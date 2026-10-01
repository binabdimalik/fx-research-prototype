# fxterminal

AI-powered FX research terminal — a lightweight Bloomberg-style pipeline for
major FX. **Research tool, not financial advice.** No automatic trade
execution anywhere in this codebase.

## Architecture (pipeline of specialized steps, not one giant agent)

```
providers (swappable interfaces)
  prices | news | cb docs | macro | rate expectations
        │
        ▼
raw_documents / fx_prices / rate_expectations / macro_events   (immutable, timestamped)
        │
        ▼
LLM steps (schema-validated, cited, never does arithmetic)
  stance scorer per CB statement (vs previous statement + vs market pricing)
  theme extractor per currency
        │
        ▼
deterministic analytics (unit-tested Python)
  currency strength score → strongest vs weakest → pair → trade idea
  (entry/stop/target/R:R computed in code; spread + event-window + correlation filters)
        │
        ▼
FastAPI backend ── static dashboard + read APIs
```

## Analyst agent (chat)

`POST /api/chat` runs a **tool-use agent loop** (`api/agent.py`): the model may
call read-only tools — `strengths`, `stances`, `themes`, `ideas`, `prices`,
`rates`, `macro` (theme-matched releases), `docs` (full-text search) —
pulling deeper data than the base context
before answering. Actions are JSON-mode (`{"action":"tool"|"answer", ...}`) so
every configured backend (Anthropic, OpenAI-compat, Ollama) supports it.

Every answer is then audited by a **deterministic verifier**: figures must
match stored rows within tolerance, strongest/weakest claims are checked
against the actual ladder, citations must name real stored documents, and
data-quality issues (stale `as_of`, low-confidence currencies) surface as
`warnings`. The response carries `mode` (`agent`|`offline`),
`verified`, `warnings`, `steps` (tools used) and `sources` (title/date/url).
The dashboard renders a verified/unverified badge, the tool trail and warning
lines under each answer. Without an API key the deterministic offline
answerer replies verbatim from stored rows (`verified` by construction).

## Dashboard + API (Phase 4) + agent chat

`uvicorn fxterminal.api.main:app` serves the dashboard at `/` plus read APIs
under `/api/*` (strength, stances, themes, ideas, prices, docs, chat) — see
`api/routes.py`.

Layout:

```
src/fxterminal/
  core/        config (env secrets), structured logging, async db plumbing
  models/      SQLAlchemy schema (raw docs, prices, expectations, events, analysis)
  ingest/      provider interfaces + implementations            (Phase 2)
  nlp/         LLM stance/theme extraction, Pydantic-validated  (Phases 3-4)
  analytics/   deterministic scoring & trade construction       (Phase 5)
  backtest/    point-in-time backtesting                        (Phase 6)
  api/         FastAPI app                                      (Phase 7)
  scheduler/   jobs, alerts                                     (Phase 8)
alembic/       migrations (async, URL from FXT_DATABASE_URL)
tests/         unit + integration tests
```

## Quickstart (works without Docker via SQLite)

```bash
# 1. Python 3.11+ required. Create the venv:
python3.12 -m venv .venv          # or: virtualenv -p python3.12 .venv
source .venv/bin/activate

# 2. Install (pinned deps live in requirements.txt):
pip install -e ".[dev]"           # or: pip install -r requirements.txt -e .

# 3. Configure:
cp .env.example .env              # defaults to local SQLite — no services needed

# 4. Database:
alembic upgrade head              # Postgres or SQLite
# dev shortcut for SQLite only:   fxterminal db init

# 5. Verify:
pytest                            # unit tests (SQLite, no network)
fxterminal config                 # prints effective config, secrets masked
fxterminal db check               # connectivity
uvicorn fxterminal.api.main:app --reload   # http://localhost:8000/docs
```

### With Docker (Postgres + TimescaleDB + pgvector, Redis)

```bash
docker compose up -d
# in .env: FXT_DATABASE_URL=postgresql+asyncpg://fxterminal:fxterminal@localhost:5432/fxterminal
alembic upgrade head              # creates hypertables automatically
```

## Design rules enforced by construction

- **Immutability**: `raw_documents` is append-only; `content_hash` dedupes.
- **LLM never does math**: it reads text, classifies, extracts, cites. All
  scores, entry/stop/target, R:R, and filters are deterministic Python.
- **Auditable**: stance/theme rows store verbatim `evidence` sentences with
  `doc_id` + `published_at`, plus model + prompt version.
- **Secrets**: env-only via `FXT_*`; held as `SecretStr`; a structlog processor
  redacts any field/value resembling a key or token.
- **Timezones**: every timestamp is UTC-aware end to end.

## Ingestion (Phase 2)

Providers satisfy Protocols in `src/fxterminal/ingest/base.py` — swap any of
them in `pipeline.build_providers` without touching the pipeline or store.

| Category | Default provider | Status | Notes |
|---|---|---|---|
| FX prices (daily) | Frankfurter (`api.frankfurter.dev`) | verified | ECB reference rates; daily only, no spreads |
| FX prices (intraday) | Yahoo Finance chart endpoint | unofficial, no key | 1h bars (`timeframe='1h'`); delayed/indicative quotes — research only |
| News + CB docs | RSS: Fed, BoJ, ECB, BoE, SNB | verified live | title-keyword doc typing; RBA/RBNZ sites block bots (403) — no feed |
| General market news | investing.com RSS (markets + top) | verified live | per-item currency tagging via title keywords |
| Macro | FRED (`fred/series/observations`) | needs `FXT_FRED_API_KEY` | US-heavy, lagged non-US; see provider docstring |
| Macro calendar | Trading Economics `/calendar` | needs `FXT_TE_API_KEY` (paid — guest tier discontinued) | parsed defensively; unverified shape |
| Rate expectations | local JSON file (`data/rate_expectations.json`) | manual | no robust free API — see `providers/rates_file.py` for upgrade paths |

```bash
fxterminal ingest prices --since 2026-09-01   # daily bars, 21 crosses (7 ccy)
fxterminal ingest tick                      # intraday '1h' bars (Yahoo, no key)
fxterminal ingest docs                      # RSS feeds -> raw_documents (deduped)
fxterminal ingest macro                     # FRED (skips cleanly without key)
fxterminal ingest rates                     # JSON file -> rate_expectations
fxterminal ingest all                       # everything; per-provider failures isolated
fxterminal db seed-demo                     # demo rows w/o LLM (model='demo-seed-no-llm')
fxterminal db clear-demo                    # remove demo rows before real nlp run
```

Ingestion is idempotent (documents dedupe on `content_hash`; bars/events on
their natural unique keys with `ON CONFLICT DO NOTHING`), so it is safe to
schedule or re-run after failures.

**Provenance**: every stored row carries its `source` and, for documents, the
origin `url`. `/api/docs` returns both and the dashboard's Recent-documents
panel shows a source tag and links the title to the publisher's page. Stance
and theme rows embed `evidence` entries with `doc_id` + verbatim sentence; API
responses carry the same grounding (`context_docs` on chat).

## NLP / stance scoring (Phase 3)

```bash
export FXT_ANTHROPIC_API_KEY=sk-ant-...
fxterminal nlp eval            # accuracy on the labeled statement eval set
fxterminal nlp score           # score all unscored statements/minutes
fxterminal nlp score --bank FED              # or ECB|BOJ|BOE|SNB|RBA|RBNZ
```

LLM backend is selectable via `FXT_LLM_PROVIDER`:
- **`anthropic`** (default) — `FXT_ANTHROPIC_API_KEY` + `FXT_LLM_MODEL`
- **`openai_compat`** — any OpenAI `/chat/completions` endpoint: Groq and
  OpenRouter have free tiers, Ollama runs fully local. Set
  `FXT_LLM_BASE_URL`, `FXT_LLM_API_KEY` (omit for Ollama), `FXT_LLM_MODEL`.
  Same schema-validated `complete_json` contract; implemented over plain
  httpx, no extra SDK.

- **LLM boundary**: `LLMClient` protocol in `nlp/llm.py` — `complete_json` returns
  Pydantic-validated output or raises `LLMError` after retries. Anthropic is the
  default impl; the model only classifies/extracts/quotes.
- **Evidence is verified**: every cited sentence must be a verbatim
  (whitespace/typographic-normalized) substring of the source — fabricated
  citations trigger a retry-with-feedback, then a safe failure.
- **Deterministic deltas**: `shift_vs_prev` (score − prior stored score) and
  `surprise_vs_pricing` (score − implied-rate drift, `compute_priced_stance`)
  are computed in code and are NULL when data is missing — never fabricated.
- **Eval set**: `nlp/eval.py` — 4 labeled synthetic statement snippets covering
  hawkish/dovish/neutral/nuanced. Extend `EVAL_CASES` as you gather real texts.

## Theme extraction (Phase 4)

```bash
fxterminal nlp themes                # all configured currencies, 7-day window
fxterminal nlp themes --days 14 --currency EUR
```

Batches recent `news`/`speech` docs per currency (max 30, truncated to 1500
chars each in the prompt). The model cites via `doc_ref` ("D1", "D2"…) which is
resolved to real `doc_id`s in code — it never touches DB identifiers. Same
verify-retry contract as stance scoring: citations must be verbatim in the
cited document, else retry → safe empty result. Persisted rows store the full
evidence (doc_id, published_at, source, sentence) for audit.

## Analytics — strength, pairs, trade ideas (Phase 5)

All deterministic Python — the LLM is not involved in any number here.

```bash
fxterminal analyze strength    # composite per currency -> currency_scores
fxterminal analyze ideas       # rank pairs, apply filters, persist trade ideas
```

- **Strength** (`analytics/strength.py`): weighted composite of stance (0.50),
  theme sentiment (0.25), implied-rate drift (0.25). Missing components drop
  out and weights renormalize; `confidence` records input coverage.
- **Pair ranking** (`analytics/pairs.py`): |strength differential|, highest
  first; pairs missing a leg are skipped, never treated as zero.
- **Trade ideas** (`analytics/ideas.py`): entry zone ±0.25σ, stop at 1.5σ,
  target at 3σ (R:R = 2.0 fixed), σ = 30d realized daily vol. Hard filters:
  min differential 20, min confidence 34%, `max_spread_pips` (when provider
  supplies spreads), high/medium-impact event blackout ±`event_blackout_hours`.
  Every idea stores `invalidation`, `key_risks`, `filters`, `inputs` and the
  disclaimer. Tunables are module constants — tune deliberately, not per-run.

## Backtesting (Phase 6)

```bash
fxterminal backtest run --pair EURUSD --start 2026-01-01 --end 2026-09-30 \
    --oos 0.25 --spread 1.0
```

Point-in-time replay (`backtest/engine.py`): every DB read is cut off at the
bar's timestamp — stance by statement `published_at`, themes by `window_end`,
rates by `time`. Signal = same constants as live ideas (min differential,
confidence, 1.5σ stop, 3σ target, 30d vol). Conservative fills: both levels
touched in a bar → stop first; no same-bar re-entry; end-of-data closes at the
last close. Costs: `(spread + slippage)` pips on entry AND exit, plus carry =
`(implied_base − implied_quote)/365` per day held, signed by direction.
`backtest/metrics.py` reports hit rate, expectancy, total return, max drawdown,
per-trade Sharpe, profit factor — in-sample vs out-of-sample on a `--oos` trade
split.

Caveats: Frankfurter bars are daily ECB fixes (no true spread/volatility
microstructure), σ-stops are simplistic, and Sharpe uses a fixed 52
periods/year scaling — treat metrics as comparative, not predictive.

## Scheduling, alerts + deployment (Phase 8)

```bash
fxterminal cycle    # one full pass: ingest -> nlp -> analytics -> alerts
fxterminal serve    # loop forever on FXT_SCHEDULE_INTERVAL_HOURS (default 24)
```

`sched/jobs.py::run_cycle` runs each step isolated — a broken provider logs and
the rest of the cycle continues; LLM steps skip cleanly without a key.

`serve` runs two loops: the full cycle on `FXT_SCHEDULE_INTERVAL_HOURS`
(default 24h) and an intraday price tick on `FXT_INTRADAY_INTERVAL_MINUTES`
(default 60m) that refreshes '1h' bars between cycles. Disable with
`FXT_INTRADAY_PRICES_ENABLED=false`.

Alerts (`alerts/notifier.py`) fire on **new trade ideas** and **stance shifts**
(|shift_vs_prev| >= 30, 26h lookback) — logged always, POSTed to
`FXT_ALERT_WEBHOOK` when set. Immutable timestamps give free dedup; no
notifications table.

Monitoring: `GET /api/status` reports per-feed latest timestamp, row count and
age-hours with a `stale_feeds` list (>48h = stale; ideas/rates exempt).

Deploy (`deploy/*.service`, `make install-services`): two user-level systemd
units — `fxterminal-api` (uvicorn on 127.0.0.1:8000) and
`fxterminal-scheduler` (`fxterminal serve`). **Loopback only — no auth yet.**

## Known limitations (Phase 1)

- SQLite is dev/test only — no TimescaleDB hypertables or JSONB there.
- Initial migration builds tables from metadata; per-column diffs start with
the first autogenerated revision.
- pgvector embedding column is intentionally deferred to Phase 4 (needs the
  Postgres backend; adding it to the SQLite-tested schema would break dev).
- `/readyz` checks DB connectivity only; no Redis/LLM checks yet.

## Phase roadmap

1. ✅ Scaffold, config/secrets, schema, logging
2. ✅ Ingestion behind provider interfaces (prices, news, CB docs, macro, rates)
3. ✅ CB stance scorer + historical evaluation set
4. ✅ News theme extraction
5. ✅ Strength scoring, pair selection, trade ideas (deterministic)
6. ✅ Backtesting (point-in-time, costs, out-of-sample)
7. ✅ Dashboard + context-grounded chat
8. ✅ Scheduling, alerts, monitoring, deployment

## Data licensing warning

Free sources used later (central-bank sites, RSS, Frankfurter/ECB rates,
Dukascopy history) are fine for **personal research** but their terms may
prohibit redistribution in a commercial product — flag any productization.
