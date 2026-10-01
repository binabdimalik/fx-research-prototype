"""Command-line interface: `fxterminal ...` (installed via pyproject scripts).

Run without install: `python -m fxterminal.cli ...` with PYTHONPATH=src.
"""

from __future__ import annotations

import asyncio

import typer

from fxterminal.core.config import get_settings
from fxterminal.core.db import get_engine, init_db
from fxterminal.core.logging import configure_logging, get_logger

app = typer.Typer(help="fxterminal — AI-powered FX research terminal", no_args_is_help=True)
db_app = typer.Typer(help="Database commands")
ingest_app = typer.Typer(help="Data ingestion commands")
nlp_app = typer.Typer(help="LLM analysis commands (needs FXT_ANTHROPIC_API_KEY)")
analyze_app = typer.Typer(help="Deterministic analytics: strength, pairs, ideas")
backtest_app = typer.Typer(help="Point-in-time backtesting")
app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(nlp_app, name="nlp")
app.add_typer(analyze_app, name="analyze")
app.add_typer(backtest_app, name="backtest")

logger = get_logger(__name__)


def _setup() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)


@db_app.command("init")
def db_init() -> None:
    """Create all tables (dev convenience — use `alembic upgrade head` for real envs)."""
    _setup()

    async def _run() -> None:
        await init_db()

    asyncio.run(_run())
    typer.echo("Tables created (metadata.create_all).")


@db_app.command("check")
def db_check() -> None:
    """Verify the database is reachable."""
    _setup()

    async def _run() -> str:
        from sqlalchemy import text

        engine = get_engine()
        async with engine.connect() as conn:
            row = (await conn.execute(text("SELECT 1"))).scalar()
        await engine.dispose()
        return f"database reachable (SELECT 1 -> {row})"

    typer.echo(asyncio.run(_run()))


@db_app.command("seed-demo")
def db_seed_demo() -> None:
    """Insert clearly-labeled demo rows (model='demo-seed-no-llm') so the
    dashboard + analytics exercise end-to-end WITHOUT an LLM key.
    Run `db clear-demo` before `nlp score`/`nlp themes` to remove them."""
    _setup()
    from fxterminal.demo import seed_demo

    counts = asyncio.run(_with_session(seed_demo))
    for table, n in counts.items():
        typer.echo(f"  {table}: {n} rows")
    if not any(counts.values()):
        typer.echo("nothing seeded — run `fxterminal ingest docs` first (need doc rows to cite)")


@db_app.command("clear-demo")
def db_clear_demo() -> None:
    """Delete every row written by `db seed-demo` (matched on demo markers)."""
    _setup()
    from fxterminal.demo import clear_demo

    counts = asyncio.run(_with_session(clear_demo))
    for table, n in counts.items():
        typer.echo(f"  {table}: {n} deleted")


@app.command("config")
def show_config() -> None:
    """Print effective configuration with secrets masked."""
    _setup()
    for key, value in get_settings().safe_dict().items():
        typer.echo(f"{key}={value}")


def _print_reports(reports: list[object]) -> None:
    from fxterminal.ingest.types import IngestReport

    for r in reports:
        assert isinstance(r, IngestReport)
        if r.error:
            typer.echo(f"  {r.provider}: ERROR {r.error}", err=True)
        else:
            typer.echo(
                f"  {r.provider}: fetched={r.fetched} inserted={r.inserted} "
                f"dupes={r.duplicates} rejected={r.rejected}"
            )


async def _with_session(fn):
    """Run `fn(session)` inside a session, then close the engine cleanly."""
    from fxterminal.core.db import get_engine, get_session_factory

    async with get_session_factory()() as session:
        result = await fn(session)
    await get_engine().dispose()
    return result


@ingest_app.command("prices")
def ingest_prices_cmd(
    since: str = typer.Option(None, help="YYYY-MM-DD start date (default: month start)"),
    until: str = typer.Option(None, help="YYYY-MM-DD end date (default: today)"),
) -> None:
    """Fetch daily FX bars (default provider: Frankfurter/ECB reference)."""
    _setup()
    from datetime import date

    from fxterminal.ingest.pipeline import build_providers, default_pairs, ingest_prices

    settings = get_settings()
    since = since or date.today().replace(day=1).isoformat()
    pairs = default_pairs(settings.currency_list)
    typer.echo(f"pairs={pairs} since={since}")

    async def run(session):
        providers = build_providers(settings)
        rep = await ingest_prices(session, providers["prices"], pairs, since, until)
        return [rep]

    _print_reports(asyncio.run(_with_session(run)))


@ingest_app.command("docs")
def ingest_docs_cmd() -> None:
    """Fetch news + central-bank documents from configured RSS feeds."""
    _setup()
    from fxterminal.ingest.pipeline import build_providers, ingest_documents

    async def run(session):
        providers = build_providers(get_settings())
        return await ingest_documents(session, providers["documents"])

    _print_reports(asyncio.run(_with_session(run)))


@ingest_app.command("macro")
def ingest_macro_cmd() -> None:
    """Fetch macro observations (FRED needs FXT_FRED_API_KEY; Trading Economics
    needs FXT_TE_API_KEY)."""
    _setup()
    from fxterminal.ingest.pipeline import build_providers, ingest_macro

    async def run(session):
        providers = build_providers(get_settings())
        macro = providers["macro"]
        if not macro:
            return [await ingest_macro(session, None, get_settings().currency_list)]
        return [
            await ingest_macro(session, mp, get_settings().currency_list) for mp in macro
        ]

    _print_reports(asyncio.run(_with_session(run)))


@ingest_app.command("rates")
def ingest_rates_cmd() -> None:
    """Load market-implied rate expectations (default: local JSON file)."""
    _setup()
    from fxterminal.ingest.pipeline import build_providers, ingest_rates

    async def run(session):
        providers = build_providers(get_settings())
        return [
            await ingest_rates(
                session, providers["rates"], get_settings().bank_list
            )
        ]

    _print_reports(asyncio.run(_with_session(run)))


@ingest_app.command("tick")
def ingest_tick_cmd() -> None:
    """One-shot intraday refresh: latest '1h' bars from the intraday
    provider (Yahoo Finance chart endpoint — unofficial, no key)."""
    _setup()
    from fxterminal.sched.jobs import intraday_tick

    async def run(session):
        return await intraday_tick(session)

    typer.echo(asyncio.run(_with_session(run)))


@ingest_app.command("all")
def ingest_all_cmd(
    since: str = typer.Option(None, help="YYYY-MM-DD price start (default: month start)"),
) -> None:
    """Run every configured provider."""
    _setup()
    from fxterminal.ingest.pipeline import run_all

    async def run(session):
        return await run_all(session, since=since)

    _print_reports(asyncio.run(_with_session(run)))


def _llm_or_exit():
    """Build the configured LLM backend or exit with a clear message."""
    from fxterminal.nlp.llm import LLMError, build_llm

    try:
        return build_llm(get_settings())
    except LLMError as e:
        typer.echo(f"LLM unavailable: {e}", err=True)
        raise typer.Exit(2) from e


@nlp_app.command("score")
def nlp_score(
    bank: str = typer.Option(
        None, help="FED|ECB|BOJ|BOE|SNB|RBA|RBNZ — default all"
    ),
    limit: int = typer.Option(20),
) -> None:
    """Score all unscored central-bank statements/minutes."""
    _setup()
    from fxterminal.nlp.stance import score_pending

    llm = _llm_or_exit()

    async def run(session):
        return await score_pending(session, llm, bank=bank.upper() if bank else None, limit=limit)

    typer.echo(asyncio.run(_with_session(run)))


@nlp_app.command("themes")
def nlp_themes(
    days: int = typer.Option(7, help="Lookback window in days"),
    currency: str = typer.Option(None, help="Single currency — default all configured"),
) -> None:
    """Extract per-currency news themes with cited evidence."""
    _setup()
    from fxterminal.nlp.themes import extract_all_themes

    llm = _llm_or_exit()
    ccys = [currency.upper()] if currency else get_settings().currency_list

    async def run(session):
        return await extract_all_themes(session, ccys, llm, window_days=days)

    typer.echo(asyncio.run(_with_session(run)))


@nlp_app.command("eval")
def nlp_eval() -> None:
    """Run the stance-classifier eval set and print accuracy."""
    _setup()
    from fxterminal.nlp.eval import run_eval

    llm = _llm_or_exit()
    results, accuracy = asyncio.run(run_eval(llm))
    for r in results:
        mark = "OK " if r.correct else "MISS"
        detail = f"score={r.score} label={r.predicted_label}"
        if r.error:
            detail += f" | {r.error[:120]}"
        typer.echo(f"  [{mark}] {r.case_id}: expected={r.expected} {detail}")
    typer.echo(f"accuracy={accuracy:.0%} ({sum(r.correct for r in results)}/{len(results)})")


@analyze_app.command("strength")
def analyze_strength() -> None:
    """Compute + persist currency strength composites."""
    _setup()
    from fxterminal.analytics.strength import compute_all

    async def run(session):
        return await compute_all(session, get_settings().currency_list)

    rows = asyncio.run(_with_session(run))
    for r in rows:
        conf = r.components.get("_confidence", 0.0)
        comps = {k: v for k, v in r.components.items() if k != "_confidence"}
        typer.echo(f"  {r.currency}: {r.score:+.1f} (conf {conf:.0%}) {comps}")
    if not rows:
        typer.echo("no components available — run ingest + nlp first")


@analyze_app.command("ideas")
def analyze_ideas() -> None:
    """Rank pairs and generate filtered trade ideas (deterministic)."""
    _setup()
    from fxterminal.analytics.ideas import generate_ideas
    from fxterminal.analytics.pairs import rank_pairs
    from fxterminal.analytics.strength import compute_strength
    from fxterminal.ingest.pipeline import default_pairs

    async def run(session):
        settings = get_settings()
        strengths = {}
        for c in settings.currency_list:
            res = await compute_strength(session, c)
            if res:
                strengths[c] = res
        views = rank_pairs(strengths, default_pairs(settings.currency_list))
        return strengths, views, await generate_ideas(session, views, strengths, settings)

    _strengths, views, ideas = asyncio.run(_with_session(run))
    typer.echo("pairs by |differential|:")
    for v in views:
        direction = "+" if v.direction > 0 else "-"
        typer.echo(f"  {v.pair} diff={v.differential:+.0f} dir={direction}")
    typer.echo("ideas:")
    for i in ideas:
        entry_mid = (i.entry_low + i.entry_high) / 2
        typer.echo(
            f"  {i.direction.upper()} {i.pair} @ {entry_mid:.5f} "
            f"stop {i.stop:.5f} target {i.target:.5f} rr={i.risk_reward:.1f}"
        )
    if not ideas:
        typer.echo("  none passed filters")


@backtest_app.command("run")
def backtest_run(
    pair: str = typer.Option("EURUSD"),
    start: str = typer.Option("2026-01-01", help="ISO date"),
    end: str = typer.Option("2026-09-30", help="ISO date"),
    oos: float = typer.Option(0.25, help="Out-of-sample fraction of trades"),
    spread: float = typer.Option(1.0, help="Assumed spread in pips"),
) -> None:
    """Replay the strength-differential strategy on stored daily bars."""
    _setup()
    from datetime import UTC, datetime

    from fxterminal.backtest.engine import SimParams, run_backtest

    params = SimParams(
        pair=pair.upper(),
        start=datetime.fromisoformat(start).replace(tzinfo=UTC),
        end=datetime.fromisoformat(end).replace(tzinfo=UTC),
        spread_pips=spread,
        oos_fraction=oos,
    )

    async def run(session):
        return await run_backtest(session, params)

    res = asyncio.run(_with_session(run))
    typer.echo(f"{res.pair}: {res.n_bars} bars, {len(res.trades)} trades")
    for t in res.trades:
        typer.echo(
            f"  {t.entry_time.date()} -> {t.exit_time.date()} "
            f"dir={'+' if t.direction > 0 else '-'} {t.reason:6} "
            f"net={t.net_return:+.3%} (cost {t.cost_paid:.3%}, carry {t.carry_earned:+.3%})"
        )
    for label, m in (("in-sample", res.metrics), ("out-of-sample", res.metrics_oos)):
        if m and m.n_trades:
            sharpe = f"{m.sharpe:.2f}" if m.sharpe is not None else "n/a"
            pf = f"{m.profit_factor:.2f}" if m.profit_factor else "n/a"
            typer.echo(
                f"{label}: n={m.n_trades} hit={m.hit_rate:.0%} "
                f"exp={m.expectancy:+.3%} total={m.total_return:+.2%} "
                f"maxDD={m.max_drawdown:.2%} sharpe={sharpe} pf={pf}"
            )


@app.command("cycle")
def run_one_cycle() -> None:
    """Run one full pipeline cycle (ingest -> NLP -> analytics -> alerts)."""
    _setup()
    from fxterminal.sched.jobs import run_cycle

    async def run(session):
        return await run_cycle(session)

    result = asyncio.run(_with_session(run))
    for step, status in result.steps.items():
        typer.echo(f"  {step}: {status}")
    if not result.ok:
        raise typer.Exit(1)


@app.command("serve")
def serve(
    interval: float = typer.Option(0, help="Override FXT_SCHEDULE_INTERVAL_HOURS (0=use config)"),
) -> None:
    """Run the scheduler loop forever (API should be run separately)."""
    _setup()
    from fxterminal.sched.jobs import serve_forever

    hours = interval or get_settings().schedule_interval_hours
    typer.echo(f"scheduler starting — cycle every {hours}h")
    try:
        asyncio.run(serve_forever(hours))
    except KeyboardInterrupt:
        typer.echo("stopped")


if __name__ == "__main__":
    app()
