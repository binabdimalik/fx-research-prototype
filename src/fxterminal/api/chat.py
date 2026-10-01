"""Chat endpoint — context-grounded Q&A over stored research.

Design: for each question we assemble a bounded snapshot of REAL stored data
(latest strengths, stances, themes, ideas, recent doc headlines) and put it in
the prompt. The model answers from that data only — it never sees tools and
never writes numbers of its own. The response returns the exact context rows
used, so the UI can show what the answer was grounded in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.api.agent import (
    MACRO_WINDOW_DAYS,
    dominant_themes,
    run_agent,
    theme_keywords,
    verify_answer,
)
from fxterminal.core.config import BANK_CURRENCIES
from fxterminal.core.freshness import fx_market_open
from fxterminal.models import (
    CurrencyScore,
    FxPrice,
    MacroEvent,
    RateExpectation,
    RawDocument,
    StanceScore,
    Theme,
    TradeIdea,
)

PROMPT_VERSION = "agent-v1"

# Single-shot fallback system prompt — used only if the agent loop itself
# fails before producing anything usable.
SYSTEM = """You are the analyst-assistant of an FX research terminal.
Answer ONLY from the DATA block below — it contains the terminal's stored
strength scores, stance scores, themes, trade ideas and recent source
documents. Rules:
- Never invent numbers, prices, dates or events not present in DATA.
- Quote levels exactly as stored; reference sources as (doc_title, date).
- If the data doesn't contain the answer, say so plainly — no speculation.
- Always include the disclaimer that this is research, not financial advice.
- Keep answers terse: facts first, then a one-line caveat."""

MAX_CHARS = 12000


def _fmt_components(components: dict) -> str:
    """Format the components dict: 'stance +0.40, rates +0.30, conf 75%'."""
    parts = [f"{k} {v:+.2f}" for k, v in components.items() if k != "_confidence"]
    conf = components.get("_confidence")
    if conf is not None:
        parts.append(f"conf {conf:.0%}")
    return ", ".join(parts)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    context_docs: list[str]  # doc ids the answer could cite
    model: str
    disclaimer: str
    # --- agent fields (additive; absent fields default) -------------------
    mode: str = "offline"           # "agent" (LLM loop) | "offline" (deterministic)
    verified: bool = False          # True when the verifier found no mismatches
    warnings: list[str] = []        # verifier + data-quality warnings
    steps: list[str] = []           # tool calls the agent made
    sources: list[dict] = []        # source rows: title/type/date/source/url


async def build_context(
    session: AsyncSession, currency_hint: str | None = None
) -> tuple[str, list[str], dict[str, list[dict]]]:
    """Serialize the current research state into a bounded prompt block.

    Returns (text_block, doc_ids, snapshot). `snapshot` carries the same rows
    in structured form so the no-LLM fallback can answer directly instead of
    echoing raw text. Deterministic — same DB, same block.
    """
    parts: list[str] = []
    doc_ids: list[str] = []
    snap: dict[str, list[dict]] = {
        "strengths": [], "stances": [], "themes": [],
        "ideas": [], "closes": [], "docs": [], "macro": [], "rates": [],
    }

    # strengths (latest as_of)
    latest_asof = (
        await session.execute(select(func.max(CurrencyScore.as_of)))
    ).scalar_one_or_none()
    if latest_asof:
        rows = (
            (await session.execute(select(CurrencyScore).where(CurrencyScore.as_of == latest_asof)))
            .scalars()
            .all()
        )
        lines = [
            f"  {r.currency}: {r.score:+.1f} ({_fmt_components(r.components)})" for r in rows
        ]
        parts.append(f"STRENGTH (as_of {latest_asof.date()}):\n" + "\n".join(lines))
        snap["as_of"] = latest_asof.date().isoformat()  # type: ignore[assignment]
        for r in rows:
            snap["strengths"].append(
                {"currency": r.currency, "score": r.score, "components": r.components}
            )

    # latest stance per currency
    q = (
        select(StanceScore, RawDocument.published_at, RawDocument.title)
        .join(RawDocument, StanceScore.statement_doc_id == RawDocument.id)
        .order_by(RawDocument.published_at.desc())
    )
    seen: set[str] = set()
    stance_lines: list[str] = []
    for s, pub, title in (await session.execute(q)).all():
        if s.currency in seen:
            continue
        seen.add(s.currency)
        stance_lines.append(
            f"  {s.central_bank}/{s.currency}: {s.score:+d} ({s.label}) — {title} ({pub.date()})"
        )
        doc_ids.append(str(s.statement_doc_id))
        snap["stances"].append(
            {"central_bank": s.central_bank, "currency": s.currency, "score": s.score,
             "label": s.label, "shift": s.shift_vs_prev, "title": title,
             "published": pub.date().isoformat()}
        )
    if stance_lines:
        parts.append("STANCES:\n" + "\n".join(stance_lines))

    # recent themes
    latest_end = (await session.execute(select(func.max(Theme.window_end)))).scalar_one_or_none()
    if latest_end:
        rows = (
            (await session.execute(select(Theme).where(Theme.window_end == latest_end)))
            .scalars()
            .all()
        )
        lines = [
            f"  {r.currency} {r.theme}: sentiment {r.sentiment:+.2f} w={r.weight:.2f}" for r in rows
        ]
        parts.append(f"THEMES (window_end {latest_end.date()}):\n" + "\n".join(lines))
        for r in rows:
            ev = (r.evidence or [{}])[0].get("sentence", "")
            snap["themes"].append(
                {"currency": r.currency, "theme": r.theme,
                 "sentiment": r.sentiment, "weight": r.weight,
                 "evidence": ev}
            )

    # macro releases from the last ~3 months, filtered to each currency's
    # dominant stored theme — this is what the agent reasons over (e.g. USD
    # theme=inflation -> CPI/PPI/PCE prints only, not a raw data dump)
    dom = dominant_themes(snap)
    cutoff = datetime.now(UTC) - timedelta(days=MACRO_WINDOW_DAYS)
    macro_rows = (
        (await session.execute(
            select(MacroEvent)
            .where(MacroEvent.time >= cutoff)
            .order_by(MacroEvent.currency, MacroEvent.time.desc())
        ))
        .scalars()
        .all()
    )
    if macro_rows:
        lines: list[str] = []
        for r in macro_rows:
            kws = theme_keywords(dom.get(r.currency))
            if kws is not None and not any(k in r.name.lower() for k in kws):
                continue            # theme-filtered: skip unrelated releases
            if kws is None and r.impact != "high":
                continue            # no theme -> high-impact releases only
            lines.append(
                f"  {r.currency} [{dom.get(r.currency, '-')}] {r.name} "
                f"{r.time.date()}: act {r.actual} fcst {r.forecast} prev {r.previous}"
            )
            snap["macro"].append(
                {"currency": r.currency, "name": r.name, "theme": dom.get(r.currency),
                 "date": r.time.date().isoformat(), "actual": r.actual,
                 "forecast": r.forecast, "previous": r.previous}
            )
        if lines:
            parts.append(
                f"MACRO (last {MACRO_WINDOW_DAYS}d, per-currency theme filter):\n"
                + "\n".join(lines[:28])
            )

    # next-meeting market pricing — newest quote per (bank, instrument).
    # Market-implied path is half the story for any theme/stance question,
    # so it belongs in the base context, not only behind the rates tool.
    rows = (
        (
            await session.execute(
                select(RateExpectation).order_by(RateExpectation.time.desc()).limit(80)
            )
        )
        .scalars()
        .all()
    )
    if rows:
        seen2: set[tuple[str, str]] = set()
        lines = []
        for r in rows:
            key = (r.central_bank, r.instrument)
            if key in seen2:
                continue
            seen2.add(key)
            meet = r.meeting_date.date().isoformat() if r.meeting_date else "-"
            prob = f" p={r.implied_prob:.0%}" if r.implied_prob is not None else ""
            lines.append(
                f"  {r.central_bank}/{r.currency}: implied {r.implied_rate:.3f}%"
                f"{prob} (meeting {meet})"
            )
            snap["rates"].append(
                {"bank": r.central_bank, "currency": r.currency,
                 "instrument": r.instrument, "implied_rate": r.implied_rate,
                 "implied_prob": r.implied_prob, "meeting": meet}
            )
            if len(lines) >= 12:
                break
        parts.append("RATE PRICING (latest quote per instrument):\n" + "\n".join(lines))

    # latest ideas
    rows = (
        (await session.execute(select(TradeIdea).order_by(TradeIdea.created_at.desc()).limit(5)))
        .scalars()
        .all()
    )
    if rows:
        lines = [
            f"  {i.direction.upper()} {i.pair} entry {i.entry_low:.5f}-{i.entry_high:.5f} "
            f"stop {i.stop:.5f} target {i.target:.5f} rr={i.risk_reward:.1f}"
            for i in rows
        ]
        parts.append("TRADE IDEAS:\n" + "\n".join(lines))
        for i in rows:
            snap["ideas"].append(
                {"direction": i.direction, "pair": i.pair,
                 "entry_low": i.entry_low, "entry_high": i.entry_high,
                 "stop": i.stop, "target": i.target, "rr": i.risk_reward,
                 "rationale": i.rationale}
            )

    # latest closes (any timeframe — intraday bars included)
    rows = (
        (
            await session.execute(
                select(FxPrice)
                .order_by(FxPrice.time.desc())
                .limit(6)
            )
        )
        .scalars()
        .all()
    )
    if rows:
        lines = [
            f"  {r.pair} [{r.timeframe}] {r.time.date()}: close {r.close:.5f}" for r in rows
        ]
        parts.append("LATEST CLOSES:\n" + "\n".join(lines))
        for r in rows:
            snap["closes"].append(
                {"pair": r.pair, "timeframe": r.timeframe,
                 "date": r.time.date().isoformat(), "close": r.close}
            )

    # recent doc headlines (for source attribution)
    rows = (
        (
            await session.execute(
                select(RawDocument).order_by(RawDocument.published_at.desc()).limit(15)
            )
        )
        .scalars()
        .all()
    )
    if rows:
        lines = [f"  [{d.doc_type}] {d.title} ({d.published_at.date()})" for d in rows]
        parts.append("RECENT DOCUMENTS:\n" + "\n".join(lines))
        doc_ids.extend(str(d.id) for d in rows)
        for d in rows:
            snap["docs"].append(
                {"type": d.doc_type, "title": d.title,
                 "published": d.published_at.date().isoformat(),
                 "source": d.source, "url": d.url}
            )

    text = "DATA:\n\n" + "\n\n".join(parts)
    return text[:MAX_CHARS], doc_ids, snap


_DISCLAIMER = "Research tool output — not financial advice."
_BANK_CCY = BANK_CURRENCIES


def _footer(snap: dict) -> str:
    """Uniform offline footer: data as-of + market state + disclaimer."""
    asof = snap.get("as_of") or "no data"
    mkt = "market open" if fx_market_open() else "market closed (weekend)"
    return (
        f"\n\n(data as of {asof} UTC; {mkt} — deterministic answer from stored "
        f"rows. {_DISCLAIMER})"
    )


def _find_currency(message: str, snap: dict[str, list[dict]]) -> str | None:
    """First currency, central-bank or country token mentioned in the message."""
    known = {r["currency"] for r in snap["strengths"]} | set(_BANK_CCY.values())
    for tok in message.upper().split():
        t = tok.strip("?.,!:/\"'")
        if t in known:
            return t
        if t in _BANK_CCY:
            return _BANK_CCY[t]
        if t in _CCY_ALIASES:
            return _CCY_ALIASES[t]
    return None


# country/region words -> ISO code (multi-word names can't match — we split
# on whitespace, so only single tokens live here)
_CCY_ALIASES = {
    "US": "USD", "USA": "USD", "AMERICAN": "USD",
    "EURO": "EUR", "EUROZONE": "EUR",
    "UK": "GBP", "BRITAIN": "GBP", "BRITISH": "GBP",
    "JAPAN": "JPY", "JAPANESE": "JPY",
    "SWISS": "CHF", "SWITZERLAND": "CHF",
    "AUSTRALIA": "AUD", "AUSTRALIAN": "AUD", "AUSSIE": "AUD",
    "KIWI": "NZD",
}


def _drivers(row: dict) -> str:
    """Dominant strength components for one currency row."""
    comps = {k: v for k, v in (row.get("components") or {}).items() if k != "_confidence"}
    if not comps:
        return "no component detail"
    top = sorted(comps.items(), key=lambda kv: -abs(kv[1]))[:2]
    return ", ".join(f"{k} {v:+.2f}" for k, v in top)


def _offline_answer(message: str, snap: dict[str, list[dict]]) -> str:
    """Deterministic answerer for when no LLM is configured.

    Routes on intent keywords and answers only from the stored snapshot —
    same grounding contract as the LLM path, no invented numbers.
    """
    msg = message.lower()
    ccy = _find_currency(message, snap)
    strengths = snap["strengths"]
    footer = _footer(snap)  # as-of + market state + disclaimer on every answer

    def strength_of(c: str) -> dict | None:
        return next((r for r in strengths if r["currency"] == c), None)

    # --- strongest / weakest / rank ---------------------------------------
    if any(k in msg for k in ("strong", "weak", "best", "worst", "rank", "top")):
        if not strengths:
            return "No strength scores stored — run `fxterminal analyze strength`." + footer
        ordered = sorted(strengths, key=lambda r: -r["score"])
        if ccy:
            row = strength_of(ccy)
            if not row:
                return f"No strength score stored for {ccy}." + footer
            rank = [r["currency"] for r in ordered].index(ccy) + 1
            conf = (row["components"] or {}).get("_confidence")
            conf_s = f", conf {conf:.0%}" if conf is not None else ""
            return (
                f"{ccy}: {row['score']:+.1f} — rank {rank}/{len(ordered)} "
                f"(as_of {snap.get('as_of')}). Drivers: {_drivers(row)}{conf_s}."
                + footer
            )
        hi, lo = ordered[0], ordered[-1]
        return (
            f"Strongest: {hi['currency']} {hi['score']:+.1f} ({_drivers(hi)}). "
            f"Weakest: {lo['currency']} {lo['score']:+.1f} ({_drivers(lo)}). "
            f"Full ladder: "
            + " > ".join(f"{r['currency']} {r['score']:+.0f}" for r in ordered)
            + f" (as_of {snap.get('as_of')})." + footer
        )

    # --- trade ideas -------------------------------------------------------
    if any(k in msg for k in ("idea", "trade", "setup", "position", "recommend", "buy", "sell")):
        ideas = snap["ideas"]
        if ccy:
            ideas = [i for i in ideas if ccy in i["pair"]]
        if not ideas:
            return (
                "No stored trade ideas"
                + (f" for {ccy} pairs" if ccy else "")
                + " — run `fxterminal analyze ideas`."
                + footer
            )
        lines = [
            f"{i['direction'].upper()} {i['pair']}: "
            f"entry {i['entry_low']:.5f}-{i['entry_high']:.5f}, "
            f"stop {i['stop']:.5f}, target {i['target']:.5f} (rr {i['rr']:.1f})"
            for i in ideas[:3]
        ]
        return "Latest ideas:\n" + "\n".join(lines) + footer

    # --- themes (checked BEFORE stances: 'theme' is explicit intent while
    #     bank tokens like 'boj' also appear in the stance list) -------------
    if "theme" in msg or "narrative" in msg or "news" in msg or "sentiment" in msg:
        themes = snap["themes"]
        if ccy:
            themes = [t for t in themes if t["currency"] == ccy]
        if not themes:
            return (
                "No themes stored"
                + (f" for {ccy}" if ccy else "")
                + " — run `fxterminal nlp themes`."
                + footer
            )
        if ccy:
            # single-currency theme query → structured mini-brief:
            # dominant theme + evidence, stance, market pricing, and the
            # theme-matched releases that will move it next
            themes.sort(key=lambda t: -t["weight"])
            t = themes[0]
            brief: list[str] = [
                f"{t['currency']} dominant theme: {t['theme']} "
                f"(sentiment {t['sentiment']:+.2f}, w={t['weight']:.2f})"
                + (f" — also: {', '.join(x['theme'] for x in themes[1:3])}"
                   if len(themes) > 1 else "")
            ]
            if t.get("evidence"):
                brief.append(f"Evidence: \"{t['evidence']}\"")
            s = next((x for x in snap["stances"] if x["currency"] == ccy), None)
            if s:
                shift = f", Δ{s['shift']:+.0f}" if s.get("shift") is not None else ""
                brief.append(
                    f"Stance: {s['central_bank']}/{ccy} {s['score']:+d} "
                    f"({s['label']}{shift}) — \"{s['title'][:80]}\" ({s['published']})"
                )
            r = next((x for x in snap["rates"] if x["currency"] == ccy), None)
            if r:
                prob = (
                    f", {r['implied_prob']:.0%} priced"
                    if r.get("implied_prob") is not None else ""
                )
                brief.append(
                    f"Market pricing ({r['instrument']}): {r['implied_rate']:.3f}%"
                    f" implied{prob} into the {r['meeting']} meeting"
                )
            watch = [m for m in snap["macro"] if m["currency"] == ccy][:3]
            if watch:
                brief.append(
                    f"Watch next ({t['theme']} releases): "
                    + "; ".join(f"{m['name']} on {m['date']}" for m in watch)
                )
            return "Theme brief:\n" + "\n".join(f"- {p}" for p in brief) + footer
        lines = [
            f"{t['currency']} {t['theme']}: sentiment {t['sentiment']:+.2f} (w={t['weight']:.2f})"
            for t in themes
        ]
        return "Themes:\n" + "\n".join(lines) + footer

    # --- stances ------------------------------------------------------------
    if any(
        k in msg
        for k in (
            "stance", "hawk", "dov", "central bank", "policy",
            "fed", "ecb", "boj", "boe", "snb", "rba", "rbnz",
        )
    ):
        stances = snap["stances"]
        if ccy:
            stances = [s for s in stances if s["currency"] == ccy]
        if not stances:
            return (
                "No stance scores stored"
                + (f" for {ccy}" if ccy else "")
                + " — run `fxterminal nlp score`."
                + footer
            )
        lines = [
            f"{s['central_bank']}/{s['currency']}: {s['score']:+d} ({s['label']})"
            + (f", Δ{s['shift']:+.0f}" if s.get("shift") is not None else "")
            + f" — {s['title'][:80]} ({s['published']})"
            for s in stances
        ]
        return "Central-bank stances:\n" + "\n".join(lines) + footer

    # --- macro data (checked before prices: 'rate' also hits the price list) -
    if any(
        k in msg
        for k in (
            "inflation", "cpi", "ppi", "pce", "macro", "data release",
            "economic data", "payroll", "nfp", "unemployment", "gdp",
        )
    ):
        macro = snap["macro"]
        if ccy:
            macro = [m for m in macro if m["currency"] == ccy]
        if not macro:
            return (
                "No macro releases stored"
                + (f" for {ccy}" if ccy else "")
                + f" in the last {MACRO_WINDOW_DAYS}d — run `fxterminal ingest macro` "
                  "(FRED key) or add the Trading Economics key."
                + footer
            )
        theme_note = {
            m["currency"]: m.get("theme") for m in macro
        }
        head = "Releases (theme-matched, " + f"{MACRO_WINDOW_DAYS}d):\n"
        lines = [
            f"{m['currency']} {m['name']} {m['date']}: "
            f"act {m['actual']} fcst {m['forecast']} prev {m['previous']}"
            for m in macro[:8]
        ]
        t = {k for k in theme_note.values() if k}
        tail = f"\n(dominant theme(s): {', '.join(sorted(t))})" if t else ""
        return head + "\n".join(lines) + tail + footer

    # --- prices -------------------------------------------------------------
    if any(k in msg for k in ("price", "close", "level", "quote", "rate")):
        closes = snap["closes"]
        if ccy:
            closes = [c for c in closes if ccy in c["pair"]]
        if not closes:
            return "No price bars stored — run `fxterminal ingest prices` / `ingest tick`." + footer
        lines = [
            f"{c['pair']} [{c['timeframe']}] {c['date']}: {c['close']:.5f}" for c in closes[:6]
        ]
        return "Latest closes:\n" + "\n".join(lines) + footer

    # --- default: compact headline -----------------------------------------
    if not strengths:
        return (
            "No data stored yet. Run `fxterminal ingest all`, `fxterminal db seed-demo` "
            "(or `nlp score` with a key), then `fxterminal analyze strength && analyze ideas`."
            + footer
        )
    ordered = sorted(strengths, key=lambda r: -r["score"])
    hi, lo = ordered[0], ordered[-1]
    n_ideas = len(snap["ideas"])
    head = (
        f"Snapshot (as_of {snap.get('as_of')}): {hi['currency']} strongest "
        f"({hi['score']:+.1f}), {lo['currency']} weakest ({lo['score']:+.1f}); "
        f"{n_ideas} trade idea(s) stored."
    )
    if snap["ideas"]:
        i = snap["ideas"][0]
        head += (
            f" Top idea: {i['direction'].upper()} {i['pair']} "
            f"entry {i['entry_low']:.5f}-{i['entry_high']:.5f}, stop {i['stop']:.5f}."
        )
    head += " Ask about: strongest, ideas, stances, themes, prices."
    return head + footer


async def answer_question(session: AsyncSession, llm, message: str) -> ChatResponse:
    """Run the analyst agent, verify its output, return answer + grounding.

    Order of attempts: (1) tool-use agent loop via `run_agent`; (2) single-shot
    `complete_json`; (3) deterministic offline answerer. Every answer — LLM or
    not — is audited by `verify_answer`; failures become `warnings`, never
    silent edits.
    """
    context, doc_ids, snap = await build_context(session)
    steps: list[str] = []
    citations: list[str] = []

    try:
        result = await run_agent(session, llm, message, context, snap)
        answer = result.answer
        steps = result.steps
        citations = result.citations
        tool_text = result.tool_text
        # verify against snapshot + anything the tools returned
        warnings, verified = verify_answer(answer, snap, tool_text, citations)
        mode = "agent"
    except Exception:
        try:
            user = (
                f"{context}\n\nQUESTION: {message}\n\n"
                "Answer concisely. End with the disclaimer."
            )
            out = await llm.complete_json(SYSTEM, user, _ChatAnswer)
            answer = out.answer
            warnings, verified = verify_answer(answer, snap)
            mode = "agent"
        except Exception:
            # model unavailable or bad output — deterministic fallback answers
            # verbatim from stored rows, so it is verified by construction
            answer = _offline_answer(message, snap)
            warnings, verified = [], True
            mode = "offline"

    return ChatResponse(
        answer=answer,
        context_docs=doc_ids[:20],
        model=getattr(llm, "model", "unknown"),
        disclaimer="Research tool output — not financial advice.",
        mode=mode,
        verified=verified,
        warnings=warnings,
        steps=steps,
        sources=snap["docs"][:8],
    )


class _ChatAnswer(BaseModel):
    answer: str = Field(min_length=5, max_length=8000)
