"""Analyst agent — tool-use loop + deterministic verifier.

Why a JSON-action loop instead of native tool_use: `complete_json` is the one
contract every backend here honours (Anthropic, OpenAI-compat, Ollama). A
ReAct-style loop over validated JSON actions gives the same capability on all
of them, with zero extra SDK surface.

The agent may call read-only tools to pull deeper data than the base context
block (per-pair bars, rate expectations, full-text doc search). After it
answers, `verify_answer` audits the output deterministically — every figure in
the answer must match a stored value within tolerance, directional claims
('X is strongest') are checked against the actual ladder, and citations must
name real stored documents. Failures surface as `warnings`, never silently.

No tool ever writes to the DB.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.models import FxPrice, MacroEvent, RateExpectation, RawDocument

MAX_ROUNDS = 5
_TOOL_TEXT_MAX = 2400  # chars per tool result fed back to the model

# Default window for macro-driven reasoning — "~3 months" per product spec.
MACRO_WINDOW_DAYS = 90

# Which stored releases feed each live theme. Keys are the Theme taxonomy
# (nlp/themes.py); values are lowercase substrings matched against
# MacroEvent.name. Themes absent here (politics, risk_sentiment, other) fall
# back to all high-impact releases.
THEME_RELEASES: dict[str, tuple[str, ...]] = {
    "inflation": ("cpi", "ppi", "pce", "hicp", "inflation", "price"),
    "growth": ("gdp", "pmi", "retail", "industrial", "production", "sales", "growth"),
    "employment": ("employ", "payroll", "unemployment", "jobless", "claim", "wage", "job", "nfp"),
    "monetary_policy": ("rate", "funds", "policy", "yield", "decision"),
    "trade": ("trade", "export", "import", "tariff", "balance", "current account"),
}


def dominant_themes(snap: dict) -> dict[str, str]:
    """currency -> highest-weight theme name in the latest stored window."""
    out: dict[str, dict] = {}
    for t in snap.get("themes", []):
        cur = out.get(t["currency"])
        if cur is None or t["weight"] > cur["weight"]:
            out[t["currency"]] = t
    return {c: t["theme"] for c, t in out.items()}


def theme_keywords(theme: str | None) -> tuple[str, ...] | None:
    """Release-name keywords for a theme; None = no filter (catch-all themes)."""
    if not theme:
        return None
    return THEME_RELEASES.get(theme.lower())

AGENT_SYSTEM = """You are the analyst agent of an FX research terminal.
You VERIFY every claim against stored data before answering — never invent
numbers, prices, dates or events.

Respond with JSON only, exactly one of:

  {"action": "tool", "tool": "<name>", "args": {...}}
  {"action": "answer", "answer": "<final text>", "citations": ["<doc title>", ...]}

Available tools (all read-only, all data is stored in the terminal DB):
  strengths {currency?: ISO code}      — latest score + components + confidence
  stances   {bank?: FED|ECB|BOJ|BOE|SNB|RBA|RBNZ} — latest stance + evidence
  themes    {currency?: ISO code}      — theme sentiment/weight, latest window
  ideas     {pair?: e.g. EURUSD}       — stored ideas: entry/stop/target/rr/risks
  prices    {pair?: e.g. EURUSD}       — latest bars across timeframes
  rates     {bank?: code}              — market-implied rate expectations
  macro     {currency?: ISO, days?: default 90, theme?: name} — releases
            filtered to the currency's dominant stored theme
  docs      {query: string}            — full-text search over stored documents

Guidance rules:
- Use tools when the DATA block lacks the detail asked for — a good analyst
  pulls prices/rates/docs to verify before advising.
- For a currency view, reason from its CURRENT theme (see THEMES) and the
  theme-matched macro releases in MACRO — e.g. an inflation theme means
  CPI/PPI/PCE prints drive the view; check their 3-month trend and surprises
  via the macro tool before advising.
- Shape "what is the theme/stance" answers like a desk brief, not a label:
  (1) dominant theme + the evidence sentence that supports it,
  (2) latest decision/stance and how it shifted,
  (3) what the market prices into the next meeting (RATE PRICING),
  (4) the upcoming theme-matched releases that will confirm or break it.
  Pull the stances/themes/docs tools when the base context lacks the
  supporting quote.
- Answer facts first, then caveats: flag low-confidence inputs, stale data,
  absent sources (e.g. a bank with no doc coverage), and idea risks.
- Cite documents by title when they back a claim.
- Keep it terse. Always end with: research output, not financial advice."""


class AgentAction(BaseModel):
    """One agent turn. `action` defaults to 'answer' so a bare {"answer": ...}
    payload — e.g. from a backend that ignores the schema — still parses."""

    action: Literal["tool", "answer"] = "answer"
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    answer: str = Field(min_length=5, max_length=8000)


class AgentResult(BaseModel):
    answer: str
    steps: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    # raw tool outputs — the verifier accepts figures from these as grounded
    tool_text: str = ""


# ---------------------------------------------------------------------------
# Tools — every one is read-only and returns a compact text block
# ---------------------------------------------------------------------------


def _comps(row: dict) -> str:
    parts = [
        f"{k} {v:+.2f}"
        for k, v in (row.get("components") or {}).items()
        if k != "_confidence"
    ]
    conf = (row.get("components") or {}).get("_confidence")
    if conf is not None:
        parts.append(f"conf {conf:.0%}")
    return ", ".join(parts)


async def _t_strengths(session: AsyncSession, args: dict, snap: dict) -> str:
    ccy = str(args.get("currency") or "").upper()
    rows = [r for r in snap["strengths"] if not ccy or r["currency"] == ccy]
    if not rows:
        scope = f" for {ccy}" if ccy else ""
        return f"no strength rows{scope} (as_of {snap.get('as_of')})"
    return "\n".join(
        f"{r['currency']} {r['score']:+.1f}  ({_comps(r)})" for r in rows
    ) + f"\nas_of {snap.get('as_of')}"


async def _t_stances(session: AsyncSession, args: dict, snap: dict) -> str:
    bank = str(args.get("bank") or "").upper()
    rows = [s for s in snap["stances"] if not bank or s["central_bank"] == bank]
    if not rows:
        return f"no stance rows{f' for {bank}' if bank else ''}"
    return "\n".join(
        f"{s['central_bank']}/{s['currency']} {s['score']:+d} {s['label']}"
        + (f" Δ{s['shift']:+.0f}" if s.get("shift") is not None else "")
        + f" — {s['title'][:90]} ({s['published']})"
        for s in rows
    )


async def _t_themes(session: AsyncSession, args: dict, snap: dict) -> str:
    ccy = str(args.get("currency") or "").upper()
    rows = [t for t in snap["themes"] if not ccy or t["currency"] == ccy]
    if not rows:
        return f"no themes{f' for {ccy}' if ccy else ''}"
    return "\n".join(
        f"{t['currency']} {t['theme']}: sentiment {t['sentiment']:+.2f} w={t['weight']:.2f}"
        for t in rows
    )


async def _t_ideas(session: AsyncSession, args: dict, snap: dict) -> str:
    pair = str(args.get("pair") or "").upper()
    rows = [i for i in snap["ideas"] if not pair or i["pair"] == pair]
    if not rows:
        return f"no stored ideas{f' for {pair}' if pair else ''}"
    return "\n".join(
        f"{i['direction'].upper()} {i['pair']} entry {i['entry_low']:.5f}-{i['entry_high']:.5f} "
        f"stop {i['stop']:.5f} target {i['target']:.5f} rr={i['rr']:.1f} — {i['rationale'][:120]}"
        for i in rows
    )


async def _t_prices(session: AsyncSession, args: dict, snap: dict) -> str:
    pair = str(args.get("pair") or "").upper()
    q = select(FxPrice).order_by(FxPrice.time.desc()).limit(12)
    if pair:
        q = select(FxPrice).where(FxPrice.pair == pair).order_by(FxPrice.time.desc()).limit(12)
    rows = (await session.execute(q)).scalars().all()
    if not rows:
        return f"no price bars{f' for {pair}' if pair else ''}"
    return "\n".join(
        f"{r.pair} [{r.timeframe}] {r.time.date()}: close {r.close:.5f}" for r in rows
    )


async def _t_rates(session: AsyncSession, args: dict, snap: dict) -> str:
    bank = str(args.get("bank") or "").upper()
    q = select(RateExpectation).order_by(RateExpectation.time.desc()).limit(80)
    if bank:
        q = q.where(RateExpectation.central_bank == bank)
    rows = (await session.execute(q)).scalars().all()
    if not rows:
        return f"no rate expectations{f' for {bank}' if bank else ''}"
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for r in rows:
        key = (r.central_bank, r.instrument)
        if key in seen:
            continue
        seen.add(key)
        meet = r.meeting_date.date().isoformat() if r.meeting_date else "-"
        lines.append(
            f"{r.central_bank}/{r.currency} {r.instrument}: implied {r.implied_rate:.3f}% "
            f"(meeting {meet})"
        )
        if len(lines) >= 12:
            break
    return "\n".join(lines)


async def _t_macro(session: AsyncSession, args: dict, snap: dict) -> str:
    ccy = str(args.get("currency") or "").upper()
    days = int(args.get("days") or MACRO_WINDOW_DAYS)
    theme = str(args.get("theme") or "").lower()
    themes = dominant_themes(snap)
    if not theme and ccy:
        theme = themes.get(ccy, "")  # fall back to the currency's live theme
    kws = theme_keywords(theme)

    cutoff = datetime.now(UTC) - timedelta(days=days)
    q = (
        select(MacroEvent)
        .where(MacroEvent.time >= cutoff)
        .order_by(MacroEvent.currency, MacroEvent.time.desc())
    )
    if ccy:
        q = q.where(MacroEvent.currency == ccy)
    rows = (await session.execute(q)).scalars().all()
    if kws:
        rows = [r for r in rows if any(k in r.name.lower() for k in kws)]
    else:
        rows = [r for r in rows if r.impact == "high"]  # catch-all: high-impact only
    if not rows:
        scope = f"{ccy} " if ccy else ""
        filt = f" matching theme {theme!r}" if theme else ""
        return f"no {scope}macro releases{filt} in the last {days}d"

    def fmt_one(r: MacroEvent) -> str:
        surprise = (
            f" surprise {r.actual - r.forecast:+.3g}"
            if r.actual is not None and r.forecast is not None
            else ""
        )
        return (
            f"{r.currency} {r.name} {r.time.date()}: "
            f"act {r.actual} fcst {r.forecast} prev {r.previous}{surprise}"
        )

    head = f"theme filter: {theme or 'none'} | window: {days}d"
    return head + "\n" + "\n".join(fmt_one(r) for r in rows[:25])


async def _t_docs(session: AsyncSession, args: dict, snap: dict) -> str:
    query = str(args.get("query") or "").strip()
    q = select(RawDocument).order_by(RawDocument.published_at.desc()).limit(6)
    if query:
        like = f"%{query}%"
        q = q.where(or_(RawDocument.title.ilike(like), RawDocument.raw_text.ilike(like)))
    rows = (await session.execute(q)).scalars().all()
    if not rows:
        return f"no documents matching {query!r}"
    return "\n".join(
        f"[{d.doc_type}] {d.title} ({d.published_at.date()}) — {d.source} — {d.url or 'no url'}"
        for d in rows
    )


TOOLS: dict[str, Any] = {
    "strengths": _t_strengths,
    "stances": _t_stances,
    "themes": _t_themes,
    "ideas": _t_ideas,
    "prices": _t_prices,
    "rates": _t_rates,
    "macro": _t_macro,
    "docs": _t_docs,
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def run_agent(
    session: AsyncSession,
    llm: Any,
    message: str,
    context: str,
    snap: dict,
) -> AgentResult:
    """JSON-action ReAct loop. Raises LLMError when the backend fails
    terminally — the caller then degrades to the deterministic answerer."""
    steps: list[str] = []
    tool_outs: list[str] = []
    transcript = (
        f"{context}\n\nQUESTION: {message}\n\nRespond with JSON only."
    )
    for _round in range(MAX_ROUNDS):
        act = await llm.complete_json(AGENT_SYSTEM, transcript, AgentAction)
        if act.action == "answer" and act.answer:
            return AgentResult(
                answer=act.answer,
                steps=steps,
                citations=act.citations,
                tool_text="\n".join(tool_outs),
            )

        name = (act.tool or "").strip()
        fn = TOOLS.get(name)
        steps.append(name + (f"({json.dumps(act.args)})" if act.args else ""))
        if fn is None:
            out = f"unknown tool {name!r}; available: {', '.join(TOOLS)}"
        else:
            try:
                out = await fn(session, act.args or {}, snap)
            except Exception as e:  # DB hiccup — tell the model, keep looping
                out = f"tool error: {e}"
        tool_outs.append(out)
        transcript += f"\n\nTOOL {name} RESULT:\n{out[:_TOOL_TEXT_MAX]}\n\nRespond with JSON only."

    # rounds exhausted — force a best-effort answer over what was gathered
    out = await llm.complete_json(
        AGENT_SYSTEM, transcript + "\n\nNow produce the final answer.", FinalAnswer
    )
    return AgentResult(
        answer=out.answer, steps=steps, tool_text="\n".join(tool_outs)
    )


# ---------------------------------------------------------------------------
# Deterministic verifier — audits the agent's output against stored rows
# ---------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?\d+\.\d+|[-+]?\d{2,}")


def _known_numbers(snap: dict, tool_text: str) -> list[tuple[float, float]]:
    """(value, tolerance) pairs the answer may legitimately cite."""
    known: list[tuple[float, float]] = []
    for r in snap["strengths"]:
        known.append((r["score"], 0.15))
        for v in (r.get("components") or {}).values():
            if isinstance(v, int | float):
                known.append((v, 0.06))
    for s in snap["stances"]:
        known.append((float(s["score"]), 1.5))
        if s.get("shift") is not None:
            known.append((float(s["shift"]), 1.5))
    for t in snap["themes"]:
        known.append((t["sentiment"], 0.06))
        known.append((t["weight"], 0.06))
    for i in snap["ideas"]:
        for key in ("entry_low", "entry_high", "stop", "target"):
            known.append((i[key], 0.001))
        known.append((i["rr"], 0.15))
    for c in snap["closes"]:
        known.append((c["close"], 0.0005))
    for m in snap.get("macro", []):
        for key in ("actual", "forecast", "previous"):
            v = m.get(key)
            if isinstance(v, int | float):
                known.append((v, max(0.05, abs(v) * 0.005)))
    for r in snap.get("rates", []):
        known.append((r["implied_rate"], 0.02))
        if r.get("implied_prob") is not None:
            # quoted as either fraction or percent in answers — accept both
            known.append((r["implied_prob"], 0.02))
            known.append((r["implied_prob"] * 100, 2.0))
    # numbers the agent pulled via tools are real stored values — accept them
    # with a proportional tolerance (rounding in the answer is fine)
    for m in _NUM.finditer(tool_text):
        v = float(m.group())
        known.append((v, max(0.05, abs(v) * 0.01)))
    return known


def verify_answer(
    answer: str, snap: dict, tool_text: str = "", citations: list[str] | None = None
) -> tuple[list[str], bool]:
    """Check every figure and directional claim in the answer.

    Returns (warnings, verified). verified=False means at least one claim
    could not be matched to stored data — shown in the UI, never hidden.
    """
    warnings: list[str] = []
    known = _known_numbers(snap, tool_text)

    # --- figure audit -------------------------------------------------------
    unchecked: list[str] = []
    for m in _NUM.finditer(answer):
        v = float(m.group())
        # dates/headline noise we don't verify (years, doc dates, counts)
        if 1990 <= v <= 2100 and v == int(v):
            continue
        if not any(abs(v - k) <= tol for k, tol in known):
            unchecked.append(m.group())
    if unchecked:
        warnings.append(
            "unverified figure(s) not in stored data: " + ", ".join(sorted(set(unchecked))[:6])
        )

    # --- directional claims --------------------------------------------------
    ordered = sorted(snap["strengths"], key=lambda r: -r["score"])
    if ordered:
        top, bottom = ordered[0]["currency"], ordered[-1]["currency"]
        for pat, actual, word in (
            (r"([A-Za-z]{3})\s+(?:is|looks|remains)\s+(?:the\s+)?strongest", top, "strongest"),
            (r"([A-Za-z]{3})\s+(?:is|looks|remains)\s+(?:the\s+)?weakest", bottom, "weakest"),
        ):
            m = re.search(pat, answer, re.IGNORECASE)
            if m and m.group(1).upper() != actual:
                warnings.append(
                    f"claim check: {m.group(1)} is not the {word} — stored ladder has {actual}"
                )

    # --- citations ------------------------------------------------------------
    titles = {d["title"] for d in snap["docs"]}
    for c in citations or []:
        if c not in titles and not any(c.lower() in t.lower() for t in titles):
            warnings.append(f"citation not in stored documents: {c!r}")

    # --- data-quality guidance ----------------------------------------------
    as_of = snap.get("as_of")
    if as_of:
        try:
            age = (datetime.now(UTC).date() - datetime.fromisoformat(as_of).date()).days
            if age > 3:
                warnings.append(f"strength data is {age}d old (as_of {as_of}) — re-ingest")
        except ValueError:
            pass
    for ccy in {r["currency"] for r in snap["strengths"]}:
        conf = (next(r for r in snap["strengths"] if r["currency"] == ccy)
                .get("components") or {}).get("_confidence")
        if conf is not None and conf < 0.5 and re.search(rf"\b{ccy}\b", answer):
            warnings.append(
                f"{ccy} strength is low-confidence ({conf:.0%}) — partial inputs only"
            )
    return warnings, not warnings
