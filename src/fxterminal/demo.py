"""Demo seed data — clearly-labeled synthetic rows so the dashboard and full
pipeline exercise end-to-end WITHOUT an LLM key.

Every seeded row carries model='demo-seed-no-llm' (stances/themes) or
source='demo_seed' (rate expectations), so it is trivially identifiable and
removable via `clear_demo` / `fxterminal db clear-demo` before running real
`nlp score`/`nlp themes` — the unique constraints then let real scores land.

Evidence fields cite REAL raw_documents (latest statement/minutes per bank,
latest news doc per currency) with the document title as the verbatim sentence
— titles are verbatim substrings of raw_text for RSS items.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.core.config import BANK_CURRENCIES
from fxterminal.ingest.store import store_rate_expectations
from fxterminal.ingest.types import RateExp
from fxterminal.models import (
    MacroEvent,
    RateExpectation,
    RawDocument,
    StanceScore,
    Theme,
)

DEMO_MODEL = "demo-seed-no-llm"
DEMO_PROMPT = "manual-v0"
DEMO_SOURCE = "demo_seed"

# Plausible demo narrative: hawkish USD/JPY repricing, dovish EUR/GBP,
# quiet SNB/RBA/RBNZ.
# (bank, ccy): (stance score, label, shift, theme, theme sentiment, weight,
#               implied-rate path over ~30d, meeting in ~35d)
_DEMO: dict[str, tuple[int, str, float, str, float, float, list[float]]] = {
    # USD theme=inflation on purpose — the agent's macro tool then surfaces
    # the CPI/PPI/PCE series seeded below
    "FED": (40, "hawkish", 15.0, "inflation", 0.55, 0.90, [4.10, 4.25, 4.40]),
    "ECB": (-45, "dovish", -20.0, "growth", -0.45, 0.80, [2.35, 2.20, 2.05]),
    "BOJ": (25, "mildly hawkish", 10.0, "monetary policy", 0.35, 0.75, [0.60, 0.70, 0.85]),
    "BOE": (-10, "neutral", -5.0, "monetary policy", -0.20, 0.80, [3.85, 3.75, 3.60]),
    "SNB": (5, "neutral", 0.0, "monetary policy", 0.10, 0.70, [0.30, 0.30, 0.35]),
    "RBA": (15, "mildly hawkish", 5.0, "inflation", 0.30, 0.70, [3.60, 3.60, 3.55]),
    "RBNZ": (-20, "dovish", -10.0, "growth", -0.30, 0.70, [3.25, 3.10, 3.00]),
}
_BANK_CCY = BANK_CURRENCIES

# ~3 months of theme-matched releases so the agent's macro reasoning demos
# without a FRED/TE key. (ccy, name, impact, [(days_ago, actual, fcst, prev)])
_DEMO_MACRO: list[tuple[str, str, str, list[tuple[int, float, float, float]]]] = [
    # USD theme=inflation → the three Fed price gauges, gently re-accelerating
    ("USD", "US CPI (index, SA)", "high", [
        (80, 324.20, 324.00, 323.60),
        (50, 324.90, 324.60, 324.20),
        (20, 325.60, 325.10, 324.90),
    ]),
    ("USD", "US PPI All Commodities (index)", "high", [
        (75, 252.40, 252.20, 251.90),
        (45, 253.10, 252.70, 252.40),
        (15, 253.80, 253.20, 253.10),
    ]),
    ("USD", "US Core PCE Price Index", "high", [
        (70, 126.80, 126.60, 126.40),
        (40, 127.20, 127.00, 126.80),
        (10, 127.60, 127.30, 127.20),
    ]),
    # AUD theme=inflation (RBA) → CPI
    ("AUD", "Australia CPI (index)", "high", [
        (60, 138.10, 137.80, 137.50),
        (12, 138.70, 138.30, 138.10),
    ]),
    # EUR theme=growth → GDP; NZD theme=growth → GDP
    ("EUR", "EA GDP YoY", "high", [
        (55, 0.9, 1.0, 1.1),
        (8, 0.7, 0.9, 0.9),
    ]),
    ("NZD", "NZ GDP QoQ", "high", [(25, 0.2, 0.4, 0.5)]),
    # GBP/CHF/JPY themes=monetary_policy → the policy rate itself
    ("GBP", "UK Bank Rate", "high", [(30, 3.75, 3.75, 4.00)]),
    ("CHF", "SNB Policy Rate", "high", [(65, 0.25, 0.25, 0.25)]),
    ("JPY", "BOJ Policy Rate", "high", [(45, 0.75, 0.75, 0.50)]),
]


async def _latest_doc(
    session: AsyncSession, *, bank: str | None, ccy: str | None, doc_types: tuple[str, ...]
) -> RawDocument | None:
    q = select(RawDocument).where(RawDocument.doc_type.in_(doc_types))
    if bank:
        q = q.where(RawDocument.central_bank == bank)
    if ccy:
        q = q.where(RawDocument.currency == ccy)
    q = q.order_by(RawDocument.published_at.desc()).limit(1)
    return (await session.execute(q)).scalars().first()


async def seed_demo(session: AsyncSession) -> dict[str, int]:
    """Insert demo rate expectations, stance scores and themes.

    Idempotent-ish: stance/theme inserts check the unique-keyed pair first;
    rate rows dedupe on their natural unique key. Returns row counts.
    """
    now = datetime.now(UTC)
    counts = {"rate_expectations": 0, "stance_scores": 0, "themes": 0, "macro_events": 0}

    # --- rate expectations: 3 snapshots per bank over ~30d -----------------
    exps: list[RateExp] = []
    for bank, ccy in _BANK_CCY.items():
        spec = _DEMO.get(bank)
        if spec is None:
            continue
        for days_ago, rate in zip((30, 15, 0), spec[6], strict=True):
            exps.append(
                RateExp(
                    time=now - timedelta(days=days_ago),
                    central_bank=bank,
                    currency=ccy,
                    instrument="MANUAL",
                    implied_rate=rate,
                    meeting_date=now + timedelta(days=35),
                    implied_prob=0.6,
                    source=DEMO_SOURCE,
                )
            )
    ins, _ = await store_rate_expectations(session, exps)
    counts["rate_expectations"] = ins

    # --- stance scores: latest statement/minutes per bank ------------------
    for bank, (score, label, shift, *_rest) in _DEMO.items():
        doc = await _latest_doc(session, bank=bank, ccy=None, doc_types=("statement", "minutes"))
        if doc is None:
            # some feeds (e.g. ecb_rss) only carry 'news' — cite the latest
            # doc for the bank so the demo pipeline still exercises fully
            doc = await _latest_doc(
                session, bank=bank, ccy=None, doc_types=("statement", "minutes", "speech", "news")
            )
        if doc is None:
            continue
        exists = (
            await session.execute(
                select(func.count(StanceScore.id)).where(
                    StanceScore.central_bank == bank,
                    StanceScore.statement_doc_id == doc.id,
                )
            )
        ).scalar_one()
        if exists:
            continue
        prev = await session.execute(
            select(RawDocument)
            .where(
                RawDocument.central_bank == bank,
                RawDocument.doc_type.in_(["statement", "minutes"]),
                RawDocument.published_at < doc.published_at,
            )
            .order_by(RawDocument.published_at.desc())
            .limit(1)
        )
        prev_doc = prev.scalars().first()
        session.add(
            StanceScore(
                central_bank=bank,
                currency=doc.currency or _BANK_CCY[bank],
                statement_doc_id=doc.id,
                previous_statement_doc_id=prev_doc.id if prev_doc else None,
                score=score,
                label=label,
                shift_vs_prev=shift,
                surprise_vs_pricing=None,
                evidence=[
                    {
                        "doc_id": str(doc.id),
                        "published_at": doc.published_at.isoformat(),
                        "sentence": doc.title,
                        "topic": "policy",
                    }
                ],
                model=DEMO_MODEL,
                prompt_version=DEMO_PROMPT,
            )
        )
        counts["stance_scores"] += 1

    # --- themes: latest news doc per currency ------------------------------
    window_start = now - timedelta(days=14)
    for bank, spec in _DEMO.items():
        ccy = _BANK_CCY[bank]
        theme, sent, w = spec[3], spec[4], spec[5]
        doc = await _latest_doc(session, bank=None, ccy=ccy, doc_types=("news",))
        if doc is None:
            continue
        # upsert on (currency, model): a reseed refreshes the demo row in
        # place instead of piling duplicate themes onto the dashboard
        existing = (
            await session.execute(
                select(Theme).where(
                    Theme.currency == ccy, Theme.model == DEMO_MODEL
                )
            )
        ).scalars().first()
        if existing is not None:
            existing.theme = theme
            existing.sentiment = sent
            existing.weight = w
            existing.window_start = window_start
            existing.window_end = now
            existing.evidence = [
                {
                    "doc_id": str(doc.id),
                    "published_at": doc.published_at.isoformat(),
                    "source": doc.source,
                    "sentence": doc.title,
                    "topic": theme,
                }
            ]
            continue
        session.add(
            Theme(
                currency=ccy,
                theme=theme,
                sentiment=sent,
                weight=w,
                window_start=window_start,
                window_end=now,
                evidence=[
                    {
                        "doc_id": str(doc.id),
                        "published_at": doc.published_at.isoformat(),
                        "source": doc.source,
                        "sentence": doc.title,
                        "topic": theme,
                    }
                ],
                model=DEMO_MODEL,
                prompt_version=DEMO_PROMPT,
            )
        )
        counts["themes"] += 1

    # --- macro events: theme-matched releases over ~3 months ----------------
    # dedupe on (currency, name, source) — the demo series set is fixed, so a
    # reseed must not pile up the same releases with fresh timestamps
    for ccy, name, impact, obs in _DEMO_MACRO:
        exists = (
            await session.execute(
                select(func.count(MacroEvent.id)).where(
                    MacroEvent.currency == ccy,
                    MacroEvent.name == name,
                    MacroEvent.source == DEMO_SOURCE,
                )
            )
        ).scalar_one()
        if exists:
            continue
        for days_ago, act, fcst, prev in obs:
            session.add(
                MacroEvent(
                    time=now - timedelta(days=days_ago),
                    currency=ccy,
                    name=name,
                    impact=impact,
                    actual=act,
                    forecast=fcst,
                    previous=prev,
                    source=DEMO_SOURCE,
                )
            )
            counts["macro_events"] += 1

    await session.commit()
    return counts


async def clear_demo(session: AsyncSession) -> dict[str, int]:
    """Delete all rows written by `seed_demo` (matched on the demo markers)."""
    counts: dict[str, int] = {}
    for table, cond, key in (
        (StanceScore, StanceScore.model == DEMO_MODEL, "stance_scores"),
        (Theme, Theme.model == DEMO_MODEL, "themes"),
        (RateExpectation, RateExpectation.source == DEMO_SOURCE, "rate_expectations"),
        (MacroEvent, MacroEvent.source == DEMO_SOURCE, "macro_events"),
    ):
        res = await session.execute(delete(table).where(cond))
        counts[key] = res.rowcount if isinstance(res.rowcount, int) else 0
    await session.commit()
    return counts
