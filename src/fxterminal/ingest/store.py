"""Persistence for normalized ingest records.

Dedup strategy per table:
  raw_documents   — sha256 content_hash, pre-select existing hashes, insert new
  fx_prices       — unique (time, pair, timeframe, source), ON CONFLICT DO NOTHING
  macro_events    — unique (time, currency, name, source), same
  rate_expectations — unique (time, bank, instrument, source), same

ON CONFLICT DO NOTHING is emitted via dialect-specific insert (postgresql /
sqlite both support it); other dialects fall back to plain insert.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.core.config import get_settings
from fxterminal.core.logging import get_logger
from fxterminal.ingest.types import MacroObs, PriceBar, RateExp, RawDoc
from fxterminal.models import FxPrice, MacroEvent, RateExpectation, RawDocument

_INSERT_CHUNK = 500

logger = get_logger(__name__)


def _insert_do_nothing(session: AsyncSession, table: type, rows: list[dict]):
    """INSERT ... ON CONFLICT DO NOTHING for sqlite/postgres; plain insert else."""
    dialect = session.bind.dialect.name if session.bind else ""
    if dialect == "postgresql":
        return pg_insert(table).values(rows).on_conflict_do_nothing()
    if dialect == "sqlite":
        return sqlite_insert(table).values(rows).on_conflict_do_nothing()
    return insert(table).values(rows)


async def store_documents(session: AsyncSession, docs: Sequence[RawDoc]) -> tuple[int, int]:
    """Insert new docs, skipping content_hash dupes. Returns (inserted, dupes)."""
    if not docs:
        return 0, 0

    hashes = {RawDocument.compute_content_hash(d.source, d.url, d.title, d.raw_text) for d in docs}
    existing: set[str] = set()
    for i in range(0, len(hashes), _INSERT_CHUNK):
        chunk = list(hashes)[i : i + _INSERT_CHUNK]
        result = await session.execute(
            select(RawDocument.content_hash).where(RawDocument.content_hash.in_(chunk))
        )
        existing.update(result.scalars().all())

    new_docs = [
        d
        for d in docs
        if RawDocument.compute_content_hash(d.source, d.url, d.title, d.raw_text) not in existing
    ]
    # Also dedupe *within* the batch itself (two feeds can carry the same item).
    seen: set[str] = set()
    rows: list[dict] = []
    for d in new_docs:
        h = RawDocument.compute_content_hash(d.source, d.url, d.title, d.raw_text)
        if h in seen:
            continue
        seen.add(h)
        rows.append(
            {
                "source": d.source,
                "doc_type": d.doc_type,
                "url": d.url,
                "central_bank": d.central_bank,
                "currency": d.currency,
                "title": d.title,
                "published_at": d.published_at,
                "content_hash": h,
                "raw_text": d.raw_text,
                "raw_metadata": d.meta,
            }
        )
    for i in range(0, len(rows), _INSERT_CHUNK):
        await session.execute(insert(RawDocument).values(rows[i : i + _INSERT_CHUNK]))
    await session.commit()
    return len(rows), len(docs) - len(rows)


def _bar_problem(b: PriceBar, prev_close: float | None, max_jump: float) -> str | None:
    """Return why a bar is invalid, or None. Rejects, never repairs — a bad
    tick upstream must not silently poison vol/levels downstream.

    Checks: non-null positive OHLC, sane bar geometry (low <= open/close <=
    high), and a close-to-close jump cap vs the previous accepted bar
    (stored or earlier in this batch).
    """
    vals = (b.open, b.high, b.low, b.close)
    if any(v is None for v in vals):
        return "null ohlc"
    if any(v <= 0 for v in vals):
        return "non-positive price"
    if b.low > b.high:
        return "low > high"
    if not (b.low <= b.open <= b.high and b.low <= b.close <= b.high):
        return "open/close outside [low,high]"
    if prev_close is not None and prev_close > 0:
        jump = abs(b.close / prev_close - 1.0)
        if jump > max_jump:
            return f"impossible jump {jump:.1%} vs prev close {prev_close:.5f}"
    return None


async def _last_closes(
    session: AsyncSession, keys: set[tuple[str, str, str]]
) -> dict[tuple[str, str, str], float]:
    """Latest stored close per (pair, timeframe, source) — the jump-check anchor."""
    out: dict[tuple[str, str, str], float] = {}
    for pair, tf, src in keys:
        q = (
            select(FxPrice.close)
            .where(FxPrice.pair == pair, FxPrice.timeframe == tf, FxPrice.source == src)
            .order_by(FxPrice.time.desc())
            .limit(1)
        )
        v = (await session.execute(q)).scalars().first()
        if v is not None:
            out[(pair, tf, src)] = float(v)
    return out


async def store_prices(session: AsyncSession, bars: Sequence[PriceBar]) -> tuple[int, int, int]:
    """Validate -> insert. Returns (inserted, dupes, rejected).

    Rejected bars are logged with the reason — validation failures are data-
    quality events worth seeing, never silently dropped.
    """
    if not bars:
        return 0, 0, 0
    max_jump = get_settings().max_price_jump_pct
    keys = {(b.pair, b.timeframe, b.source) for b in bars}
    anchors = await _last_closes(session, keys)

    rows: list[dict] = []
    rejected = 0
    # time order per series so each bar validates against the last ACCEPTED close
    for b in sorted(bars, key=lambda x: x.time):
        key = (b.pair, b.timeframe, b.source)
        problem = _bar_problem(b, anchors.get(key), max_jump)
        if problem:
            rejected += 1
            logger.warning(
                "price_rejected", pair=b.pair, timeframe=b.timeframe,
                source=b.source, time=str(b.time), reason=problem,
            )
            continue
        anchors[key] = b.close  # accepted bar becomes the new anchor
        rows.append(
            {
                "time": b.time,
                "pair": b.pair,
                "timeframe": b.timeframe,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "spread_pips": b.spread_pips,
                "source": b.source,
            }
        )
    inserted, dupes = await _insert_ignore_many(session, FxPrice, rows)
    return inserted, dupes, rejected


async def store_macro(session: AsyncSession, obs: Sequence[MacroObs]) -> tuple[int, int]:
    if not obs:
        return 0, 0
    rows = [
        {
            "time": o.time,
            "currency": o.currency,
            "name": o.name,
            "impact": o.impact,
            "actual": o.actual,
            "forecast": o.forecast,
            "previous": o.previous,
            "source": o.source,
        }
        for o in obs
    ]
    return await _insert_ignore_many(session, MacroEvent, rows)


async def store_rate_expectations(
    session: AsyncSession, exps: Sequence[RateExp]
) -> tuple[int, int]:
    if not exps:
        return 0, 0
    rows = [
        {
            "time": r.time,
            "central_bank": r.central_bank,
            "currency": r.currency,
            "instrument": r.instrument,
            "meeting_date": r.meeting_date,
            "implied_rate": r.implied_rate,
            "implied_prob": r.implied_prob,
            "source": r.source,
        }
        for r in exps
    ]
    return await _insert_ignore_many(session, RateExpectation, rows)


async def _insert_ignore_many(
    session: AsyncSession, table: type, rows: list[dict]
) -> tuple[int, int]:
    """Insert rows ignoring unique violations. Returns (inserted, skipped)."""
    inserted = 0
    for i in range(0, len(rows), _INSERT_CHUNK):
        chunk = rows[i : i + _INSERT_CHUNK]
        stmt = _insert_do_nothing(session, table, chunk)
        result = await session.execute(stmt)
        # rowcount is reliable on postgres; sqlite returns -1 for OR IGNORE
        # chunks in some drivers — treat unknown as all-inserted.
        rowcount = result.rowcount
        inserted += rowcount if isinstance(rowcount, int) and rowcount >= 0 else len(chunk)
    await session.commit()
    return inserted, len(rows) - inserted
