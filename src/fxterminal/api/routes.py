"""Read-only API endpoints for the dashboard + chat.

Everything returns stored/derived rows — the API never recomputes analytics
and never lets the LLM write to the DB. Each response carries the disclaimer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.api.chat import ChatRequest
from fxterminal.core.config import get_settings
from fxterminal.core.db import get_session
from fxterminal.models import (
    CurrencyScore,
    FxPrice,
    RawDocument,
    StanceScore,
    Theme,
    TradeIdea,
)

router = APIRouter(prefix="/api")
DISCLAIMER = "Research tool output — not financial advice."


def _doc(d: RawDocument) -> dict[str, Any]:
    return {
        "id": str(d.id),
        "source": d.source,
        "doc_type": d.doc_type,
        "central_bank": d.central_bank,
        "currency": d.currency,
        "title": d.title,
        "url": d.url,
        "published_at": d.published_at.isoformat(),
        "raw_text": d.raw_text,
    }


def _stance(s: StanceScore, published: datetime | None) -> dict[str, Any]:
    return {
        "id": s.id,
        "central_bank": s.central_bank,
        "currency": s.currency,
        "score": s.score,
        "label": s.label,
        "shift_vs_prev": s.shift_vs_prev,
        "surprise_vs_pricing": s.surprise_vs_pricing,
        "evidence": s.evidence,
        "model": s.model,
        "prompt_version": s.prompt_version,
        "statement_doc_id": str(s.statement_doc_id),
        "statement_published_at": published.isoformat() if published else None,
        "created_at": s.created_at.isoformat(),
    }


@router.get("/stances")
async def stances(
    session: Annotated[AsyncSession, Depends(get_session)],
    currency: str | None = None,
) -> dict[str, Any]:
    """Latest stance score per bank, optionally filtered to one currency."""
    q = (
        select(StanceScore, RawDocument.published_at)
        .join(RawDocument, StanceScore.statement_doc_id == RawDocument.id)
        .order_by(RawDocument.published_at.desc())
    )
    if currency:
        q = q.where(StanceScore.currency == currency.upper())
    rows = (await session.execute(q)).all()
    seen: set[str] = set()
    out = []
    for s, pub in rows:
        if s.currency in seen:
            continue  # latest per currency only
        seen.add(s.currency)
        out.append(_stance(s, pub))
    return {"stances": out, "disclaimer": DISCLAIMER}


@router.get("/themes")
async def themes(
    session: Annotated[AsyncSession, Depends(get_session)],
    currency: str | None = None,
    days: int = Query(14, ge=1, le=90),
) -> dict[str, Any]:
    """Recent themes per currency within `days` of the latest window_end."""
    latest_end = (await session.execute(select(func.max(Theme.window_end)))).scalar_one()
    if latest_end is None:
        return {"themes": [], "disclaimer": DISCLAIMER}
    cutoff = latest_end - timedelta(days=days)
    q = (
        select(Theme)
        .where(Theme.window_end >= cutoff)
        # newest window first so the (currency, theme) dedupe keeps the
        # freshest row when repeated runs/reseeds left overlapping windows
        .order_by(Theme.currency, Theme.window_end.desc(), Theme.weight.desc())
    )
    if currency:
        q = q.where(Theme.currency == currency.upper())
    rows = list((await session.execute(q)).scalars().all())
    seen: set[tuple[str, str]] = set()
    deduped: list[Theme] = []
    for t in rows:
        key = (t.currency, t.theme)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    rows = deduped
    out = [
        {
            "id": t.id,
            "currency": t.currency,
            "theme": t.theme,
            "sentiment": t.sentiment,
            "weight": t.weight,
            "window_start": t.window_start.isoformat(),
            "window_end": t.window_end.isoformat(),
            "evidence": t.evidence,
        }
        for t in rows
    ]
    return {"themes": out, "disclaimer": DISCLAIMER}


@router.get("/strength")
async def strength(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """Latest composite strength row per currency."""
    latest_asof = (await session.execute(select(func.max(CurrencyScore.as_of)))).scalar_one()
    if latest_asof is None:
        return {"strengths": [], "as_of": None, "disclaimer": DISCLAIMER}
    q = (
        select(CurrencyScore)
        .where(CurrencyScore.as_of == latest_asof)
        .order_by(CurrencyScore.score.desc())
    )
    rows = list((await session.execute(q)).scalars().all())
    out = [
        {
            "currency": r.currency,
            "score": r.score,
            "components": r.components,
            "scorer_version": r.scorer_version,
            "as_of": r.as_of.isoformat(),
        }
        for r in rows
    ]
    return {"strengths": out, "as_of": latest_asof.isoformat(), "disclaimer": DISCLAIMER}


@router.get("/ideas")
async def ideas(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """Most recent trade ideas, newest first."""
    q = select(TradeIdea).order_by(TradeIdea.created_at.desc()).limit(limit)
    rows = list((await session.execute(q)).scalars().all())
    out = [
        {
            "id": i.id,
            "pair": i.pair,
            "direction": i.direction,
            "status": i.status,
            "entry_low": i.entry_low,
            "entry_high": i.entry_high,
            "stop": i.stop,
            "target": i.target,
            "risk_reward": i.risk_reward,
            "rationale": i.rationale,
            "invalidation": i.invalidation,
            "key_risks": i.key_risks,
            "filters": i.filters,
            "valid_until": i.valid_until.isoformat() if i.valid_until else None,
            "created_at": i.created_at.isoformat(),
            "disclaimer": i.disclaimer,
        }
        for i in rows
    ]
    return {"ideas": out, "disclaimer": DISCLAIMER}


@router.get("/prices")
async def prices(
    session: Annotated[AsyncSession, Depends(get_session)],
    pair: str = "EURUSD",
    days: int = Query(30, ge=1, le=365),
) -> dict[str, Any]:
    """Daily bars for one pair — the dashboard chart series."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    q = (
        select(FxPrice)
        .where(
            FxPrice.pair == pair.upper(),
            FxPrice.timeframe == "1d",
            FxPrice.time >= cutoff,
        )
        .order_by(FxPrice.time)
    )
    rows = list((await session.execute(q)).scalars().all())
    out = [
        {"time": r.time.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
        for r in rows
    ]
    return {"pair": pair.upper(), "bars": out, "disclaimer": DISCLAIMER}


@router.get("/docs")
async def docs(
    session: Annotated[AsyncSession, Depends(get_session)],
    doc_type: str | None = None,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Recent source documents (for browsing + citations)."""
    q = select(RawDocument).order_by(RawDocument.published_at.desc()).limit(limit)
    if doc_type:
        q = q.where(RawDocument.doc_type == doc_type)
    rows = list((await session.execute(q)).scalars().all())
    return {"documents": [_doc(d) for d in rows], "disclaimer": DISCLAIMER}


@router.get("/status")
async def status(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """Data-freshness + provider-health summary for monitoring.

    Per feed kind: latest timestamp, row count, age, and a `stale` flag judged
    against the per-type limits in core/freshness.py (market-aware for
    intraday prices). `sources` carries the last recorded run per provider
    from provider_runs — the Data Health backing data.
    """
    from fxterminal.core.freshness import age_hours, fx_market_open, is_stale
    from fxterminal.models import MacroEvent, ProviderRun, RateExpectation

    now = datetime.now(UTC)

    async def latest(model_col, *conds) -> datetime | None:
        q = select(func.max(model_col))
        for c in conds:
            q = q.where(c)
        return (await session.execute(q)).scalar_one_or_none()

    async def count(model) -> int:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()

    feeds = {
        "prices_1d": (await latest(FxPrice.time, FxPrice.timeframe == "1d"), "bar"),
        "prices_1h": (await latest(FxPrice.time, FxPrice.timeframe == "1h"), "bar"),
        "documents": (await latest(RawDocument.published_at), "row"),
        "macro": (await latest(MacroEvent.time), "row"),
        "rate_expectations": (await latest(RateExpectation.time), "row"),
        "strength": (await latest(CurrencyScore.as_of), "row"),
    }
    counts = {
        "prices": await count(FxPrice),
        "documents": await count(RawDocument),
        "macro": await count(MacroEvent),
        "rate_expectations": await count(RateExpectation),
        "stances": await count(StanceScore),
        "themes": await count(Theme),
        "strength": await count(CurrencyScore),
        "ideas": await count(TradeIdea),
    }
    out: dict[str, Any] = {}
    stale: list[str] = []
    for name, (ts, _kind) in feeds.items():
        flag = is_stale(name, ts, now)
        if flag:
            stale.append(name)
        out[name] = {
            "latest": ts.isoformat() if ts else None,
            "age_hours": age_hours(ts, now),
            "stale": flag,
        }
    for name, n in counts.items():
        out.setdefault(name, {})["rows"] = n

    # per-source last run — one entry per provider, newest first
    sub = (
        select(ProviderRun.provider, func.max(ProviderRun.started_at).label("m"))
        .group_by(ProviderRun.provider)
        .subquery()
    )
    q = select(ProviderRun).join(
        sub,
        (ProviderRun.provider == sub.c.provider)
        & (ProviderRun.started_at == sub.c.m),
    )
    sources = [
        {
            "provider": r.provider,
            "last_run": r.started_at.isoformat(),
            "duration_ms": r.duration_ms,
            "inserted": r.inserted,
            "rejected": r.rejected,
            "error": r.error,
        }
        for r in (await session.execute(q)).scalars().all()
    ]
    return {
        "feeds": out,
        "stale_feeds": stale,
        "sources": sources,
        "market_open": fx_market_open(now),
        "checked_at": now.isoformat(),
        "disclaimer": DISCLAIMER,
    }


@router.post("/chat")
async def chat(
    req: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, Any]:
    """Context-grounded Q&A. LLM answers only from stored data — degrades to a
    raw context snapshot when no LLM is configured."""
    from fxterminal.api.chat import answer_question
    from fxterminal.nlp.llm import LLMError, build_llm

    try:
        llm = build_llm(get_settings())
    except LLMError:
        llm = _NullLLM()  # returns the context snapshot as the answer
    resp = await answer_question(session, llm, req.message)
    return resp.model_dump()


class _NullLLM:
    """Stand-in when no API key is configured — complete_json always raises so
    answer_question falls back to the raw context snapshot."""

    model = "none"

    async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
        from fxterminal.nlp.llm import LLMError

        raise LLMError("no LLM configured")
