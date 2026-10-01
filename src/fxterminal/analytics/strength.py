"""Currency strength composite — deterministic, auditable, unit-tested.

Composite (all components normalized to -1..+1):
    strength = sum(w_i * c_i) / sum(w_i)  over *available* components, x100.

Components:
  stance  — latest CB stance score /100.            weight 0.50
  themes  — sum(theme.weight * theme.sentiment)/sum(w).   weight 0.25
  rates   — implied-rate drift in pp over 30d, x1 scale.   weight 0.25

Missing components are dropped and weights renormalized — a currency with no
rate data is NOT silently treated as neutral. `confidence` = fraction of total
weight actually present, persisted so downstream steps can require coverage.

WEIGHTS lives here as a named constant; tune deliberately, never per-run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.models import CurrencyScore, RateExpectation, StanceScore, Theme

WEIGHTS: dict[str, float] = {"stance": 0.50, "themes": 0.25, "rates": 0.25}
SCORER_VERSION = "strength-v1"

RATES_WINDOW_DAYS = 30
THEMES_LOOKBACK_DAYS = 14
# +0.25pp of implied-rate drift ~ +0.25 on the -1..1 axis
RATE_DRIFT_SCALE = 1.0


@dataclass(frozen=True)
class StrengthResult:
    currency: str
    score: float  # -100..+100
    components: dict[str, float]
    confidence: float  # 0..1 share of weight present


def combine_components(components: dict[str, float | None]) -> StrengthResult | None:
    """Weighted composite over available components.

    Returns None when nothing is available (caller stores nothing).
    """
    present = {k: v for k, v in components.items() if v is not None and k in WEIGHTS}
    if not present:
        return None
    wsum = sum(WEIGHTS[k] for k in present)
    composite = sum(WEIGHTS[k] * present[k] for k in present) / wsum
    composite = max(-1.0, min(1.0, composite))
    return StrengthResult(
        currency="",
        score=round(composite * 100, 2),
        components=present,
        confidence=round(wsum / sum(WEIGHTS.values()), 3),
    )


async def latest_stance(
    session: AsyncSession, currency: str, as_of: datetime | None = None
) -> float | None:
    """Latest stance whose *statement published* at or before as_of.

    PIT note: we order/cut off by the statement's `published_at`, not the
    scorer's `created_at` — a backtest at time t may use a score computed later
    so long as the source text existed at t (the text is immutable, so the
    score is a fixed function of it).
    """
    from fxterminal.models import RawDocument

    q = (
        select(StanceScore.score)
        .join(RawDocument, StanceScore.statement_doc_id == RawDocument.id)
        .where(StanceScore.currency == currency)
    )
    if as_of is not None:
        q = q.where(RawDocument.published_at <= as_of)
    q = q.order_by(RawDocument.published_at.desc()).limit(1)
    v = (await session.execute(q)).scalars().first()
    return None if v is None else max(-1.0, min(1.0, v / 100.0))


async def theme_component(session: AsyncSession, currency: str, as_of: datetime) -> float | None:
    q = (
        select(Theme.sentiment, Theme.weight)
        .where(
            Theme.currency == currency,
            Theme.window_end >= as_of - timedelta(days=THEMES_LOOKBACK_DAYS),
            Theme.window_end <= as_of,  # PIT: only themes whose window closed
        )
        .order_by(Theme.created_at.desc())
    )
    rows = (await session.execute(q)).all()
    if not rows:
        return None
    wsum = sum(r.weight for r in rows)
    if wsum <= 0:
        return None
    return max(-1.0, min(1.0, sum(r.sentiment * r.weight for r in rows) / wsum))


async def rate_drift_component(
    session: AsyncSession, currency: str, as_of: datetime
) -> float | None:
    """Change in implied policy-rate expectations over the last 30d, in pp."""
    start = as_of - timedelta(days=RATES_WINDOW_DAYS)
    q = (
        select(RateExpectation.time, RateExpectation.implied_rate)
        .where(
            RateExpectation.currency == currency,
            RateExpectation.time >= start,
            RateExpectation.time <= as_of,
        )
        .order_by(RateExpectation.time)
    )
    rows = (await session.execute(q)).all()
    if len(rows) < 2:
        return None
    drift_pp = rows[-1].implied_rate - rows[0].implied_rate
    return max(-1.0, min(1.0, drift_pp * RATE_DRIFT_SCALE))


async def compute_strength(
    session: AsyncSession, currency: str, as_of: datetime | None = None
) -> StrengthResult | None:
    """Compute (not yet persist) the composite for one currency."""
    as_of = as_of or datetime.now(UTC)
    components = {
        "stance": await latest_stance(session, currency, as_of),
        "themes": await theme_component(session, currency, as_of),
        "rates": await rate_drift_component(session, currency, as_of),
    }
    result = combine_components(components)
    if result is None:
        return None
    return StrengthResult(
        currency=currency,
        score=result.score,
        components=result.components,
        confidence=result.confidence,
    )


async def compute_all(
    session: AsyncSession, currencies: list[str], as_of: datetime | None = None
) -> list[CurrencyScore]:
    """Compute + persist strength rows for the configured universe."""
    as_of = as_of or datetime.now(UTC)
    rows: list[CurrencyScore] = []
    for ccy in currencies:
        res = await compute_strength(session, ccy, as_of)
        if res is None:
            continue
        row = CurrencyScore(
            as_of=as_of,
            currency=ccy,
            score=res.score,
            components={**res.components, "_confidence": res.confidence},
            scorer_version=SCORER_VERSION,
        )
        session.add(row)
        rows.append(row)
    await session.commit()
    return rows


def realized_vol(closes: list[float], window: int = 30) -> float | None:
    """Std dev of daily log returns x price level = absolute daily sigma in
    price units. Returns None with < window+1 observations (no faking vol)."""
    if len(closes) < window + 1:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(-window, 0)]
    if any(r == 0 for r in rets) and len(set(rets)) == 1:
        return None  # constant price — no meaningful vol
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * closes[-1]
