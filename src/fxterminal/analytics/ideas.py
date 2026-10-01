"""Trade-idea generation — fully deterministic.

Pipeline per candidate pair (ranked by |strength differential|):
  entry   = latest close (entry zone of +/- 0.25 * daily sigma)
  stop    = entry -/+ k * sigma_daily   (sigma = realized daily vol, price units)
  target  = entry +/- rr * k * sigma_daily
  filters: min |differential|, min confidence, max spread, event blackout.

The rationale string is assembled deterministically from the dominant
components — no LLM, no invented levels. An LLM *may* later polish the wording
around these exact numbers, never replace them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.analytics.pairs import PairView
from fxterminal.analytics.strength import StrengthResult, realized_vol
from fxterminal.core.config import Settings
from fxterminal.core.freshness import fx_market_open, is_stale
from fxterminal.core.logging import get_logger
from fxterminal.models import FxPrice, MacroEvent, TradeIdea

logger = get_logger(__name__)

# risk params — deliberately conservative defaults; tune via Settings later
STOP_SIGMA = 1.5  # stop distance in daily sigma
TARGET_RR = 2.0  # target = STOP_SIGMA * TARGET_RR away
MIN_DIFFERENTIAL = 20.0  # ignore weak strength gaps
MIN_CONFIDENCE = 0.34  # require >= ~1/3 of component weight present
MAX_IDEAS = 3
VOL_WINDOW = 30
# Re-running analyze (or the hourly cycle) on unchanged inputs would otherwise
# append an identical idea batch every time — one open idea per pair+direction.
DUPLICATE_WINDOW_HOURS = 24


def _direction_word(direction: int) -> str:
    return "LONG" if direction > 0 else "SHORT"


def _build_rationale(view: PairView, strengths: dict[str, StrengthResult]) -> str:
    """Deterministic rationale from dominant components — text, not numbers
    beyond those already computed."""

    def top_component(ccy: str) -> str:
        c = strengths[ccy].components
        if not c:
            return "no data"
        k = max(c, key=lambda key: abs(c[key]))
        return f"{k} ({c[k]:+.2f})"

    b, q = view.base, view.quote
    return (
        f"{_direction_word(view.direction)} {view.pair}: strength differential "
        f"{view.differential:+.0f} ({b} {strengths[b].score:+.0f} vs "
        f"{q} {strengths[q].score:+.0f}). "
        f"{b} driven by {top_component(b)}; {q} driven by {top_component(q)}. "
        f"Confidence {view.confidence:.0%} of weighted inputs present."
    )


async def _latest_bar(session: AsyncSession, pair: str) -> tuple[float, datetime] | None:
    """(close, time) of the newest daily bar — the freshness check needs the
    timestamp, not just the price."""
    q = (
        select(FxPrice.close, FxPrice.time)
        .where(FxPrice.pair == pair, FxPrice.timeframe == "1d")
        .order_by(FxPrice.time.desc())
        .limit(1)
    )
    row = (await session.execute(q)).first()
    return None if row is None else (float(row.close), row.time)


async def _closes(session: AsyncSession, pair: str, window: int) -> list[float]:
    q = (
        select(FxPrice.close)
        .where(FxPrice.pair == pair, FxPrice.timeframe == "1d")
        .order_by(FxPrice.time.desc())
        .limit(window + 1)
    )
    rows = (await session.execute(q)).scalars().all()
    return [float(c) for c in reversed(rows)]  # oldest -> newest


async def _event_blackout_hit(
    session: AsyncSession, currencies: set[str], now: datetime, horizon_h: int
) -> str | None:
    """Name of a high/medium impact event inside the blackout window, else None."""
    lo = now - timedelta(hours=horizon_h)
    hi = now + timedelta(hours=horizon_h)
    q = (
        select(MacroEvent.name)
        .where(
            MacroEvent.currency.in_(currencies),
            MacroEvent.impact.in_(["high", "medium"]),
            MacroEvent.time >= lo,
            MacroEvent.time <= hi,
        )
        .limit(1)
    )
    return (await session.execute(q)).scalars().first()


def levels_for(direction: int, entry: float, sigma: float) -> tuple[float, float, float, float]:
    """(entry_low, entry_high, stop, target) in price units.

    Entry is a zone of +/- 0.25*sigma around the last close — closer to how a
    human would work an order than a single touch price. Long: stop below entry.
    """
    zone = 0.25 * sigma
    stop_d = STOP_SIGMA * sigma
    target_d = STOP_SIGMA * TARGET_RR * sigma
    if direction > 0:
        return entry - zone, entry + zone, entry - stop_d, entry + target_d
    return entry - zone, entry + zone, entry + stop_d, entry - target_d


def rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    return 0.0 if risk == 0 else abs(target - entry) / risk


def _build_key_risks(view: PairView, spread: float | None) -> list[str]:
    """Deterministic key-risks list from the data we actually have."""
    risks = []
    if view.confidence < 0.75:
        risks.append(f"partial data coverage ({view.confidence:.0%} of weighted inputs)")
    risks.append("strength composite is backward-looking; news can gap against the idea")
    if spread is None:
        risks.append("no live spread data — assumed liquid majors only")
    return risks


async def generate_ideas(
    session: AsyncSession,
    views: list[PairView],
    strengths: dict[str, StrengthResult],
    settings: Settings,
    now: datetime | None = None,
) -> list[TradeIdea]:
    """Apply hard filters, compute levels, persist TradeIdea rows."""
    now = now or datetime.now(UTC)
    ideas: list[TradeIdea] = []
    for view in views:
        if len(ideas) >= MAX_IDEAS:
            break
        if abs(view.differential) < MIN_DIFFERENTIAL:
            continue
        if view.confidence < MIN_CONFIDENCE:
            continue

        latest = await _latest_bar(session, view.pair)
        if latest is None:
            continue
        entry, bar_time = latest
        if is_stale("prices_1d", bar_time, now):
            # no idea built on a price older than the freshness limit — the
            # level/stop would be meaningless against a moved market
            logger.warning(
                "idea_blocked_stale_price", pair=view.pair,
                bar_time=str(bar_time), max_age_h="72",
            )
            continue
        closes = await _closes(session, view.pair, VOL_WINDOW)
        sigma = realized_vol(closes, VOL_WINDOW)
        if sigma is None or sigma <= 0:
            continue

        event = await _event_blackout_hit(
            session, {view.base, view.quote}, now, settings.event_blackout_hours
        )
        if event:
            continue

        # dedup: skip if an identical direction idea for this pair already
        # exists from a recent run — same inputs produce the same idea
        direction_word = "long" if view.direction > 0 else "short"
        dup = (
            await session.execute(
                select(TradeIdea.id)
                .where(
                    TradeIdea.pair == view.pair,
                    TradeIdea.direction == direction_word,
                    TradeIdea.created_at >= now - timedelta(hours=DUPLICATE_WINDOW_HOURS),
                )
                .limit(1)
            )
        ).scalars().first()
        if dup is not None:
            continue

        # spread filter — we only have a spread when the price provider gives
        # one; ECB fixes don't, so this check is skipped for them (documented).
        q = (
            select(FxPrice.spread_pips)
            .where(FxPrice.pair == view.pair)
            .order_by(FxPrice.time.desc())
            .limit(1)
        )
        spread = (await session.execute(q)).scalars().first()
        if spread is not None and spread > settings.max_spread_pips:
            continue

        entry_low, entry_high, stop, target = levels_for(view.direction, entry, sigma)
        mid = (entry_low + entry_high) / 2
        idea = TradeIdea(
            pair=view.pair,
            direction="long" if view.direction > 0 else "short",
            entry_low=entry_low,
            entry_high=entry_high,
            stop=stop,
            target=target,
            risk_reward=rr(mid, stop, target),
            rationale=_build_rationale(view, strengths),
            invalidation=(
                f"close beyond {stop:.5f} or strength differential reverses sign "
                f"(currently {view.differential:+.0f})"
            ),
            key_risks=_build_key_risks(view, spread),
            filters={
                "min_differential": abs(view.differential) >= MIN_DIFFERENTIAL,
                "min_confidence": view.confidence >= MIN_CONFIDENCE,
                "spread_ok": spread is None or spread <= settings.max_spread_pips,
                "event_blackout_clear": event is None,
                "price_fresh": True,  # is_stale gate above already ran
                # honest label: idea computed from the last close while the
                # market was shut — levels are off-market until it reopens
                "market_open_at_generation": fx_market_open(now),
            },
            inputs={
                "differential": view.differential,
                "base_components": strengths[view.base].components,
                "quote_components": strengths[view.quote].components,
                "sigma_daily": sigma,
                "entry_ref": entry,
            },
            valid_until=now + timedelta(days=7),
        )
        session.add(idea)
        ideas.append(idea)
    await session.commit()
    return ideas
