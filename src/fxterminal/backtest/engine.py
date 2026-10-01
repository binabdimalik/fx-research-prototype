"""Point-in-time backtest replay of the strength-differential strategy.

Rules (fixed constants in `analytics/ideas.py` — same params as live ideas;
no per-run tuning, never fitted on the reported data):
  signal   = |strength(base) - strength(quote)| >= MIN_DIFFERENTIAL,
             confidence >= MIN_CONFIDENCE, computed with as_of = bar time.
  entry    = signal bar close, direction = sign(differential).
  exits    = stop 1.5*sigma / target 3.0*sigma, sigma = realized daily vol of
             closes UP TO the signal bar only.
             Both touched in one bar -> assume stop first (conservative).
             End of data -> close at last close.
  costs    = (spread + slippage) pips charged on entry AND exit;
             carry = (implied_base - implied_quote)/365 per day held, signed
             by direction (long earns when base rate > quote rate).
  sizing   = 1 unit of notional per trade; returns compound multiplicatively.

Lookahead contract: every DB read is filtered to timestamps <= bar.time —
stance by statement published_at, themes by window_end, rates by time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.analytics.ideas import MIN_CONFIDENCE, MIN_DIFFERENTIAL, STOP_SIGMA, TARGET_RR
from fxterminal.analytics.strength import StrengthResult, compute_strength, realized_vol
from fxterminal.backtest.metrics import Metrics, compute_metrics, equity_curve
from fxterminal.core.logging import get_logger
from fxterminal.models import FxPrice, RateExpectation

logger = get_logger(__name__)

VOL_WINDOW = 30
SIGNAL_EVERY_BAR = True  # compute signal each bar while flat

PIP_SIZE: dict[str, float] = {}  # default below; JPY pairs override


def pip_size(pair: str) -> float:
    return 0.01 if pair[3:] == "JPY" else 0.0001


@dataclass(frozen=True)
class SimParams:
    pair: str
    start: datetime
    end: datetime
    spread_pips: float = 1.0
    slippage_pips: float = 0.5
    oos_fraction: float = 0.0  # e.g. 0.25 => last 25% of trades reported OOS


@dataclass(frozen=True)
class SimTrade:
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry: float
    exit: float
    reason: str  # "stop" | "target" | "eod"
    gross_return: float
    net_return: float  # after costs + carry
    cost_paid: float  # fractional
    carry_earned: float  # fractional


@dataclass(frozen=True)
class SimResult:
    pair: str
    n_bars: int
    trades: list[SimTrade] = field(default_factory=list)
    metrics: Metrics | None = None
    metrics_oos: Metrics | None = None
    equity: list[float] = field(default_factory=list)


async def _bars(session: AsyncSession, pair: str, start: datetime, end: datetime) -> list[FxPrice]:
    q = (
        select(FxPrice)
        .where(
            FxPrice.pair == pair,
            FxPrice.timeframe == "1d",
            FxPrice.time >= start,
            FxPrice.time <= end,
        )
        .order_by(FxPrice.time)
    )
    return list((await session.execute(q)).scalars().all())


async def _latest_implied(session: AsyncSession, ccy: str, as_of: datetime) -> float | None:
    q = (
        select(RateExpectation.implied_rate)
        .where(RateExpectation.currency == ccy, RateExpectation.time <= as_of)
        .order_by(RateExpectation.time.desc())
        .limit(1)
    )
    return (await session.execute(q)).scalars().first()


async def _signal(
    session: AsyncSession, base: str, quote: str, t: datetime
) -> tuple[int, dict[str, StrengthResult]]:
    """(direction, strengths) or (0, ...) when filters fail."""
    s_base = await compute_strength(session, base, as_of=t)
    s_quote = await compute_strength(session, quote, as_of=t)
    if s_base is None or s_quote is None:
        return 0, {}
    diff = s_base.score - s_quote.score
    conf = min(s_base.confidence, s_quote.confidence)
    if abs(diff) < MIN_DIFFERENTIAL or conf < MIN_CONFIDENCE:
        return 0, {}
    return (1 if diff > 0 else -1), {base: s_base, quote: s_quote}


async def run_backtest(session: AsyncSession, params: SimParams) -> SimResult:
    base, quote = params.pair[:3], params.pair[3:]
    bars = await _bars(session, params.pair, params.start, params.end)
    result = SimResult(pair=params.pair, n_bars=len(bars))
    if len(bars) < VOL_WINDOW + 2:
        return result

    pip = pip_size(params.pair)
    cost_frac = (params.spread_pips + params.slippage_pips) * pip  # relative to ~1.0
    trades: list[SimTrade] = []
    opens_at = 0  # index of first bar that can generate a signal

    pos: dict | None = None  # {dir, entry, stop, target, t, carry_rate}
    for i in range(opens_at, len(bars)):
        bar = bars[i]

        # ---- manage open position on this bar's OHLC
        if pos is not None:
            exit_price: float | None = None
            reason = ""
            if pos["dir"] > 0:
                if bar.low <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif bar.high >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            else:
                if bar.high >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif bar.low <= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            if i == len(bars) - 1 and exit_price is None:
                exit_price, reason = bar.close, "eod"
            if exit_price is not None:
                days_held = max(1, (bar.time.date() - pos["t"].date()).days)
                carry = pos["carry_rate"] * pos["dir"] * days_held / 365.0
                gross = (exit_price - pos["entry"]) * pos["dir"] / pos["entry"]
                net = gross - 2 * cost_frac + carry
                trades.append(
                    SimTrade(
                        entry_time=pos["t"],
                        exit_time=bar.time,
                        direction=pos["dir"],
                        entry=pos["entry"],
                        exit=exit_price,
                        reason=reason,
                        gross_return=gross,
                        net_return=net,
                        cost_paid=2 * cost_frac,
                        carry_earned=carry,
                    )
                )
                pos = None
            continue  # no same-bar re-entry (conservative)

        # ---- flat: compute signal on this bar's close (PIT)
        if i < VOL_WINDOW:  # need vol lookback
            continue
        closes = [b.close for b in bars[: i + 1]]
        sigma = realized_vol(closes, VOL_WINDOW)
        if sigma is None or sigma <= 0:
            continue
        direction, _strengths = await _signal(session, base, quote, bar.time)
        if direction == 0:
            continue
        rb = await _latest_implied(session, base, bar.time)
        rq = await _latest_implied(session, quote, bar.time)
        carry_rate = (rb - rq) if (rb is not None and rq is not None) else 0.0
        stop_d = STOP_SIGMA * sigma
        target_d = STOP_SIGMA * TARGET_RR * sigma
        pos = {
            "dir": direction,
            "entry": bar.close,
            "stop": bar.close - direction * stop_d,
            "target": bar.close + direction * target_d,
            "t": bar.time,
            "carry_rate": carry_rate,
        }

    returns = [t.net_return for t in trades]
    split = int(len(returns) * (1.0 - params.oos_fraction)) if params.oos_fraction else len(returns)
    return SimResult(
        pair=params.pair,
        n_bars=len(bars),
        trades=trades,
        metrics=compute_metrics(returns[:split]),
        metrics_oos=compute_metrics(returns[split:]) if params.oos_fraction else None,
        equity=equity_curve(returns),
    )
