"""Backtest performance metrics — pure functions, unit-tested.

All inputs are closed-trade returns (fractional, e.g. +0.02) or an equity
curve (cumulative product path). Nothing here reads the DB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    n_trades: int
    hit_rate: float  # fraction of trades with pnl > 0
    expectancy: float  # mean trade return (fractional)
    total_return: float  # product of (1 + r_i) - 1
    max_drawdown: float  # worst peak-to-trough of the equity curve, positive
    sharpe: float | None  # mean/std of trade returns, annualized per-trade
    profit_factor: float | None  # gross wins / gross losses; None if no losses


def equity_curve(returns: list[float]) -> list[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def max_drawdown(curve: list[float]) -> float:
    peak = curve[0]
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    return abs(worst)


def compute_metrics(returns: list[float], periods_per_year: float = 52.0) -> Metrics:
    """`periods_per_year` scales per-trade Sharpe to annualized terms; 52 is a
    rough default for weekly-ish trades — callers should not treat it as truth."""
    n = len(returns)
    if n == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, None, None)
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    curve = equity_curve(returns)
    mean = sum(returns) / n
    if n > 1:
        var = sum((r - mean) ** 2 for r in returns) / (n - 1)
        sharpe = (mean / math.sqrt(var)) * math.sqrt(periods_per_year) if var > 0 else None
    else:
        sharpe = None
    gross_w, gross_l = sum(wins), abs(sum(losses))
    return Metrics(
        n_trades=n,
        hit_rate=len(wins) / n,
        expectancy=mean,
        total_return=curve[-1] - 1.0,
        max_drawdown=max_drawdown(curve),
        sharpe=sharpe,
        profit_factor=(gross_w / gross_l) if gross_l > 0 else None,
    )
