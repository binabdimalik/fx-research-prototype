"""Pair ranking from currency strength differentials. Deterministic."""

from __future__ import annotations

from dataclasses import dataclass

from fxterminal.analytics.strength import StrengthResult


@dataclass(frozen=True)
class PairView:
    pair: str
    base: str
    quote: str
    direction: int  # +1 long base / -1 short base
    differential: float  # strength(base) - strength(quote), -200..+200
    confidence: float  # min(conf_base, conf_quote)


def rank_pairs(strengths: dict[str, StrengthResult], pairs: list[str]) -> list[PairView]:
    """Rank configured pairs by absolute strength differential, high to low.

    Pairs where either leg lacks a strength reading are skipped — no implicit
    zeros. Output order is total: ties broken alphabetically.
    """
    views: list[PairView] = []
    for pair in pairs:
        base, quote = pair[:3], pair[3:]
        s_base = strengths.get(base)
        s_quote = strengths.get(quote)
        if s_base is None or s_quote is None:
            continue
        diff = s_base.score - s_quote.score
        views.append(
            PairView(
                pair=pair,
                base=base,
                quote=quote,
                direction=1 if diff >= 0 else -1,
                differential=round(diff, 2),
                confidence=min(s_base.confidence, s_quote.confidence),
            )
        )
    views.sort(key=lambda v: (-abs(v.differential), v.pair))
    return views
