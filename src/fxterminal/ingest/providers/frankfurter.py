"""Daily FX reference rates from Frankfurter (ECB reference data republished).

API docs: https://frankfurter.dev/ — endpoints verified against
`api.frankfurter.dev` (the legacy `api.frankfurter.app` host 301-redirects;
use the .dev host). Free for personal research; data originates from the ECB
reference rates published ~16:00 CET on TARGET business days.

Limitations (honest): daily granularity only, ECB-day fixes not tradeable
quotes, no bid/ask spread. Suitable for strength signals and daily backtests;
not for intraday execution analysis.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from fxterminal.ingest.base import FetchClient, ProviderError
from fxterminal.ingest.types import PriceBar

# ECB publishes the fix around 16:00 CET; we stamp bars 16:00 UTC (close enough
# for daily signals; documented approximation).
_FIX_HOUR_UTC = 16


def cross_rate(pair: str, eur_rates: dict[str, float]) -> float:
    """Quote for `pair` (AB = units of B per 1 A) from EUR-based rates.

    eur_rates maps currency -> units per 1 EUR. So EURB = rates[B] and
    AB = rates[B]/rates[A] for A != EUR. Pure function — deterministic.
    """
    if len(pair) != 6:
        raise ValueError(f"pair must be 6 chars like EURUSD, got {pair!r}")
    base, quote = pair[:3], pair[3:]
    r_base = 1.0 if base == "EUR" else eur_rates.get(base)
    r_quote = 1.0 if quote == "EUR" else eur_rates.get(quote)
    if r_base is None or r_quote is None:
        raise ProviderError(f"missing leg for {pair} in frankfurter rates")
    return r_quote / r_base


class FrankfurterProvider:
    """Fetches daily EUR-reference bars and derives crosses in code.

    Provider contract: fetch_prices(pairs, since="YYYY-MM-DD", until=None).
    """

    name = "frankfurter"

    def __init__(self, client: FetchClient, base_url: str = "https://api.frankfurter.dev") -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def fetch_prices(
        self, pairs: list[str], since: str, until: str | None = None
    ) -> list[PriceBar]:
        currencies = sorted({pair[:3] for pair in pairs} | {pair[3:] for pair in pairs} - {"EUR"})
        url = f"{self._base_url}/v1/{since}..{until or ''}?base=EUR&symbols={','.join(currencies)}"
        text = await self._client.get_text(url)
        try:
            payload = json.loads(text)
            rows = payload["rates"]
        except (json.JSONDecodeError, KeyError) as e:
            raise ProviderError(f"frankfurter: unexpected payload shape: {e}") from e

        bars: list[PriceBar] = []
        for day, rates in sorted(rows.items()):
            try:
                d = date.fromisoformat(day)
            except ValueError as e:
                raise ProviderError(f"frankfurter: bad date {day!r}") from e
            t = datetime(d.year, d.month, d.day, _FIX_HOUR_UTC, tzinfo=UTC)
            for pair in pairs:
                px = cross_rate(pair, rates)
                bars.append(
                    PriceBar(
                        time=t,
                        pair=pair,
                        timeframe="1d",
                        open=px,
                        high=px,
                        low=px,
                        close=px,
                        source=self.name,
                    )
                )
        return bars
