"""Intraday FX bars from the unofficial Yahoo Finance chart endpoint.

Endpoint (no key required):
    https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval=60m&range=5d

Honest caveats:
  - UNOFFICIAL and undocumented — Yahoo may change or throttle it without
    notice. Fine for personal research; check https://legal.yahoo.com/us/en/yahoo/terms/
    before any redistribution or commercial use.
  - Quotes are delayed/indicative, not tradeable ticks; no bid/ask spread.
  - Used ONLY for '1h' bars. Frankfurter remains the provider for the
    authoritative daily ECB fix — the two timeframes coexist in fx_prices.

Satisfies the PriceProvider protocol: fetch_prices(pairs, since, until).
`since`/`until` are mapped onto Yahoo's coarse `range` token (their API does
not accept arbitrary date windows for intraday intervals).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from fxterminal.ingest.base import FetchClient, ProviderError
from fxterminal.ingest.types import PriceBar

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
_TIMEFRAME = "1h"
_INTERVAL = "60m"


def _range_for(since: str, until: str | None) -> str:
    """Map a since/until window onto Yahoo range tokens for 60m bars.

    Yahoo caps 60m history at ~730d, so the widest useful range is '2y'.
    Anything <= 5d fetches '5d' anyway — cheap over-fetch, deduped on store.
    """
    try:
        days = (
            (date.fromisoformat(until) if until else date.today())
            - date.fromisoformat(since)
        ).days
    except ValueError as e:
        raise ProviderError(f"yahoo: bad since/until date: {e}") from e
    if days <= 5:
        return "5d"
    if days <= 31:
        return "1mo"
    if days <= 93:
        return "3mo"
    if days <= 186:
        return "6mo"
    if days <= 365:
        return "1y"
    return "2y"


class YahooFxProvider:
    """Fetches hourly OHLC bars per pair (EURUSD -> EURUSD=X)."""

    name = "yahoo"

    def __init__(self, client: FetchClient, base_url: str = _BASE_URL) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def fetch_prices(
        self, pairs: list[str], since: str, until: str | None = None
    ) -> list[PriceBar]:
        rng = _range_for(since, until)
        bars: list[PriceBar] = []
        for pair in pairs:
            url = f"{self._base_url}/{pair}=X?interval={_INTERVAL}&range={rng}"
            text = await self._client.get_text(url)
            try:
                result = json.loads(text)["chart"]["result"][0]
                timestamps = result["timestamp"]
                quote = result["indicators"]["quote"][0]
                opens, highs, lows, closes, vols = (
                    quote.get("open") or [],
                    quote.get("high") or [],
                    quote.get("low") or [],
                    quote.get("close") or [],
                    quote.get("volume") or [],
                )
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                raise ProviderError(f"yahoo: unexpected payload for {pair}: {e}") from e

            for i, ts in enumerate(timestamps):
                try:
                    o, hi, lo, c = opens[i], highs[i], lows[i], closes[i]
                except IndexError:
                    continue
                if c is None or o is None or hi is None or lo is None:
                    continue  # gaps (weekends/holidays) — skip, never fake
                bars.append(
                    PriceBar(
                        time=datetime.fromtimestamp(ts, tz=UTC),
                        pair=pair,
                        timeframe=_TIMEFRAME,
                        open=float(o),
                        high=float(hi),
                        low=float(lo),
                        close=float(c),
                        volume=float(vols[i]) if i < len(vols) and vols[i] is not None else None,
                        source=self.name,
                    )
                )
        return bars
