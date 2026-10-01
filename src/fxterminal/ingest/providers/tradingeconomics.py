"""Macro calendar provider: Trading Economics (paid API).

Endpoint: GET https://api.tradingeconomics.com/calendar?c={key}&f=json
    (optional country/date filters; we fetch the broad calendar and filter
    client-side so no per-country quota is burned)

Key: FXT_TE_API_KEY — PAID plan required; the old guest:guest account was
discontinued (their response: "guest account has been discontinued... subscribe
at tradingeconomics.com/api/pricing.aspx"). Verified 2026-10-01 — provider is
key-gated in `build_providers` and reported as skipped when unset.

Response rows are parsed defensively (.get() everywhere) because the exact
field casing varies by endpoint version: Date, Country, Event, Category,
Actual, Forecast, Previous, Importance (1-3), Currency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fxterminal.ingest.base import FetchClient, ProviderError
from fxterminal.ingest.types import MacroObs

_BASE = "https://api.tradingeconomics.com"

# Country names in the TE calendar -> ISO currency of the configured universe.
_COUNTRY_CCY: dict[str, str] = {
    "United States": "USD",
    "Euro Area": "EUR",
    "Eurozone": "EUR",
    "Germany": "EUR",
    "France": "EUR",
    "Italy": "EUR",
    "Spain": "EUR",
    "Japan": "JPY",
    "United Kingdom": "GBP",
    "Switzerland": "CHF",
    "Australia": "AUD",
    "New Zealand": "NZD",
}
_IMPORTANCE: dict[int, str] = {3: "high", 2: "medium", 1: "low"}


def _row_currency(row: dict) -> str | None:
    """TE rows carry either a Currency code or a Country name."""
    cur = (row.get("Currency") or "").strip().upper()
    if cur in _COUNTRY_CCY.values():
        return cur
    return _COUNTRY_CCY.get((row.get("Country") or "").strip())


class TradingEconomicsMacroProvider:
    """Fetches the economic calendar and maps rows onto MacroObs."""

    name = "tradingeconomics"

    def __init__(self, client: FetchClient, api_key: str, base_url: str = _BASE) -> None:
        if not api_key:
            raise ProviderError("tradingeconomics: FXT_TE_API_KEY not configured")
        self._client = client
        self._key = api_key
        self._base_url = base_url.rstrip("/")

    async def fetch_macro(self, currencies: list[str]) -> list[MacroObs]:
        text = await self._client.get_text(f"{self._base_url}/calendar?c={self._key}&f=json")
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProviderError(f"tradingeconomics: bad JSON: {e}") from e
        if not isinstance(rows, list):
            raise ProviderError("tradingeconomics: expected a JSON array")

        want = set(currencies)
        obs: list[MacroObs] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ccy = _row_currency(row)
            if ccy is None or ccy not in want:
                continue
            raw_date = row.get("Date")
            try:
                t = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            obs.append(
                MacroObs(
                    time=t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC),
                    currency=ccy,
                    name=str(row.get("Event") or row.get("Category") or "calendar event"),
                    impact=_IMPORTANCE.get(int(row.get("Importance") or 2), "medium"),
                    actual=_num(row.get("Actual")),
                    forecast=_num(row.get("Forecast")),
                    previous=_num(row.get("Previous")),
                    source=self.name,
                    meta={"country": row.get("Country"), "category": row.get("Category")},
                )
            )
        return obs


def _num(v) -> float | None:
    """TE sometimes renders numeric fields as '3.1%' / '-'."""
    if v is None or v == "-":
        return None
    try:
        return float(str(v).replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
