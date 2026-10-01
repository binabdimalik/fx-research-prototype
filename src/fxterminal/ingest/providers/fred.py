"""Macro releases from FRED (Federal Reserve Bank of St. Louis).

API docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
Endpoint shape used here is the long-stable `fred/series/observations`
JSON API — verify parameter names against that page before changing.

Requires a free API key (FXT_FRED_API_KEY; register at
https://fred.stlouisfed.org/docs/api/api_key.html). Without a key the provider
reports a skip rather than failing the pipeline.

Coverage note: FRED is US-heavy. EUR/JPY/GBP series exist via OECD syncs but
release timing on FRED lags the official release — fine for trend analysis,
NOT for release-time event studies. A real economic-calendar provider
(e.g. Trading Economics, paid) is the upgrade path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fxterminal.core.logging import get_logger
from fxterminal.ingest.base import FetchClient, ProviderError
from fxterminal.ingest.types import MacroObs

logger = get_logger(__name__)

# Canonical series per currency -> (FRED series id, display name, impact).
# Coverage follows each bank's live themes: for a currency whose dominant theme
# is inflation, the agent pulls that currency's CPI/PPI/PCE-type releases; for
# employment it pulls payroll/claims; etc. Extend freely; keep names stable —
# they key the MacroEvent dedup.
DEFAULT_SERIES: dict[str, list[tuple[str, str, str]]] = {
    "USD": [
        # inflation theme: the Fed's three price gauges
        ("CPIAUCSL", "US CPI (index, SA)", "high"),
        ("CPILFESL", "US Core CPI (index, SA)", "high"),
        ("PPIACO", "US PPI All Commodities (index)", "high"),
        ("PCEPI", "US PCE Price Index", "high"),
        ("PCEPILFE", "US Core PCE Price Index", "high"),
        # employment theme
        ("UNRATE", "US Unemployment Rate", "high"),
        ("PAYEMS", "US Nonfarm Payrolls", "high"),
        ("ICSA", "US Initial Jobless Claims", "medium"),
        # policy + growth themes
        ("FEDFUNDS", "Effective Fed Funds Rate", "medium"),
        ("GDP", "US GDP", "medium"),
        ("RSAFS", "US Retail Sales ex Food Svcs", "medium"),
    ],
    "EUR": [
        ("CP0000EZ19M086NEST", "EA HICP YoY (OECD)", "high"),
    ],
    "JPY": [
        ("JPNCPIALLMINMEI", "Japan CPI (index, OECD)", "high"),
    ],
    "GBP": [
        ("GBRCPIALLMINMEI", "UK CPI (index, OECD)", "high"),
    ],
    # OECD "CPI: All Items" index pattern {ISO3}CPIALLMINMEI — covers the
    # remaining banks' inflation inputs
    "CHF": [
        ("CHECPIALLMINMEI", "Switzerland CPI (index, OECD)", "high"),
    ],
    "NZD": [
        ("NZLCPIALLMINMEI", "New Zealand CPI (index, OECD)", "high"),
    ],
    "AUD": [
        ("AUSCPIALLMINMEI", "Australia CPI (index, OECD)", "high"),
    ],
}


class FredMacroProvider:
    """Pulls recent observations for configured FRED series."""

    name = "fred"

    def __init__(
        self,
        client: FetchClient,
        api_key: str,
        base_url: str = "https://api.stlouisfed.org",
        series_map: dict[str, list[tuple[str, str, str]]] | None = None,
        limit: int = 8,
    ) -> None:
        if not api_key:
            raise ProviderError("FRED_API_KEY not configured")
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._series_map = series_map or DEFAULT_SERIES
        self._limit = limit

    async def fetch_macro(self, currencies: list[str]) -> list[MacroObs]:
        out: list[MacroObs] = []
        for ccy in currencies:
            for series_id, display, impact in self._series_map.get(ccy, []):
                url = (
                    f"{self._base_url}/fred/series/observations"
                    f"?series_id={series_id}&api_key={self._api_key}"
                    f"&file_type=json&sort_order=desc&limit={self._limit}"
                )
                try:
                    text = await self._client.get_text(url)
                    out.extend(
                        self._parse_observations(text, ccy, series_id, display, impact)
                    )
                except ProviderError as e:
                    # one bad/renamed series must not sink the whole currency run
                    logger.warning("fred_series_failed", series=series_id, error=str(e)[:160])
        return out

    def _parse_observations(
        self, text: str, ccy: str, series_id: str, display: str, impact: str
    ) -> list[MacroObs]:
        try:
            payload = json.loads(text)
            observations = payload["observations"]
        except (json.JSONDecodeError, KeyError) as e:
            raise ProviderError(f"fred {series_id}: unexpected payload: {e}") from e
        obs_list: list[MacroObs] = []
        for obs in observations:
            value = obs.get("value")
            if value in (None, "."):  # FRED uses "." for missing observations
                continue
            try:
                obs_date = obs["date"]
                day = datetime.fromisoformat(obs_date).replace(tzinfo=UTC)
                actual = float(value)
            except (KeyError, ValueError) as e:
                raise ProviderError(f"fred {series_id}: bad observation {obs!r}") from e
            obs_list.append(
                MacroObs(
                    time=day,
                    currency=ccy,
                    name=display,
                    impact=impact,  # type: ignore[arg-type]
                    actual=actual,
                    source=self.name,
                    meta={"series_id": series_id, "observation_date": obs_date},
                )
            )
        return obs_list
