"""Rate expectations from a local JSON file (manual input provider).

There is no robust free API for market-implied policy expectations. Real
sources worth evaluating later (all have cost or ToS considerations):
  - CME FedWatch page data (unofficial JSON; scraping has ToS risk — check
    https://www.cmegroup.com/terms-conditions.html before automating)
  - CME Fed Funds / SOFR futures settlement prices (published daily; you'd
    compute implied probabilities yourself — deterministic, license-cleaner)
  - Paid: Bloomberg WIRP, LSEG/Refinitiv, MacroPolis-type vendors.

Until one is chosen, drop a file at FXT_RATE_EXPECTATIONS_FILE (default
data/rate_expectations.json) shaped like:
[
  {
    "time": "2026-09-30T14:00:00Z",
    "central_bank": "FED",
    "currency": "USD",
    "instrument": "MANUAL",
    "implied_rate": 4.25,
    "meeting_date": "2026-11-05",
    "implied_prob": 0.65
  }
]
Missing file => empty result (warns, doesn't fail the pipeline).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fxterminal.core.logging import get_logger
from fxterminal.ingest.base import ProviderError
from fxterminal.ingest.types import RateExp

logger = get_logger(__name__)


def _parse_dt(v: object, field_name: str) -> datetime:
    if not isinstance(v, str):
        raise ProviderError(f"rate file: {field_name} must be an ISO string, got {v!r}")
    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class JsonRateExpectationProvider:
    name = "json_rate_file"

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    async def fetch_expectations(self, banks: list[str]) -> list[RateExp]:
        if not self._path.exists():
            logger.warning("rate_file_missing", path=str(self._path))
            return []
        try:
            rows = json.loads(self._path.read_text())
        except json.JSONDecodeError as e:
            raise ProviderError(f"rate file invalid JSON: {e}") from e
        if not isinstance(rows, list):
            raise ProviderError("rate file must contain a JSON array")

        out: list[RateExp] = []
        for i, r in enumerate(rows):
            try:
                out.append(
                    RateExp(
                        time=_parse_dt(r["time"], "time"),
                        central_bank=str(r["central_bank"]).upper(),
                        currency=str(r["currency"]).upper(),
                        instrument=str(r.get("instrument", "MANUAL")),
                        implied_rate=float(r["implied_rate"]),
                        meeting_date=(
                            _parse_dt(r["meeting_date"], "meeting_date")
                            if r.get("meeting_date")
                            else None
                        ),
                        implied_prob=(
                            float(r["implied_prob"]) if r.get("implied_prob") is not None else None
                        ),
                        source=self.name,
                    )
                )
            except KeyError as e:
                raise ProviderError(f"rate file row {i}: missing field {e}") from e
        if banks:
            wanted = {b.upper() for b in banks}
            out = [r for r in out if r.central_bank in wanted]
        return out
