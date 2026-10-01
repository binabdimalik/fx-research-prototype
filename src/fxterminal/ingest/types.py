"""Normalized record types passed between providers and the store layer.

Providers return these plain dataclasses; `store.py` maps them onto ORM rows.
Keeping providers ORM-free makes them trivially testable with fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

DocType = Literal["statement", "minutes", "speech", "news", "macro_release"]
Impact = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RawDoc:
    """A document to persist immutably in `raw_documents`."""

    source: str  # provider id, e.g. "fed_rss"
    doc_type: DocType
    title: str
    published_at: datetime  # tz-aware UTC
    raw_text: str
    url: str | None = None
    central_bank: str | None = None  # FED|ECB|BOJ|BOE
    currency: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PriceBar:
    """One FX OHLC bar."""

    time: datetime  # tz-aware UTC
    pair: str  # e.g. "EURUSD"
    timeframe: str  # "1d" for Phase 2
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    spread_pips: float | None = None
    source: str = ""


@dataclass(frozen=True)
class MacroObs:
    """A macro release observation (actual/forecast/previous for one series)."""

    time: datetime  # tz-aware UTC, release time
    currency: str
    name: str  # canonical event name, e.g. "US CPI YoY"
    impact: Impact = "medium"
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RateExp:
    """A market-implied policy-rate expectation snapshot."""

    time: datetime  # tz-aware UTC, when the quote was taken
    central_bank: str
    currency: str
    instrument: str  # e.g. "FED_FUNDS_FUT_Z5", "MANUAL"
    implied_rate: float  # percent
    meeting_date: datetime | None = None
    implied_prob: float | None = None
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestReport:
    """Outcome of one provider run — used by pipeline + CLI output."""

    provider: str
    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    rejected: int = 0  # failed validation (null/negative/impossible/jump)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
