"""Market data: FX prices, rate expectations, macro calendar events.

`fx_prices` and `rate_expectations` become TimescaleDB hypertables on Postgres
(see the initial Alembic migration); on SQLite they are ordinary tables.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from fxterminal.models.base import Base


class FxPrice(Base):
    """One OHLCV bar for an FX pair. `source` is part of the dedup key so the
    same bar from two providers never collides."""

    __tablename__ = "fx_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    pair: Mapped[str] = mapped_column(String(8), nullable=False)  # e.g. "EURUSD"
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1d")

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Typical spread at bar close in pips, when the provider reports it —
    # feeds the deterministic spread filter for trade ideas.
    spread_pips: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        # `time` must be part of any unique constraint on a TimescaleDB hypertable.
        UniqueConstraint("time", "pair", "timeframe", "source", name="uq_fx_price_bar"),
        Index("ix_fx_prices_pair_time", "pair", "time"),
    )


class RateExpectation(Base):
    """Market-implied policy rate for a central bank at a point in time.

    `instrument` identifies the pricing source (e.g. "FED_FUNDS_FUT_Z5",
    "OIS_3M") so methodologies can coexist and be audited.
    """

    __tablename__ = "rate_expectations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    central_bank: Mapped[str] = mapped_column(String(16), nullable=False)  # FED|ECB|BOJ|BOE
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    instrument: Mapped[str] = mapped_column(String(64), nullable=False)

    # Meeting the expectation refers to; null for generic tenor strips.
    meeting_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    implied_rate: Mapped[float] = mapped_column(Float, nullable=False)  # percent, e.g. 4.375
    # Optional richer shapes: implied probability of hike/cut, etc.
    implied_prob: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint("time", "central_bank", "instrument", "source", name="uq_rate_exp"),
        Index("ix_rate_exp_bank_time", "central_bank", "time"),
    )


class MacroEvent(Base):
    """Economic calendar entry — used by the deterministic event-blackout filter
    and later by the backtester for release-timing realism."""

    __tablename__ = "macro_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)  # e.g. "US CPI YoY"
    impact: Mapped[str] = mapped_column(String(8), nullable=False, default="medium")

    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint("time", "currency", "name", "source", name="uq_macro_event"),
        Index("ix_macro_events_time", "time"),
        Index("ix_macro_events_currency_time", "currency", "time"),
    )
