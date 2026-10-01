"""Operational metadata: one row per provider run.

`provider_runs` powers the Data Health view — per-source last-success,
latency and error counts. Every `_guard` call in the ingest pipeline writes
one row, including failed runs (which are the ones that matter most).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from fxterminal.models.base import Base, utcnow


class ProviderRun(Base):
    __tablename__ = "provider_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(96), nullable=False)  # e.g. "docs:fed_rss"
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(String(512), nullable=True)

    __table_args__ = (Index("ix_provider_runs_name_time", "provider", "started_at"),)
