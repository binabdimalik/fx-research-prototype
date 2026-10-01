"""Immutable raw source documents: CB statements, minutes, speeches, news, macro.

Raw payloads are append-only: we never mutate `raw_text` after insert, so every
downstream score can be traced back to exactly what the source said and when we
fetched it. Deduplication is enforced by the unique `content_hash`.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from fxterminal.models.base import Base, JSONBVariant, utcnow


class RawDocument(Base):
    __tablename__ = "raw_documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # Provenance
    source: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "fed_fomc_rss"
    doc_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # statement|minutes|speech|news|macro_release
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    # Classification (nullable: news may map to multiple/no currencies)
    central_bank: Mapped[str | None] = mapped_column(String(16), nullable=True)  # FED|ECB|BOJ|BOE
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)

    title: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # Dedup key: sha256 of normalized (source, url, title, body). Unique ⇒ retries safe.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSONBVariant, nullable=False, default=dict)

    __table_args__ = (
        UniqueConstraint("content_hash", name="content_hash"),
        Index("ix_raw_documents_published_at", "published_at"),
        Index("ix_raw_documents_bank_type_time", "central_bank", "doc_type", "published_at"),
        Index("ix_raw_documents_currency_time", "currency", "published_at"),
    )

    @staticmethod
    def compute_content_hash(source: str, url: str | None, title: str, body: str) -> str:
        """Stable dedup hash. Normalizes whitespace so trivial reposts dedupe."""
        normalized = "|".join(
            [
                source.strip().lower(),
                (url or "").strip().lower(),
                " ".join(title.split()),
                " ".join(body.split()),
            ]
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
