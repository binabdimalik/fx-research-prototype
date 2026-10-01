"""Analytical outputs: LLM-produced (validated) and deterministic scores.

Every LLM-derived row stores its evidence (exact quoted sentences + document
references) and the model/prompt version that produced it, so results are
auditable and reproducible.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from fxterminal.models.base import Base, JSONBVariant, TimestampMixin


class StanceScore(Base, TimestampMixin):
    """LLM-classified central-bank stance for one statement, with citations.

    `score` is -100 (maximally dovish) .. +100 (maximally hawkish). `label` is a
    5-bucket human-readable form. `shift_vs_prev` and `surprise_vs_pricing`
    capture *change*, which matters more than level for FX.
    """

    __tablename__ = "stance_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    central_bank: Mapped[str] = mapped_column(String(16), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)

    statement_doc_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("raw_documents.id"), nullable=False
    )
    previous_statement_doc_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("raw_documents.id"), nullable=True
    )

    score: Mapped[int] = mapped_column(Integer, nullable=False)  # -100..+100
    label: Mapped[str] = mapped_column(String(16), nullable=False)  # dovish..hawkish
    shift_vs_prev: Mapped[float | None] = mapped_column(Float, nullable=True)  # -100..+100
    surprise_vs_pricing: Mapped[float | None] = mapped_column(Float, nullable=True)

    # [{doc_id, published_at, sentence, topic}] — verbatim quotes only.
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONBVariant, nullable=False)

    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("central_bank", "statement_doc_id", name="uq_stance_statement"),
        Index("ix_stance_bank_time", "central_bank", "created_at"),
    )


class Theme(Base, TimestampMixin):
    """LLM-extracted narrative theme attached to a currency over a window."""

    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    theme: Mapped[str] = mapped_column(String(64), nullable=False)  # inflation|growth|politics|...
    sentiment: Mapped[float] = mapped_column(Float, nullable=False)  # -1..+1 for the currency
    weight: Mapped[float] = mapped_column(Float, nullable=False)  # 0..1 relevance

    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONBVariant, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (Index("ix_themes_currency_window", "currency", "window_end"),)


class CurrencyScore(Base, TimestampMixin):
    """Deterministic composite strength score per currency at an as-of time.

    `components` records each sub-score (stance, momentum, carry, themes...)
    with its weight so the total is fully explainable and recomputable.
    """

    __tablename__ = "currency_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    score: Mapped[float] = mapped_column(Float, nullable=False)
    components: Mapped[dict[str, Any]] = mapped_column(JSONBVariant, nullable=False)
    scorer_version: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("currency", "as_of", "scorer_version", name="uq_currency_score"),
        Index("ix_currency_scores_asof", "as_of"),
    )


class TradeIdea(Base, TimestampMixin):
    """Deterministic trade construction from currency scores.

    Recommendation only — nothing in this system executes trades. Every idea
    carries its risk parameters, the filters it passed, and the disclaimer.
    """

    __tablename__ = "trade_ideas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(8), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)  # long|short
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")

    # Prices are computed from market data in code — never LLM output.
    entry_low: Mapped[float] = mapped_column(Float, nullable=False)
    entry_high: Mapped[float] = mapped_column(Float, nullable=False)
    stop: Mapped[float] = mapped_column(Float, nullable=False)
    target: Mapped[float] = mapped_column(Float, nullable=False)
    risk_reward: Mapped[float] = mapped_column(Float, nullable=False)

    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    invalidation: Mapped[str] = mapped_column(Text, nullable=False)
    key_risks: Mapped[list[str]] = mapped_column(JSONBVariant, nullable=False)

    # Which deterministic filters ran and what they found (spread, event window,
    # correlation with open ideas). Flagged-but-kept ideas are recorded here.
    filters: Mapped[dict[str, Any]] = mapped_column(JSONBVariant, nullable=False)

    # Inputs used: currency score ids, stance ids — full reproducibility.
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONBVariant, nullable=False, default=dict)

    disclaimer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="Research tool output — not financial advice.",
    )
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_trade_ideas_pair_time", "pair", "created_at"),)
