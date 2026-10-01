"""Declarative base and shared column types."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on Postgres, plain JSON on SQLite — keeps dev/tests dependency-free
# while giving us real JSONB indexing/operators in production.
JSONBVariant = JSON().with_variant(JSONB, "postgresql")


def utcnow() -> datetime:
    """Timezone-aware UTC now. All timestamps in this system are UTC."""
    return datetime.now(UTC)


# Explicit constraint naming so Alembic diffs are stable across dialects
# (SQLite otherwise auto-names constraints unpredictably).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """`created_at` for append-only analytical tables."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
