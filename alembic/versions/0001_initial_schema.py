"""initial schema

Revision ID: 0001_initial
Revises: None
Create Date: 2026-09-30 21:00:00+00:00

The first revision materializes the schema directly from SQLAlchemy metadata
(the metadata is versioned with the code, so this is deterministic). On
Postgres it additionally converts `fx_prices` and `rate_expectations` into
TimescaleDB hypertables — guarded so plain Postgres still works.
"""

from __future__ import annotations

from collections.abc import Sequence

import fxterminal.models  # noqa: F401  (populate metadata)
from alembic import op
from fxterminal.models.base import Base

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Runs on plain Postgres too: missing extension/binaries degrade to a NOTICE.
_EXTENSION_SQL = """
DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'timescaledb not available (%); continuing without hypertables', SQLERRM;
END
$$;
"""

_HYPERTABLE_SQL = """
DO $$
BEGIN
    PERFORM create_hypertable('%(table)s', 'time', if_not_exists => TRUE);
EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'timescaledb extension not installed; skipping hypertable for %(table)s';
END
$$;
"""


def _create_hypertables() -> None:
    for table in ("fx_prices", "rate_expectations"):
        op.execute(_HYPERTABLE_SQL % {"table": table})


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind)
    if bind.dialect.name == "postgresql":
        bind.exec_driver_sql(_EXTENSION_SQL)
        _create_hypertables()


def downgrade() -> None:
    bind = op.get_bind()
    # Hypertables drop as ordinary tables; no special handling needed.
    Base.metadata.drop_all(bind)
