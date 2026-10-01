"""provider_runs health table

Revision ID: 0002_provider_runs
Revises: 0001_initial
Create Date: 2026-10-01 12:00:00+00:00

Adds `provider_runs` (per-provider health records). Metadata-driven like the
initial revision — create_all only materializes missing tables, so existing
deployments keep their data untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import fxterminal.models  # noqa: F401  (populate metadata)
from alembic import op
from fxterminal.models.base import Base

revision: str = "0002_provider_runs"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind)


def downgrade() -> None:
    op.drop_table("provider_runs")
