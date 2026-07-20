"""Add owner-scoped user doctor-search location."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260720_0012"
down_revision: str | None = "20260717_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("location_city", sa.String(120), nullable=True))
    op.add_column(
        "users",
        sa.Column("location_fallback_city", sa.String(120), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "location_fallback_city")
    op.drop_column("users", "location_city")
