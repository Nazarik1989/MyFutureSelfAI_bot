"""Add canonical structured temporal resolution to drafts and inbox."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260713_0008"
down_revision: str | None = "20260713_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("draft_inbox_items") as batch_op:
        batch_op.add_column(sa.Column("temporal_resolution", sa.JSON()))
    with op.batch_alter_table("inbox_items") as batch_op:
        batch_op.add_column(sa.Column("temporal_resolution", sa.JSON()))


def downgrade() -> None:
    with op.batch_alter_table("inbox_items") as batch_op:
        batch_op.drop_column("temporal_resolution")
    with op.batch_alter_table("draft_inbox_items") as batch_op:
        batch_op.drop_column("temporal_resolution")
