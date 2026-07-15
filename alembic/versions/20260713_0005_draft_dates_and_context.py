"""Persist resolved draft dates and pending calendar choices."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260713_0005"
down_revision: str | None = "20260713_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("draft_inbox_items") as batch_op:
        batch_op.add_column(sa.Column("description", sa.Text()))
        batch_op.add_column(sa.Column("resolved_date", sa.Date()))
    with op.batch_alter_table("inbox_items") as batch_op:
        batch_op.add_column(sa.Column("description", sa.Text()))
        batch_op.add_column(sa.Column("resolved_date", sa.Date()))
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.add_column(sa.Column("pending_date_options", sa.JSON()))
        batch_op.add_column(sa.Column("resolved_date", sa.Date()))


def downgrade() -> None:
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.drop_column("resolved_date")
        batch_op.drop_column("pending_date_options")
    with op.batch_alter_table("inbox_items") as batch_op:
        batch_op.drop_column("resolved_date")
        batch_op.drop_column("description")
    with op.batch_alter_table("draft_inbox_items") as batch_op:
        batch_op.drop_column("resolved_date")
        batch_op.drop_column("description")
