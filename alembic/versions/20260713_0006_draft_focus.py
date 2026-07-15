"""Add persistent draft focus and pending action state."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260713_0006"
down_revision: str | None = "20260713_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.add_column(sa.Column("focused_draft_id", sa.String(36)))
        batch_op.add_column(sa.Column("focused_draft_version", sa.Integer()))
        batch_op.add_column(sa.Column("pending_action", sa.String(20)))
        batch_op.add_column(sa.Column("focus_expires_at", sa.DateTime(timezone=True)))
        batch_op.create_foreign_key(
            "fk_conversation_sessions_focused_draft_id",
            "draft_inbox_items",
            ["focused_draft_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.drop_constraint("fk_conversation_sessions_focused_draft_id", type_="foreignkey")
        batch_op.drop_column("focus_expires_at")
        batch_op.drop_column("pending_action")
        batch_op.drop_column("focused_draft_version")
        batch_op.drop_column("focused_draft_id")
