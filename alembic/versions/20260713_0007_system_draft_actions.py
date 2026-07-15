"""Add isolated batch draft action state and last-saved receipt."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260713_0007"
down_revision: str | None = "20260713_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.add_column(sa.Column("system_pending_action", sa.String(40)))
        batch_op.add_column(sa.Column("system_draft_snapshot", sa.JSON()))
        batch_op.add_column(
            sa.Column("system_action_version", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("system_action_expires_at", sa.DateTime(timezone=True)))
        batch_op.add_column(sa.Column("last_saved_inbox_item_id", sa.Integer()))
        batch_op.add_column(sa.Column("last_saved_at", sa.DateTime(timezone=True)))
        batch_op.create_foreign_key(
            "fk_conversation_sessions_last_saved_inbox_item_id",
            "inbox_items",
            ["last_saved_inbox_item_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.drop_constraint(
            "fk_conversation_sessions_last_saved_inbox_item_id", type_="foreignkey"
        )
        batch_op.drop_column("last_saved_at")
        batch_op.drop_column("last_saved_inbox_item_id")
        batch_op.drop_column("system_action_expires_at")
        batch_op.drop_column("system_action_version")
        batch_op.drop_column("system_draft_snapshot")
        batch_op.drop_column("system_pending_action")
