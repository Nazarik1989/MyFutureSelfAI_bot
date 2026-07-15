"""Add persistent, versioned inbox preview drafts."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260713_0003"
down_revision: str | None = "20260712_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "draft_inbox_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("next_step", sa.Text()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("preview_message_id", sa.BigInteger()),
    )
    op.create_index("ix_draft_inbox_items_user_id", "draft_inbox_items", ["user_id"])
    op.create_index(
        "ix_draft_inbox_items_telegram_user_id",
        "draft_inbox_items",
        ["telegram_user_id"],
    )
    op.create_index("ix_draft_inbox_items_chat_id", "draft_inbox_items", ["chat_id"])
    op.create_index("ix_draft_inbox_items_status", "draft_inbox_items", ["status"])
    with op.batch_alter_table("inbox_items") as batch_op:
        batch_op.add_column(sa.Column("draft_id", sa.String(36)))
        batch_op.create_unique_constraint("uq_inbox_items_draft_id", ["draft_id"])
        batch_op.create_foreign_key(
            "fk_inbox_items_draft_id",
            "draft_inbox_items",
            ["draft_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("inbox_items") as batch_op:
        batch_op.drop_constraint("fk_inbox_items_draft_id", type_="foreignkey")
        batch_op.drop_constraint("uq_inbox_items_draft_id", type_="unique")
        batch_op.drop_column("draft_id")
    op.drop_table("draft_inbox_items")
