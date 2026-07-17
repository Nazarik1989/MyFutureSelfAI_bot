"""Add persistent task reminder outbox."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260717_0009"
down_revision: str | None = "20260713_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "task_reminders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "inbox_item_id",
            sa.Integer(),
            sa.ForeignKey("inbox_items.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("event_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("remind_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("delivery_key", sa.String(80), nullable=False, unique=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("claim_token", sa.String(36)),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("telegram_message_id", sa.BigInteger()),
        sa.Column("last_error_type", sa.String(120)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_task_reminders_telegram_user_id",
        "task_reminders",
        ["telegram_user_id"],
    )
    op.create_index("ix_task_reminders_remind_at", "task_reminders", ["remind_at"])
    op.create_index("ix_task_reminders_status", "task_reminders", ["status"])


def downgrade() -> None:
    op.drop_table("task_reminders")
