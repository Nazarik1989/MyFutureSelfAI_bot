"""Add persistent bounded conversation context."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260713_0004"
down_revision: str | None = "20260713_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("current_topic", sa.String(200)),
        sa.Column("summary", sa.Text()),
        sa.Column(
            "active_draft_id",
            sa.String(36),
            sa.ForeignKey("draft_inbox_items.id", ondelete="SET NULL"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.UniqueConstraint(
            "telegram_user_id", "chat_id", name="uq_conversation_session_user_chat"
        ),
    )
    op.create_index(
        "ix_conversation_sessions_telegram_user_id",
        "conversation_sessions",
        ["telegram_user_id"],
    )
    op.create_index("ix_conversation_sessions_chat_id", "conversation_sessions", ["chat_id"])
    op.create_index("ix_conversation_sessions_expires_at", "conversation_sessions", ["expires_at"])
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("intent", sa.String(40), nullable=False),
    )
    op.create_index("ix_conversation_messages_session_id", "conversation_messages", ["session_id"])
    op.create_index("ix_conversation_messages_timestamp", "conversation_messages", ["timestamp"])


def downgrade() -> None:
    op.drop_table("conversation_messages")
    op.drop_table("conversation_sessions")
