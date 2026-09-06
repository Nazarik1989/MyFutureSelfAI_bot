"""Add Nova durable dialogue state and observed episodic memory."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260822_0028"
down_revision: str | None = "20260817_0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "nova_dialogue_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("access_version", sa.Integer(), nullable=False),
        sa.Column("active_topic", sa.String(200), nullable=True),
        sa.Column("current_user_goal", sa.String(300), nullable=True),
        sa.Column("last_assistant_offer", sa.String(600), nullable=True),
        sa.Column(
            "last_assistant_offer_kinds",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("unresolved_question", sa.String(300), nullable=True),
        sa.Column("requested_action", sa.String(16), nullable=True),
        sa.Column("open_loops", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "telegram_user_id",
            "chat_id",
            name="uq_nova_dialogue_states_actor_chat",
        ),
        sa.CheckConstraint(
            "access_version > 0",
            name="ck_nova_dialogue_states_access_version",
        ),
        sa.CheckConstraint("revision > 0", name="ck_nova_dialogue_states_revision"),
        sa.CheckConstraint(
            "active_topic IS NULL OR length(active_topic) BETWEEN 1 AND 200",
            name="ck_nova_dialogue_states_active_topic_length",
        ),
        sa.CheckConstraint(
            "current_user_goal IS NULL OR length(current_user_goal) BETWEEN 1 AND 300",
            name="ck_nova_dialogue_states_user_goal_length",
        ),
        sa.CheckConstraint(
            "last_assistant_offer IS NULL OR length(last_assistant_offer) BETWEEN 1 AND 600",
            name="ck_nova_dialogue_states_offer_length",
        ),
        sa.CheckConstraint(
            "unresolved_question IS NULL OR length(unresolved_question) BETWEEN 1 AND 300",
            name="ck_nova_dialogue_states_question_length",
        ),
        sa.CheckConstraint(
            "requested_action IS NULL OR requested_action IN "
            "('capture', 'reminder', 'plan', 'memory', 'clarify')",
            name="ck_nova_dialogue_states_requested_action",
        ),
        sa.CheckConstraint(
            "json_array_length(last_assistant_offer_kinds) BETWEEN 0 AND 4",
            name="ck_nova_dialogue_states_offer_kinds_count",
        ),
        sa.CheckConstraint(
            "json_array_length(open_loops) BETWEEN 0 AND 5",
            name="ck_nova_dialogue_states_open_loops_count",
        ),
        sa.CheckConstraint(
            "length(CAST(last_assistant_offer_kinds AS TEXT)) BETWEEN 2 AND 1024",
            name="ck_nova_dialogue_states_offer_kinds_bytes",
        ),
        sa.CheckConstraint(
            "length(CAST(open_loops AS TEXT)) BETWEEN 2 AND 4096",
            name="ck_nova_dialogue_states_open_loops_bytes",
        ),
    )
    op.create_index(
        "ix_nova_dialogue_states_telegram_user_id",
        "nova_dialogue_states",
        ["telegram_user_id"],
    )
    op.create_index("ix_nova_dialogue_states_chat_id", "nova_dialogue_states", ["chat_id"])
    op.create_index(
        "ix_nova_dialogue_states_expires_at",
        "nova_dialogue_states",
        ["expires_at"],
    )
    op.create_index(
        "ix_nova_dialogue_states_owner_expiry",
        "nova_dialogue_states",
        ["owner_id", "expires_at", "id"],
    )

    op.create_table(
        "nova_observed_memories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("semantic_key", sa.String(32), nullable=True),
        sa.Column("normalized_value", sa.Text(), nullable=False),
        sa.Column("content_fingerprint", sa.String(64), nullable=False),
        sa.Column("source_kind", sa.String(16), nullable=False),
        sa.Column("source_session_id", sa.Integer(), nullable=False),
        sa.Column("source_message_id", sa.Integer(), nullable=False),
        sa.Column("source_receipt", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("salience", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("superseded_by_public_id", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("public_id", name="uq_nova_observed_memories_public_id"),
        sa.UniqueConstraint(
            "owner_id",
            "content_fingerprint",
            name="uq_nova_observed_memories_owner_fingerprint",
        ),
        sa.CheckConstraint(
            "category IN ('fact', 'preference', 'orientation', 'theme', 'identity')",
            name="ck_nova_observed_memories_category",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'forgotten')",
            name="ck_nova_observed_memories_status",
        ),
        sa.CheckConstraint(
            "source_kind IN ('conversation', 'explicit')",
            name="ck_nova_observed_memories_source_kind",
        ),
        sa.CheckConstraint(
            "length(normalized_value) BETWEEN 1 AND 500",
            name="ck_nova_observed_memories_value_length",
        ),
        sa.CheckConstraint(
            "length(content_fingerprint) = 64",
            name="ck_nova_observed_memories_fingerprint_length",
        ),
        sa.CheckConstraint(
            "length(source_receipt) = 64",
            name="ck_nova_observed_memories_receipt_length",
        ),
        sa.CheckConstraint(
            "salience BETWEEN 1 AND 5",
            name="ck_nova_observed_memories_salience",
        ),
        sa.CheckConstraint("revision > 0", name="ck_nova_observed_memories_revision"),
        sa.CheckConstraint(
            "superseded_by_public_id IS NULL OR length(superseded_by_public_id) = 36",
            name="ck_nova_observed_memories_superseded_id_length",
        ),
        sa.CheckConstraint(
            "semantic_key IS NULL OR "
            "(semantic_key = 'identity' AND category = 'identity') OR "
            "(semantic_key IN ('response_length', 'tone', 'reminder_style') "
            "AND category = 'preference')",
            name="ck_nova_observed_memories_semantic_key",
        ),
    )
    op.create_index(
        "ix_nova_observed_memories_owner_active",
        "nova_observed_memories",
        ["owner_id", "status", "updated_at", "id"],
    )
    op.create_index(
        "ix_nova_observed_memories_owner_category",
        "nova_observed_memories",
        ["owner_id", "category", "status", "updated_at", "id"],
    )
    op.create_index(
        "uq_nova_observed_memories_owner_active_semantic_key",
        "nova_observed_memories",
        ["owner_id", "semantic_key"],
        unique=True,
        sqlite_where=sa.text("status = 'active' AND semantic_key IS NOT NULL"),
        postgresql_where=sa.text("status = 'active' AND semantic_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_nova_observed_memories_owner_active_semantic_key",
        table_name="nova_observed_memories",
    )
    op.drop_index(
        "ix_nova_observed_memories_owner_category",
        table_name="nova_observed_memories",
    )
    op.drop_index(
        "ix_nova_observed_memories_owner_active",
        table_name="nova_observed_memories",
    )
    op.drop_table("nova_observed_memories")
    op.drop_index(
        "ix_nova_dialogue_states_owner_expiry",
        table_name="nova_dialogue_states",
    )
    op.drop_index("ix_nova_dialogue_states_expires_at", table_name="nova_dialogue_states")
    op.drop_index("ix_nova_dialogue_states_chat_id", table_name="nova_dialogue_states")
    op.drop_index(
        "ix_nova_dialogue_states_telegram_user_id",
        table_name="nova_dialogue_states",
    )
    op.drop_table("nova_dialogue_states")
