"""Add guest demo quota ledger and restart-safe sessions."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260806_0024"
down_revision: str | None = "20260805_0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS = "'thought_breakdown', 'first_step'"
_USAGE_STATUSES = "'reserved', 'succeeded', 'failed', 'expired'"
_SESSION_STATUSES = (
    "'awaiting_input', 'processing', 'result_ready', 'completed', 'cancelled', 'expired'"
)


def upgrade() -> None:
    op.create_table(
        "guest_quota_days",
        sa.Column("quota_day", sa.Date(), primary_key=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_table(
        "guest_usage_ledger",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("demo_kind", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("telegram_update_id", sa.BigInteger(), nullable=True),
        sa.Column("reservation_token", sa.String(64), nullable=False),
        sa.Column("quota_day", sa.Date(), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_guest_usage_user_idempotency",
        ),
        sa.UniqueConstraint(
            "reservation_token",
            name="uq_guest_usage_reservation_token",
        ),
        sa.CheckConstraint(
            f"demo_kind IN ({_KINDS})",
            name="ck_guest_usage_demo_kind",
        ),
        sa.CheckConstraint(
            f"status IN ({_USAGE_STATUSES})",
            name="ck_guest_usage_status",
        ),
        sa.CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 128",
            name="ck_guest_usage_idempotency_length",
        ),
        sa.CheckConstraint(
            "expires_at > reserved_at",
            name="ck_guest_usage_expiry_order",
        ),
        sa.CheckConstraint(
            "(status = 'reserved' AND completed_at IS NULL) OR "
            "(status IN ('succeeded', 'failed', 'expired') AND completed_at IS NOT NULL)",
            name="ck_guest_usage_completion_state",
        ),
        sa.CheckConstraint(
            "provider_started_at IS NULL OR "
            "(provider_started_at >= reserved_at AND provider_started_at < expires_at)",
            name="ck_guest_usage_provider_start_window",
        ),
        sa.CheckConstraint(
            "status <> 'succeeded' OR provider_started_at IS NOT NULL",
            name="ck_guest_usage_success_requires_provider_start",
        ),
    )
    op.create_index(
        "uq_guest_usage_one_reserved_per_user",
        "guest_usage_ledger",
        ["user_id"],
        unique=True,
        sqlite_where=sa.text("status = 'reserved'"),
        postgresql_where=sa.text("status = 'reserved'"),
    )
    op.create_index(
        "ix_guest_usage_quota_status_expiry",
        "guest_usage_ledger",
        ["quota_day", "status", "expires_at"],
    )
    op.create_index(
        "ix_guest_usage_user_status",
        "guest_usage_ledger",
        ["user_id", "status"],
    )
    op.create_index(
        "ix_guest_usage_provider_started_at",
        "guest_usage_ledger",
        ["provider_started_at"],
    )
    op.create_index(
        "ix_guest_usage_unstarted_status_expiry",
        "guest_usage_ledger",
        ["status", "expires_at"],
        sqlite_where=sa.text("provider_started_at IS NULL"),
        postgresql_where=sa.text("provider_started_at IS NULL"),
    )
    op.create_table(
        "guest_demo_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("access_version", sa.Integer(), nullable=False),
        sa.Column("demo_kind", sa.String(32), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("prompt_message_id", sa.BigInteger(), nullable=True),
        sa.Column("consumed_update_id", sa.BigInteger(), nullable=True),
        sa.Column("consumed_message_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "usage_id",
            sa.Integer(),
            sa.ForeignKey("guest_usage_ledger.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("result_payload", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("result_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.UniqueConstraint(
            "user_id",
            "chat_id",
            name="uq_guest_demo_session_user_chat",
        ),
        sa.CheckConstraint(
            f"demo_kind IN ({_KINDS})",
            name="ck_guest_demo_session_kind",
        ),
        sa.CheckConstraint(
            f"status IN ({_SESSION_STATUSES})",
            name="ck_guest_demo_session_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_guest_demo_session_version"),
        sa.CheckConstraint(
            "access_version > 0",
            name="ck_guest_demo_session_access_version",
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_guest_demo_session_expiry_order",
        ),
        sa.CheckConstraint(
            "(status = 'result_ready' AND result_payload IS NOT NULL "
            "AND result_expires_at IS NOT NULL) OR "
            "(status <> 'result_ready' AND result_payload IS NULL "
            "AND result_expires_at IS NULL)",
            name="ck_guest_demo_session_result_state",
        ),
    )
    op.create_index(
        "ix_guest_demo_sessions_user_id",
        "guest_demo_sessions",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_guest_demo_sessions_user_id", table_name="guest_demo_sessions")
    op.drop_table("guest_demo_sessions")
    op.drop_index("ix_guest_usage_unstarted_status_expiry", table_name="guest_usage_ledger")
    op.drop_index("ix_guest_usage_provider_started_at", table_name="guest_usage_ledger")
    op.drop_index("ix_guest_usage_user_status", table_name="guest_usage_ledger")
    op.drop_index("ix_guest_usage_quota_status_expiry", table_name="guest_usage_ledger")
    op.drop_index("uq_guest_usage_one_reserved_per_user", table_name="guest_usage_ledger")
    op.drop_table("guest_usage_ledger")
    op.drop_table("guest_quota_days")
