"""Add normalized recurring task reminder schedules and occurrences."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260810_0025"
down_revision: str | None = "20260806_0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recurring_task_reminder_schedules",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("inbox_item_id", sa.Integer(), nullable=False),
        sa.Column(
            "recurrence_kind",
            sa.String(16),
            nullable=False,
            server_default="daily",
        ),
        sa.Column("local_time", sa.Time(), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column(
            "timezone_source",
            sa.String(16),
            nullable=False,
            server_default="profile",
        ),
        sa.Column("start_local_date", sa.Date(), nullable=False),
        sa.Column("next_occurrence_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
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
        sa.ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_recurring_schedule_inbox_owner",
        ),
        sa.UniqueConstraint(
            "inbox_item_id",
            name="uq_recurring_schedule_inbox_item",
        ),
        sa.CheckConstraint(
            "recurrence_kind IN ('daily')",
            name="ck_recurring_schedule_kind",
        ),
        sa.CheckConstraint(
            "timezone_source IN ('profile', 'explicit')",
            name="ck_recurring_schedule_timezone_source",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'disabled', 'completed')",
            name="ck_recurring_schedule_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_recurring_schedule_version"),
        sa.CheckConstraint(
            "length(timezone) BETWEEN 1 AND 64",
            name="ck_recurring_schedule_timezone_length",
        ),
    )
    op.create_index(
        "ix_recurring_schedule_due",
        "recurring_task_reminder_schedules",
        ["status", "next_occurrence_at"],
    )
    op.create_index(
        "ix_recurring_schedule_owner_status",
        "recurring_task_reminder_schedules",
        ["owner_id", "status", "next_occurrence_at"],
    )

    op.create_table(
        "recurring_task_reminder_occurrences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "schedule_id",
            sa.Integer(),
            sa.ForeignKey("recurring_task_reminder_schedules.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("schedule_version", sa.Integer(), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("delivery_key", sa.String(128), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("claim_token", sa.String(36), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
        sa.Column("last_error_type", sa.String(120), nullable=True),
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
            "schedule_id",
            "schedule_version",
            "scheduled_for",
            name="uq_recurring_occurrence_generation_time",
        ),
        sa.UniqueConstraint(
            "schedule_id",
            "schedule_version",
            "local_date",
            name="uq_recurring_occurrence_generation_date",
        ),
        sa.UniqueConstraint(
            "delivery_key",
            name="uq_recurring_occurrence_delivery_key",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'sent', 'skipped_stale', 'cancelled')",
            name="ck_recurring_occurrence_status",
        ),
        sa.CheckConstraint(
            "schedule_version > 0",
            name="ck_recurring_occurrence_schedule_version",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name="ck_recurring_occurrence_attempt_count",
        ),
        sa.CheckConstraint(
            "(status = 'processing' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL) OR "
            "(status <> 'processing' AND claim_token IS NULL AND claimed_at IS NULL)",
            name="ck_recurring_occurrence_claim_state",
        ),
        sa.CheckConstraint(
            "(status = 'sent' AND sent_at IS NOT NULL) OR (status <> 'sent' AND sent_at IS NULL)",
            name="ck_recurring_occurrence_sent_state",
        ),
        sa.CheckConstraint(
            "(delivery_started_at IS NULL AND status <> 'sent') OR "
            "(delivery_started_at IS NOT NULL AND status IN ('processing', 'sent'))",
            name="ck_recurring_occurrence_delivery_started_state",
        ),
        sa.CheckConstraint(
            "length(delivery_key) BETWEEN 1 AND 128",
            name="ck_recurring_occurrence_delivery_key_length",
        ),
    )
    op.create_index(
        "ix_recurring_occurrence_due",
        "recurring_task_reminder_occurrences",
        ["status", "scheduled_for", "next_attempt_at"],
    )
    op.create_index(
        "ix_recurring_occurrence_claims",
        "recurring_task_reminder_occurrences",
        ["status", "delivery_started_at", "claimed_at"],
    )
    op.create_index(
        "ix_recurring_occurrence_schedule_history",
        "recurring_task_reminder_occurrences",
        ["schedule_id", "created_at"],
    )
    op.create_index(
        "ix_recurring_occurrence_cleanup",
        "recurring_task_reminder_occurrences",
        ["status", "updated_at"],
    )
    op.create_index(
        "uq_recurring_occurrence_sent_date",
        "recurring_task_reminder_occurrences",
        ["schedule_id", "local_date"],
        unique=True,
        sqlite_where=sa.text("status = 'sent'"),
        postgresql_where=sa.text("status = 'sent'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_recurring_occurrence_sent_date",
        table_name="recurring_task_reminder_occurrences",
    )
    op.drop_index(
        "ix_recurring_occurrence_cleanup",
        table_name="recurring_task_reminder_occurrences",
    )
    op.drop_index(
        "ix_recurring_occurrence_schedule_history",
        table_name="recurring_task_reminder_occurrences",
    )
    op.drop_index(
        "ix_recurring_occurrence_claims",
        table_name="recurring_task_reminder_occurrences",
    )
    op.drop_index(
        "ix_recurring_occurrence_due",
        table_name="recurring_task_reminder_occurrences",
    )
    op.drop_table("recurring_task_reminder_occurrences")
    op.drop_index(
        "ix_recurring_schedule_owner_status",
        table_name="recurring_task_reminder_schedules",
    )
    op.drop_index(
        "ix_recurring_schedule_due",
        table_name="recurring_task_reminder_schedules",
    )
    op.drop_table("recurring_task_reminder_schedules")
