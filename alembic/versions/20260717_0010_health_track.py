"""Add private health check-ins and reminder preferences."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260717_0010"
down_revision: str | None = "20260717_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "health_check_ins",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("energy", sa.Integer(), nullable=False),
        sa.Column("sleep", sa.Integer(), nullable=False),
        sa.Column("mood", sa.Integer(), nullable=False),
        sa.Column("stress", sa.Integer(), nullable=False),
        sa.Column("physical_wellbeing", sa.Integer(), nullable=False),
        sa.Column("symptoms", sa.Text()),
        sa.Column("state_score", sa.Integer(), nullable=False),
        sa.CheckConstraint("energy BETWEEN 0 AND 10", name="ck_health_energy"),
        sa.CheckConstraint("sleep BETWEEN 0 AND 10", name="ck_health_sleep"),
        sa.CheckConstraint("mood BETWEEN 0 AND 10", name="ck_health_mood"),
        sa.CheckConstraint("stress BETWEEN 0 AND 10", name="ck_health_stress"),
        sa.CheckConstraint(
            "physical_wellbeing BETWEEN 0 AND 10",
            name="ck_health_physical_wellbeing",
        ),
        sa.CheckConstraint(
            "state_score BETWEEN 0 AND 100",
            name="ck_health_state_score",
        ),
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
        sa.UniqueConstraint("user_id", "local_date", name="uq_health_checkin_user_date"),
    )
    op.create_index("ix_health_check_ins_user_id", "health_check_ins", ["user_id"])
    op.create_index("ix_health_check_ins_local_date", "health_check_ins", ["local_date"])
    op.create_table(
        "health_reminder_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("local_time", sa.Time(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
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
        "ix_health_reminder_preferences_user_id",
        "health_reminder_preferences",
        ["user_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("health_reminder_preferences")
    op.drop_table("health_check_ins")
