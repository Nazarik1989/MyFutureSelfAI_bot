"""Initial MVP schema."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260712_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False),
        sa.Column("display_name", sa.String(120)),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("onboarding_completed", sa.Boolean(), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_users_telegram_id", "users", ["telegram_id"], unique=True)
    op.create_table(
        "vision_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("raw_answers", sa.JSON(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("values", sa.JSON(), nullable=False),
        sa.Column("desired_identity", sa.JSON(), nullable=False),
        sa.Column("constraints", sa.JSON(), nullable=False),
        sa.Column("motivation_style", sa.String(120)),
        sa.Column(
            "last_updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        *timestamps(),
    )
    op.create_table(
        "goals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("life_area", sa.String(80), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("progress_criterion", sa.Text(), nullable=False),
        sa.Column("horizon", sa.String(80), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("vision_link", sa.Text(), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_goals_user_id", "goals", ["user_id"])
    op.create_index("ix_goals_status", "goals", ["status"])
    op.create_table(
        "routines",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "goal_id", sa.Integer(), sa.ForeignKey("goals.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("frequency", sa.String(100), nullable=False),
        sa.Column("minimum_version", sa.Text(), nullable=False),
        sa.Column("normal_version", sa.Text(), nullable=False),
        sa.Column("preferred_time", sa.String(80)),
        sa.Column("status", sa.String(20), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_routines_user_id", "routines", ["user_id"])
    op.create_index("ix_routines_goal_id", "routines", ["goal_id"])
    op.create_index("ix_routines_status", "routines", ["status"])
    op.create_table(
        "inbox_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("next_step", sa.Text()),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_inbox_items_user_id", "inbox_items", ["user_id"])
    op.create_index("ix_inbox_items_status", "inbox_items", ["status"])
    op.create_table(
        "daily_check_ins",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("checkin_date", sa.Date(), nullable=False),
        sa.Column("worked", sa.Text()),
        sa.Column("did_not_work", sa.Text()),
        sa.Column("energy", sa.Integer()),
        sa.Column("obstacle", sa.Text()),
        sa.Column("tomorrow_adjustment", sa.Text()),
        sa.Column("completed_actions", sa.JSON(), nullable=False),
        sa.Column("skipped_actions", sa.JSON(), nullable=False),
        sa.UniqueConstraint("user_id", "checkin_date", name="uq_checkin_user_date"),
        *timestamps(),
    )
    op.create_index("ix_daily_check_ins_user_id", "daily_check_ins", ["user_id"])
    op.create_table(
        "onboarding_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("current_step", sa.Integer(), nullable=False),
        sa.Column("answers", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        *timestamps(),
    )


def downgrade() -> None:
    for table in (
        "onboarding_states",
        "daily_check_ins",
        "inbox_items",
        "routines",
        "goals",
        "vision_profiles",
        "users",
    ):
        op.drop_table(table)
