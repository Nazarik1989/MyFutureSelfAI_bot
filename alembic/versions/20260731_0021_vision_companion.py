"""Add opt-in vision companion preferences and daily check-ins."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260731_0021"
down_revision: str | None = "20260725_0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vision_companion_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "vision_item_id",
            sa.Integer(),
            sa.ForeignKey("vision_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("morning_time", sa.Time(), nullable=False),
        sa.Column("evening_time", sa.Time(), nullable=False),
        sa.Column("extra_per_day", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("extra_times", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("extra_per_day BETWEEN 0 AND 3", name="ck_vision_companion_extra_count"),
    )
    op.create_index(
        "ix_vision_companion_preferences_owner_id",
        "vision_companion_preferences",
        ["owner_id"],
        unique=True,
    )
    op.create_index(
        "ix_vision_companion_preferences_vision_item_id",
        "vision_companion_preferences",
        ["vision_item_id"],
        unique=True,
    )

    op.create_table(
        "vision_companion_checkins",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "vision_item_id",
            sa.Integer(),
            sa.ForeignKey("vision_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("local_date", sa.Date(), nullable=False),
        sa.Column("moment", sa.String(16), nullable=False),
        sa.Column("response", sa.String(16), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "owner_id",
            "vision_item_id",
            "local_date",
            "moment",
            name="uq_vision_companion_checkin_day",
        ),
        sa.CheckConstraint(
            "moment IN ('morning', 'evening', 'extra')",
            name="ck_vision_companion_checkin_moment",
        ),
        sa.CheckConstraint(
            "response IN ('committed', 'pause', 'done', 'partial', 'missed', 'later')",
            name="ck_vision_companion_checkin_response",
        ),
    )
    op.create_index(
        "ix_vision_companion_checkins_owner_id",
        "vision_companion_checkins",
        ["owner_id"],
    )
    op.create_index(
        "ix_vision_companion_checkins_vision_item_id",
        "vision_companion_checkins",
        ["vision_item_id"],
    )
    op.create_index(
        "ix_vision_companion_checkins_local_date",
        "vision_companion_checkins",
        ["local_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_vision_companion_checkins_local_date", table_name="vision_companion_checkins")
    op.drop_index(
        "ix_vision_companion_checkins_vision_item_id", table_name="vision_companion_checkins"
    )
    op.drop_index("ix_vision_companion_checkins_owner_id", table_name="vision_companion_checkins")
    op.drop_table("vision_companion_checkins")
    op.drop_index(
        "ix_vision_companion_preferences_vision_item_id",
        table_name="vision_companion_preferences",
    )
    op.drop_index(
        "ix_vision_companion_preferences_owner_id",
        table_name="vision_companion_preferences",
    )
    op.drop_table("vision_companion_preferences")
