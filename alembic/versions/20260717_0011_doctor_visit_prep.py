"""Add private doctor visit preparation records."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260717_0011"
down_revision: str | None = "20260717_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "doctor_visit_preps",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("duration", sa.Text(), nullable=False),
        sa.Column("symptoms", sa.Text(), nullable=False),
        sa.Column("medications", sa.Text()),
        sa.Column("questions", sa.Text()),
        sa.Column("health_snapshot", sa.JSON(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "appointment_inbox_item_id",
            sa.Integer(),
            sa.ForeignKey("inbox_items.id", ondelete="SET NULL"),
            nullable=True,
            unique=True,
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
    )
    op.create_index("ix_doctor_visit_preps_user_id", "doctor_visit_preps", ["user_id"])


def downgrade() -> None:
    op.drop_table("doctor_visit_preps")
