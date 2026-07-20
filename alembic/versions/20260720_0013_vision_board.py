"""Add owner-scoped vision board items and persistent drafts."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260720_0013"
down_revision: str | None = "20260720_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CATEGORIES = (
    "health_energy",
    "relationships_family",
    "work_purpose",
    "money",
    "home",
    "travel",
    "growth_creativity",
    "other",
)


def upgrade() -> None:
    op.create_table(
        "vision_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("category", sa.String(40), nullable=False),
        sa.Column("wish_text", sa.Text(), nullable=False),
        sa.Column("why_text", sa.Text(), nullable=True),
        sa.Column("target_date", sa.Date(), nullable=True),
        sa.Column("first_step", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "linked_task_id",
            sa.Integer(),
            sa.ForeignKey("inbox_items.id", ondelete="SET NULL"),
            nullable=True,
            unique=True,
        ),
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
        sa.CheckConstraint(
            f"category IN ({', '.join(repr(value) for value in CATEGORIES)})",
            name="ck_vision_item_category",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'achieved', 'archived')",
            name="ck_vision_item_status",
        ),
    )
    op.create_index("ix_vision_items_owner_id", "vision_items", ["owner_id"])
    op.create_index("ix_vision_items_category", "vision_items", ["category"])
    op.create_index("ix_vision_items_status", "vision_items", ["status"])
    op.create_table(
        "vision_drafts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("step", sa.String(30), nullable=False, server_default="category"),
        sa.Column("category", sa.String(40), nullable=True),
        sa.Column("wish_text", sa.Text(), nullable=True),
        sa.Column("why_text", sa.Text(), nullable=True),
        sa.Column("target_date", sa.Date(), nullable=True),
        sa.Column("first_step", sa.Text(), nullable=True),
        sa.Column(
            "editing_item_id",
            sa.Integer(),
            sa.ForeignKey("vision_items.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("edit_field", sa.String(30), nullable=True),
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
    )
    op.create_index(
        "ix_vision_drafts_owner_id",
        "vision_drafts",
        ["owner_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_vision_drafts_owner_id", table_name="vision_drafts")
    op.drop_table("vision_drafts")
    op.drop_index("ix_vision_items_status", table_name="vision_items")
    op.drop_index("ix_vision_items_category", table_name="vision_items")
    op.drop_index("ix_vision_items_owner_id", table_name="vision_items")
    op.drop_table("vision_items")
