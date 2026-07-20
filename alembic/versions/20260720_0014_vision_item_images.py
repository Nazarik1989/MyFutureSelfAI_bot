"""Add normalized owner-scoped images for vision items."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260720_0014"
down_revision: str | None = "20260720_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vision_item_images",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "vision_item_id",
            sa.Integer(),
            sa.ForeignKey("vision_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("mime_type", sa.String(40), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
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
        sa.CheckConstraint(
            "width > 0 AND height > 0",
            name="ck_vision_item_image_dimensions",
        ),
        sa.CheckConstraint("version > 0", name="ck_vision_item_image_version"),
        sa.CheckConstraint(
            "mime_type IN ('image/jpeg', 'image/png', 'image/webp')",
            name="ck_vision_item_image_mime_type",
        ),
    )
    op.create_index(
        "ix_vision_item_images_vision_item_id",
        "vision_item_images",
        ["vision_item_id"],
        unique=True,
    )
    op.create_index(
        "ix_vision_item_images_owner_id",
        "vision_item_images",
        ["owner_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_vision_item_images_owner_id", table_name="vision_item_images")
    op.drop_index("ix_vision_item_images_vision_item_id", table_name="vision_item_images")
    op.drop_table("vision_item_images")
