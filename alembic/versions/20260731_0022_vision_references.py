"""Add persistent private reference images for vision generation."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260731_0022"
down_revision: str | None = "20260731_0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vision_references",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("name", sa.String(60), nullable=False),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("mime_type", sa.String(40), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("owner_id", "sha256", name="uq_vision_reference_owner_sha256"),
        sa.CheckConstraint(
            "kind IN ('self', 'person', 'place', 'object', 'style')",
            name="ck_vision_reference_kind",
        ),
        sa.CheckConstraint("length(name) BETWEEN 1 AND 60", name="ck_vision_reference_name_length"),
        sa.CheckConstraint("width > 0 AND height > 0", name="ck_vision_reference_dimensions"),
        sa.CheckConstraint("version > 0", name="ck_vision_reference_version"),
        sa.CheckConstraint(
            "mime_type IN ('image/jpeg', 'image/png', 'image/webp')",
            name="ck_vision_reference_mime_type",
        ),
    )
    op.create_index("ix_vision_references_owner_id", "vision_references", ["owner_id"])
    op.create_index("ix_vision_references_kind", "vision_references", ["kind"])


def downgrade() -> None:
    op.drop_index("ix_vision_references_kind", table_name="vision_references")
    op.drop_index("ix_vision_references_owner_id", table_name="vision_references")
    op.drop_table("vision_references")
