"""Add owner-scoped Nova memory and metadata-only audit history."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260811_0026"
down_revision: str | None = "20260810_0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "nova_memory_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("category", sa.String(24), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_fingerprint", sa.String(64), nullable=False),
        sa.Column(
            "important",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
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
        sa.UniqueConstraint(
            "public_id",
            name="uq_nova_memory_items_public_id",
        ),
        sa.UniqueConstraint(
            "owner_id",
            "content_fingerprint",
            name="uq_nova_memory_items_owner_fingerprint",
        ),
        sa.CheckConstraint(
            "category IN ('about_me', 'interaction', 'orientation')",
            name="ck_nova_memory_items_category",
        ),
        sa.CheckConstraint(
            "length(content) BETWEEN 1 AND 500",
            name="ck_nova_memory_items_content_length",
        ),
        sa.CheckConstraint(
            "length(content_fingerprint) = 64",
            name="ck_nova_memory_items_fingerprint_length",
        ),
        sa.CheckConstraint(
            "version > 0",
            name="ck_nova_memory_items_version",
        ),
    )
    op.create_index(
        "ix_nova_memory_items_owner_list",
        "nova_memory_items",
        ["owner_id", "important", "updated_at", "id"],
    )
    op.create_index(
        "ix_nova_memory_items_owner_category_list",
        "nova_memory_items",
        ["owner_id", "category", "important", "updated_at", "id"],
    )

    op.create_table(
        "nova_memory_changes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("memory_public_id", sa.String(36), nullable=True),
        sa.Column("operation", sa.String(24), nullable=False),
        sa.Column("category", sa.String(24), nullable=True),
        sa.Column("resulting_version", sa.Integer(), nullable=True),
        sa.Column(
            "affected_count",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "operation IN ('created', 'updated', 'importance_changed', 'deleted', 'deleted_all')",
            name="ck_nova_memory_changes_operation",
        ),
        sa.CheckConstraint(
            "category IS NULL OR category IN ('about_me', 'interaction', 'orientation')",
            name="ck_nova_memory_changes_category",
        ),
        sa.CheckConstraint(
            "memory_public_id IS NULL OR length(memory_public_id) = 36",
            name="ck_nova_memory_changes_public_id_length",
        ),
        sa.CheckConstraint(
            "resulting_version IS NULL OR resulting_version > 0",
            name="ck_nova_memory_changes_resulting_version",
        ),
        sa.CheckConstraint(
            "affected_count > 0",
            name="ck_nova_memory_changes_affected_count",
        ),
        sa.CheckConstraint(
            "(operation IN ('created', 'updated', 'importance_changed') "
            "AND memory_public_id IS NOT NULL AND category IS NOT NULL "
            "AND resulting_version IS NOT NULL AND affected_count = 1) OR "
            "(operation = 'deleted' AND memory_public_id IS NOT NULL "
            "AND category IS NOT NULL AND resulting_version IS NULL "
            "AND affected_count = 1) OR "
            "(operation = 'deleted_all' AND memory_public_id IS NULL "
            "AND category IS NULL AND resulting_version IS NULL)",
            name="ck_nova_memory_changes_shape",
        ),
    )
    op.create_index(
        "ix_nova_memory_changes_owner_created",
        "nova_memory_changes",
        ["owner_id", "created_at", "id"],
    )
    op.create_index(
        "ix_nova_memory_changes_item_history",
        "nova_memory_changes",
        ["owner_id", "memory_public_id", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_nova_memory_changes_item_history",
        table_name="nova_memory_changes",
    )
    op.drop_index(
        "ix_nova_memory_changes_owner_created",
        table_name="nova_memory_changes",
    )
    op.drop_table("nova_memory_changes")
    op.drop_index(
        "ix_nova_memory_items_owner_category_list",
        table_name="nova_memory_items",
    )
    op.drop_index(
        "ix_nova_memory_items_owner_list",
        table_name="nova_memory_items",
    )
    op.drop_table("nova_memory_items")
