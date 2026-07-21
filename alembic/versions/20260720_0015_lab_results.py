"""Add owner-scoped normalized lab result documents."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260720_0015"
down_revision: str | None = "20260720_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "lab_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("document_date", sa.Date(), nullable=True),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="saved"),
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
        sa.UniqueConstraint("id", "owner_id", name="uq_lab_document_id_owner"),
        sa.CheckConstraint("page_count > 0", name="ck_lab_document_page_count"),
        sa.CheckConstraint(
            "length(title) BETWEEN 1 AND 200",
            name="ck_lab_document_title_length",
        ),
        sa.CheckConstraint("version > 0", name="ck_lab_document_version"),
        sa.CheckConstraint(
            "source_type IN ('image', 'pdf')",
            name="ck_lab_document_source_type",
        ),
        sa.CheckConstraint("status IN ('saved')", name="ck_lab_document_status"),
    )
    op.create_index("ix_lab_documents_owner_id", "lab_documents", ["owner_id"])
    op.create_index("ix_lab_documents_document_date", "lab_documents", ["document_date"])
    op.create_index("ix_lab_documents_status", "lab_documents", ["status"])

    op.create_table(
        "lab_document_pages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("page_index", sa.Integer(), nullable=False),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("mime_type", sa.String(40), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "owner_id"],
            ["lab_documents.id", "lab_documents.owner_id"],
            ondelete="CASCADE",
            name="fk_lab_page_document_owner",
        ),
        sa.UniqueConstraint("document_id", "page_index", name="uq_lab_page_document_index"),
        sa.CheckConstraint("page_index >= 0", name="ck_lab_page_index"),
        sa.CheckConstraint("width > 0 AND height > 0", name="ck_lab_page_dimensions"),
        sa.CheckConstraint("length(image_bytes) > 0", name="ck_lab_page_has_bytes"),
        sa.CheckConstraint("mime_type = 'image/jpeg'", name="ck_lab_page_mime_type"),
        sa.CheckConstraint("length(sha256) = 64", name="ck_lab_page_sha256_length"),
    )
    op.create_index(
        "ix_lab_document_pages_document_id",
        "lab_document_pages",
        ["document_id"],
    )
    op.create_index("ix_lab_document_pages_owner_id", "lab_document_pages", ["owner_id"])
    op.create_index(
        "ix_lab_document_pages_owner_document",
        "lab_document_pages",
        ["owner_id", "document_id"],
    )

    op.create_table(
        "lab_delete_confirmations",
        sa.Column("token", sa.String(32), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("document_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("document_version > 0", name="ck_lab_delete_version"),
        sa.CheckConstraint(
            "status IN ('pending', 'consumed')",
            name="ck_lab_delete_status",
        ),
    )
    op.create_index(
        "ix_lab_delete_confirmations_owner_id",
        "lab_delete_confirmations",
        ["owner_id"],
    )
    op.create_index(
        "ix_lab_delete_confirmations_document_id",
        "lab_delete_confirmations",
        ["document_id"],
    )
    op.create_index(
        "ix_lab_delete_confirmations_status",
        "lab_delete_confirmations",
        ["status"],
    )
    op.create_index(
        "ix_lab_delete_confirmations_expires_at",
        "lab_delete_confirmations",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_lab_delete_confirmations_expires_at",
        table_name="lab_delete_confirmations",
    )
    op.drop_index("ix_lab_delete_confirmations_status", table_name="lab_delete_confirmations")
    op.drop_index(
        "ix_lab_delete_confirmations_document_id",
        table_name="lab_delete_confirmations",
    )
    op.drop_index("ix_lab_delete_confirmations_owner_id", table_name="lab_delete_confirmations")
    op.drop_table("lab_delete_confirmations")
    op.drop_index(
        "ix_lab_document_pages_owner_document",
        table_name="lab_document_pages",
    )
    op.drop_index("ix_lab_document_pages_owner_id", table_name="lab_document_pages")
    op.drop_index("ix_lab_document_pages_document_id", table_name="lab_document_pages")
    op.drop_table("lab_document_pages")
    op.drop_index("ix_lab_documents_status", table_name="lab_documents")
    op.drop_index("ix_lab_documents_document_date", table_name="lab_documents")
    op.drop_index("ix_lab_documents_owner_id", table_name="lab_documents")
    op.drop_table("lab_documents")
