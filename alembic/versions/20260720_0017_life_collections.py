"""Add owner-scoped life areas and smart collection links."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260720_0017"
down_revision: str | None = "20260720_0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "life_collections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("normalized_name", sa.String(100), nullable=False),
        sa.Column("starter_key", sa.String(64), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("id", "owner_id", name="uq_life_collection_id_owner"),
        sa.UniqueConstraint(
            "owner_id", "normalized_name", name="uq_life_collection_owner_normalized_name"
        ),
        sa.CheckConstraint("kind IN ('topic', 'project', 'list')", name="ck_life_collection_kind"),
        sa.CheckConstraint("status IN ('active', 'archived')", name="ck_life_collection_status"),
        sa.CheckConstraint("version > 0", name="ck_life_collection_version"),
        sa.CheckConstraint("length(name) BETWEEN 1 AND 100", name="ck_life_collection_name_length"),
        sa.CheckConstraint(
            "length(normalized_name) BETWEEN 1 AND 100",
            name="ck_life_collection_normalized_name_length",
        ),
    )
    op.create_index("ix_life_collections_owner_id", "life_collections", ["owner_id"])
    op.create_index("ix_life_collections_kind", "life_collections", ["kind"])
    op.create_index("ix_life_collections_status", "life_collections", ["status"])

    op.create_table(
        "life_collection_aliases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("collection_id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.String(100), nullable=False),
        sa.Column("normalized_alias", sa.String(100), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_alias_owner",
        ),
        sa.UniqueConstraint(
            "owner_id", "normalized_alias", name="uq_life_collection_alias_owner_name"
        ),
        sa.CheckConstraint(
            "length(alias) BETWEEN 1 AND 100", name="ck_life_collection_alias_length"
        ),
        sa.CheckConstraint(
            "length(normalized_alias) BETWEEN 1 AND 100",
            name="ck_life_collection_normalized_alias_length",
        ),
    )
    op.create_index(
        "ix_life_collection_aliases_collection_id",
        "life_collection_aliases",
        ["collection_id"],
    )
    op.create_index("ix_life_collection_aliases_owner_id", "life_collection_aliases", ["owner_id"])

    op.create_table(
        "life_collection_links",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("collection_id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("inbox_item_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_link_collection_owner",
        ),
        sa.ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_life_collection_link_inbox_owner",
        ),
        sa.UniqueConstraint("collection_id", "inbox_item_id", name="uq_life_collection_link_item"),
    )
    op.create_index(
        "ix_life_collection_links_collection_id",
        "life_collection_links",
        ["collection_id"],
    )
    op.create_index("ix_life_collection_links_owner_id", "life_collection_links", ["owner_id"])
    op.create_index(
        "ix_life_collection_links_inbox_item_id",
        "life_collection_links",
        ["inbox_item_id"],
    )

    op.create_table(
        "life_collection_preferences",
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("onboarding_completed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("version > 0", name="ck_life_collection_preference_version"),
    )

    op.create_table(
        "life_collection_contexts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=False),
        sa.Column("last_inbox_item_id", sa.Integer(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_context_collection_owner",
        ),
        sa.ForeignKeyConstraint(
            ["last_inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_life_collection_context_inbox_owner",
        ),
        sa.UniqueConstraint("owner_id", "chat_id", name="uq_life_collection_context_owner_chat"),
        sa.CheckConstraint("version > 0", name="ck_life_collection_context_version"),
    )
    op.create_index(
        "ix_life_collection_contexts_owner_id", "life_collection_contexts", ["owner_id"]
    )
    op.create_index("ix_life_collection_contexts_chat_id", "life_collection_contexts", ["chat_id"])
    op.create_index(
        "ix_life_collection_contexts_collection_id",
        "life_collection_contexts",
        ["collection_id"],
    )
    op.create_index(
        "ix_life_collection_contexts_expires_at",
        "life_collection_contexts",
        ["expires_at"],
    )

    op.create_table(
        "life_collection_action_tokens",
        sa.Column("token", sa.String(32), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=True),
        sa.Column("collection_version", sa.Integer(), nullable=True),
        sa.Column("inbox_item_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(48), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_action_collection_owner",
        ),
        sa.ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_life_collection_action_inbox_owner",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_life_collection_action_status",
        ),
        sa.CheckConstraint(
            "collection_version IS NULL OR collection_version > 0",
            name="ck_life_collection_action_version",
        ),
    )
    for column in (
        "owner_id",
        "chat_id",
        "collection_id",
        "inbox_item_id",
        "action",
        "status",
        "expires_at",
    ):
        op.create_index(
            f"ix_life_collection_action_tokens_{column}",
            "life_collection_action_tokens",
            [column],
        )


def downgrade() -> None:
    op.drop_table("life_collection_action_tokens")
    op.drop_table("life_collection_contexts")
    op.drop_table("life_collection_preferences")
    op.drop_table("life_collection_links")
    op.drop_table("life_collection_aliases")
    op.drop_table("life_collections")
