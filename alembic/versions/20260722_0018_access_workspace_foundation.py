"""Add workspace access, invitations, projects, and normalized knowledge scopes."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260722_0018"
down_revision: str | None = "20260720_0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
    return (
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


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("normalized_name", sa.String(100), nullable=False),
        sa.Column("character", sa.String(20), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("access_epoch", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(),
        sa.UniqueConstraint("id", "created_by_user_id", name="uq_workspace_id_creator"),
        sa.UniqueConstraint(
            "created_by_user_id",
            "normalized_name",
            name="uq_workspace_creator_normalized_name",
        ),
        sa.CheckConstraint(
            "character IN ('pair', 'friends', 'family', 'team', 'custom')",
            name="ck_workspace_character",
        ),
        sa.CheckConstraint("status IN ('active', 'archived')", name="ck_workspace_status"),
        sa.CheckConstraint("access_epoch > 0", name="ck_workspace_access_epoch"),
        sa.CheckConstraint("version > 0", name="ck_workspace_version"),
        sa.CheckConstraint("length(name) BETWEEN 1 AND 100", name="ck_workspace_name_length"),
        sa.CheckConstraint(
            "length(normalized_name) BETWEEN 1 AND 100",
            name="ck_workspace_normalized_name_length",
        ),
        sa.CheckConstraint(
            "description IS NULL OR length(description) BETWEEN 1 AND 500",
            name="ck_workspace_description_length",
        ),
    )
    for column in ("character", "created_by_user_id", "status"):
        op.create_index(f"ix_workspaces_{column}", "workspaces", [column])

    op.create_table(
        "workspace_members",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Integer(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "invited_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member_user"),
        sa.CheckConstraint(
            "role IN ('owner', 'editor', 'viewer')", name="ck_workspace_member_role"
        ),
        sa.CheckConstraint(
            "status IN ('active', 'revoked', 'left')", name="ck_workspace_member_status"
        ),
        sa.CheckConstraint("version > 0", name="ck_workspace_member_version"),
        sa.CheckConstraint(
            "(status = 'active' AND revoked_at IS NULL) OR "
            "(status IN ('revoked', 'left') AND revoked_at IS NOT NULL)",
            name="ck_workspace_member_revocation_time",
        ),
    )
    for column in ("workspace_id", "user_id", "role", "status"):
        op.create_index(f"ix_workspace_members_{column}", "workspace_members", [column])

    op.create_table(
        "workspace_invitations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Integer(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "inviter_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "intended_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("delivery_mode", sa.String(20), nullable=False),
        sa.Column("template_key", sa.String(64), nullable=False),
        sa.Column("custom_text", sa.String(1000), nullable=True),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(),
        sa.UniqueConstraint("id", "workspace_id", name="uq_workspace_invitation_id_workspace"),
        sa.UniqueConstraint("token_hash", name="uq_workspace_invitation_token_hash"),
        sa.CheckConstraint("role IN ('editor', 'viewer')", name="ck_workspace_invitation_role"),
        sa.CheckConstraint(
            "delivery_mode IN ('direct', 'share')",
            name="ck_workspace_invitation_delivery_mode",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'accepted', 'declined', 'revoked', 'expired')",
            name="ck_workspace_invitation_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_workspace_invitation_version"),
        sa.CheckConstraint("length(token_hash) = 64", name="ck_workspace_invitation_hash_length"),
        sa.CheckConstraint(
            "length(template_key) BETWEEN 1 AND 64",
            name="ck_workspace_invitation_template_length",
        ),
        sa.CheckConstraint(
            "custom_text IS NULL OR length(custom_text) BETWEEN 1 AND 1000",
            name="ck_workspace_invitation_custom_text_length",
        ),
        sa.CheckConstraint(
            "(delivery_mode = 'direct' AND intended_user_id IS NOT NULL) OR "
            "(delivery_mode = 'share' AND intended_user_id IS NULL)",
            name="ck_workspace_invitation_recipient",
        ),
        sa.CheckConstraint(
            "intended_user_id IS NULL OR intended_user_id != inviter_user_id",
            name="ck_workspace_invitation_not_self",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND consumed_at IS NULL AND revoked_at IS NULL) OR "
            "(status IN ('accepted', 'declined') AND consumed_at IS NOT NULL "
            "AND revoked_at IS NULL) OR "
            "(status = 'revoked' AND consumed_at IS NULL AND revoked_at IS NOT NULL) OR "
            "(status = 'expired' AND consumed_at IS NULL AND revoked_at IS NULL)",
            name="ck_workspace_invitation_terminal_time",
        ),
    )
    for column in (
        "workspace_id",
        "inviter_user_id",
        "intended_user_id",
        "status",
        "expires_at",
    ):
        op.create_index(f"ix_workspace_invitations_{column}", "workspace_invitations", [column])

    op.create_table(
        "workspace_projects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Integer(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("normalized_name", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(),
        sa.UniqueConstraint("id", "workspace_id", name="uq_workspace_project_id_workspace"),
        sa.UniqueConstraint(
            "workspace_id", "normalized_name", name="uq_workspace_project_normalized_name"
        ),
        sa.CheckConstraint("status IN ('active', 'archived')", name="ck_workspace_project_status"),
        sa.CheckConstraint("version > 0", name="ck_workspace_project_version"),
        sa.CheckConstraint(
            "length(name) BETWEEN 1 AND 100", name="ck_workspace_project_name_length"
        ),
        sa.CheckConstraint(
            "length(normalized_name) BETWEEN 1 AND 100",
            name="ck_workspace_project_normalized_name_length",
        ),
    )
    for column in ("workspace_id", "status"):
        op.create_index(f"ix_workspace_projects_{column}", "workspace_projects", [column])

    op.create_table(
        "knowledge_spaces",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column(
            "personal_owner_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "workspace_id",
            sa.Integer(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("workspace_project_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["workspace_project_id", "workspace_id"],
            ["workspace_projects.id", "workspace_projects.workspace_id"],
            ondelete="CASCADE",
            name="fk_knowledge_space_project_workspace",
        ),
        sa.CheckConstraint(
            "kind IN ('personal', 'workspace', 'project')", name="ck_knowledge_space_kind"
        ),
        sa.CheckConstraint("status IN ('active', 'archived')", name="ck_knowledge_space_status"),
        sa.CheckConstraint("version > 0", name="ck_knowledge_space_version"),
        sa.CheckConstraint(
            "(kind = 'personal' AND personal_owner_user_id IS NOT NULL "
            "AND workspace_id IS NULL AND workspace_project_id IS NULL) OR "
            "(kind = 'workspace' AND personal_owner_user_id IS NULL "
            "AND workspace_id IS NOT NULL AND workspace_project_id IS NULL) OR "
            "(kind = 'project' AND personal_owner_user_id IS NULL "
            "AND workspace_id IS NOT NULL AND workspace_project_id IS NOT NULL)",
            name="ck_knowledge_space_scope",
        ),
    )
    for column in (
        "kind",
        "personal_owner_user_id",
        "workspace_id",
        "workspace_project_id",
        "status",
    ):
        op.create_index(f"ix_knowledge_spaces_{column}", "knowledge_spaces", [column])
    op.create_index(
        "uq_knowledge_space_personal_owner",
        "knowledge_spaces",
        ["personal_owner_user_id"],
        unique=True,
        sqlite_where=sa.text("kind = 'personal'"),
        postgresql_where=sa.text("kind = 'personal'"),
    )
    op.create_index(
        "uq_knowledge_space_workspace",
        "knowledge_spaces",
        ["workspace_id"],
        unique=True,
        sqlite_where=sa.text("kind = 'workspace'"),
        postgresql_where=sa.text("kind = 'workspace'"),
    )
    op.create_index(
        "uq_knowledge_space_project",
        "knowledge_spaces",
        ["workspace_project_id"],
        unique=True,
        sqlite_where=sa.text("kind = 'project'"),
        postgresql_where=sa.text("kind = 'project'"),
    )

    op.create_table(
        "workspace_contexts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("workspace_access_epoch", sa.Integer(), nullable=False),
        sa.Column("workspace_project_id", sa.Integer(), nullable=True),
        sa.Column("workspace_project_version", sa.Integer(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_members.workspace_id", "workspace_members.user_id"],
            ondelete="CASCADE",
            name="fk_workspace_context_member",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_project_id", "workspace_id"],
            ["workspace_projects.id", "workspace_projects.workspace_id"],
            ondelete="CASCADE",
            name="fk_workspace_context_project",
        ),
        sa.UniqueConstraint("actor_user_id", "chat_id", name="uq_workspace_context_actor_chat"),
        sa.CheckConstraint("workspace_access_epoch > 0", name="ck_workspace_context_access_epoch"),
        sa.CheckConstraint("version > 0", name="ck_workspace_context_version"),
        sa.CheckConstraint(
            "(workspace_project_id IS NULL AND workspace_project_version IS NULL) OR "
            "(workspace_project_id IS NOT NULL AND workspace_project_version IS NOT NULL "
            "AND workspace_project_version > 0)",
            name="ck_workspace_context_project_version",
        ),
    )
    for column in (
        "actor_user_id",
        "chat_id",
        "workspace_id",
        "workspace_project_id",
        "expires_at",
    ):
        op.create_index(f"ix_workspace_contexts_{column}", "workspace_contexts", [column])

    op.create_table(
        "workspace_action_tokens",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("scope_kind", sa.String(20), nullable=False),
        sa.Column(
            "workspace_id",
            sa.Integer(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("workspace_access_epoch", sa.Integer(), nullable=True),
        sa.Column("workspace_version", sa.Integer(), nullable=True),
        sa.Column("workspace_status_snapshot", sa.String(20), nullable=True),
        sa.Column("workspace_project_id", sa.Integer(), nullable=True),
        sa.Column("workspace_project_version", sa.Integer(), nullable=True),
        sa.Column("workspace_project_status_snapshot", sa.String(20), nullable=True),
        sa.Column("invitation_id", sa.Integer(), nullable=True),
        sa.Column("invitation_version", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(48), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace_project_id", "workspace_id"],
            ["workspace_projects.id", "workspace_projects.workspace_id"],
            ondelete="CASCADE",
            name="fk_workspace_action_project",
        ),
        sa.ForeignKeyConstraint(
            ["invitation_id", "workspace_id"],
            ["workspace_invitations.id", "workspace_invitations.workspace_id"],
            ondelete="CASCADE",
            name="fk_workspace_action_invitation",
        ),
        sa.CheckConstraint(
            "scope_kind IN ('wizard', 'workspace', 'invitation')",
            name="ck_workspace_action_scope_kind",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_workspace_action_status",
        ),
        sa.CheckConstraint(
            "(status IN ('pending', 'awaiting_input') AND consumed_at IS NULL) OR "
            "(status = 'consumed' AND consumed_at IS NOT NULL)",
            name="ck_workspace_action_consumed_time",
        ),
        sa.CheckConstraint("length(token_hash) = 64", name="ck_workspace_action_hash_length"),
        sa.CheckConstraint("length(action) BETWEEN 1 AND 48", name="ck_workspace_action_length"),
        sa.CheckConstraint(
            "(scope_kind = 'wizard' AND workspace_id IS NULL "
            "AND workspace_access_epoch IS NULL AND workspace_version IS NULL "
            "AND workspace_status_snapshot IS NULL "
            "AND workspace_project_id IS NULL AND workspace_project_version IS NULL "
            "AND workspace_project_status_snapshot IS NULL "
            "AND invitation_id IS NULL AND invitation_version IS NULL) OR "
            "(scope_kind = 'workspace' AND workspace_id IS NOT NULL "
            "AND workspace_access_epoch IS NOT NULL AND workspace_access_epoch > 0 "
            "AND workspace_version IS NOT NULL AND workspace_version > 0 "
            "AND workspace_status_snapshot IN ('active', 'archived') "
            "AND invitation_id IS NULL AND invitation_version IS NULL) OR "
            "(scope_kind = 'invitation' AND workspace_id IS NOT NULL "
            "AND workspace_access_epoch IS NULL AND workspace_version IS NULL "
            "AND workspace_status_snapshot IS NULL "
            "AND workspace_project_id IS NULL AND workspace_project_version IS NULL "
            "AND workspace_project_status_snapshot IS NULL "
            "AND invitation_id IS NOT NULL AND invitation_version IS NOT NULL "
            "AND invitation_version > 0)",
            name="ck_workspace_action_scope",
        ),
        sa.CheckConstraint(
            "(workspace_project_id IS NULL AND workspace_project_version IS NULL "
            "AND workspace_project_status_snapshot IS NULL) OR "
            "(scope_kind = 'workspace' AND workspace_project_id IS NOT NULL "
            "AND workspace_project_version IS NOT NULL AND workspace_project_version > 0 "
            "AND workspace_project_status_snapshot IN ('active', 'archived'))",
            name="ck_workspace_action_project_version",
        ),
    )
    for column in (
        "actor_user_id",
        "chat_id",
        "workspace_id",
        "workspace_project_id",
        "invitation_id",
        "action",
        "status",
        "expires_at",
    ):
        op.create_index(f"ix_workspace_action_tokens_{column}", "workspace_action_tokens", [column])


def downgrade() -> None:
    op.drop_table("workspace_action_tokens")
    op.drop_table("workspace_contexts")
    op.drop_table("knowledge_spaces")
    op.drop_table("workspace_projects")
    op.drop_table("workspace_invitations")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
