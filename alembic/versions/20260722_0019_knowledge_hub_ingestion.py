"""Add Knowledge Hub sources, immutable revisions, capture, quotas, and jobs."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260722_0019"
down_revision: str | None = "20260722_0018"
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
    # Keep this nullable for additive rollback compatibility with the PR #23
    # image. New ORM writes always generate a UUID and KnowledgeService repairs
    # a legacy NULL before returning a space to Knowledge code.
    op.add_column(
        "knowledge_spaces",
        sa.Column(
            "public_id",
            sa.String(36),
            sa.CheckConstraint(
                "public_id IS NULL OR length(public_id) = 36",
                name="ck_knowledge_space_public_id_length",
            ),
            nullable=True,
        ),
    )
    op.create_index(
        "uq_knowledge_space_public_id",
        "knowledge_spaces",
        ["public_id"],
        unique=True,
    )
    op.create_index(
        "uq_knowledge_space_id_kind",
        "knowledge_spaces",
        ["id", "kind"],
        unique=True,
    )

    op.create_table(
        "knowledge_runtime_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("maintenance_paused", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("id = 1", name="ck_knowledge_runtime_singleton"),
        sa.CheckConstraint("version > 0", name="ck_knowledge_runtime_version"),
    )
    op.bulk_insert(
        sa.table(
            "knowledge_runtime_state",
            sa.column("id", sa.Integer()),
            sa.column("maintenance_paused", sa.Boolean()),
            sa.column("version", sa.Integer()),
        ),
        [{"id": 1, "maintenance_paused": False, "version": 1}],
    )

    op.create_table(
        "knowledge_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column("knowledge_space_id", sa.Integer(), nullable=False),
        sa.Column("space_kind", sa.String(20), nullable=False),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("provenance_kind", sa.String(40), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=True),
        sa.Column("processing_status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("lifecycle_status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("knowledge_role", sa.String(20), nullable=False, server_default="trusted"),
        sa.Column("priority", sa.String(20), nullable=False, server_default="normal"),
        sa.Column("publication_state", sa.String(24), nullable=False, server_default="draft"),
        sa.Column("system_classification", sa.String(24), nullable=False, server_default="general"),
        sa.Column("user_classification", sa.String(64), nullable=True),
        sa.Column("current_revision_number", sa.Integer(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("trashed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "trashed_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("purge_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["knowledge_space_id", "space_kind"],
            ["knowledge_spaces.id", "knowledge_spaces.kind"],
            ondelete="CASCADE",
            name="fk_knowledge_source_space_scope",
        ),
        sa.UniqueConstraint("public_id", name="uq_knowledge_source_public_id"),
        sa.UniqueConstraint("id", "knowledge_space_id", name="uq_knowledge_source_id_space"),
        sa.CheckConstraint("length(public_id) = 36", name="ck_knowledge_source_public_id_length"),
        sa.CheckConstraint(
            "source_type IN ('text', 'forward', 'document', 'image', 'url')",
            name="ck_knowledge_source_type",
        ),
        sa.CheckConstraint(
            "processing_status IN "
            "('queued', 'processing', 'ready', 'partial', 'failed', 'quarantined', "
            "'cancelled')",
            name="ck_knowledge_source_processing_status",
        ),
        sa.CheckConstraint(
            "lifecycle_status IN ('active', 'trashed', 'purge_pending', 'purge_failed', 'purged')",
            name="ck_knowledge_source_lifecycle_status",
        ),
        sa.CheckConstraint(
            "knowledge_role IN "
            "('foundation', 'trusted', 'perspective', 'discussion', 'counterpoint', "
            "'hypothesis')",
            name="ck_knowledge_source_role",
        ),
        sa.CheckConstraint(
            "priority IN ('high', 'normal', 'low')", name="ck_knowledge_source_priority"
        ),
        sa.CheckConstraint(
            "publication_state IN ('draft', 'publication_ready')",
            name="ck_knowledge_source_publication_state",
        ),
        sa.CheckConstraint(
            "system_classification IN ('general', 'health_private')",
            name="ck_knowledge_source_system_classification",
        ),
        sa.CheckConstraint(
            "system_classification != 'health_private' OR space_kind = 'personal'",
            name="ck_knowledge_source_health_personal_only",
        ),
        sa.CheckConstraint(
            "system_classification != 'health_private' OR publication_state = 'draft'",
            name="ck_knowledge_source_health_not_publication_ready",
        ),
        sa.CheckConstraint("length(title) BETWEEN 1 AND 200", name="ck_knowledge_source_title"),
        sa.CheckConstraint(
            "length(provenance_kind) BETWEEN 1 AND 40",
            name="ck_knowledge_source_provenance_kind",
        ),
        sa.CheckConstraint(
            "user_classification IS NULL OR length(user_classification) BETWEEN 1 AND 64",
            name="ck_knowledge_source_user_classification",
        ),
        sa.CheckConstraint("version > 0", name="ck_knowledge_source_version"),
        sa.CheckConstraint(
            "(lifecycle_status = 'purged' AND current_revision_number IS NULL) OR "
            "(lifecycle_status != 'purged' AND current_revision_number > 0)",
            name="ck_knowledge_source_current_revision",
        ),
        sa.CheckConstraint(
            "(lifecycle_status = 'active' AND trashed_at IS NULL "
            "AND purge_requested_at IS NULL AND purged_at IS NULL) OR "
            "(lifecycle_status = 'trashed' AND trashed_at IS NOT NULL "
            "AND purge_requested_at IS NULL AND purged_at IS NULL) OR "
            "(lifecycle_status IN ('purge_pending', 'purge_failed') "
            "AND trashed_at IS NOT NULL AND purge_requested_at IS NOT NULL "
            "AND purged_at IS NULL) OR "
            "(lifecycle_status = 'purged' AND trashed_at IS NOT NULL "
            "AND purge_requested_at IS NOT NULL AND purged_at IS NOT NULL)",
            name="ck_knowledge_source_lifecycle_times",
        ),
    )
    for column in (
        "knowledge_space_id",
        "space_kind",
        "created_by_user_id",
        "source_type",
        "processing_status",
        "lifecycle_status",
        "knowledge_role",
        "priority",
        "publication_state",
        "system_classification",
        "user_classification",
        "trashed_at",
    ):
        op.create_index(f"ix_knowledge_sources_{column}", "knowledge_sources", [column])
    op.create_index(
        "ix_knowledge_sources_space_lifecycle_updated",
        "knowledge_sources",
        ["knowledge_space_id", "lifecycle_status", "updated_at"],
    )
    op.create_index(
        "ix_knowledge_sources_space_processing",
        "knowledge_sources",
        ["knowledge_space_id", "processing_status"],
    )

    op.create_table(
        "knowledge_source_revisions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("knowledge_space_id", sa.Integer(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("original_revision_id", sa.Integer(), nullable=True),
        sa.Column("original_storage_key", sa.String(512), nullable=True),
        sa.Column("declared_mime", sa.String(127), nullable=True),
        sa.Column("detected_mime", sa.String(127), nullable=False),
        sa.Column("detected_format", sa.String(20), nullable=False),
        sa.Column("safe_display_name", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("extracted_storage_key", sa.String(512), nullable=True),
        sa.Column("extracted_sha256", sa.String(64), nullable=True),
        sa.Column("extracted_size_bytes", sa.Integer(), nullable=True),
        sa.Column("extraction_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("extraction_metadata", sa.JSON(), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("finalized_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_revision_source_space",
        ),
        sa.ForeignKeyConstraint(
            ["original_revision_id", "source_id", "knowledge_space_id"],
            [
                "knowledge_source_revisions.id",
                "knowledge_source_revisions.source_id",
                "knowledge_source_revisions.knowledge_space_id",
            ],
            ondelete="CASCADE",
            name="fk_knowledge_revision_original_scope",
        ),
        sa.UniqueConstraint("public_id", name="uq_knowledge_revision_public_id"),
        sa.UniqueConstraint("id", "source_id", name="uq_knowledge_revision_id_source"),
        sa.UniqueConstraint(
            "id",
            "source_id",
            "knowledge_space_id",
            name="uq_knowledge_revision_id_source_space",
        ),
        sa.UniqueConstraint(
            "source_id", "revision_number", name="uq_knowledge_revision_source_number"
        ),
        sa.UniqueConstraint("original_storage_key", name="uq_knowledge_revision_original_key"),
        sa.UniqueConstraint("extracted_storage_key", name="uq_knowledge_revision_extracted_key"),
        sa.CheckConstraint("length(public_id) = 36", name="ck_knowledge_revision_public_id_length"),
        sa.CheckConstraint("revision_number > 0", name="ck_knowledge_revision_number"),
        sa.CheckConstraint("length(sha256) = 64", name="ck_knowledge_revision_sha256_length"),
        sa.CheckConstraint("size_bytes >= 0", name="ck_knowledge_revision_size"),
        sa.CheckConstraint(
            "detected_format IN ('text', 'txt', 'markdown', 'pdf', 'docx', 'epub', 'image', 'url')",
            name="ck_knowledge_revision_format",
        ),
        sa.CheckConstraint(
            "extraction_status IN "
            "('pending', 'ready', 'partial', 'failed', 'quarantined', 'cancelled')",
            name="ck_knowledge_revision_extraction_status",
        ),
        sa.CheckConstraint(
            "original_storage_key IS NULL OR length(original_storage_key) BETWEEN 1 AND 512",
            name="ck_knowledge_revision_original_key_length",
        ),
        sa.CheckConstraint(
            "(original_revision_id IS NULL AND original_storage_key IS NOT NULL) OR "
            "(original_revision_id IS NOT NULL AND original_storage_key IS NULL)",
            name="ck_knowledge_revision_original_reference",
        ),
        sa.CheckConstraint(
            "length(safe_display_name) BETWEEN 1 AND 255",
            name="ck_knowledge_revision_display_name",
        ),
        sa.CheckConstraint(
            "(extracted_storage_key IS NULL AND extracted_sha256 IS NULL "
            "AND extracted_size_bytes IS NULL) OR "
            "(extracted_storage_key IS NOT NULL AND extracted_sha256 IS NOT NULL "
            "AND length(extracted_sha256) = 64 AND extracted_size_bytes >= 0)",
            name="ck_knowledge_revision_extracted_tuple",
        ),
        sa.CheckConstraint(
            "(extraction_status = 'pending' AND finalized_at IS NULL) OR "
            "(extraction_status != 'pending' AND finalized_at IS NOT NULL)",
            name="ck_knowledge_revision_finalized_time",
        ),
    )
    for column in (
        "source_id",
        "knowledge_space_id",
        "original_revision_id",
        "detected_format",
        "extraction_status",
        "created_by_user_id",
    ):
        op.create_index(
            f"ix_knowledge_source_revisions_{column}",
            "knowledge_source_revisions",
            [column],
        )
    op.create_index(
        "ix_knowledge_revisions_space_sha256",
        "knowledge_source_revisions",
        ["knowledge_space_id", "sha256"],
    )

    op.create_table(
        "knowledge_ingestion_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column("knowledge_space_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("revision_id", sa.Integer(), nullable=True),
        sa.Column(
            "requested_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("job_type", sa.String(20), nullable=False, server_default="extract"),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_owner", sa.String(64), nullable=True),
        sa.Column("lease_token", sa.String(64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("safe_error_code", sa.String(64), nullable=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("pipeline_version", sa.String(32), nullable=False, server_default="v1"),
        sa.Column("source_version", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_job_source_space",
        ),
        sa.ForeignKeyConstraint(
            ["revision_id", "source_id"],
            ["knowledge_source_revisions.id", "knowledge_source_revisions.source_id"],
            ondelete="CASCADE",
            name="fk_knowledge_job_revision_source",
        ),
        sa.UniqueConstraint("public_id", name="uq_knowledge_job_public_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_knowledge_job_idempotency"),
        sa.CheckConstraint("length(public_id) = 36", name="ck_knowledge_job_public_id_length"),
        sa.CheckConstraint("job_type IN ('extract', 'purge')", name="ck_knowledge_job_type"),
        sa.CheckConstraint(
            "(job_type = 'extract' AND revision_id IS NOT NULL) OR "
            "(job_type = 'purge' AND revision_id IS NULL)",
            name="ck_knowledge_job_revision_scope",
        ),
        sa.CheckConstraint(
            "status IN "
            "('queued', 'processing', 'ready', 'partial', 'failed', 'quarantined', "
            "'cancelled')",
            name="ck_knowledge_job_status",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0 AND max_attempts BETWEEN 1 AND 20 "
            "AND attempt_count <= max_attempts",
            name="ck_knowledge_job_attempts",
        ),
        sa.CheckConstraint("source_version > 0", name="ck_knowledge_job_source_version"),
        sa.CheckConstraint("version > 0", name="ck_knowledge_job_version"),
        sa.CheckConstraint(
            "(status = 'processing' AND lease_owner IS NOT NULL AND lease_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL "
            "AND finished_at IS NULL) OR "
            "(status = 'queued' AND lease_owner IS NULL AND lease_token IS NULL "
            "AND lease_expires_at IS NULL AND heartbeat_at IS NULL AND finished_at IS NULL) OR "
            "(status IN ('ready', 'partial', 'failed', 'quarantined', 'cancelled') "
            "AND lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL "
            "AND heartbeat_at IS NULL AND finished_at IS NOT NULL)",
            name="ck_knowledge_job_lease_state",
        ),
        sa.CheckConstraint(
            "status NOT IN ('ready', 'cancelled') OR safe_error_code IS NULL",
            name="ck_knowledge_job_safe_error",
        ),
    )
    for column in (
        "knowledge_space_id",
        "source_id",
        "revision_id",
        "requested_by_user_id",
        "job_type",
        "status",
        "available_at",
        "lease_expires_at",
    ):
        op.create_index(
            f"ix_knowledge_ingestion_jobs_{column}",
            "knowledge_ingestion_jobs",
            [column],
        )
    op.create_index(
        "ix_knowledge_jobs_poll",
        "knowledge_ingestion_jobs",
        ["status", "available_at", "id"],
    )
    op.create_index(
        "ix_knowledge_jobs_stale_lease",
        "knowledge_ingestion_jobs",
        ["status", "lease_expires_at"],
    )
    op.create_index(
        "ix_knowledge_jobs_source_status",
        "knowledge_ingestion_jobs",
        ["source_id", "status"],
    )

    op.create_table(
        "knowledge_capture_drafts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("capture_kind", sa.String(20), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("telegram_file_id", sa.String(512), nullable=True),
        sa.Column("telegram_file_unique_id_hash", sa.String(64), nullable=True),
        sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
        sa.Column("declared_mime", sa.String(127), nullable=True),
        sa.Column("safe_display_name", sa.String(255), nullable=True),
        sa.Column("declared_size_bytes", sa.Integer(), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        sa.Column(
            "knowledge_space_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_spaces.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("knowledge_space_version", sa.Integer(), nullable=True),
        sa.Column("workspace_access_epoch", sa.Integer(), nullable=True),
        sa.Column("workspace_project_version", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("knowledge_role", sa.String(20), nullable=False, server_default="trusted"),
        sa.Column("priority", sa.String(20), nullable=False, server_default="normal"),
        sa.Column("system_classification", sa.String(24), nullable=False, server_default="general"),
        sa.Column("user_classification", sa.String(64), nullable=True),
        sa.Column("status", sa.String(24), nullable=False, server_default="collecting"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "confirmed_source_id",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint("public_id", name="uq_knowledge_capture_public_id"),
        sa.UniqueConstraint("id", "knowledge_space_id", name="uq_knowledge_capture_id_space"),
        sa.ForeignKeyConstraint(
            ["confirmed_source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="RESTRICT",
            name="fk_knowledge_capture_confirmed_source_space",
        ),
        sa.CheckConstraint("length(public_id) = 36", name="ck_knowledge_capture_public_id_length"),
        sa.CheckConstraint(
            "capture_kind IN ('text', 'forward', 'document', 'image', 'url')",
            name="ck_knowledge_capture_kind",
        ),
        sa.CheckConstraint(
            "status IN "
            "('collecting', 'awaiting_confirmation', 'confirming', 'confirmed', "
            "'cancelled', 'expired')",
            name="ck_knowledge_capture_status",
        ),
        sa.CheckConstraint(
            "knowledge_role IN "
            "('foundation', 'trusted', 'perspective', 'discussion', 'counterpoint', "
            "'hypothesis')",
            name="ck_knowledge_capture_role",
        ),
        sa.CheckConstraint(
            "priority IN ('high', 'normal', 'low')", name="ck_knowledge_capture_priority"
        ),
        sa.CheckConstraint(
            "system_classification IN ('general', 'health_private')",
            name="ck_knowledge_capture_system_classification",
        ),
        sa.CheckConstraint("version > 0", name="ck_knowledge_capture_version"),
        sa.CheckConstraint(
            "declared_size_bytes IS NULL OR declared_size_bytes >= 0",
            name="ck_knowledge_capture_declared_size",
        ),
        sa.CheckConstraint(
            "(status IN ('collecting', 'confirmed', 'cancelled', 'expired') "
            "AND text_content IS NULL AND source_url IS NULL "
            "AND telegram_file_id IS NULL) OR "
            "(status IN ('awaiting_confirmation', 'confirming') "
            "AND capture_kind IN ('text', 'forward') AND text_content IS NOT NULL "
            "AND source_url IS NULL AND telegram_file_id IS NULL) OR "
            "(status IN ('awaiting_confirmation', 'confirming') "
            "AND capture_kind = 'url' AND source_url IS NOT NULL AND text_content IS NULL "
            "AND telegram_file_id IS NULL) OR "
            "(status IN ('awaiting_confirmation', 'confirming') "
            "AND capture_kind IN ('document', 'image') AND telegram_file_id IS NOT NULL "
            "AND text_content IS NULL AND source_url IS NULL)",
            name="ck_knowledge_capture_payload",
        ),
        sa.CheckConstraint(
            "(status IN ('collecting', 'cancelled', 'expired')) OR "
            "(knowledge_space_id IS NOT NULL AND knowledge_space_version IS NOT NULL "
            "AND title IS NOT NULL)",
            name="ck_knowledge_capture_configured",
        ),
        sa.CheckConstraint(
            "(status = 'confirmed' AND confirmed_source_id IS NOT NULL "
            "AND completed_at IS NOT NULL) OR "
            "(status IN ('cancelled', 'expired') AND confirmed_source_id IS NULL "
            "AND completed_at IS NOT NULL) OR "
            "(status IN ('collecting', 'awaiting_confirmation', 'confirming') "
            "AND confirmed_source_id IS NULL AND completed_at IS NULL)",
            name="ck_knowledge_capture_completion",
        ),
    )
    for column in (
        "actor_user_id",
        "chat_id",
        "knowledge_space_id",
        "status",
        "expires_at",
        "confirmed_source_id",
    ):
        op.create_index(
            f"ix_knowledge_capture_drafts_{column}",
            "knowledge_capture_drafts",
            [column],
        )
    op.create_index(
        "uq_knowledge_capture_active_actor_chat",
        "knowledge_capture_drafts",
        ["actor_user_id", "chat_id"],
        unique=True,
        sqlite_where=sa.text("status IN ('collecting', 'awaiting_confirmation', 'confirming')"),
        postgresql_where=sa.text("status IN ('collecting', 'awaiting_confirmation', 'confirming')"),
    )

    op.create_table(
        "knowledge_action_tokens",
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
            "knowledge_space_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("knowledge_space_version", sa.Integer(), nullable=False),
        sa.Column("workspace_access_epoch", sa.Integer(), nullable=True),
        sa.Column(
            "capture_draft_id",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column("capture_version", sa.Integer(), nullable=True),
        sa.Column(
            "source_id",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column("source_version", sa.Integer(), nullable=True),
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
            ["capture_draft_id", "knowledge_space_id"],
            ["knowledge_capture_drafts.id", "knowledge_capture_drafts.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_action_capture_space",
        ),
        sa.ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_action_source_space",
        ),
        sa.CheckConstraint("length(token_hash) = 64", name="ck_knowledge_action_hash"),
        sa.CheckConstraint(
            "scope_kind IN ('capture', 'source', 'space')",
            name="ck_knowledge_action_scope_kind",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_knowledge_action_status",
        ),
        sa.CheckConstraint(
            "(status IN ('pending', 'awaiting_input') AND consumed_at IS NULL) OR "
            "(status = 'consumed' AND consumed_at IS NOT NULL)",
            name="ck_knowledge_action_consumed",
        ),
        sa.CheckConstraint(
            "(scope_kind = 'capture' AND capture_draft_id IS NOT NULL "
            "AND capture_version IS NOT NULL AND source_id IS NULL) OR "
            "(scope_kind = 'source' AND capture_draft_id IS NULL "
            "AND capture_version IS NULL AND source_id IS NOT NULL "
            "AND source_version IS NOT NULL) OR "
            "(scope_kind = 'space' AND capture_draft_id IS NULL "
            "AND capture_version IS NULL AND source_id IS NULL "
            "AND source_version IS NULL)",
            name="ck_knowledge_action_scope",
        ),
    )
    for column in (
        "actor_user_id",
        "chat_id",
        "knowledge_space_id",
        "capture_draft_id",
        "source_id",
        "action",
        "status",
        "expires_at",
    ):
        op.create_index(
            f"ix_knowledge_action_tokens_{column}",
            "knowledge_action_tokens",
            [column],
        )

    op.create_table(
        "knowledge_quota_reservations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "knowledge_space_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "capture_draft_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column("reserved_bytes", sa.Integer(), nullable=False),
        sa.Column("reserved_sources", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reserved_jobs", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(20), nullable=False, server_default="reserved"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "source_id",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "revision_id",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint("public_id", name="uq_knowledge_quota_reservation_public_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_knowledge_quota_reservation_key"),
        sa.ForeignKeyConstraint(
            ["capture_draft_id", "knowledge_space_id"],
            ["knowledge_capture_drafts.id", "knowledge_capture_drafts.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_quota_capture_space",
        ),
        sa.ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_quota_source_space",
        ),
        sa.ForeignKeyConstraint(
            ["revision_id", "source_id", "knowledge_space_id"],
            [
                "knowledge_source_revisions.id",
                "knowledge_source_revisions.source_id",
                "knowledge_source_revisions.knowledge_space_id",
            ],
            ondelete="CASCADE",
            name="fk_knowledge_quota_revision_source_space",
        ),
        sa.CheckConstraint(
            "length(public_id) = 36", name="ck_knowledge_quota_reservation_public_id"
        ),
        sa.CheckConstraint(
            "status IN ('reserved', 'committed', 'released', 'expired')",
            name="ck_knowledge_quota_reservation_status",
        ),
        sa.CheckConstraint(
            "reserved_bytes >= 0 AND reserved_sources > 0 AND reserved_jobs >= 0",
            name="ck_knowledge_quota_reservation_amounts",
        ),
        sa.CheckConstraint(
            "(status = 'reserved' AND completed_at IS NULL AND source_id IS NULL "
            "AND revision_id IS NULL) OR "
            "(status = 'committed' AND completed_at IS NOT NULL AND source_id IS NOT NULL "
            "AND revision_id IS NOT NULL) OR "
            "(status IN ('released', 'expired') AND completed_at IS NOT NULL "
            "AND source_id IS NULL AND revision_id IS NULL)",
            name="ck_knowledge_quota_reservation_completion",
        ),
    )
    for column in (
        "actor_user_id",
        "knowledge_space_id",
        "capture_draft_id",
        "status",
        "expires_at",
    ):
        op.create_index(
            f"ix_knowledge_quota_reservations_{column}",
            "knowledge_quota_reservations",
            [column],
        )
    op.create_index(
        "ix_knowledge_quota_actor_status",
        "knowledge_quota_reservations",
        ["actor_user_id", "status"],
    )
    op.create_index(
        "ix_knowledge_quota_space_status",
        "knowledge_quota_reservations",
        ["knowledge_space_id", "status"],
    )

    op.create_table(
        "knowledge_audit_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column("event_type", sa.String(48), nullable=False),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "workspace_id",
            sa.Integer(),
            sa.ForeignKey("workspaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "knowledge_space_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_spaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "capture_draft_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_capture_drafts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_sources.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "revision_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_source_revisions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_ingestion_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("safe_metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("public_id", name="uq_knowledge_audit_public_id"),
        sa.CheckConstraint("length(public_id) = 36", name="ck_knowledge_audit_public_id"),
        sa.CheckConstraint(
            "event_type IN ("
            "'workspace.created', 'workspace.member_added', 'workspace.role_changed', "
            "'workspace.member_revoked', 'workspace.member_left', 'workspace.archived', "
            "'workspace.restored', 'workspace.project_created', "
            "'workspace.project_renamed', 'workspace.project_archived', "
            "'workspace.project_restored', 'space.created', 'capture.started', "
            "'capture.confirmed', "
            "'capture.cancelled', 'capture.expired', 'source.created', "
            "'revision.created', 'ingestion.status_changed', 'source.trashed', "
            "'source.restored', 'source.purge_requested', 'source.purged', "
            "'source.purge_failed', 'source.classification_changed')",
            name="ck_knowledge_audit_event_type",
        ),
    )
    for column in (
        "event_type",
        "actor_user_id",
        "workspace_id",
        "knowledge_space_id",
        "capture_draft_id",
        "source_id",
        "revision_id",
        "job_id",
    ):
        op.create_index(
            f"ix_knowledge_audit_events_{column}",
            "knowledge_audit_events",
            [column],
        )
    op.create_index(
        "ix_knowledge_audit_space_created",
        "knowledge_audit_events",
        ["knowledge_space_id", "created_at"],
    )
    op.create_index(
        "ix_knowledge_audit_source_created",
        "knowledge_audit_events",
        ["source_id", "created_at"],
    )


def downgrade() -> None:
    # Database-only downgrade for pre-activation test/rollback. It deliberately
    # never removes originals or extracted files under /data/knowledge.
    op.drop_table("knowledge_audit_events")
    op.drop_table("knowledge_quota_reservations")
    op.drop_table("knowledge_action_tokens")
    op.drop_table("knowledge_capture_drafts")
    op.drop_table("knowledge_ingestion_jobs")
    op.drop_table("knowledge_source_revisions")
    op.drop_table("knowledge_sources")
    op.drop_table("knowledge_runtime_state")
    op.drop_index("uq_knowledge_space_id_kind", table_name="knowledge_spaces")
    op.drop_index("uq_knowledge_space_public_id", table_name="knowledge_spaces")
    op.drop_column("knowledge_spaces", "public_id")
