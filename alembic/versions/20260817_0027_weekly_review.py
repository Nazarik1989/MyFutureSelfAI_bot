"""Add durable weekly review sessions and confirmed weekly focuses."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260817_0027"
down_revision: str | None = "20260811_0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_string_items_check(column: str, *, maximum_items: int) -> str:
    return " AND ".join(
        f"(json_array_length({column}) < {index + 1} OR "
        f"(coalesce(substr(CAST({column} -> {index} AS TEXT), 1, 1) = '\"', false) "
        f"AND coalesce(length({column} ->> {index}), 0) BETWEEN 1 AND 200))"
        for index in range(maximum_items)
    )


def _reminder_candidate_fields_check(column: str, *, dialect_name: str) -> str:
    checks: list[str] = []
    for index in range(5):
        if dialect_name == "postgresql":
            candidate = f"({column} -> {index})::jsonb"
            checks.append(
                f"(json_array_length({column}) < {index + 1} OR ("
                f"coalesce(jsonb_typeof({candidate}) = 'object', false) "
                f"AND coalesce(jsonb_typeof({candidate} -> 'title') = 'string', false) "
                f"AND coalesce(length({candidate} ->> 'title'), 0) BETWEEN 1 AND 200 "
                f"AND coalesce(jsonb_typeof({candidate} -> 'schedule_wording') = "
                f"'string', false) "
                f"AND coalesce(length({candidate} ->> 'schedule_wording'), 0) "
                f"BETWEEN 1 AND 200 "
                f"AND ({candidate} - 'title' - 'schedule_wording') = '{{}}'::jsonb))"
            )
            continue
        if dialect_name != "sqlite":
            raise RuntimeError("Weekly review migration supports SQLite and PostgreSQL.")
        candidate_path = f"'$[{index}]'"
        title_path = f"'$[{index}].title'"
        schedule_path = f"'$[{index}].schedule_wording'"
        checks.append(
            f"(json_array_length({column}) < {index + 1} OR ("
            f"coalesce(json_type({column}, {candidate_path}) = 'object', false) "
            f"AND coalesce(json_type({column}, {title_path}) = 'text', false) "
            f"AND coalesce(length(json_extract({column}, {title_path})), 0) "
            f"BETWEEN 1 AND 200 "
            f"AND coalesce(json_type({column}, {schedule_path}) = 'text', false) "
            f"AND coalesce(length(json_extract({column}, {schedule_path})), 0) "
            f"BETWEEN 1 AND 200 "
            f"AND json_remove(json_extract({column}, {candidate_path}), "
            f"'$.title', '$.schedule_wording') = '{{}}'))"
        )
    return " AND ".join(checks)


def upgrade() -> None:
    dialect_name = op.get_bind().dialect.name
    op.create_table(
        "weekly_focuses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("week_start", sa.Date(), nullable=False),
        sa.Column("focus", sa.Text(), nullable=False),
        sa.Column("approach", sa.Text(), nullable=True),
        sa.Column(
            "small_steps",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("source", sa.String(8), nullable=False),
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
        sa.UniqueConstraint("public_id", name="uq_weekly_focuses_public_id"),
        sa.UniqueConstraint(
            "owner_id",
            "week_start",
            name="uq_weekly_focuses_owner_week",
        ),
        sa.CheckConstraint(
            "length(public_id) = 36",
            name="ck_weekly_focuses_public_id_length",
        ),
        sa.CheckConstraint(
            "length(focus) BETWEEN 1 AND 300",
            name="ck_weekly_focuses_focus_length",
        ),
        sa.CheckConstraint(
            "approach IS NULL OR length(approach) BETWEEN 1 AND 500",
            name="ck_weekly_focuses_approach_length",
        ),
        sa.CheckConstraint(
            "json_array_length(small_steps) BETWEEN 0 AND 3",
            name="ck_weekly_focuses_small_steps_count",
        ),
        sa.CheckConstraint(
            "substr(CAST(small_steps AS TEXT), 1, 1) = '[' "
            "AND length(CAST(small_steps AS TEXT)) BETWEEN 2 AND 4096",
            name="ck_weekly_focuses_small_steps_json_shape",
        ),
        sa.CheckConstraint(
            _json_string_items_check("small_steps", maximum_items=3),
            name="ck_weekly_focuses_small_steps_lengths",
        ),
        sa.CheckConstraint(
            "source IN ('text', 'voice')",
            name="ck_weekly_focuses_source",
        ),
        sa.CheckConstraint("version > 0", name="ck_weekly_focuses_version"),
    )
    op.create_index(
        "ix_weekly_focuses_owner_history",
        "weekly_focuses",
        ["owner_id", "week_start", "id"],
    )

    op.create_table(
        "weekly_focus_changes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("focus_public_id", sa.String(36), nullable=False),
        sa.Column("operation", sa.String(12), nullable=False),
        sa.Column("resulting_version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "operation IN ('created', 'updated', 'deleted')",
            name="ck_weekly_focus_changes_operation",
        ),
        sa.CheckConstraint(
            "length(focus_public_id) = 36",
            name="ck_weekly_focus_changes_public_id_length",
        ),
        sa.CheckConstraint(
            "resulting_version > 0",
            name="ck_weekly_focus_changes_resulting_version",
        ),
    )
    op.create_index(
        "ix_weekly_focus_changes_owner_created",
        "weekly_focus_changes",
        ["owner_id", "created_at", "id"],
    )
    op.create_index(
        "ix_weekly_focus_changes_focus_history",
        "weekly_focus_changes",
        ["owner_id", "focus_public_id", "created_at", "id"],
    )

    op.create_table(
        "weekly_review_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("access_version", sa.Integer(), nullable=False),
        sa.Column("week_start", sa.Date(), nullable=False),
        sa.Column("phase", sa.String(24), nullable=False),
        sa.Column("canonical_chat_id", sa.BigInteger(), nullable=True),
        sa.Column("canonical_message_id", sa.BigInteger(), nullable=True),
        sa.Column("base_focus_public_id", sa.String(36), nullable=True),
        sa.Column("base_focus_version", sa.Integer(), nullable=True),
        sa.Column("extracted_focus", sa.Text(), nullable=True),
        sa.Column("extracted_approach", sa.Text(), nullable=True),
        sa.Column(
            "small_steps",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "reminder_candidates",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("extracted_source", sa.String(8), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.UniqueConstraint("public_id", name="uq_weekly_review_sessions_public_id"),
        sa.UniqueConstraint(
            "owner_id",
            "chat_id",
            name="uq_weekly_review_sessions_owner_chat",
        ),
        sa.CheckConstraint(
            "length(public_id) = 36",
            name="ck_weekly_review_sessions_public_id_length",
        ),
        sa.CheckConstraint(
            "phase IN ('root', 'awaiting_input', 'processing', 'preview', 'saved', "
            "'candidates', 'reminder_handoff', 'delete_preview', 'completed')",
            name="ck_weekly_review_sessions_phase",
        ),
        sa.CheckConstraint(
            "access_version > 0",
            name="ck_weekly_review_sessions_access_version",
        ),
        sa.CheckConstraint("version > 0", name="ck_weekly_review_sessions_version"),
        sa.CheckConstraint(
            "extracted_focus IS NULL OR length(extracted_focus) BETWEEN 1 AND 300",
            name="ck_weekly_review_sessions_focus_length",
        ),
        sa.CheckConstraint(
            "extracted_approach IS NULL OR length(extracted_approach) BETWEEN 1 AND 500",
            name="ck_weekly_review_sessions_approach_length",
        ),
        sa.CheckConstraint(
            "json_array_length(small_steps) BETWEEN 0 AND 3",
            name="ck_weekly_review_sessions_small_steps_count",
        ),
        sa.CheckConstraint(
            "substr(CAST(small_steps AS TEXT), 1, 1) = '[' "
            "AND length(CAST(small_steps AS TEXT)) BETWEEN 2 AND 4096",
            name="ck_weekly_review_sessions_small_steps_json_shape",
        ),
        sa.CheckConstraint(
            _json_string_items_check("small_steps", maximum_items=3),
            name="ck_weekly_review_sessions_small_steps_lengths",
        ),
        sa.CheckConstraint(
            "json_array_length(reminder_candidates) BETWEEN 0 AND 5",
            name="ck_weekly_review_sessions_candidates_count",
        ),
        sa.CheckConstraint(
            "substr(CAST(reminder_candidates AS TEXT), 1, 1) = '[' "
            "AND length(CAST(reminder_candidates AS TEXT)) BETWEEN 2 AND 20000",
            name="ck_weekly_review_sessions_candidates_json_shape",
        ),
        sa.CheckConstraint(
            _reminder_candidate_fields_check(
                "reminder_candidates",
                dialect_name=dialect_name,
            ),
            name="ck_weekly_review_sessions_candidates_fields",
        ),
        sa.CheckConstraint(
            "extracted_source IS NULL OR extracted_source IN ('text', 'voice')",
            name="ck_weekly_review_sessions_source",
        ),
        sa.CheckConstraint(
            "(canonical_chat_id IS NULL AND canonical_message_id IS NULL) OR "
            "(canonical_chat_id IS NOT NULL AND canonical_message_id IS NOT NULL "
            "AND canonical_chat_id = chat_id AND canonical_message_id > 0)",
            name="ck_weekly_review_sessions_canonical_binding",
        ),
        sa.CheckConstraint(
            "(base_focus_public_id IS NULL AND base_focus_version IS NULL) OR "
            "(base_focus_public_id IS NOT NULL AND base_focus_version IS NOT NULL "
            "AND length(base_focus_public_id) = 36 AND base_focus_version > 0)",
            name="ck_weekly_review_sessions_base_focus_generation",
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_weekly_review_sessions_expiry_order",
        ),
    )
    op.create_index(
        "ix_weekly_review_sessions_expiry",
        "weekly_review_sessions",
        ["expires_at", "id"],
    )
    op.create_index(
        "ix_weekly_review_sessions_owner_week",
        "weekly_review_sessions",
        ["owner_id", "week_start", "id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_weekly_review_sessions_owner_week",
        table_name="weekly_review_sessions",
    )
    op.drop_index(
        "ix_weekly_review_sessions_expiry",
        table_name="weekly_review_sessions",
    )
    op.drop_table("weekly_review_sessions")
    op.drop_index(
        "ix_weekly_focus_changes_focus_history",
        table_name="weekly_focus_changes",
    )
    op.drop_index(
        "ix_weekly_focus_changes_owner_created",
        table_name="weekly_focus_changes",
    )
    op.drop_table("weekly_focus_changes")
    op.drop_index(
        "ix_weekly_focuses_owner_history",
        table_name="weekly_focuses",
    )
    op.drop_table("weekly_focuses")
