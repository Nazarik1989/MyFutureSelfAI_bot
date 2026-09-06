from __future__ import annotations

from datetime import date, datetime, time
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _json_string_items_check(column: str, *, maximum_items: int) -> str:
    return " AND ".join(
        f"(json_array_length({column}) < {index + 1} OR "
        f"(coalesce(substr(CAST({column} -> {index} AS TEXT), 1, 1) = '\"', false) "
        f"AND coalesce(length({column} ->> {index}), 0) BETWEEN 1 AND 200))"
        for index in range(maximum_items)
    )


def _weekly_reminder_candidate_fields_check(column: str) -> str:
    """SQLite exact-shape check; PostgreSQL gets its own conditional CHECK."""

    checks: list[str] = []
    for index in range(5):
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


def _weekly_reminder_candidate_fields_check_postgresql(column: str) -> str:
    """PostgreSQL exact-shape check using JSONB key subtraction."""

    checks: list[str] = []
    for index in range(5):
        candidate = f"({column} -> {index})"
        candidate_jsonb = f"({candidate})::jsonb"
        checks.append(
            f"(json_array_length({column}) < {index + 1} OR ("
            f"coalesce(jsonb_typeof({candidate_jsonb}) = 'object', false) "
            f"AND coalesce(jsonb_typeof({candidate_jsonb} -> 'title') = 'string', false) "
            f"AND coalesce(length({candidate_jsonb} ->> 'title'), 0) BETWEEN 1 AND 200 "
            f"AND coalesce(jsonb_typeof({candidate_jsonb} -> 'schedule_wording') = "
            f"'string', false) "
            f"AND coalesce(length({candidate_jsonb} ->> 'schedule_wording'), 0) "
            f"BETWEEN 1 AND 200 "
            f"AND ({candidate_jsonb} - 'title' - 'schedule_wording') = '{{}}'::jsonb))"
        )
    return " AND ".join(checks)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "access_tier IN ('guest', 'subscriber', 'admin', 'blocked')",
            name="ck_users_access_tier",
        ),
        CheckConstraint("access_version > 0", name="ck_users_access_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64), default="Europe/Moscow")
    location_city: Mapped[str | None] = mapped_column(String(120))
    location_fallback_city: Mapped[str | None] = mapped_column(String(120))
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    access_tier: Mapped[str] = mapped_column(String(16), default="guest", server_default="guest")
    access_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    vision_profile: Mapped[VisionProfile | None] = relationship(
        back_populates="user", uselist=False
    )
    goals: Mapped[list[Goal]] = relationship(back_populates="user")
    routines: Mapped[list[Routine]] = relationship(back_populates="user")
    inbox_items: Mapped[list[InboxItem]] = relationship(back_populates="user")
    health_check_ins: Mapped[list[HealthCheckIn]] = relationship(back_populates="user")
    doctor_visit_preps: Mapped[list[DoctorVisitPrep]] = relationship(back_populates="user")
    vision_items: Mapped[list[VisionItem]] = relationship(back_populates="owner")
    vision_item_images: Mapped[list[VisionItemImage]] = relationship(back_populates="owner")
    vision_references: Mapped[list[VisionReference]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    vision_companion_preference: Mapped[VisionCompanionPreference | None] = relationship(
        back_populates="owner", uselist=False, cascade="all, delete-orphan"
    )
    vision_companion_checkins: Mapped[list[VisionCompanionCheckIn]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    lab_documents: Mapped[list[LabDocument]] = relationship(back_populates="owner")
    task_states: Mapped[list[TaskState]] = relationship(
        back_populates="owner", overlaps="inbox_item,task_state"
    )
    recurring_task_reminder_schedules: Mapped[list[RecurringTaskReminderSchedule]] = relationship(
        back_populates="owner",
        overlaps="inbox_item,recurring_reminder_schedule",
    )
    nova_memory_items: Mapped[list[NovaMemoryItem]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    nova_memory_changes: Mapped[list[NovaMemoryChange]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    weekly_focuses: Mapped[list[WeeklyFocus]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    weekly_focus_changes: Mapped[list[WeeklyFocusChange]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    weekly_review_sessions: Mapped[list[WeeklyReviewSession]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    nova_dialogue_states: Mapped[list[NovaDialogueState]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    nova_observed_memories: Mapped[list[NovaObservedMemory]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class NovaMemoryItem(TimestampMixin, Base):
    __tablename__ = "nova_memory_items"
    __table_args__ = (
        UniqueConstraint("public_id", name="uq_nova_memory_items_public_id"),
        UniqueConstraint(
            "owner_id",
            "content_fingerprint",
            name="uq_nova_memory_items_owner_fingerprint",
        ),
        CheckConstraint(
            "category IN ('about_me', 'interaction', 'orientation')",
            name="ck_nova_memory_items_category",
        ),
        CheckConstraint(
            "length(content) BETWEEN 1 AND 500",
            name="ck_nova_memory_items_content_length",
        ),
        CheckConstraint(
            "length(content_fingerprint) = 64",
            name="ck_nova_memory_items_fingerprint_length",
        ),
        CheckConstraint("version > 0", name="ck_nova_memory_items_version"),
        Index(
            "ix_nova_memory_items_owner_list",
            "owner_id",
            "important",
            "updated_at",
            "id",
        ),
        Index(
            "ix_nova_memory_items_owner_category_list",
            "owner_id",
            "category",
            "important",
            "updated_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(24))
    content: Mapped[str] = mapped_column(Text)
    content_fingerprint: Mapped[str] = mapped_column(String(64))
    important: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
    )
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    owner: Mapped[User] = relationship(back_populates="nova_memory_items")


class NovaMemoryChange(Base):
    __tablename__ = "nova_memory_changes"
    __table_args__ = (
        CheckConstraint(
            "operation IN ('created', 'updated', 'importance_changed', 'deleted', 'deleted_all')",
            name="ck_nova_memory_changes_operation",
        ),
        CheckConstraint(
            "category IS NULL OR category IN ('about_me', 'interaction', 'orientation')",
            name="ck_nova_memory_changes_category",
        ),
        CheckConstraint(
            "memory_public_id IS NULL OR length(memory_public_id) = 36",
            name="ck_nova_memory_changes_public_id_length",
        ),
        CheckConstraint(
            "resulting_version IS NULL OR resulting_version > 0",
            name="ck_nova_memory_changes_resulting_version",
        ),
        CheckConstraint("affected_count > 0", name="ck_nova_memory_changes_affected_count"),
        CheckConstraint(
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
        Index(
            "ix_nova_memory_changes_owner_created",
            "owner_id",
            "created_at",
            "id",
        ),
        Index(
            "ix_nova_memory_changes_item_history",
            "owner_id",
            "memory_public_id",
            "created_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    memory_public_id: Mapped[str | None] = mapped_column(String(36))
    operation: Mapped[str] = mapped_column(String(24))
    category: Mapped[str | None] = mapped_column(String(24))
    resulting_version: Mapped[int | None] = mapped_column(Integer)
    affected_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    owner: Mapped[User] = relationship(back_populates="nova_memory_changes")


class WeeklyFocus(TimestampMixin, Base):
    __tablename__ = "weekly_focuses"
    __table_args__ = (
        UniqueConstraint("public_id", name="uq_weekly_focuses_public_id"),
        UniqueConstraint("owner_id", "week_start", name="uq_weekly_focuses_owner_week"),
        CheckConstraint(
            "length(public_id) = 36",
            name="ck_weekly_focuses_public_id_length",
        ),
        CheckConstraint(
            "length(focus) BETWEEN 1 AND 300",
            name="ck_weekly_focuses_focus_length",
        ),
        CheckConstraint(
            "approach IS NULL OR length(approach) BETWEEN 1 AND 500",
            name="ck_weekly_focuses_approach_length",
        ),
        CheckConstraint(
            "json_array_length(small_steps) BETWEEN 0 AND 3",
            name="ck_weekly_focuses_small_steps_count",
        ),
        CheckConstraint(
            "substr(CAST(small_steps AS TEXT), 1, 1) = '[' "
            "AND length(CAST(small_steps AS TEXT)) BETWEEN 2 AND 4096",
            name="ck_weekly_focuses_small_steps_json_shape",
        ),
        CheckConstraint(
            _json_string_items_check("small_steps", maximum_items=3),
            name="ck_weekly_focuses_small_steps_lengths",
        ),
        CheckConstraint(
            "source IN ('text', 'voice')",
            name="ck_weekly_focuses_source",
        ),
        CheckConstraint("version > 0", name="ck_weekly_focuses_version"),
        Index("ix_weekly_focuses_owner_history", "owner_id", "week_start", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    week_start: Mapped[date] = mapped_column(Date)
    focus: Mapped[str] = mapped_column(Text)
    approach: Mapped[str | None] = mapped_column(Text)
    small_steps: Mapped[list[str]] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    source: Mapped[str] = mapped_column(String(8))
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    owner: Mapped[User] = relationship(back_populates="weekly_focuses")


class WeeklyFocusChange(Base):
    __tablename__ = "weekly_focus_changes"
    __table_args__ = (
        CheckConstraint(
            "operation IN ('created', 'updated', 'deleted')",
            name="ck_weekly_focus_changes_operation",
        ),
        CheckConstraint(
            "length(focus_public_id) = 36",
            name="ck_weekly_focus_changes_public_id_length",
        ),
        CheckConstraint(
            "resulting_version > 0",
            name="ck_weekly_focus_changes_resulting_version",
        ),
        Index(
            "ix_weekly_focus_changes_owner_created",
            "owner_id",
            "created_at",
            "id",
        ),
        Index(
            "ix_weekly_focus_changes_focus_history",
            "owner_id",
            "focus_public_id",
            "created_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    focus_public_id: Mapped[str] = mapped_column(String(36))
    operation: Mapped[str] = mapped_column(String(12))
    resulting_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    owner: Mapped[User] = relationship(back_populates="weekly_focus_changes")


class WeeklyReviewSession(TimestampMixin, Base):
    __tablename__ = "weekly_review_sessions"
    __table_args__ = (
        UniqueConstraint("public_id", name="uq_weekly_review_sessions_public_id"),
        UniqueConstraint(
            "owner_id",
            "chat_id",
            name="uq_weekly_review_sessions_owner_chat",
        ),
        CheckConstraint(
            "length(public_id) = 36",
            name="ck_weekly_review_sessions_public_id_length",
        ),
        CheckConstraint(
            "phase IN ('root', 'awaiting_input', 'processing', 'preview', 'saved', "
            "'candidates', 'reminder_handoff', 'delete_preview', 'completed')",
            name="ck_weekly_review_sessions_phase",
        ),
        CheckConstraint(
            "access_version > 0",
            name="ck_weekly_review_sessions_access_version",
        ),
        CheckConstraint(
            "version > 0",
            name="ck_weekly_review_sessions_version",
        ),
        CheckConstraint(
            "extracted_focus IS NULL OR length(extracted_focus) BETWEEN 1 AND 300",
            name="ck_weekly_review_sessions_focus_length",
        ),
        CheckConstraint(
            "extracted_approach IS NULL OR length(extracted_approach) BETWEEN 1 AND 500",
            name="ck_weekly_review_sessions_approach_length",
        ),
        CheckConstraint(
            "json_array_length(small_steps) BETWEEN 0 AND 3",
            name="ck_weekly_review_sessions_small_steps_count",
        ),
        CheckConstraint(
            "substr(CAST(small_steps AS TEXT), 1, 1) = '[' "
            "AND length(CAST(small_steps AS TEXT)) BETWEEN 2 AND 4096",
            name="ck_weekly_review_sessions_small_steps_json_shape",
        ),
        CheckConstraint(
            _json_string_items_check("small_steps", maximum_items=3),
            name="ck_weekly_review_sessions_small_steps_lengths",
        ),
        CheckConstraint(
            "json_array_length(reminder_candidates) BETWEEN 0 AND 5",
            name="ck_weekly_review_sessions_candidates_count",
        ),
        CheckConstraint(
            "substr(CAST(reminder_candidates AS TEXT), 1, 1) = '[' "
            "AND length(CAST(reminder_candidates AS TEXT)) BETWEEN 2 AND 20000",
            name="ck_weekly_review_sessions_candidates_json_shape",
        ),
        CheckConstraint(
            _weekly_reminder_candidate_fields_check("reminder_candidates"),
            name="ck_weekly_review_sessions_candidates_fields",
        ).ddl_if(dialect="sqlite"),
        CheckConstraint(
            _weekly_reminder_candidate_fields_check_postgresql("reminder_candidates"),
            name="ck_weekly_review_sessions_candidates_fields",
        ).ddl_if(dialect="postgresql"),
        CheckConstraint(
            "extracted_source IS NULL OR extracted_source IN ('text', 'voice')",
            name="ck_weekly_review_sessions_source",
        ),
        CheckConstraint(
            "(canonical_chat_id IS NULL AND canonical_message_id IS NULL) OR "
            "(canonical_chat_id IS NOT NULL AND canonical_message_id IS NOT NULL "
            "AND canonical_chat_id = chat_id AND canonical_message_id > 0)",
            name="ck_weekly_review_sessions_canonical_binding",
        ),
        CheckConstraint(
            "(base_focus_public_id IS NULL AND base_focus_version IS NULL) OR "
            "(base_focus_public_id IS NOT NULL AND base_focus_version IS NOT NULL "
            "AND length(base_focus_public_id) = 36 AND base_focus_version > 0)",
            name="ck_weekly_review_sessions_base_focus_generation",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_weekly_review_sessions_expiry_order",
        ),
        Index(
            "ix_weekly_review_sessions_expiry",
            "expires_at",
            "id",
        ),
        Index(
            "ix_weekly_review_sessions_owner_week",
            "owner_id",
            "week_start",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    telegram_user_id: Mapped[int] = mapped_column(BigInteger)
    chat_id: Mapped[int] = mapped_column(BigInteger)
    access_version: Mapped[int] = mapped_column(Integer)
    week_start: Mapped[date] = mapped_column(Date)
    phase: Mapped[str] = mapped_column(String(24))
    canonical_chat_id: Mapped[int | None] = mapped_column(BigInteger)
    canonical_message_id: Mapped[int | None] = mapped_column(BigInteger)
    base_focus_public_id: Mapped[str | None] = mapped_column(String(36))
    base_focus_version: Mapped[int | None] = mapped_column(Integer)
    extracted_focus: Mapped[str | None] = mapped_column(Text)
    extracted_approach: Mapped[str | None] = mapped_column(Text)
    small_steps: Mapped[list[str]] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    reminder_candidates: Mapped[list[dict[str, str]]] = mapped_column(
        JSON,
        default=list,
        server_default=text("'[]'"),
    )
    extracted_source: Mapped[str | None] = mapped_column(String(8))
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    owner: Mapped[User] = relationship(back_populates="weekly_review_sessions")


class AccessTierChange(Base):
    __tablename__ = "access_tier_changes"
    __table_args__ = (
        CheckConstraint(
            "from_tier IN ('guest', 'subscriber', 'admin', 'blocked')",
            name="ck_access_tier_changes_from_tier",
        ),
        CheckConstraint(
            "to_tier IN ('guest', 'subscriber', 'admin', 'blocked')",
            name="ck_access_tier_changes_to_tier",
        ),
        CheckConstraint(
            "length(source) BETWEEN 1 AND 64",
            name="ck_access_tier_changes_source_length",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    from_tier: Mapped[str] = mapped_column(String(16))
    to_tier: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GuestQuotaDay(Base):
    __tablename__ = "guest_quota_days"

    quota_day: Mapped[date] = mapped_column(Date, primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class GuestUsageLedger(Base):
    __tablename__ = "guest_usage_ledger"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_guest_usage_user_idempotency",
        ),
        UniqueConstraint("reservation_token", name="uq_guest_usage_reservation_token"),
        CheckConstraint(
            "demo_kind IN ('thought_breakdown', 'first_step')",
            name="ck_guest_usage_demo_kind",
        ),
        CheckConstraint(
            "status IN ('reserved', 'succeeded', 'failed', 'expired')",
            name="ck_guest_usage_status",
        ),
        CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 128",
            name="ck_guest_usage_idempotency_length",
        ),
        CheckConstraint(
            "expires_at > reserved_at",
            name="ck_guest_usage_expiry_order",
        ),
        CheckConstraint(
            "(status = 'reserved' AND completed_at IS NULL) OR "
            "(status IN ('succeeded', 'failed', 'expired') AND completed_at IS NOT NULL)",
            name="ck_guest_usage_completion_state",
        ),
        CheckConstraint(
            "provider_started_at IS NULL OR "
            "(provider_started_at >= reserved_at AND provider_started_at < expires_at)",
            name="ck_guest_usage_provider_start_window",
        ),
        CheckConstraint(
            "status <> 'succeeded' OR provider_started_at IS NOT NULL",
            name="ck_guest_usage_success_requires_provider_start",
        ),
        Index(
            "uq_guest_usage_one_reserved_per_user",
            "user_id",
            unique=True,
            sqlite_where=text("status = 'reserved'"),
            postgresql_where=text("status = 'reserved'"),
        ),
        Index(
            "ix_guest_usage_quota_status_expiry",
            "quota_day",
            "status",
            "expires_at",
        ),
        Index("ix_guest_usage_user_status", "user_id", "status"),
        Index("ix_guest_usage_provider_started_at", "provider_started_at"),
        Index(
            "ix_guest_usage_unstarted_status_expiry",
            "status",
            "expires_at",
            sqlite_where=text("provider_started_at IS NULL"),
            postgresql_where=text("provider_started_at IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    demo_kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    telegram_update_id: Mapped[int | None] = mapped_column(BigInteger)
    reservation_token: Mapped[str] = mapped_column(String(64))
    quota_day: Mapped[date] = mapped_column(Date)
    reserved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    provider_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GuestDemoSession(Base):
    __tablename__ = "guest_demo_sessions"
    __table_args__ = (
        UniqueConstraint("user_id", "chat_id", name="uq_guest_demo_session_user_chat"),
        CheckConstraint(
            "demo_kind IN ('thought_breakdown', 'first_step')",
            name="ck_guest_demo_session_kind",
        ),
        CheckConstraint(
            "status IN ('awaiting_input', 'processing', 'result_ready', "
            "'completed', 'cancelled', 'expired')",
            name="ck_guest_demo_session_status",
        ),
        CheckConstraint("version > 0", name="ck_guest_demo_session_version"),
        CheckConstraint("access_version > 0", name="ck_guest_demo_session_access_version"),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_guest_demo_session_expiry_order",
        ),
        CheckConstraint(
            "(status = 'result_ready' AND result_payload IS NOT NULL "
            "AND result_expires_at IS NOT NULL) OR "
            "(status <> 'result_ready' AND result_payload IS NULL "
            "AND result_expires_at IS NULL)",
            name="ck_guest_demo_session_result_state",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger)
    access_version: Mapped[int] = mapped_column(Integer)
    demo_kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(20))
    prompt_message_id: Mapped[int | None] = mapped_column(BigInteger)
    consumed_update_id: Mapped[int | None] = mapped_column(BigInteger)
    consumed_message_id: Mapped[int | None] = mapped_column(BigInteger)
    usage_id: Mapped[int | None] = mapped_column(
        ForeignKey("guest_usage_ledger.id", ondelete="SET NULL")
    )
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    result_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, default=1)


class DraftInboxItem(Base):
    __tablename__ = "draft_inbox_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    source: Mapped[str] = mapped_column(String(20))
    raw_text: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    next_step: Mapped[str | None] = mapped_column(Text)
    resolved_date: Mapped[date | None] = mapped_column(Date)
    temporal_resolution: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="preview", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, default=1)
    preview_message_id: Mapped[int | None] = mapped_column(BigInteger)


class ConversationSession(TimestampMixin, Base):
    __tablename__ = "conversation_sessions"
    __table_args__ = (
        UniqueConstraint("telegram_user_id", "chat_id", name="uq_conversation_session_user_chat"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    current_topic: Mapped[str | None] = mapped_column(String(200))
    summary: Mapped[str | None] = mapped_column(Text)
    pending_date_options: Mapped[list[dict[str, str]] | None] = mapped_column(JSON)
    resolved_date: Mapped[date | None] = mapped_column(Date)
    active_draft_id: Mapped[str | None] = mapped_column(
        ForeignKey("draft_inbox_items.id", ondelete="SET NULL")
    )
    focused_draft_id: Mapped[str | None] = mapped_column(
        ForeignKey("draft_inbox_items.id", ondelete="SET NULL")
    )
    focused_draft_version: Mapped[int | None] = mapped_column(Integer)
    pending_action: Mapped[str | None] = mapped_column(String(20))
    focus_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    system_pending_action: Mapped[str | None] = mapped_column(String(40))
    system_draft_snapshot: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    system_action_version: Mapped[int] = mapped_column(Integer, default=0)
    system_action_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_saved_inbox_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("inbox_items.id", ondelete="SET NULL")
    )
    last_saved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    messages: Mapped[list[ConversationMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    source: Mapped[str] = mapped_column(String(20), default="text")
    intent: Mapped[str] = mapped_column(String(40))
    session: Mapped[ConversationSession] = relationship(back_populates="messages")


class NovaDialogueState(TimestampMixin, Base):
    __tablename__ = "nova_dialogue_states"
    __table_args__ = (
        UniqueConstraint(
            "telegram_user_id",
            "chat_id",
            name="uq_nova_dialogue_states_actor_chat",
        ),
        CheckConstraint("access_version > 0", name="ck_nova_dialogue_states_access_version"),
        CheckConstraint("revision > 0", name="ck_nova_dialogue_states_revision"),
        CheckConstraint(
            "active_topic IS NULL OR length(active_topic) BETWEEN 1 AND 200",
            name="ck_nova_dialogue_states_active_topic_length",
        ),
        CheckConstraint(
            "current_user_goal IS NULL OR length(current_user_goal) BETWEEN 1 AND 300",
            name="ck_nova_dialogue_states_user_goal_length",
        ),
        CheckConstraint(
            "last_assistant_offer IS NULL OR length(last_assistant_offer) BETWEEN 1 AND 600",
            name="ck_nova_dialogue_states_offer_length",
        ),
        CheckConstraint(
            "unresolved_question IS NULL OR length(unresolved_question) BETWEEN 1 AND 300",
            name="ck_nova_dialogue_states_question_length",
        ),
        CheckConstraint(
            "requested_action IS NULL OR requested_action IN "
            "('capture', 'reminder', 'plan', 'memory', 'clarify')",
            name="ck_nova_dialogue_states_requested_action",
        ),
        CheckConstraint(
            "json_array_length(last_assistant_offer_kinds) BETWEEN 0 AND 4",
            name="ck_nova_dialogue_states_offer_kinds_count",
        ),
        CheckConstraint(
            "json_array_length(open_loops) BETWEEN 0 AND 5",
            name="ck_nova_dialogue_states_open_loops_count",
        ),
        CheckConstraint(
            "length(CAST(last_assistant_offer_kinds AS TEXT)) BETWEEN 2 AND 1024",
            name="ck_nova_dialogue_states_offer_kinds_bytes",
        ),
        CheckConstraint(
            "length(CAST(open_loops AS TEXT)) BETWEEN 2 AND 4096",
            name="ck_nova_dialogue_states_open_loops_bytes",
        ),
        Index(
            "ix_nova_dialogue_states_owner_expiry",
            "owner_id",
            "expires_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    access_version: Mapped[int] = mapped_column(Integer)
    active_topic: Mapped[str | None] = mapped_column(String(200))
    current_user_goal: Mapped[str | None] = mapped_column(String(300))
    last_assistant_offer: Mapped[str | None] = mapped_column(String(600))
    last_assistant_offer_kinds: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        server_default=text("'[]'"),
    )
    unresolved_question: Mapped[str | None] = mapped_column(String(300))
    requested_action: Mapped[str | None] = mapped_column(String(16))
    open_loops: Mapped[list[str]] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    revision: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    owner: Mapped[User] = relationship(back_populates="nova_dialogue_states")


class NovaObservedMemory(TimestampMixin, Base):
    __tablename__ = "nova_observed_memories"
    __table_args__ = (
        UniqueConstraint("public_id", name="uq_nova_observed_memories_public_id"),
        UniqueConstraint(
            "owner_id",
            "content_fingerprint",
            name="uq_nova_observed_memories_owner_fingerprint",
        ),
        CheckConstraint(
            "category IN ('fact', 'preference', 'orientation', 'theme', 'identity')",
            name="ck_nova_observed_memories_category",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded', 'forgotten')",
            name="ck_nova_observed_memories_status",
        ),
        CheckConstraint(
            "source_kind IN ('conversation', 'explicit')",
            name="ck_nova_observed_memories_source_kind",
        ),
        CheckConstraint(
            "length(normalized_value) BETWEEN 1 AND 500",
            name="ck_nova_observed_memories_value_length",
        ),
        CheckConstraint(
            "length(content_fingerprint) = 64",
            name="ck_nova_observed_memories_fingerprint_length",
        ),
        CheckConstraint(
            "length(source_receipt) = 64",
            name="ck_nova_observed_memories_receipt_length",
        ),
        CheckConstraint("salience BETWEEN 1 AND 5", name="ck_nova_observed_memories_salience"),
        CheckConstraint("revision > 0", name="ck_nova_observed_memories_revision"),
        CheckConstraint(
            "superseded_by_public_id IS NULL OR length(superseded_by_public_id) = 36",
            name="ck_nova_observed_memories_superseded_id_length",
        ),
        CheckConstraint(
            "semantic_key IS NULL OR "
            "(semantic_key = 'identity' AND category = 'identity') OR "
            "(semantic_key IN ('response_length', 'tone', 'reminder_style') "
            "AND category = 'preference')",
            name="ck_nova_observed_memories_semantic_key",
        ),
        Index(
            "ix_nova_observed_memories_owner_active",
            "owner_id",
            "status",
            "updated_at",
            "id",
        ),
        Index(
            "ix_nova_observed_memories_owner_category",
            "owner_id",
            "category",
            "status",
            "updated_at",
            "id",
        ),
        Index(
            "uq_nova_observed_memories_owner_active_semantic_key",
            "owner_id",
            "semantic_key",
            unique=True,
            sqlite_where=text("status = 'active' AND semantic_key IS NOT NULL"),
            postgresql_where=text("status = 'active' AND semantic_key IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(16))
    semantic_key: Mapped[str | None] = mapped_column(String(32))
    normalized_value: Mapped[str] = mapped_column(Text)
    content_fingerprint: Mapped[str] = mapped_column(String(64))
    source_kind: Mapped[str] = mapped_column(String(16))
    source_session_id: Mapped[int] = mapped_column(Integer)
    source_message_id: Mapped[int] = mapped_column(Integer)
    source_receipt: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="active", server_default="active")
    salience: Mapped[int] = mapped_column(Integer, default=3, server_default="3")
    revision: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    superseded_by_public_id: Mapped[str | None] = mapped_column(String(36))
    owner: Mapped[User] = relationship(back_populates="nova_observed_memories")


class VisionProfile(TimestampMixin, Base):
    __tablename__ = "vision_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    raw_answers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text)
    values: Mapped[list[str]] = mapped_column(JSON, default=list)
    desired_identity: Mapped[list[str]] = mapped_column(JSON, default=list)
    constraints: Mapped[list[str]] = mapped_column(JSON, default=list)
    motivation_style: Mapped[str | None] = mapped_column(String(120))
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user: Mapped[User] = relationship(back_populates="vision_profile")


class Goal(TimestampMixin, Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    life_area: Mapped[str] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(String(200))
    outcome: Mapped[str] = mapped_column(Text)
    progress_criterion: Mapped[str] = mapped_column(Text)
    horizon: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="proposed", index=True)
    priority: Mapped[int] = mapped_column(Integer, default=1)
    vision_link: Mapped[str] = mapped_column(Text)
    user: Mapped[User] = relationship(back_populates="goals")
    routines: Mapped[list[Routine]] = relationship(back_populates="goal")


class Routine(TimestampMixin, Base):
    __tablename__ = "routines"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    goal_id: Mapped[int] = mapped_column(ForeignKey("goals.id", ondelete="CASCADE"), index=True)
    frequency: Mapped[str] = mapped_column(String(100))
    minimum_version: Mapped[str] = mapped_column(Text)
    normal_version: Mapped[str] = mapped_column(Text)
    preferred_time: Mapped[str | None] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="proposed", index=True)
    user: Mapped[User] = relationship(back_populates="routines")
    goal: Mapped[Goal] = relationship(back_populates="routines")


class InboxItem(TimestampMixin, Base):
    __tablename__ = "inbox_items"
    __table_args__ = (
        UniqueConstraint("id", "user_id", name="uq_inbox_item_id_user"),
        CheckConstraint("version > 0", name="ck_inbox_item_version"),
        CheckConstraint(
            "(status = 'trashed' AND trashed_at IS NOT NULL "
            "AND pre_trash_status IN ('confirmed', 'archived')) OR "
            "(status != 'trashed' AND trashed_at IS NULL "
            "AND pre_trash_status IS NULL)",
            name="ck_inbox_item_trash_state",
        ),
        Index("ix_inbox_items_user_status_id", "user_id", "status", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    draft_id: Mapped[str | None] = mapped_column(
        ForeignKey("draft_inbox_items.id", ondelete="SET NULL"), unique=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text)
    next_step: Mapped[str | None] = mapped_column(Text)
    resolved_date: Mapped[date | None] = mapped_column(Date)
    temporal_resolution: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(20), default="text")
    status: Mapped[str] = mapped_column(String(20), default="confirmed", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    trashed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    pre_trash_status: Mapped[str | None] = mapped_column(String(20))
    user: Mapped[User] = relationship(back_populates="inbox_items")
    reminder: Mapped[TaskReminder | None] = relationship(
        back_populates="inbox_item", uselist=False, cascade="all, delete-orphan"
    )
    recurring_reminder_schedule: Mapped[RecurringTaskReminderSchedule | None] = relationship(
        back_populates="inbox_item",
        uselist=False,
        cascade="all, delete-orphan",
        overlaps="owner,recurring_task_reminder_schedules",
    )
    task_state: Mapped[TaskState | None] = relationship(
        back_populates="inbox_item",
        uselist=False,
        cascade="all, delete-orphan",
        overlaps="owner,task_states",
    )


class VisionItem(TimestampMixin, Base):
    __tablename__ = "vision_items"
    __table_args__ = (
        CheckConstraint(
            "category IN ('health_energy', 'relationships_family', 'work_purpose', "
            "'money', 'home', 'travel', 'growth_creativity', 'other')",
            name="ck_vision_item_category",
        ),
        CheckConstraint(
            "status IN ('active', 'achieved', 'archived')",
            name="ck_vision_item_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(40), index=True)
    wish_text: Mapped[str] = mapped_column(Text)
    why_text: Mapped[str | None] = mapped_column(Text)
    target_date: Mapped[date | None] = mapped_column(Date)
    first_step: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    linked_task_id: Mapped[int | None] = mapped_column(
        ForeignKey("inbox_items.id", ondelete="SET NULL"), unique=True
    )
    owner: Mapped[User] = relationship(back_populates="vision_items")
    linked_task: Mapped[InboxItem | None] = relationship()
    image: Mapped[VisionItemImage | None] = relationship(
        back_populates="vision_item",
        uselist=False,
        cascade="all, delete-orphan",
    )
    companion_preference: Mapped[VisionCompanionPreference | None] = relationship(
        back_populates="vision_item", uselist=False
    )
    companion_checkins: Mapped[list[VisionCompanionCheckIn]] = relationship(
        back_populates="vision_item", cascade="all, delete-orphan"
    )


class VisionItemImage(TimestampMixin, Base):
    __tablename__ = "vision_item_images"
    __table_args__ = (
        CheckConstraint("width > 0 AND height > 0", name="ck_vision_item_image_dimensions"),
        CheckConstraint("version > 0", name="ck_vision_item_image_version"),
        CheckConstraint(
            "mime_type IN ('image/jpeg', 'image/png', 'image/webp')",
            name="ck_vision_item_image_mime_type",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    vision_item_id: Mapped[int] = mapped_column(
        ForeignKey("vision_items.id", ondelete="CASCADE"), unique=True, index=True
    )
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    mime_type: Mapped[str] = mapped_column(String(40))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    vision_item: Mapped[VisionItem] = relationship(back_populates="image")
    owner: Mapped[User] = relationship(back_populates="vision_item_images")


class VisionReference(TimestampMixin, Base):
    __tablename__ = "vision_references"
    __table_args__ = (
        UniqueConstraint("owner_id", "sha256", name="uq_vision_reference_owner_sha256"),
        CheckConstraint(
            "kind IN ('self', 'person', 'place', 'object', 'style')",
            name="ck_vision_reference_kind",
        ),
        CheckConstraint(
            "length(name) BETWEEN 1 AND 60",
            name="ck_vision_reference_name_length",
        ),
        CheckConstraint("width > 0 AND height > 0", name="ck_vision_reference_dimensions"),
        CheckConstraint("version > 0", name="ck_vision_reference_version"),
        CheckConstraint(
            "mime_type IN ('image/jpeg', 'image/png', 'image/webp')",
            name="ck_vision_reference_mime_type",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(60))
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    mime_type: Mapped[str] = mapped_column(String(40))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    owner: Mapped[User] = relationship(back_populates="vision_references")


class VisionCompanionPreference(TimestampMixin, Base):
    __tablename__ = "vision_companion_preferences"
    __table_args__ = (
        CheckConstraint("extra_per_day BETWEEN 0 AND 3", name="ck_vision_companion_extra_count"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    vision_item_id: Mapped[int] = mapped_column(
        ForeignKey("vision_items.id", ondelete="CASCADE"), unique=True, index=True
    )
    telegram_user_id: Mapped[int] = mapped_column(BigInteger)
    chat_id: Mapped[int] = mapped_column(BigInteger)
    timezone: Mapped[str] = mapped_column(String(64))
    morning_time: Mapped[time] = mapped_column(Time)
    evening_time: Mapped[time] = mapped_column(Time)
    extra_per_day: Mapped[int] = mapped_column(Integer, default=0)
    extra_times: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    owner: Mapped[User] = relationship(back_populates="vision_companion_preference")
    vision_item: Mapped[VisionItem] = relationship(back_populates="companion_preference")


class VisionCompanionCheckIn(TimestampMixin, Base):
    __tablename__ = "vision_companion_checkins"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "vision_item_id",
            "local_date",
            "moment",
            name="uq_vision_companion_checkin_day",
        ),
        CheckConstraint(
            "moment IN ('morning', 'evening', 'extra')",
            name="ck_vision_companion_checkin_moment",
        ),
        CheckConstraint(
            "response IN ('committed', 'pause', 'done', 'partial', 'missed', 'later')",
            name="ck_vision_companion_checkin_response",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    vision_item_id: Mapped[int] = mapped_column(
        ForeignKey("vision_items.id", ondelete="CASCADE"), index=True
    )
    local_date: Mapped[date] = mapped_column(Date, index=True)
    moment: Mapped[str] = mapped_column(String(16))
    response: Mapped[str] = mapped_column(String(16))
    owner: Mapped[User] = relationship(back_populates="vision_companion_checkins")
    vision_item: Mapped[VisionItem] = relationship(back_populates="companion_checkins")


class LabDocument(TimestampMixin, Base):
    __tablename__ = "lab_documents"
    __table_args__ = (
        UniqueConstraint("id", "owner_id", name="uq_lab_document_id_owner"),
        CheckConstraint("page_count > 0", name="ck_lab_document_page_count"),
        CheckConstraint(
            "length(title) BETWEEN 1 AND 200",
            name="ck_lab_document_title_length",
        ),
        CheckConstraint("version > 0", name="ck_lab_document_version"),
        CheckConstraint(
            "source_type IN ('image', 'pdf')",
            name="ck_lab_document_source_type",
        ),
        CheckConstraint(
            "status IN ('saved')",
            name="ck_lab_document_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    document_date: Mapped[date | None] = mapped_column(Date, index=True)
    source_type: Mapped[str] = mapped_column(String(20))
    page_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="saved", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    owner: Mapped[User] = relationship(back_populates="lab_documents")
    pages: Mapped[list[LabDocumentPage]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="LabDocumentPage.page_index",
    )


class LabDocumentPage(Base):
    __tablename__ = "lab_document_pages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "owner_id"],
            ["lab_documents.id", "lab_documents.owner_id"],
            ondelete="CASCADE",
            name="fk_lab_page_document_owner",
        ),
        UniqueConstraint("document_id", "page_index", name="uq_lab_page_document_index"),
        Index("ix_lab_document_pages_owner_document", "owner_id", "document_id"),
        CheckConstraint("page_index >= 0", name="ck_lab_page_index"),
        CheckConstraint("width > 0 AND height > 0", name="ck_lab_page_dimensions"),
        CheckConstraint("length(image_bytes) > 0", name="ck_lab_page_has_bytes"),
        CheckConstraint("mime_type = 'image/jpeg'", name="ck_lab_page_mime_type"),
        CheckConstraint("length(sha256) = 64", name="ck_lab_page_sha256_length"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, index=True)
    owner_id: Mapped[int] = mapped_column(Integer, index=True)
    page_index: Mapped[int] = mapped_column(Integer)
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    mime_type: Mapped[str] = mapped_column(String(40))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document: Mapped[LabDocument] = relationship(back_populates="pages")


class LabDeleteConfirmation(Base):
    __tablename__ = "lab_delete_confirmations"
    __table_args__ = (
        CheckConstraint("document_version > 0", name="ck_lab_delete_version"),
        CheckConstraint(
            "status IN ('pending', 'consumed')",
            name="ck_lab_delete_status",
        ),
    )

    token: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger)
    document_id: Mapped[int] = mapped_column(Integer, index=True)
    document_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VisionDraft(TimestampMixin, Base):
    __tablename__ = "vision_drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    chat_id: Mapped[int] = mapped_column(BigInteger)
    step: Mapped[str] = mapped_column(String(30), default="category")
    category: Mapped[str | None] = mapped_column(String(40))
    wish_text: Mapped[str | None] = mapped_column(Text)
    why_text: Mapped[str | None] = mapped_column(Text)
    target_date: Mapped[date | None] = mapped_column(Date)
    first_step: Mapped[str | None] = mapped_column(Text)
    editing_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("vision_items.id", ondelete="CASCADE")
    )
    edit_field: Mapped[str | None] = mapped_column(String(30))
    version: Mapped[int] = mapped_column(Integer, default=1)


class TaskState(TimestampMixin, Base):
    __tablename__ = "task_states"
    __table_args__ = (
        ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_task_state_inbox_owner",
        ),
        CheckConstraint(
            "status IN ('active', 'completed', 'cancelled')",
            name="ck_task_state_status",
        ),
        CheckConstraint("version > 0", name="ck_task_state_version"),
        UniqueConstraint("owner_id", "inbox_item_id", name="uq_task_state_owner_item"),
        Index("ix_task_states_inbox_item_id", "inbox_item_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    inbox_item_id: Mapped[int] = mapped_column(Integer, unique=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    timezone: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    owner: Mapped[User] = relationship(
        back_populates="task_states", overlaps="inbox_item,task_state"
    )
    inbox_item: Mapped[InboxItem] = relationship(
        back_populates="task_state", overlaps="owner,task_states"
    )


class TaskActionToken(Base):
    __tablename__ = "task_action_tokens"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_id", "inbox_item_id"],
            ["task_states.owner_id", "task_states.inbox_item_id"],
            ondelete="CASCADE",
            name="fk_task_action_state_owner",
        ),
        CheckConstraint("task_version > 0", name="ck_task_action_version"),
        CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_task_action_status",
        ),
    )

    token: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    inbox_item_id: Mapped[int] = mapped_column(Integer, index=True)
    task_version: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(40), index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TaskReminder(TimestampMixin, Base):
    __tablename__ = "task_reminders"

    id: Mapped[int] = mapped_column(primary_key=True)
    inbox_item_id: Mapped[int] = mapped_column(
        ForeignKey("inbox_items.id", ondelete="CASCADE"), unique=True
    )
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger)
    event_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    remind_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    timezone: Mapped[str] = mapped_column(String(64))
    delivery_key: Mapped[str] = mapped_column(String(80), unique=True)
    task_version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    claim_token: Mapped[str | None] = mapped_column(String(36))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    telegram_message_id: Mapped[int | None] = mapped_column(BigInteger)
    last_error_type: Mapped[str | None] = mapped_column(String(120))
    inbox_item: Mapped[InboxItem] = relationship(back_populates="reminder")


class RecurringTaskReminderSchedule(TimestampMixin, Base):
    __tablename__ = "recurring_task_reminder_schedules"
    __table_args__ = (
        ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_recurring_schedule_inbox_owner",
        ),
        UniqueConstraint(
            "inbox_item_id",
            name="uq_recurring_schedule_inbox_item",
        ),
        CheckConstraint(
            "recurrence_kind IN ('daily')",
            name="ck_recurring_schedule_kind",
        ),
        CheckConstraint(
            "timezone_source IN ('profile', 'explicit')",
            name="ck_recurring_schedule_timezone_source",
        ),
        CheckConstraint(
            "status IN ('active', 'disabled', 'completed')",
            name="ck_recurring_schedule_status",
        ),
        CheckConstraint("version > 0", name="ck_recurring_schedule_version"),
        CheckConstraint(
            "length(timezone) BETWEEN 1 AND 64",
            name="ck_recurring_schedule_timezone_length",
        ),
        Index(
            "ix_recurring_schedule_due",
            "status",
            "next_occurrence_at",
        ),
        Index(
            "ix_recurring_schedule_owner_status",
            "owner_id",
            "status",
            "next_occurrence_at",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    inbox_item_id: Mapped[int] = mapped_column(Integer)
    recurrence_kind: Mapped[str] = mapped_column(String(16), default="daily")
    local_time: Mapped[time] = mapped_column(Time)
    timezone: Mapped[str] = mapped_column(String(64))
    timezone_source: Mapped[str] = mapped_column(String(16), default="profile")
    start_local_date: Mapped[date] = mapped_column(Date)
    next_occurrence_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="active")
    version: Mapped[int] = mapped_column(Integer, default=1)
    owner: Mapped[User] = relationship(
        back_populates="recurring_task_reminder_schedules",
        overlaps="inbox_item,recurring_reminder_schedule",
    )
    inbox_item: Mapped[InboxItem] = relationship(
        back_populates="recurring_reminder_schedule",
        overlaps="owner,recurring_task_reminder_schedules",
    )
    occurrences: Mapped[list[RecurringTaskReminderOccurrence]] = relationship(
        back_populates="schedule",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="RecurringTaskReminderOccurrence.scheduled_for",
    )


class RecurringTaskReminderOccurrence(TimestampMixin, Base):
    __tablename__ = "recurring_task_reminder_occurrences"
    __table_args__ = (
        UniqueConstraint(
            "schedule_id",
            "schedule_version",
            "scheduled_for",
            name="uq_recurring_occurrence_generation_time",
        ),
        UniqueConstraint(
            "schedule_id",
            "schedule_version",
            "local_date",
            name="uq_recurring_occurrence_generation_date",
        ),
        UniqueConstraint(
            "delivery_key",
            name="uq_recurring_occurrence_delivery_key",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'sent', 'skipped_stale', 'cancelled')",
            name="ck_recurring_occurrence_status",
        ),
        CheckConstraint(
            "schedule_version > 0",
            name="ck_recurring_occurrence_schedule_version",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_recurring_occurrence_attempt_count",
        ),
        CheckConstraint(
            "(status = 'processing' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL) OR "
            "(status <> 'processing' AND claim_token IS NULL AND claimed_at IS NULL)",
            name="ck_recurring_occurrence_claim_state",
        ),
        CheckConstraint(
            "(status = 'sent' AND sent_at IS NOT NULL) OR (status <> 'sent' AND sent_at IS NULL)",
            name="ck_recurring_occurrence_sent_state",
        ),
        CheckConstraint(
            "(delivery_started_at IS NULL AND status <> 'sent') OR "
            "(delivery_started_at IS NOT NULL AND status IN ('processing', 'sent'))",
            name="ck_recurring_occurrence_delivery_started_state",
        ),
        CheckConstraint(
            "length(delivery_key) BETWEEN 1 AND 128",
            name="ck_recurring_occurrence_delivery_key_length",
        ),
        Index(
            "ix_recurring_occurrence_due",
            "status",
            "scheduled_for",
            "next_attempt_at",
        ),
        Index(
            "ix_recurring_occurrence_claims",
            "status",
            "delivery_started_at",
            "claimed_at",
        ),
        Index(
            "ix_recurring_occurrence_schedule_history",
            "schedule_id",
            "created_at",
        ),
        Index(
            "ix_recurring_occurrence_cleanup",
            "status",
            "updated_at",
        ),
        Index(
            "uq_recurring_occurrence_sent_date",
            "schedule_id",
            "local_date",
            unique=True,
            sqlite_where=text("status = 'sent'"),
            postgresql_where=text("status = 'sent'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    schedule_id: Mapped[int] = mapped_column(
        ForeignKey("recurring_task_reminder_schedules.id", ondelete="CASCADE")
    )
    schedule_version: Mapped[int] = mapped_column(Integer)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    local_date: Mapped[date] = mapped_column(Date)
    delivery_key: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    claim_token: Mapped[str | None] = mapped_column(String(36))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    telegram_message_id: Mapped[int | None] = mapped_column(BigInteger)
    last_error_type: Mapped[str | None] = mapped_column(String(120))
    schedule: Mapped[RecurringTaskReminderSchedule] = relationship(back_populates="occurrences")


class LifeCollection(TimestampMixin, Base):
    __tablename__ = "life_collections"
    __table_args__ = (
        UniqueConstraint("id", "owner_id", name="uq_life_collection_id_owner"),
        UniqueConstraint(
            "owner_id", "normalized_name", name="uq_life_collection_owner_normalized_name"
        ),
        CheckConstraint("kind IN ('topic', 'project', 'list')", name="ck_life_collection_kind"),
        CheckConstraint("status IN ('active', 'archived')", name="ck_life_collection_status"),
        CheckConstraint("version > 0", name="ck_life_collection_version"),
        CheckConstraint("length(name) BETWEEN 1 AND 100", name="ck_life_collection_name_length"),
        CheckConstraint(
            "length(normalized_name) BETWEEN 1 AND 100",
            name="ck_life_collection_normalized_name_length",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(100))
    normalized_name: Mapped[str] = mapped_column(String(100))
    starter_key: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)


class LifeCollectionAlias(Base):
    __tablename__ = "life_collection_aliases"
    __table_args__ = (
        ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_alias_owner",
        ),
        UniqueConstraint(
            "owner_id", "normalized_alias", name="uq_life_collection_alias_owner_name"
        ),
        CheckConstraint("length(alias) BETWEEN 1 AND 100", name="ck_life_collection_alias_length"),
        CheckConstraint(
            "length(normalized_alias) BETWEEN 1 AND 100",
            name="ck_life_collection_normalized_alias_length",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    collection_id: Mapped[int] = mapped_column(Integer, index=True)
    owner_id: Mapped[int] = mapped_column(Integer, index=True)
    alias: Mapped[str] = mapped_column(String(100))
    normalized_alias: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LifeCollectionLink(Base):
    __tablename__ = "life_collection_links"
    __table_args__ = (
        ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_link_collection_owner",
        ),
        ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_life_collection_link_inbox_owner",
        ),
        UniqueConstraint("collection_id", "inbox_item_id", name="uq_life_collection_link_item"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    collection_id: Mapped[int] = mapped_column(Integer, index=True)
    owner_id: Mapped[int] = mapped_column(Integer, index=True)
    inbox_item_id: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LifeCollectionPreference(TimestampMixin, Base):
    __tablename__ = "life_collection_preferences"
    __table_args__ = (CheckConstraint("version > 0", name="ck_life_collection_preference_version"),)

    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=1)


class LifeCollectionContext(TimestampMixin, Base):
    __tablename__ = "life_collection_contexts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_context_collection_owner",
        ),
        ForeignKeyConstraint(
            ["last_inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_life_collection_context_inbox_owner",
        ),
        UniqueConstraint("owner_id", "chat_id", name="uq_life_collection_context_owner_chat"),
        CheckConstraint("version > 0", name="ck_life_collection_context_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(Integer, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    collection_id: Mapped[int] = mapped_column(Integer, index=True)
    last_inbox_item_id: Mapped[int | None] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LifeCollectionActionToken(Base):
    __tablename__ = "life_collection_action_tokens"
    __table_args__ = (
        ForeignKeyConstraint(
            ["collection_id", "owner_id"],
            ["life_collections.id", "life_collections.owner_id"],
            ondelete="CASCADE",
            name="fk_life_collection_action_collection_owner",
        ),
        ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_life_collection_action_inbox_owner",
        ),
        CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_life_collection_action_status",
        ),
        CheckConstraint(
            "collection_version IS NULL OR collection_version > 0",
            name="ck_life_collection_action_version",
        ),
    )

    token: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    collection_id: Mapped[int | None] = mapped_column(Integer, index=True)
    collection_version: Mapped[int | None] = mapped_column(Integer)
    inbox_item_id: Mapped[int | None] = mapped_column(Integer, index=True)
    action: Mapped[str] = mapped_column(String(48), index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Workspace(TimestampMixin, Base):
    """A collaboration boundary, distinct from owner-only LifeCollection rows."""

    __tablename__ = "workspaces"
    __table_args__ = (
        UniqueConstraint("id", "created_by_user_id", name="uq_workspace_id_creator"),
        UniqueConstraint(
            "created_by_user_id",
            "normalized_name",
            name="uq_workspace_creator_normalized_name",
        ),
        CheckConstraint(
            "character IN ('pair', 'friends', 'family', 'team', 'custom')",
            name="ck_workspace_character",
        ),
        CheckConstraint("status IN ('active', 'archived')", name="ck_workspace_status"),
        CheckConstraint("access_epoch > 0", name="ck_workspace_access_epoch"),
        CheckConstraint("version > 0", name="ck_workspace_version"),
        CheckConstraint("length(name) BETWEEN 1 AND 100", name="ck_workspace_name_length"),
        CheckConstraint(
            "length(normalized_name) BETWEEN 1 AND 100",
            name="ck_workspace_normalized_name_length",
        ),
        CheckConstraint(
            "description IS NULL OR length(description) BETWEEN 1 AND 500",
            name="ck_workspace_description_length",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    normalized_name: Mapped[str] = mapped_column(String(100))
    character: Mapped[str] = mapped_column(String(20), index=True)
    description: Mapped[str | None] = mapped_column(String(500))
    created_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    access_epoch: Mapped[int] = mapped_column(Integer, default=1)
    version: Mapped[int] = mapped_column(Integer, default=1)


class WorkspaceMember(TimestampMixin, Base):
    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member_user"),
        CheckConstraint("role IN ('owner', 'editor', 'viewer')", name="ck_workspace_member_role"),
        CheckConstraint(
            "status IN ('active', 'revoked', 'left')",
            name="ck_workspace_member_status",
        ),
        CheckConstraint("version > 0", name="ck_workspace_member_version"),
        CheckConstraint(
            "(status = 'active' AND revoked_at IS NULL) OR "
            "(status IN ('revoked', 'left') AND revoked_at IS NOT NULL)",
            name="ck_workspace_member_revocation_time",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    role: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    invited_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, default=1)


class WorkspaceInvitation(TimestampMixin, Base):
    __tablename__ = "workspace_invitations"
    __table_args__ = (
        UniqueConstraint("id", "workspace_id", name="uq_workspace_invitation_id_workspace"),
        UniqueConstraint("token_hash", name="uq_workspace_invitation_token_hash"),
        CheckConstraint("role IN ('editor', 'viewer')", name="ck_workspace_invitation_role"),
        CheckConstraint(
            "delivery_mode IN ('direct', 'share')",
            name="ck_workspace_invitation_delivery_mode",
        ),
        CheckConstraint(
            "status IN ('pending', 'accepted', 'declined', 'revoked', 'expired')",
            name="ck_workspace_invitation_status",
        ),
        CheckConstraint("version > 0", name="ck_workspace_invitation_version"),
        CheckConstraint("length(token_hash) = 64", name="ck_workspace_invitation_hash_length"),
        CheckConstraint(
            "length(template_key) BETWEEN 1 AND 64",
            name="ck_workspace_invitation_template_length",
        ),
        CheckConstraint(
            "custom_text IS NULL OR length(custom_text) BETWEEN 1 AND 1000",
            name="ck_workspace_invitation_custom_text_length",
        ),
        CheckConstraint(
            "(delivery_mode = 'direct' AND intended_user_id IS NOT NULL) OR "
            "(delivery_mode = 'share' AND intended_user_id IS NULL)",
            name="ck_workspace_invitation_recipient",
        ),
        CheckConstraint(
            "intended_user_id IS NULL OR intended_user_id != inviter_user_id",
            name="ck_workspace_invitation_not_self",
        ),
        CheckConstraint(
            "(status = 'pending' AND consumed_at IS NULL AND revoked_at IS NULL) OR "
            "(status IN ('accepted', 'declined') AND consumed_at IS NOT NULL "
            "AND revoked_at IS NULL) OR "
            "(status = 'revoked' AND consumed_at IS NULL AND revoked_at IS NOT NULL) OR "
            "(status = 'expired' AND consumed_at IS NULL AND revoked_at IS NULL)",
            name="ck_workspace_invitation_terminal_time",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    inviter_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    intended_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    delivery_mode: Mapped[str] = mapped_column(String(20))
    template_key: Mapped[str] = mapped_column(String(64))
    custom_text: Mapped[str | None] = mapped_column(String(1000))
    token_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, default=1)


class WorkspaceProject(TimestampMixin, Base):
    """A shared-space project; never aliases LifeCollection(kind='project')."""

    __tablename__ = "workspace_projects"
    __table_args__ = (
        UniqueConstraint("id", "workspace_id", name="uq_workspace_project_id_workspace"),
        UniqueConstraint(
            "workspace_id", "normalized_name", name="uq_workspace_project_normalized_name"
        ),
        CheckConstraint("status IN ('active', 'archived')", name="ck_workspace_project_status"),
        CheckConstraint("version > 0", name="ck_workspace_project_version"),
        CheckConstraint("length(name) BETWEEN 1 AND 100", name="ck_workspace_project_name_length"),
        CheckConstraint(
            "length(normalized_name) BETWEEN 1 AND 100",
            name="ck_workspace_project_normalized_name_length",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    normalized_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)


class KnowledgeSpace(TimestampMixin, Base):
    __tablename__ = "knowledge_spaces"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_project_id", "workspace_id"],
            ["workspace_projects.id", "workspace_projects.workspace_id"],
            ondelete="CASCADE",
            name="fk_knowledge_space_project_workspace",
        ),
        CheckConstraint(
            "kind IN ('personal', 'workspace', 'project')", name="ck_knowledge_space_kind"
        ),
        CheckConstraint("status IN ('active', 'archived')", name="ck_knowledge_space_status"),
        CheckConstraint("version > 0", name="ck_knowledge_space_version"),
        CheckConstraint(
            "(kind = 'personal' AND personal_owner_user_id IS NOT NULL "
            "AND workspace_id IS NULL AND workspace_project_id IS NULL) OR "
            "(kind = 'workspace' AND personal_owner_user_id IS NULL "
            "AND workspace_id IS NOT NULL AND workspace_project_id IS NULL) OR "
            "(kind = 'project' AND personal_owner_user_id IS NULL "
            "AND workspace_id IS NOT NULL AND workspace_project_id IS NOT NULL)",
            name="ck_knowledge_space_scope",
        ),
        Index(
            "uq_knowledge_space_personal_owner",
            "personal_owner_user_id",
            unique=True,
            sqlite_where=text("kind = 'personal'"),
            postgresql_where=text("kind = 'personal'"),
        ),
        Index(
            "uq_knowledge_space_workspace",
            "workspace_id",
            unique=True,
            sqlite_where=text("kind = 'workspace'"),
            postgresql_where=text("kind = 'workspace'"),
        ),
        Index(
            "uq_knowledge_space_project",
            "workspace_project_id",
            unique=True,
            sqlite_where=text("kind = 'project'"),
            postgresql_where=text("kind = 'project'"),
        ),
        Index("uq_knowledge_space_public_id", "public_id", unique=True),
        Index("uq_knowledge_space_id_kind", "id", "kind", unique=True),
        CheckConstraint(
            "public_id IS NULL OR length(public_id) = 36",
            name="ck_knowledge_space_public_id_length",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Nullable at the database boundary so a PR #23 rollback image can still
    # insert a space against the additive PR #24 schema. New code always
    # supplies a UUID and KnowledgeService repairs legacy NULL rows on access.
    public_id: Mapped[str | None] = mapped_column(
        String(36), default=lambda: str(uuid4()), index=False
    )
    kind: Mapped[str] = mapped_column(String(20), index=True)
    personal_owner_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    workspace_id: Mapped[int | None] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    workspace_project_id: Mapped[int | None] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)


class WorkspaceContext(TimestampMixin, Base):
    __tablename__ = "workspace_contexts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_members.workspace_id", "workspace_members.user_id"],
            ondelete="CASCADE",
            name="fk_workspace_context_member",
        ),
        ForeignKeyConstraint(
            ["workspace_project_id", "workspace_id"],
            ["workspace_projects.id", "workspace_projects.workspace_id"],
            ondelete="CASCADE",
            name="fk_workspace_context_project",
        ),
        UniqueConstraint("actor_user_id", "chat_id", name="uq_workspace_context_actor_chat"),
        CheckConstraint("workspace_access_epoch > 0", name="ck_workspace_context_access_epoch"),
        CheckConstraint("version > 0", name="ck_workspace_context_version"),
        CheckConstraint(
            "(workspace_project_id IS NULL AND workspace_project_version IS NULL) OR "
            "(workspace_project_id IS NOT NULL AND workspace_project_version IS NOT NULL "
            "AND workspace_project_version > 0)",
            name="ck_workspace_context_project_version",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    workspace_id: Mapped[int] = mapped_column(Integer, index=True)
    workspace_access_epoch: Mapped[int] = mapped_column(Integer)
    workspace_project_id: Mapped[int | None] = mapped_column(Integer, index=True)
    workspace_project_version: Mapped[int | None] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class WorkspaceActionToken(Base):
    __tablename__ = "workspace_action_tokens"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_project_id", "workspace_id"],
            ["workspace_projects.id", "workspace_projects.workspace_id"],
            ondelete="CASCADE",
            name="fk_workspace_action_project",
        ),
        ForeignKeyConstraint(
            ["invitation_id", "workspace_id"],
            ["workspace_invitations.id", "workspace_invitations.workspace_id"],
            ondelete="CASCADE",
            name="fk_workspace_action_invitation",
        ),
        CheckConstraint(
            "scope_kind IN ('wizard', 'workspace', 'invitation')",
            name="ck_workspace_action_scope_kind",
        ),
        CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_workspace_action_status",
        ),
        CheckConstraint(
            "(status IN ('pending', 'awaiting_input') AND consumed_at IS NULL) OR "
            "(status = 'consumed' AND consumed_at IS NOT NULL)",
            name="ck_workspace_action_consumed_time",
        ),
        CheckConstraint("length(token_hash) = 64", name="ck_workspace_action_hash_length"),
        CheckConstraint("length(action) BETWEEN 1 AND 48", name="ck_workspace_action_length"),
        CheckConstraint(
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
        CheckConstraint(
            "(workspace_project_id IS NULL AND workspace_project_version IS NULL "
            "AND workspace_project_status_snapshot IS NULL) OR "
            "(scope_kind = 'workspace' AND workspace_project_id IS NOT NULL "
            "AND workspace_project_version IS NOT NULL AND workspace_project_version > 0 "
            "AND workspace_project_status_snapshot IN ('active', 'archived'))",
            name="ck_workspace_action_project_version",
        ),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    scope_kind: Mapped[str] = mapped_column(String(20))
    workspace_id: Mapped[int | None] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    workspace_access_epoch: Mapped[int | None] = mapped_column(Integer)
    workspace_version: Mapped[int | None] = mapped_column(Integer)
    workspace_status_snapshot: Mapped[str | None] = mapped_column(String(20))
    workspace_project_id: Mapped[int | None] = mapped_column(Integer, index=True)
    workspace_project_version: Mapped[int | None] = mapped_column(Integer)
    workspace_project_status_snapshot: Mapped[str | None] = mapped_column(String(20))
    invitation_id: Mapped[int | None] = mapped_column(Integer, index=True)
    invitation_version: Mapped[int | None] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(48), index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeSource(TimestampMixin, Base):
    """A logical, space-owned source with a mutable lifecycle pointer."""

    __tablename__ = "knowledge_sources"
    __table_args__ = (
        ForeignKeyConstraint(
            ["knowledge_space_id", "space_kind"],
            ["knowledge_spaces.id", "knowledge_spaces.kind"],
            ondelete="CASCADE",
            name="fk_knowledge_source_space_scope",
        ),
        UniqueConstraint("public_id", name="uq_knowledge_source_public_id"),
        UniqueConstraint("id", "knowledge_space_id", name="uq_knowledge_source_id_space"),
        CheckConstraint("length(public_id) = 36", name="ck_knowledge_source_public_id_length"),
        CheckConstraint(
            "source_type IN ('text', 'forward', 'document', 'image', 'url')",
            name="ck_knowledge_source_type",
        ),
        CheckConstraint(
            "processing_status IN "
            "('queued', 'processing', 'ready', 'partial', 'failed', 'quarantined', "
            "'cancelled')",
            name="ck_knowledge_source_processing_status",
        ),
        CheckConstraint(
            "lifecycle_status IN ('active', 'trashed', 'purge_pending', 'purge_failed', 'purged')",
            name="ck_knowledge_source_lifecycle_status",
        ),
        CheckConstraint(
            "knowledge_role IN "
            "('foundation', 'trusted', 'perspective', 'discussion', 'counterpoint', "
            "'hypothesis')",
            name="ck_knowledge_source_role",
        ),
        CheckConstraint(
            "priority IN ('high', 'normal', 'low')",
            name="ck_knowledge_source_priority",
        ),
        CheckConstraint(
            "publication_state IN ('draft', 'publication_ready')",
            name="ck_knowledge_source_publication_state",
        ),
        CheckConstraint(
            "system_classification IN ('general', 'health_private')",
            name="ck_knowledge_source_system_classification",
        ),
        CheckConstraint(
            "system_classification != 'health_private' OR space_kind = 'personal'",
            name="ck_knowledge_source_health_personal_only",
        ),
        CheckConstraint(
            "system_classification != 'health_private' OR publication_state = 'draft'",
            name="ck_knowledge_source_health_not_publication_ready",
        ),
        CheckConstraint("length(title) BETWEEN 1 AND 200", name="ck_knowledge_source_title"),
        CheckConstraint(
            "length(provenance_kind) BETWEEN 1 AND 40",
            name="ck_knowledge_source_provenance_kind",
        ),
        CheckConstraint(
            "user_classification IS NULL OR length(user_classification) BETWEEN 1 AND 64",
            name="ck_knowledge_source_user_classification",
        ),
        CheckConstraint("version > 0", name="ck_knowledge_source_version"),
        CheckConstraint(
            "(lifecycle_status = 'purged' AND current_revision_number IS NULL) OR "
            "(lifecycle_status != 'purged' AND current_revision_number > 0)",
            name="ck_knowledge_source_current_revision",
        ),
        CheckConstraint(
            "(lifecycle_status = 'active' AND trashed_at IS NULL "
            "AND purge_requested_at IS NULL "
            "AND purged_at IS NULL) OR "
            "(lifecycle_status = 'trashed' AND trashed_at IS NOT NULL "
            "AND purge_requested_at IS NULL "
            "AND purged_at IS NULL) OR "
            "(lifecycle_status IN ('purge_pending', 'purge_failed') "
            "AND trashed_at IS NOT NULL "
            "AND purge_requested_at IS NOT NULL AND purged_at IS NULL) OR "
            "(lifecycle_status = 'purged' AND trashed_at IS NOT NULL "
            "AND purge_requested_at IS NOT NULL "
            "AND purged_at IS NOT NULL)",
            name="ck_knowledge_source_lifecycle_times",
        ),
        Index(
            "ix_knowledge_sources_space_lifecycle_updated",
            "knowledge_space_id",
            "lifecycle_status",
            "updated_at",
        ),
        Index(
            "ix_knowledge_sources_space_processing",
            "knowledge_space_id",
            "processing_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    knowledge_space_id: Mapped[int] = mapped_column(Integer, index=True)
    space_kind: Mapped[str] = mapped_column(String(20), index=True)
    created_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    source_type: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(200))
    provenance_kind: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    processing_status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    lifecycle_status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    knowledge_role: Mapped[str] = mapped_column(String(20), default="trusted", index=True)
    priority: Mapped[str] = mapped_column(String(20), default="normal", index=True)
    publication_state: Mapped[str] = mapped_column(String(24), default="draft", index=True)
    system_classification: Mapped[str] = mapped_column(String(24), default="general", index=True)
    user_classification: Mapped[str | None] = mapped_column(String(64), index=True)
    current_revision_number: Mapped[int | None] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer, default=1)
    trashed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    trashed_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    purge_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeSourceRevision(Base):
    """Original identity plus a write-once deterministic extraction result."""

    __tablename__ = "knowledge_source_revisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_revision_source_space",
        ),
        ForeignKeyConstraint(
            ["original_revision_id", "source_id", "knowledge_space_id"],
            [
                "knowledge_source_revisions.id",
                "knowledge_source_revisions.source_id",
                "knowledge_source_revisions.knowledge_space_id",
            ],
            ondelete="CASCADE",
            name="fk_knowledge_revision_original_scope",
        ),
        UniqueConstraint("public_id", name="uq_knowledge_revision_public_id"),
        UniqueConstraint("id", "source_id", name="uq_knowledge_revision_id_source"),
        UniqueConstraint(
            "id",
            "source_id",
            "knowledge_space_id",
            name="uq_knowledge_revision_id_source_space",
        ),
        UniqueConstraint(
            "source_id", "revision_number", name="uq_knowledge_revision_source_number"
        ),
        UniqueConstraint("original_storage_key", name="uq_knowledge_revision_original_key"),
        UniqueConstraint("extracted_storage_key", name="uq_knowledge_revision_extracted_key"),
        CheckConstraint("length(public_id) = 36", name="ck_knowledge_revision_public_id_length"),
        CheckConstraint("revision_number > 0", name="ck_knowledge_revision_number"),
        CheckConstraint("length(sha256) = 64", name="ck_knowledge_revision_sha256_length"),
        CheckConstraint("size_bytes >= 0", name="ck_knowledge_revision_size"),
        CheckConstraint(
            "detected_format IN ('text', 'txt', 'markdown', 'pdf', 'docx', 'epub', 'image', 'url')",
            name="ck_knowledge_revision_format",
        ),
        CheckConstraint(
            "extraction_status IN "
            "('pending', 'ready', 'partial', 'failed', 'quarantined', 'cancelled')",
            name="ck_knowledge_revision_extraction_status",
        ),
        CheckConstraint(
            "original_storage_key IS NULL OR length(original_storage_key) BETWEEN 1 AND 512",
            name="ck_knowledge_revision_original_key_length",
        ),
        CheckConstraint(
            "(original_revision_id IS NULL AND original_storage_key IS NOT NULL) OR "
            "(original_revision_id IS NOT NULL AND original_storage_key IS NULL)",
            name="ck_knowledge_revision_original_reference",
        ),
        CheckConstraint(
            "length(safe_display_name) BETWEEN 1 AND 255",
            name="ck_knowledge_revision_display_name",
        ),
        CheckConstraint(
            "(extracted_storage_key IS NULL AND extracted_sha256 IS NULL "
            "AND extracted_size_bytes IS NULL) OR "
            "(extracted_storage_key IS NOT NULL AND extracted_sha256 IS NOT NULL "
            "AND length(extracted_sha256) = 64 AND extracted_size_bytes >= 0)",
            name="ck_knowledge_revision_extracted_tuple",
        ),
        CheckConstraint(
            "(extraction_status = 'pending' AND finalized_at IS NULL) OR "
            "(extraction_status != 'pending' AND finalized_at IS NOT NULL)",
            name="ck_knowledge_revision_finalized_time",
        ),
        Index(
            "ix_knowledge_revisions_space_sha256",
            "knowledge_space_id",
            "sha256",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    source_id: Mapped[int] = mapped_column(Integer, index=True)
    knowledge_space_id: Mapped[int] = mapped_column(Integer, index=True)
    revision_number: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    original_revision_id: Mapped[int | None] = mapped_column(Integer, index=True)
    original_storage_key: Mapped[str | None] = mapped_column(String(512))
    declared_mime: Mapped[str | None] = mapped_column(String(127))
    detected_mime: Mapped[str] = mapped_column(String(127))
    detected_format: Mapped[str] = mapped_column(String(20), index=True)
    safe_display_name: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer)
    extracted_storage_key: Mapped[str | None] = mapped_column(String(512))
    extracted_sha256: Mapped[str | None] = mapped_column(String(64))
    extracted_size_bytes: Mapped[int | None] = mapped_column(Integer)
    extraction_status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    extraction_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeIngestionJob(TimestampMixin, Base):
    __tablename__ = "knowledge_ingestion_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_job_source_space",
        ),
        ForeignKeyConstraint(
            ["revision_id", "source_id"],
            ["knowledge_source_revisions.id", "knowledge_source_revisions.source_id"],
            ondelete="CASCADE",
            name="fk_knowledge_job_revision_source",
        ),
        UniqueConstraint("public_id", name="uq_knowledge_job_public_id"),
        UniqueConstraint("idempotency_key", name="uq_knowledge_job_idempotency"),
        CheckConstraint("length(public_id) = 36", name="ck_knowledge_job_public_id_length"),
        CheckConstraint("job_type IN ('extract', 'purge')", name="ck_knowledge_job_type"),
        CheckConstraint(
            "(job_type = 'extract' AND revision_id IS NOT NULL) OR "
            "(job_type = 'purge' AND revision_id IS NULL)",
            name="ck_knowledge_job_revision_scope",
        ),
        CheckConstraint(
            "status IN "
            "('queued', 'processing', 'ready', 'partial', 'failed', 'quarantined', "
            "'cancelled')",
            name="ck_knowledge_job_status",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND max_attempts BETWEEN 1 AND 20 "
            "AND attempt_count <= max_attempts",
            name="ck_knowledge_job_attempts",
        ),
        CheckConstraint("source_version > 0", name="ck_knowledge_job_source_version"),
        CheckConstraint("version > 0", name="ck_knowledge_job_version"),
        CheckConstraint(
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
        CheckConstraint(
            "status NOT IN ('ready', 'cancelled') OR safe_error_code IS NULL",
            name="ck_knowledge_job_safe_error",
        ),
        Index(
            "ix_knowledge_jobs_poll",
            "status",
            "available_at",
            "id",
        ),
        Index(
            "ix_knowledge_jobs_stale_lease",
            "status",
            "lease_expires_at",
        ),
        Index("ix_knowledge_jobs_source_status", "source_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    knowledge_space_id: Mapped[int] = mapped_column(Integer, index=True)
    source_id: Mapped[int] = mapped_column(Integer, index=True)
    revision_id: Mapped[int | None] = mapped_column(Integer, index=True)
    requested_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    job_type: Mapped[str] = mapped_column(String(20), default="extract", index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    safe_error_code: Mapped[str | None] = mapped_column(String(64))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    pipeline_version: Mapped[str] = mapped_column(String(32), default="v1")
    source_version: Mapped[int] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeCaptureDraft(TimestampMixin, Base):
    __tablename__ = "knowledge_capture_drafts"
    __table_args__ = (
        UniqueConstraint("public_id", name="uq_knowledge_capture_public_id"),
        UniqueConstraint("id", "knowledge_space_id", name="uq_knowledge_capture_id_space"),
        ForeignKeyConstraint(
            ["confirmed_source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="RESTRICT",
            name="fk_knowledge_capture_confirmed_source_space",
        ),
        CheckConstraint("length(public_id) = 36", name="ck_knowledge_capture_public_id_length"),
        CheckConstraint(
            "capture_kind IN ('text', 'forward', 'document', 'image', 'url')",
            name="ck_knowledge_capture_kind",
        ),
        CheckConstraint(
            "status IN "
            "('collecting', 'awaiting_confirmation', 'confirming', 'confirmed', "
            "'cancelled', 'expired')",
            name="ck_knowledge_capture_status",
        ),
        CheckConstraint(
            "knowledge_role IN "
            "('foundation', 'trusted', 'perspective', 'discussion', 'counterpoint', "
            "'hypothesis')",
            name="ck_knowledge_capture_role",
        ),
        CheckConstraint(
            "priority IN ('high', 'normal', 'low')",
            name="ck_knowledge_capture_priority",
        ),
        CheckConstraint(
            "system_classification IN ('general', 'health_private')",
            name="ck_knowledge_capture_system_classification",
        ),
        CheckConstraint("version > 0", name="ck_knowledge_capture_version"),
        CheckConstraint(
            "declared_size_bytes IS NULL OR declared_size_bytes >= 0",
            name="ck_knowledge_capture_declared_size",
        ),
        CheckConstraint(
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
        CheckConstraint(
            "(status IN ('collecting', 'cancelled', 'expired')) OR "
            "(knowledge_space_id IS NOT NULL AND knowledge_space_version IS NOT NULL "
            "AND title IS NOT NULL)",
            name="ck_knowledge_capture_configured",
        ),
        CheckConstraint(
            "(status = 'confirmed' AND confirmed_source_id IS NOT NULL "
            "AND completed_at IS NOT NULL) OR "
            "(status IN ('cancelled', 'expired') AND confirmed_source_id IS NULL "
            "AND completed_at IS NOT NULL) OR "
            "(status IN ('collecting', 'awaiting_confirmation', 'confirming') "
            "AND confirmed_source_id IS NULL AND completed_at IS NULL)",
            name="ck_knowledge_capture_completion",
        ),
        Index(
            "uq_knowledge_capture_active_actor_chat",
            "actor_user_id",
            "chat_id",
            unique=True,
            sqlite_where=text("status IN ('collecting', 'awaiting_confirmation', 'confirming')"),
            postgresql_where=text(
                "status IN ('collecting', 'awaiting_confirmation', 'confirming')"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    capture_kind: Mapped[str] = mapped_column(String(20))
    text_content: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    telegram_file_id: Mapped[str | None] = mapped_column(String(512))
    telegram_file_unique_id_hash: Mapped[str | None] = mapped_column(String(64))
    telegram_message_id: Mapped[int | None] = mapped_column(BigInteger)
    declared_mime: Mapped[str | None] = mapped_column(String(127))
    safe_display_name: Mapped[str | None] = mapped_column(String(255))
    declared_size_bytes: Mapped[int | None] = mapped_column(Integer)
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    knowledge_space_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), index=True
    )
    knowledge_space_version: Mapped[int | None] = mapped_column(Integer)
    workspace_access_epoch: Mapped[int | None] = mapped_column(Integer)
    workspace_project_version: Mapped[int | None] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(String(200))
    knowledge_role: Mapped[str] = mapped_column(String(20), default="trusted")
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    system_classification: Mapped[str] = mapped_column(String(24), default="general")
    user_classification: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="collecting", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    confirmed_source_id: Mapped[int | None] = mapped_column(Integer, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeActionToken(Base):
    __tablename__ = "knowledge_action_tokens"
    __table_args__ = (
        ForeignKeyConstraint(
            ["capture_draft_id", "knowledge_space_id"],
            ["knowledge_capture_drafts.id", "knowledge_capture_drafts.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_action_capture_space",
        ),
        ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_action_source_space",
        ),
        CheckConstraint("length(token_hash) = 64", name="ck_knowledge_action_hash"),
        CheckConstraint(
            "scope_kind IN ('capture', 'source', 'space')",
            name="ck_knowledge_action_scope_kind",
        ),
        CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_knowledge_action_status",
        ),
        CheckConstraint(
            "(status IN ('pending', 'awaiting_input') AND consumed_at IS NULL) OR "
            "(status = 'consumed' AND consumed_at IS NOT NULL)",
            name="ck_knowledge_action_consumed",
        ),
        CheckConstraint(
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

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    scope_kind: Mapped[str] = mapped_column(String(20))
    knowledge_space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), index=True
    )
    knowledge_space_version: Mapped[int] = mapped_column(Integer)
    workspace_access_epoch: Mapped[int | None] = mapped_column(Integer)
    capture_draft_id: Mapped[int | None] = mapped_column(Integer, index=True)
    capture_version: Mapped[int | None] = mapped_column(Integer)
    source_id: Mapped[int | None] = mapped_column(Integer, index=True)
    source_version: Mapped[int | None] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(48), index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeQuotaReservation(TimestampMixin, Base):
    __tablename__ = "knowledge_quota_reservations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["capture_draft_id", "knowledge_space_id"],
            ["knowledge_capture_drafts.id", "knowledge_capture_drafts.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_quota_capture_space",
        ),
        ForeignKeyConstraint(
            ["source_id", "knowledge_space_id"],
            ["knowledge_sources.id", "knowledge_sources.knowledge_space_id"],
            ondelete="CASCADE",
            name="fk_knowledge_quota_source_space",
        ),
        ForeignKeyConstraint(
            ["revision_id", "source_id", "knowledge_space_id"],
            [
                "knowledge_source_revisions.id",
                "knowledge_source_revisions.source_id",
                "knowledge_source_revisions.knowledge_space_id",
            ],
            ondelete="CASCADE",
            name="fk_knowledge_quota_revision_source_space",
        ),
        UniqueConstraint("public_id", name="uq_knowledge_quota_reservation_public_id"),
        UniqueConstraint("idempotency_key", name="uq_knowledge_quota_reservation_key"),
        CheckConstraint("length(public_id) = 36", name="ck_knowledge_quota_reservation_public_id"),
        CheckConstraint(
            "status IN ('reserved', 'committed', 'released', 'expired')",
            name="ck_knowledge_quota_reservation_status",
        ),
        CheckConstraint(
            "reserved_bytes >= 0 AND reserved_sources > 0 AND reserved_jobs >= 0",
            name="ck_knowledge_quota_reservation_amounts",
        ),
        CheckConstraint(
            "(status = 'reserved' AND completed_at IS NULL AND source_id IS NULL "
            "AND revision_id IS NULL) OR "
            "(status = 'committed' AND completed_at IS NOT NULL AND source_id IS NOT NULL "
            "AND revision_id IS NOT NULL) OR "
            "(status IN ('released', 'expired') AND completed_at IS NOT NULL "
            "AND source_id IS NULL AND revision_id IS NULL)",
            name="ck_knowledge_quota_reservation_completion",
        ),
        Index(
            "ix_knowledge_quota_actor_status",
            "actor_user_id",
            "status",
        ),
        Index(
            "ix_knowledge_quota_space_status",
            "knowledge_space_id",
            "status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    knowledge_space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), index=True
    )
    capture_draft_id: Mapped[int] = mapped_column(Integer, index=True)
    reserved_bytes: Mapped[int] = mapped_column(Integer)
    reserved_sources: Mapped[int] = mapped_column(Integer, default=1)
    reserved_jobs: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="reserved", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source_id: Mapped[int | None] = mapped_column(Integer)
    revision_id: Mapped[int | None] = mapped_column(Integer)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeAuditEvent(Base):
    """Append-only, allowlisted metadata; never stores source content or raw URLs."""

    __tablename__ = "knowledge_audit_events"
    __table_args__ = (
        UniqueConstraint("public_id", name="uq_knowledge_audit_public_id"),
        CheckConstraint("length(public_id) = 36", name="ck_knowledge_audit_public_id"),
        CheckConstraint(
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
        Index(
            "ix_knowledge_audit_space_created",
            "knowledge_space_id",
            "created_at",
        ),
        Index(
            "ix_knowledge_audit_source_created",
            "source_id",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    workspace_id: Mapped[int | None] = mapped_column(
        ForeignKey("workspaces.id", ondelete="SET NULL"), index=True
    )
    knowledge_space_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="SET NULL"), index=True
    )
    capture_draft_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_capture_drafts.id", ondelete="SET NULL"), index=True
    )
    source_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_sources.id", ondelete="SET NULL"), index=True
    )
    revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_source_revisions.id", ondelete="SET NULL"), index=True
    )
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_ingestion_jobs.id", ondelete="SET NULL"), index=True
    )
    safe_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeRuntimeState(Base):
    """Singleton DB fence coordinating capture/runner with consistent backups."""

    __tablename__ = "knowledge_runtime_state"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_knowledge_runtime_singleton"),
        CheckConstraint("version > 0", name="ck_knowledge_runtime_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    maintenance_paused: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DailyCheckIn(TimestampMixin, Base):
    __tablename__ = "daily_check_ins"
    __table_args__ = (UniqueConstraint("user_id", "checkin_date", name="uq_checkin_user_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    checkin_date: Mapped[date] = mapped_column(Date)
    worked: Mapped[str | None] = mapped_column(Text)
    did_not_work: Mapped[str | None] = mapped_column(Text)
    energy: Mapped[int | None] = mapped_column(Integer)
    obstacle: Mapped[str | None] = mapped_column(Text)
    tomorrow_adjustment: Mapped[str | None] = mapped_column(Text)
    completed_actions: Mapped[list[str]] = mapped_column(JSON, default=list)
    skipped_actions: Mapped[list[str]] = mapped_column(JSON, default=list)


class HealthCheckIn(TimestampMixin, Base):
    __tablename__ = "health_check_ins"
    __table_args__ = (
        UniqueConstraint("user_id", "local_date", name="uq_health_checkin_user_date"),
        CheckConstraint("energy BETWEEN 0 AND 10", name="ck_health_energy"),
        CheckConstraint("sleep BETWEEN 0 AND 10", name="ck_health_sleep"),
        CheckConstraint("mood BETWEEN 0 AND 10", name="ck_health_mood"),
        CheckConstraint("stress BETWEEN 0 AND 10", name="ck_health_stress"),
        CheckConstraint(
            "physical_wellbeing BETWEEN 0 AND 10",
            name="ck_health_physical_wellbeing",
        ),
        CheckConstraint("state_score BETWEEN 0 AND 100", name="ck_health_state_score"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    local_date: Mapped[date] = mapped_column(Date, index=True)
    timezone: Mapped[str] = mapped_column(String(64))
    energy: Mapped[int] = mapped_column(Integer)
    sleep: Mapped[int] = mapped_column(Integer)
    mood: Mapped[int] = mapped_column(Integer)
    stress: Mapped[int] = mapped_column(Integer)
    physical_wellbeing: Mapped[int] = mapped_column(Integer)
    symptoms: Mapped[str | None] = mapped_column(Text)
    state_score: Mapped[int] = mapped_column(Integer)
    user: Mapped[User] = relationship(back_populates="health_check_ins")


class HealthReminderPreference(TimestampMixin, Base):
    __tablename__ = "health_reminder_preferences"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    telegram_user_id: Mapped[int] = mapped_column(BigInteger)
    chat_id: Mapped[int] = mapped_column(BigInteger)
    timezone: Mapped[str] = mapped_column(String(64))
    local_time: Mapped[time] = mapped_column(Time)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class DoctorVisitPrep(TimestampMixin, Base):
    __tablename__ = "doctor_visit_preps"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    timezone: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)
    duration: Mapped[str] = mapped_column(Text)
    symptoms: Mapped[str] = mapped_column(Text)
    medications: Mapped[str | None] = mapped_column(Text)
    questions: Mapped[str | None] = mapped_column(Text)
    health_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text)
    appointment_inbox_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("inbox_items.id", ondelete="SET NULL"),
        unique=True,
    )
    user: Mapped[User] = relationship(back_populates="doctor_visit_preps")
    appointment_inbox_item: Mapped[InboxItem | None] = relationship()


class OnboardingState(TimestampMixin, Base):
    __tablename__ = "onboarding_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    answers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
