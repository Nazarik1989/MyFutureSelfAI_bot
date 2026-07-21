from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

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
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64), default="Europe/Moscow")
    location_city: Mapped[str | None] = mapped_column(String(120))
    location_fallback_city: Mapped[str | None] = mapped_column(String(120))
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
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
    lab_documents: Mapped[list[LabDocument]] = relationship(back_populates="owner")
    task_states: Mapped[list[TaskState]] = relationship(
        back_populates="owner", overlaps="inbox_item,task_state"
    )


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
    __table_args__ = (UniqueConstraint("id", "user_id", name="uq_inbox_item_id_user"),)

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
    user: Mapped[User] = relationship(back_populates="inbox_items")
    reminder: Mapped[TaskReminder | None] = relationship(
        back_populates="inbox_item", uselist=False, cascade="all, delete-orphan"
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
