from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
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
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    vision_profile: Mapped[VisionProfile | None] = relationship(
        back_populates="user", uselist=False
    )
    goals: Mapped[list[Goal]] = relationship(back_populates="user")
    routines: Mapped[list[Routine]] = relationship(back_populates="user")
    inbox_items: Mapped[list[InboxItem]] = relationship(back_populates="user")


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


class OnboardingState(TimestampMixin, Base):
    __tablename__ = "onboarding_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    answers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
