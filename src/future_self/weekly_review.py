from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Literal, cast
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import and_, case, delete, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .access import FULL_ACCESS_TIERS
from .db import Database
from .models import (
    InboxItem,
    RecurringTaskReminderSchedule,
    TaskReminder,
    TaskState,
    User,
    WeeklyFocus,
    WeeklyFocusChange,
    WeeklyReviewSession,
)

WEEKLY_REVIEW_SESSION_TTL = timedelta(minutes=30)
WEEKLY_REVIEW_MAX_CLEANUP_BATCH = 100
WEEKLY_REVIEW_MAX_TASK_SNAPSHOT = 20
WEEKLY_REVIEW_MAX_REMINDER_SNAPSHOT = 20
WEEKLY_REVIEW_MAX_FOCUS_LENGTH = 300
WEEKLY_REVIEW_MAX_APPROACH_LENGTH = 500
WEEKLY_REVIEW_MAX_SMALL_STEPS = 3
WEEKLY_REVIEW_MAX_SMALL_STEP_LENGTH = 200
WEEKLY_REVIEW_MAX_REMINDER_CANDIDATES = 5
WEEKLY_REVIEW_MAX_CANDIDATE_TITLE_LENGTH = 200
WEEKLY_REVIEW_MAX_CANDIDATE_SCHEDULE_LENGTH = 200

type WeeklyReviewSource = Literal["text", "voice"]
type WeeklyReviewAccessStatus = Literal["access_denied", "access_changed"]
type WeeklyReviewSessionStatus = Literal[
    "created",
    "updated",
    "found",
    "not_found",
    "expired",
    "stale",
    "week_changed",
    "access_denied",
    "access_changed",
]
type WeeklyFocusMutationStatus = Literal[
    "created",
    "updated",
    "duplicate",
    "deleted",
    "replay",
    "not_found",
    "expired",
    "stale",
    "week_changed",
    "access_denied",
    "access_changed",
    "focus_changed",
]

_UNSET = object()


class WeeklyReviewPhase(StrEnum):
    ROOT = "root"
    AWAITING_INPUT = "awaiting_input"
    PROCESSING = "processing"
    PREVIEW = "preview"
    SAVED = "saved"
    CANDIDATES = "candidates"
    REMINDER_HANDOFF = "reminder_handoff"
    DELETE_PREVIEW = "delete_preview"
    COMPLETED = "completed"


class WeeklyReviewValidationError(ValueError):
    """Privacy-safe invalid weekly review input."""


class WeeklyReviewStorageError(RuntimeError):
    """Privacy-safe replacement for database errors that may include parameters."""


@dataclass(frozen=True, slots=True)
class WeeklyReviewWeek:
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class WeeklyReminderCandidate:
    title: str = field(repr=False)
    schedule_wording: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class WeeklyFocusSnapshot:
    public_id: str
    week_start: date
    focus: str = field(repr=False)
    approach: str | None = field(default=None, repr=False)
    small_steps: tuple[str, ...] = field(default=(), repr=False)
    source: WeeklyReviewSource = "text"
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class WeeklyReviewSessionSnapshot:
    public_id: str
    owner_id: int
    telegram_user_id: int
    chat_id: int
    access_version: int
    week_start: date
    phase: WeeklyReviewPhase
    canonical_chat_id: int | None
    canonical_message_id: int | None
    base_focus_public_id: str | None = None
    base_focus_version: int | None = None
    focus: str | None = field(default=None, repr=False)
    approach: str | None = field(default=None, repr=False)
    small_steps: tuple[str, ...] = field(default=(), repr=False)
    reminder_candidates: tuple[WeeklyReminderCandidate, ...] = field(
        default=(),
        repr=False,
    )
    source: WeeklyReviewSource | None = None
    version: int = 1
    expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class WeeklyReviewSessionResult:
    status: WeeklyReviewSessionStatus
    session: WeeklyReviewSessionSnapshot | None = None


@dataclass(frozen=True, slots=True)
class WeeklyFocusLookup:
    status: Literal["found", "not_found", "access_denied", "access_changed"]
    focus: WeeklyFocusSnapshot | None = None


@dataclass(frozen=True, slots=True)
class WeeklyFocusMutation:
    status: WeeklyFocusMutationStatus
    focus: WeeklyFocusSnapshot | None = None
    session: WeeklyReviewSessionSnapshot | None = None
    audit_written: bool = False


@dataclass(frozen=True, slots=True)
class WeeklyReviewTaskSnapshot:
    title: str = field(repr=False)
    event_at: datetime | None = None
    requires_attention: bool = False


@dataclass(frozen=True, slots=True)
class WeeklyReviewReminderSnapshot:
    kind: Literal["one_shot", "daily"]
    title: str = field(repr=False)
    next_at: datetime | None = None
    local_time: time | None = None
    timezone: str | None = None


@dataclass(frozen=True, slots=True)
class WeeklyReviewSystemSnapshot:
    status: Literal["ready", "week_changed", "access_denied", "access_changed"]
    week_start: date
    current_focus: WeeklyFocusSnapshot | None = field(default=None, repr=False)
    completed_previous_cycle: int = 0
    active_tasks: tuple[WeeklyReviewTaskSnapshot, ...] = field(default=(), repr=False)
    reminders: tuple[WeeklyReviewReminderSnapshot, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class _Actor:
    owner_id: int
    telegram_id: int
    tier: str
    access_version: int
    timezone: str


def current_week_start(timezone_name: str, *, now: datetime | None = None) -> date:
    local_date = _local_datetime(timezone_name, now).date()
    return local_date - timedelta(days=local_date.weekday())


def target_week_start(
    timezone_name: str,
    *,
    now: datetime | None = None,
    scheduled: bool = False,
    review_weekday: int = 6,
) -> date:
    if isinstance(review_weekday, bool) or not isinstance(review_weekday, int):
        raise WeeklyReviewValidationError("Review weekday must be an integer.")
    if not 0 <= review_weekday <= 6:
        raise WeeklyReviewValidationError("Review weekday must be between 0 and 6.")
    local_date = _local_datetime(timezone_name, now).date()
    start = local_date - timedelta(days=local_date.weekday())
    if scheduled:
        start += timedelta(days=7)
    return start


def weekly_review_week(week_start: date) -> WeeklyReviewWeek:
    clean_start = _week_start(week_start)
    return WeeklyReviewWeek(start=clean_start, end=clean_start + timedelta(days=6))


def normalize_weekly_focus(value: str) -> str:
    return _normalized_text(value, label="Weekly focus", maximum=WEEKLY_REVIEW_MAX_FOCUS_LENGTH)


def normalize_weekly_approach(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WeeklyReviewValidationError("Weekly approach must be text.")
    if not value.strip():
        return None
    return _normalized_text(
        value,
        label="Weekly approach",
        maximum=WEEKLY_REVIEW_MAX_APPROACH_LENGTH,
    )


def normalize_weekly_steps(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise WeeklyReviewValidationError("Weekly steps must be a sequence.")
    if len(values) > WEEKLY_REVIEW_MAX_SMALL_STEPS:
        raise WeeklyReviewValidationError("Weekly review supports at most three steps.")
    return tuple(
        _normalized_text(
            value,
            label="Weekly step",
            maximum=WEEKLY_REVIEW_MAX_SMALL_STEP_LENGTH,
        )
        for value in values
    )


def normalize_weekly_candidates(
    values: Sequence[WeeklyReminderCandidate | Mapping[str, object]],
) -> tuple[WeeklyReminderCandidate, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise WeeklyReviewValidationError("Reminder candidates must be a sequence.")
    if len(values) > WEEKLY_REVIEW_MAX_REMINDER_CANDIDATES:
        raise WeeklyReviewValidationError("Weekly review supports at most five candidates.")
    normalized: list[WeeklyReminderCandidate] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, WeeklyReminderCandidate):
            title = value.title
            schedule_wording = value.schedule_wording
        elif isinstance(value, Mapping):
            if set(value.keys()) != {"title", "schedule_wording"}:
                raise WeeklyReviewValidationError("Reminder candidate has invalid shape.")
            title = value.get("title")
            schedule_wording = value.get("schedule_wording")
        else:
            raise WeeklyReviewValidationError("Reminder candidate has invalid shape.")
        clean_title = _normalized_text(
            title,
            label="Reminder title",
            maximum=WEEKLY_REVIEW_MAX_CANDIDATE_TITLE_LENGTH,
        )
        clean_schedule = _normalized_text(
            schedule_wording,
            label="Reminder schedule",
            maximum=WEEKLY_REVIEW_MAX_CANDIDATE_SCHEDULE_LENGTH,
        )
        key = (clean_title.casefold(), clean_schedule.casefold())
        if key in seen:
            raise WeeklyReviewValidationError("Reminder candidates must be unique.")
        seen.add(key)
        normalized.append(WeeklyReminderCandidate(clean_title, clean_schedule))
    return tuple(normalized)


class WeeklyReviewService:
    """Owner-scoped persistence for durable review state and confirmed focus."""

    def __init__(
        self,
        db: Database,
        *,
        session_ttl: timedelta = WEEKLY_REVIEW_SESSION_TTL,
        review_weekday: int = 6,
        task_snapshot_limit: int = 10,
        reminder_snapshot_limit: int = 10,
        clock: Callable[[], datetime] | None = None,
    ):
        if not timedelta(0) < session_ttl <= WEEKLY_REVIEW_SESSION_TTL:
            raise WeeklyReviewValidationError("Weekly review session TTL is invalid.")
        if isinstance(review_weekday, bool) or not isinstance(review_weekday, int):
            raise WeeklyReviewValidationError("Review weekday must be an integer.")
        if not 0 <= review_weekday <= 6:
            raise WeeklyReviewValidationError("Review weekday must be between 0 and 6.")
        self._bounded_limit(task_snapshot_limit, WEEKLY_REVIEW_MAX_TASK_SNAPSHOT, "task")
        self._bounded_limit(
            reminder_snapshot_limit,
            WEEKLY_REVIEW_MAX_REMINDER_SNAPSHOT,
            "reminder",
        )
        self.db = db
        self.session_ttl = session_ttl
        self.review_weekday = review_weekday
        self.task_snapshot_limit = task_snapshot_limit
        self.reminder_snapshot_limit = reminder_snapshot_limit
        self._clock = clock or _utc_now

    @staticmethod
    def current_week_start(timezone_name: str, *, now: datetime | None = None) -> date:
        return current_week_start(timezone_name, now=now)

    @staticmethod
    def target_week_start(
        timezone_name: str,
        *,
        now: datetime | None = None,
        scheduled: bool = False,
        review_weekday: int = 6,
    ) -> date:
        return target_week_start(
            timezone_name,
            now=now,
            scheduled=scheduled,
            review_weekday=review_weekday,
        )

    @staticmethod
    def week_range(week_start: date) -> WeeklyReviewWeek:
        return weekly_review_week(week_start)

    async def create_session(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        target_week_start: date | None = None,
        canonical_message_id: int | None = None,
        phase: WeeklyReviewPhase = WeeklyReviewPhase.ROOT,
        scheduled: bool = False,
        replace_existing: bool = True,
        now: datetime | None = None,
    ) -> WeeklyReviewSessionResult:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        destination = self._positive_id(chat_id, "chat")
        access_version = self._positive_version(expected_access_version, "access")
        clean_phase = self._phase(phase)
        canonical = self._optional_positive_id(canonical_message_id, "canonical message")
        if not isinstance(replace_existing, bool):
            raise WeeklyReviewValidationError("Weekly review replacement policy is invalid.")
        requested_week = _week_start(target_week_start) if target_week_start is not None else None
        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return WeeklyReviewSessionResult(failure)
                assert actor is not None
                existing = await session.scalar(
                    select(WeeklyReviewSession)
                    .where(
                        WeeklyReviewSession.owner_id == actor.owner_id,
                        WeeklyReviewSession.chat_id == destination,
                    )
                    .with_for_update()
                )
                current = self._current_time(now)
                current_week = target_week_start_for_actor(
                    actor,
                    now=current,
                    scheduled=False,
                    review_weekday=self.review_weekday,
                )
                default_week = target_week_start_for_actor(
                    actor,
                    now=current,
                    scheduled=scheduled,
                    review_weekday=self.review_weekday,
                )
                live_week = requested_week if requested_week is not None else default_week
                allowed_weeks = (
                    self._live_target_weeks(actor, now=current)
                    if scheduled
                    else frozenset({current_week})
                )
                if live_week not in allowed_weeks:
                    return WeeklyReviewSessionResult("week_changed")
                if existing is not None and not replace_existing:
                    if (
                        existing.access_version == access_version
                        and _as_utc(existing.expires_at) > current
                        and self._session_target_is_current(actor, existing, now=current)
                    ):
                        return WeeklyReviewSessionResult(
                            "found",
                            self._session_snapshot(existing),
                        )
                    await session.delete(existing)
                elif existing is not None:
                    await session.delete(existing)
                if existing is not None:
                    await session.flush()
                row = WeeklyReviewSession(
                    public_id=str(uuid4()),
                    owner_id=actor.owner_id,
                    telegram_user_id=actor.telegram_id,
                    chat_id=destination,
                    access_version=actor.access_version,
                    week_start=live_week,
                    phase=clean_phase.value,
                    canonical_chat_id=destination if canonical is not None else None,
                    canonical_message_id=canonical,
                    base_focus_public_id=None,
                    base_focus_version=None,
                    extracted_focus=None,
                    extracted_approach=None,
                    small_steps=[],
                    reminder_candidates=[],
                    extracted_source=None,
                    version=1,
                    created_at=current,
                    updated_at=current,
                    expires_at=current + self.session_ttl,
                )
                session.add(row)
                await session.flush()
                return WeeklyReviewSessionResult("created", self._session_snapshot(row))
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review session create failed.") from None

    async def current_session(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        now: datetime | None = None,
    ) -> WeeklyReviewSessionResult:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        destination = self._positive_id(chat_id, "chat")
        access_version = self._positive_version(expected_access_version, "access")
        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(
                    session,
                    actor_id,
                    access_version,
                )
                if failure is not None:
                    return WeeklyReviewSessionResult(failure)
                assert actor is not None
                row = await session.scalar(
                    select(WeeklyReviewSession)
                    .where(
                        WeeklyReviewSession.owner_id == actor.owner_id,
                        WeeklyReviewSession.telegram_user_id == actor.telegram_id,
                        WeeklyReviewSession.chat_id == destination,
                    )
                    .with_for_update()
                )
                if row is None:
                    return WeeklyReviewSessionResult("not_found")
                current = self._current_time(now)
                if row.access_version != access_version:
                    return WeeklyReviewSessionResult("access_changed")
                if _as_utc(row.expires_at) <= current:
                    await session.delete(row)
                    return WeeklyReviewSessionResult("expired")
                if not self._session_target_is_current(actor, row, now=current):
                    await session.delete(row)
                    return WeeklyReviewSessionResult("week_changed")
                snapshot = self._session_snapshot(row)
            if not await self._actor_generation_is_current(actor):
                return WeeklyReviewSessionResult("access_changed")
            return WeeklyReviewSessionResult("found", snapshot)
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review session read failed.") from None

    async def get_session_exact(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        canonical_message_id: int | None | object = _UNSET,
        expected_phase: WeeklyReviewPhase | None = None,
        now: datetime | None = None,
    ) -> WeeklyReviewSessionResult:
        public_id = self._public_id(session_public_id)
        session_version = self._positive_version(expected_session_version, "session")
        canonical = (
            canonical_message_id
            if canonical_message_id is _UNSET
            else self._optional_positive_id(canonical_message_id, "canonical message")
        )
        phase = self._phase(expected_phase) if expected_phase is not None else None
        result = await self.current_session(
            telegram_actor_id=telegram_actor_id,
            chat_id=chat_id,
            expected_access_version=expected_access_version,
            now=now,
        )
        if result.status != "found" or result.session is None:
            return result
        live = result.session
        if (
            live.public_id != public_id
            or live.version != session_version
            or (canonical is not _UNSET and live.canonical_message_id != canonical)
            or (phase is not None and live.phase is not phase)
        ):
            return WeeklyReviewSessionResult("stale")
        return result

    async def bind_canonical(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        canonical_message_id: int,
        now: datetime | None = None,
    ) -> WeeklyReviewSessionResult:
        return await self.transition_session(
            telegram_actor_id=telegram_actor_id,
            chat_id=chat_id,
            expected_access_version=expected_access_version,
            session_public_id=session_public_id,
            expected_session_version=expected_session_version,
            expected_canonical_message_id=None,
            canonical_message_id=canonical_message_id,
            now=now,
        )

    async def mark_processing(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        canonical_message_id: int,
        now: datetime | None = None,
    ) -> WeeklyReviewSessionResult:
        return await self.transition_session(
            telegram_actor_id=telegram_actor_id,
            chat_id=chat_id,
            expected_access_version=expected_access_version,
            session_public_id=session_public_id,
            expected_session_version=expected_session_version,
            expected_canonical_message_id=canonical_message_id,
            expected_phase=WeeklyReviewPhase.AWAITING_INPUT,
            phase=WeeklyReviewPhase.PROCESSING,
            focus=None,
            approach=None,
            small_steps=(),
            reminder_candidates=(),
            source=None,
            now=now,
        )

    async def store_extraction(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        canonical_message_id: int,
        focus: str,
        approach: str | None,
        small_steps: Sequence[str],
        reminder_candidates: Sequence[WeeklyReminderCandidate | Mapping[str, object]],
        source: WeeklyReviewSource | str,
        now: datetime | None = None,
    ) -> WeeklyReviewSessionResult:
        return await self.transition_session(
            telegram_actor_id=telegram_actor_id,
            chat_id=chat_id,
            expected_access_version=expected_access_version,
            session_public_id=session_public_id,
            expected_session_version=expected_session_version,
            expected_canonical_message_id=canonical_message_id,
            expected_phase=WeeklyReviewPhase.PROCESSING,
            phase=WeeklyReviewPhase.PREVIEW,
            focus=focus,
            approach=approach,
            small_steps=small_steps,
            reminder_candidates=reminder_candidates,
            source=source,
            now=now,
        )

    async def transition_session(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        expected_canonical_message_id: int | None | object = _UNSET,
        expected_phase: WeeklyReviewPhase | None = None,
        phase: WeeklyReviewPhase | None = None,
        canonical_message_id: int | None | object = _UNSET,
        focus: str | None | object = _UNSET,
        approach: str | None | object = _UNSET,
        small_steps: Sequence[str] | object = _UNSET,
        reminder_candidates: (
            Sequence[WeeklyReminderCandidate | Mapping[str, object]] | object
        ) = _UNSET,
        source: WeeklyReviewSource | str | None | object = _UNSET,
        now: datetime | None = None,
    ) -> WeeklyReviewSessionResult:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        destination = self._positive_id(chat_id, "chat")
        access_version = self._positive_version(expected_access_version, "access")
        public_id = self._public_id(session_public_id)
        session_version = self._positive_version(expected_session_version, "session")
        expected_canonical = (
            expected_canonical_message_id
            if expected_canonical_message_id is _UNSET
            else self._optional_positive_id(
                expected_canonical_message_id,
                "expected canonical message",
            )
        )
        expected_phase_value = self._phase(expected_phase) if expected_phase is not None else None
        target_phase = self._phase(phase) if phase is not None else None
        new_canonical = (
            canonical_message_id
            if canonical_message_id is _UNSET
            else self._optional_positive_id(canonical_message_id, "canonical message")
        )
        clean_focus = (
            focus if focus is _UNSET or focus is None else normalize_weekly_focus(cast(str, focus))
        )
        clean_approach = (
            approach
            if approach is _UNSET
            else normalize_weekly_approach(cast(str | None, approach))
        )
        clean_steps = (
            small_steps
            if small_steps is _UNSET
            else normalize_weekly_steps(cast(Sequence[str], small_steps))
        )
        clean_candidates = (
            reminder_candidates
            if reminder_candidates is _UNSET
            else normalize_weekly_candidates(
                cast(
                    Sequence[WeeklyReminderCandidate | Mapping[str, object]],
                    reminder_candidates,
                )
            )
        )
        clean_source = source if source is _UNSET or source is None else self._source(source)
        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return WeeklyReviewSessionResult(failure)
                assert actor is not None
                row = await session.scalar(
                    select(WeeklyReviewSession)
                    .where(
                        WeeklyReviewSession.owner_id == actor.owner_id,
                        WeeklyReviewSession.telegram_user_id == actor.telegram_id,
                        WeeklyReviewSession.chat_id == destination,
                    )
                    .with_for_update()
                )
                base_focus: WeeklyFocus | None | object = _UNSET
                if target_phase in {
                    WeeklyReviewPhase.PREVIEW,
                    WeeklyReviewPhase.DELETE_PREVIEW,
                }:
                    base_focus = await self._locked_focus(
                        session,
                        actor.owner_id,
                        row.week_start if row is not None else None,
                    )
                current = self._current_time(now)
                failure = self._session_failure(
                    row,
                    public_id=public_id,
                    expected_version=session_version,
                    expected_access_version=access_version,
                    expected_canonical=expected_canonical,
                    expected_phase=expected_phase_value,
                    now=current,
                )
                if failure is not None:
                    if failure == "expired" and row is not None:
                        await session.delete(row)
                    return WeeklyReviewSessionResult(failure)
                assert row is not None
                if not self._session_target_is_current(actor, row, now=current):
                    await session.delete(row)
                    return WeeklyReviewSessionResult("week_changed")
                if target_phase is not None:
                    row.phase = target_phase.value
                    if target_phase in {
                        WeeklyReviewPhase.ROOT,
                        WeeklyReviewPhase.AWAITING_INPUT,
                        WeeklyReviewPhase.PROCESSING,
                    }:
                        row.base_focus_public_id = None
                        row.base_focus_version = None
                    elif base_focus is not _UNSET:
                        locked_focus = cast(WeeklyFocus | None, base_focus)
                        row.base_focus_public_id = (
                            locked_focus.public_id if locked_focus is not None else None
                        )
                        row.base_focus_version = (
                            locked_focus.version if locked_focus is not None else None
                        )
                if new_canonical is not _UNSET:
                    row.canonical_chat_id = destination if new_canonical is not None else None
                    row.canonical_message_id = cast(int | None, new_canonical)
                if clean_focus is not _UNSET:
                    row.extracted_focus = cast(str | None, clean_focus)
                if clean_approach is not _UNSET:
                    row.extracted_approach = cast(str | None, clean_approach)
                if clean_steps is not _UNSET:
                    row.small_steps = list(cast(tuple[str, ...], clean_steps))
                if clean_candidates is not _UNSET:
                    row.reminder_candidates = [
                        {
                            "title": candidate.title,
                            "schedule_wording": candidate.schedule_wording,
                        }
                        for candidate in cast(tuple[WeeklyReminderCandidate, ...], clean_candidates)
                    ]
                if clean_source is not _UNSET:
                    row.extracted_source = cast(str | None, clean_source)
                resulting_phase = WeeklyReviewPhase(row.phase)
                if resulting_phase is WeeklyReviewPhase.PREVIEW and (
                    row.extracted_focus is None or row.extracted_source is None
                ):
                    raise WeeklyReviewValidationError("Weekly preview is incomplete.")
                row.version += 1
                row.updated_at = current
                row.expires_at = current + self.session_ttl
                await session.flush()
                return WeeklyReviewSessionResult("updated", self._session_snapshot(row))
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review session transition failed.") from None

    async def clear_session_exact(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        session_public_id: str,
        expected_session_version: int,
    ) -> bool:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        destination = self._positive_id(chat_id, "chat")
        public_id = self._public_id(session_public_id)
        version = self._positive_version(expected_session_version, "session")
        try:
            async with self.db.session() as session:
                owner_id = await session.scalar(select(User.id).where(User.telegram_id == actor_id))
                if owner_id is None:
                    return False
                removed = await session.execute(
                    delete(WeeklyReviewSession).where(
                        WeeklyReviewSession.owner_id == owner_id,
                        WeeklyReviewSession.telegram_user_id == actor_id,
                        WeeklyReviewSession.chat_id == destination,
                        WeeklyReviewSession.public_id == public_id,
                        WeeklyReviewSession.version == version,
                    )
                )
                return int(removed.rowcount or 0) == 1
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review session cleanup failed.") from None

    async def retire_access_changed_session(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        current_access_version: int,
    ) -> WeeklyReviewSessionResult:
        """Retire only an access-stale durable generation and return its frozen binding."""

        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        destination = self._positive_id(chat_id, "chat")
        observed_version = self._positive_version(current_access_version, "access")
        try:
            async with self.db.session() as session:
                # Lock the owner independent of tier so a downgrade can still
                # erase the exact stale private session generation.
                locked_id = await session.scalar(
                    update(User)
                    .where(User.telegram_id == actor_id)
                    .values(updated_at=User.updated_at)
                    .returning(User.id)
                )
                if locked_id is None:
                    return WeeklyReviewSessionResult("access_denied")
                owner = await session.scalar(
                    select(User).where(User.id == locked_id).with_for_update()
                )
                if owner is None:
                    return WeeklyReviewSessionResult("access_denied")
                row = await session.scalar(
                    select(WeeklyReviewSession)
                    .where(
                        WeeklyReviewSession.owner_id == owner.id,
                        WeeklyReviewSession.telegram_user_id == actor_id,
                        WeeklyReviewSession.chat_id == destination,
                    )
                    .with_for_update()
                )
                if row is None:
                    return WeeklyReviewSessionResult("not_found")
                snapshot = self._session_snapshot(row)
                owner_generation_changed = owner.access_version != observed_version
                session_generation_changed = (
                    owner.access_tier not in FULL_ACCESS_TIERS
                    or row.access_version != owner.access_version
                )
                if session_generation_changed:
                    await session.delete(row)
                    return WeeklyReviewSessionResult("access_changed", snapshot)
                if owner_generation_changed:
                    # A fresh allowed replacement belongs to the authoritative
                    # generation and must not be deleted by a stale caller.
                    return WeeklyReviewSessionResult("access_changed")
                return WeeklyReviewSessionResult("found", snapshot)
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review access retirement failed.") from None

    async def get_focus(
        self,
        *,
        telegram_actor_id: int,
        expected_access_version: int | None = None,
        week_start: date | None = None,
        now: datetime | None = None,
    ) -> WeeklyFocusLookup:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        access_version = (
            self._positive_version(expected_access_version, "access")
            if expected_access_version is not None
            else None
        )
        current = _as_utc(now or datetime.now(UTC))
        try:
            async with self.db.sessions() as session:
                actor, failure = await self._read_actor_for_generation(
                    session,
                    actor_id,
                    access_version,
                )
                if failure is not None:
                    return WeeklyFocusLookup(failure)
                assert actor is not None
                target = (
                    _week_start(week_start)
                    if week_start is not None
                    else current_week_start(actor.timezone, now=current)
                )
                row = await session.scalar(
                    select(WeeklyFocus).where(
                        WeeklyFocus.owner_id == actor.owner_id,
                        WeeklyFocus.week_start == target,
                    )
                )
                snapshot = self._focus_snapshot(row) if row is not None else None
            if not await self._actor_generation_is_current(actor):
                return WeeklyFocusLookup("access_changed")
            return (
                WeeklyFocusLookup("found", snapshot) if snapshot else WeeklyFocusLookup("not_found")
            )
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly focus read failed.") from None

    async def system_snapshot(
        self,
        *,
        telegram_actor_id: int,
        expected_access_version: int,
        target_week_start: date,
        now: datetime | None = None,
    ) -> WeeklyReviewSystemSnapshot:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        access_version = self._positive_version(expected_access_version, "access")
        target = _week_start(target_week_start)
        current = _as_utc(now or datetime.now(UTC))
        try:
            async with self.db.sessions() as session:
                actor, failure = await self._read_actor_for_generation(
                    session,
                    actor_id,
                    access_version,
                )
                if failure is not None:
                    return WeeklyReviewSystemSnapshot(failure, target)
                assert actor is not None
                if target not in self._live_target_weeks(actor, now=current):
                    return WeeklyReviewSystemSnapshot("week_changed", target)
                focus_row = await session.scalar(
                    select(WeeklyFocus).where(
                        WeeklyFocus.owner_id == actor.owner_id,
                        WeeklyFocus.week_start == target,
                    )
                )
                previous_start, previous_end = self._local_week_bounds_utc(
                    actor.timezone,
                    target - timedelta(days=7),
                )
                completed = int(
                    await session.scalar(
                        select(func.count(TaskState.id))
                        .join(
                            InboxItem,
                            and_(
                                InboxItem.id == TaskState.inbox_item_id,
                                InboxItem.user_id == TaskState.owner_id,
                            ),
                        )
                        .where(
                            TaskState.owner_id == actor.owner_id,
                            TaskState.status == "completed",
                            TaskState.completed_at >= previous_start,
                            TaskState.completed_at < previous_end,
                        )
                    )
                    or 0
                )
                _, target_end = self._local_week_bounds_utc(actor.timezone, target)
                attention_order = case(
                    (TaskState.event_at.is_not(None) & (TaskState.event_at < target_end), 0),
                    else_=1,
                )
                task_rows = (
                    await session.execute(
                        select(InboxItem.title, TaskState.event_at)
                        .join(
                            TaskState,
                            and_(
                                TaskState.inbox_item_id == InboxItem.id,
                                TaskState.owner_id == InboxItem.user_id,
                            ),
                        )
                        .where(
                            InboxItem.user_id == actor.owner_id,
                            InboxItem.kind == "task",
                            InboxItem.status == "confirmed",
                            TaskState.status == "active",
                        )
                        .order_by(attention_order, TaskState.event_at, InboxItem.id)
                        .limit(self.task_snapshot_limit)
                    )
                ).all()
                tasks = tuple(
                    WeeklyReviewTaskSnapshot(
                        title=row.title,
                        event_at=_optional_utc(row.event_at),
                        requires_attention=(
                            row.event_at is not None and _as_utc(row.event_at) < target_end
                        ),
                    )
                    for row in task_rows
                )
                one_shot_rows = (
                    await session.execute(
                        select(InboxItem.title, TaskReminder.remind_at, TaskReminder.timezone)
                        .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                        .join(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                        .where(
                            InboxItem.user_id == actor.owner_id,
                            InboxItem.kind == "task",
                            InboxItem.status == "confirmed",
                            TaskState.owner_id == actor.owner_id,
                            TaskState.status == "active",
                            TaskReminder.status.in_({"pending", "processing"}),
                            TaskReminder.remind_at >= current,
                            TaskReminder.remind_at < target_end,
                        )
                        .order_by(TaskReminder.remind_at, TaskReminder.id)
                        .limit(self.reminder_snapshot_limit)
                    )
                ).all()
                daily_rows = (
                    await session.execute(
                        select(
                            InboxItem.title,
                            RecurringTaskReminderSchedule.next_occurrence_at,
                            RecurringTaskReminderSchedule.local_time,
                            RecurringTaskReminderSchedule.timezone,
                        )
                        .join(
                            TaskState,
                            and_(
                                TaskState.inbox_item_id == InboxItem.id,
                                TaskState.owner_id == InboxItem.user_id,
                            ),
                        )
                        .join(
                            RecurringTaskReminderSchedule,
                            and_(
                                RecurringTaskReminderSchedule.inbox_item_id == InboxItem.id,
                                RecurringTaskReminderSchedule.owner_id == InboxItem.user_id,
                            ),
                        )
                        .where(
                            InboxItem.user_id == actor.owner_id,
                            InboxItem.kind == "task",
                            InboxItem.status == "confirmed",
                            TaskState.status == "active",
                            RecurringTaskReminderSchedule.status == "active",
                            RecurringTaskReminderSchedule.recurrence_kind == "daily",
                            RecurringTaskReminderSchedule.next_occurrence_at < target_end,
                        )
                        .order_by(
                            RecurringTaskReminderSchedule.next_occurrence_at,
                            RecurringTaskReminderSchedule.id,
                        )
                        .limit(self.reminder_snapshot_limit)
                    )
                ).all()
                reminders = [
                    WeeklyReviewReminderSnapshot(
                        "one_shot",
                        row.title,
                        next_at=_as_utc(row.remind_at),
                        timezone=row.timezone,
                    )
                    for row in one_shot_rows
                ]
                reminders.extend(
                    WeeklyReviewReminderSnapshot(
                        "daily",
                        row.title,
                        next_at=_as_utc(row.next_occurrence_at),
                        local_time=row.local_time,
                        timezone=row.timezone,
                    )
                    for row in daily_rows
                )
                reminders.sort(
                    key=lambda reminder: reminder.next_at or datetime.max.replace(tzinfo=UTC)
                )
                snapshot = WeeklyReviewSystemSnapshot(
                    "ready",
                    target,
                    current_focus=self._focus_snapshot(focus_row) if focus_row else None,
                    completed_previous_cycle=completed,
                    active_tasks=tasks,
                    reminders=tuple(reminders[: self.reminder_snapshot_limit]),
                )
            if not await self._actor_generation_is_current(actor):
                return WeeklyReviewSystemSnapshot("access_changed", target)
            return snapshot
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review snapshot failed.") from None

    async def confirm_focus(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        canonical_message_id: int,
        expected_week_start: date,
        now: datetime | None = None,
    ) -> WeeklyFocusMutation:
        return await self._confirm_focus(
            telegram_actor_id=telegram_actor_id,
            chat_id=chat_id,
            expected_access_version=expected_access_version,
            session_public_id=session_public_id,
            expected_session_version=expected_session_version,
            canonical_message_id=canonical_message_id,
            expected_week_start=expected_week_start,
            now=now,
        )

    async def _confirm_focus(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        canonical_message_id: int,
        expected_week_start: date,
        now: datetime | None,
    ) -> WeeklyFocusMutation:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        destination = self._positive_id(chat_id, "chat")
        access_version = self._positive_version(expected_access_version, "access")
        public_id = self._public_id(session_public_id)
        session_version = self._positive_version(expected_session_version, "session")
        canonical = self._positive_id(canonical_message_id, "canonical message")
        target = _week_start(expected_week_start)
        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return WeeklyFocusMutation(failure)
                assert actor is not None
                review = await self._locked_session(session, actor.owner_id, destination)
                existing = await self._locked_focus(session, actor.owner_id, target)
                current = self._current_time(now)
                replay = await self._confirm_replay(
                    review,
                    actor=actor,
                    public_id=public_id,
                    expected_version=session_version,
                    access_version=access_version,
                    canonical=canonical,
                    target=target,
                    terminal_phase=WeeklyReviewPhase.SAVED,
                    current_focus=existing,
                )
                if replay is not None:
                    if review is not None and _as_utc(review.expires_at) <= current:
                        await session.delete(review)
                        return WeeklyFocusMutation("expired")
                    if review is not None and not self._session_target_is_current(
                        actor,
                        review,
                        now=current,
                    ):
                        return WeeklyFocusMutation("week_changed")
                    return replay
                failure = self._session_failure(
                    review,
                    public_id=public_id,
                    expected_version=session_version,
                    expected_access_version=access_version,
                    expected_canonical=canonical,
                    expected_phase=WeeklyReviewPhase.PREVIEW,
                    now=current,
                )
                if failure is not None:
                    if failure == "expired" and review is not None:
                        await session.delete(review)
                    return WeeklyFocusMutation(failure)
                assert review is not None
                if review.week_start != target or not self._session_target_is_current(
                    actor,
                    review,
                    now=current,
                ):
                    return WeeklyFocusMutation("week_changed")
                if review.extracted_focus is None or review.extracted_source is None:
                    return WeeklyFocusMutation("stale")
                clean_focus = normalize_weekly_focus(review.extracted_focus)
                clean_approach = normalize_weekly_approach(review.extracted_approach)
                clean_steps = normalize_weekly_steps(review.small_steps)
                clean_source = self._source(review.extracted_source)
                if not self._focus_generation_matches(review, existing):
                    return WeeklyFocusMutation(
                        "focus_changed",
                        focus=self._focus_snapshot(existing) if existing is not None else None,
                        session=self._session_snapshot(review),
                    )
                audit_written = False
                if existing is None:
                    existing = WeeklyFocus(
                        public_id=str(uuid4()),
                        owner_id=actor.owner_id,
                        week_start=target,
                        focus=clean_focus,
                        approach=clean_approach,
                        small_steps=list(clean_steps),
                        source=clean_source,
                        version=1,
                        created_at=current,
                        updated_at=current,
                    )
                    session.add(existing)
                    await session.flush()
                    self._audit(
                        session,
                        owner_id=actor.owner_id,
                        focus_public_id=existing.public_id,
                        operation="created",
                        resulting_version=1,
                    )
                    status: WeeklyFocusMutationStatus = "created"
                    audit_written = True
                elif (
                    existing.focus == clean_focus
                    and existing.approach == clean_approach
                    and tuple(existing.small_steps) == clean_steps
                ):
                    status = "duplicate"
                else:
                    existing.focus = clean_focus
                    existing.approach = clean_approach
                    existing.small_steps = list(clean_steps)
                    existing.source = clean_source
                    existing.version += 1
                    existing.updated_at = current
                    await session.flush()
                    self._audit(
                        session,
                        owner_id=actor.owner_id,
                        focus_public_id=existing.public_id,
                        operation="updated",
                        resulting_version=existing.version,
                    )
                    status = "updated"
                    audit_written = True
                review.phase = WeeklyReviewPhase.SAVED.value
                review.base_focus_public_id = existing.public_id
                review.base_focus_version = existing.version
                review.version += 1
                review.updated_at = current
                review.expires_at = current + self.session_ttl
                await session.flush()
                return WeeklyFocusMutation(
                    status,
                    focus=self._focus_snapshot(existing),
                    session=self._session_snapshot(review),
                    audit_written=audit_written,
                )
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly focus confirmation failed.") from None

    async def confirm_delete(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_access_version: int,
        session_public_id: str,
        expected_session_version: int,
        canonical_message_id: int,
        expected_week_start: date,
        focus_public_id: str | None = None,
        expected_focus_version: int | None = None,
        now: datetime | None = None,
    ) -> WeeklyFocusMutation:
        actor_id = self._positive_id(telegram_actor_id, "Telegram actor")
        destination = self._positive_id(chat_id, "chat")
        access_version = self._positive_version(expected_access_version, "access")
        public_id = self._public_id(session_public_id)
        session_version = self._positive_version(expected_session_version, "session")
        canonical = self._positive_id(canonical_message_id, "canonical message")
        target = _week_start(expected_week_start)
        if (focus_public_id is None) != (expected_focus_version is None):
            raise WeeklyReviewValidationError("Focus generation must be a complete pair.")
        item_public_id = self._public_id(focus_public_id) if focus_public_id is not None else None
        item_version = (
            self._positive_version(expected_focus_version, "focus")
            if expected_focus_version is not None
            else None
        )
        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return WeeklyFocusMutation(failure)
                assert actor is not None
                review = await self._locked_session(session, actor.owner_id, destination)
                existing = await self._locked_focus(session, actor.owner_id, target)
                current = self._current_time(now)
                if (
                    review is not None
                    and review.public_id == public_id
                    and review.phase == WeeklyReviewPhase.COMPLETED.value
                    and review.version == session_version + 1
                    and review.canonical_message_id == canonical
                    and review.week_start == target
                ):
                    if _as_utc(review.expires_at) <= current:
                        await session.delete(review)
                        return WeeklyFocusMutation("expired")
                    if not self._session_target_is_current(actor, review, now=current):
                        return WeeklyFocusMutation("week_changed")
                    if item_public_id is not None and (
                        review.base_focus_public_id != item_public_id
                        or review.base_focus_version != item_version
                    ):
                        return WeeklyFocusMutation(
                            "focus_changed",
                            session=self._session_snapshot(review),
                        )
                    return WeeklyFocusMutation("replay", session=self._session_snapshot(review))
                failure = self._session_failure(
                    review,
                    public_id=public_id,
                    expected_version=session_version,
                    expected_access_version=access_version,
                    expected_canonical=canonical,
                    expected_phase=WeeklyReviewPhase.DELETE_PREVIEW,
                    now=current,
                )
                if failure is not None:
                    if failure == "expired" and review is not None:
                        await session.delete(review)
                    return WeeklyFocusMutation(failure)
                assert review is not None
                if review.week_start != target or not self._session_target_is_current(
                    actor,
                    review,
                    now=current,
                ):
                    return WeeklyFocusMutation("week_changed")
                if item_public_id is not None and (
                    review.base_focus_public_id != item_public_id
                    or review.base_focus_version != item_version
                ):
                    return WeeklyFocusMutation(
                        "focus_changed",
                        focus=self._focus_snapshot(existing) if existing is not None else None,
                        session=self._session_snapshot(review),
                    )
                if (
                    review.base_focus_public_id is None
                    or review.base_focus_version is None
                    or not self._focus_generation_matches(review, existing)
                ):
                    return WeeklyFocusMutation(
                        "focus_changed",
                        focus=self._focus_snapshot(existing) if existing is not None else None,
                        session=self._session_snapshot(review),
                    )
                assert existing is not None
                resulting_version = existing.version + 1
                await session.delete(existing)
                await session.flush()
                self._audit(
                    session,
                    owner_id=actor.owner_id,
                    focus_public_id=existing.public_id,
                    operation="deleted",
                    resulting_version=resulting_version,
                )
                review.phase = WeeklyReviewPhase.COMPLETED.value
                review.extracted_focus = None
                review.extracted_approach = None
                review.small_steps = []
                review.reminder_candidates = []
                review.extracted_source = None
                review.version += 1
                review.updated_at = current
                review.expires_at = current + self.session_ttl
                await session.flush()
                return WeeklyFocusMutation(
                    "deleted",
                    session=self._session_snapshot(review),
                    audit_written=True,
                )
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly focus deletion failed.") from None

    async def cleanup_expired(
        self,
        *,
        now: datetime | None = None,
        limit: int = WEEKLY_REVIEW_MAX_CLEANUP_BATCH,
        allowed_tiers: frozenset[str] | None = None,
    ) -> int:
        batch = self._bounded_limit(limit, WEEKLY_REVIEW_MAX_CLEANUP_BATCH, "cleanup")
        tiers = self._allowed_tiers(allowed_tiers)
        current = self._current_time(now)
        try:
            async with self.db.session() as session:
                ids = tuple(
                    (
                        await session.scalars(
                            select(WeeklyReviewSession.id)
                            .join(User, User.id == WeeklyReviewSession.owner_id)
                            .where(
                                WeeklyReviewSession.expires_at <= current,
                                WeeklyReviewSession.telegram_user_id == User.telegram_id,
                                User.access_tier.in_(tiers),
                            )
                            .order_by(WeeklyReviewSession.expires_at, WeeklyReviewSession.id)
                            .limit(batch)
                        )
                    ).all()
                )
                if not ids:
                    return 0
                removed = await session.execute(
                    delete(WeeklyReviewSession).where(
                        WeeklyReviewSession.id.in_(ids),
                        WeeklyReviewSession.expires_at <= current,
                    )
                )
                return int(removed.rowcount or 0)
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review expiry cleanup failed.") from None

    async def recover_processing_sessions(
        self,
        *,
        now: datetime | None = None,
        updated_before: datetime | None = None,
        limit: int = WEEKLY_REVIEW_MAX_CLEANUP_BATCH,
        allowed_tiers: frozenset[str] | None = None,
    ) -> int:
        recovered = await self.recover_processing_session_snapshots(
            now=now,
            updated_before=updated_before,
            limit=limit,
            allowed_tiers=allowed_tiers,
        )
        return len(recovered)

    async def recover_processing_session_snapshots(
        self,
        *,
        now: datetime | None = None,
        updated_before: datetime | None = None,
        limit: int = WEEKLY_REVIEW_MAX_CLEANUP_BATCH,
        allowed_tiers: frozenset[str] | None = None,
    ) -> tuple[WeeklyReviewSessionSnapshot, ...]:
        batch = self._bounded_limit(limit, WEEKLY_REVIEW_MAX_CLEANUP_BATCH, "recovery")
        tiers = self._allowed_tiers(allowed_tiers)
        current_hint = self._current_time(now)
        recovery_cutoff = _as_utc(updated_before) if updated_before is not None else None
        recovery_conditions = [
            WeeklyReviewSession.phase == WeeklyReviewPhase.PROCESSING.value,
            WeeklyReviewSession.expires_at > current_hint,
        ]
        if recovery_cutoff is not None:
            recovery_conditions.append(WeeklyReviewSession.updated_at <= recovery_cutoff)
        try:
            async with self.db.sessions() as session:
                candidates = tuple(
                    (
                        await session.execute(
                            select(
                                WeeklyReviewSession.id,
                                WeeklyReviewSession.public_id,
                                WeeklyReviewSession.owner_id,
                                WeeklyReviewSession.telegram_user_id,
                                WeeklyReviewSession.chat_id,
                                WeeklyReviewSession.access_version,
                            )
                            .join(User, User.id == WeeklyReviewSession.owner_id)
                            .where(*recovery_conditions)
                            .where(
                                WeeklyReviewSession.telegram_user_id == User.telegram_id,
                                WeeklyReviewSession.access_version == User.access_version,
                                User.access_tier.in_(tiers),
                            )
                            .order_by(WeeklyReviewSession.updated_at, WeeklyReviewSession.id)
                            .limit(batch)
                        )
                    ).all()
                )
            recovered: list[WeeklyReviewSessionSnapshot] = []
            for candidate in candidates:
                async with self.db.session() as session:
                    actor, failure = await self._lock_actor(
                        session,
                        candidate.telegram_user_id,
                        candidate.access_version,
                    )
                    if (
                        failure is not None
                        or actor is None
                        or actor.owner_id != candidate.owner_id
                        or actor.tier not in tiers
                    ):
                        continue
                    row = await session.scalar(
                        select(WeeklyReviewSession)
                        .where(
                            WeeklyReviewSession.id == candidate.id,
                            WeeklyReviewSession.public_id == candidate.public_id,
                            WeeklyReviewSession.owner_id == actor.owner_id,
                            WeeklyReviewSession.telegram_user_id == actor.telegram_id,
                            WeeklyReviewSession.chat_id == candidate.chat_id,
                            WeeklyReviewSession.access_version == actor.access_version,
                        )
                        .with_for_update()
                    )
                    if row is None or row.phase != WeeklyReviewPhase.PROCESSING.value:
                        continue
                    if recovery_cutoff is not None and _as_utc(row.updated_at) > recovery_cutoff:
                        continue
                    current = self._current_time(now)
                    if _as_utc(row.expires_at) <= current:
                        await session.delete(row)
                        continue
                    if not self._session_target_is_current(actor, row, now=current):
                        await session.delete(row)
                        continue
                    row.phase = WeeklyReviewPhase.AWAITING_INPUT.value
                    row.base_focus_public_id = None
                    row.base_focus_version = None
                    row.extracted_focus = None
                    row.extracted_approach = None
                    row.small_steps = []
                    row.reminder_candidates = []
                    row.extracted_source = None
                    row.version += 1
                    row.updated_at = current
                    row.expires_at = current + self.session_ttl
                    await session.flush()
                    recovered.append(self._session_snapshot(row))
            return tuple(recovered)
        except SQLAlchemyError:
            raise WeeklyReviewStorageError("Weekly review processing recovery failed.") from None

    async def _confirm_replay(
        self,
        review: WeeklyReviewSession | None,
        *,
        actor: _Actor,
        public_id: str,
        expected_version: int,
        access_version: int,
        canonical: int,
        target: date,
        terminal_phase: WeeklyReviewPhase,
        current_focus: WeeklyFocus | None,
    ) -> WeeklyFocusMutation | None:
        if (
            review is None
            or review.owner_id != actor.owner_id
            or review.telegram_user_id != actor.telegram_id
            or review.public_id != public_id
            or review.access_version != access_version
            or review.phase != terminal_phase.value
            or review.version != expected_version + 1
            or review.canonical_message_id != canonical
            or review.week_start != target
        ):
            return None
        return WeeklyFocusMutation(
            "replay",
            focus=(self._focus_snapshot(current_focus) if current_focus is not None else None),
            session=self._session_snapshot(review),
        )

    @staticmethod
    async def _locked_session(
        session: AsyncSession,
        owner_id: int,
        chat_id: int,
    ) -> WeeklyReviewSession | None:
        return await session.scalar(
            select(WeeklyReviewSession)
            .where(
                WeeklyReviewSession.owner_id == owner_id,
                WeeklyReviewSession.chat_id == chat_id,
            )
            .with_for_update()
        )

    @staticmethod
    async def _locked_focus(
        session: AsyncSession,
        owner_id: int,
        week_start: date | None,
    ) -> WeeklyFocus | None:
        if week_start is None:
            return None
        return await session.scalar(
            select(WeeklyFocus)
            .where(
                WeeklyFocus.owner_id == owner_id,
                WeeklyFocus.week_start == week_start,
            )
            .with_for_update()
        )

    @staticmethod
    def _focus_generation_matches(
        review: WeeklyReviewSession,
        focus: WeeklyFocus | None,
    ) -> bool:
        if review.base_focus_public_id is None or review.base_focus_version is None:
            return focus is None
        return (
            focus is not None
            and focus.public_id == review.base_focus_public_id
            and focus.version == review.base_focus_version
        )

    def _live_target_weeks(self, actor: _Actor, *, now: datetime) -> frozenset[date]:
        return frozenset(
            {
                target_week_start_for_actor(
                    actor,
                    now=now,
                    scheduled=False,
                    review_weekday=self.review_weekday,
                ),
                target_week_start_for_actor(
                    actor,
                    now=now,
                    scheduled=True,
                    review_weekday=self.review_weekday,
                ),
            }
        )

    def _session_target_is_current(
        self,
        actor: _Actor,
        row: WeeklyReviewSession,
        *,
        now: datetime,
    ) -> bool:
        """Validate the target policy at creation while preserving its exact week."""

        created_at = _as_utc(row.created_at)
        return row.week_start in self._live_target_weeks(
            actor, now=created_at
        ) and row.week_start in self._live_target_weeks(actor, now=now)

    @staticmethod
    def _session_failure(
        row: WeeklyReviewSession | None,
        *,
        public_id: str,
        expected_version: int,
        expected_access_version: int,
        expected_canonical: int | None | object,
        expected_phase: WeeklyReviewPhase | None,
        now: datetime,
    ) -> Literal["not_found", "stale", "expired", "access_changed"] | None:
        if row is None:
            return "not_found"
        if row.access_version != expected_access_version:
            return "access_changed"
        if _as_utc(row.expires_at) <= now:
            return "expired"
        if row.public_id != public_id or row.version != expected_version:
            return "stale"
        if expected_canonical is not _UNSET and row.canonical_message_id != expected_canonical:
            return "stale"
        if expected_phase is not None and row.phase != expected_phase.value:
            return "stale"
        return None

    @staticmethod
    async def _read_actor_for_generation(
        session: AsyncSession,
        telegram_actor_id: int,
        expected_access_version: int | None,
    ) -> tuple[_Actor | None, WeeklyReviewAccessStatus | None]:
        row = (
            await session.execute(
                select(
                    User.id,
                    User.telegram_id,
                    User.access_tier,
                    User.access_version,
                    User.timezone,
                ).where(User.telegram_id == telegram_actor_id)
            )
        ).one_or_none()
        if row is None or row.access_tier not in FULL_ACCESS_TIERS:
            return None, "access_denied"
        if expected_access_version is not None and row.access_version != expected_access_version:
            return None, "access_changed"
        return (
            _Actor(
                owner_id=row.id,
                telegram_id=row.telegram_id,
                tier=row.access_tier,
                access_version=row.access_version,
                timezone=row.timezone,
            ),
            None,
        )

    @classmethod
    async def _lock_actor(
        cls,
        session: AsyncSession,
        telegram_actor_id: int,
        expected_access_version: int,
    ) -> tuple[_Actor | None, WeeklyReviewAccessStatus | None]:
        locked_id = await session.scalar(
            update(User)
            .where(
                User.telegram_id == telegram_actor_id,
                User.access_tier.in_(FULL_ACCESS_TIERS),
                User.access_version == expected_access_version,
            )
            .values(updated_at=User.updated_at)
            .returning(User.id)
        )
        if locked_id is None:
            return await cls._read_actor_for_generation(
                session,
                telegram_actor_id,
                expected_access_version,
            )
        owner = await session.scalar(select(User).where(User.id == locked_id).with_for_update())
        if owner is None or owner.access_tier not in FULL_ACCESS_TIERS:
            return None, "access_denied"
        if owner.access_version != expected_access_version:
            return None, "access_changed"
        return (
            _Actor(
                owner_id=owner.id,
                telegram_id=owner.telegram_id,
                tier=owner.access_tier,
                access_version=owner.access_version,
                timezone=owner.timezone,
            ),
            None,
        )

    async def _actor_generation_is_current(self, actor: _Actor) -> bool:
        async with self.db.sessions() as session:
            current = await session.scalar(
                select(User.id).where(
                    User.id == actor.owner_id,
                    User.telegram_id == actor.telegram_id,
                    User.access_tier == actor.tier,
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                    User.access_version == actor.access_version,
                )
            )
        return current == actor.owner_id

    @staticmethod
    def _audit(
        session: AsyncSession,
        *,
        owner_id: int,
        focus_public_id: str,
        operation: Literal["created", "updated", "deleted"],
        resulting_version: int,
    ) -> None:
        session.add(
            WeeklyFocusChange(
                owner_id=owner_id,
                focus_public_id=focus_public_id,
                operation=operation,
                resulting_version=resulting_version,
            )
        )

    @staticmethod
    def _focus_snapshot(row: WeeklyFocus) -> WeeklyFocusSnapshot:
        return WeeklyFocusSnapshot(
            public_id=row.public_id,
            week_start=row.week_start,
            focus=row.focus,
            approach=row.approach,
            small_steps=tuple(row.small_steps),
            source=cast(WeeklyReviewSource, row.source),
            version=row.version,
            created_at=_optional_utc(row.created_at),
            updated_at=_optional_utc(row.updated_at),
        )

    @staticmethod
    def _session_snapshot(row: WeeklyReviewSession) -> WeeklyReviewSessionSnapshot:
        candidates = normalize_weekly_candidates(row.reminder_candidates)
        source = cast(WeeklyReviewSource | None, row.extracted_source)
        if source is not None and source not in {"text", "voice"}:
            raise WeeklyReviewStorageError("Weekly review session source is invalid.")
        return WeeklyReviewSessionSnapshot(
            public_id=row.public_id,
            owner_id=row.owner_id,
            telegram_user_id=row.telegram_user_id,
            chat_id=row.chat_id,
            access_version=row.access_version,
            week_start=row.week_start,
            phase=WeeklyReviewPhase(row.phase),
            canonical_chat_id=row.canonical_chat_id,
            canonical_message_id=row.canonical_message_id,
            base_focus_public_id=row.base_focus_public_id,
            base_focus_version=row.base_focus_version,
            focus=row.extracted_focus,
            approach=row.extracted_approach,
            small_steps=tuple(row.small_steps),
            reminder_candidates=candidates,
            source=source,
            version=row.version,
            expires_at=_optional_utc(row.expires_at),
            created_at=_optional_utc(row.created_at),
            updated_at=_optional_utc(row.updated_at),
        )

    @staticmethod
    def _local_week_bounds_utc(timezone_name: str, week_start: date) -> tuple[datetime, datetime]:
        timezone = _timezone(timezone_name)
        start = datetime.combine(week_start, time.min, tzinfo=timezone).astimezone(UTC)
        end = datetime.combine(
            week_start + timedelta(days=7),
            time.min,
            tzinfo=timezone,
        ).astimezone(UTC)
        return start, end

    @staticmethod
    def _phase(value: WeeklyReviewPhase | str) -> WeeklyReviewPhase:
        try:
            return WeeklyReviewPhase(value)
        except (TypeError, ValueError):
            raise WeeklyReviewValidationError("Unsupported weekly review phase.") from None

    @staticmethod
    def _source(value: WeeklyReviewSource | str) -> WeeklyReviewSource:
        if value not in {"text", "voice"}:
            raise WeeklyReviewValidationError("Unsupported weekly review source.")
        return cast(WeeklyReviewSource, value)

    @staticmethod
    def _public_id(value: str) -> str:
        if not isinstance(value, str):
            raise WeeklyReviewValidationError("Weekly review public id is invalid.")
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError, TypeError):
            raise WeeklyReviewValidationError("Weekly review public id is invalid.") from None
        if parsed.version != 4 or str(parsed) != value:
            raise WeeklyReviewValidationError("Weekly review public id is invalid.")
        return value

    @staticmethod
    def _positive_id(value: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WeeklyReviewValidationError(f"{label} id must be positive.")
        return value

    @classmethod
    def _optional_positive_id(cls, value: object, label: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int):
            raise WeeklyReviewValidationError(f"{label} id must be positive.")
        return cls._positive_id(value, label)

    @staticmethod
    def _positive_version(value: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WeeklyReviewValidationError(f"{label} version must be positive.")
        return value

    def _current_time(self, explicit: datetime | None) -> datetime:
        return _as_utc(explicit) if explicit is not None else _as_utc(self._clock())

    @staticmethod
    def _allowed_tiers(value: frozenset[str] | None) -> frozenset[str]:
        if value is None:
            return frozenset(FULL_ACCESS_TIERS)
        if not isinstance(value, frozenset) or not value or not value <= FULL_ACCESS_TIERS:
            raise WeeklyReviewValidationError("Weekly review allowed tiers are invalid.")
        return value

    @staticmethod
    def _bounded_limit(value: int, maximum: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise WeeklyReviewValidationError(f"Weekly review {label} limit is invalid.")
        return value


def target_week_start_for_actor(
    actor: _Actor,
    *,
    now: datetime,
    scheduled: bool,
    review_weekday: int,
) -> date:
    return target_week_start(
        actor.timezone,
        now=now,
        scheduled=scheduled,
        review_weekday=review_weekday,
    )


def _week_start(value: date | None) -> date:
    if isinstance(value, datetime) or not isinstance(value, date) or value.weekday() != 0:
        raise WeeklyReviewValidationError("Weekly review week must start on Monday.")
    return value


def _timezone(value: str) -> ZoneInfo:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise WeeklyReviewValidationError("User timezone is invalid.")
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        raise WeeklyReviewValidationError("User timezone is invalid.") from None


def _local_datetime(timezone_name: str, now: datetime | None) -> datetime:
    current = _as_utc(now or datetime.now(UTC))
    return current.astimezone(_timezone(timezone_name))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise WeeklyReviewValidationError("Weekly review timestamp is invalid.")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _optional_utc(value: datetime | None) -> datetime | None:
    return _as_utc(value) if value is not None else None


def _normalized_text(value: object, *, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise WeeklyReviewValidationError(f"{label} must be text.")
    normalized = unicodedata.normalize("NFKC", value)
    if any(
        unicodedata.category(character).startswith("C") and character not in {"\t", "\n", "\r"}
        for character in normalized
    ):
        raise WeeklyReviewValidationError(f"{label} contains unsupported characters.")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not 1 <= len(normalized) <= maximum:
        raise WeeklyReviewValidationError(f"{label} length is invalid.")
    return normalized


__all__ = [
    "WEEKLY_REVIEW_MAX_CLEANUP_BATCH",
    "WEEKLY_REVIEW_MAX_REMINDER_CANDIDATES",
    "WEEKLY_REVIEW_MAX_SMALL_STEPS",
    "WEEKLY_REVIEW_SESSION_TTL",
    "WeeklyFocusLookup",
    "WeeklyFocusMutation",
    "WeeklyFocusSnapshot",
    "WeeklyReminderCandidate",
    "WeeklyReviewPhase",
    "WeeklyReviewService",
    "WeeklyReviewSessionResult",
    "WeeklyReviewSessionSnapshot",
    "WeeklyReviewSource",
    "WeeklyReviewStorageError",
    "WeeklyReviewSystemSnapshot",
    "WeeklyReviewTaskSnapshot",
    "WeeklyReviewReminderSnapshot",
    "WeeklyReviewValidationError",
    "WeeklyReviewWeek",
    "current_week_start",
    "normalize_weekly_approach",
    "normalize_weekly_candidates",
    "normalize_weekly_focus",
    "normalize_weekly_steps",
    "target_week_start",
    "weekly_review_week",
]
