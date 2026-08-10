from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import and_, exists, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .access import FULL_ACCESS_TIERS, is_full_access_tier
from .db import Database
from .domain import canonical_timezone
from .models import (
    InboxItem,
    RecurringTaskReminderOccurrence,
    RecurringTaskReminderSchedule,
    TaskState,
    User,
)
from .reminder_intent import (
    DailyOccurrence,
    calculate_daily_occurrence,
    format_schedule_time,
)

logger = logging.getLogger(__name__)

ScheduleStatus = Literal["active", "disabled", "completed"]
TimezoneSource = Literal["profile", "explicit"]
DeliveryReadiness = Literal[
    "ready",
    "access_denied",
    "access_changed",
    "task_changed",
    "destination_changed",
    "stale",
]


class RecurringReminderError(RuntimeError):
    """Base class for safe recurring-reminder domain failures."""


class RecurringTaskNotEligible(RecurringReminderError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RecurringScheduleConflict(RecurringReminderError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RecurringFenceLost(RecurringReminderError):
    pass


@dataclass(frozen=True, slots=True)
class RecurringScheduleSnapshot:
    id: int
    owner_id: int
    inbox_item_id: int
    recurrence_kind: Literal["daily"]
    local_time: time
    timezone: str
    timezone_source: TimezoneSource
    start_local_date: date
    next_occurrence_at: datetime
    status: ScheduleStatus
    version: int


@dataclass(frozen=True, slots=True)
class RecurringScheduleMutation:
    schedule: RecurringScheduleSnapshot | None
    changed: bool


@dataclass(frozen=True, slots=True)
class ClaimedRecurringOccurrence:
    id: int
    schedule_id: int
    schedule_version: int
    owner_id: int
    inbox_item_id: int
    inbox_item_version: int
    task_version: int
    access_version: int
    claim_token: str
    delivery_key: str
    scheduled_for: datetime
    local_date: date
    destination_id: int
    title: str
    local_time: time
    timezone: str
    attempt_count: int


@dataclass(frozen=True, slots=True)
class RecurringReminderDelivery:
    occurrence_id: int
    schedule_id: int
    delivery_key: str
    destination_id: int
    title: str
    scheduled_for: datetime
    local_date: date
    local_time: time
    timezone: str


RecurringReminderSendCallback = Callable[[RecurringReminderDelivery], Awaitable[int | None]]
RecurringReminderDeleteCallback = Callable[[int, int], Awaitable[None]]


def as_utc(value: datetime) -> datetime:
    """Normalize SQLite's naive UTC values and PostgreSQL's aware values."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _clean_local_time(value: time) -> time:
    if not isinstance(value, time) or value.tzinfo is not None:
        raise ValueError("local_time must be a naive time")
    if value.second or value.microsecond:
        raise ValueError("local_time must have minute precision")
    return value.replace(second=0, microsecond=0)


def _clean_timezone_source(value: str) -> TimezoneSource:
    if value not in {"profile", "explicit"}:
        raise ValueError("timezone_source must be profile or explicit")
    return value  # type: ignore[return-value]


def calculate_next_daily_occurrence(
    local_time: time,
    timezone: str,
    *,
    now: datetime | None = None,
) -> DailyOccurrence:
    """Return today if its wall time is still ahead, otherwise the next valid day."""

    clean_time = _clean_local_time(local_time)
    timezone_name = canonical_timezone(timezone)
    zone = ZoneInfo(timezone_name)
    current = as_utc(now or datetime.now(UTC))
    local_today = current.astimezone(zone).date()
    occurrence = calculate_daily_occurrence(local_today, clean_time, timezone_name)
    if occurrence.scheduled_for > current:
        return occurrence
    return calculate_daily_occurrence(
        occurrence.local_date + timedelta(days=1),
        clean_time,
        timezone_name,
    )


def _next_daily_after(
    local_date: date,
    local_time: time,
    timezone: str,
) -> DailyOccurrence:
    return calculate_daily_occurrence(
        local_date + timedelta(days=1),
        _clean_local_time(local_time),
        canonical_timezone(timezone),
    )


def format_recurring_time_for_profile(
    local_time: time,
    schedule_timezone: str,
    profile_timezone: str,
    *,
    occurrence_date: date,
) -> str:
    """Compatibility name for the shared pure presentation helper."""

    return format_schedule_time(
        _clean_local_time(local_time),
        schedule_timezone,
        profile_timezone,
        occurrence_date,
    )


class RecurringTaskReminderService:
    """Owner-fenced daily schedules and durable per-occurrence delivery state."""

    def __init__(
        self,
        db: Database,
        *,
        grace_minutes: int = 120,
        lease_seconds: int = 120,
        batch_size: int = 20,
    ):
        if isinstance(grace_minutes, bool) or not 5 <= grace_minutes <= 360:
            raise ValueError("grace_minutes must be between 5 and 360")
        if isinstance(lease_seconds, bool) or not 1 <= lease_seconds <= 3_600:
            raise ValueError("lease_seconds must be between 1 and 3600")
        if isinstance(batch_size, bool) or not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100")
        self.db = db
        self.grace = timedelta(minutes=grace_minutes)
        self.lease = timedelta(seconds=lease_seconds)
        self.batch_size = batch_size

    @staticmethod
    def calculate_next_occurrence(
        local_time: time,
        timezone: str,
        *,
        now: datetime | None = None,
    ) -> DailyOccurrence:
        return calculate_next_daily_occurrence(local_time, timezone, now=now)

    async def create_daily(
        self,
        owner_id: int,
        inbox_item_id: int,
        local_time: time,
        *,
        timezone: str | None = None,
        timezone_source: str = "profile",
        now: datetime | None = None,
    ) -> RecurringScheduleMutation:
        owner_id = self._positive_id(owner_id, "owner_id")
        inbox_item_id = self._positive_id(inbox_item_id, "inbox_item_id")
        clean_time = _clean_local_time(local_time)
        source = _clean_timezone_source(timezone_source)
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            return await self.create_daily_in_session(
                session,
                owner_id,
                inbox_item_id,
                clean_time,
                timezone=timezone,
                timezone_source=source,
                now=current,
            )

    async def create_daily_in_session(
        self,
        session: AsyncSession,
        owner_id: int,
        inbox_item_id: int,
        local_time: time,
        *,
        timezone: str | None = None,
        timezone_source: str = "profile",
        now: datetime | None = None,
    ) -> RecurringScheduleMutation:
        """Create a daily schedule inside a caller-owned transaction.

        The caller must already hold this owner's cross-dialect row lock. The
        primitive never commits or reacquires that lock, allowing task
        confirmation and schedule creation to remain one atomic operation.
        """

        owner_id = self._positive_id(owner_id, "owner_id")
        inbox_item_id = self._positive_id(inbox_item_id, "inbox_item_id")
        clean_time = _clean_local_time(local_time)
        source = _clean_timezone_source(timezone_source)
        current = as_utc(now or datetime.now(UTC))
        owner = await session.get(User, owner_id)
        if owner is None:
            raise RecurringTaskNotEligible("owner_not_found")
        item, _state = await self._require_live_task(session, owner_id, inbox_item_id)
        timezone_name = self._schedule_timezone(owner, timezone, source)
        existing = await session.scalar(
            select(RecurringTaskReminderSchedule)
            .where(
                RecurringTaskReminderSchedule.inbox_item_id == item.id,
                RecurringTaskReminderSchedule.owner_id == owner_id,
            )
            .with_for_update()
        )
        if existing is not None:
            exact = (
                existing.recurrence_kind == "daily"
                and existing.local_time == clean_time
                and existing.timezone == timezone_name
                and existing.timezone_source == source
            )
            if not exact:
                raise RecurringScheduleConflict("schedule_already_exists")
            return RecurringScheduleMutation(self._snapshot(existing), changed=False)

        first = calculate_next_daily_occurrence(clean_time, timezone_name, now=current)
        schedule = RecurringTaskReminderSchedule(
            owner_id=owner_id,
            inbox_item_id=item.id,
            recurrence_kind="daily",
            local_time=clean_time,
            timezone=timezone_name,
            timezone_source=source,
            start_local_date=first.local_date,
            next_occurrence_at=first.scheduled_for,
            status="active",
            version=1,
        )
        session.add(schedule)
        await session.flush()
        return RecurringScheduleMutation(self._snapshot(schedule), changed=True)

    async def get(
        self,
        owner_id: int,
        inbox_item_id: int,
    ) -> RecurringScheduleSnapshot | None:
        owner_id = self._positive_id(owner_id, "owner_id")
        inbox_item_id = self._positive_id(inbox_item_id, "inbox_item_id")
        async with self.db.sessions() as session:
            schedule = await session.scalar(
                select(RecurringTaskReminderSchedule).where(
                    RecurringTaskReminderSchedule.owner_id == owner_id,
                    RecurringTaskReminderSchedule.inbox_item_id == inbox_item_id,
                )
            )
        return self._snapshot(schedule) if schedule is not None else None

    async def status(
        self,
        owner_id: int,
        inbox_item_id: int,
    ) -> RecurringScheduleSnapshot | None:
        return await self.get(owner_id, inbox_item_id)

    async def update_time(
        self,
        owner_id: int,
        inbox_item_id: int,
        local_time: time,
        *,
        timezone: str | None = None,
        timezone_source: str | None = None,
        expected_version: int | None = None,
        expected_access_version: int | None = None,
        now: datetime | None = None,
    ) -> RecurringScheduleMutation:
        owner_id = self._positive_id(owner_id, "owner_id")
        inbox_item_id = self._positive_id(inbox_item_id, "inbox_item_id")
        expected_version = self._optional_version(expected_version)
        expected_access_version = self._optional_version(expected_access_version)
        clean_time = _clean_local_time(local_time)
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            owner = await self._lock_owner(session, owner_id)
            self._require_access_generation(owner, expected_access_version)
            await self._require_live_task(session, owner_id, inbox_item_id)
            schedule = await self._schedule_for_update(session, owner_id, inbox_item_id)
            self._require_schedule_version(schedule, expected_version)
            source = _clean_timezone_source(timezone_source or schedule.timezone_source)
            timezone_name = self._schedule_timezone(
                owner,
                timezone if timezone is not None else schedule.timezone,
                source,
            )
            if schedule.status == "completed":
                raise RecurringScheduleConflict("completed_schedule_requires_reenable")
            if (
                schedule.local_time == clean_time
                and schedule.timezone == timezone_name
                and schedule.timezone_source == source
            ):
                return RecurringScheduleMutation(self._snapshot(schedule), changed=False)

            schedule.version += 1
            schedule.local_time = clean_time
            schedule.timezone = timezone_name
            schedule.timezone_source = source
            await self._cancel_live_occurrences(session, schedule.id)
            first = await self._prepare_generation_start(session, schedule, current)
            schedule.start_local_date = first.local_date
            schedule.next_occurrence_at = first.scheduled_for
            await session.flush()
            return RecurringScheduleMutation(self._snapshot(schedule), changed=True)

    async def disable(
        self,
        owner_id: int,
        inbox_item_id: int,
        *,
        expected_version: int | None = None,
        expected_access_version: int | None = None,
    ) -> RecurringScheduleMutation:
        owner_id = self._positive_id(owner_id, "owner_id")
        inbox_item_id = self._positive_id(inbox_item_id, "inbox_item_id")
        expected_version = self._optional_version(expected_version)
        expected_access_version = self._optional_version(expected_access_version)
        async with self.db.session() as session:
            owner = await self._lock_owner(session, owner_id)
            self._require_access_generation(owner, expected_access_version)
            schedule = await self._schedule_for_update(session, owner_id, inbox_item_id)
            self._require_schedule_version(schedule, expected_version)
            if schedule.status == "disabled":
                return RecurringScheduleMutation(self._snapshot(schedule), changed=False)
            if schedule.status == "completed":
                raise RecurringScheduleConflict("schedule_is_completed")
            schedule.status = "disabled"
            schedule.version += 1
            await self._cancel_live_occurrences(session, schedule.id)
            await session.flush()
            return RecurringScheduleMutation(self._snapshot(schedule), changed=True)

    async def reenable(
        self,
        owner_id: int,
        inbox_item_id: int,
        *,
        expected_version: int | None = None,
        expected_access_version: int | None = None,
        now: datetime | None = None,
    ) -> RecurringScheduleMutation:
        owner_id = self._positive_id(owner_id, "owner_id")
        inbox_item_id = self._positive_id(inbox_item_id, "inbox_item_id")
        expected_version = self._optional_version(expected_version)
        expected_access_version = self._optional_version(expected_access_version)
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            owner = await self._lock_owner(session, owner_id)
            self._require_access_generation(owner, expected_access_version)
            await self._require_live_task(session, owner_id, inbox_item_id)
            schedule = await self._schedule_for_update(session, owner_id, inbox_item_id)
            self._require_schedule_version(schedule, expected_version)
            if schedule.status == "active":
                return RecurringScheduleMutation(self._snapshot(schedule), changed=False)
            schedule.version += 1
            schedule.status = "active"
            if schedule.timezone_source == "profile":
                schedule.timezone = canonical_timezone(owner.timezone)
            await self._cancel_live_occurrences(session, schedule.id)
            first = await self._prepare_generation_start(session, schedule, current)
            schedule.start_local_date = first.local_date
            schedule.next_occurrence_at = first.scheduled_for
            await session.flush()
            return RecurringScheduleMutation(self._snapshot(schedule), changed=True)

    async def refresh_profile_timezone(
        self,
        owner_id: int,
        timezone: str,
        *,
        now: datetime | None = None,
    ) -> tuple[RecurringScheduleSnapshot, ...]:
        """Atomically update a profile timezone and its active profile schedules."""

        owner_id = self._positive_id(owner_id, "owner_id")
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            return await self.refresh_profile_timezone_in_session(
                session,
                owner_id,
                timezone,
                now=now,
            )

    async def refresh_profile_timezone_in_session(
        self,
        session: AsyncSession,
        owner_id: int,
        timezone: str,
        *,
        now: datetime | None = None,
    ) -> tuple[RecurringScheduleSnapshot, ...]:
        """Refresh profile-based schedules inside a caller-owned transaction.

        The caller must already hold this owner's cross-dialect row lock. Only
        active profile-based schedules move to a new generation; explicit,
        disabled and completed schedules remain untouched.
        """

        owner_id = self._positive_id(owner_id, "owner_id")
        timezone_name = canonical_timezone(timezone)
        current = as_utc(now or datetime.now(UTC))
        owner = await session.get(User, owner_id)
        if owner is None:
            raise RecurringTaskNotEligible("owner_not_found")
        owner.timezone = timezone_name
        schedules = (
            await session.scalars(
                select(RecurringTaskReminderSchedule)
                .where(
                    RecurringTaskReminderSchedule.owner_id == owner_id,
                    RecurringTaskReminderSchedule.status == "active",
                    RecurringTaskReminderSchedule.timezone_source == "profile",
                )
                .order_by(RecurringTaskReminderSchedule.id)
                .with_for_update()
            )
        ).all()
        changed: list[RecurringScheduleSnapshot] = []
        for schedule in schedules:
            if schedule.timezone == timezone_name:
                continue
            schedule.version += 1
            schedule.timezone = timezone_name
            await self._cancel_live_occurrences(session, schedule.id)
            first = await self._prepare_generation_start(session, schedule, current)
            schedule.start_local_date = first.local_date
            schedule.next_occurrence_at = first.scheduled_for
            changed.append(self._snapshot(schedule))
        await session.flush()
        return tuple(changed)

    async def complete_for_terminal_task(
        self,
        owner_id: int,
        inbox_item_id: int,
    ) -> RecurringScheduleMutation:
        owner_id = self._positive_id(owner_id, "owner_id")
        inbox_item_id = self._positive_id(inbox_item_id, "inbox_item_id")
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            return await self.complete_for_terminal_task_in_session(
                session,
                owner_id,
                inbox_item_id,
            )

    @classmethod
    async def complete_for_terminal_task_in_session(
        cls,
        session: AsyncSession,
        owner_id: int,
        inbox_item_id: int,
    ) -> RecurringScheduleMutation:
        """Terminalize a schedule inside a caller-owned transaction.

        The caller must already hold this owner's cross-dialect row lock. This
        primitive deliberately does not commit or reacquire that lock, so task
        state and recurring state can be changed atomically by future lifecycle
        wiring.
        """

        owner_id = cls._positive_id(owner_id, "owner_id")
        inbox_item_id = cls._positive_id(inbox_item_id, "inbox_item_id")
        task = (
            await session.execute(
                select(InboxItem, TaskState)
                .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                .where(
                    InboxItem.id == inbox_item_id,
                    InboxItem.user_id == owner_id,
                    InboxItem.kind == "task",
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == inbox_item_id,
                )
            )
        ).one_or_none()
        if task is None:
            raise RecurringTaskNotEligible("task_not_found")
        item, state = task
        if item.status == "confirmed" and state.status == "active":
            raise RecurringTaskNotEligible("task_is_not_terminal")
        schedule = await session.scalar(
            select(RecurringTaskReminderSchedule)
            .where(
                RecurringTaskReminderSchedule.owner_id == owner_id,
                RecurringTaskReminderSchedule.inbox_item_id == inbox_item_id,
            )
            .with_for_update()
        )
        if schedule is None:
            return RecurringScheduleMutation(None, changed=False)
        if schedule.status == "completed":
            return RecurringScheduleMutation(cls._snapshot(schedule), changed=False)
        await cls._complete_schedule(session, schedule)
        await session.flush()
        return RecurringScheduleMutation(cls._snapshot(schedule), changed=True)

    async def list_active(self, owner_id: int) -> tuple[RecurringScheduleSnapshot, ...]:
        owner_id = self._positive_id(owner_id, "owner_id")
        async with self.db.sessions() as session:
            schedules = (
                await session.scalars(
                    select(RecurringTaskReminderSchedule)
                    .join(InboxItem, InboxItem.id == RecurringTaskReminderSchedule.inbox_item_id)
                    .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                    .where(
                        RecurringTaskReminderSchedule.owner_id == owner_id,
                        RecurringTaskReminderSchedule.status == "active",
                        InboxItem.user_id == owner_id,
                        InboxItem.kind == "task",
                        InboxItem.status == "confirmed",
                        TaskState.owner_id == owner_id,
                        TaskState.status == "active",
                    )
                    .order_by(
                        RecurringTaskReminderSchedule.next_occurrence_at,
                        RecurringTaskReminderSchedule.id,
                    )
                )
            ).all()
        return tuple(self._snapshot(schedule) for schedule in schedules)

    async def materialize_due(self, *, now: datetime | None = None) -> int:
        current = as_utc(now or datetime.now(UTC))
        async with self.db.sessions() as session:
            candidates = (
                await session.execute(
                    select(
                        RecurringTaskReminderSchedule.id,
                        RecurringTaskReminderSchedule.owner_id,
                    )
                    .where(
                        RecurringTaskReminderSchedule.status == "active",
                        RecurringTaskReminderSchedule.next_occurrence_at <= current,
                    )
                    .order_by(
                        RecurringTaskReminderSchedule.next_occurrence_at,
                        RecurringTaskReminderSchedule.id,
                    )
                    .limit(self.batch_size)
                )
            ).all()
        materialized = 0
        for schedule_id, owner_id in candidates:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                schedule = await session.scalar(
                    select(RecurringTaskReminderSchedule)
                    .where(RecurringTaskReminderSchedule.id == schedule_id)
                    .with_for_update()
                )
                if (
                    schedule is None
                    or schedule.owner_id != owner_id
                    or schedule.status != "active"
                    or as_utc(schedule.next_occurrence_at) > current
                ):
                    continue
                if not await self._task_is_live(session, schedule.owner_id, schedule.inbox_item_id):
                    await self._complete_schedule(session, schedule)
                    continue

                scheduled_for = as_utc(schedule.next_occurrence_at)
                local_date = scheduled_for.astimezone(ZoneInfo(schedule.timezone)).date()
                existing = await session.scalar(
                    select(RecurringTaskReminderOccurrence).where(
                        RecurringTaskReminderOccurrence.schedule_id == schedule.id,
                        RecurringTaskReminderOccurrence.schedule_version == schedule.version,
                        RecurringTaskReminderOccurrence.local_date == local_date,
                    )
                )
                if existing is not None:
                    if as_utc(existing.scheduled_for) == scheduled_for and existing.status in {
                        "pending",
                        "processing",
                    }:
                        continue
                    schedule.next_occurrence_at = (
                        await self._next_unused_after(session, schedule, local_date)
                    ).scheduled_for
                    continue
                if await self._date_is_consumed(session, schedule.id, local_date):
                    schedule.next_occurrence_at = (
                        await self._next_unused_after(session, schedule, local_date)
                    ).scheduled_for
                    continue

                stale = scheduled_for < current - self.grace
                occurrence = RecurringTaskReminderOccurrence(
                    schedule_id=schedule.id,
                    schedule_version=schedule.version,
                    scheduled_for=scheduled_for,
                    local_date=local_date,
                    delivery_key=self._delivery_key(schedule.id, schedule.version, local_date),
                    status="skipped_stale" if stale else "pending",
                    attempt_count=0,
                )
                session.add(occurrence)
                await session.flush()
                materialized += 1
                if stale:
                    relevant = await self._next_relevant_after_stale(session, schedule, current)
                    schedule.next_occurrence_at = relevant.scheduled_for
                    if relevant.scheduled_for <= current:
                        session.add(
                            RecurringTaskReminderOccurrence(
                                schedule_id=schedule.id,
                                schedule_version=schedule.version,
                                scheduled_for=relevant.scheduled_for,
                                local_date=relevant.local_date,
                                delivery_key=self._delivery_key(
                                    schedule.id,
                                    schedule.version,
                                    relevant.local_date,
                                ),
                                status="pending",
                                attempt_count=0,
                            )
                        )
                        await session.flush()
                        materialized += 1
        return materialized

    async def skip_stale(self, *, now: datetime | None = None) -> int:
        current = as_utc(now or datetime.now(UTC))
        cutoff = current - self.grace
        async with self.db.sessions() as session:
            candidates = (
                await session.execute(
                    select(
                        RecurringTaskReminderOccurrence.id,
                        RecurringTaskReminderSchedule.owner_id,
                    )
                    .join(
                        RecurringTaskReminderSchedule,
                        RecurringTaskReminderSchedule.id
                        == RecurringTaskReminderOccurrence.schedule_id,
                    )
                    .where(
                        RecurringTaskReminderOccurrence.status.in_({"pending", "processing"}),
                        RecurringTaskReminderOccurrence.scheduled_for < cutoff,
                    )
                    .order_by(
                        RecurringTaskReminderOccurrence.scheduled_for,
                        RecurringTaskReminderOccurrence.id,
                    )
                    .limit(self.batch_size)
                )
            ).all()
        skipped = 0
        for occurrence_id, owner_id in candidates:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                row = (
                    await session.execute(
                        select(
                            RecurringTaskReminderOccurrence,
                            RecurringTaskReminderSchedule,
                        )
                        .join(
                            RecurringTaskReminderSchedule,
                            RecurringTaskReminderSchedule.id
                            == RecurringTaskReminderOccurrence.schedule_id,
                        )
                        .where(RecurringTaskReminderOccurrence.id == occurrence_id)
                        .with_for_update()
                    )
                ).one_or_none()
                if row is None:
                    continue
                occurrence, schedule = row
                if (
                    occurrence.status not in {"pending", "processing"}
                    or as_utc(occurrence.scheduled_for) >= cutoff
                ):
                    continue
                current_generation = (
                    schedule.status == "active" and schedule.version == occurrence.schedule_version
                )
                occurrence.status = "skipped_stale" if current_generation else "cancelled"
                self._clear_claim(occurrence, clear_delivery_started=True)
                skipped += 1
                if current_generation and as_utc(schedule.next_occurrence_at) == as_utc(
                    occurrence.scheduled_for
                ):
                    schedule.next_occurrence_at = (
                        await self._next_relevant_after_stale(session, schedule, current)
                    ).scheduled_for
        return skipped

    async def claim_due(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[ClaimedRecurringOccurrence, ...]:
        current = as_utc(now or datetime.now(UTC))
        cutoff = current - self.grace
        lease_cutoff = current - self.lease
        due_pending = and_(
            RecurringTaskReminderOccurrence.status == "pending",
            RecurringTaskReminderOccurrence.scheduled_for >= cutoff,
            RecurringTaskReminderOccurrence.scheduled_for <= current,
            or_(
                RecurringTaskReminderOccurrence.next_attempt_at.is_(None),
                RecurringTaskReminderOccurrence.next_attempt_at <= current,
            ),
        )
        abandoned_before_send = and_(
            RecurringTaskReminderOccurrence.status == "processing",
            RecurringTaskReminderOccurrence.delivery_started_at.is_(None),
            RecurringTaskReminderOccurrence.claimed_at <= lease_cutoff,
            RecurringTaskReminderOccurrence.scheduled_for >= cutoff,
            RecurringTaskReminderOccurrence.scheduled_for <= current,
        )
        async with self.db.sessions() as session:
            candidates = (
                await session.execute(
                    select(
                        RecurringTaskReminderOccurrence.id,
                        RecurringTaskReminderSchedule.owner_id,
                    )
                    .join(
                        RecurringTaskReminderSchedule,
                        RecurringTaskReminderSchedule.id
                        == RecurringTaskReminderOccurrence.schedule_id,
                    )
                    .join(
                        InboxItem,
                        InboxItem.id == RecurringTaskReminderSchedule.inbox_item_id,
                    )
                    .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                    .join(User, User.id == RecurringTaskReminderSchedule.owner_id)
                    .where(or_(due_pending, abandoned_before_send))
                    .where(
                        RecurringTaskReminderSchedule.status == "active",
                        RecurringTaskReminderSchedule.version
                        == RecurringTaskReminderOccurrence.schedule_version,
                        RecurringTaskReminderSchedule.next_occurrence_at
                        == RecurringTaskReminderOccurrence.scheduled_for,
                        InboxItem.user_id == RecurringTaskReminderSchedule.owner_id,
                        InboxItem.kind == "task",
                        InboxItem.status == "confirmed",
                        TaskState.owner_id == RecurringTaskReminderSchedule.owner_id,
                        TaskState.status == "active",
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                    )
                    .order_by(
                        RecurringTaskReminderOccurrence.scheduled_for,
                        RecurringTaskReminderOccurrence.id,
                    )
                    .limit(self.batch_size)
                )
            ).all()

        claimed: list[ClaimedRecurringOccurrence] = []
        for occurrence_id, owner_id in candidates:
            token = str(uuid4())
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                changed = await session.execute(
                    update(RecurringTaskReminderOccurrence)
                    .where(
                        RecurringTaskReminderOccurrence.id == occurrence_id,
                        or_(due_pending, abandoned_before_send),
                        self._current_delivery_exists(),
                    )
                    .values(
                        status="processing",
                        claim_token=token,
                        claimed_at=current,
                        delivery_started_at=None,
                        next_attempt_at=None,
                        attempt_count=RecurringTaskReminderOccurrence.attempt_count + 1,
                    )
                    .returning(RecurringTaskReminderOccurrence.id)
                )
                if changed.scalar_one_or_none() is None:
                    continue
                row = await self._claimed_row(session, occurrence_id, token)
                if row is None:
                    await self._release_token(session, occurrence_id, token, decrement_attempt=True)
                    continue
                occurrence, schedule, item, state, owner = row
                if not is_full_access_tier(owner.access_tier):
                    await self._release_token(session, occurrence_id, token, decrement_attempt=True)
                    continue
                claimed.append(
                    ClaimedRecurringOccurrence(
                        id=occurrence.id,
                        schedule_id=schedule.id,
                        schedule_version=schedule.version,
                        owner_id=owner.id,
                        inbox_item_id=item.id,
                        inbox_item_version=item.version,
                        task_version=state.version,
                        access_version=owner.access_version,
                        claim_token=token,
                        delivery_key=occurrence.delivery_key,
                        scheduled_for=as_utc(occurrence.scheduled_for),
                        local_date=occurrence.local_date,
                        destination_id=owner.telegram_id,
                        title=item.title,
                        local_time=schedule.local_time,
                        timezone=schedule.timezone,
                        attempt_count=occurrence.attempt_count,
                    )
                )
        return tuple(claimed)

    async def delivery_readiness(
        self,
        claimed: ClaimedRecurringOccurrence,
        *,
        delivery_started: bool | None = None,
    ) -> DeliveryReadiness:
        async with self.db.sessions() as session:
            row = (
                await session.execute(
                    select(
                        User.access_tier,
                        User.access_version,
                        User.telegram_id,
                        InboxItem.version,
                        TaskState.version,
                    )
                    .select_from(RecurringTaskReminderOccurrence)
                    .join(
                        RecurringTaskReminderSchedule,
                        RecurringTaskReminderSchedule.id
                        == RecurringTaskReminderOccurrence.schedule_id,
                    )
                    .join(
                        InboxItem,
                        InboxItem.id == RecurringTaskReminderSchedule.inbox_item_id,
                    )
                    .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                    .join(User, User.id == RecurringTaskReminderSchedule.owner_id)
                    .where(
                        *self._claimed_fence(claimed, delivery_started=delivery_started),
                        InboxItem.user_id == claimed.owner_id,
                        InboxItem.kind == "task",
                        InboxItem.status == "confirmed",
                        TaskState.owner_id == claimed.owner_id,
                        TaskState.status == "active",
                    )
                )
            ).one_or_none()
        if row is None:
            return "stale"
        tier, access_version, telegram_id, inbox_item_version, task_version = row
        if not is_full_access_tier(tier):
            return "access_denied"
        if access_version != claimed.access_version:
            return "access_changed"
        if telegram_id != claimed.destination_id:
            return "destination_changed"
        if inbox_item_version != claimed.inbox_item_version:
            return "task_changed"
        if task_version != claimed.task_version:
            return "task_changed"
        return "ready"

    async def begin_delivery(
        self,
        claimed: ClaimedRecurringOccurrence,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_owner(session, claimed.owner_id)
            if current > claimed.scheduled_for + self.grace:
                skipped = await session.execute(
                    update(RecurringTaskReminderOccurrence)
                    .where(
                        *self._claimed_fence(claimed, delivery_started=False),
                        self._snapshot_delivery_exists(claimed),
                    )
                    .values(
                        status="skipped_stale",
                        claim_token=None,
                        claimed_at=None,
                        delivery_started_at=None,
                        next_attempt_at=None,
                    )
                    .returning(RecurringTaskReminderOccurrence.id)
                )
                if skipped.scalar_one_or_none() is not None:
                    schedule = await session.scalar(
                        select(RecurringTaskReminderSchedule).where(
                            RecurringTaskReminderSchedule.id == claimed.schedule_id,
                            RecurringTaskReminderSchedule.owner_id == claimed.owner_id,
                            RecurringTaskReminderSchedule.status == "active",
                            RecurringTaskReminderSchedule.version == claimed.schedule_version,
                            RecurringTaskReminderSchedule.next_occurrence_at
                            == claimed.scheduled_for,
                        )
                    )
                    if schedule is None:
                        raise RecurringFenceLost("schedule fence lost while skipping late delivery")
                    schedule.next_occurrence_at = (
                        await self._next_relevant_after_stale(session, schedule, current)
                    ).scheduled_for
                return False
            changed = await session.execute(
                update(RecurringTaskReminderOccurrence)
                .where(
                    *self._claimed_fence(claimed, delivery_started=False),
                    self._snapshot_delivery_exists(claimed),
                    RecurringTaskReminderOccurrence.scheduled_for <= current,
                    RecurringTaskReminderOccurrence.scheduled_for >= current - self.grace,
                )
                .values(delivery_started_at=current)
                .returning(RecurringTaskReminderOccurrence.id)
            )
            return changed.scalar_one_or_none() is not None

    async def mark_sent(
        self,
        claimed: ClaimedRecurringOccurrence,
        message_id: int | None,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_owner(session, claimed.owner_id)
            changed = await session.execute(
                update(RecurringTaskReminderOccurrence)
                .where(
                    *self._claimed_fence(claimed, delivery_started=True),
                    self._snapshot_delivery_exists(claimed),
                )
                .values(
                    status="sent",
                    claim_token=None,
                    claimed_at=None,
                    next_attempt_at=None,
                    sent_at=current,
                    telegram_message_id=message_id,
                    last_error_type=None,
                )
                .returning(RecurringTaskReminderOccurrence.id)
            )
            if changed.scalar_one_or_none() is None:
                return False
            schedule = await session.scalar(
                select(RecurringTaskReminderSchedule).where(
                    RecurringTaskReminderSchedule.id == claimed.schedule_id,
                    RecurringTaskReminderSchedule.owner_id == claimed.owner_id,
                    RecurringTaskReminderSchedule.status == "active",
                    RecurringTaskReminderSchedule.version == claimed.schedule_version,
                    RecurringTaskReminderSchedule.next_occurrence_at == claimed.scheduled_for,
                )
            )
            if schedule is None:
                raise RecurringFenceLost("schedule fence lost while marking sent")
            next_occurrence = await self._next_unused_after(
                session,
                schedule,
                claimed.local_date,
            )
            schedule.next_occurrence_at = next_occurrence.scheduled_for
            await session.flush()
            return True

    async def release_claim(
        self,
        claimed: ClaimedRecurringOccurrence,
        *,
        attempted: bool = False,
        retry_at: datetime | None = None,
        error_type: str | None = None,
    ) -> bool:
        clean_error = self._safe_error_type(error_type)
        async with self.db.session() as session:
            values: dict[str, object | None] = {
                "status": "pending",
                "claim_token": None,
                "claimed_at": None,
                "delivery_started_at": None,
                "next_attempt_at": as_utc(retry_at) if retry_at is not None else None,
                "last_error_type": clean_error,
            }
            if not attempted:
                values["attempt_count"] = max(0, claimed.attempt_count - 1)
            changed = await session.execute(
                update(RecurringTaskReminderOccurrence)
                .where(
                    RecurringTaskReminderOccurrence.id == claimed.id,
                    RecurringTaskReminderOccurrence.status == "processing",
                    RecurringTaskReminderOccurrence.claim_token == claimed.claim_token,
                )
                .values(**values)
                .returning(RecurringTaskReminderOccurrence.id)
            )
            return changed.scalar_one_or_none() is not None

    async def release_retry(
        self,
        claimed: ClaimedRecurringOccurrence,
        exc: BaseException,
        *,
        now: datetime | None = None,
        immediate: bool = False,
    ) -> bool:
        current = as_utc(now or datetime.now(UTC))
        retry_at = None if immediate else current + self._retry_delay(claimed.attempt_count)
        return await self.release_claim(
            claimed,
            attempted=True,
            retry_at=retry_at,
            error_type=type(exc).__name__,
        )

    async def terminalize_after_send(
        self,
        claimed: ClaimedRecurringOccurrence,
        readiness: DeliveryReadiness,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_owner(session, claimed.owner_id)
            schedule = await session.scalar(
                select(RecurringTaskReminderSchedule).where(
                    RecurringTaskReminderSchedule.id == claimed.schedule_id,
                    RecurringTaskReminderSchedule.owner_id == claimed.owner_id,
                )
            )
            if (
                schedule is not None
                and schedule.status == "active"
                and schedule.version == claimed.schedule_version
                and not await self._task_is_live(
                    session,
                    claimed.owner_id,
                    claimed.inbox_item_id,
                )
            ):
                await self._complete_schedule(session, schedule)
                return True
            current_generation = (
                schedule is not None
                and schedule.status == "active"
                and schedule.version == claimed.schedule_version
                and as_utc(schedule.next_occurrence_at) == claimed.scheduled_for
            )
            terminal_status = (
                "skipped_stale"
                if current_generation
                and readiness
                in {
                    "access_denied",
                    "access_changed",
                    "task_changed",
                    "destination_changed",
                }
                else "cancelled"
            )
            changed = await session.execute(
                update(RecurringTaskReminderOccurrence)
                .where(
                    RecurringTaskReminderOccurrence.id == claimed.id,
                    RecurringTaskReminderOccurrence.status == "processing",
                    RecurringTaskReminderOccurrence.claim_token == claimed.claim_token,
                )
                .values(
                    status=terminal_status,
                    claim_token=None,
                    claimed_at=None,
                    delivery_started_at=None,
                    next_attempt_at=None,
                    last_error_type=None,
                )
                .returning(RecurringTaskReminderOccurrence.id)
            )
            if changed.scalar_one_or_none() is None:
                return False
            if current_generation and terminal_status == "skipped_stale" and schedule is not None:
                schedule.next_occurrence_at = (
                    await self._next_relevant_after_stale(session, schedule, current)
                ).scheduled_for
            return True

    async def _schedule_for_update(
        self,
        session: AsyncSession,
        owner_id: int,
        inbox_item_id: int,
    ) -> RecurringTaskReminderSchedule:
        schedule = await session.scalar(
            select(RecurringTaskReminderSchedule)
            .where(
                RecurringTaskReminderSchedule.owner_id == owner_id,
                RecurringTaskReminderSchedule.inbox_item_id == inbox_item_id,
            )
            .with_for_update()
        )
        if schedule is None:
            raise RecurringScheduleConflict("schedule_not_found")
        return schedule

    @staticmethod
    async def _lock_owner(session: AsyncSession, owner_id: int) -> User:
        changed = await session.execute(
            update(User).where(User.id == owner_id).values(updated_at=User.updated_at)
        )
        if changed.rowcount != 1:
            raise RecurringTaskNotEligible("owner_not_found")
        owner = await session.scalar(select(User).where(User.id == owner_id).with_for_update())
        if owner is None:
            raise RecurringTaskNotEligible("owner_not_found")
        return owner

    @staticmethod
    async def _require_live_task(
        session: AsyncSession,
        owner_id: int,
        inbox_item_id: int,
    ) -> tuple[InboxItem, TaskState]:
        row = (
            await session.execute(
                select(InboxItem, TaskState)
                .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                .where(
                    InboxItem.id == inbox_item_id,
                    InboxItem.user_id == owner_id,
                    InboxItem.kind == "task",
                    TaskState.owner_id == owner_id,
                )
            )
        ).one_or_none()
        if row is None:
            raise RecurringTaskNotEligible("task_not_found")
        item, state = row
        if item.status != "confirmed":
            raise RecurringTaskNotEligible("task_not_active")
        if state.status != "active":
            raise RecurringTaskNotEligible("task_terminal")
        return item, state

    @staticmethod
    async def _task_is_live(session: AsyncSession, owner_id: int, inbox_item_id: int) -> bool:
        return (
            await session.scalar(
                select(TaskState.id)
                .join(InboxItem, InboxItem.id == TaskState.inbox_item_id)
                .where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == inbox_item_id,
                    TaskState.status == "active",
                    InboxItem.user_id == owner_id,
                    InboxItem.kind == "task",
                    InboxItem.status == "confirmed",
                )
            )
        ) is not None

    @staticmethod
    async def _complete_schedule(
        session: AsyncSession,
        schedule: RecurringTaskReminderSchedule,
    ) -> None:
        if schedule.status != "completed":
            schedule.status = "completed"
            schedule.version += 1
        await RecurringTaskReminderService._cancel_live_occurrences(session, schedule.id)

    @staticmethod
    async def _cancel_live_occurrences(session: AsyncSession, schedule_id: int) -> None:
        # Once transport I/O started, a cancelled row is not provably unsent
        # and therefore must never be reused for another same-day generation.
        await session.execute(
            update(RecurringTaskReminderOccurrence)
            .where(
                RecurringTaskReminderOccurrence.schedule_id == schedule_id,
                RecurringTaskReminderOccurrence.status == "processing",
                RecurringTaskReminderOccurrence.delivery_started_at.is_not(None),
            )
            .values(
                status="skipped_stale",
                claim_token=None,
                claimed_at=None,
                delivery_started_at=None,
                next_attempt_at=None,
            )
        )
        await session.execute(
            update(RecurringTaskReminderOccurrence)
            .where(
                RecurringTaskReminderOccurrence.schedule_id == schedule_id,
                RecurringTaskReminderOccurrence.status.in_({"pending", "processing"}),
                RecurringTaskReminderOccurrence.delivery_started_at.is_(None),
            )
            .values(
                status="cancelled",
                claim_token=None,
                claimed_at=None,
                delivery_started_at=None,
                next_attempt_at=None,
            )
        )

    async def _prepare_generation_start(
        self,
        session: AsyncSession,
        schedule: RecurringTaskReminderSchedule,
        now: datetime,
    ) -> DailyOccurrence:
        candidate = calculate_next_daily_occurrence(
            schedule.local_time,
            schedule.timezone,
            now=now,
        )
        # Cancelled-before-I/O history does not consume its calendar date and
        # remains immutable. A new generation materializes its own row at due.
        return await self._skip_used_dates(session, schedule.id, schedule, candidate)

    async def _next_unused_after(
        self,
        session: AsyncSession,
        schedule: RecurringTaskReminderSchedule,
        local_date: date,
    ) -> DailyOccurrence:
        candidate = _next_daily_after(local_date, schedule.local_time, schedule.timezone)
        return await self._skip_used_dates(session, schedule.id, schedule, candidate)

    async def _next_relevant_after_stale(
        self,
        session: AsyncSession,
        schedule: RecurringTaskReminderSchedule,
        now: datetime,
    ) -> DailyOccurrence:
        zone = ZoneInfo(schedule.timezone)
        candidate = calculate_daily_occurrence(
            now.astimezone(zone).date(),
            schedule.local_time,
            schedule.timezone,
        )
        while candidate.scheduled_for < now - self.grace:
            candidate = _next_daily_after(
                candidate.local_date,
                schedule.local_time,
                schedule.timezone,
            )
        return await self._skip_used_dates(session, schedule.id, schedule, candidate)

    @staticmethod
    async def _skip_used_dates(
        session: AsyncSession,
        schedule_id: int,
        schedule: RecurringTaskReminderSchedule,
        candidate: DailyOccurrence,
    ) -> DailyOccurrence:
        used_dates = set(
            (
                await session.scalars(
                    select(RecurringTaskReminderOccurrence.local_date).where(
                        RecurringTaskReminderOccurrence.schedule_id == schedule_id,
                        RecurringTaskReminderOccurrence.local_date >= candidate.local_date,
                        or_(
                            RecurringTaskReminderOccurrence.status.in_({"sent", "skipped_stale"}),
                            RecurringTaskReminderOccurrence.delivery_started_at.is_not(None),
                        ),
                    )
                )
            ).all()
        )
        for _ in range(370):
            if candidate.local_date not in used_dates:
                return candidate
            candidate = _next_daily_after(
                candidate.local_date,
                schedule.local_time,
                schedule.timezone,
            )
        raise ValueError("no unused daily occurrence is available")

    @staticmethod
    async def _date_is_consumed(
        session: AsyncSession,
        schedule_id: int,
        local_date: date,
    ) -> bool:
        return (
            await session.scalar(
                select(RecurringTaskReminderOccurrence.id).where(
                    RecurringTaskReminderOccurrence.schedule_id == schedule_id,
                    RecurringTaskReminderOccurrence.local_date == local_date,
                    or_(
                        RecurringTaskReminderOccurrence.status.in_({"sent", "skipped_stale"}),
                        RecurringTaskReminderOccurrence.delivery_started_at.is_not(None),
                    ),
                )
            )
        ) is not None

    @staticmethod
    def _current_delivery_exists():
        return exists(
            select(RecurringTaskReminderSchedule.id)
            .join(
                InboxItem,
                InboxItem.id == RecurringTaskReminderSchedule.inbox_item_id,
            )
            .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
            .join(User, User.id == RecurringTaskReminderSchedule.owner_id)
            .where(
                RecurringTaskReminderSchedule.id == RecurringTaskReminderOccurrence.schedule_id,
                RecurringTaskReminderSchedule.status == "active",
                RecurringTaskReminderSchedule.version
                == RecurringTaskReminderOccurrence.schedule_version,
                RecurringTaskReminderSchedule.next_occurrence_at
                == RecurringTaskReminderOccurrence.scheduled_for,
                InboxItem.user_id == RecurringTaskReminderSchedule.owner_id,
                InboxItem.kind == "task",
                InboxItem.status == "confirmed",
                TaskState.owner_id == RecurringTaskReminderSchedule.owner_id,
                TaskState.status == "active",
                User.access_tier.in_(FULL_ACCESS_TIERS),
            )
        )

    @staticmethod
    def _snapshot_delivery_exists(claimed: ClaimedRecurringOccurrence):
        return exists(
            select(RecurringTaskReminderSchedule.id)
            .join(
                InboxItem,
                InboxItem.id == RecurringTaskReminderSchedule.inbox_item_id,
            )
            .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
            .join(User, User.id == RecurringTaskReminderSchedule.owner_id)
            .where(
                RecurringTaskReminderSchedule.id == claimed.schedule_id,
                RecurringTaskReminderSchedule.owner_id == claimed.owner_id,
                RecurringTaskReminderSchedule.inbox_item_id == claimed.inbox_item_id,
                RecurringTaskReminderSchedule.status == "active",
                RecurringTaskReminderSchedule.version == claimed.schedule_version,
                RecurringTaskReminderSchedule.next_occurrence_at == claimed.scheduled_for,
                InboxItem.id == claimed.inbox_item_id,
                InboxItem.user_id == claimed.owner_id,
                InboxItem.kind == "task",
                InboxItem.status == "confirmed",
                InboxItem.version == claimed.inbox_item_version,
                TaskState.owner_id == claimed.owner_id,
                TaskState.status == "active",
                TaskState.version == claimed.task_version,
                User.id == claimed.owner_id,
                User.telegram_id == claimed.destination_id,
                User.access_tier.in_(FULL_ACCESS_TIERS),
                User.access_version == claimed.access_version,
            )
        )

    @staticmethod
    def _claimed_fence(
        claimed: ClaimedRecurringOccurrence,
        *,
        delivery_started: bool | None,
    ) -> tuple[object, ...]:
        conditions: list[object] = [
            RecurringTaskReminderOccurrence.id == claimed.id,
            RecurringTaskReminderOccurrence.schedule_id == claimed.schedule_id,
            RecurringTaskReminderOccurrence.schedule_version == claimed.schedule_version,
            RecurringTaskReminderOccurrence.status == "processing",
            RecurringTaskReminderOccurrence.claim_token == claimed.claim_token,
            RecurringTaskReminderOccurrence.delivery_key == claimed.delivery_key,
        ]
        if delivery_started is True:
            conditions.append(RecurringTaskReminderOccurrence.delivery_started_at.is_not(None))
        elif delivery_started is False:
            conditions.append(RecurringTaskReminderOccurrence.delivery_started_at.is_(None))
        return tuple(conditions)

    @staticmethod
    async def _claimed_row(
        session: AsyncSession,
        occurrence_id: int,
        claim_token: str,
    ) -> (
        tuple[
            RecurringTaskReminderOccurrence,
            RecurringTaskReminderSchedule,
            InboxItem,
            TaskState,
            User,
        ]
        | None
    ):
        return (
            await session.execute(
                select(
                    RecurringTaskReminderOccurrence,
                    RecurringTaskReminderSchedule,
                    InboxItem,
                    TaskState,
                    User,
                )
                .join(
                    RecurringTaskReminderSchedule,
                    RecurringTaskReminderSchedule.id == RecurringTaskReminderOccurrence.schedule_id,
                )
                .join(
                    InboxItem,
                    InboxItem.id == RecurringTaskReminderSchedule.inbox_item_id,
                )
                .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                .join(User, User.id == RecurringTaskReminderSchedule.owner_id)
                .where(
                    RecurringTaskReminderOccurrence.id == occurrence_id,
                    RecurringTaskReminderOccurrence.status == "processing",
                    RecurringTaskReminderOccurrence.claim_token == claim_token,
                    RecurringTaskReminderSchedule.status == "active",
                    RecurringTaskReminderSchedule.version
                    == RecurringTaskReminderOccurrence.schedule_version,
                    RecurringTaskReminderSchedule.next_occurrence_at
                    == RecurringTaskReminderOccurrence.scheduled_for,
                    InboxItem.user_id == RecurringTaskReminderSchedule.owner_id,
                    InboxItem.kind == "task",
                    InboxItem.status == "confirmed",
                    TaskState.owner_id == RecurringTaskReminderSchedule.owner_id,
                    TaskState.status == "active",
                )
            )
        ).one_or_none()

    @staticmethod
    async def _release_token(
        session: AsyncSession,
        occurrence_id: int,
        token: str,
        *,
        decrement_attempt: bool,
    ) -> None:
        occurrence = await session.scalar(
            select(RecurringTaskReminderOccurrence).where(
                RecurringTaskReminderOccurrence.id == occurrence_id,
                RecurringTaskReminderOccurrence.status == "processing",
                RecurringTaskReminderOccurrence.claim_token == token,
            )
        )
        if occurrence is None:
            return
        occurrence.status = "pending"
        if decrement_attempt:
            occurrence.attempt_count = max(0, occurrence.attempt_count - 1)
        RecurringTaskReminderService._clear_claim(occurrence, clear_delivery_started=True)

    @staticmethod
    def _clear_claim(
        occurrence: RecurringTaskReminderOccurrence,
        *,
        clear_delivery_started: bool,
    ) -> None:
        occurrence.claim_token = None
        occurrence.claimed_at = None
        occurrence.next_attempt_at = None
        if clear_delivery_started:
            occurrence.delivery_started_at = None

    @staticmethod
    def _schedule_timezone(owner: User, timezone: str | None, source: TimezoneSource) -> str:
        if source == "profile":
            return canonical_timezone(owner.timezone)
        if timezone is None:
            raise ValueError("explicit timezone is required")
        return canonical_timezone(timezone)

    @staticmethod
    def _delivery_key(schedule_id: int, version: int, local_date: date) -> str:
        return f"recurring:{schedule_id}:v{version}:{local_date.isoformat()}"

    @staticmethod
    def _retry_delay(attempt_count: int) -> timedelta:
        return timedelta(seconds=min(300, 5 * (2 ** min(max(0, attempt_count - 1), 6))))

    @staticmethod
    def _safe_error_type(value: str | None) -> str | None:
        if value is None:
            return None
        clean = "".join(character for character in value if character.isalnum() or character == "_")
        return clean[:120] or "TransportError"

    @staticmethod
    def _snapshot(schedule: RecurringTaskReminderSchedule) -> RecurringScheduleSnapshot:
        return RecurringScheduleSnapshot(
            id=schedule.id,
            owner_id=schedule.owner_id,
            inbox_item_id=schedule.inbox_item_id,
            recurrence_kind="daily",
            local_time=schedule.local_time,
            timezone=schedule.timezone,
            timezone_source=schedule.timezone_source,  # type: ignore[arg-type]
            start_local_date=schedule.start_local_date,
            next_occurrence_at=as_utc(schedule.next_occurrence_at),
            status=schedule.status,  # type: ignore[arg-type]
            version=schedule.version,
        )

    @staticmethod
    def _optional_version(value: int | None) -> int | None:
        if value is None:
            return None
        return RecurringTaskReminderService._positive_id(value, "expected_version")

    @staticmethod
    def _require_schedule_version(
        schedule: RecurringTaskReminderSchedule,
        expected_version: int | None,
    ) -> None:
        if expected_version is not None and schedule.version != expected_version:
            raise RecurringFenceLost("schedule version changed")

    @staticmethod
    def _require_access_generation(owner: User, expected_version: int | None) -> None:
        if expected_version is None:
            return
        if not is_full_access_tier(owner.access_tier) or owner.access_version != expected_version:
            raise RecurringFenceLost("access version changed")

    @staticmethod
    def _positive_id(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value


class RecurringTaskReminderEngine:
    """Telegram-independent claim/send/advance coordinator.

    The database transaction that marks ``delivery_started_at`` is committed
    before the injected callback runs. A crash after that point is deliberately
    not lease-reclaimed, preventing an automatic duplicate after an uncertain
    transport outcome.
    """

    def __init__(
        self,
        db: Database,
        send: RecurringReminderSendCallback,
        *,
        delete_sent: RecurringReminderDeleteCallback | None = None,
        grace_minutes: int = 120,
        lease_seconds: int = 120,
        batch_size: int = 20,
        now_provider: Callable[[], datetime] | None = None,
    ):
        self.service = RecurringTaskReminderService(
            db,
            grace_minutes=grace_minutes,
            lease_seconds=lease_seconds,
            batch_size=batch_size,
        )
        self.send = send
        self.delete_sent = delete_sent
        self.now_provider = now_provider

    async def deliver_due(self, *, now: datetime | None = None) -> int:
        current, fresh_now = self._run_clock(now)
        await self.service.skip_stale(now=current)
        await self.service.materialize_due(now=current)
        delivered = 0
        for claimed in await self.service.claim_due(now=current):
            try:
                readiness = await self.service.delivery_readiness(
                    claimed,
                    delivery_started=False,
                )
            except asyncio.CancelledError:
                await self.service.release_claim(claimed, attempted=False)
                raise
            except Exception as exc:
                await self._safe_release_unattempted(claimed)
                logger.warning(
                    "Recurring reminder readiness failed occurrence_id=%s error_type=%s",
                    claimed.id,
                    type(exc).__name__,
                )
                continue
            if readiness != "ready":
                await self._handle_before_send_fence(claimed, readiness)
                continue

            try:
                began = await self.service.begin_delivery(claimed, now=fresh_now())
            except asyncio.CancelledError:
                await self.service.release_claim(claimed, attempted=False)
                raise
            except Exception as exc:
                await self._safe_release_unattempted(claimed)
                logger.warning(
                    "Recurring reminder begin failed occurrence_id=%s error_type=%s",
                    claimed.id,
                    type(exc).__name__,
                )
                continue
            if not began:
                await self._handle_before_send_fence(
                    claimed,
                    await self._safe_readiness(claimed, delivery_started=None),
                )
                continue

            delivery = RecurringReminderDelivery(
                occurrence_id=claimed.id,
                schedule_id=claimed.schedule_id,
                delivery_key=claimed.delivery_key,
                destination_id=claimed.destination_id,
                title=claimed.title,
                scheduled_for=claimed.scheduled_for,
                local_date=claimed.local_date,
                local_time=claimed.local_time,
                timezone=claimed.timezone,
            )
            try:
                message_id = await self.send(delivery)
            except asyncio.CancelledError:
                # Cancellation can arrive after the transport accepted the
                # message. Keep the delivery-started uncertainty fence; a
                # later grace expiry may skip/advance it, but no lease worker
                # is allowed to call the transport again.
                raise
            except Exception as exc:
                # No transport exception proves that Telegram rejected the
                # message. Keep the delivery-started uncertainty fence so this
                # calendar occurrence can never be sent automatically again.
                logger.warning(
                    "Recurring reminder delivery failed occurrence_id=%s error_type=%s",
                    claimed.id,
                    type(exc).__name__,
                )
                continue

            try:
                readiness = await self.service.delivery_readiness(
                    claimed,
                    delivery_started=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Recurring reminder post-send check failed occurrence_id=%s error_type=%s",
                    claimed.id,
                    type(exc).__name__,
                )
                # delivery_started_at remains as an uncertainty fence. It is
                # never lease-reclaimed and therefore cannot auto-send twice.
                continue
            if readiness != "ready":
                await self._compensate(claimed, message_id)
                await self.service.terminalize_after_send(
                    claimed,
                    readiness,
                    now=fresh_now(),
                )
                continue
            try:
                marked = await self.service.mark_sent(claimed, message_id, now=fresh_now())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Recurring reminder mark sent failed occurrence_id=%s error_type=%s",
                    claimed.id,
                    type(exc).__name__,
                )
                continue
            if marked:
                delivered += 1
                continue

            readiness = await self._safe_readiness(claimed, delivery_started=None)
            await self._compensate(claimed, message_id)
            await self.service.terminalize_after_send(
                claimed,
                readiness,
                now=fresh_now(),
            )
        return delivered

    def _run_clock(
        self,
        supplied_now: datetime | None,
    ) -> tuple[datetime, Callable[[], datetime]]:
        if self.now_provider is not None:
            initial = as_utc(supplied_now or self.now_provider())

            def fresh() -> datetime:
                return as_utc(self.now_provider())  # type: ignore[misc]

            return initial, fresh

        wall_started = datetime.now(UTC)
        initial = as_utc(supplied_now or wall_started)

        def fresh() -> datetime:
            return initial + (datetime.now(UTC) - wall_started)

        return initial, fresh

    async def _handle_before_send_fence(
        self,
        claimed: ClaimedRecurringOccurrence,
        readiness: DeliveryReadiness,
    ) -> None:
        if readiness in {
            "access_denied",
            "access_changed",
            "task_changed",
            "destination_changed",
        }:
            await self.service.release_claim(claimed, attempted=False)
            return
        await self.service.terminalize_after_send(claimed, readiness)

    async def _safe_release_unattempted(self, claimed: ClaimedRecurringOccurrence) -> None:
        try:
            await self.service.release_claim(claimed, attempted=False)
        except Exception as exc:
            logger.warning(
                "Recurring reminder release failed occurrence_id=%s error_type=%s",
                claimed.id,
                type(exc).__name__,
            )

    async def _safe_readiness(
        self,
        claimed: ClaimedRecurringOccurrence,
        *,
        delivery_started: bool | None,
    ) -> DeliveryReadiness:
        try:
            return await self.service.delivery_readiness(
                claimed,
                delivery_started=delivery_started,
            )
        except Exception as exc:
            logger.warning(
                "Recurring reminder fence check failed occurrence_id=%s error_type=%s",
                claimed.id,
                type(exc).__name__,
            )
            return "stale"

    async def _compensate(
        self,
        claimed: ClaimedRecurringOccurrence,
        message_id: int | None,
    ) -> None:
        if message_id is None or self.delete_sent is None:
            return
        try:
            await self.delete_sent(claimed.destination_id, message_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Recurring reminder compensation failed occurrence_id=%s error_type=%s",
                claimed.id,
                type(exc).__name__,
            )
