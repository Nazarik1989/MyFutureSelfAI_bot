from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import and_, or_, select, update

from .db import Database
from .models import DraftInboxItem, InboxItem, TaskReminder
from .schemas import TemporalResolution

logger = logging.getLogger(__name__)

ReminderSendCallback = Callable[[int, str], Awaitable[int | None]]


def as_utc(value: datetime) -> datetime:
    """Normalize SQLite's naive UTC values and PostgreSQL's aware values."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ReminderSchedule:
    event_at: datetime
    remind_at: datetime
    timezone: str


@dataclass(frozen=True, slots=True)
class ClaimedReminder:
    id: int
    claim_token: str
    chat_id: int
    title: str
    description: str | None
    event_at: datetime
    timezone: str
    attempt_count: int


def schedule_from_temporal(
    temporal_data: dict[str, object] | None,
    *,
    date_event_hour: int,
    lead_minutes: int,
) -> ReminderSchedule | None:
    if not temporal_data:
        return None
    temporal = TemporalResolution.model_validate(temporal_data)
    zone = ZoneInfo(temporal.timezone)
    if temporal.precision == "date":
        local_event = datetime.combine(
            temporal.resolved_local_date,
            time(hour=date_event_hour),
            tzinfo=zone,
        )
        event_at = local_event.astimezone(UTC)
    else:
        event_at = as_utc(temporal.resolved_at)
    remind_at = (
        as_utc(temporal.remind_at)
        if temporal.remind_at is not None
        else event_at - timedelta(minutes=lead_minutes)
    )
    return ReminderSchedule(
        event_at=event_at,
        remind_at=remind_at,
        timezone=temporal.timezone,
    )


def reminder_for_inbox_item(
    item: InboxItem,
    *,
    telegram_user_id: int,
    chat_id: int,
    date_event_hour: int,
    lead_minutes: int,
) -> TaskReminder | None:
    if item.kind != "task":
        return None
    schedule = schedule_from_temporal(
        item.temporal_resolution,
        date_event_hour=date_event_hour,
        lead_minutes=lead_minutes,
    )
    if schedule is None:
        return None
    return TaskReminder(
        inbox_item_id=item.id,
        telegram_user_id=telegram_user_id,
        chat_id=chat_id,
        event_at=schedule.event_at,
        remind_at=schedule.remind_at,
        timezone=schedule.timezone,
        delivery_key=f"inbox:{item.id}:task-reminder:v1",
        status="pending",
    )


class TaskReminderEngine:
    """Persistent task reminder outbox with leases and idempotent state transitions."""

    def __init__(
        self,
        db: Database,
        send: ReminderSendCallback,
        *,
        lease_seconds: int = 120,
        batch_size: int = 20,
        date_event_hour: int = 9,
        lead_minutes: int = 30,
    ):
        self.db = db
        self.send = send
        self.lease = timedelta(seconds=lease_seconds)
        self.batch_size = batch_size
        self.date_event_hour = date_event_hour
        self.lead_minutes = lead_minutes

    async def reconcile_missing(self, *, now: datetime | None = None) -> int:
        """Create reminders for future pre-engine tasks that still retain their draft chat."""
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            rows = (
                await session.execute(
                    select(InboxItem, DraftInboxItem)
                    .join(DraftInboxItem, DraftInboxItem.id == InboxItem.draft_id)
                    .outerjoin(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                    .where(
                        InboxItem.kind == "task",
                        InboxItem.status == "confirmed",
                        InboxItem.temporal_resolution.is_not(None),
                        TaskReminder.id.is_(None),
                    )
                )
            ).all()
            created = 0
            for item, draft in rows:
                try:
                    reminder = reminder_for_inbox_item(
                        item,
                        telegram_user_id=draft.telegram_user_id,
                        chat_id=draft.chat_id,
                        date_event_hour=self.date_event_hour,
                        lead_minutes=self.lead_minutes,
                    )
                except (ValueError, ZoneInfoNotFoundError) as exc:
                    logger.warning(
                        "Task reminder reconciliation skipped inbox_item_id=%s error_type=%s",
                        item.id,
                        type(exc).__name__,
                    )
                    continue
                if reminder is None or as_utc(reminder.event_at) <= current:
                    continue
                session.add(reminder)
                created += 1
            if created:
                await session.flush()
        return created

    async def deliver_due(self, *, now: datetime | None = None) -> int:
        current = as_utc(now or datetime.now(UTC))
        delivered = 0
        for reminder in await self._claim_due(current):
            try:
                message_id = await self.send(
                    reminder.chat_id,
                    self._message(reminder),
                )
            except Exception as exc:
                await self._release_after_failure(reminder, current, exc)
                logger.warning(
                    "Task reminder delivery failed reminder_id=%s error_type=%s",
                    reminder.id,
                    type(exc).__name__,
                )
                continue
            if await self._mark_sent(reminder, current, message_id):
                delivered += 1
        return delivered

    async def _claim_due(self, now: datetime) -> list[ClaimedReminder]:
        stale_before = now - self.lease
        due_pending = and_(
            TaskReminder.status == "pending",
            or_(
                and_(
                    TaskReminder.next_attempt_at.is_(None),
                    TaskReminder.remind_at <= now,
                ),
                TaskReminder.next_attempt_at <= now,
            ),
        )
        stale_processing = and_(
            TaskReminder.status == "processing",
            TaskReminder.claimed_at <= stale_before,
        )
        async with self.db.session() as session:
            candidate_ids = list(
                (
                    await session.scalars(
                        select(TaskReminder.id)
                        .where(or_(due_pending, stale_processing))
                        .order_by(TaskReminder.remind_at, TaskReminder.id)
                        .limit(self.batch_size)
                    )
                ).all()
            )

        claimed: list[ClaimedReminder] = []
        for reminder_id in candidate_ids:
            token = str(uuid4())
            async with self.db.session() as session:
                changed = await session.execute(
                    update(TaskReminder)
                    .where(
                        TaskReminder.id == reminder_id,
                        or_(due_pending, stale_processing),
                    )
                    .values(
                        status="processing",
                        claim_token=token,
                        claimed_at=now,
                        next_attempt_at=None,
                        attempt_count=TaskReminder.attempt_count + 1,
                    )
                    .returning(TaskReminder.id)
                )
                if changed.scalar_one_or_none() is None:
                    continue
                reminder = await session.get(TaskReminder, reminder_id)
                item = await session.get(InboxItem, reminder.inbox_item_id)
                claimed.append(
                    ClaimedReminder(
                        id=reminder.id,
                        claim_token=token,
                        chat_id=reminder.chat_id,
                        title=item.title,
                        description=item.description,
                        event_at=as_utc(reminder.event_at),
                        timezone=reminder.timezone,
                        attempt_count=reminder.attempt_count,
                    )
                )
        return claimed

    async def _mark_sent(
        self,
        reminder: ClaimedReminder,
        now: datetime,
        message_id: int | None,
    ) -> bool:
        async with self.db.session() as session:
            changed = await session.execute(
                update(TaskReminder)
                .where(
                    TaskReminder.id == reminder.id,
                    TaskReminder.status == "processing",
                    TaskReminder.claim_token == reminder.claim_token,
                )
                .values(
                    status="sent",
                    sent_at=now,
                    telegram_message_id=message_id,
                    claim_token=None,
                    claimed_at=None,
                    next_attempt_at=None,
                    last_error_type=None,
                )
                .returning(TaskReminder.id)
            )
            return changed.scalar_one_or_none() is not None

    async def _release_after_failure(
        self,
        reminder: ClaimedReminder,
        now: datetime,
        exc: Exception,
    ) -> None:
        backoff_seconds = min(300, 5 * (2 ** min(reminder.attempt_count - 1, 6)))
        async with self.db.session() as session:
            await session.execute(
                update(TaskReminder)
                .where(
                    TaskReminder.id == reminder.id,
                    TaskReminder.status == "processing",
                    TaskReminder.claim_token == reminder.claim_token,
                )
                .values(
                    status="pending",
                    claim_token=None,
                    claimed_at=None,
                    next_attempt_at=now + timedelta(seconds=backoff_seconds),
                    last_error_type=type(exc).__name__[:120],
                )
            )

    @staticmethod
    def _message(reminder: ClaimedReminder) -> str:
        local_event = reminder.event_at.astimezone(ZoneInfo(reminder.timezone))
        when = local_event.strftime("%d.%m.%Y %H:%M")
        description = f"\n{reminder.description}" if reminder.description else ""
        return f"⏰ Напоминание\n{reminder.title}{description}\nКогда: {when} ({reminder.timezone})"
