from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .dates import DateResolver
from .db import Database
from .models import (
    InboxItem,
    TaskActionToken,
    TaskReminder,
    TaskState,
    User,
    VisionItem,
)
from .reminders import as_utc

TaskBucket = Literal["today", "upcoming", "overdue", "no_due", "completed"]

BUCKET_LABELS: dict[TaskBucket, str] = {
    "today": "Сегодня",
    "upcoming": "Предстоящие",
    "overdue": "Просроченные",
    "no_due": "Без срока",
    "completed": "Выполненные",
}


@dataclass(frozen=True, slots=True)
class TaskRecord:
    state: TaskState
    item: InboxItem
    reminder: TaskReminder | None
    vision_linked: bool


@dataclass(frozen=True, slots=True)
class TaskPage:
    bucket: TaskBucket
    records: tuple[TaskRecord, ...]
    page: int
    pages: int
    total: int


@dataclass(frozen=True, slots=True)
class TaskResult:
    status: str
    record: TaskRecord | None = None
    tokens: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ParsedTaskDateTime:
    status: Literal["resolved", "none", "conflict", "nonexistent"]
    event_at: datetime | None = None
    timezone: str | None = None
    local_date: date | None = None
    local_time: time | None = None
    precision: Literal["date", "datetime"] = "datetime"
    message: str | None = None


def _valid_timezone(value: str | None, fallback: str = "Europe/Moscow") -> str:
    for candidate in (value, fallback, "UTC"):
        if not isinstance(candidate, str) or not candidate:
            continue
        try:
            ZoneInfo(candidate)
        except (TypeError, ZoneInfoNotFoundError):
            continue
        return candidate
    return "UTC"


def _parsed_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return as_utc(value)
    if not isinstance(value, str):
        return None
    try:
        return as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def task_state_for_inbox_item(
    item: InboxItem,
    *,
    owner_timezone: str,
    reminder: TaskReminder | None = None,
    date_event_hour: int = 9,
) -> TaskState:
    """Build the canonical state for a task without changing its source record."""
    temporal = item.temporal_resolution if isinstance(item.temporal_resolution, dict) else {}
    timezone = _valid_timezone(
        reminder.timezone if reminder is not None else temporal.get("timezone"),
        owner_timezone,
    )
    event_at = as_utc(reminder.event_at) if reminder is not None else None
    if event_at is None:
        event_at = _parsed_datetime(temporal.get("resolved_at"))
    if event_at is None and item.resolved_date is not None:
        local = datetime.combine(
            item.resolved_date,
            time(hour=date_event_hour),
            tzinfo=ZoneInfo(timezone),
        )
        event_at = local.astimezone(UTC)
    return TaskState(
        owner_id=item.user_id,
        inbox_item_id=item.id,
        status="active",
        event_at=event_at,
        timezone=timezone,
        version=1,
    )


async def add_task_state(
    session: AsyncSession,
    item: InboxItem,
    *,
    owner_timezone: str,
    reminder: TaskReminder | None = None,
    date_event_hour: int = 9,
) -> TaskState | None:
    """Atomically attach TaskState to a newly-created task."""
    if item.kind != "task":
        return None
    state = task_state_for_inbox_item(
        item,
        owner_timezone=owner_timezone,
        reminder=reminder,
        date_event_hour=date_event_hour,
    )
    session.add(state)
    await session.flush()
    return state


class TaskService:
    PAGE_SIZE = 6
    ACTION_TTL = timedelta(minutes=15)
    INPUT_TTL = timedelta(minutes=20)

    def __init__(
        self,
        db: Database,
        *,
        date_event_hour: int = 9,
        reminder_lead_minutes: int = 30,
        date_resolver: DateResolver | None = None,
    ):
        self.db = db
        self.date_event_hour = date_event_hour
        self.reminder_lead_minutes = reminder_lead_minutes
        self.date_resolver = date_resolver or DateResolver()

    async def reconcile(self) -> int:
        """Idempotently add state only to owner-matched task inbox rows."""
        async with self.db.session() as session:
            rows = (
                await session.execute(
                    select(InboxItem, User, TaskReminder)
                    .join(User, User.id == InboxItem.user_id)
                    .outerjoin(TaskState, TaskState.inbox_item_id == InboxItem.id)
                    .outerjoin(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                    .where(InboxItem.kind == "task", TaskState.id.is_(None))
                    .order_by(InboxItem.id)
                )
            ).all()
            for item, owner, reminder in rows:
                await add_task_state(
                    session,
                    item,
                    owner_timezone=owner.timezone,
                    reminder=reminder,
                    date_event_hour=self.date_event_hour,
                )
            return len(rows)

    async def cleanup_tokens(self, *, now: datetime | None = None) -> int:
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            result = await session.execute(
                delete(TaskActionToken).where(TaskActionToken.expires_at <= current)
            )
            return int(result.rowcount or 0)

    async def list_page(
        self,
        owner_id: int,
        bucket: TaskBucket,
        page: int,
        *,
        now: datetime | None = None,
    ) -> TaskPage:
        if bucket not in BUCKET_LABELS:
            raise ValueError("Unknown task bucket")
        current = as_utc(now or datetime.now(UTC))
        async with self.db.sessions() as session:
            rows = (
                await session.execute(
                    select(TaskState, InboxItem, TaskReminder)
                    .join(InboxItem, InboxItem.id == TaskState.inbox_item_id)
                    .outerjoin(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                    .where(
                        TaskState.owner_id == owner_id,
                        InboxItem.user_id == owner_id,
                        InboxItem.kind == "task",
                    )
                )
            ).all()
            vision_ids = set(
                (
                    await session.scalars(
                        select(VisionItem.linked_task_id).where(
                            VisionItem.owner_id == owner_id,
                            VisionItem.linked_task_id.is_not(None),
                        )
                    )
                ).all()
            )
        records = [
            TaskRecord(state, item, reminder, item.id in vision_ids)
            for state, item, reminder in rows
        ]
        records = [record for record in records if self._in_bucket(record, bucket, current)]
        records.sort(key=lambda record: self._sort_key(record, bucket))
        total = len(records)
        pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        safe_page = min(max(page, 0), pages - 1)
        start = safe_page * self.PAGE_SIZE
        return TaskPage(
            bucket=bucket,
            records=tuple(records[start : start + self.PAGE_SIZE]),
            page=safe_page,
            pages=pages,
            total=total,
        )

    async def record(self, owner_id: int, inbox_item_id: int) -> TaskRecord | None:
        async with self.db.sessions() as session:
            return await self._record(session, owner_id, inbox_item_id)

    async def issue_actions(
        self,
        owner_id: int,
        chat_id: int,
        inbox_item_id: int,
        version: int,
        actions: tuple[str, ...],
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        async with self.db.session() as session:
            state = await session.scalar(
                select(TaskState).where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == inbox_item_id,
                    TaskState.version == version,
                )
            )
            if state is None:
                return {}
            return {
                action: await self._new_token(
                    session,
                    owner_id,
                    chat_id,
                    inbox_item_id,
                    version,
                    action,
                    payload=payload,
                )
                for action in actions
            }

    async def open_from_token(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            capability, state = await self._consume_token(
                session, token, owner_id, chat_id, {"view"}
            )
            if capability is None or state is None:
                return TaskResult("stale")
            record = await self._record(session, owner_id, state.inbox_item_id)
            metadata = {
                key: str(value)
                for key, value in (capability.payload or {}).items()
                if key in {"bucket", "page"}
            }
            return TaskResult("ok", record, metadata)

    async def capability_action(self, token: str, owner_id: int, chat_id: int) -> str | None:
        async with self.db.sessions() as session:
            capability = await self._token(session, token, owner_id, chat_id)
            if capability is None or as_utc(capability.expires_at) <= datetime.now(UTC):
                return None
            return capability.action

    async def complete(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            capability = await self._token(session, token, owner_id, chat_id)
            if capability is not None and capability.status == "consumed":
                state = await session.scalar(
                    select(TaskState).where(
                        TaskState.owner_id == owner_id,
                        TaskState.inbox_item_id == capability.inbox_item_id,
                    )
                )
                if (
                    capability.action == "complete"
                    and state is not None
                    and state.status == "completed"
                ):
                    return TaskResult(
                        "already_completed",
                        await self._record(session, owner_id, state.inbox_item_id),
                    )
                return TaskResult("stale")
            capability, state = await self._consume_token(
                session, token, owner_id, chat_id, {"complete"}
            )
            if capability is None or state is None:
                return TaskResult("stale")
            if state.status == "completed":
                return TaskResult(
                    "already_completed", await self._record(session, owner_id, state.inbox_item_id)
                )
            if state.status != "active":
                return TaskResult("stale")
            now = datetime.now(UTC)
            state.status = "completed"
            state.completed_at = now
            state.cancelled_at = None
            state.version += 1
            await self._cancel_live_reminder(session, state.inbox_item_id)
            await session.flush()
            return TaskResult(
                "completed", await self._record(session, owner_id, state.inbox_item_id)
            )

    async def reopen(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            capability, state = await self._consume_token(
                session, token, owner_id, chat_id, {"reopen"}
            )
            if capability is None or state is None or state.status != "completed":
                return TaskResult("stale")
            state.status = "active"
            state.completed_at = None
            state.cancelled_at = None
            state.version += 1
            await self._cancel_live_reminder(session, state.inbox_item_id)
            await session.flush()
            return TaskResult(
                "reopened", await self._record(session, owner_id, state.inbox_item_id)
            )

    async def disable_reminder(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            capability, state = await self._consume_token(
                session, token, owner_id, chat_id, {"reminder_off"}
            )
            if capability is None or state is None or state.status != "active":
                return TaskResult("stale")
            reminder = await session.scalar(
                select(TaskReminder).where(TaskReminder.inbox_item_id == state.inbox_item_id)
            )
            if reminder is None or reminder.status not in {"pending", "processing"}:
                return TaskResult(
                    "already_off", await self._record(session, owner_id, state.inbox_item_id)
                )
            state.version += 1
            self._cancel_reminder(reminder)
            reminder.task_version = state.version
            await session.flush()
            return TaskResult(
                "disabled", await self._record(session, owner_id, state.inbox_item_id)
            )

    async def reschedule_menu(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            capability, state = await self._consume_token(
                session, token, owner_id, chat_id, {"reschedule_menu"}
            )
            if capability is None or state is None or state.status != "active":
                return TaskResult("stale")
            tokens = {
                key: await self._new_token(
                    session,
                    owner_id,
                    chat_id,
                    state.inbox_item_id,
                    state.version,
                    "reschedule_at",
                    payload={"preset": key},
                )
                for key in ("30m", "1h", "tomorrow", "custom")
            }
            return TaskResult(
                "menu", await self._record(session, owner_id, state.inbox_item_id), tokens
            )

    async def choose_reschedule_preset(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
        *,
        now: datetime | None = None,
    ) -> TaskResult:
        current = as_utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            capability = await self._token(session, token, owner_id, chat_id)
            if (
                capability is None
                or capability.action != "reschedule_at"
                or capability.status != "pending"
                or as_utc(capability.expires_at) <= current
            ):
                return TaskResult("stale")
            state = await session.scalar(
                select(TaskState).where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == capability.inbox_item_id,
                    TaskState.version == capability.task_version,
                    TaskState.status == "active",
                )
            )
            if state is None:
                return TaskResult("stale")
            preset = (capability.payload or {}).get("preset")
            if preset == "custom":
                await self._replace_pending_input(session, owner_id, chat_id)
                capability.action = "event_input"
                capability.status = "awaiting_input"
                capability.expires_at = current + self.INPUT_TTL
                capability.payload = None
                return TaskResult(
                    "await_event", await self._record(session, owner_id, state.inbox_item_id)
                )
            zone = ZoneInfo(state.timezone)
            if preset == "30m":
                event_at = current + timedelta(minutes=30)
            elif preset == "1h":
                event_at = current + timedelta(hours=1)
            elif preset == "tomorrow":
                local_tomorrow = current.astimezone(zone).date() + timedelta(days=1)
                event_at = datetime.combine(local_tomorrow, time(9), tzinfo=zone).astimezone(UTC)
            else:
                return TaskResult("stale")
            capability.status = "consumed"
            capability.consumed_at = current
            return await self._prepare_reschedule(
                session,
                state,
                owner_id,
                chat_id,
                event_at,
                state.timezone,
                "datetime",
            )

    async def start_reminder_input(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        current = datetime.now(UTC)
        async with self.db.session() as session:
            capability = await self._token(session, token, owner_id, chat_id)
            if (
                capability is None
                or capability.action != "reminder_edit"
                or capability.status != "pending"
                or as_utc(capability.expires_at) <= current
            ):
                return TaskResult("stale")
            state = await session.scalar(
                select(TaskState).where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == capability.inbox_item_id,
                    TaskState.version == capability.task_version,
                    TaskState.status == "active",
                )
            )
            if state is None:
                return TaskResult("stale")
            await self._replace_pending_input(session, owner_id, chat_id)
            capability.action = "reminder_input"
            capability.status = "awaiting_input"
            capability.expires_at = current + self.INPUT_TTL
            return TaskResult(
                "await_reminder", await self._record(session, owner_id, state.inbox_item_id)
            )

    async def pending_input(self, owner_id: int, chat_id: int) -> TaskActionToken | None:
        current = datetime.now(UTC)
        async with self.db.sessions() as session:
            return await session.scalar(
                select(TaskActionToken)
                .where(
                    TaskActionToken.owner_id == owner_id,
                    TaskActionToken.chat_id == chat_id,
                    TaskActionToken.status == "awaiting_input",
                    TaskActionToken.expires_at > current,
                )
                .order_by(TaskActionToken.created_at.desc(), TaskActionToken.token.desc())
                .limit(1)
            )

    async def cancel_pending_input(self, owner_id: int, chat_id: int) -> TaskRecord | None:
        current = datetime.now(UTC)
        async with self.db.session() as session:
            pending = await session.scalar(
                select(TaskActionToken)
                .where(
                    TaskActionToken.owner_id == owner_id,
                    TaskActionToken.chat_id == chat_id,
                    TaskActionToken.status == "awaiting_input",
                    TaskActionToken.expires_at > current,
                )
                .order_by(TaskActionToken.created_at.desc(), TaskActionToken.token.desc())
                .limit(1)
            )
            if pending is None:
                return None
            pending.status = "consumed"
            pending.consumed_at = current
            return await self._record(session, owner_id, pending.inbox_item_id)

    async def submit_pending_input(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
        parsed: ParsedTaskDateTime,
    ) -> TaskResult:
        if parsed.status != "resolved" or parsed.event_at is None or parsed.timezone is None:
            return TaskResult(parsed.status)
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            capability = await self._token(session, token, owner_id, chat_id)
            if (
                capability is None
                or capability.status != "awaiting_input"
                or as_utc(capability.expires_at) <= datetime.now(UTC)
            ):
                return TaskResult("stale")
            state = await session.scalar(
                select(TaskState).where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == capability.inbox_item_id,
                    TaskState.version == capability.task_version,
                    TaskState.status == "active",
                )
            )
            if state is None:
                return TaskResult("stale")
            capability.status = "consumed"
            capability.consumed_at = datetime.now(UTC)
            if capability.action == "event_input":
                return await self._prepare_reschedule(
                    session,
                    state,
                    owner_id,
                    chat_id,
                    parsed.event_at,
                    parsed.timezone,
                    parsed.precision,
                )
            if capability.action == "reminder_input":
                await self._set_explicit_reminder(session, state, parsed.event_at)
                await session.flush()
                return TaskResult(
                    "reminder_changed", await self._record(session, owner_id, state.inbox_item_id)
                )
            return TaskResult("stale")

    async def apply_reschedule_choice(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            capability = await self._token(session, token, owner_id, chat_id)
            if (
                capability is None
                or capability.status != "pending"
                or capability.action
                not in {
                    "reschedule_preserve",
                    "reschedule_new_reminder",
                }
                or as_utc(capability.expires_at) <= datetime.now(UTC)
            ):
                return TaskResult("stale")
            state = await session.scalar(
                select(TaskState).where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == capability.inbox_item_id,
                    TaskState.version == capability.task_version,
                    TaskState.status == "active",
                )
            )
            payload = capability.payload or {}
            event_at = _parsed_datetime(payload.get("event_at"))
            timezone = _valid_timezone(payload.get("timezone"), state.timezone if state else "UTC")
            precision = "date" if payload.get("precision") == "date" else "datetime"
            if state is None or event_at is None:
                return TaskResult("stale")
            capability.status = "consumed"
            capability.consumed_at = datetime.now(UTC)
            preserve = capability.action == "reschedule_preserve"
            await self._apply_event_change(
                session, state, event_at, timezone, precision, preserve=preserve
            )
            await session.flush()
            if preserve:
                return TaskResult(
                    "rescheduled", await self._record(session, owner_id, state.inbox_item_id)
                )
            token_value = await self._new_token(
                session,
                owner_id,
                chat_id,
                state.inbox_item_id,
                state.version,
                "reminder_input",
                status="awaiting_input",
                ttl=self.INPUT_TTL,
            )
            return TaskResult(
                "await_reminder",
                await self._record(session, owner_id, state.inbox_item_id),
                {"input": token_value},
            )

    async def prepare_delete(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            capability, state = await self._consume_token(
                session, token, owner_id, chat_id, {"delete_ask"}
            )
            if capability is None or state is None:
                return TaskResult("stale")
            tokens = {
                action: await self._new_token(
                    session,
                    owner_id,
                    chat_id,
                    state.inbox_item_id,
                    state.version,
                    action,
                )
                for action in ("delete_confirm", "delete_cancel")
            }
            return TaskResult(
                "confirm_delete", await self._record(session, owner_id, state.inbox_item_id), tokens
            )

    async def delete_or_cancel(self, token: str, owner_id: int, chat_id: int) -> TaskResult:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            capability = await self._token(session, token, owner_id, chat_id)
            if (
                capability is None
                or capability.status != "pending"
                or capability.action
                not in {
                    "delete_confirm",
                    "delete_cancel",
                }
                or as_utc(capability.expires_at) <= datetime.now(UTC)
            ):
                return TaskResult("stale")
            state = await session.scalar(
                select(TaskState).where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == capability.inbox_item_id,
                    TaskState.version == capability.task_version,
                )
            )
            if state is None:
                return TaskResult("stale")
            if capability.action == "delete_cancel":
                capability.status = "consumed"
                capability.consumed_at = datetime.now(UTC)
                return TaskResult(
                    "delete_cancelled", await self._record(session, owner_id, state.inbox_item_id)
                )
            item_id = state.inbox_item_id
            await session.execute(
                update(VisionItem)
                .where(VisionItem.owner_id == owner_id, VisionItem.linked_task_id == item_id)
                .values(linked_task_id=None)
            )
            await session.execute(
                update(TaskReminder)
                .where(TaskReminder.inbox_item_id == item_id)
                .values(
                    status="cancelled",
                    claim_token=None,
                    claimed_at=None,
                    next_attempt_at=None,
                )
            )
            item = await session.scalar(
                select(InboxItem).where(
                    InboxItem.id == item_id,
                    InboxItem.user_id == owner_id,
                    InboxItem.kind == "task",
                )
            )
            if item is None:
                return TaskResult("stale")
            await session.delete(item)
            await session.flush()
            return TaskResult("deleted")

    def parse_datetime(
        self,
        text: str,
        timezone: str,
        *,
        now: datetime | None = None,
    ) -> ParsedTaskDateTime:
        timezone = _valid_timezone(timezone)
        current = as_utc(now or datetime.now(UTC))
        clean = " ".join(text.strip().split())
        if not clean:
            return ParsedTaskDateTime("none")
        relative_text = clean if clean.lower().startswith("напомни") else f"Напомни {clean} задачу"
        relative = self.date_resolver.resolve_relative_reminder(
            relative_text,
            timezone,
            now=current,
        )
        if relative is not None:
            local = relative.remind_at.astimezone(ZoneInfo(timezone))
            return ParsedTaskDateTime(
                "resolved",
                relative.remind_at,
                timezone,
                local.date(),
                local.time().replace(tzinfo=None),
                "datetime",
            )
        resolved = self.date_resolver.resolve(clean, timezone, now=current)
        if resolved.status == "conflict":
            return ParsedTaskDateTime(
                "conflict",
                timezone=timezone,
                message=self.date_resolver.conflict_message(resolved),
            )
        if resolved.status != "resolved" or resolved.target_date is None:
            return ParsedTaskDateTime("none", timezone=timezone)
        local_time = self.date_resolver.extract_local_time(clean)
        if local_time is None:
            return ParsedTaskDateTime(
                "none",
                timezone=timezone,
                message="Укажи и дату, и время, например: завтра в 18:00.",
            )
        zone = ZoneInfo(timezone)
        local = datetime.combine(resolved.target_date, local_time, tzinfo=zone)
        event_at = local.astimezone(UTC)
        if event_at.astimezone(zone).replace(tzinfo=None) != local.replace(tzinfo=None):
            return ParsedTaskDateTime(
                "nonexistent",
                timezone=timezone,
                message="Такого локального времени нет из-за перехода часов. Выбери другое.",
            )
        return ParsedTaskDateTime(
            "resolved",
            event_at,
            timezone,
            resolved.target_date,
            local_time,
            "datetime",
        )

    async def _prepare_reschedule(
        self,
        session: AsyncSession,
        state: TaskState,
        owner_id: int,
        chat_id: int,
        event_at: datetime,
        timezone: str,
        precision: str,
    ) -> TaskResult:
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == state.inbox_item_id)
        )
        if reminder is not None and reminder.status != "cancelled":
            payload = {
                "event_at": as_utc(event_at).isoformat(),
                "timezone": timezone,
                "precision": precision,
            }
            tokens = {
                action: await self._new_token(
                    session,
                    owner_id,
                    chat_id,
                    state.inbox_item_id,
                    state.version,
                    action,
                    payload=payload,
                )
                for action in ("reschedule_preserve", "reschedule_new_reminder")
            }
            return TaskResult(
                "choose_reminder",
                await self._record(session, owner_id, state.inbox_item_id),
                tokens,
            )
        await self._apply_event_change(
            session,
            state,
            event_at,
            timezone,
            precision,
            preserve=False,
        )
        await session.flush()
        return TaskResult("rescheduled", await self._record(session, owner_id, state.inbox_item_id))

    async def _apply_event_change(
        self,
        session: AsyncSession,
        state: TaskState,
        event_at: datetime,
        timezone: str,
        precision: str,
        *,
        preserve: bool,
    ) -> None:
        event_at = as_utc(event_at)
        old_event = as_utc(state.event_at) if state.event_at is not None else None
        state.event_at = event_at
        state.timezone = _valid_timezone(timezone, state.timezone)
        state.version += 1
        item = await session.get(InboxItem, state.inbox_item_id)
        if item is None:
            raise ValueError("Task inbox item disappeared")
        local = event_at.astimezone(ZoneInfo(state.timezone))
        item.resolved_date = local.date()
        item.temporal_resolution = {
            "resolved_at": event_at.isoformat(),
            "remind_at": None,
            "timezone": state.timezone,
            "resolved_local_date": local.date().isoformat(),
            "resolved_local_time": local.time().replace(tzinfo=None).isoformat(),
            "precision": "date" if precision == "date" else "datetime",
            "original_expression": "task_hub_reschedule",
            "resolution_status": "resolved",
        }
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == state.inbox_item_id)
        )
        if reminder is None:
            return
        if preserve and reminder.status != "cancelled":
            lead = (
                old_event - as_utc(reminder.remind_at)
                if old_event is not None
                else timedelta(minutes=self.reminder_lead_minutes)
            )
            reminder.event_at = event_at
            reminder.remind_at = event_at - lead
            self._activate_reminder(reminder, state.version)
            item.temporal_resolution["remind_at"] = as_utc(reminder.remind_at).isoformat()
        else:
            self._cancel_reminder(reminder)
            reminder.task_version = state.version

    async def _set_explicit_reminder(
        self,
        session: AsyncSession,
        state: TaskState,
        remind_at: datetime,
    ) -> None:
        remind_at = as_utc(remind_at)
        state.version += 1
        item = await session.get(InboxItem, state.inbox_item_id)
        owner = await session.get(User, state.owner_id)
        if item is None or owner is None:
            raise ValueError("Task owner or inbox item disappeared")
        if state.event_at is None:
            state.event_at = remind_at
            local = remind_at.astimezone(ZoneInfo(state.timezone))
            item.resolved_date = local.date()
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == state.inbox_item_id)
        )
        event_at = as_utc(state.event_at)
        if reminder is None:
            reminder = TaskReminder(
                inbox_item_id=state.inbox_item_id,
                telegram_user_id=owner.telegram_id,
                chat_id=owner.telegram_id,
                event_at=event_at,
                remind_at=remind_at,
                timezone=state.timezone,
                delivery_key=f"task:{state.inbox_item_id}:reminder:v{state.version}",
                task_version=state.version,
                status="pending",
            )
            session.add(reminder)
        else:
            reminder.event_at = event_at
            reminder.remind_at = remind_at
            reminder.timezone = state.timezone
            self._activate_reminder(reminder, state.version)
        local = event_at.astimezone(ZoneInfo(state.timezone))
        item.temporal_resolution = {
            "resolved_at": event_at.isoformat(),
            "remind_at": remind_at.isoformat(),
            "timezone": state.timezone,
            "resolved_local_date": local.date().isoformat(),
            "resolved_local_time": local.time().replace(tzinfo=None).isoformat(),
            "precision": "datetime",
            "original_expression": "task_hub_reminder",
            "resolution_status": "resolved",
        }

    @staticmethod
    def _activate_reminder(reminder: TaskReminder, task_version: int) -> None:
        reminder.status = "pending"
        reminder.task_version = task_version
        reminder.delivery_key = f"task:{reminder.inbox_item_id}:reminder:v{task_version}"
        reminder.claim_token = None
        reminder.claimed_at = None
        reminder.next_attempt_at = None
        reminder.attempt_count = 0
        reminder.sent_at = None
        reminder.telegram_message_id = None
        reminder.last_error_type = None

    @staticmethod
    def _cancel_reminder(reminder: TaskReminder) -> None:
        if reminder.status in {"pending", "processing"}:
            reminder.status = "cancelled"
        reminder.claim_token = None
        reminder.claimed_at = None
        reminder.next_attempt_at = None

    async def _cancel_live_reminder(self, session: AsyncSession, inbox_item_id: int) -> None:
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == inbox_item_id)
        )
        if reminder is not None:
            self._cancel_reminder(reminder)

    async def _record(
        self, session: AsyncSession, owner_id: int, inbox_item_id: int
    ) -> TaskRecord | None:
        row = (
            await session.execute(
                select(TaskState, InboxItem, TaskReminder)
                .join(InboxItem, InboxItem.id == TaskState.inbox_item_id)
                .outerjoin(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                .where(
                    TaskState.owner_id == owner_id,
                    TaskState.inbox_item_id == inbox_item_id,
                    InboxItem.user_id == owner_id,
                    InboxItem.kind == "task",
                )
            )
        ).one_or_none()
        if row is None:
            return None
        state, item, reminder = row
        linked = (
            await session.scalar(
                select(VisionItem.id).where(
                    VisionItem.owner_id == owner_id,
                    VisionItem.linked_task_id == inbox_item_id,
                )
            )
        ) is not None
        return TaskRecord(state, item, reminder, linked)

    async def _token(
        self, session: AsyncSession, token: str, owner_id: int, chat_id: int
    ) -> TaskActionToken | None:
        return await session.scalar(
            select(TaskActionToken).where(
                TaskActionToken.token == token,
                TaskActionToken.owner_id == owner_id,
                TaskActionToken.chat_id == chat_id,
            )
        )

    async def _consume_token(
        self,
        session: AsyncSession,
        token: str,
        owner_id: int,
        chat_id: int,
        actions: set[str],
    ) -> tuple[TaskActionToken | None, TaskState | None]:
        capability = await self._token(session, token, owner_id, chat_id)
        current = datetime.now(UTC)
        if (
            capability is None
            or capability.action not in actions
            or capability.status != "pending"
            or as_utc(capability.expires_at) <= current
        ):
            return None, None
        state = await session.scalar(
            select(TaskState).where(
                TaskState.owner_id == owner_id,
                TaskState.inbox_item_id == capability.inbox_item_id,
                TaskState.version == capability.task_version,
            )
        )
        if state is None:
            return None, None
        capability.status = "consumed"
        capability.consumed_at = current
        return capability, state

    async def _new_token(
        self,
        session: AsyncSession,
        owner_id: int,
        chat_id: int,
        inbox_item_id: int,
        version: int,
        action: str,
        *,
        payload: dict[str, Any] | None = None,
        status: str = "pending",
        ttl: timedelta | None = None,
    ) -> str:
        token = secrets.token_urlsafe(18)
        session.add(
            TaskActionToken(
                token=token,
                owner_id=owner_id,
                chat_id=chat_id,
                inbox_item_id=inbox_item_id,
                task_version=version,
                action=action,
                payload=payload,
                status=status,
                expires_at=datetime.now(UTC) + (ttl or self.ACTION_TTL),
            )
        )
        await session.flush()
        return token

    async def _replace_pending_input(
        self, session: AsyncSession, owner_id: int, chat_id: int
    ) -> None:
        await session.execute(
            update(TaskActionToken)
            .where(
                TaskActionToken.owner_id == owner_id,
                TaskActionToken.chat_id == chat_id,
                TaskActionToken.status == "awaiting_input",
            )
            .values(status="consumed", consumed_at=datetime.now(UTC))
        )

    @staticmethod
    async def _lock_owner(session: AsyncSession, owner_id: int) -> None:
        await session.execute(
            update(User).where(User.id == owner_id).values(updated_at=User.updated_at)
        )

    @staticmethod
    def _in_bucket(record: TaskRecord, bucket: TaskBucket, now: datetime) -> bool:
        state = record.state
        if bucket == "completed":
            return state.status == "completed"
        if state.status != "active":
            return False
        event = as_utc(state.event_at) if state.event_at is not None else None
        if bucket == "no_due":
            return event is None
        if event is None:
            return False
        if bucket == "overdue":
            return event < now
        zone = ZoneInfo(_valid_timezone(state.timezone))
        local_event_date = event.astimezone(zone).date()
        local_today = now.astimezone(zone).date()
        if bucket == "today":
            return local_event_date == local_today
        return local_event_date > local_today

    @staticmethod
    def _sort_key(record: TaskRecord, bucket: TaskBucket) -> tuple[float, int]:
        if bucket == "completed":
            completed = record.state.completed_at or record.state.updated_at
            return (-as_utc(completed).timestamp(), -record.item.id)
        event = record.state.event_at
        timestamp = as_utc(event).timestamp() if event is not None else float("inf")
        return (timestamp, record.item.id)
