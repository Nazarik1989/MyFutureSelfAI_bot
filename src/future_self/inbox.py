from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .db import Database
from .models import InboxItem, TaskReminder, TaskState, User
from .recurring_reminders import RecurringTaskReminderService

InboxSnapshot = list[dict[str, object]]
LifecycleStatus = Literal["trashed", "restored", "changed", "empty"]


@dataclass(frozen=True, slots=True)
class InboxLifecycleResult:
    status: LifecycleStatus
    count: int = 0


class InboxLifecycleService:
    """Owner-scoped, optimistic lifecycle operations for saved Inbox rows."""

    _SNAPSHOT_KEYS = frozenset(
        {
            "id",
            "version",
            "title",
            "kind",
            "status",
            "pre_trash_status",
            "task_version",
            "task_status",
        }
    )
    _LIVE_STATUSES = frozenset({"confirmed", "archived"})
    _CLEANUP_COMMAND = re.compile(
        r"^(?:пожалуйста )?(?:"
        r"(?:удали|удалить|убери|очисти) (?:все )?(?:"
        r"просроченные(?: задачи)?|"
        r"неактуальные(?: задачи| черновики)?|"
        r"старые задачи|"
        r"черновики(?: несохраненные задачи)?|"
        r"несохраненные(?: задачи| карточки| черновики)?"
        r")|"
        r"(?:сохрани|сохраним)(?: это)?(?: в)? (?:инбокс|inbox)"
        r")(?: пожалуйста)?$"
    )

    def __init__(self, db: Database):
        self.db = db

    async def confirmed_snapshot(
        self,
        owner_id: int,
        item_ids: set[int] | tuple[int, ...] | list[int] | None = None,
    ) -> InboxSnapshot:
        return await self._snapshot(owner_id, "confirmed", item_ids)

    async def live_snapshot(
        self,
        owner_id: int,
        item_ids: set[int] | tuple[int, ...] | list[int] | None = None,
    ) -> InboxSnapshot:
        """Snapshot confirmed and legacy-archived rows for reversible cleanup."""

        return await self._snapshot(owner_id, self._LIVE_STATUSES, item_ids)

    async def overdue_snapshot(
        self,
        owner_id: int,
        *,
        now: datetime | None = None,
    ) -> InboxSnapshot:
        """Build the exact reversible snapshot from one overdue-qualified read."""

        current = self._utc(now or datetime.now(UTC))
        async with self.db.sessions() as session:
            rows = list(
                (
                    await session.execute(
                        select(InboxItem, TaskState, TaskReminder)
                        .join(
                            TaskState,
                            and_(
                                TaskState.inbox_item_id == InboxItem.id,
                                TaskState.owner_id == InboxItem.user_id,
                            ),
                        )
                        .outerjoin(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                        .where(
                            InboxItem.user_id == owner_id,
                            InboxItem.kind == "task",
                            InboxItem.status.in_(self._LIVE_STATUSES),
                            TaskState.status == "active",
                            TaskState.event_at.is_not(None),
                            TaskState.event_at < current,
                        )
                        .order_by(TaskState.event_at, InboxItem.id)
                    )
                ).all()
            )
        return [self._entry(item, state) for item, state, _reminder in rows]

    async def trashed_snapshot(
        self,
        owner_id: int,
        item_ids: set[int] | tuple[int, ...] | list[int] | None = None,
    ) -> InboxSnapshot:
        return await self._snapshot(owner_id, "trashed", item_ids)

    async def command_garbage_snapshot(self, owner_id: int) -> InboxSnapshot:
        """Return only exact, already-saved cleanup commands from text/voice capture."""

        async with self.db.sessions() as session:
            rows = await self._rows(session, owner_id, "confirmed")
            return [
                self._entry(item, state)
                for item, state, _reminder in rows
                if item.source in {"text", "voice"} and self._is_cleanup_command(item.raw_text)
            ]

    async def trash_snapshot(
        self,
        owner_id: int,
        snapshot: InboxSnapshot,
        *,
        now: datetime | None = None,
    ) -> InboxLifecycleResult:
        expected = self._validated_snapshot(snapshot, expected_status=self._LIVE_STATUSES)
        if expected is None:
            return InboxLifecycleResult("changed")
        if not expected:
            return InboxLifecycleResult("empty")
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return InboxLifecycleResult("changed")
            rows = await self._rows(session, owner_id, self._LIVE_STATUSES, set(expected))
            if not self._matches(rows, expected):
                return InboxLifecycleResult("changed")
            self._trash_rows(rows, current)
            await session.flush()
            await self._complete_recurring_for_rows(session, owner_id, rows)
            return InboxLifecycleResult("trashed", len(rows))

    async def trash_command_garbage_snapshot(
        self,
        owner_id: int,
        snapshot: InboxSnapshot,
        *,
        now: datetime | None = None,
    ) -> InboxLifecycleResult:
        """Re-evaluate the full command selector and trash it under one owner lock."""

        expected = self._validated_snapshot(snapshot, expected_status="confirmed")
        if expected is None:
            return InboxLifecycleResult("changed")
        if not expected:
            return InboxLifecycleResult("empty")
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return InboxLifecycleResult("changed")
            live_rows = await self._rows(session, owner_id, "confirmed")
            command_rows = [
                row
                for row in live_rows
                if row[0].source in {"text", "voice"} and self._is_cleanup_command(row[0].raw_text)
            ]
            actual = {item.id: self._entry(item, state) for item, state, _reminder in command_rows}
            if actual != expected:
                return InboxLifecycleResult("changed")
            self._trash_rows(command_rows, current)
            await session.flush()
            await self._complete_recurring_for_rows(session, owner_id, command_rows)
            return InboxLifecycleResult("trashed", len(command_rows))

    async def trash_live_item_in_session(
        self,
        session: AsyncSession,
        owner_id: int,
        item_id: int,
        *,
        owner_locked: bool = False,
        now: datetime | None = None,
    ) -> InboxLifecycleResult:
        """Soft-delete one current live row inside a caller-owned transaction.

        This is the integration point for domains that must validate their own
        optimistic version in the same transaction while retaining Inbox links.
        """

        if not self._positive_int(item_id):
            return InboxLifecycleResult("changed")
        if not owner_locked and not await self._lock_owner(session, owner_id):
            return InboxLifecycleResult("changed")
        rows = await self._rows(session, owner_id, self._LIVE_STATUSES, {item_id})
        if len(rows) != 1:
            return InboxLifecycleResult("changed")
        self._trash_rows(rows, self._utc(now or datetime.now(UTC)))
        await session.flush()
        await self._complete_recurring_for_rows(session, owner_id, rows)
        return InboxLifecycleResult("trashed", 1)

    async def restore_snapshot(
        self,
        owner_id: int,
        snapshot: InboxSnapshot,
    ) -> InboxLifecycleResult:
        expected = self._validated_snapshot(snapshot, expected_status="trashed")
        if expected is None:
            return InboxLifecycleResult("changed")
        if not expected:
            return InboxLifecycleResult("empty")
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return InboxLifecycleResult("changed")
            rows = await self._rows(session, owner_id, "trashed", set(expected))
            if not self._matches(rows, expected):
                return InboxLifecycleResult("changed")
            if any(
                item.pre_trash_status not in self._LIVE_STATUSES for item, _state, _reminder in rows
            ):
                return InboxLifecycleResult("changed")
            await self._complete_recurring_for_rows(session, owner_id, rows)
            for item, state, reminder in rows:
                item.status = item.pre_trash_status
                item.pre_trash_status = None
                item.trashed_at = None
                item.version += 1
                if state is not None:
                    state.version += 1
                # Restore never reactivates a reminder. This is also a defensive
                # terminalization if a legacy or manually repaired row was live.
                self._cancel_live_reminder(reminder)
            await session.flush()
            return InboxLifecycleResult("restored", len(rows))

    async def _snapshot(
        self,
        owner_id: int,
        status: str | frozenset[str],
        item_ids: set[int] | tuple[int, ...] | list[int] | None,
    ) -> InboxSnapshot:
        selected = self._item_ids(item_ids)
        if item_ids is not None and selected is None:
            return []
        if selected == set():
            return []
        async with self.db.sessions() as session:
            rows = await self._rows(session, owner_id, status, selected)
            return [self._entry(item, state) for item, state, _reminder in rows]

    @staticmethod
    async def _rows(
        session: AsyncSession,
        owner_id: int,
        status: str | frozenset[str],
        item_ids: set[int] | None = None,
    ) -> list[tuple[InboxItem, TaskState | None, TaskReminder | None]]:
        status_filter = (
            InboxItem.status == status if isinstance(status, str) else InboxItem.status.in_(status)
        )
        statement = (
            select(InboxItem, TaskState, TaskReminder)
            .outerjoin(
                TaskState,
                and_(
                    TaskState.inbox_item_id == InboxItem.id,
                    TaskState.owner_id == InboxItem.user_id,
                ),
            )
            .outerjoin(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
            .where(InboxItem.user_id == owner_id, status_filter)
            .order_by(InboxItem.id)
        )
        if item_ids is not None:
            statement = statement.where(InboxItem.id.in_(item_ids))
        return list((await session.execute(statement)).all())

    @classmethod
    def _matches(
        cls,
        rows: list[tuple[InboxItem, TaskState | None, TaskReminder | None]],
        expected: dict[int, dict[str, object]],
    ) -> bool:
        actual = {item.id: cls._entry(item, state) for item, state, _reminder in rows}
        return actual == expected

    @staticmethod
    def _entry(item: InboxItem, state: TaskState | None) -> dict[str, object]:
        return {
            "id": item.id,
            "version": item.version,
            "title": item.title,
            "kind": item.kind,
            "status": item.status,
            "pre_trash_status": item.pre_trash_status,
            "task_version": state.version if state is not None else None,
            "task_status": state.status if state is not None else None,
        }

    @classmethod
    def _validated_snapshot(
        cls,
        snapshot: object,
        *,
        expected_status: str | frozenset[str],
    ) -> dict[int, dict[str, object]] | None:
        if not isinstance(snapshot, list):
            return None
        expected: dict[int, dict[str, object]] = {}
        for value in snapshot:
            if not isinstance(value, dict) or set(value) != cls._SNAPSHOT_KEYS:
                return None
            item_id = value.get("id")
            version = value.get("version")
            task_version = value.get("task_version")
            task_status = value.get("task_status")
            if (
                not cls._positive_int(item_id)
                or not cls._positive_int(version)
                or item_id in expected
                or not isinstance(value.get("title"), str)
                or not isinstance(value.get("kind"), str)
                or (
                    value.get("status") != expected_status
                    if isinstance(expected_status, str)
                    else value.get("status") not in expected_status
                )
            ):
                return None
            if (task_version is None) != (task_status is None):
                return None
            if task_version is not None and (
                not cls._positive_int(task_version) or not isinstance(task_status, str)
            ):
                return None
            pre_trash_status = value.get("pre_trash_status")
            if expected_status != "trashed":
                if pre_trash_status is not None:
                    return None
            elif pre_trash_status not in cls._LIVE_STATUSES:
                return None
            expected[item_id] = dict(value)
        return expected

    @staticmethod
    def _item_ids(values: object) -> set[int] | None:
        if values is None:
            return None
        if not isinstance(values, (set, tuple, list)):
            return None
        if any(not InboxLifecycleService._positive_int(value) for value in values):
            return None
        return set(values)

    @staticmethod
    def _positive_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @staticmethod
    async def _lock_owner(session: AsyncSession, owner_id: int) -> bool:
        locked = await session.execute(
            update(User)
            .where(User.id == owner_id)
            .values(updated_at=User.updated_at)
            .returning(User.id)
        )
        return locked.scalar_one_or_none() is not None

    @staticmethod
    def _cancel_live_reminder(reminder: TaskReminder | None) -> None:
        if reminder is None:
            return
        if reminder.status in {"pending", "processing"}:
            reminder.status = "cancelled"
        reminder.claim_token = None
        reminder.claimed_at = None
        reminder.next_attempt_at = None

    @staticmethod
    async def _complete_recurring_for_rows(
        session: AsyncSession,
        owner_id: int,
        rows: list[tuple[InboxItem, TaskState | None, TaskReminder | None]],
    ) -> None:
        for item, _state, _reminder in rows:
            if item.kind != "task":
                continue
            await RecurringTaskReminderService.complete_for_terminal_task_in_session(
                session,
                owner_id,
                item.id,
            )

    @classmethod
    def _trash_rows(
        cls,
        rows: list[tuple[InboxItem, TaskState | None, TaskReminder | None]],
        trashed_at: datetime,
    ) -> None:
        for item, state, reminder in rows:
            item.pre_trash_status = item.status
            item.status = "trashed"
            item.trashed_at = trashed_at
            item.version += 1
            if state is not None:
                state.version += 1
            cls._cancel_live_reminder(reminder)

    @classmethod
    def _is_cleanup_command(cls, value: str) -> bool:
        return bool(cls._CLEANUP_COMMAND.fullmatch(cls._normalize_command(value)))

    @staticmethod
    def _normalize_command(value: str) -> str:
        lowered = value.casefold().replace("ё", "е")
        lowered = re.sub(r"[^a-zа-я0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


__all__ = ["InboxLifecycleResult", "InboxLifecycleService", "InboxSnapshot"]
