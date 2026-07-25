from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from future_self.inbox import InboxLifecycleService
from future_self.models import InboxItem, TaskReminder, TaskState
from future_self.repositories import UserRepository


async def create_owner(db, telegram_id: int) -> int:
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(telegram_id, "Europe/Moscow")
        return owner.id


async def create_item(
    db,
    owner_id: int,
    title: str,
    *,
    kind: str = "note",
    source: str = "text",
    raw_text: str | None = None,
    task_status: str = "active",
    reminder_status: str | None = None,
) -> int:
    now = datetime.now(UTC)
    async with db.session() as session:
        item = InboxItem(
            user_id=owner_id,
            kind=kind,
            title=title,
            description=None,
            raw_text=raw_text or title,
            next_step=None,
            resolved_date=None,
            temporal_resolution=None,
            source=source,
            status="confirmed",
        )
        session.add(item)
        await session.flush()
        if kind == "task":
            session.add(
                TaskState(
                    owner_id=owner_id,
                    inbox_item_id=item.id,
                    status=task_status,
                    event_at=now + timedelta(hours=2),
                    timezone="Europe/Moscow",
                    version=1,
                    completed_at=now if task_status == "completed" else None,
                )
            )
            if reminder_status is not None:
                session.add(
                    TaskReminder(
                        inbox_item_id=item.id,
                        telegram_user_id=owner_id,
                        chat_id=owner_id,
                        event_at=now + timedelta(hours=2),
                        remind_at=now + timedelta(hours=1),
                        timezone="Europe/Moscow",
                        delivery_key=f"lifecycle:{item.id}:v1",
                        task_version=1,
                        status=reminder_status,
                        claim_token="active-claim" if reminder_status == "processing" else None,
                        claimed_at=now if reminder_status == "processing" else None,
                        next_attempt_at=now if reminder_status == "processing" else None,
                    )
                )
        await session.flush()
        return item.id


async def test_confirmed_snapshot_is_owner_scoped_and_contains_task_state(db):
    owner_id = await create_owner(db, 81001)
    other_id = await create_owner(db, 81002)
    note_id = await create_item(db, owner_id, "Личная заметка")
    task_id = await create_item(db, owner_id, "Личная задача", kind="task")
    foreign_id = await create_item(db, other_id, "Чужая запись", kind="task")

    service = InboxLifecycleService(db)
    snapshot = await service.confirmed_snapshot(owner_id)

    assert [entry["id"] for entry in snapshot] == [note_id, task_id]
    assert snapshot[0] == {
        "id": note_id,
        "version": 1,
        "title": "Личная заметка",
        "kind": "note",
        "status": "confirmed",
        "pre_trash_status": None,
        "task_version": None,
        "task_status": None,
    }
    assert snapshot[1]["task_version"] == 1
    assert snapshot[1]["task_status"] == "active"
    assert await service.confirmed_snapshot(owner_id, [foreign_id]) == []


async def test_trash_and_restore_keep_task_status_and_leave_reminder_off(db):
    owner_id = await create_owner(db, 81101)
    note_id = await create_item(db, owner_id, "Заметка")
    active_id = await create_item(
        db,
        owner_id,
        "Активная задача",
        kind="task",
        reminder_status="processing",
    )
    completed_id = await create_item(
        db,
        owner_id,
        "Выполненная задача",
        kind="task",
        task_status="completed",
        reminder_status="sent",
    )
    service = InboxLifecycleService(db)

    snapshot = await service.confirmed_snapshot(owner_id)
    result = await service.trash_snapshot(
        owner_id,
        snapshot,
        now=datetime(2026, 7, 25, 12, tzinfo=UTC),
    )
    assert (result.status, result.count) == ("trashed", 3)

    async with db.sessions() as session:
        items = {
            item.id: item
            for item in (
                await session.scalars(
                    select(InboxItem).where(InboxItem.id.in_({note_id, active_id, completed_id}))
                )
            ).all()
        }
        states = {
            state.inbox_item_id: state
            for state in (
                await session.scalars(
                    select(TaskState).where(TaskState.inbox_item_id.in_({active_id, completed_id}))
                )
            ).all()
        }
        reminders = {
            reminder.inbox_item_id: reminder
            for reminder in (
                await session.scalars(
                    select(TaskReminder).where(
                        TaskReminder.inbox_item_id.in_({active_id, completed_id})
                    )
                )
            ).all()
        }
    assert all(item.status == "trashed" and item.version == 2 for item in items.values())
    assert all(item.pre_trash_status == "confirmed" for item in items.values())
    assert states[active_id].status == "active"
    assert states[completed_id].status == "completed"
    assert {state.version for state in states.values()} == {2}
    assert reminders[active_id].status == "cancelled"
    assert reminders[active_id].claim_token is None
    assert reminders[active_id].claimed_at is None
    assert reminders[active_id].next_attempt_at is None
    assert reminders[completed_id].status == "sent"

    trashed = await service.trashed_snapshot(owner_id)
    restored = await service.restore_snapshot(owner_id, trashed)
    assert (restored.status, restored.count) == ("restored", 3)
    async with db.sessions() as session:
        items = list(
            (
                await session.scalars(
                    select(InboxItem).where(InboxItem.id.in_({note_id, active_id, completed_id}))
                )
            ).all()
        )
        states = list(
            (
                await session.scalars(
                    select(TaskState).where(TaskState.inbox_item_id.in_({active_id, completed_id}))
                )
            ).all()
        )
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == active_id)
        )
    assert all(
        item.status == "confirmed"
        and item.version == 3
        and item.trashed_at is None
        and item.pre_trash_status is None
        for item in items
    )
    assert {state.status for state in states} == {"active", "completed"}
    assert {state.version for state in states} == {3}
    assert reminder.status == "cancelled"


async def test_stale_foreign_and_replayed_snapshots_fail_atomically(db):
    owner_id = await create_owner(db, 81201)
    other_id = await create_owner(db, 81202)
    first_id = await create_item(db, owner_id, "Первая")
    second_id = await create_item(db, owner_id, "Вторая")
    service = InboxLifecycleService(db)
    stale = await service.confirmed_snapshot(owner_id)

    async with db.session() as session:
        second = await session.get(InboxItem, second_id)
        second.version += 1
    assert (await service.trash_snapshot(owner_id, stale)).status == "changed"
    async with db.sessions() as session:
        statuses = list(
            await session.scalars(
                select(InboxItem.status)
                .where(InboxItem.id.in_({first_id, second_id}))
                .order_by(InboxItem.id)
            )
        )
    assert statuses == ["confirmed", "confirmed"]

    fresh = await service.confirmed_snapshot(owner_id)
    assert (await service.trash_snapshot(other_id, fresh)).status == "changed"
    assert (await service.trash_snapshot(owner_id, fresh)).status == "trashed"
    assert (await service.trash_snapshot(owner_id, fresh)).status == "changed"
    assert (await service.restore_snapshot(owner_id, fresh)).status == "changed"


async def test_overdue_snapshot_rejects_task_rescheduled_before_confirmation(db):
    owner_id = await create_owner(db, 81211)
    task_id = await create_item(db, owner_id, "Просроченная", kind="task")
    now = datetime.now(UTC)
    async with db.session() as session:
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == task_id))
        state.event_at = now - timedelta(days=1)
        state.version += 1
    service = InboxLifecycleService(db)
    snapshot = await service.overdue_snapshot(owner_id, now=now)
    assert [entry["id"] for entry in snapshot] == [task_id]

    async with db.session() as session:
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == task_id))
        state.event_at = now + timedelta(days=1)
        state.version += 1

    assert (await service.trash_snapshot(owner_id, snapshot)).status == "changed"
    async with db.sessions() as session:
        item = await session.get(InboxItem, task_id)
    assert item.status == "confirmed"


async def test_command_garbage_snapshot_is_conservative_and_owner_scoped(db):
    owner_id = await create_owner(db, 81301)
    other_id = await create_owner(db, 81302)
    first_id = await create_item(
        db,
        owner_id,
        "Удалить все просроченные",
        kind="task",
        source="voice",
    )
    second_id = await create_item(
        db,
        owner_id,
        "Удалить все неактуальные черновики",
        kind="task",
        source="text",
    )
    third_id = await create_item(
        db,
        owner_id,
        "Служебная команда",
        kind="task",
        source="voice",
        raw_text="Удалить все несохранённые задачи",
    )
    fourth_id = await create_item(
        db,
        owner_id,
        "Служебная команда",
        kind="note",
        source="text",
        raw_text="Сохраним инбокс",
    )
    fifth_id = await create_item(
        db,
        owner_id,
        "Служебная команда",
        kind="task",
        source="text",
        raw_text="Удалить все черновики/несохранённые задачи",
    )
    await create_item(
        db,
        owner_id,
        "Удалить все просроченные",
        kind="note",
        source="text",
        raw_text="Запиши фразу: удалить все просроченные",
    )
    await create_item(
        db,
        owner_id,
        "План: удалить все просроченные файлы проекта",
        kind="note",
        source="text",
    )
    await create_item(
        db,
        owner_id,
        "Удалить все просроченные",
        kind="task",
        source="vision",
    )
    await create_item(
        db,
        other_id,
        "Удалить все просроченные",
        kind="task",
        source="voice",
    )

    snapshot = await InboxLifecycleService(db).command_garbage_snapshot(owner_id)
    assert [entry["id"] for entry in snapshot] == [
        first_id,
        second_id,
        third_id,
        fourth_id,
        fifth_id,
    ]


async def test_empty_and_malformed_snapshots_never_mutate(db):
    owner_id = await create_owner(db, 81401)
    item_id = await create_item(db, owner_id, "Сохранить")
    service = InboxLifecycleService(db)

    assert (await service.trash_snapshot(owner_id, [])).status == "empty"
    assert (
        await service.trash_snapshot(owner_id, [{"id": item_id, "version": True}])
    ).status == "changed"
    async with db.sessions() as session:
        item = await session.get(InboxItem, item_id)
    assert item.status == "confirmed"
    assert item.version == 1
