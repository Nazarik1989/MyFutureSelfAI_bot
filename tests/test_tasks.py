import asyncio
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select

from future_self.models import InboxItem, TaskActionToken, TaskReminder, TaskState, VisionItem
from future_self.reminders import TaskReminderEngine, as_utc
from future_self.repositories import UserRepository
from future_self.tasks import TaskService, add_task_state


async def create_task(
    db,
    *,
    telegram_id: int = 100,
    title: str = "Задача",
    event_at: datetime | None = None,
    remind_at: datetime | None = None,
    timezone: str = "Europe/Moscow",
    precision: str = "datetime",
    source: str = "text",
):
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(telegram_id, timezone)
        local = event_at.astimezone(__import__("zoneinfo").ZoneInfo(timezone)) if event_at else None
        item = InboxItem(
            user_id=owner.id,
            kind="task",
            title=title,
            description="Описание",
            raw_text=title,
            next_step=None,
            resolved_date=local.date() if local else None,
            temporal_resolution=(
                {
                    "resolved_at": event_at.isoformat(),
                    "remind_at": remind_at.isoformat() if remind_at else None,
                    "timezone": timezone,
                    "resolved_local_date": local.date().isoformat(),
                    "resolved_local_time": local.time().replace(tzinfo=None).isoformat(),
                    "precision": precision,
                    "original_expression": "test",
                    "resolution_status": "resolved",
                }
                if event_at
                else None
            ),
            source=source,
            status="confirmed",
        )
        session.add(item)
        await session.flush()
        reminder = None
        if remind_at is not None:
            reminder = TaskReminder(
                inbox_item_id=item.id,
                telegram_user_id=telegram_id,
                chat_id=telegram_id,
                event_at=event_at,
                remind_at=remind_at,
                timezone=timezone,
                delivery_key=f"test:{item.id}:v1",
                task_version=1,
                status="pending",
            )
            session.add(reminder)
        state = await add_task_state(
            session,
            item,
            owner_timezone=timezone,
            reminder=reminder,
        )
        return owner.id, item.id, state.id


async def token(service, owner_id, item_id, action, *, chat_id=100):
    return (await service.issue_actions(owner_id, chat_id, item_id, 1, (action,)))[action]


async def test_reconciliation_is_idempotent_and_uses_canonical_precedence(db):
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(200, "Europe/Moscow")
        item = InboxItem(
            user_id=owner.id,
            kind="task",
            title="Старая задача",
            description=None,
            raw_text="legacy",
            next_step=None,
            resolved_date=date(2026, 7, 22),
            temporal_resolution={
                "resolved_at": "2026-07-22T08:00:00+00:00",
                "timezone": "Europe/Moscow",
            },
            source="text",
            status="confirmed",
        )
        idea = InboxItem(
            user_id=owner.id,
            kind="idea",
            title="Не задача",
            description=None,
            raw_text="idea",
            next_step=None,
            resolved_date=None,
            temporal_resolution=None,
            source="text",
            status="confirmed",
        )
        session.add_all([item, idea])
        await session.flush()
        session.add(
            TaskReminder(
                inbox_item_id=item.id,
                telegram_user_id=200,
                chat_id=200,
                event_at=datetime(2026, 7, 22, 9, tzinfo=UTC),
                remind_at=datetime(2026, 7, 22, 8, 30, tzinfo=UTC),
                timezone="Europe/Saratov",
                delivery_key="legacy:v1",
                task_version=1,
                status="pending",
            )
        )
        item_id = item.id
    service = TaskService(db)
    assert await service.reconcile() == 1
    assert await service.reconcile() == 0
    async with db.sessions() as session:
        states = (await session.scalars(select(TaskState))).all()
    assert len(states) == 1
    assert states[0].inbox_item_id == item_id
    assert as_utc(states[0].event_at) == datetime(2026, 7, 22, 9, tzinfo=UTC)
    assert states[0].timezone == "Europe/Saratov"


async def test_task_lists_use_owner_timezone_boundary_stable_sort_and_pagination(db):
    now = datetime(2026, 7, 21, 20, 30, tzinfo=UTC)  # 00:30 on July 22 in Saratov
    owner_id = None
    expected = []
    for index in range(8):
        owner_id, item_id, _ = await create_task(
            db,
            telegram_id=300,
            title=f"Сегодня {index}",
            event_at=now + timedelta(hours=index + 1),
            timezone="Europe/Saratov",
        )
        expected.append(item_id)
    _, overdue_id, _ = await create_task(
        db,
        telegram_id=300,
        title="Просрочена",
        event_at=now - timedelta(minutes=1),
        timezone="Europe/Saratov",
    )
    _, future_id, _ = await create_task(
        db,
        telegram_id=300,
        title="Завтра",
        event_at=now + timedelta(days=1),
        timezone="Europe/Saratov",
    )
    _, no_due_id, _ = await create_task(db, telegram_id=300, title="Без срока")
    service = TaskService(db)
    first = await service.list_page(owner_id, "today", 0, now=now)
    second = await service.list_page(owner_id, "today", 1, now=now)
    assert first.total == 9  # includes today's already-overdue task by calendar date
    assert first.pages == 2
    assert [record.item.id for record in first.records] == [overdue_id, *expected[:5]]
    assert [record.item.id for record in second.records] == expected[5:]
    assert [
        record.item.id
        for record in (await service.list_page(owner_id, "overdue", 0, now=now)).records
    ] == [overdue_id]
    assert [
        record.item.id
        for record in (await service.list_page(owner_id, "upcoming", 0, now=now)).records
    ] == [future_id]
    assert [
        record.item.id
        for record in (await service.list_page(owner_id, "no_due", 0, now=now)).records
    ] == [no_due_id]


async def test_complete_is_idempotent_cancels_claim_and_reopen_keeps_reminder_off(db):
    event = datetime.now(UTC) + timedelta(hours=2)
    owner_id, item_id, _ = await create_task(
        db,
        event_at=event,
        remind_at=event - timedelta(minutes=30),
    )
    service = TaskService(db)
    complete_token = await token(service, owner_id, item_id, "complete")
    first = await service.complete(complete_token, owner_id, 100)
    replay = await service.complete(complete_token, owner_id, 100)
    assert first.status == "completed"
    assert replay.status == "already_completed"
    async with db.sessions() as session:
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == item_id)
        )
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item_id))
    assert reminder.status == "cancelled"
    assert reminder.claim_token is None
    assert state.version == 2
    reopen_token = (await service.issue_actions(owner_id, 100, item_id, 2, ("reopen",)))["reopen"]
    reopened = await service.reopen(reopen_token, owner_id, 100)
    assert reopened.status == "reopened"
    assert reopened.record.reminder.status == "cancelled"
    assert reopened.record.state.version == 3


async def test_reschedule_preserves_interval_and_invalidates_competing_callback(db):
    now = datetime(2026, 7, 21, 9, tzinfo=UTC)
    event = now + timedelta(hours=3)
    owner_id, item_id, _ = await create_task(
        db,
        event_at=event,
        remind_at=event - timedelta(minutes=45),
    )
    service = TaskService(db)
    actions = await service.issue_actions(
        owner_id, 100, item_id, 1, ("reschedule_menu", "complete")
    )
    menu = await service.reschedule_menu(actions["reschedule_menu"], owner_id, 100)
    proposed = await service.choose_reschedule_preset(menu.tokens["1h"], owner_id, 100, now=now)
    assert proposed.status == "choose_reminder"
    changed = await service.apply_reschedule_choice(
        proposed.tokens["reschedule_preserve"], owner_id, 100
    )
    assert changed.status == "rescheduled"
    assert as_utc(changed.record.state.event_at) == now + timedelta(hours=1)
    assert as_utc(changed.record.reminder.remind_at) == now + timedelta(minutes=15)
    assert changed.record.reminder.task_version == 2
    assert (await service.complete(actions["complete"], owner_id, 100)).status == "stale"


async def test_custom_event_and_reminder_inputs_are_persistent_and_deterministic(db):
    now = datetime.now(UTC)
    owner_id, item_id, _ = await create_task(db, event_at=now + timedelta(days=1))
    first_service = TaskService(db)
    menu_token = await token(first_service, owner_id, item_id, "reschedule_menu")
    menu = await first_service.reschedule_menu(menu_token, owner_id, 100)
    pending = await first_service.choose_reschedule_preset(
        menu.tokens["custom"], owner_id, 100, now=now
    )
    assert pending.status == "await_event"
    # New service instance simulates a process restart; input state lives in SQLite.
    restarted = TaskService(db)
    stored = await restarted.pending_input(owner_id, 100)
    assert stored.action == "event_input"
    parsed = restarted.parse_datetime("завтра в 18:00", "Europe/Moscow", now=now)
    result = await restarted.submit_pending_input(stored.token, owner_id, 100, parsed)
    assert result.status == "rescheduled"
    assert result.record.state.event_at is not None


async def test_explicit_reminder_after_sent_gets_new_delivery_key_and_can_be_disabled(db):
    now = datetime(2026, 7, 21, 8, tzinfo=UTC)
    owner_id, item_id, _ = await create_task(
        db,
        event_at=now + timedelta(days=1),
        remind_at=now + timedelta(hours=1),
    )
    async with db.session() as session:
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == item_id)
        )
        old_key = reminder.delivery_key
        reminder.status = "sent"
        reminder.sent_at = now
    service = TaskService(db)
    edit_token = await token(service, owner_id, item_id, "reminder_edit")
    assert (
        await service.start_reminder_input(edit_token, owner_id, 100)
    ).status == "await_reminder"
    pending = await service.pending_input(owner_id, 100)
    parsed = service.parse_datetime("через 1 час", "Europe/Moscow", now=now)
    changed = await service.submit_pending_input(pending.token, owner_id, 100, parsed)
    assert changed.status == "reminder_changed"
    assert changed.record.reminder.status == "pending"
    assert changed.record.reminder.delivery_key != old_key
    assert changed.record.reminder.task_version == 2
    off = (await service.issue_actions(owner_id, 100, item_id, 2, ("reminder_off",)))[
        "reminder_off"
    ]
    disabled = await service.disable_reminder(off, owner_id, 100)
    assert disabled.status == "disabled"
    assert disabled.record.reminder.status == "cancelled"
    assert as_utc(disabled.record.state.event_at) == now + timedelta(days=1)


async def test_delete_confirmation_is_owner_chat_version_bound_and_clears_vision_link(db):
    owner_id, item_id, _ = await create_task(db)
    async with db.session() as session:
        vision = VisionItem(
            owner_id=owner_id,
            category="other",
            wish_text="Желание",
            why_text=None,
            target_date=None,
            first_step="Шаг",
            status="active",
            linked_task_id=item_id,
        )
        session.add(vision)
        await session.flush()
        vision_id = vision.id
    service = TaskService(db)
    ask = await token(service, owner_id, item_id, "delete_ask")
    confirm = await service.prepare_delete(ask, owner_id, 100)
    assert confirm.status == "confirm_delete"
    assert (
        await service.delete_or_cancel(confirm.tokens["delete_confirm"], owner_id, 999)
    ).status == "stale"
    cancelled = await service.delete_or_cancel(confirm.tokens["delete_cancel"], owner_id, 100)
    assert cancelled.status == "delete_cancelled"
    expired_ask = (await service.issue_actions(owner_id, 100, item_id, 1, ("delete_ask",)))[
        "delete_ask"
    ]
    expired = await service.prepare_delete(expired_ask, owner_id, 100)
    async with db.session() as session:
        confirmation = await session.get(TaskActionToken, expired.tokens["delete_confirm"])
        confirmation.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert (
        await service.delete_or_cancel(expired.tokens["delete_confirm"], owner_id, 100)
    ).status == "stale"
    ask = (await service.issue_actions(owner_id, 100, item_id, 1, ("delete_ask",)))["delete_ask"]
    confirm = await service.prepare_delete(ask, owner_id, 100)
    assert (
        await service.delete_or_cancel(confirm.tokens["delete_confirm"], owner_id, 100)
    ).status == "deleted"
    async with db.sessions() as session:
        assert await session.get(InboxItem, item_id) is None
        assert (await session.get(VisionItem, vision_id)).linked_task_id is None


async def test_tokens_are_owner_isolated_forged_stale_and_single_use(db):
    owner_id, item_id, _ = await create_task(db, telegram_id=400)
    other_id, _, _ = await create_task(db, telegram_id=401)
    service = TaskService(db)
    capability = await token(service, owner_id, item_id, "complete", chat_id=400)
    assert (await service.complete("forged", owner_id, 400)).status == "stale"
    assert (await service.complete(capability, other_id, 400)).status == "stale"
    assert (await service.complete(capability, owner_id, 401)).status == "stale"
    assert (await service.complete(capability, owner_id, 400)).status == "completed"
    assert (await service.complete(capability, owner_id, 400)).status == "already_completed"


async def test_complete_reschedule_delete_race_has_single_winner(db):
    now = datetime.now(UTC)
    owner_id, item_id, _ = await create_task(
        db,
        event_at=now + timedelta(hours=2),
        remind_at=now + timedelta(hours=1),
    )
    service = TaskService(db)
    actions = await service.issue_actions(
        owner_id,
        100,
        item_id,
        1,
        ("complete", "reschedule_menu", "delete_ask"),
    )
    menu = await service.reschedule_menu(actions["reschedule_menu"], owner_id, 100)
    proposal = await service.choose_reschedule_preset(menu.tokens["30m"], owner_id, 100, now=now)
    deletion = await service.prepare_delete(actions["delete_ask"], owner_id, 100)

    async def complete():
        return await service.complete(actions["complete"], owner_id, 100)

    async def reschedule():
        return await service.apply_reschedule_choice(
            proposal.tokens["reschedule_preserve"], owner_id, 100
        )

    async def prepare_delete():
        return await service.delete_or_cancel(deletion.tokens["delete_confirm"], owner_id, 100)

    results = await asyncio.gather(complete(), reschedule(), prepare_delete())
    winners = [
        result for result in results if result.status in {"completed", "rescheduled", "deleted"}
    ]
    assert len(winners) == 1
    async with db.sessions() as session:
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item_id))
    if winners[0].status == "deleted":
        assert state is None
    else:
        assert state.version == 2


async def test_reminder_claim_is_rechecked_against_task_version_before_send(db):
    now = datetime.now(UTC)
    owner_id, item_id, _ = await create_task(
        db,
        event_at=now + timedelta(minutes=1),
        remind_at=now,
    )
    sent = []

    async def send(chat_id, text):
        sent.append((chat_id, text))
        return 1

    engine = TaskReminderEngine(db, send)
    claimed = await engine._claim_due(now)
    assert len(claimed) == 1
    service = TaskService(db)
    complete_token = await token(service, owner_id, item_id, "complete")
    await service.complete(complete_token, owner_id, 100)
    assert not await engine._still_current(claimed[0])
    assert await engine.deliver_due(now=now + timedelta(seconds=1)) == 0
    assert sent == []


async def test_task_state_exists_for_preview_confirm_and_non_tasks_are_untouched(db):
    from future_self.drafts import DraftInboxService
    from future_self.schemas import ParsedThought

    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(500, "Europe/Moscow")
    service = DraftInboxService(db, 60)
    task = await service.create(
        user_id=owner.id,
        telegram_user_id=500,
        chat_id=500,
        source="text",
        raw_text="задача",
        parsed=ParsedThought(kind="task", title="Задача"),
    )
    idea = await service.create(
        user_id=owner.id,
        telegram_user_id=500,
        chat_id=500,
        source="text",
        raw_text="идея",
        parsed=ParsedThought(kind="idea", title="Идея"),
    )
    assert (await service.confirm(task.id, task.version, 500, 500)).ok
    assert (await service.confirm(idea.id, idea.version, 500, 500)).ok
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskState.id))) == 1
        assert await session.scalar(select(func.count(InboxItem.id))) == 2
        assert await session.scalar(select(func.count(TaskActionToken.token))) == 0
