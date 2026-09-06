from datetime import UTC, datetime, time, timedelta
from types import SimpleNamespace

from autotester.fakes import FakeCallbackQuery, FakeMessage, ScriptedTranscription
from sqlalchemy import select

from future_self.access import SUBSCRIBER
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.drafts import DraftInboxService
from future_self.models import (
    InboxItem,
    RecurringTaskReminderSchedule,
    TaskActionToken,
    TaskReminder,
    TaskState,
    User,
)
from future_self.recurring_reminders import RecurringTaskReminderService
from future_self.schemas import ParsedThought, TemporalResolution


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        database_url="sqlite+aiosqlite:///:memory:",
    )


def update_for(message, *, user_id=701, chat_id=701, query=None):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
    )


def context():
    return SimpleNamespace(user_data={}, args=[], bot=None)


def callback_by_label(message: FakeMessage, label: str) -> str:
    for reply in reversed(message.replies):
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        for row in markup.inline_keyboard:
            for button in row:
                if button.text == label:
                    return button.callback_data
    raise AssertionError(f"Missing button {label}")


async def create_confirmed_task(
    bot,
    *,
    user_id=701,
    source="text",
    description="Описание",
    same_local_day=False,
):
    owner = await bot._user(user_id)
    now = datetime.now(UTC)
    zone = __import__("zoneinfo").ZoneInfo(owner.timezone)
    event_at = now + timedelta(hours=2)
    if same_local_day:
        event_at = (
            now.astimezone(zone).replace(hour=12, minute=0, second=0, microsecond=0).astimezone(UTC)
        )
    temporal = TemporalResolution(
        resolved_at=event_at,
        remind_at=event_at - timedelta(hours=1),
        timezone=owner.timezone,
        resolved_local_date=event_at.astimezone(zone).date(),
        resolved_local_time=event_at.astimezone(zone).time().replace(tzinfo=None),
        precision="datetime",
        original_expression="через два часа",
    )
    service = DraftInboxService(bot.db, 60)
    draft = await service.create(
        user_id=owner.id,
        telegram_user_id=user_id,
        chat_id=user_id,
        source=source,
        raw_text="Проверить задачу",
        parsed=ParsedThought(
            kind="task",
            title="Проверить задачу",
            description=description,
            temporal_resolution=temporal,
            resolved_date=temporal.resolved_local_date,
        ),
    )
    return await service.confirm(draft.id, draft.version, user_id, user_id)


async def create_daily_task(bot, db, *, user_id=701, local_time=time(20, 30)):
    owner = await bot._user(user_id)
    async with db.session() as session:
        stored_owner = await session.get(User, owner.id)
        stored_owner.access_tier = SUBSCRIBER
        stored_owner.access_version = 3
    owner.access_tier = SUBSCRIBER
    owner.access_version = 3
    draft_service = DraftInboxService(bot.db, 60)
    draft = await draft_service.create(
        user_id=owner.id,
        telegram_user_id=user_id,
        chat_id=user_id,
        source="text",
        raw_text="Заполнить дневник",
        parsed=ParsedThought(
            kind="task",
            title="Заполнить дневник",
            description=None,
        ),
    )
    confirmed = await draft_service.confirm(draft.id, draft.version, user_id, user_id)
    recurring = await RecurringTaskReminderService(db).create_daily(
        owner.id,
        confirmed.inbox_item.id,
        local_time,
    )
    assert recurring.schedule is not None
    return confirmed, recurring.schedule


async def test_tasks_menu_has_required_buttons_and_creation_uses_existing_preview_guide(
    db, fake_ai
):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage("/tasks")
    await bot.tasks_command(update_for(message), context())
    labels = [
        button.text for row in message.replies[-1]["reply_markup"].inline_keyboard for button in row
    ]
    assert labels == [
        "Сегодня",
        "Предстоящие",
        "Просроченные",
        "Без срока",
        "Выполненные",
        "🧹 Очистить просроченные",
        "Создать задачу",
        "🔁 Ежедневные напоминания",
        "← Назад",
        "🏠 Главное меню",
    ]
    create_message = FakeMessage()
    await bot.task_create(update_for(create_message), context())
    assert "preview" in create_message.replies[-1]["text"]
    assert "Завтра в 18:00" in create_message.replies[-1]["text"]
    assert fake_ai.route_calls == []


async def test_task_hub_cleanup_button_starts_existing_overdue_preview(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_confirmed_task(bot)
    async with db.session() as session:
        state = await session.scalar(
            select(TaskState).where(TaskState.inbox_item_id == created.inbox_item.id)
        )
        state.event_at = datetime.now(UTC) - timedelta(days=2)

    message = FakeMessage("/tasks")
    await bot.tasks_command(update_for(message), context())
    callback = callback_by_label(message, "🧹 Очистить просроченные")
    assert callback == "task:cleanup:overdue"
    assert len(callback.encode()) <= 64

    query = FakeCallbackQuery(callback, message)
    await bot.task_callback(update_for(message, query=query), context())

    assert query.answers
    assert message.replies[-1]["text"].startswith("Нашёл просроченных задач: 1.")
    async with db.sessions() as session:
        item = await session.get(InboxItem, created.inbox_item.id)
        state = await session.scalar(
            select(TaskState).where(TaskState.inbox_item_id == created.inbox_item.id)
        )
    assert (item.status, state.status) == ("confirmed", "active")
    assert fake_ai.route_calls == []


async def test_card_complete_replay_reopen_and_delete_navigation_are_deterministic(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_confirmed_task(bot, same_local_day=True)
    listing = FakeMessage()
    await bot.task_today(update_for(listing), context())
    open_callback = callback_by_label(listing, "Открыть 1")
    assert len(open_callback.encode()) <= 64
    assert open_callback.count(":") == 1

    query = FakeCallbackQuery(open_callback, listing)
    await bot.task_callback(update_for(listing, query=query), context())
    assert "Проверить задачу" in listing.replies[-1]["text"]
    complete_callback = callback_by_label(listing, "Выполнено")

    complete_query = FakeCallbackQuery(complete_callback, listing)
    await bot.task_callback(update_for(listing, query=complete_query), context())
    assert "задача выполнена" in listing.replies[-1]["text"]
    assert "Вернуть в активные" == next(
        button.text
        for row in listing.replies[-1]["reply_markup"].inline_keyboard
        for button in row
        if button.text == "Вернуть в активные"
    )

    replay = FakeCallbackQuery(complete_callback, listing)
    await bot.task_callback(update_for(listing, query=replay), context())
    assert "уже выполнена" in listing.replies[-1]["text"]

    reopen_callback = callback_by_label(listing, "Вернуть в активные")
    reopened = FakeCallbackQuery(reopen_callback, listing)
    await bot.task_callback(update_for(listing, query=reopened), context())
    assert "Старое напоминание не включено" in listing.replies[-1]["text"]

    delete_callback = callback_by_label(listing, "Удалить")
    await bot.task_callback(
        update_for(listing, query=FakeCallbackQuery(delete_callback, listing)), context()
    )
    confirm_callback = callback_by_label(listing, "Да, в корзину")
    await bot.task_callback(
        update_for(listing, query=FakeCallbackQuery(confirm_callback, listing)), context()
    )
    assert "Задача перенесена в корзину" in listing.replies[-1]["text"]
    assert "восстановить задачу можно через /inbox" in listing.replies[-1]["text"]
    assert "История" in listing.replies[-1]["text"]
    async with db.sessions() as session:
        item = await session.get(InboxItem, created.inbox_item.id)
        state = await session.scalar(
            select(TaskState).where(TaskState.inbox_item_id == created.inbox_item.id)
        )
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == created.inbox_item.id)
        )
    assert (item.status, state.status, reminder.status) == (
        "trashed",
        "active",
        "cancelled",
    )
    assert item.pre_trash_status == "confirmed"
    assert item.trashed_at is not None
    assert fake_ai.route_calls == []


async def test_doctor_task_card_hides_description_and_custom_input_avoids_llm(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    result = await create_confirmed_task(
        bot,
        source="doctor_prepare",
        description="Симптомы и приватная причина обращения",
    )
    record = await bot.task_service.record(result.inbox_item.user_id, result.inbox_item.id)
    message = FakeMessage()
    await bot._send_record(
        message,
        result.inbox_item.user_id,
        701,
        record,
        "upcoming",
        0,
    )
    card = message.replies[-1]["text"]
    assert "Раздел «Врач»" in card
    assert "Симптомы" not in card
    assert "причина" not in card

    reminder_callback = callback_by_label(message, "Изменить напоминание")
    query = FakeCallbackQuery(reminder_callback, message)
    await bot.task_callback(update_for(message, query=query), context())
    input_message = FakeMessage("через 1 час")
    assert await bot.task_pending_text(update_for(input_message))
    assert "Напоминание обновлено" in input_message.replies[-1]["text"]
    assert fake_ai.route_calls == []


async def test_stale_persistent_input_is_consumed_instead_of_capturing_future_text(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    result = await create_confirmed_task(bot)
    owner_id = result.inbox_item.user_id
    item_id = result.inbox_item.id
    actions = await bot.task_service.issue_actions(
        owner_id,
        701,
        item_id,
        1,
        ("reminder_edit", "complete"),
    )
    assert (
        await bot.task_service.start_reminder_input(actions["reminder_edit"], owner_id, 701)
    ).status == "await_reminder"
    assert (
        await bot.task_service.complete(actions["complete"], owner_id, 701)
    ).status == "completed"
    message = FakeMessage("завтра в 18:00")
    assert await bot.task_pending_text(update_for(message))
    assert "задача уже изменилась" in message.replies[-1]["text"]
    assert await bot.task_service.pending_input(owner_id, 701) is None
    assert fake_ai.route_calls == []


async def test_daily_schedule_card_mutations_use_opaque_version_fenced_actions(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created, schedule = await create_daily_task(bot, db)
    owner_id = created.inbox_item.user_id

    message = FakeMessage("/tasks")
    await bot.tasks_command(update_for(message), context())
    recurring_callback = callback_by_label(message, "🔁 Ежедневные напоминания")
    assert recurring_callback == "task:recurring:0"

    recurring_query = FakeCallbackQuery(recurring_callback, message)
    await bot.task_callback(update_for(message, query=recurring_query), context())
    assert len(recurring_query.answers) == 1
    assert "Заполнить дневник — 20:30 · включено" in message.replies[-1]["text"]

    open_callback = callback_by_label(message, "Открыть 1")
    assert open_callback.count(":") == 1
    assert len(open_callback.encode()) <= 64
    open_query = FakeCallbackQuery(open_callback, message)
    await bot.task_callback(update_for(message, query=open_query), context())
    assert len(open_query.answers) == 1
    card = message.replies[-1]["text"]
    assert "Повтор: каждый день в 20:30 (Europe/Moscow)" in card
    assert "Статус повтора: включено" in card
    assert "Следующее срабатывание:" in card
    assert callback_by_label(message, "← Назад к списку") == "task:recurring:0"

    edit_callback = callback_by_label(message, "🕒 Изменить время")
    disable_callback = callback_by_label(message, "⏸ Отключить")
    assert edit_callback.count(":") == disable_callback.count(":") == 1
    async with db.sessions() as session:
        disable_token = await session.get(TaskActionToken, disable_callback.removeprefix("task:"))
    assert disable_token.action == "recurring_disable"
    assert disable_token.payload == {
        "schedule_version": schedule.version,
        "access_version": 3,
    }

    disable_query = FakeCallbackQuery(disable_callback, message)
    await bot.task_callback(update_for(message, query=disable_query), context())
    assert len(disable_query.answers) == 1
    assert "Ежедневное напоминание отключено" in message.replies[-1]["text"]
    assert "Статус повтора: отключено" in message.replies[-1]["text"]
    assert "Следующее срабатывание: —" in message.replies[-1]["text"]
    reenable_callback = callback_by_label(message, "▶️ Включить снова")

    stale_edit_query = FakeCallbackQuery(edit_callback, message)
    await bot.task_callback(update_for(message, query=stale_edit_query), context())
    assert len(stale_edit_query.answers) == 1
    assert await bot.task_service.pending_input(owner_id, 701) is None
    async with db.sessions() as session:
        unchanged = await session.get(RecurringTaskReminderSchedule, schedule.id)
        assert (unchanged.status, unchanged.version, unchanged.local_time) == (
            "disabled",
            schedule.version + 1,
            time(20, 30),
        )

    reenable_query = FakeCallbackQuery(reenable_callback, message)
    await bot.task_callback(update_for(message, query=reenable_query), context())
    assert len(reenable_query.answers) == 1
    assert "снова включено" in message.replies[-1]["text"]
    assert "Статус повтора: включено" in message.replies[-1]["text"]

    time_callback = callback_by_label(message, "🕒 Изменить время")
    time_query = FakeCallbackQuery(time_callback, message)
    await bot.task_callback(update_for(message, query=time_query), context())
    assert len(time_query.answers) == 1
    assert "Пришли новое время, например: 19:30" in message.replies[-1]["text"]

    input_message = FakeMessage("19.30")
    assert await bot.task_pending_text(update_for(input_message))
    assert "Время ежедневного напоминания обновлено" in input_message.replies[-1]["text"]
    assert "Повтор: каждый день в 19:30" in input_message.replies[-1]["text"]
    async with db.sessions() as session:
        updated = await session.get(RecurringTaskReminderSchedule, schedule.id)
        assert (updated.status, updated.version, updated.local_time) == (
            "active",
            schedule.version + 3,
            time(19, 30),
        )
        assert (
            await session.scalar(
                select(TaskReminder).where(TaskReminder.inbox_item_id == created.inbox_item.id)
            )
            is None
        )
    assert fake_ai.route_calls == []


async def test_recurring_mutation_rejects_access_version_bounce_after_token_claim(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created, schedule = await create_daily_task(bot, db, user_id=702)
    owner_id = created.inbox_item.user_id
    async with db.sessions() as session:
        state = await session.scalar(
            select(TaskState).where(TaskState.inbox_item_id == created.inbox_item.id)
        )
    tokens = await bot.task_service.issue_actions(
        owner_id,
        702,
        created.inbox_item.id,
        state.version,
        ("recurring_disable",),
        payload={"schedule_version": schedule.version},
    )
    original_disable = bot.task_service.recurring_reminders.disable

    async def bounce_then_disable(*args, **kwargs):
        async with db.session() as session:
            owner = await session.get(User, owner_id)
            owner.access_version += 2
        return await original_disable(*args, **kwargs)

    bot.task_service.recurring_reminders.disable = bounce_then_disable
    result = await bot.task_service.disable_recurring(tokens["recurring_disable"], owner_id, 702)

    assert result.status == "stale"
    async with db.sessions() as session:
        unchanged = await session.get(RecurringTaskReminderSchedule, schedule.id)
        assert (unchanged.status, unchanged.version) == ("active", schedule.version)
