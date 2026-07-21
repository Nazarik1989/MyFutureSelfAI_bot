from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from autotester.fakes import FakeCallbackQuery, FakeMessage, ScriptedTranscription

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.drafts import DraftInboxService
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


async def create_confirmed_task(bot, *, user_id=701, source="text", description="Описание"):
    owner = await bot._user(user_id)
    now = datetime.now(UTC)
    temporal = TemporalResolution(
        resolved_at=now + timedelta(hours=2),
        remind_at=now + timedelta(hours=1),
        timezone=owner.timezone,
        resolved_local_date=(now + timedelta(hours=2))
        .astimezone(__import__("zoneinfo").ZoneInfo(owner.timezone))
        .date(),
        resolved_local_time=(now + timedelta(hours=2))
        .astimezone(__import__("zoneinfo").ZoneInfo(owner.timezone))
        .time()
        .replace(tzinfo=None),
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
        "Создать задачу",
        "Как работают напоминания",
        "← Назад",
        "🏠 Главное меню",
    ]
    create_message = FakeMessage()
    await bot.task_create(update_for(create_message), context())
    assert "preview" in create_message.replies[-1]["text"]
    assert "Завтра в 18:00" in create_message.replies[-1]["text"]
    assert fake_ai.route_calls == []


async def test_card_complete_replay_reopen_and_delete_navigation_are_deterministic(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await create_confirmed_task(bot)
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
