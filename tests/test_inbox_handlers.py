from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from autotester.fakes import FakeCallbackQuery, FakeMessage, ScriptedTranscription
from sqlalchemy import select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import InboxItem, TaskReminder, TaskState


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        database_url="sqlite+aiosqlite:///:memory:",
    )


def update_for(message, *, user_id=83001, chat_id=83001, query=None):
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


async def create_saved_task(bot: FutureSelfBot, telegram_id: int, title: str) -> int:
    owner = await bot._user(telegram_id)
    now = datetime.now(UTC)
    async with bot.db.session() as session:
        item = InboxItem(
            user_id=owner.id,
            kind="task",
            title=title,
            description="Проверить безопасное удаление",
            raw_text=title,
            next_step=None,
            resolved_date=None,
            temporal_resolution=None,
            source="text",
            status="confirmed",
        )
        session.add(item)
        await session.flush()
        session.add(
            TaskState(
                owner_id=owner.id,
                inbox_item_id=item.id,
                status="active",
                event_at=now + timedelta(hours=2),
                timezone="Europe/Moscow",
                version=1,
            )
        )
        session.add(
            TaskReminder(
                inbox_item_id=item.id,
                telegram_user_id=telegram_id,
                chat_id=telegram_id,
                event_at=now + timedelta(hours=2),
                remind_at=now + timedelta(hours=1),
                timezone="Europe/Moscow",
                delivery_key=f"inbox-handler:{item.id}:v1",
                task_version=1,
                status="pending",
            )
        )
        await session.flush()
        return item.id


async def test_inbox_card_trash_confirmation_and_restore_are_safe(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    item_id = await create_saved_task(bot, 83001, "Позвонить Варваре")
    message = FakeMessage("/inbox")

    await bot.inbox(update_for(message), context())

    assert "Позвонить Варваре" in message.replies[-1]["text"]
    assert callback_by_label(message, "🧹 Убрать ошибочные команды") == ("ibox:cleanup:commands")
    assert callback_by_label(message, "В корзину 1") == f"ibox:trash:{item_id}"
    assert callback_by_label(message, "🗑 В корзину эту страницу").startswith("ibox:trashpage:0.")
    open_callback = callback_by_label(message, "Открыть 1")
    assert len(open_callback.encode()) <= 64
    await bot.saved_inbox_action(
        update_for(
            message,
            query=FakeCallbackQuery(open_callback, message),
        ),
        context(),
    )
    trash_callback = callback_by_label(message, "🗑 В корзину")
    await bot.saved_inbox_action(
        update_for(
            message,
            query=FakeCallbackQuery(trash_callback, message),
        ),
        context(),
    )
    assert "Переместить в корзину записей: 1?" in message.replies[-1]["text"]

    confirm_callback = callback_by_label(message, "Да, в корзину (1)")
    await bot.system_draft_action(
        update_for(
            message,
            query=FakeCallbackQuery(confirm_callback, message),
        ),
        context(),
    )
    async with db.sessions() as session:
        item = await session.get(InboxItem, item_id)
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item_id))
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == item_id)
        )
    assert (item.status, item.pre_trash_status, item.version) == ("trashed", "confirmed", 2)
    assert (state.status, state.version) == ("active", 2)
    assert reminder.status == "cancelled"

    trash_list = FakeCallbackQuery("ibox:trashlist:0", message)
    await bot.saved_inbox_action(update_for(message, query=trash_list), context())
    trash_open = callback_by_label(message, "Открыть 1")
    await bot.saved_inbox_action(
        update_for(message, query=FakeCallbackQuery(trash_open, message)),
        context(),
    )
    restore = callback_by_label(message, "↩️ Восстановить")
    await bot.saved_inbox_action(
        update_for(message, query=FakeCallbackQuery(restore, message)),
        context(),
    )
    async with db.sessions() as session:
        item = await session.get(InboxItem, item_id)
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item_id))
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == item_id)
        )
    assert (item.status, item.pre_trash_status, item.version) == ("confirmed", None, 3)
    assert (state.status, state.version) == ("active", 3)
    assert reminder.status == "cancelled"
    assert "прежнее напоминание осталось выключенным" in message.replies[-1]["text"].casefold()


async def test_inbox_cleanup_commands_is_previewed_and_owner_scoped(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    first = await create_saved_task(bot, 83002, "Удалить все просроченные")
    second = await create_saved_task(bot, 83002, "Удалить все неактуальные черновики")
    keep = await create_saved_task(bot, 83002, "Купить витамины")
    foreign = await create_saved_task(bot, 83999, "Удалить все просроченные")
    message = FakeMessage("/inbox")
    await bot.inbox(update_for(message, user_id=83002, chat_id=83002), context())

    cleanup = callback_by_label(message, "🧹 Убрать ошибочные команды")
    await bot.saved_inbox_action(
        update_for(
            message,
            user_id=83002,
            chat_id=83002,
            query=FakeCallbackQuery(cleanup, message),
        ),
        context(),
    )
    assert "Переместить в корзину записей: 2?" in message.replies[-1]["text"]
    confirm = callback_by_label(message, "Да, в корзину (2)")
    await bot.system_draft_action(
        update_for(
            message,
            user_id=83002,
            chat_id=83002,
            query=FakeCallbackQuery(confirm, message),
        ),
        context(),
    )
    async with db.sessions() as session:
        statuses = {
            item.id: item.status
            for item in (
                await session.scalars(
                    select(InboxItem).where(InboxItem.id.in_({first, second, keep, foreign}))
                )
            ).all()
        }
    assert statuses == {
        first: "trashed",
        second: "trashed",
        keep: "confirmed",
        foreign: "confirmed",
    }
    assert fake_ai.route_calls == []


async def test_inbox_page_bulk_trash_requires_exact_confirmation(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    first = await create_saved_task(bot, 83005, "Первая запись")
    second = await create_saved_task(bot, 83005, "Вторая запись")
    message = FakeMessage("/inbox")
    await bot.inbox(update_for(message, user_id=83005, chat_id=83005), context())

    callback = callback_by_label(message, "🗑 В корзину эту страницу")
    await bot.saved_inbox_action(
        update_for(
            message,
            user_id=83005,
            chat_id=83005,
            query=FakeCallbackQuery(callback, message),
        ),
        context(),
    )
    assert "Переместить в корзину записей: 2?" in message.replies[-1]["text"]
    confirm = callback_by_label(message, "Да, в корзину (2)")
    await bot.system_draft_action(
        update_for(
            message,
            user_id=83005,
            chat_id=83005,
            query=FakeCallbackQuery(confirm, message),
        ),
        context(),
    )

    async with db.sessions() as session:
        statuses = list(
            await session.scalars(
                select(InboxItem.status)
                .where(InboxItem.id.in_({first, second}))
                .order_by(InboxItem.id)
            )
        )
    assert statuses == ["trashed", "trashed"]


async def test_inbox_page_bulk_rejects_shifted_rendered_page(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    first = await create_saved_task(bot, 83007, "Первая запись")
    message = FakeMessage("/inbox")
    await bot.inbox(update_for(message, user_id=83007, chat_id=83007), context())
    stale_callback = callback_by_label(message, "🗑 В корзину эту страницу")
    second = await create_saved_task(bot, 83007, "Новая запись")
    query = FakeCallbackQuery(stale_callback, message)

    await bot.saved_inbox_action(
        update_for(
            message,
            user_id=83007,
            chat_id=83007,
            query=query,
        ),
        context(),
    )

    assert query.answers[-1][1] is True
    assert "страница inbox изменилась" in (query.answers[-1][0] or "").casefold()
    async with db.sessions() as session:
        statuses = list(
            await session.scalars(
                select(InboxItem.status)
                .where(InboxItem.id.in_({first, second}))
                .order_by(InboxItem.id)
            )
        )
    assert statuses == ["confirmed", "confirmed"]


async def test_command_cleanup_invalidates_when_matching_set_changes(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    first = await create_saved_task(bot, 83006, "Удалить все просроченные")
    message = FakeMessage("/inbox")
    await bot.inbox(update_for(message, user_id=83006, chat_id=83006), context())
    cleanup = callback_by_label(message, "🧹 Убрать ошибочные команды")
    await bot.saved_inbox_action(
        update_for(
            message,
            user_id=83006,
            chat_id=83006,
            query=FakeCallbackQuery(cleanup, message),
        ),
        context(),
    )
    confirm = callback_by_label(message, "Да, в корзину (1)")
    second = await create_saved_task(bot, 83006, "Сохраним инбокс")

    await bot.system_draft_action(
        update_for(
            message,
            user_id=83006,
            chat_id=83006,
            query=FakeCallbackQuery(confirm, message),
        ),
        context(),
    )

    assert "ошибочно сохранённых команд изменился" in message.replies[-1]["text"].casefold()
    async with db.sessions() as session:
        statuses = list(
            await session.scalars(
                select(InboxItem.status)
                .where(InboxItem.id.in_({first, second}))
                .order_by(InboxItem.id)
            )
        )
    assert statuses == ["confirmed", "confirmed"]


async def test_forged_foreign_inbox_callback_fails_closed(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    foreign_id = await create_saved_task(bot, 83004, "Чужая запись")
    await bot._user(83003)
    message = FakeMessage()
    query = FakeCallbackQuery(f"ibox:trash:{foreign_id}", message)

    await bot.saved_inbox_action(
        update_for(message, user_id=83003, chat_id=83003, query=query),
        context(),
    )

    assert query.answers[-1] == ("Запись уже изменилась", True)
    async with db.sessions() as session:
        item = await session.get(InboxItem, foreign_id)
    assert item.status == "confirmed"
