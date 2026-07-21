from types import SimpleNamespace

from autotester.fakes import FakeCallbackQuery, FakeMessage, ScriptedTranscription
from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import InboxItem, LifeCollection, LifeCollectionLink, TaskState


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        database_url="sqlite+aiosqlite:///:memory:",
    )


def update_for(message, *, user_id=720001, chat_id=720001, query=None):
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


async def click(bot, message, label, *, user_id=720001, chat_id=720001):
    data = callback_by_label(message, label)
    query = FakeCallbackQuery(data, message)
    await bot.collection_callback(
        update_for(message, user_id=user_id, chat_id=chat_id, query=query), context()
    )
    return data, query


async def onboard_all(bot, *, user_id=720001, chat_id=720001):
    message = FakeMessage("/collections")
    await bot.collections_command(update_for(message, user_id=user_id, chat_id=chat_id), context())
    await click(bot, message, "Создать все", user_id=user_id, chat_id=chat_id)
    return message


async def test_onboarding_all_natural_list_and_continuation_avoid_llm(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = await onboard_all(bot)
    assert "Стартовые разделы созданы" in message.replies[-1]["text"]

    add = FakeMessage("Добавь в покупки чай, сахар, бетономешалку и остров в Индийском океане")
    await bot._route_message(update_for(add), context(), add.text, "text")
    assert "Добавлено в «Покупки»: 4" in add.replies[-1]["text"]

    more = FakeMessage("Ещё добавь цемент")
    await bot._route_message(update_for(more), context(), more.text, "text")
    assert "Добавлено в «Покупки»: 1" in more.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 5
        assert await session.scalar(select(func.count(TaskState.id))) == 5
    assert fake_ai.route_calls == []


async def test_natural_create_confirmation_survives_restart_and_rejects_replay(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage("Создай проект Наз и Войд")
    await bot._route_message(update_for(message), context(), message.text, "text")
    token = callback_by_label(message, "Создать")
    assert len(token.encode()) <= 64
    assert await bot.collection_service.is_onboarded((await bot._user(720001)).id) is False

    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    forged_query = FakeCallbackQuery(token, message)
    await restarted.collection_callback(
        update_for(message, chat_id=720002, query=forged_query), context()
    )
    assert forged_query.answers[-1][1] is True

    query = FakeCallbackQuery(token, message)
    await restarted.collection_callback(update_for(message, query=query), context())
    assert "создан" in message.replies[-1]["text"]
    replay = FakeCallbackQuery(token, message)
    await restarted.collection_callback(update_for(message, query=replay), context())
    assert replay.answers[-1][1] is True
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LifeCollection.id))) == 1
    assert fake_ai.route_calls == []


async def test_low_confidence_requires_choice_and_ambiguous_split_requires_preview(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await onboard_all(bot)
    message = FakeMessage('Добавь в покупки "чай, сахар"')
    await bot._route_message(update_for(message), context(), message.text, "text")
    assert "Проверь разбиение" in message.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    await click(bot, message, "Сохранить")
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
    assert fake_ai.route_calls == []


async def test_collection_cards_have_navigation_task_hub_and_safe_unlink(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await onboard_all(bot)
    add = FakeMessage("Добавь в покупки чай")
    await bot._route_message(update_for(add), context(), add.text, "text")

    show = FakeMessage("Что находится в покупках?")
    await bot._route_message(update_for(show), context(), show.text, "text")
    labels = [
        button.text for row in show.replies[-1]["reply_markup"].inline_keyboard for button in row
    ]
    assert {"Открыть 1", "Добавить запись", "← Назад", "🏠 В меню", "❓ Помощь"} <= set(labels)
    await click(bot, show, "Открыть 1")
    labels = [
        button.text for row in show.replies[-1]["reply_markup"].inline_keyboard for button in row
    ]
    assert {
        "Открыть в Task Hub",
        "Переместить",
        "Связать ещё",
        "Убрать из раздела",
    } <= set(labels)
    await click(bot, show, "Убрать из раздела")
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LifeCollectionLink.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskState.id))) == 1
    assert fake_ai.route_calls == []


async def test_nonempty_delete_only_links_preserves_inbox_and_empty_delete_is_safe(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await onboard_all(bot)
    add = FakeMessage("Добавь в покупки чай")
    await bot._route_message(update_for(add), context(), add.text, "text")
    hub = FakeMessage("/collections")
    await bot.collections_command(update_for(hub), context())
    await click(bot, hub, "Списки")
    await click(bot, hub, "Покупки · 1")
    await click(bot, hub, "Удалить")
    assert "исходные записи останутся" in hub.replies[-1]["text"]
    await click(bot, hub, "Удалить только связи")
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert (
            await session.scalar(
                select(func.count(LifeCollection.id)).where(
                    LifeCollection.normalized_name == "покупки"
                )
            )
            == 0
        )
    assert fake_ai.route_calls == []


async def test_cancel_clears_persistent_input_and_active_context(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    hub = await onboard_all(bot)
    await click(bot, hub, "+ Проект")
    cancelled = FakeMessage("/cancel")
    await bot.cancel_draft_edit(update_for(cancelled), context())
    assert "Операция с разделом отменена" in cancelled.replies[-1]["text"]
    assert (
        await bot.collection_pending_text(
            update_for(FakeMessage("Не должно стать названием")),
            "Не должно стать названием",
            "text",
        )
        is False
    )
    assert fake_ai.route_calls == []


async def test_unknown_named_collection_is_created_only_after_confirmation(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage("Добавь в покупки чай, сахар")
    await bot._route_message(update_for(message), context(), message.text, "text")
    assert "Раздел «покупки» не найден" in message.replies[-1]["text"]
    assert "Создать список" in message.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LifeCollection.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    await click(bot, message, "Создать")
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LifeCollection.id))) == 1
        assert await session.scalar(select(func.count(InboxItem.id))) == 2
    assert fake_ai.route_calls == []
