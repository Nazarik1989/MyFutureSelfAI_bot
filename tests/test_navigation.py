from types import SimpleNamespace

import pytest
from autotester.fakes import FakeBot, FakeCallbackQuery, FakeMessage, ScriptedTranscription
from telegram.ext import CallbackQueryHandler, CommandHandler, ConversationHandler

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.navigation import (
    ACTIONS,
    ADVANCED_COMMANDS,
    HELP_TOPICS,
    PUBLIC_COMMANDS,
    SECTIONS,
    NavigationFlowStore,
    validate_catalog,
)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        database_url="sqlite+aiosqlite:///:memory:",
    )


def update_for(message, *, user_id=101, chat_id=201, query=None, chat_type="private"):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
    )


def context() -> SimpleNamespace:
    return SimpleNamespace(user_data={}, bot=FakeBot(), args=[])


def callback_from(message: FakeMessage, prefix: str) -> str:
    for reply in reversed(message.replies):
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        for row in markup.inline_keyboard:
            for button in row:
                if button.callback_data and button.callback_data.startswith(prefix):
                    return button.callback_data
    raise AssertionError(f"Missing callback {prefix}")


async def test_menu_help_sections_and_catalog_are_complete_without_llm(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage("/menu")
    await bot.menu_command(update_for(message), context())
    markup = message.replies[-1]["reply_markup"]
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert callbacks == [*(f"nav:section:{key}" for key in SECTIONS), "nav:help"]

    for section_key, section in SECTIONS.items():
        section_message = FakeMessage()
        await bot._send_navigation_section(section_message, section_key)
        section_markup = section_message.replies[-1]["reply_markup"]
        section_callbacks = [
            button.callback_data for row in section_markup.inline_keyboard for button in row
        ]
        assert [f"nav:action:{key}" for key in section.actions] == section_callbacks[
            : len(section.actions)
        ]
        assert "nav:root" in section_callbacks
        assert "nav:help" in section_callbacks

    help_message = FakeMessage("/help")
    await bot.help_command(update_for(help_message), context())
    help_callbacks = [
        button.callback_data
        for row in help_message.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert [f"nav:help:{key}" for key in HELP_TOPICS] == help_callbacks[:-1]
    assert fake_ai.route_calls == []


def test_catalog_has_no_dead_buttons_duplicates_or_sensitive_callback_data(fake_ai):
    validate_catalog()
    names = [item.command for item in PUBLIC_COMMANDS]
    assert names == [
        "menu",
        "inbox",
        "tasks",
        "vision",
        "health",
        "checkin",
        "doctor",
        "labs",
        "location",
        "help",
    ]
    assert len(names) == len(set(names))
    assert len(ACTIONS) == len(set(ACTIONS))
    assert {action for section in SECTIONS.values() for action in section.actions} == set(ACTIONS)

    from future_self.db import Database

    bot = FutureSelfBot(
        settings(), Database(settings().database_url), fake_ai, ScriptedTranscription()
    )
    for action in ACTIONS.values():
        if action.handler:
            assert callable(getattr(bot, action.handler, None))
    callbacks = [
        *(f"nav:section:{key}" for key in SECTIONS),
        *(f"nav:action:{key}" for key in ACTIONS),
        *(f"nav:help:{key}" for key in HELP_TOPICS),
        "nav:root",
        "nav:help",
    ]
    assert all(len(value.encode()) <= 64 for value in callbacks)
    assert all(not any(char.isdigit() for char in value) for value in callbacks)


def test_every_registered_command_is_catalogued_or_explicitly_advanced(db, fake_ai):
    application = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription()).build()

    def command_handlers(handler):
        if isinstance(handler, CommandHandler):
            yield from handler.commands
        if isinstance(handler, ConversationHandler):
            for child in handler.entry_points:
                yield from command_handlers(child)
            for children in handler.states.values():
                for child in children:
                    yield from command_handlers(child)
            for child in handler.fallbacks:
                yield from command_handlers(child)

    registered = {
        command
        for handlers in application.handlers.values()
        for handler in handlers
        for command in command_handlers(handler)
    }
    public = {item.command for item in PUBLIC_COMMANDS}
    assert public <= registered
    assert registered <= public | ADVANCED_COMMANDS
    assert any(isinstance(handler, CallbackQueryHandler) for handler in application.handlers[0])


@pytest.mark.parametrize(
    ("phrase", "action"),
    [
        ("Меню", "menu"),
        ("ГЛАВНОЕ МЕНЮ!!!", "menu"),
        ("Помощь", "help"),
        ("Что ты умеешь?", "help"),
        ("Как пользоваться ботом?", "help"),
        ("Какие есть команды?", "help"),
    ],
)
def test_natural_navigation_is_exact_deterministic_and_punctuation_safe(
    db, fake_ai, phrase, action
):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    assert bot.natural_command_router.route(phrase).action == action
    assert bot.natural_command_router.route("Добавь меню ужина в заметки") is None
    assert bot.natural_command_router.route("Мне нужна помощь с покупкой билетов") is None


async def test_health_flow_continue_exit_owner_binding_repeat_and_state_isolation(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    ctx = context()
    ctx.user_data.update({"health_checkin": {"energy": 7}, "unrelated": "keep"})
    message = FakeMessage("/menu")
    owner_update = update_for(message)
    await bot.menu_command(owner_update, ctx)
    token_data = callback_from(message, "nav:flow:exit:")

    forged_query = FakeCallbackQuery(token_data, message)
    forged = update_for(message, user_id=999, chat_id=999, query=forged_query)
    assert await bot.navigation_action(forged, context()) is None
    assert any(show_alert for _text, show_alert in forged_query.answers)
    assert "health_checkin" in ctx.user_data

    query = FakeCallbackQuery(token_data, message)
    result = await bot.navigation_action(update_for(message, query=query), ctx)
    assert result == ConversationHandler.END
    assert "health_checkin" not in ctx.user_data
    assert ctx.user_data["unrelated"] == "keep"

    repeat = FakeCallbackQuery(token_data, message)
    assert await bot.navigation_action(update_for(message, query=repeat), ctx) is None
    assert any(show_alert for _text, show_alert in repeat.answers)


async def test_continue_keeps_flow_and_old_message_edit_falls_back_safely(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    ctx = context()
    ctx.user_data["doctor_prepare"] = {"reason": "private"}
    message = FakeMessage("/menu")
    update = update_for(message)
    await bot.menu_command(update, ctx)
    data = callback_from(message, "nav:flow:continue:")
    query = FakeCallbackQuery(data, message)
    before = len(message.replies)
    assert await bot.navigation_action(update_for(message, query=query), ctx) is None
    assert ctx.user_data["doctor_prepare"] == {"reason": "private"}
    assert len(message.replies) == before + 1
    assert "private" not in str(message.replies[before])


async def test_old_cross_section_entry_cannot_start_second_flow(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    ctx = context()
    ctx.user_data["health_checkin"] = {"energy": 4}
    message = FakeMessage()
    query = FakeCallbackQuery("nav:action:doctor_prepare", message)
    result = await bot.navigation_doctor_entry(
        update_for(message, query=query),
        ctx,
    )
    assert result is None
    assert ctx.user_data["health_checkin"] == {"energy": 4}
    assert "doctor_prepare" not in ctx.user_data
    assert callback_from(message, "nav:flow:continue:").startswith("nav:flow:continue:")


async def test_navigation_flow_store_is_single_use_owner_chat_bound_and_expires():
    store = NavigationFlowStore(ttl_seconds=60)
    token = await store.issue(1, 10, "health")
    assert await store.claim(token, 2, 10) is None
    assert await store.claim(token, 1, 11) is None
    assert (await store.claim(token, 1, 10)).flow == "health"
    assert await store.claim(token, 1, 10) is None
    expired = NavigationFlowStore(ttl_seconds=-1)
    stale = await expired.issue(1, 10, "vision")
    assert await expired.claim(stale, 1, 10) is None


async def test_native_private_commands_are_registered_once_during_startup(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())

    class TelegramBot:
        def __init__(self):
            self.command_calls = []
            self.menu_calls = []

        async def set_my_commands(self, commands, **kwargs):
            self.command_calls.append((commands, kwargs))

        async def set_chat_menu_button(self, **kwargs):
            self.menu_calls.append(kwargs)

    telegram = TelegramBot()
    app = SimpleNamespace(bot=telegram, job_queue=None)
    await bot._post_init(app)
    assert len(telegram.command_calls) == 1
    assert [item.command for item in telegram.command_calls[0][0]] == [
        item.command for item in PUBLIC_COMMANDS
    ]
    assert len(telegram.menu_calls) == 1


async def test_start_after_onboarding_offers_main_menu_without_llm(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    user = await bot._user(777)
    async with db.session() as session:
        stored = await session.get(type(user), user.id)
        stored.onboarding_completed = True
        stored.display_name = "Друг"
    message = FakeMessage("/start")
    result = await bot.start(update_for(message, user_id=777), context())
    assert result == ConversationHandler.END
    assert callback_from(message, "nav:root") == "nav:root"
    assert fake_ai.route_calls == []
