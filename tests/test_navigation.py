import logging
from types import SimpleNamespace

import pytest
from autotester.fakes import (
    FakeBot,
    FakeCallbackQuery,
    FakeMediaCallbackQuery,
    FakeMessage,
    FakeVoice,
    ScriptedTranscription,
)
from telegram import BotCommandScopeAllPrivateChats
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
)

from future_self.access import AccessService
from future_self.bot import EVENING_WORKED, FutureSelfBot
from future_self.callback_ui import edit_callback_screen
from future_self.config import Settings
from future_self.navigation import (
    ACTIONS,
    ADVANCED_COMMANDS,
    HELP_TOPIC_LABELS,
    HELP_TOPICS,
    LEGACY_ACTIONS,
    PUBLIC_COMMANDS,
    ROOT_HELP_TOPIC_KEYS,
    SECTION_HELP_TOPICS,
    SECTIONS,
    NavigationFlowStore,
    advanced_commands,
    help_topics,
    navigation_actions,
    navigation_sections,
    public_commands,
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
    assert message.replies[-1]["text"] == "Главное меню\n\nЧто хочешь сделать?"
    markup = message.replies[-1]["reply_markup"]
    assert [
        [(button.text, button.callback_data) for button in row] for row in markup.inline_keyboard
    ] == [
        [("🌱 Сегодня", "nav:section:today"), ("✅ Задачи", "nav:section:tasks")],
        [("📝 Записи", "nav:section:records"), ("❤️ Здоровье", "nav:section:health")],
        [("🎯 Желания и визуализация", "nav:section:vision")],
        [
            ("🗂 Мои разделы", "nav:section:sections"),
            ("⚙️ Настройки", "nav:section:settings"),
        ],
        [("❓ Помощь", "nav:help")],
    ]

    expected_section_labels = {
        "today": ["Фокус на сегодня", "Задачи на сегодня", "Вечерний итог"],
        "tasks": [
            "Создать задачу",
            "Сегодня",
            "Предстоящие",
            "Просроченные",
            "Без срока",
            "Выполненные",
        ],
        "records": ["Мои записи", "Черновики", "Последнее сохранённое"],
        "health": [
            "Моё состояние",
            "Пройти check-in",
            "Найти врача",
            "Подготовиться к приёму",
            "Анализы",
            "Мои подготовки",
        ],
        "sections": ["Открыть мои разделы"],
        "settings": [
            "Мой профиль",
            "Часовой пояс",
            "Локация",
            "Настроить или продолжить настройку профиля",
        ],
    }

    for section_key, section in SECTIONS.items():
        section_message = FakeMessage()
        await bot._send_navigation_section(section_message, section_key)
        section_markup = section_message.replies[-1]["reply_markup"]
        section_callbacks = [
            button.callback_data for row in section_markup.inline_keyboard for button in row
        ]
        primary_buttons = [
            button
            for row in section_markup.inline_keyboard[: len(section.actions)]
            for button in row
        ]
        assert [button.text for button in primary_buttons] == expected_section_labels[section_key]
        assert [button.callback_data for button in primary_buttons] == [
            f"nav:action:{key}" for key in section.actions
        ]
        assert "nav:action:task_reminder_guide" not in section_callbacks
        assert f"nav:help:{SECTION_HELP_TOPICS[section_key]}" in section_callbacks
        assert "nav:root" in section_callbacks

    help_message = FakeMessage("/help")
    await bot.help_command(update_for(help_message), context())
    help_callbacks = [
        button.callback_data
        for row in help_message.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert help_callbacks == [
        *(f"nav:help:{key}" for key in ROOT_HELP_TOPIC_KEYS),
        "nav:root",
    ]
    assert [
        button.text
        for row in help_message.replies[-1]["reply_markup"].inline_keyboard[:-1]
        for button in row
    ] == [HELP_TOPIC_LABELS[key] for key in ROOT_HELP_TOPIC_KEYS]
    assert fake_ai.route_calls == []


def test_catalog_has_no_dead_buttons_duplicates_or_sensitive_callback_data(fake_ai):
    validate_catalog()
    names = [item.command for item in PUBLIC_COMMANDS]
    assert names == [
        "menu",
        "today",
        "tasks",
        "inbox",
        "vision",
        "health",
        "help",
    ]
    assert len(names) == len(set(names))
    assert len(ACTIONS) == len(set(ACTIONS))
    used_actions = {action for section in SECTIONS.values() for action in section.actions}
    assert used_actions | LEGACY_ACTIONS == set(ACTIONS)

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


def test_knowledge_catalog_is_flag_aware_and_capture_stays_advanced():
    validate_catalog(False, True, False)
    validate_catalog(True, True, True)

    disabled_public = {item.command for item in public_commands(False, False)}
    hub_public = {item.command for item in public_commands(False, True)}
    combined_public = {item.command for item in public_commands(True, True)}
    assert (
        disabled_public
        == hub_public
        == combined_public
        == {
            "menu",
            "today",
            "tasks",
            "inbox",
            "vision",
            "health",
            "help",
        }
    )

    assert "capture" not in advanced_commands(False, False)
    assert "capture" in advanced_commands(False, True)
    assert "workspaces" in advanced_commands(True, True)

    hub_only = navigation_sections(False, True, False)
    with_capture = navigation_sections(False, True, True)
    assert hub_only["sections"].actions == ("collections", "knowledge")
    assert with_capture["sections"].actions == ("collections", "knowledge", "capture")
    assert navigation_sections(True, False, False)["sections"].actions == (
        "collections",
        "spaces",
    )
    assert set(navigation_actions(False, True, False)) - set(ACTIONS) == {"knowledge"}
    assert set(navigation_actions(False, True, True)) - set(ACTIONS) == {
        "knowledge",
        "capture",
    }


def test_help_is_detailed_flag_aware_and_telegram_safe():
    disabled = help_topics(
        enable_workspace_access=False,
        enable_knowledge_hub=False,
        enable_knowledge_capture=False,
        enable_voice=False,
        enable_task_reminders=False,
    )
    assert set(ROOT_HELP_TOPIC_KEYS) <= set(disabled)
    assert set(SECTION_HELP_TOPICS.values()) <= set(disabled)
    disabled_text = "\n".join(text for _title, text in disabled.values())
    assert "/capture" not in disabled_text
    assert "/spaces" not in disabled_text
    assert "Напомни через" not in disabled_text
    assert "голос" not in disabled_text.casefold()
    assert "отключена настройкой" in disabled["tasks_section"][1]

    enabled = help_topics(True, True, True, True, True)
    assert set(enabled) == set(HELP_TOPICS)
    assert "Совместными становятся" in enabled["privacy"][1]
    assert "База знаний" in enabled["privacy"][1]
    assert "текстом или голосом" in enabled["records_section"][1]
    assert all(len(f"{title}\n\n{text}") < 4096 for title, text in enabled.values())
    assert all(len(f"nav:help:{key}".encode()) <= 64 for key in enabled)


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


@pytest.mark.parametrize(
    ("phrase", "action"),
    [
        ("Где визуализация?", "show_vision"),
        ("Как открыть визуализацию?", "show_vision"),
        ("Покажи визуализацию!", "show_vision"),
        ("Где карта желаний?", "show_vision"),
        ("Где мои задачи?", "show_tasks"),
        ("Как создать задачу?", "create_task"),
        ("Где мои записи?", "show_records"),
        ("Где здоровье?", "show_health"),
        ("Как найти врача?", "prepare_doctor"),
        ("Как подготовиться к врачу?", "prepare_doctor"),
        ("Где анализы?", "show_labs"),
        ("Где настройки?", "show_settings"),
        ("Как изменить часовой пояс?", "show_timezone"),
        ("Где мои разделы?", "show_collections"),
        ("Где совместные пространства?", "show_spaces"),
    ],
)
def test_stage_5a_natural_navigation_uses_one_exact_catalog(db, fake_ai, phrase, action):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    routed = bot.natural_command_router.route(phrase)
    assert routed is not None
    assert routed.action == action
    assert fake_ai.route_calls == []


@pytest.mark.parametrize(
    "narrative",
    [
        "У меня появилась задача позвонить врачу завтра",
        "Хочу записать желание чаще видеть море",
        "Визуализация помогла мне сформулировать идею",
        "В заметке я размышляю, где мои задачи и почему их стало много",
        "Добавь меню ужина в заметки",
        "Мне нужна помощь с покупкой билетов",
    ],
)
def test_natural_navigation_does_not_intercept_ordinary_narratives(db, fake_ai, narrative):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    assert bot.natural_command_router.route(narrative) is None
    assert not bot.natural_command_router.is_explicit_navigation_request(narrative)


@pytest.mark.parametrize(
    "phrase",
    [
        "Открой окно и проветри комнату",
        "Покажи презентацию клиенту",
        "Как создать привычку читать по утрам?",
        "Как найти время на спорт?",
        "Где поставить коробки после переезда?",
        "Покажи фотографии дизайнеру",
        "Открой документ после встречи",
        "Как создать меню питания?",
        "Как создать раздел книги?",
        "Покажи меню врача",
    ],
)
def test_navigation_verb_without_safe_ui_target_is_content(db, fake_ai, phrase):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())

    assert bot.natural_command_router.route(phrase) is None
    assert not bot.natural_command_router.is_explicit_navigation_request(phrase)


@pytest.mark.parametrize(
    "phrase",
    [
        "Где календарь?",
        "Покажи кнопку календаря",
        "Как открыть раздел в боте?",
        "Где в боте команды?",
        "Покажи задачи",
    ],
)
def test_unknown_explicit_request_requires_safe_ui_or_exact_capability(db, fake_ai, phrase):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())

    assert bot.natural_command_router.route(phrase) is None
    assert bot.natural_command_router.is_explicit_navigation_request(phrase)


def test_short_explicit_unknown_navigation_is_help_but_long_narrative_is_not(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    assert bot.natural_command_router.route("Где календарь?") is None
    assert bot.natural_command_router.is_explicit_navigation_request("Где календарь?")
    assert not bot.natural_command_router.is_explicit_navigation_request(
        "Где календарь, я записываю длинную мысль о планах на следующие несколько месяцев"
    )


@pytest.mark.parametrize(
    ("phrase", "screen_prefix", "expected_callback"),
    [
        ("Покажи визуализацию", "🎯 Желания и визуализация", "vision:add"),
        ("Как создать задачу?", "✅ Задачи", "nav:action:task_create"),
        ("Где мои записи?", "📝 Записи", "nav:action:inbox"),
        ("Как найти врача?", "❤️ Здоровье", "nav:action:doctor_find"),
        ("Как изменить часовой пояс?", "⚙️ Настройки", "nav:action:timezone"),
        ("Где календарь?", "❓ Помощь", "nav:help:quick"),
    ],
)
async def test_natural_text_gate_renders_screen_and_stops_content_pipeline(
    db,
    fake_ai,
    phrase,
    screen_prefix,
    expected_callback,
):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage(phrase)
    ctx = context()

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update_for(message), ctx)

    assert message.replies[-1]["text"].startswith(screen_prefix)
    assert callback_from(message, expected_callback) == expected_callback
    assert message.reply_text_calls == 1
    assert ctx.user_data == {}
    assert fake_ai.route_calls == []


async def test_natural_text_gate_leaves_ordinary_narrative_for_content_pipeline(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    narrative = "Хочу записать желание чаще видеть море"
    message = FakeMessage(narrative)
    ctx = context()

    assert await bot.navigation_text_gate(update_for(message), ctx) is None

    assert message.reply_text_calls == 0
    assert ctx.user_data == {}
    assert fake_ai.route_calls == []


async def test_voice_navigation_uses_same_catalog_without_ai_or_extra_chat_message(db, fake_ai):
    transcription = ScriptedTranscription()
    transcription.queue("Где мои записи?")
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    message = FakeMessage(voice=FakeVoice())

    await bot.voice(update_for(message), context())

    assert len(transcription.calls) == 1
    assert message.reply_text_calls == 1  # The single progress message becomes the screen.
    assert message.edits[-1].startswith("📝 Записи")
    assert fake_ai.route_calls == []


@pytest.mark.parametrize(
    ("phrase", "action"),
    [
        ("Покажи мои задачи", "show_tasks"),
        ("Открой задачи и напоминания", "show_tasks"),
        ("Какие задачи просрочены?", "show_overdue_tasks"),
        ("Покажи просроченные задачи", "show_overdue_tasks"),
    ],
)
def test_task_read_intents_are_deterministic_before_ai(db, fake_ai, phrase, action):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    assert bot.natural_command_router.route(phrase).action == action


async def test_navigation_callbacks_edit_in_place_and_answer_exactly_once(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage()
    query = FakeCallbackQuery("nav:section:tasks", message)

    await bot.navigation_action(update_for(message, query=query), context())

    assert query.answers == [(None, False)]
    assert len(query.edits) == 1
    assert query.edits[0].startswith("✅ Задачи")
    assert message.reply_text_calls == 0
    assert all(reply.get("text") != "Навигация" for reply in message.replies)


async def test_message_not_modified_is_success_without_duplicate(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage()

    class NotModifiedQuery(FakeCallbackQuery):
        async def edit_message_text(self, text, **kwargs):
            del text, kwargs
            raise BadRequest("Message is not modified")

    query = NotModifiedQuery("nav:root", message)
    await bot.navigation_action(update_for(message, query=query), context())

    assert query.answers == [(None, False)]
    assert message.reply_text_calls == 0
    assert query.markup_removed == 0


async def test_generic_navigation_edit_error_is_type_only_logged_without_duplicate(
    db, fake_ai, caplog
):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage()
    private_text = "PRIVATE NAVIGATION TEXT MUST NOT BE LOGGED"

    class FailedQuery(FakeCallbackQuery):
        async def edit_message_text(self, text, **kwargs):
            del text, kwargs
            raise TelegramError(private_text)

    query = FailedQuery("nav:root", message)
    with caplog.at_level(logging.WARNING):
        await bot.navigation_action(update_for(message, query=query), context())

    assert query.answers == [(None, False)]
    assert message.reply_text_calls == 0
    assert query.markup_removed == 0
    assert "TelegramError" in caplog.text
    assert private_text not in caplog.text


async def test_shared_feature_callback_editor_never_duplicates_on_generic_failure(caplog):
    message = FakeMessage()
    private_text = "PRIVATE CALLBACK BODY MUST NOT BE LOGGED"

    class FailedQuery(FakeCallbackQuery):
        async def edit_message_text(self, text, **kwargs):
            del text, kwargs
            raise TelegramError(private_text)

    query = FailedQuery("task:list:today:0", message)
    with caplog.at_level(logging.WARNING):
        changed = await edit_callback_screen(
            query,
            "safe screen",
            None,
            operation="test-feature",
        )

    assert changed is False
    assert message.reply_text_calls == 0
    assert query.markup_removed == 0
    assert "TelegramError" in caplog.text
    assert private_text not in caplog.text


async def test_media_navigation_uses_caption_edit_and_replacement_only_for_size_limit(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage()
    query = FakeMediaCallbackQuery("nav:root", message)

    await bot.navigation_action(update_for(message, query=query), context())

    assert query.answers == [(None, False)]
    assert query.text_attempts == 1
    assert query.caption_edits == ["Главное меню\n\nЧто хочешь сделать?"]
    assert query.markup_removed == 0
    assert message.reply_text_calls == 0

    oversized_message = FakeMessage()
    oversized_query = FakeMediaCallbackQuery("nav:root", oversized_message)
    replaced = await bot._edit_or_send(
        oversized_query,
        "x" * 1025,
        bot._root_keyboard(),
    )
    assert replaced is True
    assert oversized_query.text_attempts == 1
    assert oversized_query.caption_edits == []
    assert oversized_query.markup_removed == 1
    assert oversized_message.reply_text_calls == 1
    assert oversized_message.deleted is False


@pytest.mark.parametrize(
    ("legacy_key", "expected_title"),
    [
        ("day", "🌱 Сегодня"),
        ("ideas", "📝 Записи"),
        ("doctor", "❤️ Здоровье"),
        ("profile", "⚙️ Настройки"),
        ("collections", "🗂 Мои разделы"),
        ("spaces", "🗂 Мои разделы"),
    ],
)
async def test_legacy_section_callbacks_redirect_safely(db, fake_ai, legacy_key, expected_title):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage()
    query = FakeCallbackQuery(f"nav:section:{legacy_key}", message)

    await bot.navigation_action(update_for(message, query=query), context())

    assert query.answers == [(None, False)]
    assert query.edits[-1].startswith(expected_title)
    assert message.reply_text_calls == 0


async def test_legacy_vision_callback_opens_new_menu_and_stale_root_respects_flow(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage()
    legacy = FakeCallbackQuery("nav:action:vision", message)

    await bot.navigation_action(update_for(message, query=legacy), context())

    assert legacy.answers == [(None, False)]
    assert legacy.edits[-1].startswith("🎯 Желания и визуализация")
    assert message.reply_text_calls == 0

    flow_context = context()
    flow_context.user_data["health_checkin"] = {"energy": 4}
    stale_root = FakeCallbackQuery("nav:section:vision", message)
    await bot.navigation_action(
        update_for(message, query=stale_root),
        flow_context,
    )
    assert stale_root.answers == [(None, False)]
    assert stale_root.edits[-1].startswith("Сейчас не завершён сценарий")
    assert flow_context.user_data["health_checkin"] == {"energy": 4}


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


async def test_continue_keeps_flow_and_edits_same_message_safely(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    ctx = context()
    ctx.user_data["doctor_prepare"] = {"reason": "private"}
    message = FakeMessage("/menu")
    update = update_for(message)
    await bot.menu_command(update, ctx)
    data = callback_from(message, "nav:flow:continue:")
    query = FakeCallbackQuery(data, message)
    before_reply_calls = message.reply_text_calls
    assert await bot.navigation_action(update_for(message, query=query), ctx) is None
    assert ctx.user_data["doctor_prepare"] == {"reason": "private"}
    assert query.answers == [(None, False)]
    assert message.reply_text_calls == before_reply_calls
    assert "private" not in str(message.replies[-1])


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
        "start",
        "menu",
        "help",
    ]
    assert isinstance(telegram.command_calls[0][1]["scope"], BotCommandScopeAllPrivateChats)
    assert len(telegram.menu_calls) == 1


async def test_start_after_onboarding_offers_main_menu_without_llm(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    user = await bot._user(777)
    await AccessService(db).grant_subscriber(777, source="test")
    async with db.session() as session:
        stored = await session.get(type(user), user.id)
        stored.onboarding_completed = True
        stored.display_name = "Друг"
    message = FakeMessage("/start")
    result = await bot.start(update_for(message, user_id=777), context())
    assert result == ConversationHandler.END
    assert callback_from(message, "nav:root") == "nav:root"
    assert fake_ai.route_calls == []


async def test_evening_reflection_starts_from_main_menu_button(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    ctx = context()
    message = FakeMessage()
    query = FakeCallbackQuery("nav:action:evening", message)

    result = await bot.navigation_evening_entry(update_for(message, query=query), ctx)

    assert result == EVENING_WORKED
    assert ctx.user_data["evening"] == {}
    assert any("Что сегодня получилось" in reply["text"] for reply in message.replies)
    assert fake_ai.route_calls == []

    blocked_context = context()
    blocked_context.user_data["health_checkin"] = {"energy": 4}
    blocked_message = FakeMessage()
    blocked_query = FakeCallbackQuery("nav:action:evening", blocked_message)

    blocked_result = await bot.navigation_evening_entry(
        update_for(blocked_message, query=blocked_query), blocked_context
    )

    assert blocked_result is None
    assert blocked_context.user_data["health_checkin"] == {"energy": 4}
    assert "evening" not in blocked_context.user_data
    assert callback_from(blocked_message, "nav:flow:continue:").startswith("nav:flow:continue:")
