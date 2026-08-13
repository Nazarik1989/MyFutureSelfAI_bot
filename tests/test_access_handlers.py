from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select
from telegram import BotCommandScopeChat
from telegram.error import BadRequest, TelegramError
from telegram.ext import ApplicationHandlerStop, ConversationHandler

import future_self.access_handlers as access_handlers_module
from future_self.access import ADMIN, BLOCKED, GUEST, SUBSCRIBER, AccessService
from future_self.access_handlers import (
    FULL_VERSION_ALERT,
    GUEST_CALLBACK_ROUTES,
    GUEST_DISABLED_TEXT,
    GUEST_FEATURE_TEXTS,
    GUEST_FIRST_STEP_INPUT_TEXT,
    GUEST_GLOBAL_EXHAUSTED_TEXT,
    GUEST_LIFETIME_EXHAUSTED_TEXT,
    GUEST_MEDIA_NOTICE,
    GUEST_PROVIDER_ERROR_TEXT,
    GUEST_ROOT_TEXT,
    GUEST_TEMPORARY_ERROR_TEXT,
    GUEST_THOUGHT_INPUT_TEXT,
    SERVICE_UNAVAILABLE_TEXT,
    STALE_GUEST_ALERT,
)
from future_self.bot import ONBOARDING_INPUT, FutureSelfBot
from future_self.config import Settings
from future_self.models import OnboardingState, User


class ForbiddenTranscription:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls += 1
        raise AssertionError("guest media must not reach STT")


class SpyMedia:
    def __init__(self) -> None:
        self.get_file_calls = 0
        self.download_calls = 0

    async def get_file(self) -> Any:
        self.get_file_calls += 1

        async def download_as_bytearray() -> bytearray:
            self.download_calls += 1
            return bytearray(b"private-media")

        return SimpleNamespace(download_as_bytearray=download_as_bytearray)


class GateMessage:
    def __init__(
        self,
        text: str | None = None,
        *,
        voice: Any = None,
        audio: Any = None,
        photo: list[Any] | None = None,
        document: Any = None,
        users_shared: Any = None,
        message_id: int = 100,
    ) -> None:
        self.text = text
        self.voice = voice
        self.audio = audio
        self.photo = photo or []
        self.document = document
        self.users_shared = users_shared
        self.message_id = message_id
        self.replies: list[dict[str, Any]] = []

    async def reply_text(self, text: str, **kwargs: Any) -> GateMessage:
        self.replies.append({"text": text, **kwargs})
        return self


class GateQuery:
    def __init__(self, data: str, message: GateMessage, *, fail_edit: bool = False) -> None:
        self.data = data
        self.message = message
        self.fail_edit = fail_edit
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[dict[str, Any]] = []

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        if self.fail_edit:
            raise BadRequest("private Telegram edit detail")
        self.edits.append({"text": text, **kwargs})


class ScopeBot:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.command_calls: list[tuple[list[Any], dict[str, Any]]] = []

    async def set_my_commands(self, commands: list[Any], **kwargs: Any) -> None:
        self.command_calls.append((commands, kwargs))
        if self.fail:
            raise TelegramError("private Telegram API detail")


def make_bot(
    db: Any, fake_ai: Any, **overrides: Any
) -> tuple[FutureSelfBot, ForbiddenTranscription]:
    values = {
        "_env_file": None,
        "telegram_bot_token": "123456:test-token",
        "ai_api_key": "test-key",
        "database_url": db.url,
    }
    values.update(overrides)
    transcription = ForbiddenTranscription()
    return FutureSelfBot(Settings(**values), db, fake_ai, transcription), transcription


def update_for(
    message: GateMessage | None,
    *,
    user_id: int | None = 7001,
    chat_id: int | None = 8001,
    query: GateQuery | None = None,
    update_id: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        update_id=update_id,
        effective_user=SimpleNamespace(id=user_id) if user_id is not None else None,
        effective_chat=(
            SimpleNamespace(id=chat_id, type="private") if chat_id is not None else None
        ),
        effective_message=message,
        message=message,
        callback_query=query,
    )


def context(bot: ScopeBot | None = None, *, args: list[str] | None = None) -> SimpleNamespace:
    telegram = bot or ScopeBot()
    return SimpleNamespace(
        user_data={},
        args=args or [],
        bot=telegram,
        application=SimpleNamespace(create_task=lambda *_args, **_kwargs: None),
    )


async def set_user_state(
    bot: FutureSelfBot,
    telegram_id: int,
    *,
    tier: str | None = None,
    completed: bool | None = None,
) -> User:
    user = await bot._user(telegram_id)
    if tier == SUBSCRIBER:
        await AccessService(bot.db).grant_subscriber(telegram_id, source="test")
    elif tier == ADMIN:
        await AccessService(bot.db).grant_admin(telegram_id, source="test")
    elif tier == BLOCKED:
        await AccessService(bot.db).block(telegram_id, source="test")
    if completed is not None:
        async with bot.db.session() as session:
            stored = await session.get(User, user.id)
            assert stored is not None
            stored.onboarding_completed = completed
    async with bot.db.sessions() as session:
        stored = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert stored is not None
        return stored


async def run_guest_gate(
    bot: FutureSelfBot,
    update: SimpleNamespace,
    *,
    scope_bot: ScopeBot | None = None,
) -> ScopeBot:
    telegram = scope_bot or ScopeBot()
    with pytest.raises(ApplicationHandlerStop):
        await bot.access_gate(update, context(telegram))
    return telegram


async def test_new_and_completed_guest_are_stopped_at_guest_root(db, fake_ai):
    bot, _transcription = make_bot(db, fake_ai)
    first = GateMessage("/start@FutureSelfBot")
    await run_guest_gate(bot, update_for(first, user_id=7001, chat_id=7001))
    assert first.replies[-1]["text"] == GUEST_ROOT_TEXT

    user = await set_user_state(bot, 7001, completed=True)
    assert user.access_tier == GUEST
    second = GateMessage("/start")
    await run_guest_gate(bot, update_for(second, user_id=7001, chat_id=7001))
    assert second.replies[-1]["text"] == GUEST_ROOT_TEXT
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(OnboardingState.id))) == 0
    assert fake_ai.route_calls == []


async def test_direct_start_routes_full_tiers_and_blocks_guest_and_blocked(db, fake_ai):
    bot, _transcription = make_bot(db, fake_ai)

    guest_message = GateMessage("/start")
    assert (
        await bot.start(update_for(guest_message, user_id=7010, chat_id=7010), context())
        == ConversationHandler.END
    )
    assert guest_message.replies[-1]["text"] == GUEST_ROOT_TEXT

    await set_user_state(bot, 7011, tier=SUBSCRIBER, completed=False)
    subscriber_message = GateMessage("/start")
    assert (
        await bot.start(update_for(subscriber_message, user_id=7011, chat_id=7011), context())
        == ONBOARDING_INPUT
    )

    await set_user_state(bot, 7012, tier=SUBSCRIBER, completed=True)
    completed_message = GateMessage("/start")
    assert (
        await bot.start(update_for(completed_message, user_id=7012, chat_id=7012), context())
        == ConversationHandler.END
    )
    assert "nav:root" in {
        button.callback_data
        for row in completed_message.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    }

    await set_user_state(bot, 7013, tier=BLOCKED)
    blocked_message = GateMessage("/start")
    assert (
        await bot.start(update_for(blocked_message, user_id=7013, chat_id=7013), context())
        == ConversationHandler.END
    )
    assert "Доступ к боту ограничен" in blocked_message.replies[-1]["text"]


async def test_admin_passes_gate_and_blocked_is_stopped(db, fake_ai):
    bot, _transcription = make_bot(db, fake_ai)
    await set_user_state(bot, 7020, tier=ADMIN)
    admin_message = GateMessage("/today")
    telegram = ScopeBot()
    await bot.access_gate(update_for(admin_message, user_id=7020, chat_id=7020), context(telegram))
    assert admin_message.replies == []
    assert len(telegram.command_calls) == 1

    await set_user_state(bot, 7021, tier=BLOCKED)
    blocked_message = GateMessage("/today")
    await run_guest_gate(bot, update_for(blocked_message, user_id=7021, chat_id=7021))
    rendered = blocked_message.replies[-1]
    assert "Доступ к боту ограничен" in rendered["text"]
    assert "guest:" not in repr(rendered["reply_markup"])


async def test_full_access_gate_syncs_memory_before_other_ephemeral_flows(
    db,
    fake_ai,
    monkeypatch,
):
    bot, _transcription = make_bot(db, fake_ai)
    telegram_id = 7022
    chat_id = 8022
    await set_user_state(bot, telegram_id, tier=ADMIN)
    calls: list[str] = []

    async def memory_sync(*_args, **_kwargs):
        calls.append("memory")

    async def nova_sync(*_args, **_kwargs):
        calls.append("nova")

    async def reminder_sync(*_args, **_kwargs):
        calls.append("reminder")

    async def command_sync(*_args, **_kwargs):
        calls.append("commands")

    monkeypatch.setattr(bot, "nova_memory_sync_access", memory_sync)
    monkeypatch.setattr(bot, "nova_sync_access", nova_sync)
    monkeypatch.setattr(bot, "reminder_sync_access", reminder_sync)
    monkeypatch.setattr(bot, "_sync_access_commands", command_sync)

    await bot.access_gate(
        update_for(GateMessage("/menu"), user_id=telegram_id, chat_id=chat_id),
        context(),
    )

    assert calls == ["memory", "nova", "reminder", "commands"]


@pytest.mark.parametrize("command", ["/today", "/vision", "/onboarding", "/capture"])
async def test_guest_full_commands_are_consumed_without_ai(db, fake_ai, command):
    bot, _transcription = make_bot(db, fake_ai)
    message = GateMessage(command)
    await run_guest_gate(bot, update_for(message, user_id=7030, chat_id=7030))
    assert "доступна в полной версии" in message.replies[-1]["text"]
    assert fake_ai.route_calls == []


@pytest.mark.parametrize(
    ("command", "expected_text"),
    [
        ("/menu@FutureSelfBot", GUEST_ROOT_TEXT),
        ("/help@FutureSelfBot", "✨ Nova"),
    ],
)
async def test_guest_safe_commands_support_bot_username(db, fake_ai, command, expected_text):
    bot, _transcription = make_bot(db, fake_ai)
    message = GateMessage(command)
    await run_guest_gate(bot, update_for(message, user_id=7031, chat_id=7031))
    assert expected_text in message.replies[-1]["text"]


@pytest.mark.parametrize("route", sorted(GUEST_CALLBACK_ROUTES))
async def test_all_guest_callback_routes_answer_and_edit(db, fake_ai, route):
    bot, _transcription = make_bot(db, fake_ai)
    message = GateMessage()
    query = GateQuery(route, message)
    await run_guest_gate(
        bot,
        update_for(message, user_id=7040, chat_id=7040, query=query),
    )
    assert query.answers == [(None, False)]
    assert len(query.edits) == 1
    rendered = query.edits[0]
    if route == "guest:access":
        assert "7040" in rendered["text"]
        urls = {
            button.url
            for row in rendered["reply_markup"].inline_keyboard
            for button in row
            if button.url
        }
        assert urls == {"https://t.me/Nazar_38rus", "https://naz-ai-lab.ru"}
    else:
        assert "7040" not in rendered["text"]
    if route in {"guest:demo:thought", "guest:demo:first-step"}:
        expected = (
            GUEST_THOUGHT_INPUT_TEXT
            if route == "guest:demo:thought"
            else GUEST_FIRST_STEP_INPUT_TEXT
        )
        assert rendered["text"] == expected


def test_guest_root_text_and_markup_are_exact():
    assert (
        GUEST_ROOT_TEXT
        == """👋 Я — «Моя будущая версия»

Личный AI-ассистент, который помогает разгружать голову, видеть главное и превращать желаемое будущее в конкретные действия.

Я умею работать с мыслями, задачами, целями, картой желаний, самочувствием и подготовкой к важным событиям.

В гостевом режиме доступны 2 бесплатных AI-разбора."""
    )
    buttons = [
        button for row in access_handlers_module._ROOT_MARKUP.inline_keyboard for button in row
    ]
    assert [(button.text, button.callback_data, button.url) for button in buttons] == [
        ("✨ Попробовать бесплатно", "guest:demos", None),
        ("🧭 Что я умею", "guest:features", None),
        ("⚙️ Как это работает", "guest:how", None),
        ("💬 Подписка или свой бот", "guest:access", None),
        ("🧩 Другие проекты", None, "https://naz-ai-lab.ru"),
    ]


def test_guest_limit_and_temporary_messages_are_exact():
    assert (
        GUEST_LIFETIME_EXHAUSTED_TEXT
        == """🎁 Бесплатные разборы закончились

Вы уже использовали оба бесплатных разбора.

Чтобы получить подписку или заказать такого же собственного бота, напишите Назару Сергеевичу."""
    )
    assert (
        GUEST_PROVIDER_ERROR_TEXT
        == """Сейчас обработать запрос не получилось.

Количество бесплатных разборов не изменилось. Отправьте текст ещё раз."""
    )
    assert (
        GUEST_DISABLED_TEXT
        == """AI-демонстрация временно недоступна.

Количество бесплатных разборов не изменилось. Попробуйте немного позже."""
    )
    assert (
        GUEST_GLOBAL_EXHAUSTED_TEXT
        == """На сегодня общий лимит бесплатных разборов достигнут.

Количество ваших бесплатных разборов не изменилось. Попробуйте позже — дневной лимит обновится автоматически."""
    )
    assert (
        GUEST_TEMPORARY_ERROR_TEXT
        == """Сервис временно недоступен.

Количество бесплатных разборов не изменилось. Попробуйте ещё раз немного позже."""
    )


async def test_guest_features_have_medical_disclaimers_and_exact_routes(db, fake_ai):
    assert set(GUEST_FEATURE_TEXTS) == {
        "guest:feature:day",
        "guest:feature:notes",
        "guest:feature:tasks",
        "guest:feature:vision",
        "guest:feature:health",
        "guest:feature:doctor",
    }
    assert "не ставит диагнозы" in GUEST_FEATURE_TEXTS["guest:feature:health"]
    assert "не ставя диагнозов" in GUEST_FEATURE_TEXTS["guest:feature:doctor"]


@pytest.mark.parametrize(
    "callback_data",
    [
        "nav:root",
        "vision:old",
        "task:old",
        "space:old",
        "spacei:old",
        "kh:old",
        "profile:old",
        "timezone:update:old",
    ],
)
async def test_full_callbacks_are_replaced_with_guest_root(db, fake_ai, callback_data):
    bot, _transcription = make_bot(db, fake_ai)
    message = GateMessage()
    query = GateQuery(callback_data, message)
    await run_guest_gate(bot, update_for(message, user_id=7050, chat_id=7050, query=query))
    assert query.answers == [(FULL_VERSION_ALERT, True)]
    assert query.edits[-1]["text"] == GUEST_ROOT_TEXT


async def test_forged_guest_callback_is_stale_and_does_not_edit(db, fake_ai):
    bot, _transcription = make_bot(db, fake_ai)
    message = GateMessage()
    query = GateQuery("guest:forged", message)
    await run_guest_gate(bot, update_for(message, user_id=7051, chat_id=7051, query=query))
    assert query.answers == [(STALE_GUEST_ALERT, True)]
    assert query.edits == []


async def test_guest_callback_edit_error_has_no_reply_fallback(db, fake_ai):
    bot, _transcription = make_bot(db, fake_ai)
    message = GateMessage()
    query = GateQuery("guest:features", message, fail_edit=True)
    await run_guest_gate(bot, update_for(message, user_id=7052, chat_id=7052, query=query))
    assert query.answers == [(None, False)]
    assert message.replies == []


@pytest.mark.parametrize("media_kind", ["voice", "audio", "photo", "document"])
async def test_guest_media_is_not_downloaded_or_sent_to_stt_or_ai(db, fake_ai, media_kind):
    bot, transcription = make_bot(db, fake_ai)
    media = SpyMedia()
    kwargs: dict[str, Any]
    if media_kind == "photo":
        kwargs = {"photo": [media]}
    else:
        kwargs = {media_kind: media}
    message = GateMessage(**kwargs)
    await run_guest_gate(bot, update_for(message, user_id=7060, chat_id=7060))
    assert message.replies[-1]["text"] == GUEST_MEDIA_NOTICE
    assert media.get_file_calls == 0
    assert media.download_calls == 0
    assert transcription.calls == 0
    assert fake_ai.route_calls == []


async def test_guest_text_and_status_update_never_reach_features(db, fake_ai):
    bot, transcription = make_bot(db, fake_ai)
    text_message = GateMessage("Сделай мне план")
    await run_guest_gate(bot, update_for(text_message, user_id=7061, chat_id=7061))
    assert "выберите кнопку" in text_message.replies[-1]["text"].casefold()

    status_message = GateMessage(users_shared=SimpleNamespace(user_ids=[1]))
    await run_guest_gate(bot, update_for(status_message, user_id=7061, chat_id=7061))
    assert status_message.replies == []
    assert transcription.calls == 0
    assert fake_ai.route_calls == []


async def test_gate_fails_closed_on_missing_identity_and_database_error(
    db, fake_ai, monkeypatch, caplog
):
    bot, _transcription = make_bot(db, fake_ai)
    missing = GateMessage("/start")
    with pytest.raises(ApplicationHandlerStop):
        await bot.access_gate(update_for(missing, user_id=None), context())
    assert missing.replies[-1]["text"] == SERVICE_UNAVAILABLE_TEXT

    private_detail = "postgresql://secret-user:secret-password@private-db"

    async def fail_user(_telegram_id: int) -> User:
        raise RuntimeError(private_detail)

    monkeypatch.setattr(bot, "_user", fail_user)
    failed = GateMessage("private prompt text")
    with pytest.raises(ApplicationHandlerStop), caplog.at_level("ERROR"):
        await bot.access_gate(update_for(failed, user_id=7070, chat_id=7070), context())
    assert failed.replies[-1]["text"] == SERVICE_UNAVAILABLE_TEXT
    assert private_detail not in caplog.text
    assert "private prompt text" not in caplog.text

    callback_message = GateMessage()
    callback = GateQuery("guest:root", callback_message)
    with pytest.raises(ApplicationHandlerStop):
        await bot.access_gate(
            update_for(
                callback_message,
                user_id=7070,
                chat_id=7070,
                query=callback,
            ),
            context(),
        )
    assert callback.answers == [(SERVICE_UNAVAILABLE_TEXT, True)]
    assert callback.edits == []


async def test_command_scope_cache_tracks_tier_and_version(db, fake_ai):
    bot, _transcription = make_bot(
        db,
        fake_ai,
        enable_workspace_access=True,
        enable_knowledge_hub=True,
    )
    telegram = ScopeBot()
    update = update_for(GateMessage("/start"), user_id=7080, chat_id=8080)
    await run_guest_gate(bot, update, scope_bot=telegram)
    await run_guest_gate(bot, update, scope_bot=telegram)
    assert len(telegram.command_calls) == 1
    guest_commands = [item.command for item in telegram.command_calls[0][0]]
    assert guest_commands == ["start", "menu", "help"]
    assert isinstance(telegram.command_calls[0][1]["scope"], BotCommandScopeChat)

    await AccessService(db).grant_subscriber(7080, source="test")
    await bot.access_gate(update, context(telegram))
    assert len(telegram.command_calls) == 2
    subscriber_commands = [item.command for item in telegram.command_calls[-1][0]]
    assert subscriber_commands == [
        "menu",
        "today",
        "tasks",
        "inbox",
        "vision",
        "health",
        "help",
    ]

    await AccessService(db).grant_admin(7080, source="test")
    await bot.access_gate(update, context(telegram))
    assert len(telegram.command_calls) == 3
    assert [item.command for item in telegram.command_calls[-1][0]] == [
        item.command for item in telegram.command_calls[-2][0]
    ]

    await AccessService(db).block(7080, source="test")
    await run_guest_gate(bot, update, scope_bot=telegram)
    assert len(telegram.command_calls) == 4
    assert [item.command for item in telegram.command_calls[-1][0]] == ["start"]


async def test_command_scope_telegram_error_is_not_cached_or_used_as_authorization(
    db, fake_ai, caplog
):
    bot, _transcription = make_bot(db, fake_ai)
    await set_user_state(bot, 7090, tier=SUBSCRIBER)
    telegram = ScopeBot(fail=True)
    update = update_for(GateMessage("/today"), user_id=7090, chat_id=8090)
    with caplog.at_level("ERROR"):
        await bot.access_gate(update, context(telegram))
        await bot.access_gate(update, context(telegram))
    assert len(telegram.command_calls) == 2
    assert 8090 not in bot._access_scope_cache
    assert "private Telegram API detail" not in caplog.text


async def test_subscriber_command_scope_runtime_error_does_not_break_full_access(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    bot, _transcription = make_bot(db, fake_ai)
    await set_user_state(bot, 7091, tier=SUBSCRIBER)
    telegram = ScopeBot()
    private_detail = "PRIVATE_SCOPE_RUNTIME_DETAIL"

    async def fail_scope(*args, **kwargs):
        raise RuntimeError(private_detail)

    monkeypatch.setattr(telegram, "set_my_commands", fail_scope)
    update = update_for(GateMessage("/today"), user_id=7091, chat_id=8091)
    with caplog.at_level("ERROR"):
        await bot.access_gate(update, context(telegram))
    assert 8091 not in bot._access_scope_cache
    assert private_detail not in caplog.text


async def test_direct_workspace_deep_link_rejects_guest_before_service(db, fake_ai):
    bot, _transcription = make_bot(db, fake_ai, enable_workspace_access=True)
    called = False

    async def forbidden_invitation(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("guest invitation must not be processed")

    bot.workspace_service.issue_incoming_actions = forbidden_invitation
    message = GateMessage("/start space_private-token")
    handled = await bot.workspace_start_invitation(
        update_for(message, user_id=7100, chat_id=7100),
        context(args=["space_private-token"]),
    )
    assert handled is True
    assert called is False
    assert message.replies[-1]["text"] == GUEST_ROOT_TEXT
