import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop

from future_self.bot import (
    NOVA_MEMORY_APPLICATION_CHANGED_TEXT,
    NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
    FutureSelfBot,
)
from future_self.config import Settings
from future_self.domain import IntentRouter, PendingIntent
from future_self.models import DraftInboxItem, InboxItem, NovaMemoryChange
from future_self.nova_memory_application import build_nova_memory_projection
from future_self.nova_memory_handlers import NOVA_MEMORY_ACCESS_CHANGED_TEXT
from future_self.schemas import AssistantAnswer, IntentResult


class FakeMessage:
    def __init__(
        self,
        text: str | None = None,
        *,
        voice=None,
        chat_id: int = 10_501,
        message_id: int = 90_001,
    ):
        self.text = text
        self.voice = voice
        self.audio = None
        self.chat = SimpleNamespace(id=chat_id)
        self.message_id = message_id
        self.replies: list[dict[str, object]] = []
        self.edits: list[str] = []
        self.edit_kwargs: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self

    async def edit_text(self, text: str, **kwargs):
        self.edits.append(text)
        self.edit_kwargs.append(kwargs)


class FakeCallbackQuery:
    def __init__(self, data: str, message: FakeMessage):
        self.data = data
        self.message = message
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[str] = []
        self.edit_kwargs: list[dict[str, object]] = []
        self.markup_removed = 0

    async def answer(self, text: str | None = None, show_alert: bool = False):
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs):
        self.edits.append(text)
        self.edit_kwargs.append(kwargs)

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_removed += 1


class FakeTelegramFile:
    async def download_as_bytearray(self):
        return bytearray(b"voice")


class FakeVoice:
    duration = 3
    file_size = 5
    mime_type = "audio/ogg"
    file_name = "voice.ogg"

    async def get_file(self):
        return FakeTelegramFile()


class GreetingTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "Привет"


class IdeaTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "Мне пришла идея сделать совместное пространство для друзей"


class CorrectedTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "нужно заниматься спортом 3 раза в неделю"


class NavigationTranscription:
    enabled = True

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls += 1
        return self.text


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "telegram_bot_token": "123456:TEST",
        "ai_api_key": "test-key",
        "ai_model": "test-model",
        "transcription_provider": "disabled",
        "intent_confidence_threshold": 0.70,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


NON_NAVIGATION_UI_VERB_PHRASES = (
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
)


def update_for(message: FakeMessage, user_id: int = 501):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=user_id + 10_000),
    )


def preview_callback(message: FakeMessage, action: str) -> str:
    markup = message.replies[-1]["reply_markup"]
    return next(
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data.startswith(f"inbox:{action}:")
    )


def intent_answer_callback(
    message: FakeMessage,
    *,
    user_id: int,
    token: str = "callback-token",
    raw_text: str = "Как лучше спланировать неделю?",
) -> tuple[object, SimpleNamespace, PendingIntent, FakeCallbackQuery]:
    pending = PendingIntent(
        token=token,
        raw_text=raw_text,
        source="text",
        result=IntentResult(
            intent="unknown",
            confidence=0.2,
            topic="планирование недели",
        ),
        canonical_chat_id=user_id + 10_000,
        canonical_message_id=message.message_id,
    )
    message.chat = SimpleNamespace(id=user_id + 10_000)
    context = SimpleNamespace(user_data={f"intent:{token}": pending})
    query = FakeCallbackQuery(f"intent:answer:{token}", message)
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=user_id + 10_000),
    )
    return update, context, pending, query


async def enable_admin_memory(
    bot: FutureSelfBot,
    *,
    user_id: int,
    content: str = "Отвечай кратко и по пунктам",
):
    await bot._user(user_id)
    access = await bot.access_service.grant_admin(user_id, source="test")
    created = await bot.nova_memory_service.create(
        telegram_actor_id=user_id,
        expected_access_version=access.status.access_version,
        category="interaction",
        content=content,
        important=True,
    )
    assert created.status == "created"
    return access.status


async def mutate_stage7c_memory(bot, access, user_id: int, mutation: str, *, suffix: str):
    page = await bot.nova_memory_service.list(
        telegram_actor_id=user_id,
    )
    assert page.status == "ok"
    assert len(page.items) == 1
    item = page.items[0]
    if mutation == "create":
        result = await bot.nova_memory_service.create(
            telegram_actor_id=user_id,
            expected_access_version=access.access_version,
            category="about_me",
            content=f"NEW_STAGE7C_CALLBACK_{suffix}_MEMORY",
        )
        assert result.status == "created"
    elif mutation == "edit":
        result = await bot.nova_memory_service.update(
            telegram_actor_id=user_id,
            public_id=item.public_id,
            expected_version=item.version,
            expected_access_version=access.access_version,
            content=f"EDITED_STAGE7C_CALLBACK_{suffix}_MEMORY",
        )
        assert result.status == "updated"
    elif mutation == "toggle":
        result = await bot.nova_memory_service.set_important(
            telegram_actor_id=user_id,
            public_id=item.public_id,
            expected_version=item.version,
            expected_access_version=access.access_version,
            important=not item.important,
        )
        assert result.status == "importance_changed"
    elif mutation == "delete":
        result = await bot.nova_memory_service.delete(
            telegram_actor_id=user_id,
            public_id=item.public_id,
            expected_version=item.version,
            expected_access_version=access.access_version,
        )
        assert result.status == "deleted"
    else:
        assert mutation == "delete_all"
        assert page.collection_revision is not None
        result = await bot.nova_memory_service.delete_all(
            telegram_actor_id=user_id,
            expected_access_version=access.access_version,
            expected_collection_revision=page.collection_revision,
        )
        assert result.status == "deleted_all"
    assert result.affected_count == 1


async def inbox_count(db) -> int:
    async with db.sessions() as session:
        return int(await session.scalar(select(func.count(InboxItem.id))))


async def draft_count(db) -> int:
    async with db.sessions() as session:
        return int(await session.scalar(select(func.count(DraftInboxItem.id))))


@pytest.mark.parametrize(
    ("phrase", "expected_heading"),
    [
        ("Покажи мои записи", "📝 Записи"),
        ("Где настройки", "⚙️ Настройки"),
        ("Где календарь?", "✨ Nova"),
    ],
)
async def test_navigation_text_gate_consumes_known_and_explicit_unknown_requests_before_content(
    db,
    fake_ai,
    phrase,
    expected_heading,
):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage(phrase)

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update_for(message, 4901), SimpleNamespace(user_data={}))

    assert expected_heading in str(message.replies[-1]["text"])
    assert fake_ai.route_calls == []
    assert await draft_count(db) == 0
    assert await inbox_count(db) == 0


async def test_active_flow_fences_section_navigation_without_ai_or_content_capture(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    state = {"energy": 7}
    context = SimpleNamespace(user_data={"health_checkin": state})
    message = FakeMessage("Покажи мои записи")

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update_for(message, 4902), context)

    assert context.user_data["health_checkin"] is state
    assert "не завершён сценарий" in str(message.replies[-1]["text"])
    assert fake_ai.route_calls == []
    assert await draft_count(db) == 0


@pytest.mark.parametrize("narrative", NON_NAVIGATION_UI_VERB_PHRASES)
async def test_navigation_verbs_without_ui_target_continue_to_content_routing(
    db, fake_ai, narrative
):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage(narrative)
    update = update_for(message, 4903)
    context = SimpleNamespace(user_data={})

    await bot.navigation_text_gate(update, context)
    assert message.replies == []

    await bot.text(update, context)

    assert message.replies
    assert not any(str(reply["text"]).startswith("❓ Помощь") for reply in message.replies)
    assert not any(str(reply["text"]).startswith("✨ Nova") for reply in message.replies)
    assert fake_ai.route_calls == [] or fake_ai.route_calls[-1][0] == narrative
    assert await inbox_count(db) == 0


@pytest.mark.parametrize(
    ("phrase", "expected_heading"),
    [
        ("Где настройки", "⚙️ Настройки"),
        ("Где календарь?", "✨ Nova"),
    ],
)
async def test_voice_navigation_reuses_progress_message_without_ai_or_content_capture(
    db,
    fake_ai,
    phrase,
    expected_heading,
):
    transcription = NavigationTranscription(phrase)
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    message = FakeMessage(voice=FakeVoice())

    await bot.voice(update_for(message, 4904), SimpleNamespace(user_data={}))

    assert transcription.calls == 1
    assert len(message.replies) == 1
    assert message.replies[0]["text"] == "Расшифровываю голосовую мысль…"
    assert expected_heading in message.edits[-1]
    assert message.edit_kwargs[-1].get("reply_markup") is not None
    assert fake_ai.route_calls == []
    assert await draft_count(db) == 0
    assert await inbox_count(db) == 0


@pytest.mark.parametrize("phrase", NON_NAVIGATION_UI_VERB_PHRASES)
async def test_voice_navigation_verbs_without_ui_target_continue_to_content_routing(
    db,
    fake_ai,
    phrase,
):
    transcription = NavigationTranscription(phrase)
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    message = FakeMessage(voice=FakeVoice())

    await bot.voice(update_for(message, 4905), SimpleNamespace(user_data={}))

    assert transcription.calls == 1
    assert message.replies
    assert not any(str(reply["text"]).startswith("❓ Помощь") for reply in message.replies)
    assert not any(edit.startswith("❓ Помощь") for edit in message.edits)
    assert not any(str(reply["text"]).startswith("✨ Nova") for reply in message.replies)
    assert not any(edit.startswith("✨ Nova") for edit in message.edits)
    assert fake_ai.route_calls == [] or fake_ai.route_calls[-1][0] == phrase


async def test_greeting_gets_answer_and_is_not_saved(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Привет")
    await bot.text(update_for(message), SimpleNamespace(user_data={}))
    assert message.replies[-1]["text"] == "Привет!"
    assert await inbox_count(db) == 0
    assert await draft_count(db) == 0


async def test_tomorrow_question_uses_timezone_and_is_not_saved(db, fake_ai):
    router = IntentRouter(fake_ai, 0.70)
    moment = datetime(2026, 7, 12, 21, 30, tzinfo=UTC)
    moscow = await router.route("Какой завтра день недели?", "Europe/Moscow", now=moment)
    new_york = await router.route("Какой завтра день недели?", "America/New_York", now=moment)
    assert moscow.answer == "Завтра вторник."
    assert new_york.answer == "Завтра понедельник."
    assert await inbox_count(db) == 0


async def test_defer_answer_suppresses_only_internal_fallback(fake_ai):
    router = IntentRouter(fake_ai, 0.70)
    original_route = fake_ai.route_message

    async def route_without_answer(text, temporal_context, conversation_context=None):
        result = await original_route(text, temporal_context, conversation_context)
        return result.model_copy(update={"intent": "question", "answer": None})

    fake_ai.route_message = route_without_answer

    deferred = await router.route("Какой завтра день недели?", "Europe/Moscow", defer_answer=True)

    assert deferred.intent == "question"
    assert deferred.answer is None
    assert fake_ai.answer_calls == []
    assert fake_ai.answer_confirmed_memory_calls == []


async def test_legacy_route_still_calls_plain_answer_when_not_deferred(fake_ai):
    router = IntentRouter(fake_ai, 0.70)
    original_route = fake_ai.route_message

    async def route_without_answer(text, temporal_context, conversation_context=None):
        result = await original_route(text, temporal_context, conversation_context)
        return result.model_copy(update={"intent": "question", "answer": None})

    fake_ai.route_message = route_without_answer

    result = await router.route("Какой завтра день недели?", "Europe/Moscow")

    assert result.answer == "Ответ на: Какой завтра день недели?"
    assert len(fake_ai.answer_calls) == 1
    assert fake_ai.answer_confirmed_memory_calls == [None]


async def test_router_answer_forwards_only_nonempty_typed_memory(fake_ai):
    router = IntentRouter(fake_ai, 0.70)
    projection = build_nova_memory_projection(
        [
            SimpleNamespace(
                public_id="private-item",
                category="interaction",
                content="Отвечай кратко",
                important=True,
                updated_at=datetime(2026, 8, 13, tzinfo=UTC),
            )
        ],
        collection_revision="private-revision",
    )

    answer = await router.answer(
        "Объясни идею",
        "Europe/Moscow",
        confirmed_memory=projection,
    )

    assert answer.answer == "Ответ на: Объясни идею"
    assert fake_ai.answer_confirmed_memory_calls == [projection]


async def test_question_handler_creates_no_draft_or_inbox(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Какой завтра день недели?")
    await bot.text(update_for(message, 502), SimpleNamespace(user_data={}))
    assert str(message.replies[-1]["text"]).startswith("Завтра")
    assert await draft_count(db) == 0
    assert await inbox_count(db) == 0


async def test_idea_preview_requires_callback_and_saves_once(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    update = update_for(message)
    await bot.text(update, context)
    assert "Тип: идея" in str(message.replies[-1]["text"])
    assert await inbox_count(db) == 0

    callback_data = preview_callback(message, "save")
    query = FakeCallbackQuery(callback_data, message)
    callback_update = SimpleNamespace(
        callback_query=query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )
    await bot.inbox_action(callback_update, context)
    await bot.inbox_action(callback_update, context)
    assert await inbox_count(db) == 1
    assert query.answers[-1] == ("Эта карточка уже неактуальна. Создай новую.", True)


async def test_drop_preview_never_saves(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    message = FakeMessage("Не забудь сделать звонок")
    update = update_for(message)
    await bot.text(update, context)
    query = FakeCallbackQuery(preview_callback(message, "drop"), message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert await inbox_count(db) == 0


async def test_edit_repreviews_and_only_second_confirmation_saves(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    first_message = FakeMessage("Мне пришла идея создать пространство")
    update = update_for(first_message)
    await bot.text(update, context)
    first_edit_callback = preview_callback(first_message, "edit")
    first_draft_id = first_edit_callback.split(":")[2]
    edit_query = FakeCallbackQuery(first_edit_callback, first_message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=edit_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert await inbox_count(db) == 0

    corrected = "Мне пришла идея создать совместное пространство для друзей"
    corrected_message = FakeMessage(corrected)
    await bot.text(update_for(corrected_message), context)
    save_callback = preview_callback(corrected_message, "save")
    assert save_callback.split(":")[2] == first_draft_id
    assert save_callback.endswith(":2")
    assert "Тип: идея" in str(corrected_message.replies[-1]["text"])
    assert await inbox_count(db) == 0

    save_query = FakeCallbackQuery(save_callback, corrected_message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=save_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert await inbox_count(db) == 1
    async with db.sessions() as session:
        saved = await session.scalar(select(InboxItem))
    assert saved.raw_text == corrected


async def test_text_and_voice_share_intent_router(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    await bot.text(update_for(FakeMessage("Привет"), 601), SimpleNamespace(user_data={}))
    voice_message = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice_message, 601), SimpleNamespace(user_data={}))
    assert [call[0] for call in fake_ai.route_calls[-2:]] == ["Привет", "Привет"]
    assert voice_message.edits[0] == "Я услышал: «Привет»"
    assert voice_message.replies[-1]["text"] == "Привет!"


async def test_low_confidence_shows_choices_and_choice_still_previews(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    message = FakeMessage("непонятно")
    update = update_for(message, 701)
    await bot.text(update, context)
    markup = message.replies[-1]["reply_markup"]
    callback_data = {button.callback_data for row in markup.inline_keyboard for button in row}
    assert {
        next(value for value in callback_data if value.startswith("intent:answer:")),
        next(value for value in callback_data if value.startswith("intent:idea:")),
        next(value for value in callback_data if value.startswith("intent:task:")),
        next(value for value in callback_data if value.startswith("intent:note:")),
        next(value for value in callback_data if value.startswith("intent:drop:")),
    } == callback_data
    assert await inbox_count(db) == 0

    idea_callback = next(value for value in callback_data if value.startswith("intent:idea:"))
    query = FakeCallbackQuery(idea_callback, message)
    await bot.intent_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert any(
        button.callback_data.startswith("inbox:save:")
        for row in message.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    )
    assert await inbox_count(db) == 0


async def test_intent_answer_callback_flags_off_keeps_legacy_answer_contract_and_replay(
    db, fake_ai
):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage()
    update, context, _pending, query = intent_answer_callback(message, user_id=7101)

    async def forbidden_snapshot(**kwargs):
        del kwargs
        raise AssertionError("application snapshot must stay unread while flags are off")

    bot.nova_memory_service.application_snapshot = forbidden_snapshot

    await bot.intent_action(update, context)
    await bot.intent_action(update, context)

    assert fake_ai.answer_confirmed_memory_calls == [None]
    assert len(fake_ai.answer_calls) == 1
    assert query.answers == [(None, False), ("Это действие уже обработано", True)]
    assert query.edits == ["Ответ на: Как лучше спланировать неделю?"]
    assert message.replies == []


@pytest.mark.parametrize("mismatch", ["chat", "message"])
async def test_intent_answer_callback_wrong_canonical_is_stale_without_reads_or_dml(
    db,
    fake_ai,
    mismatch,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_120 if mismatch == "chat" else 7_121
    message = FakeMessage(message_id=91_200)
    update, context, pending, query = intent_answer_callback(message, user_id=user_id)
    if mismatch == "chat":
        message.chat = SimpleNamespace(id=message.chat.id + 1)
    else:
        message.message_id += 1

    async def forbidden_snapshot(**kwargs):
        del kwargs
        raise AssertionError("stale canonical must stop before memory snapshot")

    bot.nova_memory_service.application_snapshot = forbidden_snapshot
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)

    await bot.intent_action(update, context)

    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert query.answers == [("Это действие устарело", True)]
    assert query.edits == []
    assert pending.handled is False
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert audit_after == audit_before


@pytest.mark.parametrize("failure_stage", ["actor", "conversation"])
async def test_intent_answer_callback_context_read_failure_is_exact_and_memory_blind(
    db,
    fake_ai,
    caplog,
    failure_stage,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_129 if failure_stage == "actor" else 7_130
    private_memory = f"PRIVATE_CALLBACK_{failure_stage.upper()}_MEMORY"
    private_question = f"PRIVATE_CALLBACK_{failure_stage.upper()}_QUESTION"
    private_error = f"PRIVATE_CALLBACK_{failure_stage.upper()}_ERROR"
    await enable_admin_memory(bot, user_id=user_id, content=private_memory)
    message = FakeMessage(message_id=91_290)
    update, context, _pending, query = intent_answer_callback(
        message,
        user_id=user_id,
        raw_text=private_question,
    )

    class CallbackContextFailure(RuntimeError):
        pass

    async def failed_read(*args, **kwargs):
        del args, kwargs
        raise CallbackContextFailure(private_error)

    if failure_stage == "actor":
        bot._user = failed_read
    else:
        bot.conversation.get = failed_read
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)

    with caplog.at_level("WARNING"):
        await bot.intent_action(update, context)

    assert query.answers == [(None, False)]
    assert query.edits == [NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT]
    assert query.edit_kwargs == [{"reply_markup": None, "parse_mode": None}]
    assert message.replies == []
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert bot._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after == audit_before
    assert "stage=callback_context" in caplog.text
    assert "error_type=CallbackContextFailure" in caplog.text
    for private_value in (
        private_memory,
        private_question,
        private_error,
        str(user_id),
        str(message.message_id),
    ):
        assert private_value not in caplog.text


async def test_intent_answer_callbacks_on_distinct_canonicals_do_not_share_ui_lock(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    first_user_id = 7_122
    second_user_id = 7_123
    await enable_admin_memory(bot, user_id=first_user_id, content="Первый стиль ответа")
    await enable_admin_memory(bot, user_id=second_user_id, content="Второй стиль ответа")
    first_message = FakeMessage(message_id=91_220)
    second_message = FakeMessage(message_id=91_230)
    first_update, first_context, _, first_query = intent_answer_callback(
        first_message,
        user_id=first_user_id,
        token="first-canonical",
    )
    second_update, second_context, _, second_query = intent_answer_callback(
        second_message,
        user_id=second_user_id,
        token="second-canonical",
    )
    first_edit_started = asyncio.Event()
    first_edit_release = asyncio.Event()
    original_first_edit = first_query.edit_message_text

    async def blocked_first_edit(text, **kwargs):
        first_edit_started.set()
        await first_edit_release.wait()
        await original_first_edit(text, **kwargs)

    first_query.edit_message_text = blocked_first_edit
    first_task = asyncio.create_task(bot.intent_action(first_update, first_context))
    await asyncio.wait_for(first_edit_started.wait(), timeout=2)

    await asyncio.wait_for(bot.intent_action(second_update, second_context), timeout=2)

    assert not first_task.done()
    assert second_query.answers == [(None, False)]
    assert second_query.edits == ["Ответ на: Как лучше спланировать неделю?"]
    first_edit_release.set()
    await asyncio.wait_for(first_task, timeout=2)
    assert first_query.answers == [(None, False)]
    assert first_query.edits == ["Ответ на: Как лучше спланировать неделю?"]
    assert len(fake_ai.answer_calls) == 2


async def test_intent_answer_callback_applies_nonempty_memory_once_and_appends_after_fence(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7102
    await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage()
    update, context, _pending, query = intent_answer_callback(message, user_id=user_id)

    await bot.intent_action(update, context)
    await bot.intent_action(update, context)

    assert query.answers == [(None, False), ("Это действие уже обработано", True)]
    assert query.edits == ["Ответ на: Как лучше спланировать неделю?"]
    assert len(fake_ai.answer_calls) == 1
    assert len(fake_ai.answer_confirmed_memory_calls) == 1
    projection = fake_ai.answer_confirmed_memory_calls[0]
    assert projection is not None
    assert projection.provider_payload() == [
        {
            "category": "interaction",
            "important": True,
            "content": "Отвечай кратко и по пунктам",
        }
    ]
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert [entry["role"] for entry in conversation.messages] == ["assistant"]
    assert conversation.messages[0]["content"] == "Ответ на: Как лучше спланировать неделю?"


@pytest.mark.parametrize("mutation", ["create", "edit", "toggle", "delete", "delete_all"])
async def test_intent_answer_callback_revision_change_during_provider_neutralizes_without_editing_answer(
    db,
    fake_ai,
    mutation,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7103
    access = await enable_admin_memory(bot, user_id=user_id)
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()

    async def blocked_answer(
        text,
        temporal_context,
        conversation_context=None,
        *,
        confirmed_memory=None,
    ):
        del temporal_context, conversation_context, confirmed_memory
        fake_ai.answer_calls.append((text, {}))
        provider_started.set()
        await provider_release.wait()
        return AssistantAnswer(answer="Приватный персонализированный ответ")

    fake_ai.answer_message = blocked_answer
    message = FakeMessage()
    update, context, _pending, query = intent_answer_callback(message, user_id=user_id)
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)

    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await provider_started.wait()
    await mutate_stage7c_memory(
        bot,
        access,
        user_id,
        mutation,
        suffix=f"PROVIDER_{mutation.upper()}",
    )
    provider_release.set()
    await callback_task

    assert query.answers == [(None, False)]
    assert query.edits == [NOVA_MEMORY_APPLICATION_CHANGED_TEXT]
    assert query.edit_kwargs[-1] == {"reply_markup": None, "parse_mode": None}
    assert "Приватный персонализированный ответ" not in query.edits
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after - audit_before == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


@pytest.mark.parametrize("access_race", ["downgrade", "bounce"])
async def test_intent_answer_callback_access_change_during_provider_neutralizes_fail_closed(
    db,
    fake_ai,
    access_race,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7109
    await enable_admin_memory(bot, user_id=user_id)
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()

    async def blocked_answer(
        text,
        temporal_context,
        conversation_context=None,
        *,
        confirmed_memory=None,
    ):
        del temporal_context, conversation_context, confirmed_memory
        fake_ai.answer_calls.append((text, {}))
        provider_started.set()
        await provider_release.wait()
        return AssistantAnswer(answer="Приватный персонализированный ответ")

    fake_ai.answer_message = blocked_answer
    message = FakeMessage()
    update, context, _pending, query = intent_answer_callback(message, user_id=user_id)

    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await provider_started.wait()
    downgraded = await bot.access_service.set_guest(user_id, source="test")
    assert downgraded.changed is True
    if access_race == "bounce":
        restored = await bot.access_service.grant_admin(user_id, source="test")
        assert restored.changed is True
    provider_release.set()
    await callback_task

    assert query.answers == [(None, False)]
    assert query.edits == [NOVA_MEMORY_ACCESS_CHANGED_TEXT]
    assert query.edit_kwargs[-1] == {"reply_markup": None, "parse_mode": None}
    assert "Приватный персонализированный ответ" not in query.edits
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


@pytest.mark.parametrize("race", ["downgrade", "bounce", "create"])
async def test_intent_answer_callback_pre_edit_fence_blocks_access_and_revision_races(
    db,
    fake_ai,
    race,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7130
    access = await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage(message_id=91_300)
    update, context, _pending, query = intent_answer_callback(message, user_id=user_id)
    pre_edit_started = asyncio.Event()
    pre_edit_release = asyncio.Event()
    current_calls = 0
    original_current_check = bot.nova_memory_service.application_current_check

    async def blocked_pre_edit_check(**kwargs):
        nonlocal current_calls
        current_calls += 1
        if current_calls == 3:
            pre_edit_started.set()
            await pre_edit_release.wait()
        return await original_current_check(**kwargs)

    bot.nova_memory_service.application_current_check = blocked_pre_edit_check
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await asyncio.wait_for(pre_edit_started.wait(), timeout=2)

    if race == "downgrade":
        await bot.access_service.set_guest(user_id, source="test")
    elif race == "bounce":
        await bot.access_service.set_guest(user_id, source="test")
        await bot.access_service.grant_admin(user_id, source="test")
    else:
        await mutate_stage7c_memory(
            bot,
            access,
            user_id,
            race,
            suffix="PRE_EDIT_CREATE",
        )
    pre_edit_release.set()
    await asyncio.wait_for(callback_task, timeout=2)

    expected_neutral = (
        NOVA_MEMORY_ACCESS_CHANGED_TEXT
        if race in {"downgrade", "bounce"}
        else NOVA_MEMORY_APPLICATION_CHANGED_TEXT
    )
    assert current_calls == 3
    assert query.answers == [(None, False)]
    assert query.edits == [expected_neutral]
    assert query.edit_kwargs == [{"reply_markup": None, "parse_mode": None}]
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    assert bot._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after - audit_before == (0 if race in {"downgrade", "bounce"} else 1)
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


async def test_intent_answer_callback_message_not_modified_is_success_and_runs_post_fence(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7104
    await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage()
    update, context, _pending, query = intent_answer_callback(message, user_id=user_id)
    edit_attempts = 0

    async def not_modified(text, **kwargs):
        nonlocal edit_attempts
        del text, kwargs
        edit_attempts += 1
        raise BadRequest("Message is not modified")

    query.edit_message_text = not_modified

    await bot.intent_action(update, context)

    assert query.answers == [(None, False)]
    assert edit_attempts == 1
    assert len(fake_ai.answer_calls) == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert [entry["role"] for entry in conversation.messages] == ["assistant"]


@pytest.mark.parametrize(
    "race",
    ["downgrade", "bounce", "create", "edit", "toggle", "delete", "delete_all"],
)
async def test_intent_answer_callback_state_change_inside_primary_edit_neutralizes_canonical(
    db,
    fake_ai,
    race,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7105
    access = await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage()
    update, context, _pending, query = intent_answer_callback(message, user_id=user_id)
    primary_started = asyncio.Event()
    primary_release = asyncio.Event()
    original_edit = query.edit_message_text

    async def blocked_primary_edit(text, **kwargs):
        if text.startswith("Ответ на:"):
            primary_started.set()
            await primary_release.wait()
        await original_edit(text, **kwargs)

    query.edit_message_text = blocked_primary_edit
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)

    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await primary_started.wait()
    if race == "downgrade":
        await bot.access_service.set_guest(user_id, source="test")
    elif race == "bounce":
        await bot.access_service.set_guest(user_id, source="test")
        await bot.access_service.grant_admin(user_id, source="test")
    else:
        await mutate_stage7c_memory(
            bot,
            access,
            user_id,
            race,
            suffix=f"EDIT_{race.upper()}",
        )
    primary_release.set()
    await callback_task

    expected_neutral = (
        NOVA_MEMORY_ACCESS_CHANGED_TEXT
        if race in {"downgrade", "bounce"}
        else NOVA_MEMORY_APPLICATION_CHANGED_TEXT
    )
    assert query.answers == [(None, False)]
    assert query.edits == [
        "Ответ на: Как лучше спланировать неделю?",
        expected_neutral,
    ]
    assert query.edit_kwargs[-1] == {"reply_markup": None, "parse_mode": None}
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after - audit_before == (0 if race in {"downgrade", "bounce"} else 1)
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


async def test_intent_answer_callback_replacement_identity_is_preserved_and_not_overwritten(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7106
    await enable_admin_memory(bot, user_id=user_id)
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()

    async def blocked_answer(
        text,
        temporal_context,
        conversation_context=None,
        *,
        confirmed_memory=None,
    ):
        del temporal_context, conversation_context, confirmed_memory
        fake_ai.answer_calls.append((text, {}))
        provider_started.set()
        await provider_release.wait()
        return AssistantAnswer(answer="Старый приватный ответ")

    fake_ai.answer_message = blocked_answer
    message = FakeMessage()
    update, context, pending, query = intent_answer_callback(message, user_id=user_id)
    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await provider_started.wait()
    replacement = PendingIntent(
        token=pending.token,
        raw_text="Новый вопрос",
        source="text",
        result=IntentResult(intent="unknown", confidence=0.1),
    )
    context.user_data[f"intent:{pending.token}"] = replacement
    provider_release.set()
    await callback_task

    assert context.user_data[f"intent:{pending.token}"] is replacement
    assert replacement.handled is False
    assert query.answers == [(None, False)]
    assert query.edits == []
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


async def test_intent_answer_callback_replacement_during_pre_delivery_check_stops_old_paint(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_124
    await enable_admin_memory(bot, user_id=user_id)
    pre_delivery_started = asyncio.Event()
    pre_delivery_release = asyncio.Event()
    current_calls = 0
    original_current_check = bot.nova_memory_service.application_current_check

    async def blocked_pre_delivery_check(**kwargs):
        nonlocal current_calls
        current_calls += 1
        if current_calls == 3:
            pre_delivery_started.set()
            await pre_delivery_release.wait()
        return await original_current_check(**kwargs)

    bot.nova_memory_service.application_current_check = blocked_pre_delivery_check
    message = FakeMessage(message_id=91_240)
    update, context, pending, query = intent_answer_callback(message, user_id=user_id)
    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await asyncio.wait_for(pre_delivery_started.wait(), timeout=2)
    replacement = PendingIntent(
        token=pending.token,
        raw_text="Новый вопрос того же canonical",
        source="text",
        result=IntentResult(intent="unknown", confidence=0.1),
        canonical_chat_id=pending.canonical_chat_id,
        canonical_message_id=pending.canonical_message_id,
    )
    context.user_data[f"intent:{pending.token}"] = replacement

    pre_delivery_release.set()
    await asyncio.wait_for(callback_task, timeout=2)

    assert current_calls == 3
    assert context.user_data[f"intent:{pending.token}"] is replacement
    assert replacement.handled is False
    assert query.answers == [(None, False)]
    assert query.edits == []
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


async def test_intent_answer_callback_replacement_fresh_render_wins_early_neutral_race(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_128
    await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage(message_id=91_280)
    update, context, pending, old_query = intent_answer_callback(message, user_id=user_id)
    old_neutral_started = asyncio.Event()
    old_neutral_release = asyncio.Event()
    prepare_calls = 0
    neutral_calls = 0
    original_prepare = bot._prepare_nova_memory_answer
    original_neutral_if_current = bot._edit_nova_memory_application_neutral_if_current

    async def fail_only_old_preparation(**kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 1:
            return "unavailable"
        return await original_prepare(**kwargs)

    async def block_only_old_neutral(*args, **kwargs):
        nonlocal neutral_calls
        neutral_calls += 1
        if neutral_calls == 1:
            old_neutral_started.set()
            await old_neutral_release.wait()
        return await original_neutral_if_current(*args, **kwargs)

    bot._prepare_nova_memory_answer = fail_only_old_preparation
    bot._edit_nova_memory_application_neutral_if_current = block_only_old_neutral
    old_task = asyncio.create_task(bot.intent_action(update, context))
    await asyncio.wait_for(old_neutral_started.wait(), timeout=2)

    replacement = PendingIntent(
        token=pending.token,
        raw_text="Новый актуальный вопрос",
        source="text",
        result=IntentResult(intent="unknown", confidence=0.1),
        canonical_chat_id=pending.canonical_chat_id,
        canonical_message_id=pending.canonical_message_id,
    )
    pending_key = f"intent:{pending.token}"
    context.user_data[pending_key] = replacement
    replacement_query = FakeCallbackQuery(f"intent:answer:{pending.token}", message)
    replacement_update = SimpleNamespace(
        callback_query=replacement_query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )

    await asyncio.wait_for(bot.intent_action(replacement_update, context), timeout=2)
    assert replacement_query.edits == ["Ответ на: Новый актуальный вопрос"]

    old_neutral_release.set()
    await asyncio.wait_for(old_task, timeout=2)

    assert prepare_calls == 2
    assert neutral_calls == 1
    assert context.user_data[pending_key] is replacement
    assert replacement.handled is True
    assert replacement.canonical_chat_id == pending.canonical_chat_id
    assert replacement.canonical_message_id == pending.canonical_message_id
    assert old_query.answers == [(None, False)]
    assert replacement_query.answers == [(None, False)]
    assert old_query.edits == []
    assert replacement_query.edits[-1] == "Ответ на: Новый актуальный вопрос"
    assert NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT not in replacement_query.edits
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    assert bot._nova_memory_application_tasks == set()
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert [entry["content"] for entry in conversation.messages] == [
        "Ответ на: Новый актуальный вопрос"
    ]


async def test_intent_answer_callback_replacement_during_primary_edit_is_neutralized_unchanged(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_127
    await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage(message_id=91_270)
    update, context, pending, query = intent_answer_callback(message, user_id=user_id)
    primary_started = asyncio.Event()
    primary_release = asyncio.Event()
    original_edit = query.edit_message_text

    async def blocked_primary_edit(text, **kwargs):
        if text != NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT:
            primary_started.set()
            await primary_release.wait()
        await original_edit(text, **kwargs)

    query.edit_message_text = blocked_primary_edit
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await asyncio.wait_for(primary_started.wait(), timeout=2)
    replacement = PendingIntent(
        token=pending.token,
        raw_text="Новый вопрос того же canonical после primary edit",
        source="text",
        result=IntentResult(intent="unknown", confidence=0.1),
        canonical_chat_id=pending.canonical_chat_id,
        canonical_message_id=pending.canonical_message_id,
    )
    pending_key = f"intent:{pending.token}"
    context.user_data[pending_key] = replacement

    primary_release.set()
    await asyncio.wait_for(callback_task, timeout=2)

    assert context.user_data[pending_key] is replacement
    assert replacement.handled is False
    assert replacement.raw_text == "Новый вопрос того же canonical после primary edit"
    assert replacement.canonical_chat_id == pending.canonical_chat_id
    assert replacement.canonical_message_id == pending.canonical_message_id
    assert query.answers == [(None, False)]
    assert query.edits == [
        "Ответ на: Как лучше спланировать неделю?",
        NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
    ]
    assert query.edit_kwargs[-1] == {"reply_markup": None, "parse_mode": None}
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    assert bot._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after == audit_before
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


@pytest.mark.parametrize("replacement_outcome", ["failure", "cancellation"])
async def test_intent_answer_callback_replacement_renderer_failure_or_cancel_leaves_neutral_last(
    db,
    fake_ai,
    replacement_outcome,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_125 if replacement_outcome == "failure" else 7_126
    await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage(message_id=91_250)
    update, context, pending, old_query = intent_answer_callback(message, user_id=user_id)
    old_primary_started = asyncio.Event()
    old_primary_release = asyncio.Event()
    replacement_waiting_for_canonical = asyncio.Event()
    ui_lock_requests = 0
    edit_order: list[tuple[str, str, dict[str, object]]] = []
    old_edit = old_query.edit_message_text
    original_ui_lock = bot._nova_memory_application_ui_lock

    def observed_ui_lock(binding):
        nonlocal ui_lock_requests
        ui_lock_requests += 1
        lock = original_ui_lock(binding)
        if ui_lock_requests == 2:
            replacement_waiting_for_canonical.set()
        return lock

    async def blocked_old_edit(text, **kwargs):
        if text != NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT:
            old_primary_started.set()
            await old_primary_release.wait()
        edit_order.append(("old", text, kwargs))
        await old_edit(text, **kwargs)

    old_query.edit_message_text = blocked_old_edit
    bot._nova_memory_application_ui_lock = observed_ui_lock
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)

    old_task = asyncio.create_task(bot.intent_action(update, context))
    await asyncio.wait_for(old_primary_started.wait(), timeout=2)
    replacement = PendingIntent(
        token=pending.token,
        raw_text="Новый вопрос для replacement renderer",
        source="text",
        result=IntentResult(intent="unknown", confidence=0.1),
        canonical_chat_id=pending.canonical_chat_id,
        canonical_message_id=pending.canonical_message_id,
    )
    pending_key = f"intent:{pending.token}"
    context.user_data[pending_key] = replacement
    replacement_query = FakeCallbackQuery(f"intent:answer:{pending.token}", message)
    replacement_update = SimpleNamespace(
        callback_query=replacement_query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )
    replacement_edit_attempted = asyncio.Event()

    async def failed_or_cancelled_replacement_edit(text, **kwargs):
        edit_order.append(("replacement_attempt", text, kwargs))
        replacement_edit_attempted.set()
        if replacement_outcome == "cancellation":
            raise asyncio.CancelledError
        raise BadRequest("replacement renderer failed")

    replacement_query.edit_message_text = failed_or_cancelled_replacement_edit
    replacement_task = asyncio.create_task(bot.intent_action(replacement_update, context))
    await asyncio.wait_for(replacement_waiting_for_canonical.wait(), timeout=2)
    assert len(fake_ai.answer_calls) == 2
    assert not replacement_edit_attempted.is_set()

    old_primary_release.set()
    await asyncio.wait_for(old_task, timeout=2)
    if replacement_outcome == "cancellation":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(replacement_task, timeout=2)
    else:
        await asyncio.wait_for(replacement_task, timeout=2)

    assert replacement_edit_attempted.is_set()
    assert context.user_data[pending_key] is replacement
    assert replacement.handled is True
    assert replacement.canonical_chat_id == pending.canonical_chat_id
    assert replacement.canonical_message_id == pending.canonical_message_id
    assert old_query.answers == [(None, False)]
    assert replacement_query.answers == [(None, False)]
    assert [entry[0] for entry in edit_order] == [
        "old",
        "old",
        "replacement_attempt",
    ]
    assert [entry[1] for entry in edit_order[:2]] == [
        "Ответ на: Как лучше спланировать неделю?",
        NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
    ]
    assert edit_order[1][2] == {"reply_markup": None, "parse_mode": None}
    assert old_query.edits[-1] == NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT
    assert replacement_query.edits == []
    assert message.replies == []
    assert bot._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after == audit_before
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


async def test_intent_answer_callback_external_cancel_after_edit_keeps_inner_post_fence_alive(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7107
    access = await enable_admin_memory(bot, user_id=user_id)
    post_check_started = asyncio.Event()
    post_check_release = asyncio.Event()
    current_calls = 0
    original_current_check = bot.nova_memory_service.application_current_check

    async def blocked_post_check(**kwargs):
        nonlocal current_calls
        current_calls += 1
        if current_calls == 4:
            post_check_started.set()
            await post_check_release.wait()
        return await original_current_check(**kwargs)

    bot.nova_memory_service.application_current_check = blocked_post_check
    message = FakeMessage()
    update, context, _pending, query = intent_answer_callback(message, user_id=user_id)
    callback_task = asyncio.create_task(bot.intent_action(update, context))
    await post_check_started.wait()
    assert query.edits == ["Ответ на: Как лучше спланировать неделю?"]

    callback_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await callback_task
    changed = await bot.nova_memory_service.create(
        telegram_actor_id=user_id,
        expected_access_version=access.access_version,
        category="orientation",
        content="После отправки поменялся приоритет",
    )
    assert changed.status == "created"
    inner_tasks = tuple(bot._nova_memory_application_tasks)
    assert len(inner_tasks) == 1
    assert inner_tasks[0].get_name() == "nova-memory-application-edit-lifecycle"
    post_check_release.set()
    await asyncio.gather(*inner_tasks)

    assert query.answers == [(None, False)]
    assert query.edits[-1] == NOVA_MEMORY_APPLICATION_CHANGED_TEXT
    assert query.edit_kwargs[-1] == {"reply_markup": None, "parse_mode": None}
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []


async def test_intent_answer_callback_compensating_edit_cancellation_is_observed_safely(
    db,
    fake_ai,
    caplog,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7127
    private_memory = "PRIVATE_STAGE7C_CALLBACK_CLEANUP_MEMORY"
    private_question = "PRIVATE_STAGE7C_CALLBACK_CLEANUP_QUESTION"
    private_error = "PRIVATE_STAGE7C_CALLBACK_CLEANUP_CANCELLATION"
    access = await enable_admin_memory(
        bot,
        user_id=user_id,
        content=private_memory,
    )
    message = FakeMessage(message_id=91_270)
    update, context, _pending, query = intent_answer_callback(
        message,
        user_id=user_id,
        raw_text=private_question,
    )
    post_check_started = asyncio.Event()
    post_check_release = asyncio.Event()
    current_calls = 0
    original_current_check = bot.nova_memory_service.application_current_check

    async def blocked_post_check(**kwargs):
        nonlocal current_calls
        current_calls += 1
        if current_calls == 4:
            post_check_started.set()
            await post_check_release.wait()
        return await original_current_check(**kwargs)

    bot.nova_memory_service.application_current_check = blocked_post_check
    edit_attempts: list[tuple[str, dict[str, object]]] = []
    original_edit = query.edit_message_text

    async def cancel_compensating_edit(text, **kwargs):
        edit_attempts.append((text, kwargs))
        if len(edit_attempts) == 1:
            await original_edit(text, **kwargs)
            return
        raise asyncio.CancelledError(private_error)

    query.edit_message_text = cancel_compensating_edit
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)

    loop = asyncio.get_running_loop()
    prior_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, error_context: loop_errors.append(error_context))
    callback_task = None
    inner_task = None
    try:
        with caplog.at_level("WARNING"):
            callback_task = asyncio.create_task(bot.intent_action(update, context))
            await asyncio.wait_for(post_check_started.wait(), timeout=2)
            assert len(query.edits) == 1
            assert private_question in query.edits[0]
            inner_tasks = tuple(bot._nova_memory_application_tasks)
            assert len(inner_tasks) == 1
            inner_task = inner_tasks[0]
            assert inner_task.get_name() == "nova-memory-application-edit-lifecycle"

            await mutate_stage7c_memory(
                bot,
                access,
                user_id,
                "create",
                suffix="CLEANUP_CANCELLATION",
            )
            post_check_release.set()
            await asyncio.wait_for(callback_task, timeout=2)
            await asyncio.sleep(0)
    finally:
        post_check_release.set()
        if callback_task is not None and not callback_task.done():
            callback_task.cancel()
            await asyncio.gather(callback_task, return_exceptions=True)
        loop.set_exception_handler(prior_exception_handler)

    assert current_calls == 4
    assert inner_task is not None
    assert inner_task.done()
    assert not inner_task.cancelled()
    assert inner_task.result() is False
    assert bot._nova_memory_application_tasks == set()
    assert query.answers == [(None, False)]
    assert len(edit_attempts) == 2
    assert edit_attempts[1] == (
        NOVA_MEMORY_APPLICATION_CHANGED_TEXT,
        {"reply_markup": None, "parse_mode": None},
    )
    assert query.edits == [edit_attempts[0][0]]
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after - audit_before == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []
    assert loop_errors == []
    assert "operation=intent_answer_post_edit" in caplog.text
    assert "error_type=CancelledError" in caplog.text
    assert "Task exception was never retrieved" not in caplog.text
    for private_value in (
        private_memory,
        private_question,
        private_error,
        str(user_id),
    ):
        assert private_value not in caplog.text


async def test_intent_answer_callback_direct_telegram_cancellation_starts_no_post_fence(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7108
    await enable_admin_memory(bot, user_id=user_id)
    message = FakeMessage()
    update, context, pending, query = intent_answer_callback(message, user_id=user_id)
    current_calls = 0
    original_current = bot.nova_memory_service.application_current_check

    async def counted_current(**kwargs):
        nonlocal current_calls
        current_calls += 1
        return await original_current(**kwargs)

    bot.nova_memory_service.application_current_check = counted_current

    async def cancelled_edit(text, **kwargs):
        del text, kwargs
        raise asyncio.CancelledError

    query.edit_message_text = cancelled_edit

    with pytest.raises(asyncio.CancelledError):
        await bot.intent_action(update, context)

    assert query.answers == [(None, False)]
    assert current_calls == 3
    assert bot._nova_memory_application_tasks == set()
    assert message.replies == []
    assert len(fake_ai.answer_calls) == 1
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    assert conversation.messages == []
    binding = bot._nova_memory_callback_binding(update, query, pending)
    assert binding is not None
    ui_lock = bot._nova_memory_application_ui_lock(binding)
    await asyncio.wait_for(ui_lock.acquire(), timeout=2)
    ui_lock.release()


async def test_intent_answer_callback_memory_answer_is_filtered_from_next_route_and_answer(
    db,
    fake_ai,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_128
    private_memory = "PRIVATE_STAGE7C_CALLBACK_CROSS_TURN_MEMORY"
    memory_answer = "PRIVATE_STAGE7C_CALLBACK_CROSS_TURN_ANSWER"
    await enable_admin_memory(bot, user_id=user_id, content=private_memory)
    callback_message = FakeMessage(message_id=91_280)
    update, context, _pending, query = intent_answer_callback(
        callback_message,
        user_id=user_id,
        raw_text="first callback question",
    )
    original_answer = fake_ai.answer_message
    answer_number = 0

    async def sentinel_answer(*args, **kwargs):
        nonlocal answer_number
        generated = await original_answer(*args, **kwargs)
        answer_number += 1
        answer = memory_answer if answer_number == 1 else "SECOND_CALLBACK_CROSS_TURN_ANSWER"
        return generated.model_copy(update={"answer": answer})

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="conversation", confidence=0.99, answer=None)

    fake_ai.answer_message = sentinel_answer
    fake_ai.route_message = conversation_route

    await bot.intent_action(update, context)
    await bot.intent_action(update, context)

    assert query.answers == [(None, False), ("Это действие уже обработано", True)]
    assert query.edits == [memory_answer]
    assert len(fake_ai.answer_calls) == 1
    first_snapshot = await bot.conversation.get(user_id, user_id + 10_000)
    assert any(
        message["content"] == memory_answer and message["intent"] == "memory_answer"
        for message in first_snapshot.messages
    )

    next_message = FakeMessage(
        "second callback cross turn question",
        chat_id=user_id + 10_000,
        message_id=91_281,
    )
    await bot._route_message(
        update_for(next_message, user_id),
        SimpleNamespace(user_data={}),
        next_message.text,
        "text",
    )

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 2
    assert len(fake_ai.answer_confirmed_memory_calls) == 2
    assert all(projection is not None for projection in fake_ai.answer_confirmed_memory_calls)
    for provider_context in (
        fake_ai.conversation_contexts[-1],
        fake_ai.answer_conversation_contexts[-1],
    ):
        assert memory_answer not in repr(provider_context)
        assert all(
            message.get("intent") != "memory_answer"
            for message in provider_context["recent_messages"]
        )
    assert next_message.replies == [{"text": "SECOND_CALLBACK_CROSS_TURN_ANSWER"}]
    final_snapshot = await bot.conversation.get(user_id, user_id + 10_000)
    assert any(message["content"] == memory_answer for message in final_snapshot.messages)
    assert private_memory not in repr(fake_ai.conversation_contexts)
    assert private_memory not in repr(fake_ai.answer_conversation_contexts)


@pytest.mark.parametrize("post_edit_state", ["success", "replacement"])
async def test_intent_answer_callback_outer_cancel_during_primary_edit_keeps_combined_lifecycle(
    db,
    fake_ai,
    post_edit_state,
):
    bot = FutureSelfBot(
        settings(
            enable_nova_memory=True,
            enable_nova_memory_application=True,
        ),
        db,
        fake_ai,
        GreetingTranscription(),
    )
    user_id = 7_129 if post_edit_state == "success" else 7_130
    private_memory = f"PRIVATE_STAGE7C_COMBINED_EDIT_{post_edit_state.upper()}"
    await enable_admin_memory(bot, user_id=user_id, content=private_memory)
    message = FakeMessage(message_id=91_290)
    update, context, pending, query = intent_answer_callback(message, user_id=user_id)
    primary_started = asyncio.Event()
    primary_release = asyncio.Event()
    original_edit = query.edit_message_text

    async def blocked_primary_edit(text, **kwargs):
        if not query.edits:
            primary_started.set()
            await primary_release.wait()
        await original_edit(text, **kwargs)

    query.edit_message_text = blocked_primary_edit
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)

    outer = asyncio.create_task(bot.intent_action(update, context))
    await asyncio.wait_for(primary_started.wait(), timeout=2)
    inner_tasks = tuple(bot._nova_memory_application_tasks)
    assert len(inner_tasks) == 1
    inner = inner_tasks[0]
    assert inner.get_name() == "nova-memory-application-edit-lifecycle"
    assert private_memory not in inner.get_name()
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert not inner.done()

    replacement = None
    pending_key = f"intent:{pending.token}"
    if post_edit_state == "replacement":
        replacement = PendingIntent(
            token=pending.token,
            raw_text="replacement callback question",
            source="text",
            result=IntentResult(intent="unknown", confidence=0.1),
            canonical_chat_id=pending.canonical_chat_id,
            canonical_message_id=pending.canonical_message_id,
        )
        context.user_data[pending_key] = replacement
    primary_release.set()
    assert await asyncio.wait_for(inner, timeout=2) is (post_edit_state == "success")
    await asyncio.sleep(0)

    assert query.answers == [(None, False)]
    assert len(fake_ai.answer_calls) == 1
    assert bot._nova_memory_application_tasks == set()
    if post_edit_state == "success":
        assert query.edits == ["Ответ на: Как лучше спланировать неделю?"]
        assert context.user_data[pending_key] is pending
    else:
        assert query.edits == [
            "Ответ на: Как лучше спланировать неделю?",
            NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
        ]
        assert query.edit_kwargs[-1] == {"reply_markup": None, "parse_mode": None}
        assert context.user_data[pending_key] is replacement
        assert replacement is not None
        assert replacement.handled is False
        assert replacement.raw_text == "replacement callback question"
    conversation = await bot.conversation.get(user_id, user_id + 10_000)
    if post_edit_state == "success":
        assert len(conversation.messages) == 1
        assert conversation.messages[0]["intent"] == "memory_answer"
    else:
        assert conversation.messages == []
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after == audit_before


async def test_voice_idea_creates_persistent_preview_without_inbox(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    message = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(message, 801), SimpleNamespace(user_data={}))
    assert message.edits[0].startswith("Я услышал:")
    assert preview_callback(message, "save").startswith("inbox:save:")
    assert await draft_count(db) == 1
    assert await inbox_count(db) == 0


async def test_persistent_draft_can_be_saved_after_bot_restart(db, fake_ai):
    first_bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    update = update_for(message, 811)
    await first_bot.text(update, SimpleNamespace(user_data={}))
    callback_data = preview_callback(message, "save")

    restarted_bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    query = FakeCallbackQuery(callback_data, message)
    await restarted_bot.inbox_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    assert await inbox_count(db) == 1


async def test_foreign_user_cannot_confirm_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    owner_update = update_for(message, 821)
    await bot.text(owner_update, SimpleNamespace(user_data={}))
    query = FakeCallbackQuery(preview_callback(message, "save"), message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=SimpleNamespace(id=999),
            effective_chat=owner_update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    assert await inbox_count(db) == 0
    assert query.answers[-1][1] is True


async def test_two_fast_messages_create_independent_drafts(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    await bot.text(
        update_for(FakeMessage("Мне пришла идея создать пространство А"), 831),
        SimpleNamespace(user_data={}),
    )
    await bot.text(
        update_for(FakeMessage("Мне пришла идея создать пространство Б"), 831),
        SimpleNamespace(user_data={}),
    )
    assert await draft_count(db) == 2
    assert await inbox_count(db) == 0


async def test_concurrent_save_callbacks_create_one_inbox_item(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    update = update_for(message, 841)
    await bot.text(update, SimpleNamespace(user_data={}))
    callback_data = preview_callback(message, "save")
    first_query = FakeCallbackQuery(callback_data, message)
    second_query = FakeCallbackQuery(callback_data, message)

    def callback_update(query):
        return SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        )

    await asyncio.gather(
        bot.inbox_action(callback_update(first_query), SimpleNamespace(user_data={})),
        bot.inbox_action(callback_update(second_query), SimpleNamespace(user_data={})),
    )
    assert await inbox_count(db) == 1
    assert sorted(len(query.edits) for query in (first_query, second_query)) == [0, 1]


async def test_voice_can_revise_editing_draft_without_saving(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    first_message = FakeMessage("Мне пришла идея создать пространство")
    update = update_for(first_message, 851)
    await bot.text(update, SimpleNamespace(user_data={}))
    edit_query = FakeCallbackQuery(preview_callback(first_message, "edit"), first_message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=edit_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    bot.transcription = CorrectedTranscription()
    voice_message = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice_message, 851), SimpleNamespace(user_data={}))
    assert preview_callback(voice_message, "save").endswith(":2")
    assert await inbox_count(db) == 0


async def test_cancel_discards_persistent_editing_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Мне пришла идея создать пространство")
    update = update_for(message, 861)
    await bot.text(update, SimpleNamespace(user_data={}))
    edit_query = FakeCallbackQuery(preview_callback(message, "edit"), message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=edit_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    cancel_message = FakeMessage()
    await bot.cancel_draft_edit(update_for(cancel_message, 861), SimpleNamespace(user_data={}))
    assert "ничего не сохранено" in str(cancel_message.replies[-1]["text"])
    assert await inbox_count(db) == 0


async def test_local_voice_edit_save_duplicate_flow_counts(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    context = SimpleNamespace(user_data={})
    voice_message = FakeMessage(voice=FakeVoice())
    update = update_for(voice_message, 871)
    await bot.voice(update, context)
    counts = [await inbox_count(db)]

    edit_query = FakeCallbackQuery(preview_callback(voice_message, "edit"), voice_message)
    callback_update = SimpleNamespace(
        callback_query=edit_query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )
    await bot.inbox_action(callback_update, context)
    counts.append(await inbox_count(db))

    corrected_message = FakeMessage("нужно заниматься спортом 3 раза в неделю")
    await bot.text(update_for(corrected_message, 871), context)
    counts.append(await inbox_count(db))

    save_query = FakeCallbackQuery(preview_callback(corrected_message, "save"), corrected_message)
    save_update = SimpleNamespace(
        callback_query=save_query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )
    await bot.inbox_action(save_update, context)
    counts.append(await inbox_count(db))
    await bot.inbox_action(save_update, context)
    counts.append(await inbox_count(db))

    print("handler_flow_counts=" + " -> ".join(map(str, counts)))
    assert counts == [0, 0, 0, 1, 1]
