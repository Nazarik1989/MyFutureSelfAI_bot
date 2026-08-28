import asyncio
import gc
import json
import logging
import warnings
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError
from sqlalchemy import event, func, select, text, update
from telegram import CallbackQuery, Chat, Message, MessageEntity, ReplyKeyboardRemove, Update
from telegram import User as TelegramUser
from telegram.error import TelegramError
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ExtBot,
    MessageHandler,
    TypeHandler,
)

import future_self.bot as bot_module
import future_self.main as main_module
from future_self.access import FULL_ACCESS_TIERS, AccessService
from future_self.ai import NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT
from future_self.bot import (
    NOVA_MEMORY_APPLICATION_CHANGED_TEXT,
    NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
    FutureSelfBot,
    log_safe_failure,
)
from future_self.config import Settings
from future_self.doctor import run_diagnostics
from future_self.main import create_application, format_configuration_error, run
from future_self.models import (
    ConversationMessage,
    DraftInboxItem,
    Goal,
    InboxItem,
    LifeCollection,
    NovaDialogueState,
    NovaMemoryChange,
    NovaMemoryItem,
    NovaObservedMemory,
    OnboardingState,
    RecurringTaskReminderSchedule,
    TaskReminder,
    User,
    VisionItem,
    VisionProfile,
    WeeklyFocus,
    WeeklyFocusChange,
    WeeklyReviewSession,
)
from future_self.nova_brain import NovaBrainPolicy, NovaBrainService
from future_self.nova_companion_handlers import (
    NOVA_COMPANION_NOT_EXECUTED_TEXT,
    NOVA_COMPANION_RECALL_ACTION_SUPPRESSED_TEXT,
    NOVA_COMPANION_RECALL_CLARIFICATION_TEXT,
    NOVA_COMPANION_RECALL_UNAVAILABLE_TEXT,
    NOVA_COMPANION_UNAVAILABLE_TEXT,
)
from future_self.nova_handlers import NOVA_ROOT_TEXT
from future_self.nova_memory import NovaMemoryApplicationCurrent
from future_self.nova_memory_flow import NovaMemoryFlowPhase
from future_self.reminder_flow import ReminderFlowPhase
from future_self.reminder_intent import ReminderScheduleKind, ReminderTimezoneSource
from future_self.repositories import OnboardingRepository, UserRepository
from future_self.schemas import (
    IntentResult,
    NovaCompanionDialogueStateUpdate,
    NovaCompanionMemoryCandidate,
    NovaCompanionProviderCapture,
    NovaCompanionProviderReminderOffer,
    NovaCompanionProviderResponse,
    ParsedThought,
    ReminderTimezoneResolution,
    WeeklyReviewExtraction,
)
from future_self.system_actions import SystemActionRouter
from future_self.tasks import add_task_state
from future_self.timezones import extract_reminder_timezone_fragment
from future_self.weekly_review import WeeklyReviewPhase
from future_self.weekly_review_handlers import (
    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
    WEEKLY_REVIEW_QUESTION,
)


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_stage8b21_course_reminder_transcript(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    from future_self.schemas import (
        NovaCompanionProviderReminderOffer,
        NovaCompanionProviderResponse,
    )

    telegram_id = 720_210 if source == "text" else 720_211
    phrases = (
        "мне нужно, чтобы ты был моим помощником по жизни, помогал не забыть о делах, и всяком разном.",
        (
            "да, но в том числе мне нужно напоминать о самом главном, чтобы в суете я не "
            "забывал курс, по которому я могу стать лучше"
        ),
        "Информации вокруг так много, что главное легко теряется в этом потоке.",
        ("вот !!! поэтому мне нужно вспоминать об этом почаще, а ты могла бы мне помогать в этом"),
    )
    transcription = RuntimeTranscription(phrases[0])
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    markup_edits: list[dict[str, object]] = []

    async def edit_reply_markup(self, *args, **kwargs):
        del self, args
        markup_edits.append(dict(kwargs))
        return transport.sent_messages[-1]

    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_reply_markup)
    responses = (
        "Поняла. Могу быть рядом как спокойный помощник по делам и важным мелочам.",
        "Ты хочешь не терять свой курс среди суеты — это важный ориентир.",
        "Понимаю: поток информации действительно может заслонять главное.",
        (
            "Ты хочешь регулярно возвращаться к своему курсу. Давай настроим настоящее "
            "напоминание и отдельно выберем расписание."
        ),
    )
    for index, phrase in enumerate(phrases):
        fake_ai.companion_provider_result = NovaCompanionProviderResponse(
            answer=responses[index],
            reminder_offer=(
                NovaCompanionProviderReminderOffer(
                    title="курс, по которому я могу стать лучше",
                    schedule_wording=None,
                    evidence=phrases[1],
                )
                if index == 3
                else None
            ),
        )
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=52_100 + index,
                source_message_id=192_100 + index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=52_100 + index,
                source_message_id=192_100 + index,
                progress_message_id=193_100 + index,
            )
        await application.process_update(update_value)
        await asyncio.sleep(0)

    route = SystemActionRouter().route(phrases[1], pending_action=None)
    assert (route.kind, route.action) == ("none", None)
    snapshot = await core.conversation.get(telegram_id, telegram_id)
    assert snapshot.system_pending_action is None
    assert len(fake_ai.companion_calls) == 4
    assert [call[0] for call in fake_ai.companion_calls] == list(phrases)
    assert len(markup_edits) == 1
    callbacks = _runtime_callback_values(markup_edits[0]["reply_markup"])
    assert len(callbacks) == 2
    assert all(callback.startswith("nrem:") for callback in callbacks)
    live = await core.nova_companion_reminders.active(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
        access_tier=owner.access_tier,
        access_version=owner.access_version,
    )
    assert live is not None
    assert live.candidate.schedule_wording is None
    assert live.candidate.temporal is None
    assert live.candidate.evidence == phrases[1]
    rendered_text = "\n".join(
        str(entry.get("text", "")) for entry in [*transport.sent, *transport.edits]
    )
    assert "команду удаления" not in rendered_text
    assert "ничего не удалено" not in rendered_text
    assert "курс" in rendered_text
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_stage8b21_single_offer_continuation_transcript(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    from future_self.schemas import (
        NovaCompanionProviderReminderOffer,
        NovaCompanionProviderResponse,
    )

    telegram_id = 720_220 if source == "text" else 720_221
    phrases = (
        "как не улетать в мысли постоянно? нужны какие-то ментальные тренировки?",
        (
            "а если я попрошу тебя постоянно мне напоминать, на первое время, пока я "
            "не привыкну. это норм вариант? как считаешь?"
        ),
        "ооо, было бы круто)",
        "ну давай подбери)",
    )
    transcription = RuntimeTranscription(phrases[0])
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    markup_edits: list[dict[str, object]] = []

    async def edit_reply_markup(self, *args, **kwargs):
        del self, args
        markup_edits.append(dict(kwargs))
        return transport.sent_messages[-1]

    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_reply_markup)
    raw_meta = "Только уточни: что именно подобрать — вариант, план или упражнение?"
    provider_results = (
        NovaCompanionProviderResponse(
            answer="Можно тренировать короткое возвращение внимания к текущему моменту."
        ),
        NovaCompanionProviderResponse(
            answer=(
                "Да, это может помочь. Настоящее расписание выберем отдельно. "
                "Если хочешь, можем дальше просто подобрать удобный способ."
            ),
            reminder_offer=NovaCompanionProviderReminderOffer(
                title="ментальные тренировки",
                schedule_wording=None,
                evidence=phrases[0],
            ),
        ),
        NovaCompanionProviderResponse(answer=raw_meta),
        NovaCompanionProviderResponse(
            answer=(
                "Выбери короткую остановку несколько раз в день и спрашивай себя, что "
                "сейчас действительно важно. Настоящее напоминание настроим отдельно."
            )
        ),
    )
    for index, phrase in enumerate(phrases):
        fake_ai.companion_provider_result = provider_results[index]
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=52_200 + index,
                source_message_id=192_200 + index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=52_200 + index,
                source_message_id=192_200 + index,
                progress_message_id=193_200 + index,
            )
        await application.process_update(update_value)
        await asyncio.sleep(0)

    assert len(fake_ai.companion_calls) == 4
    first_anchor = fake_ai.companion_discourse_calls[2]
    second_anchor = fake_ai.companion_discourse_calls[3]
    assert first_anchor is not None and first_anchor.offer_kinds == ("method",)
    assert second_anchor is not None and second_anchor.offer_kinds == ("method",)
    rendered_text = "\n".join(
        str(entry.get("text", "")) for entry in [*transport.sent, *transport.edits]
    )
    assert raw_meta not in rendered_text
    assert "короткую практику" in rendered_text
    assert "что именно подобрать" not in rendered_text.casefold()
    assert len(markup_edits) == 1
    callbacks = _runtime_callback_values(markup_edits[0]["reply_markup"])
    assert len(callbacks) == 2
    assert all(callback.startswith("nrem:") for callback in callbacks)
    live = await core.nova_companion_reminders.active(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
        access_tier=owner.access_tier,
        access_version=owner.access_version,
    )
    assert live is not None
    assert live.candidate.schedule_wording is None
    assert live.candidate.temporal is None
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert raw_meta not in contents
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
    assert core._nova_companion_tasks == set()


class FakeTranscription:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "Тестовая расшифровка"


class RuntimeTranscription:
    enabled = True

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls.append((audio, filename))
        return self.transcript


class RuntimeTelegramFile:
    async def download_as_bytearray(self) -> bytearray:
        return bytearray(b"runtime-voice")


class RuntimeVoice:
    duration = 3
    file_size = 20
    mime_type = "audio/ogg"
    file_name = "voice.ogg"

    async def get_file(self) -> RuntimeTelegramFile:
        return RuntimeTelegramFile()


def runtime_settings(**overrides) -> Settings:
    values = {
        "telegram_bot_token": "123456:TEST-TOKEN-FOR-LOCAL-RUNTIME",
        "ai_api_key": "test-ai-key",
        "ai_model": "test-model",
        "database_url": "sqlite+aiosqlite:///:memory:",
    }
    values.update(overrides)
    return Settings(**values)


async def _runtime_subscriber(
    core: FutureSelfBot,
    db,
    telegram_id: int,
    *,
    onboarding_completed: bool,
):
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = onboarding_completed
    return await core._user(telegram_id)


async def _runtime_weekly_focus(
    core: FutureSelfBot,
    db,
    owner: User,
    focus: str,
) -> WeeklyFocus:
    baseline = await core.focus_service.materialize_today_application(
        owner.id,
        include_weekly_focus=False,
    )
    async with db.session() as session:
        weekly_focus = WeeklyFocus(
            owner_id=owner.id,
            week_start=baseline.local_week_start,
            focus=focus,
            approach=None,
            small_steps=[],
            source="text",
        )
        session.add(weekly_focus)
        await session.flush()
        public_id = weekly_focus.public_id
    async with db.sessions() as session:
        stored = await session.scalar(select(WeeklyFocus).where(WeeklyFocus.public_id == public_id))
        assert stored is not None
        return stored


async def _runtime_active_memory(core: FutureSelfBot, owner, telegram_id: int, chat_id: int):
    return await core.nova_memory_sessions.create(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat_id,
        tier=owner.access_tier,
        access_version=owner.access_version,
        canonical_message_id=9_001,
        phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
    )


def _runtime_voice_update(
    application,
    telegram_id: int,
    *,
    update_id: int,
    source_message_id: int,
    progress_message_id: int,
):
    telegram_user = TelegramUser(telegram_id, "Тест", False)
    bot_user = TelegramUser(123456, "Future Self", True)
    chat = Chat(telegram_id, "private")
    source_message = Message(
        source_message_id,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        voice=RuntimeVoice(),
    )
    progress_message = Message(
        progress_message_id,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…",
    )
    update = Update(update_id, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    progress_message.set_bot(application.bot)
    return update, progress_message


def _patch_runtime_voice_transport(monkeypatch, progress_message):
    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return progress_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return progress_message

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    return sent, edits


async def _runtime_stage7c_user(
    core: FutureSelfBot,
    db,
    telegram_id: int,
    *,
    tier: str = "subscriber",
):
    owner = await core._user(telegram_id)
    access = AccessService(db)
    if tier == "admin":
        await access.grant_admin(telegram_id, source="stage7c-test")
    else:
        await access.grant_subscriber(telegram_id, source="stage7c-test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True
    return await core._user(telegram_id)


async def _runtime_stage7c_memory(core: FutureSelfBot, owner, content: str):
    result = await core.nova_memory_service.create(
        telegram_actor_id=owner.telegram_id,
        expected_access_version=owner.access_version,
        category="about_me",
        content=content,
    )
    assert result.status == "created"
    return result


async def _runtime_stage7c_mutate_memory(
    core: FutureSelfBot,
    owner,
    mutation: str,
    initial,
    *,
    suffix: str,
):
    assert initial.item is not None
    item = initial.item
    if mutation == "create":
        result = await core.nova_memory_service.create(
            telegram_actor_id=owner.telegram_id,
            expected_access_version=owner.access_version,
            category="orientation",
            content=f"NEW_STAGE7C_{suffix}_MEMORY",
        )
        assert result.status == "created"
    elif mutation == "edit":
        result = await core.nova_memory_service.update(
            telegram_actor_id=owner.telegram_id,
            public_id=item.public_id,
            expected_version=item.version,
            expected_access_version=owner.access_version,
            content=f"EDITED_STAGE7C_{suffix}_MEMORY",
        )
        assert result.status == "updated"
    elif mutation == "toggle":
        result = await core.nova_memory_service.set_important(
            telegram_actor_id=owner.telegram_id,
            public_id=item.public_id,
            expected_version=item.version,
            expected_access_version=owner.access_version,
            important=not item.important,
        )
        assert result.status == "importance_changed"
    elif mutation == "delete":
        result = await core.nova_memory_service.delete(
            telegram_actor_id=owner.telegram_id,
            public_id=item.public_id,
            expected_version=item.version,
            expected_access_version=owner.access_version,
        )
        assert result.status == "deleted"
    else:
        assert mutation == "delete_all"
        page = await core.nova_memory_service.list(
            telegram_actor_id=owner.telegram_id,
        )
        assert page.status == "ok"
        assert page.collection_revision is not None
        result = await core.nova_memory_service.delete_all(
            telegram_actor_id=owner.telegram_id,
            expected_access_version=owner.access_version,
            expected_collection_revision=page.collection_revision,
        )
        assert result.status == "deleted_all"
    assert result.affected_count == 1


def _runtime_text_update(
    application,
    telegram_id: int,
    text: str,
    *,
    update_id: int,
    source_message_id: int,
):
    telegram_user = TelegramUser(telegram_id, False, "Тест")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        source_message_id,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=text,
    )
    update = Update(update_id, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    return update


def _runtime_bot_message(application, chat_id: int, message_id: int, text: str = "pending"):
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(chat_id, "private")
    message = Message(
        message_id,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text=text,
    )
    message.set_bot(application.bot)
    return message


def _patch_runtime_stage7c_transport(
    monkeypatch,
    returned_messages,
    *,
    send_started: asyncio.Event | None = None,
    send_release: asyncio.Event | None = None,
):
    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []
    queue = list(returned_messages)

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        if send_started is not None:
            send_started.set()
        if send_release is not None:
            await send_release.wait()
        assert queue
        return queue.pop(0)

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return returned_messages[-1]

    async def fake_delete_message(self, *args, **kwargs):
        del self, args
        deletes.append(kwargs)
        return True

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "delete_message", fake_delete_message)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    return sent, edits, deletes


def _patch_runtime_stage7c_cleanup_transport(
    monkeypatch,
    returned_message,
    *,
    delete_error: BaseException | None,
    edit_error: BaseException | None = None,
):
    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return returned_message

    async def fake_delete_message(self, *args, **kwargs):
        del self, args
        deletes.append(kwargs)
        if delete_error is not None:
            raise delete_error
        return False

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        if edit_error is not None:
            raise edit_error
        return returned_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "delete_message", fake_delete_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    return sent, edits, deletes


def _patch_runtime_voice_priority_spies(core: FutureSelfBot, fake_ai, monkeypatch):
    calls: list[str] = []

    async def memory_gate(*args, **kwargs):
        del args, kwargs
        calls.append("memory")
        return False

    async def reminder_gate(*args, **kwargs):
        del args, kwargs
        calls.append("reminder")
        return False

    async def nova_gate(*args, **kwargs):
        del args, kwargs
        calls.append("nova")
        return False

    async def generic_route(*args, **kwargs):
        del args, kwargs
        calls.append("generic")

    async def forbidden_provider(*args, **kwargs):
        del args, kwargs
        raise AssertionError("durable voice owner must stop provider routing")

    monkeypatch.setattr(core, "nova_memory_voice_gate", memory_gate)
    monkeypatch.setattr(core, "reminder_voice_gate", reminder_gate)
    monkeypatch.setattr(core, "nova_voice_gate", nova_gate)
    monkeypatch.setattr(core, "_route_message", generic_route)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_provider)
    monkeypatch.setattr(fake_ai, "answer_message", forbidden_provider)
    monkeypatch.setattr(fake_ai, "nova_help", forbidden_provider, raising=False)
    return calls


async def _assert_runtime_voice_memory_untouched(db) -> None:
    async with db.sessions() as session:
        assert list((await session.scalars(select(NovaMemoryItem))).all()) == []
        assert list((await session.scalars(select(NovaMemoryChange))).all()) == []


def _assert_runtime_voice_canonical(
    sent: list[dict[str, object]],
    edits: list[dict[str, object]],
    progress_message_id: int,
) -> None:
    assert [entry["text"] for entry in sent] == ["Расшифровываю голосовую мысль…"]
    assert edits
    assert all(entry["message_id"] == progress_message_id for entry in edits)
    assert all(
        not str(button.callback_data).startswith("nmem:")
        for entry in (*sent, *edits)
        if entry.get("reply_markup") is not None
        for row in getattr(entry["reply_markup"], "inline_keyboard", ())
        for button in row
    )


def test_missing_environment_variables_are_reported_without_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)
    message = format_configuration_error(caught.value)
    assert "TELEGRAM_BOT_TOKEN" in message
    assert "AI_API_KEY" in message
    assert "Значения не выводятся" in message


async def test_doctor_default_makes_no_network_calls(db, monkeypatch):
    async with db.session() as session:
        await session.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        await session.execute(
            text("INSERT INTO alembic_version (version_num) VALUES ('20260822_0028')")
        )

    async def forbidden_network(*args, **kwargs):
        raise AssertionError("network client must not be called")

    monkeypatch.setattr("telegram.Bot.get_me", forbidden_network)
    monkeypatch.setattr("openai.resources.models.AsyncModels.retrieve", forbidden_network)
    report = await run_diagnostics(network=False, database_url=db.url)
    assert report.exit_code == 0
    assert any(check.name == "network" and check.status == "WARN" for check in report.checks)


def test_application_starts_with_fake_services(fake_ai):
    captured = []
    run(
        runtime_settings(),
        ai=fake_ai,
        transcription=FakeTranscription(),
        application_runner=captured.append,
    )
    assert len(captured) == 1
    assert captured[0].bot.token.startswith("123456:")


async def test_guest_services_share_one_policy_instance(db, fake_ai):
    bot = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        FakeTranscription(),
    )
    assert bot.guest_session_service.policy is bot.guest_quota_policy
    assert bot.guest_quota_service is bot.guest_session_service.quota
    assert bot.guest_quota_service.policy is bot.guest_quota_policy


async def test_post_init_schedules_guest_recovery_before_job_queue_early_return(
    db,
    fake_ai,
    monkeypatch,
):
    bot = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        FakeTranscription(),
    )
    recovery_started = asyncio.Event()
    recovery_release = asyncio.Event()

    async def blocked_recovery(_telegram_bot):
        recovery_started.set()
        await recovery_release.wait()

    monkeypatch.setattr(bot, "_recover_guest_demo_results", blocked_recovery)

    class TelegramBot:
        async def set_my_commands(self, commands, **kwargs):
            return None

        async def set_chat_menu_button(self, **kwargs):
            return None

    def forbidden_application_task(*args, **kwargs):
        raise AssertionError("post_init must not use Application.create_task")

    await bot._post_init(
        SimpleNamespace(
            bot=TelegramBot(),
            job_queue=None,
            create_task=forbidden_application_task,
            post_stop=bot._post_stop,
        )
    )
    await recovery_started.wait()
    maintenance = bot._guest_maintenance_task
    assert maintenance is not None
    assert maintenance.get_name() == "guest-result-maintenance"
    assert not maintenance.done()
    await bot._post_stop(SimpleNamespace())
    assert maintenance.done()
    assert bot._guest_maintenance_task is None
    recovery_release.set()
    await bot._post_shutdown(SimpleNamespace())
    assert bot._guest_maintenance_task is None


async def test_guest_maintenance_cleanup_repeats_without_sleep_based_race(
    db,
    fake_ai,
    monkeypatch,
):
    bot = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        FakeTranscription(),
    )
    cleanup_calls = 0
    events: list[str] = []
    first_wait = asyncio.Event()
    release_wait = asyncio.Event()
    second_cleanup = asyncio.Event()

    async def cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1
        events.append("cleanup")
        if cleanup_calls == 2:
            second_cleanup.set()

    async def recovery(_telegram_bot):
        events.append("recovery")
        return None

    async def controlled_wait():
        first_wait.set()
        await release_wait.wait()
        release_wait.clear()

    monkeypatch.setattr(bot, "_cleanup_guest_results_safely", cleanup)
    monkeypatch.setattr(bot, "_recover_guest_demo_results", recovery)
    monkeypatch.setattr(bot, "_guest_maintenance_wait", controlled_wait)
    bot._start_guest_maintenance(SimpleNamespace())
    await first_wait.wait()
    assert cleanup_calls == 1
    assert events == ["cleanup", "recovery"]
    release_wait.set()
    await second_cleanup.wait()
    assert cleanup_calls == 2
    await bot._stop_guest_maintenance()


async def test_scheduler_startup_does_not_wait_for_blocked_guest_recovery(
    db,
    fake_ai,
    monkeypatch,
):
    bot = FutureSelfBot(
        runtime_settings(database_url=db.url, enable_task_reminders=False),
        db,
        fake_ai,
        FakeTranscription(),
    )
    recovery_started = asyncio.Event()
    recovery_release = asyncio.Event()
    repeating: list[str] = []

    async def blocked_recovery(_telegram_bot):
        recovery_started.set()
        await recovery_release.wait()

    monkeypatch.setattr(bot, "_recover_guest_demo_results", blocked_recovery)

    class TelegramBot:
        async def set_my_commands(self, commands, **kwargs):
            return None

        async def set_chat_menu_button(self, **kwargs):
            return None

    class Queue:
        def run_repeating(self, callback, **kwargs):
            repeating.append(kwargs["name"])

    await bot._post_init(
        SimpleNamespace(
            bot=TelegramBot(),
            job_queue=Queue(),
            post_stop=bot._post_stop,
        )
    )
    assert bot.scheduler is not None
    assert repeating == ["labs:cleanup"]
    await recovery_started.wait()
    assert bot._guest_maintenance_task is not None
    assert not bot._guest_maintenance_task.done()
    recovery_release.set()
    await bot._post_stop(SimpleNamespace())


async def test_guest_cleanup_iteration_failure_is_safe_and_loop_continues(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    bot = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    calls = 0
    wait_started = asyncio.Event()
    release_wait = asyncio.Event()
    recovered = asyncio.Event()
    private_detail = "PRIVATE_MAINTENANCE_DETAIL"

    async def cleanup_undeliverable_results():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(private_detail)
        recovered.set()
        return 0

    async def recovery(_telegram_bot):
        return None

    async def controlled_wait():
        wait_started.set()
        await release_wait.wait()
        release_wait.clear()

    monkeypatch.setattr(
        bot.guest_session_service,
        "cleanup_undeliverable_results",
        cleanup_undeliverable_results,
    )
    monkeypatch.setattr(bot, "_recover_guest_demo_results", recovery)
    monkeypatch.setattr(bot, "_guest_maintenance_wait", controlled_wait)
    with caplog.at_level(logging.ERROR):
        bot._start_guest_maintenance(SimpleNamespace())
        await wait_started.wait()
        release_wait.set()
        await recovered.wait()
        await bot._stop_guest_maintenance()
    assert calls == 2
    assert "Guest result cleanup failed error_type=RuntimeError" in caplog.text
    assert private_detail not in caplog.text


async def test_blocked_guest_recovery_does_not_delay_scheduler(db, fake_ai, monkeypatch):
    bot = FutureSelfBot(
        runtime_settings(database_url=db.url, enable_task_reminders=False),
        db,
        fake_ai,
        FakeTranscription(),
    )
    recovery_started = asyncio.Event()
    recovery_release = asyncio.Event()
    repeating: list[str] = []

    async def blocked_recovery(_telegram_bot):
        recovery_started.set()
        await recovery_release.wait()

    monkeypatch.setattr(bot, "_recover_guest_demo_results", blocked_recovery)

    class TelegramBot:
        async def set_my_commands(self, commands, **kwargs):
            return None

        async def set_chat_menu_button(self, **kwargs):
            return None

    class Queue:
        def run_repeating(self, callback, **kwargs):
            repeating.append(kwargs["name"])

    app = SimpleNamespace(
        bot=TelegramBot(),
        job_queue=Queue(),
        post_stop=bot._post_stop,
    )
    await bot._post_init(app)
    assert bot.scheduler is not None
    assert repeating == ["labs:cleanup"]
    await recovery_started.wait()
    assert bot._guest_maintenance_task is not None
    assert not bot._guest_maintenance_task.done()
    recovery_release.set()
    await bot._post_stop(app)


def test_critical_startup_error_is_safe_and_nonzero(monkeypatch, caplog):
    private_detail = "https://api.telegram.org/botSECRET/getMe"
    monkeypatch.setattr(main_module, "get_settings", runtime_settings)

    def fail_startup(settings):
        raise RuntimeError(private_detail)

    monkeypatch.setattr(main_module, "run", fail_startup)
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as stopped:
        main_module.main()
    assert "Не удалось запустить" in str(stopped.value)
    assert private_detail not in str(stopped.value)
    assert private_detail not in caplog.text


def test_key_telegram_handlers_are_registered(fake_ai):
    settings = runtime_settings()
    from future_self.db import Database

    database = Database(settings.database_url)
    bot = FutureSelfBot(settings, database, fake_ai, FakeTranscription())
    application = create_application(settings, database, fake_ai, FakeTranscription())
    handlers = application.handlers[0]
    vision_gate_handlers = application.handlers[-1]

    assert application.handlers[-5][0].callback.__name__ == "private_chat_guard"
    assert isinstance(application.handlers[-4][0], TypeHandler)
    assert application.handlers[-4][0].callback.__name__ == "access_gate"
    assert application.handlers[-3][0].callback.__name__ == "system_action_text_gate"
    assert isinstance(handlers[0], ConversationHandler)
    assert isinstance(handlers[1], ConversationHandler)
    assert isinstance(handlers[2], ConversationHandler)
    assert isinstance(handlers[3], ConversationHandler)
    onboarding_commands = {
        command
        for handler in handlers[0].entry_points
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }
    evening_commands = {
        command
        for handler in handlers[1].entry_points
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }
    assert onboarding_commands == {"start", "onboarding"}
    assert evening_commands == {"evening"}
    health_commands = {
        command
        for handler in handlers[2].entry_points
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }
    assert health_commands == {"checkin", "health_edit"}
    doctor_commands = {
        command
        for handler in handlers[3].entry_points
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }
    assert doctor_commands == {"doctor_prepare", "doctor_prepare_edit"}
    commands = {
        command
        for handler in handlers
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }
    assert {
        "help",
        "profile",
        "location",
        "timezone",
        "mynova",
        "goals",
        "inbox",
        "tasks",
        "drafts",
        "last_saved",
        "cleanup_drafts",
        "today",
        "week",
        "cancel",
        "health",
        "health_delete",
        "health_reminder_on",
        "health_reminder_off",
        "doctor_preparations",
        "doctor_prepare_show",
        "doctor_prepare_delete",
        "doctor_prepare_task",
        "doctor_find",
        "doctor_find_task",
    } <= commands
    assert sum(isinstance(handler, CallbackQueryHandler) for handler in handlers) == 23
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "weekly_review_callback"
        and getattr(handler.pattern, "pattern", None) == r"^wrev:[A-Za-z0-9_-]+$"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "nova_memory_callback"
        and getattr(handler.pattern, "pattern", None) == r"^nmem:[A-Za-z0-9_-]+$"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "nova_companion_callback"
        and getattr(handler.pattern, "pattern", None) == r"^ncap:[A-Za-z0-9_-]+$"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "nova_brain_callback"
        and getattr(handler.pattern, "pattern", None) == r"^nbrain:[A-Za-z0-9_-]+$"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "nova_companion_reminder_callback"
        and getattr(handler.pattern, "pattern", None) == r"^nrem:[A-Za-z0-9_-]+$"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "reminder_callback"
        and getattr(handler.pattern, "pattern", None) == r"^rmd:[A-Za-z0-9_-]+$"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler) and handler.callback.__name__ == "nova_callback"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler) and handler.callback.__name__ == "profile_action"
        for handler in handlers
    )
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "saved_inbox_action"
        for handler in handlers
    )
    assert sum(isinstance(handler, MessageHandler) for handler in handlers) == 2
    gate_commands = {
        command
        for handler in vision_gate_handlers
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }
    assert gate_commands == {"vision", "labs", "cancel"}
    assert sum(isinstance(handler, CallbackQueryHandler) for handler in vision_gate_handlers) == 2
    assert sum(isinstance(handler, MessageHandler) for handler in vision_gate_handlers) == 3
    assert bot.error_handler.__name__ in {
        callback.__name__ for callback in application.error_handlers
    }
    assert application.post_stop is not None
    assert application.post_stop.__name__ == "_post_stop"
    assert application.post_stop is not None
    assert application.post_stop.__self__.__class__ is FutureSelfBot


async def test_real_application_routes_cleanup_before_persistent_onboarding(
    db, fake_ai, monkeypatch
):
    core = FutureSelfBot(runtime_settings(), db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    owner = await core._user(712345)
    await AccessService(db).grant_subscriber(712345, source="test")
    async with db.session() as session:
        session.add(
            OnboardingState(
                user_id=owner.id,
                current_step=2,
                answers={"display_name": "Тест"},
                status="in_progress",
            )
        )

    sent: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return message

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    telegram_user = TelegramUser(712345, False, "Тест")
    chat = Chat(712345, "private")
    message = Message(
        91,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text="Удали все просроченные",
    )
    update = Update(991, message=message)
    update.set_bot(application.bot)
    message.set_bot(application.bot)

    await application.process_update(update)

    assert sent and sent[-1]["text"].startswith("Неактуальных задач не найдено")
    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
        inbox_count = len((await session.scalars(select(InboxItem))).all())
    assert state.current_step == 2
    assert state.answers == {"display_name": "Тест"}
    assert inbox_count == 0
    assert fake_ai.route_calls == []

    reminder_control_message = Message(
        92,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text="Отключи старые напоминания",
    )
    reminder_control_update = Update(992, message=reminder_control_message)
    reminder_control_update.set_bot(application.bot)
    reminder_control_message.set_bot(application.bot)

    await application.process_update(reminder_control_update)

    assert sent[-1]["text"].startswith("Похоже на команду удаления")
    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
    assert state.current_step == 2
    assert state.answers == {"display_name": "Тест"}

    narrative = "В будущем я хочу научиться удалять все неактуальные задачи вовремя"
    narrative_message = Message(
        93,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=narrative,
    )
    narrative_update = Update(993, message=narrative_message)
    narrative_update.set_bot(application.bot)
    narrative_message.set_bot(application.bot)

    await application.process_update(narrative_update)

    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
        inbox_count = len((await session.scalars(select(InboxItem))).all())
    assert state.current_step == 3
    assert state.answers["display_name"] == "Тест"
    assert state.answers["future_life"] == narrative
    assert inbox_count == 0
    assert fake_ai.route_calls == []

    ambiguous_narrative = "В будущем я хочу научиться удалять просроченные файлы без стресса"
    ambiguous_message = Message(
        94,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=ambiguous_narrative,
    )
    ambiguous_update = Update(994, message=ambiguous_message)
    ambiguous_update.set_bot(application.bot)
    ambiguous_message.set_bot(application.bot)

    await application.process_update(ambiguous_update)

    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
        inbox_count = len((await session.scalars(select(InboxItem))).all())
    assert state.current_step == 4
    assert state.answers["residence"] == ambiguous_narrative
    assert inbox_count == 0
    assert fake_ai.route_calls == []

    async with db.session() as session:
        stored_owner = await session.get(User, owner.id)
        stored_state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
        stored_owner.onboarding_completed = True
        stored_state.status = "completed"
    sent.clear()
    task_message = Message(
        95,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text="Открой задачи и напоминания",
    )
    task_update = Update(995, message=task_message)
    task_update.set_bot(application.bot)
    task_message.set_bot(application.bot)

    await application.process_update(task_update)

    assert sent and str(sent[-1]["text"]).startswith("✅ Задачи и напоминания")


@pytest.mark.parametrize(
    ("phrase", "expected_text", "exact"),
    [
        ("Где мои задачи?", "✅ Задачи", False),
        ("Где календарь?", NOVA_ROOT_TEXT, True),
    ],
)
async def test_real_application_stops_natural_navigation_before_downstream_content(
    db, fake_ai, monkeypatch, caplog, phrase, expected_text, exact
):
    core = FutureSelfBot(runtime_settings(), db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    telegram_id = 712346
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    sent: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def downstream_handler(update, context):
        del context
        downstream.append(update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    application.add_handler(TypeHandler(Update, downstream_handler), group=100)

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    chat = Chat(telegram_id, "private")
    message = Message(
        96,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=phrase,
    )
    update = Update(996, message=message)
    update.set_bot(application.bot)
    message.set_bot(application.bot)

    with caplog.at_level(logging.INFO):
        await application.process_update(update)

    assert downstream == []
    assert sent
    if exact:
        assert sent[-1]["text"] == expected_text
    else:
        assert str(sent[-1]["text"]).startswith(expected_text)
    assert fake_ai.route_calls == []
    assert phrase not in caplog.text
    async with db.sessions() as session:
        assert len((await session.scalars(select(InboxItem))).all()) == 0
        assert len((await session.scalars(select(DraftInboxItem))).all()) == 0
        assert len((await session.scalars(select(ConversationMessage))).all()) == 0


async def test_real_application_explicit_reminder_owns_text_and_callback_once(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712360
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        106,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text="Напомни завтра в 19:30 заполнить дневник благодарностей",
    )
    canonical_message = Message(
        206,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="🔔 Проверь напоминание",
    )
    update = Update(1006, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def fake_answer_callback_query(self, callback_query_id, *args, **kwargs):
        del self, args
        answers.append({"callback_query_id": callback_query_id, **kwargs})
        return True

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def forbidden_provider(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reminder routing must not call AI")

    async def downstream_handler(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "answer_callback_query", fake_answer_callback_query)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_provider)
    monkeypatch.setattr(fake_ai, "nova_help", forbidden_provider, raising=False)
    downstream_probe = TypeHandler(Update, downstream_handler)
    application.add_handler(downstream_probe, group=100)

    await application.process_update(update)

    assert downstream == []
    assert len(sent) == 1
    assert len(edits) == 1
    assert sent[0]["text"] == "🔔 Готовлю напоминание…"
    assert sent[0].get("reply_markup") is None
    assert edits[0]["message_id"] == canonical_message.message_id
    assert str(edits[0]["text"]).startswith("🔔 Проверь напоминание")
    reminder_callbacks = [
        str(button.callback_data)
        for row in edits[0]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert reminder_callbacks
    assert all(callback.startswith("rmd:") for callback in reminder_callbacks)
    current = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat.id,
    )
    assert current is not None
    assert current.canonical_message_id == canonical_message.message_id
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )
    application.remove_handler(downstream_probe, group=100)

    cancel_data = next(
        str(button.callback_data)
        for row in edits[0]["reply_markup"].inline_keyboard
        for button in row
        if button.text == "Отмена"
    )
    query = CallbackQuery(
        "runtime-reminder-cancel",
        telegram_user,
        "runtime-reminder-chat",
        message=canonical_message,
        data=cancel_data,
    )
    callback_update = Update(1007, callback_query=query)
    callback_update.set_bot(application.bot)
    query.set_bot(application.bot)

    await application.process_update(callback_update)

    assert len(answers) == 1
    assert answers[0]["callback_query_id"] == query.id
    assert len(sent) == 1
    assert len(edits) == 2
    assert edits[-1]["text"] == "🔔 Напоминание отменено. Ничего не сохранено."
    assert downstream == []
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )


async def test_real_application_explicit_memory_stops_every_downstream_pipeline_before_confirm(
    db,
    fake_ai,
    monkeypatch,
):
    settings = runtime_settings(
        database_url=db.url,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    core = FutureSelfBot(settings, db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    telegram_id = 712365
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    private_content = "PRIVATE_MEMORY_RUNTIME_SENTINEL"
    source_message = Message(
        116,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=f"Nova, запомни: {private_content}",
    )
    canonical_message = Message(
        216,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="🧬 Моя Nova",
    )
    update = Update(1016, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def forbidden_provider(*args, **kwargs):
        del args, kwargs
        raise AssertionError("memory CRUD routing must not call AI")

    async def downstream_handler(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_provider)
    monkeypatch.setattr(fake_ai, "nova_help", forbidden_provider, raising=False)
    application.add_handler(TypeHandler(Update, downstream_handler), group=100)

    await application.process_update(update)

    assert downstream == []
    assert [entry["text"] for entry in sent] == ["🧬 Моя Nova\n\nОткрываю…"]
    assert len(edits) == 1
    assert edits[0]["message_id"] == canonical_message.message_id
    assert private_content in str(edits[0]["text"])
    callbacks = [
        str(button.callback_data)
        for row in edits[0]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert callbacks and all(callback.startswith("nmem:") for callback in callbacks)
    assert all(private_content not in callback for callback in callbacks)
    assert fake_ai.route_calls == []
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )
    async with db.sessions() as session:
        assert list((await session.scalars(select(NovaMemoryItem))).all()) == []
        assert list((await session.scalars(select(NovaMemoryChange))).all()) == []
        assert list((await session.scalars(select(DraftInboxItem))).all()) == []
        assert list((await session.scalars(select(InboxItem))).all()) == []


async def test_real_application_ordinary_text_keeps_generic_pipeline(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712361
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    phrase = "Привет"
    source_message = Message(
        107,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=phrase,
    )
    bot_message = Message(
        207,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Привет!",
    )
    update = Update(1008, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    bot_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return bot_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)

    await application.process_update(update)

    assert [call[0] for call in fake_ai.route_calls] == [phrase]
    assert sent and sent[-1]["text"] == "Привет!"
    assert all(
        not str(button.callback_data).startswith("rmd:")
        for item in sent
        if item.get("reply_markup") is not None
        for row in getattr(item["reply_markup"], "inline_keyboard", ())
        for button in row
    )
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )


async def test_real_application_explicit_voice_reminder_reuses_progress_message(
    db,
    fake_ai,
    monkeypatch,
):
    transcript = "Каждый день в 19:30 напоминай заполнить дневник"
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712364
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        110,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        voice=RuntimeVoice(),
    )
    progress_message = Message(
        210,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…",
    )
    update = Update(1011, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    progress_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return progress_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return progress_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def forbidden_provider(*args, **kwargs):
        del args, kwargs
        raise AssertionError("voice reminder routing must not call AI")

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_provider)
    monkeypatch.setattr(fake_ai, "nova_help", forbidden_provider, raising=False)

    await application.process_update(update)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert len(sent) == 1
    assert sent[0]["text"] == "Расшифровываю голосовую мысль…"
    assert len(edits) == 1
    assert edits[0]["message_id"] == progress_message.message_id
    assert str(edits[0]["text"]).startswith("🔁 Проверь напоминание")
    assert "Я услышал" not in str(edits[0]["text"])
    assert all(
        str(button.callback_data).startswith("rmd:")
        for row in edits[0]["reply_markup"].inline_keyboard
        for button in row
    )
    current = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat.id,
    )
    assert current is not None
    assert current.canonical_message_id == progress_message.message_id
    assert transcript not in repr(current)
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )


async def test_real_application_explicit_voice_memory_reuses_progress_and_stops_generic(
    db,
    fake_ai,
    monkeypatch,
):
    private_content = "PRIVATE_VOICE_MEMORY_SENTINEL"
    transcript = f"Nova, запомни: {private_content}"
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712374
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        117,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        voice=RuntimeVoice(),
    )
    progress_message = Message(
        217,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…",
    )
    update = Update(1017, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    progress_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return progress_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return progress_message

    async def forbidden_provider(*args, **kwargs):
        del args, kwargs
        raise AssertionError("voice memory routing must not call AI")

    async def downstream_handler(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_provider)
    monkeypatch.setattr(fake_ai, "nova_help", forbidden_provider, raising=False)
    application.add_handler(TypeHandler(Update, downstream_handler), group=100)

    await application.process_update(update)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert downstream == []
    assert [entry["text"] for entry in sent] == ["Расшифровываю голосовую мысль…"]
    assert len(edits) == 1
    assert edits[0]["message_id"] == progress_message.message_id
    assert private_content in str(edits[0]["text"])
    assert "Я услышал" not in str(edits[0]["text"])
    callbacks = [
        str(button.callback_data)
        for row in edits[0]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert callbacks and all(callback.startswith("nmem:") for callback in callbacks)
    current = await core.nova_memory_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat.id,
    )
    assert current is not None
    assert current.canonical_message_id == progress_message.message_id
    assert private_content not in repr(current)
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        assert list((await session.scalars(select(NovaMemoryItem))).all()) == []
        assert list((await session.scalars(select(NovaMemoryChange))).all()) == []


@pytest.mark.parametrize(
    ("source", "telegram_id", "phrase", "fragment", "timezone", "provider_calls"),
    [
        (
            "text",
            712_365,
            "Напомни завтра в 9:00 по Светогорску позвонить врачу",
            "по Светогорску",
            "Europe/Moscow",
            1,
        ),
        (
            "voice",
            712_366,
            "Напомни завтра в 9:00 по Светогорску позвонить врачу",
            "по Светогорску",
            "Europe/Moscow",
            1,
        ),
        (
            "voice",
            712_367,
            "Напомни завтра в 10:00 по Лондону созвониться с клиентом",
            "по Лондону",
            "Europe/London",
            0,
        ),
        (
            "text",
            712_370,
            "Напомни завтра в 10:00 по МСК созвониться с клиентом",
            "по МСК",
            "Europe/Moscow",
            0,
        ),
        (
            "voice",
            712_371,
            "Напомни завтра в 10:00 по Europe/London созвониться с клиентом",
            "по Europe/London",
            "Europe/London",
            0,
        ),
    ],
)
async def test_real_application_natural_timezone_reminder_has_text_stt_parity(
    db,
    fake_ai,
    monkeypatch,
    source,
    telegram_id,
    phrase,
    fragment,
    timezone,
    provider_calls,
):
    provider_fragment = None
    if provider_calls:
        extracted = extract_reminder_timezone_fragment(phrase)
        assert extracted is not None
        provider_fragment = extracted.text
        fake_ai.reminder_timezone_results[provider_fragment] = ReminderTimezoneResolution(
            status="resolved",
            timezone=timezone,
            matched_text=fragment,
            city="Светогорск",
            country="Россия",
        )
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    message_kwargs = {"text": phrase} if source == "text" else {"voice": RuntimeVoice()}
    source_message = Message(
        111,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        **message_kwargs,
    )
    canonical_message = Message(
        211,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…" if source == "voice" else "🔔 Готовлю напоминание…",
    )
    update = Update(1012, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def forbidden_assistant(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reminder timezone routing must not enter generic AI or Nova")

    async def downstream_handler(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_assistant)
    monkeypatch.setattr(fake_ai, "nova_help", forbidden_assistant, raising=False)
    downstream_probe = TypeHandler(Update, downstream_handler)
    application.add_handler(downstream_probe, group=100)

    await application.process_update(update)

    assert downstream == ([] if source == "text" else [update.update_id])
    assert len(sent) == 1
    assert len(edits) == (2 if provider_calls else 1)
    assert all(edit["message_id"] == canonical_message.message_id for edit in edits)
    assert str(edits[-1]["text"]).startswith("🔔 Проверь напоминание")
    assert timezone in str(edits[-1]["text"])
    assert fake_ai.reminder_timezone_calls == [provider_fragment] * provider_calls
    assert fake_ai.route_calls == []
    if source == "voice":
        assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
        assert sent[0]["text"] == "Расшифровываю голосовую мысль…"
        assert "Я услышал" not in str(edits[-1]["text"])
    else:
        assert transcription.calls == []
        assert sent[0]["text"] == "🔔 Готовлю напоминание…"
    current = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat.id,
    )
    assert current is not None
    assert current.canonical_message_id == canonical_message.message_id
    assert current.timezone == timezone
    assert current.timezone_source == "explicit"
    assert phrase not in repr(current)
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )


@pytest.mark.parametrize(("checkpoint", "telegram_id"), [("stt", 712_368), ("provider", 712_369)])
async def test_real_application_voice_timezone_access_version_bounce_is_fail_closed(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    checkpoint,
    telegram_id,
):
    phrase = "Напомни завтра в 9:00 по Светогорску позвонить врачу"
    timezone_expression = "по Светогорску"
    extracted = extract_reminder_timezone_fragment(phrase)
    assert extracted is not None
    fragment = extracted.text
    access = AccessService(db)

    async def bounce_access() -> None:
        await access.set_guest(telegram_id, source="test-bounce")
        await access.grant_subscriber(telegram_id, source="test-bounce")

    class BouncingTranscription(RuntimeTranscription):
        async def transcribe(self, audio: bytes, filename: str) -> str:
            self.calls.append((audio, filename))
            await bounce_access()
            return self.transcript

    transcription = (
        BouncingTranscription(phrase) if checkpoint == "stt" else RuntimeTranscription(phrase)
    )
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text=timezone_expression,
        city="Светогорск",
        country="Россия",
    )
    if checkpoint == "provider":
        fake_ai.reminder_timezone_release.clear()

    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await core._user(telegram_id)
    await access.grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True
    initial = await access.status(telegram_id)
    assert initial is not None

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        112,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        voice=RuntimeVoice(),
    )
    progress_message = Message(
        212,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…",
    )
    update = Update(1013, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    progress_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return progress_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return progress_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def forbidden_assistant(*args, **kwargs):
        del args, kwargs
        raise AssertionError("stale reminder input must not enter generic AI or Nova")

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_assistant)
    monkeypatch.setattr(fake_ai, "nova_help", forbidden_assistant, raising=False)

    with caplog.at_level(logging.INFO):
        processing = asyncio.create_task(application.process_update(update))
        if checkpoint == "provider":
            await asyncio.wait_for(fake_ai.reminder_timezone_started.wait(), timeout=5)
            await bounce_access()
            fake_ai.reminder_timezone_release.set()
        await asyncio.wait_for(processing, timeout=10)

    current_status = await access.status(telegram_id)
    assert current_status is not None
    assert current_status.access_tier == "subscriber"
    assert current_status.access_version == initial.access_version + 2
    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert len(sent) == 1
    assert sent[0]["text"] == "Расшифровываю голосовую мысль…"
    assert edits
    assert edits[-1]["message_id"] == progress_message.message_id
    assert "Доступ изменился" in str(edits[-1]["text"])
    assert all("Проверь напоминание" not in str(edit["text"]) for edit in edits)
    assert fake_ai.reminder_timezone_calls == ([fragment] if checkpoint == "provider" else [])
    assert fake_ai.route_calls == []
    assert phrase not in caplog.text
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )
    async with db.sessions() as session:
        assert len((await session.scalars(select(InboxItem))).all()) == 0
        assert len((await session.scalars(select(DraftInboxItem))).all()) == 0
        assert len((await session.scalars(select(ConversationMessage))).all()) == 0


async def test_real_application_onboarding_owns_explicit_reminder_text(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712362
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        session.add(
            OnboardingState(
                user_id=owner.id,
                current_step=2,
                answers={"display_name": "Тест"},
                status="in_progress",
            )
        )

    phrase = "Напомни завтра в 19:30 заполнить дневник"
    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        108,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=phrase,
    )
    bot_message = Message(
        208,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Регистрация",
    )
    update = Update(1009, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    bot_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return bot_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)

    await application.process_update(update)

    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
    assert state is not None
    assert state.current_step == 3
    assert state.answers["future_life"] == phrase
    assert fake_ai.route_calls == []
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )
    assert all(
        not str(button.callback_data).startswith("rmd:")
        for item in sent
        if item.get("reply_markup") is not None
        for row in getattr(item["reply_markup"], "inline_keyboard", ())
        for button in row
    )


async def test_real_application_onboarding_owns_explicit_reminder_voice(
    db,
    fake_ai,
    monkeypatch,
):
    transcript = "Напомни завтра в 19:30 заполнить дневник"
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712363
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        session.add(
            OnboardingState(
                user_id=owner.id,
                current_step=2,
                answers={"display_name": "Тест"},
                status="in_progress",
            )
        )

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        109,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        voice=RuntimeVoice(),
    )
    progress_message = Message(
        209,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…",
    )
    update = Update(1010, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    progress_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return progress_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return progress_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)

    await application.process_update(update)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
    assert state is not None
    assert state.current_step == 3
    assert state.answers["future_life"] == transcript
    assert fake_ai.route_calls == []
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )
    assert all(
        not str(button.callback_data).startswith("rmd:")
        for item in [*sent, *edits]
        if item.get("reply_markup") is not None
        for row in getattr(item["reply_markup"], "inline_keyboard", ())
        for button in row
    )


async def test_real_application_explicit_nova_uses_one_canonical_message_and_stops_pipeline(
    db, fake_ai, monkeypatch
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    telegram_id = 712348
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        98,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text="Nova, как добавить задачу с напоминанием?",
    )
    canonical_message = Message(
        198,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="✨ Nova\n\nРазбираю вопрос…",
    )
    update = Update(998, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def downstream_handler(update, context):
        del context
        downstream.append(update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    application.add_handler(TypeHandler(Update, downstream_handler), group=100)

    await application.process_update(update)

    assert downstream == []
    assert len(sent) == 1
    assert sent[0]["text"] == "✨ Nova\n\nРазбираю вопрос…"
    assert len(edits) == 1
    assert edits[0]["message_id"] == canonical_message.message_id
    assert str(edits[0]["text"]).startswith("✨ Nova")
    assert "напомин" in str(edits[0]["text"]).casefold()
    assert fake_ai.route_calls == []
    current = await core.nova_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat.id,
    )
    assert current is not None
    assert current.canonical_message_id == canonical_message.message_id
    async with db.sessions() as session:
        assert len((await session.scalars(select(InboxItem))).all()) == 0
        assert len((await session.scalars(select(DraftInboxItem))).all()) == 0
        assert len((await session.scalars(select(ConversationMessage))).all()) == 0


async def test_real_application_natural_nova_text_stops_before_generic_content(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    telegram_id = 712351
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    phrase = (
        "Просто я знаю, что в этом боте есть визуализация, но не могу её найти "
        "в менюшке. Подскажи, пожалуйста."
    )
    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        101,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=phrase,
    )
    canonical_message = Message(
        201,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="✨ Nova\n\nРазбираю вопрос…",
    )
    update = Update(1001, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    downstream: list[int] = []
    generic_answer_calls: list[str] = []
    nova_provider_calls: list[str] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def downstream_handler(late_update, context):
        del context
        downstream.append(late_update.update_id)

    async def generic_answer(*args, **kwargs):
        del args, kwargs
        generic_answer_calls.append("answer_message")
        return None

    async def nova_provider(*args, **kwargs):
        del args, kwargs
        nova_provider_calls.append("nova_help")
        return None

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(fake_ai, "answer_message", generic_answer)
    monkeypatch.setattr(fake_ai, "nova_help", nova_provider, raising=False)
    application.add_handler(TypeHandler(Update, downstream_handler), group=100)

    await application.process_update(update)

    assert downstream == []
    assert len(sent) == 1
    assert sent[0]["text"] == "✨ Nova\n\nРазбираю вопрос…"
    assert edits
    assert all(edit["message_id"] == canonical_message.message_id for edit in edits)
    result = edits[-1]
    assert "визуализац" in str(result["text"]).casefold()
    assert any(
        button.text == "🎯 Открыть визуализацию"
        and str(button.callback_data).startswith("nova:action:vision:")
        for row in result["reply_markup"].inline_keyboard
        for button in row
    )
    assert fake_ai.route_calls == []
    assert generic_answer_calls == []
    assert nova_provider_calls == []
    current = await core.nova_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat.id,
    )
    assert current is not None
    assert current.last_action_id == "vision"
    async with db.sessions() as session:
        assert len((await session.scalars(select(InboxItem))).all()) == 0
        assert len((await session.scalars(select(DraftInboxItem))).all()) == 0
        assert len((await session.scalars(select(ConversationMessage))).all()) == 0


async def test_real_application_voice_help_runs_nova_before_generic_content(
    db,
    fake_ai,
    monkeypatch,
):
    transcript = "Привет, где у тебя находится визуализация?"
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712352
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        102,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        voice=RuntimeVoice(),
    )
    canonical_message = Message(
        202,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…",
    )
    update = Update(1002, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    generic_route_calls: list[tuple[str, str]] = []
    generic_answer_calls: list[str] = []
    nova_provider_calls: list[str] = []
    original_route = core._route_message

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def tracked_route(route_update, context, text, source):
        generic_route_calls.append((text, source))
        await original_route(route_update, context, text, source)

    async def generic_answer(*args, **kwargs):
        del args, kwargs
        generic_answer_calls.append("answer_message")
        return None

    async def nova_provider(*args, **kwargs):
        del args, kwargs
        nova_provider_calls.append("nova_help")
        return None

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(core, "_route_message", tracked_route)
    monkeypatch.setattr(fake_ai, "answer_message", generic_answer)
    monkeypatch.setattr(fake_ai, "nova_help", nova_provider, raising=False)

    await application.process_update(update)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert len(sent) == 1
    assert sent[0]["text"] == "Расшифровываю голосовую мысль…"
    assert edits
    assert all(edit["message_id"] == canonical_message.message_id for edit in edits)
    result = edits[-1]
    assert "визуализац" in str(result["text"]).casefold()
    assert "Я услышал" not in str(result["text"])
    assert any(
        button.text == "🎯 Открыть визуализацию"
        and str(button.callback_data).startswith("nova:action:vision:")
        for row in result["reply_markup"].inline_keyboard
        for button in row
    )
    assert generic_route_calls == []
    assert fake_ai.route_calls == []
    assert generic_answer_calls == []
    assert nova_provider_calls == []
    current = await core.nova_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=chat.id,
    )
    assert current is not None
    assert current.canonical_message_id == canonical_message.message_id
    assert current.last_action_id == "vision"
    assert transcript not in repr(current)
    async with db.sessions() as session:
        assert len((await session.scalars(select(InboxItem))).all()) == 0
        assert len((await session.scalars(select(DraftInboxItem))).all()) == 0
        assert len((await session.scalars(select(ConversationMessage))).all()) == 0


async def test_real_application_voice_help_does_not_steal_durable_onboarding_answer(
    db,
    fake_ai,
    monkeypatch,
):
    transcript = "Привет, где у тебя находится визуализация?"
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(database_url=db.url),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    telegram_id = 712353
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        session.add(
            OnboardingState(
                user_id=owner.id,
                current_step=2,
                answers={"display_name": "Тест"},
                status="in_progress",
            )
        )

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        103,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        voice=RuntimeVoice(),
    )
    canonical_message = Message(
        203,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="Расшифровываю голосовую мысль…",
    )
    update = Update(1003, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)

    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    generic_route_calls: list[tuple[str, str]] = []
    nova_provider_calls: list[str] = []
    original_route = core._route_message

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def tracked_route(route_update, context, text, source):
        generic_route_calls.append((text, source))
        await original_route(route_update, context, text, source)

    async def nova_provider(*args, **kwargs):
        del args, kwargs
        nova_provider_calls.append("nova_help")
        return None

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(core, "_route_message", tracked_route)
    monkeypatch.setattr(fake_ai, "nova_help", nova_provider, raising=False)

    await application.process_update(update)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert sent
    assert sent[0]["text"] == "Расшифровываю голосовую мысль…"
    assert all("✨ Nova" not in str(item["text"]) for item in sent)
    assert all("✨ Nova" not in str(edit["text"]) for edit in edits)
    assert generic_route_calls == []
    assert fake_ai.route_calls == []
    assert nova_provider_calls == []
    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
    assert state is not None
    assert state.current_step == 3
    assert state.answers["display_name"] == "Тест"
    assert state.answers["future_life"] == transcript
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )


async def test_real_application_explicit_nova_cannot_open_destructive_system_action(
    db, fake_ai, monkeypatch
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    telegram_id = 712350
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    source_message = Message(
        99,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text="Nova, удали все черновики",
    )
    canonical_message = Message(
        199,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="✨ Nova\n\nРазбираю вопрос…",
    )
    update = Update(999, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    canonical_message.set_bot(application.bot)
    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return canonical_message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        return canonical_message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def downstream_handler(update, context):
        del context
        downstream.append(update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    application.add_handler(TypeHandler(Update, downstream_handler), group=100)

    await application.process_update(update)

    assert downstream == []
    assert len(sent) == 1
    assert len(edits) == 1
    assert str(edits[0]["text"]).startswith("✨ Nova")
    snapshot = await core.conversation.get(telegram_id, chat.id)
    assert snapshot.system_pending_action is None
    assert fake_ai.route_calls == []
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is not None
    )


async def test_real_application_stale_nova_callback_preserves_pending_task_input(
    db, fake_ai, monkeypatch
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    telegram_id = 712349
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True
        item = InboxItem(
            user_id=owner.id,
            kind="task",
            title="Позвонить врачу",
            raw_text="Позвонить врачу",
            source="text",
            status="confirmed",
            version=1,
        )
        session.add(item)
        await session.flush()
        await add_task_state(session, item, owner_timezone="Europe/Moscow")
        item_id = item.id

    actions = await core.task_service.issue_actions(
        owner.id,
        telegram_id,
        item_id,
        1,
        ("reminder_edit",),
    )
    result = await core.task_service.start_reminder_input(
        actions["reminder_edit"], owner.id, telegram_id
    )
    assert result.status == "await_reminder"

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
    chat = Chat(telegram_id, "private")
    callback_message = Message(
        199,
        datetime.now(UTC),
        chat,
        from_user=bot_user,
        text="✨ Nova\n\nСтарая подсказка",
    )
    query = CallbackQuery(
        "stale-nova-callback",
        telegram_user,
        "runtime-test",
        message=callback_message,
        data="nova:action:task_create:expired-token",
    )
    update = Update(999, callback_query=query)
    update.set_bot(application.bot)
    callback_message.set_bot(application.bot)
    query.set_bot(application.bot)

    answers: list[dict[str, object]] = []

    async def fake_answer_callback_query(self, callback_query_id, *args, **kwargs):
        del self, args
        answers.append({"callback_query_id": callback_query_id, **kwargs})
        return True

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(ExtBot, "answer_callback_query", fake_answer_callback_query)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)

    await application.process_update(update)

    pending = await core.task_service.pending_input(owner.id, telegram_id)
    assert pending is not None
    assert pending.token == actions["reminder_edit"]
    assert pending.status == "awaiting_input"
    assert len(answers) == 1
    assert answers[0]["callback_query_id"] == query.id


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
async def test_real_application_keeps_non_ui_navigation_verbs_in_content_pipeline(
    db, fake_ai, monkeypatch, caplog, phrase
):
    core = FutureSelfBot(runtime_settings(), db, fake_ai, FakeTranscription())
    application = core.build()
    application._initialized = True
    telegram_id = 712347
    owner = await core._user(telegram_id)
    await AccessService(db).grant_subscriber(telegram_id, source="test")
    async with db.session() as session:
        stored = await session.get(User, owner.id)
        stored.onboarding_completed = True

    sent: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return message

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def downstream_handler(update, context):
        del context
        downstream.append(update.update_id)

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    application.add_handler(TypeHandler(Update, downstream_handler), group=100)

    telegram_user = TelegramUser(telegram_id, False, "Тест")
    chat = Chat(telegram_id, "private")
    message = Message(
        97,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=phrase,
    )
    update = Update(997, message=message)
    update.set_bot(application.bot)
    message.set_bot(application.bot)

    with caplog.at_level(logging.INFO):
        await application.process_update(update)

    response_texts = [str(item.get("text", "")) for item in sent]
    assert downstream == [997]
    assert response_texts
    assert not any(text.startswith("❓ Помощь") for text in response_texts)
    assert fake_ai.route_calls or any("Такого раздела пока нет" in text for text in response_texts)
    assert phrase not in caplog.text
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=chat.id,
        )
        is None
    )


async def test_state_survives_new_repository_and_session(db):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(700, "Europe/Moscow")
        state = await OnboardingRepository(session).get_or_create(user.id)
        state.current_step = 4
        state.answers = {"display_name": "Лена", "future_life": "Спокойная жизнь"}
        user_id = user.id
    async with db.sessions() as session:
        restored = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == user_id)
        )
    assert restored.current_step == 4
    assert restored.answers["display_name"] == "Лена"


async def test_sqlite_runtime_enables_foreign_key_enforcement(db):
    async with db.sessions() as session:
        assert await session.scalar(text("PRAGMA foreign_keys")) == 1


class FakeCallbackQuery:
    def __init__(self, data: str):
        self.data = data
        self.answers: list[tuple[str | None, bool]] = []
        self.edited: list[str] = []
        self.markup_removed = 0
        self.message = SimpleNamespace()

    async def answer(self, text: str | None = None, show_alert: bool = False):
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs):
        self.edited.append(text)

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_removed += 1


async def test_legacy_inbox_callback_is_rejected_without_save(db, fake_ai):
    bot = FutureSelfBot(runtime_settings(), db, fake_ai, FakeTranscription())
    query = FakeCallbackQuery("inbox:save")
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=123),
        effective_chat=SimpleNamespace(id=456),
    )
    await bot.inbox_action(update, SimpleNamespace(user_data={}))
    async with db.sessions() as session:
        count = len((await session.scalars(select(InboxItem))).all())
    assert count == 0
    assert query.answers[-1] == ("Эта карточка уже неактуальна. Создай новую.", True)


def test_safe_error_logging_omits_exception_message(caplog):
    secret = "secret-token-and-private-voice-text"
    with caplog.at_level(logging.ERROR):
        log_safe_failure("Voice processing failed", RuntimeError(secret), user_id=42)
    assert "Voice processing failed" in caplog.text
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ("transcript", "telegram_id", "consumed"),
    [
        ("Я хочу жить у моря и больше путешествовать", 713_001, True),
        ("Nova, запомни: PRIVATE_ONBOARDING_MEMORY_SENTINEL", 713_002, False),
    ],
    ids=("ordinary", "explicit_memory"),
)
async def test_real_application_persisted_onboarding_owns_active_memory_voice(
    db,
    fake_ai,
    monkeypatch,
    transcript,
    telegram_id,
    consumed,
):
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=False,
    )
    async with db.session() as session:
        session.add(
            OnboardingState(
                user_id=owner.id,
                current_step=2,
                answers={"display_name": "Тест"},
                status="in_progress",
            )
        )
    memory = await _runtime_active_memory(core, owner, telegram_id, telegram_id)
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=20_001 + telegram_id,
        source_message_id=301,
        progress_message_id=401,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    downstream = _patch_runtime_voice_priority_spies(core, fake_ai, monkeypatch)

    await application.process_update(update)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert downstream == []
    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
    assert state is not None
    assert state.current_step == (3 if consumed else 2)
    assert (state.answers.get("future_life") == transcript) is consumed
    assert (
        await core.nova_memory_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    assert memory.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
    await _assert_runtime_voice_memory_untouched(db)
    _assert_runtime_voice_canonical(sent, edits, progress.message_id)


@pytest.mark.parametrize(
    ("transcript", "telegram_id", "consumed"),
    [
        ("16", 713_003, True),
        ("Nova, запомни: PRIVATE_DATE_MEMORY_SENTINEL", 713_004, False),
    ],
    ids=("ordinary", "explicit_memory"),
)
async def test_real_application_pending_date_choice_owns_active_memory_voice(
    db,
    fake_ai,
    monkeypatch,
    transcript,
    telegram_id,
    consumed,
):
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    await core.conversation.set_date_conflict(
        telegram_id,
        telegram_id,
        [
            {"value": "2026-08-16", "weekday": "воскресенье"},
            {"value": "2026-08-23", "weekday": "воскресенье"},
        ],
    )
    await core.conversation.append(
        telegram_id,
        telegram_id,
        role="user",
        content="Напомни 16 августа позвонить врачу",
        source="text",
        intent="date_conflict",
    )
    memory = await _runtime_active_memory(core, owner, telegram_id, telegram_id)
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=20_001 + telegram_id,
        source_message_id=302,
        progress_message_id=402,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    downstream = _patch_runtime_voice_priority_spies(core, fake_ai, monkeypatch)
    original_route = FutureSelfBot._route_message.__get__(core, FutureSelfBot)

    async def date_choice_route(route_update, context, text, source):
        return await original_route(route_update, context, text, source)

    if consumed:
        monkeypatch.setattr(core, "_route_message", date_choice_route)

    await application.process_update(update)

    assert downstream == []
    snapshot = await core.conversation.get(telegram_id, telegram_id)
    drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
    assert bool(drafts) is consumed
    assert bool(snapshot.pending_date_options) is not consumed
    assert (
        await core.nova_memory_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    assert memory.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
    await _assert_runtime_voice_memory_untouched(db)
    _assert_runtime_voice_canonical(sent, edits, progress.message_id)


async def test_real_application_pending_system_confirmation_owns_active_memory_voice(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 713_005
    transcription = RuntimeTranscription("да, удалить")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    draft = await core.draft_service.create(
        user_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
        source="text",
        raw_text="Удаляемый черновик",
        parsed=ParsedThought(kind="note", title="Удаляемый черновик"),
    )
    await core.conversation.begin_system_action(
        telegram_id,
        telegram_id,
        "discard_all_active_drafts",
        [
            {
                "id": draft.id,
                "version": draft.version,
                "affected": True,
                "preview_message_id": None,
            }
        ],
    )
    await _runtime_active_memory(core, owner, telegram_id, telegram_id)
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=20_001 + telegram_id,
        source_message_id=303,
        progress_message_id=403,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    downstream = _patch_runtime_voice_priority_spies(core, fake_ai, monkeypatch)

    await application.process_update(update)

    assert downstream == []
    assert (await core.draft_service.get(draft.id)).status == "discarded"
    assert (await core.conversation.get(telegram_id, telegram_id)).system_pending_action is None
    assert (
        await core.nova_memory_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    await _assert_runtime_voice_memory_untouched(db)
    _assert_runtime_voice_canonical(sent, edits, progress.message_id)


@pytest.mark.parametrize(
    ("flow", "telegram_id", "transcript"),
    [
        ("workspace", 713_006, "Наш дом"),
        ("collection", 713_007, "Путешествия"),
        ("task", 713_008, "завтра в 18:00"),
        ("vision", 713_009, "Побывать у океана"),
    ],
)
async def test_real_application_durable_business_flow_owns_active_memory_voice(
    db,
    fake_ai,
    monkeypatch,
    flow,
    telegram_id,
    transcript,
):
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_workspace_access=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    item_id = None
    if flow == "workspace":
        await core.workspace_service.begin_input(
            owner.id,
            telegram_id,
            "create_name",
            payload={"character": "family"},
        )
    elif flow == "collection":
        await core.collection_service.issue_action(
            owner.id,
            telegram_id,
            "input_create",
            payload={"kind": "topic"},
            status="awaiting_input",
        )
    elif flow == "task":
        async with db.session() as session:
            item = InboxItem(
                user_id=owner.id,
                kind="task",
                title="Позвонить врачу",
                raw_text="Позвонить врачу",
                source="text",
                status="confirmed",
                version=1,
            )
            session.add(item)
            await session.flush()
            await add_task_state(session, item, owner_timezone="Europe/Moscow")
            item_id = item.id
        actions = await core.task_service.issue_actions(
            owner.id,
            telegram_id,
            item_id,
            1,
            ("reminder_edit",),
        )
        started = await core.task_service.start_reminder_input(
            actions["reminder_edit"],
            owner.id,
            telegram_id,
        )
        assert started.status == "await_reminder"
    else:
        draft = await core.vision_service.begin(owner.id, telegram_id)
        selected = await core.vision_service.choose_category(
            owner.id,
            telegram_id,
            "other",
            draft_id=draft.id,
        )
        assert selected.status == "advanced"
    await _runtime_active_memory(core, owner, telegram_id, telegram_id)
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=20_001 + telegram_id,
        source_message_id=304,
        progress_message_id=404,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    downstream = _patch_runtime_voice_priority_spies(core, fake_ai, monkeypatch)

    await application.process_update(update)

    assert downstream == []
    if flow == "workspace":
        pending = await core.workspace_service.pending_input(owner.id, telegram_id)
        assert pending is not None
        assert pending.action == "input:create_description"
        assert pending.payload["name"] == transcript
    elif flow == "collection":
        assert await core.collection_service.pending_input(owner.id, telegram_id) is None
        assert (await core.collection_service.resolve(owner.id, transcript)).match is not None
    elif flow == "task":
        assert await core.task_service.pending_input(owner.id, telegram_id) is None
        record = await core.task_service.record(owner.id, item_id)
        assert record is not None and record.reminder is not None
    else:
        draft = await core.vision_service.draft(owner.id, telegram_id)
        assert draft is not None
        assert draft.wish_text == transcript
        assert draft.step == "why"
    assert (
        await core.nova_memory_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    await _assert_runtime_voice_memory_untouched(db)
    _assert_runtime_voice_canonical(sent, edits, progress.message_id)


async def test_real_application_voice_durable_owner_clears_only_frozen_memory_generation(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 713_010
    transcription = RuntimeTranscription("Nova, запомни: PRIVATE_REPLACEMENT_SENTINEL")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    await core.conversation.set_date_conflict(
        telegram_id,
        telegram_id,
        [{"value": "2026-08-16", "weekday": "воскресенье"}],
    )
    frozen = await _runtime_active_memory(core, owner, telegram_id, telegram_id)
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=20_001 + telegram_id,
        source_message_id=305,
        progress_message_id=405,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    downstream = _patch_runtime_voice_priority_spies(core, fake_ai, monkeypatch)
    original_active_flow = core._active_navigation_flow
    replacement = None

    async def replace_during_ownership_check(route_update, context):
        nonlocal replacement
        active = await original_active_flow(route_update, context)
        if replacement is None:
            replacement = await core.nova_memory_sessions.create(
                owner_id=owner.id,
                telegram_user_id=telegram_id,
                chat_id=telegram_id,
                tier=owner.access_tier,
                access_version=owner.access_version,
                canonical_message_id=9_002,
                phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
            )
        return active

    monkeypatch.setattr(core, "_active_navigation_flow", replace_during_ownership_check)

    await application.process_update(update)

    assert downstream == []
    current = await core.nova_memory_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert replacement is not None
    assert current == replacement
    assert current.id != frozen.id
    assert current.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
    assert (await core.conversation.get(telegram_id, telegram_id)).pending_date_options
    await _assert_runtime_voice_memory_untouched(db)
    assert [entry["text"] for entry in sent] == ["Расшифровываю голосовую мысль…"]
    assert all(entry["message_id"] == progress.message_id for entry in edits)


async def test_real_application_voice_bound_clear_preserves_replacement_generation(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 713_011
    transcription = RuntimeTranscription("Nova, запомни: PRIVATE_BOUND_CLEAR_SENTINEL")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    await core.conversation.set_date_conflict(
        telegram_id,
        telegram_id,
        [{"value": "2026-08-16", "weekday": "воскресенье"}],
    )
    frozen = await _runtime_active_memory(core, owner, telegram_id, telegram_id)
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=20_001 + telegram_id,
        source_message_id=306,
        progress_message_id=406,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    downstream = _patch_runtime_voice_priority_spies(core, fake_ai, monkeypatch)
    original_clear_bound = core.nova_memory_clear_bound
    replacement = None
    clear_session_ids: list[str | None] = []

    async def replace_during_bound_clear(
        owner_id,
        telegram_user_id,
        chat_id,
        *,
        session_id=None,
    ):
        nonlocal replacement
        clear_session_ids.append(session_id)
        replacement = await core.nova_memory_sessions.create(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
            tier=owner.access_tier,
            access_version=owner.access_version,
            canonical_message_id=9_003,
            phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
        )
        return await original_clear_bound(
            owner_id,
            telegram_user_id,
            chat_id,
            session_id=session_id,
        )

    monkeypatch.setattr(core, "nova_memory_clear_bound", replace_during_bound_clear)

    await application.process_update(update)

    assert downstream == []
    assert clear_session_ids == [frozen.id]
    current = await core.nova_memory_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert replacement is not None
    assert current == replacement
    assert current.id != frozen.id
    assert current.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
    assert (await core.conversation.get(telegram_id, telegram_id)).pending_date_options
    await _assert_runtime_voice_memory_untouched(db)
    _assert_runtime_voice_canonical(sent, edits, progress.message_id)


async def test_stage7c_flags_off_preserve_legacy_route_and_skip_memory_reads(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_001
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=False,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id)
    reply = _runtime_bot_message(application, telegram_id, 8_001)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    async def forbidden_read(**kwargs):
        del kwargs
        raise AssertionError("disabled application must not read Nova memory")

    monkeypatch.setattr(core.nova_memory_service, "application_snapshot", forbidden_read)
    monkeypatch.setattr(core.nova_memory_service, "application_current_check", forbidden_read)
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_001,
        source_message_id=7_001,
    )

    await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert fake_ai.answer_calls == []
    assert [entry["text"] for entry in sent] == ["Привет!"]
    async with db.sessions() as session:
        assistant = await session.scalar(
            select(ConversationMessage).where(ConversationMessage.role == "assistant")
        )
    assert assistant is not None
    assert assistant.intent == "answer"
    async with db.sessions() as session:
        messages = list(
            (
                await session.scalars(select(ConversationMessage).order_by(ConversationMessage.id))
            ).all()
        )
    assert [(item.role, item.content) for item in messages] == [
        ("user", "привет"),
        ("assistant", "Привет!"),
    ]
    assert [item.intent for item in messages] == ["conversation", "answer"]


async def test_stage7c_non_conversation_route_never_reads_or_answers_with_memory(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_002
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id)
    reply = _runtime_bot_message(application, telegram_id, 8_002)
    _patch_runtime_stage7c_transport(monkeypatch, [reply])

    async def forbidden_read(**kwargs):
        del kwargs
        raise AssertionError("non-conversation intent must not read Nova memory")

    monkeypatch.setattr(core.nova_memory_service, "application_snapshot", forbidden_read)
    monkeypatch.setattr(core.nova_memory_service, "application_current_check", forbidden_read)
    update = _runtime_text_update(
        application,
        telegram_id,
        "идея о спокойном путешествии",
        update_id=24_002,
        source_message_id=7_002,
    )

    await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert fake_ai.answer_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 1


@pytest.mark.parametrize(
    ("tier", "crud_admin_only", "application_admin_only"),
    [
        ("subscriber", False, False),
        ("admin", True, True),
    ],
)
async def test_stage7c_ready_memory_is_answer_only_and_uses_one_extra_provider_call(
    db,
    fake_ai,
    monkeypatch,
    tier,
    crud_admin_only,
    application_admin_only,
):
    telegram_id = 714_010 if tier == "subscriber" else 714_011
    private = f"PRIVATE_STAGE7C_{tier.upper()}_MEMORY"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=crud_admin_only,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=application_admin_only,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier=tier)
    await _runtime_stage7c_memory(core, owner, private)
    reply = _runtime_bot_message(application, telegram_id, 8_010)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])
    snapshot_calls = 0
    current_calls = 0
    original_snapshot = core.nova_memory_service.application_snapshot
    original_current = core.nova_memory_service.application_current_check

    async def counted_snapshot(**kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return await original_snapshot(**kwargs)

    async def counted_current(**kwargs):
        nonlocal current_calls
        current_calls += 1
        return await original_current(**kwargs)

    monkeypatch.setattr(core.nova_memory_service, "application_snapshot", counted_snapshot)
    monkeypatch.setattr(core.nova_memory_service, "application_current_check", counted_current)
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_010 + telegram_id,
        source_message_id=7_010,
    )

    await application.process_update(update)

    assert snapshot_calls == 1
    assert current_calls == 4
    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert len(fake_ai.answer_confirmed_memory_calls) == 1
    projection = fake_ai.answer_confirmed_memory_calls[0]
    assert projection is not None
    assert [record.content for record in projection.records] == [private]
    assert private not in repr(fake_ai.conversation_contexts)
    assert all("confirmed_memory" not in context for context in fake_ai.conversation_contexts)
    assert [entry["text"] for entry in sent] == ["Ответ на: привет"]


async def test_stage7c_subscriber_is_excluded_by_admin_only_policy_without_memory_reads(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_012
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=True,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    reply = _runtime_bot_message(application, telegram_id, 8_012)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    async def forbidden_read(**kwargs):
        del kwargs
        raise AssertionError("tier-excluded application must not read Nova memory")

    monkeypatch.setattr(core.nova_memory_service, "application_snapshot", forbidden_read)
    monkeypatch.setattr(core.nova_memory_service, "application_current_check", forbidden_read)
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_012,
        source_message_id=7_012,
    )

    await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert fake_ai.answer_calls == []
    assert [entry["text"] for entry in sent] == ["Привет!"]
    async with db.sessions() as session:
        assistant = await session.scalar(
            select(ConversationMessage).where(ConversationMessage.role == "assistant")
        )
    assert assistant is not None
    assert assistant.intent == "answer"


async def test_stage7c_access_gate_generation_bounce_stops_before_memory_snapshot(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_013
    private = "PRIVATE_STAGE7C_ACCESS_GATE_BOUNCE"
    route_answer = "ROUTE_ANSWER_MUST_NOT_BE_DELIVERED_AFTER_ACCESS_BOUNCE"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private)
    reply = _runtime_bot_message(application, telegram_id, 8_013)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="conversation", confidence=0.99, answer=route_answer)

    monkeypatch.setattr(fake_ai, "route_message", conversation_route)
    original_user = core._user
    access_gate_read = asyncio.Event()
    access_gate_release = asyncio.Event()
    user_reads = 0

    async def block_first_user_read(actor_id):
        nonlocal user_reads
        user = await original_user(actor_id)
        user_reads += 1
        if user_reads == 1:
            access_gate_read.set()
            await access_gate_release.wait()
        return user

    snapshot_calls = 0
    original_snapshot = core.nova_memory_service.application_snapshot

    async def counted_snapshot(**kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return await original_snapshot(**kwargs)

    monkeypatch.setattr(core, "_user", block_first_user_read)
    monkeypatch.setattr(
        core.nova_memory_service,
        "application_snapshot",
        counted_snapshot,
    )
    update = _runtime_text_update(
        application,
        telegram_id,
        "stage7c access generation question",
        update_id=24_013,
        source_message_id=7_013,
    )

    processing = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(access_gate_read.wait(), timeout=2)
    access = AccessService(db)
    await access.set_guest(telegram_id, source="stage7c-access-gate-bounce")
    await access.grant_subscriber(telegram_id, source="stage7c-access-gate-bounce")
    access_gate_release.set()
    await asyncio.wait_for(processing, timeout=2)

    status = await access.status(telegram_id)
    assert status is not None
    assert status.access_tier == "subscriber"
    assert status.access_version == owner.access_version + 2
    assert user_reads >= 2
    assert snapshot_calls == 0
    assert fake_ai.answer_calls == []
    assert fake_ai.answer_confirmed_memory_calls == []
    assert all(entry.get("text") != route_answer for entry in sent)
    assert private not in repr(sent)
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []


@pytest.mark.parametrize(
    ("source", "checkpoint", "race"),
    [
        ("text", checkpoint, race)
        for checkpoint in (
            "before_snapshot",
            "after_materialization",
            "pre_provider",
            "pre_telegram",
            "post_io",
        )
        for race in (
            "downgrade",
            "bounce",
            "create",
            "edit",
            "toggle",
            "delete",
            "delete_all",
        )
    ]
    + [
        ("voice", "after_materialization", "downgrade"),
        ("voice", "pre_telegram", "bounce"),
        ("voice", "post_io", "create"),
    ],
)
async def test_stage7c_named_race_checkpoint_matrix_is_fenced(
    db,
    fake_ai,
    monkeypatch,
    source,
    checkpoint,
    race,
):
    telegram_id = 714_014
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        RuntimeTranscription("привет") if source == "voice" else FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    initial = await _runtime_stage7c_memory(
        core,
        owner,
        f"PRIVATE_STAGE7C_{checkpoint.upper()}_{race.upper()}",
    )
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    answer = _runtime_bot_message(application, telegram_id, 8_014)
    if source == "voice":
        update, progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=24_014,
            source_message_id=7_014,
            progress_message_id=8_013,
        )
        sent, _, deletes = _patch_runtime_stage7c_transport(
            monkeypatch,
            [progress, answer],
        )
    else:
        update = _runtime_text_update(
            application,
            telegram_id,
            "привет",
            update_id=24_014,
            source_message_id=7_014,
        )
        sent, _, deletes = _patch_runtime_stage7c_transport(monkeypatch, [answer])
    checkpoint_started = asyncio.Event()
    checkpoint_release = asyncio.Event()
    current_calls = 0
    original_snapshot = core.nova_memory_service.application_snapshot
    original_application_hook = core.nova_memory_service._before_application_generation_check
    original_current = core.nova_memory_service.application_current_check

    if checkpoint == "before_snapshot":

        async def blocked_snapshot(**kwargs):
            checkpoint_started.set()
            await checkpoint_release.wait()
            return await original_snapshot(**kwargs)

        monkeypatch.setattr(
            core.nova_memory_service,
            "application_snapshot",
            blocked_snapshot,
        )
    elif checkpoint == "after_materialization":

        async def blocked_application_hook(*args, **kwargs):
            checkpoint_started.set()
            await checkpoint_release.wait()
            await original_application_hook(*args, **kwargs)

        monkeypatch.setattr(
            core.nova_memory_service,
            "_before_application_generation_check",
            blocked_application_hook,
        )
    else:
        target_call = {"pre_provider": 1, "pre_telegram": 3, "post_io": 4}[checkpoint]

        async def blocked_current(**kwargs):
            nonlocal current_calls
            current_calls += 1
            if current_calls == target_call:
                checkpoint_started.set()
                await checkpoint_release.wait()
            return await original_current(**kwargs)

        monkeypatch.setattr(
            core.nova_memory_service,
            "application_current_check",
            blocked_current,
        )

    processing = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(checkpoint_started.wait(), timeout=2)

    if race == "downgrade":
        await AccessService(db).set_guest(telegram_id, source="stage7c-checkpoint")
    elif race == "bounce":
        access = AccessService(db)
        await access.set_guest(telegram_id, source="stage7c-checkpoint")
        await access.grant_subscriber(telegram_id, source="stage7c-checkpoint")
    else:
        await _runtime_stage7c_mutate_memory(
            core,
            owner,
            race,
            initial,
            suffix=f"{checkpoint.upper()}_{race.upper()}",
        )
    checkpoint_release.set()
    await asyncio.wait_for(processing, timeout=2)

    is_access_race = race in {"downgrade", "bounce"}
    is_pre_snapshot_mutation = checkpoint == "before_snapshot" and not is_access_race
    expected_provider_calls = 0
    if checkpoint in {"pre_telegram", "post_io"}:
        expected_provider_calls = 1
    elif is_pre_snapshot_mutation and race not in {"delete", "delete_all"}:
        expected_provider_calls = 1
    assert len(fake_ai.answer_calls) == expected_provider_calls
    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) <= 1
    answer_sends = sent[1:] if source == "voice" else sent
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert audit_after - audit_before == (0 if race in {"downgrade", "bounce"} else 1)
    if is_pre_snapshot_mutation:
        if expected_provider_calls:
            projection = fake_ai.answer_confirmed_memory_calls[0]
            assert projection is not None
            if race == "create":
                assert any("BEFORE_SNAPSHOT_CREATE" in item.content for item in projection.records)
            elif race == "edit":
                assert any("BEFORE_SNAPSHOT_EDIT" in item.content for item in projection.records)
            elif race == "toggle":
                assert any(item.important for item in projection.records)
        else:
            assert fake_ai.answer_confirmed_memory_calls == []
        assert len(assistant_messages) == 1
        assert deletes == []
    elif checkpoint == "post_io":
        assert len(answer_sends) == 1
        assert deletes == [{"chat_id": telegram_id, "message_id": answer.message_id}]
        assert assistant_messages == []
    else:
        assert all("РћС‚РІРµС‚ РЅР°:" not in str(entry.get("text")) for entry in answer_sends)
        assert deletes == []
        assert assistant_messages == []


async def test_stage7c_empty_memory_reuses_embedded_route_answer_without_answer_call(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_020
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id)
    reply = _runtime_bot_message(application, telegram_id, 8_020)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_020,
        source_message_id=7_020,
    )

    await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert fake_ai.answer_calls == []
    assert [entry["text"] for entry in sent] == ["Привет!"]
    async with db.sessions() as session:
        assistant = await session.scalar(
            select(ConversationMessage).where(ConversationMessage.role == "assistant")
        )
    assert assistant is not None
    assert assistant.intent == "answer"


async def test_stage7c_empty_memory_calls_normal_answer_once_when_route_has_no_answer(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_021
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id)
    reply = _runtime_bot_message(application, telegram_id, 8_021)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    async def route_without_answer(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="question", confidence=0.99, answer=None)

    monkeypatch.setattr(fake_ai, "route_message", route_without_answer)
    update = _runtime_text_update(
        application,
        telegram_id,
        "что важно сегодня?",
        update_id=24_021,
        source_message_id=7_021,
    )

    await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert fake_ai.answer_confirmed_memory_calls == [None]
    assert [entry["text"] for entry in sent] == ["Ответ на: что важно сегодня?"]
    async with db.sessions() as session:
        assistant = await session.scalar(
            select(ConversationMessage).where(ConversationMessage.role == "assistant")
        )
    assert assistant is not None
    assert assistant.intent == "answer"


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_stage7c_text_and_stt_share_ready_memory_answer_lifecycle(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 714_030 if source == "text" else 714_031
    private = f"PRIVATE_STAGE7C_{source.upper()}_PARITY"
    transcription = RuntimeTranscription("привет")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private)

    if source == "voice":
        update, progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=24_031,
            source_message_id=7_031,
            progress_message_id=8_031,
        )
        reply = _runtime_bot_message(application, telegram_id, 8_032)
        sent, edits, _ = _patch_runtime_stage7c_transport(
            monkeypatch,
            [progress, reply],
        )
    else:
        update = _runtime_text_update(
            application,
            telegram_id,
            "привет",
            update_id=24_030,
            source_message_id=7_030,
        )
        reply = _runtime_bot_message(application, telegram_id, 8_030)
        sent, edits, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    projection = fake_ai.answer_confirmed_memory_calls[0]
    assert projection is not None
    assert [record.content for record in projection.records] == [private]
    if source == "voice":
        assert [entry["text"] for entry in sent] == [
            "Расшифровываю голосовую мысль…",
            "Ответ на: привет",
        ]
        assert edits and str(edits[-1]["text"]).startswith("Я услышал:")
    else:
        assert [entry["text"] for entry in sent] == ["Ответ на: привет"]
    async with db.sessions() as session:
        messages = list(
            (
                await session.scalars(select(ConversationMessage).order_by(ConversationMessage.id))
            ).all()
        )
    assert [item.role for item in messages] == ["user", "assistant"]
    assert messages[0].source == source
    assert messages[1].intent == "memory_answer"
    assert private not in repr([(item.role, item.content) for item in messages])


@pytest.mark.parametrize(
    ("race", "expected_neutral"),
    [
        ("access", "Доступ изменился"),
        ("bounce", "Доступ изменился"),
        ("create", "Память Nova изменилась"),
        ("edit", "Память Nova изменилась"),
        ("toggle", "Память Nova изменилась"),
        ("delete", "Память Nova изменилась"),
        ("delete_all", "Память Nova изменилась"),
    ],
)
async def test_stage7c_provider_race_discards_answer_without_delivery_or_retry(
    db,
    fake_ai,
    monkeypatch,
    race,
    expected_neutral,
):
    telegram_id = {
        "access": 714_040,
        "bounce": 714_041,
        "create": 714_042,
        "edit": 714_043,
        "toggle": 714_044,
        "delete": 714_045,
        "delete_all": 714_046,
    }[race]
    private = f"PRIVATE_STAGE7C_PROVIDER_{race.upper()}"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    initial = await _runtime_stage7c_memory(core, owner, private)
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    neutral = _runtime_bot_message(application, telegram_id, 8_040)
    sent, _, deletes = _patch_runtime_stage7c_transport(monkeypatch, [neutral])
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()
    original_answer = fake_ai.answer_message

    async def blocked_answer(*args, **kwargs):
        provider_started.set()
        await provider_release.wait()
        return await original_answer(*args, **kwargs)

    monkeypatch.setattr(fake_ai, "answer_message", blocked_answer)
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_040 + telegram_id,
        source_message_id=7_040,
    )
    task = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(provider_started.wait(), timeout=2)
    if race == "access":
        await AccessService(db).set_guest(telegram_id, source="stage7c-provider-race")
    elif race == "bounce":
        access = AccessService(db)
        await access.set_guest(telegram_id, source="stage7c-provider-bounce")
        await access.grant_subscriber(telegram_id, source="stage7c-provider-bounce")
    else:
        await _runtime_stage7c_mutate_memory(
            core,
            owner,
            race,
            initial,
            suffix=f"PROVIDER_{race.upper()}",
        )
    provider_release.set()
    await asyncio.wait_for(task, timeout=2)

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert len(sent) == 1
    assert expected_neutral in str(sent[0]["text"])
    assert "Ответ на:" not in str(sent[0]["text"])
    assert deletes == []
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after - audit_before == (0 if race in {"access", "bounce"} else 1)
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []


@pytest.mark.parametrize(
    ("race", "expected_neutral"),
    [
        ("access", "Доступ изменился"),
        ("bounce", "Доступ изменился"),
        ("create", "Память Nova изменилась"),
        ("edit", "Память Nova изменилась"),
        ("toggle", "Память Nova изменилась"),
        ("delete", "Память Nova изменилась"),
        ("delete_all", "Память Nova изменилась"),
    ],
)
async def test_stage7c_telegram_send_race_compensates_exact_answer_and_skips_context(
    db,
    fake_ai,
    monkeypatch,
    race,
    expected_neutral,
):
    telegram_id = {
        "access": 714_050,
        "bounce": 714_051,
        "create": 714_052,
        "edit": 714_053,
        "toggle": 714_054,
        "delete": 714_055,
        "delete_all": 714_056,
    }[race]
    private = f"PRIVATE_STAGE7C_SEND_{race.upper()}"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    initial = await _runtime_stage7c_memory(core, owner, private)
    async with db.sessions() as session:
        audit_before = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    answer = _runtime_bot_message(application, telegram_id, 8_050)
    send_started = asyncio.Event()
    send_release = asyncio.Event()
    sent, edits, deletes = _patch_runtime_stage7c_transport(
        monkeypatch,
        [answer],
        send_started=send_started,
        send_release=send_release,
    )
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_050 + telegram_id,
        source_message_id=7_050,
    )
    task = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(send_started.wait(), timeout=2)
    if race == "access":
        await AccessService(db).set_guest(telegram_id, source="stage7c-send-race")
    elif race == "bounce":
        access = AccessService(db)
        await access.set_guest(telegram_id, source="stage7c-send-bounce")
        await access.grant_subscriber(telegram_id, source="stage7c-send-bounce")
    else:
        await _runtime_stage7c_mutate_memory(
            core,
            owner,
            race,
            initial,
            suffix=f"SEND_{race.upper()}",
        )
    send_release.set()
    await asyncio.wait_for(task, timeout=2)

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert [entry["text"] for entry in sent] == ["Ответ на: привет"]
    assert deletes == [{"chat_id": telegram_id, "message_id": answer.message_id}]
    assert edits == []
    assert expected_neutral not in repr(sent)
    async with db.sessions() as session:
        audit_after = int(await session.scalar(select(func.count(NovaMemoryChange.id))) or 0)
    assert audit_after - audit_before == (0 if race in {"access", "bounce"} else 1)
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []


async def test_stage7c_post_send_external_cancel_keeps_shielded_fence_running(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_060
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, "PRIVATE_STAGE7C_SHIELD_MEMORY")
    answer = _runtime_bot_message(application, telegram_id, 8_060)
    _, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [answer])
    original_current = core.nova_memory_service.application_current_check
    post_check_started = asyncio.Event()
    post_check_release = asyncio.Event()
    check_calls = 0

    async def block_post_check(**kwargs):
        nonlocal check_calls
        check_calls += 1
        if check_calls == 4:
            post_check_started.set()
            await post_check_release.wait()
        return await original_current(**kwargs)

    monkeypatch.setattr(
        core.nova_memory_service,
        "application_current_check",
        block_post_check,
    )
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_060,
        source_message_id=7_060,
    )
    handler = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(post_check_started.wait(), timeout=2)
    await AccessService(db).set_guest(telegram_id, source="stage7c-shield-race")
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    assert core._nova_memory_application_tasks

    post_check_release.set()
    await asyncio.wait_for(
        asyncio.gather(*tuple(core._nova_memory_application_tasks)),
        timeout=2,
    )

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert deletes == [{"chat_id": telegram_id, "message_id": answer.message_id}]
    assert edits == []
    assert core._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []


async def test_stage7c_direct_telegram_cancel_starts_no_post_send_fence(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 714_061
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, "PRIVATE_STAGE7C_DIRECT_CANCEL_MEMORY")
    current_calls = 0
    cleanup_calls: list[str] = []
    original_current = core.nova_memory_service.application_current_check

    async def counted_current(**kwargs):
        nonlocal current_calls
        current_calls += 1
        return await original_current(**kwargs)

    async def cancelled_send(self, *args, **kwargs):
        del self, args, kwargs
        raise asyncio.CancelledError

    async def forbidden_delete(self, *args, **kwargs):
        del self, args, kwargs
        cleanup_calls.append("delete")
        raise AssertionError("primary Telegram cancellation must not start cleanup")

    async def forbidden_edit(self, *args, **kwargs):
        del self, args, kwargs
        cleanup_calls.append("edit")
        raise AssertionError("primary Telegram cancellation must not start cleanup")

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(
        core.nova_memory_service,
        "application_current_check",
        counted_current,
    )
    monkeypatch.setattr(ExtBot, "send_message", cancelled_send)
    monkeypatch.setattr(ExtBot, "delete_message", forbidden_delete)
    monkeypatch.setattr(ExtBot, "edit_message_text", forbidden_edit)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_061,
        source_message_id=7_061,
    )

    with pytest.raises(asyncio.CancelledError):
        await application.process_update(update)

    assert current_calls == 3
    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert cleanup_calls == []
    assert core._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []


async def test_stage7c_primary_telegram_error_starts_no_post_send_fence(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    telegram_id = 714_062
    private_memory = "PRIVATE_STAGE7C_TELEGRAM_ERROR_MEMORY"
    private_error = "PRIVATE_STAGE7C_TELEGRAM_ERROR_DETAIL"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    current_calls = 0
    cleanup_calls: list[str] = []
    original_current = core.nova_memory_service.application_current_check

    async def counted_current(**kwargs):
        nonlocal current_calls
        current_calls += 1
        return await original_current(**kwargs)

    async def failed_send(self, *args, **kwargs):
        del self, args, kwargs
        raise TelegramError(private_error)

    async def forbidden_cleanup(self, *args, **kwargs):
        del self, args, kwargs
        cleanup_calls.append("unexpected")
        raise AssertionError("failed primary Telegram send must not start cleanup")

    monkeypatch.setattr(
        core.nova_memory_service,
        "application_current_check",
        counted_current,
    )
    monkeypatch.setattr(ExtBot, "send_message", failed_send)
    monkeypatch.setattr(ExtBot, "delete_message", forbidden_cleanup)
    monkeypatch.setattr(ExtBot, "edit_message_text", forbidden_cleanup)
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_062,
        source_message_id=7_062,
    )

    with caplog.at_level(logging.WARNING):
        await application.process_update(update)

    assert current_calls == 3
    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert cleanup_calls == []
    assert core._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []
    application_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Nova memory application delivery failed")
    ]
    assert application_logs == [
        "Nova memory application delivery failed stage=telegram_send error_type=TelegramError"
    ]
    for private_value in (private_memory, private_error, str(telegram_id)):
        assert private_value not in repr(application_logs)


@pytest.mark.parametrize(
    "delete_error",
    [
        RuntimeError("PRIVATE_STAGE7C_DELETE_FAILURE"),
        None,
    ],
    ids=["failure", "false-result"],
)
async def test_stage7c_post_send_delete_failure_or_false_result_edits_exact_message(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    delete_error,
):
    telegram_id = 714_070
    private_memory = "PRIVATE_STAGE7C_DELETE_MEMORY"
    private_question = "PRIVATE_STAGE7C_DELETE_QUESTION"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    answer = _runtime_bot_message(application, telegram_id, 8_070)
    sent, edits, deletes = _patch_runtime_stage7c_cleanup_transport(
        monkeypatch,
        answer,
        delete_error=delete_error,
    )

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="conversation", confidence=0.99, answer=None)

    original_current = core.nova_memory_service.application_current_check

    async def changed_after_send(**kwargs):
        if sent:
            return NovaMemoryApplicationCurrent("memory_changed")
        return await original_current(**kwargs)

    monkeypatch.setattr(fake_ai, "route_message", conversation_route)
    monkeypatch.setattr(
        core.nova_memory_service,
        "application_current_check",
        changed_after_send,
    )
    update = _runtime_text_update(
        application,
        telegram_id,
        private_question,
        update_id=24_070,
        source_message_id=7_070,
    )

    with caplog.at_level(logging.WARNING):
        await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert len(sent) == 1
    assert deletes == [{"chat_id": telegram_id, "message_id": answer.message_id}]
    assert edits == [
        {
            "chat_id": telegram_id,
            "message_id": answer.message_id,
            "text": NOVA_MEMORY_APPLICATION_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    assert core._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []
    if delete_error is None:
        assert "operation=delete" not in caplog.text
    else:
        assert "operation=delete" in caplog.text
        assert f"error_type={type(delete_error).__name__}" in caplog.text
    for private_value in (
        private_memory,
        private_question,
        str(telegram_id),
        str(delete_error),
    ):
        assert private_value not in caplog.text


async def test_stage7c_shutdown_cancel_during_blocked_compensation_delete_stops_before_edit(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    telegram_id = 714_072
    private_memory = "PRIVATE_STAGE7C_BLOCKED_DELETE_MEMORY"
    private_question = "PRIVATE_STAGE7C_BLOCKED_DELETE_QUESTION"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    answer = _runtime_bot_message(application, telegram_id, 8_072)
    sent: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    delete_started = asyncio.Event()
    delete_never_finishes = asyncio.Event()

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(kwargs)
        return answer

    async def blocked_delete_message(self, *args, **kwargs):
        del self, args
        deletes.append(kwargs)
        delete_started.set()
        await delete_never_finishes.wait()
        return True

    async def forbidden_fallback_edit(self, *args, **kwargs):
        del self, args
        edits.append(kwargs)
        raise AssertionError("cancelled compensation must not start fallback edit")

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="conversation", confidence=0.99, answer=None)

    original_current = core.nova_memory_service.application_current_check

    async def changed_after_send(**kwargs):
        if sent:
            return NovaMemoryApplicationCurrent("memory_changed")
        return await original_current(**kwargs)

    monkeypatch.setattr(fake_ai, "route_message", conversation_route)
    monkeypatch.setattr(
        core.nova_memory_service,
        "application_current_check",
        changed_after_send,
    )
    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "delete_message", blocked_delete_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", forbidden_fallback_edit)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_DRAIN_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_CANCEL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_CANCEL_RETRY_SECONDS", 0.01)
    update = _runtime_text_update(
        application,
        telegram_id,
        private_question,
        update_id=24_072,
        source_message_id=7_072,
    )

    handler = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(delete_started.wait(), timeout=2)
    inner = tuple(core._nova_memory_application_tasks)
    assert len(inner) == 1
    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(core._post_stop(SimpleNamespace()), timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await handler
    await asyncio.sleep(0)

    assert deletes == [{"chat_id": telegram_id, "message_id": answer.message_id}]
    assert edits == []
    assert inner[0].done()
    assert inner[0].cancelled()
    assert core._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []
    application_logs = [
        record.getMessage()
        for record in caplog.records
        if "Nova memory application" in record.getMessage()
    ]
    assert any(
        "operation=delete error_type=CancelledError" in message for message in application_logs
    )
    assert not any("operation=edit" in message for message in application_logs)
    for private_value in (private_memory, private_question, str(telegram_id)):
        assert private_value not in repr(application_logs)


@pytest.mark.parametrize(
    "edit_error",
    [
        RuntimeError("PRIVATE_STAGE7C_EDIT_FAILURE"),
        asyncio.CancelledError("PRIVATE_STAGE7C_EDIT_CANCELLATION"),
    ],
    ids=["failure", "cancellation"],
)
async def test_stage7c_post_send_edit_cleanup_failure_or_cancellation_is_observed_safely(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    edit_error,
):
    telegram_id = 714_071
    private_memory = "PRIVATE_STAGE7C_EDIT_MEMORY"
    private_question = "PRIVATE_STAGE7C_EDIT_QUESTION"
    delete_error = RuntimeError("PRIVATE_STAGE7C_DELETE_BEFORE_EDIT")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    answer = _runtime_bot_message(application, telegram_id, 8_071)
    sent, edits, deletes = _patch_runtime_stage7c_cleanup_transport(
        monkeypatch,
        answer,
        delete_error=delete_error,
        edit_error=edit_error,
    )

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="conversation", confidence=0.99, answer=None)

    original_current = core.nova_memory_service.application_current_check

    async def changed_after_send(**kwargs):
        if sent:
            return NovaMemoryApplicationCurrent("memory_changed")
        return await original_current(**kwargs)

    monkeypatch.setattr(fake_ai, "route_message", conversation_route)
    monkeypatch.setattr(
        core.nova_memory_service,
        "application_current_check",
        changed_after_send,
    )
    update = _runtime_text_update(
        application,
        telegram_id,
        private_question,
        update_id=24_071,
        source_message_id=7_071,
    )
    loop = asyncio.get_running_loop()
    prior_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        with caplog.at_level(logging.WARNING):
            if isinstance(edit_error, asyncio.CancelledError):
                with pytest.raises(asyncio.CancelledError):
                    await application.process_update(update)
            else:
                await application.process_update(update)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(prior_exception_handler)

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert len(sent) == 1
    assert deletes == [{"chat_id": telegram_id, "message_id": answer.message_id}]
    assert edits == [
        {
            "chat_id": telegram_id,
            "message_id": answer.message_id,
            "text": NOVA_MEMORY_APPLICATION_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    assert loop_errors == []
    assert core._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []
    assert "operation=delete" in caplog.text
    assert "operation=edit" in caplog.text
    assert f"error_type={type(edit_error).__name__}" in caplog.text
    assert "Task exception was never retrieved" not in caplog.text
    for private_value in (
        private_memory,
        private_question,
        str(telegram_id),
        str(delete_error),
        str(edit_error),
    ):
        assert private_value not in caplog.text


async def test_stage7c_enabled_route_failure_log_contains_only_safe_metadata(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    telegram_id = 714_080
    private_memory = "PRIVATE_STAGE7C_ROUTE_MEMORY"
    private_question = "PRIVATE_STAGE7C_ROUTE_QUESTION"
    private_error = "PRIVATE_STAGE7C_ROUTE_EXCEPTION"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    reply = _runtime_bot_message(application, telegram_id, 8_080)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    class Stage7CRouteFailure(RuntimeError):
        pass

    async def failed_route(*args, **kwargs):
        del args, kwargs
        raise Stage7CRouteFailure(private_error)

    monkeypatch.setattr(fake_ai, "route_message", failed_route)
    update = _runtime_text_update(
        application,
        telegram_id,
        private_question,
        update_id=24_080,
        source_message_id=7_080,
    )

    with caplog.at_level(logging.WARNING):
        await application.process_update(update)

    assert len(sent) == 1
    assert fake_ai.answer_calls == []
    assert "stage=route" in caplog.text
    assert "error_type=Stage7CRouteFailure" in caplog.text
    assert "user_id=" not in caplog.text
    for private_value in (
        private_memory,
        private_question,
        private_error,
        str(telegram_id),
    ):
        assert private_value not in caplog.text


@pytest.mark.parametrize("failure_stage", ["storage", "provider"])
async def test_stage7c_enabled_storage_and_provider_failure_logs_are_memory_blind(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    failure_stage,
):
    telegram_id = 714_081 if failure_stage == "storage" else 714_082
    private_memory = f"PRIVATE_STAGE7C_{failure_stage.upper()}_MEMORY"
    private_question = f"PRIVATE_STAGE7C_{failure_stage.upper()}_QUESTION"
    private_error = f"PRIVATE_STAGE7C_{failure_stage.upper()}_EXCEPTION"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    reply = _runtime_bot_message(application, telegram_id, 8_081)
    sent, _, _ = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="conversation", confidence=0.99, answer=None)

    class Stage7CStorageFailure(RuntimeError):
        pass

    class Stage7CProviderFailure(RuntimeError):
        pass

    async def failed_snapshot(**kwargs):
        del kwargs
        raise Stage7CStorageFailure(private_error)

    async def failed_answer(*args, **kwargs):
        del args, kwargs
        raise Stage7CProviderFailure(private_error)

    monkeypatch.setattr(fake_ai, "route_message", conversation_route)
    if failure_stage == "storage":
        monkeypatch.setattr(
            core.nova_memory_service,
            "application_snapshot",
            failed_snapshot,
        )
    else:
        monkeypatch.setattr(fake_ai, "answer_message", failed_answer)
    update = _runtime_text_update(
        application,
        telegram_id,
        private_question,
        update_id=24_081 + telegram_id,
        source_message_id=7_081,
    )

    with caplog.at_level(logging.WARNING):
        await application.process_update(update)

    assert len(fake_ai.route_calls) == 1
    assert [entry["text"] for entry in sent] == [NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT]
    expected_stage = "snapshot" if failure_stage == "storage" else "provider"
    expected_error = (
        "Stage7CStorageFailure" if failure_stage == "storage" else "Stage7CProviderFailure"
    )
    assert f"stage={expected_stage}" in caplog.text
    assert f"error_type={expected_error}" in caplog.text
    assert "user_id=" not in caplog.text
    for private_value in (
        private_memory,
        private_question,
        private_error,
        str(telegram_id),
    ):
        assert private_value not in caplog.text


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_stage7c_cross_turn_memory_answer_never_reenters_provider_context(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 714_090 if source == "text" else 714_091
    private_memory = f"PRIVATE_STAGE7C_{source.upper()}_CROSS_TURN_MEMORY"
    memory_answer = f"PRIVATE_STAGE7C_{source.upper()}_MEMORY_ANSWER_SENTINEL"
    ordinary_answer = f"ORDINARY_STAGE7C_{source.upper()}_ANSWER_SENTINEL"
    transcription = RuntimeTranscription("cross turn question")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    await core.conversation.append(
        telegram_id,
        telegram_id,
        role="assistant",
        content=ordinary_answer,
        source="text",
        intent="answer",
    )

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="conversation", confidence=0.99, answer=None)

    original_answer = fake_ai.answer_message
    answer_number = 0

    async def sentinel_answer(*args, **kwargs):
        nonlocal answer_number
        generated = await original_answer(*args, **kwargs)
        answer_number += 1
        answer = memory_answer if answer_number == 1 else "SECOND_STAGE7C_PERSONALIZED_ANSWER"
        return generated.model_copy(update={"answer": answer})

    monkeypatch.setattr(fake_ai, "route_message", conversation_route)
    monkeypatch.setattr(fake_ai, "answer_message", sentinel_answer)
    if source == "voice":
        first_update, first_progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=24_090,
            source_message_id=7_090,
            progress_message_id=8_090,
        )
        second_update, second_progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=24_091,
            source_message_id=7_091,
            progress_message_id=8_092,
        )
        first_answer = _runtime_bot_message(application, telegram_id, 8_091)
        second_answer = _runtime_bot_message(application, telegram_id, 8_093)
        sent, edits, _ = _patch_runtime_stage7c_transport(
            monkeypatch,
            [first_progress, first_answer, second_progress, second_answer],
        )
    else:
        first_update = _runtime_text_update(
            application,
            telegram_id,
            "first cross turn question",
            update_id=24_090,
            source_message_id=7_090,
        )
        second_update = _runtime_text_update(
            application,
            telegram_id,
            "second cross turn question",
            update_id=24_091,
            source_message_id=7_091,
        )
        first_answer = _runtime_bot_message(application, telegram_id, 8_090)
        second_answer = _runtime_bot_message(application, telegram_id, 8_091)
        sent, edits, _ = _patch_runtime_stage7c_transport(
            monkeypatch,
            [first_answer, second_answer],
        )

    await application.process_update(first_update)
    async with db.sessions() as session:
        first_assistant = await session.scalar(
            select(ConversationMessage).where(
                ConversationMessage.role == "assistant",
                ConversationMessage.content == memory_answer,
            )
        )
    assert first_assistant is not None
    assert first_assistant.intent == "memory_answer"

    await application.process_update(second_update)

    assert len(fake_ai.route_calls) == 2
    assert len(fake_ai.answer_calls) == 2
    assert len(fake_ai.answer_confirmed_memory_calls) == 2
    assert all(projection is not None for projection in fake_ai.answer_confirmed_memory_calls)
    assert [record.content for record in fake_ai.answer_confirmed_memory_calls[-1].records] == [
        private_memory
    ]
    for provider_context in (
        fake_ai.conversation_contexts[-1],
        fake_ai.answer_conversation_contexts[-1],
    ):
        assert memory_answer not in repr(provider_context)
        assert ordinary_answer in repr(provider_context)
        assert any(
            message["role"] == "assistant" and message["content"] == ordinary_answer
            for message in provider_context["recent_messages"]
        )
        assert any(message["role"] == "user" for message in provider_context["recent_messages"])
    local_snapshot = await core.conversation.get(telegram_id, telegram_id)
    assert any(
        message["content"] == memory_answer and message["intent"] == "memory_answer"
        for message in local_snapshot.messages
    )
    if source == "voice":
        assert [entry["text"] for entry in sent] == [
            "Расшифровываю голосовую мысль…",
            memory_answer,
            "Расшифровываю голосовую мысль…",
            "SECOND_STAGE7C_PERSONALIZED_ANSWER",
        ]
        assert len(edits) == 2
        assert all(str(edit["text"]).startswith("Я услышал:") for edit in edits)
    else:
        assert [entry["text"] for entry in sent] == [
            memory_answer,
            "SECOND_STAGE7C_PERSONALIZED_ANSWER",
        ]


@pytest.mark.parametrize("state_change", ["delete", "delete_all", "kill_switch"])
async def test_stage7c_stale_memory_answer_is_filtered_after_delete_or_kill_switch(
    db,
    fake_ai,
    monkeypatch,
    state_change,
):
    telegram_id = {
        "delete": 714_092,
        "delete_all": 714_093,
        "kill_switch": 714_094,
    }[state_change]
    private_memory = f"PRIVATE_STAGE7C_{state_change.upper()}_MEMORY"
    memory_answer = f"PRIVATE_STAGE7C_{state_change.upper()}_ANSWER_SENTINEL"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    created = await _runtime_stage7c_memory(core, owner, private_memory)
    first_answer = _runtime_bot_message(application, telegram_id, 8_094)
    second_answer = _runtime_bot_message(application, telegram_id, 8_095)
    sent, _, _ = _patch_runtime_stage7c_transport(
        monkeypatch,
        [first_answer, second_answer],
    )

    async def conversation_route(text, temporal_context, conversation_context=None):
        fake_ai.route_calls.append((text, temporal_context))
        fake_ai.conversation_contexts.append(conversation_context or {})
        return IntentResult(intent="question", confidence=0.99, answer=None)

    original_answer = fake_ai.answer_message
    answer_number = 0

    async def sentinel_answer(*args, **kwargs):
        nonlocal answer_number
        generated = await original_answer(*args, **kwargs)
        answer_number += 1
        answer = memory_answer if answer_number == 1 else "ANSWER_AFTER_STAGE7C_STATE_CHANGE"
        return generated.model_copy(update={"answer": answer})

    monkeypatch.setattr(fake_ai, "route_message", conversation_route)
    monkeypatch.setattr(fake_ai, "answer_message", sentinel_answer)
    first_update = _runtime_text_update(
        application,
        telegram_id,
        "first state change question",
        update_id=24_092,
        source_message_id=7_092,
    )
    await application.process_update(first_update)
    assert fake_ai.answer_confirmed_memory_calls[-1] is not None

    assert created.item is not None
    if state_change == "delete":
        deleted = await core.nova_memory_service.delete(
            telegram_actor_id=telegram_id,
            public_id=created.item.public_id,
            expected_version=created.item.version,
            expected_access_version=owner.access_version,
        )
        assert deleted.status == "deleted"
    elif state_change == "delete_all":
        page = await core.nova_memory_service.list(telegram_actor_id=telegram_id)
        assert page.collection_revision is not None
        deleted = await core.nova_memory_service.delete_all(
            telegram_actor_id=telegram_id,
            expected_access_version=owner.access_version,
            expected_collection_revision=page.collection_revision,
        )
        assert deleted.status == "deleted_all"
    else:
        core.settings.enable_nova_memory_application = False

    second_update = _runtime_text_update(
        application,
        telegram_id,
        "second state change question",
        update_id=24_093,
        source_message_id=7_093,
    )
    await application.process_update(second_update)

    assert [entry["text"] for entry in sent] == [
        memory_answer,
        "ANSWER_AFTER_STAGE7C_STATE_CHANGE",
    ]
    assert fake_ai.answer_confirmed_memory_calls[-1] is None
    for provider_context in (
        fake_ai.conversation_contexts[-1],
        fake_ai.answer_conversation_contexts[-1],
    ):
        assert memory_answer not in repr(provider_context)
    local_snapshot = await core.conversation.get(telegram_id, telegram_id)
    assert any(
        message["content"] == memory_answer and message["intent"] == "memory_answer"
        for message in local_snapshot.messages
    )


@pytest.mark.parametrize(
    "post_send_state",
    ["success", "memory_mutation", "downgrade", "version_bounce"],
)
async def test_stage7c_outer_cancel_during_primary_send_keeps_combined_lifecycle_tracked(
    db,
    fake_ai,
    monkeypatch,
    post_send_state,
):
    telegram_id = {
        "success": 714_095,
        "memory_mutation": 714_096,
        "downgrade": 714_099,
        "version_bounce": 714_100,
    }[post_send_state]
    private_memory = f"PRIVATE_STAGE7C_COMBINED_SEND_{post_send_state.upper()}"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    initial = await _runtime_stage7c_memory(core, owner, private_memory)
    answer = _runtime_bot_message(application, telegram_id, 8_096)
    send_started = asyncio.Event()
    send_release = asyncio.Event()
    _, edits, deletes = _patch_runtime_stage7c_transport(
        monkeypatch,
        [answer],
        send_started=send_started,
        send_release=send_release,
    )
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_095,
        source_message_id=7_095,
    )

    outer = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(send_started.wait(), timeout=2)
    inner_tasks = tuple(core._nova_memory_application_tasks)
    assert len(inner_tasks) == 1
    inner = inner_tasks[0]
    assert inner.get_name() == "nova-memory-application-send-lifecycle"
    assert private_memory not in inner.get_name()
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert not inner.done()
    if post_send_state == "memory_mutation":
        await _runtime_stage7c_mutate_memory(
            core,
            owner,
            "create",
            initial,
            suffix="COMBINED_SEND_CANCEL",
        )
    elif post_send_state == "downgrade":
        await AccessService(db).set_guest(telegram_id, source="combined-send-cancel")
    elif post_send_state == "version_bounce":
        access = AccessService(db)
        await access.set_guest(telegram_id, source="combined-send-cancel-bounce")
        await access.grant_subscriber(telegram_id, source="combined-send-cancel-bounce")
    send_release.set()
    assert await asyncio.wait_for(inner, timeout=2) is (post_send_state == "success")
    await asyncio.sleep(0)

    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert core._nova_memory_application_tasks == set()
    assert edits == []
    if post_send_state == "success":
        assert deletes == []
    else:
        assert deletes == [{"chat_id": telegram_id, "message_id": answer.message_id}]
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    if post_send_state == "success":
        assert len(assistant_messages) == 1
        assert assistant_messages[0].intent == "memory_answer"
    else:
        assert assistant_messages == []


async def test_stage7c_post_fence_context_append_failure_is_observed_without_fallback(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    telegram_id = 714_098
    private_memory = "PRIVATE_STAGE7C_APPEND_FAILURE_MEMORY"
    private_error = "PRIVATE_STAGE7C_APPEND_FAILURE_DETAIL"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
            enable_nova_memory_application=True,
            nova_memory_application_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id)
    await _runtime_stage7c_memory(core, owner, private_memory)
    answer = _runtime_bot_message(application, telegram_id, 8_098)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [answer])
    original_append = core.conversation.append

    class Stage7CAppendFailure(RuntimeError):
        pass

    async def fail_assistant_append(*args, **kwargs):
        if kwargs.get("role") == "assistant":
            raise Stage7CAppendFailure(private_error)
        return await original_append(*args, **kwargs)

    monkeypatch.setattr(core.conversation, "append", fail_assistant_append)
    update = _runtime_text_update(
        application,
        telegram_id,
        "привет",
        update_id=24_098,
        source_message_id=7_098,
    )

    with caplog.at_level(logging.WARNING):
        await application.process_update(update)
    await asyncio.sleep(0)

    assert [entry["text"] for entry in sent] == ["Ответ на: привет"]
    assert edits == []
    assert deletes == []
    assert len(fake_ai.route_calls) == 1
    assert len(fake_ai.answer_calls) == 1
    assert core._nova_memory_application_tasks == set()
    async with db.sessions() as session:
        assistant_messages = list(
            await session.scalars(
                select(ConversationMessage).where(ConversationMessage.role == "assistant")
            )
        )
    assert assistant_messages == []
    lifecycle_logs = [
        record.getMessage()
        for record in caplog.records
        if "operation=send_lifecycle" in record.getMessage()
    ]
    assert lifecycle_logs == [
        "Nova memory application task failed operation=send_lifecycle "
        "error_type=Stage7CAppendFailure"
    ]
    for private_value in (private_memory, private_error, str(telegram_id)):
        assert private_value not in repr(lifecycle_logs)


async def test_stage7c_shutdown_drain_waits_only_tracked_tasks_and_is_idempotent(
    db,
    fake_ai,
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    tracked_started = asyncio.Event()
    tracked_release = asyncio.Event()
    unrelated_release = asyncio.Event()

    async def tracked_delivery():
        tracked_started.set()
        await tracked_release.wait()
        return True

    tracked = asyncio.create_task(
        tracked_delivery(),
        name="nova-memory-application-send-lifecycle",
    )
    core._track_nova_memory_application_task(tracked)
    unrelated = asyncio.create_task(
        unrelated_release.wait(),
        name="unrelated-application-task",
    )
    loop = asyncio.get_running_loop()
    prior_debug = loop.get_debug()
    prior_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        loop.set_debug(True)
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            await asyncio.wait_for(tracked_started.wait(), timeout=2)
            stopping = asyncio.create_task(core._post_stop(SimpleNamespace()))
            await asyncio.sleep(0)
            assert not stopping.done()
            assert not tracked.done()
            assert not unrelated.done()
            await core._user(714_097)

            tracked_release.set()
            await asyncio.wait_for(stopping, timeout=2)
            assert tracked.result() is True
            assert core._nova_memory_application_tasks == set()
            assert not unrelated.done()

            await core._post_stop(SimpleNamespace())
            assert not unrelated.done()
            await core._post_shutdown(SimpleNamespace())
            await core._post_shutdown(SimpleNamespace())
            assert not unrelated.done()
            assert fake_ai.answer_calls == []
        finally:
            tracked_release.set()
            unrelated_release.set()
            await asyncio.gather(tracked, unrelated, return_exceptions=True)
            await asyncio.sleep(0)
            gc.collect()
            loop.set_exception_handler(prior_exception_handler)
            loop.set_debug(prior_debug)
    assert loop_errors == []
    assert [warning for warning in caught_warnings if warning.category is RuntimeWarning] == []


def _patch_runtime_weekly_transport(monkeypatch, application):
    sent: list[dict[str, object]] = []
    sent_messages: list[Message] = []
    edits: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    messages: dict[tuple[int, int], Message] = {}
    next_message_id = 81_000
    bot_user = TelegramUser(123456, True, "Future Self")

    def make_message(chat_id: int, message_id: int, text_value: str = "pending") -> Message:
        message = Message(
            message_id,
            datetime.now(UTC),
            Chat(chat_id, "private"),
            from_user=bot_user,
            text=text_value,
        )
        message.set_bot(application.bot)
        messages[(chat_id, message_id)] = message
        return message

    async def fake_send_message(self, *args, **kwargs):
        nonlocal next_message_id
        del self, args
        record = dict(kwargs)
        sent.append(record)
        chat_id = int(record["chat_id"])
        message = make_message(chat_id, next_message_id, str(record.get("text", "")))
        next_message_id += 1
        sent_messages.append(message)
        return message

    async def fake_edit_message_text(self, *args, **kwargs):
        del self, args
        record = dict(kwargs)
        edits.append(record)
        chat_id = int(record["chat_id"])
        message_id = int(record["message_id"])
        return messages.get((chat_id, message_id)) or make_message(
            chat_id,
            message_id,
            str(record.get("text", "")),
        )

    async def fake_delete_message(self, *args, **kwargs):
        del self, args
        deletes.append(dict(kwargs))
        return True

    async def fake_answer_callback_query(self, callback_query_id, *args, **kwargs):
        del self, args
        answers.append({"callback_query_id": callback_query_id, **kwargs})
        return True

    async def fake_set_my_commands(self, commands, **kwargs):
        del self, commands, kwargs
        return True

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(ExtBot, "delete_message", fake_delete_message)
    monkeypatch.setattr(ExtBot, "answer_callback_query", fake_answer_callback_query)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_my_commands)
    return SimpleNamespace(
        sent=sent,
        sent_messages=sent_messages,
        edits=edits,
        deletes=deletes,
        answers=answers,
        make_message=make_message,
    )


def _runtime_weekly_callback_update(
    application,
    telegram_user: TelegramUser,
    message: Message,
    data: str,
    *,
    update_id: int,
) -> tuple[Update, CallbackQuery]:
    query = CallbackQuery(
        f"runtime-weekly-{update_id}",
        telegram_user,
        "runtime-weekly-chat",
        message=message,
        data=data,
    )
    update = Update(update_id, callback_query=query)
    update.set_bot(application.bot)
    query.set_bot(application.bot)
    return update, query


def _runtime_weekly_text_update(
    application,
    telegram_id: int,
    text_value: str,
    *,
    update_id: int,
    source_message_id: int,
) -> Update:
    telegram_user = TelegramUser(telegram_id, "Варвара", False)
    source_message = Message(
        source_message_id,
        datetime.now(UTC),
        Chat(telegram_id, "private"),
        from_user=telegram_user,
        text=text_value,
    )
    update = Update(update_id, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    return update


def _runtime_weekly_command_update(
    application,
    telegram_id: int,
    command: str,
    *,
    update_id: int,
    source_message_id: int,
) -> Update:
    application.bot._bot_user = TelegramUser(
        123456,
        "Future Self",
        True,
        username="future_self_bot",
    )
    telegram_user = TelegramUser(telegram_id, "Варвара", False)
    source_message = Message(
        source_message_id,
        datetime.now(UTC),
        Chat(telegram_id, "private"),
        from_user=telegram_user,
        text=command,
        entities=[MessageEntity(MessageEntity.BOT_COMMAND, 0, len(command))],
    )
    update = Update(update_id, message=source_message)
    update.set_bot(application.bot)
    source_message.set_bot(application.bot)
    return update


async def _runtime_weekly_today_callback_update(
    core: FutureSelfBot,
    application,
    transport,
    owner: User,
    *,
    canonical_message_id: int,
    update_id: int,
) -> tuple[Update, CallbackQuery]:
    created = await core.weekly_review_service.create_session(
        telegram_actor_id=owner.telegram_id,
        chat_id=owner.telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=canonical_message_id,
        phase=WeeklyReviewPhase.SAVED,
    )
    assert created.status == "created"
    assert created.session is not None
    canonical = transport.make_message(
        owner.telegram_id,
        canonical_message_id,
        "PRIVATE_FROZEN_WEEKLY_SCREEN",
    )
    tokens = await core.weekly_review_capabilities.issue(
        actions=("today",),
        owner_id=owner.id,
        telegram_user_id=owner.telegram_id,
        chat_id=owner.telegram_id,
        canonical_message_id=canonical_message_id,
        access_version=owner.access_version,
        week_start=created.session.week_start,
        session_public_id=created.session.public_id,
        session_version=created.session.version,
    )
    telegram_user = TelegramUser(owner.telegram_id, "Варвара", False)
    return _runtime_weekly_callback_update(
        application,
        telegram_user,
        canonical,
        f"wrev:{tokens['today']}",
        update_id=update_id,
    )


async def _runtime_apply_today_race(
    db,
    owner: User,
    race: str,
    *,
    week_start: date,
    advance_clock,
) -> None:
    if race == "downgrade":
        await AccessService(db).set_guest(owner.telegram_id, source="today-fence-test")
        return
    if race == "bounce":
        access = AccessService(db)
        await access.set_guest(owner.telegram_id, source="today-fence-test")
        await access.grant_subscriber(owner.telegram_id, source="today-fence-test")
        return
    if race == "timezone":
        async with db.session() as session:
            stored = await session.get(User, owner.id)
            assert stored is not None
            stored.timezone = "UTC" if stored.timezone != "UTC" else "Europe/Moscow"
        return
    if race == "week":
        advance_clock()
        return
    async with db.session() as session:
        focus = await session.scalar(
            select(WeeklyFocus).where(
                WeeklyFocus.owner_id == owner.id,
                WeeklyFocus.week_start == week_start,
            )
        )
        if race == "focus_edit":
            assert focus is not None
            focus.focus = "PRIVATE_CHANGED_WEEKLY_FOCUS"
            focus.version += 1
        elif race == "focus_delete":
            assert focus is not None
            await session.delete(focus)
        elif race == "focus_create":
            assert focus is None
            session.add(
                WeeklyFocus(
                    owner_id=owner.id,
                    week_start=week_start,
                    focus="PRIVATE_CREATED_WEEKLY_FOCUS",
                    approach=None,
                    small_steps=[],
                    source="text",
                )
            )
        else:
            raise AssertionError(f"unknown today race: {race}")


async def _runtime_prepare_today_case(
    db,
    fake_ai,
    monkeypatch,
    *,
    surface: str,
    with_focus: bool,
    telegram_id: int,
):
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    if with_focus:
        await _runtime_weekly_focus(
            core,
            db,
            owner,
            "PRIVATE_FROZEN_WEEKLY_FOCUS",
        )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    query = None
    if surface == "command":
        update = _runtime_weekly_command_update(
            application,
            telegram_id,
            "/today",
            update_id=32_100,
            source_message_id=97_100,
        )
    else:
        assert surface == "callback"
        update, query = await _runtime_weekly_today_callback_update(
            core,
            application,
            transport,
            owner,
            canonical_message_id=97_101,
            update_id=32_101,
        )
    return SimpleNamespace(
        core=core,
        application=application,
        owner=owner,
        transport=transport,
        update=update,
        query=query,
    )


async def _assert_runtime_today_callback_terminal(
    env,
    *,
    edit_count: int,
    access_changed: bool,
) -> None:
    assert env.query is not None
    assert len(env.transport.edits) == edit_count
    canonical_message_id = env.query.message.message_id
    assert {entry["message_id"] for entry in env.transport.edits} == {canonical_message_id}
    final = env.transport.edits[-1]
    if access_changed:
        assert final["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
        assert final["reply_markup"] is None
        return
    markup = final["reply_markup"]
    assert markup is not None
    callbacks = [
        str(button.callback_data)
        for row in markup.inline_keyboard
        for button in row
        if str(button.callback_data).startswith("wrev:")
    ]
    assert callbacks
    for callback in callbacks:
        claim = await env.core.weekly_review_capabilities.peek(
            callback.removeprefix("wrev:"),
            telegram_user_id=env.owner.telegram_id,
            chat_id=env.owner.telegram_id,
            canonical_message_id=canonical_message_id,
        )
        assert claim is not None


def _runtime_callback_for_fragment(markup, fragment: str, *, prefix: str) -> str:
    matches = [
        str(button.callback_data)
        for row in markup.inline_keyboard
        for button in row
        if fragment in button.text and str(button.callback_data).startswith(prefix)
    ]
    assert len(matches) == 1
    return matches[0]


async def test_real_application_scheduled_weekly_capability_keeps_next_week_binding(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_001
    frozen_sunday = datetime(2026, 8, 16, 15, 0, tzinfo=UTC)
    next_monday = date(2026, 8, 17)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    assert (
        core.weekly_review_service.target_week_start(
            owner.timezone,
            now=frozen_sunday,
            scheduled=True,
            review_weekday=6,
        )
        == next_monday
    )
    live_scheduled_week = core.weekly_review_service.target_week_start(
        owner.timezone,
        scheduled=True,
        review_weekday=6,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    canonical = transport.make_message(telegram_id, 91_001, "weekly scheduled nudge")
    original_create_session = core.weekly_review_service.create_session
    scheduled_values: list[bool] = []

    async def record_scheduled_create(**kwargs):
        scheduled_values.append(bool(kwargs.get("scheduled")))
        return await original_create_session(**kwargs)

    monkeypatch.setattr(core.weekly_review_service, "create_session", record_scheduled_create)
    tokens = await core.weekly_review_capabilities.issue(
        actions=("start", "focus", "reminders", "cancel"),
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
        canonical_message_id=canonical.message_id,
        access_version=owner.access_version,
        week_start=live_scheduled_week,
        scheduled=True,
    )
    assert all(token.isascii() and len(token) <= 40 for token in tokens.values())
    assert all("715001" not in token and "2026" not in token for token in tokens.values())
    telegram_user = TelegramUser(telegram_id, "Варвара", False)
    callback_update, query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        canonical,
        f"wrev:{tokens['start']}",
        update_id=31_001,
    )

    await application.process_update(callback_update)

    current = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.status == "found", (
        scheduled_values,
        transport.answers,
        transport.edits,
    )
    assert current.session is not None
    assert current.session.week_start == live_scheduled_week
    assert current.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert current.session.canonical_message_id == canonical.message_id
    assert transport.sent == []
    assert str(transport.edits[-1]["text"]).endswith(WEEKLY_REVIEW_QUESTION)
    assert transport.edits[-1]["message_id"] == canonical.message_id
    assert [entry["callback_query_id"] for entry in transport.answers] == [query.id]
    assert scheduled_values == [True]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


async def test_real_application_varvara_weekly_review_requires_two_independent_confirms(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_002
    launch = "Давай скорректируем систему"
    focus_launch = "Фокус на неделю"
    exact_focus = "спокойно закрывать подтверждённые напоминания дня."
    reminder_evidence = "В 15:05 сказать Назару, что я люблю его…"
    long_answer = (
        "Хочу скорректировать.\n"
        "Можно держать в уме образ будущего и двигаться к нему\n"
        "через один небольшой шаг.\n"
        f"Фокус: {exact_focus}\n"
        f"{reminder_evidence}"
    )
    fake_ai.weekly_review_result = WeeklyReviewExtraction(
        focus="provider must not replace the explicit focus",
        approach="Держать в уме образ будущего",
        small_steps=["Сделать один небольшой шаг"],
        reminder_candidates=[
            {
                "title": "сказать Назару, что я люблю его",
                "schedule_wording": "В 15:05",
                "evidence": reminder_evidence,
            }
        ],
    )
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    downstream: list[int] = []

    async def forbidden_generic(*args, **kwargs):
        del args, kwargs
        raise AssertionError("weekly routing must not enter generic AI")

    async def downstream_probe(update, context):
        del context
        downstream.append(update.update_id)

    monkeypatch.setattr(fake_ai, "route_message", forbidden_generic)
    probe = TypeHandler(Update, downstream_probe)
    application.add_handler(probe, group=100)

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            launch,
            update_id=31_010,
            source_message_id=92_010,
        )
    )
    assert len(transport.sent) == 1
    canonical = transport.sent_messages[0]
    assert transport.edits[-1]["message_id"] == canonical.message_id
    assert all(
        str(button.callback_data).startswith("wrev:")
        for row in transport.edits[-1]["reply_markup"].inline_keyboard
        for button in row
    )
    assert fake_ai.weekly_review_calls == []
    assert downstream == []

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            focus_launch,
            update_id=31_011,
            source_message_id=92_011,
        )
    )
    assert str(transport.edits[-1]["text"]).endswith(WEEKLY_REVIEW_QUESTION)
    assert transport.edits[-1]["message_id"] == canonical.message_id
    assert len(transport.sent) == 1

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            long_answer,
            update_id=31_012,
            source_message_id=92_012,
        )
    )
    application.remove_handler(probe, group=100)

    assert downstream == []
    assert [call[0] for call in fake_ai.weekly_review_calls] == [long_answer]
    preview = str(transport.edits[-1]["text"])
    assert exact_focus in preview
    assert "Держать в уме образ будущего" in preview
    assert "Сделать один небольшой шаг" in preview
    assert "сказать Назару, что я люблю его" in preview
    assert "В 15:05" in preview
    assert "Ещё не создано" in preview
    assert transport.edits[-1]["message_id"] == canonical.message_id
    preview_markup = transport.edits[-1]["reply_markup"]
    save_data = _runtime_callback_for_fragment(
        preview_markup,
        "Сохранить на неделю",
        prefix="wrev:",
    )
    current = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.status == "found"
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.PREVIEW
    assert current.session.focus == exact_focus
    assert current.session.source == "text"
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0

    telegram_user = TelegramUser(telegram_id, "Варвара", False)
    callback_queries: list[CallbackQuery] = []

    async def click(data: str, update_id: int) -> None:
        callback_update, query = _runtime_weekly_callback_update(
            application,
            telegram_user,
            canonical,
            data,
            update_id=update_id,
        )
        callback_queries.append(query)
        before = len(transport.answers)
        await application.process_update(callback_update)
        assert len(transport.answers) == before + 1
        assert transport.answers[-1]["callback_query_id"] == query.id

    await click(save_data, 31_013)

    assert str(transport.edits[-1]["text"]).endswith("✅ Фокус недели сохранён")
    async with db.sessions() as session:
        focus_rows = list((await session.scalars(select(WeeklyFocus))).all())
        audit_rows = list((await session.scalars(select(WeeklyFocusChange))).all())
        assert len(focus_rows) == 1
        assert focus_rows[0].focus == exact_focus
        assert focus_rows[0].source == "text"
        assert len(audit_rows) == 1
        assert audit_rows[0].operation == "created"
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0

    configure_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Настроить найденные (1)",
        prefix="wrev:",
    )
    await click(configure_data, 31_014)
    candidate_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Назару",
        prefix="wrev:",
    )
    assert exact_focus not in candidate_data
    assert "Назару" not in candidate_data
    await click(candidate_data, 31_015)

    reminder_session = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert reminder_session is not None
    assert reminder_session.canonical_message_id == canonical.message_id
    assert str(transport.edits[-1]["text"]) == "🔔 Когда напомнить?"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0

    tomorrow_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Завтра",
        prefix="rmd:",
    )
    await click(tomorrow_data, 31_016)
    assert "Проверь напоминание" in str(transport.edits[-1]["text"])
    confirm_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Создать",
        prefix="rmd:",
    )
    await click(confirm_data, 31_017)

    assert "Напоминание создано" in str(transport.edits[-1]["text"])
    assert len(transport.sent) == 1
    assert all(entry["message_id"] == canonical.message_id for entry in transport.edits)
    assert len(transport.answers) == len(callback_queries)
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


async def test_real_application_active_weekly_voice_owns_reminder_like_transcript(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_003
    transcript = (
        "На этой неделе спокойно закрывать дела. Завтра в 15:05 сказать Назару, что я люблю его."
    )
    reminder_evidence = "Завтра в 15:05 сказать Назару, что я люблю его."
    transcription = RuntimeTranscription(transcript)
    fake_ai.weekly_review_result = WeeklyReviewExtraction(
        focus="Спокойно закрывать дела",
        small_steps=["Закрыть одно подтверждённое дело"],
        reminder_candidates=[
            {
                "title": "сказать Назару, что я люблю его",
                "schedule_wording": "Завтра в 15:05",
                "evidence": reminder_evidence,
            }
        ],
    )
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    created = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=91_003,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_020,
        source_message_id=92_020,
        progress_message_id=93_020,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    deletes: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_delete_message(self, *args, **kwargs):
        del self, args
        deletes.append(dict(kwargs))
        return True

    async def forbidden_generic(*args, **kwargs):
        del args, kwargs
        raise AssertionError("active weekly voice must stop generic/reminder routing")

    async def downstream_probe(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "delete_message", fake_delete_message)
    monkeypatch.setattr(fake_ai, "route_message", forbidden_generic)
    application.add_handler(TypeHandler(Update, downstream_probe), group=100)

    await application.process_update(update)

    current = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.status == "found"
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.PREVIEW
    assert current.session.source == "voice"
    assert [call[0] for call in fake_ai.weekly_review_calls] == [transcript]
    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert downstream == []
    assert [entry["text"] for entry in sent] == ["Расшифровываю голосовую мысль…"]
    assert edits[-1]["message_id"] == created.session.canonical_message_id
    assert "Спокойно закрывать дела" in str(edits[-1]["text"])
    assert [(entry["chat_id"], entry["message_id"]) for entry in deletes] == [
        (telegram_id, progress.message_id)
    ]
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


async def test_real_application_natural_weekly_voice_retires_progress_and_reuses_cleanup_send(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_004
    transcript = "Давай скорректируем систему"
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    update, _unused_progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_021,
        source_message_id=92_021,
        progress_message_id=93_021,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    await application.process_update(update)
    await asyncio.sleep(0)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert [entry["text"] for entry in transport.sent] == [
        "Расшифровываю голосовую мысль…",
        transport.edits[0]["text"],
    ]
    assert isinstance(transport.sent[1]["reply_markup"], ReplyKeyboardRemove)
    progress, canonical = transport.sent_messages
    assert [(entry["chat_id"], entry["message_id"]) for entry in transport.deletes] == [
        (telegram_id, progress.message_id)
    ]
    assert len(transport.edits) == 1
    assert transport.edits[0]["message_id"] == canonical.message_id
    assert transport.edits[0]["reply_markup"] is not None
    current = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.status == "found"
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert current.session.canonical_message_id == canonical.message_id
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert core._weekly_review_tasks == set()


@pytest.mark.parametrize("durable_owner", ["onboarding", "workspace"])
async def test_real_application_durable_owner_wins_natural_weekly_voice(
    db,
    fake_ai,
    monkeypatch,
    durable_owner,
):
    telegram_id = 715_005 if durable_owner == "onboarding" else 715_006
    transcript = "Давай скорректируем систему"
    transcription = RuntimeTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
            enable_workspace_access=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=durable_owner != "onboarding",
    )
    if durable_owner == "onboarding":
        async with db.session() as session:
            session.add(
                OnboardingState(
                    user_id=owner.id,
                    current_step=2,
                    answers={"display_name": "Варвара"},
                    status="in_progress",
                )
            )
    else:
        await core.workspace_service.begin_input(
            owner.id,
            telegram_id,
            "invite_recipient",
            payload={"request_id": 47},
        )
    update, _unused_progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_022 if durable_owner == "onboarding" else 31_023,
        source_message_id=92_022,
        progress_message_id=93_022,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    await application.process_update(update)
    await asyncio.sleep(0)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert all("Обзор недели" not in str(entry.get("text", "")) for entry in transport.sent)
    assert all("Обзор недели" not in str(entry.get("text", "")) for entry in transport.edits)
    assert all(
        not isinstance(entry.get("reply_markup"), ReplyKeyboardRemove) for entry in transport.sent
    )
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert (
        await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
    ).session is None
    if durable_owner == "onboarding":
        async with db.sessions() as session:
            state = await session.scalar(
                select(OnboardingState).where(OnboardingState.user_id == owner.id)
            )
        assert state is not None
        assert state.status == "in_progress"
        assert state.current_step == 3
        assert state.answers["future_life"] == transcript
    else:
        pending = await core.workspace_service.pending_input(owner.id, telegram_id)
        assert pending is not None
        assert pending.action == "input:invite_recipient"
        assert pending.payload == {"request_id": 47}
    assert core._weekly_review_tasks == set()


@pytest.mark.parametrize("durable_owner", ["onboarding", "workspace"])
async def test_real_application_durable_owner_wins_explicit_weekly_text(
    db,
    fake_ai,
    monkeypatch,
    durable_owner,
):
    telegram_id = 715_010 if durable_owner == "onboarding" else 715_011
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
            enable_workspace_access=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=durable_owner != "onboarding",
    )
    if durable_owner == "onboarding":
        async with db.session() as session:
            session.add(
                OnboardingState(
                    user_id=owner.id,
                    current_step=2,
                    answers={"display_name": "Варвара"},
                    status="in_progress",
                )
            )
    else:
        await core.workspace_service.begin_input(
            owner.id,
            telegram_id,
            "create_name",
            payload={"character": "family"},
        )
    created = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=91_010 + telegram_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    _patch_runtime_weekly_transport(monkeypatch, application)

    async def forbidden_generic(*args, **kwargs):
        del args, kwargs
        raise AssertionError("durable owner must stop weekly, reminder and generic routing")

    monkeypatch.setattr(fake_ai, "route_message", forbidden_generic)
    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            "Фокус на неделю",
            update_id=31_030 + telegram_id,
            source_message_id=92_030,
        )
    )

    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert (
        await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
    ).session is None
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    if durable_owner == "onboarding":
        async with db.sessions() as session:
            state = await session.scalar(
                select(OnboardingState).where(OnboardingState.user_id == owner.id)
            )
        assert state is not None
        assert state.current_step == 3
        assert state.answers["future_life"] == "Фокус на неделю"
    else:
        pending = await core.workspace_service.pending_input(owner.id, telegram_id)
        assert pending is not None
        assert pending.action == "input:create_name"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.parametrize("owner_kind", ["onboarding", "workspace"])
async def test_scheduled_weekly_waits_for_real_reply_keyboard_owner_acquisition(
    db,
    fake_ai,
    monkeypatch,
    owner_kind,
):
    telegram_id = 715_110 if owner_kind == "onboarding" else 715_111
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
            enable_workspace_access=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=owner_kind != "onboarding",
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    owner_locked = asyncio.Event()
    owner_release = asyncio.Event()

    if owner_kind == "onboarding":
        original_ask = core._ask_question

        async def blocked_ask(update, step):
            owner_locked.set()
            await owner_release.wait()
            await original_ask(update, step)

        monkeypatch.setattr(core, "_ask_question", blocked_ask)
        owner_task = asyncio.create_task(
            application.process_update(
                _runtime_weekly_command_update(
                    application,
                    telegram_id,
                    "/start",
                    update_id=31_110,
                    source_message_id=92_110,
                )
            )
        )
    else:
        original_begin = core.workspace_service.begin_input

        async def blocked_begin(*args, **kwargs):
            owner_locked.set()
            await owner_release.wait()
            return await original_begin(*args, **kwargs)

        monkeypatch.setattr(core.workspace_service, "begin_input", blocked_begin)
        owner_task = asyncio.create_task(
            core._begin_workspace_input(
                owner.id,
                telegram_id,
                "input_invite_recipient",
                payload={"request_id": 51},
            )
        )

    delivery = None
    try:
        await asyncio.wait_for(owner_locked.wait(), timeout=10)
        delivery = asyncio.create_task(
            core.weekly_review_scheduled_notification(
                application.bot,
                telegram_id,
                owner.timezone,
            )
        )
        await asyncio.sleep(0)
        assert not delivery.done()
        assert all("Обзор недели" not in str(entry.get("text", "")) for entry in transport.sent)
        owner_release.set()
        await asyncio.wait_for(asyncio.gather(owner_task, delivery), timeout=10)
    finally:
        owner_release.set()
        pending = [task for task in (owner_task, delivery) if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)

    assert all("Обзор недели" not in str(entry.get("text", "")) for entry in transport.sent)
    assert all("Обзор недели" not in str(entry.get("text", "")) for entry in transport.edits)
    assert transport.deletes == []
    if owner_kind == "onboarding":
        async with db.sessions() as session:
            state = await session.scalar(
                select(OnboardingState).where(OnboardingState.user_id == owner.id)
            )
        assert state is not None
        assert state.status == "in_progress"
    else:
        pending_owner = await core.workspace_service.pending_input(owner.id, telegram_id)
        assert pending_owner is not None
        assert pending_owner.action == "input:invite_recipient"
    assert (
        await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
    ).session is None
    assert core._weekly_review_tasks == set()


@pytest.mark.parametrize("owner_kind", ["onboarding", "workspace"])
async def test_scheduled_weekly_post_send_owner_mismatch_neutralizes_exact_message(
    db,
    fake_ai,
    monkeypatch,
    owner_kind,
):
    telegram_id = 715_112 if owner_kind == "onboarding" else 715_113
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
            enable_workspace_access=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=owner_kind != "onboarding",
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    plain_send = ExtBot.send_message
    owner_published = asyncio.Event()

    async def send_then_publish_owner(self, *args, **kwargs):
        sent = await plain_send(self, *args, **kwargs)
        if owner_kind == "onboarding":
            async with db.session() as session:
                session.add(
                    OnboardingState(
                        user_id=owner.id,
                        current_step=0,
                        answers={},
                        status="in_progress",
                    )
                )
        else:
            # This direct durable write models another process, outside the
            # process-local reply-keyboard lock, winning after Telegram accepts.
            await core.workspace_service.begin_input(
                owner.id,
                telegram_id,
                "invite_recipient",
                payload={"request_id": 52},
            )
        owner_published.set()
        return sent

    monkeypatch.setattr(ExtBot, "send_message", send_then_publish_owner)

    await core.weekly_review_scheduled_notification(
        application.bot,
        telegram_id,
        owner.timezone,
    )
    await asyncio.sleep(0)

    assert owner_published.is_set()
    assert len(transport.sent) == 1
    assert isinstance(transport.sent[0]["reply_markup"], ReplyKeyboardRemove)
    sent = transport.sent_messages[0]
    assert [(entry["chat_id"], entry["message_id"]) for entry in transport.deletes] == [
        (telegram_id, sent.message_id)
    ]
    assert transport.edits == []
    if owner_kind == "onboarding":
        async with db.sessions() as session:
            state = await session.scalar(
                select(OnboardingState).where(OnboardingState.user_id == owner.id)
            )
        assert state is not None
        assert state.status == "in_progress"
    else:
        pending_owner = await core.workspace_service.pending_input(owner.id, telegram_id)
        assert pending_owner is not None
        assert pending_owner.action == "input:invite_recipient"
    assert (
        await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
    ).session is None
    assert core.weekly_review_capabilities._capabilities == {}
    assert core._weekly_review_tasks == set()


@pytest.mark.parametrize("blocking_owner", ["onboarding", "workspace", "memory", "reminder"])
async def test_real_application_sessionless_weekly_callback_defers_without_spending_token(
    db,
    fake_ai,
    monkeypatch,
    blocking_owner,
):
    telegram_id = {
        "onboarding": 715_012,
        "workspace": 715_013,
        "memory": 715_014,
        "reminder": 715_015,
    }[blocking_owner]
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
            enable_workspace_access=True,
            enable_nova_memory=True,
            nova_memory_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=blocking_owner != "onboarding",
    )
    memory_session = None
    reminder_session = None
    if blocking_owner == "onboarding":
        async with db.session() as session:
            session.add(
                OnboardingState(
                    user_id=owner.id,
                    current_step=2,
                    answers={"display_name": "Варвара"},
                    status="in_progress",
                )
            )
    elif blocking_owner == "workspace":
        await core.workspace_service.begin_input(
            owner.id,
            telegram_id,
            "create_name",
            payload={"character": "family"},
        )
    elif blocking_owner == "memory":
        memory_session = await _runtime_active_memory(core, owner, telegram_id, telegram_id)
    else:
        reminder_session = await core.reminder_sessions.create(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
            access_version=owner.access_version,
            title="Позвонить врачу",
            schedule_kind=ReminderScheduleKind.ONCE,
            local_date=None,
            local_time=None,
            timezone=owner.timezone,
            timezone_source=ReminderTimezoneSource.PROFILE,
            phase=ReminderFlowPhase.WHEN,
            canonical_message_id=91_015,
        )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    canonical = transport.make_message(telegram_id, 91_012, "weekly scheduled nudge")
    scheduled_week = core.weekly_review_service.target_week_start(
        owner.timezone,
        scheduled=True,
        review_weekday=6,
    )
    tokens = await core.weekly_review_capabilities.issue(
        actions=("start",),
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
        canonical_message_id=canonical.message_id,
        access_version=owner.access_version,
        week_start=scheduled_week,
        scheduled=True,
    )
    callback_data = f"wrev:{tokens['start']}"
    telegram_user = TelegramUser(telegram_id, "Варвара", False)
    blocked_update, blocked_query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        canonical,
        callback_data,
        update_id=31_035,
    )

    await application.process_update(blocked_update)

    blocked = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert blocked.session is None
    assert [entry["callback_query_id"] for entry in transport.answers] == [blocked_query.id]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0

    if blocking_owner == "onboarding":
        async with db.session() as session:
            stored = await session.get(User, owner.id)
            state = await session.scalar(
                select(OnboardingState).where(OnboardingState.user_id == owner.id)
            )
            assert stored is not None and state is not None
            stored.onboarding_completed = True
            state.status = "completed"
    elif blocking_owner == "workspace":
        assert await core.workspace_service.cancel_input(owner.id, telegram_id)
    elif blocking_owner == "memory":
        assert memory_session is not None
        assert await core.nova_memory_clear_bound(
            owner.id,
            telegram_id,
            telegram_id,
            session_id=memory_session.id,
        )
    else:
        assert reminder_session is not None
        assert await core.reminder_sessions.clear(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
            session_id=reminder_session.id,
        )

    retry_update, retry_query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        canonical,
        callback_data,
        update_id=31_036,
    )
    await application.process_update(retry_update)

    current = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.status == "found"
    assert current.session is not None
    assert current.session.week_start == scheduled_week
    assert current.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert current.session.canonical_message_id == canonical.message_id
    assert str(transport.edits[-1]["text"]).endswith(WEEKLY_REVIEW_QUESTION)
    assert [entry["callback_query_id"] for entry in transport.answers] == [
        blocked_query.id,
        retry_query.id,
    ]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize("race", ["replacement", "access_bounce"])
async def test_real_application_weekly_voice_post_stt_race_is_fail_closed(
    db,
    fake_ai,
    monkeypatch,
    race,
):
    telegram_id = 715_020 if race == "replacement" else 715_021
    transcript = "Завтра в 15:05 сказать Назару, что я люблю его"
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingTranscription(RuntimeTranscription):
        async def transcribe(self, audio: bytes, filename: str) -> str:
            self.calls.append((audio, filename))
            started.set()
            await release.wait()
            return self.transcript

    transcription = BlockingTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    frozen = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=91_020,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert frozen.session is not None
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_040 + telegram_id,
        source_message_id=92_040,
        progress_message_id=93_040,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    deletes: list[dict[str, object]] = []
    downstream: list[int] = []

    async def fake_delete_message(self, *args, **kwargs):
        del self, args
        deletes.append(dict(kwargs))
        return True

    async def downstream_probe(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "delete_message", fake_delete_message)
    application.add_handler(TypeHandler(Update, downstream_probe), group=100)
    processing = asyncio.create_task(application.process_update(update))
    replacement = None
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        if race == "replacement":
            replacement_result = await core.weekly_review_service.create_session(
                telegram_actor_id=telegram_id,
                chat_id=telegram_id,
                expected_access_version=owner.access_version,
                canonical_message_id=91_021,
                phase=WeeklyReviewPhase.ROOT,
            )
            replacement = replacement_result.session
            assert replacement is not None
        else:
            access = AccessService(db)
            await access.set_guest(telegram_id, source="weekly-runtime-race")
            await access.grant_subscriber(telegram_id, source="weekly-runtime-race")
        release.set()
        await asyncio.wait_for(processing, timeout=10)
    finally:
        release.set()
        if not processing.done():
            try:
                await asyncio.wait_for(asyncio.shield(processing), timeout=10)
            except TimeoutError:
                processing.cancel()
        await asyncio.gather(processing, return_exceptions=True)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert downstream == []
    assert [entry["text"] for entry in sent] == ["Расшифровываю голосовую мысль…"]
    assert [(entry["chat_id"], entry["message_id"]) for entry in deletes] == [
        (telegram_id, progress.message_id)
    ]
    if race == "replacement":
        current = await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
        assert current.status == "found"
        assert current.session == replacement
        assert current.session is not None
        assert current.session.phase is WeeklyReviewPhase.ROOT
        assert current.session.canonical_message_id == 91_021
        assert edits == []
    else:
        refreshed = await core._user(telegram_id)
        current = await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=refreshed.access_version,
        )
        assert current.session is None
        assert edits
        assert all(entry["message_id"] == frozen.session.canonical_message_id for entry in edits)
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


async def test_real_application_weekly_voice_initial_session_lookup_error_is_fail_closed(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_022
    transcription = RuntimeTranscription("Завтра в 15:05 сказать Назару, что я люблю его")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    frozen = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=91_022,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert frozen.session is not None
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_062,
        source_message_id=92_062,
        progress_message_id=93_062,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    downstream: list[int] = []
    original_current = core.weekly_review_service.current_session

    async def failed_current(**kwargs):
        del kwargs
        raise RuntimeError("private-weekly-lookup")

    async def downstream_probe(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(core.weekly_review_service, "current_session", failed_current)
    application.add_handler(TypeHandler(Update, downstream_probe), group=100)
    await application.process_update(update)

    assert transcription.calls == []
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert downstream == []
    assert sent == []
    assert edits == []
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    current = await original_current(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.session == frozen.session
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


async def test_weekly_maintenance_consecutive_batches_drain_more_than_limit(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    expired_remaining = 237
    processing_remaining = 205
    expired_batches: list[int] = []
    processing_batches: list[int] = []
    rendered: list[int] = []

    async def cleanup_expired(**kwargs):
        assert kwargs == {"allowed_tiers": FULL_ACCESS_TIERS}
        nonlocal expired_remaining
        batch = min(100, expired_remaining)
        expired_remaining -= batch
        expired_batches.append(batch)
        return batch

    async def recover_processing(**kwargs):
        assert kwargs == {"allowed_tiers": FULL_ACCESS_TIERS}
        nonlocal processing_remaining
        batch = min(100, processing_remaining)
        processing_remaining -= batch
        processing_batches.append(batch)
        return tuple(
            SimpleNamespace(canonical_message_id=95_000 + len(rendered) + index)
            for index in range(batch)
        )

    async def no_markup(session):
        del session
        return None

    async def record_render(context, session, text_value, markup):
        del context, text_value, markup
        rendered.append(session.canonical_message_id)
        return True

    monkeypatch.setattr(core.weekly_review_service, "cleanup_expired", cleanup_expired)
    monkeypatch.setattr(
        core.weekly_review_service,
        "recover_processing_session_snapshots",
        recover_processing,
    )
    monkeypatch.setattr(core, "_weekly_review_cancel_markup", no_markup)
    monkeypatch.setattr(core, "_weekly_review_render", record_render)
    monkeypatch.setattr(core, "_weekly_review_week_heading", lambda session: "🧭 Неделя")

    for _ in range(4):
        await core._maintain_weekly_review_state(
            SimpleNamespace(),
            startup_recovery=True,
        )

    assert expired_batches == [100, 100, 37, 0]
    assert processing_batches == [100, 100, 5, 0]
    assert expired_remaining == processing_remaining == 0
    assert len(rendered) == 205
    assert len(set(rendered)) == 205
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []


async def test_weekly_maintenance_recovers_processing_to_retry_without_provider(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_030
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    created = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=95_030,
        phase=WeeklyReviewPhase.PROCESSING,
    )
    assert created.session is not None
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    transport.make_message(telegram_id, created.session.canonical_message_id, "processing")

    await core._maintain_weekly_review_state(
        application.bot,
        startup_recovery=True,
    )

    recovered = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert recovered.status == "found"
    assert recovered.session is not None
    assert recovered.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert recovered.session.version == created.session.version + 1
    assert transport.edits[-1]["message_id"] == created.session.canonical_message_id
    assert "Ничего не сохранено — отправь его ещё раз" in str(transport.edits[-1]["text"])
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    edit_count = len(transport.edits)

    await core._maintain_weekly_review_state(
        application.bot,
        startup_recovery=True,
    )

    assert len(transport.edits) == edit_count
    assert fake_ai.weekly_review_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


async def test_periodic_weekly_maintenance_does_not_reset_live_provider_generation(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_031
    private_input = "PRIVATE_WEEKLY_LIVE_PROVIDER_" + ("x" * 180)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    created = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=95_031,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    transport.make_message(telegram_id, created.session.canonical_message_id, "awaiting")
    original_recovery = core.weekly_review_service.recover_processing_session_snapshots
    recovery_calls: list[dict[str, object]] = []

    async def record_recovery(**kwargs):
        recovery_calls.append(dict(kwargs))
        return await original_recovery(**kwargs)

    monkeypatch.setattr(
        core.weekly_review_service,
        "recover_processing_session_snapshots",
        record_recovery,
    )
    fake_ai.weekly_review_release.clear()
    processing = asyncio.create_task(
        application.process_update(
            _runtime_weekly_text_update(
                application,
                telegram_id,
                private_input,
                update_id=31_053,
                source_message_id=96_031,
            )
        )
    )
    try:
        await asyncio.wait_for(fake_ai.weekly_review_started.wait(), timeout=2)
        frozen = await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
        assert frozen.status == "found"
        assert frozen.session is not None
        assert frozen.session.phase is WeeklyReviewPhase.PROCESSING
        assert frozen.session.version == created.session.version + 1
        edit_count = len(transport.edits)

        maintenance_started = datetime.now(UTC)
        await core._maintain_weekly_review_state(
            application.bot,
            startup_recovery=False,
        )
        maintenance_finished = datetime.now(UTC)

        still_processing = await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
        assert still_processing.session == frozen.session
        assert not processing.done()
        assert len(transport.edits) == edit_count
        assert len(recovery_calls) == 1
        assert "now" not in recovery_calls[0]
        recovery_cutoff = recovery_calls[0]["updated_before"]
        assert isinstance(recovery_cutoff, datetime)
        assert maintenance_started.timestamp() - 36 <= recovery_cutoff.timestamp()
        assert recovery_cutoff.timestamp() <= maintenance_finished.timestamp() - 34
        assert recovery_calls[0]["allowed_tiers"] == FULL_ACCESS_TIERS

        fake_ai.weekly_review_release.set()
        await asyncio.wait_for(processing, timeout=2)
    finally:
        fake_ai.weekly_review_release.set()
        if not processing.done():
            await asyncio.gather(processing, return_exceptions=True)
        await asyncio.gather(*tuple(core._weekly_review_tasks), return_exceptions=True)
    await asyncio.sleep(0)

    preview = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert preview.status == "found"
    assert preview.session is not None
    assert preview.session.phase is WeeklyReviewPhase.PREVIEW
    assert preview.session.version == frozen.session.version + 1
    assert [call[0] for call in fake_ai.weekly_review_calls] == [private_input]
    assert core._weekly_review_tasks == set()
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize("checkpoint", ["provider", "post_send"])
async def test_real_application_today_access_bounce_is_fail_closed(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    checkpoint,
):
    telegram_id = 715_040 if checkpoint == "provider" else 715_041
    private_focus = f"PRIVATE_TODAY_{checkpoint.upper()}_FOCUS"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    await _runtime_weekly_focus(core, db, owner, private_focus)
    plan = await fake_ai.make_today_plan({})
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    checkpoint_started = asyncio.Event()
    checkpoint_release = asyncio.Event()
    provider_calls = 0

    async def blocked_generate(snapshot):
        nonlocal provider_calls
        assert snapshot.actor_id == owner.id
        assert snapshot.weekly_focus == private_focus
        provider_calls += 1
        if checkpoint == "provider":
            checkpoint_started.set()
            await checkpoint_release.wait()
        return plan

    monkeypatch.setattr(core.focus_service, "generate_today_plan", blocked_generate)
    if checkpoint == "post_send":

        async def blocked_send(self, *args, **kwargs):
            del self, args
            record = dict(kwargs)
            transport.sent.append(record)
            message = transport.make_message(
                int(record["chat_id"]),
                96_041,
                str(record.get("text", "")),
            )
            transport.sent_messages.append(message)
            checkpoint_started.set()
            await checkpoint_release.wait()
            return message

        monkeypatch.setattr(ExtBot, "send_message", blocked_send)
    update = _runtime_weekly_command_update(
        application,
        telegram_id,
        "/today",
        update_id=31_050,
        source_message_id=96_040,
    )

    with caplog.at_level(logging.WARNING):
        processing = asyncio.create_task(application.process_update(update))
        await asyncio.wait_for(checkpoint_started.wait(), timeout=2)
        access = AccessService(db)
        await access.set_guest(telegram_id, source="today-runtime-race")
        await access.grant_subscriber(telegram_id, source="today-runtime-race")
        checkpoint_release.set()
        await asyncio.wait_for(processing, timeout=2)

    refreshed = await core._user(telegram_id)
    assert refreshed.access_tier == "subscriber"
    assert refreshed.access_version == owner.access_version + 2
    assert provider_calls == 1
    if checkpoint == "provider":
        assert transport.sent == []
        assert transport.deletes == []
        assert transport.edits == []
    else:
        assert len(transport.sent) == 1
        assert private_focus in str(transport.sent[0]["text"])
        assert [(entry["chat_id"], entry["message_id"]) for entry in transport.deletes] == [
            (telegram_id, 96_041)
        ]
        assert transport.edits == []
    assert private_focus not in caplog.text
    assert str(telegram_id) not in caplog.text


async def test_real_application_today_pre_provider_fence_error_sends_nothing(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    telegram_id = 715_044
    private_error = "PRIVATE_TODAY_PRE_PROVIDER_FENCE_ERROR"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    provider_calls = 0

    async def failed_check(snapshot):
        del snapshot
        raise RuntimeError(private_error)

    async def forbidden_provider(snapshot):
        nonlocal provider_calls
        del snapshot
        provider_calls += 1
        raise AssertionError("provider must not run after a failed pre-provider fence")

    monkeypatch.setattr(core.focus_service, "check_today_application", failed_check)
    monkeypatch.setattr(core.focus_service, "generate_today_plan", forbidden_provider)
    update = _runtime_weekly_command_update(
        application,
        telegram_id,
        "/today",
        update_id=31_054,
        source_message_id=96_044,
    )

    with caplog.at_level(logging.WARNING):
        await application.process_update(update)

    assert provider_calls == 0
    assert transport.sent == []
    assert transport.edits == []
    assert transport.deletes == []
    assert private_error not in caplog.text
    assert str(telegram_id) not in caplog.text


async def test_real_application_today_outer_cancel_keeps_shielded_lifecycle_running(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_042
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    await _runtime_weekly_focus(core, db, owner, "Сохраняемый фокус недели")
    plan = await fake_ai.make_today_plan({})
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()

    async def blocked_generate(snapshot):
        assert snapshot.actor_id == owner.id
        provider_started.set()
        await provider_release.wait()
        return plan

    monkeypatch.setattr(core.focus_service, "generate_today_plan", blocked_generate)
    update = _runtime_weekly_command_update(
        application,
        telegram_id,
        "/today",
        update_id=31_051,
        source_message_id=96_042,
    )
    outer = asyncio.create_task(application.process_update(update))
    await asyncio.wait_for(provider_started.wait(), timeout=2)
    inner = next(
        task
        for task in core._weekly_review_tasks
        if task.get_name() == "weekly-review-today-lifecycle"
    )

    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer

    assert not inner.done()
    assert transport.sent == []
    provider_release.set()
    await asyncio.wait_for(inner, timeout=2)
    await asyncio.sleep(0)

    assert len(transport.sent) == 1
    assert "Сохраняемый фокус недели" in str(transport.sent[0]["text"])
    assert transport.sent_messages[0].message_id > 0
    assert transport.deletes == []
    assert core._weekly_review_tasks == set()


async def test_real_application_today_direct_telegram_cancel_is_not_delivery(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_043
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    await _runtime_weekly_focus(core, db, owner, "Недоставленный фокус недели")
    plan = await fake_ai.make_today_plan({})
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    send_attempts = 0

    async def generate(snapshot):
        assert snapshot.actor_id == owner.id
        return plan

    async def cancelled_send(self, *args, **kwargs):
        nonlocal send_attempts
        del self, args, kwargs
        send_attempts += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(core.focus_service, "generate_today_plan", generate)
    monkeypatch.setattr(ExtBot, "send_message", cancelled_send)
    update = _runtime_weekly_command_update(
        application,
        telegram_id,
        "/today",
        update_id=31_052,
        source_message_id=96_043,
    )

    with pytest.raises(asyncio.CancelledError):
        await application.process_update(update)
    await asyncio.sleep(0)

    assert send_attempts == 1
    assert transport.sent == []
    assert transport.sent_messages == []
    assert transport.deletes == []
    assert transport.edits == []
    assert core._weekly_review_tasks == set()


@pytest.mark.parametrize("surface", ["command", "callback"])
@pytest.mark.parametrize(
    "race",
    [
        "downgrade",
        "bounce",
        "status_error",
        "timezone",
        "week",
        "focus_edit",
        "focus_delete",
        "focus_create",
    ],
)
async def test_real_application_today_surfaces_fail_closed_before_provider(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    surface,
    race,
):
    private_error = "PRIVATE_TODAY_PRE_PROVIDER_STATUS_ERROR"
    env = await _runtime_prepare_today_case(
        db,
        fake_ai,
        monkeypatch,
        surface=surface,
        with_focus=race != "focus_create",
        telegram_id=716_100,
    )
    clock = {"advanced": False}
    base_now = datetime.now(UTC)

    def frozen_clock(value):
        del value
        return base_now + (timedelta(days=8) if clock["advanced"] else timedelta())

    monkeypatch.setattr(env.core.focus_service, "_utc_now", frozen_clock)
    original_materialize = env.core.focus_service.materialize_today_application

    async def raced_materialize(user_id, *, include_weekly_focus, now=None):
        snapshot = await original_materialize(
            user_id,
            include_weekly_focus=include_weekly_focus,
            now=now,
        )
        await _runtime_apply_today_race(
            db,
            env.owner,
            race,
            week_start=snapshot.local_week_start,
            advance_clock=lambda: clock.__setitem__("advanced", True),
        )
        return snapshot

    if race == "status_error":

        async def failed_check(snapshot):
            del snapshot
            raise RuntimeError(private_error)

        monkeypatch.setattr(env.core.focus_service, "check_today_application", failed_check)
    else:
        monkeypatch.setattr(
            env.core.focus_service,
            "materialize_today_application",
            raced_materialize,
        )
    provider_calls = 0
    original_generate = env.core.focus_service.generate_today_plan

    async def forbidden_generate(snapshot):
        nonlocal provider_calls
        provider_calls += 1
        return await original_generate(snapshot)

    monkeypatch.setattr(env.core.focus_service, "generate_today_plan", forbidden_generate)

    with caplog.at_level(logging.WARNING):
        await env.application.process_update(env.update)
    await asyncio.sleep(0)

    assert provider_calls == 0
    assert env.transport.sent == []
    assert env.transport.deletes == []
    if env.query is None:
        assert env.transport.edits == []
    else:
        assert [entry["callback_query_id"] for entry in env.transport.answers] == [env.query.id]
        await _assert_runtime_today_callback_terminal(
            env,
            edit_count=1,
            access_changed=race in {"downgrade", "bounce"},
        )
    assert private_error not in caplog.text
    assert "PRIVATE_FROZEN_WEEKLY_FOCUS" not in caplog.text
    assert str(env.owner.telegram_id) not in caplog.text
    assert env.core._weekly_review_tasks == set()


@pytest.mark.parametrize("surface", ["command", "callback"])
@pytest.mark.parametrize(
    "race",
    [
        "downgrade",
        "bounce",
        "timezone",
        "week",
        "focus_edit",
        "focus_delete",
        "focus_create",
    ],
)
async def test_real_application_today_surfaces_discard_provider_result_after_race(
    db,
    fake_ai,
    monkeypatch,
    surface,
    race,
):
    env = await _runtime_prepare_today_case(
        db,
        fake_ai,
        monkeypatch,
        surface=surface,
        with_focus=race != "focus_create",
        telegram_id=716_110,
    )
    clock = {"advanced": False}
    base_now = datetime.now(UTC)

    def frozen_clock(value):
        del value
        return base_now + (timedelta(days=8) if clock["advanced"] else timedelta())

    monkeypatch.setattr(env.core.focus_service, "_utc_now", frozen_clock)
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()
    provider_snapshots = []
    original_generate = env.core.focus_service.generate_today_plan

    async def blocked_generate(snapshot):
        provider_snapshots.append(snapshot)
        provider_started.set()
        await provider_release.wait()
        return await original_generate(snapshot)

    monkeypatch.setattr(env.core.focus_service, "generate_today_plan", blocked_generate)
    processing = asyncio.create_task(env.application.process_update(env.update))
    try:
        await asyncio.wait_for(provider_started.wait(), timeout=10)
        assert len(provider_snapshots) == 1
        snapshot = provider_snapshots[0]
        await _runtime_apply_today_race(
            db,
            env.owner,
            race,
            week_start=snapshot.local_week_start,
            advance_clock=lambda: clock.__setitem__("advanced", True),
        )
        provider_release.set()
        await asyncio.wait_for(processing, timeout=10)
    finally:
        provider_release.set()
        if not processing.done():
            await asyncio.gather(processing, return_exceptions=True)
    await asyncio.sleep(0)

    assert len(provider_snapshots) == 1
    assert provider_snapshots[0].includes_weekly_focus is True
    assert fake_ai.last_today_context is not None
    assert "weekly_focus" in fake_ai.last_today_context
    if race != "focus_create":
        assert fake_ai.last_today_context["weekly_focus"] == "PRIVATE_FROZEN_WEEKLY_FOCUS"
    else:
        assert fake_ai.last_today_context["weekly_focus"] is None
    assert env.transport.sent == []
    assert env.transport.deletes == []
    if env.query is None:
        assert env.transport.edits == []
    else:
        assert [entry["callback_query_id"] for entry in env.transport.answers] == [env.query.id]
        await _assert_runtime_today_callback_terminal(
            env,
            edit_count=1,
            access_changed=race in {"downgrade", "bounce"},
        )
    assert env.core._weekly_review_tasks == set()


@pytest.mark.parametrize("surface", ["command", "callback"])
@pytest.mark.parametrize("race", ["bounce", "timezone", "week", "focus_delete"])
async def test_real_application_today_surfaces_compensate_exact_accepted_telegram_output(
    db,
    fake_ai,
    monkeypatch,
    surface,
    race,
):
    env = await _runtime_prepare_today_case(
        db,
        fake_ai,
        monkeypatch,
        surface=surface,
        with_focus=True,
        telegram_id=716_120,
    )
    clock = {"advanced": False}
    base_now = datetime.now(UTC)

    def frozen_clock(value):
        del value
        return base_now + (timedelta(days=8) if clock["advanced"] else timedelta())

    monkeypatch.setattr(env.core.focus_service, "_utc_now", frozen_clock)
    provider_snapshots = []
    original_generate = env.core.focus_service.generate_today_plan

    async def record_generate(snapshot):
        provider_snapshots.append(snapshot)
        return await original_generate(snapshot)

    monkeypatch.setattr(env.core.focus_service, "generate_today_plan", record_generate)
    telegram_accepted = asyncio.Event()
    telegram_release = asyncio.Event()
    accepted_message_id = 97_120
    if surface == "command":

        async def blocked_send(self, *args, **kwargs):
            del self, args
            record = dict(kwargs)
            env.transport.sent.append(record)
            message = env.transport.make_message(
                int(record["chat_id"]),
                accepted_message_id,
                str(record.get("text", "")),
            )
            env.transport.sent_messages.append(message)
            telegram_accepted.set()
            await telegram_release.wait()
            return message

        monkeypatch.setattr(ExtBot, "send_message", blocked_send)
    else:
        first_edit = True

        async def blocked_edit(self, *args, **kwargs):
            nonlocal first_edit
            del self, args
            record = dict(kwargs)
            env.transport.edits.append(record)
            message = env.transport.make_message(
                int(record["chat_id"]),
                int(record["message_id"]),
                str(record.get("text", "")),
            )
            if first_edit:
                first_edit = False
                telegram_accepted.set()
                await telegram_release.wait()
            return message

        monkeypatch.setattr(ExtBot, "edit_message_text", blocked_edit)
    processing = asyncio.create_task(env.application.process_update(env.update))
    try:
        await asyncio.wait_for(telegram_accepted.wait(), timeout=10)
        assert len(provider_snapshots) == 1
        await _runtime_apply_today_race(
            db,
            env.owner,
            race,
            week_start=provider_snapshots[0].local_week_start,
            advance_clock=lambda: clock.__setitem__("advanced", True),
        )
        telegram_release.set()
        await asyncio.wait_for(processing, timeout=10)
    finally:
        telegram_release.set()
        if not processing.done():
            await asyncio.gather(processing, return_exceptions=True)
    await asyncio.sleep(0)

    assert len(provider_snapshots) == 1
    if surface == "command":
        assert len(env.transport.sent) == 1
        assert "PRIVATE_FROZEN_WEEKLY_FOCUS" in str(env.transport.sent[0]["text"])
        assert [(entry["chat_id"], entry["message_id"]) for entry in env.transport.deletes] == [
            (env.owner.telegram_id, accepted_message_id)
        ]
        assert env.transport.edits == []
    else:
        assert env.query is not None
        assert [entry["callback_query_id"] for entry in env.transport.answers] == [env.query.id]
        assert len(env.transport.edits) == 3
        assert "PRIVATE_FROZEN_WEEKLY_FOCUS" in str(env.transport.edits[0]["text"])
        assert env.transport.edits[1]["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
        assert {entry["message_id"] for entry in env.transport.edits} == {97_101}
        await _assert_runtime_today_callback_terminal(
            env,
            edit_count=3,
            access_changed=race == "bounce",
        )
        assert env.transport.sent == []
        assert env.transport.deletes == []
    assert env.core._weekly_review_tasks == set()


@pytest.mark.parametrize("surface", ["command", "callback"])
@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_real_application_today_final_fence_failure_compensates_exact_output(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    surface,
    failure,
):
    private_error = "PRIVATE_TODAY_FINAL_FENCE_FAILURE"
    env = await _runtime_prepare_today_case(
        db,
        fake_ai,
        monkeypatch,
        surface=surface,
        with_focus=True,
        telegram_id=716_130,
    )
    provider_calls = 0
    original_generate = env.core.focus_service.generate_today_plan

    async def record_generate(snapshot):
        nonlocal provider_calls
        provider_calls += 1
        return await original_generate(snapshot)

    monkeypatch.setattr(env.core.focus_service, "generate_today_plan", record_generate)
    check_calls = 0
    final_call = 3 if surface == "command" else 4
    original_check = env.core.focus_service.check_today_application

    async def fail_final_check(snapshot):
        nonlocal check_calls
        check_calls += 1
        if check_calls == final_call:
            if failure == "cancel":
                raise asyncio.CancelledError(private_error)
            raise RuntimeError(private_error)
        return await original_check(snapshot)

    monkeypatch.setattr(env.core.focus_service, "check_today_application", fail_final_check)
    with caplog.at_level(logging.WARNING):
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await env.application.process_update(env.update)
        else:
            await env.application.process_update(env.update)
    await asyncio.sleep(0)

    assert provider_calls == 1
    assert check_calls == final_call
    if surface == "command":
        assert len(env.transport.sent) == 1
        sent_message = env.transport.sent_messages[0]
        assert [(entry["chat_id"], entry["message_id"]) for entry in env.transport.deletes] == [
            (env.owner.telegram_id, sent_message.message_id)
        ]
        assert env.transport.edits == []
    else:
        assert env.query is not None
        assert [entry["callback_query_id"] for entry in env.transport.answers] == [env.query.id]
        assert len(env.transport.edits) == 3
        assert env.transport.edits[1]["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
        assert {entry["message_id"] for entry in env.transport.edits} == {97_101}
        await _assert_runtime_today_callback_terminal(
            env,
            edit_count=3,
            access_changed=False,
        )
        assert env.transport.sent == []
        assert env.transport.deletes == []
    assert private_error not in caplog.text
    assert "PRIVATE_FROZEN_WEEKLY_FOCUS" not in caplog.text
    assert str(env.owner.telegram_id) not in caplog.text
    assert env.core._weekly_review_tasks == set()


@pytest.mark.parametrize(
    ("enabled", "admin_only"),
    [(False, False), (True, True)],
    ids=["disabled", "admin-only-subscriber"],
)
async def test_real_application_today_policy_denial_uses_legacy_context_without_weekly_sql(
    db,
    fake_ai,
    monkeypatch,
    enabled,
    admin_only,
):
    telegram_id = 716_140
    private_focus = "PRIVATE_LEGACY_WEEKLY_FOCUS"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=enabled,
            weekly_review_admin_only=admin_only,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    await _runtime_weekly_focus(core, db, owner, private_focus)
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    provider_calls = 0
    provider_snapshots = []
    original_generate = core.focus_service.generate_today_plan

    async def record_generate(snapshot):
        nonlocal provider_calls
        provider_calls += 1
        provider_snapshots.append(snapshot)
        return await original_generate(snapshot)

    monkeypatch.setattr(core.focus_service, "generate_today_plan", record_generate)
    statements: list[str] = []

    def record_sql(conn, cursor, statement, parameters, context, executemany):
        del conn, cursor, parameters, context, executemany
        statements.append(str(statement).casefold())

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_sql)
    try:
        await application.process_update(
            _runtime_weekly_command_update(
                application,
                telegram_id,
                "/today",
                update_id=32_140,
                source_message_id=97_140,
            )
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_sql)
    await asyncio.sleep(0)

    assert provider_calls == 1
    assert len(provider_snapshots) == 1
    assert provider_snapshots[0].includes_weekly_focus is False
    assert provider_snapshots[0].weekly_focus is None
    assert fake_ai.last_today_context is not None
    assert "weekly_focus" not in fake_ai.last_today_context
    assert len(transport.sent) == 1
    assert private_focus not in str(transport.sent[0]["text"])
    assert transport.edits == []
    assert transport.deletes == []
    assert not any(
        table in statement
        for statement in statements
        for table in (
            "weekly_focuses",
            "weekly_focus_changes",
            "weekly_review_sessions",
        )
    )
    assert core._weekly_review_tasks == set()


async def test_weekly_shutdown_repeats_cancel_and_leaves_unrelated_task_running(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    started = asyncio.Event()
    first_cancel = asyncio.Event()
    second_cancel = asyncio.Event()
    blocker = asyncio.Event()
    unrelated_release = asyncio.Event()

    async def cancellation_suppressor() -> None:
        started.set()
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            first_cancel.set()
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                second_cancel.set()
                raise

    tracked = asyncio.create_task(
        cancellation_suppressor(),
        name="weekly-review-delivery-lifecycle",
    )
    core._weekly_review_tasks.add(tracked)
    unrelated = asyncio.create_task(unrelated_release.wait(), name="unrelated-runtime-task")
    monkeypatch.setattr(bot_module, "_WEEKLY_REVIEW_DRAIN_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(bot_module, "_WEEKLY_REVIEW_CANCEL_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(bot_module, "_WEEKLY_REVIEW_CANCEL_RETRY_SECONDS", 0.01)
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.wait_for(core._drain_weekly_review_tasks(), timeout=2)
        assert first_cancel.is_set()
        assert second_cancel.is_set()
        assert tracked.cancelled()
        assert core._weekly_review_tasks == set()
        assert not unrelated.done()
    finally:
        blocker.set()
        unrelated_release.set()
        await asyncio.gather(tracked, unrelated, return_exceptions=True)


async def test_post_stop_redrains_weekly_delivery_spawned_during_maintenance_stop(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    first_empty_snapshot = asyncio.Event()
    delivery_started = asyncio.Event()
    delivery_release = asyncio.Event()
    maintenance_started = asyncio.Event()
    maintenance_cancelled = asyncio.Event()
    second_drain_started = asyncio.Event()
    unrelated_release = asyncio.Event()
    inner_tasks: list[asyncio.Task[bool]] = []
    drain_calls = 0

    async def blocked_delivery() -> bool:
        delivery_started.set()
        await delivery_release.wait()
        return True

    async def maintenance_producer() -> None:
        maintenance_started.set()
        await first_empty_snapshot.wait()
        inner = asyncio.create_task(
            blocked_delivery(),
            name="weekly-review-delivery-lifecycle",
        )
        inner_tasks.append(inner)
        core._weekly_review_track_task(inner)
        try:
            await asyncio.shield(inner)
        except asyncio.CancelledError:
            maintenance_cancelled.set()
            raise

    original_drain = core._drain_private_delivery_tasks

    async def coordinated_drain() -> None:
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 2:
            second_drain_started.set()
        await original_drain()
        if drain_calls == 1:
            first_empty_snapshot.set()
            await delivery_started.wait()

    monkeypatch.setattr(core, "_drain_private_delivery_tasks", coordinated_drain)
    maintenance = asyncio.create_task(
        maintenance_producer(),
        name="guest-result-maintenance",
    )
    core._guest_maintenance_task = maintenance
    unrelated = asyncio.create_task(
        unrelated_release.wait(),
        name="unrelated-runtime-task",
    )
    stopping: asyncio.Task[None] | None = None
    loop = asyncio.get_running_loop()
    prior_debug = loop.get_debug()
    prior_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        loop.set_debug(True)
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            await asyncio.wait_for(maintenance_started.wait(), timeout=1)
            stopping = asyncio.create_task(core._post_stop(SimpleNamespace()))
            await asyncio.wait_for(maintenance_cancelled.wait(), timeout=1)
            await asyncio.wait_for(second_drain_started.wait(), timeout=1)

            assert drain_calls == 2
            assert len(inner_tasks) == 1
            assert inner_tasks[0] in core._weekly_review_tasks
            assert not inner_tasks[0].done()
            assert not stopping.done()
            assert not unrelated.done()

            delivery_release.set()
            await asyncio.wait_for(stopping, timeout=1)

            assert maintenance.cancelled()
            assert core._guest_maintenance_task is None
            assert inner_tasks[0].result() is True
            assert core._weekly_review_tasks == set()
            assert drain_calls == 2
            assert not unrelated.done()

            await core._post_shutdown(SimpleNamespace())
            await core._post_shutdown(SimpleNamespace())
            assert core._guest_maintenance_task is None
            assert core._weekly_review_tasks == set()
            assert drain_calls == 6
            assert not unrelated.done()
        finally:
            delivery_release.set()
            unrelated_release.set()
            if stopping is not None and not stopping.done():
                stopping.cancel()
            if not maintenance.done():
                maintenance.cancel()
            await asyncio.gather(
                *(inner_tasks + [maintenance, unrelated]),
                *(() if stopping is None else (stopping,)),
                return_exceptions=True,
            )
            await asyncio.sleep(0)
            gc.collect()
            loop.set_exception_handler(prior_exception_handler)
            loop.set_debug(prior_debug)
    assert loop_errors == []
    assert [warning for warning in caught_warnings if warning.category is RuntimeWarning] == []


async def test_post_stop_stops_maintenance_and_redrains_when_initial_drain_fails(
    db,
    fake_ai,
    monkeypatch,
):
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    maintenance_started = asyncio.Event()
    maintenance_observed = asyncio.Event()
    drain_calls = 0

    class InitialDrainFailure(RuntimeError):
        pass

    failure = InitialDrainFailure("initial private delivery drain failed")
    original_drain = core._drain_private_delivery_tasks

    async def fail_once_then_drain() -> None:
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 1:
            raise failure
        await original_drain()

    async def maintenance_producer() -> None:
        maintenance_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            maintenance_observed.set()

    monkeypatch.setattr(core, "_drain_private_delivery_tasks", fail_once_then_drain)
    maintenance = asyncio.create_task(
        maintenance_producer(),
        name="guest-result-maintenance",
    )
    core._guest_maintenance_task = maintenance
    await asyncio.wait_for(maintenance_started.wait(), timeout=1)

    with pytest.raises(InitialDrainFailure) as exc_info:
        await core._post_stop(SimpleNamespace())

    assert exc_info.value is failure
    assert maintenance_observed.is_set()
    assert maintenance.cancelled()
    assert core._guest_maintenance_task is None
    assert core._weekly_review_tasks == set()
    assert drain_calls == 2


@pytest.mark.parametrize(
    "terminal_outcome",
    ["cancelled", "failure"],
)
async def test_stage7c_shutdown_terminal_drain_cancels_awaits_and_clears_tracked_tasks(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    terminal_outcome,
):
    private = "PRIVATE_STAGE7C_DRAIN_TIMEOUT_SENTINEL"
    private_error = "PRIVATE_STAGE7C_DRAIN_TERMINAL_ERROR"
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    started = asyncio.Event()
    release = asyncio.Event()
    terminal_started = asyncio.Event()
    terminal_finished = asyncio.Event()
    unrelated_release = asyncio.Event()

    class Stage7CTerminalFailure(RuntimeError):
        pass

    async def blocked_delivery():
        started.set()
        try:
            await release.wait()
            return True
        except asyncio.CancelledError:
            terminal_started.set()
            if terminal_outcome == "failure":
                raise Stage7CTerminalFailure(private_error) from None
            raise
        finally:
            terminal_finished.set()

    tracked = asyncio.create_task(
        blocked_delivery(),
        name="nova-memory-application-send-lifecycle",
    )
    core._track_nova_memory_application_task(tracked)
    unrelated = asyncio.create_task(
        unrelated_release.wait(),
        name="unrelated-application-task",
    )
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_DRAIN_TIMEOUT_SECONDS", 0.0)
    loop = asyncio.get_running_loop()
    prior_debug = loop.get_debug()
    prior_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        loop.set_debug(True)
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            with caplog.at_level(logging.WARNING):
                await core._post_stop(SimpleNamespace())

            assert terminal_started.is_set()
            assert terminal_finished.is_set()
            assert tracked.done()
            assert tracked.cancelled() is (terminal_outcome == "cancelled")
            assert core._nova_memory_application_tasks == set()
            assert not unrelated.done()

            drain_logs = [
                record.getMessage()
                for record in caplog.records
                if "Nova memory application shutdown" in record.getMessage()
            ]
            assert drain_logs == [
                "Nova memory application shutdown operation=drain "
                "error_type=TimeoutError pending_count=1"
            ]
            lifecycle_logs = [
                record.getMessage()
                for record in caplog.records
                if "Nova memory application task" in record.getMessage()
            ]
            if terminal_outcome == "cancelled":
                assert lifecycle_logs == [
                    "Nova memory application task finished operation=delivery_lifecycle "
                    "error_type=CancelledError"
                ]
            else:
                assert lifecycle_logs == [
                    "Nova memory application task failed operation=delivery_lifecycle "
                    "error_type=Stage7CTerminalFailure"
                ]

            stage7c_logs = drain_logs + lifecycle_logs
            assert "Task(" not in repr(stage7c_logs)
            for forbidden in (private, private_error, "question", "answer", "revision"):
                assert forbidden not in repr(stage7c_logs)
            assert fake_ai.answer_calls == []

            caplog.clear()
            await core._post_shutdown(SimpleNamespace())
            await core._post_shutdown(SimpleNamespace())
            assert core._nova_memory_application_tasks == set()
            assert not unrelated.done()
            assert "Nova memory application shutdown" not in caplog.text
        finally:
            release.set()
            unrelated_release.set()
            if not tracked.done():
                tracked.cancel()
            await asyncio.gather(tracked, unrelated, return_exceptions=True)
            await asyncio.sleep(0)
            gc.collect()
            loop.set_exception_handler(prior_exception_handler)
            loop.set_debug(prior_debug)
    assert loop_errors == []
    assert [warning for warning in caught_warnings if warning.category is RuntimeWarning] == []


async def test_stage7c_terminal_drain_repeats_cancel_until_task_really_finishes(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_STAGE7C_REPEATED_CANCEL_SENTINEL"
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    started = asyncio.Event()
    first_cancel_seen = asyncio.Event()
    second_cancel_seen = asyncio.Event()
    terminal_finished = asyncio.Event()
    next_await_release = asyncio.Event()
    unrelated_release = asyncio.Event()
    cancel_count = 0

    async def suppress_first_cancel_then_block_again():
        nonlocal cancel_count
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_count += 1
            first_cancel_seen.set()
        try:
            await next_await_release.wait()
            return True
        except asyncio.CancelledError:
            cancel_count += 1
            second_cancel_seen.set()
            raise
        finally:
            terminal_finished.set()

    tracked = asyncio.create_task(
        suppress_first_cancel_then_block_again(),
        name="nova-memory-application-send-lifecycle",
    )
    core._track_nova_memory_application_task(tracked)
    unrelated = asyncio.create_task(
        unrelated_release.wait(),
        name="unrelated-application-task",
    )
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_DRAIN_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_CANCEL_TIMEOUT_SECONDS", 0.6)
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_CANCEL_RETRY_SECONDS", 0.2)
    loop = asyncio.get_running_loop()
    prior_debug = loop.get_debug()
    prior_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        loop.set_debug(True)
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            started_at = loop.time()
            with caplog.at_level(logging.WARNING):
                stopping = asyncio.create_task(core._post_stop(SimpleNamespace()))
                await asyncio.wait_for(first_cancel_seen.wait(), timeout=1)

                assert tracked in core._nova_memory_application_tasks
                assert not tracked.done()
                assert not stopping.done()
                assert not unrelated.done()

                await asyncio.wait_for(stopping, timeout=1)
            elapsed = loop.time() - started_at

            assert second_cancel_seen.is_set()
            assert terminal_finished.is_set()
            assert cancel_count >= 2
            assert elapsed < 1
            assert tracked.done()
            assert tracked.cancelled()
            assert core._nova_memory_application_tasks == set()
            assert not unrelated.done()

            await core._post_shutdown(SimpleNamespace())
            await core._post_shutdown(SimpleNamespace())
            assert core._nova_memory_application_tasks == set()
            assert not unrelated.done()
            assert fake_ai.answer_calls == []

            application_logs = [
                record.getMessage()
                for record in caplog.records
                if "Nova memory application" in record.getMessage()
            ]
            assert application_logs == [
                "Nova memory application shutdown operation=drain "
                "error_type=TimeoutError pending_count=1",
                "Nova memory application task finished operation=delivery_lifecycle "
                "error_type=CancelledError",
            ]
            assert "Task(" not in repr(application_logs)
            for forbidden in (private, "question", "answer", "revision"):
                assert forbidden not in repr(application_logs)
        finally:
            next_await_release.set()
            unrelated_release.set()
            if not tracked.done():
                tracked.cancel()
            await asyncio.gather(tracked, unrelated, return_exceptions=True)
            await asyncio.sleep(0)
            gc.collect()
            loop.set_exception_handler(prior_exception_handler)
            loop.set_debug(prior_debug)
    assert loop_errors == []
    assert [warning for warning in caught_warnings if warning.category is RuntimeWarning] == []


async def test_stage7c_terminal_drain_hard_deadline_raises_and_retains_live_task(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_STAGE7C_HARD_DEADLINE_SENTINEL"
    core = FutureSelfBot(runtime_settings(database_url=db.url), db, fake_ai, FakeTranscription())
    started = asyncio.Event()
    force_exit = asyncio.Event()
    unrelated_release = asyncio.Event()
    cancel_count = 0

    async def cancellation_resistant_delivery():
        nonlocal cancel_count
        started.set()
        while not force_exit.is_set():
            try:
                await force_exit.wait()
            except asyncio.CancelledError:
                cancel_count += 1
        return True

    tracked = asyncio.create_task(
        cancellation_resistant_delivery(),
        name="nova-memory-application-send-lifecycle",
    )
    core._track_nova_memory_application_task(tracked)
    unrelated = asyncio.create_task(
        unrelated_release.wait(),
        name="unrelated-application-task",
    )
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_DRAIN_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_CANCEL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(bot_module, "_NOVA_MEMORY_APPLICATION_CANCEL_RETRY_SECONDS", 0.005)
    loop = asyncio.get_running_loop()
    prior_debug = loop.get_debug()
    prior_exception_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        loop.set_debug(True)
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            started_at = loop.time()
            with caplog.at_level(logging.WARNING):
                with pytest.raises(
                    bot_module._NovaMemoryApplicationDrainError,
                    match="Nova memory application terminal drain timed out",
                ):
                    await asyncio.wait_for(
                        core._drain_nova_memory_application_tasks(),
                        timeout=0.5,
                    )
            elapsed = loop.time() - started_at

            assert elapsed < 0.5
            assert cancel_count >= 2
            assert not tracked.done()
            assert tracked in core._nova_memory_application_tasks
            assert not unrelated.done()
            application_logs = [
                record.getMessage()
                for record in caplog.records
                if "Nova memory application shutdown" in record.getMessage()
            ]
            assert application_logs == [
                "Nova memory application shutdown operation=drain "
                "error_type=TimeoutError pending_count=1",
                "Nova memory application shutdown operation=terminal_drain "
                "error_type=TimeoutError pending_count=1",
            ]
            assert "Task(" not in repr(application_logs)
            for forbidden in (private, "question", "answer", "revision"):
                assert forbidden not in repr(application_logs)

            force_exit.set()
            assert await asyncio.wait_for(tracked, timeout=1) is True
            await asyncio.sleep(0)
            assert core._nova_memory_application_tasks == set()
            assert not unrelated.done()

            await core._post_stop(SimpleNamespace())
            await core._post_shutdown(SimpleNamespace())
            await core._post_shutdown(SimpleNamespace())
            assert core._nova_memory_application_tasks == set()
            assert not unrelated.done()
        finally:
            force_exit.set()
            unrelated_release.set()
            if not tracked.done():
                tracked.cancel()
            await asyncio.gather(tracked, unrelated, return_exceptions=True)
            await asyncio.sleep(0)
            gc.collect()
            loop.set_exception_handler(prior_exception_handler)
            loop.set_debug(prior_debug)
    assert loop_errors == []
    assert [warning for warning in caught_warnings if warning.category is RuntimeWarning] == []


async def test_real_application_weekly_voice_pre_route_outer_cancel_keeps_cleanup_alive(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_044
    stt_started = asyncio.Event()
    stt_release = asyncio.Event()

    class BlockingPreRouteTranscription(RuntimeTranscription):
        async def transcribe(self, audio: bytes, filename: str) -> str:
            self.calls.append((audio, filename))
            stt_started.set()
            await stt_release.wait()
            return self.transcript

    transcript = "Р—Р°РІС‚СЂР° РІ 15:05 СЃРґРµР»Р°С‚СЊ РЅРµР±РѕР»СЊС€РѕР№ С€Р°Рі"
    transcription = BlockingPreRouteTranscription(transcript)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    frozen = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=97_044,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert frozen.session is not None
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_054,
        source_message_id=96_044,
        progress_message_id=98_044,
    )
    sent: list[dict[str, object]] = []
    edits: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []
    edit_started = asyncio.Event()
    edit_release = asyncio.Event()
    access = AccessService(db)

    async def fake_send_message(self, *args, **kwargs):
        del self, args
        sent.append(dict(kwargs))
        return progress

    async def blocked_edit_message_text(self, *args, **kwargs):
        del self, args
        edits.append(dict(kwargs))
        edit_started.set()
        await edit_release.wait()
        return progress

    async def fake_delete_message(self, *args, **kwargs):
        del self, args
        deletes.append(dict(kwargs))
        return True

    monkeypatch.setattr(ExtBot, "send_message", fake_send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", blocked_edit_message_text)
    monkeypatch.setattr(ExtBot, "delete_message", fake_delete_message)

    outer = asyncio.create_task(application.process_update(update))
    try:
        await asyncio.wait_for(stt_started.wait(), timeout=10)
        await access.set_guest(telegram_id, source="weekly-runtime-cancel")
        await access.grant_subscriber(telegram_id, source="weekly-runtime-cancel")
        stt_release.set()
        await asyncio.wait_for(edit_started.wait(), timeout=10)
        inner = next(
            task
            for task in core._weekly_review_tasks
            if task.get_name() == "weekly-review-voice-pre-route-lifecycle"
        )
        async with db.sessions() as session:
            assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0

        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
        assert not inner.done()
        assert inner in core._weekly_review_tasks

        edit_release.set()
        assert await asyncio.wait_for(inner, timeout=10) is True
        await asyncio.sleep(0)
    finally:
        stt_release.set()
        edit_release.set()
        if not outer.done():
            outer.cancel()
        await asyncio.gather(outer, *tuple(core._weekly_review_tasks), return_exceptions=True)

    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert len(sent) == 1
    assert sent[0]["chat_id"] == telegram_id
    assert [(entry["chat_id"], entry["message_id"]) for entry in deletes] == [
        (telegram_id, progress.message_id)
    ]
    assert len(edits) == 1
    assert edits[0]["message_id"] == frozen.session.canonical_message_id
    assert edits[0]["reply_markup"] is None
    assert edits[0]["parse_mode"] is None
    assert core._weekly_review_tasks == set()
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


async def test_real_application_weekly_voice_direct_stt_cancel_is_not_delivery(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_045

    class CancellingTranscription(RuntimeTranscription):
        async def transcribe(self, audio: bytes, filename: str) -> str:
            self.calls.append((audio, filename))
            raise asyncio.CancelledError

    transcription = CancellingTranscription("private transcript must not exist")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    frozen = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=97_045,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert frozen.session is not None
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_055,
        source_message_id=96_045,
        progress_message_id=98_045,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    deletes: list[dict[str, object]] = []
    cleanup_started = asyncio.Event()
    downstream: list[int] = []

    async def cleanup_delete(self, *args, **kwargs):
        del self, args
        deletes.append(dict(kwargs))
        cleanup_started.set()
        return True

    async def downstream_probe(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "delete_message", cleanup_delete)
    application.add_handler(TypeHandler(Update, downstream_probe), group=100)

    with pytest.raises(asyncio.CancelledError):
        await application.process_update(update)
    await asyncio.wait_for(cleanup_started.wait(), timeout=2)
    await asyncio.gather(*tuple(core._weekly_review_tasks))
    await asyncio.sleep(0)

    current = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.status == "found"
    assert current.session == frozen.session
    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert len(sent) == 1
    assert sent[0]["chat_id"] == telegram_id
    assert edits == []
    assert [(entry["chat_id"], entry["message_id"]) for entry in deletes] == [
        (telegram_id, progress.message_id)
    ]
    assert downstream == []
    assert core._weekly_review_tasks == set()
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


async def test_real_application_weekly_voice_outer_stt_cancel_tracks_progress_cleanup(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 715_046
    stt_started = asyncio.Event()
    stt_blocker = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    class BlockingCancelledTranscription(RuntimeTranscription):
        async def transcribe(self, audio: bytes, filename: str) -> str:
            self.calls.append((audio, filename))
            stt_started.set()
            await stt_blocker.wait()
            return self.transcript

    transcription = BlockingCancelledTranscription("private cancelled transcript")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=True,
            weekly_review_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    frozen = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=97_046,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert frozen.session is not None
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=31_056,
        source_message_id=96_046,
        progress_message_id=98_046,
    )
    sent, edits = _patch_runtime_voice_transport(monkeypatch, progress)
    deletes: list[dict[str, object]] = []
    downstream: list[int] = []

    async def blocked_cleanup_delete(self, *args, **kwargs):
        del self, args
        deletes.append(dict(kwargs))
        cleanup_started.set()
        await cleanup_release.wait()
        return True

    async def downstream_probe(late_update, context):
        del context
        downstream.append(late_update.update_id)

    monkeypatch.setattr(ExtBot, "delete_message", blocked_cleanup_delete)
    application.add_handler(TypeHandler(Update, downstream_probe), group=100)
    outer = asyncio.create_task(application.process_update(update))
    try:
        await asyncio.wait_for(stt_started.wait(), timeout=10)
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(outer, timeout=1)
        await asyncio.wait_for(cleanup_started.wait(), timeout=10)
        inner = next(
            task
            for task in core._weekly_review_tasks
            if task.get_name() == "weekly-review-voice-cancel-cleanup-lifecycle"
        )
        assert not inner.done()
        assert inner in core._weekly_review_tasks

        current_before_release = await core.weekly_review_service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
        )
        assert current_before_release.session == frozen.session

        cleanup_release.set()
        assert await asyncio.wait_for(inner, timeout=10) is None
        await asyncio.sleep(0)
    finally:
        stt_blocker.set()
        cleanup_release.set()
        if not outer.done():
            outer.cancel()
        await asyncio.gather(outer, *tuple(core._weekly_review_tasks), return_exceptions=True)

    current = await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )
    assert current.status == "found"
    assert current.session == frozen.session
    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert len(sent) == 1
    assert sent[0]["chat_id"] == telegram_id
    assert edits == []
    assert [(entry["chat_id"], entry["message_id"]) for entry in deletes] == [
        (telegram_id, progress.message_id)
    ]
    assert downstream == []
    assert core._weekly_review_tasks == set()
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


async def _runtime_companion_content_counts(db) -> tuple[int, int, int]:
    async with db.sessions() as session:
        conversations = await session.scalar(select(func.count(ConversationMessage.id)))
        drafts = await session.scalar(select(func.count(DraftInboxItem.id)))
        inbox = await session.scalar(select(func.count(InboxItem.id)))
    return int(conversations or 0), int(drafts or 0), int(inbox or 0)


def _runtime_ncap_callbacks(markup) -> list[str]:
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if isinstance(button.callback_data, str) and button.callback_data.startswith("ncap:")
    ]


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_companion_reflection_has_text_voice_parity(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 716_001 if source == "text" else 716_002
    reflection = "Я постоянно забываю о главном, из-за каждодневной суеты"
    transcription = RuntimeTranscription(reflection)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="admin")

    if source == "text":
        answer = _runtime_bot_message(application, telegram_id, 116_001)
        sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [answer])
        update = _runtime_text_update(
            application,
            telegram_id,
            reflection,
            update_id=41_001,
            source_message_id=115_001,
        )
    else:
        update, answer = _runtime_voice_update(
            application,
            telegram_id,
            update_id=41_002,
            source_message_id=115_002,
            progress_message_id=116_002,
        )
        sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [answer])

    await application.process_update(update)
    await asyncio.sleep(0)

    assert [call[0] for call in fake_ai.companion_calls] == [reflection]
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    if source == "text":
        assert [entry["text"] for entry in sent] == [fake_ai.companion_result.answer]
        assert edits == []
    else:
        assert [entry["text"] for entry in sent] == ["Расшифровываю голосовую мысль…"]
        assert edits[-1]["message_id"] == answer.message_id
        assert edits[-1]["text"] == fake_ai.companion_result.answer
        assert edits[-1].get("reply_markup") is None
    assert deletes == []
    assert await _runtime_companion_content_counts(db) == (2, 0, 0)
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_companion_local_name_and_identity_are_provider_free(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 716_003 if source == "text" else 716_004
    transcription = RuntimeTranscription("Нова")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    phrases = ("Нова", "Ты Nova?")

    if source == "text":
        replies = [
            _runtime_bot_message(application, telegram_id, 116_003),
            _runtime_bot_message(application, telegram_id, 116_004),
        ]
        sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, replies)
        for offset, phrase in enumerate(phrases):
            await application.process_update(
                _runtime_text_update(
                    application,
                    telegram_id,
                    phrase,
                    update_id=41_010 + offset,
                    source_message_id=115_010 + offset,
                )
            )
        assert [entry["text"] for entry in sent] == [
            "Да, я здесь 🙂",
            "Да, я Nova. Я рядом — о чём хочешь поговорить?",
        ]
        assert edits == []
    else:
        first_update, first_progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=41_012,
            source_message_id=115_012,
            progress_message_id=116_012,
        )
        second_update, second_progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=41_013,
            source_message_id=115_013,
            progress_message_id=116_013,
        )
        sent, edits, deletes = _patch_runtime_stage7c_transport(
            monkeypatch,
            [first_progress, second_progress],
        )
        await application.process_update(first_update)
        transcription.transcript = phrases[1]
        await application.process_update(second_update)
        assert [entry["text"] for entry in sent] == [
            "Расшифровываю голосовую мысль…",
            "Расшифровываю голосовую мысль…",
        ]
        assert [(entry["message_id"], entry["text"]) for entry in edits] == [
            (first_progress.message_id, "Да, я здесь 🙂"),
            (
                second_progress.message_id,
                "Да, я Nova. Я рядом — о чём хочешь поговорить?",
            ),
        ]

    assert deletes == []
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert await _runtime_companion_content_counts(db) == (0, 0, 0)
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    assert core._nova_companion_tasks == set()


async def test_real_application_companion_concrete_idea_publishes_one_opaque_suggestion(
    db,
    fake_ai,
    monkeypatch,
):
    from future_self.schemas import NovaCompanionCapture, NovaCompanionResponse

    telegram_id = 716_005
    idea = "Идея: сделать тихую комнату для чтения"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Это может стать спокойным местом для восстановления.",
        capture=NovaCompanionCapture(
            kind="idea",
            title="сделать тихую комнату для чтения",
        ),
    )
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    answer = _runtime_bot_message(application, telegram_id, 116_005)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [answer])
    markup_edits: list[dict[str, object]] = []

    async def edit_reply_markup(self, *args, **kwargs):
        del self, args
        markup_edits.append(kwargs)
        return answer

    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_reply_markup)
    update = _runtime_text_update(
        application,
        telegram_id,
        idea,
        update_id=41_020,
        source_message_id=115_020,
    )

    await application.process_update(update)
    await asyncio.sleep(0)

    assert [call[0] for call in fake_ai.companion_calls] == [idea]
    assert [entry["text"] for entry in sent] == [fake_ai.companion_result.answer]
    assert edits == []
    assert len(markup_edits) == 1
    assert markup_edits[0]["chat_id"] == telegram_id
    assert markup_edits[0]["message_id"] == answer.message_id
    callbacks = _runtime_ncap_callbacks(markup_edits[0]["reply_markup"])
    assert len(callbacks) == 2
    assert len(set(callbacks)) == 2
    assert all(len(value.encode("utf-8")) <= 64 for value in callbacks)
    assert all(idea not in value for value in callbacks)
    assert (
        await core.nova_companion_captures.peek(
            callbacks[0],
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
            canonical_message_id=answer.message_id,
            access_version=owner.access_version,
        )
        is not None
    )
    assert deletes == []
    assert await _runtime_companion_content_counts(db) == (2, 0, 0)
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize(
    ("enabled", "admin_only", "tier"),
    [
        (False, True, "admin"),
        (True, True, "subscriber"),
    ],
)
async def test_real_application_companion_disabled_or_ineligible_keeps_legacy_preview(
    db,
    fake_ai,
    monkeypatch,
    enabled,
    admin_only,
    tier,
):
    telegram_id = 716_006 if enabled else 716_007
    reflection = "Я постоянно забываю о главном, из-за каждодневной суеты"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=enabled,
            nova_companion_admin_only=admin_only,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier=tier)
    preview = _runtime_bot_message(application, telegram_id, 116_006)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [preview])
    update = _runtime_text_update(
        application,
        telegram_id,
        reflection,
        update_id=41_030,
        source_message_id=115_030,
    )

    await application.process_update(update)

    assert fake_ai.companion_calls == []
    assert [call[0] for call in fake_ai.route_calls] == [reflection]
    drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
    assert len(drafts) == 1
    assert len(sent) == 1
    assert any(
        str(button.callback_data).startswith("inbox:save:")
        for row in sent[0]["reply_markup"].inline_keyboard
        for button in row
    )
    assert edits == []
    assert deletes == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


async def test_real_application_companion_does_not_steal_persisted_onboarding(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 716_008
    reflection = "Я постоянно забываю о главном, из-за каждодневной суеты"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await core._user(telegram_id)
    await AccessService(db).grant_admin(telegram_id, source="companion-runtime")
    async with db.session() as session:
        session.add(
            OnboardingState(
                user_id=owner.id,
                current_step=2,
                answers={"display_name": "Варвара"},
                status="in_progress",
            )
        )
    reply = _runtime_bot_message(application, telegram_id, 116_008)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [reply] * 4)

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            reflection,
            update_id=41_040,
            source_message_id=115_040,
        )
    )

    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == owner.id)
        )
    assert state is not None
    assert state.current_step == 3
    assert state.answers["future_life"] == reflection
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert len(sent) >= 2
    assert sent[0]["text"] == "Ответ сохранён ✓"
    assert edits == []
    assert deletes == []
    assert await _runtime_companion_content_counts(db) == (0, 0, 0)


async def test_real_application_companion_does_not_steal_active_vision_flow(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 716_011
    wish = "Нова, я хочу побывать у океана"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    draft = await core.vision_service.begin(owner.id, telegram_id)
    selected = await core.vision_service.choose_category(
        owner.id,
        telegram_id,
        "other",
        draft_id=draft.id,
    )
    assert selected.status == "advanced"
    reply = _runtime_bot_message(application, telegram_id, 116_011)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [reply] * 3)

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            wish,
            update_id=41_045,
            source_message_id=115_045,
        )
    )

    current = await core.vision_service.draft(owner.id, telegram_id)
    assert current is not None
    assert current.wish_text == wish
    assert current.step == "why"
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert sent
    assert edits == []
    assert deletes == []
    assert await _runtime_companion_content_counts(db) == (0, 0, 0)


async def test_real_application_companion_does_not_steal_active_reminder(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 716_009
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    await core.reminder_sessions.create(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
        access_version=owner.access_version,
        title="Позвонить врачу",
        schedule_kind=ReminderScheduleKind.ONCE,
        local_date=None,
        local_time=None,
        timezone=owner.timezone,
        timezone_source=ReminderTimezoneSource.PROFILE,
        phase=ReminderFlowPhase.WHEN,
    )
    reply = _runtime_bot_message(application, telegram_id, 116_009)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [reply])

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            "завтра в 10:00",
            update_id=41_050,
            source_message_id=115_050,
        )
    )

    current = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert current is not None
    assert current.phase is ReminderFlowPhase.PREVIEW
    assert current.title == "Позвонить врачу"
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert len(sent) == 1
    assert len(edits) == 1
    assert edits[0]["message_id"] == reply.message_id
    assert "Проверь напоминание" in edits[0]["text"]
    assert deletes == []
    assert await _runtime_companion_content_counts(db) == (0, 0, 0)


async def test_real_application_addressed_capability_stays_in_nova_help(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 716_010
    question = "Нова, как добавить задачу с напоминанием?"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    canonical = _runtime_bot_message(application, telegram_id, 116_010)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [canonical])

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            question,
            update_id=41_060,
            source_message_id=115_060,
        )
    )

    current = await core.nova_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert current is not None
    assert current.last_action_id == "task_create"
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert len(sent) == 1
    assert sent[0]["text"] == "✨ Nova\n\nРазбираю вопрос…"
    assert len(edits) == 1
    assert edits[0]["message_id"] == canonical.message_id
    assert any(
        str(button.callback_data).startswith("nova:action:task_create:")
        for row in edits[0]["reply_markup"].inline_keyboard
        for button in row
    )
    assert deletes == []
    assert await _runtime_companion_content_counts(db) == (0, 0, 0)


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_companion_keeps_explicit_reminder_preview_text_voice(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 716_012 if source == "text" else 716_013
    phrase = "Напомни завтра в 19:30 заполнить дневник"
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    canonical = _runtime_bot_message(application, telegram_id, 116_012)
    sent, edits, deletes = _patch_runtime_stage7c_transport(
        monkeypatch,
        [canonical] * 3,
    )
    if source == "text":
        update = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=41_070,
            source_message_id=115_070,
        )
    else:
        update, canonical = _runtime_voice_update(
            application,
            telegram_id,
            update_id=41_071,
            source_message_id=115_071,
            progress_message_id=116_012,
        )

    await application.process_update(update)

    current = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert current is not None
    assert current.phase is ReminderFlowPhase.PREVIEW
    assert current.title == "заполнить дневник"
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert deletes == []
    if source == "text":
        assert sent
        rendered = [entry["text"] for entry in (*sent, *edits)]
        assert any("Проверь напоминание" in text for text in rendered)
    else:
        assert [entry["text"] for entry in sent] == ["Расшифровываю голосовую мысль…"]
        assert edits
        assert all(entry["message_id"] == canonical.message_id for entry in edits)
        assert "Проверь напоминание" in edits[-1]["text"]
    assert await _runtime_companion_content_counts(db) == (0, 0, 0)
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("companion_enabled", [False, True])
async def test_real_application_relative_voice_fallback_matches_legacy_with_companion(
    db,
    fake_ai,
    monkeypatch,
    companion_enabled,
):
    telegram_id = 716_014 if companion_enabled else 716_015
    phrase = "Напомни через 5 минут выпить воды"
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=companion_enabled,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    update, progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=41_072,
        source_message_id=115_072,
        progress_message_id=116_014,
    )
    preview = _runtime_bot_message(application, telegram_id, 117_014)
    sent, edits, deletes = _patch_runtime_stage7c_transport(
        monkeypatch,
        [progress, preview],
    )

    await application.process_update(update)
    await asyncio.sleep(0)

    drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
    assert len(drafts) == 1
    assert drafts[0].kind == "task"
    assert drafts[0].title == "Выпить воды"
    assert drafts[0].temporal_resolution is not None
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert [entry["text"] for entry in sent[:1]] == ["Расшифровываю голосовую мысль…"]
    assert len(sent) == 2
    assert "Напоминание:" in sent[1]["text"]
    assert len(edits) == 1
    assert edits[0]["message_id"] == progress.message_id
    assert edits[0]["text"].startswith("Я услышал:")
    assert deletes == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert core._nova_companion_tasks == set()


async def test_real_application_companion_exact_wake_regression_sequence(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 716_016
    reflection = "Я постоянно забываю о главном, из-за каждодневной суеты"
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    replies = [
        _runtime_bot_message(application, telegram_id, 116_016 + offset) for offset in range(4)
    ]
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, replies)

    for offset, phrase in enumerate(("Нова", "Нова ты тут?", "Ты Нова?")):
        await application.process_update(
            _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=41_080 + offset,
                source_message_id=115_080 + offset,
            )
        )

        assert fake_ai.companion_calls == []
        assert fake_ai.route_calls == []
        assert fake_ai.answer_calls == []
        assert await _runtime_companion_content_counts(db) == (0, 0, 0)
        assert (
            await core.nova_sessions.current(
                owner_id=owner.id,
                telegram_user_id=telegram_id,
                chat_id=telegram_id,
            )
            is None
        )

    assert [entry["text"] for entry in sent] == [
        "Да, я здесь 🙂",
        "Да, я здесь 🙂",
        "Да, я Nova. Я рядом — о чём хочешь поговорить?",
    ]

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            reflection,
            update_id=41_083,
            source_message_id=115_083,
        )
    )
    await asyncio.sleep(0)

    assert [call[0] for call in fake_ai.companion_calls] == [reflection]
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert [entry["text"] for entry in sent] == [
        "Да, я здесь 🙂",
        "Да, я здесь 🙂",
        "Да, я Nova. Я рядом — о чём хочешь поговорить?",
        fake_ai.companion_result.answer,
    ]
    assert all(entry.get("reply_markup") is None for entry in sent)
    assert edits == []
    assert deletes == []
    assert await core.draft_service.active_previews(telegram_id, telegram_id) == []
    assert await _runtime_companion_content_counts(db) == (2, 0, 0)
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize(
    ("telegram_id", "command", "seed_reference", "expected_kind", "expected_title"),
    [
        (
            716_017,
            "Запиши это как заметку",
            "Подготовить короткий план разговора с Мариной",
            "note",
            "Подготовить короткий план разговора с Мариной",
        ),
        (
            716_018,
            "Создай задачу позвонить врачу",
            None,
            "task",
            "позвонить врачу",
        ),
    ],
)
async def test_real_application_companion_explicit_capture_stays_direct_preview(
    db,
    fake_ai,
    monkeypatch,
    telegram_id,
    command,
    seed_reference,
    expected_kind,
    expected_title,
):
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    if seed_reference is not None:
        await core.conversation.append(
            telegram_id,
            telegram_id,
            role="user",
            content=seed_reference,
            source="text",
            intent="companion_user",
        )
    preview = _runtime_bot_message(application, telegram_id, 116_017)
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, [preview])

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            command,
            update_id=41_090,
            source_message_id=115_090,
        )
    )

    drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
    assert len(drafts) == 1
    assert drafts[0].kind == expected_kind
    assert drafts[0].title == expected_title
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert len(sent) == 1
    assert sent[0]["reply_markup"] is not None
    callbacks = [
        button.callback_data
        for row in sent[0]["reply_markup"].inline_keyboard
        for button in row
        if isinstance(button.callback_data, str)
    ]
    assert any(callback.startswith("inbox:save:") for callback in callbacks)
    assert not any(callback.startswith("ncap:") for callback in callbacks)
    assert edits == []
    assert deletes == []
    assert (
        await core.nova_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_companion_identity_continuity_reminder_transcript(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    from future_self.schemas import NovaCompanionReminderOffer, NovaCompanionResponse

    telegram_id = 719_100 if source == "text" else 719_101
    phrases = (
        "Привет",
        "Нова",
        "Я чё-то так устал",
        "Да))) не забыть бы мне завтра на стрижку)",
        "В моей голове?",
        "А вдруг забуду?",
        "Что же делать?",
        "Меня зовут Назар. Я мужчина",
        "А ты не знала, как меня зовут?",
        "Ну так можешь помочь мне? Чтобы я не забыл завтра о стрижке?",
        "Поставь плиз",
        "На стрижку завтра 19:00",
        "Напомнишь?",
        "А мне напомнишь?",
        "В смысле? Ты записала напоминание?",
    )
    transcription = RuntimeTranscription(phrases[0])
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    async with db.session() as session:
        await session.execute(
            update(User)
            .where(User.id == owner.id)
            .values(display_name="Назар", location_city="Москва", timezone="Europe/Moscow")
        )
        session.add_all(
            [
                VisionProfile(
                    user_id=owner.id,
                    raw_answers={},
                    summary="Хочу больше энергии и спокойствия",
                    values=["Здоровье"],
                    desired_identity=["Спокойный мужчина"],
                    constraints=[],
                ),
                VisionItem(
                    owner_id=owner.id,
                    category="health_energy",
                    wish_text="Чувствовать себя бодрее",
                    why_text="Жить активнее",
                    first_step="Наладить отдых",
                    status="active",
                ),
                Goal(
                    user_id=owner.id,
                    life_area="Здоровье",
                    title="Восстановить энергию",
                    outcome="Больше сил",
                    progress_criterion="Стабильный режим",
                    horizon="Три месяца",
                    status="active",
                    priority=5,
                    vision_link="Чувствовать себя бодрее",
                ),
            ]
        )
    owner = await core._user(telegram_id)
    returned = [
        _runtime_bot_message(application, telegram_id, 190_000 + index) for index in range(40)
    ]
    sent, edits, deletes = _patch_runtime_stage7c_transport(monkeypatch, returned)
    markup_edits: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []

    async def edit_reply_markup(self, *args, **kwargs):
        del self, args
        markup_edits.append(kwargs)
        return returned[0]

    async def answer_callback_query(self, callback_query_id, *args, **kwargs):
        del self, args
        answers.append({"callback_query_id": callback_query_id, **kwargs})
        return True

    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_reply_markup)
    monkeypatch.setattr(ExtBot, "answer_callback_query", answer_callback_query)

    provider_steps = {0, 2, 3, 4, 5, 6, 7, 9}
    for index, phrase in enumerate(phrases):
        if index == 5:
            fake_ai.companion_result = NovaCompanionResponse(
                answer="Тогда лучше действительно поставить напоминание. Могу помочь 🙂",
                reminder_offer=NovaCompanionReminderOffer(
                    title="стрижку",
                    schedule_wording="завтра",
                    evidence=phrases[3],
                ),
            )
        elif index in provider_steps:
            fake_ai.companion_result = NovaCompanionResponse(
                answer=f"Продолжаю разговор о стрижке: шаг {index}."
            )
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=49_000 + index,
                source_message_id=189_000 + index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=49_000 + index,
                source_message_id=189_000 + index,
                progress_message_id=190_000 + index,
            )
        await application.process_update(update_value)
        await asyncio.sleep(0)

    assert len(fake_ai.companion_calls) == len(provider_steps)
    step_six_projection = fake_ai.companion_calls[4][2].provider_payload()
    assert step_six_projection["confirmed_identity"] == {
        "display_name": "Назар",
        "location_city": "Москва",
        "timezone": "Europe/Moscow",
    }
    assert step_six_projection["active_vision_items"][0]["wish_text"] == "Чувствовать себя бодрее"
    assert step_six_projection["active_goals"][0]["title"] == "Восстановить энергию"
    recent_text = str(step_six_projection["recent_conversation"])
    assert "стрижку" in recent_text
    assert "telegram" not in str(step_six_projection).casefold()
    assert "access_version" not in str(step_six_projection)

    nrem_edits = [
        entry
        for entry in markup_edits
        if any(
            str(button.callback_data).startswith("nrem:")
            for row in entry["reply_markup"].inline_keyboard
            for button in row
        )
    ]
    assert len(nrem_edits) == 1
    reminder = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert reminder is not None
    assert reminder.title == "стрижку"
    assert reminder.phase is ReminderFlowPhase.PREVIEW
    assert reminder.local_time is not None and reminder.local_time.strftime("%H:%M") == "19:00"
    assert reminder.timezone == "Europe/Moscow"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0

    preview_edit = next(
        entry
        for entry in reversed(edits)
        if entry.get("reply_markup") is not None
        and any(
            str(button.callback_data).startswith("rmd:")
            for row in entry["reply_markup"].inline_keyboard
            for button in row
        )
    )
    confirm_data = preview_edit["reply_markup"].inline_keyboard[0][0].callback_data
    canonical = next(
        message for message in returned if message.message_id == reminder.canonical_message_id
    )
    telegram_user = TelegramUser(telegram_id, False, "Назар")
    query = CallbackQuery(
        f"stage8b2-{source}",
        telegram_user,
        "stage8b2-chat",
        message=canonical,
        data=confirm_data,
    )
    callback_update = Update(49_100, callback_query=query)
    callback_update.set_bot(application.bot)
    query.set_bot(application.bot)
    await application.process_update(callback_update)
    await asyncio.sleep(0)

    async with db.sessions() as session:
        reminders = list((await session.scalars(select(TaskReminder))).all())
        assert len(reminders) == 1
        assert reminders[0].timezone == "Europe/Moscow"
        assert (
            reminders[0]
            .event_at.replace(tzinfo=UTC)
            .astimezone(ZoneInfo("Europe/Moscow"))
            .strftime("%H:%M")
            == "19:00"
        )
    assert len(answers) == 1
    assert core._nova_companion_tasks == set()
    assert all("я поставила" not in str(entry.get("text", "")).casefold() for entry in sent)
    assert all("напомню" not in str(entry.get("text", "")).casefold() for entry in sent)


_RUNTIME_EXPLICIT_CAPTURE_COMMANDS = (
    (
        "Добавь в мысли: я лучше работаю, когда утром не читаю новости.",
        "note",
        "я лучше работаю, когда утром не читаю новости",
        None,
    ),
    ("Добавь задачу позвонить врачу", "task", "позвонить врачу", None),
    ("Добавь идею открыть кофейню", "idea", "открыть кофейню", None),
    ("Сохрани желание увидеть океан", "desire", "увидеть океан", None),
    ("Запиши заметку купить молоко", "note", "купить молоко", None),
    ("Запиши задачу позвонить врачу", "task", "позвонить врачу", None),
    (
        "Создай задачу завтра в 10:00 позвонить врачу",
        "task",
        "завтра в 10:00 позвонить врачу",
        None,
    ),
    (
        "Сохрани это",
        "note",
        "Подготовить короткий план разговора с Мариной",
        "Подготовить короткий план разговора с Мариной",
    ),
)


def _runtime_callback_values(markup) -> list[str]:
    if markup is None:
        return []
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if isinstance(button.callback_data, str)
    ]


def _runtime_callback_by_label(markup, label: str) -> str:
    matches = [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.text == label and isinstance(button.callback_data, str)
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_companion_all_explicit_commands_are_direct_provider_free(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    fixed_now = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)
    transcription = RuntimeTranscription("unused")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    core.date_resolver._now_provider = lambda: fixed_now
    application = core.build()
    application._initialized = True
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    command_index = 0

    for address in ("", "Нова, ", "Nova, "):
        for (
            base_command,
            expected_kind,
            expected_title,
            reference,
        ) in _RUNTIME_EXPLICIT_CAPTURE_COMMANDS:
            command_index += 1
            telegram_id = 716_100 + command_index + (100 if source == "voice" else 0)
            command = f"{address}{base_command}"
            await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
            if reference is not None:
                await core.conversation.append(
                    telegram_id,
                    telegram_id,
                    role="user",
                    content=reference,
                    source="text",
                    intent="companion_user",
                )
            sent_before = len(transport.sent)
            edits_before = len(transport.edits)
            if source == "text":
                update = _runtime_text_update(
                    application,
                    telegram_id,
                    command,
                    update_id=42_000 + command_index,
                    source_message_id=117_000 + command_index,
                )
            else:
                transcription.transcript = command
                update, _unused_progress = _runtime_voice_update(
                    application,
                    telegram_id,
                    update_id=42_100 + command_index,
                    source_message_id=117_100 + command_index,
                    progress_message_id=118_100 + command_index,
                )

            await application.process_update(update)

            drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
            assert len(drafts) == 1
            draft = drafts[0]
            assert draft.kind == expected_kind
            assert draft.title == expected_title
            if base_command.startswith("Создай задачу завтра"):
                assert draft.resolved_date == date(2026, 8, 19)
                assert draft.temporal_resolution is not None
                assert draft.temporal_resolution["resolved_local_time"] == "10:00:00"
                assert draft.temporal_resolution["timezone"] == "Europe/Moscow"
            else:
                assert draft.resolved_date is None
                assert draft.temporal_resolution is None

            new_renderings = [
                *transport.sent[sent_before:],
                *transport.edits[edits_before:],
            ]
            callbacks = [
                callback
                for rendering in new_renderings
                for callback in _runtime_callback_values(rendering.get("reply_markup"))
            ]
            assert any(callback.startswith("inbox:save:") for callback in callbacks)
            assert not any(callback.startswith("ncap:") for callback in callbacks)
            async with db.sessions() as session:
                assert await session.scalar(select(func.count(InboxItem.id))) == 0
                assert await session.scalar(select(func.count(LifeCollection.id))) == 0
            assert core._nova_companion_tasks == set()

    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    if source == "voice":
        assert transcription.calls == [(b"runtime-voice", "voice.ogg")] * command_index
    else:
        assert transcription.calls == []


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_reserved_thought_capture_saves_exactly_one_inbox_item(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    phrase = "Добавь в мысли: я лучше работаю, когда утром не читаю новости."
    expected = "я лучше работаю, когда утром не читаю новости"
    telegram_id = 716_330 if source == "text" else 716_331
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    if source == "text":
        update_value = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=42_330,
            source_message_id=117_330,
        )
        canonical = None
    else:
        update_value, _unused_progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=42_331,
            source_message_id=117_331,
            progress_message_id=118_331,
        )
    await application.process_update(update_value)

    drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
    assert len(drafts) == 1
    assert drafts[0].kind == "note" and drafts[0].title == expected
    assert fake_ai.companion_calls == []
    assert fake_ai.route_calls == []
    renderings = [*transport.sent, *transport.edits]
    save_callbacks = [
        callback
        for rendering in renderings
        for callback in _runtime_callback_values(rendering.get("reply_markup"))
        if callback.startswith("inbox:save:")
    ]
    assert len(save_callbacks) == 1
    canonical = transport.sent_messages[-1]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LifeCollection.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0

    telegram_user = TelegramUser(telegram_id, "Тест", False)
    callback_update, _query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        canonical,
        save_callbacks[0],
        update_id=42_332,
    )
    await application.process_update(callback_update)

    async with db.sessions() as session:
        rows = list((await session.scalars(select(InboxItem))).all())
        assert len(rows) == 1
        assert rows[0].kind == "note" and rows[0].title == expected
        assert await session.scalar(select(func.count(LifeCollection.id))) == 0
    assert fake_ai.companion_calls == []
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_companion_implicit_dated_task_add_preserves_temporal(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    from future_self.schemas import NovaCompanionCapture, NovaCompanionResponse

    fixed_now = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)
    telegram_id = 716_350 if source == "text" else 716_351
    phrase = "Я хочу завтра в 10:00 позвонить врачу"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Давай сохраним это как конкретную задачу.",
        capture=NovaCompanionCapture(kind="task", title="позвонить врачу"),
    )
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        transcription,
    )
    core.date_resolver._now_provider = lambda: fixed_now
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    markup_edits: list[dict[str, object]] = []

    async def edit_reply_markup(self, *args, **kwargs):
        del self, args
        markup_edits.append(dict(kwargs))
        return transport.sent_messages[0]

    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_reply_markup)
    if source == "text":
        update = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=42_350,
            source_message_id=117_350,
        )
    else:
        update, _unused_progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=42_351,
            source_message_id=117_351,
            progress_message_id=118_351,
        )

    await application.process_update(update)

    assert [call[0] for call in fake_ai.companion_calls] == [phrase]
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert len(transport.sent_messages) == 1
    assert len(markup_edits) == 1
    offer_markup = markup_edits[0]["reply_markup"]
    add_data = _runtime_callback_by_label(offer_markup, "Добавить как задачу")
    assert add_data.startswith("ncap:")
    assert len(add_data.encode("utf-8")) <= 64
    assert phrase not in add_data
    assert "позвонить врачу" not in add_data
    assert await _runtime_companion_content_counts(db) == (2, 0, 0)

    telegram_user = TelegramUser(telegram_id, "Тест", False)
    callback_update, query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        transport.sent_messages[0],
        add_data,
        update_id=42_352,
    )
    await application.process_update(callback_update)

    drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.kind == "task"
    assert draft.title == "позвонить врачу"
    assert draft.resolved_date == date(2026, 8, 19)
    assert draft.temporal_resolution is not None
    assert draft.temporal_resolution["resolved_local_date"] == "2026-08-19"
    assert draft.temporal_resolution["resolved_local_time"] == "10:00:00"
    assert draft.temporal_resolution["timezone"] == "Europe/Moscow"
    assert draft.temporal_resolution["precision"] == "datetime"
    assert draft.temporal_resolution["original_expression"] == phrase
    assert len(fake_ai.companion_calls) == 1
    assert fake_ai.route_calls == []
    assert fake_ai.answer_calls == []
    assert [entry["callback_query_id"] for entry in transport.answers] == [query.id]
    assert len(transport.sent_messages) == 2
    assert await _runtime_companion_content_counts(db) == (3, 1, 0)
    assert core._nova_companion_tasks == set()
    if source == "voice":
        assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    else:
        assert transcription.calls == []


async def test_real_application_companion_temporal_conflict_keeps_opaque_choice(
    db,
    fake_ai,
    monkeypatch,
):
    from future_self.schemas import NovaCompanionCapture, NovaCompanionResponse

    fixed_now = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)
    telegram_id = 716_352
    phrase = "Я хочу в пятницу 20 августа в 10:00 позвонить врачу"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Давай сначала уточним дату задачи.",
        capture=NovaCompanionCapture(kind="task", title="позвонить врачу"),
    )
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=True,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    core.date_resolver._now_provider = lambda: fixed_now
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="admin")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    markup_edits: list[dict[str, object]] = []

    async def edit_reply_markup(self, *args, **kwargs):
        del self, args
        markup_edits.append(dict(kwargs))
        return transport.sent_messages[0]

    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_reply_markup)
    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=42_360,
            source_message_id=117_360,
        )
    )
    offer_markup = markup_edits[0]["reply_markup"]
    add_data = _runtime_callback_by_label(offer_markup, "Добавить как задачу")
    telegram_user = TelegramUser(telegram_id, "Тест", False)
    first_callback, first_query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        transport.sent_messages[0],
        add_data,
        update_id=42_361,
    )

    await application.process_update(first_callback)

    assert await core.draft_service.active_previews(telegram_id, telegram_id) == []
    assert len(fake_ai.companion_calls) == 1
    conflict_markups = [
        rendering.get("reply_markup")
        for rendering in transport.edits
        if len(_runtime_ncap_callbacks(rendering.get("reply_markup"))) == 3
    ]
    conflict_markups.extend(
        rendering.get("reply_markup")
        for rendering in markup_edits[1:]
        if len(_runtime_ncap_callbacks(rendering.get("reply_markup"))) == 3
    )
    assert len(conflict_markups) == 1
    conflict_markup = conflict_markups[0]
    choice_buttons = [
        button
        for row in conflict_markup.inline_keyboard
        for button in row
        if button.text != "Не сейчас"
    ]
    assert len(choice_buttons) == 2
    assert all(str(button.callback_data).startswith("ncap:") for button in choice_buttons)
    assert all(phrase not in str(button.callback_data) for button in choice_buttons)
    assert await _runtime_companion_content_counts(db) == (2, 0, 0)

    second_callback, second_query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        transport.sent_messages[0],
        str(choice_buttons[0].callback_data),
        update_id=42_362,
    )
    await application.process_update(second_callback)

    drafts = await core.draft_service.active_previews(telegram_id, telegram_id)
    assert len(drafts) == 1
    assert drafts[0].kind == "task"
    assert drafts[0].title == "позвонить врачу"
    assert drafts[0].resolved_date == date(2026, 8, 21)
    assert drafts[0].temporal_resolution is not None
    assert drafts[0].temporal_resolution["resolved_local_time"] == "10:00:00"
    assert drafts[0].temporal_resolution["timezone"] == "Europe/Moscow"
    assert len(fake_ai.companion_calls) == 1
    assert [entry["callback_query_id"] for entry in transport.answers] == [
        first_query.id,
        second_query.id,
    ]
    assert await _runtime_companion_content_counts(db) == (3, 1, 0)
    assert core._nova_companion_tasks == set()


async def _seed_runtime_companion_reminder_receipt(core, db, owner, *, suffix: str):
    from future_self.nova_companion_handlers import _CompanionStatusReceipt

    event_at = datetime(2026, 8, 22, 16, 0, tzinfo=UTC)
    async with db.session() as session:
        item = InboxItem(
            user_id=owner.id,
            kind="task",
            title=f"Стрижка {suffix}",
            description=None,
            raw_text=f"Стрижка {suffix}",
            next_step=None,
            resolved_date=event_at.date(),
            temporal_resolution=None,
            source="companion",
            status="confirmed",
            version=1,
        )
        session.add(item)
        await session.flush()
        session.add(
            TaskReminder(
                inbox_item_id=item.id,
                telegram_user_id=owner.telegram_id,
                chat_id=owner.telegram_id,
                event_at=event_at,
                remind_at=event_at,
                timezone="Europe/Moscow",
                delivery_key=f"stage8b2-receipt-{suffix}-{owner.id}",
                task_version=item.version,
                status="pending",
            )
        )
        await session.flush()
        receipt = _CompanionStatusReceipt(
            kind="reminder",
            owner_id=owner.id,
            telegram_user_id=owner.telegram_id,
            chat_id=owner.telegram_id,
            access_version=owner.access_version,
            inbox_item_id=item.id,
            inbox_item_version=item.version,
            title=item.title,
        )
    key = (owner.id, owner.telegram_id, owner.telegram_id)
    core._nova_companion_status_receipts[key] = receipt
    return key, receipt


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_substantive_text_voice_invalidates_exact_status_receipt(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 716_370 if source == "text" else 716_371
    phrase = "Давай обсудим новый план тренировки"
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    key, _receipt = await _seed_runtime_companion_reminder_receipt(
        core,
        db,
        owner,
        suffix=source,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    if source == "text":
        action_update = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=42_370,
            source_message_id=117_370,
        )
    else:
        action_update, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=42_371,
            source_message_id=117_371,
            progress_message_id=118_371,
        )
    await application.process_update(action_update)
    await asyncio.sleep(0)
    assert key not in core._nova_companion_status_receipts

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            "Оно уже создано?",
            update_id=42_372,
            source_message_id=117_372,
        )
    )
    assert transport.sent[-1]["text"] == "Пока нет — напоминание ещё не создано."
    assert len(fake_ai.companion_calls) == 1
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("entry", ["command", "callback"])
async def test_real_application_new_cancelled_durable_entry_invalidates_old_status_receipt(
    db,
    fake_ai,
    monkeypatch,
    entry,
):
    telegram_id = 716_372 if entry == "command" else 716_373
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    key, _receipt = await _seed_runtime_companion_reminder_receipt(
        core,
        db,
        owner,
        suffix=entry,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    if entry == "command":
        start_update = _runtime_weekly_command_update(
            application,
            telegram_id,
            "/evening",
            update_id=42_373,
            source_message_id=117_373,
        )
    else:
        telegram_user = TelegramUser(telegram_id, "Варвара", False)
        canonical = _runtime_bot_message(application, telegram_id, 118_374)
        start_update, _query = _runtime_weekly_callback_update(
            application,
            telegram_user,
            canonical,
            "nav:action:evening",
            update_id=42_374,
        )
    await application.process_update(start_update)
    assert key not in core._nova_companion_status_receipts

    await application.process_update(
        _runtime_weekly_command_update(
            application,
            telegram_id,
            "/cancel",
            update_id=42_375,
            source_message_id=117_375,
        )
    )
    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            "Оно уже создано?",
            update_id=42_376,
            source_message_id=117_376,
        )
    )

    assert transport.sent[-1]["text"] == "Пока нет — напоминание ещё не создано."
    assert fake_ai.companion_calls == []
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize(
    "section_key",
    ["today", "tasks", "records", "health", "vision", "sections", "settings"],
)
async def test_real_application_every_accepted_navigation_section_invalidates_exact_receipt(
    db,
    fake_ai,
    monkeypatch,
    section_key,
):
    telegram_id = 716_380 + [
        "today",
        "tasks",
        "records",
        "health",
        "vision",
        "sections",
        "settings",
    ].index(section_key)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    key, _receipt = await _seed_runtime_companion_reminder_receipt(
        core,
        db,
        owner,
        suffix=f"section-{section_key}",
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    telegram_user = TelegramUser(telegram_id, "Варвара", False)
    canonical = _runtime_bot_message(application, telegram_id, 118_380 + telegram_id)
    section_update, query = _runtime_weekly_callback_update(
        application,
        telegram_user,
        canonical,
        f"nav:section:{section_key}",
        update_id=42_380 + telegram_id,
    )

    await application.process_update(section_update)

    assert key not in core._nova_companion_status_receipts
    assert [answer["callback_query_id"] for answer in transport.answers].count(query.id) == 1
    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            "Оно уже создано?",
            update_id=42_390 + telegram_id,
            source_message_id=117_390 + telegram_id,
        )
    )
    assert transport.sent[-1]["text"] == "Пока нет — напоминание ещё не создано."
    assert fake_ai.companion_calls == []
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("question", ["Готово?", "Всё готово?", "Ну что, всё готово?"])
async def test_real_application_contextual_status_uses_live_receipt_before_group_minus_two_invalidation(
    db,
    fake_ai,
    monkeypatch,
    question,
):
    telegram_id = 716_400 + ["Готово?", "Всё готово?", "Ну что, всё готово?"].index(question)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    key, receipt = await _seed_runtime_companion_reminder_receipt(
        core,
        db,
        owner,
        suffix=f"generic-{telegram_id}",
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            question,
            update_id=42_400 + telegram_id,
            source_message_id=117_400 + telegram_id,
        )
    )

    assert transport.sent[-1]["text"].startswith("Да, напоминание создано на ")
    assert core._nova_companion_status_receipts[key] is receipt
    assert fake_ai.companion_calls == []
    assert core._nova_companion_tasks == set()


async def test_real_application_accepted_system_action_invalidates_old_status_receipt(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 716_374
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    draft = await core.draft_service.create(
        user_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
        source="text",
        raw_text="Временный черновик",
        parsed=ParsedThought(kind="note", title="Временный черновик"),
    )
    await core.conversation.begin_system_action(
        telegram_id,
        telegram_id,
        "discard_all_active_drafts",
        [
            {
                "id": draft.id,
                "version": draft.version,
                "affected": True,
                "preview_message_id": None,
            }
        ],
    )
    key, receipt = await _seed_runtime_companion_reminder_receipt(
        core,
        db,
        owner,
        suffix="system",
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            "да, удалить",
            update_id=42_377,
            source_message_id=117_377,
        )
    )
    assert key not in core._nova_companion_status_receipts
    assert (await core.draft_service.get(draft.id)).status == "discarded"

    newer = replace(receipt, inbox_item_version=receipt.inbox_item_version + 1)
    core._nova_companion_status_receipts[key] = newer
    assert (
        core.nova_companion_invalidate_status_receipt_exact(
            telegram_id,
            telegram_id,
            expected_receipt=receipt,
        )
        is False
    )
    assert core._nova_companion_status_receipts[key] is newer
    core._nova_companion_status_receipts.pop(key)
    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            "Оно уже создано?",
            update_id=42_378,
            source_message_id=117_378,
        )
    )
    assert transport.sent[-1]["text"] == "Пока нет — напоминание ещё не создано."
    assert fake_ai.companion_calls == []


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize(
    ("phrase", "memory_key", "memory_value", "stored_value", "display_value"),
    [
        (
            "Нова, отвечай мне коротко, пожалуйста",
            "response_length",
            "short",
            "response_length=short",
            "короткие ответы",
        ),
        (
            "Говори со мной спокойно",
            "tone",
            "calm",
            "tone=calm",
            "спокойный тон",
        ),
    ],
)
async def test_real_application_nova_brain_persists_and_recalls_exact_observed_memory(
    db,
    fake_ai,
    monkeypatch,
    source,
    phrase,
    memory_key,
    memory_value,
    stored_value,
    display_value,
):
    telegram_id = 720_300 + {"response_length": 0, "tone": 10}[memory_key] + (source == "voice")
    answer = "Могу предложить короткое упражнение. Какой вариант тебе ближе?"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer=answer,
        memory_candidate=NovaCompanionMemoryCandidate(
            category="preference",
            key=memory_key,
            value=memory_value,
            evidence=phrase,
            salience=5,
        ),
    )
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=True,
            nova_conversation_brain_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    if source == "text":
        update_value = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=43_300,
            source_message_id=118_300,
        )
    else:
        update_value, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=43_301,
            source_message_id=118_301,
            progress_message_id=119_301,
        )
    await application.process_update(update_value)

    assert len(fake_ai.companion_calls) == 1
    assert fake_ai.companion_brain_calls[0] is not None
    async with db.sessions() as session:
        state = await session.scalar(select(NovaDialogueState))
        memory = await session.scalar(select(NovaObservedMemory))
        assert state is None
        assert memory is not None and memory.normalized_value == stored_value
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            "Что ты обо мне помнишь?",
            update_id=43_302,
            source_message_id=118_302,
        )
    )
    assert display_value in transport.sent[-1]["text"]
    assert len(fake_ai.companion_calls) == 1
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_nova_brain_replaces_structured_setting_once_per_turn(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 725_000 + (source == "voice")
    turns = (
        ("Отвечай мне коротко", "short", "Поняла, буду отвечать коротко."),
        ("Отвечай мне подробно", "detailed", "Поняла, буду отвечать подробно."),
    )
    transcription = RuntimeTranscription(turns[0][0])
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=True,
            nova_conversation_brain_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    for index, (phrase, value, answer) in enumerate(turns):
        fake_ai.companion_provider_result = NovaCompanionProviderResponse(
            answer=answer,
            memory_candidate=NovaCompanionMemoryCandidate(
                category="preference",
                key="response_length",
                value=value,
                evidence=phrase,
            ),
        )
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=57_000 + index,
                source_message_id=201_000 + index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=58_000 + index,
                source_message_id=202_000 + index,
                progress_message_id=203_000 + index,
            )
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()

    assert len(fake_ai.companion_calls) == 2
    assert fake_ai.companion_brain_calls[1] is not None
    assert [item.value for item in fake_ai.companion_brain_calls[1].memories] == [
        "response_length=short"
    ]
    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    assert all(answer in delivered for _phrase, _value, answer in turns)
    async with db.sessions() as session:
        contents = tuple(
            await session.scalars(
                select(ConversationMessage.content).order_by(ConversationMessage.id)
            )
        )
        assert contents == tuple(
            item for phrase, _value, answer in turns for item in (phrase, answer)
        )
        rows = list(
            (
                await session.scalars(select(NovaObservedMemory).order_by(NovaObservedMemory.id))
            ).all()
        )
        assert [(row.normalized_value, row.status) for row in rows] == [
            ("response_length=short", "superseded"),
            ("response_length=detailed", "active"),
        ]
        assert all(row.owner_id == owner.id for row in rows)
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    restarted = NovaBrainService(db)
    current = await restarted.list_current(
        telegram_actor_id=telegram_id,
        expected_access_version=owner.access_version,
        policy=NovaBrainPolicy(enabled=True, admin_only=False),
    )
    assert [item.value for item in current] == ["response_length=detailed"]
    assert core._nova_companion_tasks == set()


async def test_real_application_nova_brain_disabled_is_exact_stage_b_fallback(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 720_302
    phrase = "Я предпочитаю короткие ответы"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Поняла.",
        dialogue_state_update=NovaCompanionDialogueStateUpdate(active_topic="короткие ответы"),
        memory_candidate=NovaCompanionMemoryCandidate(
            category="preference",
            key="response_length",
            value="short",
            evidence=phrase,
        ),
    )
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=False,
        ),
        db,
        fake_ai,
        FakeTranscription(),
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    await application.process_update(
        _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=43_303,
            source_message_id=118_303,
        )
    )

    assert transport.sent[-1]["text"] == "Поняла."
    assert fake_ai.companion_brain_calls == [None]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0


_NOVA_BRAIN_EVIDENCE_SLICING_CASES = (
    (
        "Дочь сказала: «Меня зовут Маша»",
        "Меня зовут Маша",
        "identity",
        "identity",
        "display_name=Маша",
    ),
    (
        "Мой муж говорит: «Я мужчина»",
        "Я мужчина",
        "identity",
        "identity",
        "grammatical_address=masculine",
    ),
    (
        "Коллега просит: «Отвечай мне коротко»",
        "Отвечай мне коротко",
        "preference",
        "response_length",
        "short",
    ),
    (
        "Не повторяй фразу «Говори со мной спокойно»",
        "Говори со мной спокойно",
        "preference",
        "tone",
        "calm",
    ),
    (
        "В инструкции написано: «Напоминай мне мягко»",
        "Напоминай мне мягко",
        "preference",
        "reminder_style",
        "gentle",
    ),
)


@pytest.mark.parametrize(
    ("case_index", "phrase", "evidence", "category", "memory_key", "memory_value"),
    tuple((index, *case) for index, case in enumerate(_NOVA_BRAIN_EVIDENCE_SLICING_CASES)),
)
@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("brain_enabled", [False, True], ids=["brain-off", "brain-on"])
async def test_real_application_nova_brain_rejects_provider_evidence_slicing(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    case_index,
    phrase,
    evidence,
    category,
    memory_key,
    memory_value,
    source,
    brain_enabled,
):
    telegram_id = 724_000 + case_index * 10 + (1 if source == "voice" else 0)
    safe_answer = "Поняла контекст. Давай продолжим разговор."
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer=safe_answer,
        memory_candidate=NovaCompanionMemoryCandidate(
            category=category,
            key=memory_key,
            value=memory_value,
            evidence=evidence,
        ),
    )
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=brain_enabled,
            nova_conversation_brain_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    if source == "text":
        update_value = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=55_000 + case_index,
            source_message_id=198_000 + case_index,
        )
    else:
        update_value, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=56_000 + case_index,
            source_message_id=199_000 + case_index,
            progress_message_id=200_000 + case_index,
        )
    await application.process_update(update_value)
    await core._drain_nova_companion_tasks()

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    assert safe_answer in delivered
    assert phrase not in delivered
    assert len(fake_ai.companion_calls) == 1
    projection = fake_ai.companion_brain_calls[0]
    assert projection is None or all(memory_value not in item.value for item in projection.memories)
    async with db.sessions() as session:
        contents = tuple(
            await session.scalars(
                select(ConversationMessage.content).order_by(ConversationMessage.id)
            )
        )
        assert contents == (phrase, safe_answer)
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert phrase not in caplog.text
    assert evidence not in caplog.text
    assert memory_value not in caplog.text
    assert core._nova_companion_tasks == set()


_NOVA_BRAIN_FREE_TEXT_PRIVACY_CASES = (
    "Я постоянно лечусь от биполярки",
    "Я часто употребляю кокаин",
    "Я люблю секс без презерватива",
    "Я часто хожу в мечеть",
    "Я часто поддерживаю Навального",
    "Я постоянно получаю 500 тысяч",
    "Я постоянно забираю Машу из школы",
    "Каждую неделю прохожу длительное лечение",
    "Регулярно принимаю сильные вещества",
    "Мне нравится близость без защиты",
    "По пятницам бываю на общей молитве",
    "Я за известного оппозиционного деятеля",
    "Каждый месяц мне приходит очень крупная сумма",
    "После уроков всегда встречаю Машу",
    "Я люблю готовить вместе с Мариной",
)


@pytest.mark.parametrize(
    ("case_index", "phrase"),
    tuple(enumerate(_NOVA_BRAIN_FREE_TEXT_PRIVACY_CASES)),
)
@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("brain_enabled", [False, True], ids=["brain-off", "brain-on"])
async def test_real_application_nova_brain_free_text_is_never_automatic_memory(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    case_index,
    phrase,
    source,
    brain_enabled,
):
    telegram_id = 723_000 + case_index * 10 + (1 if source == "voice" else 0)
    safe_answer = "Спасибо, что поделился. Давай продолжим разговор."
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer=safe_answer,
        memory_candidate=NovaCompanionMemoryCandidate(
            category="preference",
            key="tone",
            value="calm",
            evidence=phrase,
        ),
    )
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=brain_enabled,
            nova_conversation_brain_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    if source == "text":
        update_value = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=53_000 + case_index,
            source_message_id=195_000 + case_index,
        )
    else:
        update_value, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=54_000 + case_index,
            source_message_id=196_000 + case_index,
            progress_message_id=197_000 + case_index,
        )
    await application.process_update(update_value)
    await core._drain_nova_companion_tasks()

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    assert safe_answer in delivered
    assert phrase not in delivered
    assert len(fake_ai.companion_calls) == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert phrase not in caplog.text
    assert core._nova_companion_tasks == set()


def _nova_conversation_brain_eval_cases() -> list[dict[str, object]]:
    path = Path(__file__).parent / "evals/nova_conversation_brain_cases.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    return loaded


@pytest.mark.parametrize(
    "case",
    _nova_conversation_brain_eval_cases(),
    ids=lambda case: str(case["id"]),
)
@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("brain_enabled", [False, True], ids=["brain-off", "brain-on"])
async def test_real_application_nova_conversation_brain_eval_routes_and_effects(
    db,
    fake_ai,
    monkeypatch,
    case,
    source,
    brain_enabled,
):
    """Execute every offline eval through the real PTB Application route."""

    case_id = str(case["id"])
    phrase = str(case["input"])
    expected_route = str(case["expected_route"])
    telegram_id = 721_000 + sum(ord(char) for char in f"{case_id}:{source}:{brain_enabled}")
    transcription = RuntimeTranscription(phrase)
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=brain_enabled,
            nova_conversation_brain_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    async with db.session() as session:
        await session.execute(
            update(User)
            .where(User.id == owner.id)
            .values(display_name="Назар", location_city="Москва", timezone="Europe/Moscow")
        )
        if case_id in {"long-term-recall", "exact-forget"}:
            session.add(
                NovaObservedMemory(
                    owner_id=owner.id,
                    category="preference",
                    normalized_value=(
                        "response_length=short"
                        if case_id == "long-term-recall"
                        else "response_length=short"
                    ),
                    content_fingerprint=("a" if case_id == "long-term-recall" else "b") * 64,
                    source_kind="conversation",
                    source_session_id=1,
                    source_message_id=1,
                    source_receipt=("c" if case_id == "long-term-recall" else "d") * 64,
                    status="active",
                    salience=5,
                    revision=1,
                )
            )
        elif case_id == "structured-setting-correction":
            session.add(
                NovaObservedMemory(
                    owner_id=owner.id,
                    category="preference",
                    normalized_value="response_length=normal",
                    content_fingerprint="e" * 64,
                    source_kind="conversation",
                    source_session_id=1,
                    source_message_id=1,
                    source_receipt="f" * 64,
                    status="active",
                    salience=4,
                    revision=1,
                )
            )
    owner = await core._user(telegram_id)
    prior_assistant = case.get("prior_assistant")
    if isinstance(prior_assistant, str):
        await core.conversation.append(
            telegram_id,
            telegram_id,
            role="assistant",
            content=prior_assistant,
            source="text",
            intent="companion_answer",
        )

    safe_answer = f"Безопасный eval-ответ: {case_id}."
    provider_kwargs: dict[str, object] = {"answer": safe_answer}
    memory = case.get("memory")
    if isinstance(memory, dict):
        provider_kwargs["memory_candidate"] = NovaCompanionMemoryCandidate.model_validate(memory)
    if case_id == "false-execution-claim":
        provider_kwargs.update(
            answer=phrase,
            capture=NovaCompanionProviderCapture(
                kind="task",
                title="Стрижка",
                evidence=phrase,
            ),
        )
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(**provider_kwargs)
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    async with db.sessions() as session:
        before = {
            "conversation": await session.scalar(select(func.count(ConversationMessage.id))),
            "state": await session.scalar(select(func.count(NovaDialogueState.id))),
            "memory": await session.scalar(select(func.count(NovaObservedMemory.id))),
            "draft": await session.scalar(select(func.count(DraftInboxItem.id))),
            "inbox": await session.scalar(select(func.count(InboxItem.id))),
            "reminder": await session.scalar(select(func.count(TaskReminder.id))),
        }

    if source == "text":
        update_value = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=51_000,
            source_message_id=191_000,
        )
    else:
        update_value, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=51_001,
            source_message_id=191_001,
            progress_message_id=192_001,
        )
    await application.process_update(update_value)
    await core._drain_nova_companion_tasks()

    local_brain_route = brain_enabled and expected_route in {"memory_recall", "memory_forget"}
    expected_provider_calls = (
        1
        if expected_route == "companion"
        or (expected_route in {"memory_recall", "memory_forget"} and not brain_enabled)
        else 0
    )
    assert len(fake_ai.companion_calls) == expected_provider_calls
    if expected_provider_calls:
        assert fake_ai.companion_brain_calls == [
            None if not brain_enabled else fake_ai.companion_brain_calls[0]
        ]
        assert (fake_ai.companion_brain_calls[0] is not None) is brain_enabled

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    if case_id == "false-execution-claim":
        assert phrase not in delivered
        assert NOVA_COMPANION_NOT_EXECUTED_TEXT in delivered
    elif expected_provider_calls:
        assert safe_answer in delivered
    if expected_route == "identity":
        assert any("Назар" in item or "Москва" in item for item in delivered)
    elif expected_route == "reminder":
        reminder_session = await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        assert reminder_session is not None
        assert reminder_session.phase is ReminderFlowPhase.PREVIEW
    elif local_brain_route:
        assert delivered
        if expected_route == "memory_forget":
            assert any(
                item.get("reply_markup") is not None for item in (*transport.sent, *transport.edits)
            )
        else:
            assert any("короткие ответы" in item for item in delivered)

    async with db.sessions() as session:
        after = {
            "conversation": await session.scalar(select(func.count(ConversationMessage.id))),
            "state": await session.scalar(select(func.count(NovaDialogueState.id))),
            "memory": await session.scalar(select(func.count(NovaObservedMemory.id))),
            "draft": await session.scalar(select(func.count(DraftInboxItem.id))),
            "inbox": await session.scalar(select(func.count(InboxItem.id))),
            "reminder": await session.scalar(select(func.count(TaskReminder.id))),
        }
        memories = tuple(await session.scalars(select(NovaObservedMemory)))
        states = tuple(await session.scalars(select(NovaDialogueState)))
    assert after["conversation"] - before["conversation"] == 2 * expected_provider_calls
    assert after["state"] == before["state"]
    expected_memory_delta = 1 if brain_enabled and case_id == "structured-setting-correction" else 0
    assert after["memory"] - before["memory"] == expected_memory_delta
    assert after["draft"] == before["draft"]
    assert after["inbox"] == before["inbox"]
    assert after["reminder"] == before["reminder"]
    assert all(item.owner_id == owner.id for item in memories)
    assert all(
        item.owner_id == owner.id
        and item.telegram_user_id == telegram_id
        and item.chat_id == telegram_id
        and item.access_version == owner.access_version
        for item in states
    )
    if not brain_enabled:
        assert after["state"] == 0
        assert after["memory"] == before["memory"]
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_nova_brain_exact_identity_helper_reminder_restart_transcript(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    telegram_id = 722_000 if source == "text" else 722_001
    phrases = (
        "Меня зовут Назар. Я мужчина",
        "Будь моим жизненным помощником",
        "Мне нужно напоминать о главном",
        "Давай",
    )
    transcription = RuntimeTranscription(phrases[0])
    settings = runtime_settings(
        database_url=db.url,
        enable_nova_companion=True,
        nova_companion_admin_only=False,
        enable_nova_conversation_brain=True,
        nova_conversation_brain_admin_only=False,
        conversation_context_messages=10,
    )
    core = FutureSelfBot(settings, db, fake_ai, transcription)
    application = core.build()
    application._initialized = True
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    async def edit_reply_markup(self, *args, **kwargs):
        del self, args
        transport.edits.append(dict(kwargs))
        return transport.make_message(
            int(kwargs["chat_id"]),
            int(kwargs["message_id"]),
        )

    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_reply_markup)

    responses = (
        NovaCompanionProviderResponse(
            answer="Назар, рад знакомству. Готов помочь тебе двигаться дальше.",
            memory_candidate=NovaCompanionMemoryCandidate(
                category="identity",
                key="identity",
                value="display_name=Назар;grammatical_address=masculine",
                evidence=phrases[0],
                salience=5,
            ),
        ),
        NovaCompanionProviderResponse(
            answer="Буду помогать тебе держать курс и выбирать следующий шаг.",
            dialogue_state_update=NovaCompanionDialogueStateUpdate(
                active_topic="жизненным помощником",
            ),
        ),
        NovaCompanionProviderResponse(
            answer="Могу предложить создать настоящее напоминание о главном.",
            reminder_offer=NovaCompanionProviderReminderOffer(
                title="главном",
                evidence=phrases[2],
            ),
            dialogue_state_update=NovaCompanionDialogueStateUpdate(
                active_topic="напоминать о главном",
            ),
        ),
    )
    for index, phrase in enumerate(phrases):
        if index < len(responses):
            fake_ai.companion_provider_result = responses[index]
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=52_000 + index,
                source_message_id=193_000 + index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=52_000 + index,
                source_message_id=193_000 + index,
                progress_message_id=194_000 + index,
            )
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()

    assert len(fake_ai.companion_calls) == 3
    nrem_markups = [
        item["reply_markup"]
        for item in (*transport.sent, *transport.edits)
        if item.get("reply_markup") is not None
        and any(
            str(button.callback_data).startswith("nrem:")
            for row in item["reply_markup"].inline_keyboard
            for button in row
        )
    ]
    assert len(nrem_markups) == 1
    reminder_session = await core.reminder_sessions.current(
        owner_id=owner.id,
        telegram_user_id=telegram_id,
        chat_id=telegram_id,
    )
    assert reminder_session is not None
    assert reminder_session.title == "главном"
    assert reminder_session.phase is not ReminderFlowPhase.TITLE
    async with db.sessions() as session:
        actor = await session.get(User, owner.id)
        identity = await session.scalar(
            select(NovaObservedMemory).where(NovaObservedMemory.category == "identity")
        )
        state = await session.scalar(select(NovaDialogueState))
        assert actor is not None and actor.display_name is None
        assert identity is not None and "назар" in identity.normalized_value
        assert state is not None and state.active_topic == "напоминать о главном"
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0

    for index in range(settings.conversation_context_messages + 4):
        await core.conversation.append(
            telegram_id,
            telegram_id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"безопасная промежуточная реплика {index}",
            source="text",
            intent="companion_user" if index % 2 == 0 else "companion_answer",
        )

    restarted = FutureSelfBot(settings, db, fake_ai, transcription)
    restarted_application = restarted.build()
    restarted_application._initialized = True
    restarted_transport = _patch_runtime_weekly_transport(monkeypatch, restarted_application)
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Назар, продолжим с главным: выберем время для напоминания?",
    )
    follow_up = "Как продолжим?"
    if source == "text":
        follow_up_update = _runtime_text_update(
            restarted_application,
            telegram_id,
            follow_up,
            update_id=52_100,
            source_message_id=193_100,
        )
    else:
        transcription.transcript = follow_up
        follow_up_update, _progress = _runtime_voice_update(
            restarted_application,
            telegram_id,
            update_id=52_101,
            source_message_id=193_101,
            progress_message_id=194_101,
        )
    await restarted_application.process_update(follow_up_update)
    await restarted._drain_nova_companion_tasks()

    assert len(fake_ai.companion_calls) == 4
    restarted_projection = fake_ai.companion_brain_calls[-1]
    assert restarted_projection is not None
    assert restarted_projection.working_state.active_topic == "напоминать о главном"
    assert any(
        item.category == "identity" and "назар" in item.value
        for item in restarted_projection.memories
    )
    delivered = [
        str(item.get("text", ""))
        for item in (*restarted_transport.sent, *restarted_transport.edits)
    ]
    assert "Назар, продолжим с главным: выберем время для напоминания?" in delivered
    assert not any(
        item.get("reply_markup") is not None
        and any(
            str(button.callback_data).startswith("nrem:")
            for row in item["reply_markup"].inline_keyboard
            for button in row
        )
        for item in (*restarted_transport.sent, *restarted_transport.edits)
    )
    identity_question = "Как меня зовут?"
    if source == "text":
        identity_update = _runtime_text_update(
            restarted_application,
            telegram_id,
            identity_question,
            update_id=52_102,
            source_message_id=193_102,
        )
    else:
        transcription.transcript = identity_question
        identity_update, _progress = _runtime_voice_update(
            restarted_application,
            telegram_id,
            update_id=52_103,
            source_message_id=193_103,
            progress_message_id=194_103,
        )
    await restarted_application.process_update(identity_update)
    await restarted._drain_nova_companion_tasks()
    delivered = [
        str(item.get("text", ""))
        for item in (*restarted_transport.sent, *restarted_transport.edits)
    ]
    assert "Из твоих слов: тебя зовут Назар." in delivered
    assert len(fake_ai.companion_calls) == 4
    assert restarted._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_stage8c1_provider_proposals_fail_independently(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    source,
):
    sentinel = "PRIVATE_STAGE8C1_PROPOSAL_SENTINEL"
    cases = (
        (
            "capture",
            {
                "answer": "Безопасный ответ capture.",
                "capture": json.dumps(
                    {
                        "kind": "idea",
                        "title": "мысль",
                        "evidence": "Обычный разговор capture",
                        "extra": sentinel,
                    },
                    ensure_ascii=False,
                ),
            },
            "invalid_capture",
        ),
        (
            "reminder",
            {
                "answer": "Безопасный ответ reminder.",
                "reminder_offer": json.dumps(
                    {"title": [sentinel], "evidence": "Обычный разговор reminder"},
                    ensure_ascii=False,
                ),
            },
            "invalid_reminder_offer",
        ),
        (
            "dialogue",
            {
                "answer": "Безопасный ответ dialogue.",
                "dialogue_state_update": json.dumps(
                    {"requested_action": sentinel},
                    ensure_ascii=False,
                ),
            },
            "invalid_dialogue_state",
        ),
        (
            "memory",
            {
                "answer": "Безопасный ответ memory.",
                "memory_candidate": json.dumps(
                    {
                        "category": sentinel,
                        "key": "identity",
                        "value": "display_name=Ольга",
                        "evidence": "Меня зовут Ольга",
                    },
                    ensure_ascii=False,
                ),
            },
            "invalid_memory_candidate",
        ),
        (
            "conflict",
            {
                "answer": "Безопасный ответ conflict.",
                "capture": json.dumps(
                    {
                        "kind": "note",
                        "title": "разговор",
                        "evidence": "Обычный разговор conflict",
                    },
                    ensure_ascii=False,
                ),
                "reminder_offer": json.dumps(
                    {"title": "разговор", "evidence": "Обычный разговор conflict"},
                    ensure_ascii=False,
                ),
            },
            "conflicting_actions",
        ),
    )
    transcription = RuntimeTranscription("unused")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=True,
            nova_conversation_brain_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    with caplog.at_level(logging.WARNING):
        for index, (case_id, raw, diagnostic) in enumerate(cases):
            telegram_id = 723_000 + index + (100 if source == "voice" else 0)
            await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
            phrase = f"Обычный разговор {case_id}"
            fake_ai.companion_provider_raw_result = raw
            if source == "text":
                update_value = _runtime_text_update(
                    application,
                    telegram_id,
                    phrase,
                    update_id=53_000 + index,
                    source_message_id=195_000 + index,
                )
            else:
                transcription.transcript = phrase
                update_value, _progress = _runtime_voice_update(
                    application,
                    telegram_id,
                    update_id=53_100 + index,
                    source_message_id=195_100 + index,
                    progress_message_id=196_100 + index,
                )
            await application.process_update(update_value)
            await core._drain_nova_companion_tasks()
            assert diagnostic in caplog.text

        telegram_id = 723_090 if source == "text" else 723_190
        await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
        fake_ai.companion_provider_raw_result = {"answer": 42, "capture": None}
        phrase = "Обычный разговор invalid answer"
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=53_090,
                source_message_id=195_090,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=53_190,
                source_message_id=195_190,
                progress_message_id=196_190,
            )
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    assert all(
        (
            NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT
            if case_id in {"reminder", "conflict"}
            else str(raw["answer"])
        )
        in delivered
        for case_id, raw, _diagnostic in cases
    )
    assert NOVA_COMPANION_UNAVAILABLE_TEXT in delivered
    assert "invalid_answer" in caplog.text
    assert sentinel not in caplog.text
    assert len(fake_ai.companion_calls) == len(cases) + 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == len(cases) * 2
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_stage8c1_recall_and_identity_are_local_semantic_routes(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    recall_phrases = (
        "напомни плиз",
        "напомни про наш с тобой разговор о ментальных тренировках, что именно мы обсуждали?",
        "напомни, о чём мы говорили",
        "напомни, что именно мы обсуждали",
        "вспомни наш разговор",
        "Ты помнишь наш разговор?",
        "Помнишь, о чём мы говорили?",
        "Мы недавно разговаривали о ментальных тренировках. Ты помнишь наш разговор?",
        "Нова, ты помнишь наш разговор?",
        "Nova, помнишь, о чём мы говорили?",
    )
    transcription = RuntimeTranscription("unused")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
            enable_nova_conversation_brain=True,
            nova_conversation_brain_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    for index, phrase in enumerate(recall_phrases):
        telegram_id = 724_000 + index + (100 if source == "voice" else 0)
        owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=54_000 + index,
                source_message_id=197_000 + index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=54_100 + index,
                source_message_id=197_100 + index,
                progress_message_id=198_100 + index,
            )
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()
        assert (
            await core.reminder_sessions.current(
                owner_id=owner.id,
                telegram_user_id=telegram_id,
                chat_id=telegram_id,
            )
            is None
        )

    identity_id = 724_090 if source == "text" else 724_190
    owner = await _runtime_stage7c_user(core, db, identity_id, tier="subscriber")
    async with db.session() as session:
        await session.execute(update(User).where(User.id == owner.id).values(display_name="Назар"))
    identity_phrase = "Марина сказала: «Меня зовут Ольга». Как меня зовут?"
    if source == "text":
        identity_update = _runtime_text_update(
            application,
            identity_id,
            identity_phrase,
            update_id=54_090,
            source_message_id=197_090,
        )
    else:
        transcription.transcript = identity_phrase
        identity_update, _progress = _runtime_voice_update(
            application,
            identity_id,
            update_id=54_190,
            source_message_id=197_190,
            progress_message_id=198_190,
        )
    await application.process_update(identity_update)
    await core._drain_nova_companion_tasks()

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    assert any("содержание разговора" in item for item in delivered)
    assert any("доступном контексте" in item for item in delivered)
    assert "Да, тебя зовут Назар." in delivered
    assert fake_ai.companion_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_stage8c1_release_blocker_routes(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    transcription = RuntimeTranscription("unused")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    id_offset = 100 if source == "voice" else 0

    async def process(telegram_id: int, phrase: str, case_index: int) -> None:
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=55_000 + case_index,
                source_message_id=199_000 + case_index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=55_100 + case_index,
                source_message_id=199_100 + case_index,
                progress_message_id=200_100 + case_index,
            )
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()

    rejected_id = 725_000 + id_offset
    await _runtime_stage7c_user(core, db, rejected_id, tier="subscriber")
    raw_answer = "Могу поставить напоминание — нажми кнопку. PRIVATE_RUNTIME_REJECTED"
    fake_ai.companion_provider_raw_result = {
        "answer": raw_answer,
        "reminder_offer": {"title": ["врача"], "evidence": "А вдруг забуду?"},
    }
    await process(rejected_id, "А вдруг забуду?", 0)

    ambiguous_id = 725_010 + id_offset
    ambiguous_owner = await _runtime_stage7c_user(
        core,
        db,
        ambiguous_id,
        tier="subscriber",
    )
    await core.conversation.append(
        ambiguous_id,
        ambiguous_id,
        role="user",
        content="Люблю чай.",
        source="text",
        intent="companion_user",
    )
    await core.conversation.append(
        ambiguous_id,
        ambiguous_id,
        role="assistant",
        content="Ясно.",
        source="text",
        intent="companion_answer",
    )
    fake_ai.companion_provider_raw_result = None
    await process(ambiguous_id, "напомни плиз", 1)

    recall_id = 725_020 + id_offset
    recall_owner = await _runtime_stage7c_user(core, db, recall_id, tier="subscriber")
    await process(recall_id, "Ты помнишь наш разговор?", 2)

    temporal_id = 725_030 + id_offset
    temporal_owner = await _runtime_stage7c_user(core, db, temporal_id, tier="subscriber")
    await process(temporal_id, "напомни про наш разговор завтра в 19:00", 3)

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    assert NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT in delivered
    assert raw_answer not in delivered
    assert all("PRIVATE_RUNTIME_REJECTED" not in item for item in delivered)
    assert NOVA_COMPANION_RECALL_CLARIFICATION_TEXT in delivered
    assert NOVA_COMPANION_RECALL_UNAVAILABLE_TEXT in delivered
    assert len(fake_ai.companion_calls) == 1
    assert (
        await core.reminder_sessions.current(
            owner_id=ambiguous_owner.id,
            telegram_user_id=ambiguous_id,
            chat_id=ambiguous_id,
        )
        is None
    )
    assert (
        await core.reminder_sessions.current(
            owner_id=recall_owner.id,
            telegram_user_id=recall_id,
            chat_id=recall_id,
        )
        is None
    )
    reminder_session = await core.reminder_sessions.current(
        owner_id=temporal_owner.id,
        telegram_user_id=temporal_id,
        chat_id=temporal_id,
    )
    assert reminder_session is not None
    assert reminder_session.phase is ReminderFlowPhase.PREVIEW
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert raw_answer not in contents
        assert NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT in contents
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 4
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize(
    "scenario",
    [
        "reversed_valid_reminder",
        "malformed_reminder_action",
        "valid_capture_variant",
        "conflicting_actions",
        "answer_only_capture",
        "safe_independent_recall",
    ],
)
async def test_real_application_recall_suppresses_action_dependent_answer_atomically(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    source,
    scenario,
):
    transcription = RuntimeTranscription("unused")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    telegram_id = 726_000 + (100 if source == "voice" else 0)
    owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    prior = "Мы обсуждали ментальные тренировки завтра в 19:00."
    await core.conversation.append(
        telegram_id,
        telegram_id,
        role="user",
        content=prior,
        source="text",
        intent="companion_user",
    )
    await core.conversation.append(
        telegram_id,
        telegram_id,
        role="assistant",
        content="Это был небольшой ежедневный эксперимент.",
        source="text",
        intent="companion_answer",
    )
    phrase = "напомни, о чём мы говорили"
    sentinel = f"PRIVATE_RUNTIME_SUPPRESSED_ACTION_{scenario.upper()}"
    raw_answer = {
        "reversed_valid_reminder": (f"Ниже можно поставить напоминание. {sentinel}"),
        "malformed_reminder_action": (f"Под сообщением доступно действие «Напомнить». {sentinel}"),
        "valid_capture_variant": (f"Для этого доступен вариант «Сохранить». {sentinel}"),
        "conflicting_actions": (
            f"Под сообщением доступны действия «Сохранить» и «Напомнить». {sentinel}"
        ),
        "answer_only_capture": f"Ниже можно сохранить этот разговор. {sentinel}",
        "safe_independent_recall": (
            "Мы говорили о том, что напоминания помогают тебе не забывать о делах."
        ),
    }[scenario]
    raw_result: dict[str, object] = {"answer": raw_answer}
    if scenario in {
        "reversed_valid_reminder",
        "conflicting_actions",
        "safe_independent_recall",
    }:
        raw_result["reminder_offer"] = {
            "title": "ментальные тренировки",
            "schedule_wording": "завтра в 19:00",
            "evidence": prior,
        }
    if scenario == "malformed_reminder_action":
        raw_result["reminder_offer"] = {"title": ["разговор"], "evidence": phrase}
    if scenario in {"valid_capture_variant", "conflicting_actions"}:
        raw_result["capture"] = {
            "kind": "note",
            "title": "разговор",
            "evidence": phrase,
        }
    fake_ai.companion_provider_raw_result = raw_result
    if source == "text":
        update_value = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=56_000,
            source_message_id=201_000,
        )
    else:
        transcription.transcript = phrase
        update_value, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=56_100,
            source_message_id=201_100,
            progress_message_id=202_100,
        )

    with caplog.at_level(logging.INFO):
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    expected_answer = (
        raw_answer
        if scenario == "safe_independent_recall"
        else NOVA_COMPANION_RECALL_ACTION_SUPPRESSED_TEXT
    )
    assert expected_answer in delivered
    if scenario != "safe_independent_recall":
        assert raw_answer not in delivered
    assert all(sentinel not in item for item in delivered)
    assert len(fake_ai.companion_calls) == 1
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert contents == (
            prior,
            "Это был небольшой ежедневный эксперимент.",
            phrase,
            expected_answer,
        )
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert core.nova_companion_captures._screens == {}
    assert core.nova_companion_captures._capabilities == {}
    assert core.nova_companion_reminders._screens == {}
    assert core.nova_companion_reminders._capabilities == {}
    assert (
        await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        is None
    )
    assert sentinel not in caplog.text
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_today_recall_wording_uses_reminder_flow(
    db,
    fake_ai,
    monkeypatch,
    source,
):
    cases = (
        ("напомни про наш разговор сегодня", ReminderFlowPhase.TIME),
        ("напомни про наш разговор на сегодня", ReminderFlowPhase.TIME),
        ("напомни сегодня про наш разговор", ReminderFlowPhase.TIME),
        ("напомни про наш разговор сегодня в 19:00", ReminderFlowPhase.PREVIEW),
    )
    transcription = RuntimeTranscription("unused")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    _patch_runtime_weekly_transport(monkeypatch, application)
    id_offset = 100 if source == "voice" else 0

    for index, (phrase, expected_phase) in enumerate(cases):
        telegram_id = 727_000 + id_offset + index
        owner = await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
        if source == "text":
            update_value = _runtime_text_update(
                application,
                telegram_id,
                phrase,
                update_id=57_000 + index,
                source_message_id=203_000 + index,
            )
        else:
            transcription.transcript = phrase
            update_value, _progress = _runtime_voice_update(
                application,
                telegram_id,
                update_id=57_100 + index,
                source_message_id=203_100 + index,
                progress_message_id=204_100 + index,
            )
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()
        reminder_session = await core.reminder_sessions.current(
            owner_id=owner.id,
            telegram_user_id=telegram_id,
            chat_id=telegram_id,
        )
        assert reminder_session is not None
        assert reminder_session.phase is expected_phase

    assert fake_ai.companion_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert core._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_application_unavailable_recall_bounds_full_telegram_input(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    source,
):
    transcription = RuntimeTranscription("unused")
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_nova_companion=True,
            nova_companion_admin_only=False,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    telegram_id = 728_000 + (100 if source == "voice" else 0)
    await _runtime_stage7c_user(core, db, telegram_id, tier="subscriber")
    prefix = "напомни про наш разговор о "
    sentinel = "PRIVATE_RUNTIME_RECALL_TOPIC_BEYOND_BOUND"
    phrase = prefix + "я" * (4096 - len(prefix) - len(sentinel)) + sentinel
    assert len(phrase) == 4096
    if source == "text":
        update_value = _runtime_text_update(
            application,
            telegram_id,
            phrase,
            update_id=58_000,
            source_message_id=205_000,
        )
    else:
        transcription.transcript = phrase
        update_value, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=58_100,
            source_message_id=205_100,
            progress_message_id=206_100,
        )

    with caplog.at_level(logging.INFO):
        await application.process_update(update_value)
        await core._drain_nova_companion_tasks()

    delivered = [str(item.get("text", "")) for item in (*transport.sent, *transport.edits)]
    assert NOVA_COMPANION_RECALL_UNAVAILABLE_TEXT in delivered
    assert all(len(item.encode("utf-16-le")) // 2 <= 4096 for item in delivered)
    assert all(sentinel not in item for item in delivered)
    assert sentinel not in caplog.text
    assert fake_ai.companion_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert core._nova_companion_tasks == set()
