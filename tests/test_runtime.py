import asyncio
import gc
import logging
import warnings
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text
from telegram import CallbackQuery, Chat, Message, Update
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
from future_self.access import AccessService
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
    InboxItem,
    NovaMemoryChange,
    NovaMemoryItem,
    OnboardingState,
    User,
)
from future_self.nova_handlers import NOVA_ROOT_TEXT
from future_self.nova_memory import NovaMemoryApplicationCurrent
from future_self.nova_memory_flow import NovaMemoryFlowPhase
from future_self.repositories import OnboardingRepository, UserRepository
from future_self.schemas import IntentResult, ParsedThought, ReminderTimezoneResolution
from future_self.tasks import add_task_state
from future_self.timezones import extract_reminder_timezone_fragment


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
    telegram_user = TelegramUser(telegram_id, False, "Тест")
    bot_user = TelegramUser(123456, True, "Future Self")
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
            text("INSERT INTO alembic_version (version_num) VALUES ('20260811_0026')")
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
    assert sum(isinstance(handler, CallbackQueryHandler) for handler in handlers) == 19
    assert any(
        isinstance(handler, CallbackQueryHandler)
        and handler.callback.__name__ == "nova_memory_callback"
        and getattr(handler.pattern, "pattern", None) == r"^nmem:[A-Za-z0-9_-]+$"
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
