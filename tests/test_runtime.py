import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import select, text
from telegram import CallbackQuery, Chat, Message, Update
from telegram import User as TelegramUser
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ExtBot,
    MessageHandler,
    TypeHandler,
)

import future_self.main as main_module
from future_self.access import AccessService
from future_self.bot import FutureSelfBot, log_safe_failure
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
from future_self.nova_memory_flow import NovaMemoryFlowPhase
from future_self.repositories import OnboardingRepository, UserRepository
from future_self.schemas import ParsedThought, ReminderTimezoneResolution
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
