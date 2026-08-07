import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import select, text
from telegram import Chat, Message, Update
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
from future_self.models import InboxItem, OnboardingState, User
from future_self.repositories import OnboardingRepository, UserRepository


class FakeTranscription:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "Тестовая расшифровка"


def runtime_settings(**overrides) -> Settings:
    values = {
        "telegram_bot_token": "123456:TEST-TOKEN-FOR-LOCAL-RUNTIME",
        "ai_api_key": "test-ai-key",
        "ai_model": "test-model",
        "database_url": "sqlite+aiosqlite:///:memory:",
    }
    values.update(overrides)
    return Settings(**values)


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
            text("INSERT INTO alembic_version (version_num) VALUES ('20260806_0024')")
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
    assert sum(isinstance(handler, CallbackQueryHandler) for handler in handlers) == 16
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
