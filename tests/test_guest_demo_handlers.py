from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select
from telegram import Chat, Message, Update
from telegram import User as TelegramUser
from telegram.error import BadRequest, TelegramError
from telegram.ext import ApplicationHandlerStop, ExtBot, TypeHandler

import future_self.access_handlers as access_handlers_module
from future_self.access import AccessService
from future_self.access_handlers import (
    GUEST_ACCESS_CHANGED_TEXT,
    GUEST_DISABLED_TEXT,
    GUEST_FIRST_STEP_INPUT_TEXT,
    GUEST_GLOBAL_EXHAUSTED_TEXT,
    GUEST_LIFETIME_EXHAUSTED_TEXT,
    GUEST_PROCESSING_ALERT,
    GUEST_PROVIDER_ERROR_TEXT,
    GUEST_RESULT_EXPIRED_TEXT,
    GUEST_THOUGHT_INPUT_TEXT,
)
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.guest_access import (
    GuestDemoKind,
    GuestSessionDecision,
    GuestSessionOutcome,
    GuestSessionStatus,
)
from future_self.models import GuestDemoSession, GuestUsageLedger, User
from future_self.schemas import GuestFirstStep, GuestThoughtBreakdown


class ForbiddenTranscription:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls += 1
        raise AssertionError("guest media must not reach transcription")


class DemoMessage:
    def __init__(
        self,
        text: str | None = None,
        *,
        message_id: int = 100,
        voice: Any = None,
    ) -> None:
        self.text = text
        self.message_id = message_id
        self.voice = voice
        self.audio = None
        self.photo: list[Any] = []
        self.document = None
        self.replies: list[dict[str, Any]] = []

    async def reply_text(self, text: str, **kwargs: Any) -> DemoMessage:
        self.replies.append({"text": text, **kwargs})
        return self


class DemoQuery:
    def __init__(
        self,
        data: str,
        message: DemoMessage | None,
        *,
        answer_error: BaseException | None = None,
        edit_error: BaseException | None = None,
    ) -> None:
        self.data = data
        self.message = message
        self.answer_error = answer_error
        self.edit_error = edit_error
        self.answer_calls = 0
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[dict[str, Any]] = []

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        self.answer_calls += 1
        if self.answer_error is not None:
            raise self.answer_error
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        self.edits.append({"text": text, **kwargs})


class DemoTelegramBot:
    def __init__(self) -> None:
        self.command_calls: list[tuple[list[Any], dict[str, Any]]] = []
        self.edits: list[dict[str, Any]] = []
        self.send_calls: list[dict[str, Any]] = []
        self.next_edit_error: TelegramError | None = None
        self.next_edit_error_text: str | None = None
        self.edit_hook: Any = None

    async def set_my_commands(self, commands: list[Any], **kwargs: Any) -> None:
        self.command_calls.append((commands, kwargs))

    async def set_chat_menu_button(self, **kwargs: Any) -> None:
        return None

    async def edit_message_text(self, **kwargs: Any) -> None:
        if self.edit_hook is not None:
            consumed = await self.edit_hook(kwargs)
            if consumed:
                self.edit_hook = None
        if self.next_edit_error is not None and (
            self.next_edit_error_text is None or self.next_edit_error_text in kwargs["text"]
        ):
            error = self.next_edit_error
            self.next_edit_error = None
            self.next_edit_error_text = None
            raise error
        self.edits.append(kwargs)

    async def send_message(self, **kwargs: Any) -> None:
        self.send_calls.append(kwargs)
        raise AssertionError("guest demo must not create fallback messages")


class DemoApplication:
    def __init__(self, *, fail_create_task: bool = False) -> None:
        self.fail_create_task = fail_create_task
        self.tasks: list[asyncio.Task[None]] = []
        self.task_names: list[str | None] = []
        self.user_data_snapshots: list[dict[str, Any]] = []

    def create_task(self, coroutine, *, name: str | None = None):
        if self.fail_create_task:
            raise RuntimeError("private scheduling detail")
        task = asyncio.create_task(coroutine, name=name)
        self.tasks.append(task)
        self.task_names.append(name)
        return task

    async def drain(self) -> None:
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


def make_bot(db: Any, fake_ai: Any, **overrides: Any) -> FutureSelfBot:
    transcription = overrides.pop("_transcription", None) or ForbiddenTranscription()
    values = {
        "_env_file": None,
        "telegram_bot_token": "123456:test-token",
        "ai_api_key": "test-key",
        "database_url": db.url,
    }
    values.update(overrides)
    return FutureSelfBot(
        Settings(**values),
        db,
        fake_ai,
        transcription,
    )


def make_update(
    message: DemoMessage | None,
    *,
    telegram_id: int,
    chat_id: int,
    update_id: int,
    query: DemoQuery | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        update_id=update_id,
        effective_user=SimpleNamespace(id=telegram_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
        effective_message=message,
        message=message,
        callback_query=query,
    )


async def run_gate(
    bot: FutureSelfBot,
    update: SimpleNamespace,
    telegram: DemoTelegramBot,
    application: DemoApplication,
) -> None:
    user_data: dict[str, Any] = {}
    context = SimpleNamespace(
        bot=telegram,
        application=application,
        user_data=user_data,
        args=[],
    )
    with pytest.raises(ApplicationHandlerStop):
        await bot.access_gate(update, context)
    application.user_data_snapshots.append(dict(user_data))


async def start_demo(
    bot: FutureSelfBot,
    telegram: DemoTelegramBot,
    application: DemoApplication,
    *,
    telegram_id: int,
    chat_id: int,
    route: str,
    message_id: int = 100,
    update_id: int = 1,
) -> tuple[DemoMessage, DemoQuery]:
    message = DemoMessage(message_id=message_id)
    query = DemoQuery(route, message)
    await run_gate(
        bot,
        make_update(
            message,
            telegram_id=telegram_id,
            chat_id=chat_id,
            update_id=update_id,
            query=query,
        ),
        telegram,
        application,
    )
    return message, query


async def submit_text(
    bot: FutureSelfBot,
    telegram: DemoTelegramBot,
    application: DemoApplication,
    *,
    telegram_id: int,
    chat_id: int,
    text: str,
    message_id: int,
    update_id: int,
) -> DemoMessage:
    message = DemoMessage(text, message_id=message_id)
    await run_gate(
        bot,
        make_update(
            message,
            telegram_id=telegram_id,
            chat_id=chat_id,
            update_id=update_id,
        ),
        telegram,
        application,
    )
    return message


async def stored_session(bot: FutureSelfBot, telegram_id: int, chat_id: int) -> GuestDemoSession:
    user = await bot._user(telegram_id)
    async with bot.db.sessions() as session:
        row = await session.scalar(
            select(GuestDemoSession).where(
                GuestDemoSession.user_id == user.id,
                GuestDemoSession.chat_id == chat_id,
            )
        )
        assert row is not None
        return row


def real_update(
    application,
    *,
    telegram_id: int,
    chat_id: int,
    update_id: int,
    text: str,
    chat_type: str = "private",
) -> Update:
    telegram_user = TelegramUser(telegram_id, False, "Guest")
    chat = Chat(chat_id, chat_type)
    message = Message(
        update_id,
        datetime.now(UTC),
        chat,
        from_user=telegram_user,
        text=text,
    )
    update = Update(update_id, message=message)
    update.set_bot(application.bot)
    message.set_bot(application.bot)
    return update


@pytest.mark.parametrize(
    ("route", "expected_kind", "prompt_text", "counter_name", "result_heading"),
    [
        (
            "guest:demo:thought",
            GuestDemoKind.THOUGHT_BREAKDOWN,
            GUEST_THOUGHT_INPUT_TEXT,
            "guest_thought_calls",
            "📝 Разобранная мысль",
        ),
        (
            "guest:demo:first-step",
            GuestDemoKind.FIRST_STEP,
            GUEST_FIRST_STEP_INPUT_TEXT,
            "guest_first_step_calls",
            "🌱 Первый шаг",
        ),
    ],
)
async def test_guest_demo_happy_path_uses_one_canonical_message_and_one_ai_call(
    db,
    fake_ai,
    route,
    expected_kind,
    prompt_text,
    counter_name,
    result_heading,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    canonical, query = await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8101,
        chat_id=8101,
        route=route,
        message_id=501,
    )
    assert query.answers == [(None, False)]
    assert query.edits[0]["text"] == prompt_text
    session = await stored_session(bot, 8101, 8101)
    assert session.prompt_message_id == 501
    assert session.demo_kind == expected_kind.value

    input_message = await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8101,
        chat_id=8101,
        text="  Мой спокойный запрос  ",
        message_id=502,
        update_id=2,
    )
    assert len(application.tasks) == 1
    await application.drain()

    assert getattr(fake_ai, counter_name) == 1
    other_counter = (
        fake_ai.guest_first_step_calls
        if counter_name == "guest_thought_calls"
        else fake_ai.guest_thought_calls
    )
    assert other_counter == 0
    assert all(edit["chat_id"] == 8101 for edit in telegram.edits)
    assert all(edit["message_id"] == 501 for edit in telegram.edits)
    assert telegram.edits[0]["reply_markup"] is None
    assert result_heading in telegram.edits[-1]["text"]
    assert canonical.replies == []
    assert input_message.replies == []
    assert telegram.send_calls == []
    assert (await stored_session(bot, 8101, 8101)).status == GuestSessionStatus.COMPLETED.value


async def test_access_gate_returns_while_provider_is_blocked_and_next_update_is_consumed(
    db,
    fake_ai,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    fake_ai.guest_thought_release.clear()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8102,
        chat_id=8102,
        route="guest:demo:thought",
    )

    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8102,
        chat_id=8102,
        text="Запрос для блокировки",
        message_id=102,
        update_id=2,
    )
    assert len(application.tasks) == 1
    await fake_ai.guest_thought_started.wait()
    assert not application.tasks[0].done()

    processing_message = DemoMessage(message_id=100)
    processing_query = DemoQuery("guest:demos", processing_message)
    await run_gate(
        bot,
        make_update(
            processing_message,
            telegram_id=8102,
            chat_id=8102,
            update_id=3,
            query=processing_query,
        ),
        telegram,
        application,
    )
    assert processing_query.answers == [(GUEST_PROCESSING_ALERT, True)]
    assert processing_query.edits == []
    assert len(application.tasks) == 1

    fake_ai.guest_thought_release.set()
    await application.drain()
    assert fake_ai.guest_thought_calls == 1


@pytest.mark.parametrize("invalid_text", ["   \n\t", "x" * 1201])
async def test_invalid_guest_text_does_not_claim_reserve_or_call_provider(
    db,
    fake_ai,
    invalid_text,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8103,
        chat_id=8103,
        route="guest:demo:first-step",
    )
    before = await stored_session(bot, 8103, 8103)

    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8103,
        chat_id=8103,
        text=invalid_text,
        message_id=103,
        update_id=2,
    )
    after = await stored_session(bot, 8103, 8103)
    assert after.status == GuestSessionStatus.AWAITING_INPUT.value
    assert after.version == before.version
    assert after.usage_id is None
    assert application.tasks == []
    assert fake_ai.guest_first_step_calls == 0
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0


async def test_demo_answer_failure_keeps_visible_awaiting_session(db, fake_ai, caplog):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    private_detail = "PRIVATE_CALLBACK_ANSWER_DETAIL"
    message = DemoMessage(message_id=510)
    query = DemoQuery(
        "guest:demo:thought",
        message,
        answer_error=RuntimeError(private_detail),
    )
    with caplog.at_level("ERROR"):
        await run_gate(
            bot,
            make_update(
                message,
                telegram_id=8130,
                chat_id=8130,
                update_id=30,
                query=query,
            ),
            telegram,
            application,
        )
    stored = await stored_session(bot, 8130, 8130)
    assert query.answer_calls == 1
    assert query.edits[-1]["text"] == GUEST_THOUGHT_INPUT_TEXT
    assert stored.status == GuestSessionStatus.AWAITING_INPUT.value
    assert private_detail not in caplog.text


async def test_demo_input_edit_failure_cancels_invisible_session(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    canonical = DemoMessage(message_id=511)
    query = DemoQuery(
        "guest:demo:first-step",
        canonical,
        edit_error=TelegramError("input screen unavailable"),
    )
    await run_gate(
        bot,
        make_update(
            canonical,
            telegram_id=8131,
            chat_id=8131,
            update_id=31,
            query=query,
        ),
        telegram,
        application,
    )
    assert query.answer_calls == 1
    assert (await stored_session(bot, 8131, 8131)).status == GuestSessionStatus.CANCELLED.value
    arbitrary = await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8131,
        chat_id=8131,
        text="Этот текст не является demo input",
        message_id=512,
        update_id=32,
    )
    assert arbitrary.replies
    assert application.tasks == []
    assert fake_ai.guest_first_step_calls == 0
    assert canonical.replies == []
    assert telegram.send_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0


@pytest.mark.parametrize("cancel_stage", ["answer", "edit"])
async def test_cancelled_demo_start_fenced_cleans_exact_unshown_session(
    db,
    fake_ai,
    cancel_stage,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    canonical = DemoMessage(message_id=516)
    cancellation = asyncio.CancelledError()
    query = DemoQuery(
        "guest:demo:thought",
        canonical,
        answer_error=cancellation if cancel_stage == "answer" else None,
        edit_error=cancellation if cancel_stage == "edit" else None,
    )
    update = make_update(
        canonical,
        telegram_id=8144,
        chat_id=8144,
        update_id=516,
        query=query,
    )
    context = SimpleNamespace(
        bot=telegram,
        application=application,
        user_data={},
        args=[],
    )

    with pytest.raises(asyncio.CancelledError):
        await bot.access_gate(update, context)

    assert query.answer_calls == 1
    assert query.edits == []
    stored = await stored_session(bot, 8144, 8144)
    assert stored.status == GuestSessionStatus.CANCELLED.value
    assert stored.result_payload is None
    assert stored.result_expires_at is None

    raw_text = "PRIVATE_TEXT_AFTER_CANCELLED_DEMO_START"
    arbitrary = await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8144,
        chat_id=8144,
        text=raw_text,
        message_id=517,
        update_id=517,
    )
    assert arbitrary.replies
    assert application.tasks == []
    assert fake_ai.guest_thought_calls == 0
    assert fake_ai.guest_first_step_calls == 0
    assert telegram.send_calls == []
    assert all(raw_text not in (task.get_name() or "") for task in asyncio.all_tasks())
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0


async def test_cancel_during_failed_input_edit_keeps_shielded_cleanup_running(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_tasks: list[asyncio.Task[None]] = []
    cleanup_calls: list[dict[str, Any]] = []
    original_cleanup = bot.guest_session_service.cancel_unshown_awaiting
    original_create_task = asyncio.create_task

    async def blocked_cleanup(**kwargs):
        cleanup_calls.append(kwargs)
        cleanup_started.set()
        await cleanup_release.wait()
        return await original_cleanup(**kwargs)

    def record_create_task(coroutine, *, name=None, **kwargs):
        task = original_create_task(coroutine, name=name, **kwargs)
        if name == "guest-unshown-session-cleanup":
            cleanup_tasks.append(task)
        return task

    monkeypatch.setattr(bot.guest_session_service, "cancel_unshown_awaiting", blocked_cleanup)
    monkeypatch.setattr(access_handlers_module.asyncio, "create_task", record_create_task)
    canonical = DemoMessage(message_id=519)
    query = DemoQuery(
        "guest:demo:thought",
        canonical,
        edit_error=TelegramError("input screen unavailable"),
    )
    context = SimpleNamespace(
        bot=telegram,
        application=application,
        user_data={},
        args=[],
    )
    handler = original_create_task(
        bot.access_gate(
            make_update(
                canonical,
                telegram_id=8148,
                chat_id=8148,
                update_id=519,
                query=query,
            ),
            context,
        )
    )

    await cleanup_started.wait()
    assert len(cleanup_tasks) == 1
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    assert not cleanup_tasks[0].done()

    cleanup_release.set()
    await cleanup_tasks[0]
    assert cleanup_tasks[0].done()
    assert len(cleanup_calls) == 1
    stored = await stored_session(bot, 8148, 8148)
    assert cleanup_calls[0] == {
        "user_id": stored.user_id,
        "chat_id": stored.chat_id,
        "access_version": stored.access_version,
        "session_version": 1,
    }
    assert stored.status == GuestSessionStatus.CANCELLED.value
    assert stored.result_payload is None
    assert stored.result_expires_at is None
    assert query.answer_calls == 1
    assert query.edits == []
    assert canonical.replies == []
    assert telegram.send_calls == []

    reserve_calls = 0

    async def forbidden_reserve(**kwargs):
        nonlocal reserve_calls
        reserve_calls += 1
        raise AssertionError("cancelled invisible session must not reserve")

    monkeypatch.setattr(bot.guest_quota_service, "reserve", forbidden_reserve)
    arbitrary = await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8148,
        chat_id=8148,
        text="TEXT_AFTER_SHIELDED_UNSHOWN_CLEANUP",
        message_id=520,
        update_id=520,
    )
    assert arbitrary.replies
    assert reserve_calls == 0
    assert application.tasks == []
    assert fake_ai.guest_thought_calls == 0
    assert fake_ai.guest_first_step_calls == 0
    assert telegram.send_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0


async def test_repeated_cancellation_does_not_replace_original_callback_cancellation(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    original_cleanup = bot.guest_session_service.cancel_unshown_awaiting

    async def blocked_cleanup(**kwargs):
        cleanup_started.set()
        await cleanup_release.wait()
        try:
            return await original_cleanup(**kwargs)
        finally:
            cleanup_finished.set()

    monkeypatch.setattr(bot.guest_session_service, "cancel_unshown_awaiting", blocked_cleanup)
    original_cancellation = asyncio.CancelledError("original callback cancellation")
    canonical = DemoMessage(message_id=521)
    query = DemoQuery(
        "guest:demo:first-step",
        canonical,
        answer_error=original_cancellation,
    )
    handler = asyncio.create_task(
        bot.access_gate(
            make_update(
                canonical,
                telegram_id=8162,
                chat_id=8162,
                update_id=521,
                query=query,
            ),
            SimpleNamespace(
                bot=telegram,
                application=application,
                user_data={},
                args=[],
            ),
        )
    )

    await cleanup_started.wait()
    handler.cancel()
    with pytest.raises(asyncio.CancelledError) as captured:
        await handler
    assert captured.value is original_cancellation
    assert not cleanup_finished.is_set()

    cleanup_release.set()
    await cleanup_finished.wait()
    stored = await stored_session(bot, 8162, 8162)
    assert stored.status == GuestSessionStatus.CANCELLED.value
    assert stored.result_payload is None
    assert stored.result_expires_at is None
    assert query.answer_calls == 1
    assert query.edits == []
    assert fake_ai.guest_first_step_calls == 0


async def test_cancelled_demo_start_cleanup_failure_preserves_original_cancellation(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    private_detail = "PRIVATE_UNSHOWN_CLEANUP_DETAIL"
    task_names: list[str | None] = []
    original_create_task = asyncio.create_task

    async def fail_cleanup(**kwargs):
        raise RuntimeError(private_detail)

    def record_create_task(coroutine, *, name=None, **kwargs):
        task_names.append(name)
        return original_create_task(coroutine, name=name, **kwargs)

    monkeypatch.setattr(bot.guest_session_service, "cancel_unshown_awaiting", fail_cleanup)
    monkeypatch.setattr(access_handlers_module.asyncio, "create_task", record_create_task)
    canonical = DemoMessage(message_id=518)
    query = DemoQuery(
        "guest:demo:first-step",
        canonical,
        answer_error=asyncio.CancelledError(),
    )
    context = SimpleNamespace(
        bot=telegram,
        application=application,
        user_data={},
        args=[],
    )
    with caplog.at_level("ERROR"), pytest.raises(asyncio.CancelledError):
        await bot.access_gate(
            make_update(
                canonical,
                telegram_id=8145,
                chat_id=8145,
                update_id=518,
                query=query,
            ),
            context,
        )

    assert task_names.count("guest-unshown-session-cleanup") == 1
    assert private_detail not in caplog.text
    assert "Guest invisible session cleanup failed error_type=RuntimeError" in caplog.text
    assert all(private_detail not in (name or "") for name in task_names)
    assert query.answer_calls == 1
    assert fake_ai.guest_first_step_calls == 0


async def test_real_process_update_guest_domain_failure_cannot_reach_later_groups(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    transcription = ForbiddenTranscription()
    bot = make_bot(db, fake_ai, _transcription=transcription)
    application = bot.build()
    application._initialized = True
    telegram_id = 8132
    user = await bot._user(telegram_id)
    await bot.guest_session_service.start_session(
        user_id=user.id,
        chat_id=telegram_id,
        access_version=user.access_version,
        demo_kind=GuestDemoKind.THOUGHT_BREAKDOWN,
        prompt_message_id=513,
    )
    private_detail = "PRIVATE_CLAIM_FAILURE_DETAIL"
    claim_calls = 0

    async def fail_claim(**kwargs):
        nonlocal claim_calls
        claim_calls += 1
        raise RuntimeError(private_detail)

    send_calls: list[dict[str, Any]] = []

    async def fake_set_commands(self, commands, **kwargs):
        return None

    async def forbidden_send(self, *args, **kwargs):
        del self, args
        send_calls.append(kwargs)
        raise AssertionError("restricted failure must not create a fallback")

    downstream_calls = 0

    async def downstream(_update, _context):
        nonlocal downstream_calls
        downstream_calls += 1

    monkeypatch.setattr(bot.guest_session_service, "claim_input", fail_claim)
    monkeypatch.setattr(ExtBot, "set_my_commands", fake_set_commands)
    monkeypatch.setattr(ExtBot, "send_message", forbidden_send)
    application.add_handler(TypeHandler(Update, downstream), group=100)
    update = real_update(
        application,
        telegram_id=telegram_id,
        chat_id=telegram_id,
        update_id=513,
        text="RAW_PROCESS_UPDATE_SENTINEL",
    )
    with caplog.at_level("ERROR"):
        await application.process_update(update)

    assert claim_calls == 1
    assert downstream_calls == 0
    assert fake_ai.route_calls == []
    assert fake_ai.guest_thought_calls == 0
    assert fake_ai.guest_first_step_calls == 0
    assert transcription.calls == 0
    assert send_calls == []
    assert private_detail not in caplog.text
    assert "RAW_PROCESS_UPDATE_SENTINEL" not in caplog.text


async def test_real_process_update_non_private_delivery_failure_is_fail_closed(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    transcription = ForbiddenTranscription()
    bot = make_bot(db, fake_ai, _transcription=transcription)
    application = bot.build()
    application._initialized = True
    private_detail = "PRIVATE_GROUP_NOTIFICATION_DETAIL"
    downstream_calls = 0

    async def fail_send(self, *args, **kwargs):
        raise TelegramError(private_detail)

    async def downstream(_update, _context):
        nonlocal downstream_calls
        downstream_calls += 1

    monkeypatch.setattr(ExtBot, "send_message", fail_send)
    application.add_handler(TypeHandler(Update, downstream), group=100)
    update = real_update(
        application,
        telegram_id=8133,
        chat_id=-8133,
        update_id=514,
        text="group update",
        chat_type="group",
    )
    with caplog.at_level("ERROR"):
        await application.process_update(update)

    assert downstream_calls == 0
    assert fake_ai.route_calls == []
    assert fake_ai.guest_thought_calls == 0
    assert fake_ai.guest_first_step_calls == 0
    assert transcription.calls == 0
    assert private_detail not in caplog.text


async def test_error_handler_stops_later_groups_even_if_safe_reply_fails(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    bot = make_bot(db, fake_ai)
    application = bot.build()
    application._initialized = True
    handler_detail = "PRIVATE_ERROR_HANDLER_DETAIL"
    delivery_detail = "PRIVATE_SAFE_REPLY_DETAIL"
    downstream_calls = 0

    async def fail_early(_update, _context):
        raise RuntimeError(handler_detail)

    async def fail_send(self, *args, **kwargs):
        raise TelegramError(delivery_detail)

    async def downstream(_update, _context):
        nonlocal downstream_calls
        downstream_calls += 1

    monkeypatch.setattr(ExtBot, "send_message", fail_send)
    application.add_handler(TypeHandler(Update, fail_early), group=-10)
    application.add_handler(TypeHandler(Update, downstream), group=100)
    update = real_update(
        application,
        telegram_id=8134,
        chat_id=8134,
        update_id=515,
        text="error handler update",
    )
    with caplog.at_level("ERROR"):
        await application.process_update(update)

    assert downstream_calls == 0
    assert handler_detail not in caplog.text
    assert delivery_detail not in caplog.text


async def test_perimeter_cancellation_is_never_suppressed(db, fake_ai, monkeypatch):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8143,
        chat_id=8143,
        route="guest:demo:thought",
    )

    async def cancelled_claim(**kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(bot.guest_session_service, "claim_input", cancelled_claim)
    message = DemoMessage("cancel perimeter", message_id=544)
    with pytest.raises(asyncio.CancelledError):
        await bot.access_gate(
            make_update(message, telegram_id=8143, chat_id=8143, update_id=544),
            SimpleNamespace(
                bot=telegram,
                application=application,
                user_data={},
                args=[],
            ),
        )
    with pytest.raises(asyncio.CancelledError):
        await bot.error_handler(None, SimpleNamespace(error=asyncio.CancelledError()))

    real_application = bot.build()
    group_update = real_update(
        real_application,
        telegram_id=8143,
        chat_id=-8143,
        update_id=545,
        text="cancel private guard",
        chat_type="group",
    )

    async def cancelled_send(self, *args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(ExtBot, "send_message", cancelled_send)
    with pytest.raises(asyncio.CancelledError):
        await bot.private_chat_guard(group_update, SimpleNamespace())


async def test_reserve_not_guest_uses_neutral_terminal_ui(db, fake_ai, monkeypatch):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8135,
        chat_id=8135,
        route="guest:demo:thought",
    )
    original_reserve = bot.guest_quota_service.reserve

    async def change_access_then_reserve(**kwargs):
        await AccessService(db).grant_subscriber(8135, source="test")
        return await original_reserve(**kwargs)

    monkeypatch.setattr(bot.guest_quota_service, "reserve", change_access_then_reserve)
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8135,
        chat_id=8135,
        text="Доступ изменится перед reserve",
        message_id=535,
        update_id=535,
    )
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert (await stored_session(bot, 8135, 8135)).status == GuestSessionStatus.CANCELLED.value
    assert application.tasks == []
    assert fake_ai.guest_thought_calls == 0


async def test_bind_not_guest_fails_reservation_and_uses_neutral_ui(db, fake_ai, monkeypatch):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8136,
        chat_id=8136,
        route="guest:demo:thought",
    )
    original_bind = bot.guest_session_service.bind_reservation

    async def change_access_then_bind(**kwargs):
        await AccessService(db).grant_subscriber(8136, source="test")
        return await original_bind(**kwargs)

    monkeypatch.setattr(bot.guest_session_service, "bind_reservation", change_access_then_bind)
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8136,
        chat_id=8136,
        text="Доступ изменится перед bind",
        message_id=536,
        update_id=536,
    )
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert (await stored_session(bot, 8136, 8136)).status == GuestSessionStatus.CANCELLED.value
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        assert usage is not None and usage.status == "failed"
        assert usage.provider_started_at is None
    assert application.tasks == []
    assert fake_ai.guest_thought_calls == 0


async def test_pending_not_guest_uses_neutral_ui_and_consumes_update(db, fake_ai, monkeypatch):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8137,
        chat_id=8137,
        route="guest:demo:first-step",
        message_id=537,
    )
    stale_user = await bot._user(8137)
    await AccessService(db).grant_subscriber(8137, source="test")

    async def stale_guest(_telegram_id):
        return stale_user

    monkeypatch.setattr(bot, "_user", stale_guest)
    message = DemoMessage("Не запускать AI", message_id=538)
    await run_gate(
        bot,
        make_update(message, telegram_id=8137, chat_id=8137, update_id=538),
        telegram,
        application,
    )
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert message.replies == []
    assert application.tasks == []
    assert fake_ai.guest_first_step_calls == 0


@pytest.mark.parametrize("access_race", ["subscriber", "blocked", "guest_generation"])
async def test_claim_access_race_replaces_stale_input_without_reserve_or_provider(
    db,
    fake_ai,
    monkeypatch,
    access_race,
):
    telegram_id = {
        "subscriber": 8146,
        "blocked": 8147,
        "guest_generation": 8148,
    }[access_race]
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    canonical, _query = await start_demo(
        bot,
        telegram,
        application,
        telegram_id=telegram_id,
        chat_id=telegram_id,
        route="guest:demo:thought",
        message_id=telegram_id,
    )
    original_claim = bot.guest_session_service.claim_input
    reserve_calls = 0

    async def change_access_then_claim(**kwargs):
        if access_race == "subscriber":
            await AccessService(db).grant_subscriber(telegram_id, source="test")
        elif access_race == "blocked":
            await AccessService(db).block(telegram_id, source="test")
        else:
            async with db.session() as db_session:
                current = await db_session.scalar(
                    select(User).where(User.telegram_id == telegram_id)
                )
                assert current is not None
                current.access_version += 1
        return await original_claim(**kwargs)

    async def record_reserve(**kwargs):
        nonlocal reserve_calls
        reserve_calls += 1
        raise AssertionError("claim access race must not reserve")

    monkeypatch.setattr(bot.guest_session_service, "claim_input", change_access_then_claim)
    monkeypatch.setattr(bot.guest_quota_service, "reserve", record_reserve)
    submitted = await submit_text(
        bot,
        telegram,
        application,
        telegram_id=telegram_id,
        chat_id=telegram_id,
        text="RAW_CLAIM_ACCESS_RACE_INPUT",
        message_id=telegram_id + 1,
        update_id=telegram_id + 2,
    )

    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert submitted.replies == []
    assert canonical.replies == []
    assert telegram.send_calls == []
    assert reserve_calls == 0
    assert application.tasks == []
    assert fake_ai.guest_thought_calls == 0
    assert fake_ai.guest_first_step_calls == 0
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0


async def test_benign_stale_claim_does_not_interfere_with_same_guest_generation(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8149,
        chat_id=8149,
        route="guest:demo:first-step",
        message_id=8149,
    )
    edit_count = len(telegram.edits)
    original_claim = bot.guest_session_service.claim_input
    reserve_calls = 0

    async def competing_claim(**kwargs):
        claimed = await original_claim(**kwargs)
        assert claimed.outcome is GuestSessionOutcome.CLAIMED
        return GuestSessionDecision(GuestSessionOutcome.STALE, claimed.session, False)

    async def record_reserve(**kwargs):
        nonlocal reserve_calls
        reserve_calls += 1
        raise AssertionError("benign stale claim must not reserve")

    monkeypatch.setattr(bot.guest_session_service, "claim_input", competing_claim)
    monkeypatch.setattr(bot.guest_quota_service, "reserve", record_reserve)
    submitted = await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8149,
        chat_id=8149,
        text="concurrent claim",
        message_id=8150,
        update_id=8150,
    )

    assert len(telegram.edits) == edit_count
    assert submitted.replies == []
    assert telegram.send_calls == []
    assert reserve_calls == 0
    assert application.tasks == []
    assert fake_ai.guest_first_step_calls == 0
    assert (await stored_session(bot, 8149, 8149)).status == GuestSessionStatus.PROCESSING.value
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0


async def test_stale_claim_access_recheck_failure_is_private_and_fail_closed(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8151,
        chat_id=8151,
        route="guest:demo:thought",
        message_id=8151,
    )
    private_detail = "PRIVATE_STALE_ACCESS_RECHECK_DETAIL"
    raw_input = "RAW_STALE_ACCESS_RECHECK_INPUT"
    reserve_calls = 0

    async def stale_claim(**kwargs):
        return GuestSessionDecision(GuestSessionOutcome.STALE, None, False)

    async def fail_status(_telegram_id):
        raise RuntimeError(private_detail)

    async def record_reserve(**kwargs):
        nonlocal reserve_calls
        reserve_calls += 1
        raise AssertionError("failed access recheck must not reserve")

    monkeypatch.setattr(bot.guest_session_service, "claim_input", stale_claim)
    monkeypatch.setattr(bot.access_service, "status", fail_status)
    monkeypatch.setattr(bot.guest_quota_service, "reserve", record_reserve)
    with caplog.at_level("ERROR"):
        await submit_text(
            bot,
            telegram,
            application,
            telegram_id=8151,
            chat_id=8151,
            text=raw_input,
            message_id=8152,
            update_id=8152,
        )

    assert reserve_calls == 0
    assert application.tasks == []
    assert fake_ai.guest_thought_calls == 0
    assert fake_ai.guest_first_step_calls == 0
    assert telegram.send_calls == []
    assert private_detail not in caplog.text
    assert raw_input not in caplog.text


async def test_duplicate_input_creates_one_task_and_one_provider_permission(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    fake_ai.guest_thought_release.clear()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8104,
        chat_id=8104,
        route="guest:demo:thought",
    )
    duplicate = make_update(
        DemoMessage("Один запрос", message_id=104),
        telegram_id=8104,
        chat_id=8104,
        update_id=4,
    )
    await asyncio.gather(
        run_gate(bot, duplicate, telegram, application),
        run_gate(bot, duplicate, telegram, application),
    )
    assert len(application.tasks) == 1
    await fake_ai.guest_thought_started.wait()
    fake_ai.guest_thought_release.set()
    await application.drain()
    assert fake_ai.guest_thought_calls == 1


@pytest.mark.parametrize("create_task_failure", [False, True])
async def test_pre_provider_failure_releases_provisional_slot(
    db,
    fake_ai,
    create_task_failure,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication(fail_create_task=create_task_failure)
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8105,
        chat_id=8105,
        route="guest:demo:thought",
    )
    if not create_task_failure:
        telegram.next_edit_error = BadRequest("canonical message was deleted")
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8105,
        chat_id=8105,
        text="Проверить cleanup",
        message_id=105,
        update_id=5,
    )
    assert fake_ai.guest_thought_calls == 0
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        assert usage is not None
        assert usage.status == "failed"
        assert usage.provider_started_at is None
    snapshot = await bot.guest_quota_service.snapshot((await bot._user(8105)).id)
    assert snapshot.global_used == 0
    assert snapshot.remaining_operations == 2


@pytest.mark.parametrize("begin_mode", ["exception", "expired", "stale", "already"])
async def test_begin_provider_non_invoking_outcomes_are_reconciled(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    begin_mode,
):
    telegram_id = {"exception": 8135, "expired": 8136, "stale": 8137, "already": 8138}[begin_mode]
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=telegram_id,
        chat_id=telegram_id,
        route="guest:demo:thought",
        message_id=telegram_id,
    )
    original_begin = bot.guest_session_service.begin_provider_call
    private_detail = "PRIVATE_BEGIN_PROVIDER_DETAIL"

    async def controlled_begin(**kwargs):
        if begin_mode == "exception":
            raise RuntimeError(private_detail)
        if begin_mode == "expired":
            return await original_begin(**kwargs, now=datetime.now(UTC) + timedelta(minutes=20))
        if begin_mode == "stale":
            kwargs["session_version"] += 1
            return await original_begin(**kwargs)
        first = await original_begin(**kwargs)
        assert first.can_invoke_provider
        return await original_begin(**kwargs)

    monkeypatch.setattr(bot.guest_session_service, "begin_provider_call", controlled_begin)
    with caplog.at_level("ERROR"):
        await submit_text(
            bot,
            telegram,
            application,
            telegram_id=telegram_id,
            chat_id=telegram_id,
            text="Проверка begin outcome",
            message_id=telegram_id + 1,
            update_id=telegram_id + 2,
        )
        await application.drain()
    assert fake_ai.guest_thought_calls == 0
    stored = await stored_session(bot, telegram_id, telegram_id)
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        assert usage is not None
        if begin_mode == "already":
            assert usage.status == "reserved"
            assert usage.provider_started_at is not None
            assert stored.status == GuestSessionStatus.PROCESSING.value
            assert telegram.edits[-1]["text"] != GUEST_PROVIDER_ERROR_TEXT
        else:
            assert usage.status == ("expired" if begin_mode == "expired" else "failed")
            assert usage.provider_started_at is None
            assert stored.status == GuestSessionStatus.AWAITING_INPUT.value
            assert telegram.edits[-1]["text"] == GUEST_PROVIDER_ERROR_TEXT
    assert private_detail not in caplog.text


async def test_cancel_during_begin_provider_call_cleans_up_and_propagates(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    begin_started = asyncio.Event()
    never_release = asyncio.Event()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8139,
        chat_id=8139,
        route="guest:demo:thought",
    )

    async def blocked_begin(**kwargs):
        begin_started.set()
        await never_release.wait()

    monkeypatch.setattr(bot.guest_session_service, "begin_provider_call", blocked_begin)
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8139,
        chat_id=8139,
        text="Отмена begin",
        message_id=8140,
        update_id=8141,
    )
    await begin_started.wait()
    worker = application.tasks[-1]
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker
    assert fake_ai.guest_thought_calls == 0
    assert (await stored_session(bot, 8139, 8139)).status == GuestSessionStatus.AWAITING_INPUT.value
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        assert usage is not None and usage.status == "failed"
        assert usage.provider_started_at is None


async def test_stale_completion_reconciles_processing_screen_without_second_ai(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    original_complete = bot.guest_session_service.complete_with_result
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8142,
        chat_id=8142,
        route="guest:demo:thought",
    )

    async def stale_complete(**kwargs):
        kwargs["session_version"] += 1
        return await original_complete(**kwargs)

    monkeypatch.setattr(bot.guest_session_service, "complete_with_result", stale_complete)
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8142,
        chat_id=8142,
        text="Stale completion",
        message_id=8143,
        update_id=8144,
    )
    await application.drain()
    assert fake_ai.guest_thought_calls == 1
    assert telegram.edits[-1]["text"] == GUEST_PROVIDER_ERROR_TEXT
    assert (await stored_session(bot, 8142, 8142)).status == GuestSessionStatus.AWAITING_INPUT.value
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        assert usage is not None and usage.status == "failed"
        assert usage.provider_started_at is not None


@pytest.mark.parametrize("failure_mode", ["exception", "invalid", "timeout"])
async def test_post_start_failure_keeps_global_slot_and_personal_quota(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    failure_mode,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    provider_sentinel = "PRIVATE_PROVIDER_ERROR_SENTINEL"
    if failure_mode == "exception":
        fake_ai.guest_thought_error = RuntimeError(provider_sentinel)
    elif failure_mode == "invalid":
        fake_ai.guest_thought_result = SimpleNamespace()
    else:
        monkeypatch.setattr(access_handlers_module, "GUEST_TEXT_PROVIDER_TIMEOUT_SECONDS", 0.001)
        fake_ai.guest_thought_release.clear()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8106,
        chat_id=8106,
        route="guest:demo:thought",
    )
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8106,
        chat_id=8106,
        text="Запрос с ошибкой",
        message_id=106,
        update_id=6,
    )
    await application.drain()

    user = await bot._user(8106)
    snapshot = await bot.guest_quota_service.snapshot(user.id)
    assert snapshot.global_used == 1
    assert snapshot.successful_lifetime_count == 0
    assert snapshot.remaining_operations == 2
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        assert usage is not None
        assert usage.status == "failed"
        assert usage.provider_started_at is not None
    assert GUEST_PROVIDER_ERROR_TEXT in {edit["text"] for edit in telegram.edits}
    assert provider_sentinel not in caplog.text


async def test_access_change_after_provider_start_charges_success_but_drops_result(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    fake_ai.guest_thought_release.clear()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8107,
        chat_id=8107,
        route="guest:demo:thought",
    )
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8107,
        chat_id=8107,
        text="Успешный запрос при смене доступа",
        message_id=107,
        update_id=7,
    )
    await fake_ai.guest_thought_started.wait()
    await AccessService(db).grant_subscriber(8107, source="test")
    fake_ai.guest_thought_release.set()
    await application.drain()

    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        stored = await session.scalar(select(GuestDemoSession))
        assert usage is not None and usage.status == "succeeded"
        assert stored is not None and stored.status == GuestSessionStatus.CANCELLED.value
        assert stored.result_payload is None
    assert not any("📝 Разобранная мысль" in edit["text"] for edit in telegram.edits)
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None


async def test_access_change_before_provider_start_prevents_ai_call(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8120,
        chat_id=8120,
        route="guest:demo:thought",
    )
    original_begin = bot.guest_session_service.begin_provider_call

    async def change_access_then_begin(**kwargs):
        await AccessService(db).grant_subscriber(8120, source="test")
        return await original_begin(**kwargs)

    monkeypatch.setattr(
        bot.guest_session_service,
        "begin_provider_call",
        change_access_then_begin,
    )
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8120,
        chat_id=8120,
        text="Доступ меняется до старта",
        message_id=120,
        update_id=20,
    )
    await application.drain()
    assert fake_ai.guest_thought_calls == 0
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        stored = await session.scalar(select(GuestDemoSession))
        assert usage is not None and usage.status == "failed"
        assert usage.provider_started_at is None
        assert stored is not None and stored.status == GuestSessionStatus.CANCELLED.value
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None


@pytest.mark.parametrize("access_race", ["pending", "reserve", "bind"])
async def test_pre_provider_access_change_uses_neutral_terminal_ui(
    db,
    fake_ai,
    monkeypatch,
    access_race,
):
    telegram_id = {"pending": 8145, "reserve": 8146, "bind": 8147}[access_race]
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=telegram_id,
        chat_id=telegram_id,
        route="guest:demo:thought",
    )
    if access_race == "pending":
        original = bot.guest_session_service.pending_result

        async def change_then_call(**kwargs):
            await AccessService(db).grant_subscriber(telegram_id, source="test")
            return await original(**kwargs)

        monkeypatch.setattr(bot.guest_session_service, "pending_result", change_then_call)
    elif access_race == "reserve":
        original = bot.guest_quota_service.reserve

        async def change_then_call(**kwargs):
            await AccessService(db).grant_subscriber(telegram_id, source="test")
            return await original(**kwargs)

        monkeypatch.setattr(bot.guest_quota_service, "reserve", change_then_call)
    else:
        original = bot.guest_session_service.bind_reservation

        async def change_then_call(**kwargs):
            await AccessService(db).grant_subscriber(telegram_id, source="test")
            return await original(**kwargs)

        monkeypatch.setattr(bot.guest_session_service, "bind_reservation", change_then_call)

    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=telegram_id,
        chat_id=telegram_id,
        text="Смена доступа до provider",
        message_id=telegram_id + 1,
        update_id=telegram_id + 2,
    )
    await application.drain()
    assert fake_ai.guest_thought_calls == 0
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert GUEST_PROVIDER_ERROR_TEXT not in {edit["text"] for edit in telegram.edits}
    assert telegram.send_calls == []
    async with db.sessions() as session:
        usages = (await session.scalars(select(GuestUsageLedger))).all()
        assert all(item.provider_started_at is None for item in usages)
        assert all(item.status == "failed" for item in usages)


async def test_access_change_after_completion_before_edit_clears_result_without_edit(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8121,
        chat_id=8121,
        route="guest:demo:thought",
    )
    original_delivery = bot._deliver_guest_result

    async def change_access_then_deliver(delivery_bot, **kwargs):
        await AccessService(db).grant_subscriber(8121, source="test")
        return await original_delivery(delivery_bot, **kwargs)

    monkeypatch.setattr(bot, "_deliver_guest_result", change_access_then_deliver)
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8121,
        chat_id=8121,
        text="Доступ меняется после completion",
        message_id=121,
        update_id=21,
    )
    await application.drain()
    assert fake_ai.guest_thought_calls == 1
    assert not any("📝 Разобранная мысль" in edit["text"] for edit in telegram.edits)
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        stored = await session.scalar(select(GuestDemoSession))
        assert usage is not None and usage.status == "succeeded"
        assert stored is not None and stored.status == GuestSessionStatus.CANCELLED.value
        assert stored.result_payload is None


async def test_access_change_during_result_edit_triggers_compensating_edit(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8108,
        chat_id=8108,
        route="guest:demo:thought",
    )

    async def change_access(kwargs):
        if "📝 Разобранная мысль" in kwargs["text"]:
            await AccessService(db).grant_subscriber(8108, source="test")
            return True
        return False

    telegram.edit_hook = change_access
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8108,
        chat_id=8108,
        text="Смена во время edit",
        message_id=108,
        update_id=8,
    )
    await application.drain()
    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    stored = await stored_session(bot, 8108, 8108)
    assert stored.status == GuestSessionStatus.CANCELLED.value
    assert stored.result_payload is None


async def test_compensating_access_edit_failure_is_safe_best_effort_redaction(
    db,
    fake_ai,
    caplog,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    private_detail = "PRIVATE_COMPENSATING_EDIT_DETAIL"
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8142,
        chat_id=8142,
        route="guest:demo:thought",
    )

    async def change_access_and_break_redaction(kwargs):
        if "📝 Разобранная мысль" not in kwargs["text"]:
            return False
        await AccessService(db).grant_subscriber(8142, source="test")
        telegram.next_edit_error = TelegramError(private_detail)
        telegram.next_edit_error_text = GUEST_ACCESS_CHANGED_TEXT
        return True

    telegram.edit_hook = change_access_and_break_redaction
    with caplog.at_level("ERROR"):
        await submit_text(
            bot,
            telegram,
            application,
            telegram_id=8142,
            chat_id=8142,
            text="Redaction будет best effort",
            message_id=543,
            update_id=543,
        )
        await application.drain()
    stored = await stored_session(bot, 8142, 8142)
    assert stored.status == GuestSessionStatus.CANCELLED.value
    assert stored.result_payload is None
    assert fake_ai.guest_thought_calls == 1
    assert telegram.send_calls == []
    assert private_detail not in caplog.text
    assert "Guest canonical edit failed error_type=TelegramError user_id=8142" in caplog.text


async def test_result_edit_crossing_ttl_is_compensated_without_refunding_success(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8153,
        chat_id=8153,
        route="guest:demo:thought",
        message_id=8153,
    )
    original_mark_delivered = bot.guest_session_service.mark_delivered

    async def expire_after_successful_edit(**kwargs):
        async with db.sessions() as session:
            row = await session.scalar(
                select(GuestDemoSession).where(GuestDemoSession.chat_id == 8153)
            )
            assert row is not None and row.result_expires_at is not None
            expires_at = row.result_expires_at
        return await original_mark_delivered(**kwargs, now=expires_at)

    monkeypatch.setattr(
        bot.guest_session_service,
        "mark_delivered",
        expire_after_successful_edit,
    )
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8153,
        chat_id=8153,
        text="Результат пересечёт TTL",
        message_id=8154,
        update_id=8154,
    )
    await application.drain()

    assert "Разобранная мысль" in telegram.edits[-2]["text"]
    assert telegram.edits[-1]["text"] == GUEST_RESULT_EXPIRED_TEXT
    assert "Разобранная мысль" not in telegram.edits[-1]["text"]
    callback_data = {
        button.callback_data
        for row in telegram.edits[-1]["reply_markup"].inline_keyboard
        for button in row
    }
    assert callback_data == {"guest:demos", "guest:root"}
    stored = await stored_session(bot, 8153, 8153)
    assert stored.status == GuestSessionStatus.EXPIRED.value
    assert stored.result_payload is None
    assert stored.result_expires_at is None
    user = await bot._user(8153)
    quota = await bot.guest_quota_service.snapshot(user.id)
    assert quota.successful_lifetime_count == 1
    assert quota.remaining_operations == 1
    assert fake_ai.guest_thought_calls == 1
    assert telegram.send_calls == []
    async with db.sessions() as session:
        succeeded = await session.scalar(
            select(func.count(GuestUsageLedger.id)).where(
                GuestUsageLedger.user_id == user.id,
                GuestUsageLedger.status == "succeeded",
            )
        )
        assert succeeded == 1


async def test_concurrent_result_cleanup_stale_ack_is_compensated(db, fake_ai, monkeypatch):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8155,
        chat_id=8155,
        route="guest:demo:first-step",
        message_id=8155,
    )
    original_mark_delivered = bot.guest_session_service.mark_delivered
    outcomes: list[GuestSessionOutcome] = []

    async def cleanup_before_ack(**kwargs):
        async with db.sessions() as session:
            row = await session.scalar(
                select(GuestDemoSession).where(GuestDemoSession.chat_id == 8155)
            )
            assert row is not None and row.result_expires_at is not None
            expires_at = row.result_expires_at
        assert await bot.guest_session_service.cleanup_undeliverable_results(now=expires_at) == 1
        decision = await original_mark_delivered(**kwargs)
        outcomes.append(decision.outcome)
        return decision

    monkeypatch.setattr(bot.guest_session_service, "mark_delivered", cleanup_before_ack)
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8155,
        chat_id=8155,
        text="Cleanup выиграет ack race",
        message_id=8156,
        update_id=8156,
    )
    await application.drain()

    assert outcomes == [GuestSessionOutcome.STALE]
    assert "Первый шаг" in telegram.edits[-2]["text"]
    assert telegram.edits[-1]["text"] == GUEST_RESULT_EXPIRED_TEXT
    stored = await stored_session(bot, 8155, 8155)
    assert stored.status == GuestSessionStatus.EXPIRED.value
    assert stored.result_payload is None
    user = await bot._user(8155)
    quota = await bot.guest_quota_service.snapshot(user.id)
    assert quota.successful_lifetime_count == 1
    assert fake_ai.guest_first_step_calls == 1
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(GuestUsageLedger.id)).where(
                    GuestUsageLedger.user_id == user.id,
                    GuestUsageLedger.status == "succeeded",
                )
            )
            == 1
        )


async def test_access_change_has_priority_over_expiry_compensation(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8157,
        chat_id=8157,
        route="guest:demo:thought",
        message_id=8157,
    )

    async def expire_and_change_access(kwargs):
        if kwargs["reply_markup"] is None:
            return False
        async with db.sessions() as session:
            row = await session.scalar(
                select(GuestDemoSession).where(GuestDemoSession.chat_id == 8157)
            )
            assert row is not None and row.result_expires_at is not None
            expires_at = row.result_expires_at
        assert await bot.guest_session_service.cleanup_undeliverable_results(now=expires_at) == 1
        await AccessService(db).grant_subscriber(8157, source="test")
        return True

    telegram.edit_hook = expire_and_change_access
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8157,
        chat_id=8157,
        text="Access и TTL меняются вместе",
        message_id=8158,
        update_id=8158,
    )
    await application.drain()

    assert telegram.edits[-1]["text"] == GUEST_ACCESS_CHANGED_TEXT
    assert telegram.edits[-1]["reply_markup"] is None
    stored = await stored_session(bot, 8157, 8157)
    assert stored.status == GuestSessionStatus.EXPIRED.value
    assert stored.result_payload is None
    assert stored.result_expires_at is None
    assert fake_ai.guest_thought_calls == 1
    assert telegram.send_calls == []


async def test_expired_result_compensation_failure_is_safe_and_keeps_domain_redacted(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    private_detail = "PRIVATE_EXPIRED_COMPENSATION_DETAIL"
    raw_input = "RAW_EXPIRED_COMPENSATION_INPUT"
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8159,
        chat_id=8159,
        route="guest:demo:first-step",
        message_id=8159,
    )
    original_mark_delivered = bot.guest_session_service.mark_delivered

    async def expire_and_break_compensation(**kwargs):
        async with db.sessions() as session:
            row = await session.scalar(
                select(GuestDemoSession).where(GuestDemoSession.chat_id == 8159)
            )
            assert row is not None and row.result_expires_at is not None
            expires_at = row.result_expires_at
        telegram.next_edit_error = TelegramError(private_detail)
        telegram.next_edit_error_text = GUEST_RESULT_EXPIRED_TEXT
        return await original_mark_delivered(**kwargs, now=expires_at)

    monkeypatch.setattr(
        bot.guest_session_service,
        "mark_delivered",
        expire_and_break_compensation,
    )
    with caplog.at_level("ERROR"):
        await submit_text(
            bot,
            telegram,
            application,
            telegram_id=8159,
            chat_id=8159,
            text=raw_input,
            message_id=8160,
            update_id=8160,
        )
        await application.drain()

    stored = await stored_session(bot, 8159, 8159)
    assert stored.status == GuestSessionStatus.EXPIRED.value
    assert stored.result_payload is None
    assert stored.result_expires_at is None
    assert fake_ai.guest_first_step_calls == 1
    assert telegram.send_calls == []
    assert private_detail not in caplog.text
    assert raw_input not in caplog.text
    assert "Guest canonical edit failed error_type=TelegramError user_id=8159" in caplog.text


async def test_expired_result_message_not_modified_is_success(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    telegram.next_edit_error = BadRequest("Message is not modified")
    telegram.next_edit_error_text = GUEST_RESULT_EXPIRED_TEXT
    assert await bot._edit_guest_result_expired(
        telegram,
        chat_id=8161,
        message_id=8161,
    )
    assert telegram.send_calls == []


async def test_two_successes_show_paywall_and_third_call_is_absent(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    for index, route in enumerate(
        ("guest:demo:thought", "guest:demo:first-step"),
        start=1,
    ):
        await start_demo(
            bot,
            telegram,
            application,
            telegram_id=8109,
            chat_id=8109,
            route=route,
            message_id=109,
            update_id=index * 10,
        )
        await submit_text(
            bot,
            telegram,
            application,
            telegram_id=8109,
            chat_id=8109,
            text=f"Запрос {index}",
            message_id=109 + index,
            update_id=index * 10 + 1,
        )
        await application.drain()
    assert fake_ai.guest_thought_calls == 1
    assert fake_ai.guest_first_step_calls == 1
    final_result = telegram.edits[-1]
    assert "Осталось бесплатных операций: 0" in final_result["text"]
    assert "@Nazar_38rus" in final_result["text"]
    urls = {
        button.url
        for row in final_result["reply_markup"].inline_keyboard
        for button in row
        if button.url
    }
    assert urls == {"https://t.me/Nazar_38rus", "https://naz-ai-lab.ru"}

    _message, third = await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8109,
        chat_id=8109,
        route="guest:demo:thought",
        message_id=109,
        update_id=40,
    )
    assert third.edits[-1]["text"] == GUEST_LIFETIME_EXHAUSTED_TEXT
    assert fake_ai.guest_thought_calls + fake_ai.guest_first_step_calls == 2


@pytest.mark.parametrize(
    ("settings_override", "expected_text"),
    [({"guest_ai_enabled": False}, GUEST_DISABLED_TEXT), ({"guest_global_daily_limit": 1}, None)],
)
async def test_disabled_and_global_exhaustion_do_not_spend_personal_quota(
    db,
    fake_ai,
    settings_override,
    expected_text,
):
    bot = make_bot(db, fake_ai, **settings_override)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    if expected_text is None:
        blocker = await bot._user(8199)
        reserved = await bot.guest_quota_service.reserve(
            user_id=blocker.id,
            demo_kind=GuestDemoKind.THOUGHT_BREAKDOWN,
            idempotency_key="global:blocker",
            telegram_update_id=999,
        )
        assert reserved.is_new
        expected_text = GUEST_GLOBAL_EXHAUSTED_TEXT
    _message, query = await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8110,
        chat_id=8110,
        route="guest:demo:thought",
    )
    assert query.edits[-1]["text"] == expected_text
    snapshot = await bot.guest_quota_service.snapshot((await bot._user(8110)).id)
    assert snapshot.successful_lifetime_count == 0
    assert fake_ai.guest_thought_calls == 0


async def test_awaiting_media_never_reaches_transcription_ai_or_quota(db, fake_ai):
    transcription = ForbiddenTranscription()
    bot = make_bot(db, fake_ai, _transcription=transcription)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8122,
        chat_id=8122,
        route="guest:demo:first-step",
        message_id=122,
    )
    before = await stored_session(bot, 8122, 8122)
    media = DemoMessage(message_id=123, voice=SimpleNamespace(file_id="forbidden"))
    await run_gate(
        bot,
        make_update(media, telegram_id=8122, chat_id=8122, update_id=22),
        telegram,
        application,
    )
    after = await stored_session(bot, 8122, 8122)
    assert transcription.calls == 0
    assert fake_ai.guest_first_step_calls == 0
    assert after.status == GuestSessionStatus.AWAITING_INPUT.value
    assert after.version == before.version
    assert application.tasks == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0


async def test_awaiting_demo_replacement_and_back_keep_one_canonical_message(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    canonical, first_query = await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8123,
        chat_id=8123,
        route="guest:demo:thought",
        message_id=523,
    )
    replacement = DemoQuery("guest:demo:first-step", canonical)
    await run_gate(
        bot,
        make_update(
            canonical,
            telegram_id=8123,
            chat_id=8123,
            update_id=23,
            query=replacement,
        ),
        telegram,
        application,
    )
    replaced = await stored_session(bot, 8123, 8123)
    assert first_query.answers == [(None, False)]
    assert replacement.answers == [(None, False)]
    assert replacement.edits[-1]["text"] == GUEST_FIRST_STEP_INPUT_TEXT
    assert replaced.demo_kind == GuestDemoKind.FIRST_STEP.value
    assert replaced.prompt_message_id == 523

    back = DemoQuery("guest:demos", canonical)
    await run_gate(
        bot,
        make_update(
            canonical,
            telegram_id=8123,
            chat_id=8123,
            update_id=24,
            query=back,
        ),
        telegram,
        application,
    )
    assert back.answers == [(None, False)]
    assert back.edits == []
    assert telegram.edits[-1]["message_id"] == 523
    assert (await stored_session(bot, 8123, 8123)).status == GuestSessionStatus.CANCELLED.value
    assert canonical.replies == []
    assert telegram.send_calls == []


async def test_transient_result_edit_is_recovered_without_second_ai_call(
    db,
    fake_ai,
    monkeypatch,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8111,
        chat_id=8111,
        route="guest:demo:thought",
        message_id=111,
    )
    telegram.next_edit_error = TelegramError("deleted or transient private detail")
    telegram.next_edit_error_text = "📝 Разобранная мысль"
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8111,
        chat_id=8111,
        text="Сохранить результат для recovery",
        message_id=112,
        update_id=2,
    )
    await application.drain()
    assert (await stored_session(bot, 8111, 8111)).status == GuestSessionStatus.RESULT_READY.value
    assert fake_ai.guest_thought_calls == 1
    assert telegram.send_calls == []

    recovery_done = asyncio.Event()
    original_recovery = bot._recover_guest_demo_results

    async def observed_recovery(telegram_bot):
        await original_recovery(telegram_bot)
        recovery_done.set()

    monkeypatch.setattr(bot, "_recover_guest_demo_results", observed_recovery)
    await bot._post_init(
        SimpleNamespace(
            bot=telegram,
            job_queue=None,
            create_task=application.create_task,
            post_stop=bot._post_stop,
        )
    )
    await recovery_done.wait()
    assert (await stored_session(bot, 8111, 8111)).status == GuestSessionStatus.COMPLETED.value
    assert fake_ai.guest_thought_calls == 1
    assert len(application.task_names) == 1
    assert application.task_names[0].startswith("guest-demo:")
    assert "guest-result-recovery" not in application.task_names
    await bot._post_stop(SimpleNamespace())


async def test_concurrent_recovery_is_idempotent_and_never_repeats_ai(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8124,
        chat_id=8124,
        route="guest:demo:thought",
        message_id=124,
    )
    telegram.next_edit_error = TelegramError("initial result delivery failed")
    telegram.next_edit_error_text = "📝 Разобранная мысль"
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8124,
        chat_id=8124,
        text="Результат для конкурентного recovery",
        message_id=125,
        update_id=25,
    )
    await application.drain()
    assert (await stored_session(bot, 8124, 8124)).status == GuestSessionStatus.RESULT_READY.value

    await asyncio.gather(
        bot._recover_guest_demo_results(telegram),
        bot._recover_guest_demo_results(telegram),
    )
    assert (await stored_session(bot, 8124, 8124)).status == GuestSessionStatus.COMPLETED.value
    assert fake_ai.guest_thought_calls == 1
    async with db.sessions() as session:
        user = await session.scalar(select(User).where(User.telegram_id == 8124))
        assert user is not None
        succeeded = await session.scalar(
            select(func.count(GuestUsageLedger.id)).where(
                GuestUsageLedger.user_id == user.id,
                GuestUsageLedger.status == "succeeded",
            )
        )
        assert succeeded == 1


async def test_recovery_isolates_one_candidate_telegram_failure(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    for telegram_id in (8125, 8126):
        await start_demo(
            bot,
            telegram,
            application,
            telegram_id=telegram_id,
            chat_id=telegram_id,
            route="guest:demo:thought",
            message_id=telegram_id,
            update_id=telegram_id,
        )
        telegram.next_edit_error = TelegramError("hold result for startup recovery")
        telegram.next_edit_error_text = "📝 Разобранная мысль"
        await submit_text(
            bot,
            telegram,
            application,
            telegram_id=telegram_id,
            chat_id=telegram_id,
            text="Сохранённый результат",
            message_id=telegram_id + 10,
            update_id=telegram_id + 1,
        )
        await application.drain()
        assert (
            await stored_session(bot, telegram_id, telegram_id)
        ).status == GuestSessionStatus.RESULT_READY.value

    telegram.next_edit_error = TelegramError("first recovery candidate unavailable")
    telegram.next_edit_error_text = "📝 Разобранная мысль"
    await bot._recover_guest_demo_results(telegram)
    assert (await stored_session(bot, 8125, 8125)).status == GuestSessionStatus.RESULT_READY.value
    assert (await stored_session(bot, 8126, 8126)).status == GuestSessionStatus.COMPLETED.value
    assert fake_ai.guest_thought_calls == 2


async def test_message_not_modified_completes_real_post_edit_crash_window(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    private_detail = "PRIVATE_DELIVERY_ACK_DETAIL"
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8112,
        chat_id=8112,
        route="guest:demo:first-step",
        message_id=112,
    )

    async def fail_acknowledgement(**kwargs):
        raise RuntimeError(private_detail)

    monkeypatch.setattr(
        bot.guest_session_service,
        "mark_delivered",
        fail_acknowledgement,
    )
    with caplog.at_level("ERROR"):
        await submit_text(
            bot,
            telegram,
            application,
            telegram_id=8112,
            chat_id=8112,
            text="Результат уже был отредактирован",
            message_id=113,
            update_id=2,
        )
        await application.drain()
    assert any("🌱 Первый шаг" in edit["text"] for edit in telegram.edits)
    assert (await stored_session(bot, 8112, 8112)).status == GuestSessionStatus.RESULT_READY.value
    assert private_detail not in caplog.text

    restarted = make_bot(db, fake_ai)
    telegram.next_edit_error = BadRequest("Message is not modified")
    telegram.next_edit_error_text = "🌱 Первый шаг"
    await restarted._recover_guest_demo_results(telegram)
    assert (
        await stored_session(restarted, 8112, 8112)
    ).status == GuestSessionStatus.COMPLETED.value
    assert fake_ai.guest_first_step_calls == 1
    async with db.sessions() as session:
        user = await session.scalar(select(User).where(User.telegram_id == 8112))
        assert user is not None
        succeeded = await session.scalar(
            select(func.count(GuestUsageLedger.id)).where(
                GuestUsageLedger.user_id == user.id,
                GuestUsageLedger.status == "succeeded",
            )
        )
        assert succeeded == 1


async def test_pending_result_consumes_new_updates_and_never_creates_fallback(db, fake_ai):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8113,
        chat_id=8113,
        route="guest:demo:thought",
        message_id=113,
    )
    telegram.next_edit_error = TelegramError("canonical message unavailable")
    telegram.next_edit_error_text = "📝 Разобранная мысль"
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8113,
        chat_id=8113,
        text="Результат останется pending",
        message_id=114,
        update_id=2,
    )
    await application.drain()
    before = await stored_session(bot, 8113, 8113)
    assert before.status == GuestSessionStatus.RESULT_READY.value

    updates = [
        (
            DemoMessage("/start", message_id=115),
            None,
        ),
        (
            DemoMessage("Новый текст не заменит результат", message_id=116),
            None,
        ),
        (
            DemoMessage(message_id=113),
            DemoQuery("guest:demos", DemoMessage(message_id=113)),
        ),
    ]
    for offset, (message, query) in enumerate(updates, start=3):
        telegram.next_edit_error = TelegramError("still unavailable")
        telegram.next_edit_error_text = "📝 Разобранная мысль"
        await run_gate(
            bot,
            make_update(
                message,
                telegram_id=8113,
                chat_id=8113,
                update_id=offset,
                query=query,
            ),
            telegram,
            application,
        )
        if query is not None:
            assert query.answers == [(None, False)]
    after = await stored_session(bot, 8113, 8113)
    assert after.status == GuestSessionStatus.RESULT_READY.value
    assert after.version == before.version
    assert all(message.replies == [] for message, _query in updates)
    assert telegram.send_calls == []
    assert fake_ai.guest_thought_calls == 1


async def test_raw_input_and_provider_error_never_enter_db_logs_task_name_or_key(
    db,
    fake_ai,
    caplog,
):
    bot = make_bot(db, fake_ai)
    telegram = DemoTelegramBot()
    application = DemoApplication()
    raw_sentinel = "RAW_GUEST_HANDLER_SENTINEL"
    provider_sentinel = "PROVIDER_BODY_SENTINEL"
    fake_ai.guest_thought_error = RuntimeError(provider_sentinel)
    await start_demo(
        bot,
        telegram,
        application,
        telegram_id=8114,
        chat_id=8114,
        route="guest:demo:thought",
    )
    await submit_text(
        bot,
        telegram,
        application,
        telegram_id=8114,
        chat_id=8114,
        text=raw_sentinel,
        message_id=116,
        update_id=2,
    )
    await application.drain()
    assert all(raw_sentinel not in (name or "") for name in application.task_names)
    assert raw_sentinel not in repr(application.user_data_snapshots)
    assert raw_sentinel not in caplog.text
    assert provider_sentinel not in caplog.text
    async with db.sessions() as session:
        usage = await session.scalar(select(GuestUsageLedger))
        stored = await session.scalar(select(GuestDemoSession))
        assert usage is not None and raw_sentinel not in usage.idempotency_key
        assert stored is not None and raw_sentinel not in repr(stored.result_payload)
        assert raw_sentinel not in repr(stored.__dict__)


def test_maximum_result_payloads_are_below_telegram_utf16_limit():
    thought = GuestThoughtBreakdown(
        category="idea",
        title="😀" * 120,
        essence="😀" * 500,
        next_step="😀" * 300,
    )
    first = GuestFirstStep(
        focus="😀" * 300,
        first_step="😀" * 300,
        actions=["😀" * 200, "😀" * 200, "😀" * 200],
    )
    for result, kind in (
        (thought, GuestDemoKind.THOUGHT_BREAKDOWN),
        (first, GuestDemoKind.FIRST_STEP),
    ):
        text = FutureSelfBot._guest_result_text(result, kind, 0)
        assert len(text.encode("utf-16-le")) // 2 < 4096
