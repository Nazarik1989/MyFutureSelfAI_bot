from __future__ import annotations

import asyncio
import gc
import re
import warnings
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
from telegram.error import BadRequest, TelegramError

import future_self.weekly_review_flow as weekly_review_flow_module
from future_self.access import SUBSCRIBER, AccessService
from future_self.models import (
    InboxItem,
    RecurringTaskReminderSchedule,
    TaskReminder,
    User,
    WeeklyFocus,
    WeeklyFocusChange,
    WeeklyReviewSession,
)
from future_self.repositories import UserRepository
from future_self.schemas import (
    WeeklyReviewExtraction,
    WeeklyReviewReminderCandidate,
)
from future_self.weekly_review import (
    WeeklyFocusMutation,
    WeeklyReminderCandidate,
    WeeklyReviewPhase,
    WeeklyReviewService,
    WeeklyReviewSessionResult,
)
from future_self.weekly_review_flow import WeeklyReviewCapabilityStore, WeeklyReviewPolicy
from future_self.weekly_review_handlers import (
    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
    WEEKLY_REVIEW_RECOVERY_TEXT,
    WEEKLY_REVIEW_STALE_TEXT,
    WEEKLY_REVIEW_UNAVAILABLE_TEXT,
    WeeklyReviewHandlers,
    classify_weekly_review_intent,
)


class WeeklyMessage:
    _ids = count(120_000)

    def __init__(
        self,
        text: str | None = None,
        *,
        message_id: int | None = None,
        chat_id: int = 0,
        is_bot: bool = False,
    ) -> None:
        self.text = text
        self.message_id = message_id if message_id is not None else next(self._ids)
        self.chat_id = chat_id
        self.chat = SimpleNamespace(id=chat_id)
        self.from_user = SimpleNamespace(is_bot=is_bot)
        self.voice = None
        self.audio = None
        self.photo = None
        self.document = None
        self.replies: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted = 0

    async def reply_text(self, text: str, **kwargs: Any) -> WeeklyMessage:
        sent = WeeklyMessage(text, chat_id=self.chat_id, is_bot=True)
        self.replies.append({"text": text, "message": sent, **kwargs})
        return sent

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})

    async def delete(self) -> None:
        self.deleted += 1


class WeeklyQuery:
    def __init__(self, data: str, message: WeeklyMessage) -> None:
        self.data = data
        self.message = message
        self.answers: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        self.answers.append({"args": args, **kwargs})

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})


class BlockingWeeklyQuery(WeeklyQuery):
    def __init__(self, data: str, message: WeeklyMessage) -> None:
        super().__init__(data, message)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})
        if len(self.edits) == 1:
            self.started.set()
            await self.release.wait()


class CancellingWeeklyQuery(WeeklyQuery):
    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        del text, kwargs
        raise asyncio.CancelledError


class WeeklyBot:
    def __init__(self, *, edit_error: BaseException | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self.edit_error = edit_error

    async def send_message(self, **kwargs: Any) -> WeeklyMessage:
        sent = WeeklyMessage(kwargs["text"], chat_id=kwargs["chat_id"], is_bot=True)
        self.sent.append({**kwargs, "message": sent})
        return sent

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edits.append(kwargs)
        if self.edit_error is not None:
            raise self.edit_error

    async def delete_message(self, **kwargs: Any) -> bool:
        self.deleted.append(kwargs)
        return True


class BlockingWeeklyBot(WeeklyBot):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edits.append(kwargs)
        if len(self.edits) == 1:
            self.started.set()
            await self.release.wait()


class WeeklyHarness(WeeklyReviewHandlers):
    def __init__(self, db: Any, ai: Any) -> None:
        self.db = db
        self.ai = ai
        self.settings = SimpleNamespace(weekly_review_weekday=6)
        self.access_service = AccessService(db)
        self.weekly_review_service = WeeklyReviewService(db, review_weekday=6)
        self.weekly_review_policy = WeeklyReviewPolicy(enabled=True, admin_only=False)
        self.weekly_review_capabilities = WeeklyReviewCapabilityStore()
        self._weekly_review_launch_lock = asyncio.Lock()
        self._reply_keyboard_owner_lock = asyncio.Lock()
        self._weekly_review_tasks: set[asyncio.Task[Any]] = set()
        self._weekly_review_ui_locks: dict[tuple[int, int], asyncio.Lock] = {}
        self.reply_keyboard_owned = False
        self.reminder_handoffs: list[dict[str, Any]] = []

    async def _active_navigation_flow(self, update: Any, context: Any) -> None:
        del update, context
        return None

    async def _prompt_navigation_flow(self, message: Any, update: Any, flow: Any) -> None:
        raise AssertionError((message, update, flow))

    def _root_keyboard(self, tier: str | None = None) -> Any:
        return SimpleNamespace(tier=tier)

    async def _weekly_review_has_reply_keyboard_owner(self, user: User) -> bool:
        del user
        return self.reply_keyboard_owned

    async def reminder_from_weekly_candidate(
        self, update: Any, context: Any, **kwargs: Any
    ) -> bool:
        del update, context
        self.reminder_handoffs.append(kwargs)
        return True

    async def nova_memory_blocks_navigation(self, update: Any) -> bool:
        del update
        return False

    async def nova_memory_public_command_gate(self, update: Any, context: Any) -> bool:
        raise AssertionError((update, context))

    async def reminder_blocks_navigation(self, update: Any) -> bool:
        del update
        return False


async def weekly_user(db: Any, telegram_id: int) -> User:
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(telegram_id, "Europe/Moscow")
        user.access_tier = SUBSCRIBER
        user.access_version = 4
        user.onboarding_completed = True
        return user


def weekly_update(
    message: WeeklyMessage,
    *,
    telegram_user_id: int,
    chat_id: int,
    query: WeeklyQuery | None = None,
) -> SimpleNamespace:
    message.chat_id = chat_id
    message.chat = SimpleNamespace(id=chat_id)
    return SimpleNamespace(
        effective_message=query.message if query is not None else message,
        message=message,
        effective_user=SimpleNamespace(id=telegram_user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        callback_query=query,
    )


def weekly_context(bot: WeeklyBot | None = None) -> SimpleNamespace:
    return SimpleNamespace(bot=bot or WeeklyBot(), user_data={})


async def preview_session(
    bot: WeeklyHarness,
    user: User,
    *,
    chat_id: int,
    canonical_message_id: int,
    focus: str = "Спокойно закрывать подтверждённые задачи",
    candidates: tuple[WeeklyReminderCandidate, ...] = (),
):
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
        canonical_message_id=canonical_message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    processing = await bot.weekly_review_service.mark_processing(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=canonical_message_id,
    )
    assert processing.session is not None
    stored = await bot.weekly_review_service.store_extraction(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
        session_public_id=processing.session.public_id,
        expected_session_version=processing.session.version,
        canonical_message_id=canonical_message_id,
        focus=focus,
        approach="Выбирать один небольшой шаг",
        small_steps=("Проверить список",),
        reminder_candidates=candidates,
        source="text",
    )
    assert stored.session is not None
    return stored.session


async def issue_session_action(
    bot: WeeklyHarness,
    session: Any,
    action: str,
) -> str:
    tokens = await bot.weekly_review_capabilities.issue(
        actions=(action,),
        owner_id=session.owner_id,
        telegram_user_id=session.telegram_user_id,
        chat_id=session.chat_id,
        canonical_message_id=session.canonical_message_id,
        access_version=session.access_version,
        week_start=session.week_start,
        session_public_id=session.public_id,
        session_version=session.version,
    )
    return f"wrev:{tokens[action]}"


async def session_in_phase(
    bot: WeeklyHarness,
    user: User,
    *,
    chat_id: int,
    canonical_message_id: int,
    phase: WeeklyReviewPhase,
) -> Any:
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
        canonical_message_id=canonical_message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    assert created.session is not None
    if phase is WeeklyReviewPhase.ROOT:
        return created.session
    values: dict[str, Any] = {}
    if phase in {
        WeeklyReviewPhase.PREVIEW,
        WeeklyReviewPhase.SAVED,
        WeeklyReviewPhase.CANDIDATES,
    }:
        values = {
            "focus": "PRIVATE_ACTION_MATRIX_FOCUS",
            "approach": "PRIVATE_ACTION_MATRIX_APPROACH",
            "small_steps": ("PRIVATE_ACTION_MATRIX_STEP",),
            "reminder_candidates": (
                WeeklyReminderCandidate(
                    "PRIVATE_ACTION_MATRIX_CANDIDATE",
                    "PRIVATE_ACTION_MATRIX_SCHEDULE",
                ),
            ),
            "source": "text",
        }
    transitioned = await bot.weekly_review_service.transition_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        expected_canonical_message_id=canonical_message_id,
        expected_phase=WeeklyReviewPhase.ROOT,
        phase=phase,
        **values,
    )
    assert transitioned.session is not None
    return transitioned.session


async def assert_live_session_controls(
    bot: WeeklyHarness,
    query: WeeklyQuery,
    session: Any,
) -> None:
    assert query.edits
    markup = query.edits[-1]["reply_markup"]
    callbacks = [
        str(button.callback_data)
        for row in markup.inline_keyboard
        for button in row
        if str(button.callback_data).startswith("wrev:")
    ]
    assert callbacks
    for callback in callbacks:
        capability = await bot.weekly_review_capabilities.peek(
            callback.removeprefix("wrev:"),
            telegram_user_id=session.telegram_user_id,
            chat_id=session.chat_id,
            canonical_message_id=session.canonical_message_id,
        )
        assert capability is not None
        assert capability.session_public_id == session.public_id
        assert capability.session_version == session.version


@pytest.mark.asyncio
@pytest.mark.parametrize("answer_outcome", ("success", "bad_request", "telegram_error"))
async def test_callback_answer_once_failure_does_not_stop_confirm_or_recovery(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    answer_outcome,
):
    user = await weekly_user(db, 718_900)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 918_900
    canonical = WeeklyMessage(message_id=128_900, chat_id=chat_id, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        focus="PRIVATE_ANSWER_ONCE_FOCUS",
    )
    callback_data = await issue_session_action(bot, preview, "save")
    query = WeeklyQuery(callback_data, canonical)
    attempts = 0

    async def answer(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        query.answers.append({"args": args, **kwargs})
        if answer_outcome == "bad_request":
            raise BadRequest("PRIVATE_ANSWER_BAD_REQUEST")
        if answer_outcome == "telegram_error":
            raise TelegramError("PRIVATE_ANSWER_TELEGRAM_ERROR")

    monkeypatch.setattr(query, "answer", answer)
    context = weekly_context()
    with caplog.at_level("WARNING"):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            context,
        )
    await asyncio.sleep(0)

    assert attempts == 1
    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert "PRIVATE_ANSWER_ONCE_FOCUS" not in query.edits[-1]["text"]
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.SAVED
    await assert_live_session_controls(bot, query, current.session)
    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot.reminder_handoffs == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 1
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
    if answer_outcome == "success":
        assert "operation=callback_answer" not in caplog.text
    else:
        expected_type = "BadRequest" if answer_outcome == "bad_request" else "TelegramError"
        assert "operation=callback_answer" in caplog.text
        assert f"error_type={expected_type}" in caplog.text
    assert "PRIVATE_ANSWER_BAD_REQUEST" not in caplog.text
    assert "PRIVATE_ANSWER_TELEGRAM_ERROR" not in caplog.text
    assert "PRIVATE_ANSWER_ONCE_FOCUS" not in caplog.text
    assert callback_data not in caplog.text
    assert str(user.telegram_id) not in caplog.text


@pytest.mark.asyncio
async def test_callback_answer_direct_cancellation_leaves_token_usable(db, fake_ai, monkeypatch):
    user = await weekly_user(db, 718_901)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 918_901
    canonical = WeeklyMessage(message_id=128_901, chat_id=chat_id, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
    )
    callback_data = await issue_session_action(bot, preview, "save")
    token = callback_data.removeprefix("wrev:")
    query = WeeklyQuery(callback_data, canonical)
    attempts = 0

    async def cancel(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        del args, kwargs
        attempts += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(query, "answer", cancel)
    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            weekly_context(),
        )
    await asyncio.sleep(0)

    assert attempts == 1
    assert query.edits == []
    assert (
        await bot.weekly_review_capabilities.peek(
            token,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            canonical_message_id=canonical.message_id,
        )
        is not None
    )
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.asyncio
async def test_callback_answer_outer_cancellation_keeps_shielded_confirm_alive(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 718_902)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 918_902
    canonical = WeeklyMessage(message_id=128_902, chat_id=chat_id, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
    )
    query = WeeklyQuery(await issue_session_action(bot, preview, "save"), canonical)
    started = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    async def block(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        query.answers.append({"args": args, **kwargs})
        started.set()
        await release.wait()

    monkeypatch.setattr(query, "answer", block)
    outer = asyncio.create_task(
        bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            weekly_context(),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert bot._weekly_review_tasks
    release.set()
    await asyncio.wait_for(asyncio.gather(*tuple(bot._weekly_review_tasks)), timeout=5)
    await asyncio.sleep(0)

    assert attempts == 1
    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert query.edits[-1]["reply_markup"] is not None
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 1


@pytest.mark.parametrize(
    "expiry_offset",
    (
        pytest.param(timedelta(0), id="at-expiry"),
        pytest.param(timedelta(microseconds=1), id="after-expiry"),
    ),
)
@pytest.mark.asyncio
async def test_preview_save_expiring_during_callback_answer_fails_closed_and_recovers(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    expiry_offset,
):
    private = "PRIVATE_PREVIEW_TTL_RACE"
    user = await weekly_user(db, 718_905 + int(bool(expiry_offset)))
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 918_905 + int(bool(expiry_offset))
    canonical = WeeklyMessage(
        message_id=128_905 + int(bool(expiry_offset)),
        chat_id=chat_id,
        is_bot=True,
    )
    preview = await preview_session(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        focus=private,
    )
    callback_data = await issue_session_action(bot, preview, "save")
    token = callback_data.removeprefix("wrev:")
    claim = await bot.weekly_review_capabilities.peek(
        token,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
    )
    assert claim is not None

    clock = {"now": claim.expires_at - timedelta(microseconds=1)}
    original_utc = bot.weekly_review_capabilities._utc

    def controlled_utc(value: datetime | None) -> datetime:
        return clock["now"] if value is None else original_utc(value)

    monkeypatch.setattr(bot.weekly_review_capabilities, "_utc", controlled_utc)
    consume_results: list[bool] = []
    original_consume = bot.weekly_review_capabilities.consume

    async def tracked_consume(expected: Any, *, now: datetime | None = None) -> bool:
        result = await original_consume(expected, now=now)
        consume_results.append(result)
        return result

    confirm_calls = 0

    async def forbidden_confirm(**kwargs: Any):
        nonlocal confirm_calls
        del kwargs
        confirm_calls += 1
        raise AssertionError("expired capability must not confirm focus")

    monkeypatch.setattr(bot.weekly_review_capabilities, "consume", tracked_consume)
    monkeypatch.setattr(bot.weekly_review_service, "confirm_focus", forbidden_confirm)
    query = WeeklyQuery(callback_data, canonical)
    answer_started = asyncio.Event()
    answer_release = asyncio.Event()

    async def blocked_answer(*args: Any, **kwargs: Any) -> None:
        query.answers.append({"args": args, **kwargs})
        answer_started.set()
        await answer_release.wait()

    monkeypatch.setattr(query, "answer", blocked_answer)
    context = weekly_context()
    callback = asyncio.create_task(
        bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            context,
        )
    )
    try:
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(answer_started.wait(), timeout=10)
            assert (
                await bot.weekly_review_capabilities.peek(
                    token,
                    telegram_user_id=user.telegram_id,
                    chat_id=chat_id,
                    canonical_message_id=canonical.message_id,
                )
                == claim
            )
            clock["now"] = claim.expires_at + expiry_offset
            answer_release.set()
            await asyncio.wait_for(callback, timeout=10)
    finally:
        answer_release.set()
        if not callback.done():
            callback.cancel()
        await asyncio.gather(callback, return_exceptions=True)
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert consume_results == [False]
    assert confirm_calls == 0
    assert len(query.edits) == 1
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session == preview
    assert current.session.phase is WeeklyReviewPhase.PREVIEW
    await assert_live_session_controls(bot, query, current.session)
    assert (
        await bot.weekly_review_capabilities.peek(
            token,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            canonical_message_id=canonical.message_id,
        )
        is None
    )

    replay = WeeklyQuery(callback_data, canonical)
    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=replay,
        ),
        context,
    )
    await asyncio.sleep(0)
    assert replay.answers == [{"args": (WEEKLY_REVIEW_STALE_TEXT,), "show_alert": True}]
    assert replay.edits == []
    assert consume_results == [False]
    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot.reminder_handoffs == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyReviewSession.id))) == 1
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
    assert private not in caplog.text
    assert callback_data not in caplog.text
    assert str(user.telegram_id) not in caplog.text


@pytest.mark.asyncio
async def test_preview_save_answer_wait_preserves_new_access_generation(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_PREVIEW_ACCESS_WAIT"
    user = await weekly_user(db, 718_907)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 918_907
    canonical = WeeklyMessage(message_id=128_907, chat_id=chat_id, is_bot=True)
    old = await preview_session(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        focus=private,
    )
    callback_data = await issue_session_action(bot, old, "save")
    token = callback_data.removeprefix("wrev:")
    query = WeeklyQuery(callback_data, canonical)
    answer_started = asyncio.Event()
    answer_release = asyncio.Event()

    async def blocked_answer(*args: Any, **kwargs: Any) -> None:
        query.answers.append({"args": args, **kwargs})
        answer_started.set()
        await answer_release.wait()

    confirm_calls = 0

    async def forbidden_confirm(**kwargs: Any):
        nonlocal confirm_calls
        del kwargs
        confirm_calls += 1
        raise AssertionError("old access generation must not confirm focus")

    monkeypatch.setattr(query, "answer", blocked_answer)
    monkeypatch.setattr(bot.weekly_review_service, "confirm_focus", forbidden_confirm)
    context = weekly_context()
    callback = asyncio.create_task(
        bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            context,
        )
    )
    replacement = None
    fresh = None
    try:
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(answer_started.wait(), timeout=10)
            await bot.access_service.set_guest(user.telegram_id, source="test")
            await bot.access_service.grant_subscriber(user.telegram_id, source="test")
            fresh = await bot.access_service.status(user.telegram_id)
            assert fresh is not None
            replacement = await bot.weekly_review_service.create_session(
                telegram_actor_id=user.telegram_id,
                chat_id=chat_id,
                expected_access_version=fresh.access_version,
                canonical_message_id=canonical.message_id,
                phase=WeeklyReviewPhase.ROOT,
            )
            assert replacement.session is not None
            answer_release.set()
            await asyncio.wait_for(callback, timeout=10)
    finally:
        answer_release.set()
        if not callback.done():
            callback.cancel()
        await asyncio.gather(callback, return_exceptions=True)
    await asyncio.sleep(0)

    assert fresh is not None
    assert replacement is not None and replacement.session is not None
    assert query.answers == [{"args": ()}]
    assert confirm_calls == 0
    assert query.edits == [
        {
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=fresh.access_version,
    )
    assert current.session == replacement.session
    assert (
        await bot.weekly_review_capabilities.peek(
            token,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            canonical_message_id=canonical.message_id,
        )
        is None
    )
    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot.reminder_handoffs == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
    assert private not in caplog.text
    assert callback_data not in caplog.text
    assert str(user.telegram_id) not in caplog.text


@pytest.mark.asyncio
async def test_real_root_reminders_action_enters_weekly_input_without_reminder_dml(db, fake_ai):
    user = await weekly_user(db, 718_903)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 918_903
    canonical = WeeklyMessage(message_id=128_903, chat_id=chat_id, is_bot=True)
    root = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    markup = await bot._weekly_review_root_markup(root, user)
    assert markup is not None
    callback_data = None
    for row in markup.inline_keyboard:
        for button in row:
            value = str(button.callback_data)
            claim = await bot.weekly_review_capabilities.peek(
                value.removeprefix("wrev:"),
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                canonical_message_id=canonical.message_id,
            )
            if claim is not None and claim.action == "reminders":
                callback_data = value
    assert callback_data is not None
    query = WeeklyQuery(callback_data, canonical)
    context = weekly_context()

    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=query,
        ),
        context,
    )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert "Что важно удерживать" in query.edits[-1]["text"]
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    await assert_live_session_controls(bot, query, current.session)
    assert fake_ai.weekly_review_calls == []
    assert bot.reminder_handoffs == []
    assert canonical.replies == []
    assert context.bot.sent == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_unknown_server_issued_action_recovers_live_root_without_domain_writes(
    db,
    fake_ai,
    caplog,
):
    user = await weekly_user(db, 718_904)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 918_904
    canonical = WeeklyMessage(message_id=128_904, chat_id=chat_id, is_bot=True)
    root = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    callback_data = await issue_session_action(bot, root, "PRIVATE_UNKNOWN_ACTION")
    query = WeeklyQuery(callback_data, canonical)
    context = weekly_context()

    with caplog.at_level("WARNING"):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            context,
        )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    await assert_live_session_controls(bot, query, root)
    assert "operation=unknown_action error_type=UnknownAction" in caplog.text
    assert "PRIVATE_UNKNOWN_ACTION" not in caplog.text
    assert callback_data not in caplog.text
    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Давай скорректируем систему", "start"),
        ("Фокус на неделю!", "edit_focus"),
        ("спланировать неделю", "start"),
        ("Покажи фокус недели", "view"),
        ("На этой неделе система дала сбой", "none"),
        ("Мне важен фокус в работе", "none"),
        ("Расскажу, как прошла неделя", "none"),
    ],
)
def test_weekly_review_intent_is_strict(text: str, expected: str):
    assert classify_weekly_review_intent(text) == expected


@pytest.mark.asyncio
async def test_manual_open_uses_one_canonical_and_removes_reply_keyboard(db, fake_ai):
    user = await weekly_user(db, 7101)
    bot = WeeklyHarness(db, fake_ai)
    incoming = WeeklyMessage("/week", chat_id=9101)
    context = weekly_context()
    update = weekly_update(incoming, telegram_user_id=user.telegram_id, chat_id=9101)

    await bot.week_command(update, context)

    assert len(incoming.replies) == 1
    assert isinstance(incoming.replies[0]["reply_markup"], ReplyKeyboardRemove)
    canonical = incoming.replies[0]["message"]
    assert len(context.bot.edits) == 1
    assert context.bot.edits[0]["message_id"] == canonical.message_id
    assert context.bot.edits[0]["reply_markup"] is not None
    async with db.sessions() as session:
        durable = await session.scalar(select(WeeklyReviewSession))
        assert durable is not None
        assert durable.canonical_message_id == canonical.message_id
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize("control", ["Назад", "Пропустить", "Отменить"])
@pytest.mark.parametrize(
    "policy",
    [
        WeeklyReviewPolicy(enabled=False),
        WeeklyReviewPolicy(enabled=True, admin_only=True),
    ],
    ids=["disabled", "admin-only-subscriber"],
)
@pytest.mark.asyncio
async def test_ineligible_stale_text_controls_are_consumed_without_weekly_storage(
    db,
    fake_ai,
    monkeypatch,
    control,
    policy,
):
    user = await weekly_user(db, 710_101)
    bot = WeeklyHarness(db, fake_ai)
    bot.weekly_review_policy = policy
    incoming = WeeklyMessage(control, chat_id=910_101)
    update = weekly_update(
        incoming,
        telegram_user_id=user.telegram_id,
        chat_id=910_101,
    )

    async def forbidden_weekly_lookup(**kwargs: Any):
        del kwargs
        raise AssertionError("disabled controls must not touch weekly storage")

    monkeypatch.setattr(
        bot.weekly_review_service,
        "current_session",
        forbidden_weekly_lookup,
    )

    assert await bot.weekly_review_active_text_gate(update, weekly_context()) is True
    assert [reply["text"] for reply in incoming.replies] == [WEEKLY_REVIEW_UNAVAILABLE_TEXT]
    assert fake_ai.weekly_review_calls == []


@pytest.mark.parametrize("control", ["Назад", "Пропустить", "Отменить"])
@pytest.mark.asyncio
async def test_stale_voice_controls_retire_progress_and_stay_local(
    db,
    fake_ai,
    control,
):
    user = await weekly_user(db, 710_102)
    bot = WeeklyHarness(db, fake_ai)
    incoming = WeeklyMessage(chat_id=910_102)
    progress = WeeklyMessage(chat_id=910_102, is_bot=True)
    update = weekly_update(
        incoming,
        telegram_user_id=user.telegram_id,
        chat_id=910_102,
    )

    assert (
        await bot.weekly_review_launch_voice_gate(
            update,
            weekly_context(),
            control,
            progress,
            expected_user=user,
        )
        is True
    )

    assert progress.deleted == 1
    assert len(incoming.replies) == 1
    assert isinstance(incoming.replies[0]["reply_markup"], ReplyKeyboardRemove)
    canonical = incoming.replies[0]["message"]
    assert canonical.edits == [
        {
            "text": "Главное меню\n\nЧто хочешь сделать?",
            "reply_markup": SimpleNamespace(tier=SUBSCRIBER),
        }
    ]
    assert fake_ai.weekly_review_calls == []


@pytest.mark.asyncio
async def test_eligible_stale_back_uses_one_cleanup_send_and_edits_that_same_canonical(
    db,
    fake_ai,
):
    user = await weekly_user(db, 710_104)
    bot = WeeklyHarness(db, fake_ai)
    incoming = WeeklyMessage("Назад", chat_id=910_104)
    update = weekly_update(
        incoming,
        telegram_user_id=user.telegram_id,
        chat_id=910_104,
    )
    context = weekly_context()

    assert await bot.weekly_review_stale_control_gate(update, context) is True

    assert len(incoming.replies) == 1
    assert isinstance(incoming.replies[0]["reply_markup"], ReplyKeyboardRemove)
    canonical = incoming.replies[0]["message"]
    assert canonical.edits == [
        {
            "text": "Главное меню\n\nЧто хочешь сделать?",
            "reply_markup": SimpleNamespace(tier=SUBSCRIBER),
        }
    ]
    assert context.bot.sent == []
    assert context.bot.edits == []
    assert (
        await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=910_104,
            expected_access_version=user.access_version,
        )
    ).session is None
    assert fake_ai.weekly_review_calls == []


@pytest.mark.parametrize("control", ["Назад", "Пропустить", "Отменить"])
@pytest.mark.parametrize(
    "policy",
    [
        WeeklyReviewPolicy(enabled=False),
        WeeklyReviewPolicy(enabled=True, admin_only=True),
    ],
    ids=["disabled", "admin-only-subscriber"],
)
@pytest.mark.asyncio
async def test_ineligible_stale_voice_controls_retire_progress_without_weekly_storage(
    db,
    fake_ai,
    monkeypatch,
    control,
    policy,
):
    user = await weekly_user(db, 710_103)
    bot = WeeklyHarness(db, fake_ai)
    bot.weekly_review_policy = policy
    incoming = WeeklyMessage(chat_id=910_103)
    progress = WeeklyMessage(chat_id=910_103, is_bot=True)
    update = weekly_update(
        incoming,
        telegram_user_id=user.telegram_id,
        chat_id=910_103,
    )

    async def forbidden_weekly_lookup(**kwargs: Any):
        del kwargs
        raise AssertionError("disabled controls must not touch weekly storage")

    monkeypatch.setattr(
        bot.weekly_review_service,
        "current_session",
        forbidden_weekly_lookup,
    )

    assert (
        await bot.weekly_review_launch_voice_gate(
            update,
            weekly_context(),
            control,
            progress,
            expected_user=user,
        )
        is True
    )
    assert progress.deleted == 1
    assert [reply["text"] for reply in incoming.replies] == [WEEKLY_REVIEW_UNAVAILABLE_TEXT]
    assert fake_ai.weekly_review_calls == []


@pytest.mark.asyncio
async def test_cancel_command_completes_only_the_active_weekly_generation(db, fake_ai):
    user = await weekly_user(db, 7110)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_010, chat_id=9110, is_bot=True)
    active = await preview_session(
        bot,
        user,
        chat_id=9110,
        canonical_message_id=canonical.message_id,
    )
    command = WeeklyMessage("/cancel", chat_id=9110)
    update = weekly_update(command, telegram_user_id=user.telegram_id, chat_id=9110)
    context = weekly_context()

    assert await bot.weekly_review_cancel_gate(update, context) is True

    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9110,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.public_id == active.public_id
    assert current.session.phase is WeeklyReviewPhase.COMPLETED
    assert len(context.bot.edits) == 1
    assert context.bot.edits[0]["message_id"] == canonical.message_id
    assert context.bot.edits[0]["reply_markup"] is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.asyncio
async def test_preview_extraction_runs_outside_domain_transaction_and_writes_no_domain_rows(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7102)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_002, chat_id=9102, is_bot=True)
    update = weekly_update(canonical, telegram_user_id=user.telegram_id, chat_id=9102)
    context = weekly_context()
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9102,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    active_domain_transactions = 0
    provider_calls = 0
    original_session = db.session

    @asynccontextmanager
    async def tracked_session():
        nonlocal active_domain_transactions
        active_domain_transactions += 1
        try:
            async with original_session() as session:
                yield session
        finally:
            active_domain_transactions -= 1

    async def extract(ai: Any, text: str, temporal: dict[str, str]) -> WeeklyReviewExtraction:
        nonlocal provider_calls
        del ai, temporal
        provider_calls += 1
        assert active_domain_transactions == 0
        evidence = "В 15:05 сказать Назару"
        assert evidence in text
        return WeeklyReviewExtraction(
            focus="Спокойно закрывать подтверждённые задачи дня",
            approach="Держать в уме образ будущего",
            small_steps=["Выбрать один небольшой шаг"],
            reminder_candidates=[
                WeeklyReviewReminderCandidate(
                    title="Сказать Назару",
                    schedule_wording="В 15:05",
                    evidence=evidence,
                )
            ],
        )

    monkeypatch.setattr(db, "session", tracked_session)
    monkeypatch.setattr("future_self.weekly_review_handlers.extract_weekly_review_input", extract)
    text = "Фокус: спокойно закрывать задачи. В 15:05 сказать Назару"

    await bot._weekly_review_process_input_lifecycle(
        update,
        context,
        created.session,
        text,
        source="text",
    )

    assert provider_calls == 1
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9102,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.PREVIEW
    assert current.session.reminder_candidates == (
        WeeklyReminderCandidate("Сказать Назару", "В 15:05"),
    )
    async with db.sessions() as session:
        durable = await session.scalar(select(WeeklyReviewSession))
        assert durable is not None
        assert durable.reminder_candidates == [
            {"title": "Сказать Назару", "schedule_wording": "В 15:05"}
        ]
        assert "evidence" not in durable.reminder_candidates[0]
        assert text not in {
            durable.extracted_focus,
            durable.extracted_approach,
            *(durable.small_steps or []),
        }
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize("checkpoint", ["store", "retry"])
@pytest.mark.asyncio
async def test_same_access_replacement_at_processing_cas_is_preserved_without_access_cleanup(
    db,
    fake_ai,
    monkeypatch,
    checkpoint,
):
    user = await weekly_user(db, 710_202 if checkpoint == "store" else 710_203)
    chat_id = 910_202 if checkpoint == "store" else 910_203
    frozen_message_id = 121_202 if checkpoint == "store" else 121_203
    replacement_message_id = frozen_message_id + 1_000
    bot = WeeklyHarness(db, fake_ai)
    frozen = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
        canonical_message_id=frozen_message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert frozen.session is not None
    private_answer = "private weekly answer " + ("x" * 180)
    incoming = WeeklyMessage(private_answer, chat_id=chat_id)
    update = weekly_update(
        incoming,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
    )
    context = weekly_context()
    seam_started = asyncio.Event()
    seam_release = asyncio.Event()
    original_store = bot.weekly_review_service.store_extraction
    original_transition = bot.weekly_review_service.transition_session

    async def blocked_store(**kwargs: Any):
        if checkpoint == "store":
            seam_started.set()
            await seam_release.wait()
        return await original_store(**kwargs)

    async def blocked_transition(**kwargs: Any):
        if (
            checkpoint == "retry"
            and kwargs.get("expected_phase") is WeeklyReviewPhase.PROCESSING
            and kwargs.get("phase") is WeeklyReviewPhase.AWAITING_INPUT
        ):
            seam_started.set()
            await seam_release.wait()
        return await original_transition(**kwargs)

    async def forbidden_access_changed(*args: Any, **kwargs: Any):
        raise AssertionError((args, kwargs))

    async def forbidden_clear(**kwargs: Any):
        raise AssertionError(kwargs)

    monkeypatch.setattr(bot.weekly_review_service, "store_extraction", blocked_store)
    monkeypatch.setattr(bot.weekly_review_service, "transition_session", blocked_transition)
    monkeypatch.setattr(bot, "_weekly_review_access_changed", forbidden_access_changed)
    monkeypatch.setattr(bot.weekly_review_service, "clear_session_exact", forbidden_clear)
    if checkpoint == "retry":
        fake_ai.weekly_review_error = RuntimeError("private provider failure")

    processing = asyncio.create_task(
        bot._weekly_review_process_input_lifecycle(
            update,
            context,
            frozen.session,
            private_answer,
            source="text",
        )
    )
    try:
        await asyncio.wait_for(seam_started.wait(), timeout=10)
        replacement = await bot.weekly_review_service.create_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
            canonical_message_id=replacement_message_id,
            phase=WeeklyReviewPhase.ROOT,
        )
        assert replacement.status == "created"
        assert replacement.session is not None
        seam_release.set()
        await asyncio.wait_for(processing, timeout=10)
    finally:
        seam_release.set()
        if not processing.done():
            await asyncio.gather(processing, return_exceptions=True)

    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.status == "found"
    assert current.session == replacement.session
    assert current.session.public_id != frozen.session.public_id
    assert current.session.canonical_message_id == replacement_message_id
    assert current.session.phase is WeeklyReviewPhase.ROOT
    assert [call[0] for call in fake_ai.weekly_review_calls] == [private_answer]
    assert context.bot.edits == [
        {
            "chat_id": chat_id,
            "message_id": frozen_message_id,
            "text": WEEKLY_REVIEW_STALE_TEXT,
            "reply_markup": None,
        }
    ]
    assert all(edit["text"] != WEEKLY_REVIEW_ACCESS_CHANGED_TEXT for edit in context.bot.edits)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.asyncio
async def test_exact_confirm_writes_focus_and_metadata_audit_once_and_replay_is_stale(db, fake_ai):
    user = await weekly_user(db, 7103)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_003, chat_id=9103, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9103,
        canonical_message_id=canonical.message_id,
        candidates=(WeeklyReminderCandidate("Позвонить врачу", "завтра в 19:30"),),
    )
    callback_data = await issue_session_action(bot, preview, "save")
    context = weekly_context()

    winner = WeeklyQuery(callback_data, canonical)
    winner_update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9103,
        query=winner,
    )
    await bot.weekly_review_callback(winner_update, context)

    replay = WeeklyQuery(callback_data, canonical)
    replay_update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9103,
        query=replay,
    )
    await bot.weekly_review_callback(replay_update, context)

    assert winner.answers == [{"args": ()}]
    assert len(winner.edits) == 1
    assert replay.answers == [{"args": (WEEKLY_REVIEW_STALE_TEXT,), "show_alert": True}]
    assert replay.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1
        audit = await session.scalar(select(WeeklyFocusChange))
        assert audit is not None
        assert audit.operation == "created"
        assert not hasattr(audit, "focus")
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "service_method", "phase"),
    (
        ("start", "transition_session", WeeklyReviewPhase.ROOT),
        ("focus", "transition_session", WeeklyReviewPhase.ROOT),
        ("edit", "transition_session", WeeklyReviewPhase.PREVIEW),
        ("reminders", "transition_session", WeeklyReviewPhase.ROOT),
        ("cancel", "transition_session", WeeklyReviewPhase.ROOT),
        ("close", "transition_session", WeeklyReviewPhase.ROOT),
        ("done", "transition_session", WeeklyReviewPhase.SAVED),
        ("save", "confirm_focus", WeeklyReviewPhase.PREVIEW),
        ("configure", "transition_session", WeeklyReviewPhase.SAVED),
        ("back", "transition_session", WeeklyReviewPhase.CANDIDATES),
        ("delete", "transition_session", WeeklyReviewPhase.ROOT),
        ("confirm_delete", "confirm_delete", WeeklyReviewPhase.DELETE_PREVIEW),
    ),
)
@pytest.mark.parametrize(
    "outcome",
    (
        "success",
        "replay",
        "access_changed",
        "access_denied",
        "access_changed_replacement",
        "access_denied_replacement",
        "expired",
        "week_changed",
        "not_found",
        "stale",
        "focus_changed",
    ),
)
async def test_consumed_callback_typed_outcome_matrix_never_leaves_private_dead_screen(
    db,
    fake_ai,
    monkeypatch,
    action,
    service_method,
    phase,
    outcome,
):
    user = await weekly_user(db, 719_000)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_000
    canonical = WeeklyMessage(message_id=129_000, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=phase,
    )
    original_transition = bot.weekly_review_service.transition_session
    original_method = getattr(bot.weekly_review_service, service_method)
    replacement = None
    calls = 0

    async def typed_result(**kwargs):
        nonlocal calls, replacement
        calls += 1
        if service_method == "transition_session" and calls > 1:
            return await original_method(**kwargs)
        if outcome in {
            "success",
            "replay",
            "stale",
            "access_changed_replacement",
            "access_denied_replacement",
        }:
            changed = await original_transition(
                telegram_actor_id=frozen.telegram_user_id,
                chat_id=frozen.chat_id,
                expected_access_version=frozen.access_version,
                session_public_id=frozen.public_id,
                expected_session_version=frozen.version,
                expected_canonical_message_id=frozen.canonical_message_id,
                phase=WeeklyReviewPhase.ROOT,
                focus=None,
                approach=None,
                small_steps=(),
                reminder_candidates=(),
                source=None,
            )
            assert changed.session is not None
            replacement = changed.session
        elif outcome in {"expired", "week_changed", "not_found"}:
            assert await bot.weekly_review_service.clear_session_exact(
                telegram_actor_id=frozen.telegram_user_id,
                chat_id=frozen.chat_id,
                session_public_id=frozen.public_id,
                expected_session_version=frozen.version,
            )
        status = outcome.removesuffix("_replacement")
        if outcome == "success":
            status = (
                "deleted"
                if action == "confirm_delete"
                else "updated"
                if service_method == "transition_session"
                else "created"
            )
        session = replacement if outcome in {"success", "replay"} else None
        if service_method == "transition_session":
            return WeeklyReviewSessionResult(status, session)
        return WeeklyFocusMutation(status, session=session)

    monkeypatch.setattr(bot.weekly_review_service, service_method, typed_result)
    query = WeeklyQuery(await issue_session_action(bot, frozen, action), canonical)
    context = weekly_context()
    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=query,
        ),
        context,
    )

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert "PRIVATE_ACTION_MATRIX_FOCUS" not in query.edits[0]["text"]
    assert "PRIVATE_ACTION_MATRIX_CANDIDATE" not in query.edits[0]["text"]
    assert canonical.replies == []
    assert context.bot.sent == []
    if outcome in {"access_changed", "access_denied"}:
        assert query.edits[0]["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
        assert query.edits[0]["reply_markup"] is None
    elif outcome in {"access_changed_replacement", "access_denied_replacement"}:
        assert replacement is not None
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
        )
        assert current.session == replacement
        await assert_live_session_controls(bot, query, replacement)
    elif outcome in {"expired", "week_changed", "not_found"}:
        assert query.edits[0]["text"] == WEEKLY_REVIEW_RECOVERY_TEXT
        markup = query.edits[0]["reply_markup"]
        assert markup is not None
        assert all(
            str(button.callback_data).startswith("wrev:")
            for row in markup.inline_keyboard
            for button in row
        )
    elif outcome == "stale":
        assert replacement is not None
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
        )
        assert current.session == replacement
        await assert_live_session_controls(bot, query, replacement)
    elif outcome == "focus_changed":
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
        )
        assert current.session is not None
        assert current.session.phase is WeeklyReviewPhase.ROOT
        await assert_live_session_controls(bot, query, current.session)
    assert fake_ai.weekly_review_calls == []
    assert bot.reminder_handoffs == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    (
        "success",
        "replay",
        "access_changed",
        "access_denied",
        "access_changed_replacement",
        "access_denied_replacement",
        "expired",
        "week_changed",
        "not_found",
        "stale",
        "focus_changed",
    ),
)
async def test_candidate_action_uses_typed_outcome_recovery_without_repeated_handoff(
    db,
    fake_ai,
    monkeypatch,
    outcome,
):
    user = await weekly_user(db, 719_010)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_010
    canonical = WeeklyMessage(message_id=129_010, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    original_transition = bot.weekly_review_service.transition_session
    transition_calls = 0
    replacement = None

    async def typed_transition(**kwargs: Any):
        nonlocal transition_calls, replacement
        transition_calls += 1
        if transition_calls > 1:
            return await original_transition(**kwargs)
        if outcome in {"success", "replay"}:
            updated = await original_transition(**kwargs)
            assert updated.session is not None
            return WeeklyReviewSessionResult(
                "updated" if outcome == "success" else "replay",
                updated.session,
            )
        if outcome in {
            "stale",
            "access_changed_replacement",
            "access_denied_replacement",
        }:
            changed = await original_transition(
                telegram_actor_id=frozen.telegram_user_id,
                chat_id=frozen.chat_id,
                expected_access_version=frozen.access_version,
                session_public_id=frozen.public_id,
                expected_session_version=frozen.version,
                expected_canonical_message_id=frozen.canonical_message_id,
                phase=WeeklyReviewPhase.ROOT,
                focus=None,
                approach=None,
                small_steps=(),
                reminder_candidates=(),
                source=None,
            )
            assert changed.session is not None
            replacement = changed.session
        elif outcome in {"expired", "week_changed", "not_found"}:
            assert await bot.weekly_review_service.clear_session_exact(
                telegram_actor_id=frozen.telegram_user_id,
                chat_id=frozen.chat_id,
                session_public_id=frozen.public_id,
                expected_session_version=frozen.version,
            )
        return WeeklyReviewSessionResult(
            outcome.removesuffix("_replacement"),
            frozen if outcome == "focus_changed" else None,
        )

    handoff_calls = 0

    async def handoff(*args: Any, **kwargs: Any) -> bool:
        nonlocal handoff_calls
        del args, kwargs
        handoff_calls += 1
        await query.edit_message_text(
            "SAFE_REMINDER_PREVIEW",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Cancel", callback_data="reminder:live")]]
            ),
        )
        return True

    monkeypatch.setattr(bot.weekly_review_service, "transition_session", typed_transition)
    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", handoff)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    context = weekly_context()
    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=query,
        ),
        context,
    )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert handoff_calls == (1 if outcome == "success" else 0)
    assert len(query.edits) == 1
    if outcome == "success":
        assert query.edits[-1]["text"] == "SAFE_REMINDER_PREVIEW"
    elif outcome in {"access_changed", "access_denied"}:
        assert query.edits[-1]["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
        assert query.edits[-1]["reply_markup"] is None
    elif outcome in {"expired", "week_changed", "not_found"}:
        assert query.edits[-1]["text"] == WEEKLY_REVIEW_RECOVERY_TEXT
        assert query.edits[-1]["reply_markup"] is not None
    else:
        assert query.edits[-1]["reply_markup"] is not None
    if replacement is not None:
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
        )
        assert current.session == replacement
    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checkpoint",
    ("pre_fence", "handoff_exception", "post_fence", "stale_replacement"),
)
async def test_candidate_error_and_fence_paths_recover_once_without_repeating_handoff(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    checkpoint,
):
    private = "PRIVATE_CANDIDATE_HANDOFF_ERROR"
    user = await weekly_user(db, 719_011)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_011
    canonical = WeeklyMessage(message_id=129_011, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    replacement = None
    fence_calls = 0
    original_fence = bot._weekly_review_candidate_fence

    async def fence(session: Any) -> bool:
        nonlocal fence_calls
        fence_calls += 1
        if checkpoint == "pre_fence":
            return False
        if checkpoint == "post_fence" and fence_calls == 2:
            return False
        return await original_fence(session)

    handoff_calls = 0

    async def handoff(*args: Any, **kwargs: Any) -> bool:
        nonlocal handoff_calls, replacement
        del args, kwargs
        handoff_calls += 1
        if checkpoint == "handoff_exception":
            raise RuntimeError(private)
        if checkpoint == "stale_replacement":
            created = await bot.weekly_review_service.create_session(
                telegram_actor_id=user.telegram_id,
                chat_id=chat_id,
                expected_access_version=user.access_version,
                canonical_message_id=canonical.message_id,
                phase=WeeklyReviewPhase.ROOT,
                replace_existing=True,
            )
            assert created.session is not None
            replacement = created.session
            raise RuntimeError(private)
        await query.edit_message_text(
            "PRIVATE_REMINDER_HANDOFF_SCREEN",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Cancel", callback_data="reminder:live")]]
            ),
        )
        return True

    monkeypatch.setattr(bot, "_weekly_review_candidate_fence", fence)
    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", handoff)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    context = weekly_context()
    with caplog.at_level("WARNING"):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            context,
        )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert handoff_calls == (0 if checkpoint == "pre_fence" else 1)
    assert query.edits
    assert query.edits[-1]["reply_markup"] is not None
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    if replacement is not None:
        assert current.session == replacement
    else:
        assert current.session.public_id == frozen.public_id
        assert current.session.phase is WeeklyReviewPhase.CANDIDATES
        await assert_live_session_controls(bot, query, current.session)
    assert handoff_calls <= 1
    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot._weekly_review_tasks == set()
    assert private not in caplog.text
    assert str(user.telegram_id) not in caplog.text
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_candidate_direct_cancellation_restores_exact_lineage_then_propagates(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 719_012)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_012
    canonical = WeeklyMessage(message_id=129_012, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    handoff_calls = 0

    async def cancel(*args: Any, **kwargs: Any) -> bool:
        nonlocal handoff_calls
        del args, kwargs
        handoff_calls += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", cancel)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            weekly_context(),
        )
    await asyncio.sleep(0)

    assert handoff_calls == 1
    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.public_id == frozen.public_id
    assert current.session.phase is WeeklyReviewPhase.CANDIDATES
    await assert_live_session_controls(bot, query, current.session)
    assert fake_ai.weekly_review_calls == []
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_candidate_returned_generation_shrink_recovers_without_indexing_or_handoff(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 719_014)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_014
    canonical = WeeklyMessage(message_id=129_014, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    original_transition = bot.weekly_review_service.transition_session
    calls = 0

    async def shrink(**kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            kwargs["reminder_candidates"] = ()
        return await original_transition(**kwargs)

    async def forbidden(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(bot.weekly_review_service, "transition_session", shrink)
    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", forbidden)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=query,
        ),
        weekly_context(),
    )
    await asyncio.sleep(0)

    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert calls == 2
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.CANDIDATES
    assert current.session.reminder_candidates == ()
    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    await assert_live_session_controls(bot, query, current.session)
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_candidate_replay_renders_only_dispatchable_back_and_never_repeats_handoff(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 719_015)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_015
    canonical = WeeklyMessage(message_id=129_015, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    original_transition = bot.weekly_review_service.transition_session
    calls = 0

    async def replay_once(**kwargs: Any):
        nonlocal calls
        calls += 1
        result = await original_transition(**kwargs)
        if calls == 1:
            return WeeklyReviewSessionResult("replay", result.session)
        return result

    async def forbidden(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(bot.weekly_review_service, "transition_session", replay_once)
    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", forbidden)
    first = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        query=first,
    )
    await bot.weekly_review_callback(update, weekly_context())

    markup = first.edits[-1]["reply_markup"]
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert len(callbacks) == 1
    claim = await bot.weekly_review_capabilities.peek(
        str(callbacks[0]).removeprefix("wrev:"),
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
    )
    assert claim is not None
    assert claim.action == "back"
    second = WeeklyQuery(str(callbacks[0]), canonical)
    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=second,
        ),
        weekly_context(),
    )
    await asyncio.sleep(0)

    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.phase is WeeklyReviewPhase.SAVED
    assert first.answers == [{"args": ()}]
    assert second.answers == [{"args": ()}]
    assert second.edits[-1]["reply_markup"] is not None
    assert bot.reminder_handoffs == []
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_candidate_cancel_after_distinct_replacement_does_not_repaint_or_revoke_it(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 719_016)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_016
    canonical = WeeklyMessage(message_id=129_016, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    replacement = None
    replacement_token = None
    handoff_calls = 0

    async def replace_then_cancel(*args: Any, **kwargs: Any) -> bool:
        nonlocal replacement, replacement_token, handoff_calls
        del args, kwargs
        handoff_calls += 1
        created = await bot.weekly_review_service.create_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
            canonical_message_id=canonical.message_id,
            phase=WeeklyReviewPhase.ROOT,
            replace_existing=True,
        )
        assert created.session is not None
        replacement = created.session
        tokens = await bot.weekly_review_capabilities.issue(
            actions=("start",),
            owner_id=replacement.owner_id,
            telegram_user_id=replacement.telegram_user_id,
            chat_id=replacement.chat_id,
            canonical_message_id=replacement.canonical_message_id,
            access_version=replacement.access_version,
            week_start=replacement.week_start,
            session_public_id=replacement.public_id,
            session_version=replacement.version,
        )
        replacement_token = tokens["start"]
        raise asyncio.CancelledError

    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", replace_then_cancel)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            weekly_context(),
        )
    await asyncio.sleep(0)

    assert handoff_calls == 1
    assert replacement is not None
    assert replacement_token is not None
    assert query.edits == []
    assert (
        await bot.weekly_review_capabilities.peek(
            replacement_token,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            canonical_message_id=canonical.message_id,
        )
        is not None
    )
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session == replacement
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_candidate_cancel_does_not_clear_newer_same_lineage_generation(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 719_018)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_018
    canonical = WeeklyMessage(message_id=129_018, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    replacement = None
    replacement_token = None

    async def advance_then_cancel(*args: Any, **kwargs: Any) -> bool:
        nonlocal replacement, replacement_token
        del args, kwargs
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
        )
        assert current.session is not None
        advanced = await bot.weekly_review_service.transition_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
            session_public_id=current.session.public_id,
            expected_session_version=current.session.version,
            expected_canonical_message_id=canonical.message_id,
            expected_phase=WeeklyReviewPhase.REMINDER_HANDOFF,
            phase=WeeklyReviewPhase.ROOT,
            focus=None,
            approach=None,
            small_steps=(),
            reminder_candidates=(),
            source=None,
        )
        assert advanced.session is not None
        replacement = advanced.session
        tokens = await bot.weekly_review_capabilities.issue(
            actions=("start",),
            owner_id=replacement.owner_id,
            telegram_user_id=replacement.telegram_user_id,
            chat_id=replacement.chat_id,
            canonical_message_id=replacement.canonical_message_id,
            access_version=replacement.access_version,
            week_start=replacement.week_start,
            session_public_id=replacement.public_id,
            session_version=replacement.version,
        )
        replacement_token = tokens["start"]
        raise asyncio.CancelledError

    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", advance_then_cancel)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            weekly_context(),
        )
    await asyncio.sleep(0)

    assert replacement is not None
    assert replacement.public_id == frozen.public_id
    assert replacement.version == frozen.version + 2
    assert replacement_token is not None
    assert query.edits == []
    assert (
        await bot.weekly_review_capabilities.peek(
            replacement_token,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            canonical_message_id=canonical.message_id,
        )
        is not None
    )
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session == replacement
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_candidate_cancel_after_access_downgrade_neutralizes_exact_canonical(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_CANDIDATE_ACCESS_CANCEL"
    user = await weekly_user(db, 719_017)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_017
    canonical = WeeklyMessage(message_id=129_017, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    handoff_calls = 0

    async def downgrade_then_cancel(*args: Any, **kwargs: Any) -> bool:
        nonlocal handoff_calls
        del args, kwargs
        handoff_calls += 1
        await bot.access_service.set_guest(user.telegram_id, source="test")
        raise asyncio.CancelledError(private)

    monkeypatch.setattr(bot, "reminder_from_weekly_candidate", downgrade_then_cancel)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "candidate:0"), canonical)
    with caplog.at_level("WARNING"):
        with pytest.raises(asyncio.CancelledError):
            await bot.weekly_review_callback(
                weekly_update(
                    canonical,
                    telegram_user_id=user.telegram_id,
                    chat_id=chat_id,
                    query=query,
                ),
                weekly_context(),
            )
    await asyncio.sleep(0)

    assert handoff_calls == 1
    assert query.answers == [{"args": ()}]
    assert query.edits == [
        {
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    assert bot.weekly_review_capabilities._capabilities == {}
    assert bot._weekly_review_tasks == set()
    assert private not in caplog.text
    assert str(user.telegram_id) not in caplog.text
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checkpoint",
    (
        "materialize_error",
        "pre_provider_fence",
        "provider_error",
        "post_provider_fence",
        "render_false",
        "stale_replacement",
        "provider_cancel",
    ),
)
async def test_today_error_and_fence_paths_recover_without_repeating_provider(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    checkpoint,
):
    private = "PRIVATE_TODAY_CALLBACK_ERROR"
    user = await weekly_user(db, 719_013)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 919_013
    canonical = WeeklyMessage(message_id=129_013, chat_id=chat_id, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.SAVED,
    )
    snapshot = SimpleNamespace(weekly_focus="PRIVATE_TODAY_FOCUS")
    plan = SimpleNamespace(
        vision_reminder="PRIVATE_TODAY_VISION",
        main_focus="PRIVATE_TODAY_MAIN_FOCUS",
        actions=["PRIVATE_TODAY_ACTION"],
        hard_day_minimum="PRIVATE_TODAY_MINIMUM",
    )
    materialize_calls = 0
    provider_calls = 0

    async def materialize(*args: Any, **kwargs: Any) -> Any:
        nonlocal materialize_calls
        del args, kwargs
        materialize_calls += 1
        if checkpoint == "materialize_error":
            raise RuntimeError(private)
        return snapshot

    async def generate(value: Any) -> Any:
        nonlocal provider_calls
        assert value is snapshot
        provider_calls += 1
        if checkpoint == "provider_error":
            raise RuntimeError(private)
        if checkpoint == "provider_cancel":
            raise asyncio.CancelledError
        return plan

    bot.focus_service = SimpleNamespace(
        materialize_today_application=materialize,
        generate_today_plan=generate,
    )
    fence_calls = 0
    replacement = None

    async def fence(value: Any, session: Any) -> bool:
        nonlocal fence_calls, replacement
        assert value is snapshot
        del session
        fence_calls += 1
        if checkpoint == "pre_provider_fence" and fence_calls == 1:
            return False
        if checkpoint == "post_provider_fence" and fence_calls == 2:
            return False
        if checkpoint == "stale_replacement" and fence_calls == 2:
            created = await bot.weekly_review_service.create_session(
                telegram_actor_id=user.telegram_id,
                chat_id=chat_id,
                expected_access_version=user.access_version,
                canonical_message_id=canonical.message_id,
                phase=WeeklyReviewPhase.ROOT,
                replace_existing=True,
            )
            assert created.session is not None
            replacement = created.session
            return False
        return True

    monkeypatch.setattr(bot, "_weekly_review_today_fence", fence)
    if checkpoint == "render_false":
        original_render = bot._weekly_review_render
        render_calls = 0

        async def fail_first_render(*args: Any, **kwargs: Any) -> bool:
            nonlocal render_calls
            render_calls += 1
            if render_calls == 1:
                return False
            return await original_render(*args, **kwargs)

        monkeypatch.setattr(bot, "_weekly_review_render", fail_first_render)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "today"), canonical)
    context = weekly_context()
    with caplog.at_level("WARNING"):
        if checkpoint == "provider_cancel":
            with pytest.raises(asyncio.CancelledError):
                await bot.weekly_review_callback(
                    weekly_update(
                        canonical,
                        telegram_user_id=user.telegram_id,
                        chat_id=chat_id,
                        query=query,
                    ),
                    context,
                )
        else:
            await bot.weekly_review_callback(
                weekly_update(
                    canonical,
                    telegram_user_id=user.telegram_id,
                    chat_id=chat_id,
                    query=query,
                ),
                context,
            )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert materialize_calls == 1
    assert provider_calls == (0 if checkpoint in {"materialize_error", "pre_provider_fence"} else 1)
    assert len(query.edits) == 1
    assert query.edits[-1]["reply_markup"] is not None
    assert "PRIVATE_TODAY_VISION" not in query.edits[-1]["text"]
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    if replacement is not None:
        assert current.session == replacement
    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot._weekly_review_tasks == set()
    assert private not in caplog.text
    assert str(user.telegram_id) not in caplog.text
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "service_method", "phase"),
    (
        ("start", "transition_session", WeeklyReviewPhase.ROOT),
        ("focus", "transition_session", WeeklyReviewPhase.ROOT),
        ("edit", "transition_session", WeeklyReviewPhase.PREVIEW),
        ("reminders", "transition_session", WeeklyReviewPhase.ROOT),
        ("cancel", "transition_session", WeeklyReviewPhase.ROOT),
        ("close", "transition_session", WeeklyReviewPhase.ROOT),
        ("done", "transition_session", WeeklyReviewPhase.SAVED),
        ("save", "confirm_focus", WeeklyReviewPhase.PREVIEW),
        ("configure", "transition_session", WeeklyReviewPhase.SAVED),
        ("back", "transition_session", WeeklyReviewPhase.CANDIDATES),
        ("delete", "transition_session", WeeklyReviewPhase.ROOT),
        ("confirm_delete", "confirm_delete", WeeklyReviewPhase.DELETE_PREVIEW),
    ),
)
async def test_consumed_callback_ordinary_error_recovers_with_safe_log(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    action,
    service_method,
    phase,
):
    user = await weekly_user(db, 719_001)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=129_001, chat_id=919_001, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=919_001,
        canonical_message_id=canonical.message_id,
        phase=phase,
    )

    async def fail(**kwargs):
        del kwargs
        raise RuntimeError("PRIVATE_ERROR_TEXT")

    monkeypatch.setattr(bot.weekly_review_service, service_method, fail)
    query = WeeklyQuery(await issue_session_action(bot, frozen, action), canonical)
    with caplog.at_level("WARNING"):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=919_001,
                query=query,
            ),
            weekly_context(),
        )

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert query.edits[0]["reply_markup"] is not None
    assert "operation=" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert "PRIVATE_ACTION_MATRIX_FOCUS" not in caplog.text
    assert "PRIVATE_ERROR_TEXT" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "service_method", "phase"),
    (
        ("start", "transition_session", WeeklyReviewPhase.ROOT),
        ("focus", "transition_session", WeeklyReviewPhase.ROOT),
        ("edit", "transition_session", WeeklyReviewPhase.PREVIEW),
        ("reminders", "transition_session", WeeklyReviewPhase.ROOT),
        ("cancel", "transition_session", WeeklyReviewPhase.ROOT),
        ("close", "transition_session", WeeklyReviewPhase.ROOT),
        ("done", "transition_session", WeeklyReviewPhase.SAVED),
        ("save", "confirm_focus", WeeklyReviewPhase.PREVIEW),
        ("configure", "transition_session", WeeklyReviewPhase.SAVED),
        ("back", "transition_session", WeeklyReviewPhase.CANDIDATES),
        ("delete", "transition_session", WeeklyReviewPhase.ROOT),
        ("confirm_delete", "confirm_delete", WeeklyReviewPhase.DELETE_PREVIEW),
    ),
)
async def test_consumed_callback_direct_cancellation_propagates_and_is_observed(
    db,
    fake_ai,
    monkeypatch,
    action,
    service_method,
    phase,
):
    user = await weekly_user(db, 719_002)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=129_002, chat_id=919_002, is_bot=True)
    frozen = await session_in_phase(
        bot,
        user,
        chat_id=919_002,
        canonical_message_id=canonical.message_id,
        phase=phase,
    )

    async def cancel(**kwargs):
        del kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(bot.weekly_review_service, service_method, cancel)
    query = WeeklyQuery(await issue_session_action(bot, frozen, action), canonical)
    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=919_002,
                query=query,
            ),
            weekly_context(),
        )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert query.edits[0]["reply_markup"] is not None
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_focus_changed_recovery_transition_error_renders_live_controls_with_safe_log(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    user = await weekly_user(db, 719_003)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=129_003, chat_id=919_003, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=919_003,
        canonical_message_id=canonical.message_id,
        focus="PRIVATE_FOCUS_CHANGED_FOCUS",
    )

    async def focus_changed(**kwargs):
        del kwargs
        return WeeklyFocusMutation("focus_changed", session=frozen)

    async def fail_recovery(**kwargs):
        del kwargs
        raise RuntimeError("PRIVATE_FOCUS_CHANGED_ERROR")

    monkeypatch.setattr(bot.weekly_review_service, "confirm_focus", focus_changed)
    monkeypatch.setattr(bot.weekly_review_service, "transition_session", fail_recovery)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "save"), canonical)
    context = weekly_context()
    with caplog.at_level("WARNING"):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=919_003,
                query=query,
            ),
            context,
        )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    await assert_live_session_controls(bot, query, frozen)
    assert canonical.replies == []
    assert context.bot.sent == []
    assert bot._weekly_review_tasks == set()
    assert "operation=focus_changed_recovery" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert "PRIVATE_FOCUS_CHANGED_FOCUS" not in caplog.text
    assert "PRIVATE_FOCUS_CHANGED_ERROR" not in caplog.text
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_focus_changed_recovery_transition_cancellation_propagates_and_is_observed(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 719_004)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=129_004, chat_id=919_004, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=919_004,
        canonical_message_id=canonical.message_id,
    )

    async def focus_changed(**kwargs):
        del kwargs
        return WeeklyFocusMutation("focus_changed", session=frozen)

    async def cancel_recovery(**kwargs):
        del kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(bot.weekly_review_service, "confirm_focus", focus_changed)
    monkeypatch.setattr(bot.weekly_review_service, "transition_session", cancel_recovery)
    query = WeeklyQuery(await issue_session_action(bot, frozen, "save"), canonical)
    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=919_004,
                query=query,
            ),
            weekly_context(),
        )
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    await assert_live_session_controls(bot, query, frozen)
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_two_chat_stale_confirm_recovers_neutral_root_without_focus_dml(
    db,
    fake_ai,
):
    user = await weekly_user(db, 710_204)
    bot = WeeklyHarness(db, fake_ai)
    stale_canonical = WeeklyMessage(message_id=121_204, chat_id=910_204, is_bot=True)
    winning_canonical = WeeklyMessage(message_id=121_205, chat_id=910_205, is_bot=True)
    stale_preview = await preview_session(
        bot,
        user,
        chat_id=910_204,
        canonical_message_id=stale_canonical.message_id,
        focus="Первый конкурентный фокус",
    )
    winning_preview = await preview_session(
        bot,
        user,
        chat_id=910_205,
        canonical_message_id=winning_canonical.message_id,
        focus="Подтверждённый фокус из второго чата",
    )
    context = weekly_context()
    winning_query = WeeklyQuery(
        await issue_session_action(bot, winning_preview, "save"),
        winning_canonical,
    )
    await bot.weekly_review_callback(
        weekly_update(
            winning_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=910_205,
            query=winning_query,
        ),
        context,
    )
    async with db.sessions() as session:
        before_focus = await session.scalar(select(func.count(WeeklyFocus.id)))
        before_audit = await session.scalar(select(func.count(WeeklyFocusChange.id)))
    assert (before_focus, before_audit) == (1, 1)

    stale_query = WeeklyQuery(
        await issue_session_action(bot, stale_preview, "save"),
        stale_canonical,
    )
    await bot.weekly_review_callback(
        weekly_update(
            stale_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=910_204,
            query=stale_query,
        ),
        context,
    )

    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=910_204,
        expected_access_version=user.access_version,
    )
    assert current.status == "found"
    assert current.session is not None
    assert current.session.public_id == stale_preview.public_id
    assert current.session.canonical_message_id == stale_canonical.message_id
    assert current.session.phase is WeeklyReviewPhase.ROOT
    assert current.session.focus is None
    assert stale_query.answers == [{"args": ()}]
    assert len(stale_query.edits) == 1
    assert WEEKLY_REVIEW_ACCESS_CHANGED_TEXT not in stale_query.edits[0]["text"]
    assert stale_canonical.replies == []
    assert context.bot.sent == []
    await assert_live_session_controls(bot, stale_query, current.session)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == before_focus
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == before_audit


@pytest.mark.asyncio
async def test_two_chat_stale_delete_recovers_neutral_root_without_focus_dml(
    db,
    fake_ai,
):
    user = await weekly_user(db, 710_205)
    bot = WeeklyHarness(db, fake_ai)
    context = weekly_context()
    seed_canonical = WeeklyMessage(message_id=121_206, chat_id=910_206, is_bot=True)
    seed_preview = await preview_session(
        bot,
        user,
        chat_id=910_206,
        canonical_message_id=seed_canonical.message_id,
        focus="Исходный фокус недели",
    )
    seed_query = WeeklyQuery(
        await issue_session_action(bot, seed_preview, "save"),
        seed_canonical,
    )
    await bot.weekly_review_callback(
        weekly_update(
            seed_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=910_206,
            query=seed_query,
        ),
        context,
    )

    stale_canonical = WeeklyMessage(message_id=121_207, chat_id=910_207, is_bot=True)
    stale_root = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=910_207,
        expected_access_version=user.access_version,
        canonical_message_id=stale_canonical.message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    assert stale_root.session is not None
    stale_delete = await bot.weekly_review_service.transition_session(
        telegram_actor_id=user.telegram_id,
        chat_id=910_207,
        expected_access_version=user.access_version,
        session_public_id=stale_root.session.public_id,
        expected_session_version=stale_root.session.version,
        expected_canonical_message_id=stale_canonical.message_id,
        phase=WeeklyReviewPhase.DELETE_PREVIEW,
    )
    assert stale_delete.session is not None

    winning_canonical = WeeklyMessage(message_id=121_208, chat_id=910_208, is_bot=True)
    winning_preview = await preview_session(
        bot,
        user,
        chat_id=910_208,
        canonical_message_id=winning_canonical.message_id,
        focus="Новый фокус из второго чата",
    )
    winning_query = WeeklyQuery(
        await issue_session_action(bot, winning_preview, "save"),
        winning_canonical,
    )
    await bot.weekly_review_callback(
        weekly_update(
            winning_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=910_208,
            query=winning_query,
        ),
        context,
    )
    async with db.sessions() as session:
        before_focus = await session.scalar(select(func.count(WeeklyFocus.id)))
        before_audit = await session.scalar(select(func.count(WeeklyFocusChange.id)))
    assert (before_focus, before_audit) == (1, 2)

    stale_query = WeeklyQuery(
        await issue_session_action(bot, stale_delete.session, "confirm_delete"),
        stale_canonical,
    )
    await bot.weekly_review_callback(
        weekly_update(
            stale_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=910_207,
            query=stale_query,
        ),
        context,
    )

    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=910_207,
        expected_access_version=user.access_version,
    )
    assert current.status == "found"
    assert current.session is not None
    assert current.session.public_id == stale_delete.session.public_id
    assert current.session.canonical_message_id == stale_canonical.message_id
    assert current.session.phase is WeeklyReviewPhase.ROOT
    assert current.session.focus is None
    assert stale_query.answers == [{"args": ()}]
    assert len(stale_query.edits) == 1
    assert WEEKLY_REVIEW_ACCESS_CHANGED_TEXT not in stale_query.edits[0]["text"]
    assert stale_canonical.replies == []
    assert context.bot.sent == []
    await assert_live_session_controls(bot, stale_query, current.session)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == before_focus
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == before_audit


@pytest.mark.asyncio
async def test_wrong_callback_bindings_do_not_consume_opaque_owner_capability(db, fake_ai):
    user = await weekly_user(db, 7104)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_004, chat_id=9104, is_bot=True)
    week_start = bot.weekly_review_service.target_week_start(user.timezone, review_weekday=6)
    tokens = await bot.weekly_review_capabilities.issue(
        actions=("close",),
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9104,
        canonical_message_id=canonical.message_id,
        access_version=user.access_version,
        week_start=week_start,
    )
    data = f"wrev:{tokens['close']}"
    assert re.fullmatch(r"wrev:[A-Za-z0-9_-]+", data)
    assert "close" not in data
    context = weekly_context()

    attempts = (
        (999_104, 9104, canonical.message_id),
        (user.telegram_id, 999_104, canonical.message_id),
        (user.telegram_id, 9104, canonical.message_id + 1),
    )
    for telegram_id, chat_id, message_id in attempts:
        wrong_message = WeeklyMessage(message_id=message_id, chat_id=chat_id, is_bot=True)
        query = WeeklyQuery(data, wrong_message)
        update = weekly_update(
            wrong_message,
            telegram_user_id=telegram_id,
            chat_id=chat_id,
            query=query,
        )
        await bot.weekly_review_callback(update, context)
        assert query.answers == [{"args": (WEEKLY_REVIEW_STALE_TEXT,), "show_alert": True}]
        assert query.edits == []

    owner = WeeklyQuery(data, canonical)
    owner_update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9104,
        query=owner,
    )
    await bot.weekly_review_callback(owner_update, context)
    assert owner.answers == [{"args": ()}]
    assert len(owner.edits) == 1


@pytest.mark.asyncio
async def test_candidates_are_not_created_automatically_and_handoff_is_one_at_a_time(db, fake_ai):
    user = await weekly_user(db, 7105)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_005, chat_id=9105, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9105,
        canonical_message_id=canonical.message_id,
        candidates=(
            WeeklyReminderCandidate("Сказать Назару", "в 15:05"),
            WeeklyReminderCandidate("Позвонить врачу", "завтра в 19:30"),
        ),
    )
    confirmed = await bot.weekly_review_service.confirm_focus(
        telegram_actor_id=user.telegram_id,
        chat_id=9105,
        expected_access_version=user.access_version,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=canonical.message_id,
        expected_week_start=preview.week_start,
    )
    assert confirmed.session is not None
    candidates = await bot.weekly_review_service.transition_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9105,
        expected_access_version=user.access_version,
        session_public_id=confirmed.session.public_id,
        expected_session_version=confirmed.session.version,
        expected_canonical_message_id=canonical.message_id,
        expected_phase=WeeklyReviewPhase.SAVED,
        phase=WeeklyReviewPhase.CANDIDATES,
    )
    assert candidates.session is not None
    data = await issue_session_action(bot, candidates.session, "candidate:1")
    query = WeeklyQuery(data, canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9105,
        query=query,
    )

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
    await bot.weekly_review_callback(update, weekly_context())

    assert query.answers == [{"args": ()}]
    assert len(bot.reminder_handoffs) == 1
    assert bot.reminder_handoffs[0]["title"] == "Позвонить врачу"
    assert bot.reminder_handoffs[0]["schedule_wording"] == "завтра в 19:30"
    assert bot.reminder_handoffs[0]["canonical_message"] is canonical
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize("delivery_path", ["query", "bot"])
@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_post_edit_access_change_neutralizes_same_canonical_and_preserves_replacement(
    db,
    fake_ai,
    access_change,
    delivery_path,
):
    user = await weekly_user(db, 7106)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_006, chat_id=9106, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=9106,
        canonical_message_id=canonical.message_id,
    )
    query = BlockingWeeklyQuery("", canonical) if delivery_path == "query" else None
    blocking_bot = BlockingWeeklyBot()
    context = weekly_context(blocking_bot)

    delivery = asyncio.create_task(
        bot._weekly_review_render(
            context,
            frozen,
            "private preview",
            None,
            query=query,
        )
    )
    primary_started = query.started if query is not None else blocking_bot.started
    await asyncio.wait_for(primary_started.wait(), timeout=1)
    replacement = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9106,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    assert replacement.session is not None
    await bot.access_service.set_guest(user.telegram_id, source="test")
    if access_change == "bounce":
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    if query is not None:
        query.release.set()
    else:
        blocking_bot.release.set()

    assert await asyncio.wait_for(delivery, timeout=1) is False
    if query is not None:
        assert query.edits[-1] == {
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    else:
        assert blocking_bot.edits[-1] == {
            "chat_id": 9106,
            "message_id": canonical.message_id,
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    async with db.sessions() as session:
        current = await session.scalar(
            select(WeeklyReviewSession).where(WeeklyReviewSession.owner_id == user.id)
        )
        assert current is not None
        assert current.public_id == replacement.session.public_id
        assert current.version == replacement.session.version
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_pre_edit_access_change_clears_exact_old_generation_without_primary_edit(
    db,
    fake_ai,
    monkeypatch,
    access_change,
):
    user = await weekly_user(db, 7114)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_014, chat_id=9114, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=9114,
        canonical_message_id=canonical.message_id,
    )
    query = WeeklyQuery("", canonical)
    lookup_started = asyncio.Event()
    lookup_release = asyncio.Event()
    original_access = bot._weekly_review_access_values

    async def blocked_access(telegram_user_id: int, chat_id: int):
        lookup_started.set()
        await lookup_release.wait()
        return await original_access(telegram_user_id, chat_id)

    monkeypatch.setattr(bot, "_weekly_review_access_values", blocked_access)
    delivery = asyncio.create_task(
        bot._weekly_review_render(
            weekly_context(),
            frozen,
            "must not be painted",
            None,
            query=query,
        )
    )
    await asyncio.wait_for(lookup_started.wait(), timeout=1)
    await bot.access_service.set_guest(user.telegram_id, source="test")
    if access_change == "bounce":
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    lookup_release.set()

    assert await asyncio.wait_for(delivery, timeout=1) is False
    assert query.edits == [
        {
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_confirm_post_edit_race_keeps_exactly_once_dml_and_replacement(
    db,
    fake_ai,
    access_change,
):
    user = await weekly_user(db, 7115)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_015, chat_id=9115, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9115,
        canonical_message_id=canonical.message_id,
        candidates=(WeeklyReminderCandidate("Позвонить врачу", "завтра в 19:30"),),
    )
    callback_data = await issue_session_action(bot, preview, "save")
    query = BlockingWeeklyQuery(callback_data, canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9115,
        query=query,
    )
    callback = asyncio.create_task(bot.weekly_review_callback(update, weekly_context()))
    await asyncio.wait_for(query.started.wait(), timeout=1)
    assert query.answers == [{"args": ()}]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1

    replacement = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9115,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    assert replacement.session is not None
    await bot.access_service.set_guest(user.telegram_id, source="test")
    if access_change == "bounce":
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    query.release.set()
    await asyncio.wait_for(callback, timeout=1)

    assert query.answers == [{"args": ()}]
    assert query.edits[-1] == {
        "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
        "reply_markup": None,
        "parse_mode": None,
    }
    replay = WeeklyQuery(callback_data, canonical)
    replay_update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9115,
        query=replay,
    )
    await bot.weekly_review_callback(replay_update, weekly_context())
    assert replay.answers == [{"args": (WEEKLY_REVIEW_STALE_TEXT,), "show_alert": True}]
    async with db.sessions() as session:
        current = await session.scalar(
            select(WeeklyReviewSession).where(WeeklyReviewSession.owner_id == user.id)
        )
        assert current is not None
        assert current.public_id == replacement.session.public_id
        assert current.version == replacement.session.version
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_direct_telegram_cancellation_propagates_and_tracked_task_is_observed(db, fake_ai):
    user = await weekly_user(db, 7111)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_011, chat_id=9111, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=9111,
        canonical_message_id=canonical.message_id,
    )
    query = CancellingWeeklyQuery("", canonical)

    with pytest.raises(asyncio.CancelledError):
        await bot._weekly_review_render(
            weekly_context(),
            frozen,
            "private preview",
            None,
            query=query,
        )
    await asyncio.sleep(0)

    assert query.edits == []
    assert bot._weekly_review_tasks == set()
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9111,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.public_id == frozen.public_id
    assert current.session.version == frozen.version


@pytest.mark.asyncio
async def test_today_post_edit_fence_cancellation_compensates_then_propagates(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 711_101)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_101, chat_id=911_101, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=911_101,
        canonical_message_id=canonical.message_id,
    )
    query = WeeklyQuery("", canonical)
    fence_calls = 0

    async def cancel_after_edit(snapshot: Any, session: Any) -> bool:
        nonlocal fence_calls
        del snapshot, session
        fence_calls += 1
        if fence_calls == 2:
            raise asyncio.CancelledError
        return True

    monkeypatch.setattr(bot, "_weekly_review_today_fence_safely", cancel_after_edit)

    with pytest.raises(asyncio.CancelledError):
        await bot._weekly_review_render(
            weekly_context(),
            frozen,
            "private today plan",
            None,
            query=query,
            today_snapshot=object(),
        )
    await asyncio.sleep(0)

    assert fence_calls == 2
    assert [edit["text"] for edit in query.edits] == [
        "private today plan",
        WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
    ]
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_outer_cancellation_keeps_post_edit_fence_alive_and_clears_old_generation(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7113)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_013, chat_id=9113, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=9113,
        canonical_message_id=canonical.message_id,
    )
    query = WeeklyQuery("", canonical)
    lookup_started = asyncio.Event()
    lookup_release = asyncio.Event()
    lookup_calls = 0
    original_access = bot._weekly_review_access_values

    async def blocked_access(telegram_user_id: int, chat_id: int):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 3:
            lookup_started.set()
            await lookup_release.wait()
        return await original_access(telegram_user_id, chat_id)

    monkeypatch.setattr(bot, "_weekly_review_access_values", blocked_access)
    outer = asyncio.create_task(
        bot._weekly_review_render(
            weekly_context(),
            frozen,
            "accepted private preview",
            None,
            query=query,
        )
    )
    await asyncio.wait_for(lookup_started.wait(), timeout=1)
    assert query.edits == [{"text": "accepted private preview", "reply_markup": None}]

    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert len(bot._weekly_review_tasks) == 1
    inner_tasks = tuple(bot._weekly_review_tasks)
    await bot.access_service.set_guest(user.telegram_id, source="test")
    lookup_release.set()
    await asyncio.wait_for(asyncio.gather(*inner_tasks), timeout=1)
    await asyncio.sleep(0)
    assert bot._weekly_review_tasks == set()
    assert query.edits[-1] == {
        "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
        "reply_markup": None,
        "parse_mode": None,
    }
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0


@pytest.mark.asyncio
async def test_old_delivery_neutralizes_before_fresh_access_replacement_can_render(db, fake_ai):
    user = await weekly_user(db, 7112)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_012, chat_id=9112, is_bot=True)
    frozen = await preview_session(
        bot,
        user,
        chat_id=9112,
        canonical_message_id=canonical.message_id,
    )
    old_query = BlockingWeeklyQuery("", canonical)
    context = weekly_context()
    old_delivery = asyncio.create_task(
        bot._weekly_review_render(
            context,
            frozen,
            "old private preview",
            None,
            query=old_query,
        )
    )
    await asyncio.wait_for(old_query.started.wait(), timeout=1)

    await bot.access_service.set_guest(user.telegram_id, source="test")
    await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    async with db.sessions() as session:
        fresh_user = await session.scalar(select(User).where(User.id == user.id))
    assert fresh_user is not None
    replacement = await bot.weekly_review_service.create_session(
        telegram_actor_id=fresh_user.telegram_id,
        chat_id=9112,
        expected_access_version=fresh_user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    assert replacement.session is not None
    old_query.release.set()
    assert await asyncio.wait_for(old_delivery, timeout=1) is False
    assert old_query.edits[-1]["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT

    replacement_query = WeeklyQuery("", canonical)
    assert (
        await bot._weekly_review_render_current(
            context,
            replacement.session,
            query=replacement_query,
        )
        is True
    )
    assert len(replacement_query.edits) == 1
    assert replacement_query.edits[0]["text"] != WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=fresh_user.telegram_id,
        chat_id=9112,
        expected_access_version=fresh_user.access_version,
    )
    assert current.session is not None
    assert current.session.public_id == replacement.session.public_id
    assert current.session.version == replacement.session.version


@pytest.mark.asyncio
async def test_scheduled_notification_has_opaque_buttons_and_creates_no_session(db, fake_ai):
    user = await weekly_user(db, 7107)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()

    await bot.weekly_review_scheduled_notification(telegram, user.telegram_id, user.timezone)

    assert len(telegram.sent) == 1
    assert isinstance(telegram.sent[0]["reply_markup"], ReplyKeyboardRemove)
    assert len(telegram.edits) == 1
    assert telegram.edits[0]["message_id"] == telegram.sent[0]["message"].message_id
    markup = telegram.edits[0]["reply_markup"]
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert len(callbacks) == 4
    assert all(re.fullmatch(r"wrev:[A-Za-z0-9_-]+", value or "") for value in callbacks)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0


@pytest.mark.asyncio
async def test_sessionless_callback_preserves_active_weekly_generation_and_token(db, fake_ai):
    user = await weekly_user(db, 7140)
    bot = WeeklyHarness(db, fake_ai)
    active_message = WeeklyMessage(message_id=121_140, chat_id=9140, is_bot=True)
    active = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9140,
        expected_access_version=user.access_version,
        canonical_message_id=active_message.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert active.session is not None
    launch_message = WeeklyMessage(message_id=121_141, chat_id=9140, is_bot=True)
    tokens = await bot.weekly_review_capabilities.issue(
        actions=("start",),
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9140,
        canonical_message_id=launch_message.message_id,
        access_version=user.access_version,
        week_start=active.session.week_start,
    )
    token = tokens["start"]
    query = WeeklyQuery(f"wrev:{token}", launch_message)
    update = weekly_update(
        launch_message,
        telegram_user_id=user.telegram_id,
        chat_id=9140,
        query=query,
    )
    context = weekly_context()

    await bot.weekly_review_callback(update, context)
    capability_count = len(bot.weekly_review_capabilities._capabilities)
    await bot.weekly_review_callback(update, context)

    assert query.answers == [
        {"args": ("Обзор недели уже открыт.",), "show_alert": True},
        {"args": ("Обзор недели уже открыт.",), "show_alert": True},
    ]
    assert query.edits == []
    assert context.bot.edits == []
    assert len(bot.weekly_review_capabilities._capabilities) == capability_count == 1
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9140,
        expected_access_version=user.access_version,
    )
    assert current.session == active.session
    assert (
        await bot.weekly_review_capabilities.peek(
            token,
            telegram_user_id=user.telegram_id,
            chat_id=9140,
            canonical_message_id=launch_message.message_id,
        )
        is not None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.asyncio
async def test_concurrent_sessionless_callbacks_create_only_one_weekly_generation(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7141)
    bot = WeeklyHarness(db, fake_ai)
    week_start = bot.weekly_review_service.target_week_start(user.timezone)
    first_message = WeeklyMessage(message_id=121_142, chat_id=9141, is_bot=True)
    second_message = WeeklyMessage(message_id=121_143, chat_id=9141, is_bot=True)
    first_tokens = await bot.weekly_review_capabilities.issue(
        actions=("start",),
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9141,
        canonical_message_id=first_message.message_id,
        access_version=user.access_version,
        week_start=week_start,
    )
    second_tokens = await bot.weekly_review_capabilities.issue(
        actions=("start",),
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9141,
        canonical_message_id=second_message.message_id,
        access_version=user.access_version,
        week_start=week_start,
    )
    first_token = first_tokens["start"]
    second_token = second_tokens["start"]
    first_query = WeeklyQuery(f"wrev:{first_token}", first_message)
    second_query = WeeklyQuery(f"wrev:{second_token}", second_message)
    first_update = weekly_update(
        first_message,
        telegram_user_id=user.telegram_id,
        chat_id=9141,
        query=first_query,
    )
    second_update = weekly_update(
        second_message,
        telegram_user_id=user.telegram_id,
        chat_id=9141,
        query=second_query,
    )
    original_create = bot.weekly_review_service.create_session
    first_created = asyncio.Event()
    release_first = asyncio.Event()
    create_calls = 0

    async def blocked_first_create(**kwargs: Any):
        nonlocal create_calls
        create_calls += 1
        result = await original_create(**kwargs)
        if create_calls == 1:
            first_created.set()
            await release_first.wait()
        return result

    monkeypatch.setattr(bot.weekly_review_service, "create_session", blocked_first_create)
    context = weekly_context()
    first_task = asyncio.create_task(bot.weekly_review_callback(first_update, context))
    await asyncio.wait_for(first_created.wait(), timeout=1)
    second_task = asyncio.create_task(bot.weekly_review_callback(second_update, context))
    await asyncio.sleep(0)
    assert second_query.answers == []
    release_first.set()
    await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=2)

    assert create_calls == 1
    assert first_query.answers == [{"args": ()}]
    assert second_query.answers == [{"args": ("Обзор недели уже открыт.",), "show_alert": True}]
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9141,
        expected_access_version=user.access_version,
    )
    assert current.session is not None
    assert current.session.canonical_message_id == first_message.message_id
    assert (
        await bot.weekly_review_capabilities.peek(
            first_token,
            telegram_user_id=user.telegram_id,
            chat_id=9141,
            canonical_message_id=first_message.message_id,
        )
        is None
    )
    assert (
        await bot.weekly_review_capabilities.peek(
            second_token,
            telegram_user_id=user.telegram_id,
            chat_id=9141,
            canonical_message_id=second_message.message_id,
        )
        is not None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize(
    "replace_before_consume",
    (
        pytest.param(False, id="exact-clear-created"),
        pytest.param(True, id="preserve-concurrent-replacement"),
    ),
)
@pytest.mark.asyncio
async def test_sessionless_launch_expiring_during_answer_clears_only_created_exact(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    replace_before_consume,
):
    private = "PRIVATE_SESSIONLESS_TTL_RACE"
    suffix = int(replace_before_consume)
    user = await weekly_user(db, 7144 + suffix)
    bot = WeeklyHarness(db, fake_ai)
    chat_id = 9144 + suffix
    canonical = WeeklyMessage(
        message_id=121_146 + suffix,
        chat_id=chat_id,
        is_bot=True,
    )
    tokens = await bot.weekly_review_capabilities.issue(
        actions=("start",),
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
        access_version=user.access_version,
        week_start=bot.weekly_review_service.target_week_start(user.timezone),
    )
    token = tokens["start"]
    callback_data = f"wrev:{token}"
    claim = await bot.weekly_review_capabilities.peek(
        token,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        canonical_message_id=canonical.message_id,
    )
    assert claim is not None
    assert claim.session_public_id is None

    clock = {"now": claim.expires_at - timedelta(microseconds=1)}
    original_utc = bot.weekly_review_capabilities._utc

    def controlled_utc(value: datetime | None) -> datetime:
        return clock["now"] if value is None else original_utc(value)

    monkeypatch.setattr(bot.weekly_review_capabilities, "_utc", controlled_utc)
    consume_results: list[bool] = []
    transient_sessions: list[Any] = []
    replacements: list[Any] = []
    original_consume = bot.weekly_review_capabilities.consume

    async def race_consume(expected: Any, *, now: datetime | None = None) -> bool:
        assert expected == claim
        transient = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
        )
        assert transient.status == "found"
        assert transient.session is not None
        transient_sessions.append(transient.session)
        if replace_before_consume:
            replacement = await bot.weekly_review_service.create_session(
                telegram_actor_id=user.telegram_id,
                chat_id=chat_id,
                expected_access_version=user.access_version,
                target_week_start=claim.week_start,
                canonical_message_id=canonical.message_id,
                phase=WeeklyReviewPhase.ROOT,
            )
            assert replacement.status == "created"
            assert replacement.session is not None
            replacements.append(replacement.session)
        result = await original_consume(expected, now=now)
        consume_results.append(result)
        return result

    clear_attempts: list[dict[str, Any]] = []
    clear_results: list[bool] = []
    original_clear = bot.weekly_review_service.clear_session_exact

    async def tracked_clear(**kwargs: Any) -> bool:
        clear_attempts.append(dict(kwargs))
        result = await original_clear(**kwargs)
        clear_results.append(result)
        return result

    monkeypatch.setattr(bot.weekly_review_capabilities, "consume", race_consume)
    monkeypatch.setattr(bot.weekly_review_service, "clear_session_exact", tracked_clear)
    query = WeeklyQuery(callback_data, canonical)
    answer_started = asyncio.Event()
    answer_release = asyncio.Event()

    async def blocked_answer(*args: Any, **kwargs: Any) -> None:
        query.answers.append({"args": args, **kwargs})
        answer_started.set()
        await answer_release.wait()

    monkeypatch.setattr(query, "answer", blocked_answer)
    context = weekly_context()
    callback = asyncio.create_task(
        bot.weekly_review_callback(
            weekly_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                query=query,
            ),
            context,
        )
    )
    try:
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(answer_started.wait(), timeout=10)
            clock["now"] = claim.expires_at + timedelta(microseconds=1)
            answer_release.set()
            await asyncio.wait_for(callback, timeout=10)
    finally:
        answer_release.set()
        if not callback.done():
            callback.cancel()
        await asyncio.gather(callback, return_exceptions=True)
    await asyncio.sleep(0)

    assert query.answers == [{"args": ()}]
    assert consume_results == [False]
    assert len(transient_sessions) == 1
    transient = transient_sessions[0]
    assert transient.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert clear_attempts == [
        {
            "telegram_actor_id": user.telegram_id,
            "chat_id": chat_id,
            "session_public_id": transient.public_id,
            "expected_session_version": transient.version,
        }
    ]
    assert clear_results == [not replace_before_consume]
    assert len(query.edits) == 1
    assert query.edits[0]["reply_markup"] is not None
    assert (
        await bot.weekly_review_capabilities.peek(
            token,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            canonical_message_id=canonical.message_id,
        )
        is None
    )

    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    if replace_before_consume:
        assert len(replacements) == 1
        assert current.session == replacements[0]
        assert current.session.phase is WeeklyReviewPhase.ROOT
    else:
        assert replacements == []
        assert current.status == "not_found"
        assert current.session is None

    fresh_callbacks = tuple(
        str(button.callback_data)
        for row in query.edits[0]["reply_markup"].inline_keyboard
        for button in row
        if str(button.callback_data).startswith("wrev:")
    )
    assert fresh_callbacks
    fresh_claims = []
    for fresh_callback in fresh_callbacks:
        fresh_claim = await bot.weekly_review_capabilities.peek(
            fresh_callback.removeprefix("wrev:"),
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            canonical_message_id=canonical.message_id,
        )
        assert fresh_claim is not None
        fresh_claims.append(fresh_claim)
    if replace_before_consume:
        assert all(item.session_public_id == replacements[0].public_id for item in fresh_claims)
    else:
        assert all(item.session_public_id is None for item in fresh_claims)

    async with db.sessions() as db_session:
        sessions_before_replay = await db_session.scalar(select(func.count(WeeklyReviewSession.id)))
    replay = WeeklyQuery(callback_data, canonical)
    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=replay,
        ),
        context,
    )
    await asyncio.sleep(0)
    async with db.sessions() as db_session:
        assert (
            await db_session.scalar(select(func.count(WeeklyReviewSession.id)))
            == sessions_before_replay
        )
    assert replay.answers == [{"args": (WEEKLY_REVIEW_STALE_TEXT,), "show_alert": True}]
    assert replay.edits == []

    monkeypatch.setattr(bot.weekly_review_capabilities, "consume", original_consume)
    start_callback = next(
        fresh_callback
        for fresh_callback, fresh_claim in zip(fresh_callbacks, fresh_claims, strict=True)
        if fresh_claim.action == "start"
    )
    fresh_query = WeeklyQuery(start_callback, canonical)
    await bot.weekly_review_callback(
        weekly_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=fresh_query,
        ),
        context,
    )
    await asyncio.sleep(0)
    assert fresh_query.answers == [{"args": ()}]
    assert len(fresh_query.edits) == 1
    live = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
    )
    assert live.status == "found"
    assert live.session is not None
    assert live.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    if replace_before_consume:
        assert live.session.public_id == replacements[0].public_id
    await assert_live_session_controls(bot, fresh_query, live.session)

    assert canonical.replies == []
    assert context.bot.sent == []
    assert fake_ai.weekly_review_calls == []
    assert bot.reminder_handoffs == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(WeeklyReviewSession.id))) == 1
        assert await db_session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await db_session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
    assert private not in caplog.text
    assert callback_data not in caplog.text
    assert str(user.telegram_id) not in caplog.text


@pytest.mark.asyncio
async def test_failed_rerender_revokes_staged_controls_and_keeps_visible_screen(db, fake_ai):
    user = await weekly_user(db, 7142)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_144, chat_id=9142, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9142,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    visible_context = weekly_context()
    assert await bot._weekly_review_render_current(visible_context, created.session) is True
    visible_markup = visible_context.bot.edits[-1]["reply_markup"]
    visible_token = visible_markup.inline_keyboard[0][0].callback_data.removeprefix("wrev:")

    failed_context = weekly_context(WeeklyBot(edit_error=BadRequest("synthetic failure")))
    assert await bot._weekly_review_render_current(failed_context, created.session) is False

    assert (
        await bot.weekly_review_capabilities.peek(
            visible_token,
            telegram_user_id=user.telegram_id,
            chat_id=9142,
            canonical_message_id=canonical.message_id,
        )
        is not None
    )
    assert len(bot.weekly_review_capabilities._capabilities) == 1
    assert len(bot.weekly_review_capabilities._screens) == 1


@pytest.mark.asyncio
async def test_repeated_successful_rerender_keeps_only_latest_screen(db, fake_ai):
    user = await weekly_user(db, 7143)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_145, chat_id=9143, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9143,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    context = weekly_context()
    assert await bot._weekly_review_render_current(context, created.session) is True
    old_markup = context.bot.edits[-1]["reply_markup"]
    old_token = old_markup.inline_keyboard[0][0].callback_data.removeprefix("wrev:")

    assert await bot._weekly_review_render_current(context, created.session) is True
    latest_markup = context.bot.edits[-1]["reply_markup"]
    latest_token = latest_markup.inline_keyboard[0][0].callback_data.removeprefix("wrev:")

    assert (
        await bot.weekly_review_capabilities.peek(
            old_token,
            telegram_user_id=user.telegram_id,
            chat_id=9143,
            canonical_message_id=canonical.message_id,
        )
        is None
    )
    assert (
        await bot.weekly_review_capabilities.peek(
            latest_token,
            telegram_user_id=user.telegram_id,
            chat_id=9143,
            canonical_message_id=canonical.message_id,
        )
        is not None
    )
    assert len(bot.weekly_review_capabilities._capabilities) == 1
    assert len(bot.weekly_review_capabilities._screens) == 1


@pytest.mark.asyncio
async def test_scheduled_notification_skips_keyboard_owner_and_edit_failure_has_no_fallback(
    db,
    fake_ai,
    caplog,
):
    owner = await weekly_user(db, 7108)
    bot = WeeklyHarness(db, fake_ai)
    bot.reply_keyboard_owned = True
    skipped = WeeklyBot()
    await bot.weekly_review_scheduled_notification(skipped, owner.telegram_id, owner.timezone)
    assert skipped.sent == []

    bot.reply_keyboard_owned = False
    private_error = "synthetic edit failure private-weekly-content"
    failing = WeeklyBot(edit_error=BadRequest(private_error))
    await bot.weekly_review_scheduled_notification(failing, owner.telegram_id, owner.timezone)
    assert len(failing.sent) == 1
    assert len(failing.edits) == 1
    assert failing.deleted == [
        {
            "chat_id": owner.telegram_id,
            "message_id": failing.sent[0]["message"].message_id,
        }
    ]
    assert private_error not in caplog.text
    assert "operation=edit" in caplog.text
    assert "error_type=BadRequest" in caplog.text


@pytest.mark.asyncio
async def test_scheduled_post_send_access_loss_removes_exact_message(db, fake_ai):
    user = await weekly_user(db, 7109)
    bot = WeeklyHarness(db, fake_ai)

    class DowngradingBot(WeeklyBot):
        async def send_message(self, **kwargs: Any) -> WeeklyMessage:
            sent = await super().send_message(**kwargs)
            await bot.access_service.set_guest(user.telegram_id, source="test")
            return sent

    telegram = DowngradingBot()
    await bot.weekly_review_scheduled_notification(telegram, user.telegram_id, user.timezone)

    assert len(telegram.sent) == 1
    assert telegram.edits == []
    assert telegram.deleted == [
        {
            "chat_id": user.telegram_id,
            "message_id": telegram.sent[0]["message"].message_id,
        }
    ]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0


@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_scheduled_post_edit_access_change_removes_exact_message(
    db,
    fake_ai,
    access_change,
):
    user = await weekly_user(db, 7116)
    bot = WeeklyHarness(db, fake_ai)
    telegram = BlockingWeeklyBot()
    delivery = asyncio.create_task(
        bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    )
    await asyncio.wait_for(telegram.started.wait(), timeout=1)
    assert len(telegram.sent) == 1
    assert len(telegram.edits) == 1
    await bot.access_service.set_guest(user.telegram_id, source="test")
    if access_change == "bounce":
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    telegram.release.set()
    await asyncio.wait_for(delivery, timeout=1)

    assert telegram.deleted == [
        {
            "chat_id": user.telegram_id,
            "message_id": telegram.sent[0]["message"].message_id,
        }
    ]
    assert len(telegram.edits) == 1
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0


@pytest.mark.parametrize("failed_lookup", ["access", "session"])
@pytest.mark.asyncio
async def test_active_text_lookup_failure_is_consumed_and_neutralizes_frozen_canonical(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    failed_lookup,
):
    user = await weekly_user(db, 7120)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_020, chat_id=9120, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9120,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    update = weekly_update(canonical, telegram_user_id=user.telegram_id, chat_id=9120)
    context = weekly_context()
    secret = "private-weekly-lookup-payload"
    provider_calls = 0

    async def forbidden_extract(*args: Any, **kwargs: Any) -> WeeklyReviewExtraction:
        nonlocal provider_calls
        del args, kwargs
        provider_calls += 1
        raise AssertionError("provider must not run")

    monkeypatch.setattr(
        "future_self.weekly_review_handlers.extract_weekly_review_input",
        forbidden_extract,
    )
    if failed_lookup == "access":
        original_status = bot.access_service.status
        access_calls = 0

        async def flaky_status(telegram_id: int):
            nonlocal access_calls
            access_calls += 1
            if access_calls == 2:
                raise RuntimeError(secret)
            return await original_status(telegram_id)

        monkeypatch.setattr(bot.access_service, "status", flaky_status)
    else:
        original_current = bot.weekly_review_service.current_session
        current_calls = 0

        async def flaky_current(**kwargs: Any):
            nonlocal current_calls
            current_calls += 1
            if current_calls == 2:
                raise RuntimeError(secret)
            return await original_current(**kwargs)

        monkeypatch.setattr(bot.weekly_review_service, "current_session", flaky_current)

    assert await bot.weekly_review_active_text_gate(update, context) is True

    assert provider_calls == 0
    assert context.bot.edits[-1] == {
        "chat_id": 9120,
        "message_id": canonical.message_id,
        "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
        "reply_markup": None,
        "parse_mode": None,
    }
    assert secret not in caplog.text
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_post_stt_session_lookup_failure_consumes_voice_and_neutralizes_frozen_canonical(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7121)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_021, chat_id=9121, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9121,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    progress = WeeklyMessage(message_id=121_121, chat_id=9121, is_bot=True)
    update = weekly_update(canonical, telegram_user_id=user.telegram_id, chat_id=9121)
    context = weekly_context()

    async def failed_current(**kwargs: Any):
        del kwargs
        raise RuntimeError("private-transcript-must-not-be-logged")

    monkeypatch.setattr(bot.weekly_review_service, "current_session", failed_current)

    assert (
        await bot.weekly_review_voice_pre_route(
            update,
            context,
            progress,
            expected_user=user,
            expected_session=created.session,
        )
        is True
    )

    assert progress.deleted == 1
    assert context.bot.edits[-1]["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
    assert context.bot.edits[-1]["reply_markup"] is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.parametrize("failed_lookup", ["access", "exact"])
@pytest.mark.asyncio
async def test_post_provider_lookup_failure_neutralizes_without_store_or_domain_dml(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    failed_lookup,
):
    user = await weekly_user(db, 7122)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_022, chat_id=9122, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9122,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    update = weekly_update(canonical, telegram_user_id=user.telegram_id, chat_id=9122)
    context = weekly_context()
    provider_calls = 0
    store_calls = 0
    secret = "private-post-provider-fence"

    async def extract(*args: Any, **kwargs: Any) -> WeeklyReviewExtraction:
        nonlocal provider_calls
        del args, kwargs
        provider_calls += 1
        return WeeklyReviewExtraction(
            focus="Keep one calm weekly focus",
            approach="Choose one small next step",
            small_steps=["Review the short task list"],
        )

    async def forbidden_store(**kwargs: Any):
        nonlocal store_calls
        del kwargs
        store_calls += 1
        raise AssertionError("store must not run")

    monkeypatch.setattr(
        "future_self.weekly_review_handlers.extract_weekly_review_input",
        extract,
    )
    monkeypatch.setattr(bot.weekly_review_service, "store_extraction", forbidden_store)
    if failed_lookup == "access":
        original_status = bot.access_service.status
        access_calls = 0

        async def flaky_status(telegram_id: int):
            nonlocal access_calls
            access_calls += 1
            if access_calls == 3:
                raise RuntimeError(secret)
            return await original_status(telegram_id)

        monkeypatch.setattr(bot.access_service, "status", flaky_status)
    else:
        original_exact = bot.weekly_review_service.get_session_exact
        exact_calls = 0

        async def flaky_exact(**kwargs: Any):
            nonlocal exact_calls
            exact_calls += 1
            if exact_calls == 3:
                raise RuntimeError(secret)
            return await original_exact(**kwargs)

        monkeypatch.setattr(bot.weekly_review_service, "get_session_exact", flaky_exact)

    await bot._weekly_review_process_input_lifecycle(
        update,
        context,
        created.session,
        "private weekly answer",
        source="text",
    )

    assert provider_calls == 1
    assert store_calls == 0
    assert context.bot.edits[-1]["text"] == WEEKLY_REVIEW_ACCESS_CHANGED_TEXT
    assert context.bot.edits[-1]["reply_markup"] is None
    assert secret not in caplog.text
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.parametrize("failed_lookup", ["access", "flow", "exact"])
@pytest.mark.asyncio
async def test_callback_lookup_failure_answers_once_neutralizes_and_writes_no_domain_rows(
    db,
    fake_ai,
    monkeypatch,
    failed_lookup,
):
    user = await weekly_user(db, 7123)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_023, chat_id=9123, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9123,
        canonical_message_id=canonical.message_id,
    )
    callback_data = await issue_session_action(bot, preview, "save")
    query = WeeklyQuery(callback_data, canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9123,
        query=query,
    )
    if failed_lookup == "access":

        async def failed_status(telegram_id: int):
            del telegram_id
            raise RuntimeError("private-callback-access")

        monkeypatch.setattr(bot.access_service, "status", failed_status)
    elif failed_lookup == "flow":

        async def failed_flow(update: Any, context: Any):
            del update, context
            raise RuntimeError("private-callback-flow")

        monkeypatch.setattr(bot, "_active_navigation_flow", failed_flow)
    else:

        async def failed_exact(**kwargs: Any):
            del kwargs
            raise RuntimeError("private-callback-exact")

        monkeypatch.setattr(bot.weekly_review_service, "get_session_exact", failed_exact)

    await bot.weekly_review_callback(update, weekly_context())

    assert query.answers == [{"args": (WEEKLY_REVIEW_STALE_TEXT,), "show_alert": True}]
    if failed_lookup == "exact":
        assert len(query.edits) == 1
        assert query.edits[0]["text"] == WEEKLY_REVIEW_RECOVERY_TEXT
        assert query.edits[0]["reply_markup"] is not None
    else:
        assert query.edits == [
            {
                "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                "reply_markup": None,
                "parse_mode": None,
            }
        ]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_lookup_cancelled_error_propagates_without_callback_answer_or_edit(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7124)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_024, chat_id=9124, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9124,
        canonical_message_id=canonical.message_id,
    )
    callback_data = await issue_session_action(bot, preview, "save")
    query = WeeklyQuery(callback_data, canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9124,
        query=query,
    )

    async def cancelled_status(telegram_id: int):
        del telegram_id
        raise asyncio.CancelledError

    monkeypatch.setattr(bot.access_service, "status", cancelled_status)

    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_callback(update, weekly_context())
    assert query.answers == []
    assert query.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_manual_open_rechecks_access_and_unbound_generation_before_first_send(
    db,
    fake_ai,
    monkeypatch,
    access_change,
):
    user = await weekly_user(db, 7125)
    bot = WeeklyHarness(db, fake_ai)
    incoming = WeeklyMessage("/week", chat_id=9125)
    update = weekly_update(incoming, telegram_user_id=user.telegram_id, chat_id=9125)
    context = weekly_context()
    snapshot_ready = asyncio.Event()
    snapshot_release = asyncio.Event()
    original_root_text = bot._weekly_review_root_text

    async def blocked_root_text(session: Any, actor: User):
        text = await original_root_text(session, actor)
        snapshot_ready.set()
        await snapshot_release.wait()
        return text

    monkeypatch.setattr(bot, "_weekly_review_root_text", blocked_root_text)
    opening = asyncio.create_task(bot._weekly_review_open(update, context))
    await asyncio.wait_for(snapshot_ready.wait(), timeout=1)
    await bot.access_service.set_guest(user.telegram_id, source="test")
    if access_change == "bounce":
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    snapshot_release.set()
    await asyncio.wait_for(opening, timeout=1)

    assert incoming.replies == []
    assert context.bot.sent == []
    assert context.bot.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0


@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_manual_open_bind_failure_clears_unbound_and_neutralizes_accepted_send(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    access_change,
):
    user = await weekly_user(db, 7134)
    bot = WeeklyHarness(db, fake_ai)
    incoming = WeeklyMessage("/week", chat_id=9134)
    update = weekly_update(incoming, telegram_user_id=user.telegram_id, chat_id=9134)
    context = weekly_context()
    bind_started = asyncio.Event()
    bind_release = asyncio.Event()
    private_error = "private-bind-failure-weekly-content"

    async def failed_bind(**kwargs: Any):
        del kwargs
        bind_started.set()
        await bind_release.wait()
        raise RuntimeError(private_error)

    monkeypatch.setattr(bot.weekly_review_service, "bind_canonical", failed_bind)
    opening = asyncio.create_task(bot._weekly_review_open(update, context))
    await asyncio.wait_for(bind_started.wait(), timeout=1)
    assert len(incoming.replies) == 1
    sent = incoming.replies[0]["message"]

    await bot.access_service.set_guest(user.telegram_id, source="test")
    if access_change == "bounce":
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    bind_release.set()
    await asyncio.wait_for(opening, timeout=1)

    assert len(incoming.replies) == 1
    assert context.bot.sent == []
    assert context.bot.deleted == []
    assert context.bot.edits == [
        {
            "chat_id": 9134,
            "message_id": sent.message_id,
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
        }
    ]
    assert bot._weekly_review_tasks == set()
    assert private_error not in caplog.text
    assert "operation=bind_canonical" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.asyncio
async def test_manual_open_bind_cancellation_still_propagates_without_compensation(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7135)
    bot = WeeklyHarness(db, fake_ai)
    incoming = WeeklyMessage("/week", chat_id=9135)
    update = weekly_update(incoming, telegram_user_id=user.telegram_id, chat_id=9135)
    context = weekly_context()

    async def cancelled_bind(**kwargs: Any):
        del kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(bot.weekly_review_service, "bind_canonical", cancelled_bind)
    with pytest.raises(asyncio.CancelledError):
        await bot._weekly_review_open(update, context)
    await asyncio.sleep(0)

    assert len(incoming.replies) == 1
    assert context.bot.edits == []
    assert context.bot.deleted == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as session:
        durable = await session.scalar(select(WeeklyReviewSession))
        assert durable is not None
        assert durable.canonical_message_id is None


@pytest.mark.parametrize("failed_lookup", ["access", "exact"])
@pytest.mark.asyncio
async def test_manual_open_lookup_error_before_first_send_is_fail_closed(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    failed_lookup,
):
    user = await weekly_user(db, 7126)
    bot = WeeklyHarness(db, fake_ai)
    incoming = WeeklyMessage("/week", chat_id=9126)
    update = weekly_update(incoming, telegram_user_id=user.telegram_id, chat_id=9126)
    secret = "private-manual-pre-send"
    if failed_lookup == "access":
        original_status = bot.access_service.status
        status_calls = 0

        async def flaky_status(telegram_id: int):
            nonlocal status_calls
            status_calls += 1
            if status_calls == 2:
                raise RuntimeError(secret)
            return await original_status(telegram_id)

        monkeypatch.setattr(bot.access_service, "status", flaky_status)
    else:

        async def failed_exact(**kwargs: Any):
            del kwargs
            raise RuntimeError(secret)

        monkeypatch.setattr(bot.weekly_review_service, "get_session_exact", failed_exact)

    await bot._weekly_review_open(update, weekly_context())

    assert incoming.replies == []
    assert secret not in caplog.text
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0


@pytest.mark.asyncio
async def test_show_focus_rechecks_frozen_access_after_lookup_before_first_send(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7127)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_027, chat_id=9127, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9127,
        canonical_message_id=canonical.message_id,
    )
    confirmed = await bot.weekly_review_service.confirm_focus(
        telegram_actor_id=user.telegram_id,
        chat_id=9127,
        expected_access_version=user.access_version,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=canonical.message_id,
        expected_week_start=preview.week_start,
    )
    assert confirmed.status == "created"
    incoming = WeeklyMessage("show focus", chat_id=9127)
    update = weekly_update(incoming, telegram_user_id=user.telegram_id, chat_id=9127)
    lookup_ready = asyncio.Event()
    lookup_release = asyncio.Event()
    original_get_focus = bot.weekly_review_service.get_focus

    async def blocked_get_focus(**kwargs: Any):
        result = await original_get_focus(**kwargs)
        lookup_ready.set()
        await lookup_release.wait()
        return result

    monkeypatch.setattr(bot.weekly_review_service, "get_focus", blocked_get_focus)
    showing = asyncio.create_task(bot._weekly_review_show_focus(update, weekly_context()))
    await asyncio.wait_for(lookup_ready.wait(), timeout=1)
    await bot.access_service.set_guest(user.telegram_id, source="test")
    await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    lookup_release.set()
    await asyncio.wait_for(showing, timeout=1)

    assert incoming.replies == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1


@pytest.mark.parametrize("access_change", ["downgrade", "bounce", "lookup_error"])
@pytest.mark.asyncio
async def test_scheduled_notification_rechecks_access_after_keyboard_owner_before_send(
    db,
    fake_ai,
    monkeypatch,
    access_change,
):
    user = await weekly_user(db, 7128)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()
    owner_checked = asyncio.Event()
    owner_release = asyncio.Event()

    async def blocked_owner(actor: User) -> bool:
        assert actor.id == user.id
        owner_checked.set()
        await owner_release.wait()
        return False

    monkeypatch.setattr(bot, "_weekly_review_has_reply_keyboard_owner", blocked_owner)
    if access_change == "lookup_error":
        original_status = bot.access_service.status
        status_calls = 0

        async def flaky_status(telegram_id: int):
            nonlocal status_calls
            status_calls += 1
            if status_calls == 2:
                raise RuntimeError("private-scheduled-payload")
            return await original_status(telegram_id)

        monkeypatch.setattr(bot.access_service, "status", flaky_status)
    delivery = asyncio.create_task(
        bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    )
    await asyncio.wait_for(owner_checked.wait(), timeout=1)
    if access_change != "lookup_error":
        await bot.access_service.set_guest(user.telegram_id, source="test")
        if access_change == "bounce":
            await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    owner_release.set()
    await asyncio.wait_for(delivery, timeout=1)

    assert telegram.sent == []
    assert telegram.edits == []
    assert telegram.deleted == []
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_scheduled_notification_rechecks_keyboard_owner_immediately_before_send(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7136)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()
    second_owner_started = asyncio.Event()
    second_owner_release = asyncio.Event()
    owner_calls = 0

    async def changing_owner(actor: User) -> bool:
        nonlocal owner_calls
        assert actor.id == user.id
        owner_calls += 1
        if owner_calls == 1:
            return False
        second_owner_started.set()
        await second_owner_release.wait()
        return True

    async def unexpected_issue(**kwargs: Any):
        raise AssertionError(kwargs)

    monkeypatch.setattr(bot, "_weekly_review_has_reply_keyboard_owner", changing_owner)
    monkeypatch.setattr(bot.weekly_review_capabilities, "issue", unexpected_issue)
    delivery = asyncio.create_task(
        bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    )
    await asyncio.wait_for(second_owner_started.wait(), timeout=1)
    assert telegram.sent == []
    second_owner_release.set()
    await asyncio.wait_for(delivery, timeout=1)

    assert owner_calls == 2
    assert telegram.sent == []
    assert telegram.edits == []
    assert telegram.deleted == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0


@pytest.mark.asyncio
async def test_callback_outer_cancellation_keeps_access_cleanup_and_neutralization_alive(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7129)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_029, chat_id=9129, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9129,
        canonical_message_id=canonical.message_id,
    )
    callback_data = await issue_session_action(bot, preview, "save")
    query = WeeklyQuery(callback_data, canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9129,
        query=query,
    )
    clear_started = asyncio.Event()
    clear_release = asyncio.Event()
    original_clear = bot.weekly_review_service.clear_session_exact

    async def blocked_clear(**kwargs: Any):
        clear_started.set()
        await clear_release.wait()
        return await original_clear(**kwargs)

    monkeypatch.setattr(bot.weekly_review_service, "clear_session_exact", blocked_clear)
    await bot.access_service.set_guest(user.telegram_id, source="test")
    outer = asyncio.create_task(bot.weekly_review_callback(update, weekly_context()))
    await asyncio.wait_for(clear_started.wait(), timeout=1)
    assert query.answers == [{"args": ()}]

    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert len(bot._weekly_review_tasks) == 1
    inner = tuple(bot._weekly_review_tasks)
    clear_release.set()
    await asyncio.wait_for(asyncio.gather(*inner), timeout=1)
    await asyncio.sleep(0)

    assert bot._weekly_review_tasks == set()
    assert query.answers == [{"args": ()}]
    assert query.edits == [
        {
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.asyncio
async def test_callback_access_mismatch_clears_only_old_exact_and_preserves_replacement(
    db,
    fake_ai,
):
    user = await weekly_user(db, 7132)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_032, chat_id=9132, is_bot=True)
    old = await preview_session(
        bot,
        user,
        chat_id=9132,
        canonical_message_id=canonical.message_id,
    )
    callback_data = await issue_session_action(bot, old, "save")
    await bot.access_service.set_guest(user.telegram_id, source="test")
    await bot.access_service.grant_subscriber(user.telegram_id, source="test")
    fresh = await bot.access_service.status(user.telegram_id)
    assert fresh is not None
    replacement = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9132,
        expected_access_version=fresh.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.ROOT,
    )
    assert replacement.session is not None
    query = WeeklyQuery(callback_data, canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9132,
        query=query,
    )

    await bot.weekly_review_callback(update, weekly_context())

    assert query.answers == [{"args": ()}]
    assert query.edits[-1] == {
        "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
        "reply_markup": None,
        "parse_mode": None,
    }
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9132,
        expected_access_version=fresh.access_version,
    )
    assert current.session is not None
    assert current.session.public_id == replacement.session.public_id
    assert current.session.version == replacement.session.version
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_callback_access_change_during_exact_lookup_clears_and_neutralizes_old(
    db,
    fake_ai,
    monkeypatch,
    access_change,
):
    user = await weekly_user(db, 7133)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_033, chat_id=9133, is_bot=True)
    old = await preview_session(
        bot,
        user,
        chat_id=9133,
        canonical_message_id=canonical.message_id,
    )
    callback_data = await issue_session_action(bot, old, "save")
    query = WeeklyQuery(callback_data, canonical)
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9133,
        query=query,
    )
    exact_started = asyncio.Event()
    exact_release = asyncio.Event()
    original_exact = bot.weekly_review_service.get_session_exact

    async def blocked_exact(**kwargs: Any):
        exact_started.set()
        await exact_release.wait()
        return await original_exact(**kwargs)

    monkeypatch.setattr(bot.weekly_review_service, "get_session_exact", blocked_exact)
    callback = asyncio.create_task(bot.weekly_review_callback(update, weekly_context()))
    await asyncio.wait_for(exact_started.wait(), timeout=1)
    await bot.access_service.set_guest(user.telegram_id, source="test")
    replacement = None
    fresh = None
    if access_change == "bounce":
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")
        fresh = await bot.access_service.status(user.telegram_id)
        assert fresh is not None
        replacement = await bot.weekly_review_service.create_session(
            telegram_actor_id=user.telegram_id,
            chat_id=9133,
            expected_access_version=fresh.access_version,
            canonical_message_id=canonical.message_id,
            phase=WeeklyReviewPhase.ROOT,
        )
        assert replacement.session is not None
    exact_release.set()
    await asyncio.wait_for(callback, timeout=1)

    assert query.answers == [{"args": ()}]
    if replacement is None:
        assert query.edits == [
            {
                "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                "reply_markup": None,
                "parse_mode": None,
            }
        ]
    else:
        assert len(query.edits) == 1
        assert query.edits[0]["reply_markup"] is not None
    if replacement is None:
        async with db.sessions() as session:
            assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
    else:
        assert fresh is not None
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=9133,
            expected_access_version=fresh.access_version,
        )
        assert current.session is not None
        assert current.session.public_id == replacement.session.public_id
        assert current.session.version == replacement.session.version
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


@pytest.mark.asyncio
async def test_scheduled_far_week_during_focus_lookup_revokes_and_neutralizes(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7130)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()
    focus_started = asyncio.Event()
    focus_release = asyncio.Event()
    crossed = False
    original_target = bot.weekly_review_service.target_week_start
    original_get_focus = bot.weekly_review_service.get_focus

    def moving_target(*args: Any, **kwargs: Any):
        target = original_target(*args, **kwargs)
        return target + timedelta(days=14) if crossed else target

    async def blocked_focus(**kwargs: Any):
        focus_started.set()
        await focus_release.wait()
        return await original_get_focus(**kwargs)

    monkeypatch.setattr(bot.weekly_review_service, "target_week_start", moving_target)
    monkeypatch.setattr(bot.weekly_review_service, "get_focus", blocked_focus)
    delivery = asyncio.create_task(
        bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    )
    await asyncio.wait_for(focus_started.wait(), timeout=1)
    assert len(telegram.sent) == 1
    crossed = True
    focus_release.set()
    await asyncio.wait_for(delivery, timeout=1)

    assert telegram.edits == []
    assert telegram.deleted == [
        {
            "chat_id": user.telegram_id,
            "message_id": telegram.sent[0]["message"].message_id,
        }
    ]
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_show_focus_week_boundary_during_edit_neutralizes_exact_sent_message(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7131)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_031, chat_id=9131, is_bot=True)
    preview = await preview_session(
        bot,
        user,
        chat_id=9131,
        canonical_message_id=canonical.message_id,
    )
    confirmed = await bot.weekly_review_service.confirm_focus(
        telegram_actor_id=user.telegram_id,
        chat_id=9131,
        expected_access_version=user.access_version,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=canonical.message_id,
        expected_week_start=preview.week_start,
    )
    assert confirmed.status == "created"
    incoming = WeeklyMessage("show focus", chat_id=9131)
    update = weekly_update(incoming, telegram_user_id=user.telegram_id, chat_id=9131)
    telegram = BlockingWeeklyBot()
    context = weekly_context(telegram)
    crossed = False
    original_target = bot.weekly_review_service.target_week_start

    def moving_target(*args: Any, **kwargs: Any):
        target = original_target(*args, **kwargs)
        return target + timedelta(days=7) if crossed else target

    monkeypatch.setattr(bot.weekly_review_service, "target_week_start", moving_target)
    showing = asyncio.create_task(bot._weekly_review_show_focus(update, context))
    await asyncio.wait_for(telegram.started.wait(), timeout=1)
    assert len(incoming.replies) == 1
    crossed = True
    telegram.release.set()
    await asyncio.wait_for(showing, timeout=1)

    sent = incoming.replies[0]["message"]
    assert telegram.deleted == [{"chat_id": 9131, "message_id": sent.message_id}]
    assert len(telegram.edits) == 1
    assert telegram.edits[0]["message_id"] == sent.message_id
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1


@pytest.mark.asyncio
async def test_voice_pre_route_outer_cancel_keeps_retired_session_cleanup_alive(
    db,
    fake_ai,
):
    user = await weekly_user(db, 7132)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_032, chat_id=9132, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9132,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    progress = WeeklyMessage(message_id=122_032, chat_id=9132, is_bot=True)
    update = weekly_update(canonical, telegram_user_id=user.telegram_id, chat_id=9132)
    telegram = BlockingWeeklyBot()
    context = weekly_context(telegram)
    await bot.access_service.set_guest(user.telegram_id, source="test")

    outer = asyncio.create_task(
        bot.weekly_review_voice_pre_route(
            update,
            context,
            progress,
            expected_user=user,
            expected_session=created.session,
        )
    )
    await asyncio.wait_for(telegram.started.wait(), timeout=1)
    inner = next(
        task
        for task in bot._weekly_review_tasks
        if task.get_name() == "weekly-review-voice-pre-route-lifecycle"
    )
    assert progress.deleted == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0

    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert not inner.done()
    assert inner in bot._weekly_review_tasks

    telegram.release.set()
    assert await asyncio.wait_for(inner, timeout=1) is True
    await asyncio.sleep(0)

    assert bot._weekly_review_tasks == set()
    assert telegram.edits == [
        {
            "chat_id": 9132,
            "message_id": canonical.message_id,
            "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_voice_pre_route_direct_transient_delete_cancel_is_not_cleanup_success(
    db,
    fake_ai,
):
    class CancellingProgress(WeeklyMessage):
        def __init__(self) -> None:
            super().__init__(message_id=122_033, chat_id=9133, is_bot=True)
            self.delete_attempts = 0

        async def delete(self) -> None:
            self.delete_attempts += 1
            raise asyncio.CancelledError

    user = await weekly_user(db, 7133)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=121_033, chat_id=9133, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9133,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    progress = CancellingProgress()
    update = weekly_update(canonical, telegram_user_id=user.telegram_id, chat_id=9133)
    telegram = WeeklyBot()
    await bot.access_service.set_guest(user.telegram_id, source="test")

    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_voice_pre_route(
            update,
            weekly_context(telegram),
            progress,
            expected_user=user,
            expected_session=created.session,
        )
    await asyncio.sleep(0)

    assert progress.delete_attempts == 1
    assert telegram.edits == []
    assert bot._weekly_review_tasks == set()
    async with db.sessions() as session:
        stored = await session.scalar(
            select(WeeklyReviewSession).where(
                WeeklyReviewSession.public_id == created.session.public_id
            )
        )
        assert stored is not None
        assert stored.version == created.session.version
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.parametrize("state_change", ["unchanged", "replacement", "cancel", "bounce"])
@pytest.mark.asyncio
async def test_voice_cancel_cleanup_retires_progress_and_fences_frozen_generation(
    db,
    fake_ai,
    state_change,
):
    telegram_user_id = {
        "unchanged": 7134,
        "replacement": 7135,
        "cancel": 7136,
        "bounce": 7137,
    }[state_change]
    chat_id = telegram_user_id + 2000
    user = await weekly_user(db, telegram_user_id)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(
        message_id=121_034 + telegram_user_id,
        chat_id=chat_id,
        is_bot=True,
    )
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=chat_id,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    progress = WeeklyMessage(
        message_id=122_034 + telegram_user_id,
        chat_id=chat_id,
        is_bot=True,
    )
    update = weekly_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
    )
    telegram = WeeklyBot()
    replacement = None
    if state_change == "replacement":
        replacement_result = await bot.weekly_review_service.create_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
            canonical_message_id=canonical.message_id,
            phase=WeeklyReviewPhase.ROOT,
        )
        replacement = replacement_result.session
        assert replacement is not None
    elif state_change == "cancel":
        assert await bot._weekly_review_clear_exact(created.session) is True
    elif state_change == "bounce":
        await bot.access_service.set_guest(user.telegram_id, source="test")
        await bot.access_service.grant_subscriber(user.telegram_id, source="test")

    bot.weekly_review_schedule_voice_cancel_cleanup(
        update,
        weekly_context(telegram),
        progress,
        expected_user=user,
        expected_session=created.session,
    )
    task = next(
        task
        for task in bot._weekly_review_tasks
        if task.get_name() == "weekly-review-voice-cancel-cleanup-lifecycle"
    )
    assert await asyncio.wait_for(task, timeout=1) is None
    await asyncio.sleep(0)

    assert progress.deleted == 1
    assert bot._weekly_review_tasks == set()
    if state_change == "bounce":
        assert telegram.edits == [
            {
                "chat_id": chat_id,
                "message_id": canonical.message_id,
                "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                "reply_markup": None,
                "parse_mode": None,
            }
        ]
        refreshed = await bot._weekly_review_access(update)
        assert refreshed is not None
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=refreshed.access_version,
        )
        assert current.session is None
    else:
        assert telegram.edits == []
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            expected_access_version=user.access_version,
        )
        if state_change == "unchanged":
            assert current.session == created.session
        elif state_change == "replacement":
            assert current.session == replacement
        else:
            assert state_change == "cancel"
            assert current.session is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_voice_cancel_cleanup_direct_telegram_cancel_is_observed(
    db,
    fake_ai,
):
    class CancellingCleanupProgress(WeeklyMessage):
        async def delete(self) -> None:
            raise asyncio.CancelledError

    user = await weekly_user(db, 7138)
    bot = WeeklyHarness(db, fake_ai)
    canonical = WeeklyMessage(message_id=128_138, chat_id=9138, is_bot=True)
    created = await bot.weekly_review_service.create_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9138,
        expected_access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    progress = CancellingCleanupProgress(message_id=129_138, chat_id=9138, is_bot=True)
    update = weekly_update(canonical, telegram_user_id=user.telegram_id, chat_id=9138)
    telegram = WeeklyBot()

    bot.weekly_review_schedule_voice_cancel_cleanup(
        update,
        weekly_context(telegram),
        progress,
        expected_user=user,
        expected_session=created.session,
    )
    task = next(
        task
        for task in bot._weekly_review_tasks
        if task.get_name() == "weekly-review-voice-cancel-cleanup-lifecycle"
    )
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert task.cancelled()
    assert bot._weekly_review_tasks == set()
    assert telegram.edits == []
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=9138,
        expected_access_version=user.access_version,
    )
    assert current.session == created.session


@pytest.mark.asyncio
async def test_voice_cancel_cleanup_schedule_failure_preserves_cancellation_path(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    secret = "PRIVATE_CANCELLED_TRANSCRIPT_7139"
    user = await weekly_user(db, 7139)
    bot = WeeklyHarness(db, fake_ai)
    message = WeeklyMessage(message_id=128_139, chat_id=9139)
    update = weekly_update(message, telegram_user_id=user.telegram_id, chat_id=9139)
    progress = WeeklyMessage(message_id=129_139, chat_id=9139, is_bot=True)
    captured: list[Any] = []
    task_names: list[str | None] = []

    def failed_create_task(coroutine: Any, *, name: str | None = None, **kwargs: Any):
        del kwargs
        captured.append(coroutine)
        task_names.append(name)
        raise RuntimeError(secret)

    monkeypatch.setattr(asyncio, "create_task", failed_create_task)

    bot.weekly_review_schedule_voice_cancel_cleanup(
        update,
        weekly_context(),
        progress,
        expected_user=user,
        expected_session=None,
    )

    assert task_names == ["weekly-review-voice-cancel-cleanup-lifecycle"]
    assert len(captured) == 1
    assert captured[0].cr_frame is None
    assert bot._weekly_review_tasks == set()
    assert progress.deleted == 0
    logs = [
        record.getMessage()
        for record in caplog.records
        if "Weekly review voice cleanup" in record.getMessage()
    ]
    assert logs == ["Weekly review voice cleanup failed operation=schedule error_type=RuntimeError"]
    assert secret not in caplog.text
    assert str(user.telegram_id) not in caplog.text


@pytest.mark.parametrize(
    "checkpoint",
    ["access", "owner", "get_focus", "issue", "edit", "final_fence"],
)
@pytest.mark.parametrize("cancellation", ["none", "direct", "outer"])
@pytest.mark.asyncio
async def test_scheduled_post_send_checkpoint_failure_is_compensated_and_observed(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    checkpoint,
    cancellation,
):
    private_error = f"PRIVATE_SCHEDULED_{checkpoint}_{cancellation}"
    user = await weekly_user(db, 7200 + len(checkpoint) * 10 + len(cancellation))
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()
    started = asyncio.Event()
    release = asyncio.Event()
    injected = False

    async def fail_checkpoint() -> None:
        nonlocal injected
        if injected:
            return
        injected = True
        started.set()
        await release.wait()
        if cancellation == "direct":
            raise asyncio.CancelledError
        raise RuntimeError(private_error)

    original_access = bot._weekly_review_access_values

    async def checkpoint_access(*args: Any, **kwargs: Any):
        if checkpoint == "access" and telegram.sent and not injected:
            await fail_checkpoint()
        if checkpoint == "final_fence" and telegram.edits and not injected:
            await fail_checkpoint()
        return await original_access(*args, **kwargs)

    monkeypatch.setattr(bot, "_weekly_review_access_values", checkpoint_access)
    original_owner = bot._weekly_review_has_reply_keyboard_owner

    async def checkpoint_owner(actor: User) -> bool:
        if checkpoint == "owner" and telegram.sent and not injected:
            await fail_checkpoint()
        return await original_owner(actor)

    monkeypatch.setattr(bot, "_weekly_review_has_reply_keyboard_owner", checkpoint_owner)
    original_focus = bot.weekly_review_service.get_focus

    async def checkpoint_focus(**kwargs: Any):
        if checkpoint == "get_focus" and not injected:
            await fail_checkpoint()
        return await original_focus(**kwargs)

    monkeypatch.setattr(bot.weekly_review_service, "get_focus", checkpoint_focus)
    original_issue = bot.weekly_review_capabilities.issue

    async def checkpoint_issue(**kwargs: Any):
        if checkpoint == "issue" and not injected:
            await fail_checkpoint()
        return await original_issue(**kwargs)

    monkeypatch.setattr(bot.weekly_review_capabilities, "issue", checkpoint_issue)
    original_edit = telegram.edit_message_text

    async def checkpoint_edit(**kwargs: Any) -> None:
        if checkpoint == "edit" and not injected:
            telegram.edits.append(kwargs)
            await fail_checkpoint()
            return
        await original_edit(**kwargs)

    monkeypatch.setattr(telegram, "edit_message_text", checkpoint_edit)
    delivery = asyncio.create_task(
        bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        assert len(telegram.sent) == 1
        inner = next(
            task
            for task in bot._weekly_review_tasks
            if task.get_name() == "weekly-review-scheduled-post-send-lifecycle"
        )
        if cancellation == "outer":
            delivery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(delivery, timeout=1)
            assert inner in bot._weekly_review_tasks
            assert not inner.done()
        release.set()
        if cancellation == "direct":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(delivery, timeout=10)
        elif cancellation == "none":
            await asyncio.wait_for(delivery, timeout=10)
        await asyncio.gather(*tuple(bot._weekly_review_tasks), return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        release.set()
        if not delivery.done():
            delivery.cancel()
        await asyncio.gather(delivery, *tuple(bot._weekly_review_tasks), return_exceptions=True)

    sent = telegram.sent[0]["message"]
    assert telegram.deleted == [{"chat_id": user.telegram_id, "message_id": sent.message_id}]
    assert len(telegram.sent) == 1
    assert bot.weekly_review_capabilities._capabilities == {}
    assert bot._weekly_review_tasks == set()
    assert fake_ai.weekly_review_calls == []
    assert private_error not in caplog.text
    assert str(user.telegram_id) not in caplog.text
    lifecycle_logs = [
        record.getMessage()
        for record in caplog.records
        if "Weekly review notification failed" in record.getMessage()
    ]
    assert lifecycle_logs
    assert all("operation=" in message and "error_type=" in message for message in lifecycle_logs)


@pytest.mark.parametrize("delete_failure", ["false", "error"])
@pytest.mark.asyncio
async def test_scheduled_post_send_delete_failure_neutralizes_exact_message(
    db,
    fake_ai,
    caplog,
    delete_failure,
):
    private_error = f"PRIVATE_SCHEDULED_DELETE_{delete_failure}"
    user = await weekly_user(db, 7290 if delete_failure == "false" else 7291)
    bot = WeeklyHarness(db, fake_ai)

    class CleanupFallbackBot(WeeklyBot):
        async def edit_message_text(self, **kwargs: Any) -> None:
            self.edits.append(kwargs)
            if len(self.edits) == 1:
                raise BadRequest(private_error)

        async def delete_message(self, **kwargs: Any) -> bool:
            self.deleted.append(kwargs)
            if delete_failure == "error":
                raise RuntimeError(private_error)
            return False

    telegram = CleanupFallbackBot()
    await bot.weekly_review_scheduled_notification(telegram, user.telegram_id, user.timezone)

    sent = telegram.sent[0]["message"]
    assert len(telegram.sent) == 1
    assert telegram.deleted == [{"chat_id": user.telegram_id, "message_id": sent.message_id}]
    assert len(telegram.edits) == 2
    assert telegram.edits[1] == {
        "chat_id": user.telegram_id,
        "message_id": sent.message_id,
        "text": WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
        "reply_markup": None,
        "parse_mode": None,
    }
    assert bot.weekly_review_capabilities._capabilities == {}
    assert bot._weekly_review_tasks == set()
    assert private_error not in caplog.text
    assert str(user.telegram_id) not in caplog.text


@pytest.mark.asyncio
async def test_scheduled_post_send_cleanup_preserves_newer_capability_generation(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7292)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()
    final_started = asyncio.Event()
    final_release = asyncio.Event()
    original_access = bot._weekly_review_access_values

    async def fail_final_fence(*args: Any, **kwargs: Any):
        if telegram.edits and not final_started.is_set():
            final_started.set()
            await final_release.wait()
            raise RuntimeError("PRIVATE_REPLACEMENT_FINAL_FENCE")
        return await original_access(*args, **kwargs)

    monkeypatch.setattr(bot, "_weekly_review_access_values", fail_final_fence)
    delivery = asyncio.create_task(
        bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    )
    try:
        await asyncio.wait_for(final_started.wait(), timeout=10)
        sent = telegram.sent[0]["message"]
        old_markup = telegram.edits[0]["reply_markup"]
        old_tokens = tuple(
            button.callback_data.removeprefix("wrev:")
            for row in old_markup.inline_keyboard
            for button in row
        )
        old_claim = await bot.weekly_review_capabilities.peek(
            old_tokens[0],
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            canonical_message_id=sent.message_id,
        )
        assert old_claim is not None
        replacement = await bot.weekly_review_capabilities.issue(
            actions=("replacement",),
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            canonical_message_id=sent.message_id,
            access_version=user.access_version,
            week_start=old_claim.week_start,
            scheduled=True,
        )
        final_release.set()
        await asyncio.wait_for(delivery, timeout=10)
        await asyncio.sleep(0)
    finally:
        final_release.set()
        if not delivery.done():
            delivery.cancel()
        await asyncio.gather(delivery, *tuple(bot._weekly_review_tasks), return_exceptions=True)

    assert telegram.deleted == []
    assert len(telegram.edits) == 1
    for token in old_tokens:
        assert (
            await bot.weekly_review_capabilities.peek(
                token,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                canonical_message_id=sent.message_id,
            )
            is None
        )
    assert (
        await bot.weekly_review_capabilities.peek(
            replacement["replacement"],
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            canonical_message_id=sent.message_id,
        )
        is not None
    )
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_consumed_newer_launch_tombstone_blocks_old_cleanup_before_replacement_render(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_CONSUMED_REPLACEMENT_RACE"
    user = await weekly_user(db, 7296)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()
    old_final_started = asyncio.Event()
    old_final_release = asyncio.Event()
    callback_ready = asyncio.Event()
    replacement_render_started = asyncio.Event()
    replacement_render_release = asyncio.Event()
    old_final_injected = False
    original_access = bot._weekly_review_access_values

    async def cancel_old_final_fence(*args: Any, **kwargs: Any):
        nonlocal old_final_injected
        if telegram.edits and not old_final_injected:
            old_final_injected = True
            old_final_started.set()
            await old_final_release.wait()
            raise asyncio.CancelledError(private)
        return await original_access(*args, **kwargs)

    async def callback_owner_ready(update: Any) -> bool:
        del update
        callback_ready.set()
        return False

    original_render = bot._weekly_review_render_launched

    async def block_replacement_render(*args: Any, **kwargs: Any):
        replacement_render_started.set()
        await replacement_render_release.wait()
        return await original_render(*args, **kwargs)

    monkeypatch.setattr(bot, "_weekly_review_access_values", cancel_old_final_fence)
    monkeypatch.setattr(bot, "reminder_blocks_navigation", callback_owner_ready)
    monkeypatch.setattr(bot, "_weekly_review_render_launched", block_replacement_render)
    old_delivery = asyncio.create_task(
        bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    )
    callback_delivery: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(old_final_started.wait(), timeout=10)
        sent = telegram.sent[0]["message"]
        old_markup = telegram.edits[0]["reply_markup"]
        old_tokens = tuple(
            button.callback_data.removeprefix("wrev:")
            for row in old_markup.inline_keyboard
            for button in row
        )
        old_claim = await bot.weekly_review_capabilities.peek(
            old_tokens[0],
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            canonical_message_id=sent.message_id,
        )
        assert old_claim is not None
        newer = await bot.weekly_review_capabilities.issue(
            actions=("start",),
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            canonical_message_id=sent.message_id,
            access_version=user.access_version,
            week_start=old_claim.week_start,
            scheduled=True,
        )
        query = WeeklyQuery(f"wrev:{newer['start']}", sent)
        update = weekly_update(
            sent,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=query,
        )
        callback_delivery = asyncio.create_task(
            bot.weekly_review_callback(update, weekly_context(telegram))
        )
        await asyncio.wait_for(callback_ready.wait(), timeout=10)
        await asyncio.sleep(0)

        old_final_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(old_delivery, timeout=10)
        await asyncio.wait_for(replacement_render_started.wait(), timeout=10)
        cleanup = next(
            (
                task
                for task in bot._weekly_review_tasks
                if task.get_name() == "weekly-review-scheduled-cleanup-lifecycle"
            ),
            None,
        )
        if cleanup is not None:
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=10)

        assert telegram.deleted == []
        assert len(telegram.edits) == 1
        for token in old_tokens:
            assert (
                await bot.weekly_review_capabilities.peek(
                    token,
                    telegram_user_id=user.telegram_id,
                    chat_id=user.telegram_id,
                    canonical_message_id=sent.message_id,
                )
                is None
            )
        assert (
            await bot.weekly_review_capabilities.peek(
                newer["start"],
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                canonical_message_id=sent.message_id,
            )
            is None
        )
        current = await bot.weekly_review_service.current_session(
            telegram_actor_id=user.telegram_id,
            chat_id=user.telegram_id,
            expected_access_version=user.access_version,
        )
        assert current.status == "found"
        assert current.session is not None
        assert current.session.canonical_message_id == sent.message_id

        replacement_render_release.set()
        await asyncio.wait_for(callback_delivery, timeout=10)
        await asyncio.sleep(0)
    finally:
        old_final_release.set()
        replacement_render_release.set()
        pending = [old_delivery]
        if callback_delivery is not None:
            pending.append(callback_delivery)
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *pending,
            *tuple(bot._weekly_review_tasks),
            return_exceptions=True,
        )

    assert query.answers == [{"args": ()}]
    assert query.edits
    replacement_markup = query.edits[-1]["reply_markup"]
    replacement_tokens = tuple(
        button.callback_data.removeprefix("wrev:")
        for row in replacement_markup.inline_keyboard
        for button in row
    )
    assert replacement_tokens
    assert (
        await bot.weekly_review_capabilities.peek(
            replacement_tokens[0],
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            canonical_message_id=sent.message_id,
        )
        is not None
    )
    assert telegram.deleted == []
    assert bot._weekly_review_tasks == set()
    assert private not in caplog.text
    assert str(user.telegram_id) not in caplog.text


@pytest.mark.asyncio
async def test_scheduled_primary_send_direct_cancellation_is_not_delivery(db, fake_ai):
    user = await weekly_user(db, 7293)
    bot = WeeklyHarness(db, fake_ai)

    class CancellingSendBot(WeeklyBot):
        def __init__(self) -> None:
            super().__init__()
            self.send_attempts = 0

        async def send_message(self, **kwargs: Any) -> WeeklyMessage:
            del kwargs
            self.send_attempts += 1
            raise asyncio.CancelledError

    telegram = CancellingSendBot()
    with pytest.raises(asyncio.CancelledError):
        await bot.weekly_review_scheduled_notification(
            telegram,
            user.telegram_id,
            user.timezone,
        )
    await asyncio.sleep(0)

    assert telegram.send_attempts == 1
    assert telegram.sent == []
    assert telegram.edits == []
    assert telegram.deleted == []
    assert bot.weekly_review_capabilities._capabilities == {}
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_scheduled_sunday_capability_stays_live_after_monday_boundary(
    db,
    fake_ai,
    monkeypatch,
):
    user = await weekly_user(db, 7294)
    bot = WeeklyHarness(db, fake_ai)
    sunday = datetime(2026, 8, 23, 20, 50, tzinfo=UTC)
    monday = datetime(2026, 8, 23, 21, 5, tzinfo=UTC)
    clock = {"now": sunday}
    original_target = bot.weekly_review_service.target_week_start

    def frozen_target(timezone_name: str, **kwargs: Any):
        return original_target(timezone_name, now=clock["now"], **kwargs)

    monkeypatch.setattr(bot.weekly_review_service, "target_week_start", frozen_target)
    monkeypatch.setattr(bot.weekly_review_service, "_clock", lambda: clock["now"])

    class MondayBoundaryBot(WeeklyBot):
        async def send_message(self, **kwargs: Any) -> WeeklyMessage:
            sent = await super().send_message(**kwargs)
            clock["now"] = monday
            return sent

    telegram = MondayBoundaryBot()
    await bot.weekly_review_scheduled_notification(telegram, user.telegram_id, user.timezone)

    assert len(telegram.sent) == 1
    assert len(telegram.edits) == 1
    sent = telegram.sent[0]["message"]
    start_token = (
        telegram.edits[0]["reply_markup"].inline_keyboard[0][0].callback_data.removeprefix("wrev:")
    )
    claim = await bot.weekly_review_capabilities.peek(
        start_token,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        canonical_message_id=sent.message_id,
    )
    assert claim is not None
    assert claim.scheduled is True
    assert claim.week_start.isoformat() == "2026-08-24"

    query = WeeklyQuery(f"wrev:{start_token}", sent)
    update = weekly_update(
        sent,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        query=query,
    )
    await bot.weekly_review_callback(update, weekly_context(telegram))

    assert query.answers == [{"args": ()}]
    current = await bot.weekly_review_service.current_session(
        telegram_actor_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        now=monday,
    )
    assert current.status == "found"
    assert current.session is not None
    assert current.session.week_start.isoformat() == "2026-08-24"
    assert bot._weekly_review_tasks == set()


@pytest.mark.asyncio
async def test_scheduled_post_send_direct_cancel_has_no_orphan_or_unawaited_warning(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_SCHEDULED_CANCEL_WARNING_SENTINEL"
    user = await weekly_user(db, 7295)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()

    async def cancelled_focus(**kwargs: Any):
        del kwargs
        raise asyncio.CancelledError(private)

    monkeypatch.setattr(bot.weekly_review_service, "get_focus", cancelled_focus)
    loop = asyncio.get_running_loop()
    prior_debug = loop.get_debug()
    prior_handler = loop.get_exception_handler()
    loop_errors: list[dict[str, object]] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        loop.set_debug(True)
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            with pytest.raises(asyncio.CancelledError):
                await bot.weekly_review_scheduled_notification(
                    telegram,
                    user.telegram_id,
                    user.timezone,
                )
            await asyncio.gather(*tuple(bot._weekly_review_tasks), return_exceptions=True)
            await asyncio.sleep(0)
            gc.collect()
        finally:
            await asyncio.gather(*tuple(bot._weekly_review_tasks), return_exceptions=True)
            loop.set_exception_handler(prior_handler)
            loop.set_debug(prior_debug)

    sent = telegram.sent[0]["message"]
    assert telegram.deleted == [{"chat_id": user.telegram_id, "message_id": sent.message_id}]
    assert bot.weekly_review_capabilities._capabilities == {}
    assert bot._weekly_review_tasks == set()
    assert loop_errors == []
    assert [warning for warning in caught_warnings if warning.category is RuntimeWarning] == []
    assert private not in caplog.text
    assert str(user.telegram_id) not in caplog.text
    assert "operation=post_send error_type=CancelledError" in caplog.text


@pytest.mark.asyncio
async def test_scheduled_mid_issue_failure_leaves_no_partial_capability(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_SCHEDULED_MID_ISSUE_FAILURE"
    user = await weekly_user(db, 7297)
    bot = WeeklyHarness(db, fake_ai)
    telegram = WeeklyBot()
    generated = iter(("staged-screen", "staged-first"))

    def fail_mid_batch(_length: int) -> str:
        try:
            return next(generated)
        except StopIteration:
            raise RuntimeError(private) from None

    monkeypatch.setattr(weekly_review_flow_module.secrets, "token_urlsafe", fail_mid_batch)

    await bot.weekly_review_scheduled_notification(telegram, user.telegram_id, user.timezone)
    await asyncio.sleep(0)

    sent = telegram.sent[0]["message"]
    assert len(telegram.sent) == 1
    assert telegram.edits == []
    assert telegram.deleted == [{"chat_id": user.telegram_id, "message_id": sent.message_id}]
    assert bot.weekly_review_capabilities._capabilities == {}
    assert bot.weekly_review_capabilities._screens == {}
    assert bot.weekly_review_capabilities._canonical_generations == {}
    assert bot.weekly_review_capabilities._next_screen_order == 0
    assert bot._weekly_review_tasks == set()
    assert private not in caplog.text
    assert str(user.telegram_id) not in caplog.text
    assert "operation=post_send error_type=RuntimeError" in caplog.text
