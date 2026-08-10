from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import event, func, select

from future_self.access import GUEST, SUBSCRIBER
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import (
    DraftInboxItem,
    InboxItem,
    RecurringTaskReminderSchedule,
    TaskReminder,
    User,
)
from future_self.reminder_flow import ReminderFlowPhase, ReminderFlowStore
from future_self.reminder_handlers import REMINDER_STALE_TEXT
from future_self.reminder_intent import (
    ReminderIntentParser,
    ReminderScheduleKind,
    ReminderTimezoneSource,
)
from future_self.repositories import UserRepository

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)  # 15:00 in Moscow


class ReminderMessage:
    _ids = count(80_000)

    def __init__(self, text: str | None = None, *, message_id: int | None = None) -> None:
        self.text = text
        self.message_id = message_id if message_id is not None else next(self._ids)
        self.voice = None
        self.audio = None
        self.photo = None
        self.document = None
        self.replies: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted = 0

    async def reply_text(self, text: str, **kwargs: Any) -> ReminderMessage:
        sent = ReminderMessage(text)
        self.replies.append({"text": text, "message": sent, **kwargs})
        return sent

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})

    async def delete(self) -> None:
        self.deleted += 1


class ReminderQuery:
    def __init__(self, data: str, message: ReminderMessage) -> None:
        self.data = data
        self.message = message
        self.answers: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        self.answers.append({"args": args, **kwargs})

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})


class ReminderBot:
    def __init__(self) -> None:
        self.edits: list[dict[str, Any]] = []

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edits.append(kwargs)


class DisabledTranscription:
    enabled = False


def reminder_settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
        transcription_provider="disabled",
    )


def reminder_update(
    message: ReminderMessage,
    *,
    telegram_user_id: int,
    chat_id: int,
    query: ReminderQuery | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=query.message if query is not None else message,
        message=message,
        effective_user=SimpleNamespace(id=telegram_user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        callback_query=query,
    )


def reminder_context() -> SimpleNamespace:
    return SimpleNamespace(bot=ReminderBot(), user_data={})


async def subscriber(db, telegram_user_id: int) -> User:
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(
            telegram_user_id,
            "Europe/Moscow",
        )
        user.access_tier = SUBSCRIBER
        user.access_version = 4
        return user


def deterministic_bot(db, fake_ai) -> FutureSelfBot:
    bot = FutureSelfBot(reminder_settings(), db, fake_ai, DisabledTranscription())
    bot.reminder_intent_parser = ReminderIntentParser(now_provider=lambda: NOW)
    bot._reminder_now_provider = lambda: NOW
    return bot


async def current_session(bot: FutureSelfBot, user: User, chat_id: int):
    return await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
    )


def latest_markup(message: ReminderMessage) -> Any:
    assert message.edits
    return message.edits[-1]["reply_markup"]


def callback_for(markup: Any, label: str) -> str:
    for row in markup.inline_keyboard:
        for button in row:
            if button.text == label:
                return button.callback_data
    raise AssertionError(f"Button not found: {label}")


def assert_stale_alert(query: ReminderQuery) -> None:
    assert query.edits == []
    assert query.answers == [
        {
            "args": (REMINDER_STALE_TEXT,),
            "show_alert": True,
        }
    ]


@pytest.mark.asyncio
async def test_complete_once_uses_one_canonical_and_existing_task_reminder_path(db, fake_ai):
    user = await subscriber(db, 6101)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9101)
    context = reminder_context()

    assert await bot.reminder_text_gate(update, context) is True
    assert len(incoming.replies) == 1
    assert "reply_markup" not in incoming.replies[0]
    assert incoming.replies[0]["text"] == "🔔 Готовлю напоминание…"
    canonical = incoming.replies[0]["message"]
    assert "Проверь напоминание" in canonical.edits[-1]["text"]
    markup = latest_markup(canonical)
    data = callback_for(markup, "✅ Создать")
    assert data.startswith("rmd:")
    assert all(secret not in data for secret in ("позвонить", "19:30", "6101", "Europe"))

    query = ReminderQuery(data, canonical)
    callback_update = reminder_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9101,
        query=query,
    )
    await bot.reminder_callback(callback_update, context)

    assert len(query.answers) == 1
    assert len(query.edits) == 1
    assert "Напоминание создано" in query.edits[0]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_daily_confirm_is_atomic_single_use_and_has_no_one_shot_reminder(db, fake_ai):
    user = await subscriber(db, 6102)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Каждый день напоминай в 20:30 заполнить дневник")
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9102)
    context = reminder_context()

    assert await bot.reminder_text_gate(update, context) is True
    canonical = incoming.replies[0]["message"]
    data = callback_for(latest_markup(canonical), "✅ Включить")
    first = ReminderQuery(data, canonical)
    replay = ReminderQuery(data, canonical)
    first_update = reminder_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9102,
        query=first,
    )
    replay_update = reminder_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9102,
        query=replay,
    )

    await asyncio.gather(
        bot.reminder_callback(first_update, context),
        bot.reminder_callback(replay_update, context),
    )

    assert len(first.answers) == len(replay.answers) == 1
    winner, loser = (first, replay) if first.edits else (replay, first)
    assert len(winner.edits) == 1
    assert "Ежедневное напоминание включено" in winner.edits[0]["text"]
    assert winner.answers == [{"args": ()}]
    assert_stale_alert(loser)
    assert canonical.replies == []
    assert context.bot.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        item = await session.scalar(select(InboxItem))
        assert item.resolved_date is None
        assert item.temporal_resolution is None


@pytest.mark.asyncio
async def test_missing_when_then_daily_keeps_title_and_time(db, fake_ai):
    user = await subscriber(db, 6103)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Напомни в 19:30 заполнить дневник благодарностей")
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9103)
    context = reminder_context()

    assert await bot.reminder_text_gate(update, context) is True
    canonical = incoming.replies[0]["message"]
    assert canonical.edits[-1]["text"] == "🔔 Когда напомнить?"
    data = callback_for(latest_markup(canonical), "🔁 Каждый день")
    query = ReminderQuery(data, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9103,
            query=query,
        ),
        context,
    )

    assert len(query.answers) == 1
    assert "каждый день в 19:30" in query.edits[-1]["text"]
    assert "заполнить дневник благодарностей" in query.edits[-1]["text"]


@pytest.mark.asyncio
async def test_text_to_voice_continuation_reuses_canonical_and_deletes_progress(db, fake_ai):
    user = await subscriber(db, 6104)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Напомни завтра в 19:30")
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9104)
    context = reminder_context()

    assert await bot.reminder_text_gate(update, context) is True
    canonical = incoming.replies[0]["message"]
    current = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9104,
    )
    assert current is not None and current.phase is ReminderFlowPhase.TITLE
    progress = ReminderMessage("Расшифровываю…")
    voice_update = reminder_update(progress, telegram_user_id=user.telegram_id, chat_id=9104)

    assert await bot.reminder_voice_gate(
        voice_update,
        context,
        "позвонить врачу",
        progress,
        expected_access_version=user.access_version,
        expected_session=current,
    )
    assert progress.deleted == 1
    assert len(incoming.replies) == 1
    assert context.bot.edits[-1]["message_id"] == canonical.message_id
    assert "позвонить врачу" in context.bot.edits[-1]["text"]


@pytest.mark.parametrize("lifecycle", ["cancelled", "replaced"])
@pytest.mark.asyncio
async def test_voice_started_in_old_session_cannot_resurrect_it_after_stt(
    db,
    fake_ai,
    lifecycle,
):
    user = await subscriber(db, 6199)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    first = ReminderMessage("Напомни завтра в 19:30")
    update = reminder_update(first, telegram_user_id=user.telegram_id, chat_id=9199)
    assert await bot.reminder_text_gate(update, context)
    expected = await current_session(bot, user, 9199)
    assert expected is not None
    await bot.reminder_sessions.clear(
        owner_id=expected.owner_id,
        telegram_user_id=expected.telegram_user_id,
        chat_id=expected.chat_id,
        session_id=expected.id,
    )
    replacement = None
    if lifecycle == "replaced":
        replacement_input = ReminderMessage("Каждый день напоминай в 20:30 заполнить дневник")
        assert await bot.reminder_text_gate(
            reminder_update(
                replacement_input,
                telegram_user_id=user.telegram_id,
                chat_id=9199,
            ),
            context,
        )
        replacement = await current_session(bot, user, 9199)
        assert replacement is not None and replacement.id != expected.id
    progress = ReminderMessage("Расшифровываю…")

    assert await bot.reminder_voice_gate(
        reminder_update(progress, telegram_user_id=user.telegram_id, chat_id=9199),
        context,
        "позвонить врачу",
        progress,
        expected_access_version=expected.access_version,
        expected_session=expected,
    )

    assert progress.deleted == 1
    live = await current_session(bot, user, 9199)
    if replacement is None:
        assert live is None
    else:
        assert live is not None and live.id == replacement.id


@pytest.mark.asyncio
async def test_active_session_voice_unsupported_deletes_progress_and_edits_canonical(
    db,
    fake_ai,
):
    user = await subscriber(db, 6198)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9198),
        context,
    )
    current = await current_session(bot, user, 9198)
    canonical = incoming.replies[0]["message"]
    progress = ReminderMessage("Расшифровываю…")

    assert await bot.reminder_voice_gate(
        reminder_update(progress, telegram_user_id=user.telegram_id, chat_id=9198),
        context,
        "Напомни каждую неделю в 19:30 позвонить врачу",
        progress,
        expected_access_version=current.access_version,
        expected_session=current,
    )

    assert progress.deleted == 1
    assert await current_session(bot, user, 9198) is None
    assert context.bot.edits[-1]["message_id"] == canonical.message_id
    assert "только разовые и ежедневные" in context.bot.edits[-1]["text"]


@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_active_session_voice_access_change_deletes_transient_progress(
    db,
    fake_ai,
    change,
):
    user = await subscriber(db, 6197)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9197),
        context,
    )
    current = await current_session(bot, user, 9197)
    canonical = incoming.replies[0]["message"]
    async with db.session() as session:
        stored = await session.scalar(select(User).where(User.id == user.id))
        stored.access_tier = GUEST
        stored.access_version += 1
        if change == "bounce":
            stored.access_tier = SUBSCRIBER
            stored.access_version += 1
    progress = ReminderMessage("Расшифровываю…")

    assert await bot.reminder_voice_gate(
        reminder_update(progress, telegram_user_id=user.telegram_id, chat_id=9197),
        context,
        "позвонить врачу",
        progress,
        expected_access_version=current.access_version,
        expected_session=current,
    )

    assert progress.deleted == 1
    assert await current_session(bot, user, 9197) is None
    assert context.bot.edits[-1]["message_id"] == canonical.message_id
    assert "Доступ изменился" in context.bot.edits[-1]["text"]


@pytest.mark.asyncio
async def test_ordinary_text_is_not_intercepted(db, fake_ai):
    user = await subscriber(db, 6105)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Эта песня напомнила школу")
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9105)

    assert await bot.reminder_text_gate(update, reminder_context()) is False
    assert incoming.replies == []


@pytest.mark.asyncio
async def test_access_bounce_before_callback_clears_exact_session_without_writes(db, fake_ai):
    user = await subscriber(db, 6106)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Каждый день напоминай в 20:30 заполнить дневник")
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9106)
    context = reminder_context()
    assert await bot.reminder_text_gate(update, context) is True
    canonical = incoming.replies[0]["message"]
    data = callback_for(latest_markup(canonical), "✅ Включить")

    async with db.session() as session:
        stored = await session.scalar(select(User).where(User.id == user.id))
        stored.access_tier = GUEST
        stored.access_version += 1
        stored.access_tier = SUBSCRIBER
        stored.access_version += 1

    query = ReminderQuery(data, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9106,
            query=query,
        ),
        context,
    )

    assert len(query.answers) == 1
    assert "Доступ изменился" in canonical.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_past_today_and_unsupported_weekly_are_explicit(db, fake_ai):
    user = await subscriber(db, 6107)
    bot = deterministic_bot(db, fake_ai)
    past = ReminderMessage("Напомни сегодня в 00:01 проверить почту")
    past_update = reminder_update(past, telegram_user_id=user.telegram_id, chat_id=9107)
    assert await bot.reminder_text_gate(past_update, reminder_context()) is True
    assert "уже прошло" in past.replies[0]["message"].edits[-1]["text"]

    weekly = ReminderMessage("Напомни каждый понедельник в 19:30 проверить почту")
    weekly_update = reminder_update(weekly, telegram_user_id=user.telegram_id, chat_id=9108)
    assert await bot.reminder_text_gate(weekly_update, reminder_context()) is True
    assert "только разовые и ежедневные" in weekly.replies[0]["text"]


@pytest.mark.asyncio
async def test_one_shot_that_becomes_past_before_confirm_has_no_dml(db, fake_ai):
    user = await subscriber(db, 6220)
    clock = [NOW]
    bot = deterministic_bot(db, fake_ai)
    bot.reminder_intent_parser = ReminderIntentParser(now_provider=lambda: clock[0])
    bot._reminder_now_provider = lambda: clock[0]
    context = reminder_context()
    incoming = ReminderMessage("Напомни сегодня в 15:01 проверить почту")

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9220),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(latest_markup(canonical), "✅ Создать")
    clock[0] = NOW + timedelta(minutes=2)
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)
    query = ReminderQuery(token, canonical)
    try:
        await bot.reminder_callback(
            reminder_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=9220,
                query=query,
            ),
            context,
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert "уже прошло" in query.edits[0]["text"]
    labels = {
        button.text for row in query.edits[0]["reply_markup"].inline_keyboard for button in row
    }
    assert {"Сегодня", "Завтра", "Выбрать дату", "🔁 Каждый день"} <= labels
    live = await current_session(bot, user, 9220)
    assert live is not None and live.phase is ReminderFlowPhase.PAST
    assert not any(
        statement.lstrip().split(maxsplit=1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}
        for statement in statements
        if statement.strip()
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize("late_read", [2, 3], ids=["before_owner_lock", "after_owner_lock"])
@pytest.mark.asyncio
async def test_one_shot_crossing_time_during_save_is_rolled_back_and_shows_past(
    db,
    fake_ai,
    late_read,
):
    user = await subscriber(db, 6224)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни сегодня в 15:01 проверить почту")

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9224),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(latest_markup(canonical), "✅ Создать")
    clock_reads = 0

    def advancing_clock() -> datetime:
        nonlocal clock_reads
        clock_reads += 1
        return NOW if clock_reads < late_read else NOW + timedelta(minutes=2)

    bot._reminder_now_provider = advancing_clock
    query = ReminderQuery(token, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9224,
            query=query,
        ),
        context,
    )

    assert clock_reads >= late_read
    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 1
    assert "уже прошло" in query.edits[0]["text"]
    live = await current_session(bot, user, 9224)
    assert live is not None and live.phase is ReminderFlowPhase.PAST
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_store_expiry_and_stale_capability_are_fail_closed():
    store = ReminderFlowStore(ttl=timedelta(seconds=1))
    started = datetime(2026, 8, 10, 12, tzinfo=UTC)
    session = await store.create(
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_version=4,
        title="задача",
        schedule_kind=None,
        local_date=None,
        local_time=None,
        timezone="Europe/Moscow",
        timezone_source=ReminderTimezoneSource.PROFILE,
        phase=ReminderFlowPhase.WHEN,
        canonical_message_id=10,
        now=started,
    )
    tokens = await store.issue(session, ("cancel",), now=started)

    assert (
        await store.claim(
            tokens["cancel"],
            telegram_user_id=2,
            chat_id=3,
            canonical_message_id=10,
            now=started + timedelta(seconds=2),
        )
        is None
    )
    assert (
        await store.current(
            owner_id=1,
            telegram_user_id=2,
            chat_id=3,
            now=started + timedelta(seconds=2),
        )
        is None
    )


@pytest.mark.parametrize(
    ("phrase", "phase", "kind", "title", "local_date", "local_time"),
    [
        (
            "Напомни",
            ReminderFlowPhase.WHEN,
            ReminderScheduleKind.ONCE,
            None,
            None,
            None,
        ),
        (
            "Напомни заполнить дневник",
            ReminderFlowPhase.WHEN,
            ReminderScheduleKind.ONCE,
            "заполнить дневник",
            None,
            None,
        ),
        (
            "Напомни в 19:30 заполнить дневник",
            ReminderFlowPhase.WHEN,
            ReminderScheduleKind.ONCE,
            "заполнить дневник",
            None,
            time(19, 30),
        ),
        (
            "Напомни завтра заполнить дневник",
            ReminderFlowPhase.TIME,
            ReminderScheduleKind.ONCE,
            "заполнить дневник",
            date(2026, 8, 11),
            None,
        ),
        (
            "Напомни завтра в 19:30",
            ReminderFlowPhase.TITLE,
            ReminderScheduleKind.ONCE,
            None,
            date(2026, 8, 11),
            time(19, 30),
        ),
        (
            "Каждый день напоминай заполнить дневник",
            ReminderFlowPhase.TIME,
            ReminderScheduleKind.DAILY,
            "заполнить дневник",
            None,
            None,
        ),
        (
            "Каждый день напоминай в 19:30",
            ReminderFlowPhase.TITLE,
            ReminderScheduleKind.DAILY,
            None,
            None,
            time(19, 30),
        ),
    ],
)
@pytest.mark.asyncio
async def test_missing_field_combinations_follow_when_time_title_priority(
    db,
    fake_ai,
    phrase,
    phase,
    kind,
    title,
    local_date,
    local_time,
):
    user = await subscriber(db, 6201)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage(phrase)
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9201)

    assert await bot.reminder_text_gate(update, reminder_context()) is True

    session = await current_session(bot, user, 9201)
    assert session is not None
    assert session.phase is phase
    assert session.schedule_kind is kind
    assert session.title == title
    assert session.local_date == local_date
    assert session.local_time == local_time
    assert len(incoming.replies) == 1
    assert incoming.replies[0]["message"].replies == []


@pytest.mark.parametrize(
    ("phrase", "kind", "local_date", "timezone", "timezone_source"),
    [
        (
            "Напомни сегодня в 23:59 проверить почту",
            ReminderScheduleKind.ONCE,
            date(2026, 8, 10),
            "Europe/Moscow",
            ReminderTimezoneSource.PROFILE,
        ),
        (
            "Напомни завтра в 19:30 позвонить врачу",
            ReminderScheduleKind.ONCE,
            date(2026, 8, 11),
            "Europe/Moscow",
            ReminderTimezoneSource.PROFILE,
        ),
        (
            "Напомни 15 августа в 19:30 позвонить врачу",
            ReminderScheduleKind.ONCE,
            date(2026, 8, 15),
            "Europe/Moscow",
            ReminderTimezoneSource.PROFILE,
        ),
        (
            "Каждый день напоминай в 19:30 по Asia/Tbilisi заполнить дневник",
            ReminderScheduleKind.DAILY,
            None,
            "Asia/Tbilisi",
            ReminderTimezoneSource.EXPLICIT,
        ),
    ],
)
@pytest.mark.asyncio
async def test_complete_today_tomorrow_custom_date_and_daily_open_preview(
    db,
    fake_ai,
    phrase,
    kind,
    local_date,
    timezone,
    timezone_source,
):
    user = await subscriber(db, 6202)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage(phrase)

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9202),
        reminder_context(),
    )

    session = await current_session(bot, user, 9202)
    assert session is not None
    assert session.phase is ReminderFlowPhase.PREVIEW
    assert session.schedule_kind is kind
    assert session.local_date == local_date
    assert session.timezone == timezone
    assert session.timezone_source is timezone_source
    assert len(incoming.replies) == 1
    canonical = incoming.replies[0]["message"]
    callbacks = [
        button.callback_data for row in latest_markup(canonical).inline_keyboard for button in row
    ]
    assert callbacks
    assert all(value is not None and value.startswith("rmd:") for value in callbacks)
    assert all(
        sensitive not in value
        for value in callbacks
        for sensitive in ("позвонить", "дневник", "19:30", timezone, str(user.telegram_id))
    )


@pytest.mark.asyncio
async def test_choose_custom_date_then_text_continuation_reuses_canonical(db, fake_ai):
    user = await subscriber(db, 6203)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Напомни в 19:30 позвонить врачу")
    context = reminder_context()
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9203),
        context,
    )
    canonical = incoming.replies[0]["message"]
    choose = callback_for(latest_markup(canonical), "Выбрать дату")
    query = ReminderQuery(choose, canonical)

    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9203,
            query=query,
        ),
        context,
    )
    assert len(query.answers) == 1
    assert len(query.edits) == 1
    assert "На какую дату" in query.edits[0]["text"]

    continuation = ReminderMessage("15 августа")
    assert await bot.reminder_text_gate(
        reminder_update(continuation, telegram_user_id=user.telegram_id, chat_id=9203),
        context,
    )
    session = await current_session(bot, user, 9203)
    assert session is not None
    assert session.phase is ReminderFlowPhase.PREVIEW
    assert session.local_date == date(2026, 8, 15)
    assert session.local_time == time(19, 30)
    assert session.title == "позвонить врачу"
    assert continuation.replies == []
    assert context.bot.edits[-1]["message_id"] == canonical.message_id


@pytest.mark.parametrize(
    ("label", "expected_date"),
    [("Сегодня", date(2026, 8, 10)), ("Завтра", date(2026, 8, 11))],
)
@pytest.mark.asyncio
async def test_when_button_preserves_title_and_advances_to_missing_time(
    db,
    fake_ai,
    label,
    expected_date,
):
    user = await subscriber(db, 6215)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Напомни заполнить дневник")
    context = reminder_context()
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9215),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(latest_markup(canonical), label)
    query = ReminderQuery(token, canonical)

    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9215,
            query=query,
        ),
        context,
    )

    session = await current_session(bot, user, 9215)
    assert session is not None
    assert session.phase is ReminderFlowPhase.TIME
    assert session.local_date == expected_date
    assert session.local_time is None
    assert session.title == "заполнить дневник"
    assert len(query.answers) == 1
    assert len(query.edits) == 1
    assert "Во сколько напомнить" in query.edits[0]["text"]
    assert canonical.replies == []


@pytest.mark.parametrize(
    ("phrase", "phase"),
    [
        ("Напомни 31 февраля в 19:00 позвонить", ReminderFlowPhase.INVALID),
        ("Напомни завтра в 25:70 позвонить", ReminderFlowPhase.INVALID),
        ("Напомни сегодня завтра в 19:00 позвонить", ReminderFlowPhase.INVALID),
        ("Напомни завтра в 18:00 или 19:00 позвонить", ReminderFlowPhase.INVALID),
        (
            "Каждый день завтра в 19:00 напоминай позвонить",
            ReminderFlowPhase.INVALID,
        ),
        (
            "Напомни завтра в 19:00 по Ocean/Atlantis позвонить",
            ReminderFlowPhase.INVALID,
        ),
        ("Напомни 1 августа 2026 в 19:00 позвонить", ReminderFlowPhase.PAST),
    ],
)
@pytest.mark.asyncio
async def test_invalid_conflicting_and_past_inputs_fail_closed(
    db,
    fake_ai,
    phrase,
    phase,
):
    user = await subscriber(db, 6204)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage(phrase)

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9204),
        reminder_context(),
    )
    session = await current_session(bot, user, 9204)
    assert session is not None and session.phase is phase
    assert len(incoming.replies) == 1
    assert len(incoming.replies[0]["message"].edits) == 1
    async with db.sessions() as session_db:
        assert await session_db.scalar(select(func.count(InboxItem.id))) == 0
        assert await session_db.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session_db.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни каждую неделю в 19:30 проверить почту",
        "Напомни по будням в 19:30 проверить почту",
        "Напомни каждый понедельник в 19:30 проверить почту",
    ],
)
@pytest.mark.asyncio
async def test_unsupported_weekly_and_weekday_patterns_are_explicit(db, fake_ai, phrase):
    user = await subscriber(db, 6205)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage(phrase)

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9205),
        reminder_context(),
    )
    assert len(incoming.replies) == 1
    assert "только разовые и ежедневные" in incoming.replies[0]["text"]
    assert await current_session(bot, user, 9205) is None


@pytest.mark.asyncio
async def test_voice_to_text_continuation_keeps_voice_progress_as_canonical(db, fake_ai):
    user = await subscriber(db, 6206)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    progress = ReminderMessage("Расшифровываю…")

    assert await bot.reminder_voice_gate(
        reminder_update(progress, telegram_user_id=user.telegram_id, chat_id=9206),
        context,
        "Напомни завтра в 19:30",
        progress,
        expected_access_version=user.access_version,
        expected_session=None,
    )
    first = await current_session(bot, user, 9206)
    assert first is not None and first.phase is ReminderFlowPhase.TITLE
    assert first.canonical_message_id == progress.message_id
    assert progress.replies == []
    assert len(progress.edits) == 1

    continuation = ReminderMessage("позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(continuation, telegram_user_id=user.telegram_id, chat_id=9206),
        context,
    )
    final = await current_session(bot, user, 9206)
    assert final is not None and final.phase is ReminderFlowPhase.PREVIEW
    assert final.title == "позвонить врачу"
    assert final.canonical_message_id == progress.message_id
    assert continuation.replies == []
    assert context.bot.edits[-1]["message_id"] == progress.message_id


@pytest.mark.asyncio
async def test_new_explicit_command_replaces_session_and_invalidates_old_callback(
    db,
    fake_ai,
):
    user = await subscriber(db, 6207)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    first_message = ReminderMessage("Напомни завтра в 19:30")
    assert await bot.reminder_text_gate(
        reminder_update(first_message, telegram_user_id=user.telegram_id, chat_id=9207),
        context,
    )
    canonical = first_message.replies[0]["message"]
    old_session = await current_session(bot, user, 9207)
    assert old_session is not None
    old_cancel = callback_for(latest_markup(canonical), "Отмена")

    replacement = ReminderMessage("Каждый день напоминай в 20:30 заполнить дневник")
    assert await bot.reminder_text_gate(
        reminder_update(replacement, telegram_user_id=user.telegram_id, chat_id=9207),
        context,
    )
    new_session = await current_session(bot, user, 9207)
    assert new_session is not None
    assert new_session.id != old_session.id
    assert new_session.title == "заполнить дневник"
    assert new_session.schedule_kind is ReminderScheduleKind.DAILY
    assert new_session.canonical_message_id == canonical.message_id
    assert replacement.replies == []
    canonical_edits_before = list(canonical.edits)

    stale = ReminderQuery(old_cancel, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9207,
            query=stale,
        ),
        context,
    )
    assert_stale_alert(stale)
    assert canonical.edits == canonical_edits_before
    assert (await current_session(bot, user, 9207)).id == new_session.id


@pytest.mark.asyncio
async def test_cancel_command_clears_session_and_edits_only_canonical(db, fake_ai):
    user = await subscriber(db, 6208)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9208),
        context,
    )
    canonical = incoming.replies[0]["message"]
    cancel_message = ReminderMessage("/cancel")

    assert await bot.reminder_cancel_gate(
        reminder_update(cancel_message, telegram_user_id=user.telegram_id, chat_id=9208),
        context,
    )
    assert await current_session(bot, user, 9208) is None
    assert cancel_message.replies == []
    assert context.bot.edits[-1]["message_id"] == canonical.message_id
    assert "отменено" in context.bot.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


@pytest.mark.asyncio
async def test_cancel_callback_answers_once_and_never_sends_fallback(db, fake_ai):
    user = await subscriber(db, 6216)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9216),
        context,
    )
    canonical = incoming.replies[0]["message"]
    query = ReminderQuery(callback_for(latest_markup(canonical), "Отмена"), canonical)

    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9216,
            query=query,
        ),
        context,
    )

    assert len(query.answers) == 1
    assert len(query.edits) == 1
    assert "отменено" in query.edits[0]["text"]
    assert await current_session(bot, user, 9216) is None
    assert canonical.replies == []


@pytest.mark.asyncio
async def test_durable_flow_entry_clears_reminder_and_invalidates_old_capability(db, fake_ai):
    user = await subscriber(db, 6209)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9209),
        context,
    )
    canonical = incoming.replies[0]["message"]
    old_token = callback_for(latest_markup(canonical), "Отмена")
    flow_message = ReminderMessage("/evening")

    await bot.evening_start(
        reminder_update(flow_message, telegram_user_id=user.telegram_id, chat_id=9209),
        context,
    )

    assert await current_session(bot, user, 9209) is None
    stale_query = ReminderQuery(old_token, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9209,
            query=stale_query,
        ),
        context,
    )
    assert_stale_alert(stale_query)
    assert context.bot.edits == []


@pytest.mark.asyncio
async def test_expired_handler_callback_is_stale_and_has_no_fallback(db, fake_ai):
    user = await subscriber(db, 6209)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9209),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(latest_markup(canonical), "✅ Создать")
    live = await current_session(bot, user, 9209)
    assert live is not None
    assert await bot.reminder_sessions.cleanup(now=live.expires_at) == 1

    query = ReminderQuery(token, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9209,
            query=query,
        ),
        context,
    )
    assert_stale_alert(query)
    assert canonical.replies == []
    assert context.bot.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


@pytest.mark.parametrize("mismatch", ["owner", "chat", "canonical"])
@pytest.mark.asyncio
async def test_callback_capability_is_bound_to_owner_chat_and_canonical(
    db,
    fake_ai,
    mismatch,
):
    user = await subscriber(db, 6210)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9210),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(latest_markup(canonical), "✅ Создать")
    callback_message = canonical
    telegram_user_id = user.telegram_id
    chat_id = 9210
    if mismatch == "owner":
        telegram_user_id += 1
    elif mismatch == "chat":
        chat_id += 1
    else:
        callback_message = ReminderMessage(message_id=canonical.message_id + 1)
    query = ReminderQuery(token, callback_message)

    await bot.reminder_callback(
        reminder_update(
            callback_message,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            query=query,
        ),
        context,
    )

    assert_stale_alert(query)
    assert context.bot.edits == []
    assert await current_session(bot, user, 9210) is not None

    legitimate = ReminderQuery(token, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9210,
            query=legitimate,
        ),
        context,
    )

    assert len(legitimate.answers) == 1
    assert len(legitimate.edits) == 1
    assert "создано" in legitimate.edits[0]["text"]
    assert await current_session(bot, user, 9210) is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1


@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_access_change_during_initial_placeholder_neutralizes_without_private_ui(
    db,
    fake_ai,
    change,
):
    user = await subscriber(db, 6222)
    bot = deterministic_bot(db, fake_ai)
    private_title = "PRIVATE_PLACEHOLDER_TITLE_38e1"

    class AccessChangingMessage(ReminderMessage):
        async def reply_text(self, text: str, **kwargs: Any) -> ReminderMessage:
            sent = await super().reply_text(text, **kwargs)
            async with db.session() as session:
                stored = await session.scalar(select(User).where(User.id == user.id))
                stored.access_tier = GUEST
                stored.access_version += 1
                if change == "bounce":
                    stored.access_tier = SUBSCRIBER
                    stored.access_version += 1
            return sent

    incoming = AccessChangingMessage(f"Каждый день напоминай в 20:30 {private_title}")
    context = reminder_context()

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9222),
        context,
    )

    assert len(incoming.replies) == 1
    assert incoming.replies[0]["text"] == "🔔 Готовлю напоминание…"
    canonical = incoming.replies[0]["message"]
    assert len(canonical.edits) == 1
    assert "Доступ изменился" in canonical.edits[0]["text"]
    assert private_title not in canonical.edits[0]["text"]
    assert await current_session(bot, user, 9222) is None
    assert context.bot.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_system_action_clears_reminder_and_old_callback_cannot_overwrite_canonical(
    db,
    fake_ai,
    monkeypatch,
):
    user = await subscriber(db, 6223)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9223),
        context,
    )
    canonical = incoming.replies[0]["message"]
    old_token = callback_for(latest_markup(canonical), "Отмена")
    canonical_edits_before = list(canonical.edits)
    handled_routes = []

    async def record_system_route(_update, _context, _user, _snapshot, route):
        handled_routes.append(route)

    monkeypatch.setattr(bot, "_handle_system_action_route", record_system_route)
    action_message = ReminderMessage("удали все черновики")
    action_update = reminder_update(
        action_message,
        telegram_user_id=user.telegram_id,
        chat_id=9223,
    )

    assert await bot._try_system_action(
        action_update,
        context,
        action_message.text,
    )
    assert len(handled_routes) == 1
    assert await current_session(bot, user, 9223) is None

    stale = ReminderQuery(old_token, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9223,
            query=stale,
        ),
        context,
    )
    assert_stale_alert(stale)
    assert canonical.edits == canonical_edits_before


@pytest.mark.asyncio
async def test_guest_before_start_is_consumed_without_session_or_telegram_io(db, fake_ai):
    user = await subscriber(db, 6211)
    async with db.session() as session:
        stored = await session.scalar(select(User).where(User.id == user.id))
        stored.access_tier = GUEST
        stored.access_version += 1
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9211),
        reminder_context(),
    )
    assert incoming.replies == []
    assert await current_session(bot, user, 9211) is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_voice_access_change_during_stt_neutralizes_progress_without_writes(
    db,
    fake_ai,
    change,
):
    user = await subscriber(db, 6218)
    expected_access_version = user.access_version
    async with db.session() as session:
        stored = await session.scalar(select(User).where(User.id == user.id))
        stored.access_tier = GUEST
        stored.access_version += 1
        if change == "bounce":
            stored.access_tier = SUBSCRIBER
            stored.access_version += 1
    bot = deterministic_bot(db, fake_ai)
    progress = ReminderMessage("Расшифровываю…")

    assert await bot.reminder_voice_gate(
        reminder_update(progress, telegram_user_id=user.telegram_id, chat_id=9218),
        reminder_context(),
        "Каждый день напоминай в 20:30 заполнить дневник",
        progress,
        expected_access_version=expected_access_version,
        expected_session=None,
    )

    assert len(progress.edits) == 1
    assert "Доступ изменился" in progress.edits[0]["text"]
    assert await current_session(bot, user, 9218) is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize("failure", ["downgrade", "status_error"])
@pytest.mark.asyncio
async def test_relative_voice_access_failure_is_consumed_before_legacy_pipeline(
    db,
    fake_ai,
    monkeypatch,
    failure,
):
    user = await subscriber(db, 6225)
    expected_access_version = user.access_version
    bot = deterministic_bot(db, fake_ai)
    if failure == "downgrade":
        async with db.session() as session:
            stored = await session.scalar(select(User).where(User.id == user.id))
            stored.access_tier = GUEST
            stored.access_version += 1
    else:

        async def fail_status(_telegram_user_id):
            raise RuntimeError("PRIVATE_RELATIVE_STATUS_BODY")

        monkeypatch.setattr(bot.access_service, "status", fail_status)
    progress = ReminderMessage("Расшифровываю…")

    handled = await bot.reminder_voice_gate(
        reminder_update(progress, telegram_user_id=user.telegram_id, chat_id=9225),
        reminder_context(),
        "Напомни через 2 часа проверить духовку",
        progress,
        expected_access_version=expected_access_version,
        expected_session=None,
    )

    assert handled is True
    assert progress.replies == []
    assert len(progress.edits) == 1
    assert "Доступ изменился" in progress.edits[0]["text"]
    assert await current_session(bot, user, 9225) is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_relative_text_status_failure_is_consumed_before_legacy_pipeline(
    db,
    fake_ai,
    monkeypatch,
):
    user = await subscriber(db, 6226)
    bot = deterministic_bot(db, fake_ai)

    async def fail_status(_telegram_user_id):
        raise RuntimeError("PRIVATE_RELATIVE_TEXT_STATUS_BODY")

    monkeypatch.setattr(bot.access_service, "status", fail_status)
    incoming = ReminderMessage("Напомни через 2 часа проверить духовку")

    handled = await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9226),
        reminder_context(),
    )

    assert handled is True
    assert incoming.replies == []
    assert await current_session(bot, user, 9226) is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_access_status_exception_is_fail_closed_and_logs_no_private_body(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    user = await subscriber(db, 6217)
    bot = deterministic_bot(db, fake_ai)
    raw_input = "Напомни завтра в 19:30 PRIVATE_ACCESS_INPUT_33a7"
    exception_body = "PRIVATE_ACCESS_EXCEPTION_91de"

    async def fail_status(_telegram_user_id):
        raise RuntimeError(exception_body)

    monkeypatch.setattr(bot.access_service, "status", fail_status)
    caplog.set_level("WARNING", logger="future_self.reminder_handlers")
    incoming = ReminderMessage(raw_input)
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9217),
        reminder_context(),
    )

    assert incoming.replies == []
    assert await current_session(bot, user, 9217) is None
    assert "RuntimeError" in caplog.text
    assert raw_input not in caplog.text
    assert exception_body not in caplog.text


@pytest.mark.asyncio
async def test_access_downgrade_after_session_sync_clears_and_neutralizes(db, fake_ai):
    user = await subscriber(db, 6212)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9212),
        context,
    )
    canonical = incoming.replies[0]["message"]
    async with db.session() as session:
        stored = await session.scalar(select(User).where(User.id == user.id))
        stored.access_tier = GUEST
        stored.access_version += 1
    async with db.sessions() as session:
        downgraded = await session.scalar(select(User).where(User.id == user.id))

    await bot.reminder_sync_access(
        downgraded,
        9212,
        context=context,
        source_message=canonical,
    )

    assert await current_session(bot, user, 9212) is None
    assert "Доступ изменился" in canonical.edits[-1]["text"]
    assert canonical.replies == []
    assert context.bot.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


@pytest.mark.asyncio
async def test_old_access_cleanup_cannot_overwrite_replacement_on_same_canonical(db, fake_ai):
    user = await subscriber(db, 6219)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Напомни завтра в 19:30 позвонить врачу")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9219),
        context,
    )
    old = await current_session(bot, user, 9219)
    canonical = incoming.replies[0]["message"]
    replacement = await bot.reminder_sessions.create(
        owner_id=old.owner_id,
        telegram_user_id=old.telegram_user_id,
        chat_id=old.chat_id,
        access_version=old.access_version,
        title="новое напоминание",
        schedule_kind=ReminderScheduleKind.DAILY,
        local_date=None,
        local_time=time(20, 30),
        timezone=old.timezone,
        timezone_source=old.timezone_source,
        phase=ReminderFlowPhase.PREVIEW,
        canonical_message_id=old.canonical_message_id,
    )
    edits_before = len(canonical.edits)

    await bot._reminder_access_changed(context, old, source_message=canonical)

    live = await current_session(bot, user, 9219)
    assert live is not None and live.id == replacement.id
    assert len(canonical.edits) == edits_before
    assert context.bot.edits == []


@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_access_change_immediately_before_save_is_fail_closed(
    db,
    fake_ai,
    monkeypatch,
    change,
):
    user = await subscriber(db, 6213)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Каждый день напоминай в 20:30 заполнить дневник")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9213),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(latest_markup(canonical), "✅ Включить")
    original_access = bot._reminder_access
    calls = 0

    async def access_then_change(update):
        nonlocal calls
        binding = await original_access(update)
        calls += 1
        if calls == 1 and binding is not None:
            async with db.session() as session:
                stored = await session.scalar(select(User).where(User.id == user.id))
                stored.access_tier = GUEST
                stored.access_version += 1
                if change == "bounce":
                    stored.access_tier = SUBSCRIBER
                    stored.access_version += 1
        return binding

    monkeypatch.setattr(bot, "_reminder_access", access_then_change)
    query = ReminderQuery(token, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9213,
            query=query,
        ),
        context,
    )

    assert len(query.answers) == 1
    assert await current_session(bot, user, 9213) is None
    assert "Доступ изменился" in canonical.edits[-1]["text"]
    assert query.edits == []
    assert context.bot.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_access_change_inside_save_before_owner_fence_rolls_back_everything(
    db,
    fake_ai,
    monkeypatch,
    change,
):
    user = await subscriber(db, 6221)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Каждый день напоминай в 20:30 заполнить дневник")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9221),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(latest_markup(canonical), "✅ Включить")
    original_save = bot._reminder_save_atomic

    async def change_access_then_save(flow_session):
        async with db.session() as session:
            stored = await session.scalar(select(User).where(User.id == user.id))
            stored.access_tier = GUEST
            stored.access_version += 1
            if change == "bounce":
                stored.access_tier = SUBSCRIBER
                stored.access_version += 1
        return await original_save(flow_session)

    monkeypatch.setattr(bot, "_reminder_save_atomic", change_access_then_save)
    query = ReminderQuery(token, canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9221,
            query=query,
        ),
        context,
    )

    assert query.answers == [{"args": ()}]
    assert query.edits == []
    assert await current_session(bot, user, 9221) is None
    assert "Доступ изменился" in canonical.edits[-1]["text"]
    assert "заполнить дневник" not in canonical.edits[-1]["text"]
    assert context.bot.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.asyncio
async def test_private_input_title_and_exception_body_never_enter_callbacks_or_logs(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    source,
):
    user = await subscriber(db, 6214)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    private_title = f"PRIVATE_{source.upper()}_TITLE_8f24"
    raw_input = f"Каждый день напоминай в 20:30 {private_title}"
    if source == "voice":
        canonical = ReminderMessage("Расшифровываю…")
        assert await bot.reminder_voice_gate(
            reminder_update(canonical, telegram_user_id=user.telegram_id, chat_id=9214),
            context,
            raw_input,
            canonical,
            expected_access_version=user.access_version,
            expected_session=None,
        )
    else:
        incoming = ReminderMessage(raw_input)
        assert await bot.reminder_text_gate(
            reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9214),
            context,
        )
        canonical = incoming.replies[0]["message"]
    markup = latest_markup(canonical)
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert all(private_title not in (value or "") for value in callbacks)
    assert all(raw_input not in (value or "") for value in callbacks)
    exception_body = "PRIVATE_EXCEPTION_BODY_6df1"

    async def fail_save(_session):
        raise RuntimeError(exception_body)

    monkeypatch.setattr(bot, "_reminder_save_atomic", fail_save)
    caplog.set_level("WARNING", logger="future_self.reminder_handlers")
    query = ReminderQuery(callback_for(markup, "✅ Включить"), canonical)
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9214,
            query=query,
        ),
        context,
    )

    assert len(query.answers) == 1
    assert len(query.edits) == 1
    assert "попробуй ещё раз" in query.edits[0]["text"]
    log_text = caplog.text
    assert "RuntimeError" in log_text
    assert raw_input not in log_text
    assert private_title not in log_text
    assert exception_body not in log_text
    assert canonical.replies == []


@pytest.mark.asyncio
async def test_daily_schedule_failure_rolls_back_task_in_same_transaction(
    db,
    fake_ai,
    monkeypatch,
):
    user = await subscriber(db, 6218)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    incoming = ReminderMessage("Каждый день напоминай в 20:30 заполнить дневник")
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9218),
        context,
    )
    canonical = incoming.replies[0]["message"]

    async def fail_schedule(*_args, **_kwargs):
        raise RuntimeError("simulated schedule write failure")

    monkeypatch.setattr(
        bot.recurring_reminder_service,
        "create_daily_in_session",
        fail_schedule,
    )
    query = ReminderQuery(
        callback_for(latest_markup(canonical), "✅ Включить"),
        canonical,
    )
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9218,
            query=query,
        ),
        context,
    )

    assert len(query.answers) == 1
    assert len(query.edits) == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0
