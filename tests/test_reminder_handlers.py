from __future__ import annotations

import asyncio
from dataclasses import replace
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
    TaskState,
    User,
)
from future_self.nova_companion_flow import NovaCompanionReminderCandidate
from future_self.nova_memory_flow import NovaMemoryFlowPhase
from future_self.reminder_flow import ReminderFlowPhase, ReminderFlowStore
from future_self.reminder_handlers import (
    REMINDER_STALE_TEXT,
    _reminder_turn_understanding,
    _semantic_fallback_reminder_text,
)
from future_self.reminder_intent import (
    ReminderIntentParser,
    ReminderScheduleKind,
    ReminderTimezoneSource,
)
from future_self.repositories import UserRepository
from future_self.schemas import ReminderTimezoneResolution
from future_self.timezones import extract_reminder_timezone_fragment

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


async def save_direct_reminder(
    bot: FutureSelfBot,
    user: User,
    *,
    title: str,
    weekly_candidate_handoff: bool,
    chat_id: int,
    schedule_kind: ReminderScheduleKind = ReminderScheduleKind.ONCE,
    local_time: time = time(19, 30),
    timezone: str = "Europe/Moscow",
):
    flow_session = await bot.reminder_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        access_version=user.access_version,
        title=title,
        schedule_kind=schedule_kind,
        local_date=date(2026, 8, 11) if schedule_kind is ReminderScheduleKind.ONCE else None,
        local_time=local_time,
        timezone=timezone,
        timezone_source=ReminderTimezoneSource.PROFILE,
        phase=ReminderFlowPhase.PREVIEW,
        weekly_candidate_handoff=weekly_candidate_handoff,
    )
    result, recurring = await bot._reminder_save_atomic(flow_session)
    assert result.ok is True
    assert result.inbox_item is not None
    return flow_session, result, recurring


@pytest.mark.parametrize(
    "phrase",
    [
        "Можешь напомнить мне лечь спать сегодня в 22?",
        "Можешь, пожалуйста, напомнить мне сегодня в 22 часа лечь спать?",
        "Сможешь напомнить мне сегодня в 22:00 лечь спать?",
    ],
)
def test_operational_turn_understanding_grounds_slots_independent_of_word_order(phrase):
    parser = ReminderIntentParser(now_provider=lambda: NOW)
    base = parser.parse(phrase, "Europe/Moscow")

    understood = _reminder_turn_understanding(
        phrase,
        base,
        parser=parser,
        timezone="Europe/Moscow",
        now=NOW,
    )

    assert understood.requested_action == "collect"
    assert understood.elastic_routing is True
    assert understood.exact_action_anchor.casefold() in {"напомнить"}
    assert understood.title == "лечь спать"
    assert understood.local_date == date(2026, 8, 10)
    assert understood.local_time == time(22)
    assert understood.known_slots == ("title", "date", "time", "timezone")
    assert understood.missing_slot is None
    assert understood.confidence == "high"
    assert understood.ambiguous is False
    assert understood.allowed_transition is ReminderFlowPhase.PREVIEW


def test_semantic_fallback_canonicalization_removes_subordinate_frame() -> None:
    phrase = "Мне хотелось бы, чтобы ты напомнила сегодня в 22:00 позвонить врачу."
    canonical = _semantic_fallback_reminder_text(phrase)

    assert canonical is not None
    parsed = ReminderIntentParser(now_provider=lambda: NOW).parse(
        canonical,
        "Europe/Moscow",
    )
    assert parsed.title == "позвонить врачу"
    assert parsed.local_date == date(2026, 8, 10)
    assert parsed.local_time == time(22)
    assert parsed.status.value == "complete"


@pytest.mark.parametrize(
    ("phrase", "expected_time", "expected_status"),
    [
        ("Создай напоминание на 05.09.2026 в 22:00 позвонить врачу.", time(22), "complete"),
        ("Создай напоминание на 5.9.2026 в 22.00 позвонить врачу.", time(22), "complete"),
        ("Создай напоминание на 2026-09-05 в 22:00 позвонить врачу.", time(22), "complete"),
        ("Создай напоминание в 22:00 на 05.09.2026 позвонить врачу.", time(22), "complete"),
        ("Создай напоминание на 05.09.2026 позвонить врачу.", None, "needs_time"),
    ],
)
def test_operational_turn_understanding_protects_full_date_spans_and_base_slots(
    phrase,
    expected_time,
    expected_status,
):
    parser = ReminderIntentParser(now_provider=lambda: NOW)
    base = parser.parse(phrase, "Europe/Moscow")

    understood = _reminder_turn_understanding(
        phrase,
        base,
        parser=parser,
        timezone="Europe/Moscow",
        now=NOW,
    )

    assert understood.parser_result.status.value == expected_status
    assert understood.title == "позвонить врачу"
    assert understood.local_date == date(2026, 9, 5)
    assert understood.local_time == expected_time


@pytest.mark.parametrize(
    ("phrase", "expected_status", "error_code"),
    [
        (
            "Создай напоминание на 05.09.2026 или 06.09.2026 в 22:00 позвонить врачу.",
            "invalid",
            "ambiguous_date",
        ),
        (
            "Создай напоминание на 05.09.2026 в 21:00 или 22:00 позвонить врачу.",
            "needs_time",
            "missing_time",
        ),
    ],
)
def test_operational_turn_understanding_does_not_resolve_conflicting_temporal_spans(
    phrase,
    expected_status,
    error_code,
):
    parser = ReminderIntentParser(now_provider=lambda: NOW)
    base = parser.parse(phrase, "Europe/Moscow")

    understood = _reminder_turn_understanding(
        phrase,
        base,
        parser=parser,
        timezone="Europe/Moscow",
        now=NOW,
    )

    assert understood.parser_result.status.value == expected_status
    assert understood.parser_result.error_code.value == error_code


async def seed_memory_root(
    bot: FutureSelfBot,
    user: User,
    *,
    chat_id: int,
    canonical_message_id: int,
):
    return await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        tier=user.access_tier,
        access_version=user.access_version,
        canonical_message_id=canonical_message_id,
        phase=NovaMemoryFlowPhase.ROOT,
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
async def test_direct_confirm_records_exact_read_only_completion_receipt(db, fake_ai):
    user = await subscriber(db, 6_191)
    bot = deterministic_bot(db, fake_ai)
    bot.settings.enable_nova_companion = True
    bot.settings.nova_companion_admin_only = False
    incoming = ReminderMessage("Напомни завтра в 19:00 позвонить врачу")
    context = reminder_context()

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9_191),
        context,
    )
    canonical = incoming.replies[0]["message"]
    confirm = callback_for(latest_markup(canonical), "✅ Создать")
    await bot.reminder_callback(
        reminder_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9_191,
            query=ReminderQuery(confirm, canonical),
        ),
        context,
    )

    key = (user.id, user.telegram_id, 9_191)
    receipt = bot._nova_companion_status_receipts[key]
    assert receipt.owner_id == user.id
    assert receipt.telegram_user_id == user.telegram_id
    assert receipt.chat_id == 9_191
    assert receipt.access_version == user.access_version
    assert receipt.task_reminder_id is not None
    assert receipt.inbox_item_id > 0
    assert receipt.title == "позвонить врачу"
    assert receipt.timezone == "Europe/Moscow"
    assert receipt.schedule_kind == "once"
    assert receipt.local_date == date(2026, 8, 11)
    assert receipt.local_time == time(19)

    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        answer = await bot._nova_companion_status_answer(
            "Ты уже создала напоминание?",
            user=await bot._user(user.telegram_id),
            chat_id=9_191,
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    assert answer == "Да, напоминание создано на завтра, 19:00."
    assert bot._nova_companion_status_receipts[key] is receipt
    assert fake_ai.companion_calls == []
    assert not any(
        statement.lstrip().split(maxsplit=1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}
        for statement in statements
        if statement.strip()
    )

    bot._nova_companion_status_receipts.pop(key)
    for _attempt in range(2):
        assert (
            await bot._nova_companion_status_answer(
                "Ты уже создала напоминание?",
                user=await bot._user(user.telegram_id),
                chat_id=9_191,
            )
            == "Пока нет — напоминание ещё не создано."
        )
    other = await subscriber(db, 6_194)
    bot._nova_companion_status_receipts[key] = receipt
    assert (
        await bot._nova_companion_status_answer(
            "Ты уже создала напоминание?",
            user=other,
            chat_id=9_191,
        )
        == "Пока нет — напоминание ещё не создано."
    )
    forged = replace(receipt, local_time=time(18))
    bot._nova_companion_status_receipts[key] = forged
    assert (
        await bot._nova_companion_status_answer(
            "Ты уже создала напоминание?",
            user=await bot._user(user.telegram_id),
            chat_id=9_191,
        )
        == "Пока нет — напоминание ещё не создано."
    )
    assert key not in bot._nova_companion_status_receipts


@pytest.mark.asyncio
async def test_weekly_candidate_adapter_reuses_one_canonical_and_existing_confirm_path(
    db,
    fake_ai,
):
    user = await subscriber(db, 6199)
    bot = deterministic_bot(db, fake_ai)
    canonical = ReminderMessage(message_id=89_999)
    update = reminder_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9199,
    )
    context = reminder_context()

    handled = await bot.reminder_from_weekly_candidate(
        update,
        context,
        title="Позвонить врачу",
        schedule_wording="завтра в 19:30",
        canonical_message=canonical,
        expected_access_version=user.access_version,
    )

    assert handled is True
    assert canonical.replies == []
    assert len(canonical.edits) == 1
    reminder_session = await current_session(bot, user, 9199)
    assert reminder_session is not None
    assert reminder_session.canonical_message_id == canonical.message_id
    assert reminder_session.weekly_candidate_handoff is True
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0

    callback_data = callback_for(latest_markup(canonical), "✅ Создать")
    query = ReminderQuery(callback_data, canonical)
    callback_update = reminder_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9199,
        query=query,
    )
    await bot.reminder_callback(callback_update, context)

    assert query.answers == [{"args": ()}]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize("near_title", ["  ПОЗВОНИТЬ,   ВРАЧУ!!! ", "Позвонить врачю"])
@pytest.mark.asyncio
async def test_weekly_candidate_near_duplicate_reuses_existing_owner_reminder(
    db,
    fake_ai,
    near_title,
):
    user = await subscriber(db, 6200)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()

    first_canonical = ReminderMessage(message_id=90_001)
    first_update = reminder_update(
        first_canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9200,
    )
    assert await bot.reminder_from_weekly_candidate(
        first_update,
        context,
        title="Позвонить врачу",
        schedule_wording="завтра в 19:30",
        canonical_message=first_canonical,
        expected_access_version=user.access_version,
    )
    first_query = ReminderQuery(
        callback_for(latest_markup(first_canonical), "✅ Создать"),
        first_canonical,
    )
    await bot.reminder_callback(
        reminder_update(
            first_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9200,
            query=first_query,
        ),
        context,
    )

    near_canonical = ReminderMessage(message_id=90_002)
    near_update = reminder_update(
        near_canonical,
        telegram_user_id=user.telegram_id,
        chat_id=9200,
    )
    assert await bot.reminder_from_weekly_candidate(
        near_update,
        context,
        title=near_title,
        schedule_wording="завтра в 19:30",
        canonical_message=near_canonical,
        expected_access_version=user.access_version,
    )
    near_query = ReminderQuery(
        callback_for(latest_markup(near_canonical), "✅ Создать"),
        near_canonical,
    )
    await bot.reminder_callback(
        reminder_update(
            near_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=9200,
            query=near_query,
        ),
        context,
    )

    assert near_query.answers == [{"args": ()}]
    assert "✓ Уже настроено" in near_query.edits[-1]["text"]
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 1
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Не звонить врачу", "Звонить врачу"),
        ("Купить 2 билета", "Купить 3 билета"),
        ("Позвонить Анне", "Позвонить Алле"),
        ("Отправить письмо Анне", "Отправить письмо Алине"),
        ("Проверить важную почту", "Проверить личную почту"),
    ],
)
def test_weekly_near_duplicate_matcher_preserves_semantic_distinctions(left, right):
    assert FutureSelfBot._weekly_reminder_titles_near(left, right) is False


@pytest.mark.asyncio
async def test_weekly_duplicate_short_circuit_has_only_owner_lock_dml_and_no_audit(
    db,
    fake_ai,
    monkeypatch,
):
    user = await subscriber(db, 6230)
    bot = deterministic_bot(db, fake_ai)
    await save_direct_reminder(
        bot,
        user,
        title="Позвонить врачу",
        weekly_candidate_handoff=False,
        chat_id=9230,
    )
    audit_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "future_self.reminder_handlers.log_transition",
        lambda *args, **kwargs: audit_calls.append((*args, kwargs)),
    )
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        verb = statement.lstrip().split(maxsplit=1)[0].upper()
        if verb in {"INSERT", "UPDATE", "DELETE"}:
            statements.append(" ".join(statement.split()).lower())

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        _flow, result, recurring = await save_direct_reminder(
            bot,
            user,
            title="Позвонить врачю",
            weekly_candidate_handoff=True,
            chat_id=9230,
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    assert result.duplicate is True
    assert recurring is None
    assert audit_calls == []
    assert len(statements) == 1
    assert statements[0].startswith("update users set updated_at=users.updated_at")
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(DraftInboxItem.id))) == 1
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 1
        assert await db_session.scalar(select(func.count(TaskState.id))) == 1
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.asyncio
async def test_ordinary_reminder_flow_does_not_gain_weekly_typo_dedup(db, fake_ai):
    user = await subscriber(db, 6231)
    bot = deterministic_bot(db, fake_ai)
    first, first_result, _recurring = await save_direct_reminder(
        bot,
        user,
        title="Позвонить врачу",
        weekly_candidate_handoff=False,
        chat_id=9231,
    )
    second, second_result, _recurring = await save_direct_reminder(
        bot,
        user,
        title="Позвонить врачю",
        weekly_candidate_handoff=False,
        chat_id=9231,
    )

    assert first.weekly_candidate_handoff is False
    assert second.weekly_candidate_handoff is False
    assert first_result.duplicate is False
    assert second_result.duplicate is False
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(DraftInboxItem.id))) == 2
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 2
        assert await db_session.scalar(select(func.count(TaskState.id))) == 2
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == 2
        assert await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize(
    "excluded_scope",
    [
        "other_owner",
        "other_schedule",
        "other_timezone",
        "other_kind",
        "unconfirmed_item",
        "inactive_task",
        "inactive_reminder",
        "daily_schedule",
    ],
)
@pytest.mark.asyncio
async def test_weekly_duplicate_query_requires_exact_active_task_scope(
    db,
    fake_ai,
    excluded_scope,
):
    candidate_user = await subscriber(db, 6240)
    seed_user = await subscriber(db, 6241) if excluded_scope == "other_owner" else candidate_user
    bot = deterministic_bot(db, fake_ai)
    seed_kind = (
        ReminderScheduleKind.DAILY
        if excluded_scope == "daily_schedule"
        else ReminderScheduleKind.ONCE
    )
    seed_time = time(18, 30) if excluded_scope == "other_schedule" else time(19, 30)
    await save_direct_reminder(
        bot,
        seed_user,
        title="Позвонить врачу",
        weekly_candidate_handoff=False,
        chat_id=9240,
        schedule_kind=seed_kind,
        local_time=seed_time,
    )

    if excluded_scope not in {"other_owner", "other_schedule", "daily_schedule"}:
        async with db.session() as db_session:
            item = await db_session.scalar(select(InboxItem))
            assert item is not None
            reminder = await db_session.scalar(
                select(TaskReminder).where(TaskReminder.inbox_item_id == item.id)
            )
            state = await db_session.scalar(
                select(TaskState).where(TaskState.inbox_item_id == item.id)
            )
            assert reminder is not None and state is not None
            if excluded_scope == "other_timezone":
                reminder.timezone = "Europe/London"
            elif excluded_scope == "other_kind":
                item.kind = "note"
            elif excluded_scope == "unconfirmed_item":
                item.status = "archived"
            elif excluded_scope == "inactive_task":
                state.status = "completed"
                state.completed_at = NOW
            elif excluded_scope == "inactive_reminder":
                reminder.status = "sent"
                reminder.sent_at = NOW

    _flow, result, recurring = await save_direct_reminder(
        bot,
        candidate_user,
        title="Позвонить врачу!",
        weekly_candidate_handoff=True,
        chat_id=9242,
    )

    assert result.duplicate is False
    assert recurring is None
    async with db.sessions() as db_session:
        assert await db_session.scalar(select(func.count(DraftInboxItem.id))) == 2
        assert await db_session.scalar(select(func.count(InboxItem.id))) == 2
        assert await db_session.scalar(select(func.count(TaskState.id))) == 2
        expected_once = 1 if excluded_scope == "daily_schedule" else 2
        expected_daily = 1 if excluded_scope == "daily_schedule" else 0
        assert await db_session.scalar(select(func.count(TaskReminder.id))) == expected_once
        assert (
            await db_session.scalar(select(func.count(RecurringTaskReminderSchedule.id)))
            == expected_daily
        )


@pytest.mark.asyncio
async def test_daily_confirm_is_atomic_single_use_and_has_no_one_shot_reminder(db, fake_ai):
    user = await subscriber(db, 6102)
    bot = deterministic_bot(db, fake_ai)
    bot.settings.enable_nova_companion = True
    bot.settings.nova_companion_admin_only = False
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
    key = (user.id, user.telegram_id, 9102)
    receipt = bot._nova_companion_status_receipts[key]
    assert receipt.schedule_kind == "daily"
    assert receipt.recurring_schedule_id is not None
    assert receipt.recurring_schedule_version == 1
    assert receipt.local_time == time(20, 30)
    assert receipt.local_date is not None
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=9102,
        )
        is None
    )
    async with db.sessions() as session:
        schedule = await session.get(
            RecurringTaskReminderSchedule,
            receipt.recurring_schedule_id,
        )
        saved_item = await session.get(InboxItem, receipt.inbox_item_id)
    assert schedule is not None and saved_item is not None
    assert schedule.owner_id == receipt.owner_id
    assert schedule.inbox_item_id == receipt.inbox_item_id
    assert schedule.recurrence_kind == "daily"
    assert schedule.local_time == receipt.local_time
    assert schedule.timezone == receipt.timezone
    assert schedule.start_local_date == receipt.local_date
    assert schedule.version == receipt.recurring_schedule_version
    assert schedule.status == "active"
    assert saved_item.user_id == receipt.owner_id
    assert saved_item.kind == "task"
    assert saved_item.status == "confirmed"
    assert saved_item.version == receipt.inbox_item_version
    assert saved_item.title == receipt.title
    sql_trace: list[tuple[str, object]] = []

    def record_sql(_conn, _cursor, statement, parameters, _context, _executemany):
        sql_trace.append((statement, parameters))

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_sql)
    try:
        status_answer = await bot._nova_companion_status_answer(
            "Ты уже создала напоминание?",
            user=await bot._user(user.telegram_id),
            chat_id=9102,
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_sql)
    assert status_answer == "Да, ежедневное напоминание создано на 20:30.", sql_trace
    assert bot._nova_companion_status_receipts[key] is receipt


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
@pytest.mark.parametrize("source", ["text", "voice"])
async def test_accepted_reminder_clears_only_exact_memory_flow(db, fake_ai, source):
    user = await subscriber(db, 6196 if source == "text" else 6195)
    bot = deterministic_bot(db, fake_ai)
    chat_id = 9196 if source == "text" else 9195
    exact = await seed_memory_root(
        bot,
        user,
        chat_id=chat_id,
        canonical_message_id=91_960,
    )
    other = await seed_memory_root(
        bot,
        user,
        chat_id=chat_id + 100,
        canonical_message_id=91_961,
    )
    context = reminder_context()
    command = "Каждый день напоминай в 20:30 заполнить дневник"
    if source == "text":
        incoming = ReminderMessage(command)
        handled = await bot.reminder_text_gate(
            reminder_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
            ),
            context,
        )
    else:
        progress = ReminderMessage("Расшифровываю…")
        handled = await bot.reminder_voice_gate(
            reminder_update(
                progress,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
            ),
            context,
            command,
            progress,
            expected_access_version=user.access_version,
            expected_session=None,
        )

    assert handled is True
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
        )
        is None
    )
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id + 100,
        )
        == other
    )
    reminder = await current_session(bot, user, chat_id)
    assert reminder is not None
    assert reminder.title == "заполнить дневник"
    assert exact.id != other.id
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


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


@pytest.mark.parametrize("corrected", ["22:00", "22:00, сорри"])
@pytest.mark.asyncio
async def test_past_time_continuation_keeps_event_date_and_time_phase(
    db,
    fake_ai,
    corrected,
):
    user = await subscriber(db, 6_192)
    bot = deterministic_bot(db, fake_ai)
    chat_id = 9_192
    context = reminder_context()
    initial = ReminderMessage("Напомни про наш разговор сегодня")

    assert await bot.reminder_text_gate(
        reminder_update(initial, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    first = await current_session(bot, user, chat_id)
    assert first is not None and first.phase is ReminderFlowPhase.TIME
    assert first.title == "про наш разговор"
    assert first.local_date == date(2026, 8, 10)
    old_cancel = callback_for(latest_markup(initial.replies[0]["message"]), "Отмена")

    past = ReminderMessage("10:00")
    assert await bot.reminder_text_gate(
        reminder_update(past, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    rejected = await current_session(bot, user, chat_id)
    assert rejected is not None and rejected.phase is ReminderFlowPhase.TIME
    assert rejected.id != first.id
    assert rejected.title == first.title
    assert rejected.local_date == first.local_date
    assert rejected.local_time is None
    assert rejected.past_time_rejected is True
    assert "уже прошло" in context.bot.edits[-1]["text"]
    assert "Во сколько" in context.bot.edits[-1]["text"]

    stale = ReminderQuery(old_cancel, initial.replies[0]["message"])
    await bot.reminder_callback(
        reminder_update(
            initial.replies[0]["message"],
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=stale,
        ),
        context,
    )
    assert_stale_alert(stale)

    correction = ReminderMessage(corrected)
    assert await bot.reminder_text_gate(
        reminder_update(correction, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    preview = await current_session(bot, user, chat_id)
    assert preview is not None and preview.phase is ReminderFlowPhase.PREVIEW
    assert preview.title == first.title
    assert preview.local_date == first.local_date
    assert preview.local_time == time(22)
    assert preview.past_time_rejected is False
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.asyncio
async def test_guided_recurrence_collects_slots_and_fails_closed_to_supported_daily_option(
    db,
    fake_ai,
):
    user = await subscriber(db, 6_193)
    bot = deterministic_bot(db, fake_ai)
    chat_id = 9_193
    canonical = ReminderMessage(message_id=89_193)
    context = reminder_context()
    candidate = NovaCompanionReminderCandidate(
        title="возвращаться к главному",
        evidence="Напоминай мне почаще возвращаться к главному",
        timezone="Europe/Moscow",
        guided_recurrence=True,
    )

    assert await bot.reminder_from_companion_candidate(
        reminder_update(canonical, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
        candidate=candidate,
        canonical_message_id=canonical.message_id,
        expected_access_version=user.access_version,
    )
    current = await current_session(bot, user, chat_id)
    assert current is not None and current.phase is ReminderFlowPhase.RECURRENCE_FREQUENCY
    assert current.title == "возвращаться к главному"
    assert current.schedule_kind is None
    assert current.local_date is None and current.local_time is None

    frequency = ReminderMessage("раз десять")
    assert await bot.reminder_text_gate(
        reminder_update(frequency, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    current = await current_session(bot, user, chat_id)
    assert current is not None and current.phase is ReminderFlowPhase.RECURRENCE_PERIOD
    assert current.recurrence_frequency_per_day == 10

    period = ReminderMessage("в течение дня")
    assert await bot.reminder_text_gate(
        reminder_update(period, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    current = await current_session(bot, user, chat_id)
    assert current is not None and current.phase is ReminderFlowPhase.RECURRENCE_DAYS
    assert current.recurrence_active_period == "day"

    premature = ReminderMessage("Самое время создать")
    rejected_version = current.version
    assert await bot.reminder_text_gate(
        reminder_update(premature, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    current = await current_session(bot, user, chat_id)
    assert current is not None and current.phase is ReminderFlowPhase.RECURRENCE_DAYS
    assert current.version == rejected_version
    assert "каждый день" in premature.replies[-1]["text"]

    days = ReminderMessage("Каждый день")
    assert await bot.reminder_text_gate(
        reminder_update(days, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    current = await current_session(bot, user, chat_id)
    assert current is not None and current.phase is ReminderFlowPhase.RECURRENCE_TIMES

    times = ReminderMessage("09:00, 11:00, 13:00")
    assert await bot.reminder_text_gate(
        reminder_update(times, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    current = await current_session(bot, user, chat_id)
    assert current is not None and current.phase is ReminderFlowPhase.RECURRENCE_SINGLE_TIME
    assert "только одно ежедневное" in context.bot.edits[-1]["text"]

    fallback = ReminderMessage("19:00")
    assert await bot.reminder_text_gate(
        reminder_update(fallback, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    current = await current_session(bot, user, chat_id)
    assert current is not None and current.phase is ReminderFlowPhase.PREVIEW
    assert current.schedule_kind is ReminderScheduleKind.DAILY
    assert current.local_time == time(19)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


@pytest.mark.parametrize(
    ("phase", "reply"),
    [
        (ReminderFlowPhase.RECURRENCE_PERIOD, "раз десять"),
    ],
)
@pytest.mark.asyncio
async def test_guided_recurrence_wrong_slot_keeps_exact_generation(
    db,
    fake_ai,
    phase,
    reply,
):
    user = await subscriber(db, 6_195 + list(ReminderFlowPhase).index(phase))
    bot = deterministic_bot(db, fake_ai)
    session = await bot.reminder_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9_195,
        access_version=user.access_version,
        title="возвращаться к дыханию",
        schedule_kind=None,
        local_date=None,
        local_time=None,
        timezone="Europe/Moscow",
        timezone_source=ReminderTimezoneSource.PROFILE,
        phase=phase,
        guided_recurrence=True,
        recurrence_frequency_per_day=(10 if phase is ReminderFlowPhase.RECURRENCE_PERIOD else None),
    )

    turn = await bot._reminder_guided_recurrence_turn(session, reply)

    assert turn.session is session
    assert turn.error is not None
    live = await bot.reminder_sessions.get_exact(session)
    assert live is session
    assert live.version == session.version
    assert live.recurrence_frequency_per_day == session.recurrence_frequency_per_day


@pytest.mark.parametrize(
    ("reply", "expected_phase", "frequency", "period"),
    [
        ("раз десять", ReminderFlowPhase.RECURRENCE_PERIOD, 10, None),
        ("десять раз в день", ReminderFlowPhase.RECURRENCE_DAYS, 10, "day"),
        ("каждый час в течение дня", ReminderFlowPhase.RECURRENCE_SINGLE_TIME, 1, None),
    ],
)
@pytest.mark.asyncio
async def test_guided_recurrence_understands_natural_frequency_turns_without_guessing(
    db,
    fake_ai,
    reply,
    expected_phase,
    frequency,
    period,
):
    user = await subscriber(db, 6_198 + frequency)
    bot = deterministic_bot(db, fake_ai)
    session = await bot.reminder_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9_198,
        access_version=user.access_version,
        title="возвращаться к главному",
        schedule_kind=None,
        local_date=None,
        local_time=None,
        timezone="Europe/Moscow",
        timezone_source=ReminderTimezoneSource.PROFILE,
        phase=ReminderFlowPhase.RECURRENCE_FREQUENCY,
        guided_recurrence=True,
    )

    turn = await bot._reminder_guided_recurrence_turn(session, reply)

    assert turn.error is None
    assert turn.session is not None and turn.session.phase is expected_phase
    assert turn.session.title == "возвращаться к главному"
    assert turn.session.recurrence_frequency_per_day == frequency
    assert turn.session.recurrence_active_period == period


@pytest.mark.asyncio
async def test_guided_recurrence_unsupported_range_and_multi_time_require_one_fresh_time(
    db,
    fake_ai,
):
    user = await subscriber(db, 6_199)
    bot = deterministic_bot(db, fake_ai)
    session = await bot.reminder_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=9_199,
        access_version=user.access_version,
        title="возвращаться к дыханию",
        schedule_kind=None,
        local_date=None,
        local_time=None,
        timezone="Europe/Moscow",
        timezone_source=ReminderTimezoneSource.PROFILE,
        phase=ReminderFlowPhase.RECURRENCE_DAYS,
        guided_recurrence=True,
        recurrence_frequency_per_day=10,
        recurrence_active_period="day",
    )

    ranged = await bot._reminder_guided_recurrence_turn(
        session,
        "каждый день с 1.09 по 5.09 в 19:00",
    )
    assert ranged.session is not None
    assert ranged.session.phase is ReminderFlowPhase.RECURRENCE_SINGLE_TIME
    assert ranged.session.local_time is None
    assert ranged.session.recurrence_frequency_per_day == 1
    assert ranged.session.recurrence_days == "daily"

    version = ranged.session.version
    multiple = await bot._reminder_guided_recurrence_turn(
        ranged.session,
        "10:00 12:00 14:00 16:00",
    )
    assert multiple.session is ranged.session
    assert multiple.session.version == version
    assert multiple.error is not None

    accepted = await bot._reminder_guided_recurrence_turn(ranged.session, "19:00")
    assert accepted.session is not None
    assert accepted.session.phase is ReminderFlowPhase.PREVIEW
    assert accepted.session.local_time == time(19)


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
    assert labels == {"Отмена"}
    live = await current_session(bot, user, 9220)
    assert live is not None and live.phase is ReminderFlowPhase.TIME
    assert live.title == "проверить почту"
    assert live.local_date == date(2026, 8, 10)
    assert live.local_time is None
    assert live.rejected_local_time == time(15, 1)
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
    assert live is not None and live.phase is ReminderFlowPhase.TIME
    assert live.title == "проверить почту"
    assert live.local_date == date(2026, 8, 10)
    assert live.local_time is None
    assert live.rejected_local_time == time(15, 1)
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
        ("Напомни завтра в 18:00 или 19:00 позвонить", ReminderFlowPhase.TIME),
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


async def reminder_domain_row_counts(db) -> tuple[int, int, int, int]:
    async with db.sessions() as session:
        return (
            await session.scalar(select(func.count(DraftInboxItem.id))),
            await session.scalar(select(func.count(InboxItem.id))),
            await session.scalar(select(func.count(TaskReminder.id))),
            await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))),
        )


async def change_reminder_access(db, user: User, change: str) -> None:
    async with db.session() as session:
        stored = await session.scalar(select(User).where(User.id == user.id))
        assert stored is not None
        stored.access_tier = GUEST
        stored.access_version += 1
        if change == "bounce":
            stored.access_tier = SUBSCRIBER
            stored.access_version += 1


def reminder_rendered_texts(
    incoming: ReminderMessage,
    context: SimpleNamespace,
) -> list[str]:
    texts = [edit["text"] for edit in context.bot.edits]
    for reply in incoming.replies:
        texts.extend(edit["text"] for edit in reply["message"].edits)
    return texts


def latest_reminder_render(
    canonical: ReminderMessage,
    context: SimpleNamespace,
) -> dict[str, Any]:
    if context.bot.edits:
        return context.bot.edits[-1]
    assert canonical.edits
    return canonical.edits[-1]


def reminder_timezone_window(text: str) -> str:
    fragment = extract_reminder_timezone_fragment(text)
    assert fragment is not None
    return fragment.text


@pytest.mark.parametrize(
    ("phrase", "timezone", "kind", "title"),
    [
        (
            "Напомни завтра в 10:00 по Лондону созвониться с клиентом",
            "Europe/London",
            ReminderScheduleKind.ONCE,
            "созвониться с клиентом",
        ),
        (
            "Каждый день в 20:30 по времени Тбилиси заполнить дневник",
            "Asia/Tbilisi",
            ReminderScheduleKind.DAILY,
            "заполнить дневник",
        ),
        (
            "Напомни завтра в 10:00 по МСК проверить почту",
            "Europe/Moscow",
            ReminderScheduleKind.ONCE,
            "проверить почту",
        ),
        (
            "Каждый день в 20:30 по Europe/Berlin заполнить дневник",
            "Europe/Berlin",
            ReminderScheduleKind.DAILY,
            "заполнить дневник",
        ),
        (
            "Напомни завтра в 10:00 в часовом поясе Europe/London позвонить врачу",
            "Europe/London",
            ReminderScheduleKind.ONCE,
            "позвонить врачу",
        ),
        (
            "Напомни завтра в 18:00 в часовом поясе Нью-Йорка проверить почту",
            "America/New_York",
            ReminderScheduleKind.ONCE,
            "проверить почту",
        ),
    ],
)
@pytest.mark.asyncio
async def test_known_natural_and_exact_timezones_open_preview_without_ai(
    db,
    fake_ai,
    phrase,
    timezone,
    kind,
    title,
):
    user = await subscriber(db, 6301)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage(phrase)

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9301),
        reminder_context(),
    )

    session = await current_session(bot, user, 9301)
    assert session is not None
    assert session.phase is ReminderFlowPhase.PREVIEW
    assert session.schedule_kind is kind
    assert session.title == title
    assert session.timezone == timezone
    assert session.timezone_source is ReminderTimezoneSource.EXPLICIT
    assert fake_ai.reminder_timezone_calls == []
    assert fake_ai.timezone_calls == []
    preview = incoming.replies[0]["message"].edits[-1]["text"]
    assert timezone in preview
    if timezone != "Europe/Moscow":
        assert "В твоём часовом поясе:" in preview
        assert "Europe/Moscow" in preview
    if timezone == "Europe/London":
        assert "11.08.2026 12:00 (Europe/Moscow)" in preview
    elif timezone == "Asia/Tbilisi":
        assert "10.08.2026 19:30 (Europe/Moscow)" in preview


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.asyncio
async def test_model_timezone_flow_has_text_stt_parity_and_receives_only_bounded_fragment(
    db,
    fake_ai,
    source,
):
    user = await subscriber(db, 6302)
    bot = deterministic_bot(db, fake_ai)
    phrase = "Напомни завтра в 9:00 по светогорску позвонить врачу"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по светогорску",
        city="Светогорск",
        country="Россия",
    )
    context = reminder_context()
    incoming = ReminderMessage(phrase if source == "text" else "Расшифровываю…")
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9302)

    if source == "text":
        handled = await bot.reminder_text_gate(update, context)
        canonical = incoming.replies[0]["message"]
    else:
        handled = await bot.reminder_voice_gate(
            update,
            context,
            phrase,
            incoming,
            expected_access_version=user.access_version,
            expected_session=None,
        )
        canonical = incoming

    assert handled is True
    session = await current_session(bot, user, 9302)
    assert session is not None
    assert session.phase is ReminderFlowPhase.PREVIEW
    assert session.schedule_kind is ReminderScheduleKind.ONCE
    assert session.local_date == date(2026, 8, 11)
    assert session.local_time == time(9)
    assert session.title == "позвонить врачу"
    assert session.timezone == "Europe/Moscow"
    assert session.timezone_source is ReminderTimezoneSource.EXPLICIT
    assert fake_ai.reminder_timezone_calls == [fragment]
    assert phrase not in fake_ai.reminder_timezone_calls
    assert fake_ai.timezone_calls == []
    final_render = latest_reminder_render(canonical, context)
    assert "позвонить врачу" in final_render["text"]
    callbacks = [
        button.callback_data
        for row in final_render["reply_markup"].inline_keyboard
        for button in row
    ]
    assert all(value is not None and value.startswith("rmd:") for value in callbacks)
    assert all(fragment not in value and phrase not in value for value in callbacks)


@pytest.mark.asyncio
async def test_ambiguous_timezone_clarification_reuses_session_and_canonical_fields(
    db,
    fake_ai,
):
    user = await subscriber(db, 6303)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    phrase = "Напомни завтра в 10:00 по Сан-Хосе созвониться с клиентом"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="ambiguous",
        matched_text="по Сан-Хосе",
    )
    incoming = ReminderMessage(phrase)

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9303),
        context,
    )

    ambiguous = await current_session(bot, user, 9303)
    assert ambiguous is not None
    assert ambiguous.phase is ReminderFlowPhase.TIMEZONE_CLARIFY
    assert ambiguous.title == "созвониться с клиентом"
    assert ambiguous.local_date == date(2026, 8, 11)
    assert ambiguous.local_time == time(10)
    assert phrase not in repr(ambiguous)
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)
    canonical = incoming.replies[0]["message"]
    assert canonical.message_id == ambiguous.canonical_message_id
    assert (
        latest_reminder_render(canonical, context)["text"]
        == "Уточни город вместе со страной или регионом"
    )

    clarification = "Сан-Хосе, Калифорния, США"
    fake_ai.reminder_timezone_results[clarification] = ReminderTimezoneResolution(
        status="resolved",
        timezone="America/Los_Angeles",
        matched_text=clarification,
        city="Сан-Хосе",
        country="США",
    )
    follow_up = ReminderMessage(clarification)
    assert await bot.reminder_text_gate(
        reminder_update(follow_up, telegram_user_id=user.telegram_id, chat_id=9303),
        context,
    )

    resolved = await current_session(bot, user, 9303)
    assert resolved is not None
    assert resolved.id == ambiguous.id
    assert resolved.canonical_message_id == canonical.message_id
    assert resolved.phase is ReminderFlowPhase.PREVIEW
    assert resolved.title == ambiguous.title
    assert resolved.local_date == ambiguous.local_date
    assert resolved.local_time == ambiguous.local_time
    assert resolved.timezone == "America/Los_Angeles"
    assert resolved.timezone_source is ReminderTimezoneSource.EXPLICIT
    assert phrase not in repr(resolved)
    assert clarification not in repr(resolved)
    assert fake_ai.reminder_timezone_calls == [fragment, clarification]
    assert follow_up.replies == []
    assert context.bot.edits[-1]["message_id"] == canonical.message_id
    assert "America/Los_Angeles" in context.bot.edits[-1]["text"]
    assert "В твоём часовом поясе:" in context.bot.edits[-1]["text"]


@pytest.mark.parametrize(
    "failure",
    [
        "invalid_iana",
        "bad_evidence",
        "insufficient",
        "provider_error",
        "timeout",
    ],
)
@pytest.mark.asyncio
async def test_unresolved_model_timezone_is_retryable_without_domain_dml_or_private_logs(
    db,
    fake_ai,
    caplog,
    failure,
):
    user = await subscriber(db, 6304)
    bot = deterministic_bot(db, fake_ai)
    private_title = "PRIVATE_TIMEZONE_TITLE_8831"
    private_error = "PRIVATE_TIMEZONE_ERROR_117c"
    private_model_output = "PRIVATE_MODEL_OUTPUT_09bf"
    phrase = f"Напомни завтра в 9:00 по Светогорску {private_title}"
    fragment = reminder_timezone_window(phrase)
    if failure == "invalid_iana":
        fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
            status="resolved",
            timezone="Ocean/Atlantis",
            matched_text="по Светогорску",
            city=private_model_output,
        )
    elif failure == "bad_evidence":
        fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
            status="resolved",
            timezone="Europe/London",
            matched_text="по Лондону",
            city=private_model_output,
        )
    elif failure == "insufficient":
        fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(status=failure)
    elif failure == "provider_error":
        fake_ai.reminder_timezone_error = RuntimeError(private_error)
    else:
        fake_ai.reminder_timezone_error = TimeoutError(private_error)
    caplog.set_level("WARNING", logger="future_self.reminder_handlers")
    incoming = ReminderMessage(phrase)
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)

    context = reminder_context()
    try:
        assert await bot.reminder_text_gate(
            reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9304),
            context,
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    session = await current_session(bot, user, 9304)
    assert session is not None
    assert session.phase is ReminderFlowPhase.TIMEZONE_RETRY
    assert session.title is None
    assert session.timezone == "Europe/Moscow"
    assert session.timezone_source is ReminderTimezoneSource.PROFILE
    assert fake_ai.reminder_timezone_calls == [fragment]
    canonical = incoming.replies[0]["message"]
    final_render = latest_reminder_render(canonical, context)
    assert "Ничего не сохранено" in final_render["text"]
    callbacks = [
        button.callback_data
        for row in final_render["reply_markup"].inline_keyboard
        for button in row
    ]
    assert callbacks and all(value is not None and value.startswith("rmd:") for value in callbacks)
    assert all(
        sensitive not in value
        for value in callbacks
        for sensitive in (
            phrase,
            fragment,
            private_title,
            private_error,
            private_model_output,
        )
    )
    assert not any(
        statement.lstrip().split(maxsplit=1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}
        for statement in statements
        if statement.strip()
    )
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)
    assert phrase not in repr(session)
    assert fragment not in repr(session)
    assert private_error not in repr(session)
    assert private_model_output not in repr(session)
    assert phrase not in caplog.text
    assert fragment not in caplog.text
    assert private_title not in caplog.text
    assert private_error not in caplog.text
    assert private_model_output not in caplog.text
    expected_error_type = {
        "invalid_iana": "ValueError",
        "bad_evidence": "ValueError",
        "provider_error": "RuntimeError",
        "timeout": "TimeoutError",
    }.get(failure)
    if expected_error_type is None:
        assert caplog.text == ""
    else:
        assert expected_error_type in caplog.text


@pytest.mark.asyncio
async def test_weak_timezone_marker_not_mentioned_resumes_profile_flow_with_full_title(
    db,
    fake_ai,
):
    user = await subscriber(db, 6321)
    bot = deterministic_bot(db, fake_ai)
    phrase = "Напомни завтра в 9:00 по дороге купить лекарства"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(status="not_mentioned")
    incoming = ReminderMessage(phrase)

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9321),
        reminder_context(),
    )

    session = await current_session(bot, user, 9321)
    assert session is not None
    assert session.phase is ReminderFlowPhase.PREVIEW
    assert session.title == "по дороге купить лекарства"
    assert session.timezone == user.timezone
    assert session.timezone_source is ReminderTimezoneSource.PROFILE
    assert fake_ai.reminder_timezone_calls == [fragment]
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_timezone_provider_cancellation_propagates_and_keeps_retry_fence_without_dml(
    db,
    fake_ai,
):
    user = await subscriber(db, 6305)
    bot = deterministic_bot(db, fake_ai)
    phrase = "Напомни завтра в 9:00 по Светогорску позвонить врачу"
    fake_ai.reminder_timezone_error = asyncio.CancelledError()
    incoming = ReminderMessage(phrase)
    context = reminder_context()

    with pytest.raises(asyncio.CancelledError):
        await bot.reminder_text_gate(
            reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9305),
            context,
        )

    session = await current_session(bot, user, 9305)
    assert session is not None
    assert session.phase is ReminderFlowPhase.TIMEZONE_RETRY
    assert fake_ai.reminder_timezone_calls == [reminder_timezone_window(phrase)]
    canonical = incoming.replies[0]["message"]
    assert "Ничего не сохранено" in latest_reminder_render(canonical, context)["text"]
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("phrase", "title"),
    [
        ("Напомни завтра в 19:30 по работе позвонить", "по работе позвонить"),
        (
            "Напомни завтра в 19:30 по проекту отправить отчёт",
            "по проекту отправить отчёт",
        ),
    ],
)
@pytest.mark.asyncio
async def test_semantic_po_title_never_enters_timezone_model_flow(db, fake_ai, phrase, title):
    user = await subscriber(db, 6306)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage(phrase)

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9306),
        reminder_context(),
    )

    session = await current_session(bot, user, 9306)
    assert session is not None
    assert session.phase is ReminderFlowPhase.PREVIEW
    assert session.title == title
    assert session.timezone_source is ReminderTimezoneSource.PROFILE
    assert fake_ai.reminder_timezone_calls == []


@pytest.mark.parametrize("provider_outcome", ["success", "error", "cancelled"])
@pytest.mark.asyncio
async def test_duplicate_concurrent_timezone_input_starts_model_once_for_generation(
    db,
    fake_ai,
    provider_outcome,
):
    user = await subscriber(db, 6307)
    bot = deterministic_bot(db, fake_ai)
    phrase = "Напомни завтра в 9:00 по Светогорску проверить ёлку"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по Светогорску",
    )
    if provider_outcome == "error":
        fake_ai.reminder_timezone_error = RuntimeError("PRIVATE_DUPLICATE_ERROR")
    elif provider_outcome == "cancelled":
        fake_ai.reminder_timezone_error = asyncio.CancelledError()
    fake_ai.reminder_timezone_release.clear()
    first = ReminderMessage(phrase)
    duplicate = ReminderMessage("  НАПОМНИ   ЗАВТРА В 9:00 ПО СВЕТОГОРСКУ ПРОВЕРИТЬ е\u0308лку  ")
    context = reminder_context()

    first_task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(first, telegram_user_id=user.telegram_id, chat_id=9307),
            context,
        )
    )
    await fake_ai.reminder_timezone_started.wait()
    resolving = await current_session(bot, user, 9307)
    assert resolving is not None
    assert resolving.phase is ReminderFlowPhase.TIMEZONE_RESOLVING
    canonical = first.replies[0]["message"]
    canonical_edits = list(canonical.edits)
    bot_edits = list(context.bot.edits)
    duplicate_result = await bot.reminder_text_gate(
        reminder_update(duplicate, telegram_user_id=user.telegram_id, chat_id=9307),
        context,
    )
    assert duplicate_result is True
    assert await current_session(bot, user, 9307) == resolving
    assert canonical.edits == canonical_edits
    assert context.bot.edits == bot_edits
    assert fake_ai.reminder_timezone_calls == [fragment]
    assert duplicate.replies == []

    fake_ai.reminder_timezone_release.set()
    if provider_outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await first_task
    else:
        assert await first_task is True
    session = await current_session(bot, user, 9307)
    assert session is not None
    assert session.id == resolving.id
    assert session.phase is (
        ReminderFlowPhase.PREVIEW
        if provider_outcome == "success"
        else ReminderFlowPhase.TIMEZONE_RETRY
    )
    assert fake_ai.reminder_timezone_calls == [fragment]


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("provider_outcome", ["success", "error", "cancelled"])
@pytest.mark.asyncio
async def test_ordinary_input_during_timezone_resolution_is_absorbed_without_mutation(
    db,
    fake_ai,
    source,
    provider_outcome,
):
    user = await subscriber(db, 6315)
    bot = deterministic_bot(db, fake_ai)
    phrase = "Напомни завтра в 9:00 по Светогорску позвонить врачу"
    context = reminder_context()
    initial = ReminderMessage(phrase)
    fake_ai.reminder_timezone_release.clear()
    provider_task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(initial, telegram_user_id=user.telegram_id, chat_id=9315),
            context,
        )
    )
    await fake_ai.reminder_timezone_started.wait()
    provider_input = fake_ai.reminder_timezone_calls[-1]
    fake_ai.reminder_timezone_results[provider_input] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по Светогорску",
    )
    if provider_outcome == "error":
        fake_ai.reminder_timezone_error = RuntimeError("PRIVATE_TRANSPORT_DETAIL")
    elif provider_outcome == "cancelled":
        fake_ai.reminder_timezone_error = asyncio.CancelledError()

    resolving = await current_session(bot, user, 9315)
    assert resolving is not None
    assert resolving.phase is ReminderFlowPhase.TIMEZONE_RESOLVING
    canonical = initial.replies[0]["message"]
    canonical_edits = list(canonical.edits)
    bot_edits = list(context.bot.edits)
    ordinary = ReminderMessage("позвонить маме после работы")
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        if source == "text":
            handled = await bot.reminder_text_gate(
                reminder_update(
                    ordinary,
                    telegram_user_id=user.telegram_id,
                    chat_id=9315,
                ),
                context,
            )
        else:
            handled = await bot.reminder_voice_gate(
                reminder_update(
                    ordinary,
                    telegram_user_id=user.telegram_id,
                    chat_id=9315,
                ),
                context,
                "позвонить маме после работы",
                ordinary,
                expected_access_version=resolving.access_version,
                expected_session=resolving,
            )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    assert handled is True
    assert await current_session(bot, user, 9315) == resolving
    assert canonical.edits == canonical_edits
    assert context.bot.edits == bot_edits
    assert ordinary.replies == []
    assert ordinary.deleted == (1 if source == "voice" else 0)
    assert fake_ai.reminder_timezone_calls == [provider_input]
    assert not any(
        statement.lstrip().split(maxsplit=1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}
        for statement in statements
        if statement.strip()
    )
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)

    fake_ai.reminder_timezone_release.set()
    if provider_outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await provider_task
    else:
        assert await provider_task is True
    finished = await current_session(bot, user, 9315)
    assert finished is not None
    assert finished.id == resolving.id
    if provider_outcome == "success":
        assert finished.phase is ReminderFlowPhase.PREVIEW
        assert finished.timezone == "Europe/Moscow"
    else:
        assert finished.phase is ReminderFlowPhase.TIMEZONE_RETRY
    assert fake_ai.reminder_timezone_calls == [provider_input]
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("old_provider_outcome", ["success", "error", "cancelled"])
@pytest.mark.asyncio
async def test_distinct_explicit_command_replaces_resolving_generation_and_stales_old_result(
    db,
    fake_ai,
    source,
    old_provider_outcome,
):
    user = await subscriber(db, 6316)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    old_phrase = "Напомни завтра в 9:00 по Светогорску старый приватный заголовок"
    old_message = ReminderMessage(old_phrase)
    old_provider_started = asyncio.Event()
    old_provider_release = asyncio.Event()
    new_provider_started = asyncio.Event()
    new_provider_release = asyncio.Event()

    async def controlled_timezone_resolution(fragment: str):
        fake_ai.reminder_timezone_calls.append(fragment)
        call_number = len(fake_ai.reminder_timezone_calls)
        if call_number == 1:
            old_provider_started.set()
            await old_provider_release.wait()
            if old_provider_outcome == "error":
                raise RuntimeError("PRIVATE_STALE_PROVIDER_ERROR")
            if old_provider_outcome == "cancelled":
                raise asyncio.CancelledError
        elif call_number == 2:
            new_provider_started.set()
            await new_provider_release.wait()
        else:
            raise AssertionError("one provider call is allowed per reminder generation")
        return ReminderTimezoneResolution(
            status="resolved",
            timezone="Europe/Moscow",
            matched_text="по Светогорску",
        )

    fake_ai.resolve_reminder_timezone = controlled_timezone_resolution
    old_task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(old_message, telegram_user_id=user.telegram_id, chat_id=9316),
            context,
        )
    )
    await old_provider_started.wait()
    old = await current_session(bot, user, 9316)
    assert old is not None
    assert old.phase is ReminderFlowPhase.TIMEZONE_RESOLVING

    new_phrase = "Напомни завтра в 11:00 по Светогорску новый заголовок"
    replacement_input = ReminderMessage(new_phrase if source == "text" else "Расшифровываю…")
    replacement_update = reminder_update(
        replacement_input,
        telegram_user_id=user.telegram_id,
        chat_id=9316,
    )
    if source == "text":
        replacement_task = asyncio.create_task(bot.reminder_text_gate(replacement_update, context))
    else:
        replacement_task = asyncio.create_task(
            bot.reminder_voice_gate(
                replacement_update,
                context,
                new_phrase,
                replacement_input,
                expected_access_version=old.access_version,
                expected_session=old,
            )
        )

    await asyncio.wait_for(new_provider_started.wait(), timeout=1)
    assert len(fake_ai.reminder_timezone_calls) == 2
    replacement = await current_session(bot, user, 9316)
    assert replacement is not None
    assert replacement.id != old.id
    assert replacement.phase is ReminderFlowPhase.TIMEZONE_RESOLVING
    assert replacement.canonical_message_id == old.canonical_message_id
    canonical = old_message.replies[0]["message"]
    edits_before_results = len(canonical.edits) + len(context.bot.edits)
    old_provider_input, new_provider_input = fake_ai.reminder_timezone_calls
    new_provider_release.set()
    assert await replacement_task is True
    completed_replacement = await current_session(bot, user, 9316)
    assert completed_replacement is not None
    assert completed_replacement.id == replacement.id
    assert completed_replacement.phase is ReminderFlowPhase.PREVIEW
    assert completed_replacement.title == "новый заголовок"
    assert completed_replacement.local_time == time(11)
    assert completed_replacement.timezone == "Europe/Moscow"
    assert len(canonical.edits) + len(context.bot.edits) == edits_before_results + 1
    edits_after_replacement = len(canonical.edits) + len(context.bot.edits)

    old_provider_release.set()
    if old_provider_outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await old_task
    else:
        assert await old_task is True
    live = await current_session(bot, user, 9316)
    assert live == completed_replacement
    assert len(canonical.edits) + len(context.bot.edits) == edits_after_replacement
    assert (
        "старый приватный заголовок"
        not in latest_reminder_render(
            canonical,
            context,
        )["text"]
    )
    assert fake_ai.reminder_timezone_calls == [old_provider_input, new_provider_input]
    assert replacement_input.replies == []
    assert replacement_input.deleted == (1 if source == "voice" else 0)
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)


@pytest.mark.parametrize("checkpoint", ["before", "during", "after"])
@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_timezone_ai_access_change_is_fail_closed_at_every_checkpoint(
    db,
    fake_ai,
    monkeypatch,
    checkpoint,
    change,
):
    user = await subscriber(db, 6308)
    bot = deterministic_bot(db, fake_ai)
    phrase = "Напомни завтра в 9:00 по Светогорску позвонить врачу"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по Светогорску",
    )
    if checkpoint == "before":
        original_store = bot._reminder_timezone_store_and_render

        async def store_then_change(*args, **kwargs):
            result = await original_store(*args, **kwargs)
            await change_reminder_access(db, user, change)
            return result

        monkeypatch.setattr(bot, "_reminder_timezone_store_and_render", store_then_change)
    elif checkpoint == "after":
        original_apply = bot._reminder_timezone_apply_result

        async def change_then_apply(*args, **kwargs):
            await change_reminder_access(db, user, change)
            return await original_apply(*args, **kwargs)

        monkeypatch.setattr(bot, "_reminder_timezone_apply_result", change_then_apply)
    else:
        fake_ai.reminder_timezone_release.clear()
    incoming = ReminderMessage(phrase)
    context = reminder_context()
    update = reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9308)

    if checkpoint == "during":
        task = asyncio.create_task(bot.reminder_text_gate(update, context))
        await fake_ai.reminder_timezone_started.wait()
        await change_reminder_access(db, user, change)
        fake_ai.reminder_timezone_release.set()
        assert await task is True
    else:
        assert await bot.reminder_text_gate(update, context) is True

    assert len(fake_ai.reminder_timezone_calls) == (0 if checkpoint == "before" else 1)
    assert await current_session(bot, user, 9308) is None
    texts = reminder_rendered_texts(incoming, context)
    assert any("Доступ изменился" in text for text in texts)
    assert all("позвонить врачу" not in text for text in texts)
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)


@pytest.mark.parametrize("stale_kind", ["replacement", "canonical"])
@pytest.mark.asyncio
async def test_stale_timezone_result_cannot_mutate_replacement_or_rebound_canonical(
    db,
    fake_ai,
    stale_kind,
):
    user = await subscriber(db, 6309)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    old_phrase = "Напомни завтра в 9:00 по Светогорску старый приватный title"
    fragment = reminder_timezone_window(old_phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по Светогорску",
    )
    fake_ai.reminder_timezone_release.clear()
    incoming = ReminderMessage(old_phrase)
    old_task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9309),
            context,
        )
    )
    await fake_ai.reminder_timezone_started.wait()
    old = await current_session(bot, user, 9309)
    assert old is not None

    if stale_kind == "replacement":
        replacement = await bot.reminder_sessions.create(
            owner_id=old.owner_id,
            telegram_user_id=old.telegram_user_id,
            chat_id=old.chat_id,
            access_version=old.access_version,
            title="новое напоминание",
            schedule_kind=ReminderScheduleKind.ONCE,
            local_date=date(2026, 8, 11),
            local_time=time(11),
            timezone="Europe/London",
            timezone_source=ReminderTimezoneSource.EXPLICIT,
            phase=ReminderFlowPhase.PREVIEW,
            canonical_message_id=old.canonical_message_id,
            profile_timezone=old.profile_timezone,
        )
        assert replacement.id != old.id
    else:
        replacement = await bot.reminder_sessions.update(
            old,
            canonical_message_id=(old.canonical_message_id or 0) + 10,
        )
        assert replacement is not None

    fake_ai.reminder_timezone_release.set()
    assert await old_task is True
    live = await current_session(bot, user, 9309)
    assert live is not None
    assert live.id == replacement.id
    assert live.version == replacement.version
    assert live.canonical_message_id == replacement.canonical_message_id
    if stale_kind == "replacement":
        assert live.title == "новое напоминание"
        assert live.local_time == time(11)
        assert live.timezone == "Europe/London"
    else:
        assert live.phase is ReminderFlowPhase.TIMEZONE_RESOLVING
        assert live.timezone == "Europe/Moscow"
    assert fake_ai.reminder_timezone_calls == [fragment]


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни завтра в 10:00 по времени Лондона в часовом поясе Берлина позвонить врачу",
        "Напомни завтра в 10:00 по Europe/London в часовом поясе Берлина позвонить врачу",
        "Напомни завтра в 10:00 Europe/London в часовом поясе Берлина позвонить врачу",
        "Напомни завтра в 10:00 МСК в часовом поясе Берлина позвонить врачу",
        "Напомни завтра в 10:00 Europe/London в часовом поясе Europe/London позвонить врачу",
        "Напомни завтра в 10:00 по Светогорску позвонить Europe/London",
    ],
)
@pytest.mark.asyncio
async def test_multiple_explicit_timezone_markers_fail_closed_without_ai_or_dml(
    db,
    fake_ai,
    phrase,
):
    user = await subscriber(db, 6310)
    bot = deterministic_bot(db, fake_ai)
    incoming = ReminderMessage(phrase)
    context = reminder_context()
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        assert await bot.reminder_text_gate(
            reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9310),
            context,
        )
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    session = await current_session(bot, user, 9310)
    assert session is not None
    assert session.phase is ReminderFlowPhase.TIMEZONE_RETRY
    assert session.title is None
    assert phrase not in repr(session)
    assert fake_ai.reminder_timezone_calls == []
    assert fake_ai.timezone_calls == []
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)
    assert not any(
        statement.lstrip().split(maxsplit=1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}
        for statement in statements
        if statement.strip()
    )
    canonical = incoming.replies[0]["message"]
    final_render = latest_reminder_render(canonical, context)
    assert "Не удалось надёжно определить часовой пояс" in final_render["text"]
    assert "Проверь напоминание" not in final_render["text"]
    assert "Europe/Moscow" not in final_render["text"]
    assert phrase not in final_render["text"]


@pytest.mark.parametrize(
    ("relative_day", "local_time", "expected_date"),
    [
        ("сегодня", "23:59", date(2026, 8, 10)),
        ("завтра", "10:00", date(2026, 8, 11)),
    ],
)
@pytest.mark.asyncio
async def test_model_timezone_relative_date_is_anchored_when_provider_crosses_midnight(
    db,
    fake_ai,
    relative_day,
    local_time,
    expected_date,
):
    user = await subscriber(db, 6311)
    initial = datetime(2026, 8, 10, 22, 58, tzinfo=UTC)  # 23:58 in London
    clock = [initial]
    bot = deterministic_bot(db, fake_ai)
    bot.reminder_intent_parser = ReminderIntentParser(now_provider=lambda: clock[0])
    bot._reminder_now_provider = lambda: clock[0]
    timezone_expression = "по времени Нортгемптону"
    phrase = f"Напомни {relative_day} в {local_time} {timezone_expression} позвонить врачу"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/London",
        matched_text=timezone_expression,
        city="Нортгемптон",
        country="Великобритания",
    )
    fake_ai.reminder_timezone_release.clear()
    incoming = ReminderMessage(phrase)

    task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9311),
            reminder_context(),
        )
    )
    await fake_ai.reminder_timezone_started.wait()
    pending = await current_session(bot, user, 9311)
    assert pending is not None
    assert pending.calendar_anchor_utc == initial
    clock[0] = datetime(2026, 8, 10, 23, 1, tzinfo=UTC)  # 00:01 next day in London
    fake_ai.reminder_timezone_release.set()
    assert await task is True

    session = await current_session(bot, user, 9311)
    assert session is not None
    assert session.local_date == expected_date
    assert session.timezone == "Europe/London"
    assert session.timezone_source is ReminderTimezoneSource.EXPLICIT
    assert session.calendar_anchor_utc == initial
    assert fake_ai.reminder_timezone_calls == [fragment]
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_duplicate_bare_timezone_clarification_is_single_flight_and_keeps_session(
    db,
    fake_ai,
):
    user = await subscriber(db, 6312)
    bot = deterministic_bot(db, fake_ai)
    context = reminder_context()
    initial_phrase = "Напомни завтра в 10:00 по Сан-Хосе созвониться с клиентом"
    fragment = reminder_timezone_window(initial_phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="ambiguous",
        matched_text="по Сан-Хосе",
    )
    incoming = ReminderMessage(initial_phrase)
    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9312),
        context,
    )
    clarify = await current_session(bot, user, 9312)
    assert clarify is not None
    assert clarify.phase is ReminderFlowPhase.TIMEZONE_CLARIFY

    answer = "Сан-Хосе, Калифорния, США"
    fake_ai.reminder_timezone_results[answer] = ReminderTimezoneResolution(
        status="resolved",
        timezone="America/Los_Angeles",
        matched_text=answer,
        city="Сан-Хосе",
        country="США",
    )
    fake_ai.reminder_timezone_started.clear()
    fake_ai.reminder_timezone_release.clear()
    first_answer = ReminderMessage(answer)
    first_task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(first_answer, telegram_user_id=user.telegram_id, chat_id=9312),
            context,
        )
    )
    await fake_ai.reminder_timezone_started.wait()
    resolving = await current_session(bot, user, 9312)
    assert resolving is not None
    assert resolving.id == clarify.id
    assert resolving.phase is ReminderFlowPhase.TIMEZONE_RESOLVING

    duplicate = ReminderMessage(answer)
    duplicate_task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(duplicate, telegram_user_id=user.telegram_id, chat_id=9312),
            context,
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.shield(duplicate_task), timeout=1) is True
        during_duplicate = await current_session(bot, user, 9312)
        assert during_duplicate is not None
        assert during_duplicate.id == resolving.id
        assert during_duplicate.version == resolving.version
        assert during_duplicate.phase is ReminderFlowPhase.TIMEZONE_RESOLVING
        assert fake_ai.reminder_timezone_calls == [fragment, answer]
        assert duplicate.replies == []
    finally:
        fake_ai.reminder_timezone_release.set()
        await asyncio.gather(first_task, duplicate_task, return_exceptions=True)

    resolved = await current_session(bot, user, 9312)
    assert resolved is not None
    assert resolved.id == clarify.id
    assert resolved.phase is ReminderFlowPhase.PREVIEW
    assert resolved.title == clarify.title
    assert resolved.local_date == clarify.local_date
    assert resolved.local_time == clarify.local_time
    assert resolved.timezone == "America/Los_Angeles"
    assert fake_ai.reminder_timezone_calls == [fragment, answer]


@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_access_change_during_preview_capability_issue_is_neutralized_before_edit(
    db,
    fake_ai,
    monkeypatch,
    change,
):
    user = await subscriber(db, 6313)
    bot = deterministic_bot(db, fake_ai)
    private_title = "PRIVATE_PRE_EDIT_TITLE_6a42"
    phrase = f"Напомни завтра в 9:00 по Светогорску {private_title}"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по Светогорску",
    )
    original_issue = bot.reminder_sessions.issue
    changed = False

    async def issue_then_change(session, actions, **kwargs):
        nonlocal changed
        tokens = await original_issue(session, actions, **kwargs)
        if actions == ("confirm", "edit", "cancel") and not changed:
            changed = True
            await change_reminder_access(db, user, change)
        return tokens

    monkeypatch.setattr(bot.reminder_sessions, "issue", issue_then_change)
    incoming = ReminderMessage(phrase)
    context = reminder_context()

    assert await bot.reminder_text_gate(
        reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9313),
        context,
    )

    assert changed is True
    assert fake_ai.reminder_timezone_calls == [fragment]
    assert await current_session(bot, user, 9313) is None
    texts = reminder_rendered_texts(incoming, context)
    assert any("Доступ изменился" in text for text in texts)
    assert all(private_title not in text for text in texts)
    assert all("Проверь напоминание" not in text for text in texts)
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)


@pytest.mark.parametrize("change", ["downgrade", "bounce"])
@pytest.mark.asyncio
async def test_access_change_during_pre_provider_exact_check_prevents_model_call(
    db,
    fake_ai,
    monkeypatch,
    change,
):
    user = await subscriber(db, 6314)
    bot = deterministic_bot(db, fake_ai)
    private_title = "PRIVATE_PRE_PROVIDER_TITLE_729b"
    phrase = f"Напомни завтра в 9:00 по Светогорску {private_title}"
    fragment = reminder_timezone_window(phrase)
    fake_ai.reminder_timezone_results[fragment] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по Светогорску",
    )
    original_get_exact = bot.reminder_sessions.get_exact
    exact_calls = 0
    exact_waiting = asyncio.Event()
    exact_release = asyncio.Event()

    async def pause_pre_provider_resolving_exact(session, **kwargs):
        nonlocal exact_calls
        result = await original_get_exact(session, **kwargs)
        if session.phase is ReminderFlowPhase.TIMEZONE_RESOLVING:
            exact_calls += 1
            if exact_calls == 4:
                exact_waiting.set()
                await exact_release.wait()
        return result

    monkeypatch.setattr(
        bot.reminder_sessions,
        "get_exact",
        pause_pre_provider_resolving_exact,
    )
    incoming = ReminderMessage(phrase)
    context = reminder_context()
    task = asyncio.create_task(
        bot.reminder_text_gate(
            reminder_update(incoming, telegram_user_id=user.telegram_id, chat_id=9314),
            context,
        )
    )
    await exact_waiting.wait()
    await change_reminder_access(db, user, change)
    exact_release.set()
    assert await task is True

    assert exact_calls >= 4
    assert fake_ai.reminder_timezone_calls == []
    assert await current_session(bot, user, 9314) is None
    texts = reminder_rendered_texts(incoming, context)
    assert any("Доступ изменился" in text for text in texts)
    assert all(private_title not in text for text in texts)
    assert await reminder_domain_row_counts(db) == (0, 0, 0, 0)
