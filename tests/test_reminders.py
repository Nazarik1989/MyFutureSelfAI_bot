import asyncio
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.dates import DateResolver
from future_self.drafts import DraftInboxService
from future_self.models import DraftInboxItem, InboxItem, TaskReminder
from future_self.reminders import TaskReminderEngine, as_utc, schedule_from_temporal
from future_self.repositories import UserRepository
from future_self.scheduler import JobQueueScheduler
from future_self.schemas import ParsedThought, TemporalResolution


class RouteMessage:
    _next_id = 500

    def __init__(self, text: str | None = None, *, voice=None):
        self.text = text
        self.voice = voice
        self.audio = None
        self.message_id = self._next_id
        RouteMessage._next_id += 1
        self.replies: list[dict[str, object]] = []
        self.edits: list[str] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self

    async def edit_text(self, text: str):
        self.edits.append(text)


class RouteFile:
    async def download_as_bytearray(self):
        return bytearray(b"voice")


class RouteVoice:
    duration = 2
    file_size = 5
    mime_type = "audio/ogg"
    file_name = "voice.ogg"

    async def get_file(self):
        return RouteFile()


class PhraseTranscription:
    enabled = True

    def __init__(self, phrase: str):
        self.phrase = phrase

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return self.phrase


class RouteBot:
    async def edit_message_reply_markup(self, **kwargs):
        return None


def route_settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
        transcription_provider="disabled",
    )


def route_update(message: RouteMessage, user_id: int, chat_id: int):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
    )


def temporal(
    *,
    resolved_at: datetime,
    timezone: str = "Europe/Moscow",
    local_date: date = date(2026, 7, 20),
    local_time: time | None = time(18, 0),
    precision: str = "datetime",
) -> TemporalResolution:
    return TemporalResolution(
        resolved_at=resolved_at,
        timezone=timezone,
        resolved_local_date=local_date,
        resolved_local_time=local_time,
        precision=precision,
        original_expression="в понедельник вечером",
    )


async def create_reminder(
    db,
    *,
    telegram_user_id: int = 10,
    chat_id: int = 100,
    kind: str = "task",
    temporal_resolution: TemporalResolution | None = None,
    lead_minutes: int = 30,
) -> tuple[InboxItem, TaskReminder | None]:
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(telegram_user_id, "Europe/Moscow")
        user_id = user.id
    service = DraftInboxService(
        db,
        60,
        task_date_event_hour=9,
        task_reminder_lead_minutes=lead_minutes,
    )
    draft = await service.create(
        user_id=user_id,
        telegram_user_id=telegram_user_id,
        chat_id=chat_id,
        source="voice",
        raw_text="Записаться к врачу в понедельник в 18",
        parsed=ParsedThought(
            kind=kind,
            title="Записаться к врачу",
            description="Плановый приём",
            resolved_date=temporal_resolution.resolved_local_date if temporal_resolution else None,
            temporal_resolution=temporal_resolution,
        ),
    )
    result = await service.confirm(
        draft.id,
        draft.version,
        telegram_user_id,
        chat_id,
    )
    assert result.ok
    return result.inbox_item, result.reminder


def test_datetime_schedule_keeps_event_and_reminder_separate_in_utc():
    event = datetime(2026, 7, 20, 15, tzinfo=UTC)
    result = schedule_from_temporal(
        temporal(resolved_at=event).model_dump(mode="json"),
        date_event_hour=9,
        lead_minutes=45,
    )
    assert result.event_at == event
    assert result.remind_at == event - timedelta(minutes=45)
    assert result.timezone == "Europe/Moscow"


def test_date_only_schedule_uses_configured_local_hour_across_timezone():
    result = schedule_from_temporal(
        temporal(
            resolved_at=datetime(2026, 1, 15, 5, tzinfo=UTC),
            timezone="America/New_York",
            local_date=date(2026, 1, 15),
            local_time=None,
            precision="date",
        ).model_dump(mode="json"),
        date_event_hour=9,
        lead_minutes=30,
    )
    assert result.event_at == datetime(2026, 1, 15, 14, tzinfo=UTC)
    assert result.remind_at == datetime(2026, 1, 15, 13, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("phrase", "expected_delta", "expected_title"),
    [
        ("Напомни через 5 минут выпить воды", timedelta(minutes=5), "Выпить воды"),
        ("напомни через 1 час проверить духовку!", timedelta(hours=1), "Проверить духовку"),
        ("Напомни через час, проверить почту", timedelta(hours=1), "Проверить почту"),
        ("Напомни через минуту сделать вдох", timedelta(minutes=1), "Сделать вдох"),
        ("Напомни мне позвонить врачу через 2 часа", timedelta(hours=2), "Позвонить врачу"),
        ("НАПОМНИ ЧЕРЕЗ ПЯТЬ МИНУТ размяться", timedelta(minutes=5), "Размяться"),
    ],
)
def test_relative_reminder_parser_uses_exact_interval(phrase, expected_delta, expected_title):
    now = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
    result = DateResolver(now_provider=lambda: now).resolve_relative_reminder(
        phrase,
        "Europe/Moscow",
    )
    assert result is not None
    assert result.remind_at == now + expected_delta
    assert result.title == expected_title
    assert result.temporal.remind_at == now + expected_delta
    assert result.temporal.resolved_at == now + expected_delta


@pytest.mark.parametrize(
    "phrase",
    [
        "Не напоминай через 5 минут пить воду",
        "Напоминание через 5 минут",
        "Через 5 минут я выпью воду",
        "Напомни через 0 минут выпить воды",
        "Напомни через 999 часов выпить воды",
    ],
)
def test_non_commands_and_unsafe_relative_intervals_are_not_intercepted(phrase):
    assert (
        DateResolver(
            now_provider=lambda: datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
        ).resolve_relative_reminder(phrase, "Europe/Moscow")
        is None
    )


async def test_confirm_task_with_temporal_data_creates_one_persistent_reminder(db):
    item, reminder = await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    assert reminder is not None
    assert reminder.inbox_item_id == item.id
    assert as_utc(reminder.event_at) == datetime(2026, 7, 20, 15, tzinfo=UTC)
    assert as_utc(reminder.remind_at) == datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1


async def test_non_task_and_undated_task_do_not_create_reminders(db):
    await create_reminder(
        db,
        kind="idea",
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    await create_reminder(db, telegram_user_id=11, chat_id=101)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_real_text_and_voice_routing_persists_task_reminder(db, fake_ai, source):
    phrase = "Нужно сделать отчёт завтра в 18"
    bot = FutureSelfBot(route_settings(), db, fake_ai, PhraseTranscription(phrase))
    user_id = 801 if source == "text" else 802
    chat_id = user_id + 10_000
    if source == "text":
        capture = RouteMessage(phrase)
        await bot.text(
            route_update(capture, user_id, chat_id),
            SimpleNamespace(user_data={}, bot=RouteBot()),
        )
    else:
        capture = RouteMessage(voice=RouteVoice())
        await bot.voice(
            route_update(capture, user_id, chat_id),
            SimpleNamespace(user_data={}, bot=RouteBot()),
        )
    save = RouteMessage("сохрани")
    await bot.text(
        route_update(save, user_id, chat_id),
        SimpleNamespace(user_data={}, bot=RouteBot()),
    )
    repeated_save = RouteMessage("сохрани")
    await bot.text(
        route_update(repeated_save, user_id, chat_id),
        SimpleNamespace(user_data={}, bot=RouteBot()),
    )
    async with db.sessions() as session:
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.telegram_user_id == user_id)
        )
        reminder_count = await session.scalar(
            select(func.count(TaskReminder.id)).where(TaskReminder.telegram_user_id == user_id)
        )
        item = await session.get(InboxItem, reminder.inbox_item_id)
        inbox_count = await session.scalar(
            select(func.count(InboxItem.id)).where(InboxItem.user_id == item.user_id)
        )
    assert reminder is not None
    assert reminder_count == 1
    assert inbox_count == 1
    assert as_utc(reminder.event_at) - as_utc(reminder.remind_at) == timedelta(minutes=30)
    assert reminder.timezone == "Europe/Moscow"
    assert "Напоминание:" in save.replies[-1]["text"]


@pytest.mark.parametrize(
    ("source", "phrase", "delta"),
    [
        ("text", "Напомни через 5 минут выпить воды", timedelta(minutes=5)),
        ("voice", "Напомни через 5 минут выпить воды", timedelta(minutes=5)),
        ("text", "Напомни через 2 часа проверить духовку", timedelta(hours=2)),
        ("voice", "Напомни через 2 часа проверить духовку", timedelta(hours=2)),
    ],
)
async def test_relative_reminder_text_and_voice_route_save_and_deliver(
    db,
    fake_ai,
    source,
    phrase,
    delta,
):
    now = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
    bot = FutureSelfBot(route_settings(), db, fake_ai, PhraseTranscription(phrase))
    bot.date_resolver = DateResolver(now_provider=lambda: now)
    user_id = 901 + len(fake_ai.route_calls)
    chat_id = user_id + 10_000
    capture = RouteMessage(phrase if source == "text" else None, voice=RouteVoice())
    context = SimpleNamespace(user_data={}, bot=RouteBot())
    if source == "text":
        await bot.text(route_update(capture, user_id, chat_id), context)
    else:
        await bot.voice(route_update(capture, user_id, chat_id), context)
    assert fake_ai.route_calls == []

    async with db.sessions() as session:
        draft = await session.scalar(
            select(DraftInboxItem).where(DraftInboxItem.telegram_user_id == user_id)
        )
    assert draft.kind == "task"
    assert draft.temporal_resolution["remind_at"] == (now + delta).isoformat().replace(
        "+00:00", "Z"
    )

    save = RouteMessage("сохрани")
    await bot.text(route_update(save, user_id, chat_id), context)
    async with db.sessions() as session:
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.telegram_user_id == user_id)
        )
    assert as_utc(reminder.remind_at) == now + delta
    assert as_utc(reminder.event_at) == now + delta

    sent: list[tuple[int, str]] = []

    async def send(target_chat_id: int, text: str) -> int:
        sent.append((target_chat_id, text))
        return 700

    engine = TaskReminderEngine(db, send)
    assert await engine.deliver_due(now=now + delta - timedelta(seconds=1)) == 0
    assert await engine.deliver_due(now=now + delta) == 1
    assert await engine.deliver_due(now=now + delta + timedelta(seconds=1)) == 0
    assert len(sent) == 1
    assert sent[0][0] == chat_id


async def test_due_reminder_is_delivered_to_original_chat_and_marked_sent(db):
    _, reminder = await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> int:
        sent.append((chat_id, text))
        return 321

    engine = TaskReminderEngine(db, send)
    delivered = await engine.deliver_due(now=datetime(2026, 7, 20, 14, 30, tzinfo=UTC))
    assert delivered == 1
    assert len(sent) == 1
    assert sent[0][0] == 100
    assert "Записаться к врачу" in sent[0][1]
    assert "20.07.2026 18:00 (Europe/Moscow)" in sent[0][1]
    async with db.sessions() as session:
        saved = await session.get(TaskReminder, reminder.id)
    assert saved.status == "sent"
    assert saved.telegram_message_id == 321
    assert saved.attempt_count == 1


async def test_future_reminder_is_not_delivered(db):
    await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    sent: list[str] = []

    async def send(chat_id: int, text: str) -> int:
        sent.append(text)
        return 1

    delivered = await TaskReminderEngine(db, send).deliver_due(
        now=datetime(2026, 7, 20, 14, 29, 59, tzinfo=UTC)
    )
    assert delivered == 0
    assert sent == []


async def test_repeated_poll_does_not_duplicate_telegram_delivery(db):
    await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    sent: list[str] = []

    async def send(chat_id: int, text: str) -> int:
        sent.append(text)
        return len(sent)

    engine = TaskReminderEngine(db, send)
    now = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    assert await engine.deliver_due(now=now) == 1
    assert await engine.deliver_due(now=now + timedelta(minutes=1)) == 0
    assert len(sent) == 1


async def test_parallel_workers_claim_a_due_reminder_only_once(db):
    await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    sent: list[str] = []

    async def send(chat_id: int, text: str) -> int:
        sent.append(text)
        await asyncio.sleep(0)
        return 1

    now = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    results = await asyncio.gather(
        TaskReminderEngine(db, send).deliver_due(now=now),
        TaskReminderEngine(db, send).deliver_due(now=now),
    )
    assert sum(results) == 1
    assert len(sent) == 1


async def test_new_engine_instance_delivers_pending_state_after_restart(db):
    await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    sent: list[str] = []

    async def send(chat_id: int, text: str) -> int:
        sent.append(text)
        return 77

    restarted_engine = TaskReminderEngine(db, send)
    assert await restarted_engine.deliver_due(now=datetime(2026, 7, 20, 14, 31, tzinfo=UTC)) == 1
    assert len(sent) == 1


async def test_restart_reconciles_future_tasks_saved_before_engine_upgrade(db):
    _, reminder = await create_reminder(
        db,
        temporal_resolution=temporal(
            resolved_at=datetime(2027, 7, 20, 15, tzinfo=UTC),
            local_date=date(2027, 7, 20),
        ),
    )
    async with db.session() as session:
        persisted = await session.get(TaskReminder, reminder.id)
        await session.delete(persisted)

    async def send(chat_id: int, text: str) -> int:
        return 1

    engine = TaskReminderEngine(db, send)
    now = datetime(2026, 7, 17, tzinfo=UTC)
    assert await engine.reconcile_missing(now=now) == 1
    assert await engine.reconcile_missing(now=now) == 0
    async with db.sessions() as session:
        restored = await session.scalar(select(TaskReminder))
    assert restored is not None
    assert as_utc(restored.event_at) == datetime(2027, 7, 20, 15, tzinfo=UTC)


async def test_stale_claim_is_recovered_but_fresh_claim_is_not(db):
    _, stale = await create_reminder(
        db,
        telegram_user_id=10,
        chat_id=100,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    _, fresh = await create_reminder(
        db,
        telegram_user_id=11,
        chat_id=101,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    now = datetime(2026, 7, 20, 14, 35, tzinfo=UTC)
    async with db.session() as session:
        stale_row = await session.get(TaskReminder, stale.id)
        stale_row.status = "processing"
        stale_row.claim_token = "abandoned"
        stale_row.claimed_at = now - timedelta(minutes=3)
        fresh_row = await session.get(TaskReminder, fresh.id)
        fresh_row.status = "processing"
        fresh_row.claim_token = "active"
        fresh_row.claimed_at = now - timedelta(seconds=30)
    sent: list[int] = []

    async def send(chat_id: int, text: str) -> int:
        sent.append(chat_id)
        return chat_id

    assert await TaskReminderEngine(db, send, lease_seconds=120).deliver_due(now=now) == 1
    assert sent == [100]
    async with db.sessions() as session:
        fresh_row = await session.get(TaskReminder, fresh.id)
    assert fresh_row.status == "processing"
    assert fresh_row.claim_token == "active"


async def test_delivery_failure_is_retried_with_backoff_without_leaking_error(db):
    _, reminder = await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2026, 7, 20, 15, tzinfo=UTC)),
    )
    calls = 0

    async def fail(chat_id: int, text: str) -> int:
        nonlocal calls
        calls += 1
        raise RuntimeError("private Telegram response")

    engine = TaskReminderEngine(db, fail)
    due = datetime(2026, 7, 20, 14, 30, tzinfo=UTC)
    assert await engine.deliver_due(now=due) == 0
    assert await engine.deliver_due(now=due + timedelta(seconds=4)) == 0
    assert calls == 1
    async with db.sessions() as session:
        saved = await session.get(TaskReminder, reminder.id)
    assert saved.status == "pending"
    assert saved.last_error_type == "RuntimeError"
    assert as_utc(saved.next_attempt_at) == due + timedelta(seconds=5)


async def test_bot_startup_delivers_persisted_due_reminder_via_telegram(db, fake_ai):
    _, reminder = await create_reminder(
        db,
        temporal_resolution=temporal(resolved_at=datetime(2020, 7, 20, 15, tzinfo=UTC)),
    )
    sent: list[tuple[int, str]] = []
    repeating: list[dict[str, object]] = []

    class TelegramBot:
        async def send_message(self, *, chat_id: int, text: str):
            sent.append((chat_id, text))
            return SimpleNamespace(message_id=900)

    class Queue:
        def run_repeating(self, callback, **kwargs):
            repeating.append({"callback": callback, **kwargs})

    bot = FutureSelfBot(
        route_settings(),
        db,
        fake_ai,
        PhraseTranscription(""),
    )
    await bot._post_init(SimpleNamespace(bot=TelegramBot(), job_queue=Queue()))
    assert len(sent) == 1
    assert sent[0][0] == 100
    assert len(repeating) == 1
    async with db.sessions() as session:
        saved = await session.get(TaskReminder, reminder.id)
    assert saved.status == "sent"
    assert saved.telegram_message_id == 900


def test_scheduler_registers_single_persistent_outbox_poller():
    calls: list[dict[str, object]] = []

    class FakeQueue:
        def run_repeating(self, callback, **kwargs):
            calls.append({"callback": callback, **kwargs})

    async def send(chat_id: int, text: str) -> int:
        return 1

    scheduler = JobQueueScheduler(FakeQueue(), send, 8, 21, 6)
    engine = SimpleNamespace(deliver_due=None)
    scheduler.start_task_reminders(engine, interval_seconds=15)
    assert len(calls) == 1
    assert calls[0]["interval"] == 15
    assert calls[0]["first"] == 15
    assert calls[0]["name"] == "task-reminders:persistent-outbox"
