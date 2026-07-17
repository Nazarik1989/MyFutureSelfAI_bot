from asyncio import gather
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.parse import urlparse

from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.doctor_search import (
    OFFICIAL_BOOKING_URL,
    DoctorSearchService,
)
from future_self.models import DraftInboxItem, InboxItem, TaskReminder


class NoopTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return ""


class SearchMessage:
    def __init__(self, text: str):
        self.text = text
        self.replies: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self


def search_settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def search_update(text: str, *, user_id: int = 800, chat_id: int = 1800):
    message = SearchMessage(text)
    return (
        SimpleNamespace(
            effective_message=message,
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=chat_id),
        ),
        message,
    )


def test_directory_is_official_therapist_route_svetogorsk_then_vyborg():
    directory = DoctorSearchService.directory()
    assert directory.country == "Россия"
    assert directory.specialty == "терапевт"
    assert [option.city for option in directory.options] == ["Светогорск", "Выборг"]
    assert directory.options[0].address == "г. Светогорск, ул. Пограничная, д. 13"
    assert directory.options[1].address == "г. Выборг, ул. Ильинская, д. 8"
    assert OFFICIAL_BOOKING_URL == "https://zdrav.lenreg.ru/"
    allowed_domains = {"mb.vbglenobl.ru", "lofoms.spb.ru", "zdrav.lenreg.ru"}
    for url in directory.official_sources:
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.hostname in allowed_domains
    assert all(option.specialty == "терапевт" for option in directory.options)


async def test_doctor_find_real_handler_is_read_only_and_never_calls_llm(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    update, message = search_update("/doctor_find")
    await bot.doctor_find(update, SimpleNamespace(user_data={}, args=[]))
    output = message.replies[-1]["text"]
    assert output.index("Светогорск") < output.index("Выборг")
    assert "терапевт" in output.lower()
    assert "122" in output
    assert "+7 (81378) 36-268" in output
    assert "+7 (81378) 2-83-46" in output
    assert "https://zdrav.lenreg.ru/" in output
    assert "официальн" in output.lower()
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


async def test_doctor_find_task_creates_one_private_safe_reminder_and_is_idempotent(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    context = SimpleNamespace(user_data={}, args=["через", "2", "часа"])
    update, message = search_update("/doctor_find_task через 2 часа")
    before = datetime.now(UTC)
    await bot.doctor_find_task(update, context)
    assert "Записаться к терапевту" in message.replies[-1]["text"]
    assert "Reminder" in message.replies[-1]["text"]
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        item = await session.scalar(select(InboxItem))
        reminder = await session.scalar(select(TaskReminder))
    assert item.title == "Записаться к терапевту: Светогорск → Выборг"
    assert item.source == "doctor_search"
    assert "симптом" not in f"{item.raw_text} {item.description}".lower()
    assert reminder.remind_at.replace(tzinfo=UTC) > before

    repeat, repeat_message = search_update("/doctor_find_task через 2 часа")
    await bot.doctor_find_task(repeat, context)
    assert "дубликат не добавлен" in repeat_message.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0


async def test_invalid_time_creates_nothing_and_tasks_are_owner_isolated(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    invalid_context = SimpleNamespace(user_data={}, args=["когда-нибудь"])
    invalid, invalid_message = search_update("/doctor_find_task когда-нибудь", user_id=801)
    await bot.doctor_find_task(invalid, invalid_context)
    assert "Не понял будущее время" in invalid_message.replies[-1]["text"]

    context = SimpleNamespace(user_data={}, args=["через", "3", "часа"])
    first, _ = search_update("/doctor_find_task через 3 часа", user_id=802, chat_id=1802)
    second, _ = search_update("/doctor_find_task через 3 часа", user_id=803, chat_id=1803)
    await bot.doctor_find_task(first, context)
    await bot.doctor_find_task(second, context)
    async with db.sessions() as session:
        items = list((await session.scalars(select(InboxItem).order_by(InboxItem.id))).all())
        reminders = list((await session.scalars(select(TaskReminder))).all())
    assert len(items) == 2
    assert len({item.user_id for item in items}) == 2
    assert {reminder.telegram_user_id for reminder in reminders} == {802, 803}


async def test_concurrent_doctor_search_task_creation_is_atomic(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    user = await bot._user(804)
    temporal = bot._doctor_task_temporal("через 2 часа", user.timezone)
    results = await gather(
        bot.doctor_search_service.create_booking_task(
            user_id=user.id,
            telegram_user_id=804,
            chat_id=1804,
            temporal=temporal,
        ),
        bot.doctor_search_service.create_booking_task(
            user_id=user.id,
            telegram_user_id=804,
            chat_id=1804,
            temporal=temporal,
        ),
    )
    assert {result.status for result in results} == {"created", "existing"}
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
