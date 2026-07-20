from asyncio import gather
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.parse import urlparse

from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.doctor_search import (
    GOSUSLUGI_BOOKING_URL,
    SARATOV_BOOKING_URL,
    DoctorSearchService,
)
from future_self.location import UserLocation
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


async def set_location(bot: FutureSelfBot, telegram_id: int, value: str) -> None:
    user = await bot._user(telegram_id)
    await bot.location_service.set(
        user_id=user.id,
        telegram_user_id=telegram_id,
        value=value,
    )


def test_directory_is_location_specific_and_uses_official_sources():
    route = DoctorSearchService.directory(UserLocation("Светогорск", "Выборг"))
    assert [option.city for option in route.options] == ["Светогорск", "Выборг"]
    assert route.options[0].address == "г. Светогорск, ул. Пограничная, д. 13"
    assert route.options[1].address == "г. Выборг, ул. Ильинская, д. 8"
    allowed_domains = {"mb.vbglenobl.ru", "zdrav.lenreg.ru"}
    for url in route.official_sources:
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.hostname in allowed_domains

    saratov = DoctorSearchService.directory(UserLocation("Саратов"))
    assert saratov.options == ()
    assert saratov.booking_url == SARATOV_BOOKING_URL
    assert GOSUSLUGI_BOOKING_URL in saratov.official_sources
    assert "Саратов" in saratov.task_title
    assert "Светогорск" not in saratov.task_title


async def test_doctor_find_uses_each_owner_location_and_is_read_only(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    await set_location(bot, 800, "Светогорск → Выборг")
    await set_location(bot, 801, "Саратов")

    first, first_message = search_update("/doctor_find", user_id=800)
    second, second_message = search_update("/doctor_find", user_id=801, chat_id=1801)
    await bot.doctor_find(first, SimpleNamespace(user_data={}, args=[]))
    await bot.doctor_find(second, SimpleNamespace(user_data={}, args=[]))

    first_output = first_message.replies[-1]["text"]
    second_output = second_message.replies[-1]["text"]
    assert first_output.index("Светогорск") < first_output.index("Выборг")
    assert "+7 (81378) 36-268" in first_output
    assert "Саратов" not in first_output
    assert "Саратов" in second_output
    assert SARATOV_BOOKING_URL in second_output
    assert "Светогорск" not in second_output
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


async def test_doctor_find_requires_owner_location(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    update, message = search_update("/doctor_find")
    await bot.doctor_find(update, SimpleNamespace(user_data={}, args=[]))
    assert "/location" in message.replies[-1]["text"]
    assert "Светогорск" not in message.replies[-1]["text"]


async def test_doctor_find_task_creates_private_location_task_idempotently(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    await set_location(bot, 800, "Саратов")
    context = SimpleNamespace(user_data={}, args=["через", "2", "часа"])
    update, message = search_update("/doctor_find_task через 2 часа")
    before = datetime.now(UTC)
    await bot.doctor_find_task(update, context)
    assert "Записаться к терапевту: Саратов" in message.replies[-1]["text"]
    assert "Reminder" in message.replies[-1]["text"]
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        item = await session.scalar(select(InboxItem))
        reminder = await session.scalar(select(TaskReminder))
    assert item.title == "Записаться к терапевту: Саратов"
    assert item.source == "doctor_search"
    assert "Саратов" in item.description
    assert "Светогорск" not in f"{item.title} {item.description}"
    assert "симптом" not in f"{item.raw_text} {item.description}".lower()
    assert reminder.remind_at.replace(tzinfo=UTC) > before

    repeat, repeat_message = search_update("/doctor_find_task через 2 часа")
    await bot.doctor_find_task(repeat, context)
    assert "дубликат не добавлен" in repeat_message.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0


async def test_invalid_time_and_tasks_are_owner_isolated_by_location(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    await set_location(bot, 801, "Саратов")
    invalid_context = SimpleNamespace(user_data={}, args=["когда-нибудь"])
    invalid, invalid_message = search_update("/doctor_find_task когда-нибудь", user_id=801)
    await bot.doctor_find_task(invalid, invalid_context)
    assert "Не понял будущее время" in invalid_message.replies[-1]["text"]

    await set_location(bot, 802, "Светогорск → Выборг")
    await set_location(bot, 803, "Саратов")
    context = SimpleNamespace(user_data={}, args=["через", "3", "часа"])
    first, _ = search_update("/doctor_find_task через 3 часа", user_id=802, chat_id=1802)
    second, _ = search_update("/doctor_find_task через 3 часа", user_id=803, chat_id=1803)
    await bot.doctor_find_task(first, context)
    await bot.doctor_find_task(second, context)
    async with db.sessions() as session:
        items = list((await session.scalars(select(InboxItem).order_by(InboxItem.id))).all())
        reminders = list((await session.scalars(select(TaskReminder))).all())
    assert [item.title for item in items] == [
        "Записаться к терапевту: Светогорск → Выборг",
        "Записаться к терапевту: Саратов",
    ]
    assert len({item.user_id for item in items}) == 2
    assert {reminder.telegram_user_id for reminder in reminders} == {802, 803}


async def test_concurrent_doctor_search_task_creation_is_atomic(db, fake_ai):
    bot = FutureSelfBot(search_settings(), db, fake_ai, NoopTranscription())
    user = await bot._user(804)
    await set_location(bot, 804, "Саратов")
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
