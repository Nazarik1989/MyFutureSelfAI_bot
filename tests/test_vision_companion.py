from datetime import UTC, datetime, time
from types import SimpleNamespace

from autotester.fakes import FakeCallbackQuery, FakeMessage, ScriptedTranscription
from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import VisionCompanionCheckIn, VisionItem
from future_self.repositories import UserRepository
from future_self.scheduler import JobQueueScheduler
from future_self.vision_companion import VisionCompanionService, companion_extra_times


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
        morning_hour=8,
        evening_hour=20,
    )


def callback_update(data: str, message: FakeMessage, *, user_id: int, chat_id: int):
    query = FakeCallbackQuery(data, message)
    return (
        SimpleNamespace(
            effective_message=message,
            callback_query=query,
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=chat_id, type="private"),
        ),
        query,
    )


def callback_from(message: FakeMessage, prefix: str) -> str:
    for reply in reversed(message.replies):
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        for row in markup.inline_keyboard:
            for button in row:
                if button.callback_data and button.callback_data.startswith(prefix):
                    return button.callback_data
    raise AssertionError(f"Missing callback {prefix!r}")


async def vision_item(db, *, telegram_id: int = 501, wish: str = "Выступить уверенно"):
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(telegram_id, "Europe/Moscow")
        item = VisionItem(
            owner_id=owner.id,
            category="growth_creativity",
            wish_text=wish,
            why_text="Хочу делиться идеями",
            first_step="Подготовить первые три слайда",
            status="active",
        )
        session.add(item)
        await session.flush()
        return owner.id, item.id


def test_extra_times_are_evenly_spaced_between_morning_and_evening():
    assert companion_extra_times(time(8), time(20), 0) == []
    assert companion_extra_times(time(8), time(20), 1) == [time(14)]
    assert companion_extra_times(time(8), time(20), 2) == [time(12), time(16)]
    assert companion_extra_times(time(8), time(20), 3) == [time(11), time(14), time(17)]


async def test_companion_is_opt_in_single_focus_and_checkins_are_idempotent(db):
    service = VisionCompanionService(db)
    owner_id, first_id = await vision_item(db)
    _, second_id = await vision_item(db, telegram_id=501, wish="Пробежать пять километров")

    preference = await service.enable(
        owner_id=owner_id,
        item_id=first_id,
        telegram_user_id=501,
        chat_id=501,
        timezone="Europe/Moscow",
        morning_time=time(8),
        evening_time=time(20),
    )
    assert preference is not None
    assert preference.extra_times == []

    preference = await service.set_frequency(owner_id, first_id, 2)
    assert preference is not None
    assert preference.extra_times == ["12:00", "16:00"]

    switched = await service.enable(
        owner_id=owner_id,
        item_id=second_id,
        telegram_user_id=501,
        chat_id=501,
        timezone="Europe/Moscow",
        morning_time=time(8),
        evening_time=time(20),
    )
    assert switched is not None
    assert switched.id == preference.id
    assert switched.vision_item_id == second_id

    now = datetime(2026, 7, 31, 6, tzinfo=UTC)
    first = await service.record(
        owner_id=owner_id,
        item_id=second_id,
        moment="morning",
        response="committed",
        now=now,
    )
    repeated = await service.record(
        owner_id=owner_id,
        item_id=second_id,
        moment="morning",
        response="pause",
        now=now,
    )
    assert first is not None and repeated is not None and first.id == repeated.id
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(VisionCompanionCheckIn.id))) == 1
        stored = await session.get(VisionCompanionCheckIn, first.id)
        assert stored.response == "pause"


def test_scheduler_replaces_generic_daily_messages_and_schedules_selected_frequency():
    scheduled: list[dict[str, object]] = []
    removed: list[str] = []

    class Job:
        def __init__(self, name: str):
            self.name = name

        def schedule_removal(self):
            removed.append(self.name)

    class Queue:
        def jobs(self):
            return []

        def get_jobs_by_name(self, name):
            return [Job(name)] if name.endswith((":morning", ":evening")) else []

        def run_daily(self, callback, **kwargs):
            scheduled.append({"callback": callback, **kwargs})

    delivered: list[tuple[int, str]] = []

    async def send(_chat_id: int, _text: str):
        return 1

    async def vision_send(preference_id: int, moment: str):
        delivered.append((preference_id, moment))

    scheduler = JobQueueScheduler(Queue(), send, 8, 21, 6, True, vision_send)
    preference = SimpleNamespace(
        id=9,
        owner_id=7,
        telegram_user_id=501,
        timezone="Europe/Moscow",
        morning_time=time(8),
        evening_time=time(20),
        extra_times=["12:00", "16:00"],
    )
    scheduler.schedule_vision_companion(preference)

    assert removed == ["user:501:morning", "user:501:evening"]
    assert [job["name"] for job in scheduled] == [
        "vision-companion:7:morning",
        "vision-companion:7:evening",
        "vision-companion:7:extra-1",
        "vision-companion:7:extra-2",
    ]
    assert all(job["time"].tzinfo.key == "Europe/Moscow" for job in scheduled)


async def test_inactive_item_disables_future_companion_delivery(db):
    service = VisionCompanionService(db)
    owner_id, item_id = await vision_item(db, telegram_id=502)
    preference = await service.enable(
        owner_id=owner_id,
        item_id=item_id,
        telegram_user_id=502,
        chat_id=502,
        timezone="Europe/Moscow",
        morning_time=time(8),
        evening_time=time(20),
    )
    assert preference is not None
    async with db.session() as session:
        item = await session.get(VisionItem, item_id)
        item.status = "achieved"

    assert await service.snapshot(preference.id) is None
    stored = await service.get(owner_id)
    assert stored is not None and stored.enabled is False


async def test_card_offers_opt_in_and_frequency_without_starting_image_generation(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner_id, item_id = await vision_item(db, telegram_id=503)
    item = await bot.vision_service.get_item(owner_id, item_id)
    message = FakeMessage()
    await bot._vision_send_item(message, item)

    open_update, _ = callback_update(
        callback_from(message, "vision:companion:"), message, user_id=503, chat_id=503
    )
    await bot.vision_action(open_update, None)
    enable_update, _ = callback_update(
        callback_from(message, "vision:companionon:"), message, user_id=503, chat_id=503
    )
    await bot.vision_action(enable_update, None)
    frequency_update, _ = callback_update(
        callback_from(message, "vision:companionfreq:")[:-1] + "2",
        message,
        user_id=503,
        chat_id=503,
    )
    await bot.vision_action(frequency_update, None)

    preference = await bot.vision_companion_service.get(owner_id)
    assert preference is not None
    assert preference.enabled is True
    assert preference.extra_times == ["12:00", "16:00"]
    assert message.reply_text_calls == 1
