import logging
from datetime import time
from types import SimpleNamespace

from telegram.error import TelegramError

from future_self.access import ADMIN, BLOCKED, GUEST, SUBSCRIBER, is_full_access_tier
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.repositories import UserRepository
from future_self.scheduler import JobQueueScheduler


class NoopTranscription:
    enabled = False


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
        transcription_provider="disabled",
        enable_task_reminders=False,
    )


async def test_scheduler_checks_access_immediately_before_each_generic_send():
    one_off: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    sent: list[tuple[int, str]] = []
    checked: list[int] = []
    access = {101: True, 102: True}

    class Queue:
        def get_jobs_by_name(self, name):
            return []

        def run_once(self, callback, **kwargs):
            one_off.append({"callback": callback, **kwargs})

        def run_daily(self, callback, **kwargs):
            daily.append({"callback": callback, **kwargs})

    async def send(chat_id: int, text: str) -> int:
        sent.append((chat_id, text))
        return 1

    async def can_send(telegram_id: int) -> bool:
        checked.append(telegram_id)
        return access[telegram_id]

    scheduler = JobQueueScheduler(Queue(), send, 8, 21, 6, can_send=can_send)
    scheduler.schedule_user(101, "Europe/Moscow")
    access[101] = False
    initial = list(one_off)
    assert len(initial) == 3
    for job in initial:
        await job["callback"](SimpleNamespace(job=SimpleNamespace(data=job["data"])))

    # A denied daily/weekly run skips Telegram but schedules its next occurrence.
    assert sent == []
    assert checked == [101, 101, 101]
    assert len(one_off) == 6

    scheduler.schedule_health_reminder(
        user_id=5,
        chat_id=102,
        timezone="Europe/Moscow",
        local_time=time(20),
    )
    access[102] = False
    health_job = daily[0]
    await health_job["callback"](SimpleNamespace(job=SimpleNamespace(data=health_job["data"])))
    assert sent == []
    assert checked[-1] == 102
    assert len(daily) == 1


async def test_scheduler_allows_full_tiers_and_preserves_default_allow_behavior():
    scheduled: list[dict[str, object]] = []
    sent: list[int] = []
    tiers = {201: GUEST, 202: SUBSCRIBER, 203: ADMIN, 204: BLOCKED}

    class Queue:
        def run_once(self, callback, **kwargs):
            scheduled.append(kwargs)

    async def send(chat_id: int, text: str) -> int:
        sent.append(chat_id)
        return 1

    async def can_send(telegram_id: int) -> bool:
        return is_full_access_tier(tiers[telegram_id])

    scheduler = JobQueueScheduler(Queue(), send, 8, 21, 6, can_send=can_send)
    for telegram_id in tiers:
        context = SimpleNamespace(
            job=SimpleNamespace(data={"telegram_id": telegram_id, "timezone": "Europe/Moscow"})
        )
        await scheduler._morning(context)
        await scheduler._evening(context)
        await scheduler._weekly(context)
        await scheduler._health_checkin(
            SimpleNamespace(job=SimpleNamespace(data={"chat_id": telegram_id}))
        )

    assert sent == [202, 202, 202, 202, 203, 203, 203, 203]
    assert len(scheduled) == 12

    default_scheduler = JobQueueScheduler(Queue(), send, 8, 21, 6)
    await default_scheduler._health_checkin(
        SimpleNamespace(job=SimpleNamespace(data={"chat_id": 205}))
    )
    assert sent[-1] == 205


async def test_post_init_isolates_global_telegram_errors_and_initializes_scheduler(
    db, fake_ai, monkeypatch, caplog
):
    calls: list[str] = []
    repeating: list[str] = []
    one_off: list[str] = []
    sends: list[int] = []

    class TelegramBot:
        async def set_my_commands(self, commands, **kwargs):
            calls.append("commands")
            raise TelegramError("commands provider detail")

        async def set_chat_menu_button(self, **kwargs):
            calls.append("menu")
            raise TelegramError("menu provider detail")

        async def send_message(self, *, chat_id, text):
            sends.append(chat_id)
            return SimpleNamespace(message_id=1)

    class Queue:
        def get_jobs_by_name(self, name):
            return []

        def jobs(self):
            return []

        def run_once(self, callback, **kwargs):
            one_off.append(kwargs["name"])

        def run_daily(self, callback, **kwargs):
            raise AssertionError("empty database must not schedule a preference")

        def run_repeating(self, callback, **kwargs):
            repeating.append(kwargs["name"])

    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    with caplog.at_level(logging.ERROR):
        await bot._post_init(SimpleNamespace(bot=TelegramBot(), job_queue=Queue()))

    assert calls == ["commands", "menu"]
    assert bot.scheduler is not None
    assert repeating == ["labs:cleanup"]
    assert caplog.text.count("error_type=TelegramError") == 2
    assert "provider detail" not in caplog.text

    async def fail_access_check(telegram_id: int) -> bool:
        raise RuntimeError("database provider detail")

    monkeypatch.setattr(
        bot.access_service,
        "has_full_access_by_telegram_id",
        fail_access_check,
    )
    with caplog.at_level(logging.ERROR):
        await bot.scheduler._morning(
            SimpleNamespace(
                job=SimpleNamespace(data={"telegram_id": 999, "timezone": "Europe/Moscow"})
            )
        )
    assert sends == []
    assert one_off == ["user:999:morning"]
    assert "error_type=RuntimeError user_id=999" in caplog.text
    assert "database provider detail" not in caplog.text


async def test_startup_schedules_completed_subscribers_and_admins_only(db, fake_ai):
    tiers = [GUEST, SUBSCRIBER, ADMIN, BLOCKED, SUBSCRIBER]
    completed = [True, True, True, True, False]
    async with db.session() as session:
        for offset, (tier, onboarding_completed) in enumerate(
            zip(tiers, completed, strict=True), start=1
        ):
            user = await UserRepository(session).get_or_create(93000 + offset, "Europe/Moscow")
            user.access_tier = tier
            user.onboarding_completed = onboarding_completed

    scheduled: list[dict[str, object]] = []

    class TelegramBot:
        async def set_my_commands(self, commands, **kwargs):
            return None

        async def set_chat_menu_button(self, **kwargs):
            return None

    class Queue:
        def get_jobs_by_name(self, name):
            return []

        def jobs(self):
            return []

        def run_once(self, callback, **kwargs):
            scheduled.append(kwargs)

        def run_daily(self, callback, **kwargs):
            scheduled.append(kwargs)

        def run_repeating(self, callback, **kwargs):
            return None

    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    await bot._post_init(SimpleNamespace(bot=TelegramBot(), job_queue=Queue()))

    user_jobs = [job for job in scheduled if str(job["name"]).startswith("user:")]
    assert {int(str(job["name"]).split(":")[1]) for job in user_jobs} == {93002, 93003}
    assert len(user_jobs) == 6
