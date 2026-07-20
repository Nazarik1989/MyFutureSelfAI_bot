import logging
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from telegram.ext import ConversationHandler

from future_self.bot import (
    HEALTH_ENERGY,
    HEALTH_MOOD,
    HEALTH_PHYSICAL,
    HEALTH_SLEEP,
    HEALTH_STRESS,
    HEALTH_SYMPTOMS,
    FutureSelfBot,
)
from future_self.config import Settings
from future_self.health import (
    HealthService,
    prolonged_weakness_message,
    subjective_score,
    urgent_safety_message,
)
from future_self.models import HealthCheckIn, HealthReminderPreference
from future_self.repositories import UserRepository
from future_self.scheduler import JobQueueScheduler


class NoopTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return ""


class HealthMessage:
    def __init__(self, text: str):
        self.text = text
        self.replies: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self


def health_settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def health_update(text: str, user_id: int = 700, chat_id: int = 1700):
    message = HealthMessage(text)
    return (
        SimpleNamespace(
            effective_message=message,
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=chat_id),
        ),
        message,
    )


def answers(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "energy": 7,
        "sleep": 6,
        "mood": 8,
        "stress": 3,
        "physical_wellbeing": 7,
        "symptoms": None,
    }
    values.update(overrides)
    return values


def test_subjective_score_is_bounded_and_inverts_stress():
    assert subjective_score(10, 10, 10, 0, 10) == 100
    assert subjective_score(0, 0, 0, 10, 0) == 0
    assert subjective_score(7, 6, 8, 3, 7) == 70
    with pytest.raises(ValueError):
        subjective_score(11, 5, 5, 5, 5)


@pytest.mark.parametrize(
    "symptoms",
    [
        "Сильная боль и давление в груди",
        "Сильная одышка, не могу говорить",
        "Была потеря сознания",
        "Перекос лица и неразборчивая речь",
        "Сильное кровотечение",
        "Судороги",
        "Я задыхаюсь, мне не хватает воздуха",
        "Есть мысли причинить себе вред",
    ],
)
def test_red_flags_recommend_emergency_help_without_diagnosis(symptoms):
    message = urgent_safety_message(symptoms)
    assert "экстренную медицинскую службу" in message
    assert "не ставит диагноз" in message.lower()
    assert "лекар" not in message.lower()
    assert "анализ" not in message.lower()


def test_negated_or_mild_symptoms_do_not_trigger_emergency_message():
    assert urgent_safety_message("Нет боли в груди, просто устал") is None
    assert urgent_safety_message("Я не задыхаюсь, дыхание нормальное") is None
    assert urgent_safety_message("Небольшая усталость после работы") is None
    assert urgent_safety_message("Нет боли в груди, но сильная одышка") is not None


def test_prolonged_weakness_recommends_appointment_and_observation_list_only():
    message = prolonged_weakness_message("Слабость уже несколько недель", 1)
    assert "записаться" in message
    assert "наблюдения" in message
    assert "не диагноз" in message
    assert "назначение лечения или анализов" in message
    assert prolonged_weakness_message("Слабость сегодня", 1) is None
    assert prolonged_weakness_message("Слабость сегодня", 3) is not None


async def test_health_history_is_scoped_by_user_and_timezone(db):
    async with db.session() as session:
        first = await UserRepository(session).get_or_create(1, "Europe/Moscow")
        second = await UserRepository(session).get_or_create(2, "Asia/Tbilisi")
        first_id, second_id = first.id, second.id
    service = HealthService(db)
    first_record = await service.save(
        user_id=first_id,
        timezone="Europe/Moscow",
        answers=answers(),
        now=datetime(2026, 7, 17, 21, 30, tzinfo=UTC),
    )
    await service.save(
        user_id=second_id,
        timezone="Asia/Tbilisi",
        answers=answers(energy=2),
        now=datetime(2026, 7, 17, 21, 30, tzinfo=UTC),
    )
    assert first_record.local_date == date(2026, 7, 18)
    assert first_record.timezone == "Europe/Moscow"
    assert [record.id for record in await service.history(first_id)] == [first_record.id]
    assert await service.get_owned(second_id, first_record.id) is None
    assert await service.delete_owned(second_id, first_record.id) is False
    assert (
        await service.save(
            user_id=second_id,
            timezone="Asia/Tbilisi",
            answers=answers(),
            record_id=first_record.id,
        )
        is None
    )
    assert (
        await service.save(
            user_id=first_id,
            timezone="Europe/Moscow",
            answers=answers(),
            record_id=999_999,
        )
        is None
    )
    assert await service.delete_owned(first_id, first_record.id) is True


async def test_same_day_checkin_is_updated_not_duplicated(db):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(3, "UTC")
        user_id = user.id
    service = HealthService(db)
    now = datetime(2026, 7, 17, 10, tzinfo=UTC)
    first = await service.save(user_id=user_id, timezone="UTC", answers=answers(), now=now)
    second = await service.save(
        user_id=user_id,
        timezone="UTC",
        answers=answers(energy=9, symptoms="лучше"),
        now=now,
    )
    assert second.id == first.id
    assert second.energy == 9
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(HealthCheckIn.id))) == 1


async def test_weekly_report_compares_current_and_previous_seven_days(db):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(4, "UTC")
        user_id = user.id
    service = HealthService(db)
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    async with db.session() as session:
        session.add_all(
            [
                HealthCheckIn(
                    user_id=user_id,
                    local_date=date(2026, 7, 17) - timedelta(days=offset),
                    timezone="UTC",
                    **{
                        name: value
                        for name, value in answers(energy=energy).items()
                        if name != "symptoms"
                    },
                    symptoms=None,
                    state_score=subjective_score(
                        energy,
                        6,
                        8,
                        3,
                        7,
                    ),
                )
                for offset, energy, value in ((0, 8, 0), (1, 6, 0), (7, 4, 0), (8, 4, 0))
            ]
        )
    report = await service.weekly_report(user_id, "UTC", now=now)
    assert report.current_count == 2
    assert report.previous_count == 2
    assert report.current["energy"] == 7
    assert report.changes["energy"] == 3


async def test_real_checkin_flow_saves_score_without_llm_or_sensitive_logs(db, fake_ai, caplog):
    bot = FutureSelfBot(health_settings(), db, fake_ai, NoopTranscription())
    context = SimpleNamespace(user_data={}, args=[])
    update, message = health_update("/checkin")
    assert await bot.health_checkin_start(update, context) == HEALTH_ENERGY
    assert "Энергия" in message.replies[-1]["text"]
    route = (
        ("7", bot.health_energy, HEALTH_SLEEP),
        ("6", bot.health_sleep, HEALTH_MOOD),
        ("8", bot.health_mood, HEALTH_STRESS),
        ("3", bot.health_stress, HEALTH_PHYSICAL),
        ("7", bot.health_physical, HEALTH_SYMPTOMS),
    )
    for text, handler, expected_state in route:
        step_update, _ = health_update(text)
        assert await handler(step_update, context) == expected_state
    private_symptoms = "Слабость уже несколько недель, личная деталь"
    final_update, final_message = health_update(private_symptoms)
    with caplog.at_level(logging.INFO):
        assert await bot.health_symptoms(final_update, context) == ConversationHandler.END
    assert fake_ai.route_calls == []
    assert "70/100" in final_message.replies[-1]["text"]
    assert "не медицинский диагноз" in final_message.replies[-1]["text"]
    assert "записаться" in final_message.replies[-1]["text"]
    assert private_symptoms not in caplog.text


async def test_health_command_edit_and_delete_enforce_ownership(db, fake_ai):
    bot = FutureSelfBot(health_settings(), db, fake_ai, NoopTranscription())
    first = await bot._user(10)
    second = await bot._user(20)
    record = await bot.health_service.save(
        user_id=first.id,
        timezone=first.timezone,
        answers=answers(),
    )
    foreign_context = SimpleNamespace(user_data={}, args=[str(record.id)])
    foreign_update, foreign_message = health_update("/health_delete", user_id=20)
    await bot.health_delete_command(foreign_update, foreign_context)
    assert "не найдена" in foreign_message.replies[-1]["text"]
    assert await bot.health_service.get_owned(first.id, record.id) is not None

    own_update, own_message = health_update("/health_delete", user_id=10)
    await bot.health_delete_command(own_update, foreign_context)
    assert "удалена" in own_message.replies[-1]["text"]
    assert await bot.health_service.get_owned(first.id, record.id) is None
    assert second.id != first.id


async def test_health_reminder_opt_in_is_persistent_and_opt_out_removes_job(db, fake_ai):
    scheduled: list[dict[str, object]] = []
    removed: list[int] = []
    bot = FutureSelfBot(health_settings(), db, fake_ai, NoopTranscription())
    bot.scheduler = SimpleNamespace(
        schedule_health_reminder=lambda **kwargs: scheduled.append(kwargs),
        remove_health_reminder=removed.append,
    )
    context = SimpleNamespace(user_data={}, args=["19:30"])
    update, message = health_update("/health_reminder_on", user_id=30, chat_id=300)
    await bot.health_reminder_on(update, context)
    assert "19:30" in message.replies[-1]["text"]
    assert scheduled[0]["local_time"] == time(19, 30)
    async with db.sessions() as session:
        preference = await session.scalar(select(HealthReminderPreference))
    assert preference.enabled is True
    assert preference.chat_id == 30

    off_update, _ = health_update("/health_reminder_off", user_id=30, chat_id=300)
    await bot.health_reminder_off(off_update, SimpleNamespace(user_data={}, args=[]))
    assert removed == [preference.user_id]
    async with db.sessions() as session:
        preference = await session.get(HealthReminderPreference, preference.id)
    assert preference.enabled is False


async def test_health_reminder_rejects_non_hhmm_time(db, fake_ai):
    bot = FutureSelfBot(health_settings(), db, fake_ai, NoopTranscription())
    update, message = health_update("/health_reminder_on", user_id=31, chat_id=301)
    await bot.health_reminder_on(
        update,
        SimpleNamespace(user_data={}, args=["20:00:30"]),
    )
    assert "формате HH:MM" in message.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(HealthReminderPreference.id))) == 0


def test_scheduler_health_reminder_uses_recurring_timezone_job_and_deduplicated_name():
    jobs: list[dict[str, object]] = []
    removals: list[str] = []

    class Job:
        def schedule_removal(self):
            removals.append("removed")

    class Queue:
        def get_jobs_by_name(self, name):
            assert name == "health:12:daily"
            return [Job()]

        def run_daily(self, callback, **kwargs):
            jobs.append({"callback": callback, **kwargs})

    async def send(chat_id: int, text: str):
        return 1

    scheduler = JobQueueScheduler(Queue(), send, 8, 21, 6)
    scheduler.schedule_health_reminder(
        user_id=12,
        chat_id=99,
        timezone="Europe/Moscow",
        local_time=time(20),
    )
    assert removals == ["removed"]
    assert jobs[0]["name"] == "health:12:daily"
    assert jobs[0]["data"]["chat_id"] == 99
    assert jobs[0]["time"].hour == 20
    assert jobs[0]["time"].tzinfo.key == "Europe/Moscow"


async def test_health_reminder_delivery_is_generic_and_recurring_job_survives_error():
    jobs: list[dict[str, object]] = []
    sent: list[tuple[int, str]] = []

    class Queue:
        def get_jobs_by_name(self, name):
            return []

        def run_daily(self, callback, **kwargs):
            jobs.append({"callback": callback, **kwargs})

    async def failing_send(chat_id: int, text: str):
        sent.append((chat_id, text))
        raise RuntimeError("synthetic delivery failure")

    scheduler = JobQueueScheduler(Queue(), failing_send, 8, 21, 6)
    scheduler.schedule_health_reminder(
        user_id=12,
        chat_id=99,
        timezone="Europe/Moscow",
        local_time=time(20),
    )
    context = SimpleNamespace(job=SimpleNamespace(data=jobs[0]["data"]))

    with pytest.raises(RuntimeError, match="synthetic delivery failure"):
        await jobs[0]["callback"](context)

    assert len(jobs) == 1
    assert sent[0][0] == 99
    assert "/checkin" in sent[0][1]
    assert "диагноз" in sent[0][1]
    assert "симптом" not in sent[0][1].lower()
    assert "100" not in sent[0][1]
