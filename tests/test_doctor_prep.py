import logging
from asyncio import gather
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import func, select
from telegram.ext import ConversationHandler

from future_self.bot import (
    DOCTOR_DURATION,
    DOCTOR_MEDICATIONS,
    DOCTOR_QUESTIONS,
    DOCTOR_REASON,
    DOCTOR_SYMPTOMS,
    FutureSelfBot,
)
from future_self.config import Settings
from future_self.doctor_prep import DoctorVisitPrepService
from future_self.health import subjective_score
from future_self.models import DoctorVisitPrep, HealthCheckIn, InboxItem, TaskReminder
from future_self.repositories import UserRepository


class NoopTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return ""


class PrepMessage:
    def __init__(self, text: str):
        self.text = text
        self.replies: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self


def prep_settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def prep_update(text: str, *, user_id: int = 700, chat_id: int = 1700):
    message = PrepMessage(text)
    return (
        SimpleNamespace(
            effective_message=message,
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=chat_id),
        ),
        message,
    )


def prep_answers(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "reason": "Длительная слабость",
        "duration": "Около трёх недель",
        "symptoms": "Слабость усиливается к вечеру",
        "medications": "нет",
        "questions": "Что важно отслеживать дальше?",
    }
    values.update(overrides)
    return values


async def test_service_summary_uses_health_dynamics_and_owner_isolation(db):
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(1, "Europe/Moscow")
        stranger = await UserRepository(session).get_or_create(2, "Europe/Moscow")
        owner_id, stranger_id = owner.id, stranger.id
        session.add_all(
            [
                HealthCheckIn(
                    user_id=owner_id,
                    local_date=date.today(),
                    timezone="Europe/Moscow",
                    energy=4,
                    sleep=5,
                    mood=6,
                    stress=7,
                    physical_wellbeing=4,
                    symptoms="private historical symptom",
                    state_score=subjective_score(4, 5, 6, 7, 4),
                ),
                HealthCheckIn(
                    user_id=owner_id,
                    local_date=date.today() - timedelta(days=1),
                    timezone="Europe/Moscow",
                    energy=6,
                    sleep=6,
                    mood=6,
                    stress=5,
                    physical_wellbeing=6,
                    symptoms=None,
                    state_score=subjective_score(6, 6, 6, 5, 6),
                ),
            ]
        )
    service = DoctorVisitPrepService(db)
    record = await service.save(
        user_id=owner_id,
        timezone="Europe/Moscow",
        answers=prep_answers(),
    )
    assert "Health Track:" in record.summary
    assert "линейка" in record.summary
    assert "private historical symptom" not in record.summary
    assert "не медицинский диагноз" in record.summary
    assert await service.get_owned(stranger_id, record.id) is None
    assert await service.delete_owned(stranger_id, record.id) is False
    assert (
        await service.save(
            user_id=stranger_id,
            timezone="Europe/Moscow",
            answers=prep_answers(),
            record_id=record.id,
        )
        is None
    )
    assert await service.delete_owned(owner_id, record.id) is True


async def test_real_doctor_prepare_flow_is_deterministic_private_and_handles_red_flags(
    db, fake_ai, caplog
):
    bot = FutureSelfBot(prep_settings(), db, fake_ai, NoopTranscription())
    context = SimpleNamespace(user_data={}, args=[])
    update, message = prep_update("/doctor_prepare")
    assert await bot.doctor_prepare_start(update, context) == DOCTOR_REASON
    assert "причина обращения" in message.replies[-1]["text"]
    route = (
        ("Слабость и боль в груди", bot.doctor_prepare_reason, DOCTOR_DURATION),
        ("Три недели", bot.doctor_prepare_duration, DOCTOR_SYMPTOMS),
        ("Сильная боль в груди", bot.doctor_prepare_symptoms, DOCTOR_MEDICATIONS),
        ("нет", bot.doctor_prepare_medications, DOCTOR_QUESTIONS),
    )
    for text, handler, expected_state in route:
        step_update, step_message = prep_update(text)
        assert await handler(step_update, context) == expected_state
        if handler in {bot.doctor_prepare_reason, bot.doctor_prepare_symptoms}:
            assert "экстренную медицинскую службу" in step_message.replies[-1]["text"]
            assert "Не жди завершения опроса" in step_message.replies[-1]["text"]
    private_question = "Личный вопрос врачу 9f0c-private"
    final_update, final_message = prep_update(private_question)
    with caplog.at_level(logging.INFO):
        assert await bot.doctor_prepare_questions(final_update, context) == ConversationHandler.END
    output = final_message.replies[-1]["text"]
    assert "Краткое фактическое резюме" in output
    assert "экстренную медицинскую службу" in output
    assert "не заменяют срочную помощь" in output
    assert fake_ai.route_calls == []
    assert private_question not in caplog.text
    async with db.sessions() as session:
        record = await session.scalar(select(DoctorVisitPrep))
    assert record.medications is None


async def test_required_doctor_answers_do_not_advance_when_blank(db, fake_ai):
    bot = FutureSelfBot(prep_settings(), db, fake_ai, NoopTranscription())
    context = SimpleNamespace(user_data={}, args=[])
    start, _ = prep_update("/doctor_prepare")
    assert await bot.doctor_prepare_start(start, context) == DOCTOR_REASON

    blank_reason, reason_message = prep_update("   ")
    assert await bot.doctor_prepare_reason(blank_reason, context) == DOCTOR_REASON
    assert "не должна быть пустой" in reason_message.replies[-1]["text"]

    valid_reason, _ = prep_update("Слабость")
    assert await bot.doctor_prepare_reason(valid_reason, context) == DOCTOR_DURATION
    blank_duration, duration_message = prep_update(" ")
    assert await bot.doctor_prepare_duration(blank_duration, context) == DOCTOR_DURATION
    assert "не должна быть пустой" in duration_message.replies[-1]["text"]

    valid_duration, _ = prep_update("Две недели")
    assert await bot.doctor_prepare_duration(valid_duration, context) == DOCTOR_SYMPTOMS
    blank_symptoms, symptom_message = prep_update("  ")
    assert await bot.doctor_prepare_symptoms(blank_symptoms, context) == DOCTOR_SYMPTOMS
    assert "не должны быть пустыми" in symptom_message.replies[-1]["text"]


async def test_doctor_summary_stays_within_safe_telegram_size(db):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(3, "Europe/Moscow")
        user_id = user.id
    record = await DoctorVisitPrepService(db).save(
        user_id=user_id,
        timezone="Europe/Moscow",
        answers=prep_answers(
            reason="р" * 1000,
            duration="д" * 500,
            symptoms="с" * 2000,
            medications="л" * 2000,
            questions="в" * 2000,
        ),
    )
    assert len(record.summary) < 3000


async def test_edit_delete_and_show_enforce_owner(db, fake_ai):
    bot = FutureSelfBot(prep_settings(), db, fake_ai, NoopTranscription())
    owner = await bot._user(10)
    record = await bot.doctor_prep_service.save(
        user_id=owner.id,
        timezone=owner.timezone,
        answers=prep_answers(),
    )
    foreign_context = SimpleNamespace(user_data={}, args=[str(record.id)])
    foreign_show, foreign_message = prep_update(
        f"/doctor_prepare_show {record.id}",
        user_id=20,
    )
    await bot.doctor_prepare_show(foreign_show, foreign_context)
    assert "не найдена" in foreign_message.replies[-1]["text"]
    foreign_delete, foreign_delete_message = prep_update(
        f"/doctor_prepare_delete {record.id}",
        user_id=20,
    )
    await bot.doctor_prepare_delete(foreign_delete, foreign_context)
    assert "не найдена" in foreign_delete_message.replies[-1]["text"]

    edit_update, edit_message = prep_update(f"/doctor_prepare_edit {record.id}", user_id=10)
    assert await bot.doctor_prepare_start(edit_update, foreign_context) == DOCTOR_REASON
    assert "Исправляем" in edit_message.replies[-1]["text"]
    own_delete, own_message = prep_update(f"/doctor_prepare_delete {record.id}", user_id=10)
    await bot.doctor_prepare_delete(own_delete, foreign_context)
    assert "удалена" in own_message.replies[-1]["text"]


async def test_doctor_task_uses_existing_reminder_engine_and_is_idempotent(db, fake_ai):
    bot = FutureSelfBot(prep_settings(), db, fake_ai, NoopTranscription())
    user = await bot._user(30)
    record = await bot.doctor_prep_service.save(
        user_id=user.id,
        timezone=user.timezone,
        answers=prep_answers(reason="private reason must not enter reminder"),
    )
    context = SimpleNamespace(
        user_data={},
        args=[str(record.id), "через", "2", "часа"],
    )
    update, message = prep_update(
        f"/doctor_prepare_task {record.id} через 2 часа",
        user_id=30,
        chat_id=300,
    )
    before = datetime.now(UTC)
    await bot.doctor_prepare_task(update, context)
    assert "Задача «Записаться к врачу» создана" in message.replies[-1]["text"]
    async with db.sessions() as session:
        item = await session.scalar(select(InboxItem))
        reminder = await session.scalar(select(TaskReminder))
    assert item.title == "Записаться к врачу"
    assert item.source == "doctor_prepare"
    assert "private reason" not in (item.description or "")
    assert reminder.remind_at.replace(tzinfo=UTC) > before

    repeat_update, repeat_message = prep_update(
        f"/doctor_prepare_task {record.id} через 2 часа",
        user_id=30,
        chat_id=300,
    )
    await bot.doctor_prepare_task(repeat_update, context)
    assert "дубликат не добавлен" in repeat_message.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1

    foreign_update, foreign_message = prep_update(
        f"/doctor_prepare_task {record.id} через 2 часа",
        user_id=31,
        chat_id=301,
    )
    await bot.doctor_prepare_task(foreign_update, context)
    assert "не найдена" in foreign_message.replies[-1]["text"]


async def test_concurrent_doctor_task_creation_is_atomic(db, fake_ai):
    bot = FutureSelfBot(prep_settings(), db, fake_ai, NoopTranscription())
    user = await bot._user(40)
    record = await bot.doctor_prep_service.save(
        user_id=user.id,
        timezone=user.timezone,
        answers=prep_answers(),
    )
    temporal = bot._doctor_task_temporal("через 2 часа", user.timezone)
    results = await gather(
        bot.doctor_prep_service.create_appointment_task(
            user_id=user.id,
            record_id=record.id,
            telegram_user_id=40,
            chat_id=400,
            temporal=temporal,
        ),
        bot.doctor_prep_service.create_appointment_task(
            user_id=user.id,
            record_id=record.id,
            telegram_user_id=40,
            chat_id=400,
            temporal=temporal,
        ),
    )
    assert {result.status for result in results} <= {"created", "existing"}
    assert any(result.status == "created" for result in results)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
