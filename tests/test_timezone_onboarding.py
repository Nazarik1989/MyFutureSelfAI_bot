from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from future_self.bot import ONBOARDING_INPUT, FutureSelfBot
from future_self.config import Settings
from future_self.domain import ONBOARDING_QUESTIONS, canonical_timezone
from future_self.models import User
from future_self.repositories import OnboardingRepository
from future_self.schemas import TimezoneResolution
from future_self.timezones import (
    TimezoneCandidate,
    TimezoneResolver,
    resolve_timezone_locally,
    timezone_candidate_text,
)
from tests.autotester.fakes import FakeCallbackQuery, FakeMessage


class NoopTranscription:
    enabled = True


def make_bot(db, fake_ai) -> FutureSelfBot:
    return FutureSelfBot(
        Settings(
            _env_file=None,
            telegram_bot_token="123456:TEST",
            ai_api_key="test-key",
            ai_model="test-model",
        ),
        db,
        fake_ai,
        NoopTranscription(),
    )


def message_update(message: FakeMessage, telegram_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=telegram_id),
        effective_chat=SimpleNamespace(id=telegram_id),
    )


def callback_update(query: FakeCallbackQuery, telegram_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        callback_query=query,
        effective_message=query.message,
        effective_user=SimpleNamespace(id=telegram_id),
        effective_chat=SimpleNamespace(id=telegram_id),
    )


def callback_data(message: FakeMessage, prefix: str) -> str:
    keyboard = message.replies[-1]["reply_markup"].inline_keyboard
    return next(
        button.callback_data
        for row in keyboard
        for button in row
        if button.callback_data.startswith(prefix)
    )


async def seed_timezone_step(bot: FutureSelfBot, db, telegram_id: int) -> int:
    user = await bot._user(telegram_id)
    async with db.session() as session:
        state = await OnboardingRepository(session).get_or_create(user.id)
        state.current_step = 1
        state.answers = {"display_name": "Варвара"}
    return user.id


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Moscow", "Europe/Moscow"),
        (" Москва ", "Europe/Moscow"),
        ("МСК", "Europe/Moscow"),
        ("GMT+3", "Europe/Moscow"),
        ("Саратов", "Europe/Saratov"),
        ("GMT+4", "Europe/Saratov"),
        ("UTC +04:00", "Europe/Saratov"),
        ("Asia/Tbilisi", "Asia/Tbilisi"),
    ],
)
def test_canonical_timezone_accepts_human_friendly_values(value, expected):
    assert canonical_timezone(value) == expected


def test_canonical_timezone_explains_supported_values():
    with pytest.raises(ValueError, match="Moscow.*GMT\\+4.*Europe/Moscow"):
        canonical_timezone("Марс")


def test_common_city_is_resolved_locally_inside_a_phrase():
    candidate = resolve_timezone_locally("я сейчас живу в Казани")

    assert candidate == TimezoneCandidate("Europe/Moscow", "Казань", "local")


async def test_unknown_city_uses_model_and_validates_iana_timezone(fake_ai):
    fake_ai.timezone_results["живу в Лиссабоне"] = TimezoneResolution(
        timezone="Europe/Lisbon",
        city="Лиссабон",
        country="Португалия",
        ambiguous=False,
    )

    candidate = await TimezoneResolver(fake_ai).resolve("живу в Лиссабоне")

    assert candidate == TimezoneCandidate("Europe/Lisbon", "Лиссабон", "model")
    assert fake_ai.timezone_calls == ["живу в Лиссабоне"]


async def test_ambiguous_model_result_asks_for_country(fake_ai):
    with pytest.raises(ValueError, match="страной или регионом"):
        await TimezoneResolver(fake_ai).resolve("Сан-Хосе")


async def test_model_cannot_save_unknown_iana_timezone(fake_ai):
    fake_ai.timezone_results["Атлантида"] = TimezoneResolution(
        timezone="Ocean/Atlantis",
        city="Атлантида",
        country=None,
        ambiguous=False,
    )

    with pytest.raises(ValueError, match="не смогла подтвердить"):
        await TimezoneResolver(fake_ai).resolve("Атлантида")


def test_candidate_text_uses_dst_offset_for_selected_date():
    candidate = TimezoneCandidate("Europe/Berlin", "Берлин", "local")

    winter = timezone_candidate_text(candidate, now=datetime(2026, 1, 15, tzinfo=UTC))
    summer = timezone_candidate_text(candidate, now=datetime(2026, 7, 15, tzinfo=UTC))

    assert "UTC+1" in winter
    assert "UTC+2" in summer


@pytest.mark.parametrize(
    ("telegram_id", "answer", "expected", "expected_location"),
    [
        (5001, "Moscow", "Europe/Moscow", "Moscow"),
        (5002, "GMT+4", "Europe/Saratov", None),
        (5004, "я сейчас живу в Казани", "Europe/Moscow", "Казань"),
    ],
)
async def test_onboarding_waits_for_confirmation_then_saves_timezone(
    db, fake_ai, telegram_id, answer, expected, expected_location
):
    bot = make_bot(db, fake_ai)
    user_id = await seed_timezone_step(bot, db, telegram_id)
    message = FakeMessage(answer)
    context = SimpleNamespace(user_data={"onboarding_user_id": user_id})

    result = await bot.onboarding_answer(message_update(message, telegram_id), context)

    assert result == ONBOARDING_INPUT
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user_id)
        assert state.current_step == 1
        assert "timezone" not in state.answers
    confirm = callback_data(message, "onboarding:timezone:confirm:")
    query = FakeCallbackQuery(confirm, message)

    result = await bot.onboarding_timezone_action(callback_update(query, telegram_id), context)

    assert result == ONBOARDING_INPUT
    assert query.markup_removed == 1
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user_id)
        assert state.current_step == 2
        assert state.answers["timezone"] == expected
        assert state.answers.get("location") == expected_location
    assert any("жизнь через три года" in reply["text"] for reply in message.replies)


async def test_retry_keeps_timezone_step_and_removes_pending_candidate(db, fake_ai):
    telegram_id = 5005
    bot = make_bot(db, fake_ai)
    user_id = await seed_timezone_step(bot, db, telegram_id)
    message = FakeMessage("Казань")
    context = SimpleNamespace(user_data={"onboarding_user_id": user_id})
    await bot.onboarding_answer(message_update(message, telegram_id), context)
    retry = callback_data(message, "onboarding:timezone:retry:")
    query = FakeCallbackQuery(retry, message)

    result = await bot.onboarding_timezone_action(callback_update(query, telegram_id), context)

    assert result == ONBOARDING_INPUT
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user_id)
        assert state.current_step == 1
        assert "timezone" not in state.answers
        assert "pending_timezone" not in state.answers.get("__onboarding_flow__", {})
    assert "В каком городе" in message.replies[-1]["text"]


async def test_confirmed_city_skips_duplicate_final_location_question(db, fake_ai):
    telegram_id = 5006
    bot = make_bot(db, fake_ai)
    user_id = await seed_timezone_step(bot, db, telegram_id)
    message = FakeMessage("Казань")
    context = SimpleNamespace(user_data={"onboarding_user_id": user_id})
    await bot.onboarding_answer(message_update(message, telegram_id), context)
    query = FakeCallbackQuery(callback_data(message, "onboarding:timezone:confirm:"), message)
    await bot.onboarding_timezone_action(callback_update(query, telegram_id), context)

    support_step = next(
        index
        for index, question in enumerate(ONBOARDING_QUESTIONS)
        if question[0] == "support_style"
    )
    async with db.session() as session:
        state = await OnboardingRepository(session).get_or_create(user_id)
        answers = dict(state.answers)
        answers.update(
            {
                "future_life": "Спокойная и наполненная жизнь",
                "ideal_day": "Работа, спорт и близкие",
                "values": "Здоровье и свобода",
            }
        )
        state.current_step = support_step
        state.answers = answers

    support = FakeMessage("Мягко, но по делу")
    result = await bot.onboarding_answer(message_update(support, telegram_id), context)

    assert result != ONBOARDING_INPUT
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user_id)
        assert state.current_step == len(ONBOARDING_QUESTIONS)
        assert state.answers["location"] == "Казань"
        assert state.answers["__onboarding_flow__"]["location_autofilled"] is True
    assert not any("искать врачей" in reply["text"] for reply in support.replies)


async def test_stale_confirmation_cannot_advance_twice(db, fake_ai):
    telegram_id = 5007
    bot = make_bot(db, fake_ai)
    user_id = await seed_timezone_step(bot, db, telegram_id)
    message = FakeMessage("Казань")
    context = SimpleNamespace(user_data={"onboarding_user_id": user_id})
    await bot.onboarding_answer(message_update(message, telegram_id), context)
    confirm = callback_data(message, "onboarding:timezone:confirm:")

    first = FakeCallbackQuery(confirm, message)
    await bot.onboarding_timezone_action(callback_update(first, telegram_id), context)
    stale = FakeCallbackQuery(confirm, message)
    result = await bot.onboarding_timezone_action(callback_update(stale, telegram_id), context)

    assert result == ONBOARDING_INPUT
    assert stale.answers[-1][1] is True
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user_id)
        assert state.current_step == 2


async def test_invalid_timezone_does_not_advance_or_call_model(db, fake_ai):
    telegram_id = 5003
    bot = make_bot(db, fake_ai)
    user_id = await seed_timezone_step(bot, db, telegram_id)
    message = FakeMessage("GMT+99")
    context = SimpleNamespace(user_data={"onboarding_user_id": user_id})

    result = await bot.onboarding_answer(message_update(message, telegram_id), context)

    assert result == ONBOARDING_INPUT
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user_id)
        assert state.current_step == 1
        assert "timezone" not in state.answers
    assert fake_ai.timezone_calls == []
    assert "GMT+4" in message.replies[-1]["text"]


async def test_timezone_command_updates_completed_user_after_confirmation(db, fake_ai):
    telegram_id = 5008
    bot = make_bot(db, fake_ai)
    user = await bot._user(telegram_id)
    async with db.session() as session:
        stored = await session.get(User, user.id)
        stored.onboarding_completed = True
        stored.timezone = "Europe/Moscow"
        state = await OnboardingRepository(session).get_or_create(user.id)
        state.status = "completed"
        state.answers = {"timezone": "Europe/Moscow", "location": "Москва"}

    message = FakeMessage("/timezone Берлин")
    context = SimpleNamespace(args=["Берлин"], user_data={})
    await bot.timezone_command(message_update(message, telegram_id), context)
    confirm = callback_data(message, "timezone:update:confirm:")
    query = FakeCallbackQuery(confirm, message)

    await bot.timezone_action(callback_update(query, telegram_id), context)

    assert any("Europe/Berlin" in edit for edit in query.edits)
    async with db.sessions() as session:
        stored = await session.get(User, user.id)
        state = await OnboardingRepository(session).get_or_create(user.id)
        assert stored.timezone == "Europe/Berlin"
        assert stored.location_city == "Берлин"
        assert state.answers["timezone"] == "Europe/Berlin"
        assert state.answers["location"] == "Берлин"

    stale = FakeCallbackQuery(confirm, message)
    await bot.timezone_action(callback_update(stale, telegram_id), context)
    assert stale.answers[-1][1] is True
