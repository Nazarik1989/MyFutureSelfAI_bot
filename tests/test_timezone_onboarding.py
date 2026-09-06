from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from future_self.bot import ONBOARDING_INPUT, FutureSelfBot
from future_self.config import Settings
from future_self.domain import ONBOARDING_QUESTIONS, canonical_timezone
from future_self.models import User
from future_self.repositories import OnboardingRepository
from future_self.schemas import ReminderTimezoneResolution, TimezoneResolution
from future_self.timezones import (
    MAX_REMINDER_TIMEZONE_CANDIDATE_WORDS,
    MAX_REMINDER_TIMEZONE_FRAGMENT_CHARS,
    ReminderTimezoneFragment,
    ReminderTimezoneStatus,
    TimezoneCandidate,
    TimezoneResolver,
    extract_reminder_timezone_fragment,
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


@pytest.mark.parametrize(
    ("phrase", "window_prefix"),
    [
        (
            "Напомни завтра в 10:00 по светогорску созвониться с клиентом",
            "по светогорску",
        ),
        (
            "Каждый день в 20:30 по времени Нижнего Новгорода заполнить дневник",
            "по времени Нижнего Новгорода",
        ),
        (
            "Напомни в 18:00 в часовом поясе Сан-Хосе, Коста-Рика проверить почту",
            "в часовом поясе Сан-Хосе, Коста-Рика",
        ),
    ],
)
def test_reminder_timezone_marker_extraction_produces_a_bounded_candidate_window(
    phrase,
    window_prefix,
):
    fragment = extract_reminder_timezone_fragment(phrase)

    assert fragment is not None
    assert fragment.text.startswith(window_prefix)
    assert fragment.location_text
    assert len(fragment.text) <= MAX_REMINDER_TIMEZONE_FRAGMENT_CHARS
    assert len(fragment.text.split()) <= MAX_REMINDER_TIMEZONE_CANDIDATE_WORDS
    assert phrase[fragment.span[0] : fragment.span[1]] == fragment.text


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни в 10:00 по Нижнему Новгороду позвонить",
        "Напомни в 10:00 по Светогорску, Ленинградская область позвонить",
        "Напомни в 10:00 по Сан-Хосе, США позвонить",
        "Напомни в 10:00 по Сан-Хосе, Коста-Рика позвонить",
        "Напомни в 10:00 по Лос-Анджелесу позвонить",
        "Напомни в 10:00 по Los Angeles позвонить",
        "Напомни в 10:00 по Нью-Йорку позвонить",
        "Напомни в 10:00 по New York позвонить",
        "Напомни в 10:00 по Санкт-Петербургу позвонить",
        "Напомни в 10:00 по Санкт Петербургу позвонить",
    ],
)
def test_multiword_hyphenated_and_qualified_places_remain_in_candidate_window(phrase):
    fragment = extract_reminder_timezone_fragment(phrase)

    assert fragment is not None
    assert fragment.text.startswith(("по ", "в часовом поясе "))


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни завтра в 19:30 по работе позвонить",
        "Напомни завтра в 19:30 по проекту отправить отчёт",
        "Напомни завтра в 19:30 по плану провести встречу",
        "Эта история напомнила мне о работе",
    ],
)
def test_semantic_po_phrases_are_not_timezone_markers(phrase):
    assert extract_reminder_timezone_fragment(phrase) is None


@pytest.mark.parametrize(
    ("phrase", "timezone"),
    [
        ("Напомни завтра в 10:00 по МСК позвонить", "Europe/Moscow"),
        ("Напомни завтра в 10:00 по Лондону позвонить", "Europe/London"),
        ("Каждый день в 20:30 по времени Тбилиси писать", "Asia/Tbilisi"),
        ("Напомни в 10:00 по Санкт-Петербургу позвонить", "Europe/Moscow"),
        ("Напомни в 10:00 по Санкт Петербургу позвонить", "Europe/Moscow"),
        ("Напомни в 10:00 по Лос-Анджелесу позвонить", "America/Los_Angeles"),
        ("Напомни в 10:00 по Los Angeles позвонить", "America/Los_Angeles"),
        ("Напомни в 10:00 по Нью-Йорку позвонить", "America/New_York"),
        ("Напомни в 10:00 по New York позвонить", "America/New_York"),
    ],
)
async def test_known_reminder_timezones_are_local_and_never_call_model(fake_ai, phrase, timezone):
    fragment = extract_reminder_timezone_fragment(phrase)

    assert fragment is not None
    resolver = TimezoneResolver(fake_ai)
    local = resolver.resolve_reminder_locally(fragment)
    outcome = await resolver.resolve_reminder(fragment)

    assert local == outcome
    assert outcome.status is ReminderTimezoneStatus.RESOLVED
    assert outcome.candidate is not None
    assert outcome.candidate.timezone == timezone
    assert outcome.evidence_text is not None
    assert outcome.evidence_span is not None
    assert phrase[outcome.evidence_span[0] : outcome.evidence_span[1]] == outcome.evidence_text
    assert fake_ai.reminder_timezone_calls == []
    assert fake_ai.timezone_calls == []


async def test_exact_iana_reminder_timezone_is_local_and_never_calls_model(fake_ai):
    fragment = ReminderTimezoneFragment(
        "по Europe/London",
        "Europe/London",
        (0, len("по Europe/London")),
    )

    outcome = await TimezoneResolver(fake_ai).resolve_reminder(fragment)

    assert outcome.status is ReminderTimezoneStatus.RESOLVED
    assert outcome.candidate is not None
    assert outcome.candidate.timezone == "Europe/London"
    assert outcome.evidence_text == "по Europe/London"
    assert outcome.evidence_span == (0, len("по Europe/London"))
    assert fake_ai.reminder_timezone_calls == []
    assert fake_ai.timezone_calls == []


async def test_unknown_explicit_city_uses_only_timezone_fragment_and_validates_evidence(fake_ai):
    phrase = "Напомни в 9:00 по Светогорску позвонить врачу"
    fragment = extract_reminder_timezone_fragment(phrase)
    assert fragment is not None
    fake_ai.reminder_timezone_results[fragment.text] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Moscow",
        matched_text="по светогорску",
        city="Светогорск",
        country="Россия",
    )

    outcome = await TimezoneResolver(fake_ai).resolve_reminder(fragment)

    assert outcome.status is ReminderTimezoneStatus.RESOLVED
    assert outcome.candidate == TimezoneCandidate("Europe/Moscow", "Светогорск", "model")
    assert outcome.evidence_text == "по Светогорску"
    assert outcome.evidence_span is not None
    assert phrase[outcome.evidence_span[0] : outcome.evidence_span[1]] == outcome.evidence_text
    assert fake_ai.reminder_timezone_calls == [fragment.text]
    assert phrase not in fake_ai.reminder_timezone_calls
    assert fake_ai.timezone_calls == []


@pytest.mark.parametrize("status", ["not_mentioned", "insufficient"])
async def test_model_reminder_timezone_status_is_preserved_without_candidate(fake_ai, status):
    fragment = ReminderTimezoneFragment("по Сан-Хосе", "Сан-Хосе", (0, 11))
    fake_ai.reminder_timezone_results[fragment.text] = ReminderTimezoneResolution(status=status)

    outcome = await TimezoneResolver(fake_ai).resolve_reminder(fragment)

    assert outcome.status == status
    assert outcome.candidate is None
    assert fake_ai.reminder_timezone_calls == [fragment.text]


async def test_ambiguous_model_result_returns_exact_timezone_evidence(fake_ai):
    text = "по Сан-Хосе созвониться"
    fragment = ReminderTimezoneFragment(
        text,
        "Сан-Хосе созвониться",
        (25, 25 + len(text)),
    )
    fake_ai.reminder_timezone_results[fragment.text] = ReminderTimezoneResolution(
        status="ambiguous",
        matched_text="по Сан-Хосе",
    )

    outcome = await TimezoneResolver(fake_ai).resolve_reminder(fragment)

    assert outcome.status is ReminderTimezoneStatus.AMBIGUOUS
    assert outcome.candidate is None
    assert outcome.evidence_text == "по Сан-Хосе"
    assert outcome.evidence_span == (25, 36)


async def test_qualified_known_alias_is_never_resolved_by_local_substring_match(fake_ai):
    phrase = "Напомни в 9:00 по Лондону, Канада позвонить"
    fragment = extract_reminder_timezone_fragment(phrase)
    assert fragment is not None
    resolver = TimezoneResolver(fake_ai)
    assert resolver.resolve_reminder_locally(fragment) is None
    fake_ai.reminder_timezone_results[fragment.text] = ReminderTimezoneResolution(
        status="resolved",
        timezone="America/Toronto",
        matched_text="по Лондону, Канада",
        city="London",
        country="Canada",
    )

    outcome = await resolver.resolve_reminder(fragment)

    assert outcome.candidate is not None
    assert outcome.candidate.timezone == "America/Toronto"
    assert outcome.candidate.source == "model"
    assert outcome.evidence_text == "по Лондону, Канада"
    assert fake_ai.reminder_timezone_calls == [fragment.text]


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни в 9:00 по Лондону Канада позвонить",
        "напомни в 9:00 по лондону канада позвонить",
        "Напомни в 9:00 по London Ontario позвонить",
        "Напомни в 9:00 по Москве Московская область позвонить",
        "Напомни в 9:00 по лондону япония позвонить",
        "Напомни в 9:00 по лондону гаити позвонить",
        "Напомни в 9:00 по лондону тольятти позвонить",
    ],
)
def test_unpunctuated_country_or_region_qualifier_also_forces_model(fake_ai, phrase):
    fragment = extract_reminder_timezone_fragment(phrase)

    assert fragment is not None
    assert TimezoneResolver(fake_ai).resolve_reminder_locally(fragment) is None


async def test_reminder_model_cannot_return_invalid_iana_timezone(fake_ai):
    fragment = ReminderTimezoneFragment("по Светогорску", "Светогорску", (0, 15))
    fake_ai.reminder_timezone_results[fragment.text] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Ocean/Atlantis",
        matched_text="по Светогорску",
    )

    with pytest.raises(ValueError, match="invalid IANA"):
        await TimezoneResolver(fake_ai).resolve_reminder(fragment)


@pytest.mark.parametrize(
    ("window", "matched", "error"),
    [
        ("по Светогорску позвонить", "по Берлину", "start at"),
        ("по Светогорску позвонить", "Светогорску", "start at"),
        ("по Светогорску и по Светогорску", "по Светогорску", "unique"),
        ("по Светогорску позвонить", "по", "contain a place"),
    ],
)
async def test_reminder_model_evidence_is_exact_prefix_unique_and_contains_place(
    fake_ai,
    window,
    matched,
    error,
):
    fragment = ReminderTimezoneFragment(window, window, (0, len(window)))
    fake_ai.reminder_timezone_results[fragment.text] = ReminderTimezoneResolution(
        status="resolved",
        timezone="Europe/Berlin",
        matched_text=matched,
    )

    with pytest.raises(ValueError, match=error):
        await TimezoneResolver(fake_ai).resolve_reminder(fragment)


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
