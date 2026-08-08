from types import SimpleNamespace

import pytest
from autotester.fakes import (
    FakeCallbackQuery,
    FakeMessage,
    FakeVoice,
    ScriptedTranscription,
)
from sqlalchemy import func, select
from telegram import ReplyKeyboardRemove
from telegram.ext import ApplicationHandlerStop, ConversationHandler

from future_self.access import AccessService
from future_self.bot import ONBOARDING_INPUT, PROFILE_CONFIRM, FutureSelfBot
from future_self.config import Settings
from future_self.domain import ONBOARDING_QUESTIONS
from future_self.models import DraftInboxItem, OnboardingState, User, VisionProfile
from future_self.repositories import OnboardingRepository
from future_self.schemas import VisionSummary


def make_bot(db, ai) -> FutureSelfBot:
    return FutureSelfBot(
        Settings(
            _env_file=None,
            telegram_bot_token="123456:test-token",
            ai_api_key="test-key",
            database_url="sqlite+aiosqlite:///:memory:",
        ),
        db,
        ai,
        ScriptedTranscription(),
    )


def update_for(
    message: FakeMessage,
    *,
    user_id: int = 8801,
    chat_id: int | None = None,
    query: FakeCallbackQuery | None = None,
    update_id: int | None = None,
):
    return SimpleNamespace(
        update_id=update_id,
        effective_message=message,
        message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id or user_id, type="private"),
    )


def context() -> SimpleNamespace:
    return SimpleNamespace(user_data={}, args=[])


def callback_values(message: FakeMessage) -> list[str]:
    return [
        button.callback_data
        for reply in message.replies
        if (markup := reply.get("reply_markup")) is not None
        for row in getattr(markup, "inline_keyboard", ())
        for button in row
        if button.callback_data is not None
    ]


def complete_answers() -> dict[str, str]:
    return {
        "display_name": "Назар",
        "timezone": "Europe/Moscow",
        "future_life": "Спокойная жизнь, своё дело и время для семьи.",
        "residence": "В своём доме рядом с городом.",
        "work_income": "Развиваю полезный продукт.",
        "health_body": "Сильное и здоровое тело.",
        "relationships": "Тёплые отношения с близкими.",
        "ideal_day": "Семья, важная работа, движение и отдых.",
        "values": "Семья, свобода и польза.",
        "obstacles": "Распыление внимания.",
        "support_style": "Спокойно, прямо и по шагам.",
        "location": "Москва",
    }


async def seed_onboarding(
    bot: FutureSelfBot,
    *,
    telegram_id: int,
    step: int,
    answers: dict[str, object],
    status: str = "in_progress",
) -> int:
    user = await bot._user(telegram_id)
    async with bot.db.session() as session:
        state = await OnboardingRepository(session).get_or_create(user.id)
        state.current_step = step
        state.answers = answers
        state.status = status
    return user.id


def complete_answers_with_cached_summary() -> dict[str, object]:
    answers: dict[str, object] = complete_answers()
    answers["__onboarding_flow__"] = {
        "summary": {
            "summary": "Спокойная жизнь, своё дело и время для семьи.",
            "values": ["Семья", "Свобода", "Польза"],
            "desired_identity": ["Последовательный человек"],
            "constraints": ["Распыление внимания"],
            "motivation_style": "Спокойно, прямо и по шагам.",
        }
    }
    return answers


async def test_fresh_instance_resumes_long_multiline_answer_before_generic_routing(db, fake_ai):
    first = make_bot(db, fake_ai)
    await seed_onboarding(
        first,
        telegram_id=8801,
        step=2,
        answers={"display_name": "Назар", "timezone": "Europe/Moscow"},
    )
    long_answer = (
        "Я улучшаю мир вокруг себя и стараюсь приносить пользу людям.\n\n"
        "Организую школу, где дети учатся с детства работать с ИИ, "
        "много занимаются спортом и создают собственные проекты.\n\n"
        "Моя семья растёт, дети счастливы, а у меня остаётся время на близких, "
        "здоровье и осмысленную работу."
    )

    # A new bot and empty user_data model a process restart: no PTB conversation
    # state exists, only the durable onboarding row remains.
    restarted = make_bot(db, fake_ai)
    message = FakeMessage(long_answer)
    update = update_for(message, update_id=501)
    ctx = context()
    with pytest.raises(ApplicationHandlerStop):
        await restarted.navigation_text_gate(update, ctx)

    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(
                OnboardingState.user_id == ctx.user_data["onboarding_user_id"]
            )
        )
        draft_count = await session.scalar(select(func.count(DraftInboxItem.id)))
    assert state.current_step == 3
    assert state.answers["future_life"] == long_answer
    assert state.status == "in_progress"
    assert draft_count == 0
    assert fake_ai.route_calls == []
    assert any("Ответ сохранён" in reply["text"] for reply in message.replies)
    assert any("Шаг 4 из" in reply["text"] for reply in message.replies)


async def test_duplicate_delivery_does_not_consume_the_next_question(db, fake_ai):
    bot = make_bot(db, fake_ai)
    await seed_onboarding(
        bot,
        telegram_id=8802,
        step=2,
        answers={"display_name": "Назар", "timezone": "Europe/Moscow"},
    )
    message = FakeMessage("Большой ответ\nсо вторым абзацем")
    update = update_for(message, user_id=8802, update_id=777)
    ctx = context()

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update, ctx)
    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update, ctx)

    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(
                OnboardingState.user_id == ctx.user_data["onboarding_user_id"]
            )
        )
    assert state.current_step == 3
    assert state.answers["future_life"] == message.text
    assert "residence" not in state.answers
    assert any("уже сохранён" in reply["text"] for reply in message.replies)


async def test_drafts_command_cannot_escape_persisted_onboarding(db, fake_ai):
    bot = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8808,
        step=2,
        answers={"display_name": "Назар", "timezone": "Europe/Moscow"},
    )
    message = FakeMessage("/drafts")
    with pytest.raises(ApplicationHandlerStop):
        await bot.onboarding_command_gate(update_for(message, user_id=8808), context())

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
        draft_count = await session.scalar(select(func.count(DraftInboxItem.id)))
    assert state.current_step == 2
    assert draft_count == 0
    assert any("не завершён сценарий" in reply["text"] for reply in message.replies)


@pytest.mark.parametrize("attached", [True, False])
@pytest.mark.parametrize(
    ("phrase", "expected_text"),
    [
        ("Как пользоваться ботом?", "❓ Помощь"),
        ("Открой меню", "не завершён сценарий"),
    ],
)
async def test_natural_help_and_menu_do_not_become_onboarding_answers(
    db, fake_ai, attached, phrase, expected_text
):
    bot = make_bot(db, fake_ai)
    original_answers = {"display_name": "Назар", "timezone": "Europe/Moscow"}
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8820,
        step=2,
        answers=original_answers,
    )
    ctx = context()
    if attached:
        ctx.user_data["onboarding_user_id"] = owner_id
    message = FakeMessage(phrase)

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update_for(message, user_id=8820), ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == 2
    assert state.status == "in_progress"
    assert state.answers == original_answers
    assert any(expected_text in reply["text"] for reply in message.replies)
    if phrase == "Открой меню":
        assert not any("Главное меню" in reply["text"] for reply in message.replies)
    assert not any("Ответ сохранён" in reply["text"] for reply in message.replies)
    assert fake_ai.route_calls == []


@pytest.mark.parametrize("attached", [True, False])
async def test_section_navigation_does_not_become_a_durable_onboarding_answer_after_restart(
    db,
    fake_ai,
    attached,
):
    bot = make_bot(db, fake_ai)
    original_answers = {"display_name": "Назар", "timezone": "Europe/Moscow"}
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8825,
        step=2,
        answers=original_answers,
    )
    ctx = context()
    if attached:
        ctx.user_data["onboarding_user_id"] = owner_id
    message = FakeMessage("Покажи мои записи")

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update_for(message, user_id=8825), ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
        draft_count = await session.scalar(select(func.count(DraftInboxItem.id)))
    assert state.current_step == 2
    assert state.status == "in_progress"
    assert state.answers == original_answers
    assert draft_count == 0
    assert any("не завершён сценарий" in reply["text"] for reply in message.replies)
    assert not any("Ответ сохранён" in reply["text"] for reply in message.replies)
    assert fake_ai.route_calls == []


@pytest.mark.parametrize("attached", [True, False])
@pytest.mark.parametrize(
    ("command", "expected_text"),
    [("/help", "❓ Помощь"), ("/menu", "Главное меню")],
)
async def test_help_and_menu_commands_leave_durable_onboarding_unchanged(
    db, fake_ai, attached, command, expected_text
):
    bot = make_bot(db, fake_ai)
    original_answers = {"display_name": "Назар", "timezone": "Europe/Moscow"}
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8821,
        step=2,
        answers=original_answers,
    )
    ctx = context()
    if attached:
        ctx.user_data["onboarding_user_id"] = owner_id
    message = FakeMessage(command)

    with pytest.raises(ApplicationHandlerStop):
        await bot.onboarding_command_gate(update_for(message, user_id=8821), ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == 2
    assert state.status == "in_progress"
    assert state.answers == original_answers
    assert any(expected_text in reply["text"] for reply in message.replies)
    assert not any("не завершён сценарий" in reply["text"] for reply in message.replies)
    if command == "/help":
        assert "nav:help:quick" in callback_values(message)


@pytest.mark.parametrize("attached", [True, False])
@pytest.mark.parametrize(
    ("phrase", "expected_text"),
    [
        ("Помощь", "❓ Помощь"),
        ("Главное меню", "не завершён сценарий"),
    ],
)
async def test_voice_help_and_menu_do_not_become_onboarding_answers(
    db, fake_ai, attached, phrase, expected_text
):
    first = make_bot(db, fake_ai)
    original_answers = {"display_name": "Назар", "timezone": "Europe/Moscow"}
    owner_id = await seed_onboarding(
        first,
        telegram_id=8822,
        step=2,
        answers=original_answers,
    )
    transcription = ScriptedTranscription()
    transcription.queue(phrase)
    bot = FutureSelfBot(first.settings, db, fake_ai, transcription)
    ctx = context()
    if attached:
        ctx.user_data["onboarding_user_id"] = owner_id
    message = FakeMessage(voice=FakeVoice())

    result = await bot.voice(update_for(message, user_id=8822), ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert result == ONBOARDING_INPUT
    assert state.current_step == 2
    assert state.status == "in_progress"
    assert state.answers == original_answers
    assert transcription.calls == [(14, "autotest.ogg")]
    assert message.reply_text_calls == 1
    assert message.replies[0]["text"] == "Расшифровываю голосовую мысль…"
    assert len(message.edits) == 1
    assert expected_text in message.edits[0]
    assert not any("Ответ сохранён" in reply["text"] for reply in message.replies)
    assert fake_ai.route_calls == []


async def test_help_callbacks_remain_usable_during_durable_onboarding(db, fake_ai):
    bot = make_bot(db, fake_ai)
    original_answers = {"display_name": "Назар", "timezone": "Europe/Moscow"}
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8823,
        step=2,
        answers=original_answers,
    )
    ctx = context()
    message = FakeMessage()

    help_query = FakeCallbackQuery("nav:help", message)
    await bot.navigation_action(
        update_for(message, user_id=8823, query=help_query),
        ctx,
    )
    assert help_query.edits[-1].startswith("❓ Помощь")
    assert "nav:help:quick" in callback_values(message)

    topic_query = FakeCallbackQuery("nav:help:quick", message)
    await bot.navigation_action(
        update_for(message, user_id=8823, query=topic_query),
        ctx,
    )
    assert topic_query.edits[-1].startswith("🚀 Быстрый старт")
    assert "nav:help" in callback_values(message)

    back_query = FakeCallbackQuery("nav:help", message)
    await bot.navigation_action(
        update_for(message, user_id=8823, query=back_query),
        ctx,
    )
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == 2
    assert state.status == "in_progress"
    assert state.answers == original_answers
    assert all(
        query.answers[-1] == (None, False) for query in (help_query, topic_query, back_query)
    )
    assert back_query.edits[-1].startswith("❓ Помощь")
    assert message.reply_text_calls == 0
    assert not any("не завершён сценарий" in reply["text"] for reply in message.replies)


async def test_onboarding_flow_exit_is_durable_and_start_resumes_saved_step(db, fake_ai):
    bot = make_bot(db, fake_ai)
    original_answers = {"display_name": "Назар", "timezone": "Europe/Moscow"}
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8824,
        step=2,
        answers=original_answers,
    )
    ctx = SimpleNamespace(
        user_data={
            "onboarding_user_id": owner_id,
            "onboarding_detached": True,
            "vision_summary": {"temporary": True},
            "unrelated": "keep",
        },
        args=[],
    )
    message = FakeMessage()
    update = update_for(message, user_id=8824)
    await bot._prompt_navigation_flow(message, update, "onboarding")
    exit_callback = next(
        value for value in callback_values(message) if value.startswith("nav:flow:exit:")
    )
    query = FakeCallbackQuery(exit_callback, message)

    result = await bot.navigation_action(
        update_for(message, user_id=8824, query=query),
        ctx,
    )

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert result == ConversationHandler.END
    assert state.status == "cancelled"
    assert state.current_step == 2
    assert state.answers == original_answers
    assert "onboarding_user_id" not in ctx.user_data
    assert "onboarding_detached" not in ctx.user_data
    assert "vision_summary" not in ctx.user_data
    assert ctx.user_data["unrelated"] == "keep"
    assert await bot._active_navigation_flow(update, ctx) is None
    assert message.reply_text_calls == 1
    assert not any("Регистрация приостановлена" in reply["text"] for reply in message.replies)
    assert query.edits[-1].startswith("Главное меню")

    restarted = make_bot(db, fake_ai)
    await AccessService(db).grant_subscriber(8824, source="test")
    resume_context = context()
    resume_message = FakeMessage("/start")
    resume_result = await restarted.start(
        update_for(resume_message, user_id=8824),
        resume_context,
    )
    async with db.sessions() as session:
        resumed = await OnboardingRepository(session).get(owner_id)
    assert resume_result == ONBOARDING_INPUT
    assert resumed.status == "in_progress"
    assert resumed.current_step == 2
    assert resumed.answers == original_answers
    assert resume_context.user_data["onboarding_user_id"] == owner_id
    assert any("Продолжим с шага 3 из" in reply["text"] for reply in resume_message.replies)
    assert any("Шаг 3 из" in reply["text"] for reply in resume_message.replies)


async def test_live_profile_confirm_consumes_arbitrary_text_without_draft_routing(db, fake_ai):
    bot = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8809,
        step=len(ONBOARDING_QUESTIONS),
        answers=complete_answers_with_cached_summary(),
        status="awaiting_confirmation",
    )
    message = FakeMessage("Сохрани это как новую задачу")
    ctx = SimpleNamespace(user_data={"onboarding_user_id": owner_id}, args=[])

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(update_for(message, user_id=8809), ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
        draft_count = await session.scalar(select(func.count(DraftInboxItem.id)))
    assert state.current_step == len(ONBOARDING_QUESTIONS)
    assert state.status == "awaiting_confirmation"
    assert draft_count == 0
    assert fake_ai.route_calls == []
    assert ctx.user_data["onboarding_detached"] is True
    assert any(
        isinstance(reply.get("reply_markup"), ReplyKeyboardRemove) for reply in message.replies
    )
    assert any("Все вопросы пройдены" in reply["text"] for reply in message.replies)


@pytest.mark.parametrize(
    ("text", "is_command", "expected_step"),
    [
        ("Назад", False, len(ONBOARDING_QUESTIONS) - 1),
        ("Пропустить", False, len(ONBOARDING_QUESTIONS)),
        ("/back", True, len(ONBOARDING_QUESTIONS) - 1),
        ("/skip", True, len(ONBOARDING_QUESTIONS)),
    ],
)
async def test_live_profile_confirm_owns_old_reply_buttons_and_commands(
    db, fake_ai, text, is_command, expected_step
):
    telegram_id = 8810
    bot = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        bot,
        telegram_id=telegram_id,
        step=len(ONBOARDING_QUESTIONS),
        answers=complete_answers_with_cached_summary(),
        status="awaiting_confirmation",
    )
    message = FakeMessage(text)
    update = update_for(message, user_id=telegram_id)
    ctx = SimpleNamespace(user_data={"onboarding_user_id": owner_id}, args=[])

    with pytest.raises(ApplicationHandlerStop):
        if is_command:
            await bot.onboarding_command_gate(update, ctx)
        else:
            await bot.navigation_text_gate(update, ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
        draft_count = await session.scalar(select(func.count(DraftInboxItem.id)))
    assert state.current_step == expected_step
    assert draft_count == 0
    assert fake_ai.route_calls == []
    assert ctx.user_data["onboarding_detached"] is True


async def test_profile_confirmation_survives_restart_and_menu_survives_goal_failure(db, fake_ai):
    bot = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8803,
        step=len(ONBOARDING_QUESTIONS),
        answers=complete_answers(),
        status="awaiting_confirmation",
    )

    async def fail_goals(_summary):
        raise RuntimeError("provider unavailable")

    fake_ai.propose_goals = fail_goals
    restarted = make_bot(db, fake_ai)
    message = FakeMessage()
    query = FakeCallbackQuery("profile:confirm", message)
    result = await restarted.profile_action(
        update_for(message, user_id=8803, query=query), context()
    )

    assert result == ConversationHandler.END
    async with db.sessions() as session:
        user = await session.get(User, owner_id)
        state = await OnboardingRepository(session).get(owner_id)
        profile = await session.scalar(
            select(VisionProfile).where(VisionProfile.user_id == owner_id)
        )
    assert user.onboarding_completed is True
    assert state.status == "completed"
    assert profile is not None
    assert profile.raw_answers == complete_answers()
    assert any("Главное меню" in reply["text"] for reply in message.replies)
    assert any("цели пока не удалось" in reply["text"] for reply in message.replies)


async def test_summary_failure_keeps_final_answer_and_resumes_on_start(db, fake_ai):
    bot = make_bot(db, fake_ai)
    answers = complete_answers()
    answers.pop("location")
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8804,
        step=len(ONBOARDING_QUESTIONS) - 1,
        answers=answers,
    )
    working_summary = fake_ai.summarize_vision

    async def fail_summary(_answers):
        raise RuntimeError("provider unavailable")

    fake_ai.summarize_vision = fail_summary
    final_message = FakeMessage("Москва")
    ctx = SimpleNamespace(user_data={"onboarding_user_id": owner_id}, args=[])
    result = await bot.onboarding_answer(
        update_for(final_message, user_id=8804, update_id=900), ctx
    )
    assert result == ConversationHandler.END
    assert ctx.user_data["onboarding_detached"] is True

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == len(ONBOARDING_QUESTIONS)
    assert state.status == "awaiting_confirmation"
    assert state.answers["location"] == "Москва"
    assert any("Регистрация не потеряна" in reply["text"] for reply in final_message.replies)

    fake_ai.summarize_vision = working_summary
    restarted = make_bot(db, fake_ai)
    await AccessService(db).grant_subscriber(8804, source="test")
    resume_message = FakeMessage("/start")
    resume_result = await restarted.start(update_for(resume_message, user_id=8804), context())
    assert resume_result == PROFILE_CONFIRM
    assert any("Все вопросы пройдены" in reply["text"] for reply in resume_message.replies)


async def test_oversized_ai_summary_is_bounded_cached_and_still_reaches_confirmation(db, fake_ai):
    bot = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8811,
        step=len(ONBOARDING_QUESTIONS),
        answers=complete_answers(),
        status="awaiting_confirmation",
    )

    async def oversized_summary(_answers):
        return VisionSummary(
            summary="🌟" * 6_000,
            values=["💫" * 800 for _ in range(20)],
            desired_identity=["🚀" * 800 for _ in range(20)],
            constraints=["🧱" * 800 for _ in range(20)],
            motivation_style="✨" * 2_000,
        )

    fake_ai.summarize_vision = oversized_summary
    await AccessService(db).grant_subscriber(8811, source="test")
    message = FakeMessage("/start")
    result = await bot.start(update_for(message, user_id=8811), context())

    assert result == PROFILE_CONFIRM
    assert all(len(reply["text"].encode("utf-16-le")) // 2 <= 4_096 for reply in message.replies)
    confirmation = next(
        reply for reply in message.replies if "Все вопросы пройдены" in reply["text"]
    )
    callbacks = [
        button.callback_data
        for row in confirmation["reply_markup"].inline_keyboard
        for button in row
    ]
    assert "profile:confirm" in callbacks
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    cached = state.answers["__onboarding_flow__"]["summary"]
    assert len(cached["summary"].encode("utf-16-le")) // 2 <= 1_600
    assert len(cached["values"]) == 6


async def test_voice_answer_advances_the_active_conversation_state(db, fake_ai):
    transcription = ScriptedTranscription()
    transcription.queue("Москва")
    bot = FutureSelfBot(
        Settings(
            _env_file=None,
            telegram_bot_token="123456:test-token",
            ai_api_key="test-key",
            database_url="sqlite+aiosqlite:///:memory:",
        ),
        db,
        fake_ai,
        transcription,
    )
    answers = complete_answers()
    answers.pop("location")
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8806,
        step=len(ONBOARDING_QUESTIONS) - 1,
        answers=answers,
    )
    message = FakeMessage(voice=FakeVoice())
    result = await bot.voice(
        update_for(message, user_id=8806, update_id=901),
        SimpleNamespace(user_data={"onboarding_user_id": owner_id}, args=[]),
    )

    assert result == PROFILE_CONFIRM
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == len(ONBOARDING_QUESTIONS)
    assert state.answers["location"] == "Москва"
    assert transcription.calls == [(14, "autotest.ogg")]
    assert "Голос распознан и обработан в регистрации." in message.edits


async def test_voice_answer_resumes_onboarding_after_restart(db, fake_ai):
    first = make_bot(db, fake_ai)
    await seed_onboarding(
        first,
        telegram_id=8807,
        step=2,
        answers={"display_name": "Назар", "timezone": "Europe/Moscow"},
    )
    transcription = ScriptedTranscription()
    transcription.queue("Длинный голосовой ответ о будущем после перезапуска")
    restarted = FutureSelfBot(
        Settings(
            _env_file=None,
            telegram_bot_token="123456:test-token",
            ai_api_key="test-key",
            database_url="sqlite+aiosqlite:///:memory:",
        ),
        db,
        fake_ai,
        transcription,
    )
    message = FakeMessage(voice=FakeVoice())
    ctx = context()
    result = await restarted.voice(update_for(message, user_id=8807, update_id=902), ctx)

    assert result == ONBOARDING_INPUT
    assert ctx.user_data["onboarding_detached"] is True
    async with db.sessions() as session:
        state = await session.scalar(
            select(OnboardingState).where(
                OnboardingState.user_id == ctx.user_data["onboarding_user_id"]
            )
        )
    assert state.current_step == 3
    assert state.answers["future_life"] == ("Длинный голосовой ответ о будущем после перезапуска")
    assert fake_ai.route_calls == []


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_oversized_display_name_is_rejected_without_advancing_after_restart(
    db, fake_ai, source
):
    first = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        first,
        telegram_id=8812,
        step=0,
        answers={},
    )
    long_name = "Я" * 121
    ctx = context()
    if source == "voice":
        transcription = ScriptedTranscription()
        transcription.queue(long_name)
        bot = FutureSelfBot(first.settings, db, fake_ai, transcription)
        message = FakeMessage(voice=FakeVoice())
        result = await bot.voice(update_for(message, user_id=8812, update_id=1_012), ctx)
        assert result == ONBOARDING_INPUT
    else:
        bot = make_bot(db, fake_ai)
        message = FakeMessage(long_name)
        with pytest.raises(ApplicationHandlerStop):
            await bot.navigation_text_gate(update_for(message, user_id=8812, update_id=1_012), ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == 0
    assert "display_name" not in state.answers
    assert fake_ai.route_calls == []
    assert any("не больше 120 символов" in reply["text"] for reply in message.replies)


async def test_aggregate_answer_limit_keeps_the_same_step_and_saved_answers(db, fake_ai):
    bot = make_bot(db, fake_ai)
    answers: dict[str, object] = {
        "display_name": "Назар",
        "timezone": "Europe/Moscow",
        "future_life": "а" * 7_000,
        "residence": "б" * 7_000,
        "work_income": "в" * 7_000,
        "health_body": "г" * 7_000,
    }
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8813,
        step=10,
        answers=answers,
    )
    message = FakeMessage("д" * 3_000)
    result = await bot.onboarding_answer(
        update_for(message, user_id=8813, update_id=1_013),
        SimpleNamespace(user_data={"onboarding_user_id": owner_id}, args=[]),
    )

    assert result == ONBOARDING_INPUT
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == 10
    assert "support_style" not in state.answers
    assert any("вместе получились слишком длинными" in reply["text"] for reply in message.replies)


async def test_legacy_oversized_profile_fields_are_clipped_before_database_write(db, fake_ai):
    bot = make_bot(db, fake_ai)
    answers = complete_answers_with_cached_summary()
    answers["display_name"] = "Очень длинное имя " * 30
    answers["__onboarding_flow__"]["summary"]["motivation_style"] = "м" * 500
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8814,
        step=len(ONBOARDING_QUESTIONS),
        answers=answers,
        status="awaiting_confirmation",
    )
    message = FakeMessage()
    query = FakeCallbackQuery("profile:confirm", message)
    result = await bot.profile_action(update_for(message, user_id=8814, query=query), context())

    assert result == ConversationHandler.END
    async with db.sessions() as session:
        user = await session.get(User, owner_id)
        profile = await session.scalar(
            select(VisionProfile).where(VisionProfile.user_id == owner_id)
        )
    assert len(user.display_name) <= 120
    assert len(profile.motivation_style) <= 120
    assert len(profile.raw_answers["display_name"]) <= 120


async def test_legacy_oversized_stored_name_is_clipped_in_returning_greeting(db, fake_ai):
    bot = make_bot(db, fake_ai)
    user = await bot._user(8815)
    async with db.session() as session:
        stored = await session.get(User, user.id)
        stored.display_name = "Ж" * 500
        stored.onboarding_completed = True
    await AccessService(db).grant_subscriber(8815, source="test")
    message = FakeMessage("/start")
    result = await bot.start(update_for(message, user_id=8815), context())

    assert result == ConversationHandler.END
    greeting = message.replies[-1]["text"]
    assert greeting == f"С возвращением, {'Ж' * 120}!"


async def test_duplicate_skip_delivery_does_not_skip_two_optional_questions(db, fake_ai):
    bot = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8816,
        step=3,
        answers={
            "display_name": "Назар",
            "timezone": "Europe/Moscow",
            "future_life": "Спокойная жизнь",
        },
    )
    message = FakeMessage("/skip")
    update = update_for(message, user_id=8816, update_id=1_016)
    ctx = context()

    with pytest.raises(ApplicationHandlerStop):
        await bot.onboarding_command_gate(update, ctx)
    with pytest.raises(ApplicationHandlerStop):
        await bot.onboarding_command_gate(update, ctx)

    async with db.sessions() as session:
        state = await OnboardingRepository(session).get(owner_id)
    assert state.current_step == 4
    assert "residence" not in state.answers
    assert "work_income" not in state.answers
    assert any("уже обработана" in reply["text"] for reply in message.replies)


async def test_forged_early_profile_confirmation_cannot_complete_onboarding(db, fake_ai):
    bot = make_bot(db, fake_ai)
    owner_id = await seed_onboarding(
        bot,
        telegram_id=8805,
        step=1,
        answers={"display_name": "Назар"},
    )
    message = FakeMessage()
    query = FakeCallbackQuery("profile:confirm", message)
    result = await bot.profile_action(update_for(message, user_id=8805, query=query), context())

    assert result == ConversationHandler.END
    async with db.sessions() as session:
        user = await session.get(User, owner_id)
        profile_count = await session.scalar(
            select(func.count(VisionProfile.id)).where(VisionProfile.user_id == owner_id)
        )
    assert user.onboarding_completed is False
    assert profile_count == 0
    assert query.answers[-1][1] is True


def test_empty_required_answer_is_rejected_without_advancing():
    from future_self.domain import OnboardingFlow

    with pytest.raises(ValueError, match="пустым"):
        OnboardingFlow.answer({}, 0, " \n\t ")
