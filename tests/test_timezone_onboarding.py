from types import SimpleNamespace

import pytest

from future_self.bot import ONBOARDING_INPUT, FutureSelfBot
from future_self.config import Settings
from future_self.domain import canonical_timezone
from future_self.repositories import OnboardingRepository


class NoopTranscription:
    enabled = True


class Message:
    def __init__(self, text: str):
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append(text)
        return self


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


@pytest.mark.parametrize(
    ("telegram_id", "answer", "expected"),
    [
        (5001, "Moscow", "Europe/Moscow"),
        (5002, "GMT+4", "Europe/Saratov"),
    ],
)
async def test_onboarding_saves_canonical_timezone(db, fake_ai, telegram_id, answer, expected):
    bot = make_bot(db, fake_ai)
    user = await bot._user(telegram_id)
    async with db.session() as session:
        state = await OnboardingRepository(session).get_or_create(user.id)
        state.current_step = 1
        state.answers = {"display_name": "Варвара"}

    message = Message(answer)
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(user_data={"onboarding_user_id": user.id})
    result = await bot.onboarding_answer(update, context)

    assert result == ONBOARDING_INPUT
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user.id)
        assert state.current_step == 2
        assert state.answers["timezone"] == expected
    assert any("жизнь через три года" in reply for reply in message.replies)


async def test_invalid_timezone_does_not_advance_onboarding(db, fake_ai):
    bot = make_bot(db, fake_ai)
    user = await bot._user(5003)
    async with db.session() as session:
        state = await OnboardingRepository(session).get_or_create(user.id)
        state.current_step = 1
        state.answers = {"display_name": "Варвара"}

    message = Message("GMT+99")
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(user_data={"onboarding_user_id": user.id})
    result = await bot.onboarding_answer(update, context)

    assert result == ONBOARDING_INPUT
    async with db.sessions() as session:
        state = await OnboardingRepository(session).get_or_create(user.id)
        assert state.current_step == 1
        assert "timezone" not in state.answers
    assert "GMT+4" in message.replies[-1]
