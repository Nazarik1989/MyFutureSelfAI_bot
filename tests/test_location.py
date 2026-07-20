from types import SimpleNamespace

import pytest
from sqlalchemy import select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.location import location_from_user, parse_location
from future_self.models import OnboardingState, User


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
    settings = Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )
    return FutureSelfBot(settings, db, fake_ai, NoopTranscription())


def update_for(telegram_id: int, text: str):
    message = Message(text)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=telegram_id),
        effective_chat=SimpleNamespace(id=telegram_id),
        effective_message=message,
    )
    return update, message


def test_location_parser_accepts_city_and_route_and_rejects_unsafe_input():
    assert parse_location("  Саратов ").label == "Саратов"
    assert parse_location("Светогорск -> Выборг").label == "Светогорск → Выборг"
    with pytest.raises(ValueError):
        parse_location("Саратов; DROP TABLE users")
    with pytest.raises(ValueError):
        parse_location("A → B → C")


async def test_location_command_is_owner_scoped_and_syncs_profile_answers(db, fake_ai):
    bot = make_bot(db, fake_ai)
    first = await bot._user(1001)
    second = await bot._user(1002)
    async with db.session() as session:
        session.add_all(
            [
                OnboardingState(user_id=first.id, answers={"display_name": "Назар"}),
                OnboardingState(user_id=second.id, answers={"display_name": "Варвара"}),
            ]
        )

    first_update, first_message = update_for(1001, "/location Светогорск → Выборг")
    second_update, second_message = update_for(1002, "/location Саратов")
    await bot.location_command(
        first_update,
        SimpleNamespace(args=["Светогорск", "→", "Выборг"], user_data={}),
    )
    await bot.location_command(
        second_update,
        SimpleNamespace(args=["Саратов"], user_data={}),
    )
    assert "Светогорск → Выборг" in first_message.replies[-1]
    assert "Саратов" in second_message.replies[-1]

    async with db.sessions() as session:
        users = list((await session.scalars(select(User).order_by(User.telegram_id))).all())
        states = list(
            (await session.scalars(select(OnboardingState).order_by(OnboardingState.user_id))).all()
        )
    assert [location_from_user(user).label for user in users] == [
        "Светогорск → Выборг",
        "Саратов",
    ]
    assert [state.answers["location"] for state in states] == [
        "Светогорск → Выборг",
        "Саратов",
    ]

    check_first, check_first_message = update_for(1001, "/location")
    check_second, check_second_message = update_for(1002, "/location")
    await bot.location_command(check_first, SimpleNamespace(args=[], user_data={}))
    await bot.location_command(check_second, SimpleNamespace(args=[], user_data={}))
    assert "Саратов" not in check_first_message.replies[-1]
    assert "Светогорск" not in check_second_message.replies[-1]


async def test_location_update_rejects_foreign_user_pair(db, fake_ai):
    bot = make_bot(db, fake_ai)
    first = await bot._user(2001)
    await bot._user(2002)
    with pytest.raises(ValueError, match="не найден"):
        await bot.location_service.set(
            user_id=first.id,
            telegram_user_id=2002,
            value="Саратов",
        )
    async with db.sessions() as session:
        owners = list((await session.scalars(select(User).order_by(User.telegram_id))).all())
    assert all(location_from_user(owner) is None for owner in owners)
