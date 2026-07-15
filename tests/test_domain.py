from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from future_self.domain import (
    FocusService,
    InboxService,
    OnboardingFlow,
    next_notification_utc,
)
from future_self.drafts import DraftInboxService
from future_self.models import Goal, InboxItem, Routine, User
from future_self.repositories import ProfileRepository, UserRepository
from future_self.schemas import ParsedThought, VisionSummary


async def test_creates_user_once(db):
    async with db.session() as session:
        repository = UserRepository(session)
        first = await repository.get_or_create(1001, "Europe/Moscow")
        first_id = first.id
    async with db.session() as session:
        second = await UserRepository(session).get_or_create(1001, "UTC")
        count = await session.scalar(select(func.count(User.id)))
    assert second.id == first_id
    assert second.timezone == "Europe/Moscow"
    assert count == 1


async def test_saves_and_updates_vision_profile(db):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(1, "UTC")
        profile = await ProfileRepository(session).upsert(
            user,
            {"future_life": "Живу у моря"},
            VisionSummary(summary="Живу у моря", values=["свобода"]),
        )
        profile_id = profile.id
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(1, "UTC")
        updated = await ProfileRepository(session).upsert(
            user,
            {"future_life": "Живу у моря и работаю"},
            VisionSummary(summary="Живу у моря и работаю", values=["свобода"]),
        )
    assert updated.id == profile_id
    assert updated.summary == "Живу у моря и работаю"
    assert user.onboarding_completed is True


def test_onboarding_transitions_and_required_skip():
    answers = OnboardingFlow.answer({}, 0, "Аня")
    assert answers == {"display_name": "Аня"}
    assert OnboardingFlow.next_step(0) == 1
    assert OnboardingFlow.previous_step(1) == 0
    with pytest.raises(ValueError, match="нельзя пропустить"):
        OnboardingFlow.answer(answers, 0, None)
    optional = OnboardingFlow.answer(answers, 3, None)
    assert "residence" not in optional


async def test_classifies_thought_with_fake_ai(db, fake_ai):
    service = InboxService(db, fake_ai, "UTC")
    parsed = await service.classify("Нужно сделать отчёт")
    assert parsed.kind == "task"
    assert parsed.title == "Нужно сделать отчёт"


async def test_confirmation_and_discard_are_idempotent(db, fake_ai):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(10, "UTC")
        user_id = user.id
    service = DraftInboxService(db, 60)
    parsed = ParsedThought(kind="task", title="Отчёт", next_step="Открыть документ")
    accepted = await service.create(
        user_id=user_id,
        telegram_user_id=10,
        chat_id=100,
        source="text",
        raw_text="Сделать отчёт",
        parsed=parsed,
    )
    assert (await service.confirm(accepted.id, 1, 10, 100)).ok is True
    assert (await service.confirm(accepted.id, 1, 10, 100)).ok is False
    dropped = await service.create(
        user_id=user_id,
        telegram_user_id=10,
        chat_id=100,
        source="text",
        raw_text="Идея",
        parsed=parsed,
    )
    assert (await service.drop(dropped.id, 1, 10, 100)).ok is True
    assert (await service.drop(dropped.id, 1, 10, 100)).ok is False
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1


async def test_user_data_is_isolated(db, fake_ai):
    async with db.session() as session:
        first_user = await UserRepository(session).get_or_create(10, "UTC")
        second_user = await UserRepository(session).get_or_create(20, "UTC")
        first_id, second_id = first_user.id, second_user.id
    service = DraftInboxService(db, 60)
    first_draft = await service.create(
        user_id=first_id,
        telegram_user_id=10,
        chat_id=100,
        source="text",
        raw_text="Сделать А",
        parsed=ParsedThought(kind="task", title="А"),
    )
    second_draft = await service.create(
        user_id=second_id,
        telegram_user_id=20,
        chat_id=200,
        source="text",
        raw_text="Сделать Б",
        parsed=ParsedThought(kind="task", title="Б"),
    )
    await service.confirm(first_draft.id, 1, 10, 100)
    await service.confirm(second_draft.id, 1, 20, 200)
    async with db.sessions() as session:
        first = await session.scalar(select(User).where(User.telegram_id == 10))
        titles = list(
            (
                await session.scalars(select(InboxItem.title).where(InboxItem.user_id == first.id))
            ).all()
        )
    assert titles == ["А"]


async def test_today_uses_only_confirmed_tasks_and_active_items(db, fake_ai):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(30, "UTC")
        session.add_all(
            [
                Goal(
                    user_id=user.id,
                    life_area="здоровье",
                    title="Двигаться",
                    outcome="Больше энергии",
                    progress_criterion="3 раза",
                    horizon="месяц",
                    status="active",
                    priority=5,
                    vision_link="Энергичная жизнь",
                ),
                InboxItem(
                    user_id=user.id,
                    kind="task",
                    title="Подтверждено",
                    raw_text="текст",
                    source="text",
                    status="confirmed",
                ),
                InboxItem(
                    user_id=user.id,
                    kind="task",
                    title="Черновик",
                    raw_text="текст",
                    source="text",
                    status="pending",
                ),
            ]
        )
        await session.flush()
        goal = await session.scalar(select(Goal).where(Goal.user_id == user.id))
        session.add(
            Routine(
                user_id=user.id,
                goal_id=goal.id,
                frequency="ежедневно",
                minimum_version="2 минуты",
                normal_version="15 минут ходьбы",
                status="active",
            )
        )
        user_id = user.id
    plan = await FocusService(db, fake_ai).generate(user_id)
    assert plan.main_focus == "Один устойчивый шаг"
    assert fake_ai.last_today_context["confirmed_tasks"] == ["Подтверждено"]
    assert fake_ai.last_today_context["goals"] == ["Двигаться"]


@pytest.mark.parametrize(
    ("zone", "hour", "now", "expected"),
    [
        (
            "Europe/Moscow",
            8,
            datetime(2026, 1, 10, 2, tzinfo=UTC),
            datetime(2026, 1, 10, 5, tzinfo=UTC),
        ),
        (
            "Europe/Berlin",
            8,
            datetime(2026, 7, 10, 5, tzinfo=UTC),
            datetime(2026, 7, 10, 6, tzinfo=UTC),
        ),
        (
            "America/New_York",
            8,
            datetime(2026, 1, 10, 14, tzinfo=UTC),
            datetime(2026, 1, 11, 13, tzinfo=UTC),
        ),
    ],
)
def test_notification_timezones(zone, hour, now, expected):
    assert next_notification_utc(zone, hour, now=now) == expected
