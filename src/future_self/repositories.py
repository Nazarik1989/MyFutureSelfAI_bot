from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import DailyCheckIn, Goal, OnboardingState, Routine, User, VisionProfile
from .schemas import GoalProposal, RoutineProposal, VisionSummary


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, telegram_id: int, timezone: str) -> User:
        user = await self.session.scalar(select(User).where(User.telegram_id == telegram_id))
        if user is None:
            user = User(telegram_id=telegram_id, timezone=timezone)
            self.session.add(user)
            await self.session.flush()
        return user

    async def by_telegram_id(self, telegram_id: int) -> User | None:
        return await self.session.scalar(select(User).where(User.telegram_id == telegram_id))


class ProfileRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(
        self, user: User, answers: dict[str, Any], data: VisionSummary
    ) -> VisionProfile:
        profile = await self.session.scalar(
            select(VisionProfile).where(VisionProfile.user_id == user.id)
        )
        values = data.model_dump()
        if profile is None:
            profile = VisionProfile(user_id=user.id, raw_answers=answers, **values)
            self.session.add(profile)
        else:
            profile.raw_answers = answers
            for key, value in values.items():
                setattr(profile, key, value)
            profile.last_updated_at = datetime.now(UTC)
        user.onboarding_completed = True
        await self.session.flush()
        return profile


class GoalRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def replace_proposals(self, user_id: int, proposals: list[GoalProposal]) -> list[Goal]:
        old = (
            await self.session.scalars(
                select(Goal).where(Goal.user_id == user_id, Goal.status == "proposed")
            )
        ).all()
        for goal in old:
            await self.session.delete(goal)
        goals = [
            Goal(user_id=user_id, status="proposed", **item.model_dump()) for item in proposals
        ]
        self.session.add_all(goals)
        await self.session.flush()
        return goals

    async def active(self, user_id: int) -> list[Goal]:
        return list(
            (
                await self.session.scalars(
                    select(Goal)
                    .where(Goal.user_id == user_id, Goal.status == "active")
                    .order_by(Goal.priority.desc(), Goal.id)
                )
            ).all()
        )


class RoutineRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_for_goals(
        self, user_id: int, goals: list[Goal], proposals: list[RoutineProposal]
    ) -> list[Routine]:
        by_title = {goal.title: goal for goal in goals}
        routines: list[Routine] = []
        for item in proposals[:3]:
            goal = by_title.get(item.goal_title)
            if goal:
                data = item.model_dump(exclude={"goal_title"})
                routines.append(
                    Routine(user_id=user_id, goal_id=goal.id, status="proposed", **data)
                )
        self.session.add_all(routines)
        await self.session.flush()
        return routines


class CheckInRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_evening(self, user_id: int, day: date, answers: dict[str, Any]) -> DailyCheckIn:
        checkin = await self.session.scalar(
            select(DailyCheckIn).where(
                DailyCheckIn.user_id == user_id, DailyCheckIn.checkin_date == day
            )
        )
        if checkin is None:
            checkin = DailyCheckIn(user_id=user_id, checkin_date=day)
            self.session.add(checkin)
        for key, value in answers.items():
            setattr(checkin, key, value)
        await self.session.flush()
        return checkin


class OnboardingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, user_id: int) -> OnboardingState:
        state = await self.session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == user_id)
        )
        if state is None:
            state = OnboardingState(user_id=user_id)
            self.session.add(state)
            await self.session.flush()
        return state
