import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from future_self.db import Database
from future_self.schemas import (
    AssistantAnswer,
    GoalProposal,
    GoalProposals,
    GuestFirstStep,
    GuestThoughtBreakdown,
    IntentResult,
    ParsedThought,
    RoutineProposal,
    RoutineProposals,
    TimezoneResolution,
    TodayPlan,
    VisionSummary,
)


class FakeAI:
    def __init__(self) -> None:
        self.last_today_context: dict[str, object] | None = None
        self.route_calls: list[tuple[str, dict[str, str]]] = []
        self.conversation_contexts: list[dict[str, object]] = []
        self.timezone_calls: list[str] = []
        self.timezone_results: dict[str, TimezoneResolution] = {}
        self.guest_thought_calls = 0
        self.guest_first_step_calls = 0
        self.guest_thought_result = GuestThoughtBreakdown(
            category="idea",
            title="Спокойная мысль",
            essence="Краткая суть мысли",
            next_step="Записать один небольшой шаг",
        )
        self.guest_first_step_result = GuestFirstStep(
            focus="Главный фокус",
            first_step="Уделить задаче пять минут",
            actions=["Открыть заметки"],
        )
        self.guest_thought_error: Exception | None = None
        self.guest_first_step_error: Exception | None = None
        self.guest_thought_started = asyncio.Event()
        self.guest_first_step_started = asyncio.Event()
        self.guest_thought_release = asyncio.Event()
        self.guest_first_step_release = asyncio.Event()
        self.guest_thought_release.set()
        self.guest_first_step_release.set()

    async def summarize_vision(self, answers: dict[str, str]) -> VisionSummary:
        return VisionSummary(
            summary=answers["future_life"],
            values=[answers["values"]],
            desired_identity=["человек, который действует последовательно"],
            constraints=[answers["obstacles"]] if answers.get("obstacles") else [],
            motivation_style=answers.get("support_style"),
        )

    async def resolve_timezone(self, location_text: str) -> TimezoneResolution:
        self.timezone_calls.append(location_text)
        return self.timezone_results.get(
            location_text,
            TimezoneResolution(timezone=None, city=None, country=None, ambiguous=True),
        )

    async def propose_goals(self, profile: VisionSummary) -> GoalProposals:
        return GoalProposals(
            goals=[
                GoalProposal(
                    life_area="здоровье",
                    title=f"Цель {index}",
                    outcome="Устойчивый результат",
                    progress_criterion="3 раза в неделю",
                    horizon="3 месяца",
                    priority=5 - index,
                    vision_link=profile.summary,
                )
                for index in range(3)
            ]
        )

    async def propose_routines(self, goals: GoalProposals) -> RoutineProposals:
        return RoutineProposals(
            routines=[
                RoutineProposal(
                    goal_title=goal.title,
                    frequency="ежедневно",
                    minimum_version="2 минуты",
                    normal_version="15 минут",
                    preferred_time="утро",
                )
                for goal in goals.goals[:3]
            ]
        )

    async def parse_thought(self, text: str) -> ParsedThought:
        kind = "task" if "сделать" in text.lower() else "idea"
        return ParsedThought(kind=kind, title=text[:40], next_step="Выбрать первый шаг")

    async def guest_thought_breakdown(self, text: str) -> GuestThoughtBreakdown:
        del text
        self.guest_thought_calls += 1
        self.guest_thought_started.set()
        await self.guest_thought_release.wait()
        if self.guest_thought_error is not None:
            raise self.guest_thought_error
        return self.guest_thought_result

    async def guest_first_step(self, text: str) -> GuestFirstStep:
        del text
        self.guest_first_step_calls += 1
        self.guest_first_step_started.set()
        await self.guest_first_step_release.wait()
        if self.guest_first_step_error is not None:
            raise self.guest_first_step_error
        return self.guest_first_step_result

    async def make_today_plan(self, context: dict[str, object]) -> TodayPlan:
        self.last_today_context = context
        return TodayPlan(
            vision_reminder="Ты строишь спокойную жизнь.",
            main_focus="Один устойчивый шаг",
            actions=["Сделать рутину"],
            hard_day_minimum="Две минуты",
        )

    async def route_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> IntentResult:
        self.route_calls.append((text, temporal_context))
        self.conversation_contexts.append(conversation_context or {})
        lowered = text.lower()
        if "только что" in lowered or "как я говорил" in lowered:
            return IntentResult(
                intent="conversation",
                confidence=0.95,
                answer="Да, мы обсуждали еженедельное планирование.",
                topic="еженедельное планирование",
            )
        if "ты занес" in lowered and "задач" in lowered:
            return IntentResult(
                intent="question",
                confidence=0.98,
                answer="Пока нет. Для записи нужны preview и отдельное подтверждение.",
                topic="создание задачи",
            )
        if "сохрани это" in lowered:
            return IntentResult(
                intent="explicit_capture",
                confidence=0.95,
                inbox_kind="note",
                title="Сохранить обсуждение",
            )
        if "еженедельн" in lowered:
            return IntentResult(
                intent="conversation",
                confidence=0.95,
                answer="Еженедельное планирование поможет выбрать приоритеты.",
                topic="еженедельное планирование",
            )
        if lowered == "привет":
            return IntentResult(intent="conversation", confidence=0.99, answer="Привет!")
        if "какой завтра день недели" in lowered:
            return IntentResult(
                intent="question",
                confidence=0.99,
                answer=f"Завтра {temporal_context['tomorrow_weekday']}.",
            )
        if "иде" in lowered or "пространство" in lowered:
            return IntentResult(
                intent="inbox_idea",
                confidence=0.95,
                inbox_kind="idea",
                title="Совместное пространство",
                next_step="Кратко описать сценарий",
            )
        if "не забудь" in lowered or "сделать" in lowered:
            return IntentResult(
                intent="inbox_task",
                confidence=0.95,
                inbox_kind="task",
                title=text[:40],
                next_step="Выбрать время",
            )
        if "непонятно" in lowered:
            return IntentResult(
                intent="inbox_note", confidence=0.2, inbox_kind="note", title="Неясно"
            )
        return IntentResult(intent="inbox_note", confidence=0.9, inbox_kind="note", title=text[:40])

    async def answer_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> AssistantAnswer:
        return AssistantAnswer(answer=f"Ответ на: {text}")


@pytest.fixture
def fake_ai() -> FakeAI:
    return FakeAI()


@pytest_asyncio.fixture
async def db(tmp_path) -> AsyncIterator[Database]:
    database_path = tmp_path / "test.db"
    database = Database(f"sqlite+aiosqlite:///{database_path}")
    await database.create_all_for_tests()
    yield database
    await database.dispose()
    for suffix in ("", "-wal", "-shm"):
        database_path.with_name(database_path.name + suffix).unlink(missing_ok=True)
