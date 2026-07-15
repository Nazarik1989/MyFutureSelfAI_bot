from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from future_self.db import Database
from future_self.schemas import (
    AssistantAnswer,
    GoalProposal,
    GoalProposals,
    IntentResult,
    ParsedThought,
    RoutineProposal,
    RoutineProposals,
    TodayPlan,
    VisionSummary,
)


class FakeAI:
    def __init__(self) -> None:
        self.last_today_context: dict[str, object] | None = None
        self.route_calls: list[tuple[str, dict[str, str]]] = []
        self.conversation_contexts: list[dict[str, object]] = []

    async def summarize_vision(self, answers: dict[str, str]) -> VisionSummary:
        return VisionSummary(
            summary=answers["future_life"],
            values=[answers["values"]],
            desired_identity=["человек, который действует последовательно"],
            constraints=[answers["obstacles"]] if answers.get("obstacles") else [],
            motivation_style=answers.get("support_style"),
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
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await database.create_all_for_tests()
    yield database
    await database.dispose()
