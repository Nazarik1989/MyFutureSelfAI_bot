import json
from typing import Literal, Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from . import prompts
from .config import Settings
from .schemas import (
    AssistantAnswer,
    GoalProposals,
    GuestFirstStep,
    GuestThoughtBreakdown,
    IntentResult,
    ParsedThought,
    RoutineProposals,
    TimezoneResolution,
    TodayPlan,
    VisionSummary,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
GUEST_DEMO_MAX_INPUT_CHARS = 1200


def _guest_demo_input(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("guest demo input must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("guest demo input must not be empty")
    if len(cleaned) > GUEST_DEMO_MAX_INPUT_CHARS:
        raise ValueError(
            f"guest demo input must not exceed {GUEST_DEMO_MAX_INPUT_CHARS} characters"
        )
    return cleaned


class ProviderHealthCheck(BaseModel):
    ok: Literal[True]


class AIService(Protocol):
    async def health_check(self) -> ProviderHealthCheck: ...

    async def summarize_vision(self, answers: dict[str, str]) -> VisionSummary: ...

    async def resolve_timezone(self, location_text: str) -> TimezoneResolution: ...

    async def propose_goals(self, profile: VisionSummary) -> GoalProposals: ...

    async def propose_routines(self, goals: GoalProposals) -> RoutineProposals: ...

    async def parse_thought(self, text: str) -> ParsedThought: ...

    async def guest_thought_breakdown(self, text: str) -> GuestThoughtBreakdown: ...

    async def guest_first_step(self, text: str) -> GuestFirstStep: ...

    async def make_today_plan(self, context: dict[str, object]) -> TodayPlan: ...

    async def route_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> IntentResult: ...

    async def answer_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> AssistantAnswer: ...


class OpenAICompatibleAIService:
    """Structured-output adapter for OpenAI-compatible text endpoints."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        tone: str = "спокойный и конкретный",
    ):
        self.client = client
        self.model = model
        self.tone = tone

    async def _parse(
        self,
        schema: type[SchemaT],
        system: str,
        user: str,
        *,
        max_retries: int | None = None,
    ) -> SchemaT:
        client = (
            self.client
            if max_retries is None
            else self.client.with_options(max_retries=max_retries)
        )
        response = await client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": f"{system}\nСтиль ответа: {self.tone}."},
                {"role": "user", "content": user},
            ],
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("The model returned no structured output")
        return parsed

    async def summarize_vision(self, answers: dict[str, str]) -> VisionSummary:
        return await self._parse(VisionSummary, prompts.VISION_SYSTEM, repr(answers))

    async def resolve_timezone(self, location_text: str) -> TimezoneResolution:
        return await self._parse(TimezoneResolution, prompts.TIMEZONE_SYSTEM, location_text)

    async def health_check(self) -> ProviderHealthCheck:
        return await self._parse(
            ProviderHealthCheck,
            "Проверка доступности structured output. Верни ok=true.",
            "Проверка.",
        )

    async def propose_goals(self, profile: VisionSummary) -> GoalProposals:
        return await self._parse(GoalProposals, prompts.GOALS_SYSTEM, profile.model_dump_json())

    async def propose_routines(self, goals: GoalProposals) -> RoutineProposals:
        return await self._parse(RoutineProposals, prompts.ROUTINES_SYSTEM, goals.model_dump_json())

    async def parse_thought(self, text: str) -> ParsedThought:
        return await self._parse(ParsedThought, prompts.INBOX_SYSTEM, text)

    async def guest_thought_breakdown(self, text: str) -> GuestThoughtBreakdown:
        return await self._parse(
            GuestThoughtBreakdown,
            prompts.GUEST_THOUGHT_SYSTEM,
            _guest_demo_input(text),
            max_retries=0,
        )

    async def guest_first_step(self, text: str) -> GuestFirstStep:
        return await self._parse(
            GuestFirstStep,
            prompts.GUEST_FIRST_STEP_SYSTEM,
            _guest_demo_input(text),
            max_retries=0,
        )

    async def make_today_plan(self, context: dict[str, object]) -> TodayPlan:
        return await self._parse(TodayPlan, prompts.TODAY_SYSTEM, repr(context))

    async def route_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> IntentResult:
        payload = {
            "message": text,
            "temporal_context": temporal_context,
            "conversation_context": conversation_context or {},
        }
        return await self._parse(
            IntentResult, prompts.INTENT_SYSTEM, json.dumps(payload, ensure_ascii=False)
        )

    async def answer_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> AssistantAnswer:
        payload = {
            "message": text,
            "temporal_context": temporal_context,
            "conversation_context": conversation_context or {},
        }
        return await self._parse(
            AssistantAnswer,
            prompts.ANSWER_SYSTEM,
            json.dumps(payload, ensure_ascii=False),
        )


def create_ai_service(settings: Settings) -> OpenAICompatibleAIService:
    headers: dict[str, str] = {}
    if settings.ai_provider == "openrouter":
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name
    client_kwargs: dict[str, object] = {
        "api_key": settings.ai_api_key,
        "base_url": settings.ai_base_url,
    }
    if headers:
        client_kwargs["default_headers"] = headers
    client = AsyncOpenAI(**client_kwargs)
    return OpenAICompatibleAIService(client, settings.ai_model, settings.bot_tone)
