from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from . import prompts
from .config import Settings
from .nova_companion import NovaCompanionContextProjection
from .nova_companion_flow import validate_capture_suggestion
from .nova_memory_application import NovaMemoryProjection
from .schemas import (
    AssistantAnswer,
    GoalProposals,
    GuestFirstStep,
    GuestThoughtBreakdown,
    IntentResult,
    NovaCompanionCapture,
    NovaCompanionProviderResponse,
    NovaCompanionResponse,
    NovaHelpPlan,
    ParsedThought,
    ReminderTimezoneResolution,
    RoutineProposals,
    TimezoneResolution,
    TodayPlan,
    VisionSummary,
    WeeklyReviewExtraction,
)
from .weekly_review_extraction import validate_weekly_review_extraction, weekly_review_input

if TYPE_CHECKING:
    from .nova import NovaCatalog

SchemaT = TypeVar("SchemaT", bound=BaseModel)
GUEST_DEMO_MAX_INPUT_CHARS = 1200
NOVA_HELP_MAX_INPUT_CHARS = 600
NOVA_HELP_TIMEOUT_SECONDS = 30.0
REMINDER_TIMEZONE_MAX_INPUT_CHARS = 120
REMINDER_TIMEZONE_TIMEOUT_SECONDS = 20.0
NOVA_MEMORY_ANSWER_TIMEOUT_SECONDS = 30.0
NOVA_COMPANION_MAX_INPUT_CHARS = 4000
NOVA_COMPANION_TIMEOUT_SECONDS = 30.0
WEEKLY_REVIEW_EXTRACTION_TIMEOUT_SECONDS = 30.0
WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_ITEMS = 12
WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_KEY_CHARS = 64
WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_VALUE_CHARS = 128
WEEKLY_REVIEW_TEMPORAL_CONTEXT_FIELDS = frozenset(
    {
        "timezone",
        "local_datetime",
        "today_date",
        "today_weekday",
        "tomorrow_date",
        "tomorrow_weekday",
        "week_start",
        "week_end",
        "target_week_start",
        "target_week_end",
    }
)
NOVA_COMPANION_TEMPORAL_CONTEXT_FIELDS = frozenset(
    {
        "timezone",
        "local_datetime",
        "today_date",
        "today_weekday",
        "tomorrow_date",
        "tomorrow_weekday",
    }
)


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


def _nova_help_input(question: str) -> str:
    if not isinstance(question, str):
        raise ValueError("Nova help input must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Nova help input must not be empty")
    if len(cleaned) > NOVA_HELP_MAX_INPUT_CHARS:
        raise ValueError(f"Nova help input must not exceed {NOVA_HELP_MAX_INPUT_CHARS} characters")
    return cleaned


def _nova_companion_input(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Nova companion input must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Nova companion input must not be empty")
    if len(cleaned) > NOVA_COMPANION_MAX_INPUT_CHARS:
        raise ValueError(
            f"Nova companion input must not exceed {NOVA_COMPANION_MAX_INPUT_CHARS} characters"
        )
    return cleaned


def _nova_companion_temporal_context(context: dict[str, str]) -> dict[str, str]:
    if not isinstance(context, dict):
        raise ValueError("Nova companion temporal context must be a mapping")
    if len(context) > len(NOVA_COMPANION_TEMPORAL_CONTEXT_FIELDS):
        raise ValueError("Nova companion temporal context has too many fields")
    bounded: dict[str, str] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Nova companion temporal context must contain strings")
        clean_key = key.strip()
        clean_value = value.strip()
        if (
            clean_key not in NOVA_COMPANION_TEMPORAL_CONTEXT_FIELDS
            or not clean_value
            or len(clean_key) > WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_KEY_CHARS
            or len(clean_value) > WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_VALUE_CHARS
            or clean_key in bounded
        ):
            raise ValueError("Nova companion temporal context contains an invalid field")
        bounded[clean_key] = clean_value
    return bounded


def _grounding_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _validated_companion_response(
    text: str,
    parsed: object,
) -> NovaCompanionResponse:
    provider_result = NovaCompanionProviderResponse.model_validate(parsed)
    suggestion = provider_result.capture
    if suggestion is None:
        return NovaCompanionResponse(answer=provider_result.answer)
    message = _grounding_text(text)
    evidence = _grounding_text(suggestion.evidence)
    title = _grounding_text(suggestion.title)
    next_step = _grounding_text(suggestion.next_step) if suggestion.next_step else None
    if (
        not evidence
        or evidence not in message
        or not title
        or title not in evidence
        or (next_step is not None and next_step not in evidence)
    ):
        return NovaCompanionResponse(answer=provider_result.answer)
    validated = validate_capture_suggestion(
        kind=suggestion.kind,
        title=suggestion.title,
        next_step=suggestion.next_step,
        user_text=text,
    )
    if validated is None:
        return NovaCompanionResponse(answer=provider_result.answer)
    return NovaCompanionResponse(
        answer=provider_result.answer,
        capture=NovaCompanionCapture(
            kind=validated.kind,
            title=validated.title,
            next_step=validated.next_step,
        ),
    )


def _reminder_timezone_input(fragment: str) -> str:
    if not isinstance(fragment, str):
        raise ValueError("reminder timezone fragment must be a string")
    cleaned = " ".join(fragment.split())
    if not cleaned:
        raise ValueError("reminder timezone fragment must not be empty")
    if len(cleaned) > REMINDER_TIMEZONE_MAX_INPUT_CHARS:
        raise ValueError(
            "reminder timezone fragment must not exceed "
            f"{REMINDER_TIMEZONE_MAX_INPUT_CHARS} characters"
        )
    return cleaned


def _weekly_review_temporal_context(context: dict[str, str]) -> dict[str, str]:
    if not isinstance(context, dict):
        raise ValueError("weekly review temporal context must be a mapping")
    if len(context) > WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_ITEMS:
        raise ValueError("weekly review temporal context has too many fields")
    bounded: dict[str, str] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("weekly review temporal context must contain strings")
        clean_key = key.strip()
        clean_value = value.strip()
        if (
            not clean_key
            or not clean_value
            or len(clean_key) > WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_KEY_CHARS
            or len(clean_value) > WEEKLY_REVIEW_TEMPORAL_CONTEXT_MAX_VALUE_CHARS
            or clean_key not in WEEKLY_REVIEW_TEMPORAL_CONTEXT_FIELDS
            or clean_key in bounded
        ):
            raise ValueError("weekly review temporal context contains an invalid field")
        bounded[clean_key] = clean_value
    return bounded


def _nova_catalog_payload(capability_catalog: NovaCatalog) -> dict[str, object]:
    try:
        capabilities = capability_catalog.capabilities
        enabled_features = capability_catalog.enabled_features
    except AttributeError as exc:
        raise ValueError("invalid Nova capability catalog") from exc

    serialized_capabilities: list[dict[str, str]] = []
    for capability in capabilities:
        try:
            values = (capability.id, capability.label, capability.description)
        except AttributeError as exc:
            raise ValueError("invalid Nova capability") from exc
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("invalid Nova capability")
        serialized_capabilities.append(
            {
                "id": values[0].strip(),
                "label": values[1].strip(),
                "description": values[2].strip(),
            }
        )

    feature_names: set[str] = set()
    for feature in enabled_features:
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError("invalid Nova runtime feature")
        feature_names.add(feature.strip())

    return {
        "capabilities": serialized_capabilities,
        "enabled_features": sorted(feature_names),
    }


class ProviderHealthCheck(BaseModel):
    ok: Literal[True]


class AIService(Protocol):
    async def health_check(self) -> ProviderHealthCheck: ...

    async def summarize_vision(self, answers: dict[str, str]) -> VisionSummary: ...

    async def resolve_timezone(self, location_text: str) -> TimezoneResolution: ...

    async def resolve_reminder_timezone(
        self,
        timezone_fragment: str,
    ) -> ReminderTimezoneResolution: ...

    async def extract_weekly_review(
        self,
        text: str,
        temporal_context: dict[str, str],
    ) -> WeeklyReviewExtraction: ...

    async def propose_goals(self, profile: VisionSummary) -> GoalProposals: ...

    async def propose_routines(self, goals: GoalProposals) -> RoutineProposals: ...

    async def parse_thought(self, text: str) -> ParsedThought: ...

    async def guest_thought_breakdown(self, text: str) -> GuestThoughtBreakdown: ...

    async def guest_first_step(self, text: str) -> GuestFirstStep: ...

    async def nova_help(self, question: str, capability_catalog: NovaCatalog) -> NovaHelpPlan: ...

    async def companion_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        companion_context: NovaCompanionContextProjection,
    ) -> NovaCompanionResponse: ...

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
        *,
        confirmed_memory: NovaMemoryProjection | None = None,
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

    async def resolve_reminder_timezone(
        self,
        timezone_fragment: str,
    ) -> ReminderTimezoneResolution:
        fragment = _reminder_timezone_input(timezone_fragment)
        client = self.client.with_options(max_retries=0)
        async with asyncio.timeout(REMINDER_TIMEZONE_TIMEOUT_SECONDS):
            response = await client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": f"{prompts.REMINDER_TIMEZONE_SYSTEM}\nСтиль ответа: {self.tone}.",
                    },
                    {"role": "user", "content": fragment},
                ],
                text_format=ReminderTimezoneResolution,
                timeout=REMINDER_TIMEZONE_TIMEOUT_SECONDS,
            )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("The model returned no structured output")
        return ReminderTimezoneResolution.model_validate(parsed)

    async def extract_weekly_review(
        self,
        text: str,
        temporal_context: dict[str, str],
    ) -> WeeklyReviewExtraction:
        clean = weekly_review_input(text)
        payload = {
            "text": clean,
            "temporal_context": _weekly_review_temporal_context(temporal_context),
        }
        client = self.client.with_options(max_retries=0)
        async with asyncio.timeout(WEEKLY_REVIEW_EXTRACTION_TIMEOUT_SECONDS):
            response = await client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": prompts.WEEKLY_REVIEW_EXTRACTION_SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                text_format=WeeklyReviewExtraction,
                timeout=WEEKLY_REVIEW_EXTRACTION_TIMEOUT_SECONDS,
            )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("The model returned no structured output")
        return validate_weekly_review_extraction(
            clean,
            WeeklyReviewExtraction.model_validate(parsed),
        )

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

    async def nova_help(self, question: str, capability_catalog: NovaCatalog) -> NovaHelpPlan:
        cleaned_question = _nova_help_input(question)
        payload = {
            "question": cleaned_question,
            **_nova_catalog_payload(capability_catalog),
        }
        client = self.client.with_options(max_retries=0)
        async with asyncio.timeout(NOVA_HELP_TIMEOUT_SECONDS):
            response = await client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": prompts.NOVA_HELP_SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                text_format=NovaHelpPlan,
                timeout=NOVA_HELP_TIMEOUT_SECONDS,
            )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("The model returned no structured output")
        return NovaHelpPlan.model_validate(parsed)

    async def companion_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        companion_context: NovaCompanionContextProjection,
    ) -> NovaCompanionResponse:
        clean = _nova_companion_input(text)
        if not isinstance(companion_context, NovaCompanionContextProjection):
            raise ValueError("invalid Nova companion context")
        payload = {
            "message": clean,
            "temporal_context": _nova_companion_temporal_context(temporal_context),
            "companion_context": companion_context.provider_payload(),
        }
        client = self.client.with_options(max_retries=0)
        async with asyncio.timeout(NOVA_COMPANION_TIMEOUT_SECONDS):
            response = await client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": f"{prompts.NOVA_COMPANION_SYSTEM}\nСтиль ответа: {self.tone}.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                text_format=NovaCompanionProviderResponse,
                timeout=NOVA_COMPANION_TIMEOUT_SECONDS,
            )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("The model returned no structured output")
        return _validated_companion_response(clean, parsed)

    async def make_today_plan(self, context: dict[str, object]) -> TodayPlan:
        return await self._parse(
            TodayPlan,
            prompts.TODAY_SYSTEM,
            repr(context),
            max_retries=0,
        )

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
        *,
        confirmed_memory: NovaMemoryProjection | None = None,
    ) -> AssistantAnswer:
        payload = {
            "message": text,
            "temporal_context": temporal_context,
            "conversation_context": conversation_context or {},
        }
        if confirmed_memory is not None and confirmed_memory.records:
            payload["confirmed_memory"] = confirmed_memory.provider_payload()
            client = self.client.with_options(max_retries=0)
            async with asyncio.timeout(NOVA_MEMORY_ANSWER_TIMEOUT_SECONDS):
                response = await client.responses.parse(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": f"{prompts.ANSWER_SYSTEM}\nСтиль ответа: {self.tone}.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ],
                    text_format=AssistantAnswer,
                    timeout=NOVA_MEMORY_ANSWER_TIMEOUT_SECONDS,
                )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("The model returned no structured output")
            return AssistantAnswer.model_validate(parsed)
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
