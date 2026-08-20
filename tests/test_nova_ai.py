import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import future_self.ai as ai_module
from future_self import prompts
from future_self.ai import (
    NOVA_COMPANION_MAX_INPUT_CHARS,
    NOVA_COMPANION_TIMEOUT_SECONDS,
    NOVA_HELP_MAX_INPUT_CHARS,
    NOVA_HELP_TIMEOUT_SECONDS,
    OpenAICompatibleAIService,
)
from future_self.config import Settings
from future_self.nova_companion import build_nova_companion_context_projection
from future_self.schemas import (
    NOVA_HELP_MAX_PAYLOAD_BYTES,
    NovaCompanionProviderCapture,
    NovaCompanionProviderResponse,
    NovaHelpPlan,
)


def help_plan(**overrides: object) -> NovaHelpPlan:
    payload: dict[str, object] = {
        "response": "Открой раздел задач и следуй подсказкам.",
        "steps": ["Сформулируй задачу", "Укажи срок"],
        "action_id": "tasks:create",
        "kind": "guide",
    }
    payload.update(overrides)
    return NovaHelpPlan(**payload)


def capability_catalog() -> SimpleNamespace:
    capability = SimpleNamespace(
        id="tasks:create",
        label="Создать задачу",
        description="Открывает существующий flow создания задачи",
        handler_name="PRIVATE_HANDLER_MUST_NOT_REACH_PROVIDER",
        callback_data="PRIVATE_CALLBACK_MUST_NOT_REACH_PROVIDER",
    )
    return SimpleNamespace(
        capabilities=(capability,),
        enabled_features=frozenset({"tasks", "task_reminders"}),
        profile="PRIVATE_PROFILE_MUST_NOT_REACH_PROVIDER",
        telegram_id=530129470,
        database_content="PRIVATE_DATABASE_CONTENT_MUST_NOT_REACH_PROVIDER",
    )


class FakeResponses:
    def __init__(
        self,
        *,
        output_parsed: object | None = None,
        error: BaseException | None = None,
        wait_forever: bool = False,
    ):
        self.output_parsed = help_plan() if output_parsed is None else output_parsed
        self.error = error
        self.wait_forever = wait_forever
        self.parse_calls: list[dict[str, object]] = []

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if self.wait_forever:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.output_parsed)


class ForbiddenGlobalResponses:
    def __init__(self):
        self.parse_calls: list[dict[str, object]] = []

    async def parse(self, **kwargs):  # pragma: no cover - failure identifies client leakage
        self.parse_calls.append(kwargs)
        raise AssertionError("Nova must use the isolated no-retry client")


class FakeClient:
    def __init__(self, option_responses: FakeResponses):
        self.responses = ForbiddenGlobalResponses()
        self.option_responses = option_responses
        self.with_options_calls: list[dict[str, object]] = []
        self.option_clients: list[SimpleNamespace] = []

    def with_options(self, **kwargs):
        self.with_options_calls.append(kwargs)
        configured = SimpleNamespace(responses=self.option_responses)
        self.option_clients.append(configured)
        return configured


def service_with_fake(
    *,
    output_parsed: object | None = None,
    error: BaseException | None = None,
    wait_forever: bool = False,
) -> tuple[OpenAICompatibleAIService, FakeClient, FakeResponses]:
    responses = FakeResponses(
        output_parsed=output_parsed,
        error=error,
        wait_forever=wait_forever,
    )
    client = FakeClient(responses)
    return OpenAICompatibleAIService(client, "nova-test-model"), client, responses


def test_nova_settings_are_fail_closed_and_typed(monkeypatch):
    monkeypatch.delenv("ENABLE_NOVA_AI", raising=False)
    monkeypatch.delenv("NOVA_AI_ADMIN_ONLY", raising=False)
    configured = Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
    )

    assert configured.enable_nova_ai is False
    assert configured.nova_ai_admin_only is True

    enabled = Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        enable_nova_ai="true",
        nova_ai_admin_only="false",
    )
    assert enabled.enable_nova_ai is True
    assert enabled.nova_ai_admin_only is False


def test_nova_schema_is_strict_stripped_and_bounded():
    plan = NovaHelpPlan(
        response="  Короткий ответ  ",
        steps=["  Первый шаг  "],
        action_id="  tasks:create  ",
        kind="guide",
    )
    assert plan.model_dump() == {
        "response": "Короткий ответ",
        "steps": ["Первый шаг"],
        "action_id": "tasks:create",
        "kind": "guide",
    }

    with pytest.raises(ValidationError):
        NovaHelpPlan(**plan.model_dump(), unexpected="forbidden")

    for kind in ("guide", "clarify", "unsupported"):
        assert help_plan(kind=kind).kind == kind
    with pytest.raises(ValidationError):
        help_plan(kind="chat")


@pytest.mark.parametrize(
    ("overrides", "valid"),
    [
        ({"response": "x"}, True),
        ({"response": "x" * 800}, True),
        ({"response": " "}, False),
        ({"response": "x" * 801}, False),
        ({"steps": []}, True),
        ({"steps": ["a", "b", "c"]}, True),
        ({"steps": ["a", "b", "c", "d"]}, False),
        ({"steps": ["x" * 200]}, True),
        ({"steps": ["x" * 201]}, False),
        ({"steps": ["  "]}, False),
        ({"action_id": None}, True),
        ({"action_id": "x" * 100}, True),
        ({"action_id": "x" * 101}, False),
        ({"action_id": "  "}, False),
    ],
)
def test_nova_schema_boundaries(overrides, valid):
    if valid:
        help_plan(**overrides)
    else:
        with pytest.raises(ValidationError):
            help_plan(**overrides)


def test_nova_schema_guarantees_serialized_payload_at_most_four_kibibytes():
    largest_ascii = help_plan(
        response="x" * 800,
        steps=["x" * 200, "x" * 200, "x" * 200],
        action_id="x" * 100,
    )
    assert len(largest_ascii.model_dump_json().encode("utf-8")) <= (NOVA_HELP_MAX_PAYLOAD_BYTES)

    with pytest.raises(ValidationError, match="4 KiB"):
        help_plan(
            response="😀" * 800,
            steps=["😀" * 200, "😀" * 200, "😀" * 200],
            action_id="x" * 100,
        )


async def test_nova_help_uses_one_isolated_parse_with_minimal_private_payload():
    service, client, responses = service_with_fake()
    catalog = capability_catalog()
    question = "RAW_NOVA_QUESTION_SENTINEL"

    result = await service.nova_help(f" \n{question}\t ", catalog)

    assert result == help_plan()
    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(client.option_clients) == 1
    assert not hasattr(client, "max_retries")
    assert client.responses.parse_calls == []
    assert len(responses.parse_calls) == 1
    call = responses.parse_calls[0]
    assert call["model"] == "nova-test-model"
    assert call["text_format"] is NovaHelpPlan
    assert call["timeout"] == NOVA_HELP_TIMEOUT_SECONDS == 30.0
    assert call["input"][0] == {"role": "system", "content": prompts.NOVA_HELP_SYSTEM}
    provider_payload = json.loads(call["input"][1]["content"])
    assert provider_payload == {
        "question": question,
        "capabilities": [
            {
                "id": "tasks:create",
                "label": "Создать задачу",
                "description": "Открывает существующий flow создания задачи",
            }
        ],
        "enabled_features": ["task_reminders", "tasks"],
    }
    serialized_call = json.dumps(call, ensure_ascii=False, default=str)
    for forbidden in (
        "PRIVATE_HANDLER_MUST_NOT_REACH_PROVIDER",
        "PRIVATE_CALLBACK_MUST_NOT_REACH_PROVIDER",
        "PRIVATE_PROFILE_MUST_NOT_REACH_PROVIDER",
        "PRIVATE_DATABASE_CONTENT_MUST_NOT_REACH_PROVIDER",
        "530129470",
    ):
        assert forbidden not in serialized_call


@pytest.mark.parametrize("length", [1, NOVA_HELP_MAX_INPUT_CHARS])
async def test_nova_help_accepts_input_boundaries(length):
    service, client, responses = service_with_fake()
    question = "я" * length

    await service.nova_help(question, capability_catalog())

    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1
    payload = json.loads(responses.parse_calls[0]["input"][1]["content"])
    assert payload["question"] == question


@pytest.mark.parametrize(
    "invalid_input",
    [None, 123, "", " \n\t ", "x" * (NOVA_HELP_MAX_INPUT_CHARS + 1)],
)
async def test_nova_help_rejects_invalid_input_before_sdk(invalid_input):
    service, client, responses = service_with_fake()

    with pytest.raises(ValueError):
        await service.nova_help(invalid_input, capability_catalog())

    assert client.with_options_calls == []
    assert responses.parse_calls == []


async def test_nova_help_revalidates_invalid_provider_schema_without_retry():
    invalid_output = {
        "response": "Ответ",
        "steps": [],
        "action_id": None,
        "kind": "forged",
    }
    service, client, responses = service_with_fake(output_parsed=invalid_output)

    with pytest.raises(ValidationError):
        await service.nova_help("Где нужный раздел?", capability_catalog())

    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1


async def test_nova_help_missing_structured_output_fails_without_retry():
    service, client, responses = service_with_fake(output_parsed=SimpleNamespace())
    responses.output_parsed = None

    with pytest.raises(ValueError, match="no structured output"):
        await service.nova_help("Где нужный раздел?", capability_catalog())

    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1


async def test_nova_help_provider_exception_propagates_without_retry():
    provider_error = RuntimeError("PRIVATE_PROVIDER_ERROR_BODY")
    service, client, responses = service_with_fake(error=provider_error)

    with pytest.raises(RuntimeError) as caught:
        await service.nova_help("Где нужный раздел?", capability_catalog())

    assert caught.value is provider_error
    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1


async def test_nova_help_enforces_wall_timeout(monkeypatch):
    service, client, responses = service_with_fake(wait_forever=True)
    monkeypatch.setattr(ai_module, "NOVA_HELP_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(TimeoutError):
        await service.nova_help("Где нужный раздел?", capability_catalog())

    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1
    assert responses.parse_calls[0]["timeout"] == 0.01


async def test_nova_help_propagates_cancelled_error():
    cancelled = asyncio.CancelledError()
    service, client, responses = service_with_fake(error=cancelled)

    with pytest.raises(asyncio.CancelledError):
        await service.nova_help("Где нужный раздел?", capability_catalog())

    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1


async def test_nova_help_does_not_route_through_other_ai_methods():
    service, _client, responses = service_with_fake()
    service.parse_thought = AsyncMock(side_effect=AssertionError("parse_thought called"))
    service.route_message = AsyncMock(side_effect=AssertionError("route_message called"))
    service.answer_message = AsyncMock(side_effect=AssertionError("answer_message called"))
    service.guest_thought_breakdown = AsyncMock(
        side_effect=AssertionError("guest_thought_breakdown called")
    )
    service.guest_first_step = AsyncMock(side_effect=AssertionError("guest_first_step called"))

    await service.nova_help("Как открыть задачи?", capability_catalog())

    service.parse_thought.assert_not_awaited()
    service.route_message.assert_not_awaited()
    service.answer_message.assert_not_awaited()
    service.guest_thought_breakdown.assert_not_awaited()
    service.guest_first_step.assert_not_awaited()
    assert len(responses.parse_calls) == 1


async def test_nova_help_does_not_log_raw_input_output_or_provider_error(caplog):
    question = "RAW_NOVA_INPUT_MUST_NOT_BE_LOGGED"
    output = help_plan(response="RAW_NOVA_OUTPUT_MUST_NOT_BE_LOGGED")
    service, _client, responses = service_with_fake(output_parsed=output)

    await service.nova_help(question, capability_catalog())

    assert len(responses.parse_calls) == 1
    assert question not in caplog.text
    assert output.response not in caplog.text


def companion_projection():
    return build_nova_companion_context_projection(
        profile=SimpleNamespace(
            summary="PRIVATE_PROFILE_CONTEXT",
            values=["PRIVATE_PROFILE_VALUE"],
            desired_identity=[],
            constraints=[],
            motivation_style=None,
        ),
        conversation_context={
            "current_topic": "PRIVATE_TOPIC",
            "recent_messages": [{"role": "user", "content": "PRIVATE_RECENT_MESSAGE", "id": 999}],
            "active_draft": {"id": "PRIVATE_DRAFT"},
        },
    )


def companion_response(**overrides: object) -> NovaCompanionProviderResponse:
    payload: dict[str, object] = {
        "answer": "Это звучит как конкретная идея, которую можно спокойно обдумать.",
        "capture": NovaCompanionProviderCapture(
            kind="idea",
            title="клуб чтения",
            next_step="собрать клуб чтения",
            evidence="Хочу собрать клуб чтения",
        ),
    }
    payload.update(overrides)
    return NovaCompanionProviderResponse(**payload)


async def test_nova_companion_uses_one_no_retry_call_and_minimal_bounded_payload():
    output = companion_response()
    service, client, responses = service_with_fake(output_parsed=output)

    result = await service.companion_message(
        "  Хочу собрать клуб чтения  ",
        {
            "timezone": "Europe/Moscow",
            "local_datetime": "2026-08-18T15:00:00+03:00",
            "today_date": "2026-08-18",
        },
        companion_projection(),
    )

    assert result.answer == output.answer
    assert result.capture is not None
    assert result.capture.kind == "idea"
    assert result.capture.title == "клуб чтения"
    assert result.capture.next_step == "собрать клуб чтения"
    assert not hasattr(result.capture, "evidence")
    assert client.with_options_calls == [{"max_retries": 0}]
    assert client.responses.parse_calls == []
    assert len(responses.parse_calls) == 1
    call = responses.parse_calls[0]
    assert call["model"] == "nova-test-model"
    assert call["text_format"] is NovaCompanionProviderResponse
    assert call["timeout"] == NOVA_COMPANION_TIMEOUT_SECONDS == 30.0
    assert call["input"][0] == {
        "role": "system",
        "content": f"{prompts.NOVA_COMPANION_SYSTEM}\nСтиль ответа: спокойный и конкретный.",
    }
    provider_payload = json.loads(call["input"][1]["content"])
    assert provider_payload["message"] == "Хочу собрать клуб чтения"
    assert provider_payload["temporal_context"] == {
        "timezone": "Europe/Moscow",
        "local_datetime": "2026-08-18T15:00:00+03:00",
        "today_date": "2026-08-18",
    }
    assert provider_payload["companion_context"] == {
        "profile": {
            "summary": "PRIVATE_PROFILE_CONTEXT",
            "values": ["PRIVATE_PROFILE_VALUE"],
        },
        "recent_conversation": {
            "current_topic": "PRIVATE_TOPIC",
            "recent_messages": [{"role": "user", "content": "PRIVATE_RECENT_MESSAGE"}],
        },
    }
    assert "PRIVATE_DRAFT" not in call["input"][1]["content"]
    assert "999" not in call["input"][1]["content"]
    assert "PRIVATE_PROFILE_CONTEXT" not in call["input"][0]["content"]


@pytest.mark.parametrize(
    "text",
    [
        "Я постоянно забываю о главном, из-за каждодневной суеты",
        "Это стоит сохранить?",
        "Да не, я хотел просто пообщаться",
        "Не сохраняй это",
        "Не надо ничего записывать",
        "Сегодня идёт дождь",
        "Я обычно пью чай утром",
        "Просто думаю о новом проекте",
    ],
)
async def test_nova_companion_fail_closed_suppresses_capture_for_non_capture_text(text):
    output = NovaCompanionProviderResponse(
        answer="Я рядом и готова поговорить.",
        capture=NovaCompanionProviderCapture(
            kind="note",
            title=text[: min(20, len(text))],
            evidence=text,
        ),
    )
    service, client, responses = service_with_fake(output_parsed=output)

    result = await service.companion_message(
        text,
        {"timezone": "Europe/Moscow"},
        companion_projection(),
    )

    assert result.answer == output.answer
    assert result.capture is None
    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1


@pytest.mark.parametrize(
    "capture",
    [
        NovaCompanionProviderCapture(
            kind="idea",
            title="чужой заголовок",
            evidence="Хочу собрать клуб чтения",
        ),
        NovaCompanionProviderCapture(
            kind="idea",
            title="клуб чтения",
            evidence="Текст отсутствует в сообщении",
        ),
        NovaCompanionProviderCapture(
            kind="idea",
            title="клуб чтения",
            next_step="придуманный следующий шаг",
            evidence="Хочу собрать клуб чтения",
        ),
    ],
)
async def test_nova_companion_drops_ungrounded_capture_but_keeps_human_answer(capture):
    output = companion_response(capture=capture)
    service, client, responses = service_with_fake(output_parsed=output)

    result = await service.companion_message(
        "Хочу собрать клуб чтения",
        {"timezone": "Europe/Moscow"},
        companion_projection(),
    )

    assert result.answer == output.answer
    assert result.capture is None
    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1


@pytest.mark.parametrize(
    ("text", "temporal_context"),
    [
        (None, {"timezone": "Europe/Moscow"}),
        ("", {"timezone": "Europe/Moscow"}),
        ("x" * (NOVA_COMPANION_MAX_INPUT_CHARS + 1), {"timezone": "Europe/Moscow"}),
        ("Привет", {"telegram_id": "123"}),
        ("Привет", {"timezone": ""}),
        ("Привет", {"timezone": 123}),
    ],
)
async def test_nova_companion_rejects_invalid_input_before_sdk(text, temporal_context):
    service, client, responses = service_with_fake(output_parsed=companion_response(capture=None))

    with pytest.raises(ValueError):
        await service.companion_message(text, temporal_context, companion_projection())

    assert client.with_options_calls == []
    assert responses.parse_calls == []


async def test_nova_companion_missing_output_and_provider_error_do_not_retry():
    missing_service, missing_client, missing_responses = service_with_fake(
        output_parsed=SimpleNamespace()
    )
    missing_responses.output_parsed = None
    with pytest.raises(ValueError, match="no structured output"):
        await missing_service.companion_message(
            "Поговорим",
            {"timezone": "Europe/Moscow"},
            companion_projection(),
        )
    assert missing_client.with_options_calls == [{"max_retries": 0}]
    assert len(missing_responses.parse_calls) == 1

    provider_error = RuntimeError("PRIVATE_PROVIDER_ERROR")
    error_service, error_client, error_responses = service_with_fake(error=provider_error)
    with pytest.raises(RuntimeError) as caught:
        await error_service.companion_message(
            "Поговорим",
            {"timezone": "Europe/Moscow"},
            companion_projection(),
        )
    assert caught.value is provider_error
    assert error_client.with_options_calls == [{"max_retries": 0}]
    assert len(error_responses.parse_calls) == 1


async def test_nova_companion_propagates_cancellation_and_enforces_timeout(monkeypatch):
    cancelled = asyncio.CancelledError()
    cancel_service, cancel_client, cancel_responses = service_with_fake(error=cancelled)
    with pytest.raises(asyncio.CancelledError):
        await cancel_service.companion_message(
            "Поговорим",
            {"timezone": "Europe/Moscow"},
            companion_projection(),
        )
    assert cancel_client.with_options_calls == [{"max_retries": 0}]
    assert len(cancel_responses.parse_calls) == 1

    timeout_service, timeout_client, timeout_responses = service_with_fake(wait_forever=True)
    monkeypatch.setattr(ai_module, "NOVA_COMPANION_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError):
        await timeout_service.companion_message(
            "Поговорим",
            {"timezone": "Europe/Moscow"},
            companion_projection(),
        )
    assert timeout_client.with_options_calls == [{"max_retries": 0}]
    assert len(timeout_responses.parse_calls) == 1
    assert timeout_responses.parse_calls[0]["timeout"] == 0.01


async def test_nova_companion_does_not_route_through_mutating_or_legacy_ai_methods():
    service, _client, responses = service_with_fake(output_parsed=companion_response(capture=None))
    for method_name in (
        "parse_thought",
        "route_message",
        "answer_message",
        "extract_weekly_review",
        "propose_goals",
        "propose_routines",
        "make_today_plan",
    ):
        setattr(service, method_name, AsyncMock(side_effect=AssertionError(method_name)))

    await service.companion_message(
        "Мне хочется поговорить",
        {"timezone": "Europe/Moscow"},
        companion_projection(),
    )

    for method_name in (
        "parse_thought",
        "route_message",
        "answer_message",
        "extract_weekly_review",
        "propose_goals",
        "propose_routines",
        "make_today_plan",
    ):
        getattr(service, method_name).assert_not_awaited()
    assert len(responses.parse_calls) == 1
