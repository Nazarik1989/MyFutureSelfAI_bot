from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from future_self import prompts
from future_self.ai import GUEST_DEMO_MAX_INPUT_CHARS, OpenAICompatibleAIService
from future_self.schemas import (
    GuestFirstStep,
    GuestThoughtBreakdown,
    TimezoneResolution,
)

_AUTO_OUTPUT = object()


def thought_payload() -> dict[str, object]:
    return {
        "category": "idea",
        "title": "Короткий заголовок",
        "essence": "Краткая суть",
        "next_step": "Записать один вариант",
    }


def first_step_payload() -> dict[str, object]:
    return {
        "focus": "Главный фокус",
        "first_step": "Открыть заметки",
        "actions": ["Записать одну строку"],
    }


class FakeResponses:
    def __init__(self, output_parsed: object = _AUTO_OUTPUT):
        self.output_parsed = output_parsed
        self.parse_calls: list[dict[str, object]] = []

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        output = self.output_parsed
        if output is _AUTO_OUTPUT:
            schema = kwargs["text_format"]
            if schema is GuestThoughtBreakdown:
                output = GuestThoughtBreakdown(**thought_payload())
            elif schema is GuestFirstStep:
                output = GuestFirstStep(**first_step_payload())
            elif schema is TimezoneResolution:
                output = TimezoneResolution(
                    timezone="Europe/Moscow",
                    city="Москва",
                    country="Россия",
                )
            else:  # pragma: no cover - a failing contract should identify the schema
                raise AssertionError(f"unexpected schema: {schema!r}")
        return SimpleNamespace(output_parsed=output)


class FakeClient:
    def __init__(self, responses: FakeResponses):
        self.responses = responses
        self.with_options_calls: list[dict[str, object]] = []
        self.option_clients: list[SimpleNamespace] = []

    def with_options(self, **kwargs):
        self.with_options_calls.append(kwargs)
        configured = SimpleNamespace(responses=self.responses)
        self.option_clients.append(configured)
        return configured


def service_with_fake(
    output_parsed: object = _AUTO_OUTPUT,
) -> tuple[OpenAICompatibleAIService, FakeClient, FakeResponses]:
    responses = FakeResponses(output_parsed)
    client = FakeClient(responses)
    return OpenAICompatibleAIService(client, "guest-test-model"), client, responses


def test_valid_guest_thought_breakdown_strips_strings_and_forbids_extra_fields():
    result = GuestThoughtBreakdown(
        category="task",
        title="  Заголовок  ",
        essence="  Суть  ",
        next_step="  Следующий шаг  ",
    )
    assert result.model_dump() == {
        "category": "task",
        "title": "Заголовок",
        "essence": "Суть",
        "next_step": "Следующий шаг",
    }
    with pytest.raises(ValidationError):
        GuestThoughtBreakdown(**thought_payload(), unknown="forbidden")


def test_valid_guest_first_step_strips_strings_and_forbids_extra_fields():
    result = GuestFirstStep(
        focus="  Фокус  ",
        first_step="  Первый шаг  ",
        actions=["  Раз  ", "  Два  "],
    )
    assert result.model_dump() == {
        "focus": "Фокус",
        "first_step": "Первый шаг",
        "actions": ["Раз", "Два"],
    }
    with pytest.raises(ValidationError):
        GuestFirstStep(**first_step_payload(), unknown="forbidden")


@pytest.mark.parametrize("category", ["idea", "task", "desire", "note"])
def test_guest_thought_breakdown_accepts_only_supported_categories(category):
    payload = thought_payload()
    payload["category"] = category
    assert GuestThoughtBreakdown(**payload).category == category


def test_guest_thought_breakdown_rejects_unknown_category():
    payload = thought_payload()
    payload["category"] = "reminder"
    with pytest.raises(ValidationError):
        GuestThoughtBreakdown(**payload)


@pytest.mark.parametrize(
    ("schema", "payload_factory", "field", "maximum"),
    [
        (GuestThoughtBreakdown, thought_payload, "title", 120),
        (GuestThoughtBreakdown, thought_payload, "essence", 500),
        (GuestThoughtBreakdown, thought_payload, "next_step", 300),
        (GuestFirstStep, first_step_payload, "focus", 300),
        (GuestFirstStep, first_step_payload, "first_step", 300),
    ],
)
def test_guest_schema_string_field_boundaries(schema, payload_factory, field, maximum):
    payload = payload_factory()
    payload[field] = "x"
    assert getattr(schema(**payload), field) == "x"

    payload[field] = "x" * maximum
    assert len(getattr(schema(**payload), field)) == maximum

    payload[field] = " "
    with pytest.raises(ValidationError):
        schema(**payload)

    payload[field] = "x" * (maximum + 1)
    with pytest.raises(ValidationError):
        schema(**payload)


def test_guest_first_step_actions_have_list_and_item_boundaries():
    payload = first_step_payload()
    payload["actions"] = []
    assert GuestFirstStep(**payload).actions == []

    payload["actions"] = ["a", "b", "c"]
    assert GuestFirstStep(**payload).actions == ["a", "b", "c"]

    payload["actions"] = ["a", "b", "c", "d"]
    with pytest.raises(ValidationError):
        GuestFirstStep(**payload)

    payload["actions"] = ["x" * 200]
    assert len(GuestFirstStep(**payload).actions[0]) == 200

    for invalid_action in ("", "   ", "x" * 201):
        payload["actions"] = [invalid_action]
        with pytest.raises(ValidationError):
            GuestFirstStep(**payload)


def test_maximum_guest_payloads_fit_existing_session_limit():
    thought = GuestThoughtBreakdown(
        category="desire",
        title="😀" * 120,
        essence="😀" * 500,
        next_step="😀" * 300,
    )
    first_step = GuestFirstStep(
        focus="😀" * 300,
        first_step="😀" * 300,
        actions=["😀" * 200, "😀" * 200, "😀" * 200],
    )
    assert len(thought.model_dump_json().encode("utf-8")) <= 8 * 1024
    assert len(first_step.model_dump_json().encode("utf-8")) <= 8 * 1024


@pytest.mark.parametrize(
    ("method_name", "expected_schema", "expected_prompt"),
    [
        (
            "guest_thought_breakdown",
            GuestThoughtBreakdown,
            prompts.GUEST_THOUGHT_SYSTEM,
        ),
        ("guest_first_step", GuestFirstStep, prompts.GUEST_FIRST_STEP_SYSTEM),
    ],
)
async def test_guest_method_uses_one_parse_correct_contract_and_no_sdk_retry(
    method_name,
    expected_schema,
    expected_prompt,
):
    service, client, responses = service_with_fake()
    sentinel = "RAW_GUEST_AI_SENTINEL"

    result = await getattr(service, method_name)(f" \n{sentinel}\t ")

    assert isinstance(result, expected_schema)
    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(client.option_clients) == 1
    assert not hasattr(client, "max_retries")
    assert len(responses.parse_calls) == 1
    call = responses.parse_calls[0]
    assert call["text_format"] is expected_schema
    assert call["model"] == "guest-test-model"
    assert call["input"] == [
        {
            "role": "system",
            "content": f"{expected_prompt}\nСтиль ответа: спокойный и конкретный.",
        },
        {"role": "user", "content": sentinel},
    ]


@pytest.mark.parametrize("method_name", ["guest_thought_breakdown", "guest_first_step"])
@pytest.mark.parametrize("length", [1, GUEST_DEMO_MAX_INPUT_CHARS])
async def test_guest_method_accepts_input_boundaries(method_name, length):
    service, client, responses = service_with_fake()
    text = "я" * length

    await getattr(service, method_name)(text)

    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1
    assert responses.parse_calls[0]["input"][1] == {"role": "user", "content": text}


@pytest.mark.parametrize("method_name", ["guest_thought_breakdown", "guest_first_step"])
@pytest.mark.parametrize(
    "invalid_input",
    [None, 123, "", " \n\t ", "x" * (GUEST_DEMO_MAX_INPUT_CHARS + 1)],
)
async def test_guest_method_rejects_invalid_input_before_sdk(method_name, invalid_input):
    service, client, responses = service_with_fake()

    with pytest.raises(ValueError):
        await getattr(service, method_name)(invalid_input)

    assert client.with_options_calls == []
    assert responses.parse_calls == []


@pytest.mark.parametrize("method_name", ["guest_thought_breakdown", "guest_first_step"])
async def test_missing_structured_output_fails_without_retry(method_name):
    service, client, responses = service_with_fake(None)

    with pytest.raises(ValueError, match="no structured output"):
        await getattr(service, method_name)("короткий текст")

    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(responses.parse_calls) == 1


async def test_guest_methods_do_not_route_through_full_feature_ai_methods():
    service, _client, responses = service_with_fake()
    service.parse_thought = AsyncMock(side_effect=AssertionError("parse_thought called"))
    service.route_message = AsyncMock(side_effect=AssertionError("route_message called"))
    service.answer_message = AsyncMock(side_effect=AssertionError("answer_message called"))

    await service.guest_thought_breakdown("мысль")
    await service.guest_first_step("намерение")

    service.parse_thought.assert_not_awaited()
    service.route_message.assert_not_awaited()
    service.answer_message.assert_not_awaited()
    assert len(responses.parse_calls) == 2


async def test_guest_raw_input_is_not_logged(caplog):
    service, _client, responses = service_with_fake()
    sentinel = "RAW_GUEST_AI_INPUT_MUST_NOT_BE_LOGGED"

    await service.guest_thought_breakdown(sentinel)

    assert sentinel not in caplog.text
    assert len(responses.parse_calls) == 1


async def test_non_guest_parse_keeps_existing_client_retry_behavior():
    service, client, responses = service_with_fake()

    result = await service.resolve_timezone("Москва")

    assert result.timezone == "Europe/Moscow"
    assert client.with_options_calls == []
    assert len(responses.parse_calls) == 1
    assert responses.parse_calls[0]["text_format"] is TimezoneResolution
