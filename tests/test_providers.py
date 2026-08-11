import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, BadRequestError

from future_self.ai import (
    REMINDER_TIMEZONE_MAX_INPUT_CHARS,
    REMINDER_TIMEZONE_TIMEOUT_SECONDS,
    OpenAICompatibleAIService,
    ProviderHealthCheck,
    create_ai_service,
)
from future_self.config import LegacyConfigurationWarning, Settings, resolve_env_file
from future_self.doctor import DoctorReport, duplicate_env_keys, run_provider_check
from future_self.schemas import ReminderTimezoneResolution, TimezoneResolution
from future_self.transcription import (
    DisabledTranscriptionService,
    create_transcription_service,
)


def settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "telegram_bot_token": "123456:TEST",
        "ai_provider": "openrouter",
        "ai_api_key": "router-key",
        "ai_base_url": "https://openrouter.ai/api/v1",
        "ai_model": "openai/gpt-5.4-mini",
        "transcription_provider": "disabled",
    }
    values.update(overrides)
    return Settings(**values)


def test_openrouter_client_receives_base_url_and_optional_headers():
    service = create_ai_service(
        settings(
            openrouter_site_url="https://example.test/app",
            openrouter_app_name="MyFutureSelfAI",
        )
    )
    assert str(service.client.base_url) == "https://openrouter.ai/api/v1/"
    assert service.client.default_headers["HTTP-Referer"] == "https://example.test/app"
    assert service.client.default_headers["X-Title"] == "MyFutureSelfAI"


def test_official_openai_text_client_uses_official_base_url():
    service = create_ai_service(
        settings(
            ai_provider="openai",
            ai_api_key="official-key",
            ai_base_url=None,
            ai_model="gpt-4.1-mini",
        )
    )
    assert str(service.client.base_url) == "https://api.openai.com/v1/"


def test_text_and_transcription_use_different_clients():
    configured = settings(
        transcription_provider="openai",
        transcription_api_key="separate-stt-key",
        transcription_base_url="https://api.openai.com/v1",
    )
    text_service = create_ai_service(configured)
    transcription = create_transcription_service(configured)
    assert text_service.client is not transcription.client
    assert text_service.client.api_key == "router-key"
    assert transcription.client.api_key == "separate-stt-key"
    assert str(text_service.client.base_url) == "https://openrouter.ai/api/v1/"
    assert str(transcription.client.base_url) == "https://api.openai.com/v1/"


def test_disabled_transcription_needs_no_key():
    transcription = create_transcription_service(settings(transcription_api_key=None))
    assert isinstance(transcription, DisabledTranscriptionService)
    assert transcription.enabled is False


def test_legacy_openai_variables_are_supported_with_warnings():
    with pytest.warns(LegacyConfigurationWarning) as caught:
        configured = Settings(
            _env_file=None,
            telegram_bot_token="123456:TEST",
            openai_api_key="legacy-key",
            openai_model="legacy-model",
        )
    assert configured.ai_api_key == "legacy-key"
    assert configured.ai_model == "legacy-model"
    assert configured.ai_provider == "openai"
    assert configured.ai_base_url == "https://api.openai.com/v1"
    assert len(caught) == 2


async def test_doctor_reports_authentication_error_without_secret():
    request = httpx.Request("POST", "https://provider.example/v1/responses")
    response = httpx.Response(401, request=request)

    async def unauthorized():
        raise AuthenticationError("secret-key-was-rejected", response=response, body={})

    report = DoctorReport()
    await run_provider_check(report, "text_llm_network", "openrouter", unauthorized)
    check = report.checks[0]
    assert check.status == "FAIL"
    assert "authentication failed" in check.detail
    assert "secret-key" not in check.detail


async def test_doctor_reports_timeout_without_traceback():
    async def too_slow():
        await asyncio.sleep(1)

    report = DoctorReport()
    await run_provider_check(
        report,
        "text_llm_network",
        "openrouter",
        too_slow,
        timeout_seconds=0.001,
    )
    check = report.checks[0]
    assert check.status == "FAIL"
    assert "timed out" in check.detail


async def test_openrouter_health_check_uses_working_structured_parse_path():
    class OpenRouterResponses:
        def __init__(self):
            self.parse_kwargs = None

        async def parse(self, **kwargs):
            self.parse_kwargs = kwargs
            return SimpleNamespace(output_parsed=ProviderHealthCheck(ok=True))

        async def create(self, **kwargs):
            raise BadRequestError(
                "OpenRouter rejects the old doctor request",
                response=httpx.Response(
                    400,
                    request=httpx.Request("POST", "https://openrouter.ai/api/v1/responses"),
                ),
                body={},
            )

    responses = OpenRouterResponses()
    fake_client = SimpleNamespace(responses=responses)
    service = OpenAICompatibleAIService(fake_client, "openai/gpt-5.4-mini")
    result = await service.health_check()
    assert result.ok is True
    assert responses.parse_kwargs["text_format"] is ProviderHealthCheck
    assert "max_output_tokens" not in responses.parse_kwargs


async def test_timezone_resolution_uses_structured_output_on_text_provider():
    class Responses:
        def __init__(self):
            self.parse_kwargs = None

        async def parse(self, **kwargs):
            self.parse_kwargs = kwargs
            return SimpleNamespace(
                output_parsed=TimezoneResolution(
                    timezone="Europe/Lisbon",
                    city="Лиссабон",
                    country="Португалия",
                    ambiguous=False,
                )
            )

    responses = Responses()
    service = OpenAICompatibleAIService(SimpleNamespace(responses=responses), "openai/gpt-5.4-mini")

    result = await service.resolve_timezone("живу в Лиссабоне")

    assert result.timezone == "Europe/Lisbon"
    assert responses.parse_kwargs["text_format"] is TimezoneResolution
    assert responses.parse_kwargs["input"][1]["content"] == "живу в Лиссабоне"


async def test_reminder_timezone_resolution_uses_only_bounded_fragment_without_retries():
    class Responses:
        def __init__(self):
            self.parse_kwargs = None

        async def parse(self, **kwargs):
            self.parse_kwargs = kwargs
            return SimpleNamespace(
                output_parsed=ReminderTimezoneResolution(
                    status="resolved",
                    timezone="Europe/Moscow",
                    matched_text="по Светогорску",
                    city="Светогорск",
                    country="Россия",
                )
            )

    class Client:
        def __init__(self):
            self.responses = Responses()
            self.option_calls = []

        def with_options(self, **kwargs):
            self.option_calls.append(kwargs)
            return self

    client = Client()
    service = OpenAICompatibleAIService(client, "openai/gpt-5.4-mini")

    result = await service.resolve_reminder_timezone("  по   Светогорску  ")

    assert result.timezone == "Europe/Moscow"
    assert client.option_calls == [{"max_retries": 0}]
    assert client.responses.parse_kwargs["text_format"] is ReminderTimezoneResolution
    assert client.responses.parse_kwargs["input"][1] == {
        "role": "user",
        "content": "по Светогорску",
    }
    assert client.responses.parse_kwargs["timeout"] == REMINDER_TIMEZONE_TIMEOUT_SECONDS
    assert "Напомни" not in repr(client.responses.parse_kwargs["input"])


@pytest.mark.parametrize("value", ["", "   ", "x" * (REMINDER_TIMEZONE_MAX_INPUT_CHARS + 1)])
async def test_reminder_timezone_resolution_rejects_unbounded_or_empty_input(value):
    class Client:
        def with_options(self, **kwargs):
            raise AssertionError("provider must not be prepared for invalid input")

    service = OpenAICompatibleAIService(Client(), "test-model")

    with pytest.raises(ValueError, match="reminder timezone fragment"):
        await service.resolve_reminder_timezone(value)


async def test_reminder_timezone_resolution_has_a_fixed_timeout(monkeypatch):
    class Responses:
        async def parse(self, **kwargs):
            del kwargs
            await asyncio.sleep(1)

    class Client:
        responses = Responses()

        def with_options(self, **kwargs):
            assert kwargs == {"max_retries": 0}
            return self

    monkeypatch.setattr("future_self.ai.REMINDER_TIMEZONE_TIMEOUT_SECONDS", 0.001)
    service = OpenAICompatibleAIService(Client(), "test-model")

    with pytest.raises(TimeoutError):
        await service.resolve_reminder_timezone("по Светогорску")


@pytest.mark.parametrize("status", ["not_mentioned", "insufficient"])
def test_unresolved_reminder_timezone_statuses_cannot_smuggle_resolution_fields(status):
    clean = ReminderTimezoneResolution(status=status)

    assert clean.timezone is None
    assert clean.matched_text is None
    with pytest.raises(ValueError, match="unresolved reminder timezone"):
        ReminderTimezoneResolution(
            status=status,
            timezone="Europe/Moscow",
            matched_text="Москва",
        )


def test_resolved_reminder_timezone_requires_iana_and_evidence_fields():
    with pytest.raises(ValueError, match="requires timezone and matched_text"):
        ReminderTimezoneResolution(status="resolved", timezone="Europe/Moscow")


def test_ambiguous_reminder_timezone_requires_evidence_but_forbids_timezone():
    clean = ReminderTimezoneResolution(status="ambiguous", matched_text="по Сан-Хосе")

    assert clean.matched_text == "по Сан-Хосе"
    assert clean.timezone is None
    with pytest.raises(ValueError, match="requires matched_text only"):
        ReminderTimezoneResolution(status="ambiguous")
    with pytest.raises(ValueError, match="requires matched_text only"):
        ReminderTimezoneResolution(
            status="ambiguous",
            timezone="America/Los_Angeles",
            matched_text="по Сан-Хосе",
        )


async def test_doctor_bad_request_is_safe_and_includes_status():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    response = httpx.Response(400, request=request)

    async def rejected():
        raise BadRequestError("secret request content", response=response, body={})

    report = DoctorReport()
    await run_provider_check(report, "text_llm_network", "openrouter", rejected)
    detail = report.checks[0].detail
    assert "http_status=400" in detail
    assert "error_type=BadRequestError" in detail
    assert "secret request content" not in detail


def test_doctor_detects_duplicate_env_names_without_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ENABLE_VOICE=true\nAI_API_KEY=first-secret\nENABLE_VOICE=false\n",
        encoding="utf-8",
    )
    duplicates = duplicate_env_keys(env_file)
    assert duplicates == ["ENABLE_VOICE"]
    assert "first-secret" not in repr(duplicates)


def test_env_resolution_prefers_project_root_when_started_elsewhere(tmp_path):
    project_root = tmp_path / "project"
    other_cwd = tmp_path / "elsewhere"
    project_root.mkdir()
    other_cwd.mkdir()
    (project_root / ".env").write_text("ENABLE_VOICE=true\n", encoding="utf-8")
    (other_cwd / ".env").write_text("ENABLE_VOICE=false\n", encoding="utf-8")
    assert resolve_env_file(cwd=other_cwd, project_root=project_root) == project_root / ".env"
