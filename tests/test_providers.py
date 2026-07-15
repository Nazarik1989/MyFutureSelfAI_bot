import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, BadRequestError

from future_self.ai import OpenAICompatibleAIService, ProviderHealthCheck, create_ai_service
from future_self.config import LegacyConfigurationWarning, Settings, resolve_env_file
from future_self.doctor import DoctorReport, duplicate_env_keys, run_provider_check
from future_self.transcription import (
    DisabledTranscriptionService,
    create_transcription_service,
)


def settings(**overrides) -> Settings:
    values = {
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
