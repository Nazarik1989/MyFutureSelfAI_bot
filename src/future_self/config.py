import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LegacyConfigurationWarning(UserWarning):
    pass


def resolve_env_file(
    *,
    cwd: Path | None = None,
    project_root: Path | None = None,
    explicit: str | None = None,
) -> Path:
    override = explicit or os.getenv("FUTURE_SELF_ENV_FILE")
    if override:
        return Path(override).expanduser().resolve()
    working_directory = (cwd or Path.cwd()).resolve()
    source_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    source_env = source_root / ".env"
    cwd_env = working_directory / ".env"
    if source_env.exists():
        return source_env
    if cwd_env.exists():
        return cwd_env
    return source_env


ENV_FILE_PATH = resolve_env_file()


class Settings(BaseSettings):
    telegram_bot_token: str = Field(min_length=1, repr=False)

    ai_provider: Literal["openrouter", "openai"] | None = None
    ai_api_key: str | None = Field(default=None, repr=False)
    ai_base_url: str | None = None
    ai_model: str | None = None
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "MyFutureSelfAI"

    transcription_provider: Literal["openai", "local", "disabled"] = "disabled"
    transcription_api_key: str | None = Field(default=None, repr=False)
    transcription_base_url: str | None = "https://api.openai.com/v1"
    transcription_model: str = "gpt-4o-mini-transcribe"

    # Temporary compatibility only. New deployments must use AI_* variables.
    openai_api_key: str | None = Field(default=None, exclude=True, repr=False)
    openai_model: str | None = Field(default=None, exclude=True, repr=False)

    database_url: str = "sqlite+aiosqlite:///./future_self.db"
    default_timezone: str = "Europe/Moscow"
    bot_persona_name: str = "Моя будущая версия"
    bot_tone: str = "спокойный, конкретный, без осуждения"
    morning_hour: int = Field(default=8, ge=0, le=23)
    evening_hour: int = Field(default=21, ge=0, le=23)
    weekly_review_weekday: int = Field(default=6, ge=0, le=6)
    max_audio_seconds: int = Field(default=180, ge=10, le=600)
    max_audio_bytes: int = Field(default=20_000_000, ge=100_000)
    intent_confidence_threshold: float = Field(default=0.70, ge=0, le=1)
    inbox_draft_ttl_minutes: int = Field(default=60, ge=5, le=1440)
    draft_focus_ttl_minutes: int = Field(default=15, ge=1, le=1440)
    system_action_ttl_minutes: int = Field(default=10, ge=1, le=60)
    conversation_context_messages: int = Field(default=12, ge=10, le=20)
    conversation_context_ttl_hours: int = Field(default=24, ge=1, le=168)
    task_date_event_hour: int = Field(default=9, ge=0, le=23)
    task_reminder_lead_minutes: int = Field(default=30, ge=0, le=10080)
    task_reminder_poll_seconds: int = Field(default=15, ge=5, le=300)
    task_reminder_lease_seconds: int = Field(default=120, ge=30, le=3600)
    enable_task_reminders: bool = True
    enable_voice: bool = True
    enable_weekly_review: bool = True
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator(
        "telegram_bot_token",
        "ai_api_key",
        "transcription_api_key",
        "openai_api_key",
        mode="before",
    )
    @classmethod
    def strip_optional_secrets(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("ai_base_url", "transcription_base_url", "openrouter_site_url", mode="before")
    @classmethod
    def strip_urls(cls, value: object) -> object:
        return value.strip().rstrip("/") if isinstance(value, str) and value.strip() else None

    @field_validator("database_url")
    @classmethod
    def async_database_driver(cls, value: str) -> str:
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value

    @model_validator(mode="after")
    def resolve_providers_and_legacy_values(self) -> "Settings":
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        legacy_key_fallback = not self.ai_api_key and bool(self.openai_api_key)
        if legacy_key_fallback:
            self.ai_api_key = self.openai_api_key
            warnings.warn(
                "OPENAI_API_KEY is deprecated for text AI; use AI_API_KEY.",
                LegacyConfigurationWarning,
                stacklevel=2,
            )
        if not self.ai_api_key:
            raise ValueError("AI_API_KEY is required")
        if not self.ai_provider:
            self.ai_provider = "openai" if legacy_key_fallback else "openrouter"

        if not self.ai_model and self.openai_model:
            self.ai_model = self.openai_model
            warnings.warn(
                "OPENAI_MODEL is deprecated; use AI_MODEL.",
                LegacyConfigurationWarning,
                stacklevel=2,
            )
        if not self.ai_model:
            self.ai_model = (
                "openai/gpt-5.4-mini" if self.ai_provider == "openrouter" else "gpt-4.1-mini"
            )
        if not self.ai_base_url:
            self.ai_base_url = (
                "https://openrouter.ai/api/v1"
                if self.ai_provider == "openrouter"
                else "https://api.openai.com/v1"
            )

        if self.transcription_provider == "openai" and not self.transcription_api_key:
            raise ValueError("TRANSCRIPTION_API_KEY is required when TRANSCRIPTION_PROVIDER=openai")
        if self.transcription_provider == "local" and (
            not self.transcription_base_url
            or self.transcription_base_url == "https://api.openai.com/v1"
        ):
            raise ValueError(
                "TRANSCRIPTION_BASE_URL must point to the local service when provider=local"
            )
        if not self.transcription_base_url:
            self.transcription_base_url = "https://api.openai.com/v1"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """Used by diagnostics and tests after changing environment variables."""
    get_settings.cache_clear()
