import os
import warnings
from functools import lru_cache
from pathlib import Path, PurePosixPath
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
    collection_action_ttl_minutes: int = Field(default=15, ge=1, le=60)
    collection_input_ttl_minutes: int = Field(default=20, ge=1, le=120)
    collection_context_ttl_minutes: int = Field(default=20, ge=1, le=1440)
    sqlite_wal_enabled: bool = True
    sqlite_busy_timeout_ms: int = Field(default=5_000, ge=1_000, le=60_000)
    runtime_min_free_bytes: int = Field(default=1024 * 1024 * 1024, ge=100_000_000)
    runtime_min_free_inodes: int = Field(default=10_000, ge=1_000)

    # PR #23 exposes the Access/Workspace foundation independently. It remains
    # fail-closed unless the deployment explicitly enables its UI and ACL paths.
    enable_workspace_access: bool = False

    # PR #24 implements Hub/Capture/Runner behind independent rollout gates. All
    # later Knowledge/Council stages remain reserved and disabled by default.
    enable_knowledge_hub: bool = False
    enable_knowledge_capture: bool = False
    enable_knowledge_runner: bool = False
    enable_knowledge_retrieval: bool = False
    enable_knowledge_embeddings: bool = False
    enable_knowledge_ocr: bool = False
    enable_knowledge_media: bool = False
    enable_external_vision: bool = False
    enable_council: bool = False
    enable_scheduled_council: bool = False
    enable_knowledge_export: bool = False

    knowledge_asset_root: str = "/data/knowledge"
    knowledge_runner_concurrency: int = Field(default=1, ge=1, le=8)
    knowledge_max_source_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1_000_000, le=100 * 1024 * 1024
    )
    knowledge_daily_ingest_bytes_per_user: int = Field(
        default=100 * 1024 * 1024, ge=1_000_000, le=1024 * 1024 * 1024
    )
    knowledge_storage_quota_bytes_per_user: int = Field(
        default=1024 * 1024 * 1024, ge=10_000_000, le=50 * 1024 * 1024 * 1024
    )
    knowledge_daily_sources_per_user: int = Field(default=20, ge=1, le=1_000)
    knowledge_max_pending_jobs_per_user: int = Field(default=4, ge=1, le=100)
    knowledge_daily_ingest_bytes_per_space: int = Field(
        default=250 * 1024 * 1024, ge=1_000_000, le=5 * 1024 * 1024 * 1024
    )
    knowledge_storage_quota_bytes_per_space: int = Field(
        default=5 * 1024 * 1024 * 1024, ge=10_000_000, le=100 * 1024 * 1024 * 1024
    )
    knowledge_daily_sources_per_space: int = Field(default=100, ge=1, le=10_000)
    knowledge_max_pending_jobs_per_space: int = Field(default=20, ge=1, le=1_000)
    knowledge_capture_ttl_minutes: int = Field(default=30, ge=5, le=24 * 60)
    knowledge_action_ttl_minutes: int = Field(default=15, ge=1, le=60)
    knowledge_staging_ttl_minutes: int = Field(default=60, ge=10, le=24 * 60)
    knowledge_runner_poll_seconds: float = Field(default=2.0, ge=0.25, le=60.0)
    knowledge_runner_lease_seconds: int = Field(default=120, ge=30, le=3600)
    knowledge_runner_heartbeat_seconds: int = Field(default=30, ge=5, le=600)
    knowledge_runner_max_attempts: int = Field(default=3, ge=1, le=10)
    knowledge_extraction_wall_seconds: int = Field(default=30, ge=5, le=300)
    knowledge_extraction_max_pages: int = Field(default=500, ge=1, le=500)
    knowledge_extraction_max_archive_entries: int = Field(default=2_000, ge=10, le=5_000)
    knowledge_extraction_max_unpacked_bytes: int = Field(
        default=100 * 1024 * 1024, ge=1024 * 1024, le=256 * 1024 * 1024
    )
    knowledge_extraction_max_text_bytes: int = Field(
        default=10 * 1024 * 1024, ge=100_000, le=20_000_000
    )
    knowledge_provider_daily_token_budget_per_user: int = Field(
        default=100_000, ge=1_000, le=10_000_000
    )
    knowledge_external_processing_requires_consent: bool = True
    knowledge_default_apply_mode: Literal["brief_reminder", "explain"] = "brief_reminder"
    knowledge_max_quote_chars: int = Field(default=400, ge=50, le=2_000)
    council_daily_sessions_per_user: int = Field(default=5, ge=1, le=100)
    council_evidence_max_chunks: int = Field(default=12, ge=1, le=100)
    council_evidence_max_chars: int = Field(default=40_000, ge=1_000, le=200_000)

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

    @field_validator("knowledge_asset_root")
    @classmethod
    def safe_knowledge_asset_root(cls, value: str) -> str:
        clean = value.strip()
        path = PurePosixPath(clean)
        if (
            not clean.startswith("/")
            or clean.startswith("//")
            or "\\" in clean
            or ".." in path.parts
            or path in {PurePosixPath("/"), PurePosixPath("/data")}
            or len(path.parts) < 3
        ):
            raise ValueError("KNOWLEDGE_ASSET_ROOT must be a dedicated absolute POSIX path")
        return str(path)

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

        knowledge_children = (
            self.enable_knowledge_capture,
            self.enable_knowledge_runner,
            self.enable_knowledge_retrieval,
            self.enable_knowledge_embeddings,
            self.enable_knowledge_ocr,
            self.enable_knowledge_media,
            self.enable_external_vision,
            self.enable_council,
            self.enable_scheduled_council,
            self.enable_knowledge_export,
        )
        if any(knowledge_children) and not self.enable_knowledge_hub:
            raise ValueError("Knowledge child features require ENABLE_KNOWLEDGE_HUB")
        if self.enable_knowledge_runner and not self.enable_knowledge_capture:
            raise ValueError("Knowledge runner requires ENABLE_KNOWLEDGE_CAPTURE")
        if (self.enable_knowledge_ocr or self.enable_knowledge_media) and not (
            self.enable_knowledge_capture
        ):
            raise ValueError("Knowledge OCR/media require ENABLE_KNOWLEDGE_CAPTURE")
        if self.enable_knowledge_embeddings and not self.enable_knowledge_retrieval:
            raise ValueError("Knowledge embeddings require ENABLE_KNOWLEDGE_RETRIEVAL")
        if self.enable_council and not self.enable_knowledge_retrieval:
            raise ValueError("Council requires ENABLE_KNOWLEDGE_RETRIEVAL")
        if self.enable_scheduled_council and not self.enable_council:
            raise ValueError("Scheduled Council requires ENABLE_COUNCIL")
        if self.enable_external_vision and not self.enable_knowledge_capture:
            raise ValueError("External vision requires ENABLE_KNOWLEDGE_CAPTURE")
        if self.enable_external_vision and not (
            self.knowledge_external_processing_requires_consent
        ):
            raise ValueError("External vision requires explicit per-user consent policy")
        if not self.knowledge_external_processing_requires_consent:
            raise ValueError("External Knowledge processing consent cannot be disabled")
        if self.knowledge_max_source_bytes > self.knowledge_daily_ingest_bytes_per_user:
            raise ValueError("Knowledge source limit cannot exceed the daily ingest quota")
        if self.knowledge_max_source_bytes > self.knowledge_daily_ingest_bytes_per_space:
            raise ValueError("Knowledge source limit cannot exceed the per-space daily quota")
        if self.knowledge_daily_ingest_bytes_per_user > self.knowledge_storage_quota_bytes_per_user:
            raise ValueError("Knowledge daily ingest quota cannot exceed storage quota")
        if self.database_url.startswith("sqlite") and self.knowledge_runner_concurrency != 1:
            raise ValueError("SQLite Knowledge deployments require exactly one runner")
        if (
            self.knowledge_daily_ingest_bytes_per_space
            > self.knowledge_storage_quota_bytes_per_space
        ):
            raise ValueError("Knowledge per-space daily quota cannot exceed its storage quota")
        if (
            self.knowledge_max_source_bytes + self.knowledge_extraction_max_text_bytes
            > self.knowledge_storage_quota_bytes_per_user
        ):
            raise ValueError("One Knowledge source and extraction must fit the user storage quota")
        if (
            self.knowledge_max_source_bytes + self.knowledge_extraction_max_text_bytes
            > self.knowledge_storage_quota_bytes_per_space
        ):
            raise ValueError("One Knowledge source and extraction must fit the space storage quota")
        if self.knowledge_runner_heartbeat_seconds * 2 >= self.knowledge_runner_lease_seconds:
            raise ValueError("Knowledge runner heartbeat must be less than half the lease")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """Used by diagnostics and tests after changing environment variables."""
    get_settings.cache_clear()
