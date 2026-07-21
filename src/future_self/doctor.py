import argparse
import asyncio
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlunsplit

from alembic.config import Config
from alembic.script import ScriptDirectory
from openai import APITimeoutError, AuthenticationError, BadRequestError
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text

from .ai import create_ai_service
from .config import ENV_FILE_PATH, Settings
from .db import Database
from .lab_media import TelegramLabMetadata, process_lab_upload
from .transcription import create_transcription_service

MINIMUM_PYTHON = (3, 12)
DEFAULT_NETWORK_TIMEOUT = 15.0


class ProviderPresenceSettings(BaseSettings):
    telegram_bot_token: str | None = None
    ai_provider: str | None = None
    ai_api_key: str | None = None
    ai_model: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    transcription_provider: str = "disabled"
    transcription_api_key: str | None = None
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, env_file_encoding="utf-8", extra="ignore"
    )


@dataclass(slots=True)
class DiagnosticCheck:
    name: str
    status: str
    detail: str


@dataclass(slots=True)
class DoctorReport:
    checks: list[DiagnosticCheck] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(DiagnosticCheck(name, status, detail))

    @property
    def exit_code(self) -> int:
        return 1 if any(check.status == "FAIL" for check in self.checks) else 0


def safe_base_url(value: str | None) -> str:
    """Remove credentials, query, and fragment before printing a configured endpoint."""
    if not value:
        return "not configured"
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


def duplicate_env_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    counts: dict[str, int] = {}
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := pattern.match(line):
            key = match.group(1)
            counts[key] = counts.get(key, 0) + 1
    return sorted(key for key, count in counts.items() if count > 1)


def _project_root() -> Path:
    current = Path.cwd()
    if (current / "alembic.ini").exists():
        return current
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "alembic.ini").exists() else current


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


async def _check_database(report: DoctorReport, database_url: str) -> None:
    database = Database(database_url)
    try:
        async with database.sessions() as session:
            await session.execute(text("SELECT 1"))
        report.add("database", "OK", "Async database connection is available")
    except Exception as exc:
        report.add(
            "database",
            "FAIL",
            f"Database connection failed ({type(exc).__name__}); check DATABASE_URL",
        )
        await database.dispose()
        return
    try:
        async with database.sessions() as session:
            current_revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
        config = Config(str(_project_root() / "alembic.ini"))
        expected_revision = ScriptDirectory.from_config(config).get_current_head()
        if current_revision == expected_revision:
            report.add("migrations", "OK", f"Database is at Alembic head {expected_revision}")
        else:
            report.add(
                "migrations",
                "FAIL",
                "Database is not at Alembic head; run `alembic upgrade head`",
            )
    except Exception as exc:
        report.add(
            "migrations",
            "FAIL",
            f"Migration state is unavailable ({type(exc).__name__}); run `alembic upgrade head`",
        )
    finally:
        await database.dispose()


def _check_pdf_renderer(report: DoctorReport) -> None:
    try:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=144, height=144)
        output = BytesIO()
        writer.write(output)
        payload = output.getvalue()
        with TemporaryDirectory(prefix="future-self-pdf-doctor-") as directory:
            rendered = process_lab_upload(
                payload,
                TelegramLabMetadata("document", len(payload), "application/pdf"),
                temp_root=Path(directory),
            )
        if len(rendered.pages) != 1 or not rendered.pages[0].image_bytes.startswith(
            b"\xff\xd8\xff"
        ):
            raise RuntimeError("invalid_renderer_output")
        report.add(
            "pdf_renderer",
            "OK",
            f"Local PDFium render succeeded; pypdf {_package_version('pypdf')}; "
            f"pypdfium2 {_package_version('pypdfium2')}",
        )
    except Exception as exc:
        report.add(
            "pdf_renderer",
            "FAIL",
            f"Local PDF renderer failed ({type(exc).__name__}); reinstall application dependencies",
        )


async def run_provider_check(
    report: DoctorReport,
    name: str,
    provider: str,
    operation: Callable[[], Awaitable[object]],
    *,
    timeout_seconds: float = DEFAULT_NETWORK_TIMEOUT,
) -> None:
    try:
        await asyncio.wait_for(operation(), timeout=timeout_seconds)
        report.add(name, "OK", f"{provider} connection succeeded")
    except AuthenticationError:
        report.add(
            name,
            "FAIL",
            f"{provider} authentication failed; check the selected provider and API key",
        )
    except BadRequestError as exc:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        report.add(
            name,
            "FAIL",
            f"{provider} diagnostic request was rejected; "
            f"http_status={status or 'unknown'}; error_type=BadRequestError; "
            "check model and endpoint compatibility",
        )
    except (TimeoutError, APITimeoutError):
        report.add(name, "FAIL", f"{provider} check timed out after {timeout_seconds:g}s")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        report.add(name, "FAIL", f"{provider} connection failed ({type(exc).__name__})")


async def _check_network(
    report: DoctorReport,
    settings: Settings,
    presence: ProviderPresenceSettings,
    *,
    enabled: bool,
    timeout_seconds: float,
) -> None:
    if not enabled:
        report.add("network", "WARN", "Skipped; pass --network for low-cost API checks")
        return

    telegram_token = (presence.telegram_bot_token or "").strip()
    if telegram_token:
        from telegram import Bot

        async def telegram_check() -> object:
            async with Bot(telegram_token):
                return True

        await run_provider_check(
            report,
            "telegram_network",
            "Telegram",
            telegram_check,
            timeout_seconds=timeout_seconds,
        )
    else:
        report.add("telegram_network", "WARN", "Skipped; TELEGRAM_BOT_TOKEN is missing")

    text_key = (presence.ai_api_key or presence.openai_api_key or "").strip()
    if text_key:
        ai_service = create_ai_service(settings)

        async def text_check() -> object:
            return await ai_service.health_check()

        await run_provider_check(
            report,
            "text_llm_network",
            settings.ai_provider,
            text_check,
            timeout_seconds=timeout_seconds,
        )
        await ai_service.client.close()
    else:
        report.add("text_llm_network", "WARN", "Skipped; AI_API_KEY is missing")

    if settings.transcription_provider == "disabled":
        report.add(
            "transcription_network",
            "WARN",
            "Skipped; transcription is disabled and text inbox remains available",
        )
        return
    transcription_key = (presence.transcription_api_key or "").strip()
    if settings.transcription_provider == "openai" and not transcription_key:
        report.add("transcription_network", "WARN", "Skipped; TRANSCRIPTION_API_KEY is missing")
        return
    transcription = create_transcription_service(settings)

    async def transcription_check() -> object:
        return await transcription.client.models.retrieve(settings.transcription_model)

    await run_provider_check(
        report,
        "transcription_network",
        settings.transcription_provider,
        transcription_check,
        timeout_seconds=timeout_seconds,
    )
    await transcription.client.close()


async def run_diagnostics(
    *,
    network: bool = False,
    database_url: str | None = None,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
) -> DoctorReport:
    report = DoctorReport()
    python_ok = sys.version_info >= MINIMUM_PYTHON
    report.add(
        "python",
        "OK" if python_ok else "FAIL",
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    try:
        import openai  # noqa: F401
        import telegram  # noqa: F401

        report.add(
            "sdk_imports",
            "OK",
            f"python-telegram-bot {_package_version('python-telegram-bot')}; "
            f"openai {_package_version('openai')}",
        )
    except ImportError as exc:
        report.add("sdk_imports", "FAIL", f"SDK import failed ({type(exc).__name__})")

    presence = ProviderPresenceSettings()
    report.add("env_file", "OK", f"Loaded {ENV_FILE_PATH}")
    if duplicates := duplicate_env_keys(ENV_FILE_PATH):
        report.add(
            "env_duplicates",
            "WARN",
            f"Duplicate variable names: {', '.join(duplicates)}; the last value wins",
        )
    text_key = (presence.ai_api_key or "").strip()
    legacy_key = (presence.openai_api_key or "").strip()
    if not text_key and legacy_key:
        report.add(
            "legacy_configuration",
            "WARN",
            "OPENAI_API_KEY fallback is active; migrate to AI_API_KEY",
        )
    if presence.openai_model and not presence.ai_model:
        report.add(
            "legacy_model",
            "WARN",
            "OPENAI_MODEL fallback may be active; migrate to AI_MODEL",
        )

    missing = []
    if not (presence.telegram_bot_token or "").strip():
        missing.append("TELEGRAM_BOT_TOKEN")
    if not text_key and not legacy_key:
        missing.append("AI_API_KEY")
    if (
        presence.transcription_provider == "openai"
        and not (presence.transcription_api_key or "").strip()
    ):
        missing.append("TRANSCRIPTION_API_KEY")
    if missing:
        report.add(
            "secrets",
            "WARN",
            f"Not configured: {', '.join(missing)}; values are never printed",
        )
    else:
        report.add("secrets", "OK", "Required variables are present; values are hidden")

    try:
        settings = Settings(
            telegram_bot_token=presence.telegram_bot_token or "0:diagnostic-placeholder",
            ai_provider=(
                presence.ai_provider or ("openai" if legacy_key and not text_key else "openrouter")
            ),
            ai_api_key=text_key or legacy_key or "diagnostic-placeholder",
            ai_model=presence.ai_model or presence.openai_model,
            transcription_api_key=(
                presence.transcription_api_key
                or (
                    "diagnostic-placeholder"
                    if presence.transcription_provider == "openai"
                    else None
                )
            ),
            **({"database_url": database_url} if database_url else {}),
        )
        report.add("configuration", "OK", ".env and typed settings loaded successfully")
    except ValidationError:
        report.add("configuration", "FAIL", "Invalid non-secret value in .env")
        return report

    report.add(
        "text_llm",
        "OK",
        f"provider={settings.ai_provider}; base_url={safe_base_url(settings.ai_base_url)}; "
        f"model={settings.ai_model}",
    )
    transcription_status = "WARN" if settings.transcription_provider == "disabled" else "OK"
    report.add(
        "transcription",
        transcription_status,
        f"provider={settings.transcription_provider}; "
        f"base_url={safe_base_url(settings.transcription_base_url)}; "
        f"model={settings.transcription_model}; "
        f"enable_voice={str(settings.enable_voice).lower()}; "
        f"key_configured={str(bool(settings.transcription_api_key)).lower()}",
    )
    _check_pdf_renderer(report)
    await _check_database(report, settings.database_url)
    await _check_network(
        report,
        settings,
        presence,
        enabled=network,
        timeout_seconds=network_timeout,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe local runtime diagnostics")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Check Telegram, text LLM, and STT endpoints; sends no user messages",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_NETWORK_TIMEOUT,
        help="Timeout for each network check in seconds",
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(
            run_diagnostics(network=args.network, network_timeout=max(args.timeout, 0.1))
        )
    except KeyboardInterrupt:
        print("Doctor interrupted by operator.")
        raise SystemExit(130) from None
    for check in report.checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    raise SystemExit(report.exit_code)


if __name__ == "__main__":
    main()
