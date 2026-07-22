import asyncio
import logging
from collections.abc import Callable

from pydantic import ValidationError
from telegram import Update
from telegram.ext import Application

from .ai import AIService, create_ai_service
from .bot import FutureSelfBot
from .config import Settings, get_settings
from .db import Database
from .transcription import TranscriptionService, create_transcription_service

logger = logging.getLogger(__name__)
ApplicationRunner = Callable[[Application], None]


def format_configuration_error(exc: ValidationError) -> str:
    error_text = str(exc)
    known_variables = ("TELEGRAM_BOT_TOKEN", "AI_API_KEY", "TRANSCRIPTION_API_KEY")
    invalid = ["TELEGRAM_BOT_TOKEN", "AI_API_KEY"]
    invalid.extend(name for name in known_variables if name in error_text)
    if any(item["loc"] and item["loc"][0] == "telegram_bot_token" for item in exc.errors()):
        invalid.append("TELEGRAM_BOT_TOKEN")
    invalid = sorted(set(invalid))
    names = ", ".join(invalid) or "параметры в .env"
    return (
        f"Ошибка конфигурации. Проверьте обязательные переменные: {names}. Значения не выводятся."
    )


def create_application(
    settings: Settings,
    db: Database,
    ai: AIService,
    transcription: TranscriptionService,
) -> Application:
    return FutureSelfBot(settings, db, ai, transcription).build()


def run(
    settings: Settings,
    *,
    ai: AIService | None = None,
    transcription: TranscriptionService | None = None,
    application_runner: ApplicationRunner | None = None,
) -> None:
    db = Database(
        settings.database_url,
        sqlite_busy_timeout_ms=settings.sqlite_busy_timeout_ms,
        sqlite_wal_enabled=settings.sqlite_wal_enabled,
    )
    ai_service = ai or create_ai_service(settings)
    transcription_service = transcription or create_transcription_service(settings)
    try:
        application = create_application(settings, db, ai_service, transcription_service)
        runner = application_runner or (
            lambda app: app.run_polling(allowed_updates=Update.ALL_TYPES)
        )
        runner(application)
    finally:
        asyncio.run(db.dispose())


def main() -> None:
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise SystemExit(format_configuration_error(exc)) from None
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for sensitive_logger in ("httpx", "httpcore", "openai", "telegram"):
        logging.getLogger(sensitive_logger).setLevel(logging.CRITICAL)
    for noisy_logger in ("apscheduler", "sqlalchemy.engine"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    try:
        run(settings)
    except KeyboardInterrupt:
        logger.info("Bot stopped by operator")
    except Exception as exc:
        logger.error("Critical startup failure error_type=%s", type(exc).__name__)
        raise SystemExit(
            "Не удалось запустить бота или подключиться к сервису. "
            "Запустите `python -m future_self.doctor` и проверьте сеть, ключи и базу данных."
        ) from None


if __name__ == "__main__":
    main()
