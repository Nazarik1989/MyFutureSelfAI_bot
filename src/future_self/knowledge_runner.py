"""Secret-free, networkless Knowledge ingestion process.

This module never imports or instantiates the bot ``Settings`` object.  The runner
receives only an explicit allowlist of database/storage/limit variables and is meant
to run in a container with ``network_mode: none``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from .db import Database
from .knowledge import (
    ClaimedKnowledgeJob,
    KnowledgeExtractionResult,
    KnowledgeQuotaPolicy,
    KnowledgeService,
)
from .knowledge_backup import maintenance_paused
from .knowledge_extraction import (
    ExtractionLimits,
    KnowledgeExtractionError,
    KnowledgeExtractor,
)
from .knowledge_storage import KnowledgeAssetStore, KnowledgeStorageError
from .models import KnowledgeRuntimeState, KnowledgeSourceRevision

logger = logging.getLogger(__name__)

_ALLOWED_ENVIRONMENT = frozenset(
    {
        "DATABASE_URL",
        "ENABLE_KNOWLEDGE_RUNNER",
        "KNOWLEDGE_ASSET_ROOT",
        "KNOWLEDGE_RUNNER_CONCURRENCY",
        "KNOWLEDGE_RUNNER_POLL_SECONDS",
        "KNOWLEDGE_RUNNER_LEASE_SECONDS",
        "KNOWLEDGE_RUNNER_HEARTBEAT_SECONDS",
        "KNOWLEDGE_MAX_SOURCE_BYTES",
        "KNOWLEDGE_EXTRACTION_WALL_SECONDS",
        "KNOWLEDGE_EXTRACTION_MAX_PAGES",
        "KNOWLEDGE_EXTRACTION_MAX_ARCHIVE_ENTRIES",
        "KNOWLEDGE_EXTRACTION_MAX_UNPACKED_BYTES",
        "KNOWLEDGE_EXTRACTION_MAX_TEXT_BYTES",
        "RUNTIME_MIN_FREE_BYTES",
        "SQLITE_WAL_ENABLED",
        "SQLITE_BUSY_TIMEOUT_MS",
        "LOG_LEVEL",
    }
)
_SECRET_MARKERS = ("TOKEN", "API_KEY", "SECRET", "PASSWORD")
_FORBIDDEN_PROCESS_ENVIRONMENT = frozenset(
    {
        "TELEGRAM_BOT_TOKEN",
        "AI_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TRANSCRIPTION_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    }
)


class KnowledgeRunnerSettings(BaseSettings):
    """Minimal runner settings; notably there are no provider secret fields."""

    database_url: str = "sqlite+aiosqlite:////data/future_self.db"
    enable_knowledge_runner: bool = False
    knowledge_asset_root: str = "/data/knowledge"
    knowledge_runner_concurrency: int = Field(default=1, ge=1, le=8)
    knowledge_runner_poll_seconds: float = Field(default=2.0, ge=0.25, le=60.0)
    knowledge_runner_lease_seconds: int = Field(default=120, ge=30, le=3600)
    knowledge_runner_heartbeat_seconds: int = Field(default=30, ge=5, le=600)
    knowledge_max_source_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1_000_000, le=100 * 1024 * 1024
    )
    knowledge_extraction_wall_seconds: int = Field(default=30, ge=5, le=300)
    knowledge_extraction_max_pages: int = Field(default=500, ge=1, le=500)
    knowledge_extraction_max_archive_entries: int = Field(default=2_000, ge=10, le=5_000)
    knowledge_extraction_max_unpacked_bytes: int = Field(
        default=100 * 1024 * 1024, ge=1024 * 1024, le=256 * 1024 * 1024
    )
    knowledge_extraction_max_text_bytes: int = Field(
        default=10 * 1024 * 1024, ge=100_000, le=20_000_000
    )
    runtime_min_free_bytes: int = Field(default=1024 * 1024 * 1024, ge=0)
    sqlite_wal_enabled: bool = True
    sqlite_busy_timeout_ms: int = Field(default=5_000, ge=1_000, le=60_000)
    log_level: str = "INFO"

    # Do not auto-read the bot .env file. Docker must pass the allowlisted values.
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    @field_validator("database_url")
    @classmethod
    def async_database_driver(cls, value: str) -> str:
        clean = value.strip()
        if clean.startswith("postgresql://"):
            return clean.replace("postgresql://", "postgresql+asyncpg://", 1)
        return clean

    @field_validator("knowledge_asset_root", mode="before")
    @classmethod
    def dedicated_asset_root(cls, value: object) -> str:
        clean = str(value).strip()
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
        return clean

    @model_validator(mode="after")
    def safe_runner_configuration(self) -> KnowledgeRunnerSettings:
        if self.database_url.startswith("sqlite") and self.knowledge_runner_concurrency != 1:
            raise ValueError("SQLite Knowledge deployments require exactly one runner")
        if self.knowledge_runner_heartbeat_seconds * 2 >= self.knowledge_runner_lease_seconds:
            raise ValueError("Knowledge runner heartbeat must be less than half the lease")
        if self.database_url.startswith("sqlite"):
            database_name = make_url(self.database_url).database
            if not database_name:
                raise ValueError("SQLite Knowledge runner requires a database path")
            database_path = PurePosixPath(database_name.replace("\\", "/"))
            asset_path = PurePosixPath(self.knowledge_asset_root)
            if database_path == asset_path or asset_path in database_path.parents:
                raise ValueError("SQLite database must be outside KNOWLEDGE_ASSET_ROOT")
        return self


def runner_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Build the explicit process/container allowlist and reject secret-shaped keys."""

    selected = {key: value for key, value in source.items() if key in _ALLOWED_ENVIRONMENT}
    if any(marker in key.upper() for key in selected for marker in _SECRET_MARKERS):
        raise ValueError("runner secret allowlist violation")
    return selected


def assert_secret_free_process_environment(source: Mapping[str, str]) -> None:
    """Fail without reading or reporting credential values or key names."""

    if {key.upper() for key in source} & _FORBIDDEN_PROCESS_ENVIRONMENT:
        raise ValueError("runner process environment contains forbidden credentials")


@dataclass(frozen=True, slots=True)
class RunnerOutcome:
    claimed: bool
    status: str | None = None


class KnowledgeIngestionRunner:
    def __init__(
        self,
        service: KnowledgeService,
        storage: KnowledgeAssetStore,
        extractor: KnowledgeExtractor,
        settings: KnowledgeRunnerSettings,
        *,
        worker_id: str | None = None,
    ) -> None:
        self.service = service
        self.storage = storage
        self.extractor = extractor
        self.settings = settings
        self.worker_id = worker_id or f"runner-{uuid.uuid4().hex}"
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        if not self.settings.enable_knowledge_runner:
            raise RuntimeError("Knowledge runner is disabled")
        logger.info("knowledge_runner_started worker=%s", self.worker_id)
        try:
            while not self._stop.is_set():
                if maintenance_paused(self.storage.root):
                    await self._wait_poll()
                    continue
                outcome = await self.process_one()
                if not outcome.claimed:
                    await self._wait_poll()
        finally:
            logger.info("knowledge_runner_stopped worker=%s", self.worker_id)

    async def _wait_poll(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._stop.wait(), timeout=self.settings.knowledge_runner_poll_seconds
            )

    async def process_one(self) -> RunnerOutcome:
        if maintenance_paused(self.storage.root):
            return RunnerOutcome(False)
        job = await self.service.claim_next_job(
            self.worker_id,
            lease_seconds=self.settings.knowledge_runner_lease_seconds,
        )
        if job is None:
            return RunnerOutcome(False)
        logger.info(
            "knowledge_job_claimed job=%s kind=%s attempt=%s",
            job.public_id,
            job.job_type,
            job.attempt_count,
        )
        if job.cancel_requested:
            await self.service.cancel_claimed_job(job.id, job.lease_token)
            return RunnerOutcome(True, "cancelled")
        if job.job_type == "purge":
            return await self._purge(job)
        if job.job_type != "extract":
            await self.service.fail_job(
                job.id,
                job.lease_token,
                failure_kind="permanent",
                safe_error_code="unsupported_job_type",
            )
            return RunnerOutcome(True, "failed")
        return await self._extract(job)

    async def _purge(self, job: ClaimedKnowledgeJob) -> RunnerOutcome:
        heartbeat = asyncio.create_task(self._heartbeat(job))
        try:
            for storage_key in job.asset_keys:
                await asyncio.to_thread(self.storage.delete_asset, storage_key)
            finalized = await self.service.finalize_purge_job(job.id, job.lease_token)
            return RunnerOutcome(True, "purged" if finalized else "stale")
        except KnowledgeStorageError:
            # Some assets may already be gone. A later explicit retry is safe because
            # delete_asset is idempotent and DB state remains purge_failed.
            await self.service.fail_job(
                job.id,
                job.lease_token,
                failure_kind="permanent",
                safe_error_code="purge_io_failed",
            )
            return RunnerOutcome(True, "purge_failed")
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _extract(self, job: ClaimedKnowledgeJob) -> RunnerOutcome:
        heartbeat = asyncio.create_task(self._heartbeat(job))
        stored_key: str | None = None
        finalization_started = False
        try:
            if (
                job.original_storage_key is None
                or job.original_sha256 is None
                or job.size_bytes is None
                or job.detected_format is None
            ):
                raise KnowledgeStorageError("source_metadata_missing")
            await asyncio.to_thread(
                self.storage.verify_asset,
                job.original_storage_key,
                expected_size=job.size_bytes,
                expected_sha256=job.original_sha256,
            )
            with tempfile.TemporaryDirectory(prefix="knowledge-runner-") as temporary_name:
                temporary = Path(temporary_name)
                try:
                    await asyncio.to_thread(temporary.chmod, 0o700)
                except OSError:
                    pass
                input_path = temporary / "input.bin"
                copied = await asyncio.to_thread(
                    self.storage.copy_asset_to, job.original_storage_key, input_path
                )
                if copied.size_bytes != job.size_bytes or copied.sha256 != job.original_sha256:
                    raise KnowledgeStorageError("asset_changed_during_copy")
                result = await asyncio.to_thread(
                    self.extractor.extract_path,
                    input_path,
                    job.detected_format,
                    expected_size=job.size_bytes,
                    expected_sha256=job.original_sha256,
                )
            if job.cancel_requested or await self._cancel_requested(job):
                await self.service.cancel_claimed_job(job.id, job.lease_token)
                return RunnerOutcome(True, "cancelled")
            if result.status not in {"ready", "partial"}:
                kind = "quarantine" if result.status == "quarantined" else "permanent"
                await self.service.fail_job(
                    job.id,
                    job.lease_token,
                    failure_kind=kind,
                    safe_error_code=result.error_code or "extraction_failed",
                )
                return RunnerOutcome(True, result.status)
            extracted_key = extracted_hash = None
            extracted_size = None
            if result.text_bytes:
                staged = await asyncio.to_thread(
                    self.storage.stage_bytes, result.text_bytes, extracted=True
                )
                try:
                    stored = await asyncio.to_thread(
                        self.storage.finalize, staged, kind="extracted"
                    )
                except BaseException:
                    await asyncio.to_thread(self.storage.discard_staged, staged)
                    raise
                stored_key = stored.storage_key
                extracted_key = stored.storage_key
                extracted_hash = stored.sha256
                extracted_size = stored.size_bytes
            # Once the database finalization call starts, an exception or task
            # cancellation has an ambiguous commit outcome.  Retaining a possible
            # orphan is recoverable by the storage audit; deleting an asset that the
            # committed revision already references is not.
            finalization_started = True
            finalized = await self.service.finalize_job(
                job.id,
                job.lease_token,
                KnowledgeExtractionResult(
                    status=result.status,
                    extracted_storage_key=extracted_key,
                    extracted_sha256=extracted_hash,
                    extracted_size_bytes=extracted_size,
                    metadata=result.metadata,
                    safe_error_code=result.error_code,
                ),
            )
            if not finalized:
                if stored_key:
                    await asyncio.to_thread(self.storage.delete_asset, stored_key)
                    stored_key = None
                return RunnerOutcome(True, "stale")
            logger.info("knowledge_job_finished job=%s status=%s", job.public_id, result.status)
            return RunnerOutcome(True, result.status)
        except asyncio.CancelledError:
            if stored_key and not finalization_started:
                with contextlib.suppress(KnowledgeStorageError):
                    await asyncio.to_thread(self.storage.delete_asset, stored_key)
            raise
        except KnowledgeStorageError as exc:
            if stored_key and not finalization_started:
                with contextlib.suppress(KnowledgeStorageError):
                    await asyncio.to_thread(self.storage.delete_asset, stored_key)
            code = str(exc)
            failure_kind = (
                "quarantine"
                if code
                in {
                    "asset_hash_mismatch",
                    "asset_size_mismatch",
                    "asset_changed_during_copy",
                    "unsafe_asset",
                }
                else "permanent"
            )
            await self._safe_fail(job, failure_kind, code)
            return RunnerOutcome(True, failure_kind)
        except KnowledgeExtractionError as exc:
            if stored_key and not finalization_started:
                with contextlib.suppress(KnowledgeStorageError):
                    await asyncio.to_thread(self.storage.delete_asset, stored_key)
            failure_kind = (
                "quarantine" if exc.quarantined else "retryable" if exc.retryable else "permanent"
            )
            await self._safe_fail(job, failure_kind, str(exc))
            return RunnerOutcome(True, failure_kind)
        except Exception as exc:
            if stored_key and not finalization_started:
                with contextlib.suppress(KnowledgeStorageError):
                    await asyncio.to_thread(self.storage.delete_asset, stored_key)
            logger.error(
                "knowledge_job_error job=%s error_type=%s", job.public_id, type(exc).__name__
            )
            await self._safe_fail(job, "retryable", "runner_internal_error")
            return RunnerOutcome(True, "retryable")
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, job: ClaimedKnowledgeJob) -> None:
        while True:
            await asyncio.sleep(self.settings.knowledge_runner_heartbeat_seconds)
            alive = await self.service.heartbeat_job(
                job.id,
                job.lease_token,
                lease_seconds=self.settings.knowledge_runner_lease_seconds,
            )
            if not alive:
                return

    async def _cancel_requested(self, job: ClaimedKnowledgeJob) -> bool:
        checker = getattr(self.service, "job_cancel_requested", None)
        return bool(await checker(job.id, job.lease_token)) if checker is not None else False

    async def _safe_fail(self, job: ClaimedKnowledgeJob, kind: str, code: str) -> None:
        safe_kind = kind if kind in {"retryable", "permanent", "quarantine"} else "retryable"
        safe_code = (
            code if code and len(code) <= 64 and code.replace("_", "").isalnum() else "error"
        )
        await self.service.fail_job(
            job.id,
            job.lease_token,
            failure_kind=safe_kind,
            safe_error_code=safe_code,
        )


def _install_signal_handlers(runner: KnowledgeIngestionRunner) -> None:
    loop = asyncio.get_running_loop()
    for value in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(value, runner.request_stop)


def _database(settings: KnowledgeRunnerSettings) -> Database:
    return Database(
        settings.database_url,
        sqlite_busy_timeout_ms=settings.sqlite_busy_timeout_ms,
        sqlite_wal_enabled=settings.sqlite_wal_enabled,
    )


def build_runner(settings: KnowledgeRunnerSettings, database: Database) -> KnowledgeIngestionRunner:
    storage = KnowledgeAssetStore(
        Path(settings.knowledge_asset_root),
        max_source_bytes=settings.knowledge_max_source_bytes,
        max_extracted_bytes=settings.knowledge_extraction_max_text_bytes,
        min_free_bytes=settings.runtime_min_free_bytes,
    )
    limits = ExtractionLimits(
        max_source_bytes=settings.knowledge_max_source_bytes,
        max_text_chars=settings.knowledge_extraction_max_text_bytes // 4,
        max_pdf_pages=settings.knowledge_extraction_max_pages,
        max_archive_files=settings.knowledge_extraction_max_archive_entries,
        max_unpacked_bytes=settings.knowledge_extraction_max_unpacked_bytes,
        timeout_seconds=settings.knowledge_extraction_wall_seconds,
    )
    return KnowledgeIngestionRunner(
        KnowledgeService(
            database,
            quota_policy=KnowledgeQuotaPolicy(
                max_source_bytes=settings.knowledge_max_source_bytes,
                max_extracted_bytes=settings.knowledge_extraction_max_text_bytes,
            ),
        ),
        storage,
        KnowledgeExtractor(
            temp_root=Path(tempfile.gettempdir()) / "knowledge-runner", limits=limits
        ),
        settings,
    )


async def _run(
    settings: KnowledgeRunnerSettings,
    *,
    doctor: bool = False,
    full_audit: bool = False,
) -> None:
    database = _database(settings)
    try:
        runner = build_runner(settings, database)
        if doctor or full_audit:
            async with database.sessions() as session:
                await session.execute(text("SELECT 1"))
                runtime_row = await session.scalar(
                    select(KnowledgeRuntimeState.id).where(KnowledgeRuntimeState.id == 1)
                )
                if runtime_row != 1:
                    raise RuntimeError("Knowledge runtime state is unavailable")
                if not full_audit:
                    print(json.dumps({"status": "ok", "network_required": False}))
                    return
                revisions = (
                    await session.execute(
                        select(
                            KnowledgeSourceRevision.id,
                            KnowledgeSourceRevision.original_revision_id,
                            KnowledgeSourceRevision.original_storage_key,
                            KnowledgeSourceRevision.size_bytes,
                            KnowledgeSourceRevision.sha256,
                            KnowledgeSourceRevision.extracted_storage_key,
                            KnowledgeSourceRevision.extracted_size_bytes,
                            KnowledgeSourceRevision.extracted_sha256,
                        )
                    )
                ).all()
            originals = {
                revision_id: (original_key, original_size, original_hash)
                for (
                    revision_id,
                    _original_revision_id,
                    original_key,
                    original_size,
                    original_hash,
                    _extracted_key,
                    _extracted_size,
                    _extracted_hash,
                ) in revisions
                if original_key is not None
            }
            referenced: list[str] = []
            for (
                _revision_id,
                original_revision_id,
                original_key,
                original_size,
                original_hash,
                extracted_key,
                extracted_size,
                extracted_hash,
            ) in revisions:
                if original_key is None and original_revision_id is not None:
                    original = originals.get(original_revision_id)
                    if original is None:
                        raise RuntimeError("Knowledge original revision is unavailable")
                    original_key, root_size, root_hash = original
                    if root_size != original_size or root_hash != original_hash:
                        raise RuntimeError("Knowledge original revision metadata mismatch")
                if original_key is None:
                    raise RuntimeError("Knowledge original asset is unavailable")
                await asyncio.to_thread(
                    runner.storage.verify_asset,
                    original_key,
                    expected_size=original_size,
                    expected_sha256=original_hash,
                )
                referenced.append(original_key)
                if extracted_key is not None:
                    await asyncio.to_thread(
                        runner.storage.verify_asset,
                        extracted_key,
                        expected_size=extracted_size,
                        expected_sha256=extracted_hash,
                    )
                    referenced.append(extracted_key)
            audit = runner.storage.audit(referenced)
            if not audit.ok:
                raise RuntimeError("Knowledge storage audit failed")
            print(json.dumps({"status": "ok", "network_required": False}))
            return
        _install_signal_handlers(runner)
        await runner.run()
    finally:
        await database.dispose()


def main() -> None:
    os.umask(0o077)
    assert_secret_free_process_environment(os.environ)
    parser = argparse.ArgumentParser(description="Offline Knowledge ingestion runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--doctor", action="store_true")
    mode.add_argument("--full-audit", action="store_true")
    arguments = parser.parse_args()
    settings = KnowledgeRunnerSettings(_env_file=None)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_run(settings, doctor=arguments.doctor, full_audit=arguments.full_audit))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("knowledge_runner_fatal error_type=%s", type(exc).__name__)
        raise SystemExit("Knowledge runner failed; inspect safe operational logs.") from None


if __name__ == "__main__":
    main()
