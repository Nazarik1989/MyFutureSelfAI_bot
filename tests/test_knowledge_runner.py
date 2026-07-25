from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import future_self.knowledge_runner as runner_module
from future_self.knowledge import ClaimedKnowledgeJob, KnowledgeService
from future_self.knowledge_extraction import (
    ExtractionResult,
    KnowledgeExtractionError,
)
from future_self.knowledge_runner import (
    KnowledgeIngestionRunner,
    KnowledgeRunnerSettings,
    _run,
    assert_secret_free_process_environment,
    runner_environment,
)
from future_self.knowledge_storage import KnowledgeAssetStore


class FakeService:
    def __init__(self, job: ClaimedKnowledgeJob | None) -> None:
        self.job = job
        self.fail_calls: list[tuple[str, str]] = []
        self.finalized_results: list[object] = []
        self.cancelled = 0
        self.purged = 0
        self.finalize_result = True
        self.finalize_error: Exception | None = None
        self.finalize_started: asyncio.Event | None = None
        self.finalize_release: asyncio.Event | None = None
        self.cancel_after_extract = False
        self.heartbeat_result = True

    async def claim_next_job(self, worker_id: str, *, lease_seconds: int):
        del worker_id, lease_seconds
        claimed, self.job = self.job, None
        return claimed

    async def heartbeat_job(self, job_id: int, lease_token: str, *, lease_seconds: int) -> bool:
        del job_id, lease_token, lease_seconds
        return self.heartbeat_result

    async def finalize_job(self, job_id: int, lease_token: str, result: object) -> bool:
        del job_id, lease_token
        if self.finalize_started is not None:
            self.finalize_started.set()
        if self.finalize_release is not None:
            await self.finalize_release.wait()
        if self.finalize_error is not None:
            raise self.finalize_error
        self.finalized_results.append(result)
        return self.finalize_result

    async def fail_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        failure_kind: str,
        safe_error_code: str,
    ) -> bool:
        del job_id, lease_token
        self.fail_calls.append((failure_kind, safe_error_code))
        return True

    async def cancel_claimed_job(self, job_id: int, lease_token: str) -> bool:
        del job_id, lease_token
        self.cancelled += 1
        return True

    async def job_cancel_requested(self, job_id: int, lease_token: str) -> bool:
        del job_id, lease_token
        return self.cancel_after_extract

    async def finalize_purge_job(self, job_id: int, lease_token: str) -> bool:
        del job_id, lease_token
        self.purged += 1
        return True


class FakeExtractor:
    def __init__(self, result: ExtractionResult | Exception) -> None:
        self.result = result

    def extract_path(
        self,
        path: Path,
        source_format: str,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> ExtractionResult:
        assert path.is_file()
        assert source_format in {"txt", "image"}
        assert path.stat().st_size == expected_size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def settings(**updates: object) -> KnowledgeRunnerSettings:
    configured = KnowledgeRunnerSettings(_env_file=None)
    return configured.model_copy(update=updates)


def original(store: KnowledgeAssetStore, payload: bytes = b"private source"):
    return store.finalize(store.stage_bytes(payload), kind="original")


def extract_job(stored, *, cancel_requested: bool = False) -> ClaimedKnowledgeJob:
    return ClaimedKnowledgeJob(
        id=1,
        public_id="00000000-0000-0000-0000-000000000001",
        source_id=1,
        source_public_id="00000000-0000-0000-0000-000000000002",
        source_version=1,
        revision_id=1,
        revision_number=1,
        knowledge_space_id=1,
        job_type="extract",
        lease_token="lease-token",
        original_storage_key=stored.storage_key,
        original_sha256=stored.sha256,
        declared_mime="text/plain",
        detected_mime="text/plain",
        detected_format="txt",
        size_bytes=stored.size_bytes,
        attempt_count=1,
        max_attempts=3,
        cancel_requested=cancel_requested,
        asset_keys=(),
    )


def make_runner(tmp_path: Path, service: FakeService, extractor: FakeExtractor):
    store = KnowledgeAssetStore(
        tmp_path / "knowledge",
        max_source_bytes=1024 * 1024,
        max_extracted_bytes=1024 * 1024,
        min_free_bytes=0,
    )
    return (
        KnowledgeIngestionRunner(
            service,  # type: ignore[arg-type]
            store,
            extractor,  # type: ignore[arg-type]
            settings(
                knowledge_runner_poll_seconds=0.25,
                knowledge_runner_heartbeat_seconds=5,
                knowledge_runner_lease_seconds=30,
            ),
            worker_id="test-runner",
        ),
        store,
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            ExtractionResult(
                status="ready",
                source_format="txt",
                text_bytes=b"extracted text",
                text_sha256=hashlib.sha256(b"extracted text").hexdigest(),
                metadata={"encoding": "utf-8"},
            ),
            "ready",
        ),
        (
            ExtractionResult(
                status="partial",
                source_format="image",
                error_code="no_ocr",
                metadata={"image_format": "png"},
            ),
            "partial",
        ),
    ],
)
async def test_runner_finalizes_ready_and_honest_partial(
    tmp_path: Path, result: ExtractionResult, expected: str
) -> None:
    service = FakeService(None)
    runner, store = make_runner(tmp_path, service, FakeExtractor(result))
    stored = original(store)
    service.job = extract_job(stored)

    outcome = await runner.process_one()

    assert outcome.claimed and outcome.status == expected
    assert len(service.finalized_results) == 1
    committed = service.finalized_results[0]
    if expected == "ready":
        assert committed.extracted_storage_key is not None
        assert store.audit((stored.storage_key, committed.extracted_storage_key)).ok
    else:
        assert committed.extracted_storage_key is None
        assert store.audit((stored.storage_key,)).ok


@pytest.mark.parametrize(
    ("result", "failure_kind", "error_code"),
    [
        (
            ExtractionResult(
                status="quarantined",
                source_format="txt",
                error_code="archive_ratio_limit",
            ),
            "quarantine",
            "archive_ratio_limit",
        ),
        (
            KnowledgeExtractionError("worker_timeout", retryable=True),
            "retryable",
            "worker_timeout",
        ),
    ],
)
async def test_runner_classifies_quarantine_and_retryable_errors(
    tmp_path: Path,
    result: ExtractionResult | Exception,
    failure_kind: str,
    error_code: str,
) -> None:
    service = FakeService(None)
    runner, store = make_runner(tmp_path, service, FakeExtractor(result))
    service.job = extract_job(original(store))

    outcome = await runner.process_one()

    assert outcome.claimed
    assert service.fail_calls == [(failure_kind, error_code)]


@pytest.mark.parametrize("finalize_mode", ["stale", "error"])
async def test_runner_deletes_only_deterministically_uncommitted_extracted_asset(
    tmp_path: Path, finalize_mode: str
) -> None:
    result = ExtractionResult(
        status="ready",
        source_format="txt",
        text_bytes=b"result",
        text_sha256=hashlib.sha256(b"result").hexdigest(),
    )
    service = FakeService(None)
    runner, store = make_runner(tmp_path, service, FakeExtractor(result))
    stored = original(store)
    service.job = extract_job(stored)
    if finalize_mode == "stale":
        service.finalize_result = False
    else:
        service.finalize_error = RuntimeError("simulated")

    outcome = await runner.process_one()

    assert outcome.status == ("stale" if finalize_mode == "stale" else "retryable")
    audit = store.audit((stored.storage_key,))
    if finalize_mode == "stale":
        assert audit.ok
    else:
        assert len(audit.orphaned) == 1
        assert service.fail_calls == [("retryable", "runner_internal_error")]


async def test_runner_retains_extracted_asset_when_cancelled_during_finalize(
    tmp_path: Path,
) -> None:
    result = ExtractionResult(
        status="ready",
        source_format="txt",
        text_bytes=b"result",
        text_sha256=hashlib.sha256(b"result").hexdigest(),
    )
    service = FakeService(None)
    service.finalize_started = asyncio.Event()
    service.finalize_release = asyncio.Event()
    runner, store = make_runner(tmp_path, service, FakeExtractor(result))
    stored = original(store)
    service.job = extract_job(stored)

    task = asyncio.create_task(runner.process_one())
    await asyncio.wait_for(service.finalize_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    audit = store.audit((stored.storage_key,))
    assert len(audit.orphaned) == 1


async def test_runner_cancellation_and_idempotent_purge(tmp_path: Path) -> None:
    service = FakeService(None)
    runner, store = make_runner(
        tmp_path,
        service,
        FakeExtractor(ExtractionResult(status="partial", source_format="image")),
    )
    stored = original(store)
    service.job = extract_job(stored, cancel_requested=True)
    cancelled = await runner.process_one()
    assert cancelled.status == "cancelled" and service.cancelled == 1

    service.job = replace(
        extract_job(stored),
        id=2,
        job_type="purge",
        revision_id=None,
        revision_number=None,
        original_storage_key=None,
        original_sha256=None,
        declared_mime=None,
        detected_mime=None,
        detected_format=None,
        size_bytes=None,
        asset_keys=(stored.storage_key, stored.storage_key),
        cancel_requested=False,
    )
    purged = await runner.process_one()
    assert purged.status == "purged" and service.purged == 1
    assert store.audit(()).ok


async def test_runner_stops_without_busy_loop(tmp_path: Path) -> None:
    service = FakeService(None)
    runner, _ = make_runner(
        tmp_path,
        service,
        FakeExtractor(ExtractionResult(status="partial", source_format="image")),
    )
    runner.settings = runner.settings.model_copy(update={"enable_knowledge_runner": True})

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0)
    runner.request_stop()
    await asyncio.wait_for(task, timeout=1)


def test_runner_settings_and_environment_fail_closed() -> None:
    for unsafe_root in ("/", "/data", "../knowledge", "C:\\data\\knowledge"):
        with pytest.raises(ValidationError, match="dedicated absolute POSIX path"):
            KnowledgeRunnerSettings(_env_file=None, knowledge_asset_root=unsafe_root)
    with pytest.raises(ValidationError, match="exactly one runner"):
        KnowledgeRunnerSettings(_env_file=None, knowledge_runner_concurrency=2)
    with pytest.raises(ValidationError, match="outside KNOWLEDGE_ASSET_ROOT"):
        KnowledgeRunnerSettings(
            _env_file=None,
            knowledge_asset_root="/data/knowledge",
            database_url="sqlite+aiosqlite:////data/knowledge/unsafe.db",
        )

    environment = runner_environment(
        {
            "DATABASE_URL": "sqlite+aiosqlite:////data/future_self.db",
            "KNOWLEDGE_ASSET_ROOT": "/data/knowledge",
            "TELEGRAM_BOT_TOKEN": "private-bot-token",
            "AI_API_KEY": "private-ai-key",
            "HTTPS_PROXY": "https://private.invalid",
        }
    )
    assert set(environment) == {"DATABASE_URL", "KNOWLEDGE_ASSET_ROOT"}
    assert "private" not in repr(environment)
    with pytest.raises(ValueError, match="forbidden credentials"):
        assert_secret_free_process_environment({"TELEGRAM_BOT_TOKEN": "not-inspected"})


async def test_runner_doctor_is_cheap_and_full_audit_is_explicit(
    db, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await KnowledgeService(db).set_maintenance_paused(False)
    configured = settings(
        database_url=db.url,
        knowledge_asset_root=str(tmp_path / "doctor-knowledge"),
        runtime_min_free_bytes=0,
    )

    await _run(configured, doctor=True)
    assert '"status": "ok"' in capsys.readouterr().out
    await _run(configured, full_audit=True)
    assert '"status": "ok"' in capsys.readouterr().out


def test_runner_entrypoint_sets_private_umask(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    configured = settings()

    def close_coroutine(coroutine: object) -> None:
        coroutine.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(runner_module.os, "umask", lambda mode: calls.append(mode) or 0)
    monkeypatch.setattr(runner_module, "KnowledgeRunnerSettings", lambda **kwargs: configured)
    monkeypatch.setattr(runner_module.asyncio, "run", close_coroutine)
    monkeypatch.setattr(sys, "argv", ["future-self-knowledge-runner", "--doctor"])
    for key in runner_module._FORBIDDEN_PROCESS_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)

    runner_module.main()

    assert calls == [0o077]
