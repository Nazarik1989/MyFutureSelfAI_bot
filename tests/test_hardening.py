import subprocess
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError
from pypdf import PdfWriter
from sqlalchemy import text

from future_self.config import Settings
from future_self.db import Database
from future_self.safe_media.pdf import SafePdfError, render_pdf_pages
from future_self.safe_media.subprocess import (
    SafeSubprocessError,
    ensure_child,
    run_isolated_python_module,
    sanitized_environment,
    write_private_file,
)


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "telegram_bot_token": "123456:TEST",
        "ai_api_key": "test-key",
        "ai_model": "test-model",
    }
    values.update(overrides)
    return Settings(**values)


def pdf_bytes(*, javascript: bool = False) -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=144, height=144)
    if javascript:
        writer.add_js("app.alert('blocked')")
    output = __import__("io").BytesIO()
    writer.write(output)
    return output.getvalue()


def test_future_domains_are_disabled_and_approved_defaults_are_fixed() -> None:
    configured = settings()
    flags = {
        name: value
        for name, value in configured.model_dump().items()
        if name.startswith("enable_knowledge")
        or name in {"enable_council", "enable_scheduled_council", "enable_external_vision"}
    }
    assert flags
    assert not any(flags.values())
    assert configured.sqlite_wal_enabled is True
    assert configured.sqlite_busy_timeout_ms == 5_000
    assert configured.knowledge_asset_root == "/data/knowledge"
    assert configured.knowledge_runner_concurrency == 1
    assert configured.knowledge_external_processing_requires_consent is True
    assert configured.knowledge_default_apply_mode == "brief_reminder"


@pytest.mark.parametrize(
    "flag",
    [
        "enable_knowledge_capture",
        "enable_knowledge_runner",
        "enable_knowledge_retrieval",
        "enable_knowledge_embeddings",
        "enable_knowledge_ocr",
        "enable_knowledge_media",
        "enable_external_vision",
        "enable_council",
        "enable_scheduled_council",
        "enable_knowledge_export",
    ],
)
def test_child_feature_cannot_bypass_hub_gate(flag: str) -> None:
    with pytest.raises(ValidationError, match="ENABLE_KNOWLEDGE_HUB"):
        settings(**{flag: True})


def test_feature_dependencies_and_consent_fail_closed() -> None:
    with pytest.raises(ValidationError, match="ENABLE_KNOWLEDGE_CAPTURE"):
        settings(enable_knowledge_hub=True, enable_knowledge_runner=True)
    with pytest.raises(ValidationError, match="ENABLE_KNOWLEDGE_RETRIEVAL"):
        settings(enable_knowledge_hub=True, enable_knowledge_embeddings=True)
    with pytest.raises(ValidationError, match="ENABLE_KNOWLEDGE_RETRIEVAL"):
        settings(enable_knowledge_hub=True, enable_council=True)
    with pytest.raises(ValidationError, match="ENABLE_COUNCIL"):
        settings(enable_knowledge_hub=True, enable_scheduled_council=True)
    with pytest.raises(ValidationError, match="consent cannot be disabled"):
        settings(knowledge_external_processing_requires_consent=False)


def test_quotas_paths_and_sqlite_runner_are_bounded() -> None:
    with pytest.raises(ValidationError, match="daily ingest quota"):
        settings(
            knowledge_max_source_bytes=50_000_000,
            knowledge_daily_ingest_bytes_per_user=25_000_000,
        )
    with pytest.raises(ValidationError, match="storage quota"):
        settings(
            knowledge_daily_ingest_bytes_per_user=50_000_000,
            knowledge_storage_quota_bytes_per_user=25_000_000,
        )
    with pytest.raises(ValidationError, match="dedicated absolute POSIX path"):
        settings(knowledge_asset_root="/data")
    with pytest.raises(ValidationError, match="dedicated absolute POSIX path"):
        settings(knowledge_asset_root="../knowledge")
    with pytest.raises(ValidationError, match="exactly one runner"):
        settings(knowledge_runner_concurrency=2)


async def test_sqlite_runtime_enforces_wal_busy_timeout_and_foreign_keys(tmp_path: Path) -> None:
    database_path = tmp_path / "hardening.db"
    database = Database(
        f"sqlite+aiosqlite:///{database_path}",
        sqlite_busy_timeout_ms=12_345,
        sqlite_wal_enabled=True,
    )
    try:
        async with database.sessions() as session:
            assert str(await session.scalar(text("PRAGMA journal_mode"))).casefold() == "wal"
            assert await session.scalar(text("PRAGMA busy_timeout")) == 12_345
            assert await session.scalar(text("PRAGMA foreign_keys")) == 1
    finally:
        await database.dispose()


def test_safe_subprocess_environment_never_inherits_provider_or_bot_secrets() -> None:
    environment = sanitized_environment(
        {
            "PATH": "/safe/bin",
            "LANG": "C.UTF-8",
            "AI_API_KEY": "private-ai-key",
            "TELEGRAM_BOT_TOKEN": "private-bot-token",
            "DATABASE_URL": "private-database-url",
            "LD_LIBRARY_PATH": "/private/library/injection",
        }
    )
    assert environment["PATH"] == "/safe/bin"
    assert environment["LANG"] == "C.UTF-8"
    assert "AI_API_KEY" not in environment
    assert "TELEGRAM_BOT_TOKEN" not in environment
    assert "DATABASE_URL" not in environment
    assert "LD_LIBRARY_PATH" not in environment
    assert "private" not in repr(environment)
    assert sanitized_environment({}) == {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def test_safe_subprocess_uses_fixed_module_no_shell_and_sanitized_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["arguments"] = arguments
        captured.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("future_self.safe_media.subprocess.subprocess.run", fake_run)
    monkeypatch.setenv("AI_API_KEY", "must-not-cross-boundary")
    result = run_isolated_python_module(
        "future_self.safe_media.pdf_worker",
        ("input.pdf", "pages"),
        cwd=tmp_path,
        timeout_seconds=3,
    )
    assert result.returncode == 0
    assert captured["shell"] is False
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.DEVNULL
    assert "AI_API_KEY" not in captured["env"]
    assert captured["env"]["TEMP"] == str(tmp_path.resolve())
    assert captured["env"]["TMP"] == str(tmp_path.resolve())
    assert captured["env"]["TMPDIR"] == str(tmp_path.resolve())
    assert "must-not-cross-boundary" not in repr(captured)
    with pytest.raises(SafeSubprocessError, match="invalid_worker"):
        run_isolated_python_module(
            "os",
            (),
            cwd=tmp_path,
            timeout_seconds=3,
        )
    with pytest.raises(SafeSubprocessError, match="invalid_worker"):
        run_isolated_python_module(
            "future_self.safe_media..pdf_worker",
            (),
            cwd=tmp_path,
            timeout_seconds=3,
        )


def test_private_file_and_resolved_root_reject_symlink_and_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(SafeSubprocessError, match="unsafe_temporary_storage"):
        ensure_child(root, outside / "asset")

    target = root / "target"
    target.write_bytes(b"existing")
    symlink = root / "link"
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not available")
    with pytest.raises(SafeSubprocessError, match="unsafe_output_path"):
        write_private_file(symlink, b"replacement")
    assert target.read_bytes() == b"existing"

    directory_link = tmp_path / "work-link"
    try:
        directory_link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is not available")
    with pytest.raises(SafeSubprocessError, match="unsafe_work_directory"):
        run_isolated_python_module(
            "future_self.safe_media.pdf_worker",
            (),
            cwd=directory_link,
            timeout_seconds=3,
        )


def test_public_safe_pdf_boundary_accepts_plain_and_rejects_active_content(
    tmp_path: Path,
) -> None:
    pages = render_pdf_pages(pdf_bytes(), temp_root=tmp_path)
    assert len(pages) == 1
    assert pages[0].mime_type == "image/jpeg"
    assert pages[0].image_bytes.startswith(b"\xff\xd8\xff")
    with pytest.raises(SafePdfError, match="unsafe_or_unrenderable_pdf"):
        render_pdf_pages(pdf_bytes(javascript=True), temp_root=tmp_path)


def test_container_and_build_context_are_hardened() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    runbook = (root / "docs/operations/production-hardening.md").read_text(encoding="utf-8")

    assert "USER 10001:10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "--no-create-home" in dockerfile
    assert dockerignore[0] == "*"
    assert not any(
        line in {"!.env", "!data", "!data/**", "!.git", "!.git/**"} for line in dockerignore
    )
    assert "127.0.0.1:5432:5432" in compose
    for control in (
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges:true",
        "--pids-limit 128",
        "--log-opt max-size=10m",
        "/data/backups,readonly",
    ):
        assert control in runbook


def test_pr22_adds_no_schema_revision_or_future_domain_models() -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    assert ScriptDirectory.from_config(config).get_current_head() == "20260720_0017"
    model_source = (root / "src/future_self/models.py").read_text(encoding="utf-8")
    for future_model in (
        "Workspace",
        "KnowledgeSpace",
        "KnowledgeSource",
        "KnowledgeChunk",
        "KnowledgeEmbedding",
        "CouncilSession",
    ):
        assert f"class {future_model}" not in model_source
