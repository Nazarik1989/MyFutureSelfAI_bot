import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from future_self.knowledge_backup import (
    KnowledgeBackupError,
    create_backup,
    maintenance_paused,
    recover_maintenance,
    verify_backup,
)


def _write_private_asset(root: Path, key: str, payload: bytes) -> Path:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    current = root
    for part in Path(*key.split("/")).parts[:-1]:
        current /= part
        current.mkdir(mode=0o700, exist_ok=True)
        os.chmod(current, 0o700)
    target = root / Path(*key.split("/"))
    target.write_bytes(payload)
    os.chmod(target, 0o600)
    return target


def _database(path: Path, storage_key: str, payload: bytes) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
            INSERT INTO alembic_version VALUES ('20260722_0019');
            CREATE TABLE knowledge_source_revisions (
                id INTEGER PRIMARY KEY,
                original_revision_id INTEGER,
                original_storage_key TEXT,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                extracted_storage_key TEXT,
                extracted_size_bytes INTEGER,
                extracted_sha256 TEXT
            );
            CREATE TABLE knowledge_runtime_state (
                id INTEGER PRIMARY KEY,
                maintenance_paused INTEGER NOT NULL,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO knowledge_runtime_state (id, maintenance_paused, version)
            VALUES (1, 0, 1);
            """
        )
        connection.execute(
            "INSERT INTO knowledge_source_revisions VALUES (1, NULL, ?, ?, ?, NULL, NULL, NULL)",
            (storage_key, len(payload), hashlib.sha256(payload).hexdigest()),
        )


def test_consistent_backup_has_database_assets_and_privacy_safe_manifest(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"private bytes")
    _database(database, key, b"private bytes")

    target = tmp_path / "backup"
    result = create_backup(
        database_path=database,
        asset_root=assets,
        destination=target,
        application_sha="a" * 40,
    )

    assert result.asset_count == 1
    assert not maintenance_paused(assets)
    with sqlite3.connect(database) as live:
        assert (
            live.execute(
                "SELECT maintenance_paused FROM knowledge_runtime_state WHERE id = 1"
            ).fetchone()[0]
            == 0
        )
    verified = verify_backup(target)
    assert verified == {
        "format_version": 1,
        "alembic_head": "20260722_0019",
        "asset_count": 1,
    }
    manifest_text = (target / "manifest.json").read_text(encoding="ascii")
    manifest = json.loads(manifest_text)
    assert manifest["assets"][0]["storage_key"] == key
    assert "private bytes" not in manifest_text
    with sqlite3.connect(target / "database.sqlite3") as restored:
        assert (
            restored.execute(
                "SELECT maintenance_paused FROM knowledge_runtime_state WHERE id = 1"
            ).fetchone()[0]
            == 0
        )


def test_verifier_detects_missing_and_corrupt_assets(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")
    target = tmp_path / "backup"
    create_backup(
        database_path=database,
        asset_root=assets,
        destination=target,
        application_sha="local-test",
    )

    copied = target / "assets" / Path(*key.split("/"))
    copied.write_bytes(b"corrupt")
    with pytest.raises(KnowledgeBackupError, match="asset_checksum_mismatch"):
        verify_backup(target)
    copied.unlink()
    with pytest.raises((KnowledgeBackupError, FileNotFoundError)):
        verify_backup(target)


def test_backup_fails_closed_for_symlink_and_releases_pause(tmp_path: Path) -> None:
    assets = tmp_path / "knowledge"
    assets.mkdir(mode=0o700)
    os.chmod(assets, 0o700)
    database = tmp_path / "live.db"
    _database(database, "originals/ab/cd/0123456789abcdef0123456789abcdef", b"missing")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = assets / "link.bin"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(KnowledgeBackupError, match="unsafe_asset"):
        create_backup(
            database_path=database,
            asset_root=assets,
            destination=tmp_path / "backup",
            application_sha="local-test",
        )
    assert not maintenance_paused(assets)


def test_verifier_rejects_windows_and_posix_manifest_traversal(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")
    target = tmp_path / "backup"
    create_backup(
        database_path=database,
        asset_root=assets,
        destination=target,
        application_sha="local-test",
    )
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))

    for malicious in ("..\\outside", "../outside", "C:/outside"):
        manifest["assets"][0]["storage_key"] = malicious
        manifest_path.write_text(json.dumps(manifest), encoding="ascii")
        with pytest.raises(KnowledgeBackupError, match="unsafe_storage_key"):
            verify_backup(target)


def test_backup_rejects_asset_already_inconsistent_with_database(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"tampered")
    _database(database, key, b"expected")

    with pytest.raises(KnowledgeBackupError, match="database_asset_checksum_mismatch"):
        create_backup(
            database_path=database,
            asset_root=assets,
            destination=tmp_path / "backup",
            application_sha="local-test",
        )
    assert not maintenance_paused(assets)


def test_verifier_rejects_manifest_database_path_override(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")
    target = tmp_path / "backup"
    create_backup(
        database_path=database,
        asset_root=assets,
        destination=target,
        application_sha="local-test",
    )
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["database"]["path"] = "database.sqlite3?mode=memory"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    os.chmod(manifest_path, 0o600)

    with pytest.raises(KnowledgeBackupError, match="invalid_manifest"):
        verify_backup(target)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not Windows ACLs")
def test_verifier_rejects_public_manifest_permissions(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")
    target = tmp_path / "backup"
    create_backup(
        database_path=database,
        asset_root=assets,
        destination=target,
        application_sha="local-test",
    )
    os.chmod(target / "manifest.json", 0o644)

    with pytest.raises(KnowledgeBackupError, match="invalid_manifest"):
        verify_backup(target)


def test_backup_understands_retry_revision_reusing_immutable_original(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    payload = b"one immutable original"
    _write_private_asset(assets, key, payload)
    _database(database, key, payload)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO knowledge_source_revisions VALUES (2, 1, NULL, ?, ?, NULL, NULL, NULL)",
            (len(payload), hashlib.sha256(payload).hexdigest()),
        )

    target = tmp_path / "backup"
    result = create_backup(
        database_path=database,
        asset_root=assets,
        destination=target,
        application_sha="local-test",
    )

    assert result.asset_count == 1
    assert verify_backup(target)["asset_count"] == 1


def test_backup_does_not_wait_for_an_already_expired_lease(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE knowledge_ingestion_jobs (status TEXT, lease_expires_at TEXT)"
        )
        connection.execute(
            "INSERT INTO knowledge_ingestion_jobs VALUES ('processing', '2000-01-01 00:00:00')"
        )

    result = create_backup(
        database_path=database,
        asset_root=assets,
        destination=tmp_path / "backup",
        application_sha="local-test",
        wait_seconds=0,
    )

    assert result.asset_count == 1


@pytest.mark.parametrize(
    ("database_paused", "marker_present"),
    [(True, True), (True, False), (False, True)],
)
def test_maintenance_recovery_reconciles_both_partial_fences(
    tmp_path: Path,
    database_paused: bool,
    marker_present: bool,
) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE knowledge_runtime_state SET maintenance_paused = ? WHERE id = 1",
            (int(database_paused),),
        )
    if marker_present:
        marker = assets / ".knowledge-maintenance-pause"
        marker.write_text("pid=123\n", encoding="ascii")
        os.chmod(marker, 0o600)

    result = recover_maintenance(database_path=database, asset_root=assets)

    assert result.recovered
    assert result.database_was_paused is database_paused
    assert result.marker_was_present is marker_present
    assert not maintenance_paused(assets)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT maintenance_paused FROM knowledge_runtime_state WHERE id = 1"
        ).fetchone() == (0,)


def test_maintenance_recovery_retains_fences_until_live_lease_drains(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE knowledge_runtime_state SET maintenance_paused = 1 WHERE id = 1")
        connection.execute(
            "CREATE TABLE knowledge_ingestion_jobs (status TEXT, lease_expires_at TEXT)"
        )
        connection.execute(
            "INSERT INTO knowledge_ingestion_jobs VALUES ('processing', '2999-01-01 00:00:00')"
        )

    with pytest.raises(KnowledgeBackupError, match="active_runner_leases"):
        recover_maintenance(database_path=database, asset_root=assets)

    assert maintenance_paused(assets)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT maintenance_paused FROM knowledge_runtime_state WHERE id = 1"
        ).fetchone() == (1,)
        connection.execute(
            "UPDATE knowledge_ingestion_jobs SET lease_expires_at = '2000-01-01 00:00:00'"
        )

    result = recover_maintenance(database_path=database, asset_root=assets)
    assert result.recovered
    assert not maintenance_paused(assets)


def test_maintenance_recovery_is_idempotent_when_no_fence_exists(tmp_path: Path) -> None:
    database = tmp_path / "live.db"
    assets = tmp_path / "knowledge"
    key = "originals/ab/cd/0123456789abcdef0123456789abcdef"
    _write_private_asset(assets, key, b"original")
    _database(database, key, b"original")

    result = recover_maintenance(database_path=database, asset_root=assets)

    assert not result.recovered
    assert not maintenance_paused(assets)
