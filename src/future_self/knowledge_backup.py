"""Consistent, privacy-safe backup and verification for Knowledge assets.

The CLI intentionally accepts an SQLite path instead of loading bot settings.  It can
therefore run without Telegram, AI, or transcription credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_VERSION = 1
_PAUSE_FILE = ".knowledge-maintenance-pause"
_EXCLUDED_TOP_LEVEL = {".staging", ".runner-tmp", _PAUSE_FILE}
_APPLICATION_SHA = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_STORAGE_KEY = re.compile(
    r"^(?:originals/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{32}|"
    r"extracted/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{32}\.txt)$"
)
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class KnowledgeBackupError(RuntimeError):
    """A safe backup/verification failure without document contents."""


@dataclass(frozen=True)
class BackupResult:
    path: Path
    manifest_path: Path
    asset_count: int


@dataclass(frozen=True)
class MaintenanceRecoveryResult:
    database_was_paused: bool
    marker_was_present: bool

    @property
    def recovered(self) -> bool:
        return self.database_was_paused or self.marker_was_present


@dataclass(frozen=True)
class _PauseHandle:
    path: Path
    descriptor: int


def _is_private_mode(mode: int) -> bool:
    """POSIX permission bits are not meaningful for Windows ACL-backed paths."""

    return os.name == "nt" or mode & 0o077 == 0


def _assert_private_directory(path: Path, *, code: str) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise KnowledgeBackupError(code) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or not _is_private_mode(current.st_mode)
    ):
        raise KnowledgeBackupError(code)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or not _is_private_mode(current.st_mode)
        ):
            raise KnowledgeBackupError("unsafe_asset")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return size, digest.hexdigest()


def _read_private_manifest(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise KnowledgeBackupError("invalid_manifest") from exc
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_size <= 0
            or current.st_size > _MAX_MANIFEST_BYTES
            or not _is_private_mode(current.st_mode)
        ):
            raise KnowledgeBackupError("invalid_manifest")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > _MAX_MANIFEST_BYTES:
                raise KnowledgeBackupError("invalid_manifest")
            chunks.append(chunk)
        return b"".join(chunks).decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise KnowledgeBackupError("invalid_manifest") from exc
    finally:
        os.close(descriptor)


def _safe_relative(value: str) -> Path:
    if not value or "\\" in value or "\x00" in value or ":" in value:
        raise KnowledgeBackupError("unsafe_storage_key")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise KnowledgeBackupError("unsafe_storage_key")
    return Path(*pure.parts)


def _safe_storage_key(value: str) -> Path:
    if not _STORAGE_KEY.fullmatch(value):
        raise KnowledgeBackupError("unsafe_storage_key")
    return _safe_relative(value)


def _asset_files(root: Path) -> list[tuple[str, Path]]:
    _assert_private_directory(root, code="unsafe_asset_root")
    found: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in _EXCLUDED_TOP_LEVEL:
            continue
        if path.is_symlink():
            raise KnowledgeBackupError("unsafe_asset")
        if path.is_dir():
            _assert_private_directory(path, code="unsafe_asset")
        if path.is_file():
            key = PurePosixPath(*relative.parts).as_posix()
            _safe_storage_key(key)
            found.append((key, path))
    return sorted(found)


def _copy_private(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    output = os.open(destination, flags, 0o600)
    digest = hashlib.sha256()
    size = 0
    try:
        input_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        input_descriptor = os.open(source, input_flags)
        try:
            current = os.fstat(input_descriptor)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or not _is_private_mode(current.st_mode)
            ):
                raise KnowledgeBackupError("unsafe_asset")
            while chunk := os.read(input_descriptor, 1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(output, view)
                    view = view[written:]
        finally:
            os.close(input_descriptor)
        os.fsync(output)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(output)
    return size, digest.hexdigest()


def _alembic_head(connection: sqlite3.Connection) -> str:
    try:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    except sqlite3.DatabaseError as exc:
        raise KnowledgeBackupError("missing_alembic_version") from exc
    if row is None or not row[0]:
        raise KnowledgeBackupError("missing_alembic_version")
    return str(row[0])


def _active_runner_jobs(connection: sqlite3.Connection) -> int:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_ingestion_jobs'"
    ).fetchone()
    if table is None:
        return 0
    return int(
        connection.execute(
            "SELECT count(*) FROM knowledge_ingestion_jobs "
            "WHERE status = 'processing' AND lease_expires_at > CURRENT_TIMESTAMP"
        ).fetchone()[0]
    )


def _set_database_maintenance(database: Path, *, paused: bool) -> None:
    """Serialize the maintenance fence with every SQLite Knowledge writer."""

    try:
        with closing(sqlite3.connect(database, timeout=30, isolation_level=None)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE knowledge_runtime_state "
                "SET maintenance_paused = ?, version = version + 1, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = 1",
                (1 if paused else 0,),
            )
            if changed.rowcount != 1:
                connection.rollback()
                raise KnowledgeBackupError("maintenance_state_missing")
            connection.commit()
    except KnowledgeBackupError:
        raise
    except sqlite3.DatabaseError as exc:
        raise KnowledgeBackupError("maintenance_state_unavailable") from exc


def _lock_pause_descriptor(descriptor: int) -> None:
    if os.name != "posix":
        return
    try:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise KnowledgeBackupError("maintenance_owner_active") from exc
    except OSError as exc:
        raise KnowledgeBackupError("unsafe_maintenance_marker") from exc


def _validate_pause_descriptor(descriptor: int) -> os.stat_result:
    try:
        current = os.fstat(descriptor)
    except OSError as exc:
        raise KnowledgeBackupError("unsafe_maintenance_marker") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or current.st_size <= 0
        or current.st_size > 256
        or not _is_private_mode(current.st_mode)
    ):
        raise KnowledgeBackupError("unsafe_maintenance_marker")
    return current


def _acquire_pause(root: Path) -> _PauseHandle:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise KnowledgeBackupError("unsafe_asset_root")
    pause = root / _PAUSE_FILE
    descriptor = -1
    try:
        descriptor = os.open(
            pause,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise KnowledgeBackupError("maintenance_already_active") from exc
    try:
        _lock_pause_descriptor(descriptor)
        payload = f"pid={os.getpid()}\n".encode()
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        _validate_pause_descriptor(descriptor)
        _fsync_directory(root)
        return _PauseHandle(pause, descriptor)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        pause.unlink(missing_ok=True)
        raise


def _open_pause(root: Path) -> _PauseHandle:
    pause = root / _PAUSE_FILE
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(pause, flags)
    except OSError as exc:
        raise KnowledgeBackupError("unsafe_maintenance_marker") from exc
    try:
        _validate_pause_descriptor(descriptor)
        _lock_pause_descriptor(descriptor)
        return _PauseHandle(pause, descriptor)
    except Exception:
        os.close(descriptor)
        raise


def _release_pause(handle: _PauseHandle, *, remove: bool) -> None:
    closed = False
    try:
        if not remove:
            return
        try:
            opened = os.fstat(handle.descriptor)
            current = handle.path.lstat()
        except OSError as exc:
            raise KnowledgeBackupError("maintenance_marker_changed") from exc
        if not os.path.samestat(opened, current) or stat.S_ISLNK(current.st_mode):
            raise KnowledgeBackupError("maintenance_marker_changed")
        try:
            # Windows does not permit unlinking an open file. Production POSIX
            # keeps the advisory lock through unlink; Windows still performs the
            # same inode validation immediately before closing and removing it.
            if os.name == "nt":
                os.close(handle.descriptor)
                closed = True
            handle.path.unlink()
            _fsync_directory(handle.path.parent)
        except OSError as exc:
            raise KnowledgeBackupError("maintenance_marker_release_failed") from exc
    finally:
        if not closed:
            os.close(handle.descriptor)


def maintenance_paused(asset_root: str | Path) -> bool:
    """Return true when capture/runner leasing must remain paused."""

    return os.path.lexists(Path(asset_root) / _PAUSE_FILE)


def _database_maintenance_paused(database: Path) -> bool:
    try:
        with closing(sqlite3.connect(database, timeout=5)) as connection:
            row = connection.execute(
                "SELECT maintenance_paused FROM knowledge_runtime_state WHERE id = 1"
            ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise KnowledgeBackupError("maintenance_state_unavailable") from exc
    if row is None or row[0] not in (0, 1):
        raise KnowledgeBackupError("maintenance_state_missing")
    return bool(row[0])


def _clear_database_maintenance(database: Path) -> bool:
    """Clear the DB fence only while an exclusive writer lock proves leases drained."""

    try:
        with closing(sqlite3.connect(database, timeout=30, isolation_level=None)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT maintenance_paused FROM knowledge_runtime_state WHERE id = 1"
                ).fetchone()
                if row is None or row[0] not in (0, 1):
                    raise KnowledgeBackupError("maintenance_state_missing")
                if _active_runner_jobs(connection):
                    raise KnowledgeBackupError("active_runner_leases")
                was_paused = bool(row[0])
                if was_paused:
                    changed = connection.execute(
                        "UPDATE knowledge_runtime_state "
                        "SET maintenance_paused = 0, version = version + 1, "
                        "updated_at = CURRENT_TIMESTAMP WHERE id = 1 AND maintenance_paused = 1"
                    )
                    if changed.rowcount != 1:
                        raise KnowledgeBackupError("maintenance_state_changed")
                connection.commit()
                return was_paused
            except Exception:
                connection.rollback()
                raise
    except KnowledgeBackupError:
        raise
    except sqlite3.DatabaseError as exc:
        raise KnowledgeBackupError("maintenance_recovery_failed") from exc


def recover_maintenance(
    *,
    database_path: str | Path,
    asset_root: str | Path,
) -> MaintenanceRecoveryResult:
    """Safely reconcile a stale filesystem marker and SQLite maintenance fence.

    The marker stays present until a serialized SQLite transaction proves that no
    live processing lease exists and clears the database fence.  Any ambiguous
    failure retains the marker so leasing cannot resume on uncertain state.
    """

    database_argument = Path(database_path)
    assets_argument = Path(asset_root)
    if database_argument.is_symlink() or assets_argument.is_symlink():
        raise KnowledgeBackupError("unsafe_source_path")
    try:
        database = database_argument.resolve(strict=True)
        assets = assets_argument.resolve(strict=True)
    except OSError as exc:
        raise KnowledgeBackupError("unsafe_source_path") from exc
    if not database.is_file() or database.is_symlink():
        raise KnowledgeBackupError("unsafe_database")
    _assert_private_directory(assets, code="unsafe_asset_root")

    database_was_paused = _database_maintenance_paused(database)
    marker_was_present = maintenance_paused(assets)
    if not database_was_paused and not marker_was_present:
        return MaintenanceRecoveryResult(False, False)

    if marker_was_present:
        pause = _open_pause(assets)
    else:
        try:
            pause = _acquire_pause(assets)
        except KnowledgeBackupError as exc:
            if str(exc) != "maintenance_already_active":
                raise
            pause = _open_pause(assets)
            marker_was_present = True
    try:
        database_was_paused = _clear_database_maintenance(database)
    except Exception:
        _release_pause(pause, remove=False)
        raise
    _release_pause(pause, remove=True)
    return MaintenanceRecoveryResult(database_was_paused, marker_was_present)


def create_backup(
    *,
    database_path: str | Path,
    asset_root: str | Path,
    destination: str | Path,
    application_sha: str,
    wait_seconds: float = 30.0,
) -> BackupResult:
    database_argument = Path(database_path)
    assets_argument = Path(asset_root)
    if database_argument.is_symlink() or assets_argument.is_symlink():
        raise KnowledgeBackupError("unsafe_source_path")
    database = database_argument.resolve(strict=True)
    assets = assets_argument.resolve(strict=True)
    target = Path(destination).resolve(strict=False)
    if not database.is_file() or database.is_symlink():
        raise KnowledgeBackupError("unsafe_database")
    if target.exists():
        raise KnowledgeBackupError("backup_destination_exists")
    if not _APPLICATION_SHA.fullmatch(application_sha.strip()):
        raise KnowledgeBackupError("invalid_application_sha")
    if target == assets or target.is_relative_to(assets):
        raise KnowledgeBackupError("backup_inside_asset_root")

    pause = _acquire_pause(assets)
    database_paused = False
    temporary = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
    try:
        _set_database_maintenance(database, paused=True)
        database_paused = True
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            with closing(sqlite3.connect(database, timeout=5)) as live:
                active = _active_runner_jobs(live)
            if active == 0:
                break
            if time.monotonic() >= deadline:
                raise KnowledgeBackupError("runner_drain_timeout")
            time.sleep(0.2)

        temporary.mkdir(mode=0o700, parents=True)
        snapshot = temporary / "database.sqlite3"
        with (
            closing(sqlite3.connect(database, timeout=30)) as source,
            closing(sqlite3.connect(snapshot)) as dest,
        ):
            if source.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise KnowledgeBackupError("database_integrity_failed")
            head = _alembic_head(source)
            source.backup(dest)
            changed = dest.execute(
                "UPDATE knowledge_runtime_state "
                "SET maintenance_paused = 0, version = version + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = 1"
            )
            if changed.rowcount != 1:
                raise KnowledgeBackupError("maintenance_state_missing")
            dest.commit()
        os.chmod(snapshot, 0o600)
        database_size, database_hash = _hash_file(snapshot)

        asset_entries: list[dict[str, Any]] = []
        for key, source_path in _asset_files(assets):
            destination_path = temporary / "assets" / _safe_storage_key(key)
            size, digest = _copy_private(source_path, destination_path)
            asset_entries.append({"storage_key": key, "size_bytes": size, "sha256": digest})

        manifest: dict[str, Any] = {
            "format": "myfutureself-knowledge-backup",
            "format_version": MANIFEST_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "application_sha": application_sha.strip(),
            "alembic_head": head,
            "database": {
                "path": "database.sqlite3",
                "size_bytes": database_size,
                "sha256": database_hash,
            },
            "assets": asset_entries,
        }
        manifest_path = temporary / "manifest.json"
        payload = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True).encode() + b"\n"
        descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_tree_directories(temporary)
        verify_backup(temporary)
        # The destination is required to be absent, so rename is atomic on the
        # same filesystem and also works for directories on Windows.
        temporary.rename(target)
        _fsync_directory(target.parent)
        return BackupResult(target, target / "manifest.json", len(asset_entries))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if not database_paused:
            # Setting the DB fence may have committed before an error surfaced.
            # Preserve the marker so the explicit recovery command can reconcile it.
            _release_pause(pause, remove=False)
        else:
            try:
                _set_database_maintenance(database, paused=False)
            except KnowledgeBackupError as exc:
                # Keep the marker so neither process resumes on an ambiguous DB state.
                _release_pause(pause, remove=False)
                raise KnowledgeBackupError("maintenance_resume_failed") from exc
            _release_pause(pause, remove=True)


def _referenced_storage_assets(connection: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(knowledge_source_revisions)").fetchall()
    }
    required = {
        "id",
        "original_revision_id",
        "original_storage_key",
        "size_bytes",
        "sha256",
        "extracted_storage_key",
        "extracted_size_bytes",
        "extracted_sha256",
    }
    if not required.issubset(columns):
        raise KnowledgeBackupError("knowledge_schema_missing")
    rows = connection.execute(
        "SELECT id, original_revision_id, original_storage_key, size_bytes, sha256, "
        "extracted_storage_key, extracted_size_bytes, extracted_sha256 "
        "FROM knowledge_source_revisions"
    ).fetchall()
    expected: dict[str, tuple[int, str]] = {}
    originals = {
        int(revision_id): (str(original_key), int(size), str(digest))
        for revision_id, _original_id, original_key, size, digest, *_ in rows
        if original_key is not None
    }
    for (
        _revision_id,
        original_revision_id,
        original_key,
        size,
        digest,
        extracted_key,
        extracted_size,
        extracted_digest,
    ) in rows:
        if original_key is None:
            original = originals.get(int(original_revision_id or 0))
            if original is None or original[1:] != (size, digest):
                raise KnowledgeBackupError("invalid_database_asset_tuple")
            original_key = original[0]
        pairs = ((original_key, size, digest),)
        if extracted_key is None:
            if extracted_size is not None or extracted_digest is not None:
                raise KnowledgeBackupError("invalid_database_asset_tuple")
        else:
            pairs += ((extracted_key, extracted_size, extracted_digest),)
        for key, expected_size, expected_hash in pairs:
            clean_key = str(key)
            _safe_storage_key(clean_key)
            if (
                not isinstance(expected_size, int)
                or expected_size < 0
                or not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            ):
                raise KnowledgeBackupError("invalid_database_asset_tuple")
            value = (expected_size, expected_hash)
            if clean_key in expected and expected[clean_key] != value:
                raise KnowledgeBackupError("conflicting_database_asset_reference")
            expected[clean_key] = value
    return expected


def verify_backup(path: str | Path) -> dict[str, Any]:
    root_argument = Path(path)
    if root_argument.is_symlink():
        raise KnowledgeBackupError("unsafe_backup_root")
    root = root_argument.resolve(strict=True)
    _assert_private_directory(root, code="unsafe_backup_root")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise KnowledgeBackupError("invalid_manifest")
    try:
        manifest = json.loads(_read_private_manifest(manifest_path))
    except (OSError, ValueError, UnicodeError) as exc:
        raise KnowledgeBackupError("invalid_manifest") from exc
    if (
        manifest.get("format") != "myfutureself-knowledge-backup"
        or manifest.get("format_version") != MANIFEST_VERSION
    ):
        raise KnowledgeBackupError("unsupported_manifest")

    database_meta = manifest.get("database")
    if not isinstance(database_meta, dict):
        raise KnowledgeBackupError("invalid_manifest")
    if database_meta.get("path") != "database.sqlite3":
        raise KnowledgeBackupError("invalid_manifest")
    database_path = root / "database.sqlite3"
    size, digest = _hash_file(database_path)
    if size != database_meta.get("size_bytes") or digest != database_meta.get("sha256"):
        raise KnowledgeBackupError("database_checksum_mismatch")
    with closing(sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise KnowledgeBackupError("database_integrity_failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise KnowledgeBackupError("database_foreign_key_failed")
        if _alembic_head(connection) != manifest.get("alembic_head"):
            raise KnowledgeBackupError("alembic_head_mismatch")
        referenced = _referenced_storage_assets(connection)

    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise KnowledgeBackupError("invalid_manifest")
    seen: set[str] = set()
    for entry in assets:
        if not isinstance(entry, dict):
            raise KnowledgeBackupError("invalid_manifest")
        key = str(entry.get("storage_key", ""))
        if key in seen:
            raise KnowledgeBackupError("duplicate_manifest_key")
        seen.add(key)
        asset_path = root / "assets" / _safe_storage_key(key)
        size, digest = _hash_file(asset_path)
        if size != entry.get("size_bytes") or digest != entry.get("sha256"):
            raise KnowledgeBackupError("asset_checksum_mismatch")
        if referenced.get(key) != (size, digest):
            raise KnowledgeBackupError("database_asset_checksum_mismatch")
    actual = (
        {key for key, _ in _asset_files(root / "assets")} if (root / "assets").exists() else set()
    )
    if actual != seen:
        raise KnowledgeBackupError("asset_manifest_mismatch")
    referenced_keys = set(referenced)
    if referenced_keys != seen:
        code = (
            "database_asset_reference_missing"
            if referenced_keys - seen
            else "unreferenced_backup_asset"
        )
        raise KnowledgeBackupError(code)
    return {
        "format_version": MANIFEST_VERSION,
        "alembic_head": manifest["alembic_head"],
        "asset_count": len(seen),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge backup and offline verifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--database", required=True)
    create.add_argument("--assets", required=True)
    create.add_argument("--destination", required=True)
    create.add_argument("--application-sha", required=True)
    create.add_argument("--wait-seconds", type=float, default=30.0)
    verify = subparsers.add_parser("verify")
    verify.add_argument("backup")
    recover = subparsers.add_parser("recover-maintenance")
    recover.add_argument("--database", required=True)
    recover.add_argument("--assets", required=True)
    return parser


def main() -> None:
    os.umask(0o077)
    arguments = _parser().parse_args()
    try:
        if arguments.command == "create":
            result = create_backup(
                database_path=arguments.database,
                asset_root=arguments.assets,
                destination=arguments.destination,
                application_sha=arguments.application_sha,
                wait_seconds=arguments.wait_seconds,
            )
            print(json.dumps({"status": "ok", "assets": result.asset_count}))
        elif arguments.command == "verify":
            result = verify_backup(arguments.backup)
            print(json.dumps({"status": "ok", **result}, sort_keys=True))
        else:
            recovery = recover_maintenance(
                database_path=arguments.database,
                asset_root=arguments.assets,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "recovered": recovery.recovered,
                        "database_fence": recovery.database_was_paused,
                        "marker_fence": recovery.marker_was_present,
                    },
                    sort_keys=True,
                )
            )
    except KnowledgeBackupError as exc:
        raise SystemExit(f"Knowledge backup failed: {exc}") from None


if __name__ == "__main__":
    main()
