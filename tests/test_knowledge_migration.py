from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

KNOWLEDGE_TABLES = {
    "knowledge_runtime_state",
    "knowledge_sources",
    "knowledge_source_revisions",
    "knowledge_ingestion_jobs",
    "knowledge_capture_drafts",
    "knowledge_action_tokens",
    "knowledge_quota_reservations",
    "knowledge_audit_events",
}


def alembic(
    project_root: Path,
    environment: dict[str, str],
    command: str,
    revision: str | None = None,
) -> None:
    arguments = [sys.executable, "-m", "alembic", command]
    if revision is not None:
        arguments.append(revision)
    subprocess.run(
        arguments,
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _environment(database: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    return environment


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _insert_user(connection: sqlite3.Connection, telegram_id: int) -> int:
    return int(
        connection.execute(
            """
            INSERT INTO users (telegram_id, timezone, onboarding_completed)
            VALUES (?, 'Europe/Moscow', 0)
            """,
            (telegram_id,),
        ).lastrowid
    )


def _insert_personal_space(
    connection: sqlite3.Connection,
    user_id: int,
    *,
    public_id: str | None = None,
) -> int:
    if public_id is None:
        statement = """
            INSERT INTO knowledge_spaces (
                kind, personal_owner_user_id, status, version
            ) VALUES ('personal', ?, 'active', 1)
        """
        parameters: tuple[object, ...] = (user_id,)
    else:
        statement = """
            INSERT INTO knowledge_spaces (
                public_id, kind, personal_owner_user_id, status, version
            ) VALUES (?, 'personal', ?, 'active', 1)
        """
        parameters = (public_id, user_id)
    return int(connection.execute(statement, parameters).lastrowid)


def _insert_source(
    connection: sqlite3.Connection,
    *,
    space_id: int,
    space_kind: str,
    user_id: int,
    classification: str = "general",
    publication_state: str = "draft",
) -> int:
    return int(
        connection.execute(
            """
            INSERT INTO knowledge_sources (
                public_id, knowledge_space_id, space_kind, created_by_user_id,
                source_type, title, provenance_kind, processing_status,
                lifecycle_status, knowledge_role, priority, publication_state,
                system_classification, current_revision_number, version
            ) VALUES (?, ?, ?, ?, 'text', 'Safe title', 'telegram_text', 'queued',
                      'active', 'trusted', 'normal', ?, ?, 1, 1)
            """,
            (
                str(uuid4()),
                space_id,
                space_kind,
                user_id,
                publication_state,
                classification,
            ),
        ).lastrowid
    )


def _insert_revision(
    connection: sqlite3.Connection,
    *,
    source_id: int,
    space_id: int,
    user_id: int,
    digest: str,
    storage_key: str,
) -> int:
    return int(
        connection.execute(
            """
            INSERT INTO knowledge_source_revisions (
                public_id, source_id, knowledge_space_id, revision_number,
                sha256, original_storage_key, detected_mime, detected_format,
                safe_display_name, size_bytes, extraction_status,
                created_by_user_id
            ) VALUES (?, ?, ?, 1, ?, ?, 'text/plain', 'text',
                      'source.txt', 4, 'pending', ?)
            """,
            (str(uuid4()), source_id, space_id, digest, storage_key, user_id),
        ).lastrowid
    )


def test_knowledge_migration_is_additive_and_downgrade_never_touches_files(tmp_path):
    project_root = Path(__file__).parents[1]
    database = tmp_path / "production-like.db"
    environment = _environment(database)
    retained_original = tmp_path / "data" / "knowledge" / "originals" / "keep.bin"
    retained_original.parent.mkdir(parents=True)
    retained_original.write_bytes(b"must survive database downgrade")

    alembic(project_root, environment, "upgrade", "20260722_0018")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    first_user_id = _insert_user(connection, 9_240_001)
    second_user_id = _insert_user(connection, 9_240_002)
    legacy_space_id = _insert_personal_space(connection, first_user_id)
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    alembic(project_root, environment, "check")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260722_0019"
    )
    assert KNOWLEDGE_TABLES <= _tables(connection)
    assert connection.execute(
        "SELECT id, maintenance_paused, version FROM knowledge_runtime_state"
    ).fetchall() == [(1, 0, 1)]
    public_id_column = next(
        row
        for row in connection.execute("PRAGMA table_info(knowledge_spaces)")
        if row[1] == "public_id"
    )
    assert public_id_column[3] == 0
    assert connection.execute(
        "SELECT public_id FROM knowledge_spaces WHERE id = ?", (legacy_space_id,)
    ).fetchone() == (None,)

    # A PR #23 rollback image can still create a space without knowing public_id.
    rollback_space_id = _insert_personal_space(connection, second_user_id)
    connection.commit()
    assert rollback_space_id != legacy_space_id
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "downgrade", "20260722_0018")
    connection = sqlite3.connect(database)
    assert not (KNOWLEDGE_TABLES & _tables(connection))
    assert "public_id" not in {
        row[1] for row in connection.execute("PRAGMA table_info(knowledge_spaces)")
    }
    assert connection.execute("SELECT COUNT(*) FROM knowledge_spaces").fetchone()[0] == 2
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
    assert retained_original.read_bytes() == b"must survive database downgrade"

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM knowledge_runtime_state").fetchone()[0] == 1
    assert all(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        for table in KNOWLEDGE_TABLES - {"knowledge_runtime_state"}
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_knowledge_constraints_isolate_spaces_without_global_sha_deduplication(tmp_path):
    project_root = Path(__file__).parents[1]
    database = tmp_path / "constraints.db"
    environment = _environment(database)
    alembic(project_root, environment, "upgrade", "head")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    first_user_id = _insert_user(connection, 9_241_001)
    second_user_id = _insert_user(connection, 9_241_002)
    first_space_id = _insert_personal_space(connection, first_user_id, public_id=str(uuid4()))
    second_space_id = _insert_personal_space(connection, second_user_id, public_id=str(uuid4()))
    workspace_id = int(
        connection.execute(
            """
            INSERT INTO workspaces (
                name, normalized_name, character, created_by_user_id,
                status, access_epoch, version
            ) VALUES ('Shared', 'shared', 'pair', ?, 'active', 1, 1)
            """,
            (first_user_id,),
        ).lastrowid
    )
    connection.execute(
        """
        INSERT INTO workspace_members (
            workspace_id, user_id, role, status, invited_by_user_id, version
        ) VALUES (?, ?, 'owner', 'active', ?, 1)
        """,
        (workspace_id, first_user_id, first_user_id),
    )
    shared_space_id = int(
        connection.execute(
            """
            INSERT INTO knowledge_spaces (
                public_id, kind, workspace_id, status, version
            ) VALUES (?, 'workspace', ?, 'active', 1)
            """,
            (str(uuid4()), workspace_id),
        ).lastrowid
    )

    digest = "a" * 64
    first_source_id = _insert_source(
        connection,
        space_id=first_space_id,
        space_kind="personal",
        user_id=first_user_id,
    )
    _insert_revision(
        connection,
        source_id=first_source_id,
        space_id=first_space_id,
        user_id=first_user_id,
        digest=digest,
        storage_key="originals/first/source.txt",
    )
    second_source_id = _insert_source(
        connection,
        space_id=second_space_id,
        space_kind="personal",
        user_id=second_user_id,
    )
    _insert_revision(
        connection,
        source_id=second_source_id,
        space_id=second_space_id,
        user_id=second_user_id,
        digest=digest,
        storage_key="originals/second/source.txt",
    )
    connection.commit()
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM knowledge_source_revisions WHERE sha256 = ?", (digest,)
        ).fetchone()[0]
        == 2
    )

    with pytest.raises(sqlite3.IntegrityError):
        _insert_revision(
            connection,
            source_id=first_source_id,
            space_id=second_space_id,
            user_id=first_user_id,
            digest="b" * 64,
            storage_key="originals/cross-space/source.txt",
        )
    connection.rollback()

    with pytest.raises(sqlite3.IntegrityError):
        _insert_source(
            connection,
            space_id=shared_space_id,
            space_kind="workspace",
            user_id=first_user_id,
            classification="health_private",
        )
    connection.rollback()

    with pytest.raises(sqlite3.IntegrityError):
        _insert_source(
            connection,
            space_id=first_space_id,
            space_kind="personal",
            user_id=first_user_id,
            classification="health_private",
            publication_state="publication_ready",
        )
    connection.rollback()

    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
