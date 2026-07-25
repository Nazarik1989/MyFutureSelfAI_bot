from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE_TABLES = {
    "workspaces",
    "workspace_members",
    "workspace_invitations",
    "workspace_projects",
    "knowledge_spaces",
    "workspace_contexts",
    "workspace_action_tokens",
}


def alembic(project_root: Path, environment: dict[str, str], command: str, revision: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", command, revision],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_workspace_migration_is_additive_constrained_and_round_trips(tmp_path):
    project_root = Path(__file__).parents[1]
    database = tmp_path / "production-like.db"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    alembic(project_root, environment, "upgrade", "20260720_0017")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (9_230_001, "Europe/Moscow", 0),
    ).lastrowid
    intended_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (9_230_002, "Europe/Moscow", 0),
    ).lastrowid
    sentinel_id = connection.execute(
        """
        INSERT INTO inbox_items (user_id, kind, title, raw_text, source, status)
        VALUES (?, 'note', 'Sentinel', 'Sentinel', 'text', 'confirmed')
        """,
        (owner_id,),
    ).lastrowid
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260722_0019"
    )
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert WORKSPACE_TABLES <= tables
    assert all(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        for table in WORKSPACE_TABLES
    )

    workspace_id = connection.execute(
        """
        INSERT INTO workspaces (
            name, normalized_name, character, created_by_user_id,
            status, access_epoch, version
        ) VALUES ('Future', 'future', 'pair', ?, 'active', 1, 1)
        """,
        (owner_id,),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO workspace_members (
            workspace_id, user_id, role, status, invited_by_user_id, version
        ) VALUES (?, ?, 'owner', 'active', ?, 1)
        """,
        (workspace_id, owner_id, owner_id),
    )
    project_id = connection.execute(
        """
        INSERT INTO workspace_projects (
            workspace_id, name, normalized_name, status, version
        ) VALUES (?, 'Launch', 'launch', 'active', 1)
        """,
        (workspace_id,),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO knowledge_spaces (
            kind, workspace_id, workspace_project_id, status, version
        ) VALUES ('project', ?, ?, 'active', 1)
        """,
        (workspace_id, project_id),
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO knowledge_spaces (
                kind, personal_owner_user_id, workspace_id, status, version
            ) VALUES ('personal', ?, ?, 'active', 1)
            """,
            (owner_id, workspace_id),
        )
    connection.rollback()

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO workspace_projects (
                workspace_id, name, normalized_name, status, version
            ) VALUES (?, 'LAUNCH', 'launch', 'active', 1)
            """,
            (workspace_id,),
        )
    connection.rollback()

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO workspace_invitations (
                workspace_id, inviter_user_id, role, delivery_mode, template_key,
                token_hash, status, expires_at, version
            ) VALUES (?, ?, 'editor', 'direct', 'pair_1', ?, 'pending', ?, 1)
            """,
            (workspace_id, owner_id, "a" * 64, "2099-01-01T00:00:00+00:00"),
        )
    connection.rollback()

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO workspace_action_tokens (
                token_hash, actor_user_id, chat_id, scope_kind, action,
                status, expires_at, consumed_at
            ) VALUES (?, ?, 1, 'wizard', 'input:name', 'awaiting_input', ?, ?)
            """,
            (
                "b" * 64,
                owner_id,
                "2099-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
    connection.rollback()

    invitation_id = connection.execute(
        """
        INSERT INTO workspace_invitations (
            workspace_id, inviter_user_id, intended_user_id, role, delivery_mode,
            template_key, token_hash, status, expires_at, version
        ) VALUES (?, ?, ?, 'editor', 'direct', 'pair_1', ?, 'pending', ?, 1)
        """,
        (
            workspace_id,
            owner_id,
            intended_id,
            "c" * 64,
            "2099-01-01T00:00:00+00:00",
        ),
    ).lastrowid
    connection.commit()
    connection.execute("DELETE FROM users WHERE id = ?", (intended_id,))
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM workspace_invitations WHERE id = ?", (invitation_id,)
        ).fetchone()[0]
        == 0
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.commit()
    connection.close()

    alembic(project_root, environment, "downgrade", "20260720_0017")
    connection = sqlite3.connect(database)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert not (WORKSPACE_TABLES & tables)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM inbox_items WHERE id = ?", (sentinel_id,)
        ).fetchone()[0]
        == 1
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260722_0019"
    )
    assert all(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        for table in WORKSPACE_TABLES
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_workspace_migration_upgrades_clean_sqlite(tmp_path):
    project_root = Path(__file__).parents[1]
    database = tmp_path / "clean.db"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260722_0019"
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
