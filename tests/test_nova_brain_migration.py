import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

PARENT_REVISION = "20260817_0027"
EXPECTED_HEAD = "20260822_0028"


def _alembic(project_root: Path, environment: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _environment(database: Path) -> tuple[Path, dict[str, str]]:
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    return project_root, environment


def test_nova_brain_migration_round_trip_is_additive_and_enforces_contract(tmp_path):
    database = tmp_path / "nova-brain.db"
    project_root, environment = _environment(database)
    _alembic(project_root, environment, "upgrade", PARENT_REVISION)

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        """
        INSERT INTO users (
            telegram_id, display_name, timezone, onboarding_completed,
            access_tier, access_version
        ) VALUES (?, ?, 'Europe/Moscow', 1, 'subscriber', 1)
        """,
        (820_001, "PRE-BRAIN-OWNER"),
    ).lastrowid
    connection.commit()
    connection.close()

    _alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
        EXPECTED_HEAD,
    )
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {"nova_dialogue_states", "nova_observed_memories"} <= tables
    connection.execute(
        """
        INSERT INTO nova_dialogue_states (
            owner_id, telegram_user_id, chat_id, access_version,
            active_topic, last_assistant_offer_kinds, open_loops,
            revision, expires_at
        ) VALUES (?, ?, ?, 1, ?, '[]', '[]', 1, ?)
        """,
        (owner_id, 820_001, 820_001, "важная тема", "2026-09-22 12:00:00+00:00"),
    )
    connection.execute(
        """
        INSERT INTO nova_observed_memories (
            public_id, owner_id, category, semantic_key, normalized_value, content_fingerprint,
            source_kind, source_session_id, source_message_id, source_receipt,
            status, salience, revision
        ) VALUES (?, ?, 'preference', 'response_length', ?, ?,
                  'conversation', 1, 2, ?, 'active', 5, 1)
        """,
        (
            "00000000-0000-0000-0000-000000000001",
            owner_id,
            "response_length=short",
            "a" * 64,
            "b" * 64,
        ),
    )
    connection.execute(
        """
        INSERT INTO nova_observed_memories (
            public_id, owner_id, category, semantic_key, normalized_value, content_fingerprint,
            source_kind, source_session_id, source_message_id, source_receipt,
            status, salience, revision
        ) VALUES (?, ?, 'identity', 'identity', ?, ?,
                  'conversation', 1, 4, ?, 'active', 5, 1)
        """,
        (
            "00000000-0000-0000-0000-000000000003",
            owner_id,
            "identity:display_name=назар;grammatical_address=masculine",
            "e" * 64,
            "f" * 64,
        ),
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO nova_observed_memories (
                public_id, owner_id, category, semantic_key, normalized_value,
                content_fingerprint, source_kind, source_session_id,
                source_message_id, source_receipt, status, salience, revision
            ) VALUES (?, ?, 'preference', 'response_length', 'response_length=detailed', ?,
                      'conversation', 1, 5, ?, 'active', 4, 1)
            """,
            ("00000000-0000-0000-0000-000000000004", owner_id, "1" * 64, "2" * 64),
        )
    connection.rollback()
    connection.execute(
        """
        INSERT INTO nova_observed_memories (
            public_id, owner_id, category, semantic_key, normalized_value,
            content_fingerprint, source_kind, source_session_id,
            source_message_id, source_receipt, status, salience, revision
        ) VALUES (?, ?, 'preference', 'response_length', 'response_length=detailed', ?,
                  'conversation', 1, 6, ?, 'superseded', 4, 1)
        """,
        ("00000000-0000-0000-0000-000000000005", owner_id, "3" * 64, "4" * 64),
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO nova_observed_memories (
                public_id, owner_id, category, normalized_value, content_fingerprint,
                source_kind, source_session_id, source_message_id, source_receipt,
                status, salience, revision
            ) VALUES (?, ?, 'secret', 'PRIVATE', ?, 'conversation', 1, 3, ?, 'active', 9, 1)
            """,
            ("00000000-0000-0000-0000-000000000002", owner_id, "c" * 64, "d" * 64),
        )
    connection.rollback()
    assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    connection.close()

    _alembic(project_root, environment, "downgrade", PARENT_REVISION)
    connection = sqlite3.connect(database)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "nova_dialogue_states" not in tables
    assert "nova_observed_memories" not in tables
    assert connection.execute(
        "SELECT display_name FROM users WHERE id = ?", (owner_id,)
    ).fetchone() == ("PRE-BRAIN-OWNER",)
    assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    connection.close()

    _alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
        EXPECTED_HEAD,
    )
    connection.close()


def test_nova_brain_migration_is_the_only_head_and_contains_no_private_defaults():
    project_root = Path(__file__).parents[1]
    config = Config(str(project_root / "alembic.ini"))
    assert ScriptDirectory.from_config(config).get_heads() == [EXPECTED_HEAD]
    source = (project_root / "alembic/versions/20260822_0028_nova_conversation_brain.py").read_text(
        encoding="utf-8"
    )
    for sentinel in ("772730201", "388556024", "686774592", "PRIVATE"):
        assert sentinel not in source
