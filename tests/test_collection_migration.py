import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


def alembic(project_root: Path, environment: dict[str, str], command: str, revision: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", command, revision],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


COLLECTION_TABLES = {
    "life_collections",
    "life_collection_aliases",
    "life_collection_links",
    "life_collection_preferences",
    "life_collection_contexts",
    "life_collection_action_tokens",
}


def test_collection_migration_has_no_seed_and_round_trips_existing_schema(tmp_path):
    project_root = Path(__file__).parents[1]
    database = tmp_path / "production-copy.db"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    alembic(project_root, environment, "upgrade", "20260720_0016")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    first_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (980001, "Europe/Moscow", 0),
    ).lastrowid
    second_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (980002, "Europe/Saratov", 0),
    ).lastrowid
    item_id = connection.execute(
        """
        INSERT INTO inbox_items (user_id, kind, title, raw_text, source, status)
        VALUES (?, 'note', 'Sentinel', 'Sentinel', 'text', 'confirmed')
        """,
        (first_id,),
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
    assert COLLECTION_TABLES <= tables
    assert all(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        for table in COLLECTION_TABLES
    )
    collection_id = connection.execute(
        """
        INSERT INTO life_collections (
            owner_id, kind, name, normalized_name, status, version
        ) VALUES (?, 'topic', 'Личное', 'личное', 'active', 1)
        """,
        (first_id,),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO life_collection_links (collection_id, owner_id, inbox_item_id)
        VALUES (?, ?, ?)
        """,
        (collection_id, first_id, item_id),
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO life_collection_links (collection_id, owner_id, inbox_item_id)
            VALUES (?, ?, ?)
            """,
            (collection_id, second_id, item_id),
        )
    connection.rollback()
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("SELECT COUNT(*) FROM inbox_items").fetchone()[0] == 1
    connection.close()

    alembic(project_root, environment, "downgrade", "20260720_0016")
    connection = sqlite3.connect(database)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert not (COLLECTION_TABLES & tables)
    assert connection.execute("SELECT COUNT(*) FROM inbox_items").fetchone()[0] == 1
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260722_0019"
    )
    assert connection.execute("SELECT COUNT(*) FROM life_collections").fetchone()[0] == 0
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_collection_migration_upgrades_clean_sqlite(tmp_path):
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
