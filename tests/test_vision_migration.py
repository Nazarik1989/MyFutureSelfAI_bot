import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_vision_migration_upgrades_from_pr13_and_preserves_existing_data(tmp_path):
    database_path = tmp_path / "migration.db"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database_path.as_posix()}"
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "20260720_0012"],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        INSERT INTO users (
            telegram_id, display_name, timezone, location_city,
            location_fallback_city, onboarding_completed
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (123456, "Сохранённый пользователь", "Europe/Moscow", "Москва", None, 0),
    )
    connection.commit()
    connection.close()

    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    connection = sqlite3.connect(database_path)
    revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    display_name = connection.execute("SELECT display_name FROM users").fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()

    assert revision == "20260720_0013"
    assert display_name == "Сохранённый пользователь"
    assert {"vision_items", "vision_drafts"} <= tables
    assert integrity == "ok"
