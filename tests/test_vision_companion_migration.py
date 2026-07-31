import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def alembic(project_root: Path, environment: dict[str, str], command: str, revision: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", command, revision],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_companion_migration_round_trip_preserves_vision_items(tmp_path):
    database_path = tmp_path / "vision-companion.db"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database_path.as_posix()}"
    alembic(project_root, environment, "upgrade", "20260725_0020")

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (809001, "Europe/Moscow", 1),
    ).lastrowid
    item_id = connection.execute(
        "INSERT INTO vision_items (owner_id, category, wish_text, status) VALUES (?, ?, ?, ?)",
        (owner_id, "growth_creativity", "Выступить уверенно", "active"),
    ).lastrowid
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "20260731_0021")
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys=ON")
    preference_id = connection.execute(
        """
        INSERT INTO vision_companion_preferences (
            owner_id, vision_item_id, telegram_user_id, chat_id, timezone,
            morning_time, evening_time, extra_per_day, extra_times, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner_id,
            item_id,
            809001,
            809001,
            "Europe/Moscow",
            "08:00",
            "21:00",
            2,
            '["12:20", "16:40"]',
            1,
        ),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO vision_companion_checkins (
            owner_id, vision_item_id, local_date, moment, response
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (owner_id, item_id, "2026-07-31", "morning", "committed"),
    )
    connection.commit()
    assert preference_id > 0
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "downgrade", "20260725_0020")
    connection = sqlite3.connect(database_path)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "vision_companion_preferences" not in tables
    assert "vision_companion_checkins" not in tables
    assert (
        connection.execute(
            "SELECT wish_text FROM vision_items WHERE id = ?", (item_id,)
        ).fetchone()[0]
        == "Выступить уверенно"
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
