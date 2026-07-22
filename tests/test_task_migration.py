import json
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


def test_task_hub_migration_reconciles_existing_database_and_downgrades(tmp_path):
    project_root = Path(__file__).parents[1]
    database = tmp_path / "production-copy.db"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    alembic(project_root, environment, "upgrade", "20260720_0015")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (990001, "Europe/Moscow", 0),
    ).lastrowid
    other_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (990002, "Europe/Saratov", 0),
    ).lastrowid

    def inbox(title, *, resolved_date=None, temporal=None, kind="task"):
        return connection.execute(
            """
            INSERT INTO inbox_items (
                user_id, kind, title, raw_text, resolved_date,
                temporal_resolution, source, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_id,
                kind,
                title,
                title,
                resolved_date,
                json.dumps(temporal) if temporal else None,
                "text",
                "confirmed",
            ),
        ).lastrowid

    reminder_task = inbox(
        "Reminder canonical",
        resolved_date="2026-07-23",
        temporal={
            "resolved_at": "2026-07-23T07:00:00+00:00",
            "timezone": "Europe/Moscow",
        },
    )
    temporal_task = inbox(
        "Temporal canonical",
        temporal={
            "resolved_at": "2026-07-24T08:00:00+00:00",
            "timezone": "Europe/Saratov",
        },
    )
    date_task = inbox("Date canonical", resolved_date="2026-07-25")
    no_due_task = inbox("No due")
    inbox("Idea sentinel", kind="idea")
    connection.execute(
        """
        INSERT INTO task_reminders (
            inbox_item_id, telegram_user_id, chat_id, event_at, remind_at,
            timezone, delivery_key, status, attempt_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reminder_task,
            990001,
            990001,
            "2026-07-23 09:00:00+00:00",
            "2026-07-23 08:30:00+00:00",
            "Europe/Saratov",
            "legacy-reminder",
            "pending",
            0,
        ),
    )
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260722_0018"
    )
    states = connection.execute(
        "SELECT inbox_item_id, owner_id, status, event_at, timezone, version "
        "FROM task_states ORDER BY inbox_item_id"
    ).fetchall()
    assert len(states) == 4
    by_item = {row[0]: row for row in states}
    assert by_item[reminder_task][3].startswith("2026-07-23 09:00:00")
    assert by_item[reminder_task][4] == "Europe/Saratov"
    assert by_item[temporal_task][3].startswith("2026-07-24 08:00:00")
    assert by_item[date_task][3].startswith("2026-07-25 06:00:00")  # 09:00 Moscow
    assert by_item[no_due_task][3] is None
    assert all(row[1] == owner_id and row[2] == "active" and row[5] == 1 for row in states)
    assert connection.execute("SELECT task_version FROM task_reminders").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM task_reminders").fetchone()[0] == 1
    assert (
        connection.execute("SELECT COUNT(*) FROM inbox_items WHERE kind = 'idea'").fetchone()[0]
        == 1
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO task_states (
                owner_id, inbox_item_id, status, timezone, version
            ) VALUES (?, ?, 'active', 'Europe/Moscow', 1)
            """,
            (other_id, no_due_task),
        )
    connection.rollback()
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "downgrade", "20260720_0015")
    connection = sqlite3.connect(database)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "task_states" not in tables
    assert "task_action_tokens" not in tables
    reminder_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(task_reminders)").fetchall()
    }
    assert "task_version" not in reminder_columns
    assert (
        connection.execute("SELECT COUNT(*) FROM inbox_items WHERE kind = 'task'").fetchone()[0]
        == 4
    )
    assert connection.execute("SELECT COUNT(*) FROM task_reminders").fetchone()[0] == 1
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
