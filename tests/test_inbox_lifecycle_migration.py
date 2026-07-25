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


def test_inbox_lifecycle_upgrade_preserves_legacy_statuses_and_downgrades_safely(tmp_path):
    project_root = Path(__file__).parents[1]
    database = tmp_path / "inbox-lifecycle.db"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    alembic(project_root, environment, "upgrade", "20260722_0019")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        """
        INSERT INTO users (telegram_id, timezone, onboarding_completed)
        VALUES (920001, 'Europe/Moscow', 0)
        """
    ).lastrowid

    def inbox(title: str, status: str, *, kind: str = "note") -> int:
        return int(
            connection.execute(
                """
                INSERT INTO inbox_items (
                    user_id, kind, title, raw_text, source, status
                ) VALUES (?, ?, ?, ?, 'text', ?)
                """,
                (owner_id, kind, title, title, status),
            ).lastrowid
        )

    pending_id = inbox("Legacy pending", "pending")
    archived_id = inbox("Existing archive", "archived")
    task_id = inbox("Task to trash", "confirmed", kind="task")
    connection.execute(
        """
        INSERT INTO task_states (
            owner_id, inbox_item_id, status, event_at, timezone, version
        ) VALUES (?, ?, 'active', '2026-07-26 12:00:00+00:00', 'Europe/Moscow', 1)
        """,
        (owner_id, task_id),
    )
    connection.execute(
        """
        INSERT INTO task_reminders (
            inbox_item_id, telegram_user_id, chat_id, event_at, remind_at,
            timezone, delivery_key, task_version, status, claim_token,
            claimed_at, next_attempt_at, attempt_count
        ) VALUES (
            ?, 920001, 920001, '2026-07-26 12:00:00+00:00',
            '2026-07-26 11:30:00+00:00', 'Europe/Moscow', 'migration-reminder',
            1, 'processing', 'claim', '2026-07-25 12:00:00+00:00',
            '2026-07-25 12:01:00+00:00', 1
        )
        """,
        (task_id,),
    )
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "20260725_0020")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260725_0020"
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(inbox_items)")}
    assert {"version", "trashed_at", "pre_trash_status"} <= columns
    assert connection.execute(
        "SELECT status, version, trashed_at, pre_trash_status FROM inbox_items WHERE id = ?",
        (pending_id,),
    ).fetchone() == ("pending", 1, None, None)
    assert connection.execute(
        "SELECT status, version, trashed_at, pre_trash_status FROM inbox_items WHERE id = ?",
        (archived_id,),
    ).fetchone() == ("archived", 1, None, None)
    indexes = {row[1] for row in connection.execute("PRAGMA index_list(inbox_items)")}
    assert "ix_inbox_items_trashed_at" in indexes
    assert "ix_inbox_items_user_status_id" in indexes

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE inbox_items SET status = 'trashed' WHERE id = ?",
            (pending_id,),
        )
    connection.rollback()
    connection.execute(
        """
        UPDATE inbox_items
           SET status = 'trashed', version = version + 1,
               trashed_at = '2026-07-25 12:00:00+00:00',
               pre_trash_status = 'confirmed'
         WHERE id = ?
        """,
        (task_id,),
    )
    connection.commit()
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    with pytest.raises(subprocess.CalledProcessError):
        alembic(project_root, environment, "downgrade", "20260722_0019")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260725_0020"
    )
    assert connection.execute(
        "SELECT status, pre_trash_status FROM inbox_items WHERE id = ?", (task_id,)
    ).fetchone() == ("trashed", "confirmed")
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.execute(
        """
        UPDATE inbox_items
           SET status = pre_trash_status,
               trashed_at = NULL,
               pre_trash_status = NULL,
               version = version + 1
         WHERE status = 'trashed'
        """
    )
    connection.commit()
    connection.close()

    alembic(project_root, environment, "downgrade", "20260722_0019")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(inbox_items)")}
    assert "version" not in columns
    assert "trashed_at" not in columns
    assert "pre_trash_status" not in columns
    assert (
        connection.execute("SELECT status FROM inbox_items WHERE id = ?", (pending_id,)).fetchone()[
            0
        ]
        == "pending"
    )
    assert (
        connection.execute(
            "SELECT status FROM inbox_items WHERE id = ?", (archived_id,)
        ).fetchone()[0]
        == "archived"
    )
    assert (
        connection.execute("SELECT status FROM inbox_items WHERE id = ?", (task_id,)).fetchone()[0]
        == "confirmed"
    )
    assert connection.execute(
        "SELECT status, version, completed_at FROM task_states WHERE inbox_item_id = ?",
        (task_id,),
    ).fetchone() == ("active", 1, None)
    reminder = connection.execute(
        """
        SELECT status, claim_token, claimed_at, next_attempt_at
          FROM task_reminders
         WHERE inbox_item_id = ?
        """,
        (task_id,),
    ).fetchone()
    assert reminder == (
        "processing",
        "claim",
        "2026-07-25 12:00:00+00:00",
        "2026-07-25 12:01:00+00:00",
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
