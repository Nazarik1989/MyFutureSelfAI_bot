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
    cursor = connection.execute(
        """
        INSERT INTO users (
            telegram_id, display_name, timezone, location_city,
            location_fallback_city, onboarding_completed
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (123456, "Сохранённый пользователь", "Europe/Moscow", "Москва", None, 0),
    )
    connection.execute(
        """
        INSERT INTO inbox_items (
            user_id, kind, title, raw_text, source, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            cursor.lastrowid,
            "task",
            "Существующая задача",
            "migration-sentinel",
            "text",
            "confirmed",
        ),
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
    existing_task = connection.execute(
        "SELECT title FROM inbox_items WHERE raw_text = ?",
        ("migration-sentinel",),
    ).fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()

    assert revision == "20260720_0017"
    assert display_name == "Сохранённый пользователь"
    assert existing_task == "Существующая задача"
    assert {"vision_items", "vision_drafts", "vision_item_images"} <= tables
    assert {"lab_documents", "lab_document_pages", "lab_delete_confirmations"} <= tables
    assert integrity == "ok"
