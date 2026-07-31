import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def alembic(project_root: Path, environment: dict[str, str], operation: str, revision: str):
    subprocess.run(
        [sys.executable, "-m", "alembic", operation, revision],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_reference_migration_round_trip_preserves_existing_vision_data(tmp_path):
    database = tmp_path / "vision-reference-migration.sqlite3"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    alembic(project_root, environment, "upgrade", "20260731_0021")

    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (9001, "Europe/Moscow", 1),
    )
    owner_id = connection.execute("SELECT id FROM users WHERE telegram_id = 9001").fetchone()[0]
    connection.execute(
        "INSERT INTO vision_items (owner_id, category, wish_text, status) VALUES (?, ?, ?, ?)",
        (owner_id, "travel", "Увидеть океан", "active"),
    )
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute(
        """
        INSERT INTO vision_references
            (owner_id, kind, name, image_bytes, mime_type, width, height, sha256, version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (owner_id, "self", "Я", b"jpeg", "image/jpeg", 10, 10, "a" * 64, 1),
    )
    connection.commit()
    assert connection.execute("SELECT name FROM vision_references").fetchone()[0] == "Я"
    connection.close()

    alembic(project_root, environment, "downgrade", "20260731_0021")
    connection = sqlite3.connect(database)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "vision_references" not in tables
    assert connection.execute("SELECT wish_text FROM vision_items").fetchone()[0] == (
        "Увидеть океан"
    )
    connection.close()
