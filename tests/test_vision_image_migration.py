import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def alembic(project_root: Path, environment: dict[str, str], revision: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_personal_image_migration_preserves_items_cascades_and_downgrades(tmp_path):
    database_path = tmp_path / "personal-images.db"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database_path.as_posix()}"
    alembic(project_root, environment, "20260720_0013")

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (777001, "Europe/Moscow", 0),
    ).lastrowid
    item_id = connection.execute(
        """
        INSERT INTO vision_items (owner_id, category, wish_text, status)
        VALUES (?, ?, ?, ?)
        """,
        (owner_id, "travel", "Существующая карточка", "active"),
    ).lastrowid
    connection.commit()
    connection.close()

    alembic(project_root, environment, "20260720_0014")
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        INSERT INTO vision_item_images (
            vision_item_id, owner_id, image_bytes, mime_type,
            width, height, sha256, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (item_id, owner_id, b"normalized", "image/jpeg", 10, 10, "a" * 64, 1),
    )
    connection.commit()
    assert connection.execute("SELECT wish_text FROM vision_items").fetchone()[0] == (
        "Существующая карточка"
    )
    connection.execute("DELETE FROM vision_items WHERE id = ?", (item_id,))
    connection.execute(
        """
        INSERT INTO vision_items (owner_id, category, wish_text, status)
        VALUES (?, ?, ?, ?)
        """,
        (owner_id, "home", "Карточка переживает downgrade", "active"),
    )
    connection.commit()
    assert connection.execute("SELECT COUNT(*) FROM vision_item_images").fetchone()[0] == 0
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "20260720_0013"],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    connection = sqlite3.connect(database_path)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "vision_item_images" not in tables
    assert "vision_items" in tables
    assert connection.execute("SELECT wish_text FROM vision_items").fetchone()[0] == (
        "Карточка переживает downgrade"
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
