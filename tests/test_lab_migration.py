import os
import shutil
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


def test_lab_migration_upgrades_copy_preserves_data_enforces_constraints_and_downgrades(
    tmp_path,
):
    project_root = Path(__file__).parents[1]
    baseline = tmp_path / "baseline.db"
    baseline_environment = os.environ.copy()
    baseline_environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{baseline.as_posix()}"
    alembic(project_root, baseline_environment, "upgrade", "20260720_0014")

    connection = sqlite3.connect(baseline)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (880001, "Europe/Moscow", 0),
    ).lastrowid
    other_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (880002, "Europe/Moscow", 0),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO inbox_items (user_id, kind, title, raw_text, source, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (owner_id, "task", "Существующая задача", "pr18-sentinel", "text", "confirmed"),
    )
    vision_id = connection.execute(
        """
        INSERT INTO vision_items (owner_id, category, wish_text, status)
        VALUES (?, ?, ?, ?)
        """,
        (owner_id, "health_energy", "Существующая карточка", "active"),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO vision_item_images (
            vision_item_id, owner_id, image_bytes, mime_type, width, height, sha256, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (vision_id, owner_id, b"existing-image", "image/jpeg", 10, 10, "a" * 64, 1),
    )
    connection.commit()
    connection.close()

    migrated = tmp_path / "production-copy.db"
    shutil.copy2(baseline, migrated)
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{migrated.as_posix()}"
    alembic(project_root, environment, "upgrade", "head")

    connection = sqlite3.connect(migrated)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260722_0019"
    )
    assert (
        connection.execute(
            "SELECT title FROM inbox_items WHERE raw_text = 'pr18-sentinel'"
        ).fetchone()[0]
        == "Существующая задача"
    )
    assert (
        connection.execute(
            "SELECT wish_text FROM vision_items WHERE id = ?", (vision_id,)
        ).fetchone()[0]
        == "Существующая карточка"
    )
    assert (
        connection.execute(
            "SELECT image_bytes FROM vision_item_images WHERE vision_item_id = ?", (vision_id,)
        ).fetchone()[0]
        == b"existing-image"
    )

    document_id = connection.execute(
        """
        INSERT INTO lab_documents (
            owner_id, title, source_type, page_count, status, version
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (owner_id, "Нормализованный документ", "pdf", 1, "saved", 1),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO lab_document_pages (
            document_id, owner_id, page_index, image_bytes, mime_type,
            width, height, sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, owner_id, 0, b"jpeg", "image/jpeg", 10, 20, "b" * 64),
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO lab_document_pages (
                document_id, owner_id, page_index, image_bytes, mime_type,
                width, height, sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, other_id, 1, b"jpeg", "image/jpeg", 10, 20, "c" * 64),
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO lab_documents (
                owner_id, title, source_type, page_count, status, version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (owner_id, "", "image", 1, "saved", 1),
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO lab_document_pages (
                document_id, owner_id, page_index, image_bytes, mime_type,
                width, height, sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, owner_id, 1, b"jpeg", "image/jpeg", 10, 20, "short"),
        )
    connection.rollback()
    connection.execute("DELETE FROM lab_documents WHERE id = ?", (document_id,))
    connection.commit()
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM lab_document_pages WHERE document_id = ?", (document_id,)
        ).fetchone()[0]
        == 0
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "downgrade", "20260720_0014")
    connection = sqlite3.connect(migrated)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "lab_documents" not in tables
    assert "lab_document_pages" not in tables
    assert "lab_delete_confirmations" not in tables
    assert (
        connection.execute(
            "SELECT title FROM inbox_items WHERE raw_text = 'pr18-sentinel'"
        ).fetchone()[0]
        == "Существующая задача"
    )
    assert (
        connection.execute(
            "SELECT wish_text FROM vision_items WHERE id = ?", (vision_id,)
        ).fetchone()[0]
        == "Существующая карточка"
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
