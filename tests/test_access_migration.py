import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


def alembic(project_root: Path, environment: dict[str, str], operation: str, revision: str):
    subprocess.run(
        [sys.executable, "-m", "alembic", operation, revision],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def migration_environment(database: Path) -> tuple[Path, dict[str, str]]:
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    return project_root, environment


def test_access_migration_backfills_constraints_and_preserves_domain_data(tmp_path):
    database = tmp_path / "legacy-access.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "20260731_0022")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    completed_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (70001, "Europe/Moscow", 1),
    ).lastrowid
    incomplete_id = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (70002, "UTC", 0),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO vision_profiles (
            user_id, raw_answers, summary, "values", desired_identity, constraints
        ) VALUES (?, '{}', 'PROFILE-SENTINEL', '[]', '[]', '[]')
        """,
        (completed_id,),
    )
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260810_0025"
    )
    users = connection.execute(
        "SELECT telegram_id, access_tier, access_version FROM users ORDER BY telegram_id"
    ).fetchall()
    assert users == [(70001, "subscriber", 1), (70002, "guest", 1)]
    assert connection.execute("SELECT summary FROM vision_profiles").fetchone()[0] == (
        "PROFILE-SENTINEL"
    )

    connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (70003, "UTC", 0),
    )
    assert connection.execute(
        "SELECT access_tier, access_version FROM users WHERE telegram_id = 70003"
    ).fetchone() == ("guest", 1)
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE users SET access_tier = 'owner' WHERE id = ?", (incomplete_id,))
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE users SET access_version = 0 WHERE id = ?", (incomplete_id,))
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO access_tier_changes (user_id, from_tier, to_tier, source)
            VALUES (?, 'guest', 'owner', 'test')
            """,
            (incomplete_id,),
        )
    connection.rollback()
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "downgrade", "20260731_0022")
    connection = sqlite3.connect(database)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(users)").fetchall()}
    assert "access_tier" not in columns and "access_version" not in columns
    assert connection.execute("SELECT summary FROM vision_profiles").fetchone()[0] == (
        "PROFILE-SENTINEL"
    )
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT access_tier, access_version FROM users WHERE telegram_id = 70001"
    ).fetchone() == ("subscriber", 1)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_access_migration_upgrades_clean_sqlite_with_foreign_keys(tmp_path):
    database = tmp_path / "clean-access.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260810_0025"
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_access_migration_does_not_embed_rollout_admin_id():
    migration = Path(__file__).parents[1] / "alembic/versions/20260805_0023_access_tiers.py"
    assert "530129470" not in migration.read_text(encoding="utf-8")
