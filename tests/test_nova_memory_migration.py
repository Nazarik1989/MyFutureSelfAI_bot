import importlib.util
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

ITEM_TABLE = "nova_memory_items"
CHANGE_TABLE = "nova_memory_changes"
EXPECTED_HEAD = "20260822_0028"
PARENT_REVISION = "20260810_0025"


def alembic(
    project_root: Path,
    environment: dict[str, str],
    operation: str,
    revision: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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


def insert_user(connection: sqlite3.Connection, telegram_id: int, name: str) -> int:
    return connection.execute(
        """
        INSERT INTO users (
            telegram_id, display_name, timezone, onboarding_completed,
            access_tier, access_version
        ) VALUES (?, ?, 'UTC', 1, 'subscriber', 1)
        """,
        (telegram_id, name),
    ).lastrowid


def insert_item(
    connection: sqlite3.Connection,
    *,
    public_id: str,
    owner_id: int,
    fingerprint: str,
    category: str = "about_me",
    content: str = "Memory",
    version: int | None = None,
) -> int:
    version_columns = ", version" if version is not None else ""
    version_values = ", ?" if version is not None else ""
    parameters: list[object] = [public_id, owner_id, category, content, fingerprint]
    if version is not None:
        parameters.append(version)
    return connection.execute(
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint{version_columns}
        ) VALUES (?, ?, ?, ?, ?{version_values})
        """,
        parameters,
    ).lastrowid


def assert_integrity_error(connection: sqlite3.Connection, statement: str, parameters=()) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(statement, parameters)
    connection.rollback()


def test_nova_memory_upgrade_preserves_data_and_enforces_private_owner_scope(tmp_path):
    database = tmp_path / "nova-memory-0025.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", PARENT_REVISION)

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    owner_id = insert_user(connection, 102_601, "PRE-0026-OWNER-SENTINEL")
    other_owner_id = insert_user(connection, 102_602, "OTHER-OWNER-SENTINEL")
    existing_task_id = connection.execute(
        """
        INSERT INTO inbox_items (user_id, kind, title, raw_text, source, status, version)
        VALUES (?, 'task', 'PRE-0026-TASK-SENTINEL', 'PRE-0026-RAW-SENTINEL',
                'text', 'confirmed', 1)
        """,
        (owner_id,),
    ).lastrowid
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        EXPECTED_HEAD
    )
    assert connection.execute(
        "SELECT display_name FROM users WHERE id = ?", (owner_id,)
    ).fetchone() == ("PRE-0026-OWNER-SENTINEL",)
    assert connection.execute(
        "SELECT title, raw_text FROM inbox_items WHERE id = ?", (existing_task_id,)
    ).fetchone() == ("PRE-0026-TASK-SENTINEL", "PRE-0026-RAW-SENTINEL")

    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {ITEM_TABLE, CHANGE_TABLE} <= tables
    item_columns = {row[1]: row for row in connection.execute(f"PRAGMA table_info({ITEM_TABLE})")}
    assert set(item_columns) == {
        "id",
        "public_id",
        "owner_id",
        "category",
        "content",
        "content_fingerprint",
        "important",
        "version",
        "created_at",
        "updated_at",
    }
    assert item_columns["important"][3] == 1
    assert item_columns["important"][4] is not None
    assert item_columns["version"][3] == 1
    assert item_columns["version"][4] is not None
    change_columns = {
        row[1]: row for row in connection.execute(f"PRAGMA table_info({CHANGE_TABLE})")
    }
    assert set(change_columns) == {
        "id",
        "owner_id",
        "memory_public_id",
        "operation",
        "category",
        "resulting_version",
        "affected_count",
        "created_at",
    }
    assert {
        "content",
        "content_fingerprint",
        "old_content",
        "new_content",
        "transcript",
        "prompt",
        "model_output",
        "telegram_message_body",
    }.isdisjoint(change_columns)
    assert {
        "ix_nova_memory_items_owner_list",
        "ix_nova_memory_items_owner_category_list",
    } <= {row[1] for row in connection.execute(f"PRAGMA index_list({ITEM_TABLE})")}
    assert {
        "ix_nova_memory_changes_owner_created",
        "ix_nova_memory_changes_item_history",
    } <= {row[1] for row in connection.execute(f"PRAGMA index_list({CHANGE_TABLE})")}
    item_foreign_keys = connection.execute(f"PRAGMA foreign_key_list({ITEM_TABLE})").fetchall()
    change_foreign_keys = connection.execute(f"PRAGMA foreign_key_list({CHANGE_TABLE})").fetchall()
    assert any(row[2] == "users" and row[6] == "CASCADE" for row in item_foreign_keys)
    assert any(row[2] == "users" and row[6] == "CASCADE" for row in change_foreign_keys)

    first_public_id = "00000000-0000-4000-8000-000000000001"
    first_fingerprint = "a" * 64
    insert_item(
        connection,
        public_id=first_public_id,
        owner_id=owner_id,
        fingerprint=first_fingerprint,
        content="Normalized memory",
    )
    connection.commit()
    assert connection.execute(
        f"SELECT important, version FROM {ITEM_TABLE} WHERE public_id = ?",
        (first_public_id,),
    ).fetchone() == (0, 1)

    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint
        ) VALUES (?, ?, 'interaction', 'Duplicate owner memory', ?)
        """,
        ("00000000-0000-4000-8000-000000000002", owner_id, first_fingerprint),
    )
    insert_item(
        connection,
        public_id="00000000-0000-4000-8000-000000000003",
        owner_id=other_owner_id,
        fingerprint=first_fingerprint,
        content="Same normalized text for another owner",
    )
    connection.commit()
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint
        ) VALUES (?, ?, 'about_me', 'Duplicate public id', ?)
        """,
        (first_public_id, other_owner_id, "b" * 64),
    )
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint
        ) VALUES (?, ?, 'important', 'Invalid category', ?)
        """,
        ("00000000-0000-4000-8000-000000000004", owner_id, "c" * 64),
    )
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint
        ) VALUES (?, ?, 'about_me', '', ?)
        """,
        ("00000000-0000-4000-8000-000000000005", owner_id, "d" * 64),
    )
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint
        ) VALUES (?, ?, 'about_me', ?, ?)
        """,
        (
            "00000000-0000-4000-8000-000000000006",
            owner_id,
            "x" * 501,
            "e" * 64,
        ),
    )
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint, version
        ) VALUES (?, ?, 'about_me', 'Invalid version', ?, 0)
        """,
        ("00000000-0000-4000-8000-000000000007", owner_id, "f" * 64),
    )
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {ITEM_TABLE} (
            public_id, owner_id, category, content, content_fingerprint
        ) VALUES (?, ?, 'about_me', 'Invalid fingerprint', 'short')
        """,
        ("00000000-0000-4000-8000-000000000008", owner_id),
    )

    connection.execute(
        f"""
        INSERT INTO {CHANGE_TABLE} (
            owner_id, memory_public_id, operation, category,
            resulting_version, affected_count
        ) VALUES (?, ?, 'created', 'about_me', 1, 1)
        """,
        (owner_id, first_public_id),
    )
    connection.commit()
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {CHANGE_TABLE} (
            owner_id, memory_public_id, operation, category,
            resulting_version, affected_count
        ) VALUES (?, ?, 'read', 'about_me', 1, 1)
        """,
        (owner_id, first_public_id),
    )
    assert_integrity_error(
        connection,
        f"""
        INSERT INTO {CHANGE_TABLE} (
            owner_id, memory_public_id, operation, category,
            resulting_version, affected_count
        ) VALUES (?, NULL, 'created', 'about_me', 1, 1)
        """,
        (owner_id,),
    )
    connection.execute(
        f"""
        INSERT INTO {CHANGE_TABLE} (
            owner_id, memory_public_id, operation, category,
            resulting_version, affected_count
        ) VALUES (?, NULL, 'deleted_all', NULL, NULL, 2)
        """,
        (owner_id,),
    )
    connection.execute(
        f"""
        INSERT INTO {CHANGE_TABLE} (
            owner_id, memory_public_id, operation, category,
            resulting_version, affected_count
        ) VALUES (?, ?, 'deleted', 'about_me', NULL, 1)
        """,
        (other_owner_id, "00000000-0000-4000-8000-000000000003"),
    )
    connection.commit()

    connection.execute("DELETE FROM users WHERE id = ?", (other_owner_id,))
    connection.commit()
    assert (
        connection.execute(
            f"SELECT COUNT(*) FROM {ITEM_TABLE} WHERE owner_id = ?", (other_owner_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            f"SELECT COUNT(*) FROM {CHANGE_TABLE} WHERE owner_id = ?", (other_owner_id,)
        ).fetchone()[0]
        == 0
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "downgrade", PARENT_REVISION)
    connection = sqlite3.connect(database)
    remaining_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {ITEM_TABLE, CHANGE_TABLE}.isdisjoint(remaining_tables)
    assert connection.execute(
        "SELECT display_name FROM users WHERE id = ?", (owner_id,)
    ).fetchone() == ("PRE-0026-OWNER-SENTINEL",)
    assert connection.execute(
        "SELECT title FROM inbox_items WHERE id = ?", (existing_task_id,)
    ).fetchone() == ("PRE-0026-TASK-SENTINEL",)
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        EXPECTED_HEAD
    )
    assert connection.execute(f"SELECT COUNT(*) FROM {ITEM_TABLE}").fetchone()[0] == 0
    assert connection.execute(f"SELECT COUNT(*) FROM {CHANGE_TABLE}").fetchone()[0] == 0
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_nova_memory_revision_runs_with_sqlite_foreign_keys_enabled(tmp_path):
    database = tmp_path / "nova-memory-foreign-keys-on.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", PARENT_REVISION)
    migration_path = project_root / "alembic/versions/20260811_0026_nova_memory.py"
    spec = importlib.util.spec_from_file_location("nova_memory_sqlite_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    engine = create_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
            context = MigrationContext.configure(connection)
            migration.op = Operations(context)
            migration.upgrade()
            tables = {
                row[0]
                for row in connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            assert {ITEM_TABLE, CHANGE_TABLE} <= tables
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
            migration.downgrade()
            remaining = {
                row[0]
                for row in connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            assert {ITEM_TABLE, CHANGE_TABLE}.isdisjoint(remaining)
    finally:
        engine.dispose()


def test_nova_memory_clean_upgrade_has_one_head_and_no_private_defaults(tmp_path):
    database = tmp_path / "clean-nova-memory.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        EXPECTED_HEAD
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    config = Config(str(project_root / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [EXPECTED_HEAD]
    migration = project_root / "alembic/versions/20260811_0026_nova_memory.py"
    source = migration.read_text(encoding="utf-8")
    for forbidden in (
        "telegram_user_id",
        "chat_id",
        "transcript",
        "prompt",
        "model_output",
        "telegram_message_body",
    ):
        assert forbidden not in source
