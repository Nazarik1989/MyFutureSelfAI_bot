import importlib.util
import io
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


def insert_usage(
    connection: sqlite3.Connection,
    *,
    user_id: int,
    key: str,
    token: str,
    status: str = "reserved",
    kind: str = "first_step",
    provider_started_at: str | None = None,
) -> int:
    completed_at = None if status == "reserved" else "2026-08-06 10:01:00+00:00"
    return connection.execute(
        """
        INSERT INTO guest_usage_ledger (
            user_id, demo_kind, status, idempotency_key, telegram_update_id,
            reservation_token, quota_day, reserved_at, provider_started_at,
            expires_at, completed_at
        ) VALUES (?, ?, ?, ?, 1, ?, '2026-08-06',
                  '2026-08-06 10:00:00+00:00', ?,
                  '2026-08-06 10:10:00+00:00', ?)
        """,
        (user_id, kind, status, key, token, provider_started_at, completed_at),
    ).lastrowid


def test_guest_quota_migration_upgrades_0023_and_enforces_sqlite_contract(tmp_path):
    database = tmp_path / "legacy-guest-quota.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "20260805_0023")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    first_user = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (97001, "UTC", 1),
    ).lastrowid
    second_user = connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, ?, ?)",
        (97002, "Europe/Moscow", 0),
    ).lastrowid
    connection.execute(
        """
        INSERT INTO vision_profiles (
            user_id, raw_answers, summary, "values", desired_identity, constraints
        ) VALUES (?, '{}', 'GUEST-QUOTA-PROFILE-SENTINEL', '[]', '[]', '[]')
        """,
        (first_user,),
    )
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260822_0028"
    )
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {"guest_quota_days", "guest_usage_ledger", "guest_demo_sessions"} <= tables
    usage_indexes = {row[1] for row in connection.execute("PRAGMA index_list(guest_usage_ledger)")}
    assert {
        "uq_guest_usage_one_reserved_per_user",
        "ix_guest_usage_quota_status_expiry",
        "ix_guest_usage_user_status",
        "ix_guest_usage_provider_started_at",
        "ix_guest_usage_unstarted_status_expiry",
    } <= usage_indexes
    partial_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?",
        ("uq_guest_usage_one_reserved_per_user",),
    ).fetchone()[0]
    assert "WHERE status = 'reserved'" in partial_sql

    first_usage = insert_usage(
        connection,
        user_id=first_user,
        key="migration:first",
        token="a" * 43,
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        insert_usage(
            connection,
            user_id=first_user,
            key="migration:second-live",
            token="b" * 43,
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_usage(
            connection,
            user_id=second_user,
            key="migration:success-without-start",
            token="e" * 43,
            status="succeeded",
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_usage(
            connection,
            user_id=second_user,
            key="migration:start-before-reservation",
            token="f" * 43,
            provider_started_at="2026-08-06 09:59:59+00:00",
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_usage(
            connection,
            user_id=second_user,
            key="migration:start-at-expiry",
            token="g" * 43,
            provider_started_at="2026-08-06 10:10:00+00:00",
        )
    connection.rollback()
    insert_usage(
        connection,
        user_id=second_user,
        key="migration:valid-started-success",
        token="h" * 43,
        status="succeeded",
        provider_started_at="2026-08-06 10:00:01+00:00",
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        insert_usage(
            connection,
            user_id=second_user,
            key="migration:bad-kind",
            token="c" * 43,
            kind="free_chat",
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_usage(
            connection,
            user_id=second_user,
            key="migration:bad-status",
            token="d" * 43,
            status="charged",
        )
    connection.rollback()

    connection.execute(
        """
        INSERT INTO guest_demo_sessions (
            user_id, chat_id, access_version, demo_kind, status,
            prompt_message_id, consumed_update_id, consumed_message_id, usage_id,
            result_payload, created_at, updated_at, expires_at, result_expires_at, version
        ) VALUES (?, 97001, 1, 'first_step', 'awaiting_input',
                  NULL, NULL, NULL, NULL, NULL,
                  '2026-08-06 10:00:00+00:00', '2026-08-06 10:00:00+00:00',
                  '2026-08-06 10:15:00+00:00', NULL, 1)
        """,
        (first_user,),
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO guest_demo_sessions (
                user_id, chat_id, access_version, demo_kind, status,
                result_payload, created_at, updated_at, expires_at,
                result_expires_at, version
            ) VALUES (?, 97001, 1, 'first_step', 'awaiting_input', NULL,
                      '2026-08-06 10:00:00+00:00', '2026-08-06 10:00:00+00:00',
                      '2026-08-06 10:15:00+00:00', NULL, 1)
            """,
            (first_user,),
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO guest_demo_sessions (
                user_id, chat_id, access_version, demo_kind, status, usage_id,
                result_payload, created_at, updated_at, expires_at,
                result_expires_at, version
            ) VALUES (?, 97002, 1, 'first_step', 'result_ready', ?, NULL,
                      '2026-08-06 10:00:00+00:00', '2026-08-06 10:00:00+00:00',
                      '2026-08-06 10:15:00+00:00', NULL, 1)
            """,
            (second_user, first_usage),
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO guest_demo_sessions (
                user_id, chat_id, access_version, demo_kind, status,
                result_payload, created_at, updated_at, expires_at,
                result_expires_at, version
            ) VALUES (999999, 1, 1, 'first_step', 'awaiting_input', NULL,
                      '2026-08-06 10:00:00+00:00', '2026-08-06 10:00:00+00:00',
                      '2026-08-06 10:15:00+00:00', NULL, 1)
            """
        )
    connection.rollback()
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("SELECT summary FROM vision_profiles").fetchone()[0] == (
        "GUEST-QUOTA-PROFILE-SENTINEL"
    )
    connection.close()

    alembic(project_root, environment, "downgrade", "20260805_0023")
    connection = sqlite3.connect(database)
    remaining_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert (
        not {
            "guest_quota_days",
            "guest_usage_ledger",
            "guest_demo_sessions",
        }
        & remaining_tables
    )
    assert connection.execute("SELECT summary FROM vision_profiles").fetchone()[0] == (
        "GUEST-QUOTA-PROFILE-SENTINEL"
    )
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260822_0028"
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_guest_quota_migration_upgrades_clean_sqlite_with_foreign_keys(tmp_path):
    database = tmp_path / "clean-guest-quota.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        "20260822_0028"
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_guest_quota_migration_has_expected_head_and_no_embedded_admin_id():
    project_root = Path(__file__).parents[1]
    config = Config(str(project_root / "alembic.ini"))
    assert ScriptDirectory.from_config(config).get_current_head() == "20260822_0028"
    migration = project_root / "alembic/versions/20260806_0024_guest_demo_quota.py"
    source = migration.read_text(encoding="utf-8")
    assert "530129470" not in source
    ledger_ddl = source[
        source.index('"guest_usage_ledger"') : source.index('"guest_demo_sessions"')
    ]
    assert "provider_started_at" in ledger_ddl
    for forbidden in ("prompt", "input", "result_payload", "provider_response", "error_body"):
        assert forbidden not in ledger_ddl


def test_postgresql_offline_ddl_compiles_guest_quota_partial_index():
    project_root = Path(__file__).parents[1]
    migration_path = project_root / "alembic/versions/20260806_0024_guest_demo_quota.py"
    spec = importlib.util.spec_from_file_location("guest_quota_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    output = io.StringIO()
    context = MigrationContext.configure(
        url="postgresql://",
        opts={"as_sql": True, "output_buffer": output},
    )
    migration.op = Operations(context)
    migration.upgrade()
    ddl = output.getvalue()
    assert "CREATE TABLE guest_quota_days" in ddl
    assert "CREATE TABLE guest_usage_ledger" in ddl
    assert "CREATE TABLE guest_demo_sessions" in ddl
    assert "CREATE UNIQUE INDEX uq_guest_usage_one_reserved_per_user" in ddl
    assert "CREATE INDEX ix_guest_usage_provider_started_at" in ddl
    assert "CREATE INDEX ix_guest_usage_unstarted_status_expiry" in ddl
    assert "provider_started_at IS NULL" in ddl
    assert "status <> 'succeeded' OR provider_started_at IS NOT NULL" in ddl
    assert "WHERE status = 'reserved'" in ddl
