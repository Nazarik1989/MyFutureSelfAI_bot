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
from sqlalchemy import create_engine

SCHEDULE_TABLE = "recurring_task_reminder_schedules"
OCCURRENCE_TABLE = "recurring_task_reminder_occurrences"
EXPECTED_HEAD = "20260811_0026"


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


def insert_user(connection: sqlite3.Connection, telegram_id: int) -> int:
    return connection.execute(
        "INSERT INTO users (telegram_id, timezone, onboarding_completed) VALUES (?, 'UTC', 1)",
        (telegram_id,),
    ).lastrowid


def insert_task(connection: sqlite3.Connection, owner_id: int, title: str) -> int:
    return connection.execute(
        """
        INSERT INTO inbox_items (user_id, kind, title, raw_text, source, status, version)
        VALUES (?, 'task', ?, 'ONE-SHOT-RAW-SENTINEL', 'text', 'confirmed', 1)
        """,
        (owner_id, title),
    ).lastrowid


def insert_schedule(
    connection: sqlite3.Connection,
    *,
    owner_id: int,
    inbox_item_id: int,
    local_time: str = "19:30:00",
    start_local_date: str = "2026-08-10",
    next_occurrence_at: str = "2026-08-10 19:30:00+00:00",
) -> int:
    return connection.execute(
        f"""
        INSERT INTO {SCHEDULE_TABLE} (
            owner_id, inbox_item_id, recurrence_kind, local_time, timezone,
            timezone_source, start_local_date, next_occurrence_at, status, version
        ) VALUES (?, ?, 'daily', ?, 'UTC', 'explicit', ?, ?, 'active', 1)
        """,
        (owner_id, inbox_item_id, local_time, start_local_date, next_occurrence_at),
    ).lastrowid


def insert_occurrence(
    connection: sqlite3.Connection,
    *,
    schedule_id: int,
    scheduled_for: str,
    local_date: str,
    delivery_key: str,
    schedule_version: int = 1,
    status: str = "pending",
    claim_token: str | None = None,
    claimed_at: str | None = None,
    delivery_started_at: str | None = None,
    sent_at: str | None = None,
) -> int:
    return connection.execute(
        f"""
        INSERT INTO {OCCURRENCE_TABLE} (
            schedule_id, schedule_version, scheduled_for, local_date,
            delivery_key, status, claim_token, claimed_at, delivery_started_at,
            next_attempt_at, attempt_count, sent_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            schedule_id,
            schedule_version,
            scheduled_for,
            local_date,
            delivery_key,
            status,
            claim_token,
            claimed_at,
            delivery_started_at,
            scheduled_for,
            sent_at,
        ),
    ).lastrowid


def test_recurring_migration_upgrades_0024_preserves_one_shot_and_enforces_contract(
    tmp_path,
):
    database = tmp_path / "legacy-recurring-reminders.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "20260806_0024")

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = insert_user(connection, 99_001)
    other_owner_id = insert_user(connection, 99_002)
    one_shot_item_id = insert_task(connection, owner_id, "ONE-SHOT-TITLE-SENTINEL")
    recurring_item_id = insert_task(connection, owner_id, "RECURRING-TITLE-SENTINEL")
    connection.execute(
        """
        INSERT INTO task_reminders (
            inbox_item_id, telegram_user_id, chat_id, event_at, remind_at,
            timezone, delivery_key, task_version, status, attempt_count
        ) VALUES (?, 99001, 99001,
                  '2026-08-11 20:00:00+00:00', '2026-08-11 19:30:00+00:00',
                  'UTC', 'ONE-SHOT-DELIVERY-SENTINEL', 1, 'pending', 0)
        """,
        (one_shot_item_id,),
    )
    connection.commit()
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        EXPECTED_HEAD
    )
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {SCHEDULE_TABLE, OCCURRENCE_TABLE} <= tables
    assert connection.execute(f"SELECT COUNT(*) FROM {SCHEDULE_TABLE}").fetchone()[0] == 0
    assert connection.execute(f"SELECT COUNT(*) FROM {OCCURRENCE_TABLE}").fetchone()[0] == 0
    assert connection.execute(
        "SELECT delivery_key, status, task_version FROM task_reminders WHERE inbox_item_id = ?",
        (one_shot_item_id,),
    ).fetchone() == ("ONE-SHOT-DELIVERY-SENTINEL", "pending", 1)
    assert connection.execute(
        "SELECT title, raw_text FROM inbox_items WHERE id = ?",
        (one_shot_item_id,),
    ).fetchone() == ("ONE-SHOT-TITLE-SENTINEL", "ONE-SHOT-RAW-SENTINEL")

    schedule_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({SCHEDULE_TABLE})")
    }
    occurrence_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({OCCURRENCE_TABLE})")
    }
    assert {
        "telegram_user_id",
        "chat_id",
        "raw_text",
        "title",
        "access_version",
    }.isdisjoint(schedule_columns)
    assert {"telegram_user_id", "chat_id", "raw_text", "title", "error_body"}.isdisjoint(
        occurrence_columns
    )
    assert "delivery_started_at" in occurrence_columns
    assert {
        "ix_recurring_schedule_due",
        "ix_recurring_schedule_owner_status",
    } <= {row[1] for row in connection.execute(f"PRAGMA index_list({SCHEDULE_TABLE})")}
    assert {
        "ix_recurring_occurrence_due",
        "ix_recurring_occurrence_claims",
        "ix_recurring_occurrence_schedule_history",
        "ix_recurring_occurrence_cleanup",
        "uq_recurring_occurrence_sent_date",
    } <= {row[1] for row in connection.execute(f"PRAGMA index_list({OCCURRENCE_TABLE})")}

    schedule_id = insert_schedule(
        connection,
        owner_id=owner_id,
        inbox_item_id=one_shot_item_id,
    )
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        insert_schedule(
            connection,
            owner_id=other_owner_id,
            inbox_item_id=recurring_item_id,
        )
    connection.rollback()
    second_schedule_id = insert_schedule(
        connection,
        owner_id=owner_id,
        inbox_item_id=recurring_item_id,
        local_time="08:00:00",
        next_occurrence_at="2026-08-11 08:00:00+00:00",
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        insert_schedule(connection, owner_id=owner_id, inbox_item_id=one_shot_item_id)
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            f"UPDATE {SCHEDULE_TABLE} SET recurrence_kind = 'weekly' WHERE id = ?",
            (schedule_id,),
        )
    connection.rollback()

    insert_occurrence(
        connection,
        schedule_id=schedule_id,
        scheduled_for="2026-08-10 19:30:00+00:00",
        local_date="2026-08-10",
        delivery_key="recurring:one:2026-08-10",
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        insert_occurrence(
            connection,
            schedule_id=schedule_id,
            scheduled_for="2026-08-10 19:30:00+00:00",
            local_date="2026-08-11",
            delivery_key="recurring:duplicate-time",
        )
    connection.rollback()
    append_only_id = insert_occurrence(
        connection,
        schedule_id=schedule_id,
        schedule_version=2,
        scheduled_for="2026-08-10 19:30:00+00:00",
        local_date="2026-08-10",
        delivery_key="recurring:generation-two:2026-08-10",
    )
    connection.commit()
    assert append_only_id > 0
    insert_occurrence(
        connection,
        schedule_id=schedule_id,
        schedule_version=3,
        scheduled_for="2026-08-10 20:00:00+00:00",
        local_date="2026-08-10",
        delivery_key="recurring:sent-generation-three:2026-08-10",
        status="sent",
        delivery_started_at="2026-08-10 20:00:00+00:00",
        sent_at="2026-08-10 20:00:01+00:00",
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        insert_occurrence(
            connection,
            schedule_id=schedule_id,
            schedule_version=4,
            scheduled_for="2026-08-10 20:30:00+00:00",
            local_date="2026-08-10",
            delivery_key="recurring:duplicate-sent-day:2026-08-10",
            status="sent",
            delivery_started_at="2026-08-10 20:30:00+00:00",
            sent_at="2026-08-10 20:30:01+00:00",
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_occurrence(
            connection,
            schedule_id=schedule_id,
            scheduled_for="2026-08-10 20:30:00+00:00",
            local_date="2026-08-10",
            delivery_key="recurring:duplicate-date",
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_occurrence(
            connection,
            schedule_id=second_schedule_id,
            scheduled_for="2026-08-11 08:00:00+00:00",
            local_date="2026-08-11",
            delivery_key="recurring:one:2026-08-10",
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_occurrence(
            connection,
            schedule_id=second_schedule_id,
            scheduled_for="2026-08-11 08:00:00+00:00",
            local_date="2026-08-11",
            delivery_key="recurring:claim-required",
            status="processing",
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        insert_occurrence(
            connection,
            schedule_id=second_schedule_id,
            scheduled_for="2026-08-11 08:00:00+00:00",
            local_date="2026-08-11",
            delivery_key="recurring:send-fence-required",
            status="sent",
            sent_at="2026-08-11 08:00:01+00:00",
        )
    connection.rollback()
    insert_occurrence(
        connection,
        schedule_id=second_schedule_id,
        scheduled_for="2026-08-11 08:00:00+00:00",
        local_date="2026-08-11",
        delivery_key="recurring:processing-fenced",
        status="processing",
        claim_token="4b86d13e-c72c-456b-af55-bf268a18fb52",
        claimed_at="2026-08-11 08:00:00+00:00",
        delivery_started_at="2026-08-11 08:00:01+00:00",
    )
    connection.commit()
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    alembic(project_root, environment, "downgrade", "20260806_0024")
    connection = sqlite3.connect(database)
    remaining_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {SCHEDULE_TABLE, OCCURRENCE_TABLE}.isdisjoint(remaining_tables)
    assert (
        connection.execute(
            "SELECT delivery_key FROM task_reminders WHERE inbox_item_id = ?",
            (one_shot_item_id,),
        ).fetchone()[0]
        == "ONE-SHOT-DELIVERY-SENTINEL"
    )
    connection.close()

    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        EXPECTED_HEAD
    )
    assert connection.execute(f"SELECT COUNT(*) FROM {SCHEDULE_TABLE}").fetchone()[0] == 0
    assert connection.execute(f"SELECT COUNT(*) FROM {OCCURRENCE_TABLE}").fetchone()[0] == 0
    assert (
        connection.execute(
            "SELECT delivery_key FROM task_reminders WHERE inbox_item_id = ?",
            (one_shot_item_id,),
        ).fetchone()[0]
        == "ONE-SHOT-DELIVERY-SENTINEL"
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_recurring_migration_upgrades_clean_sqlite_with_foreign_keys(tmp_path):
    database = tmp_path / "clean-recurring-reminders.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
        EXPECTED_HEAD
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_recurring_revision_runs_with_sqlite_foreign_keys_enabled(tmp_path):
    database = tmp_path / "foreign-keys-on-recurring-reminders.db"
    project_root, environment = migration_environment(database)
    alembic(project_root, environment, "upgrade", "20260806_0024")
    migration_path = project_root / "alembic/versions/20260810_0025_recurring_task_reminders.py"
    spec = importlib.util.spec_from_file_location(
        "recurring_reminder_sqlite_migration", migration_path
    )
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
            assert {SCHEDULE_TABLE, OCCURRENCE_TABLE} <= tables
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
            migration.downgrade()
            remaining = {
                row[0]
                for row in connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            assert {SCHEDULE_TABLE, OCCURRENCE_TABLE}.isdisjoint(remaining)
    finally:
        engine.dispose()


def test_recurring_migration_is_the_only_head_and_has_no_private_defaults():
    project_root = Path(__file__).parents[1]
    config = Config(str(project_root / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [EXPECTED_HEAD]
    migration = project_root / "alembic/versions/20260810_0025_recurring_task_reminders.py"
    source = migration.read_text(encoding="utf-8")
    assert "530129470" not in source
    assert "telegram_user_id" not in source
    assert "chat_id" not in source
    assert "raw_text" not in source
    assert "access_version" not in source


def test_postgresql_offline_ddl_compiles_recurring_reminder_foundation():
    project_root = Path(__file__).parents[1]
    migration_path = project_root / "alembic/versions/20260810_0025_recurring_task_reminders.py"
    spec = importlib.util.spec_from_file_location("recurring_reminder_migration", migration_path)
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
    assert f"CREATE TABLE {SCHEDULE_TABLE}" in ddl
    assert f"CREATE TABLE {OCCURRENCE_TABLE}" in ddl
    assert "fk_recurring_schedule_inbox_owner" in ddl
    assert "uq_recurring_schedule_inbox_item" in ddl
    assert "uq_recurring_occurrence_generation_time" in ddl
    assert "uq_recurring_occurrence_generation_date" in ddl
    assert "uq_recurring_occurrence_delivery_key" in ddl
    assert "CREATE UNIQUE INDEX uq_recurring_occurrence_sent_date" in ddl
    assert "WHERE status = 'sent'" in ddl
    assert "delivery_started_at TIMESTAMP WITH TIME ZONE" in ddl
    assert "CREATE INDEX ix_recurring_schedule_due" in ddl
    assert "CREATE INDEX ix_recurring_occurrence_claims" in ddl
    assert "ALTER TABLE task_reminders" not in ddl
    assert "UPDATE task_reminders" not in ddl
