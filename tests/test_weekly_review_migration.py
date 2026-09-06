from __future__ import annotations

import importlib.util
import io
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateTable

from future_self.models import WeeklyReviewSession

EXPECTED_HEAD = "20260822_0028"
PARENT_REVISION = "20260811_0026"
WEEKLY_REVIEW_REVISION = "20260817_0027"
FOCUS_TABLE = "weekly_focuses"
AUDIT_TABLE = "weekly_focus_changes"
SESSION_TABLE = "weekly_review_sessions"


def _alembic(
    project_root: Path,
    environment: dict[str, str],
    operation: str,
    revision: str,
) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", operation, revision],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _environment(database: Path) -> tuple[Path, dict[str, str]]:
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    return project_root, environment


def _integrity_error(connection: sqlite3.Connection, statement: str, parameters=()) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(statement, parameters)
    connection.rollback()


def test_weekly_review_migration_schema_constraints_cascade_and_round_trip(tmp_path):
    database = tmp_path / "weekly-review-0027.db"
    project_root, environment = _environment(database)
    _alembic(project_root, environment, "upgrade", PARENT_REVISION)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    owner_id = connection.execute(
        """
        INSERT INTO users (
            telegram_id, display_name, timezone, onboarding_completed,
            access_tier, access_version
        ) VALUES (82001, 'PRE-0027-SENTINEL', 'Europe/Moscow', 1, 'subscriber', 1)
        """
    ).lastrowid
    connection.commit()
    connection.close()

    _alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
        EXPECTED_HEAD,
    )
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {FOCUS_TABLE, AUDIT_TABLE, SESSION_TABLE} <= tables
    assert connection.execute(
        "SELECT display_name FROM users WHERE id = ?", (owner_id,)
    ).fetchone() == ("PRE-0027-SENTINEL",)

    focus_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({FOCUS_TABLE})").fetchall()
    }
    assert focus_columns == {
        "id",
        "public_id",
        "owner_id",
        "week_start",
        "focus",
        "approach",
        "small_steps",
        "source",
        "version",
        "created_at",
        "updated_at",
    }
    audit_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({AUDIT_TABLE})").fetchall()
    }
    assert audit_columns == {
        "id",
        "owner_id",
        "focus_public_id",
        "operation",
        "resulting_version",
        "created_at",
    }
    session_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({SESSION_TABLE})").fetchall()
    }
    assert session_columns == {
        "id",
        "public_id",
        "owner_id",
        "telegram_user_id",
        "chat_id",
        "access_version",
        "week_start",
        "phase",
        "canonical_chat_id",
        "canonical_message_id",
        "base_focus_public_id",
        "base_focus_version",
        "extracted_focus",
        "extracted_approach",
        "small_steps",
        "reminder_candidates",
        "extracted_source",
        "version",
        "expires_at",
        "created_at",
        "updated_at",
    }
    private_columns = {
        "raw_text",
        "raw_input",
        "stt_input",
        "transcript",
        "provider_output",
        "model_output",
        "evidence",
        "evidence_quote",
        "telegram_payload",
    }
    assert private_columns.isdisjoint(audit_columns)
    assert private_columns.isdisjoint(session_columns)

    focus_indexes = {
        row[1] for row in connection.execute(f"PRAGMA index_list({FOCUS_TABLE})").fetchall()
    }
    session_indexes = {
        row[1] for row in connection.execute(f"PRAGMA index_list({SESSION_TABLE})").fetchall()
    }
    assert "ix_weekly_focuses_owner_history" in focus_indexes
    assert {
        "ix_weekly_review_sessions_expiry",
        "ix_weekly_review_sessions_owner_week",
    } <= session_indexes
    session_ddl = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (SESSION_TABLE,),
    ).fetchone()[0]
    assert "ck_weekly_review_sessions_candidates_fields" in session_ddl
    assert "ck_weekly_review_sessions_base_focus_generation" in session_ddl
    for table in (FOCUS_TABLE, AUDIT_TABLE, SESSION_TABLE):
        foreign_keys = connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        assert any(row[2] == "users" and row[6] == "CASCADE" for row in foreign_keys)

    public_id = "00000000-0000-4000-8000-000000000001"
    connection.execute(
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, approach, small_steps, source, version
        ) VALUES (?, ?, '2026-08-17', 'Один фокус', NULL, '["Шаг"]', 'text', 1)
        """,
        (public_id, owner_id),
    )
    connection.execute(
        f"""
        INSERT INTO {AUDIT_TABLE} (
            owner_id, focus_public_id, operation, resulting_version
        ) VALUES (?, ?, 'created', 1)
        """,
        (owner_id, public_id),
    )
    session_public_id = "00000000-0000-4000-8000-000000000002"
    connection.execute(
        f"""
        INSERT INTO {SESSION_TABLE} (
            public_id, owner_id, telegram_user_id, chat_id, access_version,
            week_start, phase, canonical_chat_id, canonical_message_id,
            extracted_focus, extracted_approach, small_steps, reminder_candidates,
            extracted_source, version, expires_at
        ) VALUES (?, ?, 82001, 82001, 1, '2026-08-17', 'preview', 82001, 55,
                  'Один фокус', NULL, '["Шаг"]',
                  '[{{"title":"Позвонить","schedule_wording":"в 15:05"}}]',
                  'voice', 1, datetime('now', '+30 minutes'))
        """,
        (session_public_id, owner_id),
    )
    connection.commit()

    _integrity_error(
        connection,
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, small_steps, source, version
        ) VALUES ('00000000-0000-4000-8000-000000000003', ?, '2026-08-17',
                  'Duplicate week', '[]', 'text', 1)
        """,
        (owner_id,),
    )
    for index, extra_key in enumerate(
        ("evidence", "evidence_quote", "raw", "private", "secret", "unknown"),
        start=20,
    ):
        _integrity_error(
            connection,
            f"""
            INSERT INTO {SESSION_TABLE} (
                public_id, owner_id, telegram_user_id, chat_id, access_version,
                week_start, phase, small_steps, reminder_candidates, version, expires_at
            ) VALUES (?, ?, 82001, ?, 1, '2026-08-17', 'processing', '[]', ?, 1,
                      datetime('now', '+30 minutes'))
            """,
            (
                f"00000000-0000-4000-8000-{index:012d}",
                owner_id,
                82_100 + index,
                (f'[{{"title":"Call","schedule_wording":"at 15:05","{extra_key}":"forbidden"}}]'),
            ),
        )
    for canonical_chat_id, canonical_message_id in ((None, 55), (82_001, None)):
        _integrity_error(
            connection,
            f"""
            INSERT INTO {SESSION_TABLE} (
                public_id, owner_id, telegram_user_id, chat_id, access_version,
                week_start, phase, canonical_chat_id, canonical_message_id,
                small_steps, reminder_candidates, version, expires_at
            ) VALUES (?, ?, 82001, 82001, 1, '2026-08-17', 'root', ?, ?,
                      '[]', '[]', 1, datetime('now', '+30 minutes'))
            """,
            (
                f"00000000-0000-4000-8000-{82_200 + int(canonical_message_id is None):012d}",
                owner_id,
                canonical_chat_id,
                canonical_message_id,
            ),
        )
    for base_public_id, base_version in (
        (None, 1),
        ("00000000-0000-4000-8000-000000000001", None),
    ):
        _integrity_error(
            connection,
            f"""
            INSERT INTO {SESSION_TABLE} (
                public_id, owner_id, telegram_user_id, chat_id, access_version,
                week_start, phase, base_focus_public_id, base_focus_version,
                small_steps, reminder_candidates, version, expires_at
            ) VALUES (?, ?, 82001, ?, 1, '2026-08-17', 'preview', ?, ?,
                      '[]', '[]', 1, datetime('now', '+30 minutes'))
            """,
            (
                f"00000000-0000-4000-8000-{82_300 + int(base_version is None):012d}",
                owner_id,
                82_300 + int(base_version is None),
                base_public_id,
                base_version,
            ),
        )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, small_steps, source, version
        ) VALUES ('short-id', ?, '2026-08-24', 'Invalid public id', '[]', 'text', 1)
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, small_steps, source, version
        ) VALUES ('00000000-0000-4000-8000-000000000009', ?, '2026-08-24',
                  'Non-string step', '[1]', 'text', 1)
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, small_steps, source, version
        ) VALUES ('00000000-0000-4000-8000-000000000008', ?, '2026-08-24',
                  'Oversized step', ?, 'text', 1)
        """,
        (owner_id, f'["{"x" * 201}"]'),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, small_steps, source, version
        ) VALUES ('00000000-0000-4000-8000-000000000004', ?, '2026-08-24',
                  '', '[]', 'text', 1)
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, small_steps, source, version
        ) VALUES ('00000000-0000-4000-8000-000000000005', ?, '2026-08-24',
                  'Invalid steps', '["1","2","3","4"]', 'text', 1)
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {FOCUS_TABLE} (
            public_id, owner_id, week_start, focus, small_steps, source, version
        ) VALUES ('00000000-0000-4000-8000-000000000006', ?, '2026-08-24',
                  'Invalid JSON shape', '{{}}', 'text', 1)
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {AUDIT_TABLE} (
            owner_id, focus_public_id, operation, resulting_version
        ) VALUES (?, ?, 'read', 1)
        """,
        (owner_id, public_id),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {SESSION_TABLE} (
            public_id, owner_id, telegram_user_id, chat_id, access_version,
            week_start, phase, small_steps, reminder_candidates, version, expires_at
        ) VALUES ('00000000-0000-4000-8000-000000000007', ?, 82001, 82002, 1,
                  '2026-08-17', 'processing', '[]', '[{{}}]', 1,
                  datetime('now', '+30 minutes'))
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {SESSION_TABLE} (
            public_id, owner_id, telegram_user_id, chat_id, access_version,
            week_start, phase, small_steps, reminder_candidates, version, expires_at
        ) VALUES ('00000000-0000-4000-8000-000000000010', ?, 82001, 82003, 1,
                  '2026-08-17', 'processing', '[]',
                  '[{{"title":1,"schedule_wording":"at 15:05"}}]', 1,
                  datetime('now', '+30 minutes'))
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {SESSION_TABLE} (
            public_id, owner_id, telegram_user_id, chat_id, access_version,
            week_start, phase, small_steps, reminder_candidates, version, expires_at
        ) VALUES ('00000000-0000-4000-8000-000000000011', ?, 82001, 82004, 1,
                  '2026-08-17', 'processing', '[]',
                  '[{{"title":"Call","schedule_wording":"at 15:05","evidence":"Call at 15:05"}}]', 1,
                  datetime('now', '+30 minutes'))
        """,
        (owner_id,),
    )
    _integrity_error(
        connection,
        f"""
        INSERT INTO {SESSION_TABLE} (
            public_id, owner_id, telegram_user_id, chat_id, access_version,
            week_start, phase, small_steps, reminder_candidates, version, expires_at
        ) VALUES ('short-id', ?, 82001, 82005, 1, '2026-08-17', 'processing',
                  '[]', '[]', 1, datetime('now', '+30 minutes'))
        """,
        (owner_id,),
    )

    connection.execute("DELETE FROM users WHERE id = ?", (owner_id,))
    connection.commit()
    for table in (FOCUS_TABLE, AUDIT_TABLE, SESSION_TABLE):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    _alembic(project_root, environment, "downgrade", PARENT_REVISION)
    connection = sqlite3.connect(database)
    remaining = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {FOCUS_TABLE, AUDIT_TABLE, SESSION_TABLE}.isdisjoint(remaining)
    connection.close()

    _alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
        EXPECTED_HEAD,
    )
    reupgraded = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {FOCUS_TABLE, AUDIT_TABLE, SESSION_TABLE} <= reupgraded
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_weekly_review_is_single_head_and_migration_has_no_private_payload_columns(tmp_path):
    database = tmp_path / "weekly-review-clean.db"
    project_root, environment = _environment(database)
    _alembic(project_root, environment, "upgrade", "head")
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
        EXPECTED_HEAD,
    )
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()

    config = Config(str(project_root / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == [EXPECTED_HEAD]
    assert script.get_revision(WEEKLY_REVIEW_REVISION).down_revision == PARENT_REVISION
    source = (project_root / "alembic/versions/20260817_0027_weekly_review.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "raw_text",
        "raw_input",
        "transcript",
        "provider_output",
        "model_output",
        "evidence",
        "evidence_quote",
        "telegram_payload",
    ):
        assert forbidden not in source


def test_weekly_review_postgresql_offline_ddl_compiles_with_portable_constraints():
    project_root = Path(__file__).parents[1]
    migration_path = project_root / "alembic/versions/20260817_0027_weekly_review.py"
    spec = importlib.util.spec_from_file_location("weekly_review_migration", migration_path)
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

    assert "CREATE TABLE weekly_focuses" in ddl
    assert "CREATE TABLE weekly_focus_changes" in ddl
    assert "CREATE TABLE weekly_review_sessions" in ddl
    assert ddl.count("FOREIGN KEY(owner_id) REFERENCES users (id) ON DELETE CASCADE") == 3
    for constraint in (
        "ck_weekly_focuses_public_id_length",
        "ck_weekly_focuses_small_steps_count",
        "ck_weekly_focuses_small_steps_lengths",
        "ck_weekly_review_sessions_public_id_length",
        "ck_weekly_review_sessions_candidates_count",
        "ck_weekly_review_sessions_candidates_json_shape",
        "ck_weekly_review_sessions_candidates_fields",
        "ck_weekly_review_sessions_canonical_binding",
        "ck_weekly_review_sessions_base_focus_generation",
        "ck_weekly_review_sessions_expiry_order",
    ):
        assert constraint in ddl

    audit_ddl = ddl.split("CREATE TABLE weekly_focus_changes", maxsplit=1)[1]
    audit_ddl = audit_ddl.split(";", maxsplit=1)[0]
    for forbidden_column in (
        " focus ",
        " approach ",
        "small_steps",
        "raw_text",
        "provider_output",
        "model_output",
        "evidence",
    ):
        assert forbidden_column not in audit_ddl

    assert "::jsonb - 'title' - 'schedule_wording'" in ddl
    assert "base_focus_public_id IS NOT NULL" in ddl
    assert "canonical_chat_id IS NOT NULL" in ddl


def test_weekly_review_orm_emits_dialect_specific_exact_candidate_checks():
    sqlite_ddl = str(CreateTable(WeeklyReviewSession.__table__).compile(dialect=sqlite.dialect()))
    postgresql_ddl = str(
        CreateTable(WeeklyReviewSession.__table__).compile(dialect=postgresql.dialect())
    )
    assert sqlite_ddl.count("ck_weekly_review_sessions_candidates_fields") == 1
    assert "json_remove" in sqlite_ddl
    assert "::jsonb" not in sqlite_ddl
    assert postgresql_ddl.count("ck_weekly_review_sessions_candidates_fields") == 1
    assert "json_remove" not in postgresql_ddl
    assert "::jsonb" in postgresql_ddl


async def test_weekly_review_optional_real_postgresql_contract():
    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if not url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEST_POSTGRES_URL must use PostgreSQL")

    project_root = Path(__file__).parents[1]
    migration_path = project_root / "alembic/versions/20260817_0027_weekly_review.py"
    spec = importlib.util.spec_from_file_location(
        "weekly_review_postgresql_contract",
        migration_path,
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    schema = f"weekly_review_{uuid4().hex}"
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text(f'SET search_path TO "{schema}"'))
            await connection.execute(
                text("CREATE TABLE users (id INTEGER PRIMARY KEY, telegram_id BIGINT NOT NULL)")
            )

            def upgrade(sync_connection):
                context = MigrationContext.configure(sync_connection)
                migration.op = Operations(context)
                migration.upgrade()

            await connection.run_sync(upgrade)
            definitions = (
                await connection.execute(
                    text(
                        """
                        SELECT conname, pg_get_constraintdef(oid)
                        FROM pg_constraint
                        WHERE connamespace = CAST(:schema AS regnamespace)
                          AND conrelid = CAST(:table AS regclass)
                        """
                    ),
                    {
                        "schema": schema,
                        "table": f'"{schema}".{SESSION_TABLE}',
                    },
                )
            ).all()
            constraint_sql = "\n".join(f"{name}: {definition}" for name, definition in definitions)
            assert "ck_weekly_review_sessions_candidates_fields" in constraint_sql
            assert "jsonb" in constraint_sql
            assert "ck_weekly_review_sessions_canonical_binding" in constraint_sql
            assert "ck_weekly_review_sessions_base_focus_generation" in constraint_sql

            def downgrade(sync_connection):
                context = MigrationContext.configure(sync_connection)
                migration.op = Operations(context)
                migration.downgrade()

            await connection.run_sync(downgrade)
            remaining = await connection.scalar(
                text("SELECT to_regclass(:table)"),
                {"table": f'"{schema}".{SESSION_TABLE}'},
            )
            assert remaining is None
    finally:
        try:
            async with engine.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            await engine.dispose()
