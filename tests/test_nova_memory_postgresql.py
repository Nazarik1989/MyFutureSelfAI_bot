import importlib.util
import io
import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from future_self.db import Database
from future_self.models import Base, NovaMemoryChange, NovaMemoryItem, User

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")


def asyncpg_url(value: str) -> str:
    clean = value.strip()
    if clean.startswith("postgresql://"):
        return clean.replace("postgresql://", "postgresql+asyncpg://", 1)
    return clean


def test_postgresql_offline_ddl_compiles_nova_memory_foundation():
    project_root = Path(__file__).parents[1]
    migration_path = project_root / "alembic/versions/20260811_0026_nova_memory.py"
    spec = importlib.util.spec_from_file_location("nova_memory_migration", migration_path)
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
    assert "CREATE TABLE nova_memory_items" in ddl
    assert "CREATE TABLE nova_memory_changes" in ddl
    assert "FOREIGN KEY(owner_id) REFERENCES users (id) ON DELETE CASCADE" in ddl
    assert "uq_nova_memory_items_public_id" in ddl
    assert "uq_nova_memory_items_owner_fingerprint" in ddl
    assert "ck_nova_memory_items_category" in ddl
    assert "ck_nova_memory_items_content_length" in ddl
    assert "ck_nova_memory_items_fingerprint_length" in ddl
    assert "ck_nova_memory_changes_operation" in ddl
    assert "ck_nova_memory_changes_shape" in ddl
    assert "important BOOLEAN DEFAULT false NOT NULL" in ddl
    assert "version INTEGER DEFAULT '1' NOT NULL" in ddl
    assert "affected_count INTEGER DEFAULT '1' NOT NULL" in ddl
    assert "CREATE INDEX ix_nova_memory_items_owner_list" in ddl
    assert "CREATE INDEX ix_nova_memory_items_owner_category_list" in ddl
    assert "CREATE INDEX ix_nova_memory_changes_owner_created" in ddl
    assert "CREATE INDEX ix_nova_memory_changes_item_history" in ddl
    change_ddl = ddl.split("CREATE TABLE nova_memory_changes", maxsplit=1)[1]
    change_ddl = change_ddl.split(";", maxsplit=1)[0]
    for forbidden_column in (
        " content ",
        "content_fingerprint",
        "old_content",
        "new_content",
        "transcript",
        "model_output",
    ):
        assert forbidden_column not in change_ddl


@pytest_asyncio.fixture
async def postgres_db() -> AsyncIterator[Database]:
    if not TEST_POSTGRES_URL:
        pytest.skip("TEST_POSTGRES_URL is not configured for the PostgreSQL contract suite")
    url = asyncpg_url(TEST_POSTGRES_URL)
    assert make_url(url).get_backend_name() == "postgresql"
    schema = f"nova_memory_contract_{uuid4().hex}"
    admin_engine = create_async_engine(url)
    async with admin_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    database = Database(url)
    await database.engine.dispose()
    database.engine = create_async_engine(
        url,
        connect_args={"server_settings": {"search_path": f"{schema},public"}},
    )
    database.sessions = async_sessionmaker(database.engine, expire_on_commit=False)
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield database
    finally:
        await database.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin_engine.dispose()


async def test_postgresql_nova_memory_uniqueness_audit_and_cascade_contract(postgres_db):
    async with postgres_db.session() as session:
        owner = User(
            telegram_id=102_611,
            timezone="UTC",
            onboarding_completed=True,
            access_tier="subscriber",
            access_version=1,
        )
        other_owner = User(
            telegram_id=102_612,
            timezone="UTC",
            onboarding_completed=True,
            access_tier="admin",
            access_version=1,
        )
        session.add_all([owner, other_owner])
        await session.flush()
        owner_id = owner.id
        other_owner_id = other_owner.id
        first = NovaMemoryItem(
            public_id="00000000-0000-4000-8000-000000000011",
            owner_id=owner_id,
            category="about_me",
            content="PostgreSQL owner memory",
            content_fingerprint="a" * 64,
            important=False,
            version=1,
        )
        cross_owner = NovaMemoryItem(
            public_id="00000000-0000-4000-8000-000000000012",
            owner_id=other_owner_id,
            category="interaction",
            content="PostgreSQL cross-owner exact duplicate",
            content_fingerprint="a" * 64,
            important=False,
            version=1,
        )
        session.add_all([first, cross_owner])
        session.add(
            NovaMemoryChange(
                owner_id=owner_id,
                memory_public_id=first.public_id,
                operation="created",
                category=first.category,
                resulting_version=first.version,
                affected_count=1,
            )
        )

    with pytest.raises(IntegrityError):
        async with postgres_db.session() as session:
            session.add(
                NovaMemoryItem(
                    public_id="00000000-0000-4000-8000-000000000013",
                    owner_id=owner_id,
                    category="orientation",
                    content="PostgreSQL duplicate within owner",
                    content_fingerprint="a" * 64,
                    important=False,
                    version=1,
                )
            )
            await session.flush()

    with pytest.raises(IntegrityError):
        async with postgres_db.session() as session:
            session.add(
                NovaMemoryChange(
                    owner_id=owner_id,
                    memory_public_id=None,
                    operation="created",
                    category="about_me",
                    resulting_version=1,
                    affected_count=1,
                )
            )
            await session.flush()

    async with postgres_db.session() as session:
        session.add(
            NovaMemoryChange(
                owner_id=owner_id,
                memory_public_id=None,
                operation="deleted_all",
                category=None,
                resulting_version=None,
                affected_count=1,
            )
        )

    async with postgres_db.session() as session:
        owner = await session.get(User, owner_id)
        assert owner is not None
        await session.delete(owner)

    async with postgres_db.session() as session:
        assert (
            await session.scalar(
                select(func.count(NovaMemoryItem.id)).where(NovaMemoryItem.owner_id == owner_id)
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count(NovaMemoryChange.id)).where(NovaMemoryChange.owner_id == owner_id)
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count(NovaMemoryItem.id)).where(
                    NovaMemoryItem.owner_id == other_owner_id
                )
            )
            == 1
        )
