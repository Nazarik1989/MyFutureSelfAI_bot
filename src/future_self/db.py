from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base


class Database:
    """Async SQLAlchemy infrastructure. Schema changes are managed by Alembic."""

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        sqlite_busy_timeout_ms: int = 5_000,
        sqlite_wal_enabled: bool = True,
    ):
        if not 1_000 <= sqlite_busy_timeout_ms <= 60_000:
            raise ValueError("sqlite_busy_timeout_ms must be between 1000 and 60000")
        self.url = url
        self.sqlite_busy_timeout_ms = sqlite_busy_timeout_ms
        self.engine = create_async_engine(url, echo=echo)
        if self.engine.url.get_backend_name() == "sqlite":
            database_name = self.engine.url.database
            self.sqlite_wal_enabled = bool(
                sqlite_wal_enabled and database_name and database_name != ":memory:"
            )
            event.listen(self.engine.sync_engine, "connect", self._configure_sqlite_connection)
        else:
            self.sqlite_wal_enabled = False
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    def _configure_sqlite_connection(
        self, dbapi_connection: object, connection_record: object
    ) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout={self.sqlite_busy_timeout_ms}")
            cursor.execute("PRAGMA foreign_keys=ON")
            if self.sqlite_wal_enabled:
                cursor.execute("PRAGMA journal_mode=WAL")
                result = cursor.fetchone()
                if result is None or str(result[0]).casefold() != "wal":
                    raise RuntimeError("SQLite WAL mode is unavailable")
        finally:
            cursor.close()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_all_for_tests(self) -> None:
        """Create an ephemeral test schema; applications must use Alembic."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()
