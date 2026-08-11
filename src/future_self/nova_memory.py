from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from .access import FULL_ACCESS_TIERS
from .db import Database
from .models import NovaMemoryChange, NovaMemoryItem, User

type NovaMemoryCategory = Literal["about_me", "interaction", "orientation"]
type NovaMemoryMutationStatus = Literal[
    "created",
    "duplicate",
    "conflict",
    "updated",
    "importance_changed",
    "unchanged",
    "deleted",
    "deleted_all",
    "not_found",
    "stale",
    "stale_access",
    "limit_reached",
    "access_denied",
]

NOVA_MEMORY_CATEGORIES = frozenset({"about_me", "interaction", "orientation"})
NOVA_MEMORY_MAX_CONTENT_LENGTH = 500
NOVA_MEMORY_DEFAULT_MAX_ITEMS = 100
NOVA_MEMORY_DEFAULT_PAGE_SIZE = 20
NOVA_MEMORY_MAX_PAGE_SIZE = 50
_COLLECTION_REVISION_DOMAIN = b"nova-memory-collection-v1\x00"
_COLLECTION_REVISION_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class NovaMemoryValidationError(ValueError):
    """A privacy-safe validation failure for a memory command."""


class NovaMemoryStorageError(RuntimeError):
    """A privacy-safe replacement for database errors that may contain parameters."""


@dataclass(frozen=True, slots=True)
class NovaMemorySnapshot:
    public_id: str
    category: NovaMemoryCategory
    content: str = field(repr=False)
    important: bool
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class NovaMemoryMutation:
    status: NovaMemoryMutationStatus
    item: NovaMemorySnapshot | None = None
    affected_count: int = 0


@dataclass(frozen=True, slots=True)
class NovaMemoryLookup:
    status: Literal["found", "not_found", "access_denied"]
    item: NovaMemorySnapshot | None = None


@dataclass(frozen=True, slots=True)
class NovaMemoryPage:
    status: Literal["ok", "access_denied"]
    items: tuple[NovaMemorySnapshot, ...] = ()
    offset: int = 0
    limit: int = NOVA_MEMORY_DEFAULT_PAGE_SIZE
    total: int = 0
    next_offset: int | None = None
    collection_revision: str | None = None


@dataclass(frozen=True, slots=True)
class NovaMemoryStatus:
    status: Literal["available", "access_denied"]
    count: int = 0
    max_items: int = NOVA_MEMORY_DEFAULT_MAX_ITEMS
    remaining: int = 0
    access_version: int | None = None
    collection_revision: str | None = None


@dataclass(frozen=True, slots=True)
class _Actor:
    owner_id: int
    access_version: int
    access_tier: str


def normalize_nova_memory_content(value: str) -> str:
    """Return the exact, bounded wording that is safe to persist."""
    if not isinstance(value, str):
        raise NovaMemoryValidationError("Memory content must be text.")
    normalized = unicodedata.normalize("NFKC", value)
    if any(
        unicodedata.category(character).startswith("C") and character not in {"\t", "\n", "\r"}
        for character in normalized
    ):
        raise NovaMemoryValidationError("Memory content contains unsupported characters.")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise NovaMemoryValidationError("Memory content cannot be empty.")
    if len(normalized) > NOVA_MEMORY_MAX_CONTENT_LENGTH:
        raise NovaMemoryValidationError("Memory content is too long.")
    return normalized


def nova_memory_fingerprint(normalized_content: str) -> str:
    """Build the internal exact-duplicate key; callers must not expose it."""
    return hashlib.sha256(normalized_content.casefold().encode("utf-8")).hexdigest()


class NovaMemoryService:
    """Durable, actor-bound storage for explicitly confirmed Nova memories."""

    def __init__(self, db: Database, *, max_items: int = NOVA_MEMORY_DEFAULT_MAX_ITEMS):
        if (
            isinstance(max_items, bool)
            or not isinstance(max_items, int)
            or not 1 <= max_items <= 100
        ):
            raise NovaMemoryValidationError("Memory item limit must be between 1 and 100.")
        self.db = db
        self.max_items = max_items

    async def create(
        self,
        *,
        telegram_actor_id: int,
        expected_access_version: int,
        category: NovaMemoryCategory | str,
        content: str,
        important: bool = False,
    ) -> NovaMemoryMutation:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        access_version = self._positive_version(expected_access_version, "access")
        clean_category = self._category(category)
        clean_content = normalize_nova_memory_content(content)
        if not isinstance(important, bool):
            raise NovaMemoryValidationError("Memory importance must be a boolean.")
        fingerprint = nova_memory_fingerprint(clean_content)
        preflight = await self._preflight_mutation(actor_id, access_version)
        if preflight is not None:
            return NovaMemoryMutation(preflight)

        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return NovaMemoryMutation(failure)
                assert actor is not None

                duplicate = await session.scalar(
                    select(NovaMemoryItem).where(
                        NovaMemoryItem.owner_id == actor.owner_id,
                        NovaMemoryItem.content_fingerprint == fingerprint,
                    )
                )
                if duplicate is not None:
                    return NovaMemoryMutation("duplicate", self._snapshot(duplicate))

                count = int(
                    await session.scalar(
                        select(func.count(NovaMemoryItem.id)).where(
                            NovaMemoryItem.owner_id == actor.owner_id
                        )
                    )
                    or 0
                )
                if count >= self.max_items:
                    return NovaMemoryMutation("limit_reached")

                item = NovaMemoryItem(
                    public_id=str(uuid4()),
                    owner_id=actor.owner_id,
                    category=clean_category,
                    content=clean_content,
                    content_fingerprint=fingerprint,
                    important=important,
                    version=1,
                )
                session.add(item)
                await session.flush()
                await session.refresh(item)
                self._audit(
                    session,
                    owner_id=actor.owner_id,
                    public_id=item.public_id,
                    operation="created",
                    category=clean_category,
                    resulting_version=item.version,
                )
                return NovaMemoryMutation("created", self._snapshot(item), affected_count=1)
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory create failed.") from None

    async def get(
        self,
        *,
        telegram_actor_id: int,
        public_id: str,
    ) -> NovaMemoryLookup:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        item_public_id = self._public_id(public_id)
        try:
            async with self.db.sessions() as session:
                actor = await self._read_actor(session, actor_id)
                if actor is None:
                    return NovaMemoryLookup("access_denied")
                item = None
                if item_public_id is not None:
                    item = await session.scalar(
                        select(NovaMemoryItem).where(
                            NovaMemoryItem.owner_id == actor.owner_id,
                            NovaMemoryItem.public_id == item_public_id,
                            self._access_generation_condition(actor_id, actor),
                        )
                    )
                snapshot = self._snapshot(item) if item is not None else None
            await self._before_read_generation_check(actor_id, actor)
            if not await self._access_generation_is_current(actor_id, actor):
                return NovaMemoryLookup("access_denied")
            if snapshot is None:
                return NovaMemoryLookup("not_found")
            return NovaMemoryLookup("found", snapshot)
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory read failed.") from None

    async def list(
        self,
        *,
        telegram_actor_id: int,
        category: NovaMemoryCategory | str | None = None,
        offset: int = 0,
        limit: int = NOVA_MEMORY_DEFAULT_PAGE_SIZE,
    ) -> NovaMemoryPage:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        clean_category = self._category(category) if category is not None else None
        clean_offset, clean_limit = self._pagination(offset, limit)
        try:
            async with self.db.sessions() as session:
                actor = await self._read_actor(session, actor_id)
                if actor is None:
                    return NovaMemoryPage("access_denied", offset=clean_offset, limit=clean_limit)
                generation = self._access_generation_condition(actor_id, actor)
                collection = tuple(
                    (
                        await session.scalars(
                            select(NovaMemoryItem).where(
                                NovaMemoryItem.owner_id == actor.owner_id,
                                generation,
                            )
                        )
                    ).all()
                )
                visible = tuple(
                    item
                    for item in collection
                    if clean_category is None or item.category == clean_category
                )
                ordered = sorted(
                    visible,
                    key=lambda item: (item.important, item.updated_at, item.id),
                    reverse=True,
                )
                total = len(ordered)
                page_items = ordered[clean_offset : clean_offset + clean_limit]
                items = tuple(self._snapshot(item) for item in page_items)
                collection_rows = tuple(
                    (item.id, item.public_id, item.version) for item in collection
                )
                consumed = clean_offset + len(items)
                revision = self._collection_revision(collection_rows)
            await self._before_read_generation_check(actor_id, actor)
            if not await self._access_generation_is_current(actor_id, actor):
                return NovaMemoryPage("access_denied", offset=clean_offset, limit=clean_limit)
            return NovaMemoryPage(
                "ok",
                items=items,
                offset=clean_offset,
                limit=clean_limit,
                total=total,
                next_offset=consumed if consumed < total else None,
                collection_revision=revision,
            )
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory list failed.") from None

    async def update(
        self,
        *,
        telegram_actor_id: int,
        public_id: str,
        expected_version: int,
        expected_access_version: int,
        content: str | None = None,
        category: NovaMemoryCategory | str | None = None,
    ) -> NovaMemoryMutation:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        item_public_id = self._public_id(public_id)
        item_version = self._positive_version(expected_version, "item")
        access_version = self._positive_version(expected_access_version, "access")
        clean_content = normalize_nova_memory_content(content) if content is not None else None
        clean_category = self._category(category) if category is not None else None
        preflight = await self._preflight_mutation(actor_id, access_version)
        if preflight is not None:
            return NovaMemoryMutation(preflight)

        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return NovaMemoryMutation(failure)
                assert actor is not None
                item = await self._owned_item(session, actor.owner_id, item_public_id)
                if item is None:
                    return NovaMemoryMutation("not_found")
                if item.version != item_version:
                    return NovaMemoryMutation("stale")

                target_content = clean_content if clean_content is not None else item.content
                target_category = clean_category if clean_category is not None else item.category
                fingerprint = nova_memory_fingerprint(target_content)
                if fingerprint != item.content_fingerprint:
                    duplicate = await session.scalar(
                        select(NovaMemoryItem).where(
                            NovaMemoryItem.owner_id == actor.owner_id,
                            NovaMemoryItem.content_fingerprint == fingerprint,
                            NovaMemoryItem.id != item.id,
                        )
                    )
                    if duplicate is not None:
                        return NovaMemoryMutation("conflict", self._snapshot(item))

                if target_content == item.content and target_category == item.category:
                    return NovaMemoryMutation("unchanged", self._snapshot(item))
                item.content = target_content
                item.content_fingerprint = fingerprint
                item.category = target_category
                item.version += 1
                item.updated_at = datetime.now(UTC)
                await session.flush()
                self._audit(
                    session,
                    owner_id=actor.owner_id,
                    public_id=item.public_id,
                    operation="updated",
                    category=self._category(item.category),
                    resulting_version=item.version,
                )
                return NovaMemoryMutation("updated", self._snapshot(item), affected_count=1)
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory update failed.") from None

    async def set_important(
        self,
        *,
        telegram_actor_id: int,
        public_id: str,
        expected_version: int,
        expected_access_version: int,
        important: bool,
    ) -> NovaMemoryMutation:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        item_public_id = self._public_id(public_id)
        item_version = self._positive_version(expected_version, "item")
        access_version = self._positive_version(expected_access_version, "access")
        if not isinstance(important, bool):
            raise NovaMemoryValidationError("Memory importance must be a boolean.")
        preflight = await self._preflight_mutation(actor_id, access_version)
        if preflight is not None:
            return NovaMemoryMutation(preflight)

        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return NovaMemoryMutation(failure)
                assert actor is not None
                item = await self._owned_item(session, actor.owner_id, item_public_id)
                if item is None:
                    return NovaMemoryMutation("not_found")
                if item.version != item_version:
                    return NovaMemoryMutation("stale")
                if item.important is important:
                    return NovaMemoryMutation("unchanged", self._snapshot(item))
                item.important = important
                item.version += 1
                item.updated_at = datetime.now(UTC)
                await session.flush()
                self._audit(
                    session,
                    owner_id=actor.owner_id,
                    public_id=item.public_id,
                    operation="importance_changed",
                    category=self._category(item.category),
                    resulting_version=item.version,
                )
                return NovaMemoryMutation(
                    "importance_changed", self._snapshot(item), affected_count=1
                )
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory importance update failed.") from None

    async def delete(
        self,
        *,
        telegram_actor_id: int,
        public_id: str,
        expected_version: int,
        expected_access_version: int,
    ) -> NovaMemoryMutation:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        item_public_id = self._public_id(public_id)
        item_version = self._positive_version(expected_version, "item")
        access_version = self._positive_version(expected_access_version, "access")
        preflight = await self._preflight_mutation(actor_id, access_version)
        if preflight is not None:
            return NovaMemoryMutation(preflight)

        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return NovaMemoryMutation(failure)
                assert actor is not None
                item = await self._owned_item(session, actor.owner_id, item_public_id)
                if item is None:
                    return NovaMemoryMutation("not_found")
                if item.version != item_version:
                    return NovaMemoryMutation("stale")
                category = self._category(item.category)
                memory_public_id = item.public_id
                await session.delete(item)
                await session.flush()
                self._audit(
                    session,
                    owner_id=actor.owner_id,
                    public_id=memory_public_id,
                    operation="deleted",
                    category=category,
                    resulting_version=None,
                )
                return NovaMemoryMutation("deleted", affected_count=1)
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory delete failed.") from None

    async def delete_all(
        self,
        *,
        telegram_actor_id: int,
        expected_access_version: int,
        expected_collection_revision: str,
    ) -> NovaMemoryMutation:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        access_version = self._positive_version(expected_access_version, "access")
        collection_revision = self._expected_collection_revision(expected_collection_revision)
        preflight = await self._preflight_mutation(actor_id, access_version)
        if preflight is not None:
            return NovaMemoryMutation(preflight)

        try:
            async with self.db.session() as session:
                actor, failure = await self._lock_actor(session, actor_id, access_version)
                if failure is not None:
                    return NovaMemoryMutation(failure)
                assert actor is not None
                collection_rows = await self._collection_rows(session, actor.owner_id)
                if self._collection_revision(collection_rows) != collection_revision:
                    return NovaMemoryMutation("stale")
                item_ids = tuple(row[0] for row in collection_rows)
                if not item_ids:
                    return NovaMemoryMutation("unchanged")
                removed = await session.execute(
                    delete(NovaMemoryItem).where(
                        NovaMemoryItem.owner_id == actor.owner_id,
                        NovaMemoryItem.id.in_(item_ids),
                    )
                )
                affected = int(removed.rowcount or 0)
                if affected != len(item_ids):
                    raise NovaMemoryStorageError("Memory collection changed during delete-all.")
                self._audit(
                    session,
                    owner_id=actor.owner_id,
                    public_id=None,
                    operation="deleted_all",
                    category=None,
                    resulting_version=None,
                    affected_count=affected,
                )
                return NovaMemoryMutation("deleted_all", affected_count=affected)
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory delete-all failed.") from None

    async def count(
        self,
        *,
        telegram_actor_id: int,
    ) -> NovaMemoryStatus:
        return await self.status(telegram_actor_id=telegram_actor_id)

    async def status(
        self,
        *,
        telegram_actor_id: int,
    ) -> NovaMemoryStatus:
        actor_id = self._telegram_actor_id(telegram_actor_id)
        try:
            async with self.db.sessions() as session:
                actor = await self._read_actor(session, actor_id)
                if actor is None:
                    return NovaMemoryStatus("access_denied", max_items=self.max_items)
                generation = self._access_generation_condition(actor_id, actor)
                collection_rows = await self._collection_rows(
                    session,
                    actor.owner_id,
                    generation_condition=generation,
                )
                count = len(collection_rows)
                revision = self._collection_revision(collection_rows)
            await self._before_read_generation_check(actor_id, actor)
            if not await self._access_generation_is_current(actor_id, actor):
                return NovaMemoryStatus("access_denied", max_items=self.max_items)
            return NovaMemoryStatus(
                "available",
                count=count,
                max_items=self.max_items,
                remaining=max(self.max_items - count, 0),
                access_version=actor.access_version,
                collection_revision=revision,
            )
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory status read failed.") from None

    async def _preflight_mutation(
        self, telegram_actor_id: int, expected_access_version: int
    ) -> Literal["access_denied", "stale_access"] | None:
        try:
            async with self.db.sessions() as session:
                actor = await self._read_actor(session, telegram_actor_id)
                if actor is None:
                    return "access_denied"
                if actor.access_version != expected_access_version:
                    return "stale_access"
                return None
        except SQLAlchemyError:
            raise NovaMemoryStorageError("Memory access check failed.") from None

    @staticmethod
    async def _read_actor(session: AsyncSession, telegram_actor_id: int) -> _Actor | None:
        row = (
            await session.execute(
                select(User.id, User.access_version, User.access_tier).where(
                    User.telegram_id == telegram_actor_id,
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                )
            )
        ).one_or_none()
        return _Actor(row.id, row.access_version, row.access_tier) if row is not None else None

    async def _access_generation_is_current(
        self,
        telegram_actor_id: int,
        actor: _Actor,
    ) -> bool:
        async with self.db.sessions() as session:
            current = await session.scalar(
                select(User.id).where(
                    User.id == actor.owner_id,
                    User.telegram_id == telegram_actor_id,
                    User.access_tier == actor.access_tier,
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                    User.access_version == actor.access_version,
                )
            )
            return current == actor.owner_id

    async def _before_read_generation_check(
        self,
        telegram_actor_id: int,
        actor: _Actor,
    ) -> None:
        """Test seam after payload materialization and before the final read fence."""
        del telegram_actor_id, actor

    @staticmethod
    def _access_generation_condition(
        telegram_actor_id: int,
        actor: _Actor,
    ) -> ColumnElement[bool]:
        return (
            select(User.id)
            .where(
                User.id == actor.owner_id,
                User.telegram_id == telegram_actor_id,
                User.access_tier == actor.access_tier,
                User.access_tier.in_(FULL_ACCESS_TIERS),
                User.access_version == actor.access_version,
            )
            .exists()
        )

    @staticmethod
    async def _collection_rows(
        session: AsyncSession,
        owner_id: int,
        *,
        generation_condition: ColumnElement[bool] | None = None,
    ) -> tuple[tuple[int, str, int], ...]:
        conditions = [NovaMemoryItem.owner_id == owner_id]
        if generation_condition is not None:
            conditions.append(generation_condition)
        rows = (
            await session.execute(
                select(
                    NovaMemoryItem.id,
                    NovaMemoryItem.public_id,
                    NovaMemoryItem.version,
                ).where(*conditions)
            )
        ).all()
        return tuple((row.id, row.public_id, row.version) for row in rows)

    @staticmethod
    def _collection_revision(rows: tuple[tuple[int, str, int], ...]) -> str:
        digest = hashlib.sha256(_COLLECTION_REVISION_DOMAIN)
        for _, public_id, version in sorted(rows, key=lambda row: row[1]):
            public_id_bytes = public_id.encode("utf-8")
            digest.update(len(public_id_bytes).to_bytes(2, "big"))
            digest.update(public_id_bytes)
            digest.update(version.to_bytes(8, "big", signed=False))
        return digest.hexdigest()

    @staticmethod
    async def _lock_actor(
        session: AsyncSession,
        telegram_actor_id: int,
        expected_access_version: int,
    ) -> tuple[_Actor | None, Literal["access_denied", "stale_access"] | None]:
        # A no-op UPDATE is the cross-dialect owner mutex used elsewhere in the
        # domain: PostgreSQL takes a row lock and SQLite acquires its writer lock.
        locked_id = await session.scalar(
            update(User)
            .where(
                User.telegram_id == telegram_actor_id,
                User.access_tier.in_(FULL_ACCESS_TIERS),
                User.access_version == expected_access_version,
            )
            .values(updated_at=User.updated_at)
            .returning(User.id)
        )
        if locked_id is None:
            current = (
                await session.execute(
                    select(User.access_tier, User.access_version).where(
                        User.telegram_id == telegram_actor_id
                    )
                )
            ).one_or_none()
            if current is None or current.access_tier not in FULL_ACCESS_TIERS:
                return None, "access_denied"
            return None, "stale_access"
        owner = await session.scalar(select(User).where(User.id == locked_id).with_for_update())
        if (
            owner is None
            or owner.access_tier not in FULL_ACCESS_TIERS
            or owner.access_version != expected_access_version
        ):
            return None, "stale_access"
        return _Actor(owner.id, owner.access_version, owner.access_tier), None

    @staticmethod
    async def _owned_item(
        session: AsyncSession,
        owner_id: int,
        public_id: str | None,
    ) -> NovaMemoryItem | None:
        if public_id is None:
            return None
        return await session.scalar(
            select(NovaMemoryItem).where(
                NovaMemoryItem.owner_id == owner_id,
                NovaMemoryItem.public_id == public_id,
            )
        )

    @staticmethod
    def _audit(
        session: AsyncSession,
        *,
        owner_id: int,
        public_id: str | None,
        operation: str,
        category: NovaMemoryCategory | None,
        resulting_version: int | None,
        affected_count: int = 1,
    ) -> None:
        session.add(
            NovaMemoryChange(
                owner_id=owner_id,
                memory_public_id=public_id,
                operation=operation,
                category=category,
                resulting_version=resulting_version,
                affected_count=affected_count,
            )
        )

    @staticmethod
    def _snapshot(item: NovaMemoryItem) -> NovaMemorySnapshot:
        return NovaMemorySnapshot(
            public_id=item.public_id,
            category=cast(NovaMemoryCategory, item.category),
            content=item.content,
            important=item.important,
            version=item.version,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    @staticmethod
    def _category(value: NovaMemoryCategory | str) -> NovaMemoryCategory:
        if not isinstance(value, str) or value not in NOVA_MEMORY_CATEGORIES:
            raise NovaMemoryValidationError("Unsupported memory category.")
        return cast(NovaMemoryCategory, value)

    @staticmethod
    def _telegram_actor_id(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NovaMemoryValidationError("Telegram actor id must be a positive integer.")
        return value

    @staticmethod
    def _positive_version(value: int, kind: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NovaMemoryValidationError(f"Expected {kind} version must be positive.")
        return value

    @staticmethod
    def _expected_collection_revision(value: str) -> str:
        if not isinstance(value, str) or _COLLECTION_REVISION_PATTERN.fullmatch(value) is None:
            raise NovaMemoryValidationError("Expected collection revision is invalid.")
        return value

    @staticmethod
    def _public_id(value: str) -> str | None:
        if not isinstance(value, str) or not 1 <= len(value) <= 64:
            return None
        if any(unicodedata.category(character).startswith("C") for character in value):
            return None
        return value

    @staticmethod
    def _pagination(offset: int, limit: int) -> tuple[int, int]:
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 100:
            raise NovaMemoryValidationError("Memory page offset must be between 0 and 100.")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= NOVA_MEMORY_MAX_PAGE_SIZE
        ):
            raise NovaMemoryValidationError("Memory page size must be between 1 and 50.")
        return offset, limit
