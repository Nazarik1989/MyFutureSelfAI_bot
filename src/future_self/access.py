from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select, update

from .db import Database
from .models import AccessTierChange, User

type AccessTier = Literal["guest", "subscriber", "admin", "blocked"]

GUEST: AccessTier = "guest"
SUBSCRIBER: AccessTier = "subscriber"
ADMIN: AccessTier = "admin"
BLOCKED: AccessTier = "blocked"
ACCESS_TIERS = frozenset({GUEST, SUBSCRIBER, ADMIN, BLOCKED})
FULL_ACCESS_TIERS = frozenset({SUBSCRIBER, ADMIN})


class AccessError(RuntimeError):
    """Base class for safe access-management domain failures."""


class AccessUserNotFound(AccessError):
    def __init__(self, telegram_id: int):
        super().__init__(f"Access user not found: {telegram_id}")
        self.telegram_id = telegram_id


class InvalidAccessTier(AccessError):
    def __init__(self, tier: str):
        super().__init__(f"Unsupported access tier: {tier}")
        self.tier = tier


@dataclass(frozen=True, slots=True)
class AccessStatus:
    telegram_id: int
    access_tier: AccessTier
    access_version: int
    onboarding_completed: bool


@dataclass(frozen=True, slots=True)
class AccessMutationResult:
    status: AccessStatus
    changed: bool


def is_access_tier(value: str) -> bool:
    return value in ACCESS_TIERS


def require_access_tier(value: str) -> AccessTier:
    if not is_access_tier(value):
        raise InvalidAccessTier(value)
    return value  # type: ignore[return-value]


def is_full_access_tier(value: str) -> bool:
    return value in FULL_ACCESS_TIERS


class AccessService:
    def __init__(self, db: Database):
        self.db = db

    async def status(self, telegram_id: int) -> AccessStatus | None:
        self._telegram_id(telegram_id)
        async with self.db.sessions() as session:
            user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
            return self._status(user) if user is not None else None

    @staticmethod
    def is_full_access(tier: str) -> bool:
        return is_full_access_tier(tier)

    async def has_full_access_by_user_id(self, user_id: int) -> bool:
        self._internal_user_id(user_id)
        async with self.db.sessions() as session:
            tier = await session.scalar(select(User.access_tier).where(User.id == user_id))
        return isinstance(tier, str) and is_full_access_tier(tier)

    async def has_full_access_by_telegram_id(self, telegram_id: int) -> bool:
        self._telegram_id(telegram_id)
        async with self.db.sessions() as session:
            tier = await session.scalar(
                select(User.access_tier).where(User.telegram_id == telegram_id)
            )
        return isinstance(tier, str) and is_full_access_tier(tier)

    async def set_tier(
        self,
        telegram_id: int,
        tier: str,
        *,
        source: str = "system",
    ) -> AccessMutationResult:
        target = require_access_tier(tier)
        return await self._mutate(telegram_id, lambda _current: target, source=source)

    async def grant_subscriber(
        self, telegram_id: int, *, source: str = "operator-cli"
    ) -> AccessMutationResult:
        return await self.set_tier(telegram_id, SUBSCRIBER, source=source)

    async def grant_admin(
        self, telegram_id: int, *, source: str = "operator-cli"
    ) -> AccessMutationResult:
        return await self.set_tier(telegram_id, ADMIN, source=source)

    async def set_guest(
        self, telegram_id: int, *, source: str = "operator-cli"
    ) -> AccessMutationResult:
        return await self.set_tier(telegram_id, GUEST, source=source)

    async def block(
        self, telegram_id: int, *, source: str = "operator-cli"
    ) -> AccessMutationResult:
        return await self.set_tier(telegram_id, BLOCKED, source=source)

    async def unblock(
        self, telegram_id: int, *, source: str = "operator-cli"
    ) -> AccessMutationResult:
        return await self._mutate(
            telegram_id,
            lambda current: GUEST if current == BLOCKED else current,
            source=source,
        )

    async def _mutate(
        self,
        telegram_id: int,
        transition: Callable[[AccessTier], AccessTier],
        *,
        source: str,
    ) -> AccessMutationResult:
        self._telegram_id(telegram_id)
        clean_source = self._source(source)
        async with self.db.session() as session:
            # The no-op UPDATE is a cross-dialect row mutex. PostgreSQL takes a
            # row lock; SQLite takes its writer lock before the following read.
            locked = await session.execute(
                update(User)
                .where(User.telegram_id == telegram_id)
                .values(updated_at=User.updated_at)
            )
            if locked.rowcount != 1:
                raise AccessUserNotFound(telegram_id)
            user = await session.scalar(
                select(User).where(User.telegram_id == telegram_id).with_for_update()
            )
            if user is None:
                raise AccessUserNotFound(telegram_id)
            current = require_access_tier(user.access_tier)
            target = transition(current)
            if target == current:
                return AccessMutationResult(self._status(user), changed=False)
            user.access_tier = target
            user.access_version += 1
            session.add(
                AccessTierChange(
                    user_id=user.id,
                    from_tier=current,
                    to_tier=target,
                    source=clean_source,
                )
            )
            await session.flush()
            return AccessMutationResult(self._status(user), changed=True)

    @staticmethod
    def _status(user: User) -> AccessStatus:
        return AccessStatus(
            telegram_id=user.telegram_id,
            access_tier=require_access_tier(user.access_tier),
            access_version=user.access_version,
            onboarding_completed=user.onboarding_completed,
        )

    @staticmethod
    def _telegram_id(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("telegram_id must be a positive integer")
        return value

    @staticmethod
    def _internal_user_id(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("user_id must be a positive integer")
        return value

    @staticmethod
    def _source(value: str) -> str:
        clean = value.strip()
        if not 1 <= len(clean) <= 64:
            raise ValueError("source must contain between 1 and 64 characters")
        return clean
