from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from .access import GUEST
from .config import GUEST_TEXT_PROVIDER_TIMEOUT_SECONDS, Settings
from .db import Database
from .models import GuestDemoSession, GuestQuotaDay, GuestUsageLedger, User

_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9:_-]{1,128}\Z")
_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "input",
        "raw_input",
        "original_input",
        "prompt",
        "provider_response",
        "provider_error",
        "error_body",
    }
)


class GuestDemoKind(StrEnum):
    THOUGHT_BREAKDOWN = "thought_breakdown"
    FIRST_STEP = "first_step"


class GuestUsageStatus(StrEnum):
    RESERVED = "reserved"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


class GuestSessionStatus(StrEnum):
    AWAITING_INPUT = "awaiting_input"
    PROCESSING = "processing"
    RESULT_READY = "result_ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class GuestQuotaDenialReason(StrEnum):
    DISABLED = "disabled"
    NOT_GUEST = "not_guest"
    LIFETIME_EXHAUSTED = "lifetime_exhausted"
    GLOBAL_DAILY_EXHAUSTED = "global_daily_exhausted"
    IN_PROGRESS = "in_progress"
    DUPLICATE_RESERVED = "duplicate_reserved"
    DUPLICATE_SUCCEEDED = "duplicate_succeeded"
    DUPLICATE_TERMINAL = "duplicate_terminal"
    UNAVAILABLE = "unavailable"


class GuestReservationOutcome(StrEnum):
    FAILED = "failed"
    EXPIRED = "expired"
    SUCCEEDED = "succeeded"
    ACCESS_CHANGED = "access_changed"
    STALE = "stale"
    NOT_EXPIRED = "not_expired"


class GuestProviderStartOutcome(StrEnum):
    STARTED = "started"
    ALREADY_STARTED = "already_started"
    EXPIRED = "expired"
    ACCESS_CHANGED = "access_changed"
    NOT_GUEST = "not_guest"
    STALE = "stale"


class GuestSessionOutcome(StrEnum):
    STARTED = "started"
    CLAIMED = "claimed"
    BOUND = "bound"
    RETRY_READY = "retry_ready"
    RESULT_READY = "result_ready"
    RESULT_PENDING = "result_pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"
    IN_PROGRESS = "in_progress"
    NOT_GUEST = "not_guest"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class GuestQuotaPolicy:
    enabled: bool = True
    operation_limit: int = 2
    global_daily_limit: int = 50
    reservation_ttl: timedelta = timedelta(minutes=10)
    input_ttl: timedelta = timedelta(minutes=15)
    result_ttl: timedelta = timedelta(minutes=15)
    max_result_payload_bytes: int = 8 * 1024

    def __post_init__(self) -> None:
        if not 1 <= self.operation_limit <= 10:
            raise ValueError("guest operation limit must be between 1 and 10")
        if not 1 <= self.global_daily_limit <= 100_000:
            raise ValueError("guest global daily limit must be between 1 and 100000")
        if not timedelta(minutes=2) <= self.reservation_ttl <= timedelta(minutes=60):
            raise ValueError("guest reservation TTL must be between 2 and 60 minutes")
        if self.reservation_ttl <= timedelta(seconds=GUEST_TEXT_PROVIDER_TIMEOUT_SECONDS):
            raise ValueError("guest reservation TTL must exceed the provider timeout")
        for value, label in (
            (self.input_ttl, "input"),
            (self.result_ttl, "result"),
        ):
            if not timedelta(minutes=5) <= value <= timedelta(minutes=120):
                raise ValueError(f"guest {label} TTL must be between 5 and 120 minutes")
        if not 1024 <= self.max_result_payload_bytes <= 64 * 1024:
            raise ValueError("guest result payload limit is outside the safe range")

    @classmethod
    def from_settings(cls, settings: Settings) -> GuestQuotaPolicy:
        return cls(
            enabled=settings.guest_ai_enabled,
            operation_limit=settings.guest_operation_limit,
            global_daily_limit=settings.guest_global_daily_limit,
            reservation_ttl=timedelta(minutes=settings.guest_reservation_ttl_minutes),
            input_ttl=timedelta(minutes=settings.guest_input_ttl_minutes),
            result_ttl=timedelta(minutes=settings.guest_result_ttl_minutes),
        )


@dataclass(frozen=True, slots=True)
class GuestReservation:
    usage_id: int
    user_id: int
    demo_kind: GuestDemoKind
    idempotency_key: str
    telegram_update_id: int | None
    reservation_token: str
    quota_day: date
    reserved_at: datetime
    expires_at: datetime
    provider_started_at: datetime | None


@dataclass(frozen=True, slots=True)
class GuestReservationDecision:
    reservation: GuestReservation | None
    denial_reason: GuestQuotaDenialReason | None
    is_new: bool

    @property
    def can_bind_session(self) -> bool:
        return self.is_new and self.reservation is not None and self.denial_reason is None


@dataclass(frozen=True, slots=True)
class GuestQuotaSnapshot:
    user_id: int
    successful_lifetime_count: int
    active_reservation_count: int
    remaining_operations: int
    global_used: int
    global_capacity: int
    exhausted: bool
    enabled: bool


@dataclass(frozen=True, slots=True)
class GuestReservationTransition:
    outcome: GuestReservationOutcome
    changed: bool


@dataclass(frozen=True, slots=True)
class GuestSessionSnapshot:
    session_id: int
    user_id: int
    chat_id: int
    access_version: int
    demo_kind: GuestDemoKind
    status: GuestSessionStatus
    prompt_message_id: int | None
    consumed_update_id: int | None
    consumed_message_id: int | None
    usage_id: int | None
    result_payload: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    result_expires_at: datetime | None
    version: int


@dataclass(frozen=True, slots=True)
class GuestSessionDecision:
    outcome: GuestSessionOutcome
    session: GuestSessionSnapshot | None
    changed: bool


@dataclass(frozen=True, slots=True)
class GuestProviderStartDecision:
    outcome: GuestProviderStartOutcome
    session: GuestSessionSnapshot | None
    provider_started_at: datetime | None
    changed: bool

    @property
    def can_invoke_provider(self) -> bool:
        return self.outcome is GuestProviderStartOutcome.STARTED and self.changed


@dataclass(frozen=True, slots=True)
class GuestCompletionDecision:
    outcome: GuestReservationOutcome
    session: GuestSessionSnapshot | None
    changed: bool


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _now(value: datetime | None) -> datetime:
    return _as_utc(value or datetime.now(UTC))


def _positive_id(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_positive_id(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    return _positive_id(value, label)


def _demo_kind(value: GuestDemoKind | str) -> GuestDemoKind:
    try:
        return GuestDemoKind(value)
    except ValueError as exc:
        raise ValueError("unsupported guest demo kind") from exc


def _idempotency_key(value: str) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_KEY.fullmatch(value) is None:
        raise ValueError("idempotency_key must contain 1-128 safe characters")
    return value


def _token(value: str) -> str:
    if not isinstance(value, str) or not 20 <= len(value) <= 64:
        raise ValueError("reservation_token is invalid")
    return value


def _contains_forbidden_result_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_RESULT_KEYS:
                return True
            if _contains_forbidden_result_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_result_key(item) for item in value)
    return False


class GuestQuotaService:
    def __init__(self, db: Database, policy: GuestQuotaPolicy | None = None):
        self.db = db
        self.policy = policy or GuestQuotaPolicy()

    async def reserve(
        self,
        *,
        user_id: int,
        demo_kind: GuestDemoKind | str,
        idempotency_key: str,
        telegram_update_id: int | None,
        now: datetime | None = None,
    ) -> GuestReservationDecision:
        if not self.policy.enabled:
            return GuestReservationDecision(None, GuestQuotaDenialReason.DISABLED, False)
        owner_id = _positive_id(user_id, "user_id")
        kind = _demo_kind(demo_kind)
        key = _idempotency_key(idempotency_key)
        update_id = _optional_positive_id(telegram_update_id, "telegram_update_id")
        current = _now(now)
        try:
            return await self._reserve_locked(owner_id, kind, key, update_id, current)
        except OperationalError:
            return GuestReservationDecision(None, GuestQuotaDenialReason.UNAVAILABLE, False)

    async def _reserve_locked(
        self,
        user_id: int,
        kind: GuestDemoKind,
        key: str,
        telegram_update_id: int | None,
        current: datetime,
    ) -> GuestReservationDecision:
        quota_day = current.date()
        async with self.db.session() as session:
            await self._lock_quota_day(session, quota_day)
            user = await self._lock_user(session, user_id)
            if user is None or user.access_tier != GUEST:
                return GuestReservationDecision(None, GuestQuotaDenialReason.NOT_GUEST, False)

            await session.execute(
                update(GuestUsageLedger)
                .where(
                    GuestUsageLedger.user_id == user_id,
                    GuestUsageLedger.status == GuestUsageStatus.RESERVED.value,
                    GuestUsageLedger.expires_at <= current,
                )
                .values(
                    status=GuestUsageStatus.EXPIRED.value,
                    completed_at=current,
                )
            )

            existing = await session.scalar(
                select(GuestUsageLedger).where(
                    GuestUsageLedger.user_id == user_id,
                    GuestUsageLedger.idempotency_key == key,
                )
            )
            if existing is not None:
                status = GuestUsageStatus(existing.status)
                if status is GuestUsageStatus.RESERVED and _as_utc(existing.expires_at) > current:
                    return GuestReservationDecision(
                        self._reservation(existing),
                        GuestQuotaDenialReason.DUPLICATE_RESERVED,
                        False,
                    )
                if status is GuestUsageStatus.SUCCEEDED:
                    reason = GuestQuotaDenialReason.DUPLICATE_SUCCEEDED
                else:
                    reason = GuestQuotaDenialReason.DUPLICATE_TERMINAL
                return GuestReservationDecision(None, reason, False)

            active = await session.scalar(
                select(GuestUsageLedger.id).where(
                    GuestUsageLedger.user_id == user_id,
                    GuestUsageLedger.status == GuestUsageStatus.RESERVED.value,
                    GuestUsageLedger.expires_at > current,
                )
            )
            if active is not None:
                return GuestReservationDecision(None, GuestQuotaDenialReason.IN_PROGRESS, False)

            successful = int(
                await session.scalar(
                    select(func.count(GuestUsageLedger.id)).where(
                        GuestUsageLedger.user_id == user_id,
                        GuestUsageLedger.status == GuestUsageStatus.SUCCEEDED.value,
                    )
                )
                or 0
            )
            if successful >= self.policy.operation_limit:
                return GuestReservationDecision(
                    None,
                    GuestQuotaDenialReason.LIFETIME_EXHAUSTED,
                    False,
                )

            global_used = await self._global_used(session, current)
            if global_used >= self.policy.global_daily_limit:
                return GuestReservationDecision(
                    None,
                    GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED,
                    False,
                )

            row = GuestUsageLedger(
                user_id=user_id,
                demo_kind=kind.value,
                status=GuestUsageStatus.RESERVED.value,
                idempotency_key=key,
                telegram_update_id=telegram_update_id,
                reservation_token=secrets.token_urlsafe(32),
                quota_day=quota_day,
                reserved_at=current,
                expires_at=current + self.policy.reservation_ttl,
                provider_started_at=None,
                completed_at=None,
            )
            session.add(row)
            await session.flush()
            return GuestReservationDecision(self._reservation(row), None, True)

    async def fail(
        self,
        reservation_token: str,
        *,
        now: datetime | None = None,
    ) -> GuestReservationTransition:
        token = _token(reservation_token)
        current = _now(now)
        identity = await self._usage_identity(token)
        if identity is None:
            return GuestReservationTransition(GuestReservationOutcome.STALE, False)
        usage_id, user_id, quota_day = identity
        async with self.db.session() as session:
            await self._lock_quota_day(session, quota_day)
            await self._lock_user(session, user_id)
            row = await self._locked_usage(session, usage_id, token)
            if row is None:
                return GuestReservationTransition(GuestReservationOutcome.STALE, False)
            if row.status == GuestUsageStatus.FAILED.value:
                return GuestReservationTransition(GuestReservationOutcome.FAILED, False)
            if row.status != GuestUsageStatus.RESERVED.value:
                return GuestReservationTransition(GuestReservationOutcome.STALE, False)
            if _as_utc(row.expires_at) <= current:
                row.status = GuestUsageStatus.EXPIRED.value
                row.completed_at = current
                return GuestReservationTransition(GuestReservationOutcome.EXPIRED, True)
            row.status = GuestUsageStatus.FAILED.value
            row.completed_at = current
            return GuestReservationTransition(GuestReservationOutcome.FAILED, True)

    async def expire(
        self,
        reservation_token: str,
        *,
        now: datetime | None = None,
    ) -> GuestReservationTransition:
        token = _token(reservation_token)
        current = _now(now)
        identity = await self._usage_identity(token)
        if identity is None:
            return GuestReservationTransition(GuestReservationOutcome.STALE, False)
        usage_id, user_id, quota_day = identity
        async with self.db.session() as session:
            await self._lock_quota_day(session, quota_day)
            await self._lock_user(session, user_id)
            row = await self._locked_usage(session, usage_id, token)
            if row is None:
                return GuestReservationTransition(GuestReservationOutcome.STALE, False)
            if row.status == GuestUsageStatus.EXPIRED.value:
                return GuestReservationTransition(GuestReservationOutcome.EXPIRED, False)
            if row.status != GuestUsageStatus.RESERVED.value:
                return GuestReservationTransition(GuestReservationOutcome.STALE, False)
            if _as_utc(row.expires_at) > current:
                return GuestReservationTransition(GuestReservationOutcome.NOT_EXPIRED, False)
            row.status = GuestUsageStatus.EXPIRED.value
            row.completed_at = current
            return GuestReservationTransition(GuestReservationOutcome.EXPIRED, True)

    async def snapshot(
        self,
        user_id: int,
        *,
        now: datetime | None = None,
    ) -> GuestQuotaSnapshot:
        owner_id = _positive_id(user_id, "user_id")
        current = _now(now)
        async with self.db.sessions() as session:
            user = await session.get(User, owner_id)
            successful = int(
                await session.scalar(
                    select(func.count(GuestUsageLedger.id)).where(
                        GuestUsageLedger.user_id == owner_id,
                        GuestUsageLedger.status == GuestUsageStatus.SUCCEEDED.value,
                    )
                )
                or 0
            )
            active = int(
                await session.scalar(
                    select(func.count(GuestUsageLedger.id)).where(
                        GuestUsageLedger.user_id == owner_id,
                        GuestUsageLedger.status == GuestUsageStatus.RESERVED.value,
                        GuestUsageLedger.expires_at > current,
                    )
                )
                or 0
            )
            global_used = await self._global_used(session, current)
        eligible = user is not None and user.access_tier == GUEST
        remaining = max(0, self.policy.operation_limit - successful - active) if eligible else 0
        exhausted = (
            not self.policy.enabled
            or not eligible
            or remaining == 0
            or global_used >= self.policy.global_daily_limit
        )
        return GuestQuotaSnapshot(
            user_id=owner_id,
            successful_lifetime_count=successful,
            active_reservation_count=active,
            remaining_operations=remaining,
            global_used=global_used,
            global_capacity=self.policy.global_daily_limit,
            exhausted=exhausted,
            enabled=self.policy.enabled,
        )

    async def _usage_identity(self, token: str) -> tuple[int, int, date] | None:
        async with self.db.sessions() as session:
            row = (
                await session.execute(
                    select(
                        GuestUsageLedger.id,
                        GuestUsageLedger.user_id,
                        GuestUsageLedger.quota_day,
                    ).where(GuestUsageLedger.reservation_token == token)
                )
            ).one_or_none()
        return (int(row.id), int(row.user_id), row.quota_day) if row is not None else None

    async def _lock_quota_day(
        self,
        session: AsyncSession,
        quota_day: date,
    ) -> None:
        dialect = session.get_bind().dialect.name
        values = {"quota_day": quota_day}
        if dialect == "sqlite":
            await session.execute(
                sqlite_insert(GuestQuotaDay)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[GuestQuotaDay.quota_day])
            )
            locked = await session.execute(
                update(GuestQuotaDay)
                .where(GuestQuotaDay.quota_day == quota_day)
                .values(updated_at=GuestQuotaDay.updated_at)
                .returning(GuestQuotaDay.quota_day)
            )
            if locked.scalar_one_or_none() is None:
                raise RuntimeError("guest quota-day mutex is unavailable")
            return
        if dialect == "postgresql":
            await session.execute(
                postgresql_insert(GuestQuotaDay)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[GuestQuotaDay.quota_day])
            )
        else:
            try:
                async with session.begin_nested():
                    session.add(GuestQuotaDay(**values))
                    await session.flush()
            except IntegrityError:
                pass
        locked_day = await session.scalar(
            select(GuestQuotaDay.quota_day)
            .where(GuestQuotaDay.quota_day == quota_day)
            .with_for_update()
        )
        if locked_day is None:
            raise RuntimeError("guest quota-day mutex is unavailable")

    @staticmethod
    async def _lock_user(session: AsyncSession, user_id: int) -> User | None:
        if session.get_bind().dialect.name == "sqlite":
            locked = await session.execute(
                update(User)
                .where(User.id == user_id)
                .values(updated_at=User.updated_at)
                .returning(User.id)
            )
            if locked.scalar_one_or_none() is None:
                return None
            return await session.get(User, user_id)
        return await session.scalar(select(User).where(User.id == user_id).with_for_update())

    @staticmethod
    async def _locked_usage(
        session: AsyncSession,
        usage_id: int,
        token: str,
    ) -> GuestUsageLedger | None:
        return await session.scalar(
            select(GuestUsageLedger)
            .where(
                GuestUsageLedger.id == usage_id,
                GuestUsageLedger.reservation_token == token,
            )
            .with_for_update()
        )

    @staticmethod
    async def _global_used(session: AsyncSession, current: datetime) -> int:
        day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        return int(
            await session.scalar(
                select(func.count(GuestUsageLedger.id)).where(
                    or_(
                        and_(
                            GuestUsageLedger.provider_started_at >= day_start,
                            GuestUsageLedger.provider_started_at < day_end,
                        ),
                        and_(
                            GuestUsageLedger.provider_started_at.is_(None),
                            GuestUsageLedger.status == GuestUsageStatus.RESERVED.value,
                            GuestUsageLedger.expires_at > current,
                        ),
                    ),
                )
            )
            or 0
        )

    @staticmethod
    def _reservation(row: GuestUsageLedger) -> GuestReservation:
        return GuestReservation(
            usage_id=row.id,
            user_id=row.user_id,
            demo_kind=GuestDemoKind(row.demo_kind),
            idempotency_key=row.idempotency_key,
            telegram_update_id=row.telegram_update_id,
            reservation_token=row.reservation_token,
            quota_day=row.quota_day,
            reserved_at=_as_utc(row.reserved_at),
            expires_at=_as_utc(row.expires_at),
            provider_started_at=(
                _as_utc(row.provider_started_at) if row.provider_started_at is not None else None
            ),
        )


class GuestSessionService:
    def __init__(self, db: Database, policy: GuestQuotaPolicy | None = None):
        self.db = db
        self.policy = policy or GuestQuotaPolicy()
        self.quota = GuestQuotaService(db, self.policy)

    async def start_session(
        self,
        *,
        user_id: int,
        chat_id: int,
        access_version: int,
        demo_kind: GuestDemoKind | str,
        prompt_message_id: int | None,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        version = _positive_id(access_version, "access_version")
        kind = _demo_kind(demo_kind)
        prompt_id = _optional_positive_id(prompt_message_id, "prompt_message_id")
        current = _now(now)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            user = await self.quota._lock_user(session, owner_id)
            if user is None or user.access_tier != GUEST or user.access_version != version:
                return GuestSessionDecision(GuestSessionOutcome.NOT_GUEST, None, False)
            row = await self._locked_session(session, owner_id, target_chat)
            if row is not None:
                self._expire_session(row, current)
                if row.status == GuestSessionStatus.PROCESSING.value:
                    return self._decision(GuestSessionOutcome.IN_PROGRESS, row, False)
                if row.status == GuestSessionStatus.RESULT_READY.value:
                    return self._decision(GuestSessionOutcome.RESULT_PENDING, row, False)
                row.access_version = version
                row.demo_kind = kind.value
                row.status = GuestSessionStatus.AWAITING_INPUT.value
                row.prompt_message_id = prompt_id
                row.consumed_update_id = None
                row.consumed_message_id = None
                row.usage_id = None
                row.result_payload = None
                row.created_at = current
                row.updated_at = current
                row.expires_at = current + self.policy.input_ttl
                row.result_expires_at = None
                row.version += 1
            else:
                row = GuestDemoSession(
                    user_id=owner_id,
                    chat_id=target_chat,
                    access_version=version,
                    demo_kind=kind.value,
                    status=GuestSessionStatus.AWAITING_INPUT.value,
                    prompt_message_id=prompt_id,
                    consumed_update_id=None,
                    consumed_message_id=None,
                    usage_id=None,
                    result_payload=None,
                    created_at=current,
                    updated_at=current,
                    expires_at=current + self.policy.input_ttl,
                    result_expires_at=None,
                    version=1,
                )
                session.add(row)
            await session.flush()
            return self._decision(GuestSessionOutcome.STARTED, row, True)

    async def claim_input(
        self,
        *,
        user_id: int,
        chat_id: int,
        access_version: int,
        telegram_update_id: int,
        telegram_message_id: int,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        version = _positive_id(access_version, "access_version")
        update_id = _positive_id(telegram_update_id, "telegram_update_id")
        message_id = _positive_id(telegram_message_id, "telegram_message_id")
        current = _now(now)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            user = await self.quota._lock_user(session, owner_id)
            if user is None or user.access_tier != GUEST:
                return GuestSessionDecision(GuestSessionOutcome.NOT_GUEST, None, False)
            if user.access_version != version:
                return GuestSessionDecision(GuestSessionOutcome.STALE, None, False)
            row = await self._locked_session(session, owner_id, target_chat)
            if row is None or row.access_version != version:
                return GuestSessionDecision(GuestSessionOutcome.STALE, None, False)
            self._expire_session(row, current)
            if row.consumed_update_id == update_id or row.consumed_message_id == message_id:
                return self._decision(GuestSessionOutcome.DUPLICATE, row, False)
            if row.status == GuestSessionStatus.PROCESSING.value:
                return self._decision(GuestSessionOutcome.IN_PROGRESS, row, False)
            if row.status == GuestSessionStatus.RESULT_READY.value:
                return self._decision(GuestSessionOutcome.RESULT_PENDING, row, False)
            if row.status != GuestSessionStatus.AWAITING_INPUT.value:
                outcome = (
                    GuestSessionOutcome.EXPIRED
                    if row.status == GuestSessionStatus.EXPIRED.value
                    else GuestSessionOutcome.STALE
                )
                return self._decision(outcome, row, False)
            row.status = GuestSessionStatus.PROCESSING.value
            row.consumed_update_id = update_id
            row.consumed_message_id = message_id
            row.usage_id = None
            row.updated_at = current
            row.expires_at = current + self.policy.reservation_ttl
            row.version += 1
            await session.flush()
            return self._decision(GuestSessionOutcome.CLAIMED, row, True)

    async def bind_reservation(
        self,
        *,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        reservation_token: str,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        access = _positive_id(access_version, "access_version")
        expected_version = _positive_id(session_version, "session_version")
        token = _token(reservation_token)
        current = _now(now)
        identity = await self.quota._usage_identity(token)
        if identity is None:
            return GuestSessionDecision(GuestSessionOutcome.STALE, None, False)
        usage_id, usage_user_id, quota_day = identity
        if usage_user_id != owner_id:
            return GuestSessionDecision(GuestSessionOutcome.STALE, None, False)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, quota_day)
            user = await self.quota._lock_user(session, owner_id)
            row = await self._locked_session(session, owner_id, target_chat)
            usage = await self.quota._locked_usage(session, usage_id, token)
            if user is None or user.access_tier != GUEST or user.access_version != access:
                return self._decision(GuestSessionOutcome.NOT_GUEST, row, False)
            if (
                row is None
                or usage is None
                or row.status != GuestSessionStatus.PROCESSING.value
                or row.version != expected_version
                or row.access_version != access
                or usage.user_id != owner_id
                or usage.demo_kind != row.demo_kind
                or usage.status != GuestUsageStatus.RESERVED.value
            ):
                return self._decision(GuestSessionOutcome.STALE, row, False)
            if _as_utc(usage.expires_at) <= current:
                usage.status = GuestUsageStatus.EXPIRED.value
                usage.completed_at = current
                self._retry_ready(row, current)
                await session.flush()
                return self._decision(GuestSessionOutcome.RETRY_READY, row, True)
            already_bound = await session.scalar(
                select(GuestDemoSession.id).where(
                    GuestDemoSession.usage_id == usage.id,
                    GuestDemoSession.id != row.id,
                )
            )
            if already_bound is not None:
                return self._decision(GuestSessionOutcome.STALE, row, False)
            row.usage_id = usage.id
            row.updated_at = current
            row.expires_at = _as_utc(usage.expires_at)
            row.version += 1
            await session.flush()
            return self._decision(GuestSessionOutcome.BOUND, row, True)

    async def begin_provider_call(
        self,
        *,
        reservation_token: str,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        now: datetime | None = None,
    ) -> GuestProviderStartDecision:
        token = _token(reservation_token)
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        access = _positive_id(access_version, "access_version")
        expected_version = _positive_id(session_version, "session_version")
        current = _now(now)
        identity = await self.quota._usage_identity(token)
        if identity is None or identity[1] != owner_id:
            return GuestProviderStartDecision(
                GuestProviderStartOutcome.STALE,
                None,
                None,
                False,
            )
        usage_id, _usage_user_id, _quota_day = identity
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            user = await self.quota._lock_user(session, owner_id)
            usage = await self.quota._locked_usage(session, usage_id, token)
            row = await self._locked_session(session, owner_id, target_chat)
            binding_matches = (
                usage is not None
                and row is not None
                and row.status == GuestSessionStatus.PROCESSING.value
                and row.version == expected_version
                and row.access_version == access
                and row.usage_id == usage.id
                and usage.user_id == owner_id
                and row.demo_kind == usage.demo_kind
            )
            if not binding_matches:
                return self._provider_start_decision(
                    GuestProviderStartOutcome.STALE,
                    row,
                    usage.provider_started_at if usage is not None else None,
                    False,
                )
            if user is None or user.access_tier != GUEST:
                return self._provider_start_decision(
                    GuestProviderStartOutcome.NOT_GUEST,
                    row,
                    usage.provider_started_at,
                    False,
                )
            if user.access_version != access:
                return self._provider_start_decision(
                    GuestProviderStartOutcome.ACCESS_CHANGED,
                    row,
                    usage.provider_started_at,
                    False,
                )
            if usage.status != GuestUsageStatus.RESERVED.value:
                return self._provider_start_decision(
                    GuestProviderStartOutcome.STALE,
                    row,
                    usage.provider_started_at,
                    False,
                )
            if _as_utc(usage.expires_at) <= current:
                usage.status = GuestUsageStatus.EXPIRED.value
                usage.completed_at = current
                self._retry_ready(row, current)
                await session.flush()
                return self._provider_start_decision(
                    GuestProviderStartOutcome.EXPIRED,
                    row,
                    usage.provider_started_at,
                    True,
                )
            if usage.provider_started_at is not None:
                return self._provider_start_decision(
                    GuestProviderStartOutcome.ALREADY_STARTED,
                    row,
                    usage.provider_started_at,
                    False,
                )
            usage.provider_started_at = current
            await session.flush()
            return self._provider_start_decision(
                GuestProviderStartOutcome.STARTED,
                row,
                current,
                True,
            )

    async def resolve_reservation_denial(
        self,
        *,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        denial_reason: GuestQuotaDenialReason | str,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        access = _positive_id(access_version, "access_version")
        expected_version = _positive_id(session_version, "session_version")
        reason = GuestQuotaDenialReason(denial_reason)
        current = _now(now)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            await self.quota._lock_user(session, owner_id)
            row = await self._locked_session(session, owner_id, target_chat)
            if (
                row is None
                or row.access_version != access
                or row.version != expected_version
                or row.status != GuestSessionStatus.PROCESSING.value
            ):
                return self._decision(GuestSessionOutcome.STALE, row, False)
            if reason in {
                GuestQuotaDenialReason.DISABLED,
                GuestQuotaDenialReason.NOT_GUEST,
                GuestQuotaDenialReason.LIFETIME_EXHAUSTED,
            }:
                self._terminalize(row, GuestSessionStatus.CANCELLED, current)
                outcome = GuestSessionOutcome.CANCELLED
            else:
                self._retry_ready(row, current)
                outcome = GuestSessionOutcome.RETRY_READY
            await session.flush()
            return self._decision(outcome, row, True)

    async def fail_processing(
        self,
        *,
        reservation_token: str,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        now: datetime | None = None,
    ) -> GuestCompletionDecision:
        token = _token(reservation_token)
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        access = _positive_id(access_version, "access_version")
        expected_version = _positive_id(session_version, "session_version")
        current = _now(now)
        identity = await self.quota._usage_identity(token)
        if identity is None or identity[1] != owner_id:
            return GuestCompletionDecision(GuestReservationOutcome.STALE, None, False)
        usage_id, _usage_user_id, quota_day = identity
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, quota_day)
            user = await self.quota._lock_user(session, owner_id)
            usage = await self.quota._locked_usage(session, usage_id, token)
            row = await self._locked_session(session, owner_id, target_chat)
            if usage is None:
                return GuestCompletionDecision(GuestReservationOutcome.STALE, None, False)
            if usage.status == GuestUsageStatus.FAILED.value:
                return self._completion(GuestReservationOutcome.FAILED, row, False)
            if usage.status != GuestUsageStatus.RESERVED.value:
                return self._completion(GuestReservationOutcome.STALE, row, False)
            if _as_utc(usage.expires_at) <= current:
                usage.status = GuestUsageStatus.EXPIRED.value
                outcome = GuestReservationOutcome.EXPIRED
            else:
                usage.status = GuestUsageStatus.FAILED.value
                outcome = GuestReservationOutcome.FAILED
            usage.completed_at = current
            if (
                row is not None
                and row.status == GuestSessionStatus.PROCESSING.value
                and row.version == expected_version
                and row.usage_id == usage.id
            ):
                if (
                    user is not None
                    and user.access_tier == GUEST
                    and user.access_version == access
                    and row.access_version == access
                ):
                    self._retry_ready(row, current)
                else:
                    self._terminalize(row, GuestSessionStatus.CANCELLED, current)
            await session.flush()
            return self._completion(outcome, row, True)

    async def complete_with_result(
        self,
        *,
        reservation_token: str,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        result_payload: dict[str, Any],
        now: datetime | None = None,
    ) -> GuestCompletionDecision:
        token = _token(reservation_token)
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        access = _positive_id(access_version, "access_version")
        expected_version = _positive_id(session_version, "session_version")
        payload = self._result_payload(result_payload)
        current = _now(now)
        identity = await self.quota._usage_identity(token)
        if identity is None or identity[1] != owner_id:
            return GuestCompletionDecision(GuestReservationOutcome.STALE, None, False)
        usage_id, _usage_user_id, quota_day = identity
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, quota_day)
            user = await self.quota._lock_user(session, owner_id)
            usage = await self.quota._locked_usage(session, usage_id, token)
            row = await self._locked_session(session, owner_id, target_chat)
            if (
                usage is None
                or usage.status != GuestUsageStatus.RESERVED.value
                or usage.provider_started_at is None
                or _as_utc(usage.expires_at) <= current
            ):
                return self._completion(GuestReservationOutcome.STALE, row, False)
            session_matches = (
                row is not None
                and row.status == GuestSessionStatus.PROCESSING.value
                and row.version == expected_version
                and row.usage_id == usage.id
                and row.demo_kind == usage.demo_kind
            )
            if not session_matches:
                return self._completion(GuestReservationOutcome.STALE, row, False)

            usage.status = GuestUsageStatus.SUCCEEDED.value
            usage.completed_at = current
            access_matches = (
                user is not None
                and user.access_tier == GUEST
                and user.access_version == access
                and row.access_version == access
            )
            if access_matches:
                row.status = GuestSessionStatus.RESULT_READY.value
                row.result_payload = payload
                row.result_expires_at = current + self.policy.result_ttl
                row.expires_at = current + self.policy.result_ttl
                row.updated_at = current
                row.version += 1
                outcome = GuestReservationOutcome.SUCCEEDED
            else:
                self._terminalize(row, GuestSessionStatus.CANCELLED, current)
                outcome = GuestReservationOutcome.ACCESS_CHANGED
            await session.flush()
            return self._completion(outcome, row, True)

    async def pending_result(
        self,
        *,
        user_id: int,
        chat_id: int,
        access_version: int,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        access = _positive_id(access_version, "access_version")
        current = _now(now)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            user = await self.quota._lock_user(session, owner_id)
            row = await self._locked_session(session, owner_id, target_chat)
            if user is None or user.access_tier != GUEST or user.access_version != access:
                changed = False
                if row is not None and row.status == GuestSessionStatus.RESULT_READY.value:
                    self._terminalize(row, GuestSessionStatus.CANCELLED, current)
                    await session.flush()
                    changed = True
                return self._decision(GuestSessionOutcome.NOT_GUEST, row, changed)
            if row is None or row.access_version != access:
                return self._decision(GuestSessionOutcome.STALE, row, False)
            expired = self._expire_session(row, current)
            if expired:
                await session.flush()
                return self._decision(GuestSessionOutcome.EXPIRED, row, True)
            if row.status != GuestSessionStatus.RESULT_READY.value:
                return self._decision(GuestSessionOutcome.STALE, row, False)
            return self._decision(GuestSessionOutcome.RESULT_READY, row, False)

    async def mark_delivered(
        self,
        *,
        user_id: int,
        chat_id: int,
        session_version: int,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        expected_version = _positive_id(session_version, "session_version")
        current = _now(now)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            await self.quota._lock_user(session, owner_id)
            row = await self._locked_session(session, owner_id, target_chat)
            if row is None:
                return self._decision(GuestSessionOutcome.STALE, None, False)
            if row.status == GuestSessionStatus.COMPLETED.value:
                return self._decision(GuestSessionOutcome.COMPLETED, row, False)
            if (
                row.status != GuestSessionStatus.RESULT_READY.value
                or row.version != expected_version
            ):
                return self._decision(GuestSessionOutcome.STALE, row, False)
            self._terminalize(row, GuestSessionStatus.COMPLETED, current)
            await session.flush()
            return self._decision(GuestSessionOutcome.COMPLETED, row, True)

    async def cancel(
        self,
        *,
        user_id: int,
        chat_id: int,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        current = _now(now)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            await self.quota._lock_user(session, owner_id)
            row = await self._locked_session(session, owner_id, target_chat)
            if row is None:
                return self._decision(GuestSessionOutcome.STALE, None, False)
            status = GuestSessionStatus(row.status)
            if status is GuestSessionStatus.PROCESSING:
                return self._decision(GuestSessionOutcome.IN_PROGRESS, row, False)
            if status is GuestSessionStatus.CANCELLED:
                return self._decision(GuestSessionOutcome.CANCELLED, row, False)
            if status is GuestSessionStatus.COMPLETED:
                return self._decision(GuestSessionOutcome.COMPLETED, row, False)
            if status is GuestSessionStatus.EXPIRED:
                return self._decision(GuestSessionOutcome.EXPIRED, row, False)
            self._terminalize(row, GuestSessionStatus.CANCELLED, current)
            await session.flush()
            return self._decision(GuestSessionOutcome.CANCELLED, row, True)

    async def cleanup_expired(
        self,
        *,
        user_id: int,
        chat_id: int,
        now: datetime | None = None,
    ) -> GuestSessionDecision:
        owner_id = _positive_id(user_id, "user_id")
        target_chat = _positive_id(chat_id, "chat_id")
        current = _now(now)
        async with self.db.session() as session:
            await self.quota._lock_quota_day(session, current.date())
            await self.quota._lock_user(session, owner_id)
            row = await self._locked_session(session, owner_id, target_chat)
            if row is None:
                return self._decision(GuestSessionOutcome.STALE, None, False)
            if not self._expire_session(row, current):
                return self._decision(GuestSessionOutcome.STALE, row, False)
            await session.flush()
            return self._decision(GuestSessionOutcome.EXPIRED, row, True)

    @staticmethod
    async def _locked_session(
        session: AsyncSession,
        user_id: int,
        chat_id: int,
    ) -> GuestDemoSession | None:
        return await session.scalar(
            select(GuestDemoSession)
            .where(
                GuestDemoSession.user_id == user_id,
                GuestDemoSession.chat_id == chat_id,
            )
            .with_for_update()
        )

    def _result_payload(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("guest result payload must be a non-empty JSON object")
        if _contains_forbidden_result_key(value):
            raise ValueError("guest result payload contains a forbidden field")
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("guest result payload must be valid JSON") from exc
        if len(serialized) > self.policy.max_result_payload_bytes:
            raise ValueError("guest result payload exceeds the size limit")
        normalized = json.loads(serialized.decode("utf-8"))
        if not isinstance(normalized, dict):
            raise ValueError("guest result payload must be a JSON object")
        return normalized

    @staticmethod
    def _expire_session(row: GuestDemoSession, current: datetime) -> bool:
        result_expired = (
            row.status == GuestSessionStatus.RESULT_READY.value
            and row.result_expires_at is not None
            and _as_utc(row.result_expires_at) <= current
        )
        flow_expired = (
            row.status
            in {
                GuestSessionStatus.AWAITING_INPUT.value,
                GuestSessionStatus.PROCESSING.value,
            }
            and _as_utc(row.expires_at) <= current
        )
        if not result_expired and not flow_expired:
            return False
        GuestSessionService._terminalize(row, GuestSessionStatus.EXPIRED, current)
        return True

    def _retry_ready(self, row: GuestDemoSession, current: datetime) -> None:
        row.status = GuestSessionStatus.AWAITING_INPUT.value
        row.usage_id = None
        row.result_payload = None
        row.result_expires_at = None
        row.updated_at = current
        row.expires_at = current + self.policy.input_ttl
        row.version += 1

    @staticmethod
    def _terminalize(
        row: GuestDemoSession,
        status: GuestSessionStatus,
        current: datetime,
    ) -> None:
        row.status = status.value
        row.result_payload = None
        row.result_expires_at = None
        row.updated_at = current
        row.expires_at = max(_as_utc(row.expires_at), current + timedelta(microseconds=1))
        row.version += 1

    @classmethod
    def _snapshot(cls, row: GuestDemoSession) -> GuestSessionSnapshot:
        payload = None
        if row.result_payload is not None:
            payload = json.loads(json.dumps(row.result_payload, ensure_ascii=False))
        return GuestSessionSnapshot(
            session_id=row.id,
            user_id=row.user_id,
            chat_id=row.chat_id,
            access_version=row.access_version,
            demo_kind=GuestDemoKind(row.demo_kind),
            status=GuestSessionStatus(row.status),
            prompt_message_id=row.prompt_message_id,
            consumed_update_id=row.consumed_update_id,
            consumed_message_id=row.consumed_message_id,
            usage_id=row.usage_id,
            result_payload=payload,
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
            expires_at=_as_utc(row.expires_at),
            result_expires_at=(
                _as_utc(row.result_expires_at) if row.result_expires_at is not None else None
            ),
            version=row.version,
        )

    @classmethod
    def _decision(
        cls,
        outcome: GuestSessionOutcome,
        row: GuestDemoSession | None,
        changed: bool,
    ) -> GuestSessionDecision:
        return GuestSessionDecision(
            outcome=outcome,
            session=cls._snapshot(row) if row is not None else None,
            changed=changed,
        )

    @classmethod
    def _completion(
        cls,
        outcome: GuestReservationOutcome,
        row: GuestDemoSession | None,
        changed: bool,
    ) -> GuestCompletionDecision:
        return GuestCompletionDecision(
            outcome=outcome,
            session=cls._snapshot(row) if row is not None else None,
            changed=changed,
        )

    @classmethod
    def _provider_start_decision(
        cls,
        outcome: GuestProviderStartOutcome,
        row: GuestDemoSession | None,
        provider_started_at: datetime | None,
        changed: bool,
    ) -> GuestProviderStartDecision:
        return GuestProviderStartDecision(
            outcome=outcome,
            session=cls._snapshot(row) if row is not None else None,
            provider_started_at=(
                _as_utc(provider_started_at) if provider_started_at is not None else None
            ),
            changed=changed,
        )
