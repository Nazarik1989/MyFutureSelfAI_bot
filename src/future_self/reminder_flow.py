from __future__ import annotations

import asyncio
import hashlib
import secrets
import unicodedata
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Literal

from .reminder_intent import (
    ReminderIntentResult,
    ReminderIntentStatus,
    ReminderScheduleKind,
    ReminderTimezoneSource,
)

ReminderFlowAction = Literal[
    "today",
    "tomorrow",
    "choose_date",
    "daily",
    "edit",
    "edit_when",
    "edit_time",
    "edit_title",
    "confirm",
    "retry_timezone",
    "cancel",
]


class ReminderFlowPhase(StrEnum):
    WHEN = "when"
    DATE = "date"
    TIME = "time"
    TITLE = "title"
    PREVIEW = "preview"
    EDIT = "edit"
    PAST = "past"
    INVALID = "invalid"
    TIMEZONE_RESOLVING = "timezone_resolving"
    TIMEZONE_CLARIFY = "timezone_clarify"
    TIMEZONE_RETRY = "timezone_retry"


@dataclass(frozen=True, slots=True)
class ReminderFlowSession:
    id: str
    owner_id: int
    telegram_user_id: int
    chat_id: int
    access_version: int
    version: int
    canonical_message_id: int | None
    title: str | None
    schedule_kind: ReminderScheduleKind | None
    local_date: date | None
    local_time: time | None
    timezone: str
    timezone_source: ReminderTimezoneSource
    phase: ReminderFlowPhase
    expires_at: datetime
    profile_timezone: str = "Europe/Moscow"
    timezone_fragment_fingerprint: str | None = None
    relative_day_offset: int | None = None
    calendar_anchor_utc: datetime | None = None
    weekly_candidate_handoff: bool = False

    def parser_state(self) -> ReminderIntentResult:
        if self.phase is ReminderFlowPhase.WHEN:
            status = ReminderIntentStatus.NEEDS_WHEN
        elif self.phase is ReminderFlowPhase.TIME:
            status = ReminderIntentStatus.NEEDS_TIME
        elif self.phase is ReminderFlowPhase.TITLE:
            status = ReminderIntentStatus.NEEDS_TITLE
        elif self.phase is ReminderFlowPhase.PREVIEW:
            status = ReminderIntentStatus.COMPLETE
        else:
            status = ReminderIntentStatus.INVALID
        return ReminderIntentResult(
            status=status,
            schedule_kind=self.schedule_kind,
            title=self.title,
            local_time=self.local_time,
            local_date=self.local_date,
            timezone=self.timezone,
            timezone_source=self.timezone_source,
        )


@dataclass(frozen=True, slots=True)
class ReminderFlowCapability:
    token: str
    session_id: str
    session_version: int
    owner_id: int
    telegram_user_id: int
    chat_id: int
    canonical_message_id: int | None
    action: ReminderFlowAction
    expires_at: datetime


class ReminderFlowStore:
    """Bounded, process-local reminder draft sessions and single-use capabilities."""

    def __init__(
        self,
        *,
        ttl: timedelta = timedelta(minutes=20),
        max_sessions: int = 1_000,
    ):
        if ttl <= timedelta(0) or ttl > timedelta(hours=2):
            raise ValueError("reminder flow ttl must be between 1 second and 2 hours")
        if not 1 <= max_sessions <= 10_000:
            raise ValueError("max_sessions must be between 1 and 10000")
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._sessions: dict[tuple[int, int, int], ReminderFlowSession] = {}
        self._capabilities: dict[str, ReminderFlowCapability] = {}
        self._lock = asyncio.Lock()
        self._fingerprint_key = secrets.token_bytes(32)

    async def create(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        title: str | None,
        schedule_kind: ReminderScheduleKind | None,
        local_date: date | None,
        local_time: time | None,
        timezone: str,
        timezone_source: ReminderTimezoneSource,
        phase: ReminderFlowPhase,
        canonical_message_id: int | None = None,
        profile_timezone: str | None = None,
        timezone_fragment_fingerprint: str | None = None,
        relative_day_offset: int | None = None,
        calendar_anchor_utc: datetime | None = None,
        weekly_candidate_handoff: bool = False,
        now: datetime | None = None,
    ) -> ReminderFlowSession:
        current = self._utc(now)
        key = self._key(owner_id, telegram_user_id, chat_id)
        async with self._lock:
            self._cleanup_locked(current)
            self._drop_locked(key)
            while len(self._sessions) >= self.max_sessions:
                oldest_key = min(
                    self._sessions,
                    key=lambda item: self._sessions[item].expires_at,
                )
                self._drop_locked(oldest_key)
            session = ReminderFlowSession(
                id=secrets.token_hex(16),
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                version=1,
                canonical_message_id=canonical_message_id,
                title=self._title(title),
                schedule_kind=schedule_kind,
                local_date=local_date,
                local_time=self._time(local_time),
                timezone=timezone,
                timezone_source=timezone_source,
                phase=phase,
                expires_at=current + self.ttl,
                profile_timezone=profile_timezone or timezone,
                timezone_fragment_fingerprint=timezone_fragment_fingerprint,
                relative_day_offset=relative_day_offset,
                calendar_anchor_utc=self._utc(calendar_anchor_utc)
                if calendar_anchor_utc is not None
                else None,
                weekly_candidate_handoff=bool(weekly_candidate_handoff),
            )
            self._sessions[key] = session
            return session

    async def current(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        now: datetime | None = None,
    ) -> ReminderFlowSession | None:
        current = self._utc(now)
        key = self._key(owner_id, telegram_user_id, chat_id)
        async with self._lock:
            self._cleanup_locked(current)
            return self._sessions.get(key)

    async def get_exact(
        self,
        session: ReminderFlowSession,
        *,
        now: datetime | None = None,
    ) -> ReminderFlowSession | None:
        current = self._utc(now)
        key = self._key(session.owner_id, session.telegram_user_id, session.chat_id)
        async with self._lock:
            self._cleanup_locked(current)
            live = self._sessions.get(key)
            return live if self._same_generation(live, session) else None

    async def update(
        self,
        session: ReminderFlowSession,
        *,
        title: str | None | object = ...,
        schedule_kind: ReminderScheduleKind | None | object = ...,
        local_date: date | None | object = ...,
        local_time: time | None | object = ...,
        timezone: str | object = ...,
        timezone_source: ReminderTimezoneSource | object = ...,
        phase: ReminderFlowPhase | object = ...,
        canonical_message_id: int | None | object = ...,
        timezone_fragment_fingerprint: str | None | object = ...,
        relative_day_offset: int | None | object = ...,
        calendar_anchor_utc: datetime | None | object = ...,
        now: datetime | None = None,
    ) -> ReminderFlowSession | None:
        current = self._utc(now)
        key = self._key(session.owner_id, session.telegram_user_id, session.chat_id)
        async with self._lock:
            self._cleanup_locked(current)
            live = self._sessions.get(key)
            if not self._same_generation(live, session):
                return None
            values: dict[str, object] = {
                "version": live.version + 1,
                "expires_at": current + self.ttl,
            }
            if title is not ...:
                values["title"] = self._title(title if isinstance(title, str) else None)
            if schedule_kind is not ...:
                values["schedule_kind"] = schedule_kind
            if local_date is not ...:
                values["local_date"] = local_date
            if local_time is not ...:
                values["local_time"] = self._time(
                    local_time if isinstance(local_time, time) else None
                )
            if timezone is not ...:
                values["timezone"] = timezone
            if timezone_source is not ...:
                values["timezone_source"] = timezone_source
            if phase is not ...:
                values["phase"] = phase
            if canonical_message_id is not ...:
                values["canonical_message_id"] = canonical_message_id
            if timezone_fragment_fingerprint is not ...:
                values["timezone_fragment_fingerprint"] = timezone_fragment_fingerprint
            if relative_day_offset is not ...:
                values["relative_day_offset"] = relative_day_offset
            if calendar_anchor_utc is not ...:
                values["calendar_anchor_utc"] = (
                    self._utc(calendar_anchor_utc)
                    if isinstance(calendar_anchor_utc, datetime)
                    else None
                )
            updated = replace(live, **values)
            self._sessions[key] = updated
            self._drop_capabilities_locked(live.id)
            return updated

    async def issue(
        self,
        session: ReminderFlowSession,
        actions: tuple[ReminderFlowAction, ...],
        *,
        now: datetime | None = None,
    ) -> dict[ReminderFlowAction, str]:
        current = self._utc(now)
        if not actions or len(actions) > 10 or len(set(actions)) != len(actions):
            raise ValueError("actions must contain 1..10 unique values")
        key = self._key(session.owner_id, session.telegram_user_id, session.chat_id)
        async with self._lock:
            self._cleanup_locked(current)
            live = self._sessions.get(key)
            if not self._same_generation(live, session):
                return {}
            self._drop_capabilities_locked(live.id)
            result: dict[ReminderFlowAction, str] = {}
            for action in actions:
                token = secrets.token_urlsafe(18)
                capability = ReminderFlowCapability(
                    token=token,
                    session_id=live.id,
                    session_version=live.version,
                    owner_id=live.owner_id,
                    telegram_user_id=live.telegram_user_id,
                    chat_id=live.chat_id,
                    canonical_message_id=live.canonical_message_id,
                    action=action,
                    expires_at=live.expires_at,
                )
                self._capabilities[token] = capability
                result[action] = token
            return result

    async def claim(
        self,
        token: str,
        *,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int | None,
        now: datetime | None = None,
    ) -> tuple[ReminderFlowCapability, ReminderFlowSession] | None:
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            capability = self._capabilities.get(token)
            if (
                capability is None
                or capability.telegram_user_id != telegram_user_id
                or capability.chat_id != chat_id
                or capability.canonical_message_id != canonical_message_id
            ):
                return None
            key = self._key(
                capability.owner_id,
                capability.telegram_user_id,
                capability.chat_id,
            )
            live = self._sessions.get(key)
            if (
                live is None
                or live.id != capability.session_id
                or live.version != capability.session_version
            ):
                self._capabilities.pop(token, None)
                return None
            self._drop_capabilities_locked(live.id)
            return capability, live

    async def clear(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        session_id: str | None = None,
    ) -> bool:
        key = self._key(owner_id, telegram_user_id, chat_id)
        async with self._lock:
            live = self._sessions.get(key)
            if live is None or (session_id is not None and live.id != session_id):
                return False
            self._drop_locked(key)
            return True

    async def clear_exact(self, session: ReminderFlowSession) -> bool:
        """Clear only the exact immutable session generation."""

        key = self._key(session.owner_id, session.telegram_user_id, session.chat_id)
        async with self._lock:
            live = self._sessions.get(key)
            if not self._same_generation(live, session):
                return False
            self._drop_locked(key)
            return True

    async def cleanup(self, *, now: datetime | None = None) -> int:
        current = self._utc(now)
        async with self._lock:
            before = len(self._sessions)
            self._cleanup_locked(current)
            return before - len(self._sessions)

    def timezone_fragment_fingerprint(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        clean = " ".join(normalized.casefold().replace("ё", "е").split())
        if not clean:
            raise ValueError("timezone fragment must not be empty")
        return hashlib.blake2b(
            clean.encode("utf-8"),
            key=self._fingerprint_key,
            digest_size=16,
        ).hexdigest()

    def _cleanup_locked(self, now: datetime) -> None:
        for key, session in tuple(self._sessions.items()):
            if session.expires_at <= now:
                self._drop_locked(key)
        for token, capability in tuple(self._capabilities.items()):
            if capability.expires_at <= now:
                self._capabilities.pop(token, None)

    def _drop_locked(self, key: tuple[int, int, int]) -> None:
        session = self._sessions.pop(key, None)
        if session is not None:
            self._drop_capabilities_locked(session.id)

    def _drop_capabilities_locked(self, session_id: str) -> None:
        for token, capability in tuple(self._capabilities.items()):
            if capability.session_id == session_id:
                self._capabilities.pop(token, None)

    @staticmethod
    def _same_generation(
        live: ReminderFlowSession | None,
        expected: ReminderFlowSession,
    ) -> bool:
        return bool(
            live is not None
            and live.id == expected.id
            and live.version == expected.version
            and live.access_version == expected.access_version
            and live.canonical_message_id == expected.canonical_message_id
        )

    @staticmethod
    def _key(owner_id: int, telegram_user_id: int, chat_id: int) -> tuple[int, int, int]:
        if min(owner_id, telegram_user_id, chat_id) <= 0:
            raise ValueError("reminder flow owner and destination ids must be positive")
        return owner_id, telegram_user_id, chat_id

    @staticmethod
    def _title(value: str | None) -> str | None:
        if value is None:
            return None
        clean = " ".join(value.split()).strip()
        if not clean:
            return None
        return clean[:200]

    @staticmethod
    def _time(value: time | None) -> time | None:
        if value is None:
            return None
        if value.tzinfo is not None or value.second or value.microsecond:
            raise ValueError("local_time must be a naive minute-precision time")
        return value

    @staticmethod
    def _utc(value: datetime | None) -> datetime:
        current = value or datetime.now(UTC)
        return current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)


__all__ = [
    "ReminderFlowAction",
    "ReminderFlowCapability",
    "ReminderFlowPhase",
    "ReminderFlowSession",
    "ReminderFlowStore",
]
