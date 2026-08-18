from __future__ import annotations

import asyncio
import re
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any

from .access import ADMIN, is_full_access_tier


class WeeklyReviewIntent(StrEnum):
    OPEN = "open"
    START = "start"
    EDIT_FOCUS = "edit_focus"
    VIEW = "view"
    BACK = "back"
    SKIP = "skip"
    CANCEL = "cancel"
    NONE = "none"


class WeeklyReviewDisposition(StrEnum):
    PASS = "pass"
    RENDER_CURRENT = "render_current"
    PROMPT_INPUT = "prompt_input"
    SHOW_FOCUS = "show_focus"
    RETURN_ROOT = "return_root"
    CANCEL = "cancel"
    REPROMPT = "reprompt"
    EXTRACT = "extract"
    ABSORB = "absorb"


@dataclass(frozen=True, slots=True)
class WeeklyReviewDecision:
    intent: WeeklyReviewIntent
    disposition: WeeklyReviewDisposition


_WEEKLY_OPEN_PHRASES = frozenset({"обзор недели"})
_WEEKLY_START_PHRASES = frozenset(
    {
        "давай скорректируем систему",
        "хочу скорректировать систему",
        "провести обзор недели",
        "спланируем неделю",
        "спланировать неделю",
        "начать обзор недели",
    }
)
_WEEKLY_EDIT_PHRASES = frozenset(
    {
        "фокус на неделю",
        "изменить фокус недели",
        "скорректировать фокус недели",
    }
)
_WEEKLY_VIEW_PHRASES = frozenset({"покажи фокус недели"})
_WEEKLY_BACK_PHRASES = frozenset({"назад"})
_WEEKLY_SKIP_PHRASES = frozenset({"пропустить"})
_WEEKLY_CANCEL_PHRASES = frozenset({"отменить"})
_WEEKLY_NON_ANSWERS = frozenset(
    {
        "не знаю",
        "не знаю пока",
        "пока не знаю",
        "затрудняюсь ответить",
    }
)
_WEEK_COMMAND = re.compile(r"^/week(?:@[a-z0-9_]{5,32})?$")


def normalize_weekly_review_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return re.sub(r"[.!?…]+$", "", " ".join(normalized.split())).strip()


def classify_weekly_review_intent(text: str) -> WeeklyReviewIntent:
    normalized = normalize_weekly_review_text(text)
    if _WEEK_COMMAND.fullmatch(normalized) or normalized in _WEEKLY_OPEN_PHRASES:
        return WeeklyReviewIntent.OPEN
    if normalized in _WEEKLY_START_PHRASES:
        return WeeklyReviewIntent.START
    if normalized in _WEEKLY_EDIT_PHRASES:
        return WeeklyReviewIntent.EDIT_FOCUS
    if normalized in _WEEKLY_VIEW_PHRASES:
        return WeeklyReviewIntent.VIEW
    if normalized in _WEEKLY_BACK_PHRASES:
        return WeeklyReviewIntent.BACK
    if normalized in _WEEKLY_SKIP_PHRASES:
        return WeeklyReviewIntent.SKIP
    if normalized in _WEEKLY_CANCEL_PHRASES:
        return WeeklyReviewIntent.CANCEL
    return WeeklyReviewIntent.NONE


def reduce_weekly_review_input(phase: str, text: str) -> WeeklyReviewDecision:
    """Return the shared text/STT action before any extraction or provider call."""

    phase_value = str(getattr(phase, "value", phase))
    intent = classify_weekly_review_intent(text)
    if phase_value == "processing":
        return WeeklyReviewDecision(intent, WeeklyReviewDisposition.ABSORB)
    if phase_value in {"preview", "delete_preview"}:
        return WeeklyReviewDecision(intent, WeeklyReviewDisposition.RENDER_CURRENT)
    if phase_value == "root":
        if intent in {WeeklyReviewIntent.OPEN, WeeklyReviewIntent.BACK}:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.RENDER_CURRENT)
        if intent in {WeeklyReviewIntent.START, WeeklyReviewIntent.EDIT_FOCUS}:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.PROMPT_INPUT)
        if intent is WeeklyReviewIntent.VIEW:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.SHOW_FOCUS)
        if intent in {WeeklyReviewIntent.SKIP, WeeklyReviewIntent.CANCEL}:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.CANCEL)
        return WeeklyReviewDecision(intent, WeeklyReviewDisposition.RENDER_CURRENT)
    if phase_value == "awaiting_input":
        if intent in {WeeklyReviewIntent.START, WeeklyReviewIntent.EDIT_FOCUS}:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.REPROMPT)
        if intent in {
            WeeklyReviewIntent.OPEN,
            WeeklyReviewIntent.BACK,
            WeeklyReviewIntent.SKIP,
        }:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.RETURN_ROOT)
        if intent is WeeklyReviewIntent.CANCEL:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.CANCEL)
        if intent is WeeklyReviewIntent.VIEW:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.SHOW_FOCUS)
        if normalize_weekly_review_text(text) in _WEEKLY_NON_ANSWERS:
            return WeeklyReviewDecision(intent, WeeklyReviewDisposition.REPROMPT)
        return WeeklyReviewDecision(intent, WeeklyReviewDisposition.EXTRACT)
    return WeeklyReviewDecision(intent, WeeklyReviewDisposition.PASS)


@dataclass(frozen=True, slots=True)
class WeeklyReviewPolicy:
    """Single fail-closed rollout policy for every weekly-review surface."""

    enabled: bool
    admin_only: bool = True

    def allows_tier(self, tier: str | None) -> bool:
        if not self.enabled or tier is None or not is_full_access_tier(tier):
            return False
        return not self.admin_only or tier == ADMIN

    def allows_actor(
        self,
        actor: Any | None,
        *,
        expected_access_version: int | None = None,
    ) -> bool:
        if actor is None or not self.allows_tier(getattr(actor, "access_tier", None)):
            return False
        access_version = getattr(actor, "access_version", None)
        if not isinstance(access_version, int) or access_version <= 0:
            return False
        return expected_access_version is None or access_version == expected_access_version


@dataclass(frozen=True, slots=True)
class WeeklyReviewCapability:
    token: str
    screen_id: str
    action: str
    owner_id: int
    telegram_user_id: int
    chat_id: int
    canonical_message_id: int
    access_version: int
    week_start: date
    scheduled: bool
    session_public_id: str | None
    session_version: int | None
    screen_order: int
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class WeeklyReviewScreen:
    screen_id: str
    session_public_id: str
    screen_order: int
    expires_at: datetime


class WeeklyReviewCapabilityStore:
    """Bounded, process-local opaque actions over durable weekly state.

    Durable recovery is provided by ``/week``.  Tokens deliberately do not
    survive a restart: an old Telegram button must fail closed instead of
    reconstructing authority from callback data.
    """

    def __init__(
        self,
        *,
        ttl: timedelta = timedelta(minutes=30),
        max_capabilities: int = 5_000,
    ) -> None:
        if ttl <= timedelta(0) or ttl > timedelta(hours=2):
            raise ValueError("weekly capability ttl must be between 1 second and 2 hours")
        if not 1 <= max_capabilities <= 20_000:
            raise ValueError("weekly capability limit must be between 1 and 20000")
        self.ttl = ttl
        self.max_capabilities = max_capabilities
        self._capabilities: dict[str, WeeklyReviewCapability] = {}
        self._screens: dict[str, WeeklyReviewScreen] = {}
        self._canonical_generations: dict[
            tuple[int, int, int],
            tuple[int, datetime],
        ] = {}
        self._next_screen_order = 0
        self._lock = asyncio.Lock()

    async def issue(
        self,
        *,
        actions: tuple[str, ...],
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
        access_version: int,
        week_start: date,
        scheduled: bool = False,
        session_public_id: str | None = None,
        session_version: int | None = None,
        replace_session: bool = True,
        now: datetime | None = None,
    ) -> dict[str, str]:
        if not actions or len(actions) > 12 or len(set(actions)) != len(actions):
            raise ValueError("weekly actions must contain 1..12 unique values")
        if len(actions) > self.max_capabilities:
            raise ValueError("weekly action batch exceeds capability limit")
        if min(owner_id, telegram_user_id, chat_id, canonical_message_id, access_version) <= 0:
            raise ValueError("weekly capability binding is invalid")
        if not isinstance(week_start, date) or isinstance(week_start, datetime):
            raise ValueError("weekly capability week is invalid")
        if not isinstance(scheduled, bool):
            raise ValueError("weekly capability schedule binding is invalid")
        if not isinstance(replace_session, bool):
            raise ValueError("weekly capability replacement policy is invalid")
        if (session_public_id is None) != (session_version is None):
            raise ValueError("weekly session binding is incomplete")
        if session_version is not None and session_version <= 0:
            raise ValueError("weekly session version must be positive")
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            screen_order = self._next_screen_order + 1
            screen_id = secrets.token_urlsafe(12)
            expires_at = current + self.ttl
            staged_screen = (
                WeeklyReviewScreen(
                    screen_id=screen_id,
                    session_public_id=session_public_id,
                    screen_order=screen_order,
                    expires_at=expires_at,
                )
                if session_public_id is not None
                else None
            )
            staged_capabilities: dict[str, WeeklyReviewCapability] = {}
            result: dict[str, str] = {}
            for action in actions:
                token = secrets.token_urlsafe(18)
                staged_capabilities[token] = WeeklyReviewCapability(
                    token=token,
                    screen_id=screen_id,
                    action=action,
                    owner_id=owner_id,
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    canonical_message_id=canonical_message_id,
                    access_version=access_version,
                    week_start=week_start,
                    scheduled=scheduled,
                    session_public_id=session_public_id,
                    session_version=session_version,
                    screen_order=screen_order,
                    expires_at=expires_at,
                )
                result[action] = token
            if len(staged_capabilities) != len(actions) or any(
                token in self._capabilities for token in staged_capabilities
            ):
                raise RuntimeError("weekly capability token collision")

            if session_public_id is not None and replace_session:
                self._drop_session_locked(session_public_id)
            while len(self._capabilities) + len(actions) > self.max_capabilities:
                oldest_token = min(
                    self._capabilities,
                    key=lambda token: self._capabilities[token].expires_at,
                )
                self._drop_screen_locked(self._capabilities[oldest_token].screen_id)
            self._next_screen_order = screen_order
            if staged_screen is not None:
                self._screens[screen_id] = staged_screen
            self._capabilities.update(staged_capabilities)
            canonical_key = (telegram_user_id, chat_id, canonical_message_id)
            self._canonical_generations[canonical_key] = (
                screen_order,
                expires_at,
            )
            while len(self._canonical_generations) > self.max_capabilities:
                live_canonicals = {
                    (
                        capability.telegram_user_id,
                        capability.chat_id,
                        capability.canonical_message_id,
                    )
                    for capability in self._capabilities.values()
                }
                oldest_key = min(
                    (key for key in self._canonical_generations if key not in live_canonicals),
                    key=lambda key: (
                        self._canonical_generations[key][1],
                        self._canonical_generations[key][0],
                    ),
                )
                self._canonical_generations.pop(oldest_key, None)
            return result

    async def stage_screen(
        self,
        session_public_id: str,
        *,
        now: datetime | None = None,
    ) -> WeeklyReviewScreen:
        if not isinstance(session_public_id, str) or not session_public_id:
            raise ValueError("weekly session binding is invalid")
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            while len(self._screens) >= self.max_capabilities:
                oldest = min(
                    self._screens.values(),
                    key=lambda screen: (screen.expires_at, screen.screen_order),
                )
                self._drop_screen_locked(oldest.screen_id)
            screen = WeeklyReviewScreen(
                screen_id=secrets.token_urlsafe(12),
                session_public_id=session_public_id,
                screen_order=self._new_screen_order_locked(),
                expires_at=current + self.ttl,
            )
            self._screens[screen.screen_id] = screen
            return screen

    async def screen_for_tokens(
        self,
        tokens: tuple[str, ...],
        *,
        now: datetime | None = None,
    ) -> WeeklyReviewScreen | None:
        if not tokens:
            return None
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            capabilities = tuple(self._capabilities.get(token) for token in tokens)
            if any(capability is None for capability in capabilities):
                return None
            first = capabilities[0]
            assert first is not None
            if first.session_public_id is None or any(
                capability is None
                or capability.screen_id != first.screen_id
                or capability.session_public_id != first.session_public_id
                or capability.screen_order != first.screen_order
                for capability in capabilities
            ):
                return None
            return self._screens.get(first.screen_id)

    async def screen_is_live(
        self,
        expected: WeeklyReviewScreen,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            return self._screens.get(expected.screen_id) == expected

    async def activate_screen(
        self,
        expected: WeeklyReviewScreen,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Publish one rendered screen without revoking newer staged screens."""

        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            if self._screens.get(expected.screen_id) != expected:
                return False
            obsolete = tuple(
                screen.screen_id
                for screen in self._screens.values()
                if screen.session_public_id == expected.session_public_id
                and screen.screen_order <= expected.screen_order
                and screen.screen_id != expected.screen_id
            )
            for screen_id in obsolete:
                self._drop_screen_locked(screen_id)
            return True

    async def revoke_screen(
        self,
        expected: WeeklyReviewScreen,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            if self._screens.get(expected.screen_id) != expected:
                return False
            self._drop_screen_locked(expected.screen_id)
            return True

    async def claim(
        self,
        token: str,
        *,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int | None,
        now: datetime | None = None,
    ) -> WeeklyReviewCapability | None:
        current = self._utc(now) if now is not None else None
        capability = await self.peek(
            token,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            canonical_message_id=canonical_message_id,
            now=current,
        )
        if capability is None or not await self.consume(capability, now=current):
            return None
        return capability

    async def peek(
        self,
        token: str,
        *,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int | None,
        now: datetime | None = None,
    ) -> WeeklyReviewCapability | None:
        if not isinstance(token, str) or not token or len(token) > 40:
            return None
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
                # Wrong identity/canonical must not spend the owner's token.
                return None
            return capability

    async def consume(
        self,
        expected: WeeklyReviewCapability,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Atomically spend the exact peeked screen generation."""

        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            stored = self._capabilities.get(expected.token)
            if stored is None or stored != expected or stored.expires_at <= current:
                return False
            # One valid control wins the whole rendered screen.  The private
            # screen id prevents a stale replacement token from spending a
            # fresh screen that reuses the same Telegram canonical.
            self._drop_screen_locked(expected.screen_id)
            return True

    async def revoke_session(self, session_public_id: str) -> None:
        async with self._lock:
            self._drop_session_locked(session_public_id)

    async def revoke_tokens(self, tokens: tuple[str, ...]) -> None:
        """Retire an exact set of issued capabilities without touching replacements."""

        async with self._lock:
            for token in tokens:
                self._capabilities.pop(token, None)

    async def revoke_tokens_if_current_screen(
        self,
        tokens: tuple[str, ...],
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
        access_version: int,
        week_start: date,
        scheduled: bool,
        now: datetime | None = None,
    ) -> bool:
        """Retire one exact screen and report whether it still owned the canonical."""

        if not tokens:
            return False
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            capabilities = tuple(self._capabilities.get(token) for token in tokens)
            first = capabilities[0]
            exact = bool(
                first is not None
                and all(
                    capability is not None
                    and capability.screen_id == first.screen_id
                    and capability.screen_order == first.screen_order
                    and capability.owner_id == owner_id
                    and capability.telegram_user_id == telegram_user_id
                    and capability.chat_id == chat_id
                    and capability.canonical_message_id == canonical_message_id
                    and capability.access_version == access_version
                    and capability.week_start == week_start
                    and capability.scheduled is scheduled
                    for capability in capabilities
                )
            )
            generation = self._canonical_generations.get(
                (telegram_user_id, chat_id, canonical_message_id)
            )
            latest = bool(
                exact
                and first is not None
                and generation is not None
                and generation[0] == first.screen_order
            )
            for token in tokens:
                self._capabilities.pop(token, None)
            return latest

    async def cleanup(self, *, now: datetime | None = None) -> int:
        current = self._utc(now)
        async with self._lock:
            before = len(self._capabilities)
            self._cleanup_locked(current)
            return before - len(self._capabilities)

    def _cleanup_locked(self, now: datetime) -> None:
        for token, capability in tuple(self._capabilities.items()):
            if capability.expires_at <= now:
                self._capabilities.pop(token, None)
        for screen_id, screen in tuple(self._screens.items()):
            if screen.expires_at <= now:
                self._drop_screen_locked(screen_id)
        for key, (_screen_order, expires_at) in tuple(self._canonical_generations.items()):
            if expires_at <= now:
                self._canonical_generations.pop(key, None)

    def _drop_session_locked(self, session_public_id: str) -> None:
        for screen_id, screen in tuple(self._screens.items()):
            if screen.session_public_id == session_public_id:
                self._drop_screen_locked(screen_id)
        for token, capability in tuple(self._capabilities.items()):
            if capability.session_public_id == session_public_id:
                self._capabilities.pop(token, None)

    def _drop_screen_locked(self, screen_id: str) -> None:
        for token, capability in tuple(self._capabilities.items()):
            if capability.screen_id == screen_id:
                self._capabilities.pop(token, None)
        self._screens.pop(screen_id, None)

    def _new_screen_order_locked(self) -> int:
        self._next_screen_order += 1
        return self._next_screen_order

    @staticmethod
    def _utc(value: datetime | None) -> datetime:
        current = value or datetime.now(UTC)
        if current.tzinfo is None:
            return current.replace(tzinfo=UTC)
        return current.astimezone(UTC)


__all__ = [
    "WeeklyReviewCapability",
    "WeeklyReviewCapabilityStore",
    "WeeklyReviewDecision",
    "WeeklyReviewDisposition",
    "WeeklyReviewIntent",
    "WeeklyReviewPolicy",
    "WeeklyReviewScreen",
    "classify_weekly_review_intent",
    "normalize_weekly_review_text",
    "reduce_weekly_review_input",
]
