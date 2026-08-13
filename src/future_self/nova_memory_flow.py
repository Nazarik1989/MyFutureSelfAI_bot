from __future__ import annotations

import asyncio
import re
import secrets
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from time import monotonic
from typing import Literal, cast

from .access import ACCESS_TIERS, FULL_ACCESS_TIERS, AccessTier
from .nova_memory import (
    NOVA_MEMORY_CATEGORIES,
    NovaMemoryCategory,
    normalize_nova_memory_content,
)

NOVA_MEMORY_FLOW_TTL_SECONDS = 15 * 60
NOVA_MEMORY_FLOW_MAX_SESSIONS = 128
NOVA_MEMORY_FLOW_MAX_CAPABILITIES = 32
NOVA_MEMORY_CALLBACK_PREFIX = "nmem:"

type NovaMemoryListFilter = NovaMemoryCategory | Literal["important"]

_ACTION_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{16,48}\Z")
_COLLECTION_REVISION_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_UNSET = object()


class NovaMemoryFlowPhase(StrEnum):
    ROOT = "root"
    LIST = "list"
    DETAIL = "detail"
    AWAITING_CREATE_CONTENT = "awaiting_create_content"
    CREATE_PREVIEW = "create_preview"
    AWAITING_UPDATE_CONTENT = "awaiting_update_content"
    UPDATE_PREVIEW = "update_preview"
    DELETE_PREVIEW = "delete_preview"
    DELETE_ALL_PREVIEW = "delete_all_preview"
    PROCESSING = "processing"


@dataclass(frozen=True, slots=True)
class NovaMemoryFlowSession:
    id: str
    owner_id: int
    telegram_user_id: int
    chat_id: int
    tier: AccessTier
    access_version: int
    version: int
    canonical_message_id: int | None
    phase: NovaMemoryFlowPhase
    created_at: float
    expires_at: float
    candidate_content: str | None = field(default=None, repr=False)
    candidate_category: NovaMemoryCategory | None = None
    candidate_important: bool = False
    item_public_id: str | None = field(default=None, repr=False)
    item_version: int | None = None
    list_filter: NovaMemoryListFilter | None = None
    page: int = 0
    collection_revision: str | None = field(default=None, repr=False)
    collection_count: int | None = None


@dataclass(frozen=True, slots=True)
class NovaMemoryCapability:
    token: str
    owner_id: int
    telegram_user_id: int
    chat_id: int
    tier: AccessTier
    access_version: int
    session_id: str
    session_version: int
    canonical_message_id: int
    action: str
    mutation: bool
    expires_at: float
    public_id: str | None = field(default=None, repr=False)
    expected_item_version: int | None = None
    expected_collection_revision: str | None = field(default=None, repr=False)
    list_filter: NovaMemoryListFilter | None = None
    page: int | None = None

    @property
    def callback_data(self) -> str:
        return f"{NOVA_MEMORY_CALLBACK_PREFIX}{self.token}"


@dataclass(frozen=True, slots=True)
class NovaMemoryCapabilityClaim:
    capability: NovaMemoryCapability
    session: NovaMemoryFlowSession


class NovaMemoryIntentKind(StrEnum):
    NONE = "none"
    CREATE = "create"
    AWAIT_CONTENT = "await_content"
    REMEMBER_THIS = "remember_this"
    OPEN = "open"
    DELETE_ALL = "delete_all"


@dataclass(frozen=True, slots=True)
class NovaMemoryIntentResult:
    kind: NovaMemoryIntentKind
    content: str | None = field(default=None, repr=False)
    category: NovaMemoryCategory | None = None
    important: bool = False

    @property
    def matched(self) -> bool:
        return self.kind is not NovaMemoryIntentKind.NONE


class NovaMemoryIntentClassifier:
    """Conservative local classifier shared by text and transcribed voice input."""

    _CONTENT_PATTERNS: tuple[tuple[re.Pattern[str], NovaMemoryCategory, bool], ...] = (
        (
            re.compile(
                r"(?:nova|нова)\s*,\s*запомни\s*,\s*как\s+со\s+мной\s+работать\s*:\s*(.+)\Z",
                re.IGNORECASE,
            ),
            "interaction",
            False,
        ),
        (
            re.compile(
                r"(?:nova|нова)\s*,\s*запомни\s+обо\s+мне\s*:\s*(.+)\Z",
                re.IGNORECASE,
            ),
            "about_me",
            False,
        ),
        (
            re.compile(
                r"(?:nova|нова)\s*,\s*запомни\s+мой\s+ориентир\s*:\s*(.+)\Z",
                re.IGNORECASE,
            ),
            "orientation",
            False,
        ),
        (
            re.compile(
                r"(?:nova|нова)\s*,\s*сохрани\s+в\s+важное\s*:\s*(.+)\Z",
                re.IGNORECASE,
            ),
            "about_me",
            True,
        ),
        (
            re.compile(r"(?:nova|нова)\s*,\s*запомни\s*:\s*(.+)\Z", re.IGNORECASE),
            "about_me",
            False,
        ),
        (
            re.compile(r"научи\s+nova\s*:\s*(.+)\Z", re.IGNORECASE),
            "about_me",
            False,
        ),
        (
            re.compile(r"запомни\s+для\s+nova\s*:\s*(.+)\Z", re.IGNORECASE),
            "about_me",
            False,
        ),
    )
    _AWAIT_CONTENT = re.compile(
        r"(?:nova|нова)\s*,\s*запомни\s*:?\s*[.!?]?\Z",
        re.IGNORECASE,
    )
    _REMEMBER_THIS = (
        re.compile(r"(?:nova|нова)\s*,\s*запомни\s+это\s*[.!?]?\Z", re.IGNORECASE),
        re.compile(r"сохрани\s+это\s+для\s+nova\s*[.!?]?\Z", re.IGNORECASE),
    )
    _DELETE_ALL = re.compile(
        r"(?:nova|нова)\s*,\s*забудь\s+вс[её]\s*[.!?]?\Z",
        re.IGNORECASE,
    )
    _OPEN = (
        re.compile(r"открой\s+мою\s+nova\s*[.!?]?\Z", re.IGNORECASE),
        re.compile(r"что\s+nova\s+помнит\s+обо\s+мне\s*[.!?]?\Z", re.IGNORECASE),
        re.compile(r"покажи\s+память\s+nova\s*[.!?]?\Z", re.IGNORECASE),
    )

    def classify(self, text: str) -> NovaMemoryIntentResult:
        if not isinstance(text, str):
            return NovaMemoryIntentResult(NovaMemoryIntentKind.NONE)
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return NovaMemoryIntentResult(NovaMemoryIntentKind.NONE)

        for pattern, category, important in self._CONTENT_PATTERNS:
            match = pattern.fullmatch(normalized)
            if match is not None:
                content = normalize_nova_memory_content(match.group(1))
                return NovaMemoryIntentResult(
                    NovaMemoryIntentKind.CREATE,
                    content=content,
                    category=category,
                    important=important,
                )
        if self._AWAIT_CONTENT.fullmatch(normalized) is not None:
            return NovaMemoryIntentResult(
                NovaMemoryIntentKind.AWAIT_CONTENT,
                category="about_me",
            )
        if any(pattern.fullmatch(normalized) is not None for pattern in self._REMEMBER_THIS):
            return NovaMemoryIntentResult(
                NovaMemoryIntentKind.REMEMBER_THIS,
                category="about_me",
            )
        if self._DELETE_ALL.fullmatch(normalized) is not None:
            return NovaMemoryIntentResult(NovaMemoryIntentKind.DELETE_ALL)
        if any(pattern.fullmatch(normalized) is not None for pattern in self._OPEN):
            return NovaMemoryIntentResult(NovaMemoryIntentKind.OPEN)
        return NovaMemoryIntentResult(NovaMemoryIntentKind.NONE)


_INTENT_CLASSIFIER = NovaMemoryIntentClassifier()


def classify_nova_memory_intent(text: str) -> NovaMemoryIntentResult:
    return _INTENT_CLASSIFIER.classify(text)


class NovaMemoryFlowStore:
    """Bounded process-local memory UI state and opaque callback capabilities."""

    def __init__(
        self,
        *,
        ttl_seconds: float = NOVA_MEMORY_FLOW_TTL_SECONDS,
        max_sessions: int = NOVA_MEMORY_FLOW_MAX_SESSIONS,
        max_capabilities_per_session: int = NOVA_MEMORY_FLOW_MAX_CAPABILITIES,
        clock: Callable[[], float] = monotonic,
    ):
        if not 0 < ttl_seconds <= NOVA_MEMORY_FLOW_TTL_SECONDS:
            raise ValueError("Nova memory flow ttl must be between 0 and 15 minutes")
        if not 1 <= max_sessions <= NOVA_MEMORY_FLOW_MAX_SESSIONS:
            raise ValueError("Nova memory flow sessions must be between 1 and 128")
        if not 1 <= max_capabilities_per_session <= 64:
            raise ValueError("Nova memory screen capabilities must be between 1 and 64")
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_capabilities_per_session = max_capabilities_per_session
        self._clock = clock
        self._sessions: dict[tuple[int, int], NovaMemoryFlowSession] = {}
        self._capabilities: dict[str, NovaMemoryCapability] = {}
        self._token_index: dict[str, tuple[int, int]] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        tier: AccessTier,
        access_version: int,
        canonical_message_id: int | None,
        phase: NovaMemoryFlowPhase = NovaMemoryFlowPhase.ROOT,
    ) -> NovaMemoryFlowSession:
        self._validate_new_binding(
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            tier=tier,
            access_version=access_version,
            canonical_message_id=canonical_message_id,
        )
        clean_phase = self._phase(phase)
        async with self._lock:
            self._prune_locked()
            key = (owner_id, chat_id)
            self._drop_locked(key)
            while len(self._sessions) >= self.max_sessions:
                self._drop_locked(next(iter(self._sessions)))
            now = self._clock()
            session = NovaMemoryFlowSession(
                id=self._random_token(),
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                tier=tier,
                access_version=access_version,
                version=1,
                canonical_message_id=canonical_message_id,
                phase=clean_phase,
                created_at=now,
                expires_at=now + self.ttl_seconds,
            )
            self._sessions[key] = session
            return session

    async def reserve(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        tier: AccessTier,
        access_version: int,
        phase: NovaMemoryFlowPhase = NovaMemoryFlowPhase.ROOT,
    ) -> NovaMemoryFlowSession:
        return await self.create(
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            tier=tier,
            access_version=access_version,
            canonical_message_id=None,
            phase=phase,
        )

    async def current(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
    ) -> NovaMemoryFlowSession | None:
        async with self._lock:
            self._prune_locked()
            live = self._sessions.get((owner_id, chat_id))
            if live is None or live.telegram_user_id != telegram_user_id:
                return None
            return live

    async def get(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        tier: AccessTier,
        access_version: int,
        canonical_message_id: int | None | object = _UNSET,
        session_id: str | None = None,
        session_version: int | None = None,
    ) -> NovaMemoryFlowSession | None:
        async with self._lock:
            self._prune_locked()
            key = (owner_id, chat_id)
            live = self._sessions.get(key)
            if live is None or live.telegram_user_id != telegram_user_id:
                return None
            if session_id is not None and live.id != session_id:
                return None
            if session_version is not None and live.version != session_version:
                return None
            if live.tier != tier or live.access_version != access_version:
                self._drop_locked(key)
                return None
            if (
                canonical_message_id is not _UNSET
                and live.canonical_message_id != canonical_message_id
            ):
                return None
            return live

    async def get_exact(
        self,
        session: NovaMemoryFlowSession,
    ) -> NovaMemoryFlowSession | None:
        async with self._lock:
            self._prune_locked()
            live = self._sessions.get((session.owner_id, session.chat_id))
            return live if live == session else None

    async def update(
        self,
        session: NovaMemoryFlowSession,
        *,
        phase: NovaMemoryFlowPhase,
        canonical_message_id: int | None | object = _UNSET,
        candidate_content: str | None | object = _UNSET,
        candidate_category: NovaMemoryCategory | None | object = _UNSET,
        candidate_important: bool | object = _UNSET,
        item_public_id: str | None | object = _UNSET,
        item_version: int | None | object = _UNSET,
        list_filter: NovaMemoryListFilter | None | object = _UNSET,
        page: int | object = _UNSET,
        collection_revision: str | None | object = _UNSET,
        collection_count: int | None | object = _UNSET,
    ) -> NovaMemoryFlowSession | None:
        clean_phase = self._phase(phase)
        values: dict[str, object] = {"phase": clean_phase}
        if canonical_message_id is not _UNSET:
            values["canonical_message_id"] = self._canonical(canonical_message_id)
        if candidate_content is not _UNSET:
            values["candidate_content"] = (
                normalize_nova_memory_content(candidate_content)
                if candidate_content is not None
                else None
            )
        if candidate_category is not _UNSET:
            values["candidate_category"] = self._category(candidate_category)
        if candidate_important is not _UNSET:
            if not isinstance(candidate_important, bool):
                raise ValueError("candidate importance must be a boolean")
            values["candidate_important"] = candidate_important
        if item_public_id is not _UNSET:
            values["item_public_id"] = self._public_id(item_public_id)
        if item_version is not _UNSET:
            values["item_version"] = self._optional_positive(item_version, "item version")
        if list_filter is not _UNSET:
            values["list_filter"] = self._list_filter(list_filter)
        if page is not _UNSET:
            values["page"] = self._page(page)
        if collection_revision is not _UNSET:
            values["collection_revision"] = self._collection_revision(collection_revision)
        if collection_count is not _UNSET:
            values["collection_count"] = self._collection_count(collection_count)

        if clean_phase not in {
            NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
            NovaMemoryFlowPhase.CREATE_PREVIEW,
            NovaMemoryFlowPhase.AWAITING_UPDATE_CONTENT,
            NovaMemoryFlowPhase.UPDATE_PREVIEW,
            NovaMemoryFlowPhase.PROCESSING,
        }:
            values.setdefault("candidate_content", None)
            values.setdefault("candidate_category", None)
            values.setdefault("candidate_important", False)
        if clean_phase in {
            NovaMemoryFlowPhase.ROOT,
            NovaMemoryFlowPhase.LIST,
            NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
            NovaMemoryFlowPhase.CREATE_PREVIEW,
            NovaMemoryFlowPhase.DELETE_ALL_PREVIEW,
        }:
            values.setdefault("item_public_id", None)
            values.setdefault("item_version", None)

        async with self._lock:
            self._prune_locked()
            key = (session.owner_id, session.chat_id)
            live = self._sessions.get(key)
            if live != session:
                return None
            updated = replace(live, version=live.version + 1, **values)
            self._sessions[key] = updated
            self._drop_capabilities_locked(live.id)
            return updated

    async def transition(
        self,
        session: NovaMemoryFlowSession,
        *,
        phase: NovaMemoryFlowPhase,
        canonical_message_id: int | None | object = _UNSET,
        candidate_content: str | None | object = _UNSET,
        candidate_category: NovaMemoryCategory | None | object = _UNSET,
        candidate_important: bool | object = _UNSET,
        item_public_id: str | None | object = _UNSET,
        item_version: int | None | object = _UNSET,
        list_filter: NovaMemoryListFilter | None | object = _UNSET,
        page: int | object = _UNSET,
        collection_revision: str | None | object = _UNSET,
        collection_count: int | None | object = _UNSET,
    ) -> NovaMemoryFlowSession | None:
        return await self.update(
            session,
            phase=phase,
            canonical_message_id=canonical_message_id,
            candidate_content=candidate_content,
            candidate_category=candidate_category,
            candidate_important=candidate_important,
            item_public_id=item_public_id,
            item_version=item_version,
            list_filter=list_filter,
            page=page,
            collection_revision=collection_revision,
            collection_count=collection_count,
        )

    async def bind_canonical(
        self,
        session: NovaMemoryFlowSession,
        *,
        canonical_message_id: int,
    ) -> NovaMemoryFlowSession | None:
        if session.canonical_message_id is not None:
            return None
        return await self.update(
            session,
            phase=session.phase,
            canonical_message_id=canonical_message_id,
        )

    async def rebind_canonical(
        self,
        session: NovaMemoryFlowSession,
        *,
        expected_message_id: int | None,
        new_message_id: int,
    ) -> NovaMemoryFlowSession | None:
        if session.canonical_message_id != expected_message_id:
            return None
        return await self.update(
            session,
            phase=session.phase,
            canonical_message_id=new_message_id,
        )

    async def issue(
        self,
        session: NovaMemoryFlowSession,
        *,
        action: str,
        mutation: bool = False,
        public_id: str | None = None,
        expected_item_version: int | None = None,
        expected_collection_revision: str | None = None,
        list_filter: NovaMemoryListFilter | None = None,
        page: int | None = None,
    ) -> str | None:
        clean_action = self._action(action)
        if not isinstance(mutation, bool):
            raise ValueError("capability mutation flag must be a boolean")
        clean_public_id = self._public_id(public_id)
        clean_item_version = self._optional_positive(expected_item_version, "item version")
        clean_revision = self._collection_revision(expected_collection_revision)
        clean_filter = self._list_filter(list_filter)
        clean_page = self._page(page) if page is not None else None
        if (clean_public_id is None) != (clean_item_version is None):
            raise ValueError("item capabilities require both public id and expected version")
        if clean_public_id is not None and clean_revision is not None:
            raise ValueError("item and collection fences cannot share one capability")

        async with self._lock:
            self._prune_locked()
            live = self._sessions.get((session.owner_id, session.chat_id))
            if live != session or live.canonical_message_id is None:
                return None
            existing = [
                capability
                for capability in self._capabilities.values()
                if capability.session_id == live.id
            ]
            while len(existing) >= self.max_capabilities_per_session:
                self._drop_capability_locked(existing.pop(0).token)
            token = self._random_token()
            while token in self._capabilities:
                token = self._random_token()
            capability = NovaMemoryCapability(
                token=token,
                owner_id=live.owner_id,
                telegram_user_id=live.telegram_user_id,
                chat_id=live.chat_id,
                tier=live.tier,
                access_version=live.access_version,
                session_id=live.id,
                session_version=live.version,
                canonical_message_id=live.canonical_message_id,
                action=clean_action,
                mutation=mutation,
                expires_at=live.expires_at,
                public_id=clean_public_id,
                expected_item_version=clean_item_version,
                expected_collection_revision=clean_revision,
                list_filter=clean_filter,
                page=clean_page,
            )
            self._capabilities[token] = capability
            self._token_index[token] = (live.owner_id, live.chat_id)
            callback_data = capability.callback_data
            if len(callback_data.encode("utf-8")) > 64:
                self._drop_capability_locked(token)
                raise RuntimeError("Nova memory callback exceeds Telegram limit")
            return callback_data

    async def claim(
        self,
        callback_data: str,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        tier: AccessTier,
        access_version: int,
        canonical_message_id: int,
        expected_action: str | None = None,
    ) -> NovaMemoryCapabilityClaim | None:
        token = self._callback_token(callback_data)
        if token is None:
            return None
        clean_expected_action = (
            self._action(expected_action) if expected_action is not None else None
        )
        async with self._lock:
            self._prune_locked()
            key = self._token_index.get(token)
            capability = self._capabilities.get(token)
            if key is None or capability is None:
                return None
            if key != (owner_id, chat_id):
                return None
            if (
                capability.telegram_user_id != telegram_user_id
                or capability.canonical_message_id != canonical_message_id
            ):
                return None
            if clean_expected_action is not None and capability.action != clean_expected_action:
                return None
            live = self._sessions.get(key)
            if live is None:
                self._drop_capability_locked(token)
                return None
            if live.telegram_user_id != telegram_user_id:
                return None
            if live.tier != tier or live.access_version != access_version:
                self._drop_locked(key)
                return None
            if (
                capability.owner_id != live.owner_id
                or capability.tier != live.tier
                or capability.access_version != live.access_version
                or capability.session_id != live.id
                or capability.session_version != live.version
                or capability.canonical_message_id != live.canonical_message_id
            ):
                self._drop_capability_locked(token)
                return None
            if capability.mutation:
                processing = replace(
                    live,
                    version=live.version + 1,
                    phase=NovaMemoryFlowPhase.PROCESSING,
                )
                self._sessions[key] = processing
                self._drop_capabilities_locked(live.id)
                live = processing
            return NovaMemoryCapabilityClaim(capability=capability, session=live)

    async def clear(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        session_id: str | None = None,
    ) -> bool:
        async with self._lock:
            self._prune_locked()
            key = (owner_id, chat_id)
            live = self._sessions.get(key)
            if (
                live is None
                or live.telegram_user_id != telegram_user_id
                or (session_id is not None and live.id != session_id)
            ):
                return False
            self._drop_locked(key)
            return True

    async def clear_exact(self, session: NovaMemoryFlowSession) -> bool:
        """Clear only the exact immutable generation supplied by the caller."""

        if not isinstance(session, NovaMemoryFlowSession):
            return False
        async with self._lock:
            self._prune_locked()
            key = (session.owner_id, session.chat_id)
            live = self._sessions.get(key)
            if live != session:
                return False
            self._drop_locked(key)
            return True

    async def count(self) -> int:
        async with self._lock:
            self._prune_locked()
            return len(self._sessions)

    async def cleanup(self) -> int:
        async with self._lock:
            before = len(self._sessions)
            self._prune_locked()
            return before - len(self._sessions)

    def _prune_locked(self) -> None:
        now = self._clock()
        for key, session in tuple(self._sessions.items()):
            if session.expires_at <= now:
                self._drop_locked(key)
        for token, capability in tuple(self._capabilities.items()):
            if capability.expires_at <= now:
                self._drop_capability_locked(token)

    def _drop_locked(self, key: tuple[int, int]) -> None:
        session = self._sessions.pop(key, None)
        if session is not None:
            self._drop_capabilities_locked(session.id)

    def _drop_capabilities_locked(self, session_id: str) -> None:
        for token, capability in tuple(self._capabilities.items()):
            if capability.session_id == session_id:
                self._drop_capability_locked(token)

    def _drop_capability_locked(self, token: str) -> None:
        self._capabilities.pop(token, None)
        self._token_index.pop(token, None)

    @staticmethod
    def _random_token() -> str:
        return secrets.token_urlsafe(18)

    @staticmethod
    def _callback_token(callback_data: str) -> str | None:
        if not isinstance(callback_data, str) or len(callback_data.encode("utf-8")) > 64:
            return None
        if not callback_data.startswith(NOVA_MEMORY_CALLBACK_PREFIX):
            return None
        token = callback_data.removeprefix(NOVA_MEMORY_CALLBACK_PREFIX)
        return token if _TOKEN_PATTERN.fullmatch(token) is not None else None

    @staticmethod
    def _phase(value: NovaMemoryFlowPhase) -> NovaMemoryFlowPhase:
        try:
            return NovaMemoryFlowPhase(value)
        except (TypeError, ValueError):
            raise ValueError("unsupported Nova memory flow phase") from None

    @staticmethod
    def _action(value: str) -> str:
        if not isinstance(value, str) or _ACTION_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid Nova memory capability action")
        return value

    @staticmethod
    def _category(value: object) -> NovaMemoryCategory | None:
        if value is None:
            return None
        if not isinstance(value, str) or value not in NOVA_MEMORY_CATEGORIES:
            raise ValueError("unsupported Nova memory category")
        return cast(NovaMemoryCategory, value)

    @staticmethod
    def _list_filter(value: object) -> NovaMemoryListFilter | None:
        if value is None:
            return None
        if not isinstance(value, str) or value not in {*NOVA_MEMORY_CATEGORIES, "important"}:
            raise ValueError("unsupported Nova memory list filter")
        return cast(NovaMemoryListFilter, value)

    @staticmethod
    def _public_id(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not 1 <= len(value) <= 64:
            raise ValueError("invalid Nova memory public id")
        if any(unicodedata.category(character).startswith("C") for character in value):
            raise ValueError("invalid Nova memory public id")
        return value

    @staticmethod
    def _collection_revision(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or _COLLECTION_REVISION_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid Nova memory collection revision")
        return value

    @staticmethod
    def _collection_count(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError("invalid Nova memory collection count")
        return value

    @staticmethod
    def _optional_positive(value: object, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be positive")
        return value

    @staticmethod
    def _page(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError("Nova memory page must be between 0 and 100")
        return value

    @staticmethod
    def _canonical(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("canonical message id must be positive")
        return value

    @classmethod
    def _validate_new_binding(
        cls,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        tier: AccessTier,
        access_version: int,
        canonical_message_id: int | None,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (owner_id, telegram_user_id, chat_id)
        ):
            raise ValueError("Nova memory owner and private destination ids must be positive")
        if tier not in ACCESS_TIERS or tier not in FULL_ACCESS_TIERS:
            raise ValueError("Nova memory flow requires full access")
        if isinstance(access_version, bool) or not isinstance(access_version, int):
            raise ValueError("access version must be positive")
        if access_version <= 0:
            raise ValueError("access version must be positive")
        cls._canonical(canonical_message_id)


__all__ = [
    "NOVA_MEMORY_CALLBACK_PREFIX",
    "NOVA_MEMORY_FLOW_MAX_CAPABILITIES",
    "NOVA_MEMORY_FLOW_MAX_SESSIONS",
    "NOVA_MEMORY_FLOW_TTL_SECONDS",
    "NovaMemoryCapability",
    "NovaMemoryCapabilityClaim",
    "NovaMemoryFlowPhase",
    "NovaMemoryFlowSession",
    "NovaMemoryFlowStore",
    "NovaMemoryIntentClassifier",
    "NovaMemoryIntentKind",
    "NovaMemoryIntentResult",
    "NovaMemoryListFilter",
    "classify_nova_memory_intent",
]
