from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Literal, Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from .access import FULL_ACCESS_TIERS
from .conversation import (
    COMPANION_CONTEXT_MAX_RAW_MESSAGES,
    COMPANION_CONTEXT_MIN_RAW_MESSAGES,
    COMPANION_PROMPT_CONTEXT_MAX_BYTES,
    CompanionConversationFence,
    ConversationExchangeReceipt,
    build_companion_prompt_context,
    companion_conversation_revision,
    fit_companion_prompt_context,
)
from .db import Database
from .models import (
    ConversationMessage,
    ConversationSession,
    Goal,
    User,
    VisionItem,
    VisionProfile,
    WeeklyFocus,
)
from .nova_memory import NOVA_MEMORY_CATEGORIES
from .nova_memory_application import (
    NOVA_MEMORY_APPLICATION_MAX_ITEMS,
    NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES,
    NovaMemoryProjection,
)
from .weekly_review import current_week_start

NOVA_COMPANION_CONTEXT_MAX_PAYLOAD_BYTES = COMPANION_PROMPT_CONTEXT_MAX_BYTES
NOVA_COMPANION_PROFILE_LIST_MAX_ITEMS = 5
NOVA_COMPANION_VISION_MAX_ITEMS = 6
NOVA_COMPANION_GOAL_MAX_ITEMS = 5
NOVA_COMPANION_RECENT_MESSAGE_MAX_ITEMS = 20
NOVA_COMPANION_VISION_QUERY_LIMIT = 24
NOVA_COMPANION_GOAL_QUERY_LIMIT = 20
NOVA_COMPANION_CONVERSATION_QUERY_LIMIT = 20

_PROFILE_SUMMARY_MAX_CHARS = 800
_PROFILE_ITEM_MAX_CHARS = 200
_PROFILE_MOTIVATION_MAX_CHARS = 120
_VISION_WISH_MAX_CHARS = 500
_VISION_WHY_MAX_CHARS = 400
_VISION_FIRST_STEP_MAX_CHARS = 300
_GOAL_LIFE_AREA_MAX_CHARS = 80
_GOAL_TITLE_MAX_CHARS = 300
_GOAL_DETAIL_MAX_CHARS = 400
_GOAL_HORIZON_MAX_CHARS = 100
_WEEKLY_FOCUS_MAX_CHARS = 300
_WEEKLY_APPROACH_MAX_CHARS = 500
_WEEKLY_STEP_MAX_CHARS = 200
_CONVERSATION_TOPIC_MAX_CHARS = 200
_CONVERSATION_SUMMARY_MAX_CHARS = 600
_CONVERSATION_MESSAGE_MAX_CHARS = 600
_IDENTITY_NAME_MAX_CHARS = 120
_IDENTITY_CITY_MAX_CHARS = 120
_IDENTITY_TIMEZONE_MAX_CHARS = 64
_VISION_CATEGORY_ORDER = {
    "health_energy": 0,
    "relationships_family": 1,
    "work_purpose": 2,
    "money": 3,
    "home": 4,
    "travel": 5,
    "growth_creativity": 6,
    "other": 7,
}

type NovaCompanionContextStatus = Literal[
    "ready", "access_changed", "context_changed", "unavailable"
]


class NovaCompanionProjectionError(ValueError):
    """A content-free failure to construct a companion context projection."""


class _ProfileSource(Protocol):
    summary: str
    values: Sequence[str]
    desired_identity: Sequence[str]
    constraints: Sequence[str]
    motivation_style: str | None


class _VisionSource(Protocol):
    category: str
    wish_text: str
    why_text: str | None
    first_step: str | None


class _GoalSource(Protocol):
    life_area: str
    title: str
    outcome: str
    progress_criterion: str
    horizon: str
    priority: int
    vision_link: str


class _WeeklyFocusSource(Protocol):
    focus: str
    approach: str | None
    small_steps: Sequence[str]


@dataclass(frozen=True, slots=True)
class NovaCompanionContextProjection:
    """Immutable bounded provider context with a content-free representation."""

    _payload_json: str = field(repr=False)
    payload_bytes: int
    profile_present: bool
    vision_count: int
    goal_count: int
    memory_count: int
    recent_message_count: int
    weekly_focus_present: bool
    omitted_count: int

    def __post_init__(self) -> None:
        integer_metrics = (
            self.payload_bytes,
            self.vision_count,
            self.goal_count,
            self.memory_count,
            self.recent_message_count,
            self.omitted_count,
        )
        if (
            not isinstance(self._payload_json, str)
            or not self._payload_json
            or any(type(value) is not int or value < 0 for value in integer_metrics)
            or not isinstance(self.profile_present, bool)
            or not isinstance(self.weekly_focus_present, bool)
            or self.payload_bytes != len(self._payload_json.encode("utf-8"))
            or self.payload_bytes > NOVA_COMPANION_CONTEXT_MAX_PAYLOAD_BYTES
            or self.vision_count > NOVA_COMPANION_VISION_MAX_ITEMS
            or self.goal_count > NOVA_COMPANION_GOAL_MAX_ITEMS
            or self.memory_count > NOVA_MEMORY_APPLICATION_MAX_ITEMS
            or self.recent_message_count > NOVA_COMPANION_RECENT_MESSAGE_MAX_ITEMS
        ):
            raise NovaCompanionProjectionError("Invalid companion context projection.")
        try:
            payload = json.loads(self._payload_json)
        except (TypeError, ValueError):
            raise NovaCompanionProjectionError("Invalid companion context projection.") from None
        if not isinstance(payload, dict):
            raise NovaCompanionProjectionError("Invalid companion context projection.")
        if _compact_json(payload) != self._payload_json:
            raise NovaCompanionProjectionError("Invalid companion context projection.")
        _validate_projection_payload_shape(payload, self)

    def provider_payload(self) -> dict[str, object]:
        """Return a deep-detached JSON object; callers cannot mutate the projection."""
        payload = json.loads(self._payload_json)
        if not isinstance(payload, dict):  # guarded by __post_init__
            raise NovaCompanionProjectionError("Invalid companion context projection.")
        return payload

    def provider_json(self) -> str:
        return self._payload_json


@dataclass(frozen=True, slots=True)
class NovaCompanionContextFence:
    """Private generation identity. Its repr deliberately exposes no metadata."""

    telegram_actor_id: int = field(repr=False)
    owner_id: int = field(repr=False)
    expected_tier: str = field(repr=False)
    expected_access_version: int = field(repr=False)
    timezone_name: str = field(repr=False)
    local_week_start: date = field(repr=False)
    source_revision: str = field(repr=False)
    memory_collection_revision: str | None = field(default=None, repr=False)
    conversation_fence: CompanionConversationFence | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class NovaCompanionContextSnapshot:
    status: NovaCompanionContextStatus
    projection: NovaCompanionContextProjection | None = field(default=None, repr=False)
    fence: NovaCompanionContextFence | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.status == "ready":
            if self.projection is None or self.fence is None:
                raise NovaCompanionProjectionError("Invalid companion context snapshot.")
        elif self.projection is not None or self.fence is not None:
            raise NovaCompanionProjectionError("Invalid companion context snapshot.")


@dataclass(frozen=True, slots=True)
class _NovaCompanionSources:
    owner_id: int
    display_name: str | None = field(repr=False)
    location_city: str | None = field(repr=False)
    timezone_name: str
    local_week_start: date
    profile: VisionProfile | None = field(repr=False)
    vision_items: tuple[VisionItem, ...] = field(repr=False)
    goals: tuple[Goal, ...] = field(repr=False)
    weekly_focus: WeeklyFocus | None = field(repr=False)


class NovaCompanionContextService:
    """Select-only, owner-scoped context materialization for one access generation."""

    def __init__(
        self,
        db: Database,
        conversation_message_limit: int = NOVA_COMPANION_CONVERSATION_QUERY_LIMIT,
    ):
        self.db = db
        self.conversation_message_limit = _bounded_positive_integer(
            conversation_message_limit,
            "Conversation message limit",
            minimum=COMPANION_CONTEXT_MIN_RAW_MESSAGES,
            maximum=COMPANION_CONTEXT_MAX_RAW_MESSAGES,
        )

    async def snapshot(
        self,
        *,
        telegram_actor_id: int,
        expected_tier: str,
        expected_access_version: int,
        conversation_context: Mapping[str, object] | None = None,
        conversation_chat_id: int | None = None,
        confirmed_memory: NovaMemoryProjection | None = None,
        now: datetime | None = None,
    ) -> NovaCompanionContextSnapshot:
        actor_id = _positive_integer(telegram_actor_id, "Telegram actor")
        access_version = _positive_integer(expected_access_version, "Access version")
        tier = _full_access_tier(expected_tier)
        current = _as_utc(now or datetime.now(UTC))
        chat_id = (
            _positive_integer(conversation_chat_id, "Conversation chat")
            if conversation_chat_id is not None
            else None
        )
        raw_message_limit = self.conversation_message_limit
        try:
            current_conversation: dict[str, object] | None = None
            async with self.db.sessions() as session:
                sources = await self._load_sources(
                    session,
                    telegram_actor_id=actor_id,
                    expected_tier=tier,
                    expected_access_version=access_version,
                    now=current,
                )
                if sources is None:
                    return NovaCompanionContextSnapshot("access_changed")
                if chat_id is not None:
                    provided_conversation = _fenced_conversation_payload(conversation_context)
                    current_conversation = await self._load_conversation_payload(
                        session,
                        telegram_actor_id=actor_id,
                        chat_id=chat_id,
                        raw_message_limit=raw_message_limit,
                        now=current,
                    )
                    if current_conversation != provided_conversation:
                        return NovaCompanionContextSnapshot("context_changed")
            projection = build_nova_companion_context_projection(
                profile=sources.profile,
                vision_items=sources.vision_items,
                goals=sources.goals,
                weekly_focus=sources.weekly_focus,
                confirmed_memory=confirmed_memory,
                conversation_context=conversation_context,
                display_name=sources.display_name,
                location_city=sources.location_city,
                timezone_name=sources.timezone_name,
            )
            conversation_fence: CompanionConversationFence | None = None
            if chat_id is not None and current_conversation is not None:
                conversation_payload_max_bytes = _conversation_payload_max_bytes(
                    sources.profile,
                    sources.weekly_focus,
                    display_name=sources.display_name,
                    location_city=sources.location_city,
                    timezone_name=sources.timezone_name,
                )
                projected_conversation = fit_companion_prompt_context(
                    current_conversation,
                    conversation_payload_max_bytes,
                )
                provider_conversation = projection.provider_payload().get(
                    "recent_conversation",
                    {},
                )
                if provider_conversation != projected_conversation:
                    raise NovaCompanionProjectionError("Invalid fitted conversation context.")
                conversation_fence = CompanionConversationFence(
                    owner_id=sources.owner_id,
                    telegram_user_id=actor_id,
                    chat_id=chat_id,
                    access_version=access_version,
                    access_tier=tier,
                    raw_message_limit=raw_message_limit,
                    conversation_payload_max_bytes=conversation_payload_max_bytes,
                    revision=companion_conversation_revision(
                        owner_id=sources.owner_id,
                        telegram_user_id=actor_id,
                        chat_id=chat_id,
                        context=projected_conversation,
                    ),
                )
            fence = NovaCompanionContextFence(
                telegram_actor_id=actor_id,
                owner_id=sources.owner_id,
                expected_tier=tier,
                expected_access_version=access_version,
                timezone_name=sources.timezone_name,
                local_week_start=sources.local_week_start,
                source_revision=_context_source_revision(sources),
                memory_collection_revision=(
                    confirmed_memory.collection_revision if confirmed_memory is not None else None
                ),
                conversation_fence=conversation_fence,
            )
            await self._before_generation_check(fence)
            if not await self.current_check(fence, now=current):
                return NovaCompanionContextSnapshot("access_changed")
            return NovaCompanionContextSnapshot("ready", projection=projection, fence=fence)
        except (SQLAlchemyError, NovaCompanionProjectionError, ValueError):
            return NovaCompanionContextSnapshot("unavailable")

    async def current_check(
        self,
        fence: NovaCompanionContextFence,
        *,
        exchange_receipt: ConversationExchangeReceipt | None = None,
        now: datetime | None = None,
    ) -> bool:
        if not isinstance(fence, NovaCompanionContextFence):
            raise NovaCompanionProjectionError("Invalid companion context fence.")
        if (
            exchange_receipt is not None
            and type(exchange_receipt) is not ConversationExchangeReceipt
        ):
            raise NovaCompanionProjectionError("Invalid companion exchange receipt.")
        current = _as_utc(now or datetime.now(UTC))
        try:
            expected_conversation = fence.conversation_fence
            if exchange_receipt is not None:
                if expected_conversation is None:
                    return False
                expected_conversation = exchange_receipt.result_fence_for(expected_conversation)
                if expected_conversation is None:
                    return False
            async with self.db.sessions() as session:
                sources = await self._load_sources(
                    session,
                    telegram_actor_id=fence.telegram_actor_id,
                    expected_tier=fence.expected_tier,
                    expected_access_version=fence.expected_access_version,
                    expected_owner_id=fence.owner_id,
                    now=current,
                )
                current_conversation: CompanionConversationFence | None = None
                if sources is not None and expected_conversation is not None:
                    conversation_payload = await self._load_conversation_payload(
                        session,
                        telegram_actor_id=fence.telegram_actor_id,
                        chat_id=expected_conversation.chat_id,
                        raw_message_limit=expected_conversation.raw_message_limit,
                        now=current,
                    )
                    projected_conversation = fit_companion_prompt_context(
                        conversation_payload,
                        expected_conversation.conversation_payload_max_bytes,
                    )
                    current_conversation = CompanionConversationFence(
                        owner_id=sources.owner_id,
                        telegram_user_id=fence.telegram_actor_id,
                        chat_id=expected_conversation.chat_id,
                        access_version=fence.expected_access_version,
                        access_tier=fence.expected_tier,
                        raw_message_limit=expected_conversation.raw_message_limit,
                        conversation_payload_max_bytes=(
                            expected_conversation.conversation_payload_max_bytes
                        ),
                        revision=companion_conversation_revision(
                            owner_id=sources.owner_id,
                            telegram_user_id=fence.telegram_actor_id,
                            chat_id=expected_conversation.chat_id,
                            context=projected_conversation,
                        ),
                    )
            return (
                sources is not None
                and sources.owner_id == fence.owner_id
                and sources.timezone_name == fence.timezone_name
                and sources.local_week_start == fence.local_week_start
                and _context_source_revision(sources) == fence.source_revision
                and current_conversation == expected_conversation
            )
        except (SQLAlchemyError, NovaCompanionProjectionError, ValueError):
            return False

    async def _load_sources(
        self,
        session,
        *,
        telegram_actor_id: int,
        expected_tier: str,
        expected_access_version: int,
        now: datetime,
        expected_owner_id: int | None = None,
    ) -> _NovaCompanionSources | None:
        actor_query = select(
            User.id,
            User.display_name,
            User.location_city,
            User.timezone,
        ).where(
            User.telegram_id == telegram_actor_id,
            User.access_tier == expected_tier,
            User.access_version == expected_access_version,
            User.access_tier.in_(FULL_ACCESS_TIERS),
        )
        if expected_owner_id is not None:
            actor_query = actor_query.where(User.id == expected_owner_id)
        actor = (await session.execute(actor_query)).one_or_none()
        if actor is None:
            return None
        owner_id, display_name, location_city, timezone_name = actor
        week_start = current_week_start(timezone_name, now=now)
        profile = await session.scalar(
            select(VisionProfile).where(VisionProfile.user_id == owner_id)
        )
        vision_items = tuple(
            (
                await session.scalars(
                    select(VisionItem)
                    .where(
                        VisionItem.owner_id == owner_id,
                        VisionItem.status == "active",
                    )
                    .order_by(VisionItem.category, VisionItem.wish_text, VisionItem.id)
                    .limit(NOVA_COMPANION_VISION_QUERY_LIMIT)
                )
            ).all()
        )
        goals = tuple(
            (
                await session.scalars(
                    select(Goal)
                    .where(Goal.user_id == owner_id, Goal.status == "active")
                    .order_by(Goal.priority.desc(), Goal.title, Goal.id)
                    .limit(NOVA_COMPANION_GOAL_QUERY_LIMIT)
                )
            ).all()
        )
        weekly_focus = await session.scalar(
            select(WeeklyFocus).where(
                WeeklyFocus.owner_id == owner_id,
                WeeklyFocus.week_start == week_start,
            )
        )
        return _NovaCompanionSources(
            owner_id=owner_id,
            display_name=_optional_text(display_name, _IDENTITY_NAME_MAX_CHARS),
            location_city=_optional_text(location_city, _IDENTITY_CITY_MAX_CHARS),
            timezone_name=timezone_name,
            local_week_start=week_start,
            profile=profile,
            vision_items=vision_items,
            goals=goals,
            weekly_focus=weekly_focus,
        )

    async def _load_conversation_payload(
        self,
        session,
        *,
        telegram_actor_id: int,
        chat_id: int,
        raw_message_limit: int,
        now: datetime,
    ) -> dict[str, object]:
        conversation = await session.scalar(
            select(ConversationSession).where(
                ConversationSession.telegram_user_id == telegram_actor_id,
                ConversationSession.chat_id == chat_id,
            )
        )
        if conversation is None or _as_utc(conversation.expires_at) <= now:
            return {}
        rows = list(
            (
                await session.scalars(
                    select(ConversationMessage)
                    .where(ConversationMessage.session_id == conversation.id)
                    .order_by(ConversationMessage.id.desc())
                    .limit(raw_message_limit)
                )
            ).all()
        )
        safe = build_companion_prompt_context(
            {
                "role": row.role,
                "content": row.content,
                "intent": row.intent,
            }
            for row in reversed(rows)
        )
        normalized, _omitted = _conversation_payload(safe)
        return normalized

    async def _before_generation_check(self, fence: NovaCompanionContextFence) -> None:
        """Deterministic test seam before the detached snapshot's generation fence."""
        del fence


def _context_source_revision(sources: _NovaCompanionSources) -> str:
    """Hash exactly the bounded durable sources that can reach the provider."""

    profile_payload, _profile_omitted = _profile_payload(sources.profile)
    profile_manifest: dict[str, object] | None = None
    if sources.profile is not None:
        profile_manifest = {
            "row": sources.profile.id,
            "payload": profile_payload,
        }
    vision_manifest = [
        {"row": item.id, "payload": _vision_payload(item)} for item in sources.vision_items
    ]
    goal_manifest = [{"row": goal.id, "payload": _goal_payload(goal)} for goal in sources.goals]
    weekly_payload, _weekly_omitted = _weekly_payload(sources.weekly_focus)
    weekly_manifest: dict[str, object] | None = None
    if sources.weekly_focus is not None:
        weekly_manifest = {
            "row": sources.weekly_focus.id,
            "version": sources.weekly_focus.version,
            "payload": weekly_payload,
        }
    manifest = {
        "owner": sources.owner_id,
        "display_name": sources.display_name,
        "location_city": sources.location_city,
        "timezone": sources.timezone_name,
        "local_week_start": sources.local_week_start.isoformat(),
        "profile": profile_manifest,
        "active_vision_items": vision_manifest,
        "active_goals": goal_manifest,
        "current_weekly_focus": weekly_manifest,
    }
    return sha256(_compact_json(manifest).encode("utf-8")).hexdigest()


def _conversation_payload_max_bytes(
    profile: _ProfileSource | None,
    weekly_focus: _WeeklyFocusSource | None,
    *,
    display_name: str | None = None,
    location_city: str | None = None,
    timezone_name: str | None = None,
) -> int:
    """Freeze the exact recent-conversation value budget used by provider fitting."""

    base: dict[str, object] = {}
    identity = _identity_payload(display_name, location_city, timezone_name)
    if identity:
        base["confirmed_identity"] = identity
    profile_payload, _profile_omitted = _profile_payload(profile)
    if profile_payload:
        base["profile"] = profile_payload
    weekly_payload, _weekly_omitted = _weekly_payload(weekly_focus)
    if weekly_payload:
        base["current_weekly_focus"] = weekly_payload
    base["recent_conversation"] = {}
    overhead = len(_compact_json(base).encode("utf-8")) - len(b"{}")
    return max(
        0,
        min(
            COMPANION_PROMPT_CONTEXT_MAX_BYTES,
            NOVA_COMPANION_CONTEXT_MAX_PAYLOAD_BYTES - overhead,
        ),
    )


def _identity_payload(
    display_name: str | None,
    location_city: str | None,
    timezone_name: str | None,
) -> dict[str, object]:
    result: dict[str, object] = {}
    name = _optional_text(display_name, _IDENTITY_NAME_MAX_CHARS)
    city = _optional_text(location_city, _IDENTITY_CITY_MAX_CHARS)
    timezone = _optional_text(timezone_name, _IDENTITY_TIMEZONE_MAX_CHARS)
    _put_optional(result, "display_name", name)
    _put_optional(result, "location_city", city)
    _put_optional(result, "timezone", timezone)
    return result


def build_nova_companion_context_projection(
    *,
    profile: _ProfileSource | None,
    vision_items: Iterable[_VisionSource] = (),
    goals: Iterable[_GoalSource] = (),
    weekly_focus: _WeeklyFocusSource | None = None,
    confirmed_memory: NovaMemoryProjection | None = None,
    conversation_context: Mapping[str, object] | None = None,
    display_name: str | None = None,
    location_city: str | None = None,
    timezone_name: str | None = None,
) -> NovaCompanionContextProjection:
    """Build a deterministic projection containing only the public provider contract."""
    payload: dict[str, object] = {}
    omitted_count = 0

    identity = _identity_payload(display_name, location_city, timezone_name)
    if identity:
        payload["confirmed_identity"] = identity

    profile_payload, profile_omitted = _profile_payload(profile)
    omitted_count += profile_omitted
    if profile_payload:
        payload["profile"] = profile_payload

    weekly_payload, weekly_omitted = _weekly_payload(weekly_focus)
    omitted_count += weekly_omitted
    if weekly_payload:
        payload["current_weekly_focus"] = weekly_payload

    conversation_payload, conversation_omitted = _conversation_payload(conversation_context)
    omitted_count += conversation_omitted
    if conversation_payload:
        payload["recent_conversation"] = conversation_payload

    goal_payloads = sorted(
        (_goal_payload(goal) for goal in goals),
        key=lambda item: (
            -int(item["priority"]),
            str(item["title"]).casefold(),
            _compact_json(item),
        ),
    )
    omitted_count += max(0, len(goal_payloads) - NOVA_COMPANION_GOAL_MAX_ITEMS)
    goal_payloads = goal_payloads[:NOVA_COMPANION_GOAL_MAX_ITEMS]
    if goal_payloads:
        payload["active_goals"] = goal_payloads

    vision_payloads = sorted(
        (_vision_payload(item) for item in vision_items),
        key=lambda item: (
            _VISION_CATEGORY_ORDER.get(str(item["category"]), len(_VISION_CATEGORY_ORDER)),
            str(item["wish_text"]).casefold(),
            _compact_json(item),
        ),
    )
    omitted_count += max(0, len(vision_payloads) - NOVA_COMPANION_VISION_MAX_ITEMS)
    vision_payloads = vision_payloads[:NOVA_COMPANION_VISION_MAX_ITEMS]
    if vision_payloads:
        payload["active_vision_items"] = vision_payloads

    memory_payloads: list[dict[str, object]] = []
    if confirmed_memory is not None:
        if not isinstance(confirmed_memory, NovaMemoryProjection):
            raise NovaCompanionProjectionError("Invalid confirmed-memory projection.")
        memory_payloads = confirmed_memory.provider_payload()
        if memory_payloads:
            payload["confirmed_memory"] = memory_payloads

    omitted_count += _fit_payload(payload)
    serialized = _compact_json(payload)
    payload_bytes = len(serialized.encode("utf-8"))
    if payload_bytes > NOVA_COMPANION_CONTEXT_MAX_PAYLOAD_BYTES:
        raise NovaCompanionProjectionError("Companion context exceeds its payload limit.")

    recent = payload.get("recent_conversation")
    recent_messages = recent.get("recent_messages", []) if isinstance(recent, dict) else []
    return NovaCompanionContextProjection(
        _payload_json=serialized,
        payload_bytes=payload_bytes,
        profile_present="profile" in payload,
        vision_count=len(payload.get("active_vision_items", [])),
        goal_count=len(payload.get("active_goals", [])),
        memory_count=len(payload.get("confirmed_memory", [])),
        recent_message_count=len(recent_messages),
        weekly_focus_present="current_weekly_focus" in payload,
        omitted_count=omitted_count,
    )


def _profile_payload(profile: _ProfileSource | None) -> tuple[dict[str, object], int]:
    if profile is None:
        return {}, 0
    try:
        summary = _optional_text(profile.summary, _PROFILE_SUMMARY_MAX_CHARS)
        values, values_omitted = _bounded_text_list(
            profile.values,
            max_items=NOVA_COMPANION_PROFILE_LIST_MAX_ITEMS,
            max_chars=_PROFILE_ITEM_MAX_CHARS,
        )
        identities, identities_omitted = _bounded_text_list(
            profile.desired_identity,
            max_items=NOVA_COMPANION_PROFILE_LIST_MAX_ITEMS,
            max_chars=_PROFILE_ITEM_MAX_CHARS,
        )
        constraints, constraints_omitted = _bounded_text_list(
            profile.constraints,
            max_items=NOVA_COMPANION_PROFILE_LIST_MAX_ITEMS,
            max_chars=_PROFILE_ITEM_MAX_CHARS,
        )
        motivation = _optional_text(profile.motivation_style, _PROFILE_MOTIVATION_MAX_CHARS)
    except (AttributeError, TypeError):
        raise NovaCompanionProjectionError("Invalid companion profile source.") from None
    result: dict[str, object] = {}
    _put_optional(result, "summary", summary)
    if values:
        result["values"] = values
    if identities:
        result["desired_identity"] = identities
    if constraints:
        result["constraints"] = constraints
    _put_optional(result, "motivation_style", motivation)
    return result, values_omitted + identities_omitted + constraints_omitted


def _vision_payload(item: _VisionSource) -> dict[str, object]:
    try:
        category = _required_text(item.category, 40)
        wish = _required_text(item.wish_text, _VISION_WISH_MAX_CHARS)
        why = _optional_text(item.why_text, _VISION_WHY_MAX_CHARS)
        first_step = _optional_text(item.first_step, _VISION_FIRST_STEP_MAX_CHARS)
    except (AttributeError, TypeError):
        raise NovaCompanionProjectionError("Invalid companion vision source.") from None
    result: dict[str, object] = {"category": category, "wish_text": wish}
    _put_optional(result, "why_text", why)
    _put_optional(result, "first_step", first_step)
    return result


def _goal_payload(goal: _GoalSource) -> dict[str, object]:
    try:
        priority = goal.priority
        if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 5:
            raise NovaCompanionProjectionError("Invalid companion goal source.")
        result: dict[str, object] = {
            "life_area": _required_text(goal.life_area, _GOAL_LIFE_AREA_MAX_CHARS),
            "title": _required_text(goal.title, _GOAL_TITLE_MAX_CHARS),
            "outcome": _required_text(goal.outcome, _GOAL_DETAIL_MAX_CHARS),
            "progress_criterion": _required_text(goal.progress_criterion, _GOAL_DETAIL_MAX_CHARS),
            "horizon": _required_text(goal.horizon, _GOAL_HORIZON_MAX_CHARS),
            "priority": priority,
            "vision_link": _required_text(goal.vision_link, _GOAL_DETAIL_MAX_CHARS),
        }
    except (AttributeError, TypeError):
        raise NovaCompanionProjectionError("Invalid companion goal source.") from None
    return result


def _weekly_payload(focus: _WeeklyFocusSource | None) -> tuple[dict[str, object], int]:
    if focus is None:
        return {}, 0
    try:
        focus_text = _required_text(focus.focus, _WEEKLY_FOCUS_MAX_CHARS)
        approach = _optional_text(focus.approach, _WEEKLY_APPROACH_MAX_CHARS)
        steps, omitted = _bounded_text_list(
            focus.small_steps,
            max_items=3,
            max_chars=_WEEKLY_STEP_MAX_CHARS,
            preserve_order=True,
        )
    except (AttributeError, TypeError):
        raise NovaCompanionProjectionError("Invalid companion weekly-focus source.") from None
    result: dict[str, object] = {"focus": focus_text}
    _put_optional(result, "approach", approach)
    if steps:
        result["small_steps"] = steps
    return result, omitted


def _conversation_payload(
    context: Mapping[str, object] | None,
) -> tuple[dict[str, object], int]:
    if context is None:
        return {}, 0
    if not isinstance(context, Mapping):
        raise NovaCompanionProjectionError("Invalid recent-conversation context.")
    result: dict[str, object] = {}
    _put_optional(
        result,
        "current_topic",
        _optional_text(context.get("current_topic"), _CONVERSATION_TOPIC_MAX_CHARS),
    )
    _put_optional(
        result,
        "summary",
        _optional_text(context.get("summary"), _CONVERSATION_SUMMARY_MAX_CHARS),
    )
    raw_messages = context.get("recent_messages", ())
    if raw_messages is None:
        raw_messages = ()
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise NovaCompanionProjectionError("Invalid recent-conversation context.")
    selected = raw_messages[-NOVA_COMPANION_RECENT_MESSAGE_MAX_ITEMS:]
    messages: list[dict[str, str]] = []
    for raw in selected:
        if not isinstance(raw, Mapping):
            raise NovaCompanionProjectionError("Invalid recent-conversation message.")
        role = raw.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = _optional_text(raw.get("content"), _CONVERSATION_MESSAGE_MAX_CHARS)
        if content:
            messages.append({"role": str(role), "content": content})
    if messages:
        result["recent_messages"] = messages
    return result, max(0, len(raw_messages) - len(selected))


def _fenced_conversation_payload(
    context: Mapping[str, object] | None,
) -> dict[str, object]:
    if context is None:
        return {}
    if not isinstance(context, Mapping) or set(context) - {"recent_messages"}:
        raise NovaCompanionProjectionError("Invalid fenced conversation context.")
    result, _omitted = _conversation_payload(context)
    return result


def _fit_payload(payload: dict[str, object]) -> int:
    """Drop lowest-priority whole records until the exact UTF-8 limit is met."""
    omitted = 0
    while len(_compact_json(payload).encode("utf-8")) > NOVA_COMPANION_CONTEXT_MAX_PAYLOAD_BYTES:
        removed = False
        for key in ("active_vision_items", "active_goals", "confirmed_memory"):
            values = payload.get(key)
            if isinstance(values, list) and values:
                values.pop()
                omitted += 1
                removed = True
                if not values:
                    payload.pop(key)
                break
        if removed:
            continue
        conversation = payload.get("recent_conversation")
        if isinstance(conversation, dict):
            messages = conversation.get("recent_messages")
            if isinstance(messages, list) and messages:
                messages.pop(0)
                omitted += 1
                removed = True
                if not messages:
                    conversation.pop("recent_messages")
                if not conversation:
                    payload.pop("recent_conversation")
        if removed:
            continue
        profile = payload.get("profile")
        if isinstance(profile, dict):
            for key in ("constraints", "desired_identity", "values"):
                values = profile.get(key)
                if isinstance(values, list) and values:
                    values.pop()
                    omitted += 1
                    removed = True
                    if not values:
                        profile.pop(key)
                    break
        if removed:
            continue
        weekly = payload.get("current_weekly_focus")
        if isinstance(weekly, dict):
            steps = weekly.get("small_steps")
            if isinstance(steps, list) and steps:
                steps.pop()
                omitted += 1
                removed = True
                if not steps:
                    weekly.pop("small_steps")
        if not removed:
            break
    return omitted


def _bounded_text_list(
    values: Sequence[str],
    *,
    max_items: int,
    max_chars: int,
    preserve_order: bool = False,
) -> tuple[list[str], int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise NovaCompanionProjectionError("Invalid companion text collection.")
    cleaned = [_required_text(value, max_chars) for value in values]
    if not preserve_order:
        cleaned = sorted(set(cleaned), key=lambda value: (value.casefold(), value))
    selected = cleaned[:max_items]
    return selected, max(0, len(cleaned) - len(selected))


def _required_text(value: object, max_chars: int) -> str:
    cleaned = _optional_text(value, max_chars)
    if not cleaned:
        raise NovaCompanionProjectionError("Invalid companion context text.")
    return cleaned


def _optional_text(value: object, max_chars: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise NovaCompanionProjectionError("Invalid companion context text.")
    normalized = unicodedata.normalize("NFKC", value)
    if any(
        unicodedata.category(character).startswith("C") and character not in {"\t", "\n", "\r"}
        for character in normalized
    ):
        raise NovaCompanionProjectionError("Invalid companion context text.")
    cleaned = re.sub(r"\s+", " ", normalized).strip()
    if not cleaned:
        return None
    return cleaned[:max_chars]


def _put_optional(target: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        target[key] = value


def _compact_json(payload: object) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    except (TypeError, ValueError, UnicodeError):
        raise NovaCompanionProjectionError("Invalid companion context payload.") from None


def _validate_projection_payload_shape(
    payload: dict[str, object], projection: NovaCompanionContextProjection
) -> None:
    allowed = {
        "confirmed_identity",
        "profile",
        "current_weekly_focus",
        "recent_conversation",
        "active_goals",
        "active_vision_items",
        "confirmed_memory",
    }
    if not set(payload).issubset(allowed):
        _invalid_projection()

    _validate_projection_identity(payload.get("confirmed_identity"))
    _validate_projection_profile(payload.get("profile"))
    _validate_projection_weekly_focus(payload.get("current_weekly_focus"))
    _validate_projection_recent_conversation(
        payload.get("recent_conversation"),
        expected_count=projection.recent_message_count,
    )
    _validate_projection_goals(payload.get("active_goals"), projection.goal_count)
    _validate_projection_vision(payload.get("active_vision_items"), projection.vision_count)
    _validate_projection_memory(payload.get("confirmed_memory"), projection.memory_count)
    if projection.profile_present != ("profile" in payload) or projection.weekly_focus_present != (
        "current_weekly_focus" in payload
    ):
        _invalid_projection()


def _validate_projection_identity(value: object) -> None:
    if value is None:
        return
    identity = _projection_mapping(
        value,
        required=frozenset(),
        allowed=frozenset({"display_name", "location_city", "timezone"}),
    )
    if not identity:
        _invalid_projection()
    if "display_name" in identity:
        _projection_text(identity["display_name"], _IDENTITY_NAME_MAX_CHARS)
    if "location_city" in identity:
        _projection_text(identity["location_city"], _IDENTITY_CITY_MAX_CHARS)
    if "timezone" in identity:
        _projection_text(identity["timezone"], _IDENTITY_TIMEZONE_MAX_CHARS)


def _validate_projection_profile(value: object) -> None:
    if value is None:
        return
    fields = {
        "summary",
        "values",
        "desired_identity",
        "constraints",
        "motivation_style",
    }
    profile = _projection_mapping(value, required=frozenset(), allowed=frozenset(fields))
    if not profile:
        _invalid_projection()
    if "summary" in profile:
        _projection_text(profile["summary"], _PROFILE_SUMMARY_MAX_CHARS)
    if "motivation_style" in profile:
        _projection_text(profile["motivation_style"], _PROFILE_MOTIVATION_MAX_CHARS)
    for key in ("values", "desired_identity", "constraints"):
        if key in profile:
            items = _projection_text_list(
                profile[key],
                max_items=NOVA_COMPANION_PROFILE_LIST_MAX_ITEMS,
                max_chars=_PROFILE_ITEM_MAX_CHARS,
            )
            if items != sorted(set(items), key=lambda item: (item.casefold(), item)):
                _invalid_projection()


def _validate_projection_weekly_focus(value: object) -> None:
    if value is None:
        return
    weekly = _projection_mapping(
        value,
        required=frozenset({"focus"}),
        allowed=frozenset({"focus", "approach", "small_steps"}),
    )
    _projection_text(weekly["focus"], _WEEKLY_FOCUS_MAX_CHARS)
    if "approach" in weekly:
        _projection_text(weekly["approach"], _WEEKLY_APPROACH_MAX_CHARS)
    if "small_steps" in weekly:
        _projection_text_list(
            weekly["small_steps"],
            max_items=3,
            max_chars=_WEEKLY_STEP_MAX_CHARS,
        )


def _validate_projection_recent_conversation(value: object, *, expected_count: int) -> None:
    if value is None:
        if expected_count != 0:
            _invalid_projection()
        return
    recent = _projection_mapping(
        value,
        required=frozenset(),
        allowed=frozenset({"current_topic", "summary", "recent_messages"}),
    )
    if not recent:
        _invalid_projection()
    if "current_topic" in recent:
        _projection_text(recent["current_topic"], _CONVERSATION_TOPIC_MAX_CHARS)
    if "summary" in recent:
        _projection_text(recent["summary"], _CONVERSATION_SUMMARY_MAX_CHARS)
    messages = recent.get("recent_messages")
    if messages is None:
        if expected_count != 0:
            _invalid_projection()
        return
    if (
        not isinstance(messages, list)
        or not 1 <= len(messages) <= NOVA_COMPANION_RECENT_MESSAGE_MAX_ITEMS
        or len(messages) != expected_count
    ):
        _invalid_projection()
    for message in messages:
        record = _projection_mapping(
            message,
            required=frozenset({"role", "content"}),
            allowed=frozenset({"role", "content"}),
        )
        if record["role"] not in {"user", "assistant"}:
            _invalid_projection()
        _projection_text(record["content"], _CONVERSATION_MESSAGE_MAX_CHARS)


def _validate_projection_goals(value: object, expected_count: int) -> None:
    records = _projection_records(
        value,
        expected_count=expected_count,
        max_items=NOVA_COMPANION_GOAL_MAX_ITEMS,
    )
    fields = frozenset(
        {
            "life_area",
            "title",
            "outcome",
            "progress_criterion",
            "horizon",
            "priority",
            "vision_link",
        }
    )
    for item in records:
        goal = _projection_mapping(item, required=fields, allowed=fields)
        _projection_text(goal["life_area"], _GOAL_LIFE_AREA_MAX_CHARS)
        _projection_text(goal["title"], _GOAL_TITLE_MAX_CHARS)
        _projection_text(goal["outcome"], _GOAL_DETAIL_MAX_CHARS)
        _projection_text(goal["progress_criterion"], _GOAL_DETAIL_MAX_CHARS)
        _projection_text(goal["horizon"], _GOAL_HORIZON_MAX_CHARS)
        _projection_text(goal["vision_link"], _GOAL_DETAIL_MAX_CHARS)
        priority = goal["priority"]
        if type(priority) is not int or not 1 <= priority <= 5:
            _invalid_projection()
    expected = sorted(
        records,
        key=lambda item: (
            -int(item["priority"]),
            str(item["title"]).casefold(),
            _compact_json(item),
        ),
    )
    if records != expected:
        _invalid_projection()


def _validate_projection_vision(value: object, expected_count: int) -> None:
    records = _projection_records(
        value,
        expected_count=expected_count,
        max_items=NOVA_COMPANION_VISION_MAX_ITEMS,
    )
    for item in records:
        vision = _projection_mapping(
            item,
            required=frozenset({"category", "wish_text"}),
            allowed=frozenset({"category", "wish_text", "why_text", "first_step"}),
        )
        category = _projection_text(vision["category"], 40)
        if category not in _VISION_CATEGORY_ORDER:
            _invalid_projection()
        _projection_text(vision["wish_text"], _VISION_WISH_MAX_CHARS)
        if "why_text" in vision:
            _projection_text(vision["why_text"], _VISION_WHY_MAX_CHARS)
        if "first_step" in vision:
            _projection_text(vision["first_step"], _VISION_FIRST_STEP_MAX_CHARS)
    expected = sorted(
        records,
        key=lambda item: (
            _VISION_CATEGORY_ORDER.get(str(item["category"]), len(_VISION_CATEGORY_ORDER)),
            str(item["wish_text"]).casefold(),
            _compact_json(item),
        ),
    )
    if records != expected:
        _invalid_projection()


def _validate_projection_memory(value: object, expected_count: int) -> None:
    records = _projection_records(
        value,
        expected_count=expected_count,
        max_items=NOVA_MEMORY_APPLICATION_MAX_ITEMS,
    )
    fields = frozenset({"category", "important", "content"})
    for item in records:
        memory = _projection_mapping(item, required=fields, allowed=fields)
        if memory["category"] not in NOVA_MEMORY_CATEGORIES:
            _invalid_projection()
        if type(memory["important"]) is not bool:
            _invalid_projection()
        # The confirmed projection is already whole-payload bounded.  Its
        # storage-independent constructor intentionally accepts records larger
        # than the CRUD field limit, so mirror that provider contract here.
        _projection_text(memory["content"], NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES)


def _projection_records(
    value: object,
    *,
    expected_count: int,
    max_items: int,
) -> list[dict[str, object]]:
    if value is None:
        if expected_count != 0:
            _invalid_projection()
        return []
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= max_items
        or len(value) != expected_count
        or any(not isinstance(item, dict) for item in value)
    ):
        _invalid_projection()
    return value  # type: ignore[return-value]


def _projection_mapping(
    value: object,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or not set(value).issubset(allowed)
    ):
        _invalid_projection()
    return value


def _projection_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        _invalid_projection()
    cleaned = _optional_text(value, max_chars)
    if cleaned is None or cleaned != value:
        _invalid_projection()
    return value


def _projection_text_list(value: object, *, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= max_items:
        _invalid_projection()
    for item in value:
        _projection_text(item, max_chars)
    return value  # type: ignore[return-value]


def _invalid_projection() -> None:
    raise NovaCompanionProjectionError("Invalid companion context projection.")


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NovaCompanionProjectionError(f"{label} must be a positive integer.")
    return value


def _bounded_positive_integer(
    value: int,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    result = _positive_integer(value, label)
    if not minimum <= result <= maximum:
        raise NovaCompanionProjectionError(f"{label} is outside its bounds.")
    return result


def _full_access_tier(value: str) -> str:
    if not isinstance(value, str) or value not in FULL_ACCESS_TIERS:
        raise NovaCompanionProjectionError("Invalid companion access tier.")
    return value


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise NovaCompanionProjectionError("Invalid companion context clock.")
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "NOVA_COMPANION_CONTEXT_MAX_PAYLOAD_BYTES",
    "NOVA_COMPANION_GOAL_MAX_ITEMS",
    "NOVA_COMPANION_PROFILE_LIST_MAX_ITEMS",
    "NOVA_COMPANION_RECENT_MESSAGE_MAX_ITEMS",
    "NOVA_COMPANION_VISION_MAX_ITEMS",
    "NovaCompanionContextFence",
    "NovaCompanionContextProjection",
    "NovaCompanionContextService",
    "NovaCompanionContextSnapshot",
    "NovaCompanionProjectionError",
    "build_nova_companion_context_projection",
]
