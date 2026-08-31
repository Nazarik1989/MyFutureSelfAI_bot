from __future__ import annotations

import asyncio
import json
import re
import secrets
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal, cast
from uuid import uuid4

from sqlalchemy import delete, func, select, update

from .access import ADMIN, FULL_ACCESS_TIERS, is_full_access_tier
from .conversation import ConversationExchangeSource
from .db import Database
from .models import NovaDialogueState, NovaObservedMemory, User
from .nova_companion_flow import NovaCompanionDiscourseReducer
from .schemas import NovaCompanionDialogueStateUpdate, NovaCompanionMemoryCandidate

NOVA_BRAIN_CONTEXT_MAX_BYTES = 8 * 1024
NOVA_BRAIN_RETRIEVAL_MAX_ITEMS = 6
NOVA_BRAIN_MEMORY_MAX_ITEMS = 100
NOVA_BRAIN_WORKING_STATE_TTL = timedelta(days=30)
NOVA_BRAIN_CALLBACK_PREFIX = "nbrain:"
NOVA_BRAIN_CAPABILITY_TTL = timedelta(minutes=15)

_STRUCTURED_SETTING_ORDER = (
    "response_length",
    "tone",
    "reminder_style",
    "display_name",
    "grammatical_address",
)
_STRUCTURED_SETTING_VALUES = {
    "response_length": frozenset({"short", "normal", "detailed"}),
    "tone": frozenset({"calm", "direct", "supportive"}),
    "reminder_style": frozenset({"gentle", "direct", "brief"}),
    "grammatical_address": frozenset({"masculine", "feminine", "neutral"}),
}

type NovaBrainStatus = Literal["ready", "access_changed", "unavailable"]
type NovaBrainMutationStatus = Literal[
    "applied",
    "unchanged",
    "stale",
    "access_changed",
    "not_found",
]

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-zа-яё0-9]{3,}", re.IGNORECASE)
_DISPLAY_NAME_DECLARATION = re.compile(
    r"\bменя\s+зовут\s+([А-ЯЁA-Z][А-ЯЁа-яёA-Za-z-]{1,49})\b",
    re.IGNORECASE,
)
_GRAMMATICAL_ADDRESS = (
    ("masculine", re.compile(r"\bя\s+мужчина\b", re.IGNORECASE)),
    ("feminine", re.compile(r"\bя\s+женщина\b", re.IGNORECASE)),
    (
        "masculine",
        re.compile(
            r"\b(?:предпочитаю\s+мужское\s+обращение|"
            r"обращайся\s+ко\s+мне\s+в\s+мужском\s+роде)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "feminine",
        re.compile(
            r"\b(?:предпочитаю\s+женское\s+обращение|"
            r"обращайся\s+ко\s+мне\s+в\s+женском\s+роде)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "neutral",
        re.compile(
            r"\b(?:предпочитаю\s+нейтральное\s+обращение|"
            r"обращайся\s+ко\s+мне\s+нейтрально)\b",
            re.IGNORECASE,
        ),
    ),
)
_RESPONSE_LENGTH = (
    (
        "short",
        re.compile(
            r"\b(?:(?:я\s+)?предпочитаю\s+короткие\s+ответы|"
            r"отвечай\s+(?:мне\s+)?коротко|"
            r"пиши\s+(?:мне\s+)?коротко)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "normal",
        re.compile(
            r"\b(?:(?:я\s+)?предпочитаю\s+ответы\s+средней\s+длины|"
            r"отвечай\s+(?:мне\s+)?(?:обычно|нормально))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "detailed",
        re.compile(
            r"\b(?:(?:я\s+)?предпочитаю\s+подробные\s+ответы|"
            r"отвечай\s+(?:мне\s+)?подробно|пиши\s+(?:мне\s+)?подробно)\b",
            re.IGNORECASE,
        ),
    ),
)
_TONE = (
    (
        "calm",
        re.compile(
            r"\b(?:(?:я\s+)?предпочитаю\s+спокойный\s+тон|"
            r"отвечай\s+(?:мне\s+)?спокойно|говори\s+(?:со\s+мной\s+)?спокойно)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "direct",
        re.compile(
            r"\b(?:(?:я\s+)?предпочитаю\s+прямой\s+тон|"
            r"отвечай\s+(?:мне\s+)?прямо|говори\s+(?:со\s+мной\s+)?прямо)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "supportive",
        re.compile(
            r"\b(?:(?:я\s+)?предпочитаю\s+поддерживающий\s+тон|"
            r"отвечай\s+(?:мне\s+)?поддерживающе)\b",
            re.IGNORECASE,
        ),
    ),
)
_REMINDER_STYLE = (
    (
        "gentle",
        re.compile(r"\bнапоминай\s+(?:мне\s+)?мягко\b", re.IGNORECASE),
    ),
    (
        "direct",
        re.compile(r"\bнапоминай\s+(?:мне\s+)?прямо\b", re.IGNORECASE),
    ),
    (
        "brief",
        re.compile(
            r"\b(?:напоминай\s+(?:мне\s+)?кратко|"
            r"(?:я\s+)?предпочитаю\s+краткие\s+напоминания)\b",
            re.IGNORECASE,
        ),
    ),
)
_EXPLICIT_MEMORY = re.compile(r"^\s*(?:нова\s*[,;:—-]?\s*)?запомни\b", re.IGNORECASE)


def _clean_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value)
    if any(
        unicodedata.category(character).startswith("C") and character not in {"\t", "\n", "\r"}
        for character in normalized
    ):
        return None
    cleaned = _SPACE.sub(" ", normalized).strip()
    if not cleaned or len(cleaned) > maximum:
        return None
    return cleaned


def _normalized_value(value: object, *, maximum: int = 500) -> str | None:
    cleaned = _clean_text(value, maximum=maximum)
    return cleaned.casefold() if cleaned is not None else None


def _fingerprint(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _tokens(value: str) -> frozenset[str]:
    return frozenset(match.group(0).casefold() for match in _WORD.finditer(value))


def _matched_enum(
    text: str,
    choices: tuple[tuple[str, re.Pattern[str]], ...],
    *,
    exact: bool = False,
) -> str | None:
    candidate = text.strip().rstrip(".!?").strip() if exact else text
    matched = {
        value
        for value, pattern in choices
        if (pattern.fullmatch(candidate) if exact else pattern.search(candidate)) is not None
    }
    return next(iter(matched)) if len(matched) == 1 else None


def _automatic_memory_utterance(user_text: str) -> str | None:
    """Remove only bounded address/politeness wrappers from a whole user turn."""

    utterance = user_text.strip()
    if "?" in utterance:
        return None
    utterance = re.sub(r"^(?:нова|nova)\s*,\s*", "", utterance, count=1, flags=re.IGNORECASE)
    utterance = re.sub(r"^пожалуйста\s*,\s*", "", utterance, count=1, flags=re.IGNORECASE)
    utterance = re.sub(r"\s*,\s*пожалуйста\s*[.!]*$", "", utterance, count=1, flags=re.IGNORECASE)
    utterance = utterance.rstrip(".!").strip()
    return utterance or None


def _identity_candidate_value(utterance: str) -> str | None:
    clauses = re.split(r"\s*\.\s*", utterance)
    if not clauses or any(not clause for clause in clauses):
        return None
    parsed: dict[str, str] = {}
    for clause in clauses:
        name_match = _DISPLAY_NAME_DECLARATION.fullmatch(clause)
        if name_match is not None:
            if "display_name" in parsed:
                return None
            parsed["display_name"] = name_match.group(1)
            continue
        grammatical_address = _matched_enum(clause, _GRAMMATICAL_ADDRESS, exact=True)
        if grammatical_address is None or "grammatical_address" in parsed:
            return None
        parsed["grammatical_address"] = grammatical_address
    if not parsed:
        return None
    fields = []
    if display_name := parsed.get("display_name"):
        fields.append(f"display_name={display_name}")
    if grammatical_address := parsed.get("grammatical_address"):
        fields.append(f"grammatical_address={grammatical_address}")
    return ";".join(fields)


def _canonical_memory_value(key: str, value: str) -> str | None:
    if key == "identity":
        parts = value.removeprefix("identity:").split(";")
        if not 1 <= len(parts) <= 2:
            return None
        parsed: dict[str, str] = {}
        for part in parts:
            name, separator, field_value = part.partition("=")
            if not separator or name in parsed:
                return None
            parsed[name] = field_value
        if set(parsed).difference({"display_name", "grammatical_address"}):
            return None
        display_name = parsed.get("display_name")
        if (
            display_name is not None
            and re.fullmatch(
                r"[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z-]{1,49}",
                display_name,
                re.IGNORECASE,
            )
            is None
        ):
            return None
        grammatical = parsed.get("grammatical_address")
        if grammatical is not None and grammatical not in {"masculine", "feminine", "neutral"}:
            return None
        if display_name is None and grammatical is None:
            return None
        canonical_parts = []
        if display_name is not None:
            canonical_parts.append(f"display_name={display_name}")
        if grammatical is not None:
            canonical_parts.append(f"grammatical_address={grammatical}")
        return f"identity:{';'.join(canonical_parts)}"
    allowed = {
        "response_length": {"short", "normal", "detailed"},
        "tone": {"calm", "direct", "supportive"},
        "reminder_style": {"gentle", "direct", "brief"},
    }.get(key)
    if allowed is None:
        return None
    raw_value = value.removeprefix(f"{key}=")
    return f"{key}={raw_value}" if raw_value in allowed else None


def _structured_memory_from_user_text(key: str, user_text: str) -> str | None:
    utterance = _automatic_memory_utterance(user_text)
    if utterance is None:
        return None
    utterance = re.sub(r"^теперь\s+", "", utterance, count=1, flags=re.IGNORECASE)
    if key == "identity":
        identity = _identity_candidate_value(utterance)
        return _canonical_memory_value(key, identity) if identity is not None else None
    choices = {
        "response_length": _RESPONSE_LENGTH,
        "tone": _TONE,
        "reminder_style": _REMINDER_STYLE,
    }.get(key)
    if choices is None:
        return None
    matched = _matched_enum(utterance, choices, exact=True)
    return f"{key}={matched}" if matched is not None else None


def grounded_structured_memory(user_text: str) -> tuple[str, str] | None:
    """Return the one exact server-parsed identity/setting declaration in a turn."""

    matches = tuple(
        (key, value)
        for key in ("identity", "response_length", "tone", "reminder_style")
        if (value := _structured_memory_from_user_text(key, user_text)) is not None
    )
    return matches[0] if len(matches) == 1 else None


def effective_structured_settings(values: tuple[str, ...]) -> dict[str, str]:
    """Project validated singleton settings without exposing historical values."""

    result: dict[str, str] = {}
    for value in values:
        if value.startswith("identity:"):
            for part in value.removeprefix("identity:").split(";"):
                name, separator, field_value = part.partition("=")
                if separator and name in {"display_name", "grammatical_address"}:
                    result.setdefault(name, field_value)
            continue
        key, separator, field_value = value.partition("=")
        if separator and key in {"response_length", "tone", "reminder_style"}:
            result.setdefault(key, field_value)
    return result


def _effective_structured_setting_items(values: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    settings = effective_structured_settings(values)
    return tuple((key, settings[key]) for key in _STRUCTURED_SETTING_ORDER if key in settings)


def _is_structured_observed_memory(category: str, value: str) -> bool:
    if value.startswith("identity:"):
        return category == "identity" and _canonical_memory_value("identity", value[9:]) == value
    key, separator, _setting = value.partition("=")
    return (
        category == "preference"
        and bool(separator)
        and _canonical_memory_value(key, value) == value
    )


def _structured_memory_semantic_key(category: str, value: str) -> str | None:
    if not _is_structured_observed_memory(category, value):
        return None
    if category == "identity":
        return "identity"
    key, separator, _setting = value.partition("=")
    return key if separator else None


def _merged_identity_value(current_values: tuple[str, ...], candidate_value: str) -> str:
    fields: dict[str, str] = {}
    for value in reversed(current_values):
        for part in value.removeprefix("identity:").split(";"):
            name, separator, field_value = part.partition("=")
            if separator and name in {"display_name", "grammatical_address"}:
                fields[name] = field_value
    for part in candidate_value.removeprefix("identity:").split(";"):
        name, separator, field_value = part.partition("=")
        if separator and name in {"display_name", "grammatical_address"}:
            fields[name] = field_value
    ordered = ";".join(
        f"{name}={fields[name]}"
        for name in ("display_name", "grammatical_address")
        if name in fields
    )
    canonical = _canonical_memory_value("identity", ordered)
    assert canonical is not None
    return canonical


def observed_memory_display_value(value: str) -> str:
    """Return a bounded human label for an already validated structured value."""

    if value.startswith("identity:"):
        fields = dict(
            part.split("=", 1) for part in value.removeprefix("identity:").split(";") if "=" in part
        )
        labels = []
        if fields.get("display_name"):
            name = fields["display_name"]
            labels.append(f"имя: {name[:1].upper() + name[1:]}")
        grammatical = {
            "masculine": "мужское обращение",
            "feminine": "женское обращение",
            "neutral": "нейтральное обращение",
        }.get(fields.get("grammatical_address", ""))
        if grammatical is not None:
            labels.append(grammatical)
        return "; ".join(labels) if labels else "настройка обращения"
    key, _separator, enum_value = value.partition("=")
    labels = {
        ("response_length", "short"): "короткие ответы",
        ("response_length", "normal"): "ответы обычной длины",
        ("response_length", "detailed"): "подробные ответы",
        ("tone", "calm"): "спокойный тон",
        ("tone", "direct"): "прямой тон",
        ("tone", "supportive"): "поддерживающий тон",
        ("reminder_style", "gentle"): "мягкие напоминания",
        ("reminder_style", "direct"): "прямые напоминания",
        ("reminder_style", "brief"): "краткие напоминания",
    }
    return labels.get((key, enum_value), "настройка общения")


@dataclass(frozen=True, slots=True)
class NovaBrainPolicy:
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
        return (
            type(access_version) is int
            and access_version > 0
            and (expected_access_version is None or access_version == expected_access_version)
        )


@dataclass(frozen=True, slots=True)
class NovaDialogueStateView:
    active_topic: str | None = field(default=None, repr=False)
    current_user_goal: str | None = field(default=None, repr=False)
    last_assistant_offer: str | None = field(default=None, repr=False)
    last_assistant_offer_kinds: tuple[str, ...] = field(default=(), repr=False)
    unresolved_question: str | None = field(default=None, repr=False)
    requested_action: str | None = field(default=None, repr=False)
    open_loops: tuple[str, ...] = field(default=(), repr=False)
    revision: int = 0

    def __post_init__(self) -> None:
        limits = {
            "active_topic": 200,
            "current_user_goal": 300,
            "last_assistant_offer": 600,
            "unresolved_question": 300,
            "requested_action": 16,
        }
        for name, maximum in limits.items():
            value = getattr(self, name)
            if value is not None and _clean_text(value, maximum=maximum) != value:
                raise ValueError("Invalid Nova dialogue state projection")
        if self.requested_action not in {None, "capture", "reminder", "plan", "memory", "clarify"}:
            raise ValueError("Invalid Nova dialogue action")
        if (
            type(self.revision) is not int
            or self.revision < 0
            or type(self.last_assistant_offer_kinds) is not tuple
            or len(self.last_assistant_offer_kinds) > 4
            or any(
                kind not in {"method", "exercise", "reminder_setup", "plan"}
                for kind in self.last_assistant_offer_kinds
            )
            or type(self.open_loops) is not tuple
            or len(self.open_loops) > 5
            or any(_clean_text(loop, maximum=300) != loop for loop in self.open_loops)
        ):
            raise ValueError("Invalid Nova dialogue state projection")

    def provider_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"revision": self.revision}
        for name in (
            "active_topic",
            "current_user_goal",
            "last_assistant_offer",
            "unresolved_question",
            "requested_action",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.last_assistant_offer_kinds:
            payload["last_assistant_offer_kinds"] = list(self.last_assistant_offer_kinds)
        if self.open_loops:
            payload["open_loops"] = list(self.open_loops)
        return payload


@dataclass(frozen=True, slots=True)
class NovaObservedMemoryView:
    public_id: str = field(repr=False)
    category: str
    value: str = field(repr=False)
    salience: int
    revision: int
    updated_at: datetime = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.public_id, str)
            or len(self.public_id) != 36
            or self.category not in {"fact", "preference", "orientation", "theme", "identity"}
            or _clean_text(self.value, maximum=500) != self.value
            or not _is_structured_observed_memory(self.category, self.value)
            or type(self.salience) is not int
            or not 1 <= self.salience <= 5
            or type(self.revision) is not int
            or self.revision <= 0
            or not isinstance(self.updated_at, datetime)
            or self.updated_at.tzinfo is None
        ):
            raise ValueError("Invalid Nova observed-memory projection")

    def provider_payload(self) -> dict[str, object]:
        return {
            "category": self.category,
            "value": self.value,
            "source_status": "observed",
        }


@dataclass(frozen=True, slots=True)
class NovaBrainProjection:
    working_state: NovaDialogueStateView = field(repr=False)
    memories: tuple[NovaObservedMemoryView, ...] = field(default=(), repr=False)
    structured_settings: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    payload_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.working_state) is not NovaDialogueStateView or not isinstance(
            self.memories, tuple
        ):
            raise ValueError("Invalid Nova brain projection")
        if len(self.memories) > 12 or any(
            type(memory) is not NovaObservedMemoryView for memory in self.memories
        ):
            raise ValueError("Invalid Nova brain projection")
        if (
            not isinstance(self.structured_settings, tuple)
            or len(self.structured_settings) > len(_STRUCTURED_SETTING_ORDER)
            or tuple(key for key, _value in self.structured_settings)
            != tuple(
                key for key in _STRUCTURED_SETTING_ORDER if key in dict(self.structured_settings)
            )
            or len(dict(self.structured_settings)) != len(self.structured_settings)
        ):
            raise ValueError("Invalid Nova brain structured settings")
        for key, value in self.structured_settings:
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > 200
                or any(unicodedata.category(character).startswith("C") for character in value)
                or (key == "display_name" and _clean_text(value, maximum=50) != value)
                or (
                    key != "display_name"
                    and value not in _STRUCTURED_SETTING_VALUES.get(key, frozenset())
                )
            ):
                raise ValueError("Invalid Nova brain structured settings")
        actual = len(
            json.dumps(
                self.provider_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        if actual > 32 * 1024 or self.payload_bytes not in {0, actual}:
            raise ValueError("Invalid Nova brain projection size")
        object.__setattr__(self, "payload_bytes", actual)

    def provider_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "working_dialogue_state": self.working_state.provider_payload(),
        }
        if self.memories:
            payload["relevant_observed_memories"] = [
                memory.provider_payload() for memory in self.memories
            ]
        if self.structured_settings:
            payload["active_structured_settings"] = dict(self.structured_settings)
        return payload


@dataclass(frozen=True, slots=True)
class NovaBrainFence:
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_tier: str = field(repr=False)
    access_version: int = field(repr=False)
    state_revision: int = field(repr=False)
    memory_revision: str = field(repr=False)
    retrieved_memory_fingerprints: tuple[str, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.owner_id) is not int
            or self.owner_id <= 0
            or type(self.telegram_user_id) is not int
            or type(self.chat_id) is not int
            or self.access_tier not in FULL_ACCESS_TIERS
            or type(self.access_version) is not int
            or self.access_version <= 0
            or type(self.state_revision) is not int
            or self.state_revision < 0
            or not isinstance(self.memory_revision, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.memory_revision)
            or type(self.retrieved_memory_fingerprints) is not tuple
            or len(self.retrieved_memory_fingerprints) > 12
            or any(
                re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in self.retrieved_memory_fingerprints
            )
        ):
            raise ValueError("Invalid Nova brain fence")


@dataclass(frozen=True, slots=True)
class NovaBrainSnapshot:
    status: NovaBrainStatus
    projection: NovaBrainProjection | None = field(default=None, repr=False)
    fence: NovaBrainFence | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class NovaBrainMutation:
    status: NovaBrainMutationStatus
    item: NovaObservedMemoryView | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _DialogueStateBackup:
    existed: bool
    state_id: int | None = field(default=None, repr=False)
    access_version: int | None = field(default=None, repr=False)
    active_topic: str | None = field(default=None, repr=False)
    current_user_goal: str | None = field(default=None, repr=False)
    last_assistant_offer: str | None = field(default=None, repr=False)
    last_assistant_offer_kinds: tuple[str, ...] = field(default=(), repr=False)
    unresolved_question: str | None = field(default=None, repr=False)
    requested_action: str | None = field(default=None, repr=False)
    open_loops: tuple[str, ...] = field(default=(), repr=False)
    revision: int | None = field(default=None, repr=False)
    expires_at: datetime | None = field(default=None, repr=False)
    updated_at: datetime | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _MemoryBackup:
    memory_id: int = field(repr=False)
    public_id: str = field(repr=False)
    owner_id: int = field(repr=False)
    category: str
    semantic_key: str | None = field(repr=False)
    normalized_value: str = field(repr=False)
    content_fingerprint: str = field(repr=False)
    source_kind: str
    source_session_id: int = field(repr=False)
    source_message_id: int = field(repr=False)
    source_receipt: str = field(repr=False)
    status: str
    salience: int
    revision: int
    superseded_by_public_id: str | None = field(default=None, repr=False)
    created_at: datetime = field(repr=False, default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(repr=False, default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class NovaBrainApplyReceipt:
    prior_fence: NovaBrainFence = field(repr=False)
    result_fence: NovaBrainFence = field(repr=False)
    state_backup: _DialogueStateBackup | None = field(default=None, repr=False)
    installed_state_id: int | None = field(default=None, repr=False)
    installed_state_revision: int | None = field(default=None, repr=False)
    memory_id: int | None = field(default=None, repr=False)
    memory_public_id: str | None = field(default=None, repr=False)
    memory_installed_revision: int | None = field(default=None, repr=False)
    memory_was_created: bool = field(default=False, repr=False)
    memory_backup: _MemoryBackup | None = field(default=None, repr=False)
    superseded_backups: tuple[_MemoryBackup, ...] = field(default=(), repr=False)
    pruned_memories: tuple[_MemoryBackup, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class NovaBrainForgetCapability:
    token: str = field(repr=False)
    action: Literal["confirm", "cancel"]
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_version: int = field(repr=False)
    memory_public_id: str = field(repr=False)
    memory_revision: int = field(repr=False)
    screen_id: str = field(repr=False)
    screen_order: int = field(repr=False)
    expires_at: datetime = field(repr=False)
    canonical_message_id: int | None = field(default=None, repr=False)

    @property
    def callback_data(self) -> str:
        return f"{NOVA_BRAIN_CALLBACK_PREFIX}{self.token}"


@dataclass(frozen=True, slots=True)
class NovaBrainForgetScreen:
    screen_id: str = field(repr=False)
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_version: int = field(repr=False)
    memory_public_id: str = field(repr=False)
    memory_revision: int = field(repr=False)
    screen_order: int = field(repr=False)
    callbacks: tuple[tuple[str, str], ...] = field(repr=False)
    expires_at: datetime = field(repr=False)
    canonical_message_id: int | None = field(default=None, repr=False)

    def callback_data(self, action: Literal["confirm", "cancel"]) -> str:
        return dict(self.callbacks)[action]


class NovaBrainForgetStore:
    """Opaque answer-before-consume capabilities for one exact memory deletion."""

    def __init__(
        self,
        *,
        ttl: timedelta = NOVA_BRAIN_CAPABILITY_TTL,
        max_screens: int = 256,
    ) -> None:
        if not timedelta(minutes=1) <= ttl <= timedelta(hours=1):
            raise ValueError("Invalid Nova brain capability TTL")
        if not 1 <= max_screens <= 4096:
            raise ValueError("Invalid Nova brain capability capacity")
        self.ttl = ttl
        self.max_screens = max_screens
        self._lock = asyncio.Lock()
        self._capabilities: dict[str, NovaBrainForgetCapability] = {}
        self._screens: dict[str, NovaBrainForgetScreen] = {}
        self._screen_order = 0
        self._canonical_generations: dict[tuple[int, int, int, int], tuple[int, datetime]] = {}

    async def stage(
        self,
        memory: NovaObservedMemoryView,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        now: datetime | None = None,
    ) -> NovaBrainForgetScreen:
        current = now or datetime.now(UTC)
        screen_id = secrets.token_urlsafe(18)
        pairs = tuple(
            (action, f"{NOVA_BRAIN_CALLBACK_PREFIX}{secrets.token_urlsafe(24)}")
            for action in ("confirm", "cancel")
        )
        screen = NovaBrainForgetScreen(
            screen_id=screen_id,
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            access_version=access_version,
            memory_public_id=memory.public_id,
            memory_revision=memory.revision,
            screen_order=0,
            callbacks=pairs,
            expires_at=current + self.ttl,
        )
        async with self._lock:
            self._cleanup_locked(current)
            known_callbacks = {
                callback
                for existing in self._screens.values()
                for _action, callback in existing.callbacks
            }
            if screen_id in self._screens or any(
                callback in known_callbacks for _action, callback in pairs
            ):
                raise RuntimeError("Nova brain capability collision")
            if len(self._screens) >= self.max_screens:
                unbound = sorted(
                    (
                        existing
                        for existing in self._screens.values()
                        if existing.canonical_message_id is None
                    ),
                    key=lambda existing: (existing.expires_at, existing.screen_id),
                )
                if not unbound:
                    raise RuntimeError("Nova brain capability capacity reached")
                self._drop_screen_locked(unbound[0].screen_id)
            self._screens[screen_id] = screen
        return screen

    async def bind(
        self,
        expected: NovaBrainForgetScreen,
        *,
        canonical_message_id: int,
        now: datetime | None = None,
    ) -> NovaBrainForgetScreen | None:
        current = now or datetime.now(UTC)
        if canonical_message_id <= 0:
            return None
        async with self._lock:
            self._cleanup_locked(current)
            stored = self._screens.get(expected.screen_id)
            if stored != expected or stored.expires_at <= current:
                return None
            self._screen_order += 1
            bound = NovaBrainForgetScreen(
                screen_id=stored.screen_id,
                owner_id=stored.owner_id,
                telegram_user_id=stored.telegram_user_id,
                chat_id=stored.chat_id,
                access_version=stored.access_version,
                memory_public_id=stored.memory_public_id,
                memory_revision=stored.memory_revision,
                screen_order=self._screen_order,
                callbacks=stored.callbacks,
                canonical_message_id=canonical_message_id,
                expires_at=stored.expires_at,
            )
            canonical_key = self._canonical_key(bound)
            for existing in tuple(self._screens.values()):
                if (
                    existing.screen_id != bound.screen_id
                    and existing.canonical_message_id is not None
                    and self._canonical_key(existing) == canonical_key
                ):
                    self._drop_screen_locked(existing.screen_id)
            self._screens[bound.screen_id] = bound
            self._canonical_generations[canonical_key] = (bound.screen_order, bound.expires_at)
            self._trim_generations_locked()
            for action, callback in bound.callbacks:
                token = callback.removeprefix(NOVA_BRAIN_CALLBACK_PREFIX)
                self._capabilities[token] = NovaBrainForgetCapability(
                    token=token,
                    action=cast(Literal["confirm", "cancel"], action),
                    owner_id=bound.owner_id,
                    telegram_user_id=bound.telegram_user_id,
                    chat_id=bound.chat_id,
                    access_version=bound.access_version,
                    memory_public_id=bound.memory_public_id,
                    memory_revision=bound.memory_revision,
                    screen_id=bound.screen_id,
                    screen_order=bound.screen_order,
                    canonical_message_id=canonical_message_id,
                    expires_at=bound.expires_at,
                )
            return bound

    async def consumed_screen_is_current(
        self,
        expected: NovaBrainForgetCapability,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = now or datetime.now(UTC)
        async with self._lock:
            self._cleanup_locked(current)
            key = self._canonical_key(expected)
            generation = self._canonical_generations.get(key)
            return (
                expected.expires_at > current
                and generation is not None
                and generation[0] == expected.screen_order
            )

    async def stage_recovery(
        self,
        expected: NovaBrainForgetCapability,
        *,
        now: datetime | None = None,
    ) -> NovaBrainForgetScreen | None:
        current = now or datetime.now(UTC)
        callbacks = tuple(
            (action, f"{NOVA_BRAIN_CALLBACK_PREFIX}{secrets.token_urlsafe(24)}")
            for action in ("confirm", "cancel")
        )
        screen_id = secrets.token_urlsafe(18)
        async with self._lock:
            self._cleanup_locked(current)
            key = self._canonical_key(expected)
            generation = self._canonical_generations.get(key)
            if (
                expected.expires_at <= current
                or generation is None
                or generation[0] != expected.screen_order
                or len(self._screens) >= self.max_screens
                or any(
                    callback.removeprefix(NOVA_BRAIN_CALLBACK_PREFIX) in self._capabilities
                    for _action, callback in callbacks
                )
            ):
                return None
            self._screen_order += 1
            screen = NovaBrainForgetScreen(
                screen_id=screen_id,
                owner_id=expected.owner_id,
                telegram_user_id=expected.telegram_user_id,
                chat_id=expected.chat_id,
                access_version=expected.access_version,
                memory_public_id=expected.memory_public_id,
                memory_revision=expected.memory_revision,
                screen_order=self._screen_order,
                callbacks=callbacks,
                expires_at=current + self.ttl,
                canonical_message_id=expected.canonical_message_id,
            )
            self._screens[screen.screen_id] = screen
            self._canonical_generations[key] = (screen.screen_order, screen.expires_at)
            self._trim_generations_locked()
            for action, callback in callbacks:
                token = callback.removeprefix(NOVA_BRAIN_CALLBACK_PREFIX)
                self._capabilities[token] = NovaBrainForgetCapability(
                    token=token,
                    action=cast(Literal["confirm", "cancel"], action),
                    owner_id=screen.owner_id,
                    telegram_user_id=screen.telegram_user_id,
                    chat_id=screen.chat_id,
                    access_version=screen.access_version,
                    memory_public_id=screen.memory_public_id,
                    memory_revision=screen.memory_revision,
                    screen_id=screen.screen_id,
                    screen_order=screen.screen_order,
                    expires_at=screen.expires_at,
                    canonical_message_id=screen.canonical_message_id,
                )
            return screen

    async def peek(
        self,
        callback_data: object,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
        now: datetime | None = None,
    ) -> NovaBrainForgetCapability | None:
        token = self._token(callback_data)
        if token is None:
            return None
        current = now or datetime.now(UTC)
        async with self._lock:
            self._cleanup_locked(current)
            capability = self._capabilities.get(token)
            if (
                capability is None
                or capability.owner_id != owner_id
                or capability.telegram_user_id != telegram_user_id
                or capability.chat_id != chat_id
                or capability.canonical_message_id != canonical_message_id
                or capability.expires_at <= current
            ):
                return None
            return capability

    async def consume(
        self,
        expected: NovaBrainForgetCapability,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = now or datetime.now(UTC)
        async with self._lock:
            self._cleanup_locked(current)
            stored = self._capabilities.get(expected.token)
            if stored != expected or stored.expires_at <= current:
                return False
            self._drop_screen_locked(stored.screen_id)
            return True

    async def revoke(self, screen: NovaBrainForgetScreen) -> None:
        async with self._lock:
            stored = self._screens.get(screen.screen_id)
            if stored == screen:
                self._drop_screen_locked(screen.screen_id)

    def _cleanup_locked(self, now: datetime) -> None:
        for screen_id, screen in tuple(self._screens.items()):
            if screen.expires_at <= now:
                self._drop_screen_locked(screen_id)
        for key, (_order, expires_at) in tuple(self._canonical_generations.items()):
            if expires_at <= now:
                self._canonical_generations.pop(key, None)

    def _drop_screen_locked(self, screen_id: str) -> None:
        screen = self._screens.pop(screen_id, None)
        if screen is None:
            return
        for _action, callback in screen.callbacks:
            self._capabilities.pop(callback.removeprefix(NOVA_BRAIN_CALLBACK_PREFIX), None)

    def _trim_generations_locked(self) -> None:
        limit = self.max_screens * 2
        if len(self._canonical_generations) <= limit:
            return
        live = {
            self._canonical_key(screen)
            for screen in self._screens.values()
            if screen.canonical_message_id is not None
        }
        candidates = sorted(
            ((key, value) for key, value in self._canonical_generations.items() if key not in live),
            key=lambda item: (item[1][1], item[1][0], item[0]),
        )
        for key, _value in candidates:
            if len(self._canonical_generations) <= limit:
                break
            self._canonical_generations.pop(key, None)

    @staticmethod
    def _canonical_key(
        value: NovaBrainForgetScreen | NovaBrainForgetCapability,
    ) -> tuple[int, int, int, int]:
        if value.canonical_message_id is None:
            raise ValueError("Nova brain screen is not bound")
        return (
            value.owner_id,
            value.telegram_user_id,
            value.chat_id,
            value.canonical_message_id,
        )

    @staticmethod
    def _token(callback_data: object) -> str | None:
        if not isinstance(callback_data, str) or not callback_data.startswith(
            NOVA_BRAIN_CALLBACK_PREFIX
        ):
            return None
        token = callback_data.removeprefix(NOVA_BRAIN_CALLBACK_PREFIX)
        return token if re.fullmatch(r"[A-Za-z0-9_-]{16,48}", token) else None


def validate_dialogue_state_update(
    proposal: NovaCompanionDialogueStateUpdate | None,
    *,
    user_text: str,
    assistant_answer: str,
    visible_action: str | None,
) -> NovaCompanionDialogueStateUpdate | None:
    if proposal is None:
        return None
    clean_user = _clean_text(user_text, maximum=4_000)
    clean_answer = _clean_text(assistant_answer, maximum=2_000)
    if clean_user is None or clean_answer is None:
        return None
    user_fold = clean_user.casefold()
    answer_fold = clean_answer.casefold()
    for name in ("active_topic", "current_user_goal"):
        value = getattr(proposal, name)
        if value is not None and value.casefold() not in user_fold:
            return None
    for loop in proposal.open_loops:
        if loop.casefold() not in user_fold:
            return None
    if (
        proposal.unresolved_question is not None
        and proposal.unresolved_question.casefold() not in answer_fold
    ):
        return None
    if proposal.last_assistant_offer is not None:
        if proposal.last_assistant_offer.casefold() not in answer_fold:
            return None
        anchor = NovaCompanionDiscourseReducer.reduce(
            "давай",
            [{"role": "assistant", "content": proposal.last_assistant_offer}],
        )
        if anchor is None or tuple(proposal.last_assistant_offer_kinds) != anchor.offer_kinds:
            return None
    if proposal.requested_action is not None:
        allowed = {"plan", "clarify"}
        if visible_action is not None:
            allowed.add(visible_action)
        if proposal.requested_action not in allowed:
            return None
    validated = proposal.model_copy(deep=True)
    # The model may propose additions grounded in this exact turn, but it never
    # receives authority to erase durable server state.  Topic replacement is
    # handled deterministically by `_apply_state_update` below.
    validated.clear_fields = []
    return validated


def validate_memory_candidate(
    proposal: NovaCompanionMemoryCandidate | None,
    *,
    user_text: str,
) -> NovaCompanionMemoryCandidate | None:
    if proposal is None:
        return None
    clean_user = _clean_text(user_text, maximum=4_000)
    value = _clean_text(proposal.value, maximum=500)
    if clean_user is None or value is None:
        return None
    key = proposal.key
    expected_category = "identity" if key == "identity" else "preference"
    if key is None or proposal.category != expected_category:
        return None
    # Provider-controlled evidence is deliberately not a semantic input.  Only
    # the complete, sanitized current user turn may establish automatic memory.
    canonical = _structured_memory_from_user_text(key, clean_user)
    proposed = _canonical_memory_value(key, value)
    if canonical is None or proposed is None or proposed.casefold() != canonical.casefold():
        return None
    return proposal.model_copy(
        deep=True,
        update={
            "category": expected_category,
            "value": canonical,
            # Replacement authority is derived under the owner lock from the
            # persisted semantic key, never from this provider-controlled hint.
            "supersedes_value": None,
            "salience": 5 if key == "identity" else 4,
        },
    )


class NovaBrainService:
    def __init__(
        self,
        db: Database,
        *,
        max_memories: int = NOVA_BRAIN_MEMORY_MAX_ITEMS,
        retrieval_max_items: int = NOVA_BRAIN_RETRIEVAL_MAX_ITEMS,
        context_max_bytes: int = NOVA_BRAIN_CONTEXT_MAX_BYTES,
        state_ttl: timedelta = NOVA_BRAIN_WORKING_STATE_TTL,
    ) -> None:
        if not 1 <= max_memories <= 500:
            raise ValueError("Invalid Nova brain memory limit")
        if not 1 <= retrieval_max_items <= 12:
            raise ValueError("Invalid Nova brain retrieval limit")
        if not 1_024 <= context_max_bytes <= 32 * 1024:
            raise ValueError("Invalid Nova brain context byte limit")
        if not timedelta(hours=1) <= state_ttl <= timedelta(days=365):
            raise ValueError("Invalid Nova brain state TTL")
        self.db = db
        self.max_memories = max_memories
        self.retrieval_max_items = retrieval_max_items
        self.context_max_bytes = context_max_bytes
        self.state_ttl = state_ttl

    async def snapshot(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_tier: str,
        expected_access_version: int,
        current_text: str,
        policy: NovaBrainPolicy,
        now: datetime | None = None,
    ) -> NovaBrainSnapshot:
        if not policy.allows_tier(expected_tier):
            return NovaBrainSnapshot("access_changed")
        current = now or datetime.now(UTC)
        await self._cleanup_expired_state(
            telegram_actor_id=telegram_actor_id,
            chat_id=chat_id,
            expected_tier=expected_tier,
            expected_access_version=expected_access_version,
            current=current,
        )
        async with self.db.sessions() as session:
            async with session.begin():
                actor = await session.scalar(
                    select(User).where(
                        User.telegram_id == telegram_actor_id,
                        User.access_tier == expected_tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == expected_access_version,
                    )
                )
                if actor is None:
                    return NovaBrainSnapshot("access_changed")
                state = await session.scalar(
                    select(NovaDialogueState).where(
                        NovaDialogueState.owner_id == actor.id,
                        NovaDialogueState.telegram_user_id == telegram_actor_id,
                        NovaDialogueState.chat_id == chat_id,
                    )
                )
                memories = list(
                    (
                        await session.scalars(
                            select(NovaObservedMemory)
                            .where(
                                NovaObservedMemory.owner_id == actor.id,
                                NovaObservedMemory.status == "active",
                            )
                            .order_by(
                                NovaObservedMemory.updated_at.desc(),
                                NovaObservedMemory.id.desc(),
                            )
                            .limit(self.max_memories)
                        )
                    ).all()
                )
        state_view = self._state_view(
            state
            if state is not None
            and state.access_version == expected_access_version
            and self._aware(state.expires_at) > current
            else None
        )
        memory_views = self._memory_views(memories)
        selected = self._retrieve(current_text, state_view, memory_views)
        structured_settings = self._active_structured_setting_items(memories)
        projection = self._fit_projection(state_view, selected, structured_settings)
        fence = NovaBrainFence(
            owner_id=actor.id,
            telegram_user_id=telegram_actor_id,
            chat_id=chat_id,
            access_tier=expected_tier,
            access_version=expected_access_version,
            state_revision=state_view.revision,
            memory_revision=self._memory_revision(actor.id, memory_views),
            retrieved_memory_fingerprints=tuple(_fingerprint(memory.value) for memory in selected),
        )
        return NovaBrainSnapshot("ready", projection=projection, fence=fence)

    async def _cleanup_expired_state(
        self,
        *,
        telegram_actor_id: int,
        chat_id: int,
        expected_tier: str,
        expected_access_version: int,
        current: datetime,
    ) -> bool:
        async with self.db.sessions() as session:
            candidate = await session.execute(
                select(
                    NovaDialogueState.id,
                    NovaDialogueState.owner_id,
                    NovaDialogueState.revision,
                    NovaDialogueState.expires_at,
                )
                .join(User, User.id == NovaDialogueState.owner_id)
                .where(
                    User.telegram_id == telegram_actor_id,
                    User.access_tier == expected_tier,
                    User.access_version == expected_access_version,
                    NovaDialogueState.telegram_user_id == telegram_actor_id,
                    NovaDialogueState.chat_id == chat_id,
                    NovaDialogueState.access_version == expected_access_version,
                    NovaDialogueState.expires_at <= current,
                )
            )
            frozen = candidate.one_or_none()
        if frozen is None:
            return False
        state_id, owner_id, revision, expires_at = frozen
        await self._before_expired_state_cleanup(state_id, revision)
        async with self.db.sessions() as session:
            async with session.begin():
                actor_lock = await session.execute(
                    update(User)
                    .where(
                        User.id == owner_id,
                        User.telegram_id == telegram_actor_id,
                        User.access_tier == expected_tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == expected_access_version,
                    )
                    .values(updated_at=User.updated_at)
                    .execution_options(synchronize_session=False)
                )
                if actor_lock.rowcount != 1:
                    return False
                removed = await session.execute(
                    delete(NovaDialogueState).where(
                        NovaDialogueState.id == state_id,
                        NovaDialogueState.owner_id == owner_id,
                        NovaDialogueState.telegram_user_id == telegram_actor_id,
                        NovaDialogueState.chat_id == chat_id,
                        NovaDialogueState.access_version == expected_access_version,
                        NovaDialogueState.revision == revision,
                        NovaDialogueState.expires_at == expires_at,
                        NovaDialogueState.expires_at <= current,
                    )
                )
                return removed.rowcount == 1

    async def _before_expired_state_cleanup(self, state_id: int, revision: int) -> None:
        del state_id, revision

    async def current_check(
        self,
        fence: NovaBrainFence,
        *,
        policy: NovaBrainPolicy,
        now: datetime | None = None,
    ) -> bool:
        if type(fence) is not NovaBrainFence or not policy.allows_tier(fence.access_tier):
            return False
        snapshot = await self.snapshot(
            telegram_actor_id=fence.telegram_user_id,
            chat_id=fence.chat_id,
            expected_tier=fence.access_tier,
            expected_access_version=fence.access_version,
            current_text="",
            policy=policy,
            now=now,
        )
        return snapshot.status == "ready" and snapshot.fence == fence

    async def apply_turn(
        self,
        fence: NovaBrainFence,
        source: ConversationExchangeSource,
        *,
        state_update: NovaCompanionDialogueStateUpdate | None,
        memory_candidate: NovaCompanionMemoryCandidate | None,
        user_text: str,
        policy: NovaBrainPolicy,
        now: datetime | None = None,
    ) -> NovaBrainApplyReceipt | None:
        if type(fence) is not NovaBrainFence or type(source) is not ConversationExchangeSource:
            raise ValueError("Invalid Nova brain turn")
        if state_update is None and memory_candidate is None:
            return None
        if not policy.allows_tier(fence.access_tier):
            return None
        current = now or datetime.now(UTC)
        receipt: NovaBrainApplyReceipt | None = None
        try:
            async with self.db.sessions() as session:
                async with session.begin():
                    actor_lock = await session.execute(
                        update(User)
                        .where(
                            User.id == fence.owner_id,
                            User.telegram_id == fence.telegram_user_id,
                            User.access_tier == fence.access_tier,
                            User.access_tier.in_(FULL_ACCESS_TIERS),
                            User.access_version == fence.access_version,
                        )
                        .values(updated_at=User.updated_at)
                        .execution_options(synchronize_session=False)
                    )
                    if actor_lock.rowcount != 1:
                        return None
                    await session.execute(
                        update(NovaDialogueState)
                        .where(
                            NovaDialogueState.owner_id == fence.owner_id,
                            NovaDialogueState.telegram_user_id == fence.telegram_user_id,
                            NovaDialogueState.chat_id == fence.chat_id,
                        )
                        .values(updated_at=NovaDialogueState.updated_at)
                        .execution_options(synchronize_session=False)
                    )
                    state = await session.scalar(
                        select(NovaDialogueState).where(
                            NovaDialogueState.owner_id == fence.owner_id,
                            NovaDialogueState.telegram_user_id == fence.telegram_user_id,
                            NovaDialogueState.chat_id == fence.chat_id,
                        )
                    )
                    memories = list(
                        (
                            await session.scalars(
                                select(NovaObservedMemory)
                                .where(
                                    NovaObservedMemory.owner_id == fence.owner_id,
                                    NovaObservedMemory.status == "active",
                                )
                                .order_by(
                                    NovaObservedMemory.updated_at.desc(),
                                    NovaObservedMemory.id.desc(),
                                )
                                .limit(self.max_memories)
                            )
                        ).all()
                    )
                    current_state = self._state_view(
                        state
                        if state is not None
                        and state.access_version == fence.access_version
                        and self._aware(state.expires_at) > current
                        else None
                    )
                    current_fence = NovaBrainFence(
                        owner_id=fence.owner_id,
                        telegram_user_id=fence.telegram_user_id,
                        chat_id=fence.chat_id,
                        access_tier=fence.access_tier,
                        access_version=fence.access_version,
                        state_revision=current_state.revision,
                        memory_revision=self._memory_revision(
                            fence.owner_id,
                            self._memory_views(memories),
                        ),
                    )
                    if current_fence != fence:
                        return None

                    state_backup = None
                    installed_state_id = None
                    installed_state_revision = None
                    if state_update is not None:
                        state_backup = self._state_backup(state)
                        if state is None:
                            state = NovaDialogueState(
                                owner_id=fence.owner_id,
                                telegram_user_id=fence.telegram_user_id,
                                chat_id=fence.chat_id,
                                access_version=fence.access_version,
                                last_assistant_offer_kinds=[],
                                open_loops=[],
                                revision=1,
                                expires_at=current + self.state_ttl,
                            )
                            session.add(state)
                            await session.flush()
                        else:
                            if current_state.revision == 0:
                                state.active_topic = None
                                state.current_user_goal = None
                                state.last_assistant_offer = None
                                state.last_assistant_offer_kinds = []
                                state.unresolved_question = None
                                state.requested_action = None
                                state.open_loops = []
                            state.access_version = fence.access_version
                            state.revision += 1
                            state.expires_at = current + self.state_ttl
                        self._apply_state_update(state, state_update)
                        await session.flush()
                        installed_state_id = state.id
                        installed_state_revision = state.revision

                    memory_id = None
                    memory_public_id = None
                    memory_installed_revision = None
                    memory_was_created = False
                    memory_backup = None
                    superseded_backups: tuple[_MemoryBackup, ...] = ()
                    if memory_candidate is not None:
                        (
                            memory_row,
                            memory_was_created,
                            memory_backup,
                            superseded_backups,
                            pruned_memories,
                        ) = await self._upsert_memory(
                            session,
                            fence=fence,
                            source=source,
                            candidate=memory_candidate,
                            user_text=user_text,
                            current=current,
                        )
                        if memory_row is not None:
                            memory_id = memory_row.id
                            memory_public_id = memory_row.public_id
                            memory_installed_revision = memory_row.revision

                    result_memories = list(
                        (
                            await session.scalars(
                                select(NovaObservedMemory)
                                .where(
                                    NovaObservedMemory.owner_id == fence.owner_id,
                                    NovaObservedMemory.status == "active",
                                )
                                .order_by(
                                    NovaObservedMemory.updated_at.desc(),
                                    NovaObservedMemory.id.desc(),
                                )
                                .limit(self.max_memories)
                            )
                        ).all()
                    )
                    result_state = self._state_view(state if state_update is not None else None)
                    if state_update is None:
                        result_state = current_state
                    result_fence = NovaBrainFence(
                        owner_id=fence.owner_id,
                        telegram_user_id=fence.telegram_user_id,
                        chat_id=fence.chat_id,
                        access_tier=fence.access_tier,
                        access_version=fence.access_version,
                        state_revision=result_state.revision,
                        memory_revision=self._memory_revision(
                            fence.owner_id,
                            self._memory_views(result_memories),
                        ),
                    )
                    receipt = NovaBrainApplyReceipt(
                        prior_fence=fence,
                        result_fence=result_fence,
                        state_backup=state_backup,
                        installed_state_id=installed_state_id,
                        installed_state_revision=installed_state_revision,
                        memory_id=memory_id,
                        memory_public_id=memory_public_id,
                        memory_installed_revision=memory_installed_revision,
                        memory_was_created=memory_was_created,
                        memory_backup=memory_backup,
                        superseded_backups=superseded_backups,
                        pruned_memories=(pruned_memories if memory_candidate is not None else ()),
                    )
            await self._after_apply_commit(receipt)
            return receipt
        except (Exception, asyncio.CancelledError):
            if receipt is not None:
                await self._shielded_compensation(receipt)
            raise

    async def compensate_turn(self, receipt: NovaBrainApplyReceipt) -> bool:
        if type(receipt) is not NovaBrainApplyReceipt:
            raise ValueError("Invalid Nova brain receipt")
        clean = True
        async with self.db.sessions() as session:
            async with session.begin():
                await session.execute(
                    update(User)
                    .where(User.id == receipt.prior_fence.owner_id)
                    .values(updated_at=User.updated_at)
                    .execution_options(synchronize_session=False)
                )
                if receipt.installed_state_id is not None:
                    state = await session.get(NovaDialogueState, receipt.installed_state_id)
                    if (
                        state is None
                        or state.revision != receipt.installed_state_revision
                        or state.owner_id != receipt.prior_fence.owner_id
                    ):
                        clean = False
                    elif receipt.state_backup is not None and receipt.state_backup.existed:
                        self._restore_state(state, receipt.state_backup)
                    else:
                        await session.delete(state)
                if receipt.memory_id is not None:
                    memory_compensated = False
                    memory = await session.get(NovaObservedMemory, receipt.memory_id)
                    if (
                        memory is None
                        or memory.owner_id != receipt.prior_fence.owner_id
                        or memory.public_id != receipt.memory_public_id
                        or memory.revision != receipt.memory_installed_revision
                    ):
                        clean = False
                    elif receipt.memory_was_created:
                        await session.delete(memory)
                        await session.flush()
                        memory_compensated = True
                    elif receipt.memory_backup is not None:
                        self._restore_memory(memory, receipt.memory_backup)
                        await session.flush()
                        memory_compensated = True
                else:
                    memory_compensated = not receipt.superseded_backups
                for superseded_backup in receipt.superseded_backups:
                    prior = await session.get(
                        NovaObservedMemory,
                        superseded_backup.memory_id,
                    )
                    if (
                        not memory_compensated
                        or prior is None
                        or prior.owner_id != receipt.prior_fence.owner_id
                        or prior.revision != superseded_backup.revision + 1
                        or prior.superseded_by_public_id != receipt.memory_public_id
                    ):
                        clean = False
                    else:
                        self._restore_memory(prior, superseded_backup)
                for backup in receipt.pruned_memories:
                    conflict = await session.scalar(
                        select(NovaObservedMemory.id).where(
                            (NovaObservedMemory.id == backup.memory_id)
                            | (NovaObservedMemory.public_id == backup.public_id)
                            | (
                                (NovaObservedMemory.owner_id == backup.owner_id)
                                & (
                                    NovaObservedMemory.content_fingerprint
                                    == backup.content_fingerprint
                                )
                            )
                        )
                    )
                    if conflict is not None:
                        clean = False
                        continue
                    session.add(self._memory_from_backup(backup))
        return clean

    async def list_current(
        self,
        *,
        telegram_actor_id: int,
        expected_access_version: int,
        policy: NovaBrainPolicy,
    ) -> tuple[NovaObservedMemoryView, ...]:
        async with self.db.sessions() as session:
            actor = await session.scalar(
                select(User).where(
                    User.telegram_id == telegram_actor_id,
                    User.access_version == expected_access_version,
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                )
            )
            if actor is None or not policy.allows_tier(actor.access_tier):
                return ()
            rows = list(
                (
                    await session.scalars(
                        select(NovaObservedMemory)
                        .where(
                            NovaObservedMemory.owner_id == actor.id,
                            NovaObservedMemory.status == "active",
                        )
                        .order_by(
                            NovaObservedMemory.salience.desc(),
                            NovaObservedMemory.updated_at.desc(),
                            NovaObservedMemory.id.desc(),
                        )
                        .limit(self.max_memories)
                    )
                ).all()
            )
        return self._memory_views(rows)

    async def forget_exact(
        self,
        *,
        telegram_actor_id: int,
        public_id: str,
        expected_revision: int,
        expected_access_version: int,
        policy: NovaBrainPolicy,
    ) -> NovaBrainMutation:
        async with self.db.sessions() as session:
            async with session.begin():
                actor = await session.scalar(
                    select(User).where(
                        User.telegram_id == telegram_actor_id,
                        User.access_version == expected_access_version,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                    )
                )
                if actor is None or not policy.allows_tier(actor.access_tier):
                    return NovaBrainMutation("access_changed")
                await session.execute(
                    update(User)
                    .where(User.id == actor.id)
                    .values(updated_at=User.updated_at)
                    .execution_options(synchronize_session=False)
                )
                item = await session.scalar(
                    select(NovaObservedMemory).where(
                        NovaObservedMemory.owner_id == actor.id,
                        NovaObservedMemory.public_id == public_id,
                    )
                )
                if item is None or item.status != "active":
                    return NovaBrainMutation("not_found")
                if item.revision != expected_revision:
                    return NovaBrainMutation("stale")
                item.status = "forgotten"
                item.revision += 1
                item.updated_at = datetime.now(UTC)
                await session.flush()
                return NovaBrainMutation("applied", self._memory_view(item))

    async def _upsert_memory(
        self,
        session: Any,
        *,
        fence: NovaBrainFence,
        source: ConversationExchangeSource,
        candidate: NovaCompanionMemoryCandidate,
        user_text: str,
        current: datetime,
    ) -> tuple[
        NovaObservedMemory | None,
        bool,
        _MemoryBackup | None,
        tuple[_MemoryBackup, ...],
        tuple[_MemoryBackup, ...],
    ]:
        validated = validate_memory_candidate(candidate, user_text=user_text)
        if validated is None:
            return None, False, None, (), ()
        value = _normalized_value(validated.value)
        assert value is not None
        semantic_key = _structured_memory_semantic_key(validated.category, value)
        assert semantic_key is not None

        active_rows = list(
            (
                await session.scalars(
                    select(NovaObservedMemory)
                    .where(
                        NovaObservedMemory.owner_id == fence.owner_id,
                        NovaObservedMemory.status == "active",
                    )
                    .order_by(
                        NovaObservedMemory.updated_at.desc(),
                        NovaObservedMemory.id.desc(),
                    )
                )
            ).all()
        )
        same_key_rows = tuple(
            row
            for row in active_rows
            if _structured_memory_semantic_key(row.category, row.normalized_value) == semantic_key
        )
        if semantic_key == "identity":
            value = _normalized_value(
                _merged_identity_value(
                    tuple(row.normalized_value for row in same_key_rows),
                    value,
                )
            )
            assert value is not None
        fingerprint = _fingerprint(value)
        row = await session.scalar(
            select(NovaObservedMemory).where(
                NovaObservedMemory.owner_id == fence.owner_id,
                NovaObservedMemory.content_fingerprint == fingerprint,
            )
        )
        if (
            row is not None
            and row.status == "active"
            and len(same_key_rows) == 1
            and same_key_rows[0].id == row.id
            and row.semantic_key == semantic_key
        ):
            return None, False, None, (), ()
        if len(active_rows) >= self.max_memories and not same_key_rows:
            return None, False, None, (), ()
        source_kind = "explicit" if _EXPLICIT_MEMORY.search(user_text) else "conversation"
        pruned_memories: tuple[_MemoryBackup, ...] = ()
        if row is None:
            history_limit = self.max_memories * 2
            total_count = int(
                await session.scalar(
                    select(func.count(NovaObservedMemory.id)).where(
                        NovaObservedMemory.owner_id == fence.owner_id
                    )
                )
                or 0
            )
            if total_count >= history_limit:
                purge_count = total_count - history_limit + 1
                purge_rows = tuple(
                    (
                        await session.scalars(
                            select(NovaObservedMemory)
                            .where(
                                NovaObservedMemory.owner_id == fence.owner_id,
                                NovaObservedMemory.status != "active",
                            )
                            .order_by(
                                NovaObservedMemory.updated_at,
                                NovaObservedMemory.id,
                            )
                            .limit(purge_count)
                        )
                    ).all()
                )
                if len(purge_rows) != purge_count:
                    return None, False, None, (), ()
                pruned_memories = tuple(self._memory_backup(item) for item in purge_rows)
                await session.execute(
                    delete(NovaObservedMemory).where(
                        NovaObservedMemory.id.in_(tuple(item.id for item in purge_rows))
                    )
                )
        public_id = row.public_id if row is not None else str(uuid4())
        superseded_rows = tuple(item for item in same_key_rows if row is None or item.id != row.id)
        superseded_backups = tuple(self._memory_backup(item) for item in superseded_rows)
        for item in superseded_rows:
            item.status = "superseded"
            item.superseded_by_public_id = public_id
            item.revision += 1
            item.updated_at = current
        if superseded_rows:
            await session.flush()

        memory_backup = None
        memory_was_created = row is None
        if row is None:
            row = NovaObservedMemory(
                public_id=public_id,
                owner_id=fence.owner_id,
                category=validated.category,
                semantic_key=semantic_key,
                normalized_value=value,
                content_fingerprint=fingerprint,
                source_kind=source_kind,
                source_session_id=source.session_id,
                source_message_id=source.user_message_id,
                source_receipt=source.receipt,
                status="active",
                salience=validated.salience,
                revision=1,
            )
            session.add(row)
            await session.flush()
            return row, True, None, superseded_backups, pruned_memories
        memory_backup = self._memory_backup(row)
        row.category = validated.category
        row.semantic_key = semantic_key
        row.normalized_value = value
        row.content_fingerprint = fingerprint
        row.source_kind = source_kind
        row.source_session_id = source.session_id
        row.source_message_id = source.user_message_id
        row.source_receipt = source.receipt
        row.status = "active"
        row.salience = validated.salience
        row.revision += 1
        row.superseded_by_public_id = None
        row.updated_at = current
        await session.flush()
        return row, memory_was_created, memory_backup, superseded_backups, pruned_memories

    def _retrieve(
        self,
        current_text: str,
        state: NovaDialogueStateView,
        memories: tuple[NovaObservedMemoryView, ...],
    ) -> tuple[NovaObservedMemoryView, ...]:
        query_tokens = set(_tokens(current_text))
        for value in (state.active_topic, state.current_user_goal, *state.open_loops):
            if value:
                query_tokens.update(_tokens(value))
        category_weight = {
            "identity": 40,
            "preference": 30,
            "orientation": 25,
            "theme": 15,
            "fact": 10,
        }
        scored: list[tuple[int, NovaObservedMemoryView]] = []
        for memory in memories:
            overlap = len(query_tokens.intersection(_tokens(memory.value)))
            score = overlap * 100 + category_weight[memory.category] + memory.salience * 10
            if overlap == 0 and memory.category not in {"identity", "preference", "orientation"}:
                continue
            scored.append((score, memory))
        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1].updated_at.timestamp(),
                item[1].public_id,
            )
        )
        selected: list[NovaObservedMemoryView] = []
        categories: dict[str, int] = {}
        for _score, memory in scored:
            if categories.get(memory.category, 0) >= 2:
                continue
            selected.append(memory)
            categories[memory.category] = categories.get(memory.category, 0) + 1
            if len(selected) >= self.retrieval_max_items:
                break
        return tuple(selected)

    def _fit_projection(
        self,
        state: NovaDialogueStateView,
        memories: tuple[NovaObservedMemoryView, ...],
        structured_settings: tuple[tuple[str, str], ...] = (),
    ) -> NovaBrainProjection:
        selected = list(memories)
        fitted_state = state
        while True:
            projection = NovaBrainProjection(
                fitted_state,
                tuple(selected),
                structured_settings=structured_settings,
            )
            if projection.payload_bytes <= self.context_max_bytes:
                return projection
            if selected:
                selected.pop()
                continue
            reduced = self._reduce_state_projection(fitted_state)
            if reduced == fitted_state:
                raise ValueError("Nova brain working state exceeds context budget")
            fitted_state = reduced

    @staticmethod
    def _reduce_state_projection(state: NovaDialogueStateView) -> NovaDialogueStateView:
        values = {
            "active_topic": state.active_topic,
            "current_user_goal": state.current_user_goal,
            "last_assistant_offer": state.last_assistant_offer,
            "last_assistant_offer_kinds": state.last_assistant_offer_kinds,
            "unresolved_question": state.unresolved_question,
            "requested_action": state.requested_action,
            "open_loops": state.open_loops,
            "revision": state.revision,
        }
        if values["open_loops"]:
            values["open_loops"] = values["open_loops"][:-1]
        elif values["last_assistant_offer"] is not None:
            values["last_assistant_offer"] = None
            values["last_assistant_offer_kinds"] = ()
        elif values["unresolved_question"] is not None:
            values["unresolved_question"] = None
        elif values["requested_action"] is not None:
            values["requested_action"] = None
        elif values["current_user_goal"] is not None:
            values["current_user_goal"] = None
        elif values["active_topic"] is not None:
            values["active_topic"] = None
        else:
            return state
        return NovaDialogueStateView(**values)

    @staticmethod
    def _state_view(state: NovaDialogueState | None) -> NovaDialogueStateView:
        if state is None:
            return NovaDialogueStateView()
        return NovaDialogueStateView(
            active_topic=state.active_topic,
            current_user_goal=state.current_user_goal,
            last_assistant_offer=state.last_assistant_offer,
            last_assistant_offer_kinds=tuple(state.last_assistant_offer_kinds or ()),
            unresolved_question=state.unresolved_question,
            requested_action=state.requested_action,
            open_loops=tuple(state.open_loops or ()),
            revision=state.revision,
        )

    @staticmethod
    def _memory_view(row: NovaObservedMemory) -> NovaObservedMemoryView:
        return NovaObservedMemoryView(
            public_id=row.public_id,
            category=row.category,
            value=row.normalized_value,
            salience=row.salience,
            revision=row.revision,
            updated_at=NovaBrainService._aware(row.updated_at),
        )

    @staticmethod
    def _memory_views(rows: list[NovaObservedMemory]) -> tuple[NovaObservedMemoryView, ...]:
        views = []
        seen_keys: set[str] = set()
        for row in rows:
            semantic_key = _structured_memory_semantic_key(row.category, row.normalized_value)
            if (
                semantic_key is None
                or row.semantic_key != semantic_key
                or semantic_key in seen_keys
            ):
                continue
            seen_keys.add(semantic_key)
            views.append(NovaBrainService._memory_view(row))
        return tuple(views)

    @staticmethod
    def _active_structured_setting_items(
        rows: list[NovaObservedMemory],
    ) -> tuple[tuple[str, str], ...]:
        values: list[str] = []
        seen_keys: set[str] = set()
        for row in rows:
            semantic_key = _structured_memory_semantic_key(row.category, row.normalized_value)
            if (
                semantic_key is None
                or row.status != "active"
                or row.semantic_key != semantic_key
                or semantic_key in seen_keys
                or semantic_key not in {"response_length", "tone", "reminder_style", "identity"}
            ):
                continue
            seen_keys.add(semantic_key)
            values.append(row.normalized_value)
        return _effective_structured_setting_items(tuple(values))

    @staticmethod
    def _memory_revision(owner_id: int, rows: tuple[NovaObservedMemoryView, ...]) -> str:
        manifest = {
            "owner": owner_id,
            "items": [
                {
                    "id": row.public_id,
                    "category": row.category,
                    "value": row.value,
                    "salience": row.salience,
                    "revision": row.revision,
                }
                for row in sorted(rows, key=lambda item: item.public_id)
            ],
        }
        return sha256(
            json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _state_backup(state: NovaDialogueState | None) -> _DialogueStateBackup:
        if state is None:
            return _DialogueStateBackup(False)
        return _DialogueStateBackup(
            True,
            state_id=state.id,
            access_version=state.access_version,
            active_topic=state.active_topic,
            current_user_goal=state.current_user_goal,
            last_assistant_offer=state.last_assistant_offer,
            last_assistant_offer_kinds=tuple(state.last_assistant_offer_kinds or ()),
            unresolved_question=state.unresolved_question,
            requested_action=state.requested_action,
            open_loops=tuple(state.open_loops or ()),
            revision=state.revision,
            expires_at=state.expires_at,
            updated_at=NovaBrainService._aware(state.updated_at),
        )

    @staticmethod
    def _memory_backup(row: NovaObservedMemory) -> _MemoryBackup:
        return _MemoryBackup(
            memory_id=row.id,
            public_id=row.public_id,
            owner_id=row.owner_id,
            category=row.category,
            semantic_key=row.semantic_key,
            normalized_value=row.normalized_value,
            content_fingerprint=row.content_fingerprint,
            source_kind=row.source_kind,
            source_session_id=row.source_session_id,
            source_message_id=row.source_message_id,
            source_receipt=row.source_receipt,
            status=row.status,
            salience=row.salience,
            revision=row.revision,
            superseded_by_public_id=row.superseded_by_public_id,
            created_at=NovaBrainService._aware(row.created_at),
            updated_at=NovaBrainService._aware(row.updated_at),
        )

    @staticmethod
    def _memory_from_backup(backup: _MemoryBackup) -> NovaObservedMemory:
        return NovaObservedMemory(
            id=backup.memory_id,
            public_id=backup.public_id,
            owner_id=backup.owner_id,
            category=backup.category,
            semantic_key=backup.semantic_key,
            normalized_value=backup.normalized_value,
            content_fingerprint=backup.content_fingerprint,
            source_kind=backup.source_kind,
            source_session_id=backup.source_session_id,
            source_message_id=backup.source_message_id,
            source_receipt=backup.source_receipt,
            status=backup.status,
            salience=backup.salience,
            revision=backup.revision,
            superseded_by_public_id=backup.superseded_by_public_id,
            created_at=backup.created_at,
            updated_at=backup.updated_at,
        )

    @staticmethod
    def _apply_state_update(
        state: NovaDialogueState,
        proposal: NovaCompanionDialogueStateUpdate,
    ) -> None:
        if proposal.active_topic is not None and proposal.active_topic != state.active_topic:
            state.last_assistant_offer = None
            state.last_assistant_offer_kinds = []
            state.unresolved_question = None
            state.requested_action = None
            state.open_loops = []
        for name in (
            "active_topic",
            "current_user_goal",
            "last_assistant_offer",
            "unresolved_question",
            "requested_action",
        ):
            value = getattr(proposal, name)
            if value is not None:
                setattr(state, name, value)
        if proposal.last_assistant_offer is not None:
            state.last_assistant_offer_kinds = list(proposal.last_assistant_offer_kinds)
        if proposal.open_loops:
            state.open_loops = list(proposal.open_loops)
        for name in proposal.clear_fields:
            if name == "open_loops":
                state.open_loops = []
            elif name == "last_assistant_offer":
                state.last_assistant_offer = None
                state.last_assistant_offer_kinds = []
            else:
                setattr(state, name, None)

    @staticmethod
    def _restore_state(state: NovaDialogueState, backup: _DialogueStateBackup) -> None:
        state.access_version = backup.access_version or state.access_version
        state.active_topic = backup.active_topic
        state.current_user_goal = backup.current_user_goal
        state.last_assistant_offer = backup.last_assistant_offer
        state.last_assistant_offer_kinds = list(backup.last_assistant_offer_kinds)
        state.unresolved_question = backup.unresolved_question
        state.requested_action = backup.requested_action
        state.open_loops = list(backup.open_loops)
        state.revision = backup.revision or 1
        state.expires_at = backup.expires_at or state.expires_at
        if backup.updated_at is not None:
            state.updated_at = backup.updated_at

    @staticmethod
    def _restore_memory(memory: NovaObservedMemory, backup: _MemoryBackup) -> None:
        memory.category = backup.category
        memory.semantic_key = backup.semantic_key
        memory.normalized_value = backup.normalized_value
        memory.content_fingerprint = backup.content_fingerprint
        memory.source_kind = backup.source_kind
        memory.source_session_id = backup.source_session_id
        memory.source_message_id = backup.source_message_id
        memory.source_receipt = backup.source_receipt
        memory.status = backup.status
        memory.salience = backup.salience
        memory.revision = backup.revision
        memory.superseded_by_public_id = backup.superseded_by_public_id
        memory.updated_at = backup.updated_at

    @staticmethod
    def _aware(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    async def _after_apply_commit(self, receipt: NovaBrainApplyReceipt) -> None:
        del receipt

    async def _shielded_compensation(self, receipt: NovaBrainApplyReceipt) -> None:
        task = asyncio.create_task(
            self.compensate_turn(receipt),
            name="nova-brain-apply-compensation",
        )
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                continue
        task.result()
        if cancelled:
            raise asyncio.CancelledError


__all__ = [
    "NOVA_BRAIN_CONTEXT_MAX_BYTES",
    "NOVA_BRAIN_CALLBACK_PREFIX",
    "NOVA_BRAIN_MEMORY_MAX_ITEMS",
    "NOVA_BRAIN_RETRIEVAL_MAX_ITEMS",
    "NovaBrainApplyReceipt",
    "NovaBrainFence",
    "NovaBrainForgetCapability",
    "NovaBrainForgetScreen",
    "NovaBrainForgetStore",
    "NovaBrainMutation",
    "NovaBrainPolicy",
    "NovaBrainProjection",
    "NovaBrainService",
    "NovaBrainSnapshot",
    "NovaDialogueStateView",
    "NovaObservedMemoryView",
    "observed_memory_display_value",
    "validate_dialogue_state_update",
    "validate_memory_candidate",
]
