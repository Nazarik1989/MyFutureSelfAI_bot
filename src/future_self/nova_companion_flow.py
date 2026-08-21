from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .access import ADMIN, is_full_access_tier
from .dates import WEEKDAYS, DateOption, DateResolution

NOVA_COMPANION_CALLBACK_PREFIX = "ncap:"
NOVA_COMPANION_REMINDER_CALLBACK_PREFIX = "nrem:"
NOVA_COMPANION_CAPABILITY_TTL = timedelta(minutes=15)
NOVA_COMPANION_MAX_CAPABILITIES = 2_000

type CaptureKind = Literal["idea", "task", "desire", "note"]
type CaptureAction = Literal["add", "not_now", "date_first", "date_second"]
type ReminderOfferAction = Literal["accept", "not_now"]

CAPTURE_KINDS = frozenset({"idea", "task", "desire", "note"})
CAPTURE_ACTIONS: tuple[CaptureAction, ...] = ("add", "not_now")
CAPTURE_DATE_ACTIONS: tuple[CaptureAction, ...] = (
    "date_first",
    "date_second",
    "not_now",
)

_ACTION_PATTERN = re.compile(r"(?:add|not_now|date_first|date_second)\Z")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{16,48}\Z")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NOVA_NAME = r"(?:nova|нова)"
_TRAILING_PUNCTUATION = r"[?!.…]*"


@dataclass(frozen=True, slots=True)
class NovaCompanionPolicy:
    """Fail-closed pilot policy shared by all companion entry points."""

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
        if (
            isinstance(access_version, bool)
            or not isinstance(access_version, int)
            or access_version <= 0
        ):
            return False
        return expected_access_version is None or access_version == expected_access_version


class NovaAddressKind(StrEnum):
    NONE = "none"
    WAKE = "wake"
    PRESENCE = "presence"
    IDENTITY = "identity"
    VOCATIVE = "vocative"


@dataclass(frozen=True, slots=True)
class NovaAddressResult:
    kind: NovaAddressKind
    content: str | None = field(default=None, repr=False)

    @property
    def matched(self) -> bool:
        return self.kind is not NovaAddressKind.NONE

    @property
    def is_local_response(self) -> bool:
        return self.kind in {
            NovaAddressKind.WAKE,
            NovaAddressKind.PRESENCE,
            NovaAddressKind.IDENTITY,
        }

    @property
    def local_response(self) -> str | None:
        if self.kind in {NovaAddressKind.WAKE, NovaAddressKind.PRESENCE}:
            return "Да, я здесь 🙂"
        if self.kind is NovaAddressKind.IDENTITY:
            return "Да, я Nova. Я рядом — о чём хочешь поговорить?"
        return None


class NovaAddressClassifier:
    """Conservative name/wake classifier for text and transcribed voice alike."""

    _WAKE = re.compile(
        rf"^[«\"'“]?{_NOVA_NAME}[»\"'”]?\s*{_TRAILING_PUNCTUATION}$",
        re.IGNORECASE,
    )
    _PRESENCE = re.compile(
        rf"^{_NOVA_NAME}\s*[,—:;-]?\s*(?:ты\s+)?(?:тут|здесь|рядом)"
        rf"(?:\s+со\s+мной)?\s*{_TRAILING_PUNCTUATION}$",
        re.IGNORECASE,
    )
    _PRESENCE_EN = re.compile(
        rf"^{_NOVA_NAME}\s*[,—:;-]?\s*(?:are\s+you\s+)?(?:here|there)"
        rf"\s*{_TRAILING_PUNCTUATION}$",
        re.IGNORECASE,
    )
    _IDENTITY = re.compile(
        rf"^ты\s*[-—:]?\s*{_NOVA_NAME}\s*{_TRAILING_PUNCTUATION}$",
        re.IGNORECASE,
    )
    _IDENTITY_EN = re.compile(
        rf"^(?:are\s+you|you(?:'re|\s+are))\s+{_NOVA_NAME}\s*"
        rf"{_TRAILING_PUNCTUATION}$",
        re.IGNORECASE,
    )
    _VOCATIVE = re.compile(
        rf"^{_NOVA_NAME}(?:\s*[,;:—-]\s*|\s+)(?P<content>.+)$",
        re.IGNORECASE,
    )

    @classmethod
    def classify(cls, text: object) -> NovaAddressResult:
        cleaned = _safe_text(text, max_chars=4_000)
        if cleaned is None:
            return NovaAddressResult(NovaAddressKind.NONE)
        if cls._WAKE.fullmatch(cleaned):
            return NovaAddressResult(NovaAddressKind.WAKE)
        if cls._PRESENCE.fullmatch(cleaned) or cls._PRESENCE_EN.fullmatch(cleaned):
            return NovaAddressResult(NovaAddressKind.PRESENCE)
        if cls._IDENTITY.fullmatch(cleaned) or cls._IDENTITY_EN.fullmatch(cleaned):
            return NovaAddressResult(NovaAddressKind.IDENTITY)
        match = cls._VOCATIVE.fullmatch(cleaned)
        if match is None:
            return NovaAddressResult(NovaAddressKind.NONE)
        content = match.group("content").strip(" \t,;:—-")
        return (
            NovaAddressResult(NovaAddressKind.VOCATIVE, content=content)
            if content
            else NovaAddressResult(NovaAddressKind.WAKE)
        )


@dataclass(frozen=True, slots=True)
class ExplicitCaptureIntent:
    kind: CaptureKind
    content: str | None = field(default=None, repr=False)
    references_context: bool = False


class ExplicitCaptureClassifier:
    """Match only unequivocal capture commands; reflections stay unmatched."""

    _KIND = r"(?:идею|идея|задачу|задача|желание|желания|заметку|заметка|запись)"
    _CREATE = re.compile(
        rf"^создай\s+(?P<kind>{_KIND})(?:(?:\s*[:—-]\s*|\s+)(?P<content>.+))?$",
        re.IGNORECASE,
    )
    _ADD_AS = re.compile(
        rf"^добавь\s+(?:(?P<reference>это)\s+)?как\s+(?P<kind>{_KIND})"
        rf"(?:(?:\s*[:—-]\s*|\s+)(?P<content>.+))?$",
        re.IGNORECASE,
    )
    _DIRECT_TYPED = re.compile(
        rf"^(?:добавь|запиши|сохрани)\s+(?P<kind>{_KIND})"
        rf"(?:(?:\s*[:—-]\s*|\s+)(?P<content>.+))?$",
        re.IGNORECASE,
    )
    _WRITE_AS_PREFIX = re.compile(
        rf"^(?:запиши|сохрани)\s+как\s+(?P<kind>{_KIND})"
        rf"(?:(?:\s*[:—-]\s*|\s+)(?P<content>.+))?$",
        re.IGNORECASE,
    )
    _WRITE_AS_SUFFIX = re.compile(
        rf"^(?:запиши|сохрани)\s+(?P<content>.+?)\s+как\s+(?P<kind>{_KIND})$",
        re.IGNORECASE,
    )
    _WRITE = re.compile(r"^(?:запиши|сохрани)\s+(?P<content>.+)$", re.IGNORECASE)
    _CONTEXT_REFERENCES = frozenset(
        {
            "это",
            "эту мысль",
            "эту идею",
            "последнее",
            "предыдущее",
            "мой ответ",
            "наш разговор",
        }
    )

    @classmethod
    def classify(cls, text: object) -> ExplicitCaptureIntent | None:
        cleaned = _safe_text(text, max_chars=4_000)
        if cleaned is None:
            return None
        addressed = NovaAddressClassifier.classify(cleaned)
        if addressed.kind is NovaAddressKind.VOCATIVE and addressed.content is not None:
            cleaned = addressed.content
        # Declarative capture commands commonly end in sentence punctuation;
        # it is not part of either the capture kind or a context reference.
        cleaned = cleaned.rstrip(" .!…")

        match = cls._CREATE.fullmatch(cleaned)
        if match is not None:
            return cls._result(match.group("kind"), match.group("content"))
        match = cls._ADD_AS.fullmatch(cleaned)
        if match is not None:
            result = cls._result(match.group("kind"), match.group("content"))
            if result is None:
                return None
            return replace(
                result,
                references_context=bool(match.group("reference")) or result.content is None,
            )
        match = cls._DIRECT_TYPED.fullmatch(cleaned)
        if match is not None:
            return cls._result(match.group("kind"), match.group("content"))
        match = cls._WRITE_AS_PREFIX.fullmatch(cleaned)
        if match is not None:
            return cls._result(match.group("kind"), match.group("content"))
        match = cls._WRITE_AS_SUFFIX.fullmatch(cleaned)
        if match is not None:
            return cls._result(match.group("kind"), match.group("content"))
        match = cls._WRITE.fullmatch(cleaned)
        if match is None:
            return None
        return cls._result("заметка", match.group("content"))

    @classmethod
    def _result(cls, kind: str, content: str | None) -> ExplicitCaptureIntent | None:
        capture_kind = _capture_kind(kind)
        if capture_kind is None:
            return None
        clean_content = _safe_text(content, max_chars=2_000) if content is not None else None
        if content is not None and clean_content is None:
            return None
        references_context = bool(
            clean_content is not None and clean_content.casefold() in cls._CONTEXT_REFERENCES
        )
        return ExplicitCaptureIntent(
            capture_kind,
            content=None if references_context else clean_content,
            references_context=references_context,
        )


@dataclass(frozen=True, slots=True)
class CaptureSuggestion:
    kind: CaptureKind
    title: str = field(repr=False)
    next_step: str | None = field(default=None, repr=False)
    fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in CAPTURE_KINDS:
            raise ValueError("unsupported companion capture kind")
        clean_title = _safe_text(self.title, max_chars=120)
        if clean_title is None or len(clean_title) < 2:
            raise ValueError("companion capture title must contain 2..120 safe characters")
        clean_next_step = (
            _safe_text(self.next_step, max_chars=240) if self.next_step is not None else None
        )
        if self.next_step is not None and clean_next_step is None:
            raise ValueError("companion capture next step must contain safe text")
        object.__setattr__(self, "title", clean_title)
        object.__setattr__(self, "next_step", clean_next_step)
        normalized_title = unicodedata.normalize("NFKC", clean_title).casefold()
        fingerprint = hashlib.sha256(f"{self.kind}\0{normalized_title}".encode()).hexdigest()
        object.__setattr__(self, "fingerprint", fingerprint)


_CAPTURE_OPTOUT = re.compile(
    r"\b(?:не\s+(?:сохраняй|записывай|надо|нужно)|просто\s+(?:поговорить|пообщаться))\b",
    re.IGNORECASE,
)
_UNCERTAIN_REFLECTION = re.compile(
    r"\b(?:может\s+быть|наверное|кажется|не\s+знаю|просто\s+думаю)\b",
    re.IGNORECASE,
)
_EMOTION_ONLY = re.compile(
    r"^(?:мне\s+(?:грустно|тяжело|страшно|одиноко)|я\s+(?:устал(?:а)?|расстроен(?:а)?)|"
    r"меня\s+(?:бесит|тревожит)|я\s+постоянно\s+забываю\b)",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(?:привет|здравствуй(?:те)?|доброе\s+(?:утро|утречко)|добрый\s+(?:день|вечер)|"
    r"hello|hi)\s*[!.,…]*$",
    re.IGNORECASE,
)
_QUESTION_LEAD = re.compile(
    r"^(?:кто|что|где|куда|откуда|когда|почему|зачем|как|какой|какая|какие|можно\s+ли|"
    r"стоит\s+ли)\b",
    re.IGNORECASE,
)
_CONCRETE_CAPTURE_CUE = re.compile(
    r"\b(?:идея|задача|инсайт|хочу|мечтаю|планирую|собираюсь|"
    r"нужно|надо|важно\s+(?:запомнить|учесть|сохранить)|"
    r"можно\s+(?:сделать|создать|добавить|попробовать|организовать)|"
    r"следующ(?:ий|ая|ее)\s+шаг)\b",
    re.IGNORECASE,
)
_EXACT_NO_CAPTURE = frozenset(
    {
        "я постоянно забываю о главном, из-за каждодневной суеты",
        "я постоянно забываю о главном из-за каждодневной суеты",
    }
)


def should_offer_capture(user_text: object) -> bool:
    """Apply deterministic fail-closed suppression before showing a provider suggestion."""

    cleaned = _safe_text(user_text, max_chars=4_000)
    if cleaned is None:
        return False
    normalized = cleaned.casefold().strip(" .!?…")
    if normalized in _EXACT_NO_CAPTURE:
        return False
    if NovaAddressClassifier.classify(cleaned).is_local_response:
        return False
    if (
        _CAPTURE_OPTOUT.search(cleaned)
        or _UNCERTAIN_REFLECTION.search(cleaned)
        or _EMOTION_ONLY.search(cleaned)
        or _GREETING.fullmatch(cleaned)
        or _QUESTION_LEAD.match(cleaned)
        or cleaned.rstrip().endswith("?")
    ):
        return False
    return _CONCRETE_CAPTURE_CUE.search(cleaned) is not None


def validate_capture_suggestion(
    *,
    kind: object,
    title: object,
    next_step: object = None,
    user_text: object,
) -> CaptureSuggestion | None:
    """Validate untrusted suggestion output and apply local semantic suppression."""

    if not should_offer_capture(user_text):
        return None
    if not isinstance(kind, str) or kind not in CAPTURE_KINDS or not isinstance(title, str):
        return None
    if next_step is not None and not isinstance(next_step, str):
        return None
    try:
        suggestion = CaptureSuggestion(cast(CaptureKind, kind), title, next_step)
    except ValueError:
        return None
    grounded_text = _grounding_text(user_text)
    if _grounding_text(suggestion.title) not in grounded_text:
        return None
    if (
        suggestion.next_step is not None
        and _grounding_text(suggestion.next_step) not in grounded_text
    ):
        return None
    return suggestion


@dataclass(frozen=True, slots=True, init=False)
class NovaCompanionCaptureTemporal:
    """Immutable, privacy-safe projection of one server-side date resolution."""

    _timezone: str = field(repr=False)
    _status: Literal["resolved", "conflict"] = field(repr=False)
    _target_date: date = field(repr=False)
    _stated_weekday: str | None = field(repr=False)
    _actual_weekday: str = field(repr=False)
    _inferred_year: bool = field(repr=False)
    _options: tuple[tuple[date, str], ...] = field(repr=False)
    _local_time: time | None = field(repr=False)

    def __init__(
        self,
        *,
        timezone: str,
        resolution: DateResolution,
        local_time: time | None,
    ) -> None:
        clean_timezone = _safe_text(timezone, max_chars=128)
        if clean_timezone is None or clean_timezone != timezone:
            raise ValueError("companion capture timezone must be a valid IANA timezone")
        try:
            timezone_key = ZoneInfo(clean_timezone).key
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("companion capture timezone must be a valid IANA timezone") from None
        if not isinstance(resolution, DateResolution) or resolution.status not in {
            "resolved",
            "conflict",
        }:
            raise ValueError("companion capture date resolution must be resolved or conflict")
        if type(resolution.target_date) is not date:
            raise ValueError("companion capture date resolution must have a target date")
        if type(resolution.inferred_year) is not bool:
            raise ValueError("companion capture inferred-year marker must be boolean")
        expected_actual_weekday = WEEKDAYS[resolution.target_date.weekday()]
        if resolution.actual_weekday != expected_actual_weekday:
            raise ValueError("companion capture actual weekday does not match target date")
        if resolution.stated_weekday is not None and resolution.stated_weekday not in WEEKDAYS:
            raise ValueError("companion capture stated weekday is invalid")
        if local_time is not None and (
            type(local_time) is not time
            or local_time.tzinfo is not None
            or local_time.second != 0
            or local_time.microsecond != 0
        ):
            raise ValueError("companion capture local time must be a timezone-naive minute")

        if not isinstance(resolution.options, list):
            raise ValueError("companion capture date options must be a list")
        options: list[tuple[date, str]] = []
        for option in resolution.options:
            if (
                not isinstance(option, DateOption)
                or type(option.value) is not date
                or option.weekday != WEEKDAYS[option.value.weekday()]
            ):
                raise ValueError("companion capture date option is invalid")
            options.append((option.value, option.weekday))
        frozen_options = tuple(options)
        if resolution.status == "resolved":
            if frozen_options or (
                resolution.stated_weekday is not None
                and resolution.stated_weekday != expected_actual_weekday
            ):
                raise ValueError("companion capture resolved date is inconsistent")
        else:
            option_dates = tuple(value for value, _weekday in frozen_options)
            if (
                resolution.stated_weekday is None
                or resolution.stated_weekday == expected_actual_weekday
                or len(frozen_options) != 2
                or len(set(option_dates)) != 2
                or resolution.target_date not in option_dates
                or not any(
                    weekday == resolution.stated_weekday for _value, weekday in frozen_options
                )
            ):
                raise ValueError("companion capture conflicting date is inconsistent")

        object.__setattr__(self, "_timezone", timezone_key)
        object.__setattr__(
            self, "_status", cast(Literal["resolved", "conflict"], resolution.status)
        )
        object.__setattr__(self, "_target_date", resolution.target_date)
        object.__setattr__(self, "_stated_weekday", resolution.stated_weekday)
        object.__setattr__(self, "_actual_weekday", expected_actual_weekday)
        object.__setattr__(self, "_inferred_year", resolution.inferred_year)
        object.__setattr__(self, "_options", frozen_options)
        object.__setattr__(self, "_local_time", local_time)

    @property
    def timezone(self) -> str:
        return self._timezone

    @property
    def resolution(self) -> DateResolution:
        return DateResolution(
            status=self._status,
            target_date=self._target_date,
            stated_weekday=self._stated_weekday,
            actual_weekday=self._actual_weekday,
            inferred_year=self._inferred_year,
            options=[DateOption(value=value, weekday=weekday) for value, weekday in self._options],
        )

    @property
    def local_time(self) -> time | None:
        return self._local_time


@dataclass(frozen=True, slots=True)
class NovaCompanionCaptureCapability:
    token: str = field(repr=False)
    screen_id: str = field(repr=False)
    action: CaptureAction
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    canonical_message_id: int | None = field(repr=False)
    access_version: int = field(repr=False)
    suggestion: CaptureSuggestion = field(repr=False)
    raw_text: str = field(repr=False)
    screen_order: int
    expires_at: datetime = field(repr=False)
    temporal: NovaCompanionCaptureTemporal | None = field(default=None, repr=False)

    @property
    def callback_data(self) -> str:
        return f"{NOVA_COMPANION_CALLBACK_PREFIX}{self.token}"


@dataclass(frozen=True, slots=True)
class NovaCompanionCaptureScreen:
    screen_id: str = field(repr=False)
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    canonical_message_id: int | None = field(repr=False)
    access_version: int = field(repr=False)
    suggestion: CaptureSuggestion = field(repr=False)
    raw_text: str = field(repr=False)
    screen_order: int
    expires_at: datetime = field(repr=False)
    _callbacks: tuple[tuple[CaptureAction, str], ...] = field(repr=False)
    temporal: NovaCompanionCaptureTemporal | None = field(default=None, repr=False)

    @property
    def is_bound(self) -> bool:
        return self.canonical_message_id is not None

    @property
    def callbacks(self) -> MappingProxyType[CaptureAction, str]:
        return MappingProxyType(dict(self._callbacks))

    def callback_data(self, action: CaptureAction) -> str:
        for candidate, callback_data in self._callbacks:
            if candidate == action:
                return callback_data
        raise KeyError(action)


class NovaCompanionCaptureStore:
    """Bounded process-local capabilities for optional companion capture.

    Staged controls are opaque but not claimable.  ``bind`` atomically adds the
    accepted Telegram message id and publishes that exact screen generation.
    No database, Telegram, or provider await occurs under the store lock.
    """

    def __init__(
        self,
        *,
        ttl: timedelta = NOVA_COMPANION_CAPABILITY_TTL,
        max_capabilities: int = NOVA_COMPANION_MAX_CAPABILITIES,
    ) -> None:
        if ttl < timedelta(seconds=1) or ttl > timedelta(hours=2):
            raise ValueError("companion capture ttl must be between 1 second and 2 hours")
        if not 2 <= max_capabilities <= 20_000:
            raise ValueError("companion capture capability limit must be between 2 and 20000")
        self.ttl = ttl
        self.max_capabilities = max_capabilities
        self._capabilities: dict[str, NovaCompanionCaptureCapability] = {}
        self._screens: dict[str, NovaCompanionCaptureScreen] = {}
        self._suppressions: dict[tuple[int, int, int, int, str], datetime] = {}
        self._canonical_generations: dict[tuple[int, int, int, int], tuple[int, datetime]] = {}
        self._next_screen_order = 0
        self._lock = asyncio.Lock()

    async def stage(
        self,
        suggestion: CaptureSuggestion,
        *,
        raw_text: str,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        temporal: NovaCompanionCaptureTemporal | None = None,
        actions: tuple[CaptureAction, ...] = CAPTURE_ACTIONS,
        now: datetime | None = None,
    ) -> NovaCompanionCaptureScreen | None:
        self._validate_binding(owner_id, telegram_user_id, chat_id, access_version)
        if not isinstance(suggestion, CaptureSuggestion):
            raise ValueError("validated companion capture suggestion is required")
        self._validate_temporal(temporal)
        validated_actions = self._validate_actions(actions)
        self._validate_temporal_actions(temporal, validated_actions)
        clean_raw_text = _safe_text(raw_text, max_chars=4_000)
        if clean_raw_text is None:
            raise ValueError("companion capture source text must contain safe text")
        current = self._utc(now)
        suppression_key = self._suppression_key(
            owner_id,
            telegram_user_id,
            chat_id,
            access_version,
            suggestion.fingerprint,
        )
        async with self._lock:
            self._cleanup_locked(current)
            if suppression_key in self._suppressions:
                return None
            if any(
                screen.owner_id == owner_id
                and screen.telegram_user_id == telegram_user_id
                and screen.chat_id == chat_id
                and screen.access_version == access_version
                and screen.suggestion.fingerprint == suggestion.fingerprint
                for screen in self._screens.values()
            ):
                # Only one offer for the same owner/topic generation may be
                # renderable.  Retiring a duplicate after it was published
                # would leave a second visible keyboard whose opaque tokens
                # are already dead when the user dismisses the first offer.
                return None
            return self._publish_screen_locked(
                suggestion,
                raw_text=clean_raw_text,
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                canonical_message_id=None,
                access_version=access_version,
                temporal=temporal,
                actions=validated_actions,
                now=current,
            )

    async def stage_recovery(
        self,
        expected_consumed: NovaCompanionCaptureCapability,
        *,
        actions: tuple[CaptureAction, ...] = CAPTURE_ACTIONS,
        now: datetime | None = None,
    ) -> NovaCompanionCaptureScreen | None:
        """Atomically publish controls replacing one exact consumed generation.

        The returned screen is already bound to the consumed capability's
        canonical Telegram message.  Checking the consumed tombstone,
        allocating the newer order, publishing its opaque capabilities, and
        advancing the tombstone happen under one lock, so a concurrent
        replacement cannot slip between a read-only current check and bind.
        """

        validated_actions = self._validate_actions(actions)
        if (
            not isinstance(expected_consumed, NovaCompanionCaptureCapability)
            or expected_consumed.canonical_message_id is None
        ):
            return None
        try:
            self._validate_binding(
                expected_consumed.owner_id,
                expected_consumed.telegram_user_id,
                expected_consumed.chat_id,
                expected_consumed.access_version,
            )
            self._positive(
                expected_consumed.canonical_message_id,
                "canonical message id",
            )
        except ValueError:
            return None
        if not isinstance(expected_consumed.suggestion, CaptureSuggestion):
            return None
        try:
            self._validate_temporal(expected_consumed.temporal)
        except ValueError:
            return None
        self._validate_temporal_actions(expected_consumed.temporal, validated_actions)
        clean_raw_text = _safe_text(expected_consumed.raw_text, max_chars=4_000)
        if clean_raw_text is None:
            return None
        current = self._utc(now)
        suppression_key = self._suppression_key(
            expected_consumed.owner_id,
            expected_consumed.telegram_user_id,
            expected_consumed.chat_id,
            expected_consumed.access_version,
            expected_consumed.suggestion.fingerprint,
        )
        canonical_key = self._canonical_key(
            expected_consumed.owner_id,
            expected_consumed.telegram_user_id,
            expected_consumed.chat_id,
            expected_consumed.canonical_message_id,
        )
        async with self._lock:
            self._cleanup_locked(current)
            latest = self._canonical_generations.get(canonical_key)
            if (
                expected_consumed.expires_at <= current
                or self._capabilities.get(expected_consumed.token) is not None
                or expected_consumed.screen_id in self._screens
                or latest is None
                or latest[0] != expected_consumed.screen_order
                or suppression_key in self._suppressions
            ):
                return None
            if any(
                screen.owner_id == expected_consumed.owner_id
                and screen.telegram_user_id == expected_consumed.telegram_user_id
                and screen.chat_id == expected_consumed.chat_id
                and screen.access_version == expected_consumed.access_version
                and screen.suggestion.fingerprint == expected_consumed.suggestion.fingerprint
                for screen in self._screens.values()
            ):
                return None
            return self._publish_screen_locked(
                expected_consumed.suggestion,
                raw_text=clean_raw_text,
                owner_id=expected_consumed.owner_id,
                telegram_user_id=expected_consumed.telegram_user_id,
                chat_id=expected_consumed.chat_id,
                canonical_message_id=expected_consumed.canonical_message_id,
                access_version=expected_consumed.access_version,
                temporal=expected_consumed.temporal,
                actions=validated_actions,
                now=current,
            )

    async def bind(
        self,
        expected: NovaCompanionCaptureScreen,
        *,
        canonical_message_id: int,
        now: datetime | None = None,
    ) -> NovaCompanionCaptureScreen | None:
        if not isinstance(expected, NovaCompanionCaptureScreen) or expected.is_bound:
            return None
        self._positive(canonical_message_id, "canonical message id")
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            stored = self._screens.get(expected.screen_id)
            if stored != expected or stored.expires_at <= current:
                return None
            canonical_key = self._canonical_key(
                expected.owner_id,
                expected.telegram_user_id,
                expected.chat_id,
                canonical_message_id,
            )
            latest = self._canonical_generations.get(canonical_key)
            if latest is not None and latest[0] > expected.screen_order:
                self._drop_screen_locked(expected.screen_id)
                return None
            newer = tuple(
                screen
                for screen in self._screens.values()
                if screen.screen_id != expected.screen_id
                and screen.canonical_message_id == canonical_message_id
                and screen.owner_id == expected.owner_id
                and screen.telegram_user_id == expected.telegram_user_id
                and screen.chat_id == expected.chat_id
                and screen.screen_order > expected.screen_order
            )
            if newer:
                self._drop_screen_locked(expected.screen_id)
                return None
            bound = replace(expected, canonical_message_id=canonical_message_id)
            self._screens[bound.screen_id] = bound
            for token, capability in tuple(self._capabilities.items()):
                if capability.screen_id == bound.screen_id:
                    self._capabilities[token] = replace(
                        capability,
                        canonical_message_id=canonical_message_id,
                    )
            self._record_canonical_generation_locked(bound)
            obsolete = tuple(
                screen.screen_id
                for screen in self._screens.values()
                if screen.screen_id != bound.screen_id
                and screen.canonical_message_id == canonical_message_id
                and screen.owner_id == bound.owner_id
                and screen.telegram_user_id == bound.telegram_user_id
                and screen.chat_id == bound.chat_id
                and screen.screen_order < bound.screen_order
            )
            for screen_id in obsolete:
                self._drop_screen_locked(screen_id)
            return bound

    async def peek(
        self,
        callback_data: object,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
        access_version: int,
        expected_action: CaptureAction | None = None,
        now: datetime | None = None,
    ) -> NovaCompanionCaptureCapability | None:
        token = self._callback_token(callback_data)
        if token is None:
            return None
        if expected_action is not None and not self._is_action(expected_action):
            return None
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            capability = self._capabilities.get(token)
            if capability is None or capability.canonical_message_id is None:
                return None
            if (
                capability.owner_id != owner_id
                or capability.telegram_user_id != telegram_user_id
                or capability.chat_id != chat_id
                or capability.canonical_message_id != canonical_message_id
            ):
                return None
            if expected_action is not None and capability.action != expected_action:
                return None
            if capability.access_version != access_version:
                self._drop_screen_locked(capability.screen_id)
                return None
            return capability

    async def peek_bound_identity(
        self,
        callback_data: object,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
        expected_action: CaptureAction | None = None,
        now: datetime | None = None,
    ) -> NovaCompanionCaptureCapability | None:
        """Read an exact owner-bound capability without applying an access fence.

        This narrow lookup lets a callback handler neutralize an old same-owner
        screen after an access-version bounce.  It never consumes or revokes a
        capability, and a cross-owner, cross-actor, cross-chat, or
        cross-message lookup remains opaque.
        """

        token = self._callback_token(callback_data)
        if token is None:
            return None
        if expected_action is not None and not self._is_action(expected_action):
            return None
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            capability = self._capabilities.get(token)
            if capability is None or capability.canonical_message_id is None:
                return None
            if (
                capability.owner_id != owner_id
                or capability.telegram_user_id != telegram_user_id
                or capability.chat_id != chat_id
                or capability.canonical_message_id != canonical_message_id
            ):
                return None
            if expected_action is not None and capability.action != expected_action:
                return None
            return capability

    async def consume(
        self,
        expected: NovaCompanionCaptureCapability,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            stored = self._capabilities.get(expected.token)
            if (
                stored is None
                or stored != expected
                or stored.canonical_message_id is None
                or stored.expires_at <= current
            ):
                return False
            self._drop_screen_locked(stored.screen_id)
            if stored.action == "not_now":
                self._mark_suppressed_locked(
                    stored.owner_id,
                    stored.telegram_user_id,
                    stored.chat_id,
                    stored.access_version,
                    stored.suggestion.fingerprint,
                    current,
                )
            return True

    async def consumed_screen_is_current(
        self,
        expected: NovaCompanionCaptureCapability,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Check a consumed screen against the latest bound canonical generation.

        Consumption removes the capability and screen, but recovery still needs
        to know whether its exact generation may edit the canonical message.  A
        bounded TTL tombstone prevents an older callback from overwriting a
        newer replacement even after that replacement was itself consumed.
        """

        if (
            not isinstance(expected, NovaCompanionCaptureCapability)
            or expected.canonical_message_id is None
        ):
            return False
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            if expected.expires_at <= current:
                return False
            key = self._canonical_key(
                expected.owner_id,
                expected.telegram_user_id,
                expected.chat_id,
                expected.canonical_message_id,
            )
            latest = self._canonical_generations.get(key)
            return latest is not None and latest[0] == expected.screen_order

    async def claim(
        self,
        callback_data: object,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
        access_version: int,
        expected_action: CaptureAction | None = None,
        now: datetime | None = None,
    ) -> NovaCompanionCaptureCapability | None:
        current = self._utc(now) if now is not None else None
        capability = await self.peek(
            callback_data,
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            canonical_message_id=canonical_message_id,
            access_version=access_version,
            expected_action=expected_action,
            now=current,
        )
        if capability is None or not await self.consume(capability, now=current):
            return None
        return capability

    async def revoke_screen(
        self,
        expected: NovaCompanionCaptureScreen,
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

    async def mark_suppressed(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        fingerprint: str,
        now: datetime | None = None,
    ) -> None:
        self._validate_binding(owner_id, telegram_user_id, chat_id, access_version)
        if not isinstance(fingerprint, str) or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError("invalid companion suggestion fingerprint")
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            self._mark_suppressed_locked(
                owner_id,
                telegram_user_id,
                chat_id,
                access_version,
                fingerprint,
                current,
            )

    async def is_suppressed(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        fingerprint: str,
        now: datetime | None = None,
    ) -> bool:
        self._validate_binding(owner_id, telegram_user_id, chat_id, access_version)
        if not isinstance(fingerprint, str) or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
            return False
        current = self._utc(now)
        async with self._lock:
            self._cleanup_locked(current)
            return (
                self._suppression_key(
                    owner_id,
                    telegram_user_id,
                    chat_id,
                    access_version,
                    fingerprint,
                )
                in self._suppressions
            )

    async def cleanup(self, *, now: datetime | None = None) -> int:
        current = self._utc(now)
        async with self._lock:
            before = (
                len(self._capabilities) + len(self._suppressions) + len(self._canonical_generations)
            )
            self._cleanup_locked(current)
            return (
                before
                - len(self._capabilities)
                - len(self._suppressions)
                - len(self._canonical_generations)
            )

    def _mark_suppressed_locked(
        self,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        fingerprint: str,
        now: datetime,
    ) -> None:
        duplicate_screens = tuple(
            screen.screen_id
            for screen in self._screens.values()
            if screen.owner_id == owner_id
            and screen.telegram_user_id == telegram_user_id
            and screen.chat_id == chat_id
            and screen.access_version == access_version
            and screen.suggestion.fingerprint == fingerprint
        )
        for screen_id in duplicate_screens:
            self._drop_screen_locked(screen_id)
        key = self._suppression_key(
            owner_id,
            telegram_user_id,
            chat_id,
            access_version,
            fingerprint,
        )
        self._suppressions[key] = now + self.ttl
        while len(self._suppressions) > self.max_capabilities:
            oldest = min(self._suppressions, key=self._suppressions.__getitem__)
            self._suppressions.pop(oldest, None)

    def _cleanup_locked(self, now: datetime) -> None:
        for screen_id, screen in tuple(self._screens.items()):
            if screen.expires_at <= now:
                self._drop_screen_locked(screen_id)
        for token, capability in tuple(self._capabilities.items()):
            if capability.expires_at <= now:
                self._capabilities.pop(token, None)
        for key, expires_at in tuple(self._suppressions.items()):
            if expires_at <= now:
                self._suppressions.pop(key, None)
        for key, (_screen_order, expires_at) in tuple(self._canonical_generations.items()):
            if expires_at <= now:
                self._canonical_generations.pop(key, None)

    def _make_room_locked(self, count: int) -> None:
        while len(self._capabilities) + count > self.max_capabilities:
            if not self._screens:
                raise RuntimeError("companion capability store is inconsistent")
            oldest = min(
                self._screens.values(),
                key=lambda screen: (screen.expires_at, screen.screen_order),
            )
            self._drop_screen_locked(oldest.screen_id)

    def _drop_screen_locked(self, screen_id: str) -> None:
        for token, capability in tuple(self._capabilities.items()):
            if capability.screen_id == screen_id:
                self._capabilities.pop(token, None)
        self._screens.pop(screen_id, None)

    def _publish_screen_locked(
        self,
        suggestion: CaptureSuggestion,
        *,
        raw_text: str,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int | None,
        access_version: int,
        temporal: NovaCompanionCaptureTemporal | None,
        actions: tuple[CaptureAction, ...],
        now: datetime,
    ) -> NovaCompanionCaptureScreen:
        screen_order = self._next_screen_order + 1
        screen_id = secrets.token_urlsafe(12)
        while screen_id in self._screens:
            screen_id = secrets.token_urlsafe(12)
        expires_at = now + self.ttl
        capabilities: list[NovaCompanionCaptureCapability] = []
        callbacks: list[tuple[CaptureAction, str]] = []
        staged_tokens: set[str] = set()
        for action in actions:
            token = self._random_token_locked()
            while token in staged_tokens:
                token = self._random_token_locked()
            staged_tokens.add(token)
            capability = NovaCompanionCaptureCapability(
                token=token,
                screen_id=screen_id,
                action=action,
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                canonical_message_id=canonical_message_id,
                access_version=access_version,
                suggestion=suggestion,
                raw_text=raw_text,
                screen_order=screen_order,
                expires_at=expires_at,
                temporal=temporal,
            )
            capabilities.append(capability)
            callbacks.append((action, capability.callback_data))
        screen = NovaCompanionCaptureScreen(
            screen_id=screen_id,
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            canonical_message_id=canonical_message_id,
            access_version=access_version,
            suggestion=suggestion,
            raw_text=raw_text,
            screen_order=screen_order,
            expires_at=expires_at,
            _callbacks=tuple(callbacks),
            temporal=temporal,
        )
        # Build the complete random batch before evicting a published screen.
        # An entropy-source failure cannot destroy live controls.
        self._make_room_locked(len(actions))
        self._next_screen_order = screen_order
        self._screens[screen_id] = screen
        for capability in capabilities:
            self._capabilities[capability.token] = capability
        if canonical_message_id is not None:
            self._record_canonical_generation_locked(screen)
            obsolete = tuple(
                candidate.screen_id
                for candidate in self._screens.values()
                if candidate.screen_id != screen.screen_id
                and candidate.canonical_message_id == canonical_message_id
                and candidate.owner_id == owner_id
                and candidate.telegram_user_id == telegram_user_id
                and candidate.chat_id == chat_id
                and candidate.screen_order < screen.screen_order
            )
            for obsolete_screen_id in obsolete:
                self._drop_screen_locked(obsolete_screen_id)
        return screen

    def _record_canonical_generation_locked(self, screen: NovaCompanionCaptureScreen) -> None:
        if screen.canonical_message_id is None:
            raise RuntimeError("bound companion screen is required")
        key = self._canonical_key(
            screen.owner_id,
            screen.telegram_user_id,
            screen.chat_id,
            screen.canonical_message_id,
        )
        latest = self._canonical_generations.get(key)
        if latest is None or latest[0] <= screen.screen_order:
            self._canonical_generations[key] = (screen.screen_order, screen.expires_at)
        while len(self._canonical_generations) > self.max_capabilities:
            live_canonicals = {
                self._canonical_key(
                    candidate.owner_id,
                    candidate.telegram_user_id,
                    candidate.chat_id,
                    candidate.canonical_message_id,
                )
                for candidate in self._screens.values()
                if candidate.canonical_message_id is not None
            }
            oldest = min(
                (
                    candidate
                    for candidate in self._canonical_generations
                    if candidate not in live_canonicals
                ),
                key=lambda candidate: (
                    self._canonical_generations[candidate][1],
                    self._canonical_generations[candidate][0],
                ),
            )
            self._canonical_generations.pop(oldest, None)

    def _random_token_locked(self) -> str:
        token = secrets.token_urlsafe(18)
        while token in self._capabilities:
            token = secrets.token_urlsafe(18)
        return token

    @staticmethod
    def _callback_token(callback_data: object) -> str | None:
        if not isinstance(callback_data, str) or len(callback_data.encode("utf-8")) > 64:
            return None
        if not callback_data.startswith(NOVA_COMPANION_CALLBACK_PREFIX):
            return None
        token = callback_data.removeprefix(NOVA_COMPANION_CALLBACK_PREFIX)
        return token if _TOKEN_PATTERN.fullmatch(token) is not None else None

    @classmethod
    def _validate_binding(
        cls,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
    ) -> None:
        cls._validate_identity(owner_id, telegram_user_id, chat_id)
        cls._positive(access_version, "access version")

    def _validate_actions(
        self,
        actions: tuple[CaptureAction, ...],
    ) -> tuple[CaptureAction, ...]:
        if type(actions) is not tuple or actions not in {
            CAPTURE_ACTIONS,
            CAPTURE_DATE_ACTIONS,
        }:
            raise ValueError("invalid companion capture action set")
        if len(actions) > self.max_capabilities:
            raise ValueError("companion capture action set exceeds store capacity")
        return actions

    @staticmethod
    def _validate_temporal(temporal: NovaCompanionCaptureTemporal | None) -> None:
        if temporal is None:
            return
        if type(temporal) is not NovaCompanionCaptureTemporal:
            raise ValueError("validated companion capture temporal payload is required")
        try:
            rebuilt = NovaCompanionCaptureTemporal(
                timezone=temporal.timezone,
                resolution=temporal.resolution,
                local_time=temporal.local_time,
            )
        except (AttributeError, TypeError, ValueError):
            raise ValueError("validated companion capture temporal payload is required") from None
        if rebuilt != temporal:
            raise ValueError("validated companion capture temporal payload is required")

    @staticmethod
    def _validate_temporal_actions(
        temporal: NovaCompanionCaptureTemporal | None,
        actions: tuple[CaptureAction, ...],
    ) -> None:
        if actions == CAPTURE_DATE_ACTIONS and (
            temporal is None or temporal.resolution.status != "conflict"
        ):
            raise ValueError("date-choice actions require a conflicting date resolution")

    @staticmethod
    def _is_action(value: object) -> bool:
        return isinstance(value, str) and _ACTION_PATTERN.fullmatch(value) is not None

    @classmethod
    def _validate_identity(cls, owner_id: int, telegram_user_id: int, chat_id: int) -> None:
        cls._positive(owner_id, "owner id")
        cls._positive(telegram_user_id, "Telegram user id")
        cls._positive(chat_id, "chat id")

    @staticmethod
    def _positive(value: object, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be positive")

    @staticmethod
    def _suppression_key(
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        fingerprint: str,
    ) -> tuple[int, int, int, int, str]:
        return owner_id, telegram_user_id, chat_id, access_version, fingerprint

    @staticmethod
    def _canonical_key(
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
    ) -> tuple[int, int, int, int]:
        return owner_id, telegram_user_id, chat_id, canonical_message_id

    @staticmethod
    def _utc(value: datetime | None) -> datetime:
        current = value or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("companion capture clock must be timezone-aware")
        return current.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class NovaCompanionReminderCandidate:
    """Validated server-side reminder proposal; never serialized into callback data."""

    title: str = field(repr=False)
    schedule_wording: str | None = field(default=None, repr=False)
    evidence: str = field(default="", repr=False)
    timezone: str = field(default="UTC", repr=False)
    temporal: NovaCompanionCaptureTemporal | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        title = _safe_text(self.title, max_chars=200)
        schedule = (
            _safe_text(self.schedule_wording, max_chars=160)
            if self.schedule_wording is not None
            else None
        )
        evidence = _safe_text(self.evidence, max_chars=600)
        timezone = _safe_text(self.timezone, max_chars=128)
        if title is None or evidence is None or timezone is None:
            raise ValueError("invalid grounded reminder candidate")
        try:
            timezone = ZoneInfo(timezone).key
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("invalid grounded reminder timezone") from None
        if _grounding_text(title) not in _grounding_text(evidence):
            raise ValueError("reminder title is not grounded")
        if schedule is not None and _grounding_text(schedule) not in _grounding_text(evidence):
            raise ValueError("reminder schedule is not grounded")
        if self.temporal is not None:
            NovaCompanionCaptureStore._validate_temporal(self.temporal)
            if self.temporal.timezone != timezone:
                raise ValueError("reminder temporal timezone changed")
            if self.temporal.resolution.status != "resolved":
                raise ValueError("ambiguous reminder candidates are not actionable")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "schedule_wording", schedule)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "timezone", timezone)


@dataclass(frozen=True, slots=True)
class NovaCompanionReminderCapability:
    token: str = field(repr=False)
    screen_id: str = field(repr=False)
    action: ReminderOfferAction
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_tier: str = field(repr=False)
    access_version: int = field(repr=False)
    canonical_message_id: int | None = field(repr=False)
    candidate: NovaCompanionReminderCandidate = field(repr=False)
    context_fence: object = field(repr=False)
    memory_revision: str | None = field(default=None, repr=False)
    exchange_receipt: object | None = field(default=None, repr=False)
    screen_order: int = 0
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC), repr=False)

    @property
    def callback_data(self) -> str:
        return f"{NOVA_COMPANION_REMINDER_CALLBACK_PREFIX}{self.token}"


@dataclass(frozen=True, slots=True)
class NovaCompanionReminderScreen:
    screen_id: str = field(repr=False)
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_tier: str = field(repr=False)
    access_version: int = field(repr=False)
    canonical_message_id: int | None = field(repr=False)
    candidate: NovaCompanionReminderCandidate = field(repr=False)
    context_fence: object = field(repr=False)
    memory_revision: str | None = field(default=None, repr=False)
    exchange_receipt: object | None = field(default=None, repr=False)
    screen_order: int = 0
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC), repr=False)
    _callbacks: tuple[tuple[ReminderOfferAction, str], ...] = field(default=(), repr=False)

    @property
    def is_bound(self) -> bool:
        return self.canonical_message_id is not None

    @property
    def callbacks(self) -> MappingProxyType[ReminderOfferAction, str]:
        return MappingProxyType(dict(self._callbacks))

    def callback_data(self, action: ReminderOfferAction) -> str:
        for candidate, callback in self._callbacks:
            if candidate == action:
                return callback
        raise KeyError(action)


class NovaCompanionReminderStore:
    """Bounded opaque two-action reminder offers with exact canonical generations."""

    _ACTIONS: tuple[ReminderOfferAction, ...] = ("accept", "not_now")

    def __init__(
        self,
        *,
        ttl: timedelta = NOVA_COMPANION_CAPABILITY_TTL,
        max_capabilities: int = NOVA_COMPANION_MAX_CAPABILITIES,
    ) -> None:
        if ttl < timedelta(seconds=1) or ttl > timedelta(hours=2):
            raise ValueError("companion reminder ttl must be between 1 second and 2 hours")
        if not 2 <= max_capabilities <= 20_000:
            raise ValueError("companion reminder capability limit must be between 2 and 20000")
        self.ttl = ttl
        self.max_capabilities = max_capabilities
        self._capabilities: dict[str, NovaCompanionReminderCapability] = {}
        self._screens: dict[str, NovaCompanionReminderScreen] = {}
        self._canonical_generations: dict[tuple[int, int, int, int], tuple[int, datetime]] = {}
        self._next_screen_order = 0
        self._lock = asyncio.Lock()

    async def stage(
        self,
        candidate: NovaCompanionReminderCandidate,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_tier: str,
        access_version: int,
        context_fence: object,
        memory_revision: str | None,
        now: datetime | None = None,
    ) -> NovaCompanionReminderScreen | None:
        NovaCompanionCaptureStore._validate_binding(
            owner_id, telegram_user_id, chat_id, access_version
        )
        if not isinstance(candidate, NovaCompanionReminderCandidate):
            raise ValueError("validated reminder candidate is required")
        if not isinstance(access_tier, str) or not access_tier or context_fence is None:
            raise ValueError("exact reminder generation is required")
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            return self._publish_locked(
                candidate,
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_tier=access_tier,
                access_version=access_version,
                canonical_message_id=None,
                context_fence=context_fence,
                memory_revision=memory_revision,
                exchange_receipt=None,
                now=current,
            )

    async def bind(
        self,
        expected: NovaCompanionReminderScreen,
        *,
        canonical_message_id: int,
        now: datetime | None = None,
    ) -> NovaCompanionReminderScreen | None:
        NovaCompanionCaptureStore._positive(canonical_message_id, "canonical message id")
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            stored = self._screens.get(expected.screen_id)
            if stored != expected or stored.is_bound or stored.expires_at <= current:
                return None
            bound = replace(stored, canonical_message_id=canonical_message_id)
            self._screens[bound.screen_id] = bound
            for token, capability in tuple(self._capabilities.items()):
                if capability.screen_id == bound.screen_id:
                    self._capabilities[token] = replace(
                        capability, canonical_message_id=canonical_message_id
                    )
            self._record_generation_locked(bound)
            obsolete = tuple(
                screen.screen_id
                for screen in self._screens.values()
                if screen.screen_id != bound.screen_id
                and screen.owner_id == bound.owner_id
                and screen.telegram_user_id == bound.telegram_user_id
                and screen.chat_id == bound.chat_id
                and screen.screen_order < bound.screen_order
            )
            for screen_id in obsolete:
                self._drop_screen_locked(screen_id)
            return bound

    async def attach_exchange(
        self,
        expected: NovaCompanionReminderScreen,
        exchange_receipt: object,
        *,
        now: datetime | None = None,
    ) -> NovaCompanionReminderScreen | None:
        if exchange_receipt is None:
            return None
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            stored = self._screens.get(expected.screen_id)
            if stored != expected or stored.expires_at <= current:
                return None
            updated = replace(stored, exchange_receipt=exchange_receipt)
            self._screens[updated.screen_id] = updated
            for token, capability in tuple(self._capabilities.items()):
                if capability.screen_id == updated.screen_id:
                    self._capabilities[token] = replace(
                        capability, exchange_receipt=exchange_receipt
                    )
            return updated

    async def advance_context(
        self,
        expected: NovaCompanionReminderCapability,
        *,
        context_fence: object,
        memory_revision: str | None,
        exchange_receipt: object,
        now: datetime | None = None,
    ) -> NovaCompanionReminderCapability | None:
        """Advance one still-live action anchor after an ordinary companion turn."""

        if context_fence is None or exchange_receipt is None:
            return None
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            stored_capability = self._capabilities.get(expected.token)
            screen = self._screens.get(expected.screen_id)
            if (
                stored_capability != expected
                or screen is None
                or stored_capability.expires_at <= current
            ):
                return None
            updated_screen = replace(
                screen,
                context_fence=context_fence,
                memory_revision=memory_revision,
                exchange_receipt=exchange_receipt,
            )
            self._screens[screen.screen_id] = updated_screen
            result = None
            for token, capability in tuple(self._capabilities.items()):
                if capability.screen_id == screen.screen_id:
                    updated = replace(
                        capability,
                        context_fence=context_fence,
                        memory_revision=memory_revision,
                        exchange_receipt=exchange_receipt,
                    )
                    self._capabilities[token] = updated
                    if token == expected.token:
                        result = updated
            return result

    async def active(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_tier: str,
        access_version: int,
        action: ReminderOfferAction = "accept",
        now: datetime | None = None,
    ) -> NovaCompanionReminderCapability | None:
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            candidates = [
                capability
                for capability in self._capabilities.values()
                if capability.action == action
                and capability.canonical_message_id is not None
                and capability.owner_id == owner_id
                and capability.telegram_user_id == telegram_user_id
                and capability.chat_id == chat_id
                and capability.access_tier == access_tier
                and capability.access_version == access_version
            ]
            return max(candidates, key=lambda item: item.screen_order, default=None)

    async def peek_bound_identity(
        self,
        callback_data: object,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int,
        now: datetime | None = None,
    ) -> NovaCompanionReminderCapability | None:
        token = self._callback_token(callback_data)
        if token is None:
            return None
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            capability = self._capabilities.get(token)
            if capability is None or capability.canonical_message_id is None:
                return None
            if (
                capability.owner_id != owner_id
                or capability.telegram_user_id != telegram_user_id
                or capability.chat_id != chat_id
                or capability.canonical_message_id != canonical_message_id
            ):
                return None
            return capability

    async def consume(
        self,
        expected: NovaCompanionReminderCapability,
        *,
        now: datetime | None = None,
    ) -> bool:
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            stored = self._capabilities.get(expected.token)
            if (
                stored != expected
                or stored is None
                or stored.canonical_message_id is None
                or stored.expires_at <= current
            ):
                return False
            self._drop_screen_locked(stored.screen_id)
            return True

    async def consumed_screen_is_current(
        self,
        expected: NovaCompanionReminderCapability,
        *,
        now: datetime | None = None,
    ) -> bool:
        if expected.canonical_message_id is None:
            return False
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            if expected.expires_at <= current:
                return False
            latest = self._canonical_generations.get(self._canonical_key(expected))
            return latest is not None and latest[0] == expected.screen_order

    async def revoke_screen(
        self,
        expected: NovaCompanionReminderScreen,
        *,
        now: datetime | None = None,
    ) -> bool:
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            self._cleanup_locked(current)
            if self._screens.get(expected.screen_id) != expected:
                return False
            self._drop_screen_locked(expected.screen_id)
            return True

    async def cleanup(self, *, now: datetime | None = None) -> int:
        injected_now = self._injected_now(now)
        async with self._lock:
            current = self._current_locked(injected_now)
            before = len(self._capabilities) + len(self._canonical_generations)
            self._cleanup_locked(current)
            return before - len(self._capabilities) - len(self._canonical_generations)

    def _publish_locked(
        self,
        candidate: NovaCompanionReminderCandidate,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_tier: str,
        access_version: int,
        canonical_message_id: int | None,
        context_fence: object,
        memory_revision: str | None,
        exchange_receipt: object | None,
        now: datetime,
    ) -> NovaCompanionReminderScreen:
        screen_id = secrets.token_urlsafe(18)
        while screen_id in self._screens:
            screen_id = secrets.token_urlsafe(18)
        tokens: list[tuple[ReminderOfferAction, str]] = []
        for action in self._ACTIONS:
            token = secrets.token_urlsafe(18)
            while token in self._capabilities or any(value == token for _, value in tokens):
                token = secrets.token_urlsafe(18)
            tokens.append((action, token))
        order = self._next_screen_order + 1
        expires_at = now + self.ttl
        callbacks = tuple(
            (action, f"{NOVA_COMPANION_REMINDER_CALLBACK_PREFIX}{token}")
            for action, token in tokens
        )
        screen = NovaCompanionReminderScreen(
            screen_id=screen_id,
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            access_tier=access_tier,
            access_version=access_version,
            canonical_message_id=canonical_message_id,
            candidate=candidate,
            context_fence=context_fence,
            memory_revision=memory_revision,
            exchange_receipt=exchange_receipt,
            screen_order=order,
            expires_at=expires_at,
            _callbacks=callbacks,
        )
        capabilities = [
            NovaCompanionReminderCapability(
                token=token,
                screen_id=screen_id,
                action=action,
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_tier=access_tier,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                candidate=candidate,
                context_fence=context_fence,
                memory_revision=memory_revision,
                exchange_receipt=exchange_receipt,
                screen_order=order,
                expires_at=expires_at,
            )
            for action, token in tokens
        ]
        if len(self._capabilities) + len(capabilities) > self.max_capabilities:
            self._make_room_locked(len(capabilities))
        self._next_screen_order = order
        self._screens[screen_id] = screen
        for capability in capabilities:
            self._capabilities[capability.token] = capability
        return screen

    def _record_generation_locked(self, screen: NovaCompanionReminderScreen) -> None:
        key = self._canonical_key(screen)
        latest = self._canonical_generations.get(key)
        if latest is None or latest[0] <= screen.screen_order:
            self._canonical_generations[key] = (screen.screen_order, screen.expires_at)
        self._trim_generations_locked()

    def _trim_generations_locked(self) -> None:
        while len(self._canonical_generations) > self.max_capabilities:
            live = {
                self._canonical_key(screen)
                for screen in self._screens.values()
                if screen.canonical_message_id is not None
            }
            candidate = min(
                (key for key in self._canonical_generations if key not in live),
                key=lambda key: (
                    self._canonical_generations[key][1],
                    self._canonical_generations[key][0],
                ),
            )
            self._canonical_generations.pop(candidate, None)

    def _make_room_locked(self, needed: int) -> None:
        while len(self._capabilities) + needed > self.max_capabilities and self._screens:
            oldest = min(
                self._screens.values(), key=lambda screen: (screen.expires_at, screen.screen_order)
            )
            self._drop_screen_locked(oldest.screen_id)

    def _drop_screen_locked(self, screen_id: str) -> None:
        self._screens.pop(screen_id, None)
        for token, capability in tuple(self._capabilities.items()):
            if capability.screen_id == screen_id:
                self._capabilities.pop(token, None)

    def _cleanup_locked(self, now: datetime) -> None:
        for screen in tuple(self._screens.values()):
            if screen.expires_at <= now:
                self._drop_screen_locked(screen.screen_id)
        for key, (_order, expires_at) in tuple(self._canonical_generations.items()):
            if expires_at <= now:
                self._canonical_generations.pop(key, None)

    @staticmethod
    def _injected_now(now: datetime | None) -> datetime | None:
        """Normalize deterministic clocks before awaiting the store lock."""

        return NovaCompanionCaptureStore._utc(now) if now is not None else None

    @staticmethod
    def _current_locked(injected_now: datetime | None) -> datetime:
        """Sample the runtime clock only after the store lock is acquired."""

        return injected_now if injected_now is not None else NovaCompanionCaptureStore._utc(None)

    @staticmethod
    def _canonical_key(
        value: NovaCompanionReminderScreen | NovaCompanionReminderCapability,
    ) -> tuple[int, int, int, int]:
        if value.canonical_message_id is None:
            raise ValueError("bound reminder screen is required")
        return value.owner_id, value.telegram_user_id, value.chat_id, value.canonical_message_id

    @staticmethod
    def _callback_token(callback_data: object) -> str | None:
        if not isinstance(callback_data, str) or len(callback_data.encode("utf-8")) > 64:
            return None
        if not callback_data.startswith(NOVA_COMPANION_REMINDER_CALLBACK_PREFIX):
            return None
        token = callback_data.removeprefix(NOVA_COMPANION_REMINDER_CALLBACK_PREFIX)
        return token if _TOKEN_PATTERN.fullmatch(token) else None


def _capture_kind(value: str) -> CaptureKind | None:
    normalized = value.casefold()
    mapping: dict[str, CaptureKind] = {
        "идею": "idea",
        "идея": "idea",
        "задачу": "task",
        "задача": "task",
        "желание": "desire",
        "желания": "desire",
        "заметку": "note",
        "заметка": "note",
        "запись": "note",
    }
    return mapping.get(normalized)


def _safe_text(value: object, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value)
    cleaned = " ".join(normalized.split())
    if not cleaned or len(cleaned) > max_chars or _CONTROL_PATTERN.search(cleaned):
        return None
    return cleaned


def _grounding_text(value: object) -> str:
    cleaned = _safe_text(value, max_chars=4_000)
    return cleaned.casefold() if cleaned is not None else ""


__all__ = [
    "CAPTURE_ACTIONS",
    "CAPTURE_DATE_ACTIONS",
    "CAPTURE_KINDS",
    "NOVA_COMPANION_CALLBACK_PREFIX",
    "NOVA_COMPANION_REMINDER_CALLBACK_PREFIX",
    "CaptureAction",
    "CaptureKind",
    "CaptureSuggestion",
    "ExplicitCaptureClassifier",
    "ExplicitCaptureIntent",
    "NovaAddressClassifier",
    "NovaAddressKind",
    "NovaAddressResult",
    "NovaCompanionCaptureCapability",
    "NovaCompanionCaptureScreen",
    "NovaCompanionCaptureStore",
    "NovaCompanionCaptureTemporal",
    "NovaCompanionReminderCandidate",
    "NovaCompanionReminderCapability",
    "NovaCompanionReminderScreen",
    "NovaCompanionReminderStore",
    "NovaCompanionPolicy",
    "should_offer_capture",
    "validate_capture_suggestion",
]
