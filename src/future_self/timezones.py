from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .ai import AIService
from .domain import canonical_timezone
from .location import parse_location

MAX_TIMEZONE_INPUT_CHARS = 200
MAX_REMINDER_TIMEZONE_FRAGMENT_CHARS = 120


@dataclass(frozen=True, slots=True)
class TimezoneCandidate:
    timezone: str
    city: str | None
    source: str


class ReminderTimezoneStatus(StrEnum):
    RESOLVED = "resolved"
    NOT_MENTIONED = "not_mentioned"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class ReminderTimezoneFragment:
    text: str
    location_text: str
    span: tuple[int, int]
    is_strong: bool = False


@dataclass(frozen=True, slots=True)
class ReminderTimezoneOutcome:
    status: ReminderTimezoneStatus
    candidate: TimezoneCandidate | None = None
    evidence_text: str | None = None
    evidence_span: tuple[int, int] | None = None


_CITY_ALIASES: dict[str, tuple[str, str]] = {
    "алматы": ("Asia/Almaty", "Алматы"),
    "астана": ("Asia/Almaty", "Астана"),
    "баку": ("Asia/Baku", "Баку"),
    "бали": ("Asia/Makassar", "Бали"),
    "берлин": ("Europe/Berlin", "Берлин"),
    "бишкек": ("Asia/Bishkek", "Бишкек"),
    "владивосток": ("Asia/Vladivostok", "Владивосток"),
    "дубай": ("Asia/Dubai", "Дубай"),
    "екатеринбург": ("Asia/Yekaterinburg", "Екатеринбург"),
    "казань": ("Europe/Moscow", "Казань"),
    "калининград": ("Europe/Kaliningrad", "Калининград"),
    "красноярск": ("Asia/Krasnoyarsk", "Красноярск"),
    "лондон": ("Europe/London", "Лондон"),
    "москва": ("Europe/Moscow", "Москва"),
    "новосибирск": ("Asia/Novosibirsk", "Новосибирск"),
    "омск": ("Asia/Omsk", "Омск"),
    "париж": ("Europe/Paris", "Париж"),
    "самара": ("Europe/Samara", "Самара"),
    "санкт петербург": ("Europe/Moscow", "Санкт-Петербург"),
    "петербург": ("Europe/Moscow", "Санкт-Петербург"),
    "саратов": ("Europe/Saratov", "Саратов"),
    "сеул": ("Asia/Seoul", "Сеул"),
    "стамбул": ("Europe/Istanbul", "Стамбул"),
    "ташкент": ("Asia/Tashkent", "Ташкент"),
    "тбилиси": ("Asia/Tbilisi", "Тбилиси"),
    "токио": ("Asia/Tokyo", "Токио"),
    "иркутск": ("Asia/Irkutsk", "Иркутск"),
    "якутск": ("Asia/Yakutsk", "Якутск"),
    "magadan": ("Asia/Magadan", "Магадан"),
    "moscow": ("Europe/Moscow", "Москва"),
    "new york": ("America/New_York", "New York"),
    "los angeles": ("America/Los_Angeles", "Los Angeles"),
    "berlin": ("Europe/Berlin", "Berlin"),
    "london": ("Europe/London", "London"),
}

# Frequent Russian prepositional forms let ordinary phrases such as
# "я сейчас в Казани" stay deterministic and avoid an unnecessary model call.
_CITY_ALIASES.update(
    {
        "алмате": _CITY_ALIASES["алматы"],
        "астане": _CITY_ALIASES["астана"],
        "баку": _CITY_ALIASES["баку"],
        "берлине": _CITY_ALIASES["берлин"],
        "берлина": _CITY_ALIASES["берлин"],
        "берлину": _CITY_ALIASES["берлин"],
        "бишкеке": _CITY_ALIASES["бишкек"],
        "владивостоке": _CITY_ALIASES["владивосток"],
        "дубае": _CITY_ALIASES["дубай"],
        "екатеринбурге": _CITY_ALIASES["екатеринбург"],
        "казани": _CITY_ALIASES["казань"],
        "калининграде": _CITY_ALIASES["калининград"],
        "красноярске": _CITY_ALIASES["красноярск"],
        "лондоне": _CITY_ALIASES["лондон"],
        "лондона": _CITY_ALIASES["лондон"],
        "лондону": _CITY_ALIASES["лондон"],
        "москве": _CITY_ALIASES["москва"],
        "новосибирске": _CITY_ALIASES["новосибирск"],
        "омске": _CITY_ALIASES["омск"],
        "париже": _CITY_ALIASES["париж"],
        "самаре": _CITY_ALIASES["самара"],
        "санкт петербурге": _CITY_ALIASES["санкт петербург"],
        "санкт петербурга": _CITY_ALIASES["санкт петербург"],
        "санкт петербургу": _CITY_ALIASES["санкт петербург"],
        "петербурге": _CITY_ALIASES["петербург"],
        "петербурга": _CITY_ALIASES["петербург"],
        "петербургу": _CITY_ALIASES["петербург"],
        "саратове": _CITY_ALIASES["саратов"],
        "сеуле": _CITY_ALIASES["сеул"],
        "стамбуле": _CITY_ALIASES["стамбул"],
        "ташкенте": _CITY_ALIASES["ташкент"],
        "токио": _CITY_ALIASES["токио"],
        "иркутске": _CITY_ALIASES["иркутск"],
        "якутске": _CITY_ALIASES["якутск"],
        "нью йорк": ("America/New_York", "Нью-Йорк"),
        "нью йорка": ("America/New_York", "Нью-Йорк"),
        "нью йорку": ("America/New_York", "Нью-Йорк"),
        "лос анджелес": ("America/Los_Angeles", "Лос-Анджелес"),
        "лос анджелеса": ("America/Los_Angeles", "Лос-Анджелес"),
        "лос анджелесе": ("America/Los_Angeles", "Лос-Анджелес"),
        "лос анджелесу": ("America/Los_Angeles", "Лос-Анджелес"),
    }
)

_INVALID_OFFSET = re.compile(r"^\s*(?:gmt|utc)\s*[+-]", re.IGNORECASE)
_NON_WORD = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
_REMINDER_CLOCK = (
    r"(?:[01]?\d|2[0-3])\s*[:.]\s*[0-5]\d|"
    r"(?:[01]?\d|2[0-3])\s+[0-5]\d|"
    r"(?:[01]?\d|2[0-3])\s+час(?:а|ов)?(?:\s+[0-5]?\d\s+минут(?:у|ы)?)?"
)
_STRONG_REMINDER_TIMEZONE_MARKER = re.compile(
    r"\b(?:по\s+времени|в\s+часовом\s+поясе)\s+",
    re.IGNORECASE,
)
_WEAK_REMINDER_TIMEZONE_MARKER = re.compile(
    rf"(?:{_REMINDER_CLOCK})(?P<marker>\s*по\s+)",
    re.IGNORECASE,
)
_ANCHORED_REMINDER_TIMEZONE_MARKER = re.compile(
    r"^(?:по\s+времени|в\s+часовом\s+поясе|по)\s+",
    re.IGNORECASE,
)
_CANDIDATE_WORD = re.compile(r"\S+")
_DIRECT_TIMEZONE_PREFIX = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9._+-]*/[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*|"
    r"мск|msk|moscow|москва|москве)\b",
    re.IGNORECASE,
)
MAX_REMINDER_TIMEZONE_CANDIDATE_WORDS = 8
_SEMANTIC_PO_OBJECTS = {
    "графику",
    "задаче",
    "инструкции",
    "плану",
    "проекту",
    "работе",
}
_TASK_ACTION_BOUNDARY = re.compile(
    r"^(?:позвонить|созвониться|написать|писать|заполнить|проверить|отправить|"
    r"купить|сделать|подготовить|оплатить|забрать|договориться|call|check|fill|send|write)\b",
    re.IGNORECASE,
)


def _normalized_words(value: str) -> str:
    return " ".join(_NON_WORD.sub(" ", value.casefold().replace("ё", "е")).split())


def _clean_fragment(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip())


def _normalized_evidence(value: str) -> str:
    return _clean_fragment(value).casefold().replace("ё", "е")


def _candidate_window(clean: str, start: int) -> tuple[str, tuple[int, int]]:
    tail = clean[start : start + MAX_REMINDER_TIMEZONE_FRAGMENT_CHARS]
    words = tuple(_CANDIDATE_WORD.finditer(tail))
    if len(words) > MAX_REMINDER_TIMEZONE_CANDIDATE_WORDS:
        tail = tail[: words[MAX_REMINDER_TIMEZONE_CANDIDATE_WORDS - 1].end()]
    elif start + len(tail) < len(clean) and tail and not tail[-1].isspace():
        boundary = tail.rfind(" ")
        if boundary > 0:
            tail = tail[:boundary]
    window = tail.rstrip()
    if not window:
        raise ValueError("reminder timezone fragment is empty")
    return window, (start, start + len(window))


def _timezone_marker_starts(clean: str) -> tuple[tuple[int, bool], ...]:
    strong_starts = {match.start() for match in _STRONG_REMINDER_TIMEZONE_MARKER.finditer(clean)}
    starts = set(strong_starts)
    for match in _WEAK_REMINDER_TIMEZONE_MARKER.finditer(clean):
        marker = match.group("marker")
        relative = marker.casefold().find("по")
        start = match.start("marker") + relative
        if start in starts:
            continue
        remainder = clean[match.end("marker") :]
        first_word_match = re.match(r"\S+", remainder)
        if first_word_match is None:
            continue
        first_word = _normalized_words(first_word_match.group())
        if first_word in _SEMANTIC_PO_OBJECTS:
            continue
        starts.add(start)
    return tuple((start, start in strong_starts) for start in sorted(starts))


def extract_reminder_timezone_fragment(value: str) -> ReminderTimezoneFragment | None:
    clean = _clean_fragment(value) if isinstance(value, str) else ""
    if not clean:
        return None
    starts = _timezone_marker_starts(clean)
    if not starts:
        return None
    if len(starts) != 1:
        raise ValueError("reminder contains multiple timezone markers")
    start, is_strong = starts[0]
    fragment, span = _candidate_window(clean, start)
    marker = _ANCHORED_REMINDER_TIMEZONE_MARKER.match(fragment)
    if marker is None or marker.end() >= len(fragment):
        raise ValueError("reminder timezone marker has no candidate location")
    return ReminderTimezoneFragment(fragment, fragment[marker.end() :], span, is_strong)


def reminder_timezone_reply_fragment(value: str) -> ReminderTimezoneFragment:
    clean = _clean_fragment(value) if isinstance(value, str) else ""
    if not clean or len(clean) > MAX_REMINDER_TIMEZONE_FRAGMENT_CHARS:
        raise ValueError("reminder timezone reply must contain at most 120 characters")
    return ReminderTimezoneFragment(clean, clean, (0, len(clean)), True)


def _alias_prefix_match(value: str) -> re.Match[str] | None:
    for alias in sorted(_CITY_ALIASES, key=len, reverse=True):
        words = alias.split()
        pattern = r"^" + r"[\s-]+".join(re.escape(word) for word in words) + r"\b"
        match = re.match(pattern, value.casefold().replace("ё", "е"))
        if match is not None:
            return match
    return None


def _local_reminder_evidence(
    fragment: ReminderTimezoneFragment,
) -> tuple[TimezoneCandidate, str, tuple[int, int]] | None:
    marker = _ANCHORED_REMINDER_TIMEZONE_MARKER.match(fragment.text)
    place_start = marker.end() if marker is not None else 0
    location = fragment.text[place_start:]
    direct = _DIRECT_TIMEZONE_PREFIX.match(location)
    alias = _alias_prefix_match(location)
    match = direct or alias
    if match is None:
        return None
    place_stop = match.end()
    remainder = location[place_stop:]
    qualifier = remainder.lstrip()
    if qualifier and _TASK_ACTION_BOUNDARY.match(qualifier) is None:
        return None
    exact_location = location[:place_stop]
    candidate = resolve_timezone_locally(exact_location)
    if candidate is None:
        return None
    evidence_stop = place_start + place_stop
    evidence = fragment.text[:evidence_stop]
    start = fragment.span[0]
    return candidate, evidence, (start, start + evidence_stop)


def _model_evidence_span(
    fragment: ReminderTimezoneFragment,
    matched_text: str | None,
) -> tuple[str, tuple[int, int]]:
    normalized_evidence = _normalized_evidence(matched_text or "")
    normalized_window = _normalized_evidence(fragment.text)
    if not normalized_evidence or not normalized_window.startswith(normalized_evidence):
        raise ValueError("model timezone evidence must start at the timezone marker")
    evidence_stop = next(
        (
            stop
            for stop in range(1, len(fragment.text) + 1)
            if _normalized_evidence(fragment.text[:stop]) == normalized_evidence
        ),
        None,
    )
    if evidence_stop is None:
        raise ValueError("model timezone evidence is not present in the candidate window")
    evidence = fragment.text[:evidence_stop]
    window_marker = _ANCHORED_REMINDER_TIMEZONE_MARKER.match(fragment.text)
    marker = _ANCHORED_REMINDER_TIMEZONE_MARKER.match(evidence)
    if window_marker is not None and marker is None:
        raise ValueError("model timezone evidence must contain a place and the full marker")
    place = evidence[marker.end() :] if marker is not None else evidence
    place_words = _normalized_words(place).split()
    if not place_words:
        raise ValueError("model timezone evidence must contain a place")
    if normalized_window.count(normalized_evidence) != 1:
        raise ValueError("model timezone evidence must be unique in the candidate window")
    start = fragment.span[0]
    return evidence, (start, start + evidence_stop)


def _safe_city(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        location = parse_location(value)
    except ValueError:
        return None
    return location.city if location.fallback_city is None else None


def resolve_timezone_locally(value: str) -> TimezoneCandidate | None:
    clean = " ".join(value.split())
    if not clean or len(clean) > MAX_TIMEZONE_INPUT_CHARS:
        raise ValueError("Напиши город или часовой пояс короче — до 200 символов.")
    try:
        timezone = canonical_timezone(clean)
    except ValueError:
        if _INVALID_OFFSET.match(clean):
            raise
    else:
        city = None
        tail = timezone.rsplit("/", maxsplit=1)[-1].replace("_", " ")
        if "/" in timezone and not timezone.startswith("Etc/") and not _INVALID_OFFSET.match(clean):
            city = _safe_city(tail)
        return TimezoneCandidate(timezone, city, "direct")

    words = f" {_normalized_words(clean)} "
    for alias in sorted(_CITY_ALIASES, key=len, reverse=True):
        if f" {alias} " in words:
            timezone, city = _CITY_ALIASES[alias]
            return TimezoneCandidate(timezone, city, "local")
    return None


class TimezoneResolver:
    def __init__(self, ai: AIService):
        self.ai = ai

    async def resolve(self, value: str) -> TimezoneCandidate:
        local = resolve_timezone_locally(value)
        if local is not None:
            return local
        resolution = await self.ai.resolve_timezone(" ".join(value.split()))
        if resolution.ambiguous or not resolution.timezone:
            raise ValueError(
                "Не получилось однозначно определить часовой пояс. "
                "Напиши город вместе со страной или регионом, например: «Кордова, Испания»."
            )
        try:
            timezone = ZoneInfo(resolution.timezone).key
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                "Модель не смогла подтвердить часовой пояс. Напиши ближайший крупный город."
            ) from exc
        return TimezoneCandidate(timezone, _safe_city(resolution.city), "model")

    def resolve_reminder_locally(
        self,
        fragment: ReminderTimezoneFragment,
    ) -> ReminderTimezoneOutcome | None:
        local = _local_reminder_evidence(fragment)
        if local is None:
            return None
        candidate, evidence, span = local
        return ReminderTimezoneOutcome(
            ReminderTimezoneStatus.RESOLVED,
            candidate,
            evidence,
            span,
        )

    async def resolve_reminder(
        self,
        fragment: ReminderTimezoneFragment,
    ) -> ReminderTimezoneOutcome:
        local = self.resolve_reminder_locally(fragment)
        if local is not None:
            return local
        resolution = await self.ai.resolve_reminder_timezone(fragment.text)
        status = ReminderTimezoneStatus(resolution.status)
        if status in {
            ReminderTimezoneStatus.NOT_MENTIONED,
            ReminderTimezoneStatus.INSUFFICIENT,
        }:
            return ReminderTimezoneOutcome(status)
        evidence, evidence_span = _model_evidence_span(fragment, resolution.matched_text)
        if status is ReminderTimezoneStatus.AMBIGUOUS:
            return ReminderTimezoneOutcome(
                status,
                evidence_text=evidence,
                evidence_span=evidence_span,
            )
        try:
            timezone = ZoneInfo(resolution.timezone or "").key
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("model returned an invalid IANA timezone") from exc
        candidate = TimezoneCandidate(timezone, _safe_city(resolution.city), "model")
        return ReminderTimezoneOutcome(
            ReminderTimezoneStatus.RESOLVED,
            candidate,
            evidence,
            evidence_span,
        )


def timezone_candidate_text(
    candidate: TimezoneCandidate,
    *,
    now: datetime | None = None,
) -> str:
    local_now = (now or datetime.now(UTC)).astimezone(ZoneInfo(candidate.timezone))
    offset = local_now.utcoffset()
    total_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    sign = "+" if total_minutes >= 0 else "−"
    absolute = abs(total_minutes)
    offset_text = f"UTC{sign}{absolute // 60}" + (f":{absolute % 60:02d}" if absolute % 60 else "")
    city = f"📍 Город: {candidate.city}\n" if candidate.city else ""
    return (
        "Проверь, правильно ли я определил время:\n\n"
        f"{city}🌍 Часовой пояс: {candidate.timezone}\n"
        f"🕒 Сейчас: {local_now.strftime('%H:%M')} ({offset_text})\n\n"
        "Верно?"
    )
