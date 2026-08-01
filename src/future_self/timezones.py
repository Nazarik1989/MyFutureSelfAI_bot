from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .ai import AIService
from .domain import canonical_timezone
from .location import parse_location

MAX_TIMEZONE_INPUT_CHARS = 200


@dataclass(frozen=True, slots=True)
class TimezoneCandidate:
    timezone: str
    city: str | None
    source: str


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
        "бишкеке": _CITY_ALIASES["бишкек"],
        "владивостоке": _CITY_ALIASES["владивосток"],
        "дубае": _CITY_ALIASES["дубай"],
        "екатеринбурге": _CITY_ALIASES["екатеринбург"],
        "казани": _CITY_ALIASES["казань"],
        "калининграде": _CITY_ALIASES["калининград"],
        "красноярске": _CITY_ALIASES["красноярск"],
        "лондоне": _CITY_ALIASES["лондон"],
        "москве": _CITY_ALIASES["москва"],
        "новосибирске": _CITY_ALIASES["новосибирск"],
        "омске": _CITY_ALIASES["омск"],
        "париже": _CITY_ALIASES["париж"],
        "самаре": _CITY_ALIASES["самара"],
        "санкт петербурге": _CITY_ALIASES["санкт петербург"],
        "петербурге": _CITY_ALIASES["петербург"],
        "саратове": _CITY_ALIASES["саратов"],
        "сеуле": _CITY_ALIASES["сеул"],
        "стамбуле": _CITY_ALIASES["стамбул"],
        "ташкенте": _CITY_ALIASES["ташкент"],
        "токио": _CITY_ALIASES["токио"],
        "иркутске": _CITY_ALIASES["иркутск"],
        "якутске": _CITY_ALIASES["якутск"],
    }
)

_INVALID_OFFSET = re.compile(r"^\s*(?:gmt|utc)\s*[+-]", re.IGNORECASE)
_NON_WORD = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)


def _normalized_words(value: str) -> str:
    return " ".join(_NON_WORD.sub(" ", value.casefold().replace("ё", "е")).split())


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
