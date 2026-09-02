from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from .dates import MONTHS
from .domain import validate_timezone


class ReminderIntentStatus(StrEnum):
    NOT_REMINDER = "not_reminder"
    COMPLETE = "complete"
    NEEDS_WHEN = "needs_when"
    NEEDS_TIME = "needs_time"
    NEEDS_TITLE = "needs_title"
    INVALID = "invalid"


class ReminderScheduleKind(StrEnum):
    ONCE = "once"
    DAILY = "daily"


class ReminderTimezoneSource(StrEnum):
    PROFILE = "profile"
    EXPLICIT = "explicit"


class ReminderIntentCode(StrEnum):
    NO_EXPLICIT_INTENT = "no_explicit_reminder_intent"
    MISSING_WHEN = "missing_when"
    MISSING_TIME = "missing_time"
    MISSING_TITLE = "missing_title"
    INVALID_PROFILE_TIMEZONE = "invalid_profile_timezone"
    INVALID_EXPLICIT_TIMEZONE = "invalid_explicit_timezone"
    INVALID_DATE = "invalid_date"
    AMBIGUOUS_DATE = "ambiguous_date"
    INVALID_TIME = "invalid_time"
    AMBIGUOUS_TIME = "ambiguous_time"
    CONFLICTING_SCHEDULE = "conflicting_schedule"
    NONEXISTENT_LOCAL_TIME = "nonexistent_local_time"
    PAST_ONCE = "past_once"


class ConversationRecallIntent(StrEnum):
    NONE = "none"
    RECALL = "recall"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ReminderIntentResult:
    """A privacy-safe, serializable description of a reminder command.

    The result intentionally has no field for the original command. ``title`` is
    only the task text left after deterministic command/temporal extraction.
    """

    status: ReminderIntentStatus
    schedule_kind: ReminderScheduleKind | None = None
    title: str | None = None
    local_time: time | None = None
    local_date: date | None = None
    timezone: str | None = None
    timezone_source: ReminderTimezoneSource | None = None
    scheduled_for: datetime | None = None
    error_code: ReminderIntentCode | None = None


@dataclass(frozen=True, slots=True)
class ReminderTimezoneHint:
    fragment: str
    timezone: str | None = None
    span: tuple[int, int] | None = None


def reminder_relative_day_offset(text: str) -> int | None:
    normalized = _normalize(text) if isinstance(text, str) else ""
    matches = tuple(_RELATIVE_DATE_PATTERN.finditer(normalized.casefold().replace("ё", "е")))
    if len(matches) != 1:
        return None
    return 0 if matches[0].group("relative") == "сегодня" else 1


@dataclass(frozen=True, slots=True)
class DailyOccurrence:
    local_date: date
    scheduled_for: datetime
    fold: int

    @property
    def scheduled_for_utc(self) -> datetime:
        return self.scheduled_for


@dataclass(frozen=True, slots=True)
class _TimezoneExtraction:
    value: str | None
    spans: tuple[tuple[int, int], ...]
    error: ReminderIntentCode | None = None


@dataclass(frozen=True, slots=True)
class _DateExtraction:
    value: date | None
    spans: tuple[tuple[int, int], ...]
    error: ReminderIntentCode | None = None


@dataclass(frozen=True, slots=True)
class _TimeExtraction:
    value: time | None
    spans: tuple[tuple[int, int], ...]
    error: ReminderIntentCode | None = None


# The first occurrence of an ambiguous wall time is used.  This is the PEP 495
# fold=0 side and is deterministic across retries and process restarts.
DAILY_AMBIGUOUS_FOLD = 0
_MAX_NONEXISTENT_DAYS_TO_SKIP = 370

_COMMAND_START = r"^\s*(?:пожалуйста\b[\s,;:—-]*)?"
_COMMAND_PATTERNS = (
    re.compile(_COMMAND_START + r"напомни(?:те)?(?:\s+мне)?\b", re.IGNORECASE),
    re.compile(_COMMAND_START + r"поставь(?:те)?\s+напоминание\b", re.IGNORECASE),
    re.compile(
        _COMMAND_START + r"(?:создай(?:те)?|установи(?:те)?)\s+напоминание\b",
        re.IGNORECASE,
    ),
    re.compile(_COMMAND_START + r"напоминай(?:те)?\b", re.IGNORECASE),
)
_DAILY_PATTERN = re.compile(r"\b(?:каждый\s+день|ежедневно)\b", re.IGNORECASE)
_IMPERATIVE_REMINDER_PATTERN = re.compile(r"\bнапоминай(?:те)?\b", re.IGNORECASE)
_MOSCOW_TIMEZONE_PATTERN = re.compile(
    r"(?:(?:(?<=\d)|\b)по\s+мск\b|\b(?:мск|по\s+москве|московское\s+время|"
    r"по\s+московскому\s+времени)\b)",
    re.IGNORECASE,
)
_IANA_TIMEZONE_PATTERN = re.compile(
    r"(?:(?P<prefix>(?:(?<=\d)|\b)по\s+))?"
    r"(?P<zone>[A-Za-z][A-Za-z0-9._+-]*/[A-Za-z0-9._+-]+"
    r"(?:/[A-Za-z0-9._+-]+)*)\b",
    re.IGNORECASE,
)
_MONTH_PATTERN = "|".join(MONTHS)
_RELATIVE_DATE_PATTERN = re.compile(r"\b(?:на\s+)?(?P<relative>сегодня|завтра)\b")
_NAMED_DATE_PATTERN = re.compile(
    rf"\b(?:на\s+)?(?P<day>\d{{1,2}})\s+"
    rf"(?P<month>{_MONTH_PATTERN})(?:\s+(?P<year>\d{{4}}))?\b"
)
_NUMERIC_DATE_PATTERN = re.compile(
    r"\b(?:на\s+)?(?P<day>\d{1,2})[./-](?P<month>\d{1,2})"
    r"[./-](?P<year>\d{4})\b"
)
_ISO_DATE_PATTERN = re.compile(r"\b(?:на\s+)?(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b")
_CLOCK_TIME_PATTERN = re.compile(
    r"\bв\s+(?P<hour>[01]?\d|2[0-3])\s*[:.]\s*"
    r"(?P<minute>[0-5]\d)(?!\d|\s*[.:]\s*\d)"
)
_ALTERNATIVE_CLOCK_TIME_PATTERN = re.compile(
    r"\b(?:или|либо)\s+(?P<hour>[01]?\d|2[0-3])\s*[:.]\s*"
    r"(?P<minute>[0-5]\d)(?!\d|\s*[.:]\s*\d)"
)
_INVALID_ALTERNATIVE_CLOCK_PATTERN = re.compile(r"\b(?:или|либо)\s+\d+\s*[:.]\s*\d+(?!\d)")
_HOUR_WORD_TIME_PATTERN = re.compile(
    r"\bв\s+(?P<hour>[01]?\d|2[0-3])\s+час(?:а|ов)?"
    r"(?:\s+(?P<minute>[0-5]?\d)\s+минут(?:у|ы)?)?"
    r"(?!\s+\d+\s+минут(?:у|ы)?)\b"
)
_SPACED_TIME_PATTERN = re.compile(
    r"\bв\s+(?P<hour>[01]?\d|2[0-3])\s+(?P<minute>[0-5]\d)(?!\s+\d)\b"
)
_INVALID_CLOCK_PATTERN = re.compile(r"\bв\s+\d+\s*[:.]\s*\d+(?!\d)")
_INVALID_SPACED_TIME_PATTERN = re.compile(r"\bв\s+\d+\s+\d+(?:\s+\d+)?\b")
_INVALID_HOUR_WORD_PATTERN = re.compile(r"\bв\s+\d+\s+час(?:а|ов)?(?:\s+\d+\s+минут(?:у|ы)?)?\b")
_EDGE_FILLER_PATTERN = re.compile(
    r"^(?:(?:мне|пожалуйста)\b[\s,;:—-]*)+|"
    r"(?:(?:мне|пожалуйста)\b[\s,;:—-]*)+$",
    re.IGNORECASE,
)
_RECALL_VOCATIVE = re.compile(r"^(?:nova|нова)\b[\s,;:—-]*", re.IGNORECASE)
_CONVERSATION_RECALL_PATTERNS = (
    re.compile(
        r"^(?:а\s+)?(?:о\s+ч[её]м\s+мы\s+(?:сейчас\s+)?говорили|"
        r"что\s+мы\s+(?:сейчас\s+)?обсуждали)[?!.…]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:пожалуйста[\s,]+)?напомни(?:\s+мне)?[\s,]+(?:про\s+)?"
        r"(?:наш(?:\s+с\s+тобой)?\s+)?разговор\b[\s\S]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:пожалуйста[\s,]+)?напомни(?:\s+мне)?[\s,]+(?:о\s+ч[её]м\s+мы\s+говорили|"
        r"что(?:\s+именно)?\s+мы\s+обсуждали)[?!.…]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:пожалуйста[\s,]+)?вспомни(?:\s+про)?\s+(?:наш(?:\s+с\s+тобой)?\s+)?"
        r"разговор\b[\s\S]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:мы\s+недавно\s+разговаривали\b[\s\S]{0,300}[.!?…]\s*)?"
        r"(?:ты\s+)?помнишь\s+(?:наш(?:\s+с\s+тобой)?\s+)?разговор)[?!.…]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:ты\s+)?помнишь[\s,]+о\s+ч[её]м\s+мы\s+говорили[?!.…]*$",
        re.IGNORECASE,
    ),
)
_AMBIGUOUS_RECALL_OR_REMINDER = re.compile(
    r"^напомни(?:\s+мне)?[\s,]+(?:плиз|пожалуйста)[?!.…]*$",
    re.IGNORECASE,
)


def _has_explicit_reminder_schedule(text: str) -> bool:
    if not any(pattern.search(text) for pattern in _COMMAND_PATTERNS):
        return False
    lowered = text.casefold().replace("ё", "е")
    relative = _RELATIVE_DATE_PATTERN.search(lowered)
    if relative is not None:
        return True
    if re.search(r"\bпослезавтра\b", lowered):
        return True
    return any(
        pattern.search(lowered)
        for pattern in (
            _DAILY_PATTERN,
            _NAMED_DATE_PATTERN,
            _NUMERIC_DATE_PATTERN,
            _ISO_DATE_PATTERN,
            _CLOCK_TIME_PATTERN,
            _HOUR_WORD_TIME_PATTERN,
            _SPACED_TIME_PATTERN,
            _INVALID_CLOCK_PATTERN,
            _INVALID_SPACED_TIME_PATTERN,
            _INVALID_HOUR_WORD_PATTERN,
        )
    )


def classify_conversation_recall(text: object) -> ConversationRecallIntent:
    """Conservatively separate conversational recall from executable reminders."""

    if not isinstance(text, str):
        return ConversationRecallIntent.NONE
    cleaned = _normalize(text)
    cleaned = _RECALL_VOCATIVE.sub("", cleaned, count=1).strip()
    if not cleaned:
        return ConversationRecallIntent.NONE
    if _has_explicit_reminder_schedule(cleaned):
        return ConversationRecallIntent.NONE
    if any(pattern.fullmatch(cleaned) for pattern in _CONVERSATION_RECALL_PATTERNS):
        return ConversationRecallIntent.RECALL
    if _AMBIGUOUS_RECALL_OR_REMINDER.fullmatch(cleaned):
        return ConversationRecallIntent.AMBIGUOUS
    return ConversationRecallIntent.NONE


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _zone(value: str) -> ZoneInfo:
    return validate_timezone(value)


def _valid_local_candidates(
    local_date: date,
    local_time: time,
    zone: ZoneInfo,
) -> list[tuple[int, datetime]]:
    naive = datetime.combine(local_date, local_time.replace(tzinfo=None))
    candidates: list[tuple[int, datetime]] = []
    seen_instants: set[datetime] = set()
    for fold in (0, 1):
        try:
            candidate = naive.replace(tzinfo=zone, fold=fold).astimezone(UTC)
            round_trip = candidate.astimezone(zone)
        except (OverflowError, ValueError):
            continue
        if round_trip.replace(tzinfo=None) != naive:
            continue
        if candidate in seen_instants:
            continue
        seen_instants.add(candidate)
        candidates.append((fold, candidate))
    return candidates


def calculate_daily_occurrence(
    local_date: date,
    local_time: time,
    timezone: str,
) -> DailyOccurrence:
    """Resolve the first valid daily occurrence on or after ``local_date``.

    Ambiguous wall times use ``fold=0``. A nonexistent wall time is skipped for
    that calendar date, rather than shifted or duplicated.
    """

    zone = _zone(timezone)
    candidate_date = local_date
    for _ in range(_MAX_NONEXISTENT_DAYS_TO_SKIP):
        candidates = _valid_local_candidates(candidate_date, local_time, zone)
        if candidates:
            by_fold = {fold: value for fold, value in candidates}
            selected_fold = (
                DAILY_AMBIGUOUS_FOLD if DAILY_AMBIGUOUS_FOLD in by_fold else min(by_fold)
            )
            return DailyOccurrence(candidate_date, by_fold[selected_fold], selected_fold)
        if candidate_date == date.max:
            break
        candidate_date += timedelta(days=1)
    raise ValueError("daily_occurrence_unresolvable")


def first_daily_occurrence_utc(
    local_time: time,
    timezone: str,
    *,
    now: datetime | None = None,
) -> datetime:
    """Return today's future occurrence, or the next valid calendar day's."""

    current = _as_utc(now or datetime.now(UTC))
    zone = _zone(timezone)
    local_today = current.astimezone(zone).date()
    occurrence = calculate_daily_occurrence(local_today, local_time, zone.key)
    if occurrence.local_date == local_today and occurrence.scheduled_for > current:
        return occurrence.scheduled_for
    return calculate_daily_occurrence(
        local_today + timedelta(days=1), local_time, zone.key
    ).scheduled_for


def next_daily_occurrence_utc(
    previous_occurrence_at: datetime,
    local_time: time,
    timezone: str,
) -> datetime:
    """Return the occurrence on the next valid local calendar date.

    This advances from the previous occurrence's local date, not by 24 hours, so
    it remains correct through offset changes.
    """

    zone = _zone(timezone)
    previous_local_date = _as_utc(previous_occurrence_at).astimezone(zone).date()
    return calculate_daily_occurrence(
        previous_local_date + timedelta(days=1), local_time, zone.key
    ).scheduled_for


def format_schedule_time(
    local_time: time,
    schedule_timezone: str,
    viewer_timezone: str,
    occurrence_date: date,
) -> str:
    """Format schedule wall time and, when useful, its viewer-local equivalent."""

    schedule_zone = _zone(schedule_timezone)
    viewer_zone = _zone(viewer_timezone)
    clock = local_time.replace(tzinfo=None).strftime("%H:%M")
    if schedule_zone.key == viewer_zone.key:
        return f"{clock} ({schedule_zone.key})"
    occurrence = calculate_daily_occurrence(occurrence_date, local_time, schedule_zone.key)
    viewer_clock = occurrence.scheduled_for.astimezone(viewer_zone).strftime("%H:%M")
    return (
        f"{clock} {_timezone_label(schedule_zone.key)} — "
        f"{viewer_clock} {_timezone_label(viewer_zone.key)}"
    )


def present_schedule_time(
    local_time: time,
    schedule_timezone: str,
    viewer_timezone: str,
    occurrence_date: date,
) -> str:
    return format_schedule_time(
        local_time,
        schedule_timezone,
        viewer_timezone,
        occurrence_date,
    )


def _timezone_label(timezone: str) -> str:
    return {
        "Europe/Moscow": "по Москве",
        "Europe/Saratov": "по Саратову",
    }.get(timezone, f"({timezone})")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


def _has_explicit_intent(lowered: str) -> bool:
    if any(pattern.match(lowered) for pattern in _COMMAND_PATTERNS):
        return True
    daily = _DAILY_PATTERN.match(lowered)
    if daily is None:
        return False
    remainder = lowered[daily.end() :].lstrip()
    if _IMPERATIVE_REMINDER_PATTERN.search(lowered) is not None:
        return True
    time_match = next(
        (
            match
            for pattern in (
                _CLOCK_TIME_PATTERN,
                _HOUR_WORD_TIME_PATTERN,
                _SPACED_TIME_PATTERN,
            )
            if (match := pattern.match(remainder)) is not None
        ),
        None,
    )
    if time_match is None:
        return False
    tail = remainder[time_match.end() :].lstrip()
    return bool(
        re.match(r"^(?:по\s+времени|в\s+часовом\s+поясе)\b", tail)
        or _MOSCOW_TIMEZONE_PATTERN.match(tail)
        or _IANA_TIMEZONE_PATTERN.match(tail)
    )


def _intent_spans(lowered: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for pattern in _COMMAND_PATTERNS:
        if match := pattern.match(lowered):
            spans.append(match.span())
            break
    spans.extend(match.span() for match in _IMPERATIVE_REMINDER_PATTERN.finditer(lowered))
    return tuple(spans)


def _extract_timezone(text: str) -> _TimezoneExtraction:
    spans: list[tuple[int, int]] = []
    values: list[str] = []
    for match in _MOSCOW_TIMEZONE_PATTERN.finditer(text):
        spans.append(match.span())
        values.append(_zone("мск").key)
    for match in _IANA_TIMEZONE_PATTERN.finditer(text):
        try:
            values.append(_zone(match.group("zone")).key)
        except (TypeError, ValueError):
            if match.group("prefix") is not None:
                spans.append(match.span())
                return _TimezoneExtraction(
                    None,
                    tuple(spans),
                    ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE,
                )
            continue
        spans.append(match.span())
    unique = set(values)
    if len(unique) > 1:
        return _TimezoneExtraction(
            None,
            tuple(spans),
            ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE,
        )
    return _TimezoneExtraction(next(iter(unique)) if unique else None, tuple(spans))


def reminder_explicit_timezone_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return exact deterministic timezone spans in normalized reminder text."""

    normalized = _normalize(text) if isinstance(text, str) else ""
    return _extract_timezone(normalized).spans if normalized else ()


def _extract_date(text: str, today: date) -> _DateExtraction:
    found: list[tuple[tuple[int, int], date]] = []
    invalid_spans: list[tuple[int, int]] = []

    for match in _RELATIVE_DATE_PATTERN.finditer(text):
        value = today if match.group("relative") == "сегодня" else today + timedelta(days=1)
        found.append((match.span(), value))

    for match in _NAMED_DATE_PATTERN.finditer(text):
        span = match.span()
        day = int(match.group("day"))
        month = MONTHS[match.group("month")]
        raw_year = match.group("year")
        try:
            value = date(int(raw_year) if raw_year is not None else today.year, month, day)
        except ValueError:
            value = None
        if value is None:
            invalid_spans.append(span)
        else:
            found.append((span, value))

    for pattern in (_NUMERIC_DATE_PATTERN, _ISO_DATE_PATTERN):
        for match in pattern.finditer(text):
            span = match.span()
            try:
                value = date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                )
            except ValueError:
                invalid_spans.append(span)
            else:
                found.append((span, value))

    all_spans = tuple(span for span, _ in found) + tuple(invalid_spans)
    if invalid_spans:
        return _DateExtraction(None, all_spans, ReminderIntentCode.INVALID_DATE)
    if len(found) > 1:
        return _DateExtraction(None, all_spans, ReminderIntentCode.AMBIGUOUS_DATE)
    if not found:
        return _DateExtraction(None, ())
    return _DateExtraction(found[0][1], (found[0][0],))


def _overlaps(span: tuple[int, int], blocked: Iterable[tuple[int, int]]) -> bool:
    return any(span[0] < stop and start < span[1] for start, stop in blocked)


def _extract_time(text: str, blocked: tuple[tuple[int, int], ...]) -> _TimeExtraction:
    found: list[tuple[tuple[int, int], time]] = []
    occupied = list(blocked)
    for pattern in (
        _HOUR_WORD_TIME_PATTERN,
        _SPACED_TIME_PATTERN,
        _CLOCK_TIME_PATTERN,
    ):
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, occupied):
                continue
            minute = match.groupdict().get("minute")
            found.append(
                (
                    span,
                    time(int(match.group("hour")), int(minute) if minute is not None else 0),
                )
            )
            occupied.append(span)

    if found:
        for match in _ALTERNATIVE_CLOCK_TIME_PATTERN.finditer(text):
            span = match.span()
            if _overlaps(span, occupied):
                continue
            found.append(
                (
                    span,
                    time(int(match.group("hour")), int(match.group("minute"))),
                )
            )
            occupied.append(span)

    invalid_spans: list[tuple[int, int]] = []
    if found:
        for match in _INVALID_ALTERNATIVE_CLOCK_PATTERN.finditer(text):
            span = match.span()
            if _overlaps(span, tuple(occupied)):
                continue
            invalid_spans.append(span)
            occupied.append(span)
    for pattern in (
        _INVALID_CLOCK_PATTERN,
        _INVALID_SPACED_TIME_PATTERN,
        _INVALID_HOUR_WORD_PATTERN,
    ):
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, tuple(occupied)):
                continue
            invalid_spans.append(span)
            occupied.append(span)

    all_spans = tuple(span for span, _ in found) + tuple(invalid_spans)
    if invalid_spans:
        return _TimeExtraction(None, all_spans, ReminderIntentCode.INVALID_TIME)
    if len(found) > 1:
        return _TimeExtraction(None, all_spans, ReminderIntentCode.AMBIGUOUS_TIME)
    if not found:
        return _TimeExtraction(None, ())
    return _TimeExtraction(found[0][1], (found[0][0],))


def _masked_title(text: str, spans: Iterable[tuple[int, int]]) -> str | None:
    characters = list(text)
    for start, stop in spans:
        for index in range(max(start, 0), min(stop, len(characters))):
            characters[index] = " "
    title = re.sub(r"\s+", " ", "".join(characters)).strip(" \t.,!?;:()[]{}\"'«»—-")
    previous = None
    while title and title != previous:
        previous = title
        title = _EDGE_FILLER_PATTERN.sub("", title).strip(" \t.,!?;:()[]{}\"'«»—-")
    return title or None


class ReminderIntentParser:
    def __init__(self, now_provider: Callable[[], datetime] | None = None):
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    def parse(
        self,
        text: str,
        profile_timezone: str = "Europe/Moscow",
        *,
        now: datetime | None = None,
        continuation: bool = False,
        previous: ReminderIntentResult | None = None,
        timezone_hint: ReminderTimezoneHint | None = None,
    ) -> ReminderIntentResult:
        normalized = _normalize(text) if isinstance(text, str) else ""
        lowered = normalized.lower().replace("ё", "е")
        is_continuation = continuation or previous is not None
        if not normalized or (not is_continuation and not _has_explicit_intent(lowered)):
            return ReminderIntentResult(
                ReminderIntentStatus.NOT_REMINDER,
                error_code=ReminderIntentCode.NO_EXPLICIT_INTENT,
            )

        current = _as_utc(now or self._now_provider())
        timezone_extraction = _extract_timezone(normalized)
        if timezone_hint is not None:
            fragment = _normalize(timezone_hint.fragment)
            haystack = normalized.casefold().replace("ё", "е")
            needle = fragment.casefold().replace("ё", "е")
            if timezone_hint.span is None:
                start = haystack.find(needle) if needle else -1
                valid_evidence = start >= 0 and haystack.find(needle, start + len(needle)) < 0
                stop = start + len(needle)
            else:
                start, stop = timezone_hint.span
                valid_evidence = bool(
                    needle
                    and 0 <= start < stop <= len(normalized)
                    and normalized[start:stop] == fragment
                )
            if not valid_evidence:
                timezone_extraction = _TimezoneExtraction(
                    None,
                    timezone_extraction.spans,
                    ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE,
                )
            elif timezone_hint.timezone is None:
                timezone_extraction = _TimezoneExtraction(
                    timezone_extraction.value,
                    (*timezone_extraction.spans, (start, stop)),
                    timezone_extraction.error,
                )
            else:
                try:
                    hinted_timezone = _zone(timezone_hint.timezone).key
                except (TypeError, ValueError):
                    timezone_extraction = _TimezoneExtraction(
                        None,
                        (*timezone_extraction.spans, (start, stop)),
                        ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE,
                    )
                else:
                    conflict = (
                        timezone_extraction.value is not None
                        and timezone_extraction.value != hinted_timezone
                    )
                    timezone_extraction = _TimezoneExtraction(
                        None if conflict else hinted_timezone,
                        (*timezone_extraction.spans, (start, stop)),
                        (
                            ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE
                            if conflict
                            else timezone_extraction.error
                        ),
                    )
        timezone: str | None = None
        timezone_source: ReminderTimezoneSource | None = None
        timezone_error = timezone_extraction.error
        if timezone_extraction.value is not None:
            timezone = timezone_extraction.value
            timezone_source = ReminderTimezoneSource.EXPLICIT
        elif timezone_error is None and previous is not None and previous.timezone is not None:
            try:
                timezone = _zone(previous.timezone).key
            except (TypeError, ValueError):
                timezone_error = ReminderIntentCode.INVALID_PROFILE_TIMEZONE
            else:
                timezone_source = previous.timezone_source or ReminderTimezoneSource.PROFILE
        elif timezone_error is None:
            try:
                timezone = _zone(profile_timezone).key
            except (TypeError, ValueError, AttributeError):
                timezone_error = ReminderIntentCode.INVALID_PROFILE_TIMEZONE
            else:
                timezone_source = ReminderTimezoneSource.PROFILE

        parsing_zone = _zone(timezone or "UTC")
        today = current.astimezone(parsing_zone).date()
        date_extraction = _extract_date(lowered, today)
        time_extraction = _extract_time(lowered, date_extraction.spans)
        daily_matches = tuple(match.span() for match in _DAILY_PATTERN.finditer(lowered))
        intent_spans = _intent_spans(lowered)
        title = _masked_title(
            normalized,
            (
                *intent_spans,
                *daily_matches,
                *timezone_extraction.spans,
                *date_extraction.spans,
                *time_extraction.spans,
            ),
        )
        if title is None and previous is not None:
            title = previous.title

        has_new_daily = bool(daily_matches)
        has_new_date = bool(date_extraction.spans)
        if has_new_daily:
            schedule_kind = ReminderScheduleKind.DAILY
            local_date = None
        elif has_new_date:
            schedule_kind = ReminderScheduleKind.ONCE
            local_date = date_extraction.value
        elif previous is not None and previous.schedule_kind is not None:
            schedule_kind = previous.schedule_kind
            local_date = previous.local_date
        else:
            schedule_kind = ReminderScheduleKind.ONCE
            local_date = None

        local_time = time_extraction.value
        if local_time is None and not time_extraction.spans and previous is not None:
            local_time = previous.local_time

        fields = {
            "schedule_kind": schedule_kind,
            "title": title,
            "local_time": local_time,
            "local_date": local_date,
            "timezone": timezone,
            "timezone_source": timezone_source,
        }
        error = (
            timezone_error
            or date_extraction.error
            or time_extraction.error
            or (ReminderIntentCode.CONFLICTING_SCHEDULE if has_new_daily and has_new_date else None)
        )
        if error is not None:
            return ReminderIntentResult(ReminderIntentStatus.INVALID, **fields, error_code=error)

        if (
            schedule_kind == ReminderScheduleKind.ONCE
            and local_date is not None
            and local_date < today
        ):
            return ReminderIntentResult(
                ReminderIntentStatus.INVALID,
                **fields,
                error_code=ReminderIntentCode.PAST_ONCE,
            )

        if title is None:
            return ReminderIntentResult(
                ReminderIntentStatus.NEEDS_TITLE,
                **fields,
                error_code=ReminderIntentCode.MISSING_TITLE,
            )
        if local_time is None:
            return ReminderIntentResult(
                ReminderIntentStatus.NEEDS_TIME,
                **fields,
                error_code=ReminderIntentCode.MISSING_TIME,
            )
        if schedule_kind == ReminderScheduleKind.ONCE and local_date is None:
            return ReminderIntentResult(
                ReminderIntentStatus.NEEDS_WHEN,
                **fields,
                error_code=ReminderIntentCode.MISSING_WHEN,
            )

        if schedule_kind == ReminderScheduleKind.DAILY:
            scheduled_for = first_daily_occurrence_utc(
                local_time,
                timezone or "UTC",
                now=current,
            )
        else:
            candidates = _valid_local_candidates(
                local_date,
                local_time,
                parsing_zone,
            )
            if not candidates:
                return ReminderIntentResult(
                    ReminderIntentStatus.INVALID,
                    **fields,
                    error_code=ReminderIntentCode.NONEXISTENT_LOCAL_TIME,
                )
            candidates_by_fold = {fold: value for fold, value in candidates}
            scheduled_for = candidates_by_fold.get(
                DAILY_AMBIGUOUS_FOLD,
                candidates_by_fold[min(candidates_by_fold)],
            )
            if scheduled_for <= current:
                return ReminderIntentResult(
                    ReminderIntentStatus.INVALID,
                    **fields,
                    scheduled_for=scheduled_for,
                    error_code=ReminderIntentCode.PAST_ONCE,
                )

        return ReminderIntentResult(
            ReminderIntentStatus.COMPLETE,
            **fields,
            scheduled_for=scheduled_for,
        )


def parse_reminder_intent(
    text: str,
    profile_timezone: str = "Europe/Moscow",
    *,
    now: datetime | None = None,
    continuation: bool = False,
    previous: ReminderIntentResult | None = None,
    timezone_hint: ReminderTimezoneHint | None = None,
) -> ReminderIntentResult:
    return ReminderIntentParser().parse(
        text,
        profile_timezone,
        now=now,
        continuation=continuation,
        previous=previous,
        timezone_hint=timezone_hint,
    )


__all__ = [
    "ConversationRecallIntent",
    "DAILY_AMBIGUOUS_FOLD",
    "DailyOccurrence",
    "ReminderIntentCode",
    "ReminderIntentParser",
    "ReminderIntentResult",
    "ReminderIntentStatus",
    "ReminderScheduleKind",
    "ReminderTimezoneSource",
    "ReminderTimezoneHint",
    "calculate_daily_occurrence",
    "classify_conversation_recall",
    "first_daily_occurrence_utc",
    "format_schedule_time",
    "next_daily_occurrence_utc",
    "parse_reminder_intent",
    "present_schedule_time",
    "reminder_explicit_timezone_spans",
    "reminder_relative_day_offset",
]
