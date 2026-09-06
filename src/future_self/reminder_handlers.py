from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .access import FULL_ACCESS_TIERS, is_full_access_tier
from .dates import full_numeric_date_spans
from .drafts import DraftInboxService, DraftResult, log_transition
from .models import (
    InboxItem,
    RecurringTaskReminderSchedule,
    TaskReminder,
    TaskState,
    User,
)
from .nova_companion_flow import NovaCompanionReminderCandidate
from .recurring_reminders import RecurringScheduleMutation
from .reminder_flow import (
    ReminderFlowAction,
    ReminderFlowPhase,
    ReminderFlowSession,
)
from .reminder_intent import (
    ConversationRecallIntent,
    ReminderIntentCode,
    ReminderIntentResult,
    ReminderIntentStatus,
    ReminderScheduleKind,
    ReminderSpeechAct,
    ReminderTimezoneHint,
    ReminderTimezoneSource,
    calculate_daily_occurrence,
    classify_conversation_recall,
    classify_reminder_speech_act,
    first_daily_occurrence_utc,
    reminder_action_is_quoted,
    reminder_explicit_timezone_spans,
    reminder_relative_day_offset,
)
from .schemas import ParsedThought, TemporalResolution
from .timezones import (
    ReminderTimezoneFragment,
    ReminderTimezoneStatus,
    extract_reminder_timezone_fragment,
    reminder_timezone_reply_fragment,
)

logger = logging.getLogger(__name__)

REMINDER_ACCESS_CHANGED_TEXT = "🔔 Напоминание\n\nДоступ изменился. Ничего не сохранено — повтори команду после проверки доступа."
REMINDER_STALE_TEXT = "Эта карточка уже неактуальна. Повтори команду напоминания."
REMINDER_TIMEZONE_RESOLVING_TEXT = "🔔 Определяю часовой пояс…"
REMINDER_TIMEZONE_CLARIFY_TEXT = "Уточни город вместе со страной или регионом"
REMINDER_TIMEZONE_RETRY_TEXT = (
    "Не удалось надёжно определить часовой пояс. Ничего не сохранено. "
    "Уточни город или попробуй ещё раз."
)
_COMPANION_REMINDER_SESSION_ATTR = "nova_companion_reminder_session"

_TIME_ONLY = re.compile(
    r"^\s*(?:в\s+)?(?:(?:[01]?\d|2[0-3])(?:\s*[:.]\s*[0-5]\d|\s*(?:ч\.?|час(?:а|ов)?)(?:\s+[0-5]?\d\s+минут(?:у|ы)?)?)|"
    r"(?:один|два|три|четыре|пять|шесть|семь|восемь|девять|десять|одиннадцать|двенадцать)\s+(?:утра|дня|вечера|ночи))"
    r"(?:\s*(?:по\s+)?(?:мск|по\s+москве|московское\s+время|[A-Za-z][A-Za-z0-9._+-]*/[A-Za-z0-9._+/-]+))?\s*$",
    re.IGNORECASE,
)
_BARE_CLOCK_FRAGMENT = re.compile(
    r"(?<![\w:.])(?P<hour>[01]?\d|2[0-3])\s*[:.]\s*(?P<minute>[0-5]\d)"
    r"(?!\d|\s*[.:]\s*\d)"
)
_UNSUPPORTED_RECURRENCE = re.compile(
    r"\b(?:кажд(?:ую|ой)\s+недел(?:ю|и|е|ей)?|по\s+будням|по\s+выходным|"
    r"кажд(?:ый|ую)\s+(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)|"
    r"через\s+день|раз\s+в\s+\d+\s+дн)\b",
    re.IGNORECASE,
)
_NATURAL_GUIDED_RECURRENCE_REQUEST = re.compile(
    r"(?=.*\b(?:напоминай\w*|напоминан\w*)\b)"
    r"(?=.*\b(?:регулярн\w*|почаще|чаще|постоянно|на\s+первое\s+время|"
    r"пока\s+не\s+привыкн\w*|раз\s+(?:\d+|[а-яё]+))\b)",
    re.IGNORECASE,
)
_NATURAL_GUIDED_RECURRENCE_IMPERATIVE = re.compile(
    r"^\s*(?:(?:нова|nova)\s*[,;:—-]?\s*)?"
    r"напоминай(?:те)?(?:\s+мне)?\b",
    re.IGNORECASE,
)
_NATURAL_GUIDED_RECURRENCE_CUE = re.compile(
    r"\b(?:регулярн\w*|почаще|чаще|постоянно|на\s+первое\s+время|"
    r"пока\s+не\s+привыкн\w*|раз\s+(?:\d+|[а-яё]+))\b",
    re.IGNORECASE,
)
_NATURAL_GUIDED_AMBIGUOUS_SUBJECT = re.compile(
    r"^(?:(?:об?|про|к|ко|для)\s+)?(?:это|этого|этому|этим|этом|том|тому|"
    r"нему|ней|них|такое|всё|все|главное)\b",
    re.IGNORECASE,
)
_NATURAL_GUIDED_CLARIFY_TEXT = (
    "Что именно тебе напоминать? Назови один конкретный предмет напоминания."
)
_REMINDER_TURN_MAX_CHARS = 4_096
_REMINDER_TURN_WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
_REMINDER_ACTION_WORD = re.compile(
    r"\b(?:напомни(?:те|ть)?|напомнил(?:а|и)?|постав(?:ь(?:те)?|ить)|ставить|"
    r"создай(?:те)?|создать|создавать|установи(?:те|ть)|устанавливать)\b",
    re.IGNORECASE,
)
_REMINDER_DIRECT_MODAL_WORDS = frozenset(
    {
        "можешь",
        "сможешь",
        "могла",
        "мог",
        "могли",
        "забудешь",
    }
)
_REMINDER_PREFIX_FILLERS = frozenset(
    {
        "nova",
        "нова",
        "ты",
        "вы",
        "мне",
        "ли",
        "бы",
        "не",
        "пожалуйста",
        "слушай",
        "подскажи",
        "а",
        "ну",
        "скажи",
        "эй",
    }
)
_REMINDER_REPORT_WORDS = frozenset(
    {
        "говорил",
        "говорила",
        "говорит",
        "сказал",
        "сказала",
        "сказали",
        "спросил",
        "спросила",
        "попросил",
        "попросила",
        "просил",
        "просила",
        "написал",
        "написала",
        "написали",
        "написано",
    }
)
_REMINDER_NOUN_WORDS = frozenset(
    {
        "напоминание",
        "напоминания",
    }
)
_REMINDER_VAGUE_TIME_WORDS = frozenset({"утром", "днём", "днем", "вечером"})
_REMINDER_FALLBACK_FRAME_WORDS = frozenset(
    {
        "я",
        "мы",
        "хочу",
        "хотел",
        "хотела",
        "хотелось",
        "чтобы",
        "было",
        "бы",
        "здорово",
        "хорошо",
        "удобно",
        "если",
        "можно",
        "попросить",
        "тебя",
        "вас",
        "ты",
        "мне",
        "пожалуйста",
    }
)
_REMINDER_AMBIGUOUS_TITLE = re.compile(
    r"^(?:(?:об?|про|к|ко|для)\s+)?"
    r"(?:это|этого|этому|этим|этом|том|тому|нему|ней|них)\b[.!?…]*$",
    re.IGNORECASE,
)
_REMINDER_SECOND_EVENT = re.compile(
    r"\b(?:или|либо|и|а\s+также)\s+"
    r"[^\W\d_][^\W_]{1,30}(?:ть|ться)\b",
    re.IGNORECASE,
)
_REMINDER_CLOCK_WITHOUT_PARSER_FORM = re.compile(
    r"\b(?P<prefix>в|на)\s+(?P<hour>[01]?\d|2[0-3])"
    r"(?:(?:\s*[:.]\s*(?P<minute>[0-5]\d))|"
    r"(?:\s*(?:ч\.?|час(?:а|ов)?)))?\b",
    re.IGNORECASE,
)
_GUIDED_RECURRENCE_PHASES = frozenset(
    {
        ReminderFlowPhase.RECURRENCE_FREQUENCY,
        ReminderFlowPhase.RECURRENCE_PERIOD,
        ReminderFlowPhase.RECURRENCE_DAYS,
        ReminderFlowPhase.RECURRENCE_TIMES,
        ReminderFlowPhase.RECURRENCE_SINGLE_TIME,
    }
)
_RECURRENCE_FREQUENCY = re.compile(
    r"(?:\b(?P<digits>\d{1,2})\s*раз(?:а)?\b|"
    r"\b(?P<leading_words>[а-яё-]+)\s+раз(?:а)?\b|"
    r"\bраз\s+(?P<words>[а-яё-]+)\b)",
    re.IGNORECASE,
)
_HOURLY_RECURRENCE = re.compile(
    r"\b(?:каждый\s+час|раз\s+в\s+час)\b",
    re.IGNORECASE,
)
_RECURRENCE_PERIODS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("day", re.compile(r"\b(?:в\s+течение\s+дня|за\s+день|в\s+день)\b", re.IGNORECASE)),
    ("morning", re.compile(r"\b(?:утром|в\s+течение\s+утра)\b", re.IGNORECASE)),
    ("afternoon", re.compile(r"\b(?:днём|днем|после\s+обеда)\b", re.IGNORECASE)),
    ("evening", re.compile(r"\b(?:вечером|в\s+течение\s+вечера)\b", re.IGNORECASE)),
)
_RECURRENCE_DAYS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("daily", re.compile(r"\b(?:каждый\s+день|ежедневно|все\s+дни)\b", re.IGNORECASE)),
    ("weekdays", re.compile(r"\b(?:по\s+будням|в\s+будни)\b", re.IGNORECASE)),
    ("weekends", re.compile(r"\b(?:по\s+выходным|в\s+выходные)\b", re.IGNORECASE)),
    (
        "date_range",
        re.compile(
            r"\b(?:с\s+\d{1,2}[./]\d{1,2}\s+по\s+\d{1,2}[./]\d{1,2}|на\s+\d+\s+дн)\b", re.IGNORECASE
        ),
    ),
)
_NUMBER_WORDS = {
    "один": 1,
    "однажды": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
}
_RECURRENCE_FREQUENCY_ERROR = (
    "Укажи только частоту числом, например: «раз десять». Фаза не изменена."
)
_RECURRENCE_PERIOD_ERROR = (
    "Укажи только часть дня, например: «утром» или «в течение дня». Фаза не изменена."
)
_RECURRENCE_DAYS_ERROR = (
    "Сейчас укажи только дни: для поддерживаемого варианта напиши «каждый день». Фаза не изменена."
)
_RECURRENCE_TIME_ERROR = "Укажи одно точное время, например: «19:00». Фаза не изменена."


class _ReminderPastAtSave(RuntimeError):
    pass


def _natural_guided_recurrence_title(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFKC", text)
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        return None
    normalized = " ".join(normalized.split()).strip()
    imperative = _NATURAL_GUIDED_RECURRENCE_IMPERATIVE.match(normalized)
    if imperative is None:
        return None
    remainder = normalized[imperative.end() :].strip(" ,:;—-")
    cues = tuple(_NATURAL_GUIDED_RECURRENCE_CUE.finditer(remainder))
    if len(cues) != 1:
        return None
    cue = cues[0]
    title = f"{remainder[: cue.start()]} {remainder[cue.end() :]}"
    title = " ".join(title.split()).strip(" ,:;—-.!?…")
    title = re.sub(r"^(?:пожалуйста\s+)?(?:об?|про)\s+", "", title, count=1, flags=re.I)
    if (
        not title
        or len(title) > 200
        or "?" in title
        or _NATURAL_GUIDED_AMBIGUOUS_SUBJECT.search(title)
    ):
        return None
    return title


def _bounded_reminder_turn(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFKC", text)
    if len(normalized) > _REMINDER_TURN_MAX_CHARS or any(
        unicodedata.category(character).startswith("C") for character in normalized
    ):
        return None
    normalized = " ".join(normalized.split()).strip()
    return normalized or None


def _canonical_elastic_reminder_turn(
    text: str,
    *,
    allow_semantic_fallback: bool = False,
) -> tuple[str, str] | None:
    """Return one direct, bounded reminder turn in the parser's canonical form.

    This pre-pass identifies a speech act by token/span relationships.  It does
    not infer a date, time or title: the regular reminder parser must ground
    those slots from the remaining exact user spans.
    """

    speech_act = classify_reminder_speech_act(text)
    allowed_speech_acts = {ReminderSpeechAct.DIRECT_REQUEST}
    if allow_semantic_fallback:
        allowed_speech_acts.add(ReminderSpeechAct.SEMANTIC_FALLBACK)
    if speech_act not in allowed_speech_acts:
        return None

    words = tuple(_REMINDER_TURN_WORD.finditer(text))
    action_matches = tuple(_REMINDER_ACTION_WORD.finditer(text))
    if not words or len(action_matches) != 1:
        return None
    action = action_matches[0]
    if reminder_action_is_quoted(text, action.start()):
        return None
    action_index = next(
        (index for index, word in enumerate(words) if word.start() == action.start()),
        None,
    )
    if action_index is None or action_index > 32:
        return None
    lowered_words = tuple(word.group(0).casefold().replace("ё", "е") for word in words)
    before = lowered_words[:action_index]
    prefix = text[: action.start()].strip()
    if re.search(r"[.!?]", prefix):
        if re.fullmatch(r"(?:nova|нова)\s*[.!?]", prefix, re.IGNORECASE) is None:
            return None

    action_word = lowered_words[action_index]
    infinitive = action_word.endswith("ть")
    if (
        infinitive
        and speech_act is not ReminderSpeechAct.SEMANTIC_FALLBACK
        and not any(word in _REMINDER_DIRECT_MODAL_WORDS for word in before)
    ):
        return None

    remove_spans: list[tuple[int, int]] = [action.span()]
    for index, word in enumerate(words):
        lowered = lowered_words[index]
        if index <= action_index and (
            lowered in _REMINDER_PREFIX_FILLERS or lowered in _REMINDER_DIRECT_MODAL_WORDS
        ):
            remove_spans.append(word.span())
        elif index == action_index + 1 and lowered == "мне":
            remove_spans.append(word.span())
        elif lowered in _REMINDER_VAGUE_TIME_WORDS:
            remove_spans.append(word.span())
        elif (
            speech_act is ReminderSpeechAct.SEMANTIC_FALLBACK
            and index <= action_index
            and lowered in _REMINDER_FALLBACK_FRAME_WORDS
        ):
            remove_spans.append(word.span())

    if action_word.startswith(("постав", "созда", "установ")):
        noun_index = next(
            (
                index
                for index in range(action_index + 1, min(len(words), action_index + 6))
                if lowered_words[index] in _REMINDER_NOUN_WORDS
            ),
            None,
        )
        if noun_index is None:
            return None
        remove_spans.append(words[noun_index].span())

    characters = list(text)
    for start, stop in remove_spans:
        for index in range(start, stop):
            characters[index] = " "
    body = " ".join("".join(characters).split()).strip(" ,:;—-.!?…")

    def canonical_clock(match: re.Match[str]) -> str:
        raw = match.group(0)
        minute = match.group("minute")
        if (
            match.group("prefix").casefold() == "на"
            and minute is None
            and not re.search(r"ч\.?|час", raw, re.IGNORECASE)
        ):
            return raw
        return f"в {int(match.group('hour')):02d}:{int(minute or 0):02d}"

    protected_dates = full_numeric_date_spans(body)

    def span_safe_clock(match: re.Match[str]) -> str:
        if any(match.start() < stop and start < match.end() for start, stop in protected_dates):
            return match.group(0)
        return canonical_clock(match)

    body = _REMINDER_CLOCK_WITHOUT_PARSER_FORM.sub(span_safe_clock, body)
    return f"напомни {body}".strip(), action.group(0)


def _monotonic_reminder_result(
    base: ReminderIntentResult,
    candidate: ReminderIntentResult,
) -> ReminderIntentResult:
    """Keep every independently grounded base slot while adding missing ones."""

    if base.status is ReminderIntentStatus.COMPLETE:
        return base
    if candidate.status is ReminderIntentStatus.NOT_REMINDER:
        return base
    if candidate.status is ReminderIntentStatus.INVALID and base.status not in {
        ReminderIntentStatus.INVALID,
        ReminderIntentStatus.NOT_REMINDER,
    }:
        return base
    if base.status is ReminderIntentStatus.NOT_REMINDER:
        return candidate
    use_candidate_title = bool(
        base.title is not None
        and candidate.title is not None
        and base.title != candidate.title
        and candidate.title.casefold() in base.title.casefold()
        and (
            (base.local_time is None and candidate.local_time is not None)
            or (base.local_date is None and candidate.local_date is not None)
        )
    )
    for name in ("schedule_kind", "local_time", "local_date", "timezone"):
        base_value = getattr(base, name)
        candidate_value = getattr(candidate, name)
        if base_value is not None and candidate_value is not None and base_value != candidate_value:
            return base
    if (
        base.title is not None
        and candidate.title is not None
        and base.title != candidate.title
        and not use_candidate_title
    ):
        return base
    return replace(
        candidate,
        schedule_kind=base.schedule_kind or candidate.schedule_kind,
        title=(candidate.title if use_candidate_title else base.title or candidate.title),
        local_time=base.local_time or candidate.local_time,
        local_date=base.local_date or candidate.local_date,
        timezone=base.timezone or candidate.timezone,
        timezone_source=base.timezone_source or candidate.timezone_source,
    )


def _semantic_fallback_reminder_text(text: object) -> str | None:
    normalized = _bounded_reminder_turn(text)
    if normalized is None:
        return None
    canonical = _canonical_elastic_reminder_turn(
        normalized,
        allow_semantic_fallback=True,
    )
    return canonical[0] if canonical is not None else None


def _reminder_turn_understanding(
    text: str,
    result: ReminderIntentResult,
    *,
    phase: ReminderFlowPhase | None = None,
    parser: Any | None = None,
    timezone: str | None = None,
    now: datetime | None = None,
) -> _ReminderTurnUnderstanding:
    """Make one transient routing decision without granting execution authority."""

    normalized = _bounded_reminder_turn(text)
    action_anchor: str | None = None
    elastic_routing = False
    semantic_ambiguity = False
    forced_transition: ReminderFlowPhase | None = None
    if phase is None and normalized is None:
        result = ReminderIntentResult(
            ReminderIntentStatus.NOT_REMINDER,
            error_code=ReminderIntentCode.NO_EXPLICIT_INTENT,
        )
    speech_act = (
        classify_reminder_speech_act(normalized)
        if phase is None and normalized is not None
        else ReminderSpeechAct.NONE
    )
    if phase is None and speech_act is ReminderSpeechAct.NON_EXECUTABLE:
        result = ReminderIntentResult(
            ReminderIntentStatus.NOT_REMINDER,
            error_code=ReminderIntentCode.NO_EXPLICIT_INTENT,
        )
    if (
        phase is None
        and normalized is not None
        and speech_act is ReminderSpeechAct.DIRECT_REQUEST
        and parser is not None
        and timezone is not None
    ):
        canonical = _canonical_elastic_reminder_turn(normalized)
        if canonical is not None:
            canonical_text, action_anchor = canonical
            base = result
            normalized_anchor = action_anchor.casefold().replace("ё", "е")
            elastic_routing = bool(
                result.status is ReminderIntentStatus.NOT_REMINDER
                or normalized_anchor.endswith("ть")
                or result.error_code is ReminderIntentCode.AMBIGUOUS_TIME
                or (
                    normalized_anchor.startswith(("постав", "созда", "установ"))
                    and re.search(
                        r"\bна\s+(?:[01]?\d|2[0-3])\s*[:.]\s*[0-5]\d\b",
                        normalized,
                        re.IGNORECASE,
                    )
                )
            )
            if elastic_routing:
                elastic = parser.parse(
                    canonical_text,
                    timezone,
                    now=now,
                    continuation=base.status is not ReminderIntentStatus.NOT_REMINDER,
                    previous=(
                        base if base.status is not ReminderIntentStatus.NOT_REMINDER else None
                    ),
                )
                result = _monotonic_reminder_result(base, elastic)
            ambiguous_title = bool(
                result.title is not None
                and base.error_code
                not in {
                    ReminderIntentCode.AMBIGUOUS_DATE,
                    ReminderIntentCode.AMBIGUOUS_TIME,
                    ReminderIntentCode.CONFLICTING_SCHEDULE,
                }
                and (
                    _REMINDER_AMBIGUOUS_TITLE.fullmatch(result.title.strip())
                    or _REMINDER_SECOND_EVENT.search(result.title)
                    or len(result.title) > 200
                )
            )
            if ambiguous_title:
                semantic_ambiguity = True
                forced_transition = ReminderFlowPhase.TITLE
                result = replace(
                    result,
                    status=ReminderIntentStatus.NEEDS_TITLE,
                    title=None,
                    scheduled_for=None,
                    error_code=ReminderIntentCode.MISSING_TITLE,
                )
            elif (
                elastic_routing
                and result.status is ReminderIntentStatus.INVALID
                and result.error_code is ReminderIntentCode.AMBIGUOUS_TIME
            ):
                semantic_ambiguity = True
                forced_transition = ReminderFlowPhase.TIME
                result = replace(
                    result,
                    status=ReminderIntentStatus.NEEDS_TIME,
                    local_time=None,
                    scheduled_for=None,
                    error_code=ReminderIntentCode.MISSING_TIME,
                )

    guided = phase in _GUIDED_RECURRENCE_PHASES or bool(
        normalized is not None and _NATURAL_GUIDED_RECURRENCE_REQUEST.search(normalized)
    )
    known = tuple(
        name
        for name, value in (
            ("title", result.title),
            ("date", result.local_date),
            ("time", result.local_time),
            ("timezone", result.timezone),
        )
        if value is not None
    )
    if result.status is ReminderIntentStatus.INVALID:
        missing = None
        allowed = (
            ReminderFlowPhase.PAST
            if result.error_code is ReminderIntentCode.PAST_ONCE
            else ReminderFlowPhase.INVALID
        )
    elif result.schedule_kind is None or (
        result.schedule_kind is ReminderScheduleKind.ONCE and result.local_date is None
    ):
        missing = "date"
        allowed = ReminderFlowPhase.WHEN
    elif result.local_time is None:
        missing = "time"
        allowed = ReminderFlowPhase.TIME
    elif result.title is None:
        missing = "title"
        allowed = ReminderFlowPhase.TITLE
    elif result.timezone is None:
        missing = None
        allowed = ReminderFlowPhase.INVALID
    else:
        missing = None
        allowed = ReminderFlowPhase.PREVIEW
    if phase in _GUIDED_RECURRENCE_PHASES:
        allowed = phase
    elif forced_transition is not None:
        missing = "title" if forced_transition is ReminderFlowPhase.TITLE else "time"
        allowed = forced_transition
    return _ReminderTurnUnderstanding(
        parser_result=result,
        elastic_routing=elastic_routing,
        conversational_intent=(
            "recurrence"
            if guided
            else "reminder"
            if result.status is not ReminderIntentStatus.NOT_REMINDER
            else "conversation"
        ),
        requested_action=(
            "continue"
            if phase is not None
            else "collect"
            if result.status is not ReminderIntentStatus.NOT_REMINDER
            else "none"
        ),
        title=result.title,
        local_date=result.local_date,
        local_time=result.local_time,
        known_slots=known,
        missing_slot=missing,
        exact_action_anchor=action_anchor,
        confidence=(
            "low"
            if semantic_ambiguity
            or result.status in {ReminderIntentStatus.INVALID, ReminderIntentStatus.NOT_REMINDER}
            else "high"
            if result.title is not None and (result.local_date or result.local_time)
            else "medium"
        ),
        ambiguous=semantic_ambiguity
        or result.error_code
        in {
            ReminderIntentCode.AMBIGUOUS_DATE,
            ReminderIntentCode.AMBIGUOUS_TIME,
            ReminderIntentCode.CONFLICTING_SCHEDULE,
        },
        allowed_transition=allowed,
    )


@dataclass(slots=True)
class _PendingReminderTimezone:
    session: ReminderFlowSession
    fragment: ReminderTimezoneFragment
    text: str
    continuation: bool
    previous: ReminderIntentResult | None
    clarification: bool
    source_message: Any | None


@dataclass(frozen=True, slots=True)
class _GuidedRecurrenceTurn:
    session: ReminderFlowSession | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _ReminderTurnUnderstanding:
    """Transient interpretation only; never durable scheduling authority."""

    parser_result: ReminderIntentResult
    elastic_routing: bool
    conversational_intent: Literal["reminder", "recurrence", "status", "conversation"]
    requested_action: Literal["collect", "continue", "inspect", "none"]
    title: str | None
    local_date: date | None
    local_time: time | None
    known_slots: tuple[str, ...]
    missing_slot: str | None
    exact_action_anchor: str | None
    confidence: Literal["high", "medium", "low"]
    ambiguous: bool
    allowed_transition: ReminderFlowPhase | None


@dataclass(slots=True)
class ReminderVoiceGateState:
    access_expected: bool = True
    access_failed: bool = False


class ReminderHandlers:
    reminder_sessions: Any
    reminder_intent_parser: Any
    recurring_reminder_service: Any
    draft_service: DraftInboxService

    @staticmethod
    def reminder_semantic_fallback_text(text: object) -> str | None:
        """Return a canonical, span-grounded fallback request without authority."""

        return _semantic_fallback_reminder_text(text)

    async def reminder_from_weekly_candidate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        title: str,
        schedule_wording: str,
        canonical_message: Any,
        expected_access_version: int,
    ) -> bool:
        """Hand one verified weekly candidate to the existing reminder flow.

        This adapter intentionally owns no parser, persistence or scheduler.  It
        accepts one normalized candidate, binds the existing flow to the weekly
        canonical message and leaves creation behind the regular reminder
        confirmation.
        """

        clean_title = " ".join(str(title).split()).strip()
        clean_schedule = " ".join(str(schedule_wording).split()).strip()
        if not clean_title or len(clean_title) > 200 or not clean_schedule:
            return False
        binding = await self._reminder_access(update)
        if (
            binding is None
            or binding.access_version != expected_access_version
            or update.effective_user is None
            or update.effective_chat is None
        ):
            return False
        current = await self.reminder_sessions.current(
            owner_id=binding.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        if current is not None:
            return False
        return await self._reminder_question_gate(
            update,
            context,
            f"Напомни {clean_schedule} {clean_title}",
            candidate_message=canonical_message,
            expected_access_version=expected_access_version,
            expected_session=None,
            voice_fenced=False,
            voice_state=None,
            weekly_candidate_handoff=True,
        )

    async def reminder_from_companion_candidate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        candidate: NovaCompanionReminderCandidate,
        canonical_message_id: int,
        expected_access_version: int,
    ) -> bool:
        """Hand one grounded offer to the existing reminder state machine."""

        if (
            not isinstance(candidate, NovaCompanionReminderCandidate)
            or type(canonical_message_id) is not int
            or canonical_message_id <= 0
            or update.effective_user is None
            or update.effective_chat is None
        ):
            return False
        binding = await self._reminder_access(update)
        if (
            binding is None
            or binding.access_version != expected_access_version
            or binding.timezone != candidate.timezone
        ):
            return False
        temporal = candidate.temporal
        schedule_kind = ReminderScheduleKind.ONCE if temporal is not None else None
        local_date = temporal.resolution.target_date if temporal is not None else None
        local_time = temporal.local_time if temporal is not None else None
        phase = (
            ReminderFlowPhase.RECURRENCE_FREQUENCY
            if candidate.guided_recurrence
            else self._reminder_fields_phase(
                title=candidate.title,
                schedule_kind=schedule_kind,
                local_date=local_date,
                local_time=local_time,
                timezone=candidate.timezone,
            )
        )
        session: ReminderFlowSession | None = None
        try:
            async with self._reminder_launch_lock:
                fresh = await self._reminder_access(update)
                if (
                    fresh is None
                    or fresh.id != binding.id
                    or fresh.access_version != expected_access_version
                    or fresh.timezone != candidate.timezone
                ):
                    return False
                await self.nova_memory_clear_current(update)
                await self.nova_clear_bound(fresh.id, update.effective_chat.id)
                session = await self.reminder_sessions.create(
                    owner_id=fresh.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    access_version=fresh.access_version,
                    title=candidate.title,
                    schedule_kind=schedule_kind,
                    local_date=local_date,
                    local_time=local_time,
                    timezone=candidate.timezone,
                    timezone_source=ReminderTimezoneSource.PROFILE,
                    phase=phase,
                    canonical_message_id=canonical_message_id,
                    profile_timezone=fresh.timezone,
                    weekly_candidate_handoff=False,
                    guided_recurrence=candidate.guided_recurrence,
                )
                delivery = await self._reminder_access(update)
                if (
                    delivery is None
                    or delivery.id != session.owner_id
                    or delivery.access_version != session.access_version
                    or delivery.timezone != candidate.timezone
                ):
                    await self._reminder_access_changed(context, session, source_message=None)
                    return False
                async with self._reminder_ui_lock:
                    live = await self.reminder_sessions.get_exact(session)
                    if live is None:
                        return False
                    session = live
                    text_value, markup = await self._reminder_screen(live)
                    exact = await self.reminder_sessions.get_exact(live)
                    if exact is None:
                        return False
                    await context.bot.edit_message_text(
                        chat_id=exact.chat_id,
                        message_id=exact.canonical_message_id,
                        text=text_value,
                        reply_markup=markup,
                    )
                bind_status = getattr(
                    self,
                    "nova_companion_bind_pending_reminder_status",
                    None,
                )
                if callable(bind_status):
                    bind_status(session, candidate)
                return True
        except BaseException as exc:
            if session is not None:
                exc.__dict__[_COMPANION_REMINDER_SESSION_ATTR] = session
            raise

    async def _reminder_weekly_return_markup(
        self,
        session: ReminderFlowSession,
    ) -> InlineKeyboardMarkup | None:
        hook = getattr(self, "weekly_review_reminder_return_markup", None)
        if not callable(hook):
            return None
        try:
            return await hook(
                owner_id=session.owner_id,
                telegram_user_id=session.telegram_user_id,
                chat_id=session.chat_id,
                canonical_message_id=session.canonical_message_id,
                access_version=session.access_version,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Reminder weekly return lookup failed error_type=%s",
                type(exc).__name__,
            )
            return None

    async def reminder_text_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        text = getattr(update.effective_message, "text", None)
        if not isinstance(text, str):
            return False
        return await self._reminder_question_gate(
            update,
            context,
            text,
            candidate_message=None,
            expected_access_version=None,
            expected_session=None,
            voice_fenced=False,
            voice_state=None,
            weekly_candidate_handoff=False,
        )

    async def reminder_voice_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        transcript: str,
        progress: Any,
        *,
        expected_access_version: int,
        expected_session: ReminderFlowSession | None,
        voice_state: ReminderVoiceGateState | None = None,
    ) -> bool:
        return await self._reminder_question_gate(
            update,
            context,
            transcript,
            candidate_message=progress,
            expected_access_version=expected_access_version,
            expected_session=expected_session,
            voice_fenced=True,
            voice_state=voice_state,
            weekly_candidate_handoff=False,
        )

    async def _reminder_question_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        *,
        candidate_message: Any | None,
        expected_access_version: int | None,
        expected_session: ReminderFlowSession | None,
        voice_fenced: bool,
        voice_state: ReminderVoiceGateState | None,
        weekly_candidate_handoff: bool,
        require_empty_generation: bool = False,
        allow_timezone_provider: bool = True,
    ) -> bool:
        recall_intent = classify_conversation_recall(text)
        binding = await self._reminder_access(update)
        if binding is None:
            if recall_intent is not ConversationRecallIntent.NONE:
                return False
            relative = self.date_resolver.resolve_relative_reminder(text, "UTC")
            if expected_session is None and relative:
                if voice_fenced and expected_access_version is not None:
                    await self._reminder_edit_access_candidate(candidate_message)
                return True
            if expected_session is not None:
                if voice_fenced:
                    await self._reminder_retire_voice_candidate(candidate_message)
                await self._reminder_access_changed(
                    context,
                    expected_session,
                    source_message=candidate_message or update.effective_message,
                )
                return True
            probe = self.reminder_intent_parser.parse(text, "UTC")
            handled = (
                probe.status is not ReminderIntentStatus.NOT_REMINDER
                or _NATURAL_GUIDED_RECURRENCE_IMPERATIVE.search(text) is not None
            )
            if handled and voice_fenced:
                await self._reminder_edit_access_candidate(candidate_message)
            elif (
                voice_fenced
                and expected_access_version is not None
                and voice_state is not None
                and voice_state.access_expected
            ):
                voice_state.access_failed = True
            return handled
        user = binding
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if await self._reminder_timezone_question_gate(
            update,
            context,
            text,
            user=user,
            candidate_message=candidate_message,
            expected_access_version=expected_access_version,
            expected_session=expected_session,
            voice_fenced=voice_fenced,
            weekly_candidate_handoff=weekly_candidate_handoff,
            require_empty_generation=require_empty_generation,
            allow_provider=allow_timezone_provider,
        ):
            return True

        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
            )
            if require_empty_generation and current is not None:
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            if voice_fenced and not self._reminder_expected_session_matches(
                current,
                expected_session,
            ):
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            if current is None and recall_intent is not ConversationRecallIntent.NONE:
                return False
            guided_title = None
            guided_imperative = bool(
                current is None
                and _NATURAL_GUIDED_RECURRENCE_REQUEST.search(text)
                and _NATURAL_GUIDED_RECURRENCE_IMPERATIVE.search(text)
            )
            if guided_imperative:
                guided_title = _natural_guided_recurrence_title(text)
                if guided_title is None:
                    delivery_binding = await self._reminder_access(update)
                    if (
                        delivery_binding is None
                        or delivery_binding.id != user.id
                        or delivery_binding.access_version != user.access_version
                    ):
                        if voice_fenced:
                            await self._reminder_edit_access_candidate(candidate_message)
                        return True
                    try:
                        if candidate_message is not None and hasattr(
                            candidate_message, "edit_text"
                        ):
                            await candidate_message.edit_text(
                                _NATURAL_GUIDED_CLARIFY_TEXT,
                                reply_markup=None,
                            )
                        else:
                            await update.effective_message.reply_text(_NATURAL_GUIDED_CLARIFY_TEXT)
                    except asyncio.CancelledError:
                        raise
                    except TelegramError as exc:
                        logger.warning(
                            "Reminder guided clarification failed error_type=%s",
                            type(exc).__name__,
                        )
                    return True
            fresh_binding = await self._reminder_access(update)
            if (
                fresh_binding is None
                or fresh_binding.id != user.id
                or fresh_binding.access_version != user.access_version
            ):
                probe = self.reminder_intent_parser.parse(text, user.timezone)
                if current is not None or expected_session is not None:
                    if voice_fenced:
                        await self._reminder_retire_voice_candidate(candidate_message)
                    await self._reminder_access_changed(
                        context,
                        current or expected_session,
                        source_message=candidate_message or update.effective_message,
                    )
                    return True
                relative = self.date_resolver.resolve_relative_reminder(text, user.timezone)
                handled = (
                    bool(relative)
                    or probe.status is not ReminderIntentStatus.NOT_REMINDER
                    or guided_imperative
                )
                if handled and voice_fenced:
                    await self._reminder_edit_access_candidate(candidate_message)
                elif voice_fenced and voice_state is not None and voice_state.access_expected:
                    voice_state.access_failed = True
                return handled
            user = fresh_binding
            if (
                expected_access_version is not None
                and user.access_version != expected_access_version
            ):
                stale_session = current or expected_session
                if stale_session is None:
                    probe = self.reminder_intent_parser.parse(text, user.timezone)
                    relative = self.date_resolver.resolve_relative_reminder(text, user.timezone)
                    handled = (
                        bool(relative)
                        or probe.status is not ReminderIntentStatus.NOT_REMINDER
                        or guided_imperative
                    )
                    if not handled:
                        if voice_fenced and voice_state is not None and voice_state.access_expected:
                            voice_state.access_failed = True
                        return False
                    if voice_fenced:
                        await self._reminder_edit_access_candidate(candidate_message)
                else:
                    if voice_fenced:
                        await self._reminder_retire_voice_candidate(candidate_message)
                    await self._reminder_access_changed(
                        context,
                        stale_session,
                        source_message=candidate_message or update.effective_message,
                    )
                return True

            status_hook = getattr(
                self,
                "nova_companion_pending_reminder_status_answer",
                None,
            )
            if current is not None and callable(status_hook):
                status_answer = status_hook(text)
                if isinstance(status_answer, str):
                    try:
                        if candidate_message is not None and hasattr(
                            candidate_message, "edit_text"
                        ):
                            await candidate_message.edit_text(
                                status_answer,
                                reply_markup=None,
                            )
                        else:
                            await update.effective_message.reply_text(status_answer)
                    except asyncio.CancelledError:
                        raise
                    except TelegramError as exc:
                        logger.warning(
                            "Reminder status delivery failed error_type=%s",
                            type(exc).__name__,
                        )
                    return True

            if current is not None and current.phase in _GUIDED_RECURRENCE_PHASES:
                await self.nova_memory_clear_current(update)
                await self.nova_clear_bound(user.id, chat_id)
                turn = await self._reminder_guided_recurrence_turn(current, text)
                updated = turn.session
                if updated is None:
                    await self._reminder_retire_voice_candidate(candidate_message)
                    return True
                delivery_binding = await self._reminder_access(update)
                if (
                    delivery_binding is None
                    or delivery_binding.id != updated.owner_id
                    or delivery_binding.access_version != updated.access_version
                ):
                    await self._reminder_access_changed(
                        context,
                        updated,
                        source_message=candidate_message or update.effective_message,
                    )
                    return True
                if turn.error is not None:
                    try:
                        if candidate_message is not None and hasattr(
                            candidate_message, "edit_text"
                        ):
                            await candidate_message.edit_text(turn.error, reply_markup=None)
                        else:
                            await update.effective_message.reply_text(turn.error)
                    except asyncio.CancelledError:
                        raise
                    except TelegramError as exc:
                        logger.warning(
                            "Reminder guided validation delivery failed error_type=%s",
                            type(exc).__name__,
                        )
                    return True
                if candidate_message is not None:
                    await self._reminder_retire_voice_candidate(candidate_message)
                async with self._reminder_ui_lock:
                    live = await self.reminder_sessions.get_exact(updated)
                    if live is not None:
                        await self._reminder_edit_canonical(
                            context,
                            live,
                            source_message=update.effective_message,
                        )
                return True

            fresh = (
                ReminderIntentResult(
                    status=ReminderIntentStatus.NEEDS_WHEN,
                    title=guided_title,
                    timezone=user.timezone,
                    timezone_source=ReminderTimezoneSource.PROFILE,
                    error_code=ReminderIntentCode.MISSING_WHEN,
                )
                if guided_title is not None
                else self.reminder_intent_parser.parse(text, user.timezone)
            )
            understanding = _reminder_turn_understanding(
                text,
                fresh,
                phase=current.phase if current is not None else None,
                parser=self.reminder_intent_parser,
                timezone=user.timezone,
                now=self._reminder_now(),
            )
            fresh = understanding.parser_result
            if (
                current is None
                and understanding.elastic_routing
                and understanding.title is not None
                and understanding.local_time is not None
                and understanding.local_date is None
                and fresh.schedule_kind is ReminderScheduleKind.ONCE
                and fresh.timezone is not None
            ):
                local_now = self._reminder_now().astimezone(ZoneInfo(fresh.timezone))
                if understanding.local_time > local_now.time().replace(
                    tzinfo=None,
                    second=0,
                    microsecond=0,
                ):
                    fresh = replace(
                        fresh,
                        status=ReminderIntentStatus.COMPLETE,
                        local_date=local_now.date(),
                        error_code=None,
                    )
                    understanding = replace(
                        understanding,
                        parser_result=fresh,
                        local_date=local_now.date(),
                        known_slots=tuple((*understanding.known_slots, "date")),
                        missing_slot=None,
                        allowed_transition=ReminderFlowPhase.PREVIEW,
                    )
            replacement = (
                current is not None and fresh.status is not ReminderIntentStatus.NOT_REMINDER
            )
            if current is None and understanding.requested_action == "none":
                return False
            if current is None and self.date_resolver.resolve_relative_reminder(
                text, user.timezone
            ):
                return False

            if current is not None and not replacement:
                parse_text = text
                if current.phase is ReminderFlowPhase.TIME and _TIME_ONLY.fullmatch(text):
                    parse_text = f"в {text.strip()}"
                elif current.phase is ReminderFlowPhase.TIME:
                    bare_clock = _BARE_CLOCK_FRAGMENT.search(text)
                    if bare_clock is not None and not text[
                        : bare_clock.start()
                    ].rstrip().casefold().endswith("в"):
                        parse_text = (
                            text[: bare_clock.start()]
                            + "в "
                            + bare_clock.group(0)
                            + text[bare_clock.end() :]
                        )
                result = self.reminder_intent_parser.parse(
                    parse_text,
                    user.timezone,
                    continuation=True,
                    previous=current.parser_state(),
                )
                if (
                    current.phase is ReminderFlowPhase.TIME
                    and current.title is not None
                    and result.title is not None
                ):
                    # A continuation may repeat the grounded subject together
                    # with the missing time ("На стрижку завтра 19:00").  The
                    # reminder handoff already owns the verified title; only
                    # the missing slot should be filled here.
                    result = replace(result, title=current.title)
            else:
                result = fresh

            await self.nova_memory_clear_current(update)
            if (
                result.status is not ReminderIntentStatus.NOT_REMINDER
                and _UNSUPPORTED_RECURRENCE.search(text)
            ):
                await self._reminder_show_unsupported(
                    update,
                    context,
                    current,
                    candidate_message,
                )
                return True

            result_timezone = result.timezone or user.timezone
            result_today = self._reminder_now().astimezone(ZoneInfo(result_timezone)).date()
            past_time_rejected = bool(
                (current is None or not replacement)
                and (current is None or current.phase is ReminderFlowPhase.TIME)
                and result.status is ReminderIntentStatus.INVALID
                and result.error_code is ReminderIntentCode.PAST_ONCE
                and result.schedule_kind is ReminderScheduleKind.ONCE
                and result.local_date == result_today
                and result.title is not None
                and result.local_time is not None
            )
            rejected_local_time = result.local_time if past_time_rejected else None
            if guided_title is not None:
                phase = ReminderFlowPhase.RECURRENCE_FREQUENCY
            elif past_time_rejected:
                result = replace(
                    result,
                    title=current.title if current is not None else result.title,
                    schedule_kind=(
                        current.schedule_kind if current is not None else result.schedule_kind
                    ),
                    local_date=current.local_date if current is not None else result.local_date,
                    local_time=None,
                    timezone=current.timezone if current is not None else result.timezone,
                    timezone_source=(
                        current.timezone_source if current is not None else result.timezone_source
                    ),
                )
                phase = ReminderFlowPhase.TIME
            else:
                phase = (
                    understanding.allowed_transition
                    if current is None and result is fresh
                    else self._reminder_phase(result)
                ) or self._reminder_phase(result)
            await self.nova_clear_bound(user.id, chat_id)
            canonical_message_id = current.canonical_message_id if current is not None else None
            if candidate_message is not None:
                candidate_id = getattr(candidate_message, "message_id", None)
                if current is None and isinstance(candidate_id, int):
                    canonical_message_id = candidate_id
                elif current is not None:
                    await self._reminder_retire_voice_candidate(candidate_message)

            session = await self.reminder_sessions.create(
                owner_id=user.id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=user.access_version,
                title=result.title,
                schedule_kind=result.schedule_kind,
                local_date=result.local_date,
                local_time=result.local_time,
                timezone=result.timezone or user.timezone,
                timezone_source=result.timezone_source or ReminderTimezoneSource.PROFILE,
                phase=phase,
                canonical_message_id=canonical_message_id,
                profile_timezone=user.timezone,
                weekly_candidate_handoff=(
                    weekly_candidate_handoff
                    or bool(
                        current is not None and not replacement and current.weekly_candidate_handoff
                    )
                ),
                past_time_rejected=past_time_rejected,
                rejected_local_time=rejected_local_time,
                guided_recurrence=guided_title is not None,
            )
            launch_message = candidate_message
            try:
                # Retire memory only while this exact reminder generation remains
                # current. If a concurrent memory launch already retired it, its
                # reciprocal cleanup must not remove the newer memory session.
                await self._nova_memory_clear_if_reminder_current(session)
                delivery_binding = await self._reminder_access(update)
                if (
                    delivery_binding is None
                    or delivery_binding.id != session.owner_id
                    or delivery_binding.access_version != session.access_version
                ):
                    await self._reminder_access_changed(
                        context,
                        session,
                        source_message=candidate_message or update.effective_message,
                    )
                    return True
                async with self._reminder_ui_lock:
                    live = await self.reminder_sessions.get_exact(session)
                    if live is None:
                        await self._reminder_retire_voice_candidate(candidate_message)
                        return True
                    session = live
                    if session.canonical_message_id is None:
                        sent = await update.effective_message.reply_text(
                            "🔔 Готовлю напоминание…",
                        )
                        launch_message = sent
                        message_id = getattr(sent, "message_id", None)
                        if not isinstance(message_id, int):
                            await self.reminder_sessions.clear(
                                owner_id=session.owner_id,
                                telegram_user_id=session.telegram_user_id,
                                chat_id=session.chat_id,
                                session_id=session.id,
                            )
                            await self._reminder_retire_voice_candidate(sent)
                            return True
                        bound = await self.reminder_sessions.update(
                            session,
                            canonical_message_id=message_id,
                        )
                        if bound is None:
                            await self._reminder_retire_voice_candidate(sent)
                            return True
                        session = bound
                        bound_access = await self._reminder_access(update)
                        if (
                            bound_access is None
                            or bound_access.id != bound.owner_id
                            or bound_access.access_version != bound.access_version
                        ):
                            cleared = await self.reminder_sessions.clear(
                                owner_id=bound.owner_id,
                                telegram_user_id=bound.telegram_user_id,
                                chat_id=bound.chat_id,
                                session_id=bound.id,
                            )
                            if cleared:
                                await self._reminder_edit_text(
                                    context,
                                    bound,
                                    REMINDER_ACCESS_CHANGED_TEXT,
                                    None,
                                    source_message=sent,
                                )
                            return True
                        delivered = await self._reminder_edit_canonical(
                            context,
                            bound,
                            source_message=sent,
                        )
                    else:
                        delivered = await self._reminder_edit_canonical(
                            context,
                            session,
                            source_message=candidate_message or update.effective_message,
                        )
                    if not delivered:
                        await self.reminder_sessions.clear(
                            owner_id=session.owner_id,
                            telegram_user_id=session.telegram_user_id,
                            chat_id=session.chat_id,
                            session_id=session.id,
                        )
                        await self._reminder_retire_voice_candidate(launch_message)
            except asyncio.CancelledError:
                await self.reminder_sessions.clear(
                    owner_id=session.owner_id,
                    telegram_user_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    session_id=session.id,
                )
                await self._reminder_retire_voice_candidate(launch_message)
                raise
            except TelegramError as exc:
                logger.warning(
                    "Reminder launch delivery failed error_type=%s",
                    type(exc).__name__,
                )
                await self.reminder_sessions.clear(
                    owner_id=session.owner_id,
                    telegram_user_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    session_id=session.id,
                )
                await self._reminder_retire_voice_candidate(launch_message)
            return True

    async def _reminder_timezone_question_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        *,
        user: User,
        candidate_message: Any | None,
        expected_access_version: int | None,
        expected_session: ReminderFlowSession | None,
        voice_fenced: bool,
        weekly_candidate_handoff: bool,
        require_empty_generation: bool = False,
        allow_provider: bool = True,
    ) -> bool:
        pending: _PendingReminderTimezone | None = None
        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if require_empty_generation and current is not None:
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            if current is not None and current.phase in _GUIDED_RECURRENCE_PHASES:
                # Guided slot grammar owns the whole turn. Date ranges, clock
                # lists and words such as "по" must not be reinterpreted as a
                # timezone fragment before the current phase validates them.
                return False
            fresh = self.reminder_intent_parser.parse(text, user.timezone)
            invalid_marker = False
            resolving_duplicate = False
            natural_exact_timezone = False
            resolving = bool(
                current is not None and current.phase is ReminderFlowPhase.TIMEZONE_RESOLVING
            )
            if voice_fenced and not self._reminder_expected_session_matches(
                current,
                expected_session,
            ):
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            fresh_user = await self._reminder_access(update)
            if (
                fresh_user is None
                or fresh_user.id != user.id
                or fresh_user.access_version != user.access_version
                or (
                    expected_access_version is not None
                    and fresh_user.access_version != expected_access_version
                )
            ):
                if (
                    current is None
                    and expected_session is None
                    and fresh.status is ReminderIntentStatus.NOT_REMINDER
                ):
                    return False
                if current is not None:
                    await self._reminder_retire_voice_candidate(candidate_message)
                    await self._reminder_access_changed(
                        context,
                        current,
                        source_message=candidate_message or update.effective_message,
                    )
                elif voice_fenced:
                    await self._reminder_edit_access_candidate(candidate_message)
                return True
            user = fresh_user
            clarification = bool(
                current is not None
                and current.phase
                in {
                    ReminderFlowPhase.TIMEZONE_CLARIFY,
                    ReminderFlowPhase.TIMEZONE_RETRY,
                }
                and fresh.status is ReminderIntentStatus.NOT_REMINDER
            )
            if clarification:
                try:
                    fragment = reminder_timezone_reply_fragment(text)
                except ValueError:
                    await self._reminder_retire_voice_candidate(candidate_message)
                    updated = await self.reminder_sessions.update(
                        current,
                        phase=ReminderFlowPhase.TIMEZONE_RETRY,
                    )
                    if updated is not None:
                        async with self._reminder_ui_lock:
                            await self._reminder_timezone_edit_fenced_locked(
                                update,
                                context,
                                updated,
                                source_message=update.effective_message,
                            )
                    return True
            else:
                try:
                    fragment = extract_reminder_timezone_fragment(text)
                except ValueError:
                    fragment = None
                    invalid_marker = True
                if (
                    not invalid_marker
                    and fragment is not None
                    and fresh.timezone_source is ReminderTimezoneSource.EXPLICIT
                ):
                    explicit_spans = reminder_explicit_timezone_spans(text)
                    marker_start = fragment.span[0]
                    location_start = marker_start + len(fragment.text) - len(fragment.location_text)
                    if len(explicit_spans) != 1 or explicit_spans[0][0] not in {
                        marker_start,
                        location_start,
                    }:
                        fragment = None
                        invalid_marker = True
                    else:
                        natural_exact_timezone = explicit_spans[0][0] == location_start
                if _UNSUPPORTED_RECURRENCE.search(text) or (
                    not invalid_marker
                    and (
                        (
                            fresh.timezone_source is ReminderTimezoneSource.EXPLICIT
                            and not natural_exact_timezone
                        )
                        or fresh.error_code is ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE
                    )
                ):
                    if resolving and fresh.status is not ReminderIntentStatus.COMPLETE:
                        await self._reminder_retire_voice_candidate(candidate_message)
                        return True
                    return False
                if fragment is None:
                    if (
                        not invalid_marker
                        and resolving
                        and current.timezone_fragment_fingerprint is not None
                    ):
                        try:
                            reply_fragment = reminder_timezone_reply_fragment(text)
                            resolving_duplicate = (
                                self.reminder_sessions.timezone_fragment_fingerprint(
                                    reply_fragment.text
                                )
                                == current.timezone_fragment_fingerprint
                            )
                        except ValueError:
                            resolving_duplicate = False
                    if (
                        not invalid_marker
                        and not resolving_duplicate
                        and resolving
                        and fresh.status is not ReminderIntentStatus.COMPLETE
                    ):
                        await self._reminder_retire_voice_candidate(candidate_message)
                        return True
                    if not invalid_marker and not resolving_duplicate:
                        return False
                if current is None and fresh.status is ReminderIntentStatus.NOT_REMINDER:
                    return False

            if resolving_duplicate:
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            if resolving and fresh.status is not ReminderIntentStatus.COMPLETE:
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            if invalid_marker:
                replace_invalid = bool(
                    current is not None and fresh.status is not ReminderIntentStatus.NOT_REMINDER
                )
                if current is not None and not replace_invalid:
                    retry_result = current.parser_state()
                else:
                    retry_result = self._reminder_timezone_pending_result(
                        fresh,
                        preserved_title=None,
                    )
                await self._reminder_timezone_store_and_render(
                    update,
                    context,
                    user,
                    retry_result,
                    current=current,
                    candidate_message=candidate_message,
                    phase=ReminderFlowPhase.TIMEZONE_RETRY,
                    preserve_session=current is not None and not replace_invalid,
                    timezone_fragment_fingerprint=None,
                    relative_day_offset=(
                        current.relative_day_offset
                        if current is not None and not replace_invalid
                        else reminder_relative_day_offset(text)
                    ),
                    calendar_anchor_utc=(
                        current.calendar_anchor_utc
                        if current is not None and not replace_invalid
                        else self._reminder_now()
                    ),
                    weekly_candidate_handoff=(
                        weekly_candidate_handoff
                        or bool(
                            current is not None
                            and not replace_invalid
                            and current.weekly_candidate_handoff
                        )
                    ),
                )
                return True

            fingerprint = self.reminder_sessions.timezone_fragment_fingerprint(text)
            if (
                current is not None
                and current.phase is ReminderFlowPhase.TIMEZONE_RESOLVING
                and current.timezone_fragment_fingerprint == fingerprint
            ):
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            replacement = bool(
                current is not None and fresh.status is not ReminderIntentStatus.NOT_REMINDER
            )
            previous = current.parser_state() if current is not None and not replacement else None
            if clarification:
                preliminary = current.parser_state()
            elif previous is not None:
                parsed = self.reminder_intent_parser.parse(
                    text,
                    user.timezone,
                    continuation=True,
                    previous=previous,
                )
                preliminary = self._reminder_timezone_pending_result(
                    parsed,
                    preserved_title=previous.title,
                )
            else:
                preliminary = self._reminder_timezone_pending_result(
                    fresh,
                    preserved_title=None,
                )

            local_outcome = self.timezone_resolver.resolve_reminder_locally(fragment)
            if local_outcome is not None:
                if (
                    local_outcome.candidate is None
                    or local_outcome.evidence_text is None
                    or local_outcome.evidence_span is None
                ):
                    raise RuntimeError("local reminder timezone outcome is incomplete")
                result = self._reminder_timezone_result(
                    preliminary,
                    timezone=local_outcome.candidate.timezone,
                    text=text,
                    evidence_text=local_outcome.evidence_text,
                    evidence_span=local_outcome.evidence_span,
                    profile_timezone=user.timezone,
                    previous=previous,
                    continuation=previous is not None,
                    clarification=clarification,
                    current=current if not replacement else None,
                )
                await self._reminder_timezone_store_and_render(
                    update,
                    context,
                    user,
                    result,
                    current=current,
                    candidate_message=candidate_message,
                    phase=self._reminder_phase(result),
                    preserve_session=clarification,
                    timezone_fragment_fingerprint=None,
                    relative_day_offset=(
                        current.relative_day_offset
                        if clarification and current is not None
                        else reminder_relative_day_offset(text)
                    ),
                    calendar_anchor_utc=(
                        current.calendar_anchor_utc
                        if clarification and current is not None
                        else self._reminder_now()
                    ),
                    weekly_candidate_handoff=(
                        weekly_candidate_handoff
                        or bool(
                            current is not None
                            and not replacement
                            and current.weekly_candidate_handoff
                        )
                    ),
                )
                return True

            relative_offset = (
                current.relative_day_offset
                if clarification and current is not None
                else reminder_relative_day_offset(text)
            )
            anchor = (
                current.calendar_anchor_utc
                if clarification and current is not None
                else self._reminder_now()
            )
            if not allow_provider:
                without_timezone = f"{text[: fragment.span[0]]} {text[fragment.span[1] :]}"
                locally_grounded = self.reminder_intent_parser.parse(
                    without_timezone,
                    user.timezone,
                    now=anchor,
                    continuation=previous is not None,
                    previous=previous,
                )
                retry_result = self._reminder_timezone_pending_result(
                    locally_grounded,
                    preserved_title=locally_grounded.title or preliminary.title,
                )
                await self._reminder_timezone_store_and_render(
                    update,
                    context,
                    user,
                    retry_result,
                    current=current,
                    candidate_message=candidate_message,
                    phase=ReminderFlowPhase.TIMEZONE_RETRY,
                    preserve_session=clarification,
                    timezone_fragment_fingerprint=None,
                    relative_day_offset=relative_offset,
                    calendar_anchor_utc=anchor,
                    weekly_candidate_handoff=(
                        weekly_candidate_handoff
                        or bool(
                            current is not None
                            and not replacement
                            and current.weekly_candidate_handoff
                        )
                    ),
                )
                return True
            stored = await self._reminder_timezone_store_and_render(
                update,
                context,
                user,
                preliminary,
                current=current,
                candidate_message=candidate_message,
                phase=ReminderFlowPhase.TIMEZONE_RESOLVING,
                preserve_session=clarification,
                timezone_fragment_fingerprint=fingerprint,
                relative_day_offset=relative_offset,
                calendar_anchor_utc=anchor,
                weekly_candidate_handoff=(
                    weekly_candidate_handoff
                    or bool(
                        current is not None and not replacement and current.weekly_candidate_handoff
                    )
                ),
            )
            if stored is None:
                return True
            session, source_message = stored
            pending = _PendingReminderTimezone(
                session=session,
                fragment=fragment,
                text=text,
                continuation=previous is not None,
                previous=previous,
                clarification=clarification,
                source_message=source_message,
            )

        async with self._reminder_launch_lock:
            if await self.reminder_sessions.get_exact(pending.session) is None:
                return True
            pre_provider = await self._reminder_access(update)
            if (
                pre_provider is None
                or pre_provider.id != pending.session.owner_id
                or pre_provider.access_version != pending.session.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    pending.session,
                    source_message=pending.source_message,
                )
                return True

        try:
            outcome = await self.timezone_resolver.resolve_reminder(pending.fragment)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._reminder_timezone_transition(
                    update,
                    context,
                    pending.session,
                    ReminderFlowPhase.TIMEZONE_RETRY,
                    source_message=pending.source_message,
                )
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise
        except Exception as exc:
            logger.warning(
                "Reminder timezone resolution failed error_type=%s",
                type(exc).__name__,
            )
            await self._reminder_timezone_transition(
                update,
                context,
                pending.session,
                ReminderFlowPhase.TIMEZONE_RETRY,
                source_message=pending.source_message,
            )
            return True

        if outcome.status is ReminderTimezoneStatus.AMBIGUOUS:
            if pending.clarification:
                await self._reminder_timezone_transition(
                    update,
                    context,
                    pending.session,
                    ReminderFlowPhase.TIMEZONE_CLARIFY,
                    source_message=pending.source_message,
                )
                return True
            if outcome.evidence_text is None or outcome.evidence_span is None:
                await self._reminder_timezone_transition(
                    update,
                    context,
                    pending.session,
                    ReminderFlowPhase.TIMEZONE_RETRY,
                    source_message=pending.source_message,
                )
                return True
            ambiguous = self._reminder_timezone_evidence_result(
                text=pending.text,
                evidence_text=outcome.evidence_text,
                evidence_span=outcome.evidence_span,
                profile_timezone=pending.session.profile_timezone,
                previous=pending.previous,
                continuation=pending.continuation,
                current=pending.session,
            )
            await self._reminder_timezone_apply_ambiguous(
                update,
                context,
                pending.session,
                ambiguous,
                source_message=pending.source_message,
            )
            return True
        if (
            outcome.status is ReminderTimezoneStatus.NOT_MENTIONED
            and not pending.fragment.is_strong
        ):
            ordinary = self.reminder_intent_parser.parse(
                pending.text,
                pending.session.profile_timezone,
                now=pending.session.calendar_anchor_utc or self._reminder_now(),
                continuation=pending.continuation,
                previous=pending.previous,
            )
            await self._reminder_timezone_apply_result(
                update,
                context,
                pending.session,
                ordinary,
                source_message=pending.source_message,
                explicit=False,
            )
            return True
        if outcome.status is not ReminderTimezoneStatus.RESOLVED or outcome.candidate is None:
            await self._reminder_timezone_transition(
                update,
                context,
                pending.session,
                ReminderFlowPhase.TIMEZONE_RETRY,
                source_message=pending.source_message,
            )
            return True

        if outcome.evidence_text is None or outcome.evidence_span is None:
            await self._reminder_timezone_transition(
                update,
                context,
                pending.session,
                ReminderFlowPhase.TIMEZONE_RETRY,
                source_message=pending.source_message,
            )
            return True
        result = self._reminder_timezone_result(
            pending.session.parser_state(),
            timezone=outcome.candidate.timezone,
            text=pending.text,
            evidence_text=outcome.evidence_text,
            evidence_span=outcome.evidence_span,
            profile_timezone=pending.session.profile_timezone,
            previous=pending.previous,
            continuation=pending.continuation,
            clarification=pending.clarification,
            current=pending.session,
        )
        await self._reminder_timezone_apply_result(
            update,
            context,
            pending.session,
            result,
            source_message=pending.source_message,
        )
        return True

    async def _reminder_timezone_store_and_render(
        self,
        update: Update,
        context: Any,
        user: User,
        result: ReminderIntentResult,
        *,
        current: ReminderFlowSession | None,
        candidate_message: Any | None,
        phase: ReminderFlowPhase,
        preserve_session: bool,
        timezone_fragment_fingerprint: str | None,
        relative_day_offset: int | None,
        calendar_anchor_utc: datetime | None,
        weekly_candidate_handoff: bool,
    ) -> tuple[ReminderFlowSession, Any | None] | None:
        await self.nova_memory_clear_current(update)
        await self.nova_clear_bound(user.id, update.effective_chat.id)
        canonical_message_id = current.canonical_message_id if current is not None else None
        if candidate_message is not None:
            candidate_id = getattr(candidate_message, "message_id", None)
            if current is None and isinstance(candidate_id, int):
                canonical_message_id = candidate_id
            elif current is not None:
                await self._reminder_retire_voice_candidate(candidate_message)

        if preserve_session and current is not None:
            session = await self.reminder_sessions.update(
                current,
                title=result.title,
                schedule_kind=result.schedule_kind,
                local_date=result.local_date,
                local_time=result.local_time,
                timezone=result.timezone or current.timezone,
                timezone_source=result.timezone_source or current.timezone_source,
                phase=phase,
                timezone_fragment_fingerprint=timezone_fragment_fingerprint,
                relative_day_offset=relative_day_offset,
                calendar_anchor_utc=calendar_anchor_utc,
            )
        else:
            session = await self.reminder_sessions.create(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                access_version=user.access_version,
                title=result.title,
                schedule_kind=result.schedule_kind,
                local_date=result.local_date,
                local_time=result.local_time,
                timezone=result.timezone or user.timezone,
                timezone_source=result.timezone_source or ReminderTimezoneSource.PROFILE,
                phase=phase,
                canonical_message_id=canonical_message_id,
                profile_timezone=user.timezone,
                timezone_fragment_fingerprint=timezone_fragment_fingerprint,
                relative_day_offset=relative_day_offset,
                calendar_anchor_utc=calendar_anchor_utc,
                weekly_candidate_handoff=weekly_candidate_handoff,
            )
        if session is None:
            return None
        delivery_snapshot = session
        try:
            await self._nova_memory_clear_if_reminder_current(session)
            delivery_user = await self._reminder_access(update)
            if (
                delivery_user is None
                or delivery_user.id != session.owner_id
                or delivery_user.access_version != session.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    session,
                    source_message=candidate_message or update.effective_message,
                )
                return None
            async with self._reminder_ui_lock:
                live = await self.reminder_sessions.get_exact(session)
                if live is None:
                    return None
                delivery_snapshot = live
                source_message = candidate_message or update.effective_message
                if live.canonical_message_id is None:
                    sent = await update.effective_message.reply_text("🔔 Готовлю напоминание…")
                    message_id = getattr(sent, "message_id", None)
                    if not isinstance(message_id, int):
                        await self.reminder_sessions.clear_exact(live)
                        return None
                    bound = await self.reminder_sessions.update(
                        live,
                        canonical_message_id=message_id,
                    )
                    if bound is None:
                        await self._reminder_retire_voice_candidate(sent)
                        return None
                    live = bound
                    delivery_snapshot = bound
                    source_message = sent
                delivered = await self._reminder_timezone_edit_fenced_locked(
                    update,
                    context,
                    live,
                    source_message=source_message,
                )
                if delivered is None:
                    return None
                return delivered, source_message
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self.reminder_sessions.clear_exact(delivery_snapshot))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise
        except TelegramError as exc:
            logger.warning(
                "Reminder timezone delivery failed error_type=%s",
                type(exc).__name__,
            )
            await self.reminder_sessions.clear_exact(delivery_snapshot)
            await self._reminder_retire_voice_candidate(candidate_message)
            return None

    @staticmethod
    def _reminder_timezone_pending_result(
        parsed: ReminderIntentResult,
        *,
        preserved_title: str | None,
    ) -> ReminderIntentResult:
        return ReminderIntentResult(
            status=parsed.status,
            schedule_kind=parsed.schedule_kind,
            title=preserved_title,
            local_time=parsed.local_time,
            local_date=parsed.local_date,
            timezone=parsed.timezone,
            timezone_source=parsed.timezone_source,
            scheduled_for=parsed.scheduled_for,
            error_code=parsed.error_code,
        )

    def _reminder_timezone_evidence_result(
        self,
        *,
        text: str,
        evidence_text: str,
        evidence_span: tuple[int, int],
        profile_timezone: str,
        previous: ReminderIntentResult | None,
        continuation: bool,
        current: ReminderFlowSession | None,
        timezone: str | None = None,
    ) -> ReminderIntentResult:
        return self.reminder_intent_parser.parse(
            text,
            profile_timezone,
            now=(
                current.calendar_anchor_utc
                if current is not None and current.calendar_anchor_utc is not None
                else self._reminder_now()
            ),
            continuation=continuation,
            previous=previous,
            timezone_hint=ReminderTimezoneHint(
                evidence_text,
                timezone,
                evidence_span,
            ),
        )

    def _reminder_timezone_result(
        self,
        preliminary: ReminderIntentResult,
        *,
        timezone: str,
        text: str,
        evidence_text: str,
        evidence_span: tuple[int, int],
        profile_timezone: str,
        previous: ReminderIntentResult | None,
        continuation: bool,
        clarification: bool,
        current: ReminderFlowSession | None,
    ) -> ReminderIntentResult:
        if not clarification:
            return self._reminder_timezone_evidence_result(
                text=text,
                evidence_text=evidence_text,
                evidence_span=evidence_span,
                profile_timezone=profile_timezone,
                previous=previous,
                continuation=continuation,
                current=current,
                timezone=timezone,
            )
        local_date = preliminary.local_date
        if (
            current is not None
            and current.relative_day_offset is not None
            and current.calendar_anchor_utc is not None
        ):
            local_date = current.calendar_anchor_utc.astimezone(
                ZoneInfo(timezone)
            ).date() + timedelta(days=current.relative_day_offset)
        return ReminderIntentResult(
            status=ReminderIntentStatus.COMPLETE,
            schedule_kind=preliminary.schedule_kind,
            title=preliminary.title,
            local_time=preliminary.local_time,
            local_date=local_date,
            timezone=timezone,
            timezone_source=ReminderTimezoneSource.EXPLICIT,
        )

    async def _reminder_timezone_apply_ambiguous(
        self,
        update: Update,
        context: Any,
        session: ReminderFlowSession,
        result: ReminderIntentResult,
        *,
        source_message: Any | None,
    ) -> None:
        async with self._reminder_launch_lock:
            live = await self.reminder_sessions.get_exact(session)
            if live is None:
                return
            user = await self._reminder_access(update)
            if (
                user is None
                or user.id != live.owner_id
                or user.access_version != live.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    live,
                    source_message=source_message,
                )
                return
            updated = await self.reminder_sessions.update(
                live,
                title=result.title,
                schedule_kind=result.schedule_kind,
                local_date=result.local_date,
                local_time=result.local_time,
                phase=ReminderFlowPhase.TIMEZONE_CLARIFY,
                timezone_fragment_fingerprint=None,
            )
            if updated is None:
                return
            async with self._reminder_ui_lock:
                await self._reminder_timezone_edit_fenced_locked(
                    update,
                    context,
                    updated,
                    source_message=source_message,
                )

    async def _reminder_timezone_apply_result(
        self,
        update: Update,
        context: Any,
        session: ReminderFlowSession,
        result: ReminderIntentResult,
        *,
        source_message: Any | None,
        explicit: bool = True,
    ) -> None:
        async with self._reminder_launch_lock:
            live = await self.reminder_sessions.get_exact(session)
            if live is None:
                return
            user = await self._reminder_access(update)
            if (
                user is None
                or user.id != live.owner_id
                or user.access_version != live.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    live,
                    source_message=source_message,
                )
                return
            updated = await self.reminder_sessions.update(
                live,
                title=result.title,
                schedule_kind=result.schedule_kind,
                local_date=result.local_date,
                local_time=result.local_time,
                timezone=result.timezone or live.timezone,
                timezone_source=(
                    ReminderTimezoneSource.EXPLICIT
                    if explicit
                    else result.timezone_source or live.timezone_source
                ),
                phase=self._reminder_phase(result),
                timezone_fragment_fingerprint=None,
            )
            if updated is None:
                return
            final_access = await self._reminder_access(update)
            if (
                final_access is None
                or final_access.id != updated.owner_id
                or final_access.access_version != updated.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    updated,
                    source_message=source_message,
                )
                return
            async with self._reminder_ui_lock:
                await self._reminder_timezone_edit_fenced_locked(
                    update,
                    context,
                    updated,
                    source_message=source_message,
                )

    async def _reminder_timezone_transition(
        self,
        update: Update,
        context: Any,
        session: ReminderFlowSession,
        phase: ReminderFlowPhase,
        *,
        source_message: Any | None,
    ) -> None:
        async with self._reminder_launch_lock:
            live = await self.reminder_sessions.get_exact(session)
            if live is None:
                return
            user = await self._reminder_access(update)
            if (
                user is None
                or user.id != live.owner_id
                or user.access_version != live.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    live,
                    source_message=source_message,
                )
                return
            updated = await self.reminder_sessions.update(live, phase=phase)
            if updated is None:
                return
            async with self._reminder_ui_lock:
                await self._reminder_timezone_edit_fenced_locked(
                    update,
                    context,
                    updated,
                    source_message=source_message,
                )

    async def _reminder_timezone_edit_fenced_locked(
        self,
        update: Update,
        context: Any,
        session: ReminderFlowSession,
        *,
        source_message: Any | None,
    ) -> ReminderFlowSession | None:
        exact = await self.reminder_sessions.get_exact(session)
        if exact is None:
            return None
        text_value, markup = await self._reminder_screen(exact)
        delivery = await self.reminder_sessions.get_exact(exact)
        if delivery is None:
            return None
        final_access = await self._reminder_access(update)
        if (
            final_access is None
            or final_access.id != delivery.owner_id
            or final_access.access_version != delivery.access_version
        ):
            cleared = await self.reminder_sessions.clear_exact(delivery)
            if cleared:
                await self._reminder_edit_text(
                    context,
                    delivery,
                    REMINDER_ACCESS_CHANGED_TEXT,
                    None,
                    source_message=source_message,
                )
            return None
        try:
            edited = await self._reminder_edit_text(
                context,
                delivery,
                text_value,
                markup,
                source_message=source_message,
            )
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self.reminder_sessions.clear_exact(delivery))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise
        if not edited:
            await self.reminder_sessions.clear_exact(delivery)
            await self._reminder_retire_voice_candidate(source_message)
            return None
        return delivery

    async def reminder_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        data = str(query.data or "")
        token = data.removeprefix("rmd:") if data.startswith("rmd:") else ""
        if not token or len(token) > 40:
            await query.answer(REMINDER_STALE_TEXT, show_alert=True)
            return
        async with self._reminder_launch_lock:
            claim = await self.reminder_sessions.claim(
                token,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                canonical_message_id=getattr(query.message, "message_id", None),
            )
            if claim is None:
                await query.answer(REMINDER_STALE_TEXT, show_alert=True)
                return
            await query.answer()
            capability, session = claim
            user = await self._reminder_access(update)
            if (
                user is None
                or user.id != session.owner_id
                or user.access_version != session.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    session,
                    source_message=query.message,
                )
                return
            action = capability.action
            if action == "cancel":
                await self.reminder_sessions.clear(
                    owner_id=session.owner_id,
                    telegram_user_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    session_id=session.id,
                )
                await self._reminder_edit_text(
                    context,
                    session,
                    "🔔 Напоминание отменено. Ничего не сохранено.",
                    await self._reminder_weekly_return_markup(session),
                    query=query,
                )
                return
            if action == "confirm":
                await self._reminder_confirm(update, context, query, user, session)
                return

            updated = await self._reminder_apply_action(session, action)
            if updated is None:
                return
            fresh_user = await self._reminder_access(update)
            if (
                fresh_user is None
                or fresh_user.id != updated.owner_id
                or fresh_user.access_version != updated.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    updated,
                    source_message=query.message,
                )
                return
            async with self._reminder_ui_lock:
                await self._reminder_edit_canonical(
                    context,
                    updated,
                    query=query,
                    source_message=query.message,
                )

    async def _reminder_apply_action(
        self,
        session: ReminderFlowSession,
        action: ReminderFlowAction,
    ) -> ReminderFlowSession | None:
        now = self._reminder_now()
        local_today = now.astimezone(ZoneInfo(session.timezone)).date()
        if action == "today":
            return await self._reminder_update_and_advance(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=local_today,
            )
        if action == "tomorrow":
            return await self._reminder_update_and_advance(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=local_today + timedelta(days=1),
            )
        if action == "daily":
            return await self._reminder_update_and_advance(
                session,
                schedule_kind=ReminderScheduleKind.DAILY,
                local_date=None,
            )
        if action == "choose_date":
            return await self.reminder_sessions.update(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=None,
                phase=ReminderFlowPhase.DATE,
            )
        if action == "edit":
            return await self.reminder_sessions.update(session, phase=ReminderFlowPhase.EDIT)
        if action == "edit_when":
            return await self.reminder_sessions.update(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=None,
                phase=ReminderFlowPhase.WHEN,
            )
        if action == "edit_time":
            return await self.reminder_sessions.update(
                session,
                local_time=None,
                phase=ReminderFlowPhase.TIME,
            )
        if action == "edit_title":
            return await self.reminder_sessions.update(
                session,
                title=None,
                phase=ReminderFlowPhase.TITLE,
            )
        if action == "retry_timezone":
            return await self.reminder_sessions.update(
                session,
                phase=ReminderFlowPhase.TIMEZONE_CLARIFY,
            )
        return None

    async def _reminder_update_and_advance(
        self,
        session: ReminderFlowSession,
        *,
        schedule_kind: ReminderScheduleKind,
        local_date: date | None,
    ) -> ReminderFlowSession | None:
        phase = self._reminder_fields_phase(
            title=session.title,
            schedule_kind=schedule_kind,
            local_date=local_date,
            local_time=session.local_time,
            timezone=session.timezone,
        )
        return await self.reminder_sessions.update(
            session,
            schedule_kind=schedule_kind,
            local_date=local_date,
            phase=phase,
        )

    async def _reminder_confirm(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        user: User,
        session: ReminderFlowSession,
    ) -> None:
        if session.phase is not ReminderFlowPhase.PREVIEW:
            return
        fresh = await self._reminder_access(update)
        if (
            fresh is None
            or fresh.id != session.owner_id
            or fresh.access_version != session.access_version
        ):
            await self._reminder_access_changed(context, session, source_message=query.message)
            return
        if session.schedule_kind is ReminderScheduleKind.ONCE:
            scheduled_for = self._reminder_scheduled_for(session)
            if scheduled_for is None or scheduled_for <= self._reminder_now():
                await self._reminder_render_past(context, query, session)
                return
        try:
            result, recurring = await self._reminder_save_atomic(session)
        except _ReminderPastAtSave:
            await self._reminder_render_past(context, query, session)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Reminder confirm failed error_type=%s", type(exc).__name__)
            live = await self.reminder_sessions.get_exact(session)
            if live is not None:
                async with self._reminder_ui_lock:
                    await self._reminder_edit_text(
                        context,
                        live,
                        "Не удалось сохранить напоминание. Ничего не изменено — попробуй ещё раз.",
                        await self._reminder_retry_keyboard(live),
                        query=query,
                    )
            return
        if not result.ok or result.inbox_item is None:
            failed_access = await self._reminder_access(update)
            if (
                failed_access is None
                or failed_access.id != session.owner_id
                or failed_access.access_version != session.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    session,
                    source_message=query.message,
                )
                return
            live = await self.reminder_sessions.get_exact(session)
            if live is not None:
                async with self._reminder_ui_lock:
                    await self._reminder_edit_text(
                        context,
                        live,
                        "Не удалось сохранить напоминание. Ничего не изменено — попробуй ещё раз.",
                        await self._reminder_retry_keyboard(live),
                        query=query,
                    )
            return
        final_user = await self._reminder_access(update)
        if (
            final_user is None
            or final_user.id != session.owner_id
            or final_user.access_version != session.access_version
        ):
            # The domain mutation itself is access-fenced. A change after commit
            # cannot be undone, but no stale private preview remains visible.
            await self._reminder_access_changed(context, session, source_message=query.message)
            return
        record_status = getattr(
            self,
            "nova_companion_record_confirmed_reminder",
            None,
        )
        if callable(record_status):
            record_status(
                session,
                result.inbox_item,
                reminder=result.reminder,
                recurring=recurring,
            )
        await self.reminder_sessions.clear(
            owner_id=session.owner_id,
            telegram_user_id=session.telegram_user_id,
            chat_id=session.chat_id,
            session_id=session.id,
        )
        if result.duplicate:
            success = f"✓ Уже настроено\n\n{result.inbox_item.title}"
        elif recurring is not None and recurring.schedule is not None:
            success = (
                "✅ Ежедневное напоминание включено\n\n"
                f"Что: {result.inbox_item.title}\n"
                f"Когда: каждый день в {recurring.schedule.local_time.strftime('%H:%M')}"
            )
        else:
            success = (
                "✅ Напоминание создано\n\n"
                f"Что: {result.inbox_item.title}\n"
                f"Когда: {self._reminder_once_label(session)}"
            )
        async with self._reminder_ui_lock:
            await self._reminder_edit_text(
                context,
                session,
                success,
                await self._reminder_weekly_return_markup(session),
                query=query,
            )

    async def _reminder_render_past(
        self,
        context: Any,
        query: Any,
        session: ReminderFlowSession,
    ) -> None:
        updated = await self.reminder_sessions.update(
            session,
            local_time=None,
            phase=ReminderFlowPhase.TIME,
            past_time_rejected=True,
            rejected_local_time=session.local_time,
        )
        if updated is not None:
            async with self._reminder_ui_lock:
                await self._reminder_edit_canonical(
                    context,
                    updated,
                    query=query,
                    source_message=query.message,
                )

    async def _reminder_save_atomic(
        self,
        session: ReminderFlowSession,
    ) -> tuple[DraftResult, RecurringScheduleMutation | None]:
        scheduled_for = self._reminder_scheduled_for(session)
        if session.title is None or session.local_time is None or scheduled_for is None:
            return DraftResult(False), None
        if (
            session.schedule_kind is ReminderScheduleKind.ONCE
            and scheduled_for <= self._reminder_now()
        ):
            raise _ReminderPastAtSave
        if session.schedule_kind is ReminderScheduleKind.ONCE:
            parsed = ParsedThought(
                kind="task",
                title=session.title,
                resolved_date=session.local_date,
                temporal_resolution=TemporalResolution(
                    resolved_at=scheduled_for,
                    remind_at=scheduled_for,
                    timezone=session.timezone,
                    resolved_local_date=session.local_date,
                    resolved_local_time=session.local_time,
                    precision="datetime",
                    original_expression="reminder_flow",
                    resolution_status="resolved",
                ),
            )
        else:
            parsed = ParsedThought(kind="task", title=session.title)

        async with self.db.session() as db_session:
            locked_owner_id = await db_session.scalar(
                update(User)
                .where(
                    User.id == session.owner_id,
                    User.telegram_id == session.telegram_user_id,
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                    User.access_version == session.access_version,
                )
                .values(updated_at=User.updated_at)
                .returning(User.id)
            )
            if locked_owner_id is None:
                return DraftResult(False), None
            if (
                session.schedule_kind is ReminderScheduleKind.ONCE
                and scheduled_for <= self._reminder_now()
            ):
                raise _ReminderPastAtSave
            if session.weekly_candidate_handoff:
                duplicate = await self._weekly_reminder_duplicate_in_session(
                    db_session,
                    session,
                    scheduled_for,
                )
                if duplicate is not None:
                    inbox_item, reminder = duplicate
                    return (
                        DraftResult(
                            True,
                            inbox_item=inbox_item,
                            reminder=reminder,
                            duplicate=True,
                        ),
                        None,
                    )
            draft = await self.draft_service.create_in_session(
                db_session,
                user_id=session.owner_id,
                telegram_user_id=session.telegram_user_id,
                chat_id=session.chat_id,
                source="reminder",
                raw_text=session.title,
                parsed=parsed,
            )
            result = await self.draft_service.confirm_in_session(
                db_session,
                draft.id,
                draft.version,
                session.telegram_user_id,
                session.chat_id,
                owner_locked=True,
                allow_saved_dedup=(
                    session.schedule_kind is ReminderScheduleKind.ONCE
                    and not session.weekly_candidate_handoff
                ),
                return_existing=True,
                expected_access_version=session.access_version,
            )
            if not result.ok or result.inbox_item is None:
                return result, None
            recurring: RecurringScheduleMutation | None = None
            if session.schedule_kind is ReminderScheduleKind.DAILY:
                recurring = await self.recurring_reminder_service.create_daily_in_session(
                    db_session,
                    session.owner_id,
                    result.inbox_item.id,
                    session.local_time,
                    timezone=session.timezone,
                    timezone_source=session.timezone_source.value,
                )
        log_transition(
            draft.id,
            session.telegram_user_id,
            "preview",
            "confirmed",
            "save_daily" if recurring is not None else "save_reminder",
            inbox_created=not result.duplicate,
        )
        return result, recurring

    async def _weekly_reminder_duplicate_in_session(
        self,
        db_session: Any,
        reminder: ReminderFlowSession,
        scheduled_for: datetime,
    ) -> tuple[InboxItem, TaskReminder | None] | None:
        """Find an exact/near owner reminder at the same schedule.

        This is deliberately part of the existing confirmation transaction:
        the weekly adapter still creates no separate reminder domain path, and
        a duplicate never creates a second Inbox/reminder row.
        """

        if (
            not reminder.weekly_candidate_handoff
            or reminder.title is None
            or reminder.schedule_kind is None
        ):
            return None
        if reminder.schedule_kind is ReminderScheduleKind.ONCE:
            rows = (
                await db_session.execute(
                    select(InboxItem, TaskReminder)
                    .join(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                    .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                    .where(
                        InboxItem.user_id == reminder.owner_id,
                        InboxItem.kind == "task",
                        InboxItem.status == "confirmed",
                        TaskState.owner_id == reminder.owner_id,
                        TaskState.status == "active",
                        TaskReminder.status.in_(("pending", "processing")),
                        TaskReminder.event_at == scheduled_for,
                        TaskReminder.timezone == reminder.timezone,
                    )
                    .order_by(InboxItem.id.desc())
                )
            ).all()
            for item, existing in rows:
                if self._weekly_reminder_titles_near(item.title, reminder.title):
                    return item, existing
            return None
        if reminder.schedule_kind is not ReminderScheduleKind.DAILY:
            return None
        rows = (
            await db_session.execute(
                select(InboxItem, RecurringTaskReminderSchedule)
                .join(
                    RecurringTaskReminderSchedule,
                    RecurringTaskReminderSchedule.inbox_item_id == InboxItem.id,
                )
                .join(TaskState, TaskState.inbox_item_id == InboxItem.id)
                .where(
                    InboxItem.user_id == reminder.owner_id,
                    InboxItem.kind == "task",
                    InboxItem.status == "confirmed",
                    TaskState.owner_id == reminder.owner_id,
                    TaskState.status == "active",
                    RecurringTaskReminderSchedule.owner_id == reminder.owner_id,
                    RecurringTaskReminderSchedule.status == "active",
                    RecurringTaskReminderSchedule.recurrence_kind == "daily",
                    RecurringTaskReminderSchedule.local_time == reminder.local_time,
                    RecurringTaskReminderSchedule.timezone == reminder.timezone,
                )
                .order_by(InboxItem.id.desc())
            )
        ).all()
        for item, _schedule in rows:
            if self._weekly_reminder_titles_near(item.title, reminder.title):
                return item, None
        return None

    @staticmethod
    def _weekly_reminder_titles_near(left: str, right: str) -> bool:
        def tokens(value: str) -> tuple[tuple[str, bool], ...]:
            normalized = unicodedata.normalize("NFKC", value)
            words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
            return tuple(
                (word.casefold(), index > 0 and word.istitle()) for index, word in enumerate(words)
            )

        left_tokens = tokens(left)
        right_tokens = tokens(right)
        if not left_tokens or len(left_tokens) != len(right_tokens):
            return False
        left_words = tuple(word for word, _proper in left_tokens)
        right_words = tuple(word for word, _proper in right_tokens)
        if left_words == right_words:
            return True

        differences = [
            (left_word, right_word, left_proper or right_proper)
            for (left_word, left_proper), (right_word, right_proper) in zip(
                left_tokens, right_tokens, strict=True
            )
            if left_word != right_word
        ]
        if len(differences) != 1:
            return False
        left_word, right_word, proper_name = differences[0]
        negations = {"без", "не", "нет", "ни", "no", "not", "never", "without"}
        if (
            proper_name
            or any(character.isdigit() for character in left_word + right_word)
            or left_word in negations
            or right_word in negations
            or min(len(left_word), len(right_word)) < 5
            or left_word[0] != right_word[0]
        ):
            return False
        return ReminderHandlers._weekly_reminder_single_edit(left_word, right_word)

    @staticmethod
    def _weekly_reminder_single_edit(left: str, right: str) -> bool:
        if left == right or abs(len(left) - len(right)) > 1:
            return False
        if len(left) == len(right):
            mismatches = [
                index
                for index, pair in enumerate(zip(left, right, strict=True))
                if pair[0] != pair[1]
            ]
            if len(mismatches) == 1:
                return True
            return bool(
                len(mismatches) == 2
                and mismatches[1] == mismatches[0] + 1
                and left[mismatches[0]] == right[mismatches[1]]
                and left[mismatches[1]] == right[mismatches[0]]
            )
        shorter, longer = (left, right) if len(left) < len(right) else (right, left)
        index = 0
        while index < len(shorter) and shorter[index] == longer[index]:
            index += 1
        return shorter[index:] == longer[index + 1 :]

    async def reminder_cancel_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        user = await self._reminder_access(update)
        if user is None:
            return False
        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if current is None:
                return False
            async with self._reminder_ui_lock:
                live = await self.reminder_sessions.get_exact(current)
                if live is None:
                    return False
                await self.reminder_sessions.clear(
                    owner_id=live.owner_id,
                    telegram_user_id=live.telegram_user_id,
                    chat_id=live.chat_id,
                    session_id=live.id,
                )
                await self._reminder_edit_text(
                    context,
                    live,
                    "🔔 Напоминание отменено. Ничего не сохранено.",
                    None,
                    source_message=update.effective_message,
                )
        return True

    async def reminder_sync_access(
        self,
        user: Any,
        chat_id: int,
        *,
        context: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
            )
            if current is None:
                return
            if (
                not is_full_access_tier(user.access_tier)
                or current.access_version != user.access_version
            ):
                await self.reminder_sessions.clear(
                    owner_id=current.owner_id,
                    telegram_user_id=current.telegram_user_id,
                    chat_id=current.chat_id,
                    session_id=current.id,
                )
                if context is not None:
                    async with self._reminder_ui_lock:
                        await self._reminder_edit_text(
                            context,
                            current,
                            REMINDER_ACCESS_CHANGED_TEXT,
                            None,
                            source_message=source_message,
                        )

    async def reminder_clear_current(self, update: Update) -> None:
        user = await self._reminder_access(update)
        if user is None:
            return
        async with self._reminder_launch_lock:
            async with self._reminder_ui_lock:
                await self.reminder_sessions.clear(
                    owner_id=user.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                )

    async def reminder_blocks_navigation(self, update: Update) -> bool:
        """Return whether an exact active reminder flow owns this chat update."""

        user = await self._reminder_access(update)
        if user is None or update.effective_user is None or update.effective_chat is None:
            return False
        current = await self.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        return current is not None

    async def reminder_public_command_gate(self, update: Update, context: Any) -> bool:
        """Keep an active reminder flow ahead of a competing public command."""

        user = await self._reminder_access(update)
        if user is None or update.effective_user is None or update.effective_chat is None:
            return False
        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if current is None:
                return False
            fresh = await self._reminder_access(update)
            live = await self.reminder_sessions.get_exact(current)
            if (
                fresh is None
                or fresh.id != user.id
                or fresh.access_version != user.access_version
                or live is None
            ):
                if fresh is None or fresh.access_version != current.access_version:
                    await self._reminder_access_changed(
                        context,
                        current,
                        source_message=update.effective_message,
                    )
                return True
            async with self._reminder_ui_lock:
                live = await self.reminder_sessions.get_exact(current)
                if live is not None:
                    await self._reminder_edit_canonical(
                        context,
                        live,
                        source_message=update.effective_message,
                    )
            return True

    async def _reminder_access(self, update: Update) -> User | None:
        telegram_user = update.effective_user
        chat = update.effective_chat
        if telegram_user is None or chat is None:
            return None
        try:
            status = await self.access_service.status(telegram_user.id)
            if status is None or not is_full_access_tier(status.access_tier):
                return None
            async with self.db.sessions() as session:
                return await session.scalar(
                    select(User).where(
                        User.telegram_id == telegram_user.id,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == status.access_version,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Reminder access lookup failed error_type=%s", type(exc).__name__)
            return None

    async def _reminder_access_changed(
        self,
        context: Any,
        session: ReminderFlowSession | None,
        *,
        source_message: Any | None,
    ) -> None:
        if session is None:
            return
        async with self._reminder_ui_lock:
            cleared = await self.reminder_sessions.clear(
                owner_id=session.owner_id,
                telegram_user_id=session.telegram_user_id,
                chat_id=session.chat_id,
                session_id=session.id,
            )
            if not cleared:
                return
            await self._reminder_edit_text(
                context,
                session,
                REMINDER_ACCESS_CHANGED_TEXT,
                None,
                source_message=source_message,
            )

    async def _reminder_guided_recurrence_turn(
        self,
        session: ReminderFlowSession,
        text: str,
    ) -> _GuidedRecurrenceTurn:
        normalized = " ".join(unicodedata.normalize("NFKC", text).split()).strip(" .!?…")
        phase = session.phase
        clocks = tuple(
            time(hour=int(match.group("hour")), minute=int(match.group("minute")))
            for match in _BARE_CLOCK_FRAGMENT.finditer(normalized)
        )
        date_range = _RECURRENCE_DAYS[-1][1].search(normalized) is not None

        # The persisted recurrence model supports one unbounded daily wall time.
        # A bounded range or several clocks is therefore an explicit request for
        # an unsupported shape, not evidence from which to guess a schedule.
        if phase is not ReminderFlowPhase.RECURRENCE_SINGLE_TIME and (
            date_range or len(clocks) > 1
        ):
            updated = await self.reminder_sessions.update(
                session,
                schedule_kind=None,
                local_date=None,
                local_time=None,
                recurrence_frequency_per_day=1,
                recurrence_active_period=None,
                recurrence_days="daily",
                phase=ReminderFlowPhase.RECURRENCE_SINGLE_TIME,
            )
            return _GuidedRecurrenceTurn(updated)

        if phase is ReminderFlowPhase.RECURRENCE_FREQUENCY:
            if _HOURLY_RECURRENCE.search(normalized) is not None:
                updated = await self.reminder_sessions.update(
                    session,
                    schedule_kind=None,
                    local_date=None,
                    local_time=None,
                    recurrence_frequency_per_day=1,
                    recurrence_active_period=None,
                    recurrence_days="daily",
                    phase=ReminderFlowPhase.RECURRENCE_SINGLE_TIME,
                )
                return _GuidedRecurrenceTurn(updated)
            frequency = self._reminder_recurrence_frequency(normalized)
            if frequency is None:
                return _GuidedRecurrenceTurn(session, _RECURRENCE_FREQUENCY_ERROR)
            period = self._reminder_recurrence_period(normalized)
            allowed = _RECURRENCE_FREQUENCY.sub(" ", normalized, count=1)
            if period is not None:
                for value, pattern in _RECURRENCE_PERIODS:
                    if value == period:
                        allowed = pattern.sub(" ", allowed, count=1)
                        break
            allowed = re.sub(r"\b(?:в|за)\b", " ", allowed, flags=re.I)
            if " ".join(allowed.split()).strip(" ,;:-"):
                return _GuidedRecurrenceTurn(session, _RECURRENCE_FREQUENCY_ERROR)
            return _GuidedRecurrenceTurn(
                await self.reminder_sessions.update(
                    session,
                    recurrence_frequency_per_day=frequency,
                    recurrence_active_period=period,
                    phase=(
                        ReminderFlowPhase.RECURRENCE_DAYS
                        if period is not None
                        else ReminderFlowPhase.RECURRENCE_PERIOD
                    ),
                )
            )
        if phase is ReminderFlowPhase.RECURRENCE_PERIOD:
            if _HOURLY_RECURRENCE.search(normalized) is not None:
                updated = await self.reminder_sessions.update(
                    session,
                    schedule_kind=None,
                    local_date=None,
                    local_time=None,
                    recurrence_frequency_per_day=1,
                    recurrence_active_period=None,
                    recurrence_days="daily",
                    phase=ReminderFlowPhase.RECURRENCE_SINGLE_TIME,
                )
                return _GuidedRecurrenceTurn(updated)
            period = self._reminder_recurrence_period(normalized, exact=True)
            if period is None:
                return _GuidedRecurrenceTurn(session, _RECURRENCE_PERIOD_ERROR)
            return _GuidedRecurrenceTurn(
                await self.reminder_sessions.update(
                    session,
                    recurrence_active_period=period,
                    phase=ReminderFlowPhase.RECURRENCE_DAYS,
                )
            )
        if phase is ReminderFlowPhase.RECURRENCE_DAYS:
            days = self._reminder_recurrence_days(normalized, exact=True)
            if days is None:
                return _GuidedRecurrenceTurn(session, _RECURRENCE_DAYS_ERROR)
            if days != "daily":
                updated = await self.reminder_sessions.update(
                    session,
                    recurrence_frequency_per_day=1,
                    recurrence_active_period=None,
                    recurrence_days="daily",
                    phase=ReminderFlowPhase.RECURRENCE_SINGLE_TIME,
                )
                return _GuidedRecurrenceTurn(updated)
            return _GuidedRecurrenceTurn(
                await self.reminder_sessions.update(
                    session,
                    recurrence_days=days,
                    phase=ReminderFlowPhase.RECURRENCE_TIMES,
                )
            )
        if phase is ReminderFlowPhase.RECURRENCE_SINGLE_TIME:
            if len(clocks) != 1 or not self._reminder_exact_clock_reply(normalized):
                return _GuidedRecurrenceTurn(session, _RECURRENCE_TIME_ERROR)
            return _GuidedRecurrenceTurn(
                await self.reminder_sessions.update(
                    session,
                    schedule_kind=ReminderScheduleKind.DAILY,
                    local_date=None,
                    local_time=clocks[0],
                    recurrence_frequency_per_day=1,
                    recurrence_active_period=None,
                    recurrence_days="daily",
                    phase=ReminderFlowPhase.PREVIEW,
                )
            )
        if len(clocks) != 1 or not self._reminder_exact_clock_reply(normalized):
            return _GuidedRecurrenceTurn(session, _RECURRENCE_TIME_ERROR)
        if session.recurrence_frequency_per_day != 1 or session.recurrence_days != "daily":
            return _GuidedRecurrenceTurn(
                await self.reminder_sessions.update(
                    session,
                    local_time=None,
                    recurrence_frequency_per_day=1,
                    recurrence_active_period=None,
                    recurrence_days="daily",
                    phase=ReminderFlowPhase.RECURRENCE_SINGLE_TIME,
                )
            )
        return _GuidedRecurrenceTurn(
            await self.reminder_sessions.update(
                session,
                schedule_kind=ReminderScheduleKind.DAILY,
                local_date=None,
                local_time=clocks[0],
                phase=ReminderFlowPhase.PREVIEW,
            )
        )

    @staticmethod
    def _reminder_recurrence_frequency(text: str, *, exact: bool = False) -> int | None:
        matched = (
            _RECURRENCE_FREQUENCY.fullmatch(text) if exact else _RECURRENCE_FREQUENCY.search(text)
        )
        if matched is None:
            pattern = re.fullmatch if exact else re.search
            return 1 if pattern(r"(?:один\s+раз|раз\s+в\s+день)", text, re.I) else None
        value = (
            int(matched.group("digits"))
            if matched.group("digits") is not None
            else _NUMBER_WORDS.get(
                (matched.group("words") or matched.group("leading_words") or "").casefold()
            )
        )
        return value if value is not None and 1 <= value <= 24 else None

    @staticmethod
    def _reminder_recurrence_period(
        text: str,
        *,
        exact: bool = False,
    ) -> Literal["day", "morning", "afternoon", "evening"] | None:
        matches = {
            value
            for value, pattern in _RECURRENCE_PERIODS
            if (pattern.fullmatch(text) if exact else pattern.search(text))
        }
        return next(iter(matches)) if len(matches) == 1 else None  # type: ignore[return-value]

    @staticmethod
    def _reminder_recurrence_days(
        text: str,
        *,
        exact: bool = False,
    ) -> Literal["daily", "weekdays", "weekends", "date_range"] | None:
        matches = {
            value
            for value, pattern in _RECURRENCE_DAYS
            if (pattern.fullmatch(text) if exact else pattern.search(text))
        }
        return next(iter(matches)) if len(matches) == 1 else None  # type: ignore[return-value]

    @staticmethod
    def _reminder_exact_clock_reply(text: str) -> bool:
        return (
            re.fullmatch(
                r"(?:в\s+)?(?:[01]?\d|2[0-3])\s*[:.]\s*[0-5]\d",
                text,
                re.IGNORECASE,
            )
            is not None
        )

    async def _reminder_screen(
        self,
        session: ReminderFlowSession,
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        phase = session.phase
        if phase is ReminderFlowPhase.RECURRENCE_FREQUENCY:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "🔁 Как часто напоминать в активный период?\n\n"
                "Например: «раз десять» или «десять раз в день».",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.RECURRENCE_PERIOD:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "🔁 В какой части дня должны действовать напоминания?\n\n"
                "Например: утром, вечером или в течение дня.",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.RECURRENCE_DAYS:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "🔁 В какие дни напоминать?\n\n"
                "Сейчас итоговый вариант поддерживает ежедневное расписание. "
                "Напиши: «каждый день».",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.RECURRENCE_TIMES:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "🔁 Назови одно точное время напоминания.\n\nНапример: 19:00.",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.RECURRENCE_SINGLE_TIME:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "Сейчас поддерживается только одно ежедневное напоминание в точное время. "
                "Несколько срабатываний, отдельные дни и конечный диапазон пока недоступны.\n\n"
                "Если подходит доступный вариант, укажи одно время, например: 19:00.",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.TIMEZONE_RESOLVING:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                REMINDER_TIMEZONE_RESOLVING_TEXT,
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.TIMEZONE_CLARIFY:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                REMINDER_TIMEZONE_CLARIFY_TEXT,
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.TIMEZONE_RETRY:
            tokens = await self.reminder_sessions.issue(
                session,
                ("retry_timezone", "cancel"),
            )
            return (
                REMINDER_TIMEZONE_RETRY_TEXT,
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Уточнить город",
                                callback_data=f"rmd:{tokens['retry_timezone']}",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "Отмена",
                                callback_data=f"rmd:{tokens['cancel']}",
                            )
                        ],
                    ]
                ),
            )
        if phase in {ReminderFlowPhase.WHEN, ReminderFlowPhase.PAST}:
            tokens = await self.reminder_sessions.issue(
                session,
                ("today", "tomorrow", "choose_date", "daily", "cancel"),
            )
            prefix = (
                "Выбранное время сегодня уже прошло. Ничего не переношу автоматически.\n\n"
                if phase is ReminderFlowPhase.PAST
                else ""
            )
            return (
                f"{prefix}🔔 Когда напомнить?",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton("Сегодня", callback_data=f"rmd:{tokens['today']}"),
                            InlineKeyboardButton(
                                "Завтра", callback_data=f"rmd:{tokens['tomorrow']}"
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Выбрать дату",
                                callback_data=f"rmd:{tokens['choose_date']}",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "🔁 Каждый день",
                                callback_data=f"rmd:{tokens['daily']}",
                            )
                        ],
                        [InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}")],
                    ]
                ),
            )
        if phase is ReminderFlowPhase.DATE:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "📅 На какую дату напомнить?\n\nНапиши или скажи дату, например: 15 августа.",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.TIME:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            prefix = (
                f"Время {session.rejected_local_time.strftime('%H:%M')} уже прошло. "
                "Дату сохраняю и ничего не переношу автоматически.\n\n"
                if session.past_time_rejected and session.rejected_local_time is not None
                else ""
            )
            question = (
                "🕒 Во сколько сегодня напомнить?"
                if session.past_time_rejected
                else "🕒 Во сколько напомнить?"
            )
            return (
                f"{prefix}{question}\n\n"
                "Напиши или скажи время, например: 19:30.\n"
                f"Использую твой часовой пояс: {self._reminder_timezone_label(session.timezone)}.",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.TITLE:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "📝 О чём напомнить?\n\n"
                "Напиши или скажи коротко, например:\n"
                "«заполнить дневник благодарностей».",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.EDIT:
            tokens = await self.reminder_sessions.issue(
                session,
                ("edit_when", "edit_time", "edit_title", "cancel"),
            )
            return (
                "✏️ Что изменить?",
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("Когда", callback_data=f"rmd:{tokens['edit_when']}")],
                        [
                            InlineKeyboardButton(
                                "Время", callback_data=f"rmd:{tokens['edit_time']}"
                            ),
                            InlineKeyboardButton(
                                "Название", callback_data=f"rmd:{tokens['edit_title']}"
                            ),
                        ],
                        [InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}")],
                    ]
                ),
            )
        if phase is ReminderFlowPhase.INVALID:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "Не удалось безопасно понять дату, время или часовой пояс. "
                "Отмени карточку и повтори команду точнее.",
                self._cancel_keyboard(tokens["cancel"]),
            )

        tokens = await self.reminder_sessions.issue(session, ("confirm", "edit", "cancel"))
        title = session.title or ""
        if session.schedule_kind is ReminderScheduleKind.DAILY:
            first = self._reminder_scheduled_for(session)
            first_label = self._reminder_datetime_label(first, session.timezone)
            text_value = (
                "🔁 Проверь напоминание\n\n"
                f"Что: {title}\n"
                f"Когда: каждый день в {session.local_time.strftime('%H:%M')} "
                f"({session.timezone})\n"
                f"Первый раз: {first_label}"
            )
            confirm_label = "✅ Включить"
        else:
            first = self._reminder_scheduled_for(session)
            text_value = (
                "🔔 Проверь напоминание\n\n"
                f"Что: {title}\n"
                f"Когда: {self._reminder_once_label(session)}"
            )
            confirm_label = "✅ Создать"
        if session.profile_timezone != session.timezone:
            text_value += (
                "\nВ твоём часовом поясе: "
                f"{self._reminder_datetime_label(first, session.profile_timezone)}"
            )
        return (
            text_value,
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            confirm_label,
                            callback_data=f"rmd:{tokens['confirm']}",
                        )
                    ],
                    [
                        InlineKeyboardButton("✏️ Изменить", callback_data=f"rmd:{tokens['edit']}"),
                        InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}"),
                    ],
                ]
            ),
        )

    async def _reminder_edit_canonical(
        self,
        context: Any,
        session: ReminderFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> bool:
        text_value, markup = await self._reminder_screen(session)
        return await self._reminder_edit_text(
            context,
            session,
            text_value,
            markup,
            query=query,
            source_message=source_message,
        )

    async def _reminder_edit_text(
        self,
        context: Any,
        session: ReminderFlowSession,
        text_value: str,
        markup: InlineKeyboardMarkup | None,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> bool:
        try:
            if query is not None:
                await query.edit_message_text(text_value, reply_markup=markup)
                return True
            if (
                source_message is not None
                and getattr(source_message, "message_id", None) == session.canonical_message_id
                and hasattr(source_message, "edit_text")
            ):
                await source_message.edit_text(text_value, reply_markup=markup)
                return True
            if session.canonical_message_id is not None:
                await context.bot.edit_message_text(
                    chat_id=session.chat_id,
                    message_id=session.canonical_message_id,
                    text=text_value,
                    reply_markup=markup,
                )
                return True
            return False
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if "message is not modified" not in str(exc).casefold():
                logger.warning("Reminder canonical edit failed error_type=%s", type(exc).__name__)
            return "message is not modified" in str(exc).casefold()
        except TelegramError as exc:
            logger.warning("Reminder canonical edit failed error_type=%s", type(exc).__name__)
            return False

    @staticmethod
    async def _reminder_edit_access_candidate(candidate: Any | None) -> None:
        if candidate is None or not hasattr(candidate, "edit_text"):
            return
        try:
            await candidate.edit_text(REMINDER_ACCESS_CHANGED_TEXT, reply_markup=None)
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if "message is not modified" not in str(exc).casefold():
                logger.warning("Reminder access edit failed error_type=%s", type(exc).__name__)
        except TelegramError as exc:
            logger.warning("Reminder access edit failed error_type=%s", type(exc).__name__)

    async def _reminder_show_unsupported(
        self,
        update: Update,
        context: Any,
        current: ReminderFlowSession | None,
        candidate_message: Any | None,
    ) -> None:
        text_value = (
            "Пока поддерживаются только разовые и ежедневные напоминания. "
            "Еженедельные, будние и произвольные интервалы ещё недоступны."
        )
        if current is not None:
            async with self._reminder_ui_lock:
                live = await self.reminder_sessions.get_exact(current)
                if live is None:
                    await self._reminder_retire_voice_candidate(candidate_message)
                    return
                await self.reminder_sessions.clear(
                    owner_id=live.owner_id,
                    telegram_user_id=live.telegram_user_id,
                    chat_id=live.chat_id,
                    session_id=live.id,
                )
                await self._reminder_edit_text(
                    context,
                    live,
                    text_value,
                    None,
                    source_message=candidate_message or update.effective_message,
                )
                await self._reminder_retire_voice_candidate(candidate_message)
        elif candidate_message is not None and hasattr(candidate_message, "edit_text"):
            await candidate_message.edit_text(text_value)
        else:
            await update.effective_message.reply_text(text_value)

    async def _reminder_retire_voice_candidate(self, candidate: Any | None) -> None:
        if candidate is None:
            return
        try:
            if hasattr(candidate, "delete"):
                await candidate.delete()
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            logger.warning(
                "Reminder voice transient cleanup failed error_type=%s", type(exc).__name__
            )

    async def _reminder_retry_keyboard(
        self,
        session: ReminderFlowSession,
    ) -> InlineKeyboardMarkup | None:
        tokens = await self.reminder_sessions.issue(session, ("confirm", "cancel"))
        if not tokens:
            return None
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Попробовать снова", callback_data=f"rmd:{tokens['confirm']}"
                    )
                ],
                [InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}")],
            ]
        )

    def _reminder_phase(self, result: ReminderIntentResult) -> ReminderFlowPhase:
        if result.status is ReminderIntentStatus.INVALID:
            return (
                ReminderFlowPhase.PAST
                if result.error_code is ReminderIntentCode.PAST_ONCE
                else ReminderFlowPhase.INVALID
            )
        return self._reminder_fields_phase(
            title=result.title,
            schedule_kind=result.schedule_kind,
            local_date=result.local_date,
            local_time=result.local_time,
            timezone=result.timezone,
        )

    def _reminder_fields_phase(
        self,
        *,
        title: str | None,
        schedule_kind: ReminderScheduleKind | None,
        local_date: date | None,
        local_time: time | None,
        timezone: str | None,
    ) -> ReminderFlowPhase:
        if schedule_kind is None or (
            schedule_kind is ReminderScheduleKind.ONCE and local_date is None
        ):
            return ReminderFlowPhase.WHEN
        if local_time is None:
            return ReminderFlowPhase.TIME
        if title is None:
            return ReminderFlowPhase.TITLE
        if timezone is None:
            return ReminderFlowPhase.INVALID
        if schedule_kind is ReminderScheduleKind.ONCE:
            occurrence = calculate_daily_occurrence(local_date, local_time, timezone)
            if (
                occurrence.local_date != local_date
                or occurrence.scheduled_for <= self._reminder_now()
            ):
                return ReminderFlowPhase.PAST
        return ReminderFlowPhase.PREVIEW

    def _reminder_scheduled_for(self, session: ReminderFlowSession) -> datetime | None:
        if session.local_time is None:
            return None
        if session.schedule_kind is ReminderScheduleKind.DAILY:
            return first_daily_occurrence_utc(
                session.local_time,
                session.timezone,
                now=self._reminder_now(),
            )
        if session.local_date is None:
            return None
        occurrence = calculate_daily_occurrence(
            session.local_date,
            session.local_time,
            session.timezone,
        )
        return occurrence.scheduled_for if occurrence.local_date == session.local_date else None

    def _reminder_once_label(self, session: ReminderFlowSession) -> str:
        value = self._reminder_scheduled_for(session)
        return ReminderHandlers._reminder_datetime_label(value, session.timezone)

    def _reminder_now(self) -> datetime:
        current = self._reminder_now_provider()
        return current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)

    @staticmethod
    def _reminder_datetime_label(value: datetime | None, timezone: str) -> str:
        if value is None:
            return "не определено"
        return f"{value.astimezone(ZoneInfo(timezone)).strftime('%d.%m.%Y %H:%M')} ({timezone})"

    @staticmethod
    def _reminder_timezone_label(timezone: str) -> str:
        return {
            "Europe/Moscow": "Москва (МСК)",
            "Europe/Saratov": "Саратов",
        }.get(timezone, timezone)

    @staticmethod
    def _cancel_keyboard(token: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Отмена", callback_data=f"rmd:{token}")]]
        )

    @staticmethod
    def _reminder_expected_session_matches(
        current: ReminderFlowSession | None,
        expected: ReminderFlowSession | None,
    ) -> bool:
        if current is None or expected is None:
            return current is expected
        return bool(
            current.id == expected.id
            and current.version == expected.version
            and current.access_version == expected.access_version
            and current.canonical_message_id == expected.canonical_message_id
        )


__all__ = [
    "REMINDER_ACCESS_CHANGED_TEXT",
    "REMINDER_STALE_TEXT",
    "ReminderHandlers",
    "ReminderVoiceGateState",
]
