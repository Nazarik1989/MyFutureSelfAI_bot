from __future__ import annotations

import re
import unicodedata
from typing import Literal, Protocol

from .schemas import WeeklyReviewExtraction

WEEKLY_REVIEW_MAX_INPUT_CHARS = 4_000
WEEKLY_REVIEW_LOCAL_MAX_INPUT_CHARS = 160

WeeklyReviewExtractionRoute = Literal["local", "provider"]

_EXPLICIT_FOCUS_LINE = re.compile(
    r"^[ \t]*фокус[ \t]*:[ \t]*(?P<focus>[^\r\n]+?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_STRUCTURED_HEADING = re.compile(
    r"\b(?:подход|шаг(?:и)?|напоминани[ея]|задач[аи]?)[ \t]*:",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"(?:^|\n)[ \t]*(?:[-*•]|\d{1,2}[.)])[ \t]+")
_COMPOSITE_CONNECTOR = re.compile(r";|\b(?:затем|а[ \t]+также|и[ \t]+затем)\b", re.IGNORECASE)
_SCHEDULE_WORDING = re.compile(
    r"(?:"
    r"\b\d{1,2}[:.]\d{2}\b|"
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|"
    r"\b(?:сегодня|завтра|послезавтра|ежедневно|каждый\s+день)\b|"
    r"\b(?:в\s+)?(?:понедельник|вторник|сред[ау]|четверг|пятниц[ау]|суббот[ау]|"
    r"воскресенье)\b"
    r")",
    re.IGNORECASE,
)
_INTERNAL_SENTENCE_END = re.compile(r"[.!?…](?=\s+\S)")
_SUPPORTED_DAILY_RECURRENCE = re.compile(
    r"\b(?:ежедневно|каждый\s+день)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_REMINDER_RECURRENCE = re.compile(
    r"\b(?:"
    r"еженедельно|ежемесячно|ежегодно|"
    r"по\s+(?:будням|выходным|понедельникам|вторникам|средам|четвергам|пятницам|субботам|воскресеньям)|"
    r"кажд(?:ую|ые|ый|ого|ой)\s+(?:недел\w*|месяц\w*|год\w*|понедельник\w*|вторник\w*|"
    r"сред\w*|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*)|"
    r"(?:каждые|раз\s+в)\s+(?:\d+|дв[ае]|три|четыре|пять|шесть|семь|восемь|девять|десять)\s+"
    r"(?:дн\w*|недел\w*|месяц\w*|год\w*)|"
    r"раз\s+в\s+(?:недел\w*|месяц\w*|год\w*)|"
    r"через\s+день"
    r")\b",
    re.IGNORECASE,
)


class WeeklyReviewExtractionAI(Protocol):
    async def extract_weekly_review(
        self,
        text: str,
        temporal_context: dict[str, str],
    ) -> WeeklyReviewExtraction: ...


def weekly_review_input(text: str) -> str:
    """Return a bounded input while preserving all evidence-visible characters."""

    if not isinstance(text, str):
        raise ValueError("weekly review input must be a string")
    clean = text.strip()
    if not clean:
        raise ValueError("weekly review input must not be empty")
    if len(clean) > WEEKLY_REVIEW_MAX_INPUT_CHARS:
        raise ValueError(
            f"weekly review input must not exceed {WEEKLY_REVIEW_MAX_INPUT_CHARS} characters"
        )
    return clean


def explicit_weekly_focus(text: str) -> str | None:
    """Extract one exact, line-scoped ``Фокус:`` value without paraphrasing it."""

    clean = weekly_review_input(text)
    matches = list(_EXPLICIT_FOCUS_LINE.finditer(clean))
    if len(matches) > 1:
        raise ValueError("weekly review input contains multiple explicit focus labels")
    if not matches:
        return None
    return matches[0].group("focus")


def classify_weekly_review_input(text: str) -> WeeklyReviewExtractionRoute:
    """Choose the deterministic local path only for one short, simple focus."""

    clean = weekly_review_input(text)
    explicit_focus = explicit_weekly_focus(clean)
    if len(clean) > WEEKLY_REVIEW_LOCAL_MAX_INPUT_CHARS or "\n" in clean or "\r" in clean:
        return "provider"
    if (
        _STRUCTURED_HEADING.search(clean)
        or _LIST_ITEM.search(clean)
        or _COMPOSITE_CONNECTOR.search(clean)
    ):
        return "provider"
    if _SCHEDULE_WORDING.search(clean) or _INTERNAL_SENTENCE_END.search(clean):
        return "provider"
    if explicit_focus is not None:
        match = _EXPLICIT_FOCUS_LINE.fullmatch(clean)
        return "local" if match is not None else "provider"
    return "local"


def extract_weekly_review_locally(text: str) -> WeeklyReviewExtraction:
    """Extract a short focus without invoking an AI provider."""

    clean = weekly_review_input(text)
    if classify_weekly_review_input(clean) != "local":
        raise ValueError("weekly review input requires provider extraction")
    return WeeklyReviewExtraction(focus=explicit_weekly_focus(clean) or clean)


def _weekly_reminder_schedule_kind(
    schedule_wording: str,
    evidence: str,
) -> Literal["one_shot", "daily"]:
    """Accept only recurrence families supported by the reminder engine."""

    if _UNSUPPORTED_REMINDER_RECURRENCE.search(evidence):
        raise ValueError("weekly reminder recurrence supports one-shot or daily only")
    evidence_is_daily = _SUPPORTED_DAILY_RECURRENCE.search(evidence) is not None
    schedule_is_daily = _SUPPORTED_DAILY_RECURRENCE.search(schedule_wording) is not None
    if evidence_is_daily != schedule_is_daily:
        raise ValueError("weekly reminder daily schedule must be copied from its evidence")
    return "daily" if schedule_is_daily else "one_shot"


def validate_weekly_review_extraction(
    text: str,
    extraction: WeeklyReviewExtraction,
) -> WeeklyReviewExtraction:
    """Fence untrusted extraction output against the exact source text."""

    clean = weekly_review_input(text)
    verified = WeeklyReviewExtraction.model_validate(extraction.model_dump())
    exact_focus = explicit_weekly_focus(clean)
    if exact_focus is not None:
        verified = WeeklyReviewExtraction.model_validate(
            {**verified.model_dump(), "focus": exact_focus}
        )

    for candidate in verified.reminder_candidates:
        first_evidence = clean.find(candidate.evidence)
        if first_evidence < 0 or clean.find(candidate.evidence, first_evidence + 1) >= 0:
            raise ValueError("weekly reminder evidence must be one exact unique input span")
        if candidate.schedule_wording not in candidate.evidence:
            raise ValueError("weekly reminder schedule must be copied from its evidence")
        normalized_title = unicodedata.normalize("NFKC", candidate.title).casefold()
        normalized_evidence = unicodedata.normalize("NFKC", candidate.evidence).casefold()
        if normalized_title not in normalized_evidence:
            raise ValueError("weekly reminder title must be copied from its evidence")
        _weekly_reminder_schedule_kind(candidate.schedule_wording, candidate.evidence)
    return verified


async def extract_weekly_review_input(
    ai: WeeklyReviewExtractionAI,
    text: str,
    temporal_context: dict[str, str],
) -> WeeklyReviewExtraction:
    """Route one weekly input to local parsing or one provider extraction call."""

    clean = weekly_review_input(text)
    if classify_weekly_review_input(clean) == "local":
        return extract_weekly_review_locally(clean)
    extracted = await ai.extract_weekly_review(clean, temporal_context)
    return validate_weekly_review_extraction(clean, extracted)
