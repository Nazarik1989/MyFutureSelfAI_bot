import asyncio

import pytest
from pydantic import ValidationError

from future_self.schemas import (
    WeeklyReviewExtraction,
    WeeklyReviewReminderCandidate,
)
from future_self.weekly_review_extraction import (
    WEEKLY_REVIEW_LOCAL_MAX_INPUT_CHARS,
    WEEKLY_REVIEW_MAX_INPUT_CHARS,
    classify_weekly_review_input,
    explicit_weekly_focus,
    extract_weekly_review_input,
    extract_weekly_review_locally,
    validate_weekly_review_extraction,
    weekly_review_input,
)


class ExtractionAI:
    def __init__(self, result: WeeklyReviewExtraction):
        self.result = result
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.error: BaseException | None = None

    async def extract_weekly_review(
        self,
        text: str,
        temporal_context: dict[str, str],
    ) -> WeeklyReviewExtraction:
        self.calls.append((text, temporal_context))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize(
    ("text", "expected_focus"),
    [
        ("Спокойно закончить отчёт", "Спокойно закончить отчёт"),
        ("  Фокус: Спокойно закончить отчёт  ", "Спокойно закончить отчёт"),
        ("Завершить подготовку.", "Завершить подготовку."),
    ],
)
def test_short_simple_focus_is_extracted_locally_verbatim(text, expected_focus):
    assert classify_weekly_review_input(text) == "local"

    result = extract_weekly_review_locally(text)

    assert result == WeeklyReviewExtraction(focus=expected_focus)


@pytest.mark.parametrize(
    "text",
    [
        "x" * (WEEKLY_REVIEW_LOCAL_MAX_INPUT_CHARS + 1),
        "Фокус: Закончить отчёт\nПодход: по одному разделу",
        "Закончить отчёт. Затем подготовить презентацию.",
        "- Закончить отчёт",
        "В 15:05 позвонить Назару",
        "Каждый день делать короткую разминку",
    ],
)
def test_long_structured_or_scheduled_input_requires_provider(text):
    assert classify_weekly_review_input(text) == "provider"
    with pytest.raises(ValueError, match="requires provider"):
        extract_weekly_review_locally(text)


@pytest.mark.parametrize("text", ["", "   ", "x" * (WEEKLY_REVIEW_MAX_INPUT_CHARS + 1)])
def test_weekly_review_input_is_nonempty_and_bounded(text):
    with pytest.raises(ValueError, match="weekly review input"):
        weekly_review_input(text)


def test_exact_focus_label_has_priority_and_is_not_paraphrased():
    text = (
        "Хочу двигаться небольшими шагами.\n"
        "Фокус: спокойно закрывать подтверждённые напоминания дня.\n"
        "В 15:05 сказать Назару, что я люблю его"
    )
    untrusted = WeeklyReviewExtraction(
        focus="Закрывать все дела",
        approach="Двигаться небольшими шагами",
        reminder_candidates=[
            WeeklyReviewReminderCandidate(
                title="Сказать Назару, что я люблю его",
                schedule_wording="В 15:05",
                evidence="В 15:05 сказать Назару, что я люблю его",
            )
        ],
    )

    result = validate_weekly_review_extraction(text, untrusted)

    assert result.focus == "спокойно закрывать подтверждённые напоминания дня."
    assert untrusted.focus == "Закрывать все дела"
    assert explicit_weekly_focus(text) == result.focus


def test_multiple_explicit_focus_labels_are_rejected_before_provider_use():
    text = "Фокус: первый\nФокус: второй"

    with pytest.raises(ValueError, match="multiple explicit focus"):
        classify_weekly_review_input(text)


@pytest.mark.parametrize(
    "text",
    [
        "В 15:05 позвонить Назару. Потом ещё раз: В 15:05 позвонить Назару.",
        "В 15:05 позвонить Назару.",
    ],
)
def test_candidate_evidence_must_be_one_exact_unique_source_span(text):
    result = WeeklyReviewExtraction(
        focus="Закончить отчёт",
        reminder_candidates=[
            WeeklyReviewReminderCandidate(
                title="Позвонить Назару",
                schedule_wording="В 15:05",
                evidence="В 15:05 позвонить Назару",
            )
        ],
    )
    if text.endswith("Назару.") and text.count("В 15:05") == 1:
        result.reminder_candidates[0].evidence = "В 15:05 написать Назару"
        result.reminder_candidates[0].title = "Написать Назару"

    with pytest.raises(ValueError, match="exact unique input span"):
        validate_weekly_review_extraction(text, result)


def test_candidate_schedule_must_be_copied_exactly_from_evidence():
    with pytest.raises(ValidationError, match="schedule must be copied"):
        WeeklyReviewReminderCandidate(
            title="Позвонить Назару",
            schedule_wording="в три часа",
            evidence="В 15:05 позвонить Назару",
        )


def test_candidate_evidence_uniqueness_detects_overlapping_spans():
    extraction = WeeklyReviewExtraction(
        focus="Сказать Назару важное",
        reminder_candidates=[
            WeeklyReviewReminderCandidate(
                title="11:11",
                schedule_wording="11:11",
                evidence="11:11",
            )
        ],
    )

    with pytest.raises(ValueError, match="exact unique input span"):
        validate_weekly_review_extraction(
            "Напомни в 11:11:11 сказать Назару",
            extraction,
        )


def test_candidate_title_must_be_grounded_and_cannot_invent_recurrence():
    with pytest.raises(ValidationError, match="title must be copied"):
        WeeklyReviewReminderCandidate(
            title="Каждый день сказать Назару",
            schedule_wording="В 15:05",
            evidence="В 15:05 сказать Назару",
        )


def test_candidate_rejects_explicit_unsupported_weekly_recurrence():
    extraction = WeeklyReviewExtraction(
        focus="Поддерживать контакт",
        reminder_candidates=[
            WeeklyReviewReminderCandidate(
                title="позвонить врачу",
                schedule_wording="Каждый понедельник в 15:05",
                evidence="Каждый понедельник в 15:05 позвонить врачу",
            )
        ],
    )

    with pytest.raises(ValueError, match="one-shot or daily only"):
        validate_weekly_review_extraction(
            "Каждый понедельник в 15:05 позвонить врачу",
            extraction,
        )


@pytest.mark.parametrize(
    "schedule_wording",
    [
        "по понедельникам в 15:05",
        "каждые две недели в 15:05",
        "раз в месяц в 15:05",
        "ежемесячно в 15:05",
    ],
)
def test_candidate_rejects_every_unsupported_recurrence_family(
    schedule_wording: str,
):
    text = f"{schedule_wording} позвонить врачу"
    extraction = WeeklyReviewExtraction(
        focus="Поддерживать контакт",
        reminder_candidates=[
            WeeklyReviewReminderCandidate(
                title="позвонить врачу",
                schedule_wording=schedule_wording,
                evidence=text,
            )
        ],
    )

    with pytest.raises(ValueError, match="one-shot or daily only"):
        validate_weekly_review_extraction(text, extraction)


def test_candidate_accepts_supported_daily_recurrence_only_when_schedule_copies_it():
    text = "Каждый день в 15:05 позвонить врачу"
    accepted = validate_weekly_review_extraction(
        text,
        WeeklyReviewExtraction(
            focus="Поддерживать контакт",
            reminder_candidates=[
                WeeklyReviewReminderCandidate(
                    title="позвонить врачу",
                    schedule_wording="Каждый день в 15:05",
                    evidence=text,
                )
            ],
        ),
    )

    assert accepted.reminder_candidates[0].schedule_wording == "Каждый день в 15:05"

    with pytest.raises(ValueError, match="daily schedule must be copied"):
        validate_weekly_review_extraction(
            text,
            WeeklyReviewExtraction(
                focus="Поддерживать контакт",
                reminder_candidates=[
                    WeeklyReviewReminderCandidate(
                        title="позвонить врачу",
                        schedule_wording="в 15:05",
                        evidence=text,
                    )
                ],
            ),
        )


def test_schema_forbids_extra_fields_and_reused_evidence():
    candidate = {
        "title": "Позвонить Назару",
        "schedule_wording": "В 15:05",
        "evidence": "В 15:05 позвонить Назару",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WeeklyReviewExtraction(focus="Отчёт", secret="value")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="evidence must not be reused"):
        WeeklyReviewExtraction(
            focus="Отчёт",
            reminder_candidates=[candidate, candidate],
        )


async def test_orchestrator_skips_provider_for_local_focus():
    ai = ExtractionAI(WeeklyReviewExtraction(focus="provider must not win"))

    result = await extract_weekly_review_input(
        ai,
        "Фокус: Подготовить запуск",
        {"timezone": "Europe/Moscow"},
    )

    assert result.focus == "Подготовить запуск"
    assert ai.calls == []


async def test_orchestrator_calls_provider_once_and_revalidates_output():
    text = (
        "Буду двигаться постепенно.\n"
        "Фокус: Подготовить запуск без спешки.\n"
        "В 15:05 позвонить Назару"
    )
    context = {"timezone": "Europe/Moscow", "today_date": "2026-08-17"}
    ai = ExtractionAI(
        WeeklyReviewExtraction(
            focus="provider paraphrase",
            approach="Двигаться постепенно",
            small_steps=["Подготовить основу"],
            reminder_candidates=[
                WeeklyReviewReminderCandidate(
                    title="Позвонить Назару",
                    schedule_wording="В 15:05",
                    evidence="В 15:05 позвонить Назару",
                )
            ],
        )
    )

    result = await extract_weekly_review_input(ai, text, context)

    assert ai.calls == [(text, context)]
    assert result.focus == "Подготовить запуск без спешки."
    assert result.reminder_candidates[0].evidence == "В 15:05 позвонить Назару"


async def test_provider_error_is_propagated_without_retry_fallback_or_raw_logging(caplog):
    text = "Приватное длинное описание. Затем ещё одно приватное предложение."
    ai = ExtractionAI(WeeklyReviewExtraction(focus="fallback"))
    ai.error = RuntimeError("private provider response body")

    with pytest.raises(RuntimeError, match="private provider response body"):
        await extract_weekly_review_input(ai, text, {"timezone": "Europe/Moscow"})

    assert len(ai.calls) == 1
    assert text not in caplog.text
    assert "private provider response body" not in caplog.text


async def test_provider_cancellation_is_propagated_without_retry():
    text = "Длинное составное описание. Затем ещё одно предложение."
    ai = ExtractionAI(WeeklyReviewExtraction(focus="fallback"))
    ai.error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await extract_weekly_review_input(ai, text, {"timezone": "Europe/Moscow"})

    assert len(ai.calls) == 1
