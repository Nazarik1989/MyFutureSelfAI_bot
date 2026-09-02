from datetime import UTC, date, datetime, time

import pytest

from future_self.reminder_intent import (
    DAILY_AMBIGUOUS_FOLD,
    ConversationRecallIntent,
    ReminderIntentCode,
    ReminderIntentParser,
    ReminderIntentStatus,
    ReminderScheduleKind,
    ReminderTimezoneHint,
    ReminderTimezoneSource,
    calculate_daily_occurrence,
    classify_conversation_recall,
    first_daily_occurrence_utc,
    format_schedule_time,
    next_daily_occurrence_utc,
    parse_reminder_intent,
    present_schedule_time,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)  # 15:00 in Moscow


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("напомни плиз", ConversationRecallIntent.AMBIGUOUS),
        ("О чём мы сейчас говорили?", ConversationRecallIntent.RECALL),
        ("Что мы сейчас обсуждали?", ConversationRecallIntent.RECALL),
        (
            "напомни про наш с тобой разговор о ментальных тренировках, что именно мы обсуждали?",
            ConversationRecallIntent.RECALL,
        ),
        ("напомни, о чём мы говорили", ConversationRecallIntent.RECALL),
        ("напомни, что именно мы обсуждали", ConversationRecallIntent.RECALL),
        ("вспомни наш разговор", ConversationRecallIntent.RECALL),
        ("Ты помнишь наш разговор?", ConversationRecallIntent.RECALL),
        ("Помнишь, о чём мы говорили?", ConversationRecallIntent.RECALL),
        (
            "Мы недавно разговаривали о ментальных тренировках. Ты помнишь наш разговор?",
            ConversationRecallIntent.RECALL,
        ),
        ("Нова, ты помнишь наш разговор?", ConversationRecallIntent.RECALL),
        ("Nova, помнишь, о чём мы говорили?", ConversationRecallIntent.RECALL),
        (
            "напомни про наш разговор завтра в 19:00",
            ConversationRecallIntent.NONE,
        ),
        ("напомни про наш разговор сегодня", ConversationRecallIntent.NONE),
        ("напомни про наш разговор на сегодня", ConversationRecallIntent.NONE),
        ("напомни сегодня про наш разговор", ConversationRecallIntent.NONE),
        ("напомни про наш разговор сегодня в 19:00", ConversationRecallIntent.NONE),
        (
            "напомни про нашу встречу завтра в 19:00",
            ConversationRecallIntent.NONE,
        ),
        (
            "напомни о нашем разговоре 27 августа в 10:00",
            ConversationRecallIntent.NONE,
        ),
        ("напомни завтра позвонить врачу", ConversationRecallIntent.NONE),
        ("напомни в 19:00 про стрижку", ConversationRecallIntent.NONE),
        ("поставь напоминание", ConversationRecallIntent.NONE),
        ("напоминай каждый день", ConversationRecallIntent.NONE),
    ],
)
def test_conversation_recall_classifier_is_conservative(phrase, expected):
    assert classify_conversation_recall(phrase) is expected


@pytest.mark.parametrize(
    ("phrase", "expected_status"),
    [
        (
            "напомни про наш разговор завтра в 19:00",
            ReminderIntentStatus.COMPLETE,
        ),
        ("напомни про наш разговор сегодня", ReminderIntentStatus.NEEDS_TIME),
        ("напомни про наш разговор на сегодня", ReminderIntentStatus.NEEDS_TIME),
        ("напомни сегодня про наш разговор", ReminderIntentStatus.NEEDS_TIME),
        (
            "напомни про наш разговор сегодня в 19:00",
            ReminderIntentStatus.COMPLETE,
        ),
        (
            "напомни про нашу встречу завтра в 19:00",
            ReminderIntentStatus.COMPLETE,
        ),
        (
            "напомни о нашем разговоре 27 августа в 10:00",
            ReminderIntentStatus.COMPLETE,
        ),
        ("напомни завтра позвонить врачу", ReminderIntentStatus.NEEDS_TIME),
    ],
)
def test_temporal_conversation_wording_remains_a_reminder(phrase, expected_status):
    result = ReminderIntentParser(now_provider=lambda: NOW).parse(phrase, "Europe/Moscow")

    assert result.status is expected_status


@pytest.mark.parametrize(
    ("phrase", "status", "kind", "title", "local_date", "local_time"),
    [
        (
            "Напомни в 19:30 заполнить дневник",
            "needs_when",
            "once",
            "заполнить дневник",
            None,
            time(19, 30),
        ),
        (
            "Напомни мне завтра в 19:30 заполнить дневник",
            "complete",
            "once",
            "заполнить дневник",
            date(2026, 8, 11),
            time(19, 30),
        ),
        (
            "Напомни 15 августа в 19:30 позвонить",
            "complete",
            "once",
            "позвонить",
            date(2026, 8, 15),
            time(19, 30),
        ),
        (
            "Каждый день в 19:30 напоминай заполнить дневник",
            "complete",
            "daily",
            "заполнить дневник",
            None,
            time(19, 30),
        ),
        (
            "Ежедневно напоминай в 19.30 по МСК заполнить дневник",
            "complete",
            "daily",
            "заполнить дневник",
            None,
            time(19, 30),
        ),
        (
            "Поставь напоминание на завтра в 8:00",
            "needs_title",
            "once",
            None,
            date(2026, 8, 11),
            time(8),
        ),
    ],
)
def test_required_russian_phrases(
    phrase,
    status,
    kind,
    title,
    local_date,
    local_time,
):
    result = parse_reminder_intent(phrase, "Europe/Moscow", now=NOW)

    assert result.status == status
    assert result.schedule_kind == kind
    assert result.title == title
    assert result.local_date == local_date
    assert result.local_time == local_time
    assert result.timezone == "Europe/Moscow"


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни в 19:30 заполнить дневник",
        "Напомни в 19.30 заполнить дневник",
        "Напомни в 19.30по мск заполнить дневник",
        "Напомни в 19 30 заполнить дневник",
        "Напомни в 19 часов заполнить дневник",
        "Напомни в 19 часов 30 минут заполнить дневник",
    ],
)
def test_supported_clock_forms_are_deterministic(phrase):
    result = parse_reminder_intent(phrase, now=NOW)
    expected = time(19, 30) if "30" in phrase else time(19)

    assert result.status == ReminderIntentStatus.NEEDS_WHEN
    assert result.local_time == expected


def test_unspaced_moscow_timezone_is_explicit_and_wins_over_profile():
    result = parse_reminder_intent(
        "Напомни завтра в 19.30по мск заполнить дневник",
        "Europe/Saratov",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.COMPLETE
    assert result.timezone == "Europe/Moscow"
    assert result.timezone_source == ReminderTimezoneSource.EXPLICIT
    assert result.scheduled_for == datetime(2026, 8, 11, 16, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("phrase", "expected_title"),
    [
        (
            "Напомни завтра в 19:30 в магазин",
            "в магазин",
        ),
        (
            "Напомни завтра в 19:30 на тренировку",
            "на тренировку",
        ),
        (
            "Напомни завтра в 19:30 по работе позвонить",
            "по работе позвонить",
        ),
    ],
)
def test_semantic_leading_prepositions_are_preserved(phrase, expected_title):
    result = parse_reminder_intent(phrase, "Europe/Moscow", now=NOW)

    assert result.status == ReminderIntentStatus.COMPLETE
    assert result.title == expected_title


@pytest.mark.parametrize(
    ("phrase", "fragment", "timezone", "expected_title"),
    [
        (
            "Напомни завтра в 10:00 по Лондону созвониться с клиентом",
            "по Лондону",
            "Europe/London",
            "созвониться с клиентом",
        ),
        (
            "Каждый день в 20:30 по времени Тбилиси заполнить дневник",
            "по времени Тбилиси",
            "Asia/Tbilisi",
            "заполнить дневник",
        ),
        (
            "Напомни в 18:00 в часовом поясе Нью-Йорка проверить почту",
            "в часовом поясе Нью-Йорка",
            "America/New_York",
            "проверить почту",
        ),
    ],
)
def test_validated_timezone_hint_masks_only_proven_span(
    phrase,
    fragment,
    timezone,
    expected_title,
):
    result = parse_reminder_intent(
        phrase,
        "Europe/Moscow",
        now=NOW,
        timezone_hint=ReminderTimezoneHint(fragment=fragment, timezone=timezone),
    )

    assert result.status in {
        ReminderIntentStatus.COMPLETE,
        ReminderIntentStatus.NEEDS_WHEN,
    }
    assert result.title == expected_title
    assert result.timezone == timezone
    assert result.timezone_source is ReminderTimezoneSource.EXPLICIT
    assert fragment.casefold() not in result.title.casefold()


def test_validated_timezone_hint_uses_exact_span_without_masking_same_title_text():
    phrase = "Напомни завтра в 10:00 по Лондону обсудить фразу по Лондону"
    evidence = "по Лондону"
    start = phrase.index(evidence)

    result = parse_reminder_intent(
        phrase,
        "Europe/Moscow",
        now=NOW,
        timezone_hint=ReminderTimezoneHint(
            evidence,
            "Europe/London",
            (start, start + len(evidence)),
        ),
    )

    assert result.status is ReminderIntentStatus.COMPLETE
    assert result.timezone == "Europe/London"
    assert result.title == "обсудить фразу по Лондону"


def test_timezone_hint_rejects_evidence_that_does_not_match_its_exact_span():
    phrase = "Напомни завтра в 10:00 по Лондону созвониться"
    evidence = "по Лондону"

    result = parse_reminder_intent(
        phrase,
        "Europe/Moscow",
        now=NOW,
        timezone_hint=ReminderTimezoneHint(
            evidence,
            "Europe/London",
            (0, len(evidence)),
        ),
    )

    assert result.status is ReminderIntentStatus.INVALID
    assert result.error_code is ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE


def test_unresolved_timezone_hint_masks_marker_but_preserves_safe_parsed_fields():
    result = parse_reminder_intent(
        "Напомни завтра в 9:00 по Светогорску позвонить врачу",
        "Europe/Saratov",
        now=NOW,
        timezone_hint=ReminderTimezoneHint(fragment="по Светогорску"),
    )

    assert result.status is ReminderIntentStatus.COMPLETE
    assert result.title == "позвонить врачу"
    assert result.local_date == date(2026, 8, 11)
    assert result.local_time == time(9)
    assert result.timezone == "Europe/Saratov"
    assert result.timezone_source is ReminderTimezoneSource.PROFILE


def test_timezone_hint_without_exact_unique_evidence_is_fail_closed():
    absent = parse_reminder_intent(
        "Напомни завтра в 10:00 созвониться",
        now=NOW,
        timezone_hint=ReminderTimezoneHint("по Лондону", "Europe/London"),
    )
    repeated = parse_reminder_intent(
        "Напомни завтра в 10:00 по Лондону и по Лондону созвониться",
        now=NOW,
        timezone_hint=ReminderTimezoneHint("по Лондону", "Europe/London"),
    )

    assert absent.status is ReminderIntentStatus.INVALID
    assert absent.error_code is ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE
    assert repeated.status is ReminderIntentStatus.INVALID
    assert repeated.error_code is ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE


def test_model_timezone_hint_cannot_override_conflicting_exact_iana():
    result = parse_reminder_intent(
        "Напомни завтра в 10:00 по Europe/Berlin по Лондону созвониться",
        now=NOW,
        timezone_hint=ReminderTimezoneHint("по Лондону", "Europe/London"),
    )

    assert result.status is ReminderIntentStatus.INVALID
    assert result.error_code is ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE


@pytest.mark.parametrize(
    ("phrase", "expected_title"),
    [
        ("Напомни завтра в 19:30 по работе позвонить", "по работе позвонить"),
        ("Напомни завтра в 19:30 по проекту отправить отчёт", "по проекту отправить отчёт"),
    ],
)
def test_semantic_po_title_is_untouched_without_timezone_hint(phrase, expected_title):
    result = parse_reminder_intent(phrase, now=NOW)

    assert result.status is ReminderIntentStatus.COMPLETE
    assert result.title == expected_title


@pytest.mark.parametrize(
    ("phrase", "expected_timezone"),
    [
        (
            "Напомни завтра в 19.30по мск заполнить дневник благодарностей",
            "Europe/Moscow",
        ),
        (
            "Напомни завтра в 19:30 по МСК заполнить дневник благодарностей",
            "Europe/Moscow",
        ),
        (
            "Напомни завтра в 19:30 по Europe/Moscow заполнить дневник благодарностей",
            "Europe/Moscow",
        ),
        (
            "Напомни завтра в 19:30 по Asia/Tbilisi заполнить дневник благодарностей",
            "Asia/Tbilisi",
        ),
    ],
)
def test_exact_timezone_span_is_removed_without_stray_service_preposition(
    phrase,
    expected_timezone,
):
    result = parse_reminder_intent(phrase, "Europe/Saratov", now=NOW)

    assert result.status == ReminderIntentStatus.COMPLETE
    assert result.timezone == expected_timezone
    assert result.timezone_source == ReminderTimezoneSource.EXPLICIT
    assert result.title == "заполнить дневник благодарностей"
    assert not result.title.startswith("по ")


@pytest.mark.parametrize(
    "phrase",
    [
        "Пожалуйста, напомни завтра в 19:30 позвонить",
        "Напомни завтра в 19:30 позвонить, пожалуйста",
    ],
)
def test_polite_filler_is_removed_at_command_edges(phrase):
    result = parse_reminder_intent(phrase, "Europe/Moscow", now=NOW)

    assert result.status == ReminderIntentStatus.COMPLETE
    assert result.title == "позвонить"


def test_concatenated_polite_word_is_not_a_command_prefix():
    result = parse_reminder_intent(
        "пожалуйстанапомни завтра в 19:30 в магазин",
        "Europe/Moscow",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.NOT_REMINDER
    assert result.title is None


@pytest.mark.parametrize(
    "timezone_text",
    ["МСК", "мск", "по Москве", "московское время", "по московскому времени"],
)
def test_moscow_aliases_use_safe_canonical_timezone(timezone_text):
    result = parse_reminder_intent(
        f"Напомни завтра в 19:30 {timezone_text} заполнить дневник",
        "Europe/Saratov",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.COMPLETE
    assert result.timezone == "Europe/Moscow"
    assert result.timezone_source == ReminderTimezoneSource.EXPLICIT


def test_valid_iana_timezone_is_explicit_and_profile_is_default_only():
    explicit = parse_reminder_intent(
        "Напомни завтра в 09:00 Asia/Tbilisi позвонить",
        "Europe/Saratov",
        now=NOW,
    )
    profile = parse_reminder_intent(
        "Напомни завтра в 09:00 позвонить",
        "Europe/Saratov",
        now=NOW,
    )

    assert explicit.timezone == "Asia/Tbilisi"
    assert explicit.timezone_source == ReminderTimezoneSource.EXPLICIT
    assert profile.timezone == "Europe/Saratov"
    assert profile.timezone_source == ReminderTimezoneSource.PROFILE


def test_unknown_iana_timezone_is_a_safe_invalid_result():
    result = parse_reminder_intent(
        "Напомни завтра в 09:00 по Ocean/Atlantis позвонить",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE


@pytest.mark.parametrize(
    "phrase",
    [
        "Напомни завтра купить 1.20 литра воды",
        "Напомни завтра купить 1.2 литра воды",
        "Напомни завтра версию 2.10 установить",
    ],
)
def test_decimal_content_is_not_guessed_as_clock_time(phrase):
    result = parse_reminder_intent(phrase, now=NOW)

    assert result.status == ReminderIntentStatus.NEEDS_TIME
    assert result.local_time is None
    assert result.title is not None
    assert any(value in result.title for value in ("1.20", "1.2", "2.10"))


@pytest.mark.parametrize("reference", ["docs/API", "https://example.com/path"])
def test_non_timezone_slash_content_remains_in_title(reference):
    result = parse_reminder_intent(
        f"Напомни завтра в 19:30 открыть {reference}",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.COMPLETE
    assert result.title is not None
    assert reference in result.title


@pytest.mark.parametrize("minute", [60, 70, 1000])
def test_invalid_hour_word_minutes_do_not_fall_back_to_whole_hour(minute):
    result = parse_reminder_intent(
        f"Напомни завтра в 19 часов {minute} минут позвонить",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.INVALID_TIME
    assert result.local_time is None


def test_invalid_alternative_time_does_not_silently_select_first_clock():
    result = parse_reminder_intent(
        "Напомни завтра в 19:30 или 25:99 позвонить",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.INVALID_TIME


def test_three_spaced_numbers_are_not_guessed_as_clock_time():
    result = parse_reminder_intent(
        "Напомни завтра в 19 30 40 позвонить",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.INVALID_TIME
    assert result.local_time is None


def test_explicit_invalid_iana_prefix_is_case_insensitive():
    result = parse_reminder_intent(
        "Напомни завтра в 19:30 По Ocean/Atlantis позвонить",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.INVALID_EXPLICIT_TIMEZONE


def test_datetime_boundary_returns_typed_invalid_instead_of_raising():
    result = parse_reminder_intent(
        "Напомни 31 декабря 9999 в 23:59 America/New_York позвонить",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.NONEXISTENT_LOCAL_TIME


@pytest.mark.parametrize(
    "phrase",
    [
        "Эта песня напомнила мне Москву",
        "Меня это напомнило о школе",
        "Каждый день я заполняю дневник в 19:30",
        "Каждый день в 19:30 я заполняю дневник",
        "Каждый день в 19:30 по дороге домой я слушаю музыку",
        "Каждый день в 19:30 тренировка",
        "Каждый день задача занимает много времени",
        "Время летит, а Москва меняется",
        "Москва напомнила о себе дождём",
    ],
)
def test_ordinary_narratives_are_not_intercepted(phrase):
    result = parse_reminder_intent(phrase, now=NOW)

    assert result.status == ReminderIntentStatus.NOT_REMINDER
    assert result.schedule_kind is None
    assert result.title is None
    assert result.timezone is None


def test_spaced_digits_need_an_unambiguous_time_preposition():
    result = parse_reminder_intent("Напомни 19 30 страниц прочитать завтра", now=NOW)

    assert result.status == ReminderIntentStatus.NEEDS_TIME
    assert result.local_time is None


def test_today_tomorrow_and_explicit_dates_use_reminder_timezone():
    today = parse_reminder_intent(
        "Напомни сегодня в 20:00 проверить духовку",
        "Europe/Moscow",
        now=NOW,
    )
    tomorrow = parse_reminder_intent(
        "Напомни завтра в 20:00 проверить духовку",
        "Europe/Moscow",
        now=NOW,
    )
    numeric = parse_reminder_intent(
        "Напомни 18.08.2026 в 20:00 проверить духовку",
        "Europe/Moscow",
        now=NOW,
    )

    assert today.local_date == date(2026, 8, 10)
    assert tomorrow.local_date == date(2026, 8, 11)
    assert numeric.local_date == date(2026, 8, 18)
    assert all(
        result.status == ReminderIntentStatus.COMPLETE for result in (today, tomorrow, numeric)
    )


def test_daily_without_time_and_once_without_date_return_typed_needs():
    daily = parse_reminder_intent("Ежедневно напоминай размяться", now=NOW)
    once = parse_reminder_intent("Напомни в 19:30 размяться", now=NOW)

    assert daily.status == ReminderIntentStatus.NEEDS_TIME
    assert daily.schedule_kind == ReminderScheduleKind.DAILY
    assert daily.title == "размяться"
    assert once.status == ReminderIntentStatus.NEEDS_WHEN
    assert once.schedule_kind == ReminderScheduleKind.ONCE
    assert once.local_time == time(19, 30)


def test_partial_results_merge_without_losing_recognized_fields():
    parser = ReminderIntentParser(now_provider=lambda: NOW)
    initial = parser.parse("Напомни заполнить дневник", "Europe/Saratov")
    with_time = parser.parse("В 19.30 мск", "Europe/Saratov", previous=initial)
    daily = parser.parse("Каждый день", "Europe/Saratov", previous=with_time)

    assert initial.title == "заполнить дневник"
    assert initial.status in {
        ReminderIntentStatus.NEEDS_WHEN,
        ReminderIntentStatus.NEEDS_TIME,
    }
    assert with_time.status == ReminderIntentStatus.NEEDS_WHEN
    assert with_time.title == initial.title
    assert with_time.local_time == time(19, 30)
    assert with_time.timezone == "Europe/Moscow"
    assert with_time.timezone_source == ReminderTimezoneSource.EXPLICIT
    assert daily.status == ReminderIntentStatus.COMPLETE
    assert daily.schedule_kind == ReminderScheduleKind.DAILY
    assert daily.title == initial.title
    assert daily.local_time == time(19, 30)
    assert daily.timezone == "Europe/Moscow"
    assert daily.timezone_source == ReminderTimezoneSource.EXPLICIT


def test_bare_time_is_accepted_only_in_continuation_mode():
    ordinary = parse_reminder_intent("В 19.30 мск", now=NOW)
    continuation = parse_reminder_intent("В 19.30 мск", now=NOW, continuation=True)

    assert ordinary.status == ReminderIntentStatus.NOT_REMINDER
    assert continuation.status == ReminderIntentStatus.NEEDS_TITLE
    assert continuation.local_time == time(19, 30)
    assert continuation.timezone == "Europe/Moscow"
    assert continuation.timezone_source == ReminderTimezoneSource.EXPLICIT


def test_past_one_shot_is_invalid_and_never_rolls_forward_silently():
    result = parse_reminder_intent(
        "Напомни сегодня в 14:00 позвонить",
        "Europe/Moscow",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.PAST_ONCE
    assert result.local_date == date(2026, 8, 10)
    assert result.local_time == time(14)


def test_past_named_date_without_year_stays_in_current_year_and_is_invalid():
    result = parse_reminder_intent(
        "Напомни 15 августа в 19:00 позвонить",
        "Europe/Moscow",
        now=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.PAST_ONCE
    assert result.local_date == date(2026, 8, 15)
    assert result.local_time == time(19)


def test_past_named_date_is_invalid_even_before_time_is_supplied():
    result = parse_reminder_intent(
        "Напомни 15 августа позвонить",
        "Europe/Moscow",
        now=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.PAST_ONCE
    assert result.local_date == date(2026, 8, 15)


@pytest.mark.parametrize(
    ("phrase", "code"),
    [
        ("Напомни 31 февраля в 19:00 позвонить", "invalid_date"),
        ("Напомни завтра в 25:70 позвонить", "invalid_time"),
        ("Напомни сегодня завтра в 19:00 позвонить", "ambiguous_date"),
        ("Напомни завтра в 18:00 или 19:00 позвонить", "ambiguous_time"),
        ("Каждый день завтра в 19:00 напоминай позвонить", "conflicting_schedule"),
    ],
)
def test_unsafe_or_ambiguous_temporal_input_is_invalid(phrase, code):
    result = parse_reminder_intent(phrase, now=NOW)

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == code


def test_nonexistent_one_shot_time_is_invalid():
    result = parse_reminder_intent(
        "Напомни 8 марта 2026 в 2:30 America/New_York позвонить",
        now=datetime(2026, 3, 1, tzinfo=UTC),
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert result.error_code == ReminderIntentCode.NONEXISTENT_LOCAL_TIME


def test_text_and_voice_transcript_have_identical_typed_parse():
    parser = ReminderIntentParser(now_provider=lambda: NOW)
    phrase = "Ежедневно напоминай в 19.30 по МСК заполнить дневник"

    text_result = parser.parse(phrase, "Europe/Saratov")
    voice_transcript_result = parser.parse(phrase, "Europe/Saratov")

    assert text_result == voice_transcript_result


def test_invalid_result_does_not_retain_raw_private_input():
    sentinel = "PRIVATE_SENTINEL_7bbfc"
    result = parse_reminder_intent(
        f"Напомни завтра в 19:00 по Secret/{sentinel} позвонить",
        now=NOW,
    )

    assert result.status == ReminderIntentStatus.INVALID
    assert sentinel not in repr(result)
    assert not hasattr(result, "raw_text")
    assert not hasattr(result, "original_expression")


def test_first_daily_occurrence_uses_today_only_while_still_future():
    before = first_daily_occurrence_utc(
        time(19, 30),
        "Europe/Moscow",
        now=datetime(2026, 8, 10, 15, 0, tzinfo=UTC),
    )
    at_due = first_daily_occurrence_utc(
        time(19, 30),
        "Europe/Moscow",
        now=datetime(2026, 8, 10, 16, 30, tzinfo=UTC),
    )
    after = first_daily_occurrence_utc(
        time(19, 30),
        "Europe/Moscow",
        now=datetime(2026, 8, 10, 18, 0, tzinfo=UTC),
    )

    assert before == datetime(2026, 8, 10, 16, 30, tzinfo=UTC)
    assert at_due == datetime(2026, 8, 11, 16, 30, tzinfo=UTC)
    assert after == datetime(2026, 8, 11, 16, 30, tzinfo=UTC)


def test_daily_occurrence_can_cross_utc_midnight_without_changing_local_date():
    occurrence = calculate_daily_occurrence(
        date(2026, 8, 11),
        time(0, 30),
        "Europe/Saratov",
    )

    assert occurrence.local_date == date(2026, 8, 11)
    assert occurrence.scheduled_for == datetime(2026, 8, 10, 20, 30, tzinfo=UTC)


def test_ambiguous_daily_time_uses_documented_fold_zero():
    occurrence = calculate_daily_occurrence(
        date(2026, 11, 1),
        time(1, 30),
        "America/New_York",
    )

    assert DAILY_AMBIGUOUS_FOLD == 0
    assert occurrence.local_date == date(2026, 11, 1)
    assert occurrence.fold == 0
    assert occurrence.scheduled_for == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)


def test_nonexistent_daily_time_skips_only_that_local_calendar_date():
    occurrence = calculate_daily_occurrence(
        date(2026, 3, 8),
        time(2, 30),
        "America/New_York",
    )

    assert occurrence.local_date == date(2026, 3, 9)
    assert occurrence.scheduled_for == datetime(2026, 3, 9, 6, 30, tzinfo=UTC)


def test_next_daily_occurrence_uses_local_calendar_not_fixed_24_hours():
    next_occurrence = next_daily_occurrence_utc(
        datetime(2026, 3, 7, 7, 30, tzinfo=UTC),
        time(2, 30),
        "America/New_York",
    )

    assert next_occurrence == datetime(2026, 3, 9, 6, 30, tzinfo=UTC)


def test_moscow_schedule_presentation_converts_for_saratov_viewer():
    expected = "19:30 по Москве — 20:30 по Саратову"

    assert (
        format_schedule_time(
            time(19, 30),
            "Europe/Moscow",
            "Europe/Saratov",
            date(2026, 8, 11),
        )
        == expected
    )
    assert (
        present_schedule_time(
            time(19, 30),
            "Europe/Moscow",
            "Europe/Saratov",
            date(2026, 8, 11),
        )
        == expected
    )


def test_same_timezone_presentation_is_not_duplicated():
    rendered = format_schedule_time(
        time(19, 30),
        "Europe/Moscow",
        "Europe/Moscow",
        date(2026, 8, 11),
    )

    assert rendered == "19:30 (Europe/Moscow)"
    assert "—" not in rendered
