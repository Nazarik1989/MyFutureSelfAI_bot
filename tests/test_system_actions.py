import pytest

from future_self.system_actions import SystemActionRouter


@pytest.mark.parametrize(
    "phrase",
    [
        "Удали все неактуальные задачи",
        "удалить все неактуальные задачи",
        "удали все не актуальные задачи",
        "удалите устаревшие задачи",
        "очистить устаревшие задачи",
        "удали все просроченные задачи",
        "убери все просроченные задачи",
        "убери просроченные задачи",
        "Удалить все просроченные",
        "удали всё просроченное",
        "убрать все просроченные",
        "пожалуйста, удалите все просроченные",
        "удали все неактуальные",
        "удали все неактуальное",
        "удали все не актуальные",
        "удали все устаревшие",
        "убери всё устаревшее",
    ],
)
def test_stale_task_cleanup_phrases_are_deterministic_control_intents(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("action", "archive_overdue_tasks")


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Удали все просроченные", True),
        ("пожалуйста, удалите все просроченные", True),
        ("Я имел в виду: удали все неактуальные черновики", True),
        ("В будущем я хочу научиться удалять все неактуальные задачи вовремя", False),
        ("Я хочу научиться удалять все неактуальные задачи вовремя", False),
        ("Хочу уметь удалять все просроченные задачи без стресса", False),
        ("Отключи старые напоминания", True),
        ("Отмени все старые напоминания", True),
        ("Сними старые напоминания", True),
        ("Не присылай больше старые напоминания", True),
        ("Больше не присылай старые напоминания", True),
        ("Пожалуйста, больше не присылай старые напоминания", True),
        ("Я имел в виду: оставь только последнюю", True),
        ("Нет, оставь только последнюю", True),
        ("В будущем мне не нужны старые напоминания", False),
    ],
)
def test_explicit_cleanup_command_is_distinct_from_onboarding_narrative(phrase, expected):
    assert SystemActionRouter.is_explicit_cleanup_command(phrase) is expected


@pytest.mark.parametrize(
    "phrase",
    [
        "Я имел в виду: удали все неактуальные черновики.",
        "Я сейчас посмотрел черновики, там все не актуальны, удали, пожалуйста, все.",
    ],
)
def test_screenshot_draft_phrases_target_drafts_not_tasks(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("action", "discard_all_active_drafts")


@pytest.mark.parametrize(
    "phrase",
    [
        "не удаляй все черновики",
        "не удали все черновики",
        "не удаляй неактуальные задачи",
        "не удали все неактуальные задачи",
        "не убирай просроченные задачи",
        "не оставь только последнюю",
    ],
)
def test_command_shaped_negated_cleanup_is_quarantined(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("clarify", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "удали всё ненужное",
        "удали всё старое",
        "удалить всё с прошлой недели",
        "очисти всё",
        "удали просроченные файлы",
        "удали просроченную задачу",
        "убери неактуальное напоминание",
        "удали просроченное",
        "очисти inbox",
    ],
)
def test_ambiguous_or_unsupported_cleanup_is_quarantined_instead_of_falling_through(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("clarify", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "Как удалить все просроченные задачи?",
        "Можно ли удалить все просроченные задачи?",
        "Мы обсуждали, как удалить просроченные задачи",
        "Создай задачу удалить все просроченные файлы",
        "Напомни удалить просроченные письма",
        "Не забудь удалить просроченные письма",
        "Запиши фразу: удали все просроченные задачи",
    ],
)
def test_questions_and_capture_wrappers_do_not_start_cleanup(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("none", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "Я хочу создать задачу удалить все просроченные файлы",
        "Я хочу запланировать удалить просроченные задачи",
        "Запланируй удалить просроченные задачи",
        "Стоит ли удалить просроченные задачи?",
        "Хочу спросить, стоит ли удалить просроченные задачи",
        "Не уверен, надо ли удалить просроченные задачи",
        "Когда я скажу удалить просроченные задачи, не делай этого",
        "Проверь команду «удали все просроченные задачи»",
        "Пожалуйста, создай задачу удалить все просроченные файлы",
        "Можешь создать задачу удалить все просроченные файлы",
        "Добавь, пожалуйста, задачу удалить все просроченные файлы",
        "Мне нужно создать задачу удалить все просроченные файлы",
        "Хочу, чтобы ты создал задачу удалить все просроченные файлы",
        "Задача: удалить все просроченные файлы проекта",
        "Моя задача — удалить все просроченные файлы проекта",
        "Что думаешь, удалить все просроченные задачи?",
        "Я хочу спросить, удалять ли просроченные задачи",
    ],
)
def test_indirect_question_capture_and_meta_framing_never_starts_cleanup(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("none", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "Не планирую удалять просроченные задачи",
        "Я не собираюсь удалять просроченные задачи",
        "Я не собираюсь сегодня удалять просроченные задачи",
        "Я не просил тебя удалять просроченные задачи",
        "Я передумал удалять просроченные задачи",
        "Я не собирался удалять просроченные задачи",
    ],
)
def test_indirectly_negated_cleanup_is_quarantined(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("clarify", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "Удали все задачи кроме просроченных",
        "Удали все не просроченные задачи",
        "Удали все задачи за исключением просроченных",
        "Удали задачи, исключая устаревшие",
        "Удали все задачи, но просроченные оставь",
        "Удали все задачи, не трогая просроченные",
        "Удали все задачи, кроме самых просроченных",
    ],
)
def test_excluded_or_negated_stale_selector_is_quarantined(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("clarify", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "Удали неактуальные напоминания",
        "Удали все просроченные напоминания",
        "Отключи старые напоминания",
        "Старые напоминания больше не нужны",
        "Отмени все старые напоминания",
        "Сними старые напоминания",
        "Деактивируй старые напоминания",
        "Выруби старые напоминания",
        "Не присылай больше старые напоминания",
    ],
)
def test_reminder_cleanup_never_falls_back_to_task_deletion(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("clarify", None)


@pytest.mark.parametrize(
    ("phrase", "expected_kind"),
    [
        ("Стоит ли удалить просроченные задачи?", "pending"),
        ("Проверь команду «удали все просроченные задачи»", "pending"),
        ("Удали все задачи кроме просроченных", "clarify"),
        ("Удали неактуальные напоминания", "clarify"),
    ],
)
def test_unsafe_framing_never_confirms_or_retargets_pending_cleanup(phrase, expected_kind):
    route = SystemActionRouter().route(phrase, pending_action="archive_overdue_tasks")
    assert (route.kind, route.action) == (expected_kind, None)


@pytest.mark.parametrize(
    "phrase",
    [
        "удали все просроченные задачи и черновики",
        "очисти inbox и черновики",
        "убери неактуальные задачи из черновиков",
    ],
)
def test_mixed_cleanup_targets_are_quarantined(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("clarify", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "покажи задачи и напоминания",
        "открой задачи и напоминания",
        "покажи inbox и черновики",
    ],
)
def test_non_destructive_mixed_targets_pass_through_to_read_only_routing(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("none", None)


@pytest.mark.parametrize(
    "phrase",
    [
        "меню",
        "главное меню",
        "открой меню",
        "помощь",
        "покажи помощь",
        "какие у тебя команды",
        "что ты умеешь",
    ],
)
def test_navigation_commands_pass_through_an_existing_cleanup_preview(phrase):
    route = SystemActionRouter().route(phrase, pending_action="archive_overdue_tasks")
    assert (route.kind, route.action) == ("none", None)


@pytest.mark.parametrize(
    ("phrase", "pending", "expected"),
    [
        (
            "Я имел в виду: удали все неактуальные черновики",
            "archive_overdue_tasks",
            "discard_all_active_drafts",
        ),
        (
            "Нет, я имел в виду удали просроченные задачи",
            "discard_all_active_drafts",
            "archive_overdue_tasks",
        ),
        ("очисти черновики", "archive_overdue_tasks", "discard_all_active_drafts"),
        ("оставь только последнюю", "archive_overdue_tasks", "discard_selected_drafts"),
    ],
)
def test_explicit_cleanup_correction_replaces_pending_preview_without_confirming(
    phrase, pending, expected
):
    route = SystemActionRouter().route(phrase, pending_action=pending)
    assert (route.kind, route.action) == ("action", expected)


@pytest.mark.parametrize(
    ("phrase", "pending"),
    [
        ("да, удалить черновики", "archive_overdue_tasks"),
        ("да, удалить задачи", "discard_all_active_drafts"),
    ],
)
def test_conflicting_target_cannot_confirm_a_different_pending_cleanup(phrase, pending):
    route = SystemActionRouter().route(phrase, pending_action=pending)
    assert (route.kind, route.action) == ("cancel", "cancel_system_action")


@pytest.mark.parametrize("phrase", ["интернет", "планета", "интернет-магазин"])
def test_cancel_words_are_token_matched_not_substrings(phrase):
    assert SystemActionRouter().route(phrase, pending_action="archive_overdue_tasks").kind == (
        "pending"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "да, но в том числе мне нужно напоминать о самом главном, чтобы в суете я не забывал курс, по которому я могу стать лучше",
        "мне нужно напоминать о самом главном",
        "мне присылай напоминания почаще",
        "мне нужно напоминание",
        "мне подтверждать удаление?",
        "мне нужно удалить старый файл, но это не команда боту",
    ],
)
def test_mne_prefix_never_collides_with_destructive_negation_tokens(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("none", None)


@pytest.mark.parametrize(
    ("phrase", "pending", "expected"),
    [
        ("мне оставь только последнюю", None, ("action", "discard_selected_drafts")),
        ("не оставь только последнюю", None, ("clarify", None)),
        ("не нужно присылать старые напоминания", None, ("clarify", None)),
        ("не надо больше присылать старые напоминания", None, ("clarify", None)),
        ("мне не нужны старые напоминания", None, ("clarify", None)),
        ("не присылай старые напоминания", None, ("clarify", None)),
        ("больше не присылай старые напоминания", None, ("clarify", None)),
        (
            "не подтверждаю удаление",
            "archive_overdue_tasks",
            ("cancel", "cancel_system_action"),
        ),
        ("да, удалить", "archive_overdue_tasks", ("confirm", None)),
    ],
)
def test_destructive_controls_require_real_adjacent_token_sequences(phrase, pending, expected):
    route = SystemActionRouter().route(phrase, pending_action=pending)
    assert (route.kind, route.action) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "не нужно больше удалять просроченные задачи",
        "не надо сегодня удалять все старые черновики",
    ],
)
def test_token_safe_modal_negation_remains_fail_closed_with_bounded_fillers(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("clarify", None)


@pytest.mark.parametrize("phrase", ["не подтверждаю удаление", "не да, удалить"])
def test_negated_confirmation_cancels_instead_of_confirming(phrase):
    route = SystemActionRouter().route(phrase, pending_action="archive_overdue_tasks")
    assert route.kind != "confirm"
    assert route.kind == "cancel"


@pytest.mark.parametrize(
    "phrase",
    [
        "удали всё ненужное",
        "удали все просроченные задачи и черновики",
    ],
)
def test_ambiguous_new_cleanup_does_not_confirm_an_existing_preview(phrase):
    route = SystemActionRouter().route(phrase, pending_action="archive_overdue_tasks")
    assert (route.kind, route.action) == ("clarify", None)
