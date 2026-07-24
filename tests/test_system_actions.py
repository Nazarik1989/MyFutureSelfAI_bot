import pytest

from future_self.system_actions import SystemActionRouter


@pytest.mark.parametrize(
    "phrase",
    [
        "Удали все неактуальные задачи",
        "удалить все неактуальные задачи",
        "удали все не актуальные задачи",
        "удали все просроченные задачи",
        "убери все просроченные задачи",
        "убери просроченные задачи",
        "удали все неактуальное",
    ],
)
def test_stale_task_cleanup_phrases_are_deterministic_control_intents(phrase):
    route = SystemActionRouter().route(phrase, pending_action=None)
    assert (route.kind, route.action) == ("action", "archive_overdue_tasks")


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
def test_negated_cleanup_never_becomes_a_destructive_control_intent(phrase):
    assert SystemActionRouter().route(phrase, pending_action=None).kind == "none"


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


@pytest.mark.parametrize("phrase", ["не подтверждаю удаление", "не да, удалить"])
def test_negated_confirmation_cancels_instead_of_confirming(phrase):
    route = SystemActionRouter().route(phrase, pending_action="archive_overdue_tasks")
    assert route.kind != "confirm"
    assert route.kind == "cancel"
