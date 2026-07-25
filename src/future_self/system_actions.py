import re
from dataclasses import dataclass
from typing import Literal

SystemAction = Literal[
    "list_drafts",
    "discard_one_draft",
    "discard_selected_drafts",
    "discard_all_active_drafts",
    "show_last_saved",
    "archive_overdue_tasks",
    "cancel_system_action",
]


@dataclass(slots=True, frozen=True)
class SystemActionRoute:
    kind: Literal["none", "action", "confirm", "cancel", "pending"]
    action: SystemAction | None = None


class SystemActionRouter:
    """Deterministic destructive/system intent routing before content routing."""

    LAST_SAVED = (
        "напомни что ты сохранил",
        "что сохранилось последним",
        "что ты сохранил",
        "покажи последнюю сохраненную запись",
        "последняя сохраненная запись",
    )
    LIST_DRAFTS = (
        "покажи что не сохранено",
        "покажи черновики",
        "список черновиков",
        "покажи drafts",
    )
    DELETE_ALL = (
        "удали все черновики",
        "очисти черновики",
        "удали все несохраненные карточки",
        "убери этот мусор из drafts",
        "все несохраненные задачи удалить",
        "из черновиков хочу все удалить",
    )
    ARCHIVE_OVERDUE_TASKS = (
        "удали все неактуальные задачи",
        "удалить все неактуальные задачи",
        "убери все неактуальные задачи",
        "очисти неактуальные задачи",
        "убери просроченные задачи",
        "удали просроченные задачи",
        "удали все неактуальное",
        "убери все неактуальное",
    )
    KEEP_LAST = ("оставь только последнюю", "оставь только самый новый")
    DELETE_CONFIRM = (
        "да удалить",
        "удаляй все",
        "подтверждаю удаление",
        "да да удалить",
        "да все удалить",
    )

    def route(self, text: str, *, pending_action: str | None) -> SystemActionRoute:
        normalized = self._normalize(text)
        tokens = set(normalized.split())
        delete_requested = bool(
            tokens & {"удали", "удалить", "удаляй", "убери", "очисти"}
        ) and not any(
            marker in normalized for marker in ("не удал", "не убир", "не очищ", "не надо удал")
        )
        keep_requested = "оставь" in tokens and "не остав" not in normalized
        stale_marker = any(
            marker in normalized for marker in ("неактуальн", "не актуальн", "просроченн")
        )
        task_target = "задач" in normalized or "напомин" in normalized
        if pending_action:
            # A clearly named new target replaces the pending preview; it never
            # confirms the previous one. This supports corrections such as
            # "нет, я имел в виду ... черновики" without mutating either set.
            desired_action: SystemAction | None = None
            if keep_requested and self._contains(normalized, self.KEEP_LAST):
                desired_action = "discard_selected_drafts"
            elif (
                delete_requested
                and "чернов" in normalized
                and (
                    self._contains(normalized, self.DELETE_ALL)
                    or "все" in tokens
                    or "неактуальн" in normalized
                    or "не актуальн" in normalized
                )
            ):
                desired_action = "discard_all_active_drafts"
            elif (
                delete_requested
                and "чернов" not in normalized
                and (
                    self._contains(normalized, self.ARCHIVE_OVERDUE_TASKS)
                    or (task_target and stale_marker)
                )
            ):
                desired_action = "archive_overdue_tasks"

            current_target = "tasks" if pending_action == "archive_overdue_tasks" else "drafts"
            mentioned_target = (
                "drafts" if "чернов" in normalized else "tasks" if task_target else None
            )
            confirm_requested = self._contains(normalized, self.DELETE_CONFIRM) or (
                "да" in tokens and "удал" in normalized
            )
            if desired_action is not None:
                if desired_action != pending_action or not confirm_requested:
                    return SystemActionRoute(kind="action", action=desired_action)
            elif mentioned_target is not None and mentioned_target != current_target:
                return SystemActionRoute(kind="cancel", action="cancel_system_action")

            negated_delete = "не" in tokens and bool(
                tokens & {"удали", "удалить", "удаляй", "удалять", "убери", "убирай"}
            )
            negated_confirmation = "не подтверж" in normalized
            if (
                "отмена" in tokens
                or "нет" in tokens
                or "отмени" in tokens
                or negated_delete
                or negated_confirmation
            ):
                return SystemActionRoute(kind="cancel", action="cancel_system_action")
            if confirm_requested:
                return SystemActionRoute(kind="confirm")
            return SystemActionRoute(kind="pending")
        if self._contains(normalized, self.LAST_SAVED):
            return SystemActionRoute(kind="action", action="show_last_saved")
        if self._contains(normalized, self.LIST_DRAFTS):
            return SystemActionRoute(kind="action", action="list_drafts")
        if keep_requested and self._contains(normalized, self.KEEP_LAST):
            return SystemActionRoute(kind="action", action="discard_selected_drafts")
        # An explicit mention of drafts wins over the broader "stale" wording.
        # Otherwise "delete everything stale" means overdue active tasks and is
        # always followed by a separate confirmation in the handler.
        if delete_requested and (
            self._contains(normalized, self.ARCHIVE_OVERDUE_TASKS)
            or (task_target and stale_marker and "чернов" not in normalized)
        ):
            return SystemActionRoute(kind="action", action="archive_overdue_tasks")
        if delete_requested and (
            self._contains(normalized, self.DELETE_ALL)
            or any(
                marker in normalized
                for marker in ("все чернов", "всех чернов", "несохраненн", "мусор")
            )
            or (
                "чернов" in normalized
                and ("все" in tokens or "неактуальн" in normalized or "не актуальн" in normalized)
            )
        ):
            return SystemActionRoute(kind="action", action="discard_all_active_drafts")
        if delete_requested and self._contains(normalized, self.DELETE_CONFIRM):
            return SystemActionRoute(kind="pending")
        return SystemActionRoute(kind="none")

    @staticmethod
    def _contains(value: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in value for pattern in patterns)

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().replace("ё", "е")
        lowered = re.sub(r"[^a-zа-я0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()
