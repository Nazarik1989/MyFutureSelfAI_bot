import re
from dataclasses import dataclass
from typing import Literal

SystemAction = Literal[
    "list_drafts",
    "discard_one_draft",
    "discard_selected_drafts",
    "discard_all_active_drafts",
    "show_last_saved",
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
    KEEP_LAST = ("оставь только последнюю", "оставь только самый новый")
    DELETE_CONFIRM = (
        "да удалить",
        "удаляй все",
        "подтверждаю удаление",
        "да да удалить",
        "да все удалить",
    )
    CANCEL = ("отмена", "отмени удаление", "не удаляй", "нет")

    def route(self, text: str, *, pending_action: str | None) -> SystemActionRoute:
        normalized = self._normalize(text)
        if pending_action:
            if self._contains(normalized, self.CANCEL):
                return SystemActionRoute(kind="cancel", action="cancel_system_action")
            if self._contains(normalized, self.DELETE_CONFIRM) or (
                "да" in normalized and "удал" in normalized
            ):
                return SystemActionRoute(kind="confirm")
            return SystemActionRoute(kind="pending")
        if self._contains(normalized, self.LAST_SAVED):
            return SystemActionRoute(kind="action", action="show_last_saved")
        if self._contains(normalized, self.LIST_DRAFTS):
            return SystemActionRoute(kind="action", action="list_drafts")
        if self._contains(normalized, self.KEEP_LAST):
            return SystemActionRoute(kind="action", action="discard_selected_drafts")
        if self._contains(normalized, self.DELETE_ALL) or (
            "удал" in normalized
            and any(
                marker in normalized
                for marker in ("все чернов", "всех чернов", "несохраненн", "мусор")
            )
        ):
            return SystemActionRoute(kind="action", action="discard_all_active_drafts")
        if self._contains(normalized, self.DELETE_CONFIRM):
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
