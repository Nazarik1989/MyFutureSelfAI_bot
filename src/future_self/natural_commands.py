import re
from dataclasses import dataclass
from typing import Literal

NaturalAction = Literal[
    "show_drafts",
    "show_inbox",
    "show_last_saved",
    "show_profile",
    "show_today",
    "help",
]


@dataclass(slots=True, frozen=True)
class NaturalCommand:
    action: NaturalAction


class NaturalCommandRouter:
    """Deterministic read-only commands routed before all content processing."""

    PATTERNS: dict[NaturalAction, tuple[str, ...]] = {
        "show_drafts": (
            "выполни команду drafts",
            "покажи что в drafts",
            "покажи мои черновики",
            "покажи активные карточки",
            "покажи drafts",
            "что в drafts",
        ),
        "show_inbox": (
            "что у меня сохранено",
            "что у меня сохранилось",
            "что у меня в inbox",
            "что в inbox",
            "покажи сохраненные записи",
            "покажи inbox",
            "покажи мой inbox",
        ),
        "show_last_saved": (
            "что сохранилось",
            "что сохранилось последним",
            "что ты сохранил",
            "покажи последнюю сохраненную запись",
            "напомни что ты сохранил",
        ),
        "show_profile": (
            "покажи мой профиль",
            "покажи vision profile",
            "что в моем профиле",
        ),
        "show_today": (
            "покажи фокус дня",
            "что у меня сегодня",
            "мой план на сегодня",
        ),
        "help": (
            "покажи помощь",
            "какие есть команды",
            "какие у тебя есть команды",
            "какие у тебя команды",
            "какие команды у тебя есть",
            "что ты умеешь",
        ),
    }
    WRITE_INBOX = (
        "сохрани inbox",
        "сохрани инбокс",
        "сохрани в inbox",
        "сохрани в инбокс",
        "добавь в inbox",
        "добавь в инбокс",
        "запиши в inbox",
        "запиши в инбокс",
    )

    def route(self, text: str) -> NaturalCommand | None:
        normalized = self._normalize(text)
        if any(pattern in normalized for pattern in self.WRITE_INBOX):
            return None
        for action, patterns in self.PATTERNS.items():
            if any(pattern in normalized for pattern in patterns):
                return NaturalCommand(action)
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().replace("ё", "е")
        lowered = re.sub(r"[^a-zа-я0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()
