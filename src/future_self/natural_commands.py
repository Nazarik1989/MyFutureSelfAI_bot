import re
from dataclasses import dataclass
from typing import Literal

INBOX_TARGETS = {"инбокс", "inbox", "инбок"}
SAVE_VERBS = {"сохрани", "сохраним"}
COMMAND_FILLERS = {"ну", "пожалуйста", "короче", "эээ"}


def normalize_command_text(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"[^a-zа-я0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _trim_command_fillers(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    for _ in range(2):
        if trimmed and trimmed[0] in COMMAND_FILLERS:
            trimmed.pop(0)
    if trimmed and trimmed[0] in COMMAND_FILLERS:
        return []
    if trimmed and trimmed[-1] == "пожалуйста":
        trimmed.pop()
    return trimmed


def is_save_inbox_command(text: str) -> bool:
    tokens = _trim_command_fillers(normalize_command_text(text).split())
    if len(tokens) == 2 and tokens[0] == "в" and tokens[1] in INBOX_TARGETS:
        return True
    if len(tokens) == 3 and tokens[0] == "это" and tokens[1] == "в" and tokens[2] in INBOX_TARGETS:
        return True
    if not tokens or tokens.pop(0) not in SAVE_VERBS:
        return False
    if tokens and tokens[0] == "пожалуйста":
        tokens.pop(0)
    if tokens and tokens[0] == "это":
        tokens.pop(0)
    if tokens and tokens[0] == "в":
        tokens.pop(0)
    return len(tokens) == 1 and tokens[0] in INBOX_TARGETS


def is_discard_inbox_command(text: str) -> bool:
    tokens = _trim_command_fillers(normalize_command_text(text).split())
    if len(tokens) < 3 or tokens[:2] not in (["не", "сохраняй"], ["не", "надо"]):
        return False

    if tokens[:2] == ["не", "сохраняй"]:
        remainder = tokens[2:]
    else:
        remainder = tokens[2:]
        if remainder and remainder[0] == "это":
            remainder.pop(0)
        if not remainder or remainder.pop(0) != "сохранять":
            return False

    if remainder and remainder[0] == "это":
        remainder.pop(0)
    if remainder and remainder[0] == "в":
        remainder.pop(0)
    return len(remainder) == 1 and remainder[0] in INBOX_TARGETS


NaturalAction = Literal[
    "menu",
    "show_drafts",
    "show_inbox",
    "show_last_saved",
    "show_profile",
    "show_today",
    "show_tasks",
    "show_overdue_tasks",
    "create_task",
    "show_vision",
    "show_records",
    "show_health",
    "show_labs",
    "prepare_doctor",
    "show_settings",
    "show_timezone",
    "show_collections",
    "show_spaces",
    "create_space",
    "invite_space_member",
    "show_space_invitations",
    "show_space_members",
    "help",
]


@dataclass(slots=True, frozen=True)
class NaturalCommand:
    action: NaturalAction


class NaturalCommandRouter:
    """Deterministic read-only commands routed before all content processing."""

    EXPLICIT_NAVIGATION_PREFIXES = (
        "как подготовиться",
        "как пользоваться",
        "как изменить",
        "как создать",
        "как открыть",
        "как найти",
        "покажи",
        "открой",
        "где",
    )
    SAFE_UI_TARGET_WORDS = frozenset(
        {
            "bot",
            "button",
            "calendar",
            "command",
            "feature",
            "menu",
            "section",
            "settings",
            "бот",
            "бота",
            "боте",
            "ботом",
            "календаре",
            "календарем",
            "календарь",
            "календаря",
            "кнопка",
            "кнопке",
            "кнопки",
            "кнопкой",
            "кнопку",
            "кнопок",
            "кнопках",
            "команда",
            "команде",
            "командой",
            "команду",
            "команды",
            "команд",
            "командах",
            "меню",
            "настройка",
            "настройке",
            "настройки",
            "настройкой",
            "настройку",
            "настроек",
            "настройках",
            "раздел",
            "раздела",
            "разделе",
            "разделом",
            "разделу",
            "разделы",
            "разделов",
            "разделах",
            "функцией",
            "функции",
            "функций",
            "функция",
            "функцию",
            "функциях",
        }
    )
    SAFE_UI_TARGET_FILLERS = frozenset(
        {
            "в",
            "во",
            "для",
            "из",
            "мой",
            "моя",
            "мое",
            "мои",
            "мою",
            "на",
            "твой",
            "твоя",
            "твое",
            "твои",
            "твою",
            "у",
            "эта",
            "эти",
            "это",
            "эту",
        }
    )
    # Capability names are matched as the whole target. This keeps ordinary
    # objects such as a document or presentation out of navigation while still
    # allowing a user to ask for a real bot capability in unfamiliar wording.
    CAPABILITY_TARGETS = frozenset(
        {
            "активные карточки",
            "анализы",
            "база знаний",
            "вечерний итог",
            "визуализация",
            "выполненные задачи",
            "желания и визуализация",
            "записи",
            "задачи",
            "задачи без срока",
            "задачи и напоминания",
            "здоровье",
            "инбокс",
            "карта желаний",
            "локация",
            "мои записи",
            "мои задачи",
            "мои подготовки",
            "мои пространства",
            "мое состояние",
            "мой профиль",
            "напоминания",
            "последнее сохраненное",
            "подготовка к врачу",
            "подготовка к приему",
            "помощь",
            "предстоящие задачи",
            "профиль",
            "просроченные задачи",
            "сегодня",
            "совместные пространства",
            "сохраненные записи",
            "фокус дня",
            "фокус на сегодня",
            "часовой пояс",
            "черновики",
            "check in",
            "drafts",
            "inbox",
        }
    )

    WORKSPACE_ACTIONS = frozenset(
        {
            "create_space",
            "invite_space_member",
            "show_space_invitations",
            "show_space_members",
        }
    )

    PATTERNS: dict[NaturalAction, tuple[str, ...]] = {
        "menu": (
            "меню",
            "главное меню",
            "открой меню",
            "покажи меню",
        ),
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
        "show_tasks": (
            "где мои задачи",
            "покажи мои задачи",
            "открой мои задачи",
            "покажи задачи и напоминания",
            "открой задачи и напоминания",
        ),
        "create_task": (
            "как создать задачу",
            "где создать задачу",
            "покажи как создать задачу",
        ),
        "show_overdue_tasks": (
            "покажи просроченные задачи",
            "какие задачи просрочены",
            "что у меня просрочено",
        ),
        "show_collections": (
            "где мои разделы",
            "покажи мои разделы",
            "открой мои разделы",
            "покажи разделы",
        ),
        "show_vision": (
            "где визуализация",
            "как открыть визуализацию",
            "покажи визуализацию",
            "открой визуализацию",
            "где карта желаний",
        ),
        "show_records": (
            "где мои записи",
            "покажи мои записи",
            "открой мои записи",
        ),
        "show_health": (
            "где здоровье",
            "покажи здоровье",
            "открой здоровье",
        ),
        "show_labs": (
            "где анализы",
            "покажи анализы",
            "открой анализы",
        ),
        "prepare_doctor": (
            "как подготовиться к врачу",
            "как подготовиться к приему",
            "как найти врача",
            "где подготовка к врачу",
        ),
        "show_settings": (
            "где настройки",
            "покажи настройки",
            "открой настройки",
        ),
        "show_timezone": (
            "как изменить часовой пояс",
            "где изменить часовой пояс",
        ),
        "show_spaces": (
            "где совместные пространства",
            "покажи мои пространства",
            "открой мои пространства",
            "покажи совместные пространства",
        ),
        "create_space": (
            "создай совместное пространство",
            "создать совместное пространство",
        ),
        "invite_space_member": (
            "пригласить участника",
            "пригласи участника",
        ),
        "show_space_invitations": (
            "покажи приглашения",
            "покажи приглашения в пространство",
        ),
        "show_space_members": (
            "покажи участников",
            "покажи участников пространства",
        ),
        "help": (
            "помощь",
            "покажи помощь",
            "какие есть команды",
            "какие у тебя есть команды",
            "какие у тебя команды",
            "какие команды у тебя есть",
            "что ты умеешь",
            "как пользоваться ботом",
        ),
    }
    WRITE_INBOX = ("добавь в inbox", "добавь в инбокс", "запиши в inbox", "запиши в инбокс")

    def __init__(self, *, enable_workspace_access: bool = False):
        self.enable_workspace_access = enable_workspace_access

    def route(self, text: str) -> NaturalCommand | None:
        normalized = self._normalize(text)
        normalized = " ".join(_trim_command_fillers(normalized.split()))
        if is_save_inbox_command(text) or any(
            pattern in normalized for pattern in self.WRITE_INBOX
        ):
            return None
        for action, patterns in self.PATTERNS.items():
            if action in self.WORKSPACE_ACTIONS and not self.enable_workspace_access:
                continue
            if normalized in patterns:
                return NaturalCommand(action)
        return None

    def is_explicit_navigation_request(self, text: str) -> bool:
        """Recognize only short, unmistakable navigation questions or commands."""

        normalized = self._normalize(text)
        normalized = " ".join(_trim_command_fillers(normalized.split()))
        tokens = normalized.split()
        if not normalized or len(tokens) > 10:
            return False
        if self.route(text) is not None:
            return True

        target = next(
            (
                normalized.removeprefix(f"{prefix} ")
                for prefix in self.EXPLICIT_NAVIGATION_PREFIXES
                if normalized.startswith(f"{prefix} ")
            ),
            None,
        )
        if not target:
            return False
        if target in self.CAPABILITY_TARGETS:
            return True
        target_tokens = set(target.split())
        allowed_tokens = self.SAFE_UI_TARGET_WORDS | self.SAFE_UI_TARGET_FILLERS
        return (
            not self.SAFE_UI_TARGET_WORDS.isdisjoint(target_tokens)
            and target_tokens <= allowed_tokens
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return normalize_command_text(text)
