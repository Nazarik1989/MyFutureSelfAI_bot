from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class CommandSpec:
    command: str
    description: str


@dataclass(frozen=True, slots=True)
class NavigationAction:
    key: str
    label: str
    description: str
    handler: str | None = None
    example: str | None = None


@dataclass(frozen=True, slots=True)
class NavigationSection:
    key: str
    emoji: str
    label: str
    description: str
    actions: tuple[str, ...]


PUBLIC_COMMANDS = (
    CommandSpec("menu", "Главное меню"),
    CommandSpec("inbox", "Сохранённые идеи и заметки"),
    CommandSpec("vision", "Карта желаний"),
    CommandSpec("health", "Состояние и динамика"),
    CommandSpec("checkin", "Новый health check-in"),
    CommandSpec("doctor", "Поиск врача и подготовка к приёму"),
    CommandSpec("labs", "Результаты анализов"),
    CommandSpec("location", "Личная локация"),
    CommandSpec("help", "Помощь и примеры"),
)

# Existing commands remain supported, but intentionally stay outside Telegram's
# compact native menu. Keeping this explicit lets tests detect catalog drift.
ADVANCED_COMMANDS = frozenset(
    {
        "start",
        "onboarding",
        "back",
        "skip",
        "cancel",
        "profile",
        "goals",
        "drafts",
        "last_saved",
        "cleanup_drafts",
        "today",
        "evening",
        "health_edit",
        "health_delete",
        "health_reminder_on",
        "health_reminder_off",
        "doctor_prepare",
        "doctor_prepare_edit",
        "doctor_preparations",
        "doctor_prepare_show",
        "doctor_prepare_delete",
        "doctor_prepare_task",
        "doctor_find",
        "doctor_find_task",
    }
)

ACTIONS = {
    action.key: action
    for action in (
        NavigationAction("inbox", "Мои записи", "Последние сохранённые идеи и заметки.", "inbox"),
        NavigationAction(
            "drafts", "Черновики", "Preview-карточки до сохранения.", "drafts_command"
        ),
        NavigationAction(
            "last_saved", "Последнее сохранённое", "Последняя запись inbox.", "last_saved_command"
        ),
        NavigationAction("today", "Фокус дня", "Короткий план на сегодня.", "today"),
        NavigationAction(
            "reminder_guide",
            "Как создать напоминание",
            "Напиши задачу и время естественной фразой.",
            example="Напомни через 30 минут позвонить в клинику",
        ),
        NavigationAction(
            "vision", "Открыть карту", "Желания, визуализация и личные фото.", "vision_command"
        ),
        NavigationAction(
            "health", "Моё состояние", "Текущее состояние и недельная динамика.", "health_command"
        ),
        NavigationAction(
            "checkin",
            "Пройти check-in",
            "Шесть коротких вопросов о самочувствии.",
            "health_checkin_start",
        ),
        NavigationAction(
            "doctor_find", "Найти врача", "Официальные варианты по личной локации.", "doctor_find"
        ),
        NavigationAction(
            "doctor_prepare",
            "Подготовиться к приёму",
            "Фактическое резюме без диагнозов.",
            "doctor_prepare_start",
        ),
        NavigationAction(
            "doctor_preparations",
            "Мои подготовки",
            "Сохранённые подготовки к визиту.",
            "doctor_preparations",
        ),
        NavigationAction(
            "labs",
            "Анализы",
            "Безопасная локальная загрузка фото и PDF результатов.",
            "labs_command",
        ),
        NavigationAction(
            "doctor_task_guide",
            "Создать задачу на запись",
            "Задача создаётся с явно указанным временем reminder.",
            example="/doctor_find_task завтра в 10:00",
        ),
        NavigationAction(
            "location", "Моя медицинская локация", "Город и запасной маршрут.", "location_command"
        ),
        NavigationAction("profile", "Мой профиль", "Vision Profile и текущая локация.", "profile"),
        NavigationAction(
            "onboarding", "Настроить профиль", "Продолжить первоначальную настройку.", "start"
        ),
    )
}

SECTIONS = {
    section.key: section
    for section in (
        NavigationSection(
            "ideas",
            "📝",
            "Идеи и заметки",
            "Сохраняй мысли через preview и управляй черновиками.",
            ("inbox", "drafts", "last_saved"),
        ),
        NavigationSection(
            "tasks",
            "✅",
            "Задачи и напоминания",
            "Задачи с раздельными временем события и напоминания.",
            ("today", "reminder_guide"),
        ),
        NavigationSection(
            "vision",
            "🎯",
            "Карта желаний",
            "Желание → смысл → первый шаг → задача.",
            ("vision",),
        ),
        NavigationSection(
            "health",
            "❤️",
            "Здоровье",
            "Субъективная динамика самочувствия — не медицинский диагноз.",
            ("health", "checkin"),
        ),
        NavigationSection(
            "doctor",
            "🩺",
            "Врач",
            "Поиск врача и подготовка к приёму в одном месте.",
            (
                "doctor_find",
                "doctor_prepare",
                "doctor_preparations",
                "labs",
                "doctor_task_guide",
                "location",
            ),
        ),
        NavigationSection(
            "profile",
            "⚙️",
            "Профиль и настройки",
            "Профиль, timezone и личная локация.",
            ("profile", "location", "onboarding"),
        ),
    )
}

HELP_TOPICS = {
    "quick": (
        "Быстрый старт",
        "1. Сохрани идею.\n2. Создай задачу или reminder.\n3. Добавь желание.\n"
        "4. Сделай health check-in.\n5. При необходимости сохрани анализы или открой "
        "раздел «Врач».",
    ),
    "features": (
        "Что умеет бот",
        "\n".join(f"{item.emoji} {item.label} — {item.description}" for item in SECTIONS.values()),
    ),
    "examples": (
        "Примеры сообщений",
        "• Сохрани идею: записывать одну победу дня.\n"
        "• Напомни через 30 минут позвонить в клинику.\n"
        "• Покажи мои заметки.\n"
        "• Открой карту желаний.\n"
        "• Хочу сделать health check-in.\n"
        "• /labs — безопасно сохранить фото или PDF результатов анализов.",
    ),
    "commands": (
        "Основные команды",
        "\n".join(f"/{item.command} — {item.description}" for item in PUBLIC_COMMANDS),
    ),
    "privacy": (
        "Конфиденциальность",
        "Бот работает только в личном чате. Личные карточки, анализы, health-данные и "
        "Telegram ID не передаются в LLM и не включаются в диагностические логи.",
    ),
    "safety": (
        "Здоровье и безопасность",
        "Health Track и раздел анализов не ставят диагнозы, не расшифровывают показатели "
        "и не назначают лечение. При тревожных симптомах обращайся за срочной медицинской "
        "помощью.",
    ),
}


@dataclass(frozen=True, slots=True)
class FlowCapability:
    token: str
    owner_id: int
    chat_id: int
    flow: str


@dataclass(slots=True)
class _FlowSession:
    owner_id: int
    chat_id: int
    flow: str
    expires_at: float


class NavigationFlowStore:
    """Short-lived capabilities for explicit continue/exit decisions."""

    def __init__(self, *, ttl_seconds: int = 10 * 60, max_sessions: int = 64):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, _FlowSession] = {}
        self._lock = asyncio.Lock()

    async def issue(self, owner_id: int, chat_id: int, flow: str) -> str:
        async with self._lock:
            self._prune()
            for token in [
                key
                for key, value in self._sessions.items()
                if value.owner_id == owner_id and value.chat_id == chat_id
            ]:
                self._sessions.pop(token, None)
            while len(self._sessions) >= self.max_sessions:
                self._sessions.pop(next(iter(self._sessions)), None)
            token = secrets.token_urlsafe(9)
            self._sessions[token] = _FlowSession(
                owner_id=owner_id,
                chat_id=chat_id,
                flow=flow,
                expires_at=monotonic() + self.ttl_seconds,
            )
            return token

    async def claim(self, token: str, owner_id: int, chat_id: int) -> FlowCapability | None:
        async with self._lock:
            self._prune()
            session = self._sessions.get(token)
            if session is None or session.owner_id != owner_id or session.chat_id != chat_id:
                return None
            self._sessions.pop(token, None)
            return FlowCapability(token, owner_id, chat_id, session.flow)

    def _prune(self) -> None:
        now = monotonic()
        for token in [key for key, value in self._sessions.items() if value.expires_at <= now]:
            self._sessions.pop(token, None)


def validate_catalog() -> None:
    command_names = [item.command for item in PUBLIC_COMMANDS]
    if len(command_names) != len(set(command_names)):
        raise ValueError("Duplicate public navigation commands")
    used_actions: set[str] = set()
    for section in SECTIONS.values():
        if not section.actions:
            raise ValueError(f"Empty navigation section: {section.key}")
        for action in section.actions:
            if action not in ACTIONS:
                raise ValueError(f"Unknown navigation action: {action}")
            used_actions.add(action)
    if used_actions != set(ACTIONS):
        raise ValueError("Unreachable navigation action")


validate_catalog()
