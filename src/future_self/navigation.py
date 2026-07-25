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
    CommandSpec("tasks", "Задачи и напоминания"),
    CommandSpec("collections", "Мои разделы"),
    CommandSpec("vision", "Карта желаний"),
    CommandSpec("health", "Состояние и динамика"),
    CommandSpec("checkin", "Новый health check-in"),
    CommandSpec("doctor", "Поиск врача и подготовка к приёму"),
    CommandSpec("labs", "Результаты анализов"),
    CommandSpec("location", "Личная локация"),
    CommandSpec("help", "Помощь и примеры"),
)

WORKSPACE_PUBLIC_COMMANDS = (CommandSpec("spaces", "Совместные пространства"),)

KNOWLEDGE_PUBLIC_COMMANDS = (CommandSpec("knowledge", "База знаний"),)

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

WORKSPACE_ADVANCED_COMMANDS = frozenset({"workspaces"})

KNOWLEDGE_ADVANCED_COMMANDS = frozenset({"capture"})

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
        NavigationAction(
            "task_today", "Сегодня", "Задачи на локальную календарную дату.", "task_today"
        ),
        NavigationAction(
            "task_upcoming", "Предстоящие", "Задачи после сегодняшнего дня.", "task_upcoming"
        ),
        NavigationAction(
            "task_overdue", "Просроченные", "Активные задачи с прошедшим сроком.", "task_overdue"
        ),
        NavigationAction("task_no_due", "Без срока", "Активные задачи без срока.", "task_no_due"),
        NavigationAction("task_completed", "Выполненные", "Завершённые задачи.", "task_completed"),
        NavigationAction(
            "task_create",
            "Создать задачу",
            "Создание через общий preview и confirm.",
            "task_create",
        ),
        NavigationAction(
            "task_reminder_guide",
            "Как работают напоминания",
            "Срок, напоминание, перенос и безопасная отмена доставки.",
            "task_reminder_guide",
        ),
        NavigationAction(
            "collections",
            "Открыть мои разделы",
            "Темы, проекты и списки поверх существующих записей Inbox и Task Hub.",
            "collections_command",
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

WORKSPACE_ACTIONS = {
    "spaces": NavigationAction(
        "spaces",
        "Открыть пространства",
        "Участники, приглашения и проекты в защищённом общем контуре.",
        "spaces_command",
    )
}

KNOWLEDGE_ACTIONS = {
    "knowledge": NavigationAction(
        "knowledge",
        "Открыть базу знаний",
        "Личные и доступные совместные материалы с безопасными статусами обработки.",
        "knowledge_command",
    ),
    "capture": NavigationAction(
        "capture",
        "Добавить материал",
        "Явный Capture текста, документа, изображения или ссылки с preview и подтверждением.",
        "capture_command",
    ),
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
            (
                "task_today",
                "task_upcoming",
                "task_overdue",
                "task_no_due",
                "task_completed",
                "task_create",
                "task_reminder_guide",
            ),
        ),
        NavigationSection(
            "collections",
            "🗂",
            "Мои разделы",
            "Сферы жизни, проекты и списки без дублирования записей.",
            ("collections",),
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

WORKSPACE_SECTION = NavigationSection(
    "spaces",
    "🤝",
    "Совместные пространства",
    "Отдельный защищённый контур с участниками, ролями, приглашениями и проектами.",
    ("spaces",),
)


def _knowledge_section(enable_capture: bool) -> NavigationSection:
    return NavigationSection(
        "knowledge",
        "📚",
        "База знаний",
        "Источники хранятся отдельно от Inbox и не используются LLM-контекстом.",
        ("knowledge", "capture") if enable_capture else ("knowledge",),
    )


HELP_TOPICS = {
    "quick": (
        "🚀 Быстрый старт",
        "1. Отправь мысль обычными словами.\n"
        "2. Проверь preview: тип, название, дата и следующий шаг.\n"
        "3. Нажми «Сохранить», «Редактировать» или «Не сохранять».\n"
        "4. Записи ищи в /inbox, задачи и сроки — в /tasks.\n\n"
        "Команды помнить не нужно: /menu открывает все разделы.",
    ),
    "features": (
        "🧭 Что умеет бот",
        "\n".join(f"{item.emoji} {item.label} — {item.description}" for item in SECTIONS.values()),
    ),
    "voice": (
        "🎙 Голосовое управление",
        "Отправь голосовое так же, как обычное сообщение. Бот сначала проверяет, не является "
        "ли распознанная речь командой для существующих записей, задач или черновиков, и только "
        "потом предлагает создать новую карточку.\n\n"
        "Примеры: «покажи мои задачи», «какие задачи просрочены», «сохрани в inbox», "
        "«удали все просроченные», «убери все неактуальные задачи». Массовое действие всегда "
        "показывает список и просит отдельное подтверждение. Неоднозначная команда удаления "
        "не превращается в новую запись.",
    ),
    "drafts": (
        "📝 Идеи, заметки и черновики",
        "Новое сообщение сначала становится preview, а не сразу постоянной записью. Проверь "
        "заголовок и тип, затем сохрани или отредактируй. /drafts показывает несохранённые "
        "preview, /inbox — сохранённые записи с карточками, кнопкой «В корзину», восстановлением "
        "и безопасной очисткой ошибочно сохранённых команд. Точная повторная запись в коротком "
        "окне не создаёт дубль. /cleanup_drafts удаляет только активные черновики после "
        "подтверждения. Сроки и состояние сохранённых задач находятся в /tasks.",
    ),
    "tasks": (
        "✅ Задачи и напоминания",
        "/tasks разделяет задачи на «Сегодня», «Предстоящие», «Просроченные», «Без срока» и "
        "«Выполненные». В карточке можно завершить, перенести, изменить или выключить "
        "напоминание и переместить задачу в восстанавливаемую корзину. Кнопка «Очистить "
        "просроченные» сначала показывает точный список и ничего не меняет без отдельного "
        "подтверждения.\n\n"
        "Просроченная задача не считается выполненной автоматически. Сильно опоздавшее "
        "напоминание скрывается из Inbox, но его история и сама задача остаются в Task Hub.",
    ),
    "collections": (
        "🗂 Мои разделы",
        "Разделы собирают уже существующие записи и задачи по темам, проектам и спискам без "
        "копирования содержимого. Открой /collections, выбери раздел или создай новый обычной "
        "фразой. Удаление связи не удаляет исходную запись Inbox.",
    ),
    "vision": (
        "🎯 Карта желаний",
        "В /vision желание проходит понятные шаги: категория → формулировка → зачем это важно → "
        "срок → первый шаг → подтверждение. Можно добавить личное фото и создать задачу из "
        "первого шага. Карточка не публикуется автоматически.",
    ),
    "health": (
        "❤️ Состояние и check-in",
        "/checkin задаёт короткие вопросы об энергии, сне, настроении, стрессе, теле и симптомах. "
        "/health показывает сохранённую динамику. Ответы можно исправлять и удалять; этот раздел "
        "не ставит диагнозы и не назначает лечение.",
    ),
    "doctor": (
        "🩺 Врач и анализы",
        "/doctor объединяет официальный поиск врача по личной локации, подготовку фактического "
        "резюме к приёму и сохранение результатов анализов. /location задаёт город или маршрут. "
        "/labs хранит загруженные фото/PDF, но не интерпретирует показатели как диагноз.",
    ),
    "registration": (
        "👤 Регистрация и профиль",
        "/start запускает или продолжает настройку профиля. Отвечай свободно — длинные и "
        "многоабзацные ответы допустимы. После каждого ответа бот сохраняет шаг и задаёт "
        "следующий вопрос; /back возвращает назад, /skip пропускает необязательное. В конце "
        "появится резюме для подтверждения или исправления.",
    ),
    "examples": (
        "💬 Примеры сообщений",
        "Inbox — сохранённые записи; /drafts — ещё не подтверждённые preview; /tasks — "
        "состояние, сроки и напоминания задач.\n\n"
        "• Сохрани идею: записывать одну победу дня.\n"
        "• Напомни через 30 минут позвонить в клинику.\n"
        "• Покажи мои задачи.\n"
        "• Какие задачи просрочены?\n"
        "• Удали все неактуальные задачи.\n"
        "• Создай проект Наз и Войд.\n"
        "• Добавь в Покупки чай, сахар и цемент.\n"
        "• Открой карту желаний.\n"
        "• Хочу сделать health check-in.\n"
        "• /labs — безопасно сохранить фото или PDF результатов анализов.",
    ),
    "commands": (
        "⌨️ Основные команды",
        "\n".join(f"/{item.command} — {item.description}" for item in PUBLIC_COMMANDS),
    ),
    "privacy": (
        "🔒 Конфиденциальность",
        "Бот работает только в личном чате. Личные карточки, анализы, health-данные и "
        "содержимое пользовательских разделов и Telegram ID не включаются в диагностические "
        "логи. Явные команды разделов обрабатываются без LLM.",
    ),
    "safety": (
        "🛟 Здоровье и безопасность",
        "Health Track и раздел анализов не ставят диагнозы, не расшифровывают показатели "
        "и не назначают лечение. При тревожных симптомах обращайся за срочной медицинской "
        "помощью.",
    ),
    "troubleshooting": (
        "🧰 Если бот не понял",
        "Сформулируй действие и объект прямо: «покажи просроченные задачи», «удали все "
        "черновики», «сохрани эту идею». Перед массовым удалением бот обязан показать "
        "количество и запросить подтверждение. Если активен пошаговый сценарий, закончи его "
        "ответом или используй /cancel. Главное меню — /menu, эта справка — /help.",
    ),
}


def public_commands(
    enable_workspace_access: bool = False,
    enable_knowledge_hub: bool = False,
) -> tuple[CommandSpec, ...]:
    """Return the native command catalog without exposing disabled features."""

    if not enable_workspace_access and not enable_knowledge_hub:
        return PUBLIC_COMMANDS
    result: list[CommandSpec] = []
    for item in PUBLIC_COMMANDS:
        result.append(item)
        if item.command == "collections":
            if enable_workspace_access:
                result.extend(WORKSPACE_PUBLIC_COMMANDS)
            if enable_knowledge_hub:
                result.extend(KNOWLEDGE_PUBLIC_COMMANDS)
    return tuple(result)


def advanced_commands(
    enable_workspace_access: bool = False,
    enable_knowledge_capture: bool = False,
) -> frozenset[str]:
    result = ADVANCED_COMMANDS
    if enable_workspace_access:
        result |= WORKSPACE_ADVANCED_COMMANDS
    if enable_knowledge_capture:
        result |= KNOWLEDGE_ADVANCED_COMMANDS
    return result


def navigation_actions(
    enable_workspace_access: bool = False,
    enable_knowledge_hub: bool = False,
    enable_knowledge_capture: bool = False,
) -> dict[str, NavigationAction]:
    result = dict(ACTIONS)
    if enable_workspace_access:
        result.update(WORKSPACE_ACTIONS)
    if enable_knowledge_hub:
        result["knowledge"] = KNOWLEDGE_ACTIONS["knowledge"]
        if enable_knowledge_capture:
            result["capture"] = KNOWLEDGE_ACTIONS["capture"]
    return result


def navigation_sections(
    enable_workspace_access: bool = False,
    enable_knowledge_hub: bool = False,
    enable_knowledge_capture: bool = False,
) -> dict[str, NavigationSection]:
    if not enable_workspace_access and not enable_knowledge_hub:
        return SECTIONS
    result: dict[str, NavigationSection] = {}
    for key, section in SECTIONS.items():
        result[key] = section
        if key == "collections":
            if enable_workspace_access:
                result[WORKSPACE_SECTION.key] = WORKSPACE_SECTION
            if enable_knowledge_hub:
                knowledge = _knowledge_section(enable_knowledge_capture)
                result[knowledge.key] = knowledge
    return result


def help_topics(
    enable_workspace_access: bool = False,
    enable_knowledge_hub: bool = False,
    enable_knowledge_capture: bool = False,
    enable_voice: bool = True,
    enable_task_reminders: bool = True,
) -> dict[str, tuple[str, str]]:
    """Build flag-aware help text while keeping the PR #22 constants stable."""

    topics = dict(HELP_TOPICS)
    if not enable_voice:
        topics.pop("voice", None)
    if not enable_task_reminders:
        topics["tasks"] = (
            "✅ Задачи",
            "/tasks разделяет задачи на «Сегодня», «Предстоящие», «Просроченные», «Без срока» "
            "и «Выполненные». Доставка Telegram-напоминаний сейчас отключена настройкой, но "
            "сроки, перенос, завершение и история задач продолжают работать.",
        )
        topics["examples"] = (
            "💬 Примеры сообщений",
            "Inbox — сохранённые записи; /drafts — ещё не подтверждённые preview; /tasks — "
            "состояние и сроки задач.\n\n"
            "• Сохрани идею: записывать одну победу дня.\n"
            "• Покажи мои задачи.\n"
            "• Какие задачи просрочены?\n"
            "• Удали все неактуальные задачи.\n"
            "• Создай проект Наз и Войд.\n"
            "• Открой карту желаний.\n"
            "• Хочу сделать health check-in.",
        )
    sections = navigation_sections(
        enable_workspace_access,
        enable_knowledge_hub,
        enable_knowledge_capture,
    )
    feature_lines = []
    for item in sections.values():
        description = item.description
        label = item.label
        if item.key == "tasks" and not enable_task_reminders:
            label = "Задачи"
            description = "Задачи со сроками, состояниями, переносом и историей."
        feature_lines.append(f"{item.emoji} {label} — {description}")
    topics["features"] = ("🧭 Что умеет бот", "\n".join(feature_lines))
    command_lines = []
    for item in public_commands(enable_workspace_access, enable_knowledge_hub):
        description = item.description
        if item.command == "tasks" and not enable_task_reminders:
            description = "Задачи и сроки"
        command_lines.append(f"/{item.command} — {description}")
    topics["commands"] = (
        "⌨️ Основные команды",
        "\n".join(command_lines),
    )
    privacy_parts = [
        "Бот работает только в личном чате. Личные карточки, анализы, health-данные и "
        "личные разделы не становятся общими автоматически."
    ]
    if enable_workspace_access:
        privacy_parts.append(
            "В совместном пространстве видны только явно добавленные данные и только "
            "участникам с действующим доступом."
        )
    if enable_knowledge_hub:
        privacy_parts.append(
            "База знаний показывает только личные или доступные участнику источники."
        )
    privacy_parts.append(
        "Telegram ID не включаются в диагностические логи. Явные команды разделов "
        "обрабатываются без LLM."
    )
    topics["privacy"] = (
        "🔒 Конфиденциальность",
        " ".join(privacy_parts),
    )
    if enable_workspace_access:
        topics["spaces"] = (
            "🤝 Совместные пространства",
            "/spaces открывает доступные пространства, участников и проекты. Общими становятся "
            "только явно добавленные данные. Адресная карточка предназначена выбранному "
            "пользователю. Если бот ещё не может написать ему, появится одноразовая ссылка: "
            "передай её лично — она закрепится за первым принявшим и имеет срок действия. "
            "Роли определяют доступные действия.",
        )
    if enable_knowledge_hub:
        capture_note = (
            " /capture добавляет текст, документ, изображение или ссылку через preview и явное "
            "подтверждение."
            if enable_knowledge_capture
            else " Добавление новых материалов сейчас отключено настройкой."
        )
        topics["knowledge"] = (
            "📚 База знаний",
            "/knowledge показывает личные и доступные совместные источники и их статус."
            f"{capture_note} Материалы не попадают в LLM-контекст автоматически.",
        )
    return topics


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


def validate_catalog(
    enable_workspace_access: bool = False,
    enable_knowledge_hub: bool = False,
    enable_knowledge_capture: bool = False,
) -> None:
    commands = public_commands(enable_workspace_access, enable_knowledge_hub)
    actions = navigation_actions(
        enable_workspace_access,
        enable_knowledge_hub,
        enable_knowledge_capture,
    )
    sections = navigation_sections(
        enable_workspace_access,
        enable_knowledge_hub,
        enable_knowledge_capture,
    )
    command_names = [item.command for item in commands]
    if len(command_names) != len(set(command_names)):
        raise ValueError("Duplicate public navigation commands")
    used_actions: set[str] = set()
    for section in sections.values():
        if not section.actions:
            raise ValueError(f"Empty navigation section: {section.key}")
        for action in section.actions:
            if action not in actions:
                raise ValueError(f"Unknown navigation action: {action}")
            used_actions.add(action)
    if used_actions != set(actions):
        raise ValueError("Unreachable navigation action")


validate_catalog()
