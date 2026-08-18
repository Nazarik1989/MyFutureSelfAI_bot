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
    CommandSpec("today", "Сегодня"),
    CommandSpec("tasks", "Задачи"),
    CommandSpec("inbox", "Записи"),
    CommandSpec("vision", "Желания и визуализация"),
    CommandSpec("health", "Здоровье"),
    CommandSpec("help", "Помощь"),
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
        "mynova",
        "week",
        "timezone",
        "evening",
        "collections",
        "checkin",
        "doctor",
        "labs",
        "location",
        "goals",
        "drafts",
        "last_saved",
        "cleanup_drafts",
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

WORKSPACE_ADVANCED_COMMANDS = frozenset({"spaces", "workspaces"})

KNOWLEDGE_ADVANCED_COMMANDS = frozenset({"knowledge", "capture"})

ACTIONS = {
    action.key: action
    for action in (
        NavigationAction(
            "today",
            "Фокус на сегодня",
            "Короткий персональный план: один фокус, до трёх действий и минимальный шаг.",
            "today",
        ),
        NavigationAction(
            "weekly_review",
            "🧭 Обзор недели",
            "Один подтверждённый ориентир недели и отдельная проверка напоминаний.",
            "week_command",
        ),
        NavigationAction(
            "evening",
            "Вечерний итог",
            "Пять спокойных вопросов о дне без оценок и давления.",
            "evening_start",
        ),
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
            "Разовые и ежедневные напоминания: создание, изменение времени и отключение.",
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
        NavigationAction("location", "Локация", "Город и запасной маршрут.", "location_command"),
        NavigationAction("profile", "Мой профиль", "Vision Profile и текущая локация.", "profile"),
        NavigationAction(
            "timezone",
            "Часовой пояс",
            "Местное время для сценариев и напоминаний.",
            "timezone_command",
        ),
        NavigationAction(
            "onboarding",
            "Настроить или продолжить настройку профиля",
            "Продолжить первоначальную настройку.",
            "start",
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
            "today",
            "🌱",
            "Сегодня",
            "Фокус, задачи на сегодня и спокойный вечерний итог.",
            ("today", "weekly_review", "task_today", "evening"),
        ),
        NavigationSection(
            "tasks",
            "✅",
            "Задачи",
            "Создавай задачи и открывай нужный список.",
            (
                "task_create",
                "task_today",
                "task_upcoming",
                "task_overdue",
                "task_no_due",
                "task_completed",
            ),
        ),
        NavigationSection(
            "records",
            "📝",
            "Записи",
            "Новую мысль можно просто отправить текстом или голосом.",
            ("inbox", "drafts", "last_saved"),
        ),
        NavigationSection(
            "health",
            "❤️",
            "Здоровье",
            "Состояние, check-in, врач, подготовка к приёму и анализы.",
            (
                "health",
                "checkin",
                "doctor_find",
                "doctor_prepare",
                "labs",
                "doctor_preparations",
            ),
        ),
        NavigationSection(
            "sections",
            "🗂",
            "Мои разделы",
            "Сферы жизни, проекты и списки без дублирования записей.",
            ("collections",),
        ),
        NavigationSection(
            "settings",
            "⚙️",
            "Настройки",
            "Профиль, timezone и личная локация.",
            ("profile", "timezone", "location", "onboarding"),
        ),
    )
}

LEGACY_SECTION_ALIASES = {
    "day": "today",
    "ideas": "records",
    "doctor": "health",
    "profile": "settings",
    "collections": "sections",
    "spaces": "sections",
    "organization": "sections",
}

LEGACY_ACTIONS = frozenset({"task_reminder_guide", "doctor_task_guide", "vision"})


_LEGACY_HELP_TOPICS = {
    "quick": (
        "🚀 Быстрый старт",
        "1. Отправь мысль обычными словами.\n"
        "2. Проверь preview: тип, название, дата и следующий шаг.\n"
        "3. Нажми «Сохранить», «Редактировать» или «Не сохранять».\n"
        "4. Записи ищи в /inbox, задачи и сроки — в /tasks.\n\n"
        "Команды помнить не нужно: /menu открывает все разделы.",
    ),
    "day": (
        "🌱 Мой день",
        "/today собирает персональный фокус из подтверждённых целей, рутин и задач: один "
        "главный ориентир, до трёх небольших действий и минимальный план на сложный день. "
        "/evening запускает короткую рефлексию из пяти вопросов и сохраняет её для будущего "
        "планирования. Оба сценария доступны кнопками в разделе «Мой день».",
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


ROOT_HELP_TOPIC_KEYS = (
    "quick",
    "requests",
    "examples",
    "privacy",
    "troubleshooting",
)

HELP_TOPIC_LABELS = {
    "quick": "🚀 Быстрый старт",
    "requests": "🧭 Что можно попросить",
    "examples": "💬 Примеры фраз",
    "privacy": "🔒 Данные и безопасность",
    "troubleshooting": "🧰 Если бот не понял",
}

SECTION_HELP_TOPICS = {
    "today": "today_section",
    "tasks": "tasks_section",
    "records": "records_section",
    "health": "health_section",
    "sections": "sections_section",
    "settings": "settings_section",
}

HELP_TOPICS = {
    "quick": (
        "🚀 Быстрый старт",
        "Открой /menu и выбери раздел. Новую мысль можно просто отправить текстом или "
        "голосом: перед сохранением бот покажет preview.",
    ),
    "requests": (
        "🧭 Что можно попросить",
        "Можно открыть задачи, записи, здоровье, настройки, свои разделы или визуализацию "
        "обычной короткой фразой. Например: «где мои задачи?» или «покажи визуализацию».",
    ),
    "examples": (
        "💬 Примеры фраз",
        "• Где мои задачи?\n"
        "• Как создать задачу?\n"
        "• Где мои записи?\n"
        "• Покажи визуализацию.\n"
        "• Где анализы?\n"
        "• Как изменить часовой пояс?",
    ),
    "privacy": (
        "🔒 Данные и безопасность",
        "Бот работает только в личном чате. Явные команды навигации обрабатываются "
        "локально: они не отправляются в AI и не сохраняются как записи. Медицинские "
        "разделы не ставят диагнозы.",
    ),
    "troubleshooting": (
        "🧰 Если бот не понял",
        "Сформулируй коротко: «где…», «как открыть…», «покажи…» или «открой…». "
        "Если активен пошаговый сценарий, бот предложит продолжить его или выйти в меню.",
    ),
    "today_section": (
        "🌱 Сегодня",
        "«Фокус на сегодня» собирает ориентир дня, «Задачи на сегодня» открывает текущий "
        "список, а «Вечерний итог» запускает короткую рефлексию.",
    ),
    "tasks_section": (
        "✅ Задачи",
        "Разовое или ежедневное напоминание можно создать обычной фразой. Для ежедневного "
        "укажи «каждый день» и время, затем проверь preview и часовой пояс. В карточке задачи "
        "можно изменить время или отключить ежедневное напоминание; срок задачи хранится "
        "отдельно.",
    ),
    "records_section": (
        "📝 Записи",
        "Отправь новую мысль текстом или голосом. «Мои записи» показывает сохранённое, "
        "«Черновики» — preview, а «Последнее сохранённое» — последнюю подтверждённую запись.",
    ),
    "health_section": (
        "❤️ Здоровье",
        "Здесь находятся состояние и check-in, поиск врача, подготовка к приёму и анализы. "
        "Локация настраивается отдельно в «Настройках».",
    ),
    "sections_section": (
        "🗂 Мои разделы",
        "Разделы объединяют существующие записи и задачи без копирования. Совместные "
        "пространства появляются здесь только когда функция включена.",
    ),
    "settings_section": (
        "⚙️ Настройки",
        "Здесь можно открыть профиль, проверить или изменить часовой пояс и локацию, "
        "а также продолжить настройку профиля.",
    ),
}


def public_commands(
    enable_workspace_access: bool = False,
    enable_knowledge_hub: bool = False,
) -> tuple[CommandSpec, ...]:
    """Return the compact native catalog; advanced commands remain supported."""

    del enable_workspace_access, enable_knowledge_hub
    return PUBLIC_COMMANDS


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
    enable_weekly_review: bool = True,
) -> dict[str, NavigationAction]:
    result = dict(ACTIONS)
    if not enable_weekly_review:
        result.pop("weekly_review", None)
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
    enable_weekly_review: bool = True,
) -> dict[str, NavigationSection]:
    if not enable_workspace_access and not enable_knowledge_hub and enable_weekly_review:
        return SECTIONS
    result = dict(SECTIONS)
    if not enable_weekly_review:
        today = result["today"]
        result["today"] = NavigationSection(
            key=today.key,
            emoji=today.emoji,
            label=today.label,
            description=today.description,
            actions=tuple(action for action in today.actions if action != "weekly_review"),
        )
    section = result["sections"]
    actions = list(section.actions)
    if enable_workspace_access:
        actions.append("spaces")
    if enable_knowledge_hub:
        actions.append("knowledge")
        if enable_knowledge_capture:
            actions.append("capture")
    result["sections"] = NavigationSection(
        key=section.key,
        emoji=section.emoji,
        label=section.label,
        description=section.description,
        actions=tuple(actions),
    )
    return result


def help_topics(
    enable_workspace_access: bool = False,
    enable_knowledge_hub: bool = False,
    enable_knowledge_capture: bool = False,
    enable_voice: bool = True,
    enable_task_reminders: bool = True,
) -> dict[str, tuple[str, str]]:
    """Build compact root help plus contextual section topics."""

    del enable_knowledge_capture
    topics = dict(HELP_TOPICS)
    if not enable_voice:
        topics["quick"] = (
            topics["quick"][0],
            "Открой /menu и выбери раздел. Новую мысль можно просто отправить текстом: "
            "перед сохранением бот покажет preview.",
        )
        topics["records_section"] = (
            topics["records_section"][0],
            "Отправь новую мысль текстом. «Мои записи» показывает сохранённое, "
            "«Черновики» — preview, а «Последнее сохранённое» — последнюю "
            "подтверждённую запись.",
        )
    if not enable_task_reminders:
        topics["tasks_section"] = (
            "✅ Задачи",
            "Задачи, сроки, перенос и история доступны. Доставка разовых и ежедневных "
            "Telegram-напоминаний сейчас отключена настройкой.",
        )
    privacy_parts = [topics["privacy"][1]]
    if enable_workspace_access:
        privacy_parts.append(
            "Совместными становятся только явно добавленные данные и только для участников "
            "с действующим доступом."
        )
    if enable_knowledge_hub:
        privacy_parts.append(
            "База знаний показывает только личные или доступные участнику источники."
        )
    topics["privacy"] = (topics["privacy"][0], " ".join(privacy_parts))
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
    enable_weekly_review: bool = True,
) -> None:
    commands = public_commands(enable_workspace_access, enable_knowledge_hub)
    actions = navigation_actions(
        enable_workspace_access,
        enable_knowledge_hub,
        enable_knowledge_capture,
        enable_weekly_review,
    )
    sections = navigation_sections(
        enable_workspace_access,
        enable_knowledge_hub,
        enable_knowledge_capture,
        enable_weekly_review,
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
    if used_actions | LEGACY_ACTIONS != set(actions):
        raise ValueError("Unreachable navigation action")


validate_catalog()
