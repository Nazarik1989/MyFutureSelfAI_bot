from __future__ import annotations

import asyncio
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic

from .access import ACCESS_TIERS, ADMIN, BLOCKED, GUEST, AccessTier
from .navigation import help_topics, navigation_actions, navigation_sections, public_commands

NOVA_SESSION_TTL_SECONDS = 15 * 60
NOVA_MAX_SESSIONS = 128
NOVA_MAX_ACTIONS_PER_SESSION = 8


class NovaCapabilityKind(StrEnum):
    ACTION = "action"
    SECTION = "section"
    HELP = "help"
    GUEST = "guest"


class NovaResolutionKind(StrEnum):
    GUIDE = "guide"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"


class NovaBackTarget(StrEnum):
    NOVA = "nova"
    MENU = "menu"
    GUEST = "guest"


@dataclass(frozen=True, slots=True)
class NovaRuntimeFlags:
    enable_workspace_access: bool = False
    enable_knowledge_hub: bool = False
    enable_knowledge_capture: bool = False
    enable_voice: bool = True
    enable_task_reminders: bool = True
    guest_ai_enabled: bool = True
    enable_vision_image_generation: bool = False
    vision_image_admin_only: bool = True
    enable_nova_ai: bool = False
    nova_ai_admin_only: bool = True

    @classmethod
    def from_settings(cls, settings: object) -> NovaRuntimeFlags:
        transcription_provider = getattr(settings, "transcription_provider", None)
        return cls(
            enable_workspace_access=bool(getattr(settings, "enable_workspace_access", False)),
            enable_knowledge_hub=bool(getattr(settings, "enable_knowledge_hub", False)),
            enable_knowledge_capture=bool(getattr(settings, "enable_knowledge_capture", False)),
            enable_voice=bool(
                getattr(
                    settings,
                    "enable_voice",
                    transcription_provider not in (None, "disabled"),
                )
            ),
            enable_task_reminders=bool(getattr(settings, "enable_task_reminders", True)),
            guest_ai_enabled=bool(getattr(settings, "guest_ai_enabled", False)),
            enable_vision_image_generation=bool(
                getattr(settings, "enable_vision_image_generation", False)
            ),
            vision_image_admin_only=bool(getattr(settings, "vision_image_admin_only", True)),
            enable_nova_ai=bool(getattr(settings, "enable_nova_ai", False)),
            nova_ai_admin_only=bool(getattr(settings, "nova_ai_admin_only", True)),
        )


@dataclass(frozen=True, slots=True)
class NovaCapability:
    id: str
    label: str
    description: str
    target: str
    kind: NovaCapabilityKind


@dataclass(frozen=True, slots=True)
class NovaCatalog:
    tier: AccessTier
    capabilities: tuple[NovaCapability, ...]
    enabled_features: tuple[str, ...]

    def capability(self, action_id: str) -> NovaCapability | None:
        return next((item for item in self.capabilities if item.id == action_id), None)

    def allows(self, action_id: str) -> bool:
        return self.capability(action_id) is not None


@dataclass(frozen=True, slots=True)
class NovaResolution:
    kind: NovaResolutionKind
    response: str
    steps: tuple[str, ...] = ()
    action_id: str | None = None
    cta_label: str | None = None
    back_target: NovaBackTarget = NovaBackTarget.NOVA


# IDs are deliberately allowlisted independently of labels and handler names. Labels and
# descriptions are still read from the canonical navigation catalog on every build.
_SAFE_NAVIGATION_ACTION_IDS = frozenset(
    {
        "today",
        "evening",
        "inbox",
        "drafts",
        "last_saved",
        "task_today",
        "task_upcoming",
        "task_overdue",
        "task_no_due",
        "task_completed",
        "task_create",
        "task_reminder_guide",
        "collections",
        "vision",
        "health",
        "checkin",
        "doctor_find",
        "doctor_prepare",
        "doctor_preparations",
        "labs",
        "location",
        "profile",
        "timezone",
        "onboarding",
        "spaces",
        "knowledge",
        "capture",
    }
)

_GUEST_CAPABILITIES = (
    NovaCapability(
        id="guest:features",
        label="🧭 Что я умею",
        description="Показать существующий обзор возможностей полной версии.",
        target="guest:features",
        kind=NovaCapabilityKind.GUEST,
    ),
    NovaCapability(
        id="guest:demos",
        label="✨ Попробовать бесплатно",
        description="Два доступных гостю AI-разбора и их текущий лимит.",
        target="guest:demos",
        kind=NovaCapabilityKind.GUEST,
    ),
    NovaCapability(
        id="guest:demo:thought",
        label="📝 Разобрать мысль",
        description="Превратить свободный текст в структуру и следующий шаг.",
        target="guest:demo:thought",
        kind=NovaCapabilityKind.GUEST,
    ),
    NovaCapability(
        id="guest:demo:first-step",
        label="🌱 Найти первый шаг",
        description="Получить небольшой первый шаг для цели или изменения.",
        target="guest:demo:first-step",
        kind=NovaCapabilityKind.GUEST,
    ),
    NovaCapability(
        id="guest:access",
        label="💬 Подписка или свой бот",
        description="Открыть существующий экран обращения за полным доступом.",
        target="guest:access",
        kind=NovaCapabilityKind.GUEST,
    ),
)


def build_nova_catalog(
    tier: AccessTier,
    flags: NovaRuntimeFlags | None = None,
) -> NovaCatalog:
    """Build the server-side allowlist from canonical navigation data and live flags."""

    if tier not in ACCESS_TIERS:
        raise ValueError("unsupported access tier")
    runtime = flags or NovaRuntimeFlags()
    features = _enabled_features(tier, runtime)
    if tier == BLOCKED:
        return NovaCatalog(tier=tier, capabilities=(), enabled_features=features)
    if tier == GUEST:
        guest_capabilities = tuple(
            capability
            for capability in _GUEST_CAPABILITIES
            if runtime.guest_ai_enabled
            or capability.id not in {"guest:demos", "guest:demo:thought", "guest:demo:first-step"}
        )
        return NovaCatalog(
            tier=tier,
            capabilities=guest_capabilities,
            enabled_features=features,
        )

    actions = navigation_actions(
        runtime.enable_workspace_access,
        runtime.enable_knowledge_hub,
        runtime.enable_knowledge_capture,
    )
    sections = navigation_sections(
        runtime.enable_workspace_access,
        runtime.enable_knowledge_hub,
        runtime.enable_knowledge_capture,
    )
    topics = help_topics(
        runtime.enable_workspace_access,
        runtime.enable_knowledge_hub,
        runtime.enable_knowledge_capture,
        runtime.enable_voice,
        runtime.enable_task_reminders,
    )
    menu_label = next(item.description for item in public_commands() if item.command == "menu")
    capabilities = [
        NovaCapability(
            id="menu",
            label=menu_label,
            description="Открыть существующее главное меню бота.",
            target="nav:root",
            kind=NovaCapabilityKind.SECTION,
        )
    ]
    capabilities.extend(
        NovaCapability(
            id=key,
            label=action.label,
            description=action.description,
            target=f"nav:action:{key}",
            kind=NovaCapabilityKind.ACTION,
        )
        for key, action in actions.items()
        if key in _SAFE_NAVIGATION_ACTION_IDS and action.handler is not None
    )
    capabilities.extend(
        NovaCapability(
            id=f"section:{key}",
            label=f"{section.emoji} {section.label}",
            description=section.description,
            target=f"nav:section:{key}",
            kind=NovaCapabilityKind.SECTION,
        )
        for key, section in sections.items()
    )
    capabilities.extend(
        NovaCapability(
            id=f"help:{key}",
            label=label,
            description=description,
            target=f"nav:help:{key}",
            kind=NovaCapabilityKind.HELP,
        )
        for key, (label, description) in topics.items()
    )
    return NovaCatalog(
        tier=tier,
        capabilities=tuple(capabilities),
        enabled_features=features,
    )


# A descriptive alias makes integration call sites read naturally while keeping one builder.
build_capability_catalog = build_nova_catalog


def _enabled_features(tier: AccessTier, flags: NovaRuntimeFlags) -> tuple[str, ...]:
    enabled: list[str] = []
    if flags.enable_workspace_access and tier not in (GUEST, BLOCKED):
        enabled.append("workspace")
    if flags.enable_knowledge_hub and tier not in (GUEST, BLOCKED):
        enabled.append("knowledge")
    if (
        flags.enable_knowledge_hub
        and flags.enable_knowledge_capture
        and tier not in (GUEST, BLOCKED)
    ):
        enabled.append("knowledge_capture")
    if flags.enable_voice and tier not in (GUEST, BLOCKED):
        enabled.append("voice")
    if flags.enable_task_reminders and tier not in (GUEST, BLOCKED):
        enabled.append("task_reminders")
    if flags.guest_ai_enabled and tier == GUEST:
        enabled.append("guest_demos")
    if flags.enable_vision_image_generation and tier not in (GUEST, BLOCKED):
        if not flags.vision_image_admin_only or tier == ADMIN:
            enabled.append("vision_image_generation")
    if flags.enable_nova_ai and tier not in (GUEST, BLOCKED):
        if not flags.nova_ai_admin_only or tier == ADMIN:
            enabled.append("nova_ai")
    return tuple(enabled)


_NOVA_PREFIX = re.compile(r"^nova(?:\s*[,.:!?;—-]\s*|\s+)(?P<question>.+)$", re.IGNORECASE)
_EXPLICIT_HELP_PREFIXES = (
    re.compile(
        r"^помоги\s+в\s+(?:этом\s+)?боте(?:\s*[,.:!?;—-]\s*|\s+)(?P<question>.+)$", re.IGNORECASE
    ),
    re.compile(
        r"^подскажи\s*,?\s*как\s+в\s+(?:этом\s+)?боте(?:\s*[,.:!?;—-]\s*|\s+)(?P<question>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^не\s+могу\s+найти\s+в\s+(?:этом\s+)?боте(?:\s*[,.:!?;—-]\s*|\s+)(?P<question>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^как\s+пользоваться\s+(?:этим\s+)?ботом(?:\s*[,.:!?;—-]\s*|\s+)(?P<question>.+)$",
        re.IGNORECASE,
    ),
)
_EXPLICIT_COMPLETE_HELP = re.compile(
    r"^(?:"
    r"как\s+пользоваться\s+(?:этим\s+)?ботом|"
    r"помощь|покажи\s+помощь|"
    r"какие\s+(?:у\s+тебя\s+)?(?:есть\s+)?команды|"
    r"что\s+ты\s+умеешь"
    r")\s*[?.!]*$",
    re.IGNORECASE,
)

# Natural bot-help routing is intentionally narrower than the local resolver. A
# capability word on its own is ordinary user content; a find/open/use cue must
# also resolve to an enabled action in the live catalog.
_NOVA_HELP_STRONG_CUES = (
    "не могу найти",
    "не могу ее найти",
    "не знаю где",
    "где находится",
    "где у тебя",
    "где мои",
    "как найти",
    "как открыть",
    "как попасть",
    "как пользоваться",
    "как использовать",
    "куда нажать",
    "покажи где",
    "подскажи где",
)
_NOVA_HELP_VERB_FRAGMENTS = (
    "найт",
    "откры",
    "пользова",
    "использова",
    "настро",
    "загруз",
    "добав",
    "созда",
    "перейт",
    "попаст",
    "покаж",
)
_NOVA_HELP_CONTEXT_FRAGMENTS = (
    "в боте",
    "в этом боте",
    "у тебя",
    "в меню",
    "в менюшк",
    "главное меню",
)
_NOVA_CONTENT_NEGATIONS = (
    "не спрашиваю",
    "это личная мысль",
    "обычная мысль",
)
_NOVA_EXTERNAL_CONTEXT_FRAGMENTS = (
    "photoshop",
    "excel",
    "powerpoint",
    "учебник",
    "презентац",
)
_NOVA_EXPLICIT_BOT_CONTEXT_FRAGMENTS = ("в боте", "в этом боте", "меню бота")
_NOVA_FOLLOW_UPS = frozenset(
    {
        "ладно объясни",
        "объясни",
        "расскажи подробнее",
        "как это работает",
        "покажи где это",
        "покажи где",
    }
)
_SAFE_ACTION_ID = re.compile(r"^[a-z][a-z0-9:_-]{0,99}$")


def extract_explicit_nova_question(text: str) -> str | None:
    """Return an explicitly addressed help question without broad content capture."""

    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    match = _NOVA_PREFIX.fullmatch(cleaned)
    if match is not None:
        return match.group("question").strip()
    for pattern in _EXPLICIT_HELP_PREFIXES:
        match = pattern.fullmatch(cleaned)
        if match is not None:
            return match.group("question").strip()
    if _EXPLICIT_COMPLETE_HELP.fullmatch(cleaned):
        return cleaned
    return None


def is_explicit_nova_invocation(text: str) -> bool:
    return extract_explicit_nova_question(text) is not None


def is_nova_help_intent(
    text: str,
    catalog: NovaCatalog,
    *,
    active_session: bool = False,
    last_action_id: str | None = None,
) -> bool:
    """Classify explicit bot/interface help without capturing ordinary content."""

    if not isinstance(text, str) or not text.strip():
        return False
    if extract_explicit_nova_question(text) is not None:
        return True
    normalized = _normalize(text)
    if active_session:
        return True
    if (
        last_action_id is not None
        and _is_nova_follow_up(normalized)
        and catalog.allows(last_action_id)
    ):
        return True
    if any(marker in normalized for marker in _NOVA_CONTENT_NEGATIONS):
        return False
    if any(marker in normalized for marker in _NOVA_EXTERNAL_CONTEXT_FRAGMENTS) and not any(
        marker in normalized for marker in _NOVA_EXPLICIT_BOT_CONTEXT_FRAGMENTS
    ):
        return False
    has_bot_context = any(fragment in normalized for fragment in _NOVA_HELP_CONTEXT_FRAGMENTS)
    has_strong_cue = has_bot_context and any(cue in normalized for cue in _NOVA_HELP_STRONG_CUES)
    has_owned_capability_cue = "подскажи" in normalized and "где мои" in normalized
    has_help_lead = (
        normalized.startswith(("как ", "где ", "куда ", "подскажи ", "помоги ", "покажи "))
        or " подскажи" in normalized
        or " помоги" in normalized
    )
    has_help_verb = any(fragment in normalized for fragment in _NOVA_HELP_VERB_FRAGMENTS)
    if (
        not has_strong_cue
        and not has_owned_capability_cue
        and not (has_help_lead and has_help_verb and has_bot_context)
    ):
        return False
    return _known_local_action_id(normalized, catalog) is not None


def resolve_nova_question(
    question: str,
    catalog: NovaCatalog,
    *,
    last_action_id: str | None = None,
) -> NovaResolution | None:
    """Resolve known bot-capability questions locally without invoking an AI provider."""

    if not isinstance(question, str):
        return None
    cleaned = extract_explicit_nova_question(question) or question.strip()
    if len(cleaned) > 600:
        return NovaResolution(
            kind=NovaResolutionKind.CLARIFY,
            response="Сократи вопрос до 600 символов, и я помогу найти функцию бота.",
        )
    normalized = _normalize(cleaned)
    if not normalized:
        return NovaResolution(
            kind=NovaResolutionKind.CLARIFY,
            response="Расскажи коротко, что хочешь найти или сделать в боте.",
        )
    if catalog.tier == BLOCKED:
        return _unsupported("Помощь Nova недоступна для текущего уровня доступа.")

    if _is_nova_follow_up(normalized):
        if last_action_id is None:
            return NovaResolution(
                kind=NovaResolutionKind.CLARIFY,
                response="Уточни, о какой функции бота рассказать подробнее.",
            )
        capability = catalog.capability(last_action_id)
        if capability is None:
            return NovaResolution(
                kind=NovaResolutionKind.CLARIFY,
                response="Эта функция сейчас недоступна. Выбери другой раздел.",
            )
        return _follow_up_guide(catalog, capability)

    unsupported = _unsupported_question(normalized)
    if unsupported is not None:
        return unsupported

    if catalog.tier == GUEST:
        guest = _resolve_guest_question(normalized, catalog)
        if guest is not None:
            return guest

    if _contains(normalized, "пространств", "workspace"):
        return _resolve_optional_feature(
            catalog,
            "spaces",
            "Совместные пространства доступны только когда функция включена.",
            ("Открой «Мои разделы».", "Выбери доступное пространство."),
        )
    if _contains(normalized, "баз знаний", "база знаний", "базу знаний", "knowledge"):
        return _resolve_optional_feature(
            catalog,
            "knowledge",
            "База знаний доступна только когда функция включена.",
            ("Открой базу знаний.", "Выбери личный или доступный материал."),
        )
    if _contains(normalized, "добавить материал", "загрузить материал", "capture"):
        return _resolve_optional_feature(
            catalog,
            "capture",
            "Добавление материалов доступно только когда Knowledge Capture включён.",
            ("Открой добавление материала.", "Проверь preview перед подтверждением."),
        )

    rule = _standard_rule(normalized)
    if rule is None:
        if catalog.tier == GUEST:
            return NovaResolution(
                kind=NovaResolutionKind.CLARIFY,
                response=(
                    "В гостевом режиме я могу подсказать про два демо, лимит и получение "
                    "полного доступа."
                ),
                back_target=NovaBackTarget.GUEST,
            )
        return None
    action_id, response, steps = rule
    if catalog.tier == GUEST:
        return _guest_full_version(catalog, response)
    return _guide(catalog, action_id, response, steps)


def _standard_rule(normalized: str) -> tuple[str, str, tuple[str, ...]] | None:
    if normalized in {"помощь", "покажи помощь"}:
        return (
            "help:quick",
            "Быстрый старт показывает, как открыть раздел или отправить новую мысль.",
            ("Открой краткую инструкцию.",),
        )
    if _contains(normalized, "какие команды", "какие у тебя команды"):
        return (
            "help:requests",
            "Возможности собраны в локальном обзоре без отправки вопроса в AI.",
            ("Открой обзор возможностей.",),
        )
    if _contains(
        normalized, "как пользоваться ботом", "как пользоваться этим ботом", "быстрый старт"
    ):
        return (
            "help:quick",
            "Быстрый старт показывает, как открыть раздел или отправить новую мысль.",
            ("Открой краткую инструкцию.",),
        )
    if _contains(normalized, "что умеет бот", "что ты умеешь", "возможност бот"):
        return (
            "help:requests",
            "Возможности собраны в локальном обзоре без отправки вопроса в AI.",
            ("Открой обзор возможностей.",),
        )
    if _contains(
        normalized,
        "пример вопрос",
        "примеры вопросов",
        "примеры фраз",
        "что спросить",
    ):
        return (
            "help:examples",
            "В справке есть короткие примеры вопросов о существующих функциях.",
            ("Открой примеры.",),
        )
    if _contains(normalized, "данные и безопас", "конфиденциаль", "приватност"):
        return (
            "help:privacy",
            "Справка объясняет локальную навигацию, личный чат и безопасное обращение с данными.",
            ("Открой раздел о данных и безопасности.",),
        )
    if _contains(normalized, "бот не понял", "не понимает бот", "не получилось найти"):
        return (
            "help:troubleshooting",
            "Краткая памятка поможет точнее сформулировать запрос о функциях бота.",
            ("Открой памятку.",),
        )
    if _contains(normalized, "напоминан") and _contains(normalized, "задач", "дело"):
        return (
            "task_create",
            "Задачу с напоминанием можно создать через обычный preview.",
            (
                "Сформулируй задачу.",
                "Укажи срок или время напоминания.",
                "Проверь preview перед сохранением.",
            ),
        )
    if _contains(normalized, "напоминан"):
        return (
            "task_reminder_guide",
            "Напоминания настраиваются в задаче и срабатывают по её локальному времени.",
            ("Открой памятку по напоминаниям.", "Создай или выбери задачу."),
        )
    if _contains(normalized, "создать задач", "добавить задач", "новую задач"):
        return (
            "task_create",
            "Новая задача сначала показывается в preview и сохраняется только после подтверждения.",
            ("Сформулируй задачу.", "Проверь срок и напоминание.", "Подтверди сохранение."),
        )
    if _contains(normalized, "задач"):
        return (
            "section:tasks",
            "В разделе задач есть текущие, предстоящие, просроченные и завершённые списки.",
            ("Открой раздел задач.", "Выбери нужный список."),
        )
    if _contains(normalized, "сегодня", "фокус дня", "план дня"):
        return (
            "section:today",
            "Раздел «Сегодня» объединяет фокус дня, текущие задачи и вечерний итог.",
            ("Открой раздел.", "Выбери фокус или задачи на сегодня."),
        )
    if _contains(normalized, "запис", "замет", "иде"):
        return (
            "inbox",
            "Сохранённые мысли и идеи находятся в записях.",
            ("Открой записи.", "Выбери нужную карточку."),
        )
    if _contains(normalized, "check in", "checkin", "чек ин", "чек-ин"):
        return (
            "checkin",
            "Check-in состоит из шести коротких вопросов о самочувствии.",
            ("Запусти check-in.", "Ответь на вопросы.", "Проверь сохранённое состояние."),
        )
    if _contains(normalized, "найти врач", "поиск врач", "врача рядом"):
        return (
            "doctor_find",
            "Поиск врача использует сохранённую личную локацию и официальные варианты.",
            ("Проверь локацию.", "Открой поиск врача."),
        )
    if _contains(normalized, "подготов", "прием", "приём") and _contains(
        normalized, "врач", "доктор", "прием", "приём"
    ):
        return (
            "doctor_prepare",
            "Подготовка к приёму собирает фактическое резюме без диагнозов.",
            ("Открой подготовку.", "Добавь факты и вопросы врачу.", "Проверь резюме."),
        )
    if _contains(normalized, "анализ"):
        return (
            "labs",
            "Раздел анализов безопасно хранит фото и PDF, но не ставит диагнозы.",
            ("Открой анализы.", "Выбери существующий файл или загрузку."),
        )
    if _contains(normalized, "здоров", "самочув", "состояни"):
        return (
            "section:health",
            "Раздел здоровья объединяет состояние, check-in, врача и анализы.",
            ("Открой раздел здоровья.", "Выбери нужное действие."),
        )
    if _contains(normalized, "референс"):
        return (
            "vision",
            "Референсы хранятся в приватной библиотеке раздела визуализации.",
            ("Открой карту.", "Выбери «Мои референсы».", "Добавь или открой изображение."),
        )
    if _contains(normalized, "png", "пнг", "общую карту", "скачать карту"):
        return (
            "vision",
            "Общая PNG-карта собирается локально из активных желаний.",
            ("Открой карту.", "Выбери сборку PNG-карты.", "Укажи нужные желания."),
        )
    if _contains(normalized, "желан"):
        return (
            "vision",
            "Желания находятся в разделе карты и визуализации.",
            ("Открой карту.", "Выбери существующее желание или добавь новое."),
        )
    if _contains(normalized, "визуализац", "карта желаний", "визуальн образ"):
        return (
            "vision",
            "В разделе визуализации доступны желания, личные фото, референсы и локальная PNG-карта.",
            ("Открой карту.", "Выбери нужный вид визуализации."),
        )
    if _contains(normalized, "часовой пояс", "часового пояса", "таймзон", "timezone"):
        return (
            "timezone",
            "Часовой пояс задаёт местное время для сценариев и напоминаний.",
            ("Открой настройку часового пояса.", "Выбери или введи подходящий город."),
        )
    if _contains(normalized, "локац", "местополож", "город"):
        return (
            "location",
            "Локация используется для официального поиска врача и запасного маршрута.",
            ("Открой локацию.", "Укажи город или маршрут."),
        )
    if _contains(normalized, "профил"):
        return (
            "profile",
            "В профиле находятся текущая настройка и основные сведения Vision Profile.",
            ("Открой профиль.", "Проверь или продолжи настройку."),
        )
    if _contains(normalized, "настройк"):
        return (
            "section:settings",
            "В настройках находятся профиль, часовой пояс и личная локация.",
            ("Открой настройки.", "Выбери нужный пункт."),
        )
    if _contains(normalized, "мои раздел", "коллекц", "проект", "список"):
        return (
            "collections",
            "«Мои разделы» объединяют существующие записи и задачи без копирования.",
            ("Открой свои разделы.", "Выбери тему, проект или список."),
        )
    if _contains(
        normalized,
        "главн меню",
        "главное меню",
        "где меню",
        "открой меню",
        "в начало",
    ):
        return (
            "menu",
            "Главное меню собирает все доступные разделы в одном экране.",
            ("Открой меню.", "Выбери нужный раздел."),
        )
    return None


def _resolve_guest_question(normalized: str, catalog: NovaCatalog) -> NovaResolution | None:
    if _contains(
        normalized,
        "что умеет бот",
        "что ты умеешь",
        "возможност",
        "полная версия",
        "как пользоваться ботом",
        "как пользоваться этим ботом",
    ):
        return _guide(
            catalog,
            "guest:features",
            "В гостевом обзоре перечислены возможности полной версии без запуска её обработчиков.",
            ("Открой обзор возможностей.",),
            back_target=NovaBackTarget.GUEST,
        )
    if _contains(normalized, "подпис", "полная версия", "полный доступ", "контакт", "свой бот"):
        return _guide(
            catalog,
            "guest:access",
            "Полный доступ и заказ собственного бота доступны через существующий экран обращения.",
            ("Открой экран обращения.",),
            back_target=NovaBackTarget.GUEST,
        )
    if _contains(normalized, "лимит", "сколько раз", "сколько разбор"):
        return _guide(
            catalog,
            "guest:demos",
            "Гостю доступны две бесплатные AI-операции за всё время.",
            ("Открой экран демо, чтобы увидеть доступные варианты.",),
            back_target=NovaBackTarget.GUEST,
        )
    if _contains(normalized, "первый шаг", "цель"):
        return _guide(
            catalog,
            "guest:demo:first-step",
            "Гостевое демо помогает найти небольшой первый шаг для цели.",
            ("Открой демо.", "Опиши цель одним сообщением."),
            back_target=NovaBackTarget.GUEST,
        )
    if _contains(normalized, "разобрать мысл", "разбор мысл", "демо", "бесплат"):
        return _guide(
            catalog,
            "guest:demo:thought",
            "Гостевое демо может разобрать мысль и предложить следующий шаг.",
            ("Открой демо.", "Отправь одну мысль."),
            back_target=NovaBackTarget.GUEST,
        )
    return None


def _guest_full_version(catalog: NovaCatalog, feature_description: str) -> NovaResolution:
    return _guide(
        catalog,
        "guest:access",
        f"{feature_description} Эта функция относится к полной версии бота.",
        ("Открой экран обращения за полным доступом.",),
        back_target=NovaBackTarget.GUEST,
    )


def _resolve_optional_feature(
    catalog: NovaCatalog,
    action_id: str,
    unavailable: str,
    steps: tuple[str, ...],
) -> NovaResolution:
    capability = catalog.capability(action_id)
    if capability is None:
        if catalog.tier == GUEST:
            return _guest_full_version(catalog, unavailable)
        return _unsupported(unavailable)
    return _guide(catalog, action_id, capability.description, steps)


def _guide(
    catalog: NovaCatalog,
    action_id: str,
    response: str,
    steps: tuple[str, ...],
    *,
    back_target: NovaBackTarget = NovaBackTarget.NOVA,
) -> NovaResolution:
    capability = catalog.capability(action_id)
    if capability is None:
        return _unsupported("Эта функция сейчас недоступна.")
    return NovaResolution(
        kind=NovaResolutionKind.GUIDE,
        response=response,
        steps=steps[:3],
        action_id=capability.id,
        cta_label=("🎯 Открыть визуализацию" if capability.id == "vision" else capability.label),
        back_target=back_target,
    )


def _follow_up_guide(catalog: NovaCatalog, capability: NovaCapability) -> NovaResolution:
    if capability.id == "vision":
        return _guide(
            catalog,
            capability.id,
            "Визуализация объединяет желания, личные фото, референсы и локальную PNG-карту.",
            (
                "Открой раздел визуализации кнопкой ниже.",
                "Выбери желания, личные фото или референсы.",
                "Для общей карты запусти локальную сборку PNG.",
            ),
        )
    return _guide(
        catalog,
        capability.id,
        capability.description,
        ("Открой нужный раздел кнопкой ниже.", "Выбери подходящее действие."),
    )


def _unsupported(response: str) -> NovaResolution:
    return NovaResolution(kind=NovaResolutionKind.UNSUPPORTED, response=response)


def _unsupported_question(normalized: str) -> NovaResolution | None:
    if _contains(normalized, "утренн") and _contains(normalized, "послан", "сообщен"):
        return _unsupported("Персональные утренние послания пока не реализованы.")
    if (
        _contains(normalized, "карт")
        and _contains(normalized, "будущ")
        and _contains(normalized, "ai", "ии", "нейросет")
    ):
        return _unsupported(
            "Общая AI-карта будущего пока не реализована. Доступна локальная PNG-карта желаний."
        )
    if _contains(
        normalized,
        "выдать доступ",
        "изменить доступ",
        "дать права",
        "дай права",
        "заблокировать пользователя",
        "удалить пользователя",
    ):
        return _unsupported("Nova не выполняет административные и destructive-действия.")
    return None


def _normalize(text: str) -> str:
    lowered = text.casefold().replace("ё", "е")
    lowered = re.sub(r"[^a-zа-я0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains(text: str, *fragments: str) -> bool:
    return any(fragment.replace("ё", "е") in text for fragment in fragments)


def _known_local_action_id(normalized: str, catalog: NovaCatalog) -> str | None:
    optional = (
        ("spaces", ("пространств", "workspace")),
        ("knowledge", ("баз знаний", "база знаний", "базу знаний", "knowledge")),
        ("capture", ("добавить материал", "загрузить материал", "capture")),
    )
    for action_id, aliases in optional:
        if catalog.allows(action_id) and any(alias in normalized for alias in aliases):
            return action_id
    rule = _standard_rule(normalized)
    if rule is None:
        return None
    action_id = rule[0]
    return action_id if catalog.allows(action_id) else None


def _is_nova_follow_up(normalized: str) -> bool:
    return normalized in _NOVA_FOLLOW_UPS


@dataclass(frozen=True, slots=True)
class NovaSession:
    id: str
    owner_id: int
    telegram_user_id: int
    chat_id: int
    access_version: int
    canonical_message_id: int | None
    tier: AccessTier
    created_at: float
    expires_at: float
    question_in_progress: bool
    last_action_id: str | None


@dataclass(frozen=True, slots=True)
class NovaActionClaim:
    action_id: str
    session: NovaSession


@dataclass(slots=True)
class _StoredNovaSession:
    id: str
    owner_id: int
    telegram_user_id: int
    chat_id: int
    access_version: int
    canonical_message_id: int | None
    tier: AccessTier
    created_at: float
    expires_at: float
    question_in_progress: bool
    last_action_id: str | None
    actions: dict[str, str]


class NovaSessionStore:
    """Bounded, process-local Nova sessions and single-use action capabilities."""

    def __init__(
        self,
        *,
        ttl_seconds: float = NOVA_SESSION_TTL_SECONDS,
        max_sessions: int = NOVA_MAX_SESSIONS,
        max_actions_per_session: int = NOVA_MAX_ACTIONS_PER_SESSION,
        clock: Callable[[], float] = monotonic,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if max_actions_per_session <= 0:
            raise ValueError("max_actions_per_session must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_actions_per_session = max_actions_per_session
        self._clock = clock
        self._sessions: dict[tuple[int, int], _StoredNovaSession] = {}
        self._token_index: dict[str, tuple[int, int]] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int,
        tier: AccessTier,
    ) -> NovaSession:
        return await self._replace_session(
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            access_version=access_version,
            canonical_message_id=canonical_message_id,
            tier=tier,
        )

    async def reserve(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        tier: AccessTier,
    ) -> NovaSession:
        """Reserve a session ID before Telegram returns the canonical message ID."""

        return await self._replace_session(
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            access_version=access_version,
            canonical_message_id=None,
            tier=tier,
        )

    async def _replace_session(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int | None,
        tier: AccessTier,
    ) -> NovaSession:
        self._validate_binding(
            owner_id,
            telegram_user_id,
            chat_id,
            access_version,
            canonical_message_id,
            tier,
        )
        async with self._lock:
            self._prune_locked()
            key = (owner_id, chat_id)
            self._drop_locked(key)
            while len(self._sessions) >= self.max_sessions:
                self._drop_locked(next(iter(self._sessions)))
            now = self._clock()
            stored = _StoredNovaSession(
                id=self._random_token(),
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                tier=tier,
                created_at=now,
                expires_at=now + self.ttl_seconds,
                question_in_progress=False,
                last_action_id=None,
                actions={},
            )
            self._sessions[key] = stored
            return self._snapshot(stored)

    async def get(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        tier: AccessTier,
        canonical_message_id: int | None = None,
        session_id: str | None = None,
    ) -> NovaSession | None:
        async with self._lock:
            stored = self._bound_locked(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                tier=tier,
                session_id=session_id,
            )
            return self._snapshot(stored) if stored is not None else None

    async def current(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
    ) -> NovaSession | None:
        """Find the current owner/chat session so its canonical message can be rediscovered."""

        async with self._lock:
            self._prune_locked()
            key = (owner_id, chat_id)
            stored = self._sessions.get(key)
            if stored is None:
                return None
            if stored.telegram_user_id != telegram_user_id:
                return None
            return self._snapshot(stored)

    async def begin_question(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int,
        tier: AccessTier,
        session_id: str | None = None,
    ) -> NovaSession | None:
        async with self._lock:
            stored = self._bound_locked(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                tier=tier,
                session_id=session_id,
            )
            if stored is None or stored.question_in_progress:
                return None
            for token in stored.actions:
                self._token_index.pop(token, None)
            stored.actions.clear()
            stored.question_in_progress = True
            return self._snapshot(stored)

    async def bind_canonical(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        tier: AccessTier,
        new_message_id: int,
        session_id: str,
    ) -> NovaSession | None:
        return await self.rebind_canonical(
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            access_version=access_version,
            tier=tier,
            expected_message_id=None,
            new_message_id=new_message_id,
            session_id=session_id,
        )

    async def rebind_canonical(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        tier: AccessTier,
        expected_message_id: int | None,
        new_message_id: int,
        session_id: str | None = None,
    ) -> NovaSession | None:
        """Fence a one-off Telegram replacement and bind the session to its new message."""

        if new_message_id <= 0:
            raise ValueError("invalid canonical message")
        async with self._lock:
            stored = self._bound_locked(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=expected_message_id,
                tier=tier,
                session_id=session_id,
            )
            if stored is None or stored.canonical_message_id != expected_message_id:
                return None
            stored.canonical_message_id = new_message_id
            return self._snapshot(stored)

    async def finish_question(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int,
        tier: AccessTier,
        session_id: str,
    ) -> bool:
        async with self._lock:
            stored = self._bound_locked(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                tier=tier,
                session_id=session_id,
            )
            if stored is None or not stored.question_in_progress:
                return False
            stored.question_in_progress = False
            return True

    async def remember_action(
        self,
        *,
        last_action_id: str | None,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int,
        tier: AccessTier,
        session_id: str,
    ) -> NovaSession | None:
        """Remember only a safe capability identifier after fenced UI delivery."""

        if last_action_id is not None and _SAFE_ACTION_ID.fullmatch(last_action_id) is None:
            raise ValueError("invalid last_action_id")
        async with self._lock:
            stored = self._bound_locked(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                tier=tier,
                session_id=session_id,
            )
            if stored is None or not stored.question_in_progress:
                return None
            stored.last_action_id = last_action_id
            return self._snapshot(stored)

    async def issue_action(
        self,
        *,
        action_id: str,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int,
        tier: AccessTier,
        session_id: str | None = None,
    ) -> str | None:
        if not action_id or len(action_id) > 100:
            raise ValueError("invalid action_id")
        async with self._lock:
            stored = self._bound_locked(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                tier=tier,
                session_id=session_id,
            )
            if stored is None:
                return None
            while len(stored.actions) >= self.max_actions_per_session:
                old_token = next(iter(stored.actions))
                stored.actions.pop(old_token, None)
                self._token_index.pop(old_token, None)
            token = self._random_token()
            while token in self._token_index:
                token = self._random_token()
            stored.actions[token] = action_id
            self._token_index[token] = (owner_id, chat_id)
            return token

    async def consume_action(
        self,
        *,
        token: str,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int,
        tier: AccessTier,
        expected_action_id: str | None = None,
    ) -> NovaActionClaim | None:
        async with self._lock:
            self._prune_locked()
            key = self._token_index.get(token)
            if key is None:
                return None
            # A foreign callback cannot consume or invalidate its owner's capability.
            if key != (owner_id, chat_id):
                return None
            stored = self._bound_locked(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=access_version,
                canonical_message_id=canonical_message_id,
                tier=tier,
            )
            if stored is None:
                return None
            action_id = stored.actions.pop(token, None)
            if action_id is None:
                return None
            if expected_action_id is not None and action_id != expected_action_id:
                stored.actions[token] = action_id
                return None
            self._token_index.pop(token, None)
            return NovaActionClaim(action_id=action_id, session=self._snapshot(stored))

    async def clear(
        self,
        *,
        owner_id: int,
        chat_id: int,
        session_id: str | None = None,
    ) -> bool:
        async with self._lock:
            self._prune_locked()
            key = (owner_id, chat_id)
            stored = self._sessions.get(key)
            if stored is None or (session_id is not None and stored.id != session_id):
                return False
            self._drop_locked(key)
            return True

    async def count(self) -> int:
        async with self._lock:
            self._prune_locked()
            return len(self._sessions)

    def _bound_locked(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int | None,
        tier: AccessTier,
        session_id: str | None = None,
    ) -> _StoredNovaSession | None:
        self._prune_locked()
        key = (owner_id, chat_id)
        stored = self._sessions.get(key)
        if stored is None:
            return None
        if session_id is not None and stored.id != session_id:
            return None
        if (
            stored.telegram_user_id != telegram_user_id
            or stored.access_version != access_version
            or stored.tier != tier
        ):
            self._drop_locked(key)
            return None
        if canonical_message_id is not None and stored.canonical_message_id != canonical_message_id:
            return None
        return stored

    def _prune_locked(self) -> None:
        now = self._clock()
        for key in [key for key, item in self._sessions.items() if item.expires_at <= now]:
            self._drop_locked(key)

    def _drop_locked(self, key: tuple[int, int]) -> None:
        stored = self._sessions.pop(key, None)
        if stored is None:
            return
        for token in stored.actions:
            self._token_index.pop(token, None)

    @staticmethod
    def _snapshot(stored: _StoredNovaSession) -> NovaSession:
        return NovaSession(
            id=stored.id,
            owner_id=stored.owner_id,
            telegram_user_id=stored.telegram_user_id,
            chat_id=stored.chat_id,
            access_version=stored.access_version,
            canonical_message_id=stored.canonical_message_id,
            tier=stored.tier,
            created_at=stored.created_at,
            expires_at=stored.expires_at,
            question_in_progress=stored.question_in_progress,
            last_action_id=stored.last_action_id,
        )

    @staticmethod
    def _random_token() -> str:
        return secrets.token_urlsafe(18)

    @staticmethod
    def _validate_binding(
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        access_version: int,
        canonical_message_id: int | None,
        tier: AccessTier,
    ) -> None:
        if owner_id <= 0 or telegram_user_id <= 0 or chat_id <= 0:
            raise ValueError("owner and private chat identifiers must be positive")
        if access_version <= 0 or (canonical_message_id is not None and canonical_message_id <= 0):
            raise ValueError("invalid Nova session version or message")
        if tier not in ACCESS_TIERS or tier == BLOCKED:
            raise ValueError("blocked or unsupported access tier")
