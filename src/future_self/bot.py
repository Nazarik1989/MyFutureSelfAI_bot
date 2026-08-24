import asyncio
import logging
import re
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from hashlib import blake2s
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy import update as sql_update
from telegram import (
    BotCommand,
    BotCommandScopeAllPrivateChats,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonCommands,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ChatType
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    TypeHandler,
    filters,
)
from telegram.warnings import PTBUserWarning

from .access import BLOCKED, FULL_ACCESS_TIERS, GUEST, AccessService, is_full_access_tier
from .access_handlers import GUEST_COMMANDS, AccessHandlers
from .actions import (
    ActionCommandRouter,
    ActionOutcome,
    ActionRoute,
    DraftAction,
    DraftActionService,
)
from .ai import WEEKLY_REVIEW_EXTRACTION_TIMEOUT_SECONDS, AIService
from .callback_ui import edit_callback_screen
from .collection_commands import CollectionCommandRouter
from .collection_handlers import CollectionHandlers
from .collections_service import LifeCollectionService
from .config import Settings
from .conversation import ConversationContextService, ConversationSnapshot
from .dates import DateResolver
from .db import Database
from .doctor_prep import DoctorVisitPrepService
from .doctor_search import DoctorSearchService
from .domain import (
    ONBOARDING_QUESTIONS,
    FocusService,
    IntentRouter,
    OnboardingFlow,
    PendingIntent,
    TodayApplicationSnapshot,
    normalize_display_name,
)
from .drafts import DraftInboxService
from .guest_access import GuestQuotaPolicy, GuestSessionService
from .health import (
    METRIC_LABELS,
    HealthService,
    prolonged_weakness_message,
    urgent_safety_message,
)
from .image_generation import ImageGenerationService, create_image_generation_service
from .inbox import InboxLifecycleService
from .knowledge import KnowledgeQuotaPolicy, KnowledgeService
from .knowledge_handlers import KnowledgeHandlers
from .knowledge_storage import KnowledgeAssetStore
from .lab_handlers import LabHandlers
from .labs import LabDocumentService, LabUploadSessionStore
from .location import LocationService, location_from_user, parse_location
from .models import (
    DraftInboxItem,
    Goal,
    HealthReminderPreference,
    InboxItem,
    Routine,
    User,
    VisionCompanionPreference,
    VisionProfile,
)
from .natural_commands import NaturalAction, NaturalCommandRouter
from .navigation import NavigationFlowStore
from .navigation_handlers import NavigationHandlers
from .nova import NovaSessionStore, is_explicit_nova_invocation
from .nova_companion_handlers import NovaCompanionHandlers
from .nova_handlers import NovaHandlers
from .nova_memory import (
    NovaMemoryApplicationSnapshot,
    NovaMemoryService,
    NovaMemoryValidationError,
)
from .nova_memory_application import (
    NovaMemoryProjection,
    NovaMemoryProjectionError,
    build_nova_memory_projection,
)
from .nova_memory_flow import (
    NovaMemoryFlowStore,
    NovaMemoryIntentKind,
    classify_nova_memory_intent,
)
from .nova_memory_handlers import NOVA_MEMORY_ACCESS_CHANGED_TEXT, NovaMemoryHandlers
from .recurring_reminders import (
    RecurringReminderDelivery,
    RecurringTaskReminderEngine,
    RecurringTaskReminderService,
)
from .reminder_flow import ReminderFlowStore
from .reminder_handlers import ReminderHandlers, ReminderVoiceGateState
from .reminder_intent import ReminderIntentParser
from .reminders import TaskReminderEngine
from .repositories import (
    CheckInRepository,
    GoalRepository,
    OnboardingRepository,
    RoutineRepository,
    UserRepository,
)
from .scheduler import JobQueueScheduler
from .schemas import IntentResult, ParsedThought, TemporalResolution, VisionSummary
from .system_actions import SystemActionRoute, SystemActionRouter
from .task_handlers import TaskHandlers
from .tasks import TaskService
from .timezones import TimezoneCandidate, TimezoneResolver, timezone_candidate_text
from .transcription import TranscriptionError, TranscriptionService
from .vision import VisionService
from .vision_companion import VisionCompanionService
from .vision_handlers import VisionHandlers
from .vision_images import VisionImageService, VisionImageSessionStore
from .vision_references import VisionReferenceService, VisionReferenceSessionStore
from .vision_renderer import (
    VisionBoardRenderer,
    VisionRenderLimiter,
    VisionRenderSessionStore,
)
from .weekly_review import WeeklyReviewService
from .weekly_review_flow import WeeklyReviewCapabilityStore, WeeklyReviewPolicy
from .weekly_review_handlers import WEEKLY_REVIEW_RETRY_TEXT, WeeklyReviewHandlers
from .workspace_access import WorkspaceAccessService
from .workspace_handlers import WorkspaceHandlers

logger = logging.getLogger(__name__)

ONBOARDING_INPUT, PROFILE_CONFIRM = range(2)
_ONBOARDING_META_KEY = "__onboarding_flow__"
_PENDING_ONBOARDING_TIMEZONE = "pending_timezone"
_PENDING_TIMEZONE_UPDATE = "pending_timezone_update"
_ACTIVE_ONBOARDING_STATUSES = frozenset({"in_progress", "awaiting_confirmation"})
_NOVA_MEMORY_APPLICATION_ACCESS_ATTR = "_nova_memory_application_access_generation"
_NOVA_MEMORY_APPLICATION_DRAIN_TIMEOUT_SECONDS = 30.0
_NOVA_MEMORY_APPLICATION_CANCEL_TIMEOUT_SECONDS = 5.0
_NOVA_MEMORY_APPLICATION_CANCEL_RETRY_SECONDS = 0.1
_WEEKLY_REVIEW_DRAIN_TIMEOUT_SECONDS = 31.0
_WEEKLY_REVIEW_CANCEL_TIMEOUT_SECONDS = 5.0
_WEEKLY_REVIEW_CANCEL_RETRY_SECONDS = 0.1
_WEEKLY_REVIEW_PROCESSING_RECOVERY_MARGIN_SECONDS = 5.0
EVENING_WORKED, EVENING_FAILED, EVENING_ENERGY, EVENING_OBSTACLE, EVENING_TOMORROW = range(10, 15)
(
    HEALTH_ENERGY,
    HEALTH_SLEEP,
    HEALTH_MOOD,
    HEALTH_STRESS,
    HEALTH_PHYSICAL,
    HEALTH_SYMPTOMS,
) = range(20, 26)
(
    DOCTOR_REASON,
    DOCTOR_DURATION,
    DOCTOR_SYMPTOMS,
    DOCTOR_MEDICATIONS,
    DOCTOR_QUESTIONS,
) = range(30, 35)
LABELS = {"idea": "идея", "task": "задача", "desire": "желание", "note": "заметка"}
ACTION_LABELS = {
    "idea": "идею",
    "task": "задачу",
    "desire": "желание",
    "note": "заметку",
}
NAVIGATION = ReplyKeyboardMarkup([["Назад", "Пропустить"], ["Отменить"]], resize_keyboard=True)

NOVA_MEMORY_APPLICATION_CHANGED_TEXT = (
    "🧬 Память Nova изменилась, пока я готовила ответ.\nПовтори вопрос — я учту актуальную версию."
)
NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT = (
    "Не удалось безопасно применить память Nova.\nПовтори вопрос чуть позже."
)


class _NovaMemoryApplicationDrainError(RuntimeError):
    """Raised when tracked delivery tasks ignore the bounded shutdown cancellation."""


class _WeeklyReviewDrainError(RuntimeError):
    """Raised when a tracked weekly lifecycle ignores bounded shutdown cancellation."""


@dataclass(frozen=True, slots=True)
class _NovaMemoryApplicationFence:
    telegram_actor_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    tier: str
    access_version: int = field(repr=False)
    collection_revision: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _NovaMemoryAccessGeneration:
    telegram_actor_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    tier: str
    access_version: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class _NovaMemoryCallbackBinding:
    chat_id: int = field(repr=False)
    message_id: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class _NovaMemoryPreparedAnswer:
    text: str = field(repr=False)
    fence: _NovaMemoryApplicationFence
    projection: NovaMemoryProjection | None = field(default=None, repr=False)


type _NovaMemoryApplicationResult = Literal[
    "ready",
    "access_changed",
    "memory_changed",
    "unavailable",
    "provider_failed",
]


def _truncate_utf16(value: str, max_units: int) -> str:
    encoded = value.encode("utf-16-le")
    if len(encoded) <= max_units * 2:
        return value
    clipped = encoded[: (max_units - 1) * 2].decode("utf-16-le", errors="ignore")
    return clipped.rstrip() + "…"


def _conversation_handler(**kwargs: object) -> ConversationHandler:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="If 'per_message=False', 'CallbackQueryHandler'.*",
            category=PTBUserWarning,
        )
        return ConversationHandler(**kwargs)


def log_safe_failure(event: str, exc: BaseException | None, *, user_id: int | None = None) -> None:
    """Log operational metadata without provider messages, prompts, audio, or secrets."""
    error_type = type(exc).__name__ if exc else "Unknown"
    if user_id is None:
        logger.error("%s error_type=%s", event, error_type)
    else:
        logger.error("%s error_type=%s user_id=%s", event, error_type, user_id)


class FutureSelfBot(
    AccessHandlers,
    LabHandlers,
    VisionHandlers,
    TaskHandlers,
    CollectionHandlers,
    WorkspaceHandlers,
    KnowledgeHandlers,
    WeeklyReviewHandlers,
    NovaCompanionHandlers,
    NovaMemoryHandlers,
    ReminderHandlers,
    NovaHandlers,
    NavigationHandlers,
):
    def __init__(
        self,
        settings: Settings,
        db: Database,
        ai: AIService,
        transcription: TranscriptionService,
        image_generation: ImageGenerationService | None = None,
    ):
        self.settings = settings
        self.db = db
        self.access_service = AccessService(db)
        self.guest_quota_policy = GuestQuotaPolicy.from_settings(settings)
        self.guest_session_service = GuestSessionService(db, self.guest_quota_policy)
        self.guest_quota_service = self.guest_session_service.quota
        self._access_scope_cache = {}
        self.ai = ai
        self.transcription = transcription
        self.image_generation = image_generation or create_image_generation_service(settings)
        self.draft_service = DraftInboxService(
            db,
            settings.inbox_draft_ttl_minutes,
            task_date_event_hour=settings.task_date_event_hour,
            task_reminder_lead_minutes=settings.task_reminder_lead_minutes,
        )
        self.action_service = DraftActionService(self.draft_service)
        self.action_router = ActionCommandRouter()
        self.system_action_router = SystemActionRouter()
        self.inbox_lifecycle = InboxLifecycleService(db)
        self.natural_command_router = NaturalCommandRouter(
            enable_workspace_access=getattr(settings, "enable_workspace_access", False)
        )
        self.navigation_flow_sessions = NavigationFlowStore()
        self.nova_sessions = NovaSessionStore()
        self._nova_ui_lock = asyncio.Lock()
        self._nova_launch_lock = asyncio.Lock()
        self.nova_memory_service = NovaMemoryService(
            db,
            max_items=settings.nova_memory_max_items,
        )
        self.nova_memory_sessions = NovaMemoryFlowStore()
        self._nova_memory_ui_lock = asyncio.Lock()
        self._nova_memory_launch_lock = asyncio.Lock()
        self._nova_memory_application_tasks: set[asyncio.Task[bool]] = set()
        self._nova_memory_application_ui_locks: weakref.WeakValueDictionary[
            tuple[int, int], asyncio.Lock
        ] = weakref.WeakValueDictionary()
        self.weekly_review_service = WeeklyReviewService(
            db,
            review_weekday=settings.weekly_review_weekday,
        )
        self.weekly_review_policy = WeeklyReviewPolicy(
            enabled=bool(getattr(settings, "enable_weekly_review", False)),
            admin_only=bool(getattr(settings, "weekly_review_admin_only", True)),
        )
        self.weekly_review_capabilities = WeeklyReviewCapabilityStore()
        self._weekly_review_launch_lock = asyncio.Lock()
        self._reply_keyboard_owner_lock = asyncio.Lock()
        self._weekly_review_tasks: set[asyncio.Task[Any]] = set()
        self._weekly_review_ui_locks: weakref.WeakValueDictionary[tuple[int, int], asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self.reminder_sessions = ReminderFlowStore()
        self._reminder_ui_lock = asyncio.Lock()
        self._reminder_launch_lock = asyncio.Lock()
        self._reminder_now_provider = lambda: datetime.now(UTC)
        self.conversation = ConversationContextService(
            db,
            settings.conversation_context_messages,
            settings.conversation_context_ttl_hours,
            settings.draft_focus_ttl_minutes,
            settings.system_action_ttl_minutes,
        )
        self._init_nova_companion()
        self.date_resolver = DateResolver()
        self.reminder_intent_parser = ReminderIntentParser()
        self.recurring_reminder_service = RecurringTaskReminderService(
            db,
            grace_minutes=settings.recurring_task_reminder_grace_minutes,
            lease_seconds=settings.task_reminder_lease_seconds,
        )
        self.task_service = TaskService(
            db,
            date_event_hour=settings.task_date_event_hour,
            reminder_lead_minutes=settings.task_reminder_lead_minutes,
            date_resolver=self.date_resolver,
        )
        self.collection_command_router = CollectionCommandRouter()
        self.collection_service = LifeCollectionService(
            db,
            action_ttl=timedelta(minutes=settings.collection_action_ttl_minutes),
            input_ttl=timedelta(minutes=settings.collection_input_ttl_minutes),
            context_ttl=timedelta(minutes=settings.collection_context_ttl_minutes),
            task_date_event_hour=settings.task_date_event_hour,
        )
        self.workspace_service = WorkspaceAccessService(db)
        self.knowledge_service = KnowledgeService(
            db,
            quota_policy=KnowledgeQuotaPolicy(
                max_source_bytes=settings.knowledge_max_source_bytes,
                max_extracted_bytes=settings.knowledge_extraction_max_text_bytes,
                daily_ingest_bytes_per_user=settings.knowledge_daily_ingest_bytes_per_user,
                storage_bytes_per_user=settings.knowledge_storage_quota_bytes_per_user,
                daily_sources_per_user=settings.knowledge_daily_sources_per_user,
                max_pending_jobs_per_user=settings.knowledge_max_pending_jobs_per_user,
                daily_ingest_bytes_per_space=settings.knowledge_daily_ingest_bytes_per_space,
                storage_bytes_per_space=settings.knowledge_storage_quota_bytes_per_space,
                daily_sources_per_space=settings.knowledge_daily_sources_per_space,
                max_pending_jobs_per_space=settings.knowledge_max_pending_jobs_per_space,
            ),
        )
        self.knowledge_storage = (
            KnowledgeAssetStore(
                Path(settings.knowledge_asset_root),
                max_source_bytes=settings.knowledge_max_source_bytes,
                max_extracted_bytes=settings.knowledge_extraction_max_text_bytes,
                min_free_bytes=settings.runtime_min_free_bytes,
            )
            if getattr(settings, "enable_knowledge_capture", False)
            else None
        )
        self.intent_router = IntentRouter(ai, settings.intent_confidence_threshold)
        self.focus_service = FocusService(db, ai)
        self.health_service = HealthService(db)
        self.location_service = LocationService(db)
        self.timezone_resolver = TimezoneResolver(ai)
        self.vision_service = VisionService(db)
        self.vision_companion_service = VisionCompanionService(db)
        self.vision_image_service = VisionImageService(db)
        self.vision_image_sessions = VisionImageSessionStore()
        self.vision_reference_service = VisionReferenceService(db)
        self.vision_reference_sessions = VisionReferenceSessionStore()
        self.lab_documents = LabDocumentService(db)
        self.lab_uploads = LabUploadSessionStore()
        self.vision_renderer = VisionBoardRenderer()
        self.vision_render_sessions = VisionRenderSessionStore()
        self.vision_render_limiter = VisionRenderLimiter()
        self.doctor_prep_service = DoctorVisitPrepService(
            db,
            task_date_event_hour=settings.task_date_event_hour,
            task_reminder_lead_minutes=settings.task_reminder_lead_minutes,
        )
        self.doctor_search_service = DoctorSearchService(
            db,
            task_date_event_hour=settings.task_date_event_hour,
            task_reminder_lead_minutes=settings.task_reminder_lead_minutes,
        )
        self.scheduler: JobQueueScheduler | None = None
        self.reminder_engine: TaskReminderEngine | None = None
        self.recurring_reminder_engine: RecurringTaskReminderEngine | None = None
        self._guest_maintenance_task: asyncio.Task[None] | None = None

    @property
    def voice_enabled(self) -> bool:
        return self.settings.enable_voice and getattr(self.transcription, "enabled", True)

    def build(self) -> Application:
        app = (
            Application.builder()
            .token(self.settings.telegram_bot_token)
            .post_init(self._post_init)
            .post_stop(self._post_stop)
            .post_shutdown(self._post_shutdown)
            .build()
        )
        # This assistant handles profiles, health notes and reminders. Telegram
        # group/channel replies would disclose that data to other chat members,
        # so stop every non-private update before any feature handler sees it.
        app.add_handler(TypeHandler(Update, self.private_chat_guard), group=-5)
        app.add_handler(TypeHandler(Update, self.access_gate), group=-4)
        # Destructive natural-language controls must win over every stateful
        # text flow (onboarding, Labs, Vision, health, doctor, and so on).
        # Keeping this in its own group also means a non-matching phrase can
        # continue to the normal flow without being consumed.
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.system_action_text_gate),
            group=-3,
        )
        gated_public_commands = [
            "menu",
            "today",
            "week",
            "inbox",
            "tasks",
            "collections",
            "vision",
            "health",
            "checkin",
            "health_edit",
            "doctor",
            "doctor_find",
            "doctor_prepare",
            "doctor_prepare_edit",
            "doctor_preparations",
            "cleanup_drafts",
            "location",
            "timezone",
            "profile",
            "drafts",
            "last_saved",
            "evening",
            "labs",
            "mynova",
        ]
        if getattr(self.settings, "enable_workspace_access", False):
            gated_public_commands.extend(("spaces", "workspaces"))
        if getattr(self.settings, "enable_knowledge_hub", False):
            gated_public_commands.append("knowledge")
        if getattr(self.settings, "enable_knowledge_capture", False):
            gated_public_commands.append("capture")
        gated_public_commands.append("help")
        app.add_handler(CommandHandler("cancel", self.nova_cancel_gate), group=-3)
        app.add_handler(
            CommandHandler(
                gated_public_commands,
                self.navigation_public_command_gate,
            ),
            group=-2,
        )
        # Commands that are not public navigation entries (for example /drafts)
        # must not escape an unfinished, durably restored onboarding flow.
        app.add_handler(
            MessageHandler(filters.COMMAND, self.onboarding_command_gate),
            group=-2,
        )
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.navigation_text_gate),
            group=-2,
        )
        app.add_handler(
            MessageHandler(
                filters.PHOTO | filters.Document.ALL | filters.VOICE | filters.AUDIO,
                self.nova_non_text_gate,
            ),
            group=-2,
        )
        app.add_handler(
            CallbackQueryHandler(self.knowledge_other_callback_gate),
            group=-2,
        )
        # Vision drafts must keep receiving text/voice even while another PTB
        # ConversationHandler is paused. Group -1 routes only an active owner/chat
        # draft and then stops the lower feature group.
        app.add_handler(CommandHandler("vision", self.vision_command_gate), group=-1)
        app.add_handler(CommandHandler("labs", self.labs_command_gate), group=-1)
        app.add_handler(
            CallbackQueryHandler(self.labs_action, pattern=r"^labs:"),
            group=-1,
        )
        app.add_handler(
            CallbackQueryHandler(self.vision_callback_gate, pattern=r"^vision:"),
            group=-1,
        )
        app.add_handler(CommandHandler("cancel", self.labs_cancel_gate), group=-1)
        app.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.ALL, self.labs_media_gate),
            group=-1,
        )
        app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self.labs_voice_gate),
            group=-1,
        )
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.labs_text_gate),
            group=-1,
        )
        # Profile callbacks belong to the per-user/per-chat conversation, not to
        # individual message IDs. PTB warns about this intentional configuration.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="If 'per_message=False', 'CallbackQueryHandler'.*",
                category=PTBUserWarning,
            )
            onboarding = ConversationHandler(
                entry_points=[
                    CommandHandler("start", self.start),
                    CommandHandler("onboarding", self.start),
                    CallbackQueryHandler(
                        self.nova_onboarding_entry,
                        pattern=r"^nova:action:onboarding:",
                    ),
                    CallbackQueryHandler(
                        self.navigation_onboarding_entry,
                        pattern=r"^nav:action:onboarding$",
                    ),
                ],
                states={
                    ONBOARDING_INPUT: [
                        CommandHandler("back", self.onboarding_back),
                        CommandHandler("skip", self.onboarding_skip),
                        MessageHandler(filters.Regex("^Назад$"), self.onboarding_back),
                        MessageHandler(filters.Regex("^Пропустить$"), self.onboarding_skip),
                        MessageHandler(filters.VOICE | filters.AUDIO, self.voice),
                        CallbackQueryHandler(
                            self.onboarding_timezone_action,
                            pattern=r"^onboarding:timezone:",
                        ),
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self.onboarding_answer),
                    ],
                    PROFILE_CONFIRM: [
                        CallbackQueryHandler(self.profile_action, pattern=r"^profile:")
                    ],
                },
                fallbacks=[
                    CommandHandler("cancel", self.cancel_onboarding),
                    MessageHandler(filters.Regex("^Отменить$"), self.cancel_onboarding),
                    CallbackQueryHandler(self.navigation_action, pattern=r"^nav:flow:"),
                ],
                allow_reentry=True,
            )
        evening = _conversation_handler(
            entry_points=[
                CommandHandler("evening", self.evening_start),
                CallbackQueryHandler(
                    self.nova_evening_entry,
                    pattern=r"^nova:action:evening:",
                ),
                CallbackQueryHandler(
                    self.navigation_evening_entry,
                    pattern=r"^nav:action:evening$",
                ),
            ],
            states={
                EVENING_WORKED: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.evening_worked)
                ],
                EVENING_FAILED: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.evening_failed)
                ],
                EVENING_ENERGY: [MessageHandler(filters.Regex("^[1-5]$"), self.evening_energy)],
                EVENING_OBSTACLE: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.evening_obstacle)
                ],
                EVENING_TOMORROW: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.evening_tomorrow)
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_evening),
                CallbackQueryHandler(self.navigation_action, pattern=r"^nav:flow:"),
            ],
        )
        health_checkin = _conversation_handler(
            entry_points=[
                CommandHandler("checkin", self.health_checkin_start),
                CommandHandler("health_edit", self.health_checkin_start),
                CallbackQueryHandler(
                    self.nova_health_entry,
                    pattern=r"^nova:action:checkin:",
                ),
                CallbackQueryHandler(
                    self.navigation_health_entry,
                    pattern=r"^nav:action:checkin$",
                ),
            ],
            states={
                HEALTH_ENERGY: [
                    MessageHandler(filters.Regex(r"^(?:10|[0-9])$"), self.health_energy),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.health_invalid_rating),
                ],
                HEALTH_SLEEP: [
                    MessageHandler(filters.Regex(r"^(?:10|[0-9])$"), self.health_sleep),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.health_invalid_rating),
                ],
                HEALTH_MOOD: [
                    MessageHandler(filters.Regex(r"^(?:10|[0-9])$"), self.health_mood),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.health_invalid_rating),
                ],
                HEALTH_STRESS: [
                    MessageHandler(filters.Regex(r"^(?:10|[0-9])$"), self.health_stress),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.health_invalid_rating),
                ],
                HEALTH_PHYSICAL: [
                    MessageHandler(filters.Regex(r"^(?:10|[0-9])$"), self.health_physical),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.health_invalid_rating),
                ],
                HEALTH_SYMPTOMS: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.health_symptoms)
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_health_checkin),
                CallbackQueryHandler(self.navigation_action, pattern=r"^nav:flow:"),
            ],
        )
        doctor_prepare = _conversation_handler(
            entry_points=[
                CommandHandler("doctor_prepare", self.doctor_prepare_start),
                CommandHandler("doctor_prepare_edit", self.doctor_prepare_start),
                CallbackQueryHandler(
                    self.nova_doctor_entry,
                    pattern=r"^nova:action:doctor_prepare:",
                ),
                CallbackQueryHandler(
                    self.navigation_doctor_entry,
                    pattern=r"^nav:action:doctor_prepare$",
                ),
            ],
            states={
                DOCTOR_REASON: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.doctor_prepare_reason)
                ],
                DOCTOR_DURATION: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.doctor_prepare_duration)
                ],
                DOCTOR_SYMPTOMS: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.doctor_prepare_symptoms)
                ],
                DOCTOR_MEDICATIONS: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.doctor_prepare_medications)
                ],
                DOCTOR_QUESTIONS: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.doctor_prepare_questions)
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_doctor_prepare),
                CallbackQueryHandler(self.navigation_action, pattern=r"^nav:flow:"),
            ],
            allow_reentry=True,
        )
        app.add_handler(onboarding)
        app.add_handler(evening)
        app.add_handler(health_checkin)
        app.add_handler(doctor_prepare)
        app.add_handler(CommandHandler("menu", self.menu_command))
        app.add_handler(CommandHandler("mynova", self.mynova_command))
        app.add_handler(CommandHandler("tasks", self.tasks_command))
        app.add_handler(CommandHandler("collections", self.collections_command))
        if getattr(self.settings, "enable_workspace_access", False):
            app.add_handler(CommandHandler(["spaces", "workspaces"], self.spaces_command))
        if getattr(self.settings, "enable_knowledge_hub", False):
            app.add_handler(CommandHandler("knowledge", self.knowledge_command))
        if getattr(self.settings, "enable_knowledge_capture", False):
            app.add_handler(CommandHandler("capture", self.capture_command))
        app.add_handler(CommandHandler("doctor", self.doctor_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("profile", self.profile))
        app.add_handler(CommandHandler("location", self.location_command))
        app.add_handler(CommandHandler("timezone", self.timezone_command))
        app.add_handler(CommandHandler("goals", self.goals_command))
        app.add_handler(CommandHandler("inbox", self.inbox))
        app.add_handler(CommandHandler("drafts", self.drafts_command))
        app.add_handler(CommandHandler("last_saved", self.last_saved_command))
        app.add_handler(CommandHandler("cleanup_drafts", self.cleanup_drafts_command))
        app.add_handler(CommandHandler("today", self.today))
        app.add_handler(CommandHandler("week", self.week_command))
        app.add_handler(CommandHandler("health", self.health_command))
        app.add_handler(CommandHandler("health_delete", self.health_delete_command))
        app.add_handler(CommandHandler("health_reminder_on", self.health_reminder_on))
        app.add_handler(CommandHandler("health_reminder_off", self.health_reminder_off))
        app.add_handler(CommandHandler("doctor_preparations", self.doctor_preparations))
        app.add_handler(CommandHandler("doctor_prepare_show", self.doctor_prepare_show))
        app.add_handler(CommandHandler("doctor_prepare_delete", self.doctor_prepare_delete))
        app.add_handler(CommandHandler("doctor_prepare_task", self.doctor_prepare_task))
        app.add_handler(CommandHandler("doctor_find", self.doctor_find))
        app.add_handler(CommandHandler("doctor_find_task", self.doctor_find_task))
        app.add_handler(CommandHandler("cancel", self.cancel_draft_edit))
        app.add_handler(
            CallbackQueryHandler(
                self.weekly_review_callback,
                pattern=r"^wrev:[A-Za-z0-9_-]+$",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self.nova_companion_callback,
                pattern=r"^ncap:[A-Za-z0-9_-]+$",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self.nova_companion_reminder_callback,
                pattern=r"^nrem:[A-Za-z0-9_-]+$",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self.nova_brain_callback,
                pattern=r"^nbrain:[A-Za-z0-9_-]+$",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self.nova_memory_callback,
                pattern=r"^nmem:[A-Za-z0-9_-]+$",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self.reminder_callback,
                pattern=r"^rmd:[A-Za-z0-9_-]+$",
            )
        )
        app.add_handler(CallbackQueryHandler(self.task_callback, pattern=r"^task:"))
        app.add_handler(CallbackQueryHandler(self.collection_callback, pattern=r"^collection:"))
        if getattr(self.settings, "enable_workspace_access", False):
            app.add_handler(CallbackQueryHandler(self.workspace_callback, pattern=r"^spacei?:"))
        if getattr(self.settings, "enable_knowledge_hub", False):
            app.add_handler(CallbackQueryHandler(self.knowledge_callback, pattern=r"^kh:"))
        # Fallback for a confirmation button sent before a process restart. The
        # ConversationHandler handles it normally while its in-memory state exists.
        app.add_handler(CallbackQueryHandler(self.profile_action, pattern=r"^profile:"))
        app.add_handler(
            CallbackQueryHandler(
                self.onboarding_timezone_action,
                pattern=r"^onboarding:timezone:",
            )
        )
        app.add_handler(CallbackQueryHandler(self.timezone_action, pattern=r"^timezone:update:"))
        app.add_handler(CallbackQueryHandler(self.nova_callback, pattern=r"^nova:"))
        app.add_handler(CallbackQueryHandler(self.navigation_action, pattern=r"^nav:"))
        app.add_handler(CallbackQueryHandler(self.intent_action, pattern=r"^intent:"))
        app.add_handler(CallbackQueryHandler(self.context_action, pattern=r"^context:"))
        app.add_handler(
            CallbackQueryHandler(self.draft_command_confirmation, pattern=r"^draftcmd:")
        )
        app.add_handler(CallbackQueryHandler(self.draft_focus_action, pattern=r"^draftfocus:"))
        app.add_handler(CallbackQueryHandler(self.drafts_action, pattern=r"^drafts:"))
        app.add_handler(CallbackQueryHandler(self.system_draft_action, pattern=r"^sysdraft:"))
        app.add_handler(CallbackQueryHandler(self.inbox_action, pattern=r"^inbox:"))
        app.add_handler(CallbackQueryHandler(self.saved_inbox_action, pattern=r"^ibox:"))
        app.add_handler(CallbackQueryHandler(self.goals_action, pattern=r"^goals:"))
        app.add_handler(CallbackQueryHandler(self.routines_action, pattern=r"^routines:"))
        if getattr(self.settings, "enable_knowledge_capture", False):
            app.add_handler(
                MessageHandler(filters.PHOTO | filters.Document.ALL, self.knowledge_media_gate)
            )
        if getattr(self.settings, "enable_workspace_access", False):
            app.add_handler(
                MessageHandler(filters.StatusUpdate.USERS_SHARED, self.workspace_users_shared)
            )
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.voice))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.text))
        app.add_error_handler(self.error_handler)
        return app

    async def _sync_access_commands(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        telegram_id: int,
        tier: Any,
        access_version: int,
    ) -> None:
        """Freeze the access generation admitted by the perimeter for this update."""

        setattr(
            context,
            _NOVA_MEMORY_APPLICATION_ACCESS_ATTR,
            _NovaMemoryAccessGeneration(
                telegram_actor_id=telegram_id,
                chat_id=chat_id,
                tier=str(tier),
                access_version=access_version,
            ),
        )
        await super()._sync_access_commands(
            context,
            chat_id,
            telegram_id,
            tier,
            access_version,
        )

    @staticmethod
    async def private_chat_guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        chat = update.effective_chat
        if chat is None or chat.type == ChatType.PRIVATE:
            return
        message = (
            "Из соображений приватности бот работает только в личном чате. "
            "Открой диалог с ботом напрямую."
        )
        try:
            if update.callback_query is not None:
                await update.callback_query.answer(message, show_alert=True)
            elif update.effective_message is not None:
                await update.effective_message.reply_text(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_safe_failure("Private-chat notification failed", exc)
        raise ApplicationHandlerStop

    async def _post_init(self, app: Application) -> None:
        self.lab_uploads.cleanup_startup()
        await self.lab_documents.cleanup_confirmations()
        await self.task_service.cleanup_tokens()
        await self.task_service.reconcile()
        await self.collection_service.cleanup()
        if getattr(self.settings, "enable_workspace_access", False):
            await self.workspace_service.cleanup()
        if getattr(self.settings, "enable_knowledge_hub", False):
            await self.knowledge_service.cleanup_expired()
        if self.knowledge_storage is not None:
            self.knowledge_storage.cleanup_staging(
                older_than_seconds=self.settings.knowledge_staging_ttl_minutes * 60
            )
        if self.weekly_review_policy.enabled:
            await self._maintain_weekly_review_state(app.bot, startup_recovery=True)
        try:
            await app.bot.set_my_commands(
                [BotCommand(item.command, item.description) for item in GUEST_COMMANDS],
                scope=BotCommandScopeAllPrivateChats(),
            )
        except TelegramError as exc:
            log_safe_failure("Global command scope setup failed", exc)
        try:
            await app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
        except TelegramError as exc:
            log_safe_failure("Global menu button setup failed", exc)

        async def send(telegram_id: int, text: str) -> int | None:
            message = await app.bot.send_message(chat_id=telegram_id, text=text)
            return getattr(message, "message_id", None)

        async def delete_stale_reminder(telegram_id: int, message_id: int) -> None:
            await app.bot.delete_message(chat_id=telegram_id, message_id=message_id)

        async def send_recurring(delivery: RecurringReminderDelivery) -> int | None:
            message = await app.bot.send_message(
                chat_id=delivery.destination_id,
                text=f"🔔 Ежедневное напоминание\n\n{delivery.title}",
            )
            return getattr(message, "message_id", None)

        async def send_vision_companion(preference_id: int, moment: str) -> None:
            await self._vision_companion_notification(app.bot, preference_id, moment)

        async def send_weekly_review(telegram_id: int, timezone: str) -> None:
            await self.weekly_review_scheduled_notification(
                app.bot,
                telegram_id,
                timezone,
            )

        async def can_send(telegram_id: int) -> bool:
            try:
                return await self.access_service.has_full_access_by_telegram_id(telegram_id)
            except Exception as exc:
                log_safe_failure("Background access check failed", exc)
                return False

        async def can_send_weekly(telegram_id: int) -> bool:
            if not self.weekly_review_policy.enabled:
                return False
            try:
                status = await self.access_service.status(telegram_id)
            except Exception as exc:
                log_safe_failure("Weekly background access check failed", exc)
                return False
            return bool(
                status is not None and self.weekly_review_policy.allows_tier(status.access_tier)
            )

        if isinstance(app, Application) or getattr(app, "post_stop", None) is not None:
            self._start_guest_maintenance(app.bot)

        if app.job_queue is None:
            logger.warning("JobQueue is unavailable; scheduled messages are disabled")
            return
        app.job_queue.run_repeating(
            self.labs_cleanup_job,
            interval=300,
            first=300,
            name="labs:cleanup",
        )
        self.scheduler = JobQueueScheduler(
            app.job_queue,
            send,
            self.settings.morning_hour,
            self.settings.evening_hour,
            self.settings.weekly_review_weekday,
            self.settings.enable_weekly_review,
            send_vision_companion,
            can_send=can_send,
            weekly_review_send=send_weekly_review,
            weekly_can_send=can_send_weekly,
        )
        if self.settings.enable_task_reminders:
            self.reminder_engine = TaskReminderEngine(
                self.db,
                send,
                delete_sent=delete_stale_reminder,
                lease_seconds=self.settings.task_reminder_lease_seconds,
                date_event_hour=self.settings.task_date_event_hour,
                lead_minutes=self.settings.task_reminder_lead_minutes,
            )
            await self.reminder_engine.reconcile_missing()
            await self.reminder_engine.deliver_due()
            self.scheduler.start_task_reminders(
                self.reminder_engine,
                interval_seconds=self.settings.task_reminder_poll_seconds,
            )
            self.recurring_reminder_engine = RecurringTaskReminderEngine(
                self.db,
                send_recurring,
                delete_sent=delete_stale_reminder,
                grace_minutes=self.settings.recurring_task_reminder_grace_minutes,
                lease_seconds=self.settings.task_reminder_lease_seconds,
            )
            await self.recurring_reminder_engine.deliver_due()
            self.scheduler.start_recurring_task_reminders(
                self.recurring_reminder_engine,
                interval_seconds=self.settings.task_reminder_poll_seconds,
            )
        async with self.db.sessions() as session:
            users = (
                await session.scalars(
                    select(User).where(
                        User.onboarding_completed.is_(True),
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                    )
                )
            ).all()
        for user in users:
            self.scheduler.schedule_user(user.telegram_id, user.timezone)
        for preference in await self.health_service.reminder_preferences():
            self.scheduler.schedule_health_reminder(
                user_id=preference.user_id,
                chat_id=preference.telegram_user_id,
                timezone=preference.timezone,
                local_time=preference.local_time,
            )
        for preference in await self.vision_companion_service.enabled_preferences():
            self.scheduler.schedule_vision_companion(preference)

    def _start_guest_maintenance(self, telegram_bot: object) -> None:
        if self._guest_maintenance_task is not None and not self._guest_maintenance_task.done():
            return
        maintenance = self._guest_maintenance_loop(telegram_bot)
        try:
            self._guest_maintenance_task = asyncio.create_task(
                maintenance,
                name="guest-result-maintenance",
            )
        except Exception as exc:
            maintenance.close()
            log_safe_failure("Guest maintenance scheduling failed", exc)

    async def _guest_maintenance_loop(self, telegram_bot: object) -> None:
        await self._cleanup_guest_results_safely()
        await self._cleanup_nova_companion_capabilities_safely()
        await self._maintain_weekly_review_state(telegram_bot, startup_recovery=False)
        await self._recover_guest_demo_results(telegram_bot)
        while True:
            await self._guest_maintenance_wait()
            await self._cleanup_guest_results_safely()
            await self._cleanup_nova_companion_capabilities_safely()
            await self._maintain_weekly_review_state(telegram_bot, startup_recovery=False)

    async def _guest_maintenance_wait(self) -> None:
        await asyncio.sleep(60)

    async def _cleanup_guest_results_safely(self) -> None:
        try:
            await self.guest_session_service.cleanup_undeliverable_results()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_safe_failure("Guest result cleanup failed", exc)

    async def _cleanup_nova_companion_capabilities_safely(self) -> None:
        try:
            await self.nova_companion_captures.cleanup()
            await self.nova_companion_reminders.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=capability_cleanup error_type=%s",
                type(exc).__name__,
            )

    async def _maintain_weekly_review_state(
        self,
        telegram_bot: object,
        *,
        startup_recovery: bool,
    ) -> None:
        """Run one bounded retention/recovery batch without provider work."""

        if not self.weekly_review_policy.enabled:
            return
        allowed_tiers = (
            frozenset({"admin"}) if self.weekly_review_policy.admin_only else FULL_ACCESS_TIERS
        )
        try:
            await self.weekly_review_service.cleanup_expired(allowed_tiers=allowed_tiers)
            if startup_recovery:
                recovered = await self.weekly_review_service.recover_processing_session_snapshots(
                    allowed_tiers=allowed_tiers,
                )
            else:
                current = datetime.now(UTC)
                processing_cutoff = current - timedelta(
                    seconds=(
                        WEEKLY_REVIEW_EXTRACTION_TIMEOUT_SECONDS
                        + _WEEKLY_REVIEW_PROCESSING_RECOVERY_MARGIN_SECONDS
                    )
                )
                recovered = await self.weekly_review_service.recover_processing_session_snapshots(
                    updated_before=processing_cutoff,
                    allowed_tiers=allowed_tiers,
                )
            await self.weekly_review_capabilities.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review maintenance failed operation=maintenance error_type=%s",
                type(exc).__name__,
            )
            return
        context = type("WeeklyRecoveryContext", (), {"bot": telegram_bot})()
        for session in recovered:
            if session.canonical_message_id is None:
                continue
            await self._weekly_review_render(
                context,
                session,
                f"{self._weekly_review_week_heading(session)}\n\n{WEEKLY_REVIEW_RETRY_TEXT}",
                await self._weekly_review_cancel_markup(session),
            )

    async def _stop_guest_maintenance(self) -> None:
        task = self._guest_maintenance_task
        self._guest_maintenance_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log_safe_failure("Guest maintenance stop failed", exc)

    async def _post_stop(self, app: Application) -> None:
        del app
        await self._quiesce_private_delivery_tasks()

    async def _post_shutdown(self, app: Application) -> None:
        del app
        try:
            await self._quiesce_private_delivery_tasks()
        finally:
            await self.image_generation.close()

    async def _quiesce_private_delivery_tasks(self) -> None:
        try:
            await self._drain_private_delivery_tasks()
        finally:
            try:
                await self._stop_guest_maintenance()
            finally:
                await self._drain_private_delivery_tasks()

    async def _drain_private_delivery_tasks(self) -> None:
        results = await asyncio.gather(
            self._drain_weekly_review_tasks(),
            self._drain_nova_memory_application_tasks(),
            self._drain_nova_companion_tasks(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _drain_weekly_review_tasks(self) -> None:
        current = asyncio.current_task()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _WEEKLY_REVIEW_DRAIN_TIMEOUT_SECONDS
        while True:
            pending = self._pending_weekly_review_tasks(current)
            if not pending:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            _done, still_pending = await asyncio.wait(pending, timeout=remaining)
            if still_pending:
                break

        pending = self._pending_weekly_review_tasks(current)
        if not pending:
            return
        logger.warning(
            "Weekly review shutdown operation=drain error_type=TimeoutError pending_count=%s",
            len(pending),
        )
        cancel_deadline = loop.time() + _WEEKLY_REVIEW_CANCEL_TIMEOUT_SECONDS
        while True:
            pending = self._pending_weekly_review_tasks(current)
            if not pending:
                return
            for task in pending:
                task.cancel()
            remaining = cancel_deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.wait(
                pending,
                timeout=min(_WEEKLY_REVIEW_CANCEL_RETRY_SECONDS, remaining),
            )

        pending = self._pending_weekly_review_tasks(current)
        if pending:
            logger.error(
                "Weekly review shutdown operation=terminal_drain error_type=TimeoutError "
                "pending_count=%s",
                len(pending),
            )
            raise _WeeklyReviewDrainError("Weekly review terminal drain timed out")

    def _pending_weekly_review_tasks(
        self,
        current: asyncio.Task[object] | None,
    ) -> set[asyncio.Task[Any]]:
        tasks = self._weekly_review_tasks
        for task in tuple(tasks):
            if task is current or not task.done():
                continue
            try:
                task.result()
            except BaseException:
                pass
            tasks.discard(task)
        return {task for task in tuple(tasks) if task is not current and not task.done()}

    async def _user(self, telegram_id: int) -> User:
        async with self.db.session() as session:
            return await UserRepository(session).get_or_create(
                telegram_id, self.settings.default_timezone
            )

    async def _weekly_review_has_reply_keyboard_owner(self, user: User) -> bool:
        if not user.onboarding_completed:
            async with self.db.sessions() as session:
                onboarding = await OnboardingRepository(session).get(user.id)
            if onboarding is not None and onboarding.status in _ACTIVE_ONBOARDING_STATUSES:
                return True
        if getattr(self.settings, "enable_workspace_access", False):
            pending = await self.workspace_service.pending_input(user.id, user.telegram_id)
            if pending is not None:
                return True
        return False

    @staticmethod
    def _onboarding_public_answers(answers: dict[str, object]) -> dict[str, str]:
        keys = {key for key, _question, _required in ONBOARDING_QUESTIONS}
        result = {
            key: value for key, value in answers.items() if key in keys and isinstance(value, str)
        }
        if display_name := result.get("display_name"):
            result["display_name"] = normalize_display_name(display_name, clip_legacy=True)
        return result

    @staticmethod
    def _bounded_vision_summary(summary: VisionSummary) -> VisionSummary:
        return VisionSummary(
            summary=_truncate_utf16(summary.summary, 1_600),
            values=[_truncate_utf16(value, 100) for value in summary.values[:6]],
            desired_identity=[
                _truncate_utf16(value, 120) for value in summary.desired_identity[:6]
            ],
            constraints=[_truncate_utf16(value, 120) for value in summary.constraints[:6]],
            motivation_style=(
                _truncate_utf16(summary.motivation_style, 120) if summary.motivation_style else None
            ),
        )

    @staticmethod
    def _onboarding_meta(answers: dict[str, object]) -> dict[str, object]:
        value = answers.get(_ONBOARDING_META_KEY)
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _onboarding_delivery_key(update: Update) -> str | None:
        update_id = getattr(update, "update_id", None)
        if update_id is not None:
            return f"update:{update_id}"
        message_id = getattr(update.effective_message, "message_id", None)
        chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
        if message_id is None or chat_id is None:
            return None
        return f"message:{chat_id}:{message_id}"

    async def _restore_onboarding_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> tuple[int, int, str, dict[str, object]] | None:
        user = await self._user(update.effective_user.id)
        if user.onboarding_completed:
            context.user_data.pop("onboarding_user_id", None)
            context.user_data.pop("onboarding_detached", None)
            context.user_data.pop("vision_summary", None)
            return None
        async with self.db.sessions() as session:
            state = await OnboardingRepository(session).get(user.id)
            if state is None or state.status not in _ACTIVE_ONBOARDING_STATUSES:
                return None
            snapshot = (user.id, state.current_step, state.status, dict(state.answers))
        if context.user_data.get("onboarding_user_id") != user.id:
            context.user_data["onboarding_user_id"] = user.id
            context.user_data["onboarding_detached"] = True
        return snapshot

    async def onboarding_persistent_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        *,
        force: bool = False,
        navigation_update: Any | None = None,
    ) -> tuple[int, str | None] | None:
        """Resume DB-backed onboarding before generic routing after a restart."""
        attached = context.user_data.get("onboarding_user_id") is not None
        detached = bool(context.user_data.get("onboarding_detached"))
        restored = await self._restore_onboarding_context(update, context)
        if restored is None:
            return None
        _user_id, step, _status, _answers = restored
        navigation = self.natural_command_router.route(text)
        if navigation is not None and navigation.action != "help":
            screen_update = navigation_update or update
            await self._prompt_navigation_flow(
                screen_update.effective_message,
                update,
                "onboarding",
            )
            action = "flow"
            state = PROFILE_CONFIRM if step >= len(ONBOARDING_QUESTIONS) else ONBOARDING_INPUT
            return state, action
        if attached and not detached and step < len(ONBOARDING_QUESTIONS) and not force:
            return None
        if attached and not detached and step >= len(ONBOARDING_QUESTIONS):
            # The group -2 gate, rather than ConversationHandler, now owns this
            # update. Keep subsequent updates on the same durable path so stale
            # in-memory PROFILE_CONFIRM state cannot leak text to generic routing.
            context.user_data["onboarding_detached"] = True
        navigation_answer = text.strip().casefold()
        if navigation_answer == "назад":
            return await self.onboarding_back(update, context), None
        if navigation_answer == "пропустить":
            if step >= len(ONBOARDING_QUESTIONS):
                return await self._present_onboarding_summary(update, context), None
            return await self.onboarding_skip(update, context), None
        if navigation_answer == "отменить":
            return await self.cancel_onboarding(update, context), None
        if step >= len(ONBOARDING_QUESTIONS):
            return await self._present_onboarding_summary(update, context), None
        # Use the recognized text supplied by the caller. For ordinary text it
        # is the Telegram body; voice can pass its STT result here too.
        return await self._accept_onboarding_answer(update, context, text), None

    async def onboarding_command_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        command = (update.effective_message.text or "").split(maxsplit=1)[0]
        command = command.split("@", maxsplit=1)[0].casefold()
        if command in {"/start", "/onboarding"}:
            return
        attached = context.user_data.get("onboarding_user_id") is not None
        detached = bool(context.user_data.get("onboarding_detached"))
        restored = await self._restore_onboarding_context(update, context)
        if restored is None:
            flow = await self._active_navigation_flow(update, context)
            if command == "/cancel":
                if flow is not None:
                    await self.nova_memory_clear_current(update)
                elif await self.nova_memory_public_command_gate(update, context):
                    raise ApplicationHandlerStop
                return
            if flow is not None:
                await self.reminder_clear_current(update)
                await self.nova_memory_clear_current(update)
                await self._prompt_navigation_flow(update.effective_message, update, flow)
                raise ApplicationHandlerStop
            if await self.nova_memory_public_command_gate(update, context):
                raise ApplicationHandlerStop
            return
        if command == "/help":
            await self.help_command(update, context)
            raise ApplicationHandlerStop
        if command == "/menu":
            user = await self._user(update.effective_user.id)
            await self._send_after_reply_keyboard_cleanup(
                update.effective_message,
                "Главное меню\n\nЧто хочешь сделать?",
                self._root_keyboard(user.access_tier),
            )
            raise ApplicationHandlerStop
        if attached and not detached and restored[1] >= len(ONBOARDING_QUESTIONS):
            context.user_data["onboarding_detached"] = True
            detached = True
        if (
            attached
            and not detached
            and restored[1] < len(ONBOARDING_QUESTIONS)
            and command in {"/back", "/skip", "/cancel"}
        ):
            # Let the active ConversationHandler update its own in-memory state.
            return
        if command == "/back":
            await self.onboarding_back(update, context)
        elif command == "/skip":
            if restored[1] >= len(ONBOARDING_QUESTIONS):
                await self._present_onboarding_summary(update, context)
            else:
                await self.onboarding_skip(update, context)
        elif command == "/cancel":
            await self.cancel_onboarding(update, context)
        else:
            await self._prompt_navigation_flow(update.effective_message, update, "onboarding")
        raise ApplicationHandlerStop

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await self.weekly_review_clear_current(update)
        await self.nova_memory_clear_current(update)
        await self.reminder_clear_current(update)
        await self.nova_clear_current(update)
        user = await self._user(update.effective_user.id)
        if user.access_tier == GUEST:
            await self.show_guest_root(update)
            return ConversationHandler.END
        if user.access_tier == BLOCKED or not is_full_access_tier(user.access_tier):
            await self.show_blocked_screen(update)
            return ConversationHandler.END
        if await self.workspace_start_invitation(update, context, access_user=user):
            return ConversationHandler.END
        if user.onboarding_completed:
            context.user_data.pop("onboarding_user_id", None)
            context.user_data.pop("onboarding_detached", None)
            context.user_data.pop("vision_summary", None)
            display_name = (
                normalize_display_name(user.display_name, clip_legacy=True)
                if user.display_name
                else "друг"
            )
            await self._send_after_reply_keyboard_cleanup(
                update.effective_message,
                f"С возвращением, {display_name}!",
                InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Открыть главное меню", callback_data="nav:root")]]
                ),
            )
            return ConversationHandler.END
        async with self._reply_keyboard_owner_lock:
            # This lock is shared with weekly proactive delivery. The durable
            # owner is published before the reply keyboard and neither can be
            # interleaved with a weekly ReplyKeyboardRemove.
            async with self.db.session() as session:
                state = await OnboardingRepository(session).get_or_create(user.id)
                if state.status == "cancelled":
                    state.status = "in_progress"
                step = max(0, min(state.current_step, len(ONBOARDING_QUESTIONS)))
                state.current_step = step
            context.user_data["onboarding_user_id"] = user.id
            context.user_data.pop("onboarding_detached", None)
            intro = (
                "Все ответы сохранены. Восстанавливаю итоговый профиль."
                if step >= len(ONBOARDING_QUESTIONS)
                else (
                    f"Я — «{self.settings.bot_persona_name}». Продолжим с шага {step + 1} из "
                    f"{len(ONBOARDING_QUESTIONS)}. Можно вернуться, пропустить необязательное или отменить."
                )
            )
            await update.effective_message.reply_text(intro)
            if step >= len(ONBOARDING_QUESTIONS):
                return await self._present_onboarding_summary(update, context)
            await self._ask_question(update, step)
            return ONBOARDING_INPUT

    async def _ask_question(self, update: Update, step: int) -> None:
        _, question, required = ONBOARDING_QUESTIONS[step]
        suffix = "" if required else " (можно пропустить)"
        await update.effective_message.reply_text(
            f"Шаг {step + 1} из {len(ONBOARDING_QUESTIONS)}\n{question}{suffix}",
            reply_markup=NAVIGATION,
        )

    async def _state(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> tuple[int, int, dict[str, object]]:
        user_id = context.user_data.get("onboarding_user_id")
        telegram_user = getattr(update, "effective_user", None)
        if telegram_user is not None:
            user = await self._user(telegram_user.id)
            user_id = user.id
            context.user_data["onboarding_user_id"] = user_id
        if user_id is None:
            raise ValueError("Регистрация не была начата. Запусти /start.")
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(int(user_id))
            return int(user_id), state.current_step, dict(state.answers)

    async def onboarding_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        return await self._accept_onboarding_answer(
            update, context, update.effective_message.text or ""
        )

    async def _accept_onboarding_answer(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
    ) -> int:
        try:
            user_id, step, answers = await self._state(update, context)
            if step >= len(ONBOARDING_QUESTIONS):
                return await self._present_onboarding_summary(update, context)
            delivery_key = self._onboarding_delivery_key(update)
            metadata = self._onboarding_meta(answers)
            if delivery_key is not None and metadata.get("last_delivery_key") == delivery_key:
                await update.effective_message.reply_text(
                    "Этот ответ уже сохранён — повторно его не учитываю."
                )
                await self._ask_question(update, step)
                return ONBOARDING_INPUT
            question_key = ONBOARDING_QUESTIONS[step][0]
            if question_key == "timezone":
                try:
                    candidate = await self.timezone_resolver.resolve(text)
                except ValueError:
                    raise
                except Exception as exc:
                    log_safe_failure("Onboarding timezone resolution failed", exc, user_id=user_id)
                    await update.effective_message.reply_text(
                        "Сейчас не удалось определить часовой пояс. "
                        "Ответ не сохранён — попробуй ещё раз немного позже."
                    )
                    return ONBOARDING_INPUT
                return await self._present_onboarding_timezone_candidate(
                    update,
                    user_id,
                    step,
                    answers,
                    candidate,
                    delivery_key=delivery_key,
                )
            answers = OnboardingFlow.answer(answers, step, text)
            if question_key == "location":
                parse_location(text)
                metadata = self._onboarding_meta(answers)
                metadata.pop("location_autofilled", None)
                metadata.pop("skip_location_once", None)
                answers[_ONBOARDING_META_KEY] = metadata
        except ValueError as exc:
            await update.effective_message.reply_text(str(exc))
            return ONBOARDING_INPUT
        except Exception as exc:
            log_safe_failure("Onboarding answer read failed", exc)
            await update.effective_message.reply_text(
                "Не удалось сохранить ответ. Шаг не изменён — попробуй ещё раз или продолжи через /start."
            )
            return ONBOARDING_INPUT
        try:
            return await self._advance_onboarding(
                update,
                context,
                user_id,
                step,
                answers,
                delivery_key=delivery_key,
            )
        except Exception as exc:
            log_safe_failure("Onboarding answer save failed", exc, user_id=user_id)
            await update.effective_message.reply_text(
                "Не удалось сохранить ответ. Шаг не изменён — попробуй ещё раз или продолжи через /start."
            )
            return ONBOARDING_INPUT

    async def _present_onboarding_timezone_candidate(
        self,
        update: Update,
        user_id: int,
        step: int,
        answers: dict[str, object],
        candidate: TimezoneCandidate,
        *,
        delivery_key: str | None,
    ) -> int:
        token = uuid4().hex[:16]
        stale_step = None
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user_id)
            if state.current_step != step or ONBOARDING_QUESTIONS[step][0] != "timezone":
                stale_step = state.current_step
            else:
                current_answers = dict(state.answers)
                metadata = self._onboarding_meta(current_answers)
                previous = metadata.get(_PENDING_ONBOARDING_TIMEZONE)
                if (
                    delivery_key is not None
                    and isinstance(previous, dict)
                    and previous.get("delivery_key") == delivery_key
                ):
                    token = str(previous.get("token") or token)
                metadata[_PENDING_ONBOARDING_TIMEZONE] = {
                    "token": token,
                    "timezone": candidate.timezone,
                    "city": candidate.city,
                    "source": candidate.source,
                    "delivery_key": delivery_key,
                }
                current_answers[_ONBOARDING_META_KEY] = metadata
                state.answers = current_answers
        if stale_step is not None:
            await update.effective_message.reply_text(
                "Этот ответ относится к предыдущему шагу. Показываю текущий вопрос."
            )
            if stale_step < len(ONBOARDING_QUESTIONS):
                await self._ask_question(update, stale_step)
                return ONBOARDING_INPUT
            return await self._present_onboarding_summary(update, SimpleNamespace(user_data={}))
        await update.effective_message.reply_text(
            timezone_candidate_text(candidate),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Верно",
                            callback_data=f"onboarding:timezone:confirm:{token}",
                        ),
                        InlineKeyboardButton(
                            "✏️ Другой город",
                            callback_data=f"onboarding:timezone:retry:{token}",
                        ),
                    ]
                ]
            ),
        )
        return ONBOARDING_INPUT

    async def onboarding_timezone_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        query = update.callback_query
        parts = (query.data or "").split(":")
        if len(parts) != 4 or parts[:2] != ["onboarding", "timezone"]:
            await query.answer("Эта кнопка больше не действует", show_alert=True)
            return ConversationHandler.END
        action, token = parts[2], parts[3]
        user = await self._user(update.effective_user.id)
        context.user_data["onboarding_user_id"] = user.id
        async with self.db.sessions() as session:
            state = await OnboardingRepository(session).get(user.id)
            if state is None:
                pending = None
                step = 0
                answers: dict[str, object] = {}
            else:
                step = state.current_step
                answers = dict(state.answers)
                pending = self._onboarding_meta(answers).get(_PENDING_ONBOARDING_TIMEZONE)
        if (
            state is None
            or step >= len(ONBOARDING_QUESTIONS)
            or ONBOARDING_QUESTIONS[step][0] != "timezone"
            or not isinstance(pending, dict)
            or pending.get("token") != token
            or action not in {"confirm", "retry"}
        ):
            await query.answer("Этот выбор устарел. Продолжи через /start.", show_alert=True)
            return ConversationHandler.END if user.onboarding_completed else ONBOARDING_INPUT

        if action == "retry":
            metadata = self._onboarding_meta(answers)
            metadata.pop(_PENDING_ONBOARDING_TIMEZONE, None)
            answers[_ONBOARDING_META_KEY] = metadata
            async with self.db.session() as session:
                latest = await OnboardingRepository(session).get_or_create(user.id)
                if latest.current_step == step:
                    latest_answers = dict(latest.answers)
                    latest_metadata = self._onboarding_meta(latest_answers)
                    current = latest_metadata.get(_PENDING_ONBOARDING_TIMEZONE)
                    if isinstance(current, dict) and current.get("token") == token:
                        latest_metadata.pop(_PENDING_ONBOARDING_TIMEZONE, None)
                        latest_answers[_ONBOARDING_META_KEY] = latest_metadata
                        latest.answers = latest_answers
            await query.answer()
            await query.edit_message_reply_markup(reply_markup=None)
            await self._ask_question(update, step)
            return ONBOARDING_INPUT

        timezone = str(pending.get("timezone") or "")
        try:
            timezone = ZoneInfo(timezone).key
        except Exception:
            await query.answer(
                "Часовой пояс больше недоступен. Выбери город заново.", show_alert=True
            )
            return ONBOARDING_INPUT
        city = pending.get("city")
        location = parse_location(str(city)) if city else None
        answers["timezone"] = timezone
        metadata = self._onboarding_meta(answers)
        metadata.pop(_PENDING_ONBOARDING_TIMEZONE, None)
        if location is not None:
            answers["location"] = location.label
            metadata["skip_location_once"] = True
        answers[_ONBOARDING_META_KEY] = metadata
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        return await self._advance_onboarding(
            update,
            context,
            user.id,
            step,
            answers,
            delivery_key=f"timezone-confirm:{token}",
        )

    async def onboarding_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id, step, answers = await self._state(update, context)
        delivery_key = self._onboarding_delivery_key(update)
        if (
            delivery_key is not None
            and self._onboarding_meta(answers).get("last_delivery_key") == delivery_key
        ):
            await update.effective_message.reply_text(
                "Эта команда уже обработана — повторно шаг не меняю."
            )
            if step < len(ONBOARDING_QUESTIONS):
                await self._ask_question(update, step)
                return ONBOARDING_INPUT
            return await self._present_onboarding_summary(update, context)
        try:
            answers = OnboardingFlow.answer(answers, step, None)
        except ValueError as exc:
            await update.effective_message.reply_text(str(exc))
            return ONBOARDING_INPUT
        return await self._advance_onboarding(
            update, context, user_id, step, answers, delivery_key=delivery_key
        )

    async def _advance_onboarding(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
        step: int,
        answers: dict[str, object],
        *,
        delivery_key: str | None = None,
    ) -> int:
        next_step = OnboardingFlow.next_step(step)
        metadata = self._onboarding_meta(answers)
        if (
            next_step < len(ONBOARDING_QUESTIONS)
            and ONBOARDING_QUESTIONS[next_step][0] == "location"
            and answers.get("location")
            and metadata.pop("skip_location_once", False)
        ):
            metadata["location_autofilled"] = True
            answers[_ONBOARDING_META_KEY] = metadata
            next_step = OnboardingFlow.next_step(next_step)
        outcome = "advanced"
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user_id)
            current_metadata = self._onboarding_meta(dict(state.answers))
            if (
                delivery_key is not None
                and current_metadata.get("last_delivery_key") == delivery_key
            ):
                outcome = "duplicate"
                next_step = state.current_step
            elif state.current_step != step:
                outcome = "stale"
                next_step = state.current_step
            else:
                metadata = self._onboarding_meta(answers)
                metadata.pop("summary", None)
                if delivery_key is not None:
                    metadata["last_delivery_key"] = delivery_key
                answers[_ONBOARDING_META_KEY] = metadata
                state.answers = answers
                state.current_step = next_step
                state.status = (
                    "awaiting_confirmation"
                    if next_step >= len(ONBOARDING_QUESTIONS)
                    else "in_progress"
                )
        if outcome == "duplicate":
            await update.effective_message.reply_text(
                "Этот ответ уже сохранён — повторно его не учитываю."
            )
        elif outcome == "stale":
            await update.effective_message.reply_text(
                "Предыдущий ответ уже перевёл регистрацию дальше. "
                "Повтори ответ на показанный ниже текущий вопрос."
            )
        else:
            await update.effective_message.reply_text("Ответ сохранён ✓")
        if next_step < len(ONBOARDING_QUESTIONS):
            await self._ask_question(update, next_step)
            return ONBOARDING_INPUT
        return await self._present_onboarding_summary(update, context)

    async def _resolve_onboarding_summary(
        self, user_id: int, stored_answers: dict[str, object]
    ) -> VisionSummary:
        metadata = self._onboarding_meta(stored_answers)
        cached = metadata.get("summary")
        summary = None
        if isinstance(cached, dict):
            try:
                summary = VisionSummary.model_validate(cached)
            except ValueError:
                pass
        answers = self._onboarding_public_answers(stored_answers)
        if summary is None:
            summary = await self.ai.summarize_vision(answers)
        summary = self._bounded_vision_summary(summary)
        bounded_payload = summary.model_dump(mode="json")
        if cached == bounded_payload:
            return summary
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user_id)
            latest = dict(state.answers)
            if (
                state.current_step < len(ONBOARDING_QUESTIONS)
                or self._onboarding_public_answers(latest) != answers
            ):
                raise RuntimeError("Onboarding answers changed while summary was prepared")
            latest_metadata = self._onboarding_meta(latest)
            latest_metadata["summary"] = bounded_payload
            latest[_ONBOARDING_META_KEY] = latest_metadata
            state.answers = latest
            state.status = "awaiting_confirmation"
        return summary

    async def _present_onboarding_summary(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        user_id, _step, stored_answers = await self._state(update, context)
        answers = self._onboarding_public_answers(stored_answers)
        await update.effective_message.reply_text(
            "Вопросы регистрации закончились — старые кнопки ответа убраны.",
            reply_markup=ReplyKeyboardRemove(),
        )
        try:
            summary = await self._resolve_onboarding_summary(user_id, stored_answers)
        except Exception as exc:
            log_safe_failure("Vision summary failed", exc, user_id=user_id)
            context.user_data["onboarding_detached"] = True
            await update.effective_message.reply_text(
                "Ответы сохранены, но итоговый профиль сейчас не удалось подготовить. "
                "Регистрация не потеряна: попробуй /start позже."
            )
            return ConversationHandler.END
        context.user_data["vision_summary"] = summary.model_dump()
        location_label = None
        if location_value := answers.get("location"):
            try:
                location_label = parse_location(location_value).label
            except ValueError:
                location_label = None
        await update.effective_message.reply_text(
            "Все вопросы пройдены. Проверь итог перед завершением регистрации.\n\n"
            + self._profile_text(summary, location_label=location_label),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Завершить регистрацию", callback_data="profile:confirm"
                        ),
                        InlineKeyboardButton("Изменить ответы", callback_data="profile:edit"),
                    ]
                ]
            ),
        )
        return PROFILE_CONFIRM

    async def onboarding_back(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id, step, answers = await self._state(update, context)
        previous = OnboardingFlow.previous_step(step)
        delivery_key = self._onboarding_delivery_key(update)
        duplicate = False
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user_id)
            metadata = self._onboarding_meta(dict(state.answers))
            if delivery_key is not None and metadata.get("last_delivery_key") == delivery_key:
                duplicate = True
                previous = state.current_step
            elif state.current_step == step:
                metadata.pop("summary", None)
                if delivery_key is not None:
                    metadata["last_delivery_key"] = delivery_key
                answers[_ONBOARDING_META_KEY] = metadata
                state.answers = answers
                state.current_step = previous
                state.status = "in_progress"
            else:
                previous = state.current_step
        if duplicate:
            await update.effective_message.reply_text(
                "Эта команда уже обработана — повторно шаг не меняю."
            )
        if previous >= len(ONBOARDING_QUESTIONS):
            return await self._present_onboarding_summary(update, context)
        await self._ask_question(update, previous)
        return ONBOARDING_INPUT

    async def cancel_onboarding(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = context.user_data.get("onboarding_user_id")
        if user_id:
            async with self.db.session() as session:
                state = await OnboardingRepository(session).get_or_create(int(user_id))
                state.status = "cancelled"
        context.user_data.pop("onboarding_user_id", None)
        context.user_data.pop("onboarding_detached", None)
        context.user_data.pop("vision_summary", None)
        await update.effective_message.reply_text(
            "Онбординг остановлен. Ответы сохранены; продолжить — /start.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    async def profile_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        user = await self._user(update.effective_user.id)
        if user.onboarding_completed:
            await query.answer("Регистрация уже завершена", show_alert=True)
            await self._send_navigation_root(query.message, tier=user.access_tier)
            return ConversationHandler.END
        async with self.db.sessions() as session:
            state = await OnboardingRepository(session).get(user.id)
            if (
                state is None
                or state.status != "awaiting_confirmation"
                or state.current_step < len(ONBOARDING_QUESTIONS)
            ):
                await query.answer("Эта кнопка больше не действует", show_alert=True)
                return ConversationHandler.END
            stored_answers = dict(state.answers)
        was_attached = context.user_data.get("onboarding_user_id") == user.id
        context.user_data["onboarding_user_id"] = user.id
        if not was_attached:
            context.user_data["onboarding_detached"] = True
        if query.data == "profile:edit":
            await query.answer()
            async with self.db.session() as session:
                state = await OnboardingRepository(session).get_or_create(user.id)
                state.current_step = 0
                state.status = "in_progress"
                answers = dict(state.answers)
                metadata = self._onboarding_meta(answers)
                metadata.pop("summary", None)
                answers[_ONBOARDING_META_KEY] = metadata
                state.answers = answers
            context.user_data.pop("vision_summary", None)
            await query.edit_message_text("Хорошо, пройдём ответы ещё раз.")
            await self._ask_question(update, 0)
            return ONBOARDING_INPUT
        if query.data != "profile:confirm":
            await query.answer("Эта кнопка больше не действует", show_alert=True)
            return ConversationHandler.END
        answers = self._onboarding_public_answers(stored_answers)
        required_answers = {key for key, _question, required in ONBOARDING_QUESTIONS if required}
        if any(not answers.get(key) for key in required_answers):
            await query.answer("Ответы неполные — продолжи через /start", show_alert=True)
            return ConversationHandler.END
        await query.answer()
        try:
            cached_summary = context.user_data.get("vision_summary")
            summary = (
                VisionSummary.model_validate(cached_summary)
                if isinstance(cached_summary, dict)
                else await self._resolve_onboarding_summary(user.id, stored_answers)
            )
        except Exception as exc:
            log_safe_failure("Vision confirmation resume failed", exc, user_id=user.id)
            await query.message.reply_text(
                "Ответы сохранены, но завершить регистрацию сейчас не удалось. "
                "Попробуй эту кнопку ещё раз или продолжи через /start."
            )
            return PROFILE_CONFIRM
        try:
            async with self.db.session() as session:
                locked_owner_id = await session.scalar(
                    sql_update(User)
                    .where(User.id == user.id)
                    .values(updated_at=User.updated_at)
                    .returning(User.id)
                )
                if locked_owner_id is None:
                    raise RuntimeError("Onboarding owner disappeared")
                stored_user = await session.get(User, locked_owner_id)
                if stored_user is None:
                    raise RuntimeError("Onboarding owner disappeared")
                from .repositories import ProfileRepository

                await ProfileRepository(session).upsert(stored_user, answers, summary)
                stored_user.display_name = (
                    normalize_display_name(display_name, clip_legacy=True)
                    if (display_name := answers.get("display_name"))
                    else None
                )
                if timezone_value := answers.get("timezone"):
                    from .domain import canonical_timezone

                    await self.recurring_reminder_service.refresh_profile_timezone_in_session(
                        session,
                        stored_user.id,
                        canonical_timezone(timezone_value),
                    )
                if location_value := answers.get("location"):
                    location = parse_location(location_value)
                    stored_user.location_city = location.city
                    stored_user.location_fallback_city = location.fallback_city
                state = await OnboardingRepository(session).get_or_create(user.id)
                state.status = "completed"
                telegram_id = stored_user.telegram_id
                timezone_name = stored_user.timezone
        except Exception as exc:
            log_safe_failure("Profile confirmation failed", exc, user_id=user.id)
            await query.message.reply_text(
                "Не удалось завершить регистрацию. Ответы сохранены — "
                "попробуй кнопку ещё раз или продолжи через /start."
            )
            return PROFILE_CONFIRM
        if self.scheduler:
            try:
                self.scheduler.schedule_user(telegram_id, timezone_name)
            except Exception as exc:
                log_safe_failure("Onboarding schedule refresh failed", exc, user_id=user.id)
        context.user_data.pop("onboarding_user_id", None)
        context.user_data.pop("onboarding_detached", None)
        context.user_data.pop("vision_summary", None)
        await query.edit_message_text(
            "Регистрация завершена ✓ Профиль сохранён. "
            "Главное меню уже доступно; цели можно принять, изменить или отложить."
        )
        await query.message.reply_text(
            "Кнопки регистрации убраны — дальше можно пользоваться обычным меню.",
            reply_markup=ReplyKeyboardRemove(),
        )
        await self._send_navigation_root(query.message, tier=user.access_tier)
        await self._propose_goals(query, user.id, summary)
        return ConversationHandler.END

    async def _propose_goals(self, query: object, user_id: int, summary: VisionSummary) -> None:
        try:
            proposals = await self.ai.propose_goals(summary)
            async with self.db.session() as session:
                goals = await GoalRepository(session).replace_proposals(user_id, proposals.goals)
            lines = ["Предлагаю цели (их можно отложить или удалить):"]
            buttons = []
            for goal in goals:
                lines.append(f"\n{goal.id}. {goal.title} — {goal.progress_criterion}")
                buttons.append(
                    [
                        InlineKeyboardButton(
                            f"Удалить {goal.id}", callback_data=f"goals:delete:{goal.id}"
                        ),
                        InlineKeyboardButton(
                            f"Отложить {goal.id}", callback_data=f"goals:postpone:{goal.id}"
                        ),
                        InlineKeyboardButton(
                            f"Переименовать {goal.id}", callback_data=f"goals:rename:{goal.id}"
                        ),
                    ]
                )
            buttons.append(
                [InlineKeyboardButton("Подтвердить оставшиеся", callback_data="goals:confirm")]
            )
            await query.message.reply_text(
                "".join(lines), reply_markup=InlineKeyboardMarkup(buttons)
            )
        except Exception as exc:
            log_safe_failure("Goal proposal failed", exc, user_id=user_id)
            await query.message.reply_text(
                "Профиль сохранён, но цели пока не удалось предложить. Попробуй позже."
            )

    async def goals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        async with self.db.sessions() as session:
            profile = await session.scalar(
                select(VisionProfile).where(VisionProfile.user_id == user.id)
            )
        if profile is None:
            await update.effective_message.reply_text("Сначала создай Vision Profile через /start.")
            return
        summary = VisionSummary(
            summary=profile.summary,
            values=profile.values,
            desired_identity=profile.desired_identity,
            constraints=profile.constraints,
            motivation_style=profile.motivation_style,
        )
        await update.effective_message.reply_text("Готовлю новый набор целей…")
        await self._propose_goals(update, user.id, summary)

    async def goals_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        user = await self._user(update.effective_user.id)
        parts = query.data.split(":")
        async with self.db.session() as session:
            if len(parts) == 3:
                goal = await session.get(Goal, int(parts[2]))
                if goal is None or goal.user_id != user.id or goal.status != "proposed":
                    await query.answer("Цель уже обработана", show_alert=True)
                    return
                await query.answer()
                if parts[1] == "rename":
                    context.user_data["rename_goal_id"] = goal.id
                    await query.message.reply_text(
                        f"Пришли новое название для цели «{goal.title}»."
                    )
                    return
                goal.status = "deleted" if parts[1] == "delete" else "postponed"
                await query.message.reply_text(
                    f"Цель «{goal.title}» {('удалена' if parts[1] == 'delete' else 'отложена')}."
                )
                return
            goals = list(
                (
                    await session.scalars(
                        select(Goal).where(Goal.user_id == user.id, Goal.status == "proposed")
                    )
                ).all()
            )
            if not 3 <= len(goals) <= 5:
                await query.answer("Для старта нужно оставить от 3 до 5 целей", show_alert=True)
                return
            await query.answer()
            for goal in goals:
                goal.status = "active"
        await query.edit_message_text("Цели подтверждены. Подбираю до трёх лёгких рутин…")
        try:
            from .schemas import GoalProposal, GoalProposals

            bundle = GoalProposals(
                goals=[
                    GoalProposal.model_validate(
                        {key: getattr(goal, key) for key in GoalProposal.model_fields}
                    )
                    for goal in goals
                ]
            )
            proposals = await self.ai.propose_routines(bundle)
            async with self.db.session() as session:
                attached_goals = list(
                    (
                        await session.scalars(
                            select(Goal).where(Goal.user_id == user.id, Goal.status == "active")
                        )
                    ).all()
                )
                routines = await RoutineRepository(session).create_for_goals(
                    user.id, attached_goals, proposals.routines
                )
            text = "\n".join(
                f"• {r.normal_version}\n  На сложный день: {r.minimum_version}" for r in routines
            )
            await query.message.reply_text(
                text or "Подходящие рутины пока не найдены.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Активировать рутины", callback_data="routines:confirm"
                            ),
                            InlineKeyboardButton("Отложить", callback_data="routines:postpone"),
                        ]
                    ]
                ),
            )
        except Exception as exc:
            log_safe_failure("Routine proposal failed", exc, user_id=user.id)
            await query.message.reply_text("Цели сохранены; рутины можно подобрать позже.")

    async def routines_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        user = await self._user(update.effective_user.id)
        status = "active" if query.data.endswith("confirm") else "postponed"
        async with self.db.session() as session:
            routines = (
                await session.scalars(
                    select(Routine).where(Routine.user_id == user.id, Routine.status == "proposed")
                )
            ).all()
            if not routines:
                await query.answer("Рутины уже обработаны", show_alert=True)
                return
            await query.answer()
            for routine in routines[:3]:
                routine.status = status
        await query.edit_message_text(
            "Рутины активированы. План на сегодня — /today."
            if status == "active"
            else "Рутины отложены. Цели остаются активными."
        )

    async def profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        async with self.db.sessions() as session:
            profile = await session.scalar(
                select(VisionProfile).where(VisionProfile.user_id == user.id)
            )
        if not profile:
            await update.effective_message.reply_text("Профиль ещё не создан. Начать — /start.")
            return
        await update.effective_message.reply_text(
            self._profile_text(
                VisionSummary(
                    summary=profile.summary,
                    values=profile.values,
                    desired_identity=profile.desired_identity,
                    constraints=profile.constraints,
                    motivation_style=profile.motivation_style,
                ),
                location_label=(location.label if (location := location_from_user(user)) else None),
            )
        )

    @staticmethod
    def _profile_text(profile: VisionSummary, *, location_label: str | None = None) -> str:
        profile = FutureSelfBot._bounded_vision_summary(profile)
        rendered = (
            f"Твой Vision Profile:\n{profile.summary}\n\n"
            f"Ценности: {', '.join(profile.values) or 'не указаны'}\n"
            f"Желаемая идентичность: {', '.join(profile.desired_identity) or 'не указана'}\n"
            f"Ограничения: {', '.join(profile.constraints) or 'не указаны'}\n"
            f"Стиль поддержки: {profile.motivation_style or 'не указан'}\n"
            f"Локация: {location_label or 'не настроена — используй /location'}"
        )
        return _truncate_utf16(rendered, 3_400)

    async def location_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        args = getattr(context, "args", [])
        if not args:
            location = location_from_user(user)
            if location is None:
                await update.effective_message.reply_text(
                    "Локация не настроена. Укажи город: /location Саратов. "
                    "Для маршрута с запасным городом: /location Саратов → Энгельс."
                )
                return
            await update.effective_message.reply_text(
                f"Твоя локация: {location.label}. Изменить: /location Новый город."
            )
            return
        try:
            location = await self.location_service.set(
                user_id=user.id,
                telegram_user_id=update.effective_user.id,
                value=" ".join(args),
            )
        except ValueError as exc:
            await update.effective_message.reply_text(str(exc))
            return
        await update.effective_message.reply_text(
            f"Локация сохранена: {location.label}. /doctor_find будет использовать её."
        )

    async def timezone_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        if not user.onboarding_completed:
            await update.effective_message.reply_text(
                "Сначала заверши настройку через /start — часовой пояс входит в неё."
            )
            return
        args = getattr(context, "args", [])
        if not args:
            local_now = datetime.now(UTC).astimezone(ZoneInfo(user.timezone))
            await update.effective_message.reply_text(
                f"Текущий часовой пояс: {user.timezone}\n"
                f"Местное время: {local_now.strftime('%H:%M')}\n\n"
                "Изменить: /timezone Казань или /timezone Берлин, Германия."
            )
            return
        raw = " ".join(args)
        try:
            candidate = await self.timezone_resolver.resolve(raw)
        except ValueError as exc:
            await update.effective_message.reply_text(str(exc))
            return
        except Exception as exc:
            log_safe_failure("Timezone resolution failed", exc, user_id=user.id)
            await update.effective_message.reply_text(
                "Сейчас не удалось определить часовой пояс. Попробуй ещё раз немного позже."
            )
            return

        token = uuid4().hex[:16]
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user.id)
            answers = dict(state.answers)
            metadata = self._onboarding_meta(answers)
            metadata[_PENDING_TIMEZONE_UPDATE] = {
                "token": token,
                "timezone": candidate.timezone,
                "city": candidate.city,
                "source": candidate.source,
            }
            answers[_ONBOARDING_META_KEY] = metadata
            state.answers = answers
        location_note = (
            "\nПосле подтверждения этот город также станет локацией для раздела врача."
            if candidate.city
            else ""
        )
        await update.effective_message.reply_text(
            timezone_candidate_text(candidate) + location_note,
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Сохранить",
                            callback_data=f"timezone:update:confirm:{token}",
                        ),
                        InlineKeyboardButton(
                            "Отмена",
                            callback_data=f"timezone:update:cancel:{token}",
                        ),
                    ]
                ]
            ),
        )

    async def timezone_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        parts = (query.data or "").split(":")
        if len(parts) != 4 or parts[:2] != ["timezone", "update"]:
            await query.answer("Эта кнопка больше не действует", show_alert=True)
            return
        action, token = parts[2], parts[3]
        user = await self._user(update.effective_user.id)
        async with self.db.sessions() as session:
            state = await OnboardingRepository(session).get(user.id)
            answers = dict(state.answers) if state is not None else {}
            pending = self._onboarding_meta(answers).get(_PENDING_TIMEZONE_UPDATE)
        if (
            not isinstance(pending, dict)
            or pending.get("token") != token
            or action not in {"confirm", "cancel"}
        ):
            await query.answer("Этот выбор устарел. Запусти /timezone заново.", show_alert=True)
            return

        if action == "cancel":
            async with self.db.session() as session:
                state = await OnboardingRepository(session).get_or_create(user.id)
                latest = dict(state.answers)
                metadata = self._onboarding_meta(latest)
                current = metadata.get(_PENDING_TIMEZONE_UPDATE)
                if isinstance(current, dict) and current.get("token") == token:
                    metadata.pop(_PENDING_TIMEZONE_UPDATE, None)
                    latest[_ONBOARDING_META_KEY] = metadata
                    state.answers = latest
            await query.answer()
            await query.edit_message_text("Изменение часового пояса отменено.")
            return

        timezone = str(pending.get("timezone") or "")
        try:
            timezone = ZoneInfo(timezone).key
            location = parse_location(str(pending["city"])) if pending.get("city") else None
        except (KeyError, ValueError):
            await query.answer("Данные устарели. Запусти /timezone заново.", show_alert=True)
            return

        health_schedule: tuple[int, time] | None = None
        companion_schedule: SimpleNamespace | None = None
        stale_selection = False
        async with self.db.session() as session:
            locked_owner_id = await session.scalar(
                sql_update(User)
                .where(
                    User.id == user.id,
                    User.telegram_id == update.effective_user.id,
                )
                .values(updated_at=User.updated_at)
                .returning(User.id)
            )
            stored_user = (
                await session.get(User, locked_owner_id) if locked_owner_id is not None else None
            )
            state = await OnboardingRepository(session).get_or_create(user.id)
            latest = dict(state.answers)
            metadata = self._onboarding_meta(latest)
            current = metadata.get(_PENDING_TIMEZONE_UPDATE)
            if (
                stored_user is None
                or not isinstance(current, dict)
                or current.get("token") != token
            ):
                stale_selection = True
            else:
                await self.recurring_reminder_service.refresh_profile_timezone_in_session(
                    session,
                    stored_user.id,
                    timezone,
                )
                latest["timezone"] = timezone
                if location is not None:
                    stored_user.location_city = location.city
                    stored_user.location_fallback_city = None
                    latest["location"] = location.label
                metadata.pop(_PENDING_TIMEZONE_UPDATE, None)
                latest[_ONBOARDING_META_KEY] = metadata
                state.answers = latest

                health = await session.scalar(
                    select(HealthReminderPreference).where(
                        HealthReminderPreference.user_id == stored_user.id
                    )
                )
                if health is not None:
                    health.timezone = timezone
                    if health.enabled:
                        health_schedule = (health.user_id, health.local_time)
                companion = await session.scalar(
                    select(VisionCompanionPreference).where(
                        VisionCompanionPreference.owner_id == stored_user.id
                    )
                )
                if companion is not None:
                    companion.timezone = timezone
                    if companion.enabled:
                        companion_schedule = SimpleNamespace(
                            id=companion.id,
                            owner_id=companion.owner_id,
                            telegram_user_id=stored_user.telegram_id,
                            timezone=timezone,
                            morning_time=companion.morning_time,
                            evening_time=companion.evening_time,
                            extra_times=tuple(companion.extra_times),
                        )

        if stale_selection:
            await query.answer("Этот выбор уже обработан.", show_alert=True)
            return

        if self.scheduler is not None:
            self.scheduler.schedule_user(user.telegram_id, timezone)
            if health_schedule is not None:
                health_user_id, local_time = health_schedule
                self.scheduler.schedule_health_reminder(
                    user_id=health_user_id,
                    chat_id=user.telegram_id,
                    timezone=timezone,
                    local_time=local_time,
                )
            if companion_schedule is not None:
                self.scheduler.schedule_vision_companion(companion_schedule)
        await query.answer()
        location_text = f"\nЛокация: {location.label}." if location is not None else ""
        await query.edit_message_text(
            f"Часовой пояс обновлён: {timezone}.{location_text}\n"
            "Новые напоминания и ежедневные сценарии будут использовать это местное время."
        )

    async def text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = update.effective_message.text or ""
        if await self._try_system_action(update, context, text):
            return
        if await self.workspace_pending_text(update, text, "text"):
            return
        if await self.collection_pending_text(update, text, "text"):
            return
        if await self.task_pending_text(update):
            return
        if await self._handle_vision_input(update, update.effective_message.text):
            return
        goal_id = context.user_data.pop("rename_goal_id", None)
        if goal_id is not None:
            user = await self._user(update.effective_user.id)
            title = update.effective_message.text.strip()[:200]
            async with self.db.session() as session:
                goal = await session.get(Goal, int(goal_id))
                if goal and goal.user_id == user.id and goal.status == "proposed" and title:
                    goal.title = title
                    await update.effective_message.reply_text("Название цели обновлено.")
                else:
                    await update.effective_message.reply_text("Не удалось переименовать эту цель.")
            return
        if await self.knowledge_pending_text(update, text, "text"):
            return
        await self._route_message(
            update,
            context,
            text,
            "text",
            companion_status_preinvalidated=True,
        )

    async def system_action_text_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Route safe destructive controls before any stateful text handler."""

        text = update.effective_message.text or ""
        status_receipt = self.nova_companion_status_receipt_anchor(
            update.effective_user.id,
            update.effective_chat.id,
        )
        if await self._try_system_action(update, context, text):
            self.nova_companion_invalidate_status_receipt_exact(
                update.effective_user.id,
                update.effective_chat.id,
                expected_receipt=status_receipt,
            )
            raise ApplicationHandlerStop

    async def voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int | None:
        medical_flow = await self._knowledge_medical_flow(update, context)
        if medical_flow is not None:
            await update.effective_message.reply_text(
                "В медицинском сценарии голос не отправляется на распознавание или в LLM. "
                "Ответь текстом либо заверши сценарий через /cancel."
            )
            return
        user_data = getattr(context, "user_data", {})
        blocked_context_flow = any(
            key in user_data
            for key in (
                "health_checkin",
                "doctor_prepare",
                "evening",
                "rename_goal_id",
            )
        )
        if blocked_context_flow:
            await update.effective_message.reply_text(
                "Сейчас активен другой сценарий. Продолжи его ожидаемым текстом или "
                "используй /cancel — аудио не отправлено на распознавание."
            )
            return
        if getattr(self.settings, "enable_knowledge_capture", False):
            # Capture never silently turns voice into Knowledge material. Check its
            # persistent state before Telegram download or STT so expiry and an
            # open preview cannot fall through to the generic assistant pipeline.
            flow = await self._knowledge_specialized_flow(update, context)
            if flow is None:
                user = await self._user(update.effective_user.id)
                state = await self.knowledge_service.capture_state(
                    user.id, update.effective_chat.id
                )
                if state.preview is not None or state.expired_now:
                    await update.effective_message.reply_text(
                        "Capture ожидает текст, ссылку, фото или документ. "
                        "Аудио не отправлено на распознавание; используй preview или /cancel."
                    )
                    return
        voice_user = None
        voice_memory_fence = None
        voice_weekly_session = None
        telegram_user = getattr(update, "effective_user", None)
        effective_chat = getattr(update, "effective_chat", None)
        if (
            getattr(telegram_user, "id", None) is not None
            and getattr(effective_chat, "id", None) is not None
        ):
            voice_user = await self._user(telegram_user.id)
            voice_memory_fence = await self.nova_memory_voice_fence(
                update,
                user=voice_user,
            )
            voice_weekly_session = await self.weekly_review_voice_fence(
                update,
                user=voice_user,
            )
            if self.weekly_review_voice_lookup_failed(voice_weekly_session):
                raise ApplicationHandlerStop
        media = update.effective_message.voice or update.effective_message.audio
        if not self.voice_enabled:
            message = (
                "Голосовой ввод сейчас отключён."
                if not self.settings.enable_voice
                else "Распознавание голосовых временно не настроено. Пришли мысль текстом."
            )
            if voice_user is not None and await self.weekly_review_voice_failure(
                update,
                context,
                expected_user=voice_user,
                expected_session=voice_weekly_session,
                progress=None,
                notice=message,
            ):
                return
            if await self.nova_memory_voice_failure(
                update,
                context,
                fence=voice_memory_fence,
                progress=None,
                notice=message,
            ):
                return
            await update.effective_message.reply_text(message)
            return
        if media.duration and media.duration > self.settings.max_audio_seconds:
            message = "Аудио слишком длинное. Пришли запись короче трёх минут."
            if voice_user is not None and await self.weekly_review_voice_failure(
                update,
                context,
                expected_user=voice_user,
                expected_session=voice_weekly_session,
                progress=None,
                notice=message,
            ):
                return
            if await self.nova_memory_voice_failure(
                update,
                context,
                fence=voice_memory_fence,
                progress=None,
                notice=message,
            ):
                return
            await update.effective_message.reply_text(message)
            return
        if media.file_size and media.file_size > self.settings.max_audio_bytes:
            message = "Аудиофайл слишком большой."
            if voice_user is not None and await self.weekly_review_voice_failure(
                update,
                context,
                expected_user=voice_user,
                expected_session=voice_weekly_session,
                progress=None,
                notice=message,
            ):
                return
            if await self.nova_memory_voice_failure(
                update,
                context,
                fence=voice_memory_fence,
                progress=None,
                notice=message,
            ):
                return
            await update.effective_message.reply_text(message)
            return
        mime = getattr(media, "mime_type", None)
        if mime and not (mime.startswith("audio/") or mime == "application/ogg"):
            message = "Этот формат аудио не поддерживается."
            if voice_user is not None and await self.weekly_review_voice_failure(
                update,
                context,
                expected_user=voice_user,
                expected_session=voice_weekly_session,
                progress=None,
                notice=message,
            ):
                return
            if await self.nova_memory_voice_failure(
                update,
                context,
                fence=voice_memory_fence,
                progress=None,
                notice=message,
            ):
                return
            await update.effective_message.reply_text(message)
            return
        # Freeze the access generation before STT. A downgrade or version bounce
        # while the transcript is being produced must not be hidden by a later
        # repository read.
        if voice_user is None:
            voice_user = await self._user(update.effective_user.id)
            voice_memory_fence = await self.nova_memory_voice_fence(
                update,
                user=voice_user,
            )
            voice_weekly_session = await self.weekly_review_voice_fence(
                update,
                user=voice_user,
            )
            if self.weekly_review_voice_lookup_failed(voice_weekly_session):
                raise ApplicationHandlerStop
        voice_status_receipt = self.nova_companion_status_receipt_anchor(
            update.effective_user.id,
            update.effective_chat.id,
        )
        voice_nova_session = await self.nova_sessions.current(
            owner_id=voice_user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        voice_reminder_session = await self.reminder_sessions.current(
            owner_id=voice_user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        voice_access_version = voice_user.access_version
        progress = await update.effective_message.reply_text("Расшифровываю голосовую мысль…")
        try:
            telegram_file = await media.get_file()
            audio = bytes(await telegram_file.download_as_bytearray())
            if len(audio) > self.settings.max_audio_bytes:
                raise ValueError("Audio exceeds limit")
            filename = getattr(media, "file_name", None) or "voice.ogg"
            text = await self.transcription.transcribe(audio, filename)
        except asyncio.CancelledError:
            self.weekly_review_schedule_voice_cancel_cleanup(
                update,
                context,
                progress,
                expected_user=voice_user,
                expected_session=voice_weekly_session,
            )
            raise
        except (TranscriptionError, TelegramError, ValueError) as exc:
            log_safe_failure("Voice processing failed", exc)
            message = "Не удалось распознать голосовое. Попробуй ещё раз или пришли текст."
            if await self.weekly_review_voice_failure(
                update,
                context,
                expected_user=voice_user,
                expected_session=voice_weekly_session,
                progress=progress,
                notice=message,
            ):
                return
            if await self.nova_memory_voice_failure(
                update,
                context,
                fence=voice_memory_fence,
                progress=progress,
                notice=message,
            ):
                return
            await progress.edit_text(message)
            return
        self.nova_companion_invalidate_status_for_input(
            update.effective_user.id,
            update.effective_chat.id,
            text,
            expected_receipt=voice_status_receipt,
        )
        if await self.nova_memory_voice_pre_route(
            update,
            context,
            progress,
            fence=voice_memory_fence,
        ):
            raise ApplicationHandlerStop
        if await self.weekly_review_voice_pre_route(
            update,
            context,
            progress,
            expected_user=voice_user,
            expected_session=voice_weekly_session,
        ):
            raise ApplicationHandlerStop
        screen_update = self._edited_screen_update(update, progress)
        memory_voice_active = (
            voice_memory_fence is not None and voice_memory_fence.session is not None
        )
        weekly_voice_active = voice_weekly_session is not None
        stateful_voice_active = memory_voice_active or weekly_voice_active
        ownership_update = screen_update if stateful_voice_active else update
        if await self._try_system_action(
            ownership_update,
            context,
            text,
            clear_current_memory=False,
            clear_current_weekly=False,
        ):
            await self._clear_frozen_voice_memory(voice_memory_fence)
            await self._clear_frozen_voice_weekly(voice_weekly_session)
            if not stateful_voice_active:
                await progress.edit_text(f"Я услышал: «{_truncate_utf16(text, 4_000)}»")
            return
        voice_flow = await self._active_navigation_flow(update, context)
        if await self.nova_memory_voice_pre_route(
            update,
            context,
            progress,
            fence=voice_memory_fence,
        ):
            raise ApplicationHandlerStop
        if await self.weekly_review_voice_pre_route(
            update,
            context,
            progress,
            expected_user=voice_user,
            expected_session=voice_weekly_session,
        ):
            raise ApplicationHandlerStop
        try:
            memory_intent = classify_nova_memory_intent(text).kind
        except NovaMemoryValidationError:
            memory_intent = NovaMemoryIntentKind.AWAIT_CONTENT
        if voice_flow == "onboarding" and memory_intent is not NovaMemoryIntentKind.NONE:
            await self._clear_frozen_voice_memory(voice_memory_fence)
            await self._clear_frozen_voice_weekly(voice_weekly_session)
            await self.reminder_clear_current(update)
            await self.nova_clear_current(update)
            await self._prompt_navigation_flow(
                screen_update.effective_message,
                update,
                voice_flow,
            )
            raise ApplicationHandlerStop
        onboarding_result = await self.onboarding_persistent_input(
            ownership_update,
            context,
            text,
            force=True,
            navigation_update=screen_update,
        )
        if onboarding_result is not None:
            await self._clear_frozen_voice_memory(voice_memory_fence)
            await self._clear_frozen_voice_weekly(voice_weekly_session)
            onboarding_state, navigation_action = onboarding_result
            if not stateful_voice_active and navigation_action is None:
                await progress.edit_text("Голос распознан и обработан в регистрации.")
            return onboarding_state
        reminder_voice_state = ReminderVoiceGateState(
            access_expected=is_full_access_tier(voice_user.access_tier)
        )
        if voice_flow is not None:
            await self._clear_frozen_voice_memory(voice_memory_fence)
            await self._clear_frozen_voice_weekly(voice_weekly_session)
            if memory_intent is not NovaMemoryIntentKind.NONE:
                await self.reminder_clear_current(update)
                await self.nova_clear_current(update)
                await self._prompt_navigation_flow(
                    screen_update.effective_message,
                    update,
                    voice_flow,
                )
                raise ApplicationHandlerStop
            await self.reminder_clear_current(update)
            if await self._route_voice_durable_flow(
                ownership_update,
                context,
                voice_flow,
                text,
            ):
                return
            await self._prompt_navigation_flow(
                screen_update.effective_message,
                update,
                voice_flow,
            )
            raise ApplicationHandlerStop
        if await self.weekly_review_voice_gate(
            update,
            context,
            text,
            progress,
            expected_user=voice_user,
            expected_session=voice_weekly_session,
        ):
            raise ApplicationHandlerStop
        if await self.nova_memory_voice_gate(
            update,
            context,
            text,
            progress,
            user=voice_user,
            fence=voice_memory_fence,
        ):
            raise ApplicationHandlerStop
        if await self.reminder_voice_gate(
            update,
            context,
            text,
            progress,
            expected_access_version=voice_access_version,
            expected_session=voice_reminder_session,
            voice_state=reminder_voice_state,
        ):
            return
        if await self.weekly_review_launch_voice_gate(
            update,
            context,
            text,
            progress,
            expected_user=voice_user,
        ):
            return
        if not reminder_voice_state.access_failed:
            natural_command = self.natural_command_router.route(text)
            explicit_unknown = (
                natural_command is None
                and self.natural_command_router.is_explicit_navigation_request(text)
                and self.collection_command_router.route(text) is None
            )
            if natural_command is not None or explicit_unknown:
                action = natural_command.action if natural_command is not None else "help"
                flow = await self._active_navigation_flow(update, context)
                if flow is not None and action == "help":
                    # Natural help words can be legitimate answers to a durable
                    # business flow. Let its existing consumer keep ownership.
                    await self.nova_clear_current(update)
                else:
                    await self.nova_clear_current(update)
                    if flow is not None:
                        await self._prompt_navigation_flow(
                            screen_update.effective_message,
                            update,
                            flow,
                        )
                    else:
                        await self.nova_memory_clear_current(update)
                        await self._handle_natural_command(screen_update, context, action)
                    return
            if await self.workspace_pending_text(ownership_update, text, "voice"):
                if not memory_voice_active:
                    await progress.edit_text("Голос распознан и обработан в пространстве.")
                return
            if await self.collection_pending_text(ownership_update, text, "voice"):
                if not memory_voice_active:
                    await progress.edit_text("Голос распознан и обработан в разделе.")
                return
            if await self.task_pending_text(ownership_update, text):
                if not memory_voice_active:
                    await progress.edit_text("Голос распознан и применён к задаче.")
                return
            if await self._handle_vision_input(ownership_update, text):
                if not memory_voice_active:
                    await progress.edit_text("Голос распознан и добавлен в карточку.")
                return
        if await self.nova_voice_gate(
            update,
            context,
            text,
            progress,
            user=voice_user,
            expected_session=voice_nova_session,
        ):
            return
        if reminder_voice_state.access_failed:
            await self._reminder_edit_access_candidate(progress)
            return
        if self.nova_companion_available_for_actor(voice_user):
            snapshot = await self.conversation.get(
                update.effective_user.id,
                update.effective_chat.id,
            )
            handled = await self.nova_companion_route(
                update,
                context,
                text,
                "voice",
                user=voice_user,
                conversation_snapshot=snapshot,
                delivery_message=progress,
                status_receipt_preinvalidated=True,
            )
            if handled:
                return
        heard_text = _truncate_utf16(text, 4_000)
        await progress.edit_text(f"Я услышал: «{heard_text}»")
        await self._route_message(
            update,
            context,
            text,
            "voice",
            frozen_user=voice_user,
            companion_status_preinvalidated=True,
        )

    async def _route_voice_durable_flow(
        self,
        update: Any,
        context: ContextTypes.DEFAULT_TYPE,
        flow: str,
        text: str,
    ) -> bool:
        """Give an already-persisted flow the STT result on the progress screen."""

        if flow == "workspace":
            return await self.workspace_pending_text(update, text, "voice")
        if flow == "collection_input":
            return await self.collection_pending_text(update, text, "voice")
        if flow == "task_edit":
            return await self.task_pending_text(update, text)
        if flow == "vision":
            return await self._handle_vision_input(update, text)
        if flow == "date_choice":
            user = await self._user(update.effective_user.id)
            snapshot = await self.conversation.get(
                update.effective_user.id,
                update.effective_chat.id,
            )
            selected = self.date_resolver.choose_option(text, snapshot.pending_date_options)
            if selected is None:
                return False
            await self._confirm_pending_date(
                update,
                context,
                user,
                snapshot,
                selected.value,
                "voice",
                input_text=text,
            )
            return True
        if flow in {"draft_action", "draft_edit"}:
            await self._route_message(
                update,
                context,
                text,
                "voice",
                companion_status_preinvalidated=True,
            )
            return True
        return False

    async def _clear_frozen_voice_memory(self, fence: Any | None) -> bool:
        session = getattr(fence, "session", None)
        if session is None:
            return False
        return await self.nova_memory_clear_bound(
            fence.owner_id,
            fence.telegram_user_id,
            fence.chat_id,
            session_id=session.id,
        )

    async def _clear_frozen_voice_weekly(self, session: Any | None) -> bool:
        if session is None or self.weekly_review_voice_lookup_failed(session):
            return False
        return await self._weekly_review_clear_exact(session)

    async def _try_system_action(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        *,
        clear_current_memory: bool = True,
        clear_current_weekly: bool = True,
    ) -> bool:
        """Consume destructive control language before any content flow or LLM."""

        user = await self._user(update.effective_user.id)
        snapshot = await self.conversation.get(
            update.effective_user.id,
            update.effective_chat.id,
        )
        if not snapshot.system_pending_action:
            if await self.nova_memory_owns_text(update, text, user=user):
                return False
            nova_session = await self.nova_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if nova_session is not None or is_explicit_nova_invocation(text):
                return False
        if snapshot.system_pending_action and self.natural_command_router.route(text) is not None:
            # An explicit navigation/read action means the user moved on. Drop
            # only the still-current preview and let normal natural routing run.
            await self.conversation.clear_system_action(
                update.effective_user.id,
                update.effective_chat.id,
                expected_version=snapshot.system_action_version,
            )
            return False
        route = self.system_action_router.route(
            text,
            pending_action=snapshot.system_pending_action,
        )
        if (
            not user.onboarding_completed
            and not snapshot.system_pending_action
            and not self.system_action_router.is_explicit_cleanup_command(text)
            and (
                route.kind in {"clarify", "pending"}
                or route.action
                in {
                    "archive_overdue_tasks",
                    "discard_all_active_drafts",
                    "discard_selected_drafts",
                }
            )
        ):
            # During registration, long free-form answers belong to the current
            # question. Only an unmistakable cleanup command may interrupt that
            # durable flow; narrative wishes such as "хочу научиться удалять..."
            # must continue to onboarding instead of opening a delete preview.
            return False
        if route.kind == "none":
            return False
        if clear_current_memory:
            await self.nova_memory_clear_current(update)
        if clear_current_weekly:
            await self.weekly_review_clear_current(update)
        await self.reminder_clear_current(update)
        await self._handle_system_action_route(update, context, user, snapshot, route)
        return True

    async def _route_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        source: str,
        *,
        frozen_user: User | None = None,
        companion_delivery_message: object | None = None,
        companion_status_preinvalidated: bool = False,
    ) -> None:
        natural_command = self.natural_command_router.route(text)
        if natural_command is not None:
            await self.nova_memory_clear_current(update)
            await self._handle_natural_command(update, context, natural_command.action)
            return
        if await self.handle_collection_natural(update, context, text, source):
            return
        if self.natural_command_router.is_explicit_navigation_request(text):
            await self.nova_memory_clear_current(update)
            await self.help_command(update, context)
            return
        user = frozen_user or await self._user(update.effective_user.id)
        access_generation = self._nova_memory_access_generation(
            context,
            telegram_actor_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            fallback=user,
        )
        application_enabled = self.nova_memory_application_available_for_tier(
            access_generation.tier
        )
        if application_enabled and not self._nova_memory_user_matches_generation(
            user, access_generation
        ):
            await update.effective_message.reply_text(NOVA_MEMORY_ACCESS_CHANGED_TEXT)
            return
        chat_id = update.effective_chat.id
        telegram_user_id = update.effective_user.id
        snapshot = await self.conversation.get(telegram_user_id, chat_id)
        system_route = self.system_action_router.route(
            text, pending_action=snapshot.system_pending_action
        )
        if system_route.kind != "none":
            await self._handle_system_action_route(update, context, user, snapshot, system_route)
            return
        if snapshot.pending_date_options:
            selected = self.date_resolver.choose_option(text, snapshot.pending_date_options)
            if selected is not None:
                await self._confirm_pending_date(
                    update, context, user, snapshot, selected.value, source
                )
                return
        action_route = self.action_router.route(
            text, has_pending_action=bool(snapshot.pending_action)
        )
        if action_route.kind != "none":
            await self._handle_action_route(update, context, user, snapshot, action_route, source)
            return
        editing = await self.draft_service.editing(update.effective_user.id, chat_id)
        if editing is None and await self.nova_companion_route(
            update,
            context,
            text,
            source,
            user=user,
            conversation_snapshot=snapshot,
            delivery_message=companion_delivery_message,
            status_receipt_preinvalidated=companion_status_preinvalidated,
        ):
            return
        relative_reminder = self.date_resolver.resolve_relative_reminder(text, user.timezone)
        if relative_reminder is not None:
            await self.conversation.append(
                telegram_user_id,
                chat_id,
                role="user",
                content=text.strip(),
                source=source,
                intent="relative_reminder",
                topic=relative_reminder.title,
            )
            await self._show_preview(
                update.effective_message,
                user.id,
                telegram_user_id,
                chat_id,
                text.strip(),
                source,
                ParsedThought(
                    kind="task",
                    title=relative_reminder.title,
                    resolved_date=relative_reminder.temporal.resolved_local_date,
                    temporal_resolution=relative_reminder.temporal,
                ),
                include_original=source != "voice",
            )
            return
        date_resolution = self.date_resolver.resolve(text, user.timezone)
        if date_resolution.status == "conflict":
            response = self.date_resolver.conflict_message(date_resolution)
            await self.conversation.set_date_conflict(
                telegram_user_id,
                chat_id,
                [option.model_dump(mode="json") for option in date_resolution.options],
            )
            await self.conversation.append(
                telegram_user_id,
                chat_id,
                role="user",
                content=text.strip(),
                source=source,
                intent="date_conflict",
            )
            await self.conversation.append(
                telegram_user_id,
                chat_id,
                role="assistant",
                content=response,
                source="text",
                intent="clarification",
                topic=snapshot.current_topic,
            )
            await update.effective_message.reply_text(response)
            return
        prompt_context = snapshot.for_prompt()
        prompt_context["date_resolution"] = date_resolution.model_dump(mode="json")
        temporal_resolution = (
            self.date_resolver.temporal_resolution(
                date_resolution.target_date,
                user.timezone,
                text,
                self.date_resolver.extract_local_time(text),
            )
            if date_resolution.status == "resolved" and date_resolution.target_date
            else None
        )
        if application_enabled and not await self._nova_memory_application_user_is_current(user):
            await update.effective_message.reply_text(NOVA_MEMORY_ACCESS_CHANGED_TEXT)
            return
        try:
            if application_enabled:
                result = await self.intent_router.route(
                    text,
                    user.timezone,
                    conversation_context=prompt_context,
                    defer_answer=True,
                )
            else:
                result = await self.intent_router.route(
                    text,
                    user.timezone,
                    conversation_context=prompt_context,
                )
        except Exception as exc:
            if application_enabled:
                logger.warning(
                    "Nova memory application failed stage=route error_type=%s",
                    type(exc).__name__,
                )
            else:
                log_safe_failure("Intent routing failed", exc, user_id=user.id)
            await update.effective_message.reply_text(
                "Не удалось понять сообщение. Ничего не сохранено — попробуй ещё раз."
            )
            return
        session_id = await self.conversation.append(
            telegram_user_id,
            chat_id,
            role="user",
            content=text.strip(),
            source=source,
            intent=result.intent,
            topic=result.topic or result.title,
        )
        if interpretation := self.date_resolver.interpretation_message(date_resolution):
            await update.effective_message.reply_text(interpretation)
            await self.conversation.append(
                telegram_user_id,
                chat_id,
                role="assistant",
                content=interpretation,
                source="text",
                intent="date_interpretation",
                topic=result.topic,
            )
        if editing:
            parsed = self._parsed_from_intent(
                result,
                text,
                fallback_kind=editing.kind,
                resolved_date=date_resolution.target_date,
                temporal_resolution=temporal_resolution,
            )
            revised = await self.draft_service.revise(
                editing.id,
                update.effective_user.id,
                chat_id,
                text.strip(),
                source,
                parsed,
            )
            if not revised.ok:
                await update.effective_message.reply_text(
                    "Эта карточка уже неактуальна. Создай новую."
                )
                return
            await self._send_draft_preview(
                update.effective_message,
                revised.draft,
                include_original=source != "voice",
            )
            await self.conversation.set_active_draft(telegram_user_id, chat_id, revised.draft.id)
            await self._remember_preview(telegram_user_id, chat_id, revised.draft)
            return
        if result.intent in {"conversation", "question"}:
            if not application_enabled:
                answer = result.answer or "Я тебя услышал. Можешь уточнить, чем помочь?"
                if self._is_task_question(text):
                    await self._show_task_choices(update.effective_message, session_id, answer)
                else:
                    await update.effective_message.reply_text(answer)
                await self._append_delivered_answer(
                    telegram_user_id,
                    chat_id,
                    answer,
                    result.topic or snapshot.current_topic,
                )
                return
            prepared = await self._prepare_nova_memory_answer(
                user=user,
                chat_id=chat_id,
                question=text,
                route_answer=result.answer,
                timezone_name=user.timezone,
                conversation_context=prompt_context,
                legacy_default_answer=False,
            )
            if isinstance(prepared, str):
                await update.effective_message.reply_text(
                    self._nova_memory_application_neutral_text(prepared)
                )
                return
            await self._deliver_nova_memory_application_answer(
                context,
                update.effective_message,
                session_id=session_id,
                question=text,
                prepared=prepared,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                topic=result.topic or snapshot.current_topic,
            )
            return
        if result.intent == "unknown":
            await self._show_unknown(update, context, text.strip(), source, result)
            return
        capture_text = text.strip()
        if result.intent == "explicit_capture" and self._is_reference_request(text):
            if snapshot.active_draft:
                draft = await self.draft_service.get(str(snapshot.active_draft["id"]))
                if draft and draft.status == "preview":
                    await self._send_draft_preview(update.effective_message, draft)
                    await self._remember_preview(telegram_user_id, chat_id, draft)
                    return
            candidate = self.conversation.reference_candidate(snapshot)
            if candidate is None:
                clarification = "Не уверен, что именно сохранить. Уточни сообщение или мысль."
                await update.effective_message.reply_text(clarification)
                await self.conversation.append(
                    telegram_user_id,
                    chat_id,
                    role="assistant",
                    content=clarification,
                    source="text",
                    intent="clarification",
                    topic=snapshot.current_topic,
                )
                return
            capture_text = candidate
        parsed = self._parsed_from_intent(
            result,
            capture_text,
            resolved_date=date_resolution.target_date,
            temporal_resolution=temporal_resolution,
        )
        if parsed.resolved_date:
            await self.conversation.set_resolved_date(
                telegram_user_id, chat_id, parsed.resolved_date
            )
        await self._show_preview(
            update.effective_message,
            user.id,
            update.effective_user.id,
            chat_id,
            capture_text,
            source,
            parsed,
            include_original=source != "voice",
        )

    async def _handle_natural_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        action: NaturalAction,
    ) -> None:
        section_actions = {
            "create_task": "tasks",
            "show_records": "records",
            "show_health": "health",
            "show_labs": "health",
            "prepare_doctor": "health",
            "show_settings": "settings",
            "show_timezone": "settings",
            "show_collections": "sections",
            "show_spaces": "sections",
        }
        if section := section_actions.get(action):
            await self._send_navigation_section(update.effective_message, section)
            return
        if action == "show_vision":
            await self.vision_command(update, context)
            return
        handlers = {
            "menu": self.menu_command,
            "show_drafts": self.drafts_command,
            "show_inbox": self.inbox,
            "show_last_saved": self.last_saved_command,
            "show_profile": self.profile,
            "show_today": self.today,
            "show_tasks": self.tasks_command,
            "show_overdue_tasks": self.task_overdue,
            "help": self.help_command,
        }
        if action in {
            "create_space",
            "invite_space_member",
            "show_space_invitations",
            "show_space_members",
        }:
            await self.handle_workspace_natural(update, context, action)
            return
        await handlers[action](update, context)

    async def _confirm_pending_date(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: User,
        snapshot: ConversationSnapshot,
        selected_date: date,
        source: str,
        *,
        input_text: str | None = None,
    ) -> None:
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        # The persisted option is authoritative and was calendar-validated by DateResolver.
        candidates = await self.draft_service.active_previews(telegram_user_id, chat_id)
        if len(candidates) > 1:
            await update.effective_message.reply_text(
                "Есть несколько актуальных карточек. Уточни, к какой относится дата."
            )
            return
        await self.conversation.set_resolved_date(telegram_user_id, chat_id, selected_date)
        selected_expression = input_text or update.effective_message.text or "Выбрана дата"
        await self.conversation.append(
            telegram_user_id,
            chat_id,
            role="user",
            content=selected_expression,
            source=source,
            intent="confirm_date",
            topic=snapshot.current_topic,
        )
        original_expression = next(
            (
                message["content"]
                for message in reversed(snapshot.messages)
                if message["role"] == "user" and message["intent"] == "date_conflict"
            ),
            selected_expression,
        )
        local_time = self.date_resolver.extract_local_time(original_expression)
        if local_time is None and candidates:
            if candidates[0].temporal_resolution:
                local_time = TemporalResolution.model_validate(
                    candidates[0].temporal_resolution
                ).resolved_local_time
            if local_time is None:
                local_time = self.date_resolver.extract_local_time(
                    " ".join(
                        value
                        for value in (
                            candidates[0].title,
                            candidates[0].description,
                            candidates[0].raw_text,
                        )
                        if value
                    )
                )
        temporal = self.date_resolver.temporal_resolution(
            selected_date,
            user.timezone,
            original_expression,
            local_time,
        )
        kind = (
            candidates[0].kind
            if candidates
            else "task"
            if any(marker in original_expression.lower() for marker in ("напом", "нужно", "задач"))
            else "idea"
        )
        parsed = self._resolved_temporal_draft(snapshot, temporal, original_expression, kind=kind)
        if candidates:
            action_source = "voice_command" if source == "voice" else "text_command"
            outcome = await self.action_service.execute(
                "confirm_date",
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source=action_source,
                draft_id=candidates[0].id,
                version=candidates[0].version,
                resolved_date=selected_date,
                task=parsed,
                raw_text=parsed.description,
            )
            if outcome.status != "ok" or not outcome.result or not outcome.result.draft:
                await update.effective_message.reply_text(
                    "Карточка изменилась. Повтори выбор даты для актуальной preview."
                )
                return
            draft = outcome.result.draft
            await self._deactivate_preview_keyboard(context, chat_id, outcome.previous_message_id)
        else:
            creation = await self.draft_service.create_or_get(
                user_id=user.id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source=source,
                raw_text=parsed.description or parsed.title,
                parsed=parsed,
            )
            draft = creation.draft
        await self.conversation.set_active_draft(telegram_user_id, chat_id, draft.id)
        await self._send_draft_preview(update.effective_message, draft)
        await self._remember_preview(telegram_user_id, chat_id, draft)

    async def _handle_system_action_route(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: User,
        snapshot: ConversationSnapshot,
        route: SystemActionRoute,
    ) -> None:
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        if route.kind == "clarify":
            if snapshot.system_pending_action:
                await self.conversation.clear_system_action(
                    telegram_user_id,
                    chat_id,
                    expected_version=snapshot.system_action_version,
                )
            await update.effective_message.reply_text(
                "Похоже на команду удаления, но безопасную цель определить нельзя. "
                "Ничего не удалено и новая запись не создана. Уточни, например: "
                "«удали все просроченные задачи» или «удали все черновики»."
            )
            return
        if route.kind == "pending":
            await update.effective_message.reply_text(
                "Ожидаю отдельное подтверждение удаления: «да, удалить» или кнопка «Отмена»."
            )
            return
        if route.kind == "cancel":
            cleared = await self.conversation.clear_system_action(
                telegram_user_id,
                chat_id,
                expected_version=snapshot.system_action_version,
            )
            if not cleared:
                await update.effective_message.reply_text(
                    "Это подтверждение уже неактуально или операция уже выполняется. "
                    "Проверь актуальное состояние через /inbox или /tasks."
                )
                return
            cancelled = self._system_cleanup_cancelled(snapshot.system_pending_action)
            await update.effective_message.reply_text(f"{cancelled}. Ничего не изменено.")
            return
        if route.kind == "confirm":
            await self._confirm_system_cleanup(
                update.effective_message,
                context,
                telegram_user_id,
                chat_id,
                snapshot,
            )
            return
        if snapshot.system_pending_action and route.action in {
            "archive_overdue_tasks",
            "discard_all_active_drafts",
            "discard_selected_drafts",
        }:
            # Explicitly retargeting a pending cleanup invalidates the old
            # capability even when the newly requested target set is empty.
            await self.conversation.clear_system_action(
                telegram_user_id,
                chat_id,
                expected_version=snapshot.system_action_version,
            )
        if route.action == "list_drafts":
            await self.drafts_command(update, context)
            return
        if route.action == "show_last_saved":
            await self.last_saved_command(update, context)
            return
        if route.action == "archive_overdue_tasks":
            task_snapshot = await self.inbox_lifecycle.overdue_snapshot(user.id)
            await self._begin_overdue_task_cleanup(
                update.effective_message,
                telegram_user_id,
                chat_id,
                task_snapshot,
            )
            return
        drafts = await self.draft_service.active_drafts(telegram_user_id, chat_id)
        if route.action == "discard_all_active_drafts":
            affected = {draft.id for draft in drafts}
        elif route.action == "discard_selected_drafts" and drafts:
            affected = {draft.id for draft in drafts[1:]}
        else:
            affected = set()
        await self._begin_system_cleanup(
            update.effective_message,
            telegram_user_id,
            chat_id,
            drafts,
            affected,
        )

    @staticmethod
    def _system_cleanup_cancelled(action: str | None) -> str:
        if action == "archive_overdue_tasks":
            return "Очистка просроченных задач отменена"
        if action in {"trash_inbox_items", "trash_inbox_commands"}:
            return "Перемещение записей Inbox в корзину отменено"
        return "Удаление черновиков отменено"

    async def _begin_overdue_task_cleanup(
        self,
        message: object,
        telegram_user_id: int,
        chat_id: int,
        snapshot: list[dict[str, object]],
    ) -> None:
        if not snapshot:
            await message.reply_text(
                "Неактуальных задач не найдено. Просроченные задачи можно проверить в /tasks."
            )
            return
        version = await self.conversation.begin_system_action(
            telegram_user_id,
            chat_id,
            "archive_overdue_tasks",
            snapshot,
        )
        preview = "\n".join(f"• {item['title']}" for item in snapshot[:5])
        extra = f"\n• …и ещё {len(snapshot) - 5}" if len(snapshot) > 5 else ""
        await message.reply_text(
            f"Нашёл просроченных задач: {len(snapshot)}.\n\n{preview}{extra}\n\n"
            "Переместить их из активных и Inbox в корзину? Связи сохранятся, ожидающие "
            "напоминания будут выключены, а записи можно будет восстановить. "
            "Ничего не изменится без подтверждения.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"Да, убрать {len(snapshot)}",
                            callback_data=f"sysdraft:confirm:{version}",
                        ),
                        InlineKeyboardButton("Отмена", callback_data=f"sysdraft:cancel:{version}"),
                    ]
                ]
            ),
        )

    async def _begin_inbox_trash(
        self,
        message: object,
        telegram_user_id: int,
        chat_id: int,
        snapshot: list[dict[str, object]],
        *,
        action: str = "trash_inbox_items",
    ) -> None:
        if not snapshot:
            await message.reply_text("Подходящих сохранённых записей Inbox не найдено.")
            return
        version = await self.conversation.begin_system_action(
            telegram_user_id,
            chat_id,
            action,
            snapshot,
        )
        preview = "\n".join(
            f"• [{LABELS.get(str(item['kind']), str(item['kind']))}] {item['title']}"
            for item in snapshot[:6]
        )
        extra = f"\n• …и ещё {len(snapshot) - 6}" if len(snapshot) > 6 else ""
        await message.reply_text(
            f"Переместить в корзину записей: {len(snapshot)}?\n\n{preview}{extra}\n\n"
            "Связанные ожидающие напоминания будут выключены. Записи можно восстановить "
            "через Inbox; старые напоминания при восстановлении сами не включатся.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"Да, в корзину ({len(snapshot)})",
                            callback_data=f"sysdraft:confirm:{version}",
                        ),
                        InlineKeyboardButton(
                            "Отмена",
                            callback_data=f"sysdraft:cancel:{version}",
                        ),
                    ]
                ]
            ),
        )

    async def _begin_system_cleanup(
        self,
        message: object,
        telegram_user_id: int,
        chat_id: int,
        drafts: list[DraftInboxItem],
        affected_ids: set[str],
    ) -> None:
        if not affected_ids:
            await message.reply_text("Нет активных черновиков для удаления.")
            return
        snapshot = [
            {
                "id": draft.id,
                "version": draft.version,
                "affected": draft.id in affected_ids,
                "preview_message_id": draft.preview_message_id,
            }
            for draft in drafts
        ]
        action = (
            "discard_all_active_drafts"
            if len(affected_ids) == len(drafts)
            else "discard_one_draft"
            if len(affected_ids) == 1
            else "discard_selected_drafts"
        )
        version = await self.conversation.begin_system_action(
            telegram_user_id,
            chat_id,
            action,
            snapshot,
        )
        count = len(affected_ids)
        await message.reply_text(
            f"Удалить {count} активных черновиков? Сохранённые записи в inbox останутся.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"Да, удалить {count}",
                            callback_data=f"sysdraft:confirm:{version}",
                        ),
                        InlineKeyboardButton("Отмена", callback_data=f"sysdraft:cancel:{version}"),
                    ]
                ]
            ),
        )

    async def _confirm_system_cleanup(
        self,
        message: object,
        context: ContextTypes.DEFAULT_TYPE,
        telegram_user_id: int,
        chat_id: int,
        snapshot: ConversationSnapshot,
    ) -> bool:
        if (
            not snapshot.system_pending_action
            or snapshot.system_action_version is None
            or not snapshot.system_draft_snapshot
        ):
            await message.reply_text(
                "Подтверждение отсутствует или истекло. Открой нужный раздел и повтори действие."
            )
            return False
        claim = await self.conversation.claim_system_action(
            telegram_user_id,
            chat_id,
            expected_version=snapshot.system_action_version,
        )
        if claim is None:
            await message.reply_text(
                "Это подтверждение уже использовано, истекло или заменено новым preview. "
                "Ничего не изменено."
            )
            return False
        try:
            if claim.action in {"trash_inbox_items", "trash_inbox_commands"}:
                user = await self._user(telegram_user_id)
                if claim.action == "trash_inbox_commands":
                    result = await self.inbox_lifecycle.trash_command_garbage_snapshot(
                        user.id,
                        claim.snapshot,
                    )
                else:
                    result = await self.inbox_lifecycle.trash_snapshot(user.id, claim.snapshot)
                if result.status != "trashed":
                    await message.reply_text(
                        "Список Inbox или ошибочно сохранённых команд изменился. "
                        "Ничего не перемещено; открой /inbox и повтори."
                    )
                    return False
                await message.reply_text(
                    f"Перемещено в корзину: {result.count}. Восстановление доступно в /inbox.",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "🗑 Открыть корзину",
                                    callback_data="ibox:trashlist:0",
                                )
                            ]
                        ]
                    ),
                )
                return True
            if claim.action == "archive_overdue_tasks":
                user = await self._user(telegram_user_id)
                result = await self.inbox_lifecycle.trash_snapshot(user.id, claim.snapshot)
                if result.status != "trashed":
                    await message.reply_text(
                        "Список просроченных задач изменился. Ничего не изменено; "
                        "повтори команду очистки."
                    )
                    return False
                await message.reply_text(
                    f"Перемещено в корзину просроченных задач: {result.count}. "
                    "Их можно восстановить через /inbox.",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "🗑 Открыть корзину",
                                    callback_data="ibox:trashlist:0",
                                )
                            ]
                        ]
                    ),
                )
                return True
            result = await self.draft_service.discard_snapshot(
                telegram_user_id, chat_id, claim.snapshot
            )
            if not result.ok:
                await message.reply_text(
                    "Набор черновиков изменился. Ничего не удалено; повтори /cleanup_drafts."
                )
                return False
            await self.conversation.clear_focus(telegram_user_id, chat_id)
            for message_id in result.preview_message_ids or []:
                await self._deactivate_preview_keyboard(context, chat_id, message_id)
            await message.reply_text(
                f"Удалено {result.count} черновиков. Сохранённые записи не затронуты"
            )
            return True
        finally:
            await self.conversation.finalize_system_action_claim(
                telegram_user_id,
                chat_id,
                expected_version=claim.version,
            )

    async def system_draft_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        parts = query.data.split(":")
        if len(parts) != 3 or parts[1] not in {"confirm", "cancel"}:
            await self._stale_callback(query)
            return
        try:
            expected_version = int(parts[2])
        except ValueError:
            await self._stale_callback(query)
            return
        snapshot = await self.conversation.get(update.effective_user.id, update.effective_chat.id)
        if snapshot.system_action_version != expected_version:
            await query.answer("Это подтверждение уже неактуально", show_alert=True)
            return
        if parts[1] == "cancel":
            cleared = await self.conversation.clear_system_action(
                update.effective_user.id,
                update.effective_chat.id,
                expected_version=expected_version,
            )
            if not cleared:
                await query.answer("Это подтверждение уже неактуально", show_alert=True)
                return
            await query.answer()
            await query.edit_message_text(
                f"{self._system_cleanup_cancelled(snapshot.system_pending_action)}."
            )
            return
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        await self._confirm_system_cleanup(
            query.message,
            context,
            update.effective_user.id,
            update.effective_chat.id,
            snapshot,
        )

    async def _handle_action_route(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: User,
        snapshot: ConversationSnapshot,
        route: ActionRoute,
        source: str,
    ) -> None:
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        if route.kind == "control":
            await update.effective_message.reply_text(
                "Это управляющая фраза, новую карточку не создаю. "
                "Выбери draft через /drafts или сформулируй действие прямо."
            )
            return
        if route.kind == "selection":
            selected = await self._resolve_draft_selection(update, telegram_user_id, chat_id, route)
            if selected is None:
                return
            await self.conversation.set_focus(
                telegram_user_id,
                chat_id,
                selected.id,
                selected.version,
                snapshot.pending_action,
            )
            if snapshot.pending_action:
                await self._prompt_pending_action(
                    update.effective_message, selected, snapshot.pending_action
                )
            else:
                await update.effective_message.reply_text(f"Выбрана карточка «{selected.title}».")
            return
        if route.kind == "confirmation":
            if (
                not snapshot.pending_action
                or not snapshot.focused_draft_id
                or snapshot.focused_draft_version is None
            ):
                await update.effective_message.reply_text(
                    "Нет ожидающего подтверждения. Новую карточку не создаю."
                )
                return
            focused = await self.draft_service.active_by_id(
                snapshot.focused_draft_id,
                snapshot.focused_draft_version,
                telegram_user_id,
                chat_id,
            )
            if focused is None:
                await self.conversation.clear_focus(telegram_user_id, chat_id)
                await update.effective_message.reply_text(
                    "Выбранная карточка устарела. Открой актуальные через /drafts."
                )
                return
            await self._execute_draft_command(
                update,
                context,
                user,
                snapshot,
                snapshot.pending_action,
                source,
                target=focused,
            )
            return
        if route.kind != "action" or route.action is None:
            return
        reply_target = await self._reply_draft(update, telegram_user_id, chat_id)
        if reply_target is not None:
            await self._execute_draft_command(
                update,
                context,
                user,
                snapshot,
                route.action,
                source,
                target=reply_target,
            )
            return
        focused = None
        if snapshot.focused_draft_id and snapshot.focused_draft_version is not None:
            focused = await self.draft_service.active_by_id(
                snapshot.focused_draft_id,
                snapshot.focused_draft_version,
                telegram_user_id,
                chat_id,
            )
        if focused is not None:
            if route.needs_confirmation:
                await self.conversation.set_focus(
                    telegram_user_id,
                    chat_id,
                    focused.id,
                    focused.version,
                    route.action,
                )
                await self._prompt_pending_action(update.effective_message, focused, route.action)
            else:
                await self._execute_draft_command(
                    update,
                    context,
                    user,
                    snapshot,
                    route.action,
                    source,
                    target=focused,
                )
            return
        candidates = await self.draft_service.active_previews(telegram_user_id, chat_id)
        if len(candidates) > 1:
            await self.conversation.set_pending_action(telegram_user_id, chat_id, route.action)
            await self._show_draft_choices(
                update.effective_message, list(reversed(candidates)), route.action
            )
            return
        if len(candidates) == 1:
            await self.conversation.set_focus(
                telegram_user_id,
                chat_id,
                candidates[0].id,
                candidates[0].version,
                route.action,
            )
            await self._prompt_pending_action(update.effective_message, candidates[0], route.action)
            return
        await self._execute_draft_command(update, context, user, snapshot, route.action, source)

    async def _resolve_draft_selection(
        self,
        update: Update,
        telegram_user_id: int,
        chat_id: int,
        route: ActionRoute,
    ) -> DraftInboxItem | None:
        if route.selector == "reply":
            selected = await self._reply_draft(update, telegram_user_id, chat_id)
            if selected is None:
                await update.effective_message.reply_text(
                    "«Вот эту» работает только ответом на сообщение preview-карточки."
                )
            return selected
        newest_first = await self.draft_service.active_previews(telegram_user_id, chat_id)
        chronological = list(reversed(newest_first))
        selected = None
        if route.selector in {"last", "newest"} and newest_first:
            selected = newest_first[0]
        elif route.selector == "first" and chronological:
            selected = chronological[0]
        elif route.selector == "second" and len(chronological) >= 2:
            selected = chronological[1]
        elif route.selector == "topic" and route.query:
            query = self._normalize_draft_text(route.query)
            matches = [
                draft
                for draft in newest_first
                if query
                and query
                in self._normalize_draft_text(
                    f"{draft.title} {draft.description or ''} {draft.raw_text}"
                )
            ]
            selected = matches[0] if len(matches) == 1 else None
        if selected is None:
            await update.effective_message.reply_text(
                "Не удалось однозначно выбрать карточку. Используй кнопки или /drafts."
            )
        return selected

    async def _reply_draft(
        self, update: Update, telegram_user_id: int, chat_id: int
    ) -> DraftInboxItem | None:
        replied = getattr(update.effective_message, "reply_to_message", None)
        message_id = getattr(replied, "message_id", None)
        if message_id is None:
            return None
        return await self.draft_service.by_preview_message(telegram_user_id, chat_id, message_id)

    @staticmethod
    async def _show_draft_choices(
        message: object, drafts: list[DraftInboxItem], action: DraftAction
    ) -> None:
        rows = [
            [
                InlineKeyboardButton(
                    f"{index}. {draft.title[:42]}",
                    callback_data=f"draftfocus:{action}:{draft.id}:{draft.version}",
                )
            ]
            for index, draft in enumerate(drafts, start=1)
        ]
        rows.append([InlineKeyboardButton("Отмена", callback_data="draftfocus:cancel")])
        await message.reply_text(
            "К какой карточке применить команду?",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    @staticmethod
    async def _prompt_pending_action(message: object, draft: DraftInboxItem, action: str) -> None:
        if action == "save":
            text = f"Сохранить {ACTION_LABELS[draft.kind]} «{draft.title}»?"
            yes_label = "Да, сохранить"
        else:
            text = f"Применить действие к карточке «{draft.title}»?"
            yes_label = "Да"
        await message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            yes_label,
                            callback_data=f"draftcmd:{action}:{draft.id}:{draft.version}",
                        ),
                        InlineKeyboardButton(
                            "Нет",
                            callback_data=f"draftcmd:no:{draft.id}:{draft.version}",
                        ),
                    ]
                ]
            ),
        )

    @staticmethod
    def _normalize_draft_text(value: str) -> str:
        return " ".join(
            part.strip(".,!?;:()[]{}\"'«»")
            for part in value.lower().replace("ё", "е").split()
            if part.strip(".,!?;:()[]{}\"'«»")
        )

    async def _execute_draft_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: User,
        snapshot: ConversationSnapshot,
        action: DraftAction,
        source: str,
        *,
        target: DraftInboxItem | None = None,
    ) -> None:
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        action_source = "voice_command" if source == "voice" else "text_command"
        task = None
        raw_text = None
        if action == "create_task":
            task = self._task_draft(snapshot, target)
            if task is None:
                await update.effective_message.reply_text(
                    "Не вижу одной активной темы или карточки для задачи. Уточни содержание."
                )
                return
            raw_text = task.description
        if target is None and action in {"save", "discard", "cancel"}:
            candidates = await self.draft_service.active_previews(telegram_user_id, chat_id)
            if len(candidates) == 1:
                target = candidates[0]
        target_message_id = getattr(target, "preview_message_id", None)
        if type(target_message_id) is not int or target_message_id <= 0:
            target_message_id = None
        pending_capture = (
            self.nova_companion_pending_capture_anchor(
                telegram_user_id,
                chat_id,
                target.id,
                target.version,
                target_message_id,
            )
            if target is not None and target_message_id is not None
            else None
        )
        outcome = await self.action_service.execute(
            action,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            source=action_source,
            task=task,
            user_id=user.id,
            raw_text=raw_text,
            draft_id=target.id if target else None,
            version=target.version if target else None,
            expected_preview_message_id=target_message_id,
            expected_access_version=(
                pending_capture.access_version if pending_capture is not None else None
            ),
        )
        if outcome.status == "ambiguous":
            candidates = await self.draft_service.active_previews(telegram_user_id, chat_id)
            await self.conversation.set_pending_action(telegram_user_id, chat_id, action)
            await self._show_draft_choices(
                update.effective_message, list(reversed(candidates)), action
            )
            return
        if outcome.status in {"missing", "stale"} or not outcome.result:
            await update.effective_message.reply_text(
                "Нет одной актуальной preview-карточки для этой команды."
            )
            return
        draft = outcome.result.draft
        if action in {"discard", "cancel"}:
            self.nova_companion_record_terminal_capture(
                telegram_user_id,
                chat_id,
                outcome,
                expected_pending=pending_capture,
            )
        await self._deactivate_preview_keyboard(context, chat_id, outcome.previous_message_id)
        if action == "save":
            await self.conversation.clear_focus(telegram_user_id, chat_id)
            await self.conversation.set_active_draft(telegram_user_id, chat_id, None)
            channel = "голосовой" if source == "voice" else "текстовой"
            receipt = await self._record_saved_receipt(
                telegram_user_id,
                chat_id,
                outcome,
                expected_companion_pending=pending_capture,
            )
            await update.effective_message.reply_text(
                f"Сохранено в inbox по {channel} команде.\n{receipt}",
                reply_markup=self._saved_receipt_markup(outcome),
            )
        elif action in {"discard", "cancel"}:
            await self.conversation.clear_focus(telegram_user_id, chat_id)
            await self.conversation.set_active_draft(telegram_user_id, chat_id, None)
            await update.effective_message.reply_text("Карточка удалена без сохранения.")
        elif action == "edit":
            await update.effective_message.reply_text("Пришли исправленный текст одним сообщением.")
        elif action == "create_task" and draft:
            await self.conversation.set_active_draft(telegram_user_id, chat_id, draft.id)
            await self._send_draft_preview(update.effective_message, draft)
            await self._remember_preview(telegram_user_id, chat_id, draft)

    @staticmethod
    def _topic_draft(
        snapshot: ConversationSnapshot, resolved_date: date, *, kind: str
    ) -> ParsedThought:
        topic = (snapshot.current_topic or "Запланированное действие").strip()
        if "еженедель" in topic.lower() and "план" in topic.lower():
            title = "Еженедельное планирование"
            description = (
                "Каждое воскресенье составлять план следующей недели и подводить итоги предыдущей"
            )
        else:
            title = topic[:200].capitalize()
            description = topic
        return ParsedThought(
            kind=kind,
            title=title,
            description=description,
            next_step=f"Начать {resolved_date.strftime('%d.%m.%Y')}",
            resolved_date=resolved_date,
        )

    @staticmethod
    def _resolved_temporal_draft(
        snapshot: ConversationSnapshot,
        temporal: TemporalResolution,
        original_expression: str,
        *,
        kind: str,
    ) -> ParsedThought:
        months = (
            "",
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        )
        local_date = temporal.resolved_local_date
        date_phrase = f"{local_date.day} {months[local_date.month]}"
        time_suffix = (
            f" в {temporal.resolved_local_time.strftime('%H:%M')}"
            if temporal.resolved_local_time
            else ""
        )
        lowered = original_expression.lower()
        topic = (snapshot.current_topic or "Запланированное действие").strip()
        if "стриж" in lowered:
            subject = "Стрижка"
            title = f"{subject} — {date_phrase}{time_suffix}"
            description = f"{subject}: {date_phrase} {local_date.year}{time_suffix}."
        elif "еженедель" in topic.lower() and "план" in topic.lower():
            title = "Еженедельное планирование"
            description = (
                "Каждое воскресенье составлять план следующей недели и подводить "
                f"итоги предыдущей. Начало: {date_phrase} {local_date.year}{time_suffix}."
            )
        else:
            subject = topic[:120].capitalize()
            title = f"{subject} — {date_phrase}{time_suffix}"
            description = f"{subject}: {date_phrase} {local_date.year}{time_suffix}."
        return ParsedThought(
            kind=kind,
            title=title,
            description=description,
            next_step=f"Дата: {local_date.strftime('%d.%m.%Y')}{time_suffix}",
            resolved_date=local_date,
            temporal_resolution=temporal,
        )

    def _task_draft(
        self, snapshot: ConversationSnapshot, target: DraftInboxItem | None
    ) -> ParsedThought | None:
        resolved = target.resolved_date if target else None
        if resolved is None and snapshot.resolved_date:
            resolved = date.fromisoformat(snapshot.resolved_date)
        topic = target.title if target else snapshot.current_topic
        if not topic:
            return None
        base = self._topic_draft(snapshot, resolved or date.today(), kind="task")
        if target and not ("еженедель" in topic.lower() and "план" in topic.lower()):
            base = base.model_copy(
                update={
                    "title": topic[:200],
                    "description": target.description or target.raw_text,
                    "resolved_date": resolved,
                }
            )
        return base.model_copy(
            update={
                "next_step": "Это черновик задачи, напоминание ещё не настроено",
                "resolved_date": resolved,
                "temporal_resolution": (
                    TemporalResolution.model_validate(target.temporal_resolution)
                    if target and target.temporal_resolution
                    else None
                ),
            }
        )

    @staticmethod
    async def _deactivate_preview_keyboard(
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        message_id: int | None,
    ) -> None:
        bot = getattr(context, "bot", None)
        if bot is None or message_id is None:
            return
        try:
            await bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=message_id, reply_markup=None
            )
        except TelegramError as exc:
            log_safe_failure("Preview keyboard cleanup failed", exc)

    @staticmethod
    def _parsed_from_intent(
        result: IntentResult,
        text: str,
        *,
        fallback_kind: str = "note",
        resolved_date: date | None = None,
        temporal_resolution: TemporalResolution | None = None,
    ) -> ParsedThought:
        return ParsedThought(
            kind=result.inbox_kind or fallback_kind,
            title=result.title or text.strip()[:80],
            next_step=result.next_step,
            resolved_date=resolved_date,
            temporal_resolution=temporal_resolution,
        )

    async def _show_preview(
        self,
        message: object,
        user_id: int,
        telegram_user_id: int,
        chat_id: int,
        text: str,
        source: str,
        parsed: ParsedThought,
        *,
        include_original: bool = True,
    ) -> None:
        creation = await self.draft_service.create_or_get(
            user_id=user_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            source=source,
            raw_text=text,
            parsed=parsed,
        )
        draft = creation.draft
        await self.conversation.set_active_draft(telegram_user_id, chat_id, draft.id)
        await self._send_draft_preview(message, draft, include_original=include_original)
        await self._remember_preview(telegram_user_id, chat_id, draft)

    async def _remember_preview(
        self, telegram_user_id: int, chat_id: int, draft: DraftInboxItem
    ) -> None:
        await self.conversation.append(
            telegram_user_id,
            chat_id,
            role="assistant",
            content=f"Подготовлена preview-карточка: {draft.title}",
            source="text",
            intent="preview",
            topic=draft.title,
        )

    @staticmethod
    def _is_reference_request(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in ("это", "туда", "выше", "последнее"))

    @staticmethod
    def _is_task_question(text: str) -> bool:
        lowered = text.lower().replace("ё", "е")
        return "занес" in lowered and "задач" in lowered

    @staticmethod
    async def _show_task_choices(message: object, session_id: int, answer: str) -> object:
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Создать задачу", callback_data=f"context:task:{session_id}"
                    ),
                    InlineKeyboardButton(
                        "Оставить идеей", callback_data=f"context:idea:{session_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Уточнить дату", callback_data=f"context:date:{session_id}"
                    ),
                    InlineKeyboardButton("Ничего", callback_data=f"context:drop:{session_id}"),
                ],
            ]
        )
        return await message.reply_text(answer, reply_markup=keyboard)

    async def _deliver_routed_answer(
        self,
        message: object,
        session_id: int,
        question: str,
        answer: str,
    ) -> object | None:
        if self._is_task_question(question):
            return await self._show_task_choices(message, session_id, answer)
        return await message.reply_text(answer)

    async def _append_delivered_answer(
        self,
        telegram_user_id: int,
        chat_id: int,
        answer: str,
        topic: str | None,
        *,
        intent: str = "answer",
    ) -> None:
        await self.conversation.append(
            telegram_user_id,
            chat_id,
            role="assistant",
            content=answer,
            source="text",
            intent=intent,
            topic=topic,
        )

    async def _prepare_nova_memory_answer(
        self,
        *,
        user: User,
        chat_id: int,
        question: str,
        route_answer: str | None,
        timezone_name: str,
        conversation_context: dict[str, object],
        legacy_default_answer: bool = False,
    ) -> _NovaMemoryPreparedAnswer | _NovaMemoryApplicationResult:
        policy = self.nova_memory_application_policy()
        try:
            snapshot = await self.nova_memory_service.application_snapshot(
                telegram_actor_id=user.telegram_id,
                expected_tier=user.access_tier,
                expected_access_version=user.access_version,
                policy=policy,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory application failed stage=snapshot error_type=%s",
                type(exc).__name__,
            )
            return "unavailable"
        mapped = self._nova_memory_snapshot_result(snapshot)
        if mapped != "ready":
            return mapped
        assert snapshot.collection_revision is not None
        fence = _NovaMemoryApplicationFence(
            telegram_actor_id=user.telegram_id,
            chat_id=chat_id,
            tier=user.access_tier,
            access_version=user.access_version,
            collection_revision=snapshot.collection_revision,
        )
        projection: NovaMemoryProjection | None = None
        if snapshot.status == "ready":
            try:
                projection = build_nova_memory_projection(
                    snapshot.items,
                    collection_revision=snapshot.collection_revision,
                )
            except NovaMemoryProjectionError as exc:
                logger.warning(
                    "Nova memory application failed stage=projection error_type=%s",
                    type(exc).__name__,
                )
                return "unavailable"

        answer = route_answer
        if answer is None and projection is None and legacy_default_answer:
            answer = "Я тебя услышал. Можешь уточнить, чем помочь?"
        if projection is not None or answer is None:
            before_provider = await self._nova_memory_application_check(fence)
            if before_provider != "ready":
                return before_provider
            try:
                generated = await self.intent_router.answer(
                    question,
                    timezone_name,
                    conversation_context=conversation_context,
                    confirmed_memory=projection,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Nova memory application failed stage=provider error_type=%s",
                    type(exc).__name__,
                )
                return "provider_failed"
            after_provider = await self._nova_memory_application_check(fence)
            if after_provider != "ready":
                return after_provider
            answer = generated.answer
        assert answer is not None
        return _NovaMemoryPreparedAnswer(answer, fence, projection)

    async def _nova_memory_application_check(
        self,
        fence: _NovaMemoryApplicationFence,
    ) -> _NovaMemoryApplicationResult:
        try:
            current = await self.nova_memory_service.application_current_check(
                telegram_actor_id=fence.telegram_actor_id,
                expected_tier=fence.tier,
                expected_access_version=fence.access_version,
                expected_collection_revision=fence.collection_revision,
                policy=self.nova_memory_application_policy(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory application failed stage=current_check error_type=%s",
                type(exc).__name__,
            )
            return "unavailable"
        if current.status in {"ready", "empty"}:
            return "ready"
        if current.status in {"disabled", "access_changed"}:
            return "access_changed"
        if current.status == "memory_changed":
            return "memory_changed"
        return "unavailable"

    async def _nova_memory_application_user_is_current(self, user: User) -> bool:
        """Bind routing to the full-access generation admitted for this update."""
        try:
            async with self.db.sessions() as session:
                current = await session.scalar(
                    select(User.id).where(
                        User.id == user.id,
                        User.telegram_id == user.telegram_id,
                        User.access_tier == user.access_tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == user.access_version,
                    )
                )
            return current == user.id
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory application failed stage=access_generation error_type=%s",
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _nova_memory_user_matches_generation(
        user: User,
        generation: _NovaMemoryAccessGeneration,
    ) -> bool:
        return (
            user.telegram_id == generation.telegram_actor_id
            and user.access_tier == generation.tier
            and user.access_version == generation.access_version
        )

    @staticmethod
    def _nova_memory_access_generation(
        context: Any,
        *,
        telegram_actor_id: int,
        chat_id: int,
        fallback: User,
    ) -> _NovaMemoryAccessGeneration:
        generation = getattr(context, _NOVA_MEMORY_APPLICATION_ACCESS_ATTR, None)
        if (
            isinstance(generation, _NovaMemoryAccessGeneration)
            and generation.telegram_actor_id == telegram_actor_id
            and generation.chat_id == chat_id
        ):
            return generation
        return _NovaMemoryAccessGeneration(
            telegram_actor_id=telegram_actor_id,
            chat_id=chat_id,
            tier=fallback.access_tier,
            access_version=fallback.access_version,
        )

    @staticmethod
    def _nova_memory_snapshot_result(
        snapshot: NovaMemoryApplicationSnapshot,
    ) -> _NovaMemoryApplicationResult:
        if snapshot.status in {"ready", "empty"} and snapshot.collection_revision:
            return "ready"
        if snapshot.status in {"disabled", "access_changed"}:
            return "access_changed"
        if snapshot.status == "memory_changed":
            return "memory_changed"
        return "unavailable"

    @staticmethod
    def _nova_memory_application_neutral_text(
        outcome: _NovaMemoryApplicationResult,
    ) -> str:
        if outcome == "access_changed":
            return NOVA_MEMORY_ACCESS_CHANGED_TEXT
        if outcome == "memory_changed":
            return NOVA_MEMORY_APPLICATION_CHANGED_TEXT
        return NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT

    @staticmethod
    def _log_nova_memory_application_outcome(
        projection: NovaMemoryProjection | None,
    ) -> None:
        if projection is None:
            logger.info(
                "Nova memory application outcome=empty selected_count=0 omitted_count=0 "
                "important_count=0 payload_bytes=0"
            )
            return
        logger.info(
            "Nova memory application outcome=applied selected_count=%s omitted_count=%s "
            "important_count=%s payload_bytes=%s",
            projection.selected_count,
            projection.omitted_count,
            projection.important_count,
            projection.payload_bytes,
        )

    async def _deliver_nova_memory_application_answer(
        self,
        context: Any,
        message: object,
        *,
        session_id: int,
        question: str,
        prepared: _NovaMemoryPreparedAnswer,
        telegram_user_id: int,
        chat_id: int,
        topic: str | None,
    ) -> bool:
        task = asyncio.create_task(
            self._nova_memory_application_send_lifecycle(
                context,
                message,
                session_id=session_id,
                question=question,
                prepared=prepared,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                topic=topic,
            ),
            name="nova-memory-application-send-lifecycle",
        )
        self._track_nova_memory_application_task(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise

    async def _nova_memory_application_send_lifecycle(
        self,
        context: Any,
        message: object,
        *,
        session_id: int,
        question: str,
        prepared: _NovaMemoryPreparedAnswer,
        telegram_user_id: int,
        chat_id: int,
        topic: str | None,
    ) -> bool:
        try:
            return await self._run_nova_memory_application_send_lifecycle(
                context,
                message,
                session_id=session_id,
                question=question,
                prepared=prepared,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                topic=topic,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory application task failed operation=send_lifecycle error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _run_nova_memory_application_send_lifecycle(
        self,
        context: Any,
        message: object,
        *,
        session_id: int,
        question: str,
        prepared: _NovaMemoryPreparedAnswer,
        telegram_user_id: int,
        chat_id: int,
        topic: str | None,
    ) -> bool:
        pre_delivery = await self._nova_memory_application_check(prepared.fence)
        if pre_delivery != "ready":
            await message.reply_text(self._nova_memory_application_neutral_text(pre_delivery))
            return False
        try:
            sent = await self._deliver_routed_answer(
                message,
                session_id,
                question,
                prepared.text,
            )
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            logger.warning(
                "Nova memory application delivery failed stage=telegram_send error_type=%s",
                type(exc).__name__,
            )
            return False
        if sent is None:
            return False
        message_id = self._positive_message_id(getattr(sent, "message_id", None))
        if message_id is None:
            logger.warning(
                "Nova memory application failed stage=delivery_binding error_type=MissingMessageId"
            )
            return False
        outcome = await self._nova_memory_application_check(prepared.fence)
        if outcome == "ready":
            self._log_nova_memory_application_outcome(prepared.projection)
            await self._append_delivered_answer(
                telegram_user_id,
                chat_id,
                prepared.text,
                topic,
                intent="memory_answer" if prepared.projection else "answer",
            )
            return True
        await self._compensate_nova_memory_application_message(
            context,
            sent,
            chat_id=prepared.fence.chat_id,
            message_id=message_id,
            neutral_text=self._nova_memory_application_neutral_text(outcome),
        )
        return False

    def _track_nova_memory_application_task(self, task: asyncio.Task[bool]) -> None:
        tasks = getattr(self, "_nova_memory_application_tasks", None)
        if tasks is None:
            tasks = set()
            self._nova_memory_application_tasks = tasks
        tasks.add(task)

        def finish(completed: asyncio.Task[bool]) -> None:
            tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError as exc:
                logger.warning(
                    "Nova memory application task finished operation=delivery_lifecycle "
                    "error_type=%s",
                    type(exc).__name__,
                )
            except BaseException as exc:
                logger.warning(
                    "Nova memory application task failed operation=delivery_lifecycle "
                    "error_type=%s",
                    type(exc).__name__,
                )
                return

        task.add_done_callback(finish)

    async def _drain_nova_memory_application_tasks(self) -> None:
        current = asyncio.current_task()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _NOVA_MEMORY_APPLICATION_DRAIN_TIMEOUT_SECONDS
        while True:
            pending = self._pending_nova_memory_application_tasks(current)
            if not pending:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            _done, still_pending = await asyncio.wait(pending, timeout=remaining)
            if still_pending:
                break

        pending = self._pending_nova_memory_application_tasks(current)
        if not pending:
            return
        logger.warning(
            "Nova memory application shutdown operation=drain error_type=TimeoutError "
            "pending_count=%s",
            len(pending),
        )
        cancel_deadline = loop.time() + _NOVA_MEMORY_APPLICATION_CANCEL_TIMEOUT_SECONDS
        while True:
            pending = self._pending_nova_memory_application_tasks(current)
            if not pending:
                return
            for task in pending:
                task.cancel()
            remaining = cancel_deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.wait(
                pending,
                timeout=min(_NOVA_MEMORY_APPLICATION_CANCEL_RETRY_SECONDS, remaining),
            )

        pending = self._pending_nova_memory_application_tasks(current)
        if not pending:
            return
        logger.error(
            "Nova memory application shutdown operation=terminal_drain "
            "error_type=TimeoutError pending_count=%s",
            len(pending),
        )
        raise _NovaMemoryApplicationDrainError("Nova memory application terminal drain timed out")

    def _pending_nova_memory_application_tasks(
        self,
        current: asyncio.Task[object] | None,
    ) -> set[asyncio.Task[bool]]:
        tasks = getattr(self, "_nova_memory_application_tasks", set())
        for task in tuple(tasks):
            if task is current or not task.done():
                continue
            try:
                task.result()
            except BaseException:
                pass
            tasks.discard(task)
        return {task for task in tuple(tasks) if task is not current and not task.done()}

    async def _compensate_nova_memory_application_message(
        self,
        context: Any,
        message: object,
        *,
        chat_id: int,
        message_id: int,
        neutral_text: str,
    ) -> None:
        bot = getattr(context, "bot", None)
        delete = getattr(bot, "delete_message", None)
        delete_succeeded = False
        try:
            if callable(delete):
                deleted = await delete(chat_id=chat_id, message_id=message_id)
                delete_succeeded = deleted is not False
            else:
                message_delete = getattr(message, "delete", None)
                if callable(message_delete):
                    deleted = await message_delete()
                    delete_succeeded = deleted is not False
        except asyncio.CancelledError as exc:
            logger.warning(
                "Nova memory application cleanup failed operation=delete error_type=%s",
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory application cleanup failed operation=delete error_type=%s",
                type(exc).__name__,
            )
        if delete_succeeded:
            return
        try:
            edit = getattr(bot, "edit_message_text", None)
            if callable(edit):
                await edit(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=neutral_text,
                    reply_markup=None,
                    parse_mode=None,
                )
                return
            message_edit = getattr(message, "edit_text", None)
            if callable(message_edit):
                await message_edit(
                    neutral_text,
                    reply_markup=None,
                    parse_mode=None,
                )
        except asyncio.CancelledError as exc:
            logger.warning(
                "Nova memory application cleanup failed operation=edit error_type=%s",
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory application cleanup failed operation=edit error_type=%s",
                type(exc).__name__,
            )

    async def _edit_nova_memory_application_answer(
        self,
        context: Any,
        query: Any,
        prepared: _NovaMemoryPreparedAnswer,
        *,
        pending_key: str,
        pending: PendingIntent,
        binding: _NovaMemoryCallbackBinding,
        telegram_user_id: int,
        chat_id: int,
        topic: str | None,
    ) -> bool:
        task = asyncio.create_task(
            self._nova_memory_application_edit_lifecycle(
                context,
                query,
                prepared,
                pending_key=pending_key,
                pending=pending,
                binding=binding,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                topic=topic,
            ),
            name="nova-memory-application-edit-lifecycle",
        )
        self._track_nova_memory_application_task(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise

    async def _nova_memory_application_edit_lifecycle(
        self,
        context: Any,
        query: Any,
        prepared: _NovaMemoryPreparedAnswer,
        *,
        pending_key: str,
        pending: PendingIntent,
        binding: _NovaMemoryCallbackBinding,
        telegram_user_id: int,
        chat_id: int,
        topic: str | None,
    ) -> bool:
        try:
            return await self._run_nova_memory_application_edit_lifecycle(
                context,
                query,
                prepared,
                pending_key=pending_key,
                pending=pending,
                binding=binding,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                topic=topic,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory application task failed operation=edit_lifecycle error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _run_nova_memory_application_edit_lifecycle(
        self,
        context: Any,
        query: Any,
        prepared: _NovaMemoryPreparedAnswer,
        *,
        pending_key: str,
        pending: PendingIntent,
        binding: _NovaMemoryCallbackBinding,
        telegram_user_id: int,
        chat_id: int,
        topic: str | None,
    ) -> bool:
        ui_lock = self._nova_memory_application_ui_lock(binding)
        async with ui_lock:
            if context.user_data.get(pending_key) is not pending:
                return False
            if not self._nova_memory_callback_binding_matches(query, binding):
                return False
            pre_edit = await self._nova_memory_application_check(prepared.fence)
            if context.user_data.get(
                pending_key
            ) is not pending or not self._nova_memory_callback_binding_matches(query, binding):
                return False
            if pre_edit != "ready":
                await self._edit_nova_memory_application_neutral(
                    query,
                    self._nova_memory_application_neutral_text(pre_edit),
                    operation="intent_answer_pre_edit",
                )
                return False
            try:
                await query.edit_message_text(prepared.text)
            except asyncio.CancelledError:
                raise
            except BadRequest as exc:
                if "message is not modified" in str(exc).casefold():
                    pass
                else:
                    logger.warning(
                        "Nova memory application delivery failed stage=telegram_edit error_type=%s",
                        type(exc).__name__,
                    )
                    return False
            except TelegramError as exc:
                logger.warning(
                    "Nova memory application delivery failed stage=telegram_edit error_type=%s",
                    type(exc).__name__,
                )
                return False
            delivered = await self._nova_memory_application_post_edit_fence(
                context,
                query,
                prepared.fence,
                pending_key=pending_key,
                pending=pending,
                binding=binding,
            )
            if delivered:
                self._log_nova_memory_application_outcome(prepared.projection)
                await self._append_delivered_answer(
                    telegram_user_id,
                    chat_id,
                    prepared.text,
                    topic,
                    intent="memory_answer" if prepared.projection else "answer",
                )
            return delivered

    async def _nova_memory_application_post_edit_fence(
        self,
        context: Any,
        query: Any,
        fence: _NovaMemoryApplicationFence,
        *,
        pending_key: str,
        pending: PendingIntent,
        binding: _NovaMemoryCallbackBinding,
    ) -> bool:
        if context.user_data.get(
            pending_key
        ) is not pending or not self._nova_memory_callback_binding_matches(query, binding):
            await self._edit_nova_memory_application_neutral(
                query,
                NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
                operation="intent_answer_replaced_post_edit",
                observe_cancellation=True,
            )
            return False
        outcome = await self._nova_memory_application_check(fence)
        if context.user_data.get(
            pending_key
        ) is not pending or not self._nova_memory_callback_binding_matches(query, binding):
            await self._edit_nova_memory_application_neutral(
                query,
                NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
                operation="intent_answer_replaced_post_edit",
                observe_cancellation=True,
            )
            return False
        if outcome == "ready":
            return True
        await self._edit_nova_memory_application_neutral(
            query,
            self._nova_memory_application_neutral_text(outcome),
            operation="intent_answer_post_edit",
            observe_cancellation=True,
        )
        return False

    @classmethod
    def _nova_memory_callback_binding(
        cls,
        update: Any,
        query: Any,
        pending: PendingIntent,
    ) -> _NovaMemoryCallbackBinding | None:
        expected_chat_id = cls._positive_message_id(pending.canonical_chat_id)
        expected_message_id = cls._positive_message_id(pending.canonical_message_id)
        message = getattr(query, "message", None)
        actual_chat_id = cls._positive_message_id(
            getattr(getattr(message, "chat", None), "id", None)
        )
        actual_message_id = cls._positive_message_id(getattr(message, "message_id", None))
        update_chat_id = cls._positive_message_id(
            getattr(getattr(update, "effective_chat", None), "id", None)
        )
        if (
            expected_chat_id is None
            or expected_message_id is None
            or actual_chat_id != expected_chat_id
            or update_chat_id != expected_chat_id
            or actual_message_id != expected_message_id
        ):
            return None
        return _NovaMemoryCallbackBinding(expected_chat_id, expected_message_id)

    def _nova_memory_application_ui_lock(
        self,
        binding: _NovaMemoryCallbackBinding,
    ) -> asyncio.Lock:
        key = (binding.chat_id, binding.message_id)
        lock = self._nova_memory_application_ui_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._nova_memory_application_ui_locks[key] = lock
        return lock

    @classmethod
    def _nova_memory_callback_binding_matches(
        cls,
        query: Any,
        binding: _NovaMemoryCallbackBinding,
    ) -> bool:
        message = getattr(query, "message", None)
        message_id = cls._positive_message_id(getattr(message, "message_id", None))
        chat_id = cls._positive_message_id(getattr(getattr(message, "chat", None), "id", None))
        return message_id == binding.message_id and chat_id == binding.chat_id

    @staticmethod
    async def _edit_nova_memory_application_neutral(
        query: Any,
        text: str,
        *,
        operation: str,
        observe_cancellation: bool = False,
    ) -> None:
        try:
            await query.edit_message_text(
                text,
                reply_markup=None,
                parse_mode=None,
            )
        except asyncio.CancelledError as exc:
            logger.warning(
                "Nova memory application cleanup failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            task = asyncio.current_task()
            if not observe_cancellation or task is None or task.cancelling():
                raise
        except Exception as exc:
            logger.warning(
                "Nova memory application cleanup failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )

    async def _edit_nova_memory_application_neutral_if_current(
        self,
        context: Any,
        query: Any,
        text: str,
        *,
        pending_key: str,
        pending: PendingIntent,
        binding: _NovaMemoryCallbackBinding,
        operation: str,
    ) -> bool:
        ui_lock = self._nova_memory_application_ui_lock(binding)
        async with ui_lock:
            if context.user_data.get(pending_key) is not pending:
                return False
            if not self._nova_memory_callback_binding_matches(query, binding):
                return False
            await self._edit_nova_memory_application_neutral(
                query,
                text,
                operation=operation,
            )
            return True

    async def _send_draft_preview(
        self,
        message: object,
        draft: DraftInboxItem,
        *,
        include_original: bool = True,
        on_sent: Callable[[object], None] | None = None,
        bind_preview: bool = True,
    ) -> object:
        step = f"\nСледующий шаг: {escape(draft.next_step)}" if draft.next_step else ""
        description = f"\nОписание: {escape(draft.description)}" if draft.description else ""
        if draft.temporal_resolution:
            temporal = TemporalResolution.model_validate(draft.temporal_resolution)
            local_value = temporal.resolved_local_date.strftime("%d.%m.%Y")
            if temporal.resolved_local_time:
                local_value += f" {temporal.resolved_local_time.strftime('%H:%M')}"
            temporal_label = "Напоминание" if temporal.remind_at is not None else "Дата"
            resolved = (
                f"\n{temporal_label}: {local_value}\nЧасовой пояс: {escape(temporal.timezone)}"
            )
        else:
            resolved = (
                f"\nДата начала: {draft.resolved_date.strftime('%d.%m.%Y')}"
                if draft.resolved_date
                else ""
            )
        task_notice = (
            "\nПосле сохранения напоминание будет настроено по указанной дате"
            if draft.kind == "task" and draft.temporal_resolution
            else ""
        )
        original = f"Исходный текст: {escape(draft.raw_text)}\n" if include_original else ""
        prefix = f"inbox:{{}}:{draft.id}:{draft.version}"
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Сохранить", callback_data=prefix.format("save")),
                    InlineKeyboardButton("Редактировать", callback_data=prefix.format("edit")),
                    InlineKeyboardButton("Не сохранять", callback_data=prefix.format("drop")),
                ]
            ]
        )
        preview_message = await message.reply_text(
            f"{original}Тип: {LABELS[draft.kind]}\nЗаголовок: <b>{escape(draft.title)}</b>"
            f"{description}{resolved}{step}{task_notice}",
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        if on_sent is not None:
            on_sent(preview_message)
        if bind_preview and (message_id := getattr(preview_message, "message_id", None)):
            await self.draft_service.set_preview_message(draft.id, message_id)
        return preview_message

    async def _show_unknown(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        source: str,
        result: IntentResult,
    ) -> None:
        token = uuid4().hex[:12]
        pending = PendingIntent(token, text, source, result)
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Ответить", callback_data=f"intent:answer:{token}")],
                [
                    InlineKeyboardButton(
                        "Сохранить как идею", callback_data=f"intent:idea:{token}"
                    ),
                    InlineKeyboardButton(
                        "Сохранить как задачу", callback_data=f"intent:task:{token}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Сохранить как заметку", callback_data=f"intent:note:{token}"
                    ),
                    InlineKeyboardButton("Ничего", callback_data=f"intent:drop:{token}"),
                ],
            ]
        )
        sent = await update.effective_message.reply_text(
            "Что сделать с этим сообщением?", reply_markup=keyboard
        )
        pending.canonical_chat_id = self._positive_message_id(
            getattr(getattr(sent, "chat", None), "id", None)
            or getattr(update.effective_chat, "id", None)
        )
        pending.canonical_message_id = self._positive_message_id(getattr(sent, "message_id", None))
        if pending.canonical_chat_id is not None and pending.canonical_message_id is not None:
            context.user_data[f"intent:{token}"] = pending
        else:
            logger.warning(
                "Unknown intent callback binding failed stage=delivery_binding "
                "error_type=MissingMessageId"
            )

    async def intent_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        _, action, token = query.data.split(":", 2)
        pending = context.user_data.get(f"intent:{token}")
        if not isinstance(pending, PendingIntent) or pending.handled:
            await query.answer("Это действие уже обработано", show_alert=True)
            return
        binding = (
            self._nova_memory_callback_binding(update, query, pending)
            if action == "answer"
            else None
        )
        if action == "answer" and binding is None:
            await query.answer("Это действие устарело", show_alert=True)
            return
        pending.handled = True
        await query.answer()
        if action == "drop":
            await query.edit_message_text("Хорошо, ничего не делаю.")
            return
        if action == "answer":
            pending_key = f"intent:{token}"
            assert binding is not None
            try:
                user = await self._user(update.effective_user.id)
                access_generation = self._nova_memory_access_generation(
                    context,
                    telegram_actor_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    fallback=user,
                )
                snapshot = await self.conversation.get(
                    update.effective_user.id, update.effective_chat.id
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Nova memory application failed stage=callback_context error_type=%s",
                    type(exc).__name__,
                )
                await self._edit_nova_memory_application_neutral_if_current(
                    context,
                    query,
                    NOVA_MEMORY_APPLICATION_UNAVAILABLE_TEXT,
                    pending_key=pending_key,
                    pending=pending,
                    binding=binding,
                    operation="intent_answer_context",
                )
                return
            application_enabled = self.nova_memory_application_available_for_tier(
                access_generation.tier
            )
            if not application_enabled:
                try:
                    answer = await self.intent_router.answer(
                        pending.raw_text,
                        user.timezone,
                        conversation_context=snapshot.for_prompt(),
                    )
                except Exception as exc:
                    log_safe_failure("Unknown intent answer failed", exc, user_id=user.id)
                    await query.edit_message_text(
                        "Не удалось ответить сейчас. Ничего не сохранено."
                    )
                    return
                await query.edit_message_text(answer.answer)
                return
            if not self._nova_memory_user_matches_generation(user, access_generation):
                await self._edit_nova_memory_application_neutral_if_current(
                    context,
                    query,
                    NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                    pending_key=pending_key,
                    pending=pending,
                    binding=binding,
                    operation="intent_answer_access_generation",
                )
                return
            prepared = await self._prepare_nova_memory_answer(
                user=user,
                chat_id=update.effective_chat.id,
                question=pending.raw_text,
                route_answer=None,
                timezone_name=user.timezone,
                conversation_context=snapshot.for_prompt(),
                legacy_default_answer=False,
            )
            if context.user_data.get(pending_key) is not pending:
                return
            if isinstance(prepared, str):
                await self._edit_nova_memory_application_neutral_if_current(
                    context,
                    query,
                    self._nova_memory_application_neutral_text(prepared),
                    pending_key=pending_key,
                    pending=pending,
                    binding=binding,
                    operation="intent_answer_preparation",
                )
                return
            try:
                await self._edit_nova_memory_application_answer(
                    context,
                    query,
                    prepared,
                    pending_key=pending_key,
                    pending=pending,
                    binding=binding,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    topic=pending.result.topic or snapshot.current_topic,
                )
            except asyncio.CancelledError:
                raise
            return
        parsed = ParsedThought(
            kind=action,
            title=pending.result.title or pending.raw_text[:80],
            next_step=pending.result.next_step,
        )
        user = await self._user(update.effective_user.id)
        await query.edit_message_reply_markup(reply_markup=None)
        await self._show_preview(
            query.message,
            user.id,
            update.effective_user.id,
            update.effective_chat.id,
            pending.raw_text,
            pending.source,
            parsed,
            include_original=True,
        )

    async def context_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        parts = query.data.split(":")
        if len(parts) != 3 or parts[1] not in {"task", "idea", "date", "drop"}:
            await query.answer("Этот выбор уже неактуален", show_alert=True)
            return
        try:
            session_id = int(parts[2])
        except ValueError:
            await query.answer("Этот выбор уже неактуален", show_alert=True)
            return
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        conversation = await self.conversation.by_id(session_id, telegram_user_id, chat_id)
        if conversation is None:
            await query.answer("Этот выбор уже неактуален", show_alert=True)
            return
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        action = parts[1]
        if action == "drop":
            await query.message.reply_text("Хорошо, ничего не создаю.")
            return
        if action == "date":
            await query.message.reply_text(
                "Уточни дату текстом. До preview и подтверждения ничего не сохранится."
            )
            return
        snapshot = await self.conversation.get(telegram_user_id, chat_id)
        candidate = self.conversation.reference_candidate(snapshot)
        if candidate is None:
            await query.message.reply_text(
                "Не уверен, к какой мысли относится выбор. Уточни её текстом."
            )
            return
        user = await self._user(telegram_user_id)
        kind = "task" if action == "task" else "idea"
        parsed = ParsedThought(
            kind=kind,
            title=candidate[:80],
            next_step=(
                "Уточнить дату; это пока inbox draft, а не reminder" if kind == "task" else None
            ),
        )
        await self._show_preview(
            query.message,
            user.id,
            telegram_user_id,
            chat_id,
            candidate,
            "text",
            parsed,
        )

    async def draft_command_confirmation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        parts = query.data.split(":")
        if len(parts) != 4 or parts[1] not in {
            "save",
            "edit",
            "discard",
            "create_task",
            "cancel",
            "no",
        }:
            await self._stale_callback(query)
            return
        _, action, draft_id, raw_version = parts
        try:
            version = int(raw_version)
        except ValueError:
            await self._stale_callback(query)
            return
        if action == "no":
            await query.answer()
            await query.edit_message_text("Хорошо, карточку не сохраняю.")
            await self.conversation.clear_focus(update.effective_user.id, update.effective_chat.id)
            return
        if action != "save":
            await query.answer("Подтверди это действие текстом или голосом", show_alert=True)
            return
        draft = await self.draft_service.get(draft_id)
        draft_message_id = getattr(draft, "preview_message_id", None)
        if type(draft_message_id) is not int or draft_message_id <= 0:
            draft_message_id = None
        pending_capture = (
            self.nova_companion_pending_capture_anchor(
                update.effective_user.id,
                update.effective_chat.id,
                draft_id,
                version,
                draft_message_id,
            )
            if draft_message_id is not None
            else None
        )
        outcome = await self.action_service.execute(
            "save",
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            source="callback",
            draft_id=draft_id,
            version=version,
            expected_preview_message_id=draft_message_id,
            expected_access_version=(
                pending_capture.access_version if pending_capture is not None else None
            ),
        )
        if outcome.status != "ok":
            await self._stale_callback(query)
            return
        await query.answer()
        receipt = await self._record_saved_receipt(
            update.effective_user.id,
            update.effective_chat.id,
            outcome,
            expected_companion_pending=pending_capture,
        )
        await query.edit_message_text(
            receipt,
            reply_markup=self._saved_receipt_markup(outcome),
        )
        await self.conversation.set_active_draft(
            update.effective_user.id, update.effective_chat.id, None
        )

    async def draft_focus_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        parts = query.data.split(":")
        if parts == ["draftfocus", "cancel"]:
            await query.answer()
            await query.edit_message_text("Выбор карточки отменён.")
            await self.conversation.clear_focus(update.effective_user.id, update.effective_chat.id)
            return
        if len(parts) != 4:
            await self._stale_callback(query)
            return
        _, action, draft_id, raw_version = parts
        try:
            version = int(raw_version)
        except ValueError:
            await self._stale_callback(query)
            return
        draft = await self.draft_service.active_by_id(
            draft_id,
            version,
            update.effective_user.id,
            update.effective_chat.id,
        )
        if draft is None:
            await self._stale_callback(query)
            return
        await self.conversation.set_focus(
            update.effective_user.id,
            update.effective_chat.id,
            draft.id,
            draft.version,
            action,
        )
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        await self._prompt_pending_action(query.message, draft, action)

    async def drafts_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        parts = query.data.split(":")
        if len(parts) != 4 or parts[1] not in {
            "open",
            "save",
            "drop",
            "group",
            "page",
            "cleanup",
        }:
            await self._stale_callback(query)
            return
        _, action, draft_id, raw_version = parts
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        if action == "page":
            try:
                page = max(0, int(draft_id))
            except ValueError:
                await self._stale_callback(query)
                return
            await query.answer()
            await self._send_drafts_page(query.message, telegram_user_id, chat_id, page)
            return
        if action == "cleanup":
            drafts = await self.draft_service.active_drafts(telegram_user_id, chat_id)
            await query.answer()
            await self._begin_system_cleanup(
                query.message,
                telegram_user_id,
                chat_id,
                drafts,
                {draft.id for draft in drafts},
            )
            return
        try:
            version = int(raw_version)
        except ValueError:
            await self._stale_callback(query)
            return
        active = await self.draft_service.active_drafts(telegram_user_id, chat_id)
        draft = next(
            (item for item in active if item.id == draft_id and item.version == version),
            None,
        )
        if draft is None:
            await self._stale_callback(query)
            return
        if action == "group":
            key = self.draft_service.semantic_key(draft)
            affected = {item.id for item in active if self.draft_service.semantic_key(item) == key}
            await query.answer()
            await self._begin_system_cleanup(
                query.message,
                telegram_user_id,
                chat_id,
                active,
                affected,
            )
            return
        if action == "open":
            if draft.status != "preview":
                await query.answer("Сначала заверши редактирование карточки", show_alert=True)
                return
            await self.conversation.set_focus(
                telegram_user_id, chat_id, draft.id, draft.version, None
            )
            await query.answer()
            await self._send_draft_preview(query.message, draft)
            return
        if draft.status != "preview":
            await query.answer("Эта карточка сейчас редактируется", show_alert=True)
            return
        draft_message_id = getattr(draft, "preview_message_id", None)
        if type(draft_message_id) is not int or draft_message_id <= 0:
            draft_message_id = None
        pending_capture = (
            self.nova_companion_pending_capture_anchor(
                telegram_user_id,
                chat_id,
                draft.id,
                draft.version,
                draft_message_id,
            )
            if draft_message_id is not None
            else None
        )
        outcome = await self.action_service.execute(
            "save" if action == "save" else "discard",
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            source="callback",
            draft_id=draft.id,
            version=draft.version,
            expected_preview_message_id=draft_message_id,
            expected_access_version=(
                pending_capture.access_version if pending_capture is not None else None
            ),
        )
        if outcome.status != "ok":
            await self._stale_callback(query)
            return
        await query.answer()
        if action == "save":
            receipt = await self._record_saved_receipt(
                telegram_user_id,
                chat_id,
                outcome,
                expected_companion_pending=pending_capture,
            )
            await query.edit_message_text(
                receipt,
                reply_markup=self._saved_receipt_markup(outcome),
            )
        else:
            self.nova_companion_record_terminal_capture(
                telegram_user_id,
                chat_id,
                outcome,
                expected_pending=pending_capture,
            )
            await query.edit_message_text("Черновик удалён.")
        await self.conversation.clear_focus(telegram_user_id, chat_id)

    async def inbox_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        parts = query.data.split(":")
        if len(parts) != 4 or parts[1] not in {"save", "edit", "drop"}:
            await self._stale_callback(query)
            return
        _, action, draft_id, raw_version = parts
        try:
            version = int(raw_version)
        except ValueError:
            await self._stale_callback(query)
            return
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        canonical_message_id = getattr(query.message, "message_id", None)
        if type(canonical_message_id) is not int or canonical_message_id <= 0:
            await self._stale_callback(query)
            return
        pending_capture = self.nova_companion_pending_capture_anchor(
            telegram_user_id,
            chat_id,
            draft_id,
            version,
            canonical_message_id,
        )
        expected_access_version = (
            pending_capture.access_version if pending_capture is not None else None
        )
        if action == "edit":
            outcome = await self.action_service.execute(
                "edit",
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source="callback",
                draft_id=draft_id,
                version=version,
                expected_preview_message_id=canonical_message_id,
                expected_access_version=expected_access_version,
            )
            if outcome.status != "ok":
                await self._stale_callback(query)
                return
            await query.answer()
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("Пришли исправленный текст одним сообщением")
        elif action == "drop":
            outcome = await self.action_service.execute(
                "discard",
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source="callback",
                draft_id=draft_id,
                version=version,
                expected_preview_message_id=canonical_message_id,
                expected_access_version=expected_access_version,
            )
            if outcome.status != "ok":
                await self._stale_callback(query)
                return
            self.nova_companion_record_terminal_capture(
                telegram_user_id,
                chat_id,
                outcome,
                expected_pending=pending_capture,
            )
            await query.answer()
            await query.edit_message_text("Не сохраняю.")
            await self.conversation.set_active_draft(telegram_user_id, chat_id, None)
        else:
            outcome = await self.action_service.execute(
                "save",
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source="callback",
                draft_id=draft_id,
                version=version,
                expected_preview_message_id=canonical_message_id,
                expected_access_version=expected_access_version,
            )
            if outcome.status != "ok":
                await self._stale_callback(query)
                return
            await query.answer()
            receipt = await self._record_saved_receipt(
                telegram_user_id,
                chat_id,
                outcome,
                expected_companion_pending=pending_capture,
            )
            await query.edit_message_text(
                receipt,
                reply_markup=self._saved_receipt_markup(outcome),
            )
            await self.conversation.set_active_draft(telegram_user_id, chat_id, None)

    @staticmethod
    async def _stale_callback(query: object) -> None:
        await query.answer("Эта карточка уже неактуальна. Создай новую.", show_alert=True)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except TelegramError:
            pass

    async def cancel_draft_edit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self.cancel_workspace_state(update):
            await update.effective_message.reply_text(
                "Операция с пространством отменена. Выбранный контекст не изменён.",
                reply_markup=ReplyKeyboardRemove(),
            )
            return
        collection_cancelled = await self.cancel_collection_state(update)
        if await self.cancel_task_input(update):
            return
        user = await self._user(update.effective_user.id)
        if await self.vision_service.cancel(user.id, update.effective_chat.id):
            await update.effective_message.reply_text(
                "Операция с карточкой отменена, ничего не сохранено и не удалено."
            )
            return
        if await self.cancel_knowledge_state(update):
            await update.effective_message.reply_text(
                "Capture отменён. Источник, задание и оригинал не создавались."
            )
            return
        await self.conversation.clear_focus(update.effective_user.id, update.effective_chat.id)
        await self.conversation.clear_system_action(
            update.effective_user.id, update.effective_chat.id
        )
        editing = await self.draft_service.editing(
            update.effective_user.id,
            update.effective_chat.id,
        )
        editing_message_id = getattr(editing, "preview_message_id", None)
        editing_pending = (
            self.nova_companion_pending_capture_anchor(
                update.effective_user.id,
                update.effective_chat.id,
                editing.id,
                editing.version,
                editing_message_id,
            )
            if editing is not None and type(editing_message_id) is int and editing_message_id > 0
            else None
        )
        discarded = await self.draft_service.cancel_editing(
            update.effective_user.id, update.effective_chat.id
        )
        if discarded:
            if editing is not None:
                self.nova_companion_clear_pending_capture_exact(
                    update.effective_user.id,
                    update.effective_chat.id,
                    editing.id,
                    editing.version,
                    expected_pending=editing_pending,
                )
            await self.conversation.set_active_draft(
                update.effective_user.id, update.effective_chat.id, None
            )
        await update.effective_message.reply_text(
            "Редактирование отменено, ничего не сохранено."
            if discarded
            else (
                "Операция с разделом отменена, активный раздел сброшен."
                if collection_cancelled
                else "Текущий выбор и ожидающее действие отменены."
            )
        )

    async def drafts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._send_drafts_page(
            update.effective_message,
            update.effective_user.id,
            update.effective_chat.id,
            0,
        )

    async def _send_drafts_page(
        self, message: object, telegram_user_id: int, chat_id: int, page: int
    ) -> None:
        drafts = await self.draft_service.active_drafts(telegram_user_id, chat_id)
        if not drafts:
            await message.reply_text("Активных черновиков нет.")
            return
        groups = self._group_drafts(drafts)
        page_size = 5
        max_page = max(0, (len(groups) - 1) // page_size)
        page = min(page, max_page)
        visible = groups[page * page_size : (page + 1) * page_size]
        rows: list[list[InlineKeyboardButton]] = []
        listing: list[str] = []
        for offset, group in enumerate(visible, start=1):
            draft = group[0]
            index = page * page_size + offset
            multiplier = f" ×{len(group)}" if len(group) > 1 else ""
            listing.append(f"{index}. [{LABELS[draft.kind]}] {draft.title}{multiplier}")
            rows.append(
                [
                    InlineKeyboardButton(
                        f"Открыть {index}",
                        callback_data=f"drafts:open:{draft.id}:{draft.version}",
                    ),
                    InlineKeyboardButton(
                        "Сохранить одну",
                        callback_data=f"drafts:save:{draft.id}:{draft.version}",
                    ),
                    InlineKeyboardButton(
                        "Удалить группу",
                        callback_data=f"drafts:group:{draft.id}:{draft.version}",
                    ),
                ]
            )
        pagination = []
        if page > 0:
            pagination.append(
                InlineKeyboardButton("Назад", callback_data=f"drafts:page:{page - 1}:0")
            )
        if page < max_page:
            pagination.append(
                InlineKeyboardButton("Далее", callback_data=f"drafts:page:{page + 1}:0")
            )
        if pagination:
            rows.append(pagination)
        rows.append([InlineKeyboardButton("Очистить активные", callback_data="drafts:cleanup:0:0")])
        await message.reply_text(
            f"Активных черновиков: {len(drafts)}\n"
            f"Страница {page + 1}/{max_page + 1}\n" + "\n".join(listing),
            reply_markup=InlineKeyboardMarkup(rows),
        )

    def _group_drafts(self, drafts: list[DraftInboxItem]) -> list[list[DraftInboxItem]]:
        groups: dict[tuple[str, ...], list[DraftInboxItem]] = {}
        for draft in drafts:
            groups.setdefault(self.draft_service.semantic_key(draft), []).append(draft)
        return list(groups.values())

    async def cleanup_drafts_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        drafts = await self.draft_service.active_drafts(
            update.effective_user.id, update.effective_chat.id
        )
        await self._begin_system_cleanup(
            update.effective_message,
            update.effective_user.id,
            update.effective_chat.id,
            drafts,
            {draft.id for draft in drafts},
        )

    async def _record_saved_receipt(
        self,
        telegram_user_id: int,
        chat_id: int,
        outcome: ActionOutcome,
        *,
        expected_companion_pending: object | None = None,
    ) -> str:
        item = outcome.result.inbox_item if outcome.result else None
        if item is None:
            return "Сохранено в inbox."
        record_companion_status = getattr(
            self,
            "nova_companion_record_confirmed_capture",
            None,
        )
        if callable(record_companion_status):
            record_companion_status(
                telegram_user_id,
                chat_id,
                outcome,
                expected_pending=expected_companion_pending,
            )
        if outcome.result and outcome.result.duplicate:
            return (
                "Такая запись уже есть в Inbox — повторную копию не создаю:\n"
                f"{LABELS[item.kind]} — {item.title}"
            )
        await self.conversation.record_saved(telegram_user_id, chat_id, item.id)
        receipt = f"Сохранено в inbox:\n{LABELS[item.kind]} — {item.title}"
        if item.kind == "task":
            reminder = outcome.result.reminder if outcome.result else None
            if reminder is None:
                receipt += "\nЗадача сохранена без напоминания: сначала укажи дату и время"
            else:
                local_reminder = reminder.remind_at
                if local_reminder.tzinfo is None:
                    local_reminder = local_reminder.replace(tzinfo=UTC)
                local_reminder = local_reminder.astimezone(ZoneInfo(reminder.timezone))
                receipt += (
                    "\nНапоминание: "
                    f"{local_reminder.strftime('%d.%m.%Y %H:%M')} ({reminder.timezone})"
                )
        return receipt

    @staticmethod
    def _saved_receipt_markup(outcome: ActionOutcome) -> InlineKeyboardMarkup | None:
        """Offer a safe, immediately discoverable trash action after saving."""

        result = outcome.result
        item = result.inbox_item if result else None
        if item is None or result.duplicate:
            return None
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🗑 В корзину",
                        callback_data=f"ibox:trash:{item.id}",
                    )
                ],
                [InlineKeyboardButton("Открыть Inbox", callback_data="ibox:page:0")],
            ]
        )

    async def last_saved_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        snapshot = await self.conversation.get(update.effective_user.id, update.effective_chat.id)
        async with self.db.sessions() as session:
            item = None
            if snapshot.last_saved_inbox_item_id is not None:
                item = await session.scalar(
                    select(InboxItem).where(
                        InboxItem.id == snapshot.last_saved_inbox_item_id,
                        InboxItem.user_id == user.id,
                        InboxItem.status == "confirmed",
                    )
                )
            if item is None:
                item = await session.scalar(
                    select(InboxItem)
                    .where(
                        InboxItem.user_id == user.id,
                        InboxItem.status == "confirmed",
                    )
                    .order_by(InboxItem.id.desc())
                    .limit(1)
                )
        if item is None:
            await update.effective_message.reply_text("В inbox пока нет сохранённых записей.")
            return
        await update.effective_message.reply_text(
            f"Последняя сохранённая запись:\n{LABELS[item.kind]} — {item.title}",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Открыть",
                            callback_data=f"ibox:view:{item.id}",
                        ),
                        InlineKeyboardButton(
                            "🗑 В корзину",
                            callback_data=f"ibox:trash:{item.id}",
                        ),
                    ],
                    [InlineKeyboardButton("Все записи Inbox", callback_data="ibox:page:0")],
                ]
            ),
        )

    async def inbox(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        await self._send_saved_inbox_page(
            update.effective_message,
            user.id,
            0,
            trashed=False,
            edit=False,
        )

    async def saved_inbox_action(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        del context
        query = update.callback_query
        parts = (query.data or "").split(":")
        if len(parts) != 3 or parts[0] != "ibox":
            await self._stale_callback(query)
            return
        action, raw_value = parts[1], parts[2]
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        if action in {"page", "trashlist"}:
            try:
                page = max(0, int(raw_value))
            except ValueError:
                await self._stale_callback(query)
                return
            await query.answer()
            await self._send_saved_inbox_page(
                query,
                user.id,
                page,
                trashed=action == "trashlist",
                edit=True,
            )
            return
        if action == "cleanup" and raw_value == "commands":
            snapshot = await self.inbox_lifecycle.command_garbage_snapshot(user.id)
            await query.answer()
            await self._begin_inbox_trash(
                query.message,
                update.effective_user.id,
                chat_id,
                snapshot,
                action="trash_inbox_commands",
            )
            return
        if action == "trashpage":
            try:
                raw_page, signature = raw_value.split(".", maxsplit=1)
                page = max(0, int(raw_page))
            except (ValueError, AttributeError):
                await self._stale_callback(query)
                return
            if not re.fullmatch(r"[0-9a-f]{12}", signature):
                await self._stale_callback(query)
                return
            page_size = 6
            async with self.db.sessions() as session:
                ordered_item_ids = list(
                    await session.scalars(
                        select(InboxItem.id)
                        .where(
                            InboxItem.user_id == user.id,
                            InboxItem.status == "confirmed",
                        )
                        .order_by(InboxItem.id.desc())
                        .offset(page * page_size)
                        .limit(page_size)
                    )
                )
            if self._inbox_page_signature(ordered_item_ids) != signature:
                await query.answer(
                    "Страница Inbox изменилась — обнови список перед массовым действием",
                    show_alert=True,
                )
                return
            item_ids = set(ordered_item_ids)
            snapshot = await self.inbox_lifecycle.confirmed_snapshot(user.id, item_ids)
            await query.answer()
            await self._begin_inbox_trash(
                query.message,
                update.effective_user.id,
                chat_id,
                snapshot,
            )
            return
        try:
            item_id = int(raw_value)
        except ValueError:
            await self._stale_callback(query)
            return
        if item_id <= 0:
            await self._stale_callback(query)
            return
        if action == "trash":
            snapshot = await self.inbox_lifecycle.confirmed_snapshot(user.id, {item_id})
            if not snapshot:
                await query.answer("Запись уже изменилась", show_alert=True)
                return
            await query.answer()
            await self._begin_inbox_trash(
                query.message,
                update.effective_user.id,
                chat_id,
                snapshot,
            )
            return
        if action == "restore":
            snapshot = await self.inbox_lifecycle.trashed_snapshot(user.id, {item_id})
            if not snapshot:
                await query.answer("Запись уже изменилась", show_alert=True)
                return
            result = await self.inbox_lifecycle.restore_snapshot(user.id, snapshot)
            if result.status != "restored":
                await query.answer("Запись изменилась — обнови корзину", show_alert=True)
                return
            await query.answer()
            restored_task = snapshot[0].get("kind") == "task"
            restored_status = snapshot[0].get("pre_trash_status")
            destination_text = (
                "Задача снова доступна в разделе «Задачи и напоминания». "
                if restored_task
                else "Запись снова доступна в Inbox. "
            )
            if restored_task and restored_status == "archived":
                destination_text = "Архивная задача снова доступна в разделе задач. "
            restore_text = f"Запись восстановлена. {destination_text}"
            if restored_task:
                restore_text += (
                    "Прежнее напоминание осталось выключенным — включи новое явно через /tasks."
                )
            await self._inbox_edit_or_reply(
                query,
                restore_text,
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Открыть задачи" if restored_task else "Открыть Inbox",
                                callback_data="task:hub" if restored_task else "ibox:page:0",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "🗑 Вернуться в корзину",
                                callback_data="ibox:trashlist:0",
                            )
                        ],
                    ]
                ),
            )
            return
        if action in {"view", "trashview"}:
            expected_status = "trashed" if action == "trashview" else "confirmed"
            async with self.db.sessions() as session:
                item = await session.scalar(
                    select(InboxItem).where(
                        InboxItem.id == item_id,
                        InboxItem.user_id == user.id,
                        InboxItem.status == expected_status,
                    )
                )
            if item is None:
                await query.answer("Запись уже изменилась", show_alert=True)
                return
            await query.answer()
            await self._send_saved_inbox_card(query, item, trashed=expected_status == "trashed")
            return
        await self._stale_callback(query)

    async def _send_saved_inbox_page(
        self,
        target: object,
        owner_id: int,
        page: int,
        *,
        trashed: bool,
        edit: bool,
    ) -> None:
        page_size = 6
        status = "trashed" if trashed else "confirmed"
        async with self.db.sessions() as session:
            total = int(
                await session.scalar(
                    select(func.count(InboxItem.id)).where(
                        InboxItem.user_id == owner_id,
                        InboxItem.status == status,
                    )
                )
                or 0
            )
            pages = max(1, (total + page_size - 1) // page_size)
            safe_page = min(max(page, 0), pages - 1)
            items = (
                await session.scalars(
                    select(InboxItem)
                    .where(InboxItem.user_id == owner_id, InboxItem.status == status)
                    .order_by(InboxItem.id.desc())
                    .offset(safe_page * page_size)
                    .limit(page_size)
                )
            ).all()
        rows: list[list[InlineKeyboardButton]] = []
        lines: list[str] = []
        for index, item in enumerate(items, start=safe_page * page_size + 1):
            label = LABELS.get(item.kind, item.kind)
            title = _truncate_utf16(item.title, 44)
            lines.append(f"{index}. [{label}] {title}")
            rows.append(
                [
                    InlineKeyboardButton(
                        f"Открыть {index}",
                        callback_data=(
                            f"ibox:trashview:{item.id}" if trashed else f"ibox:view:{item.id}"
                        ),
                    ),
                    *(
                        []
                        if trashed
                        else [
                            InlineKeyboardButton(
                                f"В корзину {index}",
                                callback_data=f"ibox:trash:{item.id}",
                            )
                        ]
                    ),
                ]
            )
        pagination: list[InlineKeyboardButton] = []
        prefix = "trashlist" if trashed else "page"
        if safe_page > 0:
            pagination.append(
                InlineKeyboardButton("← Назад", callback_data=f"ibox:{prefix}:{safe_page - 1}")
            )
        if safe_page + 1 < pages:
            pagination.append(
                InlineKeyboardButton("Далее →", callback_data=f"ibox:{prefix}:{safe_page + 1}")
            )
        if pagination:
            rows.append(pagination)
        if trashed:
            rows.extend(
                [
                    [InlineKeyboardButton("← К Inbox", callback_data="ibox:page:0")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
                ]
            )
            heading = "🗑 Корзина Inbox"
            empty = "Корзина пуста."
        else:
            page_signature = self._inbox_page_signature([item.id for item in items])
            rows.extend(
                [
                    *(
                        [
                            [
                                InlineKeyboardButton(
                                    "🗑 В корзину эту страницу",
                                    callback_data=(f"ibox:trashpage:{safe_page}.{page_signature}"),
                                )
                            ]
                        ]
                        if items
                        else []
                    ),
                    [
                        InlineKeyboardButton(
                            "🧹 Убрать ошибочные команды",
                            callback_data="ibox:cleanup:commands",
                        )
                    ],
                    [InlineKeyboardButton("🗑 Корзина", callback_data="ibox:trashlist:0")],
                    [InlineKeyboardButton("✅ Задачи", callback_data="task:hub")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
                ]
            )
            heading = "Последние сохранённые записи Inbox"
            empty = "Inbox пока пуст."
        listing = "\n".join(lines) if lines else empty
        text = f"{heading} — {total}\nСтраница {safe_page + 1}/{pages}\n\n{listing}"
        markup = InlineKeyboardMarkup(rows)
        if edit:
            await self._inbox_edit_or_reply(target, text, markup)
        else:
            await target.reply_text(text, reply_markup=markup)

    @staticmethod
    def _inbox_page_signature(item_ids: list[int]) -> str:
        payload = ",".join(str(item_id) for item_id in item_ids).encode("ascii")
        return blake2s(payload, digest_size=6).hexdigest()

    async def _send_saved_inbox_card(
        self,
        query: object,
        item: InboxItem,
        *,
        trashed: bool,
    ) -> None:
        details = ""
        if item.description:
            details += f"\n\nОписание: {_truncate_utf16(item.description, 1_800)}"
        if item.next_step:
            details += f"\nСледующий шаг: {_truncate_utf16(item.next_step, 600)}"
        text = (
            f"{LABELS.get(item.kind, item.kind).capitalize()}: "
            f"{_truncate_utf16(item.title, 700)}{details}"
        )
        if trashed:
            rows = [
                [
                    InlineKeyboardButton(
                        "↩️ Восстановить",
                        callback_data=f"ibox:restore:{item.id}",
                    )
                ],
                [InlineKeyboardButton("← К корзине", callback_data="ibox:trashlist:0")],
            ]
        else:
            rows = [
                [
                    InlineKeyboardButton(
                        "🗑 В корзину",
                        callback_data=f"ibox:trash:{item.id}",
                    )
                ],
                [InlineKeyboardButton("← К Inbox", callback_data="ibox:page:0")],
            ]
        rows.append([InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")])
        await self._inbox_edit_or_reply(query, text, InlineKeyboardMarkup(rows))

    @staticmethod
    async def _inbox_edit_or_reply(
        query: object,
        text: str,
        markup: InlineKeyboardMarkup,
    ) -> None:
        await edit_callback_screen(
            query,
            text,
            markup,
            operation="inbox",
        )

    async def today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        task = asyncio.create_task(
            self._today_lifecycle(update, context),
            name="weekly-review-today-lifecycle",
        )
        self._weekly_review_track_task(task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Today delivery failed operation=weekly_today error_type=%s",
                type(exc).__name__,
            )

    async def _today_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        user = await self._user(update.effective_user.id)
        if not user.onboarding_completed:
            await update.effective_message.reply_text(
                "Сначала заверши Vision Profile через /start."
            )
            return
        include_weekly_focus = self.weekly_review_policy.allows_actor(user)
        try:
            snapshot = await self.focus_service.materialize_today_application(
                user.id,
                include_weekly_focus=include_weekly_focus,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Today plan failed operation=materialize error_type=%s",
                type(exc).__name__,
            )
            return
        try:
            provider_allowed = await self._today_application_is_current(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Today plan failed operation=pre_provider_fence error_type=%s",
                type(exc).__name__,
            )
            return
        if not provider_allowed:
            return
        try:
            plan = await self.focus_service.generate_today_plan(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Today plan failed operation=provider error_type=%s",
                type(exc).__name__,
            )
            try:
                fallback_allowed = await self._today_application_is_current(snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as fence_exc:
                logger.warning(
                    "Today plan failed operation=provider_failure_fence error_type=%s",
                    type(fence_exc).__name__,
                )
                return
            if not fallback_allowed:
                return
            await update.effective_message.reply_text(
                "Не удалось собрать фокус дня. Попробуй немного позже."
            )
            return
        try:
            delivery_allowed = await self._today_application_is_current(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Today plan failed operation=pre_send_fence error_type=%s",
                type(exc).__name__,
            )
            return
        if not delivery_allowed:
            return
        actions = "\n".join(f"{i}. {action}" for i, action in enumerate(plan.actions, 1))
        weekly_line = (
            f"🎯 Фокус недели: {snapshot.weekly_focus}\n\n"
            if snapshot.includes_weekly_focus and snapshot.weekly_focus
            else ""
        )
        sent = await update.effective_message.reply_text(
            f"{weekly_line}{plan.vision_reminder}\n\nФокус: {plan.main_focus}\n{actions}\n\n"
            f"На сложный день: {plan.hard_day_minimum}"
        )
        try:
            delivered_current = await self._today_application_is_current(snapshot)
        except asyncio.CancelledError:
            await self._today_neutralize_delivery(
                context,
                update,
                sent,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Today delivery failed operation=post_send_fence error_type=%s",
                type(exc).__name__,
            )
            delivered_current = False
        if delivered_current:
            return
        await self._today_neutralize_delivery(context, update, sent)

    async def _today_neutralize_delivery(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        update: Update,
        sent: Any,
    ) -> None:
        message_id = getattr(sent, "message_id", None)
        if isinstance(message_id, int) and message_id > 0:
            await self._weekly_review_neutralize_sent(
                context.bot,
                update.effective_chat.id,
                message_id,
            )
            return
        delete = getattr(sent, "delete", None)
        if not callable(delete):
            logger.error(
                "Today delivery cleanup failed operation=missing_message_id "
                "error_type=MissingMessageId"
            )
            return
        try:
            await delete()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Today delivery cleanup failed operation=delete error_type=%s",
                type(exc).__name__,
            )

    async def _today_application_is_current(
        self,
        snapshot: TodayApplicationSnapshot,
    ) -> bool:
        check = await self.focus_service.check_today_application(snapshot)
        if not check.is_current:
            return False
        if not snapshot.includes_weekly_focus:
            return True
        return self.weekly_review_policy.allows_actor(
            snapshot,
            expected_access_version=snapshot.access_version,
        )

    async def evening_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await self.reminder_clear_current(update)
        context.user_data["evening"] = {}
        await update.effective_message.reply_text(
            "Что сегодня получилось? Даже небольшой шаг считается."
        )
        return EVENING_WORKED

    async def evening_worked(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["evening"]["worked"] = update.effective_message.text
        await update.effective_message.reply_text("Что не получилось или пришлось пропустить?")
        return EVENING_FAILED

    async def evening_failed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["evening"]["did_not_work"] = update.effective_message.text
        await update.effective_message.reply_text("Какой был уровень энергии от 1 до 5?")
        return EVENING_ENERGY

    async def evening_energy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["evening"]["energy"] = int(update.effective_message.text)
        await update.effective_message.reply_text("Какое препятствие было главным?")
        return EVENING_OBSTACLE

    async def evening_obstacle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["evening"]["obstacle"] = update.effective_message.text
        await update.effective_message.reply_text("Что перенести или изменить завтра?")
        return EVENING_TOMORROW

    async def evening_tomorrow(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        answers = context.user_data.pop("evening", {})
        answers["tomorrow_adjustment"] = update.effective_message.text
        answers["completed_actions"] = [answers["worked"]] if answers.get("worked") else []
        answers["skipped_actions"] = (
            [answers["did_not_work"]] if answers.get("did_not_work") else []
        )
        user = await self._user(update.effective_user.id)
        local_day = datetime.now(ZoneInfo(user.timezone)).date()
        async with self.db.session() as session:
            await CheckInRepository(session).save_evening(user.id, local_day, answers)
        await update.effective_message.reply_text(
            "Рефлексия сохранена. То, что не сработало, — данные для настройки завтрашнего плана."
        )
        return ConversationHandler.END

    async def cancel_evening(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.pop("evening", None)
        await update.effective_message.reply_text("Рефлексия отменена, ничего не сохранено.")
        return ConversationHandler.END

    async def health_checkin_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await self.reminder_clear_current(update)
        user = await self._user(update.effective_user.id)
        record_id = None
        command_text = update.effective_message.text or ""
        if command_text.startswith("/health_edit"):
            args = getattr(context, "args", [])
            if not args or not args[0].isdigit():
                await update.effective_message.reply_text(
                    "Укажи ID записи: /health_edit 12. ID виден в /health."
                )
                return ConversationHandler.END
            record_id = int(args[0])
            if await self.health_service.get_owned(user.id, record_id) is None:
                await update.effective_message.reply_text("Такой health-записи у тебя нет.")
                return ConversationHandler.END
        context.user_data["health_checkin"] = {"record_id": record_id}
        prefix = "Исправляем запись. " if record_id is not None else ""
        await update.effective_message.reply_text(
            f"{prefix}Энергия от 0 до 10? Отвечай одним числом."
        )
        return HEALTH_ENERGY

    async def health_energy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["health_checkin"]["energy"] = int(update.effective_message.text)
        await update.effective_message.reply_text("Сон от 0 до 10?")
        return HEALTH_SLEEP

    @staticmethod
    async def health_invalid_rating(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text("Нужно одно целое число от 0 до 10.")

    async def health_sleep(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["health_checkin"]["sleep"] = int(update.effective_message.text)
        await update.effective_message.reply_text("Настроение от 0 до 10?")
        return HEALTH_MOOD

    async def health_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["health_checkin"]["mood"] = int(update.effective_message.text)
        await update.effective_message.reply_text("Стресс от 0 до 10, где 10 — максимальный?")
        return HEALTH_STRESS

    async def health_stress(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["health_checkin"]["stress"] = int(update.effective_message.text)
        await update.effective_message.reply_text("Физическое самочувствие от 0 до 10?")
        return HEALTH_PHYSICAL

    async def health_physical(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data["health_checkin"]["physical_wellbeing"] = int(
            update.effective_message.text
        )
        await update.effective_message.reply_text(
            "Есть симптомы или наблюдения? Напиши кратко или ответь «нет»."
        )
        return HEALTH_SYMPTOMS

    async def health_symptoms(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        answers = context.user_data.pop("health_checkin", {})
        symptoms = (update.effective_message.text or "").strip()
        answers["symptoms"] = (
            None if symptoms.lower() in {"нет", "нет симптомов", "-"} else symptoms
        )
        record_id = answers.pop("record_id", None)
        user = await self._user(update.effective_user.id)
        record = await self.health_service.save(
            user_id=user.id,
            timezone=user.timezone,
            answers=answers,
            record_id=record_id,
        )
        if record is None:
            await update.effective_message.reply_text(
                "Не удалось изменить запись: она не найдена или принадлежит другому пользователю."
            )
            return ConversationHandler.END
        response = (
            f"Health check-in сохранён. Субъективная линейка состояния: "
            f"{record.state_score}/100.\n"
            "Это инструмент самонаблюдения, а не медицинский диагноз."
        )
        if urgent := urgent_safety_message(record.symptoms):
            response += f"\n\n{urgent}"
        weakness_days = await self.health_service.recent_weakness_days(user.id)
        if weakness := prolonged_weakness_message(record.symptoms, weakness_days):
            response += f"\n\n{weakness}"
        await update.effective_message.reply_text(response)
        return ConversationHandler.END

    async def cancel_health_checkin(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        context.user_data.pop("health_checkin", None)
        await update.effective_message.reply_text("Health check-in отменён, ничего не сохранено.")
        return ConversationHandler.END

    async def health_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        records = await self.health_service.history(user.id, limit=14)
        if not records:
            await update.effective_message.reply_text(
                "Health-история пока пуста. Начни с /checkin.\n"
                "Линейка 0–100 субъективна и не является медицинским диагнозом."
            )
            return
        latest = records[0]
        report = await self.health_service.weekly_report(user.id, user.timezone)
        current_lines = [
            f"Текущее состояние за {latest.local_date.strftime('%d.%m.%Y')}:",
            f"Линейка: {latest.state_score}/100",
            f"Энергия {latest.energy}/10 · Сон {latest.sleep}/10 · Настроение {latest.mood}/10",
            f"Стресс {latest.stress}/10 · Физическое самочувствие {latest.physical_wellbeing}/10",
        ]
        if latest.symptoms:
            current_lines.append(f"Наблюдения: {latest.symptoms}")
        current_lines.append(
            "Линейка субъективна, показывает динамику самонаблюдения и не является диагнозом."
        )
        if report.current_count:
            current_lines.append(f"\nНеделя: {report.current_count} check-in.")
            for name, value in report.current.items():
                change = report.changes[name]
                suffix = (
                    " · нет предыдущей недели" if change is None else f" · изменение {change:+.1f}"
                )
                current_lines.append(f"{METRIC_LABELS[name]}: {value:.1f}{suffix}")
        history = ", ".join(
            f"#{record.id} {record.local_date.strftime('%d.%m')} — {record.state_score}/100"
            for record in records[:7]
        )
        current_lines.append(f"\nИстория: {history}")
        current_lines.append(
            "Исправить: /health_edit ID · удалить: /health_delete ID\n"
            "Напоминание: /health_reminder_on 20:00 или /health_reminder_off"
        )
        await update.effective_message.reply_text("\n".join(current_lines))

    async def health_delete_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        args = getattr(context, "args", [])
        if not args or not args[0].isdigit():
            await update.effective_message.reply_text(
                "Укажи ID записи: /health_delete 12. ID виден в /health."
            )
            return
        user = await self._user(update.effective_user.id)
        deleted = await self.health_service.delete_owned(user.id, int(args[0]))
        await update.effective_message.reply_text(
            "Health-запись удалена." if deleted else "Такая health-запись не найдена."
        )

    async def health_reminder_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = getattr(context, "args", [])
        raw_time = args[0] if args else "20:00"
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", raw_time):
            await update.effective_message.reply_text(
                "Время нужно в формате HH:MM, например /health_reminder_on 20:00."
            )
            return
        try:
            local_time = time.fromisoformat(raw_time)
        except ValueError:
            await update.effective_message.reply_text(
                "Время нужно в формате HH:MM, например /health_reminder_on 20:00."
            )
            return
        user = await self._user(update.effective_user.id)
        await self.health_service.set_reminder(
            user_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            timezone=user.timezone,
            local_time=local_time,
            enabled=True,
        )
        if self.scheduler:
            self.scheduler.schedule_health_reminder(
                user_id=user.id,
                chat_id=update.effective_user.id,
                timezone=user.timezone,
                local_time=local_time,
            )
        await update.effective_message.reply_text(
            f"Ежедневное добровольное напоминание включено на {local_time.strftime('%H:%M')} "
            f"({user.timezone}). Отключить: /health_reminder_off."
        )

    async def health_reminder_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        disabled = await self.health_service.disable_reminder(user.id)
        if self.scheduler:
            self.scheduler.remove_health_reminder(user.id)
        await update.effective_message.reply_text(
            "Health-напоминание отключено." if disabled else "Health-напоминание не было включено."
        )

    async def doctor_prepare_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await self.reminder_clear_current(update)
        user = await self._user(update.effective_user.id)
        command = (update.effective_message.text or "").split(maxsplit=1)[0]
        command = command.split("@", maxsplit=1)[0]
        record_id = None
        if command == "/doctor_prepare_edit":
            args = getattr(context, "args", [])
            if not args or not args[0].isdigit():
                await update.effective_message.reply_text(
                    "Укажи ID: /doctor_prepare_edit 12. ID виден в /doctor_preparations."
                )
                return ConversationHandler.END
            record_id = int(args[0])
            if await self.doctor_prep_service.get_owned(user.id, record_id) is None:
                await update.effective_message.reply_text(
                    "Такая подготовка не найдена или принадлежит другому пользователю."
                )
                return ConversationHandler.END
        context.user_data["doctor_prepare"] = {"record_id": record_id}
        prefix = "Исправляем подготовку. " if record_id is not None else ""
        await update.effective_message.reply_text(
            f"{prefix}Кратко: какова основная причина обращения к врачу?"
        )
        return DOCTOR_REASON

    async def doctor_prepare_reason(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        reason = update.effective_message.text.strip()
        if not reason:
            await update.effective_message.reply_text(
                "Причина обращения не должна быть пустой. Опиши её одной фразой."
            )
            return DOCTOR_REASON
        context.user_data["doctor_prepare"]["reason"] = reason
        prompt = "Как долго это продолжается? Например: «5 дней» или «около месяца»."
        if urgent := urgent_safety_message(reason):
            prompt = f"{urgent}\nНе жди завершения опроса для обращения за помощью.\n\n{prompt}"
        await update.effective_message.reply_text(prompt)
        return DOCTOR_DURATION

    async def doctor_prepare_duration(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        duration = update.effective_message.text.strip()
        if not duration:
            await update.effective_message.reply_text(
                "Длительность не должна быть пустой. Например: «5 дней»."
            )
            return DOCTOR_DURATION
        context.user_data["doctor_prepare"]["duration"] = duration
        await update.effective_message.reply_text(
            "Перечисли симптомы и наблюдения фактически, без попытки поставить диагноз."
        )
        return DOCTOR_SYMPTOMS

    async def doctor_prepare_symptoms(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        symptoms = update.effective_message.text.strip()
        if not symptoms:
            await update.effective_message.reply_text(
                "Симптомы или наблюдения не должны быть пустыми. Если симптомов нет, "
                "так и напиши: «нет симптомов»."
            )
            return DOCTOR_SYMPTOMS
        context.user_data["doctor_prepare"]["symptoms"] = symptoms
        prompt = "Какие лекарства, витамины или добавки ты принимаешь? Если нет — ответь «нет»."
        reason = context.user_data["doctor_prepare"].get("reason", "")
        if urgent := urgent_safety_message(f"{reason}. {symptoms}"):
            prompt = f"{urgent}\nНе жди завершения опроса для обращения за помощью.\n\n{prompt}"
        await update.effective_message.reply_text(prompt)
        return DOCTOR_MEDICATIONS

    async def doctor_prepare_medications(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        context.user_data["doctor_prepare"]["medications"] = update.effective_message.text.strip()
        await update.effective_message.reply_text(
            "Какие вопросы хочешь задать врачу? Если пока нет — ответь «нет»."
        )
        return DOCTOR_QUESTIONS

    async def doctor_prepare_questions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        answers = context.user_data.pop("doctor_prepare", {})
        answers["questions"] = update.effective_message.text.strip()
        record_id = answers.pop("record_id", None)
        user = await self._user(update.effective_user.id)
        try:
            record = await self.doctor_prep_service.save(
                user_id=user.id,
                timezone=user.timezone,
                answers=answers,
                record_id=record_id,
            )
        except ValueError:
            await update.effective_message.reply_text(
                "Не удалось сохранить: обязательные ответы не должны быть пустыми. "
                "Запусти /doctor_prepare ещё раз."
            )
            return ConversationHandler.END
        if record is None:
            await update.effective_message.reply_text(
                "Не удалось изменить запись: она не найдена или принадлежит другому пользователю."
            )
            return ConversationHandler.END

        response = (
            f"Подготовка #{record.id} сохранена.\n\n{record.summary}\n\n"
            f"Исправить: /doctor_prepare_edit {record.id}\n"
            f"Удалить: /doctor_prepare_delete {record.id}\n"
            f"Создать задачу с reminder: /doctor_prepare_task {record.id} через 2 часа"
        )
        safety_text = f"{record.reason}. {record.duration}. {record.symptoms}"
        if urgent := urgent_safety_message(safety_text):
            response += (
                f"\n\n{urgent}\nОбычная запись к врачу и reminder не заменяют срочную помощь."
            )
        weakness_days = await self.health_service.recent_weakness_days(user.id)
        if weakness := prolonged_weakness_message(safety_text, weakness_days):
            response += f"\n\n{weakness}"
        await update.effective_message.reply_text(response)
        return ConversationHandler.END

    async def cancel_doctor_prepare(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        context.user_data.pop("doctor_prepare", None)
        await update.effective_message.reply_text(
            "Подготовка к визиту отменена, медицинская запись не создана."
        )
        return ConversationHandler.END

    async def doctor_preparations(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        records = await self.doctor_prep_service.history(user.id, limit=10)
        if not records:
            await update.effective_message.reply_text(
                "Подготовок к врачу пока нет. Начать: /doctor_prepare."
            )
            return
        lines = ["Твои подготовки к врачу:"]
        for record in records:
            reason = " ".join(record.reason.split())
            if len(reason) > 80:
                reason = f"{reason[:77]}..."
            lines.append(f"#{record.id} — {reason}")
        lines.append("Открыть: /doctor_prepare_show ID")
        await update.effective_message.reply_text("\n".join(lines))

    async def doctor_prepare_show(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = getattr(context, "args", [])
        if not args or not args[0].isdigit():
            await update.effective_message.reply_text("Укажи ID: /doctor_prepare_show 12.")
            return
        user = await self._user(update.effective_user.id)
        record = await self.doctor_prep_service.get_owned(user.id, int(args[0]))
        if record is None:
            await update.effective_message.reply_text(
                "Такая подготовка не найдена или принадлежит другому пользователю."
            )
            return
        await update.effective_message.reply_text(
            f"Подготовка #{record.id}\n\n{record.summary}\n\n"
            f"Исправить: /doctor_prepare_edit {record.id} · "
            f"удалить: /doctor_prepare_delete {record.id}"
        )

    async def doctor_prepare_delete(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        args = getattr(context, "args", [])
        if not args or not args[0].isdigit():
            await update.effective_message.reply_text("Укажи ID: /doctor_prepare_delete 12.")
            return
        user = await self._user(update.effective_user.id)
        deleted = await self.doctor_prep_service.delete_owned(user.id, int(args[0]))
        await update.effective_message.reply_text(
            "Подготовка к врачу удалена."
            if deleted
            else "Такая подготовка не найдена или принадлежит другому пользователю."
        )

    async def doctor_prepare_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = getattr(context, "args", [])
        if len(args) < 2 or not args[0].isdigit():
            await update.effective_message.reply_text(
                "Формат: /doctor_prepare_task ID через 2 часа "
                "или /doctor_prepare_task ID завтра в 10:00."
            )
            return
        user = await self._user(update.effective_user.id)
        record_id = int(args[0])
        record = await self.doctor_prep_service.get_owned(user.id, record_id)
        if record is None:
            await update.effective_message.reply_text(
                "Такая подготовка не найдена или принадлежит другому пользователю."
            )
            return
        expression = " ".join(args[1:]).strip()
        temporal = self._doctor_task_temporal(expression, user.timezone)
        if temporal is None:
            await update.effective_message.reply_text(
                "Не понял будущее время reminder. Примеры: «через 2 часа», "
                "«завтра в 10:00», «20 июля в 09:30»."
            )
            return
        result = await self.doctor_prep_service.create_appointment_task(
            user_id=user.id,
            record_id=record_id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            temporal=temporal,
        )
        if result.status == "existing":
            await update.effective_message.reply_text(
                "Задача «Записаться к врачу» для этой подготовки уже создана; дубликат не добавлен."
            )
            return
        if result.status != "created" or result.reminder is None:
            await update.effective_message.reply_text(
                "Не удалось создать задачу с reminder. Медицинская запись не изменена."
            )
            return
        local_reminder = result.reminder.remind_at
        if local_reminder.tzinfo is None:
            local_reminder = local_reminder.replace(tzinfo=UTC)
        local_reminder = local_reminder.astimezone(ZoneInfo(user.timezone))
        response = (
            "Задача «Записаться к врачу» создана без медицинских подробностей. "
            f"Reminder: {local_reminder.strftime('%d.%m.%Y %H:%M')} ({user.timezone})."
        )
        if urgent_safety_message(f"{record.reason}. {record.symptoms}"):
            response += " Эта задача не заменяет срочную медицинскую помощь."
        await update.effective_message.reply_text(response)

    def _doctor_task_temporal(
        self, expression: str, timezone_name: str
    ) -> TemporalResolution | None:
        relative_command = (
            expression if expression.lower().startswith("напомни") else f"Напомни {expression}"
        )
        relative = self.date_resolver.resolve_relative_reminder(
            f"{relative_command} Записаться к врачу",
            timezone_name,
        )
        if relative is not None:
            return relative.temporal
        resolution = self.date_resolver.resolve(expression, timezone_name)
        if resolution.status != "resolved" or resolution.target_date is None:
            return None
        local_time = self.date_resolver.extract_local_time(expression)
        precision = "datetime" if local_time is not None else "date"
        local_time = local_time or time(hour=self.settings.task_date_event_hour)
        local_event = datetime.combine(
            resolution.target_date,
            local_time,
            tzinfo=ZoneInfo(timezone_name),
        )
        event_at = local_event.astimezone(UTC)
        if event_at <= datetime.now(UTC):
            return None
        return TemporalResolution(
            resolved_at=event_at,
            remind_at=event_at if precision == "datetime" else None,
            timezone=timezone_name,
            resolved_local_date=resolution.target_date,
            resolved_local_time=local_time if precision == "datetime" else None,
            precision=precision,
            original_expression=expression,
            resolution_status="resolved",
        )

    async def doctor_find(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        location = location_from_user(user)
        if location is None:
            await update.effective_message.reply_text(
                "Сначала настрой локацию: /location Саратов. "
                "Для маршрута: /location Саратов → Энгельс."
            )
            return
        await update.effective_message.reply_text(
            self.doctor_search_service.format_directory(location)
        )

    async def doctor_find_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = getattr(context, "args", [])
        if not args:
            await update.effective_message.reply_text(
                "Формат: /doctor_find_task через 2 часа или /doctor_find_task завтра в 10:00."
            )
            return
        user = await self._user(update.effective_user.id)
        if location_from_user(user) is None:
            await update.effective_message.reply_text(
                "Сначала настрой локацию через /location, затем создай задачу."
            )
            return
        expression = " ".join(args).strip()
        temporal = self._doctor_task_temporal(expression, user.timezone)
        if temporal is None:
            await update.effective_message.reply_text(
                "Не понял будущее время reminder. Примеры: «через 2 часа», "
                "«завтра в 10:00», «20 июля в 09:30»."
            )
            return
        result = await self.doctor_search_service.create_booking_task(
            user_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            temporal=temporal,
        )
        if result.status == "missing_location":
            await update.effective_message.reply_text(
                "Локация не найдена. Настрой её через /location."
            )
            return
        if result.inbox_item is None:
            await update.effective_message.reply_text("Не удалось создать задачу.")
            return
        if result.status == "existing":
            await update.effective_message.reply_text(
                f"Задача «{result.inbox_item.title}» уже создана; дубликат не добавлен."
            )
            return
        if result.reminder is None:
            await update.effective_message.reply_text(
                "Не удалось создать reminder; задача не должна использоваться без времени."
            )
            return
        local_reminder = result.reminder.remind_at
        if local_reminder.tzinfo is None:
            local_reminder = local_reminder.replace(tzinfo=UTC)
        local_reminder = local_reminder.astimezone(ZoneInfo(user.timezone))
        await update.effective_message.reply_text(
            f"Задача «{result.inbox_item.title}» создана. "
            f"Reminder: {local_reminder.strftime('%d.%m.%Y %H:%M')} ({user.timezone})."
        )

    async def legacy_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "/start — онбординг, /profile — профиль, /goals — обновить цели, /today — фокус дня, "
            "/evening — рефлексия, /inbox — сохранённые мысли, /drafts — активные черновики, "
            "/last_saved — последняя запись, /cleanup_drafts — безопасная очистка черновиков, "
            "/health — состояние и динамика, /checkin — health check-in, "
            "/doctor_prepare — подготовка к визиту к врачу, "
            "/location — личный город или маршрут, "
            "/timezone — проверить или изменить часовой пояс, "
            "/vision — персональная карта желаний, "
            "/doctor_find — официальный поиск терапевта по твоей локации."
        )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        if isinstance(context.error, asyncio.CancelledError):
            raise context.error
        log_safe_failure("Unhandled Telegram update error", context.error)
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "Что-то пошло не так. Попробуй ещё раз немного позже."
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_safe_failure("Could not send safe error message", exc)
        raise ApplicationHandlerStop
