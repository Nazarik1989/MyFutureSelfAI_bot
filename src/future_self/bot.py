import logging
import re
import warnings
from datetime import UTC, date, datetime, time
from html import escape
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.constants import ChatType
from telegram.error import TelegramError
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

from .actions import (
    ActionCommandRouter,
    ActionOutcome,
    ActionRoute,
    DraftAction,
    DraftActionService,
)
from .ai import AIService
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
)
from .drafts import DraftInboxService
from .health import (
    METRIC_LABELS,
    HealthService,
    prolonged_weakness_message,
    urgent_safety_message,
)
from .models import DraftInboxItem, Goal, InboxItem, Routine, User, VisionProfile
from .natural_commands import NaturalAction, NaturalCommandRouter
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
from .transcription import TranscriptionError, TranscriptionService

logger = logging.getLogger(__name__)

ONBOARDING_INPUT, PROFILE_CONFIRM = range(2)
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


def log_safe_failure(event: str, exc: BaseException | None, *, user_id: int | None = None) -> None:
    """Log operational metadata without provider messages, prompts, audio, or secrets."""
    error_type = type(exc).__name__ if exc else "Unknown"
    if user_id is None:
        logger.error("%s error_type=%s", event, error_type)
    else:
        logger.error("%s error_type=%s user_id=%s", event, error_type, user_id)


class FutureSelfBot:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        ai: AIService,
        transcription: TranscriptionService,
    ):
        self.settings = settings
        self.db = db
        self.ai = ai
        self.transcription = transcription
        self.draft_service = DraftInboxService(
            db,
            settings.inbox_draft_ttl_minutes,
            task_date_event_hour=settings.task_date_event_hour,
            task_reminder_lead_minutes=settings.task_reminder_lead_minutes,
        )
        self.action_service = DraftActionService(self.draft_service)
        self.action_router = ActionCommandRouter()
        self.system_action_router = SystemActionRouter()
        self.natural_command_router = NaturalCommandRouter()
        self.conversation = ConversationContextService(
            db,
            settings.conversation_context_messages,
            settings.conversation_context_ttl_hours,
            settings.draft_focus_ttl_minutes,
            settings.system_action_ttl_minutes,
        )
        self.date_resolver = DateResolver()
        self.intent_router = IntentRouter(ai, settings.intent_confidence_threshold)
        self.focus_service = FocusService(db, ai)
        self.health_service = HealthService(db)
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

    @property
    def voice_enabled(self) -> bool:
        return self.settings.enable_voice and getattr(self.transcription, "enabled", True)

    def build(self) -> Application:
        app = (
            Application.builder()
            .token(self.settings.telegram_bot_token)
            .post_init(self._post_init)
            .build()
        )
        # This assistant handles profiles, health notes and reminders. Telegram
        # group/channel replies would disclose that data to other chat members,
        # so stop every non-private update before any feature handler sees it.
        app.add_handler(TypeHandler(Update, self.private_chat_guard), group=-1)
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
                ],
                states={
                    ONBOARDING_INPUT: [
                        CommandHandler("back", self.onboarding_back),
                        CommandHandler("skip", self.onboarding_skip),
                        MessageHandler(filters.Regex("^Назад$"), self.onboarding_back),
                        MessageHandler(filters.Regex("^Пропустить$"), self.onboarding_skip),
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self.onboarding_answer),
                    ],
                    PROFILE_CONFIRM: [
                        CallbackQueryHandler(self.profile_action, pattern=r"^profile:")
                    ],
                },
                fallbacks=[
                    CommandHandler("cancel", self.cancel_onboarding),
                    MessageHandler(filters.Regex("^Отменить$"), self.cancel_onboarding),
                ],
                allow_reentry=True,
            )
        evening = ConversationHandler(
            entry_points=[CommandHandler("evening", self.evening_start)],
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
            fallbacks=[CommandHandler("cancel", self.cancel_evening)],
        )
        health_checkin = ConversationHandler(
            entry_points=[
                CommandHandler("checkin", self.health_checkin_start),
                CommandHandler("health_edit", self.health_checkin_start),
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
            fallbacks=[CommandHandler("cancel", self.cancel_health_checkin)],
        )
        doctor_prepare = ConversationHandler(
            entry_points=[
                CommandHandler("doctor_prepare", self.doctor_prepare_start),
                CommandHandler("doctor_prepare_edit", self.doctor_prepare_start),
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
            fallbacks=[CommandHandler("cancel", self.cancel_doctor_prepare)],
            allow_reentry=True,
        )
        app.add_handler(onboarding)
        app.add_handler(evening)
        app.add_handler(health_checkin)
        app.add_handler(doctor_prepare)
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("profile", self.profile))
        app.add_handler(CommandHandler("goals", self.goals_command))
        app.add_handler(CommandHandler("inbox", self.inbox))
        app.add_handler(CommandHandler("drafts", self.drafts_command))
        app.add_handler(CommandHandler("last_saved", self.last_saved_command))
        app.add_handler(CommandHandler("cleanup_drafts", self.cleanup_drafts_command))
        app.add_handler(CommandHandler("today", self.today))
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
        app.add_handler(CallbackQueryHandler(self.intent_action, pattern=r"^intent:"))
        app.add_handler(CallbackQueryHandler(self.context_action, pattern=r"^context:"))
        app.add_handler(
            CallbackQueryHandler(self.draft_command_confirmation, pattern=r"^draftcmd:")
        )
        app.add_handler(CallbackQueryHandler(self.draft_focus_action, pattern=r"^draftfocus:"))
        app.add_handler(CallbackQueryHandler(self.drafts_action, pattern=r"^drafts:"))
        app.add_handler(CallbackQueryHandler(self.system_draft_action, pattern=r"^sysdraft:"))
        app.add_handler(CallbackQueryHandler(self.inbox_action, pattern=r"^inbox:"))
        app.add_handler(CallbackQueryHandler(self.goals_action, pattern=r"^goals:"))
        app.add_handler(CallbackQueryHandler(self.routines_action, pattern=r"^routines:"))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.voice))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.text))
        app.add_error_handler(self.error_handler)
        return app

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
        if update.callback_query is not None:
            await update.callback_query.answer(message, show_alert=True)
        elif update.effective_message is not None:
            await update.effective_message.reply_text(message)
        raise ApplicationHandlerStop

    async def _post_init(self, app: Application) -> None:
        async def send(telegram_id: int, text: str) -> int | None:
            message = await app.bot.send_message(chat_id=telegram_id, text=text)
            return getattr(message, "message_id", None)

        if app.job_queue is None:
            logger.warning("JobQueue is unavailable; scheduled messages are disabled")
            return
        self.scheduler = JobQueueScheduler(
            app.job_queue,
            send,
            self.settings.morning_hour,
            self.settings.evening_hour,
            self.settings.weekly_review_weekday,
            self.settings.enable_weekly_review,
        )
        if self.settings.enable_task_reminders:
            self.reminder_engine = TaskReminderEngine(
                self.db,
                send,
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
        async with self.db.sessions() as session:
            users = (await session.scalars(select(User).where(User.onboarding_completed))).all()
        for user in users:
            self.scheduler.schedule_user(user.telegram_id, user.timezone)
        for preference in await self.health_service.reminder_preferences():
            self.scheduler.schedule_health_reminder(
                user_id=preference.user_id,
                chat_id=preference.telegram_user_id,
                timezone=preference.timezone,
                local_time=preference.local_time,
            )

    async def _user(self, telegram_id: int) -> User:
        async with self.db.session() as session:
            return await UserRepository(session).get_or_create(
                telegram_id, self.settings.default_timezone
            )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = await self._user(update.effective_user.id)
        if user.onboarding_completed:
            await update.effective_message.reply_text(
                f"С возвращением, {user.display_name or 'друг'}! Команды: /today, /evening, /profile."
            )
            return ConversationHandler.END
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user.id)
            if state.status == "cancelled":
                state.status = "in_progress"
            step = state.current_step
        context.user_data["onboarding_user_id"] = user.id
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
            try:
                summary = await self.ai.summarize_vision(dict(state.answers))
            except Exception as exc:
                log_safe_failure("Vision resume failed", exc, user_id=user.id)
                await update.effective_message.reply_text(
                    "Ответы сохранены, но профиль сейчас не удалось собрать. Попробуй /start позже."
                )
                return ConversationHandler.END
            context.user_data["vision_summary"] = summary.model_dump()
            await update.effective_message.reply_text(
                self._profile_text(summary),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton("Подтвердить", callback_data="profile:confirm"),
                            InlineKeyboardButton("Редактировать", callback_data="profile:edit"),
                        ]
                    ]
                ),
            )
            return PROFILE_CONFIRM
        await self._ask_question(update, step)
        return ONBOARDING_INPUT

    async def _ask_question(self, update: Update, step: int) -> None:
        _, question, required = ONBOARDING_QUESTIONS[step]
        suffix = "" if required else " (можно пропустить)"
        await update.effective_message.reply_text(question + suffix, reply_markup=NAVIGATION)

    async def _state(self, context: ContextTypes.DEFAULT_TYPE) -> tuple[int, int, dict[str, str]]:
        user_id = int(context.user_data["onboarding_user_id"])
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user_id)
            return user_id, state.current_step, dict(state.answers)

    async def onboarding_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id, step, answers = await self._state(context)
        try:
            answers = OnboardingFlow.answer(answers, step, update.effective_message.text)
            if ONBOARDING_QUESTIONS[step][0] == "timezone":
                from .domain import validate_timezone

                validate_timezone(update.effective_message.text.strip())
        except ValueError as exc:
            await update.effective_message.reply_text(str(exc))
            return ONBOARDING_INPUT
        return await self._advance_onboarding(update, context, user_id, step, answers)

    async def onboarding_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id, step, answers = await self._state(context)
        try:
            answers = OnboardingFlow.answer(answers, step, None)
        except ValueError as exc:
            await update.effective_message.reply_text(str(exc))
            return ONBOARDING_INPUT
        return await self._advance_onboarding(update, context, user_id, step, answers)

    async def _advance_onboarding(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
        step: int,
        answers: dict[str, str],
    ) -> int:
        next_step = OnboardingFlow.next_step(step)
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user_id)
            state.answers, state.current_step = answers, next_step
        if next_step < len(ONBOARDING_QUESTIONS):
            await self._ask_question(update, next_step)
            return ONBOARDING_INPUT
        await update.effective_message.reply_text("Собираю профиль без добавления фактов от себя…")
        try:
            summary = await self.ai.summarize_vision(answers)
        except Exception as exc:
            log_safe_failure("Vision summary failed", exc, user_id=user_id)
            await update.effective_message.reply_text(
                "Не удалось подготовить профиль. Ответы сохранены — запусти /start позже."
            )
            return ConversationHandler.END
        context.user_data["vision_summary"] = summary.model_dump()
        await update.effective_message.reply_text(
            self._profile_text(summary),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("Подтвердить", callback_data="profile:confirm"),
                        InlineKeyboardButton("Редактировать", callback_data="profile:edit"),
                    ]
                ]
            ),
        )
        return PROFILE_CONFIRM

    async def onboarding_back(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id, step, _ = await self._state(context)
        previous = OnboardingFlow.previous_step(step)
        async with self.db.session() as session:
            state = await OnboardingRepository(session).get_or_create(user_id)
            state.current_step = previous
        await self._ask_question(update, previous)
        return ONBOARDING_INPUT

    async def cancel_onboarding(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = context.user_data.get("onboarding_user_id")
        if user_id:
            async with self.db.session() as session:
                state = await OnboardingRepository(session).get_or_create(int(user_id))
                state.status = "cancelled"
        await update.effective_message.reply_text(
            "Онбординг остановлен. Ответы сохранены; продолжить — /start."
        )
        return ConversationHandler.END

    async def profile_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        if query.data == "profile:edit":
            user_id = int(context.user_data["onboarding_user_id"])
            async with self.db.session() as session:
                state = await OnboardingRepository(session).get_or_create(user_id)
                state.current_step = 0
            await query.edit_message_text("Хорошо, пройдём ответы ещё раз.")
            await self._ask_question(update, 0)
            return ONBOARDING_INPUT
        user_id, _, answers = await self._state(context)
        summary = VisionSummary.model_validate(context.user_data["vision_summary"])
        async with self.db.session() as session:
            user = await session.get(User, user_id)
            from .repositories import ProfileRepository

            await ProfileRepository(session).upsert(user, answers, summary)
            user.display_name = answers.get("display_name")
            user.timezone = answers.get("timezone", user.timezone)
            state = await OnboardingRepository(session).get_or_create(user_id)
            state.status = "completed"
        if self.scheduler:
            self.scheduler.schedule_user(user.telegram_id, user.timezone)
        await query.edit_message_text("Профиль сохранён. Теперь предложу цели.")
        await self._propose_goals(query, user_id, summary)
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
                )
            )
        )

    @staticmethod
    def _profile_text(profile: VisionSummary) -> str:
        return (
            f"Твой Vision Profile:\n{profile.summary}\n\n"
            f"Ценности: {', '.join(profile.values) or 'не указаны'}\n"
            f"Желаемая идентичность: {', '.join(profile.desired_identity) or 'не указана'}\n"
            f"Ограничения: {', '.join(profile.constraints) or 'не указаны'}\n"
            f"Стиль поддержки: {profile.motivation_style or 'не указан'}"
        )

    async def text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        await self._route_message(update, context, update.effective_message.text, "text")

    async def voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        media = update.effective_message.voice or update.effective_message.audio
        if not self.voice_enabled:
            message = (
                "Голосовой ввод сейчас отключён."
                if not self.settings.enable_voice
                else "Распознавание голосовых временно не настроено. Пришли мысль текстом."
            )
            await update.effective_message.reply_text(message)
            return
        if media.duration and media.duration > self.settings.max_audio_seconds:
            await update.effective_message.reply_text(
                "Аудио слишком длинное. Пришли запись короче трёх минут."
            )
            return
        if media.file_size and media.file_size > self.settings.max_audio_bytes:
            await update.effective_message.reply_text("Аудиофайл слишком большой.")
            return
        mime = getattr(media, "mime_type", None)
        if mime and not (mime.startswith("audio/") or mime == "application/ogg"):
            await update.effective_message.reply_text("Этот формат аудио не поддерживается.")
            return
        progress = await update.effective_message.reply_text("Расшифровываю голосовую мысль…")
        try:
            telegram_file = await media.get_file()
            audio = bytes(await telegram_file.download_as_bytearray())
            if len(audio) > self.settings.max_audio_bytes:
                raise ValueError("Audio exceeds limit")
            filename = getattr(media, "file_name", None) or "voice.ogg"
            text = await self.transcription.transcribe(audio, filename)
        except (TranscriptionError, TelegramError, ValueError) as exc:
            log_safe_failure("Voice processing failed", exc)
            await progress.edit_text(
                "Не удалось распознать голосовое. Попробуй ещё раз или пришли текст."
            )
            return
        await progress.edit_text(f"Я услышал: «{text}»")
        await self._route_message(update, context, text, "voice")

    async def _route_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, source: str
    ) -> None:
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        telegram_user_id = update.effective_user.id
        snapshot = await self.conversation.get(telegram_user_id, chat_id)
        natural_command = self.natural_command_router.route(text)
        if natural_command is not None:
            await self._handle_natural_command(update, context, natural_command.action)
            return
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
        editing = await self.draft_service.editing(update.effective_user.id, chat_id)
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
        try:
            result = await self.intent_router.route(
                text, user.timezone, conversation_context=prompt_context
            )
        except Exception as exc:
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
            answer = result.answer or "Я тебя услышал. Можешь уточнить, чем помочь?"
            if self._is_task_question(text):
                await self._show_task_choices(update.effective_message, session_id, answer)
            else:
                await update.effective_message.reply_text(answer)
            await self.conversation.append(
                telegram_user_id,
                chat_id,
                role="assistant",
                content=answer,
                source="text",
                intent="answer",
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
        handlers = {
            "show_drafts": self.drafts_command,
            "show_inbox": self.inbox,
            "show_last_saved": self.last_saved_command,
            "show_profile": self.profile,
            "show_today": self.today,
            "help": self.help_command,
        }
        await handlers[action](update, context)

    async def _confirm_pending_date(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: User,
        snapshot: ConversationSnapshot,
        selected_date: date,
        source: str,
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
        await self.conversation.append(
            telegram_user_id,
            chat_id,
            role="user",
            content=update.effective_message.text or "Выбрана дата",
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
            update.effective_message.text or "Выбрана дата",
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
        if route.kind == "pending":
            await update.effective_message.reply_text(
                "Ожидаю отдельное подтверждение удаления: «да, удалить» или кнопка «Отмена»."
            )
            return
        if route.kind == "cancel":
            await self.conversation.clear_system_action(telegram_user_id, chat_id)
            await update.effective_message.reply_text(
                "Удаление черновиков отменено. Ничего не изменено."
            )
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
        if route.action == "list_drafts":
            await self.drafts_command(update, context)
            return
        if route.action == "show_last_saved":
            await self.last_saved_command(update, context)
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
                "Подтверждение удаления отсутствует или истекло. Запусти /cleanup_drafts снова."
            )
            return False
        result = await self.draft_service.discard_snapshot(
            telegram_user_id, chat_id, snapshot.system_draft_snapshot
        )
        await self.conversation.clear_system_action(telegram_user_id, chat_id)
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
            await self.conversation.clear_system_action(
                update.effective_user.id, update.effective_chat.id
            )
            await query.answer()
            await query.edit_message_text("Удаление черновиков отменено.")
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
        await self._deactivate_preview_keyboard(context, chat_id, outcome.previous_message_id)
        if action == "save":
            await self.conversation.clear_focus(telegram_user_id, chat_id)
            await self.conversation.set_active_draft(telegram_user_id, chat_id, None)
            channel = "голосовой" if source == "voice" else "текстовой"
            receipt = await self._record_saved_receipt(telegram_user_id, chat_id, outcome)
            await update.effective_message.reply_text(
                f"Сохранено в inbox по {channel} команде.\n{receipt}"
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
    async def _show_task_choices(message: object, session_id: int, answer: str) -> None:
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
        await message.reply_text(answer, reply_markup=keyboard)

    async def _send_draft_preview(
        self,
        message: object,
        draft: DraftInboxItem,
        *,
        include_original: bool = True,
    ) -> None:
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
        if message_id := getattr(preview_message, "message_id", None):
            await self.draft_service.set_preview_message(draft.id, message_id)

    async def _show_unknown(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        source: str,
        result: IntentResult,
    ) -> None:
        token = uuid4().hex[:12]
        context.user_data[f"intent:{token}"] = PendingIntent(token, text, source, result)
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
        await update.effective_message.reply_text(
            "Что сделать с этим сообщением?", reply_markup=keyboard
        )

    async def intent_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        _, action, token = query.data.split(":", 2)
        pending = context.user_data.get(f"intent:{token}")
        if not isinstance(pending, PendingIntent) or pending.handled:
            await query.answer("Это действие уже обработано", show_alert=True)
            return
        pending.handled = True
        await query.answer()
        if action == "drop":
            await query.edit_message_text("Хорошо, ничего не делаю.")
            return
        if action == "answer":
            user = await self._user(update.effective_user.id)
            snapshot = await self.conversation.get(
                update.effective_user.id, update.effective_chat.id
            )
            try:
                answer = await self.intent_router.answer(
                    pending.raw_text,
                    user.timezone,
                    conversation_context=snapshot.for_prompt(),
                )
            except Exception as exc:
                log_safe_failure("Unknown intent answer failed", exc, user_id=user.id)
                await query.edit_message_text("Не удалось ответить сейчас. Ничего не сохранено.")
                return
            await query.edit_message_text(answer.answer)
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
        outcome = await self.action_service.execute(
            "save",
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            source="callback",
            draft_id=draft_id,
            version=version,
        )
        if outcome.status != "ok":
            await self._stale_callback(query)
            return
        await query.answer()
        receipt = await self._record_saved_receipt(
            update.effective_user.id, update.effective_chat.id, outcome
        )
        await query.edit_message_text(receipt)
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
        outcome = await self.action_service.execute(
            "save" if action == "save" else "discard",
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            source="callback",
            draft_id=draft.id,
            version=draft.version,
        )
        if outcome.status != "ok":
            await self._stale_callback(query)
            return
        await query.answer()
        if action == "save":
            receipt = await self._record_saved_receipt(telegram_user_id, chat_id, outcome)
            await query.edit_message_text(receipt)
        else:
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
        if action == "edit":
            outcome = await self.action_service.execute(
                "edit",
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source="callback",
                draft_id=draft_id,
                version=version,
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
            )
            if outcome.status != "ok":
                await self._stale_callback(query)
                return
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
            )
            if outcome.status != "ok":
                await self._stale_callback(query)
                return
            await query.answer()
            receipt = await self._record_saved_receipt(telegram_user_id, chat_id, outcome)
            await query.edit_message_text(receipt)
            await self.conversation.set_active_draft(telegram_user_id, chat_id, None)

    @staticmethod
    async def _stale_callback(query: object) -> None:
        await query.answer("Эта карточка уже неактуальна. Создай новую.", show_alert=True)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except TelegramError:
            pass

    async def cancel_draft_edit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.conversation.clear_focus(update.effective_user.id, update.effective_chat.id)
        await self.conversation.clear_system_action(
            update.effective_user.id, update.effective_chat.id
        )
        discarded = await self.draft_service.cancel_editing(
            update.effective_user.id, update.effective_chat.id
        )
        if discarded:
            await self.conversation.set_active_draft(
                update.effective_user.id, update.effective_chat.id, None
            )
        await update.effective_message.reply_text(
            "Редактирование отменено, ничего не сохранено."
            if discarded
            else "Текущий выбор и ожидающее действие отменены."
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
    ) -> str:
        item = outcome.result.inbox_item if outcome.result else None
        if item is None:
            return "Сохранено в inbox."
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
            f"Последняя сохранённая запись:\n{LABELS[item.kind]} — {item.title}"
        )

    async def inbox(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        async with self.db.sessions() as session:
            items = (
                await session.scalars(
                    select(InboxItem)
                    .where(InboxItem.user_id == user.id, InboxItem.status == "confirmed")
                    .order_by(InboxItem.id.desc())
                    .limit(10)
                )
            ).all()
        text = (
            "\n".join(f"• [{LABELS[item.kind]}] {item.title}" for item in items)
            or "Inbox пока пуст."
        )
        await update.effective_message.reply_text(text)

    async def today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        if not user.onboarding_completed:
            await update.effective_message.reply_text(
                "Сначала заверши Vision Profile через /start."
            )
            return
        try:
            plan = await self.focus_service.generate(user.id)
        except Exception as exc:
            log_safe_failure("Today plan failed", exc, user_id=user.id)
            await update.effective_message.reply_text(
                "Не удалось собрать фокус дня. Попробуй немного позже."
            )
            return
        actions = "\n".join(f"{i}. {action}" for i, action in enumerate(plan.actions, 1))
        await update.effective_message.reply_text(
            f"{plan.vision_reminder}\n\nФокус: {plan.main_focus}\n{actions}\n\n"
            f"На сложный день: {plan.hard_day_minimum}"
        )

    async def evening_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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
        await update.effective_message.reply_text(self.doctor_search_service.format_directory())

    async def doctor_find_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = getattr(context, "args", [])
        if not args:
            await update.effective_message.reply_text(
                "Формат: /doctor_find_task через 2 часа или /doctor_find_task завтра в 10:00."
            )
            return
        user = await self._user(update.effective_user.id)
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
        if result.status == "existing":
            await update.effective_message.reply_text(
                "Задача «Записаться к терапевту: Светогорск → Выборг» уже создана; "
                "дубликат не добавлен."
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
            "Задача «Записаться к терапевту: Светогорск → Выборг» создана. "
            f"Reminder: {local_reminder.strftime('%d.%m.%Y %H:%M')} ({user.timezone})."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "/start — онбординг, /profile — профиль, /goals — обновить цели, /today — фокус дня, "
            "/evening — рефлексия, /inbox — сохранённые мысли, /drafts — активные черновики, "
            "/last_saved — последняя запись, /cleanup_drafts — безопасная очистка черновиков, "
            "/health — состояние и динамика, /checkin — health check-in, "
            "/doctor_prepare — подготовка к визиту к врачу, "
            "/doctor_find — официальный поиск терапевта Светогорск → Выборг."
        )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log_safe_failure("Unhandled Telegram update error", context.error)
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "Что-то пошло не так. Попробуй ещё раз немного позже."
                )
            except TelegramError as exc:
                log_safe_failure("Could not send safe error message", exc)
