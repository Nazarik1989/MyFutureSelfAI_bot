from __future__ import annotations

import asyncio
import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from .access import ADMIN, BLOCKED, GUEST, AccessTier, require_access_tier
from .navigation import help_topics, navigation_actions, navigation_sections
from .nova import (
    NovaCapability,
    NovaCapabilityKind,
    NovaCatalog,
    NovaResolution,
    NovaResolutionKind,
    NovaRuntimeFlags,
    NovaSession,
    build_nova_catalog,
    extract_explicit_nova_question,
    is_nova_help_intent,
    resolve_nova_question,
)

logger = logging.getLogger(__name__)

NOVA_ROOT_TEXT = """✨ Nova

Привет! Расскажи своими словами, что хочешь сделать.
Я найду нужную функцию и проведу тебя по шагам.

Например:
• «Как добавить задачу с напоминанием?»
• «Где мои желания?»
• «Как загрузить референс?»
• «Как изменить часовой пояс?»"""

NOVA_PROVIDER_FAILURE_TEXT = (
    "Не получилось разобрать вопрос автоматически. Выбери раздел или сформулируй короче."
)
NOVA_LOCAL_CLARIFY_TEXT = (
    "Я пока не нашла точную функцию. Выбери раздел или уточни, что именно хочешь сделать в боте."
)
NOVA_ACCESS_CHANGED_TEXT = "Доступ изменился. Открой /help, чтобы начать заново."
NOVA_STALE_ALERT = "Эта кнопка устарела или недоступна."
NOVA_BUSY_ALERT = "Nova уже разбирает вопрос."

_NOVA_ROOT_TOPICS = (
    ("quick", "🚀 Быстрый старт"),
    ("requests", "🧭 Возможности"),
    ("examples", "💬 Примеры вопросов"),
    ("privacy", "🔒 Данные и безопасность"),
)
_NOVA_CONVERSATION_ACTIONS = frozenset({"evening", "checkin", "doctor_prepare", "onboarding"})
_MEDIA_TEXT_EDIT_ERRORS = (
    "there is no text in the message to edit",
    "message is not a text message",
    "message to edit is not a text message",
)
_MISSING_MESSAGE_ERRORS = (
    "message to edit not found",
    "message_id_invalid",
    "message identifier is not specified",
)


class _NovaAccessChanged(Exception):
    pass


class _NovaScreenUpdate:
    def __init__(self, source: Any, message: Any):
        self._source = source
        self.effective_message = message
        self.message = message

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class _NovaCallbackMessage:
    def __init__(self, owner: NovaHandlers, query: Any, fallback_markup: Any):
        self._owner = owner
        self._query = query
        self._message = query.message
        self._fallback_markup = fallback_markup

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)

    async def reply_text(self, text: str, **kwargs: Any) -> Any:
        reply_markup = kwargs.pop("reply_markup", self._fallback_markup)
        await self._owner._edit_or_send(self._query, text, reply_markup, **kwargs)
        return self._message


class NovaHandlers:
    nova_sessions: Any

    def _nova_flags(self) -> NovaRuntimeFlags:
        return NovaRuntimeFlags.from_settings(self.settings)

    def _nova_catalog(self, tier: AccessTier) -> NovaCatalog:
        return build_nova_catalog(tier, self._nova_flags())

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        tier = require_access_tier(user.access_tier)
        if tier == BLOCKED:
            await self.show_blocked_screen(update)
            return
        flow = await self._active_navigation_flow(update, context) if tier != GUEST else None
        if flow is not None:
            await self._nova_flow_help(update.effective_message, update, flow)
            return
        await self._nova_open_message(update.effective_message, update, user)

    async def _nova_open_message(self, message: Any, update: Any, user: Any) -> bool:
        await self.nova_memory_clear_current(update)
        published = None
        async with self._nova_ui_lock:
            current = await self.nova_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if current is not None and current.question_in_progress:
                return True
            sent = await message.reply_text(
                NOVA_ROOT_TEXT,
                reply_markup=self._nova_root_keyboard(require_access_tier(user.access_tier)),
            )
            message_id = self._positive_message_id(getattr(sent, "message_id", None))
            if message_id is None:
                logger.warning(
                    "Nova canonical creation failed operation=reply error_type=MissingMessageId"
                )
                return False
            published = await self.nova_sessions.create(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                access_version=user.access_version,
                canonical_message_id=message_id,
                tier=require_access_tier(user.access_tier),
            )
        if published is not None:
            await self._nova_memory_clear_if_guided_current(published)
        return True

    async def nova_navigation_help_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        topic_key: str | None = None,
    ) -> None:
        query = update.callback_query
        user = await self._user(update.effective_user.id)
        tier = require_access_tier(user.access_tier)
        flow = await self._active_navigation_flow(update, context) if tier != GUEST else None
        if flow is not None:
            await query.answer()
            await self._nova_flow_help(query.message, update, flow, query=query)
            return
        await self.nova_memory_clear_current(update)
        if topic_key is None:
            text = NOVA_ROOT_TEXT
            markup = self._nova_root_keyboard(tier)
        else:
            topic = self._nova_topics().get(topic_key)
            if topic is None:
                await query.answer(NOVA_STALE_ALERT, show_alert=True)
                return
            text = f"{topic[0]}\n\n{topic[1]}"
            markup = self._nova_back_keyboard(tier)
        await query.answer()
        published = None
        async with self._nova_ui_lock:
            current = await self.nova_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if current is not None and current.question_in_progress:
                return
            message_id = await self._nova_edit_callback(
                query,
                text,
                markup,
                operation="help",
            )
            if message_id is not None:
                published = await self.nova_sessions.create(
                    owner_id=user.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    access_version=user.access_version,
                    canonical_message_id=message_id,
                    tier=tier,
                )
        if published is not None:
            await self._nova_memory_clear_if_guided_current(published)

    async def nova_non_text_gate(self, update: Update, context: Any) -> None:
        """Leave Nova for photo/document; voice and audio may continue the session."""

        message = update.effective_message
        if (
            getattr(message, "voice", None) is not None
            or getattr(message, "audio", None) is not None
        ):
            return
        if await self._active_navigation_flow(update, context) is not None:
            await self.nova_memory_clear_current(update)
            return
        if await self.nova_memory_media_gate(update, context):
            from telegram.ext import ApplicationHandlerStop

            raise ApplicationHandlerStop
        await self.nova_memory_clear_current(update)
        await self.reminder_clear_current(update)
        await self.nova_clear_current(update)

    async def nova_text_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        user: Any | None = None,
    ) -> bool:
        message = update.effective_message
        text = getattr(message, "text", None)
        if not isinstance(text, str):
            return False
        user = user or await self._user(update.effective_user.id)
        return await self._nova_question_gate(
            update,
            context,
            text,
            user=user,
            candidate_message=None,
            allow_replacement=True,
        )

    async def nova_voice_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        transcript: str,
        progress: Any,
        *,
        user: Any,
        expected_session: NovaSession | None,
    ) -> bool:
        """Route an STT transcript through Nova while reusing one canonical message."""

        return await self._nova_question_gate(
            update,
            context,
            transcript,
            user=user,
            candidate_message=progress,
            allow_replacement=False,
            expected_voice_session=expected_session,
            voice_session_fenced=True,
        )

    async def _nova_question_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        *,
        user: Any,
        candidate_message: Any | None,
        allow_replacement: bool,
        expected_voice_session: NovaSession | None = None,
        voice_session_fenced: bool = False,
    ) -> bool:
        tier = require_access_tier(user.access_tier)
        if tier == BLOCKED:
            return False
        current = await self.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        natural_router = getattr(self, "natural_command_router", None)
        if (
            candidate_message is None
            and current is None
            and natural_router is not None
            and natural_router.route(text) is not None
        ):
            return False
        catalog = self._nova_catalog(tier)
        explicit_question = extract_explicit_nova_question(text)
        standalone_intent = is_nova_help_intent(text, catalog)
        voice_generation_changed = voice_session_fenced and not self._nova_voice_session_matches(
            current,
            expected_voice_session,
        )
        intent_session = (
            (expected_voice_session or current) if voice_generation_changed else current
        )
        if not is_nova_help_intent(
            text,
            catalog,
            active_session=intent_session is not None,
            last_action_id=(intent_session.last_action_id if intent_session is not None else None),
        ):
            return False
        if tier != GUEST:
            flow = await self._active_navigation_flow(update, context)
            if flow is not None:
                if current is not None:
                    await self.nova_clear_bound(
                        owner_id=user.id,
                        chat_id=update.effective_chat.id,
                        session_id=current.id,
                    )
                return False
        question = explicit_question if explicit_question is not None else text.strip()
        await self.nova_memory_clear_current(update)
        started: NovaSession | None = None
        async with self._nova_launch_lock:
            if tier != GUEST and await self._active_navigation_flow(update, context) is not None:
                if current is not None:
                    await self.nova_clear_bound(
                        owner_id=user.id,
                        chat_id=update.effective_chat.id,
                        session_id=current.id,
                    )
                return False
            async with self._nova_ui_lock:
                current = await self.nova_sessions.current(
                    owner_id=user.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                )
                if voice_session_fenced and not self._nova_voice_session_matches(
                    current,
                    expected_voice_session,
                ):
                    live_tier = await self._nova_validate_access_values(
                        update.effective_user.id,
                        user.access_version,
                        tier,
                    )
                    if live_tier is None:
                        if expected_voice_session is None:
                            await self._nova_edit_transient_access_changed(candidate_message)
                        else:
                            await self._nova_edit_canonical(
                                context,
                                expected_voice_session,
                                NOVA_ACCESS_CHANGED_TEXT,
                                None,
                                source_message=update.effective_message,
                                allow_replacement=False,
                            )
                            await self._nova_delete_transient(candidate_message, context)
                    else:
                        await self._nova_delete_transient(candidate_message, context)
                    return True
                if current is None and not standalone_intent:
                    return False
                if current is not None and (
                    current.access_version != user.access_version or current.tier != tier
                ):
                    await self.nova_sessions.clear(
                        owner_id=user.id,
                        chat_id=update.effective_chat.id,
                        session_id=current.id,
                    )
                    await self._nova_edit_canonical(
                        context,
                        current,
                        NOVA_ACCESS_CHANGED_TEXT,
                        None,
                        source_message=update.effective_message,
                        allow_replacement=allow_replacement,
                    )
                    if candidate_message is not None:
                        await self._nova_delete_transient(candidate_message, context)
                    return True
                if current is None:
                    live_tier = await self._nova_validate_access_values(
                        update.effective_user.id,
                        user.access_version,
                        tier,
                    )
                    if live_tier is None:
                        if candidate_message is not None:
                            await self._nova_edit_transient_access_changed(candidate_message)
                        return True
                    placeholder = candidate_message
                    if placeholder is None:
                        placeholder = await update.effective_message.reply_text(
                            "✨ Nova\n\nРазбираю вопрос…"
                        )
                    message_id = self._positive_message_id(getattr(placeholder, "message_id", None))
                    if message_id is None:
                        logger.warning(
                            "Nova canonical creation failed operation=question_reply "
                            "error_type=MissingMessageId"
                        )
                        return True
                    current = await self.nova_sessions.create(
                        owner_id=user.id,
                        telegram_user_id=update.effective_user.id,
                        chat_id=update.effective_chat.id,
                        access_version=user.access_version,
                        canonical_message_id=message_id,
                        tier=live_tier,
                    )
                elif candidate_message is not None:
                    candidate_id = self._positive_message_id(
                        getattr(candidate_message, "message_id", None)
                    )
                    if candidate_id != current.canonical_message_id:
                        await self._nova_delete_transient(candidate_message, context)
                started = await self.nova_sessions.begin_question(
                    owner_id=current.owner_id,
                    telegram_user_id=current.telegram_user_id,
                    chat_id=current.chat_id,
                    access_version=current.access_version,
                    canonical_message_id=current.canonical_message_id,
                    tier=current.tier,
                    session_id=current.id,
                )
        if started is not None:
            await self._nova_memory_clear_if_guided_current(started)
            await self._nova_process_question(
                update,
                context,
                started,
                question,
                allow_replacement=allow_replacement,
            )
        return True

    @staticmethod
    def _nova_voice_session_matches(
        current: NovaSession | None,
        expected: NovaSession | None,
    ) -> bool:
        if expected is None:
            return current is None
        return current is not None and current.id == expected.id

    async def _nova_process_question(
        self,
        update: Any,
        context: Any,
        started: NovaSession,
        question: str,
        *,
        allow_replacement: bool,
    ) -> None:
        active = started
        try:
            live_tier = await self._nova_validate_access(active)
            if live_tier is None:
                await self._nova_discard_for_access(
                    context,
                    active,
                    source_message=update.effective_message,
                    allow_replacement=allow_replacement,
                )
                return
            if not question.strip() or len(question.strip()) > 600:
                resolution = NovaResolution(
                    kind=NovaResolutionKind.CLARIFY,
                    response=(
                        "Расскажи коротко, что хочешь сделать в боте. "
                        "Вопрос должен быть не длиннее 600 символов."
                    ),
                )
                active = await self._nova_render_resolution(
                    context,
                    active,
                    self._nova_catalog(live_tier),
                    resolution,
                    source_message=update.effective_message,
                    allow_replacement=allow_replacement,
                )
                return
            catalog = self._nova_catalog(live_tier)
            resolution = resolve_nova_question(
                question,
                catalog,
                last_action_id=active.last_action_id,
            )
            used_ai = resolution is None and self._nova_ai_allowed(live_tier)
            if resolution is None:
                try:
                    resolution = await self._nova_ai_resolution(question, active, catalog)
                except _NovaAccessChanged:
                    await self._nova_discard_for_access(
                        context,
                        active,
                        source_message=update.effective_message,
                        allow_replacement=allow_replacement,
                    )
                    return
            active = await self._nova_render_resolution(
                context,
                active,
                catalog,
                resolution,
                require_ai=used_ai,
                source_message=update.effective_message,
                allow_replacement=allow_replacement,
            )
        finally:
            await self.nova_sessions.finish_question(
                owner_id=active.owner_id,
                telegram_user_id=active.telegram_user_id,
                chat_id=active.chat_id,
                access_version=active.access_version,
                canonical_message_id=active.canonical_message_id,
                tier=active.tier,
                session_id=active.id,
            )

    async def _nova_ai_resolution(
        self,
        question: str,
        session: NovaSession,
        catalog: NovaCatalog,
    ) -> NovaResolution:
        if not self._nova_ai_allowed(session.tier):
            return NovaResolution(
                kind=NovaResolutionKind.CLARIFY,
                response=NOVA_LOCAL_CLARIFY_TEXT,
            )
        pre_tier = await self._nova_validate_access(session, require_ai=True)
        if pre_tier is None:
            raise _NovaAccessChanged
        plan = None
        try:
            plan = await self.ai.nova_help(question, catalog)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Nova provider failed error_type=%s", type(exc).__name__)
        post_tier = await self._nova_validate_access(session, require_ai=True)
        if post_tier is None:
            raise _NovaAccessChanged
        if plan is None:
            return NovaResolution(
                kind=NovaResolutionKind.CLARIFY,
                response=NOVA_PROVIDER_FAILURE_TEXT,
            )
        current_catalog = self._nova_catalog(post_tier)
        try:
            kind = NovaResolutionKind(plan.kind)
        except ValueError:
            kind = NovaResolutionKind.CLARIFY
        action_id = plan.action_id if kind is NovaResolutionKind.GUIDE else None
        capability = self._safe_nova_capability(current_catalog, action_id)
        return NovaResolution(
            kind=kind,
            response=plan.response,
            steps=tuple(plan.steps),
            action_id=capability.id if capability is not None else None,
            cta_label=capability.label if capability is not None else None,
        )

    async def _nova_render_resolution(
        self,
        context: Any,
        session: NovaSession,
        catalog: NovaCatalog,
        resolution: NovaResolution,
        *,
        require_ai: bool = False,
        source_message: Any,
        allow_replacement: bool = True,
    ) -> NovaSession:
        async with self._nova_ui_lock:
            live = await self.nova_sessions.get(
                owner_id=session.owner_id,
                telegram_user_id=session.telegram_user_id,
                chat_id=session.chat_id,
                access_version=session.access_version,
                canonical_message_id=session.canonical_message_id,
                tier=session.tier,
                session_id=session.id,
            )
            if live is None or not live.question_in_progress:
                return session
            final_tier = await self._nova_validate_access(live, require_ai=require_ai)
            if final_tier is None:
                await self.nova_sessions.clear(
                    owner_id=live.owner_id,
                    chat_id=live.chat_id,
                    session_id=live.id,
                )
                await self._nova_edit_canonical(
                    context,
                    live,
                    NOVA_ACCESS_CHANGED_TEXT,
                    None,
                    source_message=source_message,
                    allow_replacement=allow_replacement,
                )
                return session
            catalog = self._nova_catalog(final_tier)
            capability = self._safe_nova_capability(catalog, resolution.action_id)
            action_callback: str | None = None
            cta_label = resolution.cta_label
            if capability is not None and resolution.kind is NovaResolutionKind.GUIDE:
                if capability.kind is NovaCapabilityKind.GUEST:
                    action_callback = capability.target
                else:
                    token = await self.nova_sessions.issue_action(
                        action_id=capability.id,
                        owner_id=live.owner_id,
                        telegram_user_id=live.telegram_user_id,
                        chat_id=live.chat_id,
                        access_version=live.access_version,
                        canonical_message_id=live.canonical_message_id,
                        tier=live.tier,
                        session_id=live.id,
                    )
                    if token is not None:
                        action_callback = f"nova:action:{capability.id}:{token}"
                cta_label = cta_label or capability.label
            still_live = await self.nova_sessions.get(
                owner_id=live.owner_id,
                telegram_user_id=live.telegram_user_id,
                chat_id=live.chat_id,
                access_version=live.access_version,
                canonical_message_id=live.canonical_message_id,
                tier=live.tier,
                session_id=live.id,
            )
            if still_live is None or not still_live.question_in_progress:
                return session
            delivery_tier = await self._nova_validate_access(
                still_live,
                require_ai=require_ai,
            )
            if delivery_tier is None:
                await self.nova_sessions.clear(
                    owner_id=still_live.owner_id,
                    chat_id=still_live.chat_id,
                    session_id=still_live.id,
                )
                await self._nova_edit_canonical(
                    context,
                    still_live,
                    NOVA_ACCESS_CHANGED_TEXT,
                    None,
                    source_message=source_message,
                    allow_replacement=allow_replacement,
                )
                return session
            text = self._nova_resolution_text(resolution)
            markup = self._nova_resolution_keyboard(
                live.tier,
                action_callback=action_callback,
                cta_label=cta_label,
            )
            edited = await self._nova_edit_canonical(
                context,
                live,
                text,
                markup,
                source_message=source_message,
                allow_replacement=allow_replacement,
            )
            if edited is None:
                return live
            remembered = await self.nova_sessions.remember_action(
                last_action_id=(
                    capability.id
                    if capability is not None and resolution.kind is NovaResolutionKind.GUIDE
                    else None
                ),
                owner_id=edited.owner_id,
                telegram_user_id=edited.telegram_user_id,
                chat_id=edited.chat_id,
                access_version=edited.access_version,
                canonical_message_id=edited.canonical_message_id,
                tier=edited.tier,
                session_id=edited.id,
            )
            return remembered or edited

    async def nova_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = query.data if isinstance(query.data, str) else ""
        if data.startswith("nova:action:"):
            await self._nova_generic_action(update, context)
            return
        user = await self._user(update.effective_user.id)
        session = await self._nova_bound_callback_session(update, user)
        if session is None:
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return
        if (
            session.tier != GUEST
            and await self._active_navigation_flow(update, context) is not None
        ):
            await self.nova_sessions.clear(
                owner_id=session.owner_id,
                chat_id=session.chat_id,
                session_id=session.id,
            )
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return
        if session.question_in_progress:
            await query.answer(NOVA_BUSY_ALERT, show_alert=True)
            return
        if data == "nova:root":
            await query.answer()
            published = None
            async with self._nova_ui_lock:
                live = await self.nova_sessions.get(
                    owner_id=session.owner_id,
                    telegram_user_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    access_version=session.access_version,
                    canonical_message_id=session.canonical_message_id,
                    tier=session.tier,
                    session_id=session.id,
                )
                if live is None or live.question_in_progress:
                    return
                if (
                    live.tier != GUEST
                    and await self._active_navigation_flow(update, context) is not None
                ):
                    await self.nova_sessions.clear(
                        owner_id=live.owner_id,
                        chat_id=live.chat_id,
                        session_id=live.id,
                    )
                    return
                message_id = await self._nova_edit_callback(
                    query,
                    NOVA_ROOT_TEXT,
                    self._nova_root_keyboard(live.tier),
                    operation="root",
                )
                if message_id is not None:
                    published = await self.nova_sessions.create(
                        owner_id=live.owner_id,
                        telegram_user_id=live.telegram_user_id,
                        chat_id=live.chat_id,
                        access_version=live.access_version,
                        canonical_message_id=message_id,
                        tier=live.tier,
                    )
            if published is not None:
                await self._nova_memory_clear_if_guided_current(published)
            return
        if data.startswith("nova:topic:"):
            key = data.removeprefix("nova:topic:")
            topic = self._nova_topics().get(key)
            if topic is None:
                await query.answer(NOVA_STALE_ALERT, show_alert=True)
                return
            await query.answer()
            async with self._nova_ui_lock:
                live = await self.nova_sessions.get(
                    owner_id=session.owner_id,
                    telegram_user_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    access_version=session.access_version,
                    canonical_message_id=session.canonical_message_id,
                    tier=session.tier,
                    session_id=session.id,
                )
                if live is None or live.question_in_progress:
                    return
                if (
                    live.tier != GUEST
                    and await self._active_navigation_flow(update, context) is not None
                ):
                    await self.nova_sessions.clear(
                        owner_id=live.owner_id,
                        chat_id=live.chat_id,
                        session_id=live.id,
                    )
                    return
                message_id = await self._nova_edit_callback(
                    query,
                    f"{topic[0]}\n\n{topic[1]}",
                    self._nova_back_keyboard(live.tier),
                    operation="topic",
                )
                if message_id is not None and message_id != live.canonical_message_id:
                    await self.nova_sessions.rebind_canonical(
                        owner_id=live.owner_id,
                        telegram_user_id=live.telegram_user_id,
                        chat_id=live.chat_id,
                        access_version=live.access_version,
                        tier=live.tier,
                        expected_message_id=live.canonical_message_id,
                        new_message_id=message_id,
                        session_id=live.id,
                    )
            return
        await query.answer(NOVA_STALE_ALERT, show_alert=True)

    async def _nova_generic_action(self, update: Any, context: Any) -> None:
        async with self._nova_launch_lock:
            await self._nova_generic_action_locked(update, context)

    async def _nova_generic_action_locked(self, update: Any, context: Any) -> None:
        claimed = await self._nova_claim_action(update, context)
        if claimed is None:
            return
        capability, user = claimed
        if capability.id in _NOVA_CONVERSATION_ACTIONS:
            await update.callback_query.answer(NOVA_STALE_ALERT, show_alert=True)
            return
        await update.callback_query.answer()
        capability = await self._nova_revalidate_action(update, context, user, capability.id)
        if capability is None:
            return
        await self._nova_dispatch_capability(update, context, user, capability)

    async def nova_evening_entry(self, update: Any, context: Any) -> int | None:
        return await self._nova_conversation_entry(
            update,
            context,
            action_id="evening",
            handler_name="evening_start",
            back_target="nav:section:today",
        )

    async def nova_health_entry(self, update: Any, context: Any) -> int | None:
        return await self._nova_conversation_entry(
            update,
            context,
            action_id="checkin",
            handler_name="health_checkin_start",
            back_target="nav:section:health",
        )

    async def nova_doctor_entry(self, update: Any, context: Any) -> int | None:
        return await self._nova_conversation_entry(
            update,
            context,
            action_id="doctor_prepare",
            handler_name="doctor_prepare_start",
            back_target="nav:section:health",
        )

    async def nova_onboarding_entry(self, update: Any, context: Any) -> int | None:
        return await self._nova_conversation_entry(
            update,
            context,
            action_id="onboarding",
            handler_name="start",
            back_target="nav:section:settings",
        )

    async def _nova_conversation_entry(
        self,
        update: Any,
        context: Any,
        *,
        action_id: str,
        handler_name: str,
        back_target: str,
    ) -> int | None:
        async with self._nova_launch_lock:
            return await self._nova_conversation_entry_locked(
                update,
                context,
                action_id=action_id,
                handler_name=handler_name,
                back_target=back_target,
            )

    async def _nova_conversation_entry_locked(
        self,
        update: Any,
        context: Any,
        *,
        action_id: str,
        handler_name: str,
        back_target: str,
    ) -> int | None:
        claimed = await self._nova_claim_action(
            update,
            context,
            expected_action=action_id,
        )
        if claimed is None:
            return None
        capability, user = claimed
        await update.callback_query.answer()
        capability = await self._nova_revalidate_action(update, context, user, action_id)
        if capability is None:
            return None
        if await self.nova_memory_blocks_navigation(update):
            await self.nova_memory_public_command_gate(update, context)
            return None
        await self.nova_memory_clear_current(update)
        screen = _NovaCallbackMessage(
            self,
            update.callback_query,
            self._back_keyboard(back_target),
        )
        original_args = getattr(context, "args", None)
        context.args = []
        try:
            return await getattr(self, handler_name)(_NovaScreenUpdate(update, screen), context)
        finally:
            context.args = original_args or []

    async def _nova_claim_action(
        self,
        update: Any,
        context: Any,
        *,
        expected_action: str | None = None,
    ) -> tuple[NovaCapability, Any] | None:
        query = update.callback_query
        data = query.data if isinstance(query.data, str) else ""
        payload = data.removeprefix("nova:action:") if data.startswith("nova:action:") else ""
        if ":" not in payload:
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        action_id, token = payload.rsplit(":", maxsplit=1)
        if not action_id or not token:
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        if expected_action is not None and action_id != expected_action:
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        user = await self._user(update.effective_user.id)
        tier = await self._nova_validate_access_values(
            update.effective_user.id,
            user.access_version,
            require_access_tier(user.access_tier),
        )
        message_id = self._positive_message_id(getattr(query.message, "message_id", None))
        if tier is None or message_id is None:
            await self.nova_sessions.clear(owner_id=user.id, chat_id=update.effective_chat.id)
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        if tier != GUEST and await self._active_navigation_flow(update, context) is not None:
            await self.nova_sessions.clear(owner_id=user.id, chat_id=update.effective_chat.id)
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        catalog = self._nova_catalog(tier)
        capability = self._safe_nova_capability(catalog, action_id)
        if capability is None:
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        claim = await self.nova_sessions.consume_action(
            token=token,
            expected_action_id=action_id,
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            access_version=user.access_version,
            canonical_message_id=message_id,
            tier=tier,
        )
        if claim is None or claim.action_id != action_id:
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        await self.nova_sessions.clear(
            owner_id=user.id,
            chat_id=update.effective_chat.id,
            session_id=claim.session.id,
        )
        final_tier = await self._nova_validate_access_values(
            update.effective_user.id,
            user.access_version,
            tier,
        )
        final_capability = (
            self._safe_nova_capability(self._nova_catalog(final_tier), action_id)
            if final_tier is not None
            else None
        )
        if final_capability is None:
            await query.answer(NOVA_STALE_ALERT, show_alert=True)
            return None
        return final_capability, user

    async def _nova_revalidate_action(
        self,
        update: Any,
        context: Any,
        user: Any,
        action_id: str,
    ) -> NovaCapability | None:
        expected_tier = require_access_tier(user.access_tier)
        tier = await self._nova_validate_access_values(
            update.effective_user.id,
            user.access_version,
            expected_tier,
        )
        if tier is None:
            return None
        if tier != GUEST and await self._active_navigation_flow(update, context) is not None:
            return None
        tier = await self._nova_validate_access_values(
            update.effective_user.id,
            user.access_version,
            expected_tier,
        )
        if tier is None:
            return None
        return self._safe_nova_capability(self._nova_catalog(tier), action_id)

    async def _nova_dispatch_capability(
        self,
        update: Any,
        context: Any,
        user: Any,
        capability: NovaCapability,
    ) -> None:
        target = capability.target
        query = update.callback_query
        await self.nova_memory_clear_current(update)
        weekly_review_available = self._nova_flags().weekly_review_available_for_tier(
            require_access_tier(user.access_tier)
        )
        if target == "nav:root":
            await self._edit_or_send(
                query,
                "Главное меню\n\nЧто хочешь сделать?",
                self._root_keyboard(user.access_tier),
            )
            return
        if target.startswith("nav:section:"):
            section_key = target.removeprefix("nav:section:")
            if section_key == "vision":
                await self._vision_menu(query.message, user=user, query=query)
                return
            section = navigation_sections(
                self._workspace_enabled(),
                self._knowledge_hub_enabled(),
                self._knowledge_capture_enabled(),
                weekly_review_available,
            ).get(section_key)
            if section is None:
                return
            await self._edit_or_send(
                query,
                f"{section.emoji} {section.label}\n\n{section.description}",
                self._section_keyboard(section_key, user.access_tier),
            )
            return
        if target.startswith("nav:help:"):
            topic_key = target.removeprefix("nav:help:")
            topic = self._nova_topics().get(topic_key)
            if topic is not None:
                await self._edit_or_send(
                    query,
                    f"{topic[0]}\n\n{topic[1]}",
                    self._back_keyboard("nav:help"),
                )
            return
        if not target.startswith("nav:action:"):
            return
        action_id = target.removeprefix("nav:action:")
        action = navigation_actions(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
            weekly_review_available,
        ).get(action_id)
        if action is None or action.handler is None:
            return
        if action_id == "weekly_review":
            if await self.reminder_blocks_navigation(update):
                await self.reminder_public_command_gate(update, context)
                return
            async with self._reminder_launch_lock:
                reminder_owns = await self.reminder_blocks_navigation(update)
                if not reminder_owns:
                    screen = _NovaCallbackMessage(
                        self,
                        query,
                        self._back_keyboard(self._section_for_action(action_id)),
                    )
                    original_args = getattr(context, "args", None)
                    context.args = []
                    try:
                        await getattr(self, action.handler)(
                            _NovaScreenUpdate(update, screen), context
                        )
                    finally:
                        context.args = original_args or []
            if reminder_owns:
                await self.reminder_public_command_gate(update, context)
            return
        if action_id == "vision":
            await self._vision_menu(query.message, user=user, query=query)
            return
        screen = _NovaCallbackMessage(
            self,
            query,
            self._back_keyboard(self._section_for_action(action_id)),
        )
        original_args = getattr(context, "args", None)
        context.args = []
        try:
            await getattr(self, action.handler)(_NovaScreenUpdate(update, screen), context)
        finally:
            context.args = original_args or []

    async def _nova_bound_callback_session(self, update: Any, user: Any) -> NovaSession | None:
        tier = await self._nova_validate_access_values(
            update.effective_user.id,
            user.access_version,
            require_access_tier(user.access_tier),
        )
        message_id = self._positive_message_id(
            getattr(update.callback_query.message, "message_id", None)
        )
        if tier is None or message_id is None:
            await self.nova_sessions.clear(owner_id=user.id, chat_id=update.effective_chat.id)
            return None
        return await self.nova_sessions.get(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            access_version=user.access_version,
            canonical_message_id=message_id,
            tier=tier,
        )

    async def _nova_validate_access(
        self,
        session: NovaSession,
        *,
        require_ai: bool = False,
    ) -> AccessTier | None:
        tier = await self._nova_validate_access_values(
            session.telegram_user_id,
            session.access_version,
            session.tier,
        )
        if tier is None or (require_ai and not self._nova_ai_allowed(tier)):
            return None
        return tier

    async def _nova_validate_access_values(
        self,
        telegram_user_id: int,
        access_version: int,
        expected_tier: AccessTier,
    ) -> AccessTier | None:
        try:
            status = await self.access_service.status(telegram_user_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Nova access lookup failed error_type=%s", type(exc).__name__)
            return None
        if (
            status is None
            or status.access_version != access_version
            or status.access_tier != expected_tier
        ):
            return None
        return require_access_tier(status.access_tier)

    def _nova_ai_allowed(self, tier: AccessTier) -> bool:
        if not bool(getattr(self.settings, "enable_nova_ai", False)):
            return False
        if tier in {GUEST, BLOCKED}:
            return False
        return not bool(getattr(self.settings, "nova_ai_admin_only", True)) or tier == ADMIN

    def _safe_nova_capability(
        self,
        catalog: NovaCatalog,
        action_id: str | None,
    ) -> NovaCapability | None:
        if action_id is None:
            return None
        capability = catalog.capability(action_id)
        if capability is None:
            return None
        if capability.kind is not NovaCapabilityKind.ACTION:
            return capability
        action = navigation_actions(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
            self._nova_flags().weekly_review_available_for_tier(catalog.tier),
        ).get(capability.id)
        if action is None or action.handler is None:
            return None
        lowered = action.handler.casefold()
        if any(marker in lowered for marker in ("delete", "remove", "admin", "access", "block")):
            return None
        return capability

    async def nova_cancel_gate(self, update: Any, context: Any) -> None:
        flow = await self._active_navigation_flow(update, context)
        if flow is None and await self.weekly_review_cancel_gate(update, context):
            from telegram.ext import ApplicationHandlerStop

            raise ApplicationHandlerStop
        if flow is None and await self.nova_memory_cancel_gate(update, context):
            from telegram.ext import ApplicationHandlerStop

            raise ApplicationHandlerStop
        if flow is None and await self.reminder_cancel_gate(update, context):
            from telegram.ext import ApplicationHandlerStop

            raise ApplicationHandlerStop
        user = await self._user(update.effective_user.id)
        current = await self.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        if current is None:
            return
        if require_access_tier(user.access_tier) != GUEST:
            if flow is not None:
                return
        async with self._nova_ui_lock:
            live = await self.nova_sessions.get(
                owner_id=current.owner_id,
                telegram_user_id=current.telegram_user_id,
                chat_id=current.chat_id,
                access_version=current.access_version,
                canonical_message_id=current.canonical_message_id,
                tier=current.tier,
                session_id=current.id,
            )
            if live is not None:
                await self.nova_sessions.clear(
                    owner_id=user.id,
                    chat_id=update.effective_chat.id,
                    session_id=live.id,
                )
                await self._nova_edit_canonical(
                    context,
                    live,
                    "✨ Nova\n\nСессия завершена.",
                    None,
                    source_message=update.effective_message,
                )
        from telegram.ext import ApplicationHandlerStop

        raise ApplicationHandlerStop

    async def nova_clear_current(self, update: Any) -> None:
        user = await self._user(update.effective_user.id)
        await self.nova_clear_bound(user.id, update.effective_chat.id)

    async def nova_clear_bound(
        self,
        owner_id: int,
        chat_id: int,
        *,
        session_id: str | None = None,
    ) -> None:
        async with self._nova_ui_lock:
            await self.nova_sessions.clear(
                owner_id=owner_id,
                chat_id=chat_id,
                session_id=session_id,
            )

    async def nova_sync_access(
        self,
        user: Any,
        chat_id: int,
        *,
        context: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        tier = require_access_tier(user.access_tier)
        async with self._nova_ui_lock:
            current = await self.nova_sessions.current(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
            )
            if current is None:
                return
            if current.access_version != user.access_version or current.tier != tier:
                await self.nova_sessions.clear(
                    owner_id=user.id,
                    chat_id=chat_id,
                    session_id=current.id,
                )
                if context is not None:
                    await self._nova_edit_canonical(
                        context,
                        current,
                        NOVA_ACCESS_CHANGED_TEXT,
                        None,
                        source_message=source_message,
                    )

    async def _nova_flow_help(
        self,
        message: Any,
        update: Any,
        flow: str,
        *,
        query: Any | None = None,
    ) -> None:
        user = await self._user(update.effective_user.id)
        await self.nova_memory_clear_current(update)
        async with self._nova_ui_lock:
            current = await self.nova_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if current is not None:
                await self.nova_sessions.clear(
                    owner_id=user.id,
                    chat_id=update.effective_chat.id,
                    session_id=current.id,
                )
            token = await self.navigation_flow_sessions.issue(
                update.effective_user.id,
                update.effective_chat.id,
                flow,
            )
            label = getattr(self, "_nova_flow_label", lambda value: value)(flow)
            text = f"✨ Nova\n\nСейчас не завершён сценарий: {label}. Что сделать?"
            markup = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "▶️ Продолжить текущий шаг",
                            callback_data=f"nav:flow:continue:{token}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "🏠 Выйти в главное меню",
                            callback_data=f"nav:flow:exit:{token}",
                        )
                    ],
                ]
            )
            if query is not None:
                await self._edit_or_send(query, text, markup)
            else:
                await message.reply_text(text, reply_markup=markup)

    def _nova_root_keyboard(self, tier: AccessTier) -> InlineKeyboardMarkup:
        rows = [
            [InlineKeyboardButton(label, callback_data=f"nova:topic:{key}")]
            for key, label in _NOVA_ROOT_TOPICS
        ]
        rows.append(
            [
                InlineKeyboardButton(
                    "🏠 Главное меню" if tier != GUEST else "🏠 В начало",
                    callback_data="nav:root" if tier != GUEST else "guest:root",
                )
            ]
        )
        return InlineKeyboardMarkup(rows)

    def _nova_back_keyboard(self, tier: AccessTier) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("← К Nova", callback_data="nova:root")],
                [
                    InlineKeyboardButton(
                        "🏠 Главное меню" if tier != GUEST else "🏠 В начало",
                        callback_data="nav:root" if tier != GUEST else "guest:root",
                    )
                ],
            ]
        )

    def _nova_resolution_keyboard(
        self,
        tier: AccessTier,
        *,
        action_callback: str | None,
        cta_label: str | None,
    ) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        if action_callback is not None and cta_label is not None:
            label = f"✅ {cta_label}" if cta_label == "Создать задачу" else cta_label
            rows.append([InlineKeyboardButton(label, callback_data=action_callback)])
        rows.append(
            [
                InlineKeyboardButton("← К Nova", callback_data="nova:root"),
                InlineKeyboardButton(
                    "🏠 Главное меню" if tier != GUEST else "🏠 В начало",
                    callback_data="nav:root" if tier != GUEST else "guest:root",
                ),
            ]
        )
        return InlineKeyboardMarkup(rows)

    def _nova_topics(self) -> dict[str, tuple[str, str]]:
        return help_topics(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
            self._voice_enabled(),
            self._task_reminders_enabled(),
        )

    @staticmethod
    def _nova_resolution_text(resolution: NovaResolution) -> str:
        lines = ["✨ Nova", "", resolution.response]
        if resolution.steps:
            lines.append("")
            lines.extend(f"{index}. {step}" for index, step in enumerate(resolution.steps, 1))
        return "\n".join(lines)

    async def _nova_edit_callback(
        self,
        query: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        operation: str,
    ) -> int | None:
        message_id = self._positive_message_id(getattr(query.message, "message_id", None))
        replacement_allowed = message_id is None
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
            return message_id
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            value = str(exc).casefold()
            if "message is not modified" in value:
                return message_id
            if any(marker in value for marker in _MEDIA_TEXT_EDIT_ERRORS):
                if len(text) <= 1024:
                    edit_caption = getattr(query, "edit_message_caption", None)
                    if callable(edit_caption):
                        try:
                            await edit_caption(caption=text, reply_markup=reply_markup)
                            return message_id
                        except asyncio.CancelledError:
                            raise
                        except (TelegramError, TypeError, AttributeError) as caption_exc:
                            logger.warning(
                                "Nova callback edit failed operation=%s_caption error_type=%s",
                                operation,
                                type(caption_exc).__name__,
                            )
                            return None
                replacement_allowed = True
            elif any(marker in value for marker in _MISSING_MESSAGE_ERRORS):
                replacement_allowed = True
            else:
                logger.warning(
                    "Nova callback edit failed operation=%s_text error_type=%s",
                    operation,
                    type(exc).__name__,
                )
                return None
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Nova callback edit failed operation=%s_text error_type=%s",
                operation,
                type(exc).__name__,
            )
            return None
        if not replacement_allowed:
            return None
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Nova callback controls retirement failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
        try:
            replacement = await query.message.reply_text(text, reply_markup=reply_markup)
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Nova callback replacement failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return None
        return self._positive_message_id(getattr(replacement, "message_id", None))

    async def _nova_edit_canonical(
        self,
        context: Any,
        session: NovaSession,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        source_message: Any,
        allow_replacement: bool = True,
    ) -> NovaSession | None:
        bot = context.bot
        try:
            await bot.edit_message_text(
                chat_id=session.chat_id,
                message_id=session.canonical_message_id,
                text=text,
                reply_markup=reply_markup,
            )
            return session
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            value = str(exc).casefold()
            if "message is not modified" in value:
                return session
            if any(marker in value for marker in _MEDIA_TEXT_EDIT_ERRORS):
                if len(text) <= 1024:
                    try:
                        await bot.edit_message_caption(
                            chat_id=session.chat_id,
                            message_id=session.canonical_message_id,
                            caption=text,
                            reply_markup=reply_markup,
                        )
                        return session
                    except asyncio.CancelledError:
                        raise
                    except (TelegramError, TypeError, AttributeError) as caption_exc:
                        logger.warning(
                            "Nova canonical edit failed operation=caption error_type=%s",
                            type(caption_exc).__name__,
                        )
                        return None
            elif not any(marker in value for marker in _MISSING_MESSAGE_ERRORS):
                logger.warning(
                    "Nova canonical edit failed operation=text error_type=%s",
                    type(exc).__name__,
                )
                return None
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Nova canonical edit failed operation=text error_type=%s",
                type(exc).__name__,
            )
            return None

        if not allow_replacement:
            return None
        try:
            await bot.edit_message_reply_markup(
                chat_id=session.chat_id,
                message_id=session.canonical_message_id,
                reply_markup=None,
            )
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Nova canonical controls retirement failed error_type=%s",
                type(exc).__name__,
            )
        try:
            send_message = getattr(bot, "send_message", None)
            replacement = (
                await send_message(
                    chat_id=session.chat_id,
                    text=text,
                    reply_markup=reply_markup,
                )
                if callable(send_message)
                else await source_message.reply_text(text, reply_markup=reply_markup)
            )
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Nova canonical replacement failed error_type=%s",
                type(exc).__name__,
            )
            return None
        new_message_id = self._positive_message_id(getattr(replacement, "message_id", None))
        if new_message_id is None:
            logger.warning("Nova canonical replacement failed error_type=MissingMessageId")
            return None
        return await self.nova_sessions.rebind_canonical(
            owner_id=session.owner_id,
            telegram_user_id=session.telegram_user_id,
            chat_id=session.chat_id,
            access_version=session.access_version,
            tier=session.tier,
            expected_message_id=session.canonical_message_id,
            new_message_id=new_message_id,
            session_id=session.id,
        )

    async def _nova_discard_for_access(
        self,
        context: Any,
        session: NovaSession,
        *,
        source_message: Any,
        allow_replacement: bool = True,
    ) -> None:
        async with self._nova_ui_lock:
            live = await self.nova_sessions.get(
                owner_id=session.owner_id,
                telegram_user_id=session.telegram_user_id,
                chat_id=session.chat_id,
                access_version=session.access_version,
                canonical_message_id=session.canonical_message_id,
                tier=session.tier,
                session_id=session.id,
            )
            if live is None:
                return
            await self.nova_sessions.clear(
                owner_id=live.owner_id,
                chat_id=live.chat_id,
                session_id=live.id,
            )
            await self._nova_edit_canonical(
                context,
                live,
                NOVA_ACCESS_CHANGED_TEXT,
                None,
                source_message=source_message,
                allow_replacement=allow_replacement,
            )

    async def _nova_edit_transient_access_changed(self, message: Any) -> None:
        try:
            await message.edit_text(NOVA_ACCESS_CHANGED_TEXT, reply_markup=None)
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            if "message is not modified" in str(exc).casefold():
                return
            logger.warning(
                "Nova voice canonical edit failed operation=access error_type=%s",
                type(exc).__name__,
            )

    async def _nova_delete_transient(self, message: Any, context: Any) -> None:
        message_id = self._positive_message_id(getattr(message, "message_id", None))
        try:
            delete = getattr(message, "delete", None)
            if callable(delete):
                await delete()
            elif message_id is not None:
                await context.bot.delete_message(
                    chat_id=getattr(message, "chat_id", None),
                    message_id=message_id,
                )
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Nova voice transient cleanup failed error_type=%s",
                type(exc).__name__,
            )

    @staticmethod
    def _positive_message_id(value: Any) -> int | None:
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None
        )
