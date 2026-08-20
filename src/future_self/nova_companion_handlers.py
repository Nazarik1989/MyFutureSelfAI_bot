from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

from .access import FULL_ACCESS_TIERS
from .conversation import ConversationExchangeReceipt
from .domain import temporal_context
from .models import User
from .nova_companion import (
    NovaCompanionContextFence,
    NovaCompanionContextService,
)
from .nova_companion_flow import (
    CAPTURE_ACTIONS,
    CAPTURE_DATE_ACTIONS,
    CaptureSuggestion,
    ExplicitCaptureClassifier,
    NovaAddressClassifier,
    NovaAddressKind,
    NovaCompanionCaptureCapability,
    NovaCompanionCaptureScreen,
    NovaCompanionCaptureStore,
    NovaCompanionCaptureTemporal,
    NovaCompanionPolicy,
    validate_capture_suggestion,
)
from .nova_memory_application import (
    NovaMemoryProjection,
    NovaMemoryProjectionError,
    build_nova_memory_projection,
)
from .reminder_intent import ReminderIntentStatus
from .schemas import ParsedThought

logger = logging.getLogger(__name__)

NOVA_COMPANION_ACCESS_CHANGED_TEXT = (
    "Доступ изменился. Я не использовала прежний контекст — напиши сообщение ещё раз."
)
NOVA_COMPANION_CONTEXT_CHANGED_TEXT = (
    "Контекст успел измениться. Ничего не сохранено — напиши сообщение ещё раз."
)
NOVA_COMPANION_UNAVAILABLE_TEXT = (
    "Сейчас не получилось ответить. Ничего не сохранено — попробуй ещё раз."
)
NOVA_COMPANION_CAPTURE_FAILED_TEXT = (
    "Не удалось открыть preview. Ничего не сохранено — попробуй ещё раз."
)

_COMPANION_DRAIN_TIMEOUT_SECONDS = 30.0
_COMPANION_CANCEL_TIMEOUT_SECONDS = 5.0
_COMPANION_CANCEL_RETRY_SECONDS = 0.1
_COMPANION_CLEANUP_SCHEDULED_ATTR = "nova_companion_cleanup_scheduled"

type _CompanionCheck = Literal["ready", "access_changed", "context_changed", "unavailable"]


class _NovaCompanionDrainError(RuntimeError):
    pass


class _NovaCompanionCaptureCompensationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _CompanionGeneration:
    owner_id: int = field(repr=False)
    telegram_actor_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    tier: str = field(repr=False)
    access_version: int = field(repr=False)
    context_fence: NovaCompanionContextFence = field(repr=False)
    memory_revision: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _PreparedCompanionAnswer:
    answer: str = field(repr=False)
    generation: _CompanionGeneration = field(repr=False)
    suggestion: CaptureSuggestion | None = field(default=None, repr=False)
    user_text: str = field(default="", repr=False)
    source: str = "text"
    temporal: NovaCompanionCaptureTemporal | None = field(default=None, repr=False)
    persist_exchange: bool = True


class NovaCompanionHandlers:
    """Conversation-first Nova routing and optional capture delivery."""

    def _init_nova_companion(self) -> None:
        self.nova_companion_context = NovaCompanionContextService(
            self.db,
            conversation_message_limit=self.settings.conversation_context_messages,
        )
        self.nova_companion_captures = NovaCompanionCaptureStore()
        self._nova_companion_tasks: set[asyncio.Task[bool]] = set()

    def nova_companion_policy(self) -> NovaCompanionPolicy:
        return NovaCompanionPolicy(
            enabled=bool(getattr(self.settings, "enable_nova_companion", False)),
            admin_only=bool(getattr(self.settings, "nova_companion_admin_only", True)),
        )

    def nova_companion_available_for_actor(self, actor: Any | None) -> bool:
        return self.nova_companion_policy().allows_actor(actor)

    async def nova_companion_route(
        self,
        update: Any,
        context: Any,
        text: str,
        source: str,
        *,
        user: User,
        conversation_snapshot: Any,
        delivery_message: Any | None = None,
        resolved_date: Any | None = None,
        temporal_resolution: Any | None = None,
    ) -> bool:
        """Handle an eligible ordinary message without invoking the legacy router."""

        policy = self.nova_companion_policy()
        if not policy.allows_actor(user):
            return False
        address = NovaAddressClassifier.classify(text)
        if address.is_local_response:
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                address.local_response or "Да, я здесь 🙂",
                user=user,
            )
            return True
        semantic_text = (
            address.content
            if address.kind is NovaAddressKind.VOCATIVE and address.content is not None
            else text.strip()
        )
        explicit = ExplicitCaptureClassifier.classify(semantic_text)
        if explicit is not None:
            if explicit.kind == "task" and resolved_date is None and temporal_resolution is None:
                date_resolution = self.date_resolver.resolve(semantic_text, user.timezone)
                if date_resolution.status == "conflict":
                    return False
                if date_resolution.status == "resolved" and date_resolution.target_date:
                    resolved_date = date_resolution.target_date
                    temporal_resolution = self.date_resolver.temporal_resolution(
                        date_resolution.target_date,
                        user.timezone,
                        semantic_text,
                        self.date_resolver.extract_local_time(semantic_text),
                    )
            await self._nova_companion_explicit_capture(
                update,
                context,
                delivery_message,
                user=user,
                snapshot=conversation_snapshot,
                source=source,
                original_text=text,
                kind=explicit.kind,
                content=explicit.content,
                references_context=explicit.references_context,
                resolved_date=resolved_date,
                temporal_resolution=temporal_resolution,
            )
            return True
        reminder_probe = self.reminder_intent_parser.parse(semantic_text, user.timezone)
        if reminder_probe.status is not ReminderIntentStatus.NOT_REMINDER:
            return await self._reminder_question_gate(
                update,
                context,
                semantic_text,
                candidate_message=delivery_message,
                expected_access_version=(
                    user.access_version if delivery_message is not None else None
                ),
                expected_session=None,
                voice_fenced=delivery_message is not None,
                voice_state=None,
                weekly_candidate_handoff=False,
            )
        await self._nova_companion_prepare_and_deliver(
            update,
            context,
            delivery_message,
            user=user,
            snapshot=conversation_snapshot,
            text=semantic_text,
            source=source,
        )
        return True

    async def _nova_companion_local_response(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        response: str,
        *,
        user: User,
    ) -> None:
        message = delivery_message or update.effective_message
        if delivery_message is not None:
            coroutine = self._nova_companion_local_voice_lifecycle(
                context,
                delivery_message,
                response,
                user=user,
                chat_id=update.effective_chat.id,
            )
            try:
                task = asyncio.create_task(
                    coroutine,
                    name="nova-companion-local-voice-lifecycle",
                )
            except BaseException:
                coroutine.close()
                raise
            self._track_nova_companion_task(task)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.done():
                    task.cancel()
                raise
            return
        if not await self._nova_companion_actor_is_current(user):
            return
        try:
            await message.reply_text(response)
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            logger.warning(
                "Nova companion failed operation=local_delivery error_type=%s",
                type(exc).__name__,
            )

    async def _nova_companion_local_voice_lifecycle(
        self,
        context: Any,
        delivery_message: Any,
        response: str,
        *,
        user: User,
        chat_id: int,
    ) -> bool:
        try:
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_retire_pre_delivery(
                    context,
                    delivery_message,
                    chat_id=chat_id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            await delivery_message.edit_text(response, reply_markup=None)
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_retire_pre_delivery(
                    context,
                    delivery_message,
                    chat_id=chat_id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            return True
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=local_delivery error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            return False

    async def _nova_companion_explicit_capture(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        snapshot: Any,
        source: str,
        original_text: str,
        kind: str,
        content: str | None,
        references_context: bool,
        resolved_date: Any | None,
        temporal_resolution: Any | None,
    ) -> None:
        capture_text = content
        if references_context:
            capture_text = self.conversation.companion_reference_candidate(snapshot)
        if not capture_text:
            notice = "Не уверена, что именно сохранить. Уточни мысль одним сообщением."
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                notice,
                user=user,
            )
            return
        title = capture_text.strip()[:200]
        coroutine = self._nova_companion_explicit_capture_lifecycle(
            update,
            context,
            delivery_message,
            user=user,
            source=source,
            original_text=original_text,
            capture_text=capture_text,
            parsed=ParsedThought(
                kind=kind,
                title=title,
                resolved_date=resolved_date,
                temporal_resolution=temporal_resolution,
            ),
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-explicit-capture-lifecycle",
            )
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise

    async def _nova_companion_explicit_capture_lifecycle(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        source: str,
        original_text: str,
        capture_text: str,
        parsed: ParsedThought,
    ) -> bool:
        creation = None
        preview = None
        accepted_previews: list[Any] = []
        previous_preview_message_id: int | None = None
        preview_pointer_owned = False
        focus_lease = None
        focus_owned = False
        try:
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_retire_explicit_progress(
                    context,
                    delivery_message,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            creation = await self.draft_service.create_or_get_for_suggestion(
                user_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                expected_access_version=user.access_version,
                source=source,
                raw_text=capture_text,
                parsed=parsed,
            )
            if not creation.ok or creation.draft is None:
                await self._nova_companion_retire_explicit_progress(
                    context,
                    delivery_message,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            draft = creation.draft
            previous_preview_message_id = self._positive_companion_message_id(
                draft.preview_message_id
            )
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=None,
                    previous_preview_message_id=previous_preview_message_id,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            message = delivery_message or update.effective_message
            if delivery_message is not None:
                await delivery_message.edit_text("Голос распознан. Проверь preview ниже.")
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=None,
                    previous_preview_message_id=previous_preview_message_id,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            preview = await self._send_draft_preview(
                message,
                draft,
                include_original=source != "voice",
                on_sent=accepted_previews.append,
                bind_preview=False,
            )
            preview_message_id = self._positive_companion_message_id(
                getattr(preview, "message_id", None)
            )
            if (
                preview_message_id is None
                or not await self.draft_service.restore_preview_message_if_current(
                    draft.id,
                    draft.version,
                    user.telegram_id,
                    update.effective_chat.id,
                    expected_message_id=previous_preview_message_id,
                    restored_message_id=preview_message_id,
                )
            ):
                raise RuntimeError("Preview binding changed")
            preview_pointer_owned = True
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            await self.conversation.append(
                user.telegram_id,
                update.effective_chat.id,
                role="user",
                content=original_text.strip(),
                source=source,
                intent="explicit_capture",
                topic=parsed.title,
            )
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            await self._remember_preview(user.telegram_id, update.effective_chat.id, draft)
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            focus_lease = await self.conversation.acquire_active_draft_focus(
                user.telegram_id,
                update.effective_chat.id,
                draft.id,
            )
            if focus_lease is None:
                raise RuntimeError("Draft focus changed")
            focus_owned = True
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    restore_prior_focus=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            return True
        except asyncio.CancelledError as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            self._nova_companion_schedule_explicit_capture_cleanup(
                context,
                delivery_message,
                creation,
                preview=preview,
                previous_preview_message_id=previous_preview_message_id,
                preview_pointer_owned=preview_pointer_owned,
                focus_lease=focus_lease,
                focus_owned=focus_owned,
                focus_user=user,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
            exc.__dict__[_COMPANION_CLEANUP_SCHEDULED_ATTR] = True
            raise
        except Exception as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            logger.warning(
                "Nova companion failed operation=explicit_capture error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_explicit_capture_cleanup(
                context,
                delivery_message,
                creation,
                preview=preview,
                previous_preview_message_id=previous_preview_message_id,
                preview_pointer_owned=preview_pointer_owned,
                focus_lease=focus_lease,
                focus_owned=focus_owned,
                focus_user=user,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
            return False

    def _nova_companion_schedule_explicit_capture_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        creation: Any,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool = False,
        focus_lease: Any | None = None,
        focus_owned: bool = False,
        focus_user: User,
        restore_prior_focus: bool | None = None,
        telegram_user_id: int,
        chat_id: int,
        neutral_text: str,
    ) -> None:
        coroutine = self._nova_companion_explicit_capture_cleanup(
            context,
            delivery_message,
            creation,
            preview=preview,
            previous_preview_message_id=previous_preview_message_id,
            preview_pointer_owned=preview_pointer_owned,
            focus_lease=focus_lease,
            focus_owned=focus_owned,
            focus_user=focus_user,
            restore_prior_focus=restore_prior_focus,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-explicit-capture-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_explicit_cleanup error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_explicit_capture_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        creation: Any,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool = False,
        focus_lease: Any | None = None,
        focus_owned: bool = False,
        focus_user: User | None = None,
        restore_prior_focus: bool | None = None,
        telegram_user_id: int,
        chat_id: int,
        neutral_text: str,
    ) -> bool:
        storage_clean = True
        draft = getattr(creation, "draft", None)
        preview_message_id = self._positive_companion_message_id(
            getattr(preview, "message_id", None)
        )
        if creation is not None and bool(getattr(creation, "created", False)):
            storage_clean = await self._nova_companion_drop_created_draft(
                creation,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                expected_preview_message_id=(
                    preview_message_id if preview_pointer_owned else previous_preview_message_id
                ),
            )
        elif draft is not None and preview_message_id is not None and preview_pointer_owned:
            restored = await self._nova_companion_restore_reused_preview_pointer(
                draft,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                expected_message_id=preview_message_id,
                restored_message_id=previous_preview_message_id,
            )
            storage_clean = storage_clean and restored
        if focus_lease is not None:
            storage_clean = (
                await self._nova_companion_restore_focus_lease(
                    focus_lease,
                    user=focus_user,
                    capability=None,
                    restore_prior=restore_prior_focus,
                )
                and storage_clean
            )
        elif focus_owned and draft is not None:
            try:
                cleared = await self.conversation.clear_active_draft_if_current(
                    telegram_user_id,
                    chat_id,
                    draft.id,
                    draft.version,
                )
                storage_clean = storage_clean and cleared
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                storage_clean = False
                logger.warning(
                    "Nova companion failed operation=focus_compensation error_type=%s",
                    type(exc).__name__,
                )
        if preview is not None and preview_message_id is not None:
            await self._nova_companion_compensate(
                context,
                preview,
                chat_id=chat_id,
                message_id=preview_message_id,
                neutral_text=neutral_text,
            )
        await self._nova_companion_retire_explicit_progress(
            context,
            delivery_message,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )
        if not storage_clean:
            logger.warning(
                "Nova companion failed operation=explicit_compensation error_type=StateChanged"
            )
        return storage_clean

    async def _nova_companion_restore_focus_lease(
        self,
        focus_lease: Any,
        *,
        user: User | None,
        capability: NovaCompanionCaptureCapability | None,
        restore_prior: bool | None,
    ) -> bool:
        policy = restore_prior
        if policy is None:
            try:
                if user is None:
                    policy = False
                elif capability is None:
                    policy = await self._nova_companion_actor_is_current(user)
                else:
                    policy = await self._nova_companion_capability_actor_is_current(
                        user,
                        capability,
                    )
            except asyncio.CancelledError:
                policy = False
            except Exception as exc:
                policy = False
                logger.warning(
                    "Nova companion failed operation=focus_policy error_type=%s",
                    type(exc).__name__,
                )
        try:
            await self.conversation.restore_active_draft_focus_if_current(
                focus_lease,
                restore_prior=bool(policy),
            )
            # A CAS miss means another exact focus generation won and must be
            # preserved; it is a successful compensation outcome.
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=focus_compensation error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_restore_reused_preview_pointer(
        self,
        draft: Any,
        *,
        telegram_user_id: int,
        chat_id: int,
        expected_message_id: int,
        restored_message_id: int | None,
    ) -> bool:
        try:
            return await self.draft_service.restore_preview_message_if_current(
                draft.id,
                draft.version,
                telegram_user_id,
                chat_id,
                expected_message_id=expected_message_id,
                restored_message_id=restored_message_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=preview_pointer_compensation error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_retire_explicit_progress(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> bool:
        message_id = self._positive_companion_message_id(
            getattr(delivery_message, "message_id", None)
        )
        if delivery_message is None or message_id is None:
            return True
        await self._nova_companion_compensate(
            context,
            delivery_message,
            chat_id=chat_id,
            message_id=message_id,
            neutral_text=neutral_text,
        )
        return True

    async def _nova_companion_prepare_and_deliver(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        snapshot: Any,
        text: str,
        source: str,
    ) -> None:
        try:
            (
                memory_status,
                memory_projection,
                memory_revision,
            ) = await self._nova_companion_memory_projection(user)
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        if memory_status != "ready":
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=self._nova_companion_neutral_text(memory_status),
            )
            return
        try:
            materialized = await self.nova_companion_context.snapshot(
                telegram_actor_id=user.telegram_id,
                expected_tier=user.access_tier,
                expected_access_version=user.access_version,
                conversation_context=snapshot.for_companion_prompt(),
                conversation_chat_id=update.effective_chat.id,
                confirmed_memory=memory_projection,
            )
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=context_snapshot error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            return
        if materialized.status != "ready" or materialized.fence is None:
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=self._nova_companion_neutral_text(materialized.status),
            )
            return
        assert materialized.projection is not None
        generation = _CompanionGeneration(
            owner_id=user.id,
            telegram_actor_id=user.telegram_id,
            chat_id=update.effective_chat.id,
            tier=user.access_tier,
            access_version=user.access_version,
            context_fence=materialized.fence,
            memory_revision=memory_revision,
        )
        try:
            pre_provider_check = await self._nova_companion_current_check(generation)
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        if pre_provider_check != "ready":
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=self._nova_companion_neutral_text(pre_provider_check),
            )
            return
        capture_temporal: NovaCompanionCaptureTemporal | None = None
        temporal_failed = False
        try:
            date_resolution = self.date_resolver.resolve(text, user.timezone)
            if date_resolution.status in {"resolved", "conflict"}:
                capture_temporal = NovaCompanionCaptureTemporal(
                    timezone=user.timezone,
                    resolution=date_resolution,
                    local_time=self.date_resolver.extract_local_time(text),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            temporal_failed = True
            logger.warning(
                "Nova companion failed operation=temporal_resolution error_type=%s",
                type(exc).__name__,
            )
        provider_failed = False
        try:
            result = await self.ai.companion_message(
                text,
                temporal_context(user.timezone),
                materialized.projection,
            )
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        except Exception as exc:
            provider_failed = True
            logger.warning(
                "Nova companion failed operation=provider error_type=%s",
                type(exc).__name__,
            )
            result = None
        try:
            post_provider_check = await self._nova_companion_current_check(generation)
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        if post_provider_check != "ready":
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=self._nova_companion_neutral_text(post_provider_check),
            )
            return
        if provider_failed or result is None:
            prepared = _PreparedCompanionAnswer(
                answer=NOVA_COMPANION_UNAVAILABLE_TEXT,
                user_text=text,
                source=source,
                generation=generation,
                persist_exchange=False,
            )
        else:
            suggestion = None
            if result.capture is not None:
                suggestion = validate_capture_suggestion(
                    kind=result.capture.kind,
                    title=result.capture.title,
                    next_step=result.capture.next_step,
                    user_text=text,
                )
                if suggestion is not None and suggestion.kind == "task" and temporal_failed:
                    suggestion = None
            prepared = _PreparedCompanionAnswer(
                answer=result.answer,
                suggestion=suggestion,
                user_text=text,
                source=source,
                generation=generation,
                temporal=(
                    capture_temporal
                    if suggestion is not None and suggestion.kind == "task"
                    else None
                ),
            )
        await self._nova_companion_deliver(
            update,
            context,
            delivery_message,
            prepared,
        )

    async def _nova_companion_retire_pre_delivery(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> bool:
        task = self._nova_companion_start_pre_delivery_cleanup(
            context,
            delivery_message,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )
        if task is None:
            return False
        return await asyncio.shield(task)

    def _nova_companion_schedule_pre_delivery_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> None:
        self._nova_companion_start_pre_delivery_cleanup(
            context,
            delivery_message,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )

    def _nova_companion_start_pre_delivery_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> asyncio.Task[bool] | None:
        message_id = self._positive_companion_message_id(
            getattr(delivery_message, "message_id", None)
        )
        if delivery_message is None or message_id is None:
            return None
        coroutine = self._nova_companion_pre_delivery_cleanup(
            context,
            delivery_message,
            chat_id=chat_id,
            message_id=message_id,
            neutral_text=neutral_text,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-pre-delivery-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_pre_delivery_cleanup error_type=%s",
                type(exc).__name__,
            )
            return None
        self._track_nova_companion_task(task)
        return task

    async def _nova_companion_pre_delivery_cleanup(
        self,
        context: Any,
        delivery_message: Any,
        *,
        chat_id: int,
        message_id: int,
        neutral_text: str,
    ) -> bool:
        try:
            await self._nova_companion_neutralize_existing(
                context,
                delivery_message,
                chat_id,
                neutral_text,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=pre_delivery_cleanup error_type=%s",
                type(exc).__name__,
            )
            try:
                await self._nova_companion_compensate(
                    context,
                    delivery_message,
                    chat_id=chat_id,
                    message_id=message_id,
                    neutral_text=neutral_text,
                )
            except asyncio.CancelledError:
                raise
            return False

    async def _nova_companion_memory_projection(
        self, user: User
    ) -> tuple[_CompanionCheck, NovaMemoryProjection | None, str | None]:
        policy = self.nova_memory_application_policy()
        if not policy.allows(user.access_tier):
            return "ready", None, None
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
                "Nova companion failed operation=memory_snapshot error_type=%s",
                type(exc).__name__,
            )
            return "unavailable", None, None
        if snapshot.status in {"access_changed", "disabled"}:
            return "access_changed", None, None
        if snapshot.status not in {"ready", "empty"} or not snapshot.collection_revision:
            return "unavailable", None, None
        projection = None
        if snapshot.status == "ready":
            try:
                projection = build_nova_memory_projection(
                    snapshot.items,
                    collection_revision=snapshot.collection_revision,
                )
            except NovaMemoryProjectionError as exc:
                logger.warning(
                    "Nova companion failed operation=memory_projection error_type=%s",
                    type(exc).__name__,
                )
                return "unavailable", None, None
        return "ready", projection, snapshot.collection_revision

    async def _nova_companion_current_check(
        self,
        generation: _CompanionGeneration,
        *,
        exchange_receipt: ConversationExchangeReceipt | None = None,
    ) -> _CompanionCheck:
        if not self.nova_companion_policy().allows_tier(generation.tier):
            return "access_changed"
        try:
            if not await self.nova_companion_context.current_check(
                generation.context_fence,
                exchange_receipt=exchange_receipt,
            ):
                return (
                    "context_changed"
                    if await self._nova_companion_generation_actor_is_current(generation)
                    else "access_changed"
                )
            if generation.memory_revision is not None:
                current = await self.nova_memory_service.application_current_check(
                    telegram_actor_id=generation.telegram_actor_id,
                    expected_tier=generation.tier,
                    expected_access_version=generation.access_version,
                    expected_collection_revision=generation.memory_revision,
                    policy=self.nova_memory_application_policy(),
                )
                if current.status in {"access_changed", "disabled"}:
                    return "access_changed"
                if current.status == "memory_changed":
                    return "context_changed"
                if current.status not in {"ready", "empty"}:
                    return "unavailable"
            return "ready"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=current_check error_type=%s",
                type(exc).__name__,
            )
            return "unavailable"

    async def _nova_companion_generation_actor_is_current(
        self,
        generation: _CompanionGeneration,
    ) -> bool:
        if not self.nova_companion_policy().allows_tier(generation.tier):
            return False
        try:
            async with self.db.sessions() as session:
                owner_id = await session.scalar(
                    select(User.id).where(
                        User.id == generation.owner_id,
                        User.telegram_id == generation.telegram_actor_id,
                        User.access_tier == generation.tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == generation.access_version,
                    )
                )
            return owner_id == generation.owner_id
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=generation_actor error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_actor_is_current(self, user: User) -> bool:
        if not self.nova_companion_policy().allows_actor(user):
            return False
        try:
            async with self.db.sessions() as session:
                owner_id = await session.scalar(
                    select(User.id).where(
                        User.id == user.id,
                        User.telegram_id == user.telegram_id,
                        User.access_tier == user.access_tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == user.access_version,
                    )
                )
            return owner_id == user.id and self.nova_companion_policy().allows_tier(
                user.access_tier
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=actor_check error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_deliver(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        prepared: _PreparedCompanionAnswer,
    ) -> bool:
        coroutine = self._nova_companion_delivery_lifecycle(
            update,
            context,
            delivery_message,
            prepared,
        )
        try:
            task = asyncio.create_task(coroutine, name="nova-companion-delivery-lifecycle")
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        return await asyncio.shield(task)

    async def _nova_companion_delivery_lifecycle(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        prepared: _PreparedCompanionAnswer,
    ) -> bool:
        stage: NovaCompanionCaptureScreen | None = None
        screen: NovaCompanionCaptureScreen | None = None
        exchange_receipt: ConversationExchangeReceipt | None = None
        sent: Any | None = delivery_message
        message_id = self._positive_companion_message_id(
            getattr(delivery_message, "message_id", None)
        )
        try:
            if prepared.suggestion is not None:
                stage = await self.nova_companion_captures.stage(
                    prepared.suggestion,
                    raw_text=prepared.user_text,
                    owner_id=prepared.generation.owner_id,
                    telegram_user_id=prepared.generation.telegram_actor_id,
                    chat_id=prepared.generation.chat_id,
                    access_version=prepared.generation.access_version,
                    temporal=prepared.temporal,
                )
            check = await self._nova_companion_current_check(prepared.generation)
            if check != "ready":
                if stage is not None:
                    await self.nova_companion_captures.revoke_screen(stage)
                if delivery_message is not None:
                    await self._nova_companion_neutralize_existing(
                        context,
                        delivery_message,
                        prepared.generation.chat_id,
                        self._nova_companion_neutral_text(check),
                    )
                return False
            if delivery_message is None:
                sent = await update.effective_message.reply_text(prepared.answer)
            else:
                await sent.edit_text(prepared.answer, reply_markup=None)
            message_id = self._positive_companion_message_id(getattr(sent, "message_id", None))
            if message_id is None:
                raise ValueError("Missing companion message id")
            check = await self._nova_companion_current_check(prepared.generation)
            if check != "ready":
                if stage is not None:
                    await self.nova_companion_captures.revoke_screen(stage)
                await self._nova_companion_compensate(
                    context,
                    sent,
                    chat_id=prepared.generation.chat_id,
                    message_id=message_id,
                    neutral_text=self._nova_companion_neutral_text(check),
                )
                return False
            if stage is not None:
                screen = await self.nova_companion_captures.bind(
                    stage,
                    canonical_message_id=message_id,
                )
                if screen is not None:
                    check = await self._nova_companion_current_check(prepared.generation)
                    if check != "ready":
                        await self._nova_companion_delivery_cleanup(
                            context,
                            prepared,
                            sent=sent,
                            message_id=message_id,
                            stage=stage,
                            screen=screen,
                            exchange_receipt=None,
                            neutral_text=self._nova_companion_neutral_text(check),
                        )
                        return False
                    try:
                        await self._nova_companion_edit_markup(
                            context,
                            sent,
                            chat_id=prepared.generation.chat_id,
                            message_id=message_id,
                            markup=self._nova_companion_capture_markup(screen),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "Nova companion failed operation=suggestion_edit error_type=%s",
                            type(exc).__name__,
                        )
                        await self.nova_companion_captures.revoke_screen(screen)
                        await self._nova_companion_edit_markup(
                            context,
                            sent,
                            chat_id=prepared.generation.chat_id,
                            message_id=message_id,
                            markup=None,
                        )
                        screen = None
                    if screen is not None:
                        check = await self._nova_companion_current_check(prepared.generation)
                        if check != "ready":
                            await self._nova_companion_delivery_cleanup(
                                context,
                                prepared,
                                sent=sent,
                                message_id=message_id,
                                stage=stage,
                                screen=screen,
                                exchange_receipt=None,
                                neutral_text=self._nova_companion_neutral_text(check),
                            )
                            return False
            if prepared.persist_exchange:
                exchange_receipt = await self._nova_companion_append_exchange(prepared)
                if exchange_receipt is None:
                    check = await self._nova_companion_current_check(prepared.generation)
                    if check == "ready":
                        check = "context_changed"
                    await self._nova_companion_delivery_cleanup(
                        context,
                        prepared,
                        sent=sent,
                        message_id=message_id,
                        stage=stage,
                        screen=screen,
                        exchange_receipt=None,
                        neutral_text=self._nova_companion_neutral_text(check),
                    )
                    return False
            check = await self._nova_companion_current_check(
                prepared.generation,
                exchange_receipt=exchange_receipt,
            )
            if check != "ready":
                await self._nova_companion_delivery_cleanup(
                    context,
                    prepared,
                    sent=sent,
                    message_id=message_id,
                    stage=stage,
                    screen=screen,
                    exchange_receipt=exchange_receipt,
                    neutral_text=self._nova_companion_neutral_text(check),
                )
                return False
            return True
        except asyncio.CancelledError:
            self._nova_companion_schedule_delivery_cleanup(
                context,
                prepared,
                sent=sent,
                message_id=message_id,
                stage=stage,
                screen=screen,
                exchange_receipt=exchange_receipt,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=delivery error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_delivery_cleanup(
                context,
                prepared,
                sent=sent,
                message_id=message_id,
                stage=stage,
                screen=screen,
                exchange_receipt=exchange_receipt,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            return False

    def _nova_companion_schedule_delivery_cleanup(
        self,
        context: Any,
        prepared: _PreparedCompanionAnswer,
        *,
        sent: Any | None,
        message_id: int | None,
        stage: NovaCompanionCaptureScreen | None,
        screen: NovaCompanionCaptureScreen | None,
        exchange_receipt: ConversationExchangeReceipt | None,
        neutral_text: str,
    ) -> None:
        coroutine = self._nova_companion_delivery_cleanup(
            context,
            prepared,
            sent=sent,
            message_id=message_id,
            stage=stage,
            screen=screen,
            exchange_receipt=exchange_receipt,
            neutral_text=neutral_text,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-delivery-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_cleanup error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_delivery_cleanup(
        self,
        context: Any,
        prepared: _PreparedCompanionAnswer,
        *,
        sent: Any | None,
        message_id: int | None,
        stage: NovaCompanionCaptureScreen | None,
        screen: NovaCompanionCaptureScreen | None,
        exchange_receipt: ConversationExchangeReceipt | None,
        neutral_text: str,
    ) -> bool:
        clean = True
        canonical_owned = True
        try:
            if screen is not None:
                canonical_owned = await self.nova_companion_captures.revoke_screen(screen)
            elif stage is not None:
                await self.nova_companion_captures.revoke_screen(stage)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            clean = False
            if screen is not None:
                canonical_owned = False
            logger.warning(
                "Nova companion failed operation=capability_cleanup error_type=%s",
                type(exc).__name__,
            )
        if exchange_receipt is not None:
            try:
                compensated = await self.conversation.compensate_exchange(exchange_receipt)
                if not compensated:
                    clean = False
                    logger.warning(
                        "Nova companion failed operation=exchange_cleanup error_type=StateChanged"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                clean = False
                logger.warning(
                    "Nova companion failed operation=exchange_cleanup error_type=%s",
                    type(exc).__name__,
                )
        if canonical_owned and sent is not None and message_id is not None:
            try:
                await self._nova_companion_compensate(
                    context,
                    sent,
                    chat_id=prepared.generation.chat_id,
                    message_id=message_id,
                    neutral_text=neutral_text,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                clean = False
                logger.warning(
                    "Nova companion failed operation=delivery_cleanup error_type=%s",
                    type(exc).__name__,
                )
        return clean

    async def _nova_companion_append_exchange(
        self,
        prepared: _PreparedCompanionAnswer,
    ) -> ConversationExchangeReceipt | None:
        fence = prepared.generation.context_fence.conversation_fence
        if fence is None:
            return None
        return await self.conversation.append_exchange(
            fence,
            user_content=prepared.user_text,
            assistant_content=prepared.answer,
            user_source=prepared.source,
        )

    async def nova_companion_callback(self, update: Any, context: Any) -> None:
        query = update.callback_query
        if query is None:
            return
        user = await self._nova_companion_lookup_actor(update.effective_user.id)
        message_id = self._positive_companion_message_id(
            getattr(getattr(query, "message", None), "message_id", None)
        )
        capability = None
        if user is not None and message_id:
            capability = await self.nova_companion_captures.peek_bound_identity(
                query.data,
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                canonical_message_id=message_id,
            )
        await self._nova_companion_answer_query(query)
        if capability is None or not await self.nova_companion_captures.consume(capability):
            return
        assert user is not None
        coroutine = self._nova_companion_callback_lifecycle(
            update,
            context,
            query,
            user,
            capability,
        )
        try:
            task = asyncio.create_task(coroutine, name="nova-companion-callback-lifecycle")
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        await asyncio.shield(task)

    async def _nova_companion_answer_query(self, query: Any) -> None:
        try:
            await query.answer()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_answer error_type=%s",
                type(exc).__name__,
            )

    async def _nova_companion_lookup_actor(self, telegram_actor_id: int) -> User | None:
        try:
            async with self.db.sessions() as session:
                return await session.scalar(
                    select(User).where(User.telegram_id == telegram_actor_id)
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_actor error_type=%s",
                type(exc).__name__,
            )
            return None

    async def _nova_companion_callback_lifecycle(
        self,
        update: Any,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        try:
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                return False
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if capability.action == "not_now":
                if await self._nova_companion_clear_query_markup(query, context):
                    return True
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    "Хорошо, не сохраняю.",
                )
                return False
            return await self._nova_companion_add_capture(
                update,
                context,
                query,
                user,
                capability,
            )
        except asyncio.CancelledError as exc:
            if not bool(exc.__dict__.get(_COMPANION_CLEANUP_SCHEDULED_ATTR)):
                self._nova_companion_schedule_callback_recovery(
                    context,
                    query,
                    user,
                    capability,
                )
            raise
        except _NovaCompanionCaptureCompensationError as exc:
            logger.warning(
                "Nova companion failed operation=callback_compensation error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_neutralize_consumed_if_current(
                query,
                context,
                capability,
                NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_lifecycle error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_restore_capture(context, query, user, capability)
            return False

    async def _nova_companion_add_capture(
        self,
        update: Any,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        if not await self._nova_companion_capability_actor_is_current(user, capability):
            if await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
            return False
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        if (
            capability.action == "add"
            and capability.temporal is not None
            and capability.temporal.resolution.status == "conflict"
        ):
            return await self._nova_companion_restore_capture(
                context,
                query,
                user,
                capability,
            )
        creation = None
        preview = None
        accepted_previews: list[Any] = []
        previous_preview_message_id: int | None = None
        preview_pointer_owned = False
        focus_lease = None
        focus_owned = False
        try:
            parsed = self._nova_companion_parsed_capture(capability)
            if parsed is None:
                await self._nova_companion_restore_capture(
                    context,
                    query,
                    user,
                    capability,
                )
                return False
            creation = await self.draft_service.create_or_get_for_suggestion(
                user_id=capability.owner_id,
                telegram_user_id=capability.telegram_user_id,
                chat_id=capability.chat_id,
                expected_access_version=capability.access_version,
                source="companion",
                raw_text=capability.raw_text,
                parsed=parsed,
            )
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                if creation.ok and creation.draft is not None:
                    await self._nova_companion_cleanup_capture_preview(
                        context,
                        creation,
                        capability,
                        preview=None,
                    )
                return False
            if not creation.ok or creation.draft is None:
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            draft = creation.draft
            previous_preview_message_id = self._positive_companion_message_id(
                draft.preview_message_id
            )
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=None,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=None,
                )
                return False
            preview = await self._send_draft_preview(
                query.message,
                draft,
                include_original=True,
                on_sent=accepted_previews.append,
                bind_preview=False,
            )
            preview_message_id = self._positive_companion_message_id(
                getattr(preview, "message_id", None)
            )
            if (
                preview_message_id is None
                or not await self.draft_service.restore_preview_message_if_current(
                    draft.id,
                    draft.version,
                    capability.telegram_user_id,
                    capability.chat_id,
                    expected_message_id=previous_preview_message_id,
                    restored_message_id=preview_message_id,
                )
            ):
                raise RuntimeError("Preview binding changed")
            preview_pointer_owned = True
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                )
                return False
            await self._remember_preview(capability.telegram_user_id, capability.chat_id, draft)
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                return False
            cleared = await self._nova_companion_clear_query_markup(query, context)
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                return False
            if not cleared:
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_CAPTURE_FAILED_TEXT,
                )
            focus_lease = await self.conversation.acquire_active_draft_focus(
                capability.telegram_user_id,
                capability.chat_id,
                draft.id,
            )
            if focus_lease is None:
                raise RuntimeError("Draft focus changed")
            focus_owned = True
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    focus_capability=capability,
                    restore_prior_focus=False,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    focus_capability=capability,
                )
                return False
            return True
        except asyncio.CancelledError as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            if creation is not None:
                self._nova_companion_schedule_capture_cleanup(
                    context,
                    query,
                    user,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                )
                exc.__dict__[_COMPANION_CLEANUP_SCHEDULED_ATTR] = True
            raise
        except Exception as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            if creation is not None:
                cleaned = await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    focus_capability=capability,
                )
                if not cleaned:
                    raise _NovaCompanionCaptureCompensationError from exc
            raise

    def _nova_companion_schedule_capture_cleanup(
        self,
        context: Any,
        query: Any,
        user: User,
        creation: Any,
        capability: NovaCompanionCaptureCapability,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool,
        focus_lease: Any | None,
        focus_owned: bool,
    ) -> None:
        coroutine = self._nova_companion_capture_cleanup_lifecycle(
            context,
            query,
            user,
            creation,
            capability,
            preview=preview,
            previous_preview_message_id=previous_preview_message_id,
            preview_pointer_owned=preview_pointer_owned,
            focus_lease=focus_lease,
            focus_owned=focus_owned,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-capture-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_capture_cleanup error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    def _nova_companion_schedule_callback_recovery(
        self,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> None:
        coroutine = self._nova_companion_callback_recovery_lifecycle(
            context,
            query,
            user,
            capability,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-callback-recovery-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_callback_recovery error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_callback_recovery_lifecycle(
        self,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        if capability.action == "not_now":
            await self._nova_companion_neutralize_consumed_if_current(
                query,
                context,
                capability,
                "Хорошо, не сохраняю.",
            )
            return True
        await self._nova_companion_restore_capture(context, query, user, capability)
        return True

    async def _nova_companion_capture_cleanup_lifecycle(
        self,
        context: Any,
        query: Any,
        user: User,
        creation: Any,
        capability: NovaCompanionCaptureCapability,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool,
        focus_lease: Any | None,
        focus_owned: bool,
    ) -> bool:
        cleaned = await self._nova_companion_cleanup_capture_preview(
            context,
            creation,
            capability,
            preview=preview,
            previous_preview_message_id=previous_preview_message_id,
            preview_pointer_owned=preview_pointer_owned,
            focus_lease=focus_lease,
            focus_owned=focus_owned,
            focus_user=user,
            focus_capability=capability,
        )
        if cleaned:
            await self._nova_companion_restore_capture(context, query, user, capability)
            return True
        await self._nova_companion_neutralize_consumed_if_current(
            query,
            context,
            capability,
            NOVA_COMPANION_CAPTURE_FAILED_TEXT,
        )
        return False

    async def _nova_companion_cleanup_capture_preview(
        self,
        context: Any,
        creation: Any,
        capability: NovaCompanionCaptureCapability,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None = None,
        preview_pointer_owned: bool = False,
        focus_lease: Any | None = None,
        focus_owned: bool = False,
        focus_user: User | None = None,
        focus_capability: NovaCompanionCaptureCapability | None = None,
        restore_prior_focus: bool | None = None,
    ) -> bool:
        storage_clean = True
        preview_id = self._positive_companion_message_id(getattr(preview, "message_id", None))
        if creation.created:
            storage_clean = await self._nova_companion_drop_created_draft(
                creation,
                telegram_user_id=capability.telegram_user_id,
                chat_id=capability.chat_id,
                expected_preview_message_id=(
                    preview_id if preview_pointer_owned else previous_preview_message_id
                ),
            )
        elif creation.draft is not None and preview_id is not None and preview_pointer_owned:
            restored = await self._nova_companion_restore_reused_preview_pointer(
                creation.draft,
                telegram_user_id=capability.telegram_user_id,
                chat_id=capability.chat_id,
                expected_message_id=preview_id,
                restored_message_id=previous_preview_message_id,
            )
            storage_clean = storage_clean and restored
        if focus_lease is not None:
            storage_clean = (
                await self._nova_companion_restore_focus_lease(
                    focus_lease,
                    user=focus_user,
                    capability=focus_capability,
                    restore_prior=restore_prior_focus,
                )
                and storage_clean
            )
        elif focus_owned and creation.draft is not None:
            try:
                cleared = await self.conversation.clear_active_draft_if_current(
                    capability.telegram_user_id,
                    capability.chat_id,
                    creation.draft.id,
                    creation.draft.version,
                )
                storage_clean = storage_clean and cleared
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                storage_clean = False
                logger.warning(
                    "Nova companion failed operation=focus_compensation error_type=%s",
                    type(exc).__name__,
                )
        if preview is not None and preview_id is not None:
            await self._nova_companion_compensate(
                context,
                preview,
                chat_id=capability.chat_id,
                message_id=preview_id,
                neutral_text=NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
        return storage_clean

    async def _nova_companion_drop_created_draft(
        self,
        creation: Any,
        *,
        telegram_user_id: int,
        chat_id: int,
        expected_preview_message_id: int | None,
    ) -> bool:
        if not creation.created or creation.draft is None:
            return True
        try:
            result = await self.draft_service.drop_if_preview_message_current(
                creation.draft.id,
                creation.draft.version,
                telegram_user_id,
                chat_id,
                expected_message_id=expected_preview_message_id,
            )
            if result.ok:
                return True
            logger.warning(
                "Nova companion failed operation=draft_compensation error_type=StateChanged"
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=draft_compensation error_type=%s",
                type(exc).__name__,
            )
            return False

    def _nova_companion_parsed_capture(
        self,
        capability: NovaCompanionCaptureCapability,
    ) -> ParsedThought | None:
        resolved_date = None
        temporal_resolution = None
        temporal = capability.temporal
        if temporal is not None:
            resolution = temporal.resolution
            if capability.action == "add":
                if resolution.status != "resolved" or resolution.target_date is None:
                    return None
                resolved_date = resolution.target_date
            elif capability.action in {"date_first", "date_second"}:
                if resolution.status != "conflict" or len(resolution.options) != 2:
                    return None
                option_index = 0 if capability.action == "date_first" else 1
                resolved_date = resolution.options[option_index].value
            else:
                return None
            temporal_resolution = self.date_resolver.temporal_resolution(
                resolved_date,
                temporal.timezone,
                capability.raw_text,
                temporal.local_time,
            )
        elif capability.action != "add":
            return None
        return ParsedThought(
            kind=capability.suggestion.kind,
            title=capability.suggestion.title,
            next_step=capability.suggestion.next_step,
            resolved_date=resolved_date,
            temporal_resolution=temporal_resolution,
        )

    async def _nova_companion_capability_actor_is_current(
        self,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        expected_timezone = (
            capability.temporal.timezone if capability.temporal is not None else None
        )
        if (
            user.id != capability.owner_id
            or user.telegram_id != capability.telegram_user_id
            or user.access_version != capability.access_version
            or (expected_timezone is not None and user.timezone != expected_timezone)
            or not self.nova_companion_policy().allows_actor(
                user,
                expected_access_version=capability.access_version,
            )
        ):
            return False
        if not await self._nova_companion_actor_is_current(user):
            return False
        if expected_timezone is None:
            return True
        try:
            async with self.db.sessions() as session:
                owner_id = await session.scalar(
                    select(User.id).where(
                        User.id == capability.owner_id,
                        User.telegram_id == capability.telegram_user_id,
                        User.access_tier == user.access_tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == capability.access_version,
                        User.timezone == expected_timezone,
                    )
                )
            return owner_id == capability.owner_id
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=capability_actor error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_restore_capture(
        self,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        if not await self._nova_companion_capability_actor_is_current(user, capability):
            await self._nova_companion_neutralize_consumed_if_current(
                query,
                context,
                capability,
                NOVA_COMPANION_ACCESS_CHANGED_TEXT,
            )
            return False
        recovery: NovaCompanionCaptureScreen | None = None
        recovery_capability: NovaCompanionCaptureCapability | None = None
        actions = (
            CAPTURE_DATE_ACTIONS
            if capability.temporal is not None
            and capability.temporal.resolution.status == "conflict"
            else CAPTURE_ACTIONS
        )
        primary_action = actions[0]
        try:
            recovery = await self.nova_companion_captures.stage_recovery(
                capability,
                actions=actions,
            )
            if recovery is None:
                return False
            recovery_capability = await self.nova_companion_captures.peek(
                recovery.callback_data(primary_action),
                owner_id=recovery.owner_id,
                telegram_user_id=recovery.telegram_user_id,
                chat_id=recovery.chat_id,
                canonical_message_id=capability.canonical_message_id,
                access_version=recovery.access_version,
                expected_action=primary_action,
            )
            if recovery_capability is None:
                await self.nova_companion_captures.revoke_screen(recovery)
                return False
            await self._nova_companion_render_recovery(
                context,
                query,
                recovery,
            )
            return True
        except asyncio.CancelledError as exc:
            if recovery is not None:
                await self.nova_companion_captures.revoke_screen(recovery)
            if recovery_capability is not None:
                self._nova_companion_schedule_consumed_neutralization(
                    context,
                    query,
                    recovery_capability,
                    NOVA_COMPANION_CAPTURE_FAILED_TEXT,
                )
                exc.__dict__[_COMPANION_CLEANUP_SCHEDULED_ATTR] = True
            raise
        except Exception as exc:
            if recovery is not None:
                await self.nova_companion_captures.revoke_screen(recovery)
            logger.warning(
                "Nova companion failed operation=capture_recovery error_type=%s",
                type(exc).__name__,
            )
            if recovery_capability is not None:
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    recovery_capability,
                    NOVA_COMPANION_CAPTURE_FAILED_TEXT,
                )
            return False

    async def _nova_companion_render_recovery(
        self,
        context: Any,
        query: Any,
        screen: NovaCompanionCaptureScreen,
    ) -> None:
        markup = self._nova_companion_capture_markup(screen)
        temporal = screen.temporal
        if temporal is not None and temporal.resolution.status == "conflict":
            text = self.date_resolver.conflict_message(temporal.resolution)
            edit = getattr(query, "edit_message_text", None)
            if callable(edit):
                await edit(text, reply_markup=markup)
                return
            bot_edit = getattr(getattr(context, "bot", None), "edit_message_text", None)
            if callable(bot_edit):
                await bot_edit(
                    chat_id=screen.chat_id,
                    message_id=screen.canonical_message_id,
                    text=text,
                    reply_markup=markup,
                    parse_mode=None,
                )
                return
            raise TelegramError("Companion message cannot be edited")
        await self._nova_companion_edit_markup(
            context,
            query.message,
            chat_id=screen.chat_id,
            message_id=screen.canonical_message_id,
            markup=markup,
        )

    def _nova_companion_schedule_consumed_neutralization(
        self,
        context: Any,
        query: Any,
        capability: NovaCompanionCaptureCapability,
        text: str,
    ) -> None:
        coroutine = self._nova_companion_neutralize_consumed_if_current(
            query,
            context,
            capability,
            text,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-callback-neutralize-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_neutralize error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_clear_query_markup(self, query: Any, context: Any) -> bool:
        try:
            edit = getattr(query, "edit_message_reply_markup", None)
            if callable(edit):
                await edit(reply_markup=None)
                return True
            await self._nova_companion_edit_markup(
                context,
                query.message,
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                markup=None,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_edit error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_neutralize_consumed_if_current(
        self,
        query: Any,
        context: Any,
        capability: NovaCompanionCaptureCapability,
        text: str,
    ) -> bool:
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        if await self._nova_companion_neutralize_query(query, context, text):
            return True
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        message = getattr(query, "message", None)
        if (
            message is None
            or self._positive_companion_message_id(getattr(message, "message_id", None))
            != capability.canonical_message_id
        ):
            return False
        return await self._nova_companion_compensate(
            context,
            message,
            chat_id=capability.chat_id,
            message_id=capability.canonical_message_id,
            neutral_text=text,
        )

    async def _nova_companion_neutralize_query(self, query: Any, context: Any, text: str) -> bool:
        try:
            edit = getattr(query, "edit_message_text", None)
            if callable(edit):
                await edit(text, reply_markup=None)
                return True
            await self._nova_companion_neutralize_existing(
                context,
                query.message,
                query.message.chat_id,
                text,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_neutralize error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_edit_markup(
        self,
        context: Any,
        message: Any,
        *,
        chat_id: int,
        message_id: int,
        markup: InlineKeyboardMarkup | None,
    ) -> None:
        edit = getattr(message, "edit_reply_markup", None)
        if callable(edit):
            await edit(reply_markup=markup)
            return
        bot_edit = getattr(getattr(context, "bot", None), "edit_message_reply_markup", None)
        if callable(bot_edit):
            await bot_edit(chat_id=chat_id, message_id=message_id, reply_markup=markup)
            return
        edit_text = getattr(message, "edit_text", None)
        if callable(edit_text):
            text = getattr(message, "text", None)
            if isinstance(text, str):
                await edit_text(text, reply_markup=markup)
                return
        raise TelegramError("Companion message cannot be edited")

    async def _nova_companion_neutralize_existing(
        self,
        context: Any,
        message: Any,
        chat_id: int,
        text: str,
    ) -> bool:
        message_id = self._positive_companion_message_id(getattr(message, "message_id", None))
        if message_id is None:
            return False
        bot_edit = getattr(getattr(context, "bot", None), "edit_message_text", None)
        if callable(bot_edit):
            await bot_edit(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=None,
                parse_mode=None,
            )
            return True
        edit = getattr(message, "edit_text", None)
        if callable(edit):
            await edit(text, reply_markup=None, parse_mode=None)
            return True
        return False

    async def _nova_companion_compensate(
        self,
        context: Any,
        message: Any,
        *,
        chat_id: int,
        message_id: int,
        neutral_text: str,
    ) -> bool:
        deleted = False
        try:
            bot_delete = getattr(getattr(context, "bot", None), "delete_message", None)
            if callable(bot_delete):
                result = await bot_delete(chat_id=chat_id, message_id=message_id)
                deleted = result is not False
            else:
                delete = getattr(message, "delete", None)
                if callable(delete):
                    result = await delete()
                    deleted = result is not False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=delivery_delete error_type=%s",
                type(exc).__name__,
            )
        if deleted:
            return True
        try:
            return await self._nova_companion_neutralize_existing(
                context,
                message,
                chat_id,
                neutral_text,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=delivery_neutralize error_type=%s",
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _nova_companion_capture_markup(
        screen: NovaCompanionCaptureScreen,
    ) -> InlineKeyboardMarkup:
        callbacks = screen.callbacks
        if "date_first" in callbacks and "date_second" in callbacks:
            temporal = screen.temporal
            if temporal is None or temporal.resolution.status != "conflict":
                raise ValueError("Invalid companion date-choice screen")
            first, second = temporal.resolution.options
            return InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"{first.weekday} {first.value.strftime('%d.%m.%Y')}",
                            callback_data=screen.callback_data("date_first"),
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            f"{second.weekday} {second.value.strftime('%d.%m.%Y')}",
                            callback_data=screen.callback_data("date_second"),
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Не сейчас",
                            callback_data=screen.callback_data("not_now"),
                        )
                    ],
                ]
            )
        labels = {
            "idea": "идею",
            "task": "задачу",
            "desire": "желание",
            "note": "заметку",
        }
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        f"Добавить как {labels[screen.suggestion.kind]}",
                        callback_data=screen.callback_data("add"),
                    ),
                    InlineKeyboardButton(
                        "Не сейчас",
                        callback_data=screen.callback_data("not_now"),
                    ),
                ]
            ]
        )

    @staticmethod
    def _positive_companion_message_id(value: object) -> int | None:
        return value if type(value) is int and value > 0 else None

    @staticmethod
    def _nova_companion_neutral_text(outcome: _CompanionCheck) -> str:
        if outcome == "access_changed":
            return NOVA_COMPANION_ACCESS_CHANGED_TEXT
        if outcome == "context_changed":
            return NOVA_COMPANION_CONTEXT_CHANGED_TEXT
        return NOVA_COMPANION_UNAVAILABLE_TEXT

    def _track_nova_companion_task(self, task: asyncio.Task[bool]) -> None:
        self._nova_companion_tasks.add(task)

        def finish(completed: asyncio.Task[bool]) -> None:
            self._nova_companion_tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError as exc:
                logger.warning(
                    "Nova companion task finished operation=lifecycle error_type=%s",
                    type(exc).__name__,
                )
            except BaseException as exc:
                logger.warning(
                    "Nova companion task failed operation=lifecycle error_type=%s",
                    type(exc).__name__,
                )

        task.add_done_callback(finish)

    async def _drain_nova_companion_tasks(self) -> None:
        current = asyncio.current_task()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _COMPANION_DRAIN_TIMEOUT_SECONDS
        while True:
            pending = self._pending_nova_companion_tasks(current)
            if not pending:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            _done, still_pending = await asyncio.wait(pending, timeout=remaining)
            if still_pending:
                break
        pending = self._pending_nova_companion_tasks(current)
        if not pending:
            return
        logger.warning(
            "Nova companion shutdown operation=drain error_type=TimeoutError pending_count=%s",
            len(pending),
        )
        cancel_deadline = loop.time() + _COMPANION_CANCEL_TIMEOUT_SECONDS
        while True:
            pending = self._pending_nova_companion_tasks(current)
            if not pending:
                return
            for task in pending:
                task.cancel()
            remaining = cancel_deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.wait(
                pending,
                timeout=min(_COMPANION_CANCEL_RETRY_SECONDS, remaining),
            )
        pending = self._pending_nova_companion_tasks(current)
        if pending:
            logger.error(
                "Nova companion shutdown operation=terminal_drain error_type=TimeoutError "
                "pending_count=%s",
                len(pending),
            )
            raise _NovaCompanionDrainError("Nova companion terminal drain timed out")

    def _pending_nova_companion_tasks(
        self, current: asyncio.Task[Any] | None
    ) -> set[asyncio.Task[bool]]:
        for task in tuple(self._nova_companion_tasks):
            if task is current or not task.done():
                continue
            try:
                task.result()
            except BaseException:
                pass
            self._nova_companion_tasks.discard(task)
        return {
            task
            for task in tuple(self._nova_companion_tasks)
            if task is not current and not task.done()
        }
