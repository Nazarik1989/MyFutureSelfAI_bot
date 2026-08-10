from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .access import FULL_ACCESS_TIERS, is_full_access_tier
from .drafts import DraftInboxService, DraftResult, log_transition
from .models import User
from .recurring_reminders import RecurringScheduleMutation
from .reminder_flow import (
    ReminderFlowAction,
    ReminderFlowPhase,
    ReminderFlowSession,
)
from .reminder_intent import (
    ReminderIntentCode,
    ReminderIntentResult,
    ReminderIntentStatus,
    ReminderScheduleKind,
    ReminderTimezoneSource,
    calculate_daily_occurrence,
    first_daily_occurrence_utc,
)
from .schemas import ParsedThought, TemporalResolution

logger = logging.getLogger(__name__)

REMINDER_ACCESS_CHANGED_TEXT = "🔔 Напоминание\n\nДоступ изменился. Ничего не сохранено — повтори команду после проверки доступа."
REMINDER_STALE_TEXT = "Эта карточка уже неактуальна. Повтори команду напоминания."

_TIME_ONLY = re.compile(
    r"^\s*(?:в\s+)?(?:[01]?\d|2[0-3])(?:\s*[:.]\s*[0-5]\d|\s+час(?:а|ов)?(?:\s+[0-5]?\d\s+минут(?:у|ы)?)?)"
    r"(?:\s*(?:по\s+)?(?:мск|по\s+москве|московское\s+время|[A-Za-z][A-Za-z0-9._+-]*/[A-Za-z0-9._+/-]+))?\s*$",
    re.IGNORECASE,
)
_UNSUPPORTED_RECURRENCE = re.compile(
    r"\b(?:кажд(?:ую|ой)\s+недел(?:ю|и|е|ей)?|по\s+будням|по\s+выходным|"
    r"кажд(?:ый|ую)\s+(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)|"
    r"через\s+день|раз\s+в\s+\d+\s+дн)\b",
    re.IGNORECASE,
)


class _ReminderPastAtSave(RuntimeError):
    pass


@dataclass(slots=True)
class ReminderVoiceGateState:
    access_expected: bool = True
    access_failed: bool = False


class ReminderHandlers:
    reminder_sessions: Any
    reminder_intent_parser: Any
    recurring_reminder_service: Any
    draft_service: DraftInboxService

    async def reminder_text_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        text = getattr(update.effective_message, "text", None)
        if not isinstance(text, str):
            return False
        return await self._reminder_question_gate(
            update,
            context,
            text,
            candidate_message=None,
            expected_access_version=None,
            expected_session=None,
            voice_fenced=False,
            voice_state=None,
        )

    async def reminder_voice_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        transcript: str,
        progress: Any,
        *,
        expected_access_version: int,
        expected_session: ReminderFlowSession | None,
        voice_state: ReminderVoiceGateState | None = None,
    ) -> bool:
        return await self._reminder_question_gate(
            update,
            context,
            transcript,
            candidate_message=progress,
            expected_access_version=expected_access_version,
            expected_session=expected_session,
            voice_fenced=True,
            voice_state=voice_state,
        )

    async def _reminder_question_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        *,
        candidate_message: Any | None,
        expected_access_version: int | None,
        expected_session: ReminderFlowSession | None,
        voice_fenced: bool,
        voice_state: ReminderVoiceGateState | None,
    ) -> bool:
        binding = await self._reminder_access(update)
        if binding is None:
            relative = self.date_resolver.resolve_relative_reminder(text, "UTC")
            if expected_session is None and relative:
                if voice_fenced and expected_access_version is not None:
                    await self._reminder_edit_access_candidate(candidate_message)
                return True
            if expected_session is not None:
                if voice_fenced:
                    await self._reminder_retire_voice_candidate(candidate_message)
                await self._reminder_access_changed(
                    context,
                    expected_session,
                    source_message=candidate_message or update.effective_message,
                )
                return True
            probe = self.reminder_intent_parser.parse(text, "UTC")
            handled = probe.status is not ReminderIntentStatus.NOT_REMINDER
            if handled and voice_fenced:
                await self._reminder_edit_access_candidate(candidate_message)
            elif (
                voice_fenced
                and expected_access_version is not None
                and voice_state is not None
                and voice_state.access_expected
            ):
                voice_state.access_failed = True
            return handled
        user = binding
        telegram_user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
            )
            if voice_fenced and not self._reminder_expected_session_matches(
                current,
                expected_session,
            ):
                await self._reminder_retire_voice_candidate(candidate_message)
                return True
            fresh_binding = await self._reminder_access(update)
            if (
                fresh_binding is None
                or fresh_binding.id != user.id
                or fresh_binding.access_version != user.access_version
            ):
                probe = self.reminder_intent_parser.parse(text, user.timezone)
                if current is not None or expected_session is not None:
                    if voice_fenced:
                        await self._reminder_retire_voice_candidate(candidate_message)
                    await self._reminder_access_changed(
                        context,
                        current or expected_session,
                        source_message=candidate_message or update.effective_message,
                    )
                    return True
                relative = self.date_resolver.resolve_relative_reminder(text, user.timezone)
                handled = bool(relative) or probe.status is not ReminderIntentStatus.NOT_REMINDER
                if handled and voice_fenced:
                    await self._reminder_edit_access_candidate(candidate_message)
                elif voice_fenced and voice_state is not None and voice_state.access_expected:
                    voice_state.access_failed = True
                return handled
            user = fresh_binding
            if (
                expected_access_version is not None
                and user.access_version != expected_access_version
            ):
                stale_session = current or expected_session
                if stale_session is None:
                    probe = self.reminder_intent_parser.parse(text, user.timezone)
                    relative = self.date_resolver.resolve_relative_reminder(text, user.timezone)
                    handled = (
                        bool(relative) or probe.status is not ReminderIntentStatus.NOT_REMINDER
                    )
                    if not handled:
                        if voice_fenced and voice_state is not None and voice_state.access_expected:
                            voice_state.access_failed = True
                        return False
                    if voice_fenced:
                        await self._reminder_edit_access_candidate(candidate_message)
                else:
                    if voice_fenced:
                        await self._reminder_retire_voice_candidate(candidate_message)
                    await self._reminder_access_changed(
                        context,
                        stale_session,
                        source_message=candidate_message or update.effective_message,
                    )
                return True

            fresh = self.reminder_intent_parser.parse(text, user.timezone)
            replacement = (
                current is not None and fresh.status is not ReminderIntentStatus.NOT_REMINDER
            )
            if current is None and fresh.status is ReminderIntentStatus.NOT_REMINDER:
                return False
            if current is None and self.date_resolver.resolve_relative_reminder(
                text, user.timezone
            ):
                return False

            if current is not None and not replacement:
                parse_text = text
                if current.phase is ReminderFlowPhase.TIME and _TIME_ONLY.fullmatch(text):
                    parse_text = f"в {text.strip()}"
                result = self.reminder_intent_parser.parse(
                    parse_text,
                    user.timezone,
                    continuation=True,
                    previous=current.parser_state(),
                )
            else:
                result = fresh

            if (
                result.status is not ReminderIntentStatus.NOT_REMINDER
                and _UNSUPPORTED_RECURRENCE.search(text)
            ):
                await self._reminder_show_unsupported(
                    update,
                    context,
                    current,
                    candidate_message,
                )
                return True

            phase = self._reminder_phase(result)
            await self.nova_clear_bound(user.id, chat_id)
            canonical_message_id = current.canonical_message_id if current is not None else None
            if candidate_message is not None:
                candidate_id = getattr(candidate_message, "message_id", None)
                if current is None and isinstance(candidate_id, int):
                    canonical_message_id = candidate_id
                elif current is not None:
                    await self._reminder_retire_voice_candidate(candidate_message)

            session = await self.reminder_sessions.create(
                owner_id=user.id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                access_version=user.access_version,
                title=result.title,
                schedule_kind=result.schedule_kind,
                local_date=result.local_date,
                local_time=result.local_time,
                timezone=result.timezone or user.timezone,
                timezone_source=result.timezone_source or ReminderTimezoneSource.PROFILE,
                phase=phase,
                canonical_message_id=canonical_message_id,
            )
            delivery_binding = await self._reminder_access(update)
            if (
                delivery_binding is None
                or delivery_binding.id != session.owner_id
                or delivery_binding.access_version != session.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    session,
                    source_message=candidate_message or update.effective_message,
                )
                return True
            async with self._reminder_ui_lock:
                live = await self.reminder_sessions.get_exact(session)
                if live is None:
                    await self._reminder_retire_voice_candidate(candidate_message)
                    return True
                session = live
                if session.canonical_message_id is None:
                    sent = await update.effective_message.reply_text(
                        "🔔 Готовлю напоминание…",
                    )
                    message_id = getattr(sent, "message_id", None)
                    if not isinstance(message_id, int):
                        await self.reminder_sessions.clear(
                            owner_id=session.owner_id,
                            telegram_user_id=session.telegram_user_id,
                            chat_id=session.chat_id,
                            session_id=session.id,
                        )
                        return True
                    bound = await self.reminder_sessions.update(
                        session,
                        canonical_message_id=message_id,
                    )
                    if bound is None:
                        await self._reminder_retire_voice_candidate(sent)
                        return True
                    bound_access = await self._reminder_access(update)
                    if (
                        bound_access is None
                        or bound_access.id != bound.owner_id
                        or bound_access.access_version != bound.access_version
                    ):
                        cleared = await self.reminder_sessions.clear(
                            owner_id=bound.owner_id,
                            telegram_user_id=bound.telegram_user_id,
                            chat_id=bound.chat_id,
                            session_id=bound.id,
                        )
                        if cleared:
                            await self._reminder_edit_text(
                                context,
                                bound,
                                REMINDER_ACCESS_CHANGED_TEXT,
                                None,
                                source_message=sent,
                            )
                        return True
                    await self._reminder_edit_canonical(
                        context,
                        bound,
                        source_message=sent,
                    )
                else:
                    await self._reminder_edit_canonical(
                        context,
                        session,
                        source_message=candidate_message or update.effective_message,
                    )
            return True

    async def reminder_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        data = str(query.data or "")
        token = data.removeprefix("rmd:") if data.startswith("rmd:") else ""
        if not token or len(token) > 40:
            await query.answer(REMINDER_STALE_TEXT, show_alert=True)
            return
        async with self._reminder_launch_lock:
            claim = await self.reminder_sessions.claim(
                token,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                canonical_message_id=getattr(query.message, "message_id", None),
            )
            if claim is None:
                await query.answer(REMINDER_STALE_TEXT, show_alert=True)
                return
            await query.answer()
            capability, session = claim
            user = await self._reminder_access(update)
            if (
                user is None
                or user.id != session.owner_id
                or user.access_version != session.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    session,
                    source_message=query.message,
                )
                return
            action = capability.action
            if action == "cancel":
                await self.reminder_sessions.clear(
                    owner_id=session.owner_id,
                    telegram_user_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    session_id=session.id,
                )
                await self._reminder_edit_text(
                    context,
                    session,
                    "🔔 Напоминание отменено. Ничего не сохранено.",
                    None,
                    query=query,
                )
                return
            if action == "confirm":
                await self._reminder_confirm(update, context, query, user, session)
                return

            updated = await self._reminder_apply_action(session, action)
            if updated is None:
                return
            fresh_user = await self._reminder_access(update)
            if (
                fresh_user is None
                or fresh_user.id != updated.owner_id
                or fresh_user.access_version != updated.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    updated,
                    source_message=query.message,
                )
                return
            async with self._reminder_ui_lock:
                await self._reminder_edit_canonical(
                    context,
                    updated,
                    query=query,
                    source_message=query.message,
                )

    async def _reminder_apply_action(
        self,
        session: ReminderFlowSession,
        action: ReminderFlowAction,
    ) -> ReminderFlowSession | None:
        now = self._reminder_now()
        local_today = now.astimezone(ZoneInfo(session.timezone)).date()
        if action == "today":
            return await self._reminder_update_and_advance(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=local_today,
            )
        if action == "tomorrow":
            return await self._reminder_update_and_advance(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=local_today + timedelta(days=1),
            )
        if action == "daily":
            return await self._reminder_update_and_advance(
                session,
                schedule_kind=ReminderScheduleKind.DAILY,
                local_date=None,
            )
        if action == "choose_date":
            return await self.reminder_sessions.update(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=None,
                phase=ReminderFlowPhase.DATE,
            )
        if action == "edit":
            return await self.reminder_sessions.update(session, phase=ReminderFlowPhase.EDIT)
        if action == "edit_when":
            return await self.reminder_sessions.update(
                session,
                schedule_kind=ReminderScheduleKind.ONCE,
                local_date=None,
                phase=ReminderFlowPhase.WHEN,
            )
        if action == "edit_time":
            return await self.reminder_sessions.update(
                session,
                local_time=None,
                phase=ReminderFlowPhase.TIME,
            )
        if action == "edit_title":
            return await self.reminder_sessions.update(
                session,
                title=None,
                phase=ReminderFlowPhase.TITLE,
            )
        return None

    async def _reminder_update_and_advance(
        self,
        session: ReminderFlowSession,
        *,
        schedule_kind: ReminderScheduleKind,
        local_date: date | None,
    ) -> ReminderFlowSession | None:
        phase = self._reminder_fields_phase(
            title=session.title,
            schedule_kind=schedule_kind,
            local_date=local_date,
            local_time=session.local_time,
            timezone=session.timezone,
        )
        return await self.reminder_sessions.update(
            session,
            schedule_kind=schedule_kind,
            local_date=local_date,
            phase=phase,
        )

    async def _reminder_confirm(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        user: User,
        session: ReminderFlowSession,
    ) -> None:
        if session.phase is not ReminderFlowPhase.PREVIEW:
            return
        fresh = await self._reminder_access(update)
        if (
            fresh is None
            or fresh.id != session.owner_id
            or fresh.access_version != session.access_version
        ):
            await self._reminder_access_changed(context, session, source_message=query.message)
            return
        if session.schedule_kind is ReminderScheduleKind.ONCE:
            scheduled_for = self._reminder_scheduled_for(session)
            if scheduled_for is None or scheduled_for <= self._reminder_now():
                await self._reminder_render_past(context, query, session)
                return
        try:
            result, recurring = await self._reminder_save_atomic(session)
        except _ReminderPastAtSave:
            await self._reminder_render_past(context, query, session)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Reminder confirm failed error_type=%s", type(exc).__name__)
            live = await self.reminder_sessions.get_exact(session)
            if live is not None:
                async with self._reminder_ui_lock:
                    await self._reminder_edit_text(
                        context,
                        live,
                        "Не удалось сохранить напоминание. Ничего не изменено — попробуй ещё раз.",
                        await self._reminder_retry_keyboard(live),
                        query=query,
                    )
            return
        if not result.ok or result.inbox_item is None:
            failed_access = await self._reminder_access(update)
            if (
                failed_access is None
                or failed_access.id != session.owner_id
                or failed_access.access_version != session.access_version
            ):
                await self._reminder_access_changed(
                    context,
                    session,
                    source_message=query.message,
                )
                return
            live = await self.reminder_sessions.get_exact(session)
            if live is not None:
                async with self._reminder_ui_lock:
                    await self._reminder_edit_text(
                        context,
                        live,
                        "Не удалось сохранить напоминание. Ничего не изменено — попробуй ещё раз.",
                        await self._reminder_retry_keyboard(live),
                        query=query,
                    )
            return
        final_user = await self._reminder_access(update)
        if (
            final_user is None
            or final_user.id != session.owner_id
            or final_user.access_version != session.access_version
        ):
            # The domain mutation itself is access-fenced. A change after commit
            # cannot be undone, but no stale private preview remains visible.
            await self._reminder_access_changed(context, session, source_message=query.message)
            return
        await self.reminder_sessions.clear(
            owner_id=session.owner_id,
            telegram_user_id=session.telegram_user_id,
            chat_id=session.chat_id,
            session_id=session.id,
        )
        if recurring is not None and recurring.schedule is not None:
            success = (
                "✅ Ежедневное напоминание включено\n\n"
                f"Что: {result.inbox_item.title}\n"
                f"Когда: каждый день в {recurring.schedule.local_time.strftime('%H:%M')}"
            )
        else:
            success = (
                "✅ Напоминание создано\n\n"
                f"Что: {result.inbox_item.title}\n"
                f"Когда: {self._reminder_once_label(session)}"
            )
        async with self._reminder_ui_lock:
            await self._reminder_edit_text(
                context,
                session,
                success,
                None,
                query=query,
            )

    async def _reminder_render_past(
        self,
        context: Any,
        query: Any,
        session: ReminderFlowSession,
    ) -> None:
        updated = await self.reminder_sessions.update(
            session,
            phase=ReminderFlowPhase.PAST,
        )
        if updated is not None:
            async with self._reminder_ui_lock:
                await self._reminder_edit_canonical(
                    context,
                    updated,
                    query=query,
                    source_message=query.message,
                )

    async def _reminder_save_atomic(
        self,
        session: ReminderFlowSession,
    ) -> tuple[DraftResult, RecurringScheduleMutation | None]:
        scheduled_for = self._reminder_scheduled_for(session)
        if session.title is None or session.local_time is None or scheduled_for is None:
            return DraftResult(False), None
        if (
            session.schedule_kind is ReminderScheduleKind.ONCE
            and scheduled_for <= self._reminder_now()
        ):
            raise _ReminderPastAtSave
        if session.schedule_kind is ReminderScheduleKind.ONCE:
            parsed = ParsedThought(
                kind="task",
                title=session.title,
                resolved_date=session.local_date,
                temporal_resolution=TemporalResolution(
                    resolved_at=scheduled_for,
                    remind_at=scheduled_for,
                    timezone=session.timezone,
                    resolved_local_date=session.local_date,
                    resolved_local_time=session.local_time,
                    precision="datetime",
                    original_expression="reminder_flow",
                    resolution_status="resolved",
                ),
            )
        else:
            parsed = ParsedThought(kind="task", title=session.title)

        async with self.db.session() as db_session:
            locked_owner_id = await db_session.scalar(
                update(User)
                .where(
                    User.id == session.owner_id,
                    User.telegram_id == session.telegram_user_id,
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                    User.access_version == session.access_version,
                )
                .values(updated_at=User.updated_at)
                .returning(User.id)
            )
            if locked_owner_id is None:
                return DraftResult(False), None
            if (
                session.schedule_kind is ReminderScheduleKind.ONCE
                and scheduled_for <= self._reminder_now()
            ):
                raise _ReminderPastAtSave
            draft = await self.draft_service.create_in_session(
                db_session,
                user_id=session.owner_id,
                telegram_user_id=session.telegram_user_id,
                chat_id=session.chat_id,
                source="reminder",
                raw_text=session.title,
                parsed=parsed,
            )
            result = await self.draft_service.confirm_in_session(
                db_session,
                draft.id,
                draft.version,
                session.telegram_user_id,
                session.chat_id,
                owner_locked=True,
                allow_saved_dedup=session.schedule_kind is ReminderScheduleKind.ONCE,
                return_existing=True,
                expected_access_version=session.access_version,
            )
            if not result.ok or result.inbox_item is None:
                return result, None
            recurring: RecurringScheduleMutation | None = None
            if session.schedule_kind is ReminderScheduleKind.DAILY:
                recurring = await self.recurring_reminder_service.create_daily_in_session(
                    db_session,
                    session.owner_id,
                    result.inbox_item.id,
                    session.local_time,
                    timezone=session.timezone,
                    timezone_source=session.timezone_source.value,
                )
        log_transition(
            draft.id,
            session.telegram_user_id,
            "preview",
            "confirmed",
            "save_daily" if recurring is not None else "save_reminder",
            inbox_created=not result.duplicate,
        )
        return result, recurring

    async def reminder_cancel_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        user = await self._reminder_access(update)
        if user is None:
            return False
        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            if current is None:
                return False
            async with self._reminder_ui_lock:
                live = await self.reminder_sessions.get_exact(current)
                if live is None:
                    return False
                await self.reminder_sessions.clear(
                    owner_id=live.owner_id,
                    telegram_user_id=live.telegram_user_id,
                    chat_id=live.chat_id,
                    session_id=live.id,
                )
                await self._reminder_edit_text(
                    context,
                    live,
                    "🔔 Напоминание отменено. Ничего не сохранено.",
                    None,
                    source_message=update.effective_message,
                )
        return True

    async def reminder_sync_access(
        self,
        user: Any,
        chat_id: int,
        *,
        context: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        async with self._reminder_launch_lock:
            current = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
            )
            if current is None:
                return
            if (
                not is_full_access_tier(user.access_tier)
                or current.access_version != user.access_version
            ):
                await self.reminder_sessions.clear(
                    owner_id=current.owner_id,
                    telegram_user_id=current.telegram_user_id,
                    chat_id=current.chat_id,
                    session_id=current.id,
                )
                if context is not None:
                    async with self._reminder_ui_lock:
                        await self._reminder_edit_text(
                            context,
                            current,
                            REMINDER_ACCESS_CHANGED_TEXT,
                            None,
                            source_message=source_message,
                        )

    async def reminder_clear_current(self, update: Update) -> None:
        user = await self._reminder_access(update)
        if user is None:
            return
        async with self._reminder_launch_lock:
            async with self._reminder_ui_lock:
                await self.reminder_sessions.clear(
                    owner_id=user.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                )

    async def _reminder_access(self, update: Update) -> User | None:
        telegram_user = update.effective_user
        chat = update.effective_chat
        if telegram_user is None or chat is None:
            return None
        try:
            status = await self.access_service.status(telegram_user.id)
            if status is None or not is_full_access_tier(status.access_tier):
                return None
            async with self.db.sessions() as session:
                return await session.scalar(
                    select(User).where(
                        User.telegram_id == telegram_user.id,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == status.access_version,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Reminder access lookup failed error_type=%s", type(exc).__name__)
            return None

    async def _reminder_access_changed(
        self,
        context: Any,
        session: ReminderFlowSession | None,
        *,
        source_message: Any | None,
    ) -> None:
        if session is None:
            return
        async with self._reminder_ui_lock:
            cleared = await self.reminder_sessions.clear(
                owner_id=session.owner_id,
                telegram_user_id=session.telegram_user_id,
                chat_id=session.chat_id,
                session_id=session.id,
            )
            if not cleared:
                return
            await self._reminder_edit_text(
                context,
                session,
                REMINDER_ACCESS_CHANGED_TEXT,
                None,
                source_message=source_message,
            )

    async def _reminder_screen(
        self,
        session: ReminderFlowSession,
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        phase = session.phase
        if phase in {ReminderFlowPhase.WHEN, ReminderFlowPhase.PAST}:
            tokens = await self.reminder_sessions.issue(
                session,
                ("today", "tomorrow", "choose_date", "daily", "cancel"),
            )
            prefix = (
                "Выбранное время сегодня уже прошло. Ничего не переношу автоматически.\n\n"
                if phase is ReminderFlowPhase.PAST
                else ""
            )
            return (
                f"{prefix}🔔 Когда напомнить?",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton("Сегодня", callback_data=f"rmd:{tokens['today']}"),
                            InlineKeyboardButton(
                                "Завтра", callback_data=f"rmd:{tokens['tomorrow']}"
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Выбрать дату",
                                callback_data=f"rmd:{tokens['choose_date']}",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "🔁 Каждый день",
                                callback_data=f"rmd:{tokens['daily']}",
                            )
                        ],
                        [InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}")],
                    ]
                ),
            )
        if phase is ReminderFlowPhase.DATE:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "📅 На какую дату напомнить?\n\nНапиши или скажи дату, например: 15 августа.",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.TIME:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "🕒 Во сколько напомнить?\n\n"
                "Напиши или скажи время, например: 19:30.\n"
                f"Использую твой часовой пояс: {self._reminder_timezone_label(session.timezone)}.",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.TITLE:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "📝 О чём напомнить?\n\n"
                "Напиши или скажи коротко, например:\n"
                "«заполнить дневник благодарностей».",
                self._cancel_keyboard(tokens["cancel"]),
            )
        if phase is ReminderFlowPhase.EDIT:
            tokens = await self.reminder_sessions.issue(
                session,
                ("edit_when", "edit_time", "edit_title", "cancel"),
            )
            return (
                "✏️ Что изменить?",
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("Когда", callback_data=f"rmd:{tokens['edit_when']}")],
                        [
                            InlineKeyboardButton(
                                "Время", callback_data=f"rmd:{tokens['edit_time']}"
                            ),
                            InlineKeyboardButton(
                                "Название", callback_data=f"rmd:{tokens['edit_title']}"
                            ),
                        ],
                        [InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}")],
                    ]
                ),
            )
        if phase is ReminderFlowPhase.INVALID:
            tokens = await self.reminder_sessions.issue(session, ("cancel",))
            return (
                "Не удалось безопасно понять дату, время или часовой пояс. "
                "Отмени карточку и повтори команду точнее.",
                self._cancel_keyboard(tokens["cancel"]),
            )

        tokens = await self.reminder_sessions.issue(session, ("confirm", "edit", "cancel"))
        title = session.title or ""
        if session.schedule_kind is ReminderScheduleKind.DAILY:
            first = self._reminder_scheduled_for(session)
            first_label = self._reminder_datetime_label(first, session.timezone)
            text_value = (
                "🔁 Проверь напоминание\n\n"
                f"Что: {title}\n"
                f"Когда: каждый день в {session.local_time.strftime('%H:%M')}\n"
                f"Первый раз: {first_label}"
            )
            confirm_label = "✅ Включить"
        else:
            text_value = (
                "🔔 Проверь напоминание\n\n"
                f"Что: {title}\n"
                f"Когда: {self._reminder_once_label(session)}"
            )
            confirm_label = "✅ Создать"
        return (
            text_value,
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            confirm_label,
                            callback_data=f"rmd:{tokens['confirm']}",
                        )
                    ],
                    [
                        InlineKeyboardButton("✏️ Изменить", callback_data=f"rmd:{tokens['edit']}"),
                        InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}"),
                    ],
                ]
            ),
        )

    async def _reminder_edit_canonical(
        self,
        context: Any,
        session: ReminderFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        text_value, markup = await self._reminder_screen(session)
        await self._reminder_edit_text(
            context,
            session,
            text_value,
            markup,
            query=query,
            source_message=source_message,
        )

    async def _reminder_edit_text(
        self,
        context: Any,
        session: ReminderFlowSession,
        text_value: str,
        markup: InlineKeyboardMarkup | None,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        try:
            if query is not None:
                await query.edit_message_text(text_value, reply_markup=markup)
                return
            if (
                source_message is not None
                and getattr(source_message, "message_id", None) == session.canonical_message_id
                and hasattr(source_message, "edit_text")
            ):
                await source_message.edit_text(text_value, reply_markup=markup)
                return
            if session.canonical_message_id is not None:
                await context.bot.edit_message_text(
                    chat_id=session.chat_id,
                    message_id=session.canonical_message_id,
                    text=text_value,
                    reply_markup=markup,
                )
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if "message is not modified" not in str(exc).casefold():
                logger.warning("Reminder canonical edit failed error_type=%s", type(exc).__name__)
        except TelegramError as exc:
            logger.warning("Reminder canonical edit failed error_type=%s", type(exc).__name__)

    @staticmethod
    async def _reminder_edit_access_candidate(candidate: Any | None) -> None:
        if candidate is None or not hasattr(candidate, "edit_text"):
            return
        try:
            await candidate.edit_text(REMINDER_ACCESS_CHANGED_TEXT, reply_markup=None)
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if "message is not modified" not in str(exc).casefold():
                logger.warning("Reminder access edit failed error_type=%s", type(exc).__name__)
        except TelegramError as exc:
            logger.warning("Reminder access edit failed error_type=%s", type(exc).__name__)

    async def _reminder_show_unsupported(
        self,
        update: Update,
        context: Any,
        current: ReminderFlowSession | None,
        candidate_message: Any | None,
    ) -> None:
        text_value = (
            "Пока поддерживаются только разовые и ежедневные напоминания. "
            "Еженедельные, будние и произвольные интервалы ещё недоступны."
        )
        if current is not None:
            async with self._reminder_ui_lock:
                live = await self.reminder_sessions.get_exact(current)
                if live is None:
                    await self._reminder_retire_voice_candidate(candidate_message)
                    return
                await self.reminder_sessions.clear(
                    owner_id=live.owner_id,
                    telegram_user_id=live.telegram_user_id,
                    chat_id=live.chat_id,
                    session_id=live.id,
                )
                await self._reminder_edit_text(
                    context,
                    live,
                    text_value,
                    None,
                    source_message=candidate_message or update.effective_message,
                )
                await self._reminder_retire_voice_candidate(candidate_message)
        elif candidate_message is not None and hasattr(candidate_message, "edit_text"):
            await candidate_message.edit_text(text_value)
        else:
            await update.effective_message.reply_text(text_value)

    async def _reminder_retire_voice_candidate(self, candidate: Any | None) -> None:
        if candidate is None:
            return
        try:
            if hasattr(candidate, "delete"):
                await candidate.delete()
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            logger.warning(
                "Reminder voice transient cleanup failed error_type=%s", type(exc).__name__
            )

    async def _reminder_retry_keyboard(
        self,
        session: ReminderFlowSession,
    ) -> InlineKeyboardMarkup | None:
        tokens = await self.reminder_sessions.issue(session, ("confirm", "cancel"))
        if not tokens:
            return None
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Попробовать снова", callback_data=f"rmd:{tokens['confirm']}"
                    )
                ],
                [InlineKeyboardButton("Отмена", callback_data=f"rmd:{tokens['cancel']}")],
            ]
        )

    def _reminder_phase(self, result: ReminderIntentResult) -> ReminderFlowPhase:
        if result.status is ReminderIntentStatus.INVALID:
            return (
                ReminderFlowPhase.PAST
                if result.error_code is ReminderIntentCode.PAST_ONCE
                else ReminderFlowPhase.INVALID
            )
        return self._reminder_fields_phase(
            title=result.title,
            schedule_kind=result.schedule_kind,
            local_date=result.local_date,
            local_time=result.local_time,
            timezone=result.timezone,
        )

    def _reminder_fields_phase(
        self,
        *,
        title: str | None,
        schedule_kind: ReminderScheduleKind | None,
        local_date: date | None,
        local_time: time | None,
        timezone: str | None,
    ) -> ReminderFlowPhase:
        if schedule_kind is None or (
            schedule_kind is ReminderScheduleKind.ONCE and local_date is None
        ):
            return ReminderFlowPhase.WHEN
        if local_time is None:
            return ReminderFlowPhase.TIME
        if title is None:
            return ReminderFlowPhase.TITLE
        if timezone is None:
            return ReminderFlowPhase.INVALID
        if schedule_kind is ReminderScheduleKind.ONCE:
            occurrence = calculate_daily_occurrence(local_date, local_time, timezone)
            if (
                occurrence.local_date != local_date
                or occurrence.scheduled_for <= self._reminder_now()
            ):
                return ReminderFlowPhase.PAST
        return ReminderFlowPhase.PREVIEW

    def _reminder_scheduled_for(self, session: ReminderFlowSession) -> datetime | None:
        if session.local_time is None:
            return None
        if session.schedule_kind is ReminderScheduleKind.DAILY:
            return first_daily_occurrence_utc(
                session.local_time,
                session.timezone,
                now=self._reminder_now(),
            )
        if session.local_date is None:
            return None
        occurrence = calculate_daily_occurrence(
            session.local_date,
            session.local_time,
            session.timezone,
        )
        return occurrence.scheduled_for if occurrence.local_date == session.local_date else None

    def _reminder_once_label(self, session: ReminderFlowSession) -> str:
        value = self._reminder_scheduled_for(session)
        return ReminderHandlers._reminder_datetime_label(value, session.timezone)

    def _reminder_now(self) -> datetime:
        current = self._reminder_now_provider()
        return current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)

    @staticmethod
    def _reminder_datetime_label(value: datetime | None, timezone: str) -> str:
        if value is None:
            return "не определено"
        return f"{value.astimezone(ZoneInfo(timezone)).strftime('%d.%m.%Y %H:%M')} ({timezone})"

    @staticmethod
    def _reminder_timezone_label(timezone: str) -> str:
        return {
            "Europe/Moscow": "Москва (МСК)",
            "Europe/Saratov": "Саратов",
        }.get(timezone, timezone)

    @staticmethod
    def _cancel_keyboard(token: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Отмена", callback_data=f"rmd:{token}")]]
        )

    @staticmethod
    def _reminder_expected_session_matches(
        current: ReminderFlowSession | None,
        expected: ReminderFlowSession | None,
    ) -> bool:
        if current is None or expected is None:
            return current is expected
        return bool(
            current.id == expected.id
            and current.version == expected.version
            and current.access_version == expected.access_version
            and current.canonical_message_id == expected.canonical_message_id
        )


__all__ = [
    "REMINDER_ACCESS_CHANGED_TEXT",
    "REMINDER_STALE_TEXT",
    "ReminderHandlers",
    "ReminderVoiceGateState",
]
