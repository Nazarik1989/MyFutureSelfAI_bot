from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .domain import TodayApplicationSnapshot, temporal_context
from .models import User
from .weekly_review import (
    WeeklyFocusSnapshot,
    WeeklyReminderCandidate,
    WeeklyReviewPhase,
    WeeklyReviewService,
    WeeklyReviewSessionSnapshot,
)
from .weekly_review_extraction import extract_weekly_review_input
from .weekly_review_flow import (
    WeeklyReviewDisposition,
    WeeklyReviewIntent,
    WeeklyReviewPolicy,
    WeeklyReviewScreen,
    classify_weekly_review_intent,
    reduce_weekly_review_input,
)

logger = logging.getLogger(__name__)

WEEKLY_REVIEW_ACCESS_CHANGED_TEXT = "🧭 Обзор недели\n\nДоступ изменился. Ничего не сохранено — открой /week после проверки доступа."
WEEKLY_REVIEW_STALE_TEXT = "Эта кнопка недельного обзора устарела. Открой /week."
WEEKLY_REVIEW_RECOVERY_TEXT = (
    "🧭 Обзор недели\n\nЭкран обзора обновился. "
    "Открой актуальный обзор — ничего не было сохранено повторно."
)
WEEKLY_REVIEW_RETRY_TEXT = (
    "Не удалось безопасно разобрать ответ. Ничего не сохранено — отправь его ещё раз."
)
WEEKLY_REVIEW_UNAVAILABLE_TEXT = "Недельный обзор сейчас недоступен."
WEEKLY_REVIEW_QUESTION = (
    "Что важно удерживать на ближайшей неделе?\n\n"
    "Можно ответить одним длинным сообщением или голосом — я выделю главный фокус, "
    "небольшие шаги и отдельно покажу возможные напоминания."
)
WEEKLY_REVIEW_NONANSWER_TEXT = (
    "Ничего страшного — можно начать с одного небольшого ориентира.\n\n" + WEEKLY_REVIEW_QUESTION
)

_ACTIVE_TEXT_PHASES = frozenset(
    {
        WeeklyReviewPhase.ROOT,
        WeeklyReviewPhase.AWAITING_INPUT,
        WeeklyReviewPhase.PROCESSING,
        WeeklyReviewPhase.PREVIEW,
        WeeklyReviewPhase.DELETE_PREVIEW,
    }
)


class _WeeklyReviewLookupError(RuntimeError):
    """Privacy-safe marker for a failed authoritative weekly-review lookup."""


class _WeeklyReviewVoiceLookupFailure:
    """Opaque fail-closed voice fence used when durable state cannot be read."""


_WEEKLY_REVIEW_VOICE_LOOKUP_FAILED = _WeeklyReviewVoiceLookupFailure()


class _WeeklyReviewCallbackAnswer:
    """Make one privacy-safe callback answer attempt."""

    def __init__(self, query: Any) -> None:
        self._query = query
        self.attempted = False

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        if self.attempted:
            return
        self.attempted = True
        try:
            await self._query.answer(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            logger.warning(
                "Weekly review callback failed operation=callback_answer error_type=%s",
                type(exc).__name__,
            )


@dataclass(frozen=True, slots=True)
class _WeeklyReviewActionPlan:
    kind: Literal["transition", "confirm", "candidate", "today", "confirm_delete", "unknown"]
    operation: str
    expected_phase: WeeklyReviewPhase | None = None
    phase: WeeklyReviewPhase | None = None
    reset_input: bool = False


class WeeklyReviewHandlers:
    weekly_review_service: WeeklyReviewService
    weekly_review_capabilities: Any
    weekly_review_policy: WeeklyReviewPolicy
    _weekly_review_launch_lock: asyncio.Lock
    _reply_keyboard_owner_lock: asyncio.Lock
    _weekly_review_tasks: set[asyncio.Task[Any]]
    _weekly_review_ui_locks: Any

    async def week_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        try:
            await self._weekly_review_week_command(update, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review command failed operation=command error_type=%s",
                type(exc).__name__,
            )

    async def _weekly_review_week_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._weekly_review_policy_allows_update(update):
            await self._weekly_review_reply_unavailable(update.effective_message)
            return
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await self._prompt_navigation_flow(update.effective_message, update, flow)
            return
        await self._weekly_review_open(update, context)

    async def weekly_review_active_text_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        try:
            return await self._weekly_review_active_text_lifecycle(update, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review text failed operation=active error_type=%s",
                type(exc).__name__,
            )
            return True

    async def _weekly_review_active_text_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        text = str(getattr(update.effective_message, "text", "") or "")
        intent = classify_weekly_review_intent(text)
        if not self.weekly_review_policy.enabled:
            if intent is not WeeklyReviewIntent.NONE:
                await self._weekly_review_reply_unavailable(update.effective_message)
                return True
            return False
        if (
            intent is not WeeklyReviewIntent.NONE
            and not await self._weekly_review_policy_allows_update(update)
        ):
            await self._weekly_review_reply_unavailable(update.effective_message)
            return True
        first_user: User | None = None
        first_current: WeeklyReviewSessionSnapshot | None = None
        final_user: User | None = None
        final_current: WeeklyReviewSessionSnapshot | None = None
        lookup_failed = False
        try:
            first_user, first_current = await self._weekly_review_route_snapshot(update)
        except _WeeklyReviewLookupError:
            lookup_failed = True
        try:
            final_user, final_current = await self._weekly_review_route_snapshot(update)
        except _WeeklyReviewLookupError:
            lookup_failed = True
        if lookup_failed:
            frozen = first_current or final_current
            if frozen is not None:
                await self._weekly_review_access_changed(
                    context,
                    frozen,
                    source_message=update.effective_message,
                )
            return True
        if first_user is None and final_user is None:
            return False
        if not self._weekly_review_same_user_generation(first_user, final_user):
            if first_current is not None:
                await self._weekly_review_access_changed(
                    context,
                    first_current,
                    source_message=update.effective_message,
                )
            elif final_user is not None:
                await self.weekly_review_sync_access(
                    final_user,
                    update.effective_chat.id,
                    context=context,
                    source_message=update.effective_message,
                )
            return True
        if not self._weekly_review_same_session(first_current, final_current):
            return True
        user = final_user
        current = final_current
        if user is None:
            return False
        if current is None or current.phase not in _ACTIVE_TEXT_PHASES:
            return False
        await self._weekly_review_reduce_active_input(
            update,
            context,
            current,
            text,
            source="text",
        )
        return True

    async def weekly_review_launch_text_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        text = str(getattr(update.effective_message, "text", "") or "")
        intent = classify_weekly_review_intent(text)
        if intent not in {
            WeeklyReviewIntent.OPEN,
            WeeklyReviewIntent.START,
            WeeklyReviewIntent.EDIT_FOCUS,
            WeeklyReviewIntent.VIEW,
        }:
            return False
        if not await self._weekly_review_policy_allows_update(update):
            await self._weekly_review_reply_unavailable(update.effective_message)
            return True
        try:
            if intent is WeeklyReviewIntent.VIEW:
                await self._weekly_review_show_focus(update, context)
            else:
                await self._weekly_review_open(
                    update,
                    context,
                    start_input=intent in {WeeklyReviewIntent.START, WeeklyReviewIntent.EDIT_FOCUS},
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review text failed operation=launch error_type=%s",
                type(exc).__name__,
            )
        return True

    async def weekly_review_stale_control_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        intent = classify_weekly_review_intent(
            str(getattr(update.effective_message, "text", "") or "")
        )
        if intent not in {
            WeeklyReviewIntent.BACK,
            WeeklyReviewIntent.SKIP,
            WeeklyReviewIntent.CANCEL,
        }:
            return False
        if not await self._weekly_review_policy_allows_update(update):
            await self._weekly_review_reply_unavailable(update.effective_message)
            return True
        user = await self._weekly_review_access(update)
        if user is None:
            return True
        current = await self._weekly_review_current(update, user)
        if current is not None and current.phase in _ACTIVE_TEXT_PHASES:
            return False
        await self._weekly_review_render_stale_control_root(
            update.effective_message,
            user.access_tier,
        )
        return True

    async def _weekly_review_render_stale_control_root(
        self,
        message: Any,
        tier: str,
    ) -> None:
        text = "Главное меню\n\nЧто хочешь сделать?"
        sent = await self._weekly_review_send_keyboard_cleanup(message, text, None)
        if sent is None:
            return
        try:
            await sent.edit_text(text, reply_markup=self._root_keyboard(tier))
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Weekly review keyboard cleanup failed operation=edit error_type=%s",
                type(exc).__name__,
            )

    async def _weekly_review_reduce_active_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: WeeklyReviewSessionSnapshot,
        text: str,
        *,
        source: Literal["text", "voice"],
    ) -> None:
        decision = reduce_weekly_review_input(session.phase, text)
        disposition = decision.disposition
        if disposition in {WeeklyReviewDisposition.ABSORB, WeeklyReviewDisposition.PASS}:
            return
        if disposition is WeeklyReviewDisposition.RENDER_CURRENT:
            await self._weekly_review_render_current(context, session)
            return
        if disposition is WeeklyReviewDisposition.SHOW_FOCUS:
            await self._weekly_review_render_focus_in_session(context, session)
            return
        if disposition is WeeklyReviewDisposition.CANCEL:
            await self._weekly_review_complete_session(
                context,
                session,
                source_message=update.effective_message,
            )
            return
        if disposition is WeeklyReviewDisposition.RETURN_ROOT:
            root = await self._weekly_review_to_root(session)
            if root is not None:
                await self._weekly_review_render_current(context, root)
            return
        if disposition is WeeklyReviewDisposition.REPROMPT:
            await self._weekly_review_render(
                context,
                session,
                f"{self._weekly_review_week_heading(session)}\n\n{WEEKLY_REVIEW_NONANSWER_TEXT}",
                await self._weekly_review_cancel_markup(session),
            )
            return
        if disposition is WeeklyReviewDisposition.PROMPT_INPUT:
            awaiting = await self._weekly_review_to_input(session)
            if awaiting is not None:
                await self._weekly_review_render(
                    context,
                    awaiting,
                    self._weekly_review_question_text(awaiting),
                    await self._weekly_review_cancel_markup(awaiting),
                )
            return
        if disposition is not WeeklyReviewDisposition.EXTRACT:
            return
        awaiting = session
        if session.phase is WeeklyReviewPhase.ROOT:
            transitioned = await self._weekly_review_to_input(session)
            if transitioned is None:
                return
            awaiting = transitioned
        if awaiting.phase is not WeeklyReviewPhase.AWAITING_INPUT:
            return
        await self._weekly_review_process_input(
            update,
            context,
            awaiting,
            text,
            source=source,
        )

    async def _weekly_review_render_focus_in_session(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        lookup = await self.weekly_review_service.get_focus(
            telegram_actor_id=session.telegram_user_id,
            expected_access_version=session.access_version,
            week_start=session.week_start,
        )
        if lookup.status in {"access_denied", "access_changed"}:
            await self._weekly_review_access_changed(context, session)
            return
        if lookup.status == "found" and lookup.focus is not None:
            text = self._weekly_review_focus_text(lookup.focus)
        else:
            week = self.weekly_review_service.week_range(session.week_start)
            text = (
                f"🧭 Неделя: {self._weekly_review_date_range(week.start, week.end)}\n\n"
                "Подтверждённого фокуса пока нет."
            )
        if session.phase is WeeklyReviewPhase.ROOT:
            user = await self._weekly_review_access_values(
                session.telegram_user_id,
                session.chat_id,
                fail_closed=True,
            )
            markup = (
                await self._weekly_review_root_markup(session, user) if user is not None else None
            )
        else:
            markup = await self._weekly_review_cancel_markup(session)
            text += "\n\nМожно продолжить ответом на вопрос обзора."
        await self._weekly_review_render(context, session, text, markup)

    async def _weekly_review_to_root(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> WeeklyReviewSessionSnapshot | None:
        if session.canonical_message_id is None:
            return None
        result = await self.weekly_review_service.transition_session(
            telegram_actor_id=session.telegram_user_id,
            chat_id=session.chat_id,
            expected_access_version=session.access_version,
            session_public_id=session.public_id,
            expected_session_version=session.version,
            expected_canonical_message_id=session.canonical_message_id,
            phase=WeeklyReviewPhase.ROOT,
            focus=None,
            approach=None,
            small_steps=(),
            reminder_candidates=(),
            source=None,
        )
        return result.session if result.status == "updated" else None

    async def _weekly_review_complete_session(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        session: WeeklyReviewSessionSnapshot,
        *,
        source_message: Any | None = None,
    ) -> None:
        completed = await self.weekly_review_service.transition_session(
            telegram_actor_id=session.telegram_user_id,
            chat_id=session.chat_id,
            expected_access_version=session.access_version,
            session_public_id=session.public_id,
            expected_session_version=session.version,
            expected_canonical_message_id=session.canonical_message_id,
            phase=WeeklyReviewPhase.COMPLETED,
        )
        if completed.status != "updated" or completed.session is None:
            return
        await self._weekly_review_render(
            context,
            completed.session,
            f"{self._weekly_review_week_heading(completed.session)}\n\n"
            "Обзор недели завершён. Ничего больше не изменяю.",
            None,
            source_message=source_message,
        )

    async def weekly_review_voice_fence(
        self,
        update: Update,
        *,
        user: User,
    ) -> WeeklyReviewSessionSnapshot | _WeeklyReviewVoiceLookupFailure | None:
        if not self.weekly_review_policy.allows_actor(user):
            return None
        try:
            return await self._weekly_review_current(update, user, fail_closed=True)
        except _WeeklyReviewLookupError:
            return _WEEKLY_REVIEW_VOICE_LOOKUP_FAILED

    @staticmethod
    def weekly_review_voice_lookup_failed(value: Any | None) -> bool:
        return value is _WEEKLY_REVIEW_VOICE_LOOKUP_FAILED

    def weekly_review_schedule_voice_cancel_cleanup(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        progress: Any,
        *,
        expected_user: User,
        expected_session: WeeklyReviewSessionSnapshot | _WeeklyReviewVoiceLookupFailure | None,
    ) -> None:
        lifecycle = self._weekly_review_voice_cancel_cleanup_lifecycle(
            update,
            context,
            progress,
            expected_user=expected_user,
            expected_session=expected_session,
        )
        try:
            task = asyncio.create_task(
                lifecycle,
                name="weekly-review-voice-cancel-cleanup-lifecycle",
            )
        except Exception as exc:
            lifecycle.close()
            logger.warning(
                "Weekly review voice cleanup failed operation=schedule error_type=%s",
                type(exc).__name__,
            )
            return
        self._weekly_review_track_task(task)

    async def _weekly_review_voice_cancel_cleanup_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        progress: Any,
        *,
        expected_user: User,
        expected_session: WeeklyReviewSessionSnapshot | _WeeklyReviewVoiceLookupFailure | None,
    ) -> None:
        await self._weekly_review_retire_transient(progress)
        await self._weekly_review_voice_pre_route_lifecycle(
            update,
            context,
            None,
            expected_user=expected_user,
            expected_session=expected_session,
        )

    async def weekly_review_voice_pre_route(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        progress: Any,
        *,
        expected_user: User,
        expected_session: WeeklyReviewSessionSnapshot | _WeeklyReviewVoiceLookupFailure | None,
    ) -> bool:
        task = asyncio.create_task(
            self._weekly_review_voice_pre_route_lifecycle(
                update,
                context,
                progress,
                expected_user=expected_user,
                expected_session=expected_session,
            ),
            name="weekly-review-voice-pre-route-lifecycle",
        )
        self._weekly_review_track_task(task)
        return await asyncio.shield(task)

    async def _weekly_review_voice_pre_route_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        progress: Any,
        *,
        expected_user: User,
        expected_session: WeeklyReviewSessionSnapshot | _WeeklyReviewVoiceLookupFailure | None,
    ) -> bool:
        if self.weekly_review_voice_lookup_failed(expected_session):
            await self._weekly_review_retire_transient(progress)
            return True
        if expected_session is None:
            return False
        try:
            first, current = await self._weekly_review_route_snapshot(update)
            final, final_current = await self._weekly_review_route_snapshot(update)
        except _WeeklyReviewLookupError:
            await self._weekly_review_retire_transient(progress)
            if expected_session is not None:
                await self._weekly_review_access_changed(
                    context,
                    expected_session,
                    source_message=update.effective_message,
                )
            return True
        access_ok = (
            first is not None
            and final is not None
            and first.id == expected_user.id
            and final.id == expected_user.id
            and first.access_version == expected_user.access_version
            and final.access_version == expected_user.access_version
        )
        sessions_ok = self._weekly_review_same_session(
            current, expected_session
        ) and self._weekly_review_same_session(final_current, expected_session)
        if access_ok and sessions_ok:
            return False
        await self._weekly_review_retire_transient(progress)
        if not access_ok and expected_session is not None:
            await self._weekly_review_access_changed(
                context,
                expected_session,
                source_message=update.effective_message,
            )
        return True

    async def weekly_review_voice_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        transcript: str,
        progress: Any,
        *,
        expected_user: User,
        expected_session: WeeklyReviewSessionSnapshot | _WeeklyReviewVoiceLookupFailure | None,
    ) -> bool:
        if self.weekly_review_voice_lookup_failed(expected_session):
            await self._weekly_review_retire_transient(progress)
            return True
        if expected_session is not None and expected_session.phase in _ACTIVE_TEXT_PHASES:
            try:
                async with self._weekly_review_launch_lock:
                    first_user, first_current = await self._weekly_review_route_snapshot(update)
                    final_user, final_current = await self._weekly_review_route_snapshot(update)
            except _WeeklyReviewLookupError:
                await self._weekly_review_retire_transient(progress)
                await self._weekly_review_access_changed(
                    context,
                    expected_session,
                    source_message=update.effective_message,
                )
                return True
            access_ok = self._weekly_review_same_user_generation(
                first_user, expected_user
            ) and self._weekly_review_same_user_generation(final_user, expected_user)
            sessions_ok = self._weekly_review_same_session(
                first_current, expected_session
            ) and self._weekly_review_same_session(final_current, expected_session)
            if not access_ok or not sessions_ok:
                await self._weekly_review_retire_transient(progress)
                if not access_ok:
                    await self._weekly_review_access_changed(
                        context,
                        expected_session,
                        source_message=update.effective_message,
                    )
                return True
            assert final_current is not None
            current = final_current
            progress_retired = False
            if current.phase in {
                WeeklyReviewPhase.PREVIEW,
                WeeklyReviewPhase.DELETE_PREVIEW,
            }:
                await self._weekly_review_retire_transient(progress)
                progress_retired = True
            try:
                await self._weekly_review_reduce_active_input(
                    update,
                    context,
                    current,
                    transcript,
                    source="voice",
                )
            finally:
                if not progress_retired:
                    await self._weekly_review_retire_transient(progress)
            return True
        return False

    async def weekly_review_voice_failure(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        expected_user: User,
        expected_session: WeeklyReviewSessionSnapshot | _WeeklyReviewVoiceLookupFailure | None,
        progress: Any | None,
        notice: str,
    ) -> bool:
        if self.weekly_review_voice_lookup_failed(expected_session):
            await self._weekly_review_retire_transient(progress)
            return True
        if expected_session is None or expected_session.phase not in _ACTIVE_TEXT_PHASES:
            return False
        if await self.weekly_review_voice_pre_route(
            update,
            context,
            progress,
            expected_user=expected_user,
            expected_session=expected_session,
        ):
            return True
        await self._weekly_review_retire_transient(progress)
        live = await self._weekly_review_exact(expected_session)
        if live is None:
            return True
        await self._weekly_review_render(
            context,
            live,
            f"{self._weekly_review_week_heading(live)}\n\n{notice}",
            await self._weekly_review_cancel_markup(live),
            source_message=update.effective_message,
        )
        return True

    async def weekly_review_launch_voice_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        transcript: str,
        progress: Any,
        *,
        expected_user: User,
    ) -> bool:
        intent = classify_weekly_review_intent(transcript)
        if intent is WeeklyReviewIntent.NONE:
            return False
        if not self.weekly_review_policy.allows_actor(expected_user):
            await self._weekly_review_retire_transient(progress)
            await self._weekly_review_reply_unavailable(update.effective_message)
            return True
        try:
            first = await self._weekly_review_access(update, fail_closed=True)
            final = await self._weekly_review_access(update, fail_closed=True)
        except _WeeklyReviewLookupError:
            await self._weekly_review_retire_transient(progress)
            return True
        if not self._weekly_review_same_user_generation(
            first, expected_user
        ) or not self._weekly_review_same_user_generation(final, expected_user):
            await self._weekly_review_retire_transient(progress)
            return True
        if intent in {
            WeeklyReviewIntent.BACK,
            WeeklyReviewIntent.SKIP,
            WeeklyReviewIntent.CANCEL,
        }:
            await self._weekly_review_retire_transient(progress)
            assert final is not None
            await self._weekly_review_render_stale_control_root(
                update.effective_message,
                final.access_tier,
            )
            return True
        if intent is WeeklyReviewIntent.VIEW:
            await self._weekly_review_show_focus(
                update,
                context,
                source_message=progress,
                expected_user=expected_user,
            )
        else:
            await self._weekly_review_open(
                update,
                context,
                source_message=progress,
                expected_user=expected_user,
                start_input=intent in {WeeklyReviewIntent.START, WeeklyReviewIntent.EDIT_FOCUS},
            )
        return True

    async def weekly_review_clear_current(self, update: Update) -> bool:
        if not self.weekly_review_policy.enabled:
            return False
        user = await self._weekly_review_access(update)
        if user is None:
            return False
        current = await self._weekly_review_current(update, user)
        if current is None:
            return False
        try:
            cleared = await self._weekly_review_clear_exact(current)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review clear failed operation=clear error_type=%s",
                type(exc).__name__,
            )
            return False
        if cleared:
            await self.weekly_review_capabilities.revoke_session(current.public_id)
        return cleared

    async def weekly_review_cancel_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        if not self.weekly_review_policy.enabled:
            return False
        user = await self._weekly_review_access(update)
        if user is None:
            return False
        current = await self._weekly_review_current(update, user)
        if current is None or current.phase in {
            WeeklyReviewPhase.COMPLETED,
            WeeklyReviewPhase.REMINDER_HANDOFF,
        }:
            return False
        try:
            completed = await self.weekly_review_service.transition_session(
                telegram_actor_id=current.telegram_user_id,
                chat_id=current.chat_id,
                expected_access_version=current.access_version,
                session_public_id=current.public_id,
                expected_session_version=current.version,
                expected_canonical_message_id=current.canonical_message_id,
                phase=WeeklyReviewPhase.COMPLETED,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review cancel failed operation=cancel error_type=%s",
                type(exc).__name__,
            )
            return True
        if completed.status != "updated" or completed.session is None:
            return True
        await self._weekly_review_render(
            context,
            completed.session,
            f"{self._weekly_review_week_heading(completed.session)}\n\n"
            "Обзор недели завершён. Ничего больше не изменяю.",
            None,
            source_message=update.effective_message,
        )
        return True

    async def _weekly_review_process_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: WeeklyReviewSessionSnapshot,
        text: str,
        *,
        source: Literal["text", "voice"],
    ) -> None:
        task = asyncio.create_task(
            self._weekly_review_process_input_lifecycle(
                update,
                context,
                session,
                text,
                source=source,
            ),
            name="weekly-review-extraction-lifecycle",
        )
        self._weekly_review_track_task(task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review input failed operation=input error_type=%s",
                type(exc).__name__,
            )

    async def _weekly_review_process_input_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: WeeklyReviewSessionSnapshot,
        text: str,
        *,
        source: Literal["text", "voice"],
    ) -> None:
        if session.canonical_message_id is None:
            return
        try:
            processing = await self.weekly_review_service.mark_processing(
                telegram_actor_id=session.telegram_user_id,
                chat_id=session.chat_id,
                expected_access_version=session.access_version,
                session_public_id=session.public_id,
                expected_session_version=session.version,
                canonical_message_id=session.canonical_message_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review input failed operation=mark_processing error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_access_changed(
                context,
                session,
                source_message=update.effective_message,
            )
            return
        if processing.status != "updated" or processing.session is None:
            await self._weekly_review_handle_stale_input_cas(update, context, session)
            return
        frozen = processing.session
        provider_user = await self._weekly_review_processing_fence(
            update,
            context,
            frozen,
            operation="pre_extract",
        )
        if provider_user is None:
            return
        try:
            extraction = await extract_weekly_review_input(
                self.ai,
                text,
                {
                    **temporal_context(provider_user.timezone),
                    "week_start": frozen.week_start.isoformat(),
                    "week_end": (frozen.week_start + timedelta(days=6)).isoformat(),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review extraction failed operation=extract error_type=%s",
                type(exc).__name__,
            )
            if (
                await self._weekly_review_processing_fence(
                    update,
                    context,
                    frozen,
                    operation="retry_fence",
                )
                is None
            ):
                return
            try:
                retry = await self.weekly_review_service.transition_session(
                    telegram_actor_id=frozen.telegram_user_id,
                    chat_id=frozen.chat_id,
                    expected_access_version=frozen.access_version,
                    session_public_id=frozen.public_id,
                    expected_session_version=frozen.version,
                    expected_canonical_message_id=frozen.canonical_message_id,
                    expected_phase=WeeklyReviewPhase.PROCESSING,
                    phase=WeeklyReviewPhase.AWAITING_INPUT,
                )
            except asyncio.CancelledError:
                raise
            except Exception as transition_exc:
                logger.warning(
                    "Weekly review input failed operation=retry error_type=%s",
                    type(transition_exc).__name__,
                )
                await self._weekly_review_access_changed(
                    context,
                    frozen,
                    source_message=update.effective_message,
                )
                return
            if retry.session is None:
                await self._weekly_review_handle_stale_input_cas(update, context, frozen)
                return
            await self._weekly_review_render(
                context,
                retry.session,
                f"{self._weekly_review_week_heading(retry.session)}\n\n{WEEKLY_REVIEW_RETRY_TEXT}",
                await self._weekly_review_cancel_markup(retry.session),
            )
            return
        if (
            await self._weekly_review_processing_fence(
                update,
                context,
                frozen,
                operation="post_extract",
            )
            is None
        ):
            return
        try:
            stored = await self.weekly_review_service.store_extraction(
                telegram_actor_id=frozen.telegram_user_id,
                chat_id=frozen.chat_id,
                expected_access_version=frozen.access_version,
                session_public_id=frozen.public_id,
                expected_session_version=frozen.version,
                canonical_message_id=frozen.canonical_message_id,
                focus=extraction.focus,
                approach=extraction.approach,
                small_steps=extraction.small_steps,
                reminder_candidates=tuple(
                    WeeklyReminderCandidate(candidate.title, candidate.schedule_wording)
                    for candidate in extraction.reminder_candidates
                ),
                source=source,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review input failed operation=store error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_access_changed(
                context,
                frozen,
                source_message=update.effective_message,
            )
            return
        if stored.status == "updated" and stored.session is not None:
            await self._weekly_review_render_current(context, stored.session)
            return
        await self._weekly_review_handle_stale_input_cas(update, context, frozen)

    async def _weekly_review_handle_stale_input_cas(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        frozen: WeeklyReviewSessionSnapshot,
    ) -> None:
        """A failed generation CAS is not proof that access changed."""

        try:
            first_user = await self._weekly_review_access(update, fail_closed=True)
            final_user = await self._weekly_review_access(update, fail_closed=True)
        except _WeeklyReviewLookupError:
            await self._weekly_review_access_changed(
                context,
                frozen,
                source_message=update.effective_message,
            )
            return
        if not self._weekly_review_access_matches(
            frozen, first_user
        ) or not self._weekly_review_access_matches(frozen, final_user):
            await self._weekly_review_access_changed(
                context,
                frozen,
                source_message=update.effective_message,
            )
            return
        current = await self._weekly_review_current(update, final_user, fail_closed=True)
        if current is not None and current.canonical_message_id == frozen.canonical_message_id:
            if current.phase is not WeeklyReviewPhase.PROCESSING:
                await self._weekly_review_render_current(context, current)
            return
        if frozen.canonical_message_id is not None:
            await self._weekly_review_edit_unbound(
                context,
                chat_id=frozen.chat_id,
                message_id=frozen.canonical_message_id,
                text=WEEKLY_REVIEW_STALE_TEXT,
                markup=None,
                source_message=update.effective_message,
            )

    async def _weekly_review_processing_fence(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        frozen: WeeklyReviewSessionSnapshot,
        *,
        operation: str,
    ) -> User | None:
        try:
            first_user = await self._weekly_review_access(update, fail_closed=True)
            first_exact = await self._weekly_review_exact(frozen)
            final_user = await self._weekly_review_access(update, fail_closed=True)
            final_exact = await self._weekly_review_exact(frozen)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review input failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            await self._weekly_review_access_changed(
                context,
                frozen,
                source_message=update.effective_message,
            )
            return None
        if not self._weekly_review_access_matches(
            frozen, first_user
        ) or not self._weekly_review_access_matches(frozen, final_user):
            await self._weekly_review_access_changed(
                context,
                frozen,
                source_message=update.effective_message,
            )
            return None
        if first_exact is None or final_exact is None:
            await self._weekly_review_handle_stale_input_cas(update, context, frozen)
            return None
        return final_user

    async def _weekly_review_open(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        source_message: Any | None = None,
        expected_user: User | None = None,
        start_input: bool = False,
    ) -> None:
        task = asyncio.create_task(
            self._weekly_review_open_owned_lifecycle(
                update,
                context,
                source_message=source_message,
                expected_user=expected_user,
                start_input=start_input,
            ),
            name="weekly-review-open-lifecycle",
        )
        self._weekly_review_track_task(task)
        await asyncio.shield(task)

    async def _weekly_review_open_owned_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        source_message: Any | None,
        expected_user: User | None,
        start_input: bool,
    ) -> None:
        async with self._reply_keyboard_owner_lock:
            await self._weekly_review_open_lifecycle(
                update,
                context,
                source_message=source_message,
                expected_user=expected_user,
                start_input=start_input,
            )

    async def _weekly_review_open_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        source_message: Any | None = None,
        expected_user: User | None = None,
        start_input: bool = False,
    ) -> None:
        try:
            user = await self._weekly_review_access(update, fail_closed=True)
        except _WeeklyReviewLookupError:
            return
        if user is None or (
            expected_user is not None
            and not self._weekly_review_same_user_generation(user, expected_user)
        ):
            return
        if await self._weekly_review_reply_keyboard_owned(user):
            if source_message is not None and source_message is not update.effective_message:
                await self._weekly_review_retire_transient(source_message)
            return
        async with self._weekly_review_launch_lock:
            try:
                current = await self._weekly_review_current(update, user, fail_closed=True)
            except _WeeklyReviewLookupError:
                return
            if (
                current is not None
                and current.phase is not WeeklyReviewPhase.COMPLETED
                and current.canonical_message_id is not None
            ):
                # A bound session owns its existing canonical even when
                # Telegram rejects this particular edit.  Never fall back to a
                # second send/replacement screen.
                if start_input and current.phase is WeeklyReviewPhase.ROOT:
                    awaiting = await self._weekly_review_to_input(current)
                    if awaiting is not None:
                        await self._weekly_review_render(
                            context,
                            awaiting,
                            self._weekly_review_question_text(awaiting),
                            await self._weekly_review_cancel_markup(awaiting),
                        )
                elif start_input and current.phase is WeeklyReviewPhase.AWAITING_INPUT:
                    await self._weekly_review_render(
                        context,
                        current,
                        self._weekly_review_question_text(current),
                        await self._weekly_review_cancel_markup(current),
                    )
                else:
                    await self._weekly_review_render_current(context, current)
                return
            created = await self.weekly_review_service.create_session(
                telegram_actor_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                expected_access_version=user.access_version,
                phase=(WeeklyReviewPhase.AWAITING_INPUT if start_input else WeeklyReviewPhase.ROOT),
            )
            if created.status != "created" or created.session is None:
                return
            session = created.session
            try:
                text = (
                    self._weekly_review_question_text(session)
                    if start_input
                    else await self._weekly_review_root_text(session, user)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Weekly review launch failed operation=root_snapshot error_type=%s",
                    type(exc).__name__,
                )
                await self._weekly_review_clear_exact_safely(session)
                return
            if text is None:
                await self._weekly_review_clear_exact_safely(session)
                return
            if await self._weekly_review_pre_send_session_fence(update, session) is None:
                return
            message = await self._weekly_review_launch_message(update, source_message)
            if message is None:
                await self._weekly_review_clear_exact_safely(session)
                return
            sent = await self._weekly_review_send_keyboard_cleanup(message, text, None)
            message_id = getattr(sent, "message_id", None)
            if not isinstance(message_id, int) or message_id <= 0:
                await self._weekly_review_clear_exact(session)
                return
            try:
                bound = await self.weekly_review_service.bind_canonical(
                    telegram_actor_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    expected_access_version=session.access_version,
                    session_public_id=session.public_id,
                    expected_session_version=session.version,
                    canonical_message_id=message_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Weekly review launch failed operation=bind_canonical error_type=%s",
                    type(exc).__name__,
                )
                await self._weekly_review_clear_exact_safely(session)
                await self._weekly_review_edit_unbound(
                    context,
                    chat_id=session.chat_id,
                    message_id=message_id,
                    text=WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    markup=None,
                    source_message=sent,
                )
                return
            if bound.status != "updated" or bound.session is None:
                await self._weekly_review_clear_exact_safely(session)
                await self._weekly_review_edit_unbound(
                    context,
                    chat_id=session.chat_id,
                    message_id=message_id,
                    text=WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    markup=None,
                    source_message=sent,
                )
                return
            fresh = await self._weekly_review_access(update)
            if (
                fresh is None
                or fresh.id != bound.session.owner_id
                or fresh.access_version != bound.session.access_version
            ):
                await self._weekly_review_access_changed(
                    context,
                    bound.session,
                    source_message=sent,
                )
                return
            await self._weekly_review_render_current(
                context,
                bound.session,
                source_message=sent,
            )

    async def _weekly_review_show_focus(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        source_message: Any | None = None,
        expected_user: User | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._weekly_review_show_focus_owned_lifecycle(
                update,
                context,
                source_message=source_message,
                expected_user=expected_user,
            ),
            name="weekly-review-show-focus-lifecycle",
        )
        self._weekly_review_track_task(task)
        await asyncio.shield(task)

    async def _weekly_review_show_focus_owned_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        source_message: Any | None,
        expected_user: User | None,
    ) -> None:
        async with self._reply_keyboard_owner_lock:
            await self._weekly_review_show_focus_lifecycle(
                update,
                context,
                source_message=source_message,
                expected_user=expected_user,
            )

    async def _weekly_review_show_focus_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        source_message: Any | None = None,
        expected_user: User | None = None,
    ) -> None:
        try:
            user = await self._weekly_review_access(update, fail_closed=True)
        except _WeeklyReviewLookupError:
            return
        if user is None or (
            expected_user is not None
            and not self._weekly_review_same_user_generation(user, expected_user)
        ):
            return
        if await self._weekly_review_reply_keyboard_owned(user):
            if source_message is not None and source_message is not update.effective_message:
                await self._weekly_review_retire_transient(source_message)
            return
        week_start = self.weekly_review_service.target_week_start(
            user.timezone,
            review_weekday=self.settings.weekly_review_weekday,
        )
        lookup = await self.weekly_review_service.get_focus(
            telegram_actor_id=user.telegram_id,
            expected_access_version=user.access_version,
            week_start=week_start,
        )
        if lookup.status in {"access_denied", "access_changed"}:
            return
        try:
            pre_send_first = await self._weekly_review_access(update, fail_closed=True)
            pre_send_final = await self._weekly_review_access(update, fail_closed=True)
        except _WeeklyReviewLookupError:
            return
        if not self._weekly_review_same_user_generation(
            pre_send_first,
            user,
        ) or not self._weekly_review_same_user_generation(pre_send_final, user):
            return
        assert pre_send_final is not None
        if (
            self.weekly_review_service.target_week_start(
                pre_send_final.timezone,
                review_weekday=self.settings.weekly_review_weekday,
            )
            != week_start
        ):
            return
        user = pre_send_final
        week = self.weekly_review_service.week_range(week_start)
        if lookup.status == "found" and lookup.focus is not None:
            text = self._weekly_review_focus_text(lookup.focus)
        else:
            text = (
                f"🧭 Неделя: {self._weekly_review_date_range(week.start, week.end)}\n\n"
                "Подтверждённого фокуса пока нет."
            )
        message = await self._weekly_review_launch_message(update, source_message)
        if message is None:
            return
        sent = await self._weekly_review_send_keyboard_cleanup(message, text, None)
        message_id = getattr(sent, "message_id", None)
        if not isinstance(message_id, int) or message_id <= 0:
            return
        after_send = await self._weekly_review_access(update)
        if not self._weekly_review_same_user_generation(after_send, user) or (
            after_send is not None
            and self.weekly_review_service.target_week_start(
                after_send.timezone,
                review_weekday=self.settings.weekly_review_weekday,
            )
            != week_start
        ):
            await self._weekly_review_neutralize_sent(
                context.bot,
                update.effective_chat.id,
                message_id,
            )
            return
        tokens = await self.weekly_review_capabilities.issue(
            actions=("start", "close"),
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=update.effective_chat.id,
            canonical_message_id=message_id,
            access_version=user.access_version,
            week_start=week_start,
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✏️ Изменить фокус" if lookup.focus is not None else "🧭 Начать обзор",
                        callback_data=f"wrev:{tokens['start']}",
                    )
                ],
                [InlineKeyboardButton("Закрыть", callback_data=f"wrev:{tokens['close']}")],
            ]
        )
        async with self._weekly_review_ui_lock(update.effective_chat.id, message_id):
            before_edit = await self._weekly_review_access(update)
            if not self._weekly_review_same_user_generation(before_edit, user) or (
                before_edit is not None
                and self.weekly_review_service.target_week_start(
                    before_edit.timezone,
                    review_weekday=self.settings.weekly_review_weekday,
                )
                != week_start
            ):
                await self.weekly_review_capabilities.revoke_tokens(tuple(tokens.values()))
                await self._weekly_review_neutralize_sent(
                    context.bot,
                    update.effective_chat.id,
                    message_id,
                )
                return
            edited = await self._weekly_review_edit_unbound_locked(
                context,
                chat_id=update.effective_chat.id,
                message_id=message_id,
                text=text,
                markup=markup,
                source_message=sent,
            )
            if not edited:
                await self.weekly_review_capabilities.revoke_tokens(tuple(tokens.values()))
                return
            after_edit = await self._weekly_review_access(update)
            if self._weekly_review_same_user_generation(after_edit, user) and (
                after_edit is not None
                and self.weekly_review_service.target_week_start(
                    after_edit.timezone,
                    review_weekday=self.settings.weekly_review_weekday,
                )
                == week_start
            ):
                return
            await self.weekly_review_capabilities.revoke_tokens(tuple(tokens.values()))
            await self._weekly_review_neutralize_sent(
                context.bot,
                update.effective_chat.id,
                message_id,
            )

    async def weekly_review_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        task = asyncio.create_task(
            self._weekly_review_callback_lifecycle(update, context),
            name="weekly-review-callback-lifecycle",
        )
        self._weekly_review_track_task(task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review callback failed operation=callback error_type=%s",
                type(exc).__name__,
            )

    async def _weekly_review_callback_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        answer = _WeeklyReviewCallbackAnswer(query)
        data = str(query.data or "")
        token = data.removeprefix("wrev:") if data.startswith("wrev:") else ""
        if not self.weekly_review_policy.enabled:
            await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
            return
        try:
            claim = await self.weekly_review_capabilities.peek(
                token,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                canonical_message_id=getattr(query.message, "message_id", None),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review callback failed operation=capability_lookup error_type=%s",
                type(exc).__name__,
            )
            await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
            return
        if claim is None:
            await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
            return
        try:
            user = await self._weekly_review_access(update, fail_closed=True)
        except _WeeklyReviewLookupError:
            await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
            await self._weekly_review_clear_claim_safely(claim)
            await self._weekly_review_edit_query(
                query,
                WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                None,
                operation="access_lookup",
            )
            return
        if (
            user is None
            or user.id != claim.owner_id
            or user.access_version != claim.access_version
            or user.telegram_id != claim.telegram_user_id
        ):
            await answer.answer()
            await self._weekly_review_clear_claim_safely(claim)
            await self._weekly_review_edit_query(
                query,
                WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                None,
                operation="access_changed",
            )
            await self._weekly_review_consume_claim_safely(claim)
            return
        try:
            flow = await self._active_navigation_flow(update, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review callback failed operation=flow_lookup error_type=%s",
                type(exc).__name__,
            )
            await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
            await self._weekly_review_clear_claim_safely(claim)
            await self._weekly_review_edit_query(
                query,
                WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                None,
                operation="flow_lookup",
            )
            return
        if flow is not None:
            await answer.answer()
            await self._prompt_navigation_flow(query.message, update, flow, query=query)
            return
        if claim.session_public_id is None:
            try:
                memory_owns = await self.nova_memory_blocks_navigation(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Weekly review callback failed operation=owner_lookup error_type=%s",
                    type(exc).__name__,
                )
                await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
                await self._weekly_review_edit_query(
                    query,
                    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    None,
                    operation="owner_lookup",
                )
                return
            if memory_owns:
                await answer.answer()
                await self.nova_memory_public_command_gate(update, context)
                return
            try:
                reminder_owns = await self.reminder_blocks_navigation(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Weekly review callback failed operation=owner_lookup error_type=%s",
                    type(exc).__name__,
                )
                await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
                await self._weekly_review_edit_query(
                    query,
                    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    None,
                    operation="owner_lookup",
                )
                return
            if reminder_owns:
                await answer.answer(
                    "Сначала заверши текущее напоминание.",
                    show_alert=True,
                )
                return
            if claim.action == "close":
                await answer.answer()
                if not await self._weekly_review_consume_claim_safely(claim):
                    return
                await self._weekly_review_launch_action(update, context, query, user, claim)
                return
            existing: WeeklyReviewSessionSnapshot | None = None
            launched: WeeklyReviewSessionSnapshot | None = None
            launch_failure: str | None = None
            launch_stale = False
            try:
                # Phase one is read-only.  It selects the one callback answer
                # without holding the launch lock across Telegram I/O.
                async with self._weekly_review_launch_lock:
                    fresh_user = await self._weekly_review_access(update, fail_closed=True)
                    if not self._weekly_review_same_user_generation(fresh_user, user):
                        first_status = "access_changed"
                    else:
                        assert fresh_user is not None
                        first = await self.weekly_review_service.current_session(
                            telegram_actor_id=fresh_user.telegram_id,
                            chat_id=claim.chat_id,
                            expected_access_version=fresh_user.access_version,
                        )
                        first_status = (
                            "found"
                            if first.status == "found" and first.session is not None
                            else "available"
                            if first.status
                            in {
                                "not_found",
                                "expired",
                                "week_changed",
                            }
                            else "changed"
                        )
                if first_status == "found":
                    await answer.answer(
                        "Обзор недели уже открыт.",
                        show_alert=True,
                    )
                else:
                    await answer.answer()

                # Phase two rechecks every fence after answer.  Only this phase
                # may create a session or consume the launch capability.
                async with self._weekly_review_launch_lock:
                    fresh_user = await self._weekly_review_access(update, fail_closed=True)
                    if not self._weekly_review_same_user_generation(fresh_user, user):
                        launch_failure = "launch_access_changed"
                        await self._weekly_review_consume_claim_safely(claim)
                    else:
                        assert fresh_user is not None
                        current = await self.weekly_review_service.current_session(
                            telegram_actor_id=fresh_user.telegram_id,
                            chat_id=claim.chat_id,
                            expected_access_version=fresh_user.access_version,
                        )
                        if current.status == "found" and current.session is not None:
                            existing = current.session
                        elif current.status not in {
                            "not_found",
                            "expired",
                            "week_changed",
                        }:
                            launch_failure = "launch_current_changed"
                            await self._weekly_review_consume_claim_safely(claim)
                        else:
                            created = await self.weekly_review_service.create_session(
                                telegram_actor_id=fresh_user.telegram_id,
                                chat_id=claim.chat_id,
                                expected_access_version=fresh_user.access_version,
                                target_week_start=claim.week_start,
                                canonical_message_id=claim.canonical_message_id,
                                scheduled=claim.scheduled,
                                phase=WeeklyReviewPhase.AWAITING_INPUT,
                                replace_existing=False,
                            )
                            if created.status == "found" and created.session is not None:
                                existing = created.session
                            elif created.status != "created" or created.session is None:
                                launch_failure = "launch_failed"
                                await self._weekly_review_consume_claim_safely(claim)
                            elif not await self._weekly_review_consume_claim_safely(claim):
                                await self._weekly_review_clear_exact_safely(created.session)
                                launch_stale = True
                            else:
                                launched = created.session
            except _WeeklyReviewLookupError:
                launch_failure = "launch_lookup"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Weekly review callback failed operation=launch error_type=%s",
                    type(exc).__name__,
                )
                await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
                return
            if existing is not None:
                # Keep the sessionless token intact.  A stale launch screen
                # must neither replace nor spend the active durable generation.
                return
            if launch_stale:
                await self._weekly_review_recover_claim_callback(
                    context,
                    query,
                    claim,
                    operation="launch_stale",
                )
                return
            if launch_failure is not None:
                await answer.answer()
                await self._weekly_review_edit_query(
                    query,
                    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    None,
                    operation=launch_failure,
                )
                return
            if launched is None:
                await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
                return
            await self._weekly_review_render_launched(
                context,
                query,
                launched,
                claim.action,
            )
            return
        try:
            exact = await self.weekly_review_service.get_session_exact(
                telegram_actor_id=claim.telegram_user_id,
                chat_id=claim.chat_id,
                expected_access_version=claim.access_version,
                session_public_id=claim.session_public_id,
                expected_session_version=claim.session_version,
                canonical_message_id=claim.canonical_message_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review callback failed operation=exact_lookup error_type=%s",
                type(exc).__name__,
            )
            await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
            await self._weekly_review_clear_claim_safely(claim)
            await self._weekly_review_consume_claim_safely(claim)
            await self._weekly_review_recover_claim_callback(
                context,
                query,
                claim,
                operation="exact_lookup",
            )
            return
        if exact.status in {"access_denied", "access_changed"}:
            await answer.answer()
            await self._weekly_review_clear_claim_safely(claim)
            await self._weekly_review_consume_claim_safely(claim)
            await self._weekly_review_recover_claim_callback(
                context,
                query,
                claim,
                operation="exact_access_changed",
                access_changed=True,
            )
            return
        if exact.status != "found" or exact.session is None:
            await answer.answer(WEEKLY_REVIEW_STALE_TEXT, show_alert=True)
            await self._weekly_review_consume_claim_safely(claim)
            await self._weekly_review_recover_claim_callback(
                context,
                query,
                claim,
                operation="exact_outcome",
            )
            return
        await answer.answer()
        if not await self._weekly_review_consume_claim_safely(claim):
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                exact.session,
                operation="capability_consume",
            )
            return
        try:
            await self._weekly_review_session_action(
                update,
                context,
                query,
                user,
                exact.session,
                claim.action,
            )
        except asyncio.CancelledError as exc:
            recovery_session = getattr(
                exc,
                "_weekly_review_recovery_session",
                exact.session,
            )
            if not isinstance(recovery_session, WeeklyReviewSessionSnapshot):
                recovery_session = exact.session
            recovery = self._weekly_review_schedule_consumed_recovery(
                context,
                query,
                recovery_session,
                operation="action_cancelled",
                action=claim.action,
            )
            if recovery is not None:
                await asyncio.shield(recovery)
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=action error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                exact.session,
                operation="action_error",
            )

    async def _weekly_review_consume_claim_safely(self, claim: Any) -> bool:
        try:
            return await self.weekly_review_capabilities.consume(claim)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review callback failed operation=capability_consume error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_clear_claim_safely(self, claim: Any) -> bool:
        if claim.session_public_id is None or claim.session_version is None:
            return False
        try:
            return await self.weekly_review_service.clear_session_exact(
                telegram_actor_id=claim.telegram_user_id,
                chat_id=claim.chat_id,
                session_public_id=claim.session_public_id,
                expected_session_version=claim.session_version,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review callback failed operation=clear_exact error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_recover_claim_callback(
        self,
        context: Any,
        query: Any,
        claim: Any,
        *,
        operation: str,
        access_changed: bool = False,
    ) -> asyncio.Task[None] | None:
        """Recover a consumed claim without touching a newer exact generation."""

        try:
            user = await self._weekly_review_access_values(
                claim.telegram_user_id,
                claim.chat_id,
                fail_closed=True,
            )
            if user is not None:
                current = await self.weekly_review_service.current_session(
                    telegram_actor_id=user.telegram_id,
                    chat_id=claim.chat_id,
                    expected_access_version=user.access_version,
                )
                replacement = current.session if current.status == "found" else None
                if (
                    replacement is not None
                    and replacement.canonical_message_id == claim.canonical_message_id
                    and await self._weekly_review_exact(replacement) is not None
                ):
                    await self._weekly_review_render_current(context, replacement, query=query)
                    return
            if access_changed or user is None:
                await self._weekly_review_edit_query(
                    query,
                    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    None,
                    operation=operation,
                )
                return
            await self._weekly_review_render_sessionless_recovery(query, claim, user)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            await self._weekly_review_edit_query(
                query,
                WEEKLY_REVIEW_STALE_TEXT,
                None,
                operation="claim_recovery",
            )

    def _weekly_review_schedule_consumed_recovery(
        self,
        context: Any,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        *,
        operation: str,
        action: str | None = None,
    ) -> None:
        coroutine = (
            self._weekly_review_candidate_cancellation_recovery(context, query, session)
            if action is not None and action.startswith("candidate:")
            else self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation=operation,
                same_lineage_only=True,
            )
        )
        try:
            recovery = asyncio.create_task(
                coroutine,
                name="weekly-review-callback-recovery-lifecycle",
            )
        except Exception as exc:
            coroutine.close()
            logger.warning(
                "Weekly review action failed operation=recovery_schedule error_type=%s",
                type(exc).__name__,
            )
            return None
        self._weekly_review_track_task(recovery)
        return recovery

    async def _weekly_review_candidate_cancellation_recovery(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        try:
            user = await self._weekly_review_access_values(
                session.telegram_user_id,
                session.chat_id,
                fail_closed=True,
            )
            if user is None:
                if session.canonical_message_id is None:
                    await self._weekly_review_clear_exact_safely(session)
                    return
                async with self._weekly_review_ui_lock(
                    session.chat_id,
                    session.canonical_message_id,
                ):
                    if await self._weekly_review_clear_exact_safely(session):
                        await self._weekly_review_compensate(
                            context,
                            session,
                            query=query,
                            source_message=query.message,
                        )
                return
            current = await self.weekly_review_service.current_session(
                telegram_actor_id=user.telegram_id,
                chat_id=session.chat_id,
                expected_access_version=user.access_version,
            )
            fresh = current.session if current.status == "found" else None
            if fresh is not None and (
                fresh.public_id != session.public_id or fresh.version != session.version
            ):
                return
            if (
                fresh is not None
                and fresh.canonical_message_id == session.canonical_message_id
                and fresh.phase is WeeklyReviewPhase.REMINDER_HANDOFF
                and await self._weekly_review_exact(fresh) is not None
            ):
                await self._weekly_review_candidate_restore(context, query, fresh)
                return
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="candidate_cancelled",
                same_lineage_only=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=candidate_cancelled error_type=%s",
                type(exc).__name__,
            )

    async def _weekly_review_launch_action(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        user: User,
        claim: Any,
    ) -> None:
        if claim.action == "close":
            week = self.weekly_review_service.week_range(claim.week_start)
            await self._weekly_review_edit_query(
                query,
                f"🧭 Неделя: {self._weekly_review_date_range(week.start, week.end)}\n\n"
                "Обзор недели закрыт. Ничего не изменено.",
                None,
                operation="close_launch",
            )
            return
        created = await self.weekly_review_service.create_session(
            telegram_actor_id=user.telegram_id,
            chat_id=claim.chat_id,
            expected_access_version=user.access_version,
            target_week_start=claim.week_start,
            canonical_message_id=claim.canonical_message_id,
            scheduled=claim.scheduled,
            phase=(
                WeeklyReviewPhase.AWAITING_INPUT
                if claim.action in {"start", "focus", "reminders"}
                else WeeklyReviewPhase.ROOT
            ),
        )
        if created.status != "created" or created.session is None:
            await self._weekly_review_edit_query(
                query,
                WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                None,
                operation="launch_failed",
            )
            return
        await self._weekly_review_render_launched(
            context,
            query,
            created.session,
            claim.action,
        )

    async def _weekly_review_render_launched(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        action: str,
    ) -> None:
        if session.phase is WeeklyReviewPhase.AWAITING_INPUT or action in {
            "start",
            "focus",
            "reminders",
        }:
            await self._weekly_review_render(
                context,
                session,
                self._weekly_review_question_text(session),
                await self._weekly_review_cancel_markup(session),
                query=query,
            )
        else:
            await self._weekly_review_render_current(context, session, query=query)

    async def _weekly_review_session_action(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        user: User,
        session: WeeklyReviewSessionSnapshot,
        action: str,
    ) -> None:
        plan = self._weekly_review_reduce_action(session.phase, action)
        if plan.kind == "transition":
            assert plan.phase is not None
            await self._weekly_review_transition_action(
                context,
                query,
                session,
                operation=plan.operation,
                expected_phase=plan.expected_phase,
                phase=plan.phase,
                reset_input=plan.reset_input,
            )
            return
        if plan.kind == "confirm":
            await self._weekly_review_confirm(update, context, query, session)
            return
        if plan.kind == "candidate":
            await self._weekly_review_candidate_handoff(
                update,
                context,
                query,
                session,
                action,
            )
            return
        if plan.kind == "today":
            await self._weekly_review_render_today(context, query, user, session)
            return
        if plan.kind == "confirm_delete":
            await self._weekly_review_confirm_delete(context, query, session)
            return
        logger.warning(
            "Weekly review action failed operation=unknown_action error_type=%s",
            "UnknownAction",
        )
        await self._weekly_review_recover_consumed_callback(
            context,
            query,
            session,
            operation="unknown_action",
        )

    @staticmethod
    def _weekly_review_reduce_action(
        phase: WeeklyReviewPhase,
        action: str,
    ) -> _WeeklyReviewActionPlan:
        if phase is WeeklyReviewPhase.ROOT:
            if action in {"start", "focus", "reminders"}:
                return _WeeklyReviewActionPlan(
                    "transition",
                    action,
                    phase,
                    WeeklyReviewPhase.AWAITING_INPUT,
                    True,
                )
            if action in {"cancel", "close"}:
                return _WeeklyReviewActionPlan(
                    "transition", action, phase, WeeklyReviewPhase.COMPLETED
                )
            if action == "delete":
                return _WeeklyReviewActionPlan(
                    "transition", action, phase, WeeklyReviewPhase.DELETE_PREVIEW
                )
        elif phase is WeeklyReviewPhase.AWAITING_INPUT:
            if action in {"cancel", "close"}:
                return _WeeklyReviewActionPlan(
                    "transition", action, phase, WeeklyReviewPhase.COMPLETED
                )
        elif phase is WeeklyReviewPhase.PREVIEW:
            if action == "edit":
                return _WeeklyReviewActionPlan(
                    "transition",
                    action,
                    phase,
                    WeeklyReviewPhase.AWAITING_INPUT,
                    True,
                )
            if action in {"cancel", "close"}:
                return _WeeklyReviewActionPlan(
                    "transition", action, phase, WeeklyReviewPhase.COMPLETED
                )
            if action == "save":
                return _WeeklyReviewActionPlan("confirm", action, phase)
        elif phase is WeeklyReviewPhase.SAVED:
            if action == "edit":
                return _WeeklyReviewActionPlan(
                    "transition",
                    action,
                    phase,
                    WeeklyReviewPhase.AWAITING_INPUT,
                    True,
                )
            if action == "configure":
                return _WeeklyReviewActionPlan(
                    "transition", action, phase, WeeklyReviewPhase.CANDIDATES
                )
            if action == "today":
                return _WeeklyReviewActionPlan("today", action, phase)
            if action in {"done", "close"}:
                return _WeeklyReviewActionPlan(
                    "transition", action, phase, WeeklyReviewPhase.COMPLETED
                )
        elif phase is WeeklyReviewPhase.CANDIDATES:
            if action.startswith("candidate:"):
                return _WeeklyReviewActionPlan("candidate", "candidate", phase)
            if action == "back":
                return _WeeklyReviewActionPlan("transition", action, phase, WeeklyReviewPhase.SAVED)
        elif phase is WeeklyReviewPhase.REMINDER_HANDOFF:
            if action == "back":
                return _WeeklyReviewActionPlan("transition", action, phase, WeeklyReviewPhase.SAVED)
        elif phase is WeeklyReviewPhase.DELETE_PREVIEW:
            if action == "back":
                return _WeeklyReviewActionPlan("transition", action, phase, WeeklyReviewPhase.SAVED)
            if action == "confirm_delete":
                return _WeeklyReviewActionPlan("confirm_delete", action, phase)
        return _WeeklyReviewActionPlan("unknown", "unknown_action")

    async def _weekly_review_transition_action(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        *,
        operation: str,
        phase: WeeklyReviewPhase,
        expected_phase: WeeklyReviewPhase | None = None,
        reset_input: bool = False,
    ) -> None:
        reset = (
            {
                "focus": None,
                "approach": None,
                "small_steps": (),
                "reminder_candidates": (),
                "source": None,
            }
            if reset_input
            else {}
        )
        try:
            result = await self.weekly_review_service.transition_session(
                telegram_actor_id=session.telegram_user_id,
                chat_id=session.chat_id,
                expected_access_version=session.access_version,
                session_public_id=session.public_id,
                expected_session_version=session.version,
                expected_canonical_message_id=session.canonical_message_id,
                expected_phase=expected_phase,
                phase=phase,
                **reset,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation=f"{operation}_error",
            )
            return
        rendered = await self._weekly_review_handle_typed_outcome(
            context,
            query,
            session,
            result,
            operation=operation,
            success_statuses=frozenset({"updated", "replay"}),
        )
        if rendered is not None:
            await self._weekly_review_render_current(context, rendered, query=query)

    async def _weekly_review_confirm(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        task = asyncio.create_task(
            self._weekly_review_confirm_lifecycle(update, context, query, session),
            name="weekly-review-confirm-lifecycle",
        )
        self._weekly_review_track_task(task)
        await asyncio.shield(task)

    async def _weekly_review_confirm_lifecycle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        if session.canonical_message_id is None:
            return
        try:
            fresh = await self._weekly_review_access(update, fail_closed=True)
            exact = await self._weekly_review_exact(session)
        except _WeeklyReviewLookupError:
            await self._weekly_review_access_changed(
                context,
                session,
                query=query,
                source_message=query.message,
            )
            return
        access_ok = (
            fresh is not None
            and fresh.id == session.owner_id
            and fresh.access_version == session.access_version
        )
        if not access_ok or exact is None:
            if not access_ok:
                await self._weekly_review_access_changed(
                    context,
                    session,
                    query=query,
                    source_message=query.message,
                )
            else:
                await self._weekly_review_recover_consumed_callback(
                    context,
                    query,
                    session,
                    operation="confirm_precheck",
                )
            return
        try:
            result = await self.weekly_review_service.confirm_focus(
                telegram_actor_id=session.telegram_user_id,
                chat_id=session.chat_id,
                expected_access_version=session.access_version,
                session_public_id=session.public_id,
                expected_session_version=session.version,
                canonical_message_id=session.canonical_message_id,
                expected_week_start=session.week_start,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review save failed operation=confirm error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="confirm_error",
            )
            return
        saved = await self._weekly_review_handle_typed_outcome(
            context,
            query,
            session,
            result,
            operation="confirm",
            success_statuses=frozenset({"created", "updated", "duplicate", "replay"}),
        )
        if saved is not None:
            await self._weekly_review_render_current(context, saved, query=query)

    async def _weekly_review_confirm_delete(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        task = asyncio.create_task(
            self._weekly_review_confirm_delete_lifecycle(context, query, session),
            name="weekly-review-delete-lifecycle",
        )
        self._weekly_review_track_task(task)
        await asyncio.shield(task)

    async def _weekly_review_confirm_delete_lifecycle(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        if session.canonical_message_id is None:
            return
        try:
            result = await self.weekly_review_service.confirm_delete(
                telegram_actor_id=session.telegram_user_id,
                chat_id=session.chat_id,
                expected_access_version=session.access_version,
                session_public_id=session.public_id,
                expected_session_version=session.version,
                canonical_message_id=session.canonical_message_id,
                expected_week_start=session.week_start,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review delete failed operation=confirm_delete error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="confirm_delete_error",
            )
            return
        deleted = await self._weekly_review_handle_typed_outcome(
            context,
            query,
            session,
            result,
            operation="confirm_delete",
            success_statuses=frozenset({"deleted", "replay"}),
        )
        if deleted is not None:
            await self._weekly_review_render(
                context,
                deleted,
                f"{self._weekly_review_week_heading(deleted)}\n\nФокус недели удалён.",
                None,
                query=query,
            )

    async def _weekly_review_handle_typed_outcome(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        result: Any,
        *,
        operation: str,
        success_statuses: frozenset[str],
    ) -> WeeklyReviewSessionSnapshot | None:
        status = str(getattr(result, "status", "stale"))
        if status in success_statuses:
            resulting = getattr(result, "session", None)
            if isinstance(resulting, WeeklyReviewSessionSnapshot):
                return resulting
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation=f"{operation}_missing_success",
            )
            return None
        if status == "focus_changed":
            await self._weekly_review_recover_focus_changed(
                context,
                query,
                getattr(result, "session", None) or session,
            )
            return None
        if status in {"access_changed", "access_denied"}:
            if await self._weekly_review_render_allowed_replacement(
                context,
                query,
                session,
                operation=f"{operation}_{status}",
            ):
                return None
            await self._weekly_review_access_changed(
                context,
                session,
                query=query,
                source_message=query.message,
            )
            return None
        if status in {"expired", "week_changed", "not_found", "stale"}:
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation=f"{operation}_{status}",
            )
            return None
        logger.warning(
            "Weekly review action failed operation=%s error_type=%s",
            operation,
            "UnexpectedOutcome",
        )
        await self._weekly_review_recover_consumed_callback(
            context,
            query,
            session,
            operation=f"{operation}_unexpected",
        )
        return None

    async def _weekly_review_render_allowed_replacement(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        *,
        operation: str,
    ) -> bool:
        """Preserve a newer allowed generation before neutralizing an old one."""

        try:
            user = await self._weekly_review_access_values(
                session.telegram_user_id,
                session.chat_id,
                fail_closed=True,
            )
            if user is None:
                return False
            current = await self.weekly_review_service.current_session(
                telegram_actor_id=user.telegram_id,
                chat_id=session.chat_id,
                expected_access_version=user.access_version,
            )
            replacement = current.session if current.status == "found" else None
            if (
                replacement is None
                or replacement.canonical_message_id != session.canonical_message_id
                or self._weekly_review_same_session(replacement, session)
                or await self._weekly_review_exact(replacement) is None
            ):
                return False
            return await self._weekly_review_render_current(
                context,
                replacement,
                query=query,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False

    async def _weekly_review_recover_focus_changed(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        if session.canonical_message_id is None:
            return
        try:
            recovered = await self.weekly_review_service.transition_session(
                telegram_actor_id=session.telegram_user_id,
                chat_id=session.chat_id,
                expected_access_version=session.access_version,
                session_public_id=session.public_id,
                expected_session_version=session.version,
                expected_canonical_message_id=session.canonical_message_id,
                phase=WeeklyReviewPhase.ROOT,
                focus=None,
                approach=None,
                small_steps=(),
                reminder_candidates=(),
                source=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=focus_changed_recovery error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="focus_changed_recovery_error",
            )
            return
        if recovered.status == "updated" and recovered.session is not None:
            await self._weekly_review_render_current(
                context,
                recovered.session,
                query=query,
            )
            return
        if recovered.status in {"access_changed", "access_denied"}:
            await self._weekly_review_access_changed(
                context,
                session,
                query=query,
                source_message=query.message,
            )
            return
        await self._weekly_review_recover_consumed_callback(
            context,
            query,
            session,
            operation=f"focus_changed_{recovered.status}",
        )

    async def _weekly_review_recover_consumed_callback(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        *,
        operation: str,
        same_lineage_only: bool = False,
    ) -> None:
        """Replace a consumed dead screen without mutating a newer generation."""

        try:
            user = await self._weekly_review_access_values(
                session.telegram_user_id,
                session.chat_id,
                fail_closed=True,
            )
            if not self._weekly_review_access_matches(session, user):
                if same_lineage_only and user is not None:
                    current = await self.weekly_review_service.current_session(
                        telegram_actor_id=user.telegram_id,
                        chat_id=session.chat_id,
                        expected_access_version=user.access_version,
                    )
                    replacement = current.session if current.status == "found" else None
                    if (
                        replacement is not None
                        and replacement.public_id != session.public_id
                        and replacement.canonical_message_id == session.canonical_message_id
                    ):
                        return
                await self._weekly_review_access_changed(
                    context,
                    session,
                    query=query,
                    source_message=query.message,
                )
                return
            assert user is not None
            current = await self.weekly_review_service.current_session(
                telegram_actor_id=user.telegram_id,
                chat_id=session.chat_id,
                expected_access_version=user.access_version,
            )
            if current.status in {"access_changed", "access_denied"}:
                await self._weekly_review_access_changed(
                    context,
                    session,
                    query=query,
                    source_message=query.message,
                )
                return
            fresh = current.session if current.status == "found" else None
            if (
                fresh is not None
                and fresh.canonical_message_id == session.canonical_message_id
                and await self._weekly_review_exact(fresh) is not None
            ):
                if same_lineage_only and fresh.public_id != session.public_id:
                    return
                await self._weekly_review_render_current(context, fresh, query=query)
                return
            if same_lineage_only and fresh is not None:
                return
            await self._weekly_review_render_sessionless_recovery(query, session, user)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            await self._weekly_review_edit_query(
                query,
                WEEKLY_REVIEW_STALE_TEXT,
                None,
                operation="outcome_recovery",
            )

    async def _weekly_review_render_sessionless_recovery(
        self,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        user: User,
    ) -> None:
        if session.canonical_message_id is None:
            return
        week_start = self.weekly_review_service.target_week_start(
            user.timezone,
            review_weekday=self.settings.weekly_review_weekday,
        )
        tokens = await self.weekly_review_capabilities.issue(
            actions=("start", "close"),
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=session.chat_id,
            canonical_message_id=session.canonical_message_id,
            access_version=user.access_version,
            week_start=week_start,
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🧭 Открыть обзор",
                        callback_data=f"wrev:{tokens['start']}",
                    )
                ],
                [InlineKeyboardButton("Закрыть", callback_data=f"wrev:{tokens['close']}")],
            ]
        )
        edited = await self._weekly_review_edit_query(
            query,
            WEEKLY_REVIEW_RECOVERY_TEXT,
            markup,
            operation="outcome_recovery",
        )
        if not edited:
            await self.weekly_review_capabilities.revoke_tokens(tuple(tokens.values()))
            return
        final_user = await self._weekly_review_access_values(
            session.telegram_user_id,
            session.chat_id,
            fail_closed=True,
        )
        if self._weekly_review_same_user_generation(final_user, user) and (
            final_user is not None
            and self.weekly_review_service.target_week_start(
                final_user.timezone,
                review_weekday=self.settings.weekly_review_weekday,
            )
            == week_start
        ):
            return
        await self.weekly_review_capabilities.revoke_tokens(tuple(tokens.values()))
        await self._weekly_review_edit_query(
            query,
            WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
            None,
            operation="outcome_recovery_fence",
        )

    async def _weekly_review_candidate_handoff(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
        action: str,
    ) -> None:
        try:
            index = int(action.partition(":")[2])
        except ValueError:
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="candidate_invalid",
            )
            return
        if not 0 <= index < len(session.reminder_candidates):
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="candidate_invalid",
            )
            return
        try:
            result = await self.weekly_review_service.transition_session(
                telegram_actor_id=session.telegram_user_id,
                chat_id=session.chat_id,
                expected_access_version=session.access_version,
                session_public_id=session.public_id,
                expected_session_version=session.version,
                expected_canonical_message_id=session.canonical_message_id,
                expected_phase=WeeklyReviewPhase.CANDIDATES,
                phase=WeeklyReviewPhase.REMINDER_HANDOFF,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=candidate error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="candidate_error",
            )
            return
        updated = await self._weekly_review_handle_typed_outcome(
            context,
            query,
            session,
            result,
            operation="candidate",
            success_statuses=frozenset({"updated", "replay"}),
        )
        if updated is None:
            return
        try:
            await self._weekly_review_candidate_handoff_updated(
                update,
                context,
                query,
                updated,
                index=index,
                replay=result.status == "replay",
            )
        except asyncio.CancelledError as exc:
            exc._weekly_review_recovery_session = updated
            raise

    async def _weekly_review_candidate_handoff_updated(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        updated: WeeklyReviewSessionSnapshot,
        *,
        index: int,
        replay: bool,
    ) -> None:
        if replay:
            await self._weekly_review_render_current(context, updated, query=query)
            return
        if not 0 <= index < len(updated.reminder_candidates):
            await self._weekly_review_candidate_restore(context, query, updated)
            return
        if not await self._weekly_review_candidate_fence(updated):
            await self._weekly_review_candidate_restore(context, query, updated)
            return
        candidate = updated.reminder_candidates[index]
        try:
            handled = await self.reminder_from_weekly_candidate(
                update,
                context,
                title=candidate.title,
                schedule_wording=candidate.schedule_wording,
                canonical_message=query.message,
                expected_access_version=updated.access_version,
            )
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=candidate_handoff error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_candidate_restore(context, query, updated)
            return
        if not handled:
            await self._weekly_review_candidate_restore(context, query, updated)
            return
        if not await self._weekly_review_candidate_fence(updated):
            await self._weekly_review_candidate_restore(context, query, updated)

    async def _weekly_review_candidate_fence(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> bool:
        try:
            user = await self._weekly_review_access_values(
                session.telegram_user_id,
                session.chat_id,
                fail_closed=True,
            )
            return (
                self._weekly_review_access_matches(session, user)
                and await self._weekly_review_exact(session) is not None
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review action failed operation=candidate_fence error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_candidate_restore(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        await self._weekly_review_transition_action(
            context,
            query,
            session,
            operation="candidate_restore",
            expected_phase=WeeklyReviewPhase.REMINDER_HANDOFF,
            phase=WeeklyReviewPhase.CANDIDATES,
        )

    async def weekly_review_reminder_return_markup(
        self,
        *,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        canonical_message_id: int | None,
        access_version: int,
    ) -> InlineKeyboardMarkup | None:
        if not self.weekly_review_policy.enabled or canonical_message_id is None:
            return None
        try:
            user = await self._weekly_review_access_values(
                telegram_user_id,
                chat_id,
                fail_closed=True,
            )
        except _WeeklyReviewLookupError:
            return None
        if user is None or user.id != owner_id or user.access_version != access_version:
            return None
        result = await self.weekly_review_service.current_session(
            telegram_actor_id=telegram_user_id,
            chat_id=chat_id,
            expected_access_version=access_version,
        )
        session = result.session
        if (
            result.status != "found"
            or session is None
            or session.owner_id != owner_id
            or session.canonical_message_id != canonical_message_id
            or session.phase is not WeeklyReviewPhase.REMINDER_HANDOFF
        ):
            return None
        tokens = await self._weekly_review_issue(session, ("back",))
        if not tokens:
            return None
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("← К обзору недели", callback_data=f"wrev:{tokens['back']}")]]
        )

    async def _weekly_review_render_today(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        query: Any,
        user: User,
        session: WeeklyReviewSessionSnapshot,
    ) -> None:
        try:
            snapshot = await self.focus_service.materialize_today_application(
                user.id,
                include_weekly_focus=True,
            )
            if not await self._weekly_review_today_fence(snapshot, session):
                await self._weekly_review_recover_consumed_callback(
                    context,
                    query,
                    session,
                    operation="today_pre_fence",
                )
                return
            plan = await self.focus_service.generate_today_plan(snapshot)
            if not await self._weekly_review_today_fence(snapshot, session):
                await self._weekly_review_recover_consumed_callback(
                    context,
                    query,
                    session,
                    operation="today_post_fence",
                )
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review today failed operation=today error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="today_error",
            )
            return
        actions = "\n".join(f"{i}. {action}" for i, action in enumerate(plan.actions, 1))
        weekly_line = (
            f"🎯 Фокус недели: {snapshot.weekly_focus}\n\n" if snapshot.weekly_focus else ""
        )
        text = (
            f"{weekly_line}{plan.vision_reminder}\n\nФокус: {plan.main_focus}\n{actions}\n\n"
            f"На сложный день: {plan.hard_day_minimum}"
        )
        tokens = await self._weekly_review_issue(session, ("back",))
        markup = (
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "← К обзору недели", callback_data=f"wrev:{tokens['back']}"
                        )
                    ]
                ]
            )
            if tokens
            else None
        )
        rendered = await self._weekly_review_render(
            context,
            session,
            text,
            markup,
            query=query,
            today_snapshot=snapshot,
        )
        if not rendered:
            await self._weekly_review_recover_consumed_callback(
                context,
                query,
                session,
                operation="today_render",
            )

    async def _weekly_review_today_fence(
        self,
        snapshot: TodayApplicationSnapshot,
        session: WeeklyReviewSessionSnapshot,
    ) -> bool:
        if (
            snapshot.actor_id != session.owner_id
            or snapshot.telegram_id != session.telegram_user_id
            or snapshot.access_version != session.access_version
            or not snapshot.includes_weekly_focus
            or not self.weekly_review_policy.allows_actor(
                snapshot,
                expected_access_version=session.access_version,
            )
        ):
            return False
        user = await self._weekly_review_access_values(
            session.telegram_user_id,
            session.chat_id,
            fail_closed=True,
        )
        if not self._weekly_review_access_matches(session, user):
            return False
        if await self._weekly_review_exact(session) is None:
            return False
        check = await self.focus_service.check_today_application(snapshot)
        return bool(check.is_current)

    async def _weekly_review_today_fence_safely(
        self,
        snapshot: TodayApplicationSnapshot,
        session: WeeklyReviewSessionSnapshot,
    ) -> bool:
        try:
            return await self._weekly_review_today_fence(snapshot, session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review today failed operation=fence error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_render_current(
        self,
        context: Any,
        session: WeeklyReviewSessionSnapshot,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> bool:
        if session.phase is WeeklyReviewPhase.ROOT:
            user = await self._weekly_review_access_values(
                session.telegram_user_id,
                session.chat_id,
            )
            if user is None:
                await self._weekly_review_access_changed(
                    context,
                    session,
                    query=query,
                    source_message=source_message,
                )
                return False
            text = await self._weekly_review_root_text(session, user)
            if text is None:
                await self._weekly_review_access_changed(
                    context,
                    session,
                    query=query,
                    source_message=source_message,
                )
                return False
            markup = await self._weekly_review_root_markup(session, user)
        elif session.phase is WeeklyReviewPhase.AWAITING_INPUT:
            text = self._weekly_review_question_text(session)
            markup = await self._weekly_review_cancel_markup(session)
        elif session.phase is WeeklyReviewPhase.PROCESSING:
            text = f"{self._weekly_review_week_heading(session)}\n\nОтвет уже обрабатывается."
            markup = None
        elif session.phase is WeeklyReviewPhase.PREVIEW:
            text = self._weekly_review_preview_text(session)
            markup = await self._weekly_review_preview_markup(session)
        elif session.phase is WeeklyReviewPhase.SAVED:
            text = f"{self._weekly_review_week_heading(session)}\n\n✅ Фокус недели сохранён"
            markup = await self._weekly_review_saved_markup(session)
        elif session.phase is WeeklyReviewPhase.CANDIDATES:
            text = (
                f"{self._weekly_review_week_heading(session)}\n\n"
                "🔔 Возможные напоминания\n\n"
                "Выбери один кандидат. Каждый создаётся отдельно."
            )
            markup = await self._weekly_review_candidates_markup(session)
        elif session.phase is WeeklyReviewPhase.REMINDER_HANDOFF:
            text = (
                f"{self._weekly_review_week_heading(session)}\n\n"
                "🔔 Настройка напоминания уже открыта. Заверши её или вернись к обзору."
            )
            tokens = await self._weekly_review_issue(session, ("back",))
            markup = (
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "← К обзору недели",
                                callback_data=f"wrev:{tokens['back']}",
                            )
                        ]
                    ]
                )
                if tokens
                else None
            )
        elif session.phase is WeeklyReviewPhase.DELETE_PREVIEW:
            text = (
                f"{self._weekly_review_week_heading(session)}\n\n"
                "Удалить подтверждённый фокус этой недели? "
                "Это действие требует подтверждения."
            )
            markup = await self._weekly_review_delete_markup(session)
        else:
            text = f"{self._weekly_review_week_heading(session)}\n\nОбзор недели завершён."
            markup = None
        return await self._weekly_review_render(
            context,
            session,
            text,
            markup,
            query=query,
            source_message=source_message,
        )

    async def _weekly_review_root_text(
        self,
        session: WeeklyReviewSessionSnapshot,
        user: User,
    ) -> str | None:
        week = self.weekly_review_service.week_range(session.week_start)
        snapshot = await self.weekly_review_service.system_snapshot(
            telegram_actor_id=user.telegram_id,
            expected_access_version=user.access_version,
            target_week_start=session.week_start,
        )
        if snapshot.status != "ready":
            return None
        focus_line = (
            f"\n\nТекущий фокус: {snapshot.current_focus.focus}"
            if snapshot.current_focus is not None
            else ""
        )
        attention_count = sum(item.requires_attention for item in snapshot.active_tasks)
        factual_summary = (
            "\n\nСнимок системы: "
            f"завершено за прошлый цикл — {snapshot.completed_previous_cycle}; "
            f"активных задач — {len(snapshot.active_tasks)}"
            f" (требуют внимания — {attention_count}); "
            f"ближайших напоминаний — {len(snapshot.reminders)}."
        )
        return (
            "🧭 Обзор недели\n\n"
            f"Неделя: {self._weekly_review_date_range(week.start, week.end)}\n\n"
            "Пора спокойно посмотреть на неделю 🌿\n"
            "Выберем один ориентир и отдельно проверим напоминания.\n"
            f"Ничего не изменю без подтверждения.{focus_line}{factual_summary}"
        )

    async def _weekly_review_root_markup(
        self,
        session: WeeklyReviewSessionSnapshot,
        user: User,
    ) -> InlineKeyboardMarkup | None:
        lookup = await self.weekly_review_service.get_focus(
            telegram_actor_id=user.telegram_id,
            expected_access_version=user.access_version,
            week_start=session.week_start,
        )
        tokens = await self._weekly_review_issue(
            session,
            ("start", "focus", "reminders", "cancel") + (("delete",) if lookup.focus else ()),
        )
        if not tokens:
            return None
        focus_label = "✏️ Изменить фокус" if lookup.focus is not None else "🎯 Фокус недели"
        rows = [
            [InlineKeyboardButton("🧭 Начать обзор", callback_data=f"wrev:{tokens['start']}")],
            [
                InlineKeyboardButton(focus_label, callback_data=f"wrev:{tokens['focus']}"),
                InlineKeyboardButton("🔔 Напоминания", callback_data=f"wrev:{tokens['reminders']}"),
            ],
        ]
        if lookup.focus is not None:
            rows.append(
                [InlineKeyboardButton("🗑 Удалить фокус", callback_data=f"wrev:{tokens['delete']}")]
            )
        rows.append([InlineKeyboardButton("Не сейчас", callback_data=f"wrev:{tokens['cancel']}")])
        return InlineKeyboardMarkup(rows)

    def _weekly_review_preview_text(self, session: WeeklyReviewSessionSnapshot) -> str:
        week = self.weekly_review_service.week_range(session.week_start)
        lines = [
            f"🧭 Неделя: {self._weekly_review_date_range(week.start, week.end)}",
            "",
            "🎯 Фокус",
            session.focus or "—",
        ]
        if session.approach:
            lines.extend(("", "🌿 Подход", session.approach))
        if session.small_steps:
            lines.extend(("", "👣 Небольшие шаги"))
            lines.extend(
                f"{index}. {step}" for index, step in enumerate(session.small_steps, start=1)
            )
        lines.extend(("", "🔔 Возможные напоминания"))
        if session.reminder_candidates:
            lines.extend(
                f"{index}. {candidate.title} — {candidate.schedule_wording} · Ещё не создано"
                for index, candidate in enumerate(session.reminder_candidates, start=1)
            )
        else:
            lines.append("Нет · Ещё не созданы")
        return "\n".join(lines)

    async def _weekly_review_preview_markup(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> InlineKeyboardMarkup | None:
        tokens = await self._weekly_review_issue(session, ("save", "edit", "cancel"))
        if not tokens:
            return None
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✅ Сохранить на неделю", callback_data=f"wrev:{tokens['save']}"
                    )
                ],
                [
                    InlineKeyboardButton("✏️ Исправить", callback_data=f"wrev:{tokens['edit']}"),
                    InlineKeyboardButton("Отмена", callback_data=f"wrev:{tokens['cancel']}"),
                ],
            ]
        )

    async def _weekly_review_saved_markup(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> InlineKeyboardMarkup | None:
        actions = ["today", "done", "edit"]
        if session.reminder_candidates:
            actions.insert(0, "configure")
        tokens = await self._weekly_review_issue(session, tuple(actions))
        if not tokens:
            return None
        rows: list[list[InlineKeyboardButton]] = []
        if session.reminder_candidates:
            rows.append(
                [
                    InlineKeyboardButton(
                        f"🔔 Настроить найденные ({len(session.reminder_candidates)})",
                        callback_data=f"wrev:{tokens['configure']}",
                    )
                ]
            )
        rows.extend(
            [
                [
                    InlineKeyboardButton(
                        "🌱 Фокус на сегодня", callback_data=f"wrev:{tokens['today']}"
                    )
                ],
                [InlineKeyboardButton("✏️ Изменить", callback_data=f"wrev:{tokens['edit']}")],
                [InlineKeyboardButton("Готово", callback_data=f"wrev:{tokens['done']}")],
            ]
        )
        return InlineKeyboardMarkup(rows)

    async def _weekly_review_candidates_markup(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> InlineKeyboardMarkup | None:
        actions = tuple(
            [f"candidate:{index}" for index in range(len(session.reminder_candidates))] + ["back"]
        )
        tokens = await self._weekly_review_issue(session, actions)
        if not tokens:
            return None
        rows = [
            [
                InlineKeyboardButton(
                    self._weekly_review_button_label(
                        f"🔔 {candidate.title} — {candidate.schedule_wording}"
                    ),
                    callback_data=f"wrev:{tokens[f'candidate:{index}']}",
                )
            ]
            for index, candidate in enumerate(session.reminder_candidates)
        ]
        rows.append([InlineKeyboardButton("← К обзору", callback_data=f"wrev:{tokens['back']}")])
        return InlineKeyboardMarkup(rows)

    async def _weekly_review_cancel_markup(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> InlineKeyboardMarkup | None:
        tokens = await self._weekly_review_issue(session, ("cancel",))
        if not tokens:
            return None
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Отмена", callback_data=f"wrev:{tokens['cancel']}")]]
        )

    async def _weekly_review_delete_markup(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> InlineKeyboardMarkup | None:
        tokens = await self._weekly_review_issue(session, ("confirm_delete", "back"))
        if not tokens:
            return None
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🗑 Удалить", callback_data=f"wrev:{tokens['confirm_delete']}"
                    )
                ],
                [InlineKeyboardButton("← Назад", callback_data=f"wrev:{tokens['back']}")],
            ]
        )

    async def _weekly_review_issue(
        self,
        session: WeeklyReviewSessionSnapshot,
        actions: tuple[str, ...],
    ) -> dict[str, str]:
        if session.canonical_message_id is None:
            return {}
        return await self.weekly_review_capabilities.issue(
            actions=actions,
            owner_id=session.owner_id,
            telegram_user_id=session.telegram_user_id,
            chat_id=session.chat_id,
            canonical_message_id=session.canonical_message_id,
            access_version=session.access_version,
            week_start=session.week_start,
            session_public_id=session.public_id,
            session_version=session.version,
            # Publish replacement controls before Telegram I/O without
            # revoking the still-visible screen.  Exact durable fencing makes
            # older generations stale; TTL/cap bounds retire their tokens.
            replace_session=False,
        )

    async def _weekly_review_to_input(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> WeeklyReviewSessionSnapshot | None:
        if session.canonical_message_id is None:
            return None
        result = await self.weekly_review_service.transition_session(
            telegram_actor_id=session.telegram_user_id,
            chat_id=session.chat_id,
            expected_access_version=session.access_version,
            session_public_id=session.public_id,
            expected_session_version=session.version,
            expected_canonical_message_id=session.canonical_message_id,
            phase=WeeklyReviewPhase.AWAITING_INPUT,
            focus=None,
            approach=None,
            small_steps=(),
            reminder_candidates=(),
            source=None,
        )
        return result.session if result.status == "updated" else None

    def _weekly_review_focus_text(self, focus: WeeklyFocusSnapshot) -> str:
        week = self.weekly_review_service.week_range(focus.week_start)
        lines = [
            f"🧭 Неделя: {self._weekly_review_date_range(week.start, week.end)}",
            "",
            "🎯 Фокус недели",
            "",
            focus.focus,
        ]
        if focus.approach:
            lines.extend(("", "🌿 Подход", focus.approach))
        if focus.small_steps:
            lines.extend(("", "👣 Небольшие шаги"))
            lines.extend(
                f"{index}. {step}" for index, step in enumerate(focus.small_steps, start=1)
            )
        return "\n".join(lines)

    @staticmethod
    def _weekly_review_date_range(start: date, end: date) -> str:
        return f"{start.strftime('%d.%m.%Y')}–{end.strftime('%d.%m.%Y')}"

    def _weekly_review_week_heading(self, session: WeeklyReviewSessionSnapshot) -> str:
        week = self.weekly_review_service.week_range(session.week_start)
        return f"🧭 Неделя: {self._weekly_review_date_range(week.start, week.end)}"

    def _weekly_review_question_text(self, session: WeeklyReviewSessionSnapshot) -> str:
        return f"{self._weekly_review_week_heading(session)}\n\n{WEEKLY_REVIEW_QUESTION}"

    @staticmethod
    def _weekly_review_button_label(value: str) -> str:
        return value if len(value) <= 60 else value[:59].rstrip() + "…"

    async def _weekly_review_render(
        self,
        context: Any,
        session: WeeklyReviewSessionSnapshot,
        text: str,
        markup: InlineKeyboardMarkup | None,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
        today_snapshot: TodayApplicationSnapshot | None = None,
    ) -> bool:
        if session.canonical_message_id is None:
            return False
        screen = await self._weekly_review_stage_render_screen(session, markup)
        if screen is None:
            return False
        task = asyncio.create_task(
            self._weekly_review_delivery_lifecycle(
                context,
                session,
                text,
                markup,
                screen=screen,
                query=query,
                source_message=source_message,
                today_snapshot=today_snapshot,
            ),
            name="weekly-review-delivery-lifecycle",
        )
        self._weekly_review_track_task(task)
        return await asyncio.shield(task)

    async def _weekly_review_delivery_lifecycle(
        self,
        context: Any,
        session: WeeklyReviewSessionSnapshot,
        text: str,
        markup: InlineKeyboardMarkup | None,
        *,
        screen: WeeklyReviewScreen,
        query: Any | None,
        source_message: Any | None,
        today_snapshot: TodayApplicationSnapshot | None,
    ) -> bool:
        lock = self._weekly_review_ui_lock(session.chat_id, session.canonical_message_id)
        async with lock:
            primary_edit_accepted = False
            try:
                first_user = await self._weekly_review_access_values(
                    session.telegram_user_id,
                    session.chat_id,
                )
                first_live = await self._weekly_review_exact(session)
                final_user = await self._weekly_review_access_values(
                    session.telegram_user_id,
                    session.chat_id,
                )
                final_live = await self._weekly_review_exact(session)
                if not self._weekly_review_access_matches(
                    session, first_user
                ) or not self._weekly_review_access_matches(session, final_user):
                    # Access lookup is authoritative and fail-closed.  The exact
                    # CAS is safe even when access-gated reads can no longer
                    # materialize the frozen generation; it cannot clear a
                    # concurrent replacement with a different version.
                    await self._weekly_review_revoke_screen_safely(screen)
                    await self._weekly_review_clear_exact_safely(session)
                    await self._weekly_review_compensate(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                    return False
                if first_live is None or final_live is None:
                    await self._weekly_review_revoke_screen_safely(screen)
                    return False
                if (
                    query is not None
                    and getattr(query.message, "message_id", None) != session.canonical_message_id
                ):
                    await self._weekly_review_revoke_screen_safely(screen)
                    return False
                if not await self._weekly_review_screen_is_live_safely(screen):
                    return False
                if today_snapshot is not None and not await self._weekly_review_today_fence_safely(
                    today_snapshot,
                    session,
                ):
                    await self._weekly_review_revoke_screen_safely(screen)
                    return False
                edited = await self._weekly_review_primary_edit(
                    context,
                    session,
                    text,
                    markup,
                    query=query,
                    source_message=source_message,
                )
                if not edited:
                    await self._weekly_review_revoke_screen_safely(screen)
                    return False
                primary_edit_accepted = True
                post_first_user = await self._weekly_review_access_values(
                    session.telegram_user_id,
                    session.chat_id,
                )
                post_first_live = await self._weekly_review_exact(session)
                post_final_user = await self._weekly_review_access_values(
                    session.telegram_user_id,
                    session.chat_id,
                )
                post_final_live = await self._weekly_review_exact(session)
                access_ok = self._weekly_review_access_matches(
                    session, post_first_user
                ) and self._weekly_review_access_matches(session, post_final_user)
                exact_ok = post_first_live is not None and post_final_live is not None
                today_ok = today_snapshot is None or await self._weekly_review_today_fence_safely(
                    today_snapshot,
                    session,
                )
                if access_ok and exact_ok and today_ok:
                    if await self._weekly_review_activate_screen_safely(screen):
                        return True
                    await self._weekly_review_compensate(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                    return False
                await self._weekly_review_revoke_screen_safely(screen)
                if not today_ok:
                    await self._weekly_review_compensate(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                    return False
                if not access_ok:
                    await self._weekly_review_clear_exact_safely(session)
                # An accepted old edit is neutralized even when a same-canonical
                # replacement appeared.  Only the exact old DB generation may be
                # cleared; a fresh renderer can safely repaint after this lock.
                await self._weekly_review_compensate(
                    context,
                    session,
                    query=query,
                    source_message=source_message,
                )
                return False
            except asyncio.CancelledError:
                await self._weekly_review_revoke_screen_safely(screen)
                if today_snapshot is not None and primary_edit_accepted:
                    await self._weekly_review_compensate(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                raise
            except Exception as exc:
                logger.warning(
                    "Weekly review delivery failed operation=delivery error_type=%s",
                    type(exc).__name__,
                )
                # Lookup/storage failures are fail-closed.  The primary
                # Telegram operation may have completed before surfacing an
                # unexpected transport exception, so always neutralize the
                # frozen canonical; exact CAS cannot remove a replacement.
                await self._weekly_review_revoke_screen_safely(screen)
                await self._weekly_review_clear_exact_safely(session)
                await self._weekly_review_compensate(
                    context,
                    session,
                    query=query,
                    source_message=source_message,
                )
                return False

    async def _weekly_review_stage_render_screen(
        self,
        session: WeeklyReviewSessionSnapshot,
        markup: InlineKeyboardMarkup | None,
    ) -> WeeklyReviewScreen | None:
        tokens = tuple(
            callback.removeprefix("wrev:")
            for row in (markup.inline_keyboard if markup is not None else ())
            for button in row
            if isinstance((callback := button.callback_data), str) and callback.startswith("wrev:")
        )
        try:
            if tokens:
                return await self.weekly_review_capabilities.screen_for_tokens(tokens)
            return await self.weekly_review_capabilities.stage_screen(session.public_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review delivery failed operation=stage_screen error_type=%s",
                type(exc).__name__,
            )
            return None

    async def _weekly_review_screen_is_live_safely(
        self,
        screen: WeeklyReviewScreen,
    ) -> bool:
        try:
            return await self.weekly_review_capabilities.screen_is_live(screen)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review delivery failed operation=screen_lookup error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_activate_screen_safely(
        self,
        screen: WeeklyReviewScreen,
    ) -> bool:
        try:
            return await self.weekly_review_capabilities.activate_screen(screen)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review delivery failed operation=screen_activate error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_revoke_screen_safely(
        self,
        screen: WeeklyReviewScreen,
    ) -> bool:
        try:
            return await self.weekly_review_capabilities.revoke_screen(screen)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review delivery failed operation=screen_revoke error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_primary_edit(
        self,
        context: Any,
        session: WeeklyReviewSessionSnapshot,
        text: str,
        markup: InlineKeyboardMarkup | None,
        *,
        query: Any | None,
        source_message: Any | None,
    ) -> bool:
        try:
            if query is not None:
                await query.edit_message_text(text, reply_markup=markup)
                return True
            bot = getattr(context, "bot", None)
            edit = getattr(bot, "edit_message_text", None)
            if callable(edit):
                await edit(
                    chat_id=session.chat_id,
                    message_id=session.canonical_message_id,
                    text=text,
                    reply_markup=markup,
                )
                return True
            if (
                source_message is not None
                and getattr(source_message, "message_id", None) == session.canonical_message_id
                and hasattr(source_message, "edit_text")
            ):
                await source_message.edit_text(text, reply_markup=markup)
                return True
            return False
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if "message is not modified" in str(exc).casefold():
                return True
            logger.warning(
                "Weekly review canonical edit failed operation=primary error_type=%s",
                type(exc).__name__,
            )
            return False
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Weekly review canonical edit failed operation=primary error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_compensate(
        self,
        context: Any,
        session: WeeklyReviewSessionSnapshot,
        *,
        query: Any | None,
        source_message: Any | None,
    ) -> None:
        try:
            if query is not None:
                await query.edit_message_text(
                    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    reply_markup=None,
                    parse_mode=None,
                )
                return
            bot = getattr(context, "bot", None)
            edit = getattr(bot, "edit_message_text", None)
            if callable(edit):
                await edit(
                    chat_id=session.chat_id,
                    message_id=session.canonical_message_id,
                    text=WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    reply_markup=None,
                    parse_mode=None,
                )
                return
            if (
                source_message is not None
                and getattr(source_message, "message_id", None) == session.canonical_message_id
                and hasattr(source_message, "edit_text")
            ):
                await source_message.edit_text(
                    WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                    reply_markup=None,
                    parse_mode=None,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review canonical edit failed operation=compensation error_type=%s",
                type(exc).__name__,
            )

    async def _weekly_review_access_changed(
        self,
        context: Any,
        session: WeeklyReviewSessionSnapshot,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        if session.canonical_message_id is None:
            await self._weekly_review_clear_exact_safely(session)
            return
        async with self._weekly_review_ui_lock(
            session.chat_id,
            session.canonical_message_id,
        ):
            await self._weekly_review_clear_exact_safely(session)
            await self._weekly_review_compensate(
                context,
                session,
                query=query,
                source_message=source_message,
            )

    async def _weekly_review_pre_send_session_fence(
        self,
        update: Update,
        session: WeeklyReviewSessionSnapshot,
    ) -> User | None:
        try:
            first_user = await self._weekly_review_access(update, fail_closed=True)
            first_exact = await self._weekly_review_exact(session)
            final_user = await self._weekly_review_access(update, fail_closed=True)
            final_exact = await self._weekly_review_exact(session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review launch failed operation=pre_send_fence error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_clear_exact_safely(session)
            return None
        if (
            not self._weekly_review_access_matches(session, first_user)
            or not self._weekly_review_access_matches(session, final_user)
            or first_exact is None
            or final_exact is None
        ):
            await self._weekly_review_clear_exact_safely(session)
            return None
        return final_user

    async def _weekly_review_exact(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> WeeklyReviewSessionSnapshot | None:
        result = await self.weekly_review_service.get_session_exact(
            telegram_actor_id=session.telegram_user_id,
            chat_id=session.chat_id,
            expected_access_version=session.access_version,
            session_public_id=session.public_id,
            expected_session_version=session.version,
            canonical_message_id=session.canonical_message_id,
        )
        return result.session if result.status == "found" else None

    async def _weekly_review_clear_exact(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> bool:
        return await self.weekly_review_service.clear_session_exact(
            telegram_actor_id=session.telegram_user_id,
            chat_id=session.chat_id,
            session_public_id=session.public_id,
            expected_session_version=session.version,
        )

    async def _weekly_review_clear_exact_safely(
        self,
        session: WeeklyReviewSessionSnapshot,
    ) -> bool:
        try:
            return await self._weekly_review_clear_exact(session)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review clear failed operation=clear_exact error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _weekly_review_current(
        self,
        update: Update,
        user: User | None,
        *,
        fail_closed: bool = False,
    ) -> WeeklyReviewSessionSnapshot | None:
        if (
            user is None
            or update.effective_chat is None
            or not self.weekly_review_policy.allows_actor(user)
        ):
            return None
        try:
            result = await self.weekly_review_service.current_session(
                telegram_actor_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                expected_access_version=user.access_version,
            )
            return result.session if result.status == "found" else None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review session lookup failed operation=current error_type=%s",
                type(exc).__name__,
            )
            if fail_closed:
                raise _WeeklyReviewLookupError from None
            return None

    async def _weekly_review_route_snapshot(
        self,
        update: Update,
    ) -> tuple[User | None, WeeklyReviewSessionSnapshot | None]:
        user = await self._weekly_review_access(update, fail_closed=True)
        current = (
            await self._weekly_review_current(update, user, fail_closed=True)
            if user is not None
            else None
        )
        return user, current

    async def _weekly_review_access(
        self,
        update: Update,
        *,
        fail_closed: bool = False,
    ) -> User | None:
        telegram_user = update.effective_user
        chat = update.effective_chat
        if telegram_user is None or chat is None:
            return None
        return await self._weekly_review_access_values(
            telegram_user.id,
            chat.id,
            fail_closed=fail_closed,
        )

    async def _weekly_review_access_values(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        fail_closed: bool = False,
    ) -> User | None:
        if not self.weekly_review_policy.enabled or telegram_user_id <= 0 or chat_id <= 0:
            return None
        try:
            status = await self.access_service.status(telegram_user_id)
            if status is None or not self.weekly_review_policy.allows_tier(status.access_tier):
                return None
            async with self.db.sessions() as session:
                user = await session.scalar(
                    select(User).where(
                        User.telegram_id == telegram_user_id,
                        User.access_version == status.access_version,
                        User.access_tier == status.access_tier,
                    )
                )
            return user if self.weekly_review_policy.allows_actor(user) else None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review access lookup failed operation=access error_type=%s",
                type(exc).__name__,
            )
            if fail_closed:
                raise _WeeklyReviewLookupError from None
            return None

    def _weekly_review_access_matches(
        self,
        session: WeeklyReviewSessionSnapshot,
        user: User | None,
    ) -> bool:
        return bool(
            self.weekly_review_policy.allows_actor(
                user,
                expected_access_version=session.access_version,
            )
            and user is not None
            and user.id == session.owner_id
            and user.telegram_id == session.telegram_user_id
        )

    async def _weekly_review_policy_allows_update(self, update: Update) -> bool:
        if not self.weekly_review_policy.enabled:
            return False
        return await self._weekly_review_access(update) is not None

    def weekly_review_available_for_tier(self, tier: str | None) -> bool:
        return self.weekly_review_policy.allows_tier(tier)

    @staticmethod
    async def _weekly_review_reply_unavailable(message: Any | None) -> None:
        reply = getattr(message, "reply_text", None)
        if callable(reply):
            await reply(WEEKLY_REVIEW_UNAVAILABLE_TEXT)

    @staticmethod
    def _weekly_review_same_session(
        current: WeeklyReviewSessionSnapshot | None,
        expected: WeeklyReviewSessionSnapshot | None,
    ) -> bool:
        if current is None or expected is None:
            return current is expected
        return bool(
            current.public_id == expected.public_id
            and current.version == expected.version
            and current.access_version == expected.access_version
            and current.canonical_message_id == expected.canonical_message_id
        )

    @staticmethod
    def _weekly_review_same_user_generation(
        current: User | None,
        expected: User | None,
    ) -> bool:
        return bool(
            current is not None
            and expected is not None
            and current.id == expected.id
            and current.telegram_id == expected.telegram_id
            and current.access_tier == expected.access_tier
            and current.access_version == expected.access_version
        )

    def _weekly_review_ui_lock(
        self,
        chat_id: int,
        canonical_message_id: int,
    ) -> asyncio.Lock:
        key = (chat_id, canonical_message_id)
        lock = self._weekly_review_ui_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._weekly_review_ui_locks[key] = lock
        return lock

    def _weekly_review_track_task(self, task: asyncio.Task[Any]) -> None:
        self._weekly_review_tasks.add(task)

        def observe(done: asyncio.Task[Any]) -> None:
            self._weekly_review_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "Weekly review task failed operation=lifecycle error_type=%s",
                    type(exc).__name__,
                )

        task.add_done_callback(observe)

    async def _weekly_review_reply_keyboard_owned(self, user: User) -> bool:
        owns_keyboard = getattr(self, "_weekly_review_has_reply_keyboard_owner", None)
        return bool(callable(owns_keyboard) and await owns_keyboard(user))

    async def _weekly_review_launch_message(
        self,
        update: Update,
        source_message: Any | None,
    ) -> Any | None:
        if source_message is None or source_message is update.effective_message:
            return source_message or update.effective_message
        sender = getattr(source_message, "from_user", None)
        if not bool(getattr(sender, "is_bot", False)):
            return source_message
        # A bot-authored STT progress message cannot remove a persistent reply
        # keyboard by editing. Retire it, then use one fresh canonical reply
        # with ReplyKeyboardRemove and edit that same message to inline UI.
        await self._weekly_review_retire_transient(source_message)
        return update.effective_message

    async def _weekly_review_send_keyboard_cleanup(
        self,
        message: Any,
        text: str,
        markup: InlineKeyboardMarkup | None,
    ) -> Any | None:
        if message is None:
            return None
        sender = getattr(message, "from_user", None)
        if bool(getattr(sender, "is_bot", False)):
            try:
                await message.edit_text(text, reply_markup=markup)
                return message
            except asyncio.CancelledError:
                raise
            except (TelegramError, TypeError, AttributeError) as exc:
                logger.warning(
                    "Weekly review keyboard cleanup failed operation=edit error_type=%s",
                    type(exc).__name__,
                )
                return None
        reply = getattr(message, "reply_text", None)
        if not callable(reply):
            return None
        try:
            # Telegram cannot replace a persistent reply keyboard and attach an
            # inline keyboard in one operation.  This is the single canonical
            # send; callers add inline controls by editing the returned message.
            return await reply(text, reply_markup=ReplyKeyboardRemove())
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Weekly review keyboard cleanup failed operation=send error_type=%s",
                type(exc).__name__,
            )
            return None

    async def _weekly_review_edit_query(
        self,
        query: Any,
        text: str,
        markup: InlineKeyboardMarkup | None,
        *,
        operation: str,
    ) -> bool:
        message = getattr(query, "message", None)
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        message_id = getattr(message, "message_id", None)
        if isinstance(chat_id, int) and isinstance(message_id, int):
            async with self._weekly_review_ui_lock(chat_id, message_id):
                return await self._weekly_review_edit_query_unlocked(
                    query,
                    text,
                    markup,
                    operation=operation,
                )
        return await self._weekly_review_edit_query_unlocked(
            query,
            text,
            markup,
            operation=operation,
        )

    @staticmethod
    async def _weekly_review_edit_query_unlocked(
        query: Any,
        text: str,
        markup: InlineKeyboardMarkup | None,
        *,
        operation: str,
    ) -> bool:
        try:
            await query.edit_message_text(
                text,
                reply_markup=markup,
                parse_mode=None,
            )
            return True
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if "message is not modified" in str(exc).casefold():
                return True
            logger.warning(
                "Weekly review callback edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Weekly review callback edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False

    async def _weekly_review_edit_unbound(
        self,
        context: Any,
        *,
        chat_id: int,
        message_id: int,
        text: str,
        markup: InlineKeyboardMarkup | None,
        source_message: Any | None,
    ) -> bool:
        async with self._weekly_review_ui_lock(chat_id, message_id):
            return await self._weekly_review_edit_unbound_locked(
                context,
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                markup=markup,
                source_message=source_message,
            )

    @staticmethod
    async def _weekly_review_edit_unbound_locked(
        context: Any,
        *,
        chat_id: int,
        message_id: int,
        text: str,
        markup: InlineKeyboardMarkup | None,
        source_message: Any | None,
    ) -> bool:
        try:
            bot = getattr(context, "bot", None)
            edit = getattr(bot, "edit_message_text", None)
            if callable(edit):
                await edit(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text,
                    reply_markup=markup,
                )
                return True
            if (
                source_message is not None
                and getattr(source_message, "message_id", None) == message_id
                and hasattr(source_message, "edit_text")
            ):
                await source_message.edit_text(text, reply_markup=markup)
                return True
            return False
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if "message is not modified" in str(exc).casefold():
                return True
            logger.warning(
                "Weekly review launch edit failed operation=edit error_type=%s",
                type(exc).__name__,
            )
            return False
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Weekly review launch edit failed operation=edit error_type=%s",
                type(exc).__name__,
            )
            return False

    @staticmethod
    async def _weekly_review_retire_transient(message: Any) -> None:
        if message is None:
            return
        delete = getattr(message, "delete", None)
        if not callable(delete):
            return
        try:
            await delete()
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Weekly review transient cleanup failed operation=delete error_type=%s",
                type(exc).__name__,
            )

    async def weekly_review_sync_access(
        self,
        user: User,
        chat_id: int,
        *,
        context: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        if not self.weekly_review_policy.enabled:
            return
        previous = getattr(self, "_access_scope_cache", {}).get(chat_id)
        was_allowed = bool(
            previous is not None and self.weekly_review_policy.allows_tier(previous[0])
        )
        if not self.weekly_review_policy.allows_actor(user) and not was_allowed:
            return
        task = asyncio.create_task(
            self._weekly_review_sync_access_lifecycle(
                user,
                chat_id,
                context=context,
                source_message=source_message,
            ),
            name="weekly-review-access-sync-lifecycle",
        )
        self._weekly_review_track_task(task)
        await asyncio.shield(task)

    async def _weekly_review_sync_access_lifecycle(
        self,
        user: User,
        chat_id: int,
        *,
        context: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        retire = getattr(self.weekly_review_service, "retire_access_changed_session", None)
        if not callable(retire):
            return
        try:
            async with self._weekly_review_launch_lock:
                result = await retire(
                    telegram_actor_id=user.telegram_id,
                    chat_id=chat_id,
                    current_access_version=user.access_version,
                )
                if result.status != "access_changed" or result.session is None:
                    return
                await self.weekly_review_capabilities.revoke_session(result.session.public_id)
                if context is not None and result.session.canonical_message_id is not None:
                    async with self._weekly_review_ui_lock(
                        result.session.chat_id,
                        result.session.canonical_message_id,
                    ):
                        await self._weekly_review_compensate(
                            context,
                            result.session,
                            query=None,
                            source_message=source_message,
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review access sync failed operation=sync error_type=%s",
                type(exc).__name__,
            )
            return

    async def weekly_review_scheduled_notification(
        self,
        bot: Any,
        telegram_id: int,
        timezone_name: str,
    ) -> None:
        """Deliver one scheduled launch screen; never create a durable session."""

        if not self.weekly_review_policy.enabled:
            return
        del timezone_name  # The trusted actor timezone is re-read below.
        try:
            primary = await self._weekly_review_scheduled_owned_send(bot, telegram_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review notification failed operation=pre_send error_type=%s",
                type(exc).__name__,
            )
            return
        if primary is None:
            return
        sent, frozen_user, frozen_week_start, text, message_id = primary
        lifecycle = self._weekly_review_scheduled_post_send_lifecycle(
            bot,
            sent,
            frozen_user=frozen_user,
            frozen_week_start=frozen_week_start,
            text=text,
            message_id=message_id,
        )
        try:
            task = asyncio.create_task(
                lifecycle,
                name="weekly-review-scheduled-post-send-lifecycle",
            )
        except Exception as exc:
            lifecycle.close()
            logger.warning(
                "Weekly review notification failed operation=schedule error_type=%s",
                type(exc).__name__,
            )
            await self._weekly_review_neutralize_sent(bot, telegram_id, message_id)
            return
        self._weekly_review_track_task(task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review notification failed operation=lifecycle error_type=%s",
                type(exc).__name__,
            )

    async def _weekly_review_scheduled_owned_send(
        self,
        bot: Any,
        telegram_id: int,
    ) -> tuple[Any, User, date, str, int] | None:
        async with self._reply_keyboard_owner_lock:
            return await self._weekly_review_scheduled_send(bot, telegram_id)

    async def _weekly_review_scheduled_send(
        self,
        bot: Any,
        telegram_id: int,
    ) -> tuple[Any, User, date, str, int] | None:
        if not self.weekly_review_policy.enabled:
            return None
        try:
            initial = await self._weekly_review_access_values(
                telegram_id,
                telegram_id,
                fail_closed=True,
            )
        except _WeeklyReviewLookupError:
            return None
        if initial is None:
            return None
        owns_keyboard = getattr(self, "_weekly_review_has_reply_keyboard_owner", None)
        if callable(owns_keyboard) and await owns_keyboard(initial):
            return None
        try:
            pre_send = await self._weekly_review_access_values(
                telegram_id,
                telegram_id,
                fail_closed=True,
            )
        except _WeeklyReviewLookupError:
            return None
        if not self._weekly_review_same_user_generation(pre_send, initial):
            return None
        assert pre_send is not None
        initial_week_start = self.weekly_review_service.target_week_start(
            pre_send.timezone,
            scheduled=True,
            review_weekday=self.settings.weekly_review_weekday,
        )
        initial_week = self.weekly_review_service.week_range(initial_week_start)
        text = (
            "🧭 Обзор недели\n\n"
            f"Неделя: {self._weekly_review_date_range(initial_week.start, initial_week.end)}\n\n"
            "Пора спокойно посмотреть на неделю 🌿\n"
            "Выберем один ориентир и отдельно проверим напоминания.\n"
            "Ничего не изменю без подтверждения."
        )
        if callable(owns_keyboard) and await owns_keyboard(pre_send):
            return None
        try:
            send_fence = await self._weekly_review_access_values(
                telegram_id,
                telegram_id,
                fail_closed=True,
            )
        except _WeeklyReviewLookupError:
            return None
        if not self._weekly_review_same_user_generation(send_fence, pre_send):
            return None
        assert send_fence is not None
        if not self._weekly_review_scheduled_week_is_live(
            send_fence.timezone,
            initial_week_start,
        ):
            return None
        try:
            sent = await bot.send_message(
                chat_id=send_fence.telegram_id,
                text=text,
                reply_markup=ReplyKeyboardRemove(),
            )
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Weekly review notification failed operation=send error_type=%s",
                type(exc).__name__,
            )
            return None
        message_id = getattr(sent, "message_id", None)
        if not isinstance(message_id, int) or message_id <= 0:
            logger.warning(
                "Weekly review notification failed operation=send_binding "
                "error_type=MissingMessageId"
            )
            return None
        return sent, send_fence, initial_week_start, text, message_id

    async def _weekly_review_scheduled_post_send_lifecycle(
        self,
        bot: Any,
        sent: Any,
        *,
        frozen_user: User,
        frozen_week_start: date,
        text: str,
        message_id: int,
    ) -> None:
        issued_tokens: list[str] = []
        async with self._weekly_review_launch_lock:
            try:
                delivered = await self._weekly_review_scheduled_post_send(
                    bot,
                    sent,
                    frozen_user=frozen_user,
                    frozen_week_start=frozen_week_start,
                    text=text,
                    message_id=message_id,
                    issued_tokens=issued_tokens,
                )
            except asyncio.CancelledError as exc:
                logger.warning(
                    "Weekly review notification failed operation=post_send error_type=%s",
                    type(exc).__name__,
                )
                self._weekly_review_schedule_scheduled_cleanup(
                    bot,
                    frozen_user=frozen_user,
                    frozen_week_start=frozen_week_start,
                    message_id=message_id,
                    issued_tokens=tuple(issued_tokens),
                )
                raise
            except Exception as exc:
                logger.warning(
                    "Weekly review notification failed operation=post_send error_type=%s",
                    type(exc).__name__,
                )
                delivered = False
            if delivered:
                return
            try:
                await self._weekly_review_scheduled_cleanup_locked(
                    bot,
                    frozen_user=frozen_user,
                    frozen_week_start=frozen_week_start,
                    message_id=message_id,
                    issued_tokens=tuple(issued_tokens),
                )
            except asyncio.CancelledError:
                self._weekly_review_schedule_scheduled_cleanup(
                    bot,
                    frozen_user=frozen_user,
                    frozen_week_start=frozen_week_start,
                    message_id=message_id,
                    issued_tokens=tuple(issued_tokens),
                )
                raise

    async def _weekly_review_scheduled_post_send(
        self,
        bot: Any,
        sent: Any,
        *,
        frozen_user: User,
        frozen_week_start: date,
        text: str,
        message_id: int,
        issued_tokens: list[str],
    ) -> bool:
        owns_keyboard = getattr(self, "_weekly_review_has_reply_keyboard_owner", None)
        final = await self._weekly_review_access_values(
            frozen_user.telegram_id,
            frozen_user.telegram_id,
            fail_closed=True,
        )
        if (
            not self._weekly_review_same_user_generation(final, frozen_user)
            or final is None
            or (callable(owns_keyboard) and await owns_keyboard(final))
            or not self._weekly_review_scheduled_week_is_live(
                final.timezone,
                frozen_week_start,
            )
        ):
            return False
        focus = await self.weekly_review_service.get_focus(
            telegram_actor_id=final.telegram_id,
            expected_access_version=final.access_version,
            week_start=frozen_week_start,
        )
        tokens = await self.weekly_review_capabilities.issue(
            actions=("start", "focus", "reminders", "close"),
            owner_id=final.id,
            telegram_user_id=final.telegram_id,
            chat_id=final.telegram_id,
            canonical_message_id=message_id,
            access_version=final.access_version,
            week_start=frozen_week_start,
            scheduled=True,
        )
        issued_tokens.extend(tokens.values())
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🧭 Начать обзор",
                        callback_data=f"wrev:{tokens['start']}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        "✏️ Изменить фокус" if focus.focus is not None else "🎯 Фокус недели",
                        callback_data=f"wrev:{tokens['focus']}",
                    ),
                    InlineKeyboardButton(
                        "🔔 Напоминания",
                        callback_data=f"wrev:{tokens['reminders']}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Не сейчас",
                        callback_data=f"wrev:{tokens['close']}",
                    )
                ],
            ]
        )
        async with self._weekly_review_ui_lock(frozen_user.telegram_id, message_id):
            before_edit = await self._weekly_review_access_values(
                frozen_user.telegram_id,
                frozen_user.telegram_id,
                fail_closed=True,
            )
            if (
                not self._weekly_review_same_user_generation(before_edit, final)
                or before_edit is None
                or (callable(owns_keyboard) and await owns_keyboard(before_edit))
                or not self._weekly_review_scheduled_week_is_live(
                    before_edit.timezone,
                    frozen_week_start,
                )
            ):
                return False
            context = type("WeeklyContext", (), {"bot": bot})()
            edited = await self._weekly_review_edit_unbound_locked(
                context,
                chat_id=frozen_user.telegram_id,
                message_id=message_id,
                text=text,
                markup=markup,
                source_message=sent,
            )
            if not edited:
                return False
            after_edit = await self._weekly_review_access_values(
                frozen_user.telegram_id,
                frozen_user.telegram_id,
                fail_closed=True,
            )
            return bool(
                self._weekly_review_same_user_generation(after_edit, final)
                and after_edit is not None
                and not (callable(owns_keyboard) and await owns_keyboard(after_edit))
                and self._weekly_review_scheduled_week_is_live(
                    after_edit.timezone,
                    frozen_week_start,
                )
            )

    def _weekly_review_schedule_scheduled_cleanup(
        self,
        bot: Any,
        *,
        frozen_user: User,
        frozen_week_start: date,
        message_id: int,
        issued_tokens: tuple[str, ...],
    ) -> None:
        lifecycle = self._weekly_review_scheduled_cleanup_lifecycle(
            bot,
            frozen_user=frozen_user,
            frozen_week_start=frozen_week_start,
            message_id=message_id,
            issued_tokens=issued_tokens,
        )
        try:
            task = asyncio.create_task(
                lifecycle,
                name="weekly-review-scheduled-cleanup-lifecycle",
            )
        except Exception as exc:
            lifecycle.close()
            logger.warning(
                "Weekly review notification cleanup failed operation=schedule error_type=%s",
                type(exc).__name__,
            )
            return
        self._weekly_review_track_task(task)

    async def _weekly_review_scheduled_cleanup_lifecycle(
        self,
        bot: Any,
        *,
        frozen_user: User,
        frozen_week_start: date,
        message_id: int,
        issued_tokens: tuple[str, ...],
    ) -> None:
        async with self._weekly_review_launch_lock:
            await self._weekly_review_scheduled_cleanup_locked(
                bot,
                frozen_user=frozen_user,
                frozen_week_start=frozen_week_start,
                message_id=message_id,
                issued_tokens=issued_tokens,
            )

    async def _weekly_review_scheduled_cleanup_locked(
        self,
        bot: Any,
        *,
        frozen_user: User,
        frozen_week_start: date,
        message_id: int,
        issued_tokens: tuple[str, ...],
    ) -> None:
        async with self._weekly_review_ui_lock(frozen_user.telegram_id, message_id):
            exact_screen_current = not issued_tokens
            if issued_tokens:
                try:
                    exact_screen_current = (
                        await self.weekly_review_capabilities.revoke_tokens_if_current_screen(
                            issued_tokens,
                            owner_id=frozen_user.id,
                            telegram_user_id=frozen_user.telegram_id,
                            chat_id=frozen_user.telegram_id,
                            canonical_message_id=message_id,
                            access_version=frozen_user.access_version,
                            week_start=frozen_week_start,
                            scheduled=True,
                        )
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Weekly review notification cleanup failed "
                        "operation=capability_revoke error_type=%s",
                        type(exc).__name__,
                    )
                    return
            if exact_screen_current:
                await self._weekly_review_neutralize_sent(
                    bot,
                    frozen_user.telegram_id,
                    message_id,
                )

    def _weekly_review_scheduled_week_is_live(
        self,
        timezone_name: str,
        frozen_week_start: date,
    ) -> bool:
        current_week = self.weekly_review_service.target_week_start(
            timezone_name,
            scheduled=False,
            review_weekday=self.settings.weekly_review_weekday,
        )
        return frozen_week_start in {current_week, current_week + timedelta(days=7)}

    @staticmethod
    async def _weekly_review_neutralize_sent(
        bot: Any,
        chat_id: int,
        message_id: int,
    ) -> None:
        try:
            deleted = await bot.delete_message(chat_id=chat_id, message_id=message_id)
            if deleted is not False:
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review notification cleanup failed operation=delete error_type=%s",
                type(exc).__name__,
            )
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=WEEKLY_REVIEW_ACCESS_CHANGED_TEXT,
                reply_markup=None,
                parse_mode=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Weekly review notification cleanup failed operation=edit error_type=%s",
                type(exc).__name__,
            )
