from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from .access import ADMIN, SUBSCRIBER, AccessTier, require_access_tier
from .models import User
from .nova_memory import (
    NovaMemoryPage,
    NovaMemoryService,
    NovaMemorySnapshot,
    NovaMemoryStorageError,
    NovaMemoryValidationError,
    normalize_nova_memory_content,
)
from .nova_memory_flow import (
    NovaMemoryCapabilityClaim,
    NovaMemoryFlowPhase,
    NovaMemoryFlowSession,
    NovaMemoryIntentKind,
    NovaMemoryIntentResult,
    classify_nova_memory_intent,
)

logger = logging.getLogger(__name__)

NOVA_MEMORY_ROOT_TEXT = """🧬 Моя Nova

Здесь хранится только то, что ты сам попросил Nova запомнить и подтвердил.

Обычные сообщения сюда не попадают.
Сохранено: {count} из {max_items}."""

NOVA_MEMORY_CREATE_TEXT = """➕ Научить Nova

Напиши или скажи голосом одну вещь, которую Nova должна помнить.

До 500 символов. Перед сохранением можно изменить текст, раздел и отметку «Важное»."""

NOVA_MEMORY_HELP_TEXT = """❓ Как работает «Моя Nova»

• Nova хранит только то, что ты явно попросил запомнить и подтвердил.
• Обычная переписка сама в постоянную память не попадает.
• «Обо мне» — факты и предпочтения.
• «Как со мной работать» — удобный стиль общения.
• «Мои ориентиры» — ценности, приоритеты и направления.
• ⭐ Важное — особенно значимые записи.

Перед сохранением можно изменить текст, раздел и важность. Любую запись можно изменить или забыть."""

NOVA_MEMORY_ACCESS_CHANGED_TEXT = (
    "🧬 Моя Nova\n\nДоступ изменился. Содержимое памяти скрыто. Открой /mynova заново."
)
NOVA_MEMORY_STALE_ALERT = "Эта кнопка устарела или недоступна."
NOVA_MEMORY_BUSY_ALERT = "Nova уже обрабатывает это действие."
NOVA_MEMORY_UNAVAILABLE_TEXT = "Функция «Моя Nova» сейчас недоступна."
NOVA_MEMORY_STORAGE_FAILURE_TEXT = (
    "Не удалось выполнить действие с памятью. Ничего не повторяю автоматически — "
    "открой /mynova и проверь актуальное состояние."
)
NOVA_MEMORY_VALIDATION_FAILURE_TEXT = (
    "Не получилось подготовить эту запись. Пришли текст от 1 до 500 символов."
)
NOVA_MEMORY_REFERENCE_MISSING_TEXT = (
    "Не вижу подходящего предыдущего сообщения. Пришли нужную формулировку после "
    "«Nova, запомни: …»."
)
NOVA_MEMORY_MEDIA_TEXT = "Пришли текст или голосовое сообщение."
NOVA_MEMORY_VOICE_ACCESS_CHANGED_TEXT = (
    "Доступ изменился во время распознавания голоса. Повтори сообщение после обновления доступа."
)

NOVA_MEMORY_PAGE_SIZE = 5
_CATEGORY_LABELS = {
    "about_me": "📖 Обо мне",
    "interaction": "⚙️ Как со мной работать",
    "orientation": "🧭 Мои ориентиры",
}
_FILTER_TITLES = {
    "about_me": "📖 Что Nova знает обо мне",
    "interaction": "⚙️ Как со мной работать",
    "orientation": "🧭 Мои ориентиры",
    "important": "⭐ Важное",
}
_ACTIVE_PHASES = frozenset(
    {
        NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
        NovaMemoryFlowPhase.CREATE_PREVIEW,
        NovaMemoryFlowPhase.AWAITING_UPDATE_CONTENT,
        NovaMemoryFlowPhase.UPDATE_PREVIEW,
        NovaMemoryFlowPhase.DELETE_PREVIEW,
        NovaMemoryFlowPhase.DELETE_ALL_PREVIEW,
        NovaMemoryFlowPhase.PROCESSING,
    }
)
_AWAITING_PHASES = frozenset(
    {
        NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
        NovaMemoryFlowPhase.AWAITING_UPDATE_CONTENT,
    }
)


@dataclass(frozen=True, slots=True)
class NovaMemoryVoiceFence:
    """Privacy-safe generation captured before Telegram download and STT."""

    owner_id: int
    telegram_user_id: int
    chat_id: int
    tier: AccessTier
    access_version: int
    session: NovaMemoryFlowSession | None


@dataclass(frozen=True, slots=True)
class _PageView:
    items: tuple[NovaMemorySnapshot, ...]
    total: int
    page: int
    pages: int
    collection_revision: str


_INVALID_EXPLICIT_PREFIXES = (
    (
        re.compile(
            r"\s*(?:nova|нова)\s*,\s*запомни\s*,\s*как\s+со\s+мной\s+работать\s*:",
            re.IGNORECASE,
        ),
        "interaction",
        False,
    ),
    (
        re.compile(
            r"\s*(?:nova|нова)\s*,\s*запомни\s+мой\s+ориентир\s*:",
            re.IGNORECASE,
        ),
        "orientation",
        False,
    ),
    (
        re.compile(
            r"\s*(?:nova|нова)\s*,\s*сохрани\s+в\s+важное\s*:",
            re.IGNORECASE,
        ),
        "about_me",
        True,
    ),
    (
        re.compile(
            r"\s*(?:(?:nova|нова)\s*,\s*запомни(?:\s+обо\s+мне)?|"
            r"научи\s+nova|запомни\s+для\s+nova)\b",
            re.IGNORECASE,
        ),
        "about_me",
        False,
    ),
)


def _safe_nova_memory_intent(text: str) -> tuple[NovaMemoryIntentResult, bool]:
    try:
        return classify_nova_memory_intent(text), False
    except NovaMemoryValidationError:
        for pattern, category, important in _INVALID_EXPLICIT_PREFIXES:
            if pattern.match(text):
                return (
                    NovaMemoryIntentResult(
                        NovaMemoryIntentKind.AWAIT_CONTENT,
                        category=category,
                        important=important,
                    ),
                    True,
                )
        return NovaMemoryIntentResult(NovaMemoryIntentKind.NONE), False


class NovaMemoryHandlers:
    """Canonical, owner-bound Telegram UI for explicit durable Nova memory."""

    nova_memory_service: NovaMemoryService
    nova_memory_sessions: Any
    _nova_memory_launch_lock: asyncio.Lock
    _nova_memory_ui_lock: asyncio.Lock

    def nova_memory_available_for_tier(self, tier: str) -> bool:
        if not bool(getattr(self.settings, "enable_nova_memory", False)):
            return False
        if tier not in {SUBSCRIBER, ADMIN}:
            return False
        return not bool(getattr(self.settings, "nova_memory_admin_only", True)) or tier == ADMIN

    async def _nova_memory_identity(self, update: Any) -> User | None:
        telegram_user = getattr(update, "effective_user", None)
        chat = getattr(update, "effective_chat", None)
        telegram_id = getattr(telegram_user, "id", None)
        chat_id = getattr(chat, "id", None)
        if not self._positive_id(telegram_id) or not self._positive_id(chat_id):
            return None
        try:
            async with self.db.sessions() as session:
                return await session.scalar(select(User).where(User.telegram_id == telegram_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Nova memory identity lookup failed error_type=%s", type(exc).__name__)
            return None

    async def _nova_memory_access(self, update: Any) -> User | None:
        user = await self._nova_memory_identity(update)
        if user is None or not self.nova_memory_available_for_tier(user.access_tier):
            return None
        try:
            status = await self.access_service.status(user.telegram_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Nova memory access lookup failed error_type=%s", type(exc).__name__)
            return None
        if (
            status is None
            or status.access_tier != user.access_tier
            or status.access_version != user.access_version
            or not self.nova_memory_available_for_tier(status.access_tier)
        ):
            return None
        return user

    async def nova_memory_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        async with self._nova_memory_launch_lock:
            user = await self._nova_memory_access(update)
            if user is None:
                await self._nova_memory_reply_unavailable(update.effective_message)
                return True
            await self._nova_memory_handoff(update, user)
            current = await self.nova_memory_sessions.current(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
            )
            if current is not None and self._nova_memory_same_access(current, user):
                await self._nova_memory_show_root(
                    update,
                    context,
                    current,
                    source_message=update.effective_message,
                )
                await self._nova_memory_handoff(update, user)
                return True
            return await self._nova_memory_open_new_canonical(update, user)

    mynova_command = nova_memory_command

    async def nova_memory_open_from_navigation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        query = update.callback_query
        if query is None:
            return await self.nova_memory_command(update, context)
        async with self._nova_memory_launch_lock:
            user = await self._nova_memory_access(update)
            message_id = self._positive_message_id(getattr(query.message, "message_id", None))
            if user is None or message_id is None:
                await self._nova_memory_answer(query, NOVA_MEMORY_STALE_ALERT, show_alert=True)
                return True
            await self._nova_memory_answer(query)
            await self._nova_memory_handoff(update, user)
            async with self._nova_memory_ui_lock:
                session = await self.nova_memory_sessions.create(
                    owner_id=user.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    tier=require_access_tier(user.access_tier),
                    access_version=user.access_version,
                    canonical_message_id=message_id,
                    phase=NovaMemoryFlowPhase.ROOT,
                )
            await self._nova_memory_handoff(update, user)
            await self._nova_memory_show_root(update, context, session, query=query)
        return True

    nova_memory_navigation_entry = nova_memory_open_from_navigation

    async def _nova_memory_open_new_canonical(self, update: Any, user: User) -> bool:
        try:
            sent = await update.effective_message.reply_text(
                "🧬 Моя Nova\n\nОткрываю…",
                parse_mode=None,
            )
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Nova memory canonical creation failed error_type=%s", type(exc).__name__
            )
            return True
        message_id = self._positive_message_id(getattr(sent, "message_id", None))
        if message_id is None:
            logger.warning("Nova memory canonical creation failed error_type=MissingMessageId")
            return True
        async with self._nova_memory_ui_lock:
            session = await self.nova_memory_sessions.create(
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                tier=require_access_tier(user.access_tier),
                access_version=user.access_version,
                canonical_message_id=message_id,
                phase=NovaMemoryFlowPhase.ROOT,
            )
        await self._nova_memory_handoff(update, user)
        await self._nova_memory_show_root(update, None, session, source_message=sent)
        return True

    async def nova_memory_voice_fence(
        self,
        update: Update,
        *,
        user: User | None = None,
    ) -> NovaMemoryVoiceFence | None:
        actor = user or await self._nova_memory_access(update)
        if actor is None or not self.nova_memory_available_for_tier(actor.access_tier):
            return None
        session = await self.nova_memory_sessions.current(
            owner_id=actor.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        return NovaMemoryVoiceFence(
            owner_id=actor.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            tier=require_access_tier(actor.access_tier),
            access_version=actor.access_version,
            session=session,
        )

    async def nova_memory_owns_text(
        self,
        update: Update,
        text: str,
        user: User | None = None,
    ) -> bool:
        intent, _invalid = _safe_nova_memory_intent(text)
        if intent.kind is not NovaMemoryIntentKind.NONE:
            return True
        actor = user or await self._nova_memory_identity(update)
        if actor is None:
            return False
        current = await self.nova_memory_sessions.current(
            owner_id=actor.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        return current is not None and current.phase in _ACTIVE_PHASES

    async def nova_memory_blocks_navigation(self, update: Update) -> bool:
        user = await self._nova_memory_identity(update)
        if user is None:
            return False
        current = await self.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        return current is not None and current.phase in _ACTIVE_PHASES

    async def nova_memory_public_command_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        user = await self._nova_memory_identity(update)
        if user is None:
            return False
        current = await self.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        if current is None:
            return False
        if current.phase not in _ACTIVE_PHASES:
            await self.nova_memory_clear_bound(
                current.owner_id,
                current.telegram_user_id,
                current.chat_id,
                session_id=current.id,
            )
            return False
        if current.phase is NovaMemoryFlowPhase.PROCESSING:
            await self._nova_memory_deliver(
                context,
                current,
                "🧬 Моя Nova\n\nДействие уже обрабатывается. Подожди и затем проверь актуальную память.",
                None,
                query=None,
                source_message=update.effective_message,
                operation="public_command_processing",
            )
            return True
        fresh = await self._nova_memory_access(update)
        if not self._nova_memory_same_access(current, fresh):
            await self._nova_memory_access_changed(
                context,
                current,
                source_message=update.effective_message,
            )
            return True
        screen = await self.nova_memory_sessions.update(current, phase=current.phase)
        if screen is None:
            return True
        tokens = await self._nova_memory_tokens(
            screen,
            (("continue", False, {}), ("exit_home", False, {})),
        )
        if tokens is None:
            return True
        markup = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("▶️ Продолжить", callback_data=tokens["continue"])],
                [
                    InlineKeyboardButton(
                        "🏠 Выйти в главное меню", callback_data=tokens["exit_home"]
                    )
                ],
            ]
        )
        await self._nova_memory_deliver(
            context,
            screen,
            "🧬 Моя Nova\n\nСейчас не завершено действие с памятью. Что сделать?",
            markup,
            query=None,
            source_message=update.effective_message,
            operation="public_command",
        )
        return True

    async def nova_memory_cancel_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        user = await self._nova_memory_identity(update)
        if user is None:
            return False
        current = await self.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        if current is None or current.phase not in _ACTIVE_PHASES:
            return False
        if current.phase is NovaMemoryFlowPhase.PROCESSING:
            await self._nova_memory_deliver(
                context,
                current,
                "🧬 Моя Nova\n\nДействие уже обрабатывается. Отмена сейчас недоступна.",
                None,
                query=None,
                source_message=update.effective_message,
                operation="cancel_processing",
            )
            return True
        async with self._nova_memory_launch_lock:
            await self._nova_memory_retire_and_edit(
                context,
                current,
                "🧬 Моя Nova\n\nДействие отменено. Ничего не сохранено.",
                None,
                query=None,
                source_message=update.effective_message,
                operation="cancel",
            )
        return True

    async def nova_memory_clear_current(self, update: Update) -> None:
        user = await self._nova_memory_identity(update)
        if user is None:
            return
        # The launch lock is the mutual-flow handoff fence: callers that are
        # opening Reminder/Nova wait until a concurrent memory launch has either
        # published its session or aborted, then clear that exact generation.
        async with self._nova_memory_launch_lock:
            async with self._nova_memory_ui_lock:
                current = await self.nova_memory_sessions.current(
                    owner_id=user.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                )
                if current is not None:
                    await self.nova_memory_sessions.clear(
                        owner_id=current.owner_id,
                        telegram_user_id=current.telegram_user_id,
                        chat_id=current.chat_id,
                        session_id=current.id,
                    )

    async def nova_memory_clear_bound(
        self,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
        *,
        session_id: str | None = None,
    ) -> bool:
        async with self._nova_memory_launch_lock:
            async with self._nova_memory_ui_lock:
                current = await self.nova_memory_sessions.current(
                    owner_id=owner_id,
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                )
                if current is None or (session_id is not None and current.id != session_id):
                    return False
                return await self.nova_memory_sessions.clear(
                    owner_id=owner_id,
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    session_id=current.id,
                )

    async def nova_memory_sync_access(
        self,
        user: User,
        chat_id: int,
        *,
        context: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        current = await self.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
        )
        if current is None:
            return
        if self._nova_memory_same_access(current, user):
            return
        await self._nova_memory_access_changed(
            context,
            current,
            source_message=source_message,
        )

    async def nova_memory_text_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        user: User | None = None,
    ) -> bool:
        text = getattr(update.effective_message, "text", None)
        if not isinstance(text, str):
            return False
        intent, invalid = _safe_nova_memory_intent(text)
        identity = user or await self._nova_memory_identity(update)
        current = None
        if identity is not None:
            current = await self.nova_memory_sessions.current(
                owner_id=identity.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
        owns_active = current is not None and current.phase in _ACTIVE_PHASES
        if intent.kind is NovaMemoryIntentKind.NONE and not owns_active:
            return False

        actor = await self._nova_memory_access(update)
        if actor is None:
            if current is not None:
                await self._nova_memory_access_changed(
                    context,
                    current,
                    source_message=update.effective_message,
                )
            elif intent.kind is not NovaMemoryIntentKind.NONE:
                await self._nova_memory_reply_unavailable(update.effective_message)
            return True
        if current is not None and not self._nova_memory_same_access(current, actor):
            await self._nova_memory_access_changed(
                context,
                current,
                source_message=update.effective_message,
            )
            return True
        if current is not None and current.phase is NovaMemoryFlowPhase.PROCESSING:
            await self._nova_memory_render_phase(
                update,
                context,
                current,
                source_message=update.effective_message,
            )
            return True

        if intent.kind is NovaMemoryIntentKind.NONE:
            assert current is not None
            if current.phase in _AWAITING_PHASES:
                await self._nova_memory_accept_content(
                    update,
                    context,
                    current,
                    text,
                )
            elif current.phase in {
                NovaMemoryFlowPhase.DELETE_PREVIEW,
                NovaMemoryFlowPhase.DELETE_ALL_PREVIEW,
            }:
                # A non-callback message must not invalidate the only visible
                # delete-confirm capabilities; simply keep ownership.
                return True
            else:
                screen = await self.nova_memory_sessions.update(current, phase=current.phase)
                if screen is not None:
                    await self._nova_memory_render_phase(
                        update,
                        context,
                        screen,
                        source_message=update.effective_message,
                    )
            return True
        async with self._nova_memory_launch_lock:
            actor = await self._nova_memory_access(update)
            if actor is None:
                await self._nova_memory_reply_unavailable(update.effective_message)
                return True
            current = await self.nova_memory_sessions.current(
                owner_id=actor.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            await self._nova_memory_handoff(update, actor)
            await self._nova_memory_route_intent(
                update,
                context,
                actor,
                current,
                intent,
                canonical_candidate=None,
                invalid=invalid,
            )
            await self._nova_memory_handoff(update, actor)
        return True

    async def nova_memory_voice_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        transcript: str,
        progress: Any,
        *,
        user: User | None = None,
        expected_session: NovaMemoryFlowSession | None = None,
        expected_access_version: int | None = None,
        fence: NovaMemoryVoiceFence | None = None,
    ) -> bool:
        intent, invalid = _safe_nova_memory_intent(transcript)
        expected = fence.session if fence is not None else expected_session
        expected_owner_id = fence.owner_id if fence is not None else getattr(user, "id", None)
        expected_tier = fence.tier if fence is not None else getattr(user, "access_tier", None)
        expected_version = (
            fence.access_version
            if fence is not None
            else (
                expected_access_version
                if expected_access_version is not None
                else getattr(user, "access_version", None)
            )
        )
        expected_telegram_user_id = (
            fence.telegram_user_id
            if fence is not None
            else getattr(update.effective_user, "id", None)
        )
        identity = await self._nova_memory_identity(update)
        current = None
        if identity is not None:
            current = await self.nova_memory_sessions.current(
                owner_id=identity.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
        expected_active = expected is not None and expected.phase in _ACTIVE_PHASES
        if intent.kind is NovaMemoryIntentKind.NONE and expected is None:
            if not self._nova_memory_same_generation(current, expected):
                if current is not None or expected is not None:
                    await self._nova_memory_delete_transient(progress, context)
                    return True
            return False

        actor = await self._nova_memory_access(update)
        access_matches = self._nova_memory_voice_access_matches(
            actor,
            owner_id=expected_owner_id,
            telegram_user_id=expected_telegram_user_id,
            tier=expected_tier,
            access_version=expected_version,
        )
        # A first explicit voice-memory command has no pre-STT memory fence. It is
        # still safe when the caller froze the same current access generation.
        if expected_owner_id is None and user is not None:
            access_matches = self._nova_memory_voice_access_matches(
                actor,
                owner_id=user.id,
                telegram_user_id=getattr(update.effective_user, "id", None),
                tier=user.access_tier,
                access_version=user.access_version,
            )
        if actor is None or not access_matches:
            if expected is not None:
                await self._nova_memory_delete_transient(progress, context)
                await self._nova_memory_access_changed(
                    context,
                    expected,
                    source_message=update.effective_message,
                )
            else:
                await self._nova_memory_edit_transient(progress, NOVA_MEMORY_ACCESS_CHANGED_TEXT)
            return True
        current = await self.nova_memory_sessions.current(
            owner_id=actor.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        if not self._nova_memory_same_generation(current, expected):
            await self._nova_memory_delete_transient(progress, context)
            return True
        if intent.kind is NovaMemoryIntentKind.NONE and not expected_active:
            # Browsing does not own an ordinary voice message, but its frozen
            # access generation was still revalidated above.
            return False
        if current is not None and current.phase is NovaMemoryFlowPhase.PROCESSING:
            await self._nova_memory_delete_transient(progress, context)
            await self._nova_memory_render_phase(
                update,
                context,
                current,
                source_message=update.effective_message,
            )
            return True

        if intent.kind is NovaMemoryIntentKind.NONE:
            assert current is not None
            await self._nova_memory_delete_transient(progress, context)
            if current.phase in _AWAITING_PHASES:
                await self._nova_memory_accept_content(
                    update,
                    context,
                    current,
                    transcript,
                    source_message=update.effective_message,
                )
            elif current.phase in {
                NovaMemoryFlowPhase.DELETE_PREVIEW,
                NovaMemoryFlowPhase.DELETE_ALL_PREVIEW,
            }:
                return True
            else:
                screen = await self.nova_memory_sessions.update(current, phase=current.phase)
                if screen is not None:
                    await self._nova_memory_render_phase(
                        update,
                        context,
                        screen,
                        source_message=update.effective_message,
                    )
            return True

        async with self._nova_memory_launch_lock:
            first_actor = await self._nova_memory_access(update)
            first_access_matches = self._nova_memory_voice_access_matches(
                first_actor,
                owner_id=expected_owner_id,
                telegram_user_id=expected_telegram_user_id,
                tier=expected_tier,
                access_version=expected_version,
            )
            if expected_owner_id is None and user is not None:
                first_access_matches = self._nova_memory_voice_access_matches(
                    first_actor,
                    owner_id=user.id,
                    telegram_user_id=getattr(update.effective_user, "id", None),
                    tier=user.access_tier,
                    access_version=user.access_version,
                )
            lookup_owner_id = expected_owner_id or getattr(user, "id", None)
            first_current = None
            if lookup_owner_id is not None and expected_telegram_user_id is not None:
                first_current = await self.nova_memory_sessions.current(
                    owner_id=lookup_owner_id,
                    telegram_user_id=expected_telegram_user_id,
                    chat_id=update.effective_chat.id,
                )
            final_actor = await self._nova_memory_access(update)
            final_access_matches = self._nova_memory_voice_access_matches(
                final_actor,
                owner_id=expected_owner_id,
                telegram_user_id=expected_telegram_user_id,
                tier=expected_tier,
                access_version=expected_version,
            )
            if expected_owner_id is None and user is not None:
                final_access_matches = self._nova_memory_voice_access_matches(
                    final_actor,
                    owner_id=user.id,
                    telegram_user_id=getattr(update.effective_user, "id", None),
                    tier=user.access_tier,
                    access_version=user.access_version,
                )
            final_current = None
            if lookup_owner_id is not None and expected_telegram_user_id is not None:
                final_current = await self.nova_memory_sessions.current(
                    owner_id=lookup_owner_id,
                    telegram_user_id=expected_telegram_user_id,
                    chat_id=update.effective_chat.id,
                )
            if not first_access_matches or not final_access_matches:
                if expected is not None:
                    await self._nova_memory_delete_transient(progress, context)
                    await self._nova_memory_access_changed(
                        context,
                        expected,
                        source_message=update.effective_message,
                    )
                else:
                    await self._nova_memory_edit_transient(
                        progress,
                        NOVA_MEMORY_VOICE_ACCESS_CHANGED_TEXT,
                    )
                return True
            if not self._nova_memory_same_generation(
                first_current, expected
            ) or not self._nova_memory_same_generation(final_current, expected):
                await self._nova_memory_delete_transient(progress, context)
                return True
            assert final_actor is not None
            actor = final_actor
            current = final_current
            await self._nova_memory_handoff(update, actor)
            await self._nova_memory_route_intent(
                update,
                context,
                actor,
                current,
                intent,
                canonical_candidate=progress,
                invalid=invalid,
            )
            await self._nova_memory_handoff(update, actor)
        return True

    async def nova_memory_voice_pre_route(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        progress: Any,
        *,
        fence: NovaMemoryVoiceFence | None,
    ) -> bool:
        """Fence the pre-STT memory/access generation before any downstream route."""

        if fence is None:
            return False
        async with self._nova_memory_launch_lock:
            first_actor = await self._nova_memory_access(update)
            first_current = await self.nova_memory_sessions.current(
                owner_id=fence.owner_id,
                telegram_user_id=fence.telegram_user_id,
                chat_id=fence.chat_id,
            )
            final_actor = await self._nova_memory_access(update)
            final_current = await self.nova_memory_sessions.current(
                owner_id=fence.owner_id,
                telegram_user_id=fence.telegram_user_id,
                chat_id=fence.chat_id,
            )
        access_matches = self._nova_memory_voice_access_matches(
            first_actor,
            owner_id=fence.owner_id,
            telegram_user_id=fence.telegram_user_id,
            tier=fence.tier,
            access_version=fence.access_version,
        ) and self._nova_memory_voice_access_matches(
            final_actor,
            owner_id=fence.owner_id,
            telegram_user_id=fence.telegram_user_id,
            tier=fence.tier,
            access_version=fence.access_version,
        )
        if not access_matches:
            if fence.session is not None:
                await self._nova_memory_delete_transient(progress, context)
                await self._nova_memory_access_changed(
                    context,
                    fence.session,
                    source_message=update.effective_message,
                )
            else:
                await self._nova_memory_edit_transient(
                    progress,
                    NOVA_MEMORY_VOICE_ACCESS_CHANGED_TEXT,
                )
            return True
        if not self._nova_memory_same_generation(
            first_current, fence.session
        ) or not self._nova_memory_same_generation(final_current, fence.session):
            await self._nova_memory_delete_transient(progress, context)
            return True
        return False

    async def nova_memory_voice_failure(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        fence: NovaMemoryVoiceFence | None,
        progress: Any | None,
        notice: str,
    ) -> bool:
        """Keep an active memory flow canonical when voice processing cannot continue."""

        expected = fence.session if fence is not None else None
        if expected is None or expected.phase not in _ACTIVE_PHASES:
            return False
        if await self.nova_memory_voice_pre_route(
            update,
            context,
            progress,
            fence=fence,
        ):
            return True
        await self._nova_memory_delete_transient(progress, context)
        async with self._nova_memory_launch_lock:
            live = await self.nova_memory_sessions.get_exact(expected)
            if live is None:
                return True
            if live.phase is NovaMemoryFlowPhase.PROCESSING:
                await self._nova_memory_render_phase(
                    update,
                    context,
                    live,
                    source_message=update.effective_message,
                )
                return True
            screen = await self.nova_memory_sessions.update(live, phase=live.phase)
            if screen is None:
                return True
            tokens = await self._nova_memory_tokens(screen, (("continue", False, {}),))
            if tokens is None:
                return True
            await self._nova_memory_deliver(
                context,
                screen,
                f"🧬 Моя Nova\n\n{notice}",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "▶️ Продолжить",
                                callback_data=tokens["continue"],
                            )
                        ]
                    ]
                ),
                query=None,
                source_message=update.effective_message,
                operation="voice_failure",
            )
        return True

    async def nova_memory_media_gate(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        message = update.effective_message
        if message is None or not (
            getattr(message, "photo", None) or getattr(message, "document", None)
        ):
            return False
        user = await self._nova_memory_identity(update)
        if user is None:
            return False
        current = await self.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
        )
        if current is None or current.phase not in _ACTIVE_PHASES:
            return False
        fresh = await self._nova_memory_access(update)
        if not self._nova_memory_same_access(current, fresh):
            await self._nova_memory_access_changed(context, current, source_message=message)
            return True
        if current.phase is NovaMemoryFlowPhase.PROCESSING:
            await self._nova_memory_render_phase(
                update,
                context,
                current,
                source_message=message,
            )
            return True
        if current.phase in {
            NovaMemoryFlowPhase.DELETE_PREVIEW,
            NovaMemoryFlowPhase.DELETE_ALL_PREVIEW,
        }:
            return True
        screen = await self.nova_memory_sessions.update(current, phase=current.phase)
        if screen is None:
            return True
        if screen.phase in _AWAITING_PHASES:
            await self._nova_memory_render_awaiting(
                context,
                screen,
                notice=NOVA_MEMORY_MEDIA_TEXT,
                source_message=message,
            )
        else:
            await self._nova_memory_render_phase(
                update,
                context,
                screen,
                source_message=message,
            )
        return True

    async def _nova_memory_route_intent(
        self,
        update: Any,
        context: Any,
        user: User,
        current: NovaMemoryFlowSession | None,
        intent: NovaMemoryIntentResult,
        *,
        canonical_candidate: Any | None,
        invalid: bool = False,
    ) -> None:
        if current is not None and canonical_candidate is not None:
            await self._nova_memory_delete_transient(canonical_candidate, context)
            canonical_candidate = None
        session = current
        if session is None:
            if canonical_candidate is None:
                try:
                    canonical_candidate = await update.effective_message.reply_text(
                        "🧬 Моя Nova\n\nОткрываю…",
                        parse_mode=None,
                    )
                except asyncio.CancelledError:
                    raise
                except (TelegramError, TypeError, AttributeError) as exc:
                    logger.warning(
                        "Nova memory canonical creation failed error_type=%s",
                        type(exc).__name__,
                    )
                    return
            message_id = self._positive_message_id(getattr(canonical_candidate, "message_id", None))
            if message_id is None:
                logger.warning("Nova memory canonical creation failed error_type=MissingMessageId")
                return
            async with self._nova_memory_ui_lock:
                session = await self.nova_memory_sessions.create(
                    owner_id=user.id,
                    telegram_user_id=update.effective_user.id,
                    chat_id=update.effective_chat.id,
                    tier=require_access_tier(user.access_tier),
                    access_version=user.access_version,
                    canonical_message_id=message_id,
                    phase=NovaMemoryFlowPhase.ROOT,
                )

        source = canonical_candidate or update.effective_message
        if intent.kind is NovaMemoryIntentKind.OPEN:
            transitioned = await self.nova_memory_sessions.update(
                session,
                phase=NovaMemoryFlowPhase.ROOT,
                candidate_content=None,
                candidate_category=None,
                candidate_important=False,
                item_public_id=None,
                item_version=None,
                list_filter=None,
                page=0,
                collection_revision=None,
                collection_count=0,
            )
            if transitioned is not None:
                await self._nova_memory_show_root(
                    update,
                    context,
                    transitioned,
                    source_message=source,
                )
            return
        if intent.kind is NovaMemoryIntentKind.DELETE_ALL:
            await self._nova_memory_show_delete_all(
                update,
                context,
                session,
                source_message=source,
            )
            return
        if intent.kind is NovaMemoryIntentKind.REMEMBER_THIS:
            content = await self._nova_memory_latest_candidate(update)
            if content is None:
                awaiting = await self.nova_memory_sessions.update(
                    session,
                    phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
                    candidate_content=None,
                    candidate_category="about_me",
                    candidate_important=False,
                    item_public_id=None,
                    item_version=None,
                )
                if awaiting is not None:
                    await self._nova_memory_render_awaiting(
                        context,
                        awaiting,
                        notice=NOVA_MEMORY_REFERENCE_MISSING_TEXT,
                        source_message=source,
                    )
                return
            intent = NovaMemoryIntentResult(
                kind=NovaMemoryIntentKind.CREATE,
                content=content,
                category="about_me",
                important=False,
            )
        if intent.kind is NovaMemoryIntentKind.AWAIT_CONTENT:
            awaiting = await self.nova_memory_sessions.update(
                session,
                phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
                candidate_content=None,
                candidate_category=intent.category or "about_me",
                candidate_important=bool(intent.important),
                item_public_id=None,
                item_version=None,
                list_filter=None,
                page=0,
            )
            if awaiting is not None:
                await self._nova_memory_render_awaiting(
                    context,
                    awaiting,
                    notice=NOVA_MEMORY_VALIDATION_FAILURE_TEXT if invalid else None,
                    source_message=source,
                )
            return
        if intent.kind is NovaMemoryIntentKind.CREATE and intent.content is not None:
            prepared = await self.nova_memory_sessions.update(
                session,
                phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
                candidate_content=None,
                candidate_category=intent.category or "about_me",
                candidate_important=bool(intent.important),
                item_public_id=None,
                item_version=None,
                list_filter=None,
                page=0,
            )
            if prepared is not None:
                await self._nova_memory_accept_content(
                    update,
                    context,
                    prepared,
                    intent.content,
                    source_message=source,
                )

    async def _nova_memory_accept_content(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        content: str,
        *,
        source_message: Any | None = None,
    ) -> None:
        try:
            normalized = normalize_nova_memory_content(content)
        except NovaMemoryValidationError:
            screen = await self.nova_memory_sessions.update(session, phase=session.phase)
            if screen is not None:
                await self._nova_memory_render_awaiting(
                    context,
                    screen,
                    notice=NOVA_MEMORY_VALIDATION_FAILURE_TEXT,
                    source_message=source_message or update.effective_message,
                )
            return
        fresh = await self._nova_memory_access(update)
        if not self._nova_memory_same_access(session, fresh):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or update.effective_message,
            )
            return
        target_phase = (
            NovaMemoryFlowPhase.CREATE_PREVIEW
            if session.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
            else NovaMemoryFlowPhase.UPDATE_PREVIEW
        )
        updated = await self.nova_memory_sessions.update(
            session,
            phase=target_phase,
            candidate_content=normalized,
            candidate_category=session.candidate_category or "about_me",
        )
        if updated is not None:
            await self._nova_memory_render_phase(
                update,
                context,
                updated,
                source_message=source_message or update.effective_message,
            )

    async def _nova_memory_latest_candidate(self, update: Any) -> str | None:
        try:
            snapshot = await self.conversation.get(
                update.effective_user.id,
                update.effective_chat.id,
            )
            helper = getattr(self.conversation, "latest_nova_memory_candidate", None)
            candidate = helper(snapshot) if callable(helper) else None
            if not isinstance(candidate, str):
                return None
            if _safe_nova_memory_intent(candidate)[0].kind is not NovaMemoryIntentKind.NONE:
                return None
            return normalize_nova_memory_content(candidate)
        except asyncio.CancelledError:
            raise
        except (NovaMemoryValidationError, TypeError, ValueError):
            return None
        except Exception as exc:
            logger.warning(
                "Nova memory conversation lookup failed error_type=%s", type(exc).__name__
            )
            return None

    async def nova_memory_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        data = query.data if query is not None and isinstance(query.data, str) else ""
        if query is None:
            return
        identity = await self._nova_memory_identity(update)
        message_id = self._positive_message_id(getattr(query.message, "message_id", None))
        if identity is None or message_id is None:
            await self._nova_memory_answer(query, NOVA_MEMORY_STALE_ALERT, show_alert=True)
            return
        async with self._nova_memory_launch_lock:
            current = await self.nova_memory_sessions.current(
                owner_id=identity.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
            )
            fresh = await self._nova_memory_access(update)
            if current is not None and not self._nova_memory_same_access(current, fresh):
                if message_id != current.canonical_message_id:
                    await self._nova_memory_answer(
                        query,
                        NOVA_MEMORY_STALE_ALERT,
                        show_alert=True,
                    )
                    return
                await self._nova_memory_answer(query)
                await self._nova_memory_access_changed(
                    context,
                    current,
                    source_message=query.message,
                    query=query,
                )
                return
            if fresh is None:
                await self._nova_memory_answer(query, NOVA_MEMORY_STALE_ALERT, show_alert=True)
                return
            claim = await self.nova_memory_sessions.claim(
                data,
                owner_id=fresh.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                tier=require_access_tier(fresh.access_tier),
                access_version=fresh.access_version,
                canonical_message_id=message_id,
            )
            if claim is None:
                await self._nova_memory_answer(query, NOVA_MEMORY_STALE_ALERT, show_alert=True)
                return
            await self._nova_memory_answer(query)
            fresh = await self._nova_memory_access(update)
            live = await self.nova_memory_sessions.get_exact(claim.session)
            if live is None:
                return
            if not self._nova_memory_same_access(live, fresh):
                await self._nova_memory_access_changed(
                    context,
                    live,
                    source_message=query.message,
                    query=query,
                )
                return
            await self._nova_memory_dispatch_callback(update, context, claim)

    async def _nova_memory_dispatch_callback(
        self,
        update: Any,
        context: Any,
        claim: NovaMemoryCapabilityClaim,
    ) -> None:
        action = claim.capability.action
        session = claim.session
        query = update.callback_query
        if action == "continue":
            continued = await self.nova_memory_sessions.update(session, phase=session.phase)
            if continued is None:
                return
            await self._nova_memory_render_phase(
                update,
                context,
                continued,
                query=query,
                source_message=query.message,
            )
            return
        if action == "exit_home":
            root_keyboard = getattr(self, "_root_keyboard", None)
            text = (
                "Главное меню\n\nЧто хочешь сделать?" if callable(root_keyboard) else "Главное меню"
            )
            markup = root_keyboard(session.tier) if callable(root_keyboard) else None
            await self._nova_memory_retire_and_edit(
                context,
                session,
                text,
                markup,
                query=query,
                source_message=query.message,
                operation="exit_home",
            )
            return
        if action == "root":
            updated = await self.nova_memory_sessions.update(
                session,
                phase=NovaMemoryFlowPhase.ROOT,
                candidate_content=None,
                candidate_category=None,
                candidate_important=False,
                item_public_id=None,
                item_version=None,
                list_filter=None,
                page=0,
                collection_revision=None,
                collection_count=0,
            )
            if updated is not None:
                await self._nova_memory_show_root(update, context, updated, query=query)
            return
        if action == "create":
            updated = await self.nova_memory_sessions.update(
                session,
                phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
                candidate_content=None,
                candidate_category="about_me",
                candidate_important=False,
                item_public_id=None,
                item_version=None,
                list_filter=None,
                page=0,
            )
            if updated is not None:
                await self._nova_memory_render_awaiting(context, updated, query=query)
            return
        if action == "help":
            updated = await self.nova_memory_sessions.update(
                session,
                phase=NovaMemoryFlowPhase.ROOT,
            )
            if updated is not None:
                tokens = await self._nova_memory_tokens(
                    updated,
                    (("root", False, {}),),
                )
                if tokens is None:
                    return
                await self._nova_memory_deliver(
                    context,
                    updated,
                    NOVA_MEMORY_HELP_TEXT,
                    InlineKeyboardMarkup(
                        [[InlineKeyboardButton("← К моей Nova", callback_data=tokens["root"])]]
                    ),
                    query=query,
                    source_message=query.message,
                    operation="help",
                )
            return
        if action.startswith("list_"):
            list_filter = action.removeprefix("list_")
            await self._nova_memory_show_list(
                update,
                context,
                session,
                list_filter=list_filter,
                page=0,
                query=query,
            )
            return
        if action in {"page_prev", "page_next", "back_list"}:
            await self._nova_memory_show_list(
                update,
                context,
                session,
                list_filter=claim.capability.list_filter or session.list_filter or "about_me",
                page=claim.capability.page or 0,
                query=query,
            )
            return
        if action == "detail" or action.startswith("detail_"):
            await self._nova_memory_show_detail(
                update,
                context,
                session,
                public_id=claim.capability.public_id,
                list_filter=claim.capability.list_filter or session.list_filter,
                page=claim.capability.page if claim.capability.page is not None else session.page,
                query=query,
            )
            return
        if action == "edit":
            await self._nova_memory_begin_edit(update, context, session, query=query)
            return
        if action == "edit_text":
            phase = (
                NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
                if session.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
                else NovaMemoryFlowPhase.AWAITING_UPDATE_CONTENT
            )
            updated = await self.nova_memory_sessions.update(session, phase=phase)
            if updated is not None:
                await self._nova_memory_render_awaiting(context, updated, query=query)
            return
        if action == "choose_category":
            await self._nova_memory_show_category_picker(context, session, query=query)
            return
        if action.startswith("category_"):
            category = action.removeprefix("category_")
            if category not in _CATEGORY_LABELS:
                return
            phase = (
                NovaMemoryFlowPhase.CREATE_PREVIEW
                if session.item_public_id is None
                else NovaMemoryFlowPhase.UPDATE_PREVIEW
            )
            updated = await self.nova_memory_sessions.update(
                session,
                phase=phase,
                candidate_category=category,
            )
            if updated is not None:
                await self._nova_memory_render_phase(update, context, updated, query=query)
            return
        if action == "toggle_candidate_important":
            updated = await self.nova_memory_sessions.update(
                session,
                phase=session.phase,
                candidate_important=not session.candidate_important,
            )
            if updated is not None:
                await self._nova_memory_render_phase(update, context, updated, query=query)
            return
        if action == "cancel_create":
            updated = await self.nova_memory_sessions.update(
                session,
                phase=NovaMemoryFlowPhase.ROOT,
                candidate_content=None,
                candidate_category=None,
                candidate_important=False,
                item_public_id=None,
                item_version=None,
            )
            if updated is not None:
                await self._nova_memory_show_root(update, context, updated, query=query)
            return
        if action == "cancel_update":
            await self._nova_memory_show_detail(
                update,
                context,
                session,
                public_id=session.item_public_id,
                list_filter=session.list_filter,
                page=session.page,
                query=query,
            )
            return
        if action == "delete_preview":
            await self._nova_memory_show_delete_item(update, context, session, query=query)
            return
        if action == "delete_all_preview":
            await self._nova_memory_show_delete_all(update, context, session, query=query)
            return
        if action == "confirm_create":
            await self._nova_memory_confirm_create(update, context, claim)
            return
        if action == "confirm_update":
            await self._nova_memory_confirm_update(update, context, claim)
            return
        if action == "toggle_important":
            await self._nova_memory_toggle_important(update, context, claim)
            return
        if action == "confirm_delete":
            await self._nova_memory_confirm_delete(update, context, claim)
            return
        if action == "confirm_delete_all":
            await self._nova_memory_confirm_delete_all(update, context, claim)

    async def _nova_memory_show_root(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
        notice: str | None = None,
    ) -> None:
        try:
            status = await self.nova_memory_service.status(
                telegram_actor_id=session.telegram_user_id
            )
        except asyncio.CancelledError:
            raise
        except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
            logger.warning("Nova memory status failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_failure(
                context,
                session,
                query=query,
                source_message=source_message,
            )
            return
        if status.status != "available" or status.access_version != session.access_version:
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        fresh = await self._nova_memory_access(update)
        if not self._nova_memory_same_access(session, fresh):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.ROOT,
            collection_revision=status.collection_revision,
            collection_count=status.count,
            item_public_id=None,
            item_version=None,
            list_filter=None,
            page=0,
        )
        if updated is None:
            return
        actions: list[tuple[str, bool, dict[str, Any]]] = [
            ("create", False, {}),
            ("list_important", False, {}),
            ("list_about_me", False, {}),
            ("list_interaction", False, {}),
            ("list_orientation", False, {}),
            ("help", False, {}),
        ]
        if status.count > 0:
            actions.append(("delete_all_preview", False, {}))
        actions.append(("exit_home", False, {}))
        tokens = await self._nova_memory_tokens(updated, tuple(actions))
        if tokens is None:
            return
        rows = [
            [InlineKeyboardButton("➕ Научить Nova", callback_data=tokens["create"])],
            [InlineKeyboardButton("⭐ Важное", callback_data=tokens["list_important"])],
            [
                InlineKeyboardButton(
                    "📖 Что Nova знает обо мне", callback_data=tokens["list_about_me"]
                )
            ],
            [
                InlineKeyboardButton(
                    "⚙️ Как со мной работать", callback_data=tokens["list_interaction"]
                )
            ],
            [InlineKeyboardButton("🧭 Мои ориентиры", callback_data=tokens["list_orientation"])],
            [InlineKeyboardButton("❓ Как это работает", callback_data=tokens["help"])],
        ]
        if status.count > 0:
            rows.append(
                [InlineKeyboardButton("🗑 Забыть всё", callback_data=tokens["delete_all_preview"])]
            )
        rows.append([InlineKeyboardButton("🏠 Главное меню", callback_data=tokens["exit_home"])])
        text = NOVA_MEMORY_ROOT_TEXT.format(count=status.count, max_items=status.max_items)
        if notice:
            text = f"{notice}\n\n{text}"
        await self._nova_memory_deliver(
            context,
            updated,
            text,
            InlineKeyboardMarkup(rows),
            query=query,
            source_message=source_message,
            operation="root",
        )

    async def _nova_memory_render_awaiting(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        notice: str | None = None,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        editing = session.phase is NovaMemoryFlowPhase.AWAITING_UPDATE_CONTENT
        back_action = "cancel_update" if editing else "cancel_create"
        tokens = await self._nova_memory_tokens(
            session,
            ((back_action, False, {}),) + ((("choose_category", False, {}),) if editing else ()),
        )
        if tokens is None:
            return
        if editing:
            current_text = ""
            if session.item_public_id:
                try:
                    lookup = await self.nova_memory_service.get(
                        telegram_actor_id=session.telegram_user_id,
                        public_id=session.item_public_id,
                    )
                except asyncio.CancelledError:
                    raise
                except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
                    logger.warning("Nova memory read failed error_type=%s", type(exc).__name__)
                    await self._nova_memory_render_failure(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                    return
                if lookup.status != "found" or lookup.item is None:
                    await self._nova_memory_render_missing(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                    return
                if not self._nova_memory_same_access(
                    session,
                    await self._nova_memory_access_values(
                        session.telegram_user_id,
                        session.chat_id,
                    ),
                ):
                    await self._nova_memory_access_changed(
                        context,
                        session,
                        source_message=source_message or getattr(query, "message", None),
                        query=query,
                    )
                    return
                current_text = lookup.item.content
            text = (
                "✏️ Изменить запись\n\n"
                "Отправь новый текст или голосовое сообщение.\n\n"
                f"Текущий текст:\n«{current_text}»"
            )
            rows = [
                [
                    InlineKeyboardButton(
                        "🗂 Изменить только раздел", callback_data=tokens["choose_category"]
                    )
                ],
                [InlineKeyboardButton("← К записи", callback_data=tokens[back_action])],
            ]
        else:
            text = NOVA_MEMORY_CREATE_TEXT
            rows = [[InlineKeyboardButton("✖️ Отмена", callback_data=tokens[back_action])]]
        if notice:
            text = f"{text}\n\n{notice}"
        await self._nova_memory_deliver(
            context,
            session,
            text,
            InlineKeyboardMarkup(rows),
            query=query,
            source_message=source_message,
            operation="awaiting",
        )

    async def _nova_memory_render_preview(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        content = session.candidate_content
        category = session.candidate_category or "about_me"
        if content is None:
            await self._nova_memory_render_awaiting(
                context,
                session,
                query=query,
                source_message=source_message,
            )
            return
        fresh = await self._nova_memory_access(update)
        if not self._nova_memory_same_access(session, fresh):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        creating = session.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
        confirm_action = "confirm_create" if creating else "confirm_update"
        cancel_action = "cancel_create" if creating else "cancel_update"
        token_specs = (
            (confirm_action, True, self._mutation_fences(session)),
            ("choose_category", False, {}),
        )
        if creating:
            token_specs += (("toggle_candidate_important", False, {}),)
        token_specs += (("edit_text", False, {}), (cancel_action, False, {}))
        tokens = await self._nova_memory_tokens(session, token_specs)
        if tokens is None:
            return
        important_label = "☆ Убрать из важного" if session.candidate_important else "⭐ В важное"
        if creating:
            text = (
                "🧬 Nova запомнит\n\n"
                f"«{content}»\n\n"
                f"Раздел: {_CATEGORY_LABELS[category]}\n"
                f"Важное: {'да' if session.candidate_important else 'нет'}\n\n"
                "Сохранить?"
            )
        else:
            old_content = content
            if session.item_public_id:
                try:
                    lookup = await self.nova_memory_service.get(
                        telegram_actor_id=session.telegram_user_id,
                        public_id=session.item_public_id,
                    )
                except asyncio.CancelledError:
                    raise
                except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
                    logger.warning("Nova memory read failed error_type=%s", type(exc).__name__)
                    await self._nova_memory_render_failure(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                    return
                if lookup.status != "found" or lookup.item is None:
                    await self._nova_memory_render_missing(
                        context,
                        session,
                        query=query,
                        source_message=source_message,
                    )
                    return
                if not self._nova_memory_same_access(
                    session,
                    await self._nova_memory_access(update),
                ):
                    await self._nova_memory_access_changed(
                        context,
                        session,
                        source_message=source_message or getattr(query, "message", None),
                        query=query,
                    )
                    return
                old_content = lookup.item.content
            text = (
                "✏️ Сохранить изменения?\n\n"
                f"Было:\n«{old_content}»\n\n"
                f"Станет:\n«{content}»\n\n"
                f"Раздел: {_CATEGORY_LABELS[category]}"
            )
        category_row = [
            InlineKeyboardButton("🗂 Изменить раздел", callback_data=tokens["choose_category"])
        ]
        if creating:
            category_row.append(
                InlineKeyboardButton(
                    important_label,
                    callback_data=tokens["toggle_candidate_important"],
                )
            )
        rows = [
            [
                InlineKeyboardButton(
                    "✅ Запомнить" if creating else "✅ Сохранить изменения",
                    callback_data=tokens[confirm_action],
                )
            ],
            category_row,
            [
                InlineKeyboardButton(
                    "✏️ Изменить текст" if creating else "↩️ Изменить текст",
                    callback_data=tokens["edit_text"],
                )
            ],
            [InlineKeyboardButton("✖️ Отмена", callback_data=tokens[cancel_action])],
        ]
        await self._nova_memory_deliver(
            context,
            session,
            text,
            InlineKeyboardMarkup(rows),
            query=query,
            source_message=source_message,
            operation="preview",
        )

    async def _nova_memory_show_category_picker(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any,
    ) -> None:
        # A same-phase transition gives the picker its own screen generation.
        updated = await self.nova_memory_sessions.update(session, phase=session.phase)
        if updated is None:
            return
        specs = tuple((f"category_{category}", False, {}) for category in _CATEGORY_LABELS)
        specs += (("continue", False, {}),)
        tokens = await self._nova_memory_tokens(updated, specs)
        if tokens is None:
            return
        rows = [
            [InlineKeyboardButton("📖 Обо мне", callback_data=tokens["category_about_me"])],
            [
                InlineKeyboardButton(
                    "⚙️ Как со мной работать",
                    callback_data=tokens["category_interaction"],
                )
            ],
            [InlineKeyboardButton("🧭 Мой ориентир", callback_data=tokens["category_orientation"])],
            [InlineKeyboardButton("← К предпросмотру", callback_data=tokens["continue"])],
        ]
        await self._nova_memory_deliver(
            context,
            updated,
            "🗂 Выбери раздел\n\nТекст записи не изменится.",
            InlineKeyboardMarkup(rows),
            query=query,
            source_message=query.message,
            operation="category",
        )

    async def _nova_memory_show_list(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        list_filter: str,
        page: int,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        if list_filter not in _FILTER_TITLES:
            return
        try:
            view = await self._nova_memory_page(
                session.telegram_user_id,
                list_filter=list_filter,
                page=page,
            )
        except asyncio.CancelledError:
            raise
        except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
            logger.warning("Nova memory list failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_failure(
                context,
                session,
                query=query,
                source_message=source_message,
            )
            return
        if view is None:
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        fresh = await self._nova_memory_access(update)
        if not self._nova_memory_same_access(session, fresh):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.LIST,
            item_public_id=None,
            item_version=None,
            list_filter=list_filter,
            page=view.page,
            collection_revision=view.collection_revision,
            collection_count=view.total,
        )
        if updated is None:
            return
        if not view.items:
            tokens = await self._nova_memory_tokens(
                updated,
                (("create", False, {}), ("root", False, {})),
            )
            if tokens is None:
                return
            await self._nova_memory_deliver(
                context,
                updated,
                f"{_FILTER_TITLES[list_filter]}\n\n"
                "Здесь пока пусто.\nНаучи Nova — и запись появится здесь.",
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("➕ Научить Nova", callback_data=tokens["create"])],
                        [InlineKeyboardButton("← К моей Nova", callback_data=tokens["root"])],
                    ]
                ),
                query=query,
                source_message=source_message,
                operation="empty_list",
            )
            return

        specs: list[tuple[str, bool, dict[str, Any]]] = []
        item_keys: list[str] = []
        for index, item in enumerate(view.items):
            key = f"detail_{index}"
            item_keys.append(key)
            specs.append(
                (
                    key,
                    False,
                    {
                        "public_id": item.public_id,
                        "expected_item_version": item.version,
                        "list_filter": list_filter,
                        "page": view.page,
                    },
                )
            )
        if view.page > 0:
            specs.append(
                (
                    "page_prev",
                    False,
                    {"list_filter": list_filter, "page": view.page - 1},
                )
            )
        if view.page + 1 < view.pages:
            specs.append(
                (
                    "page_next",
                    False,
                    {"list_filter": list_filter, "page": view.page + 1},
                )
            )
        specs.append(("root", False, {}))
        tokens = await self._nova_memory_tokens(updated, tuple(specs))
        if tokens is None:
            return
        rows: list[list[InlineKeyboardButton]] = []
        for index, (key, item) in enumerate(zip(item_keys, view.items, strict=True), 1):
            prefix = "⭐ " if item.important else ""
            label = self._truncate_utf16(f"{index}. {prefix}{item.content}", 44)
            rows.append([InlineKeyboardButton(label, callback_data=tokens[key])])
        pager: list[InlineKeyboardButton] = []
        if "page_prev" in tokens:
            pager.append(InlineKeyboardButton("← Назад", callback_data=tokens["page_prev"]))
        if "page_next" in tokens:
            pager.append(InlineKeyboardButton("Далее →", callback_data=tokens["page_next"]))
        if pager:
            rows.append(pager)
        rows.append([InlineKeyboardButton("← К моей Nova", callback_data=tokens["root"])])
        await self._nova_memory_deliver(
            context,
            updated,
            f"{_FILTER_TITLES[list_filter]}\n\n"
            f"Записей: {view.total} · Страница {view.page + 1}/{view.pages}\n\n"
            "Выбери запись:",
            InlineKeyboardMarkup(rows),
            query=query,
            source_message=source_message,
            operation="list",
        )

    async def _nova_memory_page(
        self,
        telegram_actor_id: int,
        *,
        list_filter: str,
        page: int,
    ) -> _PageView | None:
        page = max(page, 0)
        if list_filter != "important":
            result = await self.nova_memory_service.list(
                telegram_actor_id=telegram_actor_id,
                category=list_filter,
                offset=page * NOVA_MEMORY_PAGE_SIZE,
                limit=NOVA_MEMORY_PAGE_SIZE,
            )
            if result.status != "ok" or result.collection_revision is None:
                return None
            pages = max(math.ceil(result.total / NOVA_MEMORY_PAGE_SIZE), 1)
            if page >= pages and result.total:
                page = pages - 1
                result = await self.nova_memory_service.list(
                    telegram_actor_id=telegram_actor_id,
                    category=list_filter,
                    offset=page * NOVA_MEMORY_PAGE_SIZE,
                    limit=NOVA_MEMORY_PAGE_SIZE,
                )
                if result.status != "ok" or result.collection_revision is None:
                    return None
            return _PageView(
                items=result.items,
                total=result.total,
                page=page,
                pages=pages,
                collection_revision=result.collection_revision,
            )

        chunks: list[NovaMemoryPage] = []
        coherent_revision: str | None = None
        offset = 0
        while offset < 100:
            result = await self.nova_memory_service.list(
                telegram_actor_id=telegram_actor_id,
                offset=offset,
                limit=50,
            )
            if result.status != "ok" or result.collection_revision is None:
                return None
            if coherent_revision is None:
                coherent_revision = result.collection_revision
            elif result.collection_revision != coherent_revision:
                # Never assemble a page from two different collection generations.
                raise NovaMemoryStorageError("Memory collection changed while listing.")
            chunks.append(result)
            if result.next_offset is None:
                break
            offset = result.next_offset
        items = tuple(item for chunk in chunks for item in chunk.items if item.important)
        total = len(items)
        pages = max(math.ceil(total / NOVA_MEMORY_PAGE_SIZE), 1)
        page = min(page, pages - 1)
        start = page * NOVA_MEMORY_PAGE_SIZE
        return _PageView(
            items=items[start : start + NOVA_MEMORY_PAGE_SIZE],
            total=total,
            page=page,
            pages=pages,
            collection_revision=coherent_revision or "",
        )

    async def _nova_memory_show_detail(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        public_id: str | None,
        list_filter: str | None,
        page: int,
        query: Any | None = None,
        source_message: Any | None = None,
        notice: str | None = None,
    ) -> None:
        if not public_id:
            await self._nova_memory_render_missing(
                context,
                session,
                query=query,
                source_message=source_message,
            )
            return
        try:
            lookup = await self.nova_memory_service.get(
                telegram_actor_id=session.telegram_user_id,
                public_id=public_id,
            )
        except asyncio.CancelledError:
            raise
        except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
            logger.warning("Nova memory detail failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_failure(
                context,
                session,
                query=query,
                source_message=source_message,
            )
            return
        if lookup.status == "access_denied":
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        if lookup.status != "found" or lookup.item is None:
            await self._nova_memory_render_missing(
                context,
                session,
                query=query,
                source_message=source_message,
            )
            return
        if not self._nova_memory_same_access(session, await self._nova_memory_access(update)):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        item = lookup.item
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.DETAIL,
            candidate_content=None,
            candidate_category=None,
            candidate_important=False,
            item_public_id=item.public_id,
            item_version=item.version,
            list_filter=list_filter,
            page=max(page, 0),
        )
        if updated is None:
            return
        specs = (
            (
                "edit",
                False,
                {"public_id": item.public_id, "expected_item_version": item.version},
            ),
            (
                "toggle_important",
                True,
                {"public_id": item.public_id, "expected_item_version": item.version},
            ),
            (
                "delete_preview",
                False,
                {"public_id": item.public_id, "expected_item_version": item.version},
            ),
            (
                "back_list" if list_filter else "root",
                False,
                {"list_filter": list_filter, "page": page} if list_filter else {},
            ),
            ("root", False, {}),
        )
        tokens = await self._nova_memory_tokens(updated, specs)
        if tokens is None:
            return
        importance = "\n⭐ Важное" if item.important else ""
        text = (
            f"🧬 Запись Nova\n\n{_CATEGORY_LABELS[item.category]}{importance}\n\n«{item.content}»"
        )
        if notice:
            text = f"{notice}\n\n{text}"
        back_action = "back_list" if list_filter else "root"
        back_label = "← К списку" if list_filter else "← К моей Nova"
        rows = [
            [InlineKeyboardButton("✏️ Изменить", callback_data=tokens["edit"])],
            [
                InlineKeyboardButton(
                    "☆ Убрать из важного" if item.important else "⭐ В важное",
                    callback_data=tokens["toggle_important"],
                )
            ],
            [InlineKeyboardButton("🗑 Забыть", callback_data=tokens["delete_preview"])],
            [
                InlineKeyboardButton(back_label, callback_data=tokens[back_action]),
                InlineKeyboardButton("🧬 Моя Nova", callback_data=tokens["root"]),
            ],
        ]
        await self._nova_memory_deliver(
            context,
            updated,
            text,
            InlineKeyboardMarkup(rows),
            query=query,
            source_message=source_message,
            operation="detail",
        )

    async def _nova_memory_begin_edit(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any,
    ) -> None:
        public_id = session.item_public_id
        if not public_id:
            await self._nova_memory_render_missing(context, session, query=query)
            return
        try:
            lookup = await self.nova_memory_service.get(
                telegram_actor_id=session.telegram_user_id,
                public_id=public_id,
            )
        except asyncio.CancelledError:
            raise
        except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
            logger.warning("Nova memory edit read failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_failure(context, session, query=query)
            return
        if lookup.status != "found" or lookup.item is None:
            if lookup.status == "access_denied":
                await self._nova_memory_access_changed(
                    context,
                    session,
                    source_message=query.message,
                    query=query,
                )
            else:
                await self._nova_memory_render_missing(context, session, query=query)
            return
        if not self._nova_memory_same_access(session, await self._nova_memory_access(update)):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=query.message,
                query=query,
            )
            return
        item = lookup.item
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.AWAITING_UPDATE_CONTENT,
            candidate_content=item.content,
            candidate_category=item.category,
            candidate_important=item.important,
            item_public_id=item.public_id,
            item_version=item.version,
        )
        if updated is not None:
            await self._nova_memory_render_awaiting(context, updated, query=query)

    async def _nova_memory_show_delete_item(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any,
    ) -> None:
        public_id = session.item_public_id
        if not public_id:
            await self._nova_memory_render_missing(context, session, query=query)
            return
        try:
            lookup = await self.nova_memory_service.get(
                telegram_actor_id=session.telegram_user_id,
                public_id=public_id,
            )
        except asyncio.CancelledError:
            raise
        except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
            logger.warning("Nova memory delete read failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_failure(context, session, query=query)
            return
        if lookup.status != "found" or lookup.item is None:
            if lookup.status == "access_denied":
                await self._nova_memory_access_changed(
                    context,
                    session,
                    source_message=query.message,
                    query=query,
                )
            else:
                await self._nova_memory_render_missing(context, session, query=query)
            return
        if not self._nova_memory_same_access(session, await self._nova_memory_access(update)):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=query.message,
                query=query,
            )
            return
        item = lookup.item
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.DELETE_PREVIEW,
            item_public_id=item.public_id,
            item_version=item.version,
        )
        if updated is None:
            return
        tokens = await self._nova_memory_tokens(
            updated,
            (
                (
                    "confirm_delete",
                    True,
                    {"public_id": item.public_id, "expected_item_version": item.version},
                ),
                ("cancel_update", False, {}),
            ),
        )
        if tokens is None:
            return
        await self._nova_memory_deliver(
            context,
            updated,
            "🗑 Забыть эту запись?\n\n"
            f"«{item.content}»\n\n"
            "После подтверждения Nova удалит её из активной памяти. "
            "Отменить это действие будет нельзя.",
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🗑 Да, забыть", callback_data=tokens["confirm_delete"])],
                    [
                        InlineKeyboardButton(
                            "← Нет, оставить", callback_data=tokens["cancel_update"]
                        )
                    ],
                ]
            ),
            query=query,
            source_message=query.message,
            operation="delete_preview",
        )

    async def _nova_memory_show_delete_all(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        try:
            status = await self.nova_memory_service.status(
                telegram_actor_id=session.telegram_user_id
            )
        except asyncio.CancelledError:
            raise
        except (NovaMemoryStorageError, NovaMemoryValidationError) as exc:
            logger.warning("Nova memory delete-all read failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_failure(
                context,
                session,
                query=query,
                source_message=source_message,
            )
            return
        if (
            status.status != "available"
            or status.access_version != session.access_version
            or status.collection_revision is None
        ):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        if not self._nova_memory_same_access(session, await self._nova_memory_access(update)):
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=source_message or getattr(query, "message", None),
                query=query,
            )
            return
        if status.count <= 0:
            await self._nova_memory_show_root(
                update,
                context,
                session,
                query=query,
                source_message=source_message,
                notice="Память уже пуста.",
            )
            return
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.DELETE_ALL_PREVIEW,
            collection_revision=status.collection_revision,
            collection_count=status.count,
            item_public_id=None,
            item_version=None,
        )
        if updated is None:
            return
        tokens = await self._nova_memory_tokens(
            updated,
            (
                (
                    "confirm_delete_all",
                    True,
                    {"expected_collection_revision": status.collection_revision},
                ),
                ("root", False, {}),
            ),
        )
        if tokens is None:
            return
        await self._nova_memory_deliver(
            context,
            updated,
            "🗑 Забыть всё?\n\n"
            f"Будут удалены все {status.count} записей из «Моей Nova».\n\n"
            "Задачи, напоминания, желания и остальные данные бота не изменятся. "
            "Отменить удаление памяти будет нельзя.",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🗑 Да, забыть всё", callback_data=tokens["confirm_delete_all"]
                        )
                    ],
                    [InlineKeyboardButton("← Нет, оставить", callback_data=tokens["root"])],
                ]
            ),
            query=query,
            source_message=source_message,
            operation="delete_all_preview",
        )

    async def _nova_memory_confirm_create(
        self,
        update: Any,
        context: Any,
        claim: NovaMemoryCapabilityClaim,
    ) -> None:
        session = claim.session
        if session.candidate_content is None or session.candidate_category is None:
            await self._nova_memory_render_failure(
                context,
                session,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_before_mutation(update, context, session):
            return
        try:
            result = await self.nova_memory_service.create(
                telegram_actor_id=session.telegram_user_id,
                expected_access_version=session.access_version,
                category=session.candidate_category,
                content=session.candidate_content,
                important=session.candidate_important,
            )
        except asyncio.CancelledError:
            raise
        except NovaMemoryValidationError:
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_VALIDATION_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        except NovaMemoryStorageError as exc:
            logger.warning("Nova memory create failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_STORAGE_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_after_mutation(update, context, session):
            return
        if result.status in {"created", "duplicate"} and result.item is not None:
            notice = (
                "✅ Nova запомнила эту запись."
                if result.status == "created"
                else "Эта запись уже сохранена. Открываю существующую карточку."
            )
            await self._nova_memory_show_detail(
                update,
                context,
                session,
                public_id=result.item.public_id,
                list_filter=result.item.category,
                page=0,
                query=update.callback_query,
                notice=notice,
            )
            return
        if result.status == "limit_reached":
            await self._nova_memory_transition_root_notice(
                update,
                context,
                session,
                "Достигнут лимит памяти. Забудь ненужную запись перед добавлением новой.",
                query=update.callback_query,
            )
            return
        await self._nova_memory_handle_domain_failure(update, context, session, result.status)

    async def _nova_memory_before_mutation(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
    ) -> bool:
        fresh = await self._nova_memory_access(update)
        # Access lookup awaits database and access-service I/O. Re-read the exact
        # PROCESSING generation afterwards so a replacement can never inherit a
        # claimed mutation.
        live = await self.nova_memory_sessions.get_exact(session)
        if live is not None and self._nova_memory_same_access(live, fresh):
            return True
        if live is not None:
            await self._nova_memory_access_changed(
                context,
                live,
                source_message=update.callback_query.message,
                query=update.callback_query,
            )
        return False

    async def _nova_memory_after_mutation(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
    ) -> bool:
        fresh = await self._nova_memory_access(update)
        if self._nova_memory_same_access(session, fresh):
            return True
        # The domain result is authoritative, but post-commit access loss must
        # never disclose the candidate or stored item on Telegram.
        await self._nova_memory_access_changed(
            context,
            session,
            source_message=update.callback_query.message,
            query=update.callback_query,
        )
        return False

    async def _nova_memory_handle_domain_failure(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        status: str,
    ) -> None:
        if status in {"access_denied", "stale_access"}:
            await self._nova_memory_access_changed(
                context,
                session,
                source_message=update.callback_query.message,
                query=update.callback_query,
            )
            return
        if status in {"stale", "not_found", "conflict"}:
            await self._nova_memory_transition_root_notice(
                update,
                context,
                session,
                "Запись успела измениться. Ничего не перезаписано — проверь актуальную память.",
                query=update.callback_query,
            )
            return
        await self._nova_memory_render_mutation_error(
            context,
            session,
            NOVA_MEMORY_STORAGE_FAILURE_TEXT,
            query=update.callback_query,
        )

    async def _nova_memory_transition_root_notice(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        notice: str,
        *,
        query: Any,
    ) -> None:
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.ROOT,
            candidate_content=None,
            candidate_category=None,
            candidate_important=False,
            item_public_id=None,
            item_version=None,
            list_filter=None,
            page=0,
            collection_revision=None,
            collection_count=0,
        )
        if updated is not None:
            await self._nova_memory_show_root(
                update,
                context,
                updated,
                query=query,
                notice=notice,
            )

    async def _nova_memory_render_mutation_error(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        text: str,
        *,
        query: Any,
    ) -> None:
        updated = await self.nova_memory_sessions.update(
            session,
            phase=NovaMemoryFlowPhase.ROOT,
            candidate_content=None,
            candidate_category=None,
            candidate_important=False,
            item_public_id=None,
            item_version=None,
        )
        if updated is None:
            return
        tokens = await self._nova_memory_tokens(updated, (("root", False, {}),))
        if tokens is None:
            return
        await self._nova_memory_deliver(
            context,
            updated,
            text,
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("🧬 Проверить мою Nova", callback_data=tokens["root"])]]
            ),
            query=query,
            source_message=query.message,
            operation="mutation_failure",
        )

    async def _nova_memory_render_failure(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        live = await self.nova_memory_sessions.get_exact(session)
        if live is None:
            return
        updated = await self.nova_memory_sessions.update(
            live,
            phase=NovaMemoryFlowPhase.ROOT,
            candidate_content=None,
            candidate_category=None,
            candidate_important=False,
            item_public_id=None,
            item_version=None,
        )
        if updated is None:
            return
        tokens = await self._nova_memory_tokens(updated, (("root", False, {}),))
        if tokens is None:
            return
        await self._nova_memory_deliver(
            context,
            updated,
            NOVA_MEMORY_STORAGE_FAILURE_TEXT,
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("🧬 Проверить мою Nova", callback_data=tokens["root"])]]
            ),
            query=query,
            source_message=source_message,
            operation="failure",
        )

    async def _nova_memory_render_missing(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        live = await self.nova_memory_sessions.get_exact(session)
        if live is None:
            return
        updated = await self.nova_memory_sessions.update(
            live,
            phase=NovaMemoryFlowPhase.ROOT,
            candidate_content=None,
            candidate_category=None,
            candidate_important=False,
            item_public_id=None,
            item_version=None,
        )
        if updated is None:
            return
        tokens = await self._nova_memory_tokens(updated, (("root", False, {}),))
        if tokens is None:
            return
        await self._nova_memory_deliver(
            context,
            updated,
            "Эта запись больше недоступна. Открой актуальную память.",
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("🧬 Моя Nova", callback_data=tokens["root"])]]
            ),
            query=query,
            source_message=source_message,
            operation="missing",
        )

    async def _nova_memory_render_phase(
        self,
        update: Any,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any | None = None,
        source_message: Any | None = None,
    ) -> None:
        phase = session.phase
        if phase is NovaMemoryFlowPhase.ROOT:
            await self._nova_memory_show_root(
                update,
                context,
                session,
                query=query,
                source_message=source_message,
            )
        elif phase is NovaMemoryFlowPhase.LIST and session.list_filter:
            await self._nova_memory_show_list(
                update,
                context,
                session,
                list_filter=session.list_filter,
                page=session.page,
                query=query,
                source_message=source_message,
            )
        elif phase is NovaMemoryFlowPhase.DETAIL:
            await self._nova_memory_show_detail(
                update,
                context,
                session,
                public_id=session.item_public_id,
                list_filter=session.list_filter,
                page=session.page,
                query=query,
                source_message=source_message,
            )
        elif phase in _AWAITING_PHASES:
            await self._nova_memory_render_awaiting(
                context,
                session,
                query=query,
                source_message=source_message,
            )
        elif phase in {
            NovaMemoryFlowPhase.CREATE_PREVIEW,
            NovaMemoryFlowPhase.UPDATE_PREVIEW,
        }:
            await self._nova_memory_render_preview(
                update,
                context,
                session,
                query=query,
                source_message=source_message,
            )
        elif phase is NovaMemoryFlowPhase.DELETE_PREVIEW:
            if query is not None:
                await self._nova_memory_show_delete_item(update, context, session, query=query)
        elif phase is NovaMemoryFlowPhase.DELETE_ALL_PREVIEW:
            await self._nova_memory_show_delete_all(
                update,
                context,
                session,
                query=query,
                source_message=source_message,
            )
        elif phase is NovaMemoryFlowPhase.PROCESSING:
            await self._nova_memory_deliver(
                context,
                session,
                "🧬 Моя Nova\n\nДействие уже обрабатывается.",
                None,
                query=query,
                source_message=source_message,
                operation="processing",
            )

    async def _nova_memory_tokens(
        self,
        session: NovaMemoryFlowSession,
        specs: tuple[tuple[str, bool, dict[str, Any]], ...],
    ) -> dict[str, str] | None:
        result: dict[str, str] = {}
        for action, mutation, payload in specs:
            token = await self.nova_memory_sessions.issue(
                session,
                action=action,
                mutation=mutation,
                **payload,
            )
            if token is None:
                # If this screen is still exact, a same-phase transition retires
                # capabilities issued before the failure. If it was replaced,
                # update is a no-op and the replacement remains untouched.
                await self.nova_memory_sessions.update(session, phase=session.phase)
                return None
            result[action] = token
        return result

    @staticmethod
    def _mutation_fences(session: NovaMemoryFlowSession) -> dict[str, Any]:
        if session.item_public_id is not None and session.item_version is not None:
            return {
                "public_id": session.item_public_id,
                "expected_item_version": session.item_version,
            }
        return {}

    async def _nova_memory_deliver(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        query: Any | None,
        source_message: Any | None,
        operation: str,
    ) -> bool:
        await self._nova_memory_ui_lock.acquire()
        lock_handed_off = False
        try:
            live = await self.nova_memory_sessions.get_exact(session)
            if live is None:
                return False
            fresh = await self._nova_memory_access_values(
                live.telegram_user_id,
                live.chat_id,
            )
            if not self._nova_memory_same_access(live, fresh):
                cleared = await self.nova_memory_sessions.clear_exact(live)
                if not cleared:
                    return False
                if query is not None:
                    await self._nova_memory_edit_query(
                        query,
                        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                        None,
                        operation="access_changed",
                    )
                elif source_message is not None:
                    await self._nova_memory_edit(
                        context,
                        live,
                        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                        None,
                        source_message=source_message,
                    )
                return False
            # Access I/O above is an await boundary. An exact re-read prevents a
            # concurrent same-session transition from painting an obsolete screen.
            live = await self.nova_memory_sessions.get_exact(live)
            if live is None:
                return False
            final_access = await self._nova_memory_access_values(
                live.telegram_user_id,
                live.chat_id,
            )
            if not self._nova_memory_same_access(live, final_access):
                cleared = await self.nova_memory_sessions.clear_exact(live)
                if not cleared:
                    return False
                if query is not None:
                    await self._nova_memory_edit_query(
                        query,
                        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                        None,
                        operation="access_changed",
                    )
                elif source_message is not None:
                    await self._nova_memory_edit(
                        context,
                        live,
                        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                        None,
                        source_message=source_message,
                    )
                return False
            live = await self.nova_memory_sessions.get_exact(live)
            if live is None:
                return False
            delivered = live
            if query is not None:
                query_message_id = self._positive_message_id(
                    getattr(query.message, "message_id", None)
                )
                if query_message_id != delivered.canonical_message_id:
                    return False
                edited = await self._nova_memory_edit_query(
                    query,
                    text,
                    reply_markup,
                    operation=operation,
                )
            else:
                edited = await self._nova_memory_edit(
                    context,
                    delivered,
                    text,
                    reply_markup,
                    source_message=source_message,
                    operation=operation,
                )
            if not edited:
                return False
            fence_task = asyncio.create_task(
                self._nova_memory_post_edit_fence_and_release(
                    context,
                    delivered,
                    query=query,
                    source_message=source_message,
                ),
                name="nova-memory-post-edit-fence",
            )
            lock_handed_off = True
            self._nova_memory_track_post_edit_fence(fence_task)
            try:
                return await asyncio.shield(fence_task)
            except asyncio.CancelledError:
                raise
        finally:
            if not lock_handed_off:
                self._nova_memory_ui_lock.release()

    async def _nova_memory_post_edit_fence_and_release(
        self,
        context: Any,
        delivered: NovaMemoryFlowSession,
        *,
        query: Any | None,
        source_message: Any | None,
    ) -> bool:
        """Finish a successful delivery fence while retaining UI serialization."""

        try:
            return await self._nova_memory_post_edit_fence(
                context,
                delivered,
                query=query,
                source_message=source_message,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory post-edit fence failed operation=post_edit_fence error_type=%s",
                type(exc).__name__,
            )
            return False
        finally:
            self._nova_memory_ui_lock.release()

    @staticmethod
    def _nova_memory_observe_post_edit_fence_result(task: asyncio.Task[bool]) -> None:
        try:
            task.result()
        except BaseException:
            return

    def _nova_memory_track_post_edit_fence(self, task: asyncio.Task[bool]) -> None:
        tasks = getattr(self, "_nova_memory_post_edit_tasks", None)
        if tasks is None:
            tasks = set()
            self._nova_memory_post_edit_tasks = tasks
        tasks.add(task)

        def finish(completed: asyncio.Task[bool]) -> None:
            tasks.discard(completed)
            self._nova_memory_observe_post_edit_fence_result(completed)

        task.add_done_callback(finish)

    async def _nova_memory_post_edit_fence(
        self,
        context: Any,
        delivered: NovaMemoryFlowSession,
        *,
        query: Any | None,
        source_message: Any | None,
    ) -> bool:
        """Revalidate private delivery after Telegram I/O and neutralize access loss."""

        first_access = await self._nova_memory_access_values(
            delivered.telegram_user_id,
            delivered.chat_id,
        )
        first_live = await self.nova_memory_sessions.get_exact(delivered)
        final_access = await self._nova_memory_access_values(
            delivered.telegram_user_id,
            delivered.chat_id,
        )
        final_live = await self.nova_memory_sessions.get_exact(delivered)
        access_matches = self._nova_memory_same_access(
            delivered, first_access
        ) and self._nova_memory_same_access(delivered, final_access)
        if access_matches:
            return first_live is not None and final_live is not None

        # Once access loss is confirmed, the already-painted frozen canonical
        # must be neutralized even if a newer generation now shares its message.
        # The CAS may retire only the exact old generation; a replacement is
        # preserved and may repaint later after its own fresh fences.
        if first_live is not None and final_live is not None:
            await self.nova_memory_sessions.clear_exact(delivered)
        await self._nova_memory_compensate_edit_locked(
            context,
            delivered,
            query=query,
            source_message=source_message,
        )
        return False

    async def _nova_memory_compensate_edit_locked(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        query: Any | None,
        source_message: Any | None,
    ) -> None:
        try:
            if query is not None:
                await query.edit_message_text(
                    NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                    reply_markup=None,
                    parse_mode=None,
                )
                return
            bot = getattr(context, "bot", None) if context is not None else None
            edit = getattr(bot, "edit_message_text", None)
            if callable(edit):
                await edit(
                    chat_id=session.chat_id,
                    message_id=session.canonical_message_id,
                    text=NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                    reply_markup=None,
                    parse_mode=None,
                )
                return
            source_id = self._positive_message_id(getattr(source_message, "message_id", None))
            if source_id == session.canonical_message_id:
                await source_message.edit_text(
                    NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                    reply_markup=None,
                    parse_mode=None,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova memory canonical edit failed operation=access_compensation error_type=%s",
                type(exc).__name__,
            )

    async def _nova_memory_retire_and_edit(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        query: Any | None,
        source_message: Any | None,
        operation: str,
    ) -> bool:
        """Atomically retire one exact canonical generation and replace its UI."""

        async with self._nova_memory_ui_lock:
            live = await self.nova_memory_sessions.get_exact(session)
            if live is None:
                return False
            if query is not None:
                query_message_id = self._positive_message_id(
                    getattr(query.message, "message_id", None)
                )
                if query_message_id != live.canonical_message_id:
                    return False
            fresh = await self._nova_memory_access_values(
                live.telegram_user_id,
                live.chat_id,
            )
            live = await self.nova_memory_sessions.get_exact(live)
            if live is None:
                return False
            access_ok = self._nova_memory_same_access(live, fresh)
            final_access = await self._nova_memory_access_values(
                live.telegram_user_id,
                live.chat_id,
            )
            access_ok = access_ok and self._nova_memory_same_access(live, final_access)
            cleared = await self.nova_memory_sessions.clear_exact(live)
            if not cleared:
                return False
            replacement = await self.nova_memory_sessions.current(
                owner_id=live.owner_id,
                telegram_user_id=live.telegram_user_id,
                chat_id=live.chat_id,
            )
            if replacement is not None:
                return False
            if not access_ok:
                text = NOVA_MEMORY_ACCESS_CHANGED_TEXT
                reply_markup = None
                operation = "access_changed"
            if query is not None:
                return await self._nova_memory_edit_query(
                    query,
                    text,
                    reply_markup,
                    operation=operation,
                )
            return await self._nova_memory_edit(
                context,
                live,
                text,
                reply_markup,
                source_message=source_message,
            )

    async def _nova_memory_edit_query(
        self,
        query: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        operation: str,
    ) -> bool:
        try:
            await query.edit_message_text(
                text,
                reply_markup=reply_markup,
                parse_mode=None,
            )
            return True
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            if "message is not modified" in str(exc).casefold():
                return True
            logger.warning(
                "Nova memory callback edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Nova memory callback edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False

    async def _nova_memory_edit(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        source_message: Any | None,
        operation: str = "canonical_edit",
    ) -> bool:
        try:
            bot = getattr(context, "bot", None) if context is not None else None
            edit = getattr(bot, "edit_message_text", None)
            if callable(edit):
                await edit(
                    chat_id=session.chat_id,
                    message_id=session.canonical_message_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=None,
                )
            else:
                source_id = self._positive_message_id(getattr(source_message, "message_id", None))
                if source_id != session.canonical_message_id:
                    return False
                await source_message.edit_text(
                    text,
                    reply_markup=reply_markup,
                    parse_mode=None,
                )
            return True
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            if "message is not modified" in str(exc).casefold():
                return True
            logger.warning(
                "Nova memory canonical edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Nova memory canonical edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False

    async def _nova_memory_access_changed(
        self,
        context: Any,
        session: NovaMemoryFlowSession,
        *,
        source_message: Any | None,
        query: Any | None = None,
    ) -> None:
        async with self._nova_memory_ui_lock:
            live = await self.nova_memory_sessions.get_exact(session)
            if live is None:
                return
            cleared = await self.nova_memory_sessions.clear_exact(live)
            if not cleared:
                return
            replacement = await self.nova_memory_sessions.current(
                owner_id=live.owner_id,
                telegram_user_id=live.telegram_user_id,
                chat_id=live.chat_id,
            )
            if replacement is not None:
                return
            if query is not None:
                await self._nova_memory_edit_query(
                    query,
                    NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                    None,
                    operation="access_changed",
                )
            elif source_message is not None:
                await self._nova_memory_edit(
                    context,
                    live,
                    NOVA_MEMORY_ACCESS_CHANGED_TEXT,
                    None,
                    source_message=source_message,
                )

    async def _nova_memory_access_values(
        self,
        telegram_user_id: int,
        chat_id: int,
    ) -> User | None:
        del chat_id
        try:
            async with self.db.sessions() as db_session:
                user = await db_session.scalar(
                    select(User).where(User.telegram_id == telegram_user_id)
                )
            if user is None or not self.nova_memory_available_for_tier(user.access_tier):
                return None
            status = await self.access_service.status(telegram_user_id)
            if (
                status is None
                or status.access_tier != user.access_tier
                or status.access_version != user.access_version
            ):
                return None
            return user
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Nova memory access lookup failed error_type=%s", type(exc).__name__)
            return None

    def _nova_memory_same_access(
        self,
        session: NovaMemoryFlowSession,
        user: User | None,
    ) -> bool:
        return bool(
            user is not None
            and self.nova_memory_available_for_tier(user.access_tier)
            and user.id == session.owner_id
            and user.telegram_id == session.telegram_user_id
            and user.access_tier == session.tier
            and user.access_version == session.access_version
        )

    def _nova_memory_voice_access_matches(
        self,
        user: User | None,
        *,
        owner_id: int | None,
        telegram_user_id: int | None,
        tier: str | AccessTier | None,
        access_version: int | None,
    ) -> bool:
        """Match an access read to the exact generation frozen before STT."""

        return bool(
            user is not None
            and owner_id is not None
            and telegram_user_id is not None
            and tier is not None
            and access_version is not None
            and self.nova_memory_available_for_tier(user.access_tier)
            and user.id == owner_id
            and user.telegram_id == telegram_user_id
            and user.access_tier == tier
            and user.access_version == access_version
        )

    @staticmethod
    def _nova_memory_same_generation(
        current: NovaMemoryFlowSession | None,
        expected: NovaMemoryFlowSession | None,
    ) -> bool:
        if expected is None:
            return current is None
        return bool(
            current is not None
            and current.id == expected.id
            and current.version == expected.version
            and current.access_version == expected.access_version
            and current.canonical_message_id == expected.canonical_message_id
        )

    async def _nova_memory_handoff(self, update: Any, user: User) -> None:
        await self._nova_memory_clear_guided_help(
            user.id,
            user.telegram_id,
            update.effective_chat.id,
        )
        reminder_store = getattr(self, "reminder_sessions", None)
        if reminder_store is None:
            return
        try:
            reminder_ui_lock = getattr(self, "_reminder_ui_lock", None)
            if reminder_ui_lock is None:
                return
            async with reminder_ui_lock:
                current = await reminder_store.current(
                    owner_id=user.id,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                )
                if current is not None:
                    await reminder_store.clear(
                        owner_id=current.owner_id,
                        telegram_user_id=current.telegram_user_id,
                        chat_id=current.chat_id,
                        session_id=current.id,
                    )
        except asyncio.CancelledError:
            raise
        except (TypeError, AttributeError) as exc:
            logger.warning("Nova memory reminder handoff failed error_type=%s", type(exc).__name__)

    async def _nova_memory_clear_if_guided_current(self, session: Any) -> bool:
        """Retire memory only while the published guided generation is still current."""

        store = getattr(self, "nova_sessions", None)
        guided_ui_lock = getattr(self, "_nova_ui_lock", None)
        if store is None or guided_ui_lock is None or session is None:
            return False
        async with self._nova_memory_launch_lock:
            async with guided_ui_lock:
                live = await store.get(
                    owner_id=session.owner_id,
                    telegram_user_id=session.telegram_user_id,
                    chat_id=session.chat_id,
                    access_version=session.access_version,
                    canonical_message_id=session.canonical_message_id,
                    tier=session.tier,
                    session_id=session.id,
                )
                if live is None:
                    return False
                async with self._nova_memory_ui_lock:
                    current = await self.nova_memory_sessions.current(
                        owner_id=session.owner_id,
                        telegram_user_id=session.telegram_user_id,
                        chat_id=session.chat_id,
                    )
                    if current is None:
                        return False
                    return await self.nova_memory_sessions.clear(
                        owner_id=current.owner_id,
                        telegram_user_id=current.telegram_user_id,
                        chat_id=current.chat_id,
                        session_id=current.id,
                    )

    async def _nova_memory_clear_if_reminder_current(self, session: Any) -> bool:
        """Retire memory only while the published reminder generation is still current."""

        store = getattr(self, "reminder_sessions", None)
        reminder_ui_lock = getattr(self, "_reminder_ui_lock", None)
        if store is None or reminder_ui_lock is None or session is None:
            return False
        async with self._nova_memory_launch_lock:
            async with reminder_ui_lock:
                live = await store.get_exact(session)
                if live is None:
                    return False
                async with self._nova_memory_ui_lock:
                    current = await self.nova_memory_sessions.current(
                        owner_id=session.owner_id,
                        telegram_user_id=session.telegram_user_id,
                        chat_id=session.chat_id,
                    )
                    if current is None:
                        return False
                    return await self.nova_memory_sessions.clear(
                        owner_id=current.owner_id,
                        telegram_user_id=current.telegram_user_id,
                        chat_id=current.chat_id,
                        session_id=current.id,
                    )

    async def _nova_memory_clear_guided_help(
        self,
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
    ) -> None:
        store = getattr(self, "nova_sessions", None)
        if store is None:
            return
        try:
            current = await store.current(
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
            )
        except (TypeError, AttributeError):
            current = None
        if current is None:
            return
        # Guided Nova performs its reciprocal post-create memory clear outside
        # its UI lock. Serializing this exact clear with that UI lock prevents a
        # reverse interleaving from retiring both freshly published flows.
        clear_bound = getattr(self, "nova_clear_bound", None)
        if callable(clear_bound):
            await clear_bound(
                owner_id,
                chat_id,
                session_id=current.id,
            )

    async def _nova_memory_reply_unavailable(self, message: Any) -> None:
        try:
            await message.reply_text(NOVA_MEMORY_UNAVAILABLE_TEXT, parse_mode=None)
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning("Nova memory unavailable reply failed error_type=%s", type(exc).__name__)

    @staticmethod
    async def _nova_memory_answer(
        query: Any,
        text: str | None = None,
        *,
        show_alert: bool = False,
    ) -> None:
        try:
            if text is None:
                await query.answer()
            else:
                await query.answer(text, show_alert=show_alert)
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning("Nova memory callback answer failed error_type=%s", type(exc).__name__)

    @staticmethod
    async def _nova_memory_edit_transient(message: Any, text: str) -> None:
        try:
            await message.edit_text(text, reply_markup=None, parse_mode=None)
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            if "message is not modified" not in str(exc).casefold():
                logger.warning("Nova memory voice edit failed error_type=%s", type(exc).__name__)

    async def _nova_memory_delete_transient(self, message: Any, context: Any) -> None:
        try:
            delete = getattr(message, "delete", None)
            if callable(delete):
                await delete()
                return
            message_id = self._positive_message_id(getattr(message, "message_id", None))
            if message_id is not None:
                await context.bot.delete_message(
                    chat_id=getattr(message, "chat_id", None),
                    message_id=message_id,
                )
        except asyncio.CancelledError:
            raise
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Nova memory voice transient cleanup failed error_type=%s", type(exc).__name__
            )

    @staticmethod
    def _positive_id(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @classmethod
    def _positive_message_id(cls, value: Any) -> int | None:
        return value if cls._positive_id(value) else None

    @staticmethod
    def _truncate_utf16(value: str, max_units: int) -> str:
        encoded = value.encode("utf-16-le")
        if len(encoded) <= max_units * 2:
            return value
        clipped = encoded[: max(0, max_units - 1) * 2].decode("utf-16-le", errors="ignore")
        return clipped.rstrip() + "…"

    async def _nova_memory_confirm_update(
        self,
        update: Any,
        context: Any,
        claim: NovaMemoryCapabilityClaim,
    ) -> None:
        session = claim.session
        capability = claim.capability
        public_id = capability.public_id or session.item_public_id
        expected_version = capability.expected_item_version or session.item_version
        if public_id is None or expected_version is None or session.candidate_category is None:
            await self._nova_memory_render_failure(
                context,
                session,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_before_mutation(update, context, session):
            return
        try:
            result = await self.nova_memory_service.update(
                telegram_actor_id=session.telegram_user_id,
                public_id=public_id,
                expected_version=expected_version,
                expected_access_version=session.access_version,
                content=session.candidate_content,
                category=session.candidate_category,
            )
        except asyncio.CancelledError:
            raise
        except NovaMemoryValidationError:
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_VALIDATION_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        except NovaMemoryStorageError as exc:
            logger.warning("Nova memory update failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_STORAGE_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_after_mutation(update, context, session):
            return
        if result.status in {"updated", "unchanged"} and result.item is not None:
            await self._nova_memory_show_detail(
                update,
                context,
                session,
                public_id=result.item.public_id,
                list_filter=session.list_filter,
                page=session.page,
                query=update.callback_query,
                notice=(
                    "✅ Изменения сохранены."
                    if result.status == "updated"
                    else "Запись уже содержит эти данные."
                ),
            )
            return
        if result.status == "conflict":
            await self._nova_memory_show_detail(
                update,
                context,
                session,
                public_id=public_id,
                list_filter=session.list_filter,
                page=session.page,
                query=update.callback_query,
                notice="Такая формулировка уже есть. Исходная запись не изменена.",
            )
            return
        await self._nova_memory_handle_domain_failure(update, context, session, result.status)

    async def _nova_memory_toggle_important(
        self,
        update: Any,
        context: Any,
        claim: NovaMemoryCapabilityClaim,
    ) -> None:
        session = claim.session
        public_id = claim.capability.public_id
        expected_version = claim.capability.expected_item_version
        if public_id is None or expected_version is None:
            await self._nova_memory_render_failure(
                context,
                session,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_before_mutation(update, context, session):
            return
        try:
            lookup = await self.nova_memory_service.get(
                telegram_actor_id=session.telegram_user_id,
                public_id=public_id,
            )
            if lookup.status != "found" or lookup.item is None:
                await self._nova_memory_handle_domain_failure(
                    update,
                    context,
                    session,
                    "access_denied" if lookup.status == "access_denied" else "not_found",
                )
                return
            if lookup.item.version != expected_version:
                await self._nova_memory_handle_domain_failure(
                    update,
                    context,
                    session,
                    "stale",
                )
                return
            if not await self._nova_memory_before_mutation(update, context, session):
                return
            result = await self.nova_memory_service.set_important(
                telegram_actor_id=session.telegram_user_id,
                public_id=public_id,
                expected_version=expected_version,
                expected_access_version=session.access_version,
                important=not lookup.item.important,
            )
        except asyncio.CancelledError:
            raise
        except NovaMemoryValidationError:
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_VALIDATION_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        except NovaMemoryStorageError as exc:
            logger.warning("Nova memory importance failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_STORAGE_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_after_mutation(update, context, session):
            return
        if result.status in {"importance_changed", "unchanged"} and result.item is not None:
            await self._nova_memory_show_detail(
                update,
                context,
                session,
                public_id=result.item.public_id,
                list_filter=session.list_filter,
                page=session.page,
                query=update.callback_query,
            )
            return
        await self._nova_memory_handle_domain_failure(update, context, session, result.status)

    async def _nova_memory_confirm_delete(
        self,
        update: Any,
        context: Any,
        claim: NovaMemoryCapabilityClaim,
    ) -> None:
        session = claim.session
        public_id = claim.capability.public_id
        expected_version = claim.capability.expected_item_version
        if public_id is None or expected_version is None:
            await self._nova_memory_render_failure(
                context,
                session,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_before_mutation(update, context, session):
            return
        try:
            result = await self.nova_memory_service.delete(
                telegram_actor_id=session.telegram_user_id,
                public_id=public_id,
                expected_version=expected_version,
                expected_access_version=session.access_version,
            )
        except asyncio.CancelledError:
            raise
        except NovaMemoryValidationError:
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_VALIDATION_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        except NovaMemoryStorageError as exc:
            logger.warning("Nova memory delete failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_STORAGE_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_after_mutation(update, context, session):
            return
        if result.status == "deleted":
            await self._nova_memory_transition_root_notice(
                update,
                context,
                session,
                "Запись удалена из активной памяти Nova.",
                query=update.callback_query,
            )
            return
        await self._nova_memory_handle_domain_failure(update, context, session, result.status)

    async def _nova_memory_confirm_delete_all(
        self,
        update: Any,
        context: Any,
        claim: NovaMemoryCapabilityClaim,
    ) -> None:
        session = claim.session
        revision = claim.capability.expected_collection_revision
        if revision is None:
            await self._nova_memory_render_failure(
                context,
                session,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_before_mutation(update, context, session):
            return
        try:
            result = await self.nova_memory_service.delete_all(
                telegram_actor_id=session.telegram_user_id,
                expected_access_version=session.access_version,
                expected_collection_revision=revision,
            )
        except asyncio.CancelledError:
            raise
        except NovaMemoryValidationError:
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_VALIDATION_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        except NovaMemoryStorageError as exc:
            logger.warning("Nova memory delete-all failed error_type=%s", type(exc).__name__)
            await self._nova_memory_render_mutation_error(
                context,
                session,
                NOVA_MEMORY_STORAGE_FAILURE_TEXT,
                query=update.callback_query,
            )
            return
        if not await self._nova_memory_after_mutation(update, context, session):
            return
        if result.status in {"deleted_all", "unchanged"}:
            await self._nova_memory_transition_root_notice(
                update,
                context,
                session,
                (
                    f"Nova забыла {result.affected_count} записей."
                    if result.status == "deleted_all"
                    else "Память уже пуста."
                ),
                query=update.callback_query,
            )
            return
        if result.status == "stale":
            await self._nova_memory_transition_root_notice(
                update,
                context,
                session,
                "Память успела измениться. Ничего не удалено — проверь записи и повтори.",
                query=update.callback_query,
            )
            return
        await self._nova_memory_handle_domain_failure(update, context, session, result.status)


__all__ = [
    "NOVA_MEMORY_ACCESS_CHANGED_TEXT",
    "NOVA_MEMORY_ROOT_TEXT",
    "NOVA_MEMORY_STALE_ALERT",
    "NovaMemoryHandlers",
    "NovaMemoryVoiceFence",
]
