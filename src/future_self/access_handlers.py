from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import ValidationError
from telegram import (
    BotCommand,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop, ContextTypes

from .access import BLOCKED, GUEST, AccessTier, is_full_access_tier, require_access_tier
from .ai import GUEST_DEMO_MAX_INPUT_CHARS
from .config import GUEST_TEXT_PROVIDER_TIMEOUT_SECONDS
from .guest_access import (
    GuestDemoKind,
    GuestProviderStartOutcome,
    GuestQuotaDenialReason,
    GuestReservationOutcome,
    GuestSessionOutcome,
    GuestSessionSnapshot,
    GuestSessionStatus,
)
from .navigation import CommandSpec, public_commands
from .schemas import GuestFirstStep, GuestThoughtBreakdown

logger = logging.getLogger(__name__)

GUEST_COMMANDS = (
    CommandSpec("start", "Начать"),
    CommandSpec("menu", "Главное меню"),
    CommandSpec("help", "Как это работает"),
)
BLOCKED_COMMANDS = (CommandSpec("start", "Начать"),)

GUEST_ROOT_TEXT = """👋 Я — «Моя будущая версия»

Личный AI-ассистент, который помогает разгружать голову, видеть главное и превращать желаемое будущее в конкретные действия.

Я умею работать с мыслями, задачами, целями, картой желаний, самочувствием и подготовкой к важным событиям.

В гостевом режиме доступны 2 бесплатных AI-разбора."""

GUEST_DEMOS_TEXT = """✨ Бесплатная демонстрация

Можно попробовать два сценария:

📝 Разобрать мысль — превратить свободный текст в понятную структуру и следующий шаг.

🌱 Найти первый шаг — превратить цель или желаемое изменение в небольшое действие."""

GUEST_THOUGHT_INPUT_TEXT = f"""📝 Разобрать мысль

Опишите мысль одним сообщением. Я выделю тип, короткий заголовок, суть и следующий шаг.

Лимит — {GUEST_DEMO_MAX_INPUT_CHARS} символов."""

GUEST_FIRST_STEP_INPUT_TEXT = f"""🌱 Найти первый шаг

Опишите цель, желание или изменение. Я предложу небольшой и конкретный первый шаг.

Лимит — {GUEST_DEMO_MAX_INPUT_CHARS} символов."""

GUEST_PROCESSING_TEXTS = {
    GuestDemoKind.THOUGHT_BREAKDOWN: "⏳ Разбираю запрос…",
    GuestDemoKind.FIRST_STEP: "⏳ Ищу небольшой первый шаг…",
}

GUEST_PROVIDER_ERROR_TEXT = """Сейчас обработать запрос не получилось.

Количество бесплатных разборов не изменилось. Отправьте текст ещё раз."""

GUEST_LIFETIME_EXHAUSTED_TEXT = """🎁 Бесплатные разборы закончились

Вы уже использовали оба бесплатных разбора.

Чтобы получить подписку или заказать такого же собственного бота, напишите Назару Сергеевичу."""

GUEST_DISABLED_TEXT = """AI-демонстрация временно недоступна.

Количество бесплатных разборов не изменилось. Попробуйте немного позже."""

GUEST_GLOBAL_EXHAUSTED_TEXT = """На сегодня общий лимит бесплатных разборов достигнут.

Количество ваших бесплатных разборов не изменилось. Попробуйте позже — дневной лимит обновится автоматически."""

GUEST_TEMPORARY_ERROR_TEXT = """Сервис временно недоступен.

Количество бесплатных разборов не изменилось. Попробуйте ещё раз немного позже."""

GUEST_PROCESSING_ALERT = "Запрос уже обрабатывается"
GUEST_ACCESS_CHANGED_TEXT = "Доступ изменился. Откройте /start, чтобы продолжить."
GUEST_RESULT_EXPIRED_TEXT = """⌛ Срок хранения результата истёк.

Выберите демо ещё раз или вернитесь в начало."""
GUEST_SESSION_EXPIRED_TEXT = """⌛ Сессия демо истекла.

Выберите демо ещё раз или вернитесь в начало."""

GUEST_HOW_TEXT = """⚙️ Как это работает

1. Вы описываете мысль или цель обычными словами.
2. Бот структурирует её и предлагает небольшой следующий шаг.
3. В полной версии результат можно сохранять, связывать с задачами и использовать в ежедневном сопровождении."""

GUEST_FEATURE_TEXTS = {
    "guest:feature:day": """🌱 Фокус дня

Бот связывает образ желаемого будущего с небольшим действием сегодня и помогает спокойно подвести итог вечером.""",
    "guest:feature:notes": """📝 Мысли и заметки

Можно написать мысль обычными словами. Бот помогает понять, что это: идея, задача, желание или заметка — и предлагает аккуратную карточку перед сохранением.""",
    "guest:feature:tasks": """✅ Задачи и напоминания

Бот помогает сформулировать задачу, отделить срок события от времени напоминания и не потерять следующий шаг.""",
    "guest:feature:vision": """🎯 Карта желаний

Желание превращается в понятную карточку: смысл, первый шаг, задача и, при необходимости, визуальный образ.""",
    "guest:feature:health": """❤️ Самочувствие

Короткие check-in помогают видеть субъективную динамику состояния. Бот не ставит диагнозы и не заменяет врача.""",
    "guest:feature:doctor": """🩺 Подготовка к врачу

Бот помогает собрать факты, вопросы и наблюдения перед визитом, не ставя диагнозов и не назначая лечение.""",
}

GUEST_CALLBACK_ROUTES = frozenset(
    {
        "guest:root",
        "guest:demos",
        "guest:demo:thought",
        "guest:demo:first-step",
        "guest:features",
        "guest:how",
        "guest:access",
        *GUEST_FEATURE_TEXTS,
    }
)

SERVICE_UNAVAILABLE_TEXT = "Сервис временно недоступен. Попробуйте немного позже."
FULL_VERSION_ALERT = "Эта функция доступна в полной версии."
STALE_GUEST_ALERT = "Эта кнопка устарела. Откройте гостевое меню заново."
GUEST_COMMAND_NOTICE = "Эта команда доступна в полной версии. Ниже — возможности гостевого режима."
GUEST_TEXT_NOTICE = "Для пробной операции выберите кнопку в гостевом меню."
GUEST_MEDIA_NOTICE = (
    "Гостевая демонстрация пока работает только через кнопки. "
    "Файлы, фото и аудио в этом режиме не обрабатываются."
)

_ROOT_MARKUP = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("✨ Попробовать бесплатно", callback_data="guest:demos")],
        [InlineKeyboardButton("🧭 Что я умею", callback_data="guest:features")],
        [InlineKeyboardButton("⚙️ Как это работает", callback_data="guest:how")],
        [InlineKeyboardButton("💬 Подписка или свой бот", callback_data="guest:access")],
        [InlineKeyboardButton("🧩 Другие проекты", url="https://naz-ai-lab.ru")],
    ]
)

_DEMOS_MARKUP = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("📝 Разобрать мысль", callback_data="guest:demo:thought")],
        [InlineKeyboardButton("🌱 Найти первый шаг", callback_data="guest:demo:first-step")],
        [InlineKeyboardButton("← Назад", callback_data="guest:root")],
    ]
)

_DEMO_INPUT_MARKUP = InlineKeyboardMarkup(
    [[InlineKeyboardButton("← Назад", callback_data="guest:demos")]]
)

_DEMO_RETRY_MARKUP = InlineKeyboardMarkup(
    [[InlineKeyboardButton("← Назад", callback_data="guest:demos")]]
)

_FEATURES_MARKUP = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("🌱 Фокус дня", callback_data="guest:feature:day")],
        [InlineKeyboardButton("📝 Мысли и заметки", callback_data="guest:feature:notes")],
        [InlineKeyboardButton("✅ Задачи и напоминания", callback_data="guest:feature:tasks")],
        [InlineKeyboardButton("🎯 Карта желаний", callback_data="guest:feature:vision")],
        [InlineKeyboardButton("❤️ Самочувствие", callback_data="guest:feature:health")],
        [InlineKeyboardButton("🩺 Подготовка к врачу", callback_data="guest:feature:doctor")],
        [InlineKeyboardButton("← Назад", callback_data="guest:root")],
    ]
)

_FEATURE_MARKUP = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("← К возможностям", callback_data="guest:features")],
        [InlineKeyboardButton("🏠 В начало", callback_data="guest:root")],
    ]
)

_HOW_MARKUP = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("✨ Попробовать", callback_data="guest:demos")],
        [InlineKeyboardButton("🏠 В начало", callback_data="guest:root")],
    ]
)


def _access_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💬 Написать @Nazar_38rus", url="https://t.me/Nazar_38rus")],
            [InlineKeyboardButton("🧩 Другие проекты Nazar AI Lab", url="https://naz-ai-lab.ru")],
            [InlineKeyboardButton("🏠 В начало", callback_data="guest:root")],
        ]
    )


def _guest_result_markup(remaining_operations: int) -> InlineKeyboardMarkup:
    if remaining_operations > 0:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✨ Попробовать ещё", callback_data="guest:demos")],
                [InlineKeyboardButton("🏠 В начало", callback_data="guest:root")],
            ]
        )
    return _access_markup()


def _guest_temporary_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✨ Попробовать ещё", callback_data="guest:demos")],
            [InlineKeyboardButton("🏠 В начало", callback_data="guest:root")],
        ]
    )


def _blocked_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💬 Написать @Nazar_38rus", url="https://t.me/Nazar_38rus")],
            [InlineKeyboardButton("🧩 Другие проекты Nazar AI Lab", url="https://naz-ai-lab.ru")],
        ]
    )


class AccessHandlers:
    settings: Any
    _access_scope_cache: dict[int, tuple[AccessTier, int]]

    async def access_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_user = update.effective_user
        chat = update.effective_chat
        if telegram_user is None or chat is None:
            await self._access_fail_closed(update)
            raise ApplicationHandlerStop

        try:
            user = await self._user(telegram_user.id)
            tier = require_access_tier(user.access_tier)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_access_failure("Access gate database failure", exc, telegram_user.id)
            await self._access_fail_closed(update)
            raise ApplicationHandlerStop from None

        await self.nova_memory_sync_access(
            user,
            chat.id,
            context=context,
            source_message=update.effective_message,
        )
        await self.nova_sync_access(
            user,
            chat.id,
            context=context,
            source_message=update.effective_message,
        )
        await self.reminder_sync_access(
            user,
            chat.id,
            context=context,
            source_message=update.effective_message,
        )
        await self._sync_access_commands(
            context, chat.id, user.telegram_id, tier, user.access_version
        )
        if is_full_access_tier(tier):
            return
        try:
            if tier == GUEST:
                await self._handle_guest_update(update, context, user)
            else:
                await self.show_blocked_screen(update)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_access_failure("Restricted access dispatch failed", exc, telegram_user.id)
        raise ApplicationHandlerStop

    async def _sync_access_commands(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        telegram_id: int,
        tier: AccessTier,
        access_version: int,
    ) -> None:
        cache_key = (tier, access_version)
        if self._access_scope_cache.get(chat_id) == cache_key:
            return
        bot = getattr(context, "bot", None)
        if bot is None:
            return
        if tier == GUEST:
            specs = GUEST_COMMANDS
        elif tier == BLOCKED:
            specs = BLOCKED_COMMANDS
        else:
            specs = public_commands(
                getattr(self.settings, "enable_workspace_access", False),
                getattr(self.settings, "enable_knowledge_hub", False),
            )
        try:
            await bot.set_my_commands(
                [BotCommand(item.command, item.description) for item in specs],
                scope=BotCommandScopeChat(chat_id),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_access_failure("Access command scope sync failed", exc, telegram_id)
            return
        self._access_scope_cache[chat_id] = cache_key

    async def _handle_guest_update(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: Any,
    ) -> None:
        chat = update.effective_chat
        if chat is None:
            return
        try:
            pending = await self.guest_session_service.pending_result(
                user_id=user.id,
                chat_id=chat.id,
                access_version=user.access_version,
            )
        except Exception as exc:
            self._log_access_failure("Guest session lookup failed", exc, user.id)
            query = update.callback_query
            if query is not None:
                await query.answer(SERVICE_UNAVAILABLE_TEXT, show_alert=True)
            return
        session = pending.session
        if pending.outcome is GuestSessionOutcome.NOT_GUEST:
            query = update.callback_query
            if query is not None:
                await query.answer()
            if session is not None:
                await self._edit_guest_access_changed(
                    context.bot,
                    chat_id=session.chat_id,
                    message_id=session.prompt_message_id,
                )
            return
        message = update.effective_message
        text = getattr(message, "text", None) if message is not None else None
        command = (
            text.split(maxsplit=1)[0].partition("@")[0].casefold()
            if isinstance(text, str) and text.startswith("/")
            else None
        )
        query = update.callback_query
        if (
            session is not None
            and session.status in {GuestSessionStatus.AWAITING_INPUT, GuestSessionStatus.PROCESSING}
            and (command == "/help" or (query is not None and str(query.data or "") == "guest:how"))
        ):
            if query is not None:
                await query.answer()
            await self._guest_nova_flow_help(update, context, session)
            return
        if query is not None and str(query.data or "").startswith("guest:nova:flow:"):
            await self._guest_nova_flow_action(update, context, user, session)
            return
        if session is not None and session.status is GuestSessionStatus.RESULT_READY:
            if query is not None:
                await query.answer()
            await self._deliver_guest_result(
                context.bot,
                user_id=user.id,
                chat_id=chat.id,
                access_version=user.access_version,
            )
            return
        if session is not None and session.status is GuestSessionStatus.PROCESSING:
            if query is not None:
                await query.answer(GUEST_PROCESSING_ALERT, show_alert=True)
            return
        if query is not None:
            await self._handle_guest_callback(update, context, user, session)
            return
        if session is not None and session.status is GuestSessionStatus.AWAITING_INPUT:
            await self._handle_guest_awaiting_update(update, context, user, session)
            return
        if message is None:
            return
        if isinstance(text, str) and text.startswith("/"):
            if command in {"/start", "/menu"}:
                await self.nova_clear_bound(user.id, chat.id)
                await self.show_guest_root(update)
            elif command == "/help":
                await self._nova_open_message(message, update, user)
            elif command == "/cancel":
                current = await self.nova_sessions.current(
                    owner_id=user.id,
                    telegram_user_id=user.telegram_id,
                    chat_id=chat.id,
                )
                if current is None:
                    await self.show_guest_root(update)
                else:
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
                                chat_id=chat.id,
                                session_id=live.id,
                            )
                            await self._nova_edit_canonical(
                                context,
                                live,
                                "✨ Nova\n\nСессия завершена.",
                                None,
                                source_message=message,
                            )
            else:
                await self._reply_guest_screen(
                    update,
                    f"{GUEST_COMMAND_NOTICE}\n\n{GUEST_ROOT_TEXT}",
                    _ROOT_MARKUP,
                )
            return
        if (
            isinstance(text, str)
            and text.strip()
            and await self.nova_text_gate(
                update,
                context,
                user=user,
            )
        ):
            return
        if self._has_guest_media(message):
            await self._reply_guest_screen(update, GUEST_MEDIA_NOTICE, _ROOT_MARKUP)
            return
        if isinstance(text, str) and text.strip():
            await self._reply_guest_screen(
                update,
                f"{GUEST_TEXT_NOTICE}\n\n{GUEST_ROOT_TEXT}",
                _ROOT_MARKUP,
            )

    async def _guest_nova_flow_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: GuestSessionSnapshot,
    ) -> None:
        if session.prompt_message_id is None:
            return
        async with self._nova_ui_lock:
            try:
                pending = await self.guest_session_service.pending_result(
                    user_id=session.user_id,
                    chat_id=session.chat_id,
                    access_version=session.access_version,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_access_failure("Guest session lookup failed", exc, session.user_id)
                return
            live = pending.session
            if (
                live is None
                or live.session_id != session.session_id
                or live.prompt_message_id != session.prompt_message_id
                or live.status
                not in {GuestSessionStatus.AWAITING_INPUT, GuestSessionStatus.PROCESSING}
            ):
                return
            token = await self.navigation_flow_sessions.issue(
                update.effective_user.id,
                update.effective_chat.id,
                self._guest_nova_flow_key(live),
            )
            label = (
                "бесплатный AI-разбор"
                if live.status is GuestSessionStatus.PROCESSING
                else "ввод для бесплатного AI-разбора"
            )
            await self._edit_canonical_message(
                context.bot,
                chat_id=live.chat_id,
                message_id=live.prompt_message_id,
                text=f"✨ Nova\n\nСейчас не завершён сценарий: {label}. Что сделать?",
                markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "▶️ Продолжить текущий шаг",
                                callback_data=f"guest:nova:flow:continue:{token}",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "🏠 Выйти в главное меню",
                                callback_data=f"guest:nova:flow:exit:{token}",
                            )
                        ],
                    ]
                ),
            )

    async def _guest_nova_flow_action(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: Any,
        session: GuestSessionSnapshot | None,
    ) -> None:
        async with self._nova_ui_lock:
            if session is not None:
                try:
                    pending = await self.guest_session_service.pending_result(
                        user_id=session.user_id,
                        chat_id=session.chat_id,
                        access_version=session.access_version,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._log_access_failure("Guest session lookup failed", exc, session.user_id)
                    await update.callback_query.answer(
                        SERVICE_UNAVAILABLE_TEXT,
                        show_alert=True,
                    )
                    return
                live = pending.session
                session = (
                    live
                    if live is not None
                    and live.session_id == session.session_id
                    and live.prompt_message_id == session.prompt_message_id
                    else None
                )
            await self._guest_nova_flow_action_locked(update, context, user, session)

    async def _guest_nova_flow_action_locked(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: Any,
        session: GuestSessionSnapshot | None,
    ) -> None:
        query = update.callback_query
        parts = str(query.data or "").split(":")
        if len(parts) != 5 or parts[3] not in {"continue", "exit"}:
            await query.answer(STALE_GUEST_ALERT, show_alert=True)
            return
        capability = await self.navigation_flow_sessions.claim(
            parts[4],
            update.effective_user.id,
            update.effective_chat.id,
        )
        message_id = getattr(query.message, "message_id", None)
        if (
            capability is None
            or session is None
            or capability.flow != self._guest_nova_flow_key(session)
            or session.status
            not in {GuestSessionStatus.AWAITING_INPUT, GuestSessionStatus.PROCESSING}
            or message_id != session.prompt_message_id
        ):
            await query.answer(STALE_GUEST_ALERT, show_alert=True)
            return
        if parts[3] == "exit":
            cancelled = await self.guest_session_service.cancel(
                user_id=user.id,
                chat_id=session.chat_id,
            )
            if cancelled.outcome is GuestSessionOutcome.IN_PROGRESS:
                await query.answer(GUEST_PROCESSING_ALERT, show_alert=True)
                await self._edit_guest_screen(
                    query,
                    GUEST_PROCESSING_TEXTS[session.demo_kind],
                    None,
                )
                return
            if cancelled.outcome is not GuestSessionOutcome.CANCELLED:
                await query.answer(STALE_GUEST_ALERT, show_alert=True)
                return
            await query.answer()
            await self._edit_guest_screen(query, GUEST_ROOT_TEXT, _ROOT_MARKUP)
            return
        await query.answer()
        if session.status is GuestSessionStatus.PROCESSING:
            await self._edit_guest_screen(
                query,
                GUEST_PROCESSING_TEXTS[session.demo_kind],
                None,
            )
            return
        await self._edit_guest_screen(
            query,
            self._demo_input_text(session.demo_kind),
            _DEMO_INPUT_MARKUP,
        )

    @staticmethod
    def _guest_nova_flow_key(session: GuestSessionSnapshot) -> str:
        return f"guest_demo:{session.session_id}:{session.version}"

    async def _handle_guest_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: Any,
        session: GuestSessionSnapshot | None,
    ) -> None:
        query = update.callback_query
        data = query.data if isinstance(query.data, str) else ""
        if data.startswith("nova:"):
            await self.nova_callback(update, context)
            return
        if data not in GUEST_CALLBACK_ROUTES:
            if data.startswith("guest:"):
                await query.answer(STALE_GUEST_ALERT, show_alert=True)
                return
            await query.answer(FULL_VERSION_ALERT, show_alert=True)
            await self._edit_guest_screen(query, GUEST_ROOT_TEXT, _ROOT_MARKUP)
            return
        if data == "guest:how":
            if session is not None and session.status is GuestSessionStatus.AWAITING_INPUT:
                await query.answer()
                await self._guest_nova_flow_help(update, context, session)
                return
            await self.nova_navigation_help_callback(update, context)
            return
        if data == "guest:demo:thought":
            await self.nova_clear_bound(user.id, update.effective_chat.id)
            await self._start_guest_demo(
                update,
                context,
                user,
                GuestDemoKind.THOUGHT_BREAKDOWN,
            )
            return
        if data == "guest:demo:first-step":
            await self.nova_clear_bound(user.id, update.effective_chat.id)
            await self._start_guest_demo(
                update,
                context,
                user,
                GuestDemoKind.FIRST_STEP,
            )
            return
        await self.nova_clear_bound(user.id, update.effective_chat.id)
        await query.answer()
        if session is not None and session.status is GuestSessionStatus.AWAITING_INPUT:
            await self.guest_session_service.cancel(user_id=user.id, chat_id=session.chat_id)
            text, markup = self._guest_screen(data, update.effective_user.id)
            if session.prompt_message_id is not None:
                await self._edit_canonical_message(
                    context.bot,
                    chat_id=session.chat_id,
                    message_id=session.prompt_message_id,
                    text=text,
                    markup=markup,
                )
            return
        text, markup = self._guest_screen(data, update.effective_user.id)
        await self._edit_guest_screen(query, text, markup)

    async def _start_guest_demo(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: Any,
        demo_kind: GuestDemoKind,
    ) -> None:
        query = update.callback_query
        message = getattr(query, "message", None)
        message_id = getattr(message, "message_id", None)
        chat = update.effective_chat
        if (
            message is None
            or isinstance(message_id, bool)
            or not isinstance(message_id, int)
            or message_id <= 0
            or chat is None
        ):
            await query.answer(STALE_GUEST_ALERT, show_alert=True)
            return
        try:
            snapshot = await self.guest_quota_service.snapshot(user.id)
        except Exception as exc:
            self._log_access_failure("Guest quota snapshot failed", exc, user.id)
            await query.answer(SERVICE_UNAVAILABLE_TEXT, show_alert=True)
            return
        quota_screen = self._quota_screen(snapshot)
        if quota_screen is not None:
            await query.answer()
            await self._edit_guest_screen(query, *quota_screen)
            return
        try:
            started = await self.guest_session_service.start_session(
                user_id=user.id,
                chat_id=chat.id,
                access_version=user.access_version,
                demo_kind=demo_kind,
                prompt_message_id=message_id,
            )
        except Exception as exc:
            self._log_access_failure("Guest session start failed", exc, user.id)
            await query.answer(SERVICE_UNAVAILABLE_TEXT, show_alert=True)
            return
        if started.outcome is GuestSessionOutcome.IN_PROGRESS:
            await query.answer(GUEST_PROCESSING_ALERT, show_alert=True)
            return
        if started.outcome is GuestSessionOutcome.RESULT_PENDING:
            await query.answer()
            await self._deliver_guest_result(
                context.bot,
                user_id=user.id,
                chat_id=chat.id,
                access_version=user.access_version,
            )
            return
        if started.outcome is not GuestSessionOutcome.STARTED:
            await query.answer(STALE_GUEST_ALERT, show_alert=True)
            return
        started_session = started.session
        if started_session is None:
            return
        try:
            await query.answer()
        except asyncio.CancelledError:
            await self._shielded_unshown_session_cleanup(
                started_session,
                preserve_original_cancellation=True,
            )
            raise
        except Exception as exc:
            self._log_access_failure("Guest demo callback answer failed", exc, user.id)
        try:
            edited = await self._edit_guest_screen(
                query,
                self._demo_input_text(demo_kind),
                _DEMO_INPUT_MARKUP,
            )
        except asyncio.CancelledError:
            await self._shielded_unshown_session_cleanup(
                started_session,
                preserve_original_cancellation=True,
            )
            raise
        if not edited:
            await self._shielded_unshown_session_cleanup(started_session)

    async def _cleanup_unshown_session(self, session: GuestSessionSnapshot) -> None:
        try:
            await self.guest_session_service.cancel_unshown_awaiting(
                user_id=session.user_id,
                chat_id=session.chat_id,
                access_version=session.access_version,
                session_version=session.version,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_access_failure(
                "Guest invisible session cleanup failed",
                exc,
                session.user_id,
            )

    async def _shielded_unshown_session_cleanup(
        self,
        session: GuestSessionSnapshot,
        *,
        preserve_original_cancellation: bool = False,
    ) -> None:
        try:
            cleanup = asyncio.create_task(
                self._cleanup_unshown_session(session),
                name="guest-unshown-session-cleanup",
            )
        except Exception as exc:
            self._log_access_failure(
                "Guest invisible session cleanup scheduling failed",
                exc,
                session.user_id,
            )
            return
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cleanup.add_done_callback(self._consume_cleanup_task_result)
            if not preserve_original_cancellation:
                raise

    @staticmethod
    def _consume_cleanup_task_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except BaseException:
            return

    async def _handle_guest_awaiting_update(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: Any,
        session: GuestSessionSnapshot,
    ) -> None:
        message = update.effective_message
        if message is None or session.prompt_message_id is None:
            return
        text = getattr(message, "text", None)
        if isinstance(text, str) and text.startswith("/"):
            await self.guest_session_service.cancel(user_id=user.id, chat_id=session.chat_id)
            command = text.split(maxsplit=1)[0].partition("@")[0].casefold()
            if command == "/help":
                screen = (GUEST_HOW_TEXT, _HOW_MARKUP)
            else:
                screen = (GUEST_ROOT_TEXT, _ROOT_MARKUP)
            await self._edit_canonical_message(
                context.bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
                text=screen[0],
                markup=screen[1],
            )
            return
        if self._has_guest_media(message):
            await self._edit_canonical_message(
                context.bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
                text=(
                    "Отправьте, пожалуйста, только текст одним сообщением.\n\n"
                    + self._demo_input_text(session.demo_kind)
                ),
                markup=_DEMO_INPUT_MARKUP,
            )
            return
        if not isinstance(text, str):
            return
        cleaned_text = text.strip()
        if not cleaned_text or len(cleaned_text) > GUEST_DEMO_MAX_INPUT_CHARS:
            hint = (
                "Сообщение не должно быть пустым."
                if not cleaned_text
                else f"Сократите текст до {GUEST_DEMO_MAX_INPUT_CHARS} символов."
            )
            await self._edit_canonical_message(
                context.bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
                text=f"{hint}\n\n{self._demo_input_text(session.demo_kind)}",
                markup=_DEMO_INPUT_MARKUP,
            )
            return
        await self._prepare_guest_operation(
            update,
            context,
            user,
            session,
            cleaned_text,
        )

    async def _resolve_stale_guest_claim(
        self,
        bot: Any,
        user: Any,
        session: GuestSessionSnapshot,
    ) -> None:
        try:
            current = await self.access_service.status(user.telegram_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_access_failure("Guest stale claim access check failed", exc, user.id)
            return
        if (
            current is None
            or current.access_tier != GUEST
            or current.access_version != session.access_version
        ):
            await self._edit_guest_access_changed(
                bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
            )

    async def _prepare_guest_operation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: Any,
        session: GuestSessionSnapshot,
        cleaned_text: str,
    ) -> None:
        message = update.effective_message
        update_id = getattr(update, "update_id", None)
        message_id = getattr(message, "message_id", None)
        if (
            isinstance(update_id, bool)
            or not isinstance(update_id, int)
            or update_id <= 0
            or isinstance(message_id, bool)
            or not isinstance(message_id, int)
            or message_id <= 0
        ):
            return
        claimed = await self.guest_session_service.claim_input(
            user_id=user.id,
            chat_id=session.chat_id,
            access_version=user.access_version,
            telegram_update_id=update_id,
            telegram_message_id=message_id,
        )
        if claimed.outcome is GuestSessionOutcome.NOT_GUEST:
            await self._edit_guest_access_changed(
                context.bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
            )
            return
        if claimed.outcome is GuestSessionOutcome.STALE:
            await self._resolve_stale_guest_claim(context.bot, user, session)
            return
        if claimed.outcome is GuestSessionOutcome.EXPIRED:
            await self._edit_guest_session_expired(
                context.bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
            )
            return
        if claimed.outcome is GuestSessionOutcome.RESULT_PENDING:
            await self._deliver_guest_result(
                context.bot,
                user_id=user.id,
                chat_id=session.chat_id,
                access_version=session.access_version,
            )
            return
        if claimed.outcome is not GuestSessionOutcome.CLAIMED or claimed.session is None:
            return
        claimed_session = claimed.session
        try:
            reservation = await self.guest_quota_service.reserve(
                user_id=user.id,
                demo_kind=claimed_session.demo_kind,
                idempotency_key=(f"guest:{user.id}:{update_id}:{claimed_session.demo_kind.value}"),
                telegram_update_id=update_id,
            )
        except Exception as exc:
            self._log_access_failure("Guest reservation failed", exc, user.id)
            await self._resolve_guest_reservation_denial(
                context.bot,
                user_id=user.id,
                access_version=user.access_version,
                session=claimed_session,
                reason=GuestQuotaDenialReason.UNAVAILABLE,
            )
            return
        if not reservation.can_bind_session or reservation.reservation is None:
            await self._resolve_guest_reservation_denial(
                context.bot,
                user_id=user.id,
                access_version=user.access_version,
                session=claimed_session,
                reason=reservation.denial_reason or GuestQuotaDenialReason.UNAVAILABLE,
            )
            return
        token = reservation.reservation.reservation_token
        try:
            bound = await self.guest_session_service.bind_reservation(
                user_id=user.id,
                chat_id=claimed_session.chat_id,
                access_version=user.access_version,
                session_version=claimed_session.version,
                reservation_token=token,
            )
        except Exception as exc:
            self._log_access_failure("Guest reservation bind failed", exc, user.id)
            await self._cleanup_failed_bind(
                context.bot,
                token=token,
                user_id=user.id,
                access_version=user.access_version,
                claimed_session=claimed_session,
            )
            return
        if bound.outcome is GuestSessionOutcome.NOT_GUEST:
            await self._cleanup_failed_bind(
                context.bot,
                token=token,
                user_id=user.id,
                access_version=user.access_version,
                claimed_session=claimed_session,
                denial_reason=GuestQuotaDenialReason.NOT_GUEST,
            )
            return
        if bound.outcome is GuestSessionOutcome.RETRY_READY and bound.session is not None:
            await self._edit_guest_retry(context.bot, bound.session)
            return
        if bound.outcome is not GuestSessionOutcome.BOUND or bound.session is None:
            await self._cleanup_failed_bind(
                context.bot,
                token=token,
                user_id=user.id,
                access_version=user.access_version,
                claimed_session=claimed_session,
            )
            return
        bound_session = bound.session
        processing_edited = await self._edit_canonical_message(
            context.bot,
            chat_id=bound_session.chat_id,
            message_id=bound_session.prompt_message_id,
            text=GUEST_PROCESSING_TEXTS[bound_session.demo_kind],
            markup=None,
            message_not_modified_is_success=True,
        )
        if not processing_edited:
            await self._pre_provider_failure(
                context.bot,
                token=token,
                user_id=user.id,
                access_version=user.access_version,
                session=bound_session,
                render_retry=False,
            )
            return
        worker = self._guest_provider_worker(
            cleaned_text=cleaned_text,
            user_id=user.id,
            chat_id=bound_session.chat_id,
            access_version=user.access_version,
            session_version=bound_session.version,
            reservation_token=token,
            demo_kind=bound_session.demo_kind,
            prompt_message_id=bound_session.prompt_message_id,
            bot=context.bot,
        )
        try:
            context.application.create_task(
                worker,
                name=(
                    f"guest-demo:{user.id}:{bound_session.chat_id}:"
                    f"{update_id}:{bound_session.demo_kind.value}"
                ),
            )
        except Exception as exc:
            worker.close()
            self._log_guest_worker_failure(
                "Guest worker scheduling failed",
                exc,
                user.id,
                bound_session.demo_kind,
            )
            await self._pre_provider_failure(
                context.bot,
                token=token,
                user_id=user.id,
                access_version=user.access_version,
                session=bound_session,
                render_retry=True,
            )

    async def _resolve_guest_reservation_denial(
        self,
        bot: Any,
        *,
        user_id: int,
        access_version: int,
        session: GuestSessionSnapshot,
        reason: GuestQuotaDenialReason,
    ) -> None:
        await self.guest_session_service.resolve_reservation_denial(
            user_id=user_id,
            chat_id=session.chat_id,
            access_version=access_version,
            session_version=session.version,
            denial_reason=reason,
        )
        if reason is GuestQuotaDenialReason.NOT_GUEST:
            await self._edit_guest_access_changed(
                bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
            )
            return
        text, markup = self._reservation_denial_screen(reason)
        await self._edit_canonical_message(
            bot,
            chat_id=session.chat_id,
            message_id=session.prompt_message_id,
            text=text,
            markup=markup,
        )

    async def _cleanup_failed_bind(
        self,
        bot: Any,
        *,
        token: str,
        user_id: int,
        access_version: int,
        claimed_session: GuestSessionSnapshot,
        denial_reason: GuestQuotaDenialReason = GuestQuotaDenialReason.UNAVAILABLE,
    ) -> None:
        try:
            await self.guest_quota_service.fail(token)
        except Exception as exc:
            self._log_access_failure("Guest provisional cleanup failed", exc, user_id)
        try:
            await self._resolve_guest_reservation_denial(
                bot,
                user_id=user_id,
                access_version=access_version,
                session=claimed_session,
                reason=denial_reason,
            )
        except Exception as exc:
            self._log_access_failure("Guest session retry cleanup failed", exc, user_id)

    async def _pre_provider_failure(
        self,
        bot: Any,
        *,
        token: str,
        user_id: int,
        access_version: int,
        session: GuestSessionSnapshot,
        render_retry: bool,
    ) -> None:
        try:
            failed = await self.guest_session_service.fail_processing(
                reservation_token=token,
                user_id=user_id,
                chat_id=session.chat_id,
                access_version=access_version,
                session_version=session.version,
            )
        except Exception as exc:
            self._log_access_failure("Guest pre-provider cleanup failed", exc, user_id)
            return
        if (
            render_retry
            and failed.session is not None
            and failed.session.status is GuestSessionStatus.AWAITING_INPUT
        ):
            await self._edit_guest_retry(bot, failed.session)
        elif failed.session is not None and failed.session.status is GuestSessionStatus.CANCELLED:
            await self._edit_guest_access_changed(
                bot,
                chat_id=failed.session.chat_id,
                message_id=failed.session.prompt_message_id,
            )

    async def _guest_provider_worker(
        self,
        *,
        cleaned_text: str,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        reservation_token: str,
        demo_kind: GuestDemoKind,
        prompt_message_id: int,
        bot: Any,
    ) -> None:
        try:
            started = await self.guest_session_service.begin_provider_call(
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
            )
        except asyncio.CancelledError:
            await self._shielded_guest_failure(
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
                demo_kind=demo_kind,
            )
            raise
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest provider start failed",
                exc,
                user_id,
                demo_kind,
            )
            await self._finish_guest_provider_failure(
                bot,
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
                demo_kind=demo_kind,
                prompt_message_id=prompt_message_id,
            )
            return
        if not started.can_invoke_provider:
            if started.outcome is GuestProviderStartOutcome.ALREADY_STARTED:
                return
            if started.outcome is GuestProviderStartOutcome.EXPIRED:
                if started.session is not None:
                    await self._edit_guest_retry(bot, started.session)
                return
            if started.outcome in {
                GuestProviderStartOutcome.ACCESS_CHANGED,
                GuestProviderStartOutcome.NOT_GUEST,
            }:
                try:
                    await self.guest_session_service.fail_processing(
                        reservation_token=reservation_token,
                        user_id=user_id,
                        chat_id=chat_id,
                        access_version=access_version,
                        session_version=session_version,
                    )
                except Exception as exc:
                    self._log_guest_worker_failure(
                        "Guest access-change cleanup failed",
                        exc,
                        user_id,
                        demo_kind,
                    )
                await self._edit_guest_access_changed(
                    bot,
                    chat_id=chat_id,
                    message_id=prompt_message_id,
                )
                return
            if started.outcome is GuestProviderStartOutcome.STALE:
                if (
                    started.session is not None
                    and started.session.status is GuestSessionStatus.AWAITING_INPUT
                ):
                    await self._edit_guest_retry(bot, started.session)
                    return
                if started.provider_started_at is not None:
                    return
                await self._finish_guest_provider_failure(
                    bot,
                    reservation_token=reservation_token,
                    user_id=user_id,
                    chat_id=chat_id,
                    access_version=access_version,
                    session_version=session_version,
                    demo_kind=demo_kind,
                    prompt_message_id=prompt_message_id,
                )
            return
        try:
            async with asyncio.timeout(GUEST_TEXT_PROVIDER_TIMEOUT_SECONDS):
                if demo_kind is GuestDemoKind.THOUGHT_BREAKDOWN:
                    result = await self.ai.guest_thought_breakdown(cleaned_text)
                elif demo_kind is GuestDemoKind.FIRST_STEP:
                    result = await self.ai.guest_first_step(cleaned_text)
                else:  # pragma: no cover - fenced by the domain enum
                    raise ValueError("unsupported guest demo kind")
            payload = result.model_dump(mode="json")
        except asyncio.CancelledError:
            await self._shielded_guest_failure(
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
                demo_kind=demo_kind,
            )
            raise
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest provider call failed",
                exc,
                user_id,
                demo_kind,
            )
            await self._finish_guest_provider_failure(
                bot,
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
                demo_kind=demo_kind,
                prompt_message_id=prompt_message_id,
            )
            return
        try:
            completion = await self.guest_session_service.complete_with_result(
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
                result_payload=payload,
            )
        except asyncio.CancelledError:
            await self._shielded_guest_failure(
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
                demo_kind=demo_kind,
            )
            raise
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest completion failed",
                exc,
                user_id,
                demo_kind,
            )
            await self._finish_guest_provider_failure(
                bot,
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
                demo_kind=demo_kind,
                prompt_message_id=prompt_message_id,
            )
            return
        if completion.outcome is GuestReservationOutcome.SUCCEEDED:
            await self._deliver_guest_result(
                bot,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
            )
            return
        if completion.outcome is GuestReservationOutcome.ACCESS_CHANGED:
            await self._edit_guest_access_changed(
                bot,
                chat_id=chat_id,
                message_id=prompt_message_id,
            )
            return
        await self._finish_guest_provider_failure(
            bot,
            reservation_token=reservation_token,
            user_id=user_id,
            chat_id=chat_id,
            access_version=access_version,
            session_version=session_version,
            demo_kind=demo_kind,
            prompt_message_id=prompt_message_id,
        )

    async def _finish_guest_provider_failure(
        self,
        bot: Any,
        *,
        reservation_token: str,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        demo_kind: GuestDemoKind,
        prompt_message_id: int,
    ) -> None:
        try:
            failed = await self.guest_session_service.fail_processing(
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
            )
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest provider failure cleanup failed",
                exc,
                user_id,
                demo_kind,
            )
            return
        if (
            failed.session is not None
            and failed.session.status is GuestSessionStatus.AWAITING_INPUT
        ):
            await self._edit_guest_retry(bot, failed.session)
        elif failed.session is not None and failed.session.status is GuestSessionStatus.CANCELLED:
            await self._edit_guest_access_changed(
                bot,
                chat_id=chat_id,
                message_id=prompt_message_id,
            )

    async def _shielded_guest_failure(
        self,
        *,
        reservation_token: str,
        user_id: int,
        chat_id: int,
        access_version: int,
        session_version: int,
        demo_kind: GuestDemoKind,
    ) -> None:
        cleanup = asyncio.create_task(
            self.guest_session_service.fail_processing(
                reservation_token=reservation_token,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session_version,
            )
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            try:
                await cleanup
            except Exception as exc:
                self._log_guest_worker_failure(
                    "Guest cancelled cleanup failed",
                    exc,
                    user_id,
                    demo_kind,
                )
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest cancelled cleanup failed",
                exc,
                user_id,
                demo_kind,
            )

    async def _deliver_guest_result(
        self,
        bot: Any,
        *,
        user_id: int,
        chat_id: int,
        access_version: int,
    ) -> bool:
        async with self._nova_ui_lock:
            return await self._deliver_guest_result_locked(
                bot,
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
            )

    async def _deliver_guest_result_locked(
        self,
        bot: Any,
        *,
        user_id: int,
        chat_id: int,
        access_version: int,
    ) -> bool:
        try:
            pending = await self.guest_session_service.pending_result(
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
            )
        except Exception as exc:
            self._log_access_failure("Guest result lookup failed", exc, user_id)
            return False
        session = pending.session
        if pending.outcome is GuestSessionOutcome.NOT_GUEST and session is not None:
            await self._edit_guest_access_changed(
                bot,
                chat_id=session.chat_id,
                message_id=session.prompt_message_id,
            )
            return False
        if (
            pending.outcome is not GuestSessionOutcome.RESULT_READY
            or session is None
            or session.prompt_message_id is None
            or session.result_payload is None
        ):
            return False
        try:
            result = self._validated_guest_result(session)
            quota = await self.guest_quota_service.snapshot(user_id)
            text = self._guest_result_text(result, session.demo_kind, quota.remaining_operations)
        except (ValidationError, ValueError) as exc:
            self._log_guest_worker_failure(
                "Guest stored result validation failed",
                exc,
                user_id,
                session.demo_kind,
            )
            await self.guest_session_service.cancel(user_id=user_id, chat_id=chat_id)
            return False
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest result rendering failed",
                exc,
                user_id,
                session.demo_kind,
            )
            return False
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=session.prompt_message_id,
                text=text,
                reply_markup=_guest_result_markup(quota.remaining_operations),
            )
        except BadRequest as exc:
            if not self._message_is_not_modified(exc):
                self._log_guest_worker_failure(
                    "Guest result edit failed",
                    exc,
                    user_id,
                    session.demo_kind,
                )
                return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest result edit failed",
                exc,
                user_id,
                session.demo_kind,
            )
            return False
        try:
            delivered = await self.guest_session_service.mark_delivered(
                user_id=user_id,
                chat_id=chat_id,
                access_version=access_version,
                session_version=session.version,
            )
        except Exception as exc:
            self._log_guest_worker_failure(
                "Guest result delivery acknowledgement failed",
                exc,
                user_id,
                session.demo_kind,
            )
            return False
        if delivered.outcome is GuestSessionOutcome.NOT_GUEST:
            await self._edit_guest_access_changed(
                bot,
                chat_id=chat_id,
                message_id=session.prompt_message_id,
            )
            return False
        if delivered.outcome is GuestSessionOutcome.EXPIRED or (
            delivered.outcome is GuestSessionOutcome.STALE
            and delivered.session is not None
            and delivered.session.status is GuestSessionStatus.EXPIRED
        ):
            await self._edit_guest_result_expired(
                bot,
                chat_id=chat_id,
                message_id=session.prompt_message_id,
            )
            return False
        return delivered.outcome is GuestSessionOutcome.COMPLETED

    async def _edit_guest_retry(
        self,
        bot: Any,
        session: GuestSessionSnapshot,
    ) -> bool:
        if session.status is not GuestSessionStatus.AWAITING_INPUT:
            return False
        return await self._edit_canonical_message(
            bot,
            chat_id=session.chat_id,
            message_id=session.prompt_message_id,
            text=GUEST_PROVIDER_ERROR_TEXT,
            markup=_DEMO_RETRY_MARKUP,
        )

    async def _edit_guest_access_changed(
        self,
        bot: Any,
        *,
        chat_id: int,
        message_id: int | None,
    ) -> bool:
        return await self._edit_canonical_message(
            bot,
            chat_id=chat_id,
            message_id=message_id,
            text=GUEST_ACCESS_CHANGED_TEXT,
            markup=None,
        )

    async def _edit_guest_result_expired(
        self,
        bot: Any,
        *,
        chat_id: int,
        message_id: int | None,
    ) -> bool:
        return await self._edit_canonical_message(
            bot,
            chat_id=chat_id,
            message_id=message_id,
            text=GUEST_RESULT_EXPIRED_TEXT,
            markup=_guest_temporary_markup(),
            message_not_modified_is_success=True,
        )

    async def _edit_guest_session_expired(
        self,
        bot: Any,
        *,
        chat_id: int,
        message_id: int | None,
    ) -> bool:
        return await self._edit_canonical_message(
            bot,
            chat_id=chat_id,
            message_id=message_id,
            text=GUEST_SESSION_EXPIRED_TEXT,
            markup=_guest_temporary_markup(),
            message_not_modified_is_success=True,
        )

    async def _recover_guest_demo_results(self, bot: Any) -> None:
        try:
            candidates = await self.guest_session_service.pending_result_candidates()
        except Exception as exc:
            self._log_access_failure("Guest result recovery lookup failed", exc, None)
            return
        for candidate in candidates:
            try:
                await self._deliver_guest_result(
                    bot,
                    user_id=candidate.user_id,
                    chat_id=candidate.chat_id,
                    access_version=candidate.access_version,
                )
            except Exception as exc:
                self._log_access_failure(
                    "Guest result recovery candidate failed",
                    exc,
                    candidate.user_id,
                )

    @staticmethod
    def _validated_guest_result(
        session: GuestSessionSnapshot,
    ) -> GuestThoughtBreakdown | GuestFirstStep:
        if session.demo_kind is GuestDemoKind.THOUGHT_BREAKDOWN:
            return GuestThoughtBreakdown.model_validate(session.result_payload)
        if session.demo_kind is GuestDemoKind.FIRST_STEP:
            return GuestFirstStep.model_validate(session.result_payload)
        raise ValueError("unsupported guest result kind")

    @staticmethod
    def _guest_result_text(
        result: GuestThoughtBreakdown | GuestFirstStep,
        demo_kind: GuestDemoKind,
        remaining_operations: int,
    ) -> str:
        if demo_kind is GuestDemoKind.THOUGHT_BREAKDOWN:
            if not isinstance(result, GuestThoughtBreakdown):
                raise ValueError("guest thought result schema mismatch")
            category_labels = {
                "idea": "Идея",
                "task": "Задача",
                "desire": "Желание",
                "note": "Заметка",
            }
            text = (
                "📝 Разобранная мысль\n\n"
                f"🏷️ Тип: {category_labels[result.category]}\n"
                f"✏️ Заголовок: {result.title}\n"
                f"💡 Суть: {result.essence}\n"
                f"➡️ Следующий шаг: {result.next_step}"
            )
        else:
            if not isinstance(result, GuestFirstStep):
                raise ValueError("guest first-step result schema mismatch")
            text = f"🌱 Первый шаг\n\n🎯 Фокус: {result.focus}\n👣 Первый шаг: {result.first_step}"
            if result.actions:
                actions = "\n".join(
                    f"{index}. {action}" for index, action in enumerate(result.actions, start=1)
                )
                text += f"\n\n📌 Дополнительные действия:\n{actions}"
        if remaining_operations > 0:
            text += f"\n\n🎁 Бесплатных разборов осталось: {remaining_operations}."
        else:
            text += (
                "\n\n🎁 Бесплатные разборы закончились.\n\n"
                "Чтобы получить подписку или заказать такого же собственного бота, "
                "напишите Назару Сергеевичу: @Nazar_38rus."
            )
        if len(text.encode("utf-16-le")) // 2 >= 4096:
            raise ValueError("guest result exceeds Telegram text limit")
        return text

    @staticmethod
    def _demo_input_text(demo_kind: GuestDemoKind) -> str:
        if demo_kind is GuestDemoKind.THOUGHT_BREAKDOWN:
            return GUEST_THOUGHT_INPUT_TEXT
        return GUEST_FIRST_STEP_INPUT_TEXT

    def _quota_screen(self, snapshot: Any) -> tuple[str, InlineKeyboardMarkup] | None:
        if not snapshot.enabled:
            return GUEST_DISABLED_TEXT, _guest_temporary_markup()
        if snapshot.successful_lifetime_count >= self.guest_quota_policy.operation_limit:
            return GUEST_LIFETIME_EXHAUSTED_TEXT, _access_markup()
        if snapshot.global_used >= snapshot.global_capacity:
            return GUEST_GLOBAL_EXHAUSTED_TEXT, _guest_temporary_markup()
        return None

    @staticmethod
    def _reservation_denial_screen(
        reason: GuestQuotaDenialReason,
    ) -> tuple[str, InlineKeyboardMarkup]:
        if reason is GuestQuotaDenialReason.LIFETIME_EXHAUSTED:
            return GUEST_LIFETIME_EXHAUSTED_TEXT, _access_markup()
        if reason is GuestQuotaDenialReason.DISABLED:
            return GUEST_DISABLED_TEXT, _guest_temporary_markup()
        if reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED:
            return GUEST_GLOBAL_EXHAUSTED_TEXT, _guest_temporary_markup()
        return GUEST_TEMPORARY_ERROR_TEXT, _guest_temporary_markup()

    async def _edit_canonical_message(
        self,
        bot: Any,
        *,
        chat_id: int,
        message_id: int | None,
        text: str,
        markup: InlineKeyboardMarkup | None,
        message_not_modified_is_success: bool = False,
    ) -> bool:
        if message_id is None or message_id <= 0:
            return False
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=markup,
            )
            return True
        except asyncio.CancelledError:
            raise
        except BadRequest as exc:
            if message_not_modified_is_success and self._message_is_not_modified(exc):
                return True
            self._log_access_failure("Guest canonical edit failed", exc, chat_id)
            return False
        except Exception as exc:
            self._log_access_failure("Guest canonical edit failed", exc, chat_id)
            return False

    @staticmethod
    def _message_is_not_modified(exc: BadRequest) -> bool:
        return "message is not modified" in str(exc).casefold()

    @staticmethod
    def _log_guest_worker_failure(
        event: str,
        exc: BaseException,
        user_id: int,
        demo_kind: GuestDemoKind,
    ) -> None:
        logger.error(
            "%s error_type=%s user_id=%s demo_kind=%s",
            event,
            type(exc).__name__,
            user_id,
            demo_kind.value,
        )

    @staticmethod
    def _guest_screen(route: str, telegram_id: int) -> tuple[str, InlineKeyboardMarkup]:
        if route == "guest:root":
            return GUEST_ROOT_TEXT, _ROOT_MARKUP
        if route == "guest:demos":
            return GUEST_DEMOS_TEXT, _DEMOS_MARKUP
        if route == "guest:features":
            return "🧭 Что я умею\n\nВыберите раздел:", _FEATURES_MARKUP
        if route in GUEST_FEATURE_TEXTS:
            return GUEST_FEATURE_TEXTS[route], _FEATURE_MARKUP
        if route == "guest:how":
            return GUEST_HOW_TEXT, _HOW_MARKUP
        return (
            "Полный доступ открывает сохранение результатов, персональный профиль, задачи, "
            "напоминания и ежедневное сопровождение.\n\n"
            "Чтобы получить подписку или заказать собственного бота, напишите @Nazar_38rus.\n\n"
            f"Ваш Telegram ID: {telegram_id}\n"
            "Отправьте его вместе с сообщением: „Нужна подписка“ или „Хочу собственного бота“.",
            _access_markup(),
        )

    async def show_guest_root(self, update: Update) -> None:
        await self._reply_guest_screen(update, GUEST_ROOT_TEXT, _ROOT_MARKUP)

    async def show_blocked_screen(self, update: Update) -> None:
        telegram_user = update.effective_user
        telegram_id = telegram_user.id if telegram_user is not None else "неизвестен"
        text = (
            "Доступ к боту ограничен.\n\n"
            "Если вы считаете, что произошла ошибка, напишите @Nazar_38rus и укажите "
            f"ваш Telegram ID: {telegram_id}."
        )
        query = update.callback_query
        if query is not None:
            await query.answer("Доступ ограничен.", show_alert=True)
            await self._edit_guest_screen(query, text, _blocked_markup())
            return
        await self._reply_guest_screen(update, text, _blocked_markup())

    @staticmethod
    async def _reply_guest_screen(update: Update, text: str, markup: InlineKeyboardMarkup) -> None:
        message = update.effective_message
        if message is not None:
            await message.reply_text(text, reply_markup=markup)

    async def _edit_guest_screen(
        self,
        query: Any,
        text: str,
        markup: InlineKeyboardMarkup,
    ) -> bool:
        try:
            await query.edit_message_text(text, reply_markup=markup)
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_access_failure("Guest screen edit failed", exc, None)
            return False

    async def _access_fail_closed(self, update: Update) -> None:
        try:
            if update.callback_query is not None:
                await update.callback_query.answer(SERVICE_UNAVAILABLE_TEXT, show_alert=True)
            elif update.effective_message is not None:
                await update.effective_message.reply_text(SERVICE_UNAVAILABLE_TEXT)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            telegram_user = update.effective_user
            user_id = telegram_user.id if telegram_user is not None else None
            self._log_access_failure("Fail-closed notification failed", exc, user_id)

    @staticmethod
    def _has_guest_media(message: Any) -> bool:
        return any(
            bool(getattr(message, name, None))
            for name in (
                "voice",
                "audio",
                "photo",
                "document",
                "video",
                "video_note",
                "animation",
                "sticker",
                "contact",
                "location",
                "venue",
                "poll",
            )
        )

    @staticmethod
    def _log_access_failure(event: str, exc: BaseException, user_id: int | None) -> None:
        if user_id is None:
            logger.error("%s error_type=%s", event, type(exc).__name__)
        else:
            logger.error("%s error_type=%s user_id=%s", event, type(exc).__name__, user_id)
