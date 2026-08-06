from __future__ import annotations

import logging
from typing import Any

from telegram import (
    BotCommand,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import ApplicationHandlerStop, ContextTypes

from .access import BLOCKED, GUEST, AccessTier, is_full_access_tier, require_access_tier
from .navigation import CommandSpec, public_commands

logger = logging.getLogger(__name__)

GUEST_COMMANDS = (
    CommandSpec("start", "Начать"),
    CommandSpec("menu", "Главное меню"),
    CommandSpec("help", "Как это работает"),
)
BLOCKED_COMMANDS = (CommandSpec("start", "Начать"),)

GUEST_ROOT_TEXT = """👋 Я — „Моя будущая версия“

Личный AI-ассистент, который помогает разгружать голову, видеть главное и превращать желаемое будущее в конкретные действия.

Я умею работать с мыслями, задачами, целями, картой желаний, самочувствием и подготовкой к важным событиям.

В гостевом режиме можно бесплатно попробовать 2 AI-действия."""

GUEST_DEMOS_TEXT = """✨ Бесплатная демонстрация

Можно попробовать два сценария:

📝 Разобрать мысль — превратить свободный текст в понятную структуру и следующий шаг.

🌱 Найти первый шаг — превратить цель или желаемое изменение в небольшое действие."""

GUEST_DEMO_PENDING_TEXT = (
    "Демонстрационные AI-операции подключаются следующим этапом. "
    "Пока можно посмотреть возможности бота и принцип работы."
)

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
        [InlineKeyboardButton("💬 Получить доступ", callback_data="guest:access")],
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

_DEMO_PENDING_MARKUP = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("🧭 Что я умею", callback_data="guest:features")],
        [InlineKeyboardButton("← Назад", callback_data="guest:demos")],
    ]
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
        except Exception as exc:
            self._log_access_failure("Access gate database failure", exc, telegram_user.id)
            await self._access_fail_closed(update)
            raise ApplicationHandlerStop from None

        await self._sync_access_commands(
            context, chat.id, user.telegram_id, tier, user.access_version
        )
        if is_full_access_tier(tier):
            return
        try:
            if tier == GUEST:
                await self._handle_guest_update(update)
            else:
                await self.show_blocked_screen(update)
        except TelegramError as exc:
            self._log_access_failure("Access screen delivery failure", exc, telegram_user.id)
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
        except TelegramError as exc:
            self._log_access_failure("Access command scope sync failed", exc, telegram_id)
            return
        self._access_scope_cache[chat_id] = cache_key

    async def _handle_guest_update(self, update: Update) -> None:
        query = update.callback_query
        if query is not None:
            await self._handle_guest_callback(update)
            return
        message = update.effective_message
        if message is None:
            return
        text = getattr(message, "text", None)
        if isinstance(text, str) and text.startswith("/"):
            command = text.split(maxsplit=1)[0].partition("@")[0].casefold()
            if command in {"/start", "/menu"}:
                await self.show_guest_root(update)
            elif command == "/help":
                await self._reply_guest_screen(update, GUEST_HOW_TEXT, _HOW_MARKUP)
            else:
                await self._reply_guest_screen(
                    update,
                    f"{GUEST_COMMAND_NOTICE}\n\n{GUEST_ROOT_TEXT}",
                    _ROOT_MARKUP,
                )
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

    async def _handle_guest_callback(self, update: Update) -> None:
        query = update.callback_query
        data = query.data if isinstance(query.data, str) else ""
        if data not in GUEST_CALLBACK_ROUTES:
            if data.startswith("guest:"):
                await query.answer(STALE_GUEST_ALERT, show_alert=True)
                return
            await query.answer(FULL_VERSION_ALERT, show_alert=True)
            await self._edit_guest_screen(query, GUEST_ROOT_TEXT, _ROOT_MARKUP)
            return
        await query.answer()
        text, markup = self._guest_screen(data, update.effective_user.id)
        await self._edit_guest_screen(query, text, markup)

    @staticmethod
    def _guest_screen(route: str, telegram_id: int) -> tuple[str, InlineKeyboardMarkup]:
        if route == "guest:root":
            return GUEST_ROOT_TEXT, _ROOT_MARKUP
        if route == "guest:demos":
            return GUEST_DEMOS_TEXT, _DEMOS_MARKUP
        if route in {"guest:demo:thought", "guest:demo:first-step"}:
            return GUEST_DEMO_PENDING_TEXT, _DEMO_PENDING_MARKUP
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

    async def _edit_guest_screen(self, query: Any, text: str, markup: InlineKeyboardMarkup) -> None:
        try:
            await query.edit_message_text(text, reply_markup=markup)
        except TelegramError as exc:
            self._log_access_failure("Guest screen edit failed", exc, None)
            message = getattr(query, "message", None)
            if message is not None:
                try:
                    await message.reply_text(text, reply_markup=markup)
                except TelegramError as fallback_exc:
                    self._log_access_failure("Guest screen fallback failed", fallback_exc, None)

    @staticmethod
    async def _access_fail_closed(update: Update) -> None:
        try:
            if update.callback_query is not None:
                await update.callback_query.answer(SERVICE_UNAVAILABLE_TEXT, show_alert=True)
            elif update.effective_message is not None:
                await update.effective_message.reply_text(SERVICE_UNAVAILABLE_TEXT)
        except TelegramError:
            return

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
