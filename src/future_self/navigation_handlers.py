from __future__ import annotations

from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ApplicationHandlerStop, ContextTypes, ConversationHandler

from .navigation import help_topics, navigation_actions, navigation_sections

FLOW_LABELS = {
    "onboarding": "настройка профиля",
    "evening": "вечерняя рефлексия",
    "health": "health check-in",
    "doctor": "подготовка к приёму",
    "vision": "создание карточки желания",
    "vision_image": "добавление личного фото",
    "labs": "загрузка результатов анализов",
    "workspace": "операция с совместным пространством",
    "knowledge_capture": "добавление материала в базу знаний",
    "rename_goal": "переименование цели",
}


class NavigationHandlers:
    navigation_flow_sessions: Any

    async def navigation_public_command_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        flow = await self._active_navigation_flow(update, context)
        if flow is None:
            if hasattr(self, "collection_service"):
                user = await self._user(update.effective_user.id)
                await self.collection_service.clear_context(user.id, update.effective_chat.id)
                await self.collection_service.cancel_input(user.id, update.effective_chat.id)
                if self._workspace_enabled():
                    await self.workspace_service.cancel_input(user.id, update.effective_chat.id)
            return
        await self._prompt_navigation_flow(update.effective_message, update, flow)
        raise ApplicationHandlerStop

    async def navigation_text_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        command = self.natural_command_router.route(update.effective_message.text or "")
        if command is None or command.action not in {"menu", "help"}:
            return
        if command.action == "menu":
            await self.menu_command(update, context)
        else:
            await self.help_command(update, context)
        raise ApplicationHandlerStop

    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await self._prompt_navigation_flow(update.effective_message, update, flow)
            return
        if hasattr(self, "collection_service"):
            user = await self._user(update.effective_user.id)
            await self.collection_service.clear_context(user.id, update.effective_chat.id)
            await self.collection_service.cancel_input(user.id, update.effective_chat.id)
        await self._send_navigation_root(update.effective_message)

    async def doctor_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await self._prompt_navigation_flow(update.effective_message, update, flow)
            return
        await self._send_navigation_section(update.effective_message, "doctor")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        await update.effective_message.reply_text(
            "Помощь\n\nВыбери короткий раздел — без длинной стены текста.",
            reply_markup=self._help_keyboard(),
        )

    async def navigation_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        query = update.callback_query
        data = query.data or ""
        if data.startswith("nav:flow:"):
            return await self._navigation_flow_action(update, context)

        flow = await self._active_navigation_flow(update, context)
        if flow is not None and not data.startswith("nav:help:"):
            await query.answer()
            await self._prompt_navigation_flow(query.message, update, flow)
            return None

        if hasattr(self, "collection_service"):
            user = await self._user(update.effective_user.id)
            await self.collection_service.clear_context(user.id, update.effective_chat.id)
            await self.collection_service.cancel_input(user.id, update.effective_chat.id)

        if data == "nav:root":
            await query.answer()
            await self._edit_or_send(
                query,
                "Главное меню\n\nВыбери раздел — команды помнить не обязательно.",
                self._root_keyboard(),
            )
            return None
        if data == "nav:help":
            await query.answer()
            await self._edit_or_send(
                query,
                "Помощь\n\nВыбери короткий раздел — без длинной стены текста.",
                self._help_keyboard(),
            )
            return None
        sections = navigation_sections(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )
        actions = navigation_actions(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )
        topics = help_topics(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )
        if data.startswith("nav:section:"):
            section_key = data.removeprefix("nav:section:")
            if section_key not in sections:
                await self._navigation_stale(query)
                return None
            await query.answer()
            section = sections[section_key]
            await self._edit_or_send(
                query,
                f"{section.emoji} {section.label}\n\n{section.description}",
                self._section_keyboard(section_key),
            )
            return None
        if data.startswith("nav:help:"):
            topic_key = data.removeprefix("nav:help:")
            topic = topics.get(topic_key)
            if topic is None:
                await self._navigation_stale(query)
                return None
            await query.answer()
            await self._edit_or_send(
                query,
                f"{topic[0]}\n\n{topic[1]}",
                self._back_keyboard("nav:help"),
            )
            return None
        if data.startswith("nav:action:"):
            action_key = data.removeprefix("nav:action:")
            action = actions.get(action_key)
            if action is None or action.handler in {
                "health_checkin_start",
                "doctor_prepare_start",
                "start",
            }:
                await self._navigation_stale(query)
                return None
            await query.answer()
            if action.handler is None:
                text = action.description
                if action.example:
                    text += f"\n\nПример: {action.example}"
                await query.message.reply_text(
                    text,
                    reply_markup=self._back_keyboard(self._section_for_action(action_key)),
                )
                return None
            original_args = getattr(context, "args", None)
            context.args = []
            try:
                await getattr(self, action.handler)(update, context)
            finally:
                context.args = original_args or []
            if action.handler.startswith("task_") or action.handler in {
                "collections_command",
                "spaces_command",
                "knowledge_command",
                "capture_command",
            }:
                return None
            await query.message.reply_text(
                "Навигация",
                reply_markup=self._back_keyboard(self._section_for_action(action_key)),
            )
            return None
        await self._navigation_stale(query)
        return None

    async def navigation_health_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await update.callback_query.answer()
            await self._prompt_navigation_flow(update.callback_query.message, update, flow)
            return None
        await update.callback_query.answer()
        return await self.health_checkin_start(update, context)

    async def navigation_doctor_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await update.callback_query.answer()
            await self._prompt_navigation_flow(update.callback_query.message, update, flow)
            return None
        await update.callback_query.answer()
        return await self.doctor_prepare_start(update, context)

    async def navigation_onboarding_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await update.callback_query.answer()
            await self._prompt_navigation_flow(update.callback_query.message, update, flow)
            return None
        await update.callback_query.answer()
        return await self.start(update, context)

    async def _navigation_flow_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        query = update.callback_query
        parts = (query.data or "").split(":")
        if len(parts) != 4 or parts[2] not in {"continue", "exit"}:
            await self._navigation_stale(query)
            return None
        capability = await self.navigation_flow_sessions.claim(
            parts[3], update.effective_user.id, update.effective_chat.id
        )
        if capability is None:
            await self._navigation_stale(query)
            return None
        current = await self._active_navigation_flow(update, context)
        if current != capability.flow:
            await self._navigation_stale(query)
            return None
        await query.answer()
        if parts[2] == "continue":
            instruction = (
                "Отправь выбранное фото или нажми «Отмена» в сообщении загрузки."
                if current == "vision_image"
                else (
                    "Отправь фото/PDF или используй кнопки preview."
                    if current == "labs"
                    else (
                        "Пришли материал или используй кнопки Capture preview."
                        if current == "knowledge_capture"
                        else (
                            "Пришли запрошенный текст или используй /cancel."
                            if current == "workspace"
                            else "Ответь на текущий вопрос."
                        )
                    )
                )
            )
            await self._edit_or_send(
                query,
                f"Продолжаем: {FLOW_LABELS[current]}. {instruction}",
                None,
            )
            if current == "vision":
                user = await self._user(update.effective_user.id)
                draft = await self.vision_service.draft(user.id, update.effective_chat.id)
                if draft is not None:
                    await self._vision_prompt(query.message, draft)
            return None

        await self._clear_navigation_flow(update, context, current)
        await self._edit_or_send(
            query,
            f"Сценарий «{FLOW_LABELS[current]}» остановлен. Остальные данные не изменены.",
            None,
        )
        await self._send_navigation_root(query.message)
        return ConversationHandler.END

    async def _active_navigation_flow(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> str | None:
        for key, name in (
            ("health_checkin", "health"),
            ("doctor_prepare", "doctor"),
            ("evening", "evening"),
            ("rename_goal_id", "rename_goal"),
        ):
            if key in context.user_data:
                return name
        user = await self._user(update.effective_user.id)
        if (
            self._workspace_enabled()
            and await self.workspace_service.pending_input(user.id, update.effective_chat.id)
            is not None
        ):
            return "workspace"
        if await self.lab_uploads.has_active(user.id, update.effective_chat.id):
            return "labs"
        edit = context.user_data.get("lab_document_edit")
        if (
            edit is not None
            and edit.get("owner_id") == user.id
            and edit.get("chat_id") == update.effective_chat.id
        ):
            return "labs"
        if "onboarding_user_id" in context.user_data and not user.onboarding_completed:
            return "onboarding"
        if await self.vision_image_sessions.has_active(user.id, update.effective_chat.id):
            return "vision_image"
        if await self.vision_service.draft(user.id, update.effective_chat.id) is not None:
            return "vision"
        if self._knowledge_capture_enabled():
            capture = await self.knowledge_service.capture_state(user.id, update.effective_chat.id)
            if capture.preview is not None:
                return "knowledge_capture"
        return None

    async def _clear_navigation_flow(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        flow: str,
    ) -> None:
        if flow == "health":
            context.user_data.pop("health_checkin", None)
        elif flow == "doctor":
            context.user_data.pop("doctor_prepare", None)
        elif flow == "evening":
            context.user_data.pop("evening", None)
        elif flow == "rename_goal":
            context.user_data.pop("rename_goal_id", None)
        elif flow == "onboarding":
            context.user_data.pop("onboarding_user_id", None)
            context.user_data.pop("vision_summary", None)
        else:
            user = await self._user(update.effective_user.id)
            if flow == "workspace":
                await self.workspace_service.cancel_input(user.id, update.effective_chat.id)
            elif flow == "labs":
                await self.lab_uploads.cancel_active(user.id, update.effective_chat.id)
                context.user_data.pop("lab_document_edit", None)
            elif flow == "vision_image":
                await self.vision_image_sessions.cancel_active(user.id, update.effective_chat.id)
            elif flow == "vision":
                await self.vision_service.cancel(user.id, update.effective_chat.id)
            elif flow == "knowledge_capture":
                state = await self.knowledge_service.capture_state(
                    user.id, update.effective_chat.id
                )
                await self.knowledge_service.cancel_pending_input(user.id, update.effective_chat.id)
                if state.preview is not None:
                    await self.knowledge_service.cancel_capture(
                        user.id,
                        update.effective_chat.id,
                        state.preview.draft_public_id,
                        state.preview.version,
                    )

    async def _prompt_navigation_flow(self, message: Any, update: Update, flow: str) -> None:
        token = await self.navigation_flow_sessions.issue(
            update.effective_user.id, update.effective_chat.id, flow
        )
        await message.reply_text(
            f"Сейчас не завершён сценарий: {FLOW_LABELS[flow]}. Что сделать?",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Продолжить", callback_data=f"nav:flow:continue:{token}"
                        ),
                        InlineKeyboardButton(
                            "Выйти в меню", callback_data=f"nav:flow:exit:{token}"
                        ),
                    ]
                ]
            ),
        )

    @staticmethod
    async def _navigation_stale(query: Any) -> None:
        await query.answer("Эта кнопка устарела или недоступна.", show_alert=True)

    @staticmethod
    async def _edit_or_send(
        query: Any, text: str, reply_markup: InlineKeyboardMarkup | None
    ) -> None:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
        except (TelegramError, TypeError):
            await query.message.reply_text(text, reply_markup=reply_markup)

    async def _send_navigation_root(self, message: Any) -> None:
        await message.reply_text(
            "Главное меню\n\nВыбери раздел — команды помнить не обязательно.",
            reply_markup=self._root_keyboard(),
        )

    async def _send_navigation_section(self, message: Any, section_key: str) -> None:
        section = navigation_sections(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )[section_key]
        await message.reply_text(
            f"{section.emoji} {section.label}\n\n{section.description}",
            reply_markup=self._section_keyboard(section_key),
        )

    def _root_keyboard(self) -> InlineKeyboardMarkup:
        sections = navigation_sections(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )
        rows = [
            [
                InlineKeyboardButton(
                    f"{section.emoji} {section.label}",
                    callback_data=f"nav:section:{section.key}",
                )
            ]
            for section in sections.values()
        ]
        rows.append([InlineKeyboardButton("❓ Помощь", callback_data="nav:help")])
        return InlineKeyboardMarkup(rows)

    def _section_keyboard(self, section_key: str) -> InlineKeyboardMarkup:
        sections = navigation_sections(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )
        actions = navigation_actions(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )
        section = sections[section_key]
        rows = [
            [
                InlineKeyboardButton(
                    actions[action].label,
                    callback_data=f"nav:action:{action}",
                )
            ]
            for action in section.actions
        ]
        rows.extend(
            [
                [InlineKeyboardButton("← Назад", callback_data="nav:root")],
                [
                    InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root"),
                    InlineKeyboardButton("❓ Помощь", callback_data="nav:help"),
                ],
            ]
        )
        return InlineKeyboardMarkup(rows)

    def _help_keyboard(self) -> InlineKeyboardMarkup:
        topics = help_topics(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        )
        labels = {
            "quick": "Быстрый старт",
            "features": "Что умеет бот",
            "examples": "Примеры сообщений",
            "commands": "Команды",
            "privacy": "Конфиденциальность",
            "safety": "Здоровье и безопасность",
            "knowledge": "📚 База знаний",
        }
        rows = [
            [InlineKeyboardButton(labels[key], callback_data=f"nav:help:{key}")] for key in topics
        ]
        rows.append([InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")])
        return InlineKeyboardMarkup(rows)

    @staticmethod
    def _back_keyboard(target: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("← Назад", callback_data=target)],
                [
                    InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root"),
                    InlineKeyboardButton("❓ Помощь", callback_data="nav:help"),
                ],
            ]
        )

    def _section_for_action(self, action_key: str) -> str:
        for section in navigation_sections(
            self._workspace_enabled(),
            self._knowledge_hub_enabled(),
            self._knowledge_capture_enabled(),
        ).values():
            if action_key in section.actions:
                return f"nav:section:{section.key}"
        return "nav:root"

    def _workspace_enabled(self) -> bool:
        return bool(getattr(self.settings, "enable_workspace_access", False))

    def _knowledge_hub_enabled(self) -> bool:
        return bool(getattr(self.settings, "enable_knowledge_hub", False))

    def _knowledge_capture_enabled(self) -> bool:
        return bool(getattr(self.settings, "enable_knowledge_capture", False))
