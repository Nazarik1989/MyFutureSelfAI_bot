from __future__ import annotations

import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ApplicationHandlerStop, ContextTypes, ConversationHandler

from .navigation import (
    HELP_TOPIC_LABELS,
    LEGACY_SECTION_ALIASES,
    ROOT_HELP_TOPIC_KEYS,
    SECTION_HELP_TOPICS,
    help_topics,
    navigation_actions,
    navigation_sections,
)

logger = logging.getLogger(__name__)

_LEGACY_HELP_ALIASES = {
    "day": "today_section",
    "features": "requests",
    "voice": "records_section",
    "drafts": "records_section",
    "tasks": "tasks_section",
    "collections": "sections_section",
    "vision": "requests",
    "health": "health_section",
    "doctor": "health_section",
    "registration": "settings_section",
    "commands": "requests",
    "safety": "privacy",
}


class _ScreenUpdate:
    def __init__(self, source: Any, message: Any):
        self._source = source
        self.effective_message = message
        self.message = message

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class _CallbackScreenMessage:
    def __init__(
        self,
        owner: NavigationHandlers,
        query: Any,
        fallback_markup: InlineKeyboardMarkup,
    ):
        self._owner = owner
        self._query = query
        self._message = query.message
        self._fallback_markup = fallback_markup

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)

    async def reply_text(self, text: str, **kwargs: Any) -> Any:
        reply_markup = kwargs.pop("reply_markup", self._fallback_markup)
        await self._owner._edit_or_send(
            self._query,
            text,
            reply_markup,
            **kwargs,
        )
        return self._message


class _EditedScreenMessage:
    def __init__(self, message: Any):
        self._message = message

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)

    async def reply_text(self, text: str, **kwargs: Any) -> Any:
        try:
            await self._message.edit_text(text, **kwargs)
        except TelegramError as exc:
            if not NavigationHandlers._message_not_modified(exc):
                logger.warning(
                    "Navigation message edit failed error_type=%s",
                    type(exc).__name__,
                )
        return self._message


FLOW_LABELS = {
    "onboarding": "настройка профиля",
    "evening": "вечерняя рефлексия",
    "health": "health check-in",
    "doctor": "подготовка к приёму",
    "vision": "создание карточки желания",
    "vision_image": "работа с изображением или личным референсом",
    "labs": "загрузка результатов анализов",
    "workspace": "операция с совместным пространством",
    "knowledge_capture": "добавление материала в базу знаний",
    "task_edit": "изменение задачи",
    "collection_input": "операция с разделом",
    "draft_edit": "редактирование черновика",
    "date_choice": "выбор даты",
    "draft_action": "подтверждение действия с черновиком",
    "system_action": "подтверждение системного действия",
    "rename_goal": "переименование цели",
}


class NavigationHandlers:
    navigation_flow_sessions: Any

    @staticmethod
    def _edited_screen_update(update: Any, message: Any) -> _ScreenUpdate:
        return _ScreenUpdate(update, _EditedScreenMessage(message))

    async def navigation_public_command_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        command = (update.effective_message.text or "").split(maxsplit=1)[0]
        command = command.split("@", maxsplit=1)[0].casefold()
        flow = await self._active_navigation_flow(update, context)
        if flow is None:
            if command != "/help":
                await self.nova_clear_current(update)
            if hasattr(self, "collection_service"):
                user = await self._user(update.effective_user.id)
                await self.collection_service.clear_context(user.id, update.effective_chat.id)
                await self.collection_service.cancel_input(user.id, update.effective_chat.id)
                if self._workspace_enabled():
                    await self.workspace_service.cancel_input(user.id, update.effective_chat.id)
            return
        if command == "/help":
            await self._nova_flow_help(update.effective_message, update, flow)
            raise ApplicationHandlerStop
        await self._prompt_navigation_flow(update.effective_message, update, flow)
        raise ApplicationHandlerStop

    async def navigation_text_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        onboarding_result = await self.onboarding_persistent_input(
            update, context, update.effective_message.text or ""
        )
        if onboarding_result is not None:
            raise ApplicationHandlerStop
        text = update.effective_message.text or ""
        if await self.nova_text_gate(update, context):
            raise ApplicationHandlerStop
        command = self.natural_command_router.route(text)
        explicit_unknown = (
            command is None and self.natural_command_router.is_explicit_navigation_request(text)
        )
        if explicit_unknown and self.collection_command_router.route(text) is not None:
            return
        if command is None and not explicit_unknown:
            return
        action = command.action if command is not None else "help"
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            if action == "help":
                return
            await self._prompt_navigation_flow(update.effective_message, update, flow)
        else:
            await self._handle_natural_command(update, context, action)
        raise ApplicationHandlerStop

    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await self._prompt_navigation_flow(update.effective_message, update, flow)
            return
        await self.nova_clear_current(update)
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
        await self._send_navigation_section(update.effective_message, "health")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        await update.effective_message.reply_text(
            "❓ Помощь\n\nВыбери направление или задай короткий вопрос о навигации.",
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
        help_navigation = data == "nav:help" or data.startswith("nav:help:")
        if flow is not None and not help_navigation:
            await query.answer()
            await self._prompt_navigation_flow(query.message, update, flow, query=query)
            return None
        if flow is not None and help_navigation:
            await self.nova_navigation_help_callback(update, context)
            return None

        user = await self._user(update.effective_user.id)
        if hasattr(self, "collection_service"):
            await self.collection_service.clear_context(user.id, update.effective_chat.id)
            await self.collection_service.cancel_input(user.id, update.effective_chat.id)

        if data == "nav:root":
            await query.answer()
            await self.nova_clear_current(update)
            await self._edit_or_send(
                query,
                "Главное меню\n\nЧто хочешь сделать?",
                self._root_keyboard(),
            )
            return None
        if data == "nav:help":
            await self.nova_navigation_help_callback(update, context)
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
            self._voice_enabled(),
            self._task_reminders_enabled(),
        )
        if data.startswith("nav:section:"):
            section_key = data.removeprefix("nav:section:")
            if section_key == "vision":
                await query.answer()
                await self._vision_menu(query.message, user=user, query=query)
                return None
            section_key = LEGACY_SECTION_ALIASES.get(section_key, section_key)
            if section_key not in sections:
                await self._navigation_stale(query)
                return None
            await query.answer()
            await self.nova_clear_current(update)
            section = sections[section_key]
            await self._edit_or_send(
                query,
                f"{section.emoji} {section.label}\n\n{section.description}",
                self._section_keyboard(section_key),
            )
            return None
        if data.startswith("nav:help:"):
            topic_key = data.removeprefix("nav:help:")
            topic_key = _LEGACY_HELP_ALIASES.get(topic_key, topic_key)
            await self.nova_navigation_help_callback(
                update,
                context,
                topic_key=topic_key,
            )
            return None
        if data.startswith("nav:action:"):
            action_key = data.removeprefix("nav:action:")
            action = actions.get(action_key)
            if action is None or action.handler in {
                "evening_start",
                "health_checkin_start",
                "doctor_prepare_start",
                "start",
            }:
                await self._navigation_stale(query)
                return None
            await self.nova_clear_current(update)
            if action_key == "vision":
                await query.answer()
                await self._vision_menu(query.message, user=user, query=query)
                return None
            if action_key == "task_reminder_guide":
                await query.answer()
                topic = topics["tasks_section"]
                await self._edit_or_send(
                    query,
                    f"{topic[0]}\n\n{topic[1]}",
                    self._back_keyboard("nav:section:tasks"),
                )
                return None
            if action_key == "doctor_task_guide":
                await query.answer()
                topic = topics["health_section"]
                await self._edit_or_send(
                    query,
                    f"{topic[0]}\n\n{topic[1]}",
                    self._back_keyboard("nav:section:health"),
                )
                return None
            await query.answer()
            screen = _CallbackScreenMessage(
                self,
                query,
                self._back_keyboard(self._section_for_action(action_key)),
            )
            if action.handler is None:
                text = action.description
                if action.example:
                    text += f"\n\nПример: {action.example}"
                await screen.reply_text(text)
                return None
            original_args = getattr(context, "args", None)
            context.args = []
            try:
                await getattr(self, action.handler)(_ScreenUpdate(update, screen), context)
            finally:
                context.args = original_args or []
            return None
        await self._navigation_stale(query)
        return None

    async def navigation_health_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await update.callback_query.answer()
            await self._prompt_navigation_flow(
                update.callback_query.message, update, flow, query=update.callback_query
            )
            return None
        await self.nova_clear_current(update)
        await update.callback_query.answer()
        screen = _CallbackScreenMessage(
            self, update.callback_query, self._back_keyboard("nav:section:health")
        )
        return await self.health_checkin_start(_ScreenUpdate(update, screen), context)

    async def navigation_evening_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await update.callback_query.answer()
            await self._prompt_navigation_flow(
                update.callback_query.message, update, flow, query=update.callback_query
            )
            return None
        await self.nova_clear_current(update)
        await update.callback_query.answer()
        screen = _CallbackScreenMessage(
            self, update.callback_query, self._back_keyboard("nav:section:today")
        )
        return await self.evening_start(_ScreenUpdate(update, screen), context)

    async def navigation_doctor_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await update.callback_query.answer()
            await self._prompt_navigation_flow(
                update.callback_query.message, update, flow, query=update.callback_query
            )
            return None
        await self.nova_clear_current(update)
        await update.callback_query.answer()
        screen = _CallbackScreenMessage(
            self, update.callback_query, self._back_keyboard("nav:section:health")
        )
        return await self.doctor_prepare_start(_ScreenUpdate(update, screen), context)

    async def navigation_onboarding_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int | None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await update.callback_query.answer()
            await self._prompt_navigation_flow(
                update.callback_query.message, update, flow, query=update.callback_query
            )
            return None
        await self.nova_clear_current(update)
        await update.callback_query.answer()
        screen = _CallbackScreenMessage(
            self, update.callback_query, self._back_keyboard("nav:section:settings")
        )
        return await self.start(_ScreenUpdate(update, screen), context)

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
                    await self._vision_prompt(query.message, draft, query=query)
            return None

        await self._clear_navigation_flow(update, context, current)
        await self._edit_or_send(
            query,
            "Главное меню\n\nЧто хочешь сделать?",
            self._root_keyboard(),
        )
        return ConversationHandler.END

    async def _active_navigation_flow(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> str | None:
        user = await self._user(update.effective_user.id)
        if not user.onboarding_completed:
            from .repositories import OnboardingRepository

            async with self.db.sessions() as session:
                state = await OnboardingRepository(session).get(user.id)
            if state is not None and state.status in {
                "in_progress",
                "awaiting_confirmation",
            }:
                if context.user_data.get("onboarding_user_id") != user.id:
                    context.user_data["onboarding_user_id"] = user.id
                    context.user_data["onboarding_detached"] = True
                return "onboarding"
        for key, name in (
            ("health_checkin", "health"),
            ("doctor_prepare", "doctor"),
            ("evening", "evening"),
            ("rename_goal_id", "rename_goal"),
        ):
            if key in context.user_data:
                return name
        conversation = await self.conversation.get(
            update.effective_user.id,
            update.effective_chat.id,
        )
        if conversation.system_pending_action:
            return "system_action"
        if conversation.pending_date_options:
            return "date_choice"
        if conversation.pending_action or conversation.focused_draft_id:
            return "draft_action"
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
        if await self.vision_image_sessions.has_active(
            user.id, update.effective_chat.id
        ) or await self.vision_reference_sessions.has_active(user.id, update.effective_chat.id):
            return "vision_image"
        if await self.vision_service.draft(user.id, update.effective_chat.id) is not None:
            return "vision"
        if (
            await self.draft_service.editing(
                update.effective_user.id,
                update.effective_chat.id,
            )
            is not None
        ):
            return "draft_edit"
        if await self.task_service.pending_input(user.id, update.effective_chat.id) is not None:
            return "task_edit"
        if (
            await self.collection_service.pending_input(user.id, update.effective_chat.id)
            is not None
        ):
            return "collection_input"
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
            from .repositories import OnboardingRepository

            user = await self._user(update.effective_user.id)
            async with self.db.session() as session:
                state = await OnboardingRepository(session).get(user.id)
                if state is not None and state.status in {
                    "in_progress",
                    "awaiting_confirmation",
                }:
                    state.status = "cancelled"
            context.user_data.pop("onboarding_user_id", None)
            context.user_data.pop("onboarding_detached", None)
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
                await self.vision_reference_sessions.cancel_active(
                    user.id, update.effective_chat.id
                )
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
            elif flow == "task_edit":
                await self.task_service.cancel_pending_input(user.id, update.effective_chat.id)
            elif flow == "collection_input":
                await self.collection_service.cancel_input(user.id, update.effective_chat.id)
            elif flow == "draft_edit":
                await self.conversation.clear_focus(
                    update.effective_user.id,
                    update.effective_chat.id,
                )
                await self.conversation.clear_system_action(
                    update.effective_user.id,
                    update.effective_chat.id,
                )
                discarded = await self.draft_service.cancel_editing(
                    update.effective_user.id,
                    update.effective_chat.id,
                )
                if discarded:
                    await self.conversation.set_active_draft(
                        update.effective_user.id,
                        update.effective_chat.id,
                        None,
                    )
            elif flow == "date_choice":
                await self.conversation.set_date_conflict(
                    update.effective_user.id,
                    update.effective_chat.id,
                    [],
                )
            elif flow == "draft_action":
                await self.conversation.clear_focus(
                    update.effective_user.id,
                    update.effective_chat.id,
                )
            elif flow == "system_action":
                snapshot = await self.conversation.get(
                    update.effective_user.id,
                    update.effective_chat.id,
                )
                if snapshot.system_action_version is not None:
                    await self.conversation.clear_system_action(
                        update.effective_user.id,
                        update.effective_chat.id,
                        expected_version=snapshot.system_action_version,
                    )

    @staticmethod
    def _nova_flow_label(flow: str) -> str:
        return FLOW_LABELS.get(flow, "текущий шаг")

    async def _prompt_navigation_flow(
        self,
        message: Any,
        update: Update,
        flow: str,
        *,
        query: Any | None = None,
    ) -> None:
        token = await self.navigation_flow_sessions.issue(
            update.effective_user.id, update.effective_chat.id, flow
        )
        text = f"Сейчас не завершён сценарий: {FLOW_LABELS[flow]}. Что сделать?"
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Продолжить", callback_data=f"nav:flow:continue:{token}"),
                    InlineKeyboardButton("Выйти в меню", callback_data=f"nav:flow:exit:{token}"),
                ]
            ]
        )
        if query is not None:
            await self._edit_or_send(query, text, markup)
            return
        await message.reply_text(
            text,
            reply_markup=markup,
        )

    @staticmethod
    async def _navigation_stale(query: Any) -> None:
        await query.answer("Эта кнопка устарела или недоступна.", show_alert=True)

    @staticmethod
    async def _edit_or_send(
        query: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        **kwargs: Any,
    ) -> bool:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup, **kwargs)
            return True
        except TelegramError as exc:
            if NavigationHandlers._message_not_modified(exc):
                return True
            if not NavigationHandlers._media_text_limitation(exc):
                logger.warning(
                    "Navigation callback edit failed operation=text error_type=%s",
                    type(exc).__name__,
                )
                return False
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Navigation callback edit failed operation=text error_type=%s",
                type(exc).__name__,
            )
            return False

        edit_caption = getattr(query, "edit_message_caption", None)
        if callable(edit_caption) and len(text) <= 1024:
            try:
                await edit_caption(caption=text, reply_markup=reply_markup, **kwargs)
                return True
            except TelegramError as exc:
                if NavigationHandlers._message_not_modified(exc):
                    return True
                logger.warning(
                    "Navigation callback edit failed operation=caption error_type=%s",
                    type(exc).__name__,
                )
                return False
            except (TypeError, AttributeError) as exc:
                logger.warning(
                    "Navigation callback edit failed operation=caption error_type=%s",
                    type(exc).__name__,
                )
                return False

        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Navigation callback controls retirement failed error_type=%s",
                type(exc).__name__,
            )
        try:
            await query.message.reply_text(text, reply_markup=reply_markup, **kwargs)
            return True
        except (TelegramError, TypeError, AttributeError) as exc:
            logger.warning(
                "Navigation callback replacement failed error_type=%s",
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _message_not_modified(exc: TelegramError) -> bool:
        return "message is not modified" in str(exc).casefold()

    @staticmethod
    def _media_text_limitation(exc: TelegramError) -> bool:
        value = str(exc).casefold()
        return any(
            marker in value
            for marker in (
                "there is no text in the message to edit",
                "message is not a text message",
            )
        )

    async def _send_navigation_root(self, message: Any) -> None:
        await message.reply_text(
            "Главное меню\n\nЧто хочешь сделать?",
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
        rows = [
            [
                InlineKeyboardButton("🌱 Сегодня", callback_data="nav:section:today"),
                InlineKeyboardButton("✅ Задачи", callback_data="nav:section:tasks"),
            ],
            [
                InlineKeyboardButton("📝 Записи", callback_data="nav:section:records"),
                InlineKeyboardButton("❤️ Здоровье", callback_data="nav:section:health"),
            ],
            [
                InlineKeyboardButton(
                    "🎯 Желания и визуализация",
                    callback_data="nav:section:vision",
                )
            ],
            [
                InlineKeyboardButton("🗂 Мои разделы", callback_data="nav:section:sections"),
                InlineKeyboardButton("⚙️ Настройки", callback_data="nav:section:settings"),
            ],
            [InlineKeyboardButton("❓ Помощь", callback_data="nav:help")],
        ]
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
        label_overrides = {
            ("today", "task_today"): "Задачи на сегодня",
        }
        rows = [
            [
                InlineKeyboardButton(
                    label_overrides.get((section_key, action), actions[action].label),
                    callback_data=f"nav:action:{action}",
                )
            ]
            for action in section.actions
        ]
        rows.extend(
            [
                [
                    InlineKeyboardButton(
                        "❓ Помощь",
                        callback_data=f"nav:help:{SECTION_HELP_TOPICS[section_key]}",
                    )
                ],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            ]
        )
        return InlineKeyboardMarkup(rows)

    def _help_keyboard(self) -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton(
                    HELP_TOPIC_LABELS[key],
                    callback_data=f"nav:help:{key}",
                )
            ]
            for key in ROOT_HELP_TOPIC_KEYS
        ]
        rows.append([InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")])
        return InlineKeyboardMarkup(rows)

    @staticmethod
    def _help_back_target(topic_key: str) -> str:
        for section_key, help_key in SECTION_HELP_TOPICS.items():
            if help_key == topic_key:
                return f"nav:section:{section_key}"
        return "nav:help"

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

    def _voice_enabled(self) -> bool:
        return bool(getattr(self.settings, "enable_voice", False))

    def _task_reminders_enabled(self) -> bool:
        return bool(getattr(self.settings, "enable_task_reminders", False))
