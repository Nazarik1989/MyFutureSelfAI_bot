from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from io import BytesIO
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ApplicationHandlerStop, ContextTypes

from .image_generation import (
    ImageGenerationError,
    ImageReferenceInput,
    build_vision_image_prompt,
)
from .vision import CATEGORY_META, PAGE_SIZE
from .vision_images import (
    MAX_IMAGE_INPUT_BYTES,
    MAX_IMAGE_OUTPUT_BYTES,
    MAX_IMAGE_PIXELS,
    TelegramImageMetadata,
    VisionImageError,
    normalize_vision_image,
    validate_telegram_metadata,
)
from .vision_references import (
    MAX_GENERATION_REFERENCES,
    MAX_VISION_REFERENCES,
    REFERENCE_KINDS,
)
from .vision_renderer import MAX_RENDER_ITEMS, VisionRenderItem

logger = logging.getLogger(__name__)


class VisionHandlers:
    """Telegram presentation layer for the persistent owner-scoped vision service."""

    vision_service: Any
    vision_companion_service: Any
    vision_image_service: Any
    vision_image_sessions: Any
    vision_reference_service: Any
    vision_reference_sessions: Any
    image_generation: Any
    vision_renderer: Any
    vision_render_sessions: Any
    vision_render_limiter: Any

    async def vision_command_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.vision_command(update, context)
        raise ApplicationHandlerStop

    async def vision_callback_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await self.vision_action(update, context)
        raise ApplicationHandlerStop

    async def vision_text_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._handle_vision_input(update, update.effective_message.text):
            raise ApplicationHandlerStop

    async def vision_voice_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        if await self.vision_service.draft(user.id, update.effective_chat.id) is None:
            return
        await self.voice(update, context)
        raise ApplicationHandlerStop

    async def vision_image_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        user = await self._user(update.effective_user.id)
        if await self.vision_reference_sessions.has_upload(user.id, update.effective_chat.id):
            await self._vision_reference_input(update, user)
            raise ApplicationHandlerStop
        if not await self.vision_image_sessions.has_upload(user.id, update.effective_chat.id):
            return
        await self._vision_image_input(update, user)
        raise ApplicationHandlerStop

    async def vision_cancel_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        if await self.vision_service.draft(user.id, update.effective_chat.id) is None:
            return
        await self.cancel_draft_edit(update, context)
        raise ApplicationHandlerStop

    async def vision_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        user = await self._user(update.effective_user.id)
        draft = await self.vision_service.draft(user.id, update.effective_chat.id)
        if draft is not None:
            await update.effective_message.reply_text(
                "У тебя есть незавершённая карточка. Продолжаем с сохранённого шага."
            )
            await self._vision_prompt(update.effective_message, draft)
            return
        await self._vision_menu(update.effective_message)

    @staticmethod
    async def _vision_edit_or_send(
        query: Any | None,
        message: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> None:
        """Reuse a callback message when Telegram permits it; otherwise retire its buttons."""
        if query is not None:
            try:
                await query.edit_message_text(text, reply_markup=reply_markup)
                return
            except TelegramError as exc:
                if "message is not modified" in str(exc).lower():
                    return
            except (TypeError, AttributeError):
                pass
            edit_caption = getattr(query, "edit_message_caption", None)
            if edit_caption is not None and len(text) <= 1024:
                try:
                    await edit_caption(caption=text, reply_markup=reply_markup)
                    return
                except TelegramError as exc:
                    if "message is not modified" in str(exc).lower():
                        return
                except (TypeError, AttributeError):
                    pass
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except (TelegramError, TypeError, AttributeError):
                pass
        await message.reply_text(text, reply_markup=reply_markup)
        if query is not None:
            delete = getattr(message, "delete", None)
            if delete is not None:
                try:
                    await delete()
                except (TelegramError, TypeError, AttributeError):
                    pass

    async def _vision_menu(self, message: Any, *, query: Any | None = None) -> None:
        await self._vision_edit_or_send(
            query,
            message,
            "Карта желаний\n\n"
            "Желание → зачем это важно → первый шаг → действие.\n\n"
            "Активные желания находятся в «Моей карте», завершённые — в «Достигнуто», "
            "отложенные — в «Архиве».",
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("➕ Добавить желание", callback_data="vision:add")],
                    [
                        InlineKeyboardButton(
                            "🖼 Создать визуализацию",
                            callback_data="vision:render",
                        )
                    ],
                    [
                        InlineKeyboardButton("🗺 Моя карта", callback_data="vision:list:active:0"),
                        InlineKeyboardButton(
                            "✅ Достигнуто", callback_data="vision:list:achieved:0"
                        ),
                    ],
                    [InlineKeyboardButton("📦 Архив", callback_data="vision:list:archived:0")],
                    [InlineKeyboardButton("🧩 Мои референсы", callback_data="vision:refs")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
                ]
            ),
        )

    @staticmethod
    def _vision_list_navigation(status: str) -> list[list[InlineKeyboardButton]]:
        destinations = (
            ("active", "🗺 Активные"),
            ("achieved", "✅ Достигнуто"),
            ("archived", "📦 Архив"),
        )
        other_statuses = [
            InlineKeyboardButton(label, callback_data=f"vision:list:{code}:0")
            for code, label in destinations
            if code != status
        ]
        return [
            other_statuses,
            [InlineKeyboardButton("← Меню карты", callback_data="vision:menu")],
        ]

    @staticmethod
    def _vision_category_keyboard(draft: Any, *, edit: bool = False) -> InlineKeyboardMarkup:
        prefix = "vision:editcat" if edit else "vision:cat"
        rows = []
        entries = list(CATEGORY_META.items())
        for index in range(0, len(entries), 2):
            row = []
            for code, (emoji, label) in entries[index : index + 2]:
                callback = (
                    f"{prefix}:{draft.id}:{code}"
                    if edit
                    else f"{prefix}:{draft.id}:{draft.version}:{code}"
                )
                row.append(InlineKeyboardButton(f"{emoji} {label}", callback_data=callback))
            rows.append(row)
        rows.append([InlineKeyboardButton("Отменить", callback_data=f"vision:cancel:{draft.id}")])
        return InlineKeyboardMarkup(rows)

    async def _vision_prompt(self, message: Any, draft: Any, *, query: Any | None = None) -> None:
        if draft.step == "category":
            await self._vision_edit_or_send(
                query,
                message,
                "Выбери категорию желания:",
                self._vision_category_keyboard(draft),
            )
            return
        if draft.step == "edit_value" and draft.edit_field == "category":
            await self._vision_edit_or_send(
                query,
                message,
                "Выбери новую категорию:",
                self._vision_category_keyboard(draft, edit=True),
            )
            return
        if draft.step == "delete_confirm":
            await self._vision_edit_or_send(
                query,
                message,
                "Удаление ожидает явного подтверждения.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Да, удалить",
                                callback_data=(
                                    f"vision:delete:{draft.editing_item_id}:"
                                    f"{draft.id}:{draft.version}"
                                ),
                            ),
                            InlineKeyboardButton(
                                "Нет",
                                callback_data=(
                                    f"vision:deletecancel:{draft.editing_item_id}:"
                                    f"{draft.id}:{draft.version}"
                                ),
                            ),
                        ]
                    ]
                ),
            )
            return
        prompts = {
            "wish": "Сформулируй желание как желаемый результат текстом или голосом.",
            "why": "Почему это важно для тебя?",
            "target_date": "Желаемая дата? Формат: ДД.ММ.ГГГГ.",
            "first_step": "Какой первый небольшой шаг можно сделать?",
        }
        if draft.step in prompts:
            rows = []
            if draft.step in {"why", "target_date", "first_step"}:
                rows.append(
                    [
                        InlineKeyboardButton(
                            "Пропустить",
                            callback_data=f"vision:skip:{draft.id}:{draft.version}",
                        )
                    ]
                )
            rows.append(
                [InlineKeyboardButton("Отменить", callback_data=f"vision:cancel:{draft.id}")]
            )
            await self._vision_edit_or_send(
                query,
                message,
                prompts[draft.step],
                InlineKeyboardMarkup(rows),
            )
            return
        if draft.step == "edit_value":
            field_name = {
                "wish": "желание",
                "why": "почему это важно",
                "target_date": "желаемую дату в формате ДД.ММ.ГГГГ",
                "first_step": "первый небольшой шаг",
            }.get(draft.edit_field, "новое значение")
            rows = []
            if draft.edit_field in {"why", "target_date", "first_step"}:
                rows.append(
                    [
                        InlineKeyboardButton(
                            "Очистить поле",
                            callback_data=f"vision:skip:{draft.id}:{draft.version}",
                        )
                    ]
                )
            rows.append(
                [InlineKeyboardButton("Отменить", callback_data=f"vision:cancel:{draft.id}")]
            )
            await self._vision_edit_or_send(
                query,
                message,
                f"Пришли {field_name} текстом или голосом.",
                InlineKeyboardMarkup(rows),
            )
            return
        if draft.step == "preview":
            await self._vision_edit_or_send(
                query,
                message,
                self._vision_preview_text(draft),
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Сохранить",
                                callback_data=f"vision:confirm:{draft.id}:{draft.version}",
                            ),
                            InlineKeyboardButton(
                                "Отменить",
                                callback_data=f"vision:cancel:{draft.id}",
                            ),
                        ]
                    ]
                ),
            )

    @staticmethod
    def _vision_preview_text(draft: Any) -> str:
        emoji, category = CATEGORY_META[draft.category]
        return (
            "Preview карточки\n\n"
            f"{emoji} {category}\n"
            f"Желание: {draft.wish_text}\n"
            f"Почему важно: {draft.why_text or 'не указано'}\n"
            f"Желаемая дата: "
            f"{draft.target_date.strftime('%d.%m.%Y') if draft.target_date else 'не указана'}\n"
            f"Первый шаг: {draft.first_step or 'не указан'}\n\n"
            "Карточка сохранится только после явного подтверждения."
        )

    async def _handle_vision_input(self, update: Update, value: str) -> bool:
        user = await self._user(update.effective_user.id)
        rename_flow = await self.vision_reference_sessions.awaiting_rename(
            user.id, update.effective_chat.id
        )
        if rename_flow is not None:
            capability = await self.vision_reference_sessions.claim_rename(
                rename_flow.token,
                user.id,
                update.effective_chat.id,
                value,
            )
            if capability is None:
                await update.effective_message.reply_text(
                    "Название должно быть от 1 до 60 символов. Попробуй короче."
                )
                return True
            result = await self.vision_reference_service.rename(
                user.id,
                capability.reference_id,
                expected_version=capability.expected_version,
                name=capability.name,
            )
            if result.status not in {"renamed", "existing"}:
                await update.effective_message.reply_text(
                    "Референс изменился или недоступен. Открой библиотеку заново."
                )
                return True
            await update.effective_message.reply_text(
                "Название не изменилось."
                if result.status == "existing"
                else "Референс переименован."
            )
            await self._vision_reference_library(update.effective_message, user.id)
            return True
        reference_flow = await self.vision_reference_sessions.awaiting_name(
            user.id, update.effective_chat.id
        )
        if reference_flow is not None:
            capability = await self.vision_reference_sessions.set_name(
                reference_flow.token,
                user.id,
                update.effective_chat.id,
                value,
            )
            if capability is None:
                await update.effective_message.reply_text(
                    "Название должно быть от 1 до 60 символов. Попробуй короче."
                )
                return True
            await self._vision_reference_upload_prompt(update.effective_message, capability.token)
            return True
        draft = await self.vision_service.draft(user.id, update.effective_chat.id)
        if draft is None:
            return False
        try:
            outcome = await self.vision_service.consume_text(
                user.id, update.effective_chat.id, value
            )
        except ValueError as exc:
            await update.effective_message.reply_text(str(exc))
            return True
        if outcome.status == "need_category":
            await self._vision_prompt(update.effective_message, outcome.draft)
        elif outcome.status == "need_confirm":
            await update.effective_message.reply_text(
                "Карточка уже собрана. Используй кнопку «Сохранить» или «Отменить»."
            )
            await self._vision_prompt(update.effective_message, outcome.draft)
        elif outcome.status == "need_delete_confirm":
            await self._vision_prompt(update.effective_message, outcome.draft)
        elif outcome.status == "invalid":
            await update.effective_message.reply_text("Ответ не должен быть пустым.")
        elif outcome.status == "edited":
            await update.effective_message.reply_text("Карточка обновлена.")
            await self._vision_send_item(update.effective_message, outcome.item)
        elif outcome.draft is not None:
            await self._vision_prompt(update.effective_message, outcome.draft)
        return True

    async def vision_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        parts = query.data.split(":")
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        action = parts[1] if len(parts) > 1 else ""

        if action == "add" and len(parts) == 2:
            await query.answer()
            try:
                draft = await self.vision_service.begin(user.id, chat_id)
            except ValueError:
                await query.message.reply_text(
                    "Незавершённая карточка уже открыта в другом личном чате."
                )
                return
            await self._vision_prompt(query.message, draft, query=query)
            return
        if action == "menu" and len(parts) == 2:
            await query.answer()
            await self._vision_menu(query.message, query=query)
            return
        if action == "refs" and len(parts) == 2:
            await query.answer()
            await self._vision_reference_library(query.message, user.id, query=query)
            return
        if action == "refadd" and len(parts) == 2:
            await self._vision_reference_add(query, user.id, chat_id)
            return
        if action == "refkind" and len(parts) == 4:
            await self._vision_reference_kind(query, user.id, chat_id, parts[2], parts[3])
            return
        if action == "refnamedefault" and len(parts) == 3:
            await self._vision_reference_default_name(query, user.id, chat_id, parts[2])
            return
        if action in {"refview", "refreplace", "refrename", "refdeleteask"} and len(parts) == 3:
            try:
                reference_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            if action == "refview":
                await self._vision_reference_view(query, user.id, reference_id)
            elif action == "refreplace":
                await self._vision_reference_replace(query, user.id, chat_id, reference_id)
            elif action == "refrename":
                await self._vision_reference_rename(query, user.id, chat_id, reference_id)
            else:
                await self._vision_reference_delete_ask(query, user.id, chat_id, reference_id)
            return
        if (
            action in {"refconfirm", "refcancel", "refdelete", "refdeletecancel"}
            and len(parts) == 3
        ):
            await self._vision_reference_capability_action(
                query, user.id, chat_id, action, parts[2]
            )
            return
        if action == "render" and len(parts) == 2:
            await self._vision_render_menu(query, user.id, chat_id)
            return
        if action == "renderpick" and len(parts) == 4:
            token, category = parts[2], parts[3]
            selection = await self.vision_render_sessions.claim_selection(
                token,
                user.id,
                chat_id,
                category,
            )
            if selection is None:
                await self._vision_render_stale(query)
                return
            await query.answer()
            try:
                await query.edit_message_text("Создаю карту желаний… Это займёт несколько секунд.")
            except TelegramError:
                pass
            await self._vision_render_and_send(
                query.message,
                user,
                None if selection == "all" else selection,
                token=token,
                as_document=False,
                remove_source=True,
            )
            return
        if action == "renderdownload" and len(parts) == 3:
            token = parts[2]
            selection = await self.vision_render_sessions.claim_download(
                token,
                user.id,
                chat_id,
            )
            if selection is None:
                await self._vision_render_stale(query)
                return
            await query.answer()
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except TelegramError:
                pass
            await self._vision_render_and_send(
                query.message,
                user,
                None if selection == "all" else selection,
                token=token,
                as_document=True,
                remove_source=False,
            )
            return
        if action == "rendercancel" and len(parts) == 3:
            if not await self.vision_render_sessions.cancel(parts[2], user.id, chat_id):
                await self._vision_render_stale(query)
                return
            await query.answer()
            await query.edit_message_text("Визуализация отменена.")
            return
        if action in {"imageadd", "imagereplace", "imagedeleteask"} and len(parts) == 3:
            try:
                item_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            await self._vision_image_action(query, user.id, chat_id, action, item_id)
            return
        if action == "imagegenerateask" and len(parts) == 3:
            try:
                item_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            await self._vision_image_generation_ask(query, user.id, chat_id, item_id)
            return
        if action == "imagegenerate" and len(parts) == 3:
            await self._vision_image_generate(query, user.id, chat_id, parts[2])
            return
        if action == "genrefs" and len(parts) == 3:
            await self._vision_generation_references_begin(query, user.id, chat_id, parts[2])
            return
        if action == "genreftoggle" and len(parts) == 4:
            try:
                reference_id = int(parts[3])
            except ValueError:
                await self._vision_stale(query)
                return
            await self._vision_generation_reference_toggle(
                query, user.id, chat_id, parts[2], reference_id
            )
            return
        if action == "genrefdone" and len(parts) == 3:
            await self._vision_generation_references_done(query, user.id, chat_id, parts[2])
            return
        if (
            action
            in {
                "imageconfirm",
                "imagecancel",
                "imagedelete",
                "imagedeletecancel",
            }
            and len(parts) == 3
        ):
            await self._vision_image_capability_action(
                query,
                user.id,
                chat_id,
                action,
                parts[2],
            )
            return
        if action in {"companion", "companionon", "companionfreq", "companionoff"}:
            try:
                item_id = int(parts[2])
                count = int(parts[3]) if action == "companionfreq" else None
            except (IndexError, ValueError):
                await self._vision_stale(query)
                return
            if action == "companion" and len(parts) == 3:
                await self._vision_companion_settings(query, user, item_id)
                return
            if action == "companionon" and len(parts) == 3:
                await self._vision_companion_enable(query, user, chat_id, item_id)
                return
            if action == "companionfreq" and len(parts) == 4 and count is not None:
                await self._vision_companion_frequency(query, user.id, item_id, count)
                return
            if action == "companionoff" and len(parts) == 3:
                await self._vision_companion_disable(query, user, item_id)
                return
            await self._vision_stale(query)
            return
        if action == "companioncancel" and len(parts) == 3:
            await query.answer()
            await query.edit_message_text("Сопровождение не включено. Вернуться можно из карточки.")
            return
        if action == "companionlog" and len(parts) == 5:
            try:
                item_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            await self._vision_companion_record(query, user.id, item_id, parts[3], parts[4])
            return
        if action == "list" and len(parts) == 4:
            try:
                page = max(int(parts[3]), 0)
            except ValueError:
                await self._vision_stale(query)
                return
            if parts[2] not in {"active", "achieved", "archived"}:
                await self._vision_stale(query)
                return
            await query.answer()
            await self._vision_send_page(query.message, user.id, parts[2], page, query=query)
            return
        if action == "cat" and len(parts) == 5:
            try:
                draft_id, version = int(parts[2]), int(parts[3])
            except ValueError:
                await self._vision_stale(query)
                return
            draft = await self.vision_service.draft(user.id, chat_id)
            if draft is None or draft.id != draft_id or draft.version != version:
                await self._vision_stale(query)
                return
            outcome = await self.vision_service.choose_category(
                user.id, chat_id, parts[4], draft_id=draft_id
            )
            if outcome.status != "advanced":
                await self._vision_stale(query)
                return
            await query.answer()
            await query.edit_message_reply_markup(reply_markup=None)
            await self._vision_prompt(query.message, outcome.draft, query=query)
            return
        if action == "editcat" and len(parts) == 4:
            try:
                draft_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            outcome = await self.vision_service.choose_category(
                user.id, chat_id, parts[3], draft_id=draft_id
            )
            if outcome.status != "edited":
                await self._vision_stale(query)
                return
            await query.answer()
            await self._vision_send_item(
                query.message, outcome.item, query=query, include_image=False
            )
            return
        if action == "skip" and len(parts) == 4:
            try:
                draft_id, version = int(parts[2]), int(parts[3])
            except ValueError:
                await self._vision_stale(query)
                return
            outcome = await self.vision_service.skip(user.id, chat_id, draft_id, version)
            if outcome.status not in {"advanced", "edited"}:
                await self._vision_stale(query)
                return
            await query.answer()
            if outcome.status == "edited":
                await self._vision_send_item(
                    query.message, outcome.item, query=query, include_image=False
                )
            else:
                await self._vision_prompt(query.message, outcome.draft, query=query)
            return
        if action == "confirm" and len(parts) == 4:
            try:
                draft_id, version = int(parts[2]), int(parts[3])
            except ValueError:
                await self._vision_stale(query)
                return
            outcome = await self.vision_service.confirm(user.id, chat_id, draft_id, version)
            if outcome.status != "created":
                await self._vision_stale(query)
                return
            await query.answer()
            await query.edit_message_text("Желание сохранено в твою карту.")
            await self._vision_send_item(
                query.message, outcome.item, query=query, include_image=False
            )
            return
        if action == "cancel" and len(parts) == 3:
            try:
                draft_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            draft = await self.vision_service.draft(user.id, chat_id)
            if draft is None or draft.id != draft_id:
                await self._vision_stale(query)
                return
            editing_item_id = draft.editing_item_id
            await self.vision_service.cancel(user.id, chat_id)
            await query.answer()
            await query.edit_message_text("Создание или редактирование отменено.")
            if editing_item_id is not None:
                item = await self.vision_service.get_item(user.id, editing_item_id)
                if item is not None:
                    await self._vision_send_item(
                        query.message, item, query=query, include_image=False
                    )
                    return
            await self._vision_menu(query.message, query=query)
            return
        if (
            action
            in {
                "view",
                "edit",
                "deleteask",
                "task",
                "archive",
            }
            and len(parts) == 3
        ):
            try:
                item_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            await self._vision_item_action(query, user.id, chat_id, action, item_id)
            return
        if action in {"delete", "deletecancel"} and len(parts) == 5:
            try:
                item_id, draft_id, version = map(int, parts[2:5])
            except ValueError:
                await self._vision_stale(query)
                return
            if action == "delete":
                outcome = await self.vision_service.confirm_delete(
                    user.id,
                    chat_id,
                    item_id,
                    draft_id,
                    version,
                )
            else:
                outcome = await self.vision_service.cancel_delete(
                    user.id,
                    chat_id,
                    item_id,
                    draft_id,
                    version,
                )
            if outcome.status not in {"deleted", "cancelled"}:
                await self._vision_stale(query)
                return
            await query.answer()
            if outcome.status == "deleted":
                await query.edit_message_text(
                    "Карточка удалена.",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "← К списку",
                                    callback_data=f"vision:list:{outcome.item.status}:0",
                                ),
                                InlineKeyboardButton("Меню карты", callback_data="vision:menu"),
                            ]
                        ]
                    ),
                )
            else:
                await query.edit_message_text("Удаление отменено.")
            if outcome.status == "cancelled":
                await self._vision_send_item(
                    query.message, outcome.item, query=query, include_image=False
                )
            return
        if action == "status" and len(parts) == 4:
            try:
                item_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            item = await self.vision_service.set_status(user.id, item_id, parts[3])
            if item is None:
                await self._vision_stale(query)
                return
            await query.answer()
            await query.edit_message_text("Статус карточки обновлён.")
            await self._vision_send_item(query.message, item, query=query, include_image=False)
            return
        if action == "editfield" and len(parts) == 4:
            try:
                item_id = int(parts[2])
            except ValueError:
                await self._vision_stale(query)
                return
            outcome = await self.vision_service.start_edit(user.id, chat_id, item_id, parts[3])
            if outcome.status == "busy":
                await query.answer("Сначала закончи или отмени текущую карточку.", show_alert=True)
                return
            if outcome.status != "editing":
                await self._vision_stale(query)
                return
            await query.answer()
            await self._vision_prompt(query.message, outcome.draft, query=query)
            return
        await self._vision_stale(query)

    async def _vision_render_menu(self, query: Any, owner_id: int, chat_id: int) -> None:
        counts = await self.vision_service.category_counts(owner_id, "active")
        available = {category for category, count in counts.items() if count > 0}
        await query.answer()
        if not available:
            await self._vision_edit_or_send(
                query,
                query.message,
                "Активных желаний пока нет. Сначала добавь желание.",
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("➕ Добавить желание", callback_data="vision:add")],
                        [InlineKeyboardButton("← Меню карты", callback_data="vision:menu")],
                    ]
                ),
            )
            return
        token = await self.vision_render_sessions.issue(owner_id, chat_id, available)
        rows = [
            [
                InlineKeyboardButton(
                    "🗺 Вся карта",
                    callback_data=f"vision:renderpick:{token}:all",
                )
            ]
        ]
        entries = [(code, CATEGORY_META[code]) for code in CATEGORY_META if code in available]
        for index in range(0, len(entries), 2):
            rows.append(
                [
                    InlineKeyboardButton(
                        f"{emoji} {label}",
                        callback_data=f"vision:renderpick:{token}:{code}",
                    )
                    for code, (emoji, label) in entries[index : index + 2]
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    "Отмена",
                    callback_data=f"vision:rendercancel:{token}",
                )
            ]
        )
        await self._vision_edit_or_send(
            query,
            query.message,
            "Что визуализировать? В изображение попадут только активные желания.",
            InlineKeyboardMarkup(rows),
        )

    async def _vision_render_and_send(
        self,
        message: Any,
        user: Any,
        category: str | None,
        *,
        token: str,
        as_document: bool,
        remove_source: bool = False,
    ) -> None:
        if not await self.vision_render_limiter.acquire(user.id):
            await message.reply_text(
                "Визуализация уже создаётся. Дождись завершения текущего запроса."
            )
            return
        try:
            items, total = await self.vision_service.active_for_render(
                user.id,
                category=category,
                limit=MAX_RENDER_ITEMS,
            )
            if not items:
                await message.reply_text(
                    "Для этого выбора активных желаний нет. Открой /vision и добавь карточку."
                )
                return
            snapshots = [
                VisionRenderItem(
                    category=item.category,
                    wish_text=item.wish_text,
                    target_date=item.target_date,
                    sort_id=item.id,
                    image_bytes=item.image.image_bytes if item.image is not None else None,
                )
                for item in items
            ]
            try:
                local_date = datetime.now(ZoneInfo(user.timezone)).date()
            except (TypeError, ZoneInfoNotFoundError):
                local_date = datetime.now(ZoneInfo("UTC")).date()
            board = await asyncio.to_thread(
                self.vision_renderer.render,
                snapshots,
                created_on=local_date,
                category=category,
                total_count=total,
            )
            category_label = "Вся карта" if category is None else CATEGORY_META[category][1]
            for page_index, page in enumerate(board.pages, start=1):
                stream = BytesIO(page.png)
                filename = f"vision-board-{page_index}-of-{len(board.pages)}.png"
                stream.name = filename
                caption = (
                    f"Активных желаний: {total} · {category_label} · "
                    f"страница {page_index}/{len(board.pages)}"
                )
                if board.omitted_count and page_index == len(board.pages):
                    caption += (
                        f"\nПоказано {board.included_count}; ещё {board.omitted_count} "
                        "доступны через /vision."
                    )
                try:
                    if as_document:
                        await message.reply_document(
                            document=stream,
                            filename=filename,
                            caption=caption,
                        )
                    else:
                        reply_markup = None
                        if page_index == len(board.pages):
                            reply_markup = InlineKeyboardMarkup(
                                [
                                    [
                                        InlineKeyboardButton(
                                            "Скачать PNG",
                                            callback_data=f"vision:renderdownload:{token}",
                                        ),
                                        InlineKeyboardButton(
                                            "← Меню карты",
                                            callback_data="vision:menu",
                                        ),
                                    ],
                                ]
                            )
                        await message.reply_photo(
                            photo=stream,
                            caption=caption,
                            reply_markup=reply_markup,
                        )
                finally:
                    stream.close()
            if remove_source:
                delete = getattr(message, "delete", None)
                if delete is not None:
                    try:
                        await delete()
                    except (TelegramError, TypeError, AttributeError):
                        pass
        except Exception as exc:  # Telegram and Pillow adapters fail closed here.
            logger.error("Vision render failed error_type=%s", type(exc).__name__)
            await message.reply_text(
                "Не удалось создать визуализацию. Попробуй ещё раз немного позже."
            )
        finally:
            await self.vision_render_limiter.release(user.id)

    @staticmethod
    async def _vision_render_stale(query: Any) -> None:
        await query.answer(
            "Запрос визуализации недоступен или устарел. Открой /vision ещё раз.",
            show_alert=True,
        )
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except TelegramError:
            pass

    async def _vision_item_action(
        self, query: Any, owner_id: int, chat_id: int, action: str, item_id: int
    ) -> None:
        item = await self.vision_service.get_item(owner_id, item_id)
        if item is None:
            await self._vision_stale(query)
            return
        if action == "view":
            await query.answer()
            await self._vision_send_item(query.message, item, query=query)
            return
        if action == "edit":
            await query.answer()
            await self._vision_edit_or_send(
                query,
                query.message,
                "Что изменить?",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Категорию",
                                callback_data=f"vision:editfield:{item.id}:category",
                            ),
                            InlineKeyboardButton(
                                "Желание",
                                callback_data=f"vision:editfield:{item.id}:wish",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Почему важно",
                                callback_data=f"vision:editfield:{item.id}:why",
                            ),
                            InlineKeyboardButton(
                                "Дату",
                                callback_data=f"vision:editfield:{item.id}:target_date",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Первый шаг",
                                callback_data=f"vision:editfield:{item.id}:first_step",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "← К карточке", callback_data=f"vision:view:{item.id}"
                            ),
                            InlineKeyboardButton("Меню карты", callback_data="vision:menu"),
                        ],
                    ]
                ),
            )
            return
        if action == "archive":
            updated = await self.vision_service.set_status(owner_id, item.id, "archived")
            await query.answer()
            await query.edit_message_text(
                "Карточка архивирована." if updated is not None else "Карточка недоступна.",
                reply_markup=(
                    InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "📦 Открыть архив",
                                    callback_data="vision:list:archived:0",
                                )
                            ],
                            [InlineKeyboardButton("← Меню карты", callback_data="vision:menu")],
                        ]
                    )
                    if updated is not None
                    else None
                ),
            )
            return
        if action == "deleteask":
            outcome = await self.vision_service.start_delete(owner_id, chat_id, item.id)
            if outcome.status == "busy":
                await query.answer(
                    "Сначала закончи или отмени текущую операцию с карточкой.",
                    show_alert=True,
                )
                return
            if outcome.status != "confirming":
                await self._vision_stale(query)
                return
            await query.answer()
            await self._vision_prompt(query.message, outcome.draft, query=query)
            return
        if action == "task":
            result = await self.vision_service.create_task(owner_id, item.id)
            if result.status == "missing_step":
                await query.answer(
                    "Сначала добавь первый шаг через «Редактировать».",
                    show_alert=True,
                )
                return
            if result.status == "stale":
                await self._vision_stale(query)
                return
            await query.answer()
            await query.edit_message_text(
                "Задача уже была создана; дубликат не добавлен."
                if result.status == "existing"
                else "Задача создана без reminder. Напоминание можно назначить отдельно."
            )
            refreshed = await self.vision_service.get_item(owner_id, item.id)
            if refreshed is not None:
                await self._vision_send_item(
                    query.message, refreshed, query=query, include_image=False
                )

    async def _vision_reference_library(
        self,
        message: Any,
        owner_id: int,
        *,
        query: Any | None = None,
        notice: str | None = None,
    ) -> None:
        references = await self.vision_reference_service.list(owner_id)
        lines = ([notice, ""] if notice else []) + [
            "Мои референсы",
            "",
            "Это приватная библиотека: фото остаются в боте и используются только "
            "после твоего выбора перед конкретной генерацией.",
        ]
        rows: list[list[InlineKeyboardButton]] = []
        if references:
            lines.extend(["", f"Сохранено: {len(references)}/{MAX_VISION_REFERENCES}"])
            for reference in references:
                kind = REFERENCE_KINDS[reference.kind]
                rows.append(
                    [
                        InlineKeyboardButton(
                            f"🧩 #{reference.id} · {kind} · {reference.name}"[:60],
                            callback_data=f"vision:refview:{reference.id}",
                        )
                    ]
                )
        else:
            lines.extend(["", "Референсов пока нет."])
        if len(references) < MAX_VISION_REFERENCES:
            rows.append(
                [InlineKeyboardButton("➕ Добавить референс", callback_data="vision:refadd")]
            )
        rows.append([InlineKeyboardButton("← К карте желаний", callback_data="vision:menu")])
        await self._vision_edit_or_send(
            query,
            message,
            "\n".join(lines),
            InlineKeyboardMarkup(rows),
        )

    async def _vision_reference_add(self, query: Any, owner_id: int, chat_id: int) -> None:
        if await self.vision_reference_service.count(owner_id) >= MAX_VISION_REFERENCES:
            await query.answer(
                f"В библиотеке уже максимум: {MAX_VISION_REFERENCES}.", show_alert=True
            )
            return
        if await self.vision_image_sessions.has_active(owner_id, chat_id):
            await query.answer(
                "Сначала заверши или отмени текущую операцию с изображением.",
                show_alert=True,
            )
            return
        token = await self.vision_reference_sessions.issue_create(owner_id, chat_id)
        if token is None:
            await query.answer(
                "Сначала заверши или отмени текущую операцию с референсом.",
                show_alert=True,
            )
            return
        await query.answer()
        rows = [
            [InlineKeyboardButton(label, callback_data=f"vision:refkind:{token}:{kind}")]
            for kind, label in REFERENCE_KINDS.items()
        ]
        rows.append([InlineKeyboardButton("Отмена", callback_data=f"vision:refcancel:{token}")])
        await self._vision_edit_or_send(
            query,
            query.message,
            "Что показывает этот референс? Тип поможет модели правильно его использовать.",
            InlineKeyboardMarkup(rows),
        )

    async def _vision_reference_kind(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        token: str,
        kind: str,
    ) -> None:
        capability = await self.vision_reference_sessions.choose_kind(
            token, owner_id, chat_id, kind
        )
        if capability is None:
            await self._vision_stale(query)
            return
        await query.answer()
        await query.edit_message_text(
            f"Тип: {REFERENCE_KINDS[kind]}.\n\n"
            "Напиши короткое понятное название, например «Я сейчас», «Дом у моря» "
            "или «Тёплая плёнка».",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"Оставить «{REFERENCE_KINDS[kind]}»",
                            callback_data=f"vision:refnamedefault:{token}",
                        )
                    ],
                    [InlineKeyboardButton("Отмена", callback_data=f"vision:refcancel:{token}")],
                ]
            ),
        )

    async def _vision_reference_default_name(
        self, query: Any, owner_id: int, chat_id: int, token: str
    ) -> None:
        capability = await self.vision_reference_sessions.set_name(token, owner_id, chat_id, None)
        if capability is None:
            await self._vision_stale(query)
            return
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        await self._vision_reference_upload_prompt(query.message, token, query=query)

    async def _vision_reference_upload_prompt(
        self, message: Any, token: str, *, query: Any | None = None
    ) -> None:
        await self._vision_edit_or_send(
            query,
            message,
            "Теперь отправь одно фото или image-document в JPEG, PNG или WebP. "
            f"До {MAX_IMAGE_INPUT_BYTES // (1024 * 1024)} МБ и "
            f"{MAX_IMAGE_PIXELS // 1_000_000} Мп. Метаданные и оригинал не сохраняются: "
            "в библиотеку попадёт безопасная нормализованная копия.",
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("Отмена", callback_data=f"vision:refcancel:{token}")]]
            ),
        )

    async def _vision_reference_input(self, update: Update, user: Any) -> None:
        message = update.effective_message
        capability = await self.vision_reference_sessions.claim_upload(
            user.id, update.effective_chat.id
        )
        if capability is None:
            await message.reply_text("Референс уже обрабатывается. Дождись preview.")
            return
        try:
            media, metadata = self._vision_telegram_image(message)
            validate_telegram_metadata(metadata)
            telegram_file = await media.get_file()
            raw = bytes(await telegram_file.download_as_bytearray())
            normalized = await asyncio.to_thread(
                normalize_vision_image, raw, declared_mime=metadata.mime_type
            )
            if not await self.vision_reference_sessions.attach_preview(
                capability.token, user.id, update.effective_chat.id, normalized
            ):
                await message.reply_text(
                    "Не удалось подготовить preview: лимит временной памяти исчерпан."
                )
                return
            stream = BytesIO(normalized.image_bytes)
            stream.name = "vision-reference-preview.jpg"
            try:
                await message.reply_photo(
                    photo=stream,
                    caption=(
                        f"Референс «{capability.name}» · "
                        f"{REFERENCE_KINDS[capability.kind]} · "
                        f"{normalized.width}×{normalized.height}.\n"
                        "Сохранить в приватную библиотеку?"
                    ),
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "Сохранить",
                                    callback_data=f"vision:refconfirm:{capability.token}",
                                ),
                                InlineKeyboardButton(
                                    "Отмена",
                                    callback_data=f"vision:refcancel:{capability.token}",
                                ),
                            ]
                        ]
                    ),
                )
            finally:
                stream.close()
        except VisionImageError:
            await self.vision_reference_sessions.retry_upload(
                capability.token, user.id, update.effective_chat.id
            )
            await message.reply_text(
                "Файл отклонён. Нужен статический JPEG, PNG или WebP без повреждений, "
                f"не больше {MAX_IMAGE_INPUT_BYTES // (1024 * 1024)} МБ и "
                f"{MAX_IMAGE_PIXELS // 1_000_000} Мп."
            )
        except TelegramError as exc:
            await self.vision_reference_sessions.cancel(
                capability.token, user.id, update.effective_chat.id
            )
            logger.error("Vision reference transport failed error_type=%s", type(exc).__name__)
            await message.reply_text(
                "Не удалось безопасно загрузить референс. Открой библиотеку и попробуй снова."
            )
        except Exception as exc:
            await self.vision_reference_sessions.cancel(
                capability.token, user.id, update.effective_chat.id
            )
            logger.error("Vision reference processing failed error_type=%s", type(exc).__name__)
            await message.reply_text(
                "Не удалось обработать референс. Открой библиотеку и попробуй снова."
            )

    async def _vision_reference_view(self, query: Any, owner_id: int, reference_id: int) -> None:
        reference = await self.vision_reference_service.get(owner_id, reference_id)
        if reference is None:
            await self._vision_stale(query)
            return
        await query.answer()
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except TelegramError:
            pass
        stream = BytesIO(reference.image_bytes)
        stream.name = "vision-reference.jpg"
        try:
            await query.message.reply_photo(
                photo=stream,
                caption=(
                    f"🧩 Референс #{reference.id}\n"
                    f"Название: {reference.name}\n"
                    f"Тип: {REFERENCE_KINDS[reference.kind]}\n"
                    f"Размер: {reference.width}×{reference.height}\n\n"
                    "Он не отправляется модели автоматически."
                ),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Переименовать",
                                callback_data=f"vision:refrename:{reference.id}",
                            ),
                            InlineKeyboardButton(
                                "Заменить фото",
                                callback_data=f"vision:refreplace:{reference.id}",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Удалить",
                                callback_data=f"vision:refdeleteask:{reference.id}",
                            )
                        ],
                        [InlineKeyboardButton("← К библиотеке", callback_data="vision:refs")],
                    ]
                ),
            )
        finally:
            stream.close()

    async def _vision_reference_replace(
        self, query: Any, owner_id: int, chat_id: int, reference_id: int
    ) -> None:
        reference = await self.vision_reference_service.get(owner_id, reference_id)
        if reference is None:
            await self._vision_stale(query)
            return
        if await self.vision_image_sessions.has_active(owner_id, chat_id):
            await query.answer("Сначала заверши или отмени текущую генерацию.", show_alert=True)
            return
        token = await self.vision_reference_sessions.issue_replace(
            owner_id,
            chat_id,
            reference.id,
            expected_version=reference.version,
            kind=reference.kind,
            name=reference.name,
        )
        if token is None:
            await query.answer(
                "Сначала заверши или отмени текущую операцию с референсом.",
                show_alert=True,
            )
            return
        await query.answer()
        await self._vision_edit_or_send(
            query,
            query.message,
            f"Замена референса «{reference.name}». Старое фото останется до подтверждения нового.\n\n"
            "Теперь отправь новое фото или image-document в JPEG, PNG или WebP.",
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("Отмена", callback_data=f"vision:refcancel:{token}")]]
            ),
        )

    async def _vision_reference_rename(
        self, query: Any, owner_id: int, chat_id: int, reference_id: int
    ) -> None:
        reference = await self.vision_reference_service.get(owner_id, reference_id)
        if reference is None:
            await self._vision_stale(query)
            return
        if await self.vision_image_sessions.has_active(owner_id, chat_id):
            await query.answer("Сначала заверши или отмени текущую генерацию.", show_alert=True)
            return
        token = await self.vision_reference_sessions.issue_rename(
            owner_id,
            chat_id,
            reference.id,
            expected_version=reference.version,
            kind=reference.kind,
            name=reference.name,
        )
        if token is None:
            await query.answer(
                "Сначала заверши или отмени текущую операцию с референсом.",
                show_alert=True,
            )
            return
        await query.answer()
        await self._vision_edit_or_send(
            query,
            query.message,
            f"Текущее название: «{reference.name}». Напиши новое название (до 60 символов).",
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("Отмена", callback_data=f"vision:refcancel:{token}")]]
            ),
        )

    async def _vision_reference_delete_ask(
        self, query: Any, owner_id: int, chat_id: int, reference_id: int
    ) -> None:
        reference = await self.vision_reference_service.get(owner_id, reference_id)
        if reference is None:
            await self._vision_stale(query)
            return
        if await self.vision_image_sessions.has_active(owner_id, chat_id):
            await query.answer("Сначала заверши или отмени текущую генерацию.", show_alert=True)
            return
        token = await self.vision_reference_sessions.issue_delete(
            owner_id,
            chat_id,
            reference_id,
            expected_version=reference.version,
        )
        if token is None:
            await query.answer(
                "Сначала заверши или отмени текущую операцию с референсом.",
                show_alert=True,
            )
            return
        await query.answer()
        await self._vision_edit_or_send(
            query,
            query.message,
            f"Удалить референс «{reference.name}» навсегда? Карточки желаний останутся.",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Удалить референс",
                            callback_data=f"vision:refdelete:{token}",
                        ),
                        InlineKeyboardButton(
                            "Отмена", callback_data=f"vision:refdeletecancel:{token}"
                        ),
                    ]
                ]
            ),
        )

    async def _vision_reference_capability_action(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        action: str,
        token: str,
    ) -> None:
        if action in {"refcancel", "refdeletecancel"}:
            if not await self.vision_reference_sessions.cancel(token, owner_id, chat_id):
                await self._vision_stale(query)
                return
            await query.answer()
            await self._vision_reference_library(
                query.message,
                owner_id,
                query=query,
                notice=(
                    "Добавление референса отменено."
                    if action == "refcancel"
                    else "Удаление референса отменено."
                ),
            )
            return
        if action == "refconfirm":
            capability = await self.vision_reference_sessions.claim_confirm(
                token, owner_id, chat_id
            )
            if (
                capability is None
                or capability.image is None
                or capability.kind is None
                or capability.name is None
                or (
                    capability.mode == "replace"
                    and (capability.reference_id is None or capability.expected_version is None)
                )
            ):
                await self._vision_stale(query)
                return
            if capability.mode == "replace":
                result = await self.vision_reference_service.replace(
                    owner_id,
                    capability.reference_id,
                    expected_version=capability.expected_version,
                    normalized=capability.image,
                )
            else:
                result = await self.vision_reference_service.save(
                    owner_id,
                    kind=capability.kind,
                    name=capability.name,
                    normalized=capability.image,
                )
            if result.status == "limit":
                await query.answer()
                await self._vision_reference_library(
                    query.message,
                    owner_id,
                    query=query,
                    notice=(
                        f"Лимит {MAX_VISION_REFERENCES} референсов достигнут. "
                        "Удалить ненужный можно в библиотеке."
                    ),
                )
                return
            if result.status == "duplicate":
                await query.answer()
                await self._vision_reference_library(
                    query.message,
                    owner_id,
                    query=query,
                    notice=("Такое фото уже сохранено как другой референс; замена не выполнена."),
                )
                return
            if result.status not in {"created", "replaced", "existing"}:
                await self._vision_stale(query)
                return
            await query.answer()
            await self._vision_reference_library(
                query.message,
                owner_id,
                query=query,
                notice=(
                    "Такое изображение уже есть в библиотеке; дубль не создан."
                    if result.status == "existing"
                    else (
                        "Фото референса заменено."
                        if result.status == "replaced"
                        else "Референс сохранён и останется в приватной библиотеке."
                    )
                ),
            )
            return
        capability = await self.vision_reference_sessions.claim_delete(token, owner_id, chat_id)
        if (
            capability is None
            or capability.reference_id is None
            or capability.expected_version is None
        ):
            await self._vision_stale(query)
            return
        result = await self.vision_reference_service.delete(
            owner_id,
            capability.reference_id,
            expected_version=capability.expected_version,
        )
        if result.status != "deleted":
            await self._vision_stale(query)
            return
        await query.answer()
        await self._vision_reference_library(
            query.message,
            owner_id,
            query=query,
            notice="Референс удалён из приватной библиотеки.",
        )

    async def _vision_image_action(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        action: str,
        item_id: int,
    ) -> None:
        if await self.vision_reference_sessions.has_active(owner_id, chat_id):
            await query.answer(
                "Сначала заверши или отмени текущую операцию с референсом.",
                show_alert=True,
            )
            return
        item = await self.vision_service.get_item(owner_id, item_id)
        image = await self.vision_image_service.get(owner_id, item_id)
        if item is None:
            await self._vision_stale(query)
            return
        if action == "imageadd" and image is not None:
            await self._vision_stale(query)
            return
        if action == "imagereplace" and image is None:
            await self._vision_stale(query)
            return
        if action == "imagedeleteask":
            if image is None:
                await self._vision_stale(query)
                return
            token = await self.vision_image_sessions.issue_delete(
                owner_id,
                chat_id,
                item_id,
                expected_version=image.version,
            )
            if token is None:
                await query.answer(
                    "Сначала заверши или отмени текущую операцию с фото.",
                    show_alert=True,
                )
                return
            await query.answer()
            await self._vision_edit_or_send(
                query,
                query.message,
                "Удалить личное фото из этой карточки? Карточка желания останется.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Удалить фото",
                                callback_data=f"vision:imagedelete:{token}",
                            ),
                            InlineKeyboardButton(
                                "Отмена",
                                callback_data=f"vision:imagedeletecancel:{token}",
                            ),
                        ]
                    ]
                ),
            )
            return
        token = await self.vision_image_sessions.issue_upload(
            owner_id,
            chat_id,
            item_id,
            mode="add" if action == "imageadd" else "replace",
            expected_version=image.version if image is not None else None,
        )
        if token is None:
            await query.answer(
                "Сначала заверши или отмени текущую операцию с фото.",
                show_alert=True,
            )
            return
        await query.answer()
        await self._vision_edit_or_send(
            query,
            query.message,
            "Отправь одно фото или image-document в JPEG, PNG или WebP. "
            f"До {MAX_IMAGE_INPUT_BYTES // (1024 * 1024)} МБ и {MAX_IMAGE_PIXELS // 1_000_000} Мп. "
            "Оригинал и его metadata сохраняться не будут.",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Отмена",
                            callback_data=f"vision:imagecancel:{token}",
                        )
                    ]
                ]
            ),
        )

    async def _vision_image_generation_ask(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        item_id: int,
    ) -> None:
        if not self.image_generation.enabled:
            await query.answer(
                "Генерация пока не подключена администратором.",
                show_alert=True,
            )
            return
        if await self.vision_reference_sessions.has_active(owner_id, chat_id):
            await query.answer(
                "Сначала заверши или отмени текущую операцию с референсом.",
                show_alert=True,
            )
            return
        item = await self.vision_service.get_item(owner_id, item_id)
        if item is None:
            await self._vision_stale(query)
            return
        image = await self.vision_image_service.get(owner_id, item_id)
        _emoji, category = CATEGORY_META[item.category]
        prompt = build_vision_image_prompt(
            wish_text=item.wish_text,
            category=category,
        )
        token = await self.vision_image_sessions.issue_generation(
            owner_id,
            chat_id,
            item_id,
            mode="replace" if image is not None else "add",
            expected_version=image.version if image is not None else None,
            prompt=prompt,
        )
        if token is None:
            await query.answer(
                "Сначала заверши или отмени текущую операцию с изображением.",
                show_alert=True,
            )
            return
        references = await self.vision_reference_service.list(owner_id)
        rows = [
            [
                InlineKeyboardButton(
                    f"✨ Сгенерировать без референсов · {self.image_generation.model}",
                    callback_data=f"vision:imagegenerate:{token}",
                )
            ]
        ]
        if references:
            rows.append(
                [
                    InlineKeyboardButton(
                        f"🧩 Выбрать референсы · {len(references)}",
                        callback_data=f"vision:genrefs:{token}",
                    )
                ]
            )
        rows.append([InlineKeyboardButton("Отмена", callback_data=f"vision:imagecancel:{token}")])
        await query.answer()
        await self._vision_edit_or_send(
            query,
            query.message,
            "Перед отправкой через OpenRouter к модели OpenAI проверь запрос. В него входят "
            "только желание и категория; остальные поля карточки не передаются. Запрос "
            "станет платным только после нажатия кнопки генерации. Сейчас референсы не "
            "выбраны.\n\n"
            f"Модель: {self.image_generation.model}\n"
            f"Размер: {self.image_generation.size}\n"
            f"Качество: {self.image_generation.quality}\n\n"
            f"Точный запрос:\n{prompt}",
            InlineKeyboardMarkup(rows),
        )

    async def _vision_generation_references_begin(
        self, query: Any, owner_id: int, chat_id: int, token: str
    ) -> None:
        capability = await self.vision_image_sessions.begin_reference_selection(
            token, owner_id, chat_id
        )
        if capability is None:
            await self._vision_stale(query)
            return
        await query.answer()
        await self._vision_generation_reference_picker(
            query, owner_id, chat_id, token, capability.reference_ids
        )

    async def _vision_generation_reference_toggle(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        token: str,
        reference_id: int,
    ) -> None:
        current = await self.vision_image_sessions.reference_selection(token, owner_id, chat_id)
        reference = await self.vision_reference_service.get(owner_id, reference_id)
        if current is None or reference is None:
            await self._vision_stale(query)
            return
        if (
            reference_id not in current.reference_ids
            and len(current.reference_ids) >= MAX_GENERATION_REFERENCES
        ):
            await query.answer(
                f"Можно выбрать до {MAX_GENERATION_REFERENCES} референсов.",
                show_alert=True,
            )
            return
        capability = await self.vision_image_sessions.toggle_reference(
            token, owner_id, chat_id, reference_id
        )
        if capability is None:
            await self._vision_stale(query)
            return
        await query.answer()
        await self._vision_generation_reference_picker(
            query, owner_id, chat_id, token, capability.reference_ids
        )

    async def _vision_generation_reference_picker(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        token: str,
        selected_ids: tuple[int, ...],
    ) -> None:
        del chat_id
        references = await self.vision_reference_service.list(owner_id)
        if not references:
            await query.edit_message_text(
                "В библиотеке больше нет доступных референсов.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Продолжить без референсов",
                                callback_data=f"vision:genrefdone:{token}",
                            )
                        ]
                    ]
                ),
            )
            return
        rows = [
            [
                InlineKeyboardButton(
                    (
                        f"{'✅' if reference.id in selected_ids else '▫️'} "
                        f"{REFERENCE_KINDS[reference.kind]} · {reference.name}"
                    )[:60],
                    callback_data=f"vision:genreftoggle:{token}:{reference.id}",
                )
            ]
            for reference in references
        ]
        rows.extend(
            [
                [
                    InlineKeyboardButton(
                        f"Готово · выбрано {len(selected_ids)}",
                        callback_data=f"vision:genrefdone:{token}",
                    )
                ],
                [InlineKeyboardButton("Отмена", callback_data=f"vision:imagecancel:{token}")],
            ]
        )
        await query.edit_message_text(
            f"Выбери до {MAX_GENERATION_REFERENCES} референсов. "
            "Только отмеченные изображения уйдут во внешний запрос.",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _vision_generation_references_done(
        self, query: Any, owner_id: int, chat_id: int, token: str
    ) -> None:
        capability = await self.vision_image_sessions.reference_selection(token, owner_id, chat_id)
        if capability is None:
            await self._vision_stale(query)
            return
        references = await self.vision_reference_service.get_many(
            owner_id, capability.reference_ids
        )
        if len(references) != len(capability.reference_ids):
            await self._vision_stale(query)
            return
        item = await self.vision_service.get_item(owner_id, capability.item_id)
        if item is None:
            await self._vision_stale(query)
            return
        _emoji, category = CATEGORY_META[item.category]
        prompt = build_vision_image_prompt(
            wish_text=item.wish_text,
            category=category,
            references=[(reference.kind, reference.name) for reference in references],
        )
        finished = await self.vision_image_sessions.finish_reference_selection(
            token, owner_id, chat_id, prompt=prompt
        )
        if finished is None:
            await self._vision_stale(query)
            return
        selected_text = (
            "\n".join(
                f"• {REFERENCE_KINDS[reference.kind]} · {reference.name}"
                for reference in references
            )
            if references
            else "нет"
        )
        await query.answer()
        await query.edit_message_text(
            "Перед платной генерацией проверь внешний запрос. Через OpenRouter к модели "
            "OpenAI будут отправлены желание, категория и только выбранные изображения.\n\n"
            f"Модель: {self.image_generation.model}\n"
            f"Размер: {self.image_generation.size}\n"
            f"Качество: {self.image_generation.quality}\n\n"
            f"Выбранные референсы:\n{selected_text}\n\n"
            f"Точный запрос:\n{prompt}",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"✨ Сгенерировать · {self.image_generation.model}",
                            callback_data=f"vision:imagegenerate:{token}",
                        )
                    ],
                    [InlineKeyboardButton("Отмена", callback_data=f"vision:imagecancel:{token}")],
                ]
            ),
        )

    async def _vision_image_generate(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        token: str,
    ) -> None:
        capability = await self.vision_image_sessions.claim_generation(
            token,
            owner_id,
            chat_id,
        )
        if capability is None or capability.prompt is None:
            await self._vision_stale(query)
            return
        await query.answer()
        await query.edit_message_text(
            f"Создаю изображение через {self.image_generation.model}. Это может занять до двух минут…"
        )
        try:
            references = await self.vision_reference_service.get_many(
                owner_id, capability.reference_ids
            )
            if len(references) != len(capability.reference_ids):
                await self.vision_image_sessions.cancel(token, owner_id, chat_id)
                await self._vision_edit_or_send(
                    query,
                    query.message,
                    "Один из выбранных референсов был удалён. Ничего не отправлено; "
                    "открой генерацию заново.",
                    InlineKeyboardMarkup(
                        [[InlineKeyboardButton("← Меню карты", callback_data="vision:menu")]]
                    ),
                )
                return
            raw = await self.image_generation.generate(
                capability.prompt,
                references=[
                    ImageReferenceInput(
                        image_bytes=reference.image_bytes,
                        mime_type=reference.mime_type,
                    )
                    for reference in references
                ],
            )
            normalized = await asyncio.to_thread(
                normalize_vision_image,
                raw,
                declared_mime="image/png",
            )
            attached = await self.vision_image_sessions.attach_preview(
                token,
                owner_id,
                chat_id,
                normalized,
            )
            if not attached:
                await self._vision_edit_or_send(
                    query,
                    query.message,
                    "Превью не удалось удержать во временной памяти. Запусти генерацию ещё раз.",
                    InlineKeyboardMarkup(
                        [[InlineKeyboardButton("← Меню карты", callback_data="vision:menu")]]
                    ),
                )
                return
            stream = BytesIO(normalized.image_bytes)
            stream.name = "vision-ai-preview.jpg"
            try:
                await query.message.reply_photo(
                    photo=stream,
                    caption=(
                        f"Превью от {self.image_generation.model}. "
                        f"Использовано референсов: {len(references)}. "
                        "Изображение ещё не сохранено в карточке."
                    ),
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "Сохранить в карточку",
                                    callback_data=f"vision:imageconfirm:{token}",
                                ),
                                InlineKeyboardButton(
                                    "Не сохранять",
                                    callback_data=f"vision:imagecancel:{token}",
                                ),
                            ]
                        ]
                    ),
                )
            finally:
                stream.close()
        except ImageGenerationError as exc:
            await self.vision_image_sessions.cancel(token, owner_id, chat_id)
            logger.error("Vision image generation failed error_code=%s", exc.code)
            await self._vision_edit_or_send(
                query,
                query.message,
                self._vision_image_generation_error(exc.code),
                InlineKeyboardMarkup(
                    [[InlineKeyboardButton("← Меню карты", callback_data="vision:menu")]]
                ),
            )
        except VisionImageError:
            await self.vision_image_sessions.cancel(token, owner_id, chat_id)
            logger.error("Vision image generation returned unsafe image")
            await self._vision_edit_or_send(
                query,
                query.message,
                "Провайдер вернул изображение, которое не прошло безопасную обработку. "
                "Ничего не сохранено; можно попробовать ещё раз.",
                InlineKeyboardMarkup(
                    [[InlineKeyboardButton("← Меню карты", callback_data="vision:menu")]]
                ),
            )
        except TelegramError as exc:
            await self.vision_image_sessions.cancel(token, owner_id, chat_id)
            logger.error(
                "Vision generated image transport failed error_type=%s", type(exc).__name__
            )
            await self._vision_edit_or_send(
                query,
                query.message,
                "Изображение создано, но Telegram не принял превью. Ничего не сохранено.",
                InlineKeyboardMarkup(
                    [[InlineKeyboardButton("← Меню карты", callback_data="vision:menu")]]
                ),
            )

    @staticmethod
    def _vision_image_generation_error(code: str) -> str:
        if code == "moderation_blocked":
            return (
                "OpenRouter не смог создать изображение для этого запроса из-за правил "
                "безопасности. Автоповтора не было; измени формулировку желания и попробуй снова."
            )
        if code in {"authentication", "rate_limit"}:
            return (
                "Сервис генерации сейчас недоступен из-за доступа или лимита. "
                "Ничего не сохранено; попробуй позже."
            )
        if code in {"timeout", "connection"}:
            return (
                "OpenRouter не успел вернуть изображение или связь прервалась. "
                "Ничего не сохранено; повтор можно запустить вручную."
            )
        if code == "invalid_reference":
            return (
                "OpenRouter не принял один из выбранных референсов. Ничего не сохранено; "
                "проверь библиотеку или выбери другой набор."
            )
        return "Не удалось создать безопасное превью. Ничего не сохранено; попробуй ещё раз."

    async def _vision_image_capability_action(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        action: str,
        token: str,
    ) -> None:
        if action in {"imagecancel", "imagedeletecancel"}:
            if not await self.vision_image_sessions.cancel(token, owner_id, chat_id):
                await self._vision_stale(query)
                return
            await query.answer()
            await query.edit_message_text(
                "Действие с изображением отменено."
                if action == "imagecancel"
                else "Удаление фото отменено."
            )
            return
        if action == "imageconfirm":
            capability = await self.vision_image_sessions.claim_confirm(
                token,
                owner_id,
                chat_id,
            )
            if capability is None or capability.image is None:
                await self._vision_stale(query)
                return
            result = await self.vision_image_service.save(
                owner_id,
                capability.item_id,
                expected_version=capability.expected_version,
                normalized=capability.image,
            )
            if result.status not in {"created", "replaced", "existing"}:
                await self._vision_stale(query)
                return
            await query.answer()
            notice = (
                "Фото уже было сохранено; дубль не создан."
                if result.status == "existing"
                else "Фото сохранено в карточке."
            )
            item = await self.vision_service.get_item(owner_id, capability.item_id)
            if item is not None:
                await self._vision_send_item(
                    query.message,
                    item,
                    query=query,
                    include_image=False,
                    notice=f"{notice}\n\nЛичное фото этой карточки.",
                )
            return
        capability = await self.vision_image_sessions.claim_delete(
            token,
            owner_id,
            chat_id,
        )
        if capability is None or capability.expected_version is None:
            await self._vision_stale(query)
            return
        result = await self.vision_image_service.delete(
            owner_id,
            capability.item_id,
            expected_version=capability.expected_version,
        )
        if result.status != "deleted":
            await self._vision_stale(query)
            return
        await query.answer()
        item = await self.vision_service.get_item(owner_id, capability.item_id)
        if item is not None:
            await self._vision_send_item(
                query.message,
                item,
                query=query,
                include_image=False,
                notice="Фото удалено. Карточка желания сохранена.",
            )

    async def _vision_image_input(self, update: Update, user: Any) -> None:
        message = update.effective_message
        capability = await self.vision_image_sessions.claim_upload(
            user.id,
            update.effective_chat.id,
        )
        if capability is None:
            await message.reply_text("Фото уже обрабатывается. Дождись preview.")
            return
        try:
            media, metadata = self._vision_telegram_image(message)
            validate_telegram_metadata(metadata)
            telegram_file = await media.get_file()
            raw = bytes(await telegram_file.download_as_bytearray())
            normalized = await asyncio.to_thread(
                normalize_vision_image,
                raw,
                declared_mime=metadata.mime_type,
            )
            if not await self.vision_image_sessions.attach_preview(
                capability.token,
                user.id,
                update.effective_chat.id,
                normalized,
            ):
                await message.reply_text(
                    "Не удалось подготовить preview: лимит временной памяти исчерпан."
                )
                return
            stream = BytesIO(normalized.image_bytes)
            stream.name = "vision-photo-preview.jpg"
            try:
                await message.reply_photo(
                    photo=stream,
                    caption=(
                        f"Preview: {normalized.width}×{normalized.height}, "
                        f"до {MAX_IMAGE_OUTPUT_BYTES // 1024} КБ. Сохранить это фото?"
                    ),
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "Сохранить",
                                    callback_data=f"vision:imageconfirm:{capability.token}",
                                ),
                                InlineKeyboardButton(
                                    "Отмена",
                                    callback_data=f"vision:imagecancel:{capability.token}",
                                ),
                            ]
                        ]
                    ),
                )
            finally:
                stream.close()
        except VisionImageError:
            await self.vision_image_sessions.retry_upload(
                capability.token,
                user.id,
                update.effective_chat.id,
            )
            await message.reply_text(
                "Файл отклонён. Нужен статический JPEG, PNG или WebP без повреждений, "
                f"не больше {MAX_IMAGE_INPUT_BYTES // (1024 * 1024)} МБ и "
                f"{MAX_IMAGE_PIXELS // 1_000_000} Мп. GIF, SVG, PDF и HEIC не поддерживаются."
            )
        except TelegramError as exc:
            await self.vision_image_sessions.cancel(
                capability.token,
                user.id,
                update.effective_chat.id,
            )
            logger.error("Vision image transport failed error_type=%s", type(exc).__name__)
            await message.reply_text(
                "Не удалось безопасно загрузить фото. Открой карточку и попробуй ещё раз."
            )
        except Exception as exc:
            await self.vision_image_sessions.cancel(
                capability.token,
                user.id,
                update.effective_chat.id,
            )
            logger.error("Vision image processing failed error_type=%s", type(exc).__name__)
            await message.reply_text(
                "Не удалось обработать фото. Открой карточку и попробуй ещё раз."
            )

    @staticmethod
    def _vision_telegram_image(message: Any) -> tuple[Any, TelegramImageMetadata]:
        photos = list(message.photo or [])
        if photos:
            media = photos[-1]
            return media, TelegramImageMetadata(
                source="photo",
                file_size=getattr(media, "file_size", None),
                mime_type=getattr(media, "mime_type", None),
                width=getattr(media, "width", None),
                height=getattr(media, "height", None),
            )
        document = message.document
        if document is None:
            raise VisionImageError("unsupported_source")
        return document, TelegramImageMetadata(
            source="document",
            file_size=getattr(document, "file_size", None),
            mime_type=getattr(document, "mime_type", None),
        )

    async def _vision_send_page(
        self,
        message: Any,
        owner_id: int,
        status: str,
        page: int,
        *,
        query: Any | None = None,
    ) -> None:
        items, total = await self.vision_service.page(owner_id, status, page)
        counts = await self.vision_service.category_counts(owner_id, status)
        title = {
            "active": "Моя карта",
            "achieved": "Достигнуто",
            "archived": "Архив",
        }[status]
        if not items:
            rows = []
            if status == "active":
                rows.append(
                    [InlineKeyboardButton("➕ Добавить желание", callback_data="vision:add")]
                )
            rows.extend(self._vision_list_navigation(status))
            await self._vision_edit_or_send(
                query,
                message,
                f"{title}: карточек пока нет.",
                InlineKeyboardMarkup(rows),
            )
            return
        lines = [f"{title} — {total}"]
        current_category = None
        rows = []
        for item in items:
            if item.category != current_category:
                current_category = item.category
                emoji, label = CATEGORY_META[item.category]
                lines.append(f"\n{emoji} {label} ({counts.get(item.category, 0)})")
            lines.append(f"• #{item.id} {item.wish_text[:90]}")
            rows.append(
                [
                    InlineKeyboardButton(
                        f"Открыть #{item.id}", callback_data=f"vision:view:{item.id}"
                    )
                ]
            )
        navigation = []
        if page > 0:
            navigation.append(
                InlineKeyboardButton("←", callback_data=f"vision:list:{status}:{page - 1}")
            )
        if (page + 1) * PAGE_SIZE < total:
            navigation.append(
                InlineKeyboardButton("→", callback_data=f"vision:list:{status}:{page + 1}")
            )
        if navigation:
            rows.append(navigation)
        rows.extend(self._vision_list_navigation(status))
        await self._vision_edit_or_send(
            query,
            message,
            "\n".join(lines),
            InlineKeyboardMarkup(rows),
        )

    async def _vision_companion_settings(self, query: Any, user: Any, item_id: int) -> None:
        item = await self.vision_service.get_item(user.id, item_id)
        if item is None or item.status != "active":
            await self._vision_stale(query)
            return
        preference = await self.vision_companion_service.get(user.id)
        await query.answer()
        if preference is None or not preference.enabled:
            await self._vision_edit_or_send(
                query,
                query.message,
                "Включить добровольное сопровождение этой карточки?\n\n"
                f"Утром в {self.settings.morning_hour:02d}:00 бот спросит о шаге, "
                f"вечером в {self.settings.evening_hour:02d}:00 — как прошёл день. "
                "Дополнительные напоминания изначально выключены. Одновременно "
                "сопровождается только одно желание, новую картинку бот не генерирует.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Включить", callback_data=f"vision:companionon:{item_id}"
                            ),
                            InlineKeyboardButton(
                                "Не сейчас", callback_data=f"vision:companioncancel:{item_id}"
                            ),
                        ]
                    ]
                ),
            )
            return
        if preference.vision_item_id != item_id:
            await self._vision_edit_or_send(
                query,
                query.message,
                "Сейчас сопровождается другая карточка. Переключить фокус на это желание? "
                "История прежних отметок сохранится.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Переключить", callback_data=f"vision:companionon:{item_id}"
                            ),
                            InlineKeyboardButton(
                                "Оставить как есть",
                                callback_data=f"vision:companioncancel:{item_id}",
                            ),
                        ]
                    ]
                ),
            )
            return
        await self._vision_companion_send_controls(query.message, preference, query=query)

    async def _vision_companion_enable(
        self, query: Any, user: Any, chat_id: int, item_id: int
    ) -> None:
        preference = await self.vision_companion_service.enable(
            owner_id=user.id,
            item_id=item_id,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            timezone=user.timezone,
            morning_time=datetime.min.time().replace(hour=self.settings.morning_hour),
            evening_time=datetime.min.time().replace(hour=self.settings.evening_hour),
        )
        if preference is None:
            await self._vision_stale(query)
            return
        if self.scheduler:
            self.scheduler.schedule_vision_companion(preference)
        await query.answer("Сопровождение включено")
        await self._vision_companion_send_controls(query.message, preference, query=query)

    async def _vision_companion_frequency(
        self, query: Any, owner_id: int, item_id: int, count: int
    ) -> None:
        if count not in {0, 1, 2, 3}:
            await self._vision_stale(query)
            return
        preference = await self.vision_companion_service.set_frequency(owner_id, item_id, count)
        if preference is None:
            await self._vision_stale(query)
            return
        if self.scheduler:
            self.scheduler.schedule_vision_companion(preference)
        await query.answer("Частота обновлена")
        await self._vision_companion_send_controls(query.message, preference, query=query)

    async def _vision_companion_disable(self, query: Any, user: Any, item_id: int) -> None:
        disabled = await self.vision_companion_service.disable(user.id, item_id)
        if self.scheduler:
            self.scheduler.remove_vision_companion(user.id)
            self.scheduler.schedule_user(user.telegram_id, user.timezone)
        await query.answer()
        await query.edit_message_text(
            "Сопровождение отключено. История отметок сохранена."
            if disabled
            else "Сопровождение уже было отключено."
        )

    async def _vision_companion_send_controls(
        self, message: Any, preference: Any, *, query: Any | None = None
    ) -> None:
        extras = ", ".join(preference.extra_times) if preference.extra_times else "выключены"
        await self._vision_edit_or_send(
            query,
            message,
            "Сопровождение включено.\n\n"
            f"Утренний вопрос: {preference.morning_time.strftime('%H:%M')}\n"
            f"Вечерний вопрос: {preference.evening_time.strftime('%H:%M')}\n"
            f"Дополнительные напоминания: {extras}\n"
            f"Часовой пояс: {preference.timezone}\n\n"
            "Выбери количество дополнительных мягких напоминаний между утром и вечером:",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Без дополнительных",
                            callback_data=(f"vision:companionfreq:{preference.vision_item_id}:0"),
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "1 раз",
                            callback_data=f"vision:companionfreq:{preference.vision_item_id}:1",
                        ),
                        InlineKeyboardButton(
                            "2 раза",
                            callback_data=f"vision:companionfreq:{preference.vision_item_id}:2",
                        ),
                        InlineKeyboardButton(
                            "3 раза",
                            callback_data=f"vision:companionfreq:{preference.vision_item_id}:3",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "Отключить сопровождение",
                            callback_data=f"vision:companionoff:{preference.vision_item_id}",
                        )
                    ],
                ]
            ),
        )

    async def _vision_companion_notification(
        self, bot: Any, preference_id: int, moment: str
    ) -> None:
        snapshot = await self.vision_companion_service.snapshot(preference_id)
        if snapshot is None:
            return
        if moment == "morning":
            title = "🌅 Утренний фокус карты"
            question = "Готов взять этот шаг сегодня?"
            buttons = [
                ("Беру шаг", "morning", "committed"),
                ("Сегодня пауза", "morning", "pause"),
            ]
        elif moment == "evening":
            title = "🌙 Вечерняя сверка"
            question = "Как сегодня получилось приблизиться к желанию?"
            buttons = [
                ("Сделал", "evening", "done"),
                ("Немного", "evening", "partial"),
                ("Не сегодня", "evening", "missed"),
            ]
        else:
            title = "🔔 Мягкое напоминание"
            question = "Есть возможность сделать маленький шаг сейчас?"
            buttons = [
                ("Шаг сделан", "extra", "done"),
                ("Вернусь позже", "extra", "later"),
            ]
        why = f"\nЗачем: {snapshot.why_text}" if snapshot.why_text else ""
        step = snapshot.first_step or "выбери один небольшой достижимый шаг"
        caption = (
            f"{title}\n\nЖелание: {snapshot.wish_text}{why}\nТекущий шаг: {step}\n\n{question}"
        )[:1000]
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        label,
                        callback_data=(
                            f"vision:companionlog:{snapshot.item_id}:{button_moment}:{response}"
                        ),
                    )
                    for label, button_moment, response in buttons
                ]
            ]
        )
        if snapshot.image_bytes is not None:
            stream = BytesIO(snapshot.image_bytes)
            stream.name = "vision-focus.jpg"
            try:
                await bot.send_photo(
                    chat_id=snapshot.chat_id,
                    photo=stream,
                    caption=caption,
                    reply_markup=markup,
                )
            finally:
                stream.close()
        else:
            await bot.send_message(chat_id=snapshot.chat_id, text=caption, reply_markup=markup)

    async def _vision_companion_record(
        self,
        query: Any,
        owner_id: int,
        item_id: int,
        moment: str,
        response: str,
    ) -> None:
        checkin = await self.vision_companion_service.record(
            owner_id=owner_id,
            item_id=item_id,
            moment=moment,
            response=response,
        )
        if checkin is None:
            await self._vision_stale(query)
            return
        replies = {
            "committed": "Шаг принят. Пусть он будет небольшим и реальным.",
            "pause": "Пауза принята без давления. Вечером можно спокойно свериться.",
            "done": "Отмечено. Маленький шаг тоже считается.",
            "partial": "Отмечено: немного — это уже движение.",
            "missed": "Отмечено без осуждения. Завтра можно уменьшить шаг.",
            "later": "Хорошо, вернёмся к нему позже.",
        }
        await query.answer(replies[response], show_alert=True)
        await query.edit_message_reply_markup(reply_markup=None)

    async def _vision_send_item(
        self,
        message: Any,
        item: Any,
        *,
        query: Any | None = None,
        include_image: bool = True,
        notice: str | None = None,
    ) -> None:
        emoji, category = CATEGORY_META[item.category]
        status = {
            "active": "активно",
            "achieved": "достигнуто",
            "archived": "в архиве",
        }[item.status]
        image = await self.vision_image_service.get(item.owner_id, item.id)
        if image is not None and include_image:
            stream = BytesIO(image.image_bytes)
            stream.name = "vision-photo.jpg"
            try:
                await message.reply_photo(
                    photo=stream,
                    caption="Личное фото этой карточки.",
                )
            finally:
                stream.close()
        image_rows = (
            [
                [
                    InlineKeyboardButton(
                        "Заменить изображение",
                        callback_data=f"vision:imagereplace:{item.id}",
                    ),
                    InlineKeyboardButton(
                        "Удалить изображение",
                        callback_data=f"vision:imagedeleteask:{item.id}",
                    ),
                ]
            ]
            if image is not None
            else [
                [
                    InlineKeyboardButton(
                        "📷 Добавить фото",
                        callback_data=f"vision:imageadd:{item.id}",
                    )
                ]
            ]
        )
        if self.image_generation.enabled:
            image_rows.append(
                [
                    InlineKeyboardButton(
                        (
                            f"✨ Создать новое · {self.image_generation.model}"
                            if image is not None
                            else f"✨ Создать с AI · {self.image_generation.model}"
                        ),
                        callback_data=f"vision:imagegenerateask:{item.id}",
                    )
                ]
            )
        image_rows.append([InlineKeyboardButton("🧩 Мои референсы", callback_data="vision:refs")])
        card_text = (
            f"{emoji} #{item.id} · {category} · {status}\n\n"
            f"Желание: {item.wish_text}\n"
            f"Почему важно: {item.why_text or 'не указано'}\n"
            f"Желаемая дата: "
            f"{item.target_date.strftime('%d.%m.%Y') if item.target_date else 'не указана'}\n"
            f"Первый шаг: {item.first_step or 'не указан'}"
        )
        if notice:
            card_text = f"{notice}\n\n{card_text}"
        await self._vision_edit_or_send(
            query,
            message,
            card_text,
            InlineKeyboardMarkup(
                image_rows
                + (
                    [
                        [
                            InlineKeyboardButton(
                                "🔔 Сопровождение",
                                callback_data=f"vision:companion:{item.id}",
                            )
                        ]
                    ]
                    if item.status == "active"
                    else []
                )
                + [
                    [
                        InlineKeyboardButton(
                            "Редактировать", callback_data=f"vision:edit:{item.id}"
                        ),
                        InlineKeyboardButton(
                            "✅ Достигнуто" if item.status == "active" else "↩️ Активно",
                            callback_data=(
                                f"vision:status:{item.id}:achieved"
                                if item.status == "active"
                                else f"vision:status:{item.id}:active"
                            ),
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "Создать задачу", callback_data=f"vision:task:{item.id}"
                        ),
                        InlineKeyboardButton(
                            "Архивировать" if item.status != "archived" else "Открыть архив",
                            callback_data=(
                                f"vision:archive:{item.id}"
                                if item.status != "archived"
                                else "vision:list:archived:0"
                            ),
                        ),
                    ],
                    [InlineKeyboardButton("Удалить", callback_data=f"vision:deleteask:{item.id}")],
                    [
                        InlineKeyboardButton(
                            "← К списку",
                            callback_data=f"vision:list:{item.status}:0",
                        ),
                        InlineKeyboardButton("Меню карты", callback_data="vision:menu"),
                    ],
                ]
            ),
        )

    @staticmethod
    async def _vision_stale(query: Any) -> None:
        await query.answer("Карточка недоступна или действие устарело.", show_alert=True)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except TelegramError:
            pass
