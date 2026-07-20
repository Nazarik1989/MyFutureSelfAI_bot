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

from .vision import CATEGORY_META, PAGE_SIZE
from .vision_renderer import MAX_RENDER_ITEMS, VisionRenderItem

logger = logging.getLogger(__name__)


class VisionHandlers:
    """Telegram presentation layer for the persistent owner-scoped vision service."""

    vision_service: Any
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
    async def _vision_menu(message: Any) -> None:
        await message.reply_text(
            "Карта желаний\n\n"
            "Желание → желаемый результат → зачем это важно → первый шаг → задача.",
            reply_markup=InlineKeyboardMarkup(
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
                ]
            ),
        )

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

    async def _vision_prompt(self, message: Any, draft: Any) -> None:
        if draft.step == "category":
            await message.reply_text(
                "Выбери категорию желания:",
                reply_markup=self._vision_category_keyboard(draft),
            )
            return
        if draft.step == "edit_value" and draft.edit_field == "category":
            await message.reply_text(
                "Выбери новую категорию:",
                reply_markup=self._vision_category_keyboard(draft, edit=True),
            )
            return
        if draft.step == "delete_confirm":
            await message.reply_text(
                "Удаление ожидает явного подтверждения.",
                reply_markup=InlineKeyboardMarkup(
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
            await message.reply_text(
                prompts[draft.step],
                reply_markup=InlineKeyboardMarkup(rows),
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
            await message.reply_text(
                f"Пришли {field_name} текстом или голосом.",
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return
        if draft.step == "preview":
            await message.reply_text(
                self._vision_preview_text(draft),
                reply_markup=InlineKeyboardMarkup(
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
            await self._vision_prompt(query.message, draft)
            return
        if action == "menu" and len(parts) == 2:
            await query.answer()
            await self._vision_menu(query.message)
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
                await query.edit_message_reply_markup(reply_markup=None)
            except TelegramError:
                pass
            await self._vision_render_and_send(
                query.message,
                user,
                None if selection == "all" else selection,
                token=token,
                as_document=False,
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
            )
            return
        if action == "rendercancel" and len(parts) == 3:
            if not await self.vision_render_sessions.cancel(parts[2], user.id, chat_id):
                await self._vision_render_stale(query)
                return
            await query.answer()
            await query.edit_message_text("Визуализация отменена.")
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
            await self._vision_send_page(query.message, user.id, parts[2], page)
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
            await self._vision_prompt(query.message, outcome.draft)
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
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("Категория обновлена.")
            await self._vision_send_item(query.message, outcome.item)
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
            await query.edit_message_reply_markup(reply_markup=None)
            if outcome.status == "edited":
                await query.message.reply_text("Поле очищено.")
                await self._vision_send_item(query.message, outcome.item)
            else:
                await self._vision_prompt(query.message, outcome.draft)
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
            await self._vision_send_item(query.message, outcome.item)
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
            await self.vision_service.cancel(user.id, chat_id)
            await query.answer()
            await query.edit_message_text("Создание или редактирование отменено.")
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
            await query.edit_message_text(
                "Карточка удалена." if outcome.status == "deleted" else "Удаление отменено."
            )
            if outcome.status == "cancelled":
                await self._vision_send_item(query.message, outcome.item)
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
            await self._vision_send_item(query.message, item)
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
            await self._vision_prompt(query.message, outcome.draft)
            return
        await self._vision_stale(query)

    async def _vision_render_menu(self, query: Any, owner_id: int, chat_id: int) -> None:
        counts = await self.vision_service.category_counts(owner_id, "active")
        available = {category for category, count in counts.items() if count > 0}
        await query.answer()
        if not available:
            await query.message.reply_text(
                "Активных желаний пока нет. Сначала добавь желание через /vision."
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
        await query.message.reply_text(
            "Что визуализировать? В изображение попадут только активные желания.",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _vision_render_and_send(
        self,
        message: Any,
        user: Any,
        category: str | None,
        *,
        token: str,
        as_document: bool,
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
                                        )
                                    ]
                                ]
                            )
                        await message.reply_photo(
                            photo=stream,
                            caption=caption,
                            reply_markup=reply_markup,
                        )
                finally:
                    stream.close()
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
            await self._vision_send_item(query.message, item)
            return
        if action == "edit":
            await query.answer()
            await query.message.reply_text(
                "Что изменить?",
                reply_markup=InlineKeyboardMarkup(
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
                    ]
                ),
            )
            return
        if action == "archive":
            updated = await self.vision_service.set_status(owner_id, item.id, "archived")
            await query.answer()
            await query.edit_message_text(
                "Карточка архивирована." if updated is not None else "Карточка недоступна."
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
            await self._vision_prompt(query.message, outcome.draft)
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

    async def _vision_send_page(self, message: Any, owner_id: int, status: str, page: int) -> None:
        items, total = await self.vision_service.page(owner_id, status, page)
        counts = await self.vision_service.category_counts(owner_id, status)
        title = {
            "active": "Моя карта",
            "achieved": "Достигнуто",
            "archived": "Архив",
        }[status]
        if not items:
            rows = [
                [
                    InlineKeyboardButton("Добавить желание", callback_data="vision:add"),
                    InlineKeyboardButton("Меню", callback_data="vision:menu"),
                ]
            ]
            if status != "archived":
                rows.append([InlineKeyboardButton("Архив", callback_data="vision:list:archived:0")])
            await message.reply_text(
                f"{title}: карточек пока нет.",
                reply_markup=InlineKeyboardMarkup(rows),
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
        if status != "archived":
            rows.append([InlineKeyboardButton("Архив", callback_data="vision:list:archived:0")])
        rows.append([InlineKeyboardButton("Меню", callback_data="vision:menu")])
        await message.reply_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(rows))

    async def _vision_send_item(self, message: Any, item: Any) -> None:
        emoji, category = CATEGORY_META[item.category]
        status = {
            "active": "активно",
            "achieved": "достигнуто",
            "archived": "в архиве",
        }[item.status]
        await message.reply_text(
            f"{emoji} #{item.id} · {category} · {status}\n\n"
            f"Желание: {item.wish_text}\n"
            f"Почему важно: {item.why_text or 'не указано'}\n"
            f"Желаемая дата: "
            f"{item.target_date.strftime('%d.%m.%Y') if item.target_date else 'не указана'}\n"
            f"Первый шаг: {item.first_step or 'не указан'}",
            reply_markup=InlineKeyboardMarkup(
                [
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
