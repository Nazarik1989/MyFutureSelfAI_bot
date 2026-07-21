from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from io import BytesIO
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ApplicationHandlerStop, ContextTypes

from .lab_media import (
    MAX_LAB_INPUT_BYTES,
    MAX_PDF_PAGES,
    PDF_RENDER_TIMEOUT_SECONDS,
    LabMediaError,
    TelegramLabMetadata,
    process_lab_upload,
    validate_telegram_lab_metadata,
)
from .labs import LAB_LIST_PAGE_SIZE, LabDraftSnapshot

logger = logging.getLogger(__name__)

_DOCUMENT_EDIT_TTL_SECONDS = 10 * 60


class LabHandlers:
    lab_documents: Any
    lab_uploads: Any

    async def labs_command_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        flow = await self._active_navigation_flow(update, context)
        if flow is not None:
            await self._prompt_navigation_flow(update.effective_message, update, flow)
            raise ApplicationHandlerStop
        await self.labs_command(update, context)
        raise ApplicationHandlerStop

    async def labs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None) -> None:
        del context
        await update.effective_message.reply_text(
            "Анализы\n\nХрани фото и PDF результатов как очищенные локальные копии.",
            reply_markup=self._labs_menu_keyboard(),
        )

    async def labs_media_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        if await self.lab_uploads.has_active(user.id, update.effective_chat.id):
            await self._labs_media_input(update, user)
            raise ApplicationHandlerStop
        if self._owned_document_edit(context, user.id, update.effective_chat.id):
            await update.effective_message.reply_text(
                "Сейчас ожидается текст названия или даты. Для выхода отправь /cancel."
            )
            raise ApplicationHandlerStop
        await self.vision_image_gate(update, context)

    async def labs_voice_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        if await self.lab_uploads.has_active(user.id, update.effective_chat.id):
            await update.effective_message.reply_text(
                "Для анализов отправь фото, image-document или PDF либо нажми «Отмена»."
            )
            raise ApplicationHandlerStop
        if self._owned_document_edit(context, user.id, update.effective_chat.id):
            await update.effective_message.reply_text(
                "Сейчас ожидается текст названия или даты. Для выхода отправь /cancel."
            )
            raise ApplicationHandlerStop
        await self.vision_voice_gate(update, context)

    async def labs_cleanup_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        await self.lab_uploads.cleanup_expired()
        await self.lab_documents.cleanup_confirmations()

    async def labs_text_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        active = await self.lab_uploads.active(user.id, update.effective_chat.id)
        if active is not None:
            await self._labs_draft_text(update, active)
            raise ApplicationHandlerStop
        edit = context.user_data.get("lab_document_edit")
        if edit is not None:
            await self._labs_document_edit_text(update, context, user.id, edit)
            raise ApplicationHandlerStop
        await self.vision_text_gate(update, context)

    async def labs_cancel_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = await self._user(update.effective_user.id)
        if await self.lab_uploads.cancel_active(user.id, update.effective_chat.id):
            context.user_data.pop("lab_document_edit", None)
            await update.effective_message.reply_text(
                "Загрузка анализов отменена. Временные файлы удалены.",
                reply_markup=self._labs_menu_keyboard(),
            )
            raise ApplicationHandlerStop
        if context.user_data.pop("lab_document_edit", None) is not None:
            await update.effective_message.reply_text(
                "Изменение документа отменено.", reply_markup=self._labs_menu_keyboard()
            )
            raise ApplicationHandlerStop
        await self.vision_cancel_gate(update, context)

    async def labs_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = query.data or ""
        user = await self._user(update.effective_user.id)
        owner_id = user.id
        chat_id = update.effective_chat.id

        if data == "labs:menu":
            await query.answer()
            await self._labs_edit_or_send(
                query,
                "Анализы\n\nХрани фото и PDF результатов как очищенные локальные копии.",
                self._labs_menu_keyboard(),
            )
            return
        if data == "labs:help":
            await query.answer()
            await self._labs_edit_or_send(
                query,
                "Как это работает\n\n"
                "Бот локально проверяет файл, удаляет metadata и сохраняет только заново "
                "закодированные изображения страниц. OCR и медицинской интерпретации нет.\n\n"
                f"Лимиты: до {MAX_LAB_INPUT_BYTES // (1024 * 1024)} МБ, PDF до "
                f"{MAX_PDF_PAGES} страниц, обработка до {PDF_RENDER_TIMEOUT_SECONDS} секунд.",
                self._labs_back_keyboard(),
            )
            return
        if data == "labs:add":
            token = await self.lab_uploads.start(owner_id, chat_id)
            if token is None:
                await self._labs_stale(query)
                return
            context.user_data.pop("lab_document_edit", None)
            await query.answer()
            await query.message.reply_text(
                "Отправь Telegram photo, JPEG/PNG/static WebP как документ или обычный PDF. "
                "Файл не сохранится без отдельного подтверждения.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Отмена", callback_data=f"labs:cancel:{token}")]]
                ),
            )
            return
        if data.startswith("labs:cancel:"):
            token = data.removeprefix("labs:cancel:")
            if not await self.lab_uploads.cancel(token, owner_id, chat_id):
                await self._labs_stale(query)
                return
            await query.answer()
            await self._labs_edit_or_send(
                query,
                "Загрузка отменена. Временные файлы удалены.",
                self._labs_menu_keyboard(),
            )
            return
        if data.startswith("labs:draft:"):
            await self._labs_draft_action(update, user, context)
            return
        if data.startswith("labs:list:"):
            page = self._safe_int(data.removeprefix("labs:list:"))
            if page is None:
                await self._labs_stale(query)
                return
            await query.answer()
            await self._labs_send_list(query.message, owner_id, page)
            return
        if data.startswith("labs:open:"):
            document_id = self._safe_int(data.removeprefix("labs:open:"), positive=True)
            if document_id is None:
                await self._labs_stale(query)
                return
            document = await self.lab_documents.get(owner_id, document_id)
            if document is None:
                await self._labs_stale(query)
                return
            await query.answer()
            await self._labs_send_document(query.message, document)
            return
        if data.startswith("labs:view:"):
            parts = data.split(":")
            if len(parts) != 4:
                await self._labs_stale(query)
                return
            document_id = self._safe_int(parts[2], positive=True)
            page_index = self._safe_int(parts[3])
            if document_id is None or page_index is None:
                await self._labs_stale(query)
                return
            page = await self.lab_documents.get_page(owner_id, document_id, page_index)
            document = await self.lab_documents.get(owner_id, document_id)
            if page is None or document is None:
                await self._labs_stale(query)
                return
            await query.answer()
            await self._labs_send_page(query.message, document, page_index, page.image_bytes)
            return
        if data.startswith("labs:rename:") or data.startswith("labs:date:"):
            parts = data.split(":")
            if len(parts) != 4:
                await self._labs_stale(query)
                return
            document_id = self._safe_int(parts[2], positive=True)
            version = self._safe_int(parts[3], positive=True)
            field = "title" if parts[1] == "rename" else "date"
            document = (
                None if document_id is None else await self.lab_documents.get(owner_id, document_id)
            )
            if document is None or version is None or document.version != version:
                await self._labs_stale(query)
                return
            context.user_data["lab_document_edit"] = {
                "owner_id": owner_id,
                "chat_id": chat_id,
                "document_id": document.id,
                "version": document.version,
                "field": field,
                "expires_at": monotonic() + _DOCUMENT_EDIT_TTL_SECONDS,
            }
            await query.answer()
            prompt = (
                "Отправь новое название (до 200 символов)."
                if field == "title"
                else "Отправь дату в формате ДД.ММ.ГГГГ или «—», чтобы убрать дату."
            )
            await query.message.reply_text(prompt)
            return
        if data.startswith("labs:delete:"):
            document_id = self._safe_int(data.removeprefix("labs:delete:"), positive=True)
            token = (
                None
                if document_id is None
                else await self.lab_documents.issue_delete(owner_id, chat_id, document_id)
            )
            if token is None:
                await self._labs_stale(query)
                return
            await query.answer()
            await query.message.reply_text(
                "Удалить этот документ и все его страницы? Действие необратимо.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Удалить", callback_data=f"labs:deleteconfirm:{token}"
                            ),
                            InlineKeyboardButton("Отмена", callback_data="labs:menu"),
                        ]
                    ]
                ),
            )
            return
        if data.startswith("labs:deleteconfirm:"):
            token = data.removeprefix("labs:deleteconfirm:")
            if not await self.lab_documents.confirm_delete(token, owner_id, chat_id):
                await self._labs_stale(query)
                return
            await query.answer()
            await self._labs_edit_or_send(query, "Документ удалён.", self._labs_menu_keyboard())
            return
        await self._labs_stale(query)

    async def _labs_draft_action(
        self, update: Update, user: Any, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        del context
        query = update.callback_query
        parts = (query.data or "").split(":")
        if len(parts) != 4:
            await self._labs_stale(query)
            return
        action, token = parts[2], parts[3]
        owner_id = user.id
        chat_id = update.effective_chat.id
        if action in {"title", "date"}:
            snapshot = await self.lab_uploads.begin_edit(
                token, owner_id, chat_id, "title" if action == "title" else "date"
            )
            if snapshot is None:
                await self._labs_stale(query)
                return
            await query.answer()
            await query.message.reply_text(
                "Отправь название до 200 символов."
                if action == "title"
                else "Отправь дату в формате ДД.ММ.ГГГГ или «—», чтобы оставить без даты."
            )
            return
        if action == "save":
            capability = await self.lab_uploads.claim_confirm(token, owner_id, chat_id)
            if capability is None:
                await self._labs_stale(query)
                return
            try:
                document = await self.lab_documents.create(
                    owner_id,
                    capability.title,
                    capability.document_date,
                    capability.source_type,
                    capability.pages,
                )
            except Exception as exc:
                logger.error("Lab document save failed error_type=%s", type(exc).__name__)
                await query.answer("Не удалось сохранить документ.", show_alert=True)
            else:
                await query.answer()
                await self._labs_edit_or_send(
                    query,
                    f"Документ сохранён: {document.page_count} стр.",
                    self._labs_menu_keyboard(),
                )
            finally:
                await self.lab_uploads.finish(token, owner_id, chat_id)
            return
        await self._labs_stale(query)

    async def _labs_media_input(self, update: Update, user: Any) -> None:
        message = update.effective_message
        capability = await self.lab_uploads.claim_upload(user.id, update.effective_chat.id)
        if capability is None:
            await message.reply_text("Файл уже обрабатывается. Дождись preview.")
            return
        try:
            media, metadata = self._telegram_lab_media(message)
            validate_telegram_lab_metadata(metadata)
            telegram_file = await media.get_file()
            raw = bytes(await telegram_file.download_as_bytearray())
            processed = await asyncio.wait_for(
                asyncio.to_thread(
                    process_lab_upload,
                    raw,
                    metadata,
                    temp_root=self.lab_uploads.root,
                ),
                timeout=PDF_RENDER_TIMEOUT_SECONDS + 5,
            )
            local_date = datetime.now(ZoneInfo(user.timezone)).date()
            title = f"Результаты анализов от {local_date:%d.%m.%Y}"
            snapshot = await self.lab_uploads.attach(
                capability.token,
                user.id,
                update.effective_chat.id,
                processed,
                title=title,
            )
            if snapshot is None or snapshot.first_page is None:
                raise LabMediaError("temporary_storage_limit")
            await self._labs_send_draft_preview(message, snapshot)
        except (LabMediaError, TimeoutError):
            await self.lab_uploads.cancel(capability.token, user.id, update.effective_chat.id)
            await message.reply_text(
                "Файл отклонён и удалён. Поддерживаются статические JPEG/PNG/WebP и "
                f"безопасные PDF до {MAX_PDF_PAGES} страниц и "
                f"{MAX_LAB_INPUT_BYTES // (1024 * 1024)} МБ."
            )
        except TelegramError as exc:
            await self.lab_uploads.cancel(capability.token, user.id, update.effective_chat.id)
            logger.error("Lab upload transport failed error_type=%s", type(exc).__name__)
            await message.reply_text("Не удалось безопасно загрузить файл. Начни заново.")
        except Exception as exc:
            await self.lab_uploads.cancel(capability.token, user.id, update.effective_chat.id)
            logger.error("Lab media processing failed error_type=%s", type(exc).__name__)
            await message.reply_text("Не удалось обработать файл. Он удалён; начни заново.")

    async def _labs_draft_text(self, update: Update, active: LabDraftSnapshot) -> None:
        message = update.effective_message
        text = message.text or ""
        try:
            if active.stage == "edit_title":
                snapshot = await self.lab_uploads.apply_title(active.owner_id, active.chat_id, text)
            elif active.stage == "edit_date":
                snapshot = await self.lab_uploads.apply_date(
                    active.owner_id, active.chat_id, self._parse_document_date(text)
                )
            else:
                await message.reply_text(
                    "Отправь файл или используй кнопки preview. Для выхода нажми «Отмена»."
                )
                return
        except ValueError:
            await message.reply_text(
                "Не удалось принять значение. Название — до 200 символов; дата — ДД.ММ.ГГГГ "
                "или «—»."
            )
            return
        if snapshot is None or snapshot.first_page is None:
            await message.reply_text("Этот preview устарел. Открой /labs и начни заново.")
            return
        await self._labs_send_draft_preview(message, snapshot)

    async def _labs_document_edit_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        owner_id: int,
        edit: dict[str, Any],
    ) -> None:
        if (
            edit.get("owner_id") != owner_id
            or edit.get("chat_id") != update.effective_chat.id
            or float(edit.get("expires_at", 0)) <= monotonic()
        ):
            context.user_data.pop("lab_document_edit", None)
            await update.effective_message.reply_text("Изменение устарело. Открой документ заново.")
            return
        try:
            if edit["field"] == "title":
                changed = await self.lab_documents.rename(
                    owner_id,
                    int(edit["document_id"]),
                    int(edit["version"]),
                    update.effective_message.text or "",
                )
            else:
                changed = await self.lab_documents.set_date(
                    owner_id,
                    int(edit["document_id"]),
                    int(edit["version"]),
                    self._parse_document_date(update.effective_message.text or ""),
                )
        except (KeyError, TypeError, ValueError):
            await update.effective_message.reply_text(
                "Не удалось принять значение. Название — до 200 символов; дата — ДД.ММ.ГГГГ "
                "или «—»."
            )
            return
        context.user_data.pop("lab_document_edit", None)
        if not changed:
            await update.effective_message.reply_text("Документ уже изменился. Открой его заново.")
            return
        document = await self.lab_documents.get(owner_id, int(edit["document_id"]))
        if document is None:
            await update.effective_message.reply_text("Документ недоступен.")
            return
        await update.effective_message.reply_text("Изменение сохранено.")
        await self._labs_send_document(update.effective_message, document)

    async def _labs_send_draft_preview(self, message: Any, snapshot: LabDraftSnapshot) -> None:
        stream = BytesIO(snapshot.first_page or b"")
        stream.name = "lab-preview.jpg"
        try:
            await message.reply_photo(
                photo=stream,
                caption=(
                    f"Preview первой страницы · {snapshot.page_count} стр.\n"
                    f"Название: {snapshot.title}\n"
                    f"Дата документа: {self._format_date(snapshot.document_date)}\n\n"
                    "Сохранение произойдёт только после кнопки «Сохранить»."
                ),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Название",
                                callback_data=f"labs:draft:title:{snapshot.token}",
                            ),
                            InlineKeyboardButton(
                                "Дата", callback_data=f"labs:draft:date:{snapshot.token}"
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Сохранить",
                                callback_data=f"labs:draft:save:{snapshot.token}",
                            ),
                            InlineKeyboardButton(
                                "Отмена", callback_data=f"labs:cancel:{snapshot.token}"
                            ),
                        ],
                    ]
                ),
            )
        finally:
            stream.close()

    async def _labs_send_list(self, message: Any, owner_id: int, page: int) -> None:
        items, total = await self.lab_documents.page(owner_id, page)
        if not items and page > 0:
            await message.reply_text(
                "Эта страница списка устарела.", reply_markup=self._labs_back_keyboard()
            )
            return
        if not items:
            await message.reply_text(
                "Мои документы\n\nПока ничего не сохранено.",
                reply_markup=self._labs_back_keyboard(),
            )
            return
        rows = [
            [
                InlineKeyboardButton(
                    self._button_title(item.title), callback_data=f"labs:open:{item.id}"
                )
            ]
            for item in items
        ]
        pagination: list[InlineKeyboardButton] = []
        if page > 0:
            pagination.append(InlineKeyboardButton("←", callback_data=f"labs:list:{page - 1}"))
        if (page + 1) * LAB_LIST_PAGE_SIZE < total:
            pagination.append(InlineKeyboardButton("→", callback_data=f"labs:list:{page + 1}"))
        if pagination:
            rows.append(pagination)
        rows.extend(self._labs_navigation_rows())
        lines = ["Мои документы"]
        for item in items:
            lines.append(
                f"• {item.title} · {self._format_date(item.document_date)} · "
                f"{item.page_count} стр. · загружен {item.created_at:%d.%m.%Y}"
            )
        await message.reply_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(rows))

    async def _labs_send_document(self, message: Any, document: Any) -> None:
        await message.reply_text(
            f"{document.title}\n"
            f"Дата документа: {self._format_date(document.document_date)}\n"
            f"Страниц: {document.page_count}\n"
            f"Загружен: {document.created_at:%d.%m.%Y}",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Открыть", callback_data=f"labs:view:{document.id}:0")],
                    [
                        InlineKeyboardButton(
                            "Переименовать",
                            callback_data=f"labs:rename:{document.id}:{document.version}",
                        ),
                        InlineKeyboardButton(
                            "Изменить дату",
                            callback_data=f"labs:date:{document.id}:{document.version}",
                        ),
                    ],
                    [InlineKeyboardButton("Удалить", callback_data=f"labs:delete:{document.id}")],
                    [InlineKeyboardButton("← Мои документы", callback_data="labs:list:0")],
                    *self._labs_navigation_rows(),
                ]
            ),
        )

    async def _labs_send_page(
        self, message: Any, document: Any, page_index: int, image_bytes: bytes
    ) -> None:
        rows: list[list[InlineKeyboardButton]] = []
        navigation: list[InlineKeyboardButton] = []
        if page_index > 0:
            navigation.append(
                InlineKeyboardButton("←", callback_data=f"labs:view:{document.id}:{page_index - 1}")
            )
        if page_index + 1 < document.page_count:
            navigation.append(
                InlineKeyboardButton("→", callback_data=f"labs:view:{document.id}:{page_index + 1}")
            )
        if navigation:
            rows.append(navigation)
        rows.append(
            [InlineKeyboardButton("← К документу", callback_data=f"labs:open:{document.id}")]
        )
        rows.extend(self._labs_navigation_rows())
        stream = BytesIO(image_bytes)
        stream.name = "lab-page.jpg"
        try:
            await message.reply_photo(
                photo=stream,
                caption=f"{document.title} · страница {page_index + 1}/{document.page_count}",
                reply_markup=InlineKeyboardMarkup(rows),
            )
        finally:
            stream.close()

    @staticmethod
    def _telegram_lab_media(message: Any) -> tuple[Any, TelegramLabMetadata]:
        photos = list(message.photo or [])
        if photos:
            media = photos[-1]
            return media, TelegramLabMetadata(
                "photo",
                getattr(media, "file_size", None),
                getattr(media, "mime_type", None),
                getattr(media, "width", None),
                getattr(media, "height", None),
            )
        document = message.document
        if document is None:
            raise LabMediaError("unsupported_source")
        return document, TelegramLabMetadata(
            "document",
            getattr(document, "file_size", None),
            getattr(document, "mime_type", None),
        )

    @staticmethod
    def _labs_menu_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Добавить результаты", callback_data="labs:add")],
                [InlineKeyboardButton("Мои документы", callback_data="labs:list:0")],
                [InlineKeyboardButton("Как это работает", callback_data="labs:help")],
                [InlineKeyboardButton("← Назад", callback_data="nav:section:doctor")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            ]
        )

    @staticmethod
    def _labs_back_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("← Назад", callback_data="labs:menu")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            ]
        )

    @staticmethod
    def _labs_navigation_rows() -> list[list[InlineKeyboardButton]]:
        return [
            [InlineKeyboardButton("← Анализы", callback_data="labs:menu")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
        ]

    @staticmethod
    async def _labs_stale(query: Any) -> None:
        await query.answer("Эта кнопка устарела или недоступна.", show_alert=True)

    @staticmethod
    async def _labs_edit_or_send(query: Any, text: str, reply_markup: InlineKeyboardMarkup) -> None:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup)
        except (TelegramError, TypeError):
            await query.message.reply_text(text, reply_markup=reply_markup)

    @staticmethod
    def _parse_document_date(value: str) -> date | None:
        clean = value.strip()
        if clean in {"-", "—", "нет", "Нет"}:
            return None
        parsed = datetime.strptime(clean, "%d.%m.%Y").date()
        if parsed.year < 1900 or parsed > date.today():
            raise ValueError("invalid_date")
        return parsed

    @staticmethod
    def _format_date(value: date | None) -> str:
        return "не указана" if value is None else value.strftime("%d.%m.%Y")

    @staticmethod
    def _button_title(value: str) -> str:
        return value if len(value) <= 48 else f"{value[:45]}…"

    @staticmethod
    def _safe_int(value: str, *, positive: bool = False) -> int | None:
        if not value.isascii() or not value.isdigit() or len(value) > 10:
            return None
        result = int(value)
        if result > 1_000_000_000 or (positive and result <= 0):
            return None
        return result

    @staticmethod
    def _owned_document_edit(context: Any, owner_id: int, chat_id: int) -> bool:
        edit = context.user_data.get("lab_document_edit")
        return bool(
            edit is not None
            and edit.get("owner_id") == owner_id
            and edit.get("chat_id") == chat_id
            and float(edit.get("expires_at", 0)) > monotonic()
        )
