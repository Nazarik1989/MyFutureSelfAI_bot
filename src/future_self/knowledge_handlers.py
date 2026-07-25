from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import AsyncIterator
from datetime import timedelta
from html import escape
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from .knowledge import (
    KnowledgeAccessDenied,
    KnowledgeCaptureError,
    KnowledgeCapturePreview,
    KnowledgeConflictError,
    KnowledgeError,
    KnowledgeJobError,
    KnowledgeQuotaError,
    KnowledgeSourceRecord,
    KnowledgeStaleError,
    StoredKnowledgeOriginal,
)
from .knowledge_extraction import KnowledgeExtractionError, inspect_upload
from .knowledge_storage import KnowledgeAssetStore, KnowledgeStorageError, StagedAsset

logger = logging.getLogger(__name__)
# Telegram file URLs carry the bot token in their path. Keep dependency loggers
# silent even when handlers are exercised outside the production logging setup.
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

ROLE_LABELS = {
    "foundation": "Основа",
    "trusted": "Доверенный материал",
    "perspective": "Позиция",
    "discussion": "Для обсуждения",
    "counterpoint": "Альтернативная позиция",
    "hypothesis": "Гипотеза",
}
PRIORITY_LABELS = {"high": "Высокий", "normal": "Обычный", "low": "Низкий"}
STATUS_LABELS = {
    "queued": "В очереди",
    "processing": "Обрабатывается",
    "ready": "Готово",
    "partial": "Оригинал сохранён, текст извлечён частично или недоступен",
    "failed": "Не удалось обработать",
    "quarantined": "Помещено в карантин",
    "cancelled": "Обработка отменена",
}
KIND_LABELS = {
    "text": "Текст",
    "forward": "Пересланное сообщение",
    "document": "Документ",
    "image": "Изображение",
    "url": "Ссылка",
}
SAFE_ERROR_MESSAGES = {
    "external_fetch_disabled": "Ссылка сохранена без загрузки страницы.",
    "image_without_ocr": "Изображение сохранено без OCR.",
    "image_only_pdf": "PDF сохранён, но текстовый слой не найден.",
    "unsupported_format": "Формат сохранён, но извлечение для него недоступно.",
    "mime_mismatch": "Формат файла не совпал с заявленным типом.",
    "extension_mismatch": "Расширение файла не совпало с его содержимым.",
    "archive_limits_exceeded": "Архив превысил безопасные ограничения.",
    "text_limit_reached": "Извлечённый текст достиг безопасного лимита.",
    "purge_io_failed": (
        "Не удалось удалить один или несколько файлов. Владелец может безопасно "
        "повторить окончательное удаление."
    ),
}
_CALLBACK_TOKEN = re.compile(r"[A-Za-z0-9_-]{20,48}\Z")
_MAX_PREVIEW_CHARS = 500
_MAX_BUTTON_CHARS = 52
_DOWNLOAD_TIMEOUT_SECONDS = 60


class KnowledgeTelegramDownloadError(ValueError):
    """Non-sensitive boundary error; never includes a Telegram file URL."""


class KnowledgeHandlers:
    knowledge_service: Any
    knowledge_storage: KnowledgeAssetStore | None
    task_service: Any
    collection_service: Any
    workspace_service: Any
    lab_uploads: Any
    vision_service: Any
    vision_image_sessions: Any

    async def knowledge_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._knowledge_hub_enabled():
            await update.effective_message.reply_text("База знаний сейчас выключена.")
            return
        flow = await self._knowledge_specialized_flow(update, context)
        if flow is not None:
            await update.effective_message.reply_text(
                "Сначала заверши текущий сценарий или используй /cancel. "
                "База знаний ничего не перехватила."
            )
            return
        user = await self._user(update.effective_user.id)
        await self._send_knowledge_hub(update.effective_message, user.id, update.effective_chat.id)

    async def capture_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._knowledge_capture_enabled():
            await update.effective_message.reply_text("Добавление материалов сейчас выключено.")
            return
        flow = await self._knowledge_specialized_flow(update, context)
        if flow is not None:
            await update.effective_message.reply_text(
                "Сначала заверши текущий сценарий или используй /cancel. Capture не запущен."
            )
            return
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        state = await self.knowledge_service.capture_state(user.id, chat_id)
        if state.preview is not None:
            await self._send_capture_preview(
                update.effective_message, user.id, chat_id, state.preview
            )
            return
        preview = await self.knowledge_service.begin_empty_capture(
            user.id, chat_id, ttl=self._knowledge_capture_ttl()
        )
        raw = " ".join(tuple(getattr(context, "args", ()) or ())).strip()
        if raw:
            preview = await self._set_text_payload(user.id, chat_id, preview, raw, update)
            await self._send_capture_preview(update.effective_message, user.id, chat_id, preview)
            return
        await self._send_capture_input_prompt(update.effective_message, user.id, chat_id, preview)

    async def knowledge_media_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._knowledge_capture_enabled():
            return
        flow = await self._knowledge_specialized_flow(update, context)
        if flow is not None:
            await update.effective_message.reply_text(
                "Сейчас активен другой сценарий. Файл не загружен и не добавлен в базу знаний."
            )
            return
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        state = await self.knowledge_service.capture_state(user.id, chat_id)
        if state.expired_now:
            await update.effective_message.reply_text(
                "Время Capture истекло. Файл не загружен; отправь его ещё раз."
            )
            return
        pending = await self.knowledge_service.pending_input(user.id, chat_id)
        if pending is not None:
            await update.effective_message.reply_text(
                "Сейчас ожидается текстовое поле Capture. Пришли текст или используй /cancel."
            )
            return
        preview = state.preview
        if preview is not None and preview.status != "collecting":
            await update.effective_message.reply_text(
                "Уже открыт Capture preview. Используй его кнопки или /cancel; новый файл не загружен."
            )
            return
        try:
            metadata = self._telegram_media_metadata(update.effective_message)
            if preview is None:
                preview = await self.knowledge_service.begin_capture(
                    user.id,
                    chat_id,
                    capture_kind=metadata["capture_kind"],
                    telegram_file_id=metadata["file_id"],
                    telegram_file_unique_id_hash=metadata["file_unique_id_hash"],
                    telegram_message_id=getattr(update.effective_message, "message_id", None),
                    declared_mime=metadata["declared_mime"],
                    safe_display_name=metadata["safe_display_name"],
                    declared_size_bytes=metadata["declared_size_bytes"],
                    provenance=self._telegram_provenance(update.effective_message),
                    ttl=self._knowledge_capture_ttl(),
                )
            else:
                preview = await self.knowledge_service.set_capture_payload(
                    user.id,
                    chat_id,
                    preview.draft_public_id,
                    preview.version,
                    capture_kind=metadata["capture_kind"],
                    telegram_file_id=metadata["file_id"],
                    telegram_file_unique_id_hash=metadata["file_unique_id_hash"],
                    telegram_message_id=getattr(update.effective_message, "message_id", None),
                    declared_mime=metadata["declared_mime"],
                    safe_display_name=metadata["safe_display_name"],
                    declared_size_bytes=metadata["declared_size_bytes"],
                    provenance=self._telegram_provenance(update.effective_message),
                )
        except KnowledgeError as exc:
            await update.effective_message.reply_text(str(exc))
            return
        await self._send_capture_preview(update.effective_message, user.id, chat_id, preview)

    async def knowledge_pending_text(
        self,
        update: Update,
        text: str,
        source: str,
    ) -> bool:
        del source
        if not self._knowledge_capture_enabled():
            return False
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        state = await self.knowledge_service.capture_state(user.id, chat_id)
        if state.expired_now:
            await update.effective_message.reply_text(
                "Время Capture истекло. Сообщение не отправлено в LLM; начни снова через /capture."
            )
            return True
        pending = await self.knowledge_service.pending_input(user.id, chat_id)
        if pending is not None:
            claim = await self.knowledge_service.claim_pending_input(user.id, chat_id)
            if claim is None:
                await update.effective_message.reply_text(
                    "Поле Capture устарело. Открой /knowledge и начни заново."
                )
                return True
            return await self._handle_knowledge_input(update, user.id, chat_id, claim, text)
        preview = state.preview
        if preview is not None:
            if preview.status == "collecting":
                try:
                    preview = await self._set_text_payload(user.id, chat_id, preview, text, update)
                except KnowledgeError as exc:
                    await update.effective_message.reply_text(str(exc))
                    return True
                await self._send_capture_preview(
                    update.effective_message, user.id, chat_id, preview
                )
            else:
                await update.effective_message.reply_text(
                    "Capture уже ждёт подтверждения. Используй кнопки preview или /cancel."
                )
            return True
        if getattr(update.effective_message, "forward_origin", None) is None:
            return False
        try:
            preview = await self.knowledge_service.begin_capture(
                user.id,
                chat_id,
                capture_kind="forward",
                text_content=text,
                telegram_message_id=getattr(update.effective_message, "message_id", None),
                declared_mime="text/plain",
                safe_display_name="forward.txt",
                declared_size_bytes=len(text.encode("utf-8")),
                provenance=self._telegram_provenance(update.effective_message),
                ttl=self._knowledge_capture_ttl(),
            )
        except KnowledgeError as exc:
            await update.effective_message.reply_text(str(exc))
            return True
        await self._send_capture_preview(update.effective_message, user.id, chat_id, preview)
        return True

    async def knowledge_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = query.data or ""
        if not self._knowledge_hub_enabled() or not data.startswith("kh:"):
            await self._knowledge_stale(query)
            return
        token = data.removeprefix("kh:")
        if not _CALLBACK_TOKEN.fullmatch(token):
            await self._knowledge_stale(query)
            return
        flow = await self._knowledge_specialized_flow(update, context)
        if flow is not None:
            await query.answer(
                "Сначала завершите текущий сценарий; база знаний ничего не изменила.",
                show_alert=True,
            )
            return
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        try:
            claim = await self.knowledge_service.claim_action(token, user.id, chat_id)
        except KnowledgeError:
            claim = None
        if claim is None:
            await self._knowledge_stale(query)
            return
        if claim.action.startswith("capture_") and not self._knowledge_capture_enabled():
            await query.answer(
                "Добавление материалов сейчас отключено; старое действие не выполнено.",
                show_alert=True,
            )
            return
        await query.answer()
        try:
            await self._dispatch_knowledge_action(query, context, user.id, chat_id, claim)
        except (KnowledgeAccessDenied, KnowledgeStaleError):
            await self._knowledge_stale_message(query)
        except (
            KnowledgeConflictError,
            KnowledgeQuotaError,
            KnowledgeCaptureError,
            KnowledgeJobError,
        ) as exc:
            await self._knowledge_edit_or_send(query, escape(str(exc)), None, parse_mode="HTML")
        except KnowledgeError as exc:
            await self._knowledge_edit_or_send(query, escape(str(exc)), None, parse_mode="HTML")

    async def cancel_knowledge_state(self, update: Update) -> bool:
        if not self._knowledge_capture_enabled():
            return False
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        state = await self.knowledge_service.capture_state(user.id, chat_id)
        pending = await self.knowledge_service.cancel_pending_input(user.id, chat_id)
        if state.preview is None:
            return pending or state.expired_now
        try:
            cancelled = await self.knowledge_service.cancel_capture(
                user.id,
                chat_id,
                state.preview.draft_public_id,
                state.preview.version,
            )
        except KnowledgeError:
            cancelled = False
        return pending or cancelled or state.expired_now

    async def knowledge_other_callback_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await self.workspace_other_callback_gate(update, context)
        query = update.callback_query
        if not self._knowledge_capture_enabled() or query is None:
            return
        if (query.data or "").startswith(("kh:", "nav:")):
            return
        await self.cancel_knowledge_state(update)

    async def _dispatch_knowledge_action(
        self,
        query: Any,
        context: Any,
        actor_id: int,
        chat_id: int,
        claim: Any,
    ) -> None:
        action = claim.action
        payload = claim.payload
        if action == "hub":
            await self._render_knowledge_hub(query, actor_id, chat_id)
            return
        if action in {"space", "space_page", "space_trash"}:
            lifecycle = (
                "trashed" if action == "space_trash" else str(payload.get("lifecycle", "active"))
            )
            await self._render_source_page(
                query,
                actor_id,
                chat_id,
                claim.space_public_id,
                page=int(payload.get("page", 1)),
                lifecycle="trashed" if lifecycle == "trashed" else "active",
            )
            return
        if action == "capture_start":
            state = await self.knowledge_service.capture_state(actor_id, chat_id)
            preview = state.preview or await self.knowledge_service.begin_empty_capture(
                actor_id, chat_id, ttl=self._knowledge_capture_ttl()
            )
            if preview.target_space_public_id != claim.space_public_id:
                preview = await self.knowledge_service.update_capture(
                    actor_id,
                    chat_id,
                    preview.draft_public_id,
                    preview.version,
                    target_space_public_id=claim.space_public_id,
                )
            await self._send_capture_input_prompt(query.message, actor_id, chat_id, preview)
            return
        if action == "capture_target_menu":
            preview = await self._capture_for_claim(actor_id, chat_id, claim)
            await self._render_capture_targets(query, actor_id, chat_id, preview)
            return
        if action == "capture_target":
            preview = await self._capture_for_claim(actor_id, chat_id, claim)
            target = str(payload.get("target", ""))
            updated = await self.knowledge_service.update_capture(
                actor_id,
                chat_id,
                preview.draft_public_id,
                preview.version,
                target_space_public_id=target,
            )
            await self._render_capture_preview(query, actor_id, chat_id, updated)
            return
        if action == "capture_role":
            preview = await self._capture_for_claim(actor_id, chat_id, claim)
            updated = await self.knowledge_service.update_capture(
                actor_id,
                chat_id,
                preview.draft_public_id,
                preview.version,
                knowledge_role=str(payload.get("role", "")),
            )
            await self._render_capture_preview(query, actor_id, chat_id, updated)
            return
        if action == "capture_priority":
            preview = await self._capture_for_claim(actor_id, chat_id, claim)
            updated = await self.knowledge_service.update_capture(
                actor_id,
                chat_id,
                preview.draft_public_id,
                preview.version,
                priority=str(payload.get("priority", "")),
            )
            await self._render_capture_preview(query, actor_id, chat_id, updated)
            return
        if action == "capture_title":
            preview = await self._capture_for_claim(actor_id, chat_id, claim)
            await self.knowledge_service.issue_action(
                actor_id,
                chat_id,
                "input_title",
                preview.target_space_public_id,
                capture_draft_public_id=preview.draft_public_id,
                status="awaiting_input",
                ttl=self._knowledge_action_ttl(),
            )
            await query.message.reply_text(
                "Пришли новое название одним сообщением. /cancel — отменить Capture."
            )
            return
        if action == "capture_cancel":
            preview = await self._capture_for_claim(actor_id, chat_id, claim)
            await self.knowledge_service.cancel_pending_input(actor_id, chat_id)
            await self.knowledge_service.cancel_capture(
                actor_id,
                chat_id,
                preview.draft_public_id,
                preview.version,
            )
            await self._knowledge_edit_or_send(
                query,
                "Capture отменён. Источник, задание и оригинал не создавались.",
                self._knowledge_navigation_markup(),
            )
            return
        if action == "capture_confirm":
            preview = await self._capture_for_claim(actor_id, chat_id, claim)
            await self._confirm_capture(query, context, actor_id, chat_id, claim, preview)
            return
        if action == "source_open":
            record = await self._source_for_claim(actor_id, claim)
            await self._render_source_card(query, actor_id, chat_id, claim.space_public_id, record)
            return
        if action == "source_refresh":
            record = await self._source_for_claim(actor_id, claim)
            await self._render_source_card(query, actor_id, chat_id, claim.space_public_id, record)
            return
        if action == "source_trash":
            record = await self._source_for_claim(actor_id, claim)
            await self.knowledge_service.trash_source(
                actor_id, record.source.public_id, record.source.version
            )
            await self._render_source_page(
                query,
                actor_id,
                chat_id,
                claim.space_public_id,
                page=1,
                lifecycle="active",
                notice="Материал перемещён в корзину.",
            )
            return
        if action == "source_restore":
            record = await self._source_for_claim(actor_id, claim)
            restored = await self.knowledge_service.restore_source(
                actor_id, record.source.public_id, record.source.version
            )
            refreshed = await self.knowledge_service.get_source(actor_id, restored.public_id)
            await self._render_source_card(
                query,
                actor_id,
                chat_id,
                claim.space_public_id,
                refreshed,
                notice="Материал восстановлен.",
            )
            return
        if action == "source_cancel_job":
            record = await self._source_for_claim(actor_id, claim)
            changed = await self.knowledge_service.cancel_source_job(
                actor_id, record.source.public_id, record.source.version
            )
            refreshed = await self.knowledge_service.get_source(actor_id, record.source.public_id)
            await self._render_source_card(
                query,
                actor_id,
                chat_id,
                claim.space_public_id,
                refreshed,
                notice=("Отмена обработки запрошена." if changed else "Активного задания уже нет."),
            )
            return
        if action == "source_retry":
            record = await self._source_for_claim(actor_id, claim)
            await self.knowledge_service.retry_source(
                actor_id,
                record.source.public_id,
                record.source.version,
                max_attempts=self.settings.knowledge_runner_max_attempts,
            )
            refreshed = await self.knowledge_service.get_source(actor_id, record.source.public_id)
            await self._render_source_card(
                query,
                actor_id,
                chat_id,
                claim.space_public_id,
                refreshed,
                notice="Повторная обработка поставлена в очередь.",
            )
            return
        if action == "source_purge_prompt":
            record = await self._source_for_claim(actor_id, claim)
            confirm = await self._knowledge_action(
                actor_id,
                chat_id,
                "source_purge",
                claim.space_public_id,
                source_public_id=record.source.public_id,
            )
            back = await self._knowledge_action(
                actor_id,
                chat_id,
                "source_open",
                claim.space_public_id,
                source_public_id=record.source.public_id,
            )
            await self._knowledge_edit_or_send(
                query,
                "<b>Удалить материал окончательно?</b>\n\n"
                "Будут удалены оригинал, извлечённые данные и задания. Это действие выполняется "
                "фоновым purge и не обещает успех до его завершения.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Удалить окончательно", callback_data=f"kh:{confirm}"
                            )
                        ],
                        [InlineKeyboardButton("Отмена", callback_data=f"kh:{back}")],
                    ]
                ),
                parse_mode="HTML",
            )
            return
        if action == "source_purge":
            record = await self._source_for_claim(actor_id, claim)
            await self.knowledge_service.request_permanent_delete(
                actor_id,
                record.source.public_id,
                record.source.version,
                max_attempts=self.settings.knowledge_runner_max_attempts,
            )
            await self._render_source_page(
                query,
                actor_id,
                chat_id,
                claim.space_public_id,
                page=1,
                lifecycle="trashed",
                notice="Безопасное окончательное удаление поставлено в очередь.",
            )
            return
        raise KnowledgeStaleError("Действие устарело.")

    async def _send_knowledge_hub(self, message: Any, actor_id: int, chat_id: int) -> None:
        text, markup = await self._knowledge_hub_view(actor_id, chat_id)
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_knowledge_hub(self, query: Any, actor_id: int, chat_id: int) -> None:
        text, markup = await self._knowledge_hub_view(actor_id, chat_id)
        await self._knowledge_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _knowledge_hub_view(
        self, actor_id: int, chat_id: int
    ) -> tuple[str, InlineKeyboardMarkup]:
        personal = await self.knowledge_service.ensure_personal_space(actor_id)
        spaces = await self.knowledge_service.list_spaces(actor_id)
        lines = [
            "<b>База знаний</b>",
            "",
            "Материалы хранятся отдельно от Inbox и пока не используются как LLM-контекст.",
        ]
        rows: list[list[InlineKeyboardButton]] = []
        for space in spaces:
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "space",
                space.access.space_public_id,
                payload={"page": 1, "lifecycle": "active"},
            )
            label = f"{self._space_icon(space.access.kind)} {space.name}"
            rows.append(
                [InlineKeyboardButton(self._button_label(label), callback_data=f"kh:{token}")]
            )
        if self._knowledge_capture_enabled():
            add = await self._knowledge_action(
                actor_id,
                chat_id,
                "capture_start",
                personal.access.space_public_id,
            )
            rows.append([InlineKeyboardButton("➕ Добавить материал", callback_data=f"kh:{add}")])
        rows.extend(self._knowledge_navigation_rows())
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    async def _render_source_page(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        space_public_id: str,
        *,
        page: int,
        lifecycle: str,
        notice: str | None = None,
    ) -> None:
        access = await self.knowledge_service.resolve_space(actor_id, space_public_id)
        listing = await self.knowledge_service.list_sources(
            actor_id,
            space_public_id,
            lifecycle_status=lifecycle,
            page=max(page, 1),
        )
        spaces = await self.knowledge_service.list_spaces(actor_id)
        space = next(item for item in spaces if item.access.space_public_id == space_public_id)
        title = "Корзина" if lifecycle == "trashed" else space.name
        lines = [f"<b>{escape(title)}</b>"]
        if notice:
            lines.extend(("", escape(notice)))
        if not listing.items:
            lines.extend(("", "Материалов пока нет."))
        else:
            lines.extend(("", f"Страница {listing.page} из {listing.pages}."))
        rows: list[list[InlineKeyboardButton]] = []
        for record in listing.items:
            source = record.source
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "source_open",
                space_public_id,
                source_public_id=source.public_id,
            )
            label = f"{self._status_icon(source.processing_status)} {source.title}"
            rows.append(
                [InlineKeyboardButton(self._button_label(label), callback_data=f"kh:{token}")]
            )
        pagination: list[InlineKeyboardButton] = []
        if listing.page > 1:
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "space_page",
                space_public_id,
                payload={"page": listing.page - 1, "lifecycle": lifecycle},
            )
            pagination.append(InlineKeyboardButton("←", callback_data=f"kh:{token}"))
        if listing.page < listing.pages:
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "space_page",
                space_public_id,
                payload={"page": listing.page + 1, "lifecycle": lifecycle},
            )
            pagination.append(InlineKeyboardButton("→", callback_data=f"kh:{token}"))
        if pagination:
            rows.append(pagination)
        if lifecycle == "active":
            if self._knowledge_capture_enabled() and access.role in {"owner", "editor"}:
                add = await self._knowledge_action(
                    actor_id, chat_id, "capture_start", space_public_id
                )
                rows.append([InlineKeyboardButton("➕ Добавить", callback_data=f"kh:{add}")])
            trash = await self._knowledge_action(
                actor_id,
                chat_id,
                "space_trash",
                space_public_id,
                payload={"page": 1, "lifecycle": "trashed"},
            )
            rows.append([InlineKeyboardButton("🗑 Корзина", callback_data=f"kh:{trash}")])
        else:
            active = await self._knowledge_action(
                actor_id,
                chat_id,
                "space_page",
                space_public_id,
                payload={"page": 1, "lifecycle": "active"},
            )
            rows.append([InlineKeyboardButton("← Материалы", callback_data=f"kh:{active}")])
        hub = await self._knowledge_action(actor_id, chat_id, "hub", space_public_id)
        rows.append([InlineKeyboardButton("← Пространства", callback_data=f"kh:{hub}")])
        rows.extend(self._knowledge_navigation_rows())
        await self._knowledge_edit_or_send(
            query, "\n".join(lines), InlineKeyboardMarkup(rows), parse_mode="HTML"
        )

    async def _render_source_card(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        space_public_id: str,
        record: KnowledgeSourceRecord,
        *,
        notice: str | None = None,
    ) -> None:
        source = record.source
        lines = [
            f"<b>{escape(source.title)}</b>",
            f"ID: <code>{escape(source.public_id)}</code>",
            f"Тип: {escape(KIND_LABELS.get(source.source_type, 'Материал'))}",
            f"Статус: {escape(STATUS_LABELS.get(source.processing_status, source.processing_status))}",
            f"Роль знания: {escape(ROLE_LABELS.get(source.knowledge_role, source.knowledge_role))}",
            f"Приоритет: {escape(PRIORITY_LABELS.get(source.priority, source.priority))}",
        ]
        if notice:
            lines.extend(("", escape(notice)))
        if record.revision is not None:
            lines.append(f"Размер оригинала: {self._format_bytes(record.revision.size_bytes)}")
        if record.active_job is not None and record.active_job.safe_error_code:
            lines.extend(
                (
                    "",
                    escape(
                        SAFE_ERROR_MESSAGES.get(
                            record.active_job.safe_error_code,
                            "Обработка завершилась с безопасно скрытой технической причиной.",
                        )
                    ),
                )
            )
        if source.lifecycle_status == "purge_failed":
            lines.extend(
                (
                    "",
                    "Окончательное удаление не завершено. Материал остаётся видимым "
                    "в корзине; владелец может безопасно повторить удаление.",
                )
            )
        rows: list[list[InlineKeyboardButton]] = []
        refresh = await self._knowledge_action(
            actor_id,
            chat_id,
            "source_refresh",
            space_public_id,
            source_public_id=source.public_id,
        )
        rows.append([InlineKeyboardButton("Обновить статус", callback_data=f"kh:{refresh}")])
        if record.role in {"owner", "editor"}:
            if source.lifecycle_status == "active":
                if source.processing_status in {"queued", "processing"}:
                    cancel = await self._knowledge_action(
                        actor_id,
                        chat_id,
                        "source_cancel_job",
                        space_public_id,
                        source_public_id=source.public_id,
                    )
                    rows.append(
                        [InlineKeyboardButton("Отменить обработку", callback_data=f"kh:{cancel}")]
                    )
                if source.processing_status in {"failed", "cancelled"}:
                    retry = await self._knowledge_action(
                        actor_id,
                        chat_id,
                        "source_retry",
                        space_public_id,
                        source_public_id=source.public_id,
                    )
                    rows.append(
                        [InlineKeyboardButton("Повторить обработку", callback_data=f"kh:{retry}")]
                    )
                trash = await self._knowledge_action(
                    actor_id,
                    chat_id,
                    "source_trash",
                    space_public_id,
                    source_public_id=source.public_id,
                )
                rows.append([InlineKeyboardButton("В корзину", callback_data=f"kh:{trash}")])
            elif source.lifecycle_status == "trashed":
                restore = await self._knowledge_action(
                    actor_id,
                    chat_id,
                    "source_restore",
                    space_public_id,
                    source_public_id=source.public_id,
                )
                rows.append([InlineKeyboardButton("Восстановить", callback_data=f"kh:{restore}")])
                if record.role == "owner":
                    purge = await self._knowledge_action(
                        actor_id,
                        chat_id,
                        "source_purge_prompt",
                        space_public_id,
                        source_public_id=source.public_id,
                    )
                    rows.append(
                        [InlineKeyboardButton("Удалить окончательно", callback_data=f"kh:{purge}")]
                    )
            elif source.lifecycle_status == "purge_failed" and record.role == "owner":
                retry_purge = await self._knowledge_action(
                    actor_id,
                    chat_id,
                    "source_purge",
                    space_public_id,
                    source_public_id=source.public_id,
                )
                rows.append(
                    [InlineKeyboardButton("Повторить удаление", callback_data=f"kh:{retry_purge}")]
                )
        lifecycle = "trashed" if source.lifecycle_status != "active" else "active"
        back = await self._knowledge_action(
            actor_id,
            chat_id,
            "space_page",
            space_public_id,
            payload={"page": 1, "lifecycle": lifecycle},
        )
        rows.append([InlineKeyboardButton("← К материалам", callback_data=f"kh:{back}")])
        rows.extend(self._knowledge_navigation_rows())
        await self._knowledge_edit_or_send(
            query, "\n".join(lines), InlineKeyboardMarkup(rows), parse_mode="HTML"
        )

    async def _send_capture_input_prompt(
        self,
        message: Any,
        actor_id: int,
        chat_id: int,
        preview: KnowledgeCapturePreview,
    ) -> None:
        cancel = await self._knowledge_action(
            actor_id,
            chat_id,
            "capture_cancel",
            self._capture_space_id(preview),
            capture_draft_public_id=preview.draft_public_id,
        )
        await message.reply_text(
            "Пришли один текст, пересланное сообщение, документ, изображение или ссылку. "
            "До отдельного подтверждения источник и оригинал не создаются.",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Отмена", callback_data=f"kh:{cancel}")]]
            ),
        )

    async def _send_capture_preview(
        self,
        message: Any,
        actor_id: int,
        chat_id: int,
        preview: KnowledgeCapturePreview,
    ) -> None:
        text, markup = await self._capture_preview_view(actor_id, chat_id, preview)
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_capture_preview(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        preview: KnowledgeCapturePreview,
    ) -> None:
        text, markup = await self._capture_preview_view(actor_id, chat_id, preview)
        await self._knowledge_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _capture_preview_view(
        self,
        actor_id: int,
        chat_id: int,
        preview: KnowledgeCapturePreview,
    ) -> tuple[str, InlineKeyboardMarkup]:
        if preview.status == "collecting":
            cancel = await self._knowledge_action(
                actor_id,
                chat_id,
                "capture_cancel",
                self._capture_space_id(preview),
                capture_draft_public_id=preview.draft_public_id,
            )
            return (
                "<b>Capture</b>\n\nПришли материал. До подтверждения ничего не сохраняется как источник.",
                InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Отмена", callback_data=f"kh:{cancel}")]]
                ),
            )
        warning = self._capture_warning(preview.capture_kind)
        content = (
            f"\nФрагмент: {escape(self._truncate(preview.content_preview, _MAX_PREVIEW_CHARS))}"
            if preview.content_preview
            else ""
        )
        lines = [
            "<b>Capture preview</b>",
            "",
            f"Что: {escape(KIND_LABELS.get(preview.capture_kind, 'Материал'))}",
            f"Куда: <b>{escape(preview.target_name or 'Личная база знаний')}</b>",
            f"Название: <b>{escape(preview.title or 'Материал')}</b>",
            f"Роль знания: {escape(ROLE_LABELS.get(preview.knowledge_role, preview.knowledge_role))}",
            f"Приоритет: {escape(PRIORITY_LABELS.get(preview.priority, preview.priority))}",
            content,
            "",
            f"<i>{escape(warning)}</i>",
            "<i>Внешний файл может содержать медицинские сведения, которые без OCR/LLM "
            "нельзя определить автоматически. Выбирай совместное пространство осознанно.</i>",
        ]
        space_id = self._capture_space_id(preview)
        target = await self._knowledge_action(
            actor_id,
            chat_id,
            "capture_target_menu",
            space_id,
            capture_draft_public_id=preview.draft_public_id,
        )
        title = await self._knowledge_action(
            actor_id,
            chat_id,
            "capture_title",
            space_id,
            capture_draft_public_id=preview.draft_public_id,
        )
        rows: list[list[InlineKeyboardButton]] = [
            [
                InlineKeyboardButton("Изменить место", callback_data=f"kh:{target}"),
                InlineKeyboardButton("Название", callback_data=f"kh:{title}"),
            ]
        ]
        role_row: list[InlineKeyboardButton] = []
        for role in ("trusted", "discussion", "perspective"):
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "capture_role",
                space_id,
                capture_draft_public_id=preview.draft_public_id,
                payload={"role": role},
            )
            role_row.append(
                InlineKeyboardButton(
                    self._button_label(ROLE_LABELS[role], maximum=24),
                    callback_data=f"kh:{token}",
                )
            )
        rows.append(role_row)
        role_row = []
        for role in ("foundation", "counterpoint", "hypothesis"):
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "capture_role",
                space_id,
                capture_draft_public_id=preview.draft_public_id,
                payload={"role": role},
            )
            role_row.append(
                InlineKeyboardButton(
                    self._button_label(ROLE_LABELS[role], maximum=24),
                    callback_data=f"kh:{token}",
                )
            )
        rows.append(role_row)
        priority_row: list[InlineKeyboardButton] = []
        for priority in ("high", "normal", "low"):
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "capture_priority",
                space_id,
                capture_draft_public_id=preview.draft_public_id,
                payload={"priority": priority},
            )
            priority_row.append(
                InlineKeyboardButton(PRIORITY_LABELS[priority], callback_data=f"kh:{token}")
            )
        rows.append(priority_row)
        confirm = await self._knowledge_action(
            actor_id,
            chat_id,
            "capture_confirm",
            space_id,
            capture_draft_public_id=preview.draft_public_id,
        )
        cancel = await self._knowledge_action(
            actor_id,
            chat_id,
            "capture_cancel",
            space_id,
            capture_draft_public_id=preview.draft_public_id,
        )
        rows.append(
            [
                InlineKeyboardButton("Сохранить и обработать", callback_data=f"kh:{confirm}"),
                InlineKeyboardButton("Отмена", callback_data=f"kh:{cancel}"),
            ]
        )
        rows.extend(self._knowledge_navigation_rows())
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    async def _render_capture_targets(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        preview: KnowledgeCapturePreview,
    ) -> None:
        rows: list[list[InlineKeyboardButton]] = []
        for space in await self.knowledge_service.list_spaces(actor_id):
            if space.access.role not in {"owner", "editor"}:
                continue
            if (
                preview.system_classification == "health_private"
                and space.access.kind != "personal"
            ):
                continue
            token = await self._knowledge_action(
                actor_id,
                chat_id,
                "capture_target",
                self._capture_space_id(preview),
                capture_draft_public_id=preview.draft_public_id,
                payload={"target": space.access.space_public_id},
            )
            rows.append(
                [
                    InlineKeyboardButton(
                        self._button_label(f"{self._space_icon(space.access.kind)} {space.name}"),
                        callback_data=f"kh:{token}",
                    )
                ]
            )
        back = await self._knowledge_action(
            actor_id,
            chat_id,
            "capture_role",
            self._capture_space_id(preview),
            capture_draft_public_id=preview.draft_public_id,
            payload={"role": preview.knowledge_role},
        )
        rows.append([InlineKeyboardButton("← К preview", callback_data=f"kh:{back}")])
        await self._knowledge_edit_or_send(
            query,
            "<b>Куда сохранить?</b>\n\nЛичное пространство выбрано по умолчанию. "
            "Совместное требует явного выбора и действующей роли editor/owner.",
            InlineKeyboardMarkup(rows),
            parse_mode="HTML",
        )

    async def _confirm_capture(
        self,
        query: Any,
        context: Any,
        actor_id: int,
        chat_id: int,
        claim: Any,
        preview: KnowledgeCapturePreview,
    ) -> None:
        if self.knowledge_storage is None:
            raise KnowledgeCaptureError("Хранилище Capture сейчас недоступно.")
        await self._knowledge_edit_or_send(
            query,
            "Проверяю и сохраняю оригинал…",
            None,
        )
        reserved_bytes = preview.declared_size_bytes or self.settings.knowledge_max_source_bytes
        reservation = await self.knowledge_service.reserve_capture(
            actor_id,
            chat_id,
            preview.draft_public_id,
            preview.version,
            reserved_bytes=reserved_bytes,
            idempotency_key=f"capture:{preview.draft_public_id}:v{preview.version}",
        )
        staged: StagedAsset | None = None
        stored_key: str | None = None
        try:
            async with asyncio.timeout(_DOWNLOAD_TIMEOUT_SECONDS):
                staged = await self.knowledge_storage.stage_async(
                    self._capture_chunks(context, reservation.material),
                    declared_size=reservation.material.declared_size_bytes,
                )
            duplicate = await self.knowledge_service.find_duplicate(
                actor_id, claim.space_public_id, staged.sha256
            )
            if duplicate is not None:
                self.knowledge_storage.discard_staged(staged)
                staged = None
                await self.knowledge_service.release_capture_reservation(
                    actor_id, chat_id, reservation.public_id
                )
                current = await self.knowledge_service.capture_state(actor_id, chat_id)
                if current.preview is not None:
                    await self.knowledge_service.cancel_capture(
                        actor_id,
                        chat_id,
                        current.preview.draft_public_id,
                        current.preview.version,
                    )
                await self._render_source_card(
                    query,
                    actor_id,
                    chat_id,
                    claim.space_public_id,
                    duplicate,
                    notice="Такой оригинал уже есть в этом доступном пространстве; дубликат не создан.",
                )
                return
            material = reservation.material
            if material.capture_kind == "url":
                stored = self.knowledge_storage.finalize(staged)
                detected_mime = "text/uri-list"
                detected_format = "url"
            else:
                inspected = self.knowledge_storage.inspect_and_finalize(
                    staged,
                    declared_mime=material.declared_mime,
                    display_name=material.safe_display_name,
                    inspector=inspect_upload,
                )
                stored = inspected.asset
                detected_mime = inspected.inspection.detected_mime
                detected_format = inspected.inspection.source_format
            staged = None
            stored_key = stored.storage_key
            receipt = await self.knowledge_service.commit_capture(
                actor_id,
                chat_id,
                reservation.public_id,
                original=StoredKnowledgeOriginal(
                    storage_key=stored.storage_key,
                    sha256=stored.sha256,
                    size_bytes=stored.size_bytes,
                    declared_mime=material.declared_mime,
                    detected_mime=detected_mime,
                    detected_format=detected_format,
                    safe_display_name=material.safe_display_name
                    or self._default_display_name(material.capture_kind),
                    provenance=material.provenance,
                ),
                max_attempts=self.settings.knowledge_runner_max_attempts,
            )
        except (
            TimeoutError,
            httpx.HTTPError,
            TelegramError,
            KnowledgeTelegramDownloadError,
            KnowledgeStorageError,
            KnowledgeExtractionError,
            KnowledgeError,
        ) as exc:
            if staged is not None:
                try:
                    self.knowledge_storage.discard_staged(staged)
                except KnowledgeStorageError:
                    pass
            if stored_key is not None:
                try:
                    self.knowledge_storage.delete_asset(stored_key)
                except KnowledgeStorageError:
                    pass
            try:
                await self.knowledge_service.release_capture_reservation(
                    actor_id, chat_id, reservation.public_id
                )
            except KnowledgeError:
                pass
            logger.error("Knowledge Capture confirmation failed error_type=%s", type(exc).__name__)
            await self._knowledge_edit_or_send(
                query,
                "Не удалось безопасно сохранить материал. Источник не создан; "
                "проверь файл или повтори позже.",
                self._knowledge_navigation_markup(),
            )
            return
        await self._knowledge_edit_or_send(
            query,
            "<b>Материал сохранён</b>\n\n"
            f"Source ID: <code>{escape(receipt.source_public_id)}</code>\n"
            f"Статус: {escape(STATUS_LABELS.get(receipt.processing_status, receipt.processing_status))}\n\n"
            "Тяжёлая обработка выполняется отдельным runner. Если он остановлен, материал "
            "останется в очереди и не потеряется.",
            self._knowledge_navigation_markup(),
            parse_mode="HTML",
        )

    async def _capture_chunks(self, context: Any, material: Any) -> AsyncIterator[bytes]:
        if material.capture_kind in {"text", "forward"}:
            if not material.text_content:
                raise KnowledgeCaptureError("Текст Capture пуст.")
            yield material.text_content.encode("utf-8")
            return
        if material.capture_kind == "url":
            if not material.source_url:
                raise KnowledgeCaptureError("Ссылка Capture пуста.")
            yield material.source_url.encode("utf-8")
            return
        if not material.telegram_file_id:
            raise KnowledgeCaptureError("Telegram-файл недоступен.")
        async for chunk in self._telegram_file_chunks(context.bot, material.telegram_file_id):
            yield chunk

    async def _telegram_file_chunks(self, bot: Any, telegram_file_id: str) -> AsyncIterator[bytes]:
        telegram_file = await bot.get_file(telegram_file_id)
        raw_path = getattr(telegram_file, "file_path", None)
        base = str(getattr(bot, "base_file_url", "")).rstrip("/")
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or not base:
            raise KnowledgeTelegramDownloadError("invalid_telegram_path")
        base_parts = urlsplit(base)
        if not self._safe_telegram_url(base_parts, require_file_path=False):
            raise KnowledgeTelegramDownloadError("invalid_telegram_base")
        base_prefix = base_parts.path.rstrip("/")
        if not base_prefix.startswith("/file/bot"):
            raise KnowledgeTelegramDownloadError("invalid_telegram_base")

        raw_parts = urlsplit(raw_path)
        if raw_parts.scheme:
            if not self._safe_telegram_url(raw_parts, require_file_path=True):
                raise KnowledgeTelegramDownloadError("invalid_telegram_path")
            if not raw_parts.path.startswith(f"{base_prefix}/"):
                raise KnowledgeTelegramDownloadError("invalid_telegram_path")
            url = raw_path
        else:
            if raw_parts.netloc or raw_parts.query or raw_parts.fragment:
                raise KnowledgeTelegramDownloadError("invalid_telegram_path")
            path = PurePosixPath(raw_parts.path)
            if (
                path.is_absolute()
                or not path.parts
                or any(
                    unquote(part) in {"", ".", ".."}
                    or "/" in unquote(part)
                    or "\\" in unquote(part)
                    for part in path.parts
                )
            ):
                raise KnowledgeTelegramDownloadError("invalid_telegram_path")
            encoded_path = "/".join(quote(unquote(part), safe="") for part in path.parts)
            url = f"{base}/{encoded_path}"
        final_parts = urlsplit(url)
        if not self._safe_telegram_url(
            final_parts, require_file_path=True
        ) or not final_parts.path.startswith(f"{base_prefix}/"):
            raise KnowledgeTelegramDownloadError("invalid_telegram_path")
        timeout = httpx.Timeout(
            _DOWNLOAD_TIMEOUT_SECONDS,
            connect=10,
            read=30,
            write=10,
            pool=10,
        )
        async with httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            timeout=timeout,
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise KnowledgeTelegramDownloadError("telegram_download_failed")
                async for chunk in response.aiter_bytes(64 * 1024):
                    if chunk:
                        yield bytes(chunk)

    @staticmethod
    def _safe_telegram_url(parts: Any, *, require_file_path: bool) -> bool:
        try:
            port = parts.port
        except ValueError:
            return False
        if (
            parts.scheme.casefold() != "https"
            or (parts.hostname or "").casefold() != "api.telegram.org"
            or port is not None
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            return False
        if not require_file_path:
            return True
        if not parts.path.startswith("/"):
            return False
        decoded_parts = tuple(unquote(part) for part in parts.path.split("/")[1:])
        return not any(
            part in {"", ".", ".."} or "/" in part or "\\" in part for part in decoded_parts
        )

    async def _handle_knowledge_input(
        self,
        update: Update,
        actor_id: int,
        chat_id: int,
        claim: Any,
        text: str,
    ) -> bool:
        if claim.action != "input_title" or claim.capture_draft_public_id is None:
            await update.effective_message.reply_text(
                "Поле Capture устарело. Открой /knowledge и начни заново."
            )
            return True
        state = await self.knowledge_service.capture_state(actor_id, chat_id)
        preview = state.preview
        if preview is None or preview.draft_public_id != claim.capture_draft_public_id:
            await update.effective_message.reply_text(
                "Capture устарел. Сообщение не отправлено в LLM."
            )
            return True
        try:
            updated = await self.knowledge_service.update_capture(
                actor_id,
                chat_id,
                preview.draft_public_id,
                preview.version,
                title=text,
            )
        except KnowledgeError as exc:
            await self.knowledge_service.issue_action(
                actor_id,
                chat_id,
                "input_title",
                self._capture_space_id(preview),
                capture_draft_public_id=preview.draft_public_id,
                status="awaiting_input",
                ttl=self._knowledge_action_ttl(),
            )
            await update.effective_message.reply_text(str(exc))
            return True
        await self._send_capture_preview(update.effective_message, actor_id, chat_id, updated)
        return True

    async def _set_text_payload(
        self,
        actor_id: int,
        chat_id: int,
        preview: KnowledgeCapturePreview,
        text: str,
        update: Update,
    ) -> KnowledgeCapturePreview:
        raw = text.strip()
        url = self._exact_url(raw)
        is_forward = getattr(update.effective_message, "forward_origin", None) is not None
        return await self.knowledge_service.set_capture_payload(
            actor_id,
            chat_id,
            preview.draft_public_id,
            preview.version,
            capture_kind=("forward" if is_forward else "url" if url else "text"),
            text_content=None if url else raw,
            source_url=url,
            telegram_message_id=getattr(update.effective_message, "message_id", None),
            provenance=self._telegram_provenance(update.effective_message),
            safe_display_name="link.url" if url else "message.txt",
            declared_mime="text/uri-list" if url else "text/plain",
            declared_size_bytes=len(raw.encode("utf-8")),
        )

    async def _capture_for_claim(
        self, actor_id: int, chat_id: int, claim: Any
    ) -> KnowledgeCapturePreview:
        state = await self.knowledge_service.capture_state(actor_id, chat_id)
        if (
            state.preview is None
            or claim.capture_draft_public_id is None
            or state.preview.draft_public_id != claim.capture_draft_public_id
        ):
            raise KnowledgeStaleError("Capture устарел.")
        return state.preview

    async def _source_for_claim(self, actor_id: int, claim: Any) -> KnowledgeSourceRecord:
        if claim.source_public_id is None:
            raise KnowledgeStaleError("Материал устарел.")
        return await self.knowledge_service.get_source(
            actor_id,
            claim.source_public_id,
            include_trashed=True,
        )

    async def _knowledge_action(
        self,
        actor_id: int,
        chat_id: int,
        action: str,
        space_public_id: str,
        *,
        capture_draft_public_id: str | None = None,
        source_public_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        issued = await self.knowledge_service.issue_action(
            actor_id,
            chat_id,
            action,
            space_public_id,
            capture_draft_public_id=capture_draft_public_id,
            source_public_id=source_public_id,
            payload=payload,
            ttl=self._knowledge_action_ttl(),
        )
        return issued.token

    def _knowledge_capture_ttl(self) -> timedelta:
        return timedelta(minutes=self.settings.knowledge_capture_ttl_minutes)

    def _knowledge_action_ttl(self) -> timedelta:
        return timedelta(minutes=self.settings.knowledge_action_ttl_minutes)

    async def _knowledge_medical_flow(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> str | None:
        user_data = getattr(context, "user_data", {})
        if "health_checkin" in user_data:
            return "health"
        if "doctor_prepare" in user_data:
            return "doctor"
        effective_user = getattr(update, "effective_user", None)
        effective_chat = getattr(update, "effective_chat", None)
        if effective_user is None or effective_chat is None:
            return None
        user = await self._user(effective_user.id)
        chat_id = effective_chat.id
        if await self.lab_uploads.has_active(user.id, chat_id):
            return "labs"
        if self._owned_document_edit(context, user.id, chat_id):
            return "labs"
        return None

    async def _knowledge_specialized_flow(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> str | None:
        user_data = getattr(context, "user_data", {})
        for key, name in (
            ("health_checkin", "health"),
            ("doctor_prepare", "doctor"),
            ("onboarding_user_id", "onboarding"),
            ("evening", "evening"),
            ("rename_goal_id", "rename_goal"),
        ):
            if key in user_data:
                return name
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        if await self.lab_uploads.has_active(user.id, chat_id):
            return "labs"
        if self._owned_document_edit(context, user.id, chat_id):
            return "labs"
        if await self.vision_image_sessions.has_active(user.id, chat_id):
            return "vision_image"
        if await self.vision_service.draft(user.id, chat_id) is not None:
            return "vision"
        if self._workspace_enabled() and await self.workspace_service.pending_input(
            user.id, chat_id
        ):
            return "workspace"
        if await self.task_service.pending_input(user.id, chat_id) is not None:
            return "task"
        if await self.collection_service.pending_input(user.id, chat_id) is not None:
            return "collection"
        return None

    @staticmethod
    def _telegram_media_metadata(message: Any) -> dict[str, Any]:
        photos = list(getattr(message, "photo", None) or ())
        if photos:
            media = photos[-1]
            file_id = getattr(media, "file_id", None)
            if not isinstance(file_id, str) or not file_id:
                raise KnowledgeCaptureError("Telegram-файл недоступен.")
            unique = getattr(media, "file_unique_id", None)
            return {
                "capture_kind": "image",
                "file_id": file_id,
                "file_unique_id_hash": KnowledgeHandlers._opaque_file_hash(unique),
                "declared_mime": "image/jpeg",
                "safe_display_name": "telegram-photo.jpg",
                "declared_size_bytes": getattr(media, "file_size", None),
            }
        media = getattr(message, "document", None)
        file_id = getattr(media, "file_id", None) if media is not None else None
        if not isinstance(file_id, str) or not file_id:
            raise KnowledgeCaptureError("Telegram-файл недоступен.")
        declared_mime = getattr(media, "mime_type", None)
        return {
            "capture_kind": (
                "image"
                if isinstance(declared_mime, str) and declared_mime.startswith("image/")
                else "document"
            ),
            "file_id": file_id,
            "file_unique_id_hash": KnowledgeHandlers._opaque_file_hash(
                getattr(media, "file_unique_id", None)
            ),
            "declared_mime": declared_mime,
            "safe_display_name": KnowledgeHandlers._safe_display_name(
                getattr(media, "file_name", None)
            ),
            "declared_size_bytes": getattr(media, "file_size", None),
        }

    @staticmethod
    def _telegram_provenance(message: Any) -> dict[str, Any]:
        origin = getattr(message, "forward_origin", None)
        if origin is None:
            return {"kind": "telegram_upload"}
        forwarded_at = getattr(origin, "date", None)
        return {
            "kind": "telegram_forward",
            "origin_type": type(origin).__name__[:40],
            "forwarded_at": (
                forwarded_at.isoformat() if hasattr(forwarded_at, "isoformat") else None
            ),
        }

    @staticmethod
    def _opaque_file_hash(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_display_name(value: Any) -> str:
        if not isinstance(value, str):
            return "telegram-document.bin"
        clean = "".join(
            character
            for character in value
            if character.isprintable() and character not in {"/", "\\"}
        )
        clean = " ".join(clean.split()).strip(" .")
        return clean[:255].strip() or "telegram-document.bin"

    @staticmethod
    def _exact_url(value: str) -> str | None:
        if not value or len(value) > 4096 or any(character.isspace() for character in value):
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        return value

    @staticmethod
    def _capture_space_id(preview: KnowledgeCapturePreview) -> str:
        if not preview.target_space_public_id:
            raise KnowledgeCaptureError("Пространство Capture недоступно.")
        return preview.target_space_public_id

    @staticmethod
    def _capture_warning(kind: str) -> str:
        if kind == "url":
            return "Страница не скачивается: URL хранится как ссылка без preview и redirect."
        if kind == "image":
            return "Оригинал будет сохранён без OCR и без распознавания содержимого."
        if kind == "document":
            return "Извлечение поддерживает PDF с текстовым слоем, TXT, Markdown, DOCX и EPUB."
        return "Материал станет источником только после явного подтверждения."

    @staticmethod
    def _default_display_name(kind: str) -> str:
        return {
            "text": "message.txt",
            "forward": "forward.txt",
            "url": "link.url",
            "image": "image.bin",
            "document": "document.bin",
        }.get(kind, "material.bin")

    @staticmethod
    def _button_label(value: str, *, maximum: int = _MAX_BUTTON_CHARS) -> str:
        clean = " ".join(value.split())
        return clean if len(clean) <= maximum else f"{clean[: maximum - 1].rstrip()}…"

    @staticmethod
    def _truncate(value: str | None, maximum: int) -> str:
        if not value:
            return ""
        clean = " ".join(value.split())
        return clean if len(clean) <= maximum else f"{clean[: maximum - 1].rstrip()}…"

    @staticmethod
    def _space_icon(kind: str) -> str:
        return {"personal": "🔒", "workspace": "🤝", "project": "📁"}.get(kind, "📚")

    @staticmethod
    def _status_icon(status: str) -> str:
        return {
            "queued": "⏳",
            "processing": "⚙️",
            "ready": "✅",
            "partial": "🟡",
            "failed": "⚠️",
            "quarantined": "🛡",
            "cancelled": "⏹",
        }.get(status, "•")

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value < 1024:
            return f"{value} Б"
        if value < 1024 * 1024:
            return f"{value / 1024:.1f} КБ"
        return f"{value / (1024 * 1024):.1f} МБ"

    @staticmethod
    def _knowledge_navigation_rows() -> list[list[InlineKeyboardButton]]:
        return [
            [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            [InlineKeyboardButton("❓ Помощь", callback_data="nav:help")],
        ]

    @classmethod
    def _knowledge_navigation_markup(cls) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(cls._knowledge_navigation_rows())

    @staticmethod
    async def _knowledge_stale(query: Any) -> None:
        await query.answer("Эта кнопка устарела или недоступна.", show_alert=True)

    async def _knowledge_stale_message(self, query: Any) -> None:
        await self._knowledge_edit_or_send(
            query,
            "Действие устарело или доступ изменился. Открой /knowledge ещё раз.",
            self._knowledge_navigation_markup(),
        )

    @staticmethod
    async def _knowledge_edit_or_send(
        query: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        parse_mode: str | None = None,
    ) -> None:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
        except (TelegramError, TypeError):
            await query.message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
