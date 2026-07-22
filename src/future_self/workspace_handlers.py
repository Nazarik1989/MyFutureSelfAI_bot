from __future__ import annotations

from html import escape
from typing import Any
from urllib.parse import quote

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes
from telegram.helpers import create_deep_linked_url

from .workspace_access import (
    AccessContext,
    InvitationActionResult,
    InvitationPreview,
    WorkspaceAccessDenied,
    WorkspaceAccessError,
    WorkspaceAccessService,
    WorkspaceConflictError,
    WorkspaceInvitationError,
    WorkspaceLastOwnerError,
    WorkspaceStaleError,
    clean_invitation_text,
    clean_workspace_description,
    clean_workspace_name,
)

CHARACTER_LABELS = {
    "pair": "Для пары",
    "friends": "Для друзей",
    "family": "Для семьи",
    "team": "Для команды",
    "custom": "Свой вариант",
}

CHARACTER_SUGGESTIONS = {
    "pair": ("Наше будущее", "Мы вместе"),
    "friends": ("Наши приключения", "Мы вместе"),
    "family": ("Семья", "Наш дом"),
    "team": ("Команда", "Общее будущее"),
    "custom": ("Наше пространство",),
}

ROLE_LABELS = {"owner": "Владелец", "editor": "Редактор", "viewer": "Читатель"}

INVITATION_TEMPLATES = {
    "pair": (
        "{inviter} приглашает тебя вместе сохранять мечты, планы и моменты вашего будущего ✨",
        "{inviter} хочет создать с тобой пространство «{workspace}» 💞",
        "Давай соберём в одном месте наши общие желания, идеи и визуализации?",
        "{inviter} приглашает тебя в пространство «{workspace}».",
    ),
    "friends": (
        "{inviter} приглашает тебя в пространство общих идей, приключений и планов 🚀",
        "Давай вместе собирать мечты, поездки и всё, что однажды стоит осуществить.",
        "{inviter} создаёт пространство «{workspace}» и зовёт тебя присоединиться.",
        "{inviter} приглашает тебя в пространство «{workspace}».",
    ),
    "family": (
        "{inviter} приглашает тебя в семейное пространство для общих планов, целей и "
        "воспоминаний 🏡",
        "Давайте вместе создавать будущее нашей семьи.",
        "Присоединяйся к пространству, где мы будем сохранять семейные идеи, мечты и "
        "важные события.",
        "{inviter} приглашает тебя в пространство «{workspace}».",
    ),
    "team": (
        "{inviter} приглашает тебя в общее пространство для целей, проектов и совместных решений.",
        "Присоединяйся к пространству «{workspace}» — здесь команда собирает идеи и "
        "превращает их в планы.",
        "{inviter} приглашает тебя вместе работать над общим будущим проекта 🚀",
        "{inviter} приглашает тебя в пространство «{workspace}».",
    ),
    "custom": (
        "{inviter} приглашает тебя в пространство «{workspace}».",
        "{inviter} предлагает вместе развивать пространство «{workspace}».",
        "Давай создадим общее пространство для будущих идей и планов.",
        "{inviter} приглашает тебя присоединиться.",
    ),
}

PRIVACY_FOOTER = (
    "Личные записи и визуализации не передаются автоматически. Общим становится только "
    "то, чем участники решат поделиться явно."
)


class WorkspaceHandlers:
    workspace_service: WorkspaceAccessService
    task_service: Any
    collection_service: Any

    async def spaces_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._workspace_enabled():
            await update.effective_message.reply_text("Совместные пространства сейчас выключены.")
            return
        active_flow = await self._active_navigation_flow(update, context)
        if active_flow is not None:
            await self._prompt_navigation_flow(update.effective_message, update, active_flow)
            return
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        await self.workspace_service.cancel_input(user.id, chat_id)
        await self.task_service.cancel_pending_input(user.id, chat_id)
        await self.collection_service.cancel_input(user.id, chat_id)
        await self.collection_service.clear_context(user.id, chat_id)
        await self._send_workspace_hub(update.effective_message, user.id, chat_id)

    async def workspace_start_invitation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        if not self._workspace_enabled():
            return False
        args = tuple(getattr(context, "args", ()) or ())
        if not args or not args[0].startswith("space_"):
            return False
        if len(args) != 1:
            await update.effective_message.reply_text("Приглашение недействительно.")
            return True
        raw_token = args[0].removeprefix("space_")
        if not raw_token or len(raw_token) > 48:
            await update.effective_message.reply_text("Приглашение недействительно.")
            return True
        active_flow = await self._active_navigation_flow(update, context)
        if active_flow is not None:
            await self._prompt_navigation_flow(update.effective_message, update, active_flow)
            return True
        user = await self._user(update.effective_user.id)
        try:
            incoming = await self.workspace_service.issue_incoming_actions(
                user.id, update.effective_chat.id, raw_token
            )
        except WorkspaceInvitationError:
            await update.effective_message.reply_text("Приглашение недействительно.")
            return True
        await self._send_incoming_invitation(update.effective_message, incoming)
        return True

    async def workspace_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = query.data or ""
        if data.startswith("spacei:"):
            active_flow = await self._active_navigation_flow(update, context)
            if active_flow is not None:
                await query.answer()
                await self._prompt_navigation_flow(query.message, update, active_flow)
                return
            await self._incoming_callback(update, context)
            return
        if not self._workspace_enabled() or not data.startswith("space:") or data.count(":") != 1:
            await self._workspace_stale(query)
            return
        token = data.removeprefix("space:")
        if not token or len(token) > 48:
            await self._workspace_stale(query)
            return
        active_flow = await self._active_navigation_flow(update, context)
        if active_flow is not None and active_flow != "workspace":
            await query.answer()
            await self._prompt_navigation_flow(query.message, update, active_flow)
            return
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        if active_flow == "workspace":
            await self.workspace_service.cancel_input(user.id, chat_id)
        try:
            claim = await self.workspace_service.claim_action(token, user.id, chat_id)
        except WorkspaceAccessError:
            claim = None
        if claim is None:
            await self._workspace_stale(query)
            return
        await query.answer()
        try:
            await self._dispatch_workspace_action(query, context, user, chat_id, claim)
        except WorkspaceLastOwnerError:
            await self._workspace_edit_or_send(
                query,
                "Нельзя убрать последнего владельца. Сначала назначь другого владельца.",
                None,
            )
        except WorkspaceConflictError as exc:
            await self._workspace_edit_or_send(query, escape(str(exc)), None, parse_mode="HTML")
        except (WorkspaceStaleError, WorkspaceAccessDenied, WorkspaceInvitationError):
            await self._workspace_stale_message(query)
        except WorkspaceAccessError as exc:
            await self._workspace_edit_or_send(query, escape(str(exc)), None, parse_mode="HTML")

    async def workspace_pending_text(self, update: Update, text: str, source: str) -> bool:
        del source
        if not self._workspace_enabled():
            return False
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        pending = None
        try:
            pending = await self.workspace_service.pending_input(user.id, chat_id)
            claim = (
                await self.workspace_service.claim_pending_input(user.id, chat_id, pending.action)
                if pending is not None
                else None
            )
        except WorkspaceAccessError:
            claim = None
        if claim is None:
            cancelled = await self.workspace_service.cancel_input(user.id, chat_id)
            if pending is not None or cancelled:
                await update.effective_message.reply_text(
                    "Время ввода для пространства истекло или доступ изменился. "
                    "Открой /spaces и начни операцию заново."
                )
                return True
            return False
        try:
            await self._dispatch_workspace_input(
                update.effective_message, user, chat_id, claim, text
            )
        except WorkspaceConflictError as exc:
            await self._rearm_workspace_input(user.id, chat_id, claim)
            await update.effective_message.reply_text(str(exc))
        except (WorkspaceStaleError, WorkspaceAccessDenied):
            await update.effective_message.reply_text(
                "Операция устарела или доступ изменился. Открой /spaces ещё раз."
            )
        except WorkspaceAccessError as exc:
            await self._rearm_workspace_input(user.id, chat_id, claim)
            await update.effective_message.reply_text(str(exc))
        return True

    async def cancel_workspace_state(self, update: Update) -> bool:
        if not self._workspace_enabled():
            return False
        user = await self._user(update.effective_user.id)
        return await self.workspace_service.cancel_input(user.id, update.effective_chat.id)

    async def workspace_other_callback_gate(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        del context
        if not self._workspace_enabled() or update.callback_query is None:
            return
        if (update.callback_query.data or "").startswith(("space:", "spacei:", "nav:")):
            return
        user = await self._user(update.effective_user.id)
        await self.workspace_service.cancel_input(user.id, update.effective_chat.id)

    async def handle_workspace_natural(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        action: str,
    ) -> None:
        if not self._workspace_enabled():
            await update.effective_message.reply_text("Совместные пространства сейчас выключены.")
            return
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        if action == "show_spaces":
            await self.spaces_command(update, context)
            return
        if action == "create_space":
            await self._prepare_workspace_flow(user.id, chat_id)
            await self._send_character_picker(update.effective_message, user.id, chat_id)
            return
        snapshot = await self.workspace_service.active_context(user.id, chat_id)
        if snapshot is None:
            await update.effective_message.reply_text(
                "Сначала открой пространство через /spaces — выбранный контекст здесь не найден."
            )
            return
        workspace = await self.workspace_service.get_workspace(snapshot.access_context)
        members = await self.workspace_service.list_members(snapshot.access_context)
        actor = next(record.member for record in members if record.member.user_id == user.id)
        if action in {"invite_space_member", "show_space_invitations"} and actor.role != "owner":
            await update.effective_message.reply_text(
                "Приглашениями может управлять только владелец пространства."
            )
            return
        if action == "invite_space_member":
            await self._send_invitation_roles(
                update.effective_message, user, chat_id, snapshot.access_context, workspace
            )
        elif action == "show_space_invitations":
            await self._send_invitations(
                update.effective_message, user.id, chat_id, snapshot.access_context
            )
        elif action == "show_space_members":
            await self._send_members(
                update.effective_message, user.id, chat_id, snapshot.access_context
            )

    async def _dispatch_workspace_action(
        self, query: Any, context: Any, user: Any, chat_id: int, claim: Any
    ) -> None:
        action = claim.action
        payload = claim.payload
        access = claim.access_context
        if action == "hub":
            await self._render_workspace_hub(
                query,
                user.id,
                chat_id,
                page=int(payload.get("page", 1)),
                status=str(payload.get("status", "active")),
            )
            return
        if action == "create":
            await self._prepare_workspace_flow(user.id, chat_id)
            await self._render_character_picker(query, user.id, chat_id)
            return
        if action == "choose_character":
            character = str(payload.get("character", ""))
            if character not in CHARACTER_LABELS:
                raise WorkspaceStaleError("Некорректный характер пространства.")
            await self._render_name_picker(query, user.id, chat_id, character)
            return
        if action == "custom_name":
            character = str(payload.get("character", ""))
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_create_name",
                payload={"character": character},
            )
            await self._workspace_edit_or_send(
                query,
                "Пришли название пространства (до 100 символов). /cancel — отменить ввод.",
                self._workspace_navigation_markup(),
            )
            return
        if action == "suggested_name":
            character = str(payload.get("character", ""))
            name = str(payload.get("name", ""))
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_create_description",
                payload={"character": character, "name": name},
            )
            await self._workspace_edit_or_send(
                query,
                "Добавь короткое описание или отправь «-», чтобы оставить без описания. "
                "/cancel — отменить ввод.",
                self._workspace_navigation_markup(),
            )
            return
        if action == "open":
            if access is None:
                raise WorkspaceAccessDenied("Пространство недоступно.")
            await self.workspace_service.set_context(access, chat_id)
            await self._render_workspace(query, user.id, chat_id, access)
            return
        if action == "open_archived":
            access = claim.access_context
            if access is None:
                raise WorkspaceAccessDenied("Пространство недоступно.")
            await self._render_workspace(query, user.id, chat_id, access)
            return
        if action == "context_personal":
            await self.workspace_service.ensure_personal_knowledge_space(user.id)
            await self.workspace_service.clear_context(user.id, chat_id)
            await self._render_workspace_hub(
                query, user.id, chat_id, notice="Включён личный контекст."
            )
            return
        if action == "restore":
            access = claim.access_context
            if access is None:
                raise WorkspaceAccessDenied("Пространство недоступно.")
            await self.workspace_service.set_workspace_archived(
                access, int(claim.workspace_version or 0), archived=False
            )
            refreshed = await self.workspace_service.access_context(user.id, access.workspace_id)
            await self.workspace_service.set_context(refreshed, chat_id)
            await self._render_workspace(
                query, user.id, chat_id, refreshed, notice="Пространство восстановлено."
            )
            return
        if access is None:
            raise WorkspaceAccessDenied("Пространство недоступно.")
        if action == "members":
            await self._render_members(query, user.id, chat_id, access)
        elif action == "member":
            await self._render_member(query, user.id, chat_id, access, payload)
        elif action == "member_role":
            await self.workspace_service.change_member_role(
                access,
                int(payload["member_user_id"]),
                str(payload["role"]),
                int(payload["member_version"]),
            )
            refreshed = await self.workspace_service.access_context(user.id, access.workspace_id)
            await self._render_members(
                query, user.id, chat_id, refreshed, notice="Роль участника обновлена."
            )
        elif action == "member_revoke":
            await self.workspace_service.revoke_member(
                access,
                int(payload["member_user_id"]),
                int(payload["member_version"]),
            )
            refreshed = await self.workspace_service.access_context(user.id, access.workspace_id)
            await self._render_members(
                query, user.id, chat_id, refreshed, notice="Доступ участника отозван."
            )
        elif action == "invite_start":
            workspace = await self.workspace_service.get_workspace(access)
            await self._render_invitation_roles(query, user, chat_id, access, workspace)
        elif action == "invite_role":
            workspace = await self.workspace_service.get_workspace(access)
            await self._render_invitation_templates(
                query, user, chat_id, access, workspace, str(payload["role"])
            )
        elif action in {"invite_template", "invite_next"}:
            workspace = await self.workspace_service.get_workspace(access)
            await self._render_invitation_preview(
                query,
                user,
                chat_id,
                access,
                workspace,
                role=str(payload["role"]),
                template_index=int(payload.get("template_index", 0)),
                custom_text=payload.get("custom_text"),
            )
        elif action == "invite_edit":
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_invite_text",
                payload={"role": str(payload["role"])},
                access=access,
                workspace_version=claim.workspace_version,
            )
            await self._workspace_edit_or_send(
                query,
                "Пришли свой текст приглашения (до 1000 символов). Privacy-уведомление "
                "будет добавлено автоматически. /cancel — отменить ввод.",
                self._workspace_navigation_markup(),
            )
        elif action == "invite_confirm":
            await self._confirm_share_invitation(query, context, user, chat_id, access, payload)
        elif action == "invitations":
            await self._render_invitations(query, user.id, chat_id, access)
        elif action == "invitation":
            await self._render_invitation_manage(query, user.id, chat_id, access, payload)
        elif action == "invite_revoke":
            await self.workspace_service.revoke_invitation(
                access, int(payload["invitation_id"]), int(payload["invitation_version"])
            )
            await self._render_invitations(
                query, user.id, chat_id, access, notice="Приглашение отозвано."
            )
        elif action == "invite_renew":
            issued = await self.workspace_service.renew_invitation(
                access, int(payload["invitation_id"]), int(payload["invitation_version"])
            )
            if issued.invitation.delivery_mode == "share":
                await self._show_issued_invitation(query, context, issued.token, renewed=True)
            else:
                await self._workspace_edit_or_send(
                    query,
                    "Адресное приглашение обновлено и по-прежнему привязано к выбранному "
                    "пользователю. Автоматическая отправка не выполнялась.",
                    None,
                )
        elif action == "rename":
            workspace = await self.workspace_service.get_workspace(access)
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_rename",
                payload={
                    "description": workspace.description,
                    "character": workspace.character,
                },
                access=access,
                workspace_version=workspace.version,
            )
            await self._workspace_edit_or_send(
                query,
                "Пришли новое название (до 100 символов). /cancel — отменить ввод.",
                self._workspace_navigation_markup(),
            )
        elif action == "description":
            workspace = await self.workspace_service.get_workspace(access)
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_description",
                payload={"name": workspace.name, "character": workspace.character},
                access=access,
                workspace_version=workspace.version,
            )
            await self._workspace_edit_or_send(
                query,
                "Пришли новое описание или «-», чтобы удалить его. /cancel — отменить ввод.",
                self._workspace_navigation_markup(),
            )
        elif action == "character":
            workspace = await self.workspace_service.get_workspace(access)
            await self._render_character_change(query, user.id, chat_id, access, workspace)
        elif action == "character_set":
            workspace = await self.workspace_service.get_workspace(access)
            updated = await self.workspace_service.rename_workspace(
                access,
                int(claim.workspace_version or workspace.version),
                workspace.name,
                description=workspace.description,
                character=str(payload["character"]),
            )
            refreshed = await self.workspace_service.access_context(user.id, updated.id)
            await self.workspace_service.set_context(refreshed, chat_id)
            await self._render_workspace(
                query, user.id, chat_id, refreshed, notice="Характер пространства обновлён."
            )
        elif action == "archive_ask":
            await self._render_archive_confirm(query, user.id, chat_id, access)
        elif action == "archive":
            await self.workspace_service.set_workspace_archived(
                access, int(claim.workspace_version or 0), archived=True
            )
            await self.workspace_service.clear_context(user.id, chat_id)
            await self._render_workspace_hub(
                query, user.id, chat_id, notice="Пространство перемещено в архив."
            )
        elif action == "leave_ask":
            await self._render_leave_confirm(query, user.id, chat_id, access)
        elif action == "leave":
            await self.workspace_service.leave_workspace(access, int(payload["member_version"]))
            await self.workspace_service.clear_context(user.id, chat_id)
            await self._render_workspace_hub(
                query, user.id, chat_id, notice="Ты вышел из пространства."
            )
        elif action == "projects":
            await self._render_projects(query, user.id, chat_id, access)
        elif action == "projects_archived":
            await self._render_projects(query, user.id, chat_id, access, status="archived")
        elif action == "project_create":
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_project_create",
                access=access,
                workspace_version=claim.workspace_version,
            )
            await self._workspace_edit_or_send(
                query,
                "Пришли название проекта (до 100 символов). /cancel — отменить ввод.",
                self._workspace_navigation_markup(),
            )
        elif action == "project":
            await self._render_project(query, user.id, chat_id, access, claim)
        elif action == "project_context":
            if claim.workspace_project_id is None:
                raise WorkspaceStaleError("Проект уже изменился.")
            snapshot = await self.workspace_service.set_context(
                access, chat_id, workspace_project_id=claim.workspace_project_id
            )
            await self._render_project(
                query,
                user.id,
                chat_id,
                access,
                claim,
                notice=f"Выбран контекст проекта «{snapshot.workspace_project.name}».",
            )
        elif action == "project_rename":
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_project_rename",
                access=access,
                workspace_version=claim.workspace_version,
                workspace_project_id=claim.workspace_project_id,
                workspace_project_version=claim.workspace_project_version,
            )
            await self._workspace_edit_or_send(
                query,
                "Пришли новое название проекта. /cancel — отменить ввод.",
                self._workspace_navigation_markup(),
            )
        elif action == "project_archive":
            await self.workspace_service.set_project_archived(
                access,
                int(claim.workspace_project_id or 0),
                int(claim.workspace_project_version or 0),
                archived=True,
            )
            await self._render_projects(
                query, user.id, chat_id, access, notice="Проект архивирован."
            )
        elif action == "project_restore":
            if claim.workspace_project_id is None:
                raise WorkspaceStaleError("Проект уже изменился.")
            await self.workspace_service.set_project_archived(
                access,
                claim.workspace_project_id,
                int(claim.workspace_project_version or 0),
                archived=False,
            )
            await self._render_projects(
                query, user.id, chat_id, access, notice="Проект восстановлен."
            )
        elif action == "context_workspace":
            await self.workspace_service.set_context(access, chat_id)
            await self._render_workspace(
                query, user.id, chat_id, access, notice="Выбран контекст пространства."
            )
        else:
            raise WorkspaceStaleError("Действие недоступно.")

    async def _dispatch_workspace_input(
        self, message: Any, user: Any, chat_id: int, claim: Any, text: str
    ) -> None:
        action = claim.action.removeprefix("input:")
        payload = claim.payload
        access = claim.access_context
        if action == "create_name":
            name, _ = clean_workspace_name(text)
            await self._begin_workspace_input(
                user.id,
                chat_id,
                "input_create_description",
                payload={"character": str(payload["character"]), "name": name},
            )
            await message.reply_text(
                "Добавь короткое описание или отправь «-», чтобы оставить без описания. "
                "/cancel — отменить ввод.",
                reply_markup=self._workspace_navigation_markup(),
            )
            return
        if action == "create_description":
            description = None if text.strip() == "-" else clean_workspace_description(text)
            workspace = await self.workspace_service.create_workspace(
                user.id,
                str(payload["character"]),
                str(payload["name"]),
                description,
            )
            access = await self.workspace_service.access_context(user.id, workspace.id)
            await self.workspace_service.set_context(access, chat_id)
            await self._send_workspace(
                message,
                user.id,
                chat_id,
                access,
                notice="Пространство создано. Теперь можно пригласить участника.",
            )
            return
        if access is None:
            raise WorkspaceAccessDenied("Пространство недоступно.")
        if action == "invite_text":
            custom = clean_invitation_text(text)
            if custom is None:
                raise WorkspaceAccessError("Текст приглашения не может быть пустым.")
            workspace = await self.workspace_service.get_workspace(access)
            await self._send_invitation_preview(
                message,
                user,
                chat_id,
                access,
                workspace,
                role=str(payload["role"]),
                template_index=0,
                custom_text=custom,
            )
        elif action == "rename":
            updated = await self.workspace_service.rename_workspace(
                access,
                int(claim.workspace_version or 0),
                text,
                description=payload.get("description"),
                character=str(payload["character"]),
            )
            refreshed = await self.workspace_service.access_context(user.id, updated.id)
            await self.workspace_service.set_context(refreshed, chat_id)
            await self._send_workspace(
                message, user.id, chat_id, refreshed, notice="Пространство переименовано."
            )
        elif action == "description":
            description = None if text.strip() == "-" else clean_workspace_description(text)
            updated = await self.workspace_service.rename_workspace(
                access,
                int(claim.workspace_version or 0),
                str(payload["name"]),
                description=description,
                character=str(payload["character"]),
            )
            refreshed = await self.workspace_service.access_context(user.id, updated.id)
            await self.workspace_service.set_context(refreshed, chat_id)
            await self._send_workspace(
                message, user.id, chat_id, refreshed, notice="Описание обновлено."
            )
        elif action == "project_create":
            await self.workspace_service.create_project(access, text)
            await self._send_projects(message, user.id, chat_id, access, notice="Проект создан.")
        elif action == "project_rename":
            if claim.workspace_project_id is None:
                raise WorkspaceStaleError("Проект уже изменился.")
            await self.workspace_service.rename_project(
                access,
                claim.workspace_project_id,
                int(claim.workspace_project_version or 0),
                text,
            )
            await self._send_projects(
                message, user.id, chat_id, access, notice="Проект переименован."
            )
        else:
            raise WorkspaceStaleError("Операция ввода устарела.")

    async def _prepare_workspace_flow(self, actor_id: int, chat_id: int) -> None:
        await self.workspace_service.cancel_input(actor_id, chat_id)
        await self.task_service.cancel_pending_input(actor_id, chat_id)
        await self.collection_service.cancel_input(actor_id, chat_id)
        await self.collection_service.clear_context(actor_id, chat_id)
        await self.lab_uploads.cancel_active(actor_id, chat_id)
        await self.vision_image_sessions.cancel_active(actor_id, chat_id)
        await self.vision_service.cancel(actor_id, chat_id)

    async def _send_workspace_hub(
        self,
        message: Any,
        actor_id: int,
        chat_id: int,
        *,
        page: int = 1,
        status: str = "active",
        notice: str | None = None,
    ) -> None:
        text, markup = await self._workspace_hub_view(
            actor_id, chat_id, page=page, status=status, notice=notice
        )
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_workspace_hub(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        *,
        page: int = 1,
        status: str = "active",
        notice: str | None = None,
    ) -> None:
        text, markup = await self._workspace_hub_view(
            actor_id, chat_id, page=page, status=status, notice=notice
        )
        await self._workspace_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _workspace_hub_view(
        self,
        actor_id: int,
        chat_id: int,
        *,
        page: int,
        status: str,
        notice: str | None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        safe_status = "archived" if status == "archived" else "active"
        listing = await self.workspace_service.list_workspaces(
            actor_id, page=max(page, 1), status=safe_status
        )
        heading = "Архив пространств" if safe_status == "archived" else "Совместные пространства"
        lines = [f"<b>{heading}</b>"]
        if notice:
            lines.extend(("", escape(notice)))
        if not listing.items:
            lines.extend(
                (
                    "",
                    "В архиве пока пусто."
                    if safe_status == "archived"
                    else "Здесь пока нет пространств. Создай первое — ничего личного "
                    "не станет общим автоматически.",
                )
            )
        else:
            lines.extend(("", f"Страница {listing.page} из {listing.pages}."))
        rows: list[list[InlineKeyboardButton]] = []
        for workspace in listing.items:
            if safe_status == "active":
                access = await self.workspace_service.access_context(actor_id, workspace.id)
                token = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "open",
                    access=access,
                    workspace_version=workspace.version,
                )
            else:
                access = await self.workspace_service.access_context(actor_id, workspace.id)
                token = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "open_archived",
                    access=access,
                    workspace_version=workspace.version,
                )
            rows.append(
                [
                    InlineKeyboardButton(
                        self._button_label(
                            f"{self._character_emoji(workspace.character)} {workspace.name}"
                        ),
                        callback_data=f"space:{token}",
                    )
                ]
            )
        navigation: list[InlineKeyboardButton] = []
        if listing.page > 1:
            back = await self._workspace_action(
                actor_id,
                chat_id,
                "hub",
                payload={"page": listing.page - 1, "status": safe_status},
            )
            navigation.append(InlineKeyboardButton("←", callback_data=f"space:{back}"))
        if listing.page < listing.pages:
            next_token = await self._workspace_action(
                actor_id,
                chat_id,
                "hub",
                payload={"page": listing.page + 1, "status": safe_status},
            )
            navigation.append(InlineKeyboardButton("→", callback_data=f"space:{next_token}"))
        if navigation:
            rows.append(navigation)
        if safe_status == "active":
            create = await self._workspace_action(actor_id, chat_id, "create")
            archived = await self._workspace_action(
                actor_id, chat_id, "hub", payload={"status": "archived", "page": 1}
            )
            personal = await self._workspace_action(actor_id, chat_id, "context_personal")
            rows.extend(
                (
                    [InlineKeyboardButton("＋ Создать", callback_data=f"space:{create}")],
                    [
                        InlineKeyboardButton("Архив", callback_data=f"space:{archived}"),
                        InlineKeyboardButton("Личный контекст", callback_data=f"space:{personal}"),
                    ],
                )
            )
        else:
            active = await self._workspace_action(
                actor_id, chat_id, "hub", payload={"status": "active", "page": 1}
            )
            rows.append(
                [InlineKeyboardButton("← К пространствам", callback_data=f"space:{active}")]
            )
        rows.append([InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")])
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    async def _send_character_picker(self, message: Any, actor_id: int, chat_id: int) -> None:
        text, markup = await self._character_picker_view(actor_id, chat_id)
        await message.reply_text(text, reply_markup=markup)

    async def _render_character_picker(self, query: Any, actor_id: int, chat_id: int) -> None:
        text, markup = await self._character_picker_view(actor_id, chat_id)
        await self._workspace_edit_or_send(query, text, markup)

    async def _character_picker_view(
        self, actor_id: int, chat_id: int
    ) -> tuple[str, InlineKeyboardMarkup]:
        rows: list[list[InlineKeyboardButton]] = []
        for character, label in CHARACTER_LABELS.items():
            token = await self._workspace_action(
                actor_id,
                chat_id,
                "choose_character",
                payload={"character": character},
            )
            rows.append([InlineKeyboardButton(label, callback_data=f"space:{token}")])
        hub = await self._workspace_action(actor_id, chat_id, "hub")
        rows.append([InlineKeyboardButton("Отмена", callback_data=f"space:{hub}")])
        return (
            "Какой характер у пространства? Это меняет только оформление и тексты, но не права доступа.",
            InlineKeyboardMarkup(rows),
        )

    async def _render_name_picker(
        self, query: Any, actor_id: int, chat_id: int, character: str
    ) -> None:
        rows: list[list[InlineKeyboardButton]] = []
        for name in CHARACTER_SUGGESTIONS[character]:
            token = await self._workspace_action(
                actor_id,
                chat_id,
                "suggested_name",
                payload={"character": character, "name": name},
            )
            rows.append([InlineKeyboardButton(name, callback_data=f"space:{token}")])
        custom = await self._workspace_action(
            actor_id,
            chat_id,
            "custom_name",
            payload={"character": character},
        )
        back = await self._workspace_action(actor_id, chat_id, "create")
        rows.extend(
            (
                [InlineKeyboardButton("Своё название", callback_data=f"space:{custom}")],
                [InlineKeyboardButton("← Назад", callback_data=f"space:{back}")],
            )
        )
        await self._workspace_edit_or_send(
            query,
            f"Выбери название для «{CHARACTER_LABELS[character]}» или введи своё.",
            InlineKeyboardMarkup(rows),
        )

    async def _send_workspace(
        self,
        message: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        notice: str | None = None,
    ) -> None:
        text, markup = await self._workspace_view(actor_id, chat_id, access, notice=notice)
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_workspace(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        notice: str | None = None,
    ) -> None:
        text, markup = await self._workspace_view(actor_id, chat_id, access, notice=notice)
        await self._workspace_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _workspace_view(
        self,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        notice: str | None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        workspace = await self.workspace_service.get_workspace(access)
        members = await self.workspace_service.list_members(access)
        actor = next(
            (record.member for record in members if record.member.user_id == actor_id),
            None,
        )
        if actor is None:
            raise WorkspaceAccessDenied("Пространство недоступно.")
        lines = [
            f"<b>{self._character_emoji(workspace.character)} {escape(workspace.name)}</b>",
            f"Характер: {escape(CHARACTER_LABELS[workspace.character])}",
            f"Твоя роль: {escape(ROLE_LABELS[actor.role])}",
        ]
        if workspace.description:
            lines.append(escape(workspace.description))
        if notice:
            lines.extend(("", escape(notice)))
        if workspace.status == "active":
            snapshot = await self.workspace_service.active_context(actor_id, chat_id)
            if snapshot is not None and snapshot.access_context.workspace_id == workspace.id:
                context_label = (
                    f"проект «{escape(snapshot.workspace_project.name)}»"
                    if snapshot.workspace_project is not None
                    else "всё пространство"
                )
                lines.extend(("", f"Текущий контекст: {context_label}."))
        rows: list[list[InlineKeyboardButton]] = []
        if workspace.status == "archived":
            owner_count = sum(record.member.role == "owner" for record in members)
            if actor.role == "owner":
                restore = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "restore",
                    access=access,
                    workspace_version=workspace.version,
                )
                rows.append(
                    [InlineKeyboardButton("Восстановить", callback_data=f"space:{restore}")]
                )
            if actor.role != "owner" or owner_count > 1:
                leave = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "leave_ask",
                    access=access,
                    workspace_version=workspace.version,
                    payload={"member_version": actor.version},
                )
                rows.append([InlineKeyboardButton("Выйти", callback_data=f"space:{leave}")])
            else:
                lines.extend(
                    (
                        "",
                        "Чтобы выйти, сначала назначь другого участника владельцем.",
                    )
                )
            archived = await self._workspace_action(
                actor_id,
                chat_id,
                "hub",
                payload={"status": "archived", "page": 1},
            )
            rows.append([InlineKeyboardButton("← В архив", callback_data=f"space:{archived}")])
            return "\n".join(lines), InlineKeyboardMarkup(rows)

        members_token = await self._workspace_action(
            actor_id,
            chat_id,
            "members",
            access=access,
            workspace_version=workspace.version,
        )
        projects = await self._workspace_action(
            actor_id,
            chat_id,
            "projects",
            access=access,
            workspace_version=workspace.version,
        )
        context_token = await self._workspace_action(
            actor_id,
            chat_id,
            "context_workspace",
            access=access,
            workspace_version=workspace.version,
        )
        rows.extend(
            (
                [InlineKeyboardButton("Участники", callback_data=f"space:{members_token}")],
                [InlineKeyboardButton("Проекты", callback_data=f"space:{projects}")],
                [
                    InlineKeyboardButton(
                        "Выбрать контекст пространства",
                        callback_data=f"space:{context_token}",
                    )
                ],
            )
        )
        if actor.role == "owner":
            invitations = await self._workspace_action(
                actor_id,
                chat_id,
                "invitations",
                access=access,
                workspace_version=workspace.version,
            )
            invite = await self._workspace_action(
                actor_id,
                chat_id,
                "invite_start",
                access=access,
                workspace_version=workspace.version,
            )
            rename = await self._workspace_action(
                actor_id,
                chat_id,
                "rename",
                access=access,
                workspace_version=workspace.version,
            )
            description = await self._workspace_action(
                actor_id,
                chat_id,
                "description",
                access=access,
                workspace_version=workspace.version,
            )
            character = await self._workspace_action(
                actor_id,
                chat_id,
                "character",
                access=access,
                workspace_version=workspace.version,
            )
            archive = await self._workspace_action(
                actor_id,
                chat_id,
                "archive_ask",
                access=access,
                workspace_version=workspace.version,
            )
            rows.extend(
                (
                    [
                        InlineKeyboardButton("Пригласить", callback_data=f"space:{invite}"),
                        InlineKeyboardButton("Приглашения", callback_data=f"space:{invitations}"),
                    ],
                    [
                        InlineKeyboardButton("Переименовать", callback_data=f"space:{rename}"),
                        InlineKeyboardButton("Описание", callback_data=f"space:{description}"),
                    ],
                    [InlineKeyboardButton("Изменить характер", callback_data=f"space:{character}")],
                    [InlineKeyboardButton("Архивировать", callback_data=f"space:{archive}")],
                )
            )
        hub = await self._workspace_action(actor_id, chat_id, "hub")
        owner_count = sum(record.member.role == "owner" for record in members)
        if actor.role != "owner" or owner_count > 1:
            leave = await self._workspace_action(
                actor_id,
                chat_id,
                "leave_ask",
                access=access,
                workspace_version=workspace.version,
                payload={"member_version": actor.version},
            )
            rows.append([InlineKeyboardButton("Выйти", callback_data=f"space:{leave}")])
        else:
            lines.extend(
                (
                    "",
                    "Чтобы выйти, сначала назначь другого участника владельцем.",
                )
            )
        rows.extend(
            (
                [InlineKeyboardButton("← К пространствам", callback_data=f"space:{hub}")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            )
        )
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    async def _render_character_change(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
    ) -> None:
        rows: list[list[InlineKeyboardButton]] = []
        for character, label in CHARACTER_LABELS.items():
            token = await self._workspace_action(
                actor_id,
                chat_id,
                "character_set",
                payload={"character": character},
                access=access,
                workspace_version=workspace.version,
            )
            rows.append([InlineKeyboardButton(label, callback_data=f"space:{token}")])
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"space:{back}")])
        await self._workspace_edit_or_send(
            query,
            "Выбери новый характер. Права доступа от него не изменятся.",
            InlineKeyboardMarkup(rows),
        )

    async def _render_archive_confirm(
        self, query: Any, actor_id: int, chat_id: int, access: AccessContext
    ) -> None:
        workspace = await self.workspace_service.get_workspace(access)
        confirm = await self._workspace_action(
            actor_id,
            chat_id,
            "archive",
            access=access,
            workspace_version=workspace.version,
        )
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        await self._workspace_edit_or_send(
            query,
            "Архивировать пространство? Участники не смогут использовать его до восстановления.",
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Архивировать", callback_data=f"space:{confirm}")],
                    [InlineKeyboardButton("Отмена", callback_data=f"space:{back}")],
                ]
            ),
        )

    async def _render_leave_confirm(
        self, query: Any, actor_id: int, chat_id: int, access: AccessContext
    ) -> None:
        workspace = await self.workspace_service.get_workspace(access)
        members = await self.workspace_service.list_members(access)
        actor = next(record.member for record in members if record.member.user_id == actor_id)
        confirm = await self._workspace_action(
            actor_id,
            chat_id,
            "leave",
            payload={"member_version": actor.version},
            access=access,
            workspace_version=workspace.version,
        )
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        await self._workspace_edit_or_send(
            query,
            "Выйти из пространства? Последний владелец выйти не сможет.",
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Выйти", callback_data=f"space:{confirm}")],
                    [InlineKeyboardButton("Отмена", callback_data=f"space:{back}")],
                ]
            ),
        )

    async def _send_members(
        self, message: Any, actor_id: int, chat_id: int, access: AccessContext
    ) -> None:
        text, markup = await self._members_view(actor_id, chat_id, access)
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_members(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        notice: str | None = None,
    ) -> None:
        text, markup = await self._members_view(actor_id, chat_id, access, notice=notice)
        await self._workspace_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _members_view(
        self,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        notice: str | None = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        workspace = await self.workspace_service.get_workspace(access)
        members = await self.workspace_service.list_members(access)
        actor = next(record.member for record in members if record.member.user_id == actor_id)
        lines = [f"<b>Участники · {escape(workspace.name)}</b>"]
        if notice:
            lines.extend(("", escape(notice)))
        rows: list[list[InlineKeyboardButton]] = []
        for record in members:
            label = f"{record.display_name} · {ROLE_LABELS[record.member.role]}"
            if actor.role == "owner":
                token = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "member",
                    payload={
                        "member_user_id": record.member.user_id,
                        "member_version": record.member.version,
                        "display_name": record.display_name,
                    },
                    access=access,
                    workspace_version=workspace.version,
                )
                rows.append(
                    [
                        InlineKeyboardButton(
                            self._button_label(label), callback_data=f"space:{token}"
                        )
                    ]
                )
            else:
                lines.append(
                    f"• {escape(record.display_name)} — {escape(ROLE_LABELS[record.member.role])}"
                )
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"space:{back}")])
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    async def _render_member(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        payload: dict[str, Any],
    ) -> None:
        workspace = await self.workspace_service.get_workspace(access)
        members = await self.workspace_service.list_members(access)
        member_user_id = int(payload["member_user_id"])
        expected_version = int(payload["member_version"])
        record = next(
            (
                item
                for item in members
                if item.member.user_id == member_user_id and item.member.version == expected_version
            ),
            None,
        )
        actor = next(item.member for item in members if item.member.user_id == actor_id)
        if record is None or actor.role != "owner":
            raise WorkspaceStaleError("Участник уже изменился.")
        owner_count = sum(item.member.role == "owner" for item in members)
        last_owner = record.member.role == "owner" and owner_count == 1
        rows: list[list[InlineKeyboardButton]] = []
        for role in ("owner", "editor", "viewer"):
            if role == record.member.role or (last_owner and role != "owner"):
                continue
            token = await self._workspace_action(
                actor_id,
                chat_id,
                "member_role",
                payload={
                    "member_user_id": member_user_id,
                    "member_version": record.member.version,
                    "role": role,
                },
                access=access,
                workspace_version=workspace.version,
            )
            rows.append(
                [
                    InlineKeyboardButton(
                        f"Сделать: {ROLE_LABELS[role]}",
                        callback_data=f"space:{token}",
                    )
                ]
            )
        if member_user_id != actor_id:
            revoke = await self._workspace_action(
                actor_id,
                chat_id,
                "member_revoke",
                payload={
                    "member_user_id": member_user_id,
                    "member_version": record.member.version,
                },
                access=access,
                workspace_version=workspace.version,
            )
            rows.append([InlineKeyboardButton("Отозвать доступ", callback_data=f"space:{revoke}")])
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "members",
            access=access,
            workspace_version=workspace.version,
        )
        rows.append([InlineKeyboardButton("← К участникам", callback_data=f"space:{back}")])
        await self._workspace_edit_or_send(
            query,
            f"<b>{escape(record.display_name)}</b>\n"
            f"Роль: {escape(ROLE_LABELS[record.member.role])}"
            + (
                "\nСначала назначь другого владельца, затем эту роль можно изменить."
                if last_owner
                else ""
            ),
            InlineKeyboardMarkup(rows),
            parse_mode="HTML",
        )

    async def _send_invitation_roles(
        self,
        message: Any,
        user: Any,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
    ) -> None:
        text, markup = await self._invitation_roles_view(user.id, chat_id, access, workspace)
        await message.reply_text(text, reply_markup=markup)

    async def _render_invitation_roles(
        self,
        query: Any,
        user: Any,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
    ) -> None:
        text, markup = await self._invitation_roles_view(user.id, chat_id, access, workspace)
        await self._workspace_edit_or_send(query, text, markup)

    async def _invitation_roles_view(
        self,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
    ) -> tuple[str, InlineKeyboardMarkup]:
        rows: list[list[InlineKeyboardButton]] = []
        for role in ("editor", "viewer"):
            token = await self._workspace_action(
                actor_id,
                chat_id,
                "invite_role",
                payload={"role": role},
                access=access,
                workspace_version=workspace.version,
            )
            rows.append([InlineKeyboardButton(ROLE_LABELS[role], callback_data=f"space:{token}")])
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        rows.append([InlineKeyboardButton("Отмена", callback_data=f"space:{back}")])
        return (
            "Выбери роль приглашённого. Редактор сможет управлять проектами; читатель — только смотреть.",
            InlineKeyboardMarkup(rows),
        )

    async def _render_invitation_templates(
        self,
        query: Any,
        user: Any,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
        role: str,
    ) -> None:
        if role not in {"editor", "viewer"}:
            raise WorkspaceStaleError("Роль недоступна.")
        rows: list[list[InlineKeyboardButton]] = []
        templates = INVITATION_TEMPLATES[workspace.character]
        for index in range(len(templates)):
            token = await self._workspace_action(
                user.id,
                chat_id,
                "invite_template",
                payload={"role": role, "template_index": index},
                access=access,
                workspace_version=workspace.version,
            )
            rows.append(
                [InlineKeyboardButton(f"Вариант {index + 1}", callback_data=f"space:{token}")]
            )
        edit = await self._workspace_action(
            user.id,
            chat_id,
            "invite_edit",
            payload={"role": role},
            access=access,
            workspace_version=workspace.version,
        )
        back = await self._workspace_action(
            user.id,
            chat_id,
            "invite_start",
            access=access,
            workspace_version=workspace.version,
        )
        rows.extend(
            (
                [InlineKeyboardButton("Свой текст", callback_data=f"space:{edit}")],
                [InlineKeyboardButton("← Назад", callback_data=f"space:{back}")],
            )
        )
        await self._workspace_edit_or_send(
            query,
            "Выбери стиль приглашения. Перед созданием будет preview.",
            InlineKeyboardMarkup(rows),
        )

    async def _send_invitation_preview(
        self,
        message: Any,
        user: Any,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
        *,
        role: str,
        template_index: int,
        custom_text: str | None,
    ) -> None:
        text, markup = await self._invitation_preview_view(
            user,
            chat_id,
            access,
            workspace,
            role=role,
            template_index=template_index,
            custom_text=custom_text,
        )
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_invitation_preview(
        self,
        query: Any,
        user: Any,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
        *,
        role: str,
        template_index: int,
        custom_text: str | None,
    ) -> None:
        text, markup = await self._invitation_preview_view(
            user,
            chat_id,
            access,
            workspace,
            role=role,
            template_index=template_index,
            custom_text=custom_text,
        )
        await self._workspace_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _invitation_preview_view(
        self,
        user: Any,
        chat_id: int,
        access: AccessContext,
        workspace: Any,
        *,
        role: str,
        template_index: int,
        custom_text: str | None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        if role not in {"editor", "viewer"}:
            raise WorkspaceStaleError("Роль недоступна.")
        templates = INVITATION_TEMPLATES[workspace.character]
        index = template_index % len(templates)
        invitation_text = custom_text or self._format_invitation(
            templates[index], user.display_name or "Участник", workspace.name
        )
        confirm = await self._workspace_action(
            user.id,
            chat_id,
            "invite_confirm",
            payload={
                "role": role,
                "template_index": index,
                "custom_text": custom_text,
            },
            access=access,
            workspace_version=workspace.version,
        )
        next_token = await self._workspace_action(
            user.id,
            chat_id,
            "invite_next",
            payload={
                "role": role,
                "template_index": (index + 1) % len(templates),
                "custom_text": None,
            },
            access=access,
            workspace_version=workspace.version,
        )
        edit = await self._workspace_action(
            user.id,
            chat_id,
            "invite_edit",
            payload={"role": role},
            access=access,
            workspace_version=workspace.version,
        )
        cancel = await self._workspace_action(
            user.id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        text = (
            "<b>Preview приглашения</b>\n\n"
            f"{escape(invitation_text)}\n\n"
            f"<i>{escape(PRIVACY_FOOTER)}</i>\n\n"
            f"Будущая роль: {escape(ROLE_LABELS[role])}.\n"
            "Ссылка будет одноразовой и действительна ограниченное время."
        )
        return text, InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Поделиться приглашением", callback_data=f"space:{confirm}")],
                [
                    InlineKeyboardButton("Другой вариант", callback_data=f"space:{next_token}"),
                    InlineKeyboardButton("Изменить текст", callback_data=f"space:{edit}"),
                ],
                [InlineKeyboardButton("Отмена", callback_data=f"space:{cancel}")],
            ]
        )

    async def _confirm_share_invitation(
        self,
        query: Any,
        context: Any,
        user: Any,
        chat_id: int,
        access: AccessContext,
        payload: dict[str, Any],
    ) -> None:
        username = self._bot_username(context)
        if not username:
            raise WorkspaceAccessError(
                "Сейчас не удалось подготовить ссылку. Открой приглашения и попробуй ещё раз."
            )
        workspace = await self.workspace_service.get_workspace(access)
        index = int(payload.get("template_index", 0)) % len(
            INVITATION_TEMPLATES[workspace.character]
        )
        custom_text = payload.get("custom_text")
        issued = await self.workspace_service.create_invitation(
            access,
            delivery_mode="share",
            template_key=("custom" if custom_text else f"{workspace.character}_{index + 1}"),
            role=str(payload["role"]),
            custom_text=str(custom_text) if custom_text is not None else None,
        )
        await self._show_issued_invitation(
            query, context, issued.token, renewed=False, username=username
        )

    async def _show_issued_invitation(
        self,
        query: Any,
        context: Any,
        raw_token: str,
        *,
        renewed: bool,
        username: str | None = None,
    ) -> None:
        clean_username = username or self._bot_username(context)
        if not clean_username:
            raise WorkspaceAccessError("Ссылка сейчас недоступна.")
        deep_link = create_deep_linked_url(clean_username, f"space_{raw_token}")
        share_url = "https://t.me/share/url?url=" + quote(deep_link, safe="")
        heading = "Приглашение обновлено." if renewed else "Приглашение создано."
        await self._workspace_edit_or_send(
            query,
            f"{heading}\n\nСсылка одноразовая и имеет срок действия. Пересланную ссылку "
            "может открыть её получатель — отправляй только нужному человеку.\n\n"
            f"{deep_link}",
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Поделиться", url=share_url)],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
                ]
            ),
        )

    async def _send_invitations(
        self, message: Any, actor_id: int, chat_id: int, access: AccessContext
    ) -> None:
        text, markup = await self._invitations_view(actor_id, chat_id, access)
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_invitations(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        notice: str | None = None,
    ) -> None:
        text, markup = await self._invitations_view(actor_id, chat_id, access, notice=notice)
        await self._workspace_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _invitations_view(
        self,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        notice: str | None = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        workspace = await self.workspace_service.get_workspace(access)
        invitations = await self.workspace_service.list_invitations(access)
        lines = [f"<b>Приглашения · {escape(workspace.name)}</b>"]
        if notice:
            lines.extend(("", escape(notice)))
        if not invitations:
            lines.extend(("", "Активных приглашений нет."))
        rows: list[list[InlineKeyboardButton]] = []
        for index, invitation in enumerate(invitations, start=1):
            token = await self._workspace_action(
                actor_id,
                chat_id,
                "invitation",
                payload={
                    "invitation_id": invitation.id,
                    "invitation_version": invitation.version,
                    "delivery_mode": invitation.delivery_mode,
                    "role": invitation.role,
                },
                access=access,
                workspace_version=workspace.version,
            )
            rows.append(
                [
                    InlineKeyboardButton(
                        f"Приглашение {index} · {ROLE_LABELS[invitation.role]}",
                        callback_data=f"space:{token}",
                    )
                ]
            )
        create = await self._workspace_action(
            actor_id,
            chat_id,
            "invite_start",
            access=access,
            workspace_version=workspace.version,
        )
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        rows.extend(
            (
                [InlineKeyboardButton("Создать приглашение", callback_data=f"space:{create}")],
                [InlineKeyboardButton("← Назад", callback_data=f"space:{back}")],
            )
        )
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    async def _render_invitation_manage(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        payload: dict[str, Any],
    ) -> None:
        workspace = await self.workspace_service.get_workspace(access)
        invitations = await self.workspace_service.list_invitations(access)
        invitation = next(
            (
                item
                for item in invitations
                if item.id == int(payload["invitation_id"])
                and item.version == int(payload["invitation_version"])
            ),
            None,
        )
        if invitation is None:
            raise WorkspaceStaleError("Приглашение уже недействительно.")
        revoke = await self._workspace_action(
            actor_id,
            chat_id,
            "invite_revoke",
            payload={
                "invitation_id": invitation.id,
                "invitation_version": invitation.version,
            },
            access=access,
            workspace_version=workspace.version,
        )
        renew = await self._workspace_action(
            actor_id,
            chat_id,
            "invite_renew",
            payload={
                "invitation_id": invitation.id,
                "invitation_version": invitation.version,
            },
            access=access,
            workspace_version=workspace.version,
        )
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "invitations",
            access=access,
            workspace_version=workspace.version,
        )
        await self._workspace_edit_or_send(
            query,
            "<b>Активное приглашение</b>\n"
            f"Роль: {escape(ROLE_LABELS[invitation.role])}\n"
            f"Способ: {'одноразовая ссылка' if invitation.delivery_mode == 'share' else 'адресное'}\n"
            f"Действует до: {escape(invitation.expires_at.strftime('%d.%m.%Y %H:%M UTC'))}",
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Обновить", callback_data=f"space:{renew}")],
                    [InlineKeyboardButton("Отозвать", callback_data=f"space:{revoke}")],
                    [InlineKeyboardButton("← Назад", callback_data=f"space:{back}")],
                ]
            ),
            parse_mode="HTML",
        )

    async def _send_incoming_invitation(self, message: Any, incoming: Any) -> None:
        await message.reply_text(
            self._incoming_invitation_text(incoming.preview),
            reply_markup=self._incoming_keyboard(incoming.actions),
            parse_mode="HTML",
        )

    async def _incoming_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        data = query.data or ""
        if not self._workspace_enabled() or not data.startswith("spacei:") or data.count(":") != 1:
            await self._workspace_stale(query)
            return
        token = data.removeprefix("spacei:")
        if not token or len(token) > 48:
            await self._workspace_stale(query)
            return
        user = await self._user(update.effective_user.id)
        chat_id = update.effective_chat.id
        try:
            result = await self.workspace_service.perform_invitation_action(token, user.id, chat_id)
        except (WorkspaceInvitationError, WorkspaceAccessError):
            await self._workspace_stale(query)
            return
        await query.answer()
        if result.action == "details":
            await query.message.reply_text(self._incoming_details_text(result), parse_mode="HTML")
            return
        if result.action == "later":
            await self._workspace_edit_or_send(
                query,
                "Приглашение осталось в ожидании. Вернуться к нему можно по той же ссылке, "
                "пока она действует.",
                None,
            )
            return
        if result.action == "decline":
            await self._workspace_edit_or_send(query, "Приглашение отклонено.", None)
            return
        if result.action == "accept" and result.access_context is not None:
            await self.workspace_service.set_context(result.access_context, chat_id)
            await self._render_workspace(
                query,
                user.id,
                chat_id,
                result.access_context,
                notice="Ты присоединился к пространству.",
            )
            return
        await self._workspace_stale(query)

    async def _send_projects(
        self,
        message: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        status: str = "active",
        notice: str | None = None,
    ) -> None:
        text, markup = await self._projects_view(
            actor_id, chat_id, access, status=status, notice=notice
        )
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_projects(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        status: str = "active",
        notice: str | None = None,
    ) -> None:
        text, markup = await self._projects_view(
            actor_id, chat_id, access, status=status, notice=notice
        )
        await self._workspace_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _projects_view(
        self,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        *,
        status: str,
        notice: str | None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        safe_status = "archived" if status == "archived" else "active"
        workspace = await self.workspace_service.get_workspace(access)
        members = await self.workspace_service.list_members(access)
        actor = next(record.member for record in members if record.member.user_id == actor_id)
        projects = await self.workspace_service.list_projects(access, status=safe_status)
        heading = "Архив проектов" if safe_status == "archived" else "Проекты"
        lines = [f"<b>{heading} · {escape(workspace.name)}</b>"]
        if notice:
            lines.extend(("", escape(notice)))
        if not projects:
            lines.extend(("", "Здесь пока нет проектов."))
        rows: list[list[InlineKeyboardButton]] = []
        for project in projects:
            if safe_status == "active":
                token = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "project",
                    access=access,
                    workspace_version=workspace.version,
                    workspace_project_id=project.id,
                    workspace_project_version=project.version,
                )
                rows.append(
                    [
                        InlineKeyboardButton(
                            self._button_label(project.name),
                            callback_data=f"space:{token}",
                        )
                    ]
                )
            elif actor.role in {"owner", "editor"}:
                token = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "project_restore",
                    access=access,
                    workspace_version=workspace.version,
                    workspace_project_id=project.id,
                    workspace_project_version=project.version,
                )
                rows.append(
                    [
                        InlineKeyboardButton(
                            self._button_label(project.name),
                            callback_data=f"space:{token}",
                        )
                    ]
                )
            else:
                lines.append(f"• {escape(project.name)}")
        if safe_status == "active":
            if actor.role in {"owner", "editor"}:
                create = await self._workspace_action(
                    actor_id,
                    chat_id,
                    "project_create",
                    access=access,
                    workspace_version=workspace.version,
                )
                rows.append([InlineKeyboardButton("＋ Проект", callback_data=f"space:{create}")])
            archived = await self._workspace_action(
                actor_id,
                chat_id,
                "projects_archived",
                access=access,
                workspace_version=workspace.version,
            )
            rows.append([InlineKeyboardButton("Архив проектов", callback_data=f"space:{archived}")])
        else:
            active = await self._workspace_action(
                actor_id,
                chat_id,
                "projects",
                access=access,
                workspace_version=workspace.version,
            )
            rows.append([InlineKeyboardButton("← К проектам", callback_data=f"space:{active}")])
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "open",
            access=access,
            workspace_version=workspace.version,
        )
        rows.append([InlineKeyboardButton("← К пространству", callback_data=f"space:{back}")])
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    async def _render_project(
        self,
        query: Any,
        actor_id: int,
        chat_id: int,
        access: AccessContext,
        claim: Any,
        *,
        notice: str | None = None,
    ) -> None:
        if claim.workspace_project_id is None:
            raise WorkspaceStaleError("Проект уже изменился.")
        project = await self.workspace_service.get_project(access, claim.workspace_project_id)
        workspace = await self.workspace_service.get_workspace(access)
        members = await self.workspace_service.list_members(access)
        actor = next(record.member for record in members if record.member.user_id == actor_id)
        lines = [
            f"<b>Проект · {escape(project.name)}</b>",
            f"Пространство: {escape(workspace.name)}",
        ]
        if notice:
            lines.extend(("", escape(notice)))
        context_token = await self._workspace_action(
            actor_id,
            chat_id,
            "project_context",
            access=access,
            workspace_version=workspace.version,
            workspace_project_id=project.id,
            workspace_project_version=project.version,
        )
        rows: list[list[InlineKeyboardButton]] = [
            [
                InlineKeyboardButton(
                    "Выбрать контекст проекта",
                    callback_data=f"space:{context_token}",
                )
            ]
        ]
        if actor.role in {"owner", "editor"}:
            rename = await self._workspace_action(
                actor_id,
                chat_id,
                "project_rename",
                access=access,
                workspace_version=workspace.version,
                workspace_project_id=project.id,
                workspace_project_version=project.version,
            )
            archive = await self._workspace_action(
                actor_id,
                chat_id,
                "project_archive",
                access=access,
                workspace_version=workspace.version,
                workspace_project_id=project.id,
                workspace_project_version=project.version,
            )
            rows.extend(
                (
                    [InlineKeyboardButton("Переименовать", callback_data=f"space:{rename}")],
                    [InlineKeyboardButton("Архивировать", callback_data=f"space:{archive}")],
                )
            )
        back = await self._workspace_action(
            actor_id,
            chat_id,
            "projects",
            access=access,
            workspace_version=workspace.version,
        )
        rows.append([InlineKeyboardButton("← К проектам", callback_data=f"space:{back}")])
        await self._workspace_edit_or_send(
            query, "\n".join(lines), InlineKeyboardMarkup(rows), parse_mode="HTML"
        )

    async def _workspace_action(
        self,
        actor_id: int,
        chat_id: int,
        action: str,
        *,
        payload: dict[str, Any] | None = None,
        access: AccessContext | None = None,
        workspace_version: int | None = None,
        workspace_project_id: int | None = None,
        workspace_project_version: int | None = None,
    ) -> str:
        return await self.workspace_service.issue_action(
            actor_id,
            chat_id,
            action,
            payload=payload,
            context=access,
            workspace_version=workspace_version,
            workspace_project_id=workspace_project_id,
            workspace_project_version=workspace_project_version,
        )

    async def _begin_workspace_input(
        self,
        actor_id: int,
        chat_id: int,
        action: str,
        *,
        payload: dict[str, Any] | None = None,
        access: AccessContext | None = None,
        workspace_version: int | None = None,
        workspace_project_id: int | None = None,
        workspace_project_version: int | None = None,
    ) -> None:
        await self.task_service.cancel_pending_input(actor_id, chat_id)
        await self.collection_service.cancel_input(actor_id, chat_id)
        await self.collection_service.clear_context(actor_id, chat_id)
        await self.lab_uploads.cancel_active(actor_id, chat_id)
        await self.vision_image_sessions.cancel_active(actor_id, chat_id)
        await self.vision_service.cancel(actor_id, chat_id)
        await self.workspace_service.begin_input(
            actor_id,
            chat_id,
            action.removeprefix("input_"),
            payload=payload,
            context=access,
            workspace_version=workspace_version,
            workspace_project_id=workspace_project_id,
            workspace_project_version=workspace_project_version,
        )

    async def _rearm_workspace_input(self, actor_id: int, chat_id: int, claim: Any) -> None:
        action = claim.action.removeprefix("input:")
        await self._begin_workspace_input(
            actor_id,
            chat_id,
            f"input_{action}",
            payload=claim.payload,
            access=claim.access_context,
            workspace_version=claim.workspace_version,
            workspace_project_id=claim.workspace_project_id,
            workspace_project_version=claim.workspace_project_version,
        )

    @staticmethod
    def _format_invitation(template: str, inviter: str, workspace: str) -> str:
        return template.format(inviter=inviter, workspace=workspace)

    @classmethod
    def _incoming_invitation_text(cls, preview: InvitationPreview) -> str:
        template = cls._template_from_key(preview.character, preview.template_key)
        body = preview.custom_text or cls._format_invitation(
            template, preview.inviter_display_name, preview.workspace_name
        )
        return (
            "<b>Приглашение в совместное пространство</b>\n\n"
            f"{escape(body)}\n\n"
            f"Пространство: <b>{escape(preview.workspace_name)}</b>\n"
            f"Характер: {escape(CHARACTER_LABELS.get(preview.character, 'Свой вариант'))}\n"
            f"Будущая роль: {escape(ROLE_LABELS.get(preview.role, 'Участник'))}\n\n"
            f"<i>{escape(PRIVACY_FOOTER)}</i>"
        )

    @classmethod
    def _incoming_details_text(cls, result: InvitationActionResult) -> str:
        preview = result.preview
        return (
            f"Пространство «{escape(preview.workspace_name)}» даёт доступ только к явно "
            "общим участникам и проектам. Личные записи, карта желаний, Health, Doctor и "
            "Labs не передаются. Роль после принятия: "
            f"{escape(ROLE_LABELS.get(preview.role, 'Участник'))}."
        )

    @staticmethod
    def _incoming_keyboard(actions: dict[str, str]) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Присоединиться", callback_data=f"spacei:{actions['accept']}"
                    )
                ],
                [
                    InlineKeyboardButton("Подробнее", callback_data=f"spacei:{actions['details']}"),
                    InlineKeyboardButton("Не сейчас", callback_data=f"spacei:{actions['later']}"),
                ],
                [InlineKeyboardButton("Отклонить", callback_data=f"spacei:{actions['decline']}")],
            ]
        )

    @staticmethod
    def _template_from_key(character: str, template_key: str) -> str:
        templates = INVITATION_TEMPLATES.get(character, INVITATION_TEMPLATES["custom"])
        try:
            index = int(template_key.rsplit("_", 1)[-1]) - 1
        except (TypeError, ValueError):
            index = 0
        return templates[index % len(templates)]

    @staticmethod
    def _character_emoji(character: str) -> str:
        return {
            "pair": "💞",
            "friends": "🚀",
            "family": "🏡",
            "team": "🤝",
            "custom": "✨",
        }.get(character, "✨")

    @staticmethod
    def _button_label(value: str, maximum: int = 56) -> str:
        clean = " ".join(value.split())
        return clean if len(clean) <= maximum else clean[: maximum - 1].rstrip() + "…"

    @staticmethod
    def _bot_username(context: Any) -> str | None:
        bot = getattr(context, "bot", None)
        try:
            username = getattr(bot, "username", None) if bot is not None else None
        except (RuntimeError, TelegramError):
            return None
        if not isinstance(username, str):
            return None
        return username.lstrip("@").strip() or None

    @staticmethod
    def _workspace_navigation_markup() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
                [InlineKeyboardButton("❓ Помощь", callback_data="nav:help")],
            ]
        )

    @staticmethod
    async def _workspace_stale(query: Any) -> None:
        await query.answer("Эта кнопка устарела или недоступна.", show_alert=True)

    async def _workspace_stale_message(self, query: Any) -> None:
        await self._workspace_edit_or_send(
            query,
            "Действие устарело или доступ изменился. Открой /spaces ещё раз.",
            None,
        )

    @staticmethod
    async def _workspace_edit_or_send(
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

    def _workspace_enabled(self) -> bool:
        return bool(getattr(self.settings, "enable_workspace_access", False))
