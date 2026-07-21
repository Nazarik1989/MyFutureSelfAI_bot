from __future__ import annotations

from html import escape
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from .collection_commands import CollectionCommand
from .collections_service import (
    COLLECTION_KIND_LABELS,
    STARTER_TEMPLATES,
    CollectionClaim,
    CollectionKind,
    CollectionNameError,
    CollectionPage,
    CollectionSummary,
    LifeCollectionService,
    clean_collection_name,
    split_list_items,
)
from .models import LifeCollection


class CollectionHandlers:
    collection_service: LifeCollectionService
    collection_command_router: Any
    task_service: Any

    async def collections_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        user = await self._user(update.effective_user.id)
        await self.task_service.cancel_pending_input(user.id, update.effective_chat.id)
        if not await self.collection_service.is_onboarded(user.id):
            await self._send_collection_onboarding(
                update.effective_message, user.id, update.effective_chat.id
            )
            return
        await self._send_collection_hub(update.effective_message, user.id, update.effective_chat.id)

    async def collection_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        data = query.data or ""
        if not data.startswith("collection:") or data.count(":") != 1:
            await self._collection_stale(query)
            return
        token = data.removeprefix("collection:")
        if not token or len(token) > 32:
            await self._collection_stale(query)
            return
        user = await self._user(update.effective_user.id)
        owner_id = user.id
        chat_id = update.effective_chat.id
        action = await self.collection_service.capability_action(token, owner_id, chat_id)
        if action is None:
            await self._collection_stale(query)
            return
        await query.answer()

        if action == "onboard_all":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None:
                await self._collection_stale(query)
                return
            result = await self.collection_service.create_starters(
                owner_id, tuple(item.key for item in STARTER_TEMPLATES)
            )
            await self._render_collection_hub(
                query,
                owner_id,
                chat_id,
                notice=(
                    "Стартовые разделы созданы."
                    if result.status == "created"
                    else "Настройка уже завершена."
                ),
            )
            return
        if action == "onboard_empty":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None:
                await self._collection_stale(query)
                return
            await self.collection_service.complete_empty_onboarding(owner_id)
            await self._render_collection_hub(
                query,
                owner_id,
                chat_id,
                notice="Готово. Новые разделы можно создать в любой момент.",
            )
            return
        if action == "onboard_select":
            if await self._claim(token, owner_id, chat_id, action) is None:
                await self._collection_stale(query)
                return
            await self._render_starter_picker(query, owner_id, chat_id, ())
            return
        if action == "onboard_home":
            if await self._claim(token, owner_id, chat_id, action) is None:
                await self._collection_stale(query)
                return
            await self._render_collection_onboarding(query, owner_id, chat_id)
            return
        if action == "starter_toggle":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None:
                await self._collection_stale(query)
                return
            selected = list(claim.payload.get("selected", []))
            key = str(claim.payload.get("key", ""))
            if key not in {item.key for item in STARTER_TEMPLATES}:
                await self._collection_stale(query)
                return
            if key in selected:
                selected.remove(key)
            else:
                selected.append(key)
            await self._render_starter_picker(query, owner_id, chat_id, tuple(selected))
            return
        if action == "starter_confirm":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None:
                await self._collection_stale(query)
                return
            keys = tuple(str(key) for key in claim.payload.get("selected", []))
            result = await self.collection_service.create_starters(owner_id, keys)
            if result.status not in {"created", "already_completed"}:
                await self._collection_stale_message(query)
                return
            await self._render_collection_hub(
                query,
                owner_id,
                chat_id,
                notice=f"Создано разделов: {len(result.item_ids)}.",
            )
            return
        if action == "create_type":
            pending = await self.collection_service.begin_input(
                token,
                owner_id,
                chat_id,
                allowed={action},
                input_action="input_create",
            )
            if pending is None:
                await self._collection_stale_message(query)
                return
            kind = COLLECTION_KIND_LABELS.get(
                str((await self._pending_payload(owner_id, chat_id)).get("kind"))
            )
            await self._collection_edit_or_send(
                query,
                f"Пришли название для нового раздела типа «{kind or 'тема'}». "
                "/cancel — отменить ввод.",
                self._navigation_markup(),
            )
            return
        if action == "hub":
            if await self._claim(token, owner_id, chat_id, action) is None:
                await self._collection_stale(query)
                return
            await self._render_collection_hub(query, owner_id, chat_id)
            return
        if action == "hub_list":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None:
                await self._collection_stale(query)
                return
            await self._render_collection_page(
                query,
                owner_id,
                chat_id,
                int(claim.payload.get("page", 0)),
                kind=self._payload_kind(claim.payload.get("kind")),
                status=("archived" if claim.payload.get("status") == "archived" else "active"),
            )
            return
        if action == "open":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None:
                await self._collection_stale(query)
                return
            await self._render_collection_card(query, owner_id, chat_id, claim.collection_id)
            return
        if action == "content":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None:
                await self._collection_stale(query)
                return
            await self._render_collection_content(
                query,
                owner_id,
                chat_id,
                claim.collection_id,
                int(claim.payload.get("page", 0)),
            )
            return
        if action in {"add_prompt", "rename_prompt", "alias_prompt"}:
            input_action = {
                "add_prompt": "input_add",
                "rename_prompt": "input_rename",
                "alias_prompt": "input_alias",
            }[action]
            pending = await self.collection_service.begin_input(
                token,
                owner_id,
                chat_id,
                allowed={action},
                input_action=input_action,
            )
            if pending is None:
                await self._collection_stale_message(query)
                return
            prompt = {
                "add_prompt": "Пришли запись. Для списка можно перечислить пункты через запятую.",
                "rename_prompt": "Пришли новое название раздела.",
                "alias_prompt": "Пришли дополнительное короткое имя раздела.",
            }[action]
            await self._collection_edit_or_send(
                query, f"{prompt} /cancel — отменить ввод.", self._navigation_markup()
            )
            return
        if action == "confirm_create":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None:
                await self._collection_stale(query)
                return
            await self._confirm_collection_creation(query, owner_id, chat_id, claim)
            return
        if action in {"confirm_add", "add_suggested"}:
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None or claim.collection_version is None:
                await self._collection_stale(query)
                return
            contents = tuple(str(item) for item in claim.payload.get("items", []))
            if not contents and claim.payload.get("content"):
                contents = (str(claim.payload["content"]),)
            result = await self.collection_service.create_items(
                owner_id,
                chat_id,
                claim.collection_id,
                claim.collection_version,
                contents,
                source=str(claim.payload.get("source", "text")),
                forced_kind=(
                    str(claim.payload["forced_kind"]) if claim.payload.get("forced_kind") else None
                ),
            )
            if result.status != "created_items":
                await self._collection_stale_message(query)
                return
            await self._render_collection_content(
                query,
                owner_id,
                chat_id,
                claim.collection_id,
                0,
                notice=f"Добавлено записей: {len(result.item_ids)}.",
            )
            return
        if action in {"archive", "restore"}:
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None or claim.collection_version is None:
                await self._collection_stale(query)
                return
            result = await self.collection_service.set_archived(
                owner_id,
                claim.collection_id,
                claim.collection_version,
                archived=action == "archive",
            )
            if result.status not in {"archived", "restored"}:
                await self._collection_stale_message(query)
                return
            await self._render_collection_hub(
                query,
                owner_id,
                chat_id,
                notice=("Раздел архивирован." if action == "archive" else "Раздел восстановлен."),
            )
            return
        if action == "delete_ask":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None:
                await self._collection_stale(query)
                return
            summary = await self.collection_service.summary(owner_id, claim.collection_id)
            if summary is None:
                await self._collection_stale_message(query)
                return
            await self._render_delete_choice(query, owner_id, chat_id, summary)
            return
        if action in {"delete_empty", "delete_links"}:
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None or claim.collection_version is None:
                await self._collection_stale(query)
                return
            result = await self.collection_service.delete_collection(
                owner_id,
                claim.collection_id,
                claim.collection_version,
                unlink_nonempty=action == "delete_links",
            )
            if result.status != "deleted":
                await self._collection_stale_message(query)
                return
            await self._render_collection_hub(
                query,
                owner_id,
                chat_id,
                notice="Раздел удалён. Исходные записи и задачи сохранены.",
            )
            return
        if action == "item_open":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None or claim.inbox_item_id is None:
                await self._collection_stale(query)
                return
            await self._render_collection_item(
                query, owner_id, chat_id, claim.collection_id, claim.inbox_item_id
            )
            return
        if action in {"move_menu", "link_menu"}:
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None or claim.inbox_item_id is None:
                await self._collection_stale(query)
                return
            await self._render_collection_targets(
                query,
                owner_id,
                chat_id,
                claim.collection_id,
                claim.inbox_item_id,
                mode="move" if action == "move_menu" else "link",
                page=int(claim.payload.get("page", 0)),
            )
            return
        if action in {"move_to", "link_to"}:
            claim = await self._claim(token, owner_id, chat_id, action)
            if (
                claim is None
                or claim.collection_id is None
                or claim.collection_version is None
                or claim.inbox_item_id is None
            ):
                await self._collection_stale(query)
                return
            target_id = int(claim.payload.get("target_id", 0))
            target_version = int(claim.payload.get("target_version", 0))
            if action == "move_to":
                result = await self.collection_service.move_item(
                    owner_id,
                    claim.collection_id,
                    claim.collection_version,
                    target_id,
                    target_version,
                    claim.inbox_item_id,
                )
                notice = "Запись перемещена."
            else:
                result = await self.collection_service.link_item(
                    owner_id, target_id, target_version, claim.inbox_item_id
                )
                notice = "Дополнительная связь добавлена."
            if result.status not in {"moved", "linked", "already_linked"}:
                await self._collection_stale_message(query)
                return
            await self.collection_service.set_context(
                owner_id,
                chat_id,
                target_id,
                last_inbox_item_id=claim.inbox_item_id,
            )
            await self._render_collection_content(
                query, owner_id, chat_id, target_id, 0, notice=notice
            )
            return
        if action == "unlink":
            claim = await self._claim(token, owner_id, chat_id, action)
            if (
                claim is None
                or claim.collection_id is None
                or claim.collection_version is None
                or claim.inbox_item_id is None
            ):
                await self._collection_stale(query)
                return
            result = await self.collection_service.unlink_item(
                owner_id,
                claim.collection_id,
                claim.collection_version,
                claim.inbox_item_id,
            )
            if result.status != "unlinked":
                await self._collection_stale_message(query)
                return
            await self._render_collection_content(
                query,
                owner_id,
                chat_id,
                claim.collection_id,
                0,
                notice="Связь убрана. Исходная запись сохранена.",
            )
            return
        if action == "delete_item_ask":
            claim = await self._claim(token, owner_id, chat_id, action)
            if claim is None or claim.collection_id is None or claim.inbox_item_id is None:
                await self._collection_stale(query)
                return
            summary = await self.collection_service.summary(owner_id, claim.collection_id)
            if summary is None:
                await self._collection_stale_message(query)
                return
            confirm = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                "delete_item_confirm",
                collection=summary.collection,
                inbox_item_id=claim.inbox_item_id,
            )
            await self._collection_edit_or_send(
                query,
                "Удалить саму запись из Inbox? Это также отменит связанную задачу и напоминание.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Удалить запись", callback_data=f"collection:{confirm}"
                            )
                        ],
                        self._navigation_row(),
                    ]
                ),
            )
            return
        if action == "delete_item_confirm":
            claim = await self._claim(token, owner_id, chat_id, action)
            if (
                claim is None
                or claim.collection_id is None
                or claim.collection_version is None
                or claim.inbox_item_id is None
            ):
                await self._collection_stale(query)
                return
            result = await self.collection_service.delete_item(
                owner_id,
                claim.collection_id,
                claim.collection_version,
                claim.inbox_item_id,
            )
            if result.status != "item_deleted":
                await self._collection_stale_message(query)
                return
            await self._render_collection_content(
                query,
                owner_id,
                chat_id,
                claim.collection_id,
                0,
                notice="Запись удалена.",
            )
            return
        await self._collection_stale(query)

    async def collection_pending_text(self, update: Update, text: str, source: str) -> bool:
        user = await self._user(update.effective_user.id)
        owner_id = user.id
        chat_id = update.effective_chat.id
        pending = await self.collection_service.pending_input(owner_id, chat_id)
        if pending is None:
            return False
        claim = await self.collection_service.claim_action(
            pending.token,
            owner_id,
            chat_id,
            {"input_create", "input_add", "input_rename", "input_alias"},
            pending_status="awaiting_input",
        )
        if claim is None:
            return False
        message = update.effective_message
        try:
            if claim.action == "input_create":
                kind = self._payload_kind(claim.payload.get("kind")) or "topic"
                result = await self.collection_service.create_collection(owner_id, kind, text)
                if result.status == "conflict":
                    await message.reply_text("Раздел с таким названием или alias уже существует.")
                    return True
                await message.reply_text(
                    f"Раздел «{result.collection.name}» создан.",
                    reply_markup=self._navigation_markup(),
                )
                return True
            if claim.collection_id is None or claim.collection_version is None:
                await message.reply_text("Действие устарело. Открой /collections ещё раз.")
                return True
            if claim.action == "input_rename":
                result = await self.collection_service.rename(
                    owner_id, claim.collection_id, claim.collection_version, text
                )
                response = (
                    f"Раздел переименован: {result.collection.name}."
                    if result.status == "renamed"
                    else "Название занято или карточка уже изменилась."
                )
                await message.reply_text(response, reply_markup=self._navigation_markup())
                return True
            if claim.action == "input_alias":
                result = await self.collection_service.add_alias(
                    owner_id, claim.collection_id, claim.collection_version, text
                )
                response = (
                    "Дополнительное имя добавлено."
                    if result.status == "aliased"
                    else "Такое имя уже используется или карточка изменилась."
                )
                await message.reply_text(response, reply_markup=self._navigation_markup())
                return True
            collection = await self.collection_service.summary(owner_id, claim.collection_id)
            if collection is None:
                await message.reply_text("Раздел больше недоступен.")
                return True
            await self._save_or_preview(
                message,
                owner_id,
                chat_id,
                collection.collection,
                text,
                source=source,
            )
            return True
        except CollectionNameError as exc:
            await message.reply_text(str(exc))
            return True

    async def handle_collection_natural(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        source: str,
    ) -> bool:
        del context
        command: CollectionCommand | None = self.collection_command_router.route(text)
        if command is None:
            return False
        user = await self._user(update.effective_user.id)
        owner_id = user.id
        chat_id = update.effective_chat.id
        await self.task_service.cancel_pending_input(owner_id, chat_id)
        message = update.effective_message
        try:
            if command.action == "create":
                name, _ = clean_collection_name(command.target or "")
                token = await self.collection_service.issue_action(
                    owner_id,
                    chat_id,
                    "confirm_create",
                    payload={"kind": command.kind or "topic", "name": name, "source": source},
                )
                await message.reply_text(
                    f"Создать {COLLECTION_KIND_LABELS[command.kind or 'topic']} «{name}»?",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [InlineKeyboardButton("Создать", callback_data=f"collection:{token}")],
                            self._navigation_row(),
                        ]
                    ),
                )
                return True
            if command.action == "show":
                resolution = await self.collection_service.resolve(owner_id, command.target or "")
                if resolution.match is not None:
                    await self.collection_service.set_context(
                        owner_id, chat_id, resolution.match.id
                    )
                    await self._send_collection_content(
                        message, owner_id, chat_id, resolution.match.id, 0
                    )
                elif resolution.candidates:
                    await self._send_collection_candidates(
                        message,
                        owner_id,
                        chat_id,
                        resolution.candidates,
                        action="open",
                    )
                else:
                    await message.reply_text(
                        "Такого раздела пока нет. Открой /collections, чтобы создать его."
                    )
                return True
            if command.action == "add_more":
                active = await self.collection_service.active_context(owner_id, chat_id)
                if active is None:
                    await message.reply_text(
                        "Неясно, в какой раздел добавить запись. Сначала назови раздел."
                    )
                    return True
                await self._save_or_preview(
                    message,
                    owner_id,
                    chat_id,
                    active.collection,
                    command.content or "",
                    source=source,
                )
                return True
            if command.action == "move_last":
                active = await self.collection_service.active_context(owner_id, chat_id)
                resolution = await self.collection_service.resolve(owner_id, command.target or "")
                if active is None or active.last_inbox_item_id is None:
                    await message.reply_text(
                        "Неясно, какую запись переносить. Открой её в разделе."
                    )
                    return True
                if resolution.match is None:
                    await message.reply_text("Целевой раздел не найден или название неоднозначно.")
                    return True
                result = await self.collection_service.move_item(
                    owner_id,
                    active.collection.id,
                    active.collection.version,
                    resolution.match.id,
                    resolution.match.version,
                    active.last_inbox_item_id,
                )
                if result.status != "moved":
                    await message.reply_text("Запись уже изменилась. Открой раздел снова.")
                    return True
                await self.collection_service.set_context(
                    owner_id,
                    chat_id,
                    resolution.match.id,
                    last_inbox_item_id=active.last_inbox_item_id,
                )
                await message.reply_text(f"Запись перемещена в «{resolution.match.name}».")
                return True
            if command.action == "add":
                return await self._handle_natural_add(message, owner_id, chat_id, command, source)
        except CollectionNameError as exc:
            await message.reply_text(str(exc))
            return True
        return False

    async def cancel_collection_state(self, update: Update) -> bool:
        user = await self._user(update.effective_user.id)
        cancelled_input = await self.collection_service.cancel_input(
            user.id, update.effective_chat.id
        )
        cleared_context = await self.collection_service.clear_context(
            user.id, update.effective_chat.id
        )
        return cancelled_input or cleared_context

    async def _handle_natural_add(
        self,
        message: Any,
        owner_id: int,
        chat_id: int,
        command: CollectionCommand,
        source: str,
    ) -> bool:
        content = command.content
        if content is not None:
            resolution = await self.collection_service.resolve(owner_id, command.target or "")
        else:
            resolution, content = await self.collection_service.resolve_leading(
                owner_id, command.target or ""
            )
        if resolution.match is not None and content:
            await self._save_or_preview(
                message,
                owner_id,
                chat_id,
                resolution.match,
                content,
                source=source,
                forced_kind=command.forced_item_kind,
            )
            return True
        if resolution.candidates and content:
            await self._send_collection_candidates(
                message,
                owner_id,
                chat_id,
                resolution.candidates,
                action="add_suggested",
                payload={
                    "content": content,
                    "source": source,
                    "forced_kind": command.forced_item_kind,
                },
            )
            return True
        target = command.target or ""
        proposed_content = content
        if not proposed_content:
            pieces = re_split_once(target)
            target, proposed_content = pieces
        try:
            display, _ = clean_collection_name(target)
        except CollectionNameError:
            await message.reply_text(
                "Укажи раздел и запись точнее, например: Добавь в Покупки чай."
            )
            return True
        kind = command.kind or "list"
        token = await self.collection_service.issue_action(
            owner_id,
            chat_id,
            "confirm_create",
            payload={
                "kind": kind,
                "name": display,
                "content": proposed_content,
                "source": source,
                "forced_kind": command.forced_item_kind,
            },
        )
        await message.reply_text(
            f"Раздел «{display}» не найден. Создать {COLLECTION_KIND_LABELS[kind]} с таким названием?",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Создать", callback_data=f"collection:{token}")],
                    self._navigation_row(),
                ]
            ),
        )
        return True

    async def _confirm_collection_creation(
        self, query: Any, owner_id: int, chat_id: int, claim: CollectionClaim
    ) -> None:
        kind = self._payload_kind(claim.payload.get("kind")) or "topic"
        name = str(claim.payload.get("name", ""))
        try:
            result = await self.collection_service.create_collection(owner_id, kind, name)
        except CollectionNameError as exc:
            await self._collection_edit_or_send(query, str(exc), self._navigation_markup())
            return
        if result.status != "created" or result.collection is None:
            await self._collection_edit_or_send(
                query,
                "Раздел с таким названием уже существует.",
                self._navigation_markup(),
            )
            return
        content = claim.payload.get("content")
        if content:
            items, ambiguous = split_list_items(str(content))
            if ambiguous:
                preview = await self.collection_service.issue_action(
                    owner_id,
                    chat_id,
                    "confirm_add",
                    collection=result.collection,
                    payload={
                        "items": list(items),
                        "source": str(claim.payload.get("source", "text")),
                        "forced_kind": claim.payload.get("forced_kind"),
                    },
                )
                await self._collection_edit_or_send(
                    query,
                    self._preview_text(result.collection, items),
                    InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "Сохранить", callback_data=f"collection:{preview}"
                                )
                            ],
                            self._navigation_row(),
                        ]
                    ),
                )
                return
            created = await self.collection_service.create_items(
                owner_id,
                chat_id,
                result.collection.id,
                result.collection.version,
                items,
                source=str(claim.payload.get("source", "text")),
                forced_kind=(
                    str(claim.payload["forced_kind"]) if claim.payload.get("forced_kind") else None
                ),
            )
            if created.status != "created_items":
                await self._collection_stale_message(query)
                return
        await self._render_collection_card(
            query,
            owner_id,
            chat_id,
            result.collection.id,
            notice=f"Раздел «{result.collection.name}» создан.",
        )

    async def _save_or_preview(
        self,
        message: Any,
        owner_id: int,
        chat_id: int,
        collection: LifeCollection,
        content: str,
        *,
        source: str,
        forced_kind: str | None = None,
    ) -> None:
        items, ambiguous = (
            split_list_items(content) if collection.kind == "list" else ((content.strip(),), False)
        )
        if not items:
            await message.reply_text("Запись не может быть пустой.")
            return
        if ambiguous:
            token = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                "confirm_add",
                collection=collection,
                payload={"items": list(items), "source": source, "forced_kind": forced_kind},
            )
            await message.reply_text(
                self._preview_text(collection, items),
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("Сохранить", callback_data=f"collection:{token}")],
                        self._navigation_row(),
                    ]
                ),
            )
            return
        result = await self.collection_service.create_items(
            owner_id,
            chat_id,
            collection.id,
            collection.version,
            items,
            source=source,
            forced_kind=forced_kind,
        )
        if result.status != "created_items":
            await message.reply_text("Раздел уже изменился. Открой его снова через /collections.")
            return
        await message.reply_text(
            f"Добавлено в «{collection.name}»: {len(result.item_ids)}.",
            reply_markup=self._navigation_markup(),
        )

    async def _send_collection_onboarding(self, message: Any, owner_id: int, chat_id: int) -> None:
        text, markup = await self._collection_onboarding_view(owner_id, chat_id)
        await message.reply_text(text, reply_markup=markup)

    async def _render_collection_onboarding(self, query: Any, owner_id: int, chat_id: int) -> None:
        text, markup = await self._collection_onboarding_view(owner_id, chat_id)
        await self._collection_edit_or_send(query, text, markup)

    async def _collection_onboarding_view(
        self, owner_id: int, chat_id: int
    ) -> tuple[str, InlineKeyboardMarkup]:
        tokens = {
            action: await self.collection_service.issue_action(owner_id, chat_id, action)
            for action in ("onboard_all", "onboard_select", "onboard_empty")
        }
        create = await self.collection_service.issue_action(
            owner_id, chat_id, "create_type", payload={"kind": "topic"}
        )
        return (
            "🗂 Мои разделы\n\nВыбери старт. Ничего не создаётся автоматически.",
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Создать все", callback_data=f"collection:{tokens['onboard_all']}"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Выбрать разделы",
                            callback_data=f"collection:{tokens['onboard_select']}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Начать с пустого",
                            callback_data=f"collection:{tokens['onboard_empty']}",
                        )
                    ],
                    [InlineKeyboardButton("Создать свой", callback_data=f"collection:{create}")],
                    self._navigation_row(),
                ]
            ),
        )

    async def _render_starter_picker(
        self, query: Any, owner_id: int, chat_id: int, selected: tuple[str, ...]
    ) -> None:
        selected_set = set(selected)
        rows: list[list[InlineKeyboardButton]] = []
        for template in STARTER_TEMPLATES:
            token = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                "starter_toggle",
                payload={"selected": list(selected), "key": template.key},
            )
            prefix = "✓ " if template.key in selected_set else ""
            rows.append(
                [InlineKeyboardButton(prefix + template.name, callback_data=f"collection:{token}")]
            )
        confirm = await self.collection_service.issue_action(
            owner_id,
            chat_id,
            "starter_confirm",
            payload={"selected": list(selected)},
        )
        rows.append(
            [
                InlineKeyboardButton(
                    f"Создать выбранные ({len(selected)})",
                    callback_data=f"collection:{confirm}",
                )
            ]
        )
        back = await self.collection_service.issue_action(owner_id, chat_id, "onboard_home")
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"collection:{back}")])
        rows.append(self._navigation_row())
        await self._collection_edit_or_send(
            query,
            "Выбери стартовые разделы. Это только шаблоны — их можно менять и удалять.",
            InlineKeyboardMarkup(rows),
        )

    async def _send_collection_hub(self, message: Any, owner_id: int, chat_id: int) -> None:
        text, markup = await self._collection_hub_view(owner_id, chat_id)
        await message.reply_text(text, reply_markup=markup)

    async def _render_collection_hub(
        self, query: Any, owner_id: int, chat_id: int, *, notice: str | None = None
    ) -> None:
        text, markup = await self._collection_hub_view(owner_id, chat_id)
        if notice:
            text = f"{notice}\n\n{text}"
        await self._collection_edit_or_send(query, text, markup)

    async def _collection_hub_view(
        self, owner_id: int, chat_id: int
    ) -> tuple[str, InlineKeyboardMarkup]:
        rows: list[list[InlineKeyboardButton]] = []
        for label, kind, status in (
            ("Все активные", None, "active"),
            ("Темы", "topic", "active"),
            ("Проекты", "project", "active"),
            ("Списки", "list", "active"),
            ("Архив", None, "archived"),
        ):
            token = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                "hub_list",
                payload={"kind": kind, "status": status, "page": 0},
            )
            rows.append([InlineKeyboardButton(label, callback_data=f"collection:{token}")])
        create_row: list[InlineKeyboardButton] = []
        for kind, label in (("topic", "+ Тема"), ("project", "+ Проект"), ("list", "+ Список")):
            token = await self.collection_service.issue_action(
                owner_id, chat_id, "create_type", payload={"kind": kind}
            )
            create_row.append(InlineKeyboardButton(label, callback_data=f"collection:{token}"))
        rows.append(create_row)
        rows.append(self._navigation_row())
        return (
            "🗂 Мои разделы\n\nТемы, проекты и списки связывают записи из Inbox и Task Hub.",
            InlineKeyboardMarkup(rows),
        )

    async def _render_collection_page(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        page: int,
        *,
        kind: CollectionKind | None,
        status: str,
    ) -> None:
        collection_page = await self.collection_service.list_page(
            owner_id, page, kind=kind, status=status
        )
        text, markup = await self._collection_page_view(owner_id, chat_id, collection_page)
        await self._collection_edit_or_send(query, text, markup)

    async def _collection_page_view(
        self, owner_id: int, chat_id: int, page: CollectionPage
    ) -> tuple[str, InlineKeyboardMarkup]:
        rows: list[list[InlineKeyboardButton]] = []
        for record in page.records:
            token = await self.collection_service.issue_action(
                owner_id, chat_id, "open", collection=record.collection
            )
            rows.append(
                [
                    InlineKeyboardButton(
                        f"{record.collection.name} · {record.item_count}",
                        callback_data=f"collection:{token}",
                    )
                ]
            )
        navigation: list[InlineKeyboardButton] = []
        for target, label in ((page.page - 1, "←"), (page.page + 1, "→")):
            if 0 <= target < page.pages:
                token = await self.collection_service.issue_action(
                    owner_id,
                    chat_id,
                    "hub_list",
                    payload={"kind": page.kind, "status": page.status, "page": target},
                )
                navigation.append(InlineKeyboardButton(label, callback_data=f"collection:{token}"))
        if navigation:
            rows.append(navigation)
        back = await self.collection_service.issue_action(owner_id, chat_id, "hub")
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"collection:{back}")])
        rows.append(self._navigation_row())
        label = COLLECTION_KIND_LABELS.get(page.kind, "разделы") if page.kind else "разделы"
        listing = "Здесь пока пусто." if not page.records else "Выбери раздел."
        return (
            f"{label.capitalize()} — {page.total}\nСтраница {page.page + 1}/{page.pages}\n\n{listing}",
            InlineKeyboardMarkup(rows),
        )

    async def _render_collection_card(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        collection_id: int,
        *,
        notice: str | None = None,
    ) -> None:
        summary = await self.collection_service.summary(owner_id, collection_id)
        if summary is None:
            await self._collection_stale_message(query)
            return
        await self.collection_service.set_context(owner_id, chat_id, collection_id)
        rows: list[list[InlineKeyboardButton]] = []
        if summary.collection.status == "active":
            for action, label in (
                ("content", "Открыть содержимое"),
                ("add_prompt", "Добавить запись"),
                ("rename_prompt", "Переименовать"),
                ("alias_prompt", "Добавить alias"),
                ("archive", "Архивировать"),
                ("delete_ask", "Удалить"),
            ):
                token = await self.collection_service.issue_action(
                    owner_id, chat_id, action, collection=summary.collection
                )
                rows.append([InlineKeyboardButton(label, callback_data=f"collection:{token}")])
        else:
            content = await self.collection_service.issue_action(
                owner_id, chat_id, "content", collection=summary.collection
            )
            restore = await self.collection_service.issue_action(
                owner_id, chat_id, "restore", collection=summary.collection
            )
            delete = await self.collection_service.issue_action(
                owner_id, chat_id, "delete_ask", collection=summary.collection
            )
            rows.extend(
                [
                    [
                        InlineKeyboardButton(
                            "Открыть содержимое", callback_data=f"collection:{content}"
                        )
                    ],
                    [InlineKeyboardButton("Восстановить", callback_data=f"collection:{restore}")],
                    [InlineKeyboardButton("Удалить", callback_data=f"collection:{delete}")],
                ]
            )
        back = await self.collection_service.issue_action(owner_id, chat_id, "hub")
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"collection:{back}")])
        rows.append(self._navigation_row())
        prefix = f"{notice}\n\n" if notice else ""
        text = (
            f"{prefix}<b>{escape(summary.collection.name)}</b>\n"
            f"Тип: {COLLECTION_KIND_LABELS[summary.collection.kind]}\n"
            f"Статус: {'активен' if summary.collection.status == 'active' else 'в архиве'}\n"
            f"Записей: {summary.item_count}"
        )
        await self._collection_edit_or_send(
            query, text, InlineKeyboardMarkup(rows), parse_mode="HTML"
        )

    async def _render_collection_content(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        collection_id: int,
        page: int,
        *,
        notice: str | None = None,
    ) -> None:
        view = await self._collection_content_view(
            owner_id, chat_id, collection_id, page, notice=notice
        )
        if view is None:
            await self._collection_stale_message(query)
            return
        await self._collection_edit_or_send(query, *view, parse_mode="HTML")

    async def _send_collection_content(
        self, message: Any, owner_id: int, chat_id: int, collection_id: int, page: int
    ) -> None:
        view = await self._collection_content_view(owner_id, chat_id, collection_id, page)
        if view is None:
            await message.reply_text("Раздел больше недоступен.")
            return
        text, markup = view
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _collection_content_view(
        self,
        owner_id: int,
        chat_id: int,
        collection_id: int,
        page: int,
        *,
        notice: str | None = None,
    ) -> tuple[str, InlineKeyboardMarkup] | None:
        item_page = await self.collection_service.item_page(owner_id, collection_id, page)
        if item_page is None:
            return None
        await self.collection_service.set_context(owner_id, chat_id, collection_id)
        rows: list[list[InlineKeyboardButton]] = []
        lines: list[str] = []
        for index, record in enumerate(item_page.records, start=1 + item_page.page * 6):
            marker = "✅" if record.task_state and record.task_state.status == "completed" else "•"
            lines.append(f"{index}. {marker} {escape(record.item.title)}")
            token = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                "item_open",
                collection=item_page.collection,
                inbox_item_id=record.item.id,
            )
            rows.append(
                [InlineKeyboardButton(f"Открыть {index}", callback_data=f"collection:{token}")]
            )
        pagination: list[InlineKeyboardButton] = []
        for target, label in ((item_page.page - 1, "←"), (item_page.page + 1, "→")):
            if 0 <= target < item_page.pages:
                token = await self.collection_service.issue_action(
                    owner_id,
                    chat_id,
                    "content",
                    collection=item_page.collection,
                    payload={"page": target},
                )
                pagination.append(InlineKeyboardButton(label, callback_data=f"collection:{token}"))
        if pagination:
            rows.append(pagination)
        if item_page.collection.status == "active":
            add = await self.collection_service.issue_action(
                owner_id, chat_id, "add_prompt", collection=item_page.collection
            )
            rows.append(
                [InlineKeyboardButton("Добавить запись", callback_data=f"collection:{add}")]
            )
        back = await self.collection_service.issue_action(
            owner_id, chat_id, "open", collection=item_page.collection
        )
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"collection:{back}")])
        rows.append(self._navigation_row())
        prefix = f"{escape(notice)}\n\n" if notice else ""
        listing = "\n".join(lines) if lines else "Здесь пока нет записей."
        text = (
            f"{prefix}<b>{escape(item_page.collection.name)}</b> — {item_page.total}\n"
            f"Страница {item_page.page + 1}/{item_page.pages}\n\n{listing}"
        )
        return text, InlineKeyboardMarkup(rows)

    async def _render_collection_item(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        collection_id: int,
        inbox_item_id: int,
    ) -> None:
        record = await self.collection_service.item_record(owner_id, collection_id, inbox_item_id)
        summary = await self.collection_service.summary(owner_id, collection_id)
        if record is None or summary is None:
            await self._collection_stale_message(query)
            return
        await self.collection_service.set_context(
            owner_id, chat_id, collection_id, last_inbox_item_id=inbox_item_id
        )
        rows: list[list[InlineKeyboardButton]] = []
        if record.task_state is not None:
            task_token = (
                await self.task_service.issue_actions(
                    owner_id,
                    chat_id,
                    inbox_item_id,
                    record.task_state.version,
                    ("view",),
                    payload={"bucket": "no_due", "page": 0},
                )
            ).get("view")
            if task_token:
                rows.append(
                    [InlineKeyboardButton("Открыть в Task Hub", callback_data=f"task:{task_token}")]
                )
        for action, label in (
            ("move_menu", "Переместить"),
            ("link_menu", "Связать ещё"),
            ("unlink", "Убрать из раздела"),
            ("delete_item_ask", "Удалить запись"),
        ):
            token = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                action,
                collection=summary.collection,
                inbox_item_id=inbox_item_id,
            )
            rows.append([InlineKeyboardButton(label, callback_data=f"collection:{token}")])
        back = await self.collection_service.issue_action(
            owner_id,
            chat_id,
            "content",
            collection=summary.collection,
            payload={"page": 0},
        )
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"collection:{back}")])
        rows.append(self._navigation_row())
        status = ""
        if record.task_state is not None:
            status = (
                "\nСтатус задачи: выполнена"
                if record.task_state.status == "completed"
                else "\nСтатус задачи: активна"
            )
        await self._collection_edit_or_send(
            query,
            f"<b>{escape(record.item.title)}</b>\nТип записи: {escape(record.item.kind)}{status}",
            InlineKeyboardMarkup(rows),
            parse_mode="HTML",
        )

    async def _render_collection_targets(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        source_id: int,
        inbox_item_id: int,
        *,
        mode: str,
        page: int,
    ) -> None:
        source = await self.collection_service.summary(owner_id, source_id)
        if source is None:
            await self._collection_stale_message(query)
            return
        targets = await self.collection_service.active_collections(owner_id, exclude_id=source_id)
        page_size = 6
        pages = max(1, (len(targets) + page_size - 1) // page_size)
        safe_page = min(max(page, 0), pages - 1)
        visible = targets[safe_page * page_size : (safe_page + 1) * page_size]
        rows: list[list[InlineKeyboardButton]] = []
        for target in visible:
            token = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                "move_to" if mode == "move" else "link_to",
                collection=source.collection,
                inbox_item_id=inbox_item_id,
                payload={"target_id": target.id, "target_version": target.version},
            )
            rows.append([InlineKeyboardButton(target.name, callback_data=f"collection:{token}")])
        pagination: list[InlineKeyboardButton] = []
        for target_page, label in ((safe_page - 1, "←"), (safe_page + 1, "→")):
            if 0 <= target_page < pages:
                token = await self.collection_service.issue_action(
                    owner_id,
                    chat_id,
                    "move_menu" if mode == "move" else "link_menu",
                    collection=source.collection,
                    inbox_item_id=inbox_item_id,
                    payload={"page": target_page},
                )
                pagination.append(InlineKeyboardButton(label, callback_data=f"collection:{token}"))
        if pagination:
            rows.append(pagination)
        back = await self.collection_service.issue_action(
            owner_id,
            chat_id,
            "item_open",
            collection=source.collection,
            inbox_item_id=inbox_item_id,
        )
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"collection:{back}")])
        rows.append(self._navigation_row())
        await self._collection_edit_or_send(
            query,
            (
                f"Выбери целевой раздел. Страница {safe_page + 1}/{pages}."
                if targets
                else "Других активных разделов пока нет."
            ),
            InlineKeyboardMarkup(rows),
        )

    async def _render_delete_choice(
        self, query: Any, owner_id: int, chat_id: int, summary: CollectionSummary
    ) -> None:
        rows: list[list[InlineKeyboardButton]] = []
        if summary.item_count:
            unlink = await self.collection_service.issue_action(
                owner_id, chat_id, "delete_links", collection=summary.collection
            )
            text = (
                "Раздел не пуст. Можно удалить только его связи — исходные записи останутся — "
                + (
                    "либо архивировать раздел."
                    if summary.collection.status == "active"
                    else "либо восстановить раздел."
                )
            )
            rows.append(
                [InlineKeyboardButton("Удалить только связи", callback_data=f"collection:{unlink}")]
            )
            state_action = "archive" if summary.collection.status == "active" else "restore"
            state_label = "Архивировать" if state_action == "archive" else "Восстановить"
            state_token = await self.collection_service.issue_action(
                owner_id, chat_id, state_action, collection=summary.collection
            )
            rows.append(
                [InlineKeyboardButton(state_label, callback_data=f"collection:{state_token}")]
            )
        else:
            delete_token = await self.collection_service.issue_action(
                owner_id, chat_id, "delete_empty", collection=summary.collection
            )
            text = "Раздел пуст. Удалить его?"
            rows.append(
                [InlineKeyboardButton("Удалить", callback_data=f"collection:{delete_token}")]
            )
        back = await self.collection_service.issue_action(
            owner_id, chat_id, "open", collection=summary.collection
        )
        rows.append([InlineKeyboardButton("← Назад", callback_data=f"collection:{back}")])
        rows.append(self._navigation_row())
        await self._collection_edit_or_send(query, text, InlineKeyboardMarkup(rows))

    async def _send_collection_candidates(
        self,
        message: Any,
        owner_id: int,
        chat_id: int,
        candidates: tuple[LifeCollection, ...],
        *,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        rows: list[list[InlineKeyboardButton]] = []
        for collection in candidates:
            token = await self.collection_service.issue_action(
                owner_id,
                chat_id,
                action,
                collection=collection,
                payload=payload,
            )
            rows.append(
                [InlineKeyboardButton(collection.name, callback_data=f"collection:{token}")]
            )
        rows.append(self._navigation_row())
        await message.reply_text(
            "Уточни раздел — ничего не сохранено до выбора.",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _claim(
        self, token: str, owner_id: int, chat_id: int, action: str
    ) -> CollectionClaim | None:
        return await self.collection_service.claim_action(token, owner_id, chat_id, {action})

    async def _pending_payload(self, owner_id: int, chat_id: int) -> dict[str, Any]:
        pending = await self.collection_service.pending_input(owner_id, chat_id)
        return dict(pending.payload or {}) if pending else {}

    @staticmethod
    def _payload_kind(value: object) -> CollectionKind | None:
        return value if value in COLLECTION_KIND_LABELS else None

    @staticmethod
    def _preview_text(collection: LifeCollection, items: tuple[str, ...]) -> str:
        lines = "\n".join(f"• {item}" for item in items)
        return f"Проверь разбиение для «{collection.name}»:\n\n{lines}"

    @staticmethod
    def _navigation_row() -> list[InlineKeyboardButton]:
        return [
            InlineKeyboardButton("🏠 В меню", callback_data="nav:root"),
            InlineKeyboardButton("❓ Помощь", callback_data="nav:help"),
        ]

    @classmethod
    def _navigation_markup(cls) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([cls._navigation_row()])

    @staticmethod
    async def _collection_stale(query: Any) -> None:
        await query.answer("Эта кнопка устарела или недоступна.", show_alert=True)

    @classmethod
    async def _collection_stale_message(cls, query: Any) -> None:
        await cls._collection_edit_or_send(
            query,
            "Действие устарело. Открой /collections ещё раз.",
            cls._navigation_markup(),
        )

    @staticmethod
    async def _collection_edit_or_send(
        query: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup,
        *,
        parse_mode: str | None = None,
    ) -> None:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
        except (TelegramError, TypeError):
            await query.message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)


def re_split_once(value: str) -> tuple[str, str]:
    parts = re_sub_spaces(value).split(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (value, "")


def re_sub_spaces(value: str) -> str:
    import re

    return re.sub(r"\s+", " ", value).strip()
