from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from .reminders import as_utc
from .tasks import BUCKET_LABELS, TaskBucket, TaskRecord, TaskResult


class TaskHandlers:
    task_service: Any

    async def tasks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        await update.effective_message.reply_text(
            "✅ Задачи и напоминания\n\nВыбери список или действие.",
            reply_markup=self._task_hub_keyboard(),
        )

    async def task_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._task_list_entry(update, context, "today")

    async def task_upcoming(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._task_list_entry(update, context, "upcoming")

    async def task_overdue(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._task_list_entry(update, context, "overdue")

    async def task_no_due(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._task_list_entry(update, context, "no_due")

    async def task_completed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._task_list_entry(update, context, "completed")

    async def task_create(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        await update.effective_message.reply_text(
            "Создание задачи\n\nНапиши или скажи задачу обычной фразой. "
            "Я покажу preview, и задача появится только после подтверждения.\n\n"
            "Примеры:\n"
            "• Завтра в 18:00 купить продукты.\n"
            "• Напомни через час позвонить.\n"
            "• Записаться на тренировку в субботу.",
            reply_markup=self._task_back_keyboard(),
        )

    async def task_reminder_guide(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        await update.effective_message.reply_text(
            "Как работают напоминания\n\n"
            "Срок задачи и время напоминания хранятся отдельно. Перенос срока может "
            "сохранить прежний интервал или назначить новое напоминание. Завершение и "
            "удаление безопасно отменяют ожидающую доставку. После возврата выполненной "
            "задачи напоминание включается только явно.",
            reply_markup=self._task_back_keyboard(),
        )

    async def task_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        data = query.data or ""
        if hasattr(self, "collection_service"):
            user = await self._user(update.effective_user.id)
            await self.collection_service.clear_context(user.id, update.effective_chat.id)
            await self.collection_service.cancel_input(user.id, update.effective_chat.id)
        if data == "task:hub":
            await query.answer()
            await self._task_edit_or_send(
                query,
                "✅ Задачи и напоминания\n\nВыбери список или действие.",
                self._task_hub_keyboard(),
            )
            return
        if data.startswith("task:list:"):
            parts = data.split(":")
            if len(parts) != 4 or parts[2] not in BUCKET_LABELS or not parts[3].isdigit():
                await self._task_stale(query)
                return
            await query.answer()
            await self._send_task_page(
                query,
                update.effective_user.id,
                update.effective_chat.id,
                parts[2],
                int(parts[3]),
                edit=True,
            )
            return
        if not data.startswith("task:") or data.count(":") != 1:
            await self._task_stale(query)
            return
        token = data.removeprefix("task:")
        if not token or len(token) > 32:
            await self._task_stale(query)
            return
        user = await self._user(update.effective_user.id)
        action = await self.task_service.capability_action(token, user.id, update.effective_chat.id)
        if action is None:
            await self._task_stale(query)
            return
        await query.answer()
        if action == "view":
            result = await self.task_service.open_from_token(
                token, user.id, update.effective_chat.id
            )
            bucket = (result.tokens or {}).get("bucket", "today")
            page = int((result.tokens or {}).get("page", "0"))
            await self._render_result(
                query, user.id, update.effective_chat.id, result, bucket, page
            )
            return
        if action == "complete":
            result = await self.task_service.complete(token, user.id, update.effective_chat.id)
            await self._render_result(
                query,
                user.id,
                update.effective_chat.id,
                result,
                "completed",
                0,
                notice=(
                    "Задача уже выполнена."
                    if result.status == "already_completed"
                    else "Готово — задача выполнена, ожидающее напоминание отменено."
                ),
            )
            return
        if action == "reopen":
            result = await self.task_service.reopen(token, user.id, update.effective_chat.id)
            await self._render_result(
                query,
                user.id,
                update.effective_chat.id,
                result,
                self._default_bucket(result.record),
                0,
                notice="Задача возвращена в активные. Старое напоминание не включено.",
            )
            return
        if action == "reminder_off":
            result = await self.task_service.disable_reminder(
                token, user.id, update.effective_chat.id
            )
            notice = (
                "Напоминание уже выключено или было отправлено."
                if result.status == "already_off"
                else "Напоминание отключено. Срок задачи не изменён."
            )
            await self._render_result(
                query,
                user.id,
                update.effective_chat.id,
                result,
                self._default_bucket(result.record),
                0,
                notice=notice,
            )
            return
        if action == "reschedule_menu":
            result = await self.task_service.reschedule_menu(
                token, user.id, update.effective_chat.id
            )
            if result.status != "menu" or not result.tokens:
                await self._task_stale_message(query)
                return
            await self._task_edit_or_send(
                query,
                "Перенести срок\n\nВыбери новый срок. Ничего не изменится до подтверждённого выбора.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Через 30 минут",
                                callback_data=f"task:{result.tokens['30m']}",
                            ),
                            InlineKeyboardButton(
                                "Через 1 час",
                                callback_data=f"task:{result.tokens['1h']}",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                "Завтра в 09:00",
                                callback_data=f"task:{result.tokens['tomorrow']}",
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                "Другая дата и время",
                                callback_data=f"task:{result.tokens['custom']}",
                            )
                        ],
                        [InlineKeyboardButton("Отмена", callback_data="task:hub")],
                    ]
                ),
            )
            return
        if action == "reschedule_at":
            result = await self.task_service.choose_reschedule_preset(
                token, user.id, update.effective_chat.id
            )
            await self._handle_reschedule_result(query, user.id, update.effective_chat.id, result)
            return
        if action in {"reschedule_preserve", "reschedule_new_reminder"}:
            result = await self.task_service.apply_reschedule_choice(
                token, user.id, update.effective_chat.id
            )
            if result.status == "await_reminder":
                await self._task_edit_or_send(
                    query,
                    "Срок перенесён. Теперь пришли новое время напоминания, например: "
                    "завтра в 17:30 или через 1 час. /cancel — отменить ввод.",
                    self._task_back_keyboard(),
                )
                return
            await self._render_result(
                query,
                user.id,
                update.effective_chat.id,
                result,
                self._default_bucket(result.record),
                0,
                notice="Срок перенесён, интервал напоминания сохранён.",
            )
            return
        if action == "reminder_edit":
            result = await self.task_service.start_reminder_input(
                token, user.id, update.effective_chat.id
            )
            if result.status != "await_reminder":
                await self._task_stale_message(query)
                return
            await self._task_edit_or_send(
                query,
                "Пришли новую дату и время напоминания без голосовой модели, например: "
                "завтра в 17:30 или через 1 час. /cancel — отменить ввод.",
                self._task_back_keyboard(),
            )
            return
        if action == "delete_ask":
            result = await self.task_service.prepare_delete(
                token, user.id, update.effective_chat.id
            )
            if result.status != "confirm_delete" or not result.tokens:
                await self._task_stale_message(query)
                return
            await self._task_edit_or_send(
                query,
                "Удалить задачу? Напоминание будет отменено, а связанная карточка желания останется.",
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Да, удалить",
                                callback_data=f"task:{result.tokens['delete_confirm']}",
                            ),
                            InlineKeyboardButton(
                                "Отмена",
                                callback_data=f"task:{result.tokens['delete_cancel']}",
                            ),
                        ]
                    ]
                ),
            )
            return
        if action in {"delete_confirm", "delete_cancel"}:
            result = await self.task_service.delete_or_cancel(
                token, user.id, update.effective_chat.id
            )
            if result.status == "deleted":
                await self._task_edit_or_send(
                    query,
                    "Задача удалена. Связанная карточка желания сохранена.",
                    self._task_hub_keyboard(),
                )
            elif result.status == "delete_cancelled":
                await self._render_result(
                    query,
                    user.id,
                    update.effective_chat.id,
                    result,
                    self._default_bucket(result.record),
                    0,
                    notice="Удаление отменено.",
                )
            else:
                await self._task_stale_message(query)
            return
        await self._task_stale_message(query)

    async def task_pending_text(self, update: Update) -> bool:
        user = await self._user(update.effective_user.id)
        pending = await self.task_service.pending_input(user.id, update.effective_chat.id)
        if pending is None:
            return False
        record = await self.task_service.record(user.id, pending.inbox_item_id)
        if record is None or record.state.version != pending.task_version:
            await self.task_service.cancel_pending_input(user.id, update.effective_chat.id)
            await update.effective_message.reply_text(
                "Это действие устарело: задача уже изменилась. Открой её снова через /tasks."
            )
            return True
        parsed = self.task_service.parse_datetime(
            update.effective_message.text or "",
            record.state.timezone,
        )
        if parsed.status != "resolved":
            await update.effective_message.reply_text(
                parsed.message
                or "Не удалось определить дату и время. Пример: завтра в 18:00 или через 1 час."
            )
            return True
        result = await self.task_service.submit_pending_input(
            pending.token,
            user.id,
            update.effective_chat.id,
            parsed,
        )
        if result.status == "choose_reminder":
            await self._send_reminder_choice(update.effective_message, result)
        elif result.status == "reminder_changed":
            await self._send_record(
                update.effective_message,
                user.id,
                update.effective_chat.id,
                result.record,
                self._default_bucket(result.record),
                0,
                notice="Напоминание обновлено.",
            )
        elif result.status == "rescheduled":
            await self._send_record(
                update.effective_message,
                user.id,
                update.effective_chat.id,
                result.record,
                self._default_bucket(result.record),
                0,
                notice="Срок перенесён.",
            )
        else:
            await update.effective_message.reply_text(
                "Действие устарело: задача уже изменилась. Открой её снова через /tasks."
            )
        return True

    async def cancel_task_input(self, update: Update) -> bool:
        user = await self._user(update.effective_user.id)
        record = await self.task_service.cancel_pending_input(user.id, update.effective_chat.id)
        if record is None:
            return False
        await self._send_record(
            update.effective_message,
            user.id,
            update.effective_chat.id,
            record,
            self._default_bucket(record),
            0,
            notice="Ввод отменён. Новое напоминание не создано.",
        )
        return True

    async def _task_list_entry(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        bucket: TaskBucket,
    ) -> None:
        del context
        await self._send_task_page(
            update.effective_message,
            update.effective_user.id,
            update.effective_chat.id,
            bucket,
            0,
            edit=False,
        )

    async def _send_task_page(
        self,
        target: Any,
        telegram_user_id: int,
        chat_id: int,
        bucket: TaskBucket,
        page: int,
        *,
        edit: bool,
    ) -> None:
        user = await self._user(telegram_user_id)
        task_page = await self.task_service.list_page(user.id, bucket, page)
        rows: list[list[InlineKeyboardButton]] = []
        lines: list[str] = []
        for index, record in enumerate(
            task_page.records, start=task_page.page * self.task_service.PAGE_SIZE + 1
        ):
            tokens = await self.task_service.issue_actions(
                user.id,
                chat_id,
                record.item.id,
                record.state.version,
                ("view",),
                payload={"bucket": bucket, "page": task_page.page},
            )
            if not tokens:
                continue
            due = self._compact_due(record)
            lines.append(f"{index}. {escape(record.item.title)}{due}")
            rows.append(
                [InlineKeyboardButton(f"Открыть {index}", callback_data=f"task:{tokens['view']}")]
            )
        pagination: list[InlineKeyboardButton] = []
        if task_page.page > 0:
            pagination.append(
                InlineKeyboardButton(
                    "← Назад", callback_data=f"task:list:{bucket}:{task_page.page - 1}"
                )
            )
        if task_page.page + 1 < task_page.pages:
            pagination.append(
                InlineKeyboardButton(
                    "Далее →", callback_data=f"task:list:{bucket}:{task_page.page + 1}"
                )
            )
        if pagination:
            rows.append(pagination)
        rows.extend(
            [
                [InlineKeyboardButton("← Назад", callback_data="task:hub")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            ]
        )
        listing = "\n".join(lines) if lines else "В этом списке пока нет задач."
        text = (
            f"{BUCKET_LABELS[bucket]} — {task_page.total}\n"
            f"Страница {task_page.page + 1}/{task_page.pages}\n\n{listing}"
        )
        markup = InlineKeyboardMarkup(rows)
        if edit:
            await self._task_edit_or_send(target, text, markup, parse_mode="HTML")
        else:
            await target.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _render_result(
        self,
        query: Any,
        owner_id: int,
        chat_id: int,
        result: TaskResult,
        bucket: str,
        page: int,
        *,
        notice: str | None = None,
    ) -> None:
        if result.record is None:
            await self._task_stale_message(query)
            return
        text, markup = await self._card(
            owner_id,
            chat_id,
            result.record,
            bucket if bucket in BUCKET_LABELS else self._default_bucket(result.record),
            page,
            notice=notice,
        )
        await self._task_edit_or_send(query, text, markup, parse_mode="HTML")

    async def _send_record(
        self,
        message: Any,
        owner_id: int,
        chat_id: int,
        record: TaskRecord | None,
        bucket: str,
        page: int,
        *,
        notice: str | None = None,
    ) -> None:
        if record is None:
            await message.reply_text("Действие устарело. Открой задачу снова через /tasks.")
            return
        text, markup = await self._card(owner_id, chat_id, record, bucket, page, notice=notice)
        await message.reply_text(text, reply_markup=markup, parse_mode="HTML")

    async def _card(
        self,
        owner_id: int,
        chat_id: int,
        record: TaskRecord,
        bucket: str,
        page: int,
        *,
        notice: str | None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        state, item, reminder = record.state, record.item, record.reminder
        status_label = {
            "active": "активна",
            "completed": "выполнена",
            "cancelled": "отменена",
        }.get(state.status, "неизвестен")
        due = "без срока"
        if state.event_at is not None:
            local_event = as_utc(state.event_at).astimezone(ZoneInfo(state.timezone))
            precision = self._task_precision(record)
            due = (
                local_event.strftime("%d.%m.%Y")
                if precision == "date"
                else local_event.strftime("%d.%m.%Y %H:%M")
            )
            due += f" ({escape(state.timezone)})"
        reminder_text = "нет"
        if reminder is not None:
            local_reminder = as_utc(reminder.remind_at).astimezone(ZoneInfo(reminder.timezone))
            if reminder.status in {"pending", "processing"}:
                reminder_text = local_reminder.strftime("%d.%m.%Y %H:%M")
            elif reminder.status == "sent":
                reminder_text = f"отправлено {local_reminder.strftime('%d.%m.%Y %H:%M')}"
            else:
                reminder_text = "отключено"
        description = ""
        if item.description and not item.source.startswith("doctor"):
            description = f"\nОписание: {escape(item.description)}"
        source = {
            "vision": "Карта желаний",
            "doctor_prepare": "Раздел «Врач»",
            "doctor_search": "Раздел «Врач»",
            "voice": "Голосовая запись",
        }.get(item.source)
        source_text = f"\nИсточник: {source}" if source else ""
        vision_text = "\nСвязь: карточка желания" if record.vision_linked else ""
        prefix = f"{escape(notice)}\n\n" if notice else ""
        text = (
            f"{prefix}<b>{escape(item.title)}</b>{description}\n"
            f"Срок: {due}\n"
            f"Напоминание: {reminder_text}\n"
            f"Статус: {status_label}{source_text}{vision_text}"
        )
        if state.status == "completed":
            actions = ("reopen", "delete_ask")
        else:
            actions = (
                "complete",
                "reschedule_menu",
                "reminder_edit",
                "reminder_off",
                "delete_ask",
            )
        tokens = await self.task_service.issue_actions(
            owner_id,
            chat_id,
            item.id,
            state.version,
            actions,
        )
        rows: list[list[InlineKeyboardButton]] = []
        if state.status == "completed":
            rows.append(
                [
                    InlineKeyboardButton(
                        "Вернуть в активные", callback_data=f"task:{tokens['reopen']}"
                    )
                ]
            )
        else:
            rows.extend(
                [
                    [InlineKeyboardButton("Выполнено", callback_data=f"task:{tokens['complete']}")],
                    [
                        InlineKeyboardButton(
                            "Перенести", callback_data=f"task:{tokens['reschedule_menu']}"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Изменить напоминание", callback_data=f"task:{tokens['reminder_edit']}"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Отключить напоминание", callback_data=f"task:{tokens['reminder_off']}"
                        )
                    ],
                ]
            )
        rows.extend(
            [
                [InlineKeyboardButton("Удалить", callback_data=f"task:{tokens['delete_ask']}")],
                [
                    InlineKeyboardButton(
                        "← Назад к списку",
                        callback_data=f"task:list:{bucket}:{max(page, 0)}",
                    )
                ],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            ]
        )
        return text, InlineKeyboardMarkup(rows)

    async def _handle_reschedule_result(
        self, query: Any, owner_id: int, chat_id: int, result: TaskResult
    ) -> None:
        if result.status == "await_event":
            await self._task_edit_or_send(
                query,
                "Пришли новую дату и время, например: завтра в 18:00 или через 1 час. "
                "/cancel — отменить ввод.",
                self._task_back_keyboard(),
            )
        elif result.status == "choose_reminder":
            await self._send_reminder_choice(query, result, edit=True)
        elif result.status == "rescheduled":
            await self._render_result(
                query,
                owner_id,
                chat_id,
                result,
                self._default_bucket(result.record),
                0,
                notice="Срок перенесён.",
            )
        else:
            await self._task_stale_message(query)

    async def _send_reminder_choice(
        self, target: Any, result: TaskResult, *, edit: bool = False
    ) -> None:
        if not result.tokens:
            if edit:
                await self._task_stale_message(target)
            else:
                await target.reply_text("Действие устарело. Открой задачу снова через /tasks.")
            return
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Сохранить прежний интервал",
                        callback_data=f"task:{result.tokens['reschedule_preserve']}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        "Выбрать новое напоминание",
                        callback_data=f"task:{result.tokens['reschedule_new_reminder']}",
                    )
                ],
                [InlineKeyboardButton("Отмена", callback_data="task:hub")],
            ]
        )
        text = "Новый срок выбран. Что сделать с напоминанием?"
        if edit:
            await self._task_edit_or_send(target, text, markup)
        else:
            await target.reply_text(text, reply_markup=markup)

    @staticmethod
    def _compact_due(record: TaskRecord) -> str:
        if record.state.event_at is None:
            return ""
        local = as_utc(record.state.event_at).astimezone(ZoneInfo(record.state.timezone))
        precision = TaskHandlers._task_precision(record)
        return " — " + (
            local.strftime("%d.%m") if precision == "date" else local.strftime("%d.%m %H:%M")
        )

    @staticmethod
    def _default_bucket(record: TaskRecord | None) -> TaskBucket:
        if record is None or record.state.status == "completed":
            return "completed"
        if record.state.event_at is None:
            return "no_due"
        now = datetime.now(UTC)
        event = as_utc(record.state.event_at)
        zone = ZoneInfo(record.state.timezone)
        if event < now:
            return "overdue"
        if event.astimezone(zone).date() == now.astimezone(zone).date():
            return "today"
        return "upcoming"

    @staticmethod
    def _task_precision(record: TaskRecord) -> str | None:
        temporal = record.item.temporal_resolution
        if isinstance(temporal, dict) and temporal.get("precision") in {"date", "datetime"}:
            return str(temporal["precision"])
        return "date" if record.item.resolved_date is not None else None

    @staticmethod
    def _task_hub_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Сегодня", callback_data="task:list:today:0"),
                    InlineKeyboardButton("Предстоящие", callback_data="task:list:upcoming:0"),
                ],
                [
                    InlineKeyboardButton("Просроченные", callback_data="task:list:overdue:0"),
                    InlineKeyboardButton("Без срока", callback_data="task:list:no_due:0"),
                ],
                [InlineKeyboardButton("Выполненные", callback_data="task:list:completed:0")],
                [InlineKeyboardButton("Создать задачу", callback_data="nav:action:task_create")],
                [
                    InlineKeyboardButton(
                        "Как работают напоминания",
                        callback_data="nav:action:task_reminder_guide",
                    )
                ],
                [InlineKeyboardButton("← Назад", callback_data="nav:root")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            ]
        )

    @staticmethod
    def _task_back_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("← Назад", callback_data="task:hub")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="nav:root")],
            ]
        )

    @staticmethod
    async def _task_stale(query: Any) -> None:
        await query.answer("Эта кнопка устарела или недоступна.", show_alert=True)

    @staticmethod
    async def _task_stale_message(query: Any) -> None:
        await TaskHandlers._task_edit_or_send(
            query,
            "Эта кнопка устарела или задача уже изменилась. Открой список через /tasks.",
            TaskHandlers._task_hub_keyboard(),
        )

    @staticmethod
    async def _task_edit_or_send(
        query: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        parse_mode: str | None = None,
    ) -> None:
        try:
            await query.edit_message_text(
                text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )
        except (TelegramError, TypeError):
            await query.message.reply_text(
                text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )
