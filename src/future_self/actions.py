import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from .drafts import DraftInboxService, DraftResult, masked_user
from .models import DraftInboxItem
from .schemas import ParsedThought

logger = logging.getLogger(__name__)

DraftAction = Literal["save", "edit", "discard", "confirm_date", "create_task", "cancel"]
ActionSource = Literal["callback", "voice_command", "text_command"]


@dataclass(slots=True, frozen=True)
class CommandMatch:
    action: DraftAction | None = None
    confidence: float = 0.0
    handled_without_action: bool = False
    needs_confirmation: bool = False


@dataclass(slots=True, frozen=True)
class ActionRoute:
    kind: Literal["none", "action", "selection", "confirmation", "control"]
    action: DraftAction | None = None
    selector: Literal["first", "second", "last", "newest", "topic", "reply"] | None = None
    query: str | None = None
    needs_confirmation: bool = False


@dataclass(slots=True)
class ActionOutcome:
    status: Literal["ok", "missing", "ambiguous", "stale"]
    result: DraftResult | None = None
    previous_message_id: int | None = None


class DraftCommandInterpreter:
    """Conservative deterministic parser for high-confidence draft commands."""

    SAVE = {
        "сохрани",
        "да сохрани",
        "можешь сохранить",
        "оставь это",
        "добавь в inbox",
        "добавь это в inbox",
        "сохрани в inbox",
        "сохрани это в inbox",
        "добавь в инбокс",
        "добавь это в инбокс",
    }
    CREATE_TASK = {
        "добавь к задачам",
        "добавь это к задачам",
        "создай из этого задачу",
        "запланируй это",
    }
    EDIT = {"редактировать", "редактируй", "измени это", "измени последнее"}
    DISCARD = {"не сохраняй", "не сохранять", "удали черновик"}
    CANCEL = {"отмена", "отмени", "отменить"}
    DEFER = (
        "не сохраняй пока",
        "пока не сохраняй",
        "возможно",
        "может быть",
        "добавим позже",
    )
    NON_ACTION = ("ты это сохранишь", "можно будет сохранить", "как думаешь")

    def parse(self, text: str) -> CommandMatch:
        normalized = self._normalize(text)
        if any(marker in normalized for marker in self.DEFER):
            return CommandMatch(handled_without_action=True)
        if normalized in self.SAVE:
            return CommandMatch(action="save", confidence=1.0)
        if normalized in self.CREATE_TASK:
            return CommandMatch(action="create_task", confidence=1.0)
        if normalized in self.EDIT:
            return CommandMatch(action="edit", confidence=1.0)
        if normalized in self.DISCARD:
            return CommandMatch(action="discard", confidence=1.0)
        if normalized in self.CANCEL:
            return CommandMatch(action="cancel", confidence=1.0)
        if any(marker in normalized for marker in self.NON_ACTION):
            return CommandMatch(handled_without_action=True)
        if "сохрани" in normalized and any(
            marker in normalized for marker in ("наверное", "пожалуй", "может")
        ):
            return CommandMatch(action="save", confidence=0.8, needs_confirmation=True)
        return CommandMatch()

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.strip().lower().replace("ё", "е")
        lowered = re.sub(r"[.!?]+$", "", lowered)
        lowered = re.sub(r"[,;:]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered


class ActionCommandRouter:
    """Route draft-control language before any content Intent Router call."""

    CONFIRMATIONS = (
        "да",
        "да да",
        "все правильно",
        "да все правильно",
        "подтверждаю",
        "пожалуйста сохрани",
    )
    CONTROL_ONLY = (
        "ну ты запишешь или как",
        "я же сказал сохранить",
        "какую именно",
    )

    def __init__(self) -> None:
        self.commands = DraftCommandInterpreter()

    def route(self, text: str, *, has_pending_action: bool) -> ActionRoute:
        normalized = self.commands._normalize(text)
        if has_pending_action and self._is_confirmation(normalized):
            return ActionRoute(kind="confirmation")
        if normalized in {"последнюю", "последний"}:
            return ActionRoute(kind="selection", selector="last")
        if normalized in {"самую новую", "новую", "самый новый"}:
            return ActionRoute(kind="selection", selector="newest")
        if normalized in {"первую", "первый"}:
            return ActionRoute(kind="selection", selector="first")
        if normalized in {"вторую", "второй"}:
            return ActionRoute(kind="selection", selector="second")
        if normalized in {"вот эту", "эту"}:
            return ActionRoute(kind="selection", selector="reply")
        topic_match = re.search(r"^(?:ту\s+что\s+)?про\s+(.+)$", normalized)
        if topic_match:
            return ActionRoute(kind="selection", selector="topic", query=topic_match.group(1))
        command = self.commands.parse(text)
        if command.action:
            return ActionRoute(
                kind="action",
                action=command.action,
                needs_confirmation=command.needs_confirmation,
            )
        if command.handled_without_action or any(
            marker in normalized for marker in self.CONTROL_ONLY
        ):
            return ActionRoute(kind="control")
        if any(
            marker in normalized for marker in ("да ее", "все правильно", "подтверждаю", "запишешь")
        ):
            return ActionRoute(kind="control")
        return ActionRoute(kind="none")

    @classmethod
    def _is_confirmation(cls, normalized: str) -> bool:
        return (
            normalized in cls.CONFIRMATIONS
            or normalized.startswith("да да")
            or "все правильно" in normalized
            or "подтверждаю" in normalized
            or ("сохрани" in normalized and "не сохрани" not in normalized)
        )


class DraftActionService:
    """Shared action path for Telegram callbacks and conversational commands."""

    def __init__(self, drafts: DraftInboxService):
        self.drafts = drafts

    async def execute(
        self,
        action: DraftAction,
        *,
        telegram_user_id: int,
        chat_id: int,
        source: ActionSource,
        draft_id: str | None = None,
        version: int | None = None,
        task: ParsedThought | None = None,
        user_id: int | None = None,
        raw_text: str | None = None,
        resolved_date: date | None = None,
    ) -> ActionOutcome:
        target: DraftInboxItem | None = None
        if draft_id is not None and version is not None:
            target = await self.drafts.get(draft_id)
        else:
            candidates = await self.drafts.active_previews(telegram_user_id, chat_id)
            if len(candidates) > 1:
                return ActionOutcome("ambiguous")
            if candidates:
                target = candidates[0]
                draft_id, version = target.id, target.version

        if action == "create_task" and target is None and task and user_id is not None:
            creation = await self.drafts.create_or_get(
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source="text" if source == "text_command" else "voice",
                raw_text=raw_text or task.description or task.title,
                parsed=task,
            )
            result = DraftResult(True, draft=creation.draft)
            self._log(action, source, telegram_user_id, True)
            return ActionOutcome("ok", result=result)
        if target is None or draft_id is None or version is None:
            return ActionOutcome("missing")

        previous_message_id = target.preview_message_id
        if action == "save":
            result = await self.drafts.confirm(draft_id, version, telegram_user_id, chat_id)
        elif action in {"discard", "cancel"}:
            result = await self.drafts.drop(draft_id, version, telegram_user_id, chat_id)
        elif action == "edit":
            result = await self.drafts.begin_edit(draft_id, version, telegram_user_id, chat_id)
        elif action == "confirm_date" and task is not None:
            result = await self.drafts.transform(
                draft_id,
                version,
                telegram_user_id,
                chat_id,
                task,
                raw_text=raw_text,
            )
        elif action == "confirm_date" and resolved_date is not None:
            result = await self.drafts.apply_resolved_date(
                draft_id, version, telegram_user_id, chat_id, resolved_date
            )
        elif action == "create_task" and task is not None:
            result = await self.drafts.transform(
                draft_id,
                version,
                telegram_user_id,
                chat_id,
                task,
                raw_text=raw_text,
            )
        else:
            return ActionOutcome("stale")
        self._log(action, source, telegram_user_id, result.ok)
        return ActionOutcome(
            "ok" if result.ok else "stale",
            result=result,
            previous_message_id=previous_message_id,
        )

    @staticmethod
    def _log(action: DraftAction, source: ActionSource, telegram_user_id: int, ok: bool) -> None:
        logger.info(
            "draft_action=%s action_source=%s user=%s success=%s",
            action,
            source,
            masked_user(telegram_user_id),
            ok,
        )
