from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import MessageLimit
from telegram.error import TelegramError

from .access import FULL_ACCESS_TIERS
from .ai import NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT
from .conversation import ConversationExchangeReceipt
from .domain import temporal_context
from .models import DraftInboxItem, InboxItem, TaskReminder, User
from .nova_brain import (
    NovaBrainApplyReceipt,
    NovaBrainFence,
    NovaBrainForgetCapability,
    NovaBrainForgetStore,
    NovaBrainPolicy,
    NovaBrainProjection,
    NovaBrainService,
    observed_memory_display_value,
    validate_dialogue_state_update,
    validate_memory_candidate,
)
from .nova_companion import (
    NovaCompanionContextFence,
    NovaCompanionContextService,
)
from .nova_companion_flow import (
    CAPTURE_ACTIONS,
    CAPTURE_DATE_ACTIONS,
    CaptureSuggestion,
    ExplicitCaptureClassifier,
    NovaAddressClassifier,
    NovaAddressKind,
    NovaCompanionCaptureCapability,
    NovaCompanionCaptureScreen,
    NovaCompanionCaptureStore,
    NovaCompanionCaptureTemporal,
    NovaCompanionDiscourseAnchor,
    NovaCompanionDiscourseReducer,
    NovaCompanionPolicy,
    NovaCompanionReminderCandidate,
    NovaCompanionReminderCapability,
    NovaCompanionReminderScreen,
    NovaCompanionReminderStore,
    validate_capture_suggestion,
)
from .nova_memory_application import (
    NovaMemoryProjection,
    NovaMemoryProjectionError,
    build_nova_memory_projection,
)
from .reminder_flow import ReminderFlowSession
from .reminder_intent import (
    ConversationRecallIntent,
    ReminderIntentStatus,
    classify_conversation_recall,
)
from .schemas import (
    NovaCompanionDiagnosticCode,
    NovaCompanionDialogueStateUpdate,
    NovaCompanionMemoryCandidate,
    ParsedThought,
)

logger = logging.getLogger(__name__)

NOVA_COMPANION_ACCESS_CHANGED_TEXT = (
    "Доступ изменился. Я не использовала прежний контекст — напиши сообщение ещё раз."
)
NOVA_COMPANION_CONTEXT_CHANGED_TEXT = (
    "Контекст успел измениться. Ничего не сохранено — напиши сообщение ещё раз."
)
NOVA_COMPANION_UNAVAILABLE_TEXT = "Сейчас не получилось ответить — попробуй ещё раз."
NOVA_COMPANION_CAPTURE_FAILED_TEXT = (
    "Не удалось открыть preview. Ничего не сохранено — попробуй ещё раз."
)
NOVA_COMPANION_NOT_EXECUTED_TEXT = "Пока ничего не создано — я не выполняла это действие."
NOVA_COMPANION_REMINDER_OFFER_ACTION_TEXT = (
    "Пока ничего не создано. Могу поставить настоящее напоминание — нажми кнопку."
)
NOVA_COMPANION_NO_ACTIVE_REMINDER_OFFER_TEXT = (
    "Сейчас нет активного предложения напоминания. Скажи, что и когда напомнить."
)
NOVA_COMPANION_RECALL_CLARIFICATION_TEXT = (
    "Ты хочешь вспомнить содержание разговора или поставить настоящее напоминание?"
)
NOVA_COMPANION_RECALL_UNAVAILABLE_TEXT = "Деталей того разговора в доступном контексте сейчас нет."
NOVA_COMPANION_RECALL_ACTION_SUPPRESSED_TEXT = (
    "В режиме воспоминания я могу только пересказать доступный контекст разговора — "
    "без кнопок и действий."
)
NOVA_COMPANION_DISCOURSE_AMBIGUOUS_TEXT = (
    "Ты хочешь подобрать разговорный способ или настроить настоящее напоминание?"
)
_DISCOURSE_KIND_LABELS = {
    "method": "подобрать разговорный способ",
    "exercise": "выбрать упражнение",
    "reminder_setup": "настроить настоящее напоминание",
    "plan": "составить простой план",
}

_COMPANION_DRAIN_TIMEOUT_SECONDS = 30.0
_COMPANION_CANCEL_TIMEOUT_SECONDS = 5.0
_COMPANION_CANCEL_RETRY_SECONDS = 0.1
_COMPANION_CLEANUP_SCHEDULED_ATTR = "nova_companion_cleanup_scheduled"
_COMPANION_REMINDER_SESSION_ATTR = "nova_companion_reminder_session"
_COMPANION_RECALL_TOPIC_MAX_CHARS = 160
_COMPANION_RECALL_TOPIC_MAX_BYTES = 512

_IDENTITY_NAME_QUESTION = re.compile(
    r"(?:^|[.!?…»]\s*)(?:а\s+)?(?:(?:ты\s+)?(?:не\s+)?знал[аи]?[,\s]+|"
    r"ты\s+знаешь[,\s]+)?как\s+меня\s+зовут[?!.…]*$",
    re.IGNORECASE,
)
_IDENTITY_CITY_QUESTION = re.compile(
    r"^(?:а\s+)?(?:ты\s+)?(?:знаешь[,\s]+)?(?:где\s+я\s+живу|"
    r"в\s+каком\s+городе\s+я\s+живу)[?!.…]*$",
    re.IGNORECASE,
)
_REMINDER_STATUS_QUESTION = re.compile(
    r"^(?:в\s+смысле[?!.…]*\s*)?(?:ты\s+)?(?:"
    r"записала\s+напоминание|поставила\s+напоминание|создала\s+напоминание|"
    r"напомнишь|а\s+мне\s+напомнишь|оно\s+уже\s+создано|"
    r"напоминание\s+(?:уже\s+)?(?:создано|готово)|готово\s+с\s+напоминанием"
    r")[?!.…]*$",
    re.IGNORECASE,
)
_CAPTURE_STATUS_QUESTION = re.compile(
    r"^(?:ты\s+)?(?:сохранила|записала|добавила|создала)\s+"
    r"(?:задачу|заметку|идею)[?!.…]*$",
    re.IGNORECASE,
)
_REMINDER_ACCEPT = re.compile(
    r"^(?:да|давай|поставь|поставь\s+плиз|напомни(?:\s+плиз)?|сделай\s+напоминание)[!.,…]*$",
    re.IGNORECASE,
)
_REMINDER_DECLINE = re.compile(
    r"^(?:нет|не\s+надо|не\s+сейчас|не\s+ставь)[!.,…]*$",
    re.IGNORECASE,
)
_REMINDER_WEAK_CONTINUATION = re.compile(
    r"^(?:да|давай|нет|не\s+надо|не\s+сейчас)[!.,…]*$",
    re.IGNORECASE,
)
_CONTEXTUAL_STATUS_QUESTION = re.compile(
    r"^(?:ну\s+что[,:]?\s*)?(?:вс[её]\s+)?готово\?+[!.…]*$",
    re.IGNORECASE,
)
_BRAIN_MEMORY_RECALL = re.compile(
    r"^(?:что\s+ты\s+(?:обо\s+мне\s+)?(?:помнишь|знаешь)|"
    r"что\s+ты\s+(?:помнишь|знаешь)\s+обо\s+мне|"
    r"что\s+ты\s+знаешь\s+о\s+моих\s+целях)[?!.…]*$",
    re.IGNORECASE,
)
_BRAIN_MEMORY_FORGET = re.compile(
    r"^(?:это\s+уже\s+неактуально|забудь(?:\s+это|\s*,?\s*что\s+я\s+говорил[аи]?\s+о\s+.+)?)"
    r"[?!.…]*$",
    re.IGNORECASE,
)
_USER_SUBJECT_OPERATIONAL_CLAIM = re.compile(
    r"\bты(?:\s+(?:уже|сама?|самостоятельно))*\s+(?:"
    r"поставил[а]?|создал[а]?|записал[а]?|сохранил[а]?|добавил[а]?|"
    r"отметил[а]?|уч(?:е|ё)?л[а]?|зафиксировал[а]?"
    r")\b",
    re.IGNORECASE,
)
_UNTRUSTED_OPERATIONAL_CLAIM = re.compile(
    r"\b(?:(?:я\s+)?(?:уже\s+)?(?:поставил[а]?|создал[а]?|записал[а]?|сохранил[а]?|добавил[а]?)|"
    r"(?:я\s+)?(?:поставлю|создам|запишу|сохраню|добавлю)|"
    r"(?:напоминание|задача|заметка|идея|черновик|стрижка)\s+(?:уже\s+)?"
    r"(?:создан[ао]?|сохранен[ао]?|сохранён[ао]?|записан[ао]?|поставлен[ао]?|"
    r"установлен[ао]?|настроен[ао]?|отмечен[ао]?)|"
    r"(?:я\s+)?(?:вс[её]\s+)?(?:отметил[а]?|уч(?:е|ё)?л[а]?|зафиксировал[а]?)|"
    r"напомню|(?:я\s+не\s+забуд(?:у|ем)|(?:я\s+)?(?:точно|обязательно)\s+не\s+забуд(?:у|ем))"
    r"(?:\s+(?:про|об?|тебе))?|"
    r"буду(?:\s+\w+){0,2}\s+напоминать|уже\s+в\s+голове\s+отмечено|"
    r"считай[,:;.!?—\s]+(?:что\s+)?напоминан\w*(?:\s+уже)?\s+готов\w*|"
    r"напоминан\w*\s+(?:уже\s+)?(?:готов\w*|поставлен\w*)|"
    r"(?:у\s+меня|держу[\s\S]{0,80})\s+(?:на|под)\s+контрол\w*|"
    r"возьм\w*[\s\S]{0,100}\b(?:на|под)\s+контрол\w*|"
    r"буду\s+держать\b[\s\S]{0,100}\b(?:в\s+уме|в\s+голове|в\s+поле\s+внимания)|"
    r"буду\s+иметь\b[\s\S]{0,100}\bв\s+виду|"
    r"(?:я\s+)?буду\s+помнить\b|(?:я\s+)?прослежу\b|"
    r"(?:я\s+)?не\s+дам\b[\s\S]{0,80}\bзабыть|"
    r"считай[,:;.!?—\s]+(?:что\s+)?(?:это|вс[её])\s+(?:на|под)\s+контрол\w*|"
    r"буду\s+возвращать\b[\s\S]{0,80}\bк\s+главн\w*)\b",
    re.IGNORECASE,
)
_UNTRUSTED_MEMORY_OR_RETURN_CLAIM = re.compile(
    r"\b(?:"
    r"(?:я\s+)?запомнил[аи]?(?:\s|[,:;.!?—-])|"
    r"(?:я\s+)?держу\b[\s\S]{0,100}\b(?:в\s+контекст\w*|как\s+напоминан\w*|в\s+голове)|"
    r"(?:я\s+)?(?:обязательно\s+)?вернусь\s+к\s+(?:этому|этому\s+вопросу|нему|ней)|"
    r"напоминан\w*\b[\s\S]{0,80}\b(?:у\s+меня\b[\s\S]{0,30}\bв\s+голове|в\s+голове)|"
    r"готово\b[\s\S]{0,100}\b(?:отмечен[аоы]?|записан[аоы]?|создан[аоы]?)"
    r")",
    re.IGNORECASE,
)
_ELLIPTICAL_OPERATIONAL_CLAIM = re.compile(
    r"^(?:(?:да|конечно|ну\s+вс[её]|считай)[,:;.!?—\s]+)?(?:что\s+)?"
    r"(?:вс[её]\s+)?готово(?:\s*[—-]\s*напоминан\w*(?:\s+на\s+завтра)?)?[.!…]*$",
    re.IGNORECASE,
)
_CONTEXTUAL_OPERATIONAL_COMMITMENT = re.compile(
    r"^(?:да[,:;.!?—\s]+|конечно[,:;.!?—\s]+)?(?:можешь\s+)?на\s+меня\s+рассчитывать[.!…]*$",
    re.IGNORECASE,
)
_ACTION_CONTEXT = re.compile(
    r"\b(?:напоминан\w*|напомн\w*|уведом\w*|задач\w*|заметк\w*|иде[яию]\w*|"
    r"черновик\w*|сохран\w*|запиш\w*|добав\w*|созда\w*|постав\w*|"
    r"забуд\w*|сегодня|завтра|послезавтра|\d{1,2}[:.]\d{2})\b",
    re.IGNORECASE,
)
_SUPPRESSED_ACTION_DEPENDENCY = re.compile(
    r"(?:\bкнопк\w*\b|"
    r"\b(?:нажм|выбер|подтверд|использу)\w*\b|"
    r"\bмогу\b[\s\S]{0,32}\b(?:постав|созда|сохран|добав|запи|откр|настро)\w*\b|"
    r"\b(?:постав|созда|сохран|добав|запи|откр|настро)\w*\b[\s\S]{0,32}"
    r"\b(?:ниже|рядом)\b)",
    re.IGNORECASE,
)
_SUPPRESSED_VISIBLE_ACTION = re.compile(
    r"\b(?:напоминан|напомн|постав|сохран|запи|добав|созда|откр|настро|"
    r"задач|заметк|иде[яию]|черновик)\w*\b",
    re.IGNORECASE,
)
_SUPPRESSED_UI_OR_AVAILABILITY_CUE = re.compile(
    r"(?:\b(?:ниже|рядом|можно|доступ\w*|вариант\w*|действи\w*|кнопк\w*|"
    r"нажм\w*|выбер\w*|подтверд\w*|использу\w*)\b|"
    r"\bпод\s+сообщени\w*\b)",
    re.IGNORECASE,
)
_RECENT_RECALL_TOPIC = re.compile(
    r"\bнедавно\s+разговаривали\s+о\s+(?P<topic>.+?)[.!?…]\s*"
    r"(?:ты\s+)?помнишь\s+(?:наш(?:\s+с\s+тобой)?\s+)?разговор",
    re.IGNORECASE,
)
_CONVERSATION_RECALL_TOPIC = re.compile(
    r"\bразговор(?:а)?\s+о\s+(?P<topic>.+?)(?:[,;:]\s*(?:что|о\s+ч[её]м)|[?!.…]|$)",
    re.IGNORECASE,
)
_DISCOURSE_NONCONTINUATION = re.compile(
    r"(?:\bчто\s+именно\b[\s\S]{0,80}\b(?:подобрать|выбрать|сделать)\b|"
    r"\bуточни\b[\s\S]{0,80}\b(?:что|какой|какую)\b|"
    r"^(?:да|конечно)[,!:.\s]+(?:это\s+)?(?:звучит\s+)?(?:неплохо|круто|хорошо)\b)",
    re.IGNORECASE,
)

_STATUS_RECEIPT_UNSET = object()

type _CompanionCheck = Literal["ready", "access_changed", "context_changed", "unavailable"]


def _has_untrusted_operational_claim(
    answer: object,
    *,
    user_text: object = "",
    action_context: bool = False,
) -> bool:
    if not isinstance(answer, str):
        return False
    normalized = " ".join(unicodedata.normalize("NFKC", answer).split())
    subject_fenced = _USER_SUBJECT_OPERATIONAL_CLAIM.sub("", normalized)
    normalized_user_text = (
        " ".join(unicodedata.normalize("NFKC", user_text).split())
        if isinstance(user_text, str)
        else ""
    )
    has_action_context = action_context or bool(_ACTION_CONTEXT.search(normalized_user_text))
    return bool(
        _UNTRUSTED_OPERATIONAL_CLAIM.search(subject_fenced)
        or _UNTRUSTED_MEMORY_OR_RETURN_CLAIM.search(subject_fenced)
        or (has_action_context and _ELLIPTICAL_OPERATIONAL_CLAIM.fullmatch(normalized))
        or (has_action_context and _CONTEXTUAL_OPERATIONAL_COMMITMENT.fullmatch(normalized))
    )


def _has_suppressed_action_dependency(answer: object, *, user_text: object = "") -> bool:
    if not isinstance(answer, str):
        return False
    normalized = " ".join(unicodedata.normalize("NFKC", answer).split())
    return bool(
        _SUPPRESSED_ACTION_DEPENDENCY.search(normalized)
        or any(
            _SUPPRESSED_VISIBLE_ACTION.search(clause)
            and _SUPPRESSED_UI_OR_AVAILABILITY_CUE.search(clause)
            for clause in re.split(r"[.!?…;]+", normalized)
        )
        or _has_untrusted_operational_claim(
            normalized,
            user_text=user_text,
            action_context=True,
        )
    )


def _bounded_recall_topic(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        return None
    normalized = " ".join(normalized.split()).strip(" ,.!?…")
    if (
        not normalized
        or len(normalized) > _COMPANION_RECALL_TOPIC_MAX_CHARS
        or len(normalized.encode("utf-8")) > _COMPANION_RECALL_TOPIC_MAX_BYTES
    ):
        return None
    return normalized


def _companion_discourse_answer(
    answer: str,
    anchor: NovaCompanionDiscourseAnchor | None,
) -> tuple[str, bool]:
    """Enforce the immediate-offer contract without granting execution rights."""

    if anchor is None:
        return answer, False
    if anchor.status == "ambiguous":
        if anchor.offer_kinds == ("method", "reminder_setup"):
            return NOVA_COMPANION_DISCOURSE_AMBIGUOUS_TEXT, True
        labels = [_DISCOURSE_KIND_LABELS[kind] for kind in anchor.offer_kinds]
        if len(labels) == 2:
            clarification = f"Ты хочешь {labels[0]} или {labels[1]}?"
        else:
            clarification = "Что продолжить: " + ", ".join(labels[:-1]) + f" или {labels[-1]}?"
        return clarification, True
    if _DISCOURSE_NONCONTINUATION.search(answer) is None:
        return answer, False
    kind = anchor.offer_kinds[0]
    if kind in {"method", "exercise"}:
        return (
            "Тогда предлагаю короткую практику: остановись на пару секунд, сделай один "
            "спокойный вдох и спроси себя: «Что для меня сейчас действительно важно?». "
            "Можем подобрать удобный вариант для этого.",
            True,
        )
    if kind == "plan":
        return (
            "Тогда начнём с простого плана: выбери один ориентир, один маленький шаг на "
            "сегодня и короткую проверку вечером.",
            True,
        )
    return (
        "Давай настроим настоящее напоминание. Что именно и когда тебе напомнить?",
        True,
    )


class _NovaCompanionDrainError(RuntimeError):
    pass


class _NovaCompanionCaptureCompensationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _CompanionGeneration:
    owner_id: int = field(repr=False)
    telegram_actor_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    tier: str = field(repr=False)
    access_version: int = field(repr=False)
    context_fence: NovaCompanionContextFence = field(repr=False)
    memory_revision: str | None = field(default=None, repr=False)
    brain_fence: NovaBrainFence | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _PreparedCompanionAnswer:
    answer: str = field(repr=False)
    generation: _CompanionGeneration = field(repr=False)
    suggestion: CaptureSuggestion | None = field(default=None, repr=False)
    user_text: str = field(default="", repr=False)
    source: str = "text"
    temporal: NovaCompanionCaptureTemporal | None = field(default=None, repr=False)
    reminder_candidate: NovaCompanionReminderCandidate | None = field(default=None, repr=False)
    dialogue_state_update: NovaCompanionDialogueStateUpdate | None = field(
        default=None,
        repr=False,
    )
    memory_candidate: NovaCompanionMemoryCandidate | None = field(default=None, repr=False)
    persist_exchange: bool = True


@dataclass(frozen=True, slots=True)
class _CompanionPendingCaptureStatus:
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_version: int = field(repr=False)
    draft_id: str = field(repr=False)
    draft_version: int = field(repr=False)
    canonical_message_id: int = field(repr=False)
    kind: Literal["task", "note", "idea"]
    title: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _CompanionPendingReminderStatus:
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_version: int = field(repr=False)
    session_id: str = field(repr=False)
    title: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _CompanionStatusReceipt:
    kind: Literal["task", "note", "idea", "reminder"]
    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_version: int = field(repr=False)
    inbox_item_id: int = field(repr=False)
    inbox_item_version: int = field(repr=False)
    title: str = field(repr=False)


class NovaCompanionHandlers:
    """Conversation-first Nova routing and optional capture delivery."""

    def _init_nova_companion(self) -> None:
        self.nova_companion_context = NovaCompanionContextService(
            self.db,
            conversation_message_limit=self.settings.conversation_context_messages,
        )
        self.nova_companion_captures = NovaCompanionCaptureStore()
        self.nova_companion_reminders = NovaCompanionReminderStore()
        self.nova_brain_service = NovaBrainService(
            self.db,
            max_memories=int(getattr(self.settings, "nova_conversation_brain_max_memories", 100)),
            retrieval_max_items=int(
                getattr(self.settings, "nova_conversation_brain_retrieval_items", 6)
            ),
            context_max_bytes=int(
                getattr(self.settings, "nova_conversation_brain_context_bytes", 8192)
            ),
        )
        self.nova_brain_forget = NovaBrainForgetStore()
        self._nova_brain_ui_lock = asyncio.Lock()
        self._nova_companion_reminder_ui_lock = asyncio.Lock()
        self._nova_companion_tasks: set[asyncio.Task[bool]] = set()
        self._nova_companion_pending_capture_status: dict[str, _CompanionPendingCaptureStatus] = {}
        self._nova_companion_pending_reminder_status: dict[
            str, _CompanionPendingReminderStatus
        ] = {}
        self._nova_companion_status_receipts: dict[
            tuple[int, int, int], _CompanionStatusReceipt
        ] = {}

    def nova_companion_policy(self) -> NovaCompanionPolicy:
        return NovaCompanionPolicy(
            enabled=bool(getattr(self.settings, "enable_nova_companion", False)),
            admin_only=bool(getattr(self.settings, "nova_companion_admin_only", True)),
        )

    def nova_companion_available_for_actor(self, actor: Any | None) -> bool:
        return self.nova_companion_policy().allows_actor(actor)

    def nova_brain_policy(self) -> NovaBrainPolicy:
        return NovaBrainPolicy(
            enabled=bool(getattr(self.settings, "enable_nova_conversation_brain", False)),
            admin_only=bool(getattr(self.settings, "nova_conversation_brain_admin_only", True)),
        )

    async def nova_companion_route(
        self,
        update: Any,
        context: Any,
        text: str,
        source: str,
        *,
        user: User,
        conversation_snapshot: Any,
        delivery_message: Any | None = None,
        resolved_date: Any | None = None,
        temporal_resolution: Any | None = None,
        status_receipt_preinvalidated: bool = False,
    ) -> bool:
        """Handle an eligible ordinary message without invoking the legacy router."""

        policy = self.nova_companion_policy()
        if not policy.allows_actor(user):
            return False
        status_receipt_anchor = (
            None
            if status_receipt_preinvalidated
            else self.nova_companion_status_receipt_anchor(
                user.telegram_id,
                update.effective_chat.id,
            )
        )
        address = NovaAddressClassifier.classify(text)
        if address.is_local_response:
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                address.local_response or "Да, я здесь 🙂",
                user=user,
            )
            return True
        semantic_text = (
            address.content
            if address.kind is NovaAddressKind.VOCATIVE and address.content is not None
            else text.strip()
        )
        identity_response = await self._nova_companion_identity_answer(semantic_text, user)
        if identity_response is not None:
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                identity_response,
                user=user,
            )
            return True
        status_response = await self._nova_companion_status_answer(
            semantic_text,
            user=user,
            chat_id=update.effective_chat.id,
        )
        if status_response is not None:
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                status_response,
                user=user,
            )
            return True
        if await self._nova_brain_memory_control(
            update,
            context,
            delivery_message,
            user=user,
            text=semantic_text,
            conversation_snapshot=conversation_snapshot,
        ):
            return True
        if not status_receipt_preinvalidated:
            self.nova_companion_invalidate_status_for_input(
                user.telegram_id,
                update.effective_chat.id,
                semantic_text,
                expected_receipt=status_receipt_anchor,
            )
        explicit = ExplicitCaptureClassifier.classify(semantic_text)
        if explicit is not None:
            if explicit.kind == "task" and resolved_date is None and temporal_resolution is None:
                date_resolution = self.date_resolver.resolve(semantic_text, user.timezone)
                if date_resolution.status == "conflict":
                    return False
                if date_resolution.status == "resolved" and date_resolution.target_date:
                    resolved_date = date_resolution.target_date
                    temporal_resolution = self.date_resolver.temporal_resolution(
                        date_resolution.target_date,
                        user.timezone,
                        semantic_text,
                        self.date_resolver.extract_local_time(semantic_text),
                    )
            await self._nova_companion_explicit_capture(
                update,
                context,
                delivery_message,
                user=user,
                snapshot=conversation_snapshot,
                source=source,
                original_text=text,
                kind=explicit.kind,
                content=explicit.content,
                references_context=explicit.references_context,
                resolved_date=resolved_date,
                temporal_resolution=temporal_resolution,
            )
            return True
        continuation = self._nova_companion_reminder_continuation_action(semantic_text)
        recall_intent = classify_conversation_recall(semantic_text)
        if continuation is not None:
            if await self._nova_companion_handle_reminder_continuation(
                update,
                context,
                delivery_message,
                user=user,
                action=continuation,
            ):
                return True
        recall_anchor = self._nova_companion_has_recall_anchor(
            conversation_snapshot,
            semantic_text,
        )
        if recall_intent is ConversationRecallIntent.AMBIGUOUS:
            if not recall_anchor:
                await self._nova_companion_local_response(
                    update,
                    context,
                    delivery_message,
                    NOVA_COMPANION_RECALL_CLARIFICATION_TEXT,
                    user=user,
                )
                return True
        elif recall_intent is ConversationRecallIntent.RECALL and not recall_anchor:
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                self._nova_companion_recall_unavailable_answer(semantic_text),
                user=user,
            )
            return True
        if recall_intent is not ConversationRecallIntent.NONE:
            await self._nova_companion_prepare_and_deliver(
                update,
                context,
                delivery_message,
                user=user,
                snapshot=conversation_snapshot,
                text=semantic_text,
                source=source,
                suppress_proposals=True,
            )
            return True
        if continuation is not None:
            if not self._nova_companion_reminder_continuation_requires_anchor(semantic_text):
                await self._nova_companion_local_response(
                    update,
                    context,
                    delivery_message,
                    NOVA_COMPANION_NO_ACTIVE_REMINDER_OFFER_TEXT,
                    user=user,
                )
                return True
        else:
            reminder_probe = self.reminder_intent_parser.parse(semantic_text, user.timezone)
            if reminder_probe.status is not ReminderIntentStatus.NOT_REMINDER:
                return await self._reminder_question_gate(
                    update,
                    context,
                    semantic_text,
                    candidate_message=delivery_message,
                    expected_access_version=(
                        user.access_version if delivery_message is not None else None
                    ),
                    expected_session=None,
                    voice_fenced=delivery_message is not None,
                    voice_state=None,
                    weekly_candidate_handoff=False,
                )
        await self._nova_companion_prepare_and_deliver(
            update,
            context,
            delivery_message,
            user=user,
            snapshot=conversation_snapshot,
            text=semantic_text,
            source=source,
        )
        return True

    @staticmethod
    def _nova_companion_has_recall_anchor(snapshot: Any, text: str) -> bool:
        if classify_conversation_recall(text) is ConversationRecallIntent.AMBIGUOUS:
            return NovaCompanionHandlers._nova_companion_has_immediate_recall_anchor(snapshot)
        helper = getattr(snapshot, "for_companion_prompt", None)
        if not callable(helper):
            return False
        context = helper()
        messages = context.get("recent_messages") if isinstance(context, dict) else None
        if not isinstance(messages, list):
            return False
        roles = {
            message.get("role")
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        }
        if not {"user", "assistant"}.issubset(roles):
            return False
        topic = NovaCompanionHandlers._nova_companion_recall_topic(text)
        if topic is None:
            return not NovaCompanionHandlers._nova_companion_has_explicit_recall_topic(text)
        normalized_topic = " ".join(unicodedata.normalize("NFKC", topic).casefold().split())
        return any(
            normalized_topic
            in " ".join(unicodedata.normalize("NFKC", str(message["content"])).casefold().split())
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )

    @staticmethod
    def _nova_companion_has_immediate_recall_anchor(snapshot: Any) -> bool:
        messages = getattr(snapshot, "messages", None)
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        user_message, assistant_message = messages[-2:]
        if not isinstance(user_message, dict) or not isinstance(assistant_message, dict):
            return False
        if (
            user_message.get("role") != "user"
            or user_message.get("intent") != "companion_user"
            or assistant_message.get("role") != "assistant"
            or assistant_message.get("intent") != "companion_answer"
        ):
            return False
        user_content = user_message.get("content")
        assistant_content = assistant_message.get("content")
        if (
            not isinstance(user_content, str)
            or not isinstance(assistant_content, str)
            or classify_conversation_recall(user_content) is not ConversationRecallIntent.RECALL
        ):
            return False
        helper = getattr(snapshot, "for_companion_prompt", None)
        if not callable(helper):
            return False
        context = helper()
        projected = context.get("recent_messages") if isinstance(context, dict) else None
        if not isinstance(projected, list) or len(projected) < 2:
            return False
        return projected[-2:] == [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

    @staticmethod
    def _nova_companion_recall_topic(text: str) -> str | None:
        recent_topic = _RECENT_RECALL_TOPIC.search(text)
        if recent_topic is not None:
            return _bounded_recall_topic(recent_topic.group("topic"))
        topic_match = _CONVERSATION_RECALL_TOPIC.search(text)
        if topic_match is None:
            return None
        return _bounded_recall_topic(topic_match.group("topic"))

    @staticmethod
    def _nova_companion_has_explicit_recall_topic(text: str) -> bool:
        return bool(_RECENT_RECALL_TOPIC.search(text) or _CONVERSATION_RECALL_TOPIC.search(text))

    @staticmethod
    def _nova_companion_recall_unavailable_answer(text: str) -> str:
        topic = NovaCompanionHandlers._nova_companion_recall_topic(text)
        if topic is None:
            return NOVA_COMPANION_RECALL_UNAVAILABLE_TEXT
        answer = (
            f"Я вижу, что речь была о {topic}, но деталей того разговора "
            "в доступном контексте сейчас нет."
        )
        if len(answer.encode("utf-16-le")) // 2 > int(MessageLimit.MAX_TEXT_LENGTH):
            return NOVA_COMPANION_RECALL_UNAVAILABLE_TEXT
        return answer

    async def _nova_companion_identity_answer(self, text: str, user: User) -> str | None:
        if _IDENTITY_NAME_QUESTION.search(text.strip()):
            name = " ".join((user.display_name or "").split())
            if name:
                return f"Да, тебя зовут {name}."
            policy = self.nova_brain_policy()
            if policy.allows_actor(user):
                try:
                    observed = await self.nova_brain_service.list_current(
                        telegram_actor_id=user.telegram_id,
                        expected_access_version=user.access_version,
                        policy=policy,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Nova companion failed operation=identity_lookup error_type=%s",
                        type(exc).__name__,
                    )
                    observed = ()
                for memory in observed:
                    if memory.category != "identity":
                        continue
                    matched = re.search(
                        r"(?:^identity:|;)display_name=([а-яёa-z-]{2,50})(?:;|$)",
                        memory.value,
                        re.IGNORECASE,
                    )
                    if matched is not None:
                        declared = matched.group(1)
                        declared = declared[:1].upper() + declared[1:]
                        return f"Из твоих слов: тебя зовут {declared}."
            return "Пока не знаю, как тебя зовут."
        if _IDENTITY_CITY_QUESTION.fullmatch(text.strip()):
            city = " ".join((user.location_city or "").split())
            return (
                f"Да, ты живёшь в городе {city}."
                if city
                else "Подтверждённый город пока не указан."
            )
        return None

    @staticmethod
    def _nova_brain_recall_answer(
        payload: dict[str, object],
        observed: tuple[Any, ...],
    ) -> str:
        sections: list[tuple[str, list[str]]] = []
        identity = payload.get("confirmed_identity")
        if isinstance(identity, dict):
            values: list[str] = []
            labels = (
                ("display_name", "имя"),
                ("location_city", "город"),
                ("timezone", "часовой пояс"),
            )
            for key, label in labels:
                value = identity.get(key)
                if isinstance(value, str) and value:
                    values.append(f"{label}: {value}")
            if values:
                sections.append(("Подтверждено в профиле", values))

        profile = payload.get("profile")
        if isinstance(profile, dict):
            values = []
            summary = profile.get("summary")
            if isinstance(summary, str) and summary:
                values.append(summary)
            for key in ("values", "desired_identity"):
                items = profile.get(key)
                if isinstance(items, list):
                    values.extend(item for item in items[:2] if isinstance(item, str))
            if values:
                sections.append(("Анкета и профиль", values[:3]))

        plans: list[str] = []
        weekly = payload.get("current_weekly_focus")
        if isinstance(weekly, dict) and isinstance(weekly.get("focus"), str):
            plans.append(f"фокус недели: {weekly['focus']}")
        goals = payload.get("active_goals")
        if isinstance(goals, list):
            plans.extend(
                f"цель: {item['title']}"
                for item in goals[:2]
                if isinstance(item, dict) and isinstance(item.get("title"), str)
            )
        visions = payload.get("active_vision_items")
        if isinstance(visions, list):
            plans.extend(
                f"желание: {item['wish_text']}"
                for item in visions[:2]
                if isinstance(item, dict) and isinstance(item.get("wish_text"), str)
            )
        if plans:
            sections.append(("Текущие планы и ориентиры", plans[:5]))

        confirmed = payload.get("confirmed_memory")
        if isinstance(confirmed, list):
            values = [
                str(item["content"])
                for item in confirmed[:4]
                if isinstance(item, dict) and isinstance(item.get("content"), str)
            ]
            if values:
                sections.append(("Подтверждено тобой в Nova Memory", values))
        if observed:
            sections.append(
                (
                    "Сохранено из твоих слов",
                    [observed_memory_display_value(str(memory.value)) for memory in observed[:4]],
                )
            )
        if not sections:
            return "Пока у меня нет актуальных сведений, которые можно честно перечислить."
        lines = [
            "Кратко по актуальным источникам — это не полный экспорт всех данных:",
        ]
        for title, values in sections:
            lines.append(f"\n{title}:")
            lines.extend(f"• {value}" for value in values)
        return "\n".join(lines)[:1_900]

    async def _nova_brain_memory_control(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        text: str,
        conversation_snapshot: Any,
    ) -> bool:
        policy = self.nova_brain_policy()
        if not policy.allows_actor(user):
            return False
        cleaned = text.strip()
        if _BRAIN_MEMORY_RECALL.fullmatch(cleaned):
            observed = await self.nova_brain_service.list_current(
                telegram_actor_id=user.telegram_id,
                expected_access_version=user.access_version,
                policy=policy,
            )
            memory_status, confirmed, _revision = await self._nova_companion_memory_projection(user)
            materialized = await self.nova_companion_context.snapshot(
                telegram_actor_id=user.telegram_id,
                expected_tier=user.access_tier,
                expected_access_version=user.access_version,
                conversation_context=conversation_snapshot.for_companion_prompt(),
                conversation_chat_id=update.effective_chat.id,
                confirmed_memory=(
                    confirmed if memory_status == "ready" and confirmed is not None else None
                ),
            )
            if materialized.status != "ready" or materialized.projection is None:
                await self._nova_companion_local_response(
                    update,
                    context,
                    delivery_message,
                    self._nova_companion_neutral_text(materialized.status),
                    user=user,
                )
                return True
            response = self._nova_brain_recall_answer(
                materialized.projection.provider_context_payload(),
                observed,
            )
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                response,
                user=user,
            )
            logger.info(
                "Nova companion trace route=memory_recall provider_called=false "
                "working_state_revision=0 retrieved_memory_count=%s",
                len(observed),
            )
            return True
        if _BRAIN_MEMORY_FORGET.fullmatch(cleaned) is None:
            return False
        observed = await self.nova_brain_service.list_current(
            telegram_actor_id=user.telegram_id,
            expected_access_version=user.access_version,
            policy=policy,
        )
        target = self._nova_brain_forget_target(cleaned, observed)
        if target is None:
            response = (
                "Уточни одним коротким фрагментом, что именно забыть."
                if observed
                else "Сейчас нет подходящей сохранённой записи, которую можно забыть."
            )
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                response,
                user=user,
            )
            return True
        stage = None
        sent = delivery_message
        message_id = self._positive_companion_message_id(
            getattr(delivery_message, "message_id", None)
        )
        try:
            stage = await self.nova_brain_forget.stage(
                target,
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                access_version=user.access_version,
            )
            markup = self._nova_brain_forget_markup(stage)
            async with self._nova_brain_ui_lock:
                if not await self._nova_companion_actor_is_current(user):
                    await self.nova_brain_forget.revoke(stage)
                    return True
                if delivery_message is None:
                    sent = await update.effective_message.reply_text(
                        "Забыть выбранную запись?",
                        reply_markup=markup,
                    )
                    message_id = self._positive_companion_message_id(
                        getattr(sent, "message_id", None)
                    )
                else:
                    await delivery_message.edit_text(
                        "Забыть выбранную запись?",
                        reply_markup=markup,
                    )
                if sent is None or message_id is None:
                    raise ValueError("Missing Nova brain confirmation message")
                bound = await self.nova_brain_forget.bind(
                    stage,
                    canonical_message_id=message_id,
                )
                if bound is None or not await self._nova_companion_actor_is_current(user):
                    await self.nova_brain_forget.revoke(bound or stage)
                    await self._nova_companion_compensate(
                        context,
                        sent,
                        chat_id=update.effective_chat.id,
                        message_id=message_id,
                        neutral_text=NOVA_COMPANION_CONTEXT_CHANGED_TEXT,
                    )
            return True
        except asyncio.CancelledError:
            if stage is not None:
                await self.nova_brain_forget.revoke(stage)
            if sent is not None and message_id is not None:
                self._nova_companion_schedule_pre_delivery_cleanup(
                    context,
                    sent,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=memory_forget_prompt error_type=%s",
                type(exc).__name__,
            )
            if stage is not None:
                await self.nova_brain_forget.revoke(stage)
            if sent is not None and message_id is not None:
                await self._nova_companion_compensate(
                    context,
                    sent,
                    chat_id=update.effective_chat.id,
                    message_id=message_id,
                    neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                )
            return True

    @staticmethod
    def _nova_brain_forget_markup(stage: Any) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Забыть",
                        callback_data=stage.callback_data("confirm"),
                    ),
                    InlineKeyboardButton(
                        "Отмена",
                        callback_data=stage.callback_data("cancel"),
                    ),
                ]
            ]
        )

    @staticmethod
    def _nova_brain_forget_target(text: str, memories: tuple[Any, ...]) -> Any | None:
        if not memories:
            return None
        if re.search(r"\b(?:это|неактуально)\b", text, re.IGNORECASE):
            return memories[0] if len(memories) == 1 else None
        query = re.sub(
            r"^забудь\s*,?\s*что\s+я\s+говорил[аи]?\s+о\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        query_tokens = set(re.findall(r"[a-zа-яё0-9]{3,}", query.casefold()))

        def token_matches(left: str, right: str) -> bool:
            return left == right or (len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5])

        scored = [
            (
                sum(
                    any(token_matches(query_token, memory_token) for query_token in query_tokens)
                    for memory_token in re.findall(
                        r"[a-zа-яё0-9]{3,}",
                        observed_memory_display_value(memory.value).casefold(),
                    )
                ),
                memory,
            )
            for memory in memories
        ]
        scored.sort(key=lambda item: (-item[0], item[1].public_id))
        if not scored or scored[0][0] == 0:
            return None
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        return scored[0][1]

    @staticmethod
    def _nova_companion_status_key(
        owner_id: int,
        telegram_user_id: int,
        chat_id: int,
    ) -> tuple[int, int, int]:
        return owner_id, telegram_user_id, chat_id

    def _nova_companion_status_input_is_read_only(
        self,
        text: str,
        telegram_user_id: int | None = None,
        chat_id: int | None = None,
    ) -> bool:
        cleaned = text.strip()
        address = NovaAddressClassifier.classify(cleaned)
        semantic_text = (
            address.content
            if address.kind is NovaAddressKind.VOCATIVE and address.content is not None
            else cleaned
        )
        exact_status = bool(
            _REMINDER_STATUS_QUESTION.fullmatch(semantic_text)
            or _CAPTURE_STATUS_QUESTION.fullmatch(semantic_text)
            or _IDENTITY_NAME_QUESTION.fullmatch(semantic_text)
            or _IDENTITY_CITY_QUESTION.fullmatch(semantic_text)
            or address.is_local_response
        )
        if exact_status:
            return True
        return bool(
            _CONTEXTUAL_STATUS_QUESTION.fullmatch(semantic_text)
            and isinstance(telegram_user_id, int)
            and isinstance(chat_id, int)
            and self._nova_companion_has_status_anchor_hint(telegram_user_id, chat_id)
        )

    def _nova_companion_has_status_anchor_hint(
        self,
        telegram_user_id: int,
        chat_id: int,
    ) -> bool:
        if self.nova_companion_status_receipt_anchor(telegram_user_id, chat_id) is not None:
            return True
        return any(
            pending.telegram_user_id == telegram_user_id and pending.chat_id == chat_id
            for pending in (
                *self._nova_companion_pending_capture_status.values(),
                *self._nova_companion_pending_reminder_status.values(),
            )
        )

    def nova_companion_status_receipt_anchor(
        self,
        telegram_user_id: int,
        chat_id: int,
    ) -> object | None:
        for key, receipt in self._nova_companion_status_receipts.items():
            if key[1] == telegram_user_id and key[2] == chat_id:
                return receipt
        return None

    def nova_companion_invalidate_status_for_input(
        self,
        telegram_user_id: int,
        chat_id: int,
        text: str,
        *,
        expected_receipt: object = _STATUS_RECEIPT_UNSET,
    ) -> bool:
        if self._nova_companion_status_input_is_read_only(
            text,
            telegram_user_id,
            chat_id,
        ):
            return False
        return self.nova_companion_invalidate_status_receipt_exact(
            telegram_user_id,
            chat_id,
            expected_receipt=expected_receipt,
        )

    def nova_companion_invalidate_status_receipt_exact(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_receipt: object = _STATUS_RECEIPT_UNSET,
    ) -> bool:
        removed = False
        for key, receipt in tuple(self._nova_companion_status_receipts.items()):
            if key[1] != telegram_user_id or key[2] != chat_id:
                continue
            if expected_receipt is not _STATUS_RECEIPT_UNSET and receipt is not expected_receipt:
                continue
            if self._nova_companion_status_receipts.get(key) is receipt:
                self._nova_companion_status_receipts.pop(key, None)
                removed = True
        return removed

    @staticmethod
    def _nova_companion_capture_status_kind(
        text: str,
    ) -> Literal["task", "note", "idea"]:
        normalized = text.casefold()
        if "заметк" in normalized:
            return "note"
        if "иде" in normalized:
            return "idea"
        return "task"

    @staticmethod
    def _nova_companion_capture_status_text(
        kind: Literal["task", "note", "idea"],
        *,
        confirmed: bool,
    ) -> str:
        nouns = {
            "task": ("задача", "сохранена", "создана"),
            "note": ("заметка", "сохранена", "создана"),
            "idea": ("идея", "сохранена", "создана"),
        }
        noun, saved, created = nouns[kind]
        return f"Да, {noun} {saved}." if confirmed else f"Пока нет — {noun} ещё не {created}."

    @staticmethod
    def _nova_companion_trim_private_map(values: dict[Any, Any], *, limit: int = 512) -> None:
        while len(values) > limit:
            values.pop(next(iter(values)), None)

    def _nova_companion_bind_pending_capture_status(
        self,
        user: User,
        chat_id: int,
        draft: Any,
        *,
        canonical_message_id: int,
    ) -> None:
        kind = getattr(draft, "kind", None)
        draft_id = getattr(draft, "id", None)
        draft_version = getattr(draft, "version", None)
        title = getattr(draft, "title", None)
        if (
            kind not in {"task", "note", "idea"}
            or not isinstance(draft_id, str)
            or not draft_id
            or not isinstance(draft_version, int)
            or draft_version < 1
            or not isinstance(title, str)
            or not title
            or type(canonical_message_id) is not int
            or canonical_message_id <= 0
        ):
            return
        self._nova_companion_pending_capture_status[draft_id] = _CompanionPendingCaptureStatus(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            access_version=user.access_version,
            draft_id=draft_id,
            draft_version=draft_version,
            canonical_message_id=canonical_message_id,
            kind=kind,
            title=title,
        )
        self._nova_companion_trim_private_map(self._nova_companion_pending_capture_status)

    def nova_companion_pending_capture_anchor(
        self,
        telegram_user_id: int,
        chat_id: int,
        draft_id: str,
        draft_version: int,
        canonical_message_id: int,
    ) -> _CompanionPendingCaptureStatus | None:
        pending = self._nova_companion_pending_capture_status.get(draft_id)
        if (
            pending is None
            or pending.telegram_user_id != telegram_user_id
            or pending.chat_id != chat_id
            or pending.draft_version != draft_version
            or pending.canonical_message_id != canonical_message_id
        ):
            return None
        return pending

    def nova_companion_clear_pending_capture_exact(
        self,
        telegram_user_id: int,
        chat_id: int,
        draft_id: str,
        draft_version: int,
        *,
        expected_pending: _CompanionPendingCaptureStatus,
    ) -> bool:
        pending = self._nova_companion_pending_capture_status.get(draft_id)
        if (
            pending is None
            or pending is not expected_pending
            or pending.telegram_user_id != telegram_user_id
            or pending.chat_id != chat_id
            or pending.draft_version != draft_version
        ):
            return False
        if self._nova_companion_pending_capture_status.get(draft_id) is not pending:
            return False
        self._nova_companion_pending_capture_status.pop(draft_id, None)
        return True

    def nova_companion_record_terminal_capture(
        self,
        telegram_user_id: int,
        chat_id: int,
        outcome: Any,
        *,
        expected_pending: _CompanionPendingCaptureStatus | None,
    ) -> bool:
        result = getattr(outcome, "result", None)
        draft = getattr(result, "draft", None)
        draft_id = getattr(draft, "id", None)
        draft_version = getattr(draft, "version", None)
        if not isinstance(draft_id, str) or not isinstance(draft_version, int):
            return False
        pending = self._nova_companion_pending_capture_status.get(draft_id)
        if (
            pending is None
            or pending is not expected_pending
            or (
                pending.kind != getattr(draft, "kind", None)
                or pending.title != getattr(draft, "title", None)
            )
        ):
            return False
        return self.nova_companion_clear_pending_capture_exact(
            telegram_user_id,
            chat_id,
            draft_id,
            draft_version,
            expected_pending=pending,
        )

    async def _nova_companion_has_active_capture_pending(
        self,
        user: User,
        chat_id: int,
        kind: Literal["task", "note", "idea"],
    ) -> bool:
        candidates = tuple(
            pending
            for pending in self._nova_companion_pending_capture_status.values()
            if pending.owner_id == user.id
            and pending.telegram_user_id == user.telegram_id
            and pending.chat_id == chat_id
            and pending.access_version == user.access_version
            and pending.kind == kind
        )
        if not candidates:
            return False
        try:
            async with self.db.sessions() as session:
                drafts = tuple(
                    await session.scalars(
                        select(DraftInboxItem).where(
                            DraftInboxItem.id.in_(
                                tuple(pending.draft_id for pending in candidates)
                            ),
                            DraftInboxItem.user_id == user.id,
                            DraftInboxItem.telegram_user_id == user.telegram_id,
                            DraftInboxItem.chat_id == chat_id,
                        )
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=capture_pending error_type=%s",
                type(exc).__name__,
            )
            return True
        now = datetime.now(UTC)
        exact_drafts = {draft.id: draft for draft in drafts}
        active = False
        for pending in candidates:
            draft = exact_drafts.get(pending.draft_id)
            expires_at = getattr(draft, "expires_at", None)
            if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            is_active = bool(
                draft is not None
                and draft.version == pending.draft_version
                and draft.preview_message_id == pending.canonical_message_id
                and draft.kind == pending.kind
                and draft.title == pending.title
                and draft.status in {"preview", "editing"}
                and isinstance(expires_at, datetime)
                and expires_at > now
            )
            if is_active:
                active = True
            elif self._nova_companion_pending_capture_status.get(pending.draft_id) is pending:
                self._nova_companion_pending_capture_status.pop(pending.draft_id, None)
        return active

    def nova_companion_bind_pending_reminder_status(
        self,
        session: ReminderFlowSession,
        candidate: NovaCompanionReminderCandidate,
    ) -> None:
        if session.title != candidate.title:
            return
        self._nova_companion_pending_reminder_status[session.id] = _CompanionPendingReminderStatus(
            owner_id=session.owner_id,
            telegram_user_id=session.telegram_user_id,
            chat_id=session.chat_id,
            access_version=session.access_version,
            session_id=session.id,
            title=candidate.title,
        )
        self._nova_companion_trim_private_map(self._nova_companion_pending_reminder_status)

    def nova_companion_record_confirmed_capture(
        self,
        telegram_user_id: int,
        chat_id: int,
        outcome: Any,
        *,
        expected_pending: _CompanionPendingCaptureStatus | None,
    ) -> None:
        result = getattr(outcome, "result", None)
        draft = getattr(result, "draft", None)
        draft_id = getattr(draft, "id", None)
        draft_version = getattr(draft, "version", None)
        if not isinstance(draft_id, str) or not isinstance(draft_version, int):
            return
        pending = self._nova_companion_pending_capture_status.get(draft_id)
        if (
            pending is None
            or pending is not expected_pending
            or pending.telegram_user_id != telegram_user_id
            or pending.chat_id != chat_id
            or pending.draft_version != draft_version
            or pending.kind != getattr(draft, "kind", None)
            or pending.title != getattr(draft, "title", None)
        ):
            return
        if self._nova_companion_pending_capture_status.get(draft_id) is not pending:
            return
        self._nova_companion_pending_capture_status.pop(draft_id, None)
        item = getattr(result, "inbox_item", None)
        if item is None or bool(getattr(result, "duplicate", False)):
            return
        if (
            pending.kind != getattr(item, "kind", None)
            or pending.title != getattr(item, "title", None)
            or pending.owner_id != getattr(item, "user_id", None)
            or draft_id != getattr(item, "draft_id", None)
        ):
            return
        key = self._nova_companion_status_key(
            pending.owner_id,
            pending.telegram_user_id,
            pending.chat_id,
        )
        self._nova_companion_status_receipts[key] = _CompanionStatusReceipt(
            kind=pending.kind,
            owner_id=pending.owner_id,
            telegram_user_id=pending.telegram_user_id,
            chat_id=pending.chat_id,
            access_version=pending.access_version,
            inbox_item_id=item.id,
            inbox_item_version=item.version,
            title=item.title,
        )
        self._nova_companion_trim_private_map(self._nova_companion_status_receipts)

    def nova_companion_record_confirmed_reminder(
        self,
        session: ReminderFlowSession,
        item: InboxItem,
    ) -> None:
        pending = self._nova_companion_pending_reminder_status.pop(session.id, None)
        if (
            pending is None
            or pending.owner_id != session.owner_id
            or pending.telegram_user_id != session.telegram_user_id
            or pending.chat_id != session.chat_id
            or pending.access_version != session.access_version
            or pending.title != session.title
            or item.user_id != session.owner_id
            or item.title != session.title
            or item.kind != "task"
            or item.status != "confirmed"
        ):
            return
        key = self._nova_companion_status_key(
            pending.owner_id,
            pending.telegram_user_id,
            pending.chat_id,
        )
        self._nova_companion_status_receipts[key] = _CompanionStatusReceipt(
            kind="reminder",
            owner_id=pending.owner_id,
            telegram_user_id=pending.telegram_user_id,
            chat_id=pending.chat_id,
            access_version=pending.access_version,
            inbox_item_id=item.id,
            inbox_item_version=item.version,
            title=item.title,
        )
        self._nova_companion_trim_private_map(self._nova_companion_status_receipts)

    async def _nova_companion_status_answer(
        self,
        text: str,
        *,
        user: User,
        chat_id: int,
    ) -> str | None:
        cleaned = text.strip()
        key = self._nova_companion_status_key(user.id, user.telegram_id, chat_id)
        receipt = self._nova_companion_status_receipts.get(key)
        contextual = bool(_CONTEXTUAL_STATUS_QUESTION.fullmatch(cleaned))
        capture_kind: Literal["task", "note", "idea"] | None = None
        reminder_question = bool(_REMINDER_STATUS_QUESTION.fullmatch(cleaned))
        if _CAPTURE_STATUS_QUESTION.fullmatch(cleaned):
            capture_kind = self._nova_companion_capture_status_kind(cleaned)
        elif contextual and receipt is not None and receipt.access_version == user.access_version:
            if receipt.kind in {"task", "note", "idea"}:
                capture_kind = receipt.kind
            elif receipt.kind == "reminder":
                reminder_question = True
        elif contextual:
            pending_kinds = tuple(
                kind
                for kind in ("task", "note", "idea")
                if any(
                    pending.owner_id == user.id
                    and pending.telegram_user_id == user.telegram_id
                    and pending.chat_id == chat_id
                    and pending.access_version == user.access_version
                    and pending.kind == kind
                    for pending in self._nova_companion_pending_capture_status.values()
                )
            )
            for kind in pending_kinds:
                if await self._nova_companion_has_active_capture_pending(user, chat_id, kind):
                    return self._nova_companion_capture_status_text(kind, confirmed=False)
            current_reminder = await self.reminder_sessions.current(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
            )
            pending_reminder = await self.nova_companion_reminders.active(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=chat_id,
                access_tier=user.access_tier,
                access_version=user.access_version,
            )
            if current_reminder is None and pending_reminder is None:
                return None
            reminder_question = True
        if capture_kind is not None:
            kind = capture_kind
            pending = await self._nova_companion_has_active_capture_pending(
                user,
                chat_id,
                kind,
            )
            if (
                pending
                or receipt is None
                or receipt.kind != kind
                or receipt.access_version != user.access_version
            ):
                return self._nova_companion_capture_status_text(kind, confirmed=False)
            if not await self._nova_companion_actor_is_current(user):
                self._nova_companion_status_receipts.pop(key, None)
                return self._nova_companion_capture_status_text(kind, confirmed=False)
            try:
                async with self.db.sessions() as session:
                    saved_item = await session.scalar(
                        select(InboxItem.id).where(
                            InboxItem.id == receipt.inbox_item_id,
                            InboxItem.user_id == receipt.owner_id,
                            InboxItem.kind == receipt.kind,
                            InboxItem.status == "confirmed",
                            InboxItem.version == receipt.inbox_item_version,
                            InboxItem.title == receipt.title,
                        )
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Nova companion failed operation=capture_status error_type=%s",
                    type(exc).__name__,
                )
                return "Не могу сейчас надёжно проверить статус сохранения."
            if saved_item is None:
                self._nova_companion_status_receipts.pop(key, None)
            return self._nova_companion_capture_status_text(
                kind,
                confirmed=saved_item is not None,
            )
        if not reminder_question:
            return None
        current = await self.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
        )
        pending = await self.nova_companion_reminders.active(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            access_tier=user.access_tier,
            access_version=user.access_version,
        )
        if current is not None or pending is not None:
            return "Пока нет — напоминание ещё не создано."
        if (
            receipt is None
            or receipt.kind != "reminder"
            or receipt.access_version != user.access_version
        ):
            return "Пока нет — напоминание ещё не создано."
        if not await self._nova_companion_actor_is_current(user):
            self._nova_companion_status_receipts.pop(key, None)
            return "Пока нет — напоминание ещё не создано."
        try:
            async with self.db.sessions() as session:
                reminder = await session.scalar(
                    select(TaskReminder)
                    .join(InboxItem, InboxItem.id == TaskReminder.inbox_item_id)
                    .where(
                        InboxItem.id == receipt.inbox_item_id,
                        InboxItem.user_id == receipt.owner_id,
                        InboxItem.kind == "task",
                        InboxItem.status == "confirmed",
                        InboxItem.version == receipt.inbox_item_version,
                        InboxItem.title == receipt.title,
                        TaskReminder.telegram_user_id == receipt.telegram_user_id,
                        TaskReminder.chat_id == receipt.chat_id,
                        TaskReminder.status.in_(("pending", "sent")),
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=reminder_status error_type=%s",
                type(exc).__name__,
            )
            return "Не могу сейчас надёжно проверить статус напоминания."
        if reminder is None:
            self._nova_companion_status_receipts.pop(key, None)
            return "Пока нет — напоминание ещё не создано."
        event_at = reminder.event_at
        if event_at.tzinfo is None:
            event_at = event_at.replace(tzinfo=UTC)
        local = event_at.astimezone(ZoneInfo(reminder.timezone))
        today = self._reminder_now().astimezone(ZoneInfo(reminder.timezone)).date()
        day = "завтра" if local.date() == today + timedelta(days=1) else local.strftime("%d.%m.%Y")
        return f"Да, напоминание создано на {day}, {local.strftime('%H:%M')}."

    @staticmethod
    def nova_companion_pending_reminder_status_answer(text: str) -> str | None:
        if _REMINDER_STATUS_QUESTION.fullmatch(text.strip()):
            return "Пока нет — напоминание ещё не создано."
        return None

    @staticmethod
    def _nova_companion_reminder_continuation_action(
        text: str,
    ) -> Literal["accept", "not_now"] | None:
        cleaned = text.strip()
        if _REMINDER_ACCEPT.fullmatch(cleaned):
            return "accept"
        if _REMINDER_DECLINE.fullmatch(cleaned):
            return "not_now"
        return None

    @staticmethod
    def _nova_companion_reminder_continuation_requires_anchor(text: str) -> bool:
        return _REMINDER_WEAK_CONTINUATION.fullmatch(text.strip()) is not None

    async def _nova_companion_handle_reminder_continuation(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        action: Literal["accept", "not_now"],
    ) -> bool:
        capability = await self.nova_companion_reminders.active(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=update.effective_chat.id,
            access_tier=user.access_tier,
            access_version=user.access_version,
            action=action,
        )
        if capability is None:
            return False
        if not await self._nova_companion_reminder_capability_is_current(user, capability):
            return False
        if not await self.nova_companion_reminders.consume(capability):
            return False
        coroutine = self._nova_companion_reminder_action_lifecycle(
            update,
            context,
            delivery_message,
            user=user,
            capability=capability,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-reminder-text-lifecycle",
            )
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        await asyncio.shield(task)
        return True

    async def _nova_companion_reminder_action_lifecycle(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        capability: NovaCompanionReminderCapability,
    ) -> bool:
        try:
            return await self._nova_companion_reminder_action_inner(
                update,
                context,
                delivery_message,
                user=user,
                capability=capability,
            )
        except asyncio.CancelledError as exc:
            exact_session = exc.__dict__.get(_COMPANION_REMINDER_SESSION_ATTR)
            coroutine = self._nova_companion_reminder_consumed_cleanup(
                context,
                capability,
                exact_session=(
                    exact_session if isinstance(exact_session, ReminderFlowSession) else None
                ),
            )
            try:
                task = asyncio.create_task(
                    coroutine,
                    name="nova-companion-reminder-consumed-cleanup",
                )
            except BaseException:
                coroutine.close()
            else:
                self._track_nova_companion_task(task)
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=reminder_action error_type=%s",
                type(exc).__name__,
            )
            exact_session = exc.__dict__.get(_COMPANION_REMINDER_SESSION_ATTR)
            await self._nova_companion_reminder_consumed_cleanup(
                context,
                capability,
                exact_session=(
                    exact_session if isinstance(exact_session, ReminderFlowSession) else None
                ),
            )
            return False

    async def _nova_companion_reminder_action_inner(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        capability: NovaCompanionReminderCapability,
    ) -> bool:
        async with self._nova_companion_reminder_ui_lock:
            if not await self.nova_companion_reminders.consumed_screen_is_current(capability):
                return False
            if not await self._nova_companion_reminder_capability_is_current(user, capability):
                await self._nova_companion_neutralize_reminder_offer_locked(
                    context, capability, NOVA_COMPANION_ACCESS_CHANGED_TEXT
                )
                return False
            if delivery_message is not None:
                try:
                    await delivery_message.delete()
                except asyncio.CancelledError:
                    raise
                except TelegramError as exc:
                    logger.warning(
                        "Nova companion failed operation=reminder_voice_cleanup error_type=%s",
                        type(exc).__name__,
                    )
            if capability.action == "not_now":
                await self._nova_companion_edit_markup(
                    context,
                    None,
                    chat_id=capability.chat_id,
                    message_id=capability.canonical_message_id,
                    markup=None,
                )
                return True
            if capability.canonical_message_id is None:
                return False
            handled = await self.reminder_from_companion_candidate(
                update,
                context,
                candidate=capability.candidate,
                canonical_message_id=capability.canonical_message_id,
                expected_access_version=capability.access_version,
            )
            if not handled:
                current = await self._nova_companion_reminder_capability_is_current(
                    user,
                    capability,
                )
                await self._nova_companion_neutralize_reminder_offer_locked(
                    context,
                    capability,
                    (
                        NOVA_COMPANION_CONTEXT_CHANGED_TEXT
                        if current
                        else NOVA_COMPANION_ACCESS_CHANGED_TEXT
                    ),
                )
            return handled

    async def _nova_companion_reminder_consumed_cleanup(
        self,
        context: Any,
        capability: NovaCompanionReminderCapability,
        *,
        exact_session: ReminderFlowSession | None = None,
    ) -> bool:
        try:
            async with self._nova_companion_reminder_ui_lock:
                if exact_session is not None:
                    cleared = await self.reminder_sessions.clear_exact(exact_session)
                    if not cleared:
                        return False
                return await self._nova_companion_neutralize_reminder_offer_locked(
                    context,
                    capability,
                    NOVA_COMPANION_UNAVAILABLE_TEXT,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=reminder_cleanup error_type=%s",
                type(exc).__name__,
            )
            return False

    async def nova_companion_reminder_callback(self, update: Any, context: Any) -> None:
        query = update.callback_query
        if query is None or update.effective_user is None or update.effective_chat is None:
            return
        status_receipt = self.nova_companion_status_receipt_anchor(
            update.effective_user.id,
            update.effective_chat.id,
        )
        user = await self._nova_companion_lookup_actor(update.effective_user.id)
        message_id = self._positive_companion_message_id(
            getattr(getattr(query, "message", None), "message_id", None)
        )
        capability = None
        if user is not None and message_id is not None:
            capability = await self.nova_companion_reminders.peek_bound_identity(
                query.data,
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                canonical_message_id=message_id,
            )
        await self._nova_companion_answer_query(query)
        if capability is None:
            return
        assert user is not None
        current = await self._nova_companion_reminder_capability_is_current(user, capability)
        if not await self.nova_companion_reminders.consume(capability):
            return
        if not current:
            await self._nova_companion_neutralize_reminder_offer(
                context,
                capability,
                NOVA_COMPANION_ACCESS_CHANGED_TEXT,
            )
            return
        self.nova_companion_invalidate_status_receipt_exact(
            update.effective_user.id,
            update.effective_chat.id,
            expected_receipt=status_receipt,
        )
        coroutine = self._nova_companion_reminder_action_lifecycle(
            update,
            context,
            None,
            user=user,
            capability=capability,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-reminder-callback-lifecycle",
            )
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        await asyncio.shield(task)

    async def _nova_companion_local_response(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        response: str,
        *,
        user: User,
    ) -> None:
        if delivery_message is not None:
            coroutine = self._nova_companion_local_voice_lifecycle(
                context,
                delivery_message,
                response,
                user=user,
                chat_id=update.effective_chat.id,
            )
            try:
                task = asyncio.create_task(
                    coroutine,
                    name="nova-companion-local-voice-lifecycle",
                )
            except BaseException:
                coroutine.close()
                raise
            self._track_nova_companion_task(task)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            return
        coroutine = self._nova_companion_local_text_lifecycle(
            update,
            context,
            response,
            user=user,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-local-text-lifecycle",
            )
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        await asyncio.shield(task)

    async def _nova_companion_local_text_lifecycle(
        self,
        update: Any,
        context: Any,
        response: str,
        *,
        user: User,
    ) -> bool:
        sent = None
        message_id = None
        try:
            if not await self._nova_companion_actor_is_current(user):
                return False
            sent = await update.effective_message.reply_text(response)
            message_id = self._positive_companion_message_id(getattr(sent, "message_id", None))
            if message_id is None:
                raise ValueError("Missing local companion message id")
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_compensate(
                    context,
                    sent,
                    chat_id=update.effective_chat.id,
                    message_id=message_id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            return True
        except asyncio.CancelledError:
            if sent is not None and message_id is not None:
                self._nova_companion_schedule_pre_delivery_cleanup(
                    context,
                    sent,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=local_delivery error_type=%s",
                type(exc).__name__,
            )
            if sent is not None and message_id is not None:
                await self._nova_companion_compensate(
                    context,
                    sent,
                    chat_id=update.effective_chat.id,
                    message_id=message_id,
                    neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                )
            return False

    async def _nova_companion_local_voice_lifecycle(
        self,
        context: Any,
        delivery_message: Any,
        response: str,
        *,
        user: User,
        chat_id: int,
    ) -> bool:
        try:
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_retire_pre_delivery(
                    context,
                    delivery_message,
                    chat_id=chat_id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            await delivery_message.edit_text(response, reply_markup=None)
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_retire_pre_delivery(
                    context,
                    delivery_message,
                    chat_id=chat_id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            return True
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=local_delivery error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            return False

    async def _nova_companion_explicit_capture(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        snapshot: Any,
        source: str,
        original_text: str,
        kind: str,
        content: str | None,
        references_context: bool,
        resolved_date: Any | None,
        temporal_resolution: Any | None,
    ) -> None:
        capture_text = content
        if references_context:
            capture_text = self.conversation.companion_reference_candidate(snapshot)
        if not capture_text:
            notice = "Не уверена, что именно сохранить. Уточни мысль одним сообщением."
            await self._nova_companion_local_response(
                update,
                context,
                delivery_message,
                notice,
                user=user,
            )
            return
        title = capture_text.strip()[:200]
        coroutine = self._nova_companion_explicit_capture_lifecycle(
            update,
            context,
            delivery_message,
            user=user,
            source=source,
            original_text=original_text,
            capture_text=capture_text,
            parsed=ParsedThought(
                kind=kind,
                title=title,
                resolved_date=resolved_date,
                temporal_resolution=temporal_resolution,
            ),
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-explicit-capture-lifecycle",
            )
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise

    async def _nova_companion_explicit_capture_lifecycle(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        source: str,
        original_text: str,
        capture_text: str,
        parsed: ParsedThought,
    ) -> bool:
        creation = None
        preview = None
        accepted_previews: list[Any] = []
        previous_preview_message_id: int | None = None
        preview_pointer_owned = False
        focus_lease = None
        focus_owned = False
        try:
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_retire_explicit_progress(
                    context,
                    delivery_message,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            creation = await self.draft_service.create_or_get_for_suggestion(
                user_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                expected_access_version=user.access_version,
                source=source,
                raw_text=capture_text,
                parsed=parsed,
            )
            if not creation.ok or creation.draft is None:
                await self._nova_companion_retire_explicit_progress(
                    context,
                    delivery_message,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            draft = creation.draft
            previous_preview_message_id = self._positive_companion_message_id(
                draft.preview_message_id
            )
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=None,
                    previous_preview_message_id=previous_preview_message_id,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            message = delivery_message or update.effective_message
            if delivery_message is not None:
                await delivery_message.edit_text("Голос распознан. Проверь preview ниже.")
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=None,
                    previous_preview_message_id=previous_preview_message_id,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            preview = await self._send_draft_preview(
                message,
                draft,
                include_original=source != "voice",
                on_sent=accepted_previews.append,
                bind_preview=False,
            )
            preview_message_id = self._positive_companion_message_id(
                getattr(preview, "message_id", None)
            )
            if (
                preview_message_id is None
                or not await self.draft_service.restore_preview_message_if_current(
                    draft.id,
                    draft.version,
                    user.telegram_id,
                    update.effective_chat.id,
                    expected_message_id=previous_preview_message_id,
                    restored_message_id=preview_message_id,
                )
            ):
                raise RuntimeError("Preview binding changed")
            preview_pointer_owned = True
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            await self.conversation.append(
                user.telegram_id,
                update.effective_chat.id,
                role="user",
                content=original_text.strip(),
                source=source,
                intent="explicit_capture",
                topic=parsed.title,
            )
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            await self._remember_preview(user.telegram_id, update.effective_chat.id, draft)
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            focus_lease = await self.conversation.acquire_active_draft_focus(
                user.telegram_id,
                update.effective_chat.id,
                draft.id,
            )
            if focus_lease is None:
                raise RuntimeError("Draft focus changed")
            focus_owned = True
            if not await self._nova_companion_actor_is_current(user):
                await self._nova_companion_explicit_capture_cleanup(
                    context,
                    delivery_message,
                    creation,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    restore_prior_focus=False,
                    telegram_user_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            self._nova_companion_bind_pending_capture_status(
                user,
                update.effective_chat.id,
                draft,
                canonical_message_id=preview_message_id,
            )
            return True
        except asyncio.CancelledError as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            self._nova_companion_schedule_explicit_capture_cleanup(
                context,
                delivery_message,
                creation,
                preview=preview,
                previous_preview_message_id=previous_preview_message_id,
                preview_pointer_owned=preview_pointer_owned,
                focus_lease=focus_lease,
                focus_owned=focus_owned,
                focus_user=user,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
            exc.__dict__[_COMPANION_CLEANUP_SCHEDULED_ATTR] = True
            raise
        except Exception as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            logger.warning(
                "Nova companion failed operation=explicit_capture error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_explicit_capture_cleanup(
                context,
                delivery_message,
                creation,
                preview=preview,
                previous_preview_message_id=previous_preview_message_id,
                preview_pointer_owned=preview_pointer_owned,
                focus_lease=focus_lease,
                focus_owned=focus_owned,
                focus_user=user,
                telegram_user_id=user.telegram_id,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
            return False

    def _nova_companion_schedule_explicit_capture_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        creation: Any,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool = False,
        focus_lease: Any | None = None,
        focus_owned: bool = False,
        focus_user: User,
        restore_prior_focus: bool | None = None,
        telegram_user_id: int,
        chat_id: int,
        neutral_text: str,
    ) -> None:
        coroutine = self._nova_companion_explicit_capture_cleanup(
            context,
            delivery_message,
            creation,
            preview=preview,
            previous_preview_message_id=previous_preview_message_id,
            preview_pointer_owned=preview_pointer_owned,
            focus_lease=focus_lease,
            focus_owned=focus_owned,
            focus_user=focus_user,
            restore_prior_focus=restore_prior_focus,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-explicit-capture-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_explicit_cleanup error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_explicit_capture_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        creation: Any,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool = False,
        focus_lease: Any | None = None,
        focus_owned: bool = False,
        focus_user: User | None = None,
        restore_prior_focus: bool | None = None,
        telegram_user_id: int,
        chat_id: int,
        neutral_text: str,
    ) -> bool:
        storage_clean = True
        draft = getattr(creation, "draft", None)
        preview_message_id = self._positive_companion_message_id(
            getattr(preview, "message_id", None)
        )
        if creation is not None and bool(getattr(creation, "created", False)):
            storage_clean = await self._nova_companion_drop_created_draft(
                creation,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                expected_preview_message_id=(
                    preview_message_id if preview_pointer_owned else previous_preview_message_id
                ),
            )
        elif draft is not None and preview_message_id is not None and preview_pointer_owned:
            restored = await self._nova_companion_restore_reused_preview_pointer(
                draft,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                expected_message_id=preview_message_id,
                restored_message_id=previous_preview_message_id,
            )
            storage_clean = storage_clean and restored
        if focus_lease is not None:
            storage_clean = (
                await self._nova_companion_restore_focus_lease(
                    focus_lease,
                    user=focus_user,
                    capability=None,
                    restore_prior=restore_prior_focus,
                )
                and storage_clean
            )
        elif focus_owned and draft is not None:
            try:
                cleared = await self.conversation.clear_active_draft_if_current(
                    telegram_user_id,
                    chat_id,
                    draft.id,
                    draft.version,
                )
                storage_clean = storage_clean and cleared
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                storage_clean = False
                logger.warning(
                    "Nova companion failed operation=focus_compensation error_type=%s",
                    type(exc).__name__,
                )
        if preview is not None and preview_message_id is not None:
            await self._nova_companion_compensate(
                context,
                preview,
                chat_id=chat_id,
                message_id=preview_message_id,
                neutral_text=neutral_text,
            )
        await self._nova_companion_retire_explicit_progress(
            context,
            delivery_message,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )
        if not storage_clean:
            logger.warning(
                "Nova companion failed operation=explicit_compensation error_type=StateChanged"
            )
        return storage_clean

    async def _nova_companion_restore_focus_lease(
        self,
        focus_lease: Any,
        *,
        user: User | None,
        capability: NovaCompanionCaptureCapability | None,
        restore_prior: bool | None,
    ) -> bool:
        policy = restore_prior
        if policy is None:
            try:
                if user is None:
                    policy = False
                elif capability is None:
                    policy = await self._nova_companion_actor_is_current(user)
                else:
                    policy = await self._nova_companion_capability_actor_is_current(
                        user,
                        capability,
                    )
            except asyncio.CancelledError:
                policy = False
            except Exception as exc:
                policy = False
                logger.warning(
                    "Nova companion failed operation=focus_policy error_type=%s",
                    type(exc).__name__,
                )
        try:
            await self.conversation.restore_active_draft_focus_if_current(
                focus_lease,
                restore_prior=bool(policy),
            )
            # A CAS miss means another exact focus generation won and must be
            # preserved; it is a successful compensation outcome.
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=focus_compensation error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_restore_reused_preview_pointer(
        self,
        draft: Any,
        *,
        telegram_user_id: int,
        chat_id: int,
        expected_message_id: int,
        restored_message_id: int | None,
    ) -> bool:
        try:
            return await self.draft_service.restore_preview_message_if_current(
                draft.id,
                draft.version,
                telegram_user_id,
                chat_id,
                expected_message_id=expected_message_id,
                restored_message_id=restored_message_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=preview_pointer_compensation error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_retire_explicit_progress(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> bool:
        message_id = self._positive_companion_message_id(
            getattr(delivery_message, "message_id", None)
        )
        if delivery_message is None or message_id is None:
            return True
        await self._nova_companion_compensate(
            context,
            delivery_message,
            chat_id=chat_id,
            message_id=message_id,
            neutral_text=neutral_text,
        )
        return True

    async def _nova_companion_prepare_and_deliver(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        *,
        user: User,
        snapshot: Any,
        text: str,
        source: str,
        suppress_proposals: bool = False,
    ) -> None:
        try:
            (
                memory_status,
                memory_projection,
                memory_revision,
            ) = await self._nova_companion_memory_projection(user)
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        if memory_status != "ready":
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=self._nova_companion_neutral_text(memory_status),
            )
            return
        brain_projection: NovaBrainProjection | None = None
        brain_fence: NovaBrainFence | None = None
        brain_policy = self.nova_brain_policy()
        if brain_policy.allows_actor(user):
            try:
                brain_snapshot = await self.nova_brain_service.snapshot(
                    telegram_actor_id=user.telegram_id,
                    chat_id=update.effective_chat.id,
                    expected_tier=user.access_tier,
                    expected_access_version=user.access_version,
                    current_text=text,
                    policy=brain_policy,
                )
            except asyncio.CancelledError:
                self._nova_companion_schedule_pre_delivery_cleanup(
                    context,
                    delivery_message,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                )
                raise
            except Exception as exc:
                logger.warning(
                    "Nova companion failed operation=brain_snapshot error_type=%s",
                    type(exc).__name__,
                )
                await self._nova_companion_retire_pre_delivery(
                    context,
                    delivery_message,
                    chat_id=update.effective_chat.id,
                    neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                )
                return
            if brain_snapshot.status != "ready" or brain_snapshot.fence is None:
                await self._nova_companion_retire_pre_delivery(
                    context,
                    delivery_message,
                    chat_id=update.effective_chat.id,
                    neutral_text=(
                        NOVA_COMPANION_ACCESS_CHANGED_TEXT
                        if brain_snapshot.status == "access_changed"
                        else NOVA_COMPANION_UNAVAILABLE_TEXT
                    ),
                )
                return
            brain_projection = brain_snapshot.projection
            brain_fence = brain_snapshot.fence
        try:
            materialized = await self.nova_companion_context.snapshot(
                telegram_actor_id=user.telegram_id,
                expected_tier=user.access_tier,
                expected_access_version=user.access_version,
                conversation_context=snapshot.for_companion_prompt(),
                conversation_chat_id=update.effective_chat.id,
                confirmed_memory=memory_projection,
            )
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=context_snapshot error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            return
        if materialized.status != "ready" or materialized.fence is None:
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=update.effective_chat.id,
                neutral_text=self._nova_companion_neutral_text(materialized.status),
            )
            return
        assert materialized.projection is not None
        generation = _CompanionGeneration(
            owner_id=user.id,
            telegram_actor_id=user.telegram_id,
            chat_id=update.effective_chat.id,
            tier=user.access_tier,
            access_version=user.access_version,
            context_fence=materialized.fence,
            memory_revision=memory_revision,
            brain_fence=brain_fence,
        )
        generation_timezone = materialized.fence.timezone_name
        try:
            pre_provider_check = await self._nova_companion_current_check(generation)
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        if pre_provider_check != "ready":
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=self._nova_companion_neutral_text(pre_provider_check),
            )
            return
        capture_temporal: NovaCompanionCaptureTemporal | None = None
        temporal_failed = False
        try:
            date_resolution = self.date_resolver.resolve(text, generation_timezone)
            if date_resolution.status in {"resolved", "conflict"}:
                capture_temporal = NovaCompanionCaptureTemporal(
                    timezone=generation_timezone,
                    resolution=date_resolution,
                    local_time=self.date_resolver.extract_local_time(text),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            temporal_failed = True
            logger.warning(
                "Nova companion failed operation=temporal_resolution error_type=%s",
                type(exc).__name__,
            )
        provider_failed = False
        discourse_anchor: NovaCompanionDiscourseAnchor | None = None
        try:
            provider_context = materialized.projection.provider_payload()
            recent = provider_context.get("recent_conversation")
            recent_messages = recent.get("recent_messages") if isinstance(recent, dict) else None
            discourse_anchor = NovaCompanionDiscourseReducer.reduce(text, recent_messages)
            result = await self.ai.companion_message(
                text,
                temporal_context(generation_timezone),
                materialized.projection,
                discourse_anchor=discourse_anchor,
                brain_context=brain_projection,
            )
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        except Exception as exc:
            provider_failed = True
            diagnostic_code = getattr(exc, "diagnostic_code", None)
            if diagnostic_code == "invalid_answer" or isinstance(exc, ValidationError):
                logger.warning(
                    "Nova companion failed operation=provider error_type=%s "
                    "diagnostic_code=invalid_answer",
                    type(exc).__name__,
                )
            else:
                logger.warning(
                    "Nova companion failed operation=provider error_type=%s",
                    type(exc).__name__,
                )
            result = None
        try:
            post_provider_check = await self._nova_companion_current_check(generation)
        except asyncio.CancelledError:
            self._nova_companion_schedule_pre_delivery_cleanup(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
            )
            raise
        if post_provider_check != "ready":
            await self._nova_companion_retire_pre_delivery(
                context,
                delivery_message,
                chat_id=generation.chat_id,
                neutral_text=self._nova_companion_neutral_text(post_provider_check),
            )
            return
        if provider_failed or result is None:
            prepared = _PreparedCompanionAnswer(
                answer=NOVA_COMPANION_UNAVAILABLE_TEXT,
                user_text=text,
                source=source,
                generation=generation,
                persist_exchange=False,
            )
        else:
            diagnostic_codes: list[NovaCompanionDiagnosticCode] = list(result.diagnostic_codes)

            def reject_proposal(code: NovaCompanionDiagnosticCode) -> None:
                if code not in diagnostic_codes:
                    diagnostic_codes.append(code)

            suggestion = None
            reminder_candidate = None
            if result.capture is not None and not suppress_proposals:
                suggestion = validate_capture_suggestion(
                    kind=result.capture.kind,
                    title=result.capture.title,
                    next_step=result.capture.next_step,
                    user_text=text,
                )
                if suggestion is not None and suggestion.kind == "task" and temporal_failed:
                    suggestion = None
                if suggestion is None:
                    reject_proposal("invalid_capture")
            reminder_offer_attempted = result.reminder_offer is not None and not suppress_proposals
            if reminder_offer_attempted:
                try:
                    reminder_resolution = self.date_resolver.resolve(
                        result.reminder_offer.evidence,
                        generation_timezone,
                    )
                    reminder_temporal = None
                    if reminder_resolution.status == "resolved":
                        reminder_temporal = NovaCompanionCaptureTemporal(
                            timezone=generation_timezone,
                            resolution=reminder_resolution,
                            local_time=self.date_resolver.extract_local_time(
                                result.reminder_offer.evidence
                            ),
                        )
                    elif (
                        reminder_resolution.status == "conflict"
                        or result.reminder_offer.schedule_wording is not None
                    ):
                        raise ValueError("ambiguous reminder offer")
                    reminder_candidate = NovaCompanionReminderCandidate(
                        title=result.reminder_offer.title,
                        schedule_wording=result.reminder_offer.schedule_wording,
                        evidence=result.reminder_offer.evidence,
                        timezone=generation_timezone,
                        temporal=reminder_temporal,
                    )
                except asyncio.CancelledError:
                    raise
                except (TypeError, ValueError):
                    reminder_candidate = None
                if reminder_candidate is None:
                    reject_proposal("invalid_reminder_offer")
            conflicting_action_proposals = (
                not suppress_proposals
                and result.capture is not None
                and result.reminder_offer is not None
            )
            if conflicting_action_proposals:
                suggestion = None
                reminder_candidate = None
                reject_proposal("conflicting_actions")
            rejected_reminder_offer = not suppress_proposals and (
                (reminder_offer_attempted and reminder_candidate is None)
                or "invalid_reminder_offer" in diagnostic_codes
                or "conflicting_actions" in diagnostic_codes
                or conflicting_action_proposals
            )
            suppressed_action_dependency = suppress_proposals and (
                _has_suppressed_action_dependency(result.answer, user_text=text)
            )
            answer = (
                NOVA_COMPANION_RECALL_ACTION_SUPPRESSED_TEXT
                if suppressed_action_dependency
                else (
                    (
                        NOVA_COMPANION_NOT_EXECUTED_TEXT
                        if _has_untrusted_operational_claim(
                            result.answer,
                            user_text=text,
                            action_context=True,
                        )
                        else NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT
                    )
                    if rejected_reminder_offer
                    else result.answer
                )
            )
            answer, discourse_recovered = _companion_discourse_answer(
                answer,
                discourse_anchor,
            )
            raw_answer_replaced = (
                suppressed_action_dependency or rejected_reminder_offer or discourse_recovered
            )
            if discourse_recovered and discourse_anchor is not None:
                suggestion = None
                if discourse_anchor.status == "ambiguous":
                    reminder_candidate = None
            if _has_untrusted_operational_claim(
                answer,
                user_text=text,
                action_context=(
                    reminder_offer_attempted
                    or result.capture is not None
                    or self._nova_companion_has_status_anchor_hint(
                        user.telegram_id,
                        update.effective_chat.id,
                    )
                ),
            ):
                answer = (
                    NOVA_COMPANION_REMINDER_OFFER_ACTION_TEXT
                    if reminder_candidate is not None
                    else NOVA_COMPANION_NOT_EXECUTED_TEXT
                )
                suggestion = None
                raw_answer_replaced = True
            visible_action = (
                "reminder"
                if reminder_candidate is not None
                else "capture"
                if suggestion is not None
                else None
            )
            dialogue_state_update = None
            memory_candidate = None
            if brain_fence is not None and not raw_answer_replaced and not suppress_proposals:
                dialogue_state_update = validate_dialogue_state_update(
                    result.dialogue_state_update,
                    user_text=text,
                    assistant_answer=answer,
                    visible_action=visible_action,
                )
                memory_candidate = validate_memory_candidate(
                    result.memory_candidate,
                    user_text=text,
                )
            if diagnostic_codes:
                logger.warning(
                    "Nova companion provider proposal rejected diagnostic_codes=%s",
                    ",".join(diagnostic_codes),
                )
            prepared = _PreparedCompanionAnswer(
                answer=answer,
                suggestion=suggestion,
                reminder_candidate=reminder_candidate,
                user_text=text,
                source=source,
                generation=generation,
                temporal=(
                    capture_temporal
                    if suggestion is not None and suggestion.kind == "task"
                    else None
                ),
                dialogue_state_update=dialogue_state_update,
                memory_candidate=memory_candidate,
                persist_exchange=True,
            )
            logger.info(
                "Nova companion trace route=companion provider_called=true "
                "profile_count=%s vision_count=%s goal_count=%s confirmed_memory_count=%s "
                "recent_message_count=%s working_state_revision=%s retrieved_memory_count=%s "
                "raw_answer_replaced=%s proposal_accepted=%s",
                int(materialized.projection.profile_present),
                materialized.projection.vision_count,
                materialized.projection.goal_count,
                materialized.projection.memory_count,
                materialized.projection.recent_message_count,
                brain_projection.working_state.revision if brain_projection is not None else 0,
                len(brain_projection.memories) if brain_projection is not None else 0,
                raw_answer_replaced,
                bool(
                    suggestion is not None
                    or reminder_candidate is not None
                    or dialogue_state_update is not None
                    or memory_candidate is not None
                ),
            )
        await self._nova_companion_deliver(
            update,
            context,
            delivery_message,
            prepared,
        )

    async def _nova_companion_retire_pre_delivery(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> bool:
        task = self._nova_companion_start_pre_delivery_cleanup(
            context,
            delivery_message,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )
        if task is None:
            return False
        return await asyncio.shield(task)

    def _nova_companion_schedule_pre_delivery_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> None:
        self._nova_companion_start_pre_delivery_cleanup(
            context,
            delivery_message,
            chat_id=chat_id,
            neutral_text=neutral_text,
        )

    def _nova_companion_start_pre_delivery_cleanup(
        self,
        context: Any,
        delivery_message: Any | None,
        *,
        chat_id: int,
        neutral_text: str,
    ) -> asyncio.Task[bool] | None:
        message_id = self._positive_companion_message_id(
            getattr(delivery_message, "message_id", None)
        )
        if delivery_message is None or message_id is None:
            return None
        coroutine = self._nova_companion_pre_delivery_cleanup(
            context,
            delivery_message,
            chat_id=chat_id,
            message_id=message_id,
            neutral_text=neutral_text,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-pre-delivery-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_pre_delivery_cleanup error_type=%s",
                type(exc).__name__,
            )
            return None
        self._track_nova_companion_task(task)
        return task

    async def _nova_companion_pre_delivery_cleanup(
        self,
        context: Any,
        delivery_message: Any,
        *,
        chat_id: int,
        message_id: int,
        neutral_text: str,
    ) -> bool:
        try:
            await self._nova_companion_neutralize_existing(
                context,
                delivery_message,
                chat_id,
                neutral_text,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=pre_delivery_cleanup error_type=%s",
                type(exc).__name__,
            )
            try:
                await self._nova_companion_compensate(
                    context,
                    delivery_message,
                    chat_id=chat_id,
                    message_id=message_id,
                    neutral_text=neutral_text,
                )
            except asyncio.CancelledError:
                raise
            return False

    async def _nova_companion_memory_projection(
        self, user: User
    ) -> tuple[_CompanionCheck, NovaMemoryProjection | None, str | None]:
        policy = self.nova_memory_application_policy()
        if not policy.allows(user.access_tier):
            return "ready", None, None
        try:
            snapshot = await self.nova_memory_service.application_snapshot(
                telegram_actor_id=user.telegram_id,
                expected_tier=user.access_tier,
                expected_access_version=user.access_version,
                policy=policy,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=memory_snapshot error_type=%s",
                type(exc).__name__,
            )
            return "unavailable", None, None
        if snapshot.status in {"access_changed", "disabled"}:
            return "access_changed", None, None
        if snapshot.status not in {"ready", "empty"} or not snapshot.collection_revision:
            return "unavailable", None, None
        projection = None
        if snapshot.status == "ready":
            try:
                projection = build_nova_memory_projection(
                    snapshot.items,
                    collection_revision=snapshot.collection_revision,
                )
            except NovaMemoryProjectionError as exc:
                logger.warning(
                    "Nova companion failed operation=memory_projection error_type=%s",
                    type(exc).__name__,
                )
                return "unavailable", None, None
        return "ready", projection, snapshot.collection_revision

    async def _nova_companion_current_check(
        self,
        generation: _CompanionGeneration,
        *,
        exchange_receipt: ConversationExchangeReceipt | None = None,
    ) -> _CompanionCheck:
        if not self.nova_companion_policy().allows_tier(generation.tier):
            return "access_changed"
        try:
            if not await self.nova_companion_context.current_check(
                generation.context_fence,
                exchange_receipt=exchange_receipt,
            ):
                return (
                    "context_changed"
                    if await self._nova_companion_generation_actor_is_current(generation)
                    else "access_changed"
                )
            if generation.memory_revision is not None:
                current = await self.nova_memory_service.application_current_check(
                    telegram_actor_id=generation.telegram_actor_id,
                    expected_tier=generation.tier,
                    expected_access_version=generation.access_version,
                    expected_collection_revision=generation.memory_revision,
                    policy=self.nova_memory_application_policy(),
                )
                if current.status in {"access_changed", "disabled"}:
                    return "access_changed"
                if current.status == "memory_changed":
                    return "context_changed"
                if current.status not in {"ready", "empty"}:
                    return "unavailable"
            if (
                generation.brain_fence is not None
                and not await self.nova_brain_service.current_check(
                    generation.brain_fence,
                    policy=self.nova_brain_policy(),
                )
            ):
                return (
                    "context_changed"
                    if await self._nova_companion_generation_actor_is_current(generation)
                    else "access_changed"
                )
            return "ready"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=current_check error_type=%s",
                type(exc).__name__,
            )
            return "unavailable"

    async def _nova_companion_generation_actor_is_current(
        self,
        generation: _CompanionGeneration,
    ) -> bool:
        if not self.nova_companion_policy().allows_tier(generation.tier):
            return False
        try:
            async with self.db.sessions() as session:
                owner_id = await session.scalar(
                    select(User.id).where(
                        User.id == generation.owner_id,
                        User.telegram_id == generation.telegram_actor_id,
                        User.access_tier == generation.tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == generation.access_version,
                    )
                )
            return owner_id == generation.owner_id
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=generation_actor error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_actor_is_current(self, user: User) -> bool:
        if not self.nova_companion_policy().allows_actor(user):
            return False
        try:
            async with self.db.sessions() as session:
                owner_id = await session.scalar(
                    select(User.id).where(
                        User.id == user.id,
                        User.telegram_id == user.telegram_id,
                        User.access_tier == user.access_tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == user.access_version,
                        User.display_name == user.display_name,
                        User.location_city == user.location_city,
                        User.timezone == user.timezone,
                    )
                )
            return owner_id == user.id and self.nova_companion_policy().allows_tier(
                user.access_tier
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=actor_check error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_deliver(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        prepared: _PreparedCompanionAnswer,
    ) -> bool:
        coroutine = self._nova_companion_delivery_lifecycle(
            update,
            context,
            delivery_message,
            prepared,
        )
        try:
            task = asyncio.create_task(coroutine, name="nova-companion-delivery-lifecycle")
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        return await asyncio.shield(task)

    async def _nova_companion_delivery_lifecycle(
        self,
        update: Any,
        context: Any,
        delivery_message: Any | None,
        prepared: _PreparedCompanionAnswer,
    ) -> bool:
        stage: NovaCompanionCaptureScreen | NovaCompanionReminderScreen | None = None
        screen: NovaCompanionCaptureScreen | NovaCompanionReminderScreen | None = None
        exchange_receipt: ConversationExchangeReceipt | None = None
        brain_receipt: NovaBrainApplyReceipt | None = None
        active_reminder_anchor: NovaCompanionReminderCapability | None = None
        sent: Any | None = delivery_message
        message_id = self._positive_companion_message_id(
            getattr(delivery_message, "message_id", None)
        )
        try:
            if prepared.reminder_candidate is None:
                candidate_anchor = await self.nova_companion_reminders.active(
                    owner_id=prepared.generation.owner_id,
                    telegram_user_id=prepared.generation.telegram_actor_id,
                    chat_id=prepared.generation.chat_id,
                    access_tier=prepared.generation.tier,
                    access_version=prepared.generation.access_version,
                )
                if (
                    candidate_anchor is not None
                    and await self._nova_companion_reminder_generation_is_current(candidate_anchor)
                ):
                    active_reminder_anchor = candidate_anchor
            if prepared.suggestion is not None:
                stage = await self.nova_companion_captures.stage(
                    prepared.suggestion,
                    raw_text=prepared.user_text,
                    owner_id=prepared.generation.owner_id,
                    telegram_user_id=prepared.generation.telegram_actor_id,
                    chat_id=prepared.generation.chat_id,
                    access_version=prepared.generation.access_version,
                    temporal=prepared.temporal,
                )
            elif prepared.reminder_candidate is not None:
                stage = await self.nova_companion_reminders.stage(
                    prepared.reminder_candidate,
                    owner_id=prepared.generation.owner_id,
                    telegram_user_id=prepared.generation.telegram_actor_id,
                    chat_id=prepared.generation.chat_id,
                    access_tier=prepared.generation.tier,
                    access_version=prepared.generation.access_version,
                    context_fence=prepared.generation.context_fence,
                    memory_revision=prepared.generation.memory_revision,
                )
            check = await self._nova_companion_current_check(prepared.generation)
            if check != "ready":
                if stage is not None:
                    await self._nova_companion_revoke_offer_screen(stage)
                if delivery_message is not None:
                    await self._nova_companion_neutralize_existing(
                        context,
                        delivery_message,
                        prepared.generation.chat_id,
                        self._nova_companion_neutral_text(check),
                    )
                return False
            if delivery_message is None:
                sent = await update.effective_message.reply_text(prepared.answer)
            else:
                await sent.edit_text(prepared.answer, reply_markup=None)
            message_id = self._positive_companion_message_id(getattr(sent, "message_id", None))
            if message_id is None:
                raise ValueError("Missing companion message id")
            check = await self._nova_companion_current_check(prepared.generation)
            if check != "ready":
                if stage is not None:
                    await self._nova_companion_revoke_offer_screen(stage)
                await self._nova_companion_compensate(
                    context,
                    sent,
                    chat_id=prepared.generation.chat_id,
                    message_id=message_id,
                    neutral_text=self._nova_companion_neutral_text(check),
                )
                return False
            if stage is not None:
                offer_lock = (
                    self._nova_companion_reminder_ui_lock
                    if isinstance(stage, NovaCompanionReminderScreen)
                    else asyncio.Lock()
                )
                async with offer_lock:
                    screen = await self._nova_companion_bind_offer_screen(stage, message_id)
                    if screen is not None:
                        check = await self._nova_companion_current_check(prepared.generation)
                        if check != "ready":
                            await self._nova_companion_delivery_cleanup(
                                context,
                                prepared,
                                sent=sent,
                                message_id=message_id,
                                stage=stage,
                                screen=screen,
                                exchange_receipt=None,
                                neutral_text=self._nova_companion_neutral_text(check),
                                reminder_lock_held=True,
                            )
                            return False
                        try:
                            await self._nova_companion_edit_markup(
                                context,
                                sent,
                                chat_id=prepared.generation.chat_id,
                                message_id=message_id,
                                markup=self._nova_companion_offer_markup(screen),
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.warning(
                                "Nova companion failed operation=suggestion_edit error_type=%s",
                                type(exc).__name__,
                            )
                            if isinstance(screen, NovaCompanionReminderScreen):
                                await self._nova_companion_delivery_cleanup(
                                    context,
                                    prepared,
                                    sent=sent,
                                    message_id=message_id,
                                    stage=stage,
                                    screen=screen,
                                    exchange_receipt=None,
                                    neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                                    reminder_lock_held=True,
                                )
                                return False
                            await self._nova_companion_revoke_offer_screen(screen)
                            await self._nova_companion_edit_markup(
                                context,
                                sent,
                                chat_id=prepared.generation.chat_id,
                                message_id=message_id,
                                markup=None,
                            )
                            screen = None
                        if screen is not None:
                            check = await self._nova_companion_current_check(prepared.generation)
                            if check != "ready":
                                await self._nova_companion_delivery_cleanup(
                                    context,
                                    prepared,
                                    sent=sent,
                                    message_id=message_id,
                                    stage=stage,
                                    screen=screen,
                                    exchange_receipt=None,
                                    neutral_text=self._nova_companion_neutral_text(check),
                                    reminder_lock_held=True,
                                )
                                return False
            if prepared.persist_exchange:
                exchange_receipt = await self._nova_companion_append_exchange(prepared)
                if exchange_receipt is None:
                    check = await self._nova_companion_current_check(prepared.generation)
                    if check == "ready":
                        check = "context_changed"
                    await self._nova_companion_delivery_cleanup(
                        context,
                        prepared,
                        sent=sent,
                        message_id=message_id,
                        stage=stage,
                        screen=screen,
                        exchange_receipt=None,
                        neutral_text=self._nova_companion_neutral_text(check),
                    )
                    return False
                if isinstance(screen, NovaCompanionReminderScreen):
                    advanced = await self.nova_companion_reminders.attach_exchange(
                        screen,
                        exchange_receipt,
                    )
                    if advanced is None:
                        await self._nova_companion_delivery_cleanup(
                            context,
                            prepared,
                            sent=sent,
                            message_id=message_id,
                            stage=stage,
                            screen=screen,
                            exchange_receipt=exchange_receipt,
                            neutral_text=NOVA_COMPANION_CONTEXT_CHANGED_TEXT,
                        )
                        return False
                    screen = advanced
                elif active_reminder_anchor is not None:
                    await self.nova_companion_reminders.advance_context(
                        active_reminder_anchor,
                        context_fence=prepared.generation.context_fence,
                        memory_revision=prepared.generation.memory_revision,
                        exchange_receipt=exchange_receipt,
                    )
            check = await self._nova_companion_current_check(
                prepared.generation,
                exchange_receipt=exchange_receipt,
            )
            if check != "ready":
                await self._nova_companion_delivery_cleanup(
                    context,
                    prepared,
                    sent=sent,
                    message_id=message_id,
                    stage=stage,
                    screen=screen,
                    exchange_receipt=exchange_receipt,
                    neutral_text=self._nova_companion_neutral_text(check),
                )
                return False
            if prepared.dialogue_state_update is not None or prepared.memory_candidate is not None:
                conversation_fence = prepared.generation.context_fence.conversation_fence
                source_identity = (
                    exchange_receipt.source_identity_for(conversation_fence)
                    if exchange_receipt is not None and conversation_fence is not None
                    else None
                )
                if prepared.generation.brain_fence is None or source_identity is None:
                    await self._nova_companion_delivery_cleanup(
                        context,
                        prepared,
                        sent=sent,
                        message_id=message_id,
                        stage=stage,
                        screen=screen,
                        exchange_receipt=exchange_receipt,
                        neutral_text=NOVA_COMPANION_CONTEXT_CHANGED_TEXT,
                    )
                    return False
                brain_receipt = await self.nova_brain_service.apply_turn(
                    prepared.generation.brain_fence,
                    source_identity,
                    state_update=prepared.dialogue_state_update,
                    memory_candidate=prepared.memory_candidate,
                    user_text=prepared.user_text,
                    policy=self.nova_brain_policy(),
                )
                if brain_receipt is None:
                    await self._nova_companion_delivery_cleanup(
                        context,
                        prepared,
                        sent=sent,
                        message_id=message_id,
                        stage=stage,
                        screen=screen,
                        exchange_receipt=exchange_receipt,
                        neutral_text=NOVA_COMPANION_CONTEXT_CHANGED_TEXT,
                    )
                    return False
                result_generation = replace(
                    prepared.generation,
                    brain_fence=brain_receipt.result_fence,
                )
                check = await self._nova_companion_current_check(
                    result_generation,
                    exchange_receipt=exchange_receipt,
                )
                if check != "ready":
                    await self._nova_companion_delivery_cleanup(
                        context,
                        prepared,
                        sent=sent,
                        message_id=message_id,
                        stage=stage,
                        screen=screen,
                        exchange_receipt=exchange_receipt,
                        neutral_text=self._nova_companion_neutral_text(check),
                        brain_receipt=brain_receipt,
                    )
                    return False
            return True
        except asyncio.CancelledError:
            self._nova_companion_schedule_delivery_cleanup(
                context,
                prepared,
                sent=sent,
                message_id=message_id,
                stage=stage,
                screen=screen,
                exchange_receipt=exchange_receipt,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                brain_receipt=brain_receipt,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=delivery error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_delivery_cleanup(
                context,
                prepared,
                sent=sent,
                message_id=message_id,
                stage=stage,
                screen=screen,
                exchange_receipt=exchange_receipt,
                neutral_text=NOVA_COMPANION_UNAVAILABLE_TEXT,
                brain_receipt=brain_receipt,
            )
            return False

    def _nova_companion_schedule_delivery_cleanup(
        self,
        context: Any,
        prepared: _PreparedCompanionAnswer,
        *,
        sent: Any | None,
        message_id: int | None,
        stage: NovaCompanionCaptureScreen | NovaCompanionReminderScreen | None,
        screen: NovaCompanionCaptureScreen | NovaCompanionReminderScreen | None,
        exchange_receipt: ConversationExchangeReceipt | None,
        neutral_text: str,
        brain_receipt: NovaBrainApplyReceipt | None = None,
    ) -> None:
        coroutine = self._nova_companion_delivery_cleanup(
            context,
            prepared,
            sent=sent,
            message_id=message_id,
            stage=stage,
            screen=screen,
            exchange_receipt=exchange_receipt,
            neutral_text=neutral_text,
            brain_receipt=brain_receipt,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-delivery-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_cleanup error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_delivery_cleanup(
        self,
        context: Any,
        prepared: _PreparedCompanionAnswer,
        *,
        sent: Any | None,
        message_id: int | None,
        stage: NovaCompanionCaptureScreen | NovaCompanionReminderScreen | None,
        screen: NovaCompanionCaptureScreen | NovaCompanionReminderScreen | None,
        exchange_receipt: ConversationExchangeReceipt | None,
        neutral_text: str,
        brain_receipt: NovaBrainApplyReceipt | None = None,
        reminder_lock_held: bool = False,
    ) -> bool:
        reminder_screen = isinstance(
            screen if screen is not None else stage,
            NovaCompanionReminderScreen,
        )
        if reminder_screen and not reminder_lock_held:
            async with self._nova_companion_reminder_ui_lock:
                return await self._nova_companion_delivery_cleanup(
                    context,
                    prepared,
                    sent=sent,
                    message_id=message_id,
                    stage=stage,
                    screen=screen,
                    exchange_receipt=exchange_receipt,
                    neutral_text=neutral_text,
                    brain_receipt=brain_receipt,
                    reminder_lock_held=True,
                )
        clean = True
        canonical_owned = True
        try:
            if screen is not None:
                canonical_owned = await self._nova_companion_revoke_offer_screen(screen)
            elif stage is not None:
                await self._nova_companion_revoke_offer_screen(stage)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            clean = False
            if screen is not None:
                canonical_owned = False
            logger.warning(
                "Nova companion failed operation=capability_cleanup error_type=%s",
                type(exc).__name__,
            )
        if exchange_receipt is not None:
            try:
                compensated = await self.conversation.compensate_exchange(exchange_receipt)
                if not compensated:
                    clean = False
                    logger.warning(
                        "Nova companion failed operation=exchange_cleanup error_type=StateChanged"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                clean = False
                logger.warning(
                    "Nova companion failed operation=exchange_cleanup error_type=%s",
                    type(exc).__name__,
                )
        if brain_receipt is not None:
            try:
                compensated = await self.nova_brain_service.compensate_turn(brain_receipt)
                if not compensated:
                    clean = False
                    logger.warning(
                        "Nova companion failed operation=brain_cleanup error_type=StateChanged"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                clean = False
                logger.warning(
                    "Nova companion failed operation=brain_cleanup error_type=%s",
                    type(exc).__name__,
                )
        if canonical_owned and sent is not None and message_id is not None:
            try:
                await self._nova_companion_compensate(
                    context,
                    sent,
                    chat_id=prepared.generation.chat_id,
                    message_id=message_id,
                    neutral_text=neutral_text,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                clean = False
                logger.warning(
                    "Nova companion failed operation=delivery_cleanup error_type=%s",
                    type(exc).__name__,
                )
        return clean

    async def _nova_companion_append_exchange(
        self,
        prepared: _PreparedCompanionAnswer,
    ) -> ConversationExchangeReceipt | None:
        fence = prepared.generation.context_fence.conversation_fence
        if fence is None:
            return None
        return await self.conversation.append_exchange(
            fence,
            user_content=prepared.user_text,
            assistant_content=prepared.answer,
            user_source=prepared.source,
        )

    async def nova_brain_callback(self, update: Any, context: Any) -> None:
        query = update.callback_query
        if query is None:
            return
        user = await self._nova_companion_lookup_actor(update.effective_user.id)
        message_id = self._positive_companion_message_id(
            getattr(getattr(query, "message", None), "message_id", None)
        )
        capability = None
        if user is not None and message_id is not None:
            capability = await self.nova_brain_forget.peek(
                query.data,
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                canonical_message_id=message_id,
            )
        await self._nova_companion_answer_query(query)
        if capability is None or not await self.nova_brain_forget.consume(capability):
            return
        assert user is not None
        coroutine = self._nova_brain_callback_lifecycle(
            query,
            context,
            user,
            capability,
        )
        try:
            task = asyncio.create_task(coroutine, name="nova-brain-memory-callback")
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        await asyncio.shield(task)

    async def _nova_brain_callback_lifecycle(
        self,
        query: Any,
        context: Any,
        user: User,
        capability: NovaBrainForgetCapability,
    ) -> bool:
        terminal_text: str | None = None
        try:
            async with self._nova_brain_ui_lock:
                if not await self.nova_brain_forget.consumed_screen_is_current(capability):
                    return False
                if (
                    capability.access_version != user.access_version
                    or not self.nova_brain_policy().allows_actor(
                        user,
                        expected_access_version=capability.access_version,
                    )
                    or not await self._nova_companion_actor_is_current(user)
                ):
                    terminal_text = NOVA_COMPANION_ACCESS_CHANGED_TEXT
                    await query.edit_message_text(terminal_text, reply_markup=None)
                    return False
                if capability.action == "cancel":
                    terminal_text = "Хорошо, память оставляю."
                    await query.edit_message_text(terminal_text, reply_markup=None)
                    return True
                result = await self.nova_brain_service.forget_exact(
                    telegram_actor_id=user.telegram_id,
                    public_id=capability.memory_public_id,
                    expected_revision=capability.memory_revision,
                    expected_access_version=capability.access_version,
                    policy=self.nova_brain_policy(),
                )
                if result.status == "applied":
                    terminal_text = "Забыла выбранную запись."
                    await query.edit_message_text(terminal_text, reply_markup=None)
                    return True
                terminal_text = "Запись уже изменилась или была удалена. Ничего не меняю."
                await query.edit_message_text(terminal_text, reply_markup=None)
                return False
        except asyncio.CancelledError:
            self._nova_brain_schedule_consumed_cleanup(
                query,
                user,
                capability,
                terminal_text=terminal_text,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=memory_forget error_type=%s",
                type(exc).__name__,
            )
            await self._nova_brain_consumed_cleanup(
                query,
                user,
                capability,
                terminal_text=terminal_text,
            )
            return False

    def _nova_brain_schedule_consumed_cleanup(
        self,
        query: Any,
        user: User,
        capability: NovaBrainForgetCapability,
        *,
        terminal_text: str | None,
    ) -> None:
        coroutine = self._nova_brain_consumed_cleanup(
            query,
            user,
            capability,
            terminal_text=terminal_text,
        )
        try:
            task = asyncio.create_task(coroutine, name="nova-brain-memory-cleanup")
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=memory_forget_cleanup_schedule error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_brain_consumed_cleanup(
        self,
        query: Any,
        user: User,
        capability: NovaBrainForgetCapability,
        *,
        terminal_text: str | None,
    ) -> bool:
        try:
            async with self._nova_brain_ui_lock:
                if not await self.nova_brain_forget.consumed_screen_is_current(capability):
                    return False
                actor_current = await self._nova_companion_actor_is_current(user)
                if terminal_text is not None or not actor_current:
                    await query.edit_message_text(
                        terminal_text or NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                        reply_markup=None,
                    )
                    return True
                recovered = await self.nova_brain_forget.stage_recovery(capability)
                if recovered is None:
                    return False
                try:
                    await query.edit_message_text(
                        "Забыть выбранную запись?",
                        reply_markup=self._nova_brain_forget_markup(recovered),
                    )
                except BaseException:
                    await self.nova_brain_forget.revoke(recovered)
                    raise
                return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=memory_forget_cleanup error_type=%s",
                type(exc).__name__,
            )
            return False

    async def nova_companion_callback(self, update: Any, context: Any) -> None:
        query = update.callback_query
        if query is None:
            return
        status_receipt = self.nova_companion_status_receipt_anchor(
            update.effective_user.id,
            update.effective_chat.id,
        )
        user = await self._nova_companion_lookup_actor(update.effective_user.id)
        message_id = self._positive_companion_message_id(
            getattr(getattr(query, "message", None), "message_id", None)
        )
        capability = None
        if user is not None and message_id:
            capability = await self.nova_companion_captures.peek_bound_identity(
                query.data,
                owner_id=user.id,
                telegram_user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                canonical_message_id=message_id,
            )
        await self._nova_companion_answer_query(query)
        if capability is None or not await self.nova_companion_captures.consume(capability):
            return
        assert user is not None
        coroutine = self._nova_companion_callback_lifecycle(
            update,
            context,
            query,
            user,
            capability,
            status_receipt=status_receipt,
        )
        try:
            task = asyncio.create_task(coroutine, name="nova-companion-callback-lifecycle")
        except BaseException:
            coroutine.close()
            raise
        self._track_nova_companion_task(task)
        await asyncio.shield(task)

    async def _nova_companion_answer_query(self, query: Any) -> None:
        try:
            await query.answer()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_answer error_type=%s",
                type(exc).__name__,
            )

    async def _nova_companion_lookup_actor(self, telegram_actor_id: int) -> User | None:
        try:
            async with self.db.sessions() as session:
                return await session.scalar(
                    select(User).where(User.telegram_id == telegram_actor_id)
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_actor error_type=%s",
                type(exc).__name__,
            )
            return None

    async def _nova_companion_callback_lifecycle(
        self,
        update: Any,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
        *,
        status_receipt: object | None,
    ) -> bool:
        try:
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                return False
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            self.nova_companion_invalidate_status_receipt_exact(
                capability.telegram_user_id,
                capability.chat_id,
                expected_receipt=status_receipt,
            )
            if capability.action == "not_now":
                if await self._nova_companion_clear_query_markup(query, context):
                    return True
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    "Хорошо, не сохраняю.",
                )
                return False
            return await self._nova_companion_add_capture(
                update,
                context,
                query,
                user,
                capability,
            )
        except asyncio.CancelledError as exc:
            if not bool(exc.__dict__.get(_COMPANION_CLEANUP_SCHEDULED_ATTR)):
                self._nova_companion_schedule_callback_recovery(
                    context,
                    query,
                    user,
                    capability,
                )
            raise
        except _NovaCompanionCaptureCompensationError as exc:
            logger.warning(
                "Nova companion failed operation=callback_compensation error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_neutralize_consumed_if_current(
                query,
                context,
                capability,
                NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_lifecycle error_type=%s",
                type(exc).__name__,
            )
            await self._nova_companion_restore_capture(context, query, user, capability)
            return False

    async def _nova_companion_add_capture(
        self,
        update: Any,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        if not await self._nova_companion_capability_actor_is_current(user, capability):
            if await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
            return False
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        if (
            capability.action == "add"
            and capability.temporal is not None
            and capability.temporal.resolution.status == "conflict"
        ):
            return await self._nova_companion_restore_capture(
                context,
                query,
                user,
                capability,
            )
        creation = None
        preview = None
        accepted_previews: list[Any] = []
        previous_preview_message_id: int | None = None
        preview_pointer_owned = False
        focus_lease = None
        focus_owned = False
        try:
            parsed = self._nova_companion_parsed_capture(capability)
            if parsed is None:
                await self._nova_companion_restore_capture(
                    context,
                    query,
                    user,
                    capability,
                )
                return False
            creation = await self.draft_service.create_or_get_for_suggestion(
                user_id=capability.owner_id,
                telegram_user_id=capability.telegram_user_id,
                chat_id=capability.chat_id,
                expected_access_version=capability.access_version,
                source="companion",
                raw_text=capability.raw_text,
                parsed=parsed,
            )
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                if creation.ok and creation.draft is not None:
                    await self._nova_companion_cleanup_capture_preview(
                        context,
                        creation,
                        capability,
                        preview=None,
                    )
                return False
            if not creation.ok or creation.draft is None:
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            draft = creation.draft
            previous_preview_message_id = self._positive_companion_message_id(
                draft.preview_message_id
            )
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=None,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=None,
                )
                return False
            preview = await self._send_draft_preview(
                query.message,
                draft,
                include_original=True,
                on_sent=accepted_previews.append,
                bind_preview=False,
            )
            preview_message_id = self._positive_companion_message_id(
                getattr(preview, "message_id", None)
            )
            if (
                preview_message_id is None
                or not await self.draft_service.restore_preview_message_if_current(
                    draft.id,
                    draft.version,
                    capability.telegram_user_id,
                    capability.chat_id,
                    expected_message_id=previous_preview_message_id,
                    restored_message_id=preview_message_id,
                )
            ):
                raise RuntimeError("Preview binding changed")
            preview_pointer_owned = True
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                )
                return False
            await self._remember_preview(capability.telegram_user_id, capability.chat_id, draft)
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                return False
            cleared = await self._nova_companion_clear_query_markup(query, context)
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_owned=focus_owned,
                )
                return False
            if not cleared:
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_CAPTURE_FAILED_TEXT,
                )
            focus_lease = await self.conversation.acquire_active_draft_focus(
                capability.telegram_user_id,
                capability.chat_id,
                draft.id,
            )
            if focus_lease is None:
                raise RuntimeError("Draft focus changed")
            focus_owned = True
            if not await self._nova_companion_capability_actor_is_current(user, capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    focus_capability=capability,
                    restore_prior_focus=False,
                )
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    capability,
                    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
                )
                return False
            if not await self.nova_companion_captures.consumed_screen_is_current(capability):
                await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    focus_capability=capability,
                )
                return False
            self._nova_companion_bind_pending_capture_status(
                user,
                capability.chat_id,
                draft,
                canonical_message_id=preview_message_id,
            )
            return True
        except asyncio.CancelledError as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            if creation is not None:
                self._nova_companion_schedule_capture_cleanup(
                    context,
                    query,
                    user,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                )
                exc.__dict__[_COMPANION_CLEANUP_SCHEDULED_ATTR] = True
            raise
        except Exception as exc:
            if preview is None and accepted_previews:
                preview = accepted_previews[-1]
            if creation is not None:
                cleaned = await self._nova_companion_cleanup_capture_preview(
                    context,
                    creation,
                    capability,
                    preview=preview,
                    previous_preview_message_id=previous_preview_message_id,
                    preview_pointer_owned=preview_pointer_owned,
                    focus_lease=focus_lease,
                    focus_owned=focus_owned,
                    focus_user=user,
                    focus_capability=capability,
                )
                if not cleaned:
                    raise _NovaCompanionCaptureCompensationError from exc
            raise

    def _nova_companion_schedule_capture_cleanup(
        self,
        context: Any,
        query: Any,
        user: User,
        creation: Any,
        capability: NovaCompanionCaptureCapability,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool,
        focus_lease: Any | None,
        focus_owned: bool,
    ) -> None:
        coroutine = self._nova_companion_capture_cleanup_lifecycle(
            context,
            query,
            user,
            creation,
            capability,
            preview=preview,
            previous_preview_message_id=previous_preview_message_id,
            preview_pointer_owned=preview_pointer_owned,
            focus_lease=focus_lease,
            focus_owned=focus_owned,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-capture-cleanup-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_capture_cleanup error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    def _nova_companion_schedule_callback_recovery(
        self,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> None:
        coroutine = self._nova_companion_callback_recovery_lifecycle(
            context,
            query,
            user,
            capability,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-callback-recovery-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_callback_recovery error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_callback_recovery_lifecycle(
        self,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        if capability.action == "not_now":
            await self._nova_companion_neutralize_consumed_if_current(
                query,
                context,
                capability,
                "Хорошо, не сохраняю.",
            )
            return True
        await self._nova_companion_restore_capture(context, query, user, capability)
        return True

    async def _nova_companion_capture_cleanup_lifecycle(
        self,
        context: Any,
        query: Any,
        user: User,
        creation: Any,
        capability: NovaCompanionCaptureCapability,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None,
        preview_pointer_owned: bool,
        focus_lease: Any | None,
        focus_owned: bool,
    ) -> bool:
        cleaned = await self._nova_companion_cleanup_capture_preview(
            context,
            creation,
            capability,
            preview=preview,
            previous_preview_message_id=previous_preview_message_id,
            preview_pointer_owned=preview_pointer_owned,
            focus_lease=focus_lease,
            focus_owned=focus_owned,
            focus_user=user,
            focus_capability=capability,
        )
        if cleaned:
            await self._nova_companion_restore_capture(context, query, user, capability)
            return True
        await self._nova_companion_neutralize_consumed_if_current(
            query,
            context,
            capability,
            NOVA_COMPANION_CAPTURE_FAILED_TEXT,
        )
        return False

    async def _nova_companion_cleanup_capture_preview(
        self,
        context: Any,
        creation: Any,
        capability: NovaCompanionCaptureCapability,
        *,
        preview: Any | None,
        previous_preview_message_id: int | None = None,
        preview_pointer_owned: bool = False,
        focus_lease: Any | None = None,
        focus_owned: bool = False,
        focus_user: User | None = None,
        focus_capability: NovaCompanionCaptureCapability | None = None,
        restore_prior_focus: bool | None = None,
    ) -> bool:
        storage_clean = True
        preview_id = self._positive_companion_message_id(getattr(preview, "message_id", None))
        if creation.created:
            storage_clean = await self._nova_companion_drop_created_draft(
                creation,
                telegram_user_id=capability.telegram_user_id,
                chat_id=capability.chat_id,
                expected_preview_message_id=(
                    preview_id if preview_pointer_owned else previous_preview_message_id
                ),
            )
        elif creation.draft is not None and preview_id is not None and preview_pointer_owned:
            restored = await self._nova_companion_restore_reused_preview_pointer(
                creation.draft,
                telegram_user_id=capability.telegram_user_id,
                chat_id=capability.chat_id,
                expected_message_id=preview_id,
                restored_message_id=previous_preview_message_id,
            )
            storage_clean = storage_clean and restored
        if focus_lease is not None:
            storage_clean = (
                await self._nova_companion_restore_focus_lease(
                    focus_lease,
                    user=focus_user,
                    capability=focus_capability,
                    restore_prior=restore_prior_focus,
                )
                and storage_clean
            )
        elif focus_owned and creation.draft is not None:
            try:
                cleared = await self.conversation.clear_active_draft_if_current(
                    capability.telegram_user_id,
                    capability.chat_id,
                    creation.draft.id,
                    creation.draft.version,
                )
                storage_clean = storage_clean and cleared
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                storage_clean = False
                logger.warning(
                    "Nova companion failed operation=focus_compensation error_type=%s",
                    type(exc).__name__,
                )
        if preview is not None and preview_id is not None:
            await self._nova_companion_compensate(
                context,
                preview,
                chat_id=capability.chat_id,
                message_id=preview_id,
                neutral_text=NOVA_COMPANION_CAPTURE_FAILED_TEXT,
            )
        return storage_clean

    async def _nova_companion_drop_created_draft(
        self,
        creation: Any,
        *,
        telegram_user_id: int,
        chat_id: int,
        expected_preview_message_id: int | None,
    ) -> bool:
        if not creation.created or creation.draft is None:
            return True
        try:
            result = await self.draft_service.drop_if_preview_message_current(
                creation.draft.id,
                creation.draft.version,
                telegram_user_id,
                chat_id,
                expected_message_id=expected_preview_message_id,
            )
            if result.ok:
                return True
            logger.warning(
                "Nova companion failed operation=draft_compensation error_type=StateChanged"
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=draft_compensation error_type=%s",
                type(exc).__name__,
            )
            return False

    def _nova_companion_parsed_capture(
        self,
        capability: NovaCompanionCaptureCapability,
    ) -> ParsedThought | None:
        resolved_date = None
        temporal_resolution = None
        temporal = capability.temporal
        if temporal is not None:
            resolution = temporal.resolution
            if capability.action == "add":
                if resolution.status != "resolved" or resolution.target_date is None:
                    return None
                resolved_date = resolution.target_date
            elif capability.action in {"date_first", "date_second"}:
                if resolution.status != "conflict" or len(resolution.options) != 2:
                    return None
                option_index = 0 if capability.action == "date_first" else 1
                resolved_date = resolution.options[option_index].value
            else:
                return None
            temporal_resolution = self.date_resolver.temporal_resolution(
                resolved_date,
                temporal.timezone,
                capability.raw_text,
                temporal.local_time,
            )
        elif capability.action != "add":
            return None
        return ParsedThought(
            kind=capability.suggestion.kind,
            title=capability.suggestion.title,
            next_step=capability.suggestion.next_step,
            resolved_date=resolved_date,
            temporal_resolution=temporal_resolution,
        )

    async def _nova_companion_capability_actor_is_current(
        self,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        expected_timezone = (
            capability.temporal.timezone if capability.temporal is not None else None
        )
        if (
            user.id != capability.owner_id
            or user.telegram_id != capability.telegram_user_id
            or user.access_version != capability.access_version
            or (expected_timezone is not None and user.timezone != expected_timezone)
            or not self.nova_companion_policy().allows_actor(
                user,
                expected_access_version=capability.access_version,
            )
        ):
            return False
        if not await self._nova_companion_actor_is_current(user):
            return False
        if expected_timezone is None:
            return True
        try:
            async with self.db.sessions() as session:
                owner_id = await session.scalar(
                    select(User.id).where(
                        User.id == capability.owner_id,
                        User.telegram_id == capability.telegram_user_id,
                        User.access_tier == user.access_tier,
                        User.access_tier.in_(FULL_ACCESS_TIERS),
                        User.access_version == capability.access_version,
                        User.timezone == expected_timezone,
                    )
                )
            return owner_id == capability.owner_id
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=capability_actor error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_reminder_capability_is_current(
        self,
        user: User,
        capability: NovaCompanionReminderCapability,
    ) -> bool:
        if (
            not isinstance(capability.context_fence, NovaCompanionContextFence)
            or user.id != capability.owner_id
            or user.telegram_id != capability.telegram_user_id
            or user.access_tier != capability.access_tier
            or user.access_version != capability.access_version
            or user.timezone != capability.candidate.timezone
            or not self.nova_companion_policy().allows_actor(
                user,
                expected_access_version=capability.access_version,
            )
            or not await self._nova_companion_actor_is_current(user)
        ):
            return False
        receipt = capability.exchange_receipt
        if receipt is not None and not isinstance(receipt, ConversationExchangeReceipt):
            return False
        generation = _CompanionGeneration(
            owner_id=capability.owner_id,
            telegram_actor_id=capability.telegram_user_id,
            chat_id=capability.chat_id,
            tier=capability.access_tier,
            access_version=capability.access_version,
            context_fence=capability.context_fence,
            memory_revision=capability.memory_revision,
        )
        return (
            await self._nova_companion_current_check(
                generation,
                exchange_receipt=receipt,
            )
            == "ready"
        )

    async def _nova_companion_reminder_generation_is_current(
        self,
        capability: NovaCompanionReminderCapability,
    ) -> bool:
        if (
            not isinstance(capability.context_fence, NovaCompanionContextFence)
            or capability.context_fence.owner_id != capability.owner_id
            or capability.context_fence.telegram_actor_id != capability.telegram_user_id
            or capability.context_fence.expected_tier != capability.access_tier
            or capability.context_fence.expected_access_version != capability.access_version
            or capability.context_fence.timezone_name != capability.candidate.timezone
        ):
            return False
        receipt = capability.exchange_receipt
        if receipt is not None and not isinstance(receipt, ConversationExchangeReceipt):
            return False
        return (
            await self._nova_companion_current_check(
                _CompanionGeneration(
                    owner_id=capability.owner_id,
                    telegram_actor_id=capability.telegram_user_id,
                    chat_id=capability.chat_id,
                    tier=capability.access_tier,
                    access_version=capability.access_version,
                    context_fence=capability.context_fence,
                    memory_revision=capability.memory_revision,
                ),
                exchange_receipt=receipt,
            )
            == "ready"
        )

    async def _nova_companion_neutralize_reminder_offer(
        self,
        context: Any,
        capability: NovaCompanionReminderCapability,
        text: str,
    ) -> bool:
        async with self._nova_companion_reminder_ui_lock:
            return await self._nova_companion_neutralize_reminder_offer_locked(
                context,
                capability,
                text,
            )

    async def _nova_companion_neutralize_reminder_offer_locked(
        self,
        context: Any,
        capability: NovaCompanionReminderCapability,
        text: str,
    ) -> bool:
        if (
            capability.canonical_message_id is None
            or not await self.nova_companion_reminders.consumed_screen_is_current(capability)
        ):
            return False
        try:
            await context.bot.edit_message_text(
                chat_id=capability.chat_id,
                message_id=capability.canonical_message_id,
                text=text,
                reply_markup=None,
                parse_mode=None,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=reminder_neutralize error_type=%s",
                type(exc).__name__,
            )
        try:
            result = await context.bot.delete_message(
                chat_id=capability.chat_id,
                message_id=capability.canonical_message_id,
            )
            if result is not False:
                return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=reminder_delete error_type=%s",
                type(exc).__name__,
            )
        try:
            await context.bot.edit_message_text(
                chat_id=capability.chat_id,
                message_id=capability.canonical_message_id,
                text=text,
                reply_markup=None,
                parse_mode=None,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=reminder_neutralize_fallback error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_restore_capture(
        self,
        context: Any,
        query: Any,
        user: User,
        capability: NovaCompanionCaptureCapability,
    ) -> bool:
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        if not await self._nova_companion_capability_actor_is_current(user, capability):
            await self._nova_companion_neutralize_consumed_if_current(
                query,
                context,
                capability,
                NOVA_COMPANION_ACCESS_CHANGED_TEXT,
            )
            return False
        recovery: NovaCompanionCaptureScreen | None = None
        recovery_capability: NovaCompanionCaptureCapability | None = None
        actions = (
            CAPTURE_DATE_ACTIONS
            if capability.temporal is not None
            and capability.temporal.resolution.status == "conflict"
            else CAPTURE_ACTIONS
        )
        primary_action = actions[0]
        try:
            recovery = await self.nova_companion_captures.stage_recovery(
                capability,
                actions=actions,
            )
            if recovery is None:
                return False
            recovery_capability = await self.nova_companion_captures.peek(
                recovery.callback_data(primary_action),
                owner_id=recovery.owner_id,
                telegram_user_id=recovery.telegram_user_id,
                chat_id=recovery.chat_id,
                canonical_message_id=capability.canonical_message_id,
                access_version=recovery.access_version,
                expected_action=primary_action,
            )
            if recovery_capability is None:
                await self.nova_companion_captures.revoke_screen(recovery)
                return False
            await self._nova_companion_render_recovery(
                context,
                query,
                recovery,
            )
            return True
        except asyncio.CancelledError as exc:
            if recovery is not None:
                await self.nova_companion_captures.revoke_screen(recovery)
            if recovery_capability is not None:
                self._nova_companion_schedule_consumed_neutralization(
                    context,
                    query,
                    recovery_capability,
                    NOVA_COMPANION_CAPTURE_FAILED_TEXT,
                )
                exc.__dict__[_COMPANION_CLEANUP_SCHEDULED_ATTR] = True
            raise
        except Exception as exc:
            if recovery is not None:
                await self.nova_companion_captures.revoke_screen(recovery)
            logger.warning(
                "Nova companion failed operation=capture_recovery error_type=%s",
                type(exc).__name__,
            )
            if recovery_capability is not None:
                await self._nova_companion_neutralize_consumed_if_current(
                    query,
                    context,
                    recovery_capability,
                    NOVA_COMPANION_CAPTURE_FAILED_TEXT,
                )
            return False

    async def _nova_companion_render_recovery(
        self,
        context: Any,
        query: Any,
        screen: NovaCompanionCaptureScreen,
    ) -> None:
        markup = self._nova_companion_capture_markup(screen)
        temporal = screen.temporal
        if temporal is not None and temporal.resolution.status == "conflict":
            text = self.date_resolver.conflict_message(temporal.resolution)
            edit = getattr(query, "edit_message_text", None)
            if callable(edit):
                await edit(text, reply_markup=markup)
                return
            bot_edit = getattr(getattr(context, "bot", None), "edit_message_text", None)
            if callable(bot_edit):
                await bot_edit(
                    chat_id=screen.chat_id,
                    message_id=screen.canonical_message_id,
                    text=text,
                    reply_markup=markup,
                    parse_mode=None,
                )
                return
            raise TelegramError("Companion message cannot be edited")
        await self._nova_companion_edit_markup(
            context,
            query.message,
            chat_id=screen.chat_id,
            message_id=screen.canonical_message_id,
            markup=markup,
        )

    def _nova_companion_schedule_consumed_neutralization(
        self,
        context: Any,
        query: Any,
        capability: NovaCompanionCaptureCapability,
        text: str,
    ) -> None:
        coroutine = self._nova_companion_neutralize_consumed_if_current(
            query,
            context,
            capability,
            text,
        )
        try:
            task = asyncio.create_task(
                coroutine,
                name="nova-companion-callback-neutralize-lifecycle",
            )
        except BaseException as exc:
            coroutine.close()
            logger.warning(
                "Nova companion failed operation=schedule_neutralize error_type=%s",
                type(exc).__name__,
            )
            return
        self._track_nova_companion_task(task)

    async def _nova_companion_clear_query_markup(self, query: Any, context: Any) -> bool:
        try:
            edit = getattr(query, "edit_message_reply_markup", None)
            if callable(edit):
                await edit(reply_markup=None)
                return True
            await self._nova_companion_edit_markup(
                context,
                query.message,
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                markup=None,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_edit error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_neutralize_consumed_if_current(
        self,
        query: Any,
        context: Any,
        capability: NovaCompanionCaptureCapability,
        text: str,
    ) -> bool:
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        if await self._nova_companion_neutralize_query(query, context, text):
            return True
        if not await self.nova_companion_captures.consumed_screen_is_current(capability):
            return False
        message = getattr(query, "message", None)
        if (
            message is None
            or self._positive_companion_message_id(getattr(message, "message_id", None))
            != capability.canonical_message_id
        ):
            return False
        return await self._nova_companion_compensate(
            context,
            message,
            chat_id=capability.chat_id,
            message_id=capability.canonical_message_id,
            neutral_text=text,
        )

    async def _nova_companion_neutralize_query(self, query: Any, context: Any, text: str) -> bool:
        try:
            edit = getattr(query, "edit_message_text", None)
            if callable(edit):
                await edit(text, reply_markup=None)
                return True
            await self._nova_companion_neutralize_existing(
                context,
                query.message,
                query.message.chat_id,
                text,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=callback_neutralize error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_edit_markup(
        self,
        context: Any,
        message: Any,
        *,
        chat_id: int,
        message_id: int,
        markup: InlineKeyboardMarkup | None,
    ) -> None:
        edit = getattr(message, "edit_reply_markup", None)
        if callable(edit):
            await edit(reply_markup=markup)
            return
        bot_edit = getattr(getattr(context, "bot", None), "edit_message_reply_markup", None)
        if callable(bot_edit):
            await bot_edit(chat_id=chat_id, message_id=message_id, reply_markup=markup)
            return
        edit_text = getattr(message, "edit_text", None)
        if callable(edit_text):
            text = getattr(message, "text", None)
            if isinstance(text, str):
                await edit_text(text, reply_markup=markup)
                return
        raise TelegramError("Companion message cannot be edited")

    async def _nova_companion_neutralize_existing(
        self,
        context: Any,
        message: Any,
        chat_id: int,
        text: str,
    ) -> bool:
        message_id = self._positive_companion_message_id(getattr(message, "message_id", None))
        if message_id is None:
            return False
        bot_edit = getattr(getattr(context, "bot", None), "edit_message_text", None)
        if callable(bot_edit):
            await bot_edit(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=None,
                parse_mode=None,
            )
            return True
        edit = getattr(message, "edit_text", None)
        if callable(edit):
            await edit(text, reply_markup=None, parse_mode=None)
            return True
        return False

    async def _nova_companion_compensate(
        self,
        context: Any,
        message: Any,
        *,
        chat_id: int,
        message_id: int,
        neutral_text: str,
    ) -> bool:
        deleted = False
        try:
            bot_delete = getattr(getattr(context, "bot", None), "delete_message", None)
            if callable(bot_delete):
                result = await bot_delete(chat_id=chat_id, message_id=message_id)
                deleted = result is not False
            else:
                delete = getattr(message, "delete", None)
                if callable(delete):
                    result = await delete()
                    deleted = result is not False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=delivery_delete error_type=%s",
                type(exc).__name__,
            )
        if deleted:
            return True
        try:
            return await self._nova_companion_neutralize_existing(
                context,
                message,
                chat_id,
                neutral_text,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Nova companion failed operation=delivery_neutralize error_type=%s",
                type(exc).__name__,
            )
            return False

    async def _nova_companion_bind_offer_screen(
        self,
        stage: NovaCompanionCaptureScreen | NovaCompanionReminderScreen,
        message_id: int,
    ) -> NovaCompanionCaptureScreen | NovaCompanionReminderScreen | None:
        if isinstance(stage, NovaCompanionReminderScreen):
            return await self.nova_companion_reminders.bind(stage, canonical_message_id=message_id)
        return await self.nova_companion_captures.bind(stage, canonical_message_id=message_id)

    async def _nova_companion_revoke_offer_screen(
        self,
        screen: NovaCompanionCaptureScreen | NovaCompanionReminderScreen,
    ) -> bool:
        if isinstance(screen, NovaCompanionReminderScreen):
            return await self.nova_companion_reminders.revoke_screen(screen)
        return await self.nova_companion_captures.revoke_screen(screen)

    def _nova_companion_offer_markup(
        self,
        screen: NovaCompanionCaptureScreen | NovaCompanionReminderScreen,
    ) -> InlineKeyboardMarkup:
        if isinstance(screen, NovaCompanionReminderScreen):
            return self._nova_companion_reminder_markup(screen)
        return self._nova_companion_capture_markup(screen)

    @staticmethod
    def _nova_companion_reminder_markup(
        screen: NovaCompanionReminderScreen,
    ) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🔔 Напомнить",
                        callback_data=screen.callback_data("accept"),
                    ),
                    InlineKeyboardButton(
                        "Не сейчас",
                        callback_data=screen.callback_data("not_now"),
                    ),
                ]
            ]
        )

    @staticmethod
    def _nova_companion_capture_markup(
        screen: NovaCompanionCaptureScreen,
    ) -> InlineKeyboardMarkup:
        callbacks = screen.callbacks
        if "date_first" in callbacks and "date_second" in callbacks:
            temporal = screen.temporal
            if temporal is None or temporal.resolution.status != "conflict":
                raise ValueError("Invalid companion date-choice screen")
            first, second = temporal.resolution.options
            return InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"{first.weekday} {first.value.strftime('%d.%m.%Y')}",
                            callback_data=screen.callback_data("date_first"),
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            f"{second.weekday} {second.value.strftime('%d.%m.%Y')}",
                            callback_data=screen.callback_data("date_second"),
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Не сейчас",
                            callback_data=screen.callback_data("not_now"),
                        )
                    ],
                ]
            )
        labels = {
            "idea": "идею",
            "task": "задачу",
            "desire": "желание",
            "note": "заметку",
        }
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        f"Добавить как {labels[screen.suggestion.kind]}",
                        callback_data=screen.callback_data("add"),
                    ),
                    InlineKeyboardButton(
                        "Не сейчас",
                        callback_data=screen.callback_data("not_now"),
                    ),
                ]
            ]
        )

    @staticmethod
    def _positive_companion_message_id(value: object) -> int | None:
        return value if type(value) is int and value > 0 else None

    @staticmethod
    def _nova_companion_neutral_text(outcome: _CompanionCheck) -> str:
        if outcome == "access_changed":
            return NOVA_COMPANION_ACCESS_CHANGED_TEXT
        if outcome == "context_changed":
            return NOVA_COMPANION_CONTEXT_CHANGED_TEXT
        return NOVA_COMPANION_UNAVAILABLE_TEXT

    def _track_nova_companion_task(self, task: asyncio.Task[bool]) -> None:
        self._nova_companion_tasks.add(task)

        def finish(completed: asyncio.Task[bool]) -> None:
            self._nova_companion_tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError as exc:
                logger.warning(
                    "Nova companion task finished operation=lifecycle error_type=%s",
                    type(exc).__name__,
                )
            except BaseException as exc:
                logger.warning(
                    "Nova companion task failed operation=lifecycle error_type=%s",
                    type(exc).__name__,
                )

        task.add_done_callback(finish)

    async def _drain_nova_companion_tasks(self) -> None:
        current = asyncio.current_task()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _COMPANION_DRAIN_TIMEOUT_SECONDS
        while True:
            pending = self._pending_nova_companion_tasks(current)
            if not pending:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            _done, still_pending = await asyncio.wait(pending, timeout=remaining)
            if still_pending:
                break
        pending = self._pending_nova_companion_tasks(current)
        if not pending:
            return
        logger.warning(
            "Nova companion shutdown operation=drain error_type=TimeoutError pending_count=%s",
            len(pending),
        )
        cancel_deadline = loop.time() + _COMPANION_CANCEL_TIMEOUT_SECONDS
        while True:
            pending = self._pending_nova_companion_tasks(current)
            if not pending:
                return
            for task in pending:
                task.cancel()
            remaining = cancel_deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.wait(
                pending,
                timeout=min(_COMPANION_CANCEL_RETRY_SECONDS, remaining),
            )
        pending = self._pending_nova_companion_tasks(current)
        if pending:
            logger.error(
                "Nova companion shutdown operation=terminal_drain error_type=TimeoutError "
                "pending_count=%s",
                len(pending),
            )
            raise _NovaCompanionDrainError("Nova companion terminal drain timed out")

    def _pending_nova_companion_tasks(
        self, current: asyncio.Task[Any] | None
    ) -> set[asyncio.Task[bool]]:
        for task in tuple(self._nova_companion_tasks):
            if task is current or not task.done():
                continue
            try:
                task.result()
            except BaseException:
                pass
            self._nova_companion_tasks.discard(task)
        return {
            task
            for task in tuple(self._nova_companion_tasks)
            if task is not current and not task.done()
        }
