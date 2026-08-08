from __future__ import annotations

import logging
from typing import Any

from telegram.error import TelegramError

logger = logging.getLogger(__name__)

_MEDIA_TEXT_EDIT_ERRORS = (
    "there is no text in the message to edit",
    "message is not a text message",
    "message to edit is not a text message",
)


async def edit_callback_screen(
    query: Any,
    text: str,
    reply_markup: Any,
    *,
    parse_mode: str | None = None,
    operation: str,
) -> bool:
    """Edit one callback screen without duplicating it on ordinary failures."""

    kwargs = {"parse_mode": parse_mode} if parse_mode is not None else {}
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, **kwargs)
        return True
    except TelegramError as exc:
        value = str(exc).casefold()
        if "message is not modified" in value:
            return True
        if not any(marker in value for marker in _MEDIA_TEXT_EDIT_ERRORS):
            logger.warning(
                "Callback screen edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False
    except (TypeError, AttributeError) as exc:
        logger.warning(
            "Callback screen edit failed operation=%s error_type=%s",
            operation,
            type(exc).__name__,
        )
        return False

    edit_caption = getattr(query, "edit_message_caption", None)
    if callable(edit_caption) and len(text) <= 1024:
        try:
            await edit_caption(caption=text, reply_markup=reply_markup, **kwargs)
            return True
        except TelegramError as exc:
            if "message is not modified" in str(exc).casefold():
                return True
            logger.warning(
                "Callback caption edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "Callback caption edit failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return False

    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except (TelegramError, TypeError, AttributeError) as exc:
        logger.warning(
            "Callback controls retirement failed operation=%s error_type=%s",
            operation,
            type(exc).__name__,
        )
    try:
        await query.message.reply_text(text, reply_markup=reply_markup, **kwargs)
        return True
    except (TelegramError, TypeError, AttributeError) as exc:
        logger.warning(
            "Callback replacement failed operation=%s error_type=%s",
            operation,
            type(exc).__name__,
        )
        return False
