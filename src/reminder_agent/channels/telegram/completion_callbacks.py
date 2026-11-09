"""Telegram completion callback parsing and provider-field normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from reminder_agent.channels.telegram.callbacks import parse_callback_data
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()


@dataclass(frozen=True)
class TelegramCompletionCallback:
    """Validated Telegram fields required by the completion service."""

    occurrence_id: UUID
    update_id: int
    callback_query_id: str
    telegram_user_id: int
    chat_id: int
    provider_message_id: str


def parse_completion_callback(
    update: Any,
    *,
    callback_secret: str,
) -> TelegramCompletionCallback | None:
    """Validate a Telegram callback and normalize its provider fields."""

    callback = update.callback_query
    message = callback.message if callback is not None else None
    if callback is None or message is None:
        return None
    log_event(
        logger,
        logging.INFO,
        "completion_callback_received",
        update_id=update.update_id,
        callback_query_id=callback.id,
        telegram_user_id=callback.from_user.id,
        chat_id=message.chat.id,
        has_data=bool(callback.data),
    )
    if message.chat.type != "private" or not callback.data:
        log_event(
            logger,
            logging.INFO,
            "completion_callback_rejected",
            update_id=update.update_id,
            reason=("non_private_chat" if message.chat.type != "private" else "missing_data"),
        )
        return None
    if not callback.data.startswith("done:"):
        log_event(
            logger,
            logging.INFO,
            "completion_callback_rejected",
            update_id=update.update_id,
            reason="unsupported_action",
        )
        return None
    occurrence_id = parse_callback_data(callback.data, callback_secret)
    if occurrence_id is None:
        log_event(
            logger,
            logging.WARNING,
            "completion_callback_rejected",
            update_id=update.update_id,
            reason="invalid_signature_or_occurrence_id",
        )
        return None
    return TelegramCompletionCallback(
        occurrence_id=occurrence_id,
        update_id=update.update_id,
        callback_query_id=callback.id,
        telegram_user_id=callback.from_user.id,
        chat_id=message.chat.id,
        provider_message_id=str(message.message_id),
    )
