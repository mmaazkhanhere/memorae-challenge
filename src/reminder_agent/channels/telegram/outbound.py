"""Telegram outbound transport and delivery policy.

Application services return provider-neutral results.  This module owns the
Telegram-specific concerns of chunking, keyboards, callback acknowledgement,
rate-limit handling, and message delivery retries.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from reminder_agent.channels.telegram.provider import TelegramAPIError, TelegramClient

TELEGRAM_MAX_MESSAGE_LENGTH = 4096

PHONE_KEYBOARD = {
    "keyboard": [
        [{"text": "Share my phone number", "request_contact": True}],
        [{"text": "Skip for now"}],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": True,
}
REMOVE_KEYBOARD = {"remove_keyboard": True}


def done_keyboard(callback_data: str) -> dict:
    """Build Telegram's inline completion action."""

    return {"inline_keyboard": [[{"text": "Done", "callback_data": callback_data}]]}


def split_text(text: str, *, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into Telegram-safe chunks without changing its content."""

    if max_length < 1:
        raise ValueError("max_length must be positive")
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        boundary = remaining.rfind("\n", 0, max_length + 1)
        if boundary <= 0:
            boundary = remaining.rfind(" ", 0, max_length + 1)
        if boundary <= 0:
            boundary = max_length
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip(" \n")
    if remaining:
        chunks.append(remaining)
    return chunks


def is_retryable(error: TelegramAPIError) -> bool:
    """Only Telegram rate limits are safe for synchronous retry."""

    return error.status_code == 429


def send_text(
    client: TelegramClient,
    *,
    chat_id: int,
    text: str,
    reply_markup: dict | None = None,
    max_attempts: int = 1,
    retry_delay_seconds: float = 0,
    on_attempt: Callable[[int, int], None] | None = None,
) -> None:
    """Deliver application text through Telegram, handling only transport policy."""

    chunks = split_text(text)
    for chunk_index, chunk in enumerate(chunks):
        for attempt in range(1, max_attempts + 1):
            if on_attempt is not None:
                on_attempt(chunk_index, attempt)
            try:
                kwargs = {
                    "reply_markup": reply_markup
                } if reply_markup is not None and chunk_index == len(chunks) - 1 else {}
                client.send_message(chat_id, chunk, **kwargs)
                break
            except TelegramAPIError as exc:
                if not is_retryable(exc) or attempt >= max_attempts:
                    raise
                if retry_delay_seconds:
                    time.sleep(retry_delay_seconds)


def acknowledge_callback(client: TelegramClient, callback_query_id: str) -> None:
    """Stop Telegram's inline-button spinner after application processing."""

    client.answer_callback_query(callback_query_id)


def edit_text(
    client: TelegramClient,
    *,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: dict | None = None,
) -> None:
    """Edit a Telegram message using the provider adapter."""

    client.edit_message_text(
        chat_id,
        message_id,
        text,
        reply_markup=reply_markup,
    )


def remove_inline_keyboard(
    client: TelegramClient,
    *,
    chat_id: int,
    message_id: int,
    text: str,
) -> None:
    """Remove an inline action keyboard while preserving the message text."""

    if hasattr(client, "edit_message_reply_markup"):
        client.edit_message_reply_markup(
            chat_id,
            message_id,
            reply_markup={"inline_keyboard": []},
        )
    else:  # pragma: no cover - compatibility with older test doubles
        edit_text(
            client,
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup={"inline_keyboard": []},
        )


def delete_message(client: TelegramClient, *, chat_id: int, message_id: int) -> None:
    """Delete a Telegram message using the provider adapter."""

    client.delete_message(chat_id, message_id)
