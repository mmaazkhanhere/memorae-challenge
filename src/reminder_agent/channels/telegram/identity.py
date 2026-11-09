"""Telegram-only identity extraction and authentication helpers.

This module is deliberately limited to translating Telegram sender and chat
fields into the provider-neutral identity lookup used by the application
boundary.  It does not inspect intent, pending workflow state, or reminder
data.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from reminder_agent.application.identity.resolve import ResolvedIdentity, resolve_linked_identity
from reminder_agent.channels.telegram.provider import TelegramUpdate

TELEGRAM = "TELEGRAM"


@dataclass(frozen=True)
class TelegramIdentityInput:
    """Provider identity fields extracted from a validated Telegram update."""

    telegram_user_id: int
    chat_id: int


def identity_input(update: TelegramUpdate) -> TelegramIdentityInput | None:
    """Extract the sender and destination identity from a message update."""

    if update.message is None or update.message.chat.type != "private":
        return None
    return TelegramIdentityInput(
        telegram_user_id=update.message.from_user.id,
        chat_id=update.message.chat.id,
    )


def resolve_identity(db: Session, update: TelegramUpdate) -> ResolvedIdentity | None:
    """Resolve a linked Telegram identity through the shared identity service."""

    fields = identity_input(update)
    if fields is None:
        return None
    return resolve_linked_identity(
        db,
        channel_type=TELEGRAM,
        external_id=str(fields.telegram_user_id),
    )
