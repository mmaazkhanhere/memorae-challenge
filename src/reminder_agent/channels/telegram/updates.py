"""Translate validated Telegram updates into application inputs.

This module contains all Telegram-specific inbound normalization: private
message extraction, linked-identity lookup, onboarding requests, and signed
completion callbacks.  It deliberately does not interpret reminder intent or
perform business operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    AuthenticatedRequestContext,
    RequestSource,
    VerifiedChannelIdentity,
)
from reminder_agent.application.conversations import telegram_session_id
from reminder_agent.application.identity.resolve import ResolvedIdentity, resolve_linked_identity
from reminder_agent.channels.telegram.callbacks import parse_callback_data
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.features.onboarding.request_contracts import (
    OnboardingIdentityContext,
    OnboardingRequest,
    classify_onboarding_text,
)
from reminder_agent.infrastructure.database.models import ChannelIdentity, User
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.idempotency import request_hash

TELEGRAM = "TELEGRAM"
logger = get_logger()


@dataclass(frozen=True)
class TelegramInbound:
    """Private-message fields required by the shared application boundary."""

    update: TelegramUpdate
    telegram_user_id: int
    chat_id: int
    text: str
    external_message_id: str


@dataclass(frozen=True)
class TelegramIdentityInput:
    """Telegram sender and destination fields for identity resolution."""

    telegram_user_id: int
    chat_id: int


def message_input(update: TelegramUpdate) -> TelegramInbound | None:
    """Extract private-message transport fields without interpreting them."""

    if update.message is None or update.message.chat.type != "private":
        return None
    return TelegramInbound(
        update=update,
        telegram_user_id=update.message.from_user.id,
        chat_id=update.message.chat.id,
        text=update.message.text or "",
        external_message_id=str(update.update_id),
    )


def identity_input(update: TelegramUpdate) -> TelegramIdentityInput | None:
    """Extract private-message identity fields without resolving a user."""

    inbound = message_input(update)
    if inbound is None:
        return None
    return TelegramIdentityInput(
        telegram_user_id=inbound.telegram_user_id,
        chat_id=inbound.chat_id,
    )


def to_application_request(
    inbound: TelegramInbound,
    *,
    user: User,
    identity: ChannelIdentity,
) -> ApplicationRequest:
    """Convert an authenticated Telegram message into a neutral request."""

    return ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.TELEGRAM,
            canonical_user_id=user.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type=identity.channel_type,
            ),
            external_session_id=telegram_session_id(inbound.chat_id),
        ),
        message_text=inbound.text,
        idempotency_key=inbound.external_message_id,
    )


def resolve_identity(db: Session, update: TelegramUpdate) -> ResolvedIdentity | None:
    """Resolve the linked user for one private Telegram message."""

    identity = identity_input(update)
    if identity is None:
        return None
    return resolve_linked_identity(
        db,
        channel_type=TELEGRAM,
        external_id=str(identity.telegram_user_id),
    )


def to_onboarding_request(update: TelegramUpdate) -> OnboardingRequest | None:
    """Translate one private Telegram update into an onboarding request."""

    inbound = message_input(update)
    if inbound is None or update.message is None:
        return None
    message = update.message
    contact = message.contact
    action, argument = classify_onboarding_text(
        inbound.text,
        has_contact=contact is not None,
        has_voice=message.voice is not None,
    )
    return OnboardingRequest(
        source=RequestSource.TELEGRAM,
        identity=OnboardingIdentityContext(
            channel_type=TELEGRAM,
            external_id=str(inbound.telegram_user_id),
            destination_id=str(inbound.chat_id),
        ),
        idempotency_key=inbound.external_message_id,
        request_hash=request_hash(update),
        action=action,
        text=inbound.text,
        argument=argument,
        display_name=" ".join(
            part for part in (message.from_user.first_name, message.from_user.last_name) if part
        ).strip()
        or message.from_user.username
        or "Telegram user",
        contact_phone=contact.phone_number if contact is not None else None,
        contact_owner_id=(str(contact.user_id) if contact is not None and contact.user_id is not None else None),
    )


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
    update: TelegramUpdate,
    *,
    callback_secret: str,
) -> TelegramCompletionCallback | None:
    """Validate a signed private-chat completion callback."""

    callback = update.callback_query
    message = callback.message if callback is not None else None
    if callback is None or message is None:
        return None
    log_event(
        logger, logging.INFO, "completion_callback_received", update_id=update.update_id,
        callback_query_id=callback.id, telegram_user_id=callback.from_user.id,
        chat_id=message.chat.id, has_data=bool(callback.data),
    )
    if message.chat.type != "private" or not callback.data:
        log_event(logger, logging.INFO, "completion_callback_rejected", update_id=update.update_id,
                  reason="non_private_chat" if message.chat.type != "private" else "missing_data")
        return None
    if not callback.data.startswith("done:"):
        log_event(logger, logging.INFO, "completion_callback_rejected", update_id=update.update_id,
                  reason="unsupported_action")
        return None
    occurrence_id = parse_callback_data(callback.data, callback_secret)
    if occurrence_id is None:
        log_event(logger, logging.WARNING, "completion_callback_rejected", update_id=update.update_id,
                  reason="invalid_signature_or_occurrence_id")
        return None
    return TelegramCompletionCallback(
        occurrence_id=occurrence_id, update_id=update.update_id,
        callback_query_id=callback.id, telegram_user_id=callback.from_user.id,
        chat_id=message.chat.id, provider_message_id=str(message.message_id),
    )
