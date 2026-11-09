"""Normalize validated Telegram updates into application requests."""

from __future__ import annotations

from dataclasses import dataclass

from reminder_agent.application.contracts import (
    ApplicationRequest,
    AuthenticatedRequestContext,
    RequestSource,
    VerifiedChannelIdentity,
)
from reminder_agent.application.conversations import telegram_session_id
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.infrastructure.db.models import ChannelIdentity, User


@dataclass(frozen=True)
class TelegramInbound:
    """Normalized fields needed by the shared application boundary."""

    update: TelegramUpdate
    telegram_user_id: int
    chat_id: int
    text: str
    external_message_id: str


def to_application_request(
    inbound: TelegramInbound,
    *,
    user: User,
    identity: ChannelIdentity,
) -> ApplicationRequest:
    """Convert a verified Telegram message into a channel-neutral request."""

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


def message_input(update: TelegramUpdate) -> TelegramInbound | None:
    """Extract only transport fields; no business interpretation occurs here."""

    if update.message is None or update.message.chat.type != "private":
        return None
    return TelegramInbound(
        update=update,
        telegram_user_id=update.message.from_user.id,
        chat_id=update.message.chat.id,
        text=update.message.text or "",
        external_message_id=str(update.update_id),
    )
