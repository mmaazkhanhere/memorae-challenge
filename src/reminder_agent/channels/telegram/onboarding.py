"""Translate Telegram onboarding updates into application requests."""

from __future__ import annotations

from reminder_agent.application.contracts import RequestSource
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.features.onboarding.request_contracts import (
    OnboardingIdentityContext,
    OnboardingRequest,
    classify_onboarding_text,
)
from reminder_agent.shared.idempotency import request_hash

TELEGRAM = "TELEGRAM"


def to_onboarding_request(update: TelegramUpdate) -> OnboardingRequest | None:
    """Extract transport fields and classify one private Telegram update."""

    if update.message is None or update.message.chat.type != "private":
        return None

    message = update.message
    contact = message.contact
    action, argument = classify_onboarding_text(
        message.text or "",
        has_contact=contact is not None,
        has_voice=message.voice is not None,
    )
    sender_id = str(message.from_user.id)
    return OnboardingRequest(
        source=RequestSource.TELEGRAM,
        identity=OnboardingIdentityContext(
            channel_type=TELEGRAM,
            external_id=sender_id,
            destination_id=str(message.chat.id),
        ),
        idempotency_key=str(update.update_id),
        request_hash=request_hash(update),
        action=action,
        text=message.text or "",
        argument=argument,
        display_name=" ".join(
            part for part in (message.from_user.first_name, message.from_user.last_name) if part
        ).strip()
        or message.from_user.username
        or "Telegram user",
        contact_phone=contact.phone_number if contact is not None else None,
        contact_owner_id=(str(contact.user_id) if contact is not None and contact.user_id is not None else None),
    )


__all__ = ["to_onboarding_request"]
