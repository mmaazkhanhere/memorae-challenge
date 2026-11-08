"""Telegram rendering helpers for onboarding results."""

from reminder_agent.application.contracts import OutboundResponse
from reminder_agent.channels.telegram.presentation.keyboards import (
    PHONE_KEYBOARD,
    REMOVE_KEYBOARD,
)


def onboarding_text(response: OutboundResponse) -> str:
    """Render channel-specific wording from a semantic onboarding result."""

    return response.content


__all__ = ["PHONE_KEYBOARD", "REMOVE_KEYBOARD", "onboarding_text"]
