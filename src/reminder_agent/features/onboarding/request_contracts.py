"""Provider-neutral onboarding input and command contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from reminder_agent.application.contracts import RequestSource


class OnboardingAction(StrEnum):
    """Semantic onboarding action selected from normalized channel input."""

    START = "START"
    LINK_PHONE = "LINK_PHONE"
    SKIP_PHONE = "SKIP_PHONE"
    PHONE_REGION = "PHONE_REGION"
    CONTACT = "CONTACT"
    VOICE = "VOICE"
    MESSAGE = "MESSAGE"


@dataclass(frozen=True)
class OnboardingIdentityContext:
    """Provider-neutral identity fields extracted by a channel adapter."""

    channel_type: str
    external_id: str
    destination_id: str


@dataclass(frozen=True)
class OnboardingRequest:
    """Normalized onboarding input handed to application orchestration."""

    source: RequestSource
    identity: OnboardingIdentityContext
    idempotency_key: str
    request_hash: str
    action: OnboardingAction
    text: str = ""
    argument: str | None = None
    display_name: str = ""
    contact_phone: str | None = None
    contact_owner_id: str | None = None


def classify_onboarding_text(
    text: str, *, has_contact: bool = False, has_voice: bool = False
) -> tuple[OnboardingAction, str | None]:
    """Classify supported onboarding input without provider payload types."""

    if has_contact:
        return OnboardingAction.CONTACT, None

    stripped = text.strip()
    lowered = stripped.lower()
    if stripped == "/start" or stripped.startswith("/start "):
        return OnboardingAction.START, None
    if lowered in {"/link_phone", "link phone"}:
        return OnboardingAction.LINK_PHONE, None
    if lowered in {"skip", "skip for now", "/skip_phone"}:
        return OnboardingAction.SKIP_PHONE, None

    parts = stripped.split()
    if parts and parts[0].lower() in {"/phone_region", "phone region"}:
        return OnboardingAction.PHONE_REGION, parts[1] if len(parts) == 2 else ""
    if has_voice:
        return OnboardingAction.VOICE, None
    return OnboardingAction.MESSAGE, None


__all__ = [
    "OnboardingAction",
    "OnboardingIdentityContext",
    "OnboardingRequest",
    "classify_onboarding_text",
]
