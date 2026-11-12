"""Provider-neutral onboarding input and command contracts."""

from __future__ import annotations

import re
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


_PHONE_LINK_REQUEST = re.compile(
    r"^(?:(?:i|i'd|i\s+would|we|we'd|we\s+would)\s+)?"
    r"(?:want|would\s+like|would\s+love|like|need|am\s+ready|would\s+prefer)\s+"
    r"(?:to\s+)?(?:share|send|provide|give)\s+(?:my\s+)?"
    r"(?:phone\s+)?(?:number|contact)(?:\s+(?:with|to)\s+(?:you|the\s+bot))?[.!?]*$",
    re.IGNORECASE,
)
_PHONE_CONNECT_REQUEST = re.compile(
    r"^(?:(?:i|i'd|i\s+would|we|we'd|we\s+would)\s+)?"
    r"(?:want|would\s+like|would\s+love|like|need|am\s+ready|would\s+prefer)\s+"
    r"(?:to\s+)?(?:link|connect)\s+(?:my\s+)?phone(?:\s+number)?[.!?]*$",
    re.IGNORECASE,
)
_PHONE_PERMISSION_REQUEST = re.compile(
    r"^(?:can|could|may)\s+i\s+(?:share|send|provide|give)\s+(?:my\s+)?"
    r"(?:phone\s+)?(?:number|contact)[.!?]*$|"
    r"^(?:is\s+it\s+okay\s+if\s+i|would\s+it\s+be\s+okay\s+if\s+i)\s+"
    r"(?:share|send|provide|give)\s+(?:my\s+)?(?:phone\s+)?(?:number|contact)[.!?]*$",
    re.IGNORECASE,
)
_PHONE_CALLS_REQUEST = re.compile(
    r"^(?:please\s+)?(?:(?:i|i'd|i\s+would)\s+)?(?:want|would\s+like|need|would\s+love|am\s+ready)\s+"
    r"(?:to\s+)?(?:enable|set\s+up|use|have|talk\s+to\s+you\s+on)\s+"
    r"(?:phone\s+)?calls?[.!?]*$",
    re.IGNORECASE,
)


def is_phone_link_request(text: str) -> bool:
    """Recognize explicit natural-language requests to open phone linking."""

    normalized = " ".join(text.strip().split())
    normalized_key = normalized.rstrip(".!?").casefold()
    if normalized_key in {
        "share my number",
        "share my phone number",
        "send my number",
        "provide my number",
        "give you my number",
        "link my phone",
        "connect my phone",
        "enable phone calls",
        "set up phone calls",
        "please enable phone calls",
        "please set up phone calls",
        "how do i share my phone number",
        "how can i share my phone number",
        "please connect my phone",
        "please connect my phone number",
        "let's connect my phone",
        "lets connect my phone",
        "i want to talk by phone",
        "i'd like to talk by phone",
        "can we set up phone calls",
    }:
        return True
    return bool(
        _PHONE_LINK_REQUEST.fullmatch(normalized)
        or _PHONE_CONNECT_REQUEST.fullmatch(normalized)
        or _PHONE_PERMISSION_REQUEST.fullmatch(normalized)
        or _PHONE_CALLS_REQUEST.fullmatch(normalized)
    )


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
    if lowered in {"/link_phone", "link phone"} or is_phone_link_request(stripped):
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
    "is_phone_link_request",
]
