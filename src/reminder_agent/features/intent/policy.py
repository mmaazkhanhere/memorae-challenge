"""Application boundary for supported and untrusted user requests.

The intent model is a parser, not an authorization mechanism. This module
keeps the small set of deterministic policy decisions that must happen before
model interpretation in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from reminder_agent.features.intent.messages import GENERAL_MESSAGES

# Keep the policy-level response stable for hard rejections that happen before
# the conversational router has loaded context. Routed unrelated messages use
# the broader conversational set in ``messages.py``.
OUT_OF_SCOPE_RESPONSE = GENERAL_MESSAGES["unrelated"][0]


class RequestDisposition(StrEnum):
    ALLOW = "ALLOW"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class RequestAssessment:
    disposition: RequestDisposition
    reason: str

    @property
    def is_rejected(self) -> bool:
        return self.disposition is RequestDisposition.OUT_OF_SCOPE


# These patterns are intentionally narrow. They identify attempts to change
# the instruction hierarchy or supply authority-like identifiers, not ordinary
# reminder text. The backend still owns identity and authorization even when
# the text does not match one of these patterns.
_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+)?"
        r"(?:previous|prior|system|developer|assistant)?\s*(?:instructions?|rules?|policies?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:system|developer)\s+(?:message|prompt|instructions?)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:act|pretend)\s+as\s+(?:the\s+)?(?:system|developer|assistant)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:user_id|chat_id|identity_id)\s*[:=]", re.IGNORECASE),
)

_OUT_OF_SCOPE_PATTERNS = (
    re.compile(
        r"\b(?:weather|forecast|news|joke|recipe|translate|translation|calculate|calculator|"
        r"math|stock price|sports score|write code|debug code)\b",
        re.IGNORECASE,
    ),
)


def assess_request(text: str, *, has_pending: bool = False) -> RequestAssessment:
    """Classify policy-sensitive input before intent interpretation.

    Free-form reminder language remains allowed so the existing deterministic
    parser and conservative provider fallback can handle it. Empty input is
    represented as a clarification rather than an out-of-scope rejection.
    """

    normalized = " ".join(text.split())
    if not normalized:
        return RequestAssessment(RequestDisposition.NEEDS_CLARIFICATION, "empty_request")

    if any(pattern.search(normalized) for pattern in _PROMPT_INJECTION_PATTERNS):
        return RequestAssessment(RequestDisposition.OUT_OF_SCOPE, "prompt_injection")

    if any(pattern.search(normalized) for pattern in _OUT_OF_SCOPE_PATTERNS):
        return RequestAssessment(RequestDisposition.OUT_OF_SCOPE, "unsupported_capability")

    # A message such as "remind me" is incomplete but plausibly within scope;
    # the interpreter should ask for the missing details rather than reject it.
    return RequestAssessment(RequestDisposition.ALLOW, "within_scope_or_requires_clarification")


def is_supported_operation(operation: object) -> bool:
    """Return whether an application service may execute a parsed operation."""

    # This boundary receives data from model output. Require a plain, canonical
    # string rather than allowing enum subclasses, containers, or other values
    # with surprising equality/hash behavior.
    return type(operation) is str and operation in {
        "CREATE_ONE_TIME_REMINDER",
        "CREATE_RECURRING_REMINDER",
        "LIST_REMINDERS",
        "SUMMARY",
        "RESCHEDULE_REMINDER",
        "CANCEL_REMINDER",
        "COMPLETE_REMINDER",
        "SNOOZE_OCCURRENCE",
        "SAVE_MEMORY",
    }
