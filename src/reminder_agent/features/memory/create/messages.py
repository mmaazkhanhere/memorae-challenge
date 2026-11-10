"""Personalized user-facing messages for memory creation."""

from __future__ import annotations

import random

MEMORY_CREATION_MESSAGES: dict[str, list[str]] = {
    "memory_clarification": [
        "Of course! What would you like me to remember?",
        "Sure! What should I keep in mind for you?",
        "Got it! What information would you like me to save?",
        "I'm listening! What would you like me to remember?",
    ],
    "memory_clarification_expired": [
        "Looks like we didn't finish saving that memory. Could you tell me what you'd like me to remember again?",
        "We didn't quite finish that memory request. Mind telling me what you'd like me to save again?",
        "That memory request timed out. No worries! Just tell me what you'd like me to remember.",
    ],
    "memory_canceled": [
        "No problem! I haven't saved anything.",
        "Got it! I won't save that information.",
        "All good! I've left that memory unsaved.",
        "Of course! Nothing has been saved.",
    ],
    "memory_confirmation": [
        "Got it! I'll remember that {fact}.",
        "Of course! I'll keep in mind that {fact}.",
        "Noted! I'll remember that {fact}.",
        "I've got it! I'll keep that in mind for you: {fact}.",
    ],
    "memory_confirmation_generic": [
        "Got it! I've saved that for you.",
        "Noted! I'll keep that in mind.",
        "All set! I've saved that information.",
        "Of course! I'll remember that.",
    ],
    "memory_idempotency_error": [
        "I couldn't process that just now. Could you try again?",
        "Something went wrong while processing that. Please try sending it again.",
    ],
    "memory_authorization_failed": [
        "I couldn't verify your account, so I haven't saved anything.",
        "I wasn't able to confirm your identity, so I couldn't save that memory.",
    ],
    "memory_save_failed": [
        "I couldn't save that just now. Could you try again?",
        "Something went wrong while saving that information. Mind trying once more?",
        "I wasn't able to remember that this time. Please try sending it again.",
    ],
}


def memory_message(key: str, **values: object) -> str:
    """Choose and render one personalized memory-creation message."""

    try:
        template = random.choice(MEMORY_CREATION_MESSAGES[key])
    except KeyError as exc:  # pragma: no cover - programmer error
        raise ValueError(f"unknown memory creation message: {key}") from exc
    return template.format(**values)
