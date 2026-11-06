"""High-precision lexical routing for overlapping creation commands.

These rules intentionally identify only commands whose operation is explicit.
Everything else, including mixed reminder/memory language, remains eligible for
the classifier fallback in the conversation router.
"""

from __future__ import annotations

import re
from enum import StrEnum


class ExplicitCreateRoute(StrEnum):
    NEW_REMINDER = "NEW_REMINDER"
    SAVE_MEMORY = "SAVE_MEMORY"
    MODEL_FALLBACK = "MODEL_FALLBACK"
    NONE = "NONE"


REMINDER_COMMAND_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"remind\s+me|"
    r"set\s+(?:a\s+)?reminder|"
    r"remember\s+to|"
    r"save\s+(?:a\s+)?reminder\s+to"
    r")(?:\s+|$)",
    re.IGNORECASE,
)
_DISAMBIGUATED_REMINDER_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:remember\s+to|save\s+(?:a\s+)?reminder\s+to)(?:\s+|$)",
    re.IGNORECASE,
)
_MEMORY_FACT_COMMAND = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"remember\s*(?:$|(?:that\s+)?(?:my|the|our|your|(?:[\w-]+\s+)*[\w-]+['']s)\b|(?:a\s+)?fact\b|(?:this|that|something|information)\b)|"
    r"save\s+(?:(?:that\s+)?(?:my|the|our|your|(?:[\w-]+\s+)*[\w-]+['']s)\b|(?:a\s+)?fact\b|(?:this|that|something|information)\b)|"
    r"(?:make|take)\s+(?:a\s+)?note(?:\s+(?:that|of|about)\b)?"
    r")",
    re.IGNORECASE,
)
_MEMORY_REMINDER_COLLISION = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"remember\s+to\b.*\b(?:keep|store|save|remember)\b.*\bin\s+mind\b|"
    r"(?:make|take)\s+(?:a\s+)?note\s+to\b"
    r")",
    re.IGNORECASE,
)


def explicit_create_route(text: str) -> ExplicitCreateRoute:
    """Return a route only when lexical evidence selects one operation safely."""

    if _MEMORY_REMINDER_COLLISION.match(text):
        return ExplicitCreateRoute.MODEL_FALLBACK
    if REMINDER_COMMAND_PREFIX.match(text):
        return ExplicitCreateRoute.NEW_REMINDER
    if _MEMORY_FACT_COMMAND.match(text):
        return ExplicitCreateRoute.SAVE_MEMORY
    return ExplicitCreateRoute.NONE


def is_memory_reminder_collision(text: str) -> bool:
    """Return whether explicit text contains incompatible creation cues."""

    return explicit_create_route(text) is ExplicitCreateRoute.MODEL_FALLBACK


def is_disambiguated_reminder_command(text: str) -> bool:
    """Return whether text clearly chose reminder creation over memory saving."""

    return _DISAMBIGUATED_REMINDER_PREFIX.match(text) is not None
