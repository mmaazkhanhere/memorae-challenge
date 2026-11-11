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
    r"(?:(?:and\s+)?also\s+)?"
    r"remind\s+me|"
    r"set\s+(?:a\s+)?reminder|"
    r"remember\s+to|"
    r"save\s+(?:a\s+)?reminder\s+to"
    r")(?:\s+|$)",
    re.IGNORECASE,
)
_DISAMBIGUATED_REMINDER_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:(?:and\s+)?also\s+)?(?:remember\s+to|save\s+(?:a\s+)?reminder\s+to)(?:\s+|$)",
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
    r"remember\s+to\b.*\b(?:keep|store|save|remember)\b.*\bin\s+mind|"
    r"(?:make|take)\s+(?:a\s+)?note\s+to\b"
    r")",
    re.IGNORECASE,
)
_SAVE_AS_REMINDER = re.compile(
    r"^\s*(?:please\s+)?save\s+(?:this|that)\s+as\s+(?:a\s+)?reminder\s+to\b",
    re.IGNORECASE,
)
_EMBEDDED_REMINDER_REQUEST = re.compile(
    r"^\s*(?:please\s+)?remember\b.*\b(?:need|want)\s+(?:a\s+)?reminder\b",
    re.IGNORECASE,
)
_UNRESOLVED_MEMORY_REFERENCE = re.compile(
    r"^\s*(?:please\s+)?(?:remember|save)\s+this\b"
    r"(?!\s+fact\s*[:,-])|"
    r"^\s*(?:please\s+)?remember\s+that\s+this\b|"
    r"^\s*(?:please\s+)?(?:make|take)\s+(?:a\s+)?note"
    r"(?:\s+(?:that|of|about))?\s+this\b(?!\s+fact\s*[:,-])",
    re.IGNORECASE,
)
_CONTEXTUAL_OR_NEGATED = re.compile(
    r"^\s*(?:do\s+not|don't|dont|never)\s+(?:remember|save|remind|make|take)\b|"
    r"^\s*(?:please\s+)?(?:remember|save|(?:make|take)\s+(?:a\s+)?note)\s+"
    r"(?:the\s+above|above|previous|prior|this|that)\s*[.!?]*\s*$|"
    r"^\s*(?:please\s+)?(?:also\s+)?remind\s+me\s+(?:about\s+)?"
    r"(?:it|that|this|the\s+above|above|previous|prior)\b",
    re.IGNORECASE,
)
_MIXED_CREATION_COMMAND = re.compile(
    r"\b(?:remind\s+me|set\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder)\b.*"
    r"\b(?:remember|save|note)\b|"
    r"\b(?:remember|save|note)\b.*"
    r"\b(?:remind\s+me|set\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder)\b",
    re.IGNORECASE,
)


def explicit_create_route(text: str) -> ExplicitCreateRoute:
    """Return the high-level creation shape for shared extraction callers."""

    # The operation is explicit in this form; the task after "to" is still
    # parsed and validated by the normal reminder interpreter.
    if _SAVE_AS_REMINDER.match(text):
        return ExplicitCreateRoute.NEW_REMINDER
    # Mixed operations and embedded reminder requests are not a single
    # deterministic operation.  Referential/incomplete memory wording is
    # retained as SAVE_MEMORY for the existing extractor contract; callers
    # that need a safe route should use is_unsafe_creation_command().
    if _MIXED_CREATION_COMMAND.search(text) or _EMBEDDED_REMINDER_REQUEST.search(text):
        return ExplicitCreateRoute.MODEL_FALLBACK
    if _MEMORY_REMINDER_COLLISION.match(text):
        return ExplicitCreateRoute.MODEL_FALLBACK
    if REMINDER_COMMAND_PREFIX.match(text):
        return ExplicitCreateRoute.NEW_REMINDER
    if _MEMORY_FACT_COMMAND.match(text):
        return ExplicitCreateRoute.SAVE_MEMORY
    return ExplicitCreateRoute.NONE


def is_unsafe_creation_command(text: str) -> bool:
    """Return whether creation wording needs semantic/model interpretation."""

    if _SAVE_AS_REMINDER.match(text):
        return False
    return bool(
        _CONTEXTUAL_OR_NEGATED.search(text)
        or _UNRESOLVED_MEMORY_REFERENCE.search(text)
        or _MIXED_CREATION_COMMAND.search(text)
        or _EMBEDDED_REMINDER_REQUEST.search(text)
    )


def is_memory_reminder_collision(text: str) -> bool:
    """Return whether explicit text contains incompatible creation cues."""

    return explicit_create_route(text) is ExplicitCreateRoute.MODEL_FALLBACK


def is_disambiguated_reminder_command(text: str) -> bool:
    """Return whether text clearly chose reminder creation over memory saving."""

    return _DISAMBIGUATED_REMINDER_PREFIX.match(text) is not None
