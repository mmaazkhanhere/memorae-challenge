"""User-facing text for memory saving and retrieval."""

from __future__ import annotations

import random

MEMORY_SAVE_MESSAGES: dict[str, list[str]] = {
    "clarification": [
        "What information should I remember?",
    ],
    "clarification_expired": [
        "That memory request expired. Please tell me what to remember.",
    ],
    "canceled": [
        "Okay, I won't save anything.",
    ],
    "confirmation": [
        "I saved that information.",
    ],
    "confirmation_generic": [
        "I saved that information.",
    ],
    "idempotency_error": [
        "I couldn't process that just now. Could you try again?",
        "Something went wrong while processing that. Please try sending it again.",
    ],
    "authorization_failed": [
        "I couldn't verify your account, so I haven't saved anything.",
        "I wasn't able to confirm your identity, so I couldn't save that memory.",
    ],
    "save_failed": [
        "I couldn't save that information. Please try again.",
    ],
}


def save_memory_message(key: str, **values: object) -> str:
    """Choose and render one personalized memory-save message."""

    try:
        template = random.choice(MEMORY_SAVE_MESSAGES[key])
    except KeyError as exc:  # pragma: no cover - programmer error
        raise ValueError(f"unknown memory-save message: {key}") from exc
    return template.format(**values)


MEMORY_LOOKUP_MESSAGES: dict[str, list[str]] = {
    "missing_subject": [
        "Of course! What would you like me to look up?",
        "Sure! What would you like me to recall?",
        "What would you like to know from the things I've saved for you?",
        "Happy to help! What information are you looking for?",
    ],
    "authorization_failed": [
        "I couldn't verify your account, so I can't access your saved information right now.",
        "I wasn't able to confirm your identity, so I couldn't check your memories.",
    ],
    "fact_found": [
        'You told me: "{fact_clause}"',
    ],
    "ambiguous_matches": [
        "I found more than one saved fact that could match. Please be more specific.",
    ],
    "unknown_fact": [
        "I don't know that yet.",
    ],
    "retrieval_failed": [
        "I'm having trouble checking your saved memories right now. Could you try again?",
        "I couldn't pull up that information just now. Mind trying again in a moment?",
        "Something went wrong while looking that up. Please try again.",
    ],
}


def lookup_memory_message(key: str, *, fact: str | None = None) -> str:
    """Select and render one memory-lookup message."""

    try:
        template = random.choice(MEMORY_LOOKUP_MESSAGES[key])
    except KeyError as exc:  # pragma: no cover - programmer error
        raise ValueError(f"unknown memory-lookup message: {key}") from exc
    if key == "fact_found":
        if fact is None:
            raise ValueError("fact_found messages require fact content")
        fact_clause = fact.rstrip(".!?")
        fact_sentence = fact if fact.endswith((".", "!", "?")) else f"{fact}."
        return template.format(fact_clause=fact_clause, fact_sentence=fact_sentence)
    return template
