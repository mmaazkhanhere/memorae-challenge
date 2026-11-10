"""Personalized user-facing messages for memory retrieval."""

from __future__ import annotations

import random

MEMORY_RETRIEVAL_MESSAGES: dict[str, list[str]] = {
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
        "I remember! You mentioned that {fact_clause}.",
        "You told me that {fact_clause}.",
        "I've got that saved! {fact_sentence}",
        "From what you've shared with me, {fact_clause}.",
    ],
    "ambiguous_matches": [
        "I found a couple of things that might match. Which one did you mean?",
        "I've got a few related things saved. Could you be a little more specific?",
        "I remember a few things about that. Which one are you looking for?",
        "There are a couple of possibilities in your saved information. Mind narrowing it down?",
    ],
    "unknown_fact": [
        "I don't think you've told me that yet. Want me to remember it?",
        "I don't have that saved yet. Would you like to tell me?",
        "That's not something I have in my memory right now.",
        "I couldn't find that in what you've shared with me so far.",
    ],
    "retrieval_failed": [
        "I'm having trouble checking your saved memories right now. Could you try again?",
        "I couldn't pull up that information just now. Mind trying again in a moment?",
        "Something went wrong while looking that up. Please try again.",
    ],
}


def retrieval_message(key: str, *, fact: str | None = None) -> str:
    """Select and render one retrieval message for the requested outcome."""

    try:
        template = random.choice(MEMORY_RETRIEVAL_MESSAGES[key])
    except KeyError as exc:  # pragma: no cover - programmer error
        raise ValueError(f"unknown memory retrieval message: {key}") from exc
    if key == "fact_found":
        if fact is None:
            raise ValueError("fact_found messages require fact content")
        fact_clause = fact.rstrip(".!?")
        fact_sentence = fact if fact.endswith((".", "!", "?")) else f"{fact}."
        return template.format(fact_clause=fact_clause, fact_sentence=fact_sentence)
    return template


__all__ = ["MEMORY_RETRIEVAL_MESSAGES", "retrieval_message"]
