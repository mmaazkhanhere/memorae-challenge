"""Safe, conversational responses for non-operational user messages."""

from __future__ import annotations

import random


GENERAL_MESSAGES: dict[str, list[str]] = {
    "unrelated": [
        (
            "That's outside what I can help with right now. "
            "I'm here to help you manage reminders "
            "and remember important details."
        ),
        (
            "I can't help with that one, but I can "
            "set reminders, manage them, and keep "
            "information saved for you."
        ),
        (
            "That's a little beyond what I do. "
            "I mainly help with reminders and "
            "things you'd like me to remember."
        ),
        (
            "I'm focused on helping you remember "
            "things and stay on top of reminders, "
            "so I can't help with that request."
        ),
        (
            "I can't take care of that particular request. "
            "What I can do is help you set reminders "
            "or remember information for later."
        ),
        (
            "That's not something I support at the moment. "
            "I can help you create and manage reminders "
            "or save details you'd like to remember."
        ),
    ],
    "greeting": [
        "Hey! What can I help you remember today?",
        "Hi there! Got something you'd like me to keep track of?",
        "Hey! I'm here. What's on your mind?",
        "Hello! Need a reminder or want me to remember something?",
    ],
    "thanks": [
        "You're welcome!",
        "Anytime!",
        "Happy to help!",
        "Of course!",
    ],
    "capabilities": [
        (
            "I can help you set and manage reminders, "
            "including recurring ones, and remember "
            "important information for you."
        ),
        (
            "Think of me as your extra memory. "
            "I help you remember details and "
            "keep track of reminders."
        ),
        (
            "I'm here to help you remember things. "
            "You can ask me to set reminders, "
            "manage existing ones, or save information."
        ),
    ],
}


def general_message(category: str) -> str:
    """Choose a conversational response without invoking the model."""

    return random.choice(GENERAL_MESSAGES[category])
