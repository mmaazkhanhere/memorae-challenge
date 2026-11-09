"""Personalized user-facing messages for reminder snoozing."""

from __future__ import annotations

from random import choice

REMINDER_SNOOZE_MESSAGES: dict[str, list[str]] = {
    "snooze_success": [
        "No problem! I'll remind you to {task} again {when} on Telegram.",
        "Got it! I'll give you another heads-up to {task} {when} on Telegram.",
        "Of course! I've snoozed your '{task}' reminder until {new_datetime}.",
        "All set! I'll check back with a reminder to {task} {when} on Telegram.",
        "Sure thing! Your '{task}' reminder is snoozed until {new_datetime}.",
    ],
    "missing_duration": [
        "Of course! How much later should I remind you?",
        "Sure! How long would you like to snooze it?",
        "No problem! Should I remind you again in 10 minutes, an hour, or sometime else?",
        "Got it! When would you like me to remind you again?",
    ],
    "missing_target": [
        "Sure! Which reminder would you like me to snooze?",
        "Of course! What reminder should I push back?",
        "No problem! Tell me which reminder you'd like to snooze.",
    ],
    "no_delivered_match": [
        "I couldn't find a recent reminder matching that. Could you tell me which one you mean?",
        "Hmm, I couldn't find that reminder among the ones I've sent you. Could you check the task?",
        "I don't see a matching reminder available to snooze. Mind giving me a little more detail?",
    ],
    "snooze_cancelled": [
        "No problem! I'll leave that reminder as it is.",
        "Got it! I won't snooze anything.",
        "All good! Your reminder will stay as it was.",
        "Of course! Nothing has been snoozed.",
    ],
    "invalid_duration": [
        "That snooze time doesn't look right. Try something like '10 minutes' or '1 hour'.",
        "I need a time in the future to snooze that reminder. How about 15 minutes or an hour?",
        "Could you give me a positive snooze duration, like 5 minutes or 30 minutes?",
    ],
    "reminder_not_found": [
        "I couldn't find that reminder among the ones available to snooze.",
        "Looks like I can't find a matching reminder to snooze right now.",
    ],
    "reminder_unavailable": [
        "Looks like that reminder can't be snoozed anymore.",
        "That reminder is no longer available to snooze. I haven't changed anything.",
        "I couldn't snooze that one because it's no longer available.",
    ],
    "wrong_telegram_chat": [
        "I can't snooze that reminder from this chat because it wasn't delivered here.",
        "That reminder wasn't sent to this Telegram chat, so I can't snooze it here.",
    ],
    "authorization_failed": [
        "I couldn't verify your Telegram account, so I haven't changed anything.",
        "I wasn't able to confirm your Telegram identity, so I couldn't snooze that reminder.",
    ],
}


def snooze_message(key: str, **values: object) -> str:
    """Select and format one snooze message variant."""

    try:
        template = choice(REMINDER_SNOOZE_MESSAGES[key])
    except KeyError as exc:
        raise ValueError(f"unknown snooze message key: {key}") from exc
    return template.format(**values)


__all__ = ["REMINDER_SNOOZE_MESSAGES", "snooze_message"]
