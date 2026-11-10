"""Personalized user-facing messages for reminder rescheduling."""

from __future__ import annotations

import random

REMINDER_RESCHEDULE_MESSAGES: dict[str, list[str]] = {
    "one_time_rescheduled": [
        "Moved '{task}' to {when} via Telegram.",
    ],
    "recurring_series_rescheduled": [
        "All set! I've updated all future '{task}' reminders to {cadence} at {time} {timezone} on Telegram. Your next reminder is {next_occurrence}.",
        "Got it! From now on, I'll remind you to {task} {cadence} at {time} {timezone} on Telegram. The next one's {next_occurrence}.",
        "Done! Your future '{task}' reminders will now follow {cadence} at {time} {timezone} on Telegram. Next up: {next_occurrence}.",
        "Consider it updated! All upcoming '{task}' reminders are now set for {cadence} at {time} {timezone} on Telegram. Your next one is {next_occurrence}.",
    ],
    "reschedule_cancelled": [
        "No problem! I've left that reminder right where it was.",
        "Got it! Your reminder will stay as scheduled.",
        "Of course! I haven't changed anything.",
        "All good! We'll keep the original reminder time.",
    ],
    "multiple_matching_reminders": [
        "I found a few reminders that match. Which one would you like to move?",
        "Looks like you have more than one reminder for that. Which one did you mean?",
        "I've found a couple of possibilities. Which reminder should I reschedule?",
    ],
    "missing_reminder_target": [
        "Of course! Which reminder would you like to reschedule?",
        "Sure thing! What reminder should I move?",
        "Happy to help! Which reminder did you have in mind?",
    ],
    "ambiguous_pending_selection": [
        "Just to make sure I get the right one, which reminder did you mean?",
        "I want to make sure I'm moving the right reminder. Which one should I update?",
        "Could you tell me which reminder you'd like to move?",
    ],
    "recurring_scope_required": [
        "Would you like me to move just the next reminder, or update all future reminders in the series?",
        "Should I change only the upcoming '{task}' reminder, or the entire series going forward?",
        "Just checking - is this change for the next '{task}' reminder, or all future ones?",
    ],
    "no_matching_reminder": [
        "I couldn't find an active reminder matching that. Could you try describing it another way?",
        "Hmm, I couldn't find that reminder on your active list. Want to try a different description?",
        "I don't see an active reminder matching that. Could you give me a little more detail?",
    ],
    "reminder_inactive": [
        "Looks like that reminder is no longer active, so I can't move it.",
        "That reminder isn't active anymore. I haven't made any changes.",
        "I couldn't reschedule that one because it's no longer active.",
    ],
    "authorization_failed": [
        "I couldn't verify your Telegram account, so I haven't changed anything. Please try again.",
        "I wasn't able to confirm your Telegram identity, so I couldn't update the reminder.",
    ],
    "reschedule_failed": [
        "I couldn't reschedule that reminder because {reason}. Nothing was changed.",
        "Something prevented me from moving that reminder: {reason}. Your original schedule is unchanged.",
        "I wasn't able to update that reminder. Here's why: {reason}.",
    ],
}


def choose_reschedule_message(key: str, **values: object) -> str:
    """Choose and render one personalized reschedule message."""

    return random.choice(REMINDER_RESCHEDULE_MESSAGES[key]).format(**values)


def reschedule_message(key: str) -> str:
    """Choose a reschedule message without template values."""

    return choose_reschedule_message(key)
