"""Personalized response variants for reminder completion outcomes."""

from __future__ import annotations

import random

REMINDER_COMPLETION_MESSAGES: dict[str, list[str]] = {
    "reminder_completed": [
        "All done! I've marked '{task}' as complete.",
        "Got it! '{task}' is checked off your list.",
        "That's taken care of! I've marked '{task}' as done.",
        "Done and dusted! '{task}' is now marked as complete.",
    ],
    "reminder_already_completed": [
        "You're already ahead of me! That one's marked as done.",
        "Looks like you've already checked that one off!",
        "That one's already taken care of. No changes needed.",
        "Already done! I've left it just as it is.",
    ],
    "reminder_not_available": [
        "I couldn't mark that one as done. It may have been deleted or is no longer active.",
        "Looks like that reminder isn't available to complete anymore.",
        "I couldn't find an active reminder to mark as done. It might have been removed or completed already.",
    ],
    "completed_total": [
        "Nice work! You've completed {count} {task_word} so far.",
        "You've checked off {count} {task_word} in total!",
        "That's {count} {task_word} completed. Keep it going!",
        "You've taken care of {count} {task_word} so far.",
    ],
    "completed_today": [
        "You've checked off {count} {task_word} today. Nice going!",
        "Look at that! {count} {task_word} completed today.",
        "You've already taken care of {count} {task_word} today!",
        "That's {count} {task_word} off your list today. Well done!",
    ],
}


def _choose(key: str, **values: object) -> str:
    return random.choice(REMINDER_COMPLETION_MESSAGES[key]).format(**values)


def reminder_completed_message(task: str) -> str:
    return _choose("reminder_completed", task=task)


def reminder_already_completed_message() -> str:
    return _choose("reminder_already_completed")


def reminder_not_available_message() -> str:
    return _choose("reminder_not_available")


def completed_total_message(count: int) -> str:
    return _choose(
        "completed_total",
        count=count,
        task_word="task" if count == 1 else "tasks",
    )


def completed_today_message(count: int) -> str:
    return _choose(
        "completed_today",
        count=count,
        task_word="task" if count == 1 else "tasks",
    )
