"""Confirmation formatting and response staging for reminder creation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from random import choice
from typing import TypeVar
from zoneinfo import ZoneInfo

from reminder_agent.features.intent.contracts import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
)

ResponseT = TypeVar("ResponseT")

REMINDER_MESSAGES: dict[str, list[str]] = {
    "one_time_created": [
        "Got it! I'll remind you to {task} {when} on Telegram.",
        "Of course! I'll remind you to {task} {when} on Telegram, so you'll get a heads-up.",
        "You got it! I'll remind you to {task} {when} right here on Telegram.",
        "Consider it taken care of! I'll remind you to {task} {when} on Telegram.",
    ],
    "recurring_created": [
        "Got it! I'll remind you to {task} {cadence} at {time} {timezone} on Telegram. Your next reminder is {next_occurrence}.",
        "Of course! I'll give you a heads-up to {task} {cadence} at {time} {timezone} on Telegram. We'll start {next_occurrence}.",
        "You're all set! I'll remind you to {task} {cadence} at {time} {timezone} on Telegram. The next one's {next_occurrence}.",
        "Consider it handled! I'll send you a reminder to {task} {cadence} at {time} {timezone} on Telegram. First up: {next_occurrence}.",
    ],
    "duplicate_reminder": [
        "You're already covered! I've got a reminder to {task} scheduled for {when}, so I didn't add another one.",
        "Looks like we already have that one down! Your reminder to {task} is set for {when}. I've left it as it is.",
        "No need to set that up twice! There's already a reminder to {task} for {when}, so I kept the existing one.",
    ],
}

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def ordinal_suffix(day: int) -> str:
    """Return the existing English ordinal suffix used in confirmations."""

    if 10 < day % 100 < 14:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def recurring_cadence(intent: CreateRecurringReminderIntent) -> str:
    """Render the recurring cadence wording used by reminder confirmations."""

    if intent.frequency == "WEEKLY":
        if intent.interval == 1 and intent.weekdays == tuple(range(7)):
            return "every day"
        if intent.interval == 1 and intent.weekdays == (0, 1, 2, 3, 4):
            return "every weekday"
        days = [_WEEKDAY_NAMES[day] for day in intent.weekdays]
        if len(days) == 1:
            day_label = days[0]
        elif len(days) == 2:
            day_label = " and ".join(days)
        else:
            day_label = ", ".join(days[:-1]) + ", and " + days[-1]
        if intent.interval == 1:
            return f"every {day_label}"
        interval = _NUMBER_WORDS.get(intent.interval, str(intent.interval))
        return f"every {interval} weeks on {day_label}"
    if intent.frequency == "MONTHLY":
        ordinal_day = f"{intent.day_of_month}{ordinal_suffix(intent.day_of_month or 0)}"
        interval = "every month" if intent.interval == 1 else f"every {_NUMBER_WORDS.get(intent.interval, str(intent.interval))} months"
        return f"on the {ordinal_day} of {interval}"
    month_name = datetime(2000, intent.month or 1, 1).strftime("%B")
    interval = "every year" if intent.interval == 1 else f"every {_NUMBER_WORDS.get(intent.interval, str(intent.interval))} years"
    return f"{interval} on {month_name} {intent.day_of_month}"


def _human_time(value: datetime) -> str:
    return value.strftime("%I:%M %p").lstrip("0")


def _timezone_label(value: datetime, timezone_name: str) -> str:
    return value.replace(tzinfo=ZoneInfo(timezone_name)).tzname() or timezone_name


def format_next_occurrence(value: datetime) -> str:
    return f"{value.strftime('%A, %B')} {value.day}, at {_human_time(value)} {value.tzname()}"


def format_when(value: datetime, *, reference: datetime) -> str:
    if value.date() == reference.date():
        prefix = "today"
    elif (value.date() - reference.date()).days == 1:
        prefix = f"tomorrow, {value.strftime('%B')} {value.day},"
    else:
        prefix = f"on {value.strftime('%A, %B')} {value.day},"
    return f"{prefix} at {_human_time(value)} {value.tzname()}"


def format_reminder_confirmation(
    intent: CreateOneTimeReminderIntent | CreateRecurringReminderIntent,
    *,
    local_display: str,
    timezone_name: str,
) -> str:
    """Render the exact reminder confirmation text."""

    if isinstance(intent, CreateRecurringReminderIntent):
        return choice(REMINDER_MESSAGES["recurring_created"]).format(
            task=intent.task,
            cadence=recurring_cadence(intent),
            time=_human_time(intent.start_local),
            timezone=_timezone_label(intent.start_local, timezone_name),
            next_occurrence=local_display,
        )
    return choice(REMINDER_MESSAGES["one_time_created"]).format(task=intent.task, when=local_display)


def format_duplicate_reminder(task: str, *, when: str) -> str:
    return choice(REMINDER_MESSAGES["duplicate_reminder"]).format(task=task, when=when)


def stage_creation_response[ResponseT](
    *,
    confirmations: Sequence[str],
    stage_response: Callable[..., ResponseT],
) -> ResponseT:
    """Stage the joined creation confirmation in the caller's transaction."""

    return stage_response(
        text="\n".join(confirmations),
        response_kind="confirmation",
    )
