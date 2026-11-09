"""Small normalization helpers shared by deterministic intent paths."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Literal

Meridiem = Literal["am", "pm"]
CalendarFrequency = Literal["MONTHLY", "YEARLY"]
RecurrenceFrequency = Literal["WEEKLY", "MONTHLY", "YEARLY"]
SummaryMetric = Literal[
    "PENDING_COUNT",
    "TODAY_REMAINING",
    "COMPLETED_COUNT",
    "TODAY_COMPLETED",
    "OVERDUE_COUNT",
]

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}


def normalise_now(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        from datetime import UTC

        return now.replace(tzinfo=UTC)
    from datetime import UTC

    return now.astimezone(UTC)


def task_from_remainder(remainder: str) -> str:
    task = remainder.strip(" .,:;")
    task = re.sub(
        r"^(?:(?:on|to|for|that|about)\s+)+|^remind\s+me(?:\s+(?:about|to))?\s+",
        "",
        task,
        flags=re.IGNORECASE,
    )
    return " ".join(task.split())


def number(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return NUMBER_WORDS.get(value.lower())


def ordinal_suffix(value: int) -> str:
    if 10 < value % 100 < 14:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def normalise_meridiem(value: str | None) -> str | None:
    return value.lower().replace(".", "") if value else None


def valid_clock(hour: int | None, minute: int, meridiem: str | None) -> bool:
    if hour is None or not 0 <= minute <= 59:
        return False
    if meridiem is not None:
        return 1 <= hour <= 12
    return 0 <= hour <= 23


def clock_to_24(hour: int, meridiem: str) -> int:
    if meridiem == "pm" and hour != 12:
        return hour + 12
    if meridiem == "am" and hour == 12:
        return 0
    return hour


def clock_label(hour: int, minute: int) -> str:
    return f"{hour:g}" if minute == 0 else f"{hour}:{minute:02d}"


def date_label(value: date, local_today: date) -> str:
    if value == local_today:
        return "today"
    if value == local_today + timedelta(days=1):
        return "tomorrow"
    return value.strftime("%A")
