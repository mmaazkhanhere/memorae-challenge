"""Timezone-aware date and recurrence calculations for deterministic intent."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from reminder_agent.features.intent.contracts import PendingReminder

Meridiem = Literal["am", "pm"]
CalendarFrequency = Literal["MONTHLY", "YEARLY"]
RecurrenceFrequency = Literal["WEEKLY", "MONTHLY", "YEARLY"]
SummaryMetric = Literal[
    "PENDING_COUNT", "TODAY_REMAINING", "COMPLETED_COUNT", "TODAY_COMPLETED", "OVERDUE_COUNT"
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
    from datetime import UTC

    return (
        now.replace(tzinfo=UTC)
        if now.tzinfo is None or now.utcoffset() is None
        else now.astimezone(UTC)
    )


def task_from_remainder(remainder: str) -> str:
    task = re.sub(
        r"^(?:(?:on|to|for|that|about)\s+)+|^remind\s+me(?:\s+(?:about|to))?\s+",
        "",
        remainder.strip(" .,:;"),
        flags=re.IGNORECASE,
    )
    return " ".join(task.split())


def strip_repeated_weekly_cadence(task: str, raw_days: str) -> str:
    """Remove a trailing copy of the weekly cadence already parsed from input."""

    cadence = re.escape(" ".join(raw_days.split()))
    return re.sub(
        rf"\s+(?:every|each)\s+{cadence}\s*$",
        "",
        task,
        flags=re.IGNORECASE,
    ).strip()


def strip_repeated_calendar_cadence(task: str, frequency: CalendarFrequency) -> str:
    """Remove a trailing copy of a parsed monthly/yearly cadence."""

    cadence = "month" if frequency == "MONTHLY" else "year"
    return re.sub(
        rf"\s+(?:(?:every|each)\s+{cadence}|{cadence}ly)\s*$",
        "",
        task,
        flags=re.IGNORECASE,
    ).strip()


def strip_repeated_temporal_token(task: str, token: str) -> str:
    """Remove a trailing copy of a date token already parsed from the request."""

    return re.sub(
        rf"\s+(?:on\s+)?{re.escape(token)}\s*$",
        "",
        task,
        flags=re.IGNORECASE,
    ).strip()


def number(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return NUMBER_WORDS.get(value.lower())


def ordinal_suffix(value: int) -> str:
    return "th" if 10 < value % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def normalise_meridiem(value: str | None) -> str | None:
    return value.lower().replace(".", "") if value else None


def meridiem_from_day_part(value: str | None) -> Meridiem | None:
    """Map an explicit natural-language day part to a safe 12-hour half."""

    if not value:
        return None
    normalized = value.casefold().strip()
    if normalized in {"morning"}:
        return "am"
    if normalized in {"afternoon", "evening", "night"}:
        return "pm"
    return None


def valid_clock(hour: int | None, minute: int, meridiem: str | None) -> bool:
    return (
        hour is not None
        and 0 <= minute <= 59
        and (1 <= hour <= 12 if meridiem is not None else 0 <= hour <= 23)
    )


def clock_to_24(hour: int, meridiem: str) -> int:
    return (
        hour + 12
        if meridiem == "pm" and hour != 12
        else 0
        if meridiem == "am" and hour == 12
        else hour
    )


def clock_label(hour: int, minute: int) -> str:
    return f"{hour:g}" if minute == 0 else f"{hour}:{minute:02d}"


def date_label(value: date, local_today: date) -> str:
    if value == local_today:
        return "today"
    if value == local_today + timedelta(days=1):
        return "tomorrow"
    return value.strftime("%A")


WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def next_recurring_start_local(
    *,
    frequency: RecurrenceFrequency,
    interval: int,
    weekdays: tuple[int, ...],
    day_of_month: int | None,
    month: int | None,
    hour: int,
    minute: int,
    timezone_name: str,
    now: datetime,
) -> datetime:
    """Derive the next local occurrence for a fully specified recurrence."""

    local_now = normalise_now(now).astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
    if frequency == "WEEKLY":
        for offset in range(8):
            weekly_candidate_date = local_now.date() + timedelta(days=offset)
            if weekly_candidate_date.weekday() not in weekdays:
                continue
            candidate = datetime.combine(weekly_candidate_date, time(hour, minute))
            if candidate > local_now:
                return candidate
        raise ValueError("could not find weekly recurrence start")

    if day_of_month is None:
        raise ValueError("calendar recurrence requires day_of_month")
    candidate_year = local_now.year
    candidate_month = month or local_now.month
    while True:
        try:
            candidate_date = date(candidate_year, candidate_month, day_of_month)
        except ValueError:
            candidate_date = None
        if candidate_date is not None:
            candidate = datetime.combine(candidate_date, time(hour, minute))
            if candidate > local_now:
                return candidate
        if frequency == "YEARLY":
            candidate_year += interval
        else:
            candidate_month += interval
            candidate_year += (candidate_month - 1) // 12
            candidate_month = (candidate_month - 1) % 12 + 1


def resolve_pending_due_at(pending: PendingReminder, *, now: datetime) -> datetime:
    """Resolve a complete pending reminder in its user's local timezone."""

    if pending.date_local is None or pending.hour is None:
        raise ValueError("pending reminder is missing date or time")
    hour = pending.hour
    if pending.meridiem:
        hour = clock_to_24(hour, pending.meridiem)
    zone = ZoneInfo(pending.timezone_name)
    return datetime.combine(
        pending.date_local,
        time(hour, pending.minute),
        tzinfo=zone,
    )


def date_from_day(match, local_now: datetime) -> date:
    assert match is not None
    value = match.group("day").lower()
    if value == "today":
        return local_now.date()
    if value == "tomorrow":
        return local_now.date() + timedelta(days=1)
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
        return date.fromisoformat(value)
    target = WEEKDAYS[value]
    days_ahead = (target - local_now.weekday()) % 7
    return local_now.date() + timedelta(days=days_ahead)
