"""Timezone-aware date and recurrence calculations for deterministic intent."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from reminder_agent.features.intent.contracts import PendingReminder
from reminder_agent.features.intent.text_normalization import (
    RecurrenceFrequency,
    clock_to_24,
    normalise_now,
)

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
    target = WEEKDAYS[value]
    days_ahead = (target - local_now.weekday()) % 7
    return local_now.date() + timedelta(days=days_ahead)
