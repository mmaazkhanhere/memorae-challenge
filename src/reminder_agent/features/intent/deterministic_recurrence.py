"""Weekly recurrence interpretation for deterministic intent."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from reminder_agent.features.intent.contracts import (
    CreateRecurringReminderIntent,
    Interpretation,
    NeedsClarification,
    PendingReminder,
)
from reminder_agent.features.intent.text_normalization import (
    clock_to_24 as _clock_to_24,
)
from reminder_agent.features.intent.text_normalization import (
    normalise_meridiem as _normalise_meridiem,
)
from reminder_agent.features.intent.text_normalization import (
    task_from_remainder as _task_from_remainder,
)
from reminder_agent.features.intent.text_normalization import (
    valid_clock as _valid_clock,
)
from reminder_agent.features.reminders.recurrence import WeeklyRecurrence

_PARTIAL_WEEKLY = re.compile(
    r"\bevery\s+(?P<days>day|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"(?:\s*(?:,|and)\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))*)"
    r"\s+(?:to|for)\s+(?P<task>.+?)\s*$",
    re.IGNORECASE,
)


class RecurrenceIntentMixin:
    def _interpret_partial_weekly(
        self,
        text: str,
        *,
        timezone_name: str,
        now: datetime,
    ) -> Interpretation | None:
        """Preserve a clear weekly task/cadence while asking only for time."""

        match = _PARTIAL_WEEKLY.search(text)
        if match is None:
            return None
        raw_days = match.group("days").lower()
        weekdays = (
            tuple(range(7))
            if raw_days == "day"
            else tuple(sorted({self._weekdays[day] for day in re.findall(
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
                raw_days,
            )}))
        )
        task = _task_from_remainder(match.group("task"))
        if not task or not weekdays:
            return None
        local_today = now.astimezone(ZoneInfo(timezone_name)).date()
        pending = PendingReminder(
            task=task,
            date_local=local_today,
            date_explicit=False,
            recurrence_weekdays=weekdays,
            timezone_name=timezone_name,
        )
        question = "What time should I use for that weekly reminder?"
        return Interpretation(
            clarification=NeedsClarification(pending=pending, question=question),
            response_text=question,
            fallback_allowed=False,
        )

    def _build_recurring(
        self,
        match: re.Match[str],
        *,
        timezone_name: str,
        now: datetime,
    ) -> Interpretation:
        raw_day_text = match.group("days").lower()
        if raw_day_text.strip() == "day":
            weekdays = tuple(range(7))
        else:
            raw_days = re.findall(
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
                raw_day_text,
            )
            weekdays = tuple(sorted({self._weekdays[day] for day in raw_days}))
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        meridiem = _normalise_meridiem(match.group("meridiem"))
        if not _valid_clock(hour, minute, meridiem):
            return self._invalid_time()
        if meridiem is not None:
            hour = _clock_to_24(hour, meridiem)
        task = _task_from_remainder(match.group("task"))
        if not task:
            question = "What should I remind you about every week?"
            return Interpretation(
                response_text=question,
                fallback_allowed=False,
            )
        if meridiem is None and hour <= 12:
            pending = PendingReminder(
                task=task,
                date_local=now.astimezone(ZoneInfo(timezone_name)).date(),
                hour=hour,
                minute=minute,
                recurrence_weekdays=weekdays,
                timezone_name=timezone_name,
            )
            question = f"Do you mean {hour:g} AM or {hour:g} PM every week?"
            return Interpretation(
                clarification=NeedsClarification(pending=pending, question=question),
                response_text=question,
                fallback_allowed=False,
            )
        local_now = now.astimezone(ZoneInfo(timezone_name))
        start_date = local_now.date()
        for offset in range(8):
            candidate = start_date + timedelta(days=offset)
            if candidate.weekday() not in weekdays:
                continue
            candidate_local = datetime.combine(candidate, time(hour, minute))
            if candidate_local > local_now.replace(tzinfo=None):
                start_local = candidate_local
                break
        else:  # pragma: no cover - a tuple of weekdays always yields a date
            raise AssertionError("could not calculate recurring start")
        try:
            WeeklyRecurrence(weekdays, hour, minute)
            return Interpretation(
                intent=CreateRecurringReminderIntent(
                    task=task,
                    weekdays=weekdays,
                    hour=hour,
                    minute=minute,
                    start_local=start_local,
                    timezone_name=timezone_name,
                )
            )
        except ValidationError:
            return Interpretation(
                response_text="Please provide a short recurring reminder task and time.",
                fallback_allowed=False,
            )


