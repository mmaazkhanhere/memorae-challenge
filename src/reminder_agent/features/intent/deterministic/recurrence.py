"""Deterministic weekly, calendar, and summary interpretation."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Literal, cast
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from reminder_agent.features.intent.contracts import (
    CreateRecurringReminderIntent,
    Interpretation,
    NeedsClarification,
    PendingReminder,
    ReminderSummaryIntent,
)
from reminder_agent.features.intent.deterministic.patterns import (
    contains_multiple_tasks as _contains_multiple_tasks,
)
from reminder_agent.features.intent.deterministic.support import (
    CalendarFrequency,
    SummaryMetric,
)
from reminder_agent.features.intent.deterministic.support import (
    clock_to_24 as _clock_to_24,
)
from reminder_agent.features.intent.deterministic.support import (
    normalise_meridiem as _normalise_meridiem,
)
from reminder_agent.features.intent.deterministic.support import (
    ordinal_suffix as _ordinal_suffix,
)
from reminder_agent.features.intent.deterministic.support import (
    strip_repeated_calendar_cadence as _strip_repeated_calendar_cadence,
)
from reminder_agent.features.intent.deterministic.support import (
    strip_repeated_weekly_cadence as _strip_repeated_weekly_cadence,
)
from reminder_agent.features.intent.deterministic.support import (
    task_from_remainder as _task_from_remainder,
)
from reminder_agent.features.intent.deterministic.support import (
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
    def _interpret_summary(
        self, raw: str, *, timezone_name: str, now: datetime
    ) -> Interpretation | None:
        """Parse common read-only progress questions without a model call."""

        normalized = " ".join(raw.lower().strip(" .!?").split())
        if self._summary_request.match(raw) is None:
            return None
        if re.search(r"\b(?:overdue|late)\b", normalized):
            metric: SummaryMetric = "OVERDUE_COUNT"
        elif re.search(r"\b(?:completed|done|finished)\b", normalized):
            metric = (
                "TODAY_COMPLETED"
                if re.search(r"\b(?:today|this\s+day)\b", normalized)
                else "COMPLETED_COUNT"
            )
        elif re.search(r"\b(?:still\s+have\s+to\s+do|have\s+left|remaining)\b", normalized):
            metric = "TODAY_REMAINING"
        else:
            metric = "PENDING_COUNT"
        return Interpretation(
            intent=ReminderSummaryIntent(metric=metric, timezone_name=timezone_name),
            fallback_allowed=False,
        )

    def _interpret_calendar_recurrence(
        self, text: str, *, timezone_name: str, now: datetime
    ) -> Interpretation | None:
        ordinal = self._monthly_ordinal_recurrence.search(text)
        if ordinal is not None:
            raw_ordinal = ordinal.group("ordinal").lower()
            day = self._ordinal_numbers.get(raw_ordinal)
            if day is None:
                day = int(re.sub(r"\D", "", raw_ordinal))
            return self._build_calendar_interpretation(
                ordinal,
                frequency="MONTHLY",
                day=day,
                month=None,
                timezone_name=timezone_name,
                now=now,
            )
        match = self._monthly_recurrence.search(text)
        frequency: CalendarFrequency = "MONTHLY"
        if match is None:
            match = self._yearly_recurrence.search(text)
            frequency = "YEARLY"
        if match is None:
            return None
        return self._build_calendar_interpretation(
            match,
            frequency=frequency,
            day=int(match.group("day")),
            month=(self._months[match.group("month").lower()] if frequency == "YEARLY" else None),
            timezone_name=timezone_name,
            now=now,
        )

    def _build_calendar_interpretation(
        self,
        match: re.Match[str],
        *,
        frequency: CalendarFrequency,
        day: int,
        month: int | None,
        timezone_name: str,
        now: datetime,
    ) -> Interpretation:
        hour_value = match.group("hour")
        hour = int(hour_value) if hour_value is not None else None
        minute = int(match.group("minute") or 0)
        meridiem = _normalise_meridiem(match.group("meridiem"))
        task = _strip_repeated_calendar_cadence(
            _task_from_remainder(f"{match.group('before') or ''} {match.group('after') or ''}"),
            frequency,
        )
        if _contains_multiple_tasks(task):
            return self._multiple_target_clarification(timezone_name)
        if hour is None:
            local_now = now.astimezone(ZoneInfo(timezone_name))
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        task=task or None,
                        date_local=local_now.date(),
                        hour=None,
                        recurrence_frequency=cast(Literal["MONTHLY", "YEARLY"], frequency),
                        recurrence_day_of_month=day,
                        recurrence_month=month,
                        timezone_name=timezone_name,
                        date_explicit=True,
                    ),
                    question=(
                        f"What time should I use on the {day}{_ordinal_suffix(day)} day "
                        f"of each {'month' if frequency == 'MONTHLY' else 'year'}?"
                    ),
                ),
                fallback_allowed=False,
            )
        if meridiem is None and hour <= 12:
            return Interpretation(
                response_text=f"Do you mean {hour:g} AM or {hour:g} PM for this {frequency.lower()} reminder?",
                fallback_allowed=False,
            )
        if meridiem is not None:
            hour = _clock_to_24(hour, meridiem)
        if frequency == "YEARLY":
            assert month is not None
        try:
            candidate_year = now.astimezone(ZoneInfo(timezone_name)).year
            local_now = now.astimezone(ZoneInfo(timezone_name))
            candidate_month = month or local_now.month
            while True:
                try:
                    candidate_date = date(candidate_year, candidate_month, day)
                except ValueError:
                    candidate_year += 1 if frequency == "YEARLY" else 0
                    if frequency == "MONTHLY":
                        candidate_month += 1
                        if candidate_month > 12:
                            candidate_month = 1
                            candidate_year += 1
                    continue
                candidate_local = datetime.combine(candidate_date, time(hour, minute))
                if candidate_local > local_now.replace(tzinfo=None):
                    break
                if frequency == "YEARLY":
                    candidate_year += 1
                else:
                    candidate_month += 1
                    if candidate_month > 12:
                        candidate_month = 1
                        candidate_year += 1
            if not task:
                return Interpretation(
                    response_text="What should I remind you about on that calendar date?",
                    fallback_allowed=False,
                )
            return Interpretation(
                intent=CreateRecurringReminderIntent(
                    task=task,
                    frequency=frequency,
                    interval=1,
                    weekdays=(),
                    day_of_month=day,
                    month=month,
                    hour=hour,
                    minute=minute,
                    start_local=candidate_local,
                    timezone_name=timezone_name,
                )
            )
        except (TypeError, ValueError, ValidationError):
            return Interpretation(
                response_text="Please provide a valid monthly or yearly reminder date.",
                fallback_allowed=False,
            )

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
            else tuple(
                sorted(
                    {
                        self._weekdays[day]
                        for day in re.findall(
                            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
                            raw_days,
                        )
                    }
                )
            )
        )
        task = _strip_repeated_weekly_cadence(
            _task_from_remainder(match.group("task")), raw_days
        )
        if not task or not weekdays:
            return None
        if _contains_multiple_tasks(task):
            return self._multiple_target_clarification(timezone_name)
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
        task = _strip_repeated_weekly_cadence(
            _task_from_remainder(match.group("task")), raw_day_text
        )
        if not task:
            question = "What should I remind you about every week?"
            return Interpretation(
                response_text=question,
                fallback_allowed=False,
            )
        if _contains_multiple_tasks(task):
            return self._multiple_target_clarification(timezone_name)
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
