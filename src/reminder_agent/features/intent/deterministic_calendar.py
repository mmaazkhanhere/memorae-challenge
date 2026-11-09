"""Calendar and summary branches for deterministic intent interpretation."""

from __future__ import annotations

import re
from datetime import date, datetime, time
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
from reminder_agent.features.intent.text_normalization import (
    CalendarFrequency,
    SummaryMetric,
)
from reminder_agent.features.intent.text_normalization import (
    clock_to_24 as _clock_to_24,
)
from reminder_agent.features.intent.text_normalization import (
    normalise_meridiem as _normalise_meridiem,
)
from reminder_agent.features.intent.text_normalization import (
    ordinal_suffix as _ordinal_suffix,
)
from reminder_agent.features.intent.text_normalization import (
    task_from_remainder as _task_from_remainder,
)


class CalendarIntentMixin:
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
        task = _task_from_remainder(f"{match.group('before') or ''} {match.group('after') or ''}")
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

