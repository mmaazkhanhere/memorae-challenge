"""Reminder-list interpretation branch for deterministic intent."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from reminder_agent.features.intent.contracts import Interpretation, ListRemindersIntent
from reminder_agent.features.intent.deterministic.patterns import extract_listing_task_query
from reminder_agent.features.intent.deterministic.support import normalise_now as _normalise_now

_NEXT_LIST_REQUEST = re.compile(
    r"^\s*what(?:'s|\s+is)\s+(?:my\s+)?(?:next|upcoming)\s+reminders?(?:\s+to\s+do)?\s*[?!.]*\s*$",
    re.IGNORECASE,
)
_TIME_FILTER = re.compile(
    r"\b(?P<operator>after|before)\s+(?P<hour>\d{1,2})"
    r"(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.m\.|p\.m\.|am|pm)?\b",
    re.IGNORECASE,
)
_DATE_BOUND_FILTER = re.compile(
    r"\b(?P<operator>after|before)\s+(?P<date>\d{4}-\d{1,2}-\d{1,2})\b",
    re.IGNORECASE,
)
_RECURRING_LIST_REQUEST = re.compile(
    r"\b(?:recurring|weekly)\s+reminders?\b",
    re.IGNORECASE,
)


def _parse_filter_time(match: re.Match[str]) -> time | None:
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    meridiem = (match.group("meridiem") or "").lower().replace(".", "")
    if meridiem:
        if hour < 1 or hour > 12:
            return None
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


class ListingIntentMixin:
    def _interpret_listing(
        self, raw: str, *, timezone_name: str, now: datetime
    ) -> Interpretation | None:
        if (
            self._list_request.match(raw) is None
            and _NEXT_LIST_REQUEST.match(raw) is None
            and extract_listing_task_query(raw) is None
            # The router recognizes natural forms such as "What recurring
            # reminders have I got?".  Keep extraction aligned with that
            # route so they do not unnecessarily fall through to the model.
            and _RECURRING_LIST_REQUEST.search(raw) is None
        ):
            return None
        zone = ZoneInfo(timezone_name)
        local_today = _normalise_now(now).astimezone(zone).date()
        normalized = raw.lower().replace("firday", "friday")
        task_query = extract_listing_task_query(raw)

        def list_intent(
            start_date: date | None = None,
            end_date: date | None = None,
            *,
            time_after: time | None = None,
            time_before: time | None = None,
            weekdays: tuple[int, ...] | None = None,
            recurrence_only: bool = False,
            recurrence_series_only: bool = False,
            recurrence_frequency: str | None = None,
            limit: int | None = None,
        ) -> ListRemindersIntent:
            return ListRemindersIntent(
                start_date=start_date,
                end_date=end_date,
                task_query=task_query,
                time_after=time_after,
                time_before=time_before,
                weekdays=weekdays,
                recurrence_only=recurrence_only,
                recurrence_series_only=recurrence_series_only,
                recurrence_frequency=recurrence_frequency,
                limit=limit,
                timezone_name=timezone_name,
            )

        def resolve(value: str) -> date:
            if value == "today":
                return local_today
            if value == "tomorrow":
                return local_today + timedelta(days=1)
            if value in self._weekdays:
                return local_today + timedelta(
                    days=(self._weekdays[value] - local_today.weekday()) % 7
                )
            return date.fromisoformat(value)

        if _NEXT_LIST_REQUEST.match(raw) is not None:
            return Interpretation(
                intent=list_intent(limit=1),
                fallback_allowed=False,
            )

        if "next month" in normalized:
            next_month = (local_today.replace(day=28) + timedelta(days=4)).replace(day=1)
            following_month = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1)
            return Interpretation(
                intent=list_intent(next_month, following_month - timedelta(days=1)),
                fallback_allowed=False,
            )

        time_match = _TIME_FILTER.search(normalized)
        if time_match is not None:
            parsed_time = _parse_filter_time(time_match)
            if parsed_time is None:
                return Interpretation(
                    response_text="Please provide a valid reminder time.",
                    fallback_allowed=False,
                )
            if time_match.group("operator").lower() == "after":
                return Interpretation(
                    intent=list_intent(time_after=parsed_time),
                    fallback_allowed=False,
                )
            return Interpretation(
                intent=list_intent(time_before=parsed_time),
                fallback_allowed=False,
            )

        date_bound_match = _DATE_BOUND_FILTER.search(normalized)
        if date_bound_match is not None:
            # ListRemindersIntent represents closed date ranges only. Do not
            # silently turn an open-ended date bound into an exact-date query.
            return Interpretation(
                response_text="Please provide a complete reminder date range.",
                fallback_allowed=True,
            )

        if re.search(r"\bweekends?\b", normalized):
            if re.search(r"\b(?:the|this)\s+weekend\b", normalized):
                days_until_saturday = (5 - local_today.weekday()) % 7
                weekend_start = local_today + timedelta(days=days_until_saturday)
                return Interpretation(
                    intent=list_intent(weekend_start, weekend_start + timedelta(days=1)),
                    fallback_allowed=False,
                )
            return Interpretation(
                intent=list_intent(weekdays=(5, 6)),
                fallback_allowed=False,
            )

        if "this week" in normalized:
            start = local_today - timedelta(days=local_today.weekday())
            return Interpretation(
                intent=list_intent(start, start + timedelta(days=6)),
                fallback_allowed=False,
            )
        if "next week" in normalized:
            start = local_today - timedelta(days=local_today.weekday()) + timedelta(days=7)
            return Interpretation(
                intent=list_intent(start, start + timedelta(days=6)),
                fallback_allowed=False,
            )

        # A request for recurring reminders names reminder series, not their
        # individual projected occurrences. Return one next occurrence per
        # active series so monthly and yearly reminders are not hidden by an
        # arbitrary short date window.
        if ("recurring" in normalized or "weekly" in normalized) and task_query is None:
            return Interpretation(
                intent=list_intent(
                    recurrence_only=True,
                    recurrence_series_only=True,
                    recurrence_frequency=(
                        "WEEKLY" if "weekly" in normalized else None
                    ),
                ),
                fallback_allowed=False,
            )

        if "all" in normalized and task_query is None:
            return Interpretation(
                intent=list_intent(),
                fallback_allowed=False,
            )


        match = self._list_range.search(raw)
        if match is not None:
            try:
                start = resolve(match.group("start").lower())
                end = resolve(match.group("end").lower())
                if end < start:
                    return Interpretation(
                        response_text="Please provide a reminder date range in chronological order.",
                        fallback_allowed=False,
                    )
                return Interpretation(
                    intent=list_intent(start, end),
                    fallback_allowed=False,
                )
            except (TypeError, ValueError, ValidationError):
                return Interpretation(
                    response_text="Please provide a valid reminder date range.",
                    fallback_allowed=False,
                )

        for phrase, offset in (("tomorrow", 1), ("today", 0)):
            if phrase in normalized:
                day = local_today + timedelta(days=offset)
                return Interpretation(
                    intent=list_intent(day, day),
                    fallback_allowed=False,
                )

        for weekday, index in self._weekdays.items():
            if re.search(rf"\b{weekday}\b", normalized):
                day = local_today + timedelta(days=(index - local_today.weekday()) % 7)
                return Interpretation(
                    intent=list_intent(day, day),
                    fallback_allowed=False,
                )

        iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
        if len(iso_dates) == 1:
            try:
                day = date.fromisoformat(iso_dates[0])
                return Interpretation(
                    intent=list_intent(day, day),
                    fallback_allowed=False,
                )
            except ValueError:
                return Interpretation(
                    response_text="Please provide a valid reminder date.",
                    fallback_allowed=False,
                )

        if re.search(r"\b(?:from|between|on|for)\b", normalized):
            return Interpretation(
                response_text="Please provide the reminder date or date range.",
                fallback_allowed=True,
            )
        return Interpretation(
            intent=list_intent(),
            fallback_allowed=False,
        )
