"""Pure schedule preparation and validation for reminder creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from reminder_agent.features.intent.contracts import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
)
from reminder_agent.features.reminders.create.presentation import (
    format_next_occurrence,
    format_when,
)
from reminder_agent.features.reminders.recurrence import (
    MonthlyRecurrence,
    WeeklyRecurrence,
    YearlyRecurrence,
    resolve_local_datetime,
)

CreationIntent = CreateOneTimeReminderIntent | CreateRecurringReminderIntent
Recurrence = WeeklyRecurrence | MonthlyRecurrence | YearlyRecurrence
CreationSchedule = tuple[CreationIntent, datetime, str]
ValidationCode = Literal[
    "INVALID_USER_TIMEZONE",
    "INVALID_TYPED_INTENT",
    "TIMEZONE_MISMATCH",
    "PAST_OR_PRESENT_DUE_TIME",
    "INVALID_RECURRING_INTENT",
]


@dataclass(frozen=True)
class CreationValidationFailure:
    code: ValidationCode
    error_type: str | None = None


@dataclass(frozen=True)
class CreationValidationResult:
    schedules: tuple[CreationSchedule, ...] = ()
    failure: CreationValidationFailure | None = None


def recurrence_for_intent(intent: CreateRecurringReminderIntent) -> Recurrence:
    """Build the recurrence value from a validated creation intent."""

    if intent.frequency == "WEEKLY":
        return WeeklyRecurrence(intent.weekdays, intent.hour, intent.minute, intent.interval)
    if intent.frequency == "MONTHLY":
        return MonthlyRecurrence(
            intent.day_of_month or 0,
            intent.hour,
            intent.minute,
            intent.interval,
        )
    return YearlyRecurrence(
        intent.month or 0,
        intent.day_of_month or 0,
        intent.hour,
        intent.minute,
        intent.interval,
    )


def resolve_creation_due_at(intent: CreationIntent, *, zone: ZoneInfo) -> datetime:
    """Resolve a validated creation intent to its UTC due instant."""

    if isinstance(intent, CreateRecurringReminderIntent):
        recurrence_for_intent(intent)
        local_due = resolve_local_datetime(
            intent.start_local.date(),
            intent.start_local.timetz(),
            zone,
        )
        due_at = local_due
    else:
        due_at = intent.due_at
    if due_at.tzinfo is None or due_at.utcoffset() is None:
        return due_at.replace(tzinfo=UTC)
    return due_at.astimezone(UTC)


def validate_creation_schedules(
    intents: list[object],
    *,
    timezone_name: str,
    reference: datetime,
) -> CreationValidationResult:
    """Validate creation intents in the existing precedence order."""

    try:
        zone = ZoneInfo(timezone_name)
        scheduled: list[CreationSchedule] = []
        for intent in intents:
            if not isinstance(intent, (CreateOneTimeReminderIntent, CreateRecurringReminderIntent)):
                return CreationValidationResult(
                    failure=CreationValidationFailure("INVALID_TYPED_INTENT")
                )
            if intent.timezone_name != timezone_name:
                return CreationValidationResult(
                    failure=CreationValidationFailure("TIMEZONE_MISMATCH")
                )
            due_at = resolve_creation_due_at(intent, zone=zone)
            if due_at <= reference:
                return CreationValidationResult(
                    failure=CreationValidationFailure("PAST_OR_PRESENT_DUE_TIME")
                )
            local_due = due_at.astimezone(zone)
            local_reference = reference.astimezone(zone)
            local_display = (
                format_next_occurrence(local_due)
                if isinstance(intent, CreateRecurringReminderIntent)
                else format_when(local_due, reference=local_reference)
            )
            scheduled.append((intent, due_at, local_display))
        return CreationValidationResult(schedules=tuple(scheduled))
    except ZoneInfoNotFoundError:
        return CreationValidationResult(failure=CreationValidationFailure("INVALID_USER_TIMEZONE"))
    except (TypeError, ValueError, ValidationError) as exc:
        return CreationValidationResult(
            failure=CreationValidationFailure(
                "INVALID_RECURRING_INTENT",
                error_type=type(exc).__name__,
            )
        )
