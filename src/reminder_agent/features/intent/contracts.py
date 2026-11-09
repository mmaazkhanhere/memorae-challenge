"""Channel-neutral intent routes and validated reminder contracts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reminder_agent.shared.memory_contracts import (
    MemoryIntentEvidence,
    RetrieveMemoryQuery,
    SaveMemoryFact,
)


def normalize_task_query(value: str) -> str:
    """Normalize a user-supplied listing subject without widening it."""

    normalized = " ".join(value.strip().split())
    normalized = re.sub(r"^(?:all\s+)?(?:the\s+)?", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+reminders?$", "", normalized, flags=re.IGNORECASE)
    return normalized.strip(" .,!?")


class IntentRoute(StrEnum):
    NEW_REQUEST = "NEW_REQUEST"
    LIST_REQUEST = "LIST_REQUEST"
    SUMMARY_REQUEST = "SUMMARY_REQUEST"
    RESCHEDULE_REQUEST = "RESCHEDULE_REQUEST"
    CANCEL_REQUEST = "CANCEL_REQUEST"
    COMPLETE_REQUEST = "COMPLETE_REQUEST"
    SNOOZE_REQUEST = "SNOOZE_REQUEST"
    SAVE_MEMORY = "SAVE_MEMORY"
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    UPDATE_PENDING = "UPDATE_PENDING"
    CANCEL_PENDING = "CANCEL_PENDING"
    SOCIAL = "SOCIAL"
    UNRELATED = "UNRELATED"
    UNCERTAIN = "UNCERTAIN"


class IntentRouteSuggestion(BaseModel):
    """A non-authoritative route proposed by the intent-classification model."""

    model_config = ConfigDict(extra="forbid")

    operation: IntentRoute
    target_text: str | None = None
    memory_intent_evidence: MemoryIntentEvidence | None = None


class CreateOneTimeReminderIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_ONE_TIME_REMINDER"] = "CREATE_ONE_TIME_REMINDER"
    task: str = Field(min_length=1, max_length=200)
    due_at: datetime
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("task must not be blank")
        return value

    @field_validator("due_at")
    @classmethod
    def due_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("due_at must include timezone information")
        return value.astimezone(UTC)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value


class CreateRecurringReminderIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_RECURRING_REMINDER"] = "CREATE_RECURRING_REMINDER"
    task: str = Field(min_length=1, max_length=200)
    frequency: Literal["WEEKLY", "MONTHLY", "YEARLY"] = "WEEKLY"
    interval: int = Field(default=1, ge=1, le=60)
    weekdays: tuple[int, ...] = Field(default=(), max_length=7)
    day_of_month: int | None = Field(default=None, ge=1, le=31)
    month: int | None = Field(default=None, ge=1, le=12)
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    start_local: datetime
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("task must not be blank")
        return value

    @field_validator("weekdays")
    @classmethod
    def weekdays_must_be_sorted_and_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(day < 0 or day > 6 for day in value):
            raise ValueError("weekdays must be between Monday (0) and Sunday (6)")
        normalized = tuple(sorted(set(value)))
        if normalized != value:
            raise ValueError("weekdays must be sorted and unique")
        return value

    @field_validator("start_local")
    @classmethod
    def start_local_must_be_naive(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("start_local must be timezone-naive")
        return value.replace(tzinfo=None)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def recurrence_fields_must_match_frequency(self) -> CreateRecurringReminderIntent:
        if self.frequency == "WEEKLY":
            if not self.weekdays or self.start_local.weekday() not in self.weekdays:
                raise ValueError("weekly start_local must match a selected weekday")
            if self.day_of_month is not None or self.month is not None:
                raise ValueError("weekly recurrence cannot include month fields")
        elif self.frequency == "MONTHLY":
            if self.day_of_month is None or self.start_local.day != self.day_of_month:
                raise ValueError("monthly start_local must match day_of_month")
            if self.weekdays or self.month is not None:
                raise ValueError("monthly recurrence has invalid fields")
        else:
            if (
                self.month is None
                or self.day_of_month is None
                or self.start_local.month != self.month
                or self.start_local.day != self.day_of_month
            ):
                raise ValueError("yearly start_local must match month and day_of_month")
            if self.weekdays:
                raise ValueError("yearly recurrence cannot include weekdays")
        return self


class ListRemindersIntent(BaseModel):
    """A read-only request with typed date, time, weekday, and task filters."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["LIST_REMINDERS"] = "LIST_REMINDERS"
    start_date: date | None = None
    end_date: date | None = None
    task_query: str | None = Field(default=None, min_length=1, max_length=200)
    time_after: time | None = None
    time_before: time | None = None
    weekdays: tuple[int, ...] | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value

    @field_validator("task_query")
    @classmethod
    def task_query_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_task_query(value)
        if not normalized:
            raise ValueError("task_query must not be blank")
        return normalized

    @field_validator("weekdays", mode="before")
    @classmethod
    def normalize_weekdays(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise ValueError("weekdays must be a sequence of weekday numbers")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 6
            for item in values
        ):
            raise ValueError("weekdays must contain only integers from 0 through 6")
        normalized = tuple(sorted(set(values)))
        return normalized or None

    @model_validator(mode="after")
    def date_range_must_be_ordered(self) -> ListRemindersIntent:
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must be supplied together")
        if (
            self.time_after is not None
            and self.time_before is not None
            and self.time_after > self.time_before
        ):
            raise ValueError("time_after must not be later than time_before")
        if self.start_date is None:
            return self
        assert self.end_date is not None
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        if (self.end_date - self.start_date).days > 366:
            raise ValueError("date range is too large")
        return self


class ReminderSummaryIntent(BaseModel):
    """A read-only request for reminder progress or task counts."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["SUMMARY"] = "SUMMARY"
    metric: Literal[
        "PENDING_COUNT",
        "TODAY_REMAINING",
        "COMPLETED_COUNT",
        "TODAY_COMPLETED",
        "OVERDUE_COUNT",
    ]
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value


class PendingReminder(BaseModel):
    """Validated, serializable partial data for a one-time or weekly reminder."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_ONE_TIME_REMINDER"] = "CREATE_ONE_TIME_REMINDER"
    task: str | None = Field(default=None, max_length=200)
    date_local: date | None = None
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    meridiem: Literal["am", "pm"] | None = None
    recurrence_weekdays: tuple[int, ...] | None = None
    recurrence_frequency: Literal["MONTHLY", "YEARLY"] | None = None
    recurrence_interval: int | None = Field(default=None, ge=1, le=60)
    recurrence_day_of_month: int | None = Field(default=None, ge=1, le=31)
    recurrence_month: int | None = Field(default=None, ge=1, le=12)
    timezone_name: str = Field(min_length=1, max_length=150)
    date_explicit: bool = False

    @field_validator("meridiem", mode="before")
    @classmethod
    def normalise_meridiem(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None

    @field_validator("recurrence_weekdays")
    @classmethod
    def recurrence_weekdays_must_be_sorted_and_unique(
        cls, value: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        if value is None:
            return value
        if not value or any(day < 0 or day > 6 for day in value):
            raise ValueError("recurrence weekdays must be between Monday (0) and Sunday (6)")
        if tuple(sorted(set(value))) != value:
            raise ValueError("recurrence weekdays must be sorted and unique")
        return value

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value


class PendingReminderFieldUpdate(BaseModel):
    """Only fields explicitly supplied by a reminder clarification reply."""

    model_config = ConfigDict(extra="forbid")

    task: str | None = Field(default=None, max_length=200)
    date_local: date | None = None
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int | None = Field(default=None, ge=0, le=59)
    meridiem: Literal["am", "pm"] | None = None
    date_explicit: bool | None = None
    recurrence_frequency: Literal["MONTHLY", "YEARLY"] | None = None
    recurrence_interval: int | None = Field(default=None, ge=1, le=60)
    recurrence_day_of_month: int | None = Field(default=None, ge=1, le=31)
    recurrence_month: int | None = Field(default=None, ge=1, le=12)
    certainty: Literal["CLEAR"] | None = None
    evidence: dict[
        Literal[
            "task",
            "date_local",
            "hour",
            "minute",
            "meridiem",
            "date_explicit",
            "recurrence_frequency",
            "recurrence_interval",
            "recurrence_day_of_month",
            "recurrence_month",
        ],
        str,
    ] = Field(default_factory=dict, exclude=True)

    @field_validator("certainty", mode="before")
    @classmethod
    def normalise_certainty(cls, value: object) -> object:
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("meridiem", mode="before")
    @classmethod
    def normalise_meridiem(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None

    def supplied_fields(self) -> set[str]:
        """Usable provider values, including ``false`` but never null clears."""
        return {
            field
            for field in self.model_fields_set
            if field not in {"certainty", "evidence"} and getattr(self, field) is not None
        }


class NeedsClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_ONE_TIME_REMINDER"] = "CREATE_ONE_TIME_REMINDER"
    pending: PendingReminder
    question: str = Field(min_length=1, max_length=500)


def merge_pending_reminder_updates(
    pending: PendingReminder, updates: PendingReminderFieldUpdate
) -> PendingReminder:
    """Apply explicitly supplied follow-up fields to trusted pending state."""

    values = pending.model_dump(mode="python")
    supplied = updates.supplied_fields()
    for field in supplied:
        values[field] = getattr(updates, field)
    if "date_local" in supplied and "date_explicit" not in supplied:
        values["date_explicit"] = updates.date_local is not None
    return PendingReminder.model_validate(values)


@dataclass(frozen=True)
class Interpretation:
    """Channel-neutral result of intent interpretation and validation."""

    intent: (
        CreateOneTimeReminderIntent
        | CreateRecurringReminderIntent
        | ListRemindersIntent
        | ReminderSummaryIntent
        | None
    ) = None
    intents: tuple[
        CreateOneTimeReminderIntent
        | CreateRecurringReminderIntent
        | ListRemindersIntent
        | ReminderSummaryIntent,
        ...,
    ] = ()
    clarification: NeedsClarification | None = None
    field_updates: PendingReminderFieldUpdate | None = None
    reschedule_request: Any | None = None
    cancellation_request: Any | None = None
    pending_reschedule_proposal: Any | None = None
    response_text: str | None = None
    fallback_allowed: bool = True
    clear_context: bool = False
    completion_target_text: str | None = None
    snooze_duration_seconds: int | None = None
    memory_fact: SaveMemoryFact | None = None
    memory_query: RetrieveMemoryQuery | None = None

    @property
    def resolved_intents(
        self,
    ) -> tuple[
        CreateOneTimeReminderIntent
        | CreateRecurringReminderIntent
        | ListRemindersIntent
        | ReminderSummaryIntent,
        ...,
    ]:
        """Return one or more validated commands represented by this result."""
        if self.intents:
            return self.intents
        return (self.intent,) if self.intent is not None else ()


class IntentInterpreter(Protocol):
    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
        reschedule_candidates: Sequence[dict[str, str]] = (),
        cancellation_candidates: Sequence[dict[str, str]] = (),
        pending_reschedule: Any | None = None,
    ) -> Interpretation: ...
