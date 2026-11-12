"""Contracts shared by reminder rescheduling parsing and execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ACTIVE = "ACTIVE"
PENDING = "PENDING"
SENDING = "SENDING"
CANCELED = "CANCELED"
RECURRING = "RECURRING"
ONE_TIME = "ONE_TIME"


class RescheduleScope(StrEnum):
    OCCURRENCE = "OCCURRENCE"
    SERIES = "SERIES"


class PendingRescheduleOperation(StrEnum):
    """Non-authoritative interpretations of an unfinished reschedule."""

    SELECT_CANDIDATE = "SELECT_CANDIDATE"
    UPDATE_PENDING_PARAMETERS = "UPDATE_PENDING_PARAMETERS"
    SELECT_AND_UPDATE = "SELECT_AND_UPDATE"
    START_NEW_OPERATION = "START_NEW_OPERATION"
    CANCEL_PENDING = "CANCEL_PENDING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class PendingReschedule(BaseModel):
    """Durable, non-authoritative context for a follow-up question."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["RESCHEDULE_REMINDER"] = "RESCHEDULE_REMINDER"
    target_text: str = Field(min_length=1, max_length=200)
    due_at: datetime | None = None
    date_explicit: bool = False
    offset_minutes: int | None = Field(default=None, ge=-10080, le=10080)
    source_text: str | None = Field(default=None, max_length=500)
    timezone_name: str = Field(min_length=1, max_length=150)
    candidate_ids: tuple[UUID, ...] = ()
    scope: RescheduleScope | None = None
    occurrence_date: date | None = None
    destination_date: date | None = None
    destination_weekday: int | None = Field(default=None, ge=0, le=6)

    @field_validator("target_text")
    @classmethod
    def target_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("target_text must not be blank")
        return value

    @field_validator("due_at")
    @classmethod
    def due_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("due_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def has_exactly_one_time_target(self) -> PendingReschedule:
        if self.due_at is None and self.offset_minutes is None and not self.source_text:
            raise ValueError("pending reschedule needs a time target or source request")
        if self.due_at is not None and self.offset_minutes is not None:
            raise ValueError("pending reschedule cannot have both absolute and relative targets")
        return self

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        ZoneInfo(value)
        return value


class PendingRescheduleProposal(BaseModel):
    """Validated model proposal, never a command to mutate the database.

    Candidate IDs are intentionally opaque UUIDs.  The application validates
    membership in ``PendingReschedule.candidate_ids`` again while holding the
    conversation lock before using one.
    """

    model_config = ConfigDict(extra="forbid")

    operation: PendingRescheduleOperation
    candidate_id: UUID | None = None
    due_at: datetime | None = None
    destination_date: date | None = None
    date_explicit: bool | None = None
    offset_minutes: int | None = Field(default=None, ge=-10080, le=10080)
    scope: RescheduleScope | None = None
    question: str | None = Field(default=None, max_length=500)

    @field_validator("due_at")
    @classmethod
    def proposal_due_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("due_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def proposal_is_coherent(self) -> PendingRescheduleProposal:
        has_time = self.due_at is not None or self.offset_minutes is not None
        has_update = has_time or self.destination_date is not None or self.scope is not None
        if self.due_at is not None and self.offset_minutes is not None:
            raise ValueError("proposal cannot have both absolute and relative targets")
        if self.destination_date is not None and self.due_at is not None:
            raise ValueError("proposal cannot have both a local date and absolute datetime")
        if (
            self.operation
            in {
                PendingRescheduleOperation.SELECT_CANDIDATE,
                PendingRescheduleOperation.SELECT_AND_UPDATE,
            }
            and self.candidate_id is None
        ):
            raise ValueError("candidate selection proposal needs a candidate ID")
        if (
            self.operation is PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS
            and not has_update
        ):
            raise ValueError("parameter update proposal needs an update")
        if self.operation is PendingRescheduleOperation.SELECT_AND_UPDATE and not has_update:
            raise ValueError("combined proposal needs an update")
        if self.operation in {
            PendingRescheduleOperation.CANCEL_PENDING,
            PendingRescheduleOperation.START_NEW_OPERATION,
            PendingRescheduleOperation.NEEDS_CLARIFICATION,
        } and (self.candidate_id is not None or has_update):
            raise ValueError("non-update proposal contains action fields")
        return self


@dataclass(frozen=True)
class RescheduleRequest:
    target_text: str
    due_at: datetime | None = None
    date_explicit: bool = False
    offset_minutes: int | None = None
    scope: RescheduleScope | None = None
    source_text: str | None = None
    occurrence_date: date | None = None
    destination_date: date | None = None
    # A named destination weekday is a cadence change only for a series.  A
    # relative or ISO destination date merely selects the first new due date.
    destination_weekday: int | None = None


@dataclass(frozen=True)
class RescheduleClarification:
    """A non-mutating, typed partial reschedule proposed by the extractor."""

    request: RescheduleRequest
    question: str


@dataclass(frozen=True)
class RescheduleCandidate:
    reminder_id: UUID
    occurrence_id: UUID
    task: str
    due_at: datetime
    schedule_type: str


@dataclass(frozen=True)
class RescheduleDecision:
    """Result of safe parsing/candidate resolution before a write transaction."""

    kind: Literal["MUTATE", "CLARIFY", "RESPONSE", "CANCEL"]
    text: str
    request: RescheduleRequest | None = None
    candidate_ids: tuple[UUID, ...] = ()
    occurrence_id: UUID | None = None


class RescheduleConflict(RuntimeError):
    """The requested reminder cannot safely be changed at this moment."""


__all__ = [
    "ACTIVE",
    "CANCELED",
    "ONE_TIME",
    "PENDING",
    "RECURRING",
    "SENDING",
    "PendingReschedule",
    "PendingRescheduleOperation",
    "PendingRescheduleProposal",
    "RescheduleCandidate",
    "RescheduleClarification",
    "RescheduleConflict",
    "RescheduleDecision",
    "RescheduleRequest",
    "RescheduleScope",
]
