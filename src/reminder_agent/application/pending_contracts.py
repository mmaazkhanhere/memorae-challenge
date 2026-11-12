"""Durable pending-operation contracts owned by the application layer."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PendingCompletion(BaseModel):
    """Durable, non-authoritative context for an ambiguous completion reply."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["COMPLETE_REMINDER"] = "COMPLETE_REMINDER"
    candidate_occurrence_ids: tuple[UUID, ...] = Field(min_length=1, max_length=2)


class PendingSnooze(BaseModel):
    """Durable, non-authoritative context for an unfinished snooze request."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["SNOOZE_OCCURRENCE"] = "SNOOZE_OCCURRENCE"
    candidate_occurrence_ids: tuple[UUID, ...] = Field(min_length=1, max_length=2)
    duration_seconds: int | None = Field(default=None, gt=0)
