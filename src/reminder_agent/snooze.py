"""Authorized snoozing of already-delivered reminder occurrences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.db.models import ChannelIdentity, Notification, Reminder, ReminderOccurrence

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
NOTIFIED = "NOTIFIED"
PENDING = "PENDING"
SENT = "SENT"


class PendingSnooze(BaseModel):
    """Durable, non-authoritative context for an unfinished snooze request."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["SNOOZE_OCCURRENCE"] = "SNOOZE_OCCURRENCE"
    candidate_occurrence_ids: tuple[UUID, ...] = Field(min_length=1, max_length=2)
    duration_seconds: int | None = Field(default=None, gt=0)


class SnoozeDuration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(gt=0)

    @field_validator("duration_seconds")
    @classmethod
    def duration_must_fit_timedelta(cls, value: int) -> int:
        try:
            timedelta(seconds=value)
        except OverflowError as exc:  # pragma: no cover - defensive boundary
            raise ValueError("duration is too large") from exc
        return value


_SNOOZE_PREFIX = re.compile(
    r"^\s*(?:(?:please\s+)?snooz(?:e|ed|ing)"
    r"(?:\s+(?:it|that|this|the\s+(?:reminder|task)))?\s*)?"
    r"(?:(?:for|by)\s+)?",
    re.IGNORECASE,
)
_SNOOZE_DURATION = re.compile(
    r"(?P<value>\d+)\s*(?:(?:more)\s+)?(?P<unit>seconds?|secs?|minutes?|mins?|hours?|days?)"
    r"\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def parse_snooze_duration(text: str) -> SnoozeDuration | None:
    """Parse only an explicit positive relative duration."""

    raw = text.strip()
    match = _SNOOZE_DURATION.search(raw)
    if match is None:
        return None
    prefix = raw[: match.start()]
    if (
        _SNOOZE_PREFIX.fullmatch(prefix) is None
        and re.match(r"^\s*(?:please\s+)?snooz(?:e|ed|ing)\b", prefix, re.IGNORECASE)
        is None
    ):
        return None
    value = int(match.group("value"))
    unit = match.group("unit").casefold()
    multiplier = (
        1 if unit.startswith(("second", "sec")) else
        60 if unit.startswith(("minute", "min")) else
        3600 if unit.startswith("hour") else
        86400
    )
    try:
        return SnoozeDuration(duration_seconds=value * multiplier)
    except (OverflowError, ValueError):
        return None


def snooze_target_text(text: str) -> str | None:
    """Return an optional task phrase from a snooze command.

    The phrase is only a candidate-selection hint.  The application service
    still resolves the occurrence from the authenticated user's delivered
    notifications and rechecks ownership in its write transaction.
    """

    raw = text.strip()
    match = _SNOOZE_DURATION.search(raw)
    if match is None:
        return None
    prefix = raw[: match.start()].strip()
    prefix = re.sub(r"^(?:please\s+)?snooz(?:e|ed|ing)\b", "", prefix, flags=re.IGNORECASE).strip()
    prefix = re.sub(
        r"^(?:it|that|this|the(?:\s+(?:reminder|task))?)\b", "", prefix, flags=re.IGNORECASE
    ).strip()
    prefix = re.sub(r"^(?:for|by)\b", "", prefix, flags=re.IGNORECASE).strip()
    prefix = re.sub(r"\s+(?:for|by)\s*$", "", prefix, flags=re.IGNORECASE).strip()
    return " ".join(prefix.split()) or None


def is_snooze_command(text: str) -> bool:
    normalized = text.strip().casefold()
    return parse_snooze_duration(text) is not None or bool(
        re.search(r"\bsnooz(?:e|ed|ing)\b", normalized)
    )


@dataclass(frozen=True)
class SnoozeResult:
    outcome: str
    text: str
    occurrence_id: UUID | None = None
    due_at: datetime | None = None


def snooze_occurrence(
    db: Session,
    *,
    user_id,
    identity_id,
    chat_id: int,
    occurrence_id: UUID,
    duration_seconds: int,
    provider_message_id: str | None = None,
    now: datetime | None = None,
) -> SnoozeResult:
    """Stage a new notification for one delivered occurrence; do not commit."""

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    if duration_seconds <= 0:
        return SnoozeResult("INVALID_DURATION", "Please provide a positive snooze duration.")
    due_at = now + timedelta(seconds=duration_seconds)

    identity = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.id == identity_id,
            ChannelIdentity.user_id == user_id,
            ChannelIdentity.channel_type == TELEGRAM,
            ChannelIdentity.status == LINKED,
            ChannelIdentity.telegram_chat_id == chat_id,
        )
    )
    occurrence = db.scalar(
        select(ReminderOccurrence)
        .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
        .where(ReminderOccurrence.id == occurrence_id, Reminder.user_id == user_id)
        .with_for_update()
    )
    if identity is None or occurrence is None:
        return SnoozeResult("NOT_FOUND", "I couldn't find that delivered reminder to snooze.")

    reminder = db.get(Reminder, occurrence.reminder_id)
    if reminder is None or reminder.status != ACTIVE or occurrence.status != NOTIFIED:
        return SnoozeResult("NOT_ELIGIBLE", "That reminder is no longer available to snooze.")

    delivery_filters = [
        Notification.occurrence_id == occurrence.id,
        Notification.status == SENT,
        Notification.destination_identity_id == identity.id,
    ]
    if provider_message_id is not None:
        delivery_filters.append(Notification.provider_message_id == provider_message_id)
    delivered = db.scalar(
        select(Notification).where(*delivery_filters).with_for_update()
    )
    if delivered is None:
        return SnoozeResult(
            "NOT_DELIVERED", "That reminder was not delivered to this Telegram chat."
        )

    # The occurrence lock serializes snoozes and makes sequence allocation safe.
    max_sequence = db.scalar(
        select(func.max(Notification.sequence)).where(Notification.occurrence_id == occurrence.id)
    ) or 0
    occurrence.status = PENDING
    occurrence.next_due_at = due_at
    occurrence.updated_at = now
    db.add(
        Notification(
            occurrence_id=occurrence.id,
            destination_identity_id=identity.id,
            sequence=int(max_sequence) + 1,
            due_at=due_at,
            status=PENDING,
            created_at=now,
        )
    )
    return SnoozeResult(
        "SNOOZED",
        f"Snoozed '{reminder.task}' until {due_at.isoformat(timespec='minutes')} UTC on Telegram.",
        occurrence.id,
        due_at,
    )
