"""Authorized snoozing of already-delivered reminder occurrences."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.features.reminders.snooze.messages import snooze_message
from reminder_agent.infrastructure.db.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
)
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
NOTIFIED = "NOTIFIED"
PENDING = "PENDING"
SENT = "SENT"


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
        and re.match(r"^\s*(?:please\s+)?snooz(?:e|ed|ing)\b", prefix, re.IGNORECASE) is None
    ):
        return None
    value = int(match.group("value"))
    unit = match.group("unit").casefold()
    multiplier = (
        1
        if unit.startswith(("second", "sec"))
        else 60
        if unit.startswith(("minute", "min"))
        else 3600
        if unit.startswith("hour")
        else 86400
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
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None = None,
    occurrence_id: UUID,
    duration_seconds: int,
    provider_message_id: str | None = None,
    channel_type: str = TELEGRAM,
    external_destination_id: str | None = None,
    delivery_label: str | None = "Telegram",
    now: datetime | None = None,
) -> SnoozeResult:
    """Stage a new notification for one delivered occurrence; do not commit."""

    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)
    log_event(
        logger,
        logging.INFO,
        "snooze_mutation_started",
        user_id=str(user_id),
        identity_id=str(identity_id),
        chat_id=chat_id,
        occurrence_id=str(occurrence_id),
        duration_seconds=duration_seconds,
        has_provider_message_id=provider_message_id is not None,
    )
    if duration_seconds <= 0:
        log_event(logger, logging.INFO, "snooze_mutation_rejected", outcome="INVALID_DURATION")
        return SnoozeResult("INVALID_DURATION", snooze_message("invalid_duration"))
    due_at = now + timedelta(seconds=duration_seconds)

    identity_query = select(ChannelIdentity).where(
        ChannelIdentity.id == identity_id,
        ChannelIdentity.user_id == user_id,
        ChannelIdentity.channel_type == channel_type,
        ChannelIdentity.status == LINKED,
    )
    if channel_type == TELEGRAM:
        identity_query = identity_query.where(ChannelIdentity.telegram_chat_id == chat_id)
    elif external_destination_id is not None:
        identity_query = identity_query.where(
            ChannelIdentity.external_id == external_destination_id
        )
    identity = db.scalar(identity_query)
    occurrence = db.scalar(
        select(ReminderOccurrence)
        .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
        .where(ReminderOccurrence.id == occurrence_id, Reminder.user_id == user_id)
        .with_for_update()
    )
    if identity is None or occurrence is None:
        log_event(
            logger,
            logging.INFO,
            "snooze_mutation_rejected",
            outcome="NOT_FOUND",
            occurrence_id=str(occurrence_id),
        )
        return SnoozeResult("NOT_FOUND", snooze_message("reminder_not_found"))

    reminder = db.get(Reminder, occurrence.reminder_id)
    if reminder is None or reminder.status != ACTIVE or occurrence.status != NOTIFIED:
        log_event(
            logger,
            logging.INFO,
            "snooze_mutation_rejected",
            outcome="NOT_ELIGIBLE",
            occurrence_id=str(occurrence_id),
            occurrence_status=occurrence.status,
        )
        return SnoozeResult("NOT_ELIGIBLE", snooze_message("reminder_unavailable"))

    delivery_filters = [
        Notification.occurrence_id == occurrence.id,
        Notification.status == SENT,
        Notification.destination_identity_id == identity.id,
    ]
    if provider_message_id is not None:
        delivery_filters.append(Notification.provider_message_id == provider_message_id)
    delivered = db.scalar(select(Notification).where(*delivery_filters).with_for_update())
    if delivered is None:
        log_event(
            logger,
            logging.INFO,
            "snooze_mutation_rejected",
            outcome="NOT_DELIVERED",
            occurrence_id=str(occurrence_id),
        )
        return SnoozeResult("NOT_DELIVERED", snooze_message("wrong_telegram_chat"))

    # The occurrence lock serializes snoozes and makes sequence allocation safe.
    max_sequence = (
        db.scalar(
            select(func.max(Notification.sequence)).where(
                Notification.occurrence_id == occurrence.id
            )
        )
        or 0
    )
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
    log_event(
        logger,
        logging.INFO,
        "snooze_mutation_staged",
        outcome="SNOOZED",
        occurrence_id=str(occurrence.id),
        notification_sequence=int(max_sequence) + 1,
        due_at=due_at.isoformat(),
    )
    new_datetime = f"{due_at.isoformat(timespec='minutes')} UTC"
    return SnoozeResult(
        "SNOOZED",
        snooze_message(
            "snooze_success",
            task=reminder.task,
            when=f"at {new_datetime}",
            new_datetime=new_datetime,
        ),
        occurrence.id,
        due_at,
    )
