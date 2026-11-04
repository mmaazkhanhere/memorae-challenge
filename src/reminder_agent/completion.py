"""Authorized completion of already-delivered Telegram reminder occurrences."""

from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import logging
import re
from typing import Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.db.models import ChannelIdentity, Notification, Reminder, ReminderOccurrence
from reminder_agent.observability import get_logger, log_event

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
NOTIFIED = "NOTIFIED"
COMPLETED = "COMPLETED"
SENT = "SENT"
logger = get_logger()


class PendingCompletion(BaseModel):
    """Durable, non-authoritative context for an ambiguous completion reply."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["COMPLETE_REMINDER"] = "COMPLETE_REMINDER"
    candidate_occurrence_ids: tuple[UUID, ...] = Field(min_length=1, max_length=2)


def _completion_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+", value.casefold()):
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        tokens.append(token)
    return tuple(tokens)


def match_completion_candidates(
    text: str,
    candidates: Sequence[tuple[ReminderOccurrence, Reminder, Notification]],
) -> list[tuple[ReminderOccurrence, Reminder, Notification]]:
    """Return a unique best task match, or no matches when the reply is unclear."""
    reply_tokens = _completion_tokens(text)
    if not reply_tokens:
        return []
    reply_text = " ".join(reply_tokens)
    scored: list[tuple[float, tuple[ReminderOccurrence, Reminder, Notification]]] = []
    for candidate in candidates:
        task_tokens = _completion_tokens(candidate[1].task)
        if not task_tokens:
            continue
        overlap = len(set(task_tokens) & set(reply_tokens))
        coverage = overlap / len(set(task_tokens))
        similarity = SequenceMatcher(None, " ".join(task_tokens), reply_text).ratio()
        score = (coverage * 0.7) + (similarity * 0.3)
        if coverage == 1.0 or score >= 0.68:
            scored.append((score, candidate))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.1:
        return [candidate for _, candidate in scored]
    return [scored[0][1]]


@dataclass(frozen=True)
class CompletionResult:
    outcome: str
    text: str
    occurrence_id: UUID | None = None


def _candidate_query(
    db: Session, *, user_id, identity_id, chat_id, include_completed: bool = False
):
    reminder_states = [ACTIVE, COMPLETED] if include_completed else [ACTIVE]
    occurrence_states = [NOTIFIED, COMPLETED] if include_completed else [NOTIFIED]
    return (
        select(ReminderOccurrence, Reminder, Notification)
        .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
        .join(Notification, Notification.occurrence_id == ReminderOccurrence.id)
        .join(ChannelIdentity, ChannelIdentity.id == Notification.destination_identity_id)
        .where(
            Reminder.user_id == user_id,
            Reminder.status.in_(reminder_states),
            ReminderOccurrence.status.in_(occurrence_states),
            Notification.status == SENT,
            Notification.destination_identity_id == identity_id,
            ChannelIdentity.user_id == user_id,
            ChannelIdentity.channel_type == TELEGRAM,
            ChannelIdentity.status == LINKED,
            ChannelIdentity.telegram_chat_id == chat_id,
        )
    )


def resolve_by_provider_message(
    db: Session, *, user_id, identity_id, chat_id: int, provider_message_id: str
) -> list[tuple[ReminderOccurrence, Reminder, Notification]]:
    candidates = list(
        db.execute(
            _candidate_query(
                db,
                user_id=user_id,
                identity_id=identity_id,
                chat_id=chat_id,
                include_completed=True,
            ).where(Notification.provider_message_id == provider_message_id)
        ).all()
    )
    log_event(
        logger,
        logging.DEBUG,
        "completion_candidates_resolved",
        resolution="provider_message",
        user_id=str(user_id),
        identity_id=str(identity_id),
        chat_id=chat_id,
        provider_message_id=provider_message_id,
        candidate_count=len(candidates),
    )
    return candidates


def resolve_by_occurrence(
    db: Session, *, user_id, identity_id, chat_id: int, occurrence_id: UUID
) -> list[tuple[ReminderOccurrence, Reminder, Notification]]:
    candidates = list(
        db.execute(
            _candidate_query(
                db, user_id=user_id, identity_id=identity_id, chat_id=chat_id
            ).where(ReminderOccurrence.id == occurrence_id)
        ).all()
    )
    log_event(
        logger,
        logging.DEBUG,
        "completion_candidates_resolved",
        resolution="occurrence_id",
        user_id=str(user_id),
        identity_id=str(identity_id),
        chat_id=chat_id,
        occurrence_id=str(occurrence_id),
        candidate_count=len(candidates),
    )
    return candidates


def resolve_latest_delivered(
    db: Session, *, user_id, identity_id, chat_id: int, limit: int = 2
) -> list[tuple[ReminderOccurrence, Reminder, Notification]]:
    """Return at most ``limit`` delivered occurrences for bare acknowledgements."""

    candidates = list(
        db.execute(
            _candidate_query(
                db, user_id=user_id, identity_id=identity_id, chat_id=chat_id
            )
            .order_by(Notification.sent_at.desc(), Notification.id.desc())
            .limit(limit)
        ).all()
    )
    log_event(
        logger,
        logging.DEBUG,
        "completion_candidates_resolved",
        resolution="latest_delivered",
        user_id=str(user_id),
        identity_id=str(identity_id),
        chat_id=chat_id,
        candidate_count=len(candidates),
        limit=limit,
    )
    return candidates


def complete_occurrence(
    db: Session,
    *,
    user_id,
    identity_id,
    chat_id: int,
    occurrence_id: UUID,
    provider_message_id: str | None = None,
    now: datetime | None = None,
) -> CompletionResult:
    """Complete one occurrence after rechecking ownership and delivery in a lock."""
    # ``complete_occurrence`` deliberately does not commit.  The caller must
    # commit the state transition and its durable idempotency response together.
    now = now or datetime.now(timezone.utc)
    log_event(
        logger,
        logging.DEBUG,
        "completion_attempt_started",
        user_id=str(user_id),
        identity_id=str(identity_id),
        chat_id=chat_id,
        occurrence_id=str(occurrence_id),
        provider_message_id=provider_message_id,
    )
    identity = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.id == identity_id,
            ChannelIdentity.user_id == user_id,
            ChannelIdentity.channel_type == TELEGRAM,
            ChannelIdentity.status == LINKED,
        )
    )
    occurrence = db.scalar(
        select(ReminderOccurrence)
        .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
        .where(ReminderOccurrence.id == occurrence_id, Reminder.user_id == user_id)
        .with_for_update()
    )
    reminder = db.get(Reminder, occurrence.reminder_id) if occurrence else None
    if identity is None or occurrence is None or reminder is None:
        log_event(
            logger,
            logging.INFO,
            "completion_attempt_rejected",
            outcome="NOT_FOUND",
            reason="ownership_or_occurrence_not_found",
            occurrence_id=str(occurrence_id),
        )
        return CompletionResult("NOT_FOUND", "I couldn't find that delivered reminder.")
    if identity.telegram_chat_id != chat_id:
        log_event(
            logger,
            logging.INFO,
            "completion_attempt_rejected",
            outcome="NOT_DELIVERED",
            reason="chat_mismatch",
            occurrence_id=str(occurrence_id),
        )
        return CompletionResult("NOT_DELIVERED", "That reminder is not available to mark as done.")

    # Recheck delivery and destination after taking the occurrence lock. This
    # prevents a caller-supplied or stale occurrence ID from becoming authority.
    delivery_filters = [
        Notification.occurrence_id == occurrence.id,
        Notification.status == SENT,
        Notification.destination_identity_id == identity.id,
    ]
    if provider_message_id is not None:
        delivery_filters.append(Notification.provider_message_id == provider_message_id)
    eligible = db.scalar(select(Notification).where(*delivery_filters).with_for_update())
    if eligible is None:
        log_event(
            logger,
            logging.INFO,
            "completion_attempt_rejected",
            outcome="NOT_DELIVERED",
            reason="notification_not_sent_to_identity",
            occurrence_id=str(occurrence_id),
        )
        return CompletionResult("NOT_DELIVERED", "That reminder is not available to mark as done.")
    if occurrence.status == COMPLETED:
        log_event(
            logger,
            logging.INFO,
            "completion_attempt_rejected",
            outcome="ALREADY_COMPLETED",
            reason="occurrence_already_completed",
            occurrence_id=str(occurrence_id),
        )
        return CompletionResult(
            "ALREADY_COMPLETED", "That reminder is already marked as done.", occurrence.id
        )
    if (
        reminder.status != ACTIVE
        or occurrence.status != NOTIFIED
    ):
        log_event(
            logger,
            logging.INFO,
            "completion_attempt_rejected",
            outcome="NOT_DELIVERED",
            reason="reminder_or_occurrence_not_active",
            occurrence_id=str(occurrence_id),
        )
        return CompletionResult("NOT_DELIVERED", "That reminder is not available to mark as done.")

    occurrence.status = COMPLETED
    occurrence.updated_at = now
    if reminder.schedule_type == "ONE_TIME":
        reminder.status = COMPLETED
    reminder.updated_at = now
    log_event(
        logger,
        logging.INFO,
        "completion_mutation_staged",
        outcome="COMPLETED",
        occurrence_id=str(occurrence.id),
        reminder_id=str(reminder.id),
        schedule_type=reminder.schedule_type,
    )
    return CompletionResult("COMPLETED", f"Marked as done: {reminder.task}.", occurrence.id)


def callback_data(occurrence_id: UUID, secret: str) -> str:
    value = str(occurrence_id)
    signature = hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:16]
    return f"done:{value}:{signature}"


def parse_callback_data(value: str, secret: str) -> UUID | None:
    if len(value) > 64:
        return None
    parts = value.split(":")
    if len(parts) != 3 or parts[0] != "done":
        return None
    try:
        occurrence_id = UUID(parts[1])
    except ValueError:
        return None
    expected = hmac.new(secret.encode(), parts[1].encode(), hashlib.sha256).hexdigest()[:16]
    return occurrence_id if hmac.compare_digest(parts[2], expected) else None
