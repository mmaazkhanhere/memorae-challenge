"""Authorized completion of already-delivered Telegram reminder occurrences."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from reminder_agent.features.reminders.complete.messages import (
    reminder_already_completed_message,
    reminder_completed_message,
    reminder_not_available_message,
)
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
)
from reminder_agent.infrastructure.observability import get_logger, log_event

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
NOTIFIED = "NOTIFIED"
COMPLETED = "COMPLETED"
SENT = "SENT"
logger = get_logger()
CompletionCandidate = tuple[ReminderOccurrence, Reminder, Notification]


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
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None,
    channel_type: str = TELEGRAM,
    external_destination_id: str | None = None,
    include_completed: bool = False,
) -> Select:
    reminder_states = [ACTIVE, COMPLETED] if include_completed else [ACTIVE]
    occurrence_states = [NOTIFIED, COMPLETED] if include_completed else [NOTIFIED]
    destination_filters = [
        ChannelIdentity.user_id == user_id,
        ChannelIdentity.channel_type == channel_type,
    ]
    if channel_type == TELEGRAM and chat_id is not None:
        destination_filters.append(ChannelIdentity.telegram_chat_id == chat_id)
    elif channel_type != TELEGRAM and external_destination_id is not None:
        destination_filters.append(ChannelIdentity.external_id == external_destination_id)
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
            *destination_filters,
            ChannelIdentity.status == LINKED,
        )
    )


def resolve_by_provider_message(
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None,
    provider_message_id: str,
    channel_type: str = TELEGRAM,
    external_destination_id: str | None = None,
) -> list[CompletionCandidate]:
    candidates = [
        cast(CompletionCandidate, row)
        for row in db.execute(
            _candidate_query(
                db,
                user_id=user_id,
                identity_id=identity_id,
                chat_id=chat_id,
                channel_type=channel_type,
                external_destination_id=external_destination_id,
                include_completed=True,
            ).where(Notification.provider_message_id == provider_message_id)
        ).all()
    ]
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
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None,
    occurrence_id: UUID,
    channel_type: str = TELEGRAM,
    external_destination_id: str | None = None,
) -> list[CompletionCandidate]:
    candidates = [
        cast(CompletionCandidate, row)
        for row in db.execute(
            _candidate_query(
                db,
                user_id=user_id,
                identity_id=identity_id,
                chat_id=chat_id,
                channel_type=channel_type,
                external_destination_id=external_destination_id,
            ).where(ReminderOccurrence.id == occurrence_id)
        ).all()
    ]
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
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None,
    limit: int = 2,
    channel_type: str = TELEGRAM,
    external_destination_id: str | None = None,
) -> list[CompletionCandidate]:
    """Return at most ``limit`` distinct delivered occurrences for bare acknowledgements.

    The row limit must be applied after collapsing historical SENT notifications;
    otherwise multiple deliveries for one occurrence can hide another eligible
    occurrence and incorrectly make a bare acknowledgement appear unambiguous.
    """

    candidates = [
        cast(CompletionCandidate, row)
        for row in db.execute(
            _candidate_query(
                db,
                user_id=user_id,
                identity_id=identity_id,
                chat_id=chat_id,
                channel_type=channel_type,
                external_destination_id=external_destination_id,
            ).order_by(Notification.sent_at.desc(), Notification.id.desc())
        ).all()
    ]
    distinct: list[tuple[ReminderOccurrence, Reminder, Notification]] = []
    seen_occurrences: set[UUID] = set()
    for candidate in candidates:
        occurrence = candidate[0]
        if occurrence.id in seen_occurrences:
            continue
        seen_occurrences.add(occurrence.id)
        distinct.append(candidate)
        if len(distinct) >= limit:
            break
    log_event(
        logger,
        logging.DEBUG,
        "completion_candidates_resolved",
        resolution="latest_delivered",
        user_id=str(user_id),
        identity_id=str(identity_id),
        chat_id=chat_id,
        candidate_count=len(distinct),
        limit=limit,
    )
    return distinct


def complete_occurrence(
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None = None,
    occurrence_id: UUID,
    provider_message_id: str | None = None,
    channel_type: str = TELEGRAM,
    external_destination_id: str | None = None,
    now: datetime | None = None,
) -> CompletionResult:
    """Complete one occurrence after rechecking ownership and delivery in a lock."""
    # ``complete_occurrence`` deliberately does not commit.  The caller must
    # commit the state transition and its durable idempotency response together.
    now = now or datetime.now(UTC)
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
            ChannelIdentity.channel_type == channel_type,
            ChannelIdentity.status == LINKED,
        )
    )
    destination_matches = identity is not None and (
        (channel_type == TELEGRAM and identity.telegram_chat_id == chat_id)
        or (
            channel_type != TELEGRAM
            and (external_destination_id is None or identity.external_id == external_destination_id)
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
        return CompletionResult("NOT_FOUND", reminder_not_available_message())
    if not destination_matches:
        log_event(
            logger,
            logging.INFO,
            "completion_attempt_rejected",
            outcome="NOT_DELIVERED",
            reason="chat_mismatch",
            occurrence_id=str(occurrence_id),
        )
        return CompletionResult("NOT_DELIVERED", reminder_not_available_message())

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
        return CompletionResult("NOT_DELIVERED", reminder_not_available_message())
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
            "ALREADY_COMPLETED", reminder_already_completed_message(), occurrence.id
        )
    if reminder.status != ACTIVE or occurrence.status != NOTIFIED:
        log_event(
            logger,
            logging.INFO,
            "completion_attempt_rejected",
            outcome="NOT_DELIVERED",
            reason="reminder_or_occurrence_not_active",
            occurrence_id=str(occurrence_id),
        )
        return CompletionResult("NOT_DELIVERED", reminder_not_available_message())

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
    return CompletionResult("COMPLETED", reminder_completed_message(reminder.task), occurrence.id)
