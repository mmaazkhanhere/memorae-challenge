from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.db.models import ConversationSession
from reminder_agent.completion import PendingCompletion
from reminder_agent.intent import PendingReminder
from reminder_agent.observability import get_logger, log_event

TELEGRAM = "TELEGRAM"
AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
ENDED = "ENDED"
logger = get_logger()


def telegram_session_id(chat_id: int) -> str:
    return str(chat_id)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def session_expired(session: ConversationSession, *, now: datetime, ttl_seconds: int) -> bool:
    return _utc(now) - _utc(session.updated_at) > timedelta(seconds=ttl_seconds)


def load_session(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    now: datetime,
    ttl_seconds: int,
    for_update: bool = False,
) -> ConversationSession | None:
    statement = select(ConversationSession).where(
        ConversationSession.user_id == user_id,
        ConversationSession.channel_type == TELEGRAM,
        ConversationSession.external_session_id == external_session_id,
    )
    if for_update:
        statement = statement.with_for_update()
    session = db.scalar(statement)
    if session is None or session.status != AWAITING_CLARIFICATION:
        return None
    if session_expired(session, now=now, ttl_seconds=ttl_seconds):
        if for_update:
            log_event(
                logger,
                logging.INFO,
                "clarification_session_expired",
                session_id=str(session.id),
                session_version=session.version,
            )
            end_session(session, now=now)
            return None
        # Return the stale row for the read-before-interpreting phase. The caller
        # can carry its context for the expiry check, while only the transaction
        # holding the row lock is allowed to end it.
        return session
    if session.pending_clarification is None:
        return None
    return session


def pending_from_session(session: ConversationSession | None) -> PendingReminder | None:
    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "CREATE_ONE_TIME_REMINDER":
        return None
    return PendingReminder.model_validate(session.pending_clarification)


def pending_reschedule_from_session(session: ConversationSession | None):
    """Load VS-09 context without treating it as a reminder-creation draft."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "RESCHEDULE_REMINDER":
        return None
    from reminder_agent.rescheduling import PendingReschedule

    return PendingReschedule.model_validate(session.pending_clarification)


def pending_cancellation_from_session(session: ConversationSession | None):
    """Load VS-10 context without trusting it as an authorization source."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "CANCEL_REMINDER":
        return None
    from reminder_agent.cancellation import PendingCancellation

    return PendingCancellation.model_validate(session.pending_clarification)


def pending_completion_from_session(session: ConversationSession | None):
    """Load VS-11 completion candidates without treating IDs as authority."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "COMPLETE_REMINDER":
        return None
    return PendingCompletion.model_validate(session.pending_clarification)


def end_session(session: ConversationSession, *, now: datetime) -> None:
    """Clear pending clarification state after completion, cancellation, or expiry."""
    session.status = ENDED
    session.pending_clarification = None
    session.active_reminder_id = None
    session.version += 1
    session.updated_at = _utc(now)
    log_event(
        logger,
        logging.INFO,
        "clarification_session_ended",
        session_id=str(session.id),
        session_version=session.version,
    )
