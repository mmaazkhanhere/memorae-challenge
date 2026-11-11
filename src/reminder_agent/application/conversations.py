from __future__ import annotations

import importlib
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from reminder_agent.application.pending_contracts import PendingCompletion, PendingSnooze
from reminder_agent.infrastructure.database.models import ConversationSession
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.memory_contracts import PendingMemory, PendingMemoryLookup

if TYPE_CHECKING:
    from reminder_agent.features.intent.contracts import PendingReminder

TELEGRAM = "TELEGRAM"
VOICE_CALL = "VOICE_CALL"
AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
ENDED = "ENDED"
ACTIVE = "ACTIVE"
logger = get_logger()


def upsert_clarification_session(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    pending: (
        PendingReminder | PendingCompletion | PendingSnooze | PendingMemory | PendingMemoryLookup
    ),
    now: datetime,
    channel_type: str = TELEGRAM,
) -> ConversationSession:
    """Upsert pending operation context inside the caller's transaction."""

    values = {
        "id": uuid4(),
        "user_id": user_id,
        "channel_type": channel_type,
        "external_session_id": external_session_id,
        "active_reminder_id": None,
        "pending_clarification": pending.model_dump(mode="json"),
        "status": AWAITING_CLARIFICATION,
        "version": 1,
        "updated_at": _utc(now),
    }
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        statement: Any = pg_insert(ConversationSession).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[
                ConversationSession.user_id,
                ConversationSession.channel_type,
                ConversationSession.external_session_id,
            ],
            set_={
                "pending_clarification": values["pending_clarification"],
                "status": AWAITING_CLARIFICATION,
                "active_reminder_id": None,
                "version": ConversationSession.version + 1,
                "updated_at": values["updated_at"],
            },
        )
        db.execute(statement)
    elif dialect == "sqlite":
        statement = sqlite_insert(ConversationSession).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[
                ConversationSession.user_id,
                ConversationSession.channel_type,
                ConversationSession.external_session_id,
            ],
            set_={
                "pending_clarification": values["pending_clarification"],
                "status": AWAITING_CLARIFICATION,
                "active_reminder_id": None,
                "version": ConversationSession.version + 1,
                "updated_at": values["updated_at"],
            },
        )
        db.execute(statement)
    else:  # pragma: no cover
        db.add(ConversationSession(**values))
    session = db.scalar(
        select(ConversationSession).where(
            ConversationSession.user_id == user_id,
            ConversationSession.channel_type == channel_type,
            ConversationSession.external_session_id == external_session_id,
        )
    )
    assert session is not None
    return session


def end_pending_session(
    db: Session,
    update: Any,
    *,
    user: Any,
    now: datetime,
    session_ttl_seconds: int,
    response_text: str | None = None,
    response_kind: str = "snooze_canceled",
    operation: str = "SNOOZE_OCCURRENCE",
) -> Any:
    """End pending context and optionally stage its response atomically."""

    response_staging = importlib.import_module("reminder_agent.application.response_staging")

    assert update.message is not None
    db.rollback()
    with db.begin():
        existing = response_staging._existing_record(db, update)
        if existing is not None:
            return response_staging._reply_from_record(existing)
        session = load_session(
            db,
            user_id=user.id,
            external_session_id=telegram_session_id(update.message.chat.id),
            now=now,
            ttl_seconds=session_ttl_seconds,
            for_update=True,
        )
        if session is not None:
            end_session(session, now=now)
        if response_text is None:
            return None
        return response_staging._store_response(
            db,
            update=update,
            user_id=user.id,
            text=response_text,
            response_kind=response_kind,
            operation=operation,
        )


def session_id_for_external_key(external_key: str | int) -> str:
    """Return the stable session key used by an authenticated channel."""

    return str(external_key)


def telegram_session_id(chat_id: int) -> str:
    """Compatibility name for Telegram's chat-based session key."""

    return session_id_for_external_key(chat_id)


def save_pending_clarification_session(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    pending: BaseModel,
    now: datetime,
    channel_type: str = TELEGRAM,
    active_reminder_id: UUID | None = None,
) -> ConversationSession:
    """Persist pending context without coupling the store to a provider."""

    session = db.scalar(
        select(ConversationSession)
        .where(
            ConversationSession.user_id == user_id,
            ConversationSession.channel_type == channel_type,
            ConversationSession.external_session_id == external_session_id,
        )
        .with_for_update()
    )
    pending_data = pending.model_dump(mode="json")
    if session is None:
        session = ConversationSession(
            user_id=user_id,
            channel_type=channel_type,
            external_session_id=external_session_id,
            active_reminder_id=active_reminder_id,
            pending_clarification=pending_data,
            status=AWAITING_CLARIFICATION,
            version=1,
            updated_at=_utc(now),
        )
        db.add(session)
    else:
        session.pending_clarification = pending_data
        session.status = AWAITING_CLARIFICATION
        session.active_reminder_id = active_reminder_id
        session.version += 1
        session.updated_at = _utc(now)
    return session


def load_voice_session(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    now: datetime,
    ttl_seconds: int,
    for_update: bool = False,
) -> ConversationSession | None:
    """Load active voice context, including completed-operation references.

    ``load_session`` intentionally serves clarification workflows and only
    returns ``AWAITING_CLARIFICATION`` rows. Voice calls also need a durable
    reference after a committed reminder operation, so they use this broader
    helper at the channel application boundary.
    """

    statement = select(ConversationSession).where(
        ConversationSession.user_id == user_id,
        ConversationSession.channel_type == VOICE_CALL,
        ConversationSession.external_session_id == external_session_id,
    )
    if for_update:
        statement = statement.with_for_update()
    session = db.scalar(statement)
    if session is None or session.status == ENDED:
        return None
    if session_expired(session, now=now, ttl_seconds=ttl_seconds):
        if for_update:
            end_session(session, now=now)
        return None
    return session


def activate_voice_reminder_reference(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    reminder_id: UUID,
    now: datetime,
) -> ConversationSession:
    """Persist one committed, owner-scoped reminder as the call's active topic."""

    session = db.scalar(
        select(ConversationSession)
        .where(
            ConversationSession.user_id == user_id,
            ConversationSession.channel_type == VOICE_CALL,
            ConversationSession.external_session_id == external_session_id,
        )
        .with_for_update()
    )
    if session is None:
        session = ConversationSession(
            user_id=user_id,
            channel_type=VOICE_CALL,
            external_session_id=external_session_id,
            status=ACTIVE,
            version=1,
            updated_at=_utc(now),
        )
        db.add(session)
        db.flush()
    session.active_reminder_id = reminder_id
    session.pending_clarification = None
    session.status = ACTIVE
    session.version += 1
    session.updated_at = _utc(now)
    return session


def clear_voice_reminder_reference(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    now: datetime,
) -> None:
    """Clear only the active voice topic while leaving the call available."""

    session = db.scalar(
        select(ConversationSession)
        .where(
            ConversationSession.user_id == user_id,
            ConversationSession.channel_type == VOICE_CALL,
            ConversationSession.external_session_id == external_session_id,
        )
        .with_for_update()
    )
    if session is None or session.status == ENDED:
        return
    session.active_reminder_id = None
    session.version += 1
    session.updated_at = _utc(now)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


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
    channel_type: str = TELEGRAM,
) -> ConversationSession | None:
    statement = select(ConversationSession).where(
        ConversationSession.user_id == user_id,
        ConversationSession.channel_type == channel_type,
        ConversationSession.external_session_id == external_session_id,
    )
    if for_update:
        statement = statement.with_for_update()
    session = db.scalar(statement)
    if session is None or (
        session.status != AWAITING_CLARIFICATION
        and not (channel_type == VOICE_CALL and session.status == ACTIVE)
    ):
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
    if session.pending_clarification is None and not (
        channel_type == VOICE_CALL and session.status == ACTIVE
    ):
        return None
    return session


def pending_from_session(session: ConversationSession | None) -> PendingReminder | None:
    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "CREATE_ONE_TIME_REMINDER":
        return None
    from reminder_agent.features.intent.contracts import PendingReminder

    return PendingReminder.model_validate(session.pending_clarification)


def pending_reschedule_from_session(session: ConversationSession | None) -> Any:
    """Load VS-09 context without treating it as a reminder-creation draft."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "RESCHEDULE_REMINDER":
        return None
    from reminder_agent.features.reminders.reschedule.decisions import PendingReschedule

    return PendingReschedule.model_validate(session.pending_clarification)


def pending_cancellation_from_session(session: ConversationSession | None) -> Any:
    """Load VS-10 context without trusting it as an authorization source."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "CANCEL_REMINDER":
        return None
    from reminder_agent.features.reminders.cancel.decisions import PendingCancellation

    return PendingCancellation.model_validate(session.pending_clarification)


def pending_completion_from_session(session: ConversationSession | None) -> Any:
    """Load VS-11 completion candidates without treating IDs as authority."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "COMPLETE_REMINDER":
        return None
    return PendingCompletion.model_validate(session.pending_clarification)


def pending_snooze_from_session(session: ConversationSession | None) -> Any:
    """Load VS-12 snooze candidates without treating IDs as authority."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "SNOOZE_OCCURRENCE":
        return None
    return PendingSnooze.model_validate(session.pending_clarification)


def pending_memory_from_session(session: ConversationSession | None) -> PendingMemory | None:
    """Load a pending memory-save clarification without treating it as authority."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "SAVE_MEMORY":
        return None
    return PendingMemory.model_validate(session.pending_clarification)


def pending_memory_lookup_from_session(
    session: ConversationSession | None,
) -> PendingMemoryLookup | None:
    """Load an ambiguous memory lookup without trusting IDs as authority."""

    if session is None or session.pending_clarification is None:
        return None
    if session.pending_clarification.get("operation") != "RETRIEVE_MEMORY":
        return None
    return PendingMemoryLookup.model_validate(session.pending_clarification)


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
