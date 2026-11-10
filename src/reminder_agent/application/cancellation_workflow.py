"""Application orchestration for an authenticated cancellation request.

The cancellation feature owns parsing, target resolution, validation, and the
state transition. This workflow owns session state and durable response
staging; the Telegram adapter supplies a validated update and neutral request.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import ApplicationRequest
from reminder_agent.application.conversations import (
    end_session,
    load_session,
    pending_cancellation_from_session,
    save_pending_clarification_session,
)
from reminder_agent.application.response_staging import (
    ReminderReply,
    _existing_record,
    _recover_from_idempotency_race,
    _reply_from_record,
    _store_response,
)
from reminder_agent.features.reminders.cancel.decisions import (
    CancellationConflict,
    CancellationRequest,
    PendingCancellation,
    decide_cancellation,
    is_cancel_command,
)
from reminder_agent.features.reminders.cancel.messages import cancel_message
from reminder_agent.features.reminders.cancel.service import (
    cancel_reminder_for_request,
    cancellation_command_for_decision,
    prepare_cancellation,
)
from reminder_agent.infrastructure.db.models import User
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()


def handle_cancellation_request(
    db: Session,
    update: Any,
    *,
    request: ApplicationRequest,
    user_id: UUID,
    user: User,
    pending: PendingCancellation | None,
    now: datetime,
    session_ttl_seconds: int = 900,
    request_override: CancellationRequest | None = None,
    cancellation_interpreter: Any | None = None,
    recent_messages: list[str] | None = None,
    force: bool = False,
) -> ReminderReply | None:
    """Process one authenticated cancellation request without channel parsing."""

    text = request.message_text
    log_event(
        logger,
        logging.INFO,
        "cancellation_command_started",
        update_id=update.update_id,
        input_length=len(text.strip()),
        has_pending=pending is not None,
    )
    if pending is None and request_override is None and not is_cancel_command(text):
        if not force:
            return None

    db.rollback()
    preparation = prepare_cancellation(
        db,
        user_id=user_id,
        text=text,
        timezone_name=user.timezone or "UTC",
        now=now,
        pending=pending,
        request_override=request_override,
        cancellation_interpreter=cancellation_interpreter,
        recent_messages=recent_messages,
    )
    decision = preparation.decision
    parsed_request = preparation.parsed_request
    pending_for_decision = preparation.pending_for_decision
    db.rollback()

    try:
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                return _reply_from_record(existing)
            assert decision is not None
            session = load_session(
                db,
                user_id=user_id,
                external_session_id=request.context.external_session_id,
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            locked_pending = pending_cancellation_from_session(session)
            if pending_for_decision is not None and locked_pending is not None:
                decision = decide_cancellation(
                    db,
                    user_id=user_id,
                    text=text,
                    timezone_name=user.timezone or "UTC",
                    now=now,
                    pending=locked_pending,
                )

            if decision.kind == "CANCEL":
                if session is not None:
                    end_session(session, now=now)
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="cancellation_abandoned",
                    operation="CANCEL_REMINDER",
                )

            if decision.kind == "RESPONSE":
                if session is not None and (
                    locked_pending is not None or parsed_request is not None
                ):
                    end_session(session, now=now)
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="cancellation_response",
                    operation="CANCEL_REMINDER",
                )

            if decision.kind == "CLARIFY":
                cancellation_request = decision.request
                if cancellation_request is None:
                    return _store_response(
                        db,
                        update=update,
                        user_id=user_id,
                        text=decision.text,
                        response_kind="cancellation_clarification",
                        operation="CANCEL_REMINDER",
                    )
                pending_record = PendingCancellation(
                    target_text=cancellation_request.target_text,
                    candidate_ids=decision.candidate_ids,
                    scope=cancellation_request.scope,
                    occurrence_date=cancellation_request.occurrence_date,
                    timezone_name=user.timezone or "UTC",
                )
                save_pending_clarification_session(
                    db,
                    user_id=user_id,
                    external_session_id=request.context.external_session_id,
                    pending=pending_record,
                    now=now,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "cancellation_clarification_saved",
                    update_id=update.update_id,
                    candidate_count=len(decision.candidate_ids),
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="cancellation_clarification",
                    operation="CANCEL_REMINDER",
                )

            cancellation_command = cancellation_command_for_decision(decision)
            if cancellation_command is None:
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=cancel_message("missing_target"),
                    response_kind="cancellation_clarification",
                    operation="CANCEL_REMINDER",
                )
            result = cancel_reminder_for_request(
                db,
                request=request,
                command=cancellation_command,
                now=now,
            )
            if result.response_kind == "AUTHORIZATION_FAILURE":
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=cancel_message("authorization_failed"),
                    response_kind="authorization_failure",
                    operation="CANCEL_REMINDER",
                )
            assert result.resource_id is not None
            log_event(
                logger,
                logging.INFO,
                "reminder_canceled",
                source_message_id=str(update.update_id),
                reminder_id=str(result.resource_id),
                scope=decision.scope.value if decision.scope is not None else None,
                occurrence_count=result.affected_count,
            )
            if session is not None:
                end_session(session, now=now)
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                durable_response=result.durable_response,
                response_kind="cancellation_confirmation",
                operation="CANCEL_REMINDER",
            )
    except IntegrityError:
        duplicate = _recover_from_idempotency_race(db, update)
        if duplicate is not None:
            return duplicate
        raise
    except CancellationConflict as exc:
        log_event(
            logger,
            logging.WARNING,
            "reminder_cancellation_conflict",
            update_id=update.update_id,
            reason_type=type(exc).__name__,
        )
        db.rollback()
        with db.begin():
            session = load_session(
                db,
                user_id=user_id,
                external_session_id=request.context.external_session_id,
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if parsed_request is not None and session is not None:
                end_session(session, now=now)
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                text=cancel_message("cancellation_failed", reason=str(exc).rstrip(".")),
                response_kind="cancellation_conflict",
                operation="CANCEL_REMINDER",
            )
