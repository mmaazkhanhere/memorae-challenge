"""Channel-neutral reminder cancellation services and dispatch coordination."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    CancelReminderCommand,
)
from reminder_agent.application.conversations import (
    ConversationSession,
    end_session,
    load_session,
    pending_cancellation_from_session,
)
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.features.reminders.cancel.decisions import (
    CancellationDecision,
    CancellationRequest,
    CancellationResolution,
    CancellationScope,
    PendingCancellation,
    _request_occurrence_date,
    apply_cancellation,
    bounded_cancellation_context,
    classify_cancellation_resolution,
    decide_cancellation,
    find_candidates,
    find_partial_candidates,
    find_unmaterialized_occurrence_candidates,
    format_candidates,
    ground_model_cancellation_request,
    is_cancel_command,
    is_semantically_uncertain_cancellation_target,
    parse_cancellation_request,
)
from reminder_agent.features.reminders.cancel.messages import cancel_message
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()

def cancel_reminder_for_request(
    db: Session,
    *,
    request: ApplicationRequest,
    command: CancelReminderCommand,
    now: datetime,
) -> ApplicationResult:
    """Stage a validated, owner-scoped cancellation without committing."""

    if request.idempotency_key is None:
        return ApplicationResult(
            content=cancel_message("missing_idempotency_key"),
            response_kind="IDEMPOTENCY_KEY_REQUIRED",
        )
    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE or not user.timezone:
        return ApplicationResult(
            content="I couldn't verify your account.",
            response_kind="AUTHORIZATION_FAILURE",
        )

    reminder, occurrence_count = apply_cancellation(
        db,
        user_id=user.id,
        reminder_id=command.reminder_id,
        occurrence_id=command.occurrence_id,
        scope=command.scope,
        now=now,
        occurrence_date=command.occurrence_date,
    )
    if command.scope is CancellationScope.SERIES:
        occurrence_word = "occurrence" if occurrence_count == 1 else "occurrences"
        content = cancel_message(
            "recurring_series_cancelled",
            task=reminder.task,
            count=occurrence_count,
            occurrence_word=occurrence_word,
        )
        response_kind = "CANCELLATION_SERIES"
    elif reminder.schedule_type == "ONE_TIME":
        content = cancel_message("one_time_cancelled", task=reminder.task)
        response_kind = "CANCELLATION_REMINDER"
    else:
        content = cancel_message("recurring_occurrence_cancelled", task=reminder.task)
        response_kind = "CANCELLATION_OCCURRENCE"
    return ApplicationResult(
        content=content,
        response_kind=response_kind,
        resource_id=reminder.id,
        affected_count=occurrence_count,
    )


@dataclass(frozen=True)
class CancellationPreparation:
    """Channel-neutral cancellation decision prepared before mutation."""

    decision: CancellationDecision | None
    parsed_request: CancellationRequest | None
    pending_for_decision: PendingCancellation | None


def cancellation_command_for_decision(
    decision: CancellationDecision,
) -> CancelReminderCommand | None:
    """Build the channel-neutral mutation command for one resolved target."""

    if len(decision.candidate_ids) != 1 or decision.scope is None:
        return None
    return CancelReminderCommand(
        reminder_id=decision.candidate_ids[0],
        occurrence_id=decision.occurrence_id,
        scope=decision.scope,
        occurrence_date=(
            decision.request.occurrence_date if decision.request is not None else None
        ),
    )


@dataclass(frozen=True)
class CancellationDispatchResult[ResponseT]:
    """Updated local state and response from cancellation dispatch."""

    response: ResponseT | None
    pending: PendingCancellation | None
    session: ConversationSession | None


def prepare_cancellation(
    db: Session,
    *,
    user_id: UUID,
    text: str,
    timezone_name: str,
    now: datetime,
    pending: PendingCancellation | None,
    request_override: CancellationRequest | None = None,
    cancellation_interpreter: Any = None,
    recent_messages: list[str] | None = None,
) -> CancellationPreparation:
    """Resolve cancellation candidates without owning the write transaction."""

    cancellation_context = bounded_cancellation_context(recent_messages or [])
    parsed_request = None
    if pending is None or is_cancel_command(text):
        parsed = parse_cancellation_request(text, timezone_name=timezone_name, now=now)
        parsed_request = None if isinstance(parsed, str) else parsed
        pending_for_decision = None
    else:
        pending_for_decision = pending

    if pending_for_decision is not None:
        decision = decide_cancellation(
            db,
            user_id=user_id,
            text=text,
            timezone_name=timezone_name,
            now=now,
            pending=pending_for_decision,
        )
    elif parsed_request is not None:
        decision = decide_cancellation(
            db,
            user_id=user_id,
            text=text,
            timezone_name=timezone_name,
            now=now,
            request_override=parsed_request,
        )
    else:
        decision = None

    partial_candidates = []
    if (
        decision is not None
        and decision.kind in {"RESPONSE", "CLARIFY"}
        and parsed_request is not None
    ):
        partial_candidates = find_partial_candidates(
            db,
            user_id=user_id,
            target_text=parsed_request.target_text,
            occurrence_date=parsed_request.occurrence_date,
            timezone_name=timezone_name,
            include_empty_series=(parsed_request.scope is not CancellationScope.OCCURRENCE),
        )

    if len(partial_candidates) > 1 and parsed_request is not None:
        decision = CancellationDecision(
            "CLARIFY",
            cancel_message("partial_ambiguous_match")
            + "\n"
            + format_candidates(partial_candidates, timezone_name=timezone_name),
            request=parsed_request,
            candidate_ids=tuple(candidate.reminder_id for candidate in partial_candidates),
            scope=parsed_request.scope,
        )
        partial_candidates = []

    resolution = classify_cancellation_resolution(decision, parsed_request, partial_candidates)
    if resolution is CancellationResolution.UNCERTAIN_EXTRACTION and pending_for_decision is None:
        proposal = request_override
        interpreter_method = getattr(cancellation_interpreter, "interpret_cancellation", None)
        if interpreter_method is not None:
            occurrence_date = (
                parsed_request.occurrence_date
                if parsed_request is not None
                else _request_occurrence_date(text, now=now, timezone_name=timezone_name)
            )
            candidates = find_candidates(
                db,
                user_id=user_id,
                target_text=None,
                occurrence_date=occurrence_date,
                timezone_name=timezone_name,
                include_empty_series=(
                    occurrence_date is None
                    and (
                        parsed_request is None
                        or parsed_request.scope is not CancellationScope.OCCURRENCE
                    )
                ),
            )
            if occurrence_date is not None:
                seen_reminders = {candidate.reminder_id for candidate in candidates}
                candidates.extend(
                    candidate
                    for candidate in find_unmaterialized_occurrence_candidates(
                        db,
                        user_id=user_id,
                        target_text=None,
                        occurrence_date=occurrence_date,
                        timezone_name=timezone_name,
                        now=now,
                    )
                    if candidate.reminder_id not in seen_reminders
                )
            candidates = candidates[:12]
            candidate_context = [
                {
                    "task": candidate.task,
                    "schedule_type": candidate.schedule_type,
                    "due_at": candidate.due_at.isoformat() if candidate.due_at else "",
                }
                for candidate in candidates
            ]
            kwargs = {
                "timezone_name": timezone_name,
                "now": now,
                "recent_messages": cancellation_context,
                "cancellation_candidates": candidate_context,
            }
            try:
                parameters = inspect.signature(interpreter_method).parameters
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                proposal = interpreter_method(
                    text,
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key in parameters or accepts_kwargs
                    },
                )
            except Exception as exc:  # model failures must leave cancellation unchanged
                log_event(
                    logger,
                    logging.WARNING,
                    "cancellation_target_fallback_failed",
                    reason_type=type(exc).__name__,
                )
                proposal = None
        if isinstance(proposal, CancellationRequest):
            try:
                proposal = ground_model_cancellation_request(
                    text,
                    proposal,
                    timezone_name=timezone_name,
                    now=now,
                    recent_messages=cancellation_context,
                )
            except ValueError as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "cancellation_target_fallback_rejected",
                    reason_type=type(exc).__name__,
                )
                proposal = None
        if isinstance(proposal, CancellationRequest):
            decision = decide_cancellation(
                db,
                user_id=user_id,
                text=text,
                timezone_name=timezone_name,
                now=now,
                request_override=proposal,
            )
        elif decision is None:
            decision = CancellationDecision("RESPONSE", cancel_message("missing_target"))
        elif partial_candidates:
            decision = CancellationDecision(
                "RESPONSE",
                cancel_message("uncertain_match"),
            )
    elif (
        resolution is CancellationResolution.NO_MATCH
        and decision is not None
        and decision.kind == "RESPONSE"
        and is_semantically_uncertain_cancellation_target(parsed_request)
    ):
        decision = CancellationDecision(
            "RESPONSE",
            cancel_message("uncertain_match"),
        )

    return CancellationPreparation(
        decision=decision,
        parsed_request=parsed_request,
        pending_for_decision=pending_for_decision,
    )


def dispatch_cancellation[ResponseT](
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    now: datetime,
    session_ttl_seconds: int,
    update_id: int,
    pending: PendingCancellation | None,
    session: ConversationSession | None,
    is_list_request: bool,
    is_new_reminder: bool,
    is_reschedule: bool,
    is_cancel_command: bool,
    handle: Callable[[PendingCancellation | None], ResponseT | None],
) -> CancellationDispatchResult[ResponseT]:
    """Supersede stale clarification state and invoke the existing handler."""

    if pending is not None and (is_list_request or is_new_reminder or is_reschedule):
        db.rollback()
        with db.begin():
            locked_session = load_session(
                db,
                user_id=user_id,
                external_session_id=external_session_id,
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if (
                locked_session is not None
                and pending_cancellation_from_session(locked_session) is not None
            ):
                end_session(locked_session, now=now)
        pending = None
        session = None
        log_event(
            logger,
            logging.INFO,
            "cancellation_clarification_superseded",
            update_id=update_id,
        )

    response = handle(pending) if is_cancel_command or pending is not None else None
    return CancellationDispatchResult(
        response=response,
        pending=pending,
        session=session,
    )
