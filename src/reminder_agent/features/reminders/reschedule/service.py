"""Pending rescheduling dispatch coordination.

The rescheduling handler continues to own parsing, mutation, and transaction
behavior.  This module only coordinates replacement requests around it.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    RescheduleReminderCommand,
)
from reminder_agent.application.conversations import (
    ConversationSession,
    end_session,
    load_session,
    pending_reschedule_from_session,
    save_pending_clarification_session,
    telegram_session_id,
)
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.application.response_staging import (
    ReminderReply,
    _authorized_telegram_identity,
    _existing_record,
    _recover_from_idempotency_race,
    _reply_from_record,
    _store_response,
)
from reminder_agent.features.intent.context import CONVERSATION_CONTEXT_TURNS
from reminder_agent.features.intent.pipeline import ReminderIntentInterpreter
from reminder_agent.features.reminders.reschedule.decisions import (
    PendingReschedule,
    PendingRescheduleOperation,
    PendingRescheduleProposal,
    RescheduleConflict,
    RescheduleDecision,
    RescheduleRequest,
    RescheduleScope,
    apply_pending_reschedule_proposal,
    decide_reschedule,
    find_all_candidates,
    find_candidates,
    has_unsupported_series_cadence_change,
    is_reschedule_command,
    parse_pending_reschedule_parameters,
    parse_reschedule_request,
    pending_candidates,
)
from reminder_agent.features.reminders.reschedule.decisions import (
    apply_reschedule as decisions_apply_reschedule,
)
from reminder_agent.features.reminders.reschedule.messages import (
    choose_reschedule_message,
    reschedule_message,
)
from reminder_agent.features.reminders.reschedule.models import RescheduleClarification
from reminder_agent.infrastructure.database.models import ChannelIdentity, Reminder, User
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()


@dataclass(frozen=True)
class RescheduleMutationResult:
    """Business outcome of a staged rescheduling mutation."""

    reminder: Reminder
    due_at: datetime


def reschedule_reminder_for_user(
    db: Session,
    *,
    user_id: UUID,
    destination_identity_id: UUID,
    command: RescheduleReminderCommand,
    now: datetime,
    destination_channel_type: str = "TELEGRAM",
    destination_external_id: str | None = None,
) -> RescheduleMutationResult:
    """Stage an owned rescheduling mutation without taking transaction ownership."""

    if command.due_at is None:
        raise ValueError("reschedule time could not be resolved")
    reminder, due_at = decisions_apply_reschedule(
        db,
        user_id=user_id,
        destination_identity_id=destination_identity_id,
        destination_channel_type=destination_channel_type,
        destination_external_id=destination_external_id,
        reminder_id=command.reminder_id,
        occurrence_id=command.occurrence_id,
        due_at=command.due_at,
        scope=command.scope,
        now=now,
        destination_weekday=command.destination_weekday,
    )
    return RescheduleMutationResult(reminder=reminder, due_at=due_at)


def reschedule_reminder_for_request(
    db: Session,
    *,
    request: ApplicationRequest,
    command: RescheduleReminderCommand,
    now: datetime,
) -> ApplicationResult:
    """Stage an owner- and destination-verified reschedule."""

    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE or not user.timezone:
        return ApplicationResult(
            content=reschedule_message("authorization_failed"),
            response_kind="AUTHORIZATION_FAILURE",
        )
    mutation = reschedule_reminder_for_user(
        db,
        user_id=user.id,
        destination_identity_id=request.context.verified_identity.identity_id,
        command=command,
        now=now,
        destination_channel_type=request.context.verified_identity.channel_type,
    )
    timezone_name = user.timezone
    if command.scope is not None and command.scope.value == "SERIES":
        local_due = mutation.due_at.astimezone(ZoneInfo(timezone_name))
        content = choose_reschedule_message(
            "recurring_series_rescheduled",
            task=mutation.reminder.task,
            cadence=(
                f"every {local_due:%A}"
                if command.destination_weekday is not None
                else "the existing schedule"
            ),
            time=f"{local_due:%I:%M %p}",
            timezone=f"{local_due:%Z}",
            next_occurrence=_local_display(mutation.due_at, timezone_name),
        )
        response_kind = "RESCHEDULE_SERIES"
    else:
        content = choose_reschedule_message(
            "one_time_rescheduled",
            task=mutation.reminder.task,
            when=_local_display(mutation.due_at, timezone_name),
        )
        response_kind = "RESCHEDULE_REMINDER"
    return ApplicationResult(
        content=content,
        response_kind=response_kind,
        resource_id=mutation.reminder.id,
    )


def _local_display(due_at: datetime, timezone_name: str) -> str:
    return due_at.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M %Z")

def _handle_reschedule_text(
    db: Session,
    update: Any,
    *,
    user_id: UUID,
    identity_id: UUID,
    telegram_user_id: int,
    user: User,
    pending: PendingReschedule | None,
    now: datetime,
    session_ttl_seconds: int,
    force: bool = False,
    defer_to_classifier: bool = False,
    reschedule_interpreter: ReminderIntentInterpreter | None = None,
    recent_messages: list[str] | None = None,
    selection_context: tuple[dict[str, str], ...] = (),
    pending_session_version: int | None = None,
) -> ReminderReply | None:
    """Resolve and apply VS-09 in one workflow-owned write transaction.

    This handler runs before the creation interpreter is invoked.
    """

    assert update.message is not None
    text = update.message.text or ""
    if not force and not is_reschedule_command(text) and pending is None:
        return None

    preparation = prepare_reschedule(
        db,
        user_id=user.id,
        text=text,
        timezone_name=user.timezone or "UTC",
        now=now,
        pending=pending,
        # The pre-handler is the deterministic stage. Incomplete initial
        # requests must go through intent classification before extraction.
        reschedule_interpreter=(
            reschedule_interpreter
            if force or pending is not None or not defer_to_classifier
            else None
        ),
        recent_messages=recent_messages,
        selection_context=list(selection_context),
        update_id=update.update_id,
    )
    if (
        defer_to_classifier
        and not force
        and pending is None
        and isinstance(preparation.parsed_request, str)
        and not has_unsupported_series_cadence_change(text)
    ):
        db.rollback()
        return None
    decision = preparation.decision
    pending_proposal = preparation.pending_proposal
    db.rollback()
    try:
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                return _reply_from_record(existing)
            locked_user = db.get(User, user_id)
            locked_identity = db.get(ChannelIdentity, identity_id)
            if not _authorized_telegram_identity(
                locked_user,
                locked_identity,
                telegram_user_id=telegram_user_id,
                chat_id=update.message.chat.id,
            ):
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=reschedule_message("authorization_failed"),
                    response_kind="authorization_failure",
                    operation="RESCHEDULE_REMINDER",
                )
            assert locked_user is not None and locked_identity is not None
            locked_session = load_session(
                db,
                user_id=user_id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            locked_pending = pending_reschedule_from_session(locked_session)
            if locked_pending is not None:
                candidate_id_override = None
                effective_pending = locked_pending
                proposal_is_current = (
                    pending_proposal is not None
                    and pending is not None
                    and locked_session is not None
                    and pending_session_version is not None
                    and locked_session.version == pending_session_version
                )
                if proposal_is_current and pending_proposal is not None:
                    if (
                        pending_proposal.candidate_id is not None
                        and pending_proposal.candidate_id not in locked_pending.candidate_ids
                    ):
                        decision = RescheduleDecision(
                            "CLARIFY",
                            choose_reschedule_message("ambiguous_pending_selection"),
                            candidate_ids=locked_pending.candidate_ids,
                        )
                    elif pending_proposal.operation is PendingRescheduleOperation.CANCEL_PENDING:
                        decision = RescheduleDecision("CANCEL", "Okay, I won't move that reminder.")
                    elif pending_proposal.operation in {
                        PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
                        PendingRescheduleOperation.SELECT_AND_UPDATE,
                    }:
                        effective_pending = apply_pending_reschedule_proposal(
                            locked_pending,
                            pending_proposal,
                            timezone_name=locked_user.timezone or "UTC",
                        )
                        candidate_id_override = pending_proposal.candidate_id
                        decision = decide_reschedule(
                            db,
                            user_id=user_id,
                            text=text,
                            timezone_name=locked_user.timezone or "UTC",
                            now=now,
                            pending=effective_pending,
                            candidate_id_override=candidate_id_override,
                        )
                    elif pending_proposal.operation is PendingRescheduleOperation.SELECT_CANDIDATE:
                        candidate_id_override = pending_proposal.candidate_id
                        decision = decide_reschedule(
                            db,
                            user_id=user_id,
                            text=text,
                            timezone_name=locked_user.timezone or "UTC",
                            now=now,
                            pending=effective_pending,
                            candidate_id_override=candidate_id_override,
                        )
                    else:
                        # START_NEW_OPERATION and NEEDS_CLARIFICATION cannot
                        # clear or replace a trusted pending reschedule.
                        decision = decide_reschedule(
                            db,
                            user_id=user_id,
                            text=text,
                            timezone_name=locked_user.timezone or "UTC",
                            now=now,
                            pending=effective_pending,
                        )
                else:
                    decision = decide_reschedule(
                        db,
                        user_id=user_id,
                        text=text,
                        timezone_name=locked_user.timezone or "UTC",
                        now=now,
                        pending=effective_pending,
                    )
                ordinal_match = re.search(
                    r"\b(?:the\s+)?(first|one|second|two|third|three|1|2|3)\b",
                    text.strip().lower().rstrip(".!?"),
                )
                log_event(
                    logger,
                    logging.INFO,
                    "reschedule_pending_clarification_resolved",
                    update_id=update.update_id,
                    candidate_count=len(locked_pending.candidate_ids),
                    ordinal=ordinal_match.group(1) if ordinal_match is not None else None,
                    selected_candidate_count=len(decision.candidate_ids),
                    decision=decision.kind,
                    pending_due_at=(
                        effective_pending.due_at.isoformat()
                        if effective_pending.due_at is not None
                        else None
                    ),
                    pending_offset_minutes=effective_pending.offset_minutes,
                    pending_scope=(
                        effective_pending.scope.value
                        if effective_pending.scope is not None
                        else None
                    ),
                )

            if decision.kind == "CANCEL":
                if locked_session is not None:
                    end_session(locked_session, now=now)
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="reschedule_canceled",
                    operation="RESCHEDULE_REMINDER",
                )

            if decision.kind == "RESPONSE":
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="reschedule_rejected",
                    operation="RESCHEDULE_REMINDER",
                )

            if decision.kind == "CLARIFY":
                request = decision.request
                if request is None and effective_pending is not None:
                    request = RescheduleRequest(
                        target_text=effective_pending.target_text,
                        due_at=effective_pending.due_at,
                        date_explicit=effective_pending.date_explicit,
                        offset_minutes=effective_pending.offset_minutes,
                        scope=effective_pending.scope,
                        source_text=effective_pending.source_text,
                        occurrence_date=effective_pending.occurrence_date,
                        destination_date=effective_pending.destination_date,
                        destination_weekday=effective_pending.destination_weekday,
                    )
                if request is None:
                    return _store_response(
                        db,
                        update=update,
                        user_id=user_id,
                        text=decision.text,
                        response_kind="reschedule_clarification",
                        operation="RESCHEDULE_REMINDER",
                    )
                pending_record = PendingReschedule(
                    target_text=request.target_text,
                    due_at=request.due_at,
                    date_explicit=request.date_explicit,
                    offset_minutes=request.offset_minutes,
                    source_text=request.source_text,
                    timezone_name=locked_user.timezone or "UTC",
                    candidate_ids=decision.candidate_ids,
                    scope=request.scope,
                    occurrence_date=request.occurrence_date,
                    destination_date=request.destination_date,
                    destination_weekday=request.destination_weekday,
                )
                save_pending_clarification_session(
                    db,
                    user_id=user_id,
                    external_session_id=telegram_session_id(update.message.chat.id),
                    pending=pending_record,
                    now=now,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "reschedule_clarification_saved",
                    update_id=update.update_id,
                    candidate_count=len(decision.candidate_ids),
                    due_at=request.due_at.isoformat() if request.due_at is not None else None,
                    offset_minutes=request.offset_minutes,
                    scope=request.scope.value if request.scope is not None else None,
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="reschedule_clarification",
                    operation="RESCHEDULE_REMINDER",
                )

            assert decision.kind == "MUTATE" and decision.request is not None
            if len(decision.candidate_ids) != 1:
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=reschedule_message("missing_reminder_target"),
                    response_kind="reschedule_clarification",
                    operation="RESCHEDULE_REMINDER",
                )
            mutation = reschedule_reminder_for_user(
                db,
                user_id=user_id,
                destination_identity_id=identity_id,
                command=RescheduleReminderCommand(
                    reminder_id=decision.candidate_ids[0],
                    occurrence_id=decision.occurrence_id,
                    due_at=decision.request.due_at,
                    scope=decision.request.scope,
                    destination_weekday=decision.request.destination_weekday,
                ),
                now=now,
            )
            reminder, due_at = mutation.reminder, mutation.due_at
            log_event(
                logger,
                logging.INFO,
                "reschedule_mutation_applied",
                update_id=update.update_id,
                due_at=due_at.isoformat(),
                scope=decision.request.scope.value if decision.request.scope is not None else None,
            )
            if locked_session is not None:
                end_session(locked_session, now=now)
            if decision.request.scope is RescheduleScope.SERIES:
                local_time = due_at.astimezone(ZoneInfo(locked_user.timezone or "UTC"))
                confirmation = choose_reschedule_message(
                    "recurring_series_rescheduled",
                    task=reminder.task,
                    cadence=(
                        f"every {local_time:%A}"
                        if decision.request.destination_weekday is not None
                        else "the existing schedule"
                    ),
                    time=f"{local_time:%I:%M %p}",
                    timezone=f"{local_time:%Z}",
                    next_occurrence=_local_display(
                        due_at, locked_user.timezone or "UTC"
                    ),
                )
            else:
                local_display = _local_display(due_at, locked_user.timezone or "UTC")
                confirmation = choose_reschedule_message(
                    "one_time_rescheduled", task=reminder.task, when=local_display
                )
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                text=confirmation,
                response_kind="reschedule_confirmation",
                operation="RESCHEDULE_REMINDER",
            )
    except IntegrityError:
        duplicate = _recover_from_idempotency_race(db, update)
        if duplicate is not None:
            return duplicate
        raise
    except (RescheduleConflict, ValueError) as exc:
        log_event(
            logger,
            logging.WARNING,
            "reschedule_rejected",
            update_id=update.update_id,
            reason_type=type(exc).__name__,
            reason=str(exc),
        )
        db.rollback()
        with db.begin():
            failed_session = load_session(
                db,
                user_id=user_id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if (
                failed_session is not None
                and pending_reschedule_from_session(failed_session) is not None
            ):
                end_session(failed_session, now=now)
                log_event(
                    logger,
                    logging.INFO,
                    "reschedule_clarification_cleared_after_rejection",
                    update_id=update.update_id,
                    reason_type=type(exc).__name__,
                )
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                text=choose_reschedule_message("reschedule_failed", reason=str(exc)),
                response_kind="reschedule_conflict",
                operation="RESCHEDULE_REMINDER",
            )


@dataclass(frozen=True)
class ReschedulePreparation:
    """Channel-neutral rescheduling decision prepared before mutation."""

    decision: RescheduleDecision
    parsed_request: RescheduleRequest | str | None
    pending_proposal: PendingRescheduleProposal | None


@dataclass(frozen=True)
class RescheduleDispatchResult[ResponseT]:
    """Updated local state and response from rescheduling dispatch."""

    response: ResponseT | None
    pending: PendingReschedule | None
    session: ConversationSession | None
    observed_session_version: int | None


def prepare_reschedule(
    db: Session,
    *,
    user_id: UUID,
    text: str,
    timezone_name: str,
    now: datetime,
    pending: PendingReschedule | None,
    reschedule_interpreter: Any = None,
    recent_messages: list[str] | None = None,
    selection_context: list[dict[str, str]] | None = None,
    update_id: int | None = None,
) -> ReschedulePreparation:
    """Resolve targets, proposals, and validation without owning writes."""

    unsupported_cadence_change = pending is None and has_unsupported_series_cadence_change(text)

    parsed_request = None
    if pending is None:
        parsed_request = parse_reschedule_request(
            text,
            timezone_name=timezone_name,
            now=now,
        )
        log_event(
            logger,
            logging.INFO,
            "reschedule_deterministic_parse",
            update_id=update_id,
            parsed=isinstance(parsed_request, RescheduleRequest),
            result_type=type(parsed_request).__name__,
            result_length=len(parsed_request) if isinstance(parsed_request, str) else None,
            due_at=(
                parsed_request.due_at.isoformat()
                if isinstance(parsed_request, RescheduleRequest)
                and parsed_request.due_at is not None
                else None
            ),
            offset_minutes=(
                parsed_request.offset_minutes
                if isinstance(parsed_request, RescheduleRequest)
                else None
            ),
            scope=(
                parsed_request.scope.value
                if isinstance(parsed_request, RescheduleRequest)
                and parsed_request.scope is not None
                else None
            ),
        )
    deterministic_pending_proposal = (
        parse_pending_reschedule_parameters(
            text,
            timezone_name=timezone_name,
            now=now,
            destination_date=pending.destination_date,
        )
        if pending is not None
        else None
    )
    pending_for_decision = (
        apply_pending_reschedule_proposal(
            pending,
            deterministic_pending_proposal,
            timezone_name=timezone_name,
        )
        if pending is not None and deterministic_pending_proposal is not None
        else pending
    )
    decision = decide_reschedule(
        db,
        user_id=user_id,
        text=text,
        timezone_name=timezone_name,
        now=now,
        pending=pending_for_decision,
        request_override=(
            parsed_request if isinstance(parsed_request, RescheduleRequest) else None
        ),
    )
    if unsupported_cadence_change:
        decision = RescheduleDecision(
            "RESPONSE", "Changing monthly or yearly recurrence dates is not supported yet."
        )
    pending_proposal: PendingRescheduleProposal | None = deterministic_pending_proposal
    if (
        pending is not None
        and decision.kind == "CLARIFY"
        and deterministic_pending_proposal is None
        and not (decision.request is not None and decision.request.scope is not pending.scope)
        and reschedule_interpreter is not None
        and hasattr(reschedule_interpreter, "interpret_pending_reschedule")
    ):
        db.rollback()
        candidate_context = [
            {
                "candidate_id": str(candidate.reminder_id),
                "task": candidate.task,
                "due_local": candidate.due_at.astimezone(ZoneInfo(timezone_name)).isoformat(
                    timespec="minutes"
                ),
                "schedule_type": candidate.schedule_type,
            }
            for candidate in pending_candidates(db, user_id=user_id, pending=pending)
        ]
        proposal_kwargs = {
            "timezone_name": timezone_name,
            "now": now,
            "pending_reschedule": pending.model_dump(mode="json"),
            "reschedule_candidates": candidate_context,
            "recent_messages": (recent_messages or [])[-CONVERSATION_CONTEXT_TURNS:],
        }
        try:
            method = reschedule_interpreter.interpret_pending_reschedule
            parameters = inspect.signature(method).parameters
            proposal = method(
                text,
                **{key: value for key, value in proposal_kwargs.items() if key in parameters},
            )
            if isinstance(proposal, PendingRescheduleProposal):
                pending_proposal = proposal
        except Exception as exc:  # provider errors always fall back to clarification
            log_event(
                logger,
                logging.WARNING,
                "reschedule_pending_llm_proposal_rejected",
                update_id=update_id,
                reason_type=type(exc).__name__,
            )
        log_event(
            logger,
            logging.INFO,
            "reschedule_pending_llm_proposal",
            update_id=update_id,
            accepted=pending_proposal is not None,
        )
    if (
        isinstance(parsed_request, str)
        and not unsupported_cadence_change
        and reschedule_interpreter is not None
    ):
        db.rollback()
        candidate_context = [
            {
                "candidate_id": str(candidate.reminder_id),
                "task": candidate.task,
                "due_local": candidate.due_at.astimezone(ZoneInfo(timezone_name)).isoformat(
                    timespec="minutes"
                ),
                "schedule_type": candidate.schedule_type,
            }
            for candidate in find_all_candidates(db, user_id=user_id)
        ]
        proposal_kwargs = {
            "timezone_name": timezone_name,
            "now": now,
            "recent_messages": (recent_messages or [])[-CONVERSATION_CONTEXT_TURNS:],
            "reschedule_candidates": candidate_context,
            "selection_context": selection_context or [],
        }
        method = reschedule_interpreter.interpret_reschedule
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        proposal = method(
            text,
            **{
                key: value
                for key, value in proposal_kwargs.items()
                if key in parameters or accepts_kwargs
            },
        )
        log_event(
            logger,
            logging.INFO,
            "reschedule_llm_proposal",
            update_id=update_id,
            result_type=type(proposal).__name__,
            accepted=isinstance(proposal, (RescheduleRequest, RescheduleClarification)),
            response_length=len(proposal) if isinstance(proposal, str) else None,
            due_at=(
                proposal.due_at.isoformat()
                if isinstance(proposal, RescheduleRequest) and proposal.due_at is not None
                else None
            ),
            offset_minutes=(
                proposal.offset_minutes if isinstance(proposal, RescheduleRequest) else None
            ),
            scope=(
                proposal.scope.value
                if isinstance(proposal, RescheduleRequest) and proposal.scope is not None
                else None
            ),
        )
        if isinstance(proposal, RescheduleRequest):
            decision = decide_reschedule(
                db,
                user_id=user_id,
                text=text,
                timezone_name=timezone_name,
                now=now,
                pending=pending,
                request_override=proposal,
            )
        elif isinstance(proposal, RescheduleClarification):
            candidates = find_candidates(
                db,
                user_id=user_id,
                target_text=proposal.request.target_text,
                occurrence_date=proposal.request.occurrence_date,
                timezone_name=timezone_name,
            )
            if len(candidates) == 1:
                decision = RescheduleDecision(
                    "CLARIFY",
                    proposal.question,
                    request=proposal.request,
                    candidate_ids=(candidates[0].reminder_id,),
                )
            elif len(candidates) > 1:
                decision = RescheduleDecision(
                    "CLARIFY",
                    reschedule_message("multiple_matching_reminders"),
                    request=proposal.request,
                    candidate_ids=tuple(candidate.reminder_id for candidate in candidates),
                )
            else:
                decision = RescheduleDecision(
                    "RESPONSE", reschedule_message("no_matching_reminder")
                )
        elif isinstance(proposal, str) and proposal.strip():
            decision = RescheduleDecision("RESPONSE", proposal.strip())

    return ReschedulePreparation(
        decision=decision,
        parsed_request=parsed_request,
        pending_proposal=pending_proposal,
    )


def dispatch_reschedule[ResponseT](
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    now: datetime,
    session_ttl_seconds: int,
    update_id: int,
    pending: PendingReschedule | None,
    session: ConversationSession | None,
    observed_session_version: int | None,
    is_list_request: bool,
    is_summary_request: bool,
    is_new_reminder: bool,
    is_reschedule: bool,
    handle: Callable[[PendingReschedule | None, int | None], ResponseT | None],
) -> RescheduleDispatchResult[ResponseT]:
    """Coordinate replacement requests before invoking reschedule handling."""

    if pending is not None and (is_list_request or is_summary_request):
        # A read-only schedule query is not an answer to the old reschedule
        # question. Leave that clarification available for a later reply and
        # let the normal listing flow handle this message.
        pending = None
    elif pending is not None and (is_new_reminder or is_reschedule):
        # An explicit reminder or reschedule command starts a separate request.
        # It must not be treated as an ordinal/scope answer to an older
        # clarification, or that clarification will hijack the new request.
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
                and pending_reschedule_from_session(locked_session) is not None
            ):
                end_session(locked_session, now=now)
        pending = None
        observed_session_version = None
        log_event(
            logger,
            logging.INFO,
            "reschedule_clarification_superseded",
            update_id=update_id,
            replacement_route="RESCHEDULE_REQUEST" if is_reschedule else "NEW_REQUEST",
        )

    response = handle(pending, observed_session_version)
    return RescheduleDispatchResult(
        response=response,
        pending=pending,
        session=session,
        observed_session_version=observed_session_version,
    )
