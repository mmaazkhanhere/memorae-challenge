"""Shared reminder creation workflow for authenticated voice calls."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    AuthenticatedRequestContext,
    CompleteReminderCommand,
    DeliveredReminderContext,
    RequestSource,
    RescheduleReminderCommand,
    SnoozeReminderCommand,
    VerifiedChannelIdentity,
)
from reminder_agent.application.conversations import (
    end_session,
    load_session,
    pending_cancellation_from_session,
    pending_completion_from_session,
    pending_from_session,
    pending_memory_from_session,
    pending_memory_lookup_from_session,
    pending_reschedule_from_session,
    pending_snooze_from_session,
    save_pending_clarification_session,
)
from reminder_agent.application.identity.context import authorized_user_for_request
from reminder_agent.application.pending_contracts import PendingCompletion, PendingSnooze
from reminder_agent.application.voice_call_authorization import resolve_vapi_call_context
from reminder_agent.features.intent.contracts import IntentInterpreter
from reminder_agent.features.intent.deterministic.routing import ConversationRoute, route_message
from reminder_agent.features.intent.model.router import IntentRouter
from reminder_agent.features.intent.policy import OUT_OF_SCOPE_RESPONSE
from reminder_agent.features.intent.routing import route_and_interpret
from reminder_agent.features.reminders.cancel.decisions import (
    PendingCancellation,
    is_cancel_command,
)
from reminder_agent.features.reminders.cancel.messages import cancel_message
from reminder_agent.features.reminders.cancel.service import (
    cancel_reminder_for_request,
    cancellation_command_for_decision,
    prepare_cancellation,
)
from reminder_agent.features.reminders.complete.operations import (
    match_completion_candidates,
    resolve_by_occurrence,
    resolve_latest_delivered,
)
from reminder_agent.features.reminders.create.presentation import (
    format_duplicate_reminder,
    format_next_occurrence,
    format_reminder_confirmation,
)
from reminder_agent.features.reminders.create.reminder_creation_persistence import ExistingReminder
from reminder_agent.features.reminders.create.service import run_reminder_creation_workflow
from reminder_agent.features.reminders.listing.service import (
    list_reminders_from_interpretation,
    summarize_reminders_from_interpretation,
)
from reminder_agent.features.reminders.reschedule.decisions import RescheduleRequest
from reminder_agent.features.reminders.reschedule.messages import choose_reschedule_message
from reminder_agent.features.reminders.reschedule.models import (
    PendingReschedule,
    RescheduleScope,
)
from reminder_agent.features.reminders.reschedule.parser import is_reschedule_command
from reminder_agent.features.reminders.reschedule.service import (
    prepare_reschedule,
    reschedule_reminder_for_user,
)
from reminder_agent.features.reminders.service import (
    complete_reminder_for_request,
    snooze_reminder_for_request,
)
from reminder_agent.features.reminders.snooze.messages import snooze_message
from reminder_agent.features.reminders.snooze.operations import (
    parse_snooze_duration,
    snooze_target_text,
)
from reminder_agent.infrastructure.database.models import ChannelIdentity, IdempotencyRecord
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.idempotency import (
    IdempotencyConflict,
    ensure_request_matches,
    find_idempotency_record,
    request_hash,
)

VOICE_CALL = "VOICE_CALL"
TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
CREATE_OPERATION = "CREATE_REMINDER"

logger = get_logger()


@dataclass(frozen=True)
class VoiceToolOutcome:
    """Provider-neutral result for one Vapi tool invocation."""

    response: ApplicationResult | None = None
    error: str | None = None


def _application_request(
    context: AuthenticatedRequestContext,
    *,
    call_id: str,
    tool_call_id: str,
    utterance: str,
) -> ApplicationRequest:
    return ApplicationRequest(
        context=context,
        message_text=utterance,
        idempotency_key=f"{call_id}:{tool_call_id}",
    )


def _request_for_identity(
    request: ApplicationRequest,
    *,
    identity: ChannelIdentity,
) -> ApplicationRequest:
    """Use the linked Telegram destination for reminder delivery operations.

    The phone identity authenticates the call.  Reminder notifications and
    delivered-occurrence actions, however, are Telegram-scoped, so the shared
    reminder services receive a request whose verified identity is the
    server-selected Telegram identity for the same canonical user.
    """

    return ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=request.context.source,
            canonical_user_id=request.context.canonical_user_id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type=identity.channel_type,
            ),
            external_session_id=request.context.external_session_id,
        ),
        message_text=request.message_text,
        idempotency_key=request.idempotency_key,
    )


def _with_idempotency(
    response: ApplicationResult,
    *,
    request: ApplicationRequest,
) -> ApplicationResult:
    """Attach the provider invocation key without changing service output."""

    return ApplicationResult(
        content=response.content,
        response_kind=response.response_kind,
        resource_id=response.resource_id,
        affected_count=response.affected_count,
        presentation_hint=response.presentation_hint,
        idempotency_key=request.idempotency_key,
        should_send=response.should_send,
        is_duplicate=response.is_duplicate,
        selection_context=response.selection_context,
    )


def _telegram_identity(db: Session, *, user_id: UUID) -> ChannelIdentity | None:
    return db.scalar(
        select(ChannelIdentity)
        .where(
            ChannelIdentity.user_id == user_id,
            ChannelIdentity.channel_type == TELEGRAM,
            ChannelIdentity.status == LINKED,
            ChannelIdentity.telegram_chat_id.is_not(None),
        )
        .order_by(ChannelIdentity.created_at, ChannelIdentity.id)
        .limit(1)
    )


def _record_response(
    db: Session,
    *,
    request: ApplicationRequest,
    operation: str,
    response: ApplicationResult,
) -> None:
    assert request.idempotency_key is not None
    db.add(
        IdempotencyRecord(
            source=RequestSource.VAPI.value,
            idempotency_key=request.idempotency_key,
            user_id=request.context.canonical_user_id,
            operation=operation,
            request_hash=request_hash(request),
            status="SUCCEEDED",
            # Vapi receives the result synchronously; there is no separate
            # outbound delivery worker for the spoken tool result.
            delivery_status="SENT",
            response={
                "text": response.content,
                "response_kind": response.response_kind,
                "request_text": request.message_text,
                "selection_context": list(response.selection_context or ()),
            },
        )
    )


def _response_from_record(record: IdempotencyRecord) -> ApplicationResult:
    response = record.response or {}
    raw_selection = response.get("selection_context")
    selection = (
        tuple(item for item in raw_selection if isinstance(item, dict))
        if isinstance(raw_selection, list)
        else None
    )
    return ApplicationResult(
        content=str(response.get("text", "I couldn't complete that request.")),
        response_kind=str(response.get("response_kind", "REPLAY")),
        idempotency_key=record.idempotency_key,
        should_send=True,
        is_duplicate=True,
        selection_context=selection,
    )


def _existing_response(db: Session, request: ApplicationRequest) -> ApplicationResult | None:
    record = find_idempotency_record(db, request, source=RequestSource.VAPI.value)
    if record is None:
        return None
    ensure_request_matches(record, request)
    return _response_from_record(record)


def _duplicate_confirmation(
    duplicate: ExistingReminder,
    *,
    timezone_name: str,
) -> str:
    local_due = duplicate.due_at.astimezone(ZoneInfo(timezone_name))
    return format_duplicate_reminder(
        duplicate.task,
        when=format_next_occurrence(local_due),
    )


def _result(
    *,
    request: ApplicationRequest,
    content: str | None = None,
    text: str | None = None,
    response_kind: str,
    duplicate: bool = False,
) -> ApplicationResult:
    rendered = content if content is not None else (text or "")
    return ApplicationResult(
        content=rendered,
        response_kind=response_kind,
        idempotency_key=request.idempotency_key,
        is_duplicate=duplicate,
    )


def _creation_response(
    db: Session,
    *,
    request: ApplicationRequest,
    route: ConversationRoute,
    interpretation: Any,
    now: datetime,
    session_ttl_seconds: int,
    user: Any,
    telegram_identity: ChannelIdentity,
) -> ApplicationResult:
    voice_session = load_session(
        db,
        user_id=user.id,
        external_session_id=request.context.external_session_id,
        now=now,
        ttl_seconds=session_ttl_seconds,
        for_update=True,
        channel_type=VOICE_CALL,
    )
    pending = pending_from_session(voice_session)
    if voice_session is not None and (
        pending_memory_from_session(voice_session) is not None
        or pending_memory_lookup_from_session(voice_session) is not None
    ):
        # A reminder tool call is an explicit new operation. Do not let a
        # previous memory clarification consume a later reminder turn.
        end_session(voice_session, now=now)

    if (
        route in {ConversationRoute.CANCELLATION, ConversationRoute.CANCEL_REQUEST}
        and pending is not None
    ) or interpretation.clear_context:
        if voice_session is not None:
            end_session(voice_session, now=now)
        return _result(
            request=request,
            content=interpretation.response_text or "Okay, I won't create that reminder.",
            response_kind="clarification_canceled",
        )

    if route not in {
        ConversationRoute.NEW_REQUEST,
        ConversationRoute.CLARIFICATION_REPLY,
        ConversationRoute.MODEL_FALLBACK,
    }:
        return _result(
            request=request,
            content=OUT_OF_SCOPE_RESPONSE,
            response_kind="unsupported_operation",
        )

    if interpretation.clarification is not None:
        save_pending_clarification_session(
            db,
            user_id=user.id,
            external_session_id=request.context.external_session_id,
            pending=interpretation.clarification.pending,
            now=now,
            channel_type=VOICE_CALL,
        )
        return _result(
            request=request,
            content=interpretation.clarification.question,
            response_kind="clarification",
        )

    intents = interpretation.resolved_intents
    if not intents or any(
        getattr(intent, "operation", "")
        not in {"CREATE_ONE_TIME_REMINDER", "CREATE_RECURRING_REMINDER"}
        for intent in intents
    ):
        return _result(
            request=request,
            content=interpretation.response_text or OUT_OF_SCOPE_RESPONSE,
            response_kind="clarification" if interpretation.response_text else "unsupported_operation",
        )

    creation = run_reminder_creation_workflow(
        db,
        intents=list(intents),
        user=user,
        # Reminder notifications created during a call always target the
        # user's linked Telegram destination, not the phone identity.
        identity=telegram_identity,
        locked_session=voice_session,
        timezone_name=user.timezone,
        now=now,
        reference=now,
        source_message_id=request.idempotency_key or request.context.external_session_id,
        stage_response=lambda **kwargs: _result(request=request, **kwargs),
        confirmation_formatter=format_reminder_confirmation,
    )
    if creation.failure is not None:
        return _result(
            request=request,
            content="Please provide a short reminder task and future time.",
            response_kind="schedule_validation_failure",
        )
    if creation.persistence is not None and creation.persistence.duplicate is not None:
        return _result(
            request=request,
            content=_duplicate_confirmation(
                creation.persistence.duplicate,
                timezone_name=user.timezone,
            ),
            response_kind="duplicate_reminder",
        )
    if creation.response is None:
        return _result(
            request=request,
            content="I couldn't create that reminder.",
            response_kind="persistence_failure",
        )
    return creation.response


def _voice_pending(db: Session, *, user_id: UUID, call_id: str, now: datetime, ttl: int) -> Any:
    return load_session(
        db,
        user_id=user_id,
        external_session_id=call_id,
        now=now,
        ttl_seconds=ttl,
        channel_type=VOICE_CALL,
    )


def _voice_recent_context(db: Session, *, user_id: UUID) -> tuple[list[str], list[dict[str, str]]]:
    """Load bounded Vapi conversation and list context from committed results."""

    records = db.scalars(
        select(IdempotencyRecord)
        .where(
            IdempotencyRecord.source == RequestSource.VAPI.value,
            IdempotencyRecord.user_id == user_id,
        )
        .order_by(IdempotencyRecord.created_at.desc())
        .limit(12)
    ).all()
    messages: list[str] = []
    selection: list[dict[str, str]] = []
    for record in reversed(records):
        if not isinstance(record.response, dict):
            continue
        request_text = record.response.get("request_text")
        response_text = record.response.get("text")
        if isinstance(request_text, str):
            messages.append(f"user: {request_text}")
        if isinstance(response_text, str):
            messages.append(f"assistant: {response_text}")
        raw_selection = record.response.get("selection_context")
        if isinstance(raw_selection, list):
            selection = [item for item in raw_selection if isinstance(item, dict)]
    return messages[-12:], selection


def _record_and_return(
    db: Session,
    *,
    request: ApplicationRequest,
    operation: str,
    response: ApplicationResult,
) -> VoiceToolOutcome:
    response = _with_idempotency(response, request=request)
    _record_response(db, request=request, operation=operation, response=response)
    return VoiceToolOutcome(response=response)


def _voice_list_response(
    db: Session,
    *,
    request: ApplicationRequest,
    interpretation: Any,
    now: datetime,
    operation: str,
) -> VoiceToolOutcome:
    response = (
        summarize_reminders_from_interpretation(
            db, request=request, interpretation=interpretation, now=now
        )
        if operation == "SUMMARY"
        else list_reminders_from_interpretation(
            db, request=request, interpretation=interpretation, now=now
        )
    )
    if response is None:
        response = ApplicationResult(
            content=OUT_OF_SCOPE_RESPONSE,
            response_kind="unsupported_operation",
        )
    db.rollback()
    with db.begin():
        return _record_and_return(
            db,
            request=request,
            operation=operation,
            response=response,
        )


def _reschedule_response(
    db: Session,
    *,
    request: ApplicationRequest,
    user: Any,
    telegram_identity: ChannelIdentity,
    interpreter: IntentInterpreter,
    text: str,
    now: datetime,
    session_ttl_seconds: int,
) -> VoiceToolOutcome:
    session = _voice_pending(
        db,
        user_id=user.id,
        call_id=request.context.external_session_id,
        now=now,
        ttl=session_ttl_seconds,
    )
    pending = pending_reschedule_from_session(session)
    recent_messages, selection_context = _voice_recent_context(db, user_id=user.id)
    preparation = prepare_reschedule(
        db,
        user_id=user.id,
        text=text,
        timezone_name=user.timezone,
        now=now,
        pending=pending,
        reschedule_interpreter=interpreter,
        recent_messages=recent_messages,
        selection_context=selection_context,
        update_id=None,
    )
    decision = preparation.decision
    if decision is None:
        return VoiceToolOutcome(error="Please tell me which reminder to move and when.")

    db.rollback()
    with db.begin():
        locked_session = _voice_pending(
            db,
            user_id=user.id,
            call_id=request.context.external_session_id,
            now=now,
            ttl=session_ttl_seconds,
        )
        if decision.kind == "CANCEL":
            if locked_session is not None:
                end_session(locked_session, now=now)
            response = ApplicationResult(
                content="Okay, I won't move that reminder.",
                response_kind="reschedule_canceled",
            )
        elif decision.kind == "RESPONSE":
            response = ApplicationResult(
                content=decision.text,
                response_kind="reschedule_rejected",
            )
        elif decision.kind == "CLARIFY":
            proposed = decision.request
            if proposed is None and pending is not None:
                proposed = RescheduleRequest(
                    target_text=pending.target_text,
                    due_at=pending.due_at,
                    date_explicit=pending.date_explicit,
                    offset_minutes=pending.offset_minutes,
                    scope=pending.scope,
                    source_text=pending.source_text,
                    occurrence_date=pending.occurrence_date,
                    destination_date=pending.destination_date,
                    destination_weekday=pending.destination_weekday,
                )
            if proposed is None:
                response = ApplicationResult(
                    content=decision.text,
                    response_kind="reschedule_clarification",
                )
            else:
                pending_record = PendingReschedule(
                    target_text=proposed.target_text,
                    due_at=proposed.due_at,
                    date_explicit=proposed.date_explicit,
                    offset_minutes=proposed.offset_minutes,
                    source_text=proposed.source_text,
                    timezone_name=user.timezone,
                    candidate_ids=decision.candidate_ids,
                    scope=proposed.scope,
                    occurrence_date=proposed.occurrence_date,
                    destination_date=proposed.destination_date,
                    destination_weekday=proposed.destination_weekday,
                )
                save_pending_clarification_session(
                    db,
                    user_id=user.id,
                    external_session_id=request.context.external_session_id,
                    pending=pending_record,
                    now=now,
                    channel_type=VOICE_CALL,
                )
                response = ApplicationResult(
                    content=decision.text,
                    response_kind="reschedule_clarification",
                )
        else:
            if decision.request is None or len(decision.candidate_ids) != 1:
                response = ApplicationResult(
                    content="Please tell me which reminder to move.",
                    response_kind="reschedule_clarification",
                )
            else:
                mutation = reschedule_reminder_for_user(
                    db,
                    user_id=user.id,
                    destination_identity_id=telegram_identity.id,
                    command=RescheduleReminderCommand(
                        reminder_id=decision.candidate_ids[0],
                        occurrence_id=decision.occurrence_id,
                        due_at=decision.request.due_at,
                        scope=decision.request.scope,
                        destination_weekday=decision.request.destination_weekday,
                    ),
                    now=now,
                )
                local_due = mutation.due_at.astimezone(ZoneInfo(user.timezone))
                if decision.request.scope is RescheduleScope.SERIES:
                    content = choose_reschedule_message(
                        "recurring_series_rescheduled",
                        task=mutation.reminder.task,
                        cadence=(
                            f"every {local_due:%A}"
                            if decision.request.destination_weekday is not None
                            else "the existing schedule"
                        ),
                        time=f"{local_due:%I:%M %p}",
                        timezone=f"{local_due:%Z}",
                        next_occurrence=local_due.strftime("%Y-%m-%d %H:%M %Z"),
                    )
                    kind = "RESCHEDULE_SERIES"
                else:
                    content = choose_reschedule_message(
                        "one_time_rescheduled",
                        task=mutation.reminder.task,
                        when=local_due.strftime("%Y-%m-%d %H:%M %Z"),
                    )
                    kind = "RESCHEDULE_REMINDER"
                if locked_session is not None:
                    end_session(locked_session, now=now)
                response = ApplicationResult(
                    content=content,
                    response_kind=kind,
                    resource_id=mutation.reminder.id,
                )
        return _record_and_return(
            db,
            request=request,
            operation="RESCHEDULE_REMINDER",
            response=response,
        )


def _cancellation_response(
    db: Session,
    *,
    request: ApplicationRequest,
    user: Any,
    interpreter: IntentInterpreter,
    text: str,
    now: datetime,
    session_ttl_seconds: int,
) -> VoiceToolOutcome:
    session = _voice_pending(
        db,
        user_id=user.id,
        call_id=request.context.external_session_id,
        now=now,
        ttl=session_ttl_seconds,
    )
    pending = pending_cancellation_from_session(session)
    preparation = prepare_cancellation(
        db,
        user_id=user.id,
        text=text,
        timezone_name=user.timezone,
        now=now,
        pending=pending,
        cancellation_interpreter=interpreter,
        recent_messages=[],
    )
    decision = preparation.decision
    if decision is None:
        return VoiceToolOutcome(error=cancel_message("missing_target"))

    db.rollback()
    with db.begin():
        locked_session = _voice_pending(
            db,
            user_id=user.id,
            call_id=request.context.external_session_id,
            now=now,
            ttl=session_ttl_seconds,
        )
        if decision.kind == "CANCEL":
            if locked_session is not None:
                end_session(locked_session, now=now)
            response = ApplicationResult(
                content="Okay, I won't cancel that reminder.",
                response_kind="cancellation_abandoned",
            )
        elif decision.kind == "RESPONSE":
            response = ApplicationResult(
                content=decision.text,
                response_kind="cancellation_response",
            )
        elif decision.kind == "CLARIFY":
            candidate_request = decision.request
            if candidate_request is None:
                response = ApplicationResult(
                    content=decision.text,
                    response_kind="cancellation_clarification",
                )
            else:
                save_pending_clarification_session(
                    db,
                    user_id=user.id,
                    external_session_id=request.context.external_session_id,
                    pending=PendingCancellation(
                        target_text=candidate_request.target_text,
                        candidate_ids=decision.candidate_ids,
                        scope=candidate_request.scope,
                        occurrence_date=candidate_request.occurrence_date,
                        timezone_name=user.timezone,
                    ),
                    now=now,
                    channel_type=VOICE_CALL,
                )
                response = ApplicationResult(
                    content=decision.text,
                    response_kind="cancellation_clarification",
                )
        else:
            command = cancellation_command_for_decision(decision)
            if command is None:
                response = ApplicationResult(
                    content=cancel_message("missing_target"),
                    response_kind="cancellation_clarification",
                )
            else:
                response = cancel_reminder_for_request(
                    db,
                    request=request,
                    command=command,
                    now=now,
                )
                if locked_session is not None:
                    end_session(locked_session, now=now)
        return _record_and_return(
            db,
            request=request,
            operation="CANCEL_REMINDER",
            response=response,
        )


def _delivered_candidates(
    db: Session,
    *,
    user_id: UUID,
    telegram_identity: ChannelIdentity,
    pending_ids: tuple[UUID, ...] = (),
) -> list[Any]:
    if pending_ids:
        candidates: list[Any] = []
        for occurrence_id in pending_ids:
            candidates.extend(
                resolve_by_occurrence(
                    db,
                    user_id=user_id,
                    identity_id=telegram_identity.id,
                    chat_id=telegram_identity.telegram_chat_id,
                    occurrence_id=occurrence_id,
                )
            )
        return candidates
    return resolve_latest_delivered(
        db,
        user_id=user_id,
        identity_id=telegram_identity.id,
        chat_id=telegram_identity.telegram_chat_id,
    )


def _completion_response(
    db: Session,
    *,
    request: ApplicationRequest,
    user: Any,
    telegram_identity: ChannelIdentity,
    target_text: str | None,
    now: datetime,
    session_ttl_seconds: int,
) -> VoiceToolOutcome:
    session = _voice_pending(
        db,
        user_id=user.id,
        call_id=request.context.external_session_id,
        now=now,
        ttl=session_ttl_seconds,
    )
    pending = pending_completion_from_session(session)
    candidates = _delivered_candidates(
        db,
        user_id=user.id,
        telegram_identity=telegram_identity,
        pending_ids=pending.candidate_occurrence_ids if pending is not None else (),
    )
    if target_text and len(candidates) > 1:
        candidates = match_completion_candidates(target_text, candidates)
    db.rollback()
    with db.begin():
        locked_session = _voice_pending(
            db,
            user_id=user.id,
            call_id=request.context.external_session_id,
            now=now,
            ttl=session_ttl_seconds,
        )
        if len(candidates) != 1:
            if candidates:
                labels = ", ".join(item[1].task for item in candidates[:3])
                text = f"Which reminder should I mark done: {labels}?"
                pending_value = PendingCompletion(
                    candidate_occurrence_ids=tuple(item[0].id for item in candidates)
                )
                save_pending_clarification_session(
                    db,
                    user_id=user.id,
                    external_session_id=request.context.external_session_id,
                    pending=pending_value,
                    now=now,
                    channel_type=VOICE_CALL,
                )
            else:
                text = "I couldn't find one delivered reminder to mark done."
            response = ApplicationResult(content=text, response_kind="completion_clarification")
        else:
            occurrence, _reminder, notification = candidates[0]
            response = complete_reminder_for_request(
                db,
                request=request,
                command=CompleteReminderCommand(
                    occurrence_id=occurrence.id,
                    delivery=DeliveredReminderContext(
                        identity_id=telegram_identity.id,
                        channel_type="TELEGRAM",
                        external_destination_id=str(telegram_identity.telegram_chat_id),
                        provider_message_id=notification.provider_message_id,
                    ),
                ),
                now=now,
            )
            if locked_session is not None:
                end_session(locked_session, now=now)
        return _record_and_return(
            db,
            request=request,
            operation="COMPLETE_REMINDER",
            response=response,
        )


def _snooze_response(
    db: Session,
    *,
    request: ApplicationRequest,
    user: Any,
    telegram_identity: ChannelIdentity,
    text: str,
    proposed_duration_seconds: int | None,
    now: datetime,
    session_ttl_seconds: int,
) -> VoiceToolOutcome:
    session = _voice_pending(
        db,
        user_id=user.id,
        call_id=request.context.external_session_id,
        now=now,
        ttl=session_ttl_seconds,
    )
    pending = pending_snooze_from_session(session)
    parsed = parse_snooze_duration(text)
    duration_seconds = (
        parsed.duration_seconds
        if parsed
        else proposed_duration_seconds
        if proposed_duration_seconds is not None
        else pending.duration_seconds
        if pending is not None
        else None
    )
    candidates = _delivered_candidates(
        db,
        user_id=user.id,
        telegram_identity=telegram_identity,
        pending_ids=pending.candidate_occurrence_ids if pending is not None else (),
    )
    target_text = snooze_target_text(text)
    if target_text and len(candidates) > 1:
        candidates = match_completion_candidates(target_text, candidates)
    db.rollback()
    with db.begin():
        locked_session = _voice_pending(
            db,
            user_id=user.id,
            call_id=request.context.external_session_id,
            now=now,
            ttl=session_ttl_seconds,
        )
        if len(candidates) != 1 or not isinstance(duration_seconds, int) or duration_seconds <= 0:
            if not candidates:
                response = ApplicationResult(
                    content=snooze_message("no_delivered_match"),
                    response_kind="snooze_response",
                )
            elif not isinstance(duration_seconds, int) or duration_seconds <= 0:
                response = ApplicationResult(
                    content=snooze_message("missing_duration"),
                    response_kind="snooze_clarification",
                )
            else:
                response = ApplicationResult(
                    content=snooze_message("missing_target"),
                    response_kind="snooze_clarification",
                )
            if candidates and (not isinstance(duration_seconds, int) or duration_seconds <= 0 or len(candidates) > 1):
                save_pending_clarification_session(
                    db,
                    user_id=user.id,
                    external_session_id=request.context.external_session_id,
                    pending=PendingSnooze(
                        candidate_occurrence_ids=tuple(item[0].id for item in candidates),
                        duration_seconds=duration_seconds if isinstance(duration_seconds, int) and duration_seconds > 0 else None,
                    ),
                    now=now,
                    channel_type=VOICE_CALL,
                )
        else:
            occurrence, _reminder, notification = candidates[0]
            response = snooze_reminder_for_request(
                db,
                request=request,
                command=SnoozeReminderCommand(
                    occurrence_id=occurrence.id,
                    duration_seconds=duration_seconds,
                    delivery=DeliveredReminderContext(
                        identity_id=telegram_identity.id,
                        channel_type="TELEGRAM",
                        external_destination_id=str(telegram_identity.telegram_chat_id),
                        provider_message_id=notification.provider_message_id,
                    ),
                ),
                now=now,
            )
            if locked_session is not None:
                end_session(locked_session, now=now)
        return _record_and_return(
            db,
            request=request,
            operation="SNOOZE_OCCURRENCE",
            response=response,
        )


def process_vapi_reminder_tool(
    db: Session,
    *,
    call_id: str,
    tool_call_id: str,
    utterance: str,
    interpreter: IntentInterpreter,
    intent_router: IntentRouter | None,
    now: datetime,
    session_ttl_seconds: int = 900,
) -> VoiceToolOutcome:
    """Process one authenticated Vapi reminder request.

    Vapi supplies only the caller's utterance.  Routing, extraction,
    authorization, pending-session handling, and state transitions remain in
    this application workflow and the existing reminder feature services.
    """

    call_id = call_id.strip()
    tool_call_id = tool_call_id.strip()
    utterance = utterance.strip()
    if (
        not call_id
        or not tool_call_id
        or len(call_id) > 255
        or len(tool_call_id) > 255
        or len(call_id) + len(tool_call_id) + 1 > 255
        or not utterance
    ):
        return VoiceToolOutcome(error="Please repeat the reminder request.")

    context = resolve_vapi_call_context(db, provider_call_id=call_id, now=now)
    if context is None:
        return VoiceToolOutcome(error="I couldn't verify this call.")
    phone_request = _application_request(
        context,
        call_id=call_id,
        tool_call_id=tool_call_id,
        utterance=utterance,
    )

    try:
        db.rollback()
        # The call authorization is the authority for the canonical user.
        # Reminder services then receive the server-selected linked Telegram
        # identity so delivery and delivered-occurrence operations remain
        # scoped to the user's actual Telegram destination.
        user = authorized_user_for_request(db, request=phone_request)
        if user is None or user.onboarding_status != ACTIVE or not user.timezone:
            return VoiceToolOutcome(error="I couldn't verify this call.")
        telegram_identity = _telegram_identity(db, user_id=user.id)
        if telegram_identity is None:
            return VoiceToolOutcome(
                error="Please finish Telegram setup before managing reminders on a call."
            )
        request = _request_for_identity(phone_request, identity=telegram_identity)
        existing = _existing_response(db, request)
        if existing is not None:
            return VoiceToolOutcome(response=existing)
        db.rollback()
        voice_session = _voice_pending(
            db,
            user_id=user.id,
            call_id=request.context.external_session_id,
            now=now,
            ttl=session_ttl_seconds,
        )
        pending_creation = pending_from_session(voice_session)
        pending_reschedule = pending_reschedule_from_session(voice_session)
        pending_cancellation = pending_cancellation_from_session(voice_session)
        pending_completion = pending_completion_from_session(voice_session)
        pending_snooze = pending_snooze_from_session(voice_session)
        recent_messages, selection_context = _voice_recent_context(db, user_id=user.id)

        deterministic_route = route_message(
            request.message_text,
            pending=pending_creation,
        ).route
        # The shared deterministic router intentionally leaves broad
        # reschedule language to the classifier.  Voice still needs an
        # explicit, unambiguous management verb to reach the reschedule
        # service when no classifier is configured.
        if (
            deterministic_route is ConversationRoute.MODEL_FALLBACK
            and is_reschedule_command(request.message_text)
            and re.search(r"\b(?:move|reschedule|change|update)\b", request.message_text, re.I)
        ):
            deterministic_route = ConversationRoute.RESCHEDULE_REQUEST

        if (
            deterministic_route is ConversationRoute.CANCEL_REQUEST
            and is_cancel_command(request.message_text)
            and voice_session is not None
            and any(
                pending is not None
                for pending in (
                    pending_reschedule,
                    pending_cancellation,
                    pending_completion,
                    pending_snooze,
                )
            )
        ):
            db.rollback()
            with db.begin():
                locked = _voice_pending(
                    db,
                    user_id=user.id,
                    call_id=request.context.external_session_id,
                    now=now,
                    ttl=session_ttl_seconds,
                )
                if locked is not None:
                    end_session(locked, now=now)
                return _record_and_return(
                    db,
                    request=request,
                    operation="CANCEL_PENDING",
                    response=ApplicationResult(
                        content="Okay, I won't continue that unfinished reminder request.",
                        response_kind="clarification_canceled",
                    ),
                )

        # Explicit read-only requests do not consume an older clarification.
        if deterministic_route in {
            ConversationRoute.LIST_REQUEST,
            ConversationRoute.SUMMARY_REQUEST,
        }:
            route, interpretation = route_and_interpret(
                interpreter,
                request.message_text,
                timezone_name=user.timezone,
                now=now,
                pending=pending_creation,
                user_name=user.name,
                intent_router=intent_router,
                recent_messages=recent_messages,
                selection_context=selection_context,
                deterministic_route=deterministic_route,
            )
            return _voice_list_response(
                db,
                request=request,
                interpretation=interpretation,
                now=now,
                operation=("SUMMARY" if route is ConversationRoute.SUMMARY_REQUEST else "LIST_REMINDERS"),
            )

        # A pending operation owns its follow-up unless the caller clearly
        # starts a new operation or asks for a read-only result.
        if pending_reschedule is not None and deterministic_route not in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.CANCEL_REQUEST,
            ConversationRoute.SNOOZE_REQUEST,
            ConversationRoute.COMPLETE_REQUEST,
        }:
            return _reschedule_response(
                db,
                request=request,
                user=user,
                telegram_identity=telegram_identity,
                interpreter=interpreter,
                text=request.message_text,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        if pending_cancellation is not None and deterministic_route not in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.RESCHEDULE_REQUEST,
            ConversationRoute.SNOOZE_REQUEST,
            ConversationRoute.COMPLETE_REQUEST,
        }:
            return _cancellation_response(
                db,
                request=request,
                user=user,
                interpreter=interpreter,
                text=request.message_text,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        if pending_completion is not None and deterministic_route not in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.RESCHEDULE_REQUEST,
            ConversationRoute.CANCEL_REQUEST,
            ConversationRoute.SNOOZE_REQUEST,
        }:
            return _completion_response(
                db,
                request=request,
                user=user,
                telegram_identity=telegram_identity,
                target_text=None,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        if pending_snooze is not None and deterministic_route not in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.RESCHEDULE_REQUEST,
            ConversationRoute.CANCEL_REQUEST,
            ConversationRoute.COMPLETE_REQUEST,
        }:
            return _snooze_response(
                db,
                request=request,
                user=user,
                telegram_identity=telegram_identity,
                text=request.message_text,
                proposed_duration_seconds=None,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )

        route, interpretation = route_and_interpret(
            interpreter,
            request.message_text,
            timezone_name=user.timezone,
            now=now,
            pending=pending_creation,
            user_name=user.name,
            intent_router=intent_router,
            recent_messages=recent_messages,
            selection_context=selection_context,
            deterministic_route=deterministic_route,
        )
        db.rollback()

        if route is ConversationRoute.RESCHEDULE_REQUEST:
            return _reschedule_response(
                db,
                request=request,
                user=user,
                telegram_identity=telegram_identity,
                interpreter=interpreter,
                text=request.message_text,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        if route is ConversationRoute.CANCEL_REQUEST:
            return _cancellation_response(
                db,
                request=request,
                user=user,
                interpreter=interpreter,
                text=request.message_text,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        if route is ConversationRoute.COMPLETE_REQUEST:
            return _completion_response(
                db,
                request=request,
                user=user,
                telegram_identity=telegram_identity,
                target_text=interpretation.completion_target_text,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        if route is ConversationRoute.SNOOZE_REQUEST:
            return _snooze_response(
                db,
                request=request,
                user=user,
                telegram_identity=telegram_identity,
                text=request.message_text,
                proposed_duration_seconds=interpretation.snooze_duration_seconds,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )

        # A reminder tool call is an explicit new reminder operation.  Clear
        # an unrelated memory or management clarification before creation;
        # the creation workflow itself retains its original pending context.
        if voice_session is not None and (
            pending_reschedule is not None
            or pending_cancellation is not None
            or pending_completion is not None
            or pending_snooze is not None
            or pending_memory_from_session(voice_session) is not None
            or pending_memory_lookup_from_session(voice_session) is not None
        ):
            db.rollback()
            with db.begin():
                locked = _voice_pending(
                    db,
                    user_id=user.id,
                    call_id=request.context.external_session_id,
                    now=now,
                    ttl=session_ttl_seconds,
                )
                if locked is not None:
                    end_session(locked, now=now)

        if route not in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.CLARIFICATION_REPLY,
            ConversationRoute.MODEL_FALLBACK,
        }:
            response = ApplicationResult(
                content=interpretation.response_text or OUT_OF_SCOPE_RESPONSE,
                response_kind="unsupported_operation",
            )
            db.rollback()
            with db.begin():
                return _record_and_return(
                    db,
                    request=request,
                    operation=CREATE_OPERATION,
                    response=response,
                )

        db.rollback()
        with db.begin():
            response = _creation_response(
                db,
                request=request,
                route=route,
                interpretation=interpretation,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                user=user,
                telegram_identity=telegram_identity,
            )
            return _record_and_return(
                db,
                request=request,
                operation=CREATE_OPERATION,
                response=response,
            )
    except IdempotencyConflict:
        raise
    except IntegrityError:
        db.rollback()
        replay = _existing_response(db, request)
        if replay is not None:
            return VoiceToolOutcome(response=replay)
        raise
    except Exception:
        db.rollback()
        log_event(
            logger,
            logging.ERROR,
            "vapi_reminder_tool_failed",
            call_id=call_id,
            tool_call_id=tool_call_id,
            exc_info=True,
        )
        return VoiceToolOutcome(error="I couldn't complete that reminder request. Please try again.")


__all__ = ["VoiceToolOutcome", "process_vapi_reminder_tool"]
