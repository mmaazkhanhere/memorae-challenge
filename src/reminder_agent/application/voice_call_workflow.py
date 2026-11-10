"""Shared reminder creation workflow for authenticated voice calls."""

from __future__ import annotations

import logging
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
    RequestSource,
)
from reminder_agent.application.conversations import (
    end_session,
    load_session,
    pending_from_session,
    pending_memory_from_session,
    pending_memory_lookup_from_session,
    save_pending_clarification_session,
)
from reminder_agent.application.identity.context import authorized_user_for_request
from reminder_agent.application.voice_call_authorization import resolve_vapi_call_context
from reminder_agent.features.intent.contracts import IntentInterpreter
from reminder_agent.features.intent.deterministic.routing import ConversationRoute
from reminder_agent.features.intent.model.router import IntentRouter
from reminder_agent.features.intent.policy import OUT_OF_SCOPE_RESPONSE
from reminder_agent.features.intent.routing import route_and_interpret
from reminder_agent.features.reminders.create.presentation import (
    format_duplicate_reminder,
    format_next_occurrence,
    format_reminder_confirmation,
)
from reminder_agent.features.reminders.create.reminder_creation_persistence import ExistingReminder
from reminder_agent.features.reminders.create.service import run_reminder_creation_workflow
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
            },
        )
    )


def _response_from_record(record: IdempotencyRecord) -> ApplicationResult:
    response = record.response or {}
    return ApplicationResult(
        content=str(response.get("text", "I couldn't complete that request.")),
        response_kind=str(response.get("response_kind", "REPLAY")),
        idempotency_key=record.idempotency_key,
        should_send=True,
        is_duplicate=True,
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
    """Process one authenticated Vapi reminder request transactionally."""

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
    request = _application_request(
        context,
        call_id=call_id,
        tool_call_id=tool_call_id,
        utterance=utterance,
    )

    try:
        db.rollback()
        existing = _existing_response(db, request)
        if existing is not None:
            return VoiceToolOutcome(response=existing)
        db.rollback()

        # Assemble identity and pending conversation context before invoking
        # the interpreter.  The deterministic stage may fall back to an
        # external classifier/extractor, so no database transaction is held
        # across that provider call.
        user = authorized_user_for_request(db, request=request)
        if user is None or user.onboarding_status != ACTIVE or not user.timezone:
            return VoiceToolOutcome(error="I couldn't verify this call.")
        telegram_identity = _telegram_identity(db, user_id=user.id)
        if telegram_identity is None:
            return VoiceToolOutcome(
                error="Please finish Telegram setup before creating reminders on a call."
            )
        voice_session = load_session(
            db,
            user_id=user.id,
            external_session_id=request.context.external_session_id,
            now=now,
            ttl_seconds=session_ttl_seconds,
            channel_type=VOICE_CALL,
        )
        pending = pending_from_session(voice_session)
        route, interpretation = route_and_interpret(
            interpreter,
            request.message_text,
            timezone_name=user.timezone,
            now=now,
            pending=pending,
            user_name=user.name,
            intent_router=intent_router,
            recent_messages=[],
        )
        db.rollback()

        with db.begin():
            user = authorized_user_for_request(db, request=request)
            if user is None or user.onboarding_status != ACTIVE or not user.timezone:
                return VoiceToolOutcome(error="I couldn't verify this call.")
            telegram_identity = _telegram_identity(db, user_id=user.id)
            if telegram_identity is None:
                return VoiceToolOutcome(
                    error="Please finish Telegram setup before creating reminders on a call."
                )
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
            _record_response(
                db,
                request=request,
                operation=CREATE_OPERATION,
                response=response,
            )
            return VoiceToolOutcome(response=response)
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
        return VoiceToolOutcome(error="I couldn't create that reminder. Please try again.")


__all__ = ["VoiceToolOutcome", "process_vapi_reminder_tool"]
