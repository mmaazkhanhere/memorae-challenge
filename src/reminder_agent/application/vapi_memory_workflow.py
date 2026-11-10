"""Memory operations for authenticated Vapi calls.

The Vapi channel supplies only an utterance and an already-authorized call ID.
Intent routing, extraction, ownership checks, persistence, and clarification
state remain in the shared memory workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    RequestSource,
)
from reminder_agent.application.conversations import (
    end_session,
    load_session,
    pending_memory_from_session,
    pending_memory_lookup_from_session,
    upsert_clarification_session,
)
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.application.voice_call_authorization import resolve_vapi_call_context
from reminder_agent.application.voice_call_workflow import VoiceToolOutcome
from reminder_agent.features.intent.contracts import IntentInterpreter
from reminder_agent.features.intent.deterministic.routing import ConversationRoute, route_message
from reminder_agent.features.intent.model.router import IntentRouter
from reminder_agent.features.intent.policy import OUT_OF_SCOPE_RESPONSE
from reminder_agent.features.intent.routing import route_and_interpret
from reminder_agent.features.memory.messages import lookup_memory_message, save_memory_message
from reminder_agent.features.memory.retrieve.answering import memory_answer_preserves_exact_values
from reminder_agent.features.memory.retrieve.contracts import (
    MemoryRetrievalPolicy,
)
from reminder_agent.features.memory.retrieve.persistence import lookup_memory_fact
from reminder_agent.features.memory.retrieve.workflow import run_memory_lookup_workflow
from reminder_agent.features.memory.saving import run_memory_save_workflow, save_memory_for_request
from reminder_agent.infrastructure.database.models import IdempotencyRecord
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.infrastructure.providers.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
)
from reminder_agent.shared.idempotency import (
    IdempotencyConflict,
    ensure_request_matches,
    find_idempotency_record,
    request_hash,
)
from reminder_agent.shared.memory_text import extract_memory_content, extract_memory_lookup_query

VOICE_CALL = "VOICE_CALL"
VAPI_SOURCE = RequestSource.VAPI.value
MAX_CALL_ID_LENGTH = 255
MAX_UTTERANCE_LENGTH = 4000

logger = get_logger()
# The project owner explicitly opted Vapi into sensitive-memory indexing and
# disclosure. Telegram continues to use its deny-by-default policy.
VAPI_EMBEDDING_PRIVACY_POLICY = EmbeddingPrivacyPolicy(allow_sensitive_external=True)


@dataclass(frozen=True)
class _VapiMemoryResponses:
    """Adapt shared memory outcomes to Vapi's committed tool-result contract."""

    db: Session
    request: ApplicationRequest

    def existing(self) -> ApplicationResult | None:
        record = find_idempotency_record(self.db, self.request, source=VAPI_SOURCE)
        if record is None:
            return None
        ensure_request_matches(record, self.request)
        response = record.response or {}
        return ApplicationResult(
            content=str(response.get("text", "I couldn't complete that request.")),
            response_kind=str(response.get("response_kind", "REPLAY")),
            idempotency_key=self.request.idempotency_key,
            should_send=True,
            is_duplicate=True,
        )

    def recover_from_race(self) -> ApplicationResult | None:
        self.db.rollback()
        return self.existing()

    def direct(self, text: str) -> ApplicationResult:
        return ApplicationResult(
            content=text,
            response_kind="direct",
            idempotency_key=self.request.idempotency_key,
        )

    def stage(
        self,
        *,
        user_id,
        text: str | None = None,
        durable_response=None,
        response_kind: str,
        operation: str,
    ) -> ApplicationResult:
        content = text if text is not None else durable_response.content
        key = self.request.idempotency_key
        if key is None:
            raise ValueError("Vapi memory operations require an idempotency key")
        result = ApplicationResult(
            content=content,
            response_kind=response_kind,
            idempotency_key=key,
        )
        self.db.add(
            IdempotencyRecord(
                source=VAPI_SOURCE,
                idempotency_key=key,
                user_id=user_id,
                operation=operation,
                request_hash=request_hash(self.request),
                status="SUCCEEDED",
                delivery_status="SENT",
                # Memory values and raw call utterances must not be retained in
                # the durable response payload.
                response={"text": result.content, "response_kind": response_kind},
            )
        )
        return result


def _request_for_call(
    db: Session,
    *,
    call_id: str,
    tool_call_id: str,
    utterance: str,
    now: datetime,
) -> ApplicationRequest | None:
    context = resolve_vapi_call_context(db, provider_call_id=call_id, now=now)
    if context is None:
        return None
    return ApplicationRequest(
        context=context,
        message_text=utterance,
        idempotency_key=f"{call_id}:{tool_call_id}",
    )


def _valid_inputs(call_id: str, tool_call_id: str, utterance: str) -> bool:
    return bool(
        call_id
        and tool_call_id
        and utterance
        and len(call_id) <= MAX_CALL_ID_LENGTH
        and len(tool_call_id) <= MAX_CALL_ID_LENGTH
        and len(call_id) + len(tool_call_id) + 1 <= MAX_CALL_ID_LENGTH
        and len(utterance) <= MAX_UTTERANCE_LENGTH
    )


def _pending_session(
    db: Session,
    *,
    request: ApplicationRequest,
    now: datetime,
    ttl_seconds: int,
):
    return load_session(
        db,
        user_id=request.context.canonical_user_id,
        external_session_id=request.context.external_session_id,
        now=now,
        ttl_seconds=ttl_seconds,
        channel_type=VOICE_CALL,
    )


def _cancel_pending(
    db: Session,
    *,
    request: ApplicationRequest,
    now: datetime,
    ttl_seconds: int,
    responses: _VapiMemoryResponses,
    operation: str,
) -> ApplicationResult | None:
    session = _pending_session(db, request=request, now=now, ttl_seconds=ttl_seconds)
    if session is None:
        return None
    db.rollback()
    with db.begin():
        existing = responses.existing()
        if existing is not None:
            return existing
        locked = _pending_session(db, request=request, now=now, ttl_seconds=ttl_seconds)
        if locked is not None:
            end_session(locked, now=now)
        return responses.stage(
            user_id=request.context.canonical_user_id,
            text=save_memory_message("canceled")
            if operation == "SAVE_MEMORY"
            else "Okay, I won't look that up.",
            response_kind="memory_canceled",
            operation=operation,
        )


def _save_memory_response(
    db: Session,
    *,
    request: ApplicationRequest,
    interpreter: IntentInterpreter,
    intent_router: IntentRouter | None,
    now: datetime,
    session_ttl_seconds: int,
    embedding_provider: EmbeddingProvider | None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy,
) -> ApplicationResult:
    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE or not user.timezone:
        return ApplicationResult(
            content=save_memory_message("authorization_failed"),
            response_kind="authorization_failed",
        )

    responses = _VapiMemoryResponses(db=db, request=request)
    session = _pending_session(
        db, request=request, now=now, ttl_seconds=session_ttl_seconds
    )
    pending = pending_memory_from_session(session)
    normalized = request.message_text.casefold().strip(" .!?")
    if pending is not None and normalized in {
        "cancel",
        "cancel it",
        "never mind",
        "nevermind",
        "not now",
    }:
        canceled = _cancel_pending(
            db,
            request=request,
            now=now,
            ttl_seconds=session_ttl_seconds,
            responses=responses,
            operation="SAVE_MEMORY",
        )
        assert canceled is not None
        return canceled

    explicit_content = extract_memory_content(request.message_text)
    if pending is not None and explicit_content is not None:
        pending = None
    if pending is not None and explicit_content is None:
        # A plain follow-up answers the saved-memory question. Explicit new
        # reminder/lookup requests must not be swallowed as fact content.
        route = route_message(request.message_text, pending=None).route
        if route in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.RETRIEVE_MEMORY_REQUEST,
        }:
            with db.begin():
                locked = _pending_session(
                    db, request=request, now=now, ttl_seconds=session_ttl_seconds
                )
                if locked is not None:
                    end_session(locked, now=now)
                return responses.stage(
                    user_id=user.id,
                    text=OUT_OF_SCOPE_RESPONSE,
                    response_kind="unsupported_operation",
                    operation="SAVE_MEMORY",
                )

    if pending is None:
        route, interpretation = route_and_interpret(
            interpreter,
            request.message_text,
            timezone_name=user.timezone,
            now=now,
            pending=None,
            user_name=user.name,
            intent_router=intent_router,
            recent_messages=[],
        )
        if route is not ConversationRoute.SAVE_MEMORY_REQUEST:
            return responses.direct(OUT_OF_SCOPE_RESPONSE)
        model_content = (
            interpretation.memory_fact.content
            if interpretation.memory_fact is not None and explicit_content is None
            else None
        )
        return run_memory_save_workflow(
            db,
            request=request,
            user_id=user.id,
            pending=None,
            text=request.message_text,
            model_content=model_content,
            from_model=model_content is not None,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            responses=responses,
            upsert_clarification_session=upsert_clarification_session,
            save_memory_operation=save_memory_for_request,
            embedding_provider=embedding_provider,
            embedding_privacy_policy=VAPI_EMBEDDING_PRIVACY_POLICY,
            conversation_channel_type=VOICE_CALL,
            event_context={"call_id": request.context.external_session_id},
        )

    return run_memory_save_workflow(
        db,
        request=request,
        user_id=user.id,
        pending=pending,
        text=request.message_text,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        responses=responses,
        upsert_clarification_session=upsert_clarification_session,
        save_memory_operation=save_memory_for_request,
        embedding_provider=embedding_provider,
        embedding_privacy_policy=VAPI_EMBEDDING_PRIVACY_POLICY,
        conversation_channel_type=VOICE_CALL,
        event_context={"call_id": request.context.external_session_id},
    )


def _lookup_memory_response(
    db: Session,
    *,
    request: ApplicationRequest,
    interpreter: IntentInterpreter,
    intent_router: IntentRouter | None,
    now: datetime,
    session_ttl_seconds: int,
    memory_answerer: Any | None,
    embedding_provider: EmbeddingProvider | None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy,
    memory_retrieval_policy: MemoryRetrievalPolicy,
) -> ApplicationResult:
    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE or not user.timezone:
        return ApplicationResult(
            content=lookup_memory_message("authorization_failed"),
            response_kind="authorization_failed",
        )

    responses = _VapiMemoryResponses(db=db, request=request)
    session = _pending_session(
        db, request=request, now=now, ttl_seconds=session_ttl_seconds
    )
    pending = pending_memory_lookup_from_session(session)
    normalized = request.message_text.casefold().strip(" .!?")
    if pending is not None and normalized in {
        "cancel",
        "cancel it",
        "never mind",
        "nevermind",
        "not now",
    }:
        canceled = _cancel_pending(
            db,
            request=request,
            now=now,
            ttl_seconds=session_ttl_seconds,
            responses=responses,
            operation="RETRIEVE_MEMORY",
        )
        assert canceled is not None
        return canceled

    deterministic_query = extract_memory_lookup_query(request.message_text)
    if pending is not None and deterministic_query is None:
        new_route = route_message(request.message_text, pending=None).route
        if new_route in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.SAVE_MEMORY_REQUEST,
        }:
            db.rollback()
            with db.begin():
                locked = _pending_session(
                    db, request=request, now=now, ttl_seconds=session_ttl_seconds
                )
                if locked is not None:
                    end_session(locked, now=now)
            return responses.direct(OUT_OF_SCOPE_RESPONSE)
        query = request.message_text
        interpretation = None
    else:
        pending = None
        route, interpretation = route_and_interpret(
            interpreter,
            request.message_text,
            timezone_name=user.timezone,
            now=now,
            pending=None,
            user_name=user.name,
            intent_router=intent_router,
            recent_messages=[],
        )
        if route is not ConversationRoute.RETRIEVE_MEMORY_REQUEST:
            return responses.direct(OUT_OF_SCOPE_RESPONSE)
        query = deterministic_query or (
            interpretation.memory_query.query
            if interpretation.memory_query is not None
            else None
        )

    return run_memory_lookup_workflow(
        db,
        user_id=user.id,
        external_session_id=request.context.external_session_id,
        channel_type=VOICE_CALL,
        pending=pending,
        query=query,
        source_text=request.message_text,
        from_model=bool(interpretation is not None and deterministic_query is None),
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        responses=responses,
        upsert_clarification_session=upsert_clarification_session,
        memory_answerer=memory_answerer,
        embedding_provider=embedding_provider,
        embedding_privacy_policy=VAPI_EMBEDDING_PRIVACY_POLICY,
        memory_retrieval_policy=memory_retrieval_policy,
        lookup_memory_operation=lookup_memory_fact,
        answer_preserves_exact_values_fn=memory_answer_preserves_exact_values,
        allow_sensitive_facts=True,
        event_context={"call_id": request.context.external_session_id},
    )


def _process(
    db: Session,
    *,
    call_id: str,
    tool_call_id: str,
    utterance: str,
    now: datetime,
    operation: str,
    interpreter: IntentInterpreter,
    intent_router: IntentRouter | None,
    memory_answerer: Any | None,
    embedding_provider: EmbeddingProvider | None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy,
    memory_retrieval_policy: MemoryRetrievalPolicy,
    session_ttl_seconds: int,
) -> VoiceToolOutcome:
    call_id = call_id.strip()
    tool_call_id = tool_call_id.strip()
    utterance = utterance.strip()
    if not _valid_inputs(call_id, tool_call_id, utterance):
        return VoiceToolOutcome(error="Please repeat that memory request.")
    request = _request_for_call(
        db,
        call_id=call_id,
        tool_call_id=tool_call_id,
        utterance=utterance,
        now=now,
    )
    if request is None:
        return VoiceToolOutcome(error="I couldn't verify this call.")
    try:
        db.rollback()
        if operation == "SAVE_MEMORY":
            result = _save_memory_response(
                db,
                request=request,
                interpreter=interpreter,
                intent_router=intent_router,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                embedding_provider=embedding_provider,
                embedding_privacy_policy=embedding_privacy_policy,
            )
        else:
            result = _lookup_memory_response(
                db,
                request=request,
                interpreter=interpreter,
                intent_router=intent_router,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                memory_answerer=memory_answerer,
                embedding_provider=embedding_provider,
                embedding_privacy_policy=embedding_privacy_policy,
                memory_retrieval_policy=memory_retrieval_policy,
            )
        return VoiceToolOutcome(response=result)
    except IdempotencyConflict:
        raise
    except Exception:
        db.rollback()
        log_event(
            logger,
            logging.ERROR,
            "vapi_memory_tool_failed",
            call_id=call_id,
            tool_call_id=tool_call_id,
            operation=operation,
            exc_info=True,
        )
        return VoiceToolOutcome(error="I couldn't complete that memory request.")


def process_vapi_memory_save_tool(db: Session, **kwargs) -> VoiceToolOutcome:
    return _process(db, operation="SAVE_MEMORY", **kwargs)


def process_vapi_memory_lookup_tool(db: Session, **kwargs) -> VoiceToolOutcome:
    return _process(db, operation="RETRIEVE_MEMORY", **kwargs)


__all__ = ["process_vapi_memory_lookup_tool", "process_vapi_memory_save_tool"]
