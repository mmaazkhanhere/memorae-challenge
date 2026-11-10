"""Memory-save workflow orchestration shared by channel adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol, TypeVar
from uuid import UUID

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    DurableResponse,
    SaveMemoryCommand,
)
from reminder_agent.application.conversations import (
    end_session,
    load_session,
    pending_memory_from_session,
)
from reminder_agent.features.memory.create.service import save_memory_for_request
from reminder_agent.features.memory.create.messages import memory_message
from reminder_agent.features.memory.create.validation import validated_memory_content
from reminder_agent.features.memory.service import index_memory_fact_for_owner
from reminder_agent.infrastructure.embeddings import EmbeddingPrivacyPolicy, EmbeddingProvider
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.memory_contracts import PendingMemory

ResponseT_co = TypeVar("ResponseT_co", covariant=True)
logger = get_logger()
DEFAULT_EMBEDDING_PRIVACY_POLICY = EmbeddingPrivacyPolicy()


class MemoryWorkflowResponses(Protocol[ResponseT_co]):
    """Adapter-owned response and idempotency operations used by the workflow."""

    def existing(self) -> ResponseT_co | None: ...

    def recover_from_race(self) -> ResponseT_co | None: ...

    def direct(self, text: str) -> ResponseT_co: ...

    def stage(
        self,
        *,
        user_id: UUID,
        text: str | None = None,
        durable_response: DurableResponse | None = None,
        response_kind: str,
        operation: str,
    ) -> ResponseT_co: ...


def _clarification_response[ResponseT_co](
    db: Session,
    *,
    request: ApplicationRequest,
    user_id: UUID,
    pending: PendingMemory | None,
    now: datetime,
    session_ttl_seconds: int,
    clarification_text: str | None,
    responses: MemoryWorkflowResponses[ResponseT_co],
    upsert_clarification_session: Callable[..., object],
) -> ResponseT_co:
    db.rollback()
    with db.begin():
        existing = responses.existing()
        if existing is not None:
            return existing
        locked_session = load_session(
            db,
            user_id=user_id,
            external_session_id=request.context.external_session_id,
            now=now,
            ttl_seconds=session_ttl_seconds,
            for_update=True,
            channel_type=request.context.verified_identity.channel_type,
        )
        if pending is not None and pending_memory_from_session(locked_session) is None:
            return responses.stage(
                user_id=user_id,
                text=memory_message("memory_clarification_expired"),
                response_kind="memory_clarification_expired",
                operation="SAVE_MEMORY",
            )
        upsert_clarification_session(
            db,
            user_id=user_id,
            external_session_id=request.context.external_session_id,
            pending=PendingMemory(),
            now=now,
            channel_type=request.context.verified_identity.channel_type,
        )
        return responses.stage(
            user_id=user_id,
            text=clarification_text or memory_message("memory_clarification"),
            response_kind="memory_clarification",
            operation="SAVE_MEMORY",
        )


def _save_transaction[ResponseT_co](
    db: Session,
    *,
    request: ApplicationRequest,
    user_id: UUID,
    pending: PendingMemory | None,
    content: str,
    now: datetime,
    session_ttl_seconds: int,
    responses: MemoryWorkflowResponses[ResponseT_co],
    save_memory_operation: Callable[..., ApplicationResult],
    log_event_fn: Callable[..., object],
    event_context: Mapping[str, object],
) -> tuple[ResponseT_co, UUID | None, bool]:
    db.rollback()
    with db.begin():
        existing = responses.existing()
        if existing is not None:
            return existing, None, False
        result = save_memory_operation(
            db,
            request=request,
            command=SaveMemoryCommand(content=content),
            now=now,
        )
        if result.response_kind != "MEMORY_CONFIRMATION":
            return responses.direct(result.content), None, False
        assert result.resource_id is not None
        locked_session = load_session(
            db,
            user_id=user_id,
            external_session_id=request.context.external_session_id,
            now=now,
            ttl_seconds=session_ttl_seconds,
            for_update=True,
            channel_type=request.context.verified_identity.channel_type,
        )
        if pending is not None and pending_memory_from_session(locked_session) is None:
            return (
                responses.stage(
                    user_id=user_id,
                    text=memory_message("memory_clarification_expired"),
                    response_kind="memory_clarification_expired",
                    operation="SAVE_MEMORY",
                ),
                None,
                False,
            )
        if locked_session is not None:
            end_session(locked_session, now=now)
        log_event_fn(
            logger,
            logging.INFO,
            "memory_fact_saved",
            **event_context,
            content_length=len(content),
        )
        return (
            responses.stage(
                user_id=user_id,
                durable_response=result.durable_response,
                response_kind="memory_confirmation",
                operation="SAVE_MEMORY",
            ),
            result.resource_id,
            True,
        )


def _index_committed_fact(
    db: Session,
    *,
    fact_id: UUID,
    user_id: UUID,
    embedding_provider: EmbeddingProvider,
    embedding_privacy_policy: EmbeddingPrivacyPolicy,
    now: datetime,
    log_event_fn: Callable[..., object],
    event_context: Mapping[str, object],
) -> None:
    try:
        with db.begin():
            embedding_status = index_memory_fact_for_owner(
                db,
                fact_id=fact_id,
                user_id=user_id,
                embedding_provider=embedding_provider,
                privacy_policy=embedding_privacy_policy,
                now=now,
            )
        log_event_fn(
            logger,
            logging.INFO,
            "memory_embedding_indexing_completed",
            **event_context,
            embedding_status=embedding_status,
        )
    except Exception as exc:
        db.rollback()
        # Saving already committed. Keep the fact PENDING for backfill and
        # never turn an indexing outage into a false save failure.
        log_event_fn(
            logger,
            logging.WARNING,
            "memory_embedding_indexing_deferred",
            **event_context,
            reason_type=type(exc).__name__,
        )


def _failed_save_response[ResponseT_co](
    db: Session,
    *,
    responses: MemoryWorkflowResponses[ResponseT_co],
    log_event_fn: Callable[..., object],
    event_context: Mapping[str, object],
    reason: str,
    exception: Exception | None = None,
) -> ResponseT_co:
    db.rollback()
    context = {**event_context, "reason": reason}
    if exception is not None:
        context["reason_type"] = type(exception).__name__
    log_event_fn(logger, logging.ERROR, "memory_fact_save_failed", **context)
    return responses.direct(memory_message("memory_save_failed"))


def run_memory_save_workflow[ResponseT_co](
    db: Session,
    *,
    request: ApplicationRequest,
    user_id: UUID,
    pending: PendingMemory | None,
    text: str,
    now: datetime,
    session_ttl_seconds: int,
    responses: MemoryWorkflowResponses[ResponseT_co],
    upsert_clarification_session: Callable[..., object],
    save_memory_operation: Callable[..., ApplicationResult] = save_memory_for_request,
    model_content: str | None = None,
    from_model: bool = False,
    clarification_text: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    log_event_fn: Callable[..., object] = log_event,
    event_context: Mapping[str, object] | None = None,
) -> ResponseT_co:
    """Save a memory or persist its clarification using owned transactions."""

    event_context = event_context or {}
    content = validated_memory_content(
        text=text,
        pending=pending,
        model_content=model_content,
        from_model=from_model,
    )
    if content is None:
        return _clarification_response(
            db,
            request=request,
            user_id=user_id,
            pending=pending,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            clarification_text=clarification_text,
            responses=responses,
            upsert_clarification_session=upsert_clarification_session,
        )
    try:
        response, fact_id, should_index = _save_transaction(
            db,
            request=request,
            user_id=user_id,
            pending=pending,
            content=content,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            responses=responses,
            save_memory_operation=save_memory_operation,
            log_event_fn=log_event_fn,
            event_context=event_context,
        )
        if should_index and embedding_provider is not None:
            assert fact_id is not None
            _index_committed_fact(
                db,
                fact_id=fact_id,
                user_id=user_id,
                embedding_provider=embedding_provider,
                embedding_privacy_policy=embedding_privacy_policy,
                now=now,
                log_event_fn=log_event_fn,
                event_context=event_context,
            )
        return response
    except IntegrityError:
        duplicate = responses.recover_from_race()
        if duplicate is not None:
            return duplicate
        return _failed_save_response(
            db,
            responses=responses,
            log_event_fn=log_event_fn,
            event_context=event_context,
            reason="integrity_error",
        )
    except SQLAlchemyError as exc:
        return _failed_save_response(
            db,
            responses=responses,
            log_event_fn=log_event_fn,
            event_context=event_context,
            reason="database_error",
            exception=exc,
        )
    except Exception as exc:
        return _failed_save_response(
            db,
            responses=responses,
            log_event_fn=log_event_fn,
            event_context=event_context,
            reason="unexpected_error",
            exception=exc,
        )
