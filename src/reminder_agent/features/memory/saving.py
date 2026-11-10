"""Memory-save workflow orchestration shared by channel adapters."""

from __future__ import annotations

import logging
import re
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
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.features.memory.messages import save_memory_message
from reminder_agent.features.memory.persistence import index_memory_fact_for_owner, save_memory_fact
from reminder_agent.features.memory.validation import is_memory_content_grounded
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.infrastructure.providers.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
)
from reminder_agent.shared.memory_contracts import PendingMemory, SaveMemoryFact
from reminder_agent.shared.memory_text import extract_memory_content

ResponseT_co = TypeVar("ResponseT_co", covariant=True)
logger = get_logger()
DEFAULT_EMBEDDING_PRIVACY_POLICY = EmbeddingPrivacyPolicy()

_INCOMPLETE_FACT_PHRASE = re.compile(
    r"^\s*(?:my|the|our|your)\s+"
    r"(?:[\w'-]+\s+){0,6}"
    r"(?:number|code|password|passcode|pin|email|e-mail|address|phone|"
    r"phone\s+number|name|birthday|birth\s+date|favorite\s+food|"
    r"favourite\s+food|favorite\s+color|favourite\s+colour|parking\s+slot|"
    r"locker\s+number)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_INCOMPLETE_FACT_CLAUSE = re.compile(r"(?:\bis\b|\bare\b|\bequals\b|[:=])\s*[?.!]*$", re.IGNORECASE)


def save_memory_for_request(
    db: Session,
    *,
    request: ApplicationRequest,
    command: SaveMemoryCommand,
    now: datetime,
) -> ApplicationResult:
    """Stage one authorized, owner-scoped memory fact without committing."""

    if request.idempotency_key is None:
        return ApplicationResult(
            content=save_memory_message("idempotency_error"),
            response_kind="IDEMPOTENCY_KEY_REQUIRED",
        )
    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE:
        return ApplicationResult(
            content=save_memory_message("authorization_failed"),
            response_kind="AUTHORIZATION_FAILURE",
        )
    saved = save_memory_fact(db, user_id=user.id, content=command.content, now=now)
    return ApplicationResult(
        content=save_memory_message("confirmation", fact=command.content),
        response_kind="MEMORY_CONFIRMATION",
        resource_id=saved.fact_id,
    )


def _is_incomplete_fact_phrase(content: str) -> bool:
    """Reject a subject-only fact such as ``my locker number``."""

    normalized = content.strip(" .!?\t")
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    if (
        len(tokens) <= 3
        and tokens
        and tokens[0].casefold() in {"my", "the", "our", "your"}
        and not any(char.isdigit() for char in normalized)
        and not re.search(r"\b(?:is|are|equals)\b|[:=]", normalized, re.IGNORECASE)
        and "@" not in normalized
    ):
        return True
    return (
        _INCOMPLETE_FACT_PHRASE.fullmatch(content) is not None
        or _INCOMPLETE_FACT_CLAUSE.search(content) is not None
    )


def validated_memory_content(
    *,
    text: str,
    pending: PendingMemory | None,
    model_content: str | None,
    from_model: bool,
) -> str | None:
    """Return validated, grounded content or require a non-mutating clarification."""

    content = model_content if from_model else (extract_memory_content(text) if pending is None else text.strip())
    if content is None:
        return None
    try:
        content = SaveMemoryFact(content=content).content
    except (TypeError, ValueError):
        return None
    if _is_incomplete_fact_phrase(content):
        return None
    if from_model and not is_memory_content_grounded(text, content):
        return None
    return content


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
    conversation_channel_type: str,
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
            channel_type=conversation_channel_type,
        )
        if pending is None and locked_session is not None:
            # A new memory request supersedes any older pending operation. Do
            # not leave stale clarification state active after it commits.
            end_session(locked_session, now=now)
        if pending is not None and pending_memory_from_session(locked_session) is None:
            return responses.stage(
                user_id=user_id,
                text=save_memory_message("clarification_expired"),
                response_kind="memory_clarification_expired",
                operation="SAVE_MEMORY",
            )
        upsert_clarification_session(
            db,
            user_id=user_id,
            external_session_id=request.context.external_session_id,
            pending=PendingMemory(),
            now=now,
            channel_type=conversation_channel_type,
        )
        return responses.stage(
            user_id=user_id,
            text=clarification_text or save_memory_message("clarification"),
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
    conversation_channel_type: str,
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
            channel_type=conversation_channel_type,
        )
        if pending is None and locked_session is not None:
            end_session(locked_session, now=now)
        if pending is not None and pending_memory_from_session(locked_session) is None:
            return (
                responses.stage(
                    user_id=user_id,
                    text=save_memory_message("clarification_expired"),
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
    return responses.direct(save_memory_message("save_failed"))


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
    conversation_channel_type: str | None = None,
) -> ResponseT_co:
    """Save a memory or persist its clarification using owned transactions."""

    event_context = event_context or {}
    conversation_channel_type = (
        conversation_channel_type or request.context.verified_identity.channel_type
    )
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
            conversation_channel_type=conversation_channel_type,
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
            conversation_channel_type=conversation_channel_type,
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
