"""Owner-scoped memory lookup workflow shared by channel adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol, TypeVar
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from reminder_agent.application.conversations import (
    end_session,
    load_session,
    pending_memory_lookup_from_session,
)
from reminder_agent.features.memory.messages import lookup_memory_message
from reminder_agent.features.memory.retrieve.answering import (
    MemoryAnswerer,
    MemoryAnswerProposal,
    memory_answer_preserves_exact_values,
)
from reminder_agent.features.memory.retrieve.contracts import (
    MemoryLookupResult,
    MemoryRetrievalPolicy,
    is_memory_lookup_query_grounded,
)
from reminder_agent.features.memory.retrieve.persistence import (
    lookup_memory_fact,
    revalidate_owned_memory_candidates,
)
from reminder_agent.infrastructure.database.models import User
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.infrastructure.providers.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
    contains_sensitive_memory,
)
from reminder_agent.shared.memory_contracts import PendingMemoryLookup, RetrieveMemoryQuery

ResponseT_co = TypeVar("ResponseT_co", covariant=True)
logger = get_logger()
DEFAULT_EMBEDDING_PRIVACY_POLICY = EmbeddingPrivacyPolicy()
DEFAULT_MEMORY_RETRIEVAL_POLICY = MemoryRetrievalPolicy()


class MemoryLookupResponses(Protocol[ResponseT_co]):
    """Adapter-owned response and replay operations for a memory read."""

    def existing(self) -> ResponseT_co | None: ...

    def direct(self, text: str) -> ResponseT_co: ...


def run_memory_lookup_workflow[ResponseT_co](
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    channel_type: str = "TELEGRAM",
    pending: PendingMemoryLookup | None,
    query: str | None,
    source_text: str,
    from_model: bool = False,
    now: datetime,
    session_ttl_seconds: int,
    responses: MemoryLookupResponses[ResponseT_co],
    upsert_clarification_session: Callable[..., object],
    memory_answerer: MemoryAnswerer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    memory_retrieval_policy: MemoryRetrievalPolicy = DEFAULT_MEMORY_RETRIEVAL_POLICY,
    lookup_memory_operation: Callable[..., MemoryLookupResult] = lookup_memory_fact,
    contains_sensitive_memory_fn: Callable[[str], bool] = contains_sensitive_memory,
    answer_preserves_exact_values_fn: Callable[..., bool] = memory_answer_preserves_exact_values,
    allow_fact_disclosure_fn: Callable[[object], bool] | None = None,
    allow_sensitive_facts: bool = False,
    log_event_fn: Callable[..., object] = log_event,
    event_context: Mapping[str, object] | None = None,
) -> ResponseT_co:
    """Resolve one owned fact or persist a clarification in one transaction."""

    event_context = event_context or {}
    log_event_fn(
        logger,
        logging.INFO,
        "memory_lookup_request_received",
        **event_context,
        extraction_source="llm" if from_model else "deterministic_or_clarification",
        query_present=query is not None,
        query_length=len(query or ""),
        pending_candidate_count=len(pending.candidate_ids) if pending is not None else 0,
        embedding_provider_configured=embedding_provider is not None,
    )
    try:
        query = RetrieveMemoryQuery(query=query or "").query
    except (TypeError, ValueError):
        query = None
    if query is None or (from_model and not is_memory_lookup_query_grounded(source_text, query)):
        log_event_fn(
            logger,
            logging.INFO,
            "memory_lookup_query_rejected",
            **event_context,
            reason="missing_or_ungrounded_query",
            from_model=from_model,
        )
        return responses.direct(lookup_memory_message("missing_subject"))

    db.rollback()
    try:
        with db.begin():
            existing = responses.existing()
            if existing is not None:
                return existing
            locked_user = db.get(User, user_id)
            if (
                locked_user is None
                or locked_user.id != user_id
                or locked_user.onboarding_status != "ACTIVE"
            ):
                return responses.direct(lookup_memory_message("authorization_failed"))

            if pending is None:
                prior_session = load_session(
                    db,
                    user_id=locked_user.id,
                    external_session_id=external_session_id,
                    now=now,
                    ttl_seconds=session_ttl_seconds,
                    for_update=True,
                    channel_type=channel_type,
                )
                if prior_session is not None:
                    end_session(prior_session, now=now)

            result = lookup_memory_operation(
                db,
                user_id=locked_user.id,
                query=query,
                semantic_query=source_text,
                candidate_ids=pending.candidate_ids if pending is not None else None,
                embedding_provider=embedding_provider,
                privacy_policy=embedding_privacy_policy,
                retrieval_policy=memory_retrieval_policy,
                allow_sensitive_facts=allow_sensitive_facts,
            )
            log_event_fn(
                logger,
                logging.INFO,
                "memory_lookup_result_observed",
                **event_context,
                status=result.status,
                candidate_count=len(result.candidate_ids),
                from_model=from_model,
            )
            answer_text: str | None = None
            if result.status == "CANDIDATES":
                safe_candidates = tuple(
                    fact
                    for fact in result.candidates
                    if fact.user_id == locked_user.id
                    and (allow_sensitive_facts or not contains_sensitive_memory_fn(fact.content))
                )
                proposal = None
                if memory_answerer is not None and safe_candidates:
                    try:
                        proposal = memory_answerer.answer(
                            question=source_text or query,
                            memories=tuple(fact.content for fact in safe_candidates),
                        )
                    except Exception as exc:
                        log_event_fn(
                            logger,
                            logging.WARNING,
                            "memory_answer_generation_failed",
                            **event_context,
                            reason=type(exc).__name__,
                        )
                if not isinstance(
                    proposal, MemoryAnswerProposal
                ) or not proposal.validates_candidate_count(len(safe_candidates)):
                    log_event_fn(
                        logger,
                        logging.ERROR,
                        "memory_answer_validation_failed",
                        **event_context,
                        reason=(
                            "malformed_answer"
                            if not isinstance(proposal, MemoryAnswerProposal)
                            else "unsupported_candidate_index"
                        ),
                        candidate_count=len(safe_candidates),
                    )
                    result = MemoryLookupResult(
                        status=result.fallback_status,
                        candidate_ids=result.fallback_candidate_ids,
                    )
                elif proposal.decision == "ANSWER":
                    proposed_selected = tuple(
                        safe_candidates[index] for index in proposal.candidate_indices
                    )
                    selected = revalidate_owned_memory_candidates(
                        db,
                        user_id=locked_user.id,
                        candidates=proposed_selected,
                    )
                    rejection_reason = None
                    if len(selected) != len(proposed_selected):
                        rejection_reason = "candidate_revalidation_failed"
                    elif any(fact.user_id != locked_user.id for fact in selected):
                        rejection_reason = "candidate_ownership_failed"
                    elif proposal.answer is None:
                        rejection_reason = "missing_answer"
                    elif not answer_preserves_exact_values_fn(
                        proposal.answer,
                        tuple(fact.content for fact in selected),
                    ):
                        rejection_reason = "exact_value_preservation_failed"
                    if rejection_reason is not None:
                        log_event_fn(
                            logger,
                            logging.ERROR,
                            "memory_answer_validation_failed",
                            **event_context,
                            reason=rejection_reason,
                            selected_candidate_count=len(selected),
                        )
                        result = MemoryLookupResult(status="UNKNOWN")
                    else:
                        answer_text = proposal.answer
                        result = MemoryLookupResult(status="FOUND", fact=selected[0])
                elif proposal.decision == "AMBIGUOUS":
                    proposed_selected = tuple(
                        safe_candidates[index] for index in proposal.candidate_indices
                    )
                    selected = revalidate_owned_memory_candidates(
                        db,
                        user_id=locked_user.id,
                        candidates=proposed_selected,
                    )
                    if len(selected) != len(proposed_selected) or any(
                        fact.user_id != locked_user.id for fact in selected
                    ):
                        log_event_fn(
                            logger,
                            logging.ERROR,
                            "memory_answer_validation_failed",
                            **event_context,
                            reason=(
                                "candidate_revalidation_failed"
                                if len(selected) != len(proposed_selected)
                                else "candidate_ownership_failed"
                            ),
                            selected_candidate_count=len(selected),
                        )
                        result = MemoryLookupResult(status="UNKNOWN")
                    else:
                        result = MemoryLookupResult(
                            status="AMBIGUOUS",
                            candidate_ids=tuple(fact.id for fact in selected),
                        )
                        answer_text = proposal.clarification_question
                else:
                    result = MemoryLookupResult(status="UNKNOWN")
            if result.status == "FOUND" and result.fact is not None:
                if allow_fact_disclosure_fn is not None and not allow_fact_disclosure_fn(
                    result.fact
                ):
                    if pending is not None:
                        session = load_session(
                            db,
                            user_id=locked_user.id,
                            external_session_id=external_session_id,
                            now=now,
                            ttl_seconds=session_ttl_seconds,
                            for_update=True,
                            channel_type=channel_type,
                        )
                        if (
                            session is not None
                            and pending_memory_lookup_from_session(session) is not None
                        ):
                            end_session(session, now=now)
                    return responses.direct(
                        lookup_memory_message("sensitive_fact_protected")
                    )
                if pending is not None:
                    session = load_session(
                        db,
                        user_id=locked_user.id,
                        external_session_id=external_session_id,
                        now=now,
                        ttl_seconds=session_ttl_seconds,
                        for_update=True,
                        channel_type=channel_type,
                    )
                    if (
                        session is not None
                        and pending_memory_lookup_from_session(session) is not None
                    ):
                        end_session(session, now=now)
                reply = responses.direct(
                    answer_text or lookup_memory_message("fact_found", fact=result.fact.content)
                )
                log_event_fn(
                    logger,
                    logging.INFO,
                    "memory_lookup_response_created",
                    **event_context,
                    status="FOUND",
                    response_contains_fact=True,
                )
                return reply
            if result.status == "AMBIGUOUS":
                lookup_pending = PendingMemoryLookup(
                    query=query,
                    candidate_ids=list(result.candidate_ids),
                )
                upsert_clarification_session(
                    db,
                    user_id=locked_user.id,
                    external_session_id=external_session_id,
                    pending=lookup_pending,
                    now=now,
                    channel_type=channel_type,
                )
                reply = responses.direct(
                    answer_text or lookup_memory_message("ambiguous_matches")
                )
                log_event_fn(
                    logger,
                    logging.INFO,
                    "memory_lookup_response_created",
                    **event_context,
                    status="AMBIGUOUS",
                    response_contains_fact=False,
                )
                return reply
            if pending is not None:
                session = load_session(
                    db,
                    user_id=locked_user.id,
                    external_session_id=external_session_id,
                    now=now,
                    ttl_seconds=session_ttl_seconds,
                    for_update=True,
                    channel_type=channel_type,
                )
                if session is not None and pending_memory_lookup_from_session(session) is not None:
                    end_session(session, now=now)
            reply = responses.direct(lookup_memory_message("unknown_fact"))
            log_event_fn(
                logger,
                logging.INFO,
                "memory_lookup_response_created",
                **event_context,
                status="UNKNOWN",
                response_contains_fact=False,
            )
            return reply
    except SQLAlchemyError:
        db.rollback()
        log_event_fn(
            logger,
            logging.ERROR,
            "memory_lookup_failed",
            **event_context,
            reason="database_error",
        )
        return responses.direct(lookup_memory_message("retrieval_failed"))


__all__ = [
    "DEFAULT_EMBEDDING_PRIVACY_POLICY",
    "DEFAULT_MEMORY_RETRIEVAL_POLICY",
    "MemoryLookupResponses",
    "run_memory_lookup_workflow",
]
