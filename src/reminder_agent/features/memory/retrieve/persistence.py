"""Owner-scoped lexical and semantic memory retrieval."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.features.memory.retrieve.contracts import (
    DEFAULT_MEMORY_RETRIEVAL_POLICY,
    MemoryLookupResult,
    MemoryRetrievalPolicy,
    cosine_similarity,
)
from reminder_agent.infrastructure.database.models import MEMORY_EMBEDDING_DIMENSIONS, MemoryFact
from reminder_agent.infrastructure.providers.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
    EmbeddingProviderError,
    contains_sensitive_memory,
    embed_query,
    validate_embedding,
)
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.memory_contracts import _lookup_tokens

logger = get_logger()
DEFAULT_EMBEDDING_PRIVACY_POLICY = EmbeddingPrivacyPolicy()


def revalidate_owned_memory_candidates(
    db: Session,
    *,
    user_id: UUID,
    candidates: tuple[MemoryFact, ...],
) -> tuple[MemoryFact, ...]:
    """Reload proposed facts and retain only unchanged facts owned by ``user_id``."""

    if not candidates:
        return ()
    refreshed = {
        fact.id: fact
        for fact in db.scalars(
            select(MemoryFact)
            .where(
                MemoryFact.user_id == user_id,
                MemoryFact.id.in_(fact.id for fact in candidates),
            )
            .execution_options(populate_existing=True)
        ).all()
    }
    return tuple(
        refreshed[fact.id]
        for fact in candidates
        if fact.id in refreshed and refreshed[fact.id].content == fact.content
    )


def lookup_memory_fact(
    db: Session,
    *,
    user_id: UUID,
    query: str,
    semantic_query: str | None = None,
    candidate_ids: list[UUID] | tuple[UUID, ...] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    retrieval_policy: MemoryRetrievalPolicy = DEFAULT_MEMORY_RETRIEVAL_POLICY,
) -> MemoryLookupResult:
    """Find one owner-scoped fact, using lexical evidence before semantics."""

    query = " ".join(query.split()).strip()
    query_tokens = _lookup_tokens(query)
    log_event(
        logger,
        logging.INFO,
        "memory_lookup_started",
        owner_scoped=True,
        query_length=len(query),
        query_token_count=len(query_tokens),
        candidate_filter_count=len(candidate_ids or ()),
        embedding_provider_configured=embedding_provider is not None,
        embedding_model=embedding_provider.model if embedding_provider is not None else None,
    )
    if not query_tokens:
        log_event(logger, logging.INFO, "memory_lookup_result", status="UNKNOWN", reason="empty_query")
        return MemoryLookupResult(status="UNKNOWN")

    filters = [MemoryFact.user_id == user_id]
    if candidate_ids is not None:
        if not candidate_ids:
            return MemoryLookupResult(status="UNKNOWN")
        filters.append(MemoryFact.id.in_(candidate_ids))
    normalized_content = func.replace(func.replace(func.lower(MemoryFact.content), "-", ""), "_", "")
    lexical_filters = [normalized_content.contains(token) for token in query_tokens]
    facts = list(
        db.scalars(
            select(MemoryFact)
            .where(*filters, *lexical_filters)
            .order_by(MemoryFact.created_at.desc(), MemoryFact.id.asc())
            .limit(retrieval_policy.lexical_candidate_limit + 1)
        ).all()
    )

    matches = [
        fact
        for fact in facts
        if set(query_tokens).issubset(set(_lookup_tokens(fact.content)))
    ]
    log_event(
        logger,
        logging.INFO,
        "memory_lexical_lookup_completed",
        owner_scoped=True,
        candidate_count=len(facts),
        match_count=len(matches),
    )
    if len(matches) == 1:
        return MemoryLookupResult(status="FOUND", fact=matches[0])

    fallback = (
        MemoryLookupResult(status="AMBIGUOUS", candidate_ids=tuple(fact.id for fact in matches))
        if matches
        else MemoryLookupResult(status="UNKNOWN")
    )
    return _semantic_lookup(
        db,
        user_id=user_id,
        query=query,
        semantic_query=semantic_query,
        candidate_ids=candidate_ids,
        embedding_provider=embedding_provider,
        privacy_policy=privacy_policy,
        retrieval_policy=retrieval_policy,
        fallback=fallback,
    )


def _semantic_lookup(
    db: Session,
    *,
    user_id: UUID,
    query: str,
    semantic_query: str | None,
    candidate_ids: list[UUID] | tuple[UUID, ...] | None,
    embedding_provider: EmbeddingProvider | None,
    privacy_policy: EmbeddingPrivacyPolicy,
    retrieval_policy: MemoryRetrievalPolicy,
    fallback: MemoryLookupResult,
) -> MemoryLookupResult:
    query_text = " ".join((semantic_query or query).split()).strip()
    log_event(
        logger,
        logging.INFO,
        "memory_semantic_lookup_started",
        owner_scoped=True,
        embedding_provider_configured=embedding_provider is not None,
        embedding_model=embedding_provider.model if embedding_provider is not None else None,
        candidate_filter_count=len(candidate_ids or ()),
        semantic_candidate_limit=retrieval_policy.semantic_candidate_limit,
        relevance_threshold=retrieval_policy.semantic_relevance_threshold,
        ambiguity_margin=retrieval_policy.semantic_ambiguity_margin,
        query_representation="full_request" if semantic_query else "extracted_subject",
        semantic_query_length=len(query_text),
    )
    if embedding_provider is None:
        log_event(logger, logging.INFO, "memory_semantic_lookup_skipped", reason="provider_unavailable")
        return fallback
    if (
        embedding_provider.is_external
        and contains_sensitive_memory(query_text)
        and not privacy_policy.allow_sensitive_external
    ):
        log_event(logger, logging.INFO, "memory_semantic_lookup_skipped", reason="sensitive_query")
        return fallback
    try:
        query_vector = validate_embedding(
            embed_query(embedding_provider, query_text),
            dimensions=MEMORY_EMBEDDING_DIMENSIONS,
        )
        log_event(
            logger,
            logging.INFO,
            "memory_semantic_query_embedding_created",
            embedding_model=embedding_provider.model,
            dimensions=len(query_vector),
        )
    except Exception as exc:
        log_event(
            logger,
            logging.WARNING,
            "memory_semantic_lookup_failed",
            reason="query_embedding_failed",
            error_code=exc.code if isinstance(exc, EmbeddingProviderError) else "PROVIDER_UNAVAILABLE",
        )
        return fallback

    filters = [
        MemoryFact.user_id == user_id,
        MemoryFact.embedding_status == "INDEXED",
        MemoryFact.embedding_model == embedding_provider.model,
        MemoryFact.embedding.is_not(None),
    ]
    if candidate_ids is not None:
        if not candidate_ids:
            return fallback
        filters.append(MemoryFact.id.in_(candidate_ids))

    dialect = db.get_bind().dialect.name
    scored: list[tuple[MemoryFact, float]] = []
    answer_candidate_limit = min(3, retrieval_policy.semantic_candidate_limit)
    scan_limit = max(12, retrieval_policy.semantic_candidate_limit)
    if dialect == "postgresql":
        distance = MemoryFact.embedding.cosine_distance(query_vector)
        rows = db.execute(
            select(MemoryFact, distance.label("distance"))
            .where(*filters)
            .order_by(distance.asc(), MemoryFact.id.asc())
            .limit(scan_limit)
        ).all()
        scored = [(row[0], 1.0 - float(row[1])) for row in rows]
    else:
        facts = list(
            db.scalars(
                select(MemoryFact).where(*filters).order_by(MemoryFact.id.asc()).limit(scan_limit)
            ).all()
        )
        scored = [
            (fact, cosine_similarity(query_vector, list(fact.embedding or []))) for fact in facts
        ]
        scored.sort(key=lambda item: (-item[1], str(item[0].id)))

    log_event(
        logger,
        logging.INFO,
        "memory_semantic_similarity_query_completed",
        dialect=dialect,
        owner_scoped=True,
        active_embedding_model=embedding_provider.model,
        candidate_count=len(scored),
        candidate_ids=[str(fact.id) for fact, _ in scored],
        similarity_scores=[round(score, 6) for _, score in scored],
    )
    safe_candidates = tuple(
        fact for fact, _score in scored if not contains_sensitive_memory(fact.content)
    )[:answer_candidate_limit]
    if not safe_candidates:
        log_event(
            logger,
            logging.INFO,
            "memory_semantic_lookup_result",
            status=fallback.status,
            safe_candidate_count=0,
            top_candidate_count=len(scored),
            reason="no_safe_candidates",
        )
        return fallback
    log_event(
        logger,
        logging.INFO,
        "memory_semantic_lookup_result",
        status="CANDIDATES",
        safe_candidate_count=len(safe_candidates),
        candidate_ids=[str(fact.id) for fact in safe_candidates],
        final_decision_deferred=True,
    )
    return MemoryLookupResult(
        status="CANDIDATES",
        candidates=safe_candidates,
        candidate_ids=tuple(fact.id for fact in safe_candidates),
        fallback_status="AMBIGUOUS" if fallback.status == "AMBIGUOUS" else "UNKNOWN",
        fallback_candidate_ids=fallback.candidate_ids,
    )


__all__ = ["lookup_memory_fact", "revalidate_owned_memory_candidates"]
