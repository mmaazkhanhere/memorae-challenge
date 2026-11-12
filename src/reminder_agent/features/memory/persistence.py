"""Owner-scoped memory fact persistence and embedding lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from reminder_agent.infrastructure.database.models import MEMORY_EMBEDDING_DIMENSIONS, MemoryFact
from reminder_agent.infrastructure.providers.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
    EmbeddingProviderError,
    contains_sensitive_memory,
    embed_document,
    validate_embedding,
)
from reminder_agent.shared.memory_contracts import SaveMemoryFact

DEFAULT_EMBEDDING_PRIVACY_POLICY = EmbeddingPrivacyPolicy()


@dataclass(frozen=True)
class SavedMemory:
    """Identity and indexing state of a newly persisted memory fact."""

    fact_id: UUID
    saved_at: datetime
    embedding_status: str


def save_memory_fact(
    db: Session,
    *,
    user_id: UUID,
    content: str,
    now: datetime | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
) -> SavedMemory:
    """Stage an owner-scoped memory fact; the caller owns commit/rollback."""

    command = SaveMemoryFact(content=content)
    saved_at = now or datetime.now(UTC)
    if saved_at.tzinfo is None or saved_at.utcoffset() is None:
        saved_at = saved_at.replace(tzinfo=UTC)
    embedding_fields = _prepare_embedding(
        command.content,
        provider=embedding_provider,
        privacy_policy=privacy_policy,
        now=saved_at,
    )
    fact = MemoryFact(
        user_id=user_id,
        content=command.content,
        created_at=saved_at,
        updated_at=saved_at,
        **embedding_fields,
    )
    db.add(fact)
    db.flush()
    return SavedMemory(fact_id=fact.id, saved_at=saved_at, embedding_status=fact.embedding_status)


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _prepare_embedding(
    content: str,
    *,
    provider: EmbeddingProvider | None,
    privacy_policy: EmbeddingPrivacyPolicy,
    now: datetime,
    prior_attempts: int = 0,
) -> dict:
    if provider is None:
        return {
            "embedding": None,
            "embedding_status": "PENDING",
            "embedding_model": None,
            "embedding_content_hash": None,
            "embedding_attempts": prior_attempts,
            "embedding_last_error": None,
            "embedding_updated_at": None,
        }
    if (
        provider.is_external
        and contains_sensitive_memory(content)
        and not privacy_policy.allow_sensitive_external
    ):
        return {
            "embedding": None,
            "embedding_status": "SKIPPED_SENSITIVE",
            "embedding_model": provider.model,
            "embedding_content_hash": _content_hash(content),
            "embedding_attempts": prior_attempts,
            "embedding_last_error": None,
            "embedding_updated_at": now,
        }
    try:
        vector = validate_embedding(
            embed_document(provider, content), dimensions=MEMORY_EMBEDDING_DIMENSIONS
        )
    except Exception as exc:
        error_code = exc.code if isinstance(exc, EmbeddingProviderError) else "PROVIDER_UNAVAILABLE"
        return {
            "embedding": None,
            "embedding_status": "FAILED",
            "embedding_model": provider.model,
            "embedding_content_hash": _content_hash(content),
            "embedding_attempts": prior_attempts + 1,
            "embedding_last_error": error_code,
            "embedding_updated_at": now,
        }
    return {
        "embedding": vector,
        "embedding_status": "INDEXED",
        "embedding_model": provider.model,
        "embedding_content_hash": _content_hash(content),
        "embedding_attempts": prior_attempts + 1,
        "embedding_last_error": None,
        "embedding_updated_at": now,
    }


def index_memory_fact(
    fact: MemoryFact,
    *,
    embedding_provider: EmbeddingProvider,
    privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    now: datetime | None = None,
) -> bool:
    """Idempotently index the fact's current authoritative content."""

    indexed_at = now or datetime.now(UTC)
    content_hash = _content_hash(fact.content)
    if (
        fact.embedding_status == "SKIPPED_SENSITIVE"
        and embedding_provider.is_external
        and not privacy_policy.allow_sensitive_external
        and fact.embedding_model == embedding_provider.model
        and fact.embedding_content_hash == content_hash
        and contains_sensitive_memory(fact.content)
    ):
        return False
    if (
        fact.embedding_status == "INDEXED"
        and fact.embedding is not None
        and fact.embedding_model == embedding_provider.model
        and fact.embedding_content_hash == content_hash
    ):
        return False
    fields = _prepare_embedding(
        fact.content,
        provider=embedding_provider,
        privacy_policy=privacy_policy,
        now=indexed_at,
        prior_attempts=fact.embedding_attempts,
    )
    for name, value in fields.items():
        setattr(fact, name, value)
    return True


def index_memory_fact_for_owner(
    db: Session,
    *,
    fact_id: UUID,
    user_id: UUID,
    embedding_provider: EmbeddingProvider,
    privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    now: datetime | None = None,
) -> str:
    """Lock and index one owned fact, returning its resulting index status."""

    fact = db.scalar(
        select(MemoryFact)
        .where(MemoryFact.id == fact_id, MemoryFact.user_id == user_id)
        .with_for_update()
    )
    if fact is None:
        return "PENDING"
    index_memory_fact(
        fact,
        embedding_provider=embedding_provider,
        privacy_policy=privacy_policy,
        now=now,
    )
    return fact.embedding_status


def update_memory_fact_content(
    db: Session,
    *,
    user_id: UUID,
    fact_id: UUID,
    content: str,
    embedding_provider: EmbeddingProvider | None = None,
    privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    now: datetime | None = None,
) -> MemoryFact | None:
    """Stage an owner-scoped content update; the caller owns commit/rollback."""

    command = SaveMemoryFact(content=content)
    changed_at = now or datetime.now(UTC)
    fact = db.scalar(
        select(MemoryFact)
        .where(MemoryFact.id == fact_id, MemoryFact.user_id == user_id)
        .with_for_update()
    )
    if fact is None:
        return None
    if fact.content == command.content:
        if embedding_provider is not None:
            index_memory_fact(
                fact,
                embedding_provider=embedding_provider,
                privacy_policy=privacy_policy,
                now=changed_at,
            )
        return fact
    fact.content = command.content
    fact.updated_at = changed_at
    fields = _prepare_embedding(
        fact.content,
        provider=embedding_provider,
        privacy_policy=privacy_policy,
        now=changed_at,
    )
    for name, value in fields.items():
        setattr(fact, name, value)
    db.flush()
    return fact


def backfill_memory_embeddings(
    db: Session,
    *,
    embedding_provider: EmbeddingProvider,
    privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    batch_size: int = 100,
    now: datetime | None = None,
) -> int:
    """Stage one locked, resumable batch; the caller owns commit/rollback."""

    if not 1 <= batch_size <= 1000:
        raise ValueError("batch size must be between 1 and 1000")
    # Re-evaluate skipped rows because the privacy classifier may have marked
    # an older fact based on a false positive, such as "email" as a channel.
    # Truly sensitive rows remain SKIPPED_SENSITIVE in _prepare_embedding when
    # external indexing is not allowed.
    retry_statuses = ["PENDING", "FAILED", "SKIPPED_SENSITIVE"]
    facts = list(
        db.scalars(
            select(MemoryFact)
            .where(
                or_(
                    MemoryFact.embedding_status.in_(retry_statuses),
                    MemoryFact.embedding_model != embedding_provider.model,
                    MemoryFact.embedding_model.is_(None),
                )
            )
            .order_by(MemoryFact.updated_at.asc(), MemoryFact.id.asc())
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        ).all()
    )
    indexed_at = now or datetime.now(UTC)
    changed = 0
    for fact in facts:
        if index_memory_fact(
            fact,
            embedding_provider=embedding_provider,
            privacy_policy=privacy_policy,
            now=indexed_at,
        ):
            changed += 1
    db.flush()
    return changed

