"""Shared, owner-scoped memory-fact commands and persistence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import sqrt
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from reminder_agent.db.models import MEMORY_EMBEDDING_DIMENSIONS, MemoryFact
from reminder_agent.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
    EmbeddingProviderError,
    contains_sensitive_memory,
    embed_document,
    embed_query,
    validate_embedding,
)
from reminder_agent.intent_routing_rules import ExplicitCreateRoute, explicit_create_route

MEMORY_OPERATION = "SAVE_MEMORY"
MEMORY_LOOKUP_OPERATION = "RETRIEVE_MEMORY"
_MEMORY_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:remember|save|(?:make|take)\s+(?:a\s+)?note)\b",
    re.IGNORECASE,
)
_LEADING_FILLER = re.compile(r"^(?:that|this|of|to|for\s+me)\s+", re.IGNORECASE)
_LOOKUP_PATTERNS = (
    re.compile(
        r"^\s*(?:what(?:['’]s|\s+is)|do\s+you\s+(?:remember|know)|tell\s+me)\s+"
        r"(?:my|the)\s+(?P<query>.+?)\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*where\s+(?:is\s+my|do\s+i)\s+(?P<query>.+?)\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*which\s+(?P<query>.+?)\s+do\s+i\s+(?:like|prefer)\s+most\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
)
_LOOKUP_IDENTIFIER = re.compile(
    r"\b(?:fact|memory|user)\s*(?:id|uuid)\b|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


class MemoryIntentEvidence(BaseModel):
    """Semantic evidence returned by the classifier, never authority to write."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "EXPLICIT_SAVE_REQUEST",
        "ORDINARY_STATEMENT",
        "QUOTED_INSTRUCTION",
        "NEGATED_SAVE_REQUEST",
        "REMINDER_REQUEST",
        "UNCERTAIN",
    ]
    confidence: Literal["CLEAR", "UNCERTAIN"]
    evidence_text: str | None = Field(default=None, max_length=500)

    @field_validator("evidence_text")
    @classmethod
    def evidence_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split()).strip()
        return value or None


class SaveMemoryFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["SAVE_MEMORY"] = MEMORY_OPERATION
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value or value.casefold().strip(" .!?,") in {
            "this",
            "that",
            "something",
            "information",
            "remember",
            "save",
            "note",
        }:
            raise ValueError("memory content must not be blank")
        return value


class PendingMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["SAVE_MEMORY"] = MEMORY_OPERATION
    content: None = None


class PendingMemoryLookup(BaseModel):
    """Typed context for narrowing an ambiguous owner-scoped lookup."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["RETRIEVE_MEMORY"] = MEMORY_LOOKUP_OPERATION
    query: str = Field(min_length=1, max_length=300)
    candidate_ids: list[UUID] = Field(min_length=1, max_length=50)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split()).strip()
        if not value:
            raise ValueError("memory lookup query must not be blank")
        return value


class RetrieveMemoryQuery(BaseModel):
    """A subject-only lookup proposal accepted from the existing extractor."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["RETRIEVE_MEMORY"] = MEMORY_LOOKUP_OPERATION
    query: str = Field(min_length=1, max_length=300)

    @field_validator("query")
    @classmethod
    def query_must_be_a_subject_not_an_identifier(cls, value: str) -> str:
        value = " ".join(value.split()).strip(" .!?")
        if not value or not _lookup_tokens(value):
            raise ValueError("memory lookup query must contain a subject")
        if _LOOKUP_IDENTIFIER.search(value):
            raise ValueError("memory lookup query must not contain an identifier")
        return value


@dataclass(frozen=True)
class MemoryLookupResult:
    status: Literal["FOUND", "UNKNOWN", "AMBIGUOUS"]
    fact: MemoryFact | None = None
    candidate_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True)
class SavedMemory:
    fact_id: UUID
    saved_at: datetime
    embedding_status: str


@dataclass(frozen=True)
class MemoryRetrievalPolicy:
    lexical_candidate_limit: int = 20
    semantic_candidate_limit: int = 5
    # Gemini's normalized 1536-dimensional retrieval vectors score common
    # paraphrases around 0.70; keep unrelated subjects below this floor.
    semantic_relevance_threshold: float = 0.68
    semantic_ambiguity_margin: float = 0.04

    def __post_init__(self) -> None:
        if not 2 <= self.lexical_candidate_limit <= 50:
            raise ValueError("lexical candidate limit must be between 2 and 50")
        if not 2 <= self.semantic_candidate_limit <= 20:
            raise ValueError("semantic candidate limit must be between 2 and 20")
        if not -1 <= self.semantic_relevance_threshold <= 1:
            raise ValueError("semantic relevance threshold must be between -1 and 1")
        if not 0 <= self.semantic_ambiguity_margin <= 1:
            raise ValueError("semantic ambiguity margin must be between 0 and 1")


def is_memory_save_command(text: str) -> bool:
    return explicit_create_route(text) is ExplicitCreateRoute.SAVE_MEMORY


def is_memory_intent_evidence_grounded(
    source: str, evidence: MemoryIntentEvidence | None
) -> bool:
    """Validate classifier evidence without reimplementing semantic intent parsing."""

    if evidence is None or evidence.kind != "EXPLICIT_SAVE_REQUEST":
        return False
    if evidence.confidence != "CLEAR" or not evidence.evidence_text:
        return False

    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    return normalize(evidence.evidence_text) in normalize(source)


def is_memory_content_grounded(source: str, content: str) -> bool:
    """Require a model-extracted fact to share meaningful words with the input."""

    source_tokens = {
        token.casefold()
        for token in re.findall(r"[\w@#%+.-]+", source, flags=re.UNICODE)
        if len(token) > 2
    }
    content_tokens = {
        token.casefold()
        for token in re.findall(r"[\w@#%+.-]+", content, flags=re.UNICODE)
        if len(token) > 2
    }
    return bool(source_tokens & content_tokens)


def extract_memory_content(text: str) -> str | None:
    """Extract content only after the shared routing rule selected memory."""

    if not is_memory_save_command(text):
        return None
    match = _MEMORY_PREFIX.match(text)
    if match is None:
        return None
    remainder = _LEADING_FILLER.sub("", text[match.end() :].strip())
    if not remainder or remainder.casefold().strip(" .!?,") in {
        "this",
        "that",
        "something",
        "information",
    }:
        return None
    try:
        return SaveMemoryFact(content=remainder).content
    except ValueError:
        return None


def extract_memory_lookup_query(text: str) -> str | None:
    """Extract only a specific, deterministic memory lookup subject."""

    for pattern in _LOOKUP_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        query = " ".join(match.group("query").split()).strip(" .!?\t")
        if query and _lookup_tokens(query):
            return query
    return None


def is_memory_lookup_query_grounded(source: str, query: str) -> bool:
    """Require a model lookup subject to be expressed by the user's request.

    The model may name only the requested subject.  It cannot select a fact,
    user, or value: the application searches the current owner's facts itself.
    """

    try:
        query = RetrieveMemoryQuery(query=query).query
    except (TypeError, ValueError):
        return False
    return set(_lookup_tokens(query)).issubset(set(_lookup_tokens(source)))


def save_memory_fact(
    db: Session,
    *,
    user_id: UUID,
    content: str,
    now: datetime | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    privacy_policy: EmbeddingPrivacyPolicy = EmbeddingPrivacyPolicy(),
) -> SavedMemory:
    command = SaveMemoryFact(content=content)
    saved_at = now or datetime.now(timezone.utc)
    if saved_at.tzinfo is None or saved_at.utcoffset() is None:
        saved_at = saved_at.replace(tzinfo=timezone.utc)
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
    return SavedMemory(
        fact_id=fact.id,
        saved_at=saved_at,
        embedding_status=fact.embedding_status,
    )


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
            embed_document(provider, content),
            dimensions=MEMORY_EMBEDDING_DIMENSIONS,
        )
    except Exception as exc:
        error_code = (
            exc.code if isinstance(exc, EmbeddingProviderError) else "PROVIDER_UNAVAILABLE"
        )
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
    privacy_policy: EmbeddingPrivacyPolicy = EmbeddingPrivacyPolicy(),
    now: datetime | None = None,
) -> bool:
    """Idempotently index the fact's current authoritative content."""

    indexed_at = now or datetime.now(timezone.utc)
    content_hash = _content_hash(fact.content)
    if (
        fact.embedding_status == "SKIPPED_SENSITIVE"
        and embedding_provider.is_external
        and not privacy_policy.allow_sensitive_external
        and fact.embedding_model == embedding_provider.model
        and fact.embedding_content_hash == content_hash
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


def update_memory_fact_content(
    db: Session,
    *,
    user_id: UUID,
    fact_id: UUID,
    content: str,
    embedding_provider: EmbeddingProvider | None = None,
    privacy_policy: EmbeddingPrivacyPolicy = EmbeddingPrivacyPolicy(),
    now: datetime | None = None,
) -> MemoryFact | None:
    """Owner-scoped content update that invalidates or regenerates its vector."""

    command = SaveMemoryFact(content=content)
    changed_at = now or datetime.now(timezone.utc)
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
    privacy_policy: EmbeddingPrivacyPolicy = EmbeddingPrivacyPolicy(),
    batch_size: int = 100,
    now: datetime | None = None,
) -> int:
    """Index one locked, resumable batch without exposing memory text in logs."""

    if not 1 <= batch_size <= 1000:
        raise ValueError("batch size must be between 1 and 1000")
    retry_statuses = ["PENDING", "FAILED"]
    if not embedding_provider.is_external or privacy_policy.allow_sensitive_external:
        retry_statuses.append("SKIPPED_SENSITIVE")
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
    indexed_at = now or datetime.now(timezone.utc)
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


_LOOKUP_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "my",
    "of",
    "the",
    "to",
}


def _normalize_lookup_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("-", "").replace("_", "")
    return " ".join(value.split())


def _lookup_tokens(value: str) -> tuple[str, ...]:
    normalized = _normalize_lookup_text(value)
    return tuple(
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if token not in _LOOKUP_STOP_WORDS and len(token) > 1
    )


def lookup_memory_fact(
    db: Session,
    *,
    user_id: UUID,
    query: str,
    candidate_ids: list[UUID] | tuple[UUID, ...] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    privacy_policy: EmbeddingPrivacyPolicy = EmbeddingPrivacyPolicy(),
    retrieval_policy: MemoryRetrievalPolicy = MemoryRetrievalPolicy(),
) -> MemoryLookupResult:
    """Find one owner-scoped fact, using lexical evidence before semantics."""

    query = " ".join(query.split()).strip()
    query_tokens = _lookup_tokens(query)
    if not query_tokens:
        return MemoryLookupResult(status="UNKNOWN")

    filters = [MemoryFact.user_id == user_id]
    if candidate_ids is not None:
        if not candidate_ids:
            return MemoryLookupResult(status="UNKNOWN")
        filters.append(MemoryFact.id.in_(candidate_ids))
    normalized_content = func.replace(
        func.replace(func.lower(MemoryFact.content), "-", ""), "_", ""
    )
    lexical_filters = [normalized_content.contains(token) for token in query_tokens]
    facts = list(
        db.scalars(
            select(MemoryFact)
            .where(*filters, *lexical_filters)
            .order_by(MemoryFact.created_at.desc(), MemoryFact.id.asc())
            .limit(retrieval_policy.lexical_candidate_limit + 1)
        ).all()
    )

    matches: list[MemoryFact] = []
    for fact in facts:
        content_tokens = set(_lookup_tokens(fact.content))
        if not set(query_tokens).issubset(content_tokens):
            continue
        matches.append(fact)

    if not matches:
        return _semantic_lookup(
            db,
            user_id=user_id,
            query=query,
            candidate_ids=candidate_ids,
            embedding_provider=embedding_provider,
            privacy_policy=privacy_policy,
            retrieval_policy=retrieval_policy,
        )
    # Do not let an exact phrase score conceal a different fact that has the
    # same requested subject. A clarification is safer than guessing whether
    # an older or more specific-looking value is still authoritative.
    if len(matches) != 1:
        return MemoryLookupResult(
            status="AMBIGUOUS",
            candidate_ids=tuple(fact.id for fact in matches),
        )
    return MemoryLookupResult(status="FOUND", fact=matches[0])


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _semantic_lookup(
    db: Session,
    *,
    user_id: UUID,
    query: str,
    candidate_ids: list[UUID] | tuple[UUID, ...] | None,
    embedding_provider: EmbeddingProvider | None,
    privacy_policy: EmbeddingPrivacyPolicy,
    retrieval_policy: MemoryRetrievalPolicy,
) -> MemoryLookupResult:
    if embedding_provider is None:
        return MemoryLookupResult(status="UNKNOWN")
    if (
        embedding_provider.is_external
        and contains_sensitive_memory(query)
        and not privacy_policy.allow_sensitive_external
    ):
        return MemoryLookupResult(status="UNKNOWN")
    try:
        query_vector = validate_embedding(
            embed_query(embedding_provider, query),
            dimensions=MEMORY_EMBEDDING_DIMENSIONS,
        )
    except Exception:
        return MemoryLookupResult(status="UNKNOWN")

    filters = [
        MemoryFact.user_id == user_id,
        MemoryFact.embedding_status == "INDEXED",
        MemoryFact.embedding_model == embedding_provider.model,
        MemoryFact.embedding.is_not(None),
    ]
    if candidate_ids is not None:
        if not candidate_ids:
            return MemoryLookupResult(status="UNKNOWN")
        filters.append(MemoryFact.id.in_(candidate_ids))

    dialect = db.get_bind().dialect.name
    scored: list[tuple[MemoryFact, float]] = []
    if dialect == "postgresql":
        distance = MemoryFact.embedding.cosine_distance(query_vector)
        rows = db.execute(
            select(MemoryFact, distance.label("distance"))
            .where(*filters)
            .order_by(distance.asc(), MemoryFact.id.asc())
            .limit(retrieval_policy.semantic_candidate_limit)
        ).all()
        scored = [(row[0], 1.0 - float(row[1])) for row in rows]
    else:
        # Unit-test fallback only. Production PostgreSQL computes distance and
        # returns a bounded owner-filtered result directly in SQL.
        facts = list(
            db.scalars(
                select(MemoryFact)
                .where(*filters)
                .order_by(MemoryFact.id.asc())
                .limit(retrieval_policy.semantic_candidate_limit)
            ).all()
        )
        scored = [
            (fact, _cosine_similarity(query_vector, list(fact.embedding or [])))
            for fact in facts
        ]
        scored.sort(key=lambda item: (-item[1], str(item[0].id)))

    relevant = [
        item
        for item in scored
        if item[1] >= retrieval_policy.semantic_relevance_threshold
    ]
    if not relevant:
        return MemoryLookupResult(status="UNKNOWN")
    best_score = relevant[0][1]
    plausible = [
        fact
        for fact, score in relevant
        if best_score - score <= retrieval_policy.semantic_ambiguity_margin
    ]
    if len(plausible) != 1:
        return MemoryLookupResult(
            status="AMBIGUOUS",
            candidate_ids=tuple(fact.id for fact in plausible),
        )
    return MemoryLookupResult(status="FOUND", fact=plausible[0])
