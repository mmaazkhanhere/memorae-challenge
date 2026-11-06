"""Shared, owner-scoped memory-fact commands and persistence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.db.models import MemoryFact
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
) -> SavedMemory:
    command = SaveMemoryFact(content=content)
    saved_at = now or datetime.now(timezone.utc)
    if saved_at.tzinfo is None or saved_at.utcoffset() is None:
        saved_at = saved_at.replace(tzinfo=timezone.utc)
    fact = MemoryFact(
        user_id=user_id,
        content=command.content,
        created_at=saved_at,
        updated_at=saved_at,
    )
    db.add(fact)
    db.flush()
    return SavedMemory(fact_id=fact.id, saved_at=saved_at)


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
) -> MemoryLookupResult:
    """Find one exact owner-scoped fact without modifying memory state."""

    query = " ".join(query.split()).strip()
    query_tokens = _lookup_tokens(query)
    if not query_tokens:
        return MemoryLookupResult(status="UNKNOWN")

    filters = [MemoryFact.user_id == user_id]
    if candidate_ids is not None:
        if not candidate_ids:
            return MemoryLookupResult(status="UNKNOWN")
        filters.append(MemoryFact.id.in_(candidate_ids))
    facts = list(
        db.scalars(
            select(MemoryFact)
            .where(*filters)
            .order_by(MemoryFact.created_at.desc(), MemoryFact.id.asc())
        ).all()
    )

    matches: list[MemoryFact] = []
    for fact in facts:
        content_tokens = set(_lookup_tokens(fact.content))
        if not set(query_tokens).issubset(content_tokens):
            continue
        matches.append(fact)

    if not matches:
        return MemoryLookupResult(status="UNKNOWN")
    # Do not let an exact phrase score conceal a different fact that has the
    # same requested subject. A clarification is safer than guessing whether
    # an older or more specific-looking value is still authoritative.
    if len(matches) != 1:
        return MemoryLookupResult(
            status="AMBIGUOUS",
            candidate_ids=tuple(fact.id for fact in matches),
        )
    return MemoryLookupResult(status="FOUND", fact=matches[0])
