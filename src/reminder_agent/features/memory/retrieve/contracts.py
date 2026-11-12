"""Pure contracts and policies for memory retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from reminder_agent.shared.memory_contracts import (
    RetrieveMemoryQuery,
    _lookup_token_forms,
)


@dataclass(frozen=True)
class MemoryLookupResult:
    status: Literal["FOUND", "CANDIDATES", "UNKNOWN", "AMBIGUOUS"]
    fact: Any | None = None
    candidates: tuple[Any, ...] = ()
    candidate_ids: tuple[UUID, ...] = ()
    fallback_status: Literal["UNKNOWN", "AMBIGUOUS"] = "UNKNOWN"
    fallback_candidate_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True)
class MemoryRetrievalPolicy:
    lexical_candidate_limit: int = 20
    semantic_candidate_limit: int = 3
    # Retained for configuration compatibility and offline retrieval evaluation.
    # Similarity ranks candidates; the final answer remains grounded separately.
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


DEFAULT_MEMORY_RETRIEVAL_POLICY = MemoryRetrievalPolicy()


def is_memory_lookup_query_grounded(source: str, query: str) -> bool:
    """Require an extracted lookup subject to be expressed by the request."""

    try:
        query = RetrieveMemoryQuery(query=query).query
    except (TypeError, ValueError):
        return False
    return _lookup_token_forms(query).issubset(_lookup_token_forms(source))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


class MemoryAnswerProposal(BaseModel):
    """Strict, non-authoritative candidate selection from supplied memories."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["ANSWER", "AMBIGUOUS", "UNKNOWN"]
    candidate_indices: list[StrictInt] = Field(default_factory=list, max_length=3)

    @field_validator("candidate_indices")
    @classmethod
    def indices_must_be_unique_and_non_negative(cls, value: list[int]) -> list[int]:
        if any(isinstance(index, bool) or index < 0 for index in value):
            raise ValueError("candidate indices must be non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("candidate indices must be unique")
        return value

    @model_validator(mode="after")
    def decision_fields_must_be_consistent(self) -> MemoryAnswerProposal:
        if self.decision == "ANSWER":
            if not self.candidate_indices:
                raise ValueError("ANSWER requires supporting candidates")
        elif self.decision == "AMBIGUOUS":
            if len(self.candidate_indices) < 2:
                raise ValueError("AMBIGUOUS requires conflicting candidates")
        elif self.candidate_indices:
            raise ValueError("UNKNOWN must not claim candidates")
        return self

    def validates_candidate_count(self, candidate_count: int) -> bool:
        return all(index < candidate_count for index in self.candidate_indices)


_MEMORY_CLAUSE_SEPARATOR = re.compile(
    r"\s*(?:[.!?;]+(?=\s|$)|,\s*(?=(?:my|the|a|an)\b)|"
    r"\s+(?:and|or)\s+(?=(?:my|the|a|an)\b))\s*",
    re.IGNORECASE,
)


def select_memory_fact_clause(content: str, query: str) -> str | None:
    """Return one safely relevant original clause, or ``None`` if ambiguous.

    A single-clause memory is already an atomic stored fact and is returned
    unchanged. Multi-clause memories are narrowed only when the normalized
    lookup subject matches exactly one clause. This intentionally declines to
    guess when a semantic query has no deterministic clause evidence.
    """

    normalized_content = content.strip(" .!?;")
    if not normalized_content:
        return None
    clauses = tuple(
        clause.strip(" .!?;,")
        for clause in _MEMORY_CLAUSE_SEPARATOR.split(normalized_content)
        if clause.strip(" .!?;,")
    )
    if len(clauses) <= 1:
        return normalized_content

    query_forms = _lookup_token_forms(query)
    if not query_forms:
        return None
    matches = tuple(
        clause
        for clause in clauses
        if query_forms.issubset(_lookup_token_forms(clause))
    )
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "DEFAULT_MEMORY_RETRIEVAL_POLICY",
    "MemoryLookupResult",
    "MemoryRetrievalPolicy",
    "MemoryAnswerProposal",
    "cosine_similarity",
    "is_memory_lookup_query_grounded",
    "select_memory_fact_clause",
]
