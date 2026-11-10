"""Pure contracts and policies for memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal
from uuid import UUID

from reminder_agent.shared.memory_contracts import RetrieveMemoryQuery, _lookup_tokens


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
    return set(_lookup_tokens(query)).issubset(set(_lookup_tokens(source)))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


__all__ = [
    "DEFAULT_MEMORY_RETRIEVAL_POLICY",
    "MemoryLookupResult",
    "MemoryRetrievalPolicy",
    "cosine_similarity",
    "is_memory_lookup_query_grounded",
]
