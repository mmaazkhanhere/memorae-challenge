"""Pure contracts and policies for memory retrieval."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class MemoryAnswerProposal(BaseModel):
    """Strict, non-authoritative answer proposed from supplied candidates."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["ANSWER", "AMBIGUOUS", "UNKNOWN"]
    candidate_indices: list[int] = Field(default_factory=list, max_length=3)
    answer: str | None = Field(default=None, min_length=1, max_length=1000)
    clarification_question: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("candidate_indices")
    @classmethod
    def indices_must_be_unique_and_non_negative(cls, value: list[int]) -> list[int]:
        if any(isinstance(index, bool) or index < 0 for index in value):
            raise ValueError("candidate indices must be non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("candidate indices must be unique")
        return value

    @field_validator("answer", "clarification_question")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split()).strip()
        if not value:
            raise ValueError("response text must not be blank")
        return value

    @model_validator(mode="after")
    def decision_fields_must_be_consistent(self) -> MemoryAnswerProposal:
        if self.decision == "ANSWER":
            if not self.candidate_indices or self.answer is None:
                raise ValueError("ANSWER requires supporting candidates and an answer")
            if self.clarification_question is not None:
                raise ValueError("ANSWER must not include a clarification question")
        elif self.decision == "AMBIGUOUS":
            if len(self.candidate_indices) < 2 or self.clarification_question is None:
                raise ValueError("AMBIGUOUS requires conflicting candidates and a question")
            if self.answer is not None:
                raise ValueError("AMBIGUOUS must not include an answer")
        elif self.candidate_indices or self.answer is not None or self.clarification_question is not None:
            raise ValueError("UNKNOWN must not claim candidates or response text")
        return self

    def validates_candidate_count(self, candidate_count: int) -> bool:
        return all(index < candidate_count for index in self.candidate_indices)


_EXACT_MEMORY_VALUE = re.compile(
    r"\b[A-Za-z]+\d[A-Za-z0-9_-]*\b|"
    r"\b\d+[A-Za-z][A-Za-z0-9_-]*\b|"
    r"\b\d+(?:\.\d+)?\b|"
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|"
    r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)"
)


def memory_answer_preserves_exact_values(answer: str, memories: Sequence[str]) -> bool:
    """Require code-like/contact values to remain byte-for-byte present."""

    exact_values = {
        match.group(0) for memory in memories for match in _EXACT_MEMORY_VALUE.finditer(memory)
    }
    return all(value in answer for value in exact_values)


__all__ = [
    "DEFAULT_MEMORY_RETRIEVAL_POLICY",
    "MemoryLookupResult",
    "MemoryRetrievalPolicy",
    "MemoryAnswerProposal",
    "cosine_similarity",
    "is_memory_lookup_query_grounded",
    "memory_answer_preserves_exact_values",
]
