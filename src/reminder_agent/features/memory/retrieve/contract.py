"""Provider-independent contracts for grounded memory answers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
        elif (
            self.candidate_indices
            or self.answer is not None
            or self.clarification_question is not None
        ):
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


__all__ = ["MemoryAnswerProposal", "memory_answer_preserves_exact_values"]
