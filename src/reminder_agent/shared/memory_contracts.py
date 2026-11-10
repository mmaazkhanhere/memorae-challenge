"""Channel-neutral contracts shared by intent and memory workflows."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MEMORY_OPERATION: Literal["SAVE_MEMORY"] = "SAVE_MEMORY"
MEMORY_LOOKUP_OPERATION: Literal["RETRIEVE_MEMORY"] = "RETRIEVE_MEMORY"
_LOOKUP_IDENTIFIER = re.compile(
    r"\b(?:fact|memory|user)\s*(?:id|uuid)\b|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_LOOKUP_STOP_WORDS = {"a", "an", "and", "for", "my", "of", "the", "to"}


def _lookup_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("-", "").replace("_", "")
    normalized = " ".join(normalized.split())
    return tuple(
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if token not in _LOOKUP_STOP_WORDS and len(token) > 1
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
