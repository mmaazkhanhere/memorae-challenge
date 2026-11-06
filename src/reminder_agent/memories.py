"""Shared, owner-scoped memory-fact commands and persistence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from reminder_agent.db.models import MemoryFact
from reminder_agent.intent_routing_rules import ExplicitCreateRoute, explicit_create_route

MEMORY_OPERATION = "SAVE_MEMORY"
_MEMORY_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:remember|save|(?:make|take)\s+(?:a\s+)?note)\b",
    re.IGNORECASE,
)
_LEADING_FILLER = re.compile(r"^(?:that|this|of|to|for\s+me)\s+", re.IGNORECASE)


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
