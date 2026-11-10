"""Compatibility façade for shared memory domain contracts.

Retrieval-specific contracts and policies live in ``memory.retrieve.domain``;
creation-specific grounding remains here because it is shared by the create
workflow and deterministic intent routing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from reminder_agent.features.memory.retrieve.domain import (
    DEFAULT_MEMORY_RETRIEVAL_POLICY,
    MemoryLookupResult,
    MemoryRetrievalPolicy,
    cosine_similarity,
    is_memory_lookup_query_grounded,
)
from reminder_agent.shared.memory_contracts import MemoryIntentEvidence


@dataclass(frozen=True)
class SavedMemory:
    fact_id: UUID
    saved_at: datetime
    embedding_status: str


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


__all__ = [
    "DEFAULT_MEMORY_RETRIEVAL_POLICY",
    "MemoryLookupResult",
    "MemoryRetrievalPolicy",
    "SavedMemory",
    "cosine_similarity",
    "is_memory_content_grounded",
    "is_memory_intent_evidence_grounded",
    "is_memory_lookup_query_grounded",
]
