"""Pure validation for memory save intent and model-proposed content."""

from __future__ import annotations

import re
import unicodedata

from reminder_agent.shared.memory_contracts import MemoryIntentEvidence


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
