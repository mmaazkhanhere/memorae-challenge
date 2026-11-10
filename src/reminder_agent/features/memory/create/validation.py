"""Validation for proposed memory creation before any mutation."""

from __future__ import annotations

import re

from reminder_agent.features.memory.memory_domain import is_memory_content_grounded
from reminder_agent.shared.memory_contracts import PendingMemory, SaveMemoryFact
from reminder_agent.shared.memory_text import extract_memory_content

_INCOMPLETE_FACT_PHRASE = re.compile(
    r"^\s*(?:my|the|our|your)\s+"
    r"(?:[\w'-]+\s+){0,6}"
    r"(?:number|code|password|passcode|pin|email|e-mail|address|phone|"
    r"phone\s+number|name|birthday|birth\s+date|favorite\s+food|"
    r"favourite\s+food|favorite\s+color|favourite\s+colour|parking\s+slot|"
    r"locker\s+number)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_INCOMPLETE_FACT_CLAUSE = re.compile(r"(?:\bis\b|\bare\b|\bequals\b|[:=])\s*[?.!]*$", re.IGNORECASE)


def _is_incomplete_fact_phrase(content: str) -> bool:
    """Reject a subject-only fact such as ``my locker number``."""

    normalized = content.strip(" .!?\t")
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    if (
        len(tokens) <= 3
        and tokens
        and tokens[0].casefold() in {"my", "the", "our", "your"}
        and not any(char.isdigit() for char in normalized)
        and not re.search(r"\b(?:is|are|equals)\b|[:=]", normalized, re.IGNORECASE)
        and "@" not in normalized
    ):
        return True
    return (
        _INCOMPLETE_FACT_PHRASE.fullmatch(content) is not None
        or _INCOMPLETE_FACT_CLAUSE.search(content) is not None
    )


def validated_memory_content(
    *,
    text: str,
    pending: PendingMemory | None,
    model_content: str | None,
    from_model: bool,
) -> str | None:
    """Return validated, grounded content or require a non-mutating clarification."""

    content = (
        model_content
        if from_model
        else (extract_memory_content(text) if pending is None else text.strip())
    )
    if content is None:
        return None
    try:
        content = SaveMemoryFact(content=content).content
    except (TypeError, ValueError):
        return None
    if _is_incomplete_fact_phrase(content):
        return None
    if from_model and not is_memory_content_grounded(text, content):
        return None
    return content
