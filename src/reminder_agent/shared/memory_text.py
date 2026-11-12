"""Deterministic memory command extraction shared by routing and workflows."""

from __future__ import annotations

import re

from reminder_agent.shared.creation_routing import ExplicitCreateRoute, explicit_create_route
from reminder_agent.shared.memory_contracts import SaveMemoryFact, _lookup_tokens

_MEMORY_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:remember|save|(?:make|take)\s+(?:a\s+)?note)\b",
    re.IGNORECASE,
)
_MEMORY_FACT_SUFFIX = re.compile(
    r"^\s*(?P<fact>.+?)\s*[.!]\s*(?:please\s+)?save\s+(?:this|that)\s*[.!?]*\s*$",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_FILLER = re.compile(r"^(?:that|this|of|to|for\s+me)\s+", re.IGNORECASE)
_LOOKUP_PATTERNS = (
    re.compile(
        r"^\s*where\s+did\s+i\s+leave\s+(?:my|the)\s+"
        r"(?P<query>.+?)\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+(?P<query>.+?)\s+did\s+i\s+(?:save|remember)\s*"
        r"[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+was\s+(?:the\s+)?name\s+of\s+(?:my|the)\s+"
        r"(?P<query>.+?)\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:what(?:['â€™]s|\s+is)|do\s+you\s+(?:remember|know)|tell\s+me)\s+"
        r"(?:my|the)\s+(?P<query>.+?)\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*where\s+(?:is\s+my|do\s+i)\s+(?P<query>.+?)\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*which\s+(?P<query>.+?)\s+do\s+i\s+(?:like|prefer)(?:\s+most)?\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
)


def is_memory_save_command(text: str) -> bool:
    return explicit_create_route(text) is ExplicitCreateRoute.SAVE_MEMORY


def extract_memory_content(text: str) -> str | None:
    """Extract content only after the shared routing rule selected memory."""

    if not is_memory_save_command(text):
        return None
    match = _MEMORY_PREFIX.match(text)
    if match is not None:
        remainder = _LEADING_FILLER.sub("", text[match.end() :].strip())
    else:
        suffix_match = _MEMORY_FACT_SUFFIX.match(text)
        if suffix_match is None:
            return None
        remainder = suffix_match.group("fact").strip()
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
