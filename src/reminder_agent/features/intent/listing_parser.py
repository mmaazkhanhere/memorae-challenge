"""Deterministic reminder-list task extraction."""

from __future__ import annotations

import re

from reminder_agent.features.intent.contracts import normalize_task_query

LIST_TASK_PATTERNS = (
    re.compile(
        r"^\s*(?:when|what)\s+(?:are|is)\s+(?:my\s+)?(?P<task>.+?)(?:\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday))?\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*when\s+do\s+i\s+have\s+to\s+(?P<task>.+?)\s*[?!.]*\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*(?:show|list)\s+(?:my\s+)?(?P<task>.+?)\s+reminders?(?:\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday))?\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:do\s+i\s+have|are\s+there)\s+(?:any\s+)?(?P<task>.+?)\s+reminders?\s*[?!.]*\s*$",
        re.IGNORECASE,
    ),
)


def extract_listing_task_query(text: str) -> str | None:
    """Return an explicit task/subject filter from supported listing wording."""

    for pattern in LIST_TASK_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        task_query = normalize_task_query(match.group("task"))
        if task_query and task_query.lower() not in {
            "all",
            "my",
            "reminder",
            "scheduled",
            "schedule",
            "any",
        }:
            return task_query
    return None
