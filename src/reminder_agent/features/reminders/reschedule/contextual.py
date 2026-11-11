"""Deterministic corrections that refer to the active voice-call reminder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from reminder_agent.features.reminders.reschedule.models import RescheduleRequest
from reminder_agent.features.reminders.reschedule.parser import parse_reschedule_request


@dataclass(frozen=True)
class ContextualCorrection:
    """A validated correction proposal with no reminder identity attached."""

    reschedule: RescheduleRequest | None = None
    task: str | None = None


_PREFIX = re.compile(
    r"^\s*(?:(?:actually|wait|instead|sorry)\s*[,:-]?\s*)+",
    re.IGNORECASE,
)
_RELATIVE_TIME = re.compile(
    r"^(?:make|set|move|change|reschedule)\s+(?:it|that|the\s+reminder)\s+"
    r"(?:(?:for|in|after)\s+)?(?P<amount>half\s+an?\s+hour|\d+\s+(?:minutes?|hours?))"
    r"(?:\s+(?:from\s+now|later))?\s*[.!?]*$",
    re.IGNORECASE,
)
_RELATIVE_BARE = re.compile(
    r"^(?P<amount>half\s+an?\s+hour|\d+\s+(?:minutes?|hours?))"
    r"(?:\s+(?:from\s+now|later))?\s*[.!?]*$",
    re.IGNORECASE,
)
_ABSOLUTE_TIME = re.compile(
    r"^(?:make|set|move|change|reschedule)\s+(?:it|that|the\s+reminder)\s+"
    r"(?:(?:for|to|until|at)\s+)?(?P<target>.+?)\s*[.!?]*$",
    re.IGNORECASE,
)
_TASK = re.compile(
    r"^(?:change|update|replace)\s+(?:the\s+)?(?:task|reminder|it)\s+"
    r"(?:to|with)\s+(?P<task>.+?)\s*[.!?]*$",
    re.IGNORECASE,
)


def _duration_minutes(value: str) -> int | None:
    normalized = " ".join(value.lower().split())
    if normalized == "half an hour":
        return 30
    match = re.fullmatch(r"(\d+)\s+(minute|minutes|hour|hours)", normalized)
    if match is None:
        return None
    amount = int(match.group(1))
    return amount * (60 if match.group(2).startswith("hour") else 1)


def parse_contextual_correction(
    text: str, *, now: datetime, timezone_name: str
) -> ContextualCorrection | None:
    """Parse only unambiguous active-reminder correction language.

    The active reminder identity is deliberately absent from this result. The
    caller must load and revalidate it from the authenticated voice session.
    """

    candidate = _PREFIX.sub("", text.strip(), count=1)
    task_match = _TASK.fullmatch(candidate)
    if task_match is not None:
        task = " ".join(task_match.group("task").split()).strip(" .,!?")
        return ContextualCorrection(task=task) if task else None

    relative_match = _RELATIVE_TIME.fullmatch(candidate) or _RELATIVE_BARE.fullmatch(candidate)
    if relative_match is not None:
        minutes = _duration_minutes(relative_match.group("amount"))
        if minutes is None or minutes <= 0:
            return None
        return ContextualCorrection(
            reschedule=RescheduleRequest(
                target_text="active reminder",
                due_at=now + timedelta(minutes=minutes),
                date_explicit=True,
            )
        )

    absolute_match = _ABSOLUTE_TIME.fullmatch(candidate)
    if absolute_match is None:
        return None
    parsed = parse_reschedule_request(
        f"change it to {absolute_match.group('target')}",
        timezone_name=timezone_name,
        now=now,
    )
    if not isinstance(parsed, RescheduleRequest):
        return None
    return ContextualCorrection(
        reschedule=RescheduleRequest(
            target_text="active reminder",
            due_at=parsed.due_at,
            date_explicit=parsed.date_explicit,
            offset_minutes=parsed.offset_minutes,
            scope=parsed.scope,
            occurrence_date=parsed.occurrence_date,
            destination_date=parsed.destination_date,
            destination_weekday=parsed.destination_weekday,
        )
    )


__all__ = ["ContextualCorrection", "parse_contextual_correction"]
