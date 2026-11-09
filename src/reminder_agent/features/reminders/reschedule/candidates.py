"""Candidate discovery and non-transactional reschedule resolution."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.features.reminders.recurrence import resolve_local_datetime
from reminder_agent.features.reminders.reschedule.models import (
    ACTIVE,
    PENDING,
    PendingReschedule,
    PendingRescheduleProposal,
    RescheduleCandidate,
    RescheduleRequest,
)

_ORDINALS = {
    "first": 0,
    "one": 0,
    "1": 0,
    "second": 1,
    "two": 1,
    "2": 1,
    "third": 2,
    "three": 2,
    "3": 2,
}
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "reminder",
    "reminders",
    "this",
    "occurrence",
    "series",
    "future",
    "all",
    "my",
    "please",
    "one",
    "first",
    "second",
    "third",
    "1",
    "2",
    "3",
}


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in _STOP_WORDS}


def _candidate_match(target: str, task: str) -> bool:
    """Match only when the user supplied enough task detail to be safe.

    A shared verb such as ``call`` must not cause "call the dentist" to
    select "call the bank".  If the target is too vague, the caller is asked
    to identify the reminder instead of guessing.
    """

    wanted = _tokens(target)
    actual = _tokens(task)
    if not wanted or not actual:
        return False
    return wanted <= actual


def find_candidates(
    db: Session,
    *,
    user_id: UUID,
    target_text: str,
    occurrence_date: date | None = None,
    timezone_name: str = "UTC",
    reminder_model: Any,
    occurrence_model: Any,
) -> list[RescheduleCandidate]:
    rows = db.execute(
        select(reminder_model, occurrence_model)
        .join(occurrence_model, occurrence_model.reminder_id == reminder_model.id)
        .where(
            reminder_model.user_id == user_id,
            reminder_model.status == ACTIVE,
            occurrence_model.status == PENDING,
        )
        .order_by(occurrence_model.next_due_at, reminder_model.id, occurrence_model.id)
    ).all()
    zone = ZoneInfo(timezone_name)
    candidates = [
        RescheduleCandidate(
            reminder_id=reminder.id,
            occurrence_id=occurrence.id,
            task=reminder.task,
            due_at=occurrence.next_due_at,
            schedule_type=reminder.schedule_type,
        )
        for reminder, occurrence in rows
        if _candidate_match(target_text, reminder.task)
        and (
            occurrence_date is None
            or occurrence.next_due_at.astimezone(zone).date() == occurrence_date
        )
    ]
    # A recurring series may have many materialized occurrences; candidate
    # selection is by reminder, with its next pending occurrence as the anchor.
    unique: dict[UUID, RescheduleCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.reminder_id, candidate)
    return list(unique.values())


def find_all_candidates(
    db: Session,
    *,
    user_id: UUID,
    reminder_model: Any,
    occurrence_model: Any,
) -> list[RescheduleCandidate]:
    """Return all active one-occurrence anchors for LLM disambiguation context."""
    rows = db.execute(
        select(reminder_model, occurrence_model)
        .join(occurrence_model, occurrence_model.reminder_id == reminder_model.id)
        .where(
            reminder_model.user_id == user_id,
            reminder_model.status == ACTIVE,
            occurrence_model.status == PENDING,
        )
        .order_by(occurrence_model.next_due_at, reminder_model.id, occurrence_model.id)
    ).all()
    unique: dict[UUID, RescheduleCandidate] = {}
    for reminder, occurrence in rows:
        unique.setdefault(
            reminder.id,
            RescheduleCandidate(
                reminder_id=reminder.id,
                occurrence_id=occurrence.id,
                task=reminder.task,
                due_at=occurrence.next_due_at,
                schedule_type=reminder.schedule_type,
            ),
        )
    return list(unique.values())


def format_candidates(candidates: list[RescheduleCandidate], *, timezone_name: str) -> str:
    """Render stored UTC candidate times in the user's local timezone."""
    zone = ZoneInfo(timezone_name)
    lines: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        due_at = candidate.due_at
        if due_at.tzinfo is None or due_at.utcoffset() is None:
            due_at = due_at.replace(tzinfo=UTC)
        local_due = due_at.astimezone(zone)
        lines.append(f"{index}. {candidate.task} (currently {local_due:%Y-%m-%d %H:%M %Z})")
    return "\n".join(lines)


def resolve_candidate_due(
    request: RescheduleRequest,
    candidate: RescheduleCandidate,
    *,
    timezone_name: str = "UTC",
) -> RescheduleRequest:
    """Resolve a model-provided relative offset against the selected occurrence."""
    if request.due_at is not None:
        if request.date_explicit:
            return request
        zone = ZoneInfo(timezone_name)
        candidate_local = candidate.due_at
        if candidate_local.tzinfo is None or candidate_local.utcoffset() is None:
            candidate_local = candidate_local.replace(tzinfo=UTC)
        candidate_local = candidate_local.astimezone(zone)
        requested_local = request.due_at.astimezone(zone)
        preserved_local = resolve_local_datetime(
            candidate_local.date(), requested_local.timetz().replace(tzinfo=None), zone
        )
        return RescheduleRequest(
            target_text=request.target_text,
            due_at=preserved_local.astimezone(UTC),
            date_explicit=False,
            scope=request.scope,
            source_text=request.source_text,
            occurrence_date=request.occurrence_date,
            destination_weekday=request.destination_weekday,
        )
    if request.offset_minutes is None:
        raise ValueError("reschedule request is missing a time target")
    candidate_due_at = candidate.due_at
    if candidate_due_at.tzinfo is None or candidate_due_at.utcoffset() is None:
        candidate_due_at = candidate_due_at.replace(tzinfo=UTC)
    return RescheduleRequest(
        target_text=request.target_text,
        due_at=candidate_due_at + timedelta(minutes=request.offset_minutes),
        date_explicit=request.date_explicit,
        scope=request.scope,
        source_text=request.source_text,
        occurrence_date=request.occurrence_date,
        destination_weekday=request.destination_weekday,
    )


def apply_pending_reschedule_proposal(
    pending: PendingReschedule,
    proposal: PendingRescheduleProposal,
    *,
    timezone_name: str,
) -> PendingReschedule:
    """Merge explicit parameters into trusted pending state without selecting a row."""

    values = pending.model_dump(mode="python")
    if proposal.due_at is not None:
        values["due_at"] = proposal.due_at
        values["offset_minutes"] = None
    elif proposal.offset_minutes is not None:
        values["due_at"] = None
        values["offset_minutes"] = proposal.offset_minutes
    elif proposal.destination_date is not None:
        if pending.due_at is None:
            raise ValueError("a date correction needs an existing reminder time")
        zone = ZoneInfo(timezone_name)
        existing_local = pending.due_at.astimezone(zone)
        local_due = resolve_local_datetime(
            proposal.destination_date, existing_local.timetz().replace(tzinfo=None), zone
        )
        values["due_at"] = local_due.astimezone(UTC)
        values["offset_minutes"] = None
        values["date_explicit"] = True
    if proposal.scope is not None:
        values["scope"] = proposal.scope
    if proposal.date_explicit is not None:
        values["date_explicit"] = proposal.date_explicit
    return PendingReschedule.model_validate(values)


def pending_candidates(
    db: Session,
    *,
    user_id: UUID,
    pending: PendingReschedule,
    find_candidates_fn: Callable[..., list[RescheduleCandidate]] = find_candidates,
) -> list[RescheduleCandidate]:
    """Reload only active, owned candidate IDs authorized by a session."""

    allowed = set(pending.candidate_ids)
    return [
        candidate
        for candidate in find_candidates_fn(
            db,
            user_id=user_id,
            target_text=pending.target_text,
            occurrence_date=pending.occurrence_date,
            timezone_name=pending.timezone_name,
        )
        if candidate.reminder_id in allowed
    ]


def select_pending_candidate_ids(
    db: Session,
    *,
    user_id: UUID,
    pending: PendingReschedule,
    text: str,
    timezone_name: str,
    now: datetime,
    find_candidates_fn: Callable[..., list[RescheduleCandidate]] | None = None,
) -> tuple[UUID, ...]:
    """Resolve ordinal or clock-based natural candidate references safely."""
    if find_candidates_fn is None:
        find_candidates_fn = find_candidates
    # Keep matching deliberately narrow for deterministic text selection;
    # IDs are still constrained to the durable pending candidate list below.
    if pending.occurrence_date is not None:
        candidates = find_candidates_fn(
            db,
            user_id=user_id,
            target_text=pending.target_text,
            occurrence_date=pending.occurrence_date,
            timezone_name=timezone_name,
        )
    else:
        candidates = find_candidates_fn(
            db,
            user_id=user_id,
            target_text=pending.target_text,
        )
    candidate_ids = list(pending.candidate_ids)
    normalized = text.strip().lower().rstrip(".!?")
    clock = re.fullmatch(
        r"(?:the\s+)?(?:at\s+|for\s+|scheduled\s+(?:at|for)\s+)?"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)(?:\s+(?:one|reminder|occurrence))?",
        normalized,
    )
    if clock is not None:
        hour = int(clock.group(1)) % 12
        if clock.group(3) == "pm":
            hour += 12
        minute = int(clock.group(2) or 0)
        zone = ZoneInfo(timezone_name)
        matches = []
        for candidate in candidates:
            due_at = candidate.due_at
            if due_at.tzinfo is None or due_at.utcoffset() is None:
                due_at = due_at.replace(tzinfo=UTC)
            if (
                candidate.reminder_id in candidate_ids
                and due_at.astimezone(zone).hour == hour
                and due_at.astimezone(zone).minute == minute
            ):
                matches.append(candidate.reminder_id)
        if len(matches) == 1:
            return (matches[0],)
        return tuple(candidate_ids)
    day_match = re.fullmatch(
        r"(?:the\s+)?(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:one|reminder|occurrence)",
        normalized,
    )
    if day_match is not None:
        zone = ZoneInfo(timezone_name)
        requested_day = day_match.group(1)
        local_now = now.astimezone(zone)
        if requested_day == "today":
            requested_date = local_now.date()
        elif requested_day == "tomorrow":
            requested_date = local_now.date() + timedelta(days=1)
        else:
            requested_date = None
        matches = []
        for candidate in candidates:
            due_at = candidate.due_at
            if due_at.tzinfo is None or due_at.utcoffset() is None:
                due_at = due_at.replace(tzinfo=UTC)
            local_due = due_at.astimezone(zone)
            if requested_date is not None:
                matches_if = local_due.date() == requested_date
            else:
                matches_if = local_due.strftime("%A").lower() == requested_day
            if candidate.reminder_id in candidate_ids and matches_if:
                matches.append(candidate.reminder_id)
        if len(matches) == 1:
            return (matches[0],)
        return tuple(candidate_ids)
    ordinal = re.fullmatch(
        r"(?:the\s+)?(first|one|second|two|third|three|1|2|3)(?:\s+(?:one|reminder|occurrence))?",
        normalized,
    )
    if ordinal is not None and len(candidate_ids) > 1:
        selected = _ORDINALS.get(ordinal.group(1))
        if selected is not None and selected < len(candidate_ids):
            return (candidate_ids[selected],)
    return tuple(candidate_ids)


__all__ = [
    "apply_pending_reschedule_proposal",
    "find_all_candidates",
    "find_candidates",
    "format_candidates",
    "pending_candidates",
    "resolve_candidate_due",
    "select_pending_candidate_ids",
]
