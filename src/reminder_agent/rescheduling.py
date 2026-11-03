"""Parsing and transactional application of reminder reschedules.

This module deliberately contains no Telegram or model code.  Callers provide
the canonical user identity resolved by the channel adapter; all reminder
ownership and state checks are repeated inside the write transaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.db.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
)
from reminder_agent.recurrence import parse_recurrence_rule, resolve_local_datetime


ACTIVE = "ACTIVE"
PENDING = "PENDING"
SENDING = "SENDING"
CANCELED = "CANCELED"
RECURRING = "RECURRING"
ONE_TIME = "ONE_TIME"


class RescheduleScope(StrEnum):
    OCCURRENCE = "OCCURRENCE"
    SERIES = "SERIES"


class PendingRescheduleOperation(StrEnum):
    """Non-authoritative interpretations of an unfinished reschedule."""

    SELECT_CANDIDATE = "SELECT_CANDIDATE"
    UPDATE_PENDING_PARAMETERS = "UPDATE_PENDING_PARAMETERS"
    SELECT_AND_UPDATE = "SELECT_AND_UPDATE"
    START_NEW_OPERATION = "START_NEW_OPERATION"
    CANCEL_PENDING = "CANCEL_PENDING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class PendingReschedule(BaseModel):
    """Durable, non-authoritative context for a follow-up question."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["RESCHEDULE_REMINDER"] = "RESCHEDULE_REMINDER"
    target_text: str = Field(min_length=1, max_length=200)
    due_at: datetime | None = None
    date_explicit: bool = False
    offset_minutes: int | None = Field(default=None, ge=-10080, le=10080)
    source_text: str | None = Field(default=None, max_length=500)
    timezone_name: str = Field(min_length=1, max_length=150)
    candidate_ids: tuple[UUID, ...] = ()
    scope: RescheduleScope | None = None
    occurrence_date: date | None = None

    @field_validator("target_text")
    @classmethod
    def target_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("target_text must not be blank")
        return value

    @field_validator("due_at")
    @classmethod
    def due_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("due_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def has_exactly_one_time_target(self) -> "PendingReschedule":
        if self.due_at is None and self.offset_minutes is None and not self.source_text:
            raise ValueError("pending reschedule needs a time target or source request")
        if self.due_at is not None and self.offset_minutes is not None:
            raise ValueError("pending reschedule cannot have both absolute and relative targets")
        return self

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        ZoneInfo(value)
        return value


class PendingRescheduleProposal(BaseModel):
    """Validated model proposal, never a command to mutate the database.

    Candidate IDs are intentionally opaque UUIDs.  The application validates
    membership in ``PendingReschedule.candidate_ids`` again while holding the
    conversation lock before using one.
    """

    model_config = ConfigDict(extra="forbid")

    operation: PendingRescheduleOperation
    candidate_id: UUID | None = None
    due_at: datetime | None = None
    destination_date: date | None = None
    date_explicit: bool | None = None
    offset_minutes: int | None = Field(default=None, ge=-10080, le=10080)
    scope: RescheduleScope | None = None
    question: str | None = Field(default=None, max_length=500)

    @field_validator("due_at")
    @classmethod
    def proposal_due_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("due_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def proposal_is_coherent(self) -> "PendingRescheduleProposal":
        has_time = self.due_at is not None or self.offset_minutes is not None
        has_update = has_time or self.destination_date is not None or self.scope is not None
        if self.due_at is not None and self.offset_minutes is not None:
            raise ValueError("proposal cannot have both absolute and relative targets")
        if self.destination_date is not None and self.due_at is not None:
            raise ValueError("proposal cannot have both a local date and absolute datetime")
        if self.operation in {
            PendingRescheduleOperation.SELECT_CANDIDATE,
            PendingRescheduleOperation.SELECT_AND_UPDATE,
        } and self.candidate_id is None:
            raise ValueError("candidate selection proposal needs a candidate ID")
        if self.operation is PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS and not has_update:
            raise ValueError("parameter update proposal needs an update")
        if self.operation is PendingRescheduleOperation.SELECT_AND_UPDATE and not has_update:
            raise ValueError("combined proposal needs an update")
        if self.operation in {
            PendingRescheduleOperation.CANCEL_PENDING,
            PendingRescheduleOperation.START_NEW_OPERATION,
            PendingRescheduleOperation.NEEDS_CLARIFICATION,
        } and (self.candidate_id is not None or has_update):
            raise ValueError("non-update proposal contains action fields")
        return self


@dataclass(frozen=True)
class RescheduleRequest:
    target_text: str
    due_at: datetime | None = None
    date_explicit: bool = False
    offset_minutes: int | None = None
    scope: RescheduleScope | None = None
    source_text: str | None = None
    occurrence_date: date | None = None


@dataclass(frozen=True)
class RescheduleCandidate:
    reminder_id: UUID
    occurrence_id: UUID
    task: str
    due_at: datetime
    schedule_type: str


@dataclass(frozen=True)
class RescheduleDecision:
    """Result of safe parsing/candidate resolution before a write transaction."""

    kind: Literal["MUTATE", "CLARIFY", "RESPONSE", "CANCEL"]
    text: str
    request: RescheduleRequest | None = None
    candidate_ids: tuple[UUID, ...] = ()
    occurrence_id: UUID | None = None


_COMMAND = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:move|reschedule|change|shift|push|postpone|delay)\s+(?:the\s+)?(?P<target>.+?)\s+"
    r"(?P<separator>to|until|for|at)\s+(?P<when>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_COMMAND_GREEDY = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:move|reschedule|change|shift|push|postpone|delay)\s+(?:the\s+)?(?P<target>.+)\s+"
    r"(?P<separator>to|until|for)\s+(?P<when>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_WHEN = re.compile(
    r"^(?:(?P<day>today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{4}-\d{1,2}-\d{1,2})\s+)?(?:at\s+)?(?P<hour>\d{1,2})"
    r"(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.?m\.?|p\.?m\.?|am|pm)?$",
    re.IGNORECASE,
)
_ORDINALS = {"first": 0, "one": 0, "1": 0, "second": 1, "two": 1, "2": 1, "third": 2, "three": 2, "3": 2}
_STOP_WORDS = {
    "a", "an", "the", "reminder", "reminders", "this", "occurrence", "series",
    "future", "all", "my", "please", "one", "first", "second", "third", "1", "2", "3",
}
_DAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_OCCURRENCE_SELECTOR = re.compile(
    r"\b(?:(?P<only>only)\s+)?(?P<day>today|tomorrow|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)'s\s+",
    re.IGNORECASE,
)


def is_reschedule_command(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:please\s+)?"
            r"(?:move|reschedule|change|shift|push|postpone|delay)\b",
            text,
            re.I,
        )
    )


def _scope_from_text(text: str) -> RescheduleScope | None:
    normalized = text.lower()
    if normalized.strip() in {"all", "all of them", "everything"}:
        return RescheduleScope.SERIES
    if re.search(
        r"\b(?:whole\s+)?series\b|\ball\s+(?:future\s+)?(?:reminders?|occurrences?)\b|"
        r"\ball\s+future\b.*\b(?:reminders?|occurrences?)\b|"
        r"\bupdate\s+all\b|\bevery\b",
        normalized,
    ):
        return RescheduleScope.SERIES
    if re.search(
        r"\b(?:next|upcoming|coming)\s+(?:occurrence|reminder|one)\b|"
        r"\bonly\s+(?:the\s+)?(?:next|upcoming|coming)(?:\s+(?:occurrence|reminder|one))?\b|"
        r"\b(?:only\s+)?(?:this|coming)\s+one\b|\bthis\s+(?:occurrence|reminder)\b",
        normalized,
    ):
        return RescheduleScope.OCCURRENCE
    return None


def _parse_date(raw: str | None, *, now: datetime, zone: ZoneInfo) -> date:
    local_now = now.astimezone(zone)
    if raw is None:
        return local_now.date()
    value = raw.lower()
    if value == "today":
        return local_now.date()
    if value == "tomorrow":
        return local_now.date() + timedelta(days=1)
    if value in _DAY_INDEX:
        delta = (_DAY_INDEX[value] - local_now.weekday()) % 7
        return local_now.date() + timedelta(days=delta or 7)
    return date.fromisoformat(value)


def _extract_target_and_occurrence(
    target: str, *, timezone_name: str, now: datetime
) -> tuple[str, RescheduleScope | None, date | None]:
    """Remove an explicit occurrence selector before matching the task."""

    scope = _scope_from_text(target)
    occurrence_date = None
    selector = _OCCURRENCE_SELECTOR.search(target)
    if selector is not None:
        occurrence_date = _parse_date(
            selector.group("day"), now=now, zone=ZoneInfo(timezone_name)
        )
        # A possessive day selector identifies one concrete recurrence
        # instance; "only" reinforces rather than creates that scope.
        scope = RescheduleScope.OCCURRENCE
        target = target[:selector.start()] + target[selector.end():]
    target = re.sub(
        r"\b(?:all\s+future\s+reminders?|all\s+reminders?|this\s+occurrence|this\s+reminder|"
        r"the\s+whole\s+series|the\s+series|whole\s+series|series|next\s+occurrence|"
        r"upcoming\s+reminder|only\s+this\s+one|only\s+the\s+next|coming\s+one|occurrence)\b",
        " ", target, flags=re.I,
    )
    return " ".join(target.split()).strip(" .,!?"), scope, occurrence_date


def parse_reschedule_request(text: str, *, timezone_name: str, now: datetime) -> RescheduleRequest | str:
    """Parse the deliberately small, unambiguous VS-09 command language."""

    match = _COMMAND.match(text)
    if match is None:
        return "Please say which reminder to move and give a new future time."
    target = " ".join(match.group("target").split())
    target = re.sub(r"^my\s+", "", target, flags=re.I)
    target = re.sub(r"^(?:my\s+)?(?:a\s+)?reminders?\s+to\s+", "", target, flags=re.I)
    if not target:
        return "Please tell me which reminder to move."
    target, scope, occurrence_date = _extract_target_and_occurrence(
        target, timezone_name=timezone_name, now=now
    )
    if not target:
        return "Please tell me which reminder to move."

    when = _WHEN.match(match.group("when"))
    if when is None:
        fallback_match = _COMMAND_GREEDY.match(text)
        if fallback_match is not None:
            fallback_when = _WHEN.match(fallback_match.group("when"))
            if fallback_when is not None:
                match = fallback_match
                target = " ".join(match.group("target").split())
                target = re.sub(r"^my\s+", "", target, flags=re.I)
                target = re.sub(r"^(?:my\s+)?(?:a\s+)?reminders?\s+to\s+", "", target, flags=re.I)
                target, scope, occurrence_date = _extract_target_and_occurrence(
                    target, timezone_name=timezone_name, now=now
                )
                when = fallback_when
    if when is None:
        return "Please give the new time as a future date and time, for example tomorrow at 10 AM."
    hour = int(when.group("hour"))
    minute = int(when.group("minute") or 0)
    meridiem = (when.group("meridiem") or "").lower().replace(".", "")
    if minute > 59 or hour > (12 if meridiem else 23) or (meridiem and hour == 0):
        return "Please provide a valid clock time."
    if not meridiem and 1 <= hour <= 12:
        return f"Do you mean {hour} AM or {hour} PM for the new reminder time?"
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    zone = ZoneInfo(timezone_name)
    try:
        local_date = _parse_date(when.group("day"), now=now, zone=zone)
        local_due = resolve_local_datetime(local_date, time(hour, minute), zone)
    except (TypeError, ValueError):
        return "Please provide a valid future date and time."
    due_at = local_due.astimezone(timezone.utc)
    if when.group("day") is None and due_at <= now.astimezone(timezone.utc):
        local_due = resolve_local_datetime(local_date + timedelta(days=1), time(hour, minute), zone)
        due_at = local_due.astimezone(timezone.utc)
    if due_at <= now.astimezone(timezone.utc):
        return "Please choose a future time for the reminder."
    return RescheduleRequest(
        target_text=target,
        due_at=due_at,
        date_explicit=when.group("day") is not None,
        scope=scope,
        occurrence_date=occurrence_date,
    )


def parse_pending_reschedule_parameters(
    text: str, *, timezone_name: str, now: datetime
) -> PendingRescheduleProposal | None:
    """Recognize only whole, unambiguous parameter-only follow-ups."""

    normalized = text.strip().rstrip(".!?")
    date_only = re.fullmatch(
        r"(?:no,?\s+)?i\s+meant\s+"
        r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        normalized,
        re.IGNORECASE,
    )
    if date_only is not None:
        try:
            value = _parse_date(date_only.group(1), now=now, zone=ZoneInfo(timezone_name))
        except ValueError:
            return None
        return PendingRescheduleProposal(
            operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
            destination_date=value,
            date_explicit=True,
        )
    match = re.fullmatch(
        r"(?:actually,?\s+)?(?:make|move|change)\s+(?:it|that)(?:\s+to)?\s+(.+)",
        normalized,
        re.IGNORECASE,
    )
    if match is None:
        return None
    parsed = parse_reschedule_request(
        f"Move placeholder to {match.group(1)}", timezone_name=timezone_name, now=now
    )
    if not isinstance(parsed, RescheduleRequest):
        return None
    return PendingRescheduleProposal(
        operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
        due_at=parsed.due_at,
        offset_minutes=parsed.offset_minutes,
        date_explicit=parsed.date_explicit,
        scope=parsed.scope,
    )


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in _STOP_WORDS
    }


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
    db: Session, *, user_id: UUID, target_text: str,
    occurrence_date: date | None = None, timezone_name: str = "UTC",
) -> list[RescheduleCandidate]:
    rows = db.execute(
        select(Reminder, ReminderOccurrence)
        .join(ReminderOccurrence, ReminderOccurrence.reminder_id == Reminder.id)
        .where(
            Reminder.user_id == user_id,
            Reminder.status == ACTIVE,
            ReminderOccurrence.status == PENDING,
        )
        .order_by(ReminderOccurrence.next_due_at, Reminder.id, ReminderOccurrence.id)
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


def find_all_candidates(db: Session, *, user_id: UUID) -> list[RescheduleCandidate]:
    """Return all active one-occurrence anchors for LLM disambiguation context."""
    rows = db.execute(
        select(Reminder, ReminderOccurrence)
        .join(ReminderOccurrence, ReminderOccurrence.reminder_id == Reminder.id)
        .where(
            Reminder.user_id == user_id,
            Reminder.status == ACTIVE,
            ReminderOccurrence.status == PENDING,
        )
        .order_by(ReminderOccurrence.next_due_at, Reminder.id, ReminderOccurrence.id)
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
            due_at = due_at.replace(tzinfo=timezone.utc)
        local_due = due_at.astimezone(zone)
        lines.append(
            f"{index}. {candidate.task} (currently {local_due:%Y-%m-%d %H:%M %Z})"
        )
    return "\n".join(lines)


def decide_reschedule(
    db: Session,
    *,
    user_id: UUID,
    text: str,
    timezone_name: str,
    now: datetime,
    pending: PendingReschedule | None = None,
    request_override: RescheduleRequest | None = None,
    candidate_id_override: UUID | None = None,
) -> RescheduleDecision:
    normalized = text.strip().lower().rstrip(".!?")
    if pending is not None:
        if normalized in {"cancel", "never mind", "nevermind", "not now"}:
            return RescheduleDecision("CANCEL", "Okay, I won't move that reminder.")
        scope = _scope_from_text(text) or pending.scope
        candidate_ids = (
            (candidate_id_override,)
            if candidate_id_override is not None and candidate_id_override in pending.candidate_ids
            else select_pending_candidate_ids(
                db,
                user_id=user_id,
                pending=pending,
                text=text,
                timezone_name=timezone_name,
                now=now,
            )
        )
        if len(candidate_ids) != 1:
            return RescheduleDecision(
                "CLARIFY",
                "Which reminder should I move?\n" + format_candidates(
                    pending_candidates(db, user_id=user_id, pending=pending),
                    timezone_name=timezone_name,
                ),
                request=RescheduleRequest(
                    target_text=pending.target_text,
                    due_at=pending.due_at,
                    date_explicit=pending.date_explicit,
                    offset_minutes=pending.offset_minutes,
                    scope=scope,
                    source_text=pending.source_text,
                    occurrence_date=pending.occurrence_date,
                ),
                candidate_ids=pending.candidate_ids,
            )
        candidates_by_id = {
            candidate.reminder_id: candidate
            for candidate in pending_candidates(db, user_id=user_id, pending=pending)
        }
        candidate = candidates_by_id.get(candidate_ids[0])
        if candidate is None:
            # The write path locks and rechecks ownership/activity.  Returning
            # a mutation proposal here preserves the existing user-facing
            # conflict response if a candidate disappeared after clarification.
            if pending.due_at is not None:
                return RescheduleDecision(
                    "MUTATE",
                    "",
                    RescheduleRequest(
                        target_text=pending.target_text,
                        due_at=pending.due_at,
                        date_explicit=pending.date_explicit,
                        scope=scope,
                        occurrence_date=pending.occurrence_date,
                    ),
                    candidate_ids,
                )
            return RescheduleDecision("RESPONSE", "That reminder is no longer active.")
        if candidate.schedule_type == RECURRING and scope is None:
            return RescheduleDecision(
                "CLARIFY",
                "Should I move only the upcoming reminder, or update all future reminders in the series?",
                candidate_ids=candidate_ids,
            )
        return RescheduleDecision(
            "MUTATE",
            "",
            _resolve_candidate_due(
                RescheduleRequest(
                    target_text=pending.target_text,
                    due_at=pending.due_at,
                    date_explicit=pending.date_explicit,
                    offset_minutes=pending.offset_minutes,
                    scope=scope,
                    occurrence_date=pending.occurrence_date,
                ),
                candidate,
                timezone_name=timezone_name,
            ),
            candidate_ids,
            candidate.occurrence_id,
        )

    parsed = request_override or parse_reschedule_request(
        text, timezone_name=timezone_name, now=now
    )
    if isinstance(parsed, str):
        return RescheduleDecision("RESPONSE", parsed)
    candidates = find_candidates(
        db, user_id=user_id, target_text=parsed.target_text,
        occurrence_date=parsed.occurrence_date, timezone_name=timezone_name,
    )
    if not candidates:
        return RescheduleDecision("RESPONSE", "I couldn't find an active reminder matching that description.")
    if len(candidates) > 1:
        return RescheduleDecision(
            "CLARIFY",
            "I found multiple matching reminders. Which one should I move?\n"
            + format_candidates(candidates, timezone_name=timezone_name),
            request=parsed,
            candidate_ids=tuple(candidate.reminder_id for candidate in candidates),
        )
    candidate = candidates[0]
    if candidate.schedule_type == RECURRING and parsed.scope is None:
        return RescheduleDecision(
            "CLARIFY",
            "Should I move only the upcoming reminder, or update all future reminders in the series?",
            request=parsed,
            candidate_ids=(candidate.reminder_id,),
        )
    return RescheduleDecision(
        "MUTATE",
        "",
        _resolve_candidate_due(parsed, candidate, timezone_name=timezone_name),
        (candidate.reminder_id,),
        candidate.occurrence_id,
    )


def _resolve_candidate_due(
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
            candidate_local = candidate_local.replace(tzinfo=timezone.utc)
        candidate_local = candidate_local.astimezone(zone)
        requested_local = request.due_at.astimezone(zone)
        preserved_local = resolve_local_datetime(
            candidate_local.date(), requested_local.timetz().replace(tzinfo=None), zone
        )
        return RescheduleRequest(
            target_text=request.target_text,
            due_at=preserved_local.astimezone(timezone.utc),
            date_explicit=False,
            scope=request.scope,
            source_text=request.source_text,
            occurrence_date=request.occurrence_date,
        )
    if request.offset_minutes is None:
        raise ValueError("reschedule request is missing a time target")
    candidate_due_at = candidate.due_at
    if candidate_due_at.tzinfo is None or candidate_due_at.utcoffset() is None:
        candidate_due_at = candidate_due_at.replace(tzinfo=timezone.utc)
    return RescheduleRequest(
        target_text=request.target_text,
        due_at=candidate_due_at + timedelta(minutes=request.offset_minutes),
        date_explicit=request.date_explicit,
        scope=request.scope,
        source_text=request.source_text,
        occurrence_date=request.occurrence_date,
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
        values["due_at"] = local_due.astimezone(timezone.utc)
        values["offset_minutes"] = None
        values["date_explicit"] = True
    if proposal.scope is not None:
        values["scope"] = proposal.scope
    if proposal.date_explicit is not None:
        values["date_explicit"] = proposal.date_explicit
    return PendingReschedule.model_validate(values)


def pending_candidates(
    db: Session, *, user_id: UUID, pending: PendingReschedule
) -> list[RescheduleCandidate]:
    """Reload only active, owned candidate IDs authorized by a session."""

    allowed = set(pending.candidate_ids)
    return [
        candidate
        for candidate in find_candidates(
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
) -> tuple[UUID, ...]:
    """Resolve ordinal or clock-based natural candidate references safely."""
    # Keep matching deliberately narrow for deterministic text selection;
    # IDs are still constrained to the durable pending candidate list below.
    candidate_kwargs = {"user_id": user_id, "target_text": pending.target_text}
    if pending.occurrence_date is not None:
        candidate_kwargs.update(
            occurrence_date=pending.occurrence_date, timezone_name=timezone_name
        )
    candidates = find_candidates(db, **candidate_kwargs)
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
                due_at = due_at.replace(tzinfo=timezone.utc)
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
                due_at = due_at.replace(tzinfo=timezone.utc)
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
        r"(?:the\s+)?(first|one|second|two|third|three|1|2|3)(?:\s+(?:one|reminder|occurrence))?", normalized
    )
    if ordinal is not None and len(candidate_ids) > 1:
        selected = _ORDINALS.get(ordinal.group(1))
        if selected is not None and selected < len(candidate_ids):
            return (candidate_ids[selected],)
    return tuple(candidate_ids)


def _next_sequence(db: Session, occurrence_id: UUID) -> int:
    current = db.scalar(select(func.max(Notification.sequence)).where(Notification.occurrence_id == occurrence_id))
    return int(current or 0) + 1


def _replace_pending_notification(
    db: Session,
    *,
    occurrence: ReminderOccurrence,
    destination_identity_id: UUID,
    due_at: datetime,
    now: datetime,
) -> None:
    pending = db.scalars(
        select(Notification)
        .where(
            Notification.occurrence_id == occurrence.id,
            Notification.status.in_((PENDING, SENDING)),
        )
        .with_for_update()
    ).all()
    if any(notification.status == SENDING for notification in pending):
        raise RescheduleConflict("A reminder notification is already being sent")
    for notification in pending:
        notification.status = CANCELED
        notification.claimed_by = None
        notification.claimed_until = None
    db.add(
        Notification(
            occurrence_id=occurrence.id,
            destination_identity_id=destination_identity_id,
            sequence=_next_sequence(db, occurrence.id),
            due_at=due_at,
            status=PENDING,
            attempt_count=0,
            created_at=now,
        )
    )


class RescheduleConflict(RuntimeError):
    """The requested reminder cannot safely be changed at this moment."""


def apply_reschedule(
    db: Session,
    *,
    user_id: UUID,
    destination_identity_id: UUID,
    reminder_id: UUID,
    due_at: datetime,
    scope: RescheduleScope | None,
    now: datetime,
    occurrence_id: UUID | None = None,
) -> tuple[Reminder, datetime]:
    """Atomically update one owned reminder and its pending notification(s)."""

    reminder = db.scalar(
        select(Reminder)
        .where(Reminder.id == reminder_id, Reminder.user_id == user_id, Reminder.status == ACTIVE)
        .with_for_update()
    )
    if reminder is None:
        raise RescheduleConflict("Reminder is no longer active")
    destination = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.id == destination_identity_id,
            ChannelIdentity.user_id == user_id,
            ChannelIdentity.channel_type == "TELEGRAM",
            ChannelIdentity.status == "LINKED",
            ChannelIdentity.telegram_chat_id.is_not(None),
        ).with_for_update()
    )
    if destination is None:
        raise RescheduleConflict("The Telegram delivery destination is not available")
    if due_at <= now.astimezone(timezone.utc):
        raise RescheduleConflict("The new reminder time is not in the future")
    if reminder.schedule_type == RECURRING and scope is None:
        raise RescheduleConflict("A recurring reminder requires an explicit scope")

    zone = ZoneInfo(reminder.timezone)
    if reminder.schedule_type == RECURRING and scope is RescheduleScope.SERIES:
        recurrence = parse_recurrence_rule(reminder.recurrence_rule or "")
        local_due = due_at.astimezone(zone)
        # VS-09 changes the series time while retaining its cadence and dates.
        if hasattr(recurrence, "hour"):
            fields = reminder.recurrence_rule.split(";")
            fields = [
                (f"BYHOUR={local_due.hour:02d}" if field.startswith("BYHOUR=") else field)
                for field in fields
            ]
            fields = [
                (f"BYMINUTE={local_due.minute:02d}" if field.startswith("BYMINUTE=") else field)
                for field in fields
            ]
            reminder.recurrence_rule = ";".join(fields)
            start_local = datetime.fromisoformat(reminder.start_local).replace(
                hour=local_due.hour, minute=local_due.minute
            )
            reminder.start_local = start_local.isoformat(timespec="minutes")
        occurrences = db.scalars(
            select(ReminderOccurrence)
            .where(ReminderOccurrence.reminder_id == reminder.id, ReminderOccurrence.status == PENDING)
            .with_for_update()
        ).all()
        if occurrences:
            sending_exists = db.scalar(
                select(func.count())
                .select_from(Notification)
                .where(
                    Notification.occurrence_id.in_([item.id for item in occurrences]),
                    Notification.status == SENDING,
                )
            )
            if sending_exists:
                raise RescheduleConflict("A reminder notification is already being sent")
        for occurrence in occurrences:
            local_date = occurrence.scheduled_at.astimezone(zone).date()
            shifted = resolve_local_datetime(local_date, time(local_due.hour, local_due.minute), zone)
            occurrence.scheduled_at = shifted.astimezone(timezone.utc)
            occurrence.next_due_at = occurrence.scheduled_at
            occurrence.recurrence_instance_key = shifted.replace(tzinfo=None).isoformat(timespec="minutes")
            _replace_pending_notification(
                db,
                occurrence=occurrence,
                destination_identity_id=destination_identity_id,
                due_at=occurrence.next_due_at,
                now=now,
            )
        reminder.version += 1
        reminder.updated_at = now
        return reminder, due_at

    occurrence_query = select(ReminderOccurrence).where(
        ReminderOccurrence.reminder_id == reminder.id,
        ReminderOccurrence.status == PENDING,
    )
    if occurrence_id is not None:
        occurrence_query = occurrence_query.where(ReminderOccurrence.id == occurrence_id)
    else:
        occurrence_query = occurrence_query.order_by(
            ReminderOccurrence.next_due_at, ReminderOccurrence.id
        )
    occurrence = db.scalar(occurrence_query.with_for_update())
    if occurrence is None:
        raise RescheduleConflict("Reminder has no pending occurrence")
    sending_exists = db.scalar(
        select(func.count()).select_from(Notification).where(
            Notification.occurrence_id == occurrence.id,
            Notification.status == SENDING,
        )
    )
    if sending_exists:
        raise RescheduleConflict("A reminder notification is already being sent")
    zone = ZoneInfo(reminder.timezone)
    occurrence.scheduled_at = due_at
    occurrence.next_due_at = due_at
    if reminder.schedule_type == ONE_TIME:
        reminder.start_local = due_at.astimezone(zone).isoformat()
    _replace_pending_notification(
        db,
        occurrence=occurrence,
        destination_identity_id=destination_identity_id,
        due_at=due_at,
        now=now,
    )
    reminder.version += 1
    reminder.updated_at = now
    return reminder, due_at
