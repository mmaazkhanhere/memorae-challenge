"""Parsing, resolving, and transactionally applying reminder cancellations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.features.reminders.cancel.messages import cancel_message
from reminder_agent.features.reminders.recurrence import (
    parse_recurrence_rule,
    recurrence_due_for_local_date,
)
from reminder_agent.infrastructure.database.models import (
    Notification,
    Reminder,
    ReminderOccurrence,
    ReminderRecurrenceException,
)

ACTIVE = "ACTIVE"
PENDING = "PENDING"
SENDING = "SENDING"
CANCELED = "CANCELED"
RECURRING = "RECURRING"
ONE_TIME = "ONE_TIME"


class CancellationScope(StrEnum):
    OCCURRENCE = "OCCURRENCE"
    SERIES = "SERIES"


class CancellationResolution(StrEnum):
    """Deterministic outcome used to decide whether fallback is appropriate."""

    CONFIDENT_MATCH = "CONFIDENT_MATCH"
    UNCERTAIN_EXTRACTION = "UNCERTAIN_EXTRACTION"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"


class PendingCancellation(BaseModel):
    """Durable, non-authoritative context for a cancellation clarification."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["CANCEL_REMINDER"] = "CANCEL_REMINDER"
    target_text: str = Field(min_length=1, max_length=200)
    candidate_ids: tuple[UUID, ...] = ()
    scope: CancellationScope | None = None
    occurrence_date: date | None = None
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("target_text")
    @classmethod
    def target_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("target_text must not be blank")
        return value

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        ZoneInfo(value)
        return value


@dataclass(frozen=True)
class CancellationRequest:
    target_text: str
    scope: CancellationScope | None = None
    occurrence_date: date | None = None

    def __post_init__(self) -> None:
        target = " ".join(self.target_text.strip().split())
        if not target or len(target) > 200:
            raise ValueError("cancellation target must be between 1 and 200 characters")
        object.__setattr__(self, "target_text", target)
        if self.scope is not None and not isinstance(self.scope, CancellationScope):
            object.__setattr__(self, "scope", CancellationScope(self.scope))


@dataclass(frozen=True)
class CancellationCandidate:
    reminder_id: UUID
    occurrence_id: UUID | None
    task: str
    due_at: datetime | None
    schedule_type: str


@dataclass(frozen=True)
class CancellationDecision:
    kind: Literal["MUTATE", "CLARIFY", "RESPONSE", "CANCEL"]
    text: str
    request: CancellationRequest | None = None
    candidate_ids: tuple[UUID, ...] = ()
    occurrence_id: UUID | None = None
    scope: CancellationScope | None = None


class CancellationConflict(RuntimeError):
    """The requested reminder cannot safely be canceled at this moment."""


_COMMAND = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:cancel|delete|remove|stop)\s+(?P<target>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_INDIRECT_COMMAND = re.compile(
    r"^\s*(?:i|we)\s+(?:do\s+not|don't|dont|no\s+longer)\s+"
    r"(?:need|want)\s+(?:the\s+)?(?P<target>.+?)\s*"
    r"(?:anymore|any\s+more|now)?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_INDIRECT_VERB_COMMAND = re.compile(
    r"^\s*(?:i|we)\s+(?:want|need)\s+to\s+"
    r"(?:cancel|delete|remove|stop)\s+(?:the\s+)?(?P<target>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_REMINDER_TARGET_PREFIX = re.compile(
    r"^(?:the\s+)?(?:reminder|reminding)\s+me\s+(?:to\s+)?",
    re.IGNORECASE,
)
_REMINDER_TO_TARGET_PREFIX = re.compile(
    r"^(?:the\s+)?reminder\s+to\s+",
    re.IGNORECASE,
)
_RECURRING_TARGET_SUFFIX = re.compile(
    r"\s+(?:every|each)\s+"
    r"(?:morning|afternoon|evening|night|day|weekday|weekdays|week|month)\s*$",
    re.IGNORECASE,
)
_DATE_PREFIX = re.compile(
    r"^(?P<qualifier>only\s+|just\s+)?(?P<day>today|tomorrow|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday)'?s\s+",
    re.IGNORECASE,
)
_ORDINAL = re.compile(
    r"^(?:the\s+)?(?P<value>first|one|second|two|third|three|1|2|3)"
    r"(?:\s+(?:one|reminder|occurrence))?$",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "my",
    "reminder",
    "reminders",
    "series",
    "future",
    "all",
    "only",
    "just",
    "next",
    "upcoming",
    "this",
    "one",
    "occurrence",
    "recurring",
    "of",
}
_CONTEXTUAL_TARGETS = {"it", "that", "this", "one", "that one", "this one"}
_UNCERTAIN_REFERENCE_WORDS = {
    "it",
    "that",
    "those",
    "these",
    "them",
    "this",
    "one",
    "ones",
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
_RELATIVE_DAY = re.compile(
    r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(?:on\s+)?(?P<value>\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_NAMED_DATE = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_CANCELLATION_CONTEXT_LIMIT = 6
_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
}


def is_cancel_command(text: str) -> bool:
    """Return whether text contains explicit direct or indirect cancellation language."""

    raw_text = text.strip()
    return any(
        pattern.match(raw_text) is not None
        for pattern in (_COMMAND, _INDIRECT_COMMAND, _INDIRECT_VERB_COMMAND)
    )


def is_contextual_cancellation_target(request: CancellationRequest | None) -> bool:
    """Return whether a parsed target is a conversational reference, not a name."""

    return request is not None and request.target_text.lower() in _CONTEXTUAL_TARGETS


def is_semantically_uncertain_cancellation_target(
    request: CancellationRequest | None,
) -> bool:
    """Return whether a descriptive target needs clarification instead of selection."""

    return request is not None and len(_tokens(request.target_text)) > 1


def is_uncertain_cancellation_extraction(request: CancellationRequest | None) -> bool:
    """Return whether wording contains a reference the parser cannot resolve safely."""

    if request is None:
        return True
    words = set(re.findall(r"[a-z0-9]+", request.target_text.lower()))
    return bool(words & _UNCERTAIN_REFERENCE_WORDS)


def classify_cancellation_resolution(
    decision: CancellationDecision | None,
    request: CancellationRequest | None,
    partial_candidates: Sequence[CancellationCandidate] = (),
) -> CancellationResolution:
    """Keep model fallback limited to uncertainty, never a concrete no-match."""

    if decision is not None:
        if decision.kind == "MUTATE":
            return CancellationResolution.CONFIDENT_MATCH
        if decision.kind == "CLARIFY":
            return CancellationResolution.AMBIGUOUS
    if decision is None or partial_candidates or is_uncertain_cancellation_extraction(request):
        return CancellationResolution.UNCERTAIN_EXTRACTION
    return CancellationResolution.NO_MATCH


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in _STOP_WORDS}


def _task_tokens(value: str) -> set[str]:
    """Normalize only simple singular/plural task wording differences."""
    normalized: set[str] = set()
    for token in _tokens(value):
        if token.endswith("ies") and len(token) > 3:
            normalized.add(token[:-3] + "y")
        elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
            normalized.add(token[:-1])
        else:
            normalized.add(token)
    return normalized


def _candidate_match(target: str, task: str) -> bool:
    wanted = _task_tokens(target)
    actual = _task_tokens(task)
    if not wanted or not wanted <= actual:
        return False
    # A single task token can still be a valid unique target (for example,
    # "call reminders" -> "call parents"). Ambiguity is handled by the
    # candidate count in decide_cancellation(), not by rejecting generic words
    # before the database has had a chance to establish uniqueness.
    return True


def _partial_candidate_match(target: str, task: str) -> bool:
    wanted = _tokens(target)
    actual = _tokens(task)
    return len(wanted) > 1 and wanted < actual


def _resolve_date(raw: str, *, now: datetime, timezone_name: str) -> date:
    zone = ZoneInfo(timezone_name)
    local_today = now.astimezone(zone).date()
    normalized = raw.lower()
    if normalized == "today":
        return local_today
    if normalized == "tomorrow":
        return local_today + timedelta(days=1)
    delta = (_DAY_INDEX[normalized] - local_today.weekday()) % 7
    return local_today + timedelta(days=delta or 7)


def _request_occurrence_date(text: str, *, now: datetime, timezone_name: str) -> date | None:
    """Derive one unambiguous relative date explicitly stated in this turn."""

    explicit_dates = {match.group("value") for match in _ISO_DATE.finditer(text)}
    if len(explicit_dates) == 1:
        try:
            return date.fromisoformat(explicit_dates.pop())
        except ValueError:
            return None
    if len(explicit_dates) > 1:
        return None
    named_date = _NAMED_DATE.search(text)
    if named_date is not None:
        try:
            return date(
                int(named_date.group("year")),
                _MONTHS[named_date.group("month").lower()],
                int(named_date.group("day")),
            )
        except ValueError:
            return None
    days = {match.group(1).lower() for match in _RELATIVE_DAY.finditer(text)}
    if len(days) != 1:
        return None
    return _resolve_date(days.pop(), now=now, timezone_name=timezone_name)


def bounded_cancellation_context(recent_messages: Sequence[str]) -> tuple[str, ...]:
    """Return the same bounded trusted history for extraction and grounding."""

    return tuple(recent_messages[-_CANCELLATION_CONTEXT_LIMIT:])


def parse_cancellation_request(
    text: str, *, timezone_name: str, now: datetime
) -> CancellationRequest | str:
    raw_text = text.strip()
    match = _COMMAND.match(raw_text)
    if match is not None:
        target = match.group("target").strip(" .,:;!?")
    else:
        indirect_match = _INDIRECT_COMMAND.match(raw_text) or _INDIRECT_VERB_COMMAND.match(raw_text)
        if indirect_match is None:
            return cancel_message("missing_target")
        target = indirect_match.group("target").strip(" .,:;!?")
    if not target:
        return cancel_message("missing_target")
    scope: CancellationScope | None = None
    occurrence_date: date | None = None

    if re.search(r"\b(?:all\s+future|entire|whole)\s+", target, re.IGNORECASE):
        scope = CancellationScope.SERIES
        target = re.sub(r"\b(?:all\s+future|entire|whole)\s+", "", target, flags=re.IGNORECASE)
    if re.search(r"\b(?:series)\b", target, re.IGNORECASE):
        scope = CancellationScope.SERIES
        target = re.sub(r"\bseries\b", "", target, flags=re.IGNORECASE)

    target = _REMINDER_TARGET_PREFIX.sub("", target)
    target = _REMINDER_TO_TARGET_PREFIX.sub("", target)
    target = _RECURRING_TARGET_SUFFIX.sub("", target)

    date_match = _DATE_PREFIX.match(target)
    if date_match is not None:
        occurrence_date = _resolve_date(
            date_match.group("day"), now=now, timezone_name=timezone_name
        )
        scope = CancellationScope.OCCURRENCE
        target = target[date_match.end() :]
    else:
        explicit_dates = list(_ISO_DATE.finditer(target))
        if len(explicit_dates) == 1:
            try:
                occurrence_date = date.fromisoformat(explicit_dates[0].group("value"))
            except ValueError:
                occurrence_date = None
            else:
                scope = CancellationScope.OCCURRENCE
                target = _ISO_DATE.sub("", target)
        elif re.match(r"^(?:only\s+|just\s+)?(?:the\s+)?(?:next|upcoming)\s+", target, re.I):
            scope = CancellationScope.OCCURRENCE
            target = re.sub(
                r"^(?:only\s+|just\s+)?(?:the\s+)?(?:next|upcoming)\s+",
                "",
                target,
                flags=re.IGNORECASE,
            )

    target = re.sub(r"^(?:only|just)\s+", "", target, flags=re.IGNORECASE)
    target = " ".join(target.split())
    if not target:
        return cancel_message("missing_target")
    return CancellationRequest(target, scope, occurrence_date)


def ground_model_cancellation_request(
    text: str,
    request: CancellationRequest,
    *,
    timezone_name: str,
    now: datetime,
    recent_messages: Sequence[str] = (),
) -> CancellationRequest:
    """Keep model proposals tied to the current turn or trusted recent context.

    The extractor is allowed to resolve conversational references (for example,
    ``cancel that``) from the bounded history supplied by the application.  The
    target is still only a proposal: reminder ownership and active-state checks
    happen later in ``decide_cancellation``.
    """

    trusted_context = bounded_cancellation_context(recent_messages)
    source_tokens = _tokens(" ".join([*trusted_context, text]))
    target_tokens = _tokens(request.target_text)
    if not target_tokens or not target_tokens <= source_tokens:
        raise ValueError("cancellation target was not grounded in user text")

    deterministic = parse_cancellation_request(text, timezone_name=timezone_name, now=now)

    # Deterministic parsing remains authoritative for scope/date language.
    # The model may only propose a different target when deterministic matching
    # later finds no owned active reminder.
    scope = deterministic.scope if not isinstance(deterministic, str) else request.scope
    occurrence_date = deterministic.occurrence_date if not isinstance(deterministic, str) else None
    if occurrence_date is None:
        occurrence_date = _request_occurrence_date(text, now=now, timezone_name=timezone_name)
        if occurrence_date is not None:
            scope = CancellationScope.OCCURRENCE
    if (
        scope is CancellationScope.OCCURRENCE
        and occurrence_date is None
        and not re.search(
            r"\b(?:only|just|next|upcoming|occurrence|today|tomorrow|monday|tuesday|"
            r"wednesday|thursday|friday|saturday|sunday)\b",
            text,
            re.IGNORECASE,
        )
    ):
        scope = None
    return CancellationRequest(request.target_text, scope, occurrence_date)


def _candidate_rows(
    db: Session,
    *,
    user_id: UUID,
    target_text: str | None = None,
    candidate_ids: tuple[UUID, ...] = (),
    occurrence_date: date | None = None,
    timezone_name: str = "UTC",
    include_empty_series: bool = False,
) -> list[CancellationCandidate]:
    rows = db.execute(
        select(Reminder, ReminderOccurrence)
        .join(ReminderOccurrence, ReminderOccurrence.reminder_id == Reminder.id)
        .where(
            Reminder.user_id == user_id,
            Reminder.status == ACTIVE,
            ReminderOccurrence.status == PENDING,
            *([Reminder.id.in_(candidate_ids)] if candidate_ids else []),
        )
        .order_by(ReminderOccurrence.next_due_at, Reminder.id, ReminderOccurrence.id)
    ).all()
    zone = ZoneInfo(timezone_name)
    candidates: list[CancellationCandidate] = []
    seen: set[UUID] = set()
    for reminder, occurrence in rows:
        if target_text is not None and not _candidate_match(target_text, reminder.task):
            continue
        due_at = occurrence.next_due_at
        if due_at.tzinfo is None or due_at.utcoffset() is None:
            due_at = due_at.replace(tzinfo=UTC)
        if occurrence_date is not None and due_at.astimezone(zone).date() != occurrence_date:
            continue
        if reminder.id in seen:
            continue
        seen.add(reminder.id)
        candidates.append(
            CancellationCandidate(
                reminder.id,
                occurrence.id,
                reminder.task,
                due_at,
                reminder.schedule_type,
            )
        )
    # A recurring parent remains cancellable even when every occurrence in the
    # materialization horizon has already been canceled or delivered. There is
    # no occurrence to use for an occurrence-scoped request, so only include
    # these parent-only candidates for series/default resolution.
    if include_empty_series and occurrence_date is None:
        series_query = (
            select(Reminder)
            .where(
                Reminder.user_id == user_id,
                Reminder.status == ACTIVE,
                Reminder.schedule_type == RECURRING,
                *([Reminder.id.in_(candidate_ids)] if candidate_ids else []),
            )
            .order_by(Reminder.created_at, Reminder.id)
        )
        for reminder in db.scalars(series_query).all():
            if reminder.id in seen:
                continue
            if target_text is not None and not _candidate_match(target_text, reminder.task):
                continue
            seen.add(reminder.id)
            candidates.append(
                CancellationCandidate(
                    reminder.id,
                    None,
                    reminder.task,
                    None,
                    reminder.schedule_type,
                )
            )
    return candidates


def find_candidates(
    db: Session,
    *,
    user_id: UUID,
    target_text: str | None,
    occurrence_date: date | None = None,
    timezone_name: str = "UTC",
    include_empty_series: bool = False,
) -> list[CancellationCandidate]:
    return _candidate_rows(
        db,
        user_id=user_id,
        target_text=target_text,
        occurrence_date=occurrence_date,
        timezone_name=timezone_name,
        include_empty_series=include_empty_series,
    )


def find_partial_candidates(
    db: Session,
    *,
    user_id: UUID,
    target_text: str,
    occurrence_date: date | None = None,
    timezone_name: str = "UTC",
    include_empty_series: bool = False,
) -> list[CancellationCandidate]:
    """Return owned active candidates that need semantic confirmation."""

    candidates = _candidate_rows(
        db,
        user_id=user_id,
        occurrence_date=occurrence_date,
        timezone_name=timezone_name,
        include_empty_series=include_empty_series,
    )
    return [
        candidate
        for candidate in candidates
        if _partial_candidate_match(target_text, candidate.task)
    ]


def find_unmaterialized_occurrence_candidates(
    db: Session,
    *,
    user_id: UUID,
    target_text: str | None,
    occurrence_date: date,
    timezone_name: str,
    now: datetime,
) -> list[CancellationCandidate]:
    """Find valid future recurrence dates that have not entered the horizon yet."""

    candidates: list[CancellationCandidate] = []
    reminders = db.scalars(
        select(Reminder)
        .where(
            Reminder.user_id == user_id,
            Reminder.status == ACTIVE,
            Reminder.schedule_type == RECURRING,
        )
        .order_by(Reminder.created_at, Reminder.id)
    ).all()
    for reminder in reminders:
        if target_text is not None and not _candidate_match(target_text, reminder.task):
            continue
        try:
            recurrence = parse_recurrence_rule(reminder.recurrence_rule or "")
            due = recurrence_due_for_local_date(
                recurrence,
                start_local=datetime.fromisoformat(reminder.start_local),
                occurrence_date=occurrence_date,
                timezone_name=reminder.timezone,
            )
        except ValueError:
            continue
        if due is None or due[1] <= now.astimezone(UTC):
            continue
        local_due, due_at = due
        recurrence_key = local_due.replace(tzinfo=None).isoformat(timespec="minutes")
        materialized = db.scalar(
            select(ReminderOccurrence.id).where(
                ReminderOccurrence.reminder_id == reminder.id,
                ReminderOccurrence.recurrence_instance_key == recurrence_key,
            )
        )
        skipped = db.scalar(
            select(ReminderRecurrenceException.id).where(
                ReminderRecurrenceException.reminder_id == reminder.id,
                ReminderRecurrenceException.occurrence_date == occurrence_date,
            )
        )
        if materialized is not None or skipped is not None:
            continue
        candidates.append(
            CancellationCandidate(reminder.id, None, reminder.task, due_at, reminder.schedule_type)
        )
    return candidates


def format_candidates(candidates: list[CancellationCandidate], *, timezone_name: str) -> str:
    zone = ZoneInfo(timezone_name)
    formatted: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        if candidate.due_at is None:
            schedule = "recurring series"
        else:
            schedule = f"{candidate.due_at.astimezone(zone):%Y-%m-%d %H:%M %Z}"
        formatted.append(f"{index}. {candidate.task} (scheduled {schedule})")
    return "\n".join(formatted)


def decide_cancellation(
    db: Session,
    *,
    user_id: UUID,
    text: str,
    timezone_name: str,
    now: datetime,
    pending: PendingCancellation | None = None,
    request_override: CancellationRequest | None = None,
    candidate_id_override: UUID | None = None,
) -> CancellationDecision:
    normalized = text.strip().lower().rstrip(".!?")
    if pending is not None:
        if normalized in {"cancel", "never mind", "nevermind", "not now"}:
            return CancellationDecision(
                "CANCEL", cancel_message("cancellation_abandoned")
            )
        ids = list(pending.candidate_ids)
        ordinal = _ORDINAL.match(normalized)
        if candidate_id_override is not None and candidate_id_override in ids:
            ids = [candidate_id_override]
        elif ordinal is not None and len(ids) > 1:
            selected = {
                "first": 0,
                "one": 0,
                "1": 0,
                "second": 1,
                "two": 1,
                "2": 1,
                "third": 2,
                "three": 2,
                "3": 2,
            }[ordinal.group("value")]
            if selected < len(ids):
                ids = [ids[selected]]
        if len(ids) != 1:
            candidates = _candidate_rows(
                db,
                user_id=user_id,
                candidate_ids=pending.candidate_ids,
                occurrence_date=pending.occurrence_date,
                timezone_name=timezone_name,
                include_empty_series=(pending.scope is not CancellationScope.OCCURRENCE),
            )
            return CancellationDecision(
                "CLARIFY",
                cancel_message("pending_selection_ambiguous")
                + "\n"
                + format_candidates(candidates, timezone_name=timezone_name),
                candidate_ids=pending.candidate_ids,
                request=CancellationRequest(
                    pending.target_text, pending.scope, pending.occurrence_date
                ),
                scope=pending.scope,
            )
        candidates = _candidate_rows(
            db,
            user_id=user_id,
            candidate_ids=(ids[0],),
            occurrence_date=pending.occurrence_date,
            timezone_name=timezone_name,
            include_empty_series=(pending.scope is not CancellationScope.OCCURRENCE),
        )
        if len(candidates) != 1:
            return CancellationDecision(
                "RESPONSE", cancel_message("reminder_inactive")
            )
        candidate = candidates[0]
        scope = pending.scope or (
            CancellationScope.SERIES
            if candidate.schedule_type == RECURRING
            else CancellationScope.OCCURRENCE
        )
        return CancellationDecision(
            "MUTATE",
            "",
            request=CancellationRequest(pending.target_text, scope, pending.occurrence_date),
            candidate_ids=(candidate.reminder_id,),
            occurrence_id=candidate.occurrence_id,
            scope=scope,
        )

    parsed = request_override or parse_cancellation_request(
        text, timezone_name=timezone_name, now=now
    )
    if isinstance(parsed, str):
        return CancellationDecision("RESPONSE", parsed)
    candidates = find_candidates(
        db,
        user_id=user_id,
        target_text=parsed.target_text,
        occurrence_date=parsed.occurrence_date,
        timezone_name=timezone_name,
        include_empty_series=(parsed.scope is not CancellationScope.OCCURRENCE),
    )
    if (
        not candidates
        and parsed.scope is CancellationScope.OCCURRENCE
        and parsed.occurrence_date is not None
    ):
        candidates = find_unmaterialized_occurrence_candidates(
            db,
            user_id=user_id,
            target_text=parsed.target_text,
            occurrence_date=parsed.occurrence_date,
            timezone_name=timezone_name,
            now=now,
        )
    if not candidates:
        return CancellationDecision(
            "RESPONSE", cancel_message("no_match")
        )
    if len(candidates) > 1:
        return CancellationDecision(
            "CLARIFY",
            cancel_message("multiple_matches")
            + "\n"
            + format_candidates(candidates, timezone_name=timezone_name),
            request=parsed,
            candidate_ids=tuple(candidate.reminder_id for candidate in candidates),
            scope=parsed.scope,
        )
    candidate = candidates[0]
    scope = parsed.scope or (
        CancellationScope.SERIES
        if candidate.schedule_type == RECURRING
        else CancellationScope.OCCURRENCE
    )
    return CancellationDecision(
        "MUTATE",
        "",
        request=CancellationRequest(parsed.target_text, scope, parsed.occurrence_date),
        candidate_ids=(candidate.reminder_id,),
        occurrence_id=candidate.occurrence_id,
        scope=scope,
    )


def apply_cancellation(
    db: Session,
    *,
    user_id: UUID,
    reminder_id: UUID,
    occurrence_id: UUID | None,
    scope: CancellationScope,
    now: datetime,
    occurrence_date: date | None = None,
) -> tuple[Reminder, int]:
    """Stage owned cancellation changes; the caller owns commit/rollback.

    The caller should include this mutation and its idempotency response in one
    transaction. This service never commits or rolls back on its own.
    """

    reminder = db.scalar(
        select(Reminder)
        .where(Reminder.id == reminder_id, Reminder.user_id == user_id, Reminder.status == ACTIVE)
        .with_for_update()
    )
    if reminder is None:
        raise CancellationConflict("Reminder is no longer active")

    if scope is CancellationScope.SERIES:
        occurrences = db.scalars(
            select(ReminderOccurrence)
            .where(
                ReminderOccurrence.reminder_id == reminder.id, ReminderOccurrence.status == PENDING
            )
            .with_for_update()
        ).all()
    elif (
        occurrence_id is None
        and occurrence_date is not None
        and reminder.schedule_type == RECURRING
    ):
        try:
            recurrence = parse_recurrence_rule(reminder.recurrence_rule or "")
            due = recurrence_due_for_local_date(
                recurrence,
                start_local=datetime.fromisoformat(reminder.start_local),
                occurrence_date=occurrence_date,
                timezone_name=reminder.timezone,
            )
        except ValueError as exc:
            raise CancellationConflict(
                "Requested date is not a valid recurrence occurrence"
            ) from exc
        if due is None or due[1] <= now.astimezone(UTC):
            raise CancellationConflict("Requested date is not a future recurrence occurrence")
        recurrence_key = due[0].replace(tzinfo=None).isoformat(timespec="minutes")
        occurrence = db.scalar(
            select(ReminderOccurrence)
            .where(
                ReminderOccurrence.reminder_id == reminder.id,
                ReminderOccurrence.recurrence_instance_key == recurrence_key,
            )
            .with_for_update()
        )
        if occurrence is None:
            existing_exception = db.scalar(
                select(ReminderRecurrenceException)
                .where(
                    ReminderRecurrenceException.reminder_id == reminder.id,
                    ReminderRecurrenceException.occurrence_date == occurrence_date,
                )
                .with_for_update()
            )
            if existing_exception is not None:
                raise CancellationConflict("That recurrence occurrence is already canceled")
            db.add(
                ReminderRecurrenceException(
                    reminder_id=reminder.id,
                    occurrence_date=occurrence_date,
                )
            )
            reminder.version += 1
            reminder.updated_at = now.astimezone(UTC)
            return reminder, 0
        occurrences = [occurrence] if occurrence.status == PENDING else []
    else:
        query = select(ReminderOccurrence).where(
            ReminderOccurrence.reminder_id == reminder.id,
            ReminderOccurrence.status == PENDING,
        )
        if occurrence_id is not None:
            query = query.where(ReminderOccurrence.id == occurrence_id)
        query = query.order_by(
            ReminderOccurrence.next_due_at, ReminderOccurrence.id
        ).with_for_update()
        occurrence = db.scalar(query)
        occurrences = [occurrence] if occurrence is not None else []

    if not occurrences and not (
        scope is CancellationScope.SERIES and reminder.schedule_type == RECURRING
    ):
        raise CancellationConflict("Reminder has no pending occurrence to cancel")
    occurrence_ids = [occurrence.id for occurrence in occurrences]
    notifications = (
        db.scalars(
            select(Notification)
            .where(
                Notification.occurrence_id.in_(occurrence_ids),
                Notification.status.in_((PENDING, SENDING)),
            )
            .with_for_update()
        ).all()
        if occurrence_ids
        else []
    )
    if any(notification.status == SENDING for notification in notifications):
        raise CancellationConflict("A reminder notification is already being sent")

    for occurrence in occurrences:
        occurrence.status = CANCELED
        occurrence.updated_at = now.astimezone(UTC)
    for notification in notifications:
        notification.status = CANCELED
        notification.claimed_by = None
        notification.claimed_until = None

    if scope is CancellationScope.SERIES or reminder.schedule_type == ONE_TIME:
        reminder.status = CANCELED
    reminder.version += 1
    reminder.updated_at = now.astimezone(UTC)
    return reminder, len(occurrences)
