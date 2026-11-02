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

from pydantic import BaseModel, ConfigDict, Field, field_validator
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


class PendingReschedule(BaseModel):
    """Durable, non-authoritative context for a follow-up question."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["RESCHEDULE_REMINDER"] = "RESCHEDULE_REMINDER"
    target_text: str = Field(min_length=1, max_length=200)
    due_at: datetime
    timezone_name: str = Field(min_length=1, max_length=150)
    candidate_ids: tuple[UUID, ...] = ()
    scope: RescheduleScope | None = None

    @field_validator("target_text")
    @classmethod
    def target_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("target_text must not be blank")
        return value

    @field_validator("due_at")
    @classmethod
    def due_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("due_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        ZoneInfo(value)
        return value


@dataclass(frozen=True)
class RescheduleRequest:
    target_text: str
    due_at: datetime
    scope: RescheduleScope | None = None


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


_COMMAND = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:move|reschedule|change|shift|push|postpone|delay)\s+(?:the\s+)?(?P<target>.+?)\s+"
    r"(?P<separator>to|until|for|at)\s+(?P<when>.+?)\s*[.!?]*\s*$",
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
    "future", "all", "please", "one", "first", "second", "third", "1", "2", "3",
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


def parse_reschedule_request(text: str, *, timezone_name: str, now: datetime) -> RescheduleRequest | str:
    """Parse the deliberately small, unambiguous VS-09 command language."""

    match = _COMMAND.match(text)
    if match is None:
        return "Please say which reminder to move and give a new future time."
    target = " ".join(match.group("target").split())
    if not target:
        return "Please tell me which reminder to move."
    scope = _scope_from_text(target)
    target = re.sub(
        r"\b(?:all\s+future\s+reminders?|all\s+reminders?|this\s+occurrence|this\s+reminder|"
        r"the\s+whole\s+series|the\s+series|whole\s+series|series|next\s+occurrence|"
        r"upcoming\s+reminder|only\s+this\s+one|only\s+the\s+next|coming\s+one|occurrence)\b",
        " ",
        target,
        flags=re.I,
    )
    target = " ".join(target.split()).strip(" .,!?")
    if not target:
        return "Please tell me which reminder to move."

    when = _WHEN.match(match.group("when"))
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
    return RescheduleRequest(target, due_at, scope)


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


def find_candidates(db: Session, *, user_id: UUID, target_text: str) -> list[RescheduleCandidate]:
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
    ]
    # A recurring series may have many materialized occurrences; candidate
    # selection is by reminder, with its next pending occurrence as the anchor.
    unique: dict[UUID, RescheduleCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.reminder_id, candidate)
    return list(unique.values())


def format_candidates(candidates: list[RescheduleCandidate]) -> str:
    return "\n".join(
        f"{index}. {candidate.task} (currently {candidate.due_at:%Y-%m-%d %H:%M UTC})"
        for index, candidate in enumerate(candidates, 1)
    )


def decide_reschedule(
    db: Session,
    *,
    user_id: UUID,
    text: str,
    timezone_name: str,
    now: datetime,
    pending: PendingReschedule | None = None,
    request_override: RescheduleRequest | None = None,
) -> RescheduleDecision:
    normalized = text.strip().lower().rstrip(".!?")
    if pending is not None:
        if normalized in {"cancel", "never mind", "nevermind", "not now"}:
            return RescheduleDecision("CANCEL", "Okay, I won't move that reminder.")
        scope = _scope_from_text(text) or pending.scope
        candidate_ids = pending.candidate_ids
        ordinal = re.search(r"\b(?:the\s+)?(first|one|second|two|third|three|1|2|3)\b", normalized)
        if ordinal is not None and len(candidate_ids) > 1:
            selected = _ORDINALS.get(ordinal.group(1))
            if selected is not None and selected < len(candidate_ids):
                candidate_ids = (candidate_ids[selected],)
        if len(candidate_ids) != 1:
            return RescheduleDecision(
                "CLARIFY",
                "Which reminder should I move?\n" + format_candidates(
                    find_candidates(db, user_id=user_id, target_text=pending.target_text)
                ),
                candidate_ids=pending.candidate_ids,
            )
        candidate = db.get(Reminder, candidate_ids[0])
        if candidate is not None and candidate.schedule_type == RECURRING and scope is None:
            return RescheduleDecision(
                "CLARIFY",
                "Should I move only the upcoming reminder, or update all future reminders in the series?",
                candidate_ids=candidate_ids,
            )
        return RescheduleDecision(
            "MUTATE",
            "",
            RescheduleRequest(pending.target_text, pending.due_at, scope),
            candidate_ids,
        )

    parsed = request_override or parse_reschedule_request(
        text, timezone_name=timezone_name, now=now
    )
    if isinstance(parsed, str):
        return RescheduleDecision("RESPONSE", parsed)
    candidates = find_candidates(db, user_id=user_id, target_text=parsed.target_text)
    if not candidates:
        return RescheduleDecision("RESPONSE", "I couldn't find an active reminder matching that description.")
    if len(candidates) > 1:
        return RescheduleDecision(
            "CLARIFY",
            "I found multiple matching reminders. Which one should I move?\n" + format_candidates(candidates),
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
    return RescheduleDecision("MUTATE", "", parsed, (candidate.reminder_id,))


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

    occurrence = db.scalar(
        select(ReminderOccurrence)
        .where(
            ReminderOccurrence.id.in_(
                select(ReminderOccurrence.id).where(
                    ReminderOccurrence.reminder_id == reminder.id,
                    ReminderOccurrence.status == PENDING,
                )
            )
        )
        .order_by(ReminderOccurrence.next_due_at, ReminderOccurrence.id)
        .with_for_update()
    )
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
