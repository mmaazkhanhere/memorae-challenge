"""Parsing and transactional application of reminder reschedules.

This module deliberately contains no Telegram or model code.  Callers provide
the canonical user identity resolved by the channel adapter; all reminder
ownership and state checks are repeated inside the write transaction.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.features.reminders.recurrence import (
    MonthlyRecurrence,
    WeeklyRecurrence,
    YearlyRecurrence,
    parse_recurrence_rule,
    resolve_local_datetime,
)
from reminder_agent.features.reminders.reschedule.candidates import (
    apply_pending_reschedule_proposal,
    format_candidates,
)
from reminder_agent.features.reminders.reschedule.candidates import (
    find_all_candidates as _find_all_candidates,
)
from reminder_agent.features.reminders.reschedule.candidates import (
    find_candidates as _find_candidates,
)
from reminder_agent.features.reminders.reschedule.candidates import (
    pending_candidates as _pending_candidates,
)
from reminder_agent.features.reminders.reschedule.candidates import (
    resolve_candidate_due as _resolve_candidate_due,
)
from reminder_agent.features.reminders.reschedule.candidates import (
    select_pending_candidate_ids as _select_pending_candidate_ids,
)
from reminder_agent.features.reminders.reschedule.messages import (
    choose_reschedule_message,
    reschedule_message,
)
from reminder_agent.features.reminders.reschedule.models import (
    ACTIVE,
    CANCELED,
    ONE_TIME,
    PENDING,
    RECURRING,
    SENDING,
    PendingReschedule,
    PendingRescheduleOperation,
    PendingRescheduleProposal,
    RescheduleCandidate,
    RescheduleConflict,
    RescheduleDecision,
    RescheduleRequest,
    RescheduleScope,
)
from reminder_agent.features.reminders.reschedule.parser import (
    extract_target_and_occurrence as _extract_target_and_occurrence,
)
from reminder_agent.features.reminders.reschedule.parser import (
    has_unsupported_series_cadence_change,
    is_reschedule_command,
    parse_pending_reschedule_parameters,
    parse_reschedule_request,
)
from reminder_agent.features.reminders.reschedule.parser import (
    parse_date as _parse_date,
)
from reminder_agent.features.reminders.reschedule.parser import (
    scope_from_text as _scope_from_text,
)
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
)

__all__ = [
    "PendingReschedule",
    "PendingRescheduleOperation",
    "PendingRescheduleProposal",
    "RescheduleCandidate",
    "RescheduleConflict",
    "RescheduleDecision",
    "RescheduleRequest",
    "RescheduleScope",
    "_extract_target_and_occurrence",
    "_parse_date",
    "apply_pending_reschedule_proposal",
    "apply_reschedule",
    "decide_reschedule",
    "find_all_candidates",
    "find_candidates",
    "format_candidates",
    "has_unsupported_series_cadence_change",
    "is_reschedule_command",
    "parse_pending_reschedule_parameters",
    "parse_reschedule_request",
    "pending_candidates",
    "select_pending_candidate_ids",
]

def find_candidates(
    db: Session,
    *,
    user_id: UUID,
    target_text: str,
    occurrence_date: date | None = None,
    timezone_name: str = "UTC",
) -> list[RescheduleCandidate]:
    """Compatibility entry point for owned candidate discovery."""

    return _find_candidates(
        db,
        user_id=user_id,
        target_text=target_text,
        occurrence_date=occurrence_date,
        timezone_name=timezone_name,
        reminder_model=Reminder,
        occurrence_model=ReminderOccurrence,
    )


def find_all_candidates(db: Session, *, user_id: UUID) -> list[RescheduleCandidate]:
    return _find_all_candidates(
        db,
        user_id=user_id,
        reminder_model=Reminder,
        occurrence_model=ReminderOccurrence,
    )


def pending_candidates(
    db: Session, *, user_id: UUID, pending: PendingReschedule
) -> list[RescheduleCandidate]:
    """Compatibility entry point that preserves candidate lookup patching."""

    return _pending_candidates(
        db,
        user_id=user_id,
        pending=pending,
        find_candidates_fn=find_candidates,
    )


def select_pending_candidate_ids(
    db: Session,
    *,
    user_id: UUID,
    pending: PendingReschedule,
    text: str,
    timezone_name: str,
    now: datetime,
) -> tuple[UUID, ...]:
    """Compatibility entry point that preserves candidate lookup patching."""

    return _select_pending_candidate_ids(
        db,
        user_id=user_id,
        pending=pending,
        text=text,
        timezone_name=timezone_name,
        now=now,
        find_candidates_fn=find_candidates,
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
    candidate_id_override: UUID | None = None,
) -> RescheduleDecision:
    normalized = text.strip().lower().rstrip(".!?")
    if pending is not None:
        if normalized in {"cancel", "never mind", "nevermind", "not now"}:
            return RescheduleDecision("CANCEL", reschedule_message("reschedule_cancelled"))
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
                choose_reschedule_message("ambiguous_pending_selection") + "\n"
                + format_candidates(
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
                    destination_weekday=pending.destination_weekday,
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
                        destination_weekday=pending.destination_weekday,
                    ),
                    candidate_ids,
                )
            return RescheduleDecision("RESPONSE", reschedule_message("reminder_inactive"))
        if pending.due_at is None and pending.offset_minutes is None:
            return RescheduleDecision(
                "CLARIFY",
                choose_reschedule_message(
                    "missing_reschedule_time", task=candidate.task
                ),
                request=RescheduleRequest(
                    target_text=pending.target_text,
                    due_at=None,
                    date_explicit=pending.date_explicit,
                    scope=scope,
                    source_text=pending.source_text,
                    occurrence_date=pending.occurrence_date,
                    destination_weekday=pending.destination_weekday,
                ),
                candidate_ids=(candidate.reminder_id,),
            )
        if candidate.schedule_type == RECURRING and scope is None:
            return RescheduleDecision(
                "CLARIFY",
                choose_reschedule_message(
                    "recurring_scope_required", task=candidate.task
                ),
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
                    destination_weekday=pending.destination_weekday,
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
    if candidate_id_override is not None:
        candidates = [
            candidate
            for candidate in find_all_candidates(db, user_id=user_id)
            if candidate.reminder_id == candidate_id_override
        ]
        if len(candidates) != 1:
            return RescheduleDecision("RESPONSE", reschedule_message("reminder_inactive"))
        candidate = candidates[0]
        if candidate.schedule_type == RECURRING and parsed.scope is None:
            return RescheduleDecision(
                "CLARIFY",
                choose_reschedule_message("recurring_scope_required", task=candidate.task),
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
    candidates = find_candidates(
        db,
        user_id=user_id,
        target_text=parsed.target_text,
        occurrence_date=parsed.occurrence_date,
        timezone_name=timezone_name,
    )
    if not candidates and re.fullmatch(
        r"(?:it|that|this|the previous|the prior)\s+(?:reminder|one)?",
        parsed.target_text.strip(),
        re.IGNORECASE,
    ):
        all_candidates = find_all_candidates(db, user_id=user_id)
        if len(all_candidates) == 1:
            parsed = replace(parsed, target_text=all_candidates[0].task)
            candidates = all_candidates
    if not candidates:
        return RescheduleDecision("RESPONSE", reschedule_message("no_matching_reminder"))
    if len(candidates) > 1:
        return RescheduleDecision(
            "CLARIFY",
            reschedule_message("multiple_matching_reminders") + "\n"
            + format_candidates(candidates, timezone_name=timezone_name),
            request=parsed,
            candidate_ids=tuple(candidate.reminder_id for candidate in candidates),
        )
    candidate = candidates[0]
    if parsed.due_at is None and parsed.offset_minutes is None:
        return RescheduleDecision(
            "CLARIFY",
            choose_reschedule_message(
                "missing_reschedule_time", task=candidate.task
            ),
            request=parsed,
            candidate_ids=(candidate.reminder_id,),
        )
    if candidate.schedule_type == RECURRING and parsed.scope is None:
        return RescheduleDecision(
            "CLARIFY",
            choose_reschedule_message("recurring_scope_required", task=candidate.task),
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


def _next_sequence(db: Session, occurrence_id: UUID) -> int:
    current = db.scalar(
        select(func.max(Notification.sequence)).where(Notification.occurrence_id == occurrence_id)
    )
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


def apply_reschedule(
    db: Session,
    *,
    user_id: UUID,
    destination_identity_id: UUID,
    destination_channel_type: str = "TELEGRAM",
    destination_external_id: str | None = None,
    reminder_id: UUID,
    due_at: datetime,
    scope: RescheduleScope | None,
    now: datetime,
    occurrence_id: UUID | None = None,
    destination_weekday: int | None = None,
) -> tuple[Reminder, datetime]:
    """Stage owned schedule changes; the caller owns commit/rollback.

    The caller should include this mutation and its idempotency response in one
    transaction. This service never commits or rolls back on its own.
    """

    reminder = db.scalar(
        select(Reminder)
        .where(Reminder.id == reminder_id, Reminder.user_id == user_id, Reminder.status == ACTIVE)
        .with_for_update()
    )
    if reminder is None:
        raise RescheduleConflict("Reminder is no longer active")
    destination_query = select(ChannelIdentity).where(
        ChannelIdentity.id == destination_identity_id,
        ChannelIdentity.user_id == user_id,
        ChannelIdentity.channel_type == destination_channel_type,
        ChannelIdentity.status == "LINKED",
    )
    if destination_channel_type == "TELEGRAM":
        destination_query = destination_query.where(ChannelIdentity.telegram_chat_id.is_not(None))
    elif destination_external_id is not None:
        destination_query = destination_query.where(
            ChannelIdentity.external_id == destination_external_id
        )
    destination = db.scalar(destination_query.with_for_update())
    if destination is None:
        raise RescheduleConflict(
            "The Telegram delivery destination is not available"
            if destination_channel_type == "TELEGRAM"
            else "The reminder delivery destination is not available"
        )
    if due_at <= now.astimezone(UTC):
        raise RescheduleConflict("The new reminder time is not in the future")
    if reminder.schedule_type == RECURRING and scope is None:
        raise RescheduleConflict("A recurring reminder requires an explicit scope")

    zone = ZoneInfo(reminder.timezone)
    if reminder.schedule_type == RECURRING and scope is RescheduleScope.SERIES:
        recurrence = parse_recurrence_rule(reminder.recurrence_rule or "")
        local_due = due_at.astimezone(zone)
        cadence_changed = destination_weekday is not None
        if cadence_changed:
            if not isinstance(recurrence, WeeklyRecurrence) or len(recurrence.weekdays) != 1:
                raise RescheduleConflict(
                    "Changing this recurrence cadence is not supported; change only its time."
                )
            if destination_weekday is None:
                raise RescheduleConflict("A destination weekday is required")
            recurrence = WeeklyRecurrence(
                (destination_weekday,),
                local_due.hour,
                local_due.minute,
                recurrence.interval,
            )
            # The named destination day establishes the new weekly anchor.
            start_local = local_due.replace(tzinfo=None)
        else:
            # Time-only series changes retain the existing cadence and anchor
            # date for weekly, monthly, and yearly rules.
            if isinstance(recurrence, WeeklyRecurrence):
                recurrence = WeeklyRecurrence(
                    recurrence.weekdays,
                    local_due.hour,
                    local_due.minute,
                    recurrence.interval,
                )
            elif isinstance(recurrence, MonthlyRecurrence):
                recurrence = MonthlyRecurrence(
                    recurrence.day_of_month,
                    local_due.hour,
                    local_due.minute,
                    recurrence.interval,
                )
            else:
                recurrence = YearlyRecurrence(
                    recurrence.month,
                    recurrence.day_of_month,
                    local_due.hour,
                    local_due.minute,
                    recurrence.interval,
                )
            start_local = datetime.fromisoformat(reminder.start_local).replace(
                hour=local_due.hour, minute=local_due.minute
            )
        reminder.recurrence_rule = recurrence.canonical_rule
        reminder.start_local = start_local.isoformat(timespec="minutes")
        occurrences = db.scalars(
            select(ReminderOccurrence)
            .where(
                ReminderOccurrence.reminder_id == reminder.id, ReminderOccurrence.status == PENDING
            )
            .order_by(ReminderOccurrence.next_due_at, ReminderOccurrence.id)
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
        for index, occurrence in enumerate(occurrences):
            local_date = (
                local_due.date() + timedelta(weeks=index * recurrence.interval)
                if cadence_changed
                else occurrence.scheduled_at.astimezone(zone).date()
            )
            shifted = resolve_local_datetime(
                local_date, time(local_due.hour, local_due.minute), zone
            )
            occurrence.scheduled_at = shifted.astimezone(UTC)
            occurrence.next_due_at = occurrence.scheduled_at
            occurrence.recurrence_instance_key = shifted.replace(tzinfo=None).isoformat(
                timespec="minutes"
            )
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
    candidate_occurrence = db.scalar(occurrence_query.with_for_update())
    if candidate_occurrence is None:
        raise RescheduleConflict("Reminder has no pending occurrence")
    occurrence = candidate_occurrence
    sending_exists = db.scalar(
        select(func.count())
        .select_from(Notification)
        .where(
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
