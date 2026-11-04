"""Read-only reminder listing for a user's local date range."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.db.models import (
    Reminder,
    ReminderOccurrence,
    ReminderRecurrenceException,
)
from reminder_agent.recurrence import (
    iter_recurrence_occurrences,
    parse_recurrence_rule,
    resolve_local_datetime,
)

ACTIVE = "ACTIVE"
PENDING = "PENDING"
NOTIFIED = "NOTIFIED"
COMPLETED = "COMPLETED"
RECURRING = "RECURRING"
_TASK_STOP_WORDS = {"my", "the", "a", "an", "reminder", "reminders"}


@dataclass(frozen=True)
class ListedReminder:
    reminder_id: UUID
    task: str
    due_at: datetime


@dataclass(frozen=True)
class ReminderSummary:
    metric: str
    count: int
    reminders: tuple[ListedReminder, ...] = ()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _local_bounds(start_date: date, end_date: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    start = resolve_local_datetime(start_date, time.min, zone).astimezone(timezone.utc)
    end = resolve_local_datetime(end_date + timedelta(days=1), time.min, zone).astimezone(
        timezone.utc
    )
    return start, end


def _task_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in value.lower().replace("’", "'").split():
        token = "".join(character for character in raw_token if character.isalnum())
        if token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token and token not in _TASK_STOP_WORDS:
            tokens.add(token)
    return tokens


def _matches_task_query(task: str, task_query: str) -> bool:
    """Require every meaningful query term to occur in the stored task."""

    query_tokens = _task_tokens(task_query)
    return bool(query_tokens) and query_tokens <= _task_tokens(task)


def list_reminders_in_range(
    db: Session,
    *,
    user_id: UUID,
    start_date: date | None,
    end_date: date | None,
    task_query: str | None = None,
    timezone_name: str,
    now: datetime | None = None,
    occurrence_statuses: tuple[str, ...] = (PENDING,),
) -> list[ListedReminder]:
    """Return active, pending reminders without writing any database state.

    Recurring instances are projected for the requested range when the worker
    has not materialized them yet. Existing occurrences (including completed or
    canceled ones) suppress projection of the same logical recurrence instance.
    """

    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be supplied together")
    if end_date is not None and start_date is not None and end_date < start_date:
        raise ValueError("end_date must not precede start_date")
    zone = ZoneInfo(timezone_name)
    all_reminders = start_date is None
    if start_date is None:
        reference = _utc(now or datetime.now(timezone.utc))
        start_date = reference.astimezone(zone).date()
        end_date = start_date + timedelta(days=365)
    assert end_date is not None
    range_start, range_end = _local_bounds(start_date, end_date, zone)

    def in_requested_window(due_at: datetime) -> bool:
        return due_at >= range_start and (all_reminders or due_at < range_end)
    reminders = db.scalars(
        select(Reminder)
        .where(Reminder.user_id == user_id, Reminder.status == ACTIVE)
        .order_by(Reminder.id)
    ).all()
    if task_query is not None:
        reminders = [
            reminder for reminder in reminders if _matches_task_query(reminder.task, task_query)
        ]
    if not reminders:
        return []

    reminder_ids = [reminder.id for reminder in reminders]
    occurrences = db.scalars(
        select(ReminderOccurrence).where(ReminderOccurrence.reminder_id.in_(reminder_ids))
    ).all()
    exceptions = db.scalars(
        select(ReminderRecurrenceException).where(
            ReminderRecurrenceException.reminder_id.in_(reminder_ids),
            ReminderRecurrenceException.occurrence_date >= start_date,
            ReminderRecurrenceException.occurrence_date <= end_date,
        )
    ).all()
    exceptions_by_reminder: dict[UUID, set[date]] = {}
    for exception in exceptions:
        exceptions_by_reminder.setdefault(exception.reminder_id, set()).add(
            exception.occurrence_date
        )
    occurrences_by_reminder: dict[UUID, list[ReminderOccurrence]] = {}
    for occurrence in occurrences:
        occurrences_by_reminder.setdefault(occurrence.reminder_id, []).append(occurrence)

    results: list[ListedReminder] = []
    projected_keys: set[tuple[UUID, str]] = set()
    for reminder in reminders:
        reminder_occurrences = occurrences_by_reminder.get(reminder.id, [])
        if reminder.schedule_type != RECURRING:
            for occurrence in reminder_occurrences:
                due_at = _utc(occurrence.next_due_at)
                if occurrence.status in occurrence_statuses and in_requested_window(due_at):
                    results.append(ListedReminder(reminder.id, reminder.task, due_at))
            continue

        existing_keys: set[str] = set()
        for occurrence in reminder_occurrences:
            local_scheduled = _utc(occurrence.scheduled_at).astimezone(zone).replace(tzinfo=None)
            existing_keys.add(local_scheduled.isoformat(timespec="minutes"))
            due_at = _utc(occurrence.next_due_at)
            if occurrence.status in occurrence_statuses and in_requested_window(due_at):
                results.append(ListedReminder(reminder.id, reminder.task, due_at))

        recurrence = parse_recurrence_rule(reminder.recurrence_rule or "")
        start_local = datetime.fromisoformat(reminder.start_local)
        skipped_dates = exceptions_by_reminder.get(reminder.id, set())
        projection_now = range_start - timedelta(microseconds=1)
        horizon_days = (end_date - start_date).days + 1
        for _local_date, local_due, due_at in iter_recurrence_occurrences(
            recurrence,
            start_local=start_local,
            now=projection_now,
            timezone_name=reminder.timezone,
            horizon_days=horizon_days,
        ):
            due_at = _utc(due_at)
            if not in_requested_window(due_at):
                continue
            if _local_date in skipped_dates:
                continue
            recurrence_key = local_due.replace(tzinfo=None).isoformat(timespec="minutes")
            projected_key = (reminder.id, recurrence_key)
            if recurrence_key in existing_keys or projected_key in projected_keys:
                continue
            projected_keys.add(projected_key)
            results.append(ListedReminder(reminder.id, reminder.task, due_at))

    return sorted(results, key=lambda item: (item.due_at, str(item.reminder_id), item.task))


def summarize_reminders(
    db: Session,
    *,
    user_id: UUID,
    metric: str,
    timezone_name: str,
    now: datetime | None = None,
) -> ReminderSummary:
    """Return owner-scoped progress information without changing database state.

    Outstanding means an active occurrence that is still ``PENDING`` or has
    been delivered as ``NOTIFIED`` but not marked complete. Recurring tasks
    scheduled for today are projected using the same rules as reminder listing.
    """

    reference = _utc(now or datetime.now(timezone.utc))
    zone = ZoneInfo(timezone_name)
    local_today = reference.astimezone(zone).date()
    outstanding_statuses = (PENDING, NOTIFIED)

    if metric == "TODAY_REMAINING":
        reminders = list_reminders_in_range(
            db,
            user_id=user_id,
            start_date=local_today,
            end_date=local_today,
            timezone_name=timezone_name,
            now=reference,
            occurrence_statuses=outstanding_statuses,
        )
        return ReminderSummary(metric=metric, count=len(reminders), reminders=tuple(reminders))

    if metric == "PENDING_COUNT":
        count = db.scalar(
            select(func.count(ReminderOccurrence.id))
            .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
            .where(
                Reminder.user_id == user_id,
                Reminder.status == ACTIVE,
                ReminderOccurrence.status.in_(outstanding_statuses),
            )
        )
        return ReminderSummary(metric=metric, count=int(count or 0))

    if metric == "OVERDUE_COUNT":
        count = db.scalar(
            select(func.count(ReminderOccurrence.id))
            .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
            .where(
                Reminder.user_id == user_id,
                Reminder.status == ACTIVE,
                ReminderOccurrence.status.in_(outstanding_statuses),
                ReminderOccurrence.next_due_at < reference,
            )
        )
        return ReminderSummary(metric=metric, count=int(count or 0))

    filters = [
        Reminder.user_id == user_id,
        ReminderOccurrence.status == COMPLETED,
    ]
    if metric == "TODAY_COMPLETED":
        start, end = _local_bounds(local_today, local_today, zone)
        filters.extend(
            [
                ReminderOccurrence.updated_at >= start,
                ReminderOccurrence.updated_at < end,
            ]
        )
    elif metric != "COMPLETED_COUNT":
        raise ValueError(f"unsupported reminder summary metric: {metric}")
    count = db.scalar(
        select(func.count(ReminderOccurrence.id))
        .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
        .where(*filters)
    )
    return ReminderSummary(metric=metric, count=int(count or 0))


def format_reminder_summary(
    summary: ReminderSummary, *, timezone_name: str
) -> str:
    """Format a concise response for a progress query."""

    count = summary.count
    if summary.metric == "PENDING_COUNT":
        return f"You have {count} pending reminder{'s' if count != 1 else ''}."
    if summary.metric == "OVERDUE_COUNT":
        return f"You have {count} overdue reminder{'s' if count != 1 else ''}."
    if summary.metric == "COMPLETED_COUNT":
        return f"You have completed {count} task{'s' if count != 1 else ''}."
    if summary.metric == "TODAY_COMPLETED":
        return f"You have completed {count} task{'s' if count != 1 else ''} today."
    if not summary.reminders:
        return "You have nothing left to do today."
    zone = ZoneInfo(timezone_name)
    header = f"You still have {count} task{'s' if count != 1 else ''} to do today:"
    lines = [
        f"• {item.due_at.astimezone(zone).strftime('%H:%M %Z')} — {item.task}"
        for item in summary.reminders
    ]
    return "\n".join([header, *lines])


def format_reminder_list(
    reminders: list[ListedReminder], *, start_date: date | None, end_date: date | None,
    timezone_name: str, task_query: str | None = None,
) -> str:
    """Format a compact, user-facing local-time listing."""

    subject = f' matching "{task_query}"' if task_query is not None else ""
    if start_date is None:
        if not reminders:
            return f"You have no upcoming active reminders{subject}."
        heading = f"All upcoming active reminders{subject} ({timezone_name}):"
    elif start_date == end_date:
        period = start_date.isoformat()
        if not reminders:
            return f"You have no active reminders{subject} on {period}."
        heading = f"Reminders{subject} for {period} ({timezone_name}):"
    else:
        period = f"{start_date.isoformat()} to {end_date.isoformat()}"
        if not reminders:
            return f"You have no active reminders{subject} from {period}."
        heading = f"Reminders{subject} from {period} ({timezone_name}):"

    zone = ZoneInfo(timezone_name)
    lines = [heading]
    lines.extend(
        f"• {item.due_at.astimezone(zone).strftime('%Y-%m-%d %H:%M %Z')} — {item.task}"
        for item in reminders
    )
    return "\n".join(lines)
