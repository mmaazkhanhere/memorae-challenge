"""Read-only reminder listing for a user's local date range."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.db.models import Reminder, ReminderOccurrence
from reminder_agent.recurrence import (
    iter_recurrence_occurrences,
    parse_recurrence_rule,
    resolve_local_datetime,
)

ACTIVE = "ACTIVE"
PENDING = "PENDING"
RECURRING = "RECURRING"
_TASK_STOP_WORDS = {"my", "the", "a", "an", "reminder", "reminders"}


@dataclass(frozen=True)
class ListedReminder:
    reminder_id: UUID
    task: str
    due_at: datetime


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
                if occurrence.status == PENDING and in_requested_window(due_at):
                    results.append(ListedReminder(reminder.id, reminder.task, due_at))
            continue

        existing_keys: set[str] = set()
        for occurrence in reminder_occurrences:
            local_scheduled = _utc(occurrence.scheduled_at).astimezone(zone).replace(tzinfo=None)
            existing_keys.add(local_scheduled.isoformat(timespec="minutes"))
            due_at = _utc(occurrence.next_due_at)
            if occurrence.status == PENDING and in_requested_window(due_at):
                results.append(ListedReminder(reminder.id, reminder.task, due_at))

        recurrence = parse_recurrence_rule(reminder.recurrence_rule or "")
        start_local = datetime.fromisoformat(reminder.start_local)
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
            recurrence_key = local_due.replace(tzinfo=None).isoformat(timespec="minutes")
            projected_key = (reminder.id, recurrence_key)
            if recurrence_key in existing_keys or projected_key in projected_keys:
                continue
            projected_keys.add(projected_key)
            results.append(ListedReminder(reminder.id, reminder.task, due_at))

    return sorted(results, key=lambda item: (item.due_at, str(item.reminder_id), item.task))


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
