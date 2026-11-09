"""Read-only reminder listing for a user's local date range."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.features.reminders.recurrence import (
    iter_recurrence_occurrences,
    parse_recurrence_rule,
    resolve_local_datetime,
)
from reminder_agent.infrastructure.db.models import (
    Reminder,
    ReminderOccurrence,
    ReminderRecurrenceException,
)

ACTIVE = "ACTIVE"
PENDING = "PENDING"
NOTIFIED = "NOTIFIED"
COMPLETED = "COMPLETED"
RECURRING = "RECURRING"
_TASK_STOP_WORDS = {"my", "the", "a", "an", "reminder", "reminders"}

REMINDER_LIST_MESSAGES: dict[str, tuple[str, ...]] = {
    "upcoming": (
        "Here's what's coming up! 📅",
        "You've got {count} upcoming {reminder_word}:",
        "Here's what's on your reminder list:",
        "Let's see what's coming up for you:",
    ),
    "specific_date": (
        "Here's what you've got on {date}:",
        "You've got {count} {reminder_word} for {date}:",
        "Here's what's on your schedule for {date}:",
        "Let's see what you've got planned for {date}:",
    ),
    "date_range": (
        "Here's what's on your schedule from {start_date} to {end_date}:",
        "You've got {count} {reminder_word} between {start_date} and {end_date}:",
        "Here's what I've got for you from {start_date} through {end_date}:",
        "Let's see what's coming up between {start_date} and {end_date}:",
    ),
    "task_search": (
        "Here's what I found for \"{query}\":",
        'Found {count} upcoming {reminder_word} for "{query}":',
        "Here are your reminders matching \"{query}\":",
        "I found these reminders related to \"{query}\":",
    ),
    "empty_upcoming": (
        "You're all caught up! No upcoming reminders for now.",
        "Nothing coming up on your reminder list just yet.",
        "Your reminder list is clear for now!",
        "No upcoming reminders at the moment. Anything you'd like me to remember?",
    ),
    "empty_specific_date": (
        "Nothing on your reminder list for {date}.",
        "Looks like you don't have any reminders set for {date}.",
        "Your reminder schedule is clear for {date}!",
        "No reminders for {date}. Want to add something?",
    ),
    "empty_date_range": (
        "Nothing scheduled between {start_date} and {end_date}.",
        "Looks like your reminder list is clear from {start_date} to {end_date}.",
        "No reminders coming up between {start_date} and {end_date}.",
        "You're all clear for {start_date} through {end_date}!",
    ),
    "empty_task_search": (
        'I couldn\'t find any upcoming reminders for "{query}".',
        'Nothing matching "{query}" on your upcoming reminder list.',
        'Looks like you don\'t have any active upcoming reminders related to "{query}".',
    ),
}


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


def list_owned_reminders(
    db: Session,
    *,
    user_id: UUID,
    status: str | None = None,
) -> list[Reminder]:
    """Return reminders owned by ``user_id`` in stable creation order.

    This is the low-level owner-scoped read used by administrative and
    characterization callers. Conversational date-range listing remains in
    :func:`list_reminders_in_range` below.
    """

    filters = [Reminder.user_id == user_id]
    if status is not None:
        filters.append(Reminder.status == status)
    return list(
        db.scalars(
            select(Reminder).where(*filters).order_by(Reminder.created_at.asc(), Reminder.id.asc())
        ).all()
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _local_bounds(start_date: date, end_date: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    start = resolve_local_datetime(start_date, time.min, zone).astimezone(UTC)
    end = resolve_local_datetime(end_date + timedelta(days=1), time.min, zone).astimezone(UTC)
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
    timezone_name: str,
    task_query: str | None = None,
    time_after: time | None = None,
    time_before: time | None = None,
    weekdays: tuple[int, ...] | None = None,
    limit: int | None = None,
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
    if time_after is not None and time_before is not None and time_after > time_before:
        raise ValueError("time_after must not be later than time_before")
    if weekdays is not None and any(day < 0 or day > 6 for day in weekdays):
        raise ValueError("weekdays must be between Monday (0) and Sunday (6)")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    zone = ZoneInfo(timezone_name)
    all_reminders = start_date is None
    if start_date is None:
        reference = _utc(now or datetime.now(UTC))
        start_date = reference.astimezone(zone).date()
        end_date = start_date + timedelta(days=365)
    assert end_date is not None
    range_start, range_end = _local_bounds(start_date, end_date, zone)

    def in_requested_window(due_at: datetime) -> bool:
        if due_at < range_start or (not all_reminders and due_at >= range_end):
            return False
        local_due = due_at.astimezone(zone)
        if weekdays is not None and local_due.weekday() not in weekdays:
            return False
        local_time = local_due.time().replace(tzinfo=None)
        if time_after is not None and local_time < time_after:
            return False
        if time_before is not None and local_time > time_before:
            return False
        return True

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

    ordered = sorted(results, key=lambda item: (item.due_at, str(item.reminder_id), item.task))
    return ordered[:limit] if limit is not None else ordered


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

    reference = _utc(now or datetime.now(UTC))
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


def format_reminder_summary(summary: ReminderSummary, *, timezone_name: str) -> str:
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


def _format_legacy_reminder_list(
    reminders: list[ListedReminder],
    *,
    start_date: date | None,
    end_date: date | None,
    timezone_name: str,
    task_query: str | None = None,
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
        assert end_date is not None
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


def format_reminder_list(
    reminders: list[ListedReminder],
    *,
    start_date: date | None,
    end_date: date | None,
    timezone_name: str,
    task_query: str | None = None,
    now: datetime | None = None,
) -> str:
    """Format a personalized, user-facing local-time listing."""

    count = len(reminders)
    reminder_word = "reminder" if count == 1 else "reminders"
    zone = ZoneInfo(timezone_name)
    reference_date = _utc(now).astimezone(zone).date() if now is not None else None

    def date_label(value: date) -> str:
        if reference_date is None:
            return value.isoformat()
        difference = (value - reference_date).days
        if difference == 0:
            return "today"
        if difference == 1:
            return "tomorrow"
        if difference == -1:
            return "yesterday"
        if 0 <= difference <= 6:
            return value.strftime("%A")
        if value.year != reference_date.year or difference > 6:
            return f"{value:%B} {value.day}, {value.year}"
        return f"{value:%B} {value.day}"

    def choose(category: str) -> str:
        options = REMINDER_LIST_MESSAGES[category]
        seed = count + (start_date.toordinal() if start_date is not None else 0)
        if task_query:
            seed += sum(ord(character) for character in task_query)
        return options[seed % len(options)]

    if not reminders:
        if task_query is not None:
            category, values = "empty_task_search", {"query": task_query}
        elif start_date is None:
            category, values = "empty_upcoming", {}
        elif start_date == end_date:
            category, values = "empty_specific_date", {"date": date_label(start_date)}
        else:
            assert end_date is not None
            category = "empty_date_range"
            values = {"start_date": date_label(start_date), "end_date": date_label(end_date)}
        return choose(category).format(**values)

    if task_query is not None:
        heading = choose("task_search").format(
            count=count, reminder_word=reminder_word, query=task_query
        )
    elif start_date is None:
        heading = choose("upcoming").format(count=count, reminder_word=reminder_word)
    elif start_date == end_date:
        heading = choose("specific_date").format(
            count=count, reminder_word=reminder_word, date=date_label(start_date)
        )
    else:
        assert end_date is not None
        heading = choose("date_range").format(
            count=count,
            reminder_word=reminder_word,
            start_date=date_label(start_date),
            end_date=date_label(end_date),
        )

    local_items = [(item.due_at.astimezone(zone), item.task) for item in reminders]
    if len(local_items) == 1:
        due_at, task = local_items[0]
        return "\n".join([heading, f"- {date_label(due_at.date())} at {due_at:%H:%M %Z} - {task}"])

    grouped: dict[date, list[tuple[datetime, str]]] = {}
    for due_at, task in local_items:
        grouped.setdefault(due_at.date(), []).append((due_at, task))
    lines = [heading]
    for local_date, items in grouped.items():
        lines.append(f"{date_label(local_date)}:")
        lines.extend(f"- {due_at:%H:%M %Z} - {task}" for due_at, task in items)
    return "\n".join(lines)
