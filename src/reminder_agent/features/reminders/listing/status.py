from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.features.reminders.recurrence import (
    MonthlyRecurrence,
    WeeklyRecurrence,
    YearlyRecurrence,
    parse_recurrence_rule,
)
from reminder_agent.infrastructure.database.models import Notification, Reminder, ReminderOccurrence


def answer_status_question_for_delivery(
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    provider_message_id: str,
) -> str | None:
    """Answer a non-mutating question about one delivered reminder."""
    row = db.execute(
        select(Reminder, ReminderOccurrence)
        .join(ReminderOccurrence, ReminderOccurrence.reminder_id == Reminder.id)
        .join(Notification, Notification.occurrence_id == ReminderOccurrence.id)
        .where(
            Reminder.user_id == user_id,
            Notification.destination_identity_id == identity_id,
            Notification.provider_message_id == provider_message_id,
            Notification.status == "SENT",
        )
        .order_by(Notification.sent_at.desc())
    ).first()
    if row is None:
        return None

    reminder, _occurrence = row
    if reminder.status != "ACTIVE":
        return "That reminder is no longer active, so I won't send future reminders for it."
    if reminder.schedule_type != "RECURRING" or not reminder.recurrence_rule:
        return "No. That was a one-time reminder, so it won't repeat."

    try:
        recurrence = parse_recurrence_rule(reminder.recurrence_rule)
    except ValueError:
        return f"Yes. Your recurring reminder to {reminder.task} is still active."

    hour = getattr(recurrence, "hour", None)
    minute = getattr(recurrence, "minute", None)
    time_text = "the scheduled time"
    if hour is not None and minute is not None:
        meridiem = "AM" if hour < 12 else "PM"
        time_text = f"{hour % 12 or 12}:{minute:02d} {meridiem}"
    if isinstance(recurrence, WeeklyRecurrence):
        days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        day_text = ", ".join(days[index] for index in recurrence.weekdays)
        return f"Yes. Your weekly {reminder.task} reminder is still active for {day_text} at {time_text}."
    if isinstance(recurrence, MonthlyRecurrence):
        return f"Yes. Your monthly {reminder.task} reminder is still active on day {recurrence.day_of_month} at {time_text}."
    if isinstance(recurrence, YearlyRecurrence):
        return f"Yes. Your yearly {reminder.task} reminder is still active at {time_text}."
    return f"Yes. Your recurring reminder to {reminder.task} is still active."
