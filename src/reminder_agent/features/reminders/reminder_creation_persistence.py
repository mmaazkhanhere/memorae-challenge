"""Transaction-scoped reminder graph persistence orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.features.intent.contracts import CreateRecurringReminderIntent
from reminder_agent.features.reminders.create.validation import (
    CreationSchedule,
    recurrence_for_intent,
)
from reminder_agent.features.reminders.recurrence_service import materialize_recurring_reminder
from reminder_agent.infrastructure.db.models import (
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.infrastructure.observability import get_logger, log_event

ACTIVE = "ACTIVE"
ONE_TIME = "ONE_TIME"
RECURRING = "RECURRING"
PENDING = "PENDING"
logger = get_logger()


@dataclass(frozen=True)
class CreationPersistenceResult:
    """Created confirmations plus strong references held through commit."""

    confirmations: tuple[str, ...]
    created_objects: tuple[object, ...]
    duplicate: ExistingReminder | None = None


@dataclass(frozen=True)
class ExistingReminder:
    """The existing reminder that prevented a duplicate creation."""

    task: str
    due_at: datetime


def _existing_reminder(
    db: Session,
    *,
    user_id: UUID,
    task: str,
    due_at: datetime,
) -> ExistingReminder | None:
    """Find an active pending reminder with the same owner, task, and due instant."""

    normalized_task = task.strip().casefold()
    row = db.execute(
        select(Reminder.task, ReminderOccurrence.scheduled_at)
        .join(ReminderOccurrence, ReminderOccurrence.reminder_id == Reminder.id)
        .where(
            Reminder.user_id == user_id,
            Reminder.status == ACTIVE,
            ReminderOccurrence.status == PENDING,
            func.lower(func.trim(Reminder.task)) == normalized_task,
            ReminderOccurrence.scheduled_at == due_at,
        )
        .order_by(Reminder.created_at, Reminder.id)
    ).first()
    if row is None:
        return None
    return ExistingReminder(task=row[0], due_at=row[1])


def persist_creation_graphs(
    db: Session,
    scheduled: Sequence[CreationSchedule],
    *,
    user_id: UUID,
    identity_id: UUID,
    timezone_name: str,
    zone: ZoneInfo,
    reference: datetime,
    source_message_id: str,
    delivery_channel: str,
    confirmation_formatter: Callable[..., str],
) -> CreationPersistenceResult:
    """Persist validated reminder graphs in the caller-owned transaction."""

    # Serialize creation attempts for one owner on PostgreSQL.  The query
    # below prevents ordinary duplicates; the owner lock closes the race where
    # two concurrent requests both query before either inserts.
    db.execute(select(User.id).where(User.id == user_id).with_for_update()).scalar_one()
    seen_keys: set[tuple[str, datetime]] = set()
    for intent, due_at, _local_display in scheduled:
        key = (intent.task.strip().casefold(), due_at)
        if key in seen_keys:
            return CreationPersistenceResult(
                confirmations=(),
                created_objects=(),
                duplicate=ExistingReminder(task=intent.task, due_at=due_at),
            )
        seen_keys.add(key)
        duplicate = _existing_reminder(
            db,
            user_id=user_id,
            task=intent.task,
            due_at=due_at,
        )
        if duplicate is not None:
            return CreationPersistenceResult(
                confirmations=(),
                created_objects=(),
                duplicate=duplicate,
            )

    confirmations: list[str] = []
    created_objects: list[object] = []
    for intent, due_at, local_display in scheduled:
        if isinstance(intent, CreateRecurringReminderIntent):
            recurrence = recurrence_for_intent(intent)
            reminder = Reminder(
                user_id=user_id,
                task=intent.task,
                schedule_type=RECURRING,
                start_local=intent.start_local.isoformat(timespec="minutes"),
                timezone=timezone_name,
                recurrence_rule=recurrence.canonical_rule,
                status=ACTIVE,
                version=1,
            )
            db.add(reminder)
            db.flush()
            created_objects.append(reminder)
            materialize_recurring_reminder(db, reminder, now=reference, horizon_days=35)
            confirmations.append(
                confirmation_formatter(
                    intent,
                    local_display=local_display,
                    timezone_name=timezone_name,
                )
            )
            continue
        reminder = Reminder(
            user_id=user_id,
            task=intent.task,
            schedule_type=ONE_TIME,
            start_local=due_at.astimezone(zone).isoformat(),
            timezone=timezone_name,
            recurrence_rule=None,
            status=ACTIVE,
            version=1,
        )
        db.add(reminder)
        db.flush()
        created_objects.append(reminder)
        occurrence = ReminderOccurrence(
            reminder_id=reminder.id,
            scheduled_at=due_at,
            next_due_at=due_at,
            status=PENDING,
        )
        db.add(occurrence)
        db.flush()
        created_objects.append(occurrence)
        notification = Notification(
            occurrence_id=occurrence.id,
            destination_identity_id=identity_id,
            sequence=1,
            due_at=due_at,
            status=PENDING,
            attempt_count=0,
        )
        db.add(notification)
        created_objects.append(notification)
        log_event(
            logger,
            logging.INFO,
            "reminder_notification_scheduled",
            source_message_id=source_message_id,
            reminder_id=str(reminder.id),
            occurrence_id=str(occurrence.id),
            due_at=due_at.isoformat(),
            delivery_channel=delivery_channel,
        )
        confirmations.append(
            confirmation_formatter(
                intent,
                local_display=local_display,
                timezone_name=timezone_name,
            )
        )
    return CreationPersistenceResult(
        confirmations=tuple(confirmations),
        created_objects=tuple(created_objects),
    )
