"""Persistence and rolling materialization for recurring reminders."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from reminder_agent.db.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
    ReminderRecurrenceException,
)
from reminder_agent.observability import get_logger, log_event
from reminder_agent.recurrence import iter_recurrence_occurrences, parse_recurrence_rule


TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
PENDING = "PENDING"
logger = get_logger()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _insert_for_dialect(db: Session):
    return pg_insert if db.get_bind().dialect.name == "postgresql" else sqlite_insert


def materialize_recurring_reminder(
    db: Session,
    reminder: Reminder,
    *,
    now: datetime,
    horizon_days: int = 35,
) -> int:
    """Ensure a recurring reminder has all future instances in the horizon.

    The recurrence key is the intended local wall-clock instant.  The database
    uniqueness constraint makes repeated worker runs and concurrent materializers
    converge on one occurrence and one notification per logical instance.
    """

    if reminder.schedule_type != "RECURRING" or reminder.status != ACTIVE:
        return 0
    recurrence = parse_recurrence_rule(reminder.recurrence_rule or "")
    start_local = datetime.fromisoformat(reminder.start_local)
    now = _utc(now)
    identity = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.user_id == reminder.user_id,
            ChannelIdentity.channel_type == TELEGRAM,
            ChannelIdentity.status == LINKED,
            ChannelIdentity.telegram_chat_id.is_not(None),
        )
    )
    if identity is None:
        return 0

    skipped_dates = set(
        db.scalars(
            select(ReminderRecurrenceException.occurrence_date).where(
                ReminderRecurrenceException.reminder_id == reminder.id
            )
        ).all()
    )

    insert = _insert_for_dialect(db)
    created = 0
    for _local_date, local_due, due_at in iter_recurrence_occurrences(
        recurrence,
        start_local=start_local,
        now=now,
        timezone_name=reminder.timezone,
        horizon_days=horizon_days,
    ):
        if _local_date in skipped_dates:
            continue
        recurrence_key = local_due.replace(tzinfo=None).isoformat(timespec="minutes")
        statement = (
            insert(ReminderOccurrence)
            .values(
                reminder_id=reminder.id,
                scheduled_at=due_at,
                next_due_at=due_at,
                recurrence_instance_key=recurrence_key,
                status=PENDING,
            )
            .on_conflict_do_nothing(
                index_elements=["reminder_id", "recurrence_instance_key"]
            )
            .returning(ReminderOccurrence.id)
        )
        inserted_id = db.scalar(statement)
        if inserted_id is None:
            continue
        occurrence = db.get(ReminderOccurrence, inserted_id)
        if occurrence is None:  # pragma: no cover - defensive database guard
            raise RuntimeError("recurrence occurrence was not returned after insert")
        db.add(
            Notification(
                occurrence_id=occurrence.id,
                destination_identity_id=identity.id,
                sequence=1,
                due_at=due_at,
                status=PENDING,
                attempt_count=0,
            )
        )
        created += 1
    return created


def materialize_active_recurring_reminders(
    session_factory: Callable[[], Session],
    *,
    now: datetime,
    horizon_days: int = 35,
    batch_size: int = 100,
) -> int:
    """Materialize active recurring parents in one short transaction."""

    with session_factory.begin() as db:
        created = 0
        last_id = None
        while True:
            filters = [Reminder.schedule_type == "RECURRING", Reminder.status == ACTIVE]
            if last_id is not None:
                filters.append(Reminder.id > last_id)
            reminders = db.scalars(
                select(Reminder)
                .where(*filters)
                .order_by(Reminder.id)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            ).all()
            if not reminders:
                break
            last_id = reminders[-1].id
            for reminder in reminders:
                try:
                    created += materialize_recurring_reminder(
                        db, reminder, now=now, horizon_days=horizon_days
                    )
                except (TypeError, ValueError) as exc:
                    # A malformed persisted rule must not prevent healthy series
                    # from being materialized on the same worker iteration.
                    log_event(
                        logger,
                        logging.WARNING,
                        "recurring_materialization_skipped",
                        reminder_id=str(reminder.id),
                        reason=type(exc).__name__,
                    )
        return created
