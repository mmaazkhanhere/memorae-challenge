"""Shared owner-scoped updates for an already selected reminder."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.infrastructure.database.models import Reminder


class ReminderUpdateConflict(RuntimeError):
    """The selected reminder is no longer an eligible update target."""


def update_reminder_task(
    db: Session,
    *,
    user_id: UUID,
    reminder_id: UUID,
    task: str,
    now: datetime,
) -> Reminder:
    """Update one owned active reminder while holding its row lock."""

    normalized = " ".join(task.split()).strip(" .,!?")
    if not normalized or len(normalized) > 500:
        raise ReminderUpdateConflict("invalid_task")
    reminder = db.scalar(
        select(Reminder)
        .where(
            Reminder.id == reminder_id,
            Reminder.user_id == user_id,
            Reminder.status == "ACTIVE",
        )
        .with_for_update()
    )
    if reminder is None:
        raise ReminderUpdateConflict("reminder_inactive")
    reminder.task = normalized
    reminder.version += 1
    reminder.updated_at = now
    return reminder


__all__ = ["ReminderUpdateConflict", "update_reminder_task"]
