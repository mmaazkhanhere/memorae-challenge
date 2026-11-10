"""Read-only context assembly for conversation coordination."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.features.reminders.complete.operations import resolve_latest_delivered
from reminder_agent.infrastructure.db.models import IdempotencyRecord

TELEGRAM = "TELEGRAM"

_RECENT_SELECTION_CONTEXT: dict[UUID, tuple[dict[str, str], ...]] = {}


def recent_messages(db: Session, *, user_id: UUID) -> list[str]:
    """Build bounded conversation context from durable response records."""

    records = db.scalars(
        select(IdempotencyRecord)
        .where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.user_id == user_id,
        )
        .order_by(IdempotencyRecord.created_at.desc())
        .limit(34)
    ).all()
    messages: list[str] = []
    for record in reversed(records):
        if not isinstance(record.response, dict):
            continue
        request_text = record.response.get("request_text")
        response_text = record.response.get("text")
        if isinstance(request_text, str):
            messages.append(f"user: {request_text}")
        if isinstance(response_text, str):
            messages.append(f"assistant: {response_text}")
    return messages[-34:]


def recent_selection_context(
    db: Session, *, user_id: UUID
) -> tuple[dict[str, str], ...]:
    """Return the latest structured reminder list for ordinal references."""

    cached = _RECENT_SELECTION_CONTEXT.get(user_id)
    if cached:
        return cached
    records = db.scalars(
        select(IdempotencyRecord)
        .where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.user_id == user_id,
        )
        .order_by(IdempotencyRecord.created_at.desc())
        .limit(1)
    ).all()
    for record in records:
        if not isinstance(record.response, dict):
            continue
        raw_context = record.response.get("selection_context")
        if not isinstance(raw_context, list):
            continue
        context = tuple(
            item
            for item in raw_context
            if isinstance(item, dict)
            and isinstance(item.get("reminder_id"), str)
            and isinstance(item.get("task"), str)
        )
        if context:
            return context
    return ()


def remember_selection_context(
    user_id: UUID, selection_context: tuple[dict[str, str], ...]
) -> None:
    """Cache a list result for short-lived ordinal follow-ups."""

    if selection_context:
        _RECENT_SELECTION_CONTEXT[user_id] = selection_context


def recent_delivered_reminders(
    db: Session, *, user_id: UUID, identity_id: UUID, chat_id: int
) -> list[dict[str, str]]:
    """Build bounded, non-authoritative delivery context for routing."""

    candidates = resolve_latest_delivered(
        db, user_id=user_id, identity_id=identity_id, chat_id=chat_id
    )
    return [
        {
            "task": reminder.task,
            "delivered_at": notification.sent_at.isoformat()
            if notification.sent_at is not None
            else "",
        }
        for _occurrence, reminder, notification in candidates
    ]
