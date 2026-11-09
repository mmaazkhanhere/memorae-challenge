"""Telegram compatibility handling for read-only reminder progress queries."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from reminder_agent.application.response_staging import (
    ReminderReply,
    _application_request_from_telegram_update,
)
from reminder_agent.features.intent.contracts import Interpretation
from reminder_agent.features.reminders.listing.service import (
    summarize_reminders_from_interpretation,
)
from reminder_agent.infrastructure.db.models import ChannelIdentity, User


def handle_reminder_summary_request(
    db: Session,
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity | None = None,
    interpretation: Interpretation,
    now: datetime,
    summary_operation: Callable = summarize_reminders_from_interpretation,
) -> ReminderReply | None:
    """Return the Telegram reply for a typed progress query, if present.

    The caller invokes this inside its existing transaction context. The
    helper performs no transaction control or idempotency work; the summary
    operation remains an owner-scoped read and the existing formatter remains
    the source of response wording.
    """

    assert update.message is not None
    workflow_kwargs = {
        "db": db,
        "request": (
            _application_request_from_telegram_update(update, user=user, identity=identity)
            if identity is not None
            else None
        ),
        "interpretation": interpretation,
        "now": now,
        "authenticated_user": user,
    }
    if summary_operation is summarize_reminders_from_interpretation:
        summary = summary_operation(**workflow_kwargs)
    else:
        # Preserve the existing test/injection seam while still routing all
        # validation and orchestration through the shared workflow.
        summary = summarize_reminders_from_interpretation(
            **workflow_kwargs,
            summary_operation=summary_operation,
        )
    if summary is None:
        return None
    return ReminderReply(
        update.message.chat.id,
        summary.content,
    )
