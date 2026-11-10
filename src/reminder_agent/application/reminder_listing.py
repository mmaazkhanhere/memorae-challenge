"""Application response staging for authenticated reminder-list requests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import ApplicationRequest
from reminder_agent.application.response_staging import ReminderReply
from reminder_agent.features.reminders.listing.service import (
    list_reminders_from_interpretation,
)
from reminder_agent.infrastructure.db.models import User


def stage_list_reminder_response(
    db: Session,
    update: Any,
    *,
    user: User,
    request: ApplicationRequest,
    interpretation: Any,
    now: datetime,
    list_operation: Callable = list_reminders_from_interpretation,
) -> ReminderReply | None:
    """Stage a list result after shared authorization and business validation."""

    assert update.message is not None
    result = list_operation(
        db,
        request=request,
        interpretation=interpretation,
        now=now,
    )
    if result is None:
        return None
    return ReminderReply(
        chat_id=update.message.chat.id,
        text=result.content,
        response_kind=result.response_kind,
        selection_context=tuple(result.selection_context),
    )
