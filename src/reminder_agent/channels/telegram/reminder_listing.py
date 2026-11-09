"""Telegram adaptation for the channel-neutral reminder-list workflow."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from reminder_agent.application.reminder_read_workflows import (
    list_reminders_from_interpretation,
)
from reminder_agent.application.response_staging import (
    ReminderReply,
    _application_request_from_telegram_update,
    _store_response,
)
from reminder_agent.infrastructure.db.models import ChannelIdentity, User


def handle_list_reminder_request(
    db: Session,
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity,
    interpretation: Any,
    now: datetime,
    list_operation: Callable = list_reminders_from_interpretation,
) -> ReminderReply | None:
    """Adapt a typed list result into a Telegram reply.

    The list workflow remains channel-neutral and owns authorization and
    reminder business rules. This adapter only translates the verified
    Telegram update and result into the provider-specific reply shape.
    """

    assert update.message is not None
    result = list_operation(
        db,
        request=_application_request_from_telegram_update(
            update,
            user=user,
            identity=identity,
        ),
        interpretation=interpretation,
        now=now,
    )
    if result is None:
        return None
    return _store_response(
        db,
        update=update,
        user_id=user.id,
        text=result.content,
        response_kind=result.response_kind,
        operation="LIST_REMINDERS",
        response_metadata=(
            {"selection_context": list(result.selection_context)}
            if result.selection_context
            else None
        ),
    )
