"""Owner-scoped memory application operations."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    SaveMemoryCommand,
)
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.features.memory.create.messages import memory_message
from reminder_agent.features.memory.service import save_memory_fact


def save_memory_for_request(
    db: Session,
    *,
    request: ApplicationRequest,
    command: SaveMemoryCommand,
    now: datetime,
) -> ApplicationResult:
    """Stage one owner-scoped memory fact without owning a transaction."""

    if request.idempotency_key is None:
        return ApplicationResult(
            content=memory_message("memory_idempotency_error"),
            response_kind="IDEMPOTENCY_KEY_REQUIRED",
        )
    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE:
        return ApplicationResult(
            content=memory_message("memory_authorization_failed"),
            response_kind="AUTHORIZATION_FAILURE",
        )

    saved = save_memory_fact(
        db,
        user_id=user.id,
        content=command.content,
        now=now,
    )
    return ApplicationResult(
        content=memory_message("memory_confirmation", fact=command.content),
        response_kind="MEMORY_CONFIRMATION",
        resource_id=saved.fact_id,
    )
