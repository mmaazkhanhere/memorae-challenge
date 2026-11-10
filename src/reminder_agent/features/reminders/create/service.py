"""Channel-neutral orchestration for validated reminder creation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from reminder_agent.application.conversations import end_session
from reminder_agent.features.intent.contracts import CreateRecurringReminderIntent
from reminder_agent.features.reminders.create.presentation import (
    format_duplicate_reminder,
    format_next_occurrence,
    format_reminder_confirmation,
    stage_creation_response,
)
from reminder_agent.features.reminders.create.validation import (
    CreationValidationFailure,
    validate_creation_schedules,
)
from reminder_agent.features.reminders.create.reminder_creation_persistence import (
    CreationPersistenceResult,
    ExistingReminder,
    persist_creation_graphs,
)
from reminder_agent.infrastructure.database.models import ChannelIdentity, ConversationSession, User
from reminder_agent.infrastructure.observability import get_logger, log_event

ResponseT = TypeVar("ResponseT")
logger = get_logger()


@dataclass(frozen=True)
class ReminderCreationWorkflowResult[ResponseT]:
    """Outcome of creation work performed inside the caller's transaction."""

    response: ResponseT | None = None
    failure: CreationValidationFailure | None = None
    reminder_count: int = 0
    persistence: CreationPersistenceResult | None = None


@dataclass(frozen=True)
class ReminderCreationHandlingResult[ResponseT]:
    """A creation response staged inside the caller's transaction."""

    response: ResponseT
    reminder_count: int = 0


def _duplicate_response_text(existing: ExistingReminder, *, timezone_name: str) -> str:
    local_due = existing.due_at.astimezone(ZoneInfo(timezone_name))
    return format_duplicate_reminder(
        existing.task,
        when=format_next_occurrence(local_due),
    )


def run_reminder_creation_workflow[ResponseT](
    db: Session,
    *,
    intents: list[object],
    user: User,
    identity: ChannelIdentity,
    locked_session: ConversationSession | None,
    timezone_name: str,
    now: datetime,
    reference: datetime,
    source_message_id: str,
    stage_response: Callable[..., ResponseT],
    confirmation_formatter: Callable[..., str],
) -> ReminderCreationWorkflowResult[ResponseT]:
    """Validate and persist creation work without taking transaction ownership."""

    validation = validate_creation_schedules(
        intents,
        timezone_name=timezone_name,
        reference=reference,
    )
    if validation.failure is not None:
        return ReminderCreationWorkflowResult(failure=validation.failure)

    zone = ZoneInfo(timezone_name)
    if locked_session is not None:
        end_session(locked_session, now=now)
    persistence = persist_creation_graphs(
        db,
        validation.schedules,
        user_id=user.id,
        identity_id=identity.id,
        timezone_name=timezone_name,
        zone=zone,
        reference=reference,
        source_message_id=source_message_id,
        delivery_channel=identity.channel_type,
        confirmation_formatter=confirmation_formatter,
    )
    if persistence.duplicate is not None:
        return ReminderCreationWorkflowResult(persistence=persistence)
    response = stage_creation_response(
        confirmations=persistence.confirmations,
        stage_response=stage_response,
    )
    return ReminderCreationWorkflowResult(
        response=response,
        reminder_count=len(validation.schedules),
        persistence=persistence,
    )


def handle_reminder_creation[ResponseT](
    db: Session,
    *,
    intents: list[object],
    user: User,
    identity: ChannelIdentity,
    locked_session: ConversationSession | None,
    timezone_name: str,
    now: datetime,
    reference: datetime,
    source_message_id: str,
    update_id: int,
    stage_response: Callable[..., ResponseT],
) -> ReminderCreationHandlingResult[ResponseT]:
    """Validate, persist, and stage one reminder-creation outcome.

    The caller owns the surrounding transaction and idempotency record.  This
    operation owns only deterministic creation validation, persistence, and
    the standard creation-specific response mapping.
    """

    creation = run_reminder_creation_workflow(
        db,
        intents=intents,
        user=user,
        identity=identity,
        locked_session=locked_session,
        timezone_name=timezone_name,
        now=now,
        reference=reference,
        source_message_id=source_message_id,
        stage_response=stage_response,
        confirmation_formatter=format_reminder_confirmation,
    )
    failure = creation.failure
    if failure is not None and failure.code == "INVALID_USER_TIMEZONE":
        log_event(
            logger,
            logging.ERROR,
            "reminder_schedule_rejected",
            update_id=update_id,
            reason="invalid_user_timezone",
        )
        return ReminderCreationHandlingResult(
            response=stage_response(
                text="Your saved time zone is invalid; please update it.",
                response_kind="schedule_validation_failure",
            )
        )
    if failure is not None and failure.code == "INVALID_TYPED_INTENT":
        log_event(
            logger,
            logging.WARNING,
            "reminder_intent_rejected",
            update_id=update_id,
            reason="invalid_typed_intent",
        )
        return ReminderCreationHandlingResult(
            response=stage_response(
                text=(
                    "I couldn't understand that recurring reminder. Please provide the task, "
                    "weekday, and time."
                ),
                response_kind="invalid_intent",
            )
        )
    if failure is not None and failure.code == "TIMEZONE_MISMATCH":
        log_event(logger, logging.ERROR, "reminder_timezone_mismatch", update_id=update_id)
        return ReminderCreationHandlingResult(
            response=stage_response(
                text="I couldn't verify the reminder time zone. Please try again.",
                response_kind="timezone_validation_failure",
            )
        )
    if failure is not None and failure.code == "PAST_OR_PRESENT_DUE_TIME":
        log_event(
            logger,
            logging.WARNING,
            "reminder_schedule_rejected",
            update_id=update_id,
            reason="past_or_present_due_time",
        )
        return ReminderCreationHandlingResult(
            response=stage_response(
                text="Please choose a future time for each reminder.",
                response_kind="schedule_validation_failure",
            )
        )
    if failure is not None:
        log_event(
            logger,
            logging.WARNING,
            "reminder_schedule_rejected",
            update_id=update_id,
            reason="invalid_recurring_intent",
            error_type=failure.error_type,
        )
        return ReminderCreationHandlingResult(
            response=stage_response(
                text=(
                    "I couldn't understand that recurring reminder. Please provide the task, "
                    "weekday, and time."
                ),
                response_kind="invalid_intent",
            )
        )

    if creation.persistence is not None and creation.persistence.duplicate is not None:
        existing = creation.persistence.duplicate
        log_event(
            logger,
            logging.INFO,
            "reminder_creation_duplicate",
            update_id=update_id,
            due_at=existing.due_at.isoformat(),
        )
        return ReminderCreationHandlingResult(
            response=stage_response(
                text=_duplicate_response_text(existing, timezone_name=timezone_name),
                response_kind="duplicate_reminder",
            )
        )

    assert creation.response is not None
    log_event(
        logger,
        logging.INFO,
        "reminder_confirmation_staged",
        update_id=update_id,
        reminder_count=creation.reminder_count,
    )
    return ReminderCreationHandlingResult(
        response=creation.response,
        reminder_count=creation.reminder_count,
    )


def log_reminder_creation_committed(
    *, update_id: int, reminder_count: int, intents: Sequence[object]
) -> None:
    """Record the committed, idempotent outcome of reminder creation."""

    log_event(
        logger,
        logging.INFO,
        "reminder_transaction_committed",
        update_id=update_id,
        reminder_count=reminder_count,
    )
    log_event(
        logger,
        logging.INFO,
        "reminder_created",
        update_id=update_id,
        intent_count=len(intents),
        schedule_type=(
            "RECURRING"
            if any(isinstance(intent, CreateRecurringReminderIntent) for intent in intents)
            else "ONE_TIME"
        ),
        notification_status="PENDING",
    )
