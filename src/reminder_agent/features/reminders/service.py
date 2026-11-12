"""Owner-scoped reminder application operations."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    CompleteReminderCommand,
    SnoozeReminderCommand,
)
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.features.reminders.complete.operations import complete_occurrence


def complete_reminder_for_request(
    db: Session,
    *,
    request: ApplicationRequest,
    command: CompleteReminderCommand,
    now: datetime,
) -> ApplicationResult:
    """Stage an owner- and delivery-verified completion without committing."""

    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE or not user.timezone:
        return ApplicationResult(
            content="I couldn't verify your account.",
            response_kind="AUTHORIZATION_FAILURE",
        )
    delivery = command.delivery
    result = complete_occurrence(
        db,
        user_id=user.id,
        identity_id=delivery.identity_id,
        occurrence_id=command.occurrence_id,
        chat_id=(
            int(delivery.external_destination_id)
            if delivery.channel_type == "TELEGRAM" and delivery.external_destination_id is not None
            else None
        ),
        provider_message_id=delivery.provider_message_id,
        channel_type=delivery.channel_type,
        external_destination_id=delivery.external_destination_id,
        now=now,
    )
    return ApplicationResult(
        content=result.text,
        response_kind="completion",
        resource_id=result.occurrence_id,
    )
def snooze_reminder_for_request(
    db: Session,
    *,
    request: ApplicationRequest,
    command: SnoozeReminderCommand,
    now: datetime,
) -> ApplicationResult:
    """Stage an owner- and delivery-verified snooze without committing."""

    user = authorized_user_for_request(db, request=request)
    if user is None or user.onboarding_status != ACTIVE or not user.timezone:
        return ApplicationResult(
            content="I couldn't verify your account.",
            response_kind="AUTHORIZATION_FAILURE",
        )
    from reminder_agent.features.reminders.snooze.operations import snooze_occurrence

    delivery = command.delivery
    result = snooze_occurrence(
        db,
        user_id=user.id,
        identity_id=delivery.identity_id,
        occurrence_id=command.occurrence_id,
        duration_seconds=command.duration_seconds,
        chat_id=(
            int(delivery.external_destination_id)
            if delivery.channel_type == "TELEGRAM" and delivery.external_destination_id is not None
            else None
        ),
        provider_message_id=delivery.provider_message_id,
        channel_type=delivery.channel_type,
        external_destination_id=delivery.external_destination_id,
        delivery_label=None,
        now=now,
    )
    return ApplicationResult(
        content=result.text,
        response_kind="snooze" if result.outcome == "SNOOZED" else "snooze_response",
        resource_id=result.occurrence_id,
    )


