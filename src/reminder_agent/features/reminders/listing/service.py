"""Channel-neutral read-only reminder workflow orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    ListRemindersCommand,
)
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.features.intent.contracts import (
    Interpretation,
    ListRemindersIntent,
    ReminderSummaryIntent,
)
from reminder_agent.features.reminders.listing.filters import (
    ReminderSummary,
    format_reminder_list,
    format_reminder_summary,
    list_reminders_in_range,
    summarize_reminders,
)
from reminder_agent.infrastructure.database.models import User


def list_reminders_for_request(
    db: Session,
    *,
    request: ApplicationRequest,
    command: ListRemindersCommand,
    now: datetime,
) -> ApplicationResult:
    """Return an owner-scoped reminder list without owning a transaction."""

    user = authorized_user_for_request(db, request=request)
    if user is None:
        return ApplicationResult(
            content="I couldn't verify your account.",
            response_kind="AUTHORIZATION_FAILURE",
        )
    if user.onboarding_status != ACTIVE or not user.timezone:
        return ApplicationResult(
            content="Please finish onboarding with your IANA time zone before creating reminders.",
            response_kind="ONBOARDING_INCOMPLETE",
        )
    if command.timezone_name != user.timezone:
        return ApplicationResult(
            content="I couldn't verify your reminder time zone. Please try again.",
            response_kind="TIMEZONE_MISMATCH",
        )

    reminders = list_reminders_in_range(
        db,
        user_id=user.id,
        start_date=command.start_date,
        end_date=command.end_date,
        task_query=command.task_query,
        time_after=command.time_after,
        time_before=command.time_before,
        weekdays=command.weekdays,
        recurrence_only=command.recurrence_only,
        recurrence_series_only=command.recurrence_series_only,
        recurrence_frequency=command.recurrence_frequency,
        limit=command.limit,
        timezone_name=user.timezone,
        now=now,
    )
    return ApplicationResult(
        content=format_reminder_list(
            reminders,
            start_date=command.start_date,
            end_date=command.end_date,
            timezone_name=user.timezone,
            task_query=command.task_query,
            now=now,
        ),
        response_kind="REMINDER_LIST",
        selection_context=tuple(
            {
                "reminder_id": str(item.reminder_id),
                "task": item.task,
                "due_at": item.due_at.isoformat(),
            }
            for item in reminders
        ),
    )


def list_reminders_from_interpretation(
    db: Session,
    *,
    request: ApplicationRequest,
    interpretation: Interpretation,
    now: datetime,
    list_operation: Callable = list_reminders_for_request,
) -> ApplicationResult | None:
    """Resolve and execute one read-only list request for an authenticated user."""

    list_intents = [
        intent
        for intent in interpretation.resolved_intents
        if isinstance(intent, ListRemindersIntent)
    ]
    if not list_intents:
        return None
    if len(list_intents) != 1 or len(interpretation.resolved_intents) != 1:
        return ApplicationResult(
            content="Please ask for one reminder date range at a time.",
            response_kind="REMINDER_LIST_CLARIFICATION",
        )

    list_intent = list_intents[0]
    user = authorized_user_for_request(db, request=request)
    if user is not None and list_intent.timezone_name != user.timezone:
        return ApplicationResult(
            content="I couldn't verify your reminder time zone. Please try again.",
            response_kind="TIMEZONE_MISMATCH",
        )
    return list_operation(
        db,
        request=request,
        command=ListRemindersCommand(
            start_date=list_intent.start_date,
            end_date=list_intent.end_date,
            task_query=list_intent.task_query,
            timezone_name=list_intent.timezone_name,
            time_after=list_intent.time_after,
            time_before=list_intent.time_before,
            weekdays=list_intent.weekdays,
            recurrence_only=list_intent.recurrence_only,
            recurrence_series_only=list_intent.recurrence_series_only,
            recurrence_frequency=list_intent.recurrence_frequency,
            limit=list_intent.limit,
        ),
        now=now,
    )


def summarize_reminders_from_interpretation(
    db: Session,
    *,
    request: ApplicationRequest | None,
    interpretation: Interpretation,
    now: datetime,
    authenticated_user: User | None = None,
    summary_operation: Callable[..., ReminderSummary] = summarize_reminders,
    summary_formatter: Callable[..., str] = format_reminder_summary,
) -> ApplicationResult | None:
    """Resolve and execute one owner-scoped progress query without mutation."""

    summary_intents = [
        intent
        for intent in interpretation.resolved_intents
        if isinstance(intent, ReminderSummaryIntent)
    ]
    if not summary_intents:
        return None
    if len(summary_intents) != 1 or len(interpretation.resolved_intents) != 1:
        return ApplicationResult(
            content="Please ask one reminder progress question at a time.",
            response_kind="REMINDER_SUMMARY_CLARIFICATION",
        )

    summary_intent = summary_intents[0]
    user = (
        authenticated_user
        if authenticated_user is not None
        else authorized_user_for_request(db, request=request)
    )
    if user is not None and summary_intent.timezone_name != user.timezone:
        return ApplicationResult(
            content="I couldn't verify your reminder time zone. Please try again.",
            response_kind="TIMEZONE_MISMATCH",
        )
    if user is None:
        return ApplicationResult(
            content="I couldn't verify your account.",
            response_kind="AUTHORIZATION_FAILURE",
        )
    summary = summary_operation(
        db,
        user_id=user.id,
        metric=summary_intent.metric,
        timezone_name=user.timezone,
        now=now,
    )
    return ApplicationResult(
        content=summary_formatter(summary, timezone_name=user.timezone),
        response_kind="REMINDER_SUMMARY",
    )
