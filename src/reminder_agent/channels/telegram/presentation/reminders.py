"""Telegram rendering helpers for reminder-facing application results."""

from __future__ import annotations

from reminder_agent.application.contracts import ApplicationResult, DurableResponse
from reminder_agent.features.reminders.create.presentation import (
    format_duplicate_reminder,
    format_reminder_confirmation,
    ordinal_suffix,
    recurring_cadence,
)


def durable_cancellation_response(result: ApplicationResult) -> DurableResponse:
    """Render a committed cancellation result for Telegram delivery."""

    return DurableResponse(content=result.content, response_kind=result.response_kind)

__all__ = [
    "format_duplicate_reminder",
    "format_reminder_confirmation",
    "durable_cancellation_response",
    "ordinal_suffix",
    "recurring_cadence",
]
