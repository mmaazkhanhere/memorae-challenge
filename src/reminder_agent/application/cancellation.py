"""Channel-neutral bridge to the reminder-cancellation feature."""

from reminder_agent.features.reminders.cancel.decisions import (
    CancellationConflict,
    CancellationRequest,
    PendingCancellation,
    decide_cancellation,
    is_cancel_command,
)
from reminder_agent.features.reminders.cancel.service import (
    cancel_reminder_for_request,
    cancellation_command_for_decision,
    prepare_cancellation,
)

__all__ = [
    "CancellationConflict",
    "CancellationRequest",
    "PendingCancellation",
    "cancel_reminder_for_request",
    "cancellation_command_for_decision",
    "decide_cancellation",
    "is_cancel_command",
    "prepare_cancellation",
]
