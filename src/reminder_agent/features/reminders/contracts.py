"""Shared reminder requests and results.

The channel-neutral command contracts remain defined at the application
boundary for now.  This module is the feature-facing import point, so reminder
operations do not need to know where an adapter obtained its trusted context.
"""

from reminder_agent.application.contracts import (
    CancelReminderCommand,
    CompleteReminderCommand,
    DeliveredReminderContext,
    ListRemindersCommand,
    RescheduleReminderCommand,
    SnoozeReminderCommand,
)

__all__ = [
    "CancelReminderCommand",
    "CompleteReminderCommand",
    "DeliveredReminderContext",
    "ListRemindersCommand",
    "RescheduleReminderCommand",
    "SnoozeReminderCommand",
]
