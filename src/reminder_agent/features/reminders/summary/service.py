"""Reminder summary orchestration.

The implementation shares the validated read-only workflow with listing;
keeping this entry point makes the summary operation explicit to callers.
"""

from reminder_agent.features.reminders.listing.service import (
    summarize_reminders_from_interpretation,
)

__all__ = ["summarize_reminders_from_interpretation"]
