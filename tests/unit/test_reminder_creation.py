from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from reminder_agent.features.intent import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
)
from reminder_agent.features.reminders.create.validation import resolve_creation_due_at


def test_one_time_creation_due_at_is_normalized_to_utc() -> None:
    intent = CreateOneTimeReminderIntent(
        task="call the bank",
        due_at=datetime(2026, 11, 4, 9, tzinfo=timezone(timedelta(hours=5))),
        timezone_name="Asia/Karachi",
    )

    assert resolve_creation_due_at(intent, zone=ZoneInfo("Asia/Karachi")) == datetime(
        2026, 11, 4, 4, tzinfo=UTC
    )


def test_recurring_creation_due_at_preserves_existing_dst_gap_policy() -> None:
    intent = CreateRecurringReminderIntent(
        task="weekly review",
        frequency="WEEKLY",
        weekdays=(6,),
        hour=2,
        minute=30,
        start_local=datetime(2027, 3, 14, 2, 30),
        timezone_name="America/New_York",
    )

    assert resolve_creation_due_at(intent, zone=ZoneInfo("America/New_York")) == datetime(
        2027, 3, 14, 7, 30, tzinfo=UTC
    )
