from datetime import datetime

from reminder_agent.features.intent import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
)
from reminder_agent.features.reminders.create.presentation import (
    format_reminder_confirmation,
    recurring_cadence,
)


def test_one_time_confirmation_uses_personalized_wording() -> None:
    intent = CreateOneTimeReminderIntent(
        task="call the bank",
        due_at=datetime.fromisoformat("2026-11-04T09:00:00+00:00"),
        timezone_name="UTC",
    )

    result = format_reminder_confirmation(
        intent,
        local_display="on Wednesday, November 4, at 9:00 AM UTC",
        timezone_name="UTC",
    )
    assert "call the bank" in result
    assert "9:00 AM UTC" in result
    assert "Telegram" in result


def test_weekly_confirmation_formats_days_and_interval() -> None:
    intent = CreateRecurringReminderIntent(
        task="weekly review",
        frequency="WEEKLY",
        weekdays=(0, 2),
        interval=2,
        hour=8,
        minute=5,
        start_local=datetime(2026, 11, 2, 8, 5),
        timezone_name="UTC",
    )

    result = format_reminder_confirmation(
        intent,
        local_display="Monday, November 2, at 8:05 AM UTC",
        timezone_name="UTC",
    )
    assert "weekly review" in result
    assert "every two weeks on Monday and Wednesday" in result
    assert "8:05 AM UTC" in result
    assert "Monday, November 2, at 8:05 AM UTC" in result
    assert "Telegram" in result


def test_monthly_and_yearly_cadence_preserve_ordinal_and_date_wording() -> None:
    monthly = CreateRecurringReminderIntent(
        task="pay rent",
        frequency="MONTHLY",
        interval=1,
        day_of_month=11,
        hour=9,
        minute=0,
        start_local=datetime(2026, 11, 11, 9),
        timezone_name="UTC",
    )
    yearly = CreateRecurringReminderIntent(
        task="renew insurance",
        frequency="YEARLY",
        interval=2,
        month=3,
        day_of_month=10,
        hour=9,
        minute=0,
        start_local=datetime(2026, 3, 10, 9),
        timezone_name="UTC",
    )

    assert recurring_cadence(monthly) == "on the 11th of every month"
    assert recurring_cadence(yearly) == "every two years on March 10"
