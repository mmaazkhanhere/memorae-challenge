from reminder_agent.features.reminders.listing.filters import (
    ReminderSummary,
    format_reminder_summary,
)


def test_pending_summary_uses_singular_wording_for_one_reminder(monkeypatch) -> None:
    monkeypatch.setattr(
        "reminder_agent.features.reminders.listing.filters.random.choice",
        lambda values: values[0],
    )
    response = format_reminder_summary(
        ReminderSummary(metric="PENDING_COUNT", count=1),
        timezone_name="Asia/Karachi",
    )
    assert response == "You've got one reminder coming up."


def test_pending_summary_handles_zero_reminders(monkeypatch) -> None:
    monkeypatch.setattr(
        "reminder_agent.features.reminders.listing.filters.random.choice",
        lambda values: values[0],
    )
    response = format_reminder_summary(
        ReminderSummary(metric="PENDING_COUNT", count=0),
        timezone_name="Asia/Karachi",
    )
    assert response == "You're all caught up! No pending reminders right now."
