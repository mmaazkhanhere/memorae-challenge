from reminder_agent.features.reminders.complete.messages import (
    REMINDER_COMPLETION_MESSAGES,
    reminder_completed_message,
)


def test_completion_message_catalog_contains_each_outcome() -> None:
    assert set(REMINDER_COMPLETION_MESSAGES) == {
        "reminder_completed",
        "reminder_already_completed",
        "reminder_not_available",
        "completed_total",
        "completed_today",
    }


def test_choose_completion_message_uses_requested_message_kind(monkeypatch) -> None:
    monkeypatch.setattr(
        "reminder_agent.features.reminders.complete.messages.random.choice",
        lambda values: values[0],
    )
    assert reminder_completed_message("standup") == (
        "All done! I've marked 'standup' as complete."
    )
