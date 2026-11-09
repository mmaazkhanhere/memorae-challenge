from datetime import UTC, datetime, timedelta

from reminder_agent.features.intent import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
)
from reminder_agent.features.reminders.reminder_creation_validation import (
    validate_creation_schedules,
)

REFERENCE = datetime(2026, 10, 30, 12, tzinfo=UTC)


def _one_time(*, due_at: datetime, timezone_name: str = "UTC") -> CreateOneTimeReminderIntent:
    return CreateOneTimeReminderIntent(
        task="call the bank",
        due_at=due_at,
        timezone_name=timezone_name,
    )


def test_invalid_typed_intent_precedes_later_schedule_failures() -> None:
    result = validate_creation_schedules(
        [object(), _one_time(due_at=REFERENCE)],
        timezone_name="UTC",
        reference=REFERENCE,
    )

    assert result.schedules == ()
    assert result.failure is not None
    assert result.failure.code == "INVALID_TYPED_INTENT"


def test_timezone_mismatch_precedes_past_schedule_failure() -> None:
    result = validate_creation_schedules(
        [_one_time(due_at=REFERENCE, timezone_name="Asia/Karachi")],
        timezone_name="UTC",
        reference=REFERENCE,
    )

    assert result.failure is not None
    assert result.failure.code == "TIMEZONE_MISMATCH"


def test_validation_classifies_invalid_user_timezone_and_past_schedule() -> None:
    invalid_zone = validate_creation_schedules(
        [_one_time(due_at=REFERENCE + timedelta(minutes=5))],
        timezone_name="Not/AZone",
        reference=REFERENCE,
    )
    past = validate_creation_schedules(
        [_one_time(due_at=REFERENCE)],
        timezone_name="UTC",
        reference=REFERENCE,
    )

    assert invalid_zone.failure is not None
    assert invalid_zone.failure.code == "INVALID_USER_TIMEZONE"
    assert past.failure is not None
    assert past.failure.code == "PAST_OR_PRESENT_DUE_TIME"


def test_invalid_recurring_intent_preserves_error_classification() -> None:
    intent = CreateRecurringReminderIntent.model_construct(
        task="weekly review",
        frequency="WEEKLY",
        interval=1,
        weekdays=(),
        day_of_month=None,
        month=None,
        hour=8,
        minute=0,
        start_local=datetime(2026, 11, 2, 8),
        timezone_name="UTC",
    )

    result = validate_creation_schedules(
        [intent],
        timezone_name="UTC",
        reference=REFERENCE,
    )

    assert result.failure is not None
    assert result.failure.code == "INVALID_RECURRING_INTENT"
    assert result.failure.error_type == "ValueError"
