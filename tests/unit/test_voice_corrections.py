from datetime import UTC, datetime

from reminder_agent.features.reminders.reschedule.contextual import parse_contextual_correction
from reminder_agent.features.reminders.reschedule.models import RescheduleRequest

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


def test_parses_relative_active_reminder_correction() -> None:
    result = parse_contextual_correction(
        "Actually, make it half an hour",
        now=NOW,
        timezone_name="UTC",
    )

    assert result is not None
    assert result.reschedule is not None
    assert result.reschedule.due_at == datetime(2026, 10, 31, 12, 30, tzinfo=UTC)
    assert result.reschedule.target_text == "active reminder"


def test_parses_absolute_active_reminder_correction() -> None:
    result = parse_contextual_correction(
        "Actually, make it tomorrow at 2 PM",
        now=NOW,
        timezone_name="UTC",
    )

    assert result is not None
    assert isinstance(result.reschedule, RescheduleRequest)
    assert result.reschedule.due_at == datetime(2026, 11, 1, 14, 0, tzinfo=UTC)


def test_parses_task_correction_without_assigning_an_identity() -> None:
    result = parse_contextual_correction(
        "Actually, change the task to call the dentist",
        now=NOW,
        timezone_name="UTC",
    )

    assert result is not None
    assert result.task == "call the dentist"
    assert result.reschedule is None


def test_does_not_parse_ambiguous_correction() -> None:
    assert (
        parse_contextual_correction(
            "Actually, change it",
            now=NOW,
            timezone_name="UTC",
        )
        is None
    )
