from datetime import datetime, timezone

from reminder_agent.rescheduling import (
    RescheduleScope,
    RescheduleRequest,
    parse_reschedule_request,
)


NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def test_parse_reschedule_request_resolves_explicit_local_time() -> None:
    result = parse_reschedule_request(
        "Move the bank reminder to tomorrow at 10 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.target_text == "bank reminder"
    assert result.due_at == datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)


def test_parse_reschedule_request_accepts_indirect_shift_wording() -> None:
    result = parse_reschedule_request(
        "Could you shift the bank reminder to tomorrow at 10 AM?",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.target_text == "bank reminder"


def test_parse_reschedule_request_requires_am_pm_for_ambiguous_clock() -> None:
    result = parse_reschedule_request(
        "Move the bank reminder to 10",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, str)
    assert "10 AM" in result and "10 PM" in result


def test_parse_reschedule_request_extracts_series_scope() -> None:
    result = parse_reschedule_request(
        "Move the standup series to Monday at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.scope is RescheduleScope.SERIES
    assert result.target_text == "standup"


def test_parse_reschedule_request_extracts_upcoming_occurrence_scope() -> None:
    result = parse_reschedule_request(
        "Move the standup upcoming reminder to Monday at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.scope is RescheduleScope.OCCURRENCE
    assert result.target_text == "standup"
