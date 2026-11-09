"""Characterization coverage for the current rescheduling contract."""

from datetime import UTC, datetime

from reminder_agent.features.reminders.reschedule.decisions import (
    PendingReschedule,
    PendingRescheduleOperation,
    PendingRescheduleProposal,
    RescheduleRequest,
    RescheduleScope,
    apply_pending_reschedule_proposal,
    has_unsupported_series_cadence_change,
    parse_pending_reschedule_parameters,
    parse_reschedule_request,
)

NOW = datetime(2026, 10, 30, 22, 30, tzinfo=UTC)


def test_parse_reschedule_uses_the_user_timezone_for_relative_calendar_words() -> None:
    result = parse_reschedule_request(
        "Move the bank reminder to tomorrow at 10 AM",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.due_at == datetime(2026, 11, 1, 5, 0, tzinfo=UTC)
    assert result.date_explicit is True


def test_parse_reschedule_keeps_ambiguous_clock_non_mutating() -> None:
    result = parse_reschedule_request(
        "Move the bank reminder to 10",
        timezone_name="UTC",
        now=NOW,
    )

    assert result == "Do you mean 10 AM or 10 PM for the new reminder time?"


def test_series_calendar_cadence_change_is_rejected_before_model_fallback() -> None:
    assert has_unsupported_series_cadence_change(
        "Move the whole series to the 15th at 10 AM"
    ) is True
    assert has_unsupported_series_cadence_change(
        "Move the whole series to Monday at 10 AM"
    ) is False


def test_pending_date_follow_up_preserves_original_time_target() -> None:
    result = parse_pending_reschedule_parameters(
        "I meant tomorrow",
        timezone_name="UTC",
        now=NOW,
    )

    assert result is not None
    assert result.operation is PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS
    assert result.destination_date.isoformat() == "2026-10-31"
    assert result.date_explicit is True

    pending = PendingReschedule(
        target_text="call the bank",
        due_at=datetime(2026, 10, 30, 10, 0, tzinfo=UTC),
        timezone_name="UTC",
    )
    updated = apply_pending_reschedule_proposal(
        pending,
        PendingRescheduleProposal(
            operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
            destination_date=result.destination_date,
        ),
        timezone_name="UTC",
    )

    assert updated.due_at == datetime(2026, 10, 31, 10, 0, tzinfo=UTC)
    assert updated.scope is None
    assert updated.date_explicit is True


def test_series_and_occurrence_scope_words_remain_distinct() -> None:
    series = parse_reschedule_request(
        "Move all future exercise reminders to Monday at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )
    occurrence = parse_reschedule_request(
        "Move only tomorrow's exercise reminder to 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(series, RescheduleRequest)
    assert series.scope is RescheduleScope.SERIES
    assert isinstance(occurrence, RescheduleRequest)
    assert occurrence.scope is RescheduleScope.OCCURRENCE
    assert occurrence.occurrence_date.isoformat() == "2026-10-31"
