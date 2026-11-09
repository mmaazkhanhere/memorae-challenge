from datetime import UTC, date, datetime

from reminder_agent.features.reminders.cancel.decisions import (
    CancellationDecision,
    CancellationRequest,
    CancellationResolution,
    CancellationScope,
    _candidate_match,
    classify_cancellation_resolution,
    ground_model_cancellation_request,
    is_cancel_command,
    parse_cancellation_request,
)

NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def test_cancel_command_requires_a_target():
    assert is_cancel_command("Cancel the standup") is True
    assert is_cancel_command("Please remove the gym reminder") is True
    assert is_cancel_command("I don't need those exercise reminders anymore") is True
    assert is_cancel_command("cancel") is False


def test_unqualified_cancellation_leaves_recurring_scope_for_service_default():
    request = parse_cancellation_request("Cancel the standup", timezone_name="UTC", now=NOW)

    assert not isinstance(request, str)
    assert request.target_text == "the standup"
    assert request.scope is None
    assert request.occurrence_date is None


def test_explicit_occurrence_cancellation_resolves_local_day():
    request = parse_cancellation_request(
        "Cancel only Monday's standup", timezone_name="UTC", now=NOW
    )

    assert not isinstance(request, str)
    assert request.target_text == "standup"
    assert request.scope is CancellationScope.OCCURRENCE
    assert request.occurrence_date is not None
    assert request.occurrence_date.isoformat() == "2026-11-02"


def test_explicit_iso_date_selects_future_occurrence():
    request = parse_cancellation_request(
        "Cancel standup on 2027-01-04", timezone_name="UTC", now=NOW
    )

    assert not isinstance(request, str)
    assert request.target_text == "standup"
    assert request.scope is CancellationScope.OCCURRENCE
    assert request.occurrence_date == date(2027, 1, 4)


def test_model_named_date_is_grounded_as_explicit_occurrence_date():
    request = ground_model_cancellation_request(
        "Skip my finance review on January 15, 2027",
        CancellationRequest("finance review", CancellationScope.OCCURRENCE, date(2027, 1, 15)),
        timezone_name="UTC",
        now=NOW,
    )

    assert request.scope is CancellationScope.OCCURRENCE
    assert request.occurrence_date == date(2027, 1, 15)


def test_target_matching_accepts_simple_singular_plural_difference():
    assert _candidate_match("finance review", "review finances") is True


def test_series_and_next_occurrence_scopes_are_explicit():
    series = parse_cancellation_request("Delete all future standups", timezone_name="UTC", now=NOW)
    occurrence = parse_cancellation_request("Remove the next standup", timezone_name="UTC", now=NOW)

    assert not isinstance(series, str)
    assert series.scope is CancellationScope.SERIES
    assert not isinstance(occurrence, str)
    assert occurrence.scope is CancellationScope.OCCURRENCE


def test_missing_cancellation_target_requests_clarification():
    result = parse_cancellation_request("Cancel", timezone_name="UTC", now=NOW)

    assert isinstance(result, str)
    assert "which reminder" in result.lower()


def test_indirect_cancellation_language_has_a_safe_target_fallback():
    request = parse_cancellation_request(
        "I do not need the bank reminder anymore", timezone_name="UTC", now=NOW
    )

    assert not isinstance(request, str)
    assert request.target_text == "bank reminder"


def test_cancellation_resolution_keeps_uncertainty_distinct_from_no_match():
    assert (
        classify_cancellation_resolution(
            CancellationDecision("MUTATE", ""), CancellationRequest("bank reminder")
        )
        is CancellationResolution.CONFIDENT_MATCH
    )
    assert (
        classify_cancellation_resolution(
            CancellationDecision("CLARIFY", ""), CancellationRequest("bank reminder")
        )
        is CancellationResolution.AMBIGUOUS
    )
    assert (
        classify_cancellation_resolution(
            CancellationDecision("RESPONSE", ""), CancellationRequest("those exercise reminders")
        )
        is CancellationResolution.UNCERTAIN_EXTRACTION
    )
    assert (
        classify_cancellation_resolution(
            CancellationDecision("RESPONSE", ""), CancellationRequest("bank reminder")
        )
        is CancellationResolution.NO_MATCH
    )


def test_model_cannot_replace_an_explicit_occurrence_date():
    request = ground_model_cancellation_request(
        "Cancel only Monday's standup",
        CancellationRequest("standup", CancellationScope.OCCURRENCE, date(2026, 11, 9)),
        timezone_name="UTC",
        now=NOW,
    )

    assert request.scope is CancellationScope.OCCURRENCE
    assert request.occurrence_date == date(2026, 11, 2)


def test_model_occurrence_date_must_match_a_relative_date_in_the_request():
    request = ground_model_cancellation_request(
        "Cancel standup tomorrow",
        CancellationRequest("standup", CancellationScope.SERIES, date(2026, 11, 9)),
        timezone_name="UTC",
        now=NOW,
    )

    assert request.scope is CancellationScope.OCCURRENCE
    assert request.occurrence_date == date(2026, 10, 31)


def test_contextual_next_occurrence_discards_a_model_invented_date():
    request = ground_model_cancellation_request(
        "Skip the next one",
        CancellationRequest("standup", CancellationScope.OCCURRENCE, date(2026, 11, 9)),
        timezone_name="UTC",
        now=NOW,
        recent_messages=("user: Remind me every Monday at 8 AM to standup",),
    )

    assert request.scope is CancellationScope.OCCURRENCE
    assert request.occurrence_date is None


def test_multiple_explicit_dates_discard_the_model_date_for_ambiguous_occurrences():
    request = ground_model_cancellation_request(
        "Cancel standup Monday or Tuesday",
        CancellationRequest("standup", CancellationScope.OCCURRENCE, date(2026, 11, 9)),
        timezone_name="UTC",
        now=NOW,
    )

    assert request.occurrence_date is None
