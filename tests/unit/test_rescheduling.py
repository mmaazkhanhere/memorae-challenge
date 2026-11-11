from datetime import UTC, datetime
from uuid import uuid4

import pytest

from reminder_agent.features.reminders.reschedule.candidates import (
    _candidate_match,
    select_pending_candidate_ids,
)
from reminder_agent.features.reminders.reschedule.decisions import (
    PendingReschedule,
    PendingRescheduleOperation,
    PendingRescheduleProposal,
    RescheduleCandidate,
    RescheduleRequest,
    RescheduleScope,
    _resolve_candidate_due,
    apply_pending_reschedule_proposal,
    parse_reschedule_request,
)


def test_reschedule_matching_normalizes_simple_verb_forms() -> None:
    assert _candidate_match("dentist booking appointment", "book dentist appointment")

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def test_parse_reschedule_request_resolves_explicit_local_time() -> None:
    result = parse_reschedule_request(
        "Move the bank reminder to tomorrow at 10 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.target_text == "bank reminder"
    assert result.due_at == datetime(2026, 10, 31, 10, 0, tzinfo=UTC)
    assert result.date_explicit is True


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("my Call John reminder", "Call John reminder"),
        ("the Call John reminder", "Call John reminder"),
        ("Call John", "Call John"),
        ("my reminder to call John", "call John"),
    ],
)
def test_reschedule_target_normalizes_conversational_wrappers(target: str, expected: str) -> None:
    result = parse_reschedule_request(
        f"Move {target} to tomorrow at 6 PM", timezone_name="UTC", now=NOW
    )

    assert isinstance(result, RescheduleRequest)
    assert result.target_text.lower() == expected.lower()


def test_time_only_reschedule_preserves_selected_candidate_local_date() -> None:
    candidate = RescheduleCandidate(
        reminder_id=uuid4(),
        occurrence_id=uuid4(),
        task="call John",
        due_at=datetime(2026, 10, 31, 9, 0, tzinfo=UTC),
        schedule_type="ONE_TIME",
    )

    request = parse_reschedule_request(
        "Move the Call John reminder to 6 PM", timezone_name="UTC", now=NOW
    )
    assert isinstance(request, RescheduleRequest)
    assert request.date_explicit is False

    resolved = _resolve_candidate_due(request, candidate, timezone_name="UTC")

    assert resolved.due_at == datetime(2026, 10, 31, 18, 0, tzinfo=UTC)


def test_pending_parameter_proposal_preserves_explicit_date_semantics() -> None:
    pending = PendingReschedule(
        target_text="call John",
        due_at=datetime(2026, 10, 31, 18, 0, tzinfo=UTC),
        date_explicit=True,
        candidate_ids=(uuid4(),),
        timezone_name="UTC",
    )

    time_only = apply_pending_reschedule_proposal(
        pending,
        PendingRescheduleProposal(
            operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
            due_at=datetime(2026, 10, 30, 19, 0, tzinfo=UTC),
            date_explicit=False,
        ),
        timezone_name="UTC",
    )
    tomorrow = apply_pending_reschedule_proposal(
        time_only,
        PendingRescheduleProposal(
            operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
            destination_date=datetime(2026, 11, 1, tzinfo=UTC).date(),
        ),
        timezone_name="UTC",
    )

    assert time_only.date_explicit is False
    assert tomorrow.due_at == datetime(2026, 11, 1, 19, 0, tzinfo=UTC)
    assert tomorrow.date_explicit is True


def test_pending_candidate_selection_prioritizes_clock_and_day_before_ordinal(monkeypatch) -> None:
    first_id, second_id, third_id = uuid4(), uuid4(), uuid4()
    candidates = [
        RescheduleCandidate(
            first_id, uuid4(), "call John", datetime(2026, 10, 30, 16, tzinfo=UTC), "ONE_TIME"
        ),
        RescheduleCandidate(
            second_id, uuid4(), "call John", datetime(2026, 10, 31, 9, tzinfo=UTC), "ONE_TIME"
        ),
        RescheduleCandidate(
            third_id, uuid4(), "call John", datetime(2026, 11, 2, 9, tzinfo=UTC), "ONE_TIME"
        ),
    ]

    pending = PendingReschedule(
        target_text="call John",
        due_at=datetime(2026, 10, 30, 18, tzinfo=UTC),
        candidate_ids=(first_id, second_id, third_id),
        timezone_name="UTC",
    )

    monkeypatch.setattr(
        "reminder_agent.features.reminders.reschedule.candidates.find_candidates",
        lambda db, user_id, target_text: candidates,
    )
    assert select_pending_candidate_ids(
        None,
        user_id=uuid4(),
        pending=pending,
        text="the 4 PM one",
        timezone_name="UTC",
        now=NOW,
    ) == (first_id,)
    assert select_pending_candidate_ids(
        None,
        user_id=uuid4(),
        pending=pending,
        text="the tomorrow one",
        timezone_name="UTC",
        now=NOW,
    ) == (second_id,)
    assert select_pending_candidate_ids(
        None,
        user_id=uuid4(),
        pending=pending,
        text="the Friday one",
        timezone_name="UTC",
        now=NOW,
    ) == (first_id,)


def test_parse_reschedule_request_accepts_indirect_shift_wording() -> None:
    result = parse_reschedule_request(
        "Could you shift the bank reminder to tomorrow at 10 AM?",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.target_text == "bank reminder"


def test_parse_recurring_occurrence_selector_is_not_part_of_task_target() -> None:
    result = parse_reschedule_request(
        "Move only tomorrow's exercise reminder to 10 AM", timezone_name="UTC", now=NOW
    )

    assert isinstance(result, RescheduleRequest)
    assert result.target_text == "exercise reminder"
    assert result.scope is RescheduleScope.OCCURRENCE
    assert result.occurrence_date.isoformat() == "2026-10-31"
    assert result.date_explicit is False


@pytest.mark.parametrize(
    ("text", "target", "scope", "occurrence_date"),
    [
        (
            "Move Friday's exercise occurrence to 7 PM",
            "exercise",
            RescheduleScope.OCCURRENCE,
            "2026-11-06",
        ),
        (
            "Move all future exercise reminders to 10 AM",
            "all future exercise reminders",
            RescheduleScope.SERIES,
            None,
        ),
        (
            "Move the next occurrence exercise reminder to 11 AM",
            "exercise reminder",
            RescheduleScope.OCCURRENCE,
            None,
        ),
    ],
)
def test_parse_recurring_scope_and_occurrence_selection(
    text, target, scope, occurrence_date
) -> None:
    result = parse_reschedule_request(text, timezone_name="UTC", now=NOW)

    assert isinstance(result, RescheduleRequest)
    assert result.target_text == target
    assert result.scope is scope
    assert (
        result.occurrence_date.isoformat() if result.occurrence_date else None
    ) == occurrence_date


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
    assert result.destination_weekday == 0


def test_parse_daily_series_target_excludes_scope_and_cadence_words() -> None:
    result = parse_reschedule_request(
        "Move every daily exercise reminder to 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.scope is RescheduleScope.SERIES
    assert result.target_text == "exercise reminder"
    assert result.destination_weekday is None


def test_parse_reschedule_request_extracts_upcoming_occurrence_scope() -> None:
    result = parse_reschedule_request(
        "Move the standup upcoming reminder to Monday at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result, RescheduleRequest)
    assert result.scope is RescheduleScope.OCCURRENCE
    assert result.target_text == "standup"


def test_relative_offset_is_resolved_from_the_selected_candidate_due_time() -> None:
    candidate = RescheduleCandidate(
        reminder_id=uuid4(),
        occurrence_id=uuid4(),
        task="call the bank",
        due_at=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
        schedule_type="ONE_TIME",
    )

    resolved = _resolve_candidate_due(
        RescheduleRequest(target_text="bank", offset_minutes=-120), candidate
    )

    assert resolved.due_at == datetime(2026, 10, 31, 10, 0, tzinfo=UTC)
    assert resolved.offset_minutes is None


def test_pending_reschedule_requires_one_time_target() -> None:
    common = {
        "target_text": "bank",
        "timezone_name": "UTC",
    }

    assert PendingReschedule(**common, offset_minutes=120).offset_minutes == 120
    with pytest.raises(ValueError, match="time target or source request"):
        PendingReschedule(**common)
    with pytest.raises(ValueError, match="both absolute and relative targets"):
        PendingReschedule(
            **common,
            due_at=datetime(2026, 10, 31, 10, 0, tzinfo=UTC),
            offset_minutes=120,
        )
