"""Characterization coverage for deterministic intent parsing.

These tests describe the current contract before the implementation is split
into smaller modules.  They intentionally assert both the interpreted intent
and the fallback/clarification decisions.
"""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from reminder_agent.features.intent import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
    IntentRoute,
    ListRemindersIntent,
    PendingReminder,
    ReminderSummaryIntent,
    RuleBasedIntentInterpreter,
    _next_recurring_start_local,
    _normalise_now,
    extract_listing_task_query,
    resolve_pending_due_at,
)
from reminder_agent.features.intent.deterministic import interpreter as deterministic_interpreter
from reminder_agent.features.intent.deterministic import patterns as deterministic_patterns
from reminder_agent.features.intent.deterministic import support as deterministic_support

NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def test_deterministic_helpers_reexport_their_owned_implementations() -> None:
    assert deterministic_interpreter.extract_listing_task_query is (
        deterministic_patterns.extract_listing_task_query
    )
    assert deterministic_interpreter._normalise_now is deterministic_support.normalise_now
    assert (
        deterministic_interpreter._next_recurring_start_local
        is deterministic_support.next_recurring_start_local
    )


def test_listing_extraction_preserves_task_filter_and_rejects_generic_words() -> None:
    assert extract_listing_task_query("When are my doctor's appointments?") == (
        "doctor's appointments"
    )
    assert extract_listing_task_query("Show me all reminders") == "me all"
    assert extract_listing_task_query("Show all reminders") is None
    assert extract_listing_task_query("show my scheduled reminders") is None


def test_deterministic_listing_precedes_model_fallback() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Show my gym reminders tomorrow",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.task_query == "gym"
    assert result.intent.start_date == date(2026, 10, 31)
    assert result.intent.end_date == result.intent.start_date
    assert result.fallback_allowed is False


def test_simple_relative_request_is_deterministic() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me in five minutes to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, CreateOneTimeReminderIntent)
    assert result.intent.task == "call the bank"
    assert result.intent.due_at == datetime(2026, 10, 30, 12, 5, tzinfo=UTC)
    assert result.fallback_allowed is True


def test_ambiguous_clock_is_a_clarification_without_an_intent() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow at 8 to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is not None
    assert result.clarification.question == "Do you mean 8 AM tomorrow or 8 PM tomorrow?"


def test_weekly_recurrence_keeps_weekday_order_and_local_start() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me every Monday and Wednesday at 8 PM to review the budget",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.weekdays == (0, 2)
    assert result.intent.hour == 20
    assert result.intent.start_local == datetime(2026, 11, 2, 20, 0)


def test_complex_calendar_language_remains_fallback_eligible() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me sometime after lunch to review the budget",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.fallback_allowed is True


def test_pending_due_at_uses_the_pending_timezone() -> None:
    pending = PendingReminder(
        date_local=date(2026, 10, 31),
        hour=9,
        minute=30,
        meridiem="am",
        timezone_name="Asia/Karachi",
    )

    assert resolve_pending_due_at(pending, now=NOW) == datetime(
        2026, 10, 31, 9, 30, tzinfo=ZoneInfo("Asia/Karachi")
    )


def test_naive_now_is_treated_as_utc_and_recurrence_helper_is_local() -> None:
    assert _normalise_now(datetime(2026, 10, 30, 12)) == NOW
    assert _next_recurring_start_local(
        frequency="MONTHLY",
        interval=1,
        weekdays=(),
        day_of_month=15,
        month=None,
        hour=8,
        minute=0,
        timezone_name="UTC",
        now=NOW,
    ) == datetime(2026, 11, 15, 8, 0)


def test_summary_route_is_read_only_and_deterministic() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "How many reminders are overdue?",
        selected_route=IntentRoute.SUMMARY_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, ReminderSummaryIntent)
    assert result.intent.metric == "OVERDUE_COUNT"
    assert result.fallback_allowed is False
