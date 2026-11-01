from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from reminder_agent.intent import CreateRecurringReminderIntent, RuleBasedIntentInterpreter
from reminder_agent.recurrence import (
    WeeklyRecurrence,
    parse_weekly_rule,
    resolve_local_datetime,
)


def test_weekly_phrase_is_parsed_into_a_typed_local_schedule() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me every Monday at 8 AM to call the bank.",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.weekdays == (0,)
    assert result.intent.hour == 8
    assert result.intent.minute == 0
    assert result.intent.start_local == datetime(2026, 11, 2, 8)


def test_weekly_phrase_does_not_require_the_remind_me_prefix() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Every Monday at 8 AM to call the bank.",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )
    assert isinstance(result.intent, CreateRecurringReminderIntent)


def test_weekly_phrase_with_an_ambiguous_hour_asks_for_am_or_pm() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me every Sunday at 8 to review the weekly plan.",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )

    assert result.intent is None
    assert result.clarification is not None
    assert result.clarification.question == "Do you mean 8 AM or 8 PM every week?"
    assert result.clarification.pending.recurrence_weekdays == (6,)


def test_weekly_ambiguous_hour_follow_up_creates_the_series() -> None:
    interpreter = RuleBasedIntentInterpreter()
    first = interpreter.interpret(
        "Remind me every Sunday at 8 to review the weekly plan.",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )
    assert first.clarification is not None

    result = interpreter.interpret(
        "PM",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
        context=first.clarification.pending,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.weekdays == (6,)
    assert result.intent.hour == 20


def test_weekly_rule_round_trips_canonically() -> None:
    recurrence = WeeklyRecurrence((0, 2), 8, 5)
    assert recurrence.canonical_rule == "FREQ=WEEKLY;BYDAY=MO,WE;BYHOUR=08;BYMINUTE=05"
    assert parse_weekly_rule(recurrence.canonical_rule) == recurrence


def test_recurring_intent_rejects_out_of_range_weekdays() -> None:
    try:
        CreateRecurringReminderIntent(
            task="check report",
            weekdays=(7,),
            hour=8,
            minute=0,
            start_local=datetime(2026, 11, 2, 8),
            timezone_name="UTC",
        )
    except ValueError as exc:
        assert "weekday" in str(exc).lower()
    else:  # pragma: no cover - assertion-style guard
        raise AssertionError("invalid weekday was accepted")


def test_recurring_intent_rejects_start_on_unselected_weekday() -> None:
    try:
        CreateRecurringReminderIntent(
            task="check report",
            weekdays=(0,),
            hour=8,
            minute=0,
            start_local=datetime(2026, 11, 3, 8),  # Tuesday
            timezone_name="UTC",
        )
    except ValueError as exc:
        assert "selected weekday" in str(exc)
    else:  # pragma: no cover - assertion-style guard
        raise AssertionError("weekday mismatch was accepted")


def test_dst_gap_shifts_forward_by_the_gap() -> None:
    resolved = resolve_local_datetime(
        date(2027, 3, 14), time(2, 30), ZoneInfo("America/New_York")
    )
    assert resolved.isoformat() == "2027-03-14T03:30:00-04:00"
    assert resolved.astimezone(timezone.utc).isoformat() == "2027-03-14T07:30:00+00:00"


def test_dst_overlap_uses_the_earlier_instant() -> None:
    resolved = resolve_local_datetime(
        date(2027, 11, 7), time(1, 30), ZoneInfo("America/New_York")
    )
    assert resolved.fold == 0
    assert resolved.astimezone(timezone.utc).isoformat() == "2027-11-07T05:30:00+00:00"
