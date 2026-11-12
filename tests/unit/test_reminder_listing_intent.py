import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from reminder_agent.features.intent import (
    IntentRoute,
    Interpretation,
    ListRemindersIntent,
    OpenAICompatibleIntentInterpreter,
    PendingReminder,
    ReminderIntentInterpreter,
    ReminderSummaryIntent,
    RuleBasedIntentInterpreter,
)


def test_rule_interpreter_parses_tomorrow_in_user_timezone():
    result = RuleBasedIntentInterpreter().interpret(
        "What do I have tomorrow?",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-10-31"
    assert result.intent.end_date == result.intent.start_date


def test_rule_interpreter_parses_date_range_and_all_reminders():
    interpreter = RuleBasedIntentInterpreter()
    result = interpreter.interpret(
        "List my reminders from 2026-11-02 to 2026-11-05",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )
    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-11-02"
    assert result.intent.end_date.isoformat() == "2026-11-05"
    all_reminders = interpreter.interpret(
        "Show me all reminders",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )
    assert isinstance(all_reminders.intent, ListRemindersIntent)
    assert all_reminders.intent.start_date is None
    assert all_reminders.intent.end_date is None


@pytest.mark.parametrize(
    "text", ["What recurring reminders have I got?", "What weekly reminders do I have?"]
)
def test_rule_interpreter_extracts_recurring_listing_without_model(text: str) -> None:
    result = RuleBasedIntentInterpreter().interpret(
        text,
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.recurrence_only is True
    assert result.intent.recurrence_series_only is True
    assert result.intent.recurrence_frequency == (
        "WEEKLY" if "weekly" in text.lower() else None
    )
    assert result.fallback_allowed is False


@pytest.mark.parametrize(
    ("text", "metric"),
    [
        ("How many reminders are pending?", "PENDING_COUNT"),
        ("What do I still have to do today?", "TODAY_REMAINING"),
        ("How many tasks have I completed?", "COMPLETED_COUNT"),
        ("How many tasks have I completed today?", "TODAY_COMPLETED"),
        ("How many reminders are overdue?", "OVERDUE_COUNT"),
    ],
)
def test_rule_interpreter_parses_progress_questions(text, metric):
    result = RuleBasedIntentInterpreter().interpret(
        text,
        selected_route=IntentRoute.SUMMARY_REQUEST,
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ReminderSummaryIntent)
    assert result.intent.metric == metric
    assert result.fallback_allowed is False


@pytest.mark.parametrize(
    ("text", "task_query"),
    [
        ("When are my doctor's appointments?", "doctor's appointments"),
        ("When do I have to call John?", "call John"),
        ("Show my gym reminders.", "gym"),
    ],
)
def test_rule_interpreter_extracts_explicit_listing_task_without_model(text, task_query):
    result = RuleBasedIntentInterpreter().interpret(
        text,
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.task_query == task_query
    assert result.fallback_allowed is False


def test_rule_interpreter_combines_listing_task_and_tomorrow():
    result = RuleBasedIntentInterpreter().interpret(
        "When are my doctor's appointments tomorrow?",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.task_query == "doctor's appointments"
    assert result.intent.start_date.isoformat() == "2026-10-31"


@pytest.mark.parametrize(
    ("text", "task_query"),
    [("Show me all my callreminders", "call")],
)
def test_rule_interpreter_preserves_a_compound_word_task_filter(text, task_query):
    result = RuleBasedIntentInterpreter().interpret(
        text,
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.task_query == task_query


@pytest.mark.parametrize(
    ("text", "start_date", "end_date"),
    [
        ("Hey, what is on my reminder list for tomorrow?", "2026-10-31", "2026-10-31"),
        ("and on Firday?", "2026-10-30", "2026-10-30"),
    ],
)
def test_rule_interpreter_handles_conversational_date_list_follow_ups(
    text, start_date, end_date
):
    result = RuleBasedIntentInterpreter().interpret(
        text,
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == start_date
    assert result.intent.end_date.isoformat() == end_date
    assert result.intent.task_query is None


def test_rule_interpreter_scopes_weekend_agenda_to_the_next_weekend():
    result = RuleBasedIntentInterpreter().interpret(
        "What's on my agenda for the weekend?",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 11, 11, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-11-14"
    assert result.intent.end_date.isoformat() == "2026-11-15"
    assert result.intent.weekdays is None


def test_rule_interpreter_maps_explicit_day_parts_to_meridiem():
    interpreter = RuleBasedIntentInterpreter()
    weekly = interpreter.interpret(
        "Every Monday at 8 in the morning, remind me about the team standup.",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 11, 11, 12, tzinfo=UTC),
    )
    follow_up = interpreter.interpret(
        "Around 2 in the afternoon.",
        selected_route=IntentRoute.UPDATE_PENDING,
        timezone_name="Asia/Karachi",
        now=datetime(2026, 11, 11, 12, tzinfo=UTC),
        context=PendingReminder(
            task="team standup",
            date_local=datetime(2026, 11, 16).date(),
            hour=8,
            timezone_name="Asia/Karachi",
            date_explicit=True,
        ),
    )

    assert weekly.intent is not None
    assert weekly.intent.hour == 8
    assert follow_up.intent is not None
    assert follow_up.intent.due_at.astimezone(ZoneInfo("Asia/Karachi")).hour == 14


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Any reminders for Friday?", {"start_date": "2026-10-30", "end_date": "2026-10-30"}),
        ("What's my next reminder to do?", {"limit": 1}),
        ("Do I have any payment reminders?", {"task_query": "payment"}),
        ("Show reminders after 6 PM.", {"time_after": "18:00:00"}),
        (
            "What do I have scheduled for next month?",
            {"start_date": "2026-11-01", "end_date": "2026-11-30"},
        ),
        ("Do I have any reminders on weekends?", {"weekdays": (5, 6)}),
    ],
)
def test_rule_interpreter_extracts_rich_listing_filters(text, expected):
    result = RuleBasedIntentInterpreter().interpret(
        text,
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    for field, expected_value in expected.items():
        actual = getattr(result.intent, field)
        if field in {"time_after", "start_date", "end_date"}:
            assert actual.isoformat() == expected_value
        else:
            assert actual == expected_value


def test_task_listing_deterministic_path_does_not_call_model():
    class Provider:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            raise AssertionError("deterministic task listing should not use the provider")

    provider = Provider()
    result = ReminderIntentInterpreter(provider).interpret(
        "Show my gym reminders.",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.task_query == "gym"
    assert provider.called is False


def test_unrecognized_natural_language_range_falls_back_to_extractor():
    result = RuleBasedIntentInterpreter().interpret(
        "show reminders between November 10 and November 15",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert result.intent is None
    assert result.fallback_allowed is True


def test_new_request_does_not_consume_an_existing_pending_draft():
    result = RuleBasedIntentInterpreter().interpret(
        "8 PM",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
        context=PendingReminder(task="call the bank", timezone_name="UTC"),
    )

    assert result.intent is None
    assert result.clarification is None


def test_list_request_ignores_existing_pending_draft():
    result = RuleBasedIntentInterpreter().interpret(
        "What do I have tomorrow?",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
        context=PendingReminder(task="call the bank", timezone_name="UTC"),
    )

    assert isinstance(result.intent, ListRemindersIntent)


def test_uncertain_route_does_not_call_the_extraction_provider():
    class Provider:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(response_text="should not be used")

    provider = Provider()
    result = ReminderIntentInterpreter(provider).interpret(
        "anything",
        selected_route=IntentRoute.UNCERTAIN,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert provider.called is False
    assert result.response_text == "I’m not sure what you want to do. Please clarify."


def test_model_interpreter_validates_read_only_date_range(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "operation": "LIST_REMINDERS",
                                    "start_date": "2026-11-02",
                                    "end_date": "2026-11-05",
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "reminder_agent.features.intent.httpx.post", lambda *args, **kwargs: Response()
    )
    result = OpenAICompatibleIntentInterpreter("https://example.test", "key", "model").interpret(
        "show my reminders early next week",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-11-02"
    assert result.intent.end_date.isoformat() == "2026-11-05"


def test_model_interpreter_accepts_all_reminders_without_a_date_range(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"operation":"LIST_REMINDERS"}'}}]}

    monkeypatch.setattr(
        "reminder_agent.features.intent.httpx.post", lambda *args, **kwargs: Response()
    )
    result = OpenAICompatibleIntentInterpreter("https://example.test", "key", "model").interpret(
        "show me everything I need to remember",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date is None
    assert result.intent.end_date is None


@pytest.mark.parametrize(
    "model_payload",
    [
        {"operation": "LIST_REMINDERS"},
        {"operation": "LIST_REMINDERS", "task_query": ["call John"]},
    ],
)
def test_model_interpreter_rejects_missing_or_malformed_explicit_task_filter(
    monkeypatch, model_payload
):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(model_payload)}}]}

    monkeypatch.setattr(
        "reminder_agent.features.intent.httpx.post", lambda *args, **kwargs: Response()
    )
    result = OpenAICompatibleIntentInterpreter("https://example.test", "key", "model").interpret(
        "When do I have to call John?",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert result.intent is None


def test_model_interpreter_rejects_reminder_records_in_a_list_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "operation": "LIST_REMINDERS",
                                    "date": "2026-11-03",
                                    "reminders": [
                                        {
                                            "task": "untrusted model result",
                                            "due_local": "2026-11-03T08:00",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "reminder_agent.features.intent.httpx.post", lambda *args, **kwargs: Response()
    )
    result = OpenAICompatibleIntentInterpreter("https://example.test", "key", "model").interpret(
        "what reminders do i have tomorrow?",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="Asia/Karachi",
        now=datetime(2026, 11, 2, 16, 59, tzinfo=UTC),
    )

    assert result.intent is None


def test_model_interpreter_normalizes_nested_date_range(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"LIST_REMINDERS",'
                                '"date_range":{"start":"2026-11-10",'
                                '"end":"2026-11-15"}}'
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "reminder_agent.features.intent.httpx.post", lambda *args, **kwargs: Response()
    )
    result = OpenAICompatibleIntentInterpreter("https://example.test", "key", "model").interpret(
        "show reminders between November 10 and November 15",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-11-10"
    assert result.intent.end_date.isoformat() == "2026-11-15"
