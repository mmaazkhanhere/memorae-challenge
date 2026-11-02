from datetime import datetime, timezone
import json

from reminder_agent.intent import (
    IntentRoute,
    Interpretation,
    ListRemindersIntent,
    OpenAICompatibleIntentInterpreter,
    PendingReminder,
    ReminderIntentInterpreter,
    RuleBasedIntentInterpreter,
)


def test_rule_interpreter_parses_tomorrow_in_user_timezone():
    result = RuleBasedIntentInterpreter().interpret(
        "What do I have tomorrow?",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-10-31"
    assert result.intent.end_date == result.intent.start_date


def test_rule_interpreter_parses_date_range_and_all_reminders():
    interpreter = RuleBasedIntentInterpreter()
    result = interpreter.interpret(
        "List my reminders from 2026-11-02 to 2026-11-05",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )
    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-11-02"
    assert result.intent.end_date.isoformat() == "2026-11-05"
    all_reminders = interpreter.interpret(
        "Show me all reminders",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )
    assert isinstance(all_reminders.intent, ListRemindersIntent)
    assert all_reminders.intent.start_date is None
    assert all_reminders.intent.end_date is None


def test_unrecognized_natural_language_range_falls_back_to_extractor():
    result = RuleBasedIntentInterpreter().interpret(
        "show reminders between November 10 and November 15",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )

    assert result.intent is None
    assert result.fallback_allowed is True


def test_new_request_does_not_consume_an_existing_pending_draft():
    result = RuleBasedIntentInterpreter().interpret(
        "8 PM",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
        context=PendingReminder(task="call the bank", timezone_name="UTC"),
    )

    assert result.intent is None
    assert result.clarification is None


def test_list_request_ignores_existing_pending_draft():
    result = RuleBasedIntentInterpreter().interpret(
        "What do I have tomorrow?",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
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
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
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

    monkeypatch.setattr("reminder_agent.intent.httpx.post", lambda *args, **kwargs: Response())
    result = OpenAICompatibleIntentInterpreter(
        "https://example.test", "key", "model"
    ).interpret(
        "show my reminders early next week",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
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

    monkeypatch.setattr("reminder_agent.intent.httpx.post", lambda *args, **kwargs: Response())
    result = OpenAICompatibleIntentInterpreter(
        "https://example.test", "key", "model"
    ).interpret(
        "show me everything I need to remember",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date is None
    assert result.intent.end_date is None


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

    monkeypatch.setattr("reminder_agent.intent.httpx.post", lambda *args, **kwargs: Response())
    result = OpenAICompatibleIntentInterpreter(
        "https://example.test", "key", "model"
    ).interpret(
        "show reminders between November 10 and November 15",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.start_date.isoformat() == "2026-11-10"
    assert result.intent.end_date.isoformat() == "2026-11-15"
