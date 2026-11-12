"""Characterization coverage for structured intent interpretation."""

from datetime import UTC, datetime

from reminder_agent.features.intent import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
    IntentRoute,
    ListRemindersIntent,
    NeedsClarification,
    OpenAICompatibleIntentInterpreter,
    PendingReminder,
    PendingReminderBatch,
)

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


class StubStructuredProvider:
    url = "https://llm.example/v1/chat/completions"

    def __init__(self, content: str | dict) -> None:
        self.content = content
        self.payload: dict | None = None

    def complete(self, payload: dict) -> tuple[dict, str | dict]:
        self.payload = payload
        return {}, self.content


def _interpreter(content: str | dict) -> tuple[OpenAICompatibleIntentInterpreter, StubStructuredProvider]:
    provider = StubStructuredProvider(content)
    return (
        OpenAICompatibleIntentInterpreter(
            "https://llm.example/v1", "secret", "model", client=provider
        ),
        provider,
    )


def test_structured_one_time_response_converts_user_local_time_to_utc() -> None:
    interpreter, provider = _interpreter(
        '{"operation":"CREATE_ONE_TIME_REMINDER",'
        '"task":"call the bank","due_local":"2026-10-31T10:00"}'
    )

    result = interpreter.interpret(
        "Remind me tomorrow at 10 AM to call the bank",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, CreateOneTimeReminderIntent)
    assert result.intent.due_at == datetime(2026, 10, 31, 5, 0, tzinfo=UTC)
    assert result.intent.task == "call the bank"
    assert provider.payload is not None
    assert provider.payload["temperature"] == 0
    assert "CREATE_ONE_TIME_REMINDER" in provider.payload["messages"][0]["content"]


def test_structured_one_time_response_infers_missing_operation_for_new_request() -> None:
    interpreter, _ = _interpreter(
        '{"task":"send the proposal to Sarah","due_local":"2026-11-12T09:00"}'
    )

    result = interpreter.interpret(
        "Oh, and tomorrow morning at 9 am, remind me to send the proposal to Sarah.",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, CreateOneTimeReminderIntent)
    assert result.intent.task == "send the proposal to Sarah"
    assert result.intent.due_at == datetime(2026, 11, 12, 4, 0, tzinfo=UTC)


def test_structured_one_time_response_does_not_infer_operation_with_extra_fields() -> None:
    interpreter, _ = _interpreter(
        '{"task":"send the proposal to Sarah","due_local":"2026-11-12T09:00",'
        '"unexpected":"value"}'
    )

    result = interpreter.interpret(
        "Oh, and tomorrow morning at 9 am, remind me to send the proposal to Sarah.",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is None


def test_structured_multi_reminder_clarification_preserves_all_drafts() -> None:
    interpreter, _ = _interpreter(
        {
            "operation": "NEEDS_CLARIFICATION",
            "question": "Should 6 be AM or PM for the electricity bill?",
            "pending": {
                "reminders": [
                    {
                        "task": "call John",
                        "date_local": "2026-10-31",
                        "hour": 10,
                        "minute": 0,
                        "meridiem": "am",
                        "date_explicit": True,
                    },
                    {
                        "task": "pay my electricity bill",
                        "date_local": "2026-11-06",
                        "hour": 6,
                        "minute": 0,
                        "date_explicit": True,
                    },
                ],
                "clarification_index": 1,
            },
        }
    )

    result = interpreter.interpret(
        "Remind me to call John tomorrow at 10, and pay my electricity bill on Friday at 6.",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.clarification, NeedsClarification)
    assert isinstance(result.clarification.pending, PendingReminderBatch)
    batch = result.clarification.pending
    assert batch.clarification_index == 1
    assert [draft.task for draft in batch.reminders] == [
        "call John",
        "pay my electricity bill",
    ]


def test_structured_recurring_response_normalizes_provider_time_aliases() -> None:
    interpreter, _ = _interpreter(
        '{"operation":"CREATE_RECURRING_REMINDER",'
        '"task":"team standup","frequency":"WEEKLY","interval":1,'
        '"day_of_week":"Monday","time":"8",'
        '"start_local":"2026-11-09T08:00"}'
    )

    result = interpreter.interpret(
        "Every Monday at 8, remind me about the team standup.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.task == "team standup"
    assert result.intent.weekdays == (0,)
    assert result.intent.hour == 8
    assert result.intent.minute == 0


def test_structured_recurring_response_accepts_legacy_recurrence_weekdays_alias() -> None:
    interpreter, _ = _interpreter(
        '{"operation":"CREATE_RECURRING_REMINDER",'
        '"task":"exercise","frequency":"WEEKLY","interval":1,'
        '"recurrence_weekdays":["Monday","Tuesday","Wednesday","Thursday",'
        '"Friday","Saturday","Sunday"],"hour":9,"minute":0,'
        '"start_local":"2026-10-31T09:00"}'
    )

    result = interpreter.interpret(
        "I want you to remind me to exercise every morning at 9 am",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.weekdays == (0, 1, 2, 3, 4, 5, 6)
    assert result.intent.hour == 9
    assert result.intent.minute == 0


def test_structured_envelope_response_is_normalized_without_changing_operation() -> None:
    interpreter, _ = _interpreter(
        {
            "LIST_REMINDERS": {
                "task_query": "bank",
                "start_date": "2026-10-31",
                "end_date": "2026-10-31",
            }
        }
    )

    result = interpreter.interpret(
        "Show my bank reminders tomorrow",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.task_query == "bank"
    assert result.intent.start_date.isoformat() == "2026-10-31"


def test_structured_listing_response_normalizes_rich_filters() -> None:
    interpreter, _ = _interpreter(
        {
            "operation": "LIST_REMINDERS",
            "time_after": "6 PM",
            "weekdays": ["Saturday", "Sunday"],
            "limit": 1,
        }
    )

    result = interpreter.interpret(
        "Show reminders after 6 PM on weekends",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.time_after.isoformat() == "18:00:00"
    assert result.intent.weekdays == (5, 6)
    assert result.intent.limit == 1


def test_structured_listing_response_reads_nested_filters() -> None:
    interpreter, _ = _interpreter(
        {
            "operation": "LIST_REMINDERS",
            "filters": {"weekdays": ["Friday"]},
        }
    )

    result = interpreter.interpret(
        "Any reminders for Friday?",
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, ListRemindersIntent)
    assert result.intent.weekdays == (4,)


def test_structured_clarification_preserves_trusted_pending_fields() -> None:
    interpreter, _ = _interpreter(
        '{"operation":"NEEDS_CLARIFICATION","question":"What time should I use?",'
        '"pending":{"task":"pay rent","hour":null}}'
    )
    context = PendingReminder(task="pay rent", timezone_name="UTC", date_explicit=False)

    result = interpreter.interpret(
        "Remind me to pay rent",
        context=context,
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.clarification, NeedsClarification)
    assert result.clarification.pending.task == "pay rent"
    assert result.clarification.pending.timezone_name == "UTC"
    assert result.clarification.question == "What time should I use?"


def test_malformed_structured_response_returns_fallback_eligible_error() -> None:
    interpreter, _ = _interpreter("{not json")

    result = interpreter.interpret(
        "Remind me to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.response_text == (
        "I couldn't understand that reminder. Please provide the task and future time."
    )
    assert result.fallback_allowed is True


def test_memory_extraction_failure_is_a_non_fallback_clarification() -> None:
    interpreter, _ = _interpreter(
        '{"operation":"SAVE_MEMORY","content":"   "}'
    )

    result = interpreter.interpret(
        "Please remember something",
        selected_route=IntentRoute.SAVE_MEMORY,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.memory_fact is None
    assert result.response_text == "What information should I remember?"
    assert result.fallback_allowed is False
