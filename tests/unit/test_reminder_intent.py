import json
import logging
from datetime import UTC, date, datetime

import httpx
import pytest

from reminder_agent.application.conversation_coordinator import _route_and_interpret
from reminder_agent.features.intent import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    NeedsClarification,
    OpenAICompatibleIntentInterpreter,
    OpenAICompatibleIntentRouter,
    OpenAICompatibleMemoryAnswerer,
    PendingReminder,
    PendingReminderFieldUpdate,
    ReminderIntentInterpreter,
    RuleBasedIntentInterpreter,
    resolve_pending_due_at,
    restrict_to_explicit_field_updates,
)
from reminder_agent.features.intent.deterministic.routing import ConversationRoute
from reminder_agent.features.intent.policy import OUT_OF_SCOPE_RESPONSE
from reminder_agent.features.memory.messages import MEMORY_LOOKUP_MESSAGES
from reminder_agent.features.memory.retrieve.contracts import MemoryAnswerProposal
from reminder_agent.features.reminders.reschedule.decisions import (
    RescheduleRequest,
    RescheduleScope,
)
from reminder_agent.features.reminders.reschedule.models import RescheduleClarification
from reminder_agent.infrastructure.config.settings import Settings
from reminder_agent.shared.memory_contracts import SaveMemoryFact

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def test_rule_interpreter_extracts_remember_to_as_a_reminder() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remember to call Ali tomorrow at 9 AM",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call Ali"


def test_rule_interpreter_clarifies_remember_to_without_a_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remember to call Ali",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "call Ali"


def test_monthly_ordinal_request_preserves_recurrence_while_asking_for_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me on the first day of every month to pay rent",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.question == "What time should I use on the 1st day of each month?"
    assert result.clarification.pending.task == "pay rent"
    assert result.clarification.pending.recurrence_frequency == "MONTHLY"
    assert result.clarification.pending.recurrence_day_of_month == 1


def test_monthly_ordinal_follow_up_builds_recurring_intent() -> None:
    parser = RuleBasedIntentInterpreter()
    first = parser.interpret(
        "Remind me on the first day of every month to pay rent",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )
    assert first.clarification is not None

    result = parser.interpret(
        "9 am",
        selected_route=IntentRoute.UPDATE_PENDING,
        context=first.clarification.pending,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.frequency == "MONTHLY"
    assert result.intent.day_of_month == 1
    assert result.intent.task == "pay rent"
    assert result.intent.hour == 9


def test_monthly_numeric_ordinal_without_suffix_preserves_recurrence() -> None:
    parser = RuleBasedIntentInterpreter()
    first = parser.interpret(
        "Remind me on the 2 of every month to pay bills",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert first.clarification is not None
    assert first.clarification.pending.task == "pay bills"
    assert first.clarification.pending.recurrence_frequency == "MONTHLY"
    assert first.clarification.pending.recurrence_day_of_month == 2

    result = parser.interpret(
        "5 pm",
        selected_route=IntentRoute.UPDATE_PENDING,
        context=first.clarification.pending,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.frequency == "MONTHLY"
    assert result.intent.day_of_month == 2
    assert result.intent.task == "pay bills"
    assert result.intent.hour == 17


def test_monthly_numeric_ordinal_with_suffix_preserves_recurrence() -> None:
    parser = RuleBasedIntentInterpreter()
    first = parser.interpret(
        "remind me on the 15th of every month to call parents",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert first.clarification is not None
    assert first.clarification.pending.task == "call parents"
    assert first.clarification.pending.recurrence_day_of_month == 15

    result = parser.interpret(
        "6 pm",
        selected_route=IntentRoute.UPDATE_PENDING,
        context=first.clarification.pending,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.frequency == "MONTHLY"
    assert result.intent.day_of_month == 15
    assert result.intent.hour == 18


def test_intent_router_classifies_only_the_operation_and_limits_history(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": '{"operation":"NEW_REQUEST"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    suggestion = OpenAICompatibleIntentRouter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).classify(
        "tommorow",
        pending=PendingReminder(task="call Bilal", hour=4, meridiem="pm", timezone_name="UTC"),
        recent_messages=[
            *(f"message {index}" for index in range(35)),
            "assistant: What date should I use?",
            "user: unrelated earlier reply",
            "assistant: Please clarify the date.",
            "user: another earlier reply",
            "assistant: What date should I use?",
        ],
    )

    assert suggestion.operation is IntentRoute.NEW_REQUEST
    assert "LIST_REQUEST" in captured["messages"][0]["content"]
    assert "RESCHEDULE_REQUEST" in captured["messages"][0]["content"]
    assert "COMPLETE_REQUEST" in captured["messages"][0]["content"]
    context = json.loads(captured["messages"][1]["content"])
    assert context["active_pending"]["task"] == "call Bilal"
    assert context["missing_fields"] == ["date"]
    assert context["last_clarification_question"] == "What date should I use?"
    assert len(context["recent_messages"]) == 8
    assert "message 31" not in context["recent_messages"]
    assert "message 34" in context["recent_messages"]


def test_intent_router_accepts_list_request_route(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": '{"operation":"LIST_REQUEST"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    suggestion = OpenAICompatibleIntentRouter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).classify(
        "Could you tell me what is scheduled next week?",
        pending=None,
        recent_messages=(),
    )

    assert suggestion.operation is IntentRoute.LIST_REQUEST


def test_intent_router_accepts_reschedule_request_route(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": '{"operation":"RESCHEDULE_REQUEST"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    suggestion = OpenAICompatibleIntentRouter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).classify(
        "Could you shift the bank reminder to tomorrow at 10 AM?",
        pending=None,
        recent_messages=(),
    )

    assert suggestion.operation is IntentRoute.RESCHEDULE_REQUEST


def test_intent_router_prompt_keeps_implicit_reschedules_in_intent_classification(
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": '{"operation":"RESCHEDULE_REQUEST"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    suggestion = OpenAICompatibleIntentRouter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).classify(
        "Actually move it to 9 PM today",
        pending=None,
        recent_messages=[
            "user: Remind me one minute to call Ali",
            "assistant: I'll remind you to call Ali today at 5:18 PM on Telegram.",
        ],
    )

    assert suggestion.operation is IntentRoute.RESCHEDULE_REQUEST
    prompt = captured["messages"][0]["content"]
    assert "Classify the operation independently of target resolution" in prompt


def test_intent_router_accepts_save_memory_route_and_teaches_conservative_prompt(
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"SAVE_MEMORY",'
                                '"memory_intent_evidence":{"kind":"EXPLICIT_SAVE_REQUEST",'
                                '"confidence":"CLEAR","evidence_text":"Make a note"}}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    suggestion = OpenAICompatibleIntentRouter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).classify(
        "Make a note that my gate code is 4512",
        pending=None,
        recent_messages=(),
    )

    assert suggestion.operation is IntentRoute.SAVE_MEMORY
    assert suggestion.memory_intent_evidence is not None
    assert suggestion.memory_intent_evidence.kind == "EXPLICIT_SAVE_REQUEST"
    prompt = captured["messages"][0]["content"]
    assert "ordinary statement" in prompt
    assert "SAVE_MEMORY" in prompt


def test_openai_compatible_extractor_returns_typed_memory_content(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"operation":"SAVE_MEMORY","content":"gate code is 4512"}'
                        }
                    },
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "Make a note that my gate code is 4512",
        selected_route=IntentRoute.SAVE_MEMORY,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.memory_fact == SaveMemoryFact(content="gate code is 4512")
    assert "SAVE_MEMORY" in captured["messages"][0]["content"]
    assert "CREATE_ONE_TIME_REMINDER" not in captured["messages"][0]["content"]


def test_malformed_memory_extraction_is_clarification_only(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [{"message": {"content": '{"operation":"SAVE_MEMORY","content":"   "}'}}]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "Please remember something",
        selected_route=IntentRoute.SAVE_MEMORY,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.memory_fact is None
    assert result.response_text == "What information should I remember?"
    assert result.fallback_allowed is False


def test_openai_compatible_extractor_returns_groundable_memory_lookup_query(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {"message": {"content": '{"operation":"RETRIEVE_MEMORY","query":"color"}'}}
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "Which color do I like?",
        selected_route=IntentRoute.RETRIEVE_MEMORY,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.memory_query is not None and result.memory_query.query == "color"
    assert "RETRIEVE_MEMORY" in captured["messages"][0]["content"]
    assert "CREATE_ONE_TIME_REMINDER" not in captured["messages"][0]["content"]


def test_memory_answerer_uses_strict_grounded_contract_on_groq(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "ANSWER",
                                    "candidate_indices": [0],
                                    "answer": "Your parking slot is B17.",
                                    "clarification_question": None,
                                }
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleMemoryAnswerer(
        "https://api.groq.com/openai/v1",
        "secret",
        "openai/gpt-oss-120b",
    ).answer(
        question="Where do I leave my car?",
        memories=("my parking slot is B17", "my favorite food is biryani"),
    )

    assert result == MemoryAnswerProposal(
        decision="ANSWER",
        candidate_indices=[0],
        answer="Your parking slot is B17.",
        clarification_question=None,
    )
    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert "untrusted data" in captured["messages"][0]["content"]
    answer_input = json.loads(captured["messages"][1]["content"])
    assert set(answer_input) == {"question", "candidate_memories"}
    assert len(answer_input["candidate_memories"]) == 2


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps(
            {
                "decision": "ANSWER",
                "candidate_indices": [3],
                "answer": "Invented",
                "clarification_question": None,
            }
        ),
        json.dumps(
            {
                "decision": "UNKNOWN",
                "candidate_indices": [],
                "answer": "Invented",
                "clarification_question": None,
            }
        ),
    ],
)
def test_memory_answerer_rejects_malformed_or_unsupported_output(monkeypatch, content) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": content}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleMemoryAnswerer(
        "https://api.groq.com/openai/v1",
        "secret",
        "openai/gpt-oss-120b",
    ).answer(question="Where do I park?", memories=("my parking slot is B17",))
    assert result is None


@pytest.mark.parametrize(
    "payload",
    [
        '{"operation":"RETRIEVE_MEMORY","query":"color","answer":"blue"}',
        '{"operation":"RETRIEVE_MEMORY","query":"color","fact_id":"9d03e6c2-ef28-40d7-80f1-c5720151d129"}',
        '{"operation":"RETRIEVE_MEMORY","query":"color","user_id":"9d03e6c2-ef28-40d7-80f1-c5720151d129"}',
        '{"operation":"RETRIEVE_MEMORY","query":"user ID 9d03e6c2-ef28-40d7-80f1-c5720151d129"}',
    ],
)
def test_malformed_memory_lookup_extraction_is_clarification_only(
    monkeypatch, payload: str
) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": payload}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "Which color do I like?",
        selected_route=IntentRoute.RETRIEVE_MEMORY,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.memory_query is None
    assert result.response_text in MEMORY_LOOKUP_MESSAGES["missing_subject"]


def test_memory_classifier_and_extractor_logs_redact_fact_content(monkeypatch) -> None:
    secret = "gate code is SECRET-4512"
    responses = iter(
        [
            '{"operation":"SAVE_MEMORY",'
            '"memory_intent_evidence":{"kind":"EXPLICIT_SAVE_REQUEST",'
            '"confidence":"CLEAR","evidence_text":"Keep in mind"}}',
            '{"operation":"SAVE_MEMORY","content":"gate code is SECRET-4512"}',
        ]
    )

    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    logger = logging.getLogger("reminder_agent")
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = Capture()
    logger.addHandler(handler)
    try:
        OpenAICompatibleIntentRouter("https://llm.example/v1", "secret", "intent-model").classify(
            f"Keep in mind that my {secret}",
            pending=None,
            recent_messages=(f"user: {secret}",),
        )
        result = OpenAICompatibleIntentInterpreter(
            "https://llm.example/v1", "secret", "extract-model"
        ).interpret(
            f"Keep in mind that my {secret}",
            selected_route=IntentRoute.SAVE_MEMORY,
            timezone_name="UTC",
            now=NOW,
            recent_messages=(f"user: {secret}",),
        )
    finally:
        logger.removeHandler(handler)

    assert result.memory_fact is not None
    log_text = " ".join(
        f"{record.getMessage()} {getattr(record, 'context', {})}" for record in captured
    )
    assert secret not in log_text


def test_malformed_memory_provider_exception_log_redacts_fact_content(monkeypatch) -> None:
    secret = "gate code is SECRET-4512"

    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": f'{{"operation":"SAVE_MEMORY","content":null,"fact":"{secret}"}}'
                        }
                    },
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    logger = logging.getLogger("reminder_agent")
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = Capture()
    logger.addHandler(handler)
    try:
        result = OpenAICompatibleIntentInterpreter(
            "https://llm.example/v1", "secret", "extract-model"
        ).interpret(
            f"Keep in mind that my {secret}",
            selected_route=IntentRoute.SAVE_MEMORY,
            timezone_name="UTC",
            now=NOW,
        )
    finally:
        logger.removeHandler(handler)

    assert result.memory_fact is None
    log_text = " ".join(
        f"{record.getMessage()} {getattr(record, 'context', {})}" for record in captured
    )
    assert secret not in log_text


def test_intent_router_accepts_cancel_request_route_and_prompt(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": '{"operation":"CANCEL_REQUEST"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    suggestion = OpenAICompatibleIntentRouter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).classify(
        "I do not need the bank reminder anymore",
        pending=None,
        recent_messages=(),
    )

    assert suggestion.operation is IntentRoute.CANCEL_REQUEST
    assert "CANCEL_REQUEST" in captured["messages"][0]["content"]


def test_list_request_route_from_classifier_reaches_interpreter() -> None:
    class ListRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.LIST_REQUEST)

    class RecordingInterpreter:
        def __init__(self) -> None:
            self.called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(response_text="listing")

    interpreter = RecordingInterpreter()
    route, result = _route_and_interpret(
        interpreter,
        "Could you tell me what is scheduled next week?",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=ListRouter(),
    )

    assert route is ConversationRoute.LIST_REQUEST
    assert interpreter.called is True
    assert result.response_text == "listing"


def test_classifier_update_pending_without_context_recovers_as_new_request() -> None:
    class UpdateRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.UPDATE_PENDING)

    class RecordingInterpreter:
        def __init__(self) -> None:
            self.selected_route = None

        def interpret(self, text, *, selected_route, **kwargs):
            self.selected_route = selected_route
            return Interpretation(response_text="recurring reminder")

    interpreter = RecordingInterpreter()
    route, result = _route_and_interpret(
        interpreter,
        "Every day, remind me at 2:10 pm to call parents",
        timezone_name="Asia/Karachi",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=UpdateRouter(),
    )

    assert route is ConversationRoute.NEW_REQUEST
    assert interpreter.selected_route is IntentRoute.NEW_REQUEST
    assert result.response_text == "recurring reminder"


def test_classifier_reschedule_route_bypasses_reminder_extractor() -> None:
    class RescheduleRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.RESCHEDULE_REQUEST)

    class RecordingInterpreter:
        def __init__(self) -> None:
            self.called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(response_text="should not be called")

    interpreter = RecordingInterpreter()
    route, result = _route_and_interpret(
        interpreter,
        "Could you shift the bank reminder to tomorrow at 10 AM?",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=RescheduleRouter(),
    )

    assert route is ConversationRoute.RESCHEDULE_REQUEST


def test_classifier_completion_route_bypasses_extractor() -> None:
    class CompletionRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.COMPLETE_REQUEST)

    class RecordingInterpreter:
        def __init__(self) -> None:
            self.called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(response_text="must not be called")

    interpreter = RecordingInterpreter()
    route, result = _route_and_interpret(
        interpreter,
        "I finished it",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=CompletionRouter(),
    )

    assert route is ConversationRoute.COMPLETE_REQUEST
    assert interpreter.called is False
    assert result.fallback_allowed is False


def test_classifier_cancel_route_defers_target_extraction_to_cancellation_service() -> None:
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    class CancelExtractor:
        def interpret_cancellation(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("target extraction must wait for a zero-match lookup")

        def interpret(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("cancel route must bypass reminder creation extraction")

    route, result = _route_and_interpret(
        CancelExtractor(),
        "I do not need the bank reminder anymore",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=CancelRouter(),
    )

    assert route is ConversationRoute.CANCEL_REQUEST
    assert result.cancellation_request is None
    assert result.fallback_allowed is False


def test_custom_cancellation_extractor_receives_supported_variadic_context() -> None:
    captured: dict = {}

    class CustomExtractor:
        def interpret_cancellation(self, text, **kwargs):
            captured["text"] = text
            captured.update(kwargs)
            return None

    ReminderIntentInterpreter(CustomExtractor()).interpret_cancellation(
        "Cancel that",
        timezone_name="UTC",
        now=NOW,
        recent_messages=("user: Remind me to call the bank",),
        cancellation_candidates=(
            {"task": "call the bank", "schedule_type": "ONE_TIME", "due_at": ""},
        ),
    )

    assert captured["text"] == "Cancel that"
    assert captured["recent_messages"] == ("user: Remind me to call the bank",)
    assert captured["cancellation_candidates"][0]["task"] == "call the bank"


def test_openai_interpreter_extracts_non_authoritative_cancellation_request(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"operation":"CANCEL_REMINDER","target_text":"bank reminder","scope":"SERIES"}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).interpret(
        "I do not need the bank reminder anymore",
        selected_route=IntentRoute.CANCEL_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.cancellation_request is not None
    assert result.cancellation_request.target_text == "bank reminder"
    assert result.cancellation_request.scope is None


def test_cancellation_uses_strict_json_schema_on_supported_provider(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "operation": "CANCEL_REMINDER",
                                    "target_text": "bank reminder",
                                    "scope": None,
                                    "occurrence_date": None,
                                }
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://api.groq.com/openai/v1", "secret", "openai/gpt-oss-20b"
    ).interpret(
        "Cancel the bank reminder",
        selected_route=IntentRoute.CANCEL_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert result.cancellation_request is not None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"cancellation": {"operation": "CANCEL_REMINDER"}}),
        json.dumps({"operation": "CANCEL_REMINDER", "scope": None}),
        json.dumps(
            {
                "operation": "CANCEL_REMINDER",
                "target_text": "bank reminder",
                "scope": "ALL",
                "occurrence_date": None,
            }
        ),
    ],
    ids=["malformed", "nested", "missing-target", "invalid-enum"],
)
def test_invalid_cancellation_provider_responses_fail_closed(monkeypatch, content) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": content}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).interpret(
        "Cancel the bank reminder",
        selected_route=IntentRoute.CANCEL_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.cancellation_request is None


def test_openai_interpreter_grounds_contextual_cancellation_in_recent_messages(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"operation":"CANCEL_REMINDER","target_text":"call Ahmed"}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).interpret(
        "Actually, cancel that",
        selected_route=IntentRoute.CANCEL_REQUEST,
        timezone_name="Asia/Karachi",
        now=NOW,
        recent_messages=(
            "user: Remind me tomorrow at 9 AM to call Ahmed",
            "assistant: I'll remind you to call Ahmed tomorrow at 09:00 PKT.",
        ),
    )

    assert result.cancellation_request is not None
    assert result.cancellation_request.target_text == "call Ahmed"


def test_openai_interpreter_rejects_target_absent_from_current_and_recent_context(
    monkeypatch,
) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"operation":"CANCEL_REMINDER","target_text":"secret project"}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).interpret(
        "Actually, cancel that",
        selected_route=IntentRoute.CANCEL_REQUEST,
        timezone_name="UTC",
        now=NOW,
        recent_messages=("user: Remind me to call Ahmed",),
    )

    assert result.cancellation_request is None
    assert result.fallback_allowed is True


@pytest.mark.parametrize(
    "text",
    ["what do i have on every Tuesday?", "can you show me all the recurring reminders?"],
)
def test_recurring_list_phrases_are_parsed_as_read_only_list_requests(text: str) -> None:
    result = RuleBasedIntentInterpreter().interpret(text, timezone_name="Asia/Karachi", now=NOW)

    assert result.intent is not None
    assert result.intent.operation == "LIST_REMINDERS"
    assert result.fallback_allowed is False


def test_pending_update_normalises_meridiem_case() -> None:
    assert PendingReminderFieldUpdate(meridiem="PM").meridiem == "pm"
    assert PendingReminderFieldUpdate(meridiem="AM").meridiem == "am"


def test_pending_update_rejects_invalid_meridiem() -> None:
    with pytest.raises(ValueError):
        PendingReminderFieldUpdate(meridiem="evening")


def test_pending_update_accepts_evidence_for_every_allowed_update_field() -> None:
    update = PendingReminderFieldUpdate(
        date_explicit=True,
        certainty="clear",
        evidence={"date_explicit": "tommorow"},
    )

    assert update.certainty == "CLEAR"
    assert update.evidence == {"date_explicit": "tommorow"}


def test_default_llm_configuration_targets_openai_compatible_api() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_extract_model == "openai/gpt-oss-120b"
    assert settings.llm_intent_model == "openai/gpt-oss-20b"


def test_relative_minutes_are_resolved_from_one_reference_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me in five minutes to call the bank.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call the bank"
    assert result.intent.due_at == datetime(2026, 10, 30, 12, 5, tzinfo=UTC)


def test_relative_minutes_support_task_before_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice in two minutes.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call Alice"
    assert result.intent.due_at == datetime(2026, 10, 30, 12, 2, tzinfo=UTC)


def test_tomorrow_at_nine_respects_user_local_timezone() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow at 9 AM to renew the domain.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "renew the domain"
    assert result.intent.due_at == datetime(2026, 10, 31, 4, 0, tzinfo=UTC)


def test_missing_task_is_clarified_without_an_intent() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me in five minutes.",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.response_text == "What should I remind you about?"


def test_missing_task_preserves_a_complete_tomorrow_schedule() -> None:
    interpreter = RuleBasedIntentInterpreter()

    initial = interpreter.interpret(
        "Remind me tomorrow at 8 AM.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert initial.clarification is not None
    assert initial.clarification.pending.task is None
    assert initial.clarification.pending.date_local == date(2026, 10, 31)
    assert initial.clarification.pending.hour == 8
    assert initial.clarification.pending.minute == 0
    assert initial.clarification.pending.meridiem == "am"
    assert initial.clarification.pending.date_explicit is True

    completed = interpreter.interpret(
        "Submit my assignment.",
        timezone_name="Asia/Karachi",
        now=NOW,
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.task == "Submit my assignment"
    assert completed.intent.due_at == datetime(2026, 10, 31, 3, 0, tzinfo=UTC)


def test_task_only_request_uses_model_fallback_for_clarification() -> None:
    class RecordingModel:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(task="call the bank", timezone_name="UTC"),
                    question="When should I remind you to call the bank?",
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Remind me to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert result.clarification is not None
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.date_local is None
    assert result.clarification.pending.hour is None


def test_tomorrow_morning_is_a_deterministic_incomplete_request() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow morning to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is not None
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.date_local == date(2026, 10, 31)
    assert result.clarification.pending.hour is None
    assert "time" in result.clarification.question.lower()


def test_day_part_without_task_uses_model_fallback() -> None:
    class RecordingModel:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(timezone_name="UTC"),
                    question="What should I remind you about, and when?",
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Remind me tomorrow morning",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert result.clarification is not None
    assert result.clarification.pending.task is None


def test_provider_clarification_does_not_erase_rule_extracted_task() -> None:
    class IncompleteProvider:
        def interpret(self, text, *, timezone_name, now, context=None):
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        date_local=date(2026, 10, 31),
                        hour=10,
                        meridiem="am",
                        timezone_name=timezone_name,
                    ),
                    question="What should I remind you about?",
                )
            )

    result = ReminderIntentInterpreter(IncompleteProvider()).interpret(
        "Remind me to submit my assignment.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "submit my assignment"
    assert result.clarification.pending.date_local == date(2026, 10, 31)


def test_provider_clarification_preserves_rule_extracted_recurrence_fields() -> None:
    class IncompleteProvider:
        def interpret(self, text, *, timezone_name, now, context=None):
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        date_local=date(2026, 10, 31),
                        hour=9,
                        meridiem="am",
                        timezone_name=timezone_name,
                    ),
                    question="What should I remind you about?",
                )
            )

    interpreter = ReminderIntentInterpreter(IncompleteProvider())
    interpreter._rules.interpret = lambda *args, **kwargs: Interpretation(
        clarification=NeedsClarification(
            pending=PendingReminder(
                task="standup",
                recurrence_weekdays=(0,),
                timezone_name="UTC",
            ),
            question="What time should I use?",
        ),
        fallback_allowed=True,
    )

    result = interpreter.interpret(
        "Remind me every Monday for standup",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "standup"
    assert result.clarification.pending.recurrence_weekdays == (0,)
    assert result.clarification.pending.hour == 9


def test_provider_clarification_normalises_uppercase_meridiem() -> None:
    class UppercaseProvider:
        def interpret(self, text, *, timezone_name, now, context=None):
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        date_local=date(2026, 10, 31),
                        hour=4,
                        meridiem="PM",
                        timezone_name=timezone_name,
                    ),
                    question="What should I remind you about?",
                )
            )

    result = ReminderIntentInterpreter(UppercaseProvider()).interpret(
        "Remind me tomorrow at 4 PM",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.date_local == date(2026, 10, 31)
    assert result.clarification.pending.hour == 4
    assert result.clarification.pending.meridiem == "pm"


def test_weekday_schedule_with_explicit_meridiem_is_parsed_deterministically() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me on Friday at 6 PM to go to the gym",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "go to the gym"
    assert result.intent.due_at == datetime(2026, 10, 30, 18, 0, tzinfo=UTC)


def test_recurring_request_is_parsed_deterministically() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call John every Friday at 6 PM",
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.task == "call John"
    assert result.intent.weekdays == (4,)
    assert result.intent.start_local == datetime(2026, 10, 30, 18)


def test_partial_weekly_request_preserves_task_and_weekday_before_asking_for_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me every Monday for standup",
        timezone_name="Europe/Madrid",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "standup"
    assert result.clarification.pending.recurrence_weekdays == (0,)
    assert result.clarification.pending.hour is None
    assert "time" in result.clarification.question.lower()


def test_partial_weekly_follow_up_fills_time_and_builds_recurring_intent() -> None:
    parser = RuleBasedIntentInterpreter()
    first = parser.interpret(
        "Remind me every Monday for standup",
        timezone_name="Europe/Madrid",
        now=NOW,
    )

    assert first.clarification is not None
    result = parser.interpret(
        "at 9 am",
        selected_route=IntentRoute.UPDATE_PENDING,
        context=first.clarification.pending,
        timezone_name="Europe/Madrid",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.task == "standup"
    assert result.intent.weekdays == (0,)
    assert result.intent.hour == 9


def test_every_day_recurring_request_expands_to_all_weekdays() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me every day at 8 AM to drink water",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.task == "drink water"
    assert result.intent.weekdays == tuple(range(7))
    assert result.intent.hour == 8


def test_everyday_recurring_request_expands_to_all_weekdays() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me everyday at 8 AM to drink water",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.task == "drink water"
    assert result.intent.weekdays == tuple(range(7))
    assert result.intent.hour == 8


@pytest.mark.parametrize(
    ("text", "frequency", "day_of_month", "month"),
    [
        ("Remind me every month on the 15th at 8 AM to pay rent", "MONTHLY", 15, None),
        ("Remind me every year on March 10 at 9 AM to renew insurance", "YEARLY", 10, 3),
    ],
)
def test_monthly_and_yearly_recurrence_requests_are_parsed(
    text: str, frequency: str, day_of_month: int, month: int | None
) -> None:
    result = RuleBasedIntentInterpreter().interpret(
        text,
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.frequency == frequency
    assert result.intent.day_of_month == day_of_month
    assert result.intent.month == month


def test_ambiguous_hour_is_clarified_without_guessing_next_morning() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call the bank at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=UTC),
    )

    assert result.intent is None
    assert isinstance(result.clarification, NeedsClarification)
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.hour == 7
    assert result.clarification.pending.meridiem is None
    assert "7 PM today" in result.clarification.question
    assert "7 AM tomorrow" in result.clarification.question
    assert "7 AM today" not in result.clarification.question


def test_tomorrow_clock_without_meridiem_preserves_all_fields_for_clarification() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me tomorrow at 7:44 to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert initial.intent is None
    assert initial.clarification is not None
    assert initial.clarification.pending.task == "call the bank"
    assert initial.clarification.pending.date_local == date(2026, 10, 31)
    assert initial.clarification.pending.hour == 7
    assert initial.clarification.pending.minute == 44
    assert initial.clarification.pending.meridiem is None
    assert "7:44 AM tomorrow" in initial.clarification.question
    assert "7:44 PM tomorrow" in initial.clarification.question

    completed = interpreter.interpret(
        "7:44 PM",
        timezone_name="UTC",
        now=NOW,
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 10, 31, 19, 44, tzinfo=UTC)


def test_also_remind_me_with_ambiguous_hour_asks_for_meridiem_deterministically() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Also remind me tomorrow at 2 to email the client",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.fallback_allowed is True
    assert result.clarification is not None
    assert result.clarification.pending.task == "email the client"
    assert result.clarification.pending.date_local == date(2026, 10, 31)
    assert result.clarification.pending.hour == 2
    assert result.clarification.pending.meridiem is None
    assert "2 AM tomorrow" in result.clarification.question
    assert "2 PM tomorrow" in result.clarification.question


def test_provider_fallback_cannot_replace_deterministic_ambiguity_with_generic_error() -> None:
    class UnsupportedProvider:
        def interpret(self, *args, **kwargs):
            return Interpretation(
                response_text="Please tell me the reminder task and a future time."
            )

    result = ReminderIntentInterpreter(UnsupportedProvider()).interpret(
        "Also remind me tomorrow at 2 to email the client",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "email the client"
    assert "2 AM tomorrow" in result.clarification.question


def test_ambiguous_time_on_an_expired_explicit_date_asks_only_for_a_new_date() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me Friday at 12 to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.date_local is None
    assert result.clarification.pending.hour == 12
    assert result.clarification.pending.minute == 0
    assert "future date" in result.clarification.question
    assert "AM" not in result.clarification.question
    assert "PM" not in result.clarification.question


def test_weekday_without_hour_is_clarified_and_keeps_task() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me Friday to renew the domain",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is not None
    assert result.clarification.pending.task == "renew the domain"
    assert result.clarification.pending.date_local is not None
    assert "time" in result.clarification.question.lower()


def test_clarification_follow_up_merges_once_complete() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call the bank at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=UTC),
    )

    assert initial.clarification is not None
    completed = interpreter.interpret(
        "7 PM",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=UTC),
        context=initial.clarification.pending,
    )

    assert completed.clarification is None
    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 10, 30, 19, 0, tzinfo=UTC)


def test_unrelated_follow_up_does_not_fill_missing_task() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=UTC),
    )

    assert initial.clarification is not None
    follow_up = interpreter.interpret(
        "thanks",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=UTC),
        context=initial.clarification.pending,
    )

    assert follow_up.intent is None
    assert follow_up.clarification is not None
    assert follow_up.clarification.pending.task is None


def test_follow_up_cancellation_clears_context_without_creating_reminder() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call the bank at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=UTC),
    )

    assert initial.clarification is not None
    canceled = interpreter.interpret(
        "Never mind.",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=UTC),
        context=initial.clarification.pending,
    )

    assert canceled.intent is None
    assert canceled.clarification is None
    assert canceled.clear_context is True
    assert "won't create" in canceled.response_text


def test_pending_time_resolution_uses_the_context_timezone() -> None:
    pending = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 31),
        hour=9,
        minute=30,
        meridiem="am",
        timezone_name="Asia/Karachi",
        date_explicit=True,
    )

    assert resolve_pending_due_at(pending, now=NOW) == datetime(2026, 10, 31, 4, 30, tzinfo=UTC)


def test_provider_updates_cannot_change_unmentioned_pending_fields() -> None:
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 31),
        hour=7,
        minute=0,
        meridiem="pm",
        timezone_name="UTC",
        date_explicit=True,
    )
    verified = restrict_to_explicit_field_updates(
        "8 AM",
        PendingReminderFieldUpdate(
            task="book a holiday",
            date_local=date(2026, 11, 2),
            hour=8,
            minute=0,
            meridiem="am",
        ),
        context=context,
        now=NOW,
    )

    assert verified.model_dump(exclude_none=True) == {
        "hour": 8,
        "minute": 0,
        "meridiem": "am",
    }


def test_provider_task_update_requires_an_explicit_correction() -> None:
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 31),
        hour=7,
        timezone_name="UTC",
        date_explicit=True,
    )
    verified = restrict_to_explicit_field_updates(
        "Change the reminder to call the doctor at 8 AM",
        PendingReminderFieldUpdate(task="call the doctor", hour=8, minute=0, meridiem="am"),
        context=context,
        now=NOW,
    )

    assert verified.task == "call the doctor"
    assert verified.hour == 8
    assert verified.meridiem == "am"


def test_standalone_meridiem_is_accepted_for_pending_time_clarification() -> None:
    context = PendingReminder(
        task="visit the dentist",
        date_local=date(2026, 10, 31),
        hour=8,
        timezone_name="UTC",
    )
    verified = restrict_to_explicit_field_updates(
        "PM",
        PendingReminderFieldUpdate(meridiem="PM"),
        context=context,
        now=NOW,
    )

    assert verified.meridiem == "pm"


def test_classifier_selected_update_can_replace_pending_task() -> None:
    context = PendingReminder(
        task="email Hassan",
        date_local=date(2026, 10, 31),
        hour=8,
        meridiem="pm",
        timezone_name="UTC",
    )
    verified = restrict_to_explicit_field_updates(
        "email Bilal",
        PendingReminderFieldUpdate(
            task="email Bilal", certainty="CLEAR", evidence={"task": "email Bilal"}
        ),
        context=context,
        now=NOW,
        allow_task_replacement=True,
        allow_semantic_field_updates=True,
    )

    assert verified.task == "email Bilal"


@pytest.mark.parametrize(
    ("message", "date_evidence"),
    [
        ("tomorow at 8 PM", "tomorrow"),
        ("tmorrow at 8 PM", "tomorrow"),
        ("at 8 PM tommorow", "tomorrow at 8 PM"),
    ],
)
def test_semantic_update_accepts_grounded_typo_or_word_order(
    message: str, date_evidence: str
) -> None:
    context = PendingReminder(
        task="call Ayesha",
        timezone_name="UTC",
    )
    verified = restrict_to_explicit_field_updates(
        message,
        PendingReminderFieldUpdate(
            date_local=date(2026, 10, 31),
            hour=8,
            minute=0,
            meridiem="PM",
            certainty="CLEAR",
            evidence={
                "date_local": date_evidence,
                "hour": date_evidence,
                "minute": date_evidence,
                "meridiem": date_evidence,
            },
        ),
        context=context,
        now=NOW,
        allow_semantic_field_updates=True,
    )

    assert verified.date_local == date(2026, 10, 31)
    assert verified.hour == 8
    assert verified.meridiem == "pm"


def test_semantic_update_does_not_accept_typo_without_classifier_authorization() -> None:
    context = PendingReminder(task="call Ayesha", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "tomorow",
        PendingReminderFieldUpdate(date_local=date(2026, 10, 31)),
        context=context,
        now=NOW,
    )

    assert verified.date_local is None


def test_semantic_update_requires_clear_grounded_evidence() -> None:
    context = PendingReminder(task="call Ayesha", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "tomorow",
        PendingReminderFieldUpdate(
            date_local=date(2026, 10, 31), certainty="CLEAR", evidence={"date_local": "today"}
        ),
        context=context,
        now=NOW,
        allow_semantic_field_updates=True,
    )

    assert verified.date_local is None


def test_semantic_update_rejects_a_different_temporal_word_as_evidence() -> None:
    context = PendingReminder(task="call Ayesha", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "Sunday",
        PendingReminderFieldUpdate(
            date_local=date(2026, 11, 2), certainty="CLEAR", evidence={"date_local": "Monday"}
        ),
        context=context,
        now=NOW,
        allow_semantic_field_updates=True,
    )

    assert verified.date_local is None


def test_semantic_task_update_accepts_a_grounded_spelling_variation() -> None:
    context = PendingReminder(task="email Hassan", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "emial Bilal",
        PendingReminderFieldUpdate(
            task="email Bilal", certainty="CLEAR", evidence={"task": "email Bilal"}
        ),
        context=context,
        now=NOW,
        allow_task_replacement=True,
        allow_semantic_field_updates=True,
    )

    assert verified.task == "email Bilal"


def test_clarification_question_mentions_only_the_missing_date() -> None:
    pending = PendingReminder(
        task="prepare slides",
        hour=4,
        meridiem="pm",
        timezone_name="UTC",
    )

    result = RuleBasedIntentInterpreter().apply_field_updates(
        pending, PendingReminderFieldUpdate(), now=NOW
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "prepare slides"
    assert result.clarification.pending.hour == 4
    assert result.clarification.pending.meridiem == "pm"
    assert result.clarification.pending.date_local is None
    assert result.clarification.question == "What date should I use?"


def test_complex_follow_up_uses_model_fallback_with_pending_context() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.context = None

        def interpret(self, text, *, timezone_name, now, context=None):
            self.context = context
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        task="call the bank",
                        date_local=date(2026, 10, 30),
                        hour=7,
                        minute=0,
                        meridiem="pm",
                        timezone_name=timezone_name,
                    ),
                    question="What time should I use?",
                )
            )

    model = RecordingModel()
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 30),
        hour=7,
        minute=0,
        timezone_name="UTC",
    )
    result = ReminderIntentInterpreter(model).interpret(
        "Could you make sure this happens after dinner?",
        timezone_name="UTC",
        now=NOW,
        context=context,
    )

    assert model.context == context
    assert result.clarification is not None


def test_complex_natural_language_without_reminder_keyword_uses_model_fallback() -> None:
    class RecordingModel:
        called = False

        def interpret(self, text, *, timezone_name, now, context=None):
            self.called = True
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="call John",
                    due_at=datetime(2026, 10, 30, 19, 0, tzinfo=UTC),
                    timezone_name=timezone_name,
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Could you make sure I call John around dinner time tonight?",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert result.intent is not None


def test_multiple_people_in_one_task_are_not_split_into_multiple_reminders() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice and Bob tomorrow at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is None
    assert result.intent is not None
    assert result.intent.task == "call Alice and Bob"


def test_multiple_tasks_with_one_schedule_request_clarification() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice and email Bob tomorrow at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is not None


def test_multiple_tasks_with_one_schedule_are_not_merged() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Ali tomorrow at 9 AM and to pay the bill",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is not None


def test_explicit_multiple_reminder_requests_are_parsed_independently() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice tomorrow at 9 AM and remind me to email Bob tomorrow at 10 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is None
    assert result.intent is None
    assert len(result.resolved_intents) == 2
    assert [intent.task for intent in result.resolved_intents] == [
        "call Alice",
        "email Bob",
    ]
    assert [intent.due_at for intent in result.resolved_intents] == [
        datetime(2026, 10, 31, 9, 0, tzinfo=UTC),
        datetime(2026, 10, 31, 10, 0, tzinfo=UTC),
    ]


def test_shared_date_multiple_times_are_parsed_independently() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow at 9 AM to pay rent and at 6 PM to call Sara",
        timezone_name="UTC",
        now=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
    )

    assert [intent.task for intent in result.resolved_intents] == [
        "pay rent",
        "call Sara",
    ]
    assert [intent.due_at for intent in result.resolved_intents] == [
        datetime(2026, 11, 1, 9, 0, tzinfo=UTC),
        datetime(2026, 11, 1, 18, 0, tzinfo=UTC),
    ]


def test_compound_single_prefix_uses_model_fallback() -> None:
    class RecordingModel:
        called = False

        def interpret(self, text, *, timezone_name, now, context=None):
            self.called = True
            return Interpretation(
                intents=(
                    CreateOneTimeReminderIntent(
                        task="call Alice",
                        due_at=datetime(2026, 10, 31, 9, 0, tzinfo=UTC),
                        timezone_name=timezone_name,
                    ),
                    CreateOneTimeReminderIntent(
                        task="email Bob",
                        due_at=datetime(2026, 10, 31, 10, 0, tzinfo=UTC),
                        timezone_name=timezone_name,
                    ),
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Remind me to call Alice tomorrow at 9 AM and email Bob tomorrow at 10 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert [intent.task for intent in result.resolved_intents] == ["call Alice", "email Bob"]


def test_single_schedule_multi_target_command_is_clarified() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call Alice and email Bob tomorrow at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert initial.clarification is not None
    assert initial.intent is None


def test_follow_up_accepts_minute_and_day_phrasing() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call John at 6:55",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 15, 0, tzinfo=UTC),
    )

    assert initial.clarification is not None
    completed = interpreter.interpret(
        "6:55 pm today",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 15, 0, tzinfo=UTC),
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.due_at == datetime(2026, 10, 29, 18, 55, tzinfo=UTC)


def test_bare_time_does_not_start_a_provider_clarification_session() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "7 pm",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is None
    assert result.fallback_allowed is False


def test_friday_follow_up_keeps_the_explicit_future_date() -> None:
    interpreter = RuleBasedIntentInterpreter()
    thursday = datetime(2026, 10, 29, 13, 0, tzinfo=UTC)
    initial = interpreter.interpret(
        "Remind me Friday to renew the domain",
        timezone_name="UTC",
        now=thursday,
    )

    assert initial.clarification is not None
    completed = interpreter.interpret(
        "8 a.m.",
        timezone_name="UTC",
        now=thursday,
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.due_at == datetime(2026, 10, 30, 8, 0, tzinfo=UTC)


def test_past_follow_up_keeps_task_and_asks_for_a_new_future_time() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to stretch at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 15, 0, tzinfo=UTC),
    )

    assert initial.clarification is not None
    past = interpreter.interpret(
        "7 pm",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 19, 2, tzinfo=UTC),
        context=initial.clarification.pending,
    )

    assert past.intent is None
    assert past.clarification is not None
    assert past.clarification.pending.task == "stretch"
    assert past.clarification.pending.date_local == date(2026, 10, 29)
    assert past.clarification.pending.hour is None
    assert "passed" in past.clarification.question

    completed = interpreter.interpret(
        "8 pm",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 19, 2, tzinfo=UTC),
        context=past.clarification.pending,
    )
    assert completed.intent is not None
    assert completed.intent.due_at == datetime(2026, 10, 29, 20, 0, tzinfo=UTC)


def test_explicit_past_date_keeps_task_and_time_and_asks_only_for_date() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me Friday at 8 AM to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert initial.intent is None
    assert initial.clarification is not None
    assert initial.clarification.pending.task == "call the bank"
    assert initial.clarification.pending.date_local is None
    assert initial.clarification.pending.hour == 8
    assert initial.clarification.pending.meridiem == "am"
    assert "future date" in initial.clarification.question

    completed = interpreter.interpret(
        "Monday",
        timezone_name="UTC",
        now=NOW,
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 11, 2, 8, 0, tzinfo=UTC)


def test_pydantic_intent_rejects_naive_due_time() -> None:
    try:
        CreateOneTimeReminderIntent(
            task="call",
            due_at="2026-10-30T12:05:00",
            timezone_name="UTC",
        )
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:  # pragma: no cover - assertion makes the intended validation explicit
        raise AssertionError("naive due time was accepted")


def test_openai_compatible_interpreter_accepts_structured_response(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        assert args[0] == "https://llm.example/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        assert kwargs["json"]["model"] == "provider-model"
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"CREATE_ONE_TIME_REMINDER",'
                                '"task":"call the bank","due_local":"2026-10-31T09:00:00"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1",
        "secret",
        "provider-model",
    ).interpret(
        "Please remind me to call the bank tomorrow morning",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call the bank"
    assert result.intent.due_at == datetime(2026, 10, 31, 4, 0, tzinfo=UTC)


def test_openai_compatible_interpreter_normalizes_keyed_recurring_response(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"CREATE_RECURRING_REMINDER":{'
                                '"task":"call parents","frequency":"DAILY",'
                                '"interval":1,"hour":14,"minute":10,'
                                '"start_local":"2026-11-04T14:10"}}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "Every day, remind me at 2:10 pm to call parents",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.task == "call parents"
    assert result.intent.frequency == "WEEKLY"
    assert result.intent.weekdays == (0, 1, 2, 3, 4, 5, 6)
    assert result.intent.hour == 14
    assert result.intent.minute == 10


def test_openai_compatible_interpreter_accepts_structured_reschedule_response(
    monkeypatch,
) -> None:
    def fake_post(*args, **kwargs):
        prompt = kwargs["json"]["messages"][0]["content"]
        assert "RESCHEDULE_REMINDER" in prompt
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"RESCHEDULE_REMINDER",'
                                '"target_text":"bank reminder",'
                                '"due_local":"2026-10-31T10:00:00",'
                                '"scope":"OCCURRENCE"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "I need you to shift the bank reminder to tomorrow at 10 AM",
        selected_route=IntentRoute.RESCHEDULE_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.reschedule_request, RescheduleRequest)
    assert result.reschedule_request.target_text == "bank reminder"
    assert result.reschedule_request.due_at == datetime(2026, 10, 31, 10, 0, tzinfo=UTC)
    assert result.reschedule_request.scope is RescheduleScope.OCCURRENCE


def test_openai_compatible_interpreter_normalizes_typed_reschedule_clarification(
    monkeypatch,
) -> None:
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"type":"NEEDS_CLARIFICATION",'
                                '"question":"What time should I use?",'
                                '"pending_fields":["time"]}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "Reschedule the dentist reminder for tomorrow",
        selected_route=IntentRoute.RESCHEDULE_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.response_text == "What time should I use?"


def test_reschedule_extractor_uses_structured_list_context_for_typed_clarification(
    monkeypatch,
) -> None:
    captured_prompt: dict[str, str] = {}

    def fake_post(*args, **kwargs):
        captured_prompt["value"] = kwargs["json"]["messages"][0]["content"]
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"NEEDS_CLARIFICATION",'
                                '"target_text":"submit project report",'
                                '"destination_date":"2026-10-31",'
                                '"date_explicit":true,'
                                '"question":"What time should I use tomorrow?"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    interpreter = ReminderIntentInterpreter(
        OpenAICompatibleIntentInterpreter("https://llm.example/v1", "secret", "provider-model")
    )

    proposal = interpreter.interpret_reschedule(
        "Actually, move that one to tomorrow",
        timezone_name="UTC",
        now=NOW,
        selection_context=(
            {
                "reminder_id": "00000000-0000-0000-0000-000000000001",
                "task": "submit project report",
                "due_at": "2026-10-30T19:47:00+00:00",
            },
        ),
    )

    assert "most recent reminder list shown to the user" in captured_prompt["value"]
    assert isinstance(proposal, RescheduleClarification)
    assert proposal.request.target_text == "submit project report"
    assert proposal.request.destination_date == date(2026, 10, 31)
    assert proposal.question == "What time should I use tomorrow?"


def test_openai_compatible_interpreter_returns_only_provider_field_updates_for_context(
    monkeypatch,
) -> None:
    captured_prompt = {}

    def fake_post(*args, **kwargs):
        captured_prompt["value"] = kwargs["json"]["messages"][0]["content"]
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"UPDATE_PENDING_REMINDER",'
                                '"updates":{"meridiem":"PM"}}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx,
        "post",
        fake_post,
    )
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 30),
        hour=7,
        timezone_name="UTC",
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("in the evening", timezone_name="UTC", now=NOW, context=context)

    assert result.intent is None
    assert result.clarification is None
    assert result.field_updates == PendingReminderFieldUpdate(meridiem="pm")
    assert "field values are primitive values directly" in captured_prompt["value"]
    assert (
        "Do not wrap any updates field in a value/certainty/evidence object"
        in captured_prompt["value"]
    )
    assert '"evidence":{"task":"pay rent"' in captured_prompt["value"]

    # The service-side state manager, not the provider, combines this patch
    # with the pending task/date/time and only then validates completion.
    completed = RuleBasedIntentInterpreter().apply_field_updates(
        context, result.field_updates, now=NOW
    )
    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 10, 30, 19, 0, tzinfo=UTC)


def test_llm_recurrence_update_fields_are_verified_and_merged() -> None:
    context = PendingReminder(
        task="pay bills",
        date_local=date(2026, 11, 2),
        hour=5,
        meridiem="pm",
        timezone_name="Asia/Karachi",
    )
    proposed = PendingReminderFieldUpdate(
        recurrence_frequency="MONTHLY",
        recurrence_day_of_month=2,
        certainty="CLEAR",
        evidence={
            "recurrence_frequency": "monthly",
            "recurrence_day_of_month": "2nd",
        },
    )

    verified = restrict_to_explicit_field_updates(
        "make it monthly on the 2nd",
        proposed,
        context=context,
        now=NOW,
        allow_semantic_field_updates=True,
    )
    completed = RuleBasedIntentInterpreter().apply_field_updates(context, verified, now=NOW)

    assert completed.intent is not None
    assert isinstance(completed.intent, CreateRecurringReminderIntent)
    assert completed.intent.frequency == "MONTHLY"
    assert completed.intent.day_of_month == 2


def test_llm_recurring_extraction_derives_start_for_every_three_months(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"CREATE_RECURRING_REMINDER",'
                                '"task":"review budget","frequency":"MONTHLY",'
                                '"interval":3,"day_of_month":15,"hour":8,"minute":0}'
                            )
                        }
                    }
                ]
            },
        ),
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "remind me every three months on the 15th at 8 AM to review budget",
        timezone_name="Asia/Karachi",
        now=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )

    assert isinstance(result.intent, CreateRecurringReminderIntent)
    assert result.intent.interval == 3
    assert result.intent.start_local == datetime(2027, 1, 15, 8)


def test_openai_compatible_interpreter_preserves_pending_on_ambiguous_update(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"NEEDS_CLARIFICATION",'
                                '"question":"Which Friday do you mean?"}'
                            )
                        }
                    }
                ]
            },
        ),
    )
    context = PendingReminder(
        task="call Ayesha",
        date_local=date(2026, 10, 31),
        hour=8,
        meridiem="pm",
        timezone_name="UTC",
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("next Friday", timezone_name="UTC", now=NOW, context=context)

    assert result.field_updates is None
    assert result.clarification is not None
    assert result.clarification.question == "Which Friday do you mean?"
    assert result.clarification.pending == context


def test_openai_compatible_interpreter_rejects_completed_reminder_during_clarification(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"CREATE_ONE_TIME_REMINDER",'
                                '"task":"wrong task",'
                                '"due_local":"2026-10-31T08:00:00"}'
                            )
                        }
                    }
                ]
            },
        ),
    )
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 30),
        hour=7,
        timezone_name="UTC",
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("8 AM", timezone_name="UTC", now=NOW, context=context)

    assert result.intent is None
    assert result.field_updates is None


def test_openai_compatible_interpreter_rejects_offset_bearing_local_time(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"CREATE_ONE_TIME_REMINDER",'
                                '"task":"call the bank",'
                                '"due_local":"2026-10-31T09:00:00+00:00"}'
                            )
                        }
                    }
                ]
            },
        ),
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("remind me", timezone_name="Asia/Karachi", now=NOW)

    assert result.intent is None
    assert result.response_text


def test_openai_compatible_interpreter_rejects_unknown_operation(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
            json={"choices": [{"message": {"content": '{"operation":"DELETE_ALL"}'}}]},
        ),
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("delete my reminders", timezone_name="UTC", now=NOW)

    assert result.intent is None
    assert result.response_text == OUT_OF_SCOPE_RESPONSE
