"""Focused regression tests for the deterministic-first pipeline boundary."""

from datetime import datetime, timezone

from reminder_agent.conversation_router import ConversationRoute
from reminder_agent.intent import (
    CreateOneTimeReminderIntent,
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    NeedsClarification,
    PendingReminder,
    ReminderIntentInterpreter,
)
from reminder_agent.reminders import _route_and_interpret
from reminder_agent.request_policy import OUT_OF_SCOPE_RESPONSE


NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def test_explicit_simple_request_stays_on_deterministic_route_without_classifier() -> None:
    class FailingClassifier:
        def classify(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("a clear deterministic route must not call the classifier")

    class RecordingInterpreter:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(response_text="deterministic result")

    interpreter = RecordingInterpreter()
    route, result = _route_and_interpret(
        interpreter,
        "Remind me tomorrow at 9 AM to call the bank",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=FailingClassifier(),
    )

    assert route is ConversationRoute.NEW_REQUEST
    assert interpreter.called is True
    assert result.response_text == "deterministic result"


def test_unmatched_route_uses_classifier_and_uncertainty_stops_before_extraction() -> None:
    class UncertainClassifier:
        called = False

        def classify(self, *args, **kwargs):
            self.called = True
            return IntentRouteSuggestion(operation=IntentRoute.UNCERTAIN)

    class FailingInterpreter:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("uncertain intent must clarify before parameter extraction")

    classifier = UncertainClassifier()
    route, result = _route_and_interpret(
        FailingInterpreter(),
        "Could you help organize my errands for next weekend?",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=classifier,
    )

    assert classifier.called is True
    assert route is ConversationRoute.UNRELATED
    assert "not sure" in (result.response_text or "").lower()
    assert result.fallback_allowed is False


def test_classifier_derived_unrelated_request_uses_scope_response() -> None:
    class UnrelatedClassifier:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.UNRELATED)

    class FailingInterpreter:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("unsupported intent must not reach extraction")

    route, result = _route_and_interpret(
        FailingInterpreter(),
        "Please organize my errands for next weekend",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=UnrelatedClassifier(),
    )

    assert route is ConversationRoute.UNRELATED
    assert result.response_text == OUT_OF_SCOPE_RESPONSE
    assert result.fallback_allowed is False


def test_complete_parameters_do_not_invoke_extractor_fallback() -> None:
    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("complete deterministic parameters must not call the extractor")

    result = ReminderIntentInterpreter(FailingExtractor()).interpret(
        "Remind me tomorrow at 9 AM to call the bank",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert isinstance(result.intent, CreateOneTimeReminderIntent)
    assert result.intent.task == "call the bank"


def test_incomplete_parameters_invoke_extractor_only_after_intent_is_resolved() -> None:
    class RecordingExtractor:
        called = False
        selected_route = None

        def interpret(self, text, *, selected_route, **kwargs):
            self.called = True
            self.selected_route = selected_route
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(task="call the bank", timezone_name="UTC"),
                    question="When should I remind you to call the bank?",
                )
            )

    extractor = RecordingExtractor()
    result = ReminderIntentInterpreter(extractor).interpret(
        "Remind me to call the bank",
        selected_route=IntentRoute.NEW_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert extractor.called is True
    assert extractor.selected_route is IntentRoute.NEW_REQUEST
    assert result.clarification is not None
    assert result.clarification.question == "When should I remind you to call the bank?"
