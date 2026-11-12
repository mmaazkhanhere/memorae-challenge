"""Focused regression tests for the deterministic-first pipeline boundary."""

from datetime import UTC, datetime

import pytest

from reminder_agent.application.conversation_coordinator import _route_and_interpret
from reminder_agent.features.intent import (
    CreateOneTimeReminderIntent,
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    NeedsClarification,
    PendingReminder,
    ReminderIntentInterpreter,
)
from reminder_agent.features.intent.deterministic.routing import ConversationRoute
from reminder_agent.features.intent.policy import OUT_OF_SCOPE_RESPONSE
from reminder_agent.shared.memory_contracts import MemoryIntentEvidence, SaveMemoryFact

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


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


def test_classifier_can_route_an_implicit_reschedule_without_pending_context() -> None:
    class RescheduleClassifier:
        called = False

        def classify(self, text, **kwargs):
            self.called = True
            assert text == "Actually move it to 9 PM today"
            assert kwargs["pending"] is None
            return IntentRouteSuggestion(operation=IntentRoute.RESCHEDULE_REQUEST)

    class FailingInterpreter:
        def interpret(self, *args, **kwargs):  # pragma: no cover - route must bypass extraction
            raise AssertionError("reschedule routing must bypass the generic reminder extractor")

    classifier = RescheduleClassifier()
    route, result = _route_and_interpret(
        FailingInterpreter(),
        "Actually move it to 9 PM today",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=classifier,
    )

    assert classifier.called is True
    assert route is ConversationRoute.RESCHEDULE_REQUEST
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


def test_classifier_selected_memory_reaches_existing_extractor_as_save_route() -> None:
    class MemoryRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(
                operation=IntentRoute.SAVE_MEMORY,
                memory_intent_evidence=MemoryIntentEvidence(
                    kind="EXPLICIT_SAVE_REQUEST",
                    confidence="CLEAR",
                    evidence_text="Make a note",
                ),
            )

    class MemoryExtractor:
        selected_route = None

        def interpret(self, text, *, selected_route, **kwargs):
            self.selected_route = selected_route
            return Interpretation(memory_fact=SaveMemoryFact(content="gate code is 4512"))

    extractor = MemoryExtractor()
    route, result = _route_and_interpret(
        extractor,
        "Make a note that my gate code is 4512",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=MemoryRouter(),
    )

    assert route is ConversationRoute.SAVE_MEMORY_REQUEST
    assert extractor.selected_route is IntentRoute.SAVE_MEMORY
    assert result.memory_fact is not None


def test_classifier_memory_route_without_save_language_does_not_extract() -> None:
    class OvereagerRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.SAVE_MEMORY)

    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("ordinary statements must not reach memory extraction")

    route, result = _route_and_interpret(
        FailingExtractor(),
        "My gate code is 4512",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=OvereagerRouter(),
    )

    assert route is ConversationRoute.UNRELATED
    assert result.memory_fact is None
    assert result.fallback_allowed is False


def test_classifier_evidence_supports_multilingual_save_intent_without_english_rules() -> None:
    text = "Por favor, recuerda que mi perro se llama Bruno"

    class MultilingualRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(
                operation=IntentRoute.SAVE_MEMORY,
                memory_intent_evidence=MemoryIntentEvidence(
                    kind="EXPLICIT_SAVE_REQUEST",
                    confidence="CLEAR",
                    evidence_text="recuerda que",
                ),
            )

    class RecordingExtractor:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(memory_fact=SaveMemoryFact(content="mi perro se llama Bruno"))

    extractor = RecordingExtractor()
    route, result = _route_and_interpret(
        extractor,
        text,
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=MultilingualRouter(),
    )

    assert route is ConversationRoute.SAVE_MEMORY_REQUEST
    assert extractor.called is True
    assert result.memory_fact is not None


@pytest.mark.parametrize(
    "kind",
    ["QUOTED_INSTRUCTION", "NEGATED_SAVE_REQUEST", "REMINDER_REQUEST", "UNCERTAIN"],
)
def test_adversarial_memory_evidence_does_not_reach_extractor(kind: str) -> None:
    class AdversarialRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(
                operation=IntentRoute.SAVE_MEMORY,
                memory_intent_evidence=MemoryIntentEvidence(
                    kind=kind,
                    confidence="CLEAR",
                    evidence_text="remember this",
                ),
            )

    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("non-save evidence must not reach extraction")

    route, result = _route_and_interpret(
        FailingExtractor(),
        "This is a quoted instruction: remember this, but do not save it.",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=AdversarialRouter(),
    )

    assert route is ConversationRoute.UNRELATED
    assert result.memory_fact is None
