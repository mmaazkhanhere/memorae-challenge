from reminder_agent.features import intent
from reminder_agent.features.intent import contracts, pipeline
from reminder_agent.features.intent.deterministic import interpreter as deterministic_interpreter
from reminder_agent.features.intent.model import interpreter as model_interpreter
from reminder_agent.features.intent.model import router as model_router
from reminder_agent.features.intent.pending import reminder_creation as pending_reminders
from reminder_agent.features.memory import service as memories
from reminder_agent.features.memory.retrieve.answering import OpenAICompatibleMemoryAnswerer


def test_intent_feature_package_exports_moved_object_identity() -> None:
    assert intent.OpenAICompatibleMemoryAnswerer is OpenAICompatibleMemoryAnswerer
    assert intent.IntentRoute is contracts.IntentRoute
    assert intent.IntentRouter is model_router.IntentRouter
    assert intent.RuleBasedIntentInterpreter is deterministic_interpreter.RuleBasedIntentInterpreter
    assert intent.ReminderIntentInterpreter is pipeline.ReminderIntentInterpreter
    assert intent.OpenAICompatibleIntentInterpreter is (
        model_interpreter.OpenAICompatibleIntentInterpreter
    )
    assert intent.restrict_to_explicit_field_updates is (
        pending_reminders.restrict_to_explicit_field_updates
    )
    assert intent.MemoryIntentEvidence is memories.MemoryIntentEvidence
    assert intent.RetrieveMemoryQuery is memories.RetrieveMemoryQuery
    assert intent.SaveMemoryFact is memories.SaveMemoryFact
    assert intent.extract_memory_content is memories.extract_memory_content
