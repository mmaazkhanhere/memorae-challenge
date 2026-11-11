"""Compatibility exports for the pre-modular intent API.

Runtime code should import contracts and interpreters from their feature-owned
modules. The aliases below deliberately preserve the old public identities.
"""

from __future__ import annotations

# Kept as a module-level provider handle so tests and integrations can patch
# ``reminder_agent.features.intent.httpx.post`` at the shared httpx boundary.
import httpx  # noqa: F401 - legacy monkeypatch/import compatibility

from reminder_agent.features.intent import contracts as _contracts
from reminder_agent.features.intent import pipeline as _pipeline
from reminder_agent.features.intent.deterministic import interpreter as _deterministic_interpreter
from reminder_agent.features.intent.model import interpreter as _structured_interpreter
from reminder_agent.features.intent.model import router as _model_routing
from reminder_agent.features.intent.pending import reminder_creation as _pending_reminders
from reminder_agent.features.memory.retrieve import answering as _memory_answering
from reminder_agent.infrastructure.providers import llm as _llm
from reminder_agent.shared import memory_contracts as _memory_contracts
from reminder_agent.shared import memory_text as _memory_text

OpenAICompatibleLLMClient = _llm.OpenAICompatibleLLMClient
StructuredIntentProvider = _llm.StructuredIntentProvider
MemoryAnswerProposal = _memory_answering.MemoryAnswerProposal
OpenAICompatibleMemoryAnswerer = _memory_answering.OpenAICompatibleMemoryAnswerer
_MEMORY_ANSWER_JSON_SCHEMA = _memory_answering._MEMORY_ANSWER_JSON_SCHEMA
_MEMORY_ANSWER_PROMPT = _memory_answering._MEMORY_ANSWER_PROMPT
ReminderIntentInterpreter = _pipeline.ReminderIntentInterpreter
MemoryIntentEvidence = _memory_contracts.MemoryIntentEvidence
RetrieveMemoryQuery = _memory_contracts.RetrieveMemoryQuery
SaveMemoryFact = _memory_contracts.SaveMemoryFact
extract_memory_content = _memory_text.extract_memory_content

CreateOneTimeReminderIntent = _contracts.CreateOneTimeReminderIntent
CreateRecurringReminderIntent = _contracts.CreateRecurringReminderIntent
IntentRoute = _contracts.IntentRoute
IntentRouteSuggestion = _contracts.IntentRouteSuggestion
Interpretation = _contracts.Interpretation
IntentInterpreter = _contracts.IntentInterpreter
ListRemindersIntent = _contracts.ListRemindersIntent
NeedsClarification = _contracts.NeedsClarification
PendingReminder = _contracts.PendingReminder
PendingReminderBatch = _contracts.PendingReminderBatch
PendingReminderFieldUpdate = _contracts.PendingReminderFieldUpdate
ReminderSummaryIntent = _contracts.ReminderSummaryIntent
normalize_task_query = _contracts.normalize_task_query
merge_pending_reminder_updates = _contracts.merge_pending_reminder_updates

IntentRouter = _model_routing.IntentRouter
OpenAICompatibleIntentRouter = _model_routing.OpenAICompatibleIntentRouter
OpenAICompatibleIntentInterpreter = _structured_interpreter.OpenAICompatibleIntentInterpreter
_supports_strict_json_schema = _structured_interpreter._supports_strict_json_schema

RuleBasedIntentInterpreter = _deterministic_interpreter.RuleBasedIntentInterpreter
extract_listing_task_query = _deterministic_interpreter.extract_listing_task_query
resolve_pending_due_at = _deterministic_interpreter.resolve_pending_due_at
_normalise_now = _deterministic_interpreter._normalise_now
_next_recurring_start_local = _deterministic_interpreter._next_recurring_start_local
restrict_to_explicit_field_updates = _pending_reminders.restrict_to_explicit_field_updates
