"""Model-backed classification for ambiguous intent routes."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Protocol

import httpx
from pydantic import ValidationError

from reminder_agent.features.intent.context import CONVERSATION_CONTEXT_TURNS
from reminder_agent.features.intent.contracts import (
    IntentRoute,
    IntentRouteSuggestion,
    PendingReminder,
)
from reminder_agent.infrastructure.observability import exception_details, get_logger, log_event
from reminder_agent.infrastructure.providers.llm import (
    OpenAICompatibleLLMClient,
    StructuredIntentProvider,
)

logger = get_logger()


def _missing_pending_fields(pending: PendingReminder | None) -> list[str]:
    if pending is None:
        return []
    missing: list[str] = []
    if pending.task is None:
        missing.append("task")
    if pending.date_local is None:
        missing.append("date")
    if pending.hour is None:
        missing.append("time")
    elif pending.hour <= 12 and pending.meridiem is None:
        missing.append("meridiem")
    return missing


def _last_clarification_question(recent_messages: Sequence[str]) -> str | None:
    for message in reversed(recent_messages):
        if message.startswith("assistant: ") and "?" in message:
            return message.removeprefix("assistant: ")
    return None


class IntentRouter(Protocol):
    def classify(
        self,
        text: str,
        *,
        pending: PendingReminder | None,
        recent_messages: Sequence[str],
        recent_delivered_reminders: Sequence[dict[str, str]] = (),
        selection_context: Sequence[dict[str, str]] = (),
    ) -> IntentRouteSuggestion: ...


class OpenAICompatibleIntentRouter:
    """Classify an ambiguous turn without extracting or modifying reminder data."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 10.0,
        *,
        client: StructuredIntentProvider | None = None,
    ) -> None:
        self._client = client or OpenAICompatibleLLMClient(base_url, api_key, timeout_seconds)
        self._url = self._client.url
        self._model = model

    def classify(
        self,
        text: str,
        *,
        pending: PendingReminder | None,
        recent_messages: Sequence[str],
        recent_delivered_reminders: Sequence[dict[str, str]] = (),
        selection_context: Sequence[dict[str, str]] = (),
    ) -> IntentRouteSuggestion:
        relevant_history = list(recent_messages)[-CONVERSATION_CONTEXT_TURNS:]
        log_event(
            logger,
            logging.INFO,
            "intent_classification_started",
            model=self._model,
            has_pending=pending is not None,
            recent_message_count=len(relevant_history),
        )
        prompt = """
                You are Memorae's intent classifier. Classify the latest message into exactly one operation using the message,
                relevant conversation history, pending state, and trusted recently delivered reminders.

                Supported operations:

                - NEW_REQUEST: Create a one-time or recurring reminder, including "remember to [action]".
                - LIST_REQUEST: List scheduled reminders.
                - SUMMARY_REQUEST: Count or summarize reminder progress.
                - RESCHEDULE_REQUEST: Change an existing reminder's schedule.
                - CANCEL_REQUEST: Cancel an existing reminder or series.
                - COMPLETE_REQUEST: Complete an already-delivered reminder occurrence.
                - SNOOZE_REQUEST: Postpone an already-delivered occurrence.
                - SAVE_MEMORY: Explicitly request saving a personal fact. Never save ordinary statements, quoted instructions, or negated requests.
                - RETRIEVE_MEMORY: Ask about previously saved personal information, including indirect or situational questions without "my", "remember",
                or "saved". Do not require proof that a matching fact exists.
                - UPDATE_PENDING: Clarify or modify a pending reminder.
                - CANCEL_PENDING: Cancel a pending reminder.
                - SOCIAL: Greetings or casual conversation.
                - UNRELATED: General knowledge, advice, or requests outside reminder and memory functionality.
                - UNCERTAIN: Intent is ambiguous or insufficiently supported.

                Rules:

                1. Classify by intended operation, not keywords.
                2. Reminder queries are not memory queries.
                3. Use conversation context to distinguish indirect personal-memory questions from general advice.
                4. "Who should I contact in an emergency?" ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ RETRIEVE_MEMORY when context indicates a saved emergency contact; otherwise UNCERTAIN.
                5. "How do I choose an emergency contact?" ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ UNRELATED.
                6. "Remember to call Ahmed tomorrow" ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ NEW_REQUEST.
                7. "Remember my emergency contact is Ahmed" ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ SAVE_MEMORY.
                8. Missing memory facts are handled by retrieval, not classification.
                9. Personal date/time questions about a saved personal fact are RETRIEVE_MEMORY;
                   do not classify them as LIST_REQUEST merely because they use "when".
                10. Classify the operation independently of target resolution. If the latest message clearly
                   asks to move, reschedule, shift, delay, or change an existing reminder, classify it as
                   RESCHEDULE_REQUEST even when the target is an indirect reference such as "it", "that",
                   or "the previous one", and even when the new time is incomplete. Target resolution and
                   parameter extraction happen in a later validated stage.
                11. Use pending and delivered-reminder context only to distinguish conversational intent;
                    missing or ambiguous target details must not turn an otherwise clear operation into UNCERTAIN.
                12. Use UNCERTAIN only when no supported operation can be inferred from the message.
                13. Never extract parameters, answer questions, or execute actions.

                For SAVE_MEMORY, provide `memory_intent_evidence` with `kind: "EXPLICIT_SAVE_REQUEST"`, `confidence: "CLEAR"`, and an exact source span
                in `evidence_text`. For every other operation, set it to null.

                Return JSON only:

                {
                "operation": "<SUPPORTED_OPERATION>",
                "target_text": null,
                "memory_intent_evidence": null
                }

                Set `target_text` only when a specific reminder task is explicitly referenced. Do not invent details or add fields.
            """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "latest_message": text,
                            "active_pending": pending.model_dump(mode="json") if pending else None,
                            "missing_fields": _missing_pending_fields(pending),
                            "last_clarification_question": _last_clarification_question(
                                recent_messages
                            ),
                            "recent_messages": relevant_history,
                            "recent_delivered_reminders": list(recent_delivered_reminders),
                            "last_listed_reminders": list(selection_context),
                        }
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        log_event(
            logger,
            logging.INFO,
            "intent_classification_input",
            model=self._model,
            input_length=len(text.strip()),
            active_pending=bool(pending),
            recent_message_count=len(relevant_history),
            recent_delivered_reminder_count=len(recent_delivered_reminders),
        )
        try:
            _response_payload, content = self._client.complete(payload)
            data = json.loads(content) if isinstance(content, str) else content
            suggestion = IntentRouteSuggestion.model_validate(data)
            log_event(
                logger,
                logging.INFO,
                "intent_classification_completed",
                model=self._model,
                operation=suggestion.operation,
                target_text_present=suggestion.target_text is not None,
                provider_response={
                    "operation": data.get("operation"),
                    "target_text_present": data.get("target_text") is not None,
                    "memory_evidence_kind": (
                        data.get("memory_intent_evidence", {}).get("kind")
                        if isinstance(data.get("memory_intent_evidence"), dict)
                        else None
                    ),
                },
            )
            return suggestion
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            log_event(
                logger,
                logging.WARNING,
                "intent_router_failed",
                model=self._model,
                reason="provider_or_schema_error",
                **exception_details(exc),
            )
            return IntentRouteSuggestion(operation=IntentRoute.UNCERTAIN)
