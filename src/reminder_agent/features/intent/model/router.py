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
from reminder_agent.features.intent.model.classifier_prompts import get_classifier_prompt
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
        prompt_version: str = "v1",
    ) -> None:
        self._client = client or OpenAICompatibleLLMClient(base_url, api_key, timeout_seconds)
        self._url = self._client.url
        self._model = model
        self._prompt_version = prompt_version
        get_classifier_prompt(prompt_version)

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
            prompt_version=self._prompt_version,
            has_pending=pending is not None,
            recent_message_count=len(relevant_history),
        )
        prompt = get_classifier_prompt(self._prompt_version)
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
