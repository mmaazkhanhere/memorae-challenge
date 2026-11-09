"""Structured LLM intent interpretation for the channel-independent pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from pydantic import ValidationError

from reminder_agent.features.intent import deterministic_interpreter as _deterministic_interpreter
from reminder_agent.features.intent import pending_updates as _pending_updates
from reminder_agent.features.intent.contracts import (
    IntentRoute,
    Interpretation,
    PendingReminder,
)
from reminder_agent.features.intent.structured_prompts import (
    CANCELLATION_JSON_SCHEMA,
    MEMORY_RETRIEVAL_EXTRACTION_PROMPT,
    MEMORY_SAVE_EXTRACTION_PROMPT,
    OPERATION_ENVELOPE_KEYS,
    build_prompt,
)
from reminder_agent.features.intent.structured_response import (
    decode_response,
    interpret_response,
)
from reminder_agent.infrastructure.llm import (
    OpenAICompatibleLLMClient,
    StructuredIntentProvider,
    supports_strict_json_schema,
)
from reminder_agent.infrastructure.observability import exception_details, get_logger, log_event

logger = get_logger()

__all__ = [
    "OpenAICompatibleIntentInterpreter",
    "RecurringFrequency",
    "SummaryMetric",
    "_CANCELLATION_JSON_SCHEMA",
    "_MEMORY_RETRIEVAL_EXTRACTION_PROMPT",
    "_MEMORY_SAVE_EXTRACTION_PROMPT",
    "_OPERATION_ENVELOPE_KEYS",
    "_normalise_now",
    "_next_recurring_start_local",
    "_supports_strict_json_schema",
    "extract_listing_task_query",
    "restrict_to_explicit_field_updates",
]

SummaryMetric = Literal[
    "PENDING_COUNT",
    "TODAY_REMAINING",
    "COMPLETED_COUNT",
    "TODAY_COMPLETED",
    "OVERDUE_COUNT",
]
RecurringFrequency = Literal["WEEKLY", "MONTHLY", "YEARLY"]

_normalise_now = _deterministic_interpreter._normalise_now
_next_recurring_start_local = _deterministic_interpreter._next_recurring_start_local
extract_listing_task_query = _deterministic_interpreter.extract_listing_task_query
restrict_to_explicit_field_updates = _pending_updates.restrict_to_explicit_field_updates

_supports_strict_json_schema = supports_strict_json_schema

_MEMORY_SAVE_EXTRACTION_PROMPT = MEMORY_SAVE_EXTRACTION_PROMPT
_MEMORY_RETRIEVAL_EXTRACTION_PROMPT = MEMORY_RETRIEVAL_EXTRACTION_PROMPT
_OPERATION_ENVELOPE_KEYS = OPERATION_ENVELOPE_KEYS
_CANCELLATION_JSON_SCHEMA = CANCELLATION_JSON_SCHEMA


class OpenAICompatibleIntentInterpreter:
    """Structured intent adapter for OpenAI-compatible chat-completion APIs."""

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

    def _response_format(self, route: IntentRoute) -> dict[str, Any]:
        """Use constrained cancellation output only where the provider supports it."""

        if route is IntentRoute.CANCEL_REQUEST and _supports_strict_json_schema(
            self._url, self._model
        ):
            return {"type": "json_schema", "json_schema": _CANCELLATION_JSON_SCHEMA}
        return {"type": "json_object"}

    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
        reschedule_candidates: Sequence[dict[str, str]] = (),
        cancellation_candidates: Sequence[dict[str, str]] = (),
        pending_reschedule: Any | None = None,
        selection_context: Sequence[dict[str, str]] = (),
    ) -> Interpretation:
        route = selected_route or (
            IntentRoute.UPDATE_PENDING if context is not None else IntentRoute.NEW_REQUEST
        )
        if route is IntentRoute.UNCERTAIN:
            return Interpretation(
                response_text="I\u2019m not sure what you want to do. Please clarify.",
                fallback_allowed=False,
            )
        if route is IntentRoute.UPDATE_PENDING and context is None:
            return Interpretation(
                response_text="There is no unfinished reminder to update.",
                fallback_allowed=False,
            )
        if route is IntentRoute.COMPLETE_REQUEST:
            # Completion is resolved from trusted Telegram delivery context;
            # the extractor must never select an occurrence or invent an ID.
            return Interpretation(fallback_allowed=False)
        if route is not IntentRoute.UPDATE_PENDING:
            context = None
        local_now = _normalise_now(now).astimezone(ZoneInfo(timezone_name))
        endpoint_host = urlsplit(self._url).hostname or "unknown"
        log_event(
            logger,
            logging.INFO,
            "reminder_intent_provider_request_started",
            endpoint_host=endpoint_host,
            model=self._model,
            input_length=len(text.strip()),
            timezone=timezone_name,
            has_pending_context=context is not None,
        )
        prompt = build_prompt(
            route=route,
            timezone_name=timezone_name,
            local_now=local_now,
            context=context,
            recent_messages=recent_messages,
            reschedule_candidates=reschedule_candidates,
            cancellation_candidates=cancellation_candidates,
            pending_reschedule=pending_reschedule,
            selection_context=selection_context,
        )
        request_payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "response_format": self._response_format(route),
        }
        log_event(
            logger,
            logging.INFO,
            "reminder_intent_provider_request",
            endpoint_host=endpoint_host,
            model=self._model,
            route=route,
            input_length=len(text.strip()),
            recent_message_count=len(recent_messages),
            has_pending_context=context is not None,
            reschedule_candidate_count=len(reschedule_candidates),
            cancellation_candidate_count=len(cancellation_candidates),
        )
        try:
            payload, content = self._client.complete(request_payload)
            log_event(
                logger,
                logging.INFO,
                "reminder_intent_provider_response",
                endpoint_host=endpoint_host,
                model=self._model,
                response_length=len(content) if isinstance(content, str) else None,
            )
            data, envelope_operation = decode_response(content)
            if envelope_operation is not None:
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_intent_provider_envelope_normalized",
                    endpoint_host=endpoint_host,
                    model=self._model,
                    operation=envelope_operation,
                )
            log_event(
                logger,
                logging.INFO,
                "reminder_intent_provider_raw_response",
                endpoint_host=endpoint_host,
                model=self._model,
                route=route,
                operation=data.get("operation"),
                field_names=sorted(data.keys()),
                content_length=(
                    len(data["content"]) if isinstance(data.get("content"), str) else None
                ),
                has_question=isinstance(data.get("question"), str),
            )
            return interpret_response(
                data,
                route=route,
                text=text,
                timezone_name=timezone_name,
                now=now,
                context=context,
                recent_messages=recent_messages,
                pending_reschedule=pending_reschedule,
                model=self._model,
                endpoint_host=endpoint_host,
                logger=logger,
                log_event_fn=log_event,
                listing_task_query=extract_listing_task_query,
                next_recurring_start_local=_next_recurring_start_local,
            )
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
                logging.ERROR,
                "reminder_intent_provider_failed",
                endpoint_host=endpoint_host,
                model=self._model,
                **exception_details(exc),
                has_pending_context=context is not None,
            )
            if route is IntentRoute.SAVE_MEMORY:
                return Interpretation(
                    response_text="What information should I remember?",
                    fallback_allowed=False,
                )
            if route is IntentRoute.RETRIEVE_MEMORY:
                return Interpretation(
                    response_text="Which saved information should I look up?",
                    fallback_allowed=False,
                )
            return Interpretation(
                response_text=(
                    "I couldn't understand that reminder. Please provide the task and future time."
                )
            )
        except Exception as exc:  # pragma: no cover - defensive provider boundary
            log_event(
                logger,
                logging.ERROR,
                "reminder_intent_provider_failed",
                endpoint_host=endpoint_host,
                model=self._model,
                **exception_details(exc),
            )
            if route is IntentRoute.SAVE_MEMORY:
                return Interpretation(
                    response_text="What information should I remember?",
                    fallback_allowed=False,
                )
            if route is IntentRoute.RETRIEVE_MEMORY:
                return Interpretation(
                    response_text="Which saved information should I look up?",
                    fallback_allowed=False,
                )
            return Interpretation(
                response_text=(
                    "I couldn't understand that reminder. Please provide the task and future time."
                )
            )
