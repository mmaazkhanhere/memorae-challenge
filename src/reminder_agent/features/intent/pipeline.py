"""Channel-neutral deterministic-first intent interpretation pipeline."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from reminder_agent.features.intent.contracts import (
    IntentInterpreter,
    IntentRoute,
    Interpretation,
    NeedsClarification,
    PendingReminder,
)
from reminder_agent.features.intent.deterministic import interpreter as _deterministic_interpreter
from reminder_agent.features.intent.model import interpreter as _structured_interpreter
from reminder_agent.features.reminders.reschedule.models import RescheduleClarification
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()


class ReminderIntentInterpreter:
    """Use deterministic parsing first and an optional configured model second."""

    def __init__(self, model_interpreter: IntentInterpreter | None = None) -> None:
        self._rules = _deterministic_interpreter.RuleBasedIntentInterpreter()
        self._model = model_interpreter

    def interpret_cancellation(
        self,
        text: str,
        *,
        timezone_name: str,
        now: datetime,
        recent_messages: Sequence[str] = (),
        cancellation_candidates: Sequence[dict[str, str]] = (),
    ) -> Any:
        """Return a typed, non-authoritative cancellation proposal."""

        if isinstance(self._model, _structured_interpreter.OpenAICompatibleIntentInterpreter):
            result = self._model.interpret(
                text,
                selected_route=IntentRoute.CANCEL_REQUEST,
                timezone_name=timezone_name,
                now=now,
                recent_messages=recent_messages,
                cancellation_candidates=cancellation_candidates,
            )
            return result.cancellation_request
        method = getattr(self._model, "interpret_cancellation", None)
        if method is None:
            return None
        kwargs = {
            "timezone_name": timezone_name,
            "now": now,
            "recent_messages": recent_messages,
            "cancellation_candidates": cancellation_candidates,
        }
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        return method(
            text,
            **{key: value for key, value in kwargs.items() if key in parameters or accepts_kwargs},
        )

    def interpret_reschedule(
        self,
        text: str,
        *,
        timezone_name: str,
        now: datetime,
        reschedule_candidates: Sequence[dict[str, str]] = (),
        recent_messages: Sequence[str] = (),
        selection_context: Sequence[dict[str, str]] = (),
    ) -> Any:
        """Return a non-authoritative reschedule proposal, if configured."""

        if not isinstance(self._model, _structured_interpreter.OpenAICompatibleIntentInterpreter):
            return None
        kwargs = {
            "selected_route": IntentRoute.RESCHEDULE_REQUEST,
            "timezone_name": timezone_name,
            "now": now,
            "recent_messages": recent_messages,
            "reschedule_candidates": reschedule_candidates,
            "selection_context": selection_context,
        }
        parameters = inspect.signature(self._model.interpret).parameters
        result = self._model.interpret(
            text,
            **{key: value for key, value in kwargs.items() if key in parameters},
        )
        request = result.reschedule_request
        if (
            request is not None
            and result.response_text is not None
            and request.due_at is None
            and request.offset_minutes is None
        ):
            return RescheduleClarification(request=request, question=result.response_text)
        if request is not None and request.offset_minutes is not None:
            normalized = text.lower()
            if re.search(r"\b(?:back|earlier)\b", normalized):
                request = replace(request, offset_minutes=-abs(request.offset_minutes))
            elif re.search(r"\b(?:forward|later)\b", normalized):
                request = replace(request, offset_minutes=abs(request.offset_minutes))
        return request or result.response_text

    def interpret_pending_reschedule(
        self,
        text: str,
        *,
        timezone_name: str,
        now: datetime,
        pending_reschedule: Any,
        reschedule_candidates: Sequence[dict[str, str]] = (),
        recent_messages: Sequence[str] = (),
    ) -> Any:
        """Interpret an unfinished move without resolving IDs or mutating state."""

        if self._model is None:
            return None
        if isinstance(self._model, _structured_interpreter.OpenAICompatibleIntentInterpreter):
            result = self._model.interpret(
                text,
                selected_route=IntentRoute.RESCHEDULE_REQUEST,
                timezone_name=timezone_name,
                now=now,
                recent_messages=recent_messages,
                reschedule_candidates=reschedule_candidates,
                pending_reschedule=pending_reschedule,
            )
            return result.pending_reschedule_proposal
        method = getattr(self._model, "interpret_pending_reschedule", None)
        if method is None:
            return None
        kwargs = {
            "timezone_name": timezone_name,
            "now": now,
            "pending_reschedule": pending_reschedule,
            "reschedule_candidates": reschedule_candidates,
            "recent_messages": recent_messages,
        }
        parameters = inspect.signature(method).parameters
        return method(text, **{key: value for key, value in kwargs.items() if key in parameters})

    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
    ) -> Interpretation:
        route = selected_route or (
            IntentRoute.UPDATE_PENDING if context is not None else IntentRoute.NEW_REQUEST
        )
        if route is IntentRoute.UNCERTAIN:
            return Interpretation(
                response_text="I\u2019m not sure what you want to do. Please clarify.",
                fallback_allowed=False,
            )
        extraction_context = context if route is IntentRoute.UPDATE_PENDING else None
        result = self._rules.interpret(
            text,
            selected_route=route,
            timezone_name=timezone_name,
            now=now,
            context=extraction_context,
        )
        if (
            result.intent is not None
            or result.intents
            or result.snooze_duration_seconds is not None
            or self._model is None
            or not result.fallback_allowed
        ):
            return result
        log_event(
            logger,
            logging.INFO,
            "reminder_intent_provider_fallback",
            reason="rules_did_not_match",
        )
        if isinstance(self._model, _structured_interpreter.OpenAICompatibleIntentInterpreter):
            provider_result = self._model.interpret(
                text,
                selected_route=route,
                timezone_name=timezone_name,
                now=now,
                context=extraction_context,
                recent_messages=recent_messages,
            )
        else:
            provider_kwargs: dict[str, Any] = {
                "timezone_name": timezone_name,
                "now": now,
                "context": extraction_context,
            }
            if "selected_route" in inspect.signature(self._model.interpret).parameters:
                provider_kwargs["selected_route"] = route
            provider_result = self._model.interpret(text, **provider_kwargs)
        if route is IntentRoute.UPDATE_PENDING:
            if (
                provider_result.field_updates is not None
                or provider_result.clear_context
                or provider_result.clarification is not None
            ):
                return provider_result
            return result
        if (
            provider_result.resolved_intents
            or provider_result.clarification is not None
            or provider_result.clear_context
            or provider_result.response_text is not None
            or provider_result.memory_fact is not None
            or provider_result.memory_query is not None
        ):
            if (
                context is None
                and result.clarification is not None
                and provider_result.clarification is not None
            ):
                provider_pending = provider_result.clarification.pending
                base_pending = result.clarification.pending
                merged_values = base_pending.model_dump(mode="python")
                for field in (
                    "task",
                    "date_local",
                    "hour",
                    "minute",
                    "meridiem",
                    "date_explicit",
                    "recurrence_weekdays",
                    "recurrence_frequency",
                    "recurrence_interval",
                    "recurrence_day_of_month",
                    "recurrence_month",
                ):
                    value = getattr(provider_pending, field)
                    if field in provider_pending.model_fields_set and value is not None:
                        merged_values[field] = value
                merged_pending = PendingReminder.model_validate(merged_values)
                return Interpretation(
                    clarification=NeedsClarification(
                        pending=merged_pending,
                        question=provider_result.clarification.question,
                    ),
                    response_text=provider_result.response_text,
                    fallback_allowed=provider_result.fallback_allowed,
                )
            return provider_result
        return result
