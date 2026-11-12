"""Public deterministic-first conversation routing boundary."""

from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Any, cast

from reminder_agent.features.intent.context import CONVERSATION_CONTEXT_TURNS
from reminder_agent.features.intent.contracts import (
    IntentInterpreter,
    IntentRoute,
    Interpretation,
    PendingReminder,
)
from reminder_agent.features.intent.deterministic.interpreter import RuleBasedIntentInterpreter
from reminder_agent.features.intent.deterministic.routing import (
    ConversationRoute,
    route_message,
    social_response,
    unrelated_response,
)
from reminder_agent.features.intent.messages import general_message
from reminder_agent.features.intent.model.router import IntentRouter
from reminder_agent.features.intent.pending.reminder_creation import (
    restrict_to_explicit_field_updates,
)
from reminder_agent.features.intent.policy import OUT_OF_SCOPE_RESPONSE
from reminder_agent.features.memory.messages import save_memory_message
from reminder_agent.features.memory.validation import is_memory_intent_evidence_grounded
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()


def interpret_for_route(
    interpreter: IntentInterpreter,
    text: str,
    *,
    selected_route: IntentRoute,
    timezone_name: str,
    now: datetime,
    context: PendingReminder | None,
    recent_messages: list[str],
    selection_context: list[dict[str, str]] | None = None,
) -> Interpretation:
    """Invoke the typed interpreter with one explicitly selected safe route."""

    kwargs: dict[str, Any] = {"timezone_name": timezone_name, "now": now}
    parameters = inspect.signature(interpreter.interpret).parameters
    if "selected_route" in parameters:
        kwargs["selected_route"] = selected_route
    if context is not None and "context" in parameters:
        kwargs["context"] = context
    if "recent_messages" in parameters:
        kwargs["recent_messages"] = recent_messages
    if "selection_context" in parameters:
        kwargs["selection_context"] = selection_context or []
    return cast(Interpretation, cast(Any, interpreter).interpret(text, **kwargs))


def _selected_intent_route(
    route: ConversationRoute, *, pending: PendingReminder | None
) -> IntentRoute:
    if route is ConversationRoute.LIST_REQUEST:
        return IntentRoute.LIST_REQUEST
    if route is ConversationRoute.SUMMARY_REQUEST:
        return IntentRoute.SUMMARY_REQUEST
    if route is ConversationRoute.CLARIFICATION_REPLY:
        return IntentRoute.UPDATE_PENDING
    if route is ConversationRoute.MODEL_FALLBACK:
        return IntentRoute.UPDATE_PENDING if pending is not None else IntentRoute.NEW_REQUEST
    return IntentRoute.NEW_REQUEST


def _classifier_route(
    suggestion: IntentRoute,
    *,
    pending: PendingReminder | None,
) -> tuple[ConversationRoute, bool] | None:
    """Map a classifier proposal without allowing it to authorize a mutation."""

    mapping = {
        IntentRoute.LIST_REQUEST: ConversationRoute.LIST_REQUEST,
        IntentRoute.SUMMARY_REQUEST: ConversationRoute.SUMMARY_REQUEST,
        IntentRoute.RESCHEDULE_REQUEST: ConversationRoute.RESCHEDULE_REQUEST,
        IntentRoute.CANCEL_REQUEST: ConversationRoute.CANCEL_REQUEST,
        IntentRoute.COMPLETE_REQUEST: ConversationRoute.COMPLETE_REQUEST,
        IntentRoute.SNOOZE_REQUEST: ConversationRoute.SNOOZE_REQUEST,
        IntentRoute.RETRIEVE_MEMORY: ConversationRoute.RETRIEVE_MEMORY_REQUEST,
        IntentRoute.SAVE_MEMORY: ConversationRoute.SAVE_MEMORY_REQUEST,
        IntentRoute.NEW_REQUEST: ConversationRoute.NEW_REQUEST,
        IntentRoute.SOCIAL: ConversationRoute.SOCIAL,
    }
    if suggestion in mapping:
        return mapping[suggestion], False
    if suggestion is IntentRoute.UPDATE_PENDING:
        return (
            (ConversationRoute.CLARIFICATION_REPLY, True)
            if pending is not None
            else (ConversationRoute.NEW_REQUEST, False)
        )
    if suggestion is IntentRoute.CANCEL_PENDING and pending is not None:
        return ConversationRoute.CANCELLATION, False
    return None


def route_and_interpret(
    interpreter: IntentInterpreter,
    text: str,
    *,
    timezone_name: str,
    now: datetime,
    pending: PendingReminder | None,
    user_name: str | None,
    intent_router: IntentRouter | None = None,
    recent_messages: list[str] | None = None,
    recent_delivered_reminders: list[dict[str, str]] | None = None,
    selection_context: list[dict[str, str]] | None = None,
    deterministic_route: ConversationRoute | None = None,
) -> tuple[ConversationRoute, Interpretation]:
    """Route first, then interpret only the route that permits extraction."""

    route = deterministic_route or route_message(
        text, pending=pending, selection_context=selection_context or ()
    ).route
    original_route = route
    recent_messages = recent_messages or []
    recent_delivered_reminders = recent_delivered_reminders or []
    selection_context = selection_context or []
    classifier_selected_update = False
    classifier_selected = False
    classifier_memory_evidence = None
    completion_target_text: str | None = None

    log_event(logger, logging.INFO, "reminder_route_deterministic", route=route, has_pending=pending is not None)
    if route is ConversationRoute.MODEL_FALLBACK and intent_router is not None:
        classifier_selected = True
        log_event(
            logger,
            logging.INFO,
            "reminder_route_classifier_requested",
            has_pending=pending is not None,
            recent_message_count=min(len(recent_messages), CONVERSATION_CONTEXT_TURNS),
        )
        parameters = inspect.signature(intent_router.classify).parameters
        kwargs: dict[str, Any] = {"pending": pending, "recent_messages": recent_messages}
        if "recent_delivered_reminders" in parameters:
            kwargs["recent_delivered_reminders"] = recent_delivered_reminders
        if "selection_context" in parameters:
            kwargs["selection_context"] = selection_context
        suggestion_result = intent_router.classify(text, **kwargs)
        suggestion = suggestion_result.operation
        classifier_memory_evidence = getattr(suggestion_result, "memory_intent_evidence", None)
        completion_target_text = getattr(suggestion_result, "target_text", None)
        mapped = _classifier_route(suggestion, pending=pending)
        if mapped is None:
            if suggestion is IntentRoute.UNRELATED:
                return ConversationRoute.UNRELATED, Interpretation(
                    response_text=OUT_OF_SCOPE_RESPONSE,
                    fallback_allowed=False,
                )
            response = (
                "I’m not sure whether you want a new reminder or to update your unfinished one. "
                "Please say which reminder you mean."
            )
            return ConversationRoute.UNRELATED, Interpretation(
                response_text=response,
                fallback_allowed=False,
            )
        route, classifier_selected_update = mapped
        log_event(
            logger,
            logging.INFO,
            "reminder_route_classifier_mapped",
            classifier_operation=suggestion,
            deterministic_route=original_route,
            mapped_route=route,
            has_pending=pending is not None,
        )
        if suggestion is IntentRoute.UNCERTAIN:
            return ConversationRoute.UNRELATED, Interpretation(
                response_text=(
                    "I’m not sure whether you want a new reminder or to update your unfinished one. "
                    "Please say which reminder you mean."
                ),
                fallback_allowed=False,
            )
    elif route is ConversationRoute.MODEL_FALLBACK:
        log_event(
            logger,
            logging.WARNING,
            "reminder_route_classifier_skipped",
            reason="not_configured",
            fallback_intent_route=_selected_intent_route(route, pending=pending),
            has_pending=pending is not None,
        )

    if route is ConversationRoute.CANCELLATION:
        interpretation = Interpretation(
            response_text="Okay, I won't create that reminder.",
            fallback_allowed=False,
            clear_context=True,
        )
    elif route is ConversationRoute.CAPABILITIES:
        interpretation = Interpretation(response_text=general_message("capabilities"), fallback_allowed=False)
    elif route is ConversationRoute.SOCIAL:
        interpretation = Interpretation(response_text=social_response(text, user_name=user_name), fallback_allowed=False)
    elif route is ConversationRoute.UNRELATED:
        interpretation = Interpretation(response_text=unrelated_response(text, pending=pending), fallback_allowed=False)
    elif route is ConversationRoute.RESCHEDULE_REQUEST or route is ConversationRoute.CANCEL_REQUEST:
        interpretation = Interpretation(fallback_allowed=False)
    elif route is ConversationRoute.COMPLETE_REQUEST:
        interpretation = Interpretation(fallback_allowed=False, completion_target_text=completion_target_text)
    elif route is ConversationRoute.SAVE_MEMORY_REQUEST:
        grounded = not classifier_selected or is_memory_intent_evidence_grounded(
            text, classifier_memory_evidence
        )
        if not grounded:
            return ConversationRoute.UNRELATED, Interpretation(
                response_text=save_memory_message("clarification"),
                fallback_allowed=False,
            )
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=IntentRoute.SAVE_MEMORY,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages,
        )
    elif route is ConversationRoute.SUMMARY_REQUEST:
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=IntentRoute.SUMMARY_REQUEST,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages,
        )
    elif route is ConversationRoute.SNOOZE_REQUEST:
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=IntentRoute.SNOOZE_REQUEST,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages,
        )
    elif route is ConversationRoute.RETRIEVE_MEMORY_REQUEST:
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=IntentRoute.RETRIEVE_MEMORY,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages,
        )
    else:
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=_selected_intent_route(route, pending=pending),
            timezone_name=timezone_name,
            now=now,
            context=pending if route in {ConversationRoute.CLARIFICATION_REPLY, ConversationRoute.MODEL_FALLBACK} else None,
            recent_messages=recent_messages,
            selection_context=selection_context,
        )

    updates = interpretation.field_updates
    if updates is not None:
        if pending is None:
            return ConversationRoute.UNRELATED, Interpretation(
                response_text="Please tell me the reminder task and a future time.",
                fallback_allowed=False,
            )
        verified = restrict_to_explicit_field_updates(
            text,
            updates,
            context=pending,
            now=now,
            allow_task_replacement=classifier_selected_update,
            allow_semantic_field_updates=classifier_selected_update,
        )
        interpretation = RuleBasedIntentInterpreter().apply_field_updates(pending, verified, now=now)
    return route, interpretation


__all__ = ["interpret_for_route", "route_and_interpret"]
