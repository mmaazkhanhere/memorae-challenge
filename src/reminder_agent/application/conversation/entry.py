"""Channel-neutral deterministic conversation entry behavior."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import ApplicationRequest, ApplicationResult
from reminder_agent.application.identity.context import ACTIVE, authorized_user_for_request
from reminder_agent.features.intent.contracts import (
    IntentInterpreter,
    IntentRoute,
    Interpretation,
    PendingReminder,
)
from reminder_agent.features.intent.deterministic.routing import ConversationRoute, route_message
from reminder_agent.features.intent.model.router import IntentRouter
from reminder_agent.features.reminders.listing.service import list_reminders_from_interpretation


@dataclass(frozen=True)
class ConversationEntryResult:
    """One deterministic route plus an optional shared-operation response."""

    route: ConversationRoute | None
    response: ApplicationResult | None = None


def interpret_for_route(
    interpreter: IntentInterpreter,
    text: str,
    *,
    selected_route: IntentRoute,
    timezone_name: str,
    now: datetime,
    context: PendingReminder | None,
    recent_messages: list[str],
) -> Interpretation:
    """Call the existing typed interpreter with an explicit safe route."""

    kwargs: dict[str, Any] = {"timezone_name": timezone_name, "now": now}
    parameters = inspect.signature(interpreter.interpret).parameters
    if "selected_route" in parameters:
        kwargs["selected_route"] = selected_route
    if context is not None and "context" in parameters:
        kwargs["context"] = context
    if "recent_messages" in parameters:
        kwargs["recent_messages"] = recent_messages
    dynamic_interpreter = cast(Any, interpreter)
    return cast(Interpretation, dynamic_interpreter.interpret(text, **kwargs))


def process_authenticated_conversation(
    db: Session,
    *,
    request: ApplicationRequest,
    interpreter: IntentInterpreter,
    now: datetime,
    has_pending_clarification: bool,
    pending_reminder: PendingReminder | None = None,
    intent_router: IntentRouter | None = None,
    recent_messages: list[str] | None = None,
) -> ConversationEntryResult:
    """Route an authenticated request before feature-specific dispatch."""

    user = authorized_user_for_request(db, request=request)
    if user is None:
        return ConversationEntryResult(
            route=None,
            response=ApplicationResult(
                content="I couldn't verify your account.",
                response_kind="AUTHORIZATION_FAILURE",
            ),
        )
    if user.onboarding_status != ACTIVE or not user.timezone:
        return ConversationEntryResult(
            route=None,
            response=ApplicationResult(
                content="Please finish onboarding with your IANA time zone before creating reminders.",
                response_kind="ONBOARDING_INCOMPLETE",
            ),
        )

    route = route_message(request.message_text, pending=pending_reminder).route
    if has_pending_clarification or route is not ConversationRoute.LIST_REQUEST:
        return ConversationEntryResult(route=route)

    _ = intent_router
    interpretation = interpret_for_route(
        interpreter,
        request.message_text,
        selected_route=IntentRoute.LIST_REQUEST,
        timezone_name=user.timezone,
        now=now,
        context=None,
        recent_messages=recent_messages or [],
    )
    return ConversationEntryResult(
        route=route,
        response=list_reminders_from_interpretation(
            db,
            request=request,
            interpretation=interpretation,
            now=now,
        ),
    )
