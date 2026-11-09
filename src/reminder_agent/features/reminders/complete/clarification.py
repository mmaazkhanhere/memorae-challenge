"""Telegram completion clarification and response coordination."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import TypeVar
from uuid import UUID

from reminder_agent.application.conversations import end_session
from reminder_agent.application.pending_contracts import PendingCompletion
from reminder_agent.features.reminders.complete.response import stage_completion_response
from reminder_agent.infrastructure.db.models import ConversationSession
from reminder_agent.infrastructure.observability import get_logger, log_event

ResponseT = TypeVar("ResponseT")
logger = get_logger()


def stage_completion_clarification[ResponseT](
    *,
    user_id: UUID,
    external_session_id: str,
    candidate_occurrence_ids: Sequence[UUID],
    now: datetime,
    upsert_clarification_session: Callable[..., object],
    stage_response: Callable[..., ResponseT],
    log_event_fn: Callable[..., object] = log_event,
    event_context: Mapping[str, object] | None = None,
) -> ResponseT:
    """Persist a direct-completion clarification and stage its response."""

    event_context = event_context or {}
    candidate_ids = tuple(candidate_occurrence_ids)
    response_text = (
        "Which delivered reminder did you complete?"
        if candidate_ids
        else "I couldn't find a delivered reminder to mark as done."
    )
    if candidate_ids:
        pending_completion = PendingCompletion(candidate_occurrence_ids=candidate_ids)
        upsert_clarification_session(
            user_id=user_id,
            external_session_id=external_session_id,
            pending=pending_completion,
            now=now,
        )
        log_event_fn(
            logger,
            logging.INFO,
            "completion_clarification_saved",
            **event_context,
            candidate_count=len(candidate_ids),
        )
    log_event_fn(
        logger,
        logging.INFO,
        "completion_target_ambiguous",
        **event_context,
        candidate_count=len(candidate_ids),
    )
    return stage_completion_response(
        stage_response=stage_response,
        text=response_text,
        response_kind="completion_ambiguous",
    )


def stage_pending_completion_unresolved[ResponseT](
    *,
    session: ConversationSession,
    candidates: Sequence[object],
    matches: Sequence[object],
    now: datetime,
    stage_response: Callable[..., ResponseT],
    log_event_fn: Callable[..., object] = log_event,
    event_context: Mapping[str, object] | None = None,
) -> tuple[ResponseT, str]:
    """Clear stale pending context when needed and stage an unresolved reply."""

    event_context = event_context or {}
    if not candidates:
        response_text = "I couldn't find a delivered reminder to mark as done."
        end_session(session, now=now)
        outcome = "NO_CANDIDATES"
    else:
        response_text = "Which delivered reminder did you complete? Please name it."
        outcome = "AMBIGUOUS"
    reply = stage_completion_response(
        stage_response=stage_response,
        text=response_text,
        response_kind="completion_ambiguous",
    )
    log_event_fn(
        logger,
        logging.INFO,
        "completion_follow_up_unresolved",
        **event_context,
        outcome=outcome,
        candidate_count=len(candidates),
        match_count=len(matches),
    )
    return reply, outcome
