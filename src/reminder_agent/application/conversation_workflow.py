"""Compatibility surface for the pre-feature-split conversation workflow.

The implementation lives in :mod:`conversation_coordinator`; this module only
keeps established imports working while callers migrate to the coordinator and
feature services.
"""

import reminder_agent.application.conversation_coordinator as _coordinator
import reminder_agent.features.reminders.complete.service as _completion_service
from reminder_agent.application.conversation_coordinator import (
    _recover_from_idempotency_race,
    _store_response,
    list_owned_reminders,
    restrict_to_explicit_field_updates,
)
from reminder_agent.application.conversation_coordinator import (
    handle_reminder_text as _handle_reminder_text,
)
from reminder_agent.shared.idempotency import IdempotencyConflict

__all__ = [
    "IdempotencyConflict",
    "handle_reminder_text",
    "list_owned_reminders",
    "restrict_to_explicit_field_updates",
]


def handle_reminder_text(*args, **kwargs):
    """Forward to the coordinator while preserving legacy patch seams."""

    _coordinator.restrict_to_explicit_field_updates = restrict_to_explicit_field_updates
    return _handle_reminder_text(*args, **kwargs)


def _handle_completion_follow_up(*args, **kwargs):
    """Forward completion follow-ups while preserving response test seams."""

    _completion_service._recover_from_idempotency_race = _recover_from_idempotency_race
    _completion_service._store_response = _store_response
    return _completion_service._handle_completion_follow_up(*args, **kwargs)
