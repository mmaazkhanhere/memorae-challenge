"""Shared handling for multi-reminder clarification drafts."""

from __future__ import annotations

from datetime import datetime

from reminder_agent.features.intent.contracts import (
    Interpretation,
    NeedsClarification,
    PendingReminderBatch,
    PendingReminderFieldUpdate,
)
from reminder_agent.features.intent.deterministic.interpreter import RuleBasedIntentInterpreter


def merge_pending_reminder_batch(
    interpretation: Interpretation,
    *,
    batch: PendingReminderBatch,
    now: datetime,
) -> Interpretation:
    """Apply a clarification reply to one draft and retain the other drafts."""

    active_index = batch.clarification_index
    if interpretation.clarification is not None:
        pending = interpretation.clarification.pending
        if isinstance(pending, PendingReminderBatch):
            return interpretation
        drafts = list(batch.reminders)
        drafts[active_index] = pending
        updated_batch = batch.model_copy(
            update={"reminders": tuple(drafts), "clarification_index": active_index}
        )
        return Interpretation(
            clarification=NeedsClarification(
                pending=updated_batch,
                question=interpretation.clarification.question,
            ),
            response_text=interpretation.response_text,
            fallback_allowed=interpretation.fallback_allowed,
        )

    if not interpretation.resolved_intents:
        return interpretation

    intents = list(interpretation.resolved_intents)
    drafts = list(batch.reminders)
    for index, draft in enumerate(batch.reminders):
        if index == active_index:
            continue
        resolved = RuleBasedIntentInterpreter().apply_field_updates(
            draft,
            PendingReminderFieldUpdate(),
            now=now,
        )
        if resolved.intent is not None:
            intents.append(resolved.intent)
            continue
        if resolved.clarification is None:
            return Interpretation(
                response_text="Please provide the remaining reminder details.",
                fallback_allowed=False,
            )
        drafts[index] = resolved.clarification.pending
        updated_batch = batch.model_copy(
            update={"reminders": tuple(drafts), "clarification_index": index}
        )
        return Interpretation(
            clarification=NeedsClarification(
                pending=updated_batch,
                question=resolved.clarification.question,
            ),
            response_text=resolved.response_text,
            fallback_allowed=False,
        )
    return Interpretation(intents=tuple(intents), fallback_allowed=False)
