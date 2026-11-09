"""Channel-neutral preparation for pending completion replies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from reminder_agent.application.conversations import (
    load_session,
    pending_completion_from_session,
)
from reminder_agent.features.reminders.complete.operations import (
    match_completion_candidates,
    resolve_by_occurrence,
)
from reminder_agent.features.reminders.complete.resolution import (
    CompletionCandidate,
    resolve_pending_completion_target,
)
from reminder_agent.infrastructure.db.models import ConversationSession


@dataclass(frozen=True)
class PendingCompletionPreparation:
    """Locked session and owner-scoped candidates prepared for one reply."""

    session: ConversationSession
    candidates: tuple[CompletionCandidate, ...]
    matches: tuple[CompletionCandidate, ...]


def prepare_pending_completion(
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    channel_type: str,
    external_session_id: str,
    chat_id: int | None,
    external_destination_id: str | None,
    text: str,
    now: datetime,
    session_ttl_seconds: int,
    resolve_by_occurrence_fn: Callable[..., list[CompletionCandidate]] = resolve_by_occurrence,
    match_candidates_fn: Callable[..., list[CompletionCandidate]] = match_completion_candidates,
) -> PendingCompletionPreparation | None:
    """Load and resolve a pending completion without opening a transaction.

    The caller intentionally invokes this from its existing transaction block.
    ``for_update=True`` therefore retains the original PostgreSQL row lock and
    transaction ownership while this module remains free of channel payloads.
    """

    session = load_session(
        db,
        user_id=user_id,
        external_session_id=external_session_id,
        channel_type=channel_type,
        now=now,
        ttl_seconds=session_ttl_seconds,
        for_update=True,
    )
    pending = pending_completion_from_session(session)
    if pending is None:
        return None
    assert session is not None

    resolution = resolve_pending_completion_target(
        db,
        user_id=user_id,
        identity_id=identity_id,
        chat_id=chat_id,
        external_destination_id=external_destination_id,
        channel_type=channel_type,
        candidate_occurrence_ids=pending.candidate_occurrence_ids,
        text=text,
        resolve_by_occurrence_fn=resolve_by_occurrence_fn,
        match_candidates_fn=match_candidates_fn,
    )
    return PendingCompletionPreparation(
        session=session,
        candidates=tuple(resolution.candidates),
        matches=tuple(resolution.matches),
    )
