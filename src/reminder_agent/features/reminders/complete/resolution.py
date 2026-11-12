"""Read-only completion target resolution for delivered reminders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from reminder_agent.features.reminders.complete.operations import (
    match_completion_candidates,
    resolve_by_occurrence,
    resolve_by_provider_message,
    resolve_latest_delivered,
)
from reminder_agent.infrastructure.database.models import Notification, Reminder, ReminderOccurrence

CompletionCandidate = tuple[ReminderOccurrence, Reminder, Notification]


@dataclass(frozen=True)
class CompletionTargetResolution:
    """Candidates and optional unique matches found without mutating state."""

    candidates: tuple[CompletionCandidate, ...]
    matches: tuple[CompletionCandidate, ...] = ()

    @property
    def selected(self) -> CompletionCandidate | None:
        """Return the unique selected candidate, if resolution is unambiguous."""

        return self.matches[0] if len(self.matches) == 1 else None


def resolve_completion_target(
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None,
    provider_message_id: str | None,
    channel_type: str = "TELEGRAM",
    external_destination_id: str | None = None,
    target_text: str | None = None,
    resolve_by_provider_message_fn: Callable[
        ..., list[CompletionCandidate]
    ] = resolve_by_provider_message,
    resolve_latest_delivered_fn: Callable[
        ..., list[CompletionCandidate]
    ] = resolve_latest_delivered,
    match_candidates_fn: Callable[..., list[CompletionCandidate]] = match_completion_candidates,
) -> CompletionTargetResolution:
    """Resolve a direct completion target using existing owner-scoped queries."""

    if provider_message_id is not None:
        candidates = resolve_by_provider_message_fn(
            db,
            user_id=user_id,
            identity_id=identity_id,
            chat_id=chat_id,
            provider_message_id=provider_message_id,
            channel_type=channel_type,
            external_destination_id=external_destination_id,
        )
    else:
        candidates = resolve_latest_delivered_fn(
            db,
            user_id=user_id,
            identity_id=identity_id,
            chat_id=chat_id,
            channel_type=channel_type,
            external_destination_id=external_destination_id,
        )
    if provider_message_id is None and target_text and candidates:
        targeted = match_candidates_fn(target_text, candidates)
        if len(targeted) == 1:
            candidates = targeted
    return CompletionTargetResolution(candidates=tuple(candidates))


def resolve_pending_completion_target(
    db: Session,
    *,
    user_id: UUID,
    identity_id: UUID,
    chat_id: int | None,
    candidate_occurrence_ids: Sequence[UUID],
    text: str,
    channel_type: str = "TELEGRAM",
    external_destination_id: str | None = None,
    resolve_by_occurrence_fn: Callable[..., list[CompletionCandidate]] = resolve_by_occurrence,
    match_candidates_fn: Callable[..., list[CompletionCandidate]] = match_completion_candidates,
) -> CompletionTargetResolution:
    """Resolve a pending completion reply against its owned candidate IDs."""

    candidates: list[CompletionCandidate] = []
    for occurrence_id in candidate_occurrence_ids:
        candidates.extend(
            resolve_by_occurrence_fn(
                db,
                user_id=user_id,
                identity_id=identity_id,
                chat_id=chat_id,
                occurrence_id=occurrence_id,
                channel_type=channel_type,
                external_destination_id=external_destination_id,
            )
        )
    matches = match_candidates_fn(text, candidates)
    return CompletionTargetResolution(
        candidates=tuple(candidates),
        matches=tuple(matches),
    )
