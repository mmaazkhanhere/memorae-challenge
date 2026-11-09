"""Channel-neutral request, command, and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum
from uuid import UUID

from reminder_agent.features.reminders.cancel.decisions import CancellationScope
from reminder_agent.features.reminders.reschedule.decisions import RescheduleScope


class RequestSource(StrEnum):
    """Authenticated ingress providers, independent of persisted identity type."""

    TELEGRAM = "TELEGRAM"
    VAPI = "VAPI"
    WHATSAPP = "WHATSAPP"
    WEB_API = "WEB_API"


@dataclass(frozen=True)
class VerifiedChannelIdentity:
    """A persisted, provider-verified identity selected by a channel adapter."""

    identity_id: UUID
    channel_type: str


@dataclass(frozen=True)
class DeliveredReminderContext:
    """Verified delivery context supplied by a channel adapter."""

    identity_id: UUID
    channel_type: str
    external_destination_id: str | None = None
    provider_message_id: str | None = None


@dataclass(frozen=True)
class AuthenticatedRequestContext:
    """Trusted canonical-user context created only after adapter verification."""

    source: RequestSource
    canonical_user_id: UUID
    verified_identity: VerifiedChannelIdentity
    external_session_id: str


@dataclass(frozen=True)
class IncomingMessage:
    """Normalized channel input handed to the application boundary."""

    source: RequestSource
    text: str
    external_message_id: str | None
    external_session_id: str
    idempotency_key: str | None


@dataclass(frozen=True)
class ApplicationRequest:
    """Channel-neutral message input and source idempotency key."""

    context: AuthenticatedRequestContext
    message_text: str
    idempotency_key: str | None

    @property
    def incoming_message(self) -> IncomingMessage:
        """Expose the normalized message without changing the legacy request shape."""

        return IncomingMessage(
            source=self.context.source,
            text=self.message_text,
            external_message_id=self.idempotency_key,
            external_session_id=self.context.external_session_id,
            idempotency_key=self.idempotency_key,
        )


@dataclass(frozen=True)
class ListRemindersCommand:
    """Validated list filters, independent of provider payload formats."""

    start_date: date | None
    end_date: date | None
    task_query: str | None
    timezone_name: str
    time_after: time | None = None
    time_before: time | None = None
    weekdays: tuple[int, ...] | None = None
    limit: int | None = None


@dataclass(frozen=True)
class SaveMemoryCommand:
    """Validated memory content selected by the deterministic/LLM pipeline."""

    content: str


@dataclass(frozen=True)
class CancelReminderCommand:
    """A validated, owner-scoped cancellation target selected by existing routing."""

    reminder_id: UUID
    occurrence_id: UUID | None
    scope: CancellationScope
    occurrence_date: date | None = None


@dataclass(frozen=True)
class RescheduleReminderCommand:
    """A validated, owner-scoped rescheduling target and destination."""

    reminder_id: UUID
    occurrence_id: UUID | None
    due_at: datetime | None
    scope: RescheduleScope | None
    destination_weekday: int | None = None


@dataclass(frozen=True)
class CompleteReminderCommand:
    """Validated completion target plus its verified delivery reference."""

    occurrence_id: UUID
    delivery: DeliveredReminderContext


@dataclass(frozen=True)
class SnoozeReminderCommand:
    """Validated snooze duration plus its verified delivery reference."""

    occurrence_id: UUID
    duration_seconds: int
    delivery: DeliveredReminderContext


@dataclass(frozen=True)
class DurableResponse:
    """Provider-independent outcome content for adapter-owned persistence."""

    content: str
    response_kind: str


# Named contracts for adapters; aliases preserve one object identity for callers
# that already use the original application names.
AuthenticatedUserContext = AuthenticatedRequestContext
OutgoingResponse = DurableResponse


@dataclass(frozen=True)
class OutboundResponse:
    """Channel-neutral meaning of an application response.

    Destination identifiers, keyboards, callback data, and provider payloads
    deliberately do not belong here. Channel adapters render this result for
    their own transport. ``presentation_hint`` is a semantic action, not a
    provider payload.
    """

    content: str
    response_kind: str
    resource_id: UUID | None = None
    affected_count: int | None = None
    presentation_hint: str | None = None
    idempotency_key: str | None = None
    should_send: bool = True
    is_duplicate: bool = False

    @property
    def text(self) -> str:
        """Compatibility spelling for callers that previously consumed text."""

        return self.content

    @property
    def durable_response(self) -> DurableResponse:
        """Return the provider-independent portion of this successful outcome."""

        return DurableResponse(content=self.content, response_kind=self.response_kind)


@dataclass(frozen=True)
class ApplicationResult(OutboundResponse):
    """Backward-compatible name for the canonical outbound response contract."""
