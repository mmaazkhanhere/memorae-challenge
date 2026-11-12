"""Telegram response, idempotency, and delivery adaptation for onboarding."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import OutboundResponse
from reminder_agent.application.delivery import SqlAlchemyResponseDelivery
from reminder_agent.application.identity.resolve import resolve_linked_identity
from reminder_agent.infrastructure.database.models import ChannelIdentity, IdempotencyRecord
from reminder_agent.shared.idempotency import (
    ensure_request_matches,
    find_idempotency_record,
    request_hash,
)

TELEGRAM = "TELEGRAM"


# Compatibility import name for callers that still use the old onboarding
# response symbol. The canonical contract has no Telegram destination or
# keyboard fields.
OnboardingReply = OutboundResponse


def existing_record(db: Session, update: Any) -> IdempotencyRecord | None:
    """Load and validate a Telegram onboarding idempotency record."""

    record = find_idempotency_record(db, update, source=TELEGRAM)
    if record is not None:
        ensure_request_matches(record, update)
    return record


def reply_from_record(record: IdempotencyRecord) -> OnboardingReply:
    assert record.response is not None
    presentation_hint = record.response.get("presentation_hint")
    if presentation_hint is None and record.response.get("reply_markup") is not None:
        # Older rows contain the provider payload. The Telegram adapter resolves
        # this sentinel by reading the original row during replay.
        presentation_hint = "LEGACY_STAGED"
    return OnboardingReply(
        content=str(record.response["text"]),
        response_kind="REPLAY",
        presentation_hint=presentation_hint,
        idempotency_key=record.idempotency_key,
        should_send=False,
        is_duplicate=True,
    )


def find_sender_identity(db: Session, update: Any) -> ChannelIdentity | None:
    """Resolve Telegram sender payload to the linked persisted identity."""

    assert update.message is not None
    resolved = resolve_linked_identity(
        db,
        channel_type=TELEGRAM,
        external_id=str(update.message.from_user.id),
    )
    return None if resolved is None else resolved.identity


def store_result(
    db: Session,
    *,
    update: Any,
    user_id: UUID | None,
    operation: str,
    text: str,
    presentation_hint: str | None = None,
) -> OnboardingReply:
    """Stage a durable Telegram onboarding response in the caller's transaction."""

    source_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    assert source_message is not None
    reply = OnboardingReply(
        content=text,
        response_kind=operation,
        presentation_hint=presentation_hint,
        idempotency_key=str(update.update_id),
    )
    now = datetime.now(UTC)
    db.add(
        IdempotencyRecord(
            source=TELEGRAM,
            idempotency_key=str(update.update_id),
            user_id=user_id,
            operation=operation,
            request_hash=request_hash(update),
            status="SUCCEEDED",
            delivery_status="PENDING",
            response={
                "chat_id": source_message.chat.id,
                "text": reply.text,
                # Keep the legacy key for persisted-row compatibility. New
                # Telegram markup is rendered from the semantic hint by the
                # channel adapter.
                "reply_markup": None,
                "presentation_hint": presentation_hint,
            },
            created_at=now,
            updated_at=now,
        )
    )
    return reply


def claim_delivery(db: Session, idempotency_key: str) -> bool:
    """Compatibility wrapper for Telegram's durable response delivery claim."""

    return SqlAlchemyResponseDelivery(db, source=TELEGRAM).claim(idempotency_key)


def record_delivery_outcome(
    db: Session,
    idempotency_key: str,
    *,
    sent: bool,
    error_kind: str | None = None,
    status_code: int | None = None,
    next_attempt_at: datetime | None = None,
) -> None:
    """Compatibility wrapper for Telegram's durable response outcome."""

    SqlAlchemyResponseDelivery(db, source=TELEGRAM).record_outcome(
        idempotency_key,
        sent=sent,
        error_kind=error_kind,
        status_code=status_code,
        next_attempt_at=next_attempt_at,
    )


