"""Telegram request/reply adaptation for reminder workflows.

This module deliberately owns only provider payload and durable-delivery shapes.
Callers retain transaction ownership and invoke shared application operations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    AuthenticatedRequestContext,
    DurableResponse,
    RequestSource,
    VerifiedChannelIdentity,
)
from reminder_agent.application.conversations import telegram_session_id
from reminder_agent.infrastructure.db.models import ChannelIdentity, IdempotencyRecord, User
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.idempotency import (
    IdempotencyConflict,
    ensure_request_matches,
    find_idempotency_record,
    request_hash,
)

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
logger = get_logger()


@dataclass(frozen=True)
class ReminderReply:
    chat_id: int
    text: str
    idempotency_key: str | None = None
    should_send: bool = True
    is_duplicate: bool = False
    reply_markup: dict | None = None
    response_kind: str = "clarification"
    cleanup_provider_message_id: str | None = None


def _existing_record(db: Session, update: Any) -> IdempotencyRecord | None:
    record = find_idempotency_record(db, update, source=TELEGRAM)
    if record is not None:
        try:
            ensure_request_matches(record, update)
        except IdempotencyConflict:
            log_event(
                logger,
                logging.WARNING,
                "reminder_idempotency_conflict",
                update_id=update.update_id,
            )
            raise
    return record


def _reply_from_record(record: IdempotencyRecord) -> ReminderReply:
    assert record.response is not None
    return ReminderReply(
        chat_id=int(record.response["chat_id"]),
        text=str(record.response["text"]),
        idempotency_key=record.idempotency_key,
        should_send=False,
        is_duplicate=True,
        reply_markup=record.response.get("reply_markup"),
        response_kind=(
            "completion" if record.operation == "COMPLETE_REMINDER" else "REPLAY"
        ),
        cleanup_provider_message_id=record.response.get("cleanup_provider_message_id"),
    )


def _recover_from_idempotency_race(db: Session, update: Any) -> ReminderReply | None:
    """Load the committed winner after a concurrent idempotency insert conflict."""

    db.rollback()
    record = find_idempotency_record(db, update, source=TELEGRAM)
    if record is None:
        db.rollback()
        return None
    try:
        ensure_request_matches(record, update)
    except IdempotencyConflict:
        log_event(
            logger,
            logging.WARNING,
            "reminder_idempotency_conflict",
            update_id=update.update_id,
            phase="race_recovery",
        )
        raise
    db.rollback()
    log_event(
        logger,
        logging.INFO,
        "reminder_command_duplicate",
        update_id=update.update_id,
        phase="race_recovery",
    )
    return _reply_from_record(record)


def _authorized_telegram_identity(
    user: User | None,
    identity: ChannelIdentity | None,
    *,
    telegram_user_id: int,
    chat_id: int,
) -> bool:
    """Check that the Telegram identity belongs to the requested user."""

    return bool(
        user is not None
        and identity is not None
        and user.onboarding_status == ACTIVE
        and bool(user.timezone)
        and identity.user_id == user.id
        and identity.channel_type == TELEGRAM
        and identity.status == LINKED
        and identity.external_id == str(telegram_user_id)
        and identity.telegram_chat_id == chat_id
    )


def _application_request_from_telegram_update(
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity,
) -> ApplicationRequest:
    """Adapt an already-authorized Telegram update to the shared request contract."""

    assert update.message is not None
    return ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.TELEGRAM,
            canonical_user_id=user.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type=identity.channel_type,
            ),
            external_session_id=telegram_session_id(update.message.chat.id),
        ),
        message_text=update.message.text or "",
        idempotency_key=str(update.update_id),
    )


def _store_response(
    db: Session,
    *,
    update: Any,
    user_id: UUID,
    text: str | None = None,
    durable_response: DurableResponse | None = None,
    response_kind: str = "clarification",
    operation: str = "CREATE_REMINDER",
    include_request_text: bool = True,
    cleanup_provider_message_id: str | None = None,
    response_metadata: dict[str, Any] | None = None,
) -> ReminderReply:
    """Stage an idempotent Telegram response in the caller's transaction."""

    source_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    assert source_message is not None
    if text is None:
        if durable_response is None:
            raise ValueError("a response text or durable response is required")
        text = durable_response.content
    elif durable_response is not None:
        raise ValueError("provide either response text or a durable response")
    reply = ReminderReply(
        chat_id=source_message.chat.id,
        text=text,
        idempotency_key=str(update.update_id),
        response_kind=response_kind,
        cleanup_provider_message_id=cleanup_provider_message_id,
    )
    now = datetime.now(UTC)
    response = {
        "chat_id": reply.chat_id,
        "text": reply.text,
        "reply_markup": None,
    }
    if include_request_text and update.message is not None and update.message.voice is None:
        response["request_text"] = update.message.text or ""
    if cleanup_provider_message_id is not None:
        response["cleanup_provider_message_id"] = cleanup_provider_message_id
    if response_metadata:
        response.update(response_metadata)
    db.add(
        IdempotencyRecord(
            source=TELEGRAM,
            idempotency_key=str(update.update_id),
            user_id=user_id,
            operation=operation,
            request_hash=request_hash(update),
            status="SUCCEEDED",
            delivery_status="PENDING",
            response=response,
            created_at=now,
            updated_at=now,
        )
    )
    log_event(
        logger,
        logging.DEBUG,
        "reminder_response_staged",
        update_id=update.update_id,
        response_kind=response_kind,
    )
    return reply


@dataclass(frozen=True)
class TelegramMemoryResponseAdapter:
    """Adapt memory workflow outcomes to the legacy Telegram response contract."""

    db: Session
    update: Any
    recover_race_callback: Callable[[], ReminderReply | None]
    find_record: Callable[[Session, Any], IdempotencyRecord | None] = _existing_record
    record_reply: Callable[[IdempotencyRecord], ReminderReply] = _reply_from_record
    store_response: Callable[..., ReminderReply] = _store_response

    def existing(self) -> ReminderReply | None:
        record = self.find_record(self.db, self.update)
        return None if record is None else self.record_reply(record)

    def recover_from_race(self) -> ReminderReply | None:
        return self.recover_race_callback()

    def direct(self, text: str) -> ReminderReply:
        assert self.update.message is not None
        return ReminderReply(self.update.message.chat.id, text)

    def stage(
        self,
        *,
        user_id: UUID,
        text: str | None = None,
        durable_response: DurableResponse | None = None,
        response_kind: str,
        operation: str,
    ) -> ReminderReply:
        return self.store_response(
            self.db,
            update=self.update,
            user_id=user_id,
            text=text,
            durable_response=durable_response,
            response_kind=response_kind,
            operation=operation,
            include_request_text=False,
        )


@dataclass(frozen=True)
class TelegramMemoryLookupResponseAdapter:
    """Adapt memory lookup responses to the legacy Telegram reply contract."""

    db: Session
    update: Any
    find_record: Callable[[Session, Any], IdempotencyRecord | None] = _existing_record
    record_reply: Callable[[IdempotencyRecord], ReminderReply] = _reply_from_record

    def existing(self) -> ReminderReply | None:
        record = self.find_record(self.db, self.update)
        return None if record is None else self.record_reply(record)

    def direct(self, text: str) -> ReminderReply:
        assert self.update.message is not None
        return ReminderReply(self.update.message.chat.id, text)
