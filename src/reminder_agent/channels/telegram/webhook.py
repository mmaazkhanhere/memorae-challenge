from __future__ import annotations

import hmac
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import OutboundResponse, RequestSource
from reminder_agent.application.delivery import SqlAlchemyResponseDelivery
from reminder_agent.channels.telegram.outbound import (
    PHONE_KEYBOARD,
    REMOVE_KEYBOARD,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    acknowledge_callback,
    is_retryable,
    remove_inline_keyboard,
    send_text,
    split_text,
)
from reminder_agent.channels.telegram.provider import (
    HttpTelegramClient,
    TelegramAPIError,
    TelegramUpdate,
)
from reminder_agent.channels.telegram.updates import (
    message_input,
    parse_completion_callback,
    resolve_identity,
    to_application_request,
    to_onboarding_request,
)
from reminder_agent.features.onboarding.service import handle_onboarding_request
from reminder_agent.features.onboarding.workflow import handle_update
from reminder_agent.infrastructure.db.session import get_db
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.idempotency import IdempotencyConflict, find_idempotency_record

router = APIRouter()
logger = get_logger()


def _telegram_error_is_retryable(error: TelegramAPIError) -> bool:
    return is_retryable(error)


def _split_telegram_text(text: str, *, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    return split_text(text, max_length=max_length)


def _send_reply_with_retry(
    client: HttpTelegramClient,
    reply: OutboundResponse,
    *,
    chat_id: int | None = None,
    reply_markup: dict | None = None,
    update_id: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> None:
    chunks = _split_telegram_text(reply.text)

    def on_attempt(chunk_index: int, attempt: int) -> None:
        log_event(
            logger,
            logging.DEBUG,
            "telegram_response_delivery_attempt",
            update_id=update_id,
            attempt=attempt,
            max_attempts=max_attempts,
            chunk_index=chunk_index,
            chunk_count=len(chunks),
        )

    if chat_id is None:
        # Compatibility for direct callers of the old Telegram delivery helper.
        chat_id = reply.chat_id
    text = reply.content if isinstance(reply, OutboundResponse) else reply.text
    send_text(
        client,
        chat_id=chat_id,
        text=text,
        reply_markup=reply_markup,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        on_attempt=on_attempt,
    )


def _telegram_chat_id(update: TelegramUpdate) -> int:
    message = update.message or (
        update.callback_query.message if update.callback_query is not None else None
    )
    if message is None:
        raise ValueError("Telegram update has no response destination")
    return message.chat.id


def _remove_completion_keyboard(
    client: Any,
    update: TelegramUpdate,
    *,
    reply: OutboundResponse,
) -> None:
    """Remove the Done button from the notification just completed."""

    if reply.response_kind != "completion" and update.callback_query is None:
        return
    if update.callback_query is not None:
        message = update.callback_query.message
    elif update.message is not None:
        message = update.message.reply_to_message
    else:
        message = None
    if message is None and getattr(reply, "cleanup_provider_message_id", None) is None:
        return
    if not hasattr(client, "edit_message_reply_markup"):
        return
    cleanup_message_id = getattr(reply, "cleanup_provider_message_id", None)
    if message is None:
        try:
            message_id = int(cleanup_message_id)
        except (TypeError, ValueError):
            return
        chat_id = reply.chat_id
        message_text = "Reminder"
    else:
        message_id = message.message_id
        chat_id = message.chat.id
        message_text = message.text or "Reminder"
    try:
        remove_inline_keyboard(
            client,
            chat_id=chat_id,
            message_id=message_id,
            text=message_text,
        )
    except TelegramAPIError as exc:
        # Completion has already committed; keyboard cleanup is best effort.
        log_event(
            logger,
            logging.WARNING,
            "telegram_completion_keyboard_cleanup_failed",
            update_id=update.update_id,
            failure_kind=exc.failure_kind,
            provider_status_code=exc.status_code,
        )


def _telegram_reply_markup(db: Session, update: TelegramUpdate, reply: OutboundResponse) -> dict | None:
    presentation_hint = getattr(reply, "presentation_hint", None)
    if presentation_hint == "PHONE_CONTACT_REQUEST":
        return PHONE_KEYBOARD
    if presentation_hint == "REMOVE_KEYBOARD":
        return REMOVE_KEYBOARD
    if presentation_hint != "LEGACY_STAGED":
        return None
    # Existing rows may contain the exact historical Telegram payload. Read it
    # only in the Telegram adapter so old replay behavior remains byte-for-byte.
    record = find_idempotency_record(db, update, source="TELEGRAM")
    return None if record is None or record.response is None else record.response.get("reply_markup")


@router.post("/webhook/telegram")
def telegram_webhook(
    payload: dict[str, Any],
    request: Request,
    secret: str | None = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
    db: Session = Depends(get_db),  # noqa: B008
) -> dict[str, bool]:
    expected_secret = request.app.state.settings.telegram_webhook_secret
    if secret is None or not hmac.compare_digest(secret, expected_secret):
        log_event(logger, logging.WARNING, "webhook_auth_failed")
        raise HTTPException(status_code=401, detail="Invalid webhook credentials")

    try:
        update = TelegramUpdate.model_validate(payload)
    except ValidationError as exc:
        log_event(logger, logging.WARNING, "telegram_update_invalid", exc_info=True)
        raise HTTPException(status_code=400, detail="Malformed Telegram update") from exc

    log_event(
        logger,
        logging.INFO,
        "telegram_update_received",
        update_id=update.update_id,
        has_message=update.message is not None,
        chat_type=update.message.chat.type if update.message is not None else None,
        has_contact=update.message is not None and update.message.contact is not None,
        has_voice=update.message is not None and update.message.voice is not None,
        contact_owner_present=(
            update.message is not None
            and update.message.contact is not None
            and update.message.contact.user_id is not None
        ),
    )

    try:
        inbound = message_input(update)
        db.rollback()
        client = request.app.state.telegram_client
        if update.message is not None and update.message.voice is not None and client is None:
            client = HttpTelegramClient(
                request.app.state.settings.telegram_bot_token,
                request.app.state.settings.telegram_api_timeout_seconds,
            )
        onboarding_request = to_onboarding_request(update)
        completion_callback = (
            parse_completion_callback(
                update,
                callback_secret=request.app.state.settings.telegram_webhook_secret,
            )
            if update.callback_query is not None
            else None
        )
        reply = (
            handle_onboarding_request(
                db,
                onboarding_request,
                default_region=request.app.state.settings.phone_default_region,
                now=request.app.state.clock(),
            )
            if onboarding_request is not None
            else None
        )
        application_request = None
        if reply is None:
            resolved_identity = resolve_identity(db, update) if inbound is not None else None
            application_request = (
                to_application_request(
                    inbound,
                    user=resolved_identity.user,
                    identity=resolved_identity.identity,
                )
                if inbound is not None and resolved_identity is not None
                else None
            )
            # Identity extraction is a read-only adapter step. Release its
            # implicit SQLAlchemy transaction before the application workflow
            # owns the next transaction for a reminder or callback.
            db.rollback()
            reply = handle_update(
                db,
                update,
                default_region=request.app.state.settings.phone_default_region,
                reminder_interpreter=request.app.state.reminder_interpreter,
                intent_router=request.app.state.intent_router,
                memory_answerer=request.app.state.memory_answerer,
                embedding_provider=request.app.state.embedding_provider,
                embedding_privacy_policy=request.app.state.embedding_privacy_policy,
                memory_retrieval_policy=request.app.state.memory_retrieval_policy,
                now=request.app.state.clock(),
                session_ttl_seconds=request.app.state.settings.conversation_session_ttl_seconds,
                callback_secret=request.app.state.settings.telegram_webhook_secret,
                telegram_client=client,
                transcriber=request.app.state.transcriber,
                voice_max_bytes=request.app.state.settings.telegram_voice_max_bytes,
                voice_max_duration_seconds=(
                    request.app.state.settings.telegram_voice_max_duration_seconds
                ),
                log_voice_transcript=request.app.state.settings.telegram_voice_log_transcript,
                application_request=application_request,
                completion_callback=completion_callback,
            )
        delivery = SqlAlchemyResponseDelivery(db, source=RequestSource.TELEGRAM)
        reply_markup = (
            _telegram_reply_markup(db, update, reply) if reply is not None else None
        )
        log_event(
            logger,
            logging.INFO,
            "telegram_update_handled",
            update_id=update.update_id,
            reply_created=reply is not None,
            reply_should_send=reply.should_send if reply is not None else False,
            duplicate=reply.is_duplicate if reply is not None else False,
        )
        if reply is not None and reply.is_duplicate:
            log_event(
                logger,
                logging.INFO,
                "telegram_update_duplicate",
                update_id=update.update_id,
            )
        if reply is not None and update.callback_query is not None:
            if client is None:
                client = HttpTelegramClient(
                    request.app.state.settings.telegram_bot_token,
                    request.app.state.settings.telegram_api_timeout_seconds,
                )
            if hasattr(client, "answer_callback_query"):
                try:
                    acknowledge_callback(client, update.callback_query.id)
                except TelegramAPIError as exc:
                    # Callback acknowledgement is only UI feedback. The
                    # completion mutation and durable response must still
                    # be delivered even if this best-effort call fails.
                    log_event(
                        logger,
                        logging.WARNING,
                        "telegram_callback_acknowledgement_failed",
                        update_id=update.update_id,
                        failure_kind=exc.failure_kind,
                        provider_status_code=exc.status_code,
                    )
        if reply is not None and reply.response_kind == "completion":
            if client is None:
                client = HttpTelegramClient(
                    request.app.state.settings.telegram_bot_token,
                    request.app.state.settings.telegram_api_timeout_seconds,
                )
            _remove_completion_keyboard(client, update, reply=reply)
        if reply is not None and reply.should_send:
            if reply.idempotency_key is not None and not delivery.claim(reply.idempotency_key):
                log_event(
                    logger,
                    logging.INFO,
                    "telegram_delivery_skipped",
                    update_id=update.update_id,
                    reason="already_attempted",
                )
                return {"ok": True}
            log_event(
                logger,
                logging.DEBUG,
                "telegram_response_delivery_started",
                update_id=update.update_id,
            )
            try:
                if client is None:
                    client = request.app.state.telegram_client or HttpTelegramClient(
                        request.app.state.settings.telegram_bot_token,
                        request.app.state.settings.telegram_api_timeout_seconds,
                    )
                _send_reply_with_retry(
                    client,
                    reply,
                    update_id=update.update_id,
                    chat_id=_telegram_chat_id(update),
                    reply_markup=reply_markup,
                    max_attempts=request.app.state.settings.telegram_confirmation_max_attempts,
                    retry_delay_seconds=(
                        request.app.state.settings.telegram_confirmation_retry_delay_seconds
                    ),
                )
            except TelegramAPIError as exc:
                retry_after = exc.retry_after_seconds or 0
                retry_delay = retry_after or (
                    request.app.state.settings.telegram_confirmation_background_retry_base_delay_seconds
                )
                if reply.idempotency_key is not None:
                    delivery.record_outcome(
                        reply.idempotency_key,
                        sent=False,
                        error_kind=exc.failure_kind,
                        status_code=exc.status_code,
                        next_attempt_at=datetime.now(UTC) + timedelta(seconds=retry_delay),
                    )
                log_event(
                    logger,
                    logging.ERROR,
                    "telegram_delivery_failed",
                    update_id=update.update_id,
                    failure_kind=exc.failure_kind,
                    network_cause=exc.cause,
                    provider_status_code=exc.status_code,
                    provider_description=exc.telegram_description,
                    exc_info=True,
                )
                # Durable mutations use the idempotency record to prevent a
                # replay. Stateless policy responses have no database record
                # by design, so a retry may deliver the same harmless response.
                return {"ok": True}
            if reply.idempotency_key is not None:
                delivery.record_outcome(reply.idempotency_key, sent=True)
            log_event(
                logger,
                logging.INFO,
                "telegram_response_delivery_succeeded",
                update_id=update.update_id,
            )
    except IdempotencyConflict as exc:
        log_event(
            logger,
            logging.WARNING,
            "telegram_update_conflict",
            update_id=update.update_id,
            exc_info=True,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        log_event(
            logger,
            logging.ERROR,
            "telegram_update_failed",
            update_id=update.update_id,
            exc_info=True,
        )
        raise

    log_event(
        logger,
        logging.INFO,
        "telegram_update_processed",
        update_id=update.update_id,
        response_sent=reply is not None,
    )
    return {"ok": True}
