from __future__ import annotations

import hmac
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.orm import Session

from reminder_agent.db.session import get_db
from reminder_agent.observability import get_logger, log_event
from reminder_agent.onboarding import (
    IdempotencyConflict,
    claim_delivery,
    handle_update,
    record_delivery_outcome,
)
from reminder_agent.telegram import HttpTelegramClient, TelegramAPIError, TelegramUpdate

router = APIRouter()
logger = get_logger()


def _telegram_error_is_retryable(error: TelegramAPIError) -> bool:
    # Network failures and 5xx responses are ambiguous: Telegram may have accepted
    # the message before the response was lost. Retrying can create a duplicate.
    # A 429 explicitly means Telegram did not accept this request, so it is safe.
    return error.status_code == 429


TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def _split_telegram_text(text: str, *, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a response into Telegram-safe chunks without breaking words."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        boundary = remaining.rfind("\n", 0, max_length + 1)
        if boundary <= 0:
            boundary = remaining.rfind(" ", 0, max_length + 1)
        if boundary <= 0:
            boundary = max_length
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip(" \n")
    if remaining:
        chunks.append(remaining)
    return chunks


def _send_reply_with_retry(
    client,
    reply,
    *,
    update_id: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> None:
    chunks = _split_telegram_text(reply.text)
    for chunk_index, chunk in enumerate(chunks):
        for attempt in range(1, max_attempts + 1):
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
            try:
                kwargs = {}
                if reply.reply_markup is not None and chunk_index == len(chunks) - 1:
                    kwargs["reply_markup"] = reply.reply_markup
                client.send_message(reply.chat_id, chunk, **kwargs)
                break
            except TelegramAPIError as exc:
                retryable = _telegram_error_is_retryable(exc)
                if not retryable or attempt >= max_attempts:
                    raise
                log_event(
                    logger,
                    logging.WARNING,
                    "telegram_response_delivery_retry",
                    update_id=update_id,
                    attempt=attempt,
                    failure_kind=exc.failure_kind,
                    provider_status_code=exc.status_code,
                    chunk_index=chunk_index,
                )
                if retry_delay_seconds:
                    time.sleep(retry_delay_seconds)


@router.post("/webhook/telegram")
def telegram_webhook(
    payload: dict[str, Any],
    request: Request,
    secret: str | None = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
    db: Session = Depends(get_db),
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
        client = request.app.state.telegram_client
        if update.message is not None and update.message.voice is not None and client is None:
            client = HttpTelegramClient(
                request.app.state.settings.telegram_bot_token,
                request.app.state.settings.telegram_api_timeout_seconds,
            )
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
            acknowledge_callback = getattr(client, "answer_callback_query", None)
            if callable(acknowledge_callback):
                try:
                    acknowledge_callback(update.callback_query.id)
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
        if reply is not None and reply.should_send:
            if reply.idempotency_key is not None and not claim_delivery(db, reply.idempotency_key):
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
                    record_delivery_outcome(
                        db,
                        reply.idempotency_key,
                        sent=False,
                        error_kind=exc.failure_kind,
                        status_code=exc.status_code,
                        next_attempt_at=datetime.now(timezone.utc) + timedelta(seconds=retry_delay),
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
                record_delivery_outcome(db, reply.idempotency_key, sent=True)
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
