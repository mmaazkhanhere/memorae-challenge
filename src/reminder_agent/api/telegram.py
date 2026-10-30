from __future__ import annotations

import hmac
import logging
import time
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
    return error.failure_kind in {"network", "network_timeout"} or error.status_code in {
        429,
        500,
        502,
        503,
        504,
    }


def _send_reply_with_retry(
    client,
    reply,
    *,
    update_id: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> None:
    for attempt in range(1, max_attempts + 1):
        log_event(
            logger,
            logging.DEBUG,
            "telegram_response_delivery_attempt",
            update_id=update_id,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        try:
            if reply.reply_markup is None:
                client.send_message(reply.chat_id, reply.text)
            else:
                client.send_message(
                    reply.chat_id,
                    reply.text,
                    reply_markup=reply.reply_markup,
                )
            return
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
        contact_owner_present=(
            update.message is not None
            and update.message.contact is not None
            and update.message.contact.user_id is not None
        ),
    )

    try:
        reply = handle_update(
            db,
            update,
            default_region=request.app.state.settings.phone_default_region,
            reminder_interpreter=request.app.state.reminder_interpreter,
            now=request.app.state.clock(),
        )
        if reply is not None and reply.is_duplicate:
            log_event(
                logger,
                logging.INFO,
                "telegram_update_duplicate",
                update_id=update.update_id,
            )
        if reply is not None and reply.should_send and reply.idempotency_key is not None:
            if not claim_delivery(db, reply.idempotency_key):
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
                record_delivery_outcome(db, reply.idempotency_key, sent=False)
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
                # The mutation is durable. A 200 prevents Telegram from replaying
                # the inbound update and creating a duplicate response.
                return {"ok": True}
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
