from __future__ import annotations

import hmac
import logging
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
            try:
                client = request.app.state.telegram_client or HttpTelegramClient(
                    request.app.state.settings.telegram_bot_token,
                    request.app.state.settings.telegram_api_timeout_seconds,
                )
                if reply.reply_markup is None:
                    client.send_message(reply.chat_id, reply.text)
                else:
                    client.send_message(
                        reply.chat_id,
                        reply.text,
                        reply_markup=reply.reply_markup,
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
