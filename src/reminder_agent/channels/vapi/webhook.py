"""Vapi server webhook adapter for VS-16 inbound call authorization."""

from __future__ import annotations

import hmac
import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from reminder_agent.application.voice_call_authorization import (
    authorize_incoming_vapi_call,
    end_vapi_call,
)
from reminder_agent.channels.vapi.contracts import VapiServerRequest
from reminder_agent.infrastructure.database.session import get_db
from reminder_agent.infrastructure.observability import get_logger, log_event

router = APIRouter()
logger = get_logger()

ASSISTANT_REQUEST = "assistant-request"
END_OF_CALL_REPORT = "end-of-call-report"
GENERIC_REJECTION = "We could not verify this call. Please link this number in Telegram."


def _check_bearer_token(
    authorization: str | None,
    *,
    expected_token: str,
) -> bool:
    """Compare a Vapi saved Bearer credential without leaking token values."""

    expected_token = expected_token.strip()
    if not authorization or not expected_token:
        return False
    scheme, separator, presented = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer":
        return False
    presented = presented.strip()
    return bool(presented) and hmac.compare_digest(presented, expected_token)


def _assistant_response(settings: Any) -> dict[str, Any]:
    """Return a user-independent transient assistant configuration."""

    return {
        "assistant": {
            "name": settings.vapi_assistant_name,
            "firstMessage": settings.vapi_assistant_first_message,
            "model": {
                "provider": settings.vapi_model_provider,
                "model": settings.vapi_model,
                "messages": [
                    {
                        "role": "system",
                        "content": settings.vapi_assistant_system_prompt,
                    }
                ],
            },
            "voice": {
                "provider": settings.vapi_voice_provider,
                "voiceId": settings.vapi_voice_id,
            },
        }
    }


def _reject_response() -> JSONResponse:
    """Return Vapi's pre-conversation error response without enumeration."""

    return JSONResponse(content={"error": GENERIC_REJECTION}, status_code=200)


@router.post("/webhook/vapi", response_model=None)
def vapi_webhook(
    request: Request,
    payload: Any = Body(...),  # noqa: B008
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),  # noqa: B008
) -> dict[str, Any] | JSONResponse:
    """Authenticate Vapi, authorize inbound calls, and return dynamic config."""

    settings = request.app.state.settings
    if not settings.vapi_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    if not _check_bearer_token(
        authorization,
        expected_token=settings.vapi_webhook_bearer_token,
    ):
        log_event(logger, logging.WARNING, "vapi_webhook_auth_failed")
        raise HTTPException(status_code=401, detail="Invalid webhook credentials")

    try:
        message = VapiServerRequest.model_validate(payload).message
    except ValidationError:
        log_event(logger, logging.WARNING, "vapi_server_message_invalid")
        raise HTTPException(status_code=400, detail="Malformed Vapi server message") from None

    if message.type == END_OF_CALL_REPORT:
        if message.call is not None:
            ended = end_vapi_call(
                db,
                provider_call_id=message.call.id,
                now=request.app.state.clock(),
            )
            log_event(
                logger,
                logging.INFO,
                "vapi_call_ended",
                call_id=message.call.id,
                authorization_ended=ended,
            )
        return {"ok": True}

    if message.type != ASSISTANT_REQUEST:
        log_event(logger, logging.DEBUG, "vapi_server_message_ignored", message_type=message.type)
        return {"ok": True}

    if message.call is None:
        log_event(logger, logging.INFO, "vapi_call_rejected", reason="caller_missing")
        return _reject_response()
    caller_phone = message.call.caller_phone_number
    if not caller_phone:
        log_event(logger, logging.INFO, "vapi_call_rejected", reason="phone_missing")
        return _reject_response()

    result = authorize_incoming_vapi_call(
        db,
        provider_call_id=message.call.id,
        caller_phone_number=caller_phone,
        now=request.app.state.clock(),
        ttl_seconds=settings.vapi_call_authorization_ttl_seconds,
    )
    if not result.authorized:
        log_event(
            logger,
            logging.INFO,
            "vapi_call_rejected",
            call_id=message.call.id,
            reason=result.outcome,
        )
        return _reject_response()

    log_event(
        logger,
        logging.INFO,
        "vapi_call_authorized",
        call_id=message.call.id,
    )
    return _assistant_response(settings)


__all__ = ["router", "vapi_webhook"]
