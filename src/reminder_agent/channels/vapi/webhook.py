"""Vapi server webhook adapter for inbound authorization and VS-17 tools."""

from __future__ import annotations

import hmac
import json
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
from reminder_agent.application.voice_call_workflow import process_vapi_reminder_tool
from reminder_agent.channels.vapi.contracts import VapiServerRequest, VapiToolCall
from reminder_agent.infrastructure.database.session import get_db
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.idempotency import IdempotencyConflict

router = APIRouter()
logger = get_logger()

ASSISTANT_REQUEST = "assistant-request"
END_OF_CALL_REPORT = "end-of-call-report"
TOOL_CALLS = "tool-calls"
REMINDER_TOOL_NAME = "submit_reminder_utterance"
GENERIC_REJECTION = "We could not verify this call. Please link this number in Telegram."
TOOL_ARGUMENT_ERROR = "I couldn't understand that reminder request."


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

    tool_instructions = (
        "Use submit_reminder_utterance for every request to create a reminder. "
        "Pass the caller's complete request as utterance without extracting or changing "
        "the task, date, time, recurrence, or channel. Wait for the tool result before "
        "claiming success, and repeat the returned confirmation exactly."
    )
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
                    },
                    {"role": "system", "content": tool_instructions},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": REMINDER_TOOL_NAME,
                            "description": "Submit a caller's complete reminder request.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "utterance": {
                                        "type": "string",
                                        "description": "The caller's complete reminder request verbatim.",
                                    }
                                },
                                "required": ["utterance"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            },
            "voice": {
                "provider": settings.vapi_voice_provider,
                "voiceId": settings.vapi_voice_id,
            },
        }
    }


def _tool_utterance(tool_call: VapiToolCall) -> str | None:
    """Accept only the raw utterance field; all reminder fields stay server-owned."""

    arguments = tool_call.function.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict) or set(arguments) != {"utterance"}:
        return None
    utterance = arguments.get("utterance")
    if not isinstance(utterance, str):
        return None
    normalized = utterance.strip()
    return normalized if normalized and len(normalized) <= 4000 else None


def _tool_call_response(
    request: Request,
    *,
    db: Session,
    call_id: str | None,
    tool_call: VapiToolCall,
) -> dict[str, str]:
    """Execute one tool call and render Vapi's flat-string result contract."""

    if call_id is None:
        return {"toolCallId": tool_call.id, "error": "I couldn't verify this call."}
    if tool_call.type != "function" or tool_call.function.name != REMINDER_TOOL_NAME:
        return {"toolCallId": tool_call.id, "error": "That voice action is not supported yet."}
    utterance = _tool_utterance(tool_call)
    if utterance is None:
        return {"toolCallId": tool_call.id, "error": TOOL_ARGUMENT_ERROR}
    try:
        outcome = process_vapi_reminder_tool(
            db,
            call_id=call_id,
            tool_call_id=tool_call.id,
            utterance=utterance,
            interpreter=request.app.state.reminder_interpreter,
            intent_router=request.app.state.intent_router,
            now=request.app.state.clock(),
            session_ttl_seconds=request.app.state.settings.conversation_session_ttl_seconds,
        )
    except IdempotencyConflict:
        return {
            "toolCallId": tool_call.id,
            "error": "That tool call ID was already used for a different request.",
        }
    except Exception:
        log_event(
            logger,
            logging.ERROR,
            "vapi_tool_call_failed",
            call_id=call_id,
            tool_call_id=tool_call.id,
            exc_info=True,
        )
        return {"toolCallId": tool_call.id, "error": "I couldn't create that reminder."}
    if outcome.error is not None:
        return {"toolCallId": tool_call.id, "error": outcome.error}
    assert outcome.response is not None
    return {"toolCallId": tool_call.id, "result": outcome.response.content}


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

    if message.type == TOOL_CALLS:
        call_id = message.call.id if message.call is not None else None
        results = [
            _tool_call_response(
                request,
                db=db,
                call_id=call_id,
                tool_call=tool_call,
            )
            for tool_call in message.tool_call_list
        ]
        return {"results": results}

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
