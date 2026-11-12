"""Structured event emission and safe diagnostic helpers."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    exc_info: bool = False,
    **context: Any,
) -> None:
    logger.log(
        level,
        event,
        exc_info=exc_info,
        extra={"event_name": event, "context": context},
    )


def exception_details(exc: BaseException) -> dict[str, Any]:
    """Return useful provider/schema diagnostics without raw payloads."""

    details: dict[str, Any] = {"exception_type": type(exc).__name__}
    message = str(exc).strip().replace("\r", " ").replace("\n", " ")
    if message:
        details["exception_message"] = message[:500]
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        reason_phrase = getattr(response, "reason_phrase", None)
        if status_code is not None:
            details["status_code"] = status_code
        if reason_phrase:
            details["reason_phrase"] = reason_phrase
    request = getattr(exc, "request", None)
    if request is not None:
        method = getattr(request, "method", None)
        url = getattr(request, "url", None)
        host = getattr(url, "host", None) if url is not None else None
        if method:
            details["request_method"] = method
        if host:
            details["request_host"] = host
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            details["validation_error_types"] = [
                item.get("type") for item in errors(include_url=False)
            ]
        except Exception:
            details["validation_error_types"] = ["unavailable"]
    return details


def request_id() -> str:
    return uuid4().hex
