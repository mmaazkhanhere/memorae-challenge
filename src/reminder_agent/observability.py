from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from colorama import Fore, Style, just_fix_windows_console


LOGGER_NAME = "reminder_agent"


class JsonFormatter(logging.Formatter):
    def __init__(self, include_traceback: bool = False) -> None:
        super().__init__()
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event_name", record.getMessage())
        context = getattr(record, "context", {})
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event,
            "message": record.getMessage(),
        }
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
            payload["exception_message"] = str(record.exc_info[1])
            if self.include_traceback:
                payload["exception_traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class ReadableFormatter(logging.Formatter):
    _level_colours = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def __init__(self, *, use_color: bool = True, include_traceback: bool = False) -> None:
        super().__init__()
        self.use_color = use_color
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event_name", record.getMessage())
        context = getattr(record, "context", {})
        fields = " ".join(
            f"{key}={value!r}" for key, value in context.items() if value is not None
        )
        suffix = f" {fields}" if fields else ""
        if record.exc_info:
            suffix += f" error={record.exc_info[0].__name__}: {record.exc_info[1]}"
            if self.include_traceback:
                suffix += f"\n{self.formatException(record.exc_info)}"
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        message = event.replace("_", " ")
        if not self.use_color:
            return f"{timestamp} {record.levelname:<8} {message}{suffix}"
        failure_event = event.endswith("_failed") or event.endswith("_error")
        if record.levelno >= logging.ERROR or failure_event:
            # Keep the entire error/failure line visibly red, including the
            # diagnostic context, so provider and application failures stand out.
            return f"{Fore.RED}{timestamp} {record.levelname:<8} {message}{suffix}{Style.RESET_ALL}"
        colour = self._level_colours.get(record.levelno, Fore.WHITE)
        return (
            f"{Fore.LIGHTBLACK_EX}{timestamp}{Style.RESET_ALL} "
            f"{colour}{record.levelname:<8}{Style.RESET_ALL} "
            f"{Style.BRIGHT}{message}{Style.RESET_ALL}{suffix}"
        )


def configure_logging(
    level: str = "INFO",
    log_format: str = "text",
    include_traceback: bool = False,
    use_color: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False
    if log_format.lower() == "json":
        formatter: logging.Formatter = JsonFormatter(include_traceback=include_traceback)
    else:
        if use_color:
            just_fix_windows_console()
        formatter = ReadableFormatter(
            use_color=use_color,
            include_traceback=include_traceback,
        )
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


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
    """Return useful provider/schema diagnostics without logging raw payloads."""

    details: dict[str, Any] = {"exception_type": type(exc).__name__}
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
