from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


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
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event_name", record.getMessage())
        context = getattr(record, "context", {})
        fields = " ".join(
            f"{key}={value!r}" for key, value in context.items() if value is not None
        )
        suffix = f" {fields}" if fields else ""
        if record.exc_info:
            suffix += f" error={record.exc_info[0].__name__}: {record.exc_info[1]}"
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return f"{timestamp} {record.levelname:<5} {event}{suffix}"


def configure_logging(
    level: str = "INFO",
    log_format: str = "text",
    include_traceback: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False
    if log_format.lower() == "json":
        formatter: logging.Formatter = JsonFormatter(include_traceback=include_traceback)
    else:
        formatter = ReadableFormatter()
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


def request_id() -> str:
    return uuid4().hex
