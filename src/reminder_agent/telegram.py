from __future__ import annotations

import logging
import re
import socket
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from reminder_agent.observability import get_logger, log_event

logger = get_logger()


def _connect_failure_kind(error: Exception) -> str:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, socket.gaierror):
            return "dns_error"
        current = current.__cause__ or current.__context__
    return "connect_error"


class TelegramUser(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None


class TelegramChat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    type: str


class TelegramContact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    phone_number: str
    user_id: int | None = None


class TelegramMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message_id: int
    from_user: TelegramUser = Field(alias="from")
    chat: TelegramChat
    text: str | None = None
    contact: TelegramContact | None = None


class TelegramUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    update_id: int
    message: TelegramMessage | None = None


@dataclass(frozen=True)
class TelegramDeliveryResult:
    provider_message_id: str | None


class TelegramClient(Protocol):
    def send_message(
        self, chat_id: int, text: str, *, reply_markup: dict[str, Any] | None = None
    ) -> TelegramDeliveryResult | None: ...


class TelegramAPIError(RuntimeError):
    """Raised when Telegram does not accept an outbound message."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        telegram_description: str | None = None,
        failure_kind: str | None = None,
        cause: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.telegram_description = telegram_description
        self.failure_kind = failure_kind
        self.cause = cause
        self.retry_after_seconds = retry_after_seconds


class HttpTelegramClient:
    def __init__(self, token: str, timeout_seconds: float = 10.0) -> None:
        if not token:
            raise TelegramAPIError("Telegram bot token is not configured")
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._timeout = timeout_seconds

    def _safe_cause(self, error: Exception) -> str | None:
        cause = str(error)
        if not cause:
            return None
        return re.sub(r"/bot[^/\s]+", "/bot<redacted>", cause)

    def send_message(
        self, chat_id: int, text: str, *, reply_markup: dict[str, Any] | None = None
    ) -> TelegramDeliveryResult:
        host = urlsplit(self._url).hostname or "unknown"
        safe_cause = self._safe_cause
        log_event(
            logger,
            logging.DEBUG,
            "telegram_api_request_started",
            provider_host=host,
            timeout_seconds=self._timeout,
            message_length=len(text),
            has_reply_markup=reply_markup is not None,
        )
        try:
            payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
            if reply_markup is not None:
                payload["reply_markup"] = reply_markup
            response = httpx.post(
                self._url,
                json=payload,
                timeout=self._timeout,
            )
        except httpx.ConnectTimeout as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="connect_timeout",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram connection timed out",
                failure_kind="connect_timeout",
                cause=safe_cause(exc),
            ) from exc
        except httpx.ReadTimeout as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="read_timeout",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram response timed out",
                failure_kind="read_timeout",
                cause=safe_cause(exc),
            ) from exc
        except httpx.WriteTimeout as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="write_timeout",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram request write timed out",
                failure_kind="write_timeout",
                cause=safe_cause(exc),
            ) from exc
        except httpx.PoolTimeout as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="pool_timeout",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram connection pool timed out",
                failure_kind="pool_timeout",
                cause=safe_cause(exc),
            ) from exc
        except httpx.TimeoutException as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="network_timeout",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram message delivery timed out",
                failure_kind="network_timeout",
                cause=safe_cause(exc),
            ) from exc
        except httpx.ReadError as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="read_error",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram response could not be read",
                failure_kind="read_error",
                cause=safe_cause(exc),
            ) from exc
        except httpx.WriteError as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="write_error",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram request could not be written",
                failure_kind="write_error",
                cause=safe_cause(exc),
            ) from exc
        except httpx.ConnectError as exc:
            failure_kind = _connect_failure_kind(exc)
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind=failure_kind,
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram connection failed",
                failure_kind=failure_kind,
                cause=safe_cause(exc),
            ) from exc
        except httpx.RequestError as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_request_failed",
                provider_host=host,
                failure_kind="network",
                exception_type=type(exc).__name__,
                cause=safe_cause(exc),
            )
            raise TelegramAPIError(
                "Telegram message delivery failed",
                failure_kind="network",
                cause=safe_cause(exc),
            ) from exc

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            log_event(
                logger,
                logging.ERROR,
                "telegram_api_response_invalid",
                provider_host=host,
                provider_status_code=response.status_code,
                exception_type=type(exc).__name__,
            )
            raise TelegramAPIError(
                "Telegram returned an invalid response",
                status_code=response.status_code,
                failure_kind="invalid_response",
                cause=safe_cause(exc),
            ) from exc

        log_event(
            logger,
            logging.DEBUG,
            "telegram_api_response_received",
            provider_host=host,
            provider_status_code=response.status_code,
            provider_ok=payload.get("ok"),
            has_result=isinstance(payload.get("result"), dict),
        )

        if not response.is_success:
            retry_after = None
            parameters = payload.get("parameters")
            if isinstance(parameters, dict):
                raw_retry_after = parameters.get("retry_after")
                if isinstance(raw_retry_after, int) and raw_retry_after > 0:
                    retry_after = raw_retry_after
            raise TelegramAPIError(
                "Telegram rejected the message",
                status_code=response.status_code,
                telegram_description=payload.get("description"),
                failure_kind="rate_limited" if response.status_code == 429 else "http_error",
                retry_after_seconds=retry_after,
            )
        if payload.get("ok") is not True:
            retry_after = None
            parameters = payload.get("parameters")
            if isinstance(parameters, dict):
                raw_retry_after = parameters.get("retry_after")
                if isinstance(raw_retry_after, int) and raw_retry_after > 0:
                    retry_after = raw_retry_after
            raise TelegramAPIError(
                "Telegram message delivery was rejected",
                status_code=response.status_code,
                telegram_description=payload.get("description"),
                failure_kind="rate_limited" if response.status_code == 429 else "api_error",
                retry_after_seconds=retry_after,
            )
        result = payload.get("result") or {}
        provider_message_id = result.get("message_id")
        return TelegramDeliveryResult(
            provider_message_id=(
                str(provider_message_id) if provider_message_id is not None else None
            )
        )
