from __future__ import annotations

import logging
import re
import socket
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from reminder_agent.infrastructure.observability import get_logger, log_event

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


class TelegramVoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_id: str
    file_unique_id: str | None = None
    duration: int = Field(gt=0)
    mime_type: str | None = None
    file_size: int | None = Field(default=None, gt=0)


class TelegramMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message_id: int
    from_user: TelegramUser = Field(alias="from")
    chat: TelegramChat
    text: str | None = None
    contact: TelegramContact | None = None
    voice: TelegramVoice | None = None
    reply_to_message: TelegramMessage | None = None


class TelegramCallbackQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    from_user: TelegramUser = Field(alias="from")
    data: str | None = None
    message: TelegramMessage | None = None


class TelegramUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    update_id: int
    message: TelegramMessage | None = None
    callback_query: TelegramCallbackQuery | None = None


@dataclass(frozen=True)
class TelegramDeliveryResult:
    provider_message_id: str | None


class TelegramClient(Protocol):
    def send_message(
        self, chat_id: int, text: str, *, reply_markup: dict[str, Any] | None = None
    ) -> TelegramDeliveryResult | None: ...

    def answer_callback_query(self, callback_query_id: str) -> None: ...

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> TelegramDeliveryResult | None: ...

    def delete_message(self, chat_id: int, message_id: int) -> None: ...

    def get_file_path(self, file_id: str) -> str: ...

    def download_file(self, file_path: str, *, max_bytes: int) -> bytes: ...


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
        self._token = token
        self._api_base = f"https://api.telegram.org/bot{token}"
        self._url = f"{self._api_base}/sendMessage"
        self._file_base = f"https://api.telegram.org/file/bot{token}"
        self._timeout = timeout_seconds

    def get_file_path(self, file_id: str) -> str:
        """Resolve a Telegram file id without exposing it in application logs."""
        if not file_id.strip():
            raise TelegramAPIError("Telegram file id is empty", failure_kind="invalid_media")
        try:
            response = httpx.post(
                f"{self._api_base}/getFile",
                json={"file_id": file_id},
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                "Telegram file lookup failed",
                failure_kind="network",
                cause=self._safe_cause(exc),
            ) from exc
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise TelegramAPIError(
                "Telegram returned an invalid file response",
                status_code=response.status_code,
                failure_kind="invalid_response",
                cause=self._safe_cause(exc),
            ) from exc
        result = payload.get("result") if isinstance(payload, dict) else None
        file_path = result.get("file_path") if isinstance(result, dict) else None
        if (
            isinstance(payload, dict)
            and response.is_success
            and payload.get("ok") is True
            and isinstance(file_path, str)
        ):
            return file_path
        raise TelegramAPIError(
            "Telegram did not return a usable file path",
            status_code=response.status_code,
            telegram_description=self._safe_description(
                payload.get("description") if isinstance(payload, dict) else None
            ),
            failure_kind="http_error" if not response.is_success else "invalid_response",
        )

    def download_file(self, file_path: str, *, max_bytes: int) -> bytes:
        """Download a bounded Telegram file into memory for immediate STT processing."""
        if not file_path or max_bytes < 1:
            raise TelegramAPIError(
                "Telegram media request is invalid", failure_kind="invalid_media"
            )
        try:
            with httpx.stream(
                "GET",
                f"{self._file_base}/{file_path.lstrip('/')}",
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                    raise TelegramAPIError(
                        "Telegram voice note is too large", failure_kind="too_large"
                    )
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise TelegramAPIError(
                            "Telegram voice note is too large", failure_kind="too_large"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except TelegramAPIError:
            raise
        except httpx.HTTPStatusError as exc:
            raise TelegramAPIError(
                "Telegram rejected the media download",
                status_code=exc.response.status_code,
                failure_kind="http_error",
            ) from exc
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                "Telegram media download failed",
                failure_kind="network",
                cause=self._safe_cause(exc),
            ) from exc

    def _redact_text(self, value: str) -> str:
        """Remove the configured token from provider-controlled diagnostic text."""

        redacted = value.replace(self._token, "<redacted>")
        return re.sub(r"/bot[^/\s]+", "/bot<redacted>", redacted)

    def _safe_cause(self, error: Exception) -> str | None:
        cause = str(error)
        if not cause:
            return None
        return self._redact_text(cause)

    def _safe_description(self, value: object) -> str | None:
        return self._redact_text(value) if isinstance(value, str) else None

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
            payload = response.json()
        except (TypeError, ValueError) as exc:
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
        if not isinstance(payload, dict):
            raise TelegramAPIError(
                "Telegram returned an invalid response",
                status_code=response.status_code,
                failure_kind="invalid_response",
            )

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
                telegram_description=self._safe_description(payload.get("description")),
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
                telegram_description=self._safe_description(payload.get("description")),
                failure_kind="rate_limited" if response.status_code == 429 else "api_error",
                retry_after_seconds=retry_after,
            )
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("message_id") is None:
            raise TelegramAPIError(
                "Telegram returned an invalid response",
                status_code=response.status_code,
                failure_kind="invalid_response",
            )
        provider_message_id = result.get("message_id")
        log_event(
            logger,
            logging.INFO,
            "telegram_api_message_accepted",
            provider_host=host,
            provider_message_id=(
                str(provider_message_id) if provider_message_id is not None else None
            ),
        )
        return TelegramDeliveryResult(
            provider_message_id=(
                str(provider_message_id) if provider_message_id is not None else None
            )
        )

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> TelegramDeliveryResult:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        result = self._call_mutation("editMessageText", payload)
        provider_message_id = result.get("message_id") if isinstance(result, dict) else None
        return TelegramDeliveryResult(
            provider_message_id=(
                str(provider_message_id) if provider_message_id is not None else None
            )
        )

    def delete_message(self, chat_id: int, message_id: int) -> None:
        self._call_mutation(
            "deleteMessage",
            {"chat_id": chat_id, "message_id": message_id},
        )

    def _call_mutation(self, method: str, payload: dict[str, Any]) -> Any:
        """Call a Telegram mutation endpoint and normalize provider failures."""

        try:
            response = httpx.post(
                f"{self._api_base}/{method}",
                json=payload,
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                f"Telegram {method} request failed",
                failure_kind="network",
                cause=self._safe_cause(exc),
            ) from exc
        try:
            body = response.json()
        except (TypeError, ValueError) as exc:
            raise TelegramAPIError(
                f"Telegram returned an invalid {method} response",
                status_code=response.status_code,
                failure_kind="invalid_response",
                cause=self._safe_cause(exc),
            ) from exc
        if not isinstance(body, dict) or not response.is_success or body.get("ok") is not True:
            parameters = body.get("parameters") if isinstance(body, dict) else None
            retry_after = (
                parameters.get("retry_after")
                if isinstance(parameters, dict)
                and isinstance(parameters.get("retry_after"), int)
                and parameters.get("retry_after") > 0
                else None
            )
            raise TelegramAPIError(
                f"Telegram rejected {method}",
                status_code=response.status_code,
                telegram_description=self._safe_description(
                    body.get("description") if isinstance(body, dict) else None
                ),
                failure_kind=(
                    "rate_limited"
                    if response.status_code == 429
                    else "http_error" if not response.is_success else "api_error"
                ),
                retry_after_seconds=retry_after,
            )
        return body.get("result")

    def answer_callback_query(self, callback_query_id: str) -> None:
        """Stop Telegram's inline-button spinner after a callback is handled."""

        url = f"https://api.telegram.org/bot{self._token}/answerCallbackQuery"
        try:
            response = httpx.post(
                url,
                json={"callback_query_id": callback_query_id},
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                "Telegram callback acknowledgement failed",
                failure_kind="network",
                cause=self._safe_cause(exc),
            ) from exc
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise TelegramAPIError(
                "Telegram returned an invalid callback acknowledgement",
                status_code=response.status_code,
                failure_kind="invalid_response",
                cause=self._safe_cause(exc),
            ) from exc
        if (
            not response.is_success
            or not isinstance(payload, dict)
            or payload.get("ok") is not True
        ):
            raise TelegramAPIError(
                "Telegram rejected the callback acknowledgement",
                status_code=response.status_code,
                telegram_description=self._safe_description(
                    payload.get("description") if isinstance(payload, dict) else None
                ),
                failure_kind="http_error" if not response.is_success else "api_error",
            )
