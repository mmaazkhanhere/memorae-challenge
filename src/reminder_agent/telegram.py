from __future__ import annotations

from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field


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


class TelegramMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message_id: int
    from_user: TelegramUser = Field(alias="from")
    chat: TelegramChat
    text: str | None = None


class TelegramUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    update_id: int
    message: TelegramMessage | None = None


class TelegramClient(Protocol):
    def send_message(self, chat_id: int, text: str) -> None:
        ...


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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.telegram_description = telegram_description
        self.failure_kind = failure_kind
        self.cause = cause


class HttpTelegramClient:
    def __init__(self, token: str, timeout_seconds: float = 10.0) -> None:
        if not token:
            raise TelegramAPIError("Telegram bot token is not configured")
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._timeout = timeout_seconds

    def send_message(self, chat_id: int, text: str) -> None:
        try:
            response = httpx.post(
                self._url,
                json={"chat_id": chat_id, "text": text},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise TelegramAPIError(
                "Telegram message delivery timed out",
                failure_kind="network_timeout",
                cause=str(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                "Telegram message delivery failed",
                failure_kind="network",
                cause=str(exc),
            ) from exc

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise TelegramAPIError(
                "Telegram returned an invalid response",
                status_code=response.status_code,
                failure_kind="invalid_response",
                cause=str(exc),
            ) from exc

        if not response.is_success:
            raise TelegramAPIError(
                "Telegram rejected the message",
                status_code=response.status_code,
                telegram_description=payload.get("description"),
                failure_kind="http_error",
            )
        if payload.get("ok") is not True:
            raise TelegramAPIError(
                "Telegram message delivery was rejected",
                status_code=response.status_code,
                telegram_description=payload.get("description"),
                failure_kind="api_error",
            )
