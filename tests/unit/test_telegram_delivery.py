from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from reminder_agent.application.response_staging import ReminderReply
from reminder_agent.channels.telegram.provider import (
    HttpTelegramClient,
    TelegramAPIError,
    _connect_failure_kind,
)
from reminder_agent.channels.telegram.webhook import (
    _send_reply_with_retry,
    _split_telegram_text,
    _telegram_error_is_retryable,
)


def test_http_telegram_client_returns_provider_message_id(monkeypatch):
    response = SimpleNamespace(
        status_code=200,
        is_success=True,
        json=lambda: {"ok": True, "result": {"message_id": 987}},
    )
    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", lambda *args, **kwargs: response
    )

    result = HttpTelegramClient("token").send_message(2001, "Reminder: call the bank")

    assert result.provider_message_id == "987"


def test_http_telegram_client_supports_edit_and_delete(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        result = {"message_id": 987} if url.endswith("editMessageText") else True
        return SimpleNamespace(status_code=200, is_success=True, json=lambda: {"ok": True, "result": result})

    monkeypatch.setattr("reminder_agent.channels.telegram.provider.httpx.post", post)
    client = HttpTelegramClient("token")

    edited = client.edit_message_text(2001, 987, "Updated", reply_markup={"inline_keyboard": []})
    client.delete_message(2001, 987)

    assert edited.provider_message_id == "987"
    assert calls[0][0].endswith("editMessageText")
    assert calls[0][1]["reply_markup"] == {"inline_keyboard": []}
    assert calls[1][0].endswith("deleteMessage")


def test_telegram_error_cause_redacts_bot_token():
    client = HttpTelegramClient("secret-token")

    cause = client._safe_cause(
        RuntimeError("request failed for https://api.telegram.org/botsecret-token/sendMessage")
    )

    assert cause == "request failed for https://api.telegram.org/bot<redacted>/sendMessage"


def test_http_telegram_client_preserves_retry_after(monkeypatch):
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.telegram.org/botsecret/sendMessage"),
        json={"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 19}},
    )
    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", lambda *args, **kwargs: response
    )

    with pytest.raises(TelegramAPIError) as raised:
        HttpTelegramClient("secret").send_message(2001, "Reminder")

    assert raised.value.status_code == 429
    assert raised.value.retry_after_seconds == 19
    assert raised.value.failure_kind == "rate_limited"


def test_dns_connect_failure_is_classified_as_safe_to_retry():
    import socket

    error = RuntimeError("connect failed")
    error.__cause__ = socket.gaierror(-2, "name resolution failed")

    assert _connect_failure_kind(error) == "dns_error"


def test_split_telegram_text_respects_limit_and_word_boundaries():
    chunks = _split_telegram_text("alpha " * 20, max_length=20)

    assert all(len(chunk) <= 20 for chunk in chunks)
    assert "".join(f"{chunk} " for chunk in chunks).strip() == ("alpha " * 20).strip()


def test_long_reply_is_delivered_as_ordered_chunks():
    class Client:
        def __init__(self):
            self.messages = []

        def send_message(self, chat_id, text, **kwargs):
            self.messages.append((chat_id, text, kwargs))

    client = Client()
    reply = ReminderReply(chat_id=42, text="word " * 2000)

    _send_reply_with_retry(
        client,
        reply,
        update_id=123,
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert len(client.messages) > 1
    assert all(len(message[1]) <= 4096 for message in client.messages)
    assert "".join(message[1] for message in client.messages).replace(" ", "").strip() == (
        reply.text.replace(" ", "").strip()
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"ok": True},
        {"ok": True, "result": []},
        {"ok": True, "result": {}},
    ],
)
def test_malformed_telegram_success_response_is_rejected(monkeypatch, payload) -> None:
    response = SimpleNamespace(status_code=200, is_success=True, json=lambda: payload)
    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", lambda *args, **kwargs: response
    )

    with pytest.raises(TelegramAPIError) as raised:
        HttpTelegramClient("secret-token").send_message(2001, "Reminder")

    assert raised.value.status_code == 200
    assert raised.value.failure_kind == "invalid_response"


def test_telegram_ok_false_http_success_is_treated_as_provider_rejection(monkeypatch) -> None:
    response = SimpleNamespace(
        status_code=200,
        is_success=True,
        json=lambda: {"ok": False, "description": "chat is unavailable"},
    )
    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", lambda *args, **kwargs: response
    )

    with pytest.raises(TelegramAPIError) as raised:
        HttpTelegramClient("secret-token").send_message(2001, "Reminder")

    assert raised.value.status_code == 200
    assert raised.value.failure_kind == "api_error"
    assert raised.value.telegram_description == "chat is unavailable"


@pytest.mark.parametrize(
    ("exception", "failure_kind"),
    [
        (httpx.ConnectTimeout("connect timeout"), "connect_timeout"),
        (httpx.ReadTimeout("read timeout"), "read_timeout"),
        (httpx.WriteTimeout("write timeout"), "write_timeout"),
        (httpx.ConnectError("connection refused"), "connect_error"),
    ],
)
def test_transport_failures_are_classified(monkeypatch, exception, failure_kind) -> None:
    def raise_transport_error(*args, **kwargs):
        raise exception

    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", raise_transport_error
    )

    with pytest.raises(TelegramAPIError) as raised:
        HttpTelegramClient("secret-token").send_message(2001, "Reminder")

    assert raised.value.failure_kind == failure_kind
    assert raised.value.status_code is None


@pytest.mark.parametrize(
    ("status_code", "payload", "failure_kind", "retryable"),
    [
        (403, {"ok": False, "description": "bot was blocked"}, "http_error", False),
        (429, {"ok": False, "description": "slow down"}, "rate_limited", True),
        (500, {"ok": False, "description": "upstream error"}, "http_error", False),
    ],
)
def test_http_failure_classification_distinguishes_retryable_rate_limits(
    monkeypatch, status_code, payload, failure_kind, retryable
) -> None:
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.telegram.org/botsecret-token/sendMessage"),
        json=payload,
    )
    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", lambda *args, **kwargs: response
    )

    with pytest.raises(TelegramAPIError) as raised:
        HttpTelegramClient("secret-token").send_message(2001, "Reminder")

    assert raised.value.failure_kind == failure_kind
    assert _telegram_error_is_retryable(raised.value) is retryable


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ConnectTimeout("https://api.telegram.org/botsecret-token/sendMessage"),
        httpx.ReadTimeout("Bearer secret-token"),
        httpx.WriteTimeout("request body includes secret-token"),
        httpx.ConnectError("POST /botsecret-token/sendMessage failed"),
        httpx.RequestError("telegram rejected secret-token"),
    ],
)
def test_transport_exception_causes_redact_tokens_in_all_supported_formats(
    monkeypatch, exception
) -> None:
    def raise_transport_error(*args, **kwargs):
        raise exception

    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", raise_transport_error
    )

    with pytest.raises(TelegramAPIError) as raised:
        HttpTelegramClient("secret-token").send_message(2001, "Reminder")

    assert "secret-token" not in (raised.value.cause or "")
    assert "secret-token" not in str(raised.value)


def test_provider_descriptions_and_invalid_json_causes_redact_tokens(monkeypatch) -> None:
    rejected = SimpleNamespace(
        status_code=200,
        is_success=True,
        json=lambda: {"ok": False, "description": "invalid botsecret-token credential"},
    )
    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", lambda *args, **kwargs: rejected
    )

    with pytest.raises(TelegramAPIError) as provider_error:
        HttpTelegramClient("secret-token").send_message(2001, "Reminder")
    assert "secret-token" not in (provider_error.value.telegram_description or "")

    def malformed_json():
        raise ValueError("response contains secret-token")

    malformed = SimpleNamespace(status_code=200, is_success=True, json=malformed_json)
    monkeypatch.setattr(
        "reminder_agent.channels.telegram.provider.httpx.post", lambda *args, **kwargs: malformed
    )
    with pytest.raises(TelegramAPIError) as invalid_response:
        HttpTelegramClient("secret-token").send_message(2001, "Reminder")
    assert "secret-token" not in (invalid_response.value.cause or "")


def test_long_reply_stops_after_a_permanent_chunk_failure():
    class FailsOnSecondChunk:
        def __init__(self):
            self.attempted_chunks: list[str] = []

        def send_message(self, chat_id, text, **kwargs):
            self.attempted_chunks.append(text)
            if len(self.attempted_chunks) == 2:
                raise TelegramAPIError("forbidden", status_code=403, failure_kind="http_error")

    reply = ReminderReply(chat_id=42, text="word " * 2_000)
    expected_chunks = _split_telegram_text(reply.text)
    client = FailsOnSecondChunk()

    with pytest.raises(TelegramAPIError, match="forbidden"):
        _send_reply_with_retry(
            client,
            reply,
            update_id=124,
            max_attempts=3,
            retry_delay_seconds=0,
        )

    assert len(expected_chunks) > 2
    assert client.attempted_chunks == expected_chunks[:2]
