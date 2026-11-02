from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from reminder_agent.telegram import HttpTelegramClient, TelegramAPIError, _connect_failure_kind
from reminder_agent.api.telegram import _send_reply_with_retry, _split_telegram_text
from reminder_agent.reminders import ReminderReply


def test_http_telegram_client_returns_provider_message_id(monkeypatch):
    response = SimpleNamespace(
        status_code=200,
        is_success=True,
        json=lambda: {"ok": True, "result": {"message_id": 987}},
    )
    monkeypatch.setattr("reminder_agent.telegram.httpx.post", lambda *args, **kwargs: response)

    result = HttpTelegramClient("token").send_message(2001, "Reminder: call the bank")

    assert result.provider_message_id == "987"


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
    monkeypatch.setattr("reminder_agent.telegram.httpx.post", lambda *args, **kwargs: response)

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
