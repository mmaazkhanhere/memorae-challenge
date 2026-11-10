from __future__ import annotations

import httpx
import pytest

from reminder_agent.channels.telegram.provider import HttpTelegramClient, TelegramAPIError


class _Response:
    status_code = 200
    is_success = True
    headers: dict[str, str] = {}

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self):
        return self._payload


def test_file_lookup_returns_provider_path(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: _Response({"ok": True, "result": {"file_path": "voice/a.ogg"}}),
    )

    path = HttpTelegramClient("token").get_file_path("file-id")

    assert path == "voice/a.ogg"


def test_file_download_streams_and_enforces_limit(monkeypatch) -> None:
    class StreamResponse:
        status_code = 200
        headers = {"content-length": "8"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            return iter([b"OggS", b"1234"])

    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: StreamResponse())
    assert HttpTelegramClient("token").download_file("voice/a.ogg", max_bytes=8) == b"OggS1234"

    with pytest.raises(TelegramAPIError) as exc_info:
        HttpTelegramClient("token").download_file("voice/a.ogg", max_bytes=4)
    assert exc_info.value.failure_kind == "too_large"
