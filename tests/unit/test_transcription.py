from __future__ import annotations

import httpx
import pytest

from reminder_agent.transcription import (
    GroqWhisperTranscriber,
    TranscriptionError,
    is_ogg_voice_note,
)


def test_ogg_guard_rejects_non_ogg_bytes() -> None:
    assert is_ogg_voice_note(b"OggS\x00")
    assert not is_ogg_voice_note(b"RIFF\x00")


def test_transcriber_validates_empty_audio() -> None:
    transcriber = GroqWhisperTranscriber("https://stt.example/v1", "key", "whisper", 5)
    with pytest.raises(TranscriptionError) as exc_info:
        transcriber.transcribe(b"", filename="voice.ogg", media_type="audio/ogg")
    assert exc_info.value.kind == "empty_audio"


def test_transcriber_posts_multipart_and_returns_text(monkeypatch) -> None:
    captured: dict = {}

    class Response:
        is_success = True

        def json(self):
            return {"text": " Remind me tomorrow at nine to call the bank ", "language": "en"}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    result = GroqWhisperTranscriber(
        "https://stt.example/v1", "key", "whisper", 5
    ).transcribe(b"OggS-audio", filename="voice.ogg", media_type="audio/ogg")

    assert result.text == "Remind me tomorrow at nine to call the bank"
    assert captured["url"] == "https://stt.example/v1/audio/transcriptions"
    assert captured["files"]["file"][0] == "voice.ogg"


def test_transcriber_maps_network_failure(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", fake_post)
    transcriber = GroqWhisperTranscriber("https://stt.example/v1", "key", "whisper", 5)
    with pytest.raises(TranscriptionError) as exc_info:
        transcriber.transcribe(b"OggS-audio", filename="voice.ogg", media_type="audio/ogg")
    assert exc_info.value.kind == "network"
