from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from reminder_agent.observability import get_logger, log_event

logger = get_logger()


class TranscriptionError(RuntimeError):
    """A safe, user-facing transcription failure category."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None


class AudioTranscriber(Protocol):
    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        media_type: str,
    ) -> TranscriptionResult: ...


class GroqWhisperTranscriber:
    """Small provider adapter; it never receives user identity or command authority."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float) -> None:
        if not api_key.strip():
            raise ValueError("STT API key is required")
        self._url = f"{base_url.rstrip('/')}/audio/transcriptions"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        media_type: str,
    ) -> TranscriptionResult:
        if not audio:
            raise TranscriptionError("No audio was received", kind="empty_audio")
        try:
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                data={"model": self._model, "response_format": "json"},
                files={"file": (filename, audio, media_type)},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise TranscriptionError("Transcription timed out", kind="timeout") from exc
        except httpx.RequestError as exc:
            raise TranscriptionError(
                "Transcription provider is unavailable", kind="network"
            ) from exc
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise TranscriptionError(
                "Transcription provider returned invalid data", kind="invalid_response"
            ) from exc
        if not response.is_success:
            raise TranscriptionError(
                "Transcription provider rejected the audio", kind="provider_error"
            )
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionError("No speech was detected", kind="no_speech")
        transcript = text.strip()
        if len(transcript) > 16_000:
            raise TranscriptionError("The voice note was too long to process", kind="too_long")
        language = payload.get("language") if isinstance(payload, dict) else None
        return TranscriptionResult(text=transcript, language=language)


def is_ogg_voice_note(audio: bytes) -> bool:
    """Cheap content guard before sending untrusted bytes to an external provider."""
    return audio.startswith(b"OggS")


def log_transcription_failure(*, update_id: int, error: TranscriptionError) -> None:
    log_event(
        logger,
        logging.WARNING,
        "telegram_voice_transcription_failed",
        update_id=update_id,
        failure_kind=error.kind,
    )
