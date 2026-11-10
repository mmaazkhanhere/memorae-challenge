"""Telegram voice-media adaptation before reusable command processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.infrastructure.transcription import (
    AudioTranscriber,
    TranscriptionError,
    is_ogg_voice_note,
    log_transcription_failure,
)

logger = get_logger()


class VoiceMediaError(ValueError):
    """Provider-neutral invalid voice media error."""


@dataclass(frozen=True)
class TelegramVoiceResult:
    """Normalized transcript or adapter-owned user-safe failure."""

    transcript: str | None = None
    failure_text: str | None = None
    failure_kind: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.transcript is not None


def transcribe_telegram_voice(
    *,
    voice: Any,
    telegram_client: Any,
    transcriber: AudioTranscriber | None,
    update_id: int,
    max_bytes: int,
    max_duration_seconds: int,
    log_transcript: bool = False,
) -> TelegramVoiceResult:
    """Download, validate, and transcribe one Telegram voice payload.

    This adapter owns Telegram media and provider failures. It returns a
    normalized transcript or the same user-safe failure text previously staged
    by the onboarding facade; it never opens a database transaction.
    """

    if voice.duration > max_duration_seconds or (
        voice.file_size is not None and voice.file_size > max_bytes
    ):
        log_event(
            logger,
            logging.WARNING,
            "telegram_voice_rejected",
            update_id=update_id,
            reason="size_or_duration_limit",
        )
        return TelegramVoiceResult(
            failure_text="That voice note is too large or too long. Please send a shorter note.",
            failure_kind="size_or_duration_limit",
        )
    if voice.mime_type is not None and voice.mime_type not in {"audio/ogg", "audio/opus"}:
        log_event(
            logger,
            logging.WARNING,
            "telegram_voice_rejected",
            update_id=update_id,
            reason="unsupported_mime_type",
        )
        return TelegramVoiceResult(
            failure_text="Please send a Telegram voice note, not another audio format.",
            failure_kind="unsupported_mime_type",
        )
    if telegram_client is None or transcriber is None:
        log_event(
            logger,
            logging.ERROR,
            "telegram_voice_processing_failed",
            update_id=update_id,
            failure_kind="dependency_unavailable",
        )
        return TelegramVoiceResult(
            failure_text="Voice notes are temporarily unavailable. Please try text instead.",
            failure_kind="dependency_unavailable",
        )
    try:
        log_event(
            logger,
            logging.INFO,
            "telegram_voice_download_started",
            update_id=update_id,
        )
        file_path = telegram_client.get_file_path(voice.file_id)
        audio = telegram_client.download_file(file_path, max_bytes=max_bytes)
        log_event(
            logger,
            logging.INFO,
            "telegram_voice_download_succeeded",
            update_id=update_id,
            downloaded_bytes=len(audio),
        )
        if not is_ogg_voice_note(audio):
            raise VoiceMediaError("Voice note format is invalid")
        log_event(
            logger,
            logging.INFO,
            "telegram_voice_transcription_started",
            update_id=update_id,
            audio_bytes=len(audio),
        )
        result = transcriber.transcribe(
            audio,
            filename="voice-note.ogg",
            media_type=voice.mime_type or "audio/ogg",
        )
    except (VoiceMediaError, TranscriptionError, ValueError) as exc:
        if isinstance(exc, TranscriptionError):
            log_transcription_failure(update_id=update_id, error=exc)
        else:
            log_event(
                logger,
                logging.WARNING,
                "telegram_voice_rejected",
                update_id=update_id,
                failure_kind=getattr(exc, "failure_kind", "invalid_media"),
            )
        return TelegramVoiceResult(
            failure_text="I couldn't understand that voice note. Please try again or send the request as text.",
            failure_kind=getattr(exc, "kind", None)
            or getattr(exc, "failure_kind", None)
            or "invalid_media",
        )
    except Exception:
        log_event(
            logger,
            logging.ERROR,
            "telegram_voice_processing_failed",
            update_id=update_id,
            failure_kind="provider_error",
            exc_info=True,
        )
        return TelegramVoiceResult(
            failure_text="I couldn't process that voice note. Please try again or send the request as text.",
            failure_kind="provider_error",
        )

    log_event(
        logger,
        logging.INFO,
        "telegram_voice_transcription_succeeded",
        update_id=update_id,
        transcript_length=len(result.text),
        transcript=result.text if log_transcript else None,
    )
    return TelegramVoiceResult(transcript=result.text)
