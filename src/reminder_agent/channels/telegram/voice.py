"""Telegram voice-media transport adapter."""

from reminder_agent.application.voice_transcription import (
    TelegramVoiceResult,
    transcribe_telegram_voice,
)

__all__ = ["TelegramVoiceResult", "transcribe_telegram_voice"]
