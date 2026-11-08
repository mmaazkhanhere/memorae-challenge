from __future__ import annotations

import pytest

from reminder_agent.application import conversation_workflow as reminders
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.features.onboarding.workflow import (
    IdempotencyConflict as PublicIdempotencyConflict,
)
from reminder_agent.infrastructure.db.models import IdempotencyRecord
from reminder_agent.shared.idempotency import IdempotencyConflict as SharedIdempotencyConflict
from reminder_agent.shared.idempotency import request_hash


def test_idempotency_conflict_identity_remains_publicly_compatible() -> None:
    assert PublicIdempotencyConflict is SharedIdempotencyConflict
    assert reminders.IdempotencyConflict is SharedIdempotencyConflict


def test_voice_transcript_overlay_does_not_change_request_hash() -> None:
    payload = {
        "update_id": 901,
        "message": {
            "message_id": 901,
            "from": {"id": 1001, "first_name": "Ada"},
            "chat": {"id": 2001, "type": "private"},
            "voice": {
                "file_id": "voice-file",
                "file_unique_id": "unique-voice-file",
                "duration": 3,
                "mime_type": "audio/ogg",
                "file_size": 128,
            },
        },
    }
    with_transcript = {
        **payload,
        "message": {**payload["message"], "text": "transcribed reminder"},
    }

    assert request_hash(TelegramUpdate.model_validate(payload)) == request_hash(
        TelegramUpdate.model_validate(with_transcript)
    )


def test_reminder_conflict_raises_the_shared_exception(session_factory, update_factory) -> None:
    update = TelegramUpdate.model_validate(update_factory(900, text="first request"))
    conflicting_update = TelegramUpdate.model_validate(
        update_factory(900, text="different request")
    )

    with session_factory.begin() as db:
        db.add(
            IdempotencyRecord(
                source="TELEGRAM",
                idempotency_key=str(update.update_id),
                operation="CREATE_REMINDER",
                request_hash=request_hash(update),
            )
        )

    with (
        session_factory() as db,
        pytest.raises(
            SharedIdempotencyConflict,
            match="Telegram update was reused with different content",
        ),
    ):
        reminders._existing_record(db, conflicting_update)
