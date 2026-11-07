from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select

from reminder_agent.db.models import (
    ConversationSession,
    IdempotencyRecord,
    MemoryFact,
    Notification,
    Reminder,
    ReminderOccurrence,
)
from reminder_agent.transcription import TranscriptionError, TranscriptionResult


HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


@dataclass
class FakeTranscriber:
    text: str
    calls: int = 0
    fail: bool = False

    def transcribe(self, audio: bytes, *, filename: str, media_type: str) -> TranscriptionResult:
        self.calls += 1
        assert audio.startswith(b"OggS")
        assert filename == "voice-note.ogg"
        assert media_type in {"audio/ogg", "audio/opus"}
        if self.fail:
            raise TranscriptionError("no speech", kind="no_speech")
        return TranscriptionResult(text=self.text)


def _voice_update(update_id: int, *, file_id: str = "voice-file", duration: int = 3) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": 1001, "first_name": "Ada"},
            "chat": {"id": 2001, "type": "private"},
            "voice": {
                "file_id": file_id,
                "file_unique_id": f"unique-{file_id}",
                "duration": duration,
                "mime_type": "audio/ogg",
                "file_size": 128,
            },
        },
    }


def _post(client, payload: dict):
    return client.post("/webhook/telegram", json=payload, headers=HEADERS)


def _onboard(client, update_factory) -> None:
    assert _post(client, update_factory(100)).status_code == 200
    assert _post(client, update_factory(101, text="Asia/Karachi")).status_code == 200


def test_voice_note_creates_and_confirms_existing_reminder_service(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    transcriber = FakeTranscriber("Remind me in five minutes to call the bank")
    api_client.app.state.transcriber = transcriber

    response = _post(api_client, _voice_update(102))

    assert response.status_code == 200
    assert transcriber.calls == 1
    assert telegram_client.downloaded_file_ids == ["voice-file"]
    assert "Telegram" in telegram_client.messages[-1][1]
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    assert reminder.task == "call the bank"
    assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 3


def test_voice_note_replay_does_not_transcribe_or_create_again(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    transcriber = FakeTranscriber("Remind me in five minutes to call the bank")
    api_client.app.state.transcriber = transcriber
    payload = _voice_update(103)

    assert _post(api_client, payload).status_code == 200
    message_count = len(telegram_client.messages)
    assert _post(api_client, payload).status_code == 200

    assert transcriber.calls == 1
    assert len(telegram_client.messages) == message_count
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1


def test_voice_note_uses_persistent_clarification_for_missing_time(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    transcriber = FakeTranscriber("Remind me to call the bank")
    api_client.app.state.transcriber = transcriber

    assert _post(api_client, _voice_update(104)).status_code == 200
    assert "when" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    session = db.scalar(select(ConversationSession))
    assert session is not None and session.status == "AWAITING_CLARIFICATION"

    assert _post(api_client, update_factory(105, text="tomorrow at 9 AM")).status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1


def test_failed_transcription_is_text_retry_and_does_not_mutate(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    transcriber = FakeTranscriber("", fail=True)
    api_client.app.state.transcriber = transcriber

    assert _post(api_client, _voice_update(106)).status_code == 200

    assert "couldn't understand" in telegram_client.messages[-1][1].lower()
    assert transcriber.calls == 1
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 3


def test_voice_note_reuses_list_move_cancel_and_memory_services(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    transcriber = FakeTranscriber("unused")
    api_client.app.state.transcriber = transcriber
    assert (
        _post(
            api_client,
            update_factory(200, text="Remind me tomorrow at 9 AM to call the bank"),
        ).status_code
        == 200
    )

    transcriber.text = "What reminders do I have tomorrow?"
    assert _post(api_client, _voice_update(201)).status_code == 200
    assert "call the bank" in telegram_client.messages[-1][1]

    transcriber.text = "Move the bank reminder to tomorrow at 10 AM"
    assert _post(api_client, _voice_update(202)).status_code == 200
    assert "Moved" in telegram_client.messages[-1][1]

    transcriber.text = "Cancel the bank reminder"
    assert _post(api_client, _voice_update(203)).status_code == 200
    assert db.scalar(select(Reminder)).status == "CANCELED"

    transcriber.text = "Remember my gate code is 4512"
    assert _post(api_client, _voice_update(204)).status_code == 200
    fact = db.scalar(select(MemoryFact))
    assert fact is not None and "4512" in fact.content


def test_voice_note_done_and_snooze_use_delivered_occurrence_context(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    transcriber = FakeTranscriber("unused")
    api_client.app.state.transcriber = transcriber
    assert (
        _post(
            api_client,
            update_factory(300, text="Remind me tomorrow at 9 AM to call the bank"),
        ).status_code
        == 200
    )
    occurrence = db.scalar(select(ReminderOccurrence))
    notification = db.scalar(select(Notification))
    assert occurrence is not None and notification is not None
    occurrence.status = "NOTIFIED"
    notification.status = "SENT"
    notification.provider_message_id = "delivered-1"
    db.commit()

    transcriber.text = "Done"
    assert _post(api_client, _voice_update(301)).status_code == 200
    db.refresh(occurrence)
    assert occurrence.status == "COMPLETED"

    assert (
        _post(
            api_client,
            update_factory(302, text="Remind me tomorrow at 10 AM to email the bank"),
        ).status_code
        == 200
    )
    second_occurrence = db.scalars(
        select(ReminderOccurrence).order_by(ReminderOccurrence.created_at.desc())
    ).first()
    second_notification = db.scalars(
        select(Notification).order_by(Notification.created_at.desc())
    ).first()
    assert second_occurrence is not None and second_notification is not None
    second_occurrence.status = "NOTIFIED"
    second_notification.status = "SENT"
    second_notification.provider_message_id = "delivered-2"
    db.commit()

    transcriber.text = "10 more minutes"
    assert _post(api_client, _voice_update(303)).status_code == 200
    db.refresh(second_occurrence)
    assert second_occurrence.status == "PENDING"


def test_invalid_voice_media_is_rejected_before_transcription_or_mutation(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    transcriber = FakeTranscriber("Remind me in five minutes to call the bank")
    api_client.app.state.transcriber = transcriber
    telegram_client.file_bytes = b"not-an-ogg-file"

    assert _post(api_client, _voice_update(400)).status_code == 200

    assert transcriber.calls == 0
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert "couldn't understand" in telegram_client.messages[-1][1].lower()
