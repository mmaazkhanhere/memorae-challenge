from datetime import UTC, datetime

from reminder_agent.channels.telegram.callbacks import callback_data, parse_callback_data
from reminder_agent.features.reminders.complete.operations import complete_occurrence
from reminder_agent.infrastructure.db.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)


def _seed(db, *, recurring=False, status="NOTIFIED"):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.flush()
    identity = ChannelIdentity(
        user_id=user.id,
        channel_type="TELEGRAM",
        external_id="1001",
        telegram_chat_id=2001,
        status="LINKED",
    )
    db.add(identity)
    db.flush()
    reminder = Reminder(
        user_id=user.id,
        task="call the bank",
        schedule_type="RECURRING" if recurring else "ONE_TIME",
        start_local="2026-11-03T12:00:00",
        timezone="UTC",
        recurrence_rule="FREQ=WEEKLY;BYDAY=TU;BYHOUR=12;BYMINUTE=00" if recurring else None,
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=datetime.now(UTC),
        next_due_at=datetime.now(UTC),
        status=status,
    )
    db.add(occurrence)
    db.flush()
    notification = Notification(
        occurrence_id=occurrence.id,
        destination_identity_id=identity.id,
        sequence=1,
        due_at=occurrence.scheduled_at,
        status="SENT",
        provider_message_id="42",
        sent_at=occurrence.scheduled_at,
    )
    db.add(notification)
    db.commit()
    return user, identity, reminder, occurrence, notification


def test_completes_one_time_occurrence_and_parent(db):
    user, identity, reminder, occurrence, _ = _seed(db)
    result = complete_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
    )
    assert result.outcome == "COMPLETED"
    assert db.get(ReminderOccurrence, occurrence.id).status == "COMPLETED"
    assert db.get(Reminder, reminder.id).status == "COMPLETED"


def test_completing_recurring_occurrence_keeps_series_active(db):
    user, identity, reminder, occurrence, _ = _seed(db, recurring=True)
    result = complete_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
    )
    assert result.outcome == "COMPLETED"
    assert db.get(Reminder, reminder.id).status == "ACTIVE"


def test_repeated_completion_is_safe(db):
    user, identity, reminder, occurrence, _ = _seed(db)
    complete_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
    )
    result = complete_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
    )
    assert result.outcome == "ALREADY_COMPLETED"


def test_notified_occurrence_requires_owned_destination(db):
    user, identity, _, occurrence, _ = _seed(db)
    result = complete_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=9999,
        occurrence_id=occurrence.id,
    )
    assert result.outcome == "NOT_DELIVERED"
    assert db.get(ReminderOccurrence, occurrence.id).status == "NOTIFIED"


def test_completion_rechecks_the_delivered_provider_message(db):
    user, identity, _, occurrence, _ = _seed(db)
    result = complete_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
        provider_message_id="not-the-sent-message",
    )
    assert result.outcome == "NOT_DELIVERED"
    assert db.get(ReminderOccurrence, occurrence.id).status == "NOTIFIED"


def test_callback_payload_is_signed_and_tamper_evident(db):
    user, _, _, occurrence, _ = _seed(db)
    value = callback_data(occurrence.id, "callback-secret")
    assert parse_callback_data(value, "callback-secret") == occurrence.id
    assert parse_callback_data(value, "wrong-secret") is None
    altered = value[:-1] + ("0" if value[-1] != "0" else "1")
    assert parse_callback_data(altered, "callback-secret") is None
    assert parse_callback_data("done:not-a-uuid:invalid", "callback-secret") is None
    assert parse_callback_data("x" * 65, "callback-secret") is None
