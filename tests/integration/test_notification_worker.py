from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from threading import Event, Thread

import pytest
from sqlalchemy import select

from reminder_agent.channels.telegram.callbacks import parse_callback_data
from reminder_agent.channels.telegram.provider import TelegramAPIError, TelegramDeliveryResult
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.infrastructure.runtime.worker import ConfirmationWorker, NotificationWorker


def _seed_notification(db, now: datetime, *, due_at: datetime | None = None, **kwargs):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.flush()
    identity = ChannelIdentity(
        user_id=user.id,
        channel_type="TELEGRAM",
        external_id=user.id.hex,
        telegram_chat_id=2001,
        status="LINKED",
    )
    db.add(identity)
    db.flush()
    reminder = Reminder(
        user_id=user.id,
        task="call the bank",
        schedule_type="ONE_TIME",
        start_local=now.isoformat(),
        timezone="UTC",
        status=kwargs.pop("reminder_status", "ACTIVE"),
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=due_at or now,
        next_due_at=due_at or now,
        status=kwargs.pop("occurrence_status", "PENDING"),
    )
    db.add(occurrence)
    db.flush()
    notification = Notification(
        occurrence_id=occurrence.id,
        destination_identity_id=identity.id,
        sequence=1,
        due_at=due_at or now,
        status=kwargs.pop("notification_status", "PENDING"),
        **kwargs,
    )
    db.add(notification)
    db.commit()
    return notification.id, occurrence.id


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class RecordingTelegramClient:
    def __init__(self, result_id: str = "tg-42", error: TelegramAPIError | None = None):
        self.messages: list[tuple[int, str]] = []
        self.reply_markups: list[dict | None] = []
        self.result_id = result_id
        self.error = error

    def send_message(self, chat_id: int, text: str, *, reply_markup=None):
        if self.error is not None:
            raise self.error
        self.messages.append((chat_id, text))
        self.reply_markups.append(reply_markup)
        return TelegramDeliveryResult(self.result_id)


def _worker(session_factory, client, now, **kwargs):
    return NotificationWorker(
        session_factory,
        client,
        worker_id=kwargs.pop("worker_id", "worker-a"),
        clock=lambda: now,
        retry_base_delay_seconds=kwargs.pop("retry_base_delay_seconds", 30),
        **kwargs,
    )


def _seed_failed_confirmation(db, now: datetime, *, attempt_count: int = 1):
    record = IdempotencyRecord(
        source="TELEGRAM",
        idempotency_key=f"confirmation-{attempt_count}-{now.timestamp()}",
        operation="ONBOARDING_START",
        request_hash="a" * 64,
        status="SUCCEEDED",
        delivery_status="UNKNOWN",
        delivery_attempt_count=attempt_count,
        delivery_next_attempt_at=now,
        response={"chat_id": 2001, "text": "Welcome", "reply_markup": None},
    )
    db.add(record)
    db.commit()
    return record.id


def test_due_notification_is_sent_and_records_provider_result(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, occurrence_id = _seed_notification(db, fixed_now)

    client = RecordingTelegramClient()
    assert (
        _worker(session_factory, client, fixed_now, callback_secret="callback-secret").run_once()
        == 1
    )

    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        occurrence = db.get(ReminderOccurrence, occurrence_id)
        assert notification is not None
        assert notification.status == "SENT"
        assert notification.provider_message_id == "tg-42"
        assert notification.sent_at is not None and _utc(notification.sent_at) == fixed_now
        assert notification.claimed_by is None
        assert occurrence is not None and occurrence.status == "NOTIFIED"
    assert client.messages == [(2001, "Reminder: call the bank")]
    assert client.reply_markups[0]["inline_keyboard"][0][0]["text"] == "Done"
    callback = client.reply_markups[0]["inline_keyboard"][0][0]["callback_data"]
    assert parse_callback_data(callback, "callback-secret") == occurrence_id


def test_delivery_grace_defers_a_notification_until_after_the_grace_window(
    session_factory, fixed_now
):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)

    client = RecordingTelegramClient()
    worker = _worker(
        session_factory,
        client,
        fixed_now,
        delivery_grace_seconds=15,
    )

    assert worker.run_once() == 0
    assert client.messages == []
    assert worker.claim_due_notifications(fixed_now + timedelta(seconds=14)) == []
    claims = worker.claim_due_notifications(fixed_now + timedelta(seconds=15))
    assert len(claims) == 1
    assert worker.deliver(claims[0], fixed_now + timedelta(seconds=15)) == "SENT"
    with session_factory() as db:
        assert db.get(Notification, notification_id).status == "SENT"


def test_confirmation_worker_retries_failed_confirmation(session_factory, fixed_now):
    with session_factory() as db:
        record_id = _seed_failed_confirmation(db, fixed_now)
    client = RecordingTelegramClient()
    worker = ConfirmationWorker(
        session_factory,
        client,
        worker_id="confirmation-worker",
        clock=lambda: fixed_now,
    )

    assert worker.run_once() == 1
    assert client.messages == [(2001, "Welcome")]
    with session_factory() as db:
        record = db.get(IdempotencyRecord, record_id)
        assert record is not None
        assert record.delivery_status == "SENT"
        assert record.delivery_next_attempt_at is None


def test_worker_restart_before_due_does_not_lose_notification(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(
            db, fixed_now, due_at=fixed_now + timedelta(minutes=5)
        )
    client = RecordingTelegramClient()
    worker = _worker(session_factory, client, fixed_now)
    assert worker.run_once() == 0
    assert client.messages == []

    due_worker = _worker(session_factory, client, fixed_now + timedelta(minutes=5, seconds=1))
    assert due_worker.run_once() == 1
    with session_factory() as db:
        assert db.get(Notification, notification_id).status == "SENT"


def test_competing_workers_cannot_claim_same_notification(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)
    first = _worker(session_factory, RecordingTelegramClient(), fixed_now, worker_id="one")
    second_client = RecordingTelegramClient()
    second = _worker(session_factory, second_client, fixed_now, worker_id="two")

    first_claims = first.claim_due_notifications(fixed_now)
    second_claims = second.claim_due_notifications(fixed_now)
    assert len(first_claims) == 1
    assert second_claims == []
    assert first.deliver(first_claims[0], fixed_now) == "SENT"
    assert second_client.messages == []
    with session_factory() as db:
        assert db.get(Notification, notification_id).status == "SENT"


def test_canceled_or_completed_parent_is_never_sent(session_factory, fixed_now):
    cases = (("CANCELED", "PENDING"), ("ACTIVE", "COMPLETED"))
    client = RecordingTelegramClient()
    for reminder_status, occurrence_status in cases:
        with session_factory() as db:
            _seed_notification(
                db,
                fixed_now,
                reminder_status=reminder_status,
                occurrence_status=occurrence_status,
            )
    assert _worker(session_factory, client, fixed_now).run_once() == 0
    assert client.messages == []
    with session_factory() as db:
        assert db.scalar(select(Notification).where(Notification.status == "PENDING")) is not None


def test_rate_limit_is_retried_with_backoff(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)
    client = RecordingTelegramClient(
        error=TelegramAPIError("rate limited", status_code=429, failure_kind="http_error")
    )
    outcome = _worker(session_factory, client, fixed_now).run_once()
    assert outcome == 1
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "PENDING"
        assert _utc(notification.due_at) == fixed_now + timedelta(seconds=30)
        assert notification.attempt_count == 1
        assert notification.last_error_kind == "http_error"


def test_permanent_telegram_error_is_failed(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)
    client = RecordingTelegramClient(
        error=TelegramAPIError("blocked", status_code=403, failure_kind="http_error")
    )
    _worker(session_factory, client, fixed_now).run_once()
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "FAILED"
        assert notification.last_error_kind == "http_error"


def test_uncertain_timeout_is_unknown_when_retries_are_disabled(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)
    client = RecordingTelegramClient(
        error=TelegramAPIError("timeout", failure_kind="network_timeout")
    )
    _worker(session_factory, client, fixed_now, max_attempts=1).run_once()
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "UNKNOWN"
        assert notification.last_error_kind == "network_timeout"


def test_uncertain_timeout_gets_two_delayed_retries_then_unknown(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)
    client = RecordingTelegramClient(
        error=TelegramAPIError("response timeout", failure_kind="read_timeout")
    )
    worker = _worker(session_factory, client, fixed_now)

    assert worker.run_once() == 1
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "PENDING"
        notification.due_at = fixed_now
        db.commit()
    assert worker.run_once() == 1
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "PENDING"
        notification.due_at = fixed_now
        db.commit()
    assert worker.run_once() == 1
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "UNKNOWN"
        assert notification.attempt_count == 3


def test_connection_timeout_retries_safely_then_fails(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)
    client = RecordingTelegramClient(
        error=TelegramAPIError("connect timeout", failure_kind="connect_timeout")
    )
    worker = _worker(session_factory, client, fixed_now, max_attempts=2)
    assert worker.run_once() == 1
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "PENDING"
        notification.due_at = fixed_now
        db.commit()
    assert worker.run_once() == 1
    with session_factory() as db:
        assert db.get(Notification, notification_id).status == "FAILED"


def test_rate_limit_retry_after_is_respected(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)
    client = RecordingTelegramClient(
        error=TelegramAPIError(
            "rate limited",
            status_code=429,
            failure_kind="rate_limited",
            retry_after_seconds=47,
        )
    )
    _worker(session_factory, client, fixed_now).run_once()
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "PENDING"
        assert _utc(notification.due_at) == fixed_now + timedelta(seconds=47)


def test_expired_claim_is_recovered_as_unknown(session_factory, fixed_now):
    with session_factory() as db:
        notification_id, _ = _seed_notification(
            db,
            fixed_now,
            notification_status="SENDING",
            claimed_by="dead-worker",
            claimed_until=fixed_now - timedelta(seconds=1),
        )
    client = RecordingTelegramClient()
    worker = _worker(session_factory, client, fixed_now)
    assert worker.recover_expired_claims(fixed_now) == 1
    with session_factory() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == "UNKNOWN"
        assert notification.last_error_kind == "lease_expired"
        assert notification.claimed_by is None
    assert client.messages == []


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires TEST_DATABASE_URL pointing to PostgreSQL for row-lock concurrency",
)
def test_postgres_workers_use_skip_locked_concurrency(session_factory, fixed_now):
    """A contending transaction is skipped rather than blocking on the locked row."""
    with session_factory() as db:
        notification_id, _ = _seed_notification(db, fixed_now)

    # Hold the same notification row lock that the first worker would hold while its
    # claim transaction is still open. The second worker must skip it immediately.
    holder = session_factory()
    holder.begin()
    holder.execute(
        select(Notification.id).where(Notification.id == notification_id).with_for_update()
    ).scalar_one()

    second = _worker(
        session_factory,
        RecordingTelegramClient(),
        fixed_now,
        worker_id="contender",
    )
    started = Event()
    result: list[list] = []

    def contend() -> None:
        started.set()
        result.append(second.claim_due_notifications(fixed_now))

    thread = Thread(target=contend)
    thread.start()
    assert started.wait(timeout=1)
    thread.join(timeout=2)
    try:
        assert not thread.is_alive(), "SKIP LOCKED contender blocked on the held row"
        assert result == [[]]
    finally:
        holder.rollback()
        holder.close()
