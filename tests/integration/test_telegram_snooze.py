import os
import subprocess
import sys
from datetime import UTC, datetime
from threading import Barrier, Thread
from time import monotonic, sleep

import pytest
from sqlalchemy import func, select

from reminder_agent.features.reminders.snooze import snooze_occurrence
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    ConversationSession,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.infrastructure.runtime.worker import NotificationWorker

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _post(client, payload):
    return client.post("/webhook/telegram", json=payload, headers=HEADERS)


def _onboard(api_client, db, update_factory, start_id=1000):
    _post(api_client, update_factory(start_id))
    _post(api_client, update_factory(start_id + 1, text="UTC"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert identity is not None
    return identity


def _seed_delivered(db, identity, *, task="call the bank", recurring=False, provider="777"):
    scheduled_at = datetime(2026, 10, 30, 11, 59, tzinfo=UTC)
    reminder = Reminder(
        user_id=identity.user_id,
        task=task,
        schedule_type="RECURRING" if recurring else "ONE_TIME",
        start_local="2026-11-03T12:00:00",
        timezone="UTC",
        recurrence_rule="FREQ=WEEKLY;BYDAY=TU;BYHOUR=12;BYMINUTE=00" if recurring else None,
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=scheduled_at,
        next_due_at=scheduled_at,
        status="NOTIFIED",
    )
    db.add(occurrence)
    db.flush()
    notification = Notification(
        occurrence_id=occurrence.id,
        destination_identity_id=identity.id,
        sequence=1,
        due_at=scheduled_at,
        status="SENT",
        provider_message_id=provider,
        sent_at=scheduled_at,
    )
    db.add(notification)
    db.commit()
    return reminder, occurrence, notification


def _reply_to(payload: dict, provider_message_id: str) -> dict:
    payload["message"]["reply_to_message"] = {
        "message_id": int(provider_message_id),
        "from": {"id": 999},
        "chat": {"id": 2001, "type": "private"},
        "text": "Reminder",
    }
    return payload


def test_snooze_reply_creates_one_replacement_notification_and_preserves_history(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    reminder, occurrence, sent = _seed_delivered(db, identity)

    payload = _reply_to(update_factory(1002, text="10 more minutes"), "777")
    assert _post(api_client, payload).status_code == 200
    message = telegram_client.messages[-1][1]
    assert "call the bank" in message
    assert "2026-10-30 12:10 UTC" in message

    db.refresh(occurrence)
    notifications = db.scalars(
        select(Notification)
        .where(Notification.occurrence_id == occurrence.id)
        .order_by(Notification.sequence)
    ).all()
    assert reminder.status == "ACTIVE"
    assert occurrence.status == "PENDING"
    assert len(notifications) == 2
    assert notifications[0].status == "SENT"
    assert notifications[0].provider_message_id == "777"
    assert notifications[1].status == "PENDING"
    assert notifications[1].sequence == 2


def test_bare_snooze_requires_an_unambiguous_delivered_occurrence(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, first, _ = _seed_delivered(db, identity, task="first", provider="780")
    _, second, _ = _seed_delivered(db, identity, task="second", provider="781")

    assert _post(api_client, update_factory(1002, text="10 more minutes")).status_code == 200
    assert "reminder" in telegram_client.messages[-1][1].lower()
    session = db.scalar(
        select(ConversationSession).where(ConversationSession.user_id == identity.user_id)
    )
    assert session is not None
    assert session.pending_clarification["operation"] == "SNOOZE_OCCURRENCE"
    db.refresh(first)
    db.refresh(second)
    assert first.status == "NOTIFIED"
    assert second.status == "NOTIFIED"


def test_bare_snooze_collapses_history_before_applying_candidate_limit(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, first, first_sent = _seed_delivered(db, identity, task="first", provider="790")
    db.add(
        Notification(
            occurrence_id=first.id,
            destination_identity_id=identity.id,
            sequence=2,
            due_at=first_sent.due_at,
            status="SENT",
            provider_message_id="791",
            sent_at=first_sent.sent_at,
        )
    )
    db.commit()
    _, second, _ = _seed_delivered(db, identity, task="second", provider="792")

    assert _post(api_client, update_factory(1002, text="10 more minutes")).status_code == 200
    assert "reminder" in telegram_client.messages[-1][1].lower()
    db.refresh(first)
    db.refresh(second)
    assert first.status == "NOTIFIED"
    assert second.status == "NOTIFIED"


def test_snooze_clarification_accepts_target_then_duration(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, first, _ = _seed_delivered(db, identity, task="call Jack", provider="782")
    _, second, _ = _seed_delivered(db, identity, task="call James", provider="783")

    _post(api_client, update_factory(1002, text="10 more minutes"))
    assert _post(api_client, update_factory(1003, text="call James")).status_code == 200
    assert "call James" in telegram_client.messages[-1][1]
    db.refresh(first)
    db.refresh(second)
    assert first.status == "NOTIFIED"
    assert second.status == "PENDING"


def test_snooze_missing_duration_is_clarified_then_completed(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, _ = _seed_delivered(db, identity, provider="784")

    assert _post(api_client, update_factory(1002, text="Snooze")).status_code == 200
    assert any(
        word in telegram_client.messages[-1][1].lower() for word in ("remind", "snooze")
    )
    assert _post(api_client, update_factory(1003, text="15 minutes")).status_code == 200
    assert "call the bank" in telegram_client.messages[-1][1]
    db.refresh(occurrence)
    assert occurrence.status == "PENDING"


def test_replayed_snooze_update_does_not_add_a_second_notification(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, _ = _seed_delivered(db, identity, provider="785")
    payload = _reply_to(update_factory(1002, text="10 minutes"), "785")

    _post(api_client, payload)
    message_count = len(telegram_client.messages)
    assert _post(api_client, payload).status_code == 200
    assert len(telegram_client.messages) == message_count
    assert (
        db.scalar(
            select(func.count())
            .select_from(Notification)
            .where(Notification.occurrence_id == occurrence.id)
        )
        == 2
    )
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == "TELEGRAM",
            IdempotencyRecord.idempotency_key == "1002",
        )
    )
    assert record is not None and record.operation == "SNOOZE_OCCURRENCE"


def test_snooze_pending_can_be_safely_canceled_without_mutation(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, _ = _seed_delivered(db, identity, provider="786")

    _post(api_client, update_factory(1002, text="Snooze"))
    assert _post(api_client, update_factory(1003, text="Never mind")).status_code == 200
    assert any(
        word in telegram_client.messages[-1][1].lower() for word in ("reminder", "snooze")
    )
    db.refresh(occurrence)
    session = db.scalar(
        select(ConversationSession).where(ConversationSession.user_id == identity.user_id)
    )
    assert occurrence.status == "NOTIFIED"
    assert session is not None and session.pending_clarification is None


def test_read_only_list_does_not_consume_pending_snooze(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, _ = _seed_delivered(db, identity, provider="789")

    _post(api_client, update_factory(1002, text="Snooze"))
    assert (
        _post(api_client, update_factory(1003, text="What do I have tomorrow?")).status_code == 200
    )
    session = db.scalar(
        select(ConversationSession).where(ConversationSession.user_id == identity.user_id)
    )
    db.refresh(occurrence)
    assert session is not None
    assert session.pending_clarification["operation"] == "SNOOZE_OCCURRENCE"
    assert occurrence.status == "NOTIFIED"


def test_explicit_snooze_supersedes_an_unfinished_creation_context(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, _ = _seed_delivered(db, identity, provider="790")

    _post(api_client, update_factory(1002, text="Remind me to call the bank"))
    assert _post(api_client, update_factory(1003, text="10 minutes")).status_code == 200
    assert "call the bank" in telegram_client.messages[-1][1]
    session = db.scalar(
        select(ConversationSession).where(ConversationSession.user_id == identity.user_id)
    )
    db.refresh(occurrence)
    assert session is not None and session.pending_clarification is None
    assert occurrence.status == "PENDING"


def test_snooze_pending_new_request_supersedes_without_hijacking_it(
    api_client, telegram_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, _ = _seed_delivered(db, identity, provider="787")

    _post(api_client, update_factory(1002, text="Snooze"))
    assert (
        _post(
            api_client,
            update_factory(1003, text="Remind me tomorrow at 9 AM to submit the report"),
        ).status_code
        == 200
    )
    assert "I'll remind you" in telegram_client.messages[-1][1]
    db.refresh(occurrence)
    assert occurrence.status == "NOTIFIED"


def test_worker_delivers_replacement_notification_and_keeps_sent_history(
    api_client, telegram_client, db, session_factory, fixed_now, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, _ = _seed_delivered(db, identity, provider="788")
    payload = _reply_to(update_factory(1002, text="10 minutes"), "788")
    _post(api_client, payload)

    worker = NotificationWorker(
        session_factory,
        telegram_client,
        clock=lambda: fixed_now,
        callback_secret="test-secret",
    )
    assert worker.run_once() == 0
    worker.clock = lambda: fixed_now.replace(minute=10, second=1)
    assert worker.run_once() == 1

    db.refresh(occurrence)
    notifications = db.scalars(
        select(Notification)
        .where(Notification.occurrence_id == occurrence.id)
        .order_by(Notification.sequence)
    ).all()
    assert occurrence.status == "NOTIFIED"
    assert notifications[0].status == "SENT"
    assert notifications[0].provider_message_id == "788"
    assert notifications[1].status == "SENT"
    assert any(text == "Reminder: call the bank" for _chat_id, text in telegram_client.messages)


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL row locking",
)
def test_postgres_concurrent_snoozes_create_one_replacement(
    session_factory, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, sent = _seed_delivered(db, identity, provider="concurrent")
    barrier = Barrier(2)
    outcomes: list[str] = []

    def snooze() -> None:
        with session_factory() as session:
            barrier.wait()
            result = snooze_occurrence(
                session,
                user_id=identity.user_id,
                identity_id=identity.id,
                chat_id=2001,
                occurrence_id=occurrence.id,
                duration_seconds=600,
                provider_message_id=sent.provider_message_id,
                now=datetime(2026, 10, 30, 12, 0, tzinfo=UTC),
            )
            session.commit()
            outcomes.append(result.outcome)

    threads = [Thread(target=snooze) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    db.refresh(occurrence)
    notifications = db.scalars(
        select(Notification)
        .where(Notification.occurrence_id == occurrence.id)
        .order_by(Notification.sequence)
    ).all()
    assert sorted(outcomes) == ["NOT_ELIGIBLE", "SNOOZED"]
    assert occurrence.status == "PENDING"
    assert [(item.sequence, item.status) for item in notifications] == [
        (1, "SENT"),
        (2, "PENDING"),
    ]


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL authorization boundary",
)
def test_postgres_cross_user_snooze_is_rejected(session_factory, db):
    owner = User(name="Owner", timezone="UTC", onboarding_status="ACTIVE")
    attacker = User(name="Attacker", timezone="UTC", onboarding_status="ACTIVE")
    db.add_all([owner, attacker])
    db.flush()
    owner_identity = ChannelIdentity(
        user_id=owner.id,
        channel_type="TELEGRAM",
        external_id="owner-user",
        telegram_chat_id=111111,
        status="LINKED",
    )
    attacker_identity = ChannelIdentity(
        user_id=attacker.id,
        channel_type="TELEGRAM",
        external_id="attacker-user",
        telegram_chat_id=222222,
        status="LINKED",
    )
    db.add_all([owner_identity, attacker_identity])
    db.flush()
    reminder = Reminder(
        user_id=owner.id,
        task="owner private reminder",
        schedule_type="ONE_TIME",
        start_local="2026-11-04T12:00:00",
        timezone="UTC",
    )
    db.add(reminder)
    db.flush()
    now = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=now,
        next_due_at=now,
        status="NOTIFIED",
    )
    db.add(occurrence)
    db.flush()
    sent = Notification(
        occurrence_id=occurrence.id,
        destination_identity_id=owner_identity.id,
        sequence=1,
        due_at=now,
        status="SENT",
        sent_at=now,
        provider_message_id="owner-provider",
    )
    db.add(sent)
    db.commit()

    result = snooze_occurrence(
        db,
        user_id=attacker.id,
        identity_id=attacker_identity.id,
        chat_id=222222,
        occurrence_id=occurrence.id,
        duration_seconds=600,
        provider_message_id=sent.provider_message_id,
        now=now,
    )
    db.commit()
    db.refresh(occurrence)
    assert result.outcome == "NOT_FOUND"
    assert occurrence.status == "NOTIFIED"
    assert (
        db.scalar(
            select(func.count())
            .select_from(Notification)
            .where(Notification.occurrence_id == occurrence.id)
        )
        == 1
    )


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL and a separate worker process",
)
def test_postgres_worker_process_restart_delivers_snooze_within_60_seconds(
    session_factory, api_client, db, update_factory, tmp_path
):
    identity = _onboard(api_client, db, update_factory)
    _, occurrence, sent = _seed_delivered(db, identity, provider="restart-original")
    output_path = tmp_path / "worker-delivery.jsonl"
    ready_path = tmp_path / "worker-ready"
    start_path = tmp_path / "worker-start"
    worker_code = """
import json
import os
import time
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from reminder_agent.infrastructure.runtime.worker import NotificationWorker

class FileTelegramClient:
    def send_message(self, chat_id, text, *, reply_markup=None):
        with open(os.environ['WORKER_OUTPUT'], 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'chat_id': chat_id, 'text': text}) + '\\n')
        return 'restart-provider'

engine = create_engine(os.environ['TEST_DATABASE_URL'], pool_pre_ping=True)
factory = sessionmaker(bind=engine, expire_on_commit=False)
worker = NotificationWorker(
    factory,
    FileTelegramClient(),
    worker_id=os.environ['WORKER_ID'],
    clock=lambda: datetime.now(timezone.utc),
    claim_lease_seconds=60.0,
)
ready_path = os.environ.get('WORKER_READY')
if ready_path:
    open(ready_path, 'a', encoding='utf-8').close()
    while not os.path.exists(os.environ['WORKER_START']):
        time.sleep(0.01)
print(worker.run_once())
"""
    environment = {
        **os.environ,
        "WORKER_OUTPUT": str(output_path),
        "WORKER_ID": "restart-test-worker",
    }
    before_due_environment = {
        **environment,
        "WORKER_READY": str(ready_path),
        "WORKER_START": str(start_path),
    }
    before_due_process = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=before_due_environment,
    )
    try:
        ready_deadline = monotonic() + 30
        while not ready_path.exists() and monotonic() < ready_deadline:
            if before_due_process.poll() is not None:
                stdout, stderr = before_due_process.communicate()
                pytest.fail(f"worker exited before readiness: {stdout=} {stderr=}")
            sleep(0.01)
        assert ready_path.exists(), "worker did not become ready within 30 seconds"

        result = snooze_occurrence(
            db,
            user_id=identity.user_id,
            identity_id=identity.id,
            chat_id=2001,
            occurrence_id=occurrence.id,
            duration_seconds=5,
            provider_message_id=sent.provider_message_id,
            now=datetime.now(UTC),
        )
        assert result.outcome == "SNOOZED"
        assert result.due_at is not None
        due_at = result.due_at
        db.commit()
        start_path.touch()
        before_due_stdout, before_due_stderr = before_due_process.communicate(timeout=30)
    finally:
        if before_due_process.poll() is None:
            before_due_process.kill()
            before_due_process.communicate()
    assert before_due_stdout.strip() == "0", before_due_stderr
    sleep(max(0.0, (due_at - datetime.now(UTC)).total_seconds()) + 0.2)
    after_restart = subprocess.run(
        [sys.executable, "-c", worker_code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert after_restart.stdout.strip() == "1"
    delivered = output_path.read_text(encoding="utf-8").splitlines()
    assert len(delivered) == 1
    db.refresh(occurrence)
    replacement = db.scalar(
        select(Notification).where(
            Notification.occurrence_id == occurrence.id,
            Notification.sequence == 2,
        )
    )
    assert replacement is not None
    assert replacement.status == "SENT"
    assert replacement.sent_at is not None
    latency = (replacement.sent_at - replacement.due_at).total_seconds()
    assert 0 <= latency <= 60
    assert occurrence.status == "NOTIFIED"
