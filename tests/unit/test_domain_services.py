from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select

from reminder_agent.application import conversation_coordinator as reminders
from reminder_agent.application.conversation_coordinator import handle_reminder_text
from reminder_agent.channels.telegram.provider import (
    TelegramAPIError,
    TelegramDeliveryResult,
    TelegramUpdate,
)
from reminder_agent.features.intent import CreateOneTimeReminderIntent, Interpretation
from reminder_agent.features.onboarding.message_templates import ONBOARDING_MESSAGES
from reminder_agent.features.onboarding.workflow import handle_update
from reminder_agent.features.reminders.cancel.decisions import CancellationScope, apply_cancellation
from reminder_agent.features.reminders.recurrence_service import materialize_recurring_reminder
from reminder_agent.features.reminders.reschedule.decisions import (
    RescheduleCandidate,
    RescheduleScope,
    apply_reschedule,
    decide_reschedule,
    find_candidates,
    format_candidates,
)
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.infrastructure.runtime.worker import NotificationWorker


class OneTimeInterpreter:
    def __init__(self, due_at: datetime) -> None:
        self.due_at = due_at

    def interpret(self, text: str, *, timezone_name: str, now: datetime) -> Interpretation:
        return Interpretation(
            intent=CreateOneTimeReminderIntent(
                task="call the bank",
                due_at=self.due_at,
                timezone_name=timezone_name,
            )
        )


class RecordingTelegramClient:
    def __init__(self, error: TelegramAPIError | None = None) -> None:
        self.error = error
        self.messages: list[tuple[int, str]] = []

    def send_message(self, chat_id: int, text: str, *, reply_markup=None):
        if self.error is not None:
            raise self.error
        self.messages.append((chat_id, text))
        return TelegramDeliveryResult("telegram-message-1")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def test_reschedule_candidates_are_displayed_in_user_timezone() -> None:
    text = format_candidates(
        [
            RescheduleCandidate(
                reminder_id=UUID(int=0),
                occurrence_id=UUID(int=0),
                task="call the bank",
                due_at=datetime(2026, 11, 3, 4, 0, tzinfo=UTC),
                schedule_type="ONE_TIME",
            )
        ],
        timezone_name="Asia/Karachi",
    )

    assert text == "1. call the bank (currently 2026-11-03 09:00 PKT)"


def _active_user_and_telegram_identity(db, *, telegram_user_id: int, chat_id: int):
    user = User(name=f"User {telegram_user_id}", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.flush()
    identity = ChannelIdentity(
        user_id=user.id,
        channel_type="TELEGRAM",
        external_id=str(telegram_user_id),
        telegram_chat_id=chat_id,
        status="LINKED",
    )
    db.add(identity)
    db.flush()
    return user, identity


def _seed_one_time_reminder(
    db,
    *,
    user: User,
    destination: ChannelIdentity,
    task: str,
    due_at: datetime,
):
    reminder = Reminder(
        user_id=user.id,
        task=task,
        schedule_type="ONE_TIME",
        start_local=due_at.isoformat(),
        timezone="UTC",
        status="ACTIVE",
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=due_at,
        next_due_at=due_at,
        status="PENDING",
    )
    db.add(occurrence)
    db.flush()
    notification = Notification(
        occurrence_id=occurrence.id,
        destination_identity_id=destination.id,
        sequence=1,
        due_at=due_at,
        status="PENDING",
        attempt_count=0,
    )
    db.add(notification)
    db.flush()
    return reminder, occurrence, notification


def test_reminder_creation_service_persists_a_single_graph_and_replays_idempotently(
    session_factory, fixed_now, update_factory
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_user_and_telegram_identity(db, telegram_user_id=1001, chat_id=2001)

    update = TelegramUpdate.model_validate(
        update_factory(700, text="Remind me in five minutes to call the bank")
    )
    interpreter = OneTimeInterpreter(fixed_now + timedelta(minutes=5))
    with session_factory() as db:
        reply = handle_reminder_text(
            db,
            update,
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=interpreter,
            now=fixed_now,
        )

    assert "call the bank" in reply.text
    assert "Telegram" in reply.text
    assert reply.is_duplicate is False
    with session_factory() as db:
        reminder = db.scalar(select(Reminder))
        occurrence = db.scalar(select(ReminderOccurrence))
        notification = db.scalar(select(Notification))
        assert reminder is not None and reminder.task == "call the bank"
        assert occurrence is not None and occurrence.reminder_id == reminder.id
        assert notification is not None and notification.occurrence_id == occurrence.id
        assert notification.destination_identity_id == identity.id

    with session_factory() as db:
        replay = handle_reminder_text(
            db,
            update,
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=interpreter,
            now=fixed_now,
        )
        assert replay.is_duplicate is True
        assert replay.should_send is False
        assert db.scalar(select(func.count()).select_from(Reminder)) == 1
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 1
        assert db.scalar(select(func.count()).select_from(Notification)) == 1


def test_reminder_creation_service_rolls_back_the_full_graph_on_failure(
    session_factory, fixed_now, update_factory, monkeypatch
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_user_and_telegram_identity(db, telegram_user_id=1001, chat_id=2001)

    def fail_after_scheduling(*args, **kwargs):
        raise RuntimeError("simulated response persistence failure")

    monkeypatch.setattr(reminders, "_store_response", fail_after_scheduling)
    with session_factory() as db, pytest.raises(RuntimeError, match="response persistence"):
        handle_reminder_text(
            db,
            TelegramUpdate.model_validate(
                update_factory(701, text="Remind me in five minutes to call the bank")
            ),
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=OneTimeInterpreter(fixed_now + timedelta(minutes=5)),
            now=fixed_now,
        )

    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 0
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_reschedule_services_select_owned_candidates_and_replace_notification(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, destination = _active_user_and_telegram_identity(
            db, telegram_user_id=1001, chat_id=2001
        )
        other_user, other_destination = _active_user_and_telegram_identity(
            db, telegram_user_id=2002, chat_id=3002
        )
        first, first_occurrence, first_notification = _seed_one_time_reminder(
            db,
            user=user,
            destination=destination,
            task="call the bank",
            due_at=fixed_now + timedelta(hours=1),
        )
        _seed_one_time_reminder(
            db,
            user=user,
            destination=destination,
            task="email the bank",
            due_at=fixed_now + timedelta(hours=2),
        )
        _seed_one_time_reminder(
            db,
            user=other_user,
            destination=other_destination,
            task="call the bank privately",
            due_at=fixed_now + timedelta(hours=1),
        )

        candidates = find_candidates(db, user_id=user.id, target_text="bank")
        decision = decide_reschedule(
            db,
            user_id=user.id,
            text="Move the bank reminder to tomorrow at 10 AM",
            timezone_name="UTC",
            now=fixed_now,
        )
        assert len(candidates) == 2
        assert candidates[0].reminder_id == first.id
        assert decision.kind == "CLARIFY"
        assert len(decision.candidate_ids) == 2

        due_at = fixed_now + timedelta(hours=3)
        _, returned_due_at = apply_reschedule(
            db,
            user_id=user.id,
            destination_identity_id=destination.id,
            reminder_id=first.id,
            due_at=due_at,
            scope=RescheduleScope.OCCURRENCE,
            now=fixed_now,
        )
        assert returned_due_at == due_at

    with session_factory() as db:
        occurrence = db.get(ReminderOccurrence, first_occurrence.id)
        notifications = db.scalars(
            select(Notification)
            .where(Notification.occurrence_id == first_occurrence.id)
            .order_by(Notification.sequence)
        ).all()
        assert occurrence is not None
        assert _as_utc(occurrence.next_due_at) == fixed_now + timedelta(hours=3)
        assert [(notification.sequence, notification.status) for notification in notifications] == [
            (first_notification.sequence, "CANCELED"),
            (2, "PENDING"),
        ]


def test_caller_rollback_discards_staged_cancellation_graph_mutation(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, destination = _active_user_and_telegram_identity(
            db, telegram_user_id=1001, chat_id=2001
        )
        reminder, occurrence, notification = _seed_one_time_reminder(
            db,
            user=user,
            destination=destination,
            task="call the bank",
            due_at=fixed_now + timedelta(hours=1),
        )

    with session_factory() as db:
        apply_cancellation(
            db,
            user_id=user.id,
            reminder_id=reminder.id,
            occurrence_id=occurrence.id,
            scope=CancellationScope.OCCURRENCE,
            now=fixed_now,
        )
        db.rollback()

    with session_factory() as db:
        persisted_reminder = db.get(Reminder, reminder.id)
        persisted_occurrence = db.get(ReminderOccurrence, occurrence.id)
        persisted_notification = db.get(Notification, notification.id)
        assert persisted_reminder is not None
        assert persisted_occurrence is not None
        assert persisted_notification is not None
        assert persisted_reminder.status == "ACTIVE"
        assert persisted_occurrence.status == "PENDING"
        assert persisted_notification.status == "PENDING"


def test_recurrence_materialization_service_is_idempotent_and_creates_deliveries(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, destination = _active_user_and_telegram_identity(
            db, telegram_user_id=1001, chat_id=2001
        )
        reminder = Reminder(
            user_id=user.id,
            task="weekly review",
            schedule_type="RECURRING",
            start_local="2026-10-31T10:00",
            timezone="UTC",
            recurrence_rule="FREQ=WEEKLY;BYDAY=SA;BYHOUR=10;BYMINUTE=00",
            status="ACTIVE",
        )
        db.add(reminder)
        db.flush()
        first_created = materialize_recurring_reminder(db, reminder, now=fixed_now, horizon_days=14)
        second_created = materialize_recurring_reminder(
            db, reminder, now=fixed_now, horizon_days=14
        )

        assert first_created > 0
        assert second_created == 0
        occurrences = db.scalars(
            select(ReminderOccurrence).where(ReminderOccurrence.reminder_id == reminder.id)
        ).all()
        notifications = db.scalars(select(Notification)).all()
        assert len(occurrences) == first_created
        assert len(notifications) == first_created
        assert {item.destination_identity_id for item in notifications} == {destination.id}
        assert len({item.recurrence_instance_key for item in occurrences}) == first_created


def test_notification_worker_claims_only_eligible_delivery_and_records_success(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, destination = _active_user_and_telegram_identity(
            db, telegram_user_id=1001, chat_id=2001
        )
        _, _, eligible = _seed_one_time_reminder(
            db,
            user=user,
            destination=destination,
            task="call the bank",
            due_at=fixed_now,
        )
        ineligible_identity = ChannelIdentity(
            user_id=user.id,
            channel_type="TELEGRAM",
            external_id="stale-identity",
            telegram_chat_id=2002,
            status="UNLINKED",
        )
        db.add(ineligible_identity)
        db.flush()
        _, _, ineligible = _seed_one_time_reminder(
            db,
            user=user,
            destination=ineligible_identity,
            task="do not send",
            due_at=fixed_now,
        )

    client = RecordingTelegramClient()
    worker = NotificationWorker(
        session_factory, client, worker_id="unit-worker", clock=lambda: fixed_now
    )
    claims = worker.claim_due_notifications(fixed_now)

    assert [claim.notification_id for claim in claims] == [eligible.id]
    assert worker.deliver(claims[0], fixed_now) == "SENT"
    assert client.messages == [(2001, "Reminder: call the bank")]
    with session_factory() as db:
        assert db.get(Notification, eligible.id).status == "SENT"
        assert db.get(Notification, ineligible.id).status == "PENDING"


def test_notification_worker_records_retryable_delivery_result(session_factory, fixed_now) -> None:
    with session_factory.begin() as db:
        user, destination = _active_user_and_telegram_identity(
            db, telegram_user_id=1001, chat_id=2001
        )
        _, _, notification = _seed_one_time_reminder(
            db,
            user=user,
            destination=destination,
            task="call the bank",
            due_at=fixed_now,
        )

    worker = NotificationWorker(
        session_factory,
        RecordingTelegramClient(
            TelegramAPIError("rate limited", status_code=429, failure_kind="rate_limited")
        ),
        worker_id="unit-worker",
        retry_base_delay_seconds=45,
        clock=lambda: fixed_now,
    )
    claim = worker.claim_due_notifications(fixed_now)[0]

    assert worker.deliver(claim, fixed_now) == "PENDING"
    with session_factory() as db:
        refreshed = db.get(Notification, notification.id)
        assert refreshed is not None
        assert refreshed.status == "PENDING"
        assert refreshed.attempt_count == 1
        assert _as_utc(refreshed.due_at) == fixed_now + timedelta(seconds=45)
        assert refreshed.last_error_kind == "rate_limited"


def test_onboarding_service_transitions_and_preserves_phone_ownership(
    session_factory, fixed_now, update_factory
) -> None:
    def apply(update: dict):
        with session_factory() as db:
            return handle_update(db, TelegramUpdate.model_validate(update), now=fixed_now)

    def onboard(telegram_user_id: int, chat_id: int, update_id: int) -> None:
        assert (
            apply(
                update_factory(update_id, user_id=telegram_user_id, chat_id=chat_id, text="/start")
            )
            is not None
        )
        completed = apply(
            update_factory(
                update_id + 1,
                user_id=telegram_user_id,
                chat_id=chat_id,
                text="UTC",
            )
        )
        assert completed is not None
        assert completed.text in {
            template.format(name="Ada", timezone_name="UTC")
            for template in ONBOARDING_MESSAGES["onboarding_complete"]
        }

    onboard(1001, 2001, 800)
    first_contact = update_factory(
        802,
        user_id=1001,
        chat_id=2001,
        text=None,
        contact={"phone_number": "+442083661177", "user_id": 1001},
    )
    first_reply = apply(first_contact)
    assert first_reply is not None
    assert first_reply.text in set(ONBOARDING_MESSAGES["phone_linked"])

    onboard(2002, 3002, 810)
    second_reply = apply(
        update_factory(
            812,
            user_id=2002,
            chat_id=3002,
            text=None,
            contact={"phone_number": "+442083661177", "user_id": 2002},
        )
    )

    assert second_reply is not None
    assert second_reply.text in set(ONBOARDING_MESSAGES["phone_conflict"])
    with session_factory() as db:
        users = db.scalars(select(User).order_by(User.name)).all()
        phones = db.scalars(
            select(ChannelIdentity).where(ChannelIdentity.channel_type == "PHONE")
        ).all()
        assert [user.onboarding_status for user in users] == ["ACTIVE", "ACTIVE"]
        assert len(phones) == 1
        assert phones[0].external_id == "+442083661177"
