from datetime import datetime, timedelta, timezone
import os
from threading import Event, Lock, Thread
from uuid import uuid4

import pytest

from sqlalchemy import func, select

from reminder_agent.db.models import (
    ChannelIdentity,
    ConversationSession,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
)
from reminder_agent.worker import NotificationWorker
from reminder_agent import rescheduling
from reminder_agent.intent import IntentRoute, IntentRouteSuggestion
from reminder_agent.rescheduling import RescheduleConflict, RescheduleScope, apply_reschedule


def post_update(api_client, update):
    return api_client.post(
        "/webhook/telegram",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
    )


def onboard(api_client, update_factory, *, user_id=1001, chat_id=2001, name="Ada", base=900) -> None:
    assert post_update(
        api_client,
        update_factory(base, user_id=user_id, chat_id=chat_id, first_name=name, text="/start"),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(
            base + 1,
            user_id=user_id,
            chat_id=chat_id,
            first_name=name,
            text="UTC",
        ),
    ).status_code == 200


def test_reschedule_updates_existing_reminder_and_replaces_pending_notification(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(902, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200

    response = post_update(
        api_client,
        update_factory(903, text="Move the bank reminder to tomorrow at 10 AM"),
    )

    assert response.status_code == 200
    assert "Moved 'call the bank'" in telegram_client.messages[-1][1]
    assert "10:00" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 1
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.version == 2
    notifications = db.scalars(
        select(Notification).order_by(Notification.sequence)
    ).all()
    assert len(notifications) == 2
    assert notifications[0].status == "CANCELED"
    assert notifications[1].status == "PENDING"
    assert notifications[1].due_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 10, 31, 10, 0, tzinfo=timezone.utc
    )


def test_classifier_routes_indirect_reschedule_to_transactional_handler(
    api_client, telegram_client, db, update_factory
) -> None:
    class RescheduleRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.RESCHEDULE_REQUEST)

    api_client.app.state.intent_router = RescheduleRouter()
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(903, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200

    response = post_update(
        api_client,
        update_factory(904, text="Could you shift the bank reminder to tomorrow at 10 AM?"),
    )

    assert response.status_code == 200
    assert "Moved 'call the bank'" in telegram_client.messages[-1][1]
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.version == 2


def test_reschedule_replay_does_not_apply_twice(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(904, text="Remind me tomorrow at 9 AM to call the bank"))
    update = update_factory(905, text="Move the bank reminder to tomorrow at 10 AM")

    assert post_update(api_client, update).status_code == 200
    assert post_update(api_client, update).status_code == 200

    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.version == 2
    assert db.scalar(select(func.count()).select_from(Notification)) == 2


def test_multiple_matching_reminders_are_not_speculatively_updated(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(906, text="Remind me tomorrow at 9 AM to call the bank"))
    post_update(api_client, update_factory(907, text="Remind me tomorrow at 11 AM to email the bank"))

    response = post_update(
        api_client,
        update_factory(908, text="Move the bank reminder to tomorrow at 10 AM"),
    )

    assert response.status_code == 200
    assert "multiple matching" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(func.count()).select_from(Reminder)) == 2
    assert all(reminder.version == 1 for reminder in db.scalars(select(Reminder)).all())
    assert db.scalar(select(func.count()).select_from(Notification).where(Notification.status == "CANCELED")) == 0


def test_shared_word_does_not_select_the_wrong_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(909, text="Remind me tomorrow at 9 AM to call the bank"))

    response = post_update(
        api_client,
        update_factory(910, text="Move the call dentist reminder to tomorrow at 10 AM"),
    )

    assert response.status_code == 200
    assert "couldn't find" in telegram_client.messages[-1][1].lower()
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.version == 1


def test_multiple_matches_can_be_resolved_without_creating_a_new_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(913, text="Remind me tomorrow at 9 AM to call the bank"))
    post_update(api_client, update_factory(914, text="Remind me tomorrow at 11 AM to email the bank"))
    post_update(api_client, update_factory(915, text="Move the bank reminder to tomorrow at 10 AM"))

    response = post_update(api_client, update_factory(916, text="the first one"))

    assert response.status_code == 200
    assert "Moved 'call the bank'" in telegram_client.messages[-1][1]
    reminders = db.scalars(select(Reminder).order_by(Reminder.task)).all()
    assert len(reminders) == 2
    assert sorted(reminder.version for reminder in reminders) == [1, 2]


def test_sending_notification_conflict_rolls_back_everything(
    api_client, db, session_factory, update_factory, fixed_now
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(917, text="Remind me tomorrow at 9 AM to call the bank"))
    reminder = db.scalar(select(Reminder))
    notification = db.scalar(select(Notification))
    assert reminder is not None and notification is not None
    notification.status = "SENDING"
    db.commit()

    with pytest.raises(RescheduleConflict):
        with db.begin():
            apply_reschedule(
                db,
                user_id=reminder.user_id,
                destination_identity_id=notification.destination_identity_id,
                reminder_id=reminder.id,
                due_at=datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc),
                scope=RescheduleScope.OCCURRENCE,
                now=fixed_now,
            )

    db.expire_all()
    reminder = db.scalar(select(Reminder))
    notification = db.scalar(select(Notification))
    assert reminder is not None and reminder.version == 1
    assert notification is not None and notification.status == "SENDING"


def test_reschedule_is_rejected_when_a_worker_has_claimed_the_notification(
    api_client, db, fixed_now, session_factory, update_factory
) -> None:
    """A real worker claim wins over a concurrent schedule replacement attempt."""

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(944, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    notification = db.scalar(select(Notification))
    assert reminder is not None and notification is not None
    reminder_id = reminder.id
    user_id = reminder.user_id
    destination_identity_id = notification.destination_identity_id
    notification_id = notification.id

    worker = NotificationWorker(
        session_factory,
        telegram_client=None,
        clock=lambda: datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc),
    )
    claims = worker.claim_due_notifications()
    assert len(claims) == 1
    db.rollback()

    with pytest.raises(RescheduleConflict, match="already being sent"):
        with db.begin():
            apply_reschedule(
                db,
                user_id=user_id,
                destination_identity_id=destination_identity_id,
                reminder_id=reminder_id,
                due_at=datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc),
                scope=RescheduleScope.OCCURRENCE,
                now=fixed_now,
            )

    db.expire_all()
    assert db.get(Reminder, reminder_id).version == 1
    claimed = db.get(Notification, notification_id)
    assert claimed is not None and claimed.status == "SENDING"


def test_database_failure_rolls_back_the_original_schedule(
    api_client, db, update_factory, monkeypatch
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(921, text="Remind me tomorrow at 9 AM to call the bank"))
    original = db.scalar(select(Reminder))
    assert original is not None
    original_due = db.scalar(select(ReminderOccurrence)).next_due_at

    def fail(*args, **kwargs):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(rescheduling, "_replace_pending_notification", fail)
    response = post_update(
        api_client,
        update_factory(922, text="Move the bank reminder to tomorrow at 10 AM"),
    )

    assert response.status_code == 500
    db.expire_all()
    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    assert reminder is not None and reminder.version == 1
    assert occurrence is not None and occurrence.next_due_at == original_due


def test_recurring_reminder_requires_scope_before_rescheduling(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(
        api_client,
        update_factory(918, text="Remind me every Saturday at 10 AM to review the report"),
    )

    response = post_update(
        api_client,
        update_factory(919, text="Move the report reminder to tomorrow at 9 AM"),
    )

    assert response.status_code == 200
    assert "upcoming reminder" in telegram_client.messages[-1][1]
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.version == 1

    response = post_update(api_client, update_factory(920, text="all"))

    assert response.status_code == 200
    assert "Updated all future 'review the report'" in telegram_client.messages[-1][1]
    db.expire_all()
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.version == 2
    assert "BYHOUR=09" in (reminder.recurrence_rule or "")
    assert {
        occurrence.scheduled_at.replace(tzinfo=timezone.utc).hour
        for occurrence in db.scalars(select(ReminderOccurrence)).all()
    } == {9}


def test_recurring_reschedule_can_update_only_the_upcoming_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(
        api_client,
        update_factory(925, text="Remind me every Saturday at 10 AM to review the report"),
    )
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    original_rule = reminder.recurrence_rule
    original_start = reminder.start_local
    original_future_occurrences = [
        (
            occurrence.id,
            occurrence.scheduled_at,
            occurrence.next_due_at,
            occurrence.recurrence_instance_key,
        )
        for occurrence in db.scalars(
            select(ReminderOccurrence)
            .where(ReminderOccurrence.reminder_id == reminder.id)
            .order_by(ReminderOccurrence.next_due_at)
        ).all()[1:]
    ]

    post_update(
        api_client,
        update_factory(926, text="Move the report reminder to tomorrow at 9 AM"),
    )
    response = post_update(api_client, update_factory(927, text="only upcoming"))

    assert response.status_code == 200
    assert "Moved 'review the report'" in telegram_client.messages[-1][1]
    db.expire_all()
    reminder = db.scalar(select(Reminder))
    occurrences = db.scalars(
        select(ReminderOccurrence).order_by(ReminderOccurrence.next_due_at)
    ).all()
    assert reminder is not None
    assert reminder.recurrence_rule == original_rule
    assert reminder.start_local == original_start
    assert occurrences[0].next_due_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 10, 31, 9, 0, tzinfo=timezone.utc
    )
    assert [
        (
            occurrence.id,
            occurrence.scheduled_at,
            occurrence.next_due_at,
            occurrence.recurrence_instance_key,
        )
        for occurrence in occurrences[1:]
    ] == original_future_occurrences


@pytest.mark.parametrize(
    ("replacement", "expected_response"),
    [
        ("tomorrow at 25:00", "valid clock time"),
        ("today at 9 AM", "future time"),
    ],
)
def test_invalid_or_past_replacement_time_leaves_reminder_unchanged(
    api_client, telegram_client, db, update_factory, replacement, expected_response
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(928, text="Remind me tomorrow at 9 AM to call the bank"))
    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    assert reminder is not None and occurrence is not None
    original_due = occurrence.next_due_at

    response = post_update(
        api_client,
        update_factory(929, text=f"Move the bank reminder to {replacement}"),
    )

    assert response.status_code == 200
    assert expected_response in telegram_client.messages[-1][1].lower()
    db.expire_all()
    assert db.scalar(select(Reminder)).version == 1
    assert db.scalar(select(ReminderOccurrence)).next_due_at == original_due
    assert db.scalar(select(func.count()).select_from(Notification).where(Notification.status == "CANCELED")) == 0


@pytest.mark.parametrize(
    "command",
    [
        "Move to tomorrow at 10 AM",
        "Move the reminder tomorrow at 10 AM",
        "Move the bank reminder to tomorrow",
    ],
)
def test_missing_or_malformed_reschedule_target_or_time_cannot_mutate(
    api_client, telegram_client, db, update_factory, command
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(930, text="Remind me tomorrow at 9 AM to call the bank"))
    original_due = db.scalar(select(ReminderOccurrence)).next_due_at

    response = post_update(api_client, update_factory(931, text=command))

    assert response.status_code == 200
    assert db.scalar(select(Reminder)).version == 1
    assert db.scalar(select(ReminderOccurrence)).next_due_at == original_due
    assert db.scalar(select(func.count()).select_from(Notification)) == 1


def test_never_mind_ends_a_reschedule_clarification_without_mutation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(
        api_client,
        update_factory(932, text="Remind me every Saturday at 10 AM to review the report"),
    )
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    original_version = reminder.version

    post_update(
        api_client,
        update_factory(933, text="Move the report reminder to tomorrow at 9 AM"),
    )
    response = post_update(api_client, update_factory(934, text="never mind"))

    assert response.status_code == 200
    assert "won't move" in telegram_client.messages[-1][1].lower()
    db.expire_all()
    reminder = db.scalar(select(Reminder))
    session = db.scalar(select(ConversationSession))
    assert reminder is not None and reminder.version == original_version
    assert session is not None and session.status == "ENDED"
    assert session.pending_clarification is None


def test_invalid_ordinal_keeps_multiple_match_clarification_open(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(935, text="Remind me tomorrow at 9 AM to call the bank"))
    post_update(api_client, update_factory(936, text="Remind me tomorrow at 11 AM to email the bank"))
    post_update(api_client, update_factory(937, text="Move the bank reminder to tomorrow at 10 AM"))

    response = post_update(api_client, update_factory(938, text="the third one"))

    assert response.status_code == 200
    assert "which reminder" in telegram_client.messages[-1][1].lower()
    assert all(reminder.version == 1 for reminder in db.scalars(select(Reminder)).all())
    session = db.scalar(select(ConversationSession))
    assert session is not None and session.status == "AWAITING_CLARIFICATION"
    assert len(session.pending_clarification["candidate_ids"]) == 2


@pytest.mark.parametrize("reminder_state", ["CANCELED", "MISSING"])
def test_reminder_removed_between_reschedule_clarification_and_mutation_is_safe(
    api_client, telegram_client, db, update_factory, reminder_state
) -> None:
    onboard(api_client, update_factory)
    post_update(
        api_client,
        update_factory(939, text="Remind me every Saturday at 10 AM to review the report"),
    )
    post_update(
        api_client,
        update_factory(940, text="Move the report reminder to tomorrow at 9 AM"),
    )
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    if reminder_state == "CANCELED":
        reminder.status = "CANCELED"
    else:
        db.delete(reminder)
    db.commit()

    response = post_update(api_client, update_factory(941, text="only the upcoming reminder"))

    assert response.status_code == 200
    assert "couldn't move" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(func.count()).select_from(Notification).where(Notification.status == "CANCELED")) == 0


@pytest.mark.parametrize("destination_kind", ["missing", "other_user"])
def test_reschedule_rejects_missing_or_unauthorized_telegram_destination(
    api_client, db, update_factory, fixed_now, destination_kind
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(942, text="Remind me tomorrow at 9 AM to call the bank"))
    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    notification = db.scalar(select(Notification))
    assert reminder is not None and occurrence is not None and notification is not None
    original_due = occurrence.next_due_at

    if destination_kind == "missing":
        destination_id = uuid4()
    else:
        onboard(api_client, update_factory, user_id=2002, chat_id=3002, name="Bob", base=950)
        destination_id = db.scalar(
            select(ChannelIdentity.id).where(ChannelIdentity.external_id == "2002")
        )
        assert destination_id is not None

    with pytest.raises(RescheduleConflict, match="delivery destination"):
        apply_reschedule(
            db,
            user_id=reminder.user_id,
            destination_identity_id=destination_id,
            reminder_id=reminder.id,
            due_at=datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc),
            scope=RescheduleScope.OCCURRENCE,
            now=fixed_now,
        )
    db.rollback()
    db.expire_all()
    assert db.scalar(select(Reminder)).version == 1
    assert db.scalar(select(ReminderOccurrence)).next_due_at == original_due
    assert db.scalar(select(Notification).where(Notification.id == notification.id)).status == "PENDING"


def test_reschedule_cannot_find_another_users_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(909, text="Remind me tomorrow at 9 AM to call the bank"))
    onboard(api_client, update_factory, user_id=2002, chat_id=3002, name="Bob", base=920)

    response = post_update(
        api_client,
        update_factory(910, user_id=2002, chat_id=3002, first_name="Bob", text="Move the bank reminder to tomorrow at 10 AM"),
    )

    assert response.status_code == 200
    assert "couldn't find" in telegram_client.messages[-1][1].lower()
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.version == 1


def test_worker_cannot_send_the_obsolete_due_notification(
    api_client, telegram_client, db, session_factory, update_factory, fixed_now
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(911, text="Remind me tomorrow at 9 AM to call the bank"))
    post_update(api_client, update_factory(912, text="Move the bank reminder to tomorrow at 10 AM"))

    telegram_client.messages.clear()
    worker = NotificationWorker(
        session_factory,
        telegram_client,
        clock=lambda: datetime(2026, 10, 31, 9, 30, tzinfo=timezone.utc),
    )
    worker.run_once()
    assert telegram_client.messages == []

    worker.clock = lambda: datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)
    worker.run_once()
    assert len(telegram_client.messages) == 1
    assert "call the bank" in telegram_client.messages[0][1]


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL row locking for concurrent reschedule coverage",
)
def test_postgres_concurrent_reschedules_are_serialized_without_duplicate_pending_notifications(
    api_client, telegram_client, db, fixed_now, monkeypatch, session_factory, update_factory
) -> None:
    """A second move waits for the first transaction and replaces its notification safely."""

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(943, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    notification = db.scalar(select(Notification))
    assert reminder is not None and notification is not None

    first_due = datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)
    second_due = datetime(2026, 10, 31, 11, 0, tzinfo=timezone.utc)
    first_replacement_started = Event()
    allow_first_commit = Event()
    second_attempted = Event()
    second_completed = Event()
    callback_lock = Lock()
    callback_calls = 0
    errors: list[BaseException] = []
    original_replace = rescheduling._replace_pending_notification

    def pause_first_replacement(*args, **kwargs) -> None:
        nonlocal callback_calls
        with callback_lock:
            callback_calls += 1
            is_first_replacement = callback_calls == 1
        if is_first_replacement:
            first_replacement_started.set()
            if not allow_first_commit.wait(timeout=5):
                raise RuntimeError("test did not allow the first reschedule to commit")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(rescheduling, "_replace_pending_notification", pause_first_replacement)

    def move(due_at: datetime, *, is_second: bool = False) -> None:
        session = session_factory()
        try:
            if is_second:
                second_attempted.set()
            with session.begin():
                apply_reschedule(
                    session,
                    user_id=reminder.user_id,
                    destination_identity_id=notification.destination_identity_id,
                    reminder_id=reminder.id,
                    due_at=due_at,
                    scope=RescheduleScope.OCCURRENCE,
                    now=fixed_now,
                )
        except BaseException as exc:  # pragma: no cover - asserted by the main test thread
            errors.append(exc)
        finally:
            session.close()
            if is_second:
                second_completed.set()

    first = Thread(target=move, args=(first_due,))
    first.start()
    assert first_replacement_started.wait(timeout=5)

    second = Thread(target=move, args=(second_due,), kwargs={"is_second": True})
    second.start()
    assert second_attempted.wait(timeout=5)
    assert not second_completed.wait(timeout=0.25), "second move bypassed the reminder row lock"

    allow_first_commit.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors

    db.expire_all()
    refreshed_reminder = db.get(Reminder, reminder.id)
    occurrence = db.scalar(select(ReminderOccurrence).where(ReminderOccurrence.reminder_id == reminder.id))
    assert occurrence is not None
    notifications = db.scalars(
        select(Notification)
        .where(Notification.occurrence_id == occurrence.id)
        .order_by(Notification.sequence)
    ).all()
    assert refreshed_reminder is not None and refreshed_reminder.version == 3
    assert occurrence.next_due_at == second_due
    assert [item.status for item in notifications] == ["CANCELED", "CANCELED", "PENDING"]
    assert [item.sequence for item in notifications] == [1, 2, 3]

    telegram_client.messages.clear()
    worker = NotificationWorker(
        session_factory,
        telegram_client,
        clock=lambda: fixed_now + timedelta(days=1),
    )
    assert worker.run_once() == 1
    db.expire_all()
    assert [item.status for item in db.scalars(
        select(Notification).where(Notification.occurrence_id == occurrence.id).order_by(Notification.sequence)
    ).all()] == ["CANCELED", "CANCELED", "SENT"]
    assert telegram_client.messages == [(2001, "Reminder: call the bank")]


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL row locking for destination lifecycle coverage",
)
def test_postgres_destination_cannot_be_unlinked_before_reschedule_commit(
    api_client, db, fixed_now, monkeypatch, session_factory, update_factory
) -> None:
    """The destination row is locked with the reminder write transaction."""

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(960, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    notification = db.scalar(select(Notification))
    assert reminder is not None and notification is not None

    replacement_started = Event()
    allow_reschedule_commit = Event()
    unlink_attempted = Event()
    unlink_completed = Event()
    errors: list[BaseException] = []
    original_replace = rescheduling._replace_pending_notification

    def pause_after_destination_lock(*args, **kwargs) -> None:
        replacement_started.set()
        if not allow_reschedule_commit.wait(timeout=5):
            raise RuntimeError("test did not allow the reschedule to commit")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(rescheduling, "_replace_pending_notification", pause_after_destination_lock)

    def reschedule() -> None:
        session = session_factory()
        try:
            with session.begin():
                apply_reschedule(
                    session,
                    user_id=reminder.user_id,
                    destination_identity_id=notification.destination_identity_id,
                    reminder_id=reminder.id,
                    due_at=datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc),
                    scope=RescheduleScope.OCCURRENCE,
                    now=fixed_now,
                )
        except BaseException as exc:  # pragma: no cover - asserted by the main test thread
            errors.append(exc)
        finally:
            session.close()

    def unlink_destination() -> None:
        session = session_factory()
        try:
            unlink_attempted.set()
            with session.begin():
                destination = session.get(ChannelIdentity, notification.destination_identity_id)
                assert destination is not None
                destination.status = "UNLINKED"
        except BaseException as exc:  # pragma: no cover - asserted by the main test thread
            errors.append(exc)
        finally:
            session.close()
            unlink_completed.set()

    reschedule_thread = Thread(target=reschedule)
    reschedule_thread.start()
    assert replacement_started.wait(timeout=5)

    unlink_thread = Thread(target=unlink_destination)
    unlink_thread.start()
    assert unlink_attempted.wait(timeout=5)
    assert not unlink_completed.wait(timeout=0.25), "destination changed before reschedule commit"

    allow_reschedule_commit.set()
    reschedule_thread.join(timeout=5)
    unlink_thread.join(timeout=5)
    assert not reschedule_thread.is_alive() and not unlink_thread.is_alive()
    assert not errors

    db.expire_all()
    notifications = db.scalars(
        select(Notification)
        .where(Notification.occurrence_id == notification.occurrence_id)
        .order_by(Notification.sequence)
    ).all()
    assert [item.status for item in notifications] == ["CANCELED", "PENDING"]
