from datetime import date, datetime, timezone
import os
from threading import Thread

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session

from reminder_agent.db.models import (
    ConversationSession,
    Notification,
    Reminder,
    ReminderOccurrence,
    ReminderRecurrenceException,
)
from reminder_agent.cancellation import (
    CancellationConflict,
    CancellationRequest,
    CancellationScope,
    apply_cancellation,
    decide_cancellation,
)
from reminder_agent.intent import IntentRoute, IntentRouteSuggestion
from reminder_agent.recurrence_service import materialize_active_recurring_reminders
from reminder_agent.worker import NotificationWorker


def post_update(api_client, update):
    return api_client.post(
        "/webhook/telegram",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
    )


def onboard(api_client, update_factory, *, user_id=1001, chat_id=2001, base=4000):
    assert post_update(
        api_client,
        update_factory(base, user_id=user_id, chat_id=chat_id, text="/start"),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(base + 1, user_id=user_id, chat_id=chat_id, text="UTC"),
    ).status_code == 200


requires_postgres = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires the dedicated PostgreSQL integration database",
)


def _reminder_state(db, reminder_id):
    reminder = db.get(Reminder, reminder_id)
    assert reminder is not None
    occurrences = db.scalars(
        select(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder_id)
        .order_by(ReminderOccurrence.id)
    ).all()
    notifications = db.scalars(
        select(Notification)
        .join(ReminderOccurrence, Notification.occurrence_id == ReminderOccurrence.id)
        .where(ReminderOccurrence.reminder_id == reminder_id)
        .order_by(Notification.id)
    ).all()
    return (
        reminder.status,
        reminder.version,
        reminder.updated_at,
        [(item.id, item.status, item.updated_at) for item in occurrences],
        [
            (item.id, item.status, item.claimed_by, item.claimed_until)
            for item in notifications
        ],
    )


def test_cancel_one_time_reminder_prevents_worker_delivery(
    api_client, telegram_client, db, session_factory, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4002, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200

    response = post_update(api_client, update_factory(4003, text="Cancel the bank reminder"))

    assert response.status_code == 200
    assert "Canceled 'call the bank'" in telegram_client.messages[-1][1]
    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    notification = db.scalar(select(Notification))
    assert reminder is not None and reminder.status == "CANCELED"
    assert occurrence is not None and occurrence.status == "CANCELED"
    assert notification is not None and notification.status == "CANCELED"

    telegram_client.messages.clear()
    worker = NotificationWorker(
        session_factory,
        telegram_client,
        clock=lambda: datetime(2026, 10, 31, 9, tzinfo=timezone.utc),
    )
    assert worker.run_once() == 0
    assert telegram_client.messages == []

    repeated = post_update(api_client, update_factory(4004, text="Cancel the bank reminder"))
    assert repeated.status_code == 200
    assert "couldn't find" in telegram_client.messages[-1][1].lower()
    db.expire_all()
    assert db.scalar(select(Reminder)).status == "CANCELED"


def test_llm_routed_indirect_cancellation_reaches_transactional_service(
    api_client, telegram_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4005, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    api_client.app.state.intent_router = CancelRouter()

    response = post_update(
        api_client,
        update_factory(4006, text="I do not need the bank reminder anymore"),
    )

    assert response.status_code == 200
    assert "Canceled 'call the bank'" in telegram_client.messages[-1][1]
    assert db.scalar(select(Reminder)).status == "CANCELED"


def test_cancel_recurring_series_preserves_history_and_blocks_future_materialization(
    api_client, telegram_client, db, session_factory, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4010, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    occurrences_before = db.scalars(
        select(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
        .order_by(ReminderOccurrence.next_due_at)
    ).all()
    assert len(occurrences_before) >= 2
    occurrences_before[0].status = "NOTIFIED"
    first_notification = db.scalar(
        select(Notification).where(Notification.occurrence_id == occurrences_before[0].id)
    )
    assert first_notification is not None
    first_notification.status = "SENT"
    db.commit()

    response = post_update(api_client, update_factory(4011, text="Cancel the standup"))

    assert response.status_code == 200
    db.expire_all()
    reminder = db.get(Reminder, reminder.id)
    assert reminder is not None and reminder.status == "CANCELED"
    occurrences = db.scalars(
        select(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
    ).all()
    assert occurrences[0].status == "NOTIFIED"
    assert all(item.status == "CANCELED" for item in occurrences[1:])
    assert db.scalar(select(Notification).where(Notification.id == first_notification.id)).status == "SENT"

    telegram_client.messages.clear()
    worker = NotificationWorker(
        session_factory,
        telegram_client,
        clock=lambda: datetime(2026, 11, 30, 12, tzinfo=timezone.utc),
    )
    assert worker.run_once() == 0
    assert telegram_client.messages == []


def test_cancel_recurring_series_when_materialized_occurrences_are_exhausted(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4015, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    occurrences = db.scalars(
        select(ReminderOccurrence).where(ReminderOccurrence.reminder_id == reminder.id)
    ).all()
    notifications = db.scalars(
        select(Notification).where(
            Notification.occurrence_id.in_([item.id for item in occurrences])
        )
    ).all()
    for occurrence in occurrences:
        occurrence.status = "CANCELED"
    for notification in notifications:
        notification.status = "CANCELED"
    db.commit()

    response = post_update(api_client, update_factory(4016, text="Cancel the standup"))

    assert response.status_code == 200
    db.expire_all()
    reminder = db.get(Reminder, reminder.id)
    assert reminder is not None and reminder.status == "CANCELED"
    assert "recurring reminder" in telegram_client.messages[-1][1]


def test_cancel_only_one_recurring_occurrence_leaves_series_active(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4020, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    response = post_update(
        api_client, update_factory(4021, text="Cancel only Monday's standup")
    )

    assert response.status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.status == "ACTIVE"
    occurrences = db.scalars(
        select(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
        .order_by(ReminderOccurrence.next_due_at)
    ).all()
    assert occurrences[0].status == "CANCELED"
    assert any(item.status == "PENDING" for item in occurrences[1:])
    assert "series remains active" in telegram_client.messages[-1][1]


def test_ambiguous_cancellation_clarifies_and_follow_up_cancels_only_selected(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4030, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(4031, text="Remind me tomorrow at 10 AM to email the bank"),
    ).status_code == 200

    response = post_update(api_client, update_factory(4032, text="Cancel the bank reminder"))
    assert response.status_code == 200
    assert "multiple matching" in telegram_client.messages[-1][1].lower()
    session = db.scalar(select(ConversationSession))
    assert session is not None and session.status == "AWAITING_CLARIFICATION"
    assert session.pending_clarification["operation"] == "CANCEL_REMINDER"

    response = post_update(api_client, update_factory(4033, text="the second one"))
    assert response.status_code == 200
    db.expire_all()
    reminders = db.scalars(select(Reminder).order_by(Reminder.created_at)).all()
    assert [item.status for item in reminders] == ["ACTIVE", "CANCELED"]
    assert db.scalar(select(ConversationSession)).status == "ENDED"


def test_cancellation_replay_is_idempotent_and_cross_user_isolated(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    update = update_factory(4040, text="Remind me tomorrow at 9 AM to call the bank")
    assert post_update(api_client, update).status_code == 200
    cancel = update_factory(4041, text="Cancel the bank reminder")
    assert post_update(api_client, cancel).status_code == 200
    first_count = db.scalar(select(func.count()).select_from(Reminder).where(Reminder.status == "CANCELED"))
    first_reply_count = len(telegram_client.messages)
    assert post_update(api_client, cancel).status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder).where(Reminder.status == "CANCELED")) == first_count
    assert len(telegram_client.messages) == first_reply_count

    onboard(api_client, update_factory, user_id=2002, chat_id=3002, base=4050)
    response = post_update(
        api_client,
        update_factory(4052, user_id=2002, chat_id=3002, text="Cancel the bank reminder"),
    )
    assert response.status_code == 200
    assert "couldn't find" in telegram_client.messages[-1][1].lower()


def test_cancellation_fails_without_mutation_when_notification_is_sending(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4060, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    notification = db.scalar(select(Notification))
    reminder = db.scalar(select(Reminder))
    assert notification is not None and reminder is not None
    notification.status = "SENDING"
    db.commit()

    response = post_update(api_client, update_factory(4061, text="Cancel the bank reminder"))

    assert response.status_code == 200
    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.get(Notification, notification.id).status == "SENDING"
    assert "already being sent" in telegram_client.messages[-1][1]


@requires_postgres
def test_canceling_a_reminder_leaves_unrelated_reminders_and_notifications_unchanged(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4065, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(4066, text="Remind me tomorrow at 10 AM to walk the dog"),
    ).status_code == 200
    unrelated = db.scalar(select(Reminder).where(Reminder.task == "walk the dog"))
    assert unrelated is not None
    before = _reminder_state(db, unrelated.id)

    response = post_update(api_client, update_factory(4067, text="Cancel the bank reminder"))

    assert response.status_code == 200
    db.expire_all()
    assert _reminder_state(db, unrelated.id) == before


@requires_postgres
def test_canceling_a_nonexistent_reminder_returns_not_found_without_reminder_mutations(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4068, text="Remind me tomorrow at 9 AM to call the dentist"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    before = _reminder_state(db, reminder.id)

    response = post_update(api_client, update_factory(4069, text="Cancel the bank reminder"))

    assert response.status_code == 200
    assert "couldn't find" in telegram_client.messages[-1][1].lower()
    db.expire_all()
    assert _reminder_state(db, reminder.id) == before


@requires_postgres
def test_database_failure_rolls_back_all_cancellation_state(
    api_client, db, session_factory, update_factory, fixed_now
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4071, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    before = _reminder_state(db, reminder.id)

    failed_session = session_factory()
    try:
        with pytest.raises(DBAPIError):
            with failed_session.begin():
                apply_cancellation(
                    failed_session,
                    user_id=reminder.user_id,
                    reminder_id=reminder.id,
                    occurrence_id=None,
                    scope=CancellationScope.OCCURRENCE,
                    now=fixed_now,
                )
                failed_session.execute(text("SELECT * FROM vs10_forced_database_failure"))
    finally:
        failed_session.close()

    db.expire_all()
    assert _reminder_state(db, reminder.id) == before


@requires_postgres
def test_canceling_a_recurring_series_cancels_every_pending_notification(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4072, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    pending_notification_ids = db.scalars(
        select(Notification.id)
        .join(ReminderOccurrence, Notification.occurrence_id == ReminderOccurrence.id)
        .where(
            ReminderOccurrence.reminder_id == reminder.id,
            Notification.status == "PENDING",
        )
    ).all()
    assert pending_notification_ids

    response = post_update(api_client, update_factory(4073, text="Cancel the standup"))

    assert response.status_code == 200
    db.expire_all()
    notifications = db.scalars(
        select(Notification).where(Notification.id.in_(pending_notification_ids))
    ).all()
    assert len(notifications) == len(pending_notification_ids)
    assert all(notification.status == "CANCELED" for notification in notifications)
    assert db.scalar(
        select(func.count())
        .select_from(Notification)
        .join(ReminderOccurrence, Notification.occurrence_id == ReminderOccurrence.id)
        .where(
            ReminderOccurrence.reminder_id == reminder.id,
            Notification.status == "PENDING",
        )
    ) == 0


@requires_postgres
def test_recurrence_materializer_skips_canceled_series(
    api_client, db, session_factory, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4076, text="Remind me every Monday at 8 AM to review the pipeline"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    occurrence_count_before = db.scalar(
        select(func.count())
        .select_from(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
    )
    notification_count_before = db.scalar(
        select(func.count())
        .select_from(Notification)
        .join(ReminderOccurrence, Notification.occurrence_id == ReminderOccurrence.id)
        .where(ReminderOccurrence.reminder_id == reminder.id)
    )

    assert post_update(
        api_client, update_factory(4077, text="Cancel the pipeline reminder")
    ).status_code == 200
    assert materialize_active_recurring_reminders(
        session_factory,
        now=datetime(2026, 11, 30, 12, tzinfo=timezone.utc),
        horizon_days=35,
    ) == 0

    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "CANCELED"
    assert db.scalar(
        select(func.count())
        .select_from(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
    ) == occurrence_count_before
    assert db.scalar(
        select(func.count())
        .select_from(Notification)
        .join(ReminderOccurrence, Notification.occurrence_id == ReminderOccurrence.id)
        .where(ReminderOccurrence.reminder_id == reminder.id)
    ) == notification_count_before


@requires_postgres
def test_unmaterialized_recurring_occurrence_skip_persists_and_blocks_materialization(
    api_client, db, session_factory, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4110, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    skipped_date = date(2027, 1, 4)

    response = post_update(
        api_client,
        update_factory(4111, text="Cancel standup on 2027-01-04"),
    )
    assert response.status_code == 200

    materialize_active_recurring_reminders(
        session_factory,
        now=datetime(2026, 12, 15, 12, tzinfo=timezone.utc),
        horizon_days=35,
    )
    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.scalar(
        select(ReminderRecurrenceException).where(
            ReminderRecurrenceException.reminder_id == reminder.id,
            ReminderRecurrenceException.occurrence_date == skipped_date,
        )
    ) is not None
    assert db.scalar(
        select(ReminderOccurrence).where(
            ReminderOccurrence.reminder_id == reminder.id,
            ReminderOccurrence.recurrence_instance_key == "2027-01-04T08:00",
        )
    ) is None


@requires_postgres
def test_unmaterialized_skip_rejects_invalid_and_duplicate_dates_without_parent_mutation(
    api_client, db, fixed_now, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4111, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    invalid_date = date(2027, 1, 5)
    valid_date = date(2027, 1, 4)

    db.rollback()
    with pytest.raises(CancellationConflict):
        with db.begin():
            apply_cancellation(
                db,
                user_id=reminder.user_id,
                reminder_id=reminder.id,
                occurrence_id=None,
                occurrence_date=invalid_date,
                scope=CancellationScope.OCCURRENCE,
                now=fixed_now,
            )
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.scalar(select(func.count()).select_from(ReminderRecurrenceException)) == 0
    db.rollback()

    with db.begin():
        apply_cancellation(
            db,
            user_id=reminder.user_id,
            reminder_id=reminder.id,
            occurrence_id=None,
            occurrence_date=valid_date,
            scope=CancellationScope.OCCURRENCE,
            now=fixed_now,
        )
    with pytest.raises(CancellationConflict):
        with db.begin():
            apply_cancellation(
                db,
                user_id=reminder.user_id,
                reminder_id=reminder.id,
                occurrence_id=None,
                occurrence_date=valid_date,
                scope=CancellationScope.OCCURRENCE,
                now=fixed_now,
            )
    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.scalar(select(func.count()).select_from(ReminderRecurrenceException)) == 1


@requires_postgres
def test_postgres_concurrent_unmaterialized_skips_are_serialized(
    api_client, db, fixed_now, session_factory, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4112, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    skipped_date = date(2027, 1, 4)
    outcomes: list[str] = []

    def skip_once() -> None:
        session = session_factory()
        try:
            with session.begin():
                apply_cancellation(
                    session,
                    user_id=reminder.user_id,
                    reminder_id=reminder.id,
                    occurrence_id=None,
                    occurrence_date=skipped_date,
                    scope=CancellationScope.OCCURRENCE,
                    now=fixed_now,
                )
            outcomes.append("skipped")
        except CancellationConflict:
            session.rollback()
            outcomes.append("conflict")
        finally:
            session.close()

    first = Thread(target=skip_once)
    second = Thread(target=skip_once)
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert sorted(outcomes) == ["conflict", "skipped"]
    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.scalar(select(func.count()).select_from(ReminderRecurrenceException)) == 1


@requires_postgres
def test_commit_failure_rolls_back_cancellation_without_sending_success(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4074, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    before = _reminder_state(db, reminder.id)
    message_count_before = len(telegram_client.messages)

    def fail_cancellation_commit(session: Session) -> None:
        if any(
            isinstance(item, Reminder) and item.status == "CANCELED"
            for item in session.dirty
        ):
            raise OperationalError("COMMIT", {}, RuntimeError("forced commit failure"))

    event.listen(Session, "before_commit", fail_cancellation_commit)
    try:
        response = post_update(
            api_client, update_factory(4075, text="Cancel the bank reminder")
        )
    finally:
        event.remove(Session, "before_commit", fail_cancellation_commit)

    assert response.status_code == 500
    assert len(telegram_client.messages) == message_count_before
    db.expire_all()
    assert _reminder_state(db, reminder.id) == before


@requires_postgres
def test_exact_cancellation_target_does_not_invoke_semantic_fallback(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4080, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200

    class FailIfCalled:
        def interpret_cancellation(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("exact target must not call the fallback")

    api_client.app.state.reminder_interpreter = FailIfCalled()
    assert post_update(
        api_client, update_factory(4081, text="Cancel the bank reminder")
    ).status_code == 200
    assert db.scalar(select(Reminder)).status == "CANCELED"


@requires_postgres
def test_unique_substantive_shorthand_cancels_deterministically(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4100, text="Remind me tomorrow at 9 AM to gym workout"),
    ).status_code == 200

    class FailIfCalled:
        def interpret_cancellation(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("unique shorthand must not require the fallback")

    api_client.app.state.reminder_interpreter = FailIfCalled()
    assert post_update(api_client, update_factory(4101, text="Cancel gym")).status_code == 200
    assert db.scalar(select(Reminder)).status == "CANCELED"


@requires_postgres
def test_unique_generic_call_target_cancels_matching_series(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4120, text="Remind me every day at 9 AM to call parents"),
    ).status_code == 200

    assert post_update(
        api_client,
        update_factory(4121, text="I don't need the call reminders anymore"),
    ).status_code == 200

    assert db.scalar(select(Reminder)).status == "CANCELED"


@requires_postgres
def test_partial_target_uses_a_model_proposal_before_cancellation(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4102, text="Remind me tomorrow at 9 AM to call bank manager"),
    ).status_code == 200

    class PartialInterpreter:
        def interpret_cancellation(self, text, **kwargs):
            assert text == "Cancel call bank"
            return CancellationRequest("call bank manager")

    api_client.app.state.reminder_interpreter = PartialInterpreter()
    assert post_update(
        api_client, update_factory(4103, text="Cancel call bank")
    ).status_code == 200
    assert db.scalar(select(Reminder)).status == "CANCELED"


@requires_postgres
def test_indirect_demonstrative_cancellation_uses_bounded_custom_fallback_context(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4115, text="Remind me every Monday at 8 AM to exercise routine"),
    ).status_code == 200
    captured: dict = {}

    class ContextAwareInterpreter:
        def interpret_cancellation(self, text, **kwargs):
            captured["text"] = text
            captured.update(kwargs)
            return CancellationRequest("exercise routine")

    api_client.app.state.reminder_interpreter = ContextAwareInterpreter()
    assert post_update(
        api_client,
        update_factory(4116, text="I don't need those exercise reminders anymore."),
    ).status_code == 200

    assert captured["text"] == "I don't need those exercise reminders anymore."
    assert len(captured["recent_messages"]) <= 6
    assert any("exercise routine" in item for item in captured["recent_messages"])
    assert captured["cancellation_candidates"][0]["task"] == "exercise routine"
    assert "id" not in captured["cancellation_candidates"][0]
    assert db.scalar(select(Reminder)).status == "CANCELED"


@requires_postgres
def test_semantically_different_target_requests_clarification_without_fallback(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4104, text="Remind me tomorrow at 9 AM to gym workout"),
    ).status_code == 200

    class FailIfCalled:
        def interpret_cancellation(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("unrelated target must not call the fallback")

    api_client.app.state.reminder_interpreter = FailIfCalled()
    assert post_update(
        api_client, update_factory(4105, text="Cancel evening exercise")
    ).status_code == 200
    assert "couldn't confidently match" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(Reminder)).status == "ACTIVE"


@requires_postgres
def test_multiple_partial_matches_require_clarification(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    for update_id, task in ((4106, "call bank manager"), (4107, "call bank support")):
        assert post_update(
            api_client,
            update_factory(update_id, text=f"Remind me tomorrow at 9 AM to {task}"),
        ).status_code == 200

    class FailIfCalled:
        def interpret_cancellation(self, *args, **kwargs):
            raise AssertionError("multiple partial matches must clarify before model selection")

    api_client.app.state.reminder_interpreter = FailIfCalled()
    assert post_update(
        api_client, update_factory(4108, text="Cancel call bank")
    ).status_code == 200
    assert "multiple possible" in telegram_client.messages[-1][1].lower()
    assert db.scalars(select(Reminder.status)).all() == ["ACTIVE", "ACTIVE"]


@requires_postgres
def test_uncertain_natural_language_cancellation_uses_owned_candidate_context(
    api_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4082, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    captured = {}

    class ParaphraseInterpreter:
        def interpret_cancellation(self, text, *, cancellation_candidates, **kwargs):
            captured["text"] = text
            captured["candidates"] = cancellation_candidates
            return CancellationRequest("call the bank")

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = ParaphraseInterpreter()
    assert post_update(
        api_client, update_factory(4083, text="Skip my banking errand")
    ).status_code == 200

    assert captured["text"] == "Skip my banking errand"
    assert captured["candidates"] == [
        {
            "task": "call the bank",
            "schedule_type": "ONE_TIME",
            "due_at": captured["candidates"][0]["due_at"],
        }
    ]
    assert "id" not in captured["candidates"][0]
    assert db.scalar(select(Reminder)).status == "CANCELED"


@requires_postgres
def test_contextual_cancellation_target_uses_recent_conversation(
    api_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4084, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200

    class ContextInterpreter:
        def interpret_cancellation(self, text, *, recent_messages, **kwargs):
            assert text == "Cancel that"
            assert any("call the bank" in message for message in recent_messages)
            return CancellationRequest("call the bank")

    api_client.app.state.reminder_interpreter = ContextInterpreter()
    assert post_update(api_client, update_factory(4085, text="Cancel that")).status_code == 200
    assert db.scalar(select(Reminder)).status == "CANCELED"


@requires_postgres
def test_contextual_next_occurrence_uses_owned_pending_occurrence_not_model_date(
    api_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    class ContextInterpreter:
        def interpret_cancellation(self, *args, **kwargs):
            return CancellationRequest(
                "standup", CancellationScope.OCCURRENCE, date(2026, 11, 30)
            )

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4095, text="Remind me every Monday at 8 AM to standup"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    occurrences = db.scalars(
        select(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
        .order_by(ReminderOccurrence.next_due_at)
    ).all()
    assert len(occurrences) >= 2

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = ContextInterpreter()
    assert post_update(api_client, update_factory(4096, text="Skip the next one")).status_code == 200

    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.get(ReminderOccurrence, occurrences[0].id).status == "CANCELED"
    assert db.get(ReminderOccurrence, occurrences[1].id).status == "PENDING"


@requires_postgres
def test_fallback_receives_unmaterialized_recurring_candidate_for_explicit_date(
    api_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    captured: dict = {}

    class CandidateInterpreter:
        def interpret_cancellation(self, text, *, cancellation_candidates, **kwargs):
            captured["text"] = text
            captured["candidates"] = cancellation_candidates
            return CancellationRequest(
                "fitness session", CancellationScope.OCCURRENCE, date(2027, 1, 4)
            )

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4117, text="Remind me every Monday at 8 AM to fitness session"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = CandidateInterpreter()
    assert post_update(
        api_client,
        update_factory(4118, text="Skip my fitness session on 2027-01-04"),
    ).status_code == 200

    assert captured["text"] == "Skip my fitness session on 2027-01-04"
    assert captured["candidates"] == [
        {
            "task": "fitness session",
            "schedule_type": "RECURRING",
            "due_at": captured["candidates"][0]["due_at"],
        }
    ]
    assert captured["candidates"][0]["due_at"].startswith("2027-01-04T08:00")
    assert "id" not in captured["candidates"][0]
    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.scalar(
        select(ReminderRecurrenceException).where(
            ReminderRecurrenceException.reminder_id == reminder.id,
            ReminderRecurrenceException.occurrence_date == date(2027, 1, 4),
        )
    ) is not None


@requires_postgres
def test_llm_routed_explicit_date_keeps_single_occurrence_scope(
    api_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    class DateInterpreter:
        def interpret_cancellation(self, *args, **kwargs):
            return CancellationRequest(
                "planning meeting", CancellationScope.OCCURRENCE, date(2026, 11, 2)
            )

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4113, text="Remind me every Monday at 8 AM to planning meeting"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    occurrences = db.scalars(
        select(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
        .order_by(ReminderOccurrence.next_due_at)
    ).all()
    assert len(occurrences) >= 2

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = DateInterpreter()
    assert post_update(
        api_client,
        update_factory(4114, text="Skip my planning meeting on 2026-11-02"),
    ).status_code == 200

    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "ACTIVE"
    assert db.get(ReminderOccurrence, occurrences[0].id).status == "CANCELED"
    assert db.get(ReminderOccurrence, occurrences[1].id).status == "PENDING"


@requires_postgres
def test_contextual_occurrence_with_multiple_owned_series_requires_clarification(
    api_client, telegram_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    class ContextInterpreter:
        def interpret_cancellation(self, *args, **kwargs):
            return CancellationRequest(
                "standup", CancellationScope.OCCURRENCE, date(2026, 11, 30)
            )

    onboard(api_client, update_factory)
    for update_id, task in ((4097, "team standup"), (4098, "personal standup")):
        assert post_update(
            api_client,
            update_factory(update_id, text=f"Remind me every Monday at 8 AM to {task}"),
        ).status_code == 200

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = ContextInterpreter()
    assert post_update(api_client, update_factory(4099, text="Skip the next one")).status_code == 200

    assert "multiple matching" in telegram_client.messages[-1][1].lower()
    assert db.scalars(select(Reminder.status)).all() == ["ACTIVE", "ACTIVE"]


@requires_postgres
def test_semantic_fallback_clarifies_when_its_proposed_target_matches_multiple_reminders(
    api_client, telegram_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4086, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(4087, text="Remind me tomorrow at 10 AM to email the bank"),
    ).status_code == 200

    class AmbiguousInterpreter:
        def interpret_cancellation(self, *args, **kwargs):
            return CancellationRequest("bank")

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = AmbiguousInterpreter()
    assert post_update(
        api_client, update_factory(4088, text="Skip my banking errand")
    ).status_code == 200
    assert "multiple matching" in telegram_client.messages[-1][1].lower()
    assert db.scalars(select(Reminder.status)).all() == ["ACTIVE", "ACTIVE"]


@requires_postgres
def test_confidently_parsed_missing_reminder_returns_not_found_without_fallback(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4089, text="Remind me tomorrow at 9 AM to call the dentist"),
    ).status_code == 200

    class FailIfCalled:
        def interpret_cancellation(self, *args, **kwargs):
            raise AssertionError("a concrete unmatched target must not use the fallback")

    api_client.app.state.reminder_interpreter = FailIfCalled()
    assert post_update(
        api_client, update_factory(4090, text="Cancel the bank reminder")
    ).status_code == 200
    assert "couldn't find" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(Reminder)).status == "ACTIVE"


@requires_postgres
def test_semantic_fallback_preserves_recurring_series_default_scope(
    api_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4091, text="Remind me every Monday at 8 AM to review the pipeline"),
    ).status_code == 200

    class SeriesInterpreter:
        def interpret_cancellation(self, *args, **kwargs):
            return CancellationRequest("review the pipeline")

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = SeriesInterpreter()
    assert post_update(
        api_client, update_factory(4092, text="Skip my weekly planning commitment")
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.status == "CANCELED"
    assert all(
        status == "CANCELED"
        for status in db.scalars(select(ReminderOccurrence.status)).all()
    )


@requires_postgres
def test_semantic_fallback_model_failure_leaves_reminder_unchanged(
    api_client, telegram_client, db, update_factory
):
    class CancelRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.CANCEL_REQUEST)

    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4093, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200

    class FailingInterpreter:
        def interpret_cancellation(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    api_client.app.state.intent_router = CancelRouter()
    api_client.app.state.reminder_interpreter = FailingInterpreter()
    assert post_update(
        api_client, update_factory(4094, text="Skip my banking errand")
    ).status_code == 200
    assert "please tell me which" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(Reminder)).status == "ACTIVE"


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL row locking for concurrent cancellation coverage",
)
def test_postgres_concurrent_cancellations_are_serialized(
    api_client, db, session_factory, update_factory, fixed_now
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(4070, text="Remind me tomorrow at 9 AM to call the bank"),
    ).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None

    outcomes: list[str] = []

    def cancel_once() -> None:
        session = session_factory()
        try:
            with session.begin():
                apply_cancellation(
                    session,
                    user_id=reminder.user_id,
                    reminder_id=reminder.id,
                    occurrence_id=None,
                    scope=CancellationScope.OCCURRENCE,
                    now=fixed_now,
                )
            outcomes.append("canceled")
        except CancellationConflict:
            session.rollback()
            outcomes.append("conflict")
        finally:
            session.close()

    first = Thread(target=cancel_once)
    second = Thread(target=cancel_once)
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert sorted(outcomes) == ["canceled", "conflict"]
    db.expire_all()
    assert db.get(Reminder, reminder.id).status == "CANCELED"
