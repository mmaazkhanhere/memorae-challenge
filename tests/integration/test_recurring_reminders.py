from datetime import datetime, timezone
import os
from threading import Thread
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select

from reminder_agent.db.models import ChannelIdentity, Notification, Reminder, ReminderOccurrence, User
from reminder_agent.intent import CreateRecurringReminderIntent, Interpretation
from reminder_agent.recurrence_service import (
    materialize_active_recurring_reminders,
    materialize_recurring_reminder,
)
from reminder_agent.telegram import TelegramAPIError
from reminder_agent.worker import NotificationWorker


def utc_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def post_update(client, update: dict):
    return client.post(
        "/webhook/telegram",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
    )


def onboard(api_client, update_factory, *, user_id=1001, chat_id=2001) -> None:
    assert (
        post_update(api_client, update_factory(1000, user_id=user_id, chat_id=chat_id)).status_code
        == 200
    )
    assert post_update(
        api_client,
        update_factory(1001, user_id=user_id, chat_id=chat_id, text="Asia/Karachi"),
    ).status_code == 200


def test_weekly_reminder_materializes_local_wall_clock_and_confirms(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    response = post_update(
        api_client,
        update_factory(300, text="Remind me every Monday at 8 AM to call the bank."),
    )

    assert response.status_code == 200
    assert "every Mon at 08:00 Asia/Karachi via Telegram" in telegram_client.messages[-1][1]
    assert "Next: 2026-11-02 08:00 PKT" in telegram_client.messages[-1][1]

    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    assert reminder.schedule_type == "RECURRING"
    assert reminder.recurrence_rule == "FREQ=WEEKLY;BYDAY=MO;BYHOUR=08;BYMINUTE=00"
    occurrences = db.scalars(
        select(ReminderOccurrence).where(ReminderOccurrence.reminder_id == reminder.id)
    ).all()
    assert len(occurrences) >= 5
    assert len({occurrence.recurrence_instance_key for occurrence in occurrences}) == len(
        occurrences
    )
    assert utc_timestamp(occurrences[0].scheduled_at) == datetime(
        2026, 11, 2, 3, 0, tzinfo=timezone.utc
    )
    assert db.scalar(select(func.count()).select_from(Notification)) == len(occurrences)


def test_replaying_recurring_update_creates_no_duplicate_series(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    update = update_factory(301, text="Remind me every Monday at 8 AM to stretch.")

    assert post_update(api_client, update).status_code == 200
    first_count = db.scalar(select(func.count()).select_from(ReminderOccurrence))
    assert post_update(api_client, update).status_code == 200

    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == first_count
    assert len(telegram_client.messages) == 3  # onboarding responses plus one confirmation


def test_recurring_ambiguous_hour_waits_for_am_pm_before_creating_series(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    first = post_update(
        api_client,
        update_factory(306, text="Remind me every Sunday at 8 to review the weekly plan."),
    )

    assert first.status_code == 200
    assert telegram_client.messages[-1][1] == "Do you mean 8 AM or 8 PM every week?"
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0

    follow_up = post_update(api_client, update_factory(307, text="8 PM"))

    assert follow_up.status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    assert reminder.recurrence_rule == "FREQ=WEEKLY;BYDAY=SU;BYHOUR=20;BYMINUTE=00"


def test_worker_restart_and_repeated_materialization_do_not_duplicate_instances(
    api_client, db, update_factory, session_factory
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(302, text="Remind me every Monday at 8 AM to review the report."),
    ).status_code == 200
    before = db.scalar(select(func.count()).select_from(ReminderOccurrence))

    worker = NotificationWorker(
        session_factory,
        telegram_client=None,  # materialization happens before delivery
        clock=lambda: datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )
    worker.run_once()
    NotificationWorker(
        session_factory,
        telegram_client=None,
        clock=lambda: datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    ).run_once()

    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == before


def test_materialization_pages_past_the_first_batch(
    api_client, db, update_factory, session_factory
) -> None:
    onboard(api_client, update_factory)
    user = db.scalar(select(User).where(User.name == "Ada"))
    assert user is not None
    with session_factory.begin() as session:
        first = Reminder(
            user_id=user.id,
            task="page one",
            schedule_type="RECURRING",
            start_local="2026-11-02T08:00",
            timezone="Asia/Karachi",
            recurrence_rule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=08;BYMINUTE=00",
            status="ACTIVE",
        )
        second = Reminder(
            user_id=user.id,
            task="page two",
            schedule_type="RECURRING",
            start_local="2026-11-02T08:00",
            timezone="Asia/Karachi",
            recurrence_rule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=08;BYMINUTE=00",
            status="ACTIVE",
        )
        session.add_all([first, second])
        session.flush()
        reminder_ids = (first.id, second.id)

    materialize_active_recurring_reminders(
        session_factory,
        now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
        horizon_days=35,
        batch_size=1,
    )

    with session_factory() as session:
        counts = [
            session.scalar(
                select(func.count())
                .select_from(ReminderOccurrence)
                .where(ReminderOccurrence.reminder_id == reminder_id)
            )
            for reminder_id in reminder_ids
        ]
    assert all(count > 0 for count in counts)


def test_failed_delivery_keeps_recurring_series_active(
    api_client, db, update_factory, session_factory
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(303, text="Remind me every Monday at 8 AM to check the report."),
    ).status_code == 200

    class PermanentFailureTelegramClient:
        def send_message(self, chat_id: int, text: str, *, reply_markup=None):
            raise TelegramAPIError("blocked", status_code=403, failure_kind="http_error")

    worker = NotificationWorker(
        session_factory,
        PermanentFailureTelegramClient(),
        clock=lambda: datetime(2026, 11, 2, 3, 0, 1, tzinfo=timezone.utc),
    )
    assert worker.run_once() == 1

    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.status == "ACTIVE"
    assert db.scalar(select(Notification).where(Notification.status == "FAILED")) is not None
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) >= 5


def test_malformed_model_intent_returns_safe_response_without_500(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    class MalformedInterpreter:
        def interpret(self, *args, **kwargs):
            malformed = CreateRecurringReminderIntent.model_construct(
                task="check report",
                weekdays=(7,),
                hour=8,
                minute=0,
                start_local=datetime(2026, 11, 2, 8),
                timezone_name="Asia/Karachi",
            )
            return Interpretation(intent=malformed)

    api_client.app.state.reminder_interpreter = MalformedInterpreter()
    response = post_update(
        api_client,
        update_factory(304, text="Remind me every Monday at 8 AM to check the report."),
    )

    assert response.status_code == 200
    assert "couldn't understand" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL row locks",
)
def test_postgres_concurrent_materializers_do_not_duplicate_instances(
    api_client, db, update_factory, session_factory
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(305, text="Remind me every Monday at 8 AM to check concurrency."),
    ).status_code == 200
    before = db.scalar(select(func.count()).select_from(ReminderOccurrence))
    errors: list[Exception] = []

    def materialize() -> None:
        try:
            materialize_active_recurring_reminders(
                session_factory,
                now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
                horizon_days=35,
                batch_size=1,
            )
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [Thread(target=materialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == before
    assert db.scalar(select(func.count()).select_from(Notification)) == before


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires PostgreSQL integration database",
)
def test_postgres_restart_replay_and_dst_preserve_recurrence(
    session_factory,
) -> None:
    with session_factory.begin() as session:
        user = User(name="DST", timezone="America/New_York", onboarding_status="ACTIVE")
        session.add(user)
        session.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel_type="TELEGRAM",
            external_id=f"dst-{uuid4().hex}",
            status="LINKED",
            telegram_chat_id=999999,
        )
        session.add(identity)
        session.flush()
        reminder = Reminder(
            user_id=user.id,
            task="DST check",
            schedule_type="RECURRING",
            start_local="2027-03-07T08:00",
            timezone="America/New_York",
            recurrence_rule="FREQ=WEEKLY;BYDAY=SU;BYHOUR=08;BYMINUTE=00",
            status="ACTIVE",
        )
        session.add(reminder)
        session.flush()
        materialize_recurring_reminder(
            session,
            reminder,
            now=datetime(2027, 3, 1, 12, tzinfo=timezone.utc),
            horizon_days=21,
        )

    with session_factory.begin() as session:
        reminder = session.scalar(select(Reminder).where(Reminder.task == "DST check"))
        assert reminder is not None
        materialize_recurring_reminder(
            session,
            reminder,
            now=datetime(2027, 3, 1, 12, tzinfo=timezone.utc),
            horizon_days=21,
        )

    with session_factory() as session:
        occurrences = session.scalars(
            select(ReminderOccurrence).where(ReminderOccurrence.reminder_id == reminder.id)
        ).all()
        assert len(occurrences) == 3
        assert len({occurrence.recurrence_instance_key for occurrence in occurrences}) == 3
        assert all(
            occurrence.scheduled_at.astimezone(ZoneInfo("America/New_York")).hour
            == 8
            for occurrence in occurrences
        )
