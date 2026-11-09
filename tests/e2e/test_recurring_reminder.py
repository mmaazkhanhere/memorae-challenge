from datetime import UTC, datetime

from sqlalchemy import select

from reminder_agent.infrastructure.db.models import Notification, Reminder, ReminderOccurrence
from reminder_agent.infrastructure.worker import NotificationWorker

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def _onboard(client, update_factory) -> None:
    assert _post_update(client, update_factory(1, text="/start")).status_code == 200
    assert _post_update(client, update_factory(2, text="Asia/Karachi")).status_code == 200


def test_recurring_telegram_reminder_survives_creation_and_delivers_its_first_instance(
    api_client, telegram_client, db, session_factory, update_factory
) -> None:
    _onboard(api_client, update_factory)

    assert _post_update(
        api_client, update_factory(3, text="Remind me every Monday at 8 AM to call the bank.")
    ).json() == {"ok": True}
    assert "every Monday at 8:00 AM PKT on Telegram" in telegram_client.messages[-1][1]

    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    assert reminder.schedule_type == "RECURRING"
    first_occurrence = db.scalar(
        select(ReminderOccurrence)
        .where(ReminderOccurrence.reminder_id == reminder.id)
        .order_by(ReminderOccurrence.next_due_at)
    )
    assert first_occurrence is not None

    worker = NotificationWorker(
        session_factory,
        telegram_client,
        clock=lambda: datetime(2026, 11, 2, 3, 0, 1, tzinfo=UTC),
    )
    assert worker.run_once() == 1

    db.expire_all()
    delivered = db.scalar(
        select(Notification).where(Notification.occurrence_id == first_occurrence.id)
    )
    remaining = db.scalars(
        select(ReminderOccurrence)
        .where(
            ReminderOccurrence.reminder_id == reminder.id,
            ReminderOccurrence.status == "PENDING",
        )
        .order_by(ReminderOccurrence.next_due_at)
    ).all()
    refreshed_reminder = db.get(Reminder, reminder.id)
    assert delivered is not None and delivered.status == "SENT"
    assert refreshed_reminder is not None and refreshed_reminder.status == "ACTIVE"
    assert remaining and remaining[0].next_due_at > first_occurrence.next_due_at
    assert telegram_client.messages[-1] == (2001, "Reminder: call the bank")
