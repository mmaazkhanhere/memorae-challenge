from datetime import timedelta

from sqlalchemy import select

from reminder_agent.infrastructure.database.models import Notification, Reminder, ReminderOccurrence
from reminder_agent.infrastructure.runtime.worker import NotificationWorker

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def _onboard(client, update_factory) -> None:
    assert _post_update(client, update_factory(1, text="/start")).status_code == 200
    assert _post_update(client, update_factory(2, text="Asia/Karachi")).status_code == 200


def test_telegram_reminder_is_confirmed_persisted_and_delivered_when_due(
    api_client, telegram_client, db, fixed_now, session_factory, update_factory
) -> None:
    _onboard(api_client, update_factory)

    assert _post_update(
        api_client, update_factory(3, text="Remind me in five minutes to call the bank.")
    ).json() == {"ok": True}
    assert "today at 5:05 PM PKT" in telegram_client.messages[-1][1]

    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    notification = db.scalar(select(Notification))
    assert reminder is not None and occurrence is not None and notification is not None
    assert (reminder.status, occurrence.status, notification.status) == (
        "ACTIVE",
        "PENDING",
        "PENDING",
    )

    worker = NotificationWorker(
        session_factory,
        telegram_client,
        clock=lambda: fixed_now + timedelta(minutes=5),
    )
    assert worker.run_once() == 1

    db.expire_all()
    delivered = db.get(Notification, notification.id)
    delivered_occurrence = db.get(ReminderOccurrence, occurrence.id)
    assert delivered is not None and delivered.status == "SENT"
    assert delivered.sent_at is not None
    assert delivered_occurrence is not None and delivered_occurrence.status == "NOTIFIED"
    assert telegram_client.messages[-1] == (2001, "Reminder: call the bank")
