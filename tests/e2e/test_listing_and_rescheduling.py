from datetime import UTC, datetime

from sqlalchemy import select

from reminder_agent.infrastructure.db.models import Notification, Reminder, ReminderOccurrence

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def _onboard(client, update_factory) -> None:
    assert _post_update(client, update_factory(1, text="/start")).status_code == 200
    assert _post_update(client, update_factory(2, text="UTC")).status_code == 200


def test_user_can_list_and_reschedule_a_reminder_without_creating_another_one(
    api_client, telegram_client, db, update_factory
) -> None:
    _onboard(api_client, update_factory)
    assert (
        _post_update(
            api_client, update_factory(3, text="Remind me tomorrow at 9 AM to call the bank")
        ).status_code
        == 200
    )
    assert (
        _post_update(
            api_client, update_factory(4, text="Remind me tomorrow at 10 AM to review the report")
        ).status_code
        == 200
    )

    assert _post_update(api_client, update_factory(5, text="What do I have tomorrow?")).json() == {
        "ok": True
    }
    listed = telegram_client.messages[-1][1]
    assert "call the bank" in listed
    assert "review the report" in listed
    assert "tomorrow:" in listed
    assert "09:00 UTC" in listed

    assert _post_update(
        api_client, update_factory(6, text="Move the bank reminder to tomorrow at 11 AM")
    ).json() == {"ok": True}
    assert "Moved 'call the bank'" in telegram_client.messages[-1][1]
    assert "11:00" in telegram_client.messages[-1][1]

    db.expire_all()
    reminders = db.scalars(select(Reminder).order_by(Reminder.task)).all()
    assert len(reminders) == 2
    bank_reminder = next(reminder for reminder in reminders if reminder.task == "call the bank")
    assert bank_reminder.version == 2
    occurrence = db.scalar(
        select(ReminderOccurrence).where(ReminderOccurrence.reminder_id == bank_reminder.id)
    )
    assert occurrence is not None
    assert occurrence.next_due_at.replace(tzinfo=UTC) == datetime(2026, 10, 31, 11, 0, tzinfo=UTC)
    notifications = db.scalars(
        select(Notification)
        .where(Notification.occurrence_id == occurrence.id)
        .order_by(Notification.sequence)
    ).all()
    assert [notification.status for notification in notifications] == ["CANCELED", "PENDING"]
