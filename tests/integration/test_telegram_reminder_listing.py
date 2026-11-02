from datetime import datetime, timezone

from sqlalchemy import func, select

from reminder_agent.db.models import (
    ConversationSession,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
)


HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def onboard(api_client, update_factory) -> None:
    assert post_update(api_client, update_factory(500)).status_code == 200
    assert post_update(api_client, update_factory(501, text="Asia/Karachi")).status_code == 200


def test_telegram_listing_returns_one_time_and_recurring_items_without_writes(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(502, text="Remind me tomorrow at 9 AM to renew the domain."),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(503, text="Remind me every Saturday at 10 AM to review the report."),
    ).status_code == 200

    before = {
        "reminders": db.scalar(select(func.count()).select_from(Reminder)),
        "occurrences": db.scalar(select(func.count()).select_from(ReminderOccurrence)),
        "notifications": db.scalar(select(func.count()).select_from(Notification)),
        "sessions": db.scalar(select(func.count()).select_from(ConversationSession)),
        "idempotency": db.scalar(select(func.count()).select_from(IdempotencyRecord)),
    }
    response = post_update(api_client, update_factory(504, text="What do I have tomorrow?"))

    assert response.status_code == 200
    text = telegram_client.messages[-1][1]
    assert "renew the domain" in text
    assert "review the report" in text
    assert "2026-10-31 09:00 PKT" in text
    assert "2026-10-31 10:00 PKT" in text
    after = {
        "reminders": db.scalar(select(func.count()).select_from(Reminder)),
        "occurrences": db.scalar(select(func.count()).select_from(ReminderOccurrence)),
        "notifications": db.scalar(select(func.count()).select_from(Notification)),
        "sessions": db.scalar(select(func.count()).select_from(ConversationSession)),
        "idempotency": db.scalar(select(func.count()).select_from(IdempotencyRecord)),
    }
    assert after == before


def test_telegram_listing_is_owner_scoped_and_empty_is_explicit(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    response = post_update(api_client, update_factory(505, text="What do I have on 2026-11-03?"))

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] == "You have no active reminders on 2026-11-03."


def test_listing_does_not_discard_an_unfinished_reminder_clarification(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert (
        post_update(api_client, update_factory(506, text="Remind me to call the bank")).status_code
        == 200
    )
    session_before = db.scalar(select(ConversationSession))
    assert session_before is not None and session_before.status == "AWAITING_CLARIFICATION"

    response = post_update(api_client, update_factory(507, text="What do I have tomorrow?"))

    assert response.status_code == 200
    session_after = db.scalar(select(ConversationSession))
    assert session_after is not None
    assert session_after.status == "AWAITING_CLARIFICATION"
    assert session_after.pending_clarification == session_before.pending_clarification
    assert "no active reminders" in telegram_client.messages[-1][1].lower()
