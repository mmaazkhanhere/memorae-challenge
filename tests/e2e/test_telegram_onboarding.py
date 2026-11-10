from sqlalchemy import select

from reminder_agent.infrastructure.database.models import ChannelIdentity, User

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def test_new_telegram_user_completes_onboarding_and_links_their_phone(
    api_client, telegram_client, db, update_factory
) -> None:
    """Exercise the complete onboarding conversation through the webhook boundary."""

    assert _post_update(api_client, update_factory(1, text="/start")).json() == {"ok": True}
    assert "Asia/Karachi" in telegram_client.messages[-1][1]

    assert _post_update(api_client, update_factory(2, text="Asia/Karachi")).json() == {"ok": True}
    assert telegram_client.reply_markups[-1] == {
        "keyboard": [
            [{"text": "Share my phone number", "request_contact": True}],
            [{"text": "Skip for now"}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }

    contact = {"phone_number": "+923001234567", "user_id": 1001}
    assert _post_update(api_client, update_factory(3, text=None, contact=contact)).json() == {
        "ok": True
    }
    assert "linked" in telegram_client.messages[-1][1].lower() or "connected" in telegram_client.messages[-1][1].lower()
    assert telegram_client.reply_markups[-1] == {"remove_keyboard": True}

    user = db.scalar(select(User).where(User.name == "Ada"))
    assert user is not None
    assert (user.onboarding_status, user.timezone) == ("ACTIVE", "Asia/Karachi")
    identities = db.scalars(select(ChannelIdentity).where(ChannelIdentity.user_id == user.id)).all()
    assert {
        (identity.channel_type, identity.status, identity.external_id) for identity in identities
    } == {
        ("TELEGRAM", "LINKED", "1001"),
        ("PHONE", "LINKED", "+923001234567"),
    }
