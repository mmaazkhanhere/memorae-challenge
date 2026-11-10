from __future__ import annotations

from reminder_agent.application import conversation_coordinator as reminders
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.infrastructure.database.models import ChannelIdentity, IdempotencyRecord, User


def _active_identity(db, *, telegram_user_id: int = 1001, chat_id: int = 2001):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
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


def test_telegram_request_adapter_revalidates_persisted_identity_and_session(
    session_factory, update_factory
) -> None:
    update = TelegramUpdate.model_validate(update_factory(903, text="list reminders"))

    with session_factory.begin() as db:
        user, identity = _active_identity(db)
        assert reminders._authorized_telegram_identity(
            user, identity, telegram_user_id=1001, chat_id=2001
        )
        assert not reminders._authorized_telegram_identity(
            user, identity, telegram_user_id=9999, chat_id=2001
        )
        request = reminders._application_request_from_telegram_update(
            update, user=user, identity=identity
        )

    assert request.context.canonical_user_id == user.id
    assert request.context.verified_identity.identity_id == identity.id
    assert request.context.external_session_id == "2001"
    assert request.message_text == "list reminders"
    assert request.idempotency_key == "903"


def test_telegram_response_adapter_stages_and_replays_the_same_reply(
    session_factory, update_factory
) -> None:
    update = TelegramUpdate.model_validate(update_factory(904, text="buy milk"))

    with session_factory.begin() as db:
        user, _ = _active_identity(db)
        staged = reminders._store_response(
            db,
            update=update,
            user_id=user.id,
            text="I'll remind you.",
            operation="CREATE_REMINDER",
        )

    with session_factory() as db:
        record = db.query(IdempotencyRecord).one()
        replay = reminders._reply_from_record(record)

    assert staged.chat_id == replay.chat_id == 2001
    assert staged.text == replay.text == "I'll remind you."
    assert staged.idempotency_key == replay.idempotency_key == "904"
    assert replay.should_send is False
    assert replay.is_duplicate is True
    assert record.response == {
        "chat_id": 2001,
        "text": "I'll remind you.",
        "reply_markup": None,
        "request_text": "buy milk",
    }
