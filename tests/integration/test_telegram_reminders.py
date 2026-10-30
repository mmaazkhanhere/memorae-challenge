from datetime import datetime, timedelta, timezone
import hashlib

from sqlalchemy import func, select

from reminder_agent.db.models import (
    ChannelIdentity,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.intent import CreateOneTimeReminderIntent, Interpretation
from reminder_agent.main import create_app
from reminder_agent.reminders import handle_reminder_text
from reminder_agent.settings import Settings
from reminder_agent.telegram import TelegramUpdate


HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def utc_timestamp(value: datetime) -> datetime:
    """Normalize SQLite-naive and PostgreSQL-aware timestamps for assertions."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def onboard(api_client, update_factory, *, user_id=1001, chat_id=2001) -> None:
    assert (
        post_update(api_client, update_factory(100, user_id=user_id, chat_id=chat_id)).status_code
        == 200
    )
    assert (
        post_update(
            api_client,
            update_factory(
                101,
                user_id=user_id,
                chat_id=chat_id,
                text="Asia/Karachi",
            ),
        ).status_code
        == 200
    )


def test_one_time_relative_reminder_persists_all_rows_and_confirms_local_time(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    update = update_factory(
        102,
        text="Remind me in five minutes to call the bank.",
    )

    response = post_update(api_client, update)

    assert response.status_code == 200
    assert "2026-10-30 17:05" in telegram_client.messages[-1][1]
    assert "Telegram" in telegram_client.messages[-1][1]
    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    notification = db.scalar(select(Notification))
    assert reminder is not None
    owner = db.get(User, reminder.user_id)
    assert owner is not None
    assert owner.name == "Ada"
    assert reminder.task == "call the bank"
    assert reminder.schedule_type == "ONE_TIME"
    assert reminder.status == "ACTIVE"
    assert occurrence is not None and occurrence.reminder_id == reminder.id
    assert utc_timestamp(occurrence.scheduled_at) == datetime(
        2026, 10, 30, 12, 5, tzinfo=timezone.utc
    )
    assert notification is not None and notification.occurrence_id == occurrence.id
    assert notification.status == "PENDING"
    assert utc_timestamp(notification.due_at) == datetime(
        2026, 10, 30, 12, 5, tzinfo=timezone.utc
    )


def test_tomorrow_schedule_is_stored_as_utc_and_displayed_in_local_time(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    response = post_update(
        api_client,
        update_factory(103, text="Remind me tomorrow at 9 to renew the domain."),
    )

    assert response.status_code == 200
    assert "2026-10-31 09:00" in telegram_client.messages[-1][1]
    occurrence = db.scalar(select(ReminderOccurrence))
    assert occurrence is not None
    assert utc_timestamp(occurrence.scheduled_at) == datetime(
        2026, 10, 31, 4, 0, tzinfo=timezone.utc
    )


def test_transient_confirmation_failure_retries_and_records_sent(
    session_factory, fixed_now, update_factory
) -> None:
    from fastapi.testclient import TestClient
    from reminder_agent.telegram import TelegramAPIError

    class FailOnceTelegramClient:
        def __init__(self) -> None:
            self.attempts = 0
            self.messages: list[tuple[int, str]] = []

        def send_message(self, chat_id: int, text: str, *, reply_markup=None) -> None:
            self.attempts += 1
            if self.attempts == 3:
                raise TelegramAPIError("temporary timeout", failure_kind="network_timeout")
            self.messages.append((chat_id, text))

    telegram_client = FailOnceTelegramClient()
    app = create_app(
        Settings(
            database_url="sqlite://",
            telegram_webhook_secret="test-secret",
            telegram_confirmation_retry_delay_seconds=0,
        ),
        telegram_client=telegram_client,
        session_factory=session_factory,
        clock=lambda: fixed_now,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        assert post_update(client, update_factory(110)).status_code == 200
        assert post_update(client, update_factory(111, text="Asia/Karachi")).status_code == 200
        response = post_update(
            client,
            update_factory(112, text="Remind me in five minutes to call the bank."),
        )

    assert response.status_code == 200
    assert telegram_client.attempts == 4
    assert len(telegram_client.messages) == 3
    assert "I'll remind you" in telegram_client.messages[-1][1]
    with session_factory() as db:
        delivery = db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.idempotency_key == "112",
                IdempotencyRecord.source == "TELEGRAM",
            )
        )
        assert delivery is not None
        assert delivery.delivery_status == "SENT"


def test_replaying_update_does_not_duplicate_rows_or_confirmation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    update = update_factory(104, text="Remind me in five minutes to stretch.")

    assert post_update(api_client, update).status_code == 200
    assert post_update(api_client, update).status_code == 200

    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 1
    assert db.scalar(select(func.count()).select_from(Notification)) == 1
    assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 3
    assert len(telegram_client.messages) == 3  # onboarding start, timezone, reminder


def test_idempotency_unique_race_recovery_returns_duplicate(
    session_factory, fixed_now, update_factory
) -> None:
    """A loser of the unique-key race reloads the committed winner."""
    from reminder_agent.reminders import _recover_from_idempotency_race

    update = TelegramUpdate.model_validate(
        update_factory(109, text="Remind me in five minutes to call the bank.")
    )
    request_hash = hashlib.sha256(
        update.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
    ).hexdigest()

    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        db.add(
            IdempotencyRecord(
                source="TELEGRAM",
                idempotency_key=str(update.update_id),
                user_id=user.id,
                operation="CREATE_REMINDER",
                request_hash=request_hash,
                status="SUCCEEDED",
                delivery_status="PENDING",
                response={
                    "chat_id": 2001,
                    "text": "The reminder was already processed.",
                    "reply_markup": None,
                },
                created_at=fixed_now,
                updated_at=fixed_now,
            )
        )

    with session_factory() as db:
        result = _recover_from_idempotency_race(db, update)

    assert result is not None
    assert result.is_duplicate is True
    assert result.should_send is False
    assert result.text == "The reminder was already processed."


def test_invalid_schedule_is_acknowledged_without_creating_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    response = post_update(
        api_client,
        update_factory(105, text="Remind me in zero minutes to call the bank."),
    )

    assert response.status_code == 200
    assert "future" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_past_typed_schedule_is_rejected_without_creating_reminder(
    session_factory, fixed_now, update_factory
) -> None:
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel_type="TELEGRAM",
            external_id="1001",
            telegram_chat_id=2001,
            status="LINKED",
        )
        db.add(identity)

    class PastInterpreter:
        def interpret(self, text, *, timezone_name, now):
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="call the bank",
                    due_at=fixed_now - timedelta(minutes=1),
                    timezone_name="UTC",
                )
            )

    with session_factory() as db:
        result = handle_reminder_text(
            db,
            TelegramUpdate.model_validate(
                update_factory(108, text="Remind me at a time that has passed")
            ),
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=PastInterpreter(),
            now=fixed_now,
        )

        assert "future" in result.text
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_intent_timezone_mismatch_is_rejected_without_creating_reminder(
    session_factory, fixed_now, update_factory
) -> None:
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel_type="TELEGRAM",
            external_id="1001",
            telegram_chat_id=2001,
            status="LINKED",
        )
        db.add(identity)

    class WrongTimezoneInterpreter:
        def interpret(self, text, *, timezone_name, now):
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="call the bank",
                    due_at=fixed_now + timedelta(minutes=5),
                    timezone_name="Asia/Karachi",
                )
            )

    with session_factory() as db:
        result = handle_reminder_text(
            db,
            TelegramUpdate.model_validate(
                update_factory(113, text="Remind me in five minutes to call the bank")
            ),
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=WrongTimezoneInterpreter(),
            now=fixed_now,
        )

        assert "time zone" in result.text
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_cross_user_identity_cannot_be_used_for_reminder_creation(
    session_factory, fixed_now, update_factory
) -> None:
    from reminder_agent.db.models import ChannelIdentity, User

    with session_factory.begin() as db:
        first = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        second = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add_all([first, second])
        db.flush()
        first_identity = ChannelIdentity(
            user_id=first.id,
            channel_type="TELEGRAM",
            external_id="1001",
            telegram_chat_id=2001,
            status="LINKED",
        )
        second_identity = ChannelIdentity(
            user_id=second.id,
            channel_type="TELEGRAM",
            external_id="2002",
            telegram_chat_id=3002,
            status="LINKED",
        )
        db.add_all([first_identity, second_identity])

    class FakeInterpreter:
        def interpret(self, text, *, timezone_name, now):
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="private task",
                    due_at=fixed_now + timedelta(minutes=5),
                    timezone_name="UTC",
                )
            )

    with session_factory() as db:
        result = handle_reminder_text(
            db,
            TelegramUpdate.model_validate(
                update_factory(106, user_id=1001, chat_id=2001, text="Remind me in five minutes")
            ),
            user_id=second.id,
            identity_id=second_identity.id,
            telegram_user_id=1001,
            interpreter=FakeInterpreter(),
            now=fixed_now,
        )
        assert "couldn't verify" in result.text
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0


def test_database_failure_rolls_back_reminder_graph_without_confirmation(
    session_factory, telegram_client, fixed_now, update_factory
) -> None:
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session as SQLAlchemySession

    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel_type="TELEGRAM",
            external_id="1001",
            telegram_chat_id=2001,
            status="LINKED",
        )
        db.add(identity)

    class FailingSession(SQLAlchemySession):
        def flush(self, *args, **kwargs) -> None:
            raise RuntimeError("database write failed")

    failing_factory = lambda: FailingSession(bind=session_factory.kw["bind"])
    app = create_app(
        Settings(database_url="sqlite://", telegram_webhook_secret="test-secret"),
        telegram_client=telegram_client,
        session_factory=failing_factory,
        clock=lambda: fixed_now,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = post_update(
            client,
            update_factory(107, text="Remind me in five minutes to call the bank."),
        )

    assert response.status_code == 500
    assert telegram_client.messages == []
    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0
