from sqlalchemy import func, select
from sqlalchemy.orm import Session

from reminder_agent.features.onboarding.message_templates import ONBOARDING_MESSAGES
from reminder_agent.infrastructure.database.models import ChannelIdentity, IdempotencyRecord, User

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def test_first_start_creates_pending_account_and_requests_timezone(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    response = post_update(api_client, update_factory(1))

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(telegram_client.messages) == 1
    assert telegram_client.messages[0][1] in {
        template.format(name="Ada") for template in ONBOARDING_MESSAGES["welcome"]
    }

    user = db.scalar(select(User))
    identity = db.scalar(select(ChannelIdentity))
    assert user is not None
    assert user.name == "Ada"
    assert user.onboarding_status == "PENDING"
    assert user.timezone is None
    assert identity is not None
    assert identity.external_id == "1001"
    assert identity.telegram_chat_id == 2001
    assert identity.user_id == user.id


def test_repeated_start_reuses_account_and_is_idempotent(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    update = update_factory(10)
    assert post_update(api_client, update).status_code == 200
    assert post_update(api_client, update).status_code == 200

    assert db.scalar(select(func.count()).select_from(User)) == 1
    assert db.scalar(select(func.count()).select_from(ChannelIdentity)) == 1
    assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 1
    assert len(telegram_client.messages) == 1


def test_invalid_timezone_keeps_user_pending_and_does_not_confirm_success(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    assert post_update(api_client, update_factory(20)).status_code == 200
    response = post_update(api_client, update_factory(21, text="Mars/Olympus"))

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["invalid_timezone"])
    user = db.scalar(select(User))
    assert user is not None
    assert user.onboarding_status == "PENDING"
    assert user.timezone is None


def test_valid_timezone_activates_user(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    assert post_update(api_client, update_factory(30)).status_code == 200
    response = post_update(api_client, update_factory(31, text="Asia/Karachi"))

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in {
        template.format(name="Ada", timezone_name="Asia/Karachi")
        for template in ONBOARDING_MESSAGES["onboarding_complete"]
    }
    user = db.scalar(select(User))
    assert user is not None
    assert user.onboarding_status == "ACTIVE"
    assert user.timezone == "Asia/Karachi"
    identities = db.scalars(select(ChannelIdentity)).all()
    assert [identity.channel_type for identity in identities] == ["TELEGRAM"]

    assert post_update(api_client, update_factory(32)).status_code == 200
    assert telegram_client.messages[-1][1] in {
        template.format(name="Ada")
        for template in ONBOARDING_MESSAGES["already_onboarded_without_phone"]
    }


def test_unknown_sender_cannot_see_existing_account(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    assert (
        post_update(api_client, update_factory(40, user_id=1001, first_name="Ada")).status_code
        == 200
    )
    response = post_update(
        api_client,
        update_factory(
            41,
            user_id=2002,
            chat_id=3002,
            text="What is my account?",
            first_name="Bob",
        ),
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["unknown_sender"])
    assert db.scalar(select(func.count()).select_from(User)) == 1


def test_group_start_is_ignored_without_creating_an_account(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    response = post_update(api_client, update_factory(50, chat_type="group"))

    assert response.status_code == 200
    assert telegram_client.messages == []
    assert db.scalar(select(func.count()).select_from(User)) == 0


def test_invalid_webhook_secret_is_rejected_without_side_effects(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    response = api_client.post(
        "/webhook/telegram",
        json=update_factory(60),
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
    )

    assert response.status_code == 401
    assert telegram_client.messages == []
    assert db.scalar(select(func.count()).select_from(User)) == 0


def test_malformed_update_is_rejected(api_client, telegram_client) -> None:
    response = post_update(api_client, {"update_id": "not-an-integer"})

    assert response.status_code == 400
    assert telegram_client.messages == []


def test_reusing_update_id_with_different_payload_is_rejected(
    api_client, telegram_client, update_factory
) -> None:
    assert post_update(api_client, update_factory(70)).status_code == 200
    response = post_update(api_client, update_factory(70, first_name="Different"))

    assert response.status_code == 409
    assert len(telegram_client.messages) == 1


def test_database_failure_does_not_send_false_success(
    session_factory, telegram_client, update_factory
) -> None:
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session as SQLAlchemySession

    from reminder_agent.application.main import create_app
    from reminder_agent.infrastructure.config.settings import Settings

    class FailingSession(SQLAlchemySession):
        def flush(self, *args, **kwargs) -> None:
            raise RuntimeError("database write failed")

    def failing_factory() -> FailingSession:
        return FailingSession(bind=session_factory.kw["bind"])

    app = create_app(
        Settings(
            database_url="sqlite://",
            telegram_webhook_secret="test-secret",
            telegram_bot_token="test",
        ),
        telegram_client=telegram_client,
        session_factory=failing_factory,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/webhook/telegram", json=update_factory(80), headers=HEADERS)

    assert response.status_code == 500
    assert telegram_client.messages == []


def test_telegram_502_is_acknowledged_without_duplicate_retry(
    session_factory, update_factory
) -> None:
    from fastapi.testclient import TestClient

        from reminder_agent.application.main import create_app
    from reminder_agent.channels.telegram.provider import TelegramAPIError
    from reminder_agent.infrastructure.config.settings import Settings

    class FailingTelegramClient:
        attempts = 0
        delivered = 0

        def send_message(self, chat_id: int, text: str) -> None:
            self.attempts += 1
            # Model Telegram accepting the message before a gateway returns 502.
            self.delivered += 1
            raise TelegramAPIError("bad gateway", status_code=502, failure_kind="http_error")

    failing_client = FailingTelegramClient()
    app = create_app(
        Settings(database_url="sqlite://", telegram_webhook_secret="test-secret"),
        telegram_client=failing_client,
        session_factory=session_factory,
    )
    update = update_factory(90)
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/webhook/telegram", json=update, headers=HEADERS)
        retry = client.post("/webhook/telegram", json=update, headers=HEADERS)

    assert first.status_code == 200
    assert retry.status_code == 200
    assert failing_client.attempts == 1
    assert failing_client.delivered == 1
    with session_factory() as db:
        delivery = db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.source == "TELEGRAM",
                IdempotencyRecord.idempotency_key == "90",
            )
        )
        assert delivery is not None
        assert delivery.delivery_status == "UNKNOWN"


def test_health_and_readiness(api_client) -> None:
    assert api_client.get("/health").json() == {"status": "ok"}
    assert api_client.get("/ready").json() == {"status": "ready"}
