from collections.abc import Iterator
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from reminder_agent.db.models import Base
from reminder_agent.main import create_app
from reminder_agent.settings import Settings


class FakeTelegramClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []
        self.reply_markups: list[dict | None] = []

    def send_message(self, chat_id: int, text: str, *, reply_markup=None) -> None:
        self.messages.append((chat_id, text))
        self.reply_markups.append(reply_markup)


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker[Session]]:
    database_url = os.getenv("TEST_DATABASE_URL", "sqlite://")
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def telegram_client() -> FakeTelegramClient:
    return FakeTelegramClient()


@pytest.fixture()
def api_client(session_factory, telegram_client) -> Iterator[TestClient]:
    settings = Settings(
        database_url="sqlite://",
        telegram_bot_token="test-token",
        telegram_webhook_secret="test-secret",
    )
    app = create_app(
        settings,
        telegram_client=telegram_client,
        session_factory=session_factory,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture()
def db(session_factory) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def telegram_update(
    update_id: int,
    *,
    user_id: int = 1001,
    chat_id: int = 2001,
    text: str | None = "/start",
    chat_type: str = "private",
    first_name: str = "Ada",
    contact: dict | None = None,
) -> dict:
    message = {
        "message_id": update_id,
        "from": {"id": user_id, "first_name": first_name},
        "chat": {"id": chat_id, "type": chat_type},
    }
    if text is not None:
        message["text"] = text
    if contact is not None:
        message["contact"] = contact
    return {"update_id": update_id, "message": message}


@pytest.fixture()
def update_factory():
    return telegram_update
