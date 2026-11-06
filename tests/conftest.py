from collections.abc import Iterator
from datetime import datetime, timezone
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from reminder_agent.db.models import Base
from reminder_agent.main import REQUIRED_SCHEMA_VERSION, create_app
from reminder_agent.settings import Settings


class FakeTelegramClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []
        self.reply_markups: list[dict | None] = []
        self.callback_acknowledgements: list[str] = []

    def send_message(self, chat_id: int, text: str, *, reply_markup=None) -> None:
        self.messages.append((chat_id, text))
        self.reply_markups.append(reply_markup)

    def answer_callback_query(self, callback_query_id: str) -> None:
        self.callback_acknowledgements.append(callback_query_id)


@pytest.fixture()
def fixed_now() -> datetime:
    return datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker[Session]]:
    database_url = os.getenv("TEST_DATABASE_URL", "sqlite://")
    application_database_url = os.getenv("DATABASE_URL")
    if application_database_url and database_url != "sqlite://":
        if make_url(database_url).render_as_string(hide_password=True) == make_url(
            application_database_url
        ).render_as_string(hide_password=True):
            raise RuntimeError(
                "TEST_DATABASE_URL must be a separate database from DATABASE_URL; "
                "refusing to drop the application database"
            )
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS alembic_version "
                "(version_num VARCHAR(32) NOT NULL)"
            )
        )
        connection.execute(text("DELETE FROM alembic_version"))
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:version)"),
            {"version": REQUIRED_SCHEMA_VERSION},
        )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        # The fixture creates Alembic's bookkeeping table separately because it
        # is not part of the ORM metadata. Remove it with the schema so a
        # subsequent migration run cannot mistake an empty database for a
        # database at the required revision.
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        engine.dispose()


@pytest.fixture()
def telegram_client() -> FakeTelegramClient:
    return FakeTelegramClient()


@pytest.fixture()
def api_client(session_factory, telegram_client, fixed_now) -> Iterator[TestClient]:
    settings = Settings(
        database_url="sqlite://",
        telegram_bot_token="test-token",
        telegram_webhook_secret="test-secret",
        # Tests must not inherit a developer's live provider key from .env.
        llm_api_key=None,
        embedding_provider="disabled",
        embedding_api_key=None,
    )
    app = create_app(
        settings,
        telegram_client=telegram_client,
        session_factory=session_factory,
        clock=lambda: fixed_now,
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
