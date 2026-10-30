from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from reminder_agent.main import create_app
from reminder_agent.settings import Settings


def test_ready_rejects_database_with_only_alembic_version() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))

    factory = sessionmaker(bind=engine, class_=Session)
    app = create_app(
        Settings(
            _env_file=None,
            database_url="sqlite://",
            telegram_webhook_secret="test-secret",
        ),
        session_factory=factory,
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/ready")
    finally:
        engine.dispose()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database schema is incomplete"}
