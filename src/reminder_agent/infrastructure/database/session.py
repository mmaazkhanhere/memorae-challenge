from collections.abc import Generator
from contextlib import AbstractContextManager
from typing import Protocol

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from reminder_agent.infrastructure.config.settings import get_settings


class SessionFactory(Protocol):
    """Sessionmaker surface shared by transaction-owning infrastructure code."""

    def __call__(self) -> Session: ...

    def begin(self) -> AbstractContextManager[Session]: ...


def build_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
        hide_parameters=True,
    )


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
