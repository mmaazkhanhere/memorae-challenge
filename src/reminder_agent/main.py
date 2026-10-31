from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from collections.abc import Callable, Generator
from datetime import datetime, timezone
from typing import Awaitable, Callable as TypingCallable

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from reminder_agent.api.telegram import router as telegram_router
from reminder_agent.db.session import get_db
from reminder_agent.intent import (
    OpenAICompatibleIntentInterpreter,
    OpenAICompatibleIntentRouter,
    ReminderIntentInterpreter,
)
from reminder_agent.observability import configure_logging, log_event, request_id
from reminder_agent.settings import Settings, get_settings, validate_runtime_settings
from reminder_agent.telegram import TelegramClient

REQUIRED_SCHEMA_TABLES = (
    "users",
    "channel_identities",
    "idempotency_records",
    "reminders",
    "reminder_occurrences",
    "notifications",
    "conversation_sessions",
)
REQUIRED_SCHEMA_VERSION = "0012_conversation_session_version"
REQUIRED_SCHEMA_COLUMNS = {
    "notifications": {
        "claimed_by",
        "claimed_until",
        "last_attempt_at",
        "last_error_kind",
    },
    "idempotency_records": {
        "delivery_attempt_count",
        "delivery_next_attempt_at",
        "delivery_claimed_by",
        "delivery_claimed_until",
        "delivery_last_error_kind",
    },
    "conversation_sessions": {
        "user_id",
        "channel_type",
        "external_session_id",
        "pending_clarification",
        "status",
        "version",
        "updated_at",
    },
}


def create_app(
    settings: Settings | None = None,
    *,
    telegram_client: TelegramClient | None = None,
    session_factory: Callable[[], Session] | None = None,
    reminder_interpreter=None,
    intent_router=None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    logger = configure_logging(
        runtime_settings.log_level,
        runtime_settings.log_format,
        runtime_settings.log_include_traceback,
        runtime_settings.log_color,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        validate_runtime_settings(
            runtime_settings,
            allow_local_database=session_factory is not None,
        )
        log_event(logger, logging.INFO, "runtime_configuration_validated")
        yield

    app = FastAPI(title="Memorae", lifespan=lifespan)
    app.state.settings = runtime_settings
    app.state.telegram_client = telegram_client
    model_interpreter = None
    if app.state.settings.llm_api_key:
        model_interpreter = OpenAICompatibleIntentInterpreter(
            app.state.settings.llm_base_url,
            app.state.settings.llm_api_key,
            app.state.settings.llm_extract_model,
            app.state.settings.llm_timeout_seconds,
        )
        app.state.intent_router = intent_router or OpenAICompatibleIntentRouter(
            app.state.settings.llm_base_url,
            app.state.settings.llm_api_key,
            app.state.settings.llm_intent_model,
            app.state.settings.llm_timeout_seconds,
        )
    else:
        app.state.intent_router = intent_router
    app.state.reminder_interpreter = reminder_interpreter or ReminderIntentInterpreter(
        model_interpreter
    )
    app.state.clock = clock or (lambda: datetime.now(timezone.utc))

    @app.middleware("http")
    async def request_logging(request: Request, call_next: TypingCallable[[Request], Awaitable[Response]]):
        correlation_id = request.headers.get("X-Request-ID") or request_id()
        request.state.request_id = correlation_id
        started = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            "http_request_started",
            request_id=correlation_id,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "http_request_failed",
                request_id=correlation_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                exc_info=True,
            )
            raise
        response.headers["X-Request-ID"] = correlation_id
        log_event(
            logger,
            logging.INFO,
            "http_request_completed",
            request_id=correlation_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return response

    if session_factory is not None:
        def override_db() -> Generator[Session, None, None]:
            db = session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(db: Session = Depends(get_db)) -> dict[str, str]:
        try:
            db.execute(text("SELECT 1"))
            inspector = inspect(db.get_bind())
            existing_tables = set(inspector.get_table_names())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc
        missing_tables = sorted(set(REQUIRED_SCHEMA_TABLES) - existing_tables)
        if missing_tables:
            log_event(
                logger,
                logging.ERROR,
                "database_schema_incomplete",
                missing_table_count=len(missing_tables),
            )
            raise HTTPException(status_code=503, detail="Database schema is incomplete")
        try:
            versions = db.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "database_migration_version_unavailable",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503, detail="Database migration state is unavailable"
            ) from exc
        if versions != [REQUIRED_SCHEMA_VERSION]:
            log_event(
                logger,
                logging.ERROR,
                "database_migration_incomplete",
                expected_version=REQUIRED_SCHEMA_VERSION,
                actual_versions=versions,
            )
            raise HTTPException(status_code=503, detail="Database migrations are incomplete")
        missing_columns = {
            table: sorted(columns - {column["name"] for column in inspector.get_columns(table)})
            for table, columns in REQUIRED_SCHEMA_COLUMNS.items()
        }
        missing_columns = {table: columns for table, columns in missing_columns.items() if columns}
        if missing_columns:
            log_event(
                logger,
                logging.ERROR,
                "database_schema_columns_incomplete",
                missing_column_table_count=len(missing_columns),
                missing_column_count=sum(len(columns) for columns in missing_columns.values()),
            )
            raise HTTPException(status_code=503, detail="Database schema columns are incomplete")
        return {"status": "ready"}

    app.include_router(telegram_router)
    return app


app = create_app()


def run() -> None:
    uvicorn.run("reminder_agent.main:app", host="0.0.0.0", port=8000, reload=False)
