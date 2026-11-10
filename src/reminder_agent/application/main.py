from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from collections.abc import Callable as TypingCallable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from reminder_agent.channels.telegram.provider import TelegramClient
from reminder_agent.channels.telegram.webhook import router as telegram_router
from reminder_agent.channels.vapi.webhook import router as vapi_router
from reminder_agent.features.intent.model.interpreter import OpenAICompatibleIntentInterpreter
from reminder_agent.features.intent.model.router import (
    IntentRouter,
    OpenAICompatibleIntentRouter,
)
from reminder_agent.features.intent.pipeline import ReminderIntentInterpreter
from reminder_agent.features.memory.retrieve.answering import (
    MemoryAnswerer,
    OpenAICompatibleMemoryAnswerer,
)
from reminder_agent.features.memory.retrieve.contracts import MemoryRetrievalPolicy
from reminder_agent.infrastructure.config.settings import (
    Settings,
    get_settings,
    validate_runtime_settings,
)
from reminder_agent.infrastructure.database.session import SessionFactory, get_db
from reminder_agent.infrastructure.observability import configure_logging, log_event, request_id
from reminder_agent.infrastructure.providers.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
    create_embedding_provider,
)
from reminder_agent.infrastructure.providers.llm import OpenAICompatibleLLMClient
from reminder_agent.infrastructure.providers.transcription import (
    AudioTranscriber,
    GroqWhisperTranscriber,
)

REQUIRED_SCHEMA_TABLES = (
    "users",
    "channel_identities",
    "idempotency_records",
    "reminders",
    "reminder_occurrences",
    "reminder_recurrence_exceptions",
    "notifications",
    "conversation_sessions",
    "memory_facts",
    "voice_call_authorizations",
)
REQUIRED_SCHEMA_VERSION = "0017_voice_call_authorizations"
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
    "reminder_occurrences": {"recurrence_instance_key"},
    "reminder_recurrence_exceptions": {"reminder_id", "occurrence_date"},
    "memory_facts": {
        "user_id",
        "content",
        "embedding",
        "embedding_status",
        "embedding_model",
        "embedding_content_hash",
        "embedding_attempts",
        "embedding_last_error",
        "embedding_updated_at",
        "created_at",
        "updated_at",
    },
    "voice_call_authorizations": {
        "provider_call_id",
        "user_id",
        "phone_identity_id",
        "status",
        "created_at",
        "expires_at",
        "ended_at",
    },
}


def create_app(
    settings: Settings | None = None,
    *,
    telegram_client: TelegramClient | None = None,
    session_factory: SessionFactory | None = None,
    reminder_interpreter: ReminderIntentInterpreter | None = None,
    intent_router: IntentRouter | None = None,
    memory_answerer: MemoryAnswerer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    transcriber: AudioTranscriber | None = None,
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
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        validate_runtime_settings(
            runtime_settings,
            allow_local_database=session_factory is not None,
        )
        log_event(logger, logging.INFO, "runtime_configuration_validated")
        yield

    app = FastAPI(title="Memorae", lifespan=lifespan)
    app.state.settings = runtime_settings
    app.state.telegram_client = telegram_client
    app.state.transcriber = transcriber or (
        GroqWhisperTranscriber(
            runtime_settings.stt_base_url,
            runtime_settings.stt_api_key,
            runtime_settings.stt_model,
            runtime_settings.stt_timeout_seconds,
        )
        if runtime_settings.stt_api_key
        else None
    )
    model_interpreter = None
    configured_memory_answerer = None
    if app.state.settings.llm_api_key:
        llm_client = OpenAICompatibleLLMClient(
            app.state.settings.llm_base_url,
            app.state.settings.llm_api_key,
            app.state.settings.llm_timeout_seconds,
        )
        model_interpreter = OpenAICompatibleIntentInterpreter(
            app.state.settings.llm_base_url,
            app.state.settings.llm_api_key,
            app.state.settings.llm_extract_model,
            app.state.settings.llm_timeout_seconds,
            client=llm_client,
        )
        app.state.intent_router = intent_router or OpenAICompatibleIntentRouter(
            app.state.settings.llm_base_url,
            app.state.settings.llm_api_key,
            app.state.settings.llm_intent_model,
            app.state.settings.llm_timeout_seconds,
            client=llm_client,
        )
        configured_memory_answerer = OpenAICompatibleMemoryAnswerer(
            app.state.settings.llm_base_url,
            app.state.settings.llm_api_key,
            app.state.settings.llm_extract_model,
            app.state.settings.llm_timeout_seconds,
            client=llm_client,
        )
    else:
        app.state.intent_router = intent_router
    app.state.reminder_interpreter = reminder_interpreter or ReminderIntentInterpreter(
        model_interpreter
    )
    app.state.memory_answerer = memory_answerer or configured_memory_answerer
    app.state.clock = clock or (lambda: datetime.now(UTC))
    app.state.embedding_provider = embedding_provider or create_embedding_provider(runtime_settings)
    app.state.embedding_privacy_policy = EmbeddingPrivacyPolicy(
        allow_sensitive_external=runtime_settings.embedding_allow_sensitive_external
    )
    app.state.memory_retrieval_policy = MemoryRetrievalPolicy(
        lexical_candidate_limit=runtime_settings.memory_lexical_candidate_limit,
        semantic_candidate_limit=runtime_settings.memory_semantic_candidate_limit,
        semantic_relevance_threshold=runtime_settings.memory_semantic_relevance_threshold,
        semantic_ambiguity_margin=runtime_settings.memory_semantic_ambiguity_margin,
    )

    @app.middleware("http")
    async def request_logging(
        request: Request, call_next: TypingCallable[[Request], Awaitable[Response]]
    ) -> Response:
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
    def ready(db: Session = Depends(get_db)) -> dict[str, str]:  # noqa: B008
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
    app.include_router(vapi_router)
    return app


app = create_app()


def run() -> None:
    uvicorn.run("reminder_agent.application.main:app", host="0.0.0.0", port=8000, reload=False)
