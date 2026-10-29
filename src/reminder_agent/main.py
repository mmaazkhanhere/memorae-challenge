from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from typing import Awaitable, Callable as TypingCallable

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from reminder_agent.api.telegram import router as telegram_router
from reminder_agent.db.session import get_db
from reminder_agent.observability import configure_logging, log_event, request_id
from reminder_agent.settings import Settings, get_settings
from reminder_agent.telegram import TelegramClient


def create_app(
    settings: Settings | None = None,
    *,
    telegram_client: TelegramClient | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> FastAPI:
    app = FastAPI(title="Memorae")
    app.state.settings = settings or get_settings()
    app.state.telegram_client = telegram_client
    logger = configure_logging(
        app.state.settings.log_level,
        app.state.settings.log_format,
        app.state.settings.log_include_traceback,
    )

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
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc
        return {"status": "ready"}

    app.include_router(telegram_router)
    return app


app = create_app()


def run() -> None:
    uvicorn.run("reminder_agent.main:app", host="0.0.0.0", port=8000, reload=False)
