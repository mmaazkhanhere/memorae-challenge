"""Background-worker process composition and backward-compatible imports."""

import logging

from sqlalchemy.orm import sessionmaker

from reminder_agent.channels.telegram.provider import HttpTelegramClient
from reminder_agent.infrastructure.database.session import build_engine
from reminder_agent.infrastructure.runtime.confirmations import (
    ClaimedConfirmation,
    ConfirmationWorker,
)
from reminder_agent.infrastructure.runtime.scheduling import (
    ClaimedNotification,
    NotificationWorker,
)
from reminder_agent.infrastructure.observability import configure_logging, get_logger, log_event
from reminder_agent.infrastructure.config.settings import get_settings, validate_runtime_settings

logger = get_logger()

__all__ = [
    "ClaimedConfirmation",
    "ClaimedNotification",
    "ConfirmationWorker",
    "NotificationWorker",
    "run",
]


def run() -> None:
    """Configure and run the durable Telegram-delivery worker process."""

    settings = get_settings()
    configure_logging(
        settings.log_level,
        settings.log_format,
        settings.log_include_traceback,
        settings.log_color,
    )
    validate_runtime_settings(settings)
    engine = build_engine(settings.database_url)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    worker = NotificationWorker(
        session_factory,
        HttpTelegramClient(settings.telegram_bot_token, settings.telegram_api_timeout_seconds),
        poll_interval_seconds=settings.worker_poll_interval_seconds,
        claim_lease_seconds=settings.worker_claim_lease_seconds,
        delivery_grace_seconds=settings.worker_delivery_grace_seconds,
        max_attempts=settings.notification_retry_max_attempts,
        uncertain_max_retries=settings.notification_uncertain_max_retries,
        retry_base_delay_seconds=settings.notification_retry_base_delay_seconds,
        recurrence_materialization_horizon_days=settings.recurrence_materialization_horizon_days,
        recurrence_materialization_batch_size=settings.recurrence_materialization_batch_size,
        callback_secret=settings.telegram_webhook_secret,
    )
    confirmation_worker = ConfirmationWorker(
        session_factory,
        HttpTelegramClient(settings.telegram_bot_token, settings.telegram_api_timeout_seconds),
        claim_lease_seconds=settings.worker_claim_lease_seconds,
        max_attempts=settings.telegram_confirmation_background_max_attempts,
        retry_base_delay_seconds=(
            settings.telegram_confirmation_background_retry_base_delay_seconds
        ),
    )
    log_event(
        logger,
        logging.INFO,
        "notification_worker_started",
        worker_id=worker.worker_id,
        poll_interval_seconds=settings.worker_poll_interval_seconds,
        claim_lease_seconds=settings.worker_claim_lease_seconds,
        max_attempts=settings.notification_retry_max_attempts,
        uncertain_max_retries=settings.notification_uncertain_max_retries,
    )
    try:
        worker.run_forever(confirmation_worker)
    finally:
        log_event(logger, logging.INFO, "notification_worker_stopped", worker_id=worker.worker_id)
        engine.dispose()
