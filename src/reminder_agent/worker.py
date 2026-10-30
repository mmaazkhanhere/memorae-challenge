from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.db.models import (
    ChannelIdentity,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
)
from reminder_agent.db.session import build_engine
from reminder_agent.observability import configure_logging, get_logger, log_event
from reminder_agent.settings import get_settings, validate_runtime_settings
from reminder_agent.telegram import HttpTelegramClient, TelegramAPIError, TelegramClient

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
PENDING = "PENDING"
SENDING = "SENDING"
SENT = "SENT"
FAILED = "FAILED"
UNKNOWN = "UNKNOWN"
NOTIFIED = "NOTIFIED"
logger = get_logger()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ClaimedNotification:
    notification_id: object
    worker_id: str
    attempt_count: int
    due_at: datetime


@dataclass(frozen=True)
class ClaimedConfirmation:
    record_id: object
    attempt_count: int
    chat_id: int
    text: str
    reply_markup: dict | None


class ConfirmationWorker:
    """Retry durable Telegram confirmations after an outbound failure."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        telegram_client: TelegramClient,
        *,
        worker_id: str | None = None,
        claim_lease_seconds: float = 120.0,
        max_attempts: int = 3,
        retry_base_delay_seconds: float = 30.0,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.session_factory = session_factory
        self.telegram_client = telegram_client
        self.worker_id = worker_id or f"confirmation-{uuid4().hex}"
        self.claim_lease_seconds = claim_lease_seconds
        self.max_attempts = max_attempts
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.clock = clock

    def recover_expired_claims(self, now: datetime) -> None:
        with self.session_factory.begin() as db:
            records = db.scalars(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.source == TELEGRAM,
                    IdempotencyRecord.delivery_status == "ATTEMPTED",
                    IdempotencyRecord.delivery_claimed_until <= now,
                )
            ).all()
            for record in records:
                record.delivery_status = "UNKNOWN"
                record.delivery_claimed_by = None
                record.delivery_claimed_until = None
                if record.delivery_attempt_count < self.max_attempts:
                    record.delivery_next_attempt_at = now
                log_event(
                    logger,
                    logging.WARNING,
                    "confirmation_claim_recovered",
                    worker_id=self.worker_id,
                    record_id=str(record.id),
                    attempt_count=record.delivery_attempt_count,
                )

    def claim_due(self, now: datetime, *, limit: int = 25) -> list[ClaimedConfirmation]:
        claimed: list[ClaimedConfirmation] = []
        lease_until = now + timedelta(seconds=self.claim_lease_seconds)
        with self.session_factory.begin() as db:
            rows = db.execute(
                select(IdempotencyRecord)
                .where(
                    IdempotencyRecord.source == TELEGRAM,
                    IdempotencyRecord.delivery_status == "UNKNOWN",
                    IdempotencyRecord.delivery_next_attempt_at <= now,
                    IdempotencyRecord.delivery_attempt_count < self.max_attempts,
                    IdempotencyRecord.delivery_claimed_until.is_(None),
                )
                .order_by(IdempotencyRecord.delivery_next_attempt_at, IdempotencyRecord.id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).scalars().all()
            for record in rows:
                response = record.response or {}
                if not response.get("chat_id") or not response.get("text"):
                    record.delivery_last_error_kind = "invalid_stored_response"
                    record.delivery_next_attempt_at = None
                    continue
                record.delivery_status = "ATTEMPTED"
                record.delivery_claimed_by = self.worker_id
                record.delivery_claimed_until = lease_until
                record.delivery_attempted_at = now
                record.delivery_attempt_count += 1
                claimed.append(
                    ClaimedConfirmation(
                        record.id,
                        record.delivery_attempt_count,
                        int(response["chat_id"]),
                        str(response["text"]),
                        response.get("reply_markup"),
                    )
                )
                log_event(
                    logger,
                    logging.DEBUG,
                    "confirmation_claimed",
                    worker_id=self.worker_id,
                    record_id=str(record.id),
                    attempt_count=record.delivery_attempt_count,
                    claim_expires_at=lease_until.isoformat(),
                )
        return claimed

    def _finish(self, claim: ClaimedConfirmation, now: datetime, *, sent: bool, error=None) -> str:
        with self.session_factory.begin() as db:
            record = db.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.id == claim.record_id,
                    IdempotencyRecord.delivery_status == "ATTEMPTED",
                    IdempotencyRecord.delivery_claimed_by == self.worker_id,
                )
            )
            if record is None:
                return "lost_claim"
            record.delivery_claimed_by = None
            record.delivery_claimed_until = None
            if sent:
                record.delivery_status = "SENT"
                record.delivery_next_attempt_at = None
                record.delivery_last_error_kind = None
                outcome = "SENT"
            else:
                permanent = error.status_code in {400, 401, 403, 404}
                safe_retryable = error.failure_kind in {
                    "dns_error",
                    "connect_error",
                    "connect_timeout",
                    "pool_timeout",
                }
                uncertain = error.failure_kind in {
                    "network",
                    "network_timeout",
                    "read_timeout",
                    "write_timeout",
                    "read_error",
                    "write_error",
                } or (
                    error.status_code is not None and 500 <= error.status_code <= 599
                )
                rate_limited = error.status_code == 429
                record.delivery_status = "FAILED" if permanent else "UNKNOWN"
                record.delivery_last_error_kind = error.failure_kind
                can_retry = (
                    (safe_retryable or uncertain or rate_limited)
                    and record.delivery_attempt_count < self.max_attempts
                )
                if can_retry:
                    delay = error.retry_after_seconds or (
                        self.retry_base_delay_seconds
                        * (2 ** max(record.delivery_attempt_count - 1, 0))
                    )
                    record.delivery_next_attempt_at = now + timedelta(seconds=delay)
                    outcome = "RETRY_PENDING"
                    record.delivery_status = "UNKNOWN"
                elif permanent or safe_retryable:
                    record.delivery_next_attempt_at = None
                    outcome = "FAILED"
                else:
                    record.delivery_next_attempt_at = None
                    outcome = "UNKNOWN"
                log_event(
                    logger,
                    logging.INFO,
                    "confirmation_delivery_outcome_recorded",
                    worker_id=self.worker_id,
                    record_id=str(claim.record_id),
                    outcome=outcome,
                    failure_kind=error.failure_kind,
                    retry_after_seconds=error.retry_after_seconds,
                )
            return outcome

    def deliver(self, claim: ClaimedConfirmation, now: datetime) -> str:
        log_event(
            logger,
            logging.DEBUG,
            "confirmation_delivery_attempt_started",
            worker_id=self.worker_id,
            record_id=str(claim.record_id),
            attempt_count=claim.attempt_count,
        )
        try:
            self.telegram_client.send_message(
                claim.chat_id, claim.text, reply_markup=claim.reply_markup
            )
        except TelegramAPIError as exc:
            outcome = self._finish(claim, now, sent=False, error=exc)
            log_event(
                logger,
                logging.WARNING,
                "confirmation_delivery_failed",
                worker_id=self.worker_id,
                record_id=str(claim.record_id),
                outcome=outcome,
                failure_kind=exc.failure_kind,
                network_cause=exc.cause,
            )
            return outcome
        outcome = self._finish(claim, now, sent=True)
        log_event(
            logger,
            logging.INFO,
            "confirmation_delivered",
            worker_id=self.worker_id,
            record_id=str(claim.record_id),
            outcome=outcome,
        )
        return outcome

    def run_once(self) -> int:
        now = _as_utc(self.clock())
        self.recover_expired_claims(now)
        claims = self.claim_due(now)
        for claim in claims:
            self.deliver(claim, now)
        return len(claims)


class NotificationWorker:
    """Poll, claim, and deliver due Telegram notifications.

    Claims are short-lived database leases. Telegram is called only after the claim
    transaction commits, and the result is finalized conditionally on the same lease.
    An expired in-flight claim is marked UNKNOWN because retrying it could duplicate a
    message that Telegram accepted before the worker crashed.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        telegram_client: TelegramClient,
        *,
        worker_id: str | None = None,
        poll_interval_seconds: float = 5.0,
        claim_lease_seconds: float = 60.0,
        max_attempts: int = 3,
        uncertain_max_retries: int = 2,
        retry_base_delay_seconds: float = 30.0,
        clock: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.session_factory = session_factory
        self.telegram_client = telegram_client
        self.worker_id = worker_id or f"worker-{uuid4().hex}"
        self.poll_interval_seconds = poll_interval_seconds
        self.claim_lease_seconds = claim_lease_seconds
        self.max_attempts = max_attempts
        self.uncertain_max_retries = min(max(uncertain_max_retries, 0), 2)
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.clock = clock
        self.sleep = sleep

    def recover_expired_claims(self, now: datetime | None = None) -> int:
        now = _as_utc(now or self.clock())
        recovered = 0
        with self.session_factory.begin() as db:
            claims = db.scalars(
                select(Notification).where(
                    Notification.status == SENDING,
                    Notification.claimed_until.is_not(None),
                    Notification.claimed_until <= now,
                )
            ).all()
            for notification in claims:
                notification.status = UNKNOWN
                notification.last_error_kind = "lease_expired"
                notification.claimed_by = None
                notification.claimed_until = None
                recovered += 1
        if recovered:
            log_event(
                logger,
                logging.WARNING,
                "notification_claims_recovered_as_unknown",
                worker_id=self.worker_id,
                count=recovered,
            )
        return recovered

    def claim_due_notifications(
        self, now: datetime | None = None, *, limit: int = 25
    ) -> list[ClaimedNotification]:
        now = _as_utc(now or self.clock())
        lease_until = now + timedelta(seconds=self.claim_lease_seconds)
        claimed: list[ClaimedNotification] = []
        with self.session_factory.begin() as db:
            rows = db.execute(
                select(Notification.id)
                .join(ReminderOccurrence, ReminderOccurrence.id == Notification.occurrence_id)
                .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
                .join(ChannelIdentity, ChannelIdentity.id == Notification.destination_identity_id)
                .where(
                    Notification.status == PENDING,
                    Notification.due_at <= now,
                    Reminder.status == ACTIVE,
                    ReminderOccurrence.status == PENDING,
                    ChannelIdentity.channel_type == TELEGRAM,
                    ChannelIdentity.status == LINKED,
                    ChannelIdentity.telegram_chat_id.is_not(None),
                    ChannelIdentity.user_id == Reminder.user_id,
                )
                .order_by(Notification.due_at, Notification.id)
                .limit(limit)
                .with_for_update(of=Notification, skip_locked=True)
            ).all()
            for (notification_id,) in rows:
                notification = db.get(Notification, notification_id)
                if notification is None or notification.status != PENDING:
                    continue
                notification.status = SENDING
                notification.claimed_by = self.worker_id
                notification.claimed_until = lease_until
                notification.last_attempt_at = now
                notification.attempt_count += 1
                claimed_notification = ClaimedNotification(
                    notification.id,
                    self.worker_id,
                    notification.attempt_count,
                    _as_utc(notification.due_at),
                )
                claimed.append(claimed_notification)
                log_event(
                    logger,
                    logging.DEBUG,
                    "notification_claimed",
                    worker_id=self.worker_id,
                    notification_id=str(notification.id),
                    attempt_count=notification.attempt_count,
                    due_at=_as_utc(notification.due_at).isoformat(),
                    claim_expires_at=lease_until.isoformat(),
                )
        if claimed:
            log_event(
                logger,
                logging.INFO,
                "notifications_claimed",
                worker_id=self.worker_id,
                count=len(claimed),
            )
        return claimed

    def _load_claim(self, notification_id: object, now: datetime):
        with self.session_factory() as db:
            row = db.execute(
                select(Notification, Reminder.task, ChannelIdentity.telegram_chat_id)
                .join(ReminderOccurrence, ReminderOccurrence.id == Notification.occurrence_id)
                .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
                .join(ChannelIdentity, ChannelIdentity.id == Notification.destination_identity_id)
                .where(
                    Notification.id == notification_id,
                    Notification.status == SENDING,
                    Notification.claimed_by == self.worker_id,
                    Notification.claimed_until > now,
                    Reminder.status == ACTIVE,
                    ReminderOccurrence.status == PENDING,
                    ChannelIdentity.channel_type == TELEGRAM,
                    ChannelIdentity.status == LINKED,
                    ChannelIdentity.telegram_chat_id.is_not(None),
                    ChannelIdentity.user_id == Reminder.user_id,
                )
            ).first()
            if row is None:
                return None
            notification, task, chat_id = row
            return notification, str(task), int(chat_id)

    def _finalize_success(
        self, notification_id: object, now: datetime, provider_message_id: str | None
    ) -> bool:
        with self.session_factory.begin() as db:
            notification = db.scalar(
                select(Notification).where(
                    Notification.id == notification_id,
                    Notification.status == SENDING,
                    Notification.claimed_by == self.worker_id,
                )
            )
            if notification is None:
                return False
            notification.status = SENT
            notification.sent_at = now
            notification.provider_message_id = provider_message_id
            notification.claimed_by = None
            notification.claimed_until = None
            occurrence = db.get(ReminderOccurrence, notification.occurrence_id)
            if occurrence is not None and occurrence.status == PENDING:
                occurrence.status = NOTIFIED
            return True

    def _finalize_error(
        self, notification_id: object, now: datetime, error: TelegramAPIError
    ) -> str:
        with self.session_factory.begin() as db:
            notification = db.scalar(
                select(Notification).where(
                    Notification.id == notification_id,
                    Notification.status == SENDING,
                    Notification.claimed_by == self.worker_id,
                )
            )
            if notification is None:
                return "lost_claim"

            safe_retryable = error.failure_kind in {
                "dns_error",
                "connect_error",
                "connect_timeout",
                "pool_timeout",
            }
            uncertain = error.failure_kind in {
                "network",
                "network_timeout",
                "read_timeout",
                "write_timeout",
                "read_error",
                "write_error",
            } or (
                error.status_code is not None and 500 <= error.status_code <= 599
            )
            rate_limited = error.status_code == 429
            retries_so_far = max(notification.attempt_count - 1, 0)
            can_retry = notification.attempt_count < self.max_attempts
            if uncertain:
                can_retry = can_retry and retries_so_far < self.uncertain_max_retries

            delay: float | None = None
            if (safe_retryable or rate_limited or uncertain) and can_retry:
                if rate_limited and error.retry_after_seconds is not None:
                    delay = float(error.retry_after_seconds)
                else:
                    delay = self.retry_base_delay_seconds * (2 ** retries_so_far)
                notification.due_at = now + timedelta(seconds=delay)
                notification.status = PENDING
                outcome = PENDING
            elif uncertain:
                notification.status = UNKNOWN
                outcome = UNKNOWN
            else:
                notification.status = FAILED
                outcome = FAILED
            notification.last_error_kind = error.failure_kind or f"http_{error.status_code}"
            notification.claimed_by = None
            notification.claimed_until = None
            log_event(
                logger,
                logging.INFO,
                "notification_delivery_outcome_recorded",
                worker_id=self.worker_id,
                notification_id=str(notification_id),
                outcome=outcome,
                failure_kind=error.failure_kind,
                provider_status_code=error.status_code,
                retry_after_seconds=error.retry_after_seconds,
                retry_delay_seconds=delay,
                retries_so_far=retries_so_far,
            )
        return outcome

    def deliver(self, claim: ClaimedNotification, now: datetime | None = None) -> str:
        now = _as_utc(now or self.clock())
        loaded = self._load_claim(claim.notification_id, now)
        if loaded is None:
            log_event(
                logger,
                logging.INFO,
                "notification_claim_ineligible",
                worker_id=self.worker_id,
                notification_id=str(claim.notification_id),
            )
            return "skipped"
        notification, task, chat_id = loaded
        log_event(
            logger,
            logging.INFO,
            "notification_delivery_attempt_started",
            worker_id=self.worker_id,
            notification_id=str(claim.notification_id),
            attempt_count=notification.attempt_count,
            due_at=_as_utc(notification.due_at).isoformat(),
            lateness_seconds=max(
                0.0, (now - _as_utc(notification.due_at)).total_seconds()
            ),
        )
        try:
            result = self.telegram_client.send_message(chat_id, f"Reminder: {task}")
        except TelegramAPIError as exc:
            outcome = self._finalize_error(claim.notification_id, now, exc)
            log_event(
                logger,
                logging.WARNING if outcome in {PENDING, UNKNOWN} else logging.ERROR,
                "notification_delivery_failed",
                worker_id=self.worker_id,
                notification_id=str(claim.notification_id),
                outcome=outcome,
                failure_kind=exc.failure_kind,
                network_cause=exc.cause,
                provider_status_code=exc.status_code,
            )
            return outcome
        provider_message_id = getattr(result, "provider_message_id", None)
        finalized = self._finalize_success(
            claim.notification_id, now, provider_message_id
        )
        outcome = SENT if finalized else "lost_claim"
        log_event(
            logger,
            logging.INFO,
            "notification_delivered",
            worker_id=self.worker_id,
            notification_id=str(claim.notification_id),
            outcome=outcome,
            provider_message_id=provider_message_id,
        )
        return outcome

    def run_once(self) -> int:
        now = _as_utc(self.clock())
        log_event(
            logger,
            logging.DEBUG,
            "notification_worker_iteration_started",
            worker_id=self.worker_id,
            now=now.isoformat(),
        )
        self.recover_expired_claims(now)
        claims = self.claim_due_notifications(now)
        for claim in claims:
            self.deliver(claim, now)
        log_event(
            logger,
            logging.DEBUG,
            "notification_worker_iteration_completed",
            worker_id=self.worker_id,
            claimed_count=len(claims),
        )
        return len(claims)

    def run_forever(self, confirmation_worker: ConfirmationWorker | None = None) -> None:
        while True:
            try:
                self.run_once()
                if confirmation_worker is not None:
                    confirmation_worker.run_once()
            except Exception:
                log_event(
                    logger,
                    logging.ERROR,
                    "notification_worker_iteration_failed",
                    worker_id=self.worker_id,
                    exc_info=True,
                )
            self.sleep(self.poll_interval_seconds)


def run() -> None:
    settings = get_settings()
    configure_logging(
        settings.log_level,
        settings.log_format,
        settings.log_include_traceback,
        settings.log_color,
    )
    validate_runtime_settings(settings)
    engine = build_engine(settings.database_url)
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    worker = NotificationWorker(
        session_factory,
        HttpTelegramClient(settings.telegram_bot_token, settings.telegram_api_timeout_seconds),
        poll_interval_seconds=settings.worker_poll_interval_seconds,
        claim_lease_seconds=settings.worker_claim_lease_seconds,
        max_attempts=settings.notification_retry_max_attempts,
        uncertain_max_retries=settings.notification_uncertain_max_retries,
        retry_base_delay_seconds=settings.notification_retry_base_delay_seconds,
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
