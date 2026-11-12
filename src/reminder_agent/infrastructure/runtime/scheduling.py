"""Durable delivery of scheduled Telegram reminder notifications."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import select

from reminder_agent.channels.telegram.callbacks import callback_data
from reminder_agent.channels.telegram.provider import (
    TelegramAPIError,
    TelegramClient,
)
from reminder_agent.features.reminders.recurrence_service import (
    materialize_active_recurring_reminders,
)
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
)
from reminder_agent.infrastructure.database.session import SessionFactory
from reminder_agent.infrastructure.observability import get_logger, log_event

if TYPE_CHECKING:
    from reminder_agent.infrastructure.runtime.confirmations import ConfirmationWorker

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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class ClaimedNotification:
    notification_id: object
    worker_id: str
    attempt_count: int
    due_at: datetime


class NotificationWorker:
    """Poll, claim, and deliver due Telegram notifications.

    Claims are short-lived database leases. Telegram is called only after the claim
    transaction commits, and the result is finalized conditionally on the same lease.
    An expired in-flight claim is marked UNKNOWN because retrying it could duplicate a
    message that Telegram accepted before the worker crashed.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        telegram_client: TelegramClient,
        *,
        worker_id: str | None = None,
        poll_interval_seconds: float = 5.0,
        claim_lease_seconds: float = 60.0,
        delivery_grace_seconds: float = 0.0,
        max_attempts: int = 3,
        uncertain_max_retries: int = 2,
        retry_base_delay_seconds: float = 30.0,
        recurrence_materialization_horizon_days: int = 35,
        recurrence_materialization_batch_size: int = 100,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
        callback_secret: str = "development",
    ) -> None:
        self.session_factory = session_factory
        self.telegram_client = telegram_client
        self.worker_id = worker_id or f"worker-{uuid4().hex}"
        self.poll_interval_seconds = poll_interval_seconds
        self.claim_lease_seconds = claim_lease_seconds
        self.delivery_grace_seconds = max(delivery_grace_seconds, 0.0)
        self.max_attempts = max_attempts
        self.uncertain_max_retries = min(max(uncertain_max_retries, 0), 2)
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.recurrence_materialization_horizon_days = recurrence_materialization_horizon_days
        self.recurrence_materialization_batch_size = recurrence_materialization_batch_size
        self.clock = clock
        self.sleep = sleep
        self.callback_secret = callback_secret

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
        eligible_before = now - timedelta(seconds=self.delivery_grace_seconds)
        claimed: list[ClaimedNotification] = []
        with self.session_factory.begin() as db:
            rows = db.execute(
                select(Notification.id)
                .join(ReminderOccurrence, ReminderOccurrence.id == Notification.occurrence_id)
                .join(Reminder, Reminder.id == ReminderOccurrence.reminder_id)
                .join(ChannelIdentity, ChannelIdentity.id == Notification.destination_identity_id)
                .where(
                    Notification.status == PENDING,
                    Notification.due_at <= eligible_before,
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

    def _load_deliverable_claim(
        self, notification_id: object, now: datetime
    ) -> tuple[Notification, str, int] | None:
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
            if chat_id is None:
                return None
            return notification, str(task), int(chat_id)

    def _finalize_successful_delivery(
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

    def _finalize_failed_delivery(
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
            } or (error.status_code is not None and 500 <= error.status_code <= 599)
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
                    delay = self.retry_base_delay_seconds * (2**retries_so_far)
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
        loaded = self._load_deliverable_claim(claim.notification_id, now)
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
            lateness_seconds=max(0.0, (now - _as_utc(notification.due_at)).total_seconds()),
        )
        done_markup = {
            "inline_keyboard": [
                [
                    {
                        "text": "Done",
                        "callback_data": callback_data(
                            notification.occurrence_id, self.callback_secret
                        ),
                    }
                ]
            ]
        }
        log_event(
            logger,
            logging.DEBUG,
            "completion_action_attached",
            worker_id=self.worker_id,
            notification_id=str(claim.notification_id),
            occurrence_id=str(notification.occurrence_id),
            action="done",
        )
        try:
            result = self.telegram_client.send_message(
                chat_id,
                f"Reminder: {task}",
                reply_markup=done_markup,
            )
        except TelegramAPIError as exc:
            outcome = self._finalize_failed_delivery(claim.notification_id, now, exc)
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
        finalized = self._finalize_successful_delivery(
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
        materialized = materialize_active_recurring_reminders(
            self.session_factory,
            now=now,
            horizon_days=self.recurrence_materialization_horizon_days,
            batch_size=self.recurrence_materialization_batch_size,
        )
        if materialized:
            log_event(
                logger,
                logging.INFO,
                "recurring_occurrences_materialized",
                worker_id=self.worker_id,
                count=materialized,
            )
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
        """Run notification and response-retry delivery until the process stops."""

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
