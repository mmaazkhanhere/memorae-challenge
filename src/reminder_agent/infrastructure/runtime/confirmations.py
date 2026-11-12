"""Durable retry delivery for committed Telegram command responses."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select

from reminder_agent.channels.telegram.provider import TelegramAPIError, TelegramClient
from reminder_agent.infrastructure.database.models import IdempotencyRecord
from reminder_agent.infrastructure.database.session import SessionFactory
from reminder_agent.infrastructure.observability import get_logger, log_event

TELEGRAM = "TELEGRAM"
logger = get_logger()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class ClaimedConfirmation:
    record_id: UUID
    attempt_count: int
    chat_id: int
    text: str
    reply_markup: dict | None


class ConfirmationWorker:
    """Retry durable Telegram confirmations after an outbound failure."""

    def __init__(
        self,
        session_factory: SessionFactory,
        telegram_client: TelegramClient,
        *,
        worker_id: str | None = None,
        claim_lease_seconds: float = 120.0,
        max_attempts: int = 3,
        retry_base_delay_seconds: float = 30.0,
        clock: Callable[[], datetime] = _utc_now,
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
            records = (
                db.execute(
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
                )
                .scalars()
                .all()
            )
            for record in records:
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

    def _finalize_delivery(
        self,
        claim: ClaimedConfirmation,
        now: datetime,
        *,
        sent: bool,
        error: TelegramAPIError | None = None,
    ) -> str:
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
                if error is None:
                    raise RuntimeError("failed delivery requires a Telegram error")
                permanent = error.status_code in {400, 401, 403, 404}
                safe_retryable = error.failure_kind in {
                    "dns_error", "connect_error", "connect_timeout", "pool_timeout"
                }
                uncertain = error.failure_kind in {
                    "network", "network_timeout", "read_timeout", "write_timeout",
                    "read_error", "write_error",
                } or (error.status_code is not None and 500 <= error.status_code <= 599)
                rate_limited = error.status_code == 429
                record.delivery_status = "FAILED" if permanent else "UNKNOWN"
                record.delivery_last_error_kind = error.failure_kind
                can_retry = (
                    safe_retryable or uncertain or rate_limited
                ) and record.delivery_attempt_count < self.max_attempts
                if can_retry:
                    delay = error.retry_after_seconds or (
                        self.retry_base_delay_seconds * (2 ** max(record.delivery_attempt_count - 1, 0))
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
                    logger, logging.INFO, "confirmation_delivery_outcome_recorded",
                    worker_id=self.worker_id, record_id=str(claim.record_id), outcome=outcome,
                    failure_kind=error.failure_kind, retry_after_seconds=error.retry_after_seconds,
                )
            return outcome

    def deliver(self, claim: ClaimedConfirmation, now: datetime) -> str:
        log_event(
            logger, logging.DEBUG, "confirmation_delivery_attempt_started",
            worker_id=self.worker_id, record_id=str(claim.record_id), attempt_count=claim.attempt_count,
        )
        try:
            self.telegram_client.send_message(
                claim.chat_id, claim.text, reply_markup=claim.reply_markup
            )
        except TelegramAPIError as exc:
            outcome = self._finalize_delivery(claim, now, sent=False, error=exc)
            log_event(
                logger, logging.WARNING, "confirmation_delivery_failed",
                worker_id=self.worker_id, record_id=str(claim.record_id), outcome=outcome,
                failure_kind=exc.failure_kind, network_cause=exc.cause,
            )
            return outcome
        outcome = self._finalize_delivery(claim, now, sent=True)
        log_event(
            logger, logging.INFO, "confirmation_delivered",
            worker_id=self.worker_id, record_id=str(claim.record_id), outcome=outcome,
        )
        return outcome

    def run_once(self) -> int:
        now = _as_utc(self.clock())
        self.recover_expired_claims(now)
        claims = self.claim_due(now)
        for claim in claims:
            self.deliver(claim, now)
        return len(claims)
