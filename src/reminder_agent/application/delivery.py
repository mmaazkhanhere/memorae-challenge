"""Provider-neutral durable response delivery coordination."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.infrastructure.db.models import IdempotencyRecord


class ResponseDeliveryPort(Protocol):
    """Atomic claim and outcome operations for a durable response."""

    def claim(self, idempotency_key: str) -> bool: ...

    def record_outcome(
        self,
        idempotency_key: str,
        *,
        sent: bool,
        error_kind: str | None = None,
        status_code: int | None = None,
        next_attempt_at: datetime | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class SqlAlchemyResponseDelivery:
    """Database-backed delivery ledger independent of a provider payload."""

    db: Session
    source: str
    claim_owner_prefix: str = "webhook"
    lease_seconds: int = 120

    def claim(self, idempotency_key: str) -> bool:
        with self.db.begin():
            record = self.db.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.source == self.source,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            )
            if record is None or record.delivery_status != "PENDING":
                return False
            now = datetime.now(UTC)
            record.delivery_status = "ATTEMPTED"
            record.delivery_attempted_at = now
            record.delivery_attempt_count += 1
            record.delivery_claimed_by = f"{self.claim_owner_prefix}:{idempotency_key}"
            record.delivery_claimed_until = now + timedelta(seconds=self.lease_seconds)
            return True

    def record_outcome(
        self,
        idempotency_key: str,
        *,
        sent: bool,
        error_kind: str | None = None,
        status_code: int | None = None,
        next_attempt_at: datetime | None = None,
    ) -> None:
        with self.db.begin():
            record = self.db.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.source == self.source,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            )
            if record is None:
                return
            permanent_failure = status_code in {400, 401, 403, 404}
            record.delivery_status = (
                "SENT" if sent else ("FAILED" if permanent_failure else "UNKNOWN")
            )
            record.delivery_claimed_by = None
            record.delivery_claimed_until = None
            record.delivery_last_error_kind = None if sent else error_kind
            record.delivery_next_attempt_at = (
                None if sent or permanent_failure else next_attempt_at
            )
