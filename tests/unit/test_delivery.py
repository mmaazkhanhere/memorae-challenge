from datetime import UTC, datetime, timedelta

from reminder_agent.application import SqlAlchemyResponseDelivery
from reminder_agent.infrastructure.database.models import IdempotencyRecord


def test_provider_neutral_delivery_ledger_claims_and_records_outcome(db):
    now = datetime.now(UTC)
    db.add(
        IdempotencyRecord(
            source="VAPI",
            idempotency_key="fake-delivery-1",
            operation="READ_ONLY",
            request_hash="hash",
            status="SUCCEEDED",
            delivery_status="PENDING",
            response={"content": "done"},
            created_at=now,
            updated_at=now,
        )
    )
    db.commit()

    ledger = SqlAlchemyResponseDelivery(db, source="VAPI", claim_owner_prefix="fake")
    assert ledger.claim("fake-delivery-1") is True
    assert ledger.claim("fake-delivery-1") is False

    ledger.record_outcome(
        "fake-delivery-1",
        sent=False,
        error_kind="network",
        next_attempt_at=now + timedelta(minutes=1),
    )

    record = db.query(IdempotencyRecord).one()
    assert record.delivery_status == "UNKNOWN"
    assert record.delivery_last_error_kind == "network"
