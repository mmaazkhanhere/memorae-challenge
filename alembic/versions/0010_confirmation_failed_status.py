"""Allow terminal permanent confirmation-delivery failures.

Revision ID: 0010_confirmation_failed_status
Revises: 0009_confirmation_delivery
"""

from alembic import op


revision = "0010_confirmation_failed_status"
down_revision = "0009_confirmation_delivery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_idempotency_delivery_status", "idempotency_records", type_="check"
    )
    op.create_check_constraint(
        "ck_idempotency_delivery_status",
        "idempotency_records",
        "delivery_status IN ('PENDING', 'ATTEMPTED', 'SENT', 'FAILED', 'UNKNOWN')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_idempotency_delivery_status", "idempotency_records", type_="check"
    )
    op.create_check_constraint(
        "ck_idempotency_delivery_status",
        "idempotency_records",
        "delivery_status IN ('PENDING', 'ATTEMPTED', 'SENT', 'UNKNOWN')",
    )
