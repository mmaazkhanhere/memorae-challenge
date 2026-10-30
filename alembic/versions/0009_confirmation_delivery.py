"""Add durable confirmation-delivery retry metadata.

Revision ID: 0009_confirmation_delivery
Revises: 0008_notification_claims
"""

from alembic import op
import sqlalchemy as sa


revision = "0009_confirmation_delivery"
down_revision = "0008_notification_claims"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("idempotency_records")}
    additions = (
        (
            "delivery_attempt_count",
            sa.Integer(),
            {"server_default": "0", "nullable": False},
        ),
        ("delivery_next_attempt_at", sa.DateTime(timezone=True), {"nullable": True}),
        ("delivery_claimed_by", sa.String(length=128), {"nullable": True}),
        ("delivery_claimed_until", sa.DateTime(timezone=True), {"nullable": True}),
        ("delivery_last_error_kind", sa.String(length=80), {"nullable": True}),
    )
    for name, column_type, kwargs in additions:
        if name not in columns:
            op.add_column("idempotency_records", sa.Column(name, column_type, **kwargs))
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("idempotency_records")}
    if "ix_idempotency_delivery_retry" not in indexes:
        op.create_index(
            "ix_idempotency_delivery_retry",
            "idempotency_records",
            ["source", "delivery_status", "delivery_next_attempt_at"],
        )
    op.execute(
        "UPDATE idempotency_records "
        "SET delivery_next_attempt_at = CURRENT_TIMESTAMP "
        "WHERE source = 'TELEGRAM' AND delivery_status = 'UNKNOWN' "
        "AND delivery_next_attempt_at IS NULL"
    )
    op.execute(
        "UPDATE idempotency_records "
        "SET delivery_status = 'UNKNOWN', delivery_next_attempt_at = CURRENT_TIMESTAMP "
        "WHERE source = 'TELEGRAM' AND delivery_status = 'ATTEMPTED' "
        "AND delivery_claimed_until IS NULL"
    )


def downgrade() -> None:
    op.drop_index("ix_idempotency_delivery_retry", table_name="idempotency_records")
    op.drop_column("idempotency_records", "delivery_last_error_kind")
    op.drop_column("idempotency_records", "delivery_claimed_until")
    op.drop_column("idempotency_records", "delivery_claimed_by")
    op.drop_column("idempotency_records", "delivery_next_attempt_at")
    op.drop_column("idempotency_records", "delivery_attempt_count")
