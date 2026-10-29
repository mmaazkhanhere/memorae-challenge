"""Track one outbound response attempt per Telegram update.

Revision ID: 0002_idempotency_delivery
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_idempotency_delivery"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "idempotency_records",
        sa.Column("delivery_status", sa.String(length=20), server_default="PENDING", nullable=False),
    )
    op.add_column(
        "idempotency_records",
        sa.Column("delivery_attempted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_idempotency_delivery_status",
        "idempotency_records",
        "delivery_status IN ('PENDING', 'ATTEMPTED', 'SENT', 'UNKNOWN')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_idempotency_delivery_status", "idempotency_records", type_="check")
    op.drop_column("idempotency_records", "delivery_attempted_at")
    op.drop_column("idempotency_records", "delivery_status")
