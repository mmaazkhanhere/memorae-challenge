"""Add delivery-worker claim and attempt metadata.

Revision ID: 0008_notification_claims
Revises: 0007_repair_schema_drift
"""

from alembic import op
import sqlalchemy as sa


revision = "0008_notification_claims"
down_revision = "0007_repair_schema_drift"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("notifications")}
    additions = (
        ("claimed_by", sa.String(length=128)),
        ("claimed_until", sa.DateTime(timezone=True)),
        ("last_attempt_at", sa.DateTime(timezone=True)),
        ("last_error_kind", sa.String(length=80)),
    )
    for name, column_type in additions:
        if name not in columns:
            op.add_column("notifications", sa.Column(name, column_type, nullable=True))
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("notifications")}
    if "ix_notifications_claimable" not in indexes:
        op.create_index(
            "ix_notifications_claimable",
            "notifications",
            ["status", "due_at", "claimed_until"],
        )


def downgrade() -> None:
    op.drop_index("ix_notifications_claimable", table_name="notifications")
    op.drop_column("notifications", "last_error_kind")
    op.drop_column("notifications", "last_attempt_at")
    op.drop_column("notifications", "claimed_until")
    op.drop_column("notifications", "claimed_by")
