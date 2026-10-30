"""Allow only one active owner per channel identity and phone number.

Revision ID: 0003_active_phone_identity_uniqueness
Revises: 0002_idempotency_delivery
"""

import sqlalchemy as sa

from alembic import op


revision = "0003_phone_link_uniqueness"
down_revision = "0002_idempotency_delivery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_identity_channel_external", "channel_identities", type_="unique")
    op.create_index(
        "uq_channel_identities_active_external",
        "channel_identities",
        ["channel_type", "external_id"],
        unique=True,
        postgresql_where=sa.text("status = 'LINKED'"),
        sqlite_where=sa.text("status = 'LINKED'"),
    )
    op.create_index(
        "uq_channel_identities_active_phone_user",
        "channel_identities",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("channel_type = 'PHONE' AND status = 'LINKED'"),
        sqlite_where=sa.text("channel_type = 'PHONE' AND status = 'LINKED'"),
    )


def downgrade() -> None:
    op.drop_index("uq_channel_identities_active_phone_user", table_name="channel_identities")
    op.drop_index("uq_channel_identities_active_external", table_name="channel_identities")
    op.create_unique_constraint(
        "uq_identity_channel_external",
        "channel_identities",
        ["channel_type", "external_id"],
    )
