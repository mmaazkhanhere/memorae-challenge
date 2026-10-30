"""Store each user's phone parsing region.

Revision ID: 0004_user_phone_region
Revises: 0003_phone_link_uniqueness
"""

import sqlalchemy as sa
from alembic import op


revision = "0004_user_phone_region"
down_revision = "0003_phone_link_uniqueness"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("phone_region", sa.String(length=2), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "phone_region")
