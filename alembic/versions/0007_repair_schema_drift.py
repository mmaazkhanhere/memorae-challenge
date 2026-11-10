"""Repair databases stamped at head without their domain tables.

Revision ID: 0007_repair_schema_drift
Revises: 0006_repair_missing_schema

This migration is intentionally additive and idempotent.  It is a safety net for
deployments where the Alembic version row survived a database reset while the
application tables did not.
"""

from alembic import op

from reminder_agent.infrastructure.database.models import Base


revision = "0007_repair_schema_drift"
down_revision = "0006_repair_missing_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The model metadata is the source of truth for this additive repair.  With
    # checkfirst=True, existing tables and data are left untouched.
    # The metadata now includes the VS-16 pgvector column, so the PostgreSQL
    # type must exist before create_all runs on a fresh database. Migration
    # 0016 remains the authoritative extension/schema migration.
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    raise RuntimeError("0007_repair_schema_drift is additive and cannot be downgraded safely")
