"""Repair schema drift when migration history survived a database reset.

Revision ID: 0006_repair_missing_schema
Revises: 0005_one_time_reminders

This migration is intentionally additive and conditional. It repairs databases whose
``alembic_version`` row survived while application tables were removed. Existing tables and
data are never dropped or overwritten.
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_repair_missing_schema"
down_revision = "0005_one_time_reminders"
branch_labels = None
depends_on = None


def _tables(bind) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def _indexes(bind, table: str) -> set[str]:
    return {index["name"] for index in sa.inspect(bind).get_indexes(table)}


def _create_indexes(bind) -> None:
    index_definitions = (
        (
            "channel_identities",
            "ix_channel_identities_user_id",
            ["user_id"],
            {},
        ),
        (
            "channel_identities",
            "uq_channel_identities_active_external",
            ["channel_type", "external_id"],
            {
                "unique": True,
                "postgresql_where": sa.text("status = 'LINKED'"),
                "sqlite_where": sa.text("status = 'LINKED'"),
            },
        ),
        (
            "channel_identities",
            "uq_channel_identities_active_phone_user",
            ["user_id"],
            {
                "unique": True,
                "postgresql_where": sa.text("channel_type = 'PHONE' AND status = 'LINKED'"),
                "sqlite_where": sa.text("channel_type = 'PHONE' AND status = 'LINKED'"),
            },
        ),
        ("idempotency_records", "ix_idempotency_user_id", ["user_id"], {}),
        ("reminders", "ix_reminders_user_status", ["user_id", "status"], {}),
        ("reminder_occurrences", "ix_occurrences_due_status", ["next_due_at", "status"], {}),
        ("notifications", "ix_notifications_due_status", ["due_at", "status"], {}),
    )
    for table, name, columns, kwargs in index_definitions:
        if name not in _indexes(bind, table):
            op.create_index(name, table, columns, **kwargs)


def upgrade() -> None:
    bind = op.get_bind()
    tables = _tables(bind)

    if "users" not in tables:
        op.create_table(
            "users",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("name", sa.String(length=150), nullable=False),
            sa.Column("timezone", sa.String(length=150), nullable=True),
            sa.Column("phone_region", sa.String(length=2), nullable=True),
            sa.Column(
                "onboarding_status", sa.String(length=20), server_default="PENDING", nullable=False
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "onboarding_status IN ('PENDING', 'ACTIVE')", name="ck_users_onboarding_status"
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    if "channel_identities" not in tables:
        op.create_table(
            "channel_identities",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("user_id", sa.Uuid(), nullable=False),
            sa.Column("channel_type", sa.String(length=20), nullable=False),
            sa.Column("external_id", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=20), server_default="LINKED", nullable=False),
            sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("telegram_chat_id", sa.BigInteger(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "channel_type IN ('TELEGRAM', 'PHONE')", name="ck_identity_channel_type"
            ),
            sa.CheckConstraint(
                "status IN ('PENDING', 'LINKED', 'UNLINKED')", name="ck_identity_status"
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if "idempotency_records" not in tables:
        op.create_table(
            "idempotency_records",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("source", sa.String(length=20), nullable=False),
            sa.Column("idempotency_key", sa.String(length=255), nullable=False),
            sa.Column("user_id", sa.Uuid(), nullable=True),
            sa.Column("operation", sa.String(length=50), nullable=False),
            sa.Column("request_hash", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=20), server_default="SUCCEEDED", nullable=False),
            sa.Column(
                "delivery_status", sa.String(length=20), server_default="PENDING", nullable=False
            ),
            sa.Column("delivery_attempted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("response", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint("source IN ('TELEGRAM', 'VAPI')", name="ck_idempotency_source"),
            sa.CheckConstraint(
                "status IN ('PROCESSING', 'SUCCEEDED', 'FAILED')", name="ck_idempotency_status"
            ),
            sa.CheckConstraint(
                "delivery_status IN ('PENDING', 'ATTEMPTED', 'SENT', 'UNKNOWN')",
                name="ck_idempotency_delivery_status",
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("source", "idempotency_key", name="uq_idempotency_source_key"),
        )

    if "reminders" not in tables:
        op.create_table(
            "reminders",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("user_id", sa.Uuid(), nullable=False),
            sa.Column("task", sa.Text(), nullable=False),
            sa.Column("schedule_type", sa.String(length=20), nullable=False),
            sa.Column("start_local", sa.String(length=64), nullable=False),
            sa.Column("timezone", sa.String(length=150), nullable=False),
            sa.Column("recurrence_rule", sa.Text(), nullable=True),
            sa.Column("status", sa.String(length=20), server_default="ACTIVE", nullable=False),
            sa.Column("version", sa.Integer(), server_default="1", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "schedule_type IN ('ONE_TIME', 'RECURRING')", name="ck_reminders_schedule_type"
            ),
            sa.CheckConstraint(
                "status IN ('ACTIVE', 'COMPLETED', 'CANCELED')", name="ck_reminders_status"
            ),
            sa.CheckConstraint("version >= 1", name="ck_reminders_version"),
            sa.CheckConstraint(
                "(schedule_type = 'RECURRING' AND recurrence_rule IS NOT NULL) "
                "OR (schedule_type = 'ONE_TIME' AND recurrence_rule IS NULL)",
                name="ck_reminders_recurrence_rule",
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if "reminder_occurrences" not in tables:
        op.create_table(
            "reminder_occurrences",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("reminder_id", sa.Uuid(), nullable=False),
            sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("next_due_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "status IN ('PENDING', 'NOTIFIED', 'COMPLETED', 'CANCELED')",
                name="ck_occurrences_status",
            ),
            sa.ForeignKeyConstraint(["reminder_id"], ["reminders.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "reminder_id", "scheduled_at", name="uq_occurrence_reminder_scheduled"
            ),
        )

    if "notifications" not in tables:
        op.create_table(
            "notifications",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("occurrence_id", sa.Uuid(), nullable=False),
            sa.Column("destination_identity_id", sa.Uuid(), nullable=False),
            sa.Column("sequence", sa.Integer(), server_default="1", nullable=False),
            sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
            sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("provider_message_id", sa.String(length=255), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
            sa.CheckConstraint(
                "status IN ('PENDING', 'SENDING', 'SENT', 'FAILED', 'UNKNOWN', 'CANCELED')",
                name="ck_notifications_status",
            ),
            sa.CheckConstraint("attempt_count >= 0", name="ck_notifications_attempt_count"),
            sa.CheckConstraint("sequence >= 1", name="ck_notifications_sequence"),
            sa.ForeignKeyConstraint(
                ["occurrence_id"], ["reminder_occurrences.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["destination_identity_id"], ["channel_identities.id"], ondelete="RESTRICT"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "occurrence_id", "sequence", name="uq_notification_occurrence_sequence"
            ),
        )

    _create_indexes(bind)


def downgrade() -> None:
    raise RuntimeError("0006_repair_missing_schema is additive and cannot be downgraded safely")
