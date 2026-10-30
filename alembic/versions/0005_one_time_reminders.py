"""Add VS-03 one-time reminder persistence.

Revision ID: 0005_one_time_reminders
Revises: 0004_user_phone_region
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_one_time_reminders"
down_revision = "0004_user_phone_region"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
    op.create_index("ix_reminders_user_status", "reminders", ["user_id", "status"])

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
        sa.UniqueConstraint("reminder_id", "scheduled_at", name="uq_occurrence_reminder_scheduled"),
    )
    op.create_index("ix_occurrences_due_status", "reminder_occurrences", ["next_due_at", "status"])

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
            ["destination_identity_id"], ["channel_identities.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["occurrence_id"], ["reminder_occurrences.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "occurrence_id", "sequence", name="uq_notification_occurrence_sequence"
        ),
    )
    op.create_index("ix_notifications_due_status", "notifications", ["due_at", "status"])


def downgrade() -> None:
    op.drop_index("ix_notifications_due_status", table_name="notifications")
    op.drop_table("notifications")
    op.drop_index("ix_occurrences_due_status", table_name="reminder_occurrences")
    op.drop_table("reminder_occurrences")
    op.drop_index("ix_reminders_user_status", table_name="reminders")
    op.drop_table("reminders")
