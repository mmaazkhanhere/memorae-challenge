"""Add durable skips for future recurring reminder instances."""

from alembic import op
import sqlalchemy as sa


revision = "0014_recurrence_exceptions"
down_revision = "0013_recurring_occurrence_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "reminder_recurrence_exceptions" not in inspector.get_table_names():
        op.create_table(
            "reminder_recurrence_exceptions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("reminder_id", sa.Uuid(), nullable=False),
            sa.Column("occurrence_date", sa.Date(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["reminder_id"], ["reminders.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "reminder_id", "occurrence_date", name="uq_recurrence_exception_reminder_date"
            ),
        )
        op.create_index(
            "ix_recurrence_exceptions_reminder_date",
            "reminder_recurrence_exceptions",
            ["reminder_id", "occurrence_date"],
        )


def downgrade() -> None:
    op.drop_table("reminder_recurrence_exceptions")
