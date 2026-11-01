"""Add stable recurrence instance keys for VS-07."""

from alembic import op
import sqlalchemy as sa


revision = "0013_recurring_occurrence_keys"
down_revision = "0012_conversation_session_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("reminder_occurrences")}
    if "recurrence_instance_key" not in columns:
        op.add_column(
            "reminder_occurrences",
            sa.Column("recurrence_instance_key", sa.String(length=64), nullable=True),
        )
    constraints = {
        constraint.get("name") for constraint in inspector.get_unique_constraints("reminder_occurrences")
    }
    if "uq_occurrence_reminder_recurrence_key" not in constraints:
        if op.get_bind().dialect.name == "sqlite":
            with op.batch_alter_table("reminder_occurrences") as batch:
                batch.create_unique_constraint(
                    "uq_occurrence_reminder_recurrence_key",
                    ["reminder_id", "recurrence_instance_key"],
                )
        else:
            op.create_unique_constraint(
                "uq_occurrence_reminder_recurrence_key",
                "reminder_occurrences",
                ["reminder_id", "recurrence_instance_key"],
            )


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("reminder_occurrences") as batch:
            batch.drop_constraint("uq_occurrence_reminder_recurrence_key", type_="unique")
    else:
        op.drop_constraint(
            "uq_occurrence_reminder_recurrence_key", "reminder_occurrences", type_="unique"
        )
    op.drop_column("reminder_occurrences", "recurrence_instance_key")
