"""Create durable user-owned memory facts."""

from alembic import op
import sqlalchemy as sa


revision = "0015_memory_facts"
down_revision = "0014_recurrence_exceptions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "memory_facts" in inspector.get_table_names():
        return
    op.create_table(
        "memory_facts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(trim(content)) > 0",
            name="ck_memory_facts_content_nonempty",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_memory_facts_user_created",
        "memory_facts",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_facts_user_created", table_name="memory_facts")
    op.drop_table("memory_facts")
