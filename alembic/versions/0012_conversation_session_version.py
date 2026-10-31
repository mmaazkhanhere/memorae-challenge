"""Add optimistic versions to clarification sessions."""

from alembic import op
import sqlalchemy as sa


revision = "0012_conversation_session_version"
down_revision = "0011_conversation_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("conversation_sessions")}
    if "version" not in columns:
        op.add_column(
            "conversation_sessions",
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("conversation_sessions")}
    if "version" in columns:
        op.drop_column("conversation_sessions", "version")
