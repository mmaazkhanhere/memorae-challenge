"""Add optimistic versions to clarification sessions."""

from alembic import op
import sqlalchemy as sa


revision = "0012_conversation_session_version"
down_revision = "0011_conversation_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The initial migration created alembic_version.version_num as VARCHAR(32),
    # but this revision ID is longer. Widen it before Alembic writes the new
    # revision, otherwise a clean PostgreSQL upgrade fails before VS-07's
    # migration can run.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        version_columns = {
            column["name"]: column for column in sa.inspect(bind).get_columns("alembic_version")
        }
        version_column = version_columns.get("version_num")
        if version_column is not None and (
            getattr(version_column["type"], "length", None) or 0
        ) < 64:
            op.alter_column(
                "alembic_version",
                "version_num",
                existing_type=sa.String(length=32),
                type_=sa.String(length=64),
                existing_nullable=False,
            )
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
