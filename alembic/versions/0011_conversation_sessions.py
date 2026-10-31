"""Add durable clarification conversation sessions for VS-05."""

from alembic import op
import sqlalchemy as sa


revision = "0011_conversation_sessions"
down_revision = "0010_confirmation_failed_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Migration 0007 uses the current ORM metadata for additive repair. On a
    # fresh database that repair may already have created this new table, so
    # make this migration safe for both fresh and repaired histories.
    inspector = sa.inspect(op.get_bind())
    if "conversation_sessions" not in inspector.get_table_names():
        op.create_table(
            "conversation_sessions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("user_id", sa.Uuid(), nullable=False),
            sa.Column("channel_type", sa.String(length=20), nullable=False),
            sa.Column("external_session_id", sa.String(length=255), nullable=False),
            sa.Column("active_reminder_id", sa.Uuid(), nullable=True),
            sa.Column("pending_clarification", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(length=30), server_default="ACTIVE", nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "channel_type IN ('TELEGRAM', 'VOICE_CALL')",
                name="ck_conversation_sessions_channel_type",
            ),
            sa.CheckConstraint(
                "status IN ('ACTIVE', 'AWAITING_CLARIFICATION', 'ENDED')",
                name="ck_conversation_sessions_status",
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["active_reminder_id"], ["reminders.id"], ondelete="SET NULL"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id",
                "channel_type",
                "external_session_id",
                name="uq_conversation_sessions_external",
            ),
        )
    if "ix_conversation_sessions_user_status" not in {
        index["name"] for index in inspector.get_indexes("conversation_sessions")
    }:
        op.create_index(
            "ix_conversation_sessions_user_status",
            "conversation_sessions",
            ["user_id", "status"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "conversation_sessions" in inspector.get_table_names():
        if "ix_conversation_sessions_user_status" in {
            index["name"] for index in inspector.get_indexes("conversation_sessions")
        }:
            op.drop_index("ix_conversation_sessions_user_status", table_name="conversation_sessions")
        op.drop_table("conversation_sessions")
