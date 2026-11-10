"""Persist server-side authorization for authenticated Vapi calls."""

import sqlalchemy as sa

from alembic import op

revision = "0017_voice_call_authorizations"
down_revision = "0016_memory_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "voice_call_authorizations" in inspector.get_table_names():
        return
    op.create_table(
        "voice_call_authorizations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider_call_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("phone_identity_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('ACTIVE', 'ENDED', 'EXPIRED')",
            name="ck_voice_call_authorizations_status",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["phone_identity_id"], ["channel_identities.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_call_id", name="uq_voice_call_authorizations_call_id"),
    )
    op.create_index(
        "ix_voice_call_authorizations_status_expiry",
        "voice_call_authorizations",
        ["status", "expires_at"],
    )
    op.create_index(
        "ix_voice_call_authorizations_user_id",
        "voice_call_authorizations",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_voice_call_authorizations_user_id",
        table_name="voice_call_authorizations",
    )
    op.drop_index(
        "ix_voice_call_authorizations_status_expiry",
        table_name="voice_call_authorizations",
    )
    op.drop_table("voice_call_authorizations")
