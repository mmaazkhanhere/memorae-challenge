"""Add pgvector-backed semantic indexing metadata to memory facts."""

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


revision = "0016_memory_embeddings"
down_revision = "0015_memory_facts"
branch_labels = None
depends_on = None

EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("memory embeddings require PostgreSQL with pgvector")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    inspector = sa.inspect(bind)
    existing_columns = {column["name"] for column in inspector.get_columns("memory_facts")}
    columns = (
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=True),
        sa.Column(
            "embedding_status",
            sa.String(length=24),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("embedding_model", sa.String(length=200), nullable=True),
        sa.Column("embedding_content_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "embedding_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("embedding_last_error", sa.String(length=64), nullable=True),
        sa.Column("embedding_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    for column in columns:
        if column.name not in existing_columns:
            op.add_column("memory_facts", column)

    constraint_names = {
        constraint["name"] for constraint in inspector.get_check_constraints("memory_facts")
    }
    if "ck_memory_facts_embedding_status" not in constraint_names:
        op.create_check_constraint(
            "ck_memory_facts_embedding_status",
            "memory_facts",
            "embedding_status IN ('PENDING', 'INDEXED', 'FAILED', 'SKIPPED_SENSITIVE')",
        )
    index_names = {index["name"] for index in inspector.get_indexes("memory_facts")}
    if "ix_memory_facts_user_embedding_status" not in index_names:
        op.create_index(
            "ix_memory_facts_user_embedding_status",
            "memory_facts",
            ["user_id", "embedding_status"],
        )


def downgrade() -> None:
    op.drop_index("ix_memory_facts_user_embedding_status", table_name="memory_facts")
    op.drop_constraint(
        "ck_memory_facts_embedding_status", "memory_facts", type_="check"
    )
    for column in (
        "embedding_updated_at",
        "embedding_last_error",
        "embedding_attempts",
        "embedding_content_hash",
        "embedding_model",
        "embedding_status",
        "embedding",
    ):
        op.drop_column("memory_facts", column)
    # The extension may be shared by other tables; downgrades intentionally do
    # not drop it.
