import os
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from reminder_agent.features.memory.persistence import save_memory_fact
from reminder_agent.features.memory.retrieve.persistence import lookup_memory_fact
from reminder_agent.infrastructure.db.models import (
    MemoryFact,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TEST_DATABASE_URL", "sqlite://").startswith("sqlite"),
    reason="requires the dedicated PostgreSQL pgvector test database",
)


def _vector(axis: int) -> list[float]:
    values = [0.0] * 1536
    values[axis] = 1.0
    return values


class FakePgvectorEmbeddings:
    model = "fake-pgvector-v1"
    dimensions = 1536
    is_external = False

    def embed(self, text: str) -> list[float]:
        normalized = text.casefold()
        if "park" in normalized or "leave my car" in normalized or "b17" in normalized:
            return _vector(0)
        return _vector(1)


def test_pgvector_owner_scoped_semantic_lookup_survives_new_session(
    session_factory,
) -> None:
    provider = FakePgvectorEmbeddings()
    owner_id = uuid4()
    other_id = uuid4()
    with session_factory() as db, db.begin():
        db.add_all(
            [
                User(
                    id=owner_id,
                    name="Owner",
                    timezone="Asia/Karachi",
                    onboarding_status="ACTIVE",
                ),
                User(
                    id=other_id,
                    name="Other",
                    timezone="Asia/Karachi",
                    onboarding_status="ACTIVE",
                ),
            ]
        )
        db.flush()
        owned = save_memory_fact(
            db,
            user_id=owner_id,
            content="my parking slot is B17",
            embedding_provider=provider,
        )
        save_memory_fact(
            db,
            user_id=other_id,
            content="my parking slot is A99",
            embedding_provider=provider,
        )

    # A new Session represents an application restart/new process and proves
    # that retrieval uses the persisted pgvector column, not process memory.
    with session_factory() as db:
        before_updated_at = db.scalar(
            select(MemoryFact.updated_at).where(MemoryFact.id == owned.fact_id)
        )
        result = lookup_memory_fact(
            db,
            user_id=owner_id,
            query="where do I park",
            embedding_provider=provider,
        )
        assert result.status == "CANDIDATES"
        assert result.candidates[0].id == owned.fact_id
        assert result.candidates[0].content == "my parking slot is B17"
        assert (
            db.scalar(
                select(func.count())
                .select_from(MemoryFact)
                .where(MemoryFact.user_id.in_((owner_id, other_id)))
            )
            == 2
        )
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0
        db.expire_all()
        assert (
            db.scalar(select(MemoryFact.updated_at).where(MemoryFact.id == owned.fact_id))
            == before_updated_at
        )


def test_pgvector_old_fact_with_null_embedding_uses_lexical_fallback_only(
    session_factory,
) -> None:
    provider = FakePgvectorEmbeddings()
    owner_id = uuid4()
    with session_factory() as db, db.begin():
        db.add(
            User(
                id=owner_id,
                name="Owner",
                timezone="Asia/Karachi",
                onboarding_status="ACTIVE",
            )
        )
        db.flush()
        saved = save_memory_fact(
            db,
            user_id=owner_id,
            content="my favorite food is biryani",
            embedding_provider=None,
        )
        fact = db.get(MemoryFact, saved.fact_id)
        assert fact is not None
        assert fact.embedding is None
        assert fact.embedding_status == "PENDING"

    with session_factory() as db:
        lexical = lookup_memory_fact(
            db,
            user_id=owner_id,
            query="favorite food",
            embedding_provider=provider,
        )
        assert lexical.status == "FOUND"
        assert lexical.fact is not None and lexical.fact.content.endswith("biryani")

        semantic_without_index = lookup_memory_fact(
            db,
            user_id=owner_id,
            query="what do I eat",
            embedding_provider=provider,
        )
        assert semantic_without_index.status == "UNKNOWN"


def test_pgvector_exact_production_parking_query_uses_full_semantic_request(
    session_factory,
) -> None:
    provider = FakePgvectorEmbeddings()
    owner_id = uuid4()
    with session_factory() as db, db.begin():
        db.add(
            User(
                id=owner_id,
                name="Owner",
                timezone="Asia/Karachi",
                onboarding_status="ACTIVE",
            )
        )
        db.flush()
        saved = save_memory_fact(
            db,
            user_id=owner_id,
            content="my parking slot is B17",
            embedding_provider=provider,
        )

    with session_factory() as db:
        result = lookup_memory_fact(
            db,
            user_id=owner_id,
            query="leave my car",
            semantic_query="Where do I leave my car?",
            embedding_provider=provider,
        )
        assert result.status == "CANDIDATES"
        assert result.candidates[0].id == saved.fact_id
