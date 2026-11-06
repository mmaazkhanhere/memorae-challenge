from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from reminder_agent.db.models import Base, MemoryFact
from reminder_agent.embeddings import EmbeddingProviderError
from reminder_agent.memories import (
    MemoryRetrievalPolicy,
    MemoryIntentEvidence,
    PendingMemory,
    RetrieveMemoryQuery,
    SaveMemoryFact,
    backfill_memory_embeddings,
    extract_memory_content,
    extract_memory_lookup_query,
    is_memory_intent_evidence_grounded,
    is_memory_lookup_query_grounded,
    is_memory_save_command,
    lookup_memory_fact,
    save_memory_fact,
    update_memory_fact_content,
)


def _vector(*coordinates: float) -> list[float]:
    return [*coordinates, *([0.0] * (1536 - len(coordinates)))]


class FakeEmbeddingProvider:
    model = "fake-semantic-v1"
    dimensions = 1536
    is_external = False

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        for phrase, vector in self.vectors.items():
            if phrase.casefold() in text.casefold():
                return vector
        return _vector(0, 0, 0, 1)


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        raise EmbeddingProviderError()


def test_explicit_memory_command_is_detected_and_extracted() -> None:
    cases = {
        "Remember my gate code is 4512": "my gate code is 4512",
        "Make a note that my laptop model is ThinkPad T14": (
            "my laptop model is ThinkPad T14"
        ),
        "Take note of my backup email": "my backup email",
    }
    for text, expected_content in cases.items():
        assert is_memory_save_command(text) is True
        assert extract_memory_content(text) == expected_content


@pytest.mark.parametrize(
    "text",
    [
        "I'd like you to remember that my dog's name is Bruno",
        "I’d like you to remember that my dog's name is Bruno",
    ],
)
def test_contracted_intentional_memory_request_is_authorized(text: str) -> None:
    assert is_memory_intent_evidence_grounded(
        text,
        MemoryIntentEvidence(
            kind="EXPLICIT_SAVE_REQUEST",
            confidence="CLEAR",
            evidence_text="remember",
        ),
    ) is True


@pytest.mark.parametrize(
    "kind",
    ["ORDINARY_STATEMENT", "QUOTED_INSTRUCTION", "NEGATED_SAVE_REQUEST", "REMINDER_REQUEST"],
)
def test_non_save_memory_evidence_is_never_authorized(kind: str) -> None:
    evidence = MemoryIntentEvidence(
        kind=kind,
        confidence="CLEAR",
        evidence_text="remember this",
    )
    assert is_memory_intent_evidence_grounded("Please remember this", evidence) is False


@pytest.mark.parametrize(
    "text",
    [
        "Remember",
        "Remember this",
        "save that",
        "Make a note",
        "Make a note of this",
        "Take note that",
    ],
)
def test_incomplete_memory_command_has_no_content(text: str) -> None:
    assert is_memory_save_command(text) is True
    assert extract_memory_content(text) is None


def test_memory_extractor_rejects_reminder_shaped_text() -> None:
    assert is_memory_save_command("Remember to call Ali") is False
    assert extract_memory_content("Remember to call Ali") is None


def test_memory_content_validation_rejects_blank() -> None:
    with pytest.raises(ValueError):
        SaveMemoryFact(content="   ")


def test_pending_memory_is_typed_and_serializable() -> None:
    pending = PendingMemory()
    assert pending.model_dump(mode="json") == {"operation": "SAVE_MEMORY", "content": None}


def test_save_memory_fact_persists_only_to_memory_table() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    user_id = uuid4()
    with Session(engine) as db:
        saved = save_memory_fact(
            db,
            user_id=user_id,
            content="Wi-Fi password is Andorra2026",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        db.commit()
        fact = db.scalar(select(MemoryFact).where(MemoryFact.id == saved.fact_id))
        assert fact is not None
        assert fact.user_id == user_id
        assert fact.content == "Wi-Fi password is Andorra2026"
        assert fact.embedding_status == "PENDING"
    engine.dispose()


def test_memory_lookup_returns_exact_owned_fact_without_mutation() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    owner_id = uuid4()
    other_id = uuid4()
    with Session(engine) as db:
        saved = save_memory_fact(
            db,
            user_id=owner_id,
            content="Wi-Fi password is Andorra2026",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        save_memory_fact(
            db,
            user_id=other_id,
            content="Wi-Fi password is NotYours",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        db.commit()
        before = db.get(MemoryFact, saved.fact_id)
        assert before is not None
        result = lookup_memory_fact(db, user_id=owner_id, query="wifi password")
        assert result.status == "FOUND"
        assert result.fact is not None
        assert result.fact.id == saved.fact_id
        assert result.fact.content == "Wi-Fi password is Andorra2026"
        db.expire_all()
        after = db.get(MemoryFact, saved.fact_id)
        assert after is not None
        assert after.updated_at == before.updated_at
    engine.dispose()


def test_memory_lookup_is_unknown_or_ambiguous_without_guessing() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    user_id = uuid4()
    with Session(engine) as db:
        first = save_memory_fact(
            db,
            user_id=user_id,
            content="Gate code is 4512",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        second = save_memory_fact(
            db,
            user_id=user_id,
            content="Backup gate code is 9921",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        db.commit()
        ambiguous = lookup_memory_fact(db, user_id=user_id, query="gate code")
        assert ambiguous.status == "AMBIGUOUS"
        assert set(ambiguous.candidate_ids) == {first.fact_id, second.fact_id}
        unknown = lookup_memory_fact(db, user_id=user_id, query="passport number")
        assert unknown.status == "UNKNOWN"
    engine.dispose()


def test_memory_lookup_does_not_rank_away_a_plausible_conflicting_fact() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    user_id = uuid4()
    with Session(engine) as db:
        first = save_memory_fact(
            db,
            user_id=user_id,
            content="Gate code is 4512",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        second = save_memory_fact(
            db,
            user_id=user_id,
            content="Code for the gate is 9921",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        db.commit()

        # The first fact contains the exact phrase, but both describe the
        # requested subject. Choosing it would silently return an arbitrary
        # conflicting value.
        result = lookup_memory_fact(db, user_id=user_id, query="gate code")
        assert result.status == "AMBIGUOUS"
        assert set(result.candidate_ids) == {first.fact_id, second.fact_id}
    engine.dispose()


def test_model_lookup_query_is_subject_only_and_grounded() -> None:
    query = RetrieveMemoryQuery(query="color")
    assert is_memory_lookup_query_grounded("Which color do I like?", query.query) is True
    assert is_memory_lookup_query_grounded("Which color do I like?", "gate code") is False
    with pytest.raises(ValueError):
        RetrieveMemoryQuery(query="user ID 9d03e6c2-ef28-40d7-80f1-c5720151d129")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What's my gate code?", "gate code"),
        ("what is my Wi-Fi password", "Wi-Fi password"),
        ("Do you remember my backup email?", "backup email"),
    ],
)
def test_memory_lookup_query_is_high_precision(text: str, expected: str) -> None:
    assert extract_memory_lookup_query(text) == expected


@pytest.mark.parametrize("text", ["What do I have tomorrow?", "Remember my gate code is 4512"])
def test_non_lookup_text_is_not_parsed_as_memory_lookup(text: str) -> None:
    assert extract_memory_lookup_query(text) is None


def test_embedding_failure_keeps_authoritative_fact_pending_for_retry() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    provider = FailingEmbeddingProvider()
    with Session(engine) as db:
        saved = save_memory_fact(
            db,
            user_id=uuid4(),
            content="my favorite food is biryani",
            embedding_provider=provider,
        )
        db.commit()
        fact = db.get(MemoryFact, saved.fact_id)
        assert fact is not None and fact.content == "my favorite food is biryani"
        assert saved.embedding_status == "FAILED"
        assert fact.embedding is None
        assert fact.embedding_attempts == 1
        assert fact.embedding_last_error == "PROVIDER_UNAVAILABLE"
    engine.dispose()


def test_sensitive_memory_is_not_sent_to_external_provider_by_default() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    provider = FakeEmbeddingProvider()
    provider.is_external = True
    with Session(engine) as db:
        saved = save_memory_fact(
            db,
            user_id=uuid4(),
            content="my gate code is 4512",
            embedding_provider=provider,
        )
        db.commit()
        fact = db.get(MemoryFact, saved.fact_id)
        assert fact is not None and fact.embedding_status == "SKIPPED_SENSITIVE"
        assert fact.embedding is None
        assert provider.calls == []
    engine.dispose()


def test_backfill_and_content_regeneration_are_resumable_and_idempotent() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    provider = FakeEmbeddingProvider(
        {"biryani": _vector(1, 0), "nihari": _vector(0, 1)}
    )
    owner_id = uuid4()
    with Session(engine) as db:
        saved = save_memory_fact(
            db,
            user_id=owner_id,
            content="my favorite food is biryani",
        )
        db.commit()
        assert backfill_memory_embeddings(db, embedding_provider=provider) == 1
        db.commit()
        assert backfill_memory_embeddings(db, embedding_provider=provider) == 0
        db.commit()
        assert provider.calls == ["my favorite food is biryani"]

        fact = update_memory_fact_content(
            db,
            user_id=owner_id,
            fact_id=saved.fact_id,
            content="my favorite food is biryani",
            embedding_provider=provider,
        )
        assert fact is not None
        assert provider.calls == ["my favorite food is biryani"]
        fact = update_memory_fact_content(
            db,
            user_id=owner_id,
            fact_id=saved.fact_id,
            content="my favorite food is nihari",
            embedding_provider=provider,
        )
        db.commit()
        assert fact is not None and fact.content == "my favorite food is nihari"
        assert fact.embedding_status == "INDEXED"
        assert provider.calls[-1] == "my favorite food is nihari"
    engine.dispose()


def test_semantic_lookup_is_owner_scoped_and_threshold_calibrated() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    provider = FakeEmbeddingProvider(
        {
            "parking slot": _vector(1, 0),
            "where do i park": _vector(1, 0),
            "unrelated": _vector(0, 1),
        }
    )
    owner_id = uuid4()
    other_id = uuid4()
    with Session(engine) as db:
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
        save_memory_fact(
            db,
            user_id=owner_id,
            content="an unrelated fact",
            embedding_provider=provider,
        )
        db.commit()

        result = lookup_memory_fact(
            db,
            user_id=owner_id,
            query="where do I park",
            embedding_provider=provider,
            retrieval_policy=MemoryRetrievalPolicy(
                semantic_relevance_threshold=0.80,
                semantic_ambiguity_margin=0.04,
            ),
        )
        assert result.status == "FOUND"
        assert result.fact is not None and result.fact.id == owned.fact_id

        unknown = lookup_memory_fact(
            db,
            user_id=owner_id,
            query="unknown subject",
            embedding_provider=provider,
            retrieval_policy=MemoryRetrievalPolicy(
                semantic_relevance_threshold=0.80,
                semantic_ambiguity_margin=0.04,
            ),
        )
        assert unknown.status == "UNKNOWN"
    engine.dispose()


def test_close_semantic_conflicts_trigger_typed_ambiguity() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    provider = FakeEmbeddingProvider(
        {
            "office is in lahore": _vector(1, 0),
            "workplace is karachi": _vector(0.999, 0.02),
            "office located": _vector(1, 0),
        }
    )
    owner_id = uuid4()
    with Session(engine) as db:
        first = save_memory_fact(
            db,
            user_id=owner_id,
            content="my office is in Lahore",
            embedding_provider=provider,
        )
        second = save_memory_fact(
            db,
            user_id=owner_id,
            content="my workplace is Karachi",
            embedding_provider=provider,
        )
        db.commit()
        result = lookup_memory_fact(
            db,
            user_id=owner_id,
            query="office located",
            embedding_provider=provider,
            retrieval_policy=MemoryRetrievalPolicy(
                semantic_relevance_threshold=0.80,
                semantic_ambiguity_margin=0.04,
            ),
        )
        assert result.status == "AMBIGUOUS"
        assert set(result.candidate_ids) == {first.fact_id, second.fact_id}
    engine.dispose()
