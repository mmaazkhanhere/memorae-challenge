from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from reminder_agent.db.models import Base, MemoryFact
from reminder_agent.memories import (
    MemoryIntentEvidence,
    PendingMemory,
    RetrieveMemoryQuery,
    SaveMemoryFact,
    extract_memory_content,
    extract_memory_lookup_query,
    is_memory_intent_evidence_grounded,
    is_memory_lookup_query_grounded,
    is_memory_save_command,
    lookup_memory_fact,
    save_memory_fact,
)


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
