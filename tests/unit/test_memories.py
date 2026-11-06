from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from reminder_agent.db.models import Base, MemoryFact
from reminder_agent.memories import (
    MemoryIntentEvidence,
    PendingMemory,
    SaveMemoryFact,
    extract_memory_content,
    is_memory_intent_evidence_grounded,
    is_memory_save_command,
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
