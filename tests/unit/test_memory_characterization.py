"""Characterization coverage for owner-scoped memory behavior."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from reminder_agent.features.memory import memory_domain
from reminder_agent.features.memory import service as memories
from reminder_agent.features.memory.retrieve import answering as memory_answering
from reminder_agent.features.memory.retrieve import contract as memory_answer_contract
from reminder_agent.features.memory.retrieve import domain as retrieve_domain
from reminder_agent.features.memory.service import (
    MemoryAnswerProposal,
    MemoryLookupResult,
    MemoryRetrievalPolicy,
    SavedMemory,
    is_memory_content_grounded,
    is_memory_intent_evidence_grounded,
    is_memory_lookup_query_grounded,
    memory_answer_preserves_exact_values,
)
from reminder_agent.shared.memory_contracts import MemoryIntentEvidence


def test_memory_retrieval_policy_preserves_safe_bounds() -> None:
    policy = MemoryRetrievalPolicy(
        lexical_candidate_limit=20,
        semantic_candidate_limit=3,
        semantic_relevance_threshold=0.68,
        semantic_ambiguity_margin=0.04,
    )

    assert policy.semantic_candidate_limit == 3
    with pytest.raises(ValueError, match="lexical candidate limit"):
        MemoryRetrievalPolicy(lexical_candidate_limit=1)
    with pytest.raises(ValueError, match="semantic ambiguity margin"):
        MemoryRetrievalPolicy(semantic_ambiguity_margin=1.1)


def test_memory_grounding_requires_explicit_evidence_and_shared_subject_words() -> None:
    evidence = MemoryIntentEvidence(
        kind="EXPLICIT_SAVE_REQUEST",
        confidence="CLEAR",
        evidence_text="remember",
    )

    assert is_memory_intent_evidence_grounded("Please remember my gate code", evidence)
    assert not is_memory_intent_evidence_grounded(
        "My gate code is 4512", evidence
    )
    assert is_memory_content_grounded("Remember my gate code is 4512", "gate code 4512")
    assert not is_memory_content_grounded("Remember my gate code", "favorite food")


def test_memory_lookup_query_grounding_rejects_identifiers_and_unrequested_subjects() -> None:
    assert is_memory_lookup_query_grounded("Which color do I like?", "color")
    assert not is_memory_lookup_query_grounded("Which color do I like?", "gate code")
    assert not is_memory_lookup_query_grounded(
        "Which color do I like?", "user ID 9d03e6c2-ef28-40d7-80f1-c5720151d129"
    )


def test_memory_result_and_saved_memory_contracts_keep_status_and_identity() -> None:
    fact_id = uuid4()
    saved_at = datetime(2026, 10, 30, tzinfo=UTC)
    saved = SavedMemory(fact_id=fact_id, saved_at=saved_at, embedding_status="PENDING")
    result = MemoryLookupResult(status="AMBIGUOUS", candidate_ids=(fact_id,))

    assert saved.fact_id == fact_id
    assert saved.embedding_status == "PENDING"
    assert result.status == "AMBIGUOUS"
    assert result.candidate_ids == (fact_id,)


def test_memory_answer_contract_preserves_exact_values() -> None:
    answer = MemoryAnswerProposal(
        decision="ANSWER",
        candidate_indices=[0],
        answer="Your parking slot is B17.",
        clarification_question=None,
    )

    assert memory_answer_preserves_exact_values(answer.answer or "", ("my parking slot is B17",))
    assert not memory_answer_preserves_exact_values(
        "Your parking slot is B 17.", ("my parking slot is B17",)
    )


def test_memory_retrieval_exports_preserve_canonical_identity() -> None:
    assert memories.MemoryLookupResult is retrieve_domain.MemoryLookupResult
    assert memories.MemoryRetrievalPolicy is retrieve_domain.MemoryRetrievalPolicy
    assert memories.SavedMemory is memory_domain.SavedMemory
    assert memories.MemoryAnswerProposal is memory_answer_contract.MemoryAnswerProposal
    assert memory_answering.MemoryAnswerProposal is memory_answer_contract.MemoryAnswerProposal
    assert (
        memories.memory_answer_preserves_exact_values
        is memory_answer_contract.memory_answer_preserves_exact_values
    )
