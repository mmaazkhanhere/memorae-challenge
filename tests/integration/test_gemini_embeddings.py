import math
import os

import httpx
import pytest

from reminder_agent.features.intent import OpenAICompatibleMemoryAnswerer
from reminder_agent.features.memory.retrieve.service import lookup_memory_fact
from reminder_agent.features.memory.service import save_memory_fact
from reminder_agent.infrastructure.db.models import User
from reminder_agent.infrastructure.embeddings import (
    EmbeddingProviderError,
    GeminiEmbeddingProvider,
)
from reminder_agent.infrastructure.settings import Settings


def _configured_gemini_key() -> str:
    return (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or (Settings().embedding_api_key or "")
    ).strip()


def _configured_llm_key() -> str:
    return (
        os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or (Settings().llm_api_key or "")
    ).strip()


class _FakeResponse:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("fake failure", request=request, response=response)

    def json(self):
        return self._payload


def test_fake_gemini_response_uses_document_and_query_tasks(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        # A deliberately oversized vector exercises Google's required manual
        # normalization after truncating the MRL prefix to 1536 dimensions.
        return _FakeResponse({"embedding": {"values": [3.0] * 3072}})

    monkeypatch.setattr("reminder_agent.infrastructure.embeddings.httpx.post", fake_post)
    provider = GeminiEmbeddingProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
    )

    document = provider.embed_document("My favorite food is biryani")
    query = provider.embed_query("Which food do I like most?")

    assert len(document) == 1536
    assert math.isclose(math.sqrt(sum(value * value for value in document)), 1.0)
    assert math.isclose(document[0], 1 / math.sqrt(1536))
    assert calls[0]["headers"] == {"x-goog-api-key": "test-key"}
    assert calls[0]["json"]["embedContentConfig"]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert calls[1]["json"]["embedContentConfig"]["taskType"] == "RETRIEVAL_QUERY"
    assert all(call["json"]["embedContentConfig"]["outputDimensionality"] == 1536 for call in calls)
    assert all(call["json"]["model"] == "models/gemini-embedding-001" for call in calls)
    assert query == document


def test_fake_gemini_malformed_response_is_sanitized(monkeypatch) -> None:
    monkeypatch.setattr(
        "reminder_agent.infrastructure.embeddings.httpx.post",
        lambda *args, **kwargs: _FakeResponse({"unexpected": []}),
    )
    provider = GeminiEmbeddingProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
    )

    with pytest.raises(EmbeddingProviderError) as error:
        provider.embed_query("test")
    assert error.value.code == "PROVIDER_UNAVAILABLE"
    assert "test" not in str(error.value)


def test_gemini_rejects_incompatible_model_or_dimensions() -> None:
    with pytest.raises(ValueError, match="gemini-embedding-001"):
        GeminiEmbeddingProvider(
            base_url="https://example.test",
            api_key="test-key",
            model="gemini-embedding-2",
        )
    with pytest.raises(ValueError, match=r"vector\(1536\)"):
        GeminiEmbeddingProvider(
            base_url="https://example.test",
            api_key="test-key",
            dimensions=768,
        )


def test_live_gemini_embedding_round_trip() -> None:
    if os.getenv("RUN_LIVE_GEMINI") != "1":
        pytest.skip("set RUN_LIVE_GEMINI=1 to enable the network live test")
    api_key = _configured_gemini_key()
    if not api_key:
        pytest.skip("EMBEDDING_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY is not configured")
    provider = GeminiEmbeddingProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key=api_key,
    )
    document = provider.embed_document("Memorae live embedding verification")
    query = provider.embed_query("Which memory is being verified?")

    assert len(document) == 1536
    assert len(query) == 1536
    assert math.isclose(math.sqrt(sum(value * value for value in document)), 1.0, rel_tol=1e-5)


def test_live_gemini_save_then_retrieve_against_postgresql(session_factory) -> None:
    if os.getenv("RUN_LIVE_GEMINI") != "1":
        pytest.skip("set RUN_LIVE_GEMINI=1 to enable the network live test")
    if not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"):
        pytest.skip("requires TEST_DATABASE_URL backed by PostgreSQL")
    api_key = _configured_gemini_key()
    if not api_key:
        pytest.skip("EMBEDDING_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY is not configured")

    provider = GeminiEmbeddingProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key=api_key,
    )
    owner = User(
        name="Gemini live verification",
        timezone="UTC",
        onboarding_status="ACTIVE",
    )
    with session_factory() as db, db.begin():
        db.add(owner)
        db.flush()
        saved = save_memory_fact(
            db,
            user_id=owner.id,
            content="my favorite food is biryani",
            embedding_provider=provider,
        )
        assert saved.embedding_status == "INDEXED"

    with session_factory() as db:
        result = lookup_memory_fact(
            db,
            user_id=owner.id,
            query="food like most",
            semantic_query="Which food do I like most?",
            embedding_provider=provider,
        )
        assert result.status == "CANDIDATES"
        assert result.candidates[0].content == "my favorite food is biryani"


def test_live_gemini_groq_postgresql_memory_retrieval_end_to_end(session_factory) -> None:
    if os.getenv("RUN_LIVE_MEMORY_RETRIEVAL") != "1":
        pytest.skip("set RUN_LIVE_MEMORY_RETRIEVAL=1 to enable the live provider test")
    if not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"):
        pytest.skip("requires TEST_DATABASE_URL backed by PostgreSQL")
    gemini_key = _configured_gemini_key()
    llm_key = _configured_llm_key()
    if not gemini_key or not llm_key:
        pytest.skip("Gemini and Groq/LLM API keys are both required")

    settings = Settings()
    provider = GeminiEmbeddingProvider(
        base_url=settings.embedding_gemini_base_url,
        api_key=gemini_key,
        model=settings.embedding_gemini_model,
        dimensions=settings.embedding_dimensions,
    )
    answerer = OpenAICompatibleMemoryAnswerer(
        "https://api.groq.com/openai/v1",
        llm_key,
        "openai/gpt-oss-120b",
        settings.llm_timeout_seconds,
    )
    owner = User(
        name="Live retrieval owner",
        timezone="UTC",
        onboarding_status="ACTIVE",
    )
    with session_factory() as db, db.begin():
        db.add(owner)
        db.flush()
        for content in (
            "my parking slot is B17",
            "my favorite food is biryani",
            "my office location is Lahore",
            "my dog's name is Milo",
        ):
            saved = save_memory_fact(
                db,
                user_id=owner.id,
                content=content,
                embedding_provider=provider,
            )
            assert saved.embedding_status == "INDEXED"

    with session_factory() as db:
        result = lookup_memory_fact(
            db,
            user_id=owner.id,
            query="leave my car",
            semantic_query="Where do I leave my car?",
            embedding_provider=provider,
        )
        assert result.status == "CANDIDATES"
        assert 1 <= len(result.candidates) <= 3
        assert {fact.user_id for fact in result.candidates} == {owner.id}
        proposal = answerer.answer(
            question="Where do I leave my car?",
            memories=tuple(fact.content for fact in result.candidates),
        )
        assert proposal is not None and proposal.decision == "ANSWER"
        assert proposal.answer is not None and "B17" in proposal.answer
        assert proposal.validates_candidate_count(len(result.candidates))
