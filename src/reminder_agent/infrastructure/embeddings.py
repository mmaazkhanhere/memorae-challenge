"""Provider-agnostic embedding boundary with privacy-safe failure semantics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from reminder_agent.infrastructure.db.models import MEMORY_EMBEDDING_DIMENSIONS
from reminder_agent.infrastructure.settings import Settings


class EmbeddingProviderError(RuntimeError):
    """A sanitized provider failure that never includes memory or response data."""

    def __init__(self, code: str = "PROVIDER_UNAVAILABLE") -> None:
        self.code = code
        super().__init__(code)


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str
    dimensions: int
    is_external: bool

    def embed(self, text: str) -> list[float]: ...


def embed_document(provider: EmbeddingProvider, text: str) -> list[float]:
    """Embed saved content with the provider's document task when supported."""

    method = getattr(provider, "embed_document", None)
    return method(text) if callable(method) else provider.embed(text)


def embed_query(provider: EmbeddingProvider, text: str) -> list[float]:
    """Embed a retrieval question with the provider's query task when supported."""

    method = getattr(provider, "embed_query", None)
    return method(text) if callable(method) else provider.embed(text)


@dataclass(frozen=True)
class EmbeddingPrivacyPolicy:
    """External providers receive sensitive text only after explicit opt-in."""

    allow_sensitive_external: bool = False


_SENSITIVE_MEMORY = re.compile(
    r"\b(?:password|passphrase|passcode|pin|secret|api[\s_-]*key|token|"
    r"access\s+code|gate\s+code|door\s+code|security\s+code|cvv|cvc|"
    r"private\s+key|recovery\s+(?:code|phrase)|seed\s+phrase|"
    r"e-?mail(?:\s+address)?|phone\s+number|mobile\s+number|telephone\s+number|"
    r"account\s+number)\b",
    re.IGNORECASE,
)


def contains_sensitive_memory(value: str) -> bool:
    return _SENSITIVE_MEMORY.search(value) is not None


def validate_embedding(values: list[float], *, dimensions: int) -> list[float]:
    if len(values) != dimensions:
        raise EmbeddingProviderError("INVALID_DIMENSIONS")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingProviderError("INVALID_VECTOR")
    if not any(value != 0.0 for value in vector):
        raise EmbeddingProviderError("INVALID_VECTOR")
    return vector


class OpenAICompatibleEmbeddingProvider:
    """Small adapter for OpenAI-compatible ``/embeddings`` endpoints."""

    is_external = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int = MEMORY_EMBEDDING_DIMENSIONS,
        timeout_seconds: float = 10.0,
    ) -> None:
        if dimensions != MEMORY_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding dimensions must match vector({MEMORY_EMBEDDING_DIMENSIONS})"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "input": text,
                    "dimensions": self.dimensions,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            values = payload["data"][0]["embedding"]
            return validate_embedding(values, dimensions=self.dimensions)
        except EmbeddingProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            # Do not retain provider response bodies or input text in errors/logs.
            raise EmbeddingProviderError() from exc

    embed_document = embed
    embed_query = embed


def _normalize_gemini_embedding(values: list[float], *, dimensions: int) -> list[float]:
    """Normalize the MRL prefix required by gemini-embedding-001.

    Google documents that gemini-embedding-001 needs manual normalization when
    output_dimensionality is smaller than its native 3072 dimensions.
    """

    raw = [float(value) for value in values]
    if len(raw) < dimensions:
        raise EmbeddingProviderError("INVALID_DIMENSIONS")
    vector = raw[:dimensions]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingProviderError("INVALID_VECTOR")
    return validate_embedding([value / norm for value in vector], dimensions=dimensions)


class GeminiEmbeddingProvider:
    """Google Gemini Embedding API adapter for document/query retrieval tasks."""

    is_external = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimensions: int = MEMORY_EMBEDDING_DIMENSIONS,
        timeout_seconds: float = 10.0,
    ) -> None:
        if model != "gemini-embedding-001":
            raise ValueError("Gemini embedding model must be gemini-embedding-001")
        if dimensions != MEMORY_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding dimensions must match vector({MEMORY_EMBEDDING_DIMENSIONS})"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_QUERY")

    def _embed(self, text: str, *, task_type: str) -> list[float]:
        try:
            response = httpx.post(
                f"{self.base_url}/models/{self.model}:embedContent",
                headers={"x-goog-api-key": self.api_key},
                json={
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "embedContentConfig": {
                        "taskType": task_type,
                        "outputDimensionality": self.dimensions,
                    },
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            values = payload["embedding"]["values"]
            return _normalize_gemini_embedding(values, dimensions=self.dimensions)
        except EmbeddingProviderError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            # Do not retain provider response bodies or input text in errors/logs.
            raise EmbeddingProviderError() from exc


def create_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    provider_name = settings.embedding_provider.strip().casefold()
    if provider_name == "disabled":
        return None
    if provider_name == "gemini":
        api_key = (settings.embedding_api_key or "").strip()
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for the configured provider")
        return GeminiEmbeddingProvider(
            base_url=settings.embedding_gemini_base_url,
            api_key=api_key,
            model=settings.embedding_gemini_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    if provider_name != "openai_compatible":
        raise ValueError("unsupported embedding provider")
    api_key = (settings.embedding_api_key or "").strip()
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY is required for the configured provider")
    return OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url,
        api_key=api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
