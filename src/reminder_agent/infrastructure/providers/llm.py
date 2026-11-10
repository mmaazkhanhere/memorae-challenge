"""Provider boundary for OpenAI-compatible structured model calls."""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

_STRICT_JSON_SCHEMA_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}


class StructuredIntentProvider(Protocol):
    """Minimal provider contract used by classifiers and extractors."""

    url: str

    def complete(self, payload: dict[str, Any]) -> tuple[dict[str, Any], Any]: ...


class OpenAICompatibleLLMClient:
    """Shared low-level client for the configured OpenAI-compatible provider."""

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 10.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete(self, payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        response = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        response_payload: dict[str, Any] = response.json()
        return response_payload, response_payload["choices"][0]["message"]["content"]


def supports_strict_json_schema(base_url: str, model: str) -> bool:
    """Return whether the configured provider/model accepts strict schemas."""

    host = (urlsplit(base_url).hostname or "").lower()
    return host == "api.openai.com" or (
        host == "api.groq.com" and model in _STRICT_JSON_SCHEMA_MODELS
    )
