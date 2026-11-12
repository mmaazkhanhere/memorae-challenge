"""Provider boundary for OpenAI-compatible structured model calls."""

from __future__ import annotations

import time
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
        for attempt in range(3):
            try:
                response = httpx.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except (httpx.ConnectError, httpx.ReadTimeout):
                if attempt == 2:
                    raise
                time.sleep(0.25 * (2**attempt))
                continue
            status_code = getattr(response, "status_code", 200)
            if status_code == 429 or status_code >= 500:
                if attempt == 2:
                    response.raise_for_status()
                retry_after = response.headers.get("retry-after")
                try:
                    delay = min(float(retry_after), 2.0) if retry_after else 0.25 * (2**attempt)
                except ValueError:
                    delay = 0.25 * (2**attempt)
                time.sleep(delay)
                continue
            response.raise_for_status()
            response_payload: dict[str, Any] = response.json()
            return response_payload, response_payload["choices"][0]["message"]["content"]
        raise RuntimeError("LLM request retry loop exited unexpectedly")


def supports_strict_json_schema(base_url: str, model: str) -> bool:
    """Return whether the configured provider/model accepts strict schemas."""

    host = (urlsplit(base_url).hostname or "").lower()
    return host == "api.openai.com" or (
        host == "api.groq.com" and model in _STRICT_JSON_SCHEMA_MODELS
    )
