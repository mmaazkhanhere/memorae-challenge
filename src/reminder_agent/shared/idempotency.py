"""Shared idempotency contracts used by channel-facing services."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.infrastructure.database.models import IdempotencyRecord


class IdempotencyConflict(ValueError):
    """Raised when an idempotency key is reused with different request data."""


def _request_key(request: object) -> str:
    """Read a provider's stable idempotency key without importing its payload type."""

    key = getattr(request, "update_id", getattr(request, "idempotency_key", request))
    return str(key)


def request_hash(request: object) -> str:
    """Hash a normalized provider request using its retry-stable representation.

    Telegram keeps its existing Pydantic representation and voice transcript
    rule; the duck-typed boundary means this shared store no longer imports the
    Telegram adapter.
    """

    message = getattr(request, "message", None)
    voice = getattr(message, "voice", None)
    if message is None or voice is None:
        model_dump_json = getattr(request, "model_dump_json", None)
        if model_dump_json is None:
            canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)
        else:
            canonical = model_dump_json(by_alias=True, exclude_none=True)
        if not isinstance(canonical, str):
            canonical = str(canonical)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    model_dump = cast(Any, request).model_dump
    payload = model_dump(by_alias=True, exclude_none=True)
    # A transient STT overlay must not change the digest of the original voice
    # update. This keeps retries idempotent without persisting the transcript.
    if message is not None and voice is not None:
        payload.get("message", {}).pop("text", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_idempotency_record(
    db: Session,
    request: object,
    *,
    source: str = "TELEGRAM",
) -> IdempotencyRecord | None:
    """Load the record for an update without changing the session transaction."""
    return db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == source,
            IdempotencyRecord.idempotency_key == _request_key(request),
        )
    )


def ensure_request_matches(record: IdempotencyRecord, request: object) -> None:
    """Raise the stable conflict error when a key is reused with new content."""
    if record.request_hash != request_hash(request):
        raise IdempotencyConflict("Telegram update was reused with different content")
