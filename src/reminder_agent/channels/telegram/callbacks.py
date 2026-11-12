"""Telegram callback encoding and signature validation."""

from __future__ import annotations

import hashlib
import hmac
from uuid import UUID


def callback_data(occurrence_id: UUID, secret: str) -> str:
    value = str(occurrence_id)
    signature = hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:16]
    return f"done:{value}:{signature}"


def parse_callback_data(value: str, secret: str) -> UUID | None:
    if len(value) > 64:
        return None
    parts = value.split(":")
    if len(parts) != 3 or parts[0] != "done":
        return None
    try:
        occurrence_id = UUID(parts[1])
    except ValueError:
        return None
    expected = hmac.new(secret.encode(), parts[1].encode(), hashlib.sha256).hexdigest()[:16]
    return occurrence_id if hmac.compare_digest(parts[2], expected) else None
