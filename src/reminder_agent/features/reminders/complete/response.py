"""Response staging and replay coordination for completion workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

ResponseT = TypeVar("ResponseT")


def stage_completion_response[ResponseT](
    *,
    stage_response: Callable[..., ResponseT],
    text: str,
    response_kind: str,
    operation: str = "COMPLETE_REMINDER",
    cleanup_provider_message_id: str | None = None,
) -> ResponseT:
    """Stage a completion response with the established durable metadata."""

    kwargs: dict[str, object] = {
        "text": text,
        "response_kind": response_kind,
        "operation": operation,
    }
    if cleanup_provider_message_id is not None:
        kwargs["cleanup_provider_message_id"] = cleanup_provider_message_id
    return stage_response(
        **kwargs,
    )


def commit_completion_response[ResponseT](
    db: Session,
    *,
    reply: ResponseT,
    recover_from_race: Callable[[], ResponseT | None],
) -> ResponseT:
    """Commit a directly staged response and replay the winner on a race."""

    try:
        db.commit()
    except IntegrityError:
        duplicate = recover_from_race()
        if duplicate is not None:
            return duplicate
        raise
    return reply
