"""Application services for resolving, linking, and authenticating identities."""

from reminder_agent.application.identity.context import (
    authorized_user_for_request,
    context_matches_persisted_identity,
)
from reminder_agent.application.identity.resolve import (
    ResolvedIdentity,
    resolve_linked_identity,
)

__all__ = [
    "ResolvedIdentity",
    "authorized_user_for_request",
    "context_matches_persisted_identity",
    "resolve_linked_identity",
]
