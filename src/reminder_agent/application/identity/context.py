"""Reload and validate trusted user/identity request context."""

from __future__ import annotations

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    ApplicationRequest,
    AuthenticatedRequestContext,
)
from reminder_agent.infrastructure.db.models import ChannelIdentity, User

ACTIVE = "ACTIVE"
LINKED = "LINKED"


def context_matches_persisted_identity(
    *,
    context: AuthenticatedRequestContext,
    user: User | None,
    identity: ChannelIdentity | None,
) -> bool:
    """Verify canonical ownership and linked identity state."""

    return bool(
        user is not None
        and identity is not None
        and user.id == context.canonical_user_id
        and identity.id == context.verified_identity.identity_id
        and identity.user_id == user.id
        and identity.channel_type == context.verified_identity.channel_type
        and identity.status == LINKED
    )


def authorized_user_for_request(db: Session, *, request: ApplicationRequest) -> User | None:
    """Reload and verify the trusted user/identity pair before application work."""

    context = request.context
    user = db.get(User, context.canonical_user_id)
    identity = db.get(ChannelIdentity, context.verified_identity.identity_id)
    if not context_matches_persisted_identity(context=context, user=user, identity=identity):
        return None
    return user
