"""Resolve provider identity evidence to the canonical application user."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.infrastructure.db.models import ChannelIdentity, User

LINKED = "LINKED"


@dataclass(frozen=True)
class ResolvedIdentity:
    """Canonical user and the linked identity used to resolve it."""

    user: User
    identity: ChannelIdentity


def resolve_linked_identity(
    db: Session,
    *,
    channel_type: str,
    external_id: str,
) -> ResolvedIdentity | None:
    """Resolve one active external identity to its canonical user."""

    identity = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.channel_type == channel_type,
            ChannelIdentity.external_id == external_id,
            ChannelIdentity.status == LINKED,
        )
    )
    if identity is None:
        return None
    user = db.get(User, identity.user_id)
    if user is None:
        return None
    return ResolvedIdentity(user=user, identity=identity)
