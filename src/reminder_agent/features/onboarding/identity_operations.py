"""Application-owned onboarding identity compatibility operations."""

from reminder_agent.application.identity.link import (
    PhoneLinkMutationResult,
    PhoneLinkPreparation,
    find_pending_phone_identity,
    find_phone_identity,
    link_phone_identity,
    prepare_phone_link,
)
from reminder_agent.application.identity.resolve import ResolvedIdentity as ResolvedLinkedIdentity
from reminder_agent.application.identity.resolve import resolve_linked_identity
from reminder_agent.infrastructure.database.models import ChannelIdentity

ACTIVE = "ACTIVE"
LINKED = "LINKED"
PENDING = "PENDING"
PHONE = "PHONE"
UNLINKED = "UNLINKED"


def find_linked_identity(db, *, channel_type: str, external_id: str) -> ChannelIdentity | None:
    """Find a linked identity row for legacy callers."""

    from sqlalchemy import select

    return db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.channel_type == channel_type,
            ChannelIdentity.external_id == external_id,
            ChannelIdentity.status == LINKED,
        )
    )


__all__ = [
    "ACTIVE",
    "LINKED",
    "PENDING",
    "PHONE",
    "UNLINKED",
    "PhoneLinkMutationResult",
    "PhoneLinkPreparation",
    "ResolvedLinkedIdentity",
    "find_linked_identity",
    "find_pending_phone_identity",
    "find_phone_identity",
    "link_phone_identity",
    "prepare_phone_link",
    "resolve_linked_identity",
]
