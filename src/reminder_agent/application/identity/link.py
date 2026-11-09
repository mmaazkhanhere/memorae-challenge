"""Application coordination for linking verified external identities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.features.onboarding.onboarding_validation import (
    InvalidPhoneNumber,
    normalize_phone_number,
)
from reminder_agent.infrastructure.db.models import ChannelIdentity, User

LINKED = "LINKED"
PENDING = "PENDING"
PHONE = "PHONE"
UNLINKED = "UNLINKED"


@dataclass(frozen=True)
class PhoneLinkMutationResult:
    """Outcome of a staged phone identity mutation."""

    outcome: str
    reason: str | None = None
    replaced_existing: bool = False


@dataclass(frozen=True)
class PhoneLinkPreparation:
    """Validated phone input and any staged local-number identity context."""

    outcome: str
    normalized: str | None = None
    reason: str | None = None
    region: str | None = None


def prepare_phone_link(
    db: Session,
    *,
    user: User,
    phone_number: str,
    default_region: str | None,
) -> PhoneLinkPreparation:
    """Validate a number and stage missing-region context."""

    region = user.phone_region or default_region
    try:
        normalized = normalize_phone_number(phone_number, default_region=region)
    except InvalidPhoneNumber as exc:
        if exc.reason != "missing_region":
            return PhoneLinkPreparation("INVALID", reason=exc.reason, region=region)
        for pending in db.scalars(
            select(ChannelIdentity).where(
                ChannelIdentity.user_id == user.id,
                ChannelIdentity.channel_type == PHONE,
                ChannelIdentity.status == PENDING,
            )
        ):
            pending.status = UNLINKED
        db.add(
            ChannelIdentity(
                user_id=user.id,
                channel_type=PHONE,
                external_id=phone_number,
                status=PENDING,
            )
        )
        return PhoneLinkPreparation("MISSING_REGION", reason=exc.reason, region=region)
    return PhoneLinkPreparation("NORMALIZED", normalized=normalized, region=region)


def link_phone_identity(
    db: Session,
    *,
    user: User,
    normalized: str,
    pending: ChannelIdentity | None = None,
) -> PhoneLinkMutationResult:
    """Stage phone linking inside the caller's transaction and savepoint."""

    other_owner = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.channel_type == PHONE,
            ChannelIdentity.external_id == normalized,
            ChannelIdentity.status == LINKED,
        )
    )
    if other_owner is not None and other_owner.user_id != user.id:
        return PhoneLinkMutationResult("CONFLICT", reason="active_number_conflict")

    current = find_phone_identity(db, user_id=user.id)
    if current is not None and current.external_id == normalized:
        return PhoneLinkMutationResult("ALREADY_LINKED")

    try:
        with db.begin_nested():
            if current is not None:
                current.status = UNLINKED
            if pending is not None:
                pending.external_id = normalized
                pending.status = LINKED
                pending.verified_at = datetime.now(UTC)
            else:
                db.add(
                    ChannelIdentity(
                        user_id=user.id,
                        channel_type=PHONE,
                        external_id=normalized,
                        status=LINKED,
                        verified_at=datetime.now(UTC),
                    )
                )
            db.flush()
    except IntegrityError:
        return PhoneLinkMutationResult("CONFLICT", reason="database_uniqueness_conflict")

    return PhoneLinkMutationResult("LINKED", replaced_existing=current is not None)


def find_phone_identity(
    db: Session,
    *,
    user_id: UUID,
    external_id: str | None = None,
) -> ChannelIdentity | None:
    """Find the active phone identity owned by a canonical user."""

    filters = [
        ChannelIdentity.user_id == user_id,
        ChannelIdentity.channel_type == PHONE,
        ChannelIdentity.status == LINKED,
    ]
    if external_id is not None:
        filters.append(ChannelIdentity.external_id == external_id)
    return db.scalar(select(ChannelIdentity).where(*filters))


def find_pending_phone_identity(
    db: Session,
    *,
    user_id: UUID,
) -> ChannelIdentity | None:
    """Find the newest pending phone identity for a canonical user."""

    return db.scalar(
        select(ChannelIdentity)
        .where(
            ChannelIdentity.user_id == user_id,
            ChannelIdentity.channel_type == PHONE,
            ChannelIdentity.status == PENDING,
        )
        .order_by(ChannelIdentity.created_at.desc())
    )


__all__ = [
    "PhoneLinkMutationResult",
    "PhoneLinkPreparation",
    "find_pending_phone_identity",
    "find_phone_identity",
    "link_phone_identity",
    "prepare_phone_link",
]
