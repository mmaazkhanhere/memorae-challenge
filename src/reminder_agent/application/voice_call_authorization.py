"""Application-owned authorization for inbound Vapi calls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import (
    AuthenticatedRequestContext,
    RequestSource,
    VerifiedChannelIdentity,
)
from reminder_agent.application.identity.resolve import resolve_linked_identity
from reminder_agent.features.onboarding.onboarding_validation import (
    InvalidPhoneNumber,
    normalize_phone_number,
)
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    User,
    VoiceCallAuthorization,
)

PHONE = "PHONE"
LINKED = "LINKED"
ACTIVE_USER = "ACTIVE"
ACTIVE_CALL = "ACTIVE"
ENDED_CALL = "ENDED"
EXPIRED_CALL = "EXPIRED"


@dataclass(frozen=True)
class VoiceCallAuthorizationResult:
    """Non-provider-specific result of an inbound call authorization attempt."""

    outcome: str
    authorization: VoiceCallAuthorization | None = None
    context: AuthenticatedRequestContext | None = None

    @property
    def authorized(self) -> bool:
        return self.outcome == "AUTHORIZED" and self.context is not None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _context_for(
    authorization: VoiceCallAuthorization,
) -> AuthenticatedRequestContext:
    return AuthenticatedRequestContext(
        source=RequestSource.VAPI,
        canonical_user_id=authorization.user_id,
        verified_identity=VerifiedChannelIdentity(
            identity_id=authorization.phone_identity_id,
            channel_type=PHONE,
        ),
        external_session_id=authorization.provider_call_id,
    )


def _rejected(outcome: str = "REJECTED") -> VoiceCallAuthorizationResult:
    return VoiceCallAuthorizationResult(outcome=outcome)


def _existing_authorization(
    db: Session,
    *,
    provider_call_id: str,
    now: datetime,
    user_id: UUID,
    identity_id: UUID,
) -> VoiceCallAuthorizationResult:
    existing = db.scalar(
        select(VoiceCallAuthorization)
        .where(VoiceCallAuthorization.provider_call_id == provider_call_id)
        .with_for_update()
    )
    if existing is None:
        return _rejected("ERROR")
    if existing.user_id != user_id or existing.phone_identity_id != identity_id:
        return _rejected("CONFLICT")
    if existing.status != ACTIVE_CALL or _utc(existing.expires_at) <= now:
        if existing.status == ACTIVE_CALL and _utc(existing.expires_at) <= now:
            existing.status = EXPIRED_CALL
        return _rejected("EXPIRED")
    return VoiceCallAuthorizationResult(
        outcome="AUTHORIZED",
        authorization=existing,
        context=_context_for(existing),
    )


def authorize_incoming_vapi_call(
    db: Session,
    *,
    provider_call_id: str,
    caller_phone_number: str,
    now: datetime,
    ttl_seconds: int = 3600,
) -> VoiceCallAuthorizationResult:
    """Authorize one Vapi call from a currently linked phone identity.

    The caller number is normalized and resolved inside the same short
    transaction that creates the call binding.  Replays are idempotent, and all
    identity or persistence failures fail closed.
    """

    provider_call_id = provider_call_id.strip()
    if not provider_call_id or len(provider_call_id) > 255 or ttl_seconds < 1:
        return _rejected("INVALID")
    try:
        normalized_phone = normalize_phone_number(caller_phone_number)
    except (InvalidPhoneNumber, TypeError, AttributeError):
        return _rejected()

    now = _utc(now)
    try:
        db.rollback()
        with db.begin():
            resolved = resolve_linked_identity(
                db,
                channel_type=PHONE,
                external_id=normalized_phone,
            )
            if resolved is None:
                return _rejected()
            user = resolved.user
            identity = resolved.identity
            if user.onboarding_status != ACTIVE_USER or not (user.timezone or "").strip():
                return _rejected()

            existing = db.scalar(
                select(VoiceCallAuthorization)
                .where(VoiceCallAuthorization.provider_call_id == provider_call_id)
                .with_for_update()
            )
            if existing is not None:
                return _existing_authorization(
                    db,
                    provider_call_id=provider_call_id,
                    now=now,
                    user_id=user.id,
                    identity_id=identity.id,
                )

            authorization = VoiceCallAuthorization(
                provider_call_id=provider_call_id,
                user_id=user.id,
                phone_identity_id=identity.id,
                status=ACTIVE_CALL,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
            db.add(authorization)
            db.flush()
            return VoiceCallAuthorizationResult(
                outcome="AUTHORIZED",
                authorization=authorization,
                context=_context_for(authorization),
            )
    except IntegrityError:
        # Another webhook may have won the unique call-ID race.  Re-read the
        # row and apply the same identity checks rather than creating a second
        # binding or allowing an inconsistent replay.
        try:
            db.rollback()
            with db.begin():
                resolved = resolve_linked_identity(
                    db,
                    channel_type=PHONE,
                    external_id=normalized_phone,
                )
                if resolved is None:
                    return _rejected()
                return _existing_authorization(
                    db,
                    provider_call_id=provider_call_id,
                    now=now,
                    user_id=resolved.user.id,
                    identity_id=resolved.identity.id,
                )
        except Exception:
            db.rollback()
            return _rejected("ERROR")
    except Exception:
        db.rollback()
        return _rejected("ERROR")


def resolve_vapi_call_context(
    db: Session,
    *,
    provider_call_id: str,
    now: datetime,
) -> AuthenticatedRequestContext | None:
    """Reload a call authorization and its linked identity before tool work."""

    call_id = provider_call_id.strip()
    if not call_id or len(call_id) > 255:
        return None
    try:
        authorization = db.scalar(
            select(VoiceCallAuthorization).where(
                VoiceCallAuthorization.provider_call_id == call_id,
                VoiceCallAuthorization.status == ACTIVE_CALL,
            )
        )
        if authorization is None or _utc(authorization.expires_at) <= _utc(now):
            return None
        user = db.get(User, authorization.user_id)
        identity = db.get(ChannelIdentity, authorization.phone_identity_id)
        if (
            user is None
            or identity is None
            or user.onboarding_status != ACTIVE_USER
            or not (user.timezone or "").strip()
            or identity.user_id != user.id
            or identity.channel_type != PHONE
            or identity.status != LINKED
        ):
            return None
        return _context_for(authorization)
    except Exception:
        db.rollback()
        return None


def end_vapi_call(
    db: Session,
    *,
    provider_call_id: str,
    now: datetime,
) -> bool:
    """End a call authorization after a provider end-of-call event."""

    try:
        authorization = db.scalar(
            select(VoiceCallAuthorization).where(
                VoiceCallAuthorization.provider_call_id == provider_call_id.strip()
            )
        )
        if authorization is None:
            return False
        authorization.status = ENDED_CALL
        authorization.ended_at = _utc(now)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False


# Short aliases keep the adapter seam easy to discover for future Vapi tools.
authorize_vapi_call = authorize_incoming_vapi_call
resolve_vapi_authorization = resolve_vapi_call_context


__all__ = [
    "VoiceCallAuthorizationResult",
    "authorize_incoming_vapi_call",
    "authorize_vapi_call",
    "end_vapi_call",
    "resolve_vapi_authorization",
    "resolve_vapi_call_context",
]
