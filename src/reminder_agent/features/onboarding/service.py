"""Provider-neutral onboarding application service.

The service consumes normalized onboarding requests and returns channel-neutral
responses. Provider payloads, keyboards, and delivery retries stay in channel
adapters.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import OutboundResponse
from reminder_agent.application.identity.link import (
    find_pending_phone_identity,
    find_phone_identity,
)
from reminder_agent.application.identity.resolve import resolve_linked_identity
from reminder_agent.features.onboarding import message_templates as messages
from reminder_agent.features.onboarding.onboarding_validation import (
    InvalidPhoneRegion,
    InvalidTimezone,
    validate_phone_region,
    validate_timezone,
)
from reminder_agent.features.onboarding.phone_linking_service import process_phone_link
from reminder_agent.features.onboarding.request_contracts import (
    OnboardingAction,
    OnboardingRequest,
)
from reminder_agent.infrastructure.db.models import ChannelIdentity, IdempotencyRecord, User
from reminder_agent.shared.idempotency import IdempotencyConflict

ACTIVE: Final = "ACTIVE"
LINKED: Final = "LINKED"
PENDING: Final = "PENDING"
PHONE: Final = "PHONE"
UNLINKED: Final = "UNLINKED"


def _existing_record(db: Session, request: OnboardingRequest) -> IdempotencyRecord | None:
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == request.source.value,
            IdempotencyRecord.idempotency_key == request.idempotency_key,
        )
    )
    if record is not None and record.request_hash != request.request_hash:
        raise IdempotencyConflict("Onboarding request was reused with different content")
    return record


def _reply_from_record(record: IdempotencyRecord) -> OutboundResponse:
    assert record.response is not None
    return OutboundResponse(
        content=str(record.response["text"]),
        response_kind="REPLAY",
        presentation_hint=record.response.get("presentation_hint"),
        idempotency_key=record.idempotency_key,
        should_send=False,
        is_duplicate=True,
    )


def _store_result(
    db: Session,
    *,
    request: OnboardingRequest,
    user_id: UUID | None,
    operation: str,
    text: str,
    presentation_hint: str | None = None,
    now: datetime,
) -> OutboundResponse:
    response = OutboundResponse(
        content=text,
        response_kind=operation,
        presentation_hint=presentation_hint,
        idempotency_key=request.idempotency_key,
    )
    db.add(
        IdempotencyRecord(
            source=request.source.value,
            idempotency_key=request.idempotency_key,
            user_id=user_id,
            operation=operation,
            request_hash=request.request_hash,
            status="SUCCEEDED",
            delivery_status="PENDING",
            response={
                "destination_id": request.identity.destination_id,
                "text": response.content,
                "presentation_hint": presentation_hint,
                # Retain the old key for existing replay rows. New adapters
                # use their own destination field when rendering a response.
                "chat_id": request.identity.destination_id,
                "reply_markup": None,
            },
            created_at=now,
            updated_at=now,
        )
    )
    return response


def _unknown_sender_response(
    db: Session, *, request: OnboardingRequest, now: datetime
) -> OutboundResponse:
    return _store_result(
        db,
        request=request,
        user_id=None,
        operation="ONBOARDING_UNKNOWN_SENDER",
        text=messages.unknown_sender(),
        now=now,
    )


def _channel_identity(request: OnboardingRequest, *, user_id: UUID) -> ChannelIdentity:
    identity = ChannelIdentity(
        user_id=user_id,
        channel_type=request.identity.channel_type,
        external_id=request.identity.external_id,
        status=LINKED,
        verified_at=datetime.now(UTC),
    )
    # The current schema retains Telegram's historical destination column. The
    # application treats it as persistence metadata; adapters own its meaning.
    if request.identity.channel_type == "TELEGRAM":
        identity.telegram_chat_id = int(request.identity.destination_id)
    return identity


def _handle_start(
    db: Session, *, request: OnboardingRequest, now: datetime
) -> OutboundResponse:
    identity = resolve_linked_identity(
        db,
        channel_type=request.identity.channel_type,
        external_id=request.identity.external_id,
    )
    if identity is None:
        user = User(name=request.display_name or "User", onboarding_status=PENDING)
        db.add(user)
        db.flush()
        db.add(_channel_identity(request, user_id=user.id))
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="ONBOARDING_START",
            text=messages.welcome(user.name),
            now=now,
        )

    user = identity.user
    if user.onboarding_status == ACTIVE:
        phone = find_phone_identity(db, user_id=user.id)
        if phone is None:
            return _store_result(
                db,
                request=request,
                user_id=user.id,
                operation="ONBOARDING_START",
                text=messages.already_onboarded_without_phone(user.name),
                presentation_hint="PHONE_CONTACT_REQUEST",
                now=now,
            )
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="ONBOARDING_START",
            text=messages.already_onboarded(user.name),
            presentation_hint="REMOVE_KEYBOARD",
            now=now,
        )

    return _store_result(
        db,
        request=request,
        user_id=user.id,
        operation="ONBOARDING_START",
        text=messages.welcome_back(user.name),
        now=now,
    )


def _phone_link_response(
    db: Session,
    *,
    request: OnboardingRequest,
    user: User,
    decision,
    now: datetime,
) -> OutboundResponse:
    if decision.outcome == "CONFLICT":
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_LINK_CONFLICT",
            text=messages.phone_conflict(),
            presentation_hint="PHONE_CONTACT_REQUEST",
            now=now,
        )
    if decision.outcome == "ALREADY_LINKED":
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_LINK_ALREADY_LINKED",
            text=messages.phone_already_linked(),
            presentation_hint="REMOVE_KEYBOARD",
            now=now,
        )
    return _store_result(
        db,
        request=request,
        user_id=user.id,
        operation="PHONE_LINKED",
        text=messages.phone_linked(),
        presentation_hint="REMOVE_KEYBOARD",
        now=now,
    )


def _handle_contact(
    db: Session,
    *,
    request: OnboardingRequest,
    user: User,
    default_region: str | None,
    now: datetime,
) -> OutboundResponse:
    if user.onboarding_status != ACTIVE:
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_LINK_REQUIRES_ONBOARDING",
            text=messages.phone_link_requires_onboarding(),
            now=now,
        )
    if request.contact_owner_id != request.identity.external_id:
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_LINK_OWNERSHIP_REJECTED",
            text=messages.phone_ownership_rejected(),
            presentation_hint="PHONE_CONTACT_REQUEST",
            now=now,
        )
    if not request.contact_phone:
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_LINK_INVALID",
            text=messages.phone_number_invalid_retry(),
            presentation_hint="PHONE_CONTACT_REQUEST",
            now=now,
        )
    decision = process_phone_link(
        db,
        user=user,
        phone_number=request.contact_phone,
        default_region=default_region,
    )
    if decision.outcome not in {"LINKED", "CONFLICT", "ALREADY_LINKED"}:
        if decision.outcome == "MISSING_REGION":
            text = messages.local_phone_region_required()
        else:
            text = messages.phone_number_invalid_retry()
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_LINK_INVALID",
            text=text,
            presentation_hint=(
                "PHONE_REGION_REQUIRED"
                if decision.outcome == "MISSING_REGION"
                else "PHONE_CONTACT_REQUEST"
            ),
            now=now,
        )
    return _phone_link_response(
        db, request=request, user=user, decision=decision, now=now
    )


def _handle_phone_region(
    db: Session,
    *,
    request: OnboardingRequest,
    user: User,
    now: datetime,
) -> OutboundResponse:
    if user.onboarding_status != ACTIVE:
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_REGION_REQUIRES_ONBOARDING",
            text=messages.phone_link_requires_onboarding(),
            now=now,
        )
    if not request.argument:
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_REGION_MISSING",
            text=messages.phone_region_missing(),
            presentation_hint="PHONE_REGION_REQUIRED",
            now=now,
        )
    try:
        region = validate_phone_region(request.argument)
    except InvalidPhoneRegion:
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_REGION_INVALID",
            text=messages.phone_region_invalid(),
            now=now,
        )
    user.phone_region = region
    pending = find_pending_phone_identity(db, user_id=user.id)
    if pending is None:
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_REGION_SAVED",
            text=messages.phone_region_saved(region),
            presentation_hint="PHONE_CONTACT_REQUEST",
            now=now,
        )
    decision = process_phone_link(
        db,
        user=user,
        phone_number=pending.external_id,
        default_region=region,
        pending=pending,
    )
    if decision.outcome == "INVALID":
        return _store_result(
            db,
            request=request,
            user_id=user.id,
            operation="PHONE_LINK_INVALID",
            text=messages.phone_number_invalid_retry(),
            presentation_hint="PHONE_CONTACT_REQUEST",
            now=now,
        )
    return _phone_link_response(
        db, request=request, user=user, decision=decision, now=now
    )


def handle_onboarding_request(
    db: Session,
    request: OnboardingRequest,
    *,
    default_region: str | None = None,
    now: datetime | None = None,
) -> OutboundResponse | None:
    """Process one normalized onboarding request.

    ``None`` means the canonical user is already active and the request should
    continue through the shared conversation service.
    """

    handled_at = now or datetime.now(UTC)
    identity = resolve_linked_identity(
        db,
        channel_type=request.identity.channel_type,
        external_id=request.identity.external_id,
    )
    if request.action in {OnboardingAction.MESSAGE, OnboardingAction.VOICE}:
        if identity is not None and identity.user.onboarding_status == ACTIVE:
            db.rollback()
            return None

    db.rollback()
    existing = _existing_record(db, request)
    if existing is not None:
        db.rollback()
        return _reply_from_record(existing)

    db.rollback()
    with db.begin():
        if request.action is OnboardingAction.START:
            return _handle_start(db, request=request, now=handled_at)

        if identity is None:
            return _unknown_sender_response(db, request=request, now=handled_at)

        user = identity.user
        if request.action is OnboardingAction.CONTACT:
            return _handle_contact(
                db,
                request=request,
                user=user,
                default_region=default_region,
                now=handled_at,
            )
        if request.action is OnboardingAction.LINK_PHONE:
            if user.onboarding_status != ACTIVE:
                return _store_result(
                    db,
                    request=request,
                    user_id=user.id,
                    operation="PHONE_LINK_REQUIRES_ONBOARDING",
                    text=messages.phone_link_requires_onboarding(),
                    now=handled_at,
                )
            if find_phone_identity(db, user_id=user.id) is not None:
                return _store_result(
                    db,
                    request=request,
                    user_id=user.id,
                    operation="PHONE_LINK_ALREADY_LINKED",
                    text=messages.phone_already_linked(),
                    presentation_hint="REMOVE_KEYBOARD",
                    now=handled_at,
                )
            return _store_result(
                db,
                request=request,
                user_id=user.id,
                operation="PHONE_LINK_PROMPT",
                text=messages.phone_link_prompt(),
                presentation_hint="PHONE_CONTACT_REQUEST",
                now=handled_at,
            )
        if request.action is OnboardingAction.SKIP_PHONE:
            if user.onboarding_status != ACTIVE:
                return _store_result(
                    db,
                    request=request,
                    user_id=user.id,
                    operation="PHONE_LINK_SKIPPED",
                    text=messages.phone_link_requires_onboarding(),
                    now=handled_at,
                )
            phone = find_phone_identity(db, user_id=user.id)
            text = (
                messages.phone_remains_linked()
                if phone is not None
                else messages.phone_link_skipped()
            )
            return _store_result(
                db,
                request=request,
                user_id=user.id,
                operation="PHONE_LINK_SKIPPED",
                text=text,
                presentation_hint="PHONE_LINK_SKIPPED",
                now=handled_at,
            )
        if request.action is OnboardingAction.PHONE_REGION:
            return _handle_phone_region(
                db, request=request, user=user, now=handled_at
            )
        if request.action is OnboardingAction.MESSAGE:
            try:
                timezone_name = validate_timezone(request.text.strip())
            except InvalidTimezone:
                return _store_result(
                    db,
                    request=request,
                    user_id=user.id,
                    operation="ONBOARDING_INVALID_TIMEZONE",
                    text=messages.invalid_timezone(),
                    now=handled_at,
                )
            user.timezone = timezone_name
            user.onboarding_status = ACTIVE
            return _store_result(
                db,
                request=request,
                user_id=user.id,
                operation="ONBOARDING_COMPLETE",
                text=messages.onboarding_complete(user.name, timezone_name),
                presentation_hint="PHONE_CONTACT_REQUEST",
                now=handled_at,
            )
        if request.action is OnboardingAction.VOICE:
            return _store_result(
                db,
                request=request,
                user_id=user.id,
                operation="VOICE_NOTE_REQUIRES_ONBOARDING",
                text=messages.requires_timezone_for_phone(),
                now=handled_at,
            )

        raise ValueError(f"unsupported onboarding action: {request.action}")


__all__ = ["handle_onboarding_request"]
