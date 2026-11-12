"""Transactional onboarding branches behind the public workflow faÃ§ade.

This module owns the onboarding-specific decisions and persistence calls.  The
legacy update-shaped arguments are intentionally retained here for one
increment so replay hashes, response staging, and transaction boundaries stay
unchanged while callers migrate to :mod:`onboarding_requests`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from reminder_agent.application.identity.link import (
    find_pending_phone_identity,
    find_phone_identity,
)
from reminder_agent.features.onboarding import message_templates as messages
from reminder_agent.features.onboarding.onboarding_validation import (
    InvalidPhoneRegion,
    phone_shape,
    validate_phone_region,
)
from reminder_agent.features.onboarding.phone_linking_service import (
    PhoneLinkDecision,
    process_phone_link,
)
from reminder_agent.features.onboarding.request_contracts import is_phone_link_request
from reminder_agent.features.onboarding.responses import (
    OnboardingReply,
)
from reminder_agent.features.onboarding.responses import (
    existing_record as _existing_record,
)
from reminder_agent.features.onboarding.responses import (
    find_sender_identity as _find_identity,
)
from reminder_agent.features.onboarding.responses import (
    reply_from_record as _reply_from_record,
)
from reminder_agent.features.onboarding.responses import (
    store_result as _store_result,
)
from reminder_agent.infrastructure.database.models import ChannelIdentity, User
from reminder_agent.infrastructure.observability import get_logger, log_event

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
PENDING = "PENDING"
ACTIVE = "ACTIVE"
PHONE = "PHONE"
UNLINKED = "UNLINKED"

logger = get_logger()
_phone_shape = phone_shape


def _display_name(update: Any) -> str:
    assert update.message is not None
    sender = update.message.from_user
    name = " ".join(part for part in (sender.first_name, sender.last_name) if part).strip()
    return name or sender.username or "Telegram user"


def _is_start(update: Any) -> bool:
    assert update.message is not None
    text = (update.message.text or "").strip()
    return text == "/start" or text.startswith("/start ")


def _is_link_phone(update: Any) -> bool:
    assert update.message is not None
    text = (update.message.text or "").strip()
    return text.lower() in {"/link_phone", "link phone"} or is_phone_link_request(text)


def _is_skip_phone(update: Any) -> bool:
    assert update.message is not None
    return (update.message.text or "").strip().lower() in {
        "skip",
        "skip for now",
        "/skip_phone",
    }


def _phone_region_from_text(update: Any) -> str | None:
    assert update.message is not None
    parts = (update.message.text or "").strip().split()
    if parts and parts[0].lower() in {"/phone_region", "phone region"}:
        return parts[1] if len(parts) == 2 else ""
    return None


def _find_phone_identity(
    db: Session, *, user_id: UUID, external_id: str | None = None
) -> ChannelIdentity | None:
    return find_phone_identity(db, user_id=user_id, external_id=external_id)


def _find_pending_phone_identity(db: Session, *, user_id: UUID) -> ChannelIdentity | None:
    return find_pending_phone_identity(db, user_id=user_id)


def handle_start(db: Session, update: Any) -> OnboardingReply:
    assert update.message is not None
    with db.begin():
        existing = _existing_record(db, update)
        if existing is not None:
            return _reply_from_record(existing)

        identity = _find_identity(db, update)
        if identity is None:
            user = User(name=_display_name(update), onboarding_status=PENDING)
            db.add(user)
            db.flush()
            identity = ChannelIdentity(
                user_id=user.id,
                channel_type=TELEGRAM,
                external_id=str(update.message.from_user.id),
                status=LINKED,
                verified_at=datetime.now(UTC),
                telegram_chat_id=update.message.chat.id,
            )
            db.add(identity)
            text = messages.welcome(user.name)
            return _store_result(
                db, update=update, user_id=user.id, operation="ONBOARDING_START", text=text
            )

        existing_user = db.get(User, identity.user_id)
        assert existing_user is not None
        if existing_user.onboarding_status == ACTIVE:
            phone = _find_phone_identity(db, user_id=existing_user.id)
            if phone is None:
                text = messages.already_onboarded_without_phone(existing_user.name)
                presentation_hint: str | None = "PHONE_CONTACT_REQUEST"
            else:
                text = messages.already_onboarded(existing_user.name)
                presentation_hint = "REMOVE_KEYBOARD"
        else:
            text = messages.welcome_back(existing_user.name)
            presentation_hint = None
        return _store_result(
            db,
            update=update,
            user_id=existing_user.id,
            operation="ONBOARDING_START",
            text=text,
            presentation_hint=presentation_hint,
        )


def handle_link_phone(db: Session, update: Any) -> OnboardingReply:
    assert update.message is not None
    with db.begin():
        existing = _existing_record(db, update)
        if existing is not None:
            return _reply_from_record(existing)

        identity = _find_identity(db, update)
        if identity is None:
            return _store_result(
                db,
                update=update,
                user_id=None,
                operation="ONBOARDING_UNKNOWN_SENDER",
                text=messages.unknown_sender(),
            )

        user = db.get(User, identity.user_id)
        assert user is not None
        phone = _find_phone_identity(db, user_id=user.id)
        if user.onboarding_status != ACTIVE:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_REQUIRES_ONBOARDING",
                text=messages.requires_timezone_for_phone(),
            )
        if phone is not None:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_ALREADY_LINKED",
                text=messages.phone_already_linked(),
                presentation_hint="REMOVE_KEYBOARD",
            )
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_LINK_PROMPT",
            text=messages.phone_link_prompt(),
            presentation_hint="PHONE_CONTACT_REQUEST",
        )


def handle_skip_phone(db: Session, update: Any) -> OnboardingReply:
    assert update.message is not None
    with db.begin():
        existing = _existing_record(db, update)
        if existing is not None:
            return _reply_from_record(existing)
        identity = _find_identity(db, update)
        if identity is None:
            return _store_result(
                db,
                update=update,
                user_id=None,
                operation="ONBOARDING_UNKNOWN_SENDER",
                text=messages.unknown_sender(),
            )
        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status != ACTIVE:
            text = messages.requires_timezone_for_phone()
            presentation_hint = None
        elif _find_phone_identity(db, user_id=user.id) is not None:
            text = messages.phone_remains_linked()
            presentation_hint = "REMOVE_KEYBOARD"
        else:
            text = messages.phone_link_skipped()
            presentation_hint = "REMOVE_KEYBOARD"
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_LINK_SKIPPED",
            text=text,
            presentation_hint=presentation_hint,
        )


def handle_phone_region(db: Session, update: Any, raw_region: str) -> OnboardingReply:
    assert update.message is not None
    with db.begin():
        existing = _existing_record(db, update)
        if existing is not None:
            return _reply_from_record(existing)
        identity = _find_identity(db, update)
        if identity is None:
            return _store_result(
                db,
                update=update,
                user_id=None,
                operation="PHONE_REGION_UNKNOWN_SENDER",
                text=messages.unknown_sender(),
            )
        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status != ACTIVE:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_REGION_REQUIRES_ONBOARDING",
                text=messages.requires_timezone_for_phone(),
            )
        if not raw_region:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_REGION_MISSING",
                text=messages.phone_region_missing(),
            )
        try:
            region = validate_phone_region(raw_region)
        except InvalidPhoneRegion:
            log_event(
                logger,
                logging.WARNING,
                "phone_region_rejected",
                update_id=update.update_id,
                reason="unsupported_region",
            )
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_REGION_INVALID",
                text=messages.phone_region_invalid(),
            )
        user.phone_region = region
        log_event(
            logger,
            logging.INFO,
            "phone_region_saved",
            update_id=update.update_id,
            region=region,
        )
        pending = _find_pending_phone_identity(db, user_id=user.id)
        if pending is None:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_REGION_SAVED",
                text=messages.phone_region_saved(region),
                presentation_hint="PHONE_CONTACT_REQUEST",
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
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_INVALID",
                text=messages.phone_number_invalid(region),
                presentation_hint="PHONE_CONTACT_REQUEST",
            )
        return phone_link_response(db, update=update, user=user, decision=decision)


def handle_contact(
    db: Session,
    update: Any,
    *,
    default_region: str | None,
) -> OnboardingReply:
    assert update.message is not None
    assert update.message.contact is not None
    contact = update.message.contact
    shape = _phone_shape(contact.phone_number)
    log_event(
        logger,
        logging.INFO,
        "phone_contact_received",
        update_id=update.update_id,
        phone_length=shape["phone_length"],
        phone_digit_count=shape["phone_digit_count"],
        phone_has_plus=shape["phone_has_plus"],
        contact_owner_present=contact.user_id is not None,
        contact_owner_matches=contact.user_id == update.message.from_user.id,
    )
    with db.begin():
        existing = _existing_record(db, update)
        if existing is not None:
            return _reply_from_record(existing)

        identity = _find_identity(db, update)
        if identity is None:
            return _store_result(
                db,
                update=update,
                user_id=None,
                operation="PHONE_LINK_UNKNOWN_SENDER",
                text=messages.phone_link_unknown_sender(),
            )

        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status != ACTIVE:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_REQUIRES_ONBOARDING",
                text=messages.phone_link_requires_onboarding(),
            )
        if contact.user_id is None or contact.user_id != update.message.from_user.id:
            log_event(
                logger,
                logging.WARNING,
                "phone_link_rejected",
                update_id=update.update_id,
                reason="contact_owner_mismatch",
            )
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_OWNERSHIP_REJECTED",
                text=messages.phone_ownership_rejected(),
                presentation_hint="PHONE_CONTACT_REQUEST",
            )
        decision = process_phone_link(
            db,
            user=user,
            phone_number=contact.phone_number,
            default_region=default_region,
        )
        if decision.outcome not in {"LINKED", "CONFLICT", "ALREADY_LINKED"}:
            log_event(
                logger,
                logging.WARNING,
                "phone_link_rejected",
                update_id=update.update_id,
                reason=decision.reason,
                region_configured=decision.region is not None,
            )
            if decision.outcome == "MISSING_REGION":
                text = messages.local_phone_region_required()
            else:
                text = messages.phone_number_invalid_retry()
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_INVALID",
                text=text,
                presentation_hint="PHONE_CONTACT_REQUEST",
            )

        return phone_link_response(db, update=update, user=user, decision=decision)


def phone_link_response(
    db: Session,
    *,
    update: Any,
    user: User,
    decision: PhoneLinkDecision,
) -> OnboardingReply:
    if decision.outcome == "CONFLICT":
        log_event(
            logger,
            logging.WARNING,
            "phone_link_rejected",
            update_id=update.update_id,
            reason=decision.reason,
        )
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_LINK_CONFLICT",
            text=messages.phone_conflict(),
            presentation_hint="PHONE_CONTACT_REQUEST",
        )

    if decision.outcome == "ALREADY_LINKED":
        log_event(
            logger,
            logging.INFO,
            "phone_link_reused",
            update_id=update.update_id,
        )
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_LINK_ALREADY_LINKED",
            text=messages.phone_already_linked(),
            presentation_hint="REMOVE_KEYBOARD",
        )

    log_event(
        logger,
        logging.INFO,
        "phone_link_succeeded",
        update_id=update.update_id,
        replaced_existing=decision.replaced_existing,
    )
    return _store_result(
        db,
        update=update,
        user_id=user.id,
        operation="PHONE_LINKED",
        text=messages.phone_linked(),
        presentation_hint="REMOVE_KEYBOARD",
    )


__all__ = [
    "_is_link_phone",
    "_is_skip_phone",
    "_is_start",
    "_phone_region_from_text",
    "handle_contact",
    "handle_link_phone",
    "handle_phone_region",
    "handle_skip_phone",
    "handle_start",
    "phone_link_response",
]


