from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones

import phonenumbers
from phonenumbers import NumberParseException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.db.models import ChannelIdentity, IdempotencyRecord, User
from reminder_agent.observability import get_logger, log_event
from reminder_agent.telegram import TelegramUpdate

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
PENDING = "PENDING"
ACTIVE = "ACTIVE"
PHONE = "PHONE"
UNLINKED = "UNLINKED"

PHONE_KEYBOARD = {
    "keyboard": [
        [{"text": "Share my phone number", "request_contact": True}],
        [{"text": "Skip for now"}],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": True,
}
REMOVE_KEYBOARD = {"remove_keyboard": True}
logger = get_logger()


class InvalidTimezone(ValueError):
    pass


class InvalidPhoneNumber(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class InvalidPhoneRegion(ValueError):
    pass


class IdempotencyConflict(ValueError):
    pass


@dataclass(frozen=True)
class OnboardingReply:
    chat_id: int
    text: str
    idempotency_key: str | None = None
    should_send: bool = True
    is_duplicate: bool = False
    reply_markup: dict | None = None


def validate_timezone(value: str) -> str:
    candidate = value.strip()
    if not candidate or candidate not in available_timezones():
        raise InvalidTimezone(candidate)
    try:
        ZoneInfo(candidate)
    except Exception as exc:  # pragma: no cover - defensive guard for platform tzdata
        raise InvalidTimezone(candidate) from exc
    return candidate


def normalize_phone_number(value: str, *, default_region: str | None = None) -> str:
    """Return a valid phone number in E.164 form without retaining raw input."""
    candidate = value.strip()
    if not candidate:
        raise InvalidPhoneNumber("empty phone number")
    region = default_region.strip().upper() if default_region else None
    if (
        region is None
        and not candidate.startswith("+")
        and any(character.isdigit() for character in candidate)
    ):
        raise InvalidPhoneNumber("missing_region")
    try:
        parsed = phonenumbers.parse(candidate, region)
    except NumberParseException as exc:
        raise InvalidPhoneNumber("unparseable") from exc
    if not phonenumbers.is_possible_number(parsed) or not phonenumbers.is_valid_number(parsed):
        raise InvalidPhoneNumber("invalid")
    normalized = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    if not normalized.startswith("+"):
        raise InvalidPhoneNumber("phone number is not international")
    return normalized


def validate_phone_region(value: str) -> str:
    region = value.strip().upper()
    if region not in phonenumbers.SUPPORTED_REGIONS:
        raise InvalidPhoneRegion(region)
    return region


def _phone_shape(value: str) -> dict[str, int | bool]:
    stripped = value.strip()
    return {
        "phone_length": len(stripped),
        "phone_digit_count": sum(character.isdigit() for character in stripped),
        "phone_has_plus": stripped.startswith("+"),
    }


def _request_hash(update: TelegramUpdate) -> str:
    canonical = update.model_dump_json(by_alias=True, exclude_none=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _display_name(update: TelegramUpdate) -> str:
    assert update.message is not None
    sender = update.message.from_user
    name = " ".join(part for part in (sender.first_name, sender.last_name) if part).strip()
    return name or sender.username or "Telegram user"


def _existing_record(db: Session, update: TelegramUpdate) -> IdempotencyRecord | None:
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.idempotency_key == str(update.update_id),
        )
    )
    if record is not None and record.request_hash != _request_hash(update):
        raise IdempotencyConflict("Telegram update was reused with different content")
    return record


def _reply_from_record(record: IdempotencyRecord) -> OnboardingReply:
    assert record.response is not None
    return OnboardingReply(
        chat_id=int(record.response["chat_id"]),
        text=str(record.response["text"]),
        idempotency_key=record.idempotency_key,
        should_send=False,
        is_duplicate=True,
        reply_markup=record.response.get("reply_markup"),
    )


def _store_result(
    db: Session,
    *,
    update: TelegramUpdate,
    user_id,
    operation: str,
    text: str,
    reply_markup: dict | None = None,
) -> OnboardingReply:
    assert update.message is not None
    reply = OnboardingReply(
        chat_id=update.message.chat.id,
        text=text,
        idempotency_key=str(update.update_id),
        reply_markup=reply_markup,
    )
    now = datetime.now(timezone.utc)
    db.add(
        IdempotencyRecord(
            source=TELEGRAM,
            idempotency_key=str(update.update_id),
            user_id=user_id,
            operation=operation,
            request_hash=_request_hash(update),
            status="SUCCEEDED",
            delivery_status="PENDING",
            response={
                "chat_id": reply.chat_id,
                "text": reply.text,
                "reply_markup": reply.reply_markup,
            },
            created_at=now,
            updated_at=now,
        )
    )
    return reply


def _is_start(update: TelegramUpdate) -> bool:
    assert update.message is not None
    text = (update.message.text or "").strip()
    return text == "/start" or text.startswith("/start ")


def _is_link_phone(update: TelegramUpdate) -> bool:
    assert update.message is not None
    return (update.message.text or "").strip().lower() in {"/link_phone", "link phone"}


def _is_skip_phone(update: TelegramUpdate) -> bool:
    assert update.message is not None
    return (update.message.text or "").strip().lower() in {
        "skip",
        "skip for now",
        "/skip_phone",
    }


def _phone_region_from_text(update: TelegramUpdate) -> str | None:
    assert update.message is not None
    parts = (update.message.text or "").strip().split()
    if len(parts) == 2 and parts[0].lower() in {"/phone_region", "phone region"}:
        return parts[1]
    return None


def _find_identity(db: Session, update: TelegramUpdate) -> ChannelIdentity | None:
    assert update.message is not None
    return db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.channel_type == TELEGRAM,
            ChannelIdentity.external_id == str(update.message.from_user.id),
            ChannelIdentity.status == LINKED,
        )
    )


def _find_phone_identity(
    db: Session, *, user_id, external_id: str | None = None
) -> ChannelIdentity | None:
    filters = [
        ChannelIdentity.user_id == user_id,
        ChannelIdentity.channel_type == PHONE,
        ChannelIdentity.status == LINKED,
    ]
    if external_id is not None:
        filters.append(ChannelIdentity.external_id == external_id)
    return db.scalar(select(ChannelIdentity).where(*filters))


def _handle_start(db: Session, update: TelegramUpdate) -> OnboardingReply:
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
                verified_at=datetime.now(timezone.utc),
                telegram_chat_id=update.message.chat.id,
            )
            db.add(identity)
            text = (
                f"Welcome, {user.name}! Reply with your time zone using an IANA identifier, "
                "for example Asia/Karachi."
            )
            return _store_result(
                db, update=update, user_id=user.id, operation="ONBOARDING_START", text=text
            )

        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status == ACTIVE:
            phone = _find_phone_identity(db, user_id=user.id)
            if phone is None:
                text = f"You're already onboarded, {user.name}. You can link your phone when you're ready."
                markup = PHONE_KEYBOARD
            else:
                text = f"You're already onboarded, {user.name}."
                markup = REMOVE_KEYBOARD
        else:
            text = (
                f"Welcome back, {user.name}! Please reply with your IANA time zone, "
                "for example Asia/Karachi."
            )
            markup = None
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="ONBOARDING_START",
            text=text,
            reply_markup=markup,
        )


def _handle_message(db: Session, update: TelegramUpdate) -> OnboardingReply:
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
                text="Please send /start to begin onboarding.",
            )

        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status == ACTIVE:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="ONBOARDING_ALREADY_ACTIVE",
                text=f"You're already onboarded, {user.name}.",
            )

        raw_timezone = (update.message.text or "").strip()
        try:
            timezone_name = validate_timezone(raw_timezone)
        except InvalidTimezone:
            text = (
                "I don't recognize that time zone. Reply with an IANA identifier, "
                "for example Asia/Karachi."
            )
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="ONBOARDING_INVALID_TIMEZONE",
                text=text,
            )

        user.timezone = timezone_name
        user.onboarding_status = ACTIVE
        text = f"You're all set, {user.name}! Your time zone is {timezone_name}."
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="ONBOARDING_COMPLETE",
            text=text,
            reply_markup=PHONE_KEYBOARD,
        )


def _handle_link_phone(db: Session, update: TelegramUpdate) -> OnboardingReply:
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
                text="Please send /start to begin onboarding.",
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
                text="Please finish Telegram onboarding by sharing your time zone first.",
            )
        if phone is not None:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_ALREADY_LINKED",
                text="Your phone number is already linked.",
                reply_markup=REMOVE_KEYBOARD,
            )
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_LINK_PROMPT",
            text="Please use the button to share your own phone number, or choose Skip for now.",
            reply_markup=PHONE_KEYBOARD,
        )


def _handle_skip_phone(db: Session, update: TelegramUpdate) -> OnboardingReply:
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
                text="Please send /start to begin onboarding.",
            )
        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status != ACTIVE:
            text = "Please finish Telegram onboarding by sharing your time zone first."
            markup = None
        elif _find_phone_identity(db, user_id=user.id) is not None:
            text = "Your phone number remains linked."
            markup = REMOVE_KEYBOARD
        else:
            text = "No problem. Telegram will continue to work, and you can link your phone later."
            markup = REMOVE_KEYBOARD
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_LINK_SKIPPED",
            text=text,
            reply_markup=markup,
        )


def _handle_phone_region(db: Session, update: TelegramUpdate, raw_region: str) -> OnboardingReply:
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
                text="Please send /start to begin onboarding.",
            )
        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status != ACTIVE:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_REGION_REQUIRES_ONBOARDING",
                text="Please finish Telegram onboarding by sharing your time zone first.",
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
                text="Please use a valid two-letter country code, such as PK, GB, or US.",
            )
        user.phone_region = region
        log_event(
            logger,
            logging.INFO,
            "phone_region_saved",
            update_id=update.update_id,
            region=region,
        )
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_REGION_SAVED",
            text=f"Saved phone country {region}. Please tap Share my phone number again.",
            reply_markup=PHONE_KEYBOARD,
        )


def _handle_contact(
    db: Session,
    update: TelegramUpdate,
    *,
    default_region: str | None,
) -> OnboardingReply:
    assert update.message is not None
    assert update.message.contact is not None
    contact = update.message.contact
    log_event(
        logger,
        logging.INFO,
        "phone_contact_received",
        update_id=update.update_id,
        **_phone_shape(contact.phone_number),
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
                text="Please send /start to begin onboarding before linking a phone.",
            )

        user = db.get(User, identity.user_id)
        assert user is not None
        if user.onboarding_status != ACTIVE:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_REQUIRES_ONBOARDING",
                text="Please finish onboarding by sharing your time zone before linking a phone.",
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
                text="Please use Telegram's button to share your own phone number.",
                reply_markup=PHONE_KEYBOARD,
            )
        region = user.phone_region or default_region
        try:
            normalized = normalize_phone_number(
                contact.phone_number,
                default_region=region,
            )
        except InvalidPhoneNumber as exc:
            log_event(
                logger,
                logging.WARNING,
                "phone_link_rejected",
                update_id=update.update_id,
                reason=exc.reason,
                region_configured=region is not None,
            )
            if exc.reason == "missing_region":
                text = (
                    "Telegram sent a local-format number. Reply with /phone_region followed "
                    "by your two-letter country code (for example /phone_region PK), then tap "
                    "Share my phone number again."
                )
            else:
                text = "I couldn't validate that phone number. Please try sharing it again."
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_INVALID",
                text=text,
                reply_markup=PHONE_KEYBOARD,
            )

        other_owner = db.scalar(
            select(ChannelIdentity).where(
                ChannelIdentity.channel_type == PHONE,
                ChannelIdentity.external_id == normalized,
                ChannelIdentity.status == LINKED,
            )
        )
        if other_owner is not None and other_owner.user_id != user.id:
            log_event(
                logger,
                logging.WARNING,
                "phone_link_rejected",
                update_id=update.update_id,
                reason="active_number_conflict",
            )
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_CONFLICT",
                text="That phone number is already linked to another account.",
                reply_markup=PHONE_KEYBOARD,
            )

        current = _find_phone_identity(db, user_id=user.id)
        if current is not None and current.external_id == normalized:
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
                text="Your phone number is already linked.",
                reply_markup=REMOVE_KEYBOARD,
            )

        try:
            with db.begin_nested():
                if current is not None:
                    current.status = UNLINKED
                db.add(
                    ChannelIdentity(
                        user_id=user.id,
                        channel_type=PHONE,
                        external_id=normalized,
                        status=LINKED,
                        verified_at=datetime.now(timezone.utc),
                    )
                )
                db.flush()
        except IntegrityError:
            log_event(
                logger,
                logging.WARNING,
                "phone_link_rejected",
                update_id=update.update_id,
                reason="database_uniqueness_conflict",
            )
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_CONFLICT",
                text="That phone number is already linked to another account.",
                reply_markup=PHONE_KEYBOARD,
            )

        log_event(
            logger,
            logging.INFO,
            "phone_link_succeeded",
            update_id=update.update_id,
            replaced_existing=current is not None,
        )
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="PHONE_LINKED",
            text="Your phone number is now linked. You can call the assistant using this number.",
            reply_markup=REMOVE_KEYBOARD,
        )


def handle_update(
    db: Session,
    update: TelegramUpdate,
    *,
    default_region: str | None = None,
) -> OnboardingReply | None:
    """Apply one Telegram update and return the committed response, if any."""
    if update.message is None or update.message.chat.type != "private":
        return None
    if update.message.contact is not None:
        return _handle_contact(db, update, default_region=default_region)
    if _is_start(update):
        return _handle_start(db, update)
    if _is_link_phone(update):
        return _handle_link_phone(db, update)
    if _is_skip_phone(update):
        return _handle_skip_phone(db, update)
    phone_region = _phone_region_from_text(update)
    if phone_region is not None:
        return _handle_phone_region(db, update, phone_region)
    return _handle_message(db, update)


def claim_delivery(db: Session, idempotency_key: str) -> bool:
    """Claim one outbound response so webhook retries cannot send duplicates."""
    with db.begin():
        record = db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.source == TELEGRAM,
                IdempotencyRecord.idempotency_key == idempotency_key,
            )
        )
        if record is None or record.delivery_status != "PENDING":
            return False
        record.delivery_status = "ATTEMPTED"
        record.delivery_attempted_at = datetime.now(timezone.utc)
        return True


def record_delivery_outcome(db: Session, idempotency_key: str, *, sent: bool) -> None:
    with db.begin():
        record = db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.source == TELEGRAM,
                IdempotencyRecord.idempotency_key == idempotency_key,
            )
        )
        if record is not None:
            record.delivery_status = "SENT" if sent else "UNKNOWN"
