from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones

import phonenumbers
from phonenumbers import NumberParseException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.db.models import ChannelIdentity, IdempotencyRecord, User
from reminder_agent.completion import complete_occurrence, parse_callback_data
from reminder_agent.observability import get_logger, log_event
from reminder_agent.telegram import TelegramAPIError, TelegramClient, TelegramUpdate
from reminder_agent.transcription import (
    AudioTranscriber,
    TranscriptionError,
    is_ogg_voice_note,
    log_transcription_failure,
)

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
    if update.message is None or update.message.voice is None:
        canonical = update.model_dump_json(by_alias=True, exclude_none=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload = update.model_dump(by_alias=True, exclude_none=True)
    # A transient STT overlay must not change the digest of the original voice
    # update. This keeps retries idempotent without persisting the transcript.
    if update.message is not None and update.message.voice is not None:
        payload.get("message", {}).pop("text", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
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
    source_message = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    assert source_message is not None
    reply = OnboardingReply(
        chat_id=source_message.chat.id,
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
    if parts and parts[0].lower() in {"/phone_region", "phone region"}:
        return parts[1] if len(parts) == 2 else ""
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


def _find_pending_phone_identity(db: Session, *, user_id) -> ChannelIdentity | None:
    return db.scalar(
        select(ChannelIdentity)
        .where(
            ChannelIdentity.user_id == user_id,
            ChannelIdentity.channel_type == PHONE,
            ChannelIdentity.status == PENDING,
        )
        .order_by(ChannelIdentity.created_at.desc())
    )


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


def _handle_message(
    db: Session,
    update: TelegramUpdate,
    *,
    reminder_interpreter=None,
    intent_router=None,
    memory_answerer=None,
    embedding_provider=None,
    embedding_privacy_policy=None,
    memory_retrieval_policy=None,
    now: datetime | None = None,
    session_ttl_seconds: int = 900,
) -> OnboardingReply:
    assert update.message is not None
    # Keep model calls outside a database transaction. Onboarding messages still use
    # the existing transactional path below; active users are handed to the shared
    # reminder service after the read transaction is closed.
    db.rollback()
    existing = _existing_record(db, update)
    if existing is not None:
        db.rollback()
        return _reply_from_record(existing)

    identity = _find_identity(db, update)
    if identity is None:
        db.rollback()
        with db.begin():
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
        db.rollback()
        from reminder_agent.intent import ReminderIntentInterpreter
        from reminder_agent.reminders import handle_reminder_text

        return handle_reminder_text(
            db,
            update,
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=update.message.from_user.id,
            interpreter=reminder_interpreter or ReminderIntentInterpreter(),
            intent_router=intent_router,
            memory_answerer=memory_answerer,
            embedding_provider=embedding_provider,
            embedding_privacy_policy=embedding_privacy_policy,
            memory_retrieval_policy=memory_retrieval_policy,
            now=now or datetime.now(timezone.utc),
            session_ttl_seconds=session_ttl_seconds,
        )

    db.rollback()
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
        text = (
            f"You're all set, {user.name}! Your time zone is {timezone_name}. "
            "To enable voice-call access, share your own phone number using the button below."
        )
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
            text=(
                "To enable voice-call access, share your own phone number using the button, "
                "or choose Skip for now."
            ),
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
        if not raw_region:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_REGION_MISSING",
                text=(
                    "Please include your two-letter country code, for example "
                    "/phone_region PK."
                ),
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
        pending = _find_pending_phone_identity(db, user_id=user.id)
        if pending is None:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_REGION_SAVED",
                text=f"Saved phone country {region}. Please tap Share my phone number.",
                reply_markup=PHONE_KEYBOARD,
            )
        try:
            normalized = normalize_phone_number(pending.external_id, default_region=region)
        except InvalidPhoneNumber:
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="PHONE_LINK_INVALID",
                text=(
                    f"I couldn't validate the phone number with country {region}. "
                    "Try another country code or share your phone number again."
                ),
                reply_markup=PHONE_KEYBOARD,
            )
        return _link_verified_phone(
            db,
            update=update,
            user=user,
            normalized=normalized,
            pending=pending,
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
                        external_id=contact.phone_number,
                        status=PENDING,
                    )
                )
                text = (
                    "Telegram sent a local-format number. Reply with /phone_region followed "
                    "by your two-letter country code (for example /phone_region PK)."
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

        return _link_verified_phone(db, update=update, user=user, normalized=normalized)


def _link_verified_phone(
    db: Session,
    *,
    update: TelegramUpdate,
    user: User,
    normalized: str,
    pending: ChannelIdentity | None = None,
) -> OnboardingReply:
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
            if pending is not None:
                pending.external_id = normalized
                pending.status = LINKED
                pending.verified_at = datetime.now(timezone.utc)
            else:
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
    reminder_interpreter=None,
    intent_router=None,
    memory_answerer=None,
    embedding_provider=None,
    embedding_privacy_policy=None,
    memory_retrieval_policy=None,
    now: datetime | None = None,
    session_ttl_seconds: int = 900,
    callback_secret: str = "development",
    telegram_client: TelegramClient | None = None,
    transcriber: AudioTranscriber | None = None,
    voice_max_bytes: int = 20 * 1024 * 1024,
    voice_max_duration_seconds: int = 300,
    log_voice_transcript: bool = False,
) -> OnboardingReply | None:
    """Apply one Telegram update and return the committed response, if any."""
    if update.callback_query is not None and update.callback_query.message is not None:
        callback = update.callback_query
        message = callback.message
        log_event(
            logger,
            logging.INFO,
            "completion_callback_received",
            update_id=update.update_id,
            callback_query_id=callback.id,
            telegram_user_id=callback.from_user.id,
            chat_id=message.chat.id,
            has_data=bool(callback.data),
        )
        if message.chat.type != "private" or not callback.data:
            log_event(
                logger,
                logging.INFO,
                "completion_callback_rejected",
                update_id=update.update_id,
                reason=("non_private_chat" if message.chat.type != "private" else "missing_data"),
            )
            return None
        if not callback.data.startswith("done:"):
            log_event(
                logger,
                logging.INFO,
                "completion_callback_rejected",
                update_id=update.update_id,
                reason="unsupported_action",
            )
            return None
        occurrence_id = parse_callback_data(callback.data, callback_secret)
        if occurrence_id is None:
            log_event(
                logger,
                logging.WARNING,
                "completion_callback_rejected",
                update_id=update.update_id,
                reason="invalid_signature_or_occurrence_id",
            )
            return None
        existing = _existing_record(db, update)
        if existing is not None:
            db.rollback()
            log_event(
                logger,
                logging.INFO,
                "completion_callback_duplicate",
                update_id=update.update_id,
                occurrence_id=str(occurrence_id),
            )
            return _reply_from_record(existing)
        identity = db.scalar(
            select(ChannelIdentity)
            .join(User, User.id == ChannelIdentity.user_id)
            .where(
                User.onboarding_status == ACTIVE,
                User.timezone.is_not(None),
                ChannelIdentity.channel_type == TELEGRAM,
                ChannelIdentity.external_id == str(callback.from_user.id),
                ChannelIdentity.status == LINKED,
                ChannelIdentity.telegram_chat_id == message.chat.id,
            )
        )
        if identity is None:
            log_event(
                logger,
                logging.WARNING,
                "completion_callback_rejected",
                update_id=update.update_id,
                occurrence_id=str(occurrence_id),
                reason="telegram_identity_not_authorized",
            )
            return None
        log_event(
            logger,
            logging.DEBUG,
            "completion_callback_authorized",
            update_id=update.update_id,
            occurrence_id=str(occurrence_id),
            user_id=str(identity.user_id),
            identity_id=str(identity.id),
        )
        result = complete_occurrence(
            db, user_id=identity.user_id, identity_id=identity.id,
            chat_id=message.chat.id,
            occurrence_id=occurrence_id,
            provider_message_id=str(message.message_id),
        )
        log_event(
            logger,
            logging.INFO,
            "completion_callback_service_result",
            update_id=update.update_id,
            occurrence_id=str(result.occurrence_id) if result.occurrence_id else str(occurrence_id),
            outcome=result.outcome,
        )
        reply = _store_result(
            db, update=update, user_id=identity.user_id,
            operation="COMPLETE_REMINDER", text=result.text,
        )
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            duplicate = _existing_record(db, update)
            if duplicate is not None:
                log_event(
                    logger,
                    logging.INFO,
                    "completion_callback_duplicate",
                    update_id=update.update_id,
                    occurrence_id=str(occurrence_id),
                    reason="idempotency_race",
                )
                return _reply_from_record(duplicate)
            raise
        log_event(
            logger,
            logging.INFO,
            "completion_callback_committed",
            update_id=update.update_id,
            occurrence_id=str(result.occurrence_id) if result.occurrence_id else str(occurrence_id),
            outcome=result.outcome,
        )
        log_event(
            logger,
            logging.INFO,
            "reminder_occurrence_completed",
            update_id=update.update_id,
            occurrence_id=str(result.occurrence_id) if result.occurrence_id else None,
            outcome=result.outcome,
            resolution="callback",
        )
        return reply
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
    if update.message.voice is not None:
        return _handle_voice_message(
            db,
            update,
            telegram_client=telegram_client,
            transcriber=transcriber,
            max_bytes=voice_max_bytes,
            max_duration_seconds=voice_max_duration_seconds,
            log_transcript=log_voice_transcript,
            reminder_interpreter=reminder_interpreter,
            intent_router=intent_router,
            memory_answerer=memory_answerer,
            embedding_provider=embedding_provider,
            embedding_privacy_policy=embedding_privacy_policy,
            memory_retrieval_policy=memory_retrieval_policy,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
        )
    return _handle_message(
        db,
        update,
        reminder_interpreter=reminder_interpreter,
        intent_router=intent_router,
        memory_answerer=memory_answerer,
        embedding_provider=embedding_provider,
        embedding_privacy_policy=embedding_privacy_policy,
        memory_retrieval_policy=memory_retrieval_policy,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
    )


def _handle_voice_message(
    db: Session,
    update: TelegramUpdate,
    *,
    telegram_client: TelegramClient | None,
    transcriber: AudioTranscriber | None,
    max_bytes: int,
    max_duration_seconds: int,
    log_transcript: bool = False,
    reminder_interpreter=None,
    intent_router=None,
    memory_answerer=None,
    embedding_provider=None,
    embedding_privacy_policy=None,
    memory_retrieval_policy=None,
    now: datetime | None = None,
    session_ttl_seconds: int = 900,
) -> OnboardingReply:
    """Download and transcribe one authenticated voice note, then reuse text handling."""
    assert update.message is not None and update.message.voice is not None
    db.rollback()
    log_event(
        logger,
        logging.INFO,
        "telegram_voice_processing_started",
        update_id=update.update_id,
        telegram_user_id=update.message.from_user.id,
        chat_id=update.message.chat.id,
        duration_seconds=update.message.voice.duration,
        declared_file_size=update.message.voice.file_size,
        mime_type=update.message.voice.mime_type,
    )
    existing = _existing_record(db, update)
    if existing is not None:
        log_event(logger, logging.INFO, "telegram_voice_replay_skipped", update_id=update.update_id)
        return _reply_from_record(existing)
    identity = _find_identity(db, update)
    if identity is None:
        log_event(logger, logging.WARNING, "telegram_voice_rejected", update_id=update.update_id, reason="unknown_sender")
        with db.begin():
            return _store_result(
                db,
                update=update,
                user_id=None,
                operation="VOICE_NOTE_UNKNOWN_SENDER",
                text="Please send /start to begin onboarding.",
            )
    user = db.get(User, identity.user_id)
    assert user is not None
    db.rollback()
    if user.onboarding_status != ACTIVE:
        log_event(logger, logging.INFO, "telegram_voice_rejected", update_id=update.update_id, reason="onboarding_incomplete")
        with db.begin():
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="VOICE_NOTE_REQUIRES_ONBOARDING",
                text="Please finish Telegram onboarding with your IANA time zone first.",
            )

    voice = update.message.voice
    if voice.duration > max_duration_seconds or (
        voice.file_size is not None and voice.file_size > max_bytes
    ):
        log_event(logger, logging.WARNING, "telegram_voice_rejected", update_id=update.update_id, reason="size_or_duration_limit")
        with db.begin():
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="VOICE_NOTE_REJECTED",
                text="That voice note is too large or too long. Please send a shorter note.",
            )
    if voice.mime_type is not None and voice.mime_type not in {"audio/ogg", "audio/opus"}:
        log_event(logger, logging.WARNING, "telegram_voice_rejected", update_id=update.update_id, reason="unsupported_mime_type")
        with db.begin():
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="VOICE_NOTE_REJECTED",
                text="Please send a Telegram voice note, not another audio format.",
            )
    if telegram_client is None or transcriber is None:
        log_event(logger, logging.ERROR, "telegram_voice_processing_failed", update_id=update.update_id, failure_kind="dependency_unavailable")
        return _store_voice_failure(
            db, update, user.id, "Voice notes are temporarily unavailable. Please try text instead."
        )
    try:
        log_event(logger, logging.INFO, "telegram_voice_download_started", update_id=update.update_id)
        file_path = telegram_client.get_file_path(voice.file_id)
        audio = telegram_client.download_file(file_path, max_bytes=max_bytes)
        log_event(logger, logging.INFO, "telegram_voice_download_succeeded", update_id=update.update_id, downloaded_bytes=len(audio))
        if not is_ogg_voice_note(audio):
            raise TelegramAPIError("Voice note format is invalid", failure_kind="invalid_media")
        log_event(logger, logging.INFO, "telegram_voice_transcription_started", update_id=update.update_id, audio_bytes=len(audio))
        result = transcriber.transcribe(
            audio,
            filename="voice-note.ogg",
            media_type=voice.mime_type or "audio/ogg",
        )
    except (TelegramAPIError, TranscriptionError, ValueError) as exc:
        if isinstance(exc, TranscriptionError):
            log_transcription_failure(update_id=update.update_id, error=exc)
        else:
            log_event(
                logger,
                logging.WARNING,
                "telegram_voice_rejected",
                update_id=update.update_id,
                failure_kind=getattr(exc, "failure_kind", "invalid_media"),
            )
        return _store_voice_failure(
            db,
            update,
            user.id,
            "I couldn't understand that voice note. Please try again or send the request as text.",
        )
    except Exception:
        log_event(
            logger,
            logging.ERROR,
            "telegram_voice_processing_failed",
            update_id=update.update_id,
            failure_kind="provider_error",
            exc_info=True,
        )
        return _store_voice_failure(
            db,
            update,
            user.id,
            "I couldn't process that voice note. Please try again or send the request as text.",
        )

    # Keep the original voice metadata and update digest while exposing the
    # transient transcript to the already-tested text command pipeline.
    command_update = update.model_copy(
        deep=True,
        update={
            "message": update.message.model_copy(update={"text": result.text}),
        },
    )
    log_event(
        logger,
        logging.INFO,
        "telegram_voice_transcription_succeeded",
        update_id=update.update_id,
        transcript_length=len(result.text),
        transcript=result.text if log_transcript else None,
    )
    log_event(logger, logging.INFO, "telegram_voice_command_handoff", update_id=update.update_id)
    return _handle_message(
        db,
        command_update,
        reminder_interpreter=reminder_interpreter,
        intent_router=intent_router,
        memory_answerer=memory_answerer,
        embedding_provider=embedding_provider,
        embedding_privacy_policy=embedding_privacy_policy,
        memory_retrieval_policy=memory_retrieval_policy,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
    )


def _store_voice_failure(
    db: Session, update: TelegramUpdate, user_id, text: str
) -> OnboardingReply:
    db.rollback()
    with db.begin():
        return _store_result(
            db,
            update=update,
            user_id=user_id,
            operation="VOICE_NOTE_FAILED",
            text=text,
        )


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
        record.delivery_attempt_count += 1
        record.delivery_claimed_by = f"webhook:{idempotency_key}"
        record.delivery_claimed_until = datetime.now(timezone.utc) + timedelta(minutes=2)
        return True


def record_delivery_outcome(
    db: Session,
    idempotency_key: str,
    *,
    sent: bool,
    error_kind: str | None = None,
    status_code: int | None = None,
    next_attempt_at: datetime | None = None,
) -> None:
    with db.begin():
        record = db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.source == TELEGRAM,
                IdempotencyRecord.idempotency_key == idempotency_key,
            )
        )
        if record is not None:
            permanent_failure = status_code in {400, 401, 403, 404}
            record.delivery_status = "SENT" if sent else (
                "FAILED" if permanent_failure else "UNKNOWN"
            )
            record.delivery_claimed_by = None
            record.delivery_claimed_until = None
            record.delivery_last_error_kind = None if sent else error_kind
            record.delivery_next_attempt_at = (
                None if sent or permanent_failure else next_attempt_at
            )
