from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones

from sqlalchemy import select
from sqlalchemy.orm import Session

from reminder_agent.db.models import ChannelIdentity, IdempotencyRecord, User
from reminder_agent.telegram import TelegramUpdate

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
PENDING = "PENDING"
ACTIVE = "ACTIVE"


class InvalidTimezone(ValueError):
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


def validate_timezone(value: str) -> str:
    candidate = value.strip()
    if not candidate or candidate not in available_timezones():
        raise InvalidTimezone(candidate)
    try:
        ZoneInfo(candidate)
    except Exception as exc:  # pragma: no cover - defensive guard for platform tzdata
        raise InvalidTimezone(candidate) from exc
    return candidate


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
    )


def _store_result(
    db: Session,
    *,
    update: TelegramUpdate,
    user_id,
    operation: str,
    text: str,
) -> OnboardingReply:
    assert update.message is not None
    reply = OnboardingReply(
        chat_id=update.message.chat.id,
        text=text,
        idempotency_key=str(update.update_id),
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
            response={"chat_id": reply.chat_id, "text": reply.text},
            created_at=now,
            updated_at=now,
        )
    )
    return reply


def _is_start(update: TelegramUpdate) -> bool:
    assert update.message is not None
    text = (update.message.text or "").strip()
    return text == "/start" or text.startswith("/start ")


def _find_identity(db: Session, update: TelegramUpdate) -> ChannelIdentity | None:
    assert update.message is not None
    return db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.channel_type == TELEGRAM,
            ChannelIdentity.external_id == str(update.message.from_user.id),
            ChannelIdentity.status == LINKED,
        )
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
            text = f"You're already onboarded, {user.name}."
        else:
            text = (
                f"Welcome back, {user.name}! Please reply with your IANA time zone, "
                "for example Asia/Karachi."
            )
        return _store_result(
            db, update=update, user_id=user.id, operation="ONBOARDING_START", text=text
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
            db, update=update, user_id=user.id, operation="ONBOARDING_COMPLETE", text=text
        )


def handle_update(db: Session, update: TelegramUpdate) -> OnboardingReply | None:
    """Apply one Telegram update and return the committed response, if any."""
    if update.message is None or update.message.chat.type != "private":
        return None
    if _is_start(update):
        return _handle_start(db, update)
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
