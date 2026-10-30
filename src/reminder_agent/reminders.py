from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.db.models import (
    ChannelIdentity,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.intent import IntentInterpreter
from reminder_agent.observability import get_logger, log_event
from reminder_agent.telegram import TelegramUpdate

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
ONE_TIME = "ONE_TIME"
PENDING = "PENDING"
logger = get_logger()


@dataclass(frozen=True)
class ReminderReply:
    chat_id: int
    text: str
    idempotency_key: str | None = None
    should_send: bool = True
    is_duplicate: bool = False
    reply_markup: dict | None = None


def _request_hash(update: TelegramUpdate) -> str:
    canonical = update.model_dump_json(by_alias=True, exclude_none=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _existing_record(db: Session, update: TelegramUpdate) -> IdempotencyRecord | None:
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.idempotency_key == str(update.update_id),
        )
    )
    if record is not None and record.request_hash != _request_hash(update):
        log_event(
            logger,
            logging.WARNING,
            "reminder_idempotency_conflict",
            update_id=update.update_id,
        )
        from reminder_agent.onboarding import IdempotencyConflict

        raise IdempotencyConflict("Telegram update was reused with different content")
    return record


def _reply_from_record(record: IdempotencyRecord) -> ReminderReply:
    assert record.response is not None
    return ReminderReply(
        chat_id=int(record.response["chat_id"]),
        text=str(record.response["text"]),
        idempotency_key=record.idempotency_key,
        should_send=False,
        is_duplicate=True,
        reply_markup=record.response.get("reply_markup"),
    )


def _now_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _local_display(due_at: datetime, timezone_name: str) -> str:
    return due_at.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M %Z")


def _store_response(
    db: Session,
    *,
    update: TelegramUpdate,
    user_id: UUID,
    text: str,
    response_kind: str = "clarification",
) -> ReminderReply:
    assert update.message is not None
    reply = ReminderReply(
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
            operation="CREATE_REMINDER",
            request_hash=_request_hash(update),
            status="SUCCEEDED",
            delivery_status="PENDING",
            response={"chat_id": reply.chat_id, "text": reply.text, "reply_markup": None},
            created_at=now,
            updated_at=now,
        )
    )
    log_event(
        logger,
        logging.DEBUG,
        "reminder_response_staged",
        update_id=update.update_id,
        response_kind=response_kind,
    )
    return reply


def _recover_from_idempotency_race(
    db: Session, update: TelegramUpdate
) -> ReminderReply | None:
    """Load the committed winner after a concurrent idempotency insert conflict."""
    db.rollback()
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.idempotency_key == str(update.update_id),
        )
    )
    if record is None:
        db.rollback()
        return None
    if record.request_hash != _request_hash(update):
        log_event(
            logger,
            logging.WARNING,
            "reminder_idempotency_conflict",
            update_id=update.update_id,
            phase="race_recovery",
        )
        from reminder_agent.onboarding import IdempotencyConflict

        raise IdempotencyConflict("Telegram update was reused with different content")
    db.rollback()
    log_event(
        logger,
        logging.INFO,
        "reminder_command_duplicate",
        update_id=update.update_id,
        phase="race_recovery",
    )
    return _reply_from_record(record)


def handle_reminder_text(
    db: Session,
    update: TelegramUpdate,
    *,
    user_id: UUID,
    identity_id: UUID,
    telegram_user_id: int,
    interpreter: IntentInterpreter,
    now: datetime,
) -> ReminderReply:
    """Interpret and transactionally create one one-time reminder for a Telegram user."""
    assert update.message is not None
    log_event(
        logger,
        logging.INFO,
        "reminder_command_started",
        update_id=update.update_id,
        input_length=len((update.message.text or "").strip()),
    )

    # Resolve the trusted sender and duplicate state before invoking a model. This avoids
    # holding a database transaction open during an external request.
    existing = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.idempotency_key == str(update.update_id),
        )
    )
    if existing is not None:
        if existing.request_hash != _request_hash(update):
            log_event(
                logger,
                logging.WARNING,
                "reminder_idempotency_conflict",
                update_id=update.update_id,
            )
            from reminder_agent.onboarding import IdempotencyConflict

            raise IdempotencyConflict("Telegram update was reused with different content")
        db.rollback()
        log_event(
            logger,
            logging.INFO,
            "reminder_command_duplicate",
            update_id=update.update_id,
        )
        return _reply_from_record(existing)

    user = db.get(User, user_id)
    identity = db.get(ChannelIdentity, identity_id)
    if user is None or identity is None:
        db.rollback()
        log_event(
            logger,
            logging.WARNING,
            "reminder_identity_resolution_failed",
            update_id=update.update_id,
            reason="missing_user_or_identity",
        )
        return ReminderReply(update.message.chat.id, "I couldn't verify your account.")
    timezone_name = user.timezone
    if user.onboarding_status != ACTIVE or not timezone_name:
        db.rollback()
        log_event(
            logger,
            logging.WARNING,
            "reminder_command_rejected",
            update_id=update.update_id,
            reason="onboarding_incomplete",
        )
        return ReminderReply(
            update.message.chat.id,
            "Please finish onboarding with your IANA time zone before creating reminders.",
        )
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        db.rollback()
        log_event(
            logger,
            logging.ERROR,
            "reminder_command_rejected",
            update_id=update.update_id,
            reason="invalid_user_timezone",
        )
        return ReminderReply(
            update.message.chat.id,
            "Your saved time zone is invalid; please update it.",
        )

    interpretation = interpreter.interpret(
        update.message.text or "", timezone_name=timezone_name, now=now
    )
    log_event(
        logger,
        logging.INFO,
        "reminder_intent_interpreted",
        update_id=update.update_id,
        has_intent=interpretation.intent is not None,
        task_length=len(interpretation.intent.task) if interpretation.intent else None,
        fallback_allowed=interpretation.fallback_allowed,
    )
    db.rollback()

    try:
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                log_event(
                    logger,
                    logging.INFO,
                    "reminder_command_duplicate",
                    update_id=update.update_id,
                    phase="transaction_recheck",
                )
                return _reply_from_record(existing)

            user = db.get(User, user_id)
            identity = db.get(ChannelIdentity, identity_id)
            if (
                user is None
                or identity is None
                or user.onboarding_status != ACTIVE
                or not user.timezone
                or identity.user_id != user.id
                or identity.channel_type != TELEGRAM
                or identity.status != LINKED
                or identity.external_id != str(telegram_user_id)
                or identity.telegram_chat_id != update.message.chat.id
            ):
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_identity_authorization_failed",
                    update_id=update.update_id,
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text="I couldn't verify your Telegram account.",
                    response_kind="authorization_failure",
                )

            if interpretation.intent is None:
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=interpretation.response_text
                    or "Please tell me the reminder task and a future time.",
                    response_kind="clarification",
                )

            if interpretation.intent.timezone_name != user.timezone:
                log_event(
                    logger,
                    logging.ERROR,
                    "reminder_timezone_mismatch",
                    update_id=update.update_id,
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text="I couldn't verify the reminder time zone. Please try again.",
                    response_kind="timezone_validation_failure",
                )

            due_at = _now_utc(interpretation.intent.due_at)
            reference = _now_utc(now)
            if due_at <= reference:
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_schedule_rejected",
                    update_id=update.update_id,
                    reason="past_or_present_due_time",
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text="Please choose a future time for the reminder.",
                    response_kind="schedule_validation_failure",
                )
            try:
                local_display = _local_display(due_at, user.timezone)
            except ZoneInfoNotFoundError:
                log_event(
                    logger,
                    logging.ERROR,
                    "reminder_schedule_rejected",
                    update_id=update.update_id,
                    reason="invalid_user_timezone",
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text="Your saved time zone is invalid; please update it.",
                    response_kind="schedule_validation_failure",
                )

            reminder = Reminder(
                user_id=user.id,
                task=interpretation.intent.task,
                schedule_type=ONE_TIME,
                start_local=due_at.astimezone(ZoneInfo(user.timezone)).isoformat(),
                timezone=user.timezone,
                recurrence_rule=None,
                status=ACTIVE,
                version=1,
            )
            db.add(reminder)
            db.flush()
            occurrence = ReminderOccurrence(
                reminder_id=reminder.id,
                scheduled_at=due_at,
                next_due_at=due_at,
                status=PENDING,
            )
            db.add(occurrence)
            db.flush()
            db.add(
                Notification(
                    occurrence_id=occurrence.id,
                    destination_identity_id=identity.id,
                    sequence=1,
                    due_at=due_at,
                    status=PENDING,
                    attempt_count=0,
                )
            )
            text = f"I'll remind you to {reminder.task} on {local_display} via Telegram."
            reply = _store_response(
                db,
                update=update,
                user_id=user.id,
                text=text,
                response_kind="confirmation",
            )
    except IntegrityError as exc:
        duplicate = _recover_from_idempotency_race(db, update)
        if duplicate is not None:
            return duplicate
        log_event(
            logger,
            logging.ERROR,
            "reminder_persistence_failed",
            update_id=update.update_id,
            error_type=type(exc).__name__,
        )
        raise
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "reminder_persistence_failed",
            update_id=update.update_id,
            error_type=type(exc).__name__,
        )
        raise

    log_event(
        logger,
        logging.INFO,
        "reminder_created",
        update_id=update.update_id,
        task_length=len(interpretation.intent.task),
        schedule_type=ONE_TIME,
        notification_status=PENDING,
    )
    return reply
