"""Channel-neutral pending snooze session and response coordination."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, TypeVar, cast
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.conversations import (
    end_pending_session,
    end_session,
    load_session,
    telegram_session_id,
    upsert_clarification_session,
)
from reminder_agent.application.pending_contracts import PendingSnooze
from reminder_agent.application.response_staging import (
    ReminderReply,
    _authorized_telegram_identity,
    _existing_record,
    _recover_from_idempotency_race,
    _reply_from_record,
    _store_response,
)
from reminder_agent.features.intent.contracts import IntentInterpreter, IntentRoute
from reminder_agent.features.reminders.complete.operations import (
    match_completion_candidates,
    resolve_by_occurrence,
    resolve_by_provider_message,
    resolve_latest_delivered,
)
from reminder_agent.features.reminders.snooze.messages import snooze_message
from reminder_agent.features.reminders.snooze.operations import (
    SnoozeDuration,
    is_snooze_command,
    parse_snooze_duration,
    snooze_occurrence,
    snooze_target_text,
)
from reminder_agent.infrastructure.db.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.infrastructure.observability import get_logger, log_event

ResponseT = TypeVar("ResponseT")
logger = get_logger()


def _local_display(due_at: datetime, timezone_name: str) -> str:
    return due_at.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M %Z")


def _end_pending_snooze(
    db: Session,
    update: Any,
    *,
    user: User,
    now: datetime,
    session_ttl_seconds: int,
    response_text: str | None = None,
) -> ReminderReply | None:
    """End a pending snooze under lock, optionally recording a response."""

    return end_pending_session(
        db,
        update,
        user=user,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        response_text=response_text,
    )

def _snooze_duration_from_interpreter(
    interpreter: IntentInterpreter,
    text: str,
    *,
    timezone_name: str,
    now: datetime,
    recent_messages: list[str],
) -> int | None:
    """Ask the configured extractor for duration only after rules fail."""

    method = getattr(interpreter, "interpret", None)
    if method is None:
        return None
    kwargs = {"timezone_name": timezone_name, "now": now}
    parameters = inspect.signature(method).parameters
    if "selected_route" in parameters:
        kwargs["selected_route"] = IntentRoute.SNOOZE_REQUEST
    if "recent_messages" in parameters:
        kwargs["recent_messages"] = recent_messages
    try:
        proposal = method(text, **kwargs)
    except Exception as exc:  # provider failures are clarification-only
        log_event(
            logger,
            logging.WARNING,
            "snooze_duration_fallback_failed",
            reason_type=type(exc).__name__,
        )
        return None
    value = getattr(proposal, "snooze_duration_seconds", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _distinct_delivered_candidates(
    candidates: list[tuple[ReminderOccurrence, Reminder, Notification]],
) -> list[tuple[ReminderOccurrence, Reminder, Notification]]:
    """Collapse historical deliveries to one latest row per occurrence."""

    selected: dict[UUID, tuple[ReminderOccurrence, Reminder, Notification]] = {}

    def rank(candidate: tuple[ReminderOccurrence, Reminder, Notification]) -> tuple[int, float]:
        notification = candidate[2]
        sent_at = notification.sent_at
        return (
            int(notification.sequence),
            sent_at.timestamp() if sent_at is not None else float("-inf"),
        )

    for candidate in candidates:
        occurrence, _reminder, notification = candidate
        previous = selected.get(occurrence.id)
        if previous is None or rank(candidate) > rank(previous):
            selected[occurrence.id] = candidate
    return list(selected.values())


def _handle_snooze_text(
    db: Session,
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity,
    interpreter: IntentInterpreter,
    now: datetime,
    pending: PendingSnooze | None = None,
    recent_messages: list[str] | None = None,
    session_ttl_seconds: int = 900,
    proposed_duration_seconds: int | None = None,
) -> ReminderReply:
    """Resolve and atomically snooze one delivered occurrence in a caller-owned block."""

    assert update.message is not None
    text = update.message.text or ""
    log_event(
        logger,
        logging.INFO,
        "snooze_request_received",
        update_id=update.update_id,
        user_id=str(user.id),
        identity_id=str(identity.id),
        chat_id=update.message.chat.id,
        has_pending_context=pending is not None,
        has_reply_target=update.message.reply_to_message is not None,
    )
    duration = parse_snooze_duration(text)
    if duration is None and proposed_duration_seconds is not None:
        try:
            duration = SnoozeDuration(duration_seconds=proposed_duration_seconds)
        except ValueError:
            duration = None
    if (
        duration is None
        and pending is not None
        and pending.duration_seconds is None
        and is_snooze_command(text)
    ):
        duration_seconds = _snooze_duration_from_interpreter(
            interpreter,
            text,
            timezone_name=user.timezone or "UTC",
            now=now,
            recent_messages=recent_messages or [],
        )
        duration = SnoozeDuration(duration_seconds=duration_seconds) if duration_seconds else None
    if duration is None and pending is not None and pending.duration_seconds is not None:
        duration = SnoozeDuration(duration_seconds=pending.duration_seconds)
    log_event(
        logger,
        logging.DEBUG,
        "snooze_duration_resolved",
        update_id=update.update_id,
        duration_seconds=duration.duration_seconds if duration is not None else None,
        source="deterministic"
        if parse_snooze_duration(text) is not None
        else "fallback_or_pending"
        if duration is not None
        else "unresolved",
    )

    provider_id = (
        str(update.message.reply_to_message.message_id)
        if update.message.reply_to_message is not None
        else None
    )
    if pending is not None:
        candidates: list[tuple[ReminderOccurrence, Reminder, Notification]] = []
        for occurrence_id in pending.candidate_occurrence_ids:
            candidates.extend(
                resolve_by_occurrence(
                    db,
                    user_id=user.id,
                    identity_id=identity.id,
                    chat_id=update.message.chat.id,
                    occurrence_id=occurrence_id,
                )
            )
    elif provider_id is not None:
        candidates = resolve_by_provider_message(
            db,
            user_id=user.id,
            identity_id=identity.id,
            chat_id=update.message.chat.id,
            provider_message_id=provider_id,
        )
    else:
        candidates = resolve_latest_delivered(
            db,
            user_id=user.id,
            identity_id=identity.id,
            chat_id=update.message.chat.id,
        )

    candidates = _distinct_delivered_candidates(candidates)
    log_event(
        logger,
        logging.DEBUG,
        "snooze_candidates_resolved",
        update_id=update.update_id,
        candidate_count=len(candidates),
        resolution="pending" if pending is not None else "reply" if provider_id else "latest",
    )

    target_text = snooze_target_text(text)
    if target_text is None and pending is not None and parse_snooze_duration(text) is None:
        target_text = text.strip() or None
    if provider_id is None and target_text and len(candidates) > 1:
        targeted = match_completion_candidates(target_text, candidates)
        if len(targeted) == 1:
            candidates = targeted

    if len(candidates) != 1 or duration is None:
        if not candidates:
            response_text = snooze_message("no_delivered_match")
        elif duration is None:
            response_text = snooze_message("missing_duration")
        else:
            response_text = snooze_message("missing_target")
        log_event(
            logger,
            logging.INFO,
            "snooze_clarification_required",
            update_id=update.update_id,
            candidate_count=len(candidates),
            missing_duration=duration is None,
        )
        pending_value = None
        if candidates:
            pending_value = PendingSnooze(
                candidate_occurrence_ids=tuple(item[0].id for item in candidates),
                duration_seconds=duration.duration_seconds if duration is not None else None,
            )
        return cast(
            ReminderReply,
            handle_snooze_clarification(
                db,
                user_id=user.id,
                channel_type=identity.channel_type,
                external_session_id=telegram_session_id(update.message.chat.id),
                pending=pending_value,
                response_text=response_text,
                response_kind="snooze_clarification" if pending_value else "snooze_response",
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                upsert_pending=lambda **kwargs: upsert_clarification_session(db, **kwargs),
                find_existing=lambda: _existing_record(db, update),
                stage_response=lambda **kwargs: _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    **kwargs,
                ),
                recover_from_race=lambda: _recover_from_idempotency_race(db, update),
                event_context={"update_id": update.update_id},
            ),
        )

    selected_occurrence, _reminder, selected_notification = candidates[0]
    db.rollback()
    try:
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                log_event(
                    logger, logging.INFO, "snooze_request_replayed", update_id=update.update_id
                )
                return _reply_from_record(existing)
            locked_user = db.get(User, user.id)
            locked_identity = db.get(ChannelIdentity, identity.id)
            if not _authorized_telegram_identity(
                locked_user,
                locked_identity,
                telegram_user_id=update.message.from_user.id,
                chat_id=update.message.chat.id,
            ):
                log_event(
                    logger,
                    logging.WARNING,
                    "snooze_authorization_failed",
                    update_id=update.update_id,
                    user_id=str(user.id),
                    identity_id=str(identity.id),
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=snooze_message("authorization_failed"),
                    response_kind="authorization_failure",
                    operation="SNOOZE_OCCURRENCE",
                )
            assert locked_user is not None and locked_identity is not None
            locked_session = load_session(
                db,
                user_id=user.id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            result = snooze_occurrence(
                db,
                user_id=user.id,
                identity_id=identity.id,
                chat_id=update.message.chat.id,
                occurrence_id=selected_occurrence.id,
                duration_seconds=duration.duration_seconds,
                provider_message_id=(provider_id or selected_notification.provider_message_id),
                now=now,
            )
            log_event(
                logger,
                logging.INFO if result.outcome == "SNOOZED" else logging.WARNING,
                "snooze_request_completed",
                update_id=update.update_id,
                occurrence_id=str(selected_occurrence.id),
                outcome=result.outcome,
                duration_seconds=duration.duration_seconds,
            )
            if locked_session is not None:
                end_session(locked_session, now=now)
            text_result = result.text
            if result.outcome == "SNOOZED" and result.due_at is not None:
                assert locked_user.timezone is not None
                task = (
                    db.scalar(
                        select(Reminder.task)
                        .join(ReminderOccurrence, ReminderOccurrence.reminder_id == Reminder.id)
                        .where(ReminderOccurrence.id == selected_occurrence.id)
                    )
                    or "that reminder"
                )
                text_result = (
                    snooze_message(
                        "snooze_success",
                        task=task,
                        when=f"at {_local_display(result.due_at, locked_user.timezone)}",
                        new_datetime=_local_display(result.due_at, locked_user.timezone),
                    )
                )
            return _store_response(
                db,
                update=update,
                user_id=user.id,
                text=text_result,
                response_kind="snooze" if result.outcome == "SNOOZED" else "snooze_response",
                operation="SNOOZE_OCCURRENCE",
            )
    except IntegrityError:
        duplicate = _recover_from_idempotency_race(db, update)
        if duplicate is not None:
            return duplicate
        raise


def _log_context(values: Mapping[str, object]) -> dict[str, Any]:
    return cast(dict[str, Any], dict(values))


def handle_snooze_clarification[ResponseT](
    db: Session,
    *,
    user_id: UUID,
    channel_type: str,
    external_session_id: str,
    pending: PendingSnooze | None,
    response_text: str,
    response_kind: str,
    now: datetime,
    session_ttl_seconds: int,
    upsert_pending: Callable[..., object],
    find_existing: Callable[[], ResponseT | None],
    stage_response: Callable[..., ResponseT],
    recover_from_race: Callable[[], ResponseT | None],
    event_context: Mapping[str, object] | None = None,
) -> ResponseT:
    """Stage snooze clarification/session state in the caller's established pattern."""

    event_context = event_context or {}
    db.rollback()
    try:
        with db.begin():
            existing = find_existing()
            if existing is not None:
                log_event(
                    logger, logging.INFO, "snooze_request_replayed", **_log_context(event_context)
                )
                return existing
            locked_session = load_session(
                db,
                user_id=user_id,
                external_session_id=external_session_id,
                channel_type=channel_type,
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if pending is not None:
                upsert_pending(
                    user_id=user_id,
                    external_session_id=external_session_id,
                    pending=pending,
                    now=now,
                    channel_type=channel_type,
                )
            elif locked_session is not None:
                end_session(locked_session, now=now)
            return stage_response(
                text=response_text,
                response_kind=response_kind,
                operation="SNOOZE_OCCURRENCE",
            )
    except IntegrityError:
        duplicate = recover_from_race()
        if duplicate is not None:
            return duplicate
        raise
