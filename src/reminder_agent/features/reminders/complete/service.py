"""Transaction-owned pending completion coordination."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.conversations import (
    end_session,
    save_pending_clarification_session,
    telegram_session_id,
)
from reminder_agent.application.response_staging import (
    ReminderReply,
    _existing_record,
    _recover_from_idempotency_race,
    _reply_from_record,
    _store_response,
)
from reminder_agent.features.reminders.complete.clarification import (
    stage_completion_clarification,
    stage_pending_completion_unresolved,
)
from reminder_agent.features.reminders.complete.operations import (
    complete_occurrence,
    match_completion_candidates,
    resolve_by_provider_message,
    resolve_latest_delivered,
)
from reminder_agent.features.reminders.complete.pending import prepare_pending_completion
from reminder_agent.features.reminders.complete.resolution import resolve_completion_target
from reminder_agent.features.reminders.complete.response import (
    commit_completion_response,
    stage_completion_response,
)
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    User,
)
from reminder_agent.infrastructure.observability import get_logger, log_event

logger = get_logger()


def _authorized_telegram_completion_identity(
    db: Session,
    callback: Any,
) -> ChannelIdentity | None:
    """Resolve the linked Telegram identity for a validated callback."""

    return db.scalar(
        select(ChannelIdentity)
        .join(User, User.id == ChannelIdentity.user_id)
        .where(
            User.onboarding_status == "ACTIVE",
            User.timezone.is_not(None),
            ChannelIdentity.channel_type == "TELEGRAM",
            ChannelIdentity.external_id == str(callback.telegram_user_id),
            ChannelIdentity.status == "LINKED",
            ChannelIdentity.telegram_chat_id == callback.chat_id,
        )
    )


def handle_completion_callback(
    db: Session,
    update: Any,
    *,
    callback: Any,
    now: datetime | None = None,
) -> ReminderReply | None:
    """Authorize, complete, and durably acknowledge a Telegram callback.

    Telegram parsing remains in the channel adapter. This function owns the
    completion-specific authorization, mutation, idempotency, and commit.
    """

    existing = _existing_record(db, update)
    if existing is not None:
        db.rollback()
        log_event(
            logger,
            logging.INFO,
            "completion_callback_duplicate",
            update_id=update.update_id,
            occurrence_id=str(callback.occurrence_id),
        )
        return _reply_from_record(existing)

    identity = _authorized_telegram_completion_identity(db, callback)
    if identity is None:
        log_event(
            logger,
            logging.WARNING,
            "completion_callback_rejected",
            update_id=update.update_id,
            occurrence_id=str(callback.occurrence_id),
            reason="telegram_identity_not_authorized",
        )
        return None

    result = complete_occurrence(
        db,
        user_id=identity.user_id,
        identity_id=identity.id,
        chat_id=callback.chat_id,
        occurrence_id=callback.occurrence_id,
        provider_message_id=callback.provider_message_id,
        now=now,
    )
    if result.outcome == "ALREADY_COMPLETED":
        db.rollback()
        log_event(
            logger,
            logging.INFO,
            "completion_callback_noop",
            update_id=update.update_id,
            occurrence_id=str(callback.occurrence_id),
            outcome=result.outcome,
        )
        return ReminderReply(
            chat_id=callback.chat_id,
            text=result.text,
            should_send=False,
            response_kind="completion_noop",
        )
    reply = _store_response(
        db,
        update=update,
        user_id=identity.user_id,
        text=result.text,
        response_kind="completion",
        cleanup_provider_message_id=callback.provider_message_id,
        operation="COMPLETE_REMINDER",
        include_request_text=False,
    )
    try:
        db.commit()
    except IntegrityError:
        duplicate = _recover_from_idempotency_race(db, update)
        if duplicate is not None:
            return duplicate
        raise
    log_event(
        logger,
        logging.INFO,
        "completion_callback_committed",
        update_id=update.update_id,
        occurrence_id=str(result.occurrence_id)
        if result.occurrence_id
        else str(callback.occurrence_id),
        outcome=result.outcome,
    )
    return reply


def _handle_completion_follow_up(
    db: Session,
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity,
    now: datetime,
    session_ttl_seconds: int,
) -> ReminderReply | None:
    """Compatibility wrapper for the transaction-owned pending workflow."""

    assert update.message is not None
    return handle_pending_completion(
        db,
        update,
        user=user,
        identity=identity,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        stage_response=lambda **kwargs: _store_response(
            db,
            update=update,
            user_id=user.id,
            **kwargs,
        ),
        recover_from_race=lambda: _recover_from_idempotency_race(db, update),
    )

def _handle_completion_text(
    db: Session,
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity,
    now: datetime,
    target_text: str | None = None,
) -> ReminderReply:
    """Resolve and complete one occurrence with explicit commit ownership.

    ``complete_occurrence`` only stages the state change; this workflow commits
    it together with the idempotency response.
    """
    assert update.message is not None
    provider_id = (
        str(update.message.reply_to_message.message_id)
        if update.message.reply_to_message is not None
        else None
    )
    log_event(
        logger,
        logging.INFO,
        "completion_text_received",
        update_id=update.update_id,
        user_id=str(user.id),
        identity_id=str(identity.id),
        chat_id=update.message.chat.id,
        resolution="reply" if provider_id is not None else "latest_delivered",
        has_provider_message_reference=provider_id is not None,
        target_text=target_text,
    )
    # A reply explicitly identifies the Telegram notification. Never fall
    # back to a different reminder when that reference is stale or unrelated.
    resolution = resolve_completion_target(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=update.message.chat.id,
        provider_message_id=provider_id,
        target_text=target_text,
        resolve_by_provider_message_fn=resolve_by_provider_message,
        resolve_latest_delivered_fn=resolve_latest_delivered,
        match_candidates_fn=match_completion_candidates,
    )
    candidates = list(resolution.candidates)
    log_event(
        logger,
        logging.DEBUG,
        "completion_target_resolution_finished",
        update_id=update.update_id,
        candidate_count=len(candidates),
        resolution="reply" if provider_id is not None else "latest_delivered",
    )
    if len(candidates) != 1:
        db.rollback()
        reply = stage_completion_clarification(
            user_id=user.id,
            external_session_id=telegram_session_id(update.message.chat.id),
            candidate_occurrence_ids=tuple(candidate[0].id for candidate in candidates),
            now=now,
            upsert_clarification_session=lambda **kwargs: save_pending_clarification_session(
                db,
                **kwargs,
            ),
            stage_response=lambda **kwargs: _store_response(
                db,
                update=update,
                user_id=user.id,
                **kwargs,
            ),
            event_context={"update_id": update.update_id},
        )
        reply = commit_completion_response(
            db,
            reply=reply,
            recover_from_race=lambda: _recover_from_idempotency_race(db, update),
        )
        log_event(
            logger,
            logging.INFO,
            "completion_response_committed",
            update_id=update.update_id,
            outcome="AMBIGUOUS",
            candidate_count=len(candidates),
        )
        return reply

    result = complete_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=update.message.chat.id,
        occurrence_id=candidates[0][0].id,
        provider_message_id=provider_id,
        now=now,
    )
    log_event(
        logger,
        logging.INFO,
        "completion_service_result",
        update_id=update.update_id,
        occurrence_id=str(result.occurrence_id) if result.occurrence_id else None,
        outcome=result.outcome,
        resolution="reply" if provider_id is not None else "unambiguous_last_delivered",
    )
    reply = stage_completion_response(
        stage_response=lambda **kwargs: _store_response(
            db,
            update=update,
            user_id=user.id,
            **kwargs,
        ),
        text=result.text,
        response_kind="completion",
        cleanup_provider_message_id=(
            candidates[0][2].provider_message_id if provider_id is None else provider_id
        ),
    )
    reply = commit_completion_response(
        db,
        reply=reply,
        recover_from_race=lambda: _recover_from_idempotency_race(db, update),
    )
    log_event(
        logger,
        logging.INFO,
        "completion_response_committed",
        update_id=update.update_id,
        occurrence_id=str(result.occurrence_id) if result.occurrence_id else None,
        outcome=result.outcome,
    )
    log_event(
        logger,
        logging.INFO,
        "reminder_occurrence_completed",
        update_id=update.update_id,
        occurrence_id=str(result.occurrence_id) if result.occurrence_id else None,
        outcome=result.outcome,
        resolution="reply" if provider_id is not None else "unambiguous_last_delivered",
    )
    return reply


def handle_pending_completion(
    db: Session,
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity,
    now: datetime,
    session_ttl_seconds: int,
    stage_response: Callable[..., ReminderReply],
    recover_from_race: Callable[[], ReminderReply | None],
) -> ReminderReply | None:
    """Resolve a completion clarification in one owned write transaction."""

    assert update.message is not None
    db.rollback()
    reply: ReminderReply | None = None
    result = None
    candidate_count = 0
    try:
        with db.begin():
            preparation = prepare_pending_completion(
                db,
                user_id=user.id,
                identity_id=identity.id,
                channel_type=identity.channel_type,
                external_session_id=telegram_session_id(update.message.chat.id),
                chat_id=update.message.chat.id,
                external_destination_id=None,
                text=update.message.text or "",
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
            if preparation is None:
                return None
            session = preparation.session
            candidates = list(preparation.candidates)
            candidate_count = len(candidates)
            matches = list(preparation.matches)
            if len(matches) != 1:
                reply, outcome = stage_pending_completion_unresolved(
                    session=session,
                    candidates=candidates,
                    matches=matches,
                    now=now,
                    stage_response=stage_response,
                    event_context={"update_id": update.update_id},
                )
            else:
                selected = matches[0]
                result = complete_occurrence(
                    db,
                    user_id=user.id,
                    identity_id=identity.id,
                    chat_id=update.message.chat.id,
                    occurrence_id=selected[0].id,
                    now=now,
                )
                end_session(session, now=now)
                reply = stage_completion_response(
                    stage_response=stage_response,
                    text=result.text,
                    response_kind="completion",
                    cleanup_provider_message_id=selected[2].provider_message_id,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "completion_follow_up_resolved",
                    update_id=update.update_id,
                    occurrence_id=str(selected[0].id),
                    outcome=result.outcome,
                )
    except IntegrityError:
        duplicate = recover_from_race()
        if duplicate is not None:
            return duplicate
        raise
    assert reply is not None
    log_event(
        logger,
        logging.INFO,
        "completion_follow_up_committed",
        update_id=update.update_id,
        outcome=result.outcome if result is not None else "UNRESOLVED",
        candidate_count=candidate_count,
    )
    return reply
