from __future__ import annotations

import hashlib
import inspect
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from pydantic import ValidationError

from reminder_agent.db.models import (
    ChannelIdentity,
    ConversationSession,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.conversations import (
    AWAITING_CLARIFICATION,
    end_session,
    load_session,
    pending_from_session,
    pending_reschedule_from_session,
    pending_cancellation_from_session,
    pending_completion_from_session,
    pending_snooze_from_session,
    session_expired,
    telegram_session_id,
)
from reminder_agent.cancellation import (
    CancellationConflict,
    CancellationDecision,
    CancellationResolution,
    CancellationRequest,
    CancellationScope,
    PendingCancellation,
    apply_cancellation,
    decide_cancellation,
    bounded_cancellation_context,
    classify_cancellation_resolution,
    find_candidates,
    find_partial_candidates,
    find_unmaterialized_occurrence_candidates,
    format_candidates,
    ground_model_cancellation_request,
    is_cancel_command,
    is_semantically_uncertain_cancellation_target,
    _request_occurrence_date,
    parse_cancellation_request,
)
from reminder_agent.conversation_router import (
    ConversationRoute,
    is_completion_command,
    is_new_reminder_command,
    is_list_request_command,
    is_summary_request_command,
    route_message,
    social_response,
    unrelated_response,
)
from reminder_agent.completion import (
    complete_occurrence,
    match_completion_candidates,
    PendingCompletion,
    resolve_latest_delivered,
    resolve_by_provider_message,
    resolve_by_occurrence,
)
from reminder_agent.snooze import (
    PendingSnooze,
    SnoozeDuration,
    is_snooze_command,
    parse_snooze_duration,
    snooze_target_text,
    snooze_occurrence,
)
from reminder_agent.intent import (
    CreateRecurringReminderIntent,
    CreateOneTimeReminderIntent,
    IntentRoute,
    IntentRouter,
    IntentInterpreter,
    Interpretation,
    ListRemindersIntent,
    ReminderSummaryIntent,
    PendingReminder,
    ReminderIntentInterpreter,
    RuleBasedIntentInterpreter,
    restrict_to_explicit_field_updates,
)
from reminder_agent.reminder_listing import (
    format_reminder_list,
    format_reminder_summary,
    list_reminders_in_range,
    summarize_reminders,
)
from reminder_agent.recurrence import (
    MonthlyRecurrence,
    WeeklyRecurrence,
    YearlyRecurrence,
    resolve_local_datetime,
)
from reminder_agent.recurrence_service import materialize_recurring_reminder
from reminder_agent.observability import get_logger, log_event
from reminder_agent.request_policy import (
    OUT_OF_SCOPE_RESPONSE,
    assess_request,
    is_supported_operation,
)
from reminder_agent.rescheduling import (
    PendingReschedule,
    PendingRescheduleOperation,
    PendingRescheduleProposal,
    RescheduleDecision,
    RescheduleRequest,
    RescheduleConflict,
    RescheduleScope,
    apply_reschedule,
    apply_pending_reschedule_proposal,
    decide_reschedule,
    find_all_candidates,
    pending_candidates,
    select_pending_candidate_ids,
    is_reschedule_command,
    has_unsupported_series_cadence_change,
    parse_reschedule_request,
    parse_pending_reschedule_parameters,
)
from reminder_agent.telegram import TelegramUpdate

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
ONE_TIME = "ONE_TIME"
RECURRING = "RECURRING"
PENDING = "PENDING"
logger = get_logger()


def _required_due_at(request: RescheduleRequest) -> datetime:
    """Return the application-resolved due time immediately before mutation."""
    if request.due_at is None:
        raise ValueError("reschedule time could not be resolved")
    return request.due_at


def _recurrence_for_intent(intent: CreateRecurringReminderIntent):
    if intent.frequency == "WEEKLY":
        return WeeklyRecurrence(intent.weekdays, intent.hour, intent.minute, intent.interval)
    if intent.frequency == "MONTHLY":
        return MonthlyRecurrence(
            intent.day_of_month or 0, intent.hour, intent.minute, intent.interval
        )
    return YearlyRecurrence(
        intent.month or 0,
        intent.day_of_month or 0,
        intent.hour,
        intent.minute,
        intent.interval,
    )


def _ordinal_suffix(day: int) -> str:
    if 10 < day % 100 < 14:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


@dataclass(frozen=True)
class ReminderReply:
    chat_id: int
    text: str
    idempotency_key: str | None = None
    should_send: bool = True
    is_duplicate: bool = False
    reply_markup: dict | None = None


def list_owned_reminders(
    db: Session,
    *,
    user_id: UUID,
    status: str | None = None,
) -> list[Reminder]:
    """Read reminders owned by ``user_id`` and never another account.

    This is intentionally a small application-service primitive rather than a
    conversational list feature. Callers must provide the canonical user ID
    resolved by their authenticated channel context.
    """

    filters = [Reminder.user_id == user_id]
    if status is not None:
        filters.append(Reminder.status == status)
    return list(
        db.scalars(
            select(Reminder)
            .where(*filters)
            .order_by(Reminder.created_at.asc(), Reminder.id.asc())
        ).all()
    )


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


def _authorized_telegram_identity(
    user: User | None,
    identity: ChannelIdentity | None,
    *,
    telegram_user_id: int,
    chat_id: int,
) -> bool:
    """Check that the channel identity belongs to the requested user."""

    return bool(
        user is not None
        and identity is not None
        and user.onboarding_status == ACTIVE
        and bool(user.timezone)
        and identity.user_id == user.id
        and identity.channel_type == TELEGRAM
        and identity.status == LINKED
        and identity.external_id == str(telegram_user_id)
        and identity.telegram_chat_id == chat_id
    )


def _local_display(due_at: datetime, timezone_name: str) -> str:
    return due_at.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M %Z")


def _store_response(
    db: Session,
    *,
    update: TelegramUpdate,
    user_id: UUID,
    text: str,
    response_kind: str = "clarification",
    operation: str = "CREATE_REMINDER",
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
            operation=operation,
            request_hash=_request_hash(update),
            status="SUCCEEDED",
            delivery_status="PENDING",
            response={
                "chat_id": reply.chat_id,
                "text": reply.text,
                "reply_markup": None,
                # Kept only in the durable idempotency record so the intent
                # classifier can receive the last 34 turns; never log it.
                "request_text": update.message.text or "",
            },
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


def _save_reschedule_session(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    pending: PendingReschedule,
    now: datetime,
) -> ConversationSession:
    session = db.scalar(
        select(ConversationSession)
        .where(
            ConversationSession.user_id == user_id,
            ConversationSession.channel_type == TELEGRAM,
            ConversationSession.external_session_id == external_session_id,
        )
        .with_for_update()
    )
    if session is None:
        session = ConversationSession(
            user_id=user_id,
            channel_type=TELEGRAM,
            external_session_id=external_session_id,
            pending_clarification=pending.model_dump(mode="json"),
            status=AWAITING_CLARIFICATION,
            version=1,
            updated_at=_now_utc(now),
        )
        db.add(session)
    else:
        session.pending_clarification = pending.model_dump(mode="json")
        session.status = AWAITING_CLARIFICATION
        session.active_reminder_id = None
        session.version += 1
        session.updated_at = _now_utc(now)
    return session


def _save_cancellation_session(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    pending: PendingCancellation,
    now: datetime,
) -> ConversationSession:
    session = db.scalar(
        select(ConversationSession)
        .where(
            ConversationSession.user_id == user_id,
            ConversationSession.channel_type == TELEGRAM,
            ConversationSession.external_session_id == external_session_id,
        )
        .with_for_update()
    )
    if session is None:
        session = ConversationSession(
            user_id=user_id,
            channel_type=TELEGRAM,
            external_session_id=external_session_id,
            pending_clarification=pending.model_dump(mode="json"),
            status=AWAITING_CLARIFICATION,
            version=1,
            updated_at=_now_utc(now),
        )
        db.add(session)
    else:
        session.pending_clarification = pending.model_dump(mode="json")
        session.status = AWAITING_CLARIFICATION
        session.active_reminder_id = None
        session.version += 1
        session.updated_at = _now_utc(now)
    return session


def _handle_cancellation_text(
    db: Session,
    update: TelegramUpdate,
    *,
    user_id: UUID,
    identity_id: UUID,
    telegram_user_id: int,
    user: User,
    pending: PendingCancellation | None,
    now: datetime,
    session_ttl_seconds: int = 900,
    request_override: CancellationRequest | None = None,
    cancellation_interpreter: IntentInterpreter | None = None,
    recent_messages: list[str] | None = None,
    force: bool = False,
) -> ReminderReply | None:
    """Resolve and apply VS-10 cancellation requests through one transaction."""

    assert update.message is not None
    text = update.message.text or ""
    cancellation_context = bounded_cancellation_context(recent_messages or [])
    log_event(
        logger,
        logging.INFO,
        "cancellation_command_started",
        update_id=update.update_id,
        input_length=len(text.strip()),
        has_pending=pending is not None,
    )
    if pending is None and request_override is None and not is_cancel_command(text):
        if not force:
            return None

    parsed_request = None
    if pending is None or is_cancel_command(text):
        parsed = parse_cancellation_request(
            text, timezone_name=user.timezone or "UTC", now=now
        )
        parsed_request = None if isinstance(parsed, str) else parsed
        pending_for_decision = None
    else:
        pending_for_decision = pending

    db.rollback()
    if pending_for_decision is not None:
        decision = decide_cancellation(
            db,
            user_id=user_id,
            text=text,
            timezone_name=user.timezone or "UTC",
            now=now,
            pending=pending_for_decision,
        )
    elif parsed_request is not None:
        decision = decide_cancellation(
            db,
            user_id=user_id,
            text=text,
            timezone_name=user.timezone or "UTC",
            now=now,
            request_override=parsed_request,
        )
    else:
        decision = None

    partial_candidates = []
    if decision is not None and decision.kind == "RESPONSE" and parsed_request is not None:
        partial_candidates = find_partial_candidates(
            db,
            user_id=user_id,
            target_text=parsed_request.target_text,
            occurrence_date=parsed_request.occurrence_date,
            timezone_name=user.timezone or "UTC",
            include_empty_series=(
                parsed_request.scope is not CancellationScope.OCCURRENCE
            ),
        )

    if len(partial_candidates) > 1 and parsed_request is not None:
        decision = CancellationDecision(
            "CLARIFY",
            "I found multiple possible reminder matches. Which one should I cancel?\n"
            + format_candidates(partial_candidates, timezone_name=user.timezone or "UTC"),
            request=parsed_request,
            candidate_ids=tuple(candidate.reminder_id for candidate in partial_candidates),
            scope=parsed_request.scope,
        )
        partial_candidates = []

    resolution = classify_cancellation_resolution(
        decision, parsed_request, partial_candidates
    )
    if (
        resolution is CancellationResolution.UNCERTAIN_EXTRACTION
        and pending_for_decision is None
    ):
        proposal = request_override
        interpreter_method = getattr(cancellation_interpreter, "interpret_cancellation", None)
        if interpreter_method is not None:
            occurrence_date = (
                parsed_request.occurrence_date
                if parsed_request is not None
                else _request_occurrence_date(
                    text, now=now, timezone_name=user.timezone or "UTC"
                )
            )
            candidates = find_candidates(
                db,
                user_id=user_id,
                target_text=None,
                occurrence_date=occurrence_date,
                timezone_name=user.timezone or "UTC",
                include_empty_series=(
                    occurrence_date is None
                    and (
                        parsed_request is None
                        or parsed_request.scope is not CancellationScope.OCCURRENCE
                    )
                ),
            )
            if occurrence_date is not None:
                seen_reminders = {candidate.reminder_id for candidate in candidates}
                candidates.extend(
                    candidate
                    for candidate in find_unmaterialized_occurrence_candidates(
                        db,
                        user_id=user_id,
                        target_text=None,
                        occurrence_date=occurrence_date,
                        timezone_name=user.timezone or "UTC",
                        now=now,
                    )
                    if candidate.reminder_id not in seen_reminders
                )
            candidates = candidates[:12]
            candidate_context = [
                {
                    "task": candidate.task,
                    "schedule_type": candidate.schedule_type,
                    "due_at": candidate.due_at.isoformat() if candidate.due_at else "",
                }
                for candidate in candidates
            ]
            kwargs = {
                "timezone_name": user.timezone or "UTC",
                "now": now,
                "recent_messages": cancellation_context,
                "cancellation_candidates": candidate_context,
            }
            try:
                parameters = inspect.signature(interpreter_method).parameters
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                proposal = interpreter_method(
                    text,
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key in parameters or accepts_kwargs
                    },
                )
            except Exception as exc:  # model failures must leave cancellation unchanged
                log_event(
                    logger,
                    logging.WARNING,
                    "cancellation_target_fallback_failed",
                    reason_type=type(exc).__name__,
                )
                proposal = None
        if isinstance(proposal, CancellationRequest):
            try:
                proposal = ground_model_cancellation_request(
                    text,
                    proposal,
                    timezone_name=user.timezone or "UTC",
                    now=now,
                    recent_messages=cancellation_context,
                )
            except ValueError as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "cancellation_target_fallback_rejected",
                    reason_type=type(exc).__name__,
                )
                proposal = None
        if isinstance(proposal, CancellationRequest):
            decision = decide_cancellation(
                db,
                user_id=user_id,
                text=text,
                timezone_name=user.timezone or "UTC",
                now=now,
                request_override=proposal,
            )
        elif decision is None:
            decision = CancellationDecision(
                "RESPONSE", "Please tell me which reminder to cancel."
            )
        elif partial_candidates:
            decision = CancellationDecision(
                "RESPONSE",
                "I found a possible reminder match. Please use more of the reminder's task.",
            )
    elif (
        resolution is CancellationResolution.NO_MATCH
        and decision is not None
        and decision.kind == "RESPONSE"
        and is_semantically_uncertain_cancellation_target(parsed_request)
    ):
        decision = CancellationDecision(
            "RESPONSE",
            "I couldn't confidently match that description. Please use more of the reminder's task.",
        )
    db.rollback()

    try:
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                return _reply_from_record(existing)
            locked_user = db.get(User, user_id)
            locked_identity = db.get(ChannelIdentity, identity_id)
            if not _authorized_telegram_identity(
                locked_user,
                locked_identity,
                telegram_user_id=telegram_user_id,
                chat_id=update.message.chat.id,
            ):
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text="I couldn't verify your Telegram account.",
                    response_kind="authorization_failure",
                    operation="CANCEL_REMINDER",
                )
            assert locked_user is not None
            session = load_session(
                db,
                user_id=user_id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            locked_pending = pending_cancellation_from_session(session)
            if pending_for_decision is not None and locked_pending is not None:
                decision = decide_cancellation(
                    db,
                    user_id=user_id,
                    text=text,
                    timezone_name=locked_user.timezone or "UTC",
                    now=now,
                    pending=locked_pending,
                )

            if decision.kind == "CANCEL":
                if session is not None:
                    end_session(session, now=now)
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="cancellation_abandoned",
                    operation="CANCEL_REMINDER",
                )

            if decision.kind == "RESPONSE":
                if session is not None and (locked_pending is not None or parsed_request is not None):
                    end_session(session, now=now)
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="cancellation_response",
                    operation="CANCEL_REMINDER",
                )

            if decision.kind == "CLARIFY":
                request = decision.request
                if request is None:
                    return _store_response(
                        db,
                        update=update,
                        user_id=user_id,
                        text=decision.text,
                        response_kind="cancellation_clarification",
                        operation="CANCEL_REMINDER",
                    )
                pending_record = PendingCancellation(
                    target_text=request.target_text,
                    candidate_ids=decision.candidate_ids,
                    scope=request.scope,
                    occurrence_date=request.occurrence_date,
                    timezone_name=locked_user.timezone or "UTC",
                )
                _save_cancellation_session(
                    db,
                    user_id=user_id,
                    external_session_id=telegram_session_id(update.message.chat.id),
                    pending=pending_record,
                    now=now,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "cancellation_clarification_saved",
                    update_id=update.update_id,
                    candidate_count=len(decision.candidate_ids),
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="cancellation_clarification",
                    operation="CANCEL_REMINDER",
                )

            if len(decision.candidate_ids) != 1 or decision.scope is None:
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text="Please tell me which reminder to cancel.",
                    response_kind="cancellation_clarification",
                    operation="CANCEL_REMINDER",
                )
            reminder, occurrence_count = apply_cancellation(
                db,
                user_id=user_id,
                reminder_id=decision.candidate_ids[0],
                occurrence_id=decision.occurrence_id,
                scope=decision.scope,
                now=now,
                occurrence_date=decision.request.occurrence_date if decision.request else None,
            )
            log_event(
                logger,
                logging.INFO,
                "reminder_canceled",
                update_id=update.update_id,
                reminder_id=str(reminder.id),
                scope=decision.scope.value,
                occurrence_count=occurrence_count,
            )
            if session is not None:
                end_session(session, now=now)
            if decision.scope is CancellationScope.SERIES:
                confirmation = (
                    f"Canceled the recurring reminder '{reminder.task}' and its "
                    f"{occurrence_count} pending future occurrence(s) via Telegram."
                )
            elif reminder.schedule_type == ONE_TIME:
                confirmation = f"Canceled '{reminder.task}' via Telegram."
            else:
                confirmation = (
                    f"Canceled the selected upcoming '{reminder.task}' occurrence. "
                    "The recurring series remains active."
                )
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                text=confirmation,
                response_kind="cancellation_confirmation",
                operation="CANCEL_REMINDER",
            )
    except IntegrityError:
        duplicate = _recover_from_idempotency_race(db, update)
        if duplicate is not None:
            return duplicate
        raise
    except CancellationConflict as exc:
        log_event(
            logger,
            logging.WARNING,
            "reminder_cancellation_conflict",
            update_id=update.update_id,
            reason_type=type(exc).__name__,
        )
        db.rollback()
        with db.begin():
            session = load_session(
                db,
                user_id=user_id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if parsed_request is not None and session is not None:
                end_session(session, now=now)
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                text=f"I couldn't cancel that reminder safely: {exc}.",
                response_kind="cancellation_conflict",
                operation="CANCEL_REMINDER",
            )


def _handle_reschedule_text(
    db: Session,
    update: TelegramUpdate,
    *,
    user_id: UUID,
    identity_id: UUID,
    telegram_user_id: int,
    user: User,
    pending: PendingReschedule | None,
    now: datetime,
    session_ttl_seconds: int,
    force: bool = False,
    reschedule_interpreter: ReminderIntentInterpreter | None = None,
    recent_messages: list[str] | None = None,
    pending_session_version: int | None = None,
) -> ReminderReply | None:
    """Handle VS-09 before the creation interpreter is invoked."""

    assert update.message is not None
    text = update.message.text or ""
    if not force and not is_reschedule_command(text) and pending is None:
        return None

    unsupported_cadence_change = (
        pending is None and has_unsupported_series_cadence_change(text)
    )

    parsed_request = None
    if pending is None:
        parsed_request = parse_reschedule_request(
            text,
            timezone_name=user.timezone or "UTC",
            now=now,
        )
        log_event(
            logger,
            logging.INFO,
            "reschedule_deterministic_parse",
            update_id=update.update_id,
            parsed=isinstance(parsed_request, RescheduleRequest),
            result_type=type(parsed_request).__name__,
            result_length=len(parsed_request) if isinstance(parsed_request, str) else None,
            due_at=(
                parsed_request.due_at.isoformat()
                if isinstance(parsed_request, RescheduleRequest) and parsed_request.due_at is not None
                else None
            ),
            offset_minutes=(
                parsed_request.offset_minutes
                if isinstance(parsed_request, RescheduleRequest)
                else None
            ),
            scope=(
                parsed_request.scope.value
                if isinstance(parsed_request, RescheduleRequest)
                and parsed_request.scope is not None
                else None
            ),
        )
    deterministic_pending_proposal = (
        parse_pending_reschedule_parameters(
            text, timezone_name=user.timezone or "UTC", now=now
        )
        if pending is not None
        else None
    )
    pending_for_decision = (
        apply_pending_reschedule_proposal(
            pending,
            deterministic_pending_proposal,
            timezone_name=user.timezone or "UTC",
        )
        if pending is not None and deterministic_pending_proposal is not None
        else pending
    )
    decision = decide_reschedule(
        db,
        user_id=user.id,
        text=text,
        timezone_name=user.timezone or "UTC",
        now=now,
        pending=pending_for_decision,
        request_override=(
            parsed_request if isinstance(parsed_request, RescheduleRequest) else None
        ),
    )
    if unsupported_cadence_change:
        decision = RescheduleDecision(
            "RESPONSE", "Changing monthly or yearly recurrence dates is not supported yet."
        )
    pending_proposal: PendingRescheduleProposal | None = deterministic_pending_proposal
    # Pending replies are deterministic first.  Only an unresolved reply is
    # sent to the contextual extractor, and the result remains a proposal.
    if (
        pending is not None
        and decision.kind == "CLARIFY"
        and deterministic_pending_proposal is None
        and not (
            decision.request is not None
            and decision.request.scope is not pending.scope
        )
        and reschedule_interpreter is not None
        and hasattr(reschedule_interpreter, "interpret_pending_reschedule")
    ):
        db.rollback()
        candidate_context = [
            {
                "candidate_id": str(candidate.reminder_id),
                "task": candidate.task,
                "due_local": candidate.due_at.astimezone(
                    ZoneInfo(user.timezone or "UTC")
                ).isoformat(timespec="minutes"),
                "schedule_type": candidate.schedule_type,
            }
            for candidate in pending_candidates(db, user_id=user.id, pending=pending)
        ]
        proposal_kwargs = {
            "timezone_name": user.timezone or "UTC",
            "now": now,
            "pending_reschedule": pending.model_dump(mode="json"),
            "reschedule_candidates": candidate_context,
            "recent_messages": (recent_messages or [])[-6:],
        }
        try:
            method = reschedule_interpreter.interpret_pending_reschedule
            parameters = inspect.signature(method).parameters
            proposal = method(
                text,
                **{key: value for key, value in proposal_kwargs.items() if key in parameters},
            )
            if isinstance(proposal, PendingRescheduleProposal):
                pending_proposal = proposal
        except Exception as exc:  # provider errors always fall back to clarification
            log_event(
                logger,
                logging.WARNING,
                "reschedule_pending_llm_proposal_rejected",
                update_id=update.update_id,
                reason_type=type(exc).__name__,
            )
        log_event(
            logger,
            logging.INFO,
            "reschedule_pending_llm_proposal",
            update_id=update.update_id,
            accepted=pending_proposal is not None,
        )
    if (
        isinstance(parsed_request, str)
        and not unsupported_cadence_change
        and reschedule_interpreter is not None
    ):
        db.rollback()
        candidate_context = [
            {
                "candidate_id": str(candidate.reminder_id),
                "task": candidate.task,
                "due_local": candidate.due_at.astimezone(
                    ZoneInfo(user.timezone or "UTC")
                ).isoformat(timespec="minutes"),
                "schedule_type": candidate.schedule_type,
            }
            for candidate in find_all_candidates(db, user_id=user.id)
        ]
        proposal_kwargs = {
            "timezone_name": user.timezone or "UTC",
            "now": now,
        }
        if "reschedule_candidates" in inspect.signature(
            reschedule_interpreter.interpret_reschedule
        ).parameters:
            proposal_kwargs["reschedule_candidates"] = candidate_context
        proposal = reschedule_interpreter.interpret_reschedule(text, **proposal_kwargs)
        log_event(
            logger,
            logging.INFO,
            "reschedule_llm_proposal",
            update_id=update.update_id,
            result_type=type(proposal).__name__,
            accepted=isinstance(proposal, RescheduleRequest),
            response_length=len(proposal) if isinstance(proposal, str) else None,
            due_at=(
                proposal.due_at.isoformat()
                if isinstance(proposal, RescheduleRequest) and proposal.due_at is not None
                else None
            ),
            offset_minutes=(
                proposal.offset_minutes if isinstance(proposal, RescheduleRequest) else None
            ),
            scope=(
                proposal.scope.value
                if isinstance(proposal, RescheduleRequest) and proposal.scope is not None
                else None
            ),
        )
        if isinstance(proposal, RescheduleRequest):
            decision = decide_reschedule(
                db,
                user_id=user.id,
                text=text,
                timezone_name=user.timezone or "UTC",
                now=now,
                pending=pending,
                request_override=proposal,
            )
        elif isinstance(proposal, str) and proposal.strip():
            decision = RescheduleDecision("RESPONSE", proposal.strip())
    db.rollback()
    try:
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                return _reply_from_record(existing)
            locked_user = db.get(User, user_id)
            locked_identity = db.get(ChannelIdentity, identity_id)
            if not _authorized_telegram_identity(
                locked_user,
                locked_identity,
                telegram_user_id=telegram_user_id,
                chat_id=update.message.chat.id,
            ):
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text="I couldn't verify your Telegram account.",
                    response_kind="authorization_failure",
                    operation="RESCHEDULE_REMINDER",
                )
            assert locked_user is not None and locked_identity is not None
            locked_session = load_session(
                db,
                user_id=user_id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            locked_pending = pending_reschedule_from_session(locked_session)
            if locked_pending is not None:
                candidate_id_override = None
                effective_pending = locked_pending
                proposal_is_current = (
                    pending_proposal is not None
                    and pending is not None
                    and locked_session is not None
                    and pending_session_version is not None
                    and locked_session.version == pending_session_version
                )
                if proposal_is_current and pending_proposal is not None:
                    if (
                        pending_proposal.candidate_id is not None
                        and pending_proposal.candidate_id not in locked_pending.candidate_ids
                    ):
                        decision = RescheduleDecision(
                            "CLARIFY",
                            "Which reminder should I move?",
                            candidate_ids=locked_pending.candidate_ids,
                        )
                    elif pending_proposal.operation is PendingRescheduleOperation.CANCEL_PENDING:
                        decision = RescheduleDecision(
                            "CANCEL", "Okay, I won't move that reminder."
                        )
                    elif pending_proposal.operation in {
                        PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
                        PendingRescheduleOperation.SELECT_AND_UPDATE,
                    }:
                        effective_pending = apply_pending_reschedule_proposal(
                            locked_pending,
                            pending_proposal,
                            timezone_name=locked_user.timezone or "UTC",
                        )
                        candidate_id_override = pending_proposal.candidate_id
                        decision = decide_reschedule(
                            db,
                            user_id=user_id,
                            text=text,
                            timezone_name=locked_user.timezone or "UTC",
                            now=now,
                            pending=effective_pending,
                            candidate_id_override=candidate_id_override,
                        )
                    elif pending_proposal.operation is PendingRescheduleOperation.SELECT_CANDIDATE:
                        candidate_id_override = pending_proposal.candidate_id
                        decision = decide_reschedule(
                            db,
                            user_id=user_id,
                            text=text,
                            timezone_name=locked_user.timezone or "UTC",
                            now=now,
                            pending=effective_pending,
                            candidate_id_override=candidate_id_override,
                        )
                    else:
                        # START_NEW_OPERATION and NEEDS_CLARIFICATION cannot
                        # clear or replace a trusted pending reschedule.
                        decision = decide_reschedule(
                            db,
                            user_id=user_id,
                            text=text,
                            timezone_name=locked_user.timezone or "UTC",
                            now=now,
                            pending=effective_pending,
                        )
                else:
                    decision = decide_reschedule(
                        db,
                        user_id=user_id,
                        text=text,
                        timezone_name=locked_user.timezone or "UTC",
                        now=now,
                        pending=effective_pending,
                    )
                ordinal_match = re.search(
                    r"\b(?:the\s+)?(first|one|second|two|third|three|1|2|3)\b",
                    text.strip().lower().rstrip(".!?"),
                )
                log_event(
                    logger,
                    logging.INFO,
                    "reschedule_pending_clarification_resolved",
                    update_id=update.update_id,
                    candidate_count=len(locked_pending.candidate_ids),
                    ordinal=ordinal_match.group(1) if ordinal_match is not None else None,
                    selected_candidate_count=len(decision.candidate_ids),
                    decision=decision.kind,
                    pending_due_at=(
                        effective_pending.due_at.isoformat()
                        if effective_pending.due_at is not None
                        else None
                    ),
                    pending_offset_minutes=effective_pending.offset_minutes,
                    pending_scope=(
                        effective_pending.scope.value
                        if effective_pending.scope is not None
                        else None
                    ),
                )

            if decision.kind == "CANCEL":
                if locked_session is not None:
                    end_session(locked_session, now=now)
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="reschedule_canceled",
                    operation="RESCHEDULE_REMINDER",
                )

            if decision.kind == "RESPONSE":
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="reschedule_rejected",
                    operation="RESCHEDULE_REMINDER",
                )

            if decision.kind == "CLARIFY":
                request = decision.request
                if request is None and effective_pending is not None:
                    request = RescheduleRequest(
                        target_text=effective_pending.target_text,
                        due_at=effective_pending.due_at,
                        date_explicit=effective_pending.date_explicit,
                        offset_minutes=effective_pending.offset_minutes,
                        scope=effective_pending.scope,
                        source_text=effective_pending.source_text,
                        occurrence_date=effective_pending.occurrence_date,
                        destination_weekday=effective_pending.destination_weekday,
                    )
                if request is None:
                    return _store_response(
                        db,
                        update=update,
                        user_id=user_id,
                        text=decision.text,
                        response_kind="reschedule_clarification",
                        operation="RESCHEDULE_REMINDER",
                    )
                pending_record = PendingReschedule(
                    target_text=request.target_text,
                    due_at=request.due_at,
                    date_explicit=request.date_explicit,
                    offset_minutes=request.offset_minutes,
                    source_text=request.source_text,
                    timezone_name=locked_user.timezone or "UTC",
                    candidate_ids=decision.candidate_ids,
                    scope=request.scope,
                    occurrence_date=request.occurrence_date,
                    destination_weekday=request.destination_weekday,
                )
                _save_reschedule_session(
                    db,
                    user_id=user_id,
                    external_session_id=telegram_session_id(update.message.chat.id),
                    pending=pending_record,
                    now=now,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "reschedule_clarification_saved",
                    update_id=update.update_id,
                    candidate_count=len(decision.candidate_ids),
                    due_at=request.due_at.isoformat() if request.due_at is not None else None,
                    offset_minutes=request.offset_minutes,
                    scope=request.scope.value if request.scope is not None else None,
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text=decision.text,
                    response_kind="reschedule_clarification",
                    operation="RESCHEDULE_REMINDER",
                )

            assert decision.kind == "MUTATE" and decision.request is not None
            if len(decision.candidate_ids) != 1:
                return _store_response(
                    db,
                    update=update,
                    user_id=user_id,
                    text="Please tell me which reminder to move.",
                    response_kind="reschedule_clarification",
                    operation="RESCHEDULE_REMINDER",
                )
            reminder, due_at = apply_reschedule(
                db,
                user_id=user_id,
                destination_identity_id=identity_id,
                reminder_id=decision.candidate_ids[0],
                occurrence_id=decision.occurrence_id,
                due_at=_required_due_at(decision.request),
                scope=decision.request.scope,
                now=now,
                destination_weekday=decision.request.destination_weekday,
            )
            log_event(
                logger,
                logging.INFO,
                "reschedule_mutation_applied",
                update_id=update.update_id,
                due_at=due_at.isoformat(),
                scope=decision.request.scope.value if decision.request.scope is not None else None,
            )
            if locked_session is not None:
                end_session(locked_session, now=now)
            if decision.request.scope is RescheduleScope.SERIES:
                local_time = due_at.astimezone(ZoneInfo(locked_user.timezone or "UTC"))
                cadence = (
                    f"every {local_time:%A} at "
                    if decision.request.destination_weekday is not None
                    else ""
                )
                confirmation = (
                    f"Updated all future '{reminder.task}' reminders to {cadence}"
                    f"{local_time:%H:%M %Z} via Telegram."
                )
            else:
                local_display = _local_display(due_at, locked_user.timezone or "UTC")
                confirmation = f"Moved '{reminder.task}' to {local_display} via Telegram."
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                text=confirmation,
                response_kind="reschedule_confirmation",
                operation="RESCHEDULE_REMINDER",
            )
    except IntegrityError:
        duplicate = _recover_from_idempotency_race(db, update)
        if duplicate is not None:
            return duplicate
        raise
    except (RescheduleConflict, ValueError) as exc:
        log_event(
            logger,
            logging.WARNING,
            "reschedule_rejected",
            update_id=update.update_id,
            reason_type=type(exc).__name__,
            reason=str(exc),
        )
        db.rollback()
        with db.begin():
            failed_session = load_session(
                db,
                user_id=user_id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if (
                failed_session is not None
                and pending_reschedule_from_session(failed_session) is not None
            ):
                end_session(failed_session, now=now)
                log_event(
                    logger,
                    logging.INFO,
                    "reschedule_clarification_cleared_after_rejection",
                    update_id=update.update_id,
                    reason_type=type(exc).__name__,
                )
            return _store_response(
                db,
                update=update,
                user_id=user_id,
                text=f"I couldn't move that reminder safely: {exc}.",
                response_kind="reschedule_conflict",
                operation="RESCHEDULE_REMINDER",
            )


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


def _interpret(
    interpreter: IntentInterpreter,
    text: str,
    *,
    selected_route: IntentRoute,
    timezone_name: str,
    now: datetime,
    context: PendingReminder | None,
    recent_messages: list[str],
):
    """Pass an explicit classifier route into the extraction boundary."""
    kwargs = {"timezone_name": timezone_name, "now": now}
    parameters = inspect.signature(interpreter.interpret).parameters
    if "selected_route" in parameters:
        kwargs["selected_route"] = selected_route
    if context is not None and "context" in parameters:
        kwargs["context"] = context
    if "recent_messages" in parameters:
        kwargs["recent_messages"] = recent_messages
    return interpreter.interpret(text, **kwargs)


def _selected_intent_route(
    route: ConversationRoute, *, pending: PendingReminder | None
) -> IntentRoute:
    if route is ConversationRoute.LIST_REQUEST:
        return IntentRoute.LIST_REQUEST
    if route is ConversationRoute.SUMMARY_REQUEST:
        return IntentRoute.SUMMARY_REQUEST
    if route is ConversationRoute.CLARIFICATION_REPLY:
        return IntentRoute.UPDATE_PENDING
    if route is ConversationRoute.MODEL_FALLBACK:
        return IntentRoute.UPDATE_PENDING if pending is not None else IntentRoute.NEW_REQUEST
    return IntentRoute.NEW_REQUEST


def _recent_messages(db: Session, *, user_id: UUID) -> list[str]:
    records = db.scalars(
        select(IdempotencyRecord)
        .where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.user_id == user_id,
        )
        .order_by(IdempotencyRecord.created_at.desc())
        .limit(34)
    ).all()
    messages: list[str] = []
    for record in reversed(records):
        if not isinstance(record.response, dict):
            continue
        request_text = record.response.get("request_text")
        response_text = record.response.get("text")
        if isinstance(request_text, str):
            messages.append(f"user: {request_text}")
        if isinstance(response_text, str):
            messages.append(f"assistant: {response_text}")
    return messages[-34:]


def _recent_delivered_reminders(
    db: Session, *, user_id: UUID, identity_id: UUID, chat_id: int
) -> list[dict[str, str]]:
    """Build non-authoritative context for intent classification only."""
    candidates = resolve_latest_delivered(
        db, user_id=user_id, identity_id=identity_id, chat_id=chat_id
    )
    return [
        {
            "task": reminder.task,
            "delivered_at": notification.sent_at.isoformat()
            if notification.sent_at is not None
            else "",
        }
        for _occurrence, reminder, notification in candidates
    ]


def _route_and_interpret(
    interpreter: IntentInterpreter,
    text: str,
    *,
    timezone_name: str,
    now: datetime,
    pending: PendingReminder | None,
    user_name: str | None,
    intent_router: IntentRouter | None = None,
    recent_messages: list[str] | None = None,
    recent_delivered_reminders: list[dict[str, str]] | None = None,
) -> tuple[ConversationRoute, Interpretation]:
    """Route first, then interpret only the route that permits it."""
    route = route_message(text, pending=pending).route
    classifier_selected_update = False
    completion_target_text: str | None = None
    log_event(
        logger,
        logging.INFO,
        "reminder_route_deterministic",
        route=route,
        has_pending=pending is not None,
    )
    if route is ConversationRoute.MODEL_FALLBACK and intent_router is not None:
        log_event(
            logger,
            logging.INFO,
            "reminder_route_classifier_requested",
            has_pending=pending is not None,
            recent_message_count=min(len(recent_messages or []), 34),
        )
        classify_kwargs = {
            "pending": pending,
            "recent_messages": recent_messages or [],
            "recent_delivered_reminders": recent_delivered_reminders or [],
        }
        if "recent_delivered_reminders" not in inspect.signature(intent_router.classify).parameters:
            classify_kwargs.pop("recent_delivered_reminders")
        suggestion_result = intent_router.classify(text, **classify_kwargs)
        suggestion = suggestion_result.operation
        completion_target_text = getattr(suggestion_result, "target_text", None)
        log_event(
            logger,
            logging.INFO,
            "reminder_route_classifier_selected",
            operation=suggestion,
        )
        if suggestion is IntentRoute.LIST_REQUEST:
            route = ConversationRoute.LIST_REQUEST
        elif suggestion is IntentRoute.SUMMARY_REQUEST:
            route = ConversationRoute.SUMMARY_REQUEST
        elif suggestion is IntentRoute.RESCHEDULE_REQUEST:
            route = ConversationRoute.RESCHEDULE_REQUEST
        elif suggestion is IntentRoute.CANCEL_REQUEST:
            route = ConversationRoute.CANCEL_REQUEST
        elif suggestion is IntentRoute.COMPLETE_REQUEST:
            route = ConversationRoute.COMPLETE_REQUEST
        elif suggestion is IntentRoute.SNOOZE_REQUEST:
            route = ConversationRoute.SNOOZE_REQUEST
        elif suggestion is IntentRoute.NEW_REQUEST:
            route = ConversationRoute.NEW_REQUEST
        elif suggestion is IntentRoute.UPDATE_PENDING:
            if pending is not None:
                route = ConversationRoute.CLARIFICATION_REPLY
                classifier_selected_update = True
            else:
                # UPDATE_PENDING has no valid target without an active draft.
                # Treat the message as a new request so a classifier mistake
                # cannot discard a complete reminder command.
                route = ConversationRoute.NEW_REQUEST
                log_event(
                    logger,
                    logging.INFO,
                    "reminder_route_classifier_recovered",
                    requested_operation=suggestion,
                    replacement_route=route,
                    reason="no_pending_clarification",
                )
        elif suggestion is IntentRoute.CANCEL_PENDING and pending is not None:
            route = ConversationRoute.CANCELLATION
        elif suggestion is IntentRoute.SOCIAL:
            route = ConversationRoute.SOCIAL
        elif suggestion is IntentRoute.UNRELATED:
            # A classifier has identified this as a clearly unsupported
            # request. Keep deterministic conversational handling separate,
            # but return the project-wide scope limitation here.
            return ConversationRoute.UNRELATED, Interpretation(
                response_text=OUT_OF_SCOPE_RESPONSE,
                fallback_allowed=False,
            )
        else:
            log_event(
                logger,
                logging.INFO,
                "reminder_route_uncertain",
                has_pending=pending is not None,
            )
            return ConversationRoute.UNRELATED, Interpretation(
                response_text=(
                    "I’m not sure whether you want a new reminder or to update your unfinished "
                    "one. Please say which reminder you mean."
                ),
                fallback_allowed=False,
            )
    if route is ConversationRoute.CANCELLATION:
        interpretation = Interpretation(
            response_text="Okay, I won't create that reminder.",
            fallback_allowed=False,
            clear_context=True,
        )
    elif route is ConversationRoute.SOCIAL:
        interpretation = Interpretation(
            response_text=social_response(text, user_name=user_name),
            fallback_allowed=False,
        )
    elif route is ConversationRoute.UNRELATED:
        interpretation = Interpretation(
            response_text=unrelated_response(text, pending=pending),
            fallback_allowed=False,
        )
    elif route is ConversationRoute.SUMMARY_REQUEST:
        interpretation = _interpret(
            interpreter,
            text,
            selected_route=IntentRoute.SUMMARY_REQUEST,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages or [],
        )
    elif route is ConversationRoute.RESCHEDULE_REQUEST:
        interpretation = Interpretation(fallback_allowed=False)
    elif route is ConversationRoute.CANCEL_REQUEST:
        # Deterministic parsing and candidate matching run before an optional
        # model proposal. The cancellation service invokes that fallback only
        # after a zero-match result, with owned active candidates as context.
        interpretation = Interpretation(fallback_allowed=False)
    elif route is ConversationRoute.COMPLETE_REQUEST:
        interpretation = Interpretation(
            fallback_allowed=False,
            completion_target_text=completion_target_text,
        )
    elif route is ConversationRoute.SNOOZE_REQUEST:
        interpretation = _interpret(
            interpreter,
            text,
            selected_route=IntentRoute.SNOOZE_REQUEST,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages or [],
        )
    else:
        selected_route = _selected_intent_route(route, pending=pending)
        interpretation = _interpret(
            interpreter,
            text,
            selected_route=selected_route,
            timezone_name=timezone_name,
            now=now,
            context=(
                pending
                if route
                in {ConversationRoute.CLARIFICATION_REPLY, ConversationRoute.MODEL_FALLBACK}
                else None
            ),
            recent_messages=recent_messages or [],
        )
    if interpretation.field_updates is not None:
        if pending is None:
            interpretation = Interpretation(
                response_text="Please tell me the reminder task and a future time.",
                fallback_allowed=False,
            )
        else:
            proposed_fields = interpretation.field_updates.supplied_fields()
            verified_updates = restrict_to_explicit_field_updates(
                text,
                interpretation.field_updates,
                context=pending,
                now=now,
                allow_task_replacement=classifier_selected_update,
                allow_semantic_field_updates=classifier_selected_update,
            )
            interpretation = RuleBasedIntentInterpreter().apply_field_updates(
                pending,
                verified_updates,
                now=now,
            )
            accepted_fields = verified_updates.supplied_fields()
            log_event(
                logger,
                logging.INFO,
                "pending_update_applied",
                proposed_fields=sorted(proposed_fields),
                accepted_fields=sorted(accepted_fields),
                rejected_fields=sorted(proposed_fields - accepted_fields),
                completed=interpretation.intent is not None,
                needs_clarification=interpretation.clarification is not None,
            )
    return route, interpretation


def _upsert_clarification_session(
    db: Session,
    *,
    user_id: UUID,
    external_session_id: str,
    pending: PendingReminder | PendingCompletion | PendingSnooze,
    now: datetime,
) -> ConversationSession:
    values = {
        "id": uuid4(),
        "user_id": user_id,
        "channel_type": TELEGRAM,
        "external_session_id": external_session_id,
        "active_reminder_id": None,
        "pending_clarification": pending.model_dump(mode="json"),
        "status": AWAITING_CLARIFICATION,
        "version": 1,
        "updated_at": _now_utc(now),
    }
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        statement = pg_insert(ConversationSession).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[
                ConversationSession.user_id,
                ConversationSession.channel_type,
                ConversationSession.external_session_id,
            ],
            set_={
                "pending_clarification": values["pending_clarification"],
                "status": AWAITING_CLARIFICATION,
                "active_reminder_id": None,
                "version": ConversationSession.version + 1,
                "updated_at": values["updated_at"],
            },
        )
        db.execute(statement)
    elif dialect == "sqlite":
        statement = sqlite_insert(ConversationSession).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[
                ConversationSession.user_id,
                ConversationSession.channel_type,
                ConversationSession.external_session_id,
            ],
            set_={
                "pending_clarification": values["pending_clarification"],
                "status": AWAITING_CLARIFICATION,
                "active_reminder_id": None,
                "version": ConversationSession.version + 1,
                "updated_at": values["updated_at"],
            },
        )
        db.execute(statement)
    else:  # pragma: no cover - supported deployments use PostgreSQL or SQLite tests
        db.add(ConversationSession(**values))
    session = db.scalar(
        select(ConversationSession).where(
            ConversationSession.user_id == user_id,
            ConversationSession.channel_type == TELEGRAM,
            ConversationSession.external_session_id == external_session_id,
        )
    )
    assert session is not None
    return session


def _handle_completion_follow_up(
    db: Session,
    update: TelegramUpdate,
    *,
    user: User,
    identity: ChannelIdentity,
    now: datetime,
    session_ttl_seconds: int,
) -> ReminderReply | None:
    """Resolve a reply to an earlier ambiguous delivered-reminder completion."""
    assert update.message is not None
    db.rollback()
    reply: ReminderReply | None = None
    result = None
    candidate_count = 0
    try:
        with db.begin():
            session = load_session(
                db,
                user_id=user.id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            locked_pending = pending_completion_from_session(session)
            if locked_pending is None:
                return None
            candidates: list[tuple[ReminderOccurrence, Reminder, Notification]] = []
            for occurrence_id in locked_pending.candidate_occurrence_ids:
                candidates.extend(
                    resolve_by_occurrence(
                        db,
                        user_id=user.id,
                        identity_id=identity.id,
                        chat_id=update.message.chat.id,
                        occurrence_id=occurrence_id,
                    )
                )
            candidate_count = len(candidates)
            matches = match_completion_candidates(update.message.text or "", candidates)
            if len(matches) != 1:
                if not candidates:
                    response_text = "I couldn't find a delivered reminder to mark as done."
                    end_session(session, now=now)
                    outcome = "NO_CANDIDATES"
                else:
                    response_text = "Which delivered reminder did you complete? Please name it."
                    outcome = "AMBIGUOUS"
                reply = _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=response_text,
                    response_kind="completion_ambiguous",
                    operation="COMPLETE_REMINDER",
                )
                log_event(
                    logger,
                    logging.INFO,
                    "completion_follow_up_unresolved",
                    update_id=update.update_id,
                    outcome=outcome,
                    candidate_count=candidate_count,
                    match_count=len(matches),
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
                reply = _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=result.text,
                    response_kind="completion",
                    operation="COMPLETE_REMINDER",
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
        duplicate = _recover_from_idempotency_race(db, update)
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
    update: TelegramUpdate,
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
    """Resolve and atomically snooze one already-delivered occurrence."""

    assert update.message is not None
    text = update.message.text or ""
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

    target_text = snooze_target_text(text)
    if target_text is None and pending is not None and parse_snooze_duration(text) is None:
        target_text = text.strip() or None
    if provider_id is None and target_text and len(candidates) > 1:
        targeted = match_completion_candidates(target_text, candidates)
        if len(targeted) == 1:
            candidates = targeted

    if len(candidates) != 1 or duration is None:
        if not candidates:
            response_text = "I couldn't find a delivered reminder to snooze."
        elif duration is None:
            response_text = "How long should I snooze that reminder?"
        else:
            response_text = "Which delivered reminder should I snooze? Please name it."
        pending_value = None
        if candidates:
            pending_value = PendingSnooze(
                candidate_occurrence_ids=tuple(item[0].id for item in candidates),
                duration_seconds=duration.duration_seconds if duration is not None else None,
            )
        db.rollback()
        try:
            with db.begin():
                existing = _existing_record(db, update)
                if existing is not None:
                    return _reply_from_record(existing)
                locked_session = load_session(
                    db,
                    user_id=user.id,
                    external_session_id=telegram_session_id(update.message.chat.id),
                    now=now,
                    ttl_seconds=session_ttl_seconds,
                    for_update=True,
                )
                if pending_value is not None:
                    _upsert_clarification_session(
                        db,
                        user_id=user.id,
                        external_session_id=telegram_session_id(update.message.chat.id),
                        pending=pending_value,
                        now=now,
                    )
                elif locked_session is not None:
                    end_session(locked_session, now=now)
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=response_text,
                    response_kind="snooze_clarification" if pending_value else "snooze_response",
                    operation="SNOOZE_OCCURRENCE",
                )
        except IntegrityError:
            duplicate = _recover_from_idempotency_race(db, update)
            if duplicate is not None:
                return duplicate
            raise

    selected_occurrence, _reminder, selected_notification = candidates[0]
    db.rollback()
    try:
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                return _reply_from_record(existing)
            locked_user = db.get(User, user.id)
            locked_identity = db.get(ChannelIdentity, identity.id)
            if not _authorized_telegram_identity(
                locked_user,
                locked_identity,
                telegram_user_id=update.message.from_user.id,
                chat_id=update.message.chat.id,
            ):
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text="I couldn't verify your Telegram account.",
                    response_kind="authorization_failure",
                    operation="SNOOZE_OCCURRENCE",
                )
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
                provider_message_id=(
                    provider_id or selected_notification.provider_message_id
                ),
                now=now,
            )
            if locked_session is not None:
                end_session(locked_session, now=now)
            text_result = result.text
            if result.outcome == "SNOOZED" and result.due_at is not None:
                task = db.scalar(
                    select(Reminder.task)
                    .join(ReminderOccurrence, ReminderOccurrence.reminder_id == Reminder.id)
                    .where(ReminderOccurrence.id == selected_occurrence.id)
                ) or "that reminder"
                text_result = (
                    f"Snoozed '{task}' until {_local_display(result.due_at, locked_user.timezone)} "
                    "on Telegram."
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


def _end_pending_session(
    db: Session,
    update: TelegramUpdate,
    *,
    user: User,
    now: datetime,
    session_ttl_seconds: int,
    response_text: str | None = None,
    response_kind: str = "snooze_canceled",
    operation: str = "SNOOZE_OCCURRENCE",
) -> ReminderReply | None:
    """End any pending clarification under lock, optionally recording a response."""

    assert update.message is not None
    db.rollback()
    with db.begin():
        existing = _existing_record(db, update)
        if existing is not None:
            return _reply_from_record(existing)
        session = load_session(
            db,
            user_id=user.id,
            external_session_id=telegram_session_id(update.message.chat.id),
            now=now,
            ttl_seconds=session_ttl_seconds,
            for_update=True,
        )
        if session is not None:
            end_session(session, now=now)
        if response_text is None:
            return None
        return _store_response(
            db,
            update=update,
            user_id=user.id,
            text=response_text,
            response_kind=response_kind,
            operation=operation,
        )


def _end_pending_snooze(
    db: Session,
    update: TelegramUpdate,
    *,
    user: User,
    now: datetime,
    session_ttl_seconds: int,
    response_text: str | None = None,
) -> ReminderReply | None:
    """End a pending snooze under lock, optionally recording a response."""

    return _end_pending_session(
        db,
        update,
        user=user,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        response_text=response_text,
    )


def _handle_completion_text(
    db: Session,
    update: TelegramUpdate,
    *,
    user: User,
    identity: ChannelIdentity,
    now: datetime,
    target_text: str | None = None,
) -> ReminderReply:
    """Resolve and complete the single delivered occurrence targeted by a reply."""
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
    if provider_id is not None:
        # A reply explicitly identifies the Telegram notification. Never fall
        # back to a different reminder when that reference is stale or unrelated.
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
    if provider_id is None and target_text and candidates:
        targeted = match_completion_candidates(target_text, candidates)
        if len(targeted) == 1:
            candidates = targeted
    log_event(
        logger,
        logging.DEBUG,
        "completion_target_resolution_finished",
        update_id=update.update_id,
        candidate_count=len(candidates),
        resolution="reply" if provider_id is not None else "latest_delivered",
    )
    if len(candidates) != 1:
        reply_text = (
            "Which delivered reminder did you complete?"
            if candidates
            else "I couldn't find a delivered reminder to mark as done."
        )
        db.rollback()
        if candidates:
            pending_completion = PendingCompletion(
                candidate_occurrence_ids=tuple(candidate[0].id for candidate in candidates)
            )
            _upsert_clarification_session(
                db,
                user_id=user.id,
                external_session_id=telegram_session_id(update.message.chat.id),
                pending=pending_completion,
                now=now,
            )
            log_event(
                logger,
                logging.INFO,
                "completion_clarification_saved",
                update_id=update.update_id,
                candidate_count=len(candidates),
            )
        log_event(
            logger,
            logging.INFO,
            "completion_target_ambiguous",
            update_id=update.update_id,
            candidate_count=len(candidates),
        )
        reply = _store_response(
            db,
            update=update,
            user_id=user.id,
            text=reply_text,
            response_kind="completion_ambiguous",
            operation="COMPLETE_REMINDER",
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
    reply = _store_response(
        db,
        update=update,
        user_id=user.id,
        text=result.text,
        response_kind="completion",
        operation="COMPLETE_REMINDER",
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


def handle_reminder_text(
    db: Session,
    update: TelegramUpdate,
    *,
    user_id: UUID,
    identity_id: UUID,
    telegram_user_id: int,
    interpreter: IntentInterpreter,
    intent_router: IntentRouter | None = None,
    now: datetime,
    session_ttl_seconds: int = 900,
) -> ReminderReply:
    """Interpret and transactionally create one or more reminders for a Telegram user."""
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
    if not _authorized_telegram_identity(
        user,
        identity,
        telegram_user_id=telegram_user_id,
        chat_id=update.message.chat.id,
    ):
        db.rollback()
        log_event(
            logger,
            logging.WARNING,
            "reminder_identity_resolution_failed",
            update_id=update.update_id,
            reason="missing_user_or_identity",
        )
        return ReminderReply(update.message.chat.id, "I couldn't verify your account.")

    # Completion is a deterministic acknowledgement of a delivered occurrence;
    # never send it through the natural-language extractor.
    text = (update.message.text or "").strip()
    if is_completion_command(text):
        pending_session = load_session(
            db,
            user_id=user.id,
            external_session_id=telegram_session_id(update.message.chat.id),
            now=now,
            ttl_seconds=session_ttl_seconds,
        )
        if pending_session is not None:
            _end_pending_session(
                db,
                update,
                user=user,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        log_event(
            logger,
            logging.INFO,
            "completion_route_selected",
            update_id=update.update_id,
            source="deterministic",
        )
        return _handle_completion_text(
            db,
            update,
            user=user,
            identity=identity,
            now=now,
        )
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

    session = load_session(
        db,
        user_id=user.id,
        external_session_id=telegram_session_id(update.message.chat.id),
        now=now,
        ttl_seconds=session_ttl_seconds,
    )
    session_is_expired = session is not None and session_expired(
        session, now=now, ttl_seconds=session_ttl_seconds
    )
    pending = None if session_is_expired else pending_from_session(session)
    pending_reschedule = None if session_is_expired else pending_reschedule_from_session(session)
    pending_cancellation = (
        None if session_is_expired else pending_cancellation_from_session(session)
    )
    pending_completion = (
        None if session_is_expired else pending_completion_from_session(session)
    )
    pending_snooze = (
        None if session_is_expired else pending_snooze_from_session(session)
    )
    text = update.message.text or ""
    recent_messages = _recent_messages(db, user_id=user.id)
    recent_delivered_reminders = _recent_delivered_reminders(
        db, user_id=user.id, identity_id=identity.id, chat_id=update.message.chat.id
    )
    log_event(
        logger,
        logging.INFO,
        "reminder_context_loaded",
        update_id=update.update_id,
        has_pending=pending is not None,
        session_expired=session_is_expired,
        pending_field_count=(
            sum(value is not None for value in pending.model_dump().values())
            if pending is not None
            else 0
        ),
        has_pending_cancellation=pending_cancellation is not None,
        has_pending_completion=pending_completion is not None,
        has_pending_snooze=pending_snooze is not None,
        recent_message_count=len(recent_messages),
    )
    observed_session_version = None if session_is_expired or session is None else session.version
    policy = assess_request(text, has_pending=pending is not None)
    if policy.is_rejected:
        log_event(
            logger,
            logging.WARNING,
            "reminder_request_rejected_by_policy",
            update_id=update.update_id,
            reason=policy.reason,
        )
        # Policy rejections are deliberately stateless. In particular, do not
        # create an IdempotencyRecord merely to remember a limitation response.
        # The webhook sends this reply directly and supported state-changing
        # requests continue to use durable idempotency below.
        db.rollback()
        return ReminderReply(update.message.chat.id, OUT_OF_SCOPE_RESPONSE)

    # A pending snooze remains in its original operation context unless the
    # user explicitly starts a new mutation or safely cancels it. Read-only
    # list/summary requests do not consume the pending question.
    normalized_text = re.sub(r"[.!?]+$", "", text.strip().casefold()).strip()
    if is_snooze_command(text) and pending_snooze is None and (
        pending is not None
        or pending_reschedule is not None
        or pending_cancellation is not None
        or pending_completion is not None
    ):
        _end_pending_session(
            db,
            update,
            user=user,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
        )
        pending = None
        pending_reschedule = None
        pending_cancellation = None
        pending_completion = None
        observed_session_version = None
    if pending_snooze is not None and normalized_text in {
        "cancel",
        "cancel it",
        "never mind",
        "nevermind",
        "not now",
    }:
        return _end_pending_snooze(
            db,
            update,
            user=user,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            response_text="Okay, I won't snooze that reminder.",
        )
    if pending_snooze is not None and (
        is_new_reminder_command(text)
        or is_reschedule_command(text)
        or is_cancel_command(text)
    ):
        _end_pending_snooze(
            db,
            update,
            user=user,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
        )
        pending_snooze = None
        observed_session_version = None
    if pending_snooze is not None and (
        is_list_request_command(text) or is_summary_request_command(text)
    ):
        pending_snooze = None

    # Snooze is an occurrence-scoped operation. Handle it before generic
    # reminder parsing so a delivered acknowledgement cannot become a new
    # reminder or a reschedule request.
    if is_snooze_command(text) or pending_snooze is not None:
        return _handle_snooze_text(
            db,
            update,
            user=user,
            identity=identity,
            interpreter=interpreter,
            now=now,
            pending=pending_snooze,
            recent_messages=recent_messages,
            session_ttl_seconds=session_ttl_seconds,
        )
    if pending_cancellation is not None and (
        is_list_request_command(text)
        or is_new_reminder_command(text)
        or is_reschedule_command(text)
    ):
        db.rollback()
        with db.begin():
            locked_session = load_session(
                db,
                user_id=user.id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if pending_cancellation_from_session(locked_session) is not None:
                end_session(locked_session, now=now)
        session = None
        pending_cancellation = None
        log_event(
            logger,
            logging.INFO,
            "cancellation_clarification_superseded",
            update_id=update.update_id,
        )
    if is_cancel_command(text) or pending_cancellation is not None:
        cancellation_reply = _handle_cancellation_text(
            db,
            update,
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=telegram_user_id,
            user=user,
            pending=pending_cancellation,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            cancellation_interpreter=interpreter,
            recent_messages=recent_messages,
        )
        if cancellation_reply is not None:
            return cancellation_reply
    if pending_reschedule is not None and (
        is_list_request_command(text) or is_summary_request_command(text)
    ):
        # A read-only schedule query is not an answer to the old reschedule
        # question. Leave that clarification available for a later reply and
        # let the normal listing flow handle this message.
        pending_reschedule = None
    elif pending_reschedule is not None and (
        is_new_reminder_command(text) or is_reschedule_command(text)
    ):
        # An explicit reminder or reschedule command starts a separate request.
        # It must not be treated as an ordinal/scope answer to an older
        # clarification, or that clarification will hijack the new request.
        db.rollback()
        with db.begin():
            locked_session = load_session(
                db,
                user_id=user.id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if pending_reschedule_from_session(locked_session) is not None:
                end_session(locked_session, now=now)
        session = None
        pending_reschedule = None
        observed_session_version = None
        log_event(
            logger,
            logging.INFO,
            "reschedule_clarification_superseded",
            update_id=update.update_id,
                replacement_route=(
                    "RESCHEDULE_REQUEST"
                    if is_reschedule_command(text)
                    else "NEW_REQUEST"
                ),
        )
    reschedule_reply = _handle_reschedule_text(
        db,
        update,
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=telegram_user_id,
        user=user,
        pending=pending_reschedule,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        reschedule_interpreter=interpreter,
        recent_messages=recent_messages,
        pending_session_version=observed_session_version,
    )
    if reschedule_reply is not None:
        return reschedule_reply
    if pending_completion is not None and (
        is_new_reminder_command(text)
        or is_reschedule_command(text)
        or is_list_request_command(text)
        or is_summary_request_command(text)
    ):
        db.rollback()
        with db.begin():
            locked_session = load_session(
                db,
                user_id=user.id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if pending_completion_from_session(locked_session) is not None:
                end_session(locked_session, now=now)
        pending_completion = None
        log_event(
            logger,
            logging.INFO,
            "completion_clarification_superseded",
            update_id=update.update_id,
            replacement_route=(
                "RESCHEDULE_REQUEST" if is_reschedule_command(text) else "NEW_REQUEST"
            ),
        )
    if pending_completion is not None:
        completion_reply = _handle_completion_follow_up(
            db,
            update,
            user=user,
            identity=identity,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
        )
        if completion_reply is not None:
            return completion_reply
    if session_is_expired:
        interpretation = None
    else:
        route, interpretation = _route_and_interpret(
            interpreter,
            text,
            timezone_name=timezone_name,
            now=now,
            pending=pending,
            user_name=user.name,
            intent_router=intent_router,
            recent_messages=recent_messages,
        )
        if route is ConversationRoute.RESCHEDULE_REQUEST:
            reschedule_reply = _handle_reschedule_text(
                db,
                update,
                user_id=user.id,
                identity_id=identity.id,
                telegram_user_id=telegram_user_id,
                user=user,
                pending=pending_reschedule,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                force=True,
                reschedule_interpreter=interpreter,
                recent_messages=recent_messages,
                pending_session_version=observed_session_version,
            )
            if reschedule_reply is not None:
                return reschedule_reply
        elif route is ConversationRoute.CANCEL_REQUEST:
            cancellation_reply = _handle_cancellation_text(
                db,
                update,
                user_id=user.id,
                identity_id=identity.id,
                telegram_user_id=telegram_user_id,
                user=user,
                pending=pending_cancellation,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                cancellation_interpreter=interpreter,
                recent_messages=recent_messages,
                force=True,
            )
            if cancellation_reply is not None:
                return cancellation_reply
        elif route is ConversationRoute.COMPLETE_REQUEST:
            # The classifier is only a routing hint. Completion still uses the
            # same delivered-notification resolution, authorization, transaction,
            # and idempotency path as deterministic commands.
            log_event(
                logger,
                logging.INFO,
                "completion_route_selected",
                update_id=update.update_id,
                source="llm_classifier",
            )
            db.rollback()
            return _handle_completion_text(
                db,
                update,
                user=user,
                identity=identity,
                now=now,
                target_text=interpretation.completion_target_text,
            )
        elif route is ConversationRoute.SNOOZE_REQUEST:
            return _handle_snooze_text(
                db,
                update,
                user=user,
                identity=identity,
                interpreter=interpreter,
                now=now,
                pending=pending_snooze,
                recent_messages=recent_messages,
                session_ttl_seconds=session_ttl_seconds,
                proposed_duration_seconds=interpretation.snooze_duration_seconds,
            )
    if interpretation is None:
        interpretation = Interpretation(
            response_text="That clarification expired. Please send the complete reminder again.",
            fallback_allowed=False,
        )
        route = ConversationRoute.UNRELATED
    log_event(
        logger,
        logging.INFO,
        "reminder_intent_interpreted",
        update_id=update.update_id,
        has_intent=bool(interpretation.resolved_intents),
        intent_count=len(interpretation.resolved_intents),
        needs_clarification=interpretation.clarification is not None,
        route=route,
        task_length=(
            len(getattr(interpretation.resolved_intents[0], "task", ""))
            if interpretation.resolved_intents
            else None
        ),
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
            if not _authorized_telegram_identity(
                user,
                identity,
                telegram_user_id=telegram_user_id,
                chat_id=update.message.chat.id,
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

            locked_session = load_session(
                db,
                user_id=user.id,
                external_session_id=telegram_session_id(update.message.chat.id),
                now=now,
                ttl_seconds=session_ttl_seconds,
                for_update=True,
            )
            if pending is not None and locked_session is None:
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text="That clarification expired. Please send the complete reminder again.",
                    response_kind="clarification_expired",
                )
            if session_is_expired and locked_session is None:
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text="That clarification expired. Please send the complete reminder again.",
                    response_kind="clarification_expired",
                )

            if (
                locked_session is not None
                and (
                    observed_session_version is None
                    or locked_session.version != observed_session_version
                )
            ):
                # Another update committed while this message was being
                # interpreted (potentially during a provider call).  Re-route
                # and re-interpret against the locked, latest session instead
                # of saving an interpretation derived from stale fields.
                pending = pending_from_session(locked_session)
                route, interpretation = _route_and_interpret(
                    interpreter,
                    text,
                    timezone_name=user.timezone,
                    now=now,
                    pending=pending,
                    user_name=user.name,
                    intent_router=intent_router,
                    recent_messages=_recent_messages(db, user_id=user.id),
                    recent_delivered_reminders=_recent_delivered_reminders(
                        db,
                        user_id=user.id,
                        identity_id=identity.id,
                        chat_id=update.message.chat.id,
                    ),
                )
                observed_session_version = locked_session.version
                log_event(
                    logger,
                    logging.INFO,
                    "reminder_session_reinterpreted",
                    update_id=update.update_id,
                    session_version=observed_session_version,
                    route=route,
                )

            if route in {ConversationRoute.UNRELATED, ConversationRoute.SOCIAL}:
                # Do not parse, extend, or overwrite a pending reminder for
                # conversational text that the router found unrelated.
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=interpretation.response_text
                    or "I didn't find a reminder request in that message.",
                    response_kind=("social" if route is ConversationRoute.SOCIAL else "unrelated"),
                )

            list_intents = [
                intent
                for intent in interpretation.resolved_intents
                if isinstance(intent, ListRemindersIntent)
            ]
            if list_intents:
                if len(list_intents) != 1 or len(interpretation.resolved_intents) != 1:
                    return ReminderReply(
                        update.message.chat.id,
                        "Please ask for one reminder date range at a time.",
                    )
                list_intent = list_intents[0]
                if list_intent.timezone_name != user.timezone:
                    return ReminderReply(
                        update.message.chat.id,
                        "I couldn't verify your reminder time zone. Please try again.",
                    )
                listed = list_reminders_in_range(
                    db,
                    user_id=user.id,
                    start_date=list_intent.start_date,
                    end_date=list_intent.end_date,
                    task_query=list_intent.task_query,
                    timezone_name=user.timezone,
                    now=now,
                )
                return ReminderReply(
                    update.message.chat.id,
                    format_reminder_list(
                        listed,
                        start_date=list_intent.start_date,
                        end_date=list_intent.end_date,
                        timezone_name=user.timezone,
                        task_query=list_intent.task_query,
                    ),
                )

            summary_intents = [
                intent
                for intent in interpretation.resolved_intents
                if isinstance(intent, ReminderSummaryIntent)
            ]
            if summary_intents:
                if len(summary_intents) != 1 or len(interpretation.resolved_intents) != 1:
                    return ReminderReply(
                        update.message.chat.id,
                        "Please ask one reminder progress question at a time.",
                    )
                summary_intent = summary_intents[0]
                if summary_intent.timezone_name != user.timezone:
                    return ReminderReply(
                        update.message.chat.id,
                        "I couldn't verify your reminder time zone. Please try again.",
                    )
                summary = summarize_reminders(
                    db,
                    user_id=user.id,
                    metric=summary_intent.metric,
                    timezone_name=user.timezone,
                    now=now,
                )
                return ReminderReply(
                    update.message.chat.id,
                    format_reminder_summary(summary, timezone_name=user.timezone),
                )

            if interpretation.clarification is not None:
                clarification = interpretation.clarification
                if locked_session is None:
                    locked_session = _upsert_clarification_session(
                        db,
                        user_id=user.id,
                        external_session_id=telegram_session_id(update.message.chat.id),
                        pending=clarification.pending,
                        now=now,
                    )
                else:
                    if locked_session.version != observed_session_version:
                        raise RuntimeError("conversation session version changed before save")
                    locked_session.pending_clarification = clarification.pending.model_dump(mode="json")
                    locked_session.status = AWAITING_CLARIFICATION
                    locked_session.active_reminder_id = None
                    locked_session.version += 1
                    locked_session.updated_at = _now_utc(now)
                log_event(
                    logger,
                    logging.INFO,
                    "clarification_session_saved",
                    update_id=update.update_id,
                    session_id=str(locked_session.id),
                    session_version=locked_session.version,
                    missing_task=clarification.pending.task is None,
                    missing_date=clarification.pending.date_local is None,
                    missing_time=clarification.pending.hour is None,
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=clarification.question,
                    response_kind="clarification",
                )

            intents = interpretation.resolved_intents
            if any(
                not is_supported_operation(getattr(intent, "operation", None))
                for intent in intents
            ):
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_operation_rejected_by_policy",
                    update_id=update.update_id,
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=OUT_OF_SCOPE_RESPONSE,
                    response_kind="unsupported_operation",
                )
            if not intents:
                if interpretation.clear_context and locked_session is not None:
                    end_session(locked_session, now=now)
                    return _store_response(
                        db,
                        update=update,
                        user_id=user.id,
                        text=interpretation.response_text
                        or "Okay, I won't create that reminder.",
                        response_kind="clarification_canceled",
                    )
                if locked_session is not None and locked_session.pending_clarification:
                    pending_session = pending_from_session(locked_session)
                    if pending_session is not None:
                        return _store_response(
                            db,
                            update=update,
                            user_id=user.id,
                            text="Please answer the pending clarification before creating a reminder.",
                            response_kind="clarification",
                        )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text=interpretation.response_text
                    or "Please tell me the reminder task and a future time.",
                    response_kind="clarification",
                )

            reference = _now_utc(now)
            try:
                zone = ZoneInfo(user.timezone)
                scheduled: list[
                    tuple[
                        CreateOneTimeReminderIntent | CreateRecurringReminderIntent,
                        datetime,
                        str,
                    ]
                ] = []
                for intent in intents:
                    if not isinstance(
                        intent, (CreateOneTimeReminderIntent, CreateRecurringReminderIntent)
                    ):
                        log_event(
                            logger,
                            logging.WARNING,
                            "reminder_intent_rejected",
                            update_id=update.update_id,
                            reason="invalid_typed_intent",
                        )
                        return _store_response(
                            db,
                            update=update,
                            user_id=user.id,
                            text="I couldn't understand that recurring reminder. Please provide the task, weekday, and time.",
                            response_kind="invalid_intent",
                        )
                    if intent.timezone_name != user.timezone:
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
                    if isinstance(intent, CreateRecurringReminderIntent):
                        recurrence = _recurrence_for_intent(intent)
                        local_due = resolve_local_datetime(
                            intent.start_local.date(),
                            intent.start_local.timetz(),
                            zone,
                        )
                        due_at = _now_utc(local_due)
                    else:
                        due_at = _now_utc(intent.due_at)
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
                            text="Please choose a future time for each reminder.",
                            response_kind="schedule_validation_failure",
                        )
                    scheduled.append((intent, due_at, _local_display(due_at, user.timezone)))
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
            except (TypeError, ValueError, ValidationError) as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_schedule_rejected",
                    update_id=update.update_id,
                    reason="invalid_recurring_intent",
                    error_type=type(exc).__name__,
                )
                return _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    text="I couldn't understand that recurring reminder. Please provide the task, weekday, and time.",
                    response_kind="invalid_intent",
                )

            if locked_session is not None:
                end_session(locked_session, now=now)

            confirmations: list[str] = []
            for intent, due_at, local_display in scheduled:
                if isinstance(intent, CreateRecurringReminderIntent):
                    recurrence = _recurrence_for_intent(intent)
                    reminder = Reminder(
                        user_id=user.id,
                        task=intent.task,
                        schedule_type=RECURRING,
                        start_local=intent.start_local.isoformat(timespec="minutes"),
                        timezone=user.timezone,
                        recurrence_rule=recurrence.canonical_rule,
                        status=ACTIVE,
                        version=1,
                    )
                    db.add(reminder)
                    db.flush()
                    materialize_recurring_reminder(db, reminder, now=reference, horizon_days=35)
                    if intent.frequency == "WEEKLY":
                        day_label = ", ".join(
                            ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[day]
                            for day in intent.weekdays
                        )
                        cadence = (
                            f"every {intent.interval} weeks on {day_label}"
                            if intent.interval != 1
                            else f"every {day_label}"
                        )
                    elif intent.frequency == "MONTHLY":
                        cadence = (
                            f"every {intent.interval} months on the "
                            f"{intent.day_of_month}{_ordinal_suffix(intent.day_of_month or 0)}"
                            if intent.interval != 1
                            else f"every month on the {intent.day_of_month}{_ordinal_suffix(intent.day_of_month or 0)}"
                        )
                    else:
                        cadence = (
                            f"every {intent.interval} years on {intent.month:02d}-{intent.day_of_month:02d}"
                            if intent.interval != 1
                            else f"every year on {intent.month:02d}-{intent.day_of_month:02d}"
                        )
                    confirmations.append(
                        f"I'll remind you to {reminder.task} {cadence} at "
                        f"{intent.hour:02d}:{intent.minute:02d} {user.timezone} via Telegram. "
                        f"Next: {local_display}."
                    )
                    continue
                reminder = Reminder(
                    user_id=user.id,
                    task=intent.task,
                    schedule_type=ONE_TIME,
                    start_local=due_at.astimezone(zone).isoformat(),
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
                log_event(
                    logger,
                    logging.INFO,
                    "reminder_notification_scheduled",
                    update_id=update.update_id,
                    reminder_id=str(reminder.id),
                    occurrence_id=str(occurrence.id),
                    due_at=due_at.isoformat(),
                    delivery_channel=TELEGRAM,
                )
                confirmations.append(
                    f"I'll remind you to {reminder.task} on {local_display} via Telegram."
                )
            text = "\n".join(confirmations)
            reply = _store_response(
                db,
                update=update,
                user_id=user.id,
                text=text,
                response_kind="confirmation",
            )
            log_event(
                logger,
                logging.INFO,
                "reminder_confirmation_staged",
                update_id=update.update_id,
                reminder_count=len(scheduled),
            )
        log_event(
            logger,
            logging.INFO,
            "reminder_transaction_committed",
            update_id=update.update_id,
            reminder_count=len(scheduled),
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
        intent_count=len(interpretation.resolved_intents),
        schedule_type=(
            RECURRING
            if any(
                isinstance(intent, CreateRecurringReminderIntent)
                for intent in interpretation.resolved_intents
            )
            else ONE_TIME
        ),
        notification_status=PENDING,
    )
    return reply
