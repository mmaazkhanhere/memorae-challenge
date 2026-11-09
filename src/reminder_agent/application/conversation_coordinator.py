from __future__ import annotations

import importlib
import logging
import re
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.contracts import ApplicationRequest
from reminder_agent.application.conversation import (
    process_authenticated_conversation,
)
from reminder_agent.application.conversations import (
    AWAITING_CLARIFICATION,
    end_session,
    load_session,
    pending_cancellation_from_session,
    pending_completion_from_session,
    pending_from_session,
    pending_memory_from_session,
    pending_memory_lookup_from_session,
    pending_reschedule_from_session,
    pending_snooze_from_session,
    session_expired,
    telegram_session_id,
)
from reminder_agent.application.conversations import (
    end_pending_session as _end_pending_session,
)
from reminder_agent.application.conversations import (
    upsert_clarification_session as _upsert_clarification_session,
)
from reminder_agent.application.response_staging import (
    ReminderReply,
    _application_request_from_telegram_update,
    _authorized_telegram_identity,
    _existing_record,
    _recover_from_idempotency_race,
    _reply_from_record,
    _store_response,
)
from reminder_agent.channels.telegram.cancellation import handle_cancellation_text
from reminder_agent.channels.telegram.reminder_listing import (
    handle_list_reminder_request,
)
from reminder_agent.features.intent.contracts import (
    IntentInterpreter,
    Interpretation,
    ReminderSummaryIntent,  # noqa: F401 - adapter compatibility export
)
from reminder_agent.features.intent.conversation_routing import (
    route_and_interpret as feature_route_and_interpret,
)
from reminder_agent.features.intent.model_routing import IntentRouter
from reminder_agent.features.intent.pending_updates import restrict_to_explicit_field_updates
from reminder_agent.features.intent.pipeline import ReminderIntentInterpreter
from reminder_agent.features.intent.request_policy import (
    OUT_OF_SCOPE_RESPONSE,
    assess_request,
    is_supported_operation,
)
from reminder_agent.features.intent.routing import (
    ConversationRoute,
    is_completion_command,
    is_list_request_command,
    is_new_reminder_command,
    is_reminder_status_question,
    is_summary_request_command,
)
from reminder_agent.features.memory.adapters import (
    _handle_memory_lookup_text,
    _handle_memory_text,
)
from reminder_agent.features.memory.memories import (
    MemoryRetrievalPolicy,
)
from reminder_agent.features.memory.memory_answering import (
    MemoryAnswerer,
)
from reminder_agent.features.reminders.cancel.decisions import (
    is_cancel_command,
)
from reminder_agent.features.reminders.cancel.service import (
    dispatch_cancellation,
)
from reminder_agent.features.reminders.complete.operations import (
    resolve_latest_delivered,
)
from reminder_agent.features.reminders.complete.service import (
    _handle_completion_follow_up,
    _handle_completion_text,
)
from reminder_agent.features.reminders.create.service import (
    handle_reminder_creation,
    log_reminder_creation_committed,
)
from reminder_agent.features.reminders.listing.filters import (
    list_owned_reminders,  # noqa: F401 - adapter compatibility export
)
from reminder_agent.features.reminders.listing.status import (
    answer_status_question_for_delivery,
)
from reminder_agent.features.reminders.reschedule.decisions import (
    is_reschedule_command,
    select_pending_candidate_ids,  # noqa: F401 - adapter compatibility export
)
from reminder_agent.features.reminders.reschedule.service import (
    _handle_reschedule_text,
    dispatch_reschedule,
)
from reminder_agent.features.reminders.snooze.operations import (
    is_snooze_command,
)
from reminder_agent.features.reminders.snooze.service import (
    _end_pending_snooze,
    _handle_snooze_text,
)
from reminder_agent.features.reminders.summary.telegram_adapter import (
    handle_reminder_summary_request,
)
from reminder_agent.infrastructure.db.models import ChannelIdentity, IdempotencyRecord, User
from reminder_agent.infrastructure.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
)
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.creation_routing import is_memory_reminder_collision
from reminder_agent.shared.idempotency import IdempotencyConflict  # noqa: F401 - compatibility export
from reminder_agent.shared.memory_contracts import (
    RetrieveMemoryQuery,  # noqa: F401 - adapter compatibility export
)
from reminder_agent.shared.memory_text import extract_memory_lookup_query

TELEGRAM = "TELEGRAM"
LINKED = "LINKED"
ACTIVE = "ACTIVE"
logger = get_logger()


# Compatibility wrapper; routing implementation lives in features.intent.
def _route_and_interpret(*args: Any, **kwargs: Any):
    """Preserve the old patch seam while delegating routing to the feature."""

    import reminder_agent.features.intent.conversation_routing as routing

    compatibility_restrict = restrict_to_explicit_field_updates
    try:
        compatibility_module = importlib.import_module(
            "reminder_agent.application.conversation_workflow"
        )
        compatibility_restrict = getattr(
            compatibility_module,
            "restrict_to_explicit_field_updates",
            compatibility_restrict,
        )
    except ImportError:
        pass
    routing.restrict_to_explicit_field_updates = compatibility_restrict
    return feature_route_and_interpret(*args, **kwargs)


def _recent_messages(db: Session, *, user_id: UUID) -> list[str]:
    """Build bounded conversation context from durable response records."""

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


def _recent_selection_context(
    db: Session, *, user_id: UUID
) -> tuple[dict[str, str], ...]:
    """Return the most recent structured reminder list for ordinal references."""

    records = db.scalars(
        select(IdempotencyRecord)
        .where(
            IdempotencyRecord.source == TELEGRAM,
            IdempotencyRecord.user_id == user_id,
        )
        .order_by(IdempotencyRecord.created_at.desc())
        .limit(1)
    ).all()
    for record in records:
        if not isinstance(record.response, dict):
            continue
        raw_context = record.response.get("selection_context")
        if not isinstance(raw_context, list):
            continue
        context = tuple(
            item
            for item in raw_context
            if isinstance(item, dict)
            and isinstance(item.get("reminder_id"), str)
            and isinstance(item.get("task"), str)
        )
        if context:
            return context
    return ()


def _recent_delivered_reminders(
    db: Session, *, user_id: UUID, identity_id: UUID, chat_id: int
) -> list[dict[str, str]]:
    """Build bounded, non-authoritative context for route classification."""

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


def _now_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _local_display(due_at: datetime, timezone_name: str) -> str:
    return due_at.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M %Z")






def _legacy_routing_tail() -> None:
    """Retained only as a temporary compatibility reference."""

    """

                    "Iâ€™m not sure whether you want a new reminder or to update your unfinished "
                    "one. Please say which reminder you mean."
                ),
                fallback_allowed=False,
            )
        log_event(
            logger,
            logging.INFO,
            "reminder_route_classifier_mapped",
            classifier_operation=suggestion,
            deterministic_route=deterministic_route,
            mapped_route=route,
            has_pending=pending is not None,
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
        interpretation = interpret_for_route(
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
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=IntentRoute.SNOOZE_REQUEST,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages or [],
        )
    elif route is ConversationRoute.RETRIEVE_MEMORY_REQUEST:
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=IntentRoute.RETRIEVE_MEMORY,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages or [],
        )
    elif route is ConversationRoute.SAVE_MEMORY_REQUEST:
        # Intent classification is still only a route proposal. On the model
        # fallback path, require typed, source-grounded classifier evidence;
        # the extractor and its content cannot authorize a write by themselves.
        evidence_grounded = not classifier_selected or is_memory_intent_evidence_grounded(
            text, classifier_memory_evidence
        )
        log_event(
            logger,
            logging.INFO,
            "memory_route_selected",
            source=("llm_classifier" if classifier_selected else "deterministic_router"),
            has_pending=pending is not None,
            memory_evidence_grounded=evidence_grounded,
            memory_evidence_kind=(
                getattr(classifier_memory_evidence, "kind", None) if classifier_selected else None
            ),
        )
        if not evidence_grounded:
            log_event(
                logger,
                logging.WARNING,
                "memory_route_rejected",
                reason="missing_or_ungrounded_memory_intent_evidence",
                source=("llm_classifier" if classifier_selected else "deterministic_router"),
            )
            return ConversationRoute.UNRELATED, Interpretation(
                response_text="What information would you like me to remember?",
                fallback_allowed=False,
            )
        interpretation = interpret_for_route(
            interpreter,
            text,
            selected_route=IntentRoute.SAVE_MEMORY,
            timezone_name=timezone_name,
            now=now,
            context=None,
            recent_messages=recent_messages or [],
        )
    else:
        selected_route = _selected_intent_route(route, pending=pending)
        interpretation = interpret_for_route(
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
    updates = interpretation.field_updates
    if updates is not None:
        if pending is None:
            interpretation = Interpretation(
                response_text="Please tell me the reminder task and a future time.",
                fallback_allowed=False,
            )
        else:
            proposed_fields = updates.supplied_fields()
            verified_updates = restrict_to_explicit_field_updates(
                text,
                updates,
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


    """

def handle_reminder_text(
    db: Session,
    update: Any,
    *,
    user_id: UUID,
    identity_id: UUID,
    telegram_user_id: int,
    interpreter: IntentInterpreter,
    intent_router: IntentRouter | None = None,
    now: datetime,
    session_ttl_seconds: int = 900,
    memory_answerer: MemoryAnswerer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy | None = None,
    memory_retrieval_policy: MemoryRetrievalPolicy | None = None,
    application_request: ApplicationRequest | None = None,
) -> ReminderReply:
    """Route one authenticated update and transactionally create reminders.

    The final mutation workflow owns the transaction for the update.
    """
    assert update.message is not None
    embedding_privacy_policy = embedding_privacy_policy or EmbeddingPrivacyPolicy()
    memory_retrieval_policy = memory_retrieval_policy or MemoryRetrievalPolicy()
    log_event(
        logger,
        logging.INFO,
        "reminder_command_started",
        update_id=update.update_id,
        input_length=len((update.message.text or "").strip()),
    )

    # Resolve the trusted sender and duplicate state before invoking a model. This avoids
    # holding a database transaction open during an external request.
    existing = _existing_record(db, update)
    if existing is not None:
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
    assert user is not None and identity is not None
    normalized_request = application_request or _application_request_from_telegram_update(
        update,
        user=user,
        identity=identity,
    )

    # Completion is a deterministic acknowledgement of a delivered occurrence;
    # never send it through the natural-language extractor.
    text = (update.message.text or "").strip()
    reply_to = update.message.reply_to_message
    if reply_to is not None and is_reminder_status_question(text):
        status_response = answer_status_question_for_delivery(
            db,
            user_id=user.id,
            identity_id=identity.id,
            provider_message_id=str(reply_to.message_id),
        )
        if status_response is not None:
            reply = _store_response(
                db,
                update=update,
                user_id=user.id,
                text=status_response,
                response_kind="reminder_status",
                operation="REMINDER_STATUS",
            )
            db.commit()
            log_event(
                logger,
                logging.INFO,
                "reminder_status_question_answered",
                update_id=update.update_id,
                provider_message_id=str(reply_to.message_id),
            )
            return reply
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
    pending_completion = None if session_is_expired else pending_completion_from_session(session)
    pending_snooze = None if session_is_expired else pending_snooze_from_session(session)
    pending_memory = None if session_is_expired else pending_memory_from_session(session)
    pending_memory_lookup = (
        None if session_is_expired else pending_memory_lookup_from_session(session)
    )
    text = update.message.text or ""
    recent_messages = _recent_messages(db, user_id=user.id)
    selection_context = _recent_selection_context(db, user_id=user.id)
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
        has_pending_memory=pending_memory is not None,
        has_pending_memory_lookup=pending_memory_lookup is not None,
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

    normalized_text = re.sub(r"[.!?]+$", "", text.strip().casefold()).strip()
    conversation_entry = process_authenticated_conversation(
        db,
        request=normalized_request,
        interpreter=interpreter,
        now=now,
        has_pending_clarification=session is not None,
        pending_reminder=pending,
        intent_router=intent_router,
        recent_messages=recent_messages,
    )
    if conversation_entry.response is not None:
        db.rollback()
        with db.begin():
            existing = _existing_record(db, update)
            if existing is not None:
                return _reply_from_record(existing)
            result = conversation_entry.response
            return _store_response(
                db,
                update=update,
                user_id=user.id,
                text=result.content,
                response_kind=result.response_kind,
                operation="LIST_REMINDERS",
                response_metadata=(
                    {"selection_context": list(result.selection_context)}
                    if result.selection_context
                    else None
                ),
            )
    assert conversation_entry.route is not None
    deterministic_route = conversation_entry.route
    log_event(
        logger,
        logging.INFO,
        "reminder_route_decision",
        update_id=update.update_id,
        route=deterministic_route,
        route_source="deterministic",
        has_pending=pending is not None,
        classifier_invoked=False,
    )

    # An ambiguous memory lookup keeps its own operation context. Read-only
    # reminder queries leave it available; explicit new operations safely
    # supersede it, and a plain follow-up narrows only its trusted candidates.
    if pending_memory_lookup is not None:
        if normalized_text in {
            "cancel",
            "cancel it",
            "never mind",
            "nevermind",
            "not now",
        }:
            return _end_pending_session(
                db,
                update,
                user=user,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                response_text="Okay, I won't look that up.",
                response_kind="memory_lookup_canceled",
                operation="RETRIEVE_MEMORY",
            ) or ReminderReply(update.message.chat.id, "Okay, I won't look that up.")
        if deterministic_route in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.SAVE_MEMORY_REQUEST,
            ConversationRoute.RESCHEDULE_REQUEST,
            ConversationRoute.CANCEL_REQUEST,
            ConversationRoute.SNOOZE_REQUEST,
        } or is_memory_reminder_collision(text):
            _end_pending_session(
                db,
                update,
                user=user,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
            pending_memory_lookup = None
        elif deterministic_route in {
            ConversationRoute.LIST_REQUEST,
            ConversationRoute.SUMMARY_REQUEST,
        }:
            pending_memory_lookup = None
        else:
            explicit_lookup = extract_memory_lookup_query(text)
            if explicit_lookup is None:
                return _handle_memory_lookup_text(
                    db,
                    update,
                    user=user,
                    pending=pending_memory_lookup,
                    query=text,
                    now=now,
                    session_ttl_seconds=session_ttl_seconds,
                    memory_answerer=memory_answerer,
                    embedding_provider=embedding_provider,
                    embedding_privacy_policy=embedding_privacy_policy,
                    memory_retrieval_policy=memory_retrieval_policy,
                )
            _end_pending_session(
                db,
                update,
                user=user,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
            pending_memory_lookup = None

    # A free-form narrowing reply must never be applied to candidates from an
    # expired lookup session. A complete, explicit lookup is a new read-only
    # request and may proceed independently after the old session is ended.
    expired_memory_lookup = (
        session_is_expired
        and session is not None
        and isinstance(session.pending_clarification, dict)
        and session.pending_clarification.get("operation") == "RETRIEVE_MEMORY"
    )
    if (
        expired_memory_lookup
        and deterministic_route is not ConversationRoute.RETRIEVE_MEMORY_REQUEST
    ):
        return _end_pending_session(
            db,
            update,
            user=user,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            response_text="That memory lookup expired. Please ask what you want to look up again.",
            response_kind="memory_lookup_clarification_expired",
            operation="RETRIEVE_MEMORY",
        ) or ReminderReply(
            update.message.chat.id,
            "That memory lookup expired. Please ask what you want to look up again.",
        )

    if deterministic_route is ConversationRoute.RETRIEVE_MEMORY_REQUEST:
        if pending_memory is not None:
            _end_pending_session(
                db,
                update,
                user=user,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
        extracted_query = extract_memory_lookup_query(text)
        log_event(
            logger,
            logging.INFO,
            "memory_lookup_query_extracted",
            update_id=update.update_id,
            extraction_source="deterministic",
            query_present=extracted_query is not None,
            query_length=len(extracted_query or ""),
            classifier_invoked=False,
        )
        log_event(
            logger,
            logging.INFO,
            "memory_lookup_classifier_skipped",
            update_id=update.update_id,
            reason="deterministic_retrieve_memory_route",
        )
        return _handle_memory_lookup_text(
            db,
            update,
            user=user,
            pending=None,
            query=extracted_query,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            memory_answerer=memory_answerer,
            embedding_provider=embedding_provider,
            embedding_privacy_policy=embedding_privacy_policy,
            memory_retrieval_policy=memory_retrieval_policy,
        )

    # A stale SAVE_MEMORY draft must not consume an arbitrary next statement as
    # its fact. Explicit memory and reminder commands remain new operations.
    expired_memory_pending = (
        session_is_expired
        and session is not None
        and isinstance(session.pending_clarification, dict)
        and session.pending_clarification.get("operation") == "SAVE_MEMORY"
    )
    if expired_memory_pending and deterministic_route not in {
        ConversationRoute.SAVE_MEMORY_REQUEST,
        ConversationRoute.NEW_REQUEST,
    }:
        return _end_pending_session(
            db,
            update,
            user=user,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            response_text="That memory request expired. Please send the complete request again.",
            response_kind="memory_clarification_expired",
            operation="SAVE_MEMORY",
        ) or ReminderReply(
            update.message.chat.id,
            "That memory request expired. Please send the complete request again.",
        )

    # A pending memory accepts a plain fact as its answer. Explicit commands
    # continue through the shared route decision so they can supersede this
    # operation safely instead of being silently saved as content.
    if pending_memory is not None:
        if normalized_text in {
            "cancel",
            "cancel it",
            "never mind",
            "nevermind",
            "not now",
        }:
            return _end_pending_session(
                db,
                update,
                user=user,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                response_text="Okay, I won't save anything.",
                response_kind="memory_canceled",
                operation="SAVE_MEMORY",
            ) or ReminderReply(update.message.chat.id, "Okay, I won't save anything.")

        if deterministic_route in {
            ConversationRoute.NEW_REQUEST,
            ConversationRoute.RESCHEDULE_REQUEST,
            ConversationRoute.CANCEL_REQUEST,
            ConversationRoute.SNOOZE_REQUEST,
        } or is_memory_reminder_collision(text):
            _end_pending_session(
                db,
                update,
                user=user,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
            )
            pending_memory = None
        elif deterministic_route not in {
            ConversationRoute.LIST_REQUEST,
            ConversationRoute.SUMMARY_REQUEST,
        }:
            return _handle_memory_text(
                db,
                update,
                user=user,
                identity=identity,
                pending=pending_memory,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                embedding_provider=embedding_provider,
                embedding_privacy_policy=embedding_privacy_policy,
            )

    # A pending snooze remains in its original operation context unless the
    # user explicitly starts a new mutation or safely cancels it. Read-only
    # list/summary requests do not consume the pending question.
    if (
        is_snooze_command(text)
        and pending_snooze is None
        and (
            pending is not None
            or pending_reschedule is not None
            or pending_cancellation is not None
            or pending_completion is not None
        )
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
        canceled_reply = _end_pending_snooze(
            db,
            update,
            user=user,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            response_text="Okay, I won't snooze that reminder.",
        )
        assert canceled_reply is not None
        return canceled_reply
    if pending_snooze is not None and (
        is_new_reminder_command(text) or is_reschedule_command(text) or is_cancel_command(text)
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
    cancellation_dispatch = dispatch_cancellation(
        db,
        user_id=user.id,
        external_session_id=telegram_session_id(update.message.chat.id),
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        update_id=update.update_id,
        pending=pending_cancellation,
        session=session,
        is_list_request=is_list_request_command(text),
        is_new_reminder=is_new_reminder_command(text),
        is_reschedule=is_reschedule_command(text),
        is_cancel_command=is_cancel_command(text),
        handle=lambda pending_value: handle_cancellation_text(
            db,
            update,
            user_id=cast(User, user).id,
            user=cast(User, user),
            identity=cast(ChannelIdentity, identity),
            pending=pending_value,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            cancellation_interpreter=interpreter,
            recent_messages=recent_messages,
        ),
    )
    pending_cancellation = cancellation_dispatch.pending
    session = cancellation_dispatch.session
    if cancellation_dispatch.response is not None:
        return cancellation_dispatch.response

    reschedule_dispatch = dispatch_reschedule(
        db,
        user_id=user.id,
        external_session_id=telegram_session_id(update.message.chat.id),
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        update_id=update.update_id,
        pending=pending_reschedule,
        session=session,
        observed_session_version=observed_session_version,
        is_list_request=is_list_request_command(text),
        is_summary_request=is_summary_request_command(text),
        is_new_reminder=is_new_reminder_command(text),
        is_reschedule=is_reschedule_command(text),
        handle=lambda pending_value, session_version: _handle_reschedule_text(
            db,
            update,
            user_id=cast(User, user).id,
            identity_id=cast(ChannelIdentity, identity).id,
            telegram_user_id=telegram_user_id,
            user=cast(User, user),
            pending=pending_value,
            now=now,
            session_ttl_seconds=session_ttl_seconds,
            defer_to_classifier=intent_router is not None,
            reschedule_interpreter=cast(ReminderIntentInterpreter, interpreter),
            recent_messages=recent_messages,
            selection_context=selection_context,
            pending_session_version=session_version,
        ),
    )
    pending_reschedule = reschedule_dispatch.pending
    session = reschedule_dispatch.session
    observed_session_version = reschedule_dispatch.observed_session_version
    reschedule_reply = reschedule_dispatch.response
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
                assert locked_session is not None
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
            recent_delivered_reminders=recent_delivered_reminders,
            selection_context=list(selection_context),
            deterministic_route=deterministic_route,
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
                reschedule_interpreter=cast(ReminderIntentInterpreter, interpreter),
                recent_messages=recent_messages,
                selection_context=selection_context,
                pending_session_version=observed_session_version,
            )
            if reschedule_reply is not None:
                return reschedule_reply
        elif route is ConversationRoute.CANCEL_REQUEST:
            cancellation_reply = handle_cancellation_text(
                db,
                update,
                user_id=user.id,
                user=user,
                identity=identity,
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
        elif route is ConversationRoute.RETRIEVE_MEMORY_REQUEST:
            return _handle_memory_lookup_text(
                db,
                update,
                user=user,
                pending=None,
                query=(
                    interpretation.memory_query.query
                    if interpretation.memory_query is not None
                    else None
                ),
                from_model=deterministic_route is ConversationRoute.MODEL_FALLBACK,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                memory_answerer=memory_answerer,
                embedding_provider=embedding_provider,
                embedding_privacy_policy=embedding_privacy_policy,
                memory_retrieval_policy=memory_retrieval_policy,
            )
        elif route is ConversationRoute.SAVE_MEMORY_REQUEST:
            # A classifier-selected memory request is a new operation. End any
            # older pending reminder context, then use the same canonical-user,
            # transaction, and idempotency path as deterministic memory saves.
            if (
                pending is not None
                or pending_reschedule is not None
                or pending_cancellation is not None
                or pending_completion is not None
                or pending_snooze is not None
            ):
                _end_pending_session(
                    db,
                    update,
                    user=user,
                    now=now,
                    session_ttl_seconds=session_ttl_seconds,
                )
            memory_fact = interpretation.memory_fact
            log_event(
                logger,
                logging.INFO,
                "memory_execution_selected",
                route=route,
                has_memory_fact=memory_fact is not None,
                has_clarification=interpretation.response_text is not None,
            )
            return _handle_memory_text(
                db,
                update,
                user=user,
                identity=identity,
                pending=None,
                now=now,
                session_ttl_seconds=session_ttl_seconds,
                model_content=memory_fact.content if memory_fact is not None else None,
                from_model=deterministic_route is ConversationRoute.MODEL_FALLBACK,
                clarification_text=interpretation.response_text,
                embedding_provider=embedding_provider,
                embedding_privacy_policy=embedding_privacy_policy,
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
            assert user is not None and identity is not None
            timezone_name = user.timezone
            assert timezone_name is not None

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

            if locked_session is not None and (
                observed_session_version is None
                or locked_session.version != observed_session_version
            ):
                # Another update committed while this message was being
                # interpreted (potentially during a provider call).  Re-route
                # and re-interpret against the locked, latest session instead
                # of saving an interpretation derived from stale fields.
                pending = pending_from_session(locked_session)
                route, interpretation = _route_and_interpret(
                    interpreter,
                    text,
                    timezone_name=timezone_name,
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

            list_reply = handle_list_reminder_request(
                db,
                update,
                user=user,
                identity=identity,
                interpretation=interpretation,
                now=now,
            )
            if list_reply is not None:
                return list_reply

            summary_reply = handle_reminder_summary_request(
                db,
                update,
                user=user,
                identity=identity,
                interpretation=interpretation,
                now=now,
            )
            if summary_reply is not None:
                return summary_reply

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
                    locked_session.pending_clarification = clarification.pending.model_dump(
                        mode="json"
                    )
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
                not is_supported_operation(getattr(intent, "operation", None)) for intent in intents
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
                        text=interpretation.response_text or "Okay, I won't create that reminder.",
                        response_kind="clarification_canceled",
                    )
                if locked_session is not None and locked_session.pending_clarification:
                    pending_draft = pending_from_session(locked_session)
                    if pending_draft is not None:
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
            creation = handle_reminder_creation(
                db,
                intents=list(intents),
                user=user,
                identity=identity,
                locked_session=locked_session,
                timezone_name=timezone_name,
                now=now,
                reference=reference,
                source_message_id=str(update.update_id),
                update_id=update.update_id,
                stage_response=lambda **kwargs: _store_response(
                    db,
                    update=update,
                    user_id=user.id,
                    **kwargs,
                ),
            )
            reply = creation.response
            reminder_count = creation.reminder_count
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

    log_reminder_creation_committed(
        update_id=update.update_id,
        reminder_count=reminder_count,
        intents=interpretation.resolved_intents,
    )
    return reply
