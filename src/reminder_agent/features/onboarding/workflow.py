from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.application.completion_callbacks import (
    authorized_completion_identity,
    parse_completion_callback,
)
from reminder_agent.application.contracts import ApplicationRequest
from reminder_agent.application.voice_transcription import transcribe_telegram_voice
from reminder_agent.features.intent.model_routing import IntentRouter
from reminder_agent.features.intent.pipeline import ReminderIntentInterpreter
from reminder_agent.features.memory.memories import MemoryRetrievalPolicy
from reminder_agent.features.memory.memory_answering import MemoryAnswerer
from reminder_agent.features.onboarding.handlers import (
    _is_link_phone,
    _is_skip_phone,
    _is_start,
    _phone_region_from_text,
)
from reminder_agent.features.onboarding.handlers import (
    handle_contact as _handle_contact,
)
from reminder_agent.features.onboarding.handlers import (
    handle_link_phone as _handle_link_phone,
)
from reminder_agent.features.onboarding.handlers import (
    handle_phone_region as _handle_phone_region,
)
from reminder_agent.features.onboarding.handlers import (
    handle_skip_phone as _handle_skip_phone,
)
from reminder_agent.features.onboarding.handlers import (
    handle_start as _handle_start,
)
from reminder_agent.features.onboarding.message import handle_message as _handle_message
from reminder_agent.features.onboarding.onboarding_validation import (
    InvalidPhoneNumber,  # noqa: F401 - legacy public re-export
    InvalidPhoneRegion,  # noqa: F401 - legacy public re-export
    InvalidTimezone,  # noqa: F401 - legacy public re-export
    normalize_phone_number,  # noqa: F401 - legacy public re-export
    validate_phone_region,  # noqa: F401 - legacy public re-export
    validate_timezone,  # noqa: F401 - legacy public re-export
)
from reminder_agent.features.onboarding.request_contracts import (
    OnboardingAction,
    OnboardingRequest,
)
from reminder_agent.features.onboarding.responses import (
    OnboardingReply,
)
from reminder_agent.features.onboarding.responses import (
    claim_delivery as claim_delivery,  # noqa: F401 - compatibility export
)
from reminder_agent.features.onboarding.responses import (
    existing_record as _existing_record,
)
from reminder_agent.features.onboarding.responses import (
    find_sender_identity as _find_identity,
)
from reminder_agent.features.onboarding.responses import (
    record_delivery_outcome as record_delivery_outcome,  # noqa: F401 - compatibility export
)
from reminder_agent.features.onboarding.responses import (
    reply_from_record as _reply_from_record,
)
from reminder_agent.features.onboarding.responses import (
    store_result as _store_result,
)
from reminder_agent.features.reminders.completion import complete_occurrence
from reminder_agent.infrastructure.db.models import User
from reminder_agent.infrastructure.embeddings import EmbeddingPrivacyPolicy, EmbeddingProvider
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.infrastructure.transcription import AudioTranscriber
from reminder_agent.shared.idempotency import IdempotencyConflict as _IdempotencyConflict

ACTIVE = "ACTIVE"

logger = get_logger()
IdempotencyConflict = _IdempotencyConflict


def handle_update(
    db: Session,
    update: Any,
    *,
    default_region: str | None = None,
    reminder_interpreter: ReminderIntentInterpreter | None = None,
    intent_router: IntentRouter | None = None,
    memory_answerer: MemoryAnswerer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy | None = None,
    memory_retrieval_policy: MemoryRetrievalPolicy | None = None,
    now: datetime | None = None,
    session_ttl_seconds: int = 900,
    callback_secret: str = "development",
    telegram_client: Any = None,
    transcriber: AudioTranscriber | None = None,
    voice_max_bytes: int = 20 * 1024 * 1024,
    voice_max_duration_seconds: int = 300,
    log_voice_transcript: bool = False,
    application_request: ApplicationRequest | None = None,
    onboarding_request: OnboardingRequest | None = None,
) -> OnboardingReply | None:
    """Apply one Telegram update and return its committed response, if any.

    Branch handlers own their short write transactions. The completion callback
    branch uses the explicit session commit below so the occurrence transition
    and idempotency response remain one transaction.
    """
    if update.callback_query is not None and update.callback_query.message is not None:
        callback = parse_completion_callback(update, callback_secret=callback_secret)
        if callback is None:
            return None
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
        identity = authorized_completion_identity(db, callback)
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
        log_event(
            logger,
            logging.DEBUG,
            "completion_callback_authorized",
            update_id=update.update_id,
            occurrence_id=str(callback.occurrence_id),
            user_id=str(identity.user_id),
            identity_id=str(identity.id),
        )
        result = complete_occurrence(
            db,
            user_id=identity.user_id,
            identity_id=identity.id,
            chat_id=callback.chat_id,
            occurrence_id=callback.occurrence_id,
            provider_message_id=callback.provider_message_id,
        )
        log_event(
            logger,
            logging.INFO,
            "completion_callback_service_result",
            update_id=callback.update_id,
            occurrence_id=str(result.occurrence_id)
            if result.occurrence_id
            else str(callback.occurrence_id),
            outcome=result.outcome,
        )
        reply = _store_result(
            db,
            update=update,
            user_id=identity.user_id,
            operation="COMPLETE_REMINDER",
            text=result.text,
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
                    occurrence_id=str(callback.occurrence_id),
                    reason="idempotency_race",
                )
                return _reply_from_record(duplicate)
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
    action = onboarding_request.action if onboarding_request is not None else None
    if action == OnboardingAction.CONTACT or (
        action is None and update.message.contact is not None
    ):
        return _handle_contact(db, update, default_region=default_region)
    if action == OnboardingAction.START or (action is None and _is_start(update)):
        return _handle_start(db, update)
    if action == OnboardingAction.LINK_PHONE or (action is None and _is_link_phone(update)):
        return _handle_link_phone(db, update)
    if action == OnboardingAction.SKIP_PHONE or (action is None and _is_skip_phone(update)):
        return _handle_skip_phone(db, update)
    phone_region = (
        onboarding_request.argument
        if action == OnboardingAction.PHONE_REGION
        else _phone_region_from_text(update)
        if action is None
        else None
    )
    if phone_region is not None:
        return _handle_phone_region(db, update, phone_region)
    if action == OnboardingAction.VOICE or (action is None and update.message.voice is not None):
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
            application_request=application_request,
    )


def _handle_voice_message(
    db: Session,
    update: Any,
    *,
    telegram_client: Any,
    transcriber: AudioTranscriber | None,
    max_bytes: int,
    max_duration_seconds: int,
    log_transcript: bool = False,
    reminder_interpreter: ReminderIntentInterpreter | None = None,
    intent_router: IntentRouter | None = None,
    memory_answerer: MemoryAnswerer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy | None = None,
    memory_retrieval_policy: MemoryRetrievalPolicy | None = None,
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
        log_event(
            logger,
            logging.WARNING,
            "telegram_voice_rejected",
            update_id=update.update_id,
            reason="unknown_sender",
        )
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
        log_event(
            logger,
            logging.INFO,
            "telegram_voice_rejected",
            update_id=update.update_id,
            reason="onboarding_incomplete",
        )
        with db.begin():
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="VOICE_NOTE_REQUIRES_ONBOARDING",
                text="Please finish Telegram onboarding with your IANA time zone first.",
            )

    voice_result = transcribe_telegram_voice(
        voice=update.message.voice,
        telegram_client=telegram_client,
        transcriber=transcriber,
        update_id=update.update_id,
        max_bytes=max_bytes,
        max_duration_seconds=max_duration_seconds,
        log_transcript=log_transcript,
    )
    if not voice_result.succeeded:
        assert voice_result.failure_text is not None
        return _store_voice_failure(db, update, user.id, voice_result.failure_text)

    # Keep the original voice metadata and update digest while exposing the
    # transient transcript to the already-tested text command pipeline.
    command_update = update.model_copy(
        deep=True,
        update={
            "message": update.message.model_copy(update={"text": voice_result.transcript}),
        },
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
    db: Session, update: Any, user_id: UUID | None, text: str
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


# Telegram response and delivery exports remain compatibility imports above.


