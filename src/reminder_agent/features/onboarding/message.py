"""Normalized onboarding-message branch orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import ApplicationRequest
from reminder_agent.features.intent.contracts import IntentInterpreter
from reminder_agent.features.intent.model.router import IntentRouter
from reminder_agent.features.intent.pipeline import ReminderIntentInterpreter
from reminder_agent.features.memory.retrieve.answering import MemoryAnswerer
from reminder_agent.features.memory.retrieve.domain import MemoryRetrievalPolicy
from reminder_agent.features.onboarding import message_templates as messages
from reminder_agent.features.onboarding.onboarding_validation import (
    InvalidTimezone,
    validate_timezone,
)
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
from reminder_agent.infrastructure.db.models import User
from reminder_agent.infrastructure.embeddings import EmbeddingPrivacyPolicy, EmbeddingProvider

ACTIVE = "ACTIVE"


def handle_message(
    db: Session,
    update: Any,
    *,
    reminder_interpreter: ReminderIntentInterpreter | None = None,
    intent_router: IntentRouter | None = None,
    memory_answerer: MemoryAnswerer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy | None = None,
    memory_retrieval_policy: MemoryRetrievalPolicy | None = None,
    now: datetime | None = None,
    session_ttl_seconds: int = 900,
    application_request: ApplicationRequest | None = None,
) -> OnboardingReply:
    """Handle free text for onboarding or hand active users to conversation."""

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
                text=messages.unknown_sender(),
            )

    user = db.get(User, identity.user_id)
    assert user is not None
    if user.onboarding_status == ACTIVE:
        db.rollback()
        from reminder_agent.application.conversation_workflow import handle_reminder_text

        return cast(
            OnboardingReply,
            handle_reminder_text(
                db,
                update,
                user_id=user.id,
                identity_id=identity.id,
                telegram_user_id=update.message.from_user.id,
                interpreter=cast(
                    IntentInterpreter,
                    reminder_interpreter or ReminderIntentInterpreter(),
                ),
                intent_router=intent_router,
                memory_answerer=memory_answerer,
                embedding_provider=embedding_provider,
                embedding_privacy_policy=embedding_privacy_policy,
                memory_retrieval_policy=memory_retrieval_policy,
                now=now or datetime.now(UTC),
                session_ttl_seconds=session_ttl_seconds,
                application_request=application_request,
            ),
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
            text = messages.invalid_timezone()
            return _store_result(
                db,
                update=update,
                user_id=user.id,
                operation="ONBOARDING_INVALID_TIMEZONE",
                text=text,
            )

        user.timezone = timezone_name
        user.onboarding_status = ACTIVE
        text = messages.onboarding_complete(user.name, timezone_name)
        return _store_result(
            db,
            update=update,
            user_id=user.id,
            operation="ONBOARDING_COMPLETE",
            text=text,
            presentation_hint="PHONE_CONTACT_REQUEST",
        )


__all__ = ["handle_message"]


