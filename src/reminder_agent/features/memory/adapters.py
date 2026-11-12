"""Memory save and lookup adapters for the authenticated conversation flow."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from reminder_agent.application.conversations import (
    telegram_session_id,
    upsert_clarification_session,
)
from reminder_agent.application.response_staging import (
    ReminderReply,
    TelegramMemoryLookupResponseAdapter,
    TelegramMemoryResponseAdapter,
    _application_request_from_telegram_update,
    _existing_record,
    _recover_from_idempotency_race,
    _reply_from_record,
    _store_response,
)
from reminder_agent.features.memory.persistence import DEFAULT_EMBEDDING_PRIVACY_POLICY
from reminder_agent.features.memory.retrieve.answering import (
    MemoryAnswerer,
    memory_answer_preserves_exact_values,
)
from reminder_agent.features.memory.retrieve.contracts import (
    DEFAULT_MEMORY_RETRIEVAL_POLICY,
    MemoryRetrievalPolicy,
)
from reminder_agent.features.memory.retrieve.persistence import lookup_memory_fact
from reminder_agent.features.memory.retrieve.workflow import run_memory_lookup_workflow
from reminder_agent.features.memory.saving import run_memory_save_workflow, save_memory_for_request
from reminder_agent.infrastructure.database.models import ChannelIdentity, User
from reminder_agent.infrastructure.providers.embeddings import (
    EmbeddingPrivacyPolicy,
    EmbeddingProvider,
    contains_sensitive_memory,
)
from reminder_agent.infrastructure.observability import log_event
from reminder_agent.shared.memory_contracts import PendingMemory, PendingMemoryLookup


def _handle_memory_text(
    db: Session,
    update: Any,
    *,
    user: User,
    identity: ChannelIdentity | None,
    pending: PendingMemory | None,
    now: datetime,
    session_ttl_seconds: int,
    model_content: str | None = None,
    from_model: bool = False,
    clarification_text: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
) -> ReminderReply:
    """Adapt an authenticated Telegram update into the memory workflow."""

    assert update.message is not None
    if identity is None:
        return ReminderReply(update.message.chat.id, "I couldn't verify your account.")
    request = _application_request_from_telegram_update(
        update,
        user=user,
        identity=identity,
    )
    response_adapter = TelegramMemoryResponseAdapter(
        db=db,
        update=update,
        recover_race_callback=lambda: _recover_from_idempotency_race(db, update),
        # Keep the established reminders.py monkeypatch seams for callers and
        # characterization tests while the adapter owns response formatting.
        find_record=_existing_record,
        record_reply=_reply_from_record,
        store_response=_store_response,
    )
    return run_memory_save_workflow(
        db,
        request=request,
        user_id=user.id,
        pending=pending,
        text=update.message.text or "",
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        responses=response_adapter,
        upsert_clarification_session=upsert_clarification_session,
        save_memory_operation=save_memory_for_request,
        model_content=model_content,
        from_model=from_model,
        clarification_text=clarification_text,
        embedding_provider=embedding_provider,
        embedding_privacy_policy=embedding_privacy_policy,
        log_event_fn=log_event,
        event_context={"update_id": update.update_id},
    )


def _handle_memory_lookup_text(
    db: Session,
    update: Any,
    *,
    user: User,
    pending: PendingMemoryLookup | None,
    query: str | None,
    clarification_text: str | None = None,
    from_model: bool = False,
    now: datetime,
    session_ttl_seconds: int,
    memory_answerer: MemoryAnswerer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_privacy_policy: EmbeddingPrivacyPolicy = DEFAULT_EMBEDDING_PRIVACY_POLICY,
    memory_retrieval_policy: MemoryRetrievalPolicy = DEFAULT_MEMORY_RETRIEVAL_POLICY,
) -> ReminderReply:
    """Adapt a Telegram update into the shared memory lookup workflow."""

    assert update.message is not None
    responses = TelegramMemoryLookupResponseAdapter(
        db=db,
        update=update,
        find_record=_existing_record,
        record_reply=_reply_from_record,
    )
    return run_memory_lookup_workflow(
        db,
        user_id=user.id,
        external_session_id=telegram_session_id(update.message.chat.id),
        pending=pending,
        query=query,
        clarification_text=clarification_text,
        source_text=update.message.text or "",
        from_model=from_model,
        now=now,
        session_ttl_seconds=session_ttl_seconds,
        responses=responses,
        upsert_clarification_session=upsert_clarification_session,
        memory_answerer=memory_answerer,
        embedding_provider=embedding_provider,
        embedding_privacy_policy=embedding_privacy_policy,
        memory_retrieval_policy=memory_retrieval_policy,
        lookup_memory_operation=lookup_memory_fact,
        contains_sensitive_memory_fn=contains_sensitive_memory,
        answer_preserves_exact_values_fn=memory_answer_preserves_exact_values,
        log_event_fn=log_event,
        event_context={"update_id": update.update_id},
    )
