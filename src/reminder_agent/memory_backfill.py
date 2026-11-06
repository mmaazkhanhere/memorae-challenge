"""Resumable command for indexing existing memory facts in bounded batches."""

from __future__ import annotations

import logging

from reminder_agent.db.session import SessionLocal
from reminder_agent.embeddings import EmbeddingPrivacyPolicy, create_embedding_provider
from reminder_agent.memories import backfill_memory_embeddings
from reminder_agent.observability import configure_logging, log_event
from reminder_agent.settings import get_settings


def run() -> None:
    settings = get_settings()
    logger = configure_logging(
        settings.log_level,
        settings.log_format,
        settings.log_include_traceback,
        settings.log_color,
    )
    provider = create_embedding_provider(settings)
    if provider is None:
        raise RuntimeError("Configure EMBEDDING_PROVIDER before running the backfill")
    privacy_policy = EmbeddingPrivacyPolicy(
        allow_sensitive_external=settings.embedding_allow_sensitive_external
    )
    with SessionLocal() as db, db.begin():
        changed = backfill_memory_embeddings(
            db,
            embedding_provider=provider,
            privacy_policy=privacy_policy,
            batch_size=100,
        )
    # Report only counts/model metadata, never fact or vector contents.
    log_event(
        logger,
        logging.INFO,
        "memory_embedding_backfill_completed",
        changed_count=changed,
        model=provider.model,
        dimensions=provider.dimensions,
    )
