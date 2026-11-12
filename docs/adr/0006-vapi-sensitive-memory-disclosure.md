# Vapi Sensitive Memory Disclosure

Status: Accepted
Date: 2026-11-10

## Context

The memory subsystem originally denied external embedding of sensitive facts and
prevented Vapi from speaking them aloud. That default protected passwords and
other secrets, but it also meant that an explicitly authenticated caller could
not retrieve a saved Wi-Fi password through the voice channel.

The project owner explicitly requested that authenticated Vapi callers be able
to retrieve and hear sensitive memories.

## Decision

Authenticated Vapi memory operations may embed and disclose sensitive facts.
The Vapi workflow opts into sensitive external embedding and passes sensitive
candidate facts through the shared, owner-scoped retrieval and answer-validation
pipeline. Telegram keeps the existing deny-by-default behavior.

All existing authorization, ownership, deterministic-first routing, idempotency,
and exact-value validation rules remain unchanged. Sensitive values are still
excluded from durable Vapi idempotency response payloads and application logs.

Existing facts with `SKIPPED_SENSITIVE` embeddings must be reindexed with the
approved embedding provider before semantic Vapi lookup can use them; exact
lexical lookup remains available without backfill.

## Alternatives Considered

- Keep sensitive facts unavailable over Vapi. This preserves the previous
  privacy boundary but does not meet the requested voice behavior.
- Allow sensitive lexical retrieval but never send sensitive text to the
  embedding provider. This avoids external disclosure but provides weaker
  paraphrase retrieval and was not the requested behavior.
- Allow sensitive disclosure globally across every channel. This broadens the
  change unnecessarily, so the opt-in remains limited to authenticated Vapi
  memory operations.

## Consequences

Positive consequences:

- Authenticated callers can retrieve saved passwords and other sensitive facts
  through Vapi.
- New sensitive facts can participate in semantic retrieval when indexing is
  configured and permitted.
- Telegram behavior remains unchanged.

Trade-offs:

- Sensitive memory content is sent to the configured external embedding provider
  for Vapi saves and semantic lookups.
- Operators must explicitly backfill existing `SKIPPED_SENSITIVE` rows if they
  need semantic retrieval for those facts.

## Implementation References

- `src/reminder_agent/application/vapi_memory_workflow.py`
- `src/reminder_agent/features/memory/retrieve/workflow.py`
- `src/reminder_agent/features/memory/retrieve/persistence.py`
- `src/reminder_agent/features/memory/persistence.py`
- `tests/unit/test_vapi_adapter.py`

## Relationship to Original Architecture

This decision is a scoped exception to the deny-by-default embedding and
sensitive-candidate filtering described in [Architecture.md](../Architecture.md)
section 9.11. It applies only to authenticated Vapi memory operations and does
not change Telegram behavior or the mandatory deterministic-first pipeline.
