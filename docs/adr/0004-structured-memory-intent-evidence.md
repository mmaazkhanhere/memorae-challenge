# ADR 0004: Structured Memory-Intent Evidence

## Status

Accepted

## Date

2026-11-06

This is the ADR documentation and acceptance date, not necessarily the original implementation decision date.

## Context

VS-13 used a deterministic English phrase guard after the intent classifier selected `SAVE_MEMORY`. That guard prevented ordinary statements from becoming memories, but it also rejected valid natural-language and multilingual requests whose wording was not in the growing phrase set.

The existing architecture already separates deterministic routing, LLM classification, typed extraction, and transactional execution. The classifier can perform semantic intent recognition, but its output must remain untrusted and must not authorize a mutation by itself.

## Decision

Keep explicit deterministic memory commands on the existing fast path. For `MODEL_FALLBACK` requests, extend the existing typed classifier response with optional `MemoryIntentEvidence`:

- `kind` identifies `EXPLICIT_SAVE_REQUEST`, `ORDINARY_STATEMENT`, `QUOTED_INSTRUCTION`, `NEGATED_SAVE_REQUEST`, `REMINDER_REQUEST`, or `UNCERTAIN`;
- `confidence` must be `CLEAR` for a memory save; and
- `evidence_text` must be an exact, source-grounded span from the user message.

Only `EXPLICIT_SAVE_REQUEST` evidence that passes deterministic schema and source-grounding checks may proceed to the existing memory extractor. The extractor returns typed `SaveMemoryFact` content, which is independently schema-validated and grounded in the source text. The canonical authenticated user, ownership, authorization, transaction, and Telegram idempotency path remain unchanged.

Missing, uncertain, contradictory, ungrounded, malformed, quoted, negated, ordinary-statement, or reminder evidence clarifies without extraction or mutation. No additional LLM call is introduced.

Application and provider diagnostic logs contain metadata only for classifier and extractor calls: message lengths/counts, operation names, field names, and content lengths. Raw memory text, recent conversation contents, provider payloads, and raw provider responses are not logged.

## Alternatives Considered

### Expand the English regular-expression allowlist

Rejected. It creates an unbounded maintenance problem, remains language-specific, and still cannot reliably distinguish semantic negation or quoted instructions.

### Trust the classifier operation alone

Rejected. An over-eager or malformed classifier could turn an ordinary fact into a memory write, violating deterministic verification and the no-unintended-mutation requirement.

### Add a second LLM verifier

Rejected. It increases latency and failure surface without replacing deterministic schema, grounding, ownership, authorization, and idempotency checks.

## Consequences

### Positive consequences

Natural-language variation and supported multilingual requests are handled by the configured classifier rather than an English phrase list. Ordinary statements and adversarial memory-like text do not reach extraction unless the classifier supplies clear, grounded explicit-save evidence. The deterministic-first invariant and existing transactional memory service are preserved.

### Trade-offs

Semantic classification quality remains dependent on the configured intent model. A model that cannot reliably interpret a language should produce uncertainty and clarification; deterministic code does not attempt to infer unsupported language semantics.

## Implementation References

- `src/reminder_agent/memories.py`: `MemoryIntentEvidence` and source-grounding validation; typed `SaveMemoryFact` validation remains unchanged.
- `src/reminder_agent/intent.py`: classifier schema/prompt, typed memory extraction, and metadata-only diagnostic logging.
- `src/reminder_agent/reminders.py`: deterministic route gate before extraction and existing authenticated, owner-scoped, idempotent memory execution.
- `tests/unit/test_architecture_invariant.py`: deterministic-first, multilingual, and adversarial evidence cases.
- `tests/unit/test_memories.py`: evidence grounding and non-save evidence cases.
- `tests/unit/test_reminder_intent.py`: structured classifier output and diagnostic-log redaction.
- `tests/integration/test_telegram_memories.py`: owner-scoped persistence, replay idempotency, malformed extraction, and no-mutation regressions.

## Relationship to Original Architecture

This ADR refines the original deterministic-first hybrid NLP and deterministic-verification decisions in [ADR 0001](0001-deterministic-first-hybrid-natural-language-processing.md), [ADR 0002](0002-separate-intent-classification-and-parameter-extraction.md), and [ADR 0003](0003-deterministic-verification-of-llm-proposed-commands.md). It removes a language-specific validation bottleneck without allowing an LLM to authorize or execute a memory mutation.
