# Owner-Scoped pgvector Semantic Memory

## Status

Accepted

## Date

2026-11-06

This is the ADR documentation and acceptance date; no earlier decision date is
assumed.

## Context

VS-13 stores authoritative user memory text in PostgreSQL, and VS-14 retrieves
it through deterministic owner-scoped lexical matching. Lexical lookup is safe
and precise for exact subjects but cannot reliably match paraphrases such as
"Where do I park?" to "my parking slot is B17." Semantic retrieval must scale
without adding a separate source of truth, weakening canonical-user isolation,
or allowing an LLM or vector score to invent or authorize an answer.

Existing rows must remain valid during rollout. Provider failures and sensitive
content also require explicit policies: a fact must not be lost when indexing
fails, and secrets must not be transmitted to an external provider by default.

## Decision

Keep `memory_facts.content` as the authoritative fact and add nullable
PostgreSQL pgvector storage plus indexing metadata through additive migration
`0016_memory_embeddings`. Use `vector(1536)` and cosine distance with a
provider-agnostic embedding interface. The initial configured adapter targets
OpenAI-compatible `text-embedding-3-small`; automated tests use deterministic
fakes and never require a live provider.

Preserve deterministic-first routing and retrieval. Run a bounded,
canonical-owner lexical search first. Only when it is insufficient may the
shared service generate a query embedding and execute a bounded exact cosine
query constrained by canonical `user_id` and active embedding model. Return
only original `MemoryFact.content`. Treat close plausible matches as ambiguous
through the existing typed clarification context, and treat sub-threshold or
unavailable results as unknown.

Commit the authoritative fact and save idempotency record before calling the
embedding provider. Index in a second owner-checked transaction so failures or
crashes leave the fact `PENDING` or `FAILED` and eligible for resumable
backfill. Track the model and authoritative-content hash so regeneration is
idempotent and content changes invalidate old vectors.

Do not send detected passwords, PINs, access codes, keys, tokens, or recovery
secrets to an external provider unless an approved deployment policy explicitly
opts in. Mark these facts `SKIPPED_SENSITIVE` and retain lexical retrieval.

Use exact owner-filtered vector search initially. Do not add HNSW until query
plans, latency, and per-owner cardinality show that approximate search is
necessary and its recall tradeoff is acceptable.

## Alternatives Considered

- A separate vector database was rejected because it adds synchronization,
  authorization, operational, and consistency risks while PostgreSQL remains
  the required source of truth.
- LangChain or a full RAG framework was rejected because the use case needs one
  small embedding boundary and conservative candidate selection, not a second
  orchestration layer.
- Semantic-first retrieval was rejected because it would add provider cost and
  latency to exact lookups and could let a score conceal deterministic conflicts.
- HNSW at initial rollout was rejected because current scale does not justify
  approximate recall and tuning tradeoffs.
- Rolling back a fact when embedding fails was rejected because semantic
  indexing is derived data and must not determine whether authoritative content
  is durably saved.

## Consequences

Positive consequences:

- PostgreSQL remains the single owner-authorized source of truth.
- Exact lookups stay fast, deterministic, provider-independent, and LLM-free.
- Paraphrases can be retrieved without allowing generated answer text.
- Existing rows remain usable and can be backfilled incrementally.
- Provider failures, retries, replays, and model regeneration have explicit,
  testable states.
- External secret disclosure is denied by default.

Trade-offs:

- `vector(1536)` couples the current column to one dimension; changing dimensions
  requires another migration and model backfill.
- Exact cosine search may eventually require an approximate index at higher
  per-owner cardinality.
- Post-commit synchronous indexing adds provider latency to a new save response,
  although it cannot roll back or falsify the saved fact.
- Sensitive or unembedded facts support lexical retrieval but not semantic
  paraphrases under the default privacy policy.
- Relevance and ambiguity thresholds require ongoing production-corpus
  calibration even though deterministic tests cover their enforced behavior.

## Implementation References

- `alembic/versions/0016_memory_embeddings.py`
- `src/reminder_agent/embeddings.py`
- `src/reminder_agent/memories.py`
- `src/reminder_agent/memory_backfill.py`
- `tests/unit/test_memories.py`
- `tests/integration/test_telegram_memories.py`
- `tests/integration/test_pgvector_memories.py`

## Relationship to Original Architecture

This decision extends VS-13 and VS-14 without changing the deterministic-first,
LLM-fallback invariant defined in
[0001-deterministic-first-hybrid-natural-language-processing.md](0001-deterministic-first-hybrid-natural-language-processing.md)
or the separation of classification and extraction defined in
[0002-separate-intent-classification-and-parameter-extraction.md](0002-separate-intent-classification-and-parameter-extraction.md).
It preserves structured memory intent evidence from
[0004-structured-memory-intent-evidence.md](0004-structured-memory-intent-evidence.md):
embeddings are generated only after the existing route, extraction, grounding,
ownership, and authorization checks succeed, and vectors never authorize a
mutation or supply a fact value.
