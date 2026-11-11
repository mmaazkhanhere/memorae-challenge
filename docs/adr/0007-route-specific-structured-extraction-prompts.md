# Route-Specific Structured Extraction Prompts

## Status

Accepted

## Date

2026-11-11

This is the ADR documentation and acceptance date; no earlier decision date is
assumed.

## Context

Memorae performs optional structured extraction only after deterministic routing
has selected a supported operation. The previous extraction prompt sent one
omnibus reminder prompt for every reminder route. That prompt included creation,
listing, summary, completion, snooze, reschedule, cancellation, and pending
update instructions even when only one operation was selected. This increased
input-token cost and exposed the model to unrelated operation contracts.

The prompt also appended conversation history and application context broadly.
Conversation text, reminder candidates, selection lists, and pending state may
contain user-controlled or otherwise untrusted text. Context must help resolve
legitimate multi-turn references without becoming an instruction channel or
silently dropping information needed for safe clarification.

The optimization must preserve the existing deterministic-first routing,
route-specific response contracts, typed validation, target grounding,
ownership checks, authorization, idempotent execution, and non-mutating
clarification behavior.

## Decision

Compose structured extraction prompts from a small shared instruction block and
only the instruction block for the already selected route. Keep the dedicated
memory-save and memory-retrieval prompts separate from reminder extraction.

Preserve the existing route semantics and operation names for new reminders,
listing, summaries, snoozing, rescheduling, cancellation, and pending reminder
updates. Keep material interpretation rules in the prompt even where the
application validates the same values, including local-time interpretation,
recurrence representation, clarification payloads, candidate-selection rules,
scope semantics, explicit-field preservation, and prohibition on invented IDs
or values.

Include dynamic context only where it is relevant:

- recent conversation for creation, pending updates, rescheduling, and
  cancellation;
- trusted pending reminder state for pending updates;
- reschedule candidates, selection context, and pending reschedule state only
  for rescheduling;
- owned cancellation candidates only for cancellation;
- no reminder context for memory extraction, listing, summary, or snooze.

Wrap supplied context in explicit data delimiters and instruct the model that
the values are untrusted application data, not instructions. Bound recent
history to the existing eight-turn policy and 2,000 characters per message.
Bound candidate and selection context to 50 records. When records are omitted,
include an explicit truncation notice directing the model not to resolve an
indirect target from the incomplete set.

Do not change provider response schemas, response decoding, typed validation,
grounding, authorization, ownership checks, deterministic fallback routing, or
execution services.

## Alternatives Considered

- Keep the omnibus prompt. This preserves the existing wording but continues to
  pay the full prompt cost and exposes each route to unrelated instructions.
- Remove most shared interpretation rules because schemas validate them. This
  was rejected because prompt guidance still materially affects date, timezone,
  recurrence, clarification, and candidate interpretation before validation.
- Send all context to every route but add stronger warnings. This retains
  unnecessary token cost and does not provide operation-level context isolation.
- Truncate context without an explicit marker. This was rejected because the
  model could treat an incomplete candidate set as complete and make an unsafe
  indirect selection.
- Change schemas or downstream validation as part of the optimization. This was
  rejected because prompt composition should not alter the established safety
  boundary or application contracts.

## Consequences

Positive consequences:

- Ordinary reminder routes receive only their selected operation's instructions.
- Static prompt size is reduced by approximately 73% to 89% for the measured
  reminder routes using the representative contexts.
- Memory extraction avoids irrelevant reminder context.
- Candidate, history, and pending context are operation-scoped and explicitly
  marked as data.
- Incomplete bounded context is visible to the model and results in
  clarification rather than an implied complete candidate set.
- Existing schemas, validation, authorization, grounding, and execution remain
  unchanged.

Trade-offs:

- Prompt wording is distributed across route-specific constants and must remain
  synchronized with the typed response contracts.
- Context limits can require clarification for unusually large histories or
  candidate sets.
- Exact token savings depend on the deployed model tokenizer; the repository
  does not include the production tokenizer.
- Prompt-size reduction does not by itself establish improved extraction
  accuracy and requires real-provider evaluation.

## Implementation References

- `src/reminder_agent/features/intent/model/prompts.py`
- `src/reminder_agent/features/intent/model/interpreter.py`
- `src/reminder_agent/features/intent/model/response.py`
- `src/reminder_agent/features/intent/contracts.py`
- `tests/unit/test_intent_prompt_construction.py`
- `tests/unit/test_reminder_intent.py`
- `tests/unit/test_reminder_listing_intent.py`
- `tests/unit/test_snooze.py`
- `tests/unit/test_rescheduling.py`

## Relationship to Original Architecture

This decision refines the prompt-composition boundary without changing the
deterministic-first, LLM-fallback invariant defined in
[0001-deterministic-first-hybrid-natural-language-processing.md](0001-deterministic-first-hybrid-natural-language-processing.md).
It preserves the separation between route classification and parameter
extraction defined in
[0002-separate-intent-classification-and-parameter-extraction.md](0002-separate-intent-classification-and-parameter-extraction.md),
and retains the deterministic verification of model proposals required by
[0003-deterministic-verification-of-llm-proposed-commands.md](0003-deterministic-verification-of-llm-proposed-commands.md).
It also preserves the structured memory intent evidence boundary documented in
[0004-structured-memory-intent-evidence.md](0004-structured-memory-intent-evidence.md):
memory extraction proposes content or a query only after routing, and it does
not authorize persistence or supply ownership information.
