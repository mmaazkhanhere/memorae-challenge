# ADR 0001: Deterministic-First Hybrid Natural-Language Processing

## Status

Accepted

## Date

2026-11-03

This is the ADR documentation and acceptance date, not necessarily the original implementation decision date.

## Context

The first reminder-command handling path used deterministic rules and parsers. It handles predefined command forms predictably, but supported requests can be expressed in forms that those rules cannot recognize or fully parse. Treating every message as an LLM request would make straightforward commands dependent on an external provider.

The implemented system therefore needs to support natural-language variation while preserving a deterministic path for recognized, safe requests and a useful response when an LLM provider is unavailable.

## Decision

Use deterministic processing first. Invoke an LLM only as an optional fallback when deterministic routing or parsing is insufficient and the relevant deterministic result permits fallback.

The deterministic conversation router first identifies explicit reminder, listing, rescheduling, cancellation, clarification, social, and unrelated paths. For an ambiguous route, the optional classification adapter may propose a route. The applicable deterministic parser then handles the selected operation. If that parser cannot produce a supported result and permits it, the optional extraction adapter may propose typed parameters. A deterministic result does not require either LLM call.

LLM fallback is advisory: subsequent validation, grounding, authorization, and application-service rules determine whether any operation occurs.

## Alternatives Considered

### Deterministic-only processing

This was the initial implemented and tested approach. It is predictable for supported patterns, but cannot cover all supported natural-language forms without continually expanding rule complexity.

### LLM-only processing

This would expand language coverage, but would make simple requests incur external-model cost and latency and would increase the risk of accepting an incorrect proposal. It was considered undesirable; this ADR does not assert that it was implemented or benchmarked.

### Deterministic-first hybrid processing

Selected. It preserves the deterministic path and uses a configured LLM only for eligible uncertainty or unsupported language variation.

## Consequences

### Positive consequences

Benefits include fewer unnecessary LLM API calls, lower LLM cost exposure, lower latency for deterministic requests, less hallucination exposure for straightforward commands, continued basic behavior when the provider fails or is not configured, and support for language beyond the predefined rules.

### Trade-offs

The approach does not make deterministic rules complete. It adds routing and fallback boundaries that must remain aligned with operation parsers, and ambiguous input can still require clarification. Provider failure or invalid structured output results in a safe clarification or non-mutating response rather than an alternate execution path.

## Implementation References

- `src/reminder_agent/conversation_router.py`: `route_message` performs the deterministic first routing decision and only returns `MODEL_FALLBACK` for eligible text.
- `src/reminder_agent/intent.py`: `RuleBasedIntentInterpreter` provides deterministic interpretation; `ReminderIntentInterpreter.interpret` calls its optional model only when no deterministic intent is available and `fallback_allowed` is true.
- `src/reminder_agent/main.py`: `create_app` configures optional intent-routing and extraction adapters only when an LLM API key is configured.
- `tests/unit/test_conversation_router.py`, `tests/unit/test_reminder_listing_intent.py`, and `tests/integration/test_telegram_rescheduling.py`: cover deterministic routing and the constrained fallback paths.

## Relationship to Original Architecture

The original architecture called for model- and provider-agnostic LLM integration and validated natural-language requests. This ADR records the implemented VS-01 through VS-10 refinement: deterministic interpretation is the normal first path, with optional LLM fallback rather than LLM-first command execution. It does not define architecture for later slices.

Related ADRs: [ADR 0002](0002-separate-intent-classification-and-parameter-extraction.md) and [ADR 0003](0003-deterministic-verification-of-llm-proposed-commands.md).
