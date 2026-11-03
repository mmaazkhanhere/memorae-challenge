# ADR 0003: Deterministic Verification of LLM-Proposed Commands

## Status

Accepted

## Date

2026-11-03

This is the ADR documentation and acceptance date, not necessarily the original implementation decision date.

## Context

An LLM can propose an incorrect operation, date, reminder identifier, target, scope, or parameter. A proposal must not be sufficient to modify application state: reminder operations are user-scoped, scheduled work is stateful, and recurring operations distinguish an occurrence from a series.

Validation and authorization were already requirements of the original architecture. This ADR specifically records the additional LLM boundary: model output is untrusted and must undergo deterministic grounding and operation-specific checks before execution.

The implemented VS-01 through VS-10 flows therefore require deterministic and database-backed checks after parsing or model fallback and before execution.

## Decision

Treat LLM outputs as untrusted proposals. Before an operation is executed, deterministic code and database-backed checks confirm, as applicable:

- a supported operation and typed parameter validity;
- authenticated user identity and reminder ownership;
- grounding against authorized reminder candidates;
- scheduling constraints and a valid IANA time zone;
- recurrence constraints and occurrence-versus-series scope;
- the current reminder and occurrence state; and
- a valid linked Telegram delivery identity for scheduled work.

Invalid or ambiguous proposals are rejected or clarified rather than executed speculatively. Verification is intentionally distributed across parsers, grounding logic, application services, and worker eligibility checks; it is not a single centralized verifier.

## Alternatives Considered

### Direct execution of LLM proposals

This is unsuitable because a hallucinated or malformed proposal could cause an unauthorized or incorrect change. It was not implemented or tested as an execution strategy.

### Deterministic verification of proposed commands

Selected. Each operation validates the facts that it owns and reloads authorized state before a mutation or delivery.

## Consequences

### Positive consequences

The system gains safety and predictability: a model cannot select another user's reminder, invent a candidate, bypass schedule and state checks, or cause a delivery without an eligible destination identity. Ambiguity is preserved for clarification rather than resolved by guesswork.

### Trade-offs

This increases verification complexity because validation is distributed across several modules and transactions. Valid-looking model proposals can still require a clarification when current state, ownership, grounding, or scope cannot be confirmed.

## Implementation References

- `src/reminder_agent/intent.py`: strict typed intent schemas reject unsupported fields and invalid values; `OpenAICompatibleIntentInterpreter` instructs the provider to return proposals only and never invent missing records or fields.
- `src/reminder_agent/reminders.py`: `handle_reminder_text` resolves the trusted Telegram identity and active user, checks onboarding and time zone, applies policy, routes operations, and rechecks typed intents before creating, listing, rescheduling, or cancelling reminders.
- `src/reminder_agent/cancellation.py`: `ground_model_cancellation_request` ties a proposal to the current message or bounded history; candidate lookup filters by owner and active state; cancellation resolution distinguishes ambiguity and scope.
- `src/reminder_agent/rescheduling.py`: deterministic parsing checks future scheduling; candidate lookup is user-scoped; `pending_candidates` reloads owned active candidates; `apply_reschedule` locks and revalidates the reminder, destination identity, due time, state, and scope.
- `src/reminder_agent/worker.py`: notification claiming and delivery rechecks reminder/occurrence state and the linked Telegram destination, including destination ownership.
- `tests/integration/test_telegram_rescheduling.py` (`test_hallucinated_pending_candidate_proposal_cannot_mutate`), `tests/integration/test_telegram_reminders.py` (cross-user and invalid-schedule cases), and `tests/unit/test_cancellation.py` (grounding cases) cover representative verification boundaries.

## Relationship to Original Architecture

The original architecture already required validation of requested operations, parameters, and authorization before execution. This ADR records the VS-01 through VS-10 LLM-specific refinement: model output is advisory and must be grounded and verified in operation-specific deterministic paths before execution. It does not change or specify later-slice functionality.

Related ADRs: [ADR 0001](0001-deterministic-first-hybrid-natural-language-processing.md) and [ADR 0002](0002-separate-intent-classification-and-parameter-extraction.md).
