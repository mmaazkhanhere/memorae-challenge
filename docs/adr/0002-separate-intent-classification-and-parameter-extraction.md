# ADR 0002: Separate Intent Classification and Parameter Extraction

## Status

Accepted

## Date

2026-11-03

This is the ADR documentation and acceptance date, not necessarily the original implementation decision date.

## Context

Selecting an operation and extracting the parameters for that operation are different responsibilities. Combining them makes it harder to identify whether an error arose from operation selection or field extraction, and risks mixing routing decisions with proposed data changes. Supplying conversational history to extraction also needs careful limits because irrelevant or hostile context can influence a proposal.

The implemented reminder flows need optional LLM assistance for both ambiguous intent and difficult parameter extraction without allowing either stage to become an authority for user data or operation execution.

## Decision

Use separate stages.

1. **Intent classification** uses deterministic routing, with an optional dedicated classification LLM for an eligible ambiguous route. It proposes only an operation route and does not extract fields or execute actions.
2. **Parameter extraction** uses deterministic parsing, with an optional dedicated extraction LLM when the applicable parser allows fallback. It proposes structured parameters only for the selected route.

The stages use independently configurable models. The intended operating profile is a smaller, cheaper model for classification and a more capable model for extraction, where temporal or contextual parameter reasoning can be needed. Those model choices remain configuration, rather than a permanent provider or model requirement.

The LLM adapters use an OpenAI-compatible chat-completions interface so deployments can select a compatible provider. Conversation context is bounded for classification and extraction where it is supplied; separation reduces responsibility mixing but does not eliminate hallucinations or context-poisoning risk.

## Alternatives Considered

### One combined classification-and-extraction model call

This would reduce orchestration steps, but would conflate choosing an operation with proposing its parameters, complicating debugging and validation.

### Deterministic classification and extraction only

This remains the preferred first path, but alone does not handle all supported natural-language variation.

### Separate classification and extraction stages

Selected. It keeps the route proposal narrow, scopes extraction to a selected operation, and permits independent model configuration.

## Consequences

### Positive consequences

The split improves separation of concerns, makes routing and extraction failures easier to diagnose, permits independent model configuration, and reduces mixing of operation selection with parameter proposals.

### Trade-offs

It also introduces orchestration complexity: an eligible request can make a classification fallback call and a later extraction fallback call. Bounded context and downstream validation remain required because separation does not guarantee that a model proposal is correct or safe.

## Implementation References

- `src/reminder_agent/intent.py`: `OpenAICompatibleIntentRouter.classify` returns the narrow `IntentRouteSuggestion`; its prompt prohibits field extraction and execution. `OpenAICompatibleIntentInterpreter.interpret` produces structured extraction proposals for a selected route. `ReminderIntentInterpreter` keeps deterministic parsing ahead of extraction fallback.
- `src/reminder_agent/intent.py`: typed schemas such as `IntentRouteSuggestion`, `CreateOneTimeReminderIntent`, `ListRemindersIntent`, and pending-update proposals use strict Pydantic validation.
- `src/reminder_agent/main.py` and `src/reminder_agent/settings.py`: configure separate `llm_intent_model` and `llm_extract_model` values through an OpenAI-compatible base URL, API key, and timeout.
- `.env.example`: records the current configuration defaults (`LLM_INTENT_MODEL=openai/gpt-oss-20b` and `LLM_EXTRACT_MODEL=gpt-5.4-mini-2026-03-17`) as deployment defaults, not permanent architectural commitments.
- `tests/integration/test_telegram_reminders.py`, `tests/integration/test_telegram_rescheduling.py`, and `tests/unit/test_reminder_listing_intent.py`: exercise classifier-selected routes, structured extraction, and fallback boundaries.

## Relationship to Original Architecture

The original architecture required provider-agnostic LLM integration and validated typed commands. This ADR documents the VS-01 through VS-10 implementation choice to divide optional LLM assistance into route classification and parameter extraction. It neither fixes a provider/model for later work nor defines later-slice behavior.

Related ADRs: [ADR 0001](0001-deterministic-first-hybrid-natural-language-processing.md) and [ADR 0003](0003-deterministic-verification-of-llm-proposed-commands.md).
