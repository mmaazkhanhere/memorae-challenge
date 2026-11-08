# Architectural boundaries

The package dependency direction is executable and the Telegram channel
boundary is now enforced without allowing provider-specific orchestration to
grow back into the adapter.

## Enforced rules

- `features/` is the current domain-oriented layer. It cannot import
  `application/` or `channels/`.
- `application/` is channel-neutral and cannot import concrete channel
  adapters. `application/bootstrap.py` is the composition-root exception.
- New feature-to-infrastructure imports are rejected unless the exact edge is
  listed in the temporary allowlist in
  `tests/unit/test_architecture_boundaries.py`.
- Infrastructure may not import channel code, except the worker's two explicit
  Telegram delivery imports.
- Channel modules must use application services. Compatibility modules under
  `channels/telegram` are import aliases only and may not import feature code.
- Durable response claiming and delivery-outcome persistence use the
  provider-neutral `application.delivery.ResponseDeliveryPort`; Telegram only
  supplies the provider send/retry implementation.
- `application.contracts.OutboundResponse` is the canonical response meaning;
  it contains no destination, keyboard, callback, or provider message fields.
- `features.onboarding.request_contracts.OnboardingRequest` is the normalized
  onboarding input; channel adapters translate provider payloads into it before
  application dispatch. Its identity and destination fields are opaque channel
  values, not Telegram payload objects.
- The canonical package graph must remain acyclic, and the existing feature
  private-boundary rules continue to apply.

There is currently no standalone `domain/` package. The `features/` packages
contain the domain responsibilities described by `docs/Architecture.md`.

## Remaining composition exceptions

- Feature modules import SQLAlchemy models, provider clients, observability,
  embeddings, settings, and sessions. These are exact importer/importee pairs
  in the test allowlist and are candidates for Phase 2+ extraction.
- `infrastructure/worker.py` imports Telegram callback encoding and the
  Telegram client because delivery is still Telegram-specific. This is a
  composition concern, not reusable business logic.
- `features/onboarding/workflow.py` remains the raw-update entry point.
  Transactional onboarding branches live in
  `features/onboarding/handlers.py`, while free-text onboarding and active
  conversation handoff live in `features/onboarding/message.py`.
  `features/onboarding/responses.py` retains compatibility DTO behavior; its
  transaction and delivery ledger behavior is provider-neutral.
  `ApplicationResult` is now a backward-compatible subclass of the canonical
  `OutboundResponse` contract.

The allowlists are intentionally narrow and module-pair based. They should be
removed as extraction work lands; they must not be broadened merely to silence
future failures.

## Boundary implementation

The Telegram package is organized as:

```text
channels/telegram/
├── callbacks.py       # signed callback token encoding/validation
├── identity.py        # sender/chat extraction and identity lookup entrypoint
├── inbound.py         # TelegramUpdate -> ApplicationRequest normalization
├── outbound.py        # Telegram delivery, chunking, retries, rate limits
├── provider.py        # Telegram HTTP client and payload schemas
├── voice.py           # voice transport compatibility entrypoint
└── presentation/      # keyboards and Telegram-facing rendering
```

Reminder, completion, memory, intent, clarification, and idempotent application
orchestration live under `application/`. Reusable onboarding validation and
phone-link decision mapping live under `features/onboarding`; application
identity services retain persistence, ownership checks, and transaction
boundaries. Old Telegram module paths remain only as compatibility aliases while
callers migrate.

Onboarding orchestration and feature logic are grouped under `features/onboarding/`:

```text
workflow.py   # public dispatch façade and compatibility workflow entry point
handlers.py   # transactional onboarding branches
message.py    # timezone completion and active-user handoff
responses.py  # idempotency and staged-response persistence
phone_linking_service.py # application injection bridge to feature phone linking
```

The former root-level `application/onboarding_*.py` modules have been removed;
callers use the descriptive grouped modules directly.
