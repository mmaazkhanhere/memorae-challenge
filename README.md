# Memorae

Memorae is a Python/FastAPI reminder and memory agent. Users interact through Telegram text,
Telegram voice notes, or an authenticated Vapi phone call. The application stores reminders and
user-owned facts in PostgreSQL, delivers reminders through Telegram, and runs delivery in a
separate worker process.

## Overview

Memorae supports the following workflows:

- Telegram onboarding with IANA time-zone selection and optional verified phone linking.
- One-time and weekly recurring reminders.
- Reminder listing, rescheduling, cancellation, completion, snoozing, and progress summaries.
- User-owned memory save and retrieval, with optional PostgreSQL `pgvector` semantic indexing.
- Telegram voice notes transcribed through a Groq Whisper-compatible endpoint.
- Vapi inbound call authorization and voice tools for reminder and memory operations.

Requests use a deterministic-first routing and extraction pipeline. When deterministic handling is
insufficient, the configured model may classify or propose typed parameters. The application then
validates intent, ownership, authorization, grounding, time zones, and state transitions before a
shared service can mutate data. Ambiguous, unsupported, or incomplete requests do not mutate data.

## Technology stack

- Python `>=3.12`
- FastAPI and Uvicorn
- SQLAlchemy and Alembic
- PostgreSQL with the `pgvector` extension
- Telegram Bot API
- Vapi inbound voice webhooks
- OpenAI-compatible LLM and STT endpoints; Groq Whisper is the default STT configuration
- Optional OpenAI-compatible or Gemini embeddings
- `uv` for dependency management and command execution
- pytest, pytest-cov, Ruff, and mypy for development and testing

## Prerequisites

Install the following before starting local development:

- Git
- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Docker with Docker Compose support

Provider credentials are optional for the deterministic local test suite, but are required for the
corresponding live Telegram, LLM, STT, Vapi, or embedding integrations.

## Getting started

### Option 1: Run the complete stack with Docker Compose

From the repository root:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Compose loads `.env.example` and optionally overlays the untracked `.env`. It starts:

- `postgres`: PostgreSQL with `pgvector`, exposed on host port `5433`.
- `migrate`: runs `alembic upgrade head`.
- `api`: the FastAPI service, exposed on `http://localhost:8000`.
- `worker`: the PostgreSQL-backed Telegram notification and confirmation worker.

The sample configuration uses local placeholder values. Configure real provider credentials in
`.env` before using external integrations.

Stop the stack with:

```powershell
docker compose down
```

The equivalent Make targets are `make run`, `make stop`, and `make logs`.

### Option 2: Run the API and worker from the host

Use this workflow when PostgreSQL is running in Docker but the Python processes run locally:

```powershell
Copy-Item .env.example .env
uv sync --extra test
docker compose up -d postgres
uv run alembic upgrade head
```

Start the API in one terminal:

```powershell
uv run uvicorn reminder_agent.application.main:app --reload
```

Start the worker in another terminal:

```powershell
uv run reminder-worker
```

The host configuration in `.env.example` points to PostgreSQL at
`postgresql+psycopg://memorae:memorae@localhost:5433/memorae`. Do not use that local database
for PostgreSQL integration-test cleanup; use the separate test database described below.

## Configuration

Copy [`.env.example`](.env.example) to `.env` for host-based development. Do not commit `.env`,
provider credentials, database URLs, or webhook secrets.

| Variable | Required when | Purpose |
| --- | --- | --- |
| `APP_ENV` | Always | Use `development` locally. Production validation is stricter. |
| `DATABASE_URL` | API, worker, and migrations | PostgreSQL connection string. Compose overrides it with the `postgres` service URL. |
| `TELEGRAM_BOT_TOKEN` | Telegram integration and production | Telegram Bot API credential. |
| `TELEGRAM_WEBHOOK_SECRET` | Telegram webhook and production | Value checked in `X-Telegram-Bot-Api-Secret-Token`; production requires a unique value of at least 32 characters. |
| `LLM_API_KEY` and `LLM_BASE_URL` | Model fallback and production | OpenAI-compatible classifier, extraction, and memory-answer endpoint. |
| `LLM_INTENT_MODEL` and `LLM_EXTRACT_MODEL` | Optional | Models used for intent fallback and typed extraction. |
| `STT_API_KEY`, `STT_BASE_URL`, and `STT_MODEL` | Telegram voice notes and production | Whisper-compatible voice-note transcription. |
| `VAPI_ENABLED` and `VAPI_WEBHOOK_BEARER_TOKEN` | Vapi calls | Enable and authenticate `POST /webhook/vapi`. |
| `EMBEDDING_PROVIDER` and `EMBEDDING_API_KEY` | Semantic memory | Leave the provider `disabled` for lexical-only memory, or configure `openai_compatible`/`gemini`. |
| `TEST_DATABASE_URL` | PostgreSQL integration tests | Dedicated test database, normally on host port `5434`; it must not equal `DATABASE_URL`. |
| `PHONE_DEFAULT_REGION` | Optional | Deployment-wide fallback for local-format phone numbers. Users can set their own region with `/phone_region XX`. |
| `LOG_LEVEL`, `LOG_FORMAT`, and `LOG_COLOR` | Optional | Configure text or JSON operational logs. |

The sample also documents worker polling, notification retry, recurrence materialization, voice
limits, conversation-session expiry, memory retrieval, and embedding privacy settings. Review
those values before deployment. In particular, `EMBEDDING_ALLOW_SENSITIVE_EXTERNAL` controls
whether sensitive facts may be sent to an external embedding provider.

Production startup requires PostgreSQL, a Telegram token and webhook secret, LLM and STT keys,
and valid values for any enabled optional provider. When Vapi is enabled, its bearer token and
assistant model and voice settings are also validated.

## Usage

### Telegram

Configure a Telegram webhook that sends `POST /webhook/telegram` with the configured secret header.
After onboarding with `/start`, users can send requests such as:

```text
Remind me in five minutes to call the bank
Remind me tomorrow at 9 AM to check my email
Remind me every Monday at 8 AM to call the bank
What do I still have to do today?
Cancel the standup
```

Users can link a verified phone through the contact-sharing flow, reopen linking with
`/link_phone`, and provide a country code for a national-format contact with `/phone_region PK`.
Voice-note responses are returned as Telegram text. Failed or inaudible voice notes do not create
reminders or memories.

Weekly recurrence is supported. Arbitrary monthly or calendar recurrence rules are not supported
by the current deterministic parser.

### Vapi phone calls

To enable calls:

1. Link the caller's E.164 phone number through Telegram.
2. Set `VAPI_ENABLED=true` and configure `VAPI_WEBHOOK_BEARER_TOKEN`.
3. Apply the current migrations with `uv run alembic upgrade head`.
4. Configure the Vapi Server URL as `POST /webhook/vapi` with the matching Bearer credential.
5. Leave the phone number without a fixed assistant so the `assistant-request` authorization can
   reject unknown callers before a conversation starts.

The Vapi adapter exposes tools for reminder management and memory save/lookup. Tools receive the
caller's complete utterance; user IDs, reminder IDs, fact IDs, and delivery fields are resolved
and authorized by the backend.

### Health and readiness

The API exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process health check. |
| `GET /ready` | Database, schema, migration-head, and required-column readiness check. |
| `POST /webhook/telegram` | Authenticated Telegram update entry point. |
| `POST /webhook/vapi` | Authenticated Vapi entry point when Vapi is enabled. |

For example:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
```

## Project structure

```text
.
├── src/reminder_agent/
│   ├── application/       # Shared workflows, coordination, and application contracts
│   ├── channels/          # Telegram and Vapi adapters and webhooks
│   ├── features/          # Intent, reminder, memory, onboarding, and voice features
│   ├── infrastructure/   # Configuration, database, providers, logging, and runtime
│   └── shared/            # Cross-feature contracts and idempotency helpers
├── tests/
│   ├── unit/              # Fast isolated tests
│   ├── integration/       # Provider, database, webhook, and worker integration tests
│   └── e2e/               # End-to-end workflow tests
├── alembic/               # Database migrations
├── docs/                  # Architecture, implementation plan, and ADRs
├── benchmarks/            # Benchmark and evaluation assets
├── docker-compose.yml     # Local application, worker, and PostgreSQL services
├── Dockerfile             # Locked production-style image build
├── pyproject.toml         # Project metadata, scripts, and tool configuration
├── uv.lock                # Locked Python dependency resolution
└── .env.example           # Local and deployment configuration template
```

## Development

Install the locked project and test dependencies with:

```powershell
uv sync --extra test
```

Useful commands from the repository root:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run alembic upgrade head
uv run reminder-worker
uv run memory-embedding-backfill
```

`memory-embedding-backfill` requires an enabled embedding provider and applies bounded,
resumable indexing to existing memory facts. It does not replace the durable fact text.

## Testing

Run the default suite:

```powershell
uv run pytest
```

Run a specific layer or collect coverage:

```powershell
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest tests/e2e
uv run pytest --cov=src/reminder_agent --cov-report=term-missing
```

Most tests use injected SQLite sessions by default. PostgreSQL-backed tests require the dedicated
test service and connection string:

```powershell
docker compose --profile test up -d postgres-test
$env:TEST_DATABASE_URL = "postgresql+psycopg://memorae:memorae@localhost:5434/memorae_test"
uv run pytest tests/integration
```

The test fixture refuses to clean a database when `TEST_DATABASE_URL` resolves to the application
database. Live Gemini and memory-retrieval tests are opt-in and require provider credentials and
the environment flags used by the integration tests.

## Build and deployment

The [`Dockerfile`](Dockerfile) builds a Python 3.12 image with the locked dependencies and no
development extras. Docker Compose runs migrations before starting the API and worker.

For a deployment using an external PostgreSQL service:

1. Provide a PostgreSQL `DATABASE_URL` and the required provider secrets through the deployment
   secret store.
2. Run `uv run alembic upgrade head` from the application image or deployment environment.
3. Run the API and `reminder-worker` as separate processes.
4. Configure the Telegram and, if enabled, Vapi webhooks to reach the API.
5. Confirm `GET /ready` reports the current migration head,
   `0017_voice_call_authorizations`.

This repository does not contain a provider-specific deployment manifest or CI/CD workflow.

## Architecture

Channel adapters authenticate the external request and resolve the canonical user. Shared
application services then process typed commands and enforce ownership, authorization, business
rules, and legal state transitions. PostgreSQL is the source of truth for reminders, memory facts,
idempotency records, and durable delivery state.

### C4 context-level diagram

```mermaid
flowchart LR
    USER["User"]
    TELEGRAM["Telegram\nText, voice notes, notifications"]
    VAPI["Vapi\nInbound phone calls"]
    MEMORAE["Memorae\nReminder and memory agent"]
    PROVIDERS["Configured AI providers\nLLM, STT, optional embeddings"]

    USER -->|"Text and voice notes"| TELEGRAM
    USER -->|"Phone calls"| VAPI
    TELEGRAM <-->|"Messages, contact details, reminders"| MEMORAE
    VAPI <-->|"Caller authorization and voice operations"| MEMORAE
    MEMORAE <-->|"Intent, extraction, transcription, indexing"| PROVIDERS

    classDef actor fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef external fill:#f8fafc,stroke:#94a3b8,color:#0f172a
    classDef system fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#172554

    class USER actor
    class TELEGRAM,VAPI,PROVIDERS external
    class MEMORAE system
```

### C4 container-level diagram

```mermaid
flowchart LR
    USER["User"]

    subgraph EXTERNAL["External systems"]
        direction TB
        TELEGRAM["Telegram Bot API"]
        VAPI["Vapi voice platform"]
        PROVIDERS["Configured AI providers\nOpenAI-compatible LLM/STT\nOptional embeddings"]
    end

    subgraph MEMORAE["Memorae"]
        direction TB
        API["FastAPI application\nWebhooks, authentication, routing, shared services"]
        WORKER["reminder-worker\nNotification and confirmation delivery"]
        DATABASE[("PostgreSQL\nOperational state, idempotency, pgvector")]
    end

    USER <-->|"Text and voice notes"| TELEGRAM
    USER <-->|"Phone calls"| VAPI
    TELEGRAM <-->|"Telegram webhook and Bot API"| API
    VAPI <-->|"Call authorization and tool calls"| API
    API <-->|"LLM, STT, embedding requests"| PROVIDERS
    API <-->|"Users, reminders, memories, sessions"| DATABASE
    WORKER <-->|"Claims and delivery state"| DATABASE
    WORKER -->|"Reminder notifications and confirmations"| TELEGRAM

    classDef core fill:#dbeafe,stroke:#2563eb,color:#172554,stroke-width:2px
    classDef data fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef outside fill:#f1f5f9,stroke:#94a3b8,color:#334155

    class API,WORKER core
    class DATABASE data
    class USER,TELEGRAM,VAPI,PROVIDERS outside
```

The worker claims due notifications with PostgreSQL row locks and leases, rechecks eligibility,
and records provider outcomes. External delivery cannot provide a strict end-to-end exactly-once
guarantee after an ambiguous provider response; such outcomes are recorded as `UNKNOWN` for
monitoring or recovery.

See [`docs/Architecture.md`](docs/Architecture.md) for system behavior, domain rules, channel
flows, reliability constraints, and the implemented architecture.Architecture decisions are recorded in [`docs/adr/`](docs/adr/).

## Troubleshooting

- If `/ready` returns `503`, run `uv run alembic upgrade head` and verify that the API can reach
  the configured PostgreSQL database. Readiness checks both schema objects and the migration head.
- If scheduled reminders are not delivered, confirm that `reminder-worker` is running and that
  `TELEGRAM_BOT_TOKEN` is configured.
- If Vapi returns `404`, confirm `VAPI_ENABLED=true`. If it returns `401`, verify the Bearer token
  configured in Vapi matches `VAPI_WEBHOOK_BEARER_TOKEN`.
- If model or voice-note behavior falls back to clarification, configure `LLM_API_KEY` or
  `STT_API_KEY` and check the corresponding `*_BASE_URL` and model settings.
- If PostgreSQL integration tests refuse to start, ensure `TEST_DATABASE_URL` points to the
  dedicated `postgres-test` database and is different from `DATABASE_URL`.

## Contributing

No separate contributing guide or automated CI workflow is included. For changes, preserve the
deterministic-first invariant, keep provider-specific parsing at channel boundaries, add tests for
authorization, idempotency, failure behavior, and cross-user isolation, and update the relevant
architecture or ADR documentation when behavior changes. Repository-specific development and
review rules are in [`AGENTS.md`](AGENTS.md).

## License

This project is provided solely for recruitment evaluation. The repository's
[`LICENSE.md`](LICENSE.md) states that commercial use, redistribution, production deployment,
and incorporation into other projects require prior written authorization from the copyright
holder.
