# Memorae

## Local development

VS-03 and VS-07 implement Telegram onboarding, verified phone linking, one-time reminders, and
weekly recurring reminders
with FastAPI, SQLAlchemy, Alembic, and PostgreSQL.

```powershell
uv sync --extra test
docker compose up -d postgres
docker compose --profile test up -d postgres-test
$env:DATABASE_URL = "postgresql+psycopg://memorae:memorae@localhost:5433/memorae"
$env:TEST_DATABASE_URL = "postgresql+psycopg://memorae:memorae@localhost:5434/memorae_test"
uv run alembic upgrade head
uv run uvicorn reminder_agent.application.main:app --reload
# In a second process:
uv run reminder-worker
```

For a supervised two-process local deployment, copy `.env.example` to `.env`, then run:

```powershell
docker compose up -d
```

Compose runs migrations once, then supervises the API and `reminder-worker` as separate services
with independent restart policies. The API is exposed on port 8000; the worker has no public port.

Copy `.env.example` to `.env` and configure `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_WEBHOOK_SECRET`. `PHONE_DEFAULT_REGION` is only a deployment-wide
fallback; in a multi-country deployment, users with local-format contacts can
set their own region with `/phone_region XX`. Set the Telegram webhook to
`POST /webhook/telegram` with the same secret token.

Telegram voice notes are downloaded only after the sender is authenticated,
validated as bounded OGG voice media, and transcribed through the configured
Groq Whisper-compatible `STT_*` endpoint. The transcript then uses the same
deterministic-first command router and shared reminder/memory services as typed
text. Audio and transcripts are not retained or written to logs; voice replies
are Telegram text messages. Failed or inaudible notes never mutate reminders
or memories.

### VS-16 Vapi inbound authorization

Set `VAPI_ENABLED=true`, configure a unique `VAPI_WEBHOOK_BEARER_TOKEN`, and
run `uv run alembic upgrade head`. In Vapi, attach the HTTPS Server URL
`POST /webhook/vapi` and the matching saved Bearer credential to the inbound
phone-number or organization server configuration. Leave the phone number
without a fixed assistant so the `assistant-request` decision can reject
unknown callers before a conversation starts. Link the caller's E.164 number
through Telegram first. The transient assistant exposes server tools for
reminder creation and memory save/lookup. Each tool accepts only the caller's
complete utterance and uses the shared deterministic-first application
pipeline; caller-supplied user IDs, fact IDs, and reminder fields are never
trusted. Sensitive memory facts are not disclosed over the phone and must be
checked in the linked Telegram chat.

Production startup fails fast unless `DATABASE_URL` uses PostgreSQL,
`TELEGRAM_BOT_TOKEN` is configured, and `TELEGRAM_WEBHOOK_SECRET` is a unique value of at
least 32 characters. The sample placeholders and local SQLite defaults are intended only for
injected test applications. `/ready` also verifies that the domain tables exist, so a stale
`alembic_version` row cannot make an empty database appear healthy. Readiness also requires the
current migration head (`0017_voice_call_authorizations`) and the worker/confirmation claim
columns, so a partially migrated database is rejected before webhooks are accepted.
The additive schema-repair migrations also rebuild missing domain tables when a
database was reset without resetting its Alembic version row.

Reminder text uses a confidence-first, strict typed boundary. The built-in parser
only handles short greetings and thanks plus unambiguous one-time reminders such as
`Remind me in five minutes to call the bank`, `Remind me tomorrow at 9 AM to check my email`,
`Remind me on Friday at 6 PM to go to the gym`, and `Remind me to pay the bill at 8 PM today`.
It also accepts weekly schedules such as `Remind me every Monday at 8 AM to call the bank` and
stores a typed weekly rule rather than trusting a model-generated recurrence string. Anything
requiring context, a clarification, or conversational interpretation falls back to the configured
model. By default, intent fallback uses `openai/gpt-oss-20b`, while typed extraction
and grounded memory answering use `openai/gpt-oss-120b` through the same configured
OpenAI-compatible client. `LLM_BASE_URL`,
`LLM_API_KEY`, `LLM_INTENT_MODEL`, and `LLM_EXTRACT_MODEL` can instead target any OpenAI-compatible `/chat/completions`
endpoint, including a Groq or Gemini-compatible gateway, without coupling reminder business
rules to a provider. The model can only propose a task and time; the authenticated Telegram
sender, ownership checks, user-timezone attachment, future time validation, and database
writes are controlled by the application service. Provider output containing its own UTC
offset is rejected; the user's validated IANA timezone is applied by the backend.

Creating a reminder transactionally writes one active reminder, one pending occurrence, one
pending Telegram notification, and the Telegram update's idempotency result. The confirmation
is sent only after commit and includes the user's local time and Telegram destination. VS-03
does not run delivery; the pending notification is consumed by the VS-04 scheduler worker.

VS-07 recurring reminders store the canonical weekly rule, intended IANA timezone, and a stable
local recurrence-instance key. The worker materializes a rolling 35-day horizon before claiming
due notifications and pages through active series so later reminders are not starved by the first
batch. PostgreSQL uniqueness and `INSERT ... ON CONFLICT` make replayed updates, worker restarts,
and concurrent materializers idempotent. Recurrence uses local wall-clock time: fall-back overlaps
choose the earlier instant, while spring-forward gaps shift forward by the DST gap while preserving
minutes. Delivery still uses the leased PostgreSQL notification worker, so provider timeouts can
remain `UNKNOWN` and strict exactly-once external delivery is not guaranteed. Ambiguous task,
weekday, or time input is clarified or rejected; weekly schedules are supported, not arbitrary
monthly/calendar rules.

VS-04 runs `reminder-worker` as an independent PostgreSQL-backed process. It polls due
notifications every `WORKER_POLL_INTERVAL_SECONDS`, claims rows with PostgreSQL
`FOR UPDATE SKIP LOCKED` and a lease, rechecks reminder/occurrence/destination eligibility,
then sends through Telegram. Confirmed successes are stored as `SENT` with Telegram's
provider message ID and are never retried. DNS/connect failures, connect timeouts, and pool
timeouts use exponential backoff because the request was not transmitted. Read/write failures
and Telegram 5xx responses are uncertain and receive at most two delayed retries. HTTP 429
responses use Telegram's `retry_after` value. Permanent errors become `FAILED`; exhausted
uncertain attempts become `UNKNOWN` for monitoring or manual recovery. Strict end-to-end
exactly-once delivery cannot be guaranteed after an ambiguous provider outcome.

VS-10 supports Telegram text cancellation. `Cancel the standup` cancels an active recurring
series and all of its outstanding pending occurrences; `Cancel only Monday's standup` cancels
just that pending occurrence and leaves the series active. Ambiguous task matches are clarified
before any state changes. Cancellation locks the owned reminder and pending notifications in
one transaction, preserves sent delivery history, and refuses to claim success if a notification
is already `SENDING`. Canceled reminders and occurrences are excluded by the existing worker
eligibility checks, so no stale pending notification is delivered after a committed cancellation.
Indirect cancellation language can be routed by the configured intent model; the model may only
propose target text and scope, while the backend resolves owned candidates and performs the write.

VS-11 supports marking a delivered occurrence done from Telegram. Reminder notifications include
a signed `Done` button, and replying `Done` to the original notification uses Telegram's stored
provider message ID. A bare `Done` is accepted only when exactly one delivered occurrence is
eligible; otherwise the bot asks which reminder was intended. Completion locks and rechecks the
owned delivered occurrence, completes the parent only for a one-time reminder, and leaves a
recurring series active. The occurrence state change and Telegram update idempotency response are
committed together, so replaying a message or button callback is safe. Callback signatures,
sender identity, private chat, and notification destination are all validated; callback UI
acknowledgement is best-effort and does not determine completion success.

Telegram also supports read-only progress questions such as `How many reminders are pending?`,
`What do I still have to do today?`, `How many tasks have I completed?`, and overdue counts.
These summaries are computed from the authenticated user's own reminder occurrences; delivered
but uncompleted reminders remain outstanding, and recurring instances are projected for today's
view when they have not yet been materialized.

The same worker also retries durable Telegram confirmations that failed after the inbound
command was committed. Confirmation responses are stored with the idempotency record, retried
after connectivity returns, and marked `SENT` only after Telegram confirms them. This prevents a
successful reminder from being silently accompanied by a permanently missing confirmation.

After a user completes timezone onboarding, the bot offers Telegram's native
`request_contact` button. The backend accepts only a contact whose `user_id`
matches the Telegram sender, normalizes valid numbers to E.164, and stores the
phone as a linked channel identity. Active phone numbers are database-unique, so
concurrent link attempts cannot authorize the same number for two users. Users may
choose `Skip for now` and continue using Telegram; `/link_phone` reopens the prompt.
If Telegram supplies a national-format number, the bot asks for a two-letter
country code such as `/phone_region PK` and then links the verified contact.

Run tests with:

```powershell
uv run pytest
```

Semantic memory uses PostgreSQL pgvector with `vector(1536)`, cosine distance,
and a provider-agnostic embedding boundary. It is disabled by default. Configure
the `EMBEDDING_*` values in `.env` (set `EMBEDDING_PROVIDER=gemini` and provide
`EMBEDDING_API_KEY` to use Google's `gemini-embedding-001`), apply migrations, then index one resumable
batch at a time with:

```powershell
uv run memory-embedding-backfill
```

Telegram's default memory indexing excludes detected passwords, PINs, access
codes, keys, tokens, and recovery secrets from external embedding unless the
deployment explicitly approves `EMBEDDING_ALLOW_SENSITIVE_EXTERNAL=true`.
Authenticated Vapi memory calls are the explicit voice-channel exception: they
may embed and speak sensitive facts. Original memory text remains the source of
truth; failed or disabled indexing never deletes a fact and lexical retrieval
continues to work. Existing rows previously marked `SKIPPED_SENSITIVE` need the
backfill command run with `EMBEDDING_ALLOW_SENSITIVE_EXTERNAL=true` before
semantic Vapi lookup can use them.

Memory retrieval remains deterministic-first. A single exact lexical match is
returned without Gemini or GPT-OSS. Otherwise Gemini/pgvector ranks at most the
authenticated user's top three non-sensitive candidates; similarity does not
reject them before the dedicated 120B answer prompt returns a validated
`ANSWER`, `AMBIGUOUS`, or `UNKNOWN` decision. Candidate indices and ownership are
rechecked, and exact values must remain unchanged. Telegram keeps sensitive
candidate filtering; authenticated Vapi retrieval explicitly allows sensitive
facts so the committed answer can be spoken to the caller. Provider failure
falls back to the deterministic unknown or clarification result.

Gemini's free tier is available for eligible projects, but Google does not
publish one universal embedding RPM/TPM/RPD value: quotas are project- and
tier-specific and must be checked in AI Studio. The adapter sends
`RETRIEVAL_DOCUMENT` for saved facts and `RETRIEVAL_QUERY` for questions and
normalizes the 1,536-dimensional truncated prefix. A Gemini key is not present
in this checkout, so the live-provider test is skipped; fake-response coverage
is included.

When running PostgreSQL integration tests, set `TEST_DATABASE_URL` to the dedicated
`memorae_test` database on port 5434. The test fixture refuses to run destructive cleanup if
`TEST_DATABASE_URL` is the same database as `DATABASE_URL`; the application database is never a
test target.

The VS-07 schema change is migration `0013_recurring_occurrence_keys`; run
`uv run alembic upgrade head` before starting the API or worker. A real Telegram smoke test
requires a configured bot token, a linked Telegram user, a public webhook endpoint, and a user
message such as `Remind me every Monday at 8 AM to check the smoke test`; verify the confirmation,
the recurring rows, and the next Telegram delivery. Do not use a model- or client-supplied user ID
to choose the destination.

Smoke-test record (2026-11-01): the configured bot token was present, but both the Python and
host PowerShell calls to Telegram `getMe` failed during TLS connection establishment. No inbound
message or outbound smoke notification was claimed as successful; rerun the procedure above from
a network path that can reach `api.telegram.org`.

## Structured logs

The API and worker write concise Colorama-coloured text lines to stdout by default. Set
`LOG_COLOR=false` for plain text, `LOG_LEVEL=DEBUG` for lower-level diagnostics, or
`LOG_FORMAT=json` for machine-readable output (JSON never contains ANSI colour codes).
Set `LOG_INCLUDE_TRACEBACK=true` when a full traceback is needed. Useful events include
`http_request_completed`,
`webhook_auth_failed`, `telegram_update_received`, `telegram_api_request_started`,
`telegram_api_response_received`, `telegram_api_request_failed`, `telegram_delivery_failed`,
`telegram_response_delivery_attempt`, `telegram_response_delivery_retry`,
`telegram_update_handled`, `telegram_update_processed`, `reminder_command_started`,
`reminder_intent_rules_finished`,
`reminder_intent_interpreted`, `reminder_schedule_rejected`, `reminder_created`,
`clarification_session_saved`, `reminder_notification_scheduled`,
`reminder_confirmation_staged`, `reminder_persistence_failed`, `notification_worker_started`,
`notification_worker_iteration_started`, `notification_claimed`,
`notification_delivery_attempt_started`, `notification_delivered`, and
`notification_delivery_failed`, and `notification_delivery_outcome_recorded`. Logs include request IDs, update IDs, notification IDs, status
codes, due-time lateness, parser/provider path, outcome, exception type, and sanitized network
cause. Confirmation events include `confirmation_claimed`, `confirmation_delivery_attempt_started`,
`confirmation_delivery_failed`, `confirmation_delivery_outcome_recorded`, and
`confirmation_delivered`. VS-10 adds `cancellation_command_started`,
`cancellation_clarification_saved`, `cancellation_extraction_failed`, `reminder_canceled`, and
`reminder_cancellation_conflict`. VS-11 adds `completion_target_ambiguous`,
`reminder_occurrence_completed`, and `telegram_callback_acknowledgement_failed`. Logs never contain tokens, secrets, prompts, reminder text, or personal
data.

Telegram webhook updates are idempotent. Each update gets one logical outbound response. A
Telegram rate-limit response (429) may be retried up to the configured limit; network timeouts,
connection failures, and 5xx responses are never retried because Telegram may have accepted the
message before the response was lost. Those ambiguous outcomes are recorded as `UNKNOWN`, and a
replayed update is acknowledged without sending again. Inspect `telegram_delivery_failed` and
`telegram_response_delivery_retry` logs for the network/provider cause.
