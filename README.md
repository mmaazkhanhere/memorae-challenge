# Memorae

## Local development

VS-03 implements Telegram onboarding, verified phone linking, and one-time reminder creation
with FastAPI, SQLAlchemy, Alembic, and PostgreSQL.

```powershell
uv sync --extra test
docker compose up -d postgres
docker compose --profile test up -d postgres-test
$env:DATABASE_URL = "postgresql+psycopg://memorae:memorae@localhost:5433/memorae"
$env:TEST_DATABASE_URL = "postgresql+psycopg://memorae:memorae@localhost:5434/memorae_test"
uv run alembic upgrade head
uv run uvicorn reminder_agent.main:app --reload
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

Production startup fails fast unless `DATABASE_URL` uses PostgreSQL,
`TELEGRAM_BOT_TOKEN` is configured, and `TELEGRAM_WEBHOOK_SECRET` is a unique value of at
least 32 characters. The sample placeholders and local SQLite defaults are intended only for
injected test applications. `/ready` also verifies that the domain tables exist, so a stale
`alembic_version` row cannot make an empty database appear healthy. Readiness also requires the
current migration head (`0012_conversation_session_version`) and the worker/confirmation claim
columns, so a partially migrated database is rejected before webhooks are accepted.
The additive schema-repair migrations also rebuild missing domain tables when a
database was reset without resetting its Alembic version row.

One-time reminder text uses a confidence-first, strict typed boundary. The built-in parser
only handles short greetings and thanks plus unambiguous one-time reminders such as
`Remind me in five minutes to call the bank`, `Remind me tomorrow at 9 AM to check my email`,
`Remind me on Friday at 6 PM to go to the gym`, and `Remind me to pay the bill at 8 PM today`.
Anything requiring context, a clarification, recurrence, or conversational interpretation falls
back to the configured model. By default, a configured API key uses
OpenAI's `https://api.openai.com/v1` endpoint and the `gpt-6-luna` model. `LLM_BASE_URL`,
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

When running PostgreSQL integration tests, set `TEST_DATABASE_URL` to the dedicated
`memorae_test` database on port 5434. The test fixture refuses to run destructive cleanup if
`TEST_DATABASE_URL` is the same database as `DATABASE_URL`; the application database is never a
test target.

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
`confirmation_delivered`. Logs never contain tokens, secrets, prompts, reminder text, or personal
data.

Telegram webhook updates are idempotent. Each update gets one logical outbound response. A
Telegram rate-limit response (429) may be retried up to the configured limit; network timeouts,
connection failures, and 5xx responses are never retried because Telegram may have accepted the
message before the response was lost. Those ambiguous outcomes are recorded as `UNKNOWN`, and a
replayed update is acknowledged without sending again. Inspect `telegram_delivery_failed` and
`telegram_response_delivery_retry` logs for the network/provider cause.
