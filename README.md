# Memorae

## Local development

VS-03 implements Telegram onboarding, verified phone linking, and one-time reminder creation
with FastAPI, SQLAlchemy, Alembic, and PostgreSQL.

```powershell
uv sync --extra test
docker compose up -d postgres
$env:DATABASE_URL = "postgresql+psycopg://memorae:memorae@localhost:5433/memorae"
uv run alembic upgrade head
uv run uvicorn reminder_agent.main:app --reload
```

Copy `.env.example` to `.env` and configure `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_WEBHOOK_SECRET`. `PHONE_DEFAULT_REGION` is only a deployment-wide
fallback; in a multi-country deployment, users with local-format contacts can
set their own region with `/phone_region XX`. Set the Telegram webhook to
`POST /webhook/telegram` with the same secret token.

Production startup fails fast unless `DATABASE_URL` uses PostgreSQL,
`TELEGRAM_BOT_TOKEN` is configured, and `TELEGRAM_WEBHOOK_SECRET` is a unique value of at
least 32 characters. The sample placeholders and local SQLite defaults are intended only for
injected test applications. `/ready` also verifies that the domain tables exist, so a stale
`alembic_version` row cannot make an empty database appear healthy.
The additive schema-repair migrations also rebuild missing domain tables when a
database was reset without resetting its Alembic version row.

One-time reminder text is interpreted through a strict typed boundary. The built-in parser
handles relative requests such as `Remind me in five minutes to call the bank` and
`Remind me tomorrow at 9 to renew the domain`. By default, a configured API key uses
OpenAI's `https://api.openai.com/v1` endpoint and the `gpt-6-luna` model. `LLM_BASE_URL`,
`LLM_API_KEY`, and `LLM_MODEL` can instead target any OpenAI-compatible `/chat/completions`
endpoint, including a Groq or Gemini-compatible gateway, without coupling reminder business
rules to a provider. The model can only propose a task and time; the authenticated Telegram
sender, ownership checks, user-timezone attachment, future time validation, and database
writes are controlled by the application service. Provider output containing its own UTC
offset is rejected; the user's validated IANA timezone is applied by the backend.

Creating a reminder transactionally writes one active reminder, one pending occurrence, one
pending Telegram notification, and the Telegram update's idempotency result. The confirmation
is sent only after commit and includes the user's local time and Telegram destination. VS-03
does not run delivery; the pending notification is consumed by the VS-04 scheduler worker.

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

## Structured logs

The backend writes concise structured text lines to stdout by default. Set `LOG_FORMAT=json` for JSON output or `LOG_INCLUDE_TRACEBACK=true` when a full traceback is needed. Useful events include `http_request_completed`, `webhook_auth_failed`, `telegram_update_received`, `telegram_delivery_failed`, `telegram_response_delivery_attempt`, `telegram_response_delivery_retry`, `telegram_update_processed`, `reminder_command_started`, `reminder_intent_interpreted`, `reminder_schedule_rejected`, `reminder_created`, `reminder_persistence_failed`, and `telegram_response_delivery_succeeded`. Logs include request IDs, update IDs, status codes, durations, parser/provider path, outcome, and error type, but never tokens, secrets, prompts, reminder text, or personal data.

Telegram webhook updates are idempotent. Each update gets one logical outbound response with up to three attempts for transient network, rate-limit, or 5xx failures; a retry is acknowledged without sending a duplicate. Permanent Telegram failures are not retried. If delivery remains uncertain after the configured attempts, the mutation stays durable and the delivery is recorded as `UNKNOWN`; the update is acknowledged to stop Telegram replay. Inspect `telegram_delivery_failed` and `telegram_response_delivery_retry` logs for the network/provider cause.
