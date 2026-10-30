# Memorae

## Local development

VS-02 implements Telegram onboarding and verified phone linking with FastAPI, SQLAlchemy,
Alembic, and PostgreSQL.

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

After a user completes timezone onboarding, the bot offers Telegram's native
`request_contact` button. The backend accepts only a contact whose `user_id`
matches the Telegram sender, normalizes valid numbers to E.164, and stores the
phone as a linked channel identity. Active phone numbers are database-unique, so
concurrent link attempts cannot authorize the same number for two users. Users may
choose `Skip for now` and continue using Telegram; `/link_phone` reopens the prompt.
If Telegram supplies a national-format number, the bot asks for a two-letter
country code such as `/phone_region PK` before retrying the native contact share.

Run tests with:

```powershell
uv run pytest
```

## Structured logs

The backend writes concise structured text lines to stdout by default. Set `LOG_FORMAT=json` for JSON output or `LOG_INCLUDE_TRACEBACK=true` when a full traceback is needed. Useful events include `http_request_completed`, `webhook_auth_failed`, `telegram_update_received`, `telegram_delivery_failed`, and `telegram_update_processed`. Logs include request IDs, update IDs, status codes, durations, and safe Telegram API error details, but never tokens, secrets, message text, or personal data.

Telegram webhook updates are idempotent. Each update gets one outbound response attempt; a retry is acknowledged without sending a duplicate. If Telegram delivery times out after the database mutation commits, the delivery is recorded as `UNKNOWN` and the update is acknowledged to stop Telegram replay. Inspect `telegram_delivery_failed` logs for the network/provider cause.
