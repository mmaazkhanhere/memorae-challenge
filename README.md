# Memorae

## Local development

VS-01 implements Telegram onboarding with FastAPI, SQLAlchemy, Alembic, and PostgreSQL.

```powershell
uv sync --extra test
docker compose up -d postgres
$env:DATABASE_URL = "postgresql+psycopg://memorae:memorae@localhost:5433/memorae"
uv run alembic upgrade head
uv run uvicorn reminder_agent.main:app --reload
```

Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_WEBHOOK_SECRET` using `.env.example`. Set the Telegram webhook to `POST /webhook/telegram` with the same secret token.

Run tests with:

```powershell
uv run pytest
```

## Structured logs

The backend writes concise structured text lines to stdout by default. Set `LOG_FORMAT=json` for JSON output or `LOG_INCLUDE_TRACEBACK=true` when a full traceback is needed. Useful events include `http_request_completed`, `webhook_auth_failed`, `telegram_update_received`, `telegram_delivery_failed`, and `telegram_update_processed`. Logs include request IDs, update IDs, status codes, durations, and safe Telegram API error details, but never tokens, secrets, message text, or personal data.

Telegram webhook updates are idempotent. Each update gets one outbound response attempt; a retry is acknowledged without sending a duplicate. If Telegram delivery times out after the database mutation commits, the delivery is recorded as `UNKNOWN` and the update is acknowledged to stop Telegram replay. Inspect `telegram_delivery_failed` logs for the network/provider cause.
