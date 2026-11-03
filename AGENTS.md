# Repository Guidelines

## Project Structure & Module Organization

Read [docs/Architecture.md](docs/Architecture.md) for system behavior, data concepts, channel flows, and reliability constraints; use [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the vertical-slice delivery order. The Python/FastAPI application lives under `src/reminder_agent/`, tests under `tests/`, and database migrations under `alembic/`. Operational files such as `docker-compose.yml` and `.env.example` live at the repository root. Keep Telegram, Vapi, and other provider adapters at the channel/integration boundary; put authorization and reminder/memory business rules in shared services.

## Architecture Decision Records

Store Architecture Decision Records (ADRs) in `docs/adr/`, using zero-padded sequential filenames such as `0001-decision-name.md`. ADRs use the title, Status, Date, Context, Decision, Alternatives Considered, Consequences, Implementation References, and Relationship to Original Architecture sections. Use `Accepted` for confirmed decisions; when the original decision date is not reliably known, record the ADR documentation/acceptance date and say so. Use relative Markdown links for related ADRs, distinguish positive consequences from trade-offs, and do not alter an existing ADR unless explicitly requested.

## Build, Test, and Development Commands

The local workflow is:

- Use `uv` exclusively for Python dependency installation, locking, and command execution; do not use `pip` directly.
- `uv sync --extra test` — install the locked project and test dependencies.
- `docker compose up -d postgres` — start local PostgreSQL.
- `uv run alembic upgrade head` — apply database migrations.
- `uv run uvicorn reminder_agent.main:app --reload` — run the FastAPI service locally.
- `uv run pytest` — run the automated suite.

Keep commands and configuration documentation current as new slices add workers or provider integrations.

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, type hints, and small modules organized around domain responsibilities. Prefer `snake_case` for files, functions, and variables; `PascalCase` for classes and typed schemas; and uppercase names for constants. Keep provider-specific payload parsing in adapters and validate all external input before calling services. Adopt Ruff/Black (or document the chosen equivalent) when code is introduced.

## Testing Guidelines

Use pytest. Cover schedule parsing and time zones, authorization and cross-user isolation, idempotency/replay behavior, reminder delivery, and worker concurrency. Mock external Telegram/Vapi/Groq calls in unit tests, while retaining provider smoke tests and at least one real-PostgreSQL integration test. Name tests `test_<behavior>` and run `pytest` before submitting changes.

## Commit & Pull Request Guidelines

Use Conventional Commits with a concise, imperative subject: `<type>(<scope>): <description>`. Keep commits focused and atomic; stage only files relevant to that change. PRs should explain the behavior changed, link the relevant plan item or issue, include tests and configuration notes, and document any webhook, migration, or API changes. Add screenshots or a short demo transcript when changing user-facing Telegram or voice behavior.

## Security & Configuration

Never commit secrets, personal data, or real provider credentials. Use environment variables documented in `.env.example`; keep `.env` local. Enforce verified user/channel ownership in services rather than trusting LLM or tool arguments, and avoid logging saved facts, tokens, phone numbers, or webhook secrets.

## Operational Notes

The application and worker run as separate processes. Use `uv run reminder-worker` for the
local worker, or `docker compose up -d` to run migrations and supervise both services.
PostgreSQL integration tests must use the dedicated `TEST_DATABASE_URL` database, started with
`docker compose --profile test up -d postgres-test`; never point it at `DATABASE_URL`.
