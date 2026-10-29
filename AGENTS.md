# Repository Guidelines

## Project Structure & Module Organization

This repository is currently design-first. Read [docs/Architecture.md](docs/Architecture.md) for system behavior, data concepts, channel flows, and reliability constraints; use [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the vertical-slice delivery order. `README.md` is currently minimal. As implementation is added, keep the Python/FastAPI application under `app/`, tests under `tests/`, database migrations under `alembic/`, and operational files such as `docker-compose.yml` and `.env.example` at the repository root. Keep Telegram, Vapi, and other provider adapters at the channel/integration boundary; put authorization and reminder/memory business rules in shared services.

## Build, Test, and Development Commands

No build or test tooling is committed yet. The planned local workflow is:

- `docker compose up` — start the application and local PostgreSQL once Compose is added.
- `alembic upgrade head` — apply database migrations.
- `uvicorn app.main:app --reload` — run the FastAPI service locally.
- `pytest` — run the automated suite.

Update this section when the project adopts a definitive package manager, scripts, or Make targets.

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, type hints, and small modules organized around domain responsibilities. Prefer `snake_case` for files, functions, and variables; `PascalCase` for classes and typed schemas; and uppercase names for constants. Keep provider-specific payload parsing in adapters and validate all external input before calling services. Adopt Ruff/Black (or document the chosen equivalent) when code is introduced.

## Testing Guidelines

Use pytest. Cover schedule parsing and time zones, authorization and cross-user isolation, idempotency/replay behavior, reminder delivery, and worker concurrency. Mock external Telegram/Vapi/Groq calls in unit tests, while retaining provider smoke tests and at least one real-PostgreSQL integration test. Name tests `test_<behavior>` and run `pytest` before submitting changes.

## Commit & Pull Request Guidelines

History currently contains only `Initial commit`, so no established convention exists. Use short, imperative subjects such as `Add reminder creation service`; keep commits focused. PRs should explain the behavior changed, link the relevant plan item or issue, include tests and configuration notes, and document any webhook, migration, or API changes. Add screenshots or a short demo transcript when changing user-facing Telegram or voice behavior.

## Security & Configuration

Never commit secrets, personal data, or real provider credentials. Use environment variables documented in `.env.example`; keep `.env` local. Enforce verified user/channel ownership in services rather than trusting LLM or tool arguments, and avoid logging saved facts, tokens, phone numbers, or webhook secrets.
