.PHONY: run stop logs test

run:
	docker compose up --build

stop:
	docker compose down

logs:
	docker compose logs -f api worker

test:
	uv run pytest
