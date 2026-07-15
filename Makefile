.PHONY: install migrate run test lint format postgres

install:
	pip install -e ".[dev]"

migrate:
	alembic upgrade head

run:
	future-self-bot

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

postgres:
	docker compose up -d postgres
