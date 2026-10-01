PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup install test lint format db-init migrate run

# Create the venv (uses virtualenv because python3.12-venv/ensurepip is absent
# on this machine; `python -m venv` works on a normal install).
setup:
	/usr/bin/python3.12 -m venv .venv 2>/dev/null || virtualenv -p /usr/bin/python3.12 .venv
	$(PIP) install -e ".[dev]"

install:
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

format:
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests

# Alembic migrations (works on Postgres and SQLite).
migrate:
	.venv/bin/alembic upgrade head

# Dev convenience: create tables without Alembic (fine for throwaway SQLite dbs).
db-init:
	$(PY) -m fxterminal.cli db init

db-check:
	$(PY) -m fxterminal.cli db check

# Start embedded Postgres via pgserver (persists across runs; data in pgdata/)
pg-start:
	$(PY) -c "import pgserver; s=pgserver.get_server('pgdata', cleanup_mode=None); print(s.get_uri())"
	$(PY) -c "import pgserver; s=pgserver.get_server('pgdata', cleanup_mode=None); s.psql('CREATE DATABASE fxterminal')" 2>/dev/null || true

pg-stop:
	$(PY) -c "import pgserver; pgserver.pg_ctl('pgdata', 'stop -m fast')" || true

ingest:
	$(PY) -m fxterminal.cli ingest all

run:
	.venv/bin/uvicorn fxterminal.api.main:app --reload

# Phase 8: scheduling + deployment
cycle:
	$(PY) -m fxterminal.cli cycle

schedule:
	$(PY) -m fxterminal.cli serve

install-services:
	install -m644 deploy/fxterminal-api.service deploy/fxterminal-scheduler.service ~/.config/systemd/user/
	systemctl --user daemon-reload
	@echo "enabled: systemctl --user enable --now fxterminal-api@$(USER) fxterminal-scheduler@$(USER)"
