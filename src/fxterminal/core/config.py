"""Application configuration.

Secrets are read from environment variables / a local `.env` file and held as
`SecretStr` so they are never rendered by `repr()`/logging. Use
`Settings.safe_dict()` when you need a loggable snapshot of configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Central bank -> currency coverage. Every currency in `currencies` is
# expected to have an entry here; used for rates ingest, stance scoring,
# theme doc selection and the offline chat answerer.
BANK_CURRENCIES: dict[str, str] = {
    "FED": "USD",
    "ECB": "EUR",
    "BOJ": "JPY",
    "BOE": "GBP",
    "SNB": "CHF",
    "RBA": "AUD",
    "RBNZ": "NZD",
}


class Settings(BaseSettings):
    """Runtime settings, overridable via `FXT_*` environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FXT_",
        extra="ignore",
    )

    app_env: Literal["dev", "test", "prod"] = "dev"

    # e.g. postgresql+asyncpg://user:pass@host:5432/fxterminal
    #      sqlite+aiosqlite:///./fxterminal.db
    database_url: str = "sqlite+aiosqlite:///./fxterminal.db"
    database_echo: bool = False

    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: SecretStr | None = None
    # Verify current model names at https://docs.anthropic.com/en/docs/about-claude/models
    llm_model: str = "claude-sonnet-4-5"
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # LLM backend selection. "anthropic" uses anthropic_api_key + llm_model.
    # "openai_compat" talks to any OpenAI-compatible /chat/completions endpoint
    # (Groq, OpenRouter, DeepSeek, Together, Ollama at http://localhost:11434/v1)
    # using llm_api_key + llm_model + llm_base_url.
    llm_provider: Literal["anthropic", "openai_compat"] = "anthropic"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None

    http_timeout_seconds: float = 30.0
    http_user_agent: str = "fxterminal/0.1 (research; contact: local user)"

    # --- Ingestion providers (Phase 2) ---
    # Free key: https://fred.stlouisfed.org/docs/api/api_key.html
    fred_api_key: SecretStr | None = None
    # Trading Economics calendar — PAID key required (guest tier was
    # discontinued). https://tradingeconomics.com/api/pricing.aspx
    te_api_key: SecretStr | None = None
    # Frankfurter API root — verify at https://frankfurter.dev/ (the legacy
    # api.frankfurter.app host redirects here as of Sep 2026).
    frankfurter_base_url: str = "https://api.frankfurter.dev"
    # Local JSON file for market-implied rate expectations (see
    # src/fxterminal/ingest/providers/rates_file.py for the shape).
    rate_expectations_file: str = "data/rate_expectations.json"

    # Intraday prices: unofficial Yahoo Finance chart endpoint (no key).
    # Provides '1h' bars on a fast tick; Frankfurter stays the official daily fix.
    intraday_prices_enabled: bool = True
    intraday_interval_minutes: int = 60

    # Comma-separated, e.g. FXT_CURRENCIES=USD,EUR,JPY,GBP. Kept as `str`
    # because pydantic-settings JSON-decodes list fields before validators run;
    # use `currency_list` for the parsed, validated form.
    currencies: str = "USD,EUR,JPY,GBP,CHF,NZD,AUD"

    # Deterministic risk filters applied to every trade idea.
    max_spread_pips: float = 3.0
    event_blackout_hours: int = 4

    # Ingestion validation: reject a price bar whose close deviates from the
    # previous close for the same pair/timeframe by more than this fraction.
    # 10% catches unit errors/bad ticks; no real FX daily move approaches it.
    max_price_jump_pct: float = 0.10

    # --- Scheduling + alerts (Phase 8) ---
    # Hours between pipeline cycles when `fxterminal serve` runs the scheduler.
    schedule_interval_hours: float = 24.0
    # Optional generic webhook (Slack/Discord/custom) for alert delivery —
    # alerts always go to the structured log regardless.
    alert_webhook: str | None = None

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator("currencies")
    @classmethod
    def _validate_currencies(cls, v: str) -> str:
        """Normalize `usd, chf` -> `USD,CHF` and reject malformed/duplicate codes."""
        items = [c.strip().upper() for c in v.split(",") if c.strip()]
        if not items:
            raise ValueError("at least one currency is required")
        for c in items:
            if len(c) != 3 or not c.isalpha():
                raise ValueError(f"invalid ISO-4217-style currency code: {c!r}")
        if len(set(items)) != len(items):
            raise ValueError("duplicate currency codes")
        return ",".join(items)

    @property
    def currency_list(self) -> list[str]:
        """Parsed, validated currency universe."""
        return self.currencies.split(",")

    @property
    def bank_list(self) -> list[str]:
        """Central banks covering the configured currencies, in universe order."""
        wanted = set(self.currency_list)
        return [b for b, c in BANK_CURRENCIES.items() if c in wanted]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def safe_dict(self) -> dict[str, object]:
        """Config snapshot safe to log: secrets replaced with a boolean flag."""
        data = self.model_dump()
        for name, field in type(self).model_fields.items():
            value = data.get(name)
            is_secret_field = "SecretStr" in str(field.annotation)
            if isinstance(value, SecretStr):
                data[name] = "***set***" if value.get_secret_value() else "***unset***"
            elif is_secret_field:
                data[name] = "***set***" if value else "***unset***"
        return data


@lru_cache
def get_settings() -> Settings:
    """Cached settings; call `get_settings.cache_clear()` after env changes."""
    return Settings()
