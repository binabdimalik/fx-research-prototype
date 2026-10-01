"""Config: env overrides, validation, and secret masking."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from fxterminal.core.config import Settings


def test_defaults_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from ambient FXT_* env vars (env beats _env_file=None).
    for var in list(__import__("os").environ):
        if var.startswith("FXT_"):
            monkeypatch.delenv(var)
    s = Settings(_env_file=None)
    assert s.app_env == "dev"
    assert s.is_sqlite  # clean-checkout default must not require a server
    assert s.currency_list == ["USD", "EUR", "JPY", "GBP", "CHF", "NZD", "AUD"]
    assert s.bank_list == ["FED", "ECB", "BOJ", "BOE", "SNB", "RBA", "RBNZ"]
    assert s.anthropic_api_key is None


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FXT_APP_ENV", "prod")
    monkeypatch.setenv("FXT_DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/fx")
    monkeypatch.setenv("FXT_CURRENCIES", "usd, chf ,cad")
    s = Settings(_env_file=None)
    assert s.app_env == "prod"
    assert not s.is_sqlite
    assert s.currency_list == ["USD", "CHF", "CAD"]  # normalized + split


def test_currency_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(currencies="USD,usd", _env_file=None)  # dup after normalize
    with pytest.raises(ValidationError):
        Settings(currencies="TOOLONG", _env_file=None)
    with pytest.raises(ValidationError):
        Settings(currencies="", _env_file=None)


def test_secret_never_rendered() -> None:
    s = Settings(anthropic_api_key=SecretStr("sk-ant-real-key-123"), _env_file=None)
    dump = s.safe_dict()
    assert dump["anthropic_api_key"] == "***set***"
    # No path through repr/dump exposes the raw key.
    assert "sk-ant-real-key-123" not in repr(s)
    assert "sk-ant-real-key-123" not in repr(dump)


def test_unset_secret_masks_as_unset() -> None:
    assert Settings(_env_file=None).safe_dict()["anthropic_api_key"] == "***unset***"
