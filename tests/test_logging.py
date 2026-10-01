"""Logging: secrets must never reach log output."""

from __future__ import annotations

import structlog

from fxterminal.core.logging import REDACTED, configure_logging, redact_secrets


def test_redact_secret_fields_and_values() -> None:
    event = {
        "event": "calling llm",
        "api_key": "sk-ant-secret123456",
        "note": "auth failed for sk-ant-secret123456",
        "nested": {"token": "abc", "items": [{"password": "p"}, "ok"]},
    }
    out = redact_secrets(structlog.get_logger(__name__), "info", dict(event))
    assert out["api_key"] == REDACTED
    assert "sk-ant-secret123456" in event["api_key"]  # input untouched
    assert out["note"] == "auth failed for " + REDACTED
    assert out["nested"]["token"] == REDACTED
    assert out["nested"]["items"][0]["password"] == REDACTED


def test_configure_logging_json(capsys) -> None:
    configure_logging("INFO", "json")
    structlog.get_logger("t").info("hello", api_key="sk-ant-should-hide")
    line = capsys.readouterr().out.strip()
    assert "sk-ant-should-hide" not in line
    assert REDACTED in line
