"""Structured logging via structlog.

`redact_secrets` scrubs any processor-field whose key looks sensitive so API
keys/tokens can never reach log output even if a caller passes them.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

_SECRET_KEY_RE = re.compile(r"(key|token|secret|password|credential)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(r"(sk-ant-[A-Za-z0-9_-]{8,}|Bearer\s+\S+)", re.IGNORECASE)
REDACTED = "***redacted***"


def _scrub(value: Any) -> Any:
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return _SECRET_VALUE_RE.sub(REDACTED, value)
    if isinstance(value, dict):
        return {
            k: (REDACTED if _SECRET_KEY_RE.search(str(k)) else _scrub(v)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_scrub(v) for v in value)
    return value


def redact_secrets(
    logger: logging.Logger, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor: redact secrets in field names and string values."""
    return {
        k: (REDACTED if _SECRET_KEY_RE.search(str(k)) else _scrub(v)) for k, v in event_dict.items()
    }


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure stdlib logging + structlog. Call once at process start."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        redact_secrets,
    ]

    renderer: Any = (
        structlog.dev.ConsoleRenderer() if fmt == "console" else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
