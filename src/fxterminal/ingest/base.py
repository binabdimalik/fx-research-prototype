"""Provider interfaces + shared resilient HTTP client.

Every data category has a Protocol: swap a provider by writing a new class that
satisfies it and registering it in `pipeline.py`. No provider writes to the DB
directly — they return normalized types and `store.py` persists them.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

from fxterminal.core.logging import get_logger
from fxterminal.ingest.types import MacroObs, PriceBar, RateExp, RawDoc

logger = get_logger(__name__)


class ProviderError(Exception):
    """Raised when a provider cannot fetch or parse its source. The pipeline
    catches this per-provider so one failing source never stops the others."""


class RateLimitError(ProviderError):
    """HTTP 429 / explicit rate-limit response from a provider."""


class FetchClient:
    """httpx wrapper with retry/backoff for transient + rate-limit failures.

    Only retries idempotent GETs. Retry budget is deliberately small: ingestion
    is re-runnable and idempotent, so failing fast + retrying on the next
    scheduled run is preferred over long waits.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str = "fxterminal/0.1",
        max_retries: int = 3,
        base_backoff: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
        breaker_threshold: int = 5,
        breaker_cooldown_s: float = 300.0,
    ) -> None:
        # `transport` is injectable for tests (httpx.MockTransport).
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "*/*"},
            follow_redirects=True,
            transport=transport,
        )
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        # circuit breaker: per-host consecutive-failure count + open-until.
        # After `breaker_threshold` straight transient failures we fail fast
        # for `breaker_cooldown_s` instead of hammering a dead endpoint.
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown = timedelta(seconds=breaker_cooldown_s)
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, datetime] = {}

    def _circuit_check(self, url: str) -> str:
        """Return host, raising ProviderError fast if its circuit is open."""
        host = urlparse(url).netloc
        until = self._open_until.get(host)
        if until is not None and datetime.now(UTC) < until:
            raise ProviderError(
                f"circuit open for {host} until {until.isoformat()} "
                f"({self._failures.get(host, 0)} consecutive failures)"
            )
        return host

    def _record(self, host: str, ok: bool) -> None:
        if ok:
            self._failures.pop(host, None)
            self._open_until.pop(host, None)
            return
        n = self._failures.get(host, 0) + 1
        self._failures[host] = n
        if n >= self._breaker_threshold:
            self._open_until[host] = datetime.now(UTC) + self._breaker_cooldown
            logger.error("circuit_opened", host=host, failures=n)

    async def get_text(self, url: str) -> str:
        """GET url → response text. Raises ProviderError on failure."""
        host = self._circuit_check(url)
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = await self._client.get(url)
            except httpx.TransportError as e:
                last_err = e
                logger.warning("http_retry", url=url, attempt=attempt, error=str(e)[:200])
            else:
                if resp.status_code == 429:
                    last_err = RateLimitError(f"{url} -> 429")
                    retry_after = resp.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else self._base_backoff * attempt
                    await asyncio.sleep(min(wait, 30.0))
                    continue
                if 500 <= resp.status_code < 600:
                    last_err = ProviderError(f"{url} -> {resp.status_code}")
                    await asyncio.sleep(self._base_backoff * attempt)
                    continue
                if resp.status_code >= 400:
                    # 4xx other than 429 are not retryable
                    raise ProviderError(f"{url} -> HTTP {resp.status_code}") from None
                self._record(host, ok=True)
                return resp.text
            if attempt < self._max_retries:
                await asyncio.sleep(self._base_backoff * attempt)
        self._record(host, ok=False)
        raise ProviderError(f"{url} failed after {self._max_retries} attempts: {last_err}")

    async def aclose(self) -> None:
        await self._client.aclose()


@runtime_checkable
class PriceProvider(Protocol):
    """Yields historical FX bars. `pairs` uses 6-char convention (EURUSD)."""

    @property
    def name(self) -> str: ...

    async def fetch_prices(
        self, pairs: list[str], since: str, until: str | None = None
    ) -> list[PriceBar]: ...


@runtime_checkable
class DocumentProvider(Protocol):
    """Yields documents (news, statements, minutes, speeches)."""

    @property
    def name(self) -> str: ...

    async def fetch_documents(self) -> list[RawDoc]: ...


@runtime_checkable
class MacroProvider(Protocol):
    """Yields macro release observations."""

    @property
    def name(self) -> str: ...

    async def fetch_macro(self, currencies: list[str]) -> list[MacroObs]: ...


@runtime_checkable
class RateExpectationProvider(Protocol):
    """Yields market-implied policy rate expectations."""

    @property
    def name(self) -> str: ...

    async def fetch_expectations(self, banks: list[str]) -> list[RateExp]: ...
