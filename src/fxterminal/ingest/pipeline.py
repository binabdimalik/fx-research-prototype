"""Ingestion orchestrator: run providers, persist, isolate failures.

Each provider runs independently — a failure is logged and recorded in the
report, never propagated. That keeps one flaky feed from starving the rest.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.core.config import Settings, get_settings
from fxterminal.core.logging import get_logger
from fxterminal.ingest.base import FetchClient
from fxterminal.ingest.providers.frankfurter import FrankfurterProvider
from fxterminal.ingest.providers.fred import FredMacroProvider
from fxterminal.ingest.providers.rates_file import JsonRateExpectationProvider
from fxterminal.ingest.providers.rss import DEFAULT_FEEDS, FeedSpec, RssDocumentProvider
from fxterminal.ingest.providers.tradingeconomics import TradingEconomicsMacroProvider
from fxterminal.ingest.providers.yahoo import YahooFxProvider
from fxterminal.ingest.store import (
    store_documents,
    store_macro,
    store_prices,
    store_rate_expectations,
)
from fxterminal.ingest.types import IngestReport

logger = get_logger(__name__)


# Market quoting priority: the currency earlier in this list is the BASE.
# EUR > GBP > USD > JPY gives EURUSD, GBPUSD, USDJPY, EURJPY, EURGBP, GBPJPY.
_QUOTE_PRIORITY = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]


def default_pairs(currencies: list[str]) -> list[str]:
    """All crosses among configured currencies in standard market convention.

    For USD,EUR,JPY,GBP -> EURUSD, GBPUSD, USDJPY, EURJPY, EURGBP, GBPJPY.
    """
    have = [c for c in _QUOTE_PRIORITY if c in currencies]
    return [f"{a}{b}" for i, a in enumerate(have) for b in have[i + 1 :]]


def build_providers(
    settings: Settings | None = None,
    client: FetchClient | None = None,
    extra_feeds: Sequence[FeedSpec] = (),
) -> dict[str, Any]:
    """Construct the default provider set. Swap/replace entries to change
    sources — the pipeline only needs objects satisfying the Protocols."""
    settings = settings or get_settings()
    client = client or FetchClient(
        timeout=settings.http_timeout_seconds,
        user_agent=settings.http_user_agent,
    )
    feeds = list(extra_feeds) if extra_feeds else list(DEFAULT_FEEDS)

    providers: dict[str, Any] = {
        "prices": FrankfurterProvider(client, base_url=settings.frankfurter_base_url),
        "documents": [RssDocumentProvider(client, spec) for spec in feeds],
        "rates": JsonRateExpectationProvider(settings.rate_expectations_file),
        # '1h' intraday bars on a fast tick; Frankfurter stays the daily fix.
        "intraday_prices": (
            YahooFxProvider(client) if settings.intraday_prices_enabled else None
        ),
    }
    # macro providers — zero or more; run_all reports 'skipped' when empty
    macro: list[Any] = []
    if settings.fred_api_key and settings.fred_api_key.get_secret_value():
        macro.append(FredMacroProvider(client, api_key=settings.fred_api_key.get_secret_value()))
    if settings.te_api_key and settings.te_api_key.get_secret_value():
        macro.append(
            TradingEconomicsMacroProvider(client, api_key=settings.te_api_key.get_secret_value())
        )
    providers["macro"] = macro
    return providers


async def _record_run(session: AsyncSession, rep: IngestReport, started: datetime) -> None:
    """Persist one provider_runs row — feeds the Data Health view and
    staleness alerts. A failing ingest still gets recorded, after rollback."""
    from fxterminal.models import ProviderRun

    row = ProviderRun(
        provider=rep.provider,
        started_at=started,
        duration_ms=int((datetime.now(UTC) - started).total_seconds() * 1000),
        fetched=rep.fetched,
        inserted=rep.inserted,
        duplicates=rep.duplicates,
        rejected=rep.rejected,
        error=rep.error,
    )
    try:
        session.add(row)
        await session.commit()
    except Exception as e:
        logger.error("provider_run_record_failed", provider=rep.provider, error=str(e)[:200])


async def _guard(
    session: AsyncSession,
    name: str,
    coro_fn: Callable[[], Awaitable[tuple[int, int] | tuple[int, int, int]]],
) -> IngestReport:
    """Run one provider+store step; convert any failure into a report row,
    and persist the run for per-source health tracking."""
    report = IngestReport(provider=name)
    started = datetime.now(UTC)
    try:
        result = await coro_fn()
        report.inserted = result[0]
        report.duplicates = result[1]
        report.rejected = result[2] if len(result) > 2 else 0
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        logger.error("ingest_failed", provider=name, error=report.error)
        await session.rollback()
    await _record_run(session, report, started)
    return report


async def ingest_prices(
    session: AsyncSession,
    provider: Any,
    pairs: list[str],
    since: str,
    until: str | None = None,
) -> IngestReport:
    name = getattr(provider, "name", type(provider).__name__)

    async def run() -> tuple[int, int, int]:
        bars = await provider.fetch_prices(pairs, since, until)
        report_holder.append(len(bars))
        return await store_prices(session, bars)

    report_holder: list[int] = []
    rep = await _guard(session, f"prices:{name}", run)
    rep.fetched = report_holder[0] if report_holder else 0
    return rep


async def ingest_documents(session: AsyncSession, providers: Sequence[Any]) -> list[IngestReport]:
    reports: list[IngestReport] = []
    for provider in providers:
        name = getattr(provider, "name", type(provider).__name__)
        holder: list[int] = []

        async def run(p: Any = provider, h: list[int] = holder) -> tuple[int, int]:
            docs = await p.fetch_documents()
            h.append(len(docs))
            return await store_documents(session, docs)

        rep = await _guard(session, f"docs:{name}", run)
        rep.fetched = holder[0] if holder else 0
        reports.append(rep)
    return reports


async def ingest_macro(session: AsyncSession, provider: Any, currencies: list[str]) -> IngestReport:
    if provider is None:
        return IngestReport(provider="macro:none", error="skipped: no provider configured")
    name = getattr(provider, "name", type(provider).__name__)
    holder: list[int] = []

    async def run() -> tuple[int, int]:
        obs = await provider.fetch_macro(currencies)
        holder.append(len(obs))
        return await store_macro(session, obs)

    rep = await _guard(session, f"macro:{name}", run)
    rep.fetched = holder[0] if holder else 0
    return rep


async def ingest_rates(session: AsyncSession, provider: Any, banks: list[str]) -> IngestReport:
    name = getattr(provider, "name", type(provider).__name__)
    holder: list[int] = []

    async def run() -> tuple[int, int]:
        exps = await provider.fetch_expectations(banks)
        holder.append(len(exps))
        return await store_rate_expectations(session, exps)

    rep = await _guard(session, f"rates:{name}", run)
    rep.fetched = holder[0] if holder else 0
    return rep


async def run_all(
    session: AsyncSession,
    settings: Settings | None = None,
    client: FetchClient | None = None,
    since: str | None = None,
) -> list[IngestReport]:
    """Full sweep: prices + docs + macro + rates. Returns one report per provider."""
    settings = settings or get_settings()
    providers = build_providers(settings, client)
    since = since or date.today().replace(day=1).isoformat()  # month start

    reports: list[IngestReport] = []
    reports.append(
        await ingest_prices(
            session, providers["prices"], default_pairs(settings.currency_list), since
        )
    )
    reports.extend(await ingest_documents(session, providers["documents"]))
    macro_providers = providers["macro"]
    if not macro_providers:
        reports.append(await ingest_macro(session, None, settings.currency_list))
    else:
        for mp in macro_providers:
            reports.append(await ingest_macro(session, mp, settings.currency_list))
    reports.append(
        await ingest_rates(session, providers["rates"], settings.bank_list)
    )
    if client:
        await client.aclose()
    return reports
