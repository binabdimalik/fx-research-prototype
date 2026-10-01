"""Phase 2 ingestion tests: parsing fixtures, dedup, retries, error isolation.

No live network — httpx.MockTransport serves fixture payloads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.ingest.base import FetchClient, ProviderError
from fxterminal.ingest.pipeline import (
    default_pairs,
    ingest_documents,
    ingest_macro,
    ingest_prices,
    ingest_rates,
)
from fxterminal.ingest.providers.frankfurter import FrankfurterProvider, cross_rate
from fxterminal.ingest.providers.fred import FredMacroProvider
from fxterminal.ingest.providers.rates_file import JsonRateExpectationProvider
from fxterminal.ingest.providers.rss import FeedSpec, RssDocumentProvider, classify_cb_doc
from fxterminal.ingest.types import RawDoc
from fxterminal.models import FxPrice, MacroEvent, RateExpectation, RawDocument

FRANKFURTER_JSON = json.dumps(
    {
        "amount": 1.0,
        "base": "EUR",
        "start_date": "2026-09-28",
        "end_date": "2026-09-29",
        "rates": {
            "2026-09-28": {"USD": 1.17, "JPY": 173.4, "GBP": 0.87},
            "2026-09-29": {"USD": 1.175, "JPY": 174.0, "GBP": 0.872},
        },
    }
)

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Fed Press</title>
<item>
  <title>Federal Reserve issues FOMC statement</title>
  <link>https://www.federalreserve.gov/x.htm</link>
  <pubDate>Wed, 16 Sep 2026 18:00:00 GMT</pubDate>
  <description>&lt;p&gt;The Committee decided to maintain the target range.&lt;/p&gt;</description>
  <guid>x1</guid>
</item>
<item>
  <title>Minutes of the Federal Open Market Committee</title>
  <link>https://www.federalreserve.gov/y.htm</link>
  <pubDate>Wed, 07 Oct 2026 18:00:00 GMT</pubDate>
  <description>Minutes body</description>
  <guid>x2</guid>
</item>
</channel></rss>"""

FRED_JSON = json.dumps(
    {
        "observations": [
            {"date": "2026-08-01", "value": "330.12"},
            {"date": "2026-09-01", "value": "."},  # FRED missing-value marker
            {"date": "2026-09-02", "value": "331.5"},
        ]
    }
)


def _client(handler) -> FetchClient:
    return FetchClient(transport=httpx.MockTransport(handler), max_retries=2, base_backoff=0.0)


def _handler_ok(body: str):
    def h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    return h


# --- unit: cross-rate math (deterministic) ---


def test_cross_rate() -> None:
    rates = {"USD": 1.2, "JPY": 150.0, "GBP": 0.8}
    assert cross_rate("EURUSD", rates) == pytest.approx(1.2)
    assert cross_rate("USDJPY", rates) == pytest.approx(150.0 / 1.2)
    assert cross_rate("EURGBP", rates) == pytest.approx(0.8)
    assert cross_rate("GBPJPY", rates) == pytest.approx(150.0 / 0.8)
    with pytest.raises(ProviderError):
        cross_rate("EURXXX", rates)
    with pytest.raises(ValueError):
        cross_rate("EURUSDX", rates)


def test_default_pairs() -> None:
    assert default_pairs(["USD", "EUR", "JPY", "GBP"]) == [
        "EURGBP",
        "EURUSD",
        "EURJPY",
        "GBPUSD",
        "GBPJPY",
        "USDJPY",
    ]


def test_classify_cb_doc() -> None:
    assert classify_cb_doc("Federal Reserve issues FOMC statement") == "statement"
    assert classify_cb_doc("Minutes of the Federal Open Market Committee") == "minutes"
    assert classify_cb_doc("Speech by Governor X") == "speech"
    assert classify_cb_doc("Speech: remarks by the Chair") == "speech"
    assert classify_cb_doc("Federal Reserve announces fee schedule") == "news"


# --- provider parsing with mocked HTTP ---


async def test_frankfurter_parses_range() -> None:
    provider = FrankfurterProvider(_client(_handler_ok(FRANKFURTER_JSON)))
    bars = await provider.fetch_prices(["EURUSD", "USDJPY"], "2026-09-28", "2026-09-29")
    assert len(bars) == 4
    eurusd = [b for b in bars if b.pair == "EURUSD"]
    assert eurusd[0].close == pytest.approx(1.17)
    assert eurusd[1].close == pytest.approx(1.175)
    usdjpy = [b for b in bars if b.pair == "USDJPY"]
    assert usdjpy[0].close == pytest.approx(173.4 / 1.17)
    assert all(b.time.tzinfo == UTC and b.time.hour == 16 for b in bars)


async def test_rss_parses_and_classifies() -> None:
    spec = FeedSpec(
        url="https://fed.example/rss", source="fed_rss", central_bank="FED", currency="USD"
    )
    provider = RssDocumentProvider(_client(_handler_ok(RSS_XML)), spec)
    docs = await provider.fetch_documents()
    assert len(docs) == 2
    assert docs[0].doc_type == "statement"
    assert docs[1].doc_type == "minutes"
    assert docs[0].central_bank == "FED"
    assert "target range" in docs[0].raw_text  # html stripped
    assert docs[0].published_at == datetime(2026, 9, 16, 18, 0, tzinfo=UTC)


async def test_fred_parses_and_skips_missing() -> None:
    provider = FredMacroProvider(
        _client(_handler_ok(FRED_JSON)),
        api_key="test-key",
        series_map={"USD": [("CPIAUCSL", "US CPI", "high")]},
    )
    obs = await provider.fetch_macro(["USD"])
    assert len(obs) == 2  # "." row skipped
    assert obs[0].actual == pytest.approx(330.12)
    assert obs[0].name == "US CPI"


async def test_fred_requires_key() -> None:
    with pytest.raises(ProviderError):
        FredMacroProvider(_client(_handler_ok("{}")), api_key="")


async def test_rates_file_provider(tmp_path) -> None:
    f = tmp_path / "rates.json"
    f.write_text(
        json.dumps(
            [
                {
                    "time": "2026-09-30T14:00:00Z",
                    "central_bank": "fed",
                    "currency": "usd",
                    "implied_rate": 4.25,
                    "meeting_date": "2026-11-05",
                }
            ]
        )
    )
    provider = JsonRateExpectationProvider(str(f))
    exps = await provider.fetch_expectations(["FED"])
    assert len(exps) == 1
    assert exps[0].central_bank == "FED" and exps[0].implied_rate == 4.25
    # bank filter
    assert await provider.fetch_expectations(["ECB"]) == []
    # missing file -> empty, not error
    missing = JsonRateExpectationProvider(str(tmp_path / "nope.json"))
    assert await missing.fetch_expectations([]) == []


# --- http client behavior ---


async def test_client_retries_then_succeeds() -> None:
    calls = []

    def h(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500 if len(calls) == 1 else 200, text="ok")

    client = _client(h)
    assert await client.get_text("http://x") == "ok"
    assert len(calls) == 2


async def test_client_429_retry_after() -> None:
    calls = []

    def h(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        status = 429 if len(calls) == 1 else 200
        return httpx.Response(status, text="ok", headers={"retry-after": "0"})

    assert await _client(h).get_text("http://x") == "ok"
    assert len(calls) == 2


async def test_client_404_no_retry() -> None:
    calls = []

    def h(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, text="nope")

    with pytest.raises(ProviderError):
        await _client(h).get_text("http://x")
    assert len(calls) == 1  # 4xx != 429 is not retried


# --- store + pipeline (SQLite session fixture) ---


async def test_document_dedup_idempotent(session: AsyncSession) -> None:
    spec = FeedSpec(url="http://x/rss", source="fed_rss", central_bank="FED", currency="USD")
    provider = RssDocumentProvider(_client(_handler_ok(RSS_XML)), spec)

    r1 = (await ingest_documents(session, [provider]))[0]
    assert r1.inserted == 2 and r1.fetched == 2

    r2 = (await ingest_documents(session, [provider]))[0]  # same feed again
    assert r2.inserted == 0 and r2.duplicates == 2

    count = (await session.execute(select(func.count(RawDocument.id)))).scalar_one()
    assert count == 2


async def test_price_dedup_idempotent(session: AsyncSession) -> None:
    provider = FrankfurterProvider(_client(_handler_ok(FRANKFURTER_JSON)))
    r1 = await ingest_prices(session, provider, ["EURUSD"], "2026-09-28")
    assert r1.inserted == 2
    r2 = await ingest_prices(session, provider, ["EURUSD"], "2026-09-28")
    assert r2.inserted == 0
    assert (await session.execute(select(func.count(FxPrice.id)))).scalar_one() == 2


async def test_macro_and_rates_store(session: AsyncSession, tmp_path) -> None:
    fred = FredMacroProvider(
        _client(_handler_ok(FRED_JSON)),
        api_key="k",
        series_map={"USD": [("CPIAUCSL", "US CPI", "high")]},
    )
    rep = await ingest_macro(session, fred, ["USD"])
    assert rep.inserted == 2

    f = tmp_path / "rates_test.json"
    f.write_text(json.dumps([]))
    rep2 = await ingest_rates(session, JsonRateExpectationProvider(str(f)), [])
    assert rep2.inserted == 0 and rep2.ok


async def test_provider_failure_isolated(session: AsyncSession) -> None:
    """A failing provider must not stop others or raise."""

    class Broken:
        name = "broken"

        async def fetch_documents(self) -> list[RawDoc]:
            raise RuntimeError("boom")

    ok = RssDocumentProvider(
        _client(_handler_ok(RSS_XML)),
        FeedSpec(url="http://x/rss", source="fed_rss", central_bank="FED", currency="USD"),
    )
    reports = await ingest_documents(session, [Broken(), ok])
    assert reports[0].error and "boom" in reports[0].error
    assert reports[1].ok and reports[1].inserted == 2


async def test_macro_no_provider_is_skip_not_error(session: AsyncSession) -> None:
    rep = await ingest_macro(session, None, ["USD"])
    assert rep.error is not None and "skipped" in rep.error
    count = (await session.execute(select(func.count(MacroEvent.id)))).scalar_one()
    assert count == 0


async def test_rate_exp_roundtrip(session: AsyncSession) -> None:
    session.add(
        RateExpectation(
            time=datetime(2026, 9, 30, tzinfo=UTC),
            central_bank="FED",
            currency="USD",
            instrument="MANUAL",
            implied_rate=4.25,
            source="test",
        )
    )
    await session.commit()
    row = (await session.execute(select(RateExpectation))).scalar_one()
    assert row.implied_rate == 4.25


# --- Stage 2: price validation ---


async def test_price_validation_rejects(session: AsyncSession) -> None:
    """Bad bars are rejected + counted; good ones still insert."""
    from fxterminal.ingest.store import store_prices
    from fxterminal.ingest.types import PriceBar

    t0 = datetime(2026, 9, 30, 16, tzinfo=UTC)
    t1 = datetime(2026, 10, 1, 16, tzinfo=UTC)

    def bar(px, t, o=None, hi=None, lo=None):
        return PriceBar(
            time=t, pair="EURUSD", timeframe="1d",
            open=o if o is not None else px, high=hi if hi is not None else px,
            low=lo if lo is not None else px, close=px, source="test",
        )

    ins, dupes, rej = await store_prices(session, [
        bar(1.17, t0),            # ok — becomes anchor
        bar(-1.18, t1),           # non-positive
        bar(1.50, t1, o=1.50, hi=1.51, lo=0.90),  # jump +49% vs prev? no —
        # 1.50/1.17 = +28% jump -> rejected on jump
    ])
    assert ins == 1 and rej == 2 and dupes == 0

    # geometry: low > high
    ins2, _, rej2 = await store_prices(
        session, [bar(1.18, t1, o=1.18, hi=1.18, lo=1.30)]
    )
    assert ins2 == 0 and rej2 == 1

    # a sane 0.9% move passes
    t2 = datetime(2026, 10, 2, 16, tzinfo=UTC)
    ins3, _, rej3 = await store_prices(session, [bar(1.181, t2)])
    assert ins3 == 1 and rej3 == 0


async def test_circuit_breaker_opens() -> None:
    """After threshold consecutive failures the client fails fast, no request."""
    calls: list[int] = []

    def always_500(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, text="dead")

    client = FetchClient(
        transport=httpx.MockTransport(always_500),
        max_retries=1, base_backoff=0.0,
        breaker_threshold=2, breaker_cooldown_s=60.0,
    )
    for _ in range(2):
        with pytest.raises(ProviderError):
            await client.get_text("http://dead.example/x")
    assert len(calls) == 2
    with pytest.raises(ProviderError, match="circuit open"):
        await client.get_text("http://dead.example/y")
    assert len(calls) == 2  # no request made — breaker short-circuited
    # a different host is unaffected
    assert client._failures.get("other.example") is None
    await client.aclose()


async def test_provider_run_recorded(session: AsyncSession) -> None:
    """Every ingest step — success or failure — leaves a provider_runs row."""
    from fxterminal.models import ProviderRun

    class Broken:
        name = "broken"

        async def fetch_documents(self) -> list[RawDoc]:
            raise RuntimeError("boom")

    ok = RssDocumentProvider(
        _client(_handler_ok(RSS_XML)),
        FeedSpec(url="http://x/rss", source="fed_rss", central_bank="FED", currency="USD"),
    )
    await ingest_documents(session, [Broken(), ok])
    rows = list((await session.execute(select(ProviderRun))).scalars().all())
    assert len(rows) == 2
    by_name = {r.provider: r for r in rows}
    assert "boom" in (by_name["docs:broken"].error or "")
    assert by_name["docs:fed_rss"].inserted == 2
    assert by_name["docs:fed_rss"].error is None
