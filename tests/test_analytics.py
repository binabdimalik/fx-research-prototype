"""Phase 5 tests: strength composite, pair ranking, trade-idea filters/levels."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.analytics.ideas import (
    MIN_CONFIDENCE,
    STOP_SIGMA,
    TARGET_RR,
    generate_ideas,
    levels_for,
    rr,
)
from fxterminal.analytics.pairs import rank_pairs
from fxterminal.analytics.strength import (
    StrengthResult,
    combine_components,
    compute_strength,
    realized_vol,
)
from fxterminal.core.config import Settings
from fxterminal.models import (
    CurrencyScore,
    FxPrice,
    MacroEvent,
    RateExpectation,
    RawDocument,
    StanceScore,
    Theme,
)

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _strength(ccy: str, score: float, conf: float = 1.0) -> StrengthResult:
    return StrengthResult(currency=ccy, score=score, components={}, confidence=conf)


# --- combine_components (pure) ---


def test_combine_renormalizes_missing() -> None:
    r = combine_components({"stance": 1.0, "themes": None, "rates": None})
    assert r is not None
    assert r.score == 100.0  # only stance present -> its value
    assert r.confidence == pytest.approx(0.5)
    assert combine_components({"stance": None, "themes": None, "rates": None}) is None


def test_combine_weighted() -> None:
    # stance .5 @1.0, themes .25 @0.0, rates .25 @-1.0 -> (0.5+0-0.25)/1.0
    r = combine_components({"stance": 1.0, "themes": 0.0, "rates": -1.0})
    assert r.score == pytest.approx(25.0)


# --- realized_vol ---


def test_realized_vol() -> None:
    flat = [1.1] * 40
    assert realized_vol(flat) is None  # constant series
    drifting = [1.0 * (1 + 0.001 * i) for i in range(40)]
    v = realized_vol(drifting)
    assert v is not None and v > 0
    assert realized_vol([1.1] * 5) is None  # insufficient data


# --- rank_pairs ---


def test_rank_pairs_skips_missing_and_sorts() -> None:
    strengths = {
        "USD": _strength("USD", 50),
        "EUR": _strength("EUR", -30),
        "JPY": _strength("JPY", 10),
    }
    views = rank_pairs(strengths, ["EURUSD", "USDJPY", "EURJPY", "GBPJPY"])
    pairs = [v.pair for v in views]
    assert "GBPJPY" not in pairs  # GBP has no strength -> skipped
    assert pairs[0] == "EURUSD"  # |50-(-30)|=80 biggest
    eurusd = views[0]
    assert eurusd.direction == -1  # long USD = short EURUSD


# --- levels + rr (pure risk math) ---


def test_levels_and_rr() -> None:
    lo, hi, stop, target = levels_for(1, 1.1000, 0.0100)
    assert lo < 1.1000 < hi  # entry zone straddles the reference price
    assert stop == pytest.approx(1.1000 - STOP_SIGMA * 0.0100)
    assert target == pytest.approx(1.1000 + STOP_SIGMA * TARGET_RR * 0.0100)
    assert rr(1.1000, stop, target) == pytest.approx(TARGET_RR)
    _l, _h, s2, t2 = levels_for(-1, 1.1000, 0.0100)
    assert s2 > 1.1000 and t2 < 1.1000  # short: stop above, target below


# --- generate_ideas end-to-end on fixtures ---


def _settings() -> Settings:
    return Settings(_env_file=None)


async def _seed_pair(session: AsyncSession, pair: str, days: int = 40) -> None:
    bars = [
        FxPrice(
            time=NOW - timedelta(days=days - i),
            pair=pair,
            timeframe="1d",
            open=1.10 + 0.001 * i,
            high=1.10 + 0.001 * i,
            low=1.10 + 0.001 * i,
            close=1.10 + 0.001 * i,
            source="test",
        )
        for i in range(days)
    ]
    session.add_all(bars)
    await session.commit()


async def test_idea_generated_when_filters_pass(session: AsyncSession) -> None:
    await _seed_pair(session, "EURUSD")
    strengths = {"EUR": _strength("EUR", 60), "USD": _strength("USD", -40)}
    views = rank_pairs(strengths, ["EURUSD"])
    ideas = await generate_ideas(session, views, strengths, _settings(), now=NOW)
    assert len(ideas) == 1
    i = ideas[0]
    assert i.direction == "long" and i.pair == "EURUSD"
    assert i.stop < i.entry_low < i.entry_high < i.target
    assert i.risk_reward == pytest.approx(TARGET_RR)
    assert i.filters["event_blackout_clear"]
    assert i.invalidation and i.key_risks and i.disclaimer


async def test_idea_rejected_low_differential(session: AsyncSession) -> None:
    await _seed_pair(session, "EURUSD")
    strengths = {"EUR": _strength("EUR", 10), "USD": _strength("USD", -5)}
    views = rank_pairs(strengths, ["EURUSD"])
    ideas = await generate_ideas(session, views, strengths, _settings(), now=NOW)
    assert ideas == []  # |15| < MIN_DIFFERENTIAL=20


async def test_idea_rejected_event_blackout(session: AsyncSession) -> None:
    await _seed_pair(session, "EURUSD")
    session.add(
        MacroEvent(
            time=NOW + timedelta(hours=1),  # inside 4h blackout
            currency="USD",
            name="US CPI",
            impact="high",
            source="test",
        )
    )
    await session.commit()
    strengths = {"EUR": _strength("EUR", 60), "USD": _strength("USD", -40)}
    views = rank_pairs(strengths, ["EURUSD"])
    ideas = await generate_ideas(session, views, strengths, _settings(), now=NOW)
    assert ideas == []


async def test_idea_rejected_low_confidence(session: AsyncSession) -> None:
    await _seed_pair(session, "EURUSD")
    strengths = {
        "EUR": _strength("EUR", 60, conf=0.2),  # below MIN_CONFIDENCE
        "USD": _strength("USD", -40, conf=0.2),
    }
    views = rank_pairs(strengths, ["EURUSD"])
    assert views[0].confidence < MIN_CONFIDENCE
    ideas = await generate_ideas(session, views, strengths, _settings(), now=NOW)
    assert ideas == []


async def test_idea_no_prices_no_crash(session: AsyncSession) -> None:
    strengths = {"EUR": _strength("EUR", 60), "USD": _strength("USD", -40)}
    views = rank_pairs(strengths, ["EURUSD"])
    assert await generate_ideas(session, views, strengths, _settings()) == []


# --- compute_strength integration on sqlite ---


async def test_compute_strength_components(session: AsyncSession) -> None:
    doc = RawDocument(
        source="t",
        doc_type="statement",
        central_bank="FED",
        currency="USD",
        url="https://x/1",
        title="t",
        published_at=NOW - timedelta(days=2),
        content_hash=uuid.uuid4().hex,
        raw_text="statement text",
    )
    session.add(doc)
    await session.flush()
    session.add(
        StanceScore(
            central_bank="FED",
            currency="USD",
            statement_doc_id=doc.id,
            score=60,
            label="hawkish",
            evidence=[],
            model="t",
            prompt_version="v1",
        )
    )
    session.add_all(
        [
            Theme(
                currency="USD",
                theme="inflation",
                sentiment=0.8,
                weight=1.0,
                window_start=NOW - timedelta(days=7),
                window_end=NOW,
                evidence=[],
                model="t",
                prompt_version="v1",
            ),
        ]
    )
    session.add_all(
        [
            RateExpectation(
                time=NOW - timedelta(days=29),
                central_bank="FED",
                currency="USD",
                instrument="M",
                implied_rate=4.0,
                source="t",
            ),
            RateExpectation(
                time=NOW - timedelta(days=1),
                central_bank="FED",
                currency="USD",
                instrument="M",
                implied_rate=4.5,
                source="t",
            ),
        ]
    )
    await session.commit()

    res = await compute_strength(session, "USD", as_of=NOW)
    assert res is not None
    # stance .6, themes .8, rates +0.5 clamped->0.5; composite:
    # (0.5*.6 + .25*.8 + .25*.5) = .3+.2+.125 = .625 -> 62.5
    assert res.score == pytest.approx(62.5)
    assert res.confidence == pytest.approx(1.0)

    # persists via compute_all
    from fxterminal.analytics.strength import compute_all

    rows = await compute_all(session, ["USD"], as_of=NOW)
    assert len(rows) == 1 and isinstance(rows[0], CurrencyScore)
    assert len(rows) == 1 and isinstance(rows[0], CurrencyScore)
