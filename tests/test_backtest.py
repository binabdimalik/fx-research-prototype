"""Phase 6 tests: metrics, replay engine, costs, lookahead safety."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.backtest.engine import SimParams, pip_size, run_backtest
from fxterminal.backtest.metrics import compute_metrics, equity_curve, max_drawdown
from fxterminal.models import FxPrice, RateExpectation, RawDocument, StanceScore

START = datetime(2026, 6, 1, tzinfo=UTC)


def test_metrics_math() -> None:
    m = compute_metrics([0.02, -0.01, 0.03, -0.01])
    assert m.n_trades == 4
    assert m.hit_rate == pytest.approx(0.5)
    assert m.expectancy == pytest.approx(0.0075)
    assert m.total_return == pytest.approx((1.02 * 0.99 * 1.03 * 0.99) - 1)
    assert m.profit_factor == pytest.approx(0.05 / 0.02)
    assert m.sharpe is not None
    assert compute_metrics([]).n_trades == 0
    # all winners -> profit factor None (no losses)
    assert compute_metrics([0.01, 0.02]).profit_factor is None


def test_max_drawdown() -> None:
    curve = equity_curve([0.1, -0.2, 0.05])  # 1.1 -> 0.88 -> 0.924
    assert max_drawdown(curve) == pytest.approx(1 - 0.88 / 1.1)


def test_pip_size() -> None:
    assert pip_size("EURUSD") == pytest.approx(0.0001)
    assert pip_size("USDJPY") == pytest.approx(0.01)


# --- engine on synthetic bars ---


async def _seed(session: AsyncSession, pair: str, closes: list[float]) -> None:
    bars = [
        FxPrice(
            time=START + timedelta(days=i),
            pair=pair,
            timeframe="1d",
            open=c,
            high=c + 0.002,
            low=c - 0.002,
            close=c,
            source="test",
        )
        for i, c in enumerate(closes)
    ]
    session.add_all(bars)
    await session.commit()


async def _stance_doc(session: AsyncSession, ccy: str, score: int, when: datetime) -> None:
    doc = RawDocument(
        source="t",
        doc_type="statement",
        central_bank="FED",
        currency=ccy,
        url="https://x",
        title="t",
        published_at=when,
        content_hash=uuid.uuid4().hex,
        raw_text="text",
    )
    session.add(doc)
    await session.flush()
    session.add(
        StanceScore(
            central_bank="FED",
            currency=ccy,
            statement_doc_id=doc.id,
            score=score,
            label="hawkish" if score > 0 else "dovish",
            evidence=[],
            model="t",
            prompt_version="v1",
        )
    )


async def test_backtest_enters_and_exits(session: AsyncSession) -> None:
    # mild uptrend with daily variation so sigma > 0
    closes = [1.10 + 0.001 * i + (0.0008 if i % 3 == 0 else -0.0004) for i in range(60)]
    await _seed(session, "EURUSD", closes)
    await _stance_doc(session, "EUR", 70, START - timedelta(days=10))  # hawkish EUR
    await _stance_doc(session, "USD", -60, START - timedelta(days=10))  # dovish USD
    await session.commit()

    params = SimParams(
        pair="EURUSD",
        start=START,
        end=START + timedelta(days=60),
        spread_pips=1.0,
        slippage_pips=0.5,
        oos_fraction=0.0,
    )
    res = await run_backtest(session, params)
    assert res.n_bars == 60
    assert res.trades  # the +130 differential must trigger
    t = res.trades[0]
    assert t.direction == 1
    # costs charged: net < gross
    assert t.net_return == pytest.approx(t.gross_return - t.cost_paid + t.carry_earned)
    assert t.cost_paid > 0


async def test_backtest_no_signal_when_weak(session: AsyncSession) -> None:
    closes = [1.10 + 0.0005 * i for i in range(60)]
    await _seed(session, "EURUSD", closes)
    await _stance_doc(session, "EUR", 8, START - timedelta(days=10))
    await _stance_doc(session, "USD", -5, START - timedelta(days=10))
    await session.commit()
    res = await run_backtest(
        session,
        SimParams(pair="EURUSD", start=START, end=START + timedelta(days=60)),
    )
    assert res.trades == []  # |13| < 20


async def test_no_lookahead(session: AsyncSession) -> None:
    """A stance published AFTER bar t must not affect the signal at t."""
    closes = [1.10 + 0.001 * i for i in range(60)]
    await _seed(session, "EURUSD", closes)
    # EUR statement published halfway through the window
    late = START + timedelta(days=40)
    await _stance_doc(session, "EUR", 70, late)
    await _stance_doc(session, "USD", -60, late)
    await session.commit()
    res = await run_backtest(
        session,
        SimParams(pair="EURUSD", start=START, end=START + timedelta(days=60)),
    )
    # every trade must enter strictly after the stance became public
    # (sqlite stores naive datetimes — treat them as UTC)
    for t in res.trades:
        entry = t.entry_time.replace(tzinfo=UTC) if t.entry_time.tzinfo is None else t.entry_time
        assert entry >= late


async def test_carry_from_rate_expectations(session: AsyncSession) -> None:
    closes = [1.10 + 0.001 * i + (0.001 if i % 4 == 0 else -0.0005) for i in range(80)]
    await _seed(session, "EURUSD", closes)
    await _stance_doc(session, "EUR", 70, START - timedelta(days=5))
    await _stance_doc(session, "USD", -60, START - timedelta(days=5))
    # EUR yields much more -> long EURUSD earns positive carry
    session.add_all(
        [
            RateExpectation(
                time=START - timedelta(days=1),
                central_bank="ECB",
                currency="EUR",
                instrument="M",
                implied_rate=5.0,
                source="t",
            ),
            RateExpectation(
                time=START - timedelta(days=1),
                central_bank="FED",
                currency="USD",
                instrument="M",
                implied_rate=3.0,
                source="t",
            ),
        ]
    )
    await session.commit()
    res = await run_backtest(
        session, SimParams(pair="EURUSD", start=START, end=START + timedelta(days=80))
    )
    longs = [t for t in res.trades if t.direction > 0]
    if longs:
        assert all(t.carry_earned > 0 for t in longs)


async def test_oos_split(session: AsyncSession) -> None:
    closes = [1.10 + 0.001 * i + (0.001 if i % 3 == 0 else -0.0006) for i in range(90)]
    await _seed(session, "EURUSD", closes)
    await _stance_doc(session, "EUR", 70, START - timedelta(days=5))
    await _stance_doc(session, "USD", -60, START - timedelta(days=5))
    await session.commit()
    res = await run_backtest(
        session,
        SimParams(pair="EURUSD", start=START, end=START + timedelta(days=90), oos_fraction=0.5),
    )
    if len(res.trades) >= 2:
        assert res.metrics is not None and res.metrics_oos is not None
        assert res.metrics.n_trades + res.metrics_oos.n_trades == len(res.trades)
