"""Stage 2: freshness policy + market-hours tests. Pure functions, no DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fxterminal.core.freshness import age_hours, fx_market_open, is_stale


def test_market_open_weekday() -> None:
    assert fx_market_open(datetime(2026, 9, 30, 12, tzinfo=UTC)) is True  # Wed


def test_market_closed_weekend() -> None:
    assert fx_market_open(datetime(2026, 10, 3, 12, tzinfo=UTC)) is False  # Sat
    assert fx_market_open(datetime(2026, 10, 2, 22, tzinfo=UTC)) is False  # Fri 22:00
    assert fx_market_open(datetime(2026, 10, 4, 20, tzinfo=UTC)) is False  # Sun 20:00
    assert fx_market_open(datetime(2026, 10, 4, 22, tzinfo=UTC)) is True   # Sun 22:00


def test_is_stale_thresholds() -> None:
    now = datetime(2026, 10, 1, 12, tzinfo=UTC)
    fresh = now - timedelta(hours=10)
    old = now - timedelta(days=10)
    assert is_stale("prices_1d", fresh, now) is False
    assert is_stale("prices_1d", old, now) is True
    assert is_stale("prices_1d", None, now) is True  # no data = stale
    # intraday exempt while market closed (weekend)
    weekend = datetime(2026, 10, 3, 12, tzinfo=UTC)  # Saturday
    assert is_stale("prices_1h", weekend - timedelta(hours=50), weekend) is False
    # but flags during the week
    wk = datetime(2026, 9, 30, 12, tzinfo=UTC)  # Wednesday
    assert is_stale("prices_1h", wk - timedelta(hours=50), wk) is True


def test_age_hours_naive_treated_utc() -> None:
    now = datetime(2026, 10, 1, 12, tzinfo=UTC)
    assert age_hours(datetime(2026, 10, 1, 10), now) == 2.0  # naive -> UTC
    assert age_hours(None, now) is None
