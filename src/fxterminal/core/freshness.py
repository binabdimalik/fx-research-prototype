"""Data-freshness rules and FX market-hours.

Two concerns live here so every layer (API status, dashboard, idea generation,
chat) reads ONE policy:

- `MAX_AGE_HOURS` — per feed kind, the age after which stored data is STALE.
  Values tolerate legitimate gaps: ECB publishes daily ~16:00 CET weekdays, so
  the 1d price bar over a weekend is ~72h old but expected; intraday bars are
  only flagged stale while the FX market is actually open.
- `fx_market_open` — FX trades Sun ~21:00 UTC through Fri ~21:00 UTC (the
  17:00 New York close/open, approximated; DST shifts not modeled — the
  approximation is honest for a freshness label, never used for execution).

Nothing here invents data; it only labels how old the data we have is.
"""

from __future__ import annotations

from datetime import UTC, datetime

# Feed kind -> max acceptable age in hours.
MAX_AGE_HOURS: dict[str, float] = {
    "prices_1d": 72.0,   # ECB daily fix; weekend gap is expected, not stale
    "prices_1h": 4.0,    # intraday bars — evaluated only while market is open
    "documents": 72.0,   # news feeds flow daily; CB statements are sporadic and
    #                    are judged on their own published_at, not this feed
    "macro": 45 * 24.0,  # monthly releases — stale after ~1.5 release cycles
    "rate_expectations": 48.0,  # manual/market quotes — daily-cadence source
    "strength": 30.0,    # recomputed each cycle; >~1 cycle old = stale
    "ideas": 8 * 24.0,   # ideas carry a 7d valid_until; beyond that they're dead
}

FX_OPEN_UTC = ("sun", 21)  # market opens Sunday ~21:00 UTC
FX_CLOSE_UTC = ("fri", 21)  # closes Friday ~21:00 UTC


def fx_market_open(now: datetime | None = None) -> bool:
    """Approximate FX trading week: Sun 21:00 UTC -> Fri 21:00 UTC."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    wd, h = now.weekday(), now.hour  # Mon=0 .. Sun=6
    if wd == 6:  # Sunday
        return h >= FX_OPEN_UTC[1]
    if wd == 5:  # Saturday — always closed
        return False
    if wd == 4:  # Friday
        return h < FX_CLOSE_UTC[1]
    return True  # Mon-Thu always open


def age_hours(latest: datetime | None, now: datetime | None = None) -> float | None:
    """Hours since `latest` (naive datetimes treated as UTC). None -> no data."""
    if latest is None:
        return None
    now = now or datetime.now(UTC)
    ts = latest.replace(tzinfo=UTC) if latest.tzinfo is None else latest
    return round((now - ts).total_seconds() / 3600, 1)


def is_stale(kind: str, latest: datetime | None, now: datetime | None = None) -> bool:
    """True when the newest stored row for `kind` exceeds its max age.

    Intraday prices get a market-aware pass: no data during the weekend is
    expected, so 'prices_1h' only flags stale while `fx_market_open`.
    """
    now = now or datetime.now(UTC)
    age = age_hours(latest, now)
    if age is None:
        return True
    limit = MAX_AGE_HOURS.get(kind)
    if limit is None:
        return False  # unknown kind — don't flag what we have no policy for
    if kind == "prices_1h" and not fx_market_open(now):
        return False
    return age > limit
