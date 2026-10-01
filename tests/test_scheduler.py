"""Phase 8 tests: cycle orchestration, alert detection/delivery, /api/status."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from httpx import ASGITransport, AsyncClient, MockTransport
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.alerts.notifier import (
    _deliver_webhook,
    check_alerts,
    find_new_ideas,
    find_stance_shifts,
)
from fxterminal.api.main import app
from fxterminal.core.config import Settings
from fxterminal.models import CurrencyScore, FxPrice, RawDocument, StanceScore, TradeIdea
from fxterminal.sched.jobs import CycleResult, run_cycle

NOW = datetime.now(UTC)


# --- alert detection ---


async def test_new_ideas_alerted(session: AsyncSession) -> None:
    session.add(
        TradeIdea(
            pair="EURUSD",
            direction="long",
            entry_low=1.09,
            entry_high=1.10,
            stop=1.08,
            target=1.12,
            risk_reward=2.0,
            rationale="r",
            invalidation="x",
            key_risks=[],
            filters={},
        )
    )
    await session.commit()
    alerts = await find_new_ideas(session, NOW - timedelta(hours=2))
    assert len(alerts) == 1 and alerts[0].kind == "new_idea"
    assert "EURUSD" in alerts[0].message
    # outside lookback -> nothing
    assert await find_new_ideas(session, NOW + timedelta(hours=1)) == []


async def test_stance_shift_threshold(session: AsyncSession) -> None:
    for score, shift in [(60, 45), (10, 5)]:
        doc = RawDocument(
            source="t",
            doc_type="statement",
            central_bank="FED",
            currency="USD",
            url=f"https://x/{score}",
            title="t",
            published_at=NOW - timedelta(minutes=30),
            content_hash=uuid.uuid4().hex,
            raw_text="x",
        )
        session.add(doc)
        await session.flush()
        session.add(
            StanceScore(
                central_bank="FED",
                currency="USD",
                statement_doc_id=doc.id,
                score=score,
                label="hawkish",
                shift_vs_prev=shift,
                evidence=[],
                model="m",
                prompt_version="v1",
            )
        )
    await session.commit()
    alerts = await find_stance_shifts(session, NOW - timedelta(hours=1))
    assert len(alerts) == 1  # only |45| >= SHIFT_THRESHOLD
    assert alerts[0].payload["shift"] == 45


async def test_check_alerts_no_rows(session: AsyncSession) -> None:
    assert await check_alerts(session, Settings(_env_file=None)) == "ok (no alerts)"


async def test_webhook_delivery_mocked() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.read().decode())
        return httpx.Response(200)

    transport = MockTransport(handler)
    s = Settings(_env_file=None, alert_webhook="https://hooks.example/t")

    # inject transport via a patched client
    import fxterminal.alerts.notifier as notifier

    orig = httpx.AsyncClient
    notifier.httpx.AsyncClient = lambda *a, **kw: orig(*a, **{**kw, "transport": transport})
    try:
        from fxterminal.alerts.notifier import Alert

        await _deliver_webhook(s, [Alert(kind="new_idea", message="test alert")])
    finally:
        notifier.httpx.AsyncClient = orig
    assert captured and "test alert" in captured[0]


async def test_webhook_failure_swallowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = MockTransport(handler)
    s = Settings(_env_file=None, alert_webhook="https://hooks.example/t")
    import fxterminal.alerts.notifier as notifier

    orig = httpx.AsyncClient
    notifier.httpx.AsyncClient = lambda *a, **kw: orig(*a, **{**kw, "transport": transport})
    try:
        from fxterminal.alerts.notifier import Alert

        await _deliver_webhook(s, [Alert(kind="x", message="m")])  # must not raise
    finally:
        notifier.httpx.AsyncClient = orig


# --- run_cycle orchestration ---


async def test_run_cycle_step_isolation(session: AsyncSession, monkeypatch) -> None:
    """A failing step must not abort the cycle — later steps still run."""
    import fxterminal.sched.jobs as jobs

    async def boom(session_, settings_):
        raise RuntimeError("ingest broken")

    monkeypatch.setattr(jobs, "_ingest_all", boom)
    # LLM skipped by default (no key in test env)
    settings = Settings(_env_file=None)
    res = await run_cycle(session, settings)
    assert "error" in res.steps["ingest"]
    assert res.steps["strength"].startswith("ok")  # later steps still ran
    assert res.steps["alerts"].startswith("ok")
    assert res.steps["stance"].startswith("skip")


async def test_cycle_result_ok_property() -> None:
    r = CycleResult(started_at=NOW, steps={"a": "ok", "b": "skip"})
    assert r.ok
    r.steps["c"] = "error: x"
    assert not r.ok


# --- /api/status ---


async def test_status_endpoint(session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Status reports freshness per feed with staleness flags."""
    from fxterminal.core.db import get_session

    async def override():
        yield session

    app.dependency_overrides[get_session] = override
    try:
        session.add_all(
            [
                FxPrice(
                    time=NOW - timedelta(hours=5),
                    pair="EURUSD",
                    timeframe="1d",
                    open=1.1,
                    high=1.1,
                    low=1.1,
                    close=1.1,
                    source="t",
                ),
                CurrencyScore(
                    currency="USD",
                    as_of=NOW - timedelta(hours=3),
                    score=10.0,
                    components={},
                    scorer_version="v1",
                ),
            ]
        )
        await session.commit()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/api/status")
        body = r.json()
        assert body["feeds"]["prices"]["rows"] == 1
        assert body["feeds"]["prices_1d"]["age_hours"] < 48
        assert body["feeds"]["prices_1d"]["stale"] is False
        # documents/strength have no rows -> flagged stale
        assert "documents" in body["stale_feeds"]
        assert "prices_1d" not in body["stale_feeds"]
        assert "market_open" in body and "sources" in body
    finally:
        app.dependency_overrides.clear()
