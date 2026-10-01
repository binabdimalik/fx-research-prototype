"""API smoke tests (SQLite, no external services)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

import fxterminal.core.db as db_module
from fxterminal.api.main import app
from fxterminal.core.config import get_settings
from fxterminal.core.db import init_db, make_engine
from fxterminal.models import (
    CurrencyScore,
    FxPrice,
    MacroEvent,
    RateExpectation,
    RawDocument,
    StanceScore,
    Theme,
    TradeIdea,
)

NOW = datetime(2026, 9, 30, tzinfo=UTC)


@pytest.fixture
def _sqlite_engine(monkeypatch: pytest.MonkeyPatch):
    """Point the app at in-memory SQLite so /readyz doesn't need Postgres."""
    monkeypatch.setenv("FXT_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    get_settings.cache_clear()
    monkeypatch.setattr(db_module, "_engine", None)
    engine = make_engine(get_settings())
    monkeypatch.setattr(db_module, "_engine", engine)


@pytest.fixture
async def seeded(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Temp-file sqlite (shared across the pool, unlike :memory:) with the
    full schema + a seed row per model. Returns the engine's session factory."""
    url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    monkeypatch.setenv("FXT_DATABASE_URL", url)
    get_settings.cache_clear()
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_session_factory", None)
    engine = make_engine(get_settings())
    monkeypatch.setattr(db_module, "_engine", engine)
    await init_db(engine)
    from fxterminal.core.db import get_session_factory

    factory = get_session_factory()
    async with factory() as s:
        doc = RawDocument(
            source="t",
            doc_type="statement",
            central_bank="FED",
            currency="USD",
            url="https://x/1",
            title="FOMC statement",
            published_at=NOW - timedelta(days=2),
            content_hash=uuid.uuid4().hex,
            raw_text="Rates raised.",
        )
        s.add(doc)
        await s.flush()
        s.add_all(
            [
                StanceScore(
                    central_bank="FED",
                    currency="USD",
                    statement_doc_id=doc.id,
                    score=55,
                    label="hawkish",
                    evidence=[
                        {
                            "doc_id": str(doc.id),
                            "published_at": NOW.isoformat(),
                            "sentence": "Rates raised.",
                            "topic": "guidance",
                            "direction": "hawkish",
                        }
                    ],
                    model="m",
                    prompt_version="v1",
                ),
                Theme(
                    currency="USD",
                    theme="inflation",
                    sentiment=0.7,
                    weight=0.9,
                    window_start=NOW - timedelta(days=7),
                    window_end=NOW,
                    evidence=[],
                    model="m",
                    prompt_version="v1",
                ),
                CurrencyScore(
                    currency="USD",
                    as_of=NOW,
                    score=55.0,
                    components={"stance": 0.55, "_confidence": 0.5},
                    scorer_version="v1",
                ),
                CurrencyScore(
                    currency="EUR",
                    as_of=NOW,
                    score=-20.0,
                    components={"stance": -0.2, "_confidence": 0.5},
                    scorer_version="v1",
                ),
                TradeIdea(
                    pair="EURUSD",
                    direction="short",
                    entry_low=1.09,
                    entry_high=1.10,
                    stop=1.11,
                    target=1.06,
                    risk_reward=2.0,
                    rationale="rationale",
                    invalidation="diff flips",
                    key_risks=["thin data"],
                    filters={"min_differential": True},
                ),
                FxPrice(
                    time=NOW - timedelta(days=1),
                    pair="EURUSD",
                    timeframe="1d",
                    open=1.1,
                    high=1.1,
                    low=1.1,
                    close=1.1,
                    source="t",
                ),
                RateExpectation(
                    time=NOW - timedelta(hours=6),
                    central_bank="FED",
                    currency="USD",
                    instrument="FED_FUNDS_FUT_Z6",
                    meeting_date=NOW + timedelta(days=30),
                    implied_rate=4.25,
                    implied_prob=0.30,
                    source="t",
                ),
                MacroEvent(
                    time=NOW - timedelta(days=20),
                    currency="USD",
                    name="US CPI (index, SA)",
                    impact="high",
                    actual=325.6,
                    forecast=325.1,
                    previous=324.9,
                    source="t",
                ),
                MacroEvent(
                    time=NOW - timedelta(days=25),
                    currency="USD",
                    name="US GDP",
                    impact="medium",
                    actual=2.1,
                    forecast=2.0,
                    previous=2.0,
                    source="t",
                ),
            ]
        )
        # theme-matched releases seeded above: USD theme=inflation → CPI row
        # surfaces in MACRO; GDP row is filtered out of the context block
        await s.commit()
    yield factory
    await engine.dispose()


async def test_healthz() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


async def test_readyz_sqlite(_sqlite_engine) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    assert "not financial advice" in r.json()["disclaimer"]


async def test_dashboard_serves() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/")
    assert r.status_code == 200 and "fxterminal" in r.text


async def test_read_endpoints(seeded) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        s = (await c.get("/api/strength")).json()
        assert len(s["strengths"]) == 2 and s["strengths"][0]["currency"] == "USD"
        st = (await c.get("/api/stances")).json()
        assert st["stances"][0]["score"] == 55
        assert st["stances"][0]["evidence"][0]["sentence"] == "Rates raised."
        t = (await c.get("/api/themes")).json()
        assert t["themes"][0]["theme"] == "inflation"
        i = (await c.get("/api/ideas")).json()
        assert i["ideas"][0]["pair"] == "EURUSD" and i["ideas"][0]["disclaimer"]
        d = (await c.get("/api/docs?doc_type=statement")).json()
        assert d["documents"][0]["title"] == "FOMC statement"
        p = (await c.get("/api/prices?pair=EURUSD&days=30")).json()
        assert p["bars"][0]["close"] == pytest.approx(1.1)
        # every payload carries the disclaimer
        assert "disclaimer" in s and "disclaimer" in i


async def test_chat_no_key_returns_snapshot(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """No API key: endpoint degrades to the raw context snapshot — no 500."""
    monkeypatch.delenv("FXT_ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/chat", json={"message": "what is USD doing?"})
    assert r.status_code == 200
    body = r.json()
    assert "USD" in body["answer"] or "STRENGTH" in body["answer"]
    assert body["disclaimer"]
    assert body["model"] in ("none", "unknown")


async def test_chat_with_mock_llm(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """A working LLM's answer is returned verbatim, grounded on real context."""
    import fxterminal.nlp.llm as llm_mod

    class FakeLLM:
        model = "fake-1"

        def __init__(self, settings):
            pass

        async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
            assert "STRENGTH" in prompt  # real DB data injected
            return schema.model_validate({"answer": "USD is strongest at +55."})

    # routes.py does `from fxterminal.nlp.llm import AnthropicLLM` lazily —
    # patch the name on the source module so the lazy import picks the fake
    monkeypatch.setattr(llm_mod, "AnthropicLLM", FakeLLM)
    monkeypatch.setenv("FXT_ANTHROPIC_API_KEY", "x")
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/chat", json={"message": "strongest currency?"})
    body = r.json()
    assert body["answer"] == "USD is strongest at +55."
    assert body["model"] == "fake-1"


async def test_chat_agent_tool_loop(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent loop: model emits a tool call, sees the result, then answers.
    The verifier must confirm stored figures and flag invented ones."""
    import fxterminal.nlp.llm as llm_mod

    calls = {"n": 0}

    class AgentLLM:
        model = "agent-fake"

        def __init__(self, settings):
            pass

        async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return schema.model_validate(
                    {"action": "tool", "tool": "prices", "args": {"pair": "EURUSD"}}
                )
            assert "TOOL prices RESULT" in prompt  # tool output fed back
            assert "EURUSD" in prompt
            return schema.model_validate(
                {
                    "action": "answer",
                    "answer": "USD is the strongest at +55.0. "
                    "Research output, not financial advice.",
                }
            )

    monkeypatch.setattr(llm_mod, "AnthropicLLM", AgentLLM)
    monkeypatch.setenv("FXT_ANTHROPIC_API_KEY", "x")
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/chat", json={"message": "strongest currency?"})
    body = r.json()
    assert body["mode"] == "agent"
    assert body["verified"] is True           # +55.0 matches stored USD score
    assert body["warnings"] == []
    assert body["steps"] == ['prices({"pair": "EURUSD"})']
    assert "USD is the strongest" in body["answer"]


async def test_chat_verifier_flags_bad_figures(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that invents numbers gets warnings + verified=False."""
    import fxterminal.nlp.llm as llm_mod

    class LyingLLM:
        model = "lying-fake"

        def __init__(self, settings):
            pass

        async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
            return schema.model_validate(
                {"action": "answer", "answer": "EUR is the strongest at +99.5."}
            )

    monkeypatch.setattr(llm_mod, "AnthropicLLM", LyingLLM)
    monkeypatch.setenv("FXT_ANTHROPIC_API_KEY", "x")
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/chat", json={"message": "strongest?"})
    body = r.json()
    assert body["mode"] == "agent"
    assert body["verified"] is False
    assert any("unverified figure" in w for w in body["warnings"])  # +99.5 invented
    assert any("not the strongest" in w for w in body["warnings"])  # EUR vs stored USD


async def test_chat_offline_macro_answer(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """No key → offline answerer serves theme-matched releases: USD's
    inflation theme surfaces the CPI print and filters out GDP."""
    monkeypatch.delenv("FXT_ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/chat", json={"message": "what is USD inflation data doing?"})
    body = r.json()
    assert body["mode"] == "offline" and body["verified"] is True
    assert "US CPI" in body["answer"]          # theme-matched release present
    assert "US GDP" not in body["answer"]      # off-theme release filtered
    assert "325.6" in body["answer"]           # actual printed
    assert "inflation" in body["answer"]       # dominant theme named


async def test_chat_offline_theme_brief(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-currency theme question → structured brief: theme, stance,
    market pricing and the releases to watch — not a bare label."""
    monkeypatch.delenv("FXT_ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/api/chat",
            json={"message": "what is the FED current theme and stance?"},
        )
    body = r.json()
    assert body["mode"] == "offline" and body["verified"] is True
    ans = body["answer"]
    assert "dominant theme: inflation" in ans        # theme named
    assert "FED/USD" in ans and "hawkish" in ans    # stance line
    assert "4.250%" in ans and "30% priced" in ans   # market pricing
    assert "US CPI" in ans                           # releases to watch


async def test_chat_agent_uses_macro_tool(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent calls the macro tool and its result reaches the transcript."""
    import fxterminal.nlp.llm as llm_mod

    calls = {"n": 0}

    class MacroLLM:
        model = "macro-fake"

        def __init__(self, settings):
            pass

        async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return schema.model_validate(
                    {"action": "tool", "tool": "macro", "args": {"currency": "USD"}}
                )
            assert "TOOL macro RESULT" in prompt
            assert "US CPI" in prompt          # filtered release came back
            assert "US GDP" not in prompt      # off-theme row excluded
            return schema.model_validate(
                {
                    "action": "answer",
                    "answer": "USD inflation is re-accelerating: CPI 325.6 vs "
                    "forecast 325.1. Research output, not financial advice.",
                }
            )

    monkeypatch.setattr(llm_mod, "AnthropicLLM", MacroLLM)
    monkeypatch.setenv("FXT_ANTHROPIC_API_KEY", "x")
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/chat", json={"message": "USD inflation view?"})
    body = r.json()
    assert body["mode"] == "agent"
    assert body["steps"] == ['macro({"currency": "USD"})']
    assert "325.6" in body["answer"]


async def test_endpoints_empty_db(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Empty DB → every endpoint returns 200 with empty payloads, no crashes."""
    monkeypatch.setenv("FXT_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/empty.db")
    get_settings.cache_clear()
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_session_factory", None)
    engine = make_engine(get_settings())
    monkeypatch.setattr(db_module, "_engine", engine)
    await init_db(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        for url in (
            "/api/strength",
            "/api/stances",
            "/api/themes",
            "/api/ideas",
            "/api/docs",
            "/api/prices?pair=EURUSD",
        ):
            r = await c.get(url)
            assert r.status_code == 200, url
    await engine.dispose()
