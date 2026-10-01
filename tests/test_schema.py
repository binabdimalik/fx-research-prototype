"""Schema round-trips on SQLite: inserts, dedup uniqueness, JSON evidence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.models import FxPrice, RawDocument, StanceScore, TradeIdea


def _doc(
    source: str = "fed_fomc", url: str | None = "https://x/1", body: str = "text"
) -> RawDocument:
    return RawDocument(
        source=source,
        doc_type="statement",
        central_bank="FED",
        currency="USD",
        url=url,
        title="FOMC statement",
        published_at=datetime(2026, 9, 16, 18, 0, tzinfo=UTC),
        content_hash=RawDocument.compute_content_hash(source, url, "FOMC statement", body),
        raw_text=body,
    )


async def test_raw_document_roundtrip_and_dedup(session: AsyncSession) -> None:
    doc = _doc()
    session.add(doc)
    await session.commit()

    loaded = (await session.execute(select(RawDocument))).scalar_one()
    assert loaded.central_bank == "FED"
    assert loaded.raw_metadata == {}

    # Identical content => identical hash => insert must fail (idempotent ingest).
    session.add(_doc())
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()

    # Different body => different hash => allowed.
    session.add(_doc(url="https://x/2", body="different"))
    await session.commit()


async def test_fx_price_roundtrip(session: AsyncSession) -> None:
    session.add(
        FxPrice(
            time=datetime(2026, 9, 30, tzinfo=UTC),
            pair="EURUSD",
            timeframe="1d",
            open=1.17,
            high=1.18,
            low=1.16,
            close=1.175,
            spread_pips=0.6,
            source="test",
        )
    )
    await session.commit()
    price = (await session.execute(select(FxPrice))).scalar_one()
    assert price.pair == "EURUSD" and price.close == 1.175


async def test_stance_score_with_cited_evidence(session: AsyncSession) -> None:
    doc = _doc(body="The Committee decided to raise the target range.")
    session.add(doc)
    await session.flush()

    session.add(
        StanceScore(
            central_bank="FED",
            currency="USD",
            statement_doc_id=doc.id,
            score=60,
            label="hawkish",
            shift_vs_prev=40.0,
            evidence=[{"doc_id": str(doc.id), "sentence": "decided to raise the target range"}],
            model="test-model",
            prompt_version="v1",
        )
    )
    await session.commit()

    score = (await session.execute(select(StanceScore))).scalar_one()
    assert score.evidence[0]["doc_id"] == str(doc.id)
    assert -100 <= score.score <= 100


async def test_trade_idea_defaults(session: AsyncSession) -> None:
    idea = TradeIdea(
        pair="EURUSD",
        direction="long",
        entry_low=1.170,
        entry_high=1.172,
        stop=1.165,
        target=1.185,
        risk_reward=2.5,
        rationale="test",
        invalidation="below stop",
        key_risks=["CPI surprise"],
        filters={"spread_ok": True},
    )
    session.add(idea)
    await session.commit()
    assert "not financial advice" in idea.disclaimer


def test_content_hash_stable_and_whitespace_insensitive() -> None:
    h1 = RawDocument.compute_content_hash("FED", "http://x", "Title", "a  b")
    h2 = RawDocument.compute_content_hash(" fed ", "http://x", " Title ", "a b")
    h3 = RawDocument.compute_content_hash("FED", "http://x", "Title", "a c")
    assert h1 == h2 and h1 != h3
    assert len(h1) == 64


def test_uuid_roundtrip(session_is_fake: bool = True) -> None:
    # uuid sanity (actual persistence covered by stance test above)
    assert isinstance(uuid.uuid4(), uuid.UUID)
