"""Phase 4 tests: theme extraction schema, citation verification, persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.models import RawDocument
from fxterminal.nlp.schemas import ThemesOutput, verify_theme_evidence
from fxterminal.nlp.themes import (
    build_theme_prompt,
    extract_themes,
    fetch_theme_docs,
)

NOW = datetime(2026, 9, 30, tzinfo=UTC)

DOC1_TEXT = "US CPI rose more than expected. The dollar rallied on the print."
DOC2_TEXT = "Fed officials signaled patience on cuts amid sticky services inflation."

THEMES_PAYLOAD = {
    "themes": [
        {
            "theme": "inflation",
            "sentiment": 0.6,
            "weight": 0.9,
            "summary": "Hot CPI supports USD via higher-for-longer rates.",
            "evidence": [
                {"doc_ref": "D1", "sentence": "US CPI rose more than expected."},
                {"doc_ref": "D2", "sentence": "sticky services inflation"},
            ],
        }
    ]
}


class ThemeMockLLM:
    model = "mock-llm"

    def __init__(self, payload: dict | None = None):
        self.payload = payload or THEMES_PAYLOAD
        self.calls: list[str | None] = []

    async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
        self.calls.append(validation_feedback)
        return schema.model_validate(self.payload)


def _news(text: str, days_ago: int, ccy: str = "USD", source: str = "fed_rss") -> RawDocument:
    return RawDocument(
        source=source,
        doc_type="news",
        central_bank="FED" if source == "fed_rss" else None,
        currency=ccy,
        url="https://x/1",
        title="t",
        published_at=NOW - timedelta(days=days_ago),
        content_hash=uuid.uuid4().hex,
        raw_text=text,
    )


async def test_theme_schema_and_verify() -> None:
    out = ThemesOutput.model_validate(THEMES_PAYLOAD)
    doc_map = {"D1": DOC1_TEXT, "D2": DOC2_TEXT}
    assert verify_theme_evidence(out, doc_map) == []

    bad_ref = ThemesOutput.model_validate(
        {
            **THEMES_PAYLOAD,
            "themes": [
                {
                    **THEMES_PAYLOAD["themes"][0],
                    "evidence": [{"doc_ref": "D9", "sentence": "anything at all."}],
                }
            ],
        }
    )
    problems = verify_theme_evidence(bad_ref, doc_map)
    assert problems and "unknown doc_ref" in problems[0]

    bad_quote = ThemesOutput.model_validate(
        {
            **THEMES_PAYLOAD,
            "themes": [
                {
                    **THEMES_PAYLOAD["themes"][0],
                    "evidence": [{"doc_ref": "D1", "sentence": "CPI fell sharply today."}],
                }
            ],
        }
    )
    problems = verify_theme_evidence(bad_quote, doc_map)
    assert problems and "not verbatim" in problems[0]


def test_theme_prompt_builds_refs() -> None:
    docs = [_news(DOC1_TEXT, 1), _news(DOC2_TEXT, 2)]
    _sys, user, doc_map = build_theme_prompt("USD", docs)
    assert "<D1" in user and "<D2" in user
    assert doc_map["D1"] == DOC1_TEXT
    assert "Currency under analysis: USD" in user


async def test_fetch_theme_docs_window(session: AsyncSession) -> None:
    session.add_all(
        [
            _news(DOC1_TEXT, 1),
            _news(DOC2_TEXT, 2),
            _news("Old news far away.", 30),  # outside 7d window
            _news(DOC1_TEXT, 1, ccy="EUR", source="ecb_rss"),  # other currency
        ]
    )
    await session.commit()
    docs = await fetch_theme_docs(session, "USD", NOW, window_days=7)
    assert len(docs) == 2  # only USD/fed docs inside window


async def test_extract_themes_persists(session: AsyncSession) -> None:
    session.add_all([_news(DOC1_TEXT, 1), _news(DOC2_TEXT, 2)])
    await session.commit()

    rows = await extract_themes(session, "USD", ThemeMockLLM(), window_end=NOW)
    assert len(rows) == 1
    t = rows[0]
    assert t.currency == "USD" and t.theme == "inflation"
    assert t.sentiment == 0.6 and t.weight == 0.9
    # citations resolved to real doc ids
    assert len(t.evidence) == 2
    assert all("doc_id" in e and "sentence" in e for e in t.evidence)
    assert t.prompt_version == "themes-v1"


async def test_extract_themes_no_docs(session: AsyncSession) -> None:
    assert await extract_themes(session, "JPY", ThemeMockLLM()) == []


async def test_extract_themes_retries_bad_evidence(session: AsyncSession) -> None:
    session.add(_news(DOC1_TEXT, 1))
    await session.commit()

    class FlipLLM(ThemeMockLLM):
        def __init__(self) -> None:
            super().__init__()
            self.payloads = [
                {
                    "themes": [
                        {
                            "theme": "inflation",
                            "sentiment": 0.5,
                            "weight": 0.5,
                            "summary": "bad quote test",
                            "evidence": [{"doc_ref": "D1", "sentence": "not in doc"}],
                        }
                    ]
                },
                THEMES_PAYLOAD,
            ]

        async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
            self.calls.append(validation_feedback)
            idx = min(len(self.calls) - 1, len(self.payloads) - 1)
            return schema.model_validate(self.payloads[idx])

    llm = FlipLLM()
    rows = await extract_themes(session, "USD", llm, window_end=NOW)
    # second payload's D2 evidence fails (only 1 doc in batch) — check behavior:
    # D2 is an unknown doc_ref -> retried -> third call same payload again fails.
    assert rows == []
    assert len(llm.calls) == 3  # retried until exhausted
    assert llm.calls[1] is not None  # feedback was passed back
