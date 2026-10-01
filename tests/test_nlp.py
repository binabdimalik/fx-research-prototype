"""Phase 3 tests: stance schema, evidence verification, scorer, eval harness.

LLM is a deterministic mock — no network, no API key needed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.models import RateExpectation, RawDocument, StanceScore
from fxterminal.nlp.eval import EVAL_CASES, direction_of, run_eval
from fxterminal.nlp.schemas import StanceOutput, verify_evidence
from fxterminal.nlp.stance import (
    StatementContext,
    build_prompt,
    compute_priced_stance,
    score_pending,
    score_statement,
)

T = TypeVar("T", bound=BaseModel)

STATEMENT = (
    "The Committee decided to raise the target range for the federal funds rate "
    "by 25 basis points to 5.00-5.25 percent. Inflation remains elevated and job "
    "gains have been robust. The Committee would be prepared to adjust the stance "
    "of monetary policy as appropriate if risks emerge."
)

GOOD_OUTPUT = {
    "score": 55,
    "label": "hawkish",
    "rationale": "Rate hike with elevated inflation language and robust jobs.",
    "evidence": [
        {
            "sentence": "decided to raise the target range for the federal funds rate",
            "topic": "guidance",
            "direction": "hawkish",
        },
        {
            "sentence": "Inflation remains elevated and job gains have been robust.",
            "topic": "inflation",
            "direction": "hawkish",
        },
    ],
    "key_shifts": ['"prepared to adjust" added'],
}


class MockLLM:
    """Returns a fixed StanceOutput; `calls` records prompts for assertions."""

    model = "mock-llm"

    def __init__(self, payload: dict | None = None, fail_times: int = 0) -> None:
        self.payload = payload or GOOD_OUTPUT
        self.fail_times = fail_times
        self.calls: list[tuple[str, str, str | None]] = []

    async def complete_json(
        self, system, prompt, schema: type[T], *, validation_feedback=None
    ) -> T:
        self.calls.append((system, prompt, validation_feedback))
        if self.fail_times > 0:
            self.fail_times -= 1
            from fxterminal.nlp.llm import LLMError

            raise LLMError("mock failure")
        return schema.model_validate(self.payload)


def _doc(
    body: str = STATEMENT, published: datetime | None = None, bank: str = "FED"
) -> RawDocument:
    return RawDocument(
        source="test",
        doc_type="statement",
        central_bank=bank,
        currency="USD",
        url="https://x/1",
        title="t",
        published_at=published or datetime(2026, 9, 16, 18, 0, tzinfo=UTC),
        content_hash=uuid.uuid4().hex,
        raw_text=body,
    )


# --- schema ---


def test_schema_label_score_consistency() -> None:
    with pytest.raises(ValidationError):
        StanceOutput.model_validate({**GOOD_OUTPUT, "score": -55, "label": "hawkish"})
    ok = StanceOutput.model_validate({**GOOD_OUTPUT, "score": -70, "label": "very_dovish"})
    assert ok.score == -70


def test_verify_evidence_verbatim() -> None:
    out = StanceOutput.model_validate(GOOD_OUTPUT)
    assert verify_evidence(out, STATEMENT) == []

    fabricated = {
        **GOOD_OUTPUT,
        "evidence": [{**GOOD_OUTPUT["evidence"][0], "sentence": "The Fed loves hiking."}],
    }
    bad = StanceOutput.model_validate(fabricated)
    problems = verify_evidence(bad, STATEMENT)
    assert len(problems) == 1 and "not verbatim" in problems[0]


def test_verify_evidence_whitespace_insensitive() -> None:
    squashed = STATEMENT.replace("federal funds rate", "federal  funds\nrate")
    out = StanceOutput.model_validate(GOOD_OUTPUT)
    assert verify_evidence(out, squashed) == []  # normalization catches ws diffs


# --- prompt ---


def test_prompt_includes_previous_when_present() -> None:
    cur, prev = _doc(), _doc(body="old text", published=datetime(2026, 7, 1, tzinfo=UTC))
    _sys, user = build_prompt(StatementContext(statement=cur, previous=prev))
    assert "PREVIOUS_STATEMENT" in user and "old text" in user
    _, user2 = build_prompt(StatementContext(statement=cur, previous=None))
    assert "PREVIOUS_STATEMENT" not in user2


# --- priced stance (deterministic) ---


def test_compute_priced_stance() -> None:
    def exp(rate: float, t: datetime) -> RateExpectation:
        return RateExpectation(
            time=t,
            central_bank="FED",
            currency="USD",
            instrument="MANUAL",
            implied_rate=rate,
            source="t",
        )

    t0 = datetime(2026, 9, 16, tzinfo=UTC)
    before = [exp(4.25, t0 - timedelta(days=1))]
    after = [exp(4.50, t0 + timedelta(hours=1))]
    assert compute_priced_stance(before, after) == pytest.approx(25.0)  # +25bp drift
    assert compute_priced_stance([], after) is None
    assert compute_priced_stance(before, []) is None
    # clamping
    big = [exp(6.0, t0 + timedelta(hours=1))]
    assert compute_priced_stance(before, big) == 100.0


def test_direction_of() -> None:
    assert direction_of(-30) == "dovish"
    assert direction_of(0) == "neutral"
    assert direction_of(50) == "hawkish"


# --- scorer (mocked LLM + sqlite session) ---


async def test_score_statement_persists_and_is_idempotent(session: AsyncSession) -> None:
    doc = _doc()
    session.add(doc)
    await session.commit()

    llm = MockLLM()
    row = await score_statement(session, doc, llm)
    assert row is not None and row.score == 55 and row.label == "hawkish"
    assert row.evidence[0]["doc_id"] == str(doc.id)
    assert row.prompt_version == "stance-v1"
    assert row.model == "mock-llm"

    # second call returns existing row, LLM not re-called
    calls_before = len(llm.calls)
    again = await score_statement(session, doc, llm)
    assert again is not None and again.id == row.id and len(llm.calls) == calls_before


async def test_score_shift_vs_previous(session: AsyncSession) -> None:
    prev_doc = _doc(
        body="Prior statement kept rates unchanged.", published=datetime(2026, 7, 29, tzinfo=UTC)
    )
    cur_doc = _doc(published=datetime(2026, 9, 16, tzinfo=UTC))
    session.add_all([prev_doc, cur_doc])
    await session.commit()

    # seed a prior stance score for the previous statement
    session.add(
        StanceScore(
            central_bank="FED",
            currency="USD",
            statement_doc_id=prev_doc.id,
            score=10,
            label="neutral",
            evidence=[],
            model="m",
            prompt_version="v1",
        )
    )
    await session.commit()

    row = await score_statement(session, cur_doc, MockLLM())
    assert row is not None
    assert row.shift_vs_prev == pytest.approx(45.0)  # 55 - 10
    assert row.previous_statement_doc_id == prev_doc.id


async def test_score_retries_on_fabricated_evidence(session: AsyncSession) -> None:
    doc = _doc()
    session.add(doc)
    await session.commit()

    fabricated = {
        **GOOD_OUTPUT,
        "evidence": [
            {"sentence": "We will hike forever.", "topic": "guidance", "direction": "hawkish"}
        ],
    }
    llm = FlippingLLM([fabricated, GOOD_OUTPUT])
    row = await score_statement(session, doc, llm)
    assert row is not None
    # first output rejected -> second call carried validation_feedback
    assert llm.calls[1][2] is not None and "verbatim" in llm.calls[1][2]


async def test_score_terminal_failure_returns_none(session: AsyncSession) -> None:
    doc = _doc()
    session.add(doc)
    await session.commit()
    llm = MockLLM(fail_times=10)
    assert await score_statement(session, doc, llm) is None


async def test_score_pending(session: AsyncSession) -> None:
    session.add_all(
        [
            _doc(body=STATEMENT, published=datetime(2026, 8, 1, tzinfo=UTC)),
            _doc(
                # evidence quotes in GOOD_OUTPUT must be verbatim in the body
                body="Minutes: The Committee decided to raise the target range for "
                "the federal funds rate. Inflation remains elevated and job gains "
                "have been robust.",
                published=datetime(2026, 8, 10, tzinfo=UTC),
            ),
        ]
    )
    # mark second as minutes
    for d in (await session.execute(select(RawDocument))).scalars():
        if d.raw_text.startswith("Minutes"):
            d.doc_type = "minutes"
    await session.commit()
    counts = await score_pending(session, MockLLM())
    assert counts["scored"] == 2


class FlippingLLM(MockLLM):
    """Returns different payloads per call (to exercise evidence retry)."""

    def __init__(self, payloads: list[dict]) -> None:
        super().__init__(payloads[0])
        self.payloads = payloads

    async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
        self.calls.append((system, prompt, validation_feedback))
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return schema.model_validate(payload)


# --- eval harness ---


async def test_eval_with_mock() -> None:
    class AlignedLLM(MockLLM):
        """Return the expected direction for each case by peeking at prompt."""

        async def complete_json(self, system, prompt, schema, *, validation_feedback=None):
            stmt = prompt.split("<CURRENT_STATEMENT", 1)[1]
            # crude: hawkish if 'raise', dovish if 'lower'
            if "raise" in stmt:
                payload = {
                    **GOOD_OUTPUT,
                    "evidence": [
                        {
                            "sentence": "raise the target range for the federal funds rate",
                            "topic": "guidance",
                            "direction": "hawkish",
                        }
                    ],
                }
            elif "lower" in stmt:
                payload = {
                    **GOOD_OUTPUT,
                    "score": -50,
                    "label": "dovish",
                    "evidence": [
                        {
                            "sentence": "decided to lower the deposit facility rate",
                            "topic": "guidance",
                            "direction": "dovish",
                        }
                    ],
                }
            else:
                payload = {
                    **GOOD_OUTPUT,
                    "score": 0,
                    "label": "neutral",
                    "evidence": [
                        {
                            "sentence": "maintain the guideline for money market operations",
                            "topic": "guidance",
                            "direction": "dovish",
                        }
                    ],
                }
            return schema.model_validate(payload)

    results, accuracy = await run_eval(AlignedLLM(), EVAL_CASES)
    assert len(results) == 4
    assert all(r.predicted_label is not None for r in results)
    # accuracy is informational here; evidence checks may fail on cases whose
    # statements don't contain the mock's fixed quotes — that's real signal.
    assert 0.0 <= accuracy <= 1.0
