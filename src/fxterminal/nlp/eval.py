"""Evaluation harness for the stance classifier.

The eval set is small synthetic statement excerpts with hand-labeled expected
direction (real FOMC/ECB texts are copyrighted-ish and huge; synthetic snippets
that mirror their phrasing give a smoke-test of prompt + schema + verification
without redistributing full statements). Extend `EVAL_CASES` over time.

Run: `fxterminal nlp eval` (requires FXT_ANTHROPIC_API_KEY) or with a mock LLM
in tests. Report: per-case predicted label/score vs expected direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from fxterminal.core.logging import get_logger
from fxterminal.nlp.llm import LLMClient
from fxterminal.nlp.schemas import StanceLabel, StanceOutput, verify_evidence
from fxterminal.nlp.stance import SYSTEM_PROMPT, USER_TEMPLATE

logger = get_logger(__name__)

ExpectedDirection = Literal["hawkish", "dovish", "neutral"]


class EvalCase(BaseModel):
    id: str
    bank: str
    statement: str
    expected: ExpectedDirection


def direction_of(score: int) -> ExpectedDirection:
    if score <= -15:
        return "dovish"
    if score < 15:
        return "neutral"
    return "hawkish"


@dataclass
class EvalResult:
    case_id: str
    expected: ExpectedDirection
    predicted_label: StanceLabel | None
    predicted_direction: ExpectedDirection | None
    score: int | None
    correct: bool
    evidence_ok: bool
    error: str | None = None


# Synthetic snippets modeled on typical CB phrasing. Deliberately unambiguous —
# this set checks the pipeline works, not fine-grained edge cases.
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="fed-hike-hawkish",
        bank="FED",
        statement=(
            "The Committee decided to raise the target range for the federal funds "
            "rate by 25 basis points. Inflation remains elevated and job gains have "
            "been robust. The Committee anticipates that ongoing increases in the "
            "target range will be appropriate to return inflation to 2 percent."
        ),
        expected="hawkish",
    ),
    EvalCase(
        id="ecb-cut-dovish",
        bank="ECB",
        statement=(
            "The Governing Council today decided to lower the deposit facility rate "
            "by 25 basis points. Inflation has continued to decline and the outlook "
            "for economic growth has weakened markedly. The Council stands ready to "
            "adjust all of its instruments to ensure inflation stabilises at target."
        ),
        expected="dovish",
    ),
    EvalCase(
        id="boj-hold-neutral",
        bank="BOJ",
        statement=(
            "The Bank decided, by a unanimous vote, to maintain the guideline for "
            "money market operations. The economy has recovered moderately, although "
            "some weakness has been seen in part. Uncertainty regarding overseas "
            "economies warrants careful monitoring."
        ),
        expected="neutral",
    ),
    EvalCase(
        id="boe-cut-hawkish-signal",
        bank="BOE",
        statement=(
            "The Committee voted to reduce Bank Rate by 25 basis points while noting "
            "that services inflation remains persistent. The Committee will remain "
            "vigilant to inflationary pressures and does not pre-commit to any path."
        ),
        expected="hawkish",  # cut but vigilant — mixed-to-hawkish; tests nuance
    ),
]


async def run_eval(
    llm: LLMClient, cases: list[EvalCase] | None = None
) -> tuple[list[EvalResult], float]:
    """Run the stance prompt over eval cases. Returns (results, accuracy)."""
    results: list[EvalResult] = []
    for case in cases or EVAL_CASES:
        user = USER_TEMPLATE.format(
            bank=case.bank, published="2026-01-01", statement=case.statement, prev_block=""
        )
        try:
            out: StanceOutput = await llm.complete_json(SYSTEM_PROMPT, user, StanceOutput)
            problems = verify_evidence(out, case.statement)
            pred = direction_of(out.score)
            results.append(
                EvalResult(
                    case_id=case.id,
                    expected=case.expected,
                    predicted_label=out.label,
                    predicted_direction=pred,
                    score=out.score,
                    correct=pred == case.expected,
                    evidence_ok=not problems,
                    error="; ".join(problems) if problems else None,
                )
            )
        except Exception as e:
            results.append(
                EvalResult(
                    case_id=case.id,
                    expected=case.expected,
                    predicted_label=None,
                    predicted_direction=None,
                    score=None,
                    correct=False,
                    evidence_ok=False,
                    error=f"{type(e).__name__}: {e}",
                )
            )
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / len(results) if results else 0.0
    return results, accuracy
