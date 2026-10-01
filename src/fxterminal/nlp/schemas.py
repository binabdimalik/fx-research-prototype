"""Pydantic schemas for LLM outputs.

The LLM may only classify/extract/cite. These schemas bound what it can return;
`evidence` sentences are additionally verified as verbatim substrings of the
source document by `verify_evidence` — citations it invented are rejected.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

StanceLabel = Literal["very_dovish", "dovish", "neutral", "hawkish", "very_hawkish"]
Direction = Literal["hawkish", "dovish"]
Topic = Literal[
    "inflation", "growth", "employment", "guidance", "balance_sheet", "risk", "fx", "other"
]


class EvidenceItem(BaseModel):
    """One verbatim citation supporting the stance."""

    sentence: str = Field(min_length=10, max_length=1000)
    topic: Topic = "other"
    direction: Direction


class StanceOutput(BaseModel):
    """Validated stance classification for one central-bank statement."""

    score: int = Field(ge=-100, le=100)  # -100 dovish .. +100 hawkish
    label: StanceLabel
    rationale: str = Field(min_length=20, max_length=4000)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=12)
    # Short quoted phrases that changed vs the previous statement (e.g.
    # '"transitory" removed', '"patient" added'). Empty list = no notable shift.
    key_shifts: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def label_matches_score(self) -> StanceOutput:
        """Label must be consistent with the numeric score sign/magnitude."""
        s = self.score
        if s <= -60:
            expected = "very_dovish"
        elif s <= -15:
            expected = "dovish"
        elif s < 15:
            expected = "neutral"
        elif s < 60:
            expected = "hawkish"
        else:
            expected = "very_hawkish"
        if self.label != expected:
            raise ValueError(f"label {self.label!r} inconsistent with score {s} (want {expected})")
        return self


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Whitespace/punctuation-insensitive normalization for citation checks."""
    t = _WS.sub(" ", text).strip().lower()
    # normalize curly quotes/apostrophes so LLM quotes with typographic
    # variants still verify as verbatim
    return t.replace("’", "'").replace("“", '"').replace("”", '"')  # noqa: RUF001


def verify_evidence(output: StanceOutput, source_text: str) -> list[str]:
    """Return list of problems: each evidence sentence must appear verbatim
    (whitespace-insensitive) in the source. Empty list => all citations check out."""
    src = _norm(source_text)
    problems = [
        f"evidence[{i}] not verbatim in source: {item.sentence[:80]!r}"
        for i, item in enumerate(output.evidence)
        if _norm(item.sentence) not in src
    ]
    return problems


# ---------------------------------------------------------------- themes


ThemeName = Literal[
    "inflation",
    "growth",
    "employment",
    "monetary_policy",
    "politics",
    "risk_sentiment",
    "trade",
    "other",
]


class ThemeEvidence(BaseModel):
    """Citation into one document of the analyzed batch.

    `doc_ref` is the model-facing index ("D1", "D2", ...) resolved to a real
    document id in code — the model never sees or invents DB identifiers.
    """

    doc_ref: str = Field(pattern=r"^D\d+$")
    sentence: str = Field(min_length=10, max_length=1000)


class ThemeItem(BaseModel):
    theme: ThemeName
    sentiment: float = Field(ge=-1.0, le=1.0)  # -1 bearish ccy .. +1 bullish
    weight: float = Field(ge=0.0, le=1.0)  # relevance/confidence
    summary: str = Field(min_length=10, max_length=1000)
    evidence: list[ThemeEvidence] = Field(min_length=1, max_length=6)


class ThemesOutput(BaseModel):
    """Per-currency themes extracted from a batch of documents."""

    themes: list[ThemeItem] = Field(min_length=1, max_length=8)


def verify_theme_evidence(output: ThemesOutput, doc_map: dict[str, str]) -> list[str]:
    """Verify each citation: doc_ref exists AND sentence is verbatim in it.

    `doc_map`: {"D1": raw_text_of_doc_1, ...} built by the caller.
    """
    problems: list[str] = []
    for i, theme in enumerate(output.themes):
        for j, ev in enumerate(theme.evidence):
            text = doc_map.get(ev.doc_ref)
            if text is None:
                problems.append(f"themes[{i}].evidence[{j}]: unknown doc_ref {ev.doc_ref!r}")
            elif _norm(ev.sentence) not in _norm(text):
                problems.append(
                    f"themes[{i}].evidence[{j}] not verbatim in {ev.doc_ref}: {ev.sentence[:80]!r}"
                )
    return problems
