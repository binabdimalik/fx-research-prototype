"""Central-bank stance scorer.

Flow per statement:
  1. Build prompt: current statement (+ previous statement text when available).
  2. LLM classifies -> StanceOutput (validated schema, verbatim evidence).
  3. Deterministic post-processing in code:
       shift_vs_prev      = current score - previous score (when known)
       surprise_vs_pricing = current score - priced_stance, where priced_stance
         is derived from the change in market-implied rates between statements
       (see `compute_priced_stance` — an approximation, documented, unit-tested).
  4. Persist StanceScore row with evidence + model + prompt_version.

PROMPT_VERSION must be bumped on any prompt change so stored scores stay
traceable to the exact instructions that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.core.logging import get_logger
from fxterminal.models import RateExpectation, RawDocument, StanceScore
from fxterminal.nlp.llm import LLMClient, LLMError
from fxterminal.nlp.schemas import StanceOutput, verify_evidence

logger = get_logger(__name__)

PROMPT_VERSION = "stance-v1"

SYSTEM_PROMPT = """You are a central-bank language analyst for an FX research terminal.
Classify the stance of the statement. Rules:
- score: integer -100 (maximally dovish) .. +100 (maximally hawkish).
- label: very_dovish (<= -60), dovish (-60..-15], neutral (-15..15),
  hawkish (15..60), very_hawkish (>60).
- Base the score ONLY on the statement text: policy action, guidance, inflation/
  growth assessment, risk language. Ignore boilerplate formatting.
- evidence: 1-8 items, each a VERBATIM sentence or clause copied exactly from
  the statement (never paraphrase), tagged with topic and the direction it implies.
- key_shifts: if a previous statement is provided, list short quoted phrases
  that changed (removed/added/reworded); else an empty list.
- Reply with JSON only, matching the requested schema exactly. No commentary."""

USER_TEMPLATE = """Classify the {bank} statement below.

<CURRENT_STATEMENT published="{published}">
{statement}
</CURRENT_STATEMENT>

{prev_block}
Return JSON:
{{"score": int, "label": "...", "rationale": "...",
  "evidence": [{{"sentence": "...",
                "topic": "inflation|growth|employment|guidance|balance_sheet|risk|fx|other",
                "direction": "hawkish|dovish"}}],
  "key_shifts": ["..."]}}"""

PREV_BLOCK = """<PREVIOUS_STATEMENT published="{prev_published}">
{prev_text}
</PREVIOUS_STATEMENT>
Compare CURRENT against PREVIOUS for key_shifts."""


@dataclass(frozen=True)
class StatementContext:
    """Inputs for one scoring call."""

    statement: RawDocument
    previous: RawDocument | None


def build_prompt(ctx: StatementContext) -> tuple[str, str]:
    """Returns (system, user) prompt pair. Pure function — unit-testable."""
    prev_block = ""
    if ctx.previous is not None:
        prev_block = PREV_BLOCK.format(
            prev_published=ctx.previous.published_at.date().isoformat(),
            prev_text=ctx.previous.raw_text[:20000],
        )
    user = USER_TEMPLATE.format(
        bank=ctx.statement.central_bank or "central bank",
        published=ctx.statement.published_at.date().isoformat(),
        statement=ctx.statement.raw_text[:30000],
        prev_block=prev_block,
    )
    return SYSTEM_PROMPT, user


async def find_previous_statement(
    session: AsyncSession, central_bank: str, before: datetime
) -> RawDocument | None:
    """Most recent prior statement/minutes for the same bank."""
    q = (
        select(RawDocument)
        .where(
            RawDocument.central_bank == central_bank,
            RawDocument.doc_type.in_(["statement", "minutes"]),
            RawDocument.published_at < before,
        )
        .order_by(RawDocument.published_at.desc())
        .limit(1)
    )
    return (await session.execute(q)).scalars().first()


async def previous_score(session: AsyncSession, bank: str, doc_id) -> float | None:
    """Score of the most recent StanceScore for this bank before `doc_id`."""
    res = await session.execute(select(RawDocument.published_at).where(RawDocument.id == doc_id))
    pub = res.scalar_one_or_none()
    if pub is None:
        return None
    q = (
        select(StanceScore.score)
        .join(RawDocument, StanceScore.statement_doc_id == RawDocument.id)
        .where(StanceScore.central_bank == bank, RawDocument.published_at < pub)
        .order_by(RawDocument.published_at.desc())
        .limit(1)
    )
    return (await session.execute(q)).scalars().first()


def compute_priced_stance(
    exp_before: list[RateExpectation], exp_after: list[RateExpectation]
) -> float | None:
    """Approximate what the market was pricing, from implied-rate drift.

    Idea: if implied policy rates drifted up between the two statements,
    markets had already priced a hawkish turn — so a hawkish statement is less
    surprising. We map the drift (percentage points) onto the -100..+100 scale
    with a saturating scale: 25bp of drift ~= 25 points.

    Returns None when there is no usable expectation data — the caller stores
    NULL rather than a fabricated value.
    """
    if not exp_before or not exp_after:
        return None
    before = max(exp_before, key=lambda r: r.time).implied_rate
    after = min(exp_after, key=lambda r: r.time).implied_rate
    drift_pp = after - before  # positive = market moved more hawkish
    priced = drift_pp * 100.0  # 1pp of drift -> 100 (full scale)
    return max(-100.0, min(100.0, priced))


async def _expectations_around(
    session: AsyncSession, bank: str, t1: datetime, t2: datetime
) -> tuple[list, list]:
    """Latest expectations before the statement vs earliest after it.

    Window: 30 days either side — enough for daily snapshots to exist.
    """
    from datetime import timedelta

    win = timedelta(days=30)
    before_q = (
        select(RateExpectation)
        .where(
            RateExpectation.central_bank == bank,
            RateExpectation.time >= t1 - win,
            RateExpectation.time <= t1,
        )
        .order_by(RateExpectation.time)
    )
    after_q = (
        select(RateExpectation)
        .where(
            RateExpectation.central_bank == bank,
            RateExpectation.time >= t1,
            RateExpectation.time <= t2 + win,
        )
        .order_by(RateExpectation.time)
    )
    before = list((await session.execute(before_q)).scalars().all())
    after = list((await session.execute(after_q)).scalars().all())
    return before, after


async def score_statement(
    session: AsyncSession,
    statement: RawDocument,
    llm: LLMClient,
    prompt_version: str = PROMPT_VERSION,
) -> StanceScore | None:
    """Score one statement; persist + return the row. Returns existing row if
    already scored (idempotent), None on terminal LLM failure (logged)."""
    existing = (
        (
            await session.execute(
                select(StanceScore).where(StanceScore.statement_doc_id == statement.id)
            )
        )
        .scalars()
        .first()
    )
    if existing:
        return existing

    previous = await find_previous_statement(
        session, statement.central_bank or "", statement.published_at
    )
    system, user = build_prompt(StatementContext(statement=statement, previous=previous))

    feedback: str | None = None
    output: StanceOutput | None = None
    for _attempt in range(3):
        try:
            output = await llm.complete_json(
                system, user, StanceOutput, validation_feedback=feedback
            )
        except LLMError as e:
            logger.error("stance_llm_failed", doc=str(statement.id), error=str(e))
            return None
        problems = verify_evidence(output, statement.raw_text)
        if not problems:
            break
        feedback = "Evidence must be verbatim quotes from the statement. Problems: " + "; ".join(
            problems
        )
        output = None
    if output is None:
        logger.error("stance_evidence_failed", doc=str(statement.id))
        return None

    # deterministic deltas in code — never delegated to the model
    prev_score = await previous_score(session, statement.central_bank or "", statement.id)
    shift = float(output.score - prev_score) if prev_score is not None else None

    exp_before, exp_after = await _expectations_around(
        session,
        statement.central_bank or "",
        statement.published_at,
        statement.published_at,
    )
    priced = compute_priced_stance(exp_before, exp_after)
    surprise = float(output.score - priced) if priced is not None else None

    evidence_rows = [
        {
            "doc_id": str(statement.id),
            "published_at": statement.published_at.isoformat(),
            "sentence": item.sentence,
            "topic": item.topic,
            "direction": item.direction,
        }
        for item in output.evidence
    ]
    row = StanceScore(
        central_bank=statement.central_bank or "",
        currency=statement.currency or "",
        statement_doc_id=statement.id,
        previous_statement_doc_id=previous.id if previous else None,
        score=output.score,
        label=output.label,
        shift_vs_prev=shift,
        surprise_vs_pricing=surprise,
        evidence=evidence_rows,
        model=getattr(llm, "model", "unknown"),
        prompt_version=prompt_version,
    )
    session.add(row)
    await session.commit()
    return row


async def score_pending(
    session: AsyncSession,
    llm: LLMClient,
    bank: str | None = None,
    limit: int = 20,
) -> dict[str, int]:
    """Score all unscored statements/minutes. Returns counts for reporting."""
    scored_ids = select(StanceScore.statement_doc_id)
    q = (
        select(RawDocument)
        .where(
            RawDocument.doc_type.in_(["statement", "minutes"]),
            RawDocument.id.notin_(scored_ids),
        )
        .order_by(RawDocument.published_at)
        .limit(limit)
    )
    if bank:
        q = q.where(RawDocument.central_bank == bank)
    docs = list((await session.execute(q)).scalars().all())

    counts = {"scored": 0, "failed": 0, "skipped_no_bank": 0}
    for doc in docs:
        if not doc.central_bank:
            counts["skipped_no_bank"] += 1
            continue
        row = await score_statement(session, doc, llm)
        counts["scored" if row else "failed"] += 1
    return counts
