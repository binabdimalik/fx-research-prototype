"""News theme extraction per currency over a rolling window.

Flow:
  1. Pull recent `news`/`speech` docs mentioning the currency (or from that
     currency's CB feed) in [window_start, window_end].
  2. LLM extracts themes + sentiment + cited evidence (doc_ref "D1" indexes
     into the batch — resolved to real doc ids in code afterwards).
  3. verify_theme_evidence: every citation must point at a real batch doc and
     be verbatim. Invalid output -> retry with feedback -> safe failure.
  4. Persist one Theme row per theme.

The LLM sees only truncated doc bodies + refs; doc ids/dates are attached in
code, so a theme row is fully traceable to its sources.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.core.config import BANK_CURRENCIES
from fxterminal.core.logging import get_logger
from fxterminal.models import RawDocument, Theme
from fxterminal.nlp.llm import LLMClient
from fxterminal.nlp.schemas import ThemesOutput, verify_theme_evidence

logger = get_logger(__name__)

PROMPT_VERSION = "themes-v1"
_MAX_DOC_CHARS = 1500  # per-doc truncation keeps prompts bounded

SYSTEM_PROMPT = """You are an FX macro analyst extracting currency-relevant themes from news.
Rules:
- Identify the 1-8 dominant themes for the currency over the window.
- theme: one of inflation|growth|employment|monetary_policy|politics|risk_sentiment|trade|other.
- sentiment: -1.0 (very bearish for the currency) .. +1.0 (very bullish).
- weight: 0..1 — how important this theme is right now (they need not sum to 1).
- evidence: cite supporting sentences. For each citation give doc_ref (the "Dn"
  label of the source document) and a sentence copied VERBATIM from that
  document's text. Never paraphrase citations.
- Reply with JSON only matching the requested schema."""


def build_theme_prompt(currency: str, docs: list[RawDocument]) -> tuple[str, str, dict[str, str]]:
    """Returns (system, user, doc_map). doc_map: "Dn" -> full raw_text used for
    citation verification (untruncated source of truth)."""
    blocks: list[str] = []
    doc_map: dict[str, str] = {}
    for i, doc in enumerate(docs, start=1):
        ref = f"D{i}"
        doc_map[ref] = doc.raw_text
        blocks.append(
            f'<{ref} source="{doc.source}" published="{doc.published_at.date().isoformat()}">\n'
            f"{doc.raw_text[:_MAX_DOC_CHARS]}\n</{ref}>"
        )
    user = (
        f"Currency under analysis: {currency}\n\n"
        + "\n\n".join(blocks)
        + '\n\nReturn JSON: {"themes": [{"theme": "...", "sentiment": float, '
        '"weight": float, "summary": "...", "evidence": [{"doc_ref": "Dn", "sentence": "..."}]}]}'
    )
    return SYSTEM_PROMPT, user, doc_map


async def fetch_theme_docs(
    session: AsyncSession,
    currency: str,
    window_end: datetime,
    window_days: int = 7,
    limit: int = 30,
) -> list[RawDocument]:
    """Docs for theme extraction: currency-tagged or that currency's CB feed."""
    bank_for_ccy = {ccy: bank for bank, ccy in BANK_CURRENCIES.items()}
    start = window_end - timedelta(days=window_days)
    q = (
        select(RawDocument)
        .where(
            RawDocument.published_at >= start,
            RawDocument.published_at <= window_end,
            RawDocument.doc_type.in_(["news", "speech"]),
            (RawDocument.currency == currency)
            | (RawDocument.central_bank == bank_for_ccy.get(currency)),
        )
        .order_by(RawDocument.published_at.desc())
        .limit(limit)
    )
    return list((await session.execute(q)).scalars().all())


async def extract_themes(
    session: AsyncSession,
    currency: str,
    llm: LLMClient,
    window_days: int = 7,
    window_end: datetime | None = None,
    limit: int = 30,
    prompt_version: str = PROMPT_VERSION,
) -> list[Theme]:
    """Extract + persist themes for one currency. Empty list => no docs or
    terminal failure (logged)."""
    end = window_end or datetime.now(UTC)
    start = end - timedelta(days=window_days)
    docs = await fetch_theme_docs(session, currency, end, window_days, limit)
    if not docs:
        logger.info("themes_no_docs", currency=currency)
        return []

    system, user, doc_map = build_theme_prompt(currency, docs)
    ref_to_doc = {f"D{i}": doc for i, doc in enumerate(docs, start=1)}

    feedback: str | None = None
    output: ThemesOutput | None = None
    for _attempt in range(3):
        output = await llm.complete_json(system, user, ThemesOutput, validation_feedback=feedback)
        problems = verify_theme_evidence(output, doc_map)
        if not problems:
            break
        feedback = (
            "Citations must be verbatim sentences from the cited document. "
            "Problems: " + "; ".join(problems)
        )
        output = None
    if output is None:
        logger.error("themes_evidence_failed", currency=currency)
        return []

    rows: list[Theme] = []
    for item in output.themes:
        evidence_rows = [
            {
                "doc_id": str(ref_to_doc[ev.doc_ref].id),
                "doc_ref": ev.doc_ref,
                "published_at": ref_to_doc[ev.doc_ref].published_at.isoformat(),
                "source": ref_to_doc[ev.doc_ref].source,
                "sentence": ev.sentence,
            }
            for ev in item.evidence
            if ev.doc_ref in ref_to_doc  # verified above; belt-and-braces
        ]
        rows.append(
            Theme(
                currency=currency,
                theme=item.theme,
                sentiment=item.sentiment,
                weight=item.weight,
                window_start=start,
                window_end=end,
                evidence=evidence_rows,
                model=getattr(llm, "model", "unknown"),
                prompt_version=prompt_version,
            )
        )
    session.add_all(rows)
    await session.commit()
    return rows


async def extract_all_themes(
    session: AsyncSession,
    currencies: list[str],
    llm: LLMClient,
    window_days: int = 7,
    limit: int = 30,
) -> dict[str, int]:
    """Run extraction for every configured currency; returns per-ccy counts."""
    results: dict[str, int] = {}
    for ccy in currencies:
        try:
            rows = await extract_themes(session, ccy, llm, window_days=window_days, limit=limit)
            results[ccy] = len(rows)
        except Exception as e:  # one currency's failure must not stop others
            logger.error("themes_failed", currency=ccy, error=str(e)[:200])
            results[ccy] = -1
    return results
