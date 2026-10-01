"""Daily pipeline scheduler.

`run_cycle` executes the full refresh in dependency order, each step isolated:
    ingest (prices+docs+macro+rates) -> stance scoring -> theme extraction
    -> strength composite -> trade-idea generation -> alert check.

A step raising doesn't abort the cycle — it's logged and the remaining steps
still run (a broken price feed shouldn't suppress stance scoring). LLM steps
are skipped cleanly when no key is configured.

`serve_forever` is a plain asyncio loop on `schedule_interval_hours` — no
Celery/Redis dependency for a single-user research box.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.core.config import Settings, get_settings
from fxterminal.core.db import get_session_factory
from fxterminal.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CycleResult:
    started_at: datetime
    steps: dict[str, str] = field(default_factory=dict)  # name -> "ok"|"skip"|error

    @property
    def ok(self) -> bool:
        return all(v in ("ok", "skip") for v in self.steps.values())


async def _ingest_all(session: AsyncSession, settings: Settings) -> str:
    """Delegate to the canonical sweep — `run_all` wires providers and pair
    derivation; we pass a rolling 35-day window (not its month-start default)
    so weekends/outages always get filled. Upserts dedup overlap."""
    from fxterminal.ingest.pipeline import run_all

    since = (datetime.now(UTC) - timedelta(days=35)).date().isoformat()
    reports = await run_all(session, settings, since=since)
    errors = [r for r in reports if r.error]
    inserted = sum(r.inserted for r in reports)
    return (
        f"ok ({inserted} rows, {len(errors)} provider errors: "
        + "; ".join(f"{r.provider}" for r in errors)[:120]
        + ")"
        if errors
        else f"ok ({inserted} rows)"
    )


def _llm_or_none(settings: Settings):
    try:
        from fxterminal.nlp.llm import build_llm

        return build_llm(settings)
    except Exception:
        return None


async def run_cycle(session: AsyncSession, settings: Settings | None = None) -> CycleResult:
    """One full pipeline pass. Each step is isolated — failures are recorded,
    not raised."""
    settings = settings or get_settings()
    result = CycleResult(started_at=datetime.now(UTC))

    async def step(name: str, coro) -> None:
        try:
            result.steps[name] = await coro
        except Exception as e:
            result.steps[name] = f"error: {e}"[:150]
            logger.exception("cycle_step_failed", step=name)
        logger.info("cycle_step", step=name, status=result.steps[name])

    await step("ingest", _ingest_all(session, settings))

    llm = _llm_or_none(settings)
    if llm is None:
        result.steps["stance"] = "skip: no LLM key"
        result.steps["themes"] = "skip: no LLM key"
    else:
        from fxterminal.nlp.stance import score_pending
        from fxterminal.nlp.themes import extract_all_themes

        async def score_step() -> str:
            n = await score_pending(session, llm)
            return f"ok ({n} scored)" if isinstance(n, int) else "ok"

        await step("stance", score_step())

        async def themes_step() -> str:
            res = await extract_all_themes(session, settings.currency_list, llm)
            return f"ok ({res})" if isinstance(res, dict) else "ok"

        await step("themes", themes_step())

    async def strength_step() -> str:
        from fxterminal.analytics.strength import compute_all

        rows = await compute_all(session, settings.currency_list)
        return f"ok ({len(rows)} scores)"

    await step("strength", strength_step())

    async def ideas_step() -> str:
        from fxterminal.analytics.ideas import generate_ideas
        from fxterminal.analytics.pairs import rank_pairs
        from fxterminal.analytics.strength import compute_strength
        from fxterminal.ingest.pipeline import default_pairs

        strengths = {}
        for c in settings.currency_list:
            res = await compute_strength(session, c)
            if res:
                strengths[c] = res
        views = rank_pairs(strengths, default_pairs(settings.currency_list))
        ideas = await generate_ideas(session, views, strengths, settings)
        return f"ok ({len(ideas)} ideas)"

    await step("ideas", ideas_step())

    async def alerts_step() -> str:
        from fxterminal.alerts.notifier import check_alerts

        return await check_alerts(session, settings)

    await step("alerts", alerts_step())

    logger.info("cycle_done", steps=result.steps)
    return result


async def intraday_tick(session: AsyncSession, settings: Settings | None = None) -> str:
    """Cheap between-cycle refresh: latest '1h' bars from the intraday
    provider. Runs in its own session; failures are reported, never raised."""
    settings = settings or get_settings()
    if not settings.intraday_prices_enabled:
        return "skip: intraday prices disabled"
    from fxterminal.ingest.pipeline import build_providers, default_pairs, ingest_prices

    provider = build_providers(settings).get("intraday_prices")
    if provider is None:
        return "skip: no intraday provider"
    since = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()
    rep = await ingest_prices(session, provider, default_pairs(settings.currency_list), since)
    if rep.error:
        return f"error: {rep.error}"[:150]
    return f"ok ({rep.inserted} new bars, {rep.duplicates} dupes)"


async def _intraday_loop(settings: Settings) -> None:
    """Fast price-only loop running alongside the full daily cycle."""
    factory = get_session_factory()
    tick_s = max(60, settings.intraday_interval_minutes * 60)
    while True:
        if settings.intraday_prices_enabled:
            async with factory() as session:
                try:
                    status = await intraday_tick(session, settings)
                except Exception as e:
                    status = f"error: {type(e).__name__}: {e}"[:150]
                    logger.exception("intraday_tick_failed")
                logger.info("intraday_tick", status=status)
        await asyncio.sleep(tick_s)


async def serve_forever(interval_hours: float = 24.0) -> None:
    """Full cycle every `interval_hours` + intraday price tick every
    FXT_INTRADAY_INTERVAL_MINUTES. Two loops on one event loop — each opens
    its own session, so a slow cycle never blocks the price tick."""
    settings = get_settings()

    async def _cycle_loop() -> None:
        factory = get_session_factory()
        while True:
            async with factory() as session:
                result = await run_cycle(session)
                logger.info("cycle_result", ok=result.ok, steps=result.steps)
            await asyncio.sleep(interval_hours * 3600)

    await asyncio.gather(_cycle_loop(), _intraday_loop(settings))
