"""Alert checks — notify on new trade ideas and large stance shifts.

Runs at the end of each scheduler cycle. Notifications go to the log always,
and to `FXT_ALERT_WEBHOOK` (generic JSON POST — Slack/Discord/custom) when
configured. Dedup is free: alerts key off immutable created_at/published_at
timestamps, so a second cycle inside the window only re-fires genuinely new
rows — no separate notifications table needed for a research tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fxterminal.core.config import Settings
from fxterminal.core.logging import get_logger
from fxterminal.models import RawDocument, StanceScore, TradeIdea

logger = get_logger(__name__)

LOOKBACK_HOURS = 26  # matches a daily cadence with slack for clock jitter
SHIFT_THRESHOLD = 30  # |shift_vs_prev| that counts as a real regime change


@dataclass
class Alert:
    kind: str  # "new_idea" | "stance_shift"
    message: str
    payload: dict = field(default_factory=dict)


async def find_new_ideas(session: AsyncSession, since: datetime) -> list[Alert]:
    q = select(TradeIdea).where(TradeIdea.created_at >= since)
    rows = (await session.execute(q)).scalars().all()
    out: list[Alert] = []
    for i in rows:
        ts = i.created_at
        if ts.tzinfo is None:  # sqlite rows come back naive — treat as UTC
            ts = ts.replace(tzinfo=UTC)
        if ts < since:
            continue
        out.append(
            Alert(
                kind="new_idea",
                message=(
                    f"NEW IDEA: {i.direction.upper()} {i.pair} "
                    f"entry {i.entry_low:.5f}-{i.entry_high:.5f} "
                    f"stop {i.stop:.5f} target {i.target:.5f} rr={i.risk_reward:.1f}"
                ),
                payload={"idea_id": i.id, "pair": i.pair, "direction": i.direction},
            )
        )
    return out


async def find_stance_shifts(session: AsyncSession, since: datetime) -> list[Alert]:
    """New stance scores since `since` with |shift_vs_prev| >= threshold."""
    q = (
        select(StanceScore, RawDocument.published_at)
        .join(RawDocument, StanceScore.statement_doc_id == RawDocument.id)
        .where(RawDocument.published_at >= since, StanceScore.shift_vs_prev.is_not(None))
    )
    out: list[Alert] = []
    for s, _pub in (await session.execute(q)).all():
        if abs(s.shift_vs_prev or 0) < SHIFT_THRESHOLD:
            continue
        out.append(
            Alert(
                kind="stance_shift",
                message=(
                    f"STANCE SHIFT: {s.central_bank}/{s.currency} "
                    f"{s.score:+.0f} ({s.label}), shift {s.shift_vs_prev:+.0f} vs prior"
                ),
                payload={"stance_id": s.id, "currency": s.currency, "shift": s.shift_vs_prev},
            )
        )
    return out


async def _deliver_webhook(settings: Settings, alerts: list[Alert]) -> None:
    """POST alerts to the configured webhook; failure logged, never raised."""
    if not settings.alert_webhook:
        return
    payload = {
        "text": "\n".join(f"• {a.message}" for a in alerts),
        "alerts": [{"kind": a.kind, "message": a.message, **a.payload} for a in alerts],
        "source": "fxterminal",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(settings.alert_webhook, json=payload)
            resp.raise_for_status()
    except Exception as e:
        logger.error("webhook_failed", error=str(e)[:200])


async def check_alerts(
    session: AsyncSession, settings: Settings, now: datetime | None = None
) -> str:
    """Detect + log + dispatch alerts. Returns a status string for the cycle."""
    now = now or datetime.now(UTC)
    since = now - timedelta(hours=LOOKBACK_HOURS)
    alerts = await find_new_ideas(session, since) + await find_stance_shifts(session, since)
    if not alerts:
        return "ok (no alerts)"
    for a in alerts:
        logger.info("alert", kind=a.kind, message=a.message)
    await _deliver_webhook(settings, alerts)
    return f"ok ({len(alerts)} alerts)"
