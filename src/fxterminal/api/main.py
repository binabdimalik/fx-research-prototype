"""FastAPI application — health + read endpoints + dashboard + chat."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from fxterminal import __version__
from fxterminal.api.routes import router as api_router
from fxterminal.core.config import get_settings
from fxterminal.core.db import get_engine
from fxterminal.core.logging import configure_logging

_DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    yield


app = FastAPI(
    title="fxterminal",
    version=__version__,
    description="AI-powered FX research terminal. Research tool, not financial advice.",
    lifespan=lifespan,
)
app.include_router(api_router)

DISCLAIMER = "Research tool output — not financial advice."


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the single-page dashboard."""
    return HTMLResponse(_DASHBOARD.read_text(encoding="utf-8"))


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    """Readiness: DB reachable. Used by monitoring/compose healthchecks."""
    engine = get_engine()
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"status": "ready", "disclaimer": DISCLAIMER}
