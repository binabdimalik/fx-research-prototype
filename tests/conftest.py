from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from fxterminal.core.config import Settings, get_settings
from fxterminal.core.db import make_engine, make_session_factory
from fxterminal.models.base import Base


@pytest.fixture
def settings() -> Settings:
    """Isolated settings pointing at a throwaway in-memory-ish SQLite file."""
    get_settings.cache_clear()
    return Settings(
        app_env="test",
        database_url="sqlite+aiosqlite:///:memory:",
        _env_file=None,
    )


@pytest.fixture
async def engine(settings: Settings) -> AsyncEngine:
    eng = make_engine(settings)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncSession:
    factory = make_session_factory(engine)
    async with factory() as s:
        yield s
