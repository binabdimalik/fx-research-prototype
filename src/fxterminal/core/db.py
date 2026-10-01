"""Async database session plumbing.

The engine is built lazily from `get_settings()` so tests can override
`FXT_DATABASE_URL` before first use. `init_db()` (metadata.create_all) is a
dev/SQLite convenience — real environments should use `alembic upgrade head`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from fxterminal.core.config import Settings, get_settings


def make_engine(settings: Settings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    kwargs: dict[str, object] = {"echo": settings.database_echo, "pool_pre_ping": True}
    if settings.is_sqlite:
        # SQLite has no server to ping; pool_pre_ping is a no-op there anyway.
        kwargs.pop("pool_pre_ping")
    return create_async_engine(settings.database_url, **kwargs)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# Module-level defaults for application code. Tests create their own engine.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = make_session_factory(get_engine())
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session, rolling back on error."""
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db(engine: AsyncEngine | None = None) -> None:
    """Create all tables. Dev/SQLite convenience — prefer Alembic elsewhere."""
    from fxterminal.models.base import Base

    engine = engine or get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
