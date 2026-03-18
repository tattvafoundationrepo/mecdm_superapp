"""Async SQLAlchemy engine for mecdm_superapp_user_db."""

import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_async_url() -> str | None:
    url = os.environ.get("DATABASE_URL_USER")
    if not url:
        return None
    # Convert postgresql:// to postgresql+asyncpg://
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


def get_user_db_engine():
    global _engine
    if _engine is None:
        async_url = _get_async_url()
        if not async_url:
            return None
        _engine = create_async_engine(async_url, pool_size=5, max_overflow=2)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    global _session_factory
    if _session_factory is None:
        engine = get_user_db_engine()
        if engine is None:
            return None
        _session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory
