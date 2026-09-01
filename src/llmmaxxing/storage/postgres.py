"""Postgres (asyncpg) Control storage engine.

Publication transactions run at ``SERIALIZABLE`` and take
``pg_advisory_xact_lock(hashtext(installation_id))`` immediately after the
transaction opens, before any head compare-and-swap.  The advisory lock
serializes publishers of one installation (a conflicting publisher loses as
a plain :class:`StaleStateError`, never through retry); SERIALIZABLE still
backstops every other concurrent write anomaly.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from llmmaxxing.storage.migrations import sqlalchemy_async_url
from llmmaxxing.storage.repositories import SqlUnitOfWork


def _advisory_lock(installation_id: str) -> str:
    escaped = installation_id.replace("'", "''")
    return f"SELECT pg_advisory_xact_lock(hashtext('{escaped}'))"


class PostgresStorage:
    """Open one Control Postgres database and mint its units of work."""

    __slots__ = ("url", "installation_id", "engine")

    def __init__(self, url: str, *, installation_id: str) -> None:
        self.url = url
        self.installation_id = installation_id
        self.engine: AsyncEngine = create_async_engine(
            sqlalchemy_async_url(url),
            isolation_level="SERIALIZABLE",
        )

    def unit_of_work(self) -> SqlUnitOfWork:
        return SqlUnitOfWork(
            self.engine,
            _on_begin=(_advisory_lock(self.installation_id),),
        )

    async def dispose(self) -> None:
        await self.engine.dispose()
