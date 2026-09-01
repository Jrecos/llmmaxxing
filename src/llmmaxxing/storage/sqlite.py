"""SQLite (aiosqlite) Control storage engine.

Pragmas: WAL, ``synchronous=FULL``, ``foreign_keys=ON``, ``busy_timeout=5000``.
Every write transaction opens with ``BEGIN IMMEDIATE`` so two publishers on
independent connections serialize at the database-file level; the file is
created ``0600`` because it holds key records and audit history.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from llmmaxxing.storage.repositories import SqlUnitOfWork

PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("synchronous", "FULL"),
    ("foreign_keys", "ON"),
    ("busy_timeout", "5000"),
)


def _apply_pragmas(dbapi_connection: sqlite3.Connection, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        for key, value in PRAGMAS:
            cursor.execute(f"PRAGMA {key}={value}")
    finally:
        cursor.close()


def _install_immediate_begin(engine: AsyncEngine) -> None:
    """Force ``BEGIN IMMEDIATE`` for every SQLAlchemy transaction.

    The pysqlite/aiosqlite dialect only ever emits bare ``BEGIN`` (deferred);
    listening on the engine ``begin`` event and issuing the driver-level
    statement first is the documented control for write serialization.
    """

    @sa.event.listens_for(engine.sync_engine, "begin")
    def _begin(conn: sa.Connection) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")


def _ensure_private_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(handle)
    os.chmod(path, 0o600)


class SQLiteStorage:
    """Open one Control SQLite database and mint its units of work."""

    __slots__ = ("path", "engine")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        _ensure_private_file(self.path)
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.path}",
            poolclass=sa.pool.NullPool,
        )
        sa.event.listen(engine.sync_engine, "connect", _apply_pragmas)
        _install_immediate_begin(engine)
        self.engine = engine

    def unit_of_work(self) -> SqlUnitOfWork:
        return SqlUnitOfWork(self.engine)

    async def dispose(self) -> None:
        await self.engine.dispose()
