"""Async Alembic environment for the Control database.

The calling code (``llmmaxxing.storage.migrations``) injects the async
database URL and the ``inject_failure`` interruption-test hook through
``config.attributes``; nothing checked in carries credentials.  All revisions
run over one async connection with ``transaction_per_migration``, so each
revision's DDL rolls back entirely on Postgres and the SQLite copy-and-swap
path never touches the live file until a verified copy exists.
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from llmmaxxing.storage import models as storage_models

config = context.config
target_metadata = storage_models.metadata


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table="alembic_version",
        render_as_batch=False,
        transaction_per_migration=True,
    )
    # ``inject_failure`` travels via Connection.info: revisions that do not
    # care keep a plain ``upgrade()`` signature, and 0002+ never have to
    # accept test-only keyword arguments.
    connection.info["inject_failure"] = bool(
        config.attributes.get("inject_failure", False)
    )
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        {"sqlalchemy.url": config.attributes["database_url"]},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    raise RuntimeError("offline (--sql) migrations are not supported for Control")
asyncio.run(run_async_migrations())
