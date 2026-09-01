from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from llmmaxxing.cli.main import build_parser, main
from llmmaxxing.storage.migrations import (
    HEAD_REVISION,
    MigrationRefusedError,
    PendingMigrationError,
    ensure_schema_ready,
    migrate_database,
    schema_status,
    sqlalchemy_async_url,
)

POSTGRES_DSN = os.environ.get("LLMMAXXING_TEST_POSTGRES")
EXPECTED_TABLES = {
    "accounts",
    "account_bindings",
    "activation_acknowledgements",
    "activations",
    "alembic_version",
    "audit_checkpoints",
    "audit_log",
    "backup_receipts",
    "bundles",
    "evidence",
    "identity_records",
    "key_tombstones",
    "keys",
    "outbox",
    "policies",
    "publications",
    "request_lifecycle",
    "route_group_legs",
    "route_groups",
    "schema_meta",
    "telemetry_checkpoints",
    "terminal_ledger",
}


@dataclass(frozen=True, slots=True)
class Column:
    table: str
    name: str
    kind: str
    not_null: bool
    primary_key_position: int


@dataclass(frozen=True, slots=True)
class Index:
    table: str
    columns: tuple[str, ...]
    unique: bool
    predicate: str | None


@dataclass(frozen=True, slots=True)
class ForeignKey:
    table: str
    column: str
    target_table: str
    target_column: str


@dataclass(frozen=True, slots=True)
class Schema:
    tables: frozenset[str]
    columns: frozenset[Column]
    indexes: frozenset[Index]
    foreign_keys: frozenset[ForeignKey]


def _kind(value: str) -> str:
    lowered = value.lower()
    if "int" in lowered:
        return "integer"
    if lowered in {"blob", "bytea"} or "binary" in lowered:
        return "binary"
    if "bool" in lowered:
        return "boolean"
    return "text"


def _sql(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().replace('"', "").replace("::text", "")
    normalized = normalized.replace("<>", "!=")
    normalized = re.sub(r"\s+", "", normalized)
    while normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1]
    return normalized


def _sqlite_expression(sql: str | None) -> str:
    if sql is None:
        return "<expression>"
    match = re.search(r"\((.*?)\)(?:\s+WHERE|$)", sql, re.IGNORECASE)
    return _sql(match.group(1) if match else "<expression>") or "<expression>"


async def _sqlite_schema(path: Path) -> Schema:
    connection = await aiosqlite.connect(path)
    try:
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {row[0] for row in await cursor.fetchall()}
        columns: set[Column] = set()
        indexes: set[Index] = set()
        foreign_keys: set[ForeignKey] = set()
        for table in tables:
            cursor = await connection.execute(f'PRAGMA table_info("{table}")')
            for _, name, kind, not_null, _, pk in await cursor.fetchall():
                columns.add(Column(table, name, _kind(kind), bool(not_null or pk), int(pk)))

            cursor = await connection.execute(f'PRAGMA index_list("{table}")')
            for _, index_name, unique, origin, partial in await cursor.fetchall():
                if origin == "pk":
                    continue
                sql_cursor = await connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (index_name,)
                )
                sql_row = await sql_cursor.fetchone()
                index_sql = None if sql_row is None else sql_row[0]
                info_cursor = await connection.execute(f'PRAGMA index_xinfo("{index_name}")')
                names = []
                for _, column_id, name, _, _, key in await info_cursor.fetchall():
                    if not key:
                        continue
                    names.append(name if column_id >= 0 else _sqlite_expression(index_sql))
                predicate = None
                if partial and index_sql:
                    predicate = _sql(index_sql.split(" WHERE ", 1)[1])
                indexes.add(Index(table, tuple(names), bool(unique), predicate))

            cursor = await connection.execute(f'PRAGMA foreign_key_list("{table}")')
            for _, _, target_table, source, target, *_ in await cursor.fetchall():
                foreign_keys.add(ForeignKey(table, source, target_table, target))
        return Schema(
            frozenset(tables),
            frozenset(columns),
            frozenset(indexes),
            frozenset(foreign_keys),
        )
    finally:
        await connection.close()


async def _postgres_schema(url: str) -> Schema:
    engine = create_async_engine(sqlalchemy_async_url(url))
    try:
        async with engine.connect() as connection:
            table_rows = (
                await connection.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='public' AND table_type='BASE TABLE'"
                    )
                )
            ).all()
            tables = {row[0] for row in table_rows}
            column_rows = (
                await connection.execute(
                    text(
                        "SELECT c.table_name,c.column_name,c.data_type,c.is_nullable,"
                        "COALESCE(k.ordinal_position,0) "
                        "FROM information_schema.columns c "
                        "LEFT JOIN ("
                        " SELECT ku.table_name,ku.column_name,ku.ordinal_position"
                        " FROM information_schema.table_constraints tc"
                        " JOIN information_schema.key_column_usage ku"
                        " ON tc.constraint_name=ku.constraint_name"
                        " AND tc.constraint_schema=ku.constraint_schema"
                        " WHERE tc.constraint_schema='public' AND tc.constraint_type='PRIMARY KEY'"
                        ") k ON k.table_name=c.table_name AND k.column_name=c.column_name "
                        "WHERE c.table_schema='public'"
                    )
                )
            ).all()
            columns = {
                Column(table, name, _kind(kind), nullable == "NO" or bool(pk), int(pk))
                for table, name, kind, nullable, pk in column_rows
            }
            index_rows = (
                await connection.execute(
                    text(
                        "SELECT tbl.relname,idx.relname,i.indisunique,i.indisprimary,"
                        "pg_get_expr(i.indpred,i.indrelid),"
                        "ARRAY(SELECT pg_get_indexdef(i.indexrelid,n,true) "
                        "FROM generate_series(1,i.indnkeyatts) n ORDER BY n) "
                        "FROM pg_index i "
                        "JOIN pg_class idx ON idx.oid=i.indexrelid "
                        "JOIN pg_class tbl ON tbl.oid=i.indrelid "
                        "JOIN pg_namespace ns ON ns.oid=tbl.relnamespace "
                        "WHERE ns.nspname='public'"
                    )
                )
            ).all()
            indexes = {
                Index(table, tuple(_sql(item) or "" for item in expressions), bool(unique), _sql(pred))
                for table, _, unique, primary, pred, expressions in index_rows
                if not primary
            }
            fk_rows = (
                await connection.execute(
                    text(
                        "SELECT tc.table_name,kcu.column_name,ccu.table_name,ccu.column_name "
                        "FROM information_schema.table_constraints tc "
                        "JOIN information_schema.key_column_usage kcu "
                        "ON tc.constraint_name=kcu.constraint_name "
                        "AND tc.constraint_schema=kcu.constraint_schema "
                        "JOIN information_schema.constraint_column_usage ccu "
                        "ON tc.constraint_name=ccu.constraint_name "
                        "AND tc.constraint_schema=ccu.constraint_schema "
                        "WHERE tc.constraint_schema='public' AND tc.constraint_type='FOREIGN KEY'"
                    )
                )
            ).all()
            foreign_keys = {ForeignKey(*row) for row in fk_rows}
        return Schema(
            frozenset(tables),
            frozenset(columns),
            frozenset(indexes),
            frozenset(foreign_keys),
        )
    finally:
        await engine.dispose()


async def _reset_postgres(url: str) -> None:
    engine = create_async_engine(sqlalchemy_async_url(url))
    try:
        async with engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_0001_schema_is_equivalent_on_sqlite_and_postgres16(tmp_path: Path) -> None:
    if POSTGRES_DSN is None:
        pytest.skip("LLMMAXXING_TEST_POSTGRES is required for schema equivalence")
    sqlite_path = tmp_path / "schema.db"
    sqlite_url = f"sqlite+aiosqlite:///{sqlite_path}"
    await _reset_postgres(POSTGRES_DSN)
    await asyncio.to_thread(migrate_database, sqlite_url)
    await asyncio.to_thread(migrate_database, POSTGRES_DSN)

    sqlite_schema, postgres_schema = await asyncio.gather(
        _sqlite_schema(sqlite_path), _postgres_schema(POSTGRES_DSN)
    )
    assert sqlite_schema.tables == EXPECTED_TABLES
    assert postgres_schema.tables == EXPECTED_TABLES
    assert sqlite_schema.columns == postgres_schema.columns
    assert sqlite_schema.indexes == postgres_schema.indexes
    assert sqlite_schema.foreign_keys == postgres_schema.foreign_keys
    assert schema_status(sqlite_url).current_revision == HEAD_REVISION
    assert schema_status(POSTGRES_DSN).current_revision == HEAD_REVISION


def test_sqlite_interrupted_copy_verify_swap_preserves_original_file(tmp_path: Path) -> None:
    path = tmp_path / "control.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE legacy_marker (value TEXT NOT NULL)")
    connection.commit()
    connection.close()
    before = path.read_bytes()
    before_digest = hashlib.sha256(before).hexdigest()

    with pytest.raises(RuntimeError, match="injected migration failure"):
        migrate_database(f"sqlite+aiosqlite:///{path}", inject_failure=True)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_digest
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall() == [("legacy_marker",)]
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_postgres_interrupted_ddl_transaction_rolls_back() -> None:
    if POSTGRES_DSN is None:
        pytest.skip("LLMMAXXING_TEST_POSTGRES is required for DDL rollback")
    await _reset_postgres(POSTGRES_DSN)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        await asyncio.to_thread(migrate_database, POSTGRES_DSN, inject_failure=True)
    schema = await _postgres_schema(POSTGRES_DSN)
    assert schema.tables == frozenset()


def test_pending_and_newer_reader_floors_refuse_but_previous_minor_reads_expansion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reader.db"
    url = f"sqlite+aiosqlite:///{path}"
    with pytest.raises(PendingMigrationError):
        ensure_schema_ready(url, reader_version="0.1.0", read_only=True)

    migrate_database(url)
    connection = sqlite3.connect(path)
    connection.execute("ALTER TABLE schema_meta ADD COLUMN additive_probe TEXT")
    connection.execute("UPDATE schema_meta SET value='0.0.0' WHERE key='min_reader_floor'")
    connection.commit()
    connection.close()
    ensure_schema_ready(url, reader_version="0.1.0", read_only=True)

    connection = sqlite3.connect(path)
    connection.execute("UPDATE schema_meta SET value='0.2.0' WHERE key='min_reader_floor'")
    connection.commit()
    connection.close()
    with pytest.raises(MigrationRefusedError, match="minimum reader"):
        ensure_schema_ready(url, reader_version="0.1.0", read_only=True)


def test_db_cli_is_explicit_and_refuses_nonempty_database_without_verified_backup(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    for action in ("migrate", "status", "doctor"):
        parsed = parser.parse_args(["db", action, "--database-url", "sqlite+aiosqlite:///:memory:"])
        assert parsed.command == "db"
        assert parsed.db_action == action

    path = tmp_path / "production.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE production_data (value TEXT NOT NULL)")
    connection.execute("INSERT INTO production_data VALUES ('keep-me')")
    connection.commit()
    connection.close()
    url = f"sqlite+aiosqlite:///{path}"

    assert main(["db", "migrate", "--database-url", url]) != 0
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT value FROM production_data").fetchall() == [("keep-me",)]
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='alembic_version'"
        ).fetchall() == []
    finally:
        connection.close()
