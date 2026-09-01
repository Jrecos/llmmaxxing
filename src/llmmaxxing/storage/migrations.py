"""Explicit, refusal-first Control schema migration control.

Rules enforced here (the daemon never auto-migrates):

* ``migrate`` refuses a non-empty production database unless a *verified*
  backup receipt exists that was created after the last applied schema
  change, and refuses when the schema state is not a known revision;
* ``0001`` is expand-only; there is no reverse SQL anywhere in this package;
* ``ensure_schema_ready`` lets a previous-minor reader open an expanded
  schema (honouring ``min_reader_floor``) but refuses pending or newer
  floors, and never migrates;
* SQLite migrations run against a temporary copy which is integrity-checked,
  version-checked and row-loss-checked, and only then swapped over the
  original, so an interrupted migration can never damage the live file;
* Postgres migrations run inside Alembic's per-revision DDL transaction, so
  an interruption rolls the whole schema back.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from llmmaxxing.storage import models as m

HEAD_REVISION = "0001"
#: Lowest reader version allowed to open this schema (mirrored by 0001).
MIN_READER_FLOOR = "0.1.0"

#: Tables whose seeds/rows are not "production data" for the backup gate.
_NON_DATA_TABLES = frozenset({"alembic_version", "publications", "schema_meta"})


class StorageError(Exception):
    """Base class for migration control failures."""


class MigrationRefusedError(StorageError):
    """A migration was demanded but its preconditions are not met."""


class PendingMigrationError(StorageError):
    """The database is behind the code's head revision."""


class UnknownRevisionError(StorageError):
    """The database carries a revision this build does not know."""


def sqlalchemy_async_url(url: str) -> str:
    """Normalize a user-facing DSN into its SQLAlchemy async URL.

    Accepts ``postgres://``/``postgresql://`` (asyncpg), plain ``sqlite`` and
    already-async ``+aiosqlite``/``+asyncpg`` URLs.
    """
    if url.startswith("sqlite+aiosqlite://") or url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return f"postgresql+asyncpg://{url[len(prefix) :]}"
    raise ValueError(f"unsupported database URL scheme: {url!r}")


def _sync_url(async_url: str) -> str:
    if async_url.startswith("sqlite+aiosqlite://"):
        return async_url.replace("sqlite+aiosqlite://", "sqlite://", 1)
    if async_url.startswith("postgresql+asyncpg://"):
        return async_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    raise ValueError(f"cannot derive sync URL from {async_url!r}")


def _migrations_dir() -> Path:
    override = os.environ.get("LLMMAXXING_MIGRATIONS_DIR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "migrations"
        if (candidate / "env.py").is_file():
            return candidate
    raise MigrationRefusedError("Alembic migrations directory not found")


def _config(async_url: str, *, inject_failure: bool) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_migrations_dir()))
    config.attributes["database_url"] = async_url
    if inject_failure:
        config.attributes["inject_failure"] = True
    return config


@dataclass(frozen=True, slots=True)
class SchemaStatus:
    current_revision: str | None
    head_revision: str
    pending: bool
    installed: bool

    @property
    def at_head(self) -> bool:
        return self.installed and not self.pending and self.current_revision == self.head_revision


def schema_status(url: str) -> SchemaStatus:
    """Read ``alembic_version`` without ever writing schema state."""
    async_url = sqlalchemy_async_url(url)
    engine = sa.create_engine(_sync_url(async_url))
    try:
        if "alembic_version" not in sa.inspect(engine).get_table_names():
            return SchemaStatus(None, HEAD_REVISION, True, False)
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current = context.get_current_revision()
        return SchemaStatus(current, HEAD_REVISION, current != HEAD_REVISION, True)
    finally:
        engine.dispose()


def _table_names(sync_url: str) -> set[str]:
    engine = sa.create_engine(sync_url)
    try:
        return set(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _last_applied_change_ms(sync_url: str) -> int:
    """Timestamp anchor for backup freshness (0 when the schema is pristine)."""
    if "schema_meta" not in _table_names(sync_url):
        return 0
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                sa.text("SELECT value FROM schema_meta WHERE key='last_applied_change_ms'")
            ).first()
        return 0 if row is None else int(row[0])
    finally:
        engine.dispose()


def _has_production_rows(sync_url: str) -> bool:
    """True when any table beyond seeds/metadata holds a row."""
    engine = sa.create_engine(sync_url)
    try:
        tables = set(sa.inspect(engine).get_table_names()) - _NON_DATA_TABLES
        with engine.connect() as connection:
            for table in sorted(tables):
                count = connection.execute(
                    sa.text(f'SELECT COUNT(*) FROM "{table}"')
                ).scalar_one()
                if count:
                    return True
        return False
    finally:
        engine.dispose()


def _read_receipts(sync_url: str) -> list[tuple[str, int, int | None]]:
    if "backup_receipts" not in _table_names(sync_url):
        return []
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sa.text(
                    "SELECT verifier_status, created_at_ms, verified_at_ms FROM backup_receipts"
                )
            ).all()
        return [(str(r[0]), int(r[1]), None if r[2] is None else int(r[2])) for r in rows]
    finally:
        engine.dispose()


def assert_backup_precondition(
    sync_url: str,
    *,
    receipts: list[tuple[str, int, int | None]] | None = None,
) -> None:
    """Abort before any DDL unless a verified backup post-dates last change.

    ``receipts`` are ``(verifier_status, created_at_ms, verified_at_ms)``;
    omitted means read from the ``backup_receipts`` table if present.
    """
    if not _has_production_rows(sync_url):
        return
    last_change = _last_applied_change_ms(sync_url)
    if receipts is None:
        receipts = _read_receipts(sync_url)
    fresh = [
        row
        for row in receipts
        if row[0] == "verified"
        and row[1] >= last_change
        and row[2] is not None
        and row[2] >= row[1]
    ]
    if not fresh:
        raise MigrationRefusedError(
            "non-empty database requires a verified backup receipt created after "
            "the last applied schema change"
        )


def migrate_database(url: str, *, inject_failure: bool = False) -> SchemaStatus:
    """Apply ``0001`` (or the next pending revision) atomically.

    SQLite: copy → migrate copy → verify → swap; the original file is only
    ever replaced by a verified result.  Postgres: Alembic's per-revision DDL
    transaction rolls back entirely on failure.
    """
    async_url = sqlalchemy_async_url(url)
    logging.getLogger("alembic").setLevel(logging.CRITICAL)
    if async_url.startswith("sqlite"):
        path = Path(_sync_url(async_url).removeprefix("sqlite:///"))
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
        if str(path).endswith(":memory:") or "mode=memory" in str(path):
            raise MigrationRefusedError("in-memory SQLite cannot be migrated on disk")
        return _migrate_sqlite(async_url, path, inject_failure=inject_failure)
    return _migrate_postgres(async_url, inject_failure=inject_failure)


def _migrate_postgres(async_url: str, *, inject_failure: bool) -> SchemaStatus:
    sync_url = _sync_url(async_url)
    assert_backup_precondition(sync_url)
    command.upgrade(_config(async_url, inject_failure=inject_failure), "head")
    status = schema_status(async_url)
    if status.pending:
        raise MigrationRefusedError(f"schema not at head after migrate: {status}")
    return status


def _migrate_sqlite(async_url: str, path: Path, *, inject_failure: bool) -> SchemaStatus:
    original_bytes = path.read_bytes() if path.exists() else None
    if original_bytes is not None:
        assert_backup_precondition(_sync_url(async_url))
    temp_dir = tempfile.mkdtemp(prefix="llmmaxxing-migrate-")
    staged = Path(temp_dir) / "staged.db"
    final = f"sqlite:///{path}"
    try:
        if original_bytes is not None:
            staged.write_bytes(original_bytes)
        command.upgrade(
            _config(f"sqlite+aiosqlite:///{staged}", inject_failure=inject_failure),
            "head",
        )
        _verify_sqlite_copy(staged, original_bytes)
        if path.exists():
            path.unlink()
        shutil.move(str(staged), path)
        os.chmod(path, 0o600)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return schema_status(final)


def _verify_sqlite_copy(staged: Path, original_bytes: bytes | None) -> None:
    connection = sqlite3.connect(staged)
    try:
        ok = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise MigrationRefusedError("migrated SQLite copy fails integrity_check")
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        if version is None or version[0] != HEAD_REVISION:
            raise MigrationRefusedError("migrated copy is not at head")
        if original_bytes is not None:
            before = _row_counts(original_bytes)
            after = _row_counts(staged.read_bytes())
            for table, count in before.items():
                if after.get(table, 0) < count:
                    raise MigrationRefusedError(f"migrated copy lost rows in {table}")
    finally:
        connection.close()


def _row_counts(blob: bytes) -> dict[str, int]:
    temp_dir = tempfile.mkdtemp(prefix="llmmaxxing-verify-")
    probe = Path(temp_dir) / "probe.db"
    try:
        probe.write_bytes(blob)
        connection = sqlite3.connect(probe)
        try:
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            ]
            return {
                table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                for table in tables
            }
        finally:
            connection.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def ensure_schema_ready(
    url: str, *, reader_version: str, read_only: bool = False
) -> SchemaStatus:
    """Gate a process that opens the database against its code version.

    Refuses unknown revisions, pending migrations (even read-only) and
    minimum-reader floors above the caller; a previous-minor reader may
    open an expanded schema read-only.
    """
    status = schema_status(url)
    if status.current_revision is None:
        raise PendingMigrationError("database has no schema; migration required")
    known = {revision.revision for revision in ScriptDirectory(str(_migrations_dir())).walk_revisions()}
    if status.current_revision not in known:
        raise UnknownRevisionError(
            f"database revision {status.current_revision!r} is newer than this build"
        )
    if status.pending:
        raise PendingMigrationError(
            f"database at {status.current_revision!r} is behind head {HEAD_REVISION!r}"
        )
    floor = _min_reader_floor(url)
    if floor is not None and _version_tuple(reader_version) < _version_tuple(floor):
        raise MigrationRefusedError(
            f"reader {reader_version} below minimum reader floor {floor}"
        )
    return status


def _min_reader_floor(url: str) -> str | None:
    sync_url = _sync_url(sqlalchemy_async_url(url))
    if "schema_meta" not in _table_names(sync_url):
        return None
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                sa.text("SELECT value FROM schema_meta WHERE key='min_reader_floor'")
            ).first()
        return None if row is None else str(row[0])
    finally:
        engine.dispose()


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise MigrationRefusedError(f"malformed version {value!r}")
    return tuple(int(part) for part in parts)


def doctor(url: str) -> dict[str, Any]:
    """Non-mutating health report for ``llmmaxxing db doctor``."""
    async_url = sqlalchemy_async_url(url)
    sync_url = _sync_url(async_url)
    scheme = sync_url.split("://", 1)[0]
    report: dict[str, Any] = {"url_scheme": scheme}
    status = schema_status(url)
    report["revision"] = status.current_revision
    report["at_head"] = status.at_head
    if not status.installed:
        report["ok"] = False
        report["detail"] = "no schema applied"
        return report
    missing: list[str] = []
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as connection:
            if scheme == "sqlite":
                report["integrity"] = connection.execute(
                    sa.text("PRAGMA integrity_check")
                ).scalar_one()
                report["journal_mode"] = connection.execute(
                    sa.text("PRAGMA journal_mode")
                ).scalar_one()
            else:
                report["integrity"] = "n/a"
                report["server_version_num"] = connection.execute(
                    sa.text("SHOW server_version_num")
                ).scalar_one()
        existing = set(sa.inspect(engine).get_table_names())
        missing = sorted(set(m.metadata.tables) - existing)
    finally:
        engine.dispose()
    report["missing_tables"] = missing
    report["ok"] = (
        status.at_head and report.get("integrity") in {"ok", "n/a"} and not missing
    )
    return report


def digest_file(path: str | Path) -> str:
    """SHA-256 of a file; used by backup verification before adoption."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
