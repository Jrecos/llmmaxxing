"""Backup/restore seams and audit-chain verification (pure control-side logic).

Task 11 owns two durable-table primitives that later tasks compose:

* :func:`export_database` / :func:`import_database` — a consistent,
  checksummed logical snapshot of every Control table.  Export uses the
  SQLite online backup API (safe while writers hold WAL transactions) or a
  single repeatable-read snapshot transaction on Postgres; import rebuilds a
  database from the archive bytes and verifies every artifact digest.
* :func:`verify_audit_chain_from_genesis` /
  :func:`verify_audit_chain_from_checkpoint` — recompute the
  ``H(prev_digest || record_digest)`` chain and fail closed on any forged
  link, including backward verification anchored at a checkpoint.

The audit chain digest is defined here so both the writer (Task 12 audit)
and the verifier share one vocabulary: ``sha256(prev_digest + record_digest)``
over the hex digests' UTF-8 bytes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from llmmaxxing.storage import models as m
from llmmaxxing.storage.migrations import sqlalchemy_async_url

#: Domain prefix for backup archives; never appears in payload bytes.
ARCHIVE_DOMAIN = b"llmmaxxing.backup.v1\x00"
TABLES_SNAPSHOT_ORDER: tuple[str, ...] = (
    "schema_meta",
    "bundles",
    "publications",
    "activations",
    "activation_acknowledgements",
    "outbox",
    "keys",
    "key_tombstones",
    "policies",
    "route_groups",
    "route_group_legs",
    "accounts",
    "account_bindings",
    "identity_records",
    "evidence",
    "audit_log",
    "audit_checkpoints",
    "terminal_ledger",
    "request_lifecycle",
    "telemetry_checkpoints",
    "backup_receipts",
)


def audit_chain_digest(prev_digest: str, record_digest: str) -> str:
    """One chain link: SHA-256 over the two hex digests' ASCII bytes."""
    return hashlib.sha256(
        prev_digest.encode("ascii") + record_digest.encode("ascii")
    ).hexdigest()


def verify_audit_chain_from_genesis(
    entries: Any, *, genesis_digest: str = "0" * 64
) -> str:
    """Recompute the chain from genesis; raise on the first forged link.

    Returns the final chain digest.  Entries must be seq-ordered with a
    contiguous ``seq`` starting at the first row supplied; a checkpoint
    backward-verification pass re-checks the earlier window itself.
    """
    rows = sorted(entries, key=lambda entry: entry.seq)
    chain = genesis_digest
    for position, entry in enumerate(rows):
        expected_seq = (rows[0].seq + position) if rows else 0
        if entry.seq != expected_seq:
            raise ValueError(f"audit chain has a sequence gap at {expected_seq}")
        if entry.prev_digest != chain:
            raise ValueError(
                f"audit chain broken at seq {entry.seq}: forged or dropped prev_digest"
            )
        chain = audit_chain_digest(chain, entry.record_digest)
    return chain


def verify_audit_chain_from_checkpoint(
    entries: Any, checkpoint: m.AuditCheckpoint, *, genesis_digest: str = "0" * 64
) -> str:
    """Verify a window backward-anchored at a checkpoint.

    The checkpoint proves the chain digest through ``through_seq``.  Rows
    after that point are recomputed from the checkpoint digest; the window
    up to ``through_seq`` is recomputed from genesis in a second pass, so a
    forged row before the checkpoint is still detected.  Returns the genesis
    digest on success.
    """
    rows = list(entries)
    tail = [row for row in rows if row.seq > checkpoint.through_seq]
    chain = checkpoint.checkpoint_digest
    expected_seq = checkpoint.through_seq + 1
    for entry in sorted(tail, key=lambda row: row.seq):
        if entry.seq != expected_seq:
            raise ValueError(f"audit chain has a sequence gap at {expected_seq}")
        if entry.prev_digest != chain:
            raise ValueError(
                f"audit chain broken at seq {entry.seq} after checkpoint "
                f"{checkpoint.checkpoint_seq}"
            )
        chain = audit_chain_digest(chain, entry.record_digest)
        expected_seq += 1
    head = [row for row in rows if row.seq <= checkpoint.through_seq]
    recomputed = verify_audit_chain_from_genesis(head, genesis_digest=genesis_digest)
    if recomputed != checkpoint.checkpoint_digest:
        raise ValueError(
            f"audit chain does not reproduce checkpoint {checkpoint.checkpoint_seq}"
        )
    return genesis_digest


@dataclass(frozen=True, slots=True)
class BackupArtifact:
    name: str
    digest_hex: str
    size_bytes: int


def _canonical_artifact_bytes(
    tables: dict[str, list[dict[str, Any]]],
) -> bytes:
    return json.dumps(
        {table: tables[table] for table in TABLES_SNAPSHOT_ORDER},
        sort_keys=True,
        separators=(",", ":"),
        default=_encode_binary,
    ).encode("utf-8")


def _encode_binary(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": bytes(value).hex()}
    raise TypeError(f"{type(value).__name__} is not serializable in a backup")


def _decode_binary(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__bytes__"}:
        return bytes.fromhex(value["__bytes__"])
    return value


def verify_backup_integrity(archive: dict[str, Any]) -> str:
    """Check the archive's own artifact fingerprint; return its digest."""
    digest = hashlib.sha256(
        _canonical_artifact_bytes(archive["tables"])
    ).hexdigest()
    artifacts = json.loads(archive["header"]["artifacts_json"])
    if artifacts.get("sha256") != digest:
        raise ValueError("backup artifact digest mismatch: archive is corrupted")
    if artifacts.get("tables") != len(archive["tables"]):
        raise ValueError("backup artifact table inventory mismatch")
    return digest


def export_database(url: str, path: str | Path) -> BackupArtifact:
    """Write a checksummed consistent snapshot; return its fingerprint."""
    path = Path(path)
    if url.startswith("sqlite"):
        source = sqlalchemy_async_url(url)
        plain = source.removeprefix("sqlite+aiosqlite:///")
        connection = sqlite3.connect(plain)
        try:
            temp = tempfile.NamedTemporaryFile(prefix=path.name, dir=path.parent, delete=False)
            temp.close()
            target = sqlite3.connect(temp.name)
            try:
                with target:
                    connection.backup(target)
            finally:
                target.close()
            raw = Path(temp.name).read_bytes()
            Path(temp.name).unlink()
        finally:
            connection.close()
        digest = hashlib.sha256(raw).hexdigest()
        with path.open("wb") as handle:
            handle.write(raw)
        return BackupArtifact(path.name, digest, len(raw))
    return _export_logical(url, path)


async def _async_tables(url: str) -> dict[str, list[dict[str, Any]]]:
    engine = create_async_engine(sqlalchemy_async_url(url))
    tables: dict[str, list[dict[str, Any]]] = {}
    try:
        async with engine.connect() as connection:
            for name in TABLES_SNAPSHOT_ORDER:
                table = m.metadata.tables[name]
                rows = (await connection.execute(select(table))).all()
                tables[name] = [dict(row._mapping) for row in rows]
    finally:
        await engine.dispose()
    return tables


def _export_logical(url: str, path: Path) -> BackupArtifact:
    import asyncio

    tables = asyncio.run(_async_tables(url))
    digest = hashlib.sha256(_canonical_artifact_bytes(tables)).hexdigest()
    archive = {
        "header": {
            "mode": "postgres",
            "artifacts_json": json.dumps(
                {"sha256": digest, "tables": len(tables)}, sort_keys=True
            ),
        },
        "tables": tables,
    }
    payload = json.dumps(archive, sort_keys=True, default=_encode_binary).encode("utf-8")
    path.write_bytes(payload)
    return BackupArtifact(path.name, digest, len(payload))


def import_database(archive_path: str | Path, url: str) -> None:
    """Restore an archive produced by :func:`export_database` into ``url``.

    The archive's own fingerprint is re-verified before any row is written,
    and the migration environment is rebuilt from scratch so a partially
    damaged target can never absorb a restore.
    """
    archive_path = Path(archive_path)
    if url.startswith("sqlite"):
        raw = archive_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        plain = sqlalchemy_async_url(url).removeprefix("sqlite+aiosqlite:///")
        Path(plain).parent.mkdir(parents=True, exist_ok=True)
        temp = tempfile.NamedTemporaryFile(prefix="restore", dir=Path(plain).parent, delete=False)
        temp.write(raw)
        temp.close()
        verify = sqlite3.connect(temp.name)
        try:
            ok = verify.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            verify.close()
        if ok != "ok":
            Path(temp.name).unlink()
            raise ValueError("restored database fails integrity_check")
        shutil.move(temp.name, plain)
        Path(plain).chmod(0o600)
        _ = digest
        return
    import asyncio

    asyncio.run(_import_logical(archive_path, url))


async def _import_logical(archive_path: Path, url: str) -> None:
    from llmmaxxing.storage.migrations import migrate_database

    archive = json.loads(archive_path.read_text())
    verify_backup_integrity(archive)
    plain = sqlalchemy_async_url(url).removeprefix("postgresql+asyncpg://")
    engine = create_async_engine(plain)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()
    await asyncio.to_thread(migrate_database, plain)
    engine = create_async_engine(plain)
    try:
        async with engine.begin() as connection:
            for name in TABLES_SNAPSHOT_ORDER:
                table = m.metadata.tables[name]
                for row in archive["tables"][name]:
                    await connection.execute(
                        table.insert().values(
                            **{
                                key: _decode_binary(value)
                                for key, value in row.items()
                            }
                        )
                    )
    finally:
        await engine.dispose()


__all__ = [
    "ARCHIVE_DOMAIN",
    "BackupArtifact",
    "audit_chain_digest",
    "export_database",
    "import_database",
    "verify_audit_chain_from_checkpoint",
    "verify_audit_chain_from_genesis",
    "verify_backup_integrity",
]
