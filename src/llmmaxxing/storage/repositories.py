"""Engine-neutral Control repositories and the UnitOfWork transaction scope.

One ``UnitOfWork`` == one database transaction.  All mutation goes through
exactly three shapes:

* ``add``/``append`` — immutable inserts; a duplicate identity raises
  :class:`ConflictError` (the row is never silently replaced),
* ``compare_and_swap`` / ``cas_head`` — ``UPDATE ... WHERE <digest> = :expected``;
  a zero-row effect raises :class:`StaleStateError`,
* ``get``/``list_*`` — reads returning dataclasses.

Key-record mutation always runs the pure ``core.key_lifecycle`` delta
validation *before* the digest CAS, so terminal-tombstone, expiry-extension,
high-water and verifier-reuse rules are enforced by core, not duplicated in
SQL.  Digests (``record_digest``/``prev_digest``/...) are opaque lowercase
hex64 supplied by callers; storage validates shape, never content.

Repositories are retry-free: serialization conflicts surface as
:class:`StaleStateError` to the caller, and the transaction is never
replayed.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import Select, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection

from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.key_lifecycle import PolicyReassignment, validate_key_record_delta
from llmmaxxing.core.models import ClientKeyRecord
from llmmaxxing.storage import models as m


class StorageError(Exception):
    """Base for all Control storage failures."""


class ConflictError(StorageError):
    """An immutable insert collided with an existing identity."""


class StaleStateError(StorageError):
    """A compare-and-swap observed a row that changed since it was read."""


_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_OUTBOX_STATUSES = frozenset({"ready", "sent", "acked", "dead"})
_EVIDENCE_STATUSES = frozenset(
    {"pending", "fresh", "revalidation_due", "failed", "invalidated"}
)
_BACKUP_MODES = frozenset({"file", "sqlite", "postgres"})
_VERIFIER_STATUSES = frozenset({"pending", "verified", "failed"})
_ACTIVATION_STAGES = frozenset(
    {
        "preparing_backend",
        "backend_ready",
        "staging_gateway",
        "gateway_staged",
        "committing",
        "applied",
    }
)


def _require_hex64(value: str, label: str) -> str:
    if not _HEX64.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def _row_to_dataclass(row: Any, cls: type[Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{key: value for key, value in row.items() if key in names})


def canonical_record_digest(record: ClientKeyRecord) -> str:
    """Content digest of one key record: SHA-256 over its canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(record.model_dump(mode="json"))).hexdigest()


def _key_columns(record: ClientKeyRecord, updated_at_ms: int) -> dict[str, Any]:
    record_json = canonical_json_bytes(record.model_dump(mode="json")).decode("utf-8")
    return {
        "key_id": record.key_id,
        "policy_revision": str(record.policy_id),
        "record_json": record_json,
        "record_digest": hashlib.sha256(record_json.encode("utf-8")).hexdigest(),
        "state": str(record.state),
        "issued_at_s": record.issued_at_s,
        "expires_at_s": record.expires_at_s,
        "time_high_water_s": record.time_high_water_s,
        "generation_high_water": record.generation_high_water,
        "updated_at_ms": updated_at_ms,
    }


class Transaction:
    """Late-bound transaction shared by the repositories of one UnitOfWork.

    ``unit_of_work()`` is synchronous, but the engine connection only exists
    once ``begin()`` runs; repositories bind to this stable indirection.
    """

    __slots__ = ("connection",)

    def __init__(self) -> None:
        self.connection: AsyncConnection | None = None

    def require(self) -> AsyncConnection:
        if self.connection is None:
            raise StorageError("transaction is not open; use the unit-of-work scope")
        return self.connection

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        return await self.require().execute(*args, **kwargs)


class _BaseRepository:
    __slots__ = ("_tx",)

    def __init__(self, tx: Transaction) -> None:
        self._tx = tx

    async def _insert(self, table: Any, values: dict[str, Any]) -> None:
        try:
            await self._tx.execute(insert(table).values(**values))
        except IntegrityError as error:
            raise ConflictError(f"{table.name} row already exists") from error

    async def _cas(
        self,
        table: Any,
        key: dict[str, Any],
        expected_digest_column: str,
        expected_digest: str,
        replacement_values: dict[str, Any],
    ) -> None:
        _require_hex64(expected_digest, f"expected {expected_digest_column}")
        statement = update(table)
        for column, value in key.items():
            statement = statement.where(table.c[column] == value)
        statement = statement.where(table.c[expected_digest_column] == expected_digest)
        result = await self._tx.execute(statement.values(**replacement_values))
        if result.rowcount != 1:
            raise StaleStateError(f"{table.name} compare-and-swap lost")

    async def _get(self, table: Any, cls: type[Any], key: dict[str, Any]) -> Any:
        statement = select(table)
        for column, value in key.items():
            statement = statement.where(table.c[column] == value)
        row = (await self._tx.execute(statement)).first()
        return None if row is None else _row_to_dataclass(row._mapping, cls)

    async def _list(self, statement: Select[Any], cls: type[Any]) -> list[Any]:
        rows = (await self._tx.execute(statement)).all()
        return [_row_to_dataclass(row._mapping, cls) for row in rows]


class PublicationRepository(_BaseRepository):
    """Bundle store, activation pipeline, singleton-head CAS, command outbox."""

    async def add_bundle(self, bundle: m.Bundle) -> None:
        if len(bundle.canonical_bytes) > 16 * 1024 * 1024:
            raise ValueError("bundle canonical bytes exceed the 16 MiB ceiling")
        await self._insert(m.bundles, m.bundle_row(bundle))

    async def get_bundle(self, generation: int) -> m.Bundle | None:
        return await self._get(m.bundles, m.Bundle, {"generation": generation})

    async def list_bundles(self, *, limit: int = 1000) -> list[m.Bundle]:
        statement = select(m.bundles).order_by(m.bundles.c.generation).limit(limit)
        return await self._list(statement, m.Bundle)

    async def add_activation(self, activation: m.Activation) -> None:
        if activation.stage not in _ACTIVATION_STAGES:
            raise ValueError(f"unknown activation stage {activation.stage!r}")
        await self._insert(m.activations, m.activation_row(activation))

    async def get_activation(self, activation_id: str) -> m.Activation | None:
        return await self._get(
            m.activations, m.Activation, {"activation_id": activation_id}
        )

    async def advance_activation_stage(
        self,
        activation_id: str,
        expected_stage_record_digest: str,
        replacement: m.Activation,
    ) -> m.Activation:
        if replacement.activation_id != activation_id:
            raise ValueError("activation identity cannot change")
        if replacement.stage not in _ACTIVATION_STAGES:
            raise ValueError(f"unknown activation stage {replacement.stage!r}")
        await self._cas(
            m.activations,
            {"activation_id": activation_id},
            "stage_record_digest",
            expected_stage_record_digest,
            m.activation_update_row(replacement),
        )
        loaded = await self.get_activation(activation_id)
        assert loaded is not None
        return loaded

    async def get_head(self) -> m.PublicationHead | None:
        return await self._get(m.publications, m.PublicationHead, {"id": 1})

    async def cas_head(
        self,
        expected: m.PublicationHead | None,
        replacement: m.PublicationHead,
    ) -> m.PublicationHead:
        """Serialized singleton-head swap; exactly one concurrent writer wins.

        SQLite holds ``BEGIN IMMEDIATE`` for the whole transaction; Postgres
        holds ``pg_advisory_xact_lock(hashtext(installation_id))`` taken when
        the transaction begins, so the losing publisher observes a plainly
        stale CAS and is never retried.
        """
        if replacement.id != 1:
            raise ValueError("publication head id must be 1")
        if expected is None:
            raise StaleStateError("publication head was read as missing")
        result = await self._tx.execute(
            update(m.publications)
            .where(m.publications.c.id == 1)
            .where(m.publications.c.applied_generation == expected.applied_generation)
            .where(m.publications.c.applied_bundle_hash == expected.applied_bundle_hash)
            .where(m.publications.c.dispatcher_fence == expected.dispatcher_fence)
            .values(
                applied_generation=replacement.applied_generation,
                applied_bundle_hash=replacement.applied_bundle_hash,
                dispatcher_fence=replacement.dispatcher_fence,
                updated_at_ms=replacement.updated_at_ms,
            )
        )
        if result.rowcount != 1:
            raise StaleStateError("publication head compare-and-swap lost")
        return replacement

    async def enqueue_outbox(self, item: m.OutboxItem) -> m.OutboxItem:
        """Idempotent enqueue keyed by ``dedupe_key``.

        A replay of the same logical command returns the row stored by the
        first enqueue regardless of its sequence number.
        """
        if item.status not in _OUTBOX_STATUSES:
            raise ValueError(f"unknown outbox status {item.status!r}")
        existing = await self._get(m.outbox, m.OutboxItem, {"dedupe_key": item.dedupe_key})
        if existing is not None:
            return existing
        try:
            await self._insert(m.outbox, m.outbox_row(item))
        except ConflictError:
            raced = await self._get(
                m.outbox, m.OutboxItem, {"dedupe_key": item.dedupe_key}
            )
            assert raced is not None
            return raced
        return item

    async def get_outbox(self, dedupe_key: str) -> m.OutboxItem | None:
        return await self._get(m.outbox, m.OutboxItem, {"dedupe_key": dedupe_key})


class IdentityRepository(_BaseRepository):
    """Per-installation identity roots, CAS-only mutation."""

    async def add(self, record: m.IdentityRecord) -> None:
        _require_hex64(record.identity_root_digest, "identity_root_digest")
        await self._insert(m.identity_records, m.identity_row(record))

    async def get(self, installation_id: str) -> m.IdentityRecord | None:
        return await self._get(
            m.identity_records, m.IdentityRecord, {"installation_id": installation_id}
        )

    async def compare_and_swap(
        self,
        installation_id: str,
        expected_record_digest: str,
        replacement: m.IdentityRecord,
    ) -> m.IdentityRecord:
        if replacement.installation_id != installation_id:
            raise ValueError("identity installation cannot change")
        _require_hex64(replacement.identity_root_digest, "identity_root_digest")
        await self._cas(
            m.identity_records,
            {"installation_id": installation_id},
            "identity_root_digest",
            expected_record_digest,
            {
                "channel_key_epoch": replacement.channel_key_epoch,
                "security_epoch": replacement.security_epoch,
                "identity_root_digest": replacement.identity_root_digest,
                "updated_at_ms": replacement.updated_at_ms,
            },
        )
        return replacement


class KeyRepository(_BaseRepository):
    """Client-key records and their irreversible tombstones."""

    async def add(self, record: ClientKeyRecord, *, updated_at_ms: int = 0) -> None:
        await self._insert(m.keys, _key_columns(record, updated_at_ms))

    async def get(self, key_id: str) -> ClientKeyRecord | None:
        row = (
            await self._tx.execute(
                select(m.keys.c.record_json).where(m.keys.c.key_id == key_id)
            )
        ).first()
        return None if row is None else ClientKeyRecord.model_validate_json(row.record_json)

    async def list_keys(self, *, limit: int = 1000) -> list[ClientKeyRecord]:
        rows = (
            await self._tx.execute(
                select(m.keys.c.record_json).order_by(m.keys.c.key_id).limit(limit)
            )
        ).all()
        return [ClientKeyRecord.model_validate_json(row.record_json) for row in rows]

    async def update_record(
        self,
        key_id: str,
        before: ClientKeyRecord,
        after: ClientKeyRecord,
        *,
        policy_reassignment: PolicyReassignment | None = None,
        updated_at_ms: int = 0,
    ) -> ClientKeyRecord:
        """Core-validated delta first, then digest CAS against the stored row.

        ``before`` is caller-supplied (what it believed it read); the digest
        CAS still compares against the *stored* row, so a record mutated
        between load and update loses here exactly like every other CAS.
        """
        validate_key_record_delta(before, after, policy_reassignment=policy_reassignment)
        if before.key_id != key_id or after.key_id != key_id:
            raise ValueError("client key identity cannot change")
        await self._cas(
            m.keys,
            {"key_id": key_id},
            "record_digest",
            canonical_record_digest(before),
            _key_columns(after, updated_at_ms),
        )
        return after

    async def append_tombstone(self, tombstone: m.KeyTombstone) -> None:
        _require_hex64(tombstone.terminal_record_digest, "terminal_record_digest")
        await self._insert(m.key_tombstones, m.tombstone_row(tombstone))

    async def get_tombstone(self, key_id: str) -> m.KeyTombstone | None:
        return await self._get(m.key_tombstones, m.KeyTombstone, {"key_id": key_id})


class EvidenceRepository(_BaseRepository):
    """Provider-onboarding evidence rows."""

    async def add(self, record: m.EvidenceRecord) -> None:
        if record.status not in _EVIDENCE_STATUSES:
            raise ValueError(f"unknown evidence status {record.status!r}")
        _require_hex64(record.artifact_digest, "artifact_digest")
        await self._insert(m.evidence, m.evidence_row(record))

    async def get(self, evidence_id: str) -> m.EvidenceRecord | None:
        return await self._get(m.evidence, m.EvidenceRecord, {"evidence_id": evidence_id})

    async def list_evidence(
        self,
        *,
        account_id: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[m.EvidenceRecord]:
        statement = select(m.evidence).order_by(m.evidence.c.expires_at_ms).limit(limit)
        if account_id is not None:
            statement = statement.where(m.evidence.c.account_id == account_id)
        if status is not None:
            statement = statement.where(m.evidence.c.status == status)
        return await self._list(statement, m.EvidenceRecord)


class AuditRepository(_BaseRepository):
    """Append-only audit chain with checkpoints and the terminal ledger."""

    async def append(
        self,
        *,
        occurred_at_ms: int,
        actor_principal_id: str,
        action: str,
        subject_type: str,
        subject_id: str,
        record_digest: str,
        prev_digest: str,
    ) -> m.AuditLog:
        _require_hex64(record_digest, "record_digest")
        _require_hex64(prev_digest, "prev_digest")
        seq = (
            await self._tx.execute(
                insert(m.audit_log)
                .values(
                    occurred_at_ms=occurred_at_ms,
                    actor_principal_id=actor_principal_id,
                    action=action,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    record_digest=record_digest,
                    prev_digest=prev_digest,
                )
                .returning(m.audit_log.c.seq)
            )
        ).scalar_one()
        return m.AuditLog(
            seq=int(seq),
            occurred_at_ms=occurred_at_ms,
            actor_principal_id=actor_principal_id,
            action=action,
            subject_type=subject_type,
            subject_id=subject_id,
            record_digest=record_digest,
            prev_digest=prev_digest,
        )

    async def add_checkpoint(self, checkpoint: m.AuditCheckpoint) -> None:
        _require_hex64(checkpoint.checkpoint_digest, "checkpoint_digest")
        await self._insert(m.audit_checkpoints, m.checkpoint_row(checkpoint))

    async def get_checkpoint(self, checkpoint_seq: int) -> m.AuditCheckpoint | None:
        return await self._get(
            m.audit_checkpoints, m.AuditCheckpoint, {"checkpoint_seq": checkpoint_seq}
        )

    async def list_entries(
        self, *, first_seq: int | None = None, last_seq: int | None = None
    ) -> list[m.AuditLog]:
        statement = select(m.audit_log).order_by(m.audit_log.c.seq)
        if first_seq is not None:
            statement = statement.where(m.audit_log.c.seq >= first_seq)
        if last_seq is not None:
            statement = statement.where(m.audit_log.c.seq <= last_seq)
        return await self._list(statement, m.AuditLog)

    async def append_ledger(self, entry: m.TerminalLedgerEntry) -> None:
        _require_hex64(entry.tombstone_digest, "tombstone_digest")
        _require_hex64(entry.prev_digest, "prev_digest")
        await self._insert(m.terminal_ledger, m.ledger_row(entry))

    async def latest_ledger_seq(self) -> int:
        value = (
            await self._tx.execute(
                select(m.terminal_ledger.c.seq)
                .order_by(m.terminal_ledger.c.seq.desc())
                .limit(1)
            )
        ).scalar()
        return 0 if value is None else int(value)


class TelemetryRepository(_BaseRepository):
    """Request-lifecycle ingestion with checkpoint-replay idempotency."""

    async def ingest(
        self,
        checkpoint: m.TelemetryCheckpoint,
        requests: Sequence[m.RequestLifecycle],
    ) -> bool:
        """Store one telemetry batch; ``False`` means already ingested."""
        _require_hex64(checkpoint.ingest_digest, "ingest_digest")
        try:
            await self._insert(
                m.telemetry_checkpoints, m.telemetry_checkpoint_row(checkpoint)
            )
        except ConflictError:
            return False
        for record in requests:
            await self._insert(m.request_lifecycle, m.request_row(record))
        return True

    async def list_requests(
        self,
        *,
        key_id: str | None = None,
        route_group_id: str | None = None,
        terminal_outcome: str | None = None,
        admitted_after_ms: int | None = None,
        admitted_before_ms: int | None = None,
        limit: int = 1000,
    ) -> list[m.RequestLifecycle]:
        statement = (
            select(m.request_lifecycle)
            .order_by(
                m.request_lifecycle.c.admitted_at_ms, m.request_lifecycle.c.request_id
            )
            .limit(limit)
        )
        if key_id is not None:
            statement = statement.where(m.request_lifecycle.c.key_id == key_id)
        if route_group_id is not None:
            statement = statement.where(
                m.request_lifecycle.c.route_group_id == route_group_id
            )
        if terminal_outcome is not None:
            statement = statement.where(
                m.request_lifecycle.c.terminal_outcome == terminal_outcome
            )
        if admitted_after_ms is not None:
            statement = statement.where(
                m.request_lifecycle.c.admitted_at_ms >= admitted_after_ms
            )
        if admitted_before_ms is not None:
            statement = statement.where(
                m.request_lifecycle.c.admitted_at_ms <= admitted_before_ms
            )
        return await self._list(statement, m.RequestLifecycle)

    async def list_checkpoints(self, *, limit: int = 1000) -> list[m.TelemetryCheckpoint]:
        statement = (
            select(m.telemetry_checkpoints)
            .order_by(m.telemetry_checkpoints.c.at_ms)
            .limit(limit)
        )
        return await self._list(statement, m.TelemetryCheckpoint)


class BackupReceiptRepository(_BaseRepository):
    """Verified-backup receipts gating every destructive migration."""

    async def add(self, receipt: m.BackupReceipt) -> None:
        if receipt.mode not in _BACKUP_MODES:
            raise ValueError(f"unknown backup mode {receipt.mode!r}")
        if receipt.verifier_status not in _VERIFIER_STATUSES:
            raise ValueError(f"unknown verifier status {receipt.verifier_status!r}")
        await self._insert(m.backup_receipts, m.backup_row(receipt))

    async def get(self, receipt_id: str) -> m.BackupReceipt | None:
        return await self._get(
            m.backup_receipts, m.BackupReceipt, {"receipt_id": receipt_id}
        )

    async def mark_verified(self, receipt_id: str, verified_at_ms: int) -> m.BackupReceipt:
        result = await self._tx.execute(
            update(m.backup_receipts)
            .where(m.backup_receipts.c.receipt_id == receipt_id)
            .where(m.backup_receipts.c.verifier_status == "pending")
            .values(verifier_status="verified", verified_at_ms=verified_at_ms)
        )
        if result.rowcount != 1:
            raise StaleStateError("backup receipt not pending or missing")
        loaded = await self.get(receipt_id)
        assert loaded is not None
        return loaded

    async def latest_verified(
        self, *, created_after_ms: int | None = None
    ) -> m.BackupReceipt | None:
        statement = select(m.backup_receipts).where(
            m.backup_receipts.c.verifier_status == "verified"
        )
        if created_after_ms is not None:
            statement = statement.where(m.backup_receipts.c.created_at_ms > created_after_ms)
        statement = statement.order_by(m.backup_receipts.c.created_at_ms.desc()).limit(1)
        rows = await self._list(statement, m.BackupReceipt)
        return rows[0] if rows else None


@runtime_checkable
class UnitOfWork(Protocol):
    """One Control transaction across every repository."""

    publications: PublicationRepository
    identities: IdentityRepository
    keys: KeyRepository
    evidence: EvidenceRepository
    audit: AuditRepository
    telemetry: TelemetryRepository
    backups: BackupReceiptRepository

    async def begin(self) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def __aenter__(self) -> "UnitOfWork": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None: ...


@dataclass(slots=True)
class SqlUnitOfWork:
    """Concrete engine-neutral UnitOfWork over one AsyncConnection.

    The engine hands in ``in_transaction`` (how to open the transaction —
    SQLite ``BEGIN IMMEDIATE`` via the dialect's begin hook, Postgres
    ``SERIALIZABLE``) and ``on_begin`` statements (Postgres advisory lock),
    which run inside the opened transaction before any repository call.
    """

    _engine: Any
    _on_begin: tuple[str, ...] = ()
    _txn: Transaction = field(default_factory=Transaction)
    _conn: AsyncConnection | None = None
    publications: PublicationRepository | None = None
    identities: IdentityRepository | None = None
    keys: KeyRepository | None = None
    evidence: EvidenceRepository | None = None
    audit: AuditRepository | None = None
    telemetry: TelemetryRepository | None = None
    backups: BackupReceiptRepository | None = None

    async def begin(self) -> None:
        assert self.publications is None, "unit of work already begun"
        connection = await self._engine.connect()
        self._conn = connection
        await connection.begin()
        for statement in self._on_begin:
            await connection.execute(text(statement))
        tx = self._txn
        tx.connection = connection
        self.publications = PublicationRepository(tx)
        self.identities = IdentityRepository(tx)
        self.keys = KeyRepository(tx)
        self.evidence = EvidenceRepository(tx)
        self.audit = AuditRepository(tx)
        self.telemetry = TelemetryRepository(tx)
        self.backups = BackupReceiptRepository(tx)

    async def commit(self) -> None:
        assert self._conn is not None
        await self._conn.commit()
        await self._close_connection()

    async def rollback(self) -> None:
        assert self._conn is not None
        await self._conn.rollback()
        await self._close_connection()

    async def _close_connection(self) -> None:
        assert self._conn is not None
        await self._conn.close()
        self._conn = None
        self._txn.connection = None

    async def __aenter__(self) -> "SqlUnitOfWork":
        await self.begin()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()


__all__ = [
    "AuditRepository",
    "BackupReceiptRepository",
    "ConflictError",
    "EvidenceRepository",
    "IdentityRepository",
    "KeyRepository",
    "PublicationRepository",
    "SqlUnitOfWork",
    "StaleStateError",
    "StorageError",
    "TelemetryRepository",
    "Transaction",
    "UnitOfWork",
    "canonical_record_digest",
]
