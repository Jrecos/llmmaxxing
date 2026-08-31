"""Authenticated activation, durable local state, deny overlays and dispatcher fencing."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, Self, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from llmmaxxing.core.canonical import bundle_hash, canonical_bundle_bytes, canonical_json_bytes
from llmmaxxing.core.ids import AckId, AuthLeaseId, BundleHash, CommandId, GatewayBootId, InstallationId
from llmmaxxing.core.models import AuthorizedLeg, ClientKeyRecord, PolicyBundleV1
from llmmaxxing.core.wire import (
    ActivationEnvelope,
    AuthLeaseV1,
    BaseReference,
    BundleReference,
    ChannelSigner,
    ChannelTrustSet,
    ClearDenyCommandPayload,
    ChannelSealV1,
    CommandChainGap,
    DenyCommandPayload,
    DenyOverlayV1,
    DenyReason,
    DenySubjectType,
    DuplicateCommand,
    FenceReceiptPayloadV1,
    FenceReceiptV1,
    GatewayAckStatus,
    GatewayAckV1,
    GatewayCommandV1,
    GatewayLifecycle,
    GatewayReadiness,
    GatewayStatusV1,
    PrepareCommandPayload,
    ReadinessReason,
    StatusCommandPayload,
    StaleFenceEpoch,
    TakeoverState,
    VerifiedGatewayCommand,
    WireCommandKind,
    seal_gateway_ack,
    seal_fence_receipt,
    verify_fence_receipt,
    verify_gateway_command,
)
from llmmaxxing.gateway.auth import AuthRuntimeView, LegacyKeyIndexEntry, build_legacy_key_index
from llmmaxxing.gateway.routing import GenerationOperationalGate

_SCHEMA_VERSION = 1
_ZERO_DIGEST = "0" * 64
_DENY_FRESH_MS = 10_000
_AUTH_LEASE_MS = 15 * 60 * 1000


class Clock(Protocol):
    def now_ms(self) -> int: ...


class _SystemClock:
    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000


class BundleLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_bundle_bytes: int = Field(default=16 * 1024 * 1024, ge=1, le=16 * 1024 * 1024)
    max_keys: int = Field(default=100_000, ge=1, le=100_000)
    max_policies: int = Field(default=100_000, ge=1, le=100_000)
    max_accounts: int = Field(default=4_096, ge=1, le=4_096)
    max_route_groups: int = Field(default=100_000, ge=1, le=100_000)
    max_legs: int = Field(default=100_000, ge=1, le=100_000)
    max_authorized_legs: int = Field(default=1_000_000, ge=1, le=1_000_000)


class DispatcherGate:
    """One short process-wide critical section shared by dispatch and activation."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def hold_dispatch(self, _request_id: object) -> AsyncIterator[None]:
        async with self._lock:
            yield

    @asynccontextmanager
    async def hold_activation(self) -> AsyncIterator[None]:
        async with self._lock:
            yield


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    command_id: CommandId
    digest: str
    ack_id: AckId
    status: GatewayAckStatus
    result: dict[str, JsonValue]
    ack: GatewayAckV1 | None


@dataclass(frozen=True, slots=True)
class _AuthView(AuthRuntimeView):
    key_index: Mapping[str, ClientKeyRecord]
    applied_bundle_generation: int
    applied_bundle_hash: BundleHash
    denied_key_ids: frozenset[str]
    accepted_peppers: Mapping[str, bytes]
    legacy_key_index: Mapping[str, LegacyKeyIndexEntry]
    trusted_now_s: int


def _canonical(value: object) -> str:
    return canonical_json_bytes(value).decode()


def _checksum(table: str, value: object) -> str:
    return hashlib.sha256(
        b"llmmaxxing.gateway-state.v1\0" + table.encode() + b"\0" + canonical_json_bytes(value)
    ).hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(fd, content[offset:])
        if written <= 0:
            raise OSError("short durable write")
        offset += written


class GatewayLocalState:
    """SQLite FULL state plus immutable content-addressed canonical bundle files."""

    def __init__(
        self,
        path: str | Path,
        *,
        installation_id: InstallationId,
        boot_id: GatewayBootId,
        channel_epoch: int,
        security_epoch: int,
        accepted_peppers: Mapping[str, bytes],
        clock: Clock | None = None,
        crash_injector: Callable[[str], None] | None = None,
        limits: BundleLimits | None = None,
    ) -> None:
        if channel_epoch < 1 or security_epoch < 1:
            raise ValueError("channel and security epochs must be positive")
        if not accepted_peppers:
            raise ValueError("client-key pepper trust must be injected")
        self.path = Path(path)
        self.installation_id = installation_id
        self.boot_id = boot_id
        self.channel_epoch = channel_epoch
        self.security_epoch = security_epoch
        self.accepted_peppers = MappingProxyType(dict(accepted_peppers))
        self.clock = clock or _SystemClock()
        self.limits = limits or BundleLimits()
        self._crash = crash_injector or (lambda _point: None)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._active: BundleReference | None = None
        self._staged: BundleReference | None = None
        self._previous: BundleReference | None = None
        self._base: BaseReference | None = None
        self._active_bundle: PolicyBundleV1 | None = None
        self._denies: dict[tuple[DenySubjectType, str], DenyOverlayV1] = {}
        self._deny_heartbeat_ms = 0
        self._recovery_required = False
        self._takeover_state = TakeoverState.NONE
        self._fence_epoch = 1
        self._closed = False

    @property
    def db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Gateway local state is not open")
        return self._conn

    def crash(self, point: str) -> None:
        self._crash(point)

    @property
    def bundles_path(self) -> Path:
        return self.path / "bundles"

    @property
    def pointer_path(self) -> Path:
        return self.path / "active.pointer.json"

    def bundle_path(self, value: BundleHash) -> Path:
        return self.bundles_path / f"{value}.json"

    @property
    def takeover_state(self) -> TakeoverState:
        return self._takeover_state

    @property
    def fence_epoch(self) -> int:
        return self._fence_epoch

    @property
    def lifecycle(self) -> GatewayLifecycle:
        if self._recovery_required:
            return GatewayLifecycle.RECOVERY_REQUIRED
        if self._takeover_state is TakeoverState.FENCED_OLD:
            return GatewayLifecycle.FENCED_OLD
        if self._active is not None:
            return GatewayLifecycle.APPLIED
        if self._staged is not None:
            return GatewayLifecycle.STAGED
        return GatewayLifecycle.NONE

    @property
    def active_reference(self) -> BundleReference | None:
        return self._active

    @property
    def staged_reference(self) -> BundleReference | None:
        return self._staged

    def staged(self) -> BundleReference | None:
        return self._staged

    @property
    def previous_reference(self) -> BundleReference | None:
        return self._previous

    @property
    def active_bundle(self) -> PolicyBundleV1 | None:
        return self._active_bundle

    @property
    def command_count(self) -> int:
        row = self.db.execute("SELECT count(*) FROM commands").fetchone()
        assert row is not None
        return int(row[0])

    def open(self) -> Self:
        if self._conn is not None:
            return self
        if self.path.is_symlink():
            raise RuntimeError("Gateway state directory may not be a symlink")
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path, 0o700)
        self.bundles_path.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.bundles_path, 0o700)
        for temporary in self.bundles_path.glob(".tmp-*"):
            with contextlib.suppress(OSError):
                temporary.unlink()
        for temporary in self.path.glob(".active-*"):
            with contextlib.suppress(OSError):
                temporary.unlink()
        conn = sqlite3.connect(
            self.path / "gateway-state.sqlite3",
            isolation_level=None,
            check_same_thread=False,
            timeout=30,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA fullfsync=ON")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA wal_autocheckpoint=100")
        self._conn = conn
        self._create_schema()
        self._load_identity()
        self._recover()
        return self

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL,checksum TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS bundles(bundle_hash TEXT PRIMARY KEY,generation INTEGER NOT NULL,size INTEGER NOT NULL,checksum TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS pointers(name TEXT PRIMARY KEY,generation INTEGER NOT NULL,bundle_hash TEXT NOT NULL,checksum TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS commands(command_id TEXT PRIMARY KEY,digest TEXT NOT NULL,channel_epoch INTEGER NOT NULL,sequence INTEGER NOT NULL,ack_id TEXT NOT NULL,status TEXT NOT NULL,result_json TEXT NOT NULL,ack_json TEXT,checksum TEXT NOT NULL) STRICT;
            CREATE UNIQUE INDEX IF NOT EXISTS commands_channel_sequence ON commands(channel_epoch,sequence);
            CREATE TABLE IF NOT EXISTS channel_state(channel_epoch INTEGER PRIMARY KEY,sequence INTEGER NOT NULL,digest TEXT NOT NULL,checksum TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS denies(subject_type TEXT NOT NULL,subject_id TEXT NOT NULL,deny_epoch INTEGER NOT NULL,reason TEXT NOT NULL,floor_generation INTEGER,heartbeat_at_ms INTEGER NOT NULL,checksum TEXT NOT NULL,PRIMARY KEY(subject_type,subject_id)) STRICT;
            CREATE TABLE IF NOT EXISTS auth_leases(lease_id TEXT PRIMARY KEY,lease_json TEXT NOT NULL,checksum TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS fence_receipts(receipt_digest TEXT PRIMARY KEY,receipt_json TEXT NOT NULL,checksum TEXT NOT NULL) STRICT;
            """
        )

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            else:
                self.db.execute("COMMIT")

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value,checksum FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value = str(row["value"])
        if row["checksum"] != _checksum("meta", {"key": key, "value": value}):
            raise ValueError("corrupt Gateway meta")
        return value

    def _set_meta(self, key: str, value: str) -> None:
        checksum = _checksum("meta", {"key": key, "value": value})
        self.db.execute(
            "INSERT INTO meta VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,checksum=excluded.checksum",
            (key, value, checksum),
        )

    def _load_identity(self) -> None:
        with self.transaction():
            schema = self._meta("schema_version")
            if schema is None:
                for key, value in (
                    ("schema_version", str(_SCHEMA_VERSION)),
                    ("installation_id", str(self.installation_id)),
                    ("channel_epoch", str(self.channel_epoch)),
                    ("security_epoch", str(self.security_epoch)),
                    ("fence_epoch", "1"),
                    ("takeover_state", TakeoverState.NONE.value),
                    ("deny_heartbeat_ms", "0"),
                ):
                    self._set_meta(key, value)
            else:
                if int(schema) != _SCHEMA_VERSION:
                    raise RuntimeError("unsupported Gateway state schema")
                expected = {
                    "installation_id": str(self.installation_id),
                    "channel_epoch": str(self.channel_epoch),
                    "security_epoch": str(self.security_epoch),
                }
                if any(self._meta(key) != value for key, value in expected.items()):
                    raise RuntimeError("Gateway state identity/epoch mismatch")
            self._fence_epoch = int(self._meta("fence_epoch") or 1)
            self._takeover_state = TakeoverState(self._meta("takeover_state") or "none")
            self._deny_heartbeat_ms = int(self._meta("deny_heartbeat_ms") or 0)

    @staticmethod
    def _pointer_record(name: str, ref: BundleReference | BaseReference) -> dict[str, object]:
        return {"name": name, "generation": ref.generation, "bundle_hash": str(ref.bundle_hash)}

    def _put_pointer(self, name: str, ref: BundleReference | BaseReference | None) -> None:
        if ref is None:
            self.db.execute("DELETE FROM pointers WHERE name=?", (name,))
            return
        record = self._pointer_record(name, ref)
        self.db.execute(
            "INSERT INTO pointers VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET generation=excluded.generation,bundle_hash=excluded.bundle_hash,checksum=excluded.checksum",
            (name, ref.generation, str(ref.bundle_hash), _checksum("pointers", record)),
        )

    def _read_pointer(self, name: str, *, base: bool = False) -> BundleReference | BaseReference | None:
        row = self.db.execute("SELECT * FROM pointers WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        record = {"name": name, "generation": int(row["generation"]), "bundle_hash": str(row["bundle_hash"])}
        if row["checksum"] != _checksum("pointers", record):
            raise ValueError("corrupt bundle pointer")
        if base:
            return BaseReference(
                generation=int(cast(Any, record["generation"])),
                bundle_hash=BundleHash(str(record["bundle_hash"])),
            )
        return BundleReference(
            generation=int(cast(Any, record["generation"])),
            bundle_hash=BundleHash(str(record["bundle_hash"])),
        )

    def _recover(self) -> None:
        try:
            check = self.db.execute("PRAGMA quick_check").fetchone()
            if check is None or check[0] != "ok":
                raise ValueError("SQLite integrity failure")
            self._previous = cast(BundleReference | None, self._read_pointer("previous"))
            self._active = cast(BundleReference | None, self._read_pointer("active"))
            self._staged = cast(BundleReference | None, self._read_pointer("staged"))
            self._base = cast(BaseReference | None, self._read_pointer("base", base=True))
            if self._previous is not None:
                self.load_bundle(self._previous.bundle_hash)
            if self._active is not None:
                self._active_bundle = self.load_bundle(self._active.bundle_hash)
                if self._active_bundle.generation != self._active.generation:
                    raise ValueError("active generation mismatch")
            if self._staged is not None:
                try:
                    staged = self.load_bundle(self._staged.bundle_hash)
                    if staged.generation != self._staged.generation:
                        raise ValueError("staged generation mismatch")
                except (OSError, ValueError):
                    with self.transaction():
                        self._put_pointer("staged", None)
                        self._put_pointer("base", None)
                    self._staged = None
                    self._base = None
            self._load_denies()
            self._quarantine_corrupt_commands()
            self._reconcile_pointer()
        except (OSError, sqlite3.DatabaseError, ValueError):
            self._active = None
            self._active_bundle = None
            self._recovery_required = True

    def _reconcile_pointer(self) -> None:
        if self._active is None:
            if self.pointer_path.exists():
                self.pointer_path.unlink()
                _fsync_dir(self.path)
            return
        expected = canonical_json_bytes(self._active.model_dump(mode="json"))
        actual = self.pointer_path.read_bytes() if self.pointer_path.exists() else b""
        if actual != expected:
            self._write_active_pointer(expected, inject=False)

    def _load_denies(self) -> None:
        loaded: dict[tuple[DenySubjectType, str], DenyOverlayV1] = {}
        for row in self.db.execute("SELECT * FROM denies"):
            record = {
                "subject_type": str(row["subject_type"]),
                "subject_id": str(row["subject_id"]),
                "deny_epoch": int(row["deny_epoch"]),
                "reason": str(row["reason"]),
                "floor_generation": None if row["floor_generation"] is None else int(row["floor_generation"]),
                "heartbeat_at_ms": int(row["heartbeat_at_ms"]),
            }
            if row["checksum"] != _checksum("denies", record):
                raise ValueError("corrupt deny overlay")
            overlay = DenyOverlayV1(
                subject_type=DenySubjectType(str(record["subject_type"])),
                subject_id=str(record["subject_id"]),
                deny_epoch=int(cast(Any, record["deny_epoch"])),
                reason=DenyReason(str(record["reason"])),
                deny_floor_generation=cast(int | None, record["floor_generation"]),
                heartbeat_at_ms=int(cast(Any, record["heartbeat_at_ms"])),
            )
            loaded[(overlay.subject_type, overlay.subject_id)] = overlay
        self._denies = loaded

    @staticmethod
    def _command_values(row: sqlite3.Row | Mapping[str, object]) -> dict[str, object]:
        return {
            "command_id": str(row["command_id"]), "digest": str(row["digest"]),
            "channel_epoch": int(cast(Any, row["channel_epoch"])), "sequence": int(cast(Any, row["sequence"])),
            "ack_id": str(row["ack_id"]), "status": str(row["status"]),
            "result_json": str(row["result_json"]),
            "ack_json": None if row["ack_json"] is None else str(row["ack_json"]),
        }

    def _decode_command(self, row: sqlite3.Row) -> _CommandRecord:
        values = self._command_values(row)
        if row["checksum"] != _checksum("commands", values):
            raise ValueError("corrupt command row")
        result = json.loads(str(row["result_json"]))
        if not isinstance(result, dict):
            raise ValueError("invalid command result")
        ack = None if row["ack_json"] is None else GatewayAckV1.model_validate(json.loads(str(row["ack_json"])))
        return _CommandRecord(
            CommandId(str(row["command_id"])), str(row["digest"]), AckId(str(row["ack_id"])),
            GatewayAckStatus(str(row["status"])), cast(dict[str, JsonValue], result), ack,
        )

    def _quarantine_corrupt_commands(self) -> None:
        corrupt: list[str] = []
        for row in self.db.execute("SELECT * FROM commands"):
            try:
                self._decode_command(row)
            except (ValueError, ValidationError):
                corrupt.append(str(row["command_id"]))
        if corrupt:
            with self.transaction():
                self.db.executemany("DELETE FROM commands WHERE command_id=?", ((x,) for x in corrupt))

    def store_bundle(self, payload: bytes, expected: BundleHash) -> Path:
        if len(payload) > self.limits.max_bundle_bytes:
            raise ValueError("bundle size exceeds the configured limit")
        if bundle_hash(payload) != expected:
            raise ValueError("bundle content hash differs from target")
        destination = self.bundle_path(expected)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError("content-addressed path contains different bytes")
            return destination
        fd, name = tempfile.mkstemp(prefix=".tmp-", dir=self.bundles_path)
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, payload)
            self.crash("bundle_after_write")
            os.fsync(fd)
            self.crash("bundle_after_fsync")
        finally:
            os.close(fd)
        os.replace(name, destination)
        self.crash("bundle_after_rename")
        _fsync_dir(self.bundles_path)
        self.crash("bundle_after_dir_fsync")
        return destination

    def load_bundle(self, expected: BundleHash) -> PolicyBundleV1:
        payload = self.bundle_path(expected).read_bytes()
        if len(payload) > self.limits.max_bundle_bytes or bundle_hash(payload) != expected:
            raise ValueError("stored bundle is corrupt")
        try:
            bundle = PolicyBundleV1.model_validate(json.loads(payload))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("stored bundle is invalid") from error
        if canonical_bundle_bytes(bundle) != payload:
            raise ValueError("stored bundle is not canonical")
        return bundle

    def validate_new_command(self, verified: VerifiedGatewayCommand) -> _CommandRecord | None:
        row = self.db.execute("SELECT * FROM commands WHERE command_id=?", (str(verified.command.command_id),)).fetchone()
        if row is not None:
            record = self._decode_command(row)
            if record.digest != verified.command_digest:
                raise DuplicateCommand("command ID reused with different bytes")
            return record
        state = self.db.execute("SELECT * FROM channel_state WHERE channel_epoch=?", (verified.command.channel_epoch,)).fetchone()
        sequence, digest = 1, _ZERO_DIGEST
        if state is not None:
            values = {"channel_epoch": verified.command.channel_epoch, "sequence": int(state["sequence"]), "digest": str(state["digest"])}
            if state["checksum"] != _checksum("channel_state", values):
                raise ValueError("corrupt command chain")
            sequence = int(cast(Any, values["sequence"])) + 1
            digest = str(values["digest"])
        if verified.command.sequence != sequence or verified.command.previous_digest != digest:
            raise CommandChainGap("command is not next in the digest chain")
        return None

    def ack_for(self, command_id: CommandId) -> GatewayAckV1 | None:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM commands WHERE command_id=?",
                (str(command_id),),
            ).fetchone()
            if row is None:
                return None
            return self._decode_command(row).ack

    def record_command(
        self,
        verified: VerifiedGatewayCommand,
        *,
        ack_id: AckId,
        status: GatewayAckStatus,
        result: Mapping[str, JsonValue],
    ) -> None:
        with self.transaction():
            self._record_pending(verified, ack_id, status, result)

    def _record_pending(
        self, verified: VerifiedGatewayCommand, ack_id: AckId,
        status: GatewayAckStatus, result: Mapping[str, JsonValue],
    ) -> None:
        if self.validate_new_command(verified) is not None:
            raise DuplicateCommand("command already recorded")
        command = verified.command
        result_json = _canonical(dict(result))
        values = {
            "command_id": str(command.command_id), "digest": verified.command_digest,
            "channel_epoch": command.channel_epoch, "sequence": command.sequence,
            "ack_id": str(ack_id), "status": status.value, "result_json": result_json, "ack_json": None,
        }
        self.db.execute(
            "INSERT INTO commands VALUES(?,?,?,?,?,?,?,NULL,?)",
            (*tuple(values[key] for key in ("command_id", "digest", "channel_epoch", "sequence", "ack_id", "status", "result_json")), _checksum("commands", values)),
        )
        chain = {"channel_epoch": command.channel_epoch, "sequence": command.sequence, "digest": verified.command_digest}
        self.db.execute(
            "INSERT INTO channel_state VALUES(?,?,?,?) ON CONFLICT(channel_epoch) DO UPDATE SET sequence=excluded.sequence,digest=excluded.digest,checksum=excluded.checksum",
            (command.channel_epoch, command.sequence, verified.command_digest, _checksum("channel_state", chain)),
        )

    def push_ack(self, ack: GatewayAckV1) -> None:
        with self.transaction():
            row = self.db.execute("SELECT * FROM commands WHERE command_id=?", (str(ack.command_id),)).fetchone()
            if row is None:
                raise ValueError("unknown command acknowledgement")
            record = self._decode_command(row)
            if record.digest != ack.command_digest or record.ack_id != ack.ack_id:
                raise ValueError("ack differs from pending command")
            if record.ack is not None:
                if record.ack != ack:
                    raise ValueError("different ACK already exists")
                return
            ack_json = _canonical(ack.model_dump(mode="json"))
            values = self._command_values(row)
            values["ack_json"] = ack_json
            self.db.execute(
                "UPDATE commands SET ack_json=?,checksum=? WHERE command_id=?",
                (ack_json, _checksum("commands", values), str(ack.command_id)),
            )
        self.crash("ack_after_commit")

    @property
    def deny_floor_generation(self) -> int | None:
        return max((x.deny_floor_generation for x in self._denies.values() if x.deny_floor_generation is not None), default=None)

    def validate_prepare_reference(
        self,
        base: BaseReference | None,
        ref: BundleReference,
    ) -> None:
        current = (
            None
            if self._active is None
            else BaseReference(
                generation=self._active.generation,
                bundle_hash=self._active.bundle_hash,
            )
        )
        if current != base:
            raise ValueError("activation base differs from active bundle")
        if self._staged is not None:
            raise ValueError("another activation is staged")
        floor = self.deny_floor_generation
        if floor is not None and ref.generation <= floor:
            raise ValueError("target generation does not exceed deny floor")
        if base is not None and ref.generation <= base.generation:
            raise ValueError("target generation must increase")

    def stage_prepare(
        self, verified: VerifiedGatewayCommand, bundle: PolicyBundleV1,
        ref: BundleReference, base: BaseReference | None,
        *, ack_id: AckId, result: Mapping[str, JsonValue],
    ) -> None:
        self.validate_prepare_reference(base, ref)
        self.crash("staged_before_commit")
        with self.transaction():
            record = {"bundle_hash": str(ref.bundle_hash), "generation": ref.generation, "size": len(canonical_bundle_bytes(bundle))}
            self.db.execute("INSERT OR IGNORE INTO bundles VALUES(?,?,?,?)", (*record.values(), _checksum("bundles", record)))
            self._put_pointer("staged", ref)
            self._put_pointer("base", base)
            self._record_pending(verified, ack_id, GatewayAckStatus.PREPARED, result)
        self._staged, self._base = ref, base
        self.crash("staged_after_commit")

    def _write_active_pointer(self, payload: bytes, *, inject: bool = True) -> None:
        fd, name = tempfile.mkstemp(prefix=".active-", dir=self.path)
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, payload)
            if inject: self.crash("active_pointer_after_write")
            os.fsync(fd)
            if inject: self.crash("active_pointer_after_fsync")
        finally:
            os.close(fd)
        os.replace(name, self.pointer_path)
        if inject: self.crash("active_pointer_after_rename")
        _fsync_dir(self.path)
        if inject: self.crash("active_pointer_after_dir_fsync")

    def commit_staged(
        self, verified: VerifiedGatewayCommand, target: BundleReference,
        *, ack_id: AckId, result: Mapping[str, JsonValue],
    ) -> PolicyBundleV1:
        if self._staged != target:
            raise ValueError("commit target differs from staged bundle")
        bundle = self.load_bundle(target.bundle_hash)
        self._write_active_pointer(canonical_json_bytes(target.model_dump(mode="json")))
        with self.transaction():
            self._put_pointer("previous", self._active)
            self._put_pointer("active", target)
            self._put_pointer("staged", None)
            self._put_pointer("base", None)
            self._record_pending(verified, ack_id, GatewayAckStatus.APPLIED, result)
        self._previous, self._active, self._active_bundle = self._active, target, bundle
        self._staged = self._base = None
        self.crash("active_after_commit")
        return bundle

    def apply_genesis(
        self,
        verified: VerifiedGatewayCommand,
        target: BundleReference,
        *,
        ack_id: AckId,
        result: Mapping[str, JsonValue],
    ) -> PolicyBundleV1:
        if self._active is not None or self._base is not None:
            raise ValueError("genesis requires no active record and an absent base")
        return self.commit_staged(
            verified,
            target,
            ack_id=ack_id,
            result=result,
        )

    def _verify_deny_store(self) -> None:
        try:
            self._load_denies()
        except (sqlite3.DatabaseError, ValueError):
            self._recovery_required = True
            raise

    def issue_deny(
        self, verified: VerifiedGatewayCommand, payload: DenyCommandPayload,
        *, ack_id: AckId, result: Mapping[str, JsonValue],
    ) -> None:
        key = (payload.subject_type, payload.subject_id)
        self._verify_deny_store()
        existing = self._denies.get(key)
        if existing is not None and payload.deny_epoch <= existing.deny_epoch:
            raise ValueError("deny epoch must strictly increase")
        now = self.clock.now_ms()
        overlay = DenyOverlayV1(**payload.model_dump(mode="python"), heartbeat_at_ms=now)
        record = {
            "subject_type": overlay.subject_type.value, "subject_id": overlay.subject_id,
            "deny_epoch": overlay.deny_epoch, "reason": overlay.reason.value,
            "floor_generation": overlay.deny_floor_generation, "heartbeat_at_ms": now,
        }
        self.crash("deny_before_commit")
        with self.transaction():
            self.db.execute(
                "INSERT INTO denies VALUES(?,?,?,?,?,?,?) ON CONFLICT(subject_type,subject_id) DO UPDATE SET deny_epoch=excluded.deny_epoch,reason=excluded.reason,floor_generation=excluded.floor_generation,heartbeat_at_ms=excluded.heartbeat_at_ms,checksum=excluded.checksum",
                (*record.values(), _checksum("denies", record)),
            )
            self._set_meta("deny_heartbeat_ms", str(now))
            self._record_pending(verified, ack_id, GatewayAckStatus.DENIED, result)
        self._denies[key], self._deny_heartbeat_ms = overlay, now
        self.crash("deny_after_commit")

    def clear_deny(
        self, verified: VerifiedGatewayCommand, payload: ClearDenyCommandPayload,
        *, ack_id: AckId, result: Mapping[str, JsonValue],
    ) -> None:
        key = (payload.subject_type, payload.subject_id)
        self._verify_deny_store()
        existing = self._denies.get(key)
        if existing is None or existing.deny_epoch != payload.deny_epoch:
            raise ValueError("clear-deny requires exact current epoch")
        now = self.clock.now_ms()
        with self.transaction():
            self.db.execute("DELETE FROM denies WHERE subject_type=? AND subject_id=?", (payload.subject_type.value, payload.subject_id))
            self._set_meta("deny_heartbeat_ms", str(now))
            self._record_pending(verified, ack_id, GatewayAckStatus.DENY_CLEARED, result)
        del self._denies[key]
        self._deny_heartbeat_ms = now

    def renew_deny_heartbeats(self, now_ms: int | None = None) -> None:
        now = self.clock.now_ms() if now_ms is None else now_ms
        if now < 1:
            raise ValueError("heartbeat time must be positive")
        updated: dict[tuple[DenySubjectType, str], DenyOverlayV1] = {}
        try:
            with self.transaction():
                self._load_denies()
                for key, overlay in self._denies.items():
                    fresh = overlay.model_copy(update={"heartbeat_at_ms": now})
                    record = {
                        "subject_type": fresh.subject_type.value,
                        "subject_id": fresh.subject_id,
                        "deny_epoch": fresh.deny_epoch,
                        "reason": fresh.reason.value,
                        "floor_generation": fresh.deny_floor_generation,
                        "heartbeat_at_ms": now,
                    }
                    self.db.execute(
                        "UPDATE denies SET heartbeat_at_ms=?,checksum=? "
                        "WHERE subject_type=? AND subject_id=?",
                        (
                            now,
                            _checksum("denies", record),
                            fresh.subject_type.value,
                            fresh.subject_id,
                        ),
                    )
                    updated[key] = fresh
                self._set_meta("deny_heartbeat_ms", str(now))
        except (sqlite3.DatabaseError, ValueError):
            self._recovery_required = True
            raise
        self._denies, self._deny_heartbeat_ms = updated, now

    def _heartbeat_fresh(self, now: int) -> bool:
        return 0 <= now - self._deny_heartbeat_ms <= _DENY_FRESH_MS

    @property
    def deny_all(self) -> bool:
        return self._recovery_required or self._takeover_state is TakeoverState.FENCED_OLD or not self._heartbeat_fresh(self.clock.now_ms())

    def permits(self, leg: AuthorizedLeg, route_group_id: object | None = None) -> bool:
        if route_group_id is None and self._active_bundle is not None:
            route_group_id = next(
                (
                    group.route_group_id
                    for group in self._active_bundle.route_groups
                    if any(candidate.leg_id == leg.leg_id for candidate in group.legs)
                ),
                None,
            )
        return (
            route_group_id is not None
            and not self.deny_all
            and not any(
                key in self._denies
                for key in (
                    (DenySubjectType.ACCOUNT, str(leg.account_id)),
                    (DenySubjectType.LEG, str(leg.leg_id)),
                    (DenySubjectType.ROUTE_GROUP, str(route_group_id)),
                )
            )
        )

    def issue_auth_lease(self, now: int) -> AuthLeaseV1:
        if self._active is None or self.deny_all:
            raise RuntimeError("healthy active state is required for auth lease")
        lease = AuthLeaseV1(
            lease_id=AuthLeaseId.new(), installation_id=self.installation_id,
            security_epoch=self.security_epoch, bundle=self._active,
            issued_at_ms=now, expires_at_ms=now + _AUTH_LEASE_MS,
        )
        document = lease.model_dump(mode="json")
        self.db.execute("INSERT INTO auth_leases VALUES(?,?,?)", (str(lease.lease_id), _canonical(document), _checksum("auth_leases", document)))
        return lease

    def _current_lease(self, now: int) -> AuthLeaseV1 | None:
        if self._active is None:
            return None
        for row in self.db.execute("SELECT * FROM auth_leases ORDER BY rowid DESC"):
            try:
                raw = json.loads(str(row["lease_json"]))
                if row["checksum"] != _checksum("auth_leases", raw): continue
                lease = AuthLeaseV1.model_validate(raw)
            except (ValueError, ValidationError):
                continue
            if lease.bundle == self._active and lease.security_epoch == self.security_epoch and lease.issued_at_ms <= now < lease.expires_at_ms:
                return lease
        return None

    def record_status(
        self, verified: VerifiedGatewayCommand, *, ack_id: AckId,
        issue_lease: bool, result_factory: Callable[[AuthLeaseV1 | None], Mapping[str, JsonValue]],
    ) -> None:
        with self.transaction():
            lease = self.issue_auth_lease(self.clock.now_ms()) if issue_lease else None
            self._record_pending(verified, ack_id, GatewayAckStatus.STATUS, result_factory(lease))

    def current_auth_view(self) -> AuthRuntimeView:
        if self._active_bundle is None or self._active is None:
            raise RuntimeError("no verified active auth state")
        keys = {record.key_id: record for record in self._active_bundle.keys}
        return _AuthView(
            MappingProxyType(keys), self._active.generation, self._active.bundle_hash,
            frozenset(keys) if self.deny_all else frozenset(), self.accepted_peppers,
            MappingProxyType(build_legacy_key_index(self._active_bundle.keys)), self.clock.now_ms() // 1000,
        )

    def status(
        self,
        *,
        singleton_held: bool,
        backend_ready: bool,
        capacities_ready: bool,
        now_ms: int | None = None,
    ) -> GatewayStatusV1:
        with self._lock:
            return self._status_unlocked(
                singleton_held=singleton_held,
                backend_ready=backend_ready,
                capacities_ready=capacities_ready,
                now_ms=now_ms,
            )

    def _status_unlocked(
        self, *, singleton_held: bool, backend_ready: bool, capacities_ready: bool,
        now_ms: int | None = None,
    ) -> GatewayStatusV1:
        now = self.clock.now_ms() if now_ms is None else now_ms
        fresh = self._heartbeat_fresh(now) and not self._recovery_required
        lease = self._current_lease(now)
        reasons: list[ReadinessReason] = []
        if not singleton_held: reasons.append(ReadinessReason.SINGLETON)
        if self._recovery_required: reasons.append(ReadinessReason.RECOVERY)
        elif self._takeover_state is TakeoverState.FENCED_OLD: reasons.append(ReadinessReason.FENCED)
        elif self._active is None: reasons.append(ReadinessReason.NO_ACTIVE)
        if not fresh: reasons.append(ReadinessReason.DENY_STALE)
        if lease is None: reasons.append(ReadinessReason.AUTH_LEASE)
        if not backend_ready: reasons.append(ReadinessReason.BACKEND)
        if not capacities_ready: reasons.append(ReadinessReason.CAPACITIES)
        return GatewayStatusV1(
            installation_id=self.installation_id, channel_epoch=self.channel_epoch,
            security_epoch=self.security_epoch, boot_id=self.boot_id,
            singleton_held=singleton_held, lifecycle=self.lifecycle,
            active=self._active, staged=self._staged, previous=self._previous, base=self._base,
            deny_overlay=tuple(sorted(self._denies.values(), key=lambda x: (x.subject_type.value, x.subject_id))),
            deny_floor_generation=self.deny_floor_generation, deny_heartbeat_fresh=fresh,
            auth_lease=lease, dispatcher_fence=self._fence_epoch, takeover_state=self._takeover_state,
            readiness=GatewayReadiness.READY if not reasons else GatewayReadiness.UNREADY,
            unready_reasons=tuple(reasons),
        )

    @staticmethod
    def _receipt_digest(receipt: FenceReceiptV1) -> str:
        return hashlib.sha256(
            canonical_json_bytes(receipt.model_dump(mode="json"))
        ).hexdigest()

    def _store_receipt(self, receipt: FenceReceiptV1) -> None:
        document = receipt.model_dump(mode="json")
        digest = self._receipt_digest(receipt)
        self.db.execute(
            "INSERT OR IGNORE INTO fence_receipts VALUES(?,?,?)",
            (digest, _canonical(document), _checksum("fence_receipts", document)),
        )

    def record_old_fence(self, receipt: FenceReceiptV1) -> None:
        payload = receipt.payload
        if payload.old_installation_id != self.installation_id:
            raise ValueError("fence receipt does not name this old installation")
        if payload.fence_epoch <= self._fence_epoch:
            raise ValueError("fence epoch must strictly advance")
        with self.transaction():
            self._store_receipt(receipt)
            self._set_meta("fence_epoch", str(payload.fence_epoch))
            self._set_meta("takeover_state", TakeoverState.FENCED_OLD.value)
        self._fence_epoch = payload.fence_epoch
        self._takeover_state = TakeoverState.FENCED_OLD
        self.crash("old_fence_after_commit")

    def record_fence_receipt(self, receipt: FenceReceiptV1) -> None:
        with self.transaction():
            self._store_receipt(receipt)

    def accept_takeover(self, receipt: FenceReceiptV1) -> None:
        payload = receipt.payload
        if payload.target_installation_id != self.installation_id:
            raise ValueError("fence receipt targets another installation")
        if payload.fence_epoch <= self._fence_epoch:
            raise ValueError("fence receipt does not advance dispatcher fence")
        with self.transaction():
            self._store_receipt(receipt)
            self._set_meta("takeover_state", TakeoverState.FENCED_OLD.value)
        self._takeover_state = TakeoverState.FENCED_OLD
        self.crash("takeover_after_receipt")
        with self.transaction():
            self._set_meta("fence_epoch", str(payload.fence_epoch))
            self._set_meta("takeover_state", TakeoverState.RELEASED.value)
        self._fence_epoch = payload.fence_epoch
        self._takeover_state = TakeoverState.RELEASED
        self.crash("takeover_after_advance")

    def fence_receipts(self) -> tuple[FenceReceiptV1, ...]:
        receipts: list[FenceReceiptV1] = []
        for row in self.db.execute(
            "SELECT receipt_json,checksum FROM fence_receipts ORDER BY receipt_digest"
        ):
            document = json.loads(str(row["receipt_json"]))
            if row["checksum"] != _checksum("fence_receipts", document):
                raise ValueError("corrupt fence receipt")
            receipts.append(FenceReceiptV1.model_validate(document))
        return tuple(receipts)

    def export_state(self) -> dict[str, JsonValue]:
        with self._lock:
            return self._export_state_unlocked()

    def _export_state_unlocked(self) -> dict[str, JsonValue]:
        def dump(ref: BaseModel | None) -> JsonValue:
            return None if ref is None else cast(JsonValue, ref.model_dump(mode="json"))
        return {
            "schema_version": _SCHEMA_VERSION, "installation_id": str(self.installation_id),
            "channel_epoch": self.channel_epoch, "security_epoch": self.security_epoch,
            "dispatcher_fence": self._fence_epoch, "takeover_state": self._takeover_state.value,
            "active": dump(self._active), "staged": dump(self._staged),
            "previous": dump(self._previous), "base": dump(self._base),
            "denies": cast(JsonValue, [x.model_dump(mode="json") for x in self._denies.values()]),
            "deny_heartbeat_ms": self._deny_heartbeat_ms, "command_count": self.command_count,
            "recovery_required": self._recovery_required,
        }

    def checkpoint(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(FULL)")
        _fsync_dir(self.path)

    def gc_bundles(self) -> None:
        retained = {x.bundle_hash for x in (self._active, self._staged, self._previous) if x is not None}
        floor = self.deny_floor_generation
        if floor is not None:
            retained.update(BundleHash(str(x[0])) for x in self.db.execute("SELECT bundle_hash FROM bundles WHERE generation=?", (floor,)))
        for path in self.bundles_path.glob("bh_*.json"):
            with contextlib.suppress(ValueError):
                if BundleHash(path.stem) not in retained: path.unlink()
        _fsync_dir(self.bundles_path)

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        if self._conn is not None:
            with contextlib.suppress(sqlite3.DatabaseError): self.checkpoint()
            self._conn.close()
            self._conn = None




class TakeoverCoordinator:
    """Old-host fence receipt emission and new-host monotonic acceptance."""

    def __init__(
        self,
        state: GatewayLocalState,
        signer: ChannelSigner,
        old_channel_trust: ChannelTrustSet,
        dispatcher_gate: DispatcherGate,
    ) -> None:
        self._state = state
        self._signer = signer
        self._old_channel_trust = old_channel_trust
        self._dispatcher_gate = dispatcher_gate

    async def fence_old(
        self,
        *,
        target_installation_id: InstallationId,
        credential_digest: str,
        network_digest: str,
        fenced_at_ms: int,
    ) -> FenceReceiptV1:
        async with self._dispatcher_gate.hold_activation():
            unsigned = FenceReceiptV1(
                payload=FenceReceiptPayloadV1(
                    old_installation_id=self._state.installation_id,
                    target_installation_id=target_installation_id,
                    channel_epoch=self._state.channel_epoch,
                    security_epoch=self._state.security_epoch,
                    old_boot_id=self._state.boot_id,
                    credential_digest=credential_digest,
                    network_digest=network_digest,
                    fence_epoch=self._state.fence_epoch + 1,
                    fenced_at_ms=fenced_at_ms,
                ),
                channel_seal=ChannelSealV1(
                    seal_id=self._signer.seal_id,
                    trust_epoch=self._signer.trust_epoch,
                    signature="0" * 128,
                ),
            )
            receipt = seal_fence_receipt(unsigned, self._signer)
            verify_fence_receipt(
                receipt,
                self._old_channel_trust,
                target_installation_id=target_installation_id,
                minimum_fence_epoch=self._state.fence_epoch,
            )
            await asyncio.to_thread(self._state.record_old_fence, receipt)
            return receipt

    async def accept(self, receipt: FenceReceiptV1) -> None:
        if receipt.payload.old_installation_id != self._old_channel_trust.installation_id:
            raise ValueError("fence receipt is not from the trusted old installation")
        async with self._dispatcher_gate.hold_activation():
            verify_fence_receipt(
                receipt,
                self._old_channel_trust,
                target_installation_id=self._state.installation_id,
                minimum_fence_epoch=self._state.fence_epoch,
            )
            await asyncio.to_thread(self._state.accept_takeover, receipt)


class ActivationGenerationGate:
    """Task5 generation evidence intersected with Task9's live deny overlay."""

    def __init__(
        self,
        generation_gate: GenerationOperationalGate,
        state: GatewayLocalState,
    ) -> None:
        self._generation_gate = generation_gate
        self._state = state

    def permits(self, leg: AuthorizedLeg, backend_manifest_hash: str) -> bool:
        return self._state.permits(leg) and self._generation_gate.permits(
            leg,
            backend_manifest_hash,
        )
class ActivationService:
    """Auth-first executor; bundle parsing happens only after command authentication."""

    def __init__(
        self, *, state: GatewayLocalState,
        policy_keys: Mapping[int, Mapping[str, Ed25519PublicKey]],
        channel_trust: ChannelTrustSet, ack_signer: ChannelSigner,
        generation_gate: GenerationOperationalGate, backend_manifest_hash: str,
        dispatcher_gate: DispatcherGate, apply_bundle: Callable[[PolicyBundleV1], None],
        clock: Clock | None = None, limits: BundleLimits | None = None,
        readiness_probe: Callable[[], tuple[bool, bool, bool]] | None = None,
    ) -> None:
        if not policy_keys or not channel_trust.channel_epochs:
            raise ValueError("policy and channel trust must be injected")
        if len(backend_manifest_hash) != 64:
            raise ValueError("backend manifest hash must be exact")
        self.state, self.policy_keys, self.channel_trust = state, policy_keys, channel_trust
        self.ack_signer, self.generation_gate = ack_signer, generation_gate
        self.backend_manifest_hash, self.dispatcher_gate = backend_manifest_hash, dispatcher_gate
        self.apply_bundle, self.clock = apply_bundle, clock or _SystemClock()
        self.limits = limits or BundleLimits()
        self._validation_slots = asyncio.Semaphore(2)
        self._command_lock = asyncio.Lock()
        self._readiness = readiness_probe or (lambda: (False, False, False))

    def _verify(self, command: GatewayCommandV1) -> VerifiedGatewayCommand:
        return verify_gateway_command(
            command, self.policy_keys, self.channel_trust,
            expected_boot_id=self.state.boot_id, expected_fence_epoch=self.state.fence_epoch,
        )

    def _assert_current_fence(self, verified: VerifiedGatewayCommand) -> None:
        if verified.command.dispatcher_fence != self.state.fence_epoch:
            raise StaleFenceEpoch("command dispatcher fence changed while waiting")

    @staticmethod
    def _policy_matches(policy: ActivationEnvelope | None, base: BaseReference | None, target: BundleReference) -> None:
        if policy is None or policy.base != base or policy.target != target:
            raise ValueError("signed policy does not bind exact base/target")

    def _validate_bundle(self, raw: bytes, target: BundleReference) -> PolicyBundleV1:
        if len(raw) > self.limits.max_bundle_bytes:
            raise ValueError("bundle size exceeds the configured limit")
        if bundle_hash(raw) != target.bundle_hash:
            raise ValueError("bundle hash differs from target")
        try: bundle = PolicyBundleV1.model_validate(json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error: raise ValueError("bundle schema is invalid") from error
        if canonical_bundle_bytes(bundle) != raw: raise ValueError("bundle is not canonical JSON")
        if bundle.generation != target.generation: raise ValueError("bundle generation differs")
        if bundle.backend_manifest_hash != self.backend_manifest_hash: raise ValueError("backend manifest differs")
        counts = (
            (len(bundle.keys), self.limits.max_keys), (len(bundle.policies), self.limits.max_policies),
            (len(bundle.accounts), self.limits.max_accounts), (len(bundle.route_groups), self.limits.max_route_groups),
            (sum(len(x.legs) for x in bundle.route_groups), self.limits.max_legs),
            (sum(len(x.authorized_legs) for x in bundle.policies), self.limits.max_authorized_legs),
        )
        if any(actual > maximum for actual, maximum in counts): raise ValueError("bundle object count exceeds limit")
        if any(not self.generation_gate.permits(leg, bundle.backend_manifest_hash) for policy in bundle.policies for leg in policy.authorized_legs):
            raise ValueError("bundle deployment generation is not operational")
        return bundle

    def _ack(self, verified: VerifiedGatewayCommand, record: _CommandRecord) -> GatewayAckV1:
        if record.ack is not None: return record.ack
        command = verified.command
        unsigned = GatewayAckV1(
            ack_id=record.ack_id,
            command_id=command.command_id,
            command_digest=verified.command_digest,
            installation_id=self.state.installation_id,
            channel_epoch=command.channel_epoch,
            security_epoch=self.state.security_epoch,
            dispatcher_fence=self.state.fence_epoch,
            boot_id=self.state.boot_id,
            sequence=command.sequence,
            status=record.status,
            acknowledged_at_ms=self.clock.now_ms(),
            result=record.result,
            channel_seal=ChannelSealV1(
                seal_id=self.ack_signer.seal_id,
                trust_epoch=self.ack_signer.trust_epoch,
                signature="0" * 128,
            ),
        )
        ack = seal_gateway_ack(unsigned, self.ack_signer)
        self.state.push_ack(ack)
        return ack

    async def execute(self, command: GatewayCommandV1) -> GatewayAckV1:
        verified = self._verify(command)
        async with self._command_lock:
            existing = self.state.validate_new_command(verified)
            if existing is not None:
                return self._ack(verified, existing)
            if command.kind is WireCommandKind.PREPARE:
                await self._prepare(verified)
            elif command.kind is WireCommandKind.COMMIT:
                return await self._commit(verified)
            elif command.kind is WireCommandKind.DENY:
                return await self._deny(verified)
            elif command.kind is WireCommandKind.CLEAR_DENY:
                return await self._clear(verified)
            elif command.kind is WireCommandKind.STATUS:
                self._status(verified)
            else:
                raise ValueError("takeover uses the fence-receipt API")
            recorded = self.state.validate_new_command(verified)
            assert recorded is not None
            return self._ack(verified, recorded)

    async def _prepare(self, verified: VerifiedGatewayCommand) -> None:
        payload = PrepareCommandPayload.model_validate(verified.command.payload)
        raw = payload.bundle_bytes()
        async with self._validation_slots: bundle = await asyncio.to_thread(self._validate_bundle, raw, payload.target)
        self._policy_matches(verified.policy, payload.base, payload.target)
        self.state.validate_prepare_reference(payload.base, payload.target)
        self.state.store_bundle(raw, payload.target.bundle_hash)
        self.state.stage_prepare(verified, bundle, payload.target, payload.base, ack_id=AckId.new(), result={"target": payload.target.model_dump(mode="json")})

    async def _commit(self, verified: VerifiedGatewayCommand) -> GatewayAckV1:
        target = BundleReference.model_validate(verified.command.payload)
        self._policy_matches(verified.policy, self.state._base, target)
        async with self.dispatcher_gate.hold_activation():
            self._assert_current_fence(verified)
            bundle = await asyncio.to_thread(
                self.state.commit_staged,
                verified,
                target,
                ack_id=AckId.new(),
                result={"active": target.model_dump(mode="json")},
            )
            self.apply_bundle(bundle)
            self.state.crash("memory_after_swap")
            recorded = self.state.validate_new_command(verified)
            assert recorded is not None
            return self._ack(verified, recorded)

    def _active_policy(self, policy: ActivationEnvelope | None) -> None:
        if self.state.active_reference is None or policy is None or policy.target != self.state.active_reference:
            raise ValueError("signed policy does not bind active bundle")

    async def _deny(self, verified: VerifiedGatewayCommand) -> GatewayAckV1:
        payload = DenyCommandPayload.model_validate(verified.command.payload)
        self._active_policy(verified.policy)
        async with self.dispatcher_gate.hold_activation():
            await asyncio.to_thread(
                self.state.issue_deny,
                verified,
                payload,
                ack_id=AckId.new(),
                result={"deny": payload.model_dump(mode="json")},
            )
            recorded = self.state.validate_new_command(verified)
            assert recorded is not None
            return self._ack(verified, recorded)

    async def _clear(self, verified: VerifiedGatewayCommand) -> GatewayAckV1:
        payload = ClearDenyCommandPayload.model_validate(verified.command.payload)
        self._active_policy(verified.policy)
        async with self.dispatcher_gate.hold_activation():
            await asyncio.to_thread(
                self.state.clear_deny,
                verified,
                payload,
                ack_id=AckId.new(),
                result={"cleared": payload.model_dump(mode="json")},
            )
            recorded = self.state.validate_new_command(verified)
            assert recorded is not None
            return self._ack(verified, recorded)

    def _status(self, verified: VerifiedGatewayCommand) -> None:
        payload = StatusCommandPayload.model_validate(verified.command.payload)
        self.state.renew_deny_heartbeats(self.clock.now_ms())
        singleton, backend, capacities = self._readiness()
        def result(_lease: AuthLeaseV1 | None) -> Mapping[str, JsonValue]:
            status = self.state.status(singleton_held=singleton, backend_ready=backend, capacities_ready=capacities, now_ms=self.clock.now_ms())
            return {"status": status.model_dump(mode="json")}
        self.state.record_status(verified, ack_id=AckId.new(), issue_lease=payload.issue_auth_lease, result_factory=result)
