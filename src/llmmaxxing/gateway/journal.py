"""Segmented, fsync-before-send attempt journal with conservative recovery."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import statistics
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Self, cast

type JsonScalar = str | int | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

_ZERO_DIGEST = "0" * 64
_MANIFEST = "manifest.json"
_VERSION = 1
_SAFE_KEY = re.compile(r"^[a-z0-9_.:-]{1,180}$")
_SAFE_ID = re.compile(
    r"^(?:acc|att|boot|inst|leg|probe|req)_[0-9a-f-]{32,36}$"
    r"|^dg1_[0-9a-f]{64}$|^bh_[0-9a-f]{64}$"
)
_SAFE_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SAFE_WORDS = frozenset(
    {
        "active",
        "available",
        "capacity_exhausted",
        "capacity",
        "account",
        "deployment",
        "client_cancelled",
        "closed",
        "auth_state_unavailable",
        "authz_denied",
        "backpressure_rejected",
        "completed",
        "crash_recovery",
        "deadline_exceeded",
        "manual",
        "disabled",
        "draft",
        "half_open",
        "inconclusive",
        "open",
        "response_stream_failed",
        "tombstoned",
        "quota",
        "transient_failure",
        "route_unavailable",
        "unsupported_request",
        "upstream_failed",
        "uncertain",
        "upstream_unknown",
    }
)
_RECORD_FIELDS: dict[str, frozenset[str]] = {
    "account_registered": frozenset(
        {
            "account_id",
            "binding_digest",
            "credential_attestation_digest",
            "credential_epoch",
            "state",
        }
    ),
    "attempt_reserved": frozenset(
        {
            "request_id",
            "attempt_id",
            "account_id",
            "leg_id",
            "installation_id",
            "deployment_generation_id",
            "bundle_generation",
            "bundle_hash",
            "fence_token",
            "boot_id",
            "deadline_at_ms",
            "profile_digest",
            "started_at_ms",
            "reserved_tokens",
            "quota_units",
            "monthly_reset_at_ms",
            "circuit_epoch",
            "account_circuit_epoch",
            "circuit_probe_id",
            "account_circuit_probe_id",
        }
    ),
    "attempt_dispatched": frozenset({"attempt_id", "dispatched_at_ms"}),
    "attempt_uncertain": frozenset({"attempt_id", "reason"}),
    "attempt_resolved": frozenset(
        {
            "attempt_id",
            "outcome",
            "release_capacity",
            "actual_starts",
            "actual_token_units",
            "actual_quota_units",
            "resolution_digest",
            "resolved_at_ms",
        }
    ),
    "circuit_updated": frozenset(
        {
            "account_id",
            "deployment_generation_id",
            "scope",
            "state",
            "cause",
            "opened_at_ms",
            "backoff_step",
            "evidence_digest",
            "epoch",
            "retry_at_ms",
            "probe_id",
        }
    ),
    "authoritative_active_count": frozenset({"account_id", "active_count"}),
    "recovery_probe_started": frozenset({"account_id", "probe_id"}),
    "recovery_probe_finished": frozenset({"account_id", "probe_id", "classification"}),
}


class Clock(Protocol):
    def now_ms(self) -> int: ...


class _SystemClock:
    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000


class JournalStatus(StrEnum):
    HEALTHY = "healthy"
    ADMISSION_STOP = "admission_stop"
    RECOVERY_REQUIRED = "recovery_required"
    CLOSED = "closed"


class InjectedCrash(RuntimeError):
    """Deterministic hard-crash boundary used by focused recovery tests."""


class JournalUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WriterLimits:
    queue_capacity: int = 4096
    max_group_records: int = 256
    max_group_delay_ms: int = 2


@dataclass(frozen=True, slots=True)
class JournalReceipt:
    durable_lsn: int
    record_digest: str


@dataclass(frozen=True, slots=True)
class JournalRecord:
    sequence: int
    kind: str
    payload: dict[str, JsonValue]
    previous_digest: str
    digest: str


@dataclass(frozen=True, slots=True)
class JournalRecovery:
    status: JournalStatus
    reason: str | None
    checkpoint: dict[str, JsonValue] | None
    records: tuple[JournalRecord, ...]
    checkpoint_sequence: int
    last_sequence: int
    last_digest: str
    replayed_records: int
    elapsed_ms: int
    verified_checkpoints: int


@dataclass(frozen=True, slots=True)
class JournalHealth:
    status: JournalStatus
    reason: str | None
    journal_bytes: int
    disk_usage_ratio: float
    last_sequence: int
    checkpoint_sequence: int
    verified_checkpoints: int
    replayed_records: int
    fdatasync_p99_ms: float | None
    storage_doctor_ok: bool


@dataclass(frozen=True, slots=True)
class DurableReservation:
    request_id: str
    attempt_id: str
    account_id: str
    leg_id: str
    deployment_generation_id: str
    installation_id: str
    bundle_generation: int
    bundle_hash: str
    fence_token: int
    boot_id: str
    deadline_at_ms: int
    profile_digest: str
    started_at_ms: int
    reserved_tokens: int
    quota_units: int
    monthly_reset_at_ms: int
    circuit_epoch: int
    account_circuit_epoch: int
    circuit_probe_id: str | None
    account_circuit_probe_id: str | None


@dataclass(slots=True)
class _AppendCommand:
    kind: str
    payload: dict[str, JsonValue]
    boundary: str | None
    done: threading.Event = field(default_factory=threading.Event)
    result: JournalRecord | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class _CheckpointCommand:
    snapshot: dict[str, JsonValue]
    force: bool
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(slots=True)
class _StopCommand:
    done: threading.Event = field(default_factory=threading.Event)


type _Command = _AppendCommand | _CheckpointCommand | _StopCommand
type CrashInjector = Callable[[str], None]
type DiskUsage = Callable[[Path], float]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_all(fd: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(fd, content[offset:])
        if written <= 0:
            raise OSError("short journal write")
        offset += written


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _default_disk_usage(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return 0.0 if usage.total == 0 else usage.used / usage.total


def _validate_safe_value(value: JsonValue, *, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("journal value exceeds maximum nesting")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if value < 0 or value > 2**63 - 1:
            raise ValueError("journal integer is out of range")
        return
    if isinstance(value, str):
        if len(value) > 180 or not (
            _SAFE_ID.fullmatch(value) or _SAFE_DIGEST.fullmatch(value) or value in _SAFE_WORDS
        ):
            raise ValueError("journal strings must be bounded IDs, digests, or reasons")
        return
    if isinstance(value, list):
        if len(value) > 100_000:
            raise ValueError("journal list is too large")
        for item in value:
            _validate_safe_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 100_000:
            raise ValueError("journal object is too large")
        for key, item in value.items():
            if not _SAFE_KEY.fullmatch(key):
                raise ValueError("journal object key is not bounded")
            _validate_safe_value(item, depth=depth + 1)
        return
    raise TypeError("journal values must be JSON values")


def _validate_payload(kind: str, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    expected = _RECORD_FIELDS.get(kind)
    if expected is None or frozenset(payload) != expected:
        raise ValueError(f"invalid closed journal payload for {kind!r}")
    copied = dict(payload)
    _validate_safe_value(cast(JsonValue, copied))
    return copied


def _manifest_document(last_sequence: int, last_digest: str, journal_id: str) -> dict[str, object]:
    body: dict[str, object] = {
        "version": _VERSION,
        "journal_id": journal_id,
        "last_sequence": last_sequence,
        "last_digest": last_digest,
    }
    return {**body, "digest": _digest(body)}


def _validate_manifest(value: object) -> tuple[int, str, str]:
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    expected = {"version", "journal_id", "last_sequence", "last_digest", "digest"}
    if set(value) != expected:
        raise ValueError("manifest fields are invalid")
    body = {key: value[key] for key in expected - {"digest"}}
    if value["digest"] != _digest(body) or value["version"] != _VERSION:
        raise ValueError("manifest digest or version is invalid")
    journal_id = value["journal_id"]
    last_sequence = value["last_sequence"]
    last_digest = value["last_digest"]
    if (
        not isinstance(journal_id, str)
        or not _SAFE_ID.fullmatch(journal_id)
        or not journal_id.startswith("boot_")
        or not isinstance(last_sequence, int)
        or last_sequence < 0
        or not isinstance(last_digest, str)
        or not _SAFE_DIGEST.fullmatch(last_digest)
    ):
        raise ValueError("manifest values are invalid")
    return last_sequence, last_digest, journal_id


class AttemptJournal:
    """One-replica segmented writer; reservations return only after fdatasync."""

    writer_limits = WriterLimits()

    def __init__(
        self,
        root: Path,
        *,
        create: bool,
        clock: Clock | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        crash_injector: CrashInjector | None = None,
        disk_usage: DiskUsage = _default_disk_usage,
        group_commit_delay_ms: int = 2,
        max_group_records: int = 256,
        checkpoint_every_records: int = 10_000,
        checkpoint_every_ms: int = 60_000,
        segment_bytes: int = 16 * 1024 * 1024,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
        disk_stop_ratio: float = 0.8,
        recovery_deadline_seconds: float = 30.0,
        max_replay_records: int = 10_256,
    ) -> None:
        if not 0 <= group_commit_delay_ms <= 2:
            raise ValueError("group commit delay must be at most 2ms")
        if not 1 <= max_group_records <= 256:
            raise ValueError("group commit must contain at most 256 records")
        if checkpoint_every_records < 1 or checkpoint_every_ms < 1:
            raise ValueError("checkpoint intervals must be positive")
        if segment_bytes < 1 or max_bytes < 1:
            raise ValueError("journal byte limits must be positive")
        if not 0 < disk_stop_ratio <= 1:
            raise ValueError("disk stop ratio must be in (0, 1]")

        self.root = root
        self._clock = clock or _SystemClock()
        self._monotonic = monotonic
        self._crash_injector = crash_injector
        self._disk_usage = disk_usage
        self._group_commit_delay_ms = group_commit_delay_ms
        self._max_group_records = max_group_records
        self._checkpoint_every_records = checkpoint_every_records
        self._checkpoint_every_ms = checkpoint_every_ms
        self._segment_bytes = segment_bytes
        self._max_bytes = max_bytes
        self._disk_stop_ratio = disk_stop_ratio
        self._recovery_deadline_seconds = recovery_deadline_seconds
        self._max_replay_records = max_replay_records
        self._sync_samples_ms: list[float] = []
        self._queue: queue.Queue[_Command] = queue.Queue(maxsize=self.writer_limits.queue_capacity)
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._segment_fd: int | None = None
        self._segment_size = 0
        self._segment_start = 0
        self._records_since_checkpoint = 0
        self._last_checkpoint_ms = self._clock.now_ms()
        self._checkpoint_nonce = 0
        self._journal_id = f"boot_{uuid.uuid4()}"

        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(root, 0o700)
            manifest = root / _MANIFEST
            if not manifest.exists():
                unexpected = [path for path in root.iterdir() if not path.name.endswith(".tmp")]
                if unexpected:
                    self.recovery = self._failed_recovery("missing_manifest")
                else:
                    self._write_manifest(0, _ZERO_DIGEST)
                    self.recovery = JournalRecovery(
                        status=JournalStatus.HEALTHY,
                        reason=None,
                        checkpoint=None,
                        records=(),
                        checkpoint_sequence=0,
                        last_sequence=0,
                        last_digest=_ZERO_DIGEST,
                        replayed_records=0,
                        elapsed_ms=0,
                        verified_checkpoints=0,
                    )
            else:
                self.recovery = self._recover()
        else:
            if not root.is_dir() or not (root / _MANIFEST).is_file():
                self.recovery = self._failed_recovery("missing_journal")
            else:
                self.recovery = self._recover()

        self._base_status = self.recovery.status
        self._reason = self.recovery.reason
        self._last_sequence = self.recovery.last_sequence
        self._last_digest = self.recovery.last_digest
        self._checkpoint_sequence = self.recovery.checkpoint_sequence
        self._verified_checkpoints = self.recovery.verified_checkpoints
        self._records_since_checkpoint = (
            self.recovery.last_sequence - self.recovery.checkpoint_sequence
        )
        if self._base_status is JournalStatus.HEALTHY:
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="llmmaxxing-attempt-journal",
                daemon=True,
            )
            self._thread.start()

    def enter_recovery_required(self, reason: str) -> None:
        """Fail closed and stop a still-healthy writer after invalid recovery state."""
        command: _StopCommand | None = None
        thread = self._thread
        with self._lifecycle_lock:
            if self._base_status is JournalStatus.HEALTHY and thread is not None:
                command = _StopCommand()
                self._queue.put(command)
            self._base_status = JournalStatus.RECOVERY_REQUIRED
            self._reason = reason
        if command is not None and thread is not None:
            command.done.wait()
            thread.join()

    @classmethod
    def create(cls, root: Path, **kwargs: Any) -> Self:
        return cls(root, create=True, **kwargs)

    @classmethod
    def open(cls, root: Path, **kwargs: Any) -> Self:
        return cls(root, create=False, **kwargs)

    @property
    def status(self) -> JournalStatus:
        return self.health.status

    @property
    def checkpoint_due(self) -> bool:
        return (
            self._records_since_checkpoint >= self._checkpoint_every_records
            or self._clock.now_ms() - self._last_checkpoint_ms >= self._checkpoint_every_ms
        )

    @property
    def records_until_checkpoint(self) -> int:
        return max(0, self._checkpoint_every_records - self._records_since_checkpoint)

    @property
    def health(self) -> JournalHealth:
        if self._closed:
            status = JournalStatus.CLOSED
        elif self._base_status is JournalStatus.RECOVERY_REQUIRED:
            status = JournalStatus.RECOVERY_REQUIRED
        else:
            status = (
                JournalStatus.ADMISSION_STOP
                if self._admission_capacity_exhausted()
                else JournalStatus.HEALTHY
            )
        p99 = self._fdatasync_p99()
        return JournalHealth(
            status=status,
            reason=self._reason,
            journal_bytes=self._journal_bytes(),
            disk_usage_ratio=self._disk_usage(self.root) if self.root.exists() else 1.0,
            last_sequence=self._last_sequence,
            checkpoint_sequence=self._checkpoint_sequence,
            verified_checkpoints=self._verified_checkpoints,
            replayed_records=self.recovery.replayed_records,
            fdatasync_p99_ms=p99,
            storage_doctor_ok=p99 is not None and p99 <= 2.0,
        )

    def register_account(
        self,
        *,
        account_id: str,
        binding_digest: str,
        credential_attestation_digest: str,
        credential_epoch: int,
        state: str,
    ) -> JournalRecord:
        return self._append(
            "account_registered",
            {
                "account_id": account_id,
                "binding_digest": binding_digest,
                "credential_attestation_digest": credential_attestation_digest,
                "credential_epoch": credential_epoch,
                "state": state,
            },
        )

    def reserve_before_send(self, reservation: DurableReservation) -> JournalReceipt:
        payload = cast(dict[str, JsonValue], asdict(reservation))
        health = self.health
        if health.status is not JournalStatus.HEALTHY:
            raise JournalUnavailable(health.status.value)
        if health.journal_bytes + len(_canonical(payload)) + 512 > int(self._max_bytes * 0.8):
            raise JournalUnavailable(JournalStatus.ADMISSION_STOP.value)
        record = self._append(
            "attempt_reserved",
            payload,
            boundary="reservation",
        )
        return JournalReceipt(record.sequence, record.digest)

    def record_dispatch(self, *, attempt_id: str, dispatched_at_ms: int) -> JournalReceipt:
        record = self._append(
            "attempt_dispatched",
            {"attempt_id": attempt_id, "dispatched_at_ms": dispatched_at_ms},
            boundary="reservation",
        )
        return JournalReceipt(record.sequence, record.digest)

    def provider_send_completed(self, attempt_id: str) -> None:
        del attempt_id
        self._inject("after_provider_send_before_terminal")

    def record_resolution(
        self,
        *,
        attempt_id: str,
        outcome: str,
        release_capacity: bool,
        actual_starts: int | None,
        actual_token_units: int | None,
        actual_quota_units: int | None,
        resolution_digest: str,
        resolved_at_ms: int,
    ) -> JournalReceipt:
        record = self._append(
            "attempt_resolved",
            {
                "attempt_id": attempt_id,
                "outcome": outcome,
                "release_capacity": release_capacity,
                "actual_starts": actual_starts,
                "actual_token_units": actual_token_units,
                "actual_quota_units": actual_quota_units,
                "resolution_digest": resolution_digest,
                "resolved_at_ms": resolved_at_ms,
            },
            boundary="terminal_update",
        )
        return JournalReceipt(record.sequence, record.digest)

    def mark_attempt_uncertain(self, *, attempt_id: str, reason: str) -> JournalRecord:
        return self._append(
            "attempt_uncertain",
            {"attempt_id": attempt_id, "reason": reason},
            boundary="terminal_update",
        )

    def update_circuit(
        self,
        *,
        account_id: str,
        scope: str,
        deployment_generation_id: str | None,
        state: str,
        cause: str | None,
        opened_at_ms: int,
        backoff_step: int,
        evidence_digest: str | None,
        epoch: int,
        retry_at_ms: int,
        probe_id: str | None,
    ) -> JournalRecord:
        return self._append(
            "circuit_updated",
            {
                "account_id": account_id,
                "scope": scope,
                "deployment_generation_id": deployment_generation_id,
                "state": state,
                "epoch": epoch,
                "retry_at_ms": retry_at_ms,
                "probe_id": probe_id,
                "cause": cause,
                "opened_at_ms": opened_at_ms,
                "backoff_step": backoff_step,
                "evidence_digest": evidence_digest,
            },
        )

    def record_authoritative_active_count(self, *, account_id: str, active_count: int) -> None:
        self._append(
            "authoritative_active_count",
            {"account_id": account_id, "active_count": active_count},
        )

    def record_recovery_probe_started(self, *, account_id: str, probe_id: str) -> None:
        self._append("recovery_probe_started", {"account_id": account_id, "probe_id": probe_id})

    def record_recovery_probe_finished(
        self, *, account_id: str, probe_id: str, classification: str
    ) -> None:
        self._append(
            "recovery_probe_finished",
            {"account_id": account_id, "probe_id": probe_id, "classification": classification},
        )

    def maybe_checkpoint(self, snapshot: Mapping[str, JsonValue]) -> bool:
        due = (
            self._records_since_checkpoint >= self._checkpoint_every_records
            or self._clock.now_ms() - self._last_checkpoint_ms >= self._checkpoint_every_ms
        )
        if not due:
            return False
        self._checkpoint(dict(snapshot), force=False)
        return True

    def force_checkpoint(self, snapshot: Mapping[str, JsonValue]) -> None:
        self._checkpoint(dict(snapshot), force=True)

    def storage_doctor(self, samples: int = 3) -> bool:
        if samples < 1:
            raise ValueError("doctor samples must be positive")
        path = self.root / ".doctor.tmp"
        fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        try:
            for _ in range(samples):
                _write_all(fd, b"0")
                self._timed_fdatasync(fd)
        finally:
            os.close(fd)
            path.unlink(missing_ok=True)
        p99 = self._fdatasync_p99()
        return p99 is not None and p99 <= 2.0

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            command: _StopCommand | None = None
            if (
                thread is not None
                and thread.is_alive()
                and self._base_status is JournalStatus.HEALTHY
            ):
                command = _StopCommand()
                self._queue.put(command)
        if thread is not None:
            if command is not None:
                command.done.wait()
            thread.join()
        if self._segment_fd is not None and (thread is None or not thread.is_alive()):
            os.close(self._segment_fd)
            self._segment_fd = None

    def _failed_recovery(self, reason: str) -> JournalRecovery:
        return JournalRecovery(
            status=JournalStatus.RECOVERY_REQUIRED,
            reason=reason,
            checkpoint=None,
            records=(),
            checkpoint_sequence=0,
            last_sequence=0,
            last_digest=_ZERO_DIGEST,
            replayed_records=0,
            elapsed_ms=0,
            verified_checkpoints=0,
        )

    def _recover(self) -> JournalRecovery:
        started = self._monotonic()
        try:
            manifest_value = json.loads((self.root / _MANIFEST).read_text())
            manifest_sequence, manifest_digest, self._journal_id = _validate_manifest(
                manifest_value
            )
            checkpoints: list[tuple[int, str, dict[str, JsonValue], Path]] = []
            checkpoint_paths = sorted(self.root.glob("checkpoint-*.json"))
            for path in checkpoint_paths:
                self._check_recovery_deadline(started)
                try:
                    sequence, last_digest, snapshot = self._read_checkpoint(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                checkpoints.append((sequence, last_digest, snapshot, path))
            checkpoints.sort(key=lambda item: (item[0], item[3].name))
            checkpoint_sequence = 0
            if checkpoint_paths and not checkpoints:
                raise ValueError("all retained checkpoints are invalid")
            checkpoint_digest = _ZERO_DIGEST
            checkpoint: dict[str, JsonValue] | None = None
            if checkpoints:
                checkpoint_sequence, checkpoint_digest, checkpoint, _ = checkpoints[-1]

            expected_sequence = checkpoint_sequence + 1
            previous_digest = checkpoint_digest
            records: list[JournalRecord] = []
            segments = sorted(self.root.glob("segment-*.jsonl"))
            for path in segments:
                self._check_recovery_deadline(started)
                valid_offset = 0
                with path.open("rb") as stream:
                    while raw := stream.readline():
                        self._check_recovery_deadline(started)
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError:
                            newest_incomplete_tail = (
                                path == segments[-1]
                                and not raw.endswith(b"\n")
                                and stream.tell() == path.stat().st_size
                            )
                            if not newest_incomplete_tail:
                                raise
                            fd = os.open(path, os.O_WRONLY)
                            try:
                                os.ftruncate(fd, valid_offset)
                                self._timed_fdatasync(fd)
                            finally:
                                os.close(fd)
                            _fsync_directory(self.root)
                            break
                        record = self._decode_record(value)
                        valid_offset = stream.tell()
                        if record.sequence <= checkpoint_sequence:
                            continue
                        if (
                            record.sequence != expected_sequence
                            or record.previous_digest != previous_digest
                        ):
                            raise ValueError("journal sequence or digest chain gap")
                        records.append(record)
                        expected_sequence += 1
                        previous_digest = record.digest
                        if len(records) > self._max_replay_records:
                            raise TimeoutError("replay_record_bound_exceeded")
            last_sequence = expected_sequence - 1
            if manifest_sequence > last_sequence:
                raise ValueError("manifest high-water exceeds durable journal")
            if manifest_sequence == last_sequence and manifest_digest != previous_digest:
                raise ValueError("manifest digest disagrees with durable journal")
            self._check_recovery_deadline(started)
            elapsed_ms = int((self._monotonic() - started) * 1000)
            return JournalRecovery(
                status=JournalStatus.HEALTHY,
                reason=None,
                checkpoint=checkpoint,
                records=tuple(records),
                checkpoint_sequence=checkpoint_sequence,
                last_sequence=last_sequence,
                last_digest=previous_digest,
                replayed_records=len(records),
                elapsed_ms=elapsed_ms,
                verified_checkpoints=min(2, len(checkpoints)),
            )
        except TimeoutError as error:
            reason = str(error) or "recovery_deadline_exceeded"
            if reason != "replay_record_bound_exceeded":
                reason = "recovery_deadline_exceeded"
            return self._failed_recovery(reason)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return self._failed_recovery("corrupt_journal")

    def _check_recovery_deadline(self, started: float) -> None:
        if self._monotonic() - started > self._recovery_deadline_seconds:
            raise TimeoutError("recovery_deadline_exceeded")

    def _read_checkpoint(self, path: Path) -> tuple[int, str, dict[str, JsonValue]]:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value) != {
            "version",
            "sequence",
            "last_record_digest",
            "created_at_ms",
            "snapshot",
            "digest",
        }:
            raise ValueError("checkpoint fields are invalid")
        body = {key: value[key] for key in value if key != "digest"}
        if value["version"] != _VERSION or value["digest"] != _digest(body):
            raise ValueError("checkpoint digest or version is invalid")
        sequence = value["sequence"]
        last_digest = value["last_record_digest"]
        snapshot = value["snapshot"]
        if (
            not isinstance(sequence, int)
            or sequence < 0
            or not isinstance(last_digest, str)
            or not _SAFE_DIGEST.fullmatch(last_digest)
            or not isinstance(snapshot, dict)
        ):
            raise ValueError("checkpoint values are invalid")
        typed_snapshot = cast(dict[str, JsonValue], snapshot)
        _validate_safe_value(typed_snapshot)
        return sequence, last_digest, typed_snapshot

    def _decode_record(self, value: object) -> JournalRecord:
        if not isinstance(value, dict) or set(value) != {
            "sequence",
            "kind",
            "payload",
            "previous_digest",
            "digest",
        }:
            raise ValueError("journal record fields are invalid")
        sequence = value["sequence"]
        kind = value["kind"]
        payload = value["payload"]
        previous_digest = value["previous_digest"]
        digest = value["digest"]
        if (
            not isinstance(sequence, int)
            or sequence < 1
            or not isinstance(kind, str)
            or not isinstance(payload, dict)
            or not isinstance(previous_digest, str)
            or not isinstance(digest, str)
        ):
            raise ValueError("journal record values are invalid")
        typed_payload = _validate_payload(kind, cast(dict[str, JsonValue], payload))
        body = {
            "sequence": sequence,
            "kind": kind,
            "payload": typed_payload,
            "previous_digest": previous_digest,
        }
        if digest != _digest(body):
            raise ValueError("journal record digest is invalid")
        return JournalRecord(sequence, kind, typed_payload, previous_digest, digest)

    def _append(
        self,
        kind: str,
        payload: Mapping[str, JsonValue],
        *,
        boundary: str | None = None,
    ) -> JournalRecord:
        command = _AppendCommand(kind, _validate_payload(kind, payload), boundary)
        with self._lifecycle_lock:
            if self._closed or self._base_status is not JournalStatus.HEALTHY:
                raise JournalUnavailable(self._reason or self._base_status.value)
            try:
                self._queue.put_nowait(command)
            except queue.Full:
                raise JournalUnavailable("writer_queue_full") from None
        command.done.wait()
        if command.error is not None:
            if isinstance(command.error, OSError):
                raise JournalUnavailable("writer_failed") from command.error
            raise command.error
        if command.result is None:
            raise JournalUnavailable("writer_stopped")
        return command.result

    def _checkpoint(self, snapshot: dict[str, JsonValue], *, force: bool) -> None:
        _validate_safe_value(snapshot)
        command = _CheckpointCommand(snapshot, force)
        with self._lifecycle_lock:
            if self._closed or self._base_status is not JournalStatus.HEALTHY:
                raise JournalUnavailable(self._reason or self._base_status.value)
            try:
                self._queue.put_nowait(command)
            except queue.Full:
                raise JournalUnavailable("writer_queue_full") from None
        command.done.wait()
        if command.error is not None:
            if isinstance(command.error, OSError):
                raise JournalUnavailable("writer_failed") from command.error
            raise command.error

    def _writer_loop(self) -> None:
        pending: _Command | None = None
        while True:
            command = pending or self._queue.get()
            pending = None
            if isinstance(command, _StopCommand):
                self._flush_and_close_segment()
                command.done.set()
                return
            try:
                if isinstance(command, _CheckpointCommand):
                    self._write_checkpoint(command.snapshot)
                    command.done.set()
                    continue
                batch = [command]
                deadline = time.monotonic() + self._group_commit_delay_ms / 1000
                while len(batch) < self._max_group_records:
                    try:
                        if self._group_commit_delay_ms == 0:
                            candidate = self._queue.get_nowait()
                        else:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            candidate = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if not isinstance(candidate, _AppendCommand):
                        pending = candidate
                        break
                    batch.append(candidate)
                self._write_append_batch(batch)
                for item in batch:
                    item.done.set()
            except BaseException as error:
                normalized: BaseException = (
                    JournalUnavailable("writer_failed") if isinstance(error, OSError) else error
                )
                self._base_status = JournalStatus.RECOVERY_REQUIRED
                self._reason = "writer_failed"
                if isinstance(command, _CheckpointCommand):
                    command.error = normalized
                    command.done.set()
                else:
                    for item in batch:
                        item.error = normalized
                        item.done.set()
                if pending is not None:
                    self._fail_command(pending, normalized)
                    pending = None
                while True:
                    try:
                        queued = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    self._fail_command(queued, normalized)
                self._flush_and_close_segment(sync=False)
                return

    @staticmethod
    def _fail_command(command: _Command, error: BaseException) -> None:
        if isinstance(command, _StopCommand):
            command.done.set()
            return
        command.error = error
        command.done.set()

    def _write_append_batch(self, batch: list[_AppendCommand]) -> None:
        boundaries = tuple(
            dict.fromkeys(item.boundary for item in batch if item.boundary is not None)
        )
        for boundary in boundaries:
            self._inject(f"before_{boundary}_fsync")

        touched: set[int] = set()
        for item in batch:
            sequence = self._last_sequence + 1
            body = {
                "sequence": sequence,
                "kind": item.kind,
                "payload": item.payload,
                "previous_digest": self._last_digest,
            }
            digest = _digest(body)
            record = JournalRecord(
                sequence=sequence,
                kind=item.kind,
                payload=item.payload,
                previous_digest=self._last_digest,
                digest=digest,
            )
            content = _canonical({**body, "digest": digest}) + b"\n"
            self._ensure_segment(sequence, len(content))
            assert self._segment_fd is not None
            _write_all(self._segment_fd, content)
            self._segment_size += len(content)
            touched.add(self._segment_fd)
            self._last_sequence = sequence
            self._last_digest = digest
            item.result = record

        for fd in touched:
            self._timed_fdatasync(fd)
        for boundary in boundaries:
            self._inject(f"after_{boundary}_fsync")
        self._write_manifest(self._last_sequence, self._last_digest)
        self._records_since_checkpoint += len(batch)

    def _ensure_segment(self, sequence: int, content_length: int) -> None:
        if self._segment_fd is not None and self._segment_size > 0:
            if self._segment_size + content_length <= self._segment_bytes:
                return
            self._flush_and_close_segment()
        if self._segment_fd is None:
            path = self.root / f"segment-{sequence:020d}.jsonl"
            self._segment_fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            os.chmod(path, 0o600)
            self._segment_start = sequence
            self._segment_size = path.stat().st_size

    def _flush_and_close_segment(self, *, sync: bool = True) -> None:
        if self._segment_fd is None:
            return
        if sync:
            self._timed_fdatasync(self._segment_fd)
        os.close(self._segment_fd)
        self._segment_fd = None
        self._segment_size = 0
        self._segment_start = 0

    def _write_checkpoint(self, snapshot: dict[str, JsonValue]) -> None:
        self._checkpoint_nonce += 1
        body: dict[str, JsonValue] = {
            "version": _VERSION,
            "sequence": self._last_sequence,
            "last_record_digest": self._last_digest,
            "created_at_ms": self._clock.now_ms(),
            "snapshot": snapshot,
        }
        document = {**body, "digest": _digest(body)}
        name = (
            f"checkpoint-{self._last_sequence:020d}-"
            f"{self._clock.now_ms():016d}-{self._checkpoint_nonce:06d}.json"
        )
        target = self.root / name
        temporary = self.root / f".{name}.tmp"
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            _write_all(fd, _canonical(document) + b"\n")
            self._timed_fdatasync(fd)
        finally:
            os.close(fd)
        self._inject("checkpoint_before_rename")
        os.replace(temporary, target)
        _fsync_directory(self.root)
        self._inject("checkpoint_after_rename")
        self._read_checkpoint(target)

        checkpoints: list[tuple[int, Path]] = []
        for path in self.root.glob("checkpoint-*.json"):
            try:
                sequence, _, _ = self._read_checkpoint(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            checkpoints.append((sequence, path))
        checkpoints.sort(key=lambda item: (item[0], item[1].name))
        while len(checkpoints) > 2:
            _, obsolete = checkpoints.pop(0)
            obsolete.unlink(missing_ok=True)
        self._verified_checkpoints = len(checkpoints)
        self._checkpoint_sequence = self._last_sequence
        self._records_since_checkpoint = 0
        self._last_checkpoint_ms = self._clock.now_ms()
        if len(checkpoints) == 2:
            self._compact_segments(checkpoints[0][0])

    def _compact_segments(self, safe_sequence: int) -> None:
        self._flush_and_close_segment()
        segments = sorted(self.root.glob("segment-*.jsonl"))
        starts = [int(path.stem.split("-")[1]) for path in segments]
        for index, path in enumerate(segments[:-1]):
            end_sequence = starts[index + 1] - 1
            if end_sequence > safe_sequence:
                continue
            self._inject("segment_before_delete")
            path.unlink(missing_ok=True)
            _fsync_directory(self.root)
            self._inject("segment_after_delete")

    def _write_manifest(self, last_sequence: int, last_digest: str) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        document = _manifest_document(last_sequence, last_digest, self._journal_id)
        temporary = self.root / ".manifest.tmp"
        target = self.root / _MANIFEST
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        try:
            _write_all(fd, _canonical(document) + b"\n")
            self._timed_fdatasync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, target)
        _fsync_directory(self.root)

    def _timed_fdatasync(self, fd: int) -> None:
        started = time.monotonic_ns()
        os.fdatasync(fd)
        elapsed = (time.monotonic_ns() - started) / 1_000_000
        self._sync_samples_ms.append(elapsed)
        if len(self._sync_samples_ms) > 1024:
            del self._sync_samples_ms[: len(self._sync_samples_ms) - 1024]

    def _fdatasync_p99(self) -> float | None:
        if not self._sync_samples_ms:
            return None
        if len(self._sync_samples_ms) == 1:
            return self._sync_samples_ms[0]
        return statistics.quantiles(self._sync_samples_ms, n=100, method="inclusive")[98]

    def _journal_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.iterdir() if path.is_file())

    def _admission_capacity_exhausted(self) -> bool:
        if not self.root.exists():
            return True
        return (
            self._journal_bytes() >= int(self._max_bytes * 0.8)
            or self._disk_usage(self.root) >= self._disk_stop_ratio
        )

    def _inject(self, boundary: str) -> None:
        if self._crash_injector is not None:
            self._crash_injector(boundary)
