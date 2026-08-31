"""Bounded nonblocking lifecycle spool with durable replay and strict ACK deletion."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.ids import EventId
from llmmaxxing.telemetry.events import MAX_EVENT_BYTES, LifecycleEventV1, canonical_event_bytes

MIN_SPOOL_BYTES = 8 * 1024**3
_ZERO_DIGEST = "0" * 64
_MANIFEST = "manifest-v1.json"
_ACK = "ack-v1.json"
_FATAL = "fatal.jsonl"
_RECORD_DOMAIN = b"llmmaxxing.lifecycle-spool.v1\x00"


class SpoolStatus(StrEnum):
    HEALTHY = "healthy"
    ADMISSION_STOP = "admission_stop"
    FAILED = "failed"
    CRASHED = "crashed"
    CLOSED = "closed"


class SpoolCorruptionError(RuntimeError):
    pass


class SpoolUnavailable(RuntimeError):
    pass


class SpoolInvariantError(RuntimeError):
    pass


class InjectedCrash(RuntimeError):
    """Deterministic hard-crash boundary for recovery tests."""


@dataclass(frozen=True, slots=True)
class SpoolSizing:
    rate_rps: int = 250
    outage_window_s: int = 600
    max_events_per_request: int = 12

    def __post_init__(self) -> None:
        if self.rate_rps < 1 or self.outage_window_s < 1 or self.max_events_per_request < 1:
            raise ValueError("spool sizing inputs must be positive")

    @property
    def formula_bytes(self) -> int:
        return self.rate_rps * self.outage_window_s * self.max_events_per_request * MAX_EVENT_BYTES

    @property
    def minimum_bytes(self) -> int:
        return max(MIN_SPOOL_BYTES, self.formula_bytes)

    def validate(self, configured_bytes: int, physical_bytes: int) -> None:
        if configured_bytes < self.minimum_bytes:
            raise ValueError(
                f"configured spool is {configured_bytes} bytes; requires {self.minimum_bytes}"
            )
        if physical_bytes < self.minimum_bytes:
            raise ValueError(
                f"physical spool is {physical_bytes} bytes; requires {self.minimum_bytes}"
            )
        if configured_bytes > physical_bytes:
            raise ValueError("configured spool exceeds the physical spool volume")


@dataclass(frozen=True, slots=True)
class AckCursor:
    segment: int
    sequence: int

    def __post_init__(self) -> None:
        if self.segment < 0 or self.sequence < 0:
            raise ValueError("ACK cursor cannot be negative")
        if (self.segment == 0) != (self.sequence == 0):
            raise ValueError("only the origin cursor may contain zero")

    @classmethod
    def origin(cls) -> Self:
        return cls(0, 0)


@dataclass(frozen=True, slots=True)
class SpoolGap:
    crash_gap: bool
    from_ms: int
    to_ms: int

    def __post_init__(self) -> None:
        if not self.crash_gap or self.from_ms < 1 or self.to_ms < self.from_ms:
            raise ValueError("invalid crash gap")


@dataclass(frozen=True, slots=True)
class SpoolRecord:
    sequence: int
    segment: int
    previous_digest: str
    digest: str
    event: LifecycleEventV1 | None = None
    gap: SpoolGap | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1 or self.segment < 1 or (self.event is None) == (self.gap is None):
            raise ValueError("invalid spool record")

    @property
    def cursor(self) -> AckCursor:
        return AckCursor(self.segment, self.sequence)


@dataclass(frozen=True, slots=True)
class SegmentManifest:
    index: int
    name: str
    first_sequence: int
    last_sequence: int
    first_event_id: EventId | None
    last_event_id: EventId | None
    digest: str
    bytes: int
    opened_at_ms: int
    open: bool


@dataclass(frozen=True, slots=True)
class SpoolStats:
    batches: int
    max_batch_records: int
    fdatasyncs: int
    max_fdatasync_interval_ms: float
    invariant_violations: int


@dataclass(slots=True)
class _MutableStats:
    batches: int = 0
    max_batch_records: int = 0
    fdatasyncs: int = 0
    max_fdatasync_interval_ms: float = 0.0
    invariant_violations: int = 0


@dataclass(slots=True)
class _ReservationState:
    remaining: int
    emitted: int = 0
    closed: bool = False


@dataclass(frozen=True, slots=True)
class SpoolReservation:
    """Frozen handle over writer-owned mutable accounting."""

    _spool: LifecycleSpool
    _state: _ReservationState
    budget_events: int

    @property
    def remaining_events(self) -> int:
        with self._spool._lock:
            return self._state.remaining

    @property
    def emitted_events(self) -> int:
        with self._spool._lock:
            return self._state.emitted

    def emit(self, event: LifecycleEventV1, *, terminal: bool = False) -> None:
        self._spool._submit(self._state, event, terminal=terminal)

    def release_unused(self, count: int | None = None) -> int:
        return self._spool._release_unused(self._state, count)


@dataclass(frozen=True, slots=True)
class _EventCommand:
    state: _ReservationState
    event: LifecycleEventV1
    terminal: bool


@dataclass(frozen=True, slots=True)
class _ControlCommand:
    kind: str
    done: threading.Event = field(default_factory=threading.Event)


_Command = _EventCommand | _ControlCommand


class LifecycleSpool:
    """One-process spool. Request-path methods take one lock and use ``put_nowait``."""

    def __init__(
        self,
        root: Path,
        *,
        create: bool,
        max_bytes: int,
        queue_capacity: int = 32_768,
        batch_records: int = 256,
        batch_delay_ms: int = 100,
        fdatasync_ms: int = 250,
        segment_bytes: int = 64 * 1024 * 1024,
        segment_seconds: int = 300,
        watermark_events: int = 12,
        clock_ms: Callable[[], int] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        crash_injector: Callable[[str], None] | None = None,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        if max_bytes < MAX_EVENT_BYTES or queue_capacity < 1:
            raise ValueError("spool and queue capacities must be positive")
        if not 1 <= batch_records <= 256 or not 1 <= batch_delay_ms <= 100:
            raise ValueError("writer batch exceeds the 256-record/100ms contract")
        if not 1 <= fdatasync_ms <= 250:
            raise ValueError("fdatasync cadence exceeds 250ms")
        if segment_bytes < 1 or not 1 <= segment_seconds <= 300 or watermark_events < 1:
            raise ValueError("invalid segment or watermark limit")

        self.root = root
        self.max_bytes = max_bytes
        self.queue_capacity = queue_capacity
        self._batch_records = batch_records
        self._batch_delay_s = batch_delay_ms / 1_000
        self._fdatasync_s = fdatasync_ms / 1_000
        self._sync_trigger_s = max(0.001, self._fdatasync_s - self._batch_delay_s)
        self._segment_bytes = segment_bytes
        self._segment_seconds = segment_seconds
        self._watermark_bytes = watermark_events * MAX_EVENT_BYTES
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._monotonic = monotonic
        self._crash_injector = crash_injector
        self._failure_injector = failure_injector
        # One serialized control slot cannot steal capacity reserved for live events.
        self._queue: queue.Queue[_Command] = queue.Queue(maxsize=queue_capacity + 1)
        self._lock = threading.RLock()
        self._control_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._status = SpoolStatus.HEALTHY
        self._accepting = True
        self._outstanding_slots = 0
        self._queued_slots = 0
        self._stats = _MutableStats()
        self._segments: list[SegmentManifest] = []
        self._physical_bytes = 0
        self._anchor_sequence = 0
        self._anchor_digest = _ZERO_DIGEST
        self._last_sequence = 0
        self._last_digest = _ZERO_DIGEST
        self._last_clean_at_ms = self._clock_ms()
        self._ack_cursor = AckCursor.origin()
        self._segment_fd: int | None = None
        self._segment_started_monotonic = self._monotonic()
        self._dirty_since: float | None = None
        self._crashed = False
        self._failed_terminal_events: list[LifecycleEventV1] = []

        self._initialize(create=create)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="llmmaxxing-lifecycle-spool",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def create(cls, root: Path, **kwargs: Any) -> Self:
        return cls(root, create=True, **kwargs)

    @classmethod
    def open(cls, root: Path, **kwargs: Any) -> Self:
        return cls(root, create=False, **kwargs)

    @property
    def status(self) -> SpoolStatus:
        with self._lock:
            if self._status is SpoolStatus.HEALTHY and self.usable_bytes < self._watermark_bytes:
                return SpoolStatus.ADMISSION_STOP
            return self._status

    @property
    def usable_bytes(self) -> int:
        with self._lock:
            return max(0, self.max_bytes - self.protected_bytes)

    @property
    def protected_bytes(self) -> int:
        with self._lock:
            return (
                self._physical_bytes
                + (self._outstanding_slots + self._queued_slots) * MAX_EVENT_BYTES
            )

    @property
    def backlog_bytes(self) -> int:
        with self._lock:
            return self._physical_bytes

    @property
    def outstanding_reserved_bytes(self) -> int:
        with self._lock:
            return (self._outstanding_slots + self._queued_slots) * MAX_EVENT_BYTES

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return self._queued_slots

    @property
    def ack_cursor(self) -> AckCursor:
        with self._lock:
            return self._ack_cursor

    @property
    def segment_manifests(self) -> tuple[SegmentManifest, ...]:
        with self._lock:
            return tuple(self._segments)

    @property
    def stats(self) -> SpoolStats:
        with self._lock:
            return SpoolStats(
                batches=self._stats.batches,
                max_batch_records=self._stats.max_batch_records,
                fdatasyncs=self._stats.fdatasyncs,
                max_fdatasync_interval_ms=self._stats.max_fdatasync_interval_ms,
                invariant_violations=self._stats.invariant_violations,
            )

    @property
    def failed_terminal_events(self) -> tuple[LifecycleEventV1, ...]:
        with self._lock:
            return tuple(self._failed_terminal_events)

    def try_reserve(self, events: int) -> SpoolReservation | None:
        if events < 1:
            raise ValueError("event reservation must be positive")
        with self._lock:
            live_slots = self._outstanding_slots + self._queued_slots
            if (
                not self._accepting
                or self._status is not SpoolStatus.HEALTHY
                or self.usable_bytes < self._watermark_bytes
                or self.usable_bytes < events * MAX_EVENT_BYTES
                or live_slots + events > self.queue_capacity
            ):
                return None
            state = _ReservationState(events)
            self._outstanding_slots += events
            return SpoolReservation(self, state, events)

    def _submit(
        self,
        state: _ReservationState,
        event: LifecycleEventV1,
        *,
        terminal: bool,
    ) -> None:
        with self._lock:
            if state.closed or state.remaining < 1:
                self._invariant("reservation_exhausted")
            if self._status in {SpoolStatus.FAILED, SpoolStatus.CRASHED} and terminal:
                state.remaining -= 1
                state.emitted += 1
                self._outstanding_slots -= 1
                state.closed = state.remaining == 0
                if len(self._failed_terminal_events) >= self.queue_capacity:
                    self._invariant("failed_terminal_capacity_exhausted")
                self._failed_terminal_events.append(event)
                self._write_fatal("terminal_not_durable")
                return
            if self._status in {SpoolStatus.FAILED, SpoolStatus.CRASHED, SpoolStatus.CLOSED}:
                self._invariant("writer_unavailable")
            state.remaining -= 1
            state.emitted += 1
            self._outstanding_slots -= 1
            self._queued_slots += 1
            if state.remaining == 0:
                state.closed = True
            try:
                self._queue.put_nowait(_EventCommand(state, event, terminal))
            except queue.Full as error:  # impossible when reservations are authoritative
                state.remaining += 1
                state.emitted -= 1
                state.closed = False
                self._outstanding_slots += 1
                self._queued_slots -= 1
                self._invariant("reserved_queue_full", cause=error)

    def _release_unused(self, state: _ReservationState, count: int | None) -> int:
        with self._lock:
            if count is not None and count < 0:
                raise ValueError("released event count cannot be negative")
            if state.remaining == 0:
                state.closed = True
                return 0
            released = state.remaining if count is None else count
            if released > state.remaining:
                self._invariant("reservation_release_exceeds_remaining")
            state.remaining -= released
            self._outstanding_slots -= released
            if state.remaining == 0:
                state.closed = True
            return released

    def _invariant(self, reason: str, *, cause: BaseException | None = None) -> None:
        self._stats.invariant_violations += 1
        self._accepting = False
        self._status = SpoolStatus.FAILED
        self._write_fatal(reason)
        error = SpoolInvariantError(reason)
        if cause is not None:
            raise error from cause
        raise error

    def flush(self, timeout: float = 5) -> None:
        with self._control_lock:
            with self._lock:
                if self._status in {SpoolStatus.FAILED, SpoolStatus.CRASHED}:
                    raise SpoolUnavailable(self._status.value)
                if self._status is SpoolStatus.CLOSED:
                    return
            command = _ControlCommand("flush")
            self._queue.put(command)
            if not command.done.wait(timeout):
                raise TimeoutError("lifecycle spool flush timed out")
            if not self._thread.is_alive():
                raise SpoolUnavailable("writer stopped during flush")

    def wait_stopped(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def close(self, timeout: float = 10) -> None:
        with self._close_lock, self._control_lock:
            with self._lock:
                if self._status is SpoolStatus.CLOSED:
                    return
                self._accepting = False
                failed = self._status in {SpoolStatus.FAILED, SpoolStatus.CRASHED}
            if failed:
                self._thread.join(timeout)
                self._close_fd()
                return
            command = _ControlCommand("stop")
            self._queue.put(command)
            if not command.done.wait(timeout):
                raise TimeoutError("lifecycle spool shutdown timed out")
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("lifecycle spool writer did not exit")
            with self._lock:
                self._status = SpoolStatus.CLOSED

    def replay(self, cursor: AckCursor | None = None) -> Iterator[SpoolRecord]:
        with self._lock:
            healthy = self._status is SpoolStatus.HEALTHY
        if healthy:
            self.flush()
        start = cursor or self.ack_cursor
        records, _, _, _ = self._scan_segments(repair_tail=False, update_manifests=False)
        return (record for record in records if record.sequence > start.sequence)

    def ack(self, cursor: AckCursor) -> None:
        self.flush()
        with self._lock:
            if cursor.sequence < self._ack_cursor.sequence:
                raise ValueError("ACK cursor cannot move backwards")
            if cursor.sequence > self._last_sequence:
                raise ValueError("ACK cursor exceeds the durable spool")
            if cursor != self._ack_cursor:
                records, _, _, _ = self._scan_segments(
                    repair_tail=False,
                    update_manifests=False,
                )
                if not any(record.cursor == cursor for record in records):
                    raise ValueError("ACK cursor does not identify a durable record")
            self._ack_cursor = cursor
            self._write_ack()
            current = self._segments[-1].index if self._segments else 0
            deleted = [
                segment
                for segment in self._segments
                if segment.index != current
                and segment.last_sequence > 0
                and segment.last_sequence < cursor.sequence
            ]
            if deleted:
                retained = [segment for segment in self._segments if segment not in deleted]
                renamed: list[tuple[Path, SegmentManifest]] = []
                try:
                    for segment in deleted:
                        source = self.root / segment.name
                        staged = source.with_name(source.name + ".acked")
                        os.replace(source, staged)
                        renamed.append((staged, segment))
                    self._fsync_directory()
                    newest = deleted[-1]
                    self._anchor_sequence = newest.last_sequence
                    self._anchor_digest = newest.digest
                    self._segments = retained
                    self._write_manifest(clean=False)
                    for staged, segment in renamed:
                        staged.unlink()
                        self._physical_bytes -= segment.bytes
                    self._fsync_directory()
                except OSError as error:
                    self._accepting = False
                    self._status = SpoolStatus.FAILED
                    self._write_fatal("ack_cleanup_failure")
                    raise SpoolUnavailable("durable ACK cleanup failed") from error

    def _initialize(self, *, create: bool) -> None:
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if any(self.root.iterdir()):
                raise SpoolCorruptionError("new spool directory is not empty")
            os.chmod(self.root, 0o700)
            self._write_manifest(clean=True)
            self._write_ack()
            was_clean = True
        else:
            if not self.root.is_dir():
                raise SpoolCorruptionError("missing spool directory")
            manifest = self._read_json(self.root / _MANIFEST, "manifest")
            if manifest.get("version") != 1 or not isinstance(manifest.get("clean"), bool):
                raise SpoolCorruptionError("invalid spool manifest")
            was_clean = bool(manifest["clean"])
            self._last_clean_at_ms = self._positive_int(
                manifest.get("last_clean_at_ms"), "last clean timestamp"
            )
            anchor = manifest.get("chain_anchor")
            if not isinstance(anchor, dict):
                raise SpoolCorruptionError("invalid chain anchor")
            self._anchor_sequence = self._nonnegative_int(anchor.get("sequence"), "anchor sequence")
            self._anchor_digest = self._digest(anchor.get("digest"), "anchor digest")
            raw_segments = manifest.get("segments")
            if not isinstance(raw_segments, list):
                raise SpoolCorruptionError("invalid segment manifest list")
            manifest_names: set[str] = set()
            for item in raw_segments:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise SpoolCorruptionError("invalid segment manifest entry")
                name = item["name"]
                if (
                    name in manifest_names
                    or not name.startswith("segment-")
                    or not name.endswith(".jsonl")
                ):
                    raise SpoolCorruptionError("invalid segment manifest name")
                manifest_names.add(name)
            self._repair_ack_deletions(manifest_names)
            self._ack_cursor = self._read_ack()

        records, partial, last_sequence, last_digest = self._scan_segments(
            repair_tail=not was_clean
        )
        self._last_sequence = last_sequence
        self._last_digest = last_digest
        self._physical_bytes = sum(segment.bytes for segment in self._segments)
        self._open_new_segment()
        if not was_clean:
            gap = SpoolGap(
                crash_gap=True,
                from_ms=self._last_clean_at_ms,
                to_ms=max(self._last_clean_at_ms, self._clock_ms()),
            )
            self._append_gap(gap)
            self._sync(force=True)
        elif partial:
            raise SpoolCorruptionError("clean spool contains an incomplete record")
        self._write_manifest(clean=False)

    def _writer_loop(self) -> None:
        stop: _ControlCommand | None = None
        try:
            while stop is None:
                try:
                    first = self._queue.get(timeout=min(self._batch_delay_s, self._fdatasync_s))
                except queue.Empty:
                    self._sync(force=False)
                    continue
                if isinstance(first, _ControlCommand):
                    if first.kind == "flush":
                        self._sync(force=True)
                        first.done.set()
                        continue
                    stop = first
                    break

                batch = [first]
                force = first.terminal
                deadline = self._monotonic() + self._batch_delay_s
                while len(batch) < self._batch_records and not force:
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        break
                    try:
                        command = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if isinstance(command, _ControlCommand):
                        if command.kind == "flush":
                            self._write_batch(batch)
                            batch = []
                            self._sync(force=True)
                            command.done.set()
                            continue
                        stop = command
                        break
                    batch.append(command)
                    force = command.terminal
                if batch:
                    self._write_batch(batch)
                self._sync(force=force)
            self._sync(force=True)
            self._close_current_segment()
            self._write_manifest(clean=True)
            if stop is not None:
                stop.done.set()
        except InjectedCrash:
            with self._lock:
                self._crashed = True
                self._accepting = False
                self._status = SpoolStatus.CRASHED
            if stop is not None:
                stop.done.set()
        except BaseException:
            with self._lock:
                self._accepting = False
                self._status = SpoolStatus.FAILED
            self._write_fatal("writer_io_failure")
            if stop is not None:
                stop.done.set()
            self._release_control_waiters()
        finally:
            if self._crashed:
                self._close_fd()

    def _write_batch(self, batch: list[_EventCommand]) -> None:
        if self._failure_injector is not None:
            self._failure_injector("before_batch")
        for command in batch:
            self._append_event(command.event)
            with self._lock:
                self._queued_slots -= 1
        with self._lock:
            self._stats.batches += 1
            self._stats.max_batch_records = max(self._stats.max_batch_records, len(batch))

    def _append_event(self, event: LifecycleEventV1) -> None:
        document = json.loads(canonical_event_bytes(event))
        self._append_payload("event", {"event": document}, event.event_id)

    def _append_gap(self, gap: SpoolGap) -> None:
        self._append_payload(
            "gap",
            {
                "gap": {
                    "crash_gap": gap.crash_gap,
                    "from_ms": gap.from_ms,
                    "to_ms": gap.to_ms,
                }
            },
            None,
        )

    def _append_payload(
        self,
        record_type: str,
        body: dict[str, object],
        event_id: EventId | None,
    ) -> None:
        sequence = self._last_sequence + 1
        segment = self._segments[-1].index
        payload: dict[str, object] = {
            "previous_digest": self._last_digest,
            "record_type": record_type,
            "segment": segment,
            "sequence": sequence,
            **body,
        }
        digest = hashlib.sha256(_RECORD_DOMAIN + canonical_json_bytes(payload)).hexdigest()
        line = canonical_json_bytes({**payload, "digest": digest}) + b"\n"
        self._rotate_if_needed(len(line))
        if self._segments[-1].index != segment:
            segment = self._segments[-1].index
            payload["segment"] = segment
            digest = hashlib.sha256(_RECORD_DOMAIN + canonical_json_bytes(payload)).hexdigest()
            line = canonical_json_bytes({**payload, "digest": digest}) + b"\n"
        if len(line) > MAX_EVENT_BYTES:
            raise SpoolInvariantError("spool record exceeds its reserved four-kib slot")
        fd = self._require_fd()
        if self._crash_injector is None:
            self._write_all(fd, line)
        else:
            split = max(1, len(line) // 2)
            self._write_all(fd, line[:split])
            self._crash_injector("after_partial_record")
            self._write_all(fd, line[split:])
        if self._dirty_since is None:
            self._dirty_since = self._monotonic()
        current = self._segments[-1]
        first_sequence = current.first_sequence or sequence
        first_event_id = current.first_event_id or event_id
        self._segments[-1] = replace(
            current,
            first_sequence=first_sequence,
            last_sequence=sequence,
            first_event_id=first_event_id,
            last_event_id=event_id or current.last_event_id,
            digest=digest,
            bytes=current.bytes + len(line),
        )
        with self._lock:
            self._physical_bytes += len(line)
            self._last_sequence = sequence
            self._last_digest = digest

    def _rotate_if_needed(self, line_bytes: int) -> None:
        current = self._segments[-1]
        age = self._monotonic() - self._segment_started_monotonic
        if current.bytes and (
            current.bytes + line_bytes > self._segment_bytes or age >= self._segment_seconds
        ):
            self._sync(force=True)
            self._close_current_segment()
            self._open_new_segment()

    def _open_new_segment(self) -> None:
        index = max((segment.index for segment in self._segments), default=0) + 1
        name = f"segment-{index:020d}.jsonl"
        path = self.root / name
        self._segment_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        now = self._clock_ms()
        self._segments.append(
            SegmentManifest(
                index=index,
                name=name,
                first_sequence=0,
                last_sequence=0,
                first_event_id=None,
                last_event_id=None,
                digest=self._last_digest,
                bytes=0,
                opened_at_ms=now,
                open=True,
            )
        )
        self._segment_started_monotonic = self._monotonic()

    def _close_current_segment(self) -> None:
        if self._segment_fd is None:
            return
        self._sync(force=True)
        os.close(self._segment_fd)
        self._segment_fd = None
        if self._segments:
            self._segments[-1] = replace(self._segments[-1], open=False)

    def _close_fd(self) -> None:
        fd = self._segment_fd
        self._segment_fd = None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _sync(self, *, force: bool) -> None:
        fd = self._segment_fd
        dirty_since = self._dirty_since
        if fd is None or dirty_since is None:
            return
        now = self._monotonic()
        if not force and now - dirty_since < self._sync_trigger_s:
            return
        if self._failure_injector is not None:
            self._failure_injector("before_fdatasync")
        os.fdatasync(fd)
        elapsed_ms = (self._monotonic() - dirty_since) * 1_000
        with self._lock:
            self._stats.fdatasyncs += 1
            self._stats.max_fdatasync_interval_ms = max(
                self._stats.max_fdatasync_interval_ms, elapsed_ms
            )
        self._dirty_since = None
        self._write_manifest(clean=False)

    def _scan_segments(
        self,
        *,
        repair_tail: bool,
        update_manifests: bool = True,
    ) -> tuple[list[SpoolRecord], bool, int, str]:
        paths = sorted(self.root.glob("segment-*.jsonl"))
        records: list[SpoolRecord] = []
        manifests: list[SegmentManifest] = []
        previous = self._anchor_digest
        sequence = self._anchor_sequence
        partial = False
        for path_index, path in enumerate(paths):
            try:
                index = int(path.stem.split("-")[1])
            except (IndexError, ValueError) as error:
                raise SpoolCorruptionError("invalid segment name") from error
            first_sequence = 0
            last_sequence = 0
            first_event_id: EventId | None = None
            last_event_id: EventId | None = None
            last_digest = previous
            mode = "r+b" if repair_tail and path_index == len(paths) - 1 else "rb"
            with path.open(mode) as handle:
                offset = 0
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        if mode != "r+b":
                            raise SpoolCorruptionError("incomplete record in a closed segment")
                        handle.truncate(offset)
                        handle.flush()
                        os.fdatasync(handle.fileno())
                        partial = True
                        break
                    record = self._decode_record(line[:-1], previous, sequence + 1, index)
                    records.append(record)
                    sequence = record.sequence
                    previous = record.digest
                    last_digest = record.digest
                    first_sequence = first_sequence or record.sequence
                    last_sequence = record.sequence
                    if record.event is not None:
                        first_event_id = first_event_id or record.event.event_id
                        last_event_id = record.event.event_id
                    offset += len(line)
            size = path.stat().st_size
            manifests.append(
                SegmentManifest(
                    index=index,
                    name=path.name,
                    first_sequence=first_sequence,
                    last_sequence=last_sequence,
                    first_event_id=first_event_id,
                    last_event_id=last_event_id,
                    digest=last_digest,
                    bytes=size,
                    opened_at_ms=max(1, int(path.stat().st_mtime_ns // 1_000_000)),
                    open=False,
                )
            )
        if update_manifests:
            with self._lock:
                self._segments = manifests
        return records, partial, sequence, previous

    @staticmethod
    def _decode_record(
        line: bytes,
        expected_previous: str,
        expected_sequence: int,
        expected_segment: int,
    ) -> SpoolRecord:
        try:
            document = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SpoolCorruptionError("invalid complete spool JSON") from error
        if not isinstance(document, dict):
            raise SpoolCorruptionError("spool record is not an object")
        digest = document.pop("digest", None)
        if (
            not isinstance(digest, str)
            or digest != hashlib.sha256(_RECORD_DOMAIN + canonical_json_bytes(document)).hexdigest()
        ):
            raise SpoolCorruptionError("spool digest mismatch")
        previous = document.get("previous_digest")
        sequence = document.get("sequence")
        segment = document.get("segment")
        if (
            previous != expected_previous
            or sequence != expected_sequence
            or segment != expected_segment
        ):
            raise SpoolCorruptionError("spool sequence or chain mismatch")
        record_type = document.get("record_type")
        if record_type == "event" and set(document) == {
            "previous_digest",
            "record_type",
            "segment",
            "sequence",
            "event",
        }:
            event_document = document["event"]
            event = LifecycleEventV1.model_validate(event_document)
            if json.loads(canonical_event_bytes(event)) != event_document:
                raise SpoolCorruptionError("event encoding is not canonical")
            return SpoolRecord(sequence, segment, previous, digest, event=event)
        if record_type == "gap" and set(document) == {
            "previous_digest",
            "record_type",
            "segment",
            "sequence",
            "gap",
        }:
            gap_document = document["gap"]
            if not isinstance(gap_document, dict) or set(gap_document) != {
                "crash_gap",
                "from_ms",
                "to_ms",
            }:
                raise SpoolCorruptionError("invalid gap marker")
            gap = SpoolGap(
                crash_gap=gap_document["crash_gap"],
                from_ms=gap_document["from_ms"],
                to_ms=gap_document["to_ms"],
            )
            return SpoolRecord(sequence, segment, previous, digest, gap=gap)
        raise SpoolCorruptionError("unknown spool record type or fields")

    def _write_manifest(self, *, clean: bool) -> None:
        if clean:
            self._last_clean_at_ms = self._clock_ms()
        document = {
            "version": 1,
            "clean": clean,
            "last_clean_at_ms": self._last_clean_at_ms,
            "chain_anchor": {
                "sequence": self._anchor_sequence,
                "digest": self._anchor_digest,
            },
            "segments": [
                {
                    "index": item.index,
                    "name": item.name,
                    "first_sequence": item.first_sequence,
                    "last_sequence": item.last_sequence,
                    "first_event_id": item.first_event_id,
                    "last_event_id": item.last_event_id,
                    "digest": item.digest,
                    "bytes": item.bytes,
                    "opened_at_ms": item.opened_at_ms,
                    "open": item.open and not clean,
                }
                for item in self._segments
            ],
        }
        self._atomic_json(self.root / _MANIFEST, document)

    def _read_ack(self) -> AckCursor:
        document = self._read_json(self.root / _ACK, "ACK cursor")
        if set(document) != {"segment", "sequence", "version"} or document["version"] != 1:
            raise SpoolCorruptionError("invalid ACK cursor")
        try:
            return AckCursor(document["segment"], document["sequence"])
        except (TypeError, ValueError) as error:
            raise SpoolCorruptionError("invalid ACK cursor") from error

    def _write_ack(self) -> None:
        self._atomic_json(
            self.root / _ACK,
            {
                "version": 1,
                "segment": self._ack_cursor.segment,
                "sequence": self._ack_cursor.sequence,
            },
        )

    def _atomic_json(self, path: Path, document: object) -> None:
        data = canonical_json_bytes(document) + b"\n"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            self._write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
        self._fsync_directory()

    @staticmethod
    def _read_json(path: Path, name: str) -> dict[str, Any]:
        try:
            document = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SpoolCorruptionError(f"invalid {name}") from error
        if not isinstance(document, dict):
            raise SpoolCorruptionError(f"invalid {name}")
        return document

    def _repair_ack_deletions(self, manifest_names: set[str]) -> None:
        changed = False
        for staged in self.root.glob("segment-*.jsonl.acked"):
            original = staged.with_name(staged.name.removesuffix(".acked"))
            if original.name in manifest_names:
                if original.exists():
                    raise SpoolCorruptionError("duplicate staged ACK segment")
                os.replace(staged, original)
            else:
                staged.unlink()
            changed = True
        if changed:
            self._fsync_directory()

    def _fsync_directory(self) -> None:
        directory = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _write_fatal(self, reason: str) -> None:
        try:
            line = (
                canonical_json_bytes(
                    {"kind": "writer_fatal", "occurred_at_ms": self._clock_ms(), "reason": reason}
                )
                + b"\n"
            )
            fd = os.open(self.root / _FATAL, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                self._write_all(fd, line)
                os.fdatasync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _release_control_waiters(self) -> None:
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(command, _ControlCommand):
                command.done.set()

    def _require_fd(self) -> int:
        if self._segment_fd is None:
            raise SpoolUnavailable("spool segment is closed")
        return self._segment_fd

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written < 1:
                raise OSError("short spool write")
            view = view[written:]

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SpoolCorruptionError(f"invalid {name}")
        return value

    @staticmethod
    def _nonnegative_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SpoolCorruptionError(f"invalid {name}")
        return value

    @staticmethod
    def _digest(value: object, name: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SpoolCorruptionError(f"invalid {name}")
        return value


def dedupe_replay(
    records: Iterable[SpoolRecord],
    seen_event_ids: set[EventId] | None = None,
) -> Iterator[SpoolRecord]:
    """Consumer-side helper mirroring Task 11's unique ``event_id`` constraint."""
    seen = seen_event_ids if seen_event_ids is not None else set()
    for record in records:
        if record.event is None:
            yield record
            continue
        if record.event.event_id in seen:
            continue
        seen.add(record.event.event_id)
        yield record
