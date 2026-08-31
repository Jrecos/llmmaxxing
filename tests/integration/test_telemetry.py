from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry import trace as api_trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState
from pydantic import ValidationError

from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    BundleHash,
    DeploymentGenerationId,
    EventId,
    GatewayBootId,
    InstallationId,
    PolicyRevisionId,
    RequestId,
    RouteGroupId,
)
from llmmaxxing.core.models import RequestProfile
from llmmaxxing.core.reasons import (
    DispatchCause,
    EndpointKind,
    Modality,
    RouteTrigger,
    TerminalOutcome,
)
from llmmaxxing.core.state_machines import KeyLifecycleState
from llmmaxxing.gateway.auth import AuthenticatedClient
from llmmaxxing.gateway.telemetry import TelemetryLifecycleCapacity
from llmmaxxing.telemetry.events import (
    MAX_EVENT_BYTES,
    LifecycleEventKind,
    LifecycleEventV1,
    LifecycleReason,
    LifecycleTimingsV1,
    canonical_event_bytes,
    emit_event_schema,
    load_event,
)
from llmmaxxing.telemetry.metrics import TelemetryMetrics
from llmmaxxing.telemetry.otel import OptionalOtel, deterministic_head_sample
from llmmaxxing.telemetry.writer import (
    MIN_SPOOL_BYTES,
    AckCursor,
    InjectedCrash,
    LifecycleSpool,
    SpoolCorruptionError,
    SpoolSizing,
    SpoolStatus,
    dedupe_replay,
)

KEY_ID = "1" * 32
INSTALLATION_ID = InstallationId("inst_00000000-0000-4000-8000-000000000001")
BOOT_ID = GatewayBootId("boot_00000000-0000-4000-8000-000000000002")
REQUEST_ID = RequestId("req_00000000-0000-4000-8000-000000000003")
ROUTE_ID = RouteGroupId("rg_00000000-0000-4000-8000-000000000004")
ACCOUNT_ID = AccountId("acc_00000000-0000-4000-8000-000000000005")
OTHER_ACCOUNT_ID = AccountId("acc_00000000-0000-4000-8000-000000000006")
GENERATION_ID = DeploymentGenerationId.from_digest("a" * 64)


class Clock:
    def __init__(self, value: int = 1_000_000) -> None:
        self.value = value
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        with self._lock:
            self.value += 1
            return self.value


@dataclass(frozen=True, slots=True)
class FakeExporter(SpanExporter):
    fail: bool = False
    block: threading.Event | None = None
    exported: list[object] | None = None

    def export(self, spans):  # type: ignore[no-untyped-def]
        if self.block is not None:
            self.block.wait(timeout=2)
        if self.exported is not None:
            self.exported.extend(spans)
        if self.fail:
            raise RuntimeError("collector unavailable")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

@dataclass(frozen=True, slots=True)
class HungExporter(SpanExporter):
    entered: threading.Event
    release: threading.Event

    def export(self, spans):  # type: ignore[no-untyped-def]
        self.entered.set()
        self.release.wait()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.release.wait()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return False


def make_event(
    kind: LifecycleEventKind = LifecycleEventKind.REQUEST_ADMITTED,
    *,
    event_id: EventId | None = None,
    request_id: RequestId = REQUEST_ID,
    outcome: TerminalOutcome | None = None,
    reason: LifecycleReason | None = None,
) -> LifecycleEventV1:
    attempt = kind in {
        LifecycleEventKind.ATTEMPT_RESERVED,
        LifecycleEventKind.ATTEMPT_RESOLVED,
    }
    resolved = kind is LifecycleEventKind.ATTEMPT_RESOLVED
    terminal = kind is LifecycleEventKind.REQUEST_TERMINAL
    return LifecycleEventV1(
        event_id=event_id or EventId.new(),
        request_id=request_id,
        attempt_id=AttemptId.new() if attempt else None,
        kind=kind,
        occurred_at_ms=1_000_001,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        key_id=KEY_ID,
        route_group_id=None if kind is LifecycleEventKind.REQUEST_ADMITTED else ROUTE_ID,
        account_id=ACCOUNT_ID if attempt else None,
        deployment_generation_id=GENERATION_ID if attempt else None,
        trigger=RouteTrigger.PRIMARY if attempt else None,
        outcome=(
            outcome
            if outcome is not None
            else TerminalOutcome.COMPLETED
            if terminal or resolved
            else None
        ),
        reason=reason,
        timings_ms=LifecycleTimingsV1(duration_ms=7) if resolved or terminal else None,
        uncertain=False if resolved else None,
        lease_released_at_ms=1_000_001 if resolved else None,
        attempts_used=1 if terminal else None,
    )


def client() -> AuthenticatedClient:
    return AuthenticatedClient(
        key_id=KEY_ID,
        accepted_credential_generation=1,
        policy_id=PolicyRevisionId.new(),
        key_state=KeyLifecycleState.ENABLED,
        key_expires_at_s=10_000,
        applied_bundle_generation=1,
        applied_bundle_hash=BundleHash.from_digest("b" * 64),
    )


def profile() -> RequestProfile:
    return RequestProfile(
        route_group_id=ROUTE_ID,
        model_alias="never-export-this-alias",
        endpoint=EndpointKind.CHAT,
        modality=Modality.TEXT,
        stream=True,
        input_tokens_max=12,
        output_tokens_max=8,
        reasoning_tokens_max=4,
        tools_count=1,
        forced_tool_required=False,
        response_schema_present=False,
        history_turns=2,
        deadline_ms=30_000,
    )


def dispatch(
    *,
    account_id: AccountId = ACCOUNT_ID,
    cause: DispatchCause = DispatchCause.PRIMARY,
) -> SimpleNamespace:
    request = SimpleNamespace(profile=profile())
    return SimpleNamespace(
        attempt_id=AttemptId.new(),
        candidate=SimpleNamespace(
            account_id=account_id,
            generation_id=GENERATION_ID,
            cause=cause,
        ),
        lease=SimpleNamespace(request=request),
    )


def spool(tmp_path: Path, **kwargs: object) -> LifecycleSpool:
    return LifecycleSpool.create(
        tmp_path,
        max_bytes=int(kwargs.pop("max_bytes", 512 * 4096)),
        queue_capacity=int(kwargs.pop("queue_capacity", 128)),
        segment_bytes=int(kwargs.pop("segment_bytes", 64 * 1024 * 1024)),
        segment_seconds=int(kwargs.pop("segment_seconds", 300)),
        **kwargs,
    )


def test_event_schema_is_closed_private_canonical_and_checked_in() -> None:
    event = make_event(
        LifecycleEventKind.ATTEMPT_RESOLVED,
        outcome=TerminalOutcome.CLIENT_CANCELLED,
        reason=LifecycleReason.CLIENT_CANCELLED,
    )
    encoded = canonical_event_bytes(event)
    assert encoded == canonical_event_bytes(load_event(encoded))
    assert len(encoded) <= MAX_EVENT_BYTES
    assert json.loads(encoded)["schema_version"] == 1

    lower = encoded.lower()
    for sentinel in (
        b"prompt-secret",
        b"request-body",
        b"tool-argument",
        b"tool-result",
        b"bearer credential",
        b"never-export-this-alias",
        b"raw provider error",
        b"192.0.2.42",
        b"arbitrary_metadata",
    ):
        assert sentinel not in lower

    document = event.model_dump(mode="json")
    document["arbitrary_metadata"] = {"prompt-secret": "leak"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LifecycleEventV1.model_validate(document)
    document.pop("arbitrary_metadata")
    document["reason"] = "raw provider error"
    with pytest.raises(ValidationError):
        LifecycleEventV1.model_validate(document)

    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "events-v1.json"
    assert schema_path.read_bytes() == emit_event_schema()


def test_event_kind_invariants_reject_missing_or_cross_kind_fields() -> None:
    base = make_event().model_dump(mode="python")
    base["outcome"] = TerminalOutcome.COMPLETED
    with pytest.raises(ValidationError, match="terminal fields"):
        LifecycleEventV1.model_validate(base)

    resolved = make_event(LifecycleEventKind.ATTEMPT_RESOLVED).model_dump(mode="python")
    resolved["account_id"] = None
    with pytest.raises(ValidationError, match="attempt identity"):
        LifecycleEventV1.model_validate(resolved)

    terminal = make_event(LifecycleEventKind.REQUEST_TERMINAL).model_dump(mode="python")
    terminal["attempt_id"] = AttemptId.new()
    with pytest.raises(ValidationError, match="attempt fields"):
        LifecycleEventV1.model_validate(terminal)


def test_production_lifecycle_reserves_exact_budget_and_one_terminal(tmp_path: Path) -> None:
    writer = spool(tmp_path, queue_capacity=10)
    metrics = TelemetryMetrics(account_ids=(ACCOUNT_ID,), route_group_ids=(ROUTE_ID,))
    capacity = TelemetryLifecycleCapacity(
        writer,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        clock=Clock(),
        metrics=metrics,
    )

    async def scenario() -> None:
        lifecycle = await capacity.reserve(REQUEST_ID, client(), 10)
        assert lifecycle is not None
        assert lifecycle.budget_events == 10
        assert writer.outstanding_reserved_bytes == 10 * MAX_EVENT_BYTES
        await lifecycle.profile_accepted(profile())
        await lifecycle.queued()
        for _ in range(3):
            lease = dispatch()
            await lifecycle.attempt_started(lease, shadow=False)
            lifecycle.attempt_headers(lease)
            lifecycle.attempt_first_byte(lease)
            await lifecycle.attempt_finished(
                lease,
                TerminalOutcome.UPSTREAM_FAILED,
                uncertain=False,
                capacity_released=True,
            )
        with pytest.raises(RuntimeError, match="attempt budget"):
            await lifecycle.attempt_started(dispatch(), shadow=False)
        await lifecycle.finished(TerminalOutcome.UPSTREAM_FAILED)
        await lifecycle.finished(TerminalOutcome.COMPLETED)
        await lifecycle.release()
        assert lifecycle.emitted_events == 10
        await capacity.aclose()
        assert writer.outstanding_reserved_bytes == 0

    import asyncio

    asyncio.run(scenario())
    reopened = LifecycleSpool.open(tmp_path, max_bytes=512 * 4096)
    try:
        records = tuple(reopened.replay())
        events = [record.event for record in records if record.event is not None]
        assert len(events) == 10
        assert sum(event.kind is LifecycleEventKind.REQUEST_TERMINAL for event in events) == 1
        assert events[-1].outcome is TerminalOutcome.UPSTREAM_FAILED
        resolved = [event for event in events if event.kind is LifecycleEventKind.ATTEMPT_RESOLVED]
        assert all(event.headers_at_ms is not None for event in resolved)
        assert all(event.first_byte_at_ms is not None for event in resolved)
        assert all(event.timings_ms is not None and event.timings_ms.ttft_ms == 2 for event in resolved)
        assert all(len(canonical_event_bytes(event)) <= MAX_EVENT_BYTES for event in events)
    finally:
        reopened.close()


def test_shadow_budget_cancel_and_spill_are_bounded(tmp_path: Path) -> None:
    writer = spool(tmp_path, queue_capacity=16)
    metrics = TelemetryMetrics(
        account_ids=(ACCOUNT_ID, OTHER_ACCOUNT_ID), route_group_ids=(ROUTE_ID,)
    )
    capacity = TelemetryLifecycleCapacity(
        writer,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        clock=Clock(),
        metrics=metrics,
    )

    async def scenario() -> None:
        lifecycle = await capacity.reserve(REQUEST_ID, client(), 12)
        assert lifecycle is not None and lifecycle.budget_events == 12
        await lifecycle.profile_accepted(profile())
        await lifecycle.queued()
        primary = dispatch()
        await lifecycle.attempt_started(primary, shadow=False)
        await lifecycle.attempt_finished(
            primary,
            TerminalOutcome.UPSTREAM_FAILED,
            uncertain=False,
            capacity_released=True,
        )
        spill = dispatch(account_id=OTHER_ACCOUNT_ID, cause=DispatchCause.CAPACITY)
        await lifecycle.attempt_started(spill, shadow=False)
        await lifecycle.attempt_finished(
            spill,
            TerminalOutcome.CLIENT_CANCELLED,
            uncertain=True,
            capacity_released=False,
        )
        await lifecycle.finished(TerminalOutcome.CLIENT_CANCELLED)
        await lifecycle.release()
        await capacity.aclose()

    import asyncio

    asyncio.run(scenario())
    reopened = LifecycleSpool.open(tmp_path, max_bytes=512 * 4096)
    try:
        events = [record.event for record in reopened.replay() if record.event is not None]
        resolved = [event for event in events if event.kind is LifecycleEventKind.ATTEMPT_RESOLVED]
        assert resolved[-1].uncertain is True
        assert resolved[-1].reason is LifecycleReason.CLIENT_CANCELLED
        assert resolved[-1].spill_from_account_id == ACCOUNT_ID
        assert resolved[-1].lease_released_at_ms is None
        assert sum(event.kind is LifecycleEventKind.REQUEST_TERMINAL for event in events) == 1
    finally:
        reopened.close()


def test_reserved_watermark_stops_only_new_admission_without_overwrite(tmp_path: Path) -> None:
    writer = spool(tmp_path, max_bytes=24 * MAX_EVENT_BYTES, queue_capacity=24)
    capacity = TelemetryLifecycleCapacity(
        writer,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        clock=Clock(),
    )
    first = writer.try_reserve(12)
    second = writer.try_reserve(12)
    assert first is not None and second is not None
    assert writer.try_reserve(10) is None
    assert writer.status is SpoolStatus.ADMISSION_STOP
    assert capacity.ready is True
    import asyncio

    assert asyncio.run(capacity.reserve(RequestId.new(), client(), 10)) is None

    expected = make_event(event_id=EventId.new())
    second.emit(expected)
    second.release_unused()
    writer.flush()
    assert [record.event for record in writer.replay() if record.event is not None] == [expected]

    first.release_unused()
    third = writer.try_reserve(10)
    assert third is not None
    third.release_unused()
    assert [record.event for record in writer.replay() if record.event is not None] == [expected]
    asyncio.run(capacity.aclose())


def test_replay_is_at_least_once_deduped_and_ack_is_strict(tmp_path: Path) -> None:
    writer = spool(tmp_path, segment_bytes=900, queue_capacity=32)
    reservation = writer.try_reserve(12)
    assert reservation is not None
    for _ in range(8):
        reservation.emit(make_event(event_id=EventId.new()))
    reservation.release_unused()
    writer.flush()

    original = tuple(writer.replay(AckCursor.origin()))
    assert len(original) == 8
    twice = tuple(dedupe_replay((*original, *original)))
    assert [record.event.event_id for record in twice if record.event is not None] == [
        record.event.event_id for record in original if record.event is not None
    ]

    assert writer.segment_manifests[-1].open is True
    with pytest.raises(ValueError, match="durable record"):
        writer.ack(AckCursor(original[-1].segment + 100, original[-1].sequence))

    segments = writer.segment_manifests
    assert len(segments) >= 2
    first_boundary = next(
        record.cursor for record in original if record.segment == segments[0].index
    )
    first_boundary = AckCursor(first_boundary.segment, segments[0].last_sequence)
    first_path = tmp_path / segments[0].name
    writer.ack(first_boundary)
    assert first_path.exists(), "a segment at the ACK boundary is not strictly below it"

    writer.ack(original[-1].cursor)
    assert not first_path.exists()
    assert tuple(writer.replay()) == ()
    assert tuple(writer.replay(AckCursor.origin())), "a lost ACK must redeliver"
    writer.close()

def test_replay_and_ack_decode_incrementally_outside_hot_accounting_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = spool(tmp_path, segment_bytes=1_200)
    reservation = writer.try_reserve(6)
    assert reservation is not None
    for _ in range(6):
        reservation.emit(make_event(event_id=EventId.new()))
    reservation.release_unused()
    writer.flush()
    cursor = AckCursor(
        writer.segment_manifests[-1].index,
        writer.segment_manifests[-1].last_sequence,
    )

    original = LifecycleSpool._decode_record
    decoded = 0

    def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal decoded
        decoded += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(LifecycleSpool, "_decode_record", staticmethod(counted))
    replay = writer.replay(AckCursor.origin())
    assert decoded == 0
    assert next(replay).event is not None
    assert decoded == 1
    tuple(replay)
    decoded_before_ack = decoded
    writer.ack(cursor)
    assert decoded == decoded_before_ack
    writer.close()


def test_unclean_partial_tail_becomes_explicit_gap_and_clean_close_does_not(
    tmp_path: Path,
) -> None:
    fired = False

    def crash(boundary: str) -> None:
        nonlocal fired
        if boundary == "after_partial_record" and not fired:
            fired = True
            raise InjectedCrash("power loss")

    crashed = spool(tmp_path / "crash", crash_injector=crash)
    reservation = crashed.try_reserve(1)
    assert reservation is not None
    reservation.emit(make_event())
    assert crashed.wait_stopped(2)
    crashed.close()

    recovered = LifecycleSpool.open(tmp_path / "crash", max_bytes=512 * 4096)
    try:
        replay = tuple(recovered.replay(AckCursor.origin()))
        assert len(replay) == 1
        assert replay[0].gap is not None
        assert replay[0].gap.crash_gap is True
        assert replay[0].gap.to_ms >= replay[0].gap.from_ms
    finally:
        recovered.close()

    clean = spool(tmp_path / "clean")
    reserved = clean.try_reserve(1)
    assert reserved is not None
    reserved.emit(make_event())
    reserved.release_unused()
    clean.close()
    reopened = LifecycleSpool.open(tmp_path / "clean", max_bytes=512 * 4096)
    try:
        assert all(record.gap is None for record in reopened.replay(AckCursor.origin()))
    finally:
        reopened.close()


def test_complete_corruption_fails_closed_instead_of_becoming_a_gap(tmp_path: Path) -> None:
    writer = spool(tmp_path)
    reservation = writer.try_reserve(1)
    assert reservation is not None
    reservation.emit(make_event())
    writer.close()

    segment = next(tmp_path.glob("segment-*.jsonl"))
    content = segment.read_bytes()
    segment.write_bytes(content.replace(b"request_admitted", b"request_admitteX", 1))
    with pytest.raises(SpoolCorruptionError):
        LifecycleSpool.open(tmp_path, max_bytes=512 * 4096)

def test_recovery_rejects_missing_manifest_segment_and_forward_ack(tmp_path: Path) -> None:
    missing = spool(tmp_path / "missing", segment_bytes=900)
    reservation = missing.try_reserve(4)
    assert reservation is not None
    for _ in range(4):
        reservation.emit(make_event(event_id=EventId.new()))
    missing.close()
    newest = sorted((tmp_path / "missing").glob("segment-*.jsonl"))[-1]
    newest.unlink()
    with pytest.raises(SpoolCorruptionError, match="manifest segment"):
        LifecycleSpool.open(tmp_path / "missing", max_bytes=512 * 4096)

    forward = spool(tmp_path / "forward")
    reserved = forward.try_reserve(1)
    assert reserved is not None
    reserved.emit(make_event())
    forward.close()
    (tmp_path / "forward" / "ack-v1.json").write_text(
        json.dumps({"segment": 99, "sequence": 99, "version": 1}, sort_keys=True) + "\n"
    )
    with pytest.raises(SpoolCorruptionError, match="ACK"):
        LifecycleSpool.open(tmp_path / "forward", max_bytes=512 * 4096)


def test_writer_failure_stops_admission_but_records_one_live_terminal(tmp_path: Path) -> None:
    def fail(boundary: str) -> None:
        if boundary == "before_batch":
            raise OSError("raw disk error must not enter telemetry")

    writer = spool(tmp_path, failure_injector=fail)
    metrics = TelemetryMetrics(account_ids=(ACCOUNT_ID,), route_group_ids=(ROUTE_ID,))
    capacity = TelemetryLifecycleCapacity(
        writer,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        clock=Clock(),
        metrics=metrics,
    )

    async def scenario() -> None:
        lifecycle = await capacity.reserve(REQUEST_ID, client(), 10)
        assert lifecycle is not None
        assert writer.wait_stopped(2)
        assert writer.status is SpoolStatus.FAILED
        assert await capacity.reserve(RequestId.new(), client(), 10) is None
        await lifecycle.finished(TerminalOutcome.UPSTREAM_FAILED)
        await lifecycle.finished(TerminalOutcome.COMPLETED)
        await lifecycle.release()
        assert len(writer.failed_terminal_events) == 1
        await capacity.aclose()

    import asyncio

    asyncio.run(scenario())
    fatal = (tmp_path / "fatal.jsonl").read_text()
    assert "writer_io_failure" in fatal
    assert "raw disk error" not in fatal

@pytest.mark.parametrize("batch_records", (1, 256))
def test_writer_failure_preserves_terminal_from_failing_batch_or_drained_queue(
    tmp_path: Path,
    batch_records: int,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def fail(boundary: str) -> None:
        if boundary == "before_batch":
            entered.set()
            release.wait(timeout=2)
            raise OSError("disk unavailable")

    writer = spool(
        tmp_path,
        queue_capacity=16,
        batch_records=batch_records,
        failure_injector=fail,
    )
    capacity = TelemetryLifecycleCapacity(
        writer,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        clock=Clock(),
    )

    async def scenario() -> None:
        lifecycle = await capacity.reserve(REQUEST_ID, client(), 10)
        assert lifecycle is not None
        await lifecycle.profile_accepted(profile())
        await lifecycle.queued()
        await lifecycle.finished(TerminalOutcome.UPSTREAM_FAILED)
        assert entered.wait(timeout=1)
        release.set()
        assert await asyncio.to_thread(writer.wait_stopped, 2)
        terminals = writer.failed_terminal_events
        assert len(terminals) == 1
        assert terminals[0].kind is LifecycleEventKind.REQUEST_TERMINAL
        assert terminals[0].outcome is TerminalOutcome.UPSTREAM_FAILED
        await lifecycle.release()
        await capacity.aclose()

    import asyncio

    asyncio.run(scenario())


def test_full_spool_records_fit_the_reserved_four_kib_slot(tmp_path: Path) -> None:
    writer = spool(tmp_path)
    reservation = writer.try_reserve(1)
    assert reservation is not None
    reservation.emit(make_event(LifecycleEventKind.ATTEMPT_RESOLVED))
    writer.close()
    lines = [
        line
        for segment in tmp_path.glob("segment-*.jsonl")
        for line in segment.read_bytes().splitlines()
    ]
    assert lines and all(len(line) <= MAX_EVENT_BYTES for line in lines)


def test_spool_sizing_rejects_below_floor_formula_or_physical_volume() -> None:
    defaults = SpoolSizing()
    assert defaults.formula_bytes == 250 * 600 * 12 * MAX_EVENT_BYTES
    assert defaults.minimum_bytes == MIN_SPOOL_BYTES == 8 * 1024**3
    defaults.validate(MIN_SPOOL_BYTES, MIN_SPOOL_BYTES)
    with pytest.raises(ValueError, match="configured spool"):
        defaults.validate(MIN_SPOOL_BYTES - 1, MIN_SPOOL_BYTES)
    with pytest.raises(ValueError, match="physical spool"):
        defaults.validate(MIN_SPOOL_BYTES, MIN_SPOOL_BYTES - 1)
    with pytest.raises(ValueError, match="at least 12"):
        SpoolSizing(max_events_per_request=11)

    larger = SpoolSizing(rate_rps=1_000, outage_window_s=900, max_events_per_request=12)
    assert larger.minimum_bytes == larger.formula_bytes > MIN_SPOOL_BYTES
    with pytest.raises(ValueError, match="configured spool"):
        larger.validate(MIN_SPOOL_BYTES, larger.minimum_bytes)
    with pytest.raises(ValueError, match="configured spool"):
        TelemetryLifecycleCapacity.start(
            Path("/tmp/unused-undersized-lifecycle-spool"),
            installation_id=INSTALLATION_ID,
            boot_id=BOOT_ID,
            max_bytes=MIN_SPOOL_BYTES - 1,
            physical_bytes=MIN_SPOOL_BYTES,
        )

def test_writer_activity_updates_prometheus_after_async_idle(tmp_path: Path) -> None:
    metrics = TelemetryMetrics(account_ids=(ACCOUNT_ID,), route_group_ids=(ROUTE_ID,))
    writer = spool(tmp_path)
    capacity = TelemetryLifecycleCapacity(
        writer,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        clock=Clock(),
        metrics=metrics,
    )

    async def scenario() -> None:
        lifecycle = await capacity.reserve(REQUEST_ID, client(), 10)
        assert lifecycle is not None
        await lifecycle.finished(TerminalOutcome.UPSTREAM_FAILED)
        await lifecycle.release()
        await capacity.aclose()

    import asyncio

    asyncio.run(scenario())
    samples = {
        line.split()[0]: float(line.split()[1])
        for line in metrics.render().decode().splitlines()
        if line and not line.startswith("#") and "{" not in line
    }
    assert samples["llmmaxxing_writer_batch_records_count"] >= 1
    assert samples["llmmaxxing_writer_fdatasync_seconds_count"] >= 1
    assert samples["llmmaxxing_spool_bytes"] == writer.backlog_bytes


def test_deterministic_otel_sampling_and_exporter_outage_never_touch_spool(
    tmp_path: Path,
) -> None:
    sampled = RequestId("req_00000000-0000-4000-8000-000000000003")
    unsampled = RequestId("req_00000000-0000-4000-8000-000000000001")
    assert deterministic_head_sample(sampled) is True
    assert deterministic_head_sample(sampled) is True
    assert deterministic_head_sample(unsampled) is False

    metrics = TelemetryMetrics(account_ids=(ACCOUNT_ID,), route_group_ids=(ROUTE_ID,))
    otel = OptionalOtel(
        metrics,
        exporter=FakeExporter(fail=True),
        queue_size=8,
        batch_size=1,
        export_interval_ms=1,
    )
    writer = spool(tmp_path)
    capacity = TelemetryLifecycleCapacity(
        writer,
        installation_id=INSTALLATION_ID,
        boot_id=BOOT_ID,
        clock=Clock(),
        metrics=metrics,
        otel=otel,
    )

    async def scenario() -> None:
        lifecycle = await capacity.reserve(sampled, client(), 10)
        assert lifecycle is not None
        await lifecycle.profile_accepted(profile())
        await lifecycle.queued()
        lease = dispatch()
        await lifecycle.attempt_started(lease, shadow=False)
        await lifecycle.attempt_finished(
            lease,
            TerminalOutcome.COMPLETED,
            uncertain=False,
            capacity_released=True,
        )
        await lifecycle.finished(TerminalOutcome.COMPLETED)
        await lifecycle.release()
        otel.flush(2)
        assert otel.failed_exports >= 1
        assert writer.status in {SpoolStatus.HEALTHY, SpoolStatus.ADMISSION_STOP}
        await capacity.aclose()

    import asyncio

    asyncio.run(scenario())
    reopened = LifecycleSpool.open(tmp_path, max_bytes=512 * 4096)
    try:
        events = [record.event for record in reopened.replay() if record.event is not None]
        assert events and events[-1].kind is LifecycleEventKind.REQUEST_TERMINAL
    finally:
        reopened.close()


def test_otel_queue_is_bounded_drop_oldest_and_metadata_only() -> None:
    released = threading.Event()
    exported: list[object] = []
    metrics = TelemetryMetrics(account_ids=(ACCOUNT_ID,), route_group_ids=(ROUTE_ID,))
    otel = OptionalOtel(
        metrics,
        exporter=FakeExporter(block=released, exported=exported),
        queue_size=2,
        batch_size=1,
        export_interval_ms=1,
    )
    sampled = RequestId("req_00000000-0000-4000-8000-000000000003")
    for _ in range(20):
        otel.request_started(sampled)
        otel.request_finished(sampled, ROUTE_ID, TerminalOutcome.COMPLETED)
    deadline = time.monotonic() + 1
    while otel.dropped_spans == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert otel.dropped_spans > 0
    released.set()
    otel.flush(2)
    otel.shutdown()

    serialized = " ".join(
        f"{span.name} {dict(span.attributes or {})}"
        for span in exported  # type: ignore[attr-defined]
    ).lower()
    for banned in ("prompt", "body", "tool_argument", "credential", "model_alias", "client_ip"):
        assert banned not in serialized


def test_sampled_request_ignores_ambient_trace_context() -> None:
    exported: list[object] = []
    metrics = TelemetryMetrics(account_ids=(ACCOUNT_ID,), route_group_ids=(ROUTE_ID,))
    otel = OptionalOtel(
        metrics,
        exporter=FakeExporter(exported=exported),
        queue_size=8,
        batch_size=1,
        export_interval_ms=1,
    )
    sampled = RequestId("req_00000000-0000-4000-8000-000000000003")
    ambient = NonRecordingSpan(
        SpanContext(
            trace_id=0x123456789ABCDEF123456789ABCDEF,
            span_id=0x123456789ABCDEF,
            is_remote=True,
            trace_flags=TraceFlags(0),
            trace_state=TraceState(),
        )
    )
    with api_trace.use_span(ambient, end_on_exit=False):
        otel.request_started(sampled)
        otel.request_finished(sampled, ROUTE_ID, TerminalOutcome.COMPLETED)
    assert otel.flush(2)
    otel.shutdown()
    roots = [span for span in exported if span.name == "llmmaxxing.request"]  # type: ignore[attr-defined]
    assert len(roots) == 1
    assert roots[0].parent is None  # type: ignore[attr-defined]


def test_hung_optional_otel_exporter_cannot_block_shutdown() -> None:
    entered = threading.Event()
    release = threading.Event()
    metrics = TelemetryMetrics()
    otel = OptionalOtel(
        metrics,
        exporter=HungExporter(entered, release),
        queue_size=2,
        batch_size=1,
        export_interval_ms=1,
    )
    sampled = RequestId("req_00000000-0000-4000-8000-000000000003")
    otel.request_started(sampled)
    otel.request_finished(sampled, ROUTE_ID, TerminalOutcome.COMPLETED)
    assert entered.wait(timeout=1)

    returned = threading.Event()
    shutdown = threading.Thread(target=lambda: (otel.shutdown(), returned.set()), daemon=True)
    shutdown.start()
    bounded = returned.wait(timeout=0.75)
    release.set()
    shutdown.join(timeout=3)
    assert bounded, "optional exporter shutdown must have a short outer deadline"
