from __future__ import annotations

import time
import uuid
from pathlib import Path

from llmmaxxing.core.ids import (
    AccountId,
    EventId,
    GatewayBootId,
    InstallationId,
    RequestId,
    RouteGroupId,
)
from llmmaxxing.core.reasons import RouteTrigger, TerminalOutcome
from llmmaxxing.telemetry.events import LifecycleEventKind, LifecycleEventV1
from llmmaxxing.telemetry.metrics import MAX_SCRAPE_BYTES, MAX_SERIES, TelemetryMetrics
from llmmaxxing.telemetry.writer import LifecycleSpool


def uuid4_at(value: int) -> str:
    return str(uuid.UUID(int=value, version=4))


def accounts(count: int) -> tuple[AccountId, ...]:
    return tuple(AccountId(f"acc_{uuid4_at(index + 1)}") for index in range(count))


def routes(count: int) -> tuple[RouteGroupId, ...]:
    return tuple(RouteGroupId(f"rg_{uuid4_at(index + 1_000)}") for index in range(count))


def test_hostile_label_churn_stays_finite_and_private() -> None:
    allowed_accounts = accounts(128)
    allowed_routes = routes(256)
    metrics = TelemetryMetrics(
        account_ids=allowed_accounts,
        route_group_ids=allowed_routes,
        max_series=MAX_SERIES,
        max_scrape_bytes=MAX_SCRAPE_BYTES,
    )

    for index in range(1_000):
        route = allowed_routes[index % len(allowed_routes)] if index < 256 else f"alias-{index}"
        account = (
            allowed_accounts[index % len(allowed_accounts)] if index < 500 else f"tenant-{index}"
        )
        outcome = tuple(TerminalOutcome)[index % len(TerminalOutcome)]
        trigger = tuple(RouteTrigger)[index % len(RouteTrigger)]
        metrics.request(route, account, outcome)
        metrics.attempt(route, account, trigger, spill=index % 2 == 0, outcome=outcome)
        metrics.reject(f"untrusted-reason-{index}", tier=f"untrusted-tier-{index}")
        metrics.queue_wait(route, tier=index, wait_seconds=index / 1_000)
        metrics.circuit(account, f"untrusted-circuit-{index}")
        metrics.outcome(outcome, f"untrusted-outcome-{index}")

    first = metrics.render()
    first_series = metrics.series_count
    assert first_series <= MAX_SERIES
    assert len(first) <= MAX_SCRAPE_BYTES
    assert b'account="other"' in first
    assert b'route_group="other"' in first
    assert b'reason="other"' in first

    text = first.decode()
    for forbidden_label in (
        "key_id=",
        "request_id=",
        "attempt_id=",
        "event_id=",
        "deployment_generation_id=",
        "model_alias=",
    ):
        assert forbidden_label not in text
    for leaked in ("alias-999", "tenant-999", "untrusted-reason-999", "untrusted-tier-999"):
        assert leaked not in text

    for index in range(1_000, 2_000):
        metrics.request(f"rotating-route-{index}", f"rotating-account-{index}", "unknown")
        metrics.attempt(
            f"rotating-route-{index}",
            f"rotating-account-{index}",
            "unknown",
            spill=False,
            outcome="unknown",
        )
    assert metrics.series_count == first_series
    assert len(metrics.render()) <= MAX_SCRAPE_BYTES


def event(index: int) -> LifecycleEventV1:
    return LifecycleEventV1(
        event_id=EventId.new(),
        request_id=RequestId(f"req_{uuid4_at(index + 10_000)}"),
        kind=LifecycleEventKind.REQUEST_ADMITTED,
        occurred_at_ms=index + 1,
        installation_id=InstallationId("inst_00000000-0000-4000-8000-000000000001"),
        boot_id=GatewayBootId("boot_00000000-0000-4000-8000-000000000002"),
        key_id="1" * 32,
    )


def test_writer_sustains_reference_event_rate_without_blocking_request_path(
    tmp_path: Path,
) -> None:
    writer = LifecycleSpool.create(
        tmp_path,
        max_bytes=16 * 1024 * 1024,
        queue_capacity=4_096,
        segment_bytes=64 * 1024 * 1024,
        segment_seconds=300,
    )
    started = time.monotonic()
    index = 0
    for _ in range(250):
        reservation = writer.try_reserve(12)
        assert reservation is not None
        for _ in range(12):
            reservation.emit(event(index))
            index += 1
        reservation.release_unused()
    enqueue_elapsed = time.monotonic() - started
    writer.flush(timeout=10)
    stats = writer.stats
    records = tuple(writer.replay())
    writer.close()

    assert len(records) == 3_000
    assert stats.max_batch_records <= 256
    assert stats.max_fdatasync_interval_ms <= 250
    assert stats.invariant_violations == 0
    assert enqueue_elapsed < 5, "put_nowait hot path must not wait for disk batches"
