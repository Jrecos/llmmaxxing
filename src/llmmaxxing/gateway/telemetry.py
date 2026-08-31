"""Production implementation of Task 8's mandatory lifecycle-capacity protocol."""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    EventId,
    GatewayBootId,
    InstallationId,
    RequestId,
    RouteGroupId,
)
from llmmaxxing.core.models import RequestProfile
from llmmaxxing.core.reasons import DispatchCause, RouteTrigger, TerminalOutcome
from llmmaxxing.gateway.auth import AuthenticatedClient
from llmmaxxing.gateway.scheduler import DispatchLease
from llmmaxxing.telemetry.events import (
    LifecycleEventKind,
    LifecycleEventV1,
    LifecycleReason,
    LifecycleTimingsV1,
)
from llmmaxxing.telemetry.metrics import TelemetryMetrics
from llmmaxxing.telemetry.otel import OptionalOtel
from llmmaxxing.telemetry.writer import (
    LifecycleSpool,
    SpoolReservation,
    SpoolSizing,
    SpoolStatus,
)


class TelemetryClock(Protocol):
    def now_ms(self) -> int: ...


class _SystemClock:
    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000


_CAUSE_TO_TRIGGER = {
    DispatchCause.PRIMARY: RouteTrigger.PRIMARY,
    DispatchCause.CAPACITY: RouteTrigger.CAPACITY_SPILL,
    DispatchCause.FAILURE: RouteTrigger.FAILURE_FALLBACK,
    DispatchCause.QUOTA: RouteTrigger.QUOTA_FALLBACK,
    DispatchCause.MANUAL_EMERGENCY: RouteTrigger.MANUAL_EMERGENCY,
    DispatchCause.SHADOW: RouteTrigger.SHADOW,
}


@dataclass(slots=True)
class _AttemptState:
    account_id: AccountId
    route_group_id: RouteGroupId
    trigger: RouteTrigger
    started_at_ms: int
    resolved: bool = False


@dataclass(slots=True)
class _LifecycleState:
    route_group_id: RouteGroupId | None = None
    queued_at_ms: int | None = None
    first_account_id: AccountId | None = None
    attempts: dict[AttemptId, _AttemptState] = field(default_factory=dict)
    profile_recorded: bool = False
    queued_recorded: bool = False
    terminal_recorded: bool = False
    released: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass(frozen=True, slots=True)
class LifecycleReservation:
    """Request-local frozen handle; all mutation stays in writer-owned state."""

    _capacity: TelemetryLifecycleCapacity
    _spool: SpoolReservation
    _state: _LifecycleState
    request_id: RequestId
    client: AuthenticatedClient
    admitted_at_ms: int
    max_attempts: int

    @property
    def budget_events(self) -> int:
        return self._spool.budget_events

    @property
    def emitted_events(self) -> int:
        return self._spool.emitted_events

    def release_unused(self, count: int | None = None) -> int:
        return self._spool.release_unused(count)

    async def profile_accepted(self, profile: RequestProfile) -> None:
        with self._state.lock:
            self._open()
            if self._state.profile_recorded:
                raise RuntimeError("request profile lifecycle event already recorded")
            now = self._capacity.clock.now_ms()
            event = self._event(
                LifecycleEventKind.REQUEST_ADMISSION_EVENT,
                now,
                route_group_id=profile.route_group_id,
                reason=LifecycleReason.PROFILE_ACCEPTED,
            )
            self._spool.emit(event)
            self._state.route_group_id = profile.route_group_id
            self._state.profile_recorded = True
        self._capacity.otel.request_profiled(self.request_id, profile.route_group_id)
        self._capacity._refresh_metrics()

    async def queued(self) -> None:
        with self._state.lock:
            self._open()
            if not self._state.profile_recorded or self._state.route_group_id is None:
                raise RuntimeError("request cannot queue before profile acceptance")
            if self._state.queued_recorded:
                raise RuntimeError("request queued lifecycle event already recorded")
            now = self._capacity.clock.now_ms()
            self._spool.emit(
                self._event(
                    LifecycleEventKind.REQUEST_QUEUED,
                    now,
                    route_group_id=self._state.route_group_id,
                )
            )
            self._state.queued_at_ms = now
            self._state.queued_recorded = True
        self._capacity._refresh_metrics()

    async def attempt_started(self, lease: DispatchLease, *, shadow: bool) -> None:
        attempt_id = lease.attempt_id
        request = lease.lease.request
        route_group_id = request.profile.route_group_id
        trigger = RouteTrigger.SHADOW if shadow else _CAUSE_TO_TRIGGER[lease.candidate.cause]
        account_id = lease.candidate.account_id
        with self._state.lock:
            self._open()
            if not self._state.queued_recorded:
                raise RuntimeError("provider attempt cannot start before queue admission")
            if attempt_id in self._state.attempts:
                raise RuntimeError("attempt lifecycle event already recorded")
            if len(self._state.attempts) >= self.max_attempts or self._spool.remaining_events < 3:
                raise RuntimeError("provider send exceeds reserved attempt budget")
            now = self._capacity.clock.now_ms()
            queue_wait = (
                None if self._state.queued_at_ms is None else max(0, now - self._state.queued_at_ms)
            )
            spill_from = (
                self._state.first_account_id
                if self._state.first_account_id is not None
                and self._state.first_account_id != account_id
                else None
            )
            event = self._event(
                LifecycleEventKind.ATTEMPT_RESERVED,
                now,
                route_group_id=route_group_id,
                attempt_id=attempt_id,
                account_id=account_id,
                deployment_generation_id=lease.candidate.generation_id,
                trigger=trigger,
                spill_from_account_id=spill_from,
                timings_ms=(
                    LifecycleTimingsV1(queue_wait_ms=queue_wait) if queue_wait is not None else None
                ),
            )
            # This put_nowait is the send fence: failure prevents the provider call.
            self._spool.emit(event)
            if self._state.first_account_id is None:
                self._state.first_account_id = account_id
            self._state.route_group_id = route_group_id
            self._state.attempts[attempt_id] = _AttemptState(
                account_id=account_id,
                route_group_id=route_group_id,
                trigger=trigger,
                started_at_ms=now,
            )
        self._capacity.otel.attempt_started(
            self.request_id,
            attempt_id,
            route_group_id,
            account_id,
            lease.candidate.generation_id,
            trigger,
        )
        self._capacity._refresh_metrics()

    async def attempt_finished(
        self,
        lease: DispatchLease,
        outcome: TerminalOutcome,
        *,
        uncertain: bool,
    ) -> None:
        attempt_id = lease.attempt_id
        with self._state.lock:
            self._open()
            attempt = self._state.attempts.get(attempt_id)
            if attempt is None:
                raise RuntimeError("attempt resolution has no reserved event pair")
            if attempt.resolved:
                return
            now = self._capacity.clock.now_ms()
            reason = _attempt_reason(outcome, uncertain)
            spill_from = (
                self._state.first_account_id
                if self._state.first_account_id is not None
                and self._state.first_account_id != attempt.account_id
                else None
            )
            self._spool.emit(
                self._event(
                    LifecycleEventKind.ATTEMPT_RESOLVED,
                    now,
                    route_group_id=attempt.route_group_id,
                    attempt_id=attempt_id,
                    account_id=attempt.account_id,
                    deployment_generation_id=lease.candidate.generation_id,
                    trigger=attempt.trigger,
                    outcome=outcome,
                    reason=reason,
                    spill_from_account_id=spill_from,
                    uncertain=uncertain,
                    timings_ms=LifecycleTimingsV1(duration_ms=max(0, now - attempt.started_at_ms)),
                    final_byte_at_ms=(now if outcome is TerminalOutcome.COMPLETED else None),
                    lease_released_at_ms=now,
                )
            )
            attempt.resolved = True
        self._capacity.metrics.attempt(
            attempt.route_group_id,
            attempt.account_id,
            attempt.trigger,
            spill=spill_from is not None,
            outcome=outcome,
        )
        self._capacity.otel.attempt_finished(
            self.request_id,
            attempt_id,
            outcome,
            uncertain=uncertain,
        )
        self._capacity._refresh_metrics()

    async def finished(self, outcome: TerminalOutcome) -> None:
        with self._state.lock:
            if self._state.terminal_recorded:
                return
            if self._state.released:
                raise RuntimeError("released request cannot emit a terminal event")
            now = self._capacity.clock.now_ms()
            route_group_id = self._state.route_group_id
            self._spool.emit(
                self._event(
                    LifecycleEventKind.REQUEST_TERMINAL,
                    now,
                    route_group_id=route_group_id,
                    outcome=outcome,
                    reason=LifecycleReason.from_outcome(outcome),
                    timings_ms=LifecycleTimingsV1(duration_ms=max(0, now - self.admitted_at_ms)),
                    attempts_used=len(self._state.attempts),
                ),
                terminal=True,
            )
            self._state.terminal_recorded = True
            self._spool.release_unused()
            account_id = self._state.first_account_id
        self._capacity.metrics.request(route_group_id, account_id, outcome)
        reason = LifecycleReason.from_outcome(outcome)
        self._capacity.metrics.outcome(outcome, reason)
        self._capacity.otel.request_finished(self.request_id, route_group_id, outcome)
        self._capacity._refresh_metrics()

    async def release(self) -> None:
        with self._state.lock:
            terminal = self._state.terminal_recorded
            released = self._state.released
        if released:
            return
        if not terminal:
            await self.finished(TerminalOutcome.UPSTREAM_FAILED)
        with self._state.lock:
            if self._state.released:
                return
            self._spool.release_unused()
            self._state.released = True
        self._capacity._release(self.request_id)
        self._capacity._refresh_metrics()

    def _open(self) -> None:
        if self._state.released or self._state.terminal_recorded:
            raise RuntimeError("request lifecycle is terminal")

    def _event(
        self,
        kind: LifecycleEventKind,
        occurred_at_ms: int,
        **fields: object,
    ) -> LifecycleEventV1:
        payload: dict[str, object] = {
            "event_id": EventId.new(),
            "request_id": self.request_id,
            "kind": kind,
            "occurred_at_ms": occurred_at_ms,
            "installation_id": self._capacity.installation_id,
            "boot_id": self._capacity.boot_id,
            "key_id": self.client.key_id,
        }
        payload.update(fields)
        return LifecycleEventV1.model_validate(payload)


class TelemetryLifecycleCapacity:
    """Bounded process-wide lifecycle reservations, independent from optional OTel."""

    def __init__(
        self,
        spool: LifecycleSpool,
        *,
        installation_id: InstallationId,
        boot_id: GatewayBootId,
        clock: TelemetryClock | None = None,
        metrics: TelemetryMetrics | None = None,
        otel: OptionalOtel | None = None,
    ) -> None:
        self.spool = spool
        self.installation_id = installation_id
        self.boot_id = boot_id
        self.clock = clock or _SystemClock()
        self.metrics = metrics or TelemetryMetrics()
        self.otel = otel or OptionalOtel(self.metrics)
        self._active: dict[RequestId, LifecycleReservation] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._refresh_metrics()

    @classmethod
    def start(
        cls,
        root: Path,
        *,
        installation_id: InstallationId,
        boot_id: GatewayBootId,
        max_bytes: int = 8 * 1024**3,
        sizing: SpoolSizing | None = None,
        physical_bytes: int | None = None,
        reopen: bool = False,
        clock: TelemetryClock | None = None,
        metrics: TelemetryMetrics | None = None,
        otel: OptionalOtel | None = None,
    ) -> TelemetryLifecycleCapacity:
        policy = sizing or SpoolSizing()
        volume = physical_bytes
        if volume is None:
            probe = root if root.exists() else root.parent
            volume = shutil.disk_usage(probe).total
        policy.validate(max_bytes, volume)
        factory = LifecycleSpool.open if reopen else LifecycleSpool.create
        spool = factory(root, max_bytes=max_bytes)
        return cls(
            spool,
            installation_id=installation_id,
            boot_id=boot_id,
            clock=clock,
            metrics=metrics,
            otel=otel,
        )

    @property
    def ready(self) -> bool:
        with self._lock:
            return not self._closed and self.spool.status is SpoolStatus.HEALTHY

    async def reserve(
        self,
        request_id: RequestId,
        client: AuthenticatedClient,
        events: int,
    ) -> LifecycleReservation | None:
        if events not in (10, 12) or (events - 4) % 2:
            raise ValueError("lifecycle budget must be 4 + 2*max_attempts (10 or 12 in V1)")
        with self._lock:
            if self._closed:
                return None
            if request_id in self._active:
                raise RuntimeError("request already owns lifecycle capacity")
            spool_reservation = self.spool.try_reserve(events)
            if spool_reservation is None:
                self.metrics.reject(LifecycleReason.TELEMETRY_SPOOL_EXHAUSTED, tier="other")
                self._refresh_metrics()
                return None
            admitted_at_ms = self.clock.now_ms()
            lifecycle = LifecycleReservation(
                self,
                spool_reservation,
                _LifecycleState(),
                request_id,
                client,
                admitted_at_ms,
                (events - 4) // 2,
            )
            try:
                spool_reservation.emit(
                    LifecycleEventV1(
                        event_id=EventId.new(),
                        request_id=request_id,
                        kind=LifecycleEventKind.REQUEST_ADMITTED,
                        occurred_at_ms=admitted_at_ms,
                        installation_id=self.installation_id,
                        boot_id=self.boot_id,
                        key_id=client.key_id,
                    )
                )
            except Exception:
                spool_reservation.release_unused()
                self.metrics.invariant_violation()
                self._refresh_metrics()
                return None
            self._active[request_id] = lifecycle
        self.otel.request_started(request_id)
        self._refresh_metrics()
        return lifecycle

    async def aclose(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = tuple(self._active.values())
        for lifecycle in active:
            try:
                await lifecycle.release()
            except Exception:
                self.metrics.invariant_violation()
        await asyncio.to_thread(self.spool.close)
        await asyncio.to_thread(self.otel.shutdown)
        self._refresh_metrics()

    def _release(self, request_id: RequestId) -> None:
        with self._lock:
            self._active.pop(request_id, None)

    def _refresh_metrics(self) -> None:
        self.metrics.set_reservations(self.spool.outstanding_reserved_bytes)
        self.metrics.set_spool(
            self.spool.backlog_bytes,
            self.spool.protected_bytes,
            self.spool.max_bytes,
            len(self.spool.segment_manifests),
        )


def _attempt_reason(outcome: TerminalOutcome, uncertain: bool) -> LifecycleReason:
    if not uncertain:
        return LifecycleReason.from_outcome(outcome)
    if outcome in {
        TerminalOutcome.CLIENT_CANCELLED,
        TerminalOutcome.DEADLINE_EXCEEDED,
        TerminalOutcome.RESPONSE_STREAM_FAILED,
    }:
        return LifecycleReason(outcome.value)
    return LifecycleReason.UPSTREAM_UNKNOWN
