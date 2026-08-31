"""Finite-label Prometheus registry for lifecycle admission and delivery."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from llmmaxxing.core.ids import AccountId, RouteGroupId
from llmmaxxing.core.reasons import RouteTrigger, TerminalOutcome
from llmmaxxing.telemetry.events import LifecycleReason

MAX_ACCOUNTS = 128
MAX_ROUTE_GROUPS = 256
MAX_SERIES = 10_000
MAX_SCRAPE_BYTES = 5 * 1024 * 1024
_OTHER = "other"
_QUEUE_BUCKETS = (0.001, 0.01, 0.1, 1.0, 5.0, 30.0, 120.0, 600.0)
_WRITER_BATCH_BUCKETS = (1, 8, 32, 64, 128, 256)
_FSYNC_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.25)


@dataclass(frozen=True, slots=True)
class _Family:
    metric: Any
    labels: tuple[str, ...]
    cost: int
    kind: Literal["counter", "gauge", "histogram"]


class TelemetryMetrics:
    """Own registry with configured finite identities and a hard series budget."""

    def __init__(
        self,
        *,
        account_ids: Iterable[AccountId | str] = (),
        route_group_ids: Iterable[RouteGroupId | str] = (),
        tiers: Iterable[int | str] = (0, 10, 20, 100),
        max_series: int = MAX_SERIES,
        max_scrape_bytes: int = MAX_SCRAPE_BYTES,
    ) -> None:
        if max_series < 256 or max_series > MAX_SERIES:
            raise ValueError(f"series cap must be in [256, {MAX_SERIES}]")
        if max_scrape_bytes < 1024 or max_scrape_bytes > MAX_SCRAPE_BYTES:
            raise ValueError(f"scrape cap must be in [1024, {MAX_SCRAPE_BYTES}]")
        self.registry = CollectorRegistry(auto_describe=True)
        self.max_series = max_series
        self.max_scrape_bytes = max_scrape_bytes
        self._accounts = frozenset(sorted({str(value) for value in account_ids})[:MAX_ACCOUNTS])
        self._routes = frozenset(
            sorted({str(value) for value in route_group_ids})[:MAX_ROUTE_GROUPS]
        )
        self._tiers = frozenset(str(value) for value in tiers)
        self._outcomes = frozenset(value.value for value in TerminalOutcome)
        self._triggers = frozenset(value.value for value in RouteTrigger)
        self._reasons = frozenset(
            {
                *(value.value for value in LifecycleReason),
                "capacity",
                "transient_failure",
                "quota",
                "manual",
            }
        )
        self._lock = threading.Lock()
        self._seen: dict[str, set[tuple[str, ...]]] = {}
        self._estimated_series = 0

        self._overflow = Counter(
            "llmmaxxing_metrics_series_overflow",
            "New metric labelsets folded into finite fallback series.",
            registry=self.registry,
        )
        self._invariants = Counter(
            "llmmaxxing_event_invariant_violations",
            "Lifecycle reservation or terminal invariant violations.",
            registry=self.registry,
        )
        self._dedupe = Counter(
            "llmmaxxing_dedupe_replayed",
            "At-least-once lifecycle records deduplicated by event_id.",
            registry=self.registry,
        )
        self._otel_dropped = Counter(
            "llmmaxxing_otel_export_dropped",
            "Best-effort OTel spans dropped from the bounded queue.",
            registry=self.registry,
        )
        self._otel_failed = Counter(
            "llmmaxxing_otel_export_failed",
            "Best-effort OTel spans whose export failed.",
            registry=self.registry,
        )

        families = {
            "requests": _Family(
                Counter(
                    "llmmaxxing_requests",
                    "Terminal requests by finite route/account/outcome.",
                    ("route_group", "account", "outcome"),
                    registry=self.registry,
                ),
                ("route_group", "account", "outcome"),
                2,
                "counter",
            ),
            "rejections": _Family(
                Counter(
                    "llmmaxxing_admission_rejections",
                    "Admission rejections by closed reason and tier.",
                    ("reason", "tier"),
                    registry=self.registry,
                ),
                ("reason", "tier"),
                2,
                "counter",
            ),
            "queue_depth": _Family(
                Gauge(
                    "llmmaxxing_admission_queue_depth",
                    "Current admitted queue depth by tier.",
                    ("tier",),
                    registry=self.registry,
                ),
                ("tier",),
                1,
                "gauge",
            ),
            "queue_wait": _Family(
                Histogram(
                    "llmmaxxing_admission_queue_wait_seconds",
                    "Queue wait by finite route and tier.",
                    ("route_group", "tier"),
                    buckets=_QUEUE_BUCKETS,
                    registry=self.registry,
                ),
                ("route_group", "tier"),
                len(_QUEUE_BUCKETS) + 4,
                "histogram",
            ),
            "circuits": _Family(
                Counter(
                    "llmmaxxing_circuit_transitions",
                    "Circuit transitions by finite account and closed reason.",
                    ("account", "reason"),
                    registry=self.registry,
                ),
                ("account", "reason"),
                2,
                "counter",
            ),
            "attempts": _Family(
                Counter(
                    "llmmaxxing_attempts",
                    "Provider attempts by finite route/account/trigger/spill/outcome.",
                    ("route_group", "account", "trigger", "spill", "outcome"),
                    registry=self.registry,
                ),
                ("route_group", "account", "trigger", "spill", "outcome"),
                2,
                "counter",
            ),
            "outcomes": _Family(
                Counter(
                    "llmmaxxing_outcomes",
                    "Request outcomes by closed outcome and reason.",
                    ("outcome", "reason"),
                    registry=self.registry,
                ),
                ("outcome", "reason"),
                2,
                "counter",
            ),
        }
        self._families = families

        self._reserved_bytes = Gauge(
            "llmmaxxing_reservations_outstanding_bytes",
            "Conservative lifecycle bytes reserved or awaiting durable write.",
            registry=self.registry,
        )
        self._spool_bytes = Gauge(
            "llmmaxxing_spool_bytes",
            "Physical lifecycle delivery backlog bytes.",
            registry=self.registry,
        )
        self._spool_ratio = Gauge(
            "llmmaxxing_spool_watermark_ratio",
            "Protected lifecycle bytes divided by configured spool bytes.",
            registry=self.registry,
        )
        self._spool_segments = Gauge(
            "llmmaxxing_spool_segments",
            "Retained lifecycle spool segments.",
            registry=self.registry,
        )
        self._writer_batch = Histogram(
            "llmmaxxing_writer_batch_records",
            "Records in each lifecycle disk batch.",
            buckets=_WRITER_BATCH_BUCKETS,
            registry=self.registry,
        )
        self._writer_sync = Histogram(
            "llmmaxxing_writer_fdatasync_seconds",
            "Lifecycle fdatasync latency.",
            buckets=_FSYNC_BUCKETS,
            registry=self.registry,
        )

        # These fixed families exist even at zero, so reserve them before admitting labels.
        self._estimated_series = self._count_exposed_series(generate_latest(self.registry))
        for name, family in families.items():
            self._create_labelset(name, family, self._fallback(family))

    @property
    def series_count(self) -> int:
        return self._count_exposed_series(self.render())

    def request(
        self,
        route_group: RouteGroupId | str | None,
        account: AccountId | str | None,
        outcome: TerminalOutcome | str,
    ) -> None:
        self._increment(
            "requests",
            {
                "route_group": self._route(route_group),
                "account": self._account(account),
                "outcome": self._closed(outcome, self._outcomes),
            },
        )

    def reject(self, reason: LifecycleReason | str, *, tier: int | str) -> None:
        self._increment(
            "rejections",
            {"reason": self._closed(reason, self._reasons), "tier": self._tier(tier)},
        )

    def set_queue_depth(self, tier: int | str, depth: int) -> None:
        if depth < 0:
            raise ValueError("queue depth cannot be negative")
        self._apply("queue_depth", {"tier": self._tier(tier)}, "set", depth)

    def queue_wait(
        self,
        route_group: RouteGroupId | str | None,
        *,
        tier: int | str,
        wait_seconds: float,
    ) -> None:
        self._apply(
            "queue_wait",
            {"route_group": self._route(route_group), "tier": self._tier(tier)},
            "observe",
            max(0.0, wait_seconds),
        )

    def circuit(self, account: AccountId | str | None, reason: LifecycleReason | str) -> None:
        self._increment(
            "circuits",
            {"account": self._account(account), "reason": self._closed(reason, self._reasons)},
        )

    def attempt(
        self,
        route_group: RouteGroupId | str | None,
        account: AccountId | str | None,
        trigger: RouteTrigger | str,
        *,
        spill: bool,
        outcome: TerminalOutcome | str,
    ) -> None:
        self._increment(
            "attempts",
            {
                "route_group": self._route(route_group),
                "account": self._account(account),
                "trigger": self._closed(trigger, self._triggers),
                "spill": "true" if spill else "false",
                "outcome": self._closed(outcome, self._outcomes),
            },
        )

    def outcome(self, outcome: TerminalOutcome | str, reason: LifecycleReason | str) -> None:
        self._increment(
            "outcomes",
            {
                "outcome": self._closed(outcome, self._outcomes),
                "reason": self._closed(reason, self._reasons),
            },
        )

    def set_reservations(self, value: int) -> None:
        self._reserved_bytes.set(max(0, value))

    def set_spool(
        self, physical_bytes: int, protected_bytes: int, max_bytes: int, segments: int
    ) -> None:
        self._spool_bytes.set(max(0, physical_bytes))
        self._spool_ratio.set(
            0 if max_bytes <= 0 else min(1.0, max(0, protected_bytes) / max_bytes)
        )
        self._spool_segments.set(max(0, segments))

    def writer_batch(self, records: int) -> None:
        self._writer_batch.observe(max(0, records))

    def writer_fdatasync(self, seconds: float) -> None:
        self._writer_sync.observe(max(0.0, seconds))

    def invariant_violation(self, count: int = 1) -> None:
        self._invariants.inc(max(0, count))

    def dedupe_replayed(self, count: int = 1) -> None:
        self._dedupe.inc(max(0, count))

    def otel_dropped(self, count: int = 1) -> None:
        self._otel_dropped.inc(max(0, count))

    def otel_failed(self, count: int = 1) -> None:
        self._otel_failed.inc(max(0, count))

    def render(self) -> bytes:
        payload = generate_latest(self.registry)
        if len(payload) > self.max_scrape_bytes:
            raise RuntimeError("bounded metrics scrape exceeded its configured cap")
        if self._count_exposed_series(payload) > self.max_series:
            raise RuntimeError("bounded metrics registry exceeded its configured series cap")
        return payload

    def _increment(self, family: str, labels: Mapping[str, str]) -> None:
        self._apply(family, labels, "inc", 1)

    def _apply(
        self,
        name: str,
        labels: Mapping[str, str],
        operation: str,
        value: float,
    ) -> None:
        family = self._families[name]
        values = tuple(labels[label] for label in family.labels)
        child = self._child(name, family, values)
        getattr(child, operation)(value)

    def _child(self, name: str, family: _Family, values: tuple[str, ...]) -> Any:
        with self._lock:
            seen = self._seen[name]
            if values not in seen:
                if self._estimated_series + family.cost <= self.max_series:
                    self._create_labelset(name, family, values)
                else:
                    self._overflow.inc()
                    values = self._fallback(family)
            return family.metric.labels(*values)

    def _create_labelset(
        self,
        name: str,
        family: _Family,
        values: tuple[str, ...],
    ) -> None:
        family.metric.labels(*values)
        self._seen.setdefault(name, set()).add(values)
        self._estimated_series += family.cost

    @staticmethod
    def _fallback(family: _Family) -> tuple[str, ...]:
        return tuple("false" if label == "spill" else _OTHER for label in family.labels)

    def _route(self, value: RouteGroupId | str | None) -> str:
        normalized = str(value) if value is not None else _OTHER
        return normalized if normalized in self._routes else _OTHER

    def _account(self, value: AccountId | str | None) -> str:
        normalized = str(value) if value is not None else _OTHER
        return normalized if normalized in self._accounts else _OTHER

    def _tier(self, value: int | str) -> str:
        normalized = str(value)
        return normalized if normalized in self._tiers else _OTHER

    @staticmethod
    def _closed(value: Any, allowed: Sequence[str] | frozenset[str]) -> str:
        normalized = value.value if hasattr(value, "value") else str(value)
        return normalized if normalized in allowed else _OTHER

    @staticmethod
    def _count_exposed_series(payload: bytes) -> int:
        return sum(1 for line in payload.splitlines() if line and not line.startswith(b"#"))
