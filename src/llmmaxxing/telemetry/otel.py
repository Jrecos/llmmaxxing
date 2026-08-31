"""Optional metadata-only OpenTelemetry with deterministic sampling and isolation."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    DeploymentGenerationId,
    RequestId,
    RouteGroupId,
)
from llmmaxxing.core.reasons import RouteTrigger, TerminalOutcome
from llmmaxxing.telemetry.metrics import TelemetryMetrics

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, Status, StatusCode


def deterministic_head_sample(request_id: RequestId) -> bool:
    """Stable 10% head sample, independent of process restarts and replay."""
    prefix = hashlib.sha256(str(request_id).encode()).hexdigest()[:8]
    return int(prefix, 16) % 10 == 0


class _BoundedSpanProcessor(SpanProcessor):
    """Small SDK processor whose only legal loss policy is OTel drop-oldest."""

    def __init__(
        self,
        exporter: SpanExporter,
        metrics: TelemetryMetrics,
        *,
        queue_size: int,
        batch_size: int,
        export_interval_ms: int,
    ) -> None:
        if queue_size < 1 or not 1 <= batch_size <= queue_size:
            raise ValueError("invalid OTel queue or batch size")
        if export_interval_ms < 1:
            raise ValueError("OTel export interval must be positive")
        self._exporter = exporter
        self._metrics = metrics
        self._queue_size = queue_size
        self._batch_size = batch_size
        self._interval_s = export_interval_ms / 1_000
        self._queue: deque[ReadableSpan] = deque()
        self._condition = threading.Condition()
        self._stopping = False
        self._exporting = False
        self._dropped = 0
        self._failed = 0
        self._thread = threading.Thread(
            target=self._run,
            name="llmmaxxing-otel-export",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped

    @property
    def failed(self) -> int:
        with self._condition:
            return self._failed

    def on_start(self, span: Span, parent_context: Any | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        context = span.context
        if context is None or not context.trace_flags.sampled:
            return
        with self._condition:
            if self._stopping:
                return
            if len(self._queue) >= self._queue_size:
                self._queue.popleft()
                self._dropped += 1
                self._metrics.otel_dropped()
            self._queue.append(span)
            self._condition.notify()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        deadline = time.monotonic() + max(0, timeout_millis) / 1_000
        with self._condition:
            self._condition.notify_all()
            while (self._queue or self._exporting) and time.monotonic() < deadline:
                self._condition.wait(max(0.0, deadline - time.monotonic()))
            return not self._queue and not self._exporting

    def shutdown(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=30)
        try:
            self._exporter.shutdown()
        except Exception:
            with self._condition:
                self._failed += 1
            self._metrics.otel_failed()

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._queue and not self._stopping:
                    self._condition.wait(self._interval_s)
                if self._stopping and not self._queue:
                    self._condition.notify_all()
                    return
                if not self._queue:
                    continue
                batch = tuple(self._queue.popleft() for _ in range(min(self._batch_size, len(self._queue))))
                self._exporting = True
            self._export(batch)
            with self._condition:
                self._exporting = False
                self._condition.notify_all()

    def _export(self, batch: Sequence[ReadableSpan]) -> None:
        failed = False
        try:
            result = self._exporter.export(batch)
            failed = SpanExportResult is not None and result is not SpanExportResult.SUCCESS
        except Exception:
            failed = True
        if failed:
            with self._condition:
                self._failed += len(batch)
            self._metrics.otel_failed(len(batch))


class OptionalOtel:
    """Best-effort spans; every method degrades to a bounded no-op."""

    def __init__(
        self,
        metrics: TelemetryMetrics,
        *,
        exporter: SpanExporter | None = None,
        endpoint: str | None = None,
        queue_size: int = 2048,
        batch_size: int = 256,
        export_interval_ms: int = 100,
    ) -> None:
        self._metrics = metrics
        self._provider: TracerProvider | None = None
        self._processor: _BoundedSpanProcessor | None = None
        self._tracer: Any | None = None
        self._lock = threading.Lock()
        self._requests: dict[RequestId, Span] = {}
        self._attempts: dict[tuple[RequestId, AttemptId], Span] = {}
        self._initial_failures = 0
        self._closed = False
        if exporter is None and endpoint is None:
            return
        try:
            if TracerProvider is None or Resource is None or trace is None:
                raise ImportError("OpenTelemetry SDK is unavailable")
            if exporter is None:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=endpoint)
            provider = TracerProvider(
                resource=Resource.create({"service.name": "llmmaxxing-gateway"})
            )
            processor = _BoundedSpanProcessor(
                exporter,
                metrics,
                queue_size=queue_size,
                batch_size=batch_size,
                export_interval_ms=export_interval_ms,
            )
            provider.add_span_processor(processor)
            self._provider = provider
            self._processor = processor
            self._tracer = provider.get_tracer("llmmaxxing.telemetry", "1")
        except Exception:
            self._initial_failures = 1
            metrics.otel_failed()

    @property
    def enabled(self) -> bool:
        return self._tracer is not None and not self._closed

    @property
    def dropped_spans(self) -> int:
        processor = self._processor
        return 0 if processor is None else processor.dropped

    @property
    def failed_exports(self) -> int:
        processor = self._processor
        return self._initial_failures + (0 if processor is None else processor.failed)

    def request_started(self, request_id: RequestId) -> None:
        tracer = self._tracer
        if tracer is None or not deterministic_head_sample(request_id):
            return
        try:
            span = tracer.start_span(
                "llmmaxxing.request",
                kind=SpanKind.SERVER,
                attributes={
                    "llmmaxxing.schema_version": 1,
                    "llmmaxxing.request_id": str(request_id),
                },
            )
            with self._lock:
                if self._closed or request_id in self._requests:
                    span.end()
                    return
                self._requests[request_id] = span
        except Exception:
            self._disable()

    def request_profiled(self, request_id: RequestId, route_group_id: RouteGroupId) -> None:
        span = self._request(request_id)
        if span is not None:
            self._safe_set(span, "llmmaxxing.route_group", str(route_group_id))

    def attempt_started(
        self,
        request_id: RequestId,
        attempt_id: AttemptId,
        route_group_id: RouteGroupId,
        account_id: AccountId,
        generation_id: DeploymentGenerationId,
        trigger: RouteTrigger,
    ) -> None:
        tracer = self._tracer
        root = self._request(request_id)
        if tracer is None or root is None:
            return
        try:
            context = trace.set_span_in_context(root)
            span = tracer.start_span(
                "llmmaxxing.attempt",
                context=context,
                kind=SpanKind.CLIENT,
                attributes={
                    "llmmaxxing.request_id": str(request_id),
                    "llmmaxxing.attempt_id": str(attempt_id),
                    "llmmaxxing.route_group": str(route_group_id),
                    "llmmaxxing.account": str(account_id),
                    "llmmaxxing.deployment_generation": str(generation_id),
                    "llmmaxxing.trigger": trigger.value,
                },
            )
            with self._lock:
                key = (request_id, attempt_id)
                if self._closed or key in self._attempts:
                    span.end()
                    return
                self._attempts[key] = span
        except Exception:
            self._disable()

    def attempt_finished(
        self,
        request_id: RequestId,
        attempt_id: AttemptId,
        outcome: TerminalOutcome,
        *,
        uncertain: bool,
    ) -> None:
        with self._lock:
            span = self._attempts.pop((request_id, attempt_id), None)
        if span is None:
            return
        self._safe_set(span, "llmmaxxing.outcome", outcome.value)
        self._safe_set(span, "llmmaxxing.uncertain", uncertain)
        self._set_status(span, outcome)
        self._safe_end(span)

    def request_finished(
        self,
        request_id: RequestId,
        route_group_id: RouteGroupId | None,
        outcome: TerminalOutcome,
    ) -> None:
        with self._lock:
            span = self._requests.pop(request_id, None)
            attempts = [
                self._attempts.pop(key)
                for key in tuple(self._attempts)
                if key[0] == request_id
            ]
        for attempt in attempts:
            self._safe_end(attempt)
        if span is None:
            return
        if route_group_id is not None:
            self._safe_set(span, "llmmaxxing.route_group", str(route_group_id))
        self._safe_set(span, "llmmaxxing.outcome", outcome.value)
        self._set_status(span, outcome)
        self._safe_end(span)

    def flush(self, timeout: float = 5) -> bool:
        processor = self._processor
        if processor is None:
            return True
        try:
            return processor.force_flush(max(0, int(timeout * 1_000)))
        except Exception:
            self._metrics.otel_failed()
            return False

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            spans = (*self._attempts.values(), *self._requests.values())
            self._attempts.clear()
            self._requests.clear()
        for span in spans:
            self._safe_end(span)
        provider = self._provider
        if provider is not None:
            try:
                provider.shutdown()
            except Exception:
                self._metrics.otel_failed()

    def _request(self, request_id: RequestId) -> Span | None:
        with self._lock:
            return self._requests.get(request_id)

    def _disable(self) -> None:
        self._initial_failures += 1
        self._metrics.otel_failed()
        self.shutdown()
        self._tracer = None

    @staticmethod
    def _safe_set(span: Span, name: str, value: str | bool) -> None:
        try:
            span.set_attribute(name, value)
        except Exception:
            pass

    @staticmethod
    def _safe_end(span: Span) -> None:
        try:
            span.end()
        except Exception:
            pass

    @staticmethod
    def _set_status(span: Span, outcome: TerminalOutcome) -> None:
        if Status is None or StatusCode is None:
            return
        try:
            code = StatusCode.OK if outcome is TerminalOutcome.COMPLETED else StatusCode.ERROR
            span.set_status(Status(code))
        except Exception:
            pass
