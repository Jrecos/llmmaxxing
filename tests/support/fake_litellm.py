"""Deterministic HTTPX LiteLLM boundary used by Task 8 and release gates."""

from __future__ import annotations

import asyncio
import gzip
import json
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "contract"
    / "litellm"
    / "fixtures"
    / "normalized-errors.json"
)


class FaultMode(StrEnum):
    JSON = "json"
    SSE = "sse"
    UTF8_SPLIT = "utf8_split"
    BINARY = "binary"
    COMPRESSED = "compressed"
    BACKPRESSURE = "backpressure"
    DELAY_HEADERS = "delay_headers"
    STALL_AFTER_BYTE = "stall_after_byte"
    RESET_BEFORE_HEADERS = "reset_before_headers"
    RESET_AFTER_BYTE = "reset_after_byte"
    MISSING_RECEIPT = "missing_receipt"
    WRONG_RECEIPT = "wrong_receipt"
    NAN_429 = "nan_parallel_429"
    ARLIAI_403 = "arliai_parallel_403"
    UPSTREAM_503 = "upstream_unavailable_503"
    UNHANDLED_500 = "unhandled_500"


@dataclass(frozen=True, slots=True)
class FaultPlan:
    mode: FaultMode
    deployment_id: str = "runtime-primary"
    header_gate: asyncio.Event | None = None
    stream_gate: asyncio.Event | None = None


class ControlledStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        gate_after_first: asyncio.Event | None = None,
        reset_after_first: bool = False,
        stall_after_first: bool = False,
    ) -> None:
        self.chunks = chunks
        self.gate_after_first = gate_after_first
        self.reset_after_first = reset_after_first
        self.stall_after_first = stall_after_first
        self.next_calls = 0
        self.closed = False

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for index, chunk in enumerate(self.chunks):
            self.next_calls += 1
            yield chunk
            if index == 0:
                if self.reset_after_first:
                    raise httpx.ReadError("deterministic reset")
                if self.stall_after_first:
                    await asyncio.Event().wait()
                if self.gate_after_first is not None:
                    await self.gate_after_first.wait()

    async def aclose(self) -> None:
        self.closed = True


def _normalized_errors() -> dict[str, tuple[int, dict[str, str], bytes]]:
    document = json.loads(_FIXTURES.read_text())
    result: dict[str, tuple[int, dict[str, str], bytes]] = {}
    for fixture in document["fixtures"]:
        result[fixture["name"]] = (
            fixture["status"],
            fixture["headers"],
            json.dumps(fixture["body"], separators=(",", ":")).encode(),
        )
    return result


_ERRORS = _normalized_errors()


@dataclass(slots=True)
class FakeLiteLLM:
    plans: deque[FaultPlan]
    calls: list[httpx.Request] = field(default_factory=list)
    request_bodies: list[bytes] = field(default_factory=list)
    streams: list[ControlledStream] = field(default_factory=list)

    @classmethod
    def with_plans(cls, *plans: FaultPlan) -> FakeLiteLLM:
        return cls(deque(plans))

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if not self.plans:
            raise AssertionError("fake LiteLLM received an unplanned call")
        plan = self.plans.popleft()
        self.calls.append(request)
        self.request_bodies.append(await request.aread())
        if plan.header_gate is not None:
            await plan.header_gate.wait()
        if plan.mode is FaultMode.RESET_BEFORE_HEADERS:
            raise httpx.ConnectError("deterministic connect reset", request=request)
        if plan.mode.value in _ERRORS:
            status, error_headers, body = _ERRORS[plan.mode.value]
            stream = ControlledStream((body,))
            self.streams.append(stream)
            return httpx.Response(
                status,
                headers=error_headers,
                stream=stream,
                request=request,
            )

        headers: dict[str, str] = {"content-type": "application/octet-stream"}
        if plan.mode is not FaultMode.MISSING_RECEIPT:
            headers["x-litellm-model-id"] = (
                "runtime-wrong" if plan.mode is FaultMode.WRONG_RECEIPT else plan.deployment_id
            )

        chunks: tuple[bytes, ...]
        if plan.mode in {FaultMode.JSON, FaultMode.MISSING_RECEIPT, FaultMode.WRONG_RECEIPT}:
            headers["content-type"] = "application/json"
            chunks = (b'{"ok":true}',)
        elif plan.mode is FaultMode.SSE:
            headers["content-type"] = "text/event-stream; charset=utf-8"
            chunks = (b'data: {"token":"a"}\n\n', b"data: [DONE]\n\n")
        elif plan.mode is FaultMode.UTF8_SPLIT:
            headers["content-type"] = "text/plain; charset=utf-8"
            encoded = b"uno \xf0\x9f\x9a\x80 dos"
            chunks = (encoded[:6], encoded[6:8], encoded[8:])
        elif plan.mode is FaultMode.BINARY:
            chunks = (bytes(range(128)), bytes(range(128, 256)))
        elif plan.mode is FaultMode.COMPRESSED:
            headers["content-encoding"] = "gzip"
            chunks = (gzip.compress(b"compressed-exact-bytes", mtime=0),)
        elif plan.mode is FaultMode.BACKPRESSURE:
            chunks = (b"a" * 65_536, b"b" * 65_536)
        elif plan.mode in {FaultMode.STALL_AFTER_BYTE, FaultMode.RESET_AFTER_BYTE}:
            chunks = (b"x", b"never")
        elif plan.mode is FaultMode.DELAY_HEADERS:
            chunks = (b"delayed",)
        else:
            raise AssertionError(f"unsupported fake fault mode: {plan.mode}")

        stream = ControlledStream(
            chunks,
            gate_after_first=plan.stream_gate,
            reset_after_first=plan.mode is FaultMode.RESET_AFTER_BYTE,
            stall_after_first=plan.mode is FaultMode.STALL_AFTER_BYTE,
        )
        self.streams.append(stream)
        return httpx.Response(200, headers=headers, stream=stream, request=request)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    @property
    def remaining(self) -> int:
        return len(self.plans)


async def response_bytes(response: httpx.Response) -> bytes:
    return b"".join([chunk async for chunk in response.aiter_raw()])


def fixture_body(name: str) -> bytes:
    return _ERRORS[name][2]


def fixture_status(name: str) -> int:
    return _ERRORS[name][0]


def fixture_headers(name: str) -> dict[str, str]:
    return dict(_ERRORS[name][1])


def fixture_document() -> dict[str, Any]:
    return json.loads(_FIXTURES.read_text())
