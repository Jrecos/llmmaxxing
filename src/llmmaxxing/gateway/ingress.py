"""Fail-closed ASGI request taxonomy, framing, and retained-body accounting."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llmmaxxing.adapters.litellm.contract import AdapterContract, CertifiedEndpoint

DEFAULT_INBOUND = 384
DEFAULT_PREAUTH = 64
DEFAULT_BODY_READERS_GLOBAL = 16
DEFAULT_BODY_READERS_PER_KEY = 4
DEFAULT_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_RETAINED_BYTES_GLOBAL = 256 * 1024 * 1024
DEFAULT_RETAINED_BYTES_PER_KEY = 64 * 1024 * 1024
DEFAULT_TARGET_BYTES = 8 * 1024
DEFAULT_HEADER_COUNT = 100
DEFAULT_HEADER_BYTES = 64 * 1024
DEFAULT_HEADER_NAME_BYTES = 256
DEFAULT_HEADER_VALUE_BYTES = 8 * 1024
DEFAULT_MULTIPART_SPOOL_BYTES = 1024 * 1024
DEFAULT_BODY_EVENTS = 100_000
DEFAULT_BODY_CHUNK_TIMEOUT_S = 5.0
DEFAULT_BODY_TOTAL_TIMEOUT_S = 300.0

_HEADER_NAME = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_SINGLETON_HEADERS = frozenset(
    {
        b"authorization",
        b"content-length",
        b"content-type",
        b"content-encoding",
        b"expect",
        b"host",
        b"transfer-encoding",
        b"upgrade",
    }
)

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]


class IngressError(ValueError):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class IngressLimits:
    inbound: int = DEFAULT_INBOUND
    preauth: int = DEFAULT_PREAUTH
    body_readers_global: int = DEFAULT_BODY_READERS_GLOBAL
    body_readers_per_key: int = DEFAULT_BODY_READERS_PER_KEY
    max_body_bytes: int = DEFAULT_BODY_BYTES
    retained_bytes_global: int = DEFAULT_RETAINED_BYTES_GLOBAL
    retained_bytes_per_key: int = DEFAULT_RETAINED_BYTES_PER_KEY
    max_target_bytes: int = DEFAULT_TARGET_BYTES
    max_header_count: int = DEFAULT_HEADER_COUNT
    max_header_bytes: int = DEFAULT_HEADER_BYTES
    max_header_name_bytes: int = DEFAULT_HEADER_NAME_BYTES
    max_header_value_bytes: int = DEFAULT_HEADER_VALUE_BYTES
    multipart_spool_bytes: int = DEFAULT_MULTIPART_SPOOL_BYTES
    max_body_events: int = DEFAULT_BODY_EVENTS
    body_chunk_timeout_s: float = DEFAULT_BODY_CHUNK_TIMEOUT_S
    body_total_timeout_s: float = DEFAULT_BODY_TOTAL_TIMEOUT_S

    def __post_init__(self) -> None:
        integer_limits = (
            self.inbound,
            self.preauth,
            self.body_readers_global,
            self.body_readers_per_key,
            self.max_body_bytes,
            self.retained_bytes_global,
            self.retained_bytes_per_key,
            self.max_target_bytes,
            self.max_header_count,
            self.max_header_bytes,
            self.max_header_name_bytes,
            self.max_header_value_bytes,
            self.multipart_spool_bytes,
            self.max_body_events,
        )
        if any(value < 1 for value in integer_limits):
            raise ValueError("ingress limits must be positive")
        if self.preauth > self.inbound:
            raise ValueError("pre-auth limit cannot exceed inbound limit")
        if self.body_readers_per_key > self.body_readers_global:
            raise ValueError("per-key body readers cannot exceed global readers")
        if self.max_body_bytes > self.retained_bytes_per_key:
            raise ValueError("one body cannot exceed the per-key retained limit")
        if self.retained_bytes_per_key > self.retained_bytes_global:
            raise ValueError("per-key retained limit cannot exceed global retained limit")
        if self.body_chunk_timeout_s <= 0 or self.body_total_timeout_s <= 0:
            raise ValueError("body timeouts must be positive")


@dataclass(frozen=True, slots=True)
class IngressRequest:
    endpoint: CertifiedEndpoint
    headers: tuple[tuple[bytes, bytes], ...]
    content_length: int | None
    content_type: str
    chunked: bool

    def values(self, name: bytes) -> tuple[bytes, ...]:
        lowered = name.lower()
        return tuple(value for header, value in self.headers if header == lowered)

    @property
    def multipart(self) -> bool:
        return self.endpoint.model_locator == "multipart.model"


class _CounterLease:
    __slots__ = ("_owner", "_released")

    def __init__(self, owner: _CounterLimiter) -> None:
        self._owner = owner
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._owner._release()


class _CounterLimiter:
    __slots__ = ("limit", "active", "_lock")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> _CounterLease | None:
        async with self._lock:
            if self.active >= self.limit:
                return None
            self.active += 1
        return _CounterLease(self)

    async def _release(self) -> None:
        async with self._lock:
            if self.active < 1:
                raise RuntimeError("counter lease underflow")
            self.active -= 1


class BodyReaderLease:
    __slots__ = ("_owner", "key_id", "_released")

    def __init__(self, owner: IngressResources, key_id: str) -> None:
        self._owner = owner
        self.key_id = key_id
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._owner._release_body_reader(self.key_id)


class BodyReservation:
    __slots__ = ("_owner", "key_id", "size", "_released")

    def __init__(self, owner: IngressResources, key_id: str) -> None:
        self._owner = owner
        self.key_id = key_id
        self.size = 0
        self._released = False

    async def grow(self, amount: int) -> bool:
        if self._released or amount < 0:
            return False
        if amount == 0:
            return True
        grown = await self._owner._grow_body(self.key_id, self.size, amount)
        if grown:
            self.size += amount
        return grown

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._owner._release_body(self.key_id, self.size)


class IngressResources:
    """All request-side count and retained-byte bounds for one Gateway process."""

    def __init__(self, limits: IngressLimits | None = None) -> None:
        self.limits = limits or IngressLimits()
        self._inbound = _CounterLimiter(self.limits.inbound)
        self._preauth = _CounterLimiter(self.limits.preauth)
        self._body_lock = asyncio.Lock()
        self._body_readers = 0
        self._body_readers_by_key: dict[str, int] = {}
        self._retained = 0
        self._retained_by_key: dict[str, int] = {}

    @property
    def inbound(self) -> int:
        return self._inbound.active

    @property
    def preauth(self) -> int:
        return self._preauth.active

    @property
    def body_readers(self) -> int:
        return self._body_readers

    @property
    def body_readers_by_key(self) -> Mapping[str, int]:
        return dict(self._body_readers_by_key)

    @property
    def retained_bytes(self) -> int:
        return self._retained

    @property
    def retained_by_key(self) -> Mapping[str, int]:
        return dict(self._retained_by_key)

    async def try_inbound(self) -> _CounterLease | None:
        return await self._inbound.try_acquire()

    async def try_preauth(self) -> _CounterLease | None:
        return await self._preauth.try_acquire()

    async def try_body_reader(self, key_id: str) -> BodyReaderLease | None:
        async with self._body_lock:
            key_count = self._body_readers_by_key.get(key_id, 0)
            if (
                self._body_readers >= self.limits.body_readers_global
                or key_count >= self.limits.body_readers_per_key
            ):
                return None
            self._body_readers += 1
            self._body_readers_by_key[key_id] = key_count + 1
        return BodyReaderLease(self, key_id)

    async def _release_body_reader(self, key_id: str) -> None:
        async with self._body_lock:
            key_count = self._body_readers_by_key.get(key_id, 0)
            if self._body_readers < 1 or key_count < 1:
                raise RuntimeError("body-reader lease underflow")
            self._body_readers -= 1
            if key_count == 1:
                self._body_readers_by_key.pop(key_id)
            else:
                self._body_readers_by_key[key_id] = key_count - 1

    def new_body_reservation(self, key_id: str) -> BodyReservation:
        return BodyReservation(self, key_id)

    async def _grow_body(self, key_id: str, current: int, amount: int) -> bool:
        async with self._body_lock:
            key_total = self._retained_by_key.get(key_id, 0)
            if (
                current + amount > self.limits.max_body_bytes
                or self._retained + amount > self.limits.retained_bytes_global
                or key_total + amount > self.limits.retained_bytes_per_key
            ):
                return False
            self._retained += amount
            self._retained_by_key[key_id] = key_total + amount
            return True

    async def _release_body(self, key_id: str, amount: int) -> None:
        async with self._body_lock:
            key_total = self._retained_by_key.get(key_id, 0)
            if amount > self._retained or amount > key_total:
                raise RuntimeError("retained-body reservation underflow")
            self._retained -= amount
            remaining = key_total - amount
            if remaining:
                self._retained_by_key[key_id] = remaining
            else:
                self._retained_by_key.pop(key_id, None)


class RetainedBody:
    """One quota-counted body retained through queueing and every attempt."""

    __slots__ = ("_file", "_reservation", "size", "spooled_to_disk", "_released")

    def __init__(
        self,
        file: tempfile.SpooledTemporaryFile[bytes],
        reservation: BodyReservation,
        size: int,
        spooled_to_disk: bool,
    ) -> None:
        self._file = file
        self._reservation = reservation
        self.size = size
        self.spooled_to_disk = spooled_to_disk
        self._released = False

    @property
    def spool_mode(self) -> int | None:
        if not self.spooled_to_disk:
            return None
        return os.fstat(self._file.fileno()).st_mode & 0o777

    async def read(self) -> bytes:
        if self._released:
            raise RuntimeError("retained body is released")
        self._file.seek(0)
        return self._file.read()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._file.close()
        await self._reservation.release()


def _headers(scope: Mapping[str, Any], limits: IngressLimits) -> tuple[tuple[bytes, bytes], ...]:
    raw_headers = scope.get("headers")
    if not isinstance(raw_headers, Sequence) or isinstance(raw_headers, (bytes, bytearray, str)):
        raise IngressError(400, "malformed_headers")
    if len(raw_headers) > limits.max_header_count:
        raise IngressError(431, "too_many_headers")
    result: list[tuple[bytes, bytes]] = []
    total = 0
    counts: dict[bytes, int] = {}
    for item in raw_headers:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise IngressError(400, "malformed_headers")
        name, value = item
        if not isinstance(name, bytes) or not isinstance(value, bytes):
            raise IngressError(400, "malformed_headers")
        if (
            not name
            or len(name) > limits.max_header_name_bytes
            or len(value) > limits.max_header_value_bytes
            or _HEADER_NAME.fullmatch(name) is None
            or b"\r" in value
            or b"\n" in value
            or b"\x00" in value
        ):
            raise IngressError(431, "header_limit")
        lowered = name.lower()
        total += len(lowered) + len(value) + 4
        if total > limits.max_header_bytes:
            raise IngressError(431, "headers_too_large")
        counts[lowered] = counts.get(lowered, 0) + 1
        result.append((lowered, value))
    if any(counts.get(name, 0) > 1 for name in _SINGLETON_HEADERS):
        raise IngressError(400, "duplicate_singleton_header")
    return tuple(result)


def _one(headers: tuple[tuple[bytes, bytes], ...], name: bytes) -> bytes | None:
    return next((value for header, value in headers if header == name), None)


def validate_http_request(
    scope: Mapping[str, Any],
    contract: AdapterContract,
    limits: IngressLimits | None = None,
) -> IngressRequest:
    selected = limits or IngressLimits()
    if scope.get("type") != "http":
        raise IngressError(400, "unsupported_asgi_scope")
    method = scope.get("method")
    path = scope.get("path")
    raw_path = scope.get("raw_path")
    query = scope.get("query_string", b"")
    if not isinstance(method, str) or not isinstance(path, str):
        raise IngressError(400, "malformed_target")
    if not isinstance(raw_path, bytes) or not isinstance(query, bytes):
        raise IngressError(400, "malformed_target")
    target_bytes = len(raw_path) + (1 + len(query) if query else 0)
    if target_bytes > selected.max_target_bytes:
        raise IngressError(414, "target_too_long")
    try:
        canonical_raw_path = path.encode("ascii")
    except UnicodeEncodeError:
        raise IngressError(400, "noncanonical_target") from None
    if raw_path != canonical_raw_path or b"%" in raw_path or query:
        raise IngressError(400, "noncanonical_target")

    headers = _headers(scope, selected)
    connection = (_one(headers, b"connection") or b"").lower()
    if _one(headers, b"upgrade") is not None or b"upgrade" in {
        token.strip() for token in connection.split(b",")
    }:
        raise IngressError(426, "upgrade_not_supported")
    if _one(headers, b"expect") is not None:
        raise IngressError(417, "expect_not_supported")
    if _one(headers, b"proxy-authorization") is not None:
        raise IngressError(400, "proxy_authorization_not_supported")

    content_length_value = _one(headers, b"content-length")
    transfer_encoding = _one(headers, b"transfer-encoding")
    if content_length_value is not None and transfer_encoding is not None:
        raise IngressError(400, "ambiguous_framing")
    content_length: int | None = None
    if content_length_value is not None:
        if not content_length_value.isdigit():
            raise IngressError(400, "malformed_content_length")
        content_length = int(content_length_value)
        if content_length > selected.max_body_bytes:
            raise IngressError(413, "body_too_large")
    chunked = transfer_encoding is not None
    if transfer_encoding is not None and transfer_encoding.lower() != b"chunked":
        raise IngressError(400, "unsupported_transfer_encoding")
    content_encoding = (_one(headers, b"content-encoding") or b"identity").lower()
    if content_encoding not in (b"", b"identity"):
        raise IngressError(415, "unsupported_content_encoding")

    paths = {endpoint.path for endpoint in contract.endpoints}
    endpoint = next(
        (
            candidate
            for candidate in contract.endpoints
            if candidate.path == path and candidate.method == method
        ),
        None,
    )
    if endpoint is None:
        if path in paths:
            raise IngressError(405, "method_not_allowed")
        raise IngressError(404, "route_not_found")
    content_type_bytes = _one(headers, b"content-type") or b""
    try:
        content_type = content_type_bytes.decode("ascii")
    except UnicodeDecodeError:
        raise IngressError(415, "unsupported_content_type") from None
    media_type = content_type.split(";", 1)[0].strip().lower()
    if endpoint.model_locator == "json.model" and not (
        media_type == "application/json" or media_type.endswith("+json")
    ):
        raise IngressError(415, "unsupported_content_type")
    if endpoint.model_locator == "multipart.model" and media_type != "multipart/form-data":
        raise IngressError(415, "unsupported_content_type")
    return IngressRequest(endpoint, headers, content_length, content_type, chunked)


async def read_retained_body(
    receive: ASGIReceive,
    request: IngressRequest,
    client_key_id: str,
    resources: IngressResources,
) -> RetainedBody:
    reader = await resources.try_body_reader(client_key_id)
    if reader is None:
        raise IngressError(429, "body_reader_limit")
    reservation = resources.new_body_reservation(client_key_id)
    threshold = (
        resources.limits.multipart_spool_bytes
        if request.multipart
        else resources.limits.max_body_bytes + 1
    )
    # Ownership intentionally transfers to RetainedBody and closes on its release.
    file = tempfile.SpooledTemporaryFile(max_size=threshold, mode="w+b")  # noqa: SIM115
    size = 0
    events = 0
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + resources.limits.body_total_timeout_s
        while True:
            events += 1
            if events > resources.limits.max_body_events:
                raise IngressError(408, "body_event_limit")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise IngressError(408, "body_timeout")
            try:
                async with asyncio.timeout(min(resources.limits.body_chunk_timeout_s, remaining)):
                    message = await receive()
            except TimeoutError:
                raise IngressError(408, "body_timeout") from None
            message_type = message.get("type")
            if message_type == "http.disconnect":
                raise IngressError(499, "client_disconnected")
            if message_type != "http.request":
                raise IngressError(400, "invalid_asgi_body_event")
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                raise IngressError(400, "invalid_asgi_body_event")
            if request.content_length is not None and size + len(chunk) > request.content_length:
                raise IngressError(400, "content_length_mismatch")
            if not await reservation.grow(len(chunk)):
                if size + len(chunk) > resources.limits.max_body_bytes:
                    raise IngressError(413, "body_too_large")
                raise IngressError(429, "retained_body_limit")
            file.write(chunk)
            size += len(chunk)
            if not message.get("more_body", False):
                break
        if request.content_length is not None and size != request.content_length:
            raise IngressError(400, "content_length_mismatch")
        file.seek(0)
        spooled = bool(getattr(file, "_rolled", False))
        if spooled and (os.fstat(file.fileno()).st_mode & 0o777) != 0o600:
            raise IngressError(500, "unsafe_spool_mode")
        return RetainedBody(file, reservation, size, spooled)
    except BaseException:
        file.close()
        await reservation.release()
        raise
    finally:
        await reader.release()
