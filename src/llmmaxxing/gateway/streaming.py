"""Process HTTPX ownership, partitioned attempt permits, and byte-exact relay."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

HTTP_POOL_CONNECTIONS = 160
ATTEMPT_PERMITS = 144
FOREGROUND_PERMITS = 128
RECOVERY_PERMITS = 8
DISCOVERY_QUALIFICATION_PERMITS = 8
RAW_CHUNK_BYTES = 64 * 1024
MAX_PRESTART_ERROR_BYTES = 128 * 1024

ASGISend = Any


class PermitClass(StrEnum):
    FOREGROUND = "foreground"
    RECOVERY = "recovery"
    DISCOVERY = "discovery"
    QUALIFICATION = "qualification"


@dataclass(frozen=True, slots=True)
class PermitSnapshot:
    foreground_active: int
    recovery_active: int
    discovery_qualification_active: int
    foreground_waiters: int
    recovery_waiters: int
    discovery_qualification_waiters: int

    @property
    def total_active(self) -> int:
        return self.foreground_active + self.recovery_active + self.discovery_qualification_active

    @property
    def total_waiters(self) -> int:
        return (
            self.foreground_waiters + self.recovery_waiters + self.discovery_qualification_waiters
        )


class AttemptPermit:
    __slots__ = ("_pool", "partition", "shadow", "_released")

    def __init__(self, pool: AttemptPermitPool, partition: str, *, shadow: bool) -> None:
        self._pool = pool
        self.partition = partition
        self.shadow = shadow
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._pool._release(self.partition)


class AttemptPermitPool:
    """One cancellation-safe 144-permit pool with non-borrowable reservations."""

    _LIMITS = {
        "foreground": FOREGROUND_PERMITS,
        "recovery": RECOVERY_PERMITS,
        "discovery_qualification": DISCOVERY_QUALIFICATION_PERMITS,
    }

    def __init__(self) -> None:
        if sum(self._LIMITS.values()) != ATTEMPT_PERMITS:
            raise RuntimeError("attempt permit partitions do not sum to 144")
        self._condition = asyncio.Condition()
        self._active = {name: 0 for name in self._LIMITS}
        self._waiters = {name: 0 for name in self._LIMITS}

    @staticmethod
    def _partition(kind: PermitClass) -> str:
        if kind is PermitClass.FOREGROUND:
            return "foreground"
        if kind is PermitClass.RECOVERY:
            return "recovery"
        return "discovery_qualification"

    async def acquire(self, kind: PermitClass) -> AttemptPermit:
        partition = self._partition(kind)
        async with self._condition:
            self._waiters[partition] += 1
            try:
                while self._active[partition] >= self._LIMITS[partition]:
                    await self._condition.wait()
                self._active[partition] += 1
            finally:
                self._waiters[partition] -= 1
        return AttemptPermit(self, partition, shadow=False)

    async def try_shadow(self) -> AttemptPermit | None:
        """Borrow only an immediately idle foreground permit; never queue or bypass waiters."""
        async with self._condition:
            if self._waiters["foreground"] or self._active["foreground"] >= FOREGROUND_PERMITS:
                return None
            self._active["foreground"] += 1
        return AttemptPermit(self, "foreground", shadow=True)

    async def _release(self, partition: str) -> None:
        async with self._condition:
            if self._active[partition] < 1:
                raise RuntimeError("attempt permit underflow")
            self._active[partition] -= 1
            self._condition.notify_all()

    def snapshot(self) -> PermitSnapshot:
        return PermitSnapshot(
            foreground_active=self._active["foreground"],
            recovery_active=self._active["recovery"],
            discovery_qualification_active=self._active["discovery_qualification"],
            foreground_waiters=self._waiters["foreground"],
            recovery_waiters=self._waiters["recovery"],
            discovery_qualification_waiters=self._waiters["discovery_qualification"],
        )


@dataclass(frozen=True, slots=True)
class HTTPClientConfig:
    origin: str
    max_connections: int
    max_keepalive_connections: int
    trust_env: bool
    follow_redirects: bool
    verify_tls: bool
    retries: int
    connect_timeout_s: float
    write_timeout_s: float
    read_inactivity_timeout_s: float
    pool_timeout_s: float


class ProcessHTTPClient:
    """The one HTTPX client owned by one Task-9 single-worker Gateway process."""

    def __init__(
        self,
        origin: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        connect_timeout_s: float = 10.0,
        write_timeout_s: float = 30.0,
        read_inactivity_timeout_s: float = 120.0,
        pool_timeout_s: float = 30.0,
    ) -> None:
        if not origin.startswith(("http://", "https://")) or origin.endswith("/"):
            raise ValueError("LiteLLM origin must be an absolute URL without a trailing slash")
        if (
            min(
                connect_timeout_s,
                write_timeout_s,
                read_inactivity_timeout_s,
                pool_timeout_s,
            )
            <= 0
        ):
            raise ValueError("HTTP timeouts must be positive")
        self.config = HTTPClientConfig(
            origin=origin,
            max_connections=HTTP_POOL_CONNECTIONS,
            max_keepalive_connections=HTTP_POOL_CONNECTIONS,
            trust_env=False,
            follow_redirects=False,
            verify_tls=True,
            retries=0,
            connect_timeout_s=connect_timeout_s,
            write_timeout_s=write_timeout_s,
            read_inactivity_timeout_s=read_inactivity_timeout_s,
            pool_timeout_s=pool_timeout_s,
        )
        self._client = httpx.AsyncClient(
            base_url=origin,
            limits=httpx.Limits(
                max_connections=HTTP_POOL_CONNECTIONS,
                max_keepalive_connections=HTTP_POOL_CONNECTIONS,
            ),
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                write=write_timeout_s,
                read=read_inactivity_timeout_s,
                pool=pool_timeout_s,
            ),
            trust_env=False,
            follow_redirects=False,
            verify=True,
            transport=transport,
        )
        self._closed = False

    @property
    def client_count(self) -> int:
        return 0 if self._closed else 1

    async def send(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        content: bytes,
    ) -> httpx.Response:
        if self._closed:
            raise RuntimeError("process HTTP client is closed")
        if not path.startswith("/v1/") or "?" in path or "#" in path:
            raise ValueError("upstream path is outside the certified route table")
        request = self._client.build_request(method, path, headers=headers, content=content)
        return await self._client.send(request, stream=True, follow_redirects=False)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()


_HOP_HEADERS = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    }
)


def response_headers(response: httpx.Response) -> tuple[tuple[bytes, bytes], ...]:
    connection_tokens: set[bytes] = set()
    for name, value in response.headers.raw:
        if name.lower() == b"connection":
            connection_tokens.update(token.strip().lower() for token in value.split(b","))
    return tuple(
        (name.lower(), value)
        for name, value in response.headers.raw
        if name.lower() not in _HOP_HEADERS
        and name.lower() not in connection_tokens
        and name.lower() != b"content-length"
    )


class UpstreamStreamError(RuntimeError):
    pass


class DownstreamStreamError(RuntimeError):
    pass


class DownstreamDisconnected(DownstreamStreamError):
    pass


async def _watch_disconnect(receive: Any) -> None:
    while True:
        try:
            message = await receive()
        except Exception:
            return
        if message.get("type") == "http.disconnect":
            return


async def _await_or_disconnect(awaitable: Any, disconnect: asyncio.Task[None]) -> Any:
    operation = asyncio.create_task(awaitable)
    try:
        done, _ = await asyncio.wait(
            (operation, disconnect),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect in done:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise DownstreamDisconnected("ASGI client disconnected")
        return await operation
    except asyncio.CancelledError:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        raise


async def _next_with_timeout(iterator: Any, timeout_s: float) -> bytes | None:
    try:
        async with asyncio.timeout(timeout_s):
            return await anext(iterator)
    except StopAsyncIteration:
        return None
    except TimeoutError as error:
        raise UpstreamStreamError("upstream stream stalled") from error
    except (httpx.HTTPError, OSError) as error:
        raise UpstreamStreamError("upstream stream failed") from error


async def relay_raw_response(
    response: httpx.Response,
    receive: Any,
    send: ASGISend,
    *,
    read_inactivity_timeout_s: float,
) -> int:
    """Start once and relay bounded raw bytes while observing ASGI disconnects."""
    disconnect = asyncio.create_task(_watch_disconnect(receive))
    try:
        try:
            await _await_or_disconnect(
                send(
                    {
                        "type": "http.response.start",
                        "status": response.status_code,
                        "headers": list(response_headers(response)),
                    }
                ),
                disconnect,
            )
        except asyncio.CancelledError:
            raise
        except DownstreamDisconnected:
            raise
        except Exception as error:
            raise DownstreamStreamError("downstream rejected response headers") from error
        sent = 0
        iterator = response.aiter_raw(RAW_CHUNK_BYTES).__aiter__()
        while True:
            chunk = await _await_or_disconnect(
                _next_with_timeout(iterator, read_inactivity_timeout_s),
                disconnect,
            )
            if chunk is None:
                break
            try:
                await _await_or_disconnect(
                    send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": True,
                        }
                    ),
                    disconnect,
                )
            except asyncio.CancelledError:
                raise
            except DownstreamDisconnected:
                raise
            except Exception as error:
                raise DownstreamStreamError("downstream stream failed") from error
            sent += len(chunk)
        try:
            await _await_or_disconnect(
                send({"type": "http.response.body", "body": b"", "more_body": False}),
                disconnect,
            )
        except asyncio.CancelledError:
            raise
        except DownstreamDisconnected:
            raise
        except Exception as error:
            raise DownstreamStreamError("downstream stream finalization failed") from error
        return sent
    finally:
        disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)


async def read_prestart_error(
    response: httpx.Response,
    *,
    read_inactivity_timeout_s: float,
) -> bytes:
    result = bytearray()
    iterator = response.aiter_raw(RAW_CHUNK_BYTES).__aiter__()
    while True:
        chunk = await _next_with_timeout(iterator, read_inactivity_timeout_s)
        if chunk is None:
            return bytes(result)
        if len(result) + len(chunk) > MAX_PRESTART_ERROR_BYTES:
            raise UpstreamStreamError(
                "upstream error envelope exceeds the bounded classifier input"
            )
        result.extend(chunk)


async def drain_raw_response(
    response: httpx.Response,
    *,
    read_inactivity_timeout_s: float,
) -> int:
    """Consume one non-serving response with the same bounded raw iterator."""
    total = 0
    iterator = response.aiter_raw(RAW_CHUNK_BYTES).__aiter__()
    while True:
        chunk = await _next_with_timeout(iterator, read_inactivity_timeout_s)
        if chunk is None:
            return total
        total += len(chunk)


async def relay_buffered_response(
    response: httpx.Response,
    body: bytes,
    send: ASGISend,
) -> None:
    try:
        await send(
            {
                "type": "http.response.start",
                "status": response.status_code,
                "headers": list(response_headers(response)),
            }
        )
        for offset in range(0, len(body), RAW_CHUNK_BYTES):
            await send(
                {
                    "type": "http.response.body",
                    "body": body[offset : offset + RAW_CHUNK_BYTES],
                    "more_body": True,
                }
            )
        await send({"type": "http.response.body", "body": b"", "more_body": False})
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise DownstreamStreamError("downstream buffered response failed") from error
