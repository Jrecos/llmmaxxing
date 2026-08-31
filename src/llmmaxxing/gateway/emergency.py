"""Host singleton, contraction-only root UDS, readiness and process lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import json
import os
import socket
import stat
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol
from llmmaxxing.core.wire import (
    GatewayCommandV1,
    GatewayReadiness,
    GatewayStatusV1,
    StatusCommandPayload,
    WireCommandKind,
)
from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.gateway.activation import ActivationService, GatewayLocalState


class ClosableASGI(Protocol):
    async def __call__(
        self,
        scope: Mapping[str, Any],
        receive: Callable[[], Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None: ...

    async def aclose(self) -> None: ...


class GatewaySingleton:
    """Nonblocking process-lifetime flock on the persistent Gateway data directory."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "gateway.lock"
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True
        if self.data_dir.is_symlink():
            raise RuntimeError("singleton data directory may not be a symlink")
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.EMLINK):
                raise RuntimeError("singleton lock may not be a symlink") from error
            raise
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> GatewaySingleton:
        if not self.acquire():
            raise RuntimeError("another Gateway dispatcher owns the singleton")
        return self

    def __exit__(self, *_error: object) -> None:
        self.release()


class EmergencyServer:
    """Root-peer UDS accepting only status and monotonic deny contractions."""

    def __init__(
        self,
        path: str | Path,
        service: ActivationService,
        *,
        required_uid: int = 0,
        crash_injector: Callable[[str], None] | None = None,
        max_command_bytes: int = 64 * 1024,
    ) -> None:
        if required_uid < 0 or max_command_bytes < 1:
            raise ValueError("invalid emergency server bounds")
        self.path = Path(path)
        self.service = service
        self.required_uid = required_uid
        self.max_command_bytes = max_command_bytes
        self._crash = crash_injector or (lambda _point: None)
        self._server: asyncio.AbstractServer | None = None
        self._inode: tuple[int, int] | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        if os.geteuid() != self.required_uid:
            label = "root" if self.required_uid == 0 else f"uid {self.required_uid}"
            raise PermissionError(f"emergency socket requires {label}")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.lstat(self.path)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("refusing existing emergency socket path")
        server = await asyncio.start_unix_server(
            self._handle,
            path=self.path,
            limit=self.max_command_bytes + 1,
        )
        try:
            os.chmod(self.path, 0o600)
            metadata = os.lstat(self.path)
            if not stat.S_ISSOCK(metadata.st_mode):
                raise RuntimeError("emergency path is not a fresh socket")
            if metadata.st_uid != self.required_uid:
                raise PermissionError("emergency socket owner is not the required uid")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise PermissionError("emergency socket mode is not 0600")
        except BaseException:
            server.close()
            await server.wait_closed()
            with contextlib.suppress(OSError):
                self.path.unlink()
            raise
        self._server = server
        self._inode = (metadata.st_dev, metadata.st_ino)

    def _peer_allowed(self, writer: asyncio.StreamWriter) -> bool:
        connection = writer.get_extra_info("socket")
        if connection is None or not hasattr(socket, "SO_PEERCRED"):
            return False
        try:
            _pid, uid, _gid = struct.unpack(
                "3i",
                connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")),
            )
        except OSError:
            return False
        return bool(uid == self.required_uid)

    async def _reply(self, writer: asyncio.StreamWriter, value: object) -> None:
        writer.write(canonical_json_bytes(value) + b"\n")
        await writer.drain()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if not self._peer_allowed(writer):
                await self._reply(writer, {"error": "emergency_peer_unauthorized"})
                return
            try:
                line = await reader.readuntil(b"\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                await self._reply(writer, {"error": "emergency_command_invalid"})
                return
            raw = line[:-1]
            if len(raw) > self.max_command_bytes:
                await self._reply(writer, {"error": "emergency_command_invalid"})
                return
            try:
                command = GatewayCommandV1.model_validate(json.loads(raw))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                await self._reply(writer, {"error": "emergency_command_invalid"})
                return
            if canonical_json_bytes(command.model_dump(mode="json")) != raw:
                await self._reply(writer, {"error": "emergency_command_noncanonical"})
                return
            if command.kind not in (WireCommandKind.STATUS, WireCommandKind.DENY):
                await self._reply(writer, {"error": "emergency_command_not_contraction"})
                return
            if (
                command.kind is WireCommandKind.STATUS
                and StatusCommandPayload.model_validate(command.payload).issue_auth_lease
            ):
                await self._reply(writer, {"error": "emergency_command_not_contraction"})
                return
            try:
                ack = await self.service.execute(command)
            except Exception:
                await self._reply(writer, {"error": "emergency_command_rejected"})
                return
            self._crash("uds_after_ack_before_reply")
            await self._reply(writer, ack.model_dump(mode="json"))
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._inode is not None:
            try:
                metadata = os.lstat(self.path)
            except FileNotFoundError:
                pass
            else:
                if (metadata.st_dev, metadata.st_ino) == self._inode and stat.S_ISSOCK(
                    metadata.st_mode
                ):
                    self.path.unlink()
                    _fsync_parent(self.path)
        self._inode = None


def _fsync_parent(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class GatewayRuntime:
    """Single-worker ASGI lifecycle and local health ownership around Task8's app."""

    def __init__(
        self,
        app: ClosableASGI,
        *,
        state: GatewayLocalState,
        singleton: GatewaySingleton,
        backend_ready: Callable[[], bool],
        capacities_ready: Callable[[], bool],
        emergency: EmergencyServer | None = None,
    ) -> None:
        self.app = app
        self.state = state
        self.singleton = singleton
        self.backend_ready = backend_ready
        self.capacities_ready = capacities_ready
        self.emergency = emergency
        self._started = False

    async def startup(self) -> None:
        if self._started:
            return
        if not self.singleton.acquire():
            raise RuntimeError("another Gateway dispatcher owns the singleton")
        try:
            if self.emergency is not None:
                await self.emergency.start()
            self._started = True
        except BaseException:
            self.singleton.release()
            raise

    def status(self) -> GatewayStatusV1:
        try:
            backend_ready = bool(self.backend_ready())
        except Exception:
            backend_ready = False
        try:
            capacities_ready = bool(self.capacities_ready())
        except Exception:
            capacities_ready = False
        return self.state.status(
            singleton_held=self.singleton.held,
            backend_ready=backend_ready,
            capacities_ready=capacities_ready,
        )

    async def shutdown(self) -> None:
        try:
            if self.emergency is not None:
                await self.emergency.stop()
        finally:
            try:
                await self.app.aclose()
            finally:
                await asyncio.to_thread(self.state.close)
                self.singleton.release()
                self._started = False

    async def __call__(
        self,
        scope: Mapping[str, Any],
        receive: Callable[[], Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None:
        if scope.get("type") == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope.get("type") == "http" and scope.get("path") in (
            "/health/liveness",
            "/health/readiness",
        ):
            await self._health(str(scope["path"]), send)
            return
        if not self._started or self.status().readiness is not GatewayReadiness.READY:
            await _send_json(send, 503, {"error": "gateway_unready"})
            return
        await self.app(scope, receive, send)

    async def _lifespan(
        self,
        receive: Callable[[], Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await self.startup()
                except Exception as error:
                    await send({"type": "lifespan.startup.failed", "message": str(error)})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _health(
        self,
        path: str,
        send: Callable[[dict[str, Any]], Any],
    ) -> None:
        status_value = self.status()
        if path == "/health/liveness":
            await _send_json(send, 200, {"status": "live"})
        else:
            await _send_json(
                send,
                200 if status_value.readiness is GatewayReadiness.READY else 503,
                status_value.model_dump(mode="json"),
            )


async def _send_json(
    send: Callable[[dict[str, Any]], Any],
    status: int,
    value: object,
) -> None:
    body = canonical_json_bytes(value)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
