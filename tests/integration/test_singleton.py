from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from integration.test_activation import (
    activate,
    async_test,
    make_harness,
    signed_policy,
)
from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.ids import GatewayBootId, InstallationId
from llmmaxxing.core.wire import (
    ChannelSigner,
    ChannelTrustSet,
    ClearDenyCommandPayload,
    DenyCommandPayload,
    DenyReason,
    DenySubjectType,
    GatewayReadiness,
    StatusCommandPayload,
    TakeoverState,
    WireCommandKind,
)
from llmmaxxing.gateway.activation import GatewayLocalState, TakeoverCoordinator
from llmmaxxing.gateway.emergency import EmergencyServer, GatewayRuntime, GatewaySingleton
from support.gateway_stack import ACCOUNT_ID, PEPPER, PEPPER_VERSION, make_bundle


def _hold_lock(path: str, ready: multiprocessing.connection.Connection) -> None:  # type: ignore[name-defined]
    lock = GatewaySingleton(path)
    ready.send(lock.acquire())
    ready.close()
    while True:
        __import__("time").sleep(1)


def test_singleton_is_process_lifetime_nonblocking_and_crash_releases(tmp_path: Path) -> None:
    first = GatewaySingleton(tmp_path)
    second = GatewaySingleton(tmp_path)
    assert first.acquire()
    assert first.held
    assert not second.acquire()
    assert not second.held
    first.release()
    assert second.acquire()
    second.release()

    parent, child = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=_hold_lock, args=(str(tmp_path), child))
    process.start()
    assert parent.recv() is True
    contender = GatewaySingleton(tmp_path)
    assert not contender.acquire()
    process.kill()
    process.join(timeout=5)
    assert process.exitcode is not None
    assert contender.acquire()
    contender.release()


def test_singleton_and_uds_refuse_symlink_attachment(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("not a lock")
    lock = tmp_path / "gateway.lock"
    lock.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        GatewaySingleton(tmp_path).acquire()


async def _request(path: Path, command: Any) -> bytes:
    reader, writer = await asyncio.open_unix_connection(path)
    writer.write(canonical_json_bytes(command.model_dump(mode="json")) + b"\n")
    await writer.drain()
    response = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return response


@async_test
async def test_root_uds_is_0600_contraction_only_and_replays_durable_ack(tmp_path: Path) -> None:
    harness = make_harness(tmp_path / "state")
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    harness.state.renew_deny_heartbeats(harness.clock.now_ms())
    socket_path = tmp_path / "gateway.sock"
    crashes = {"enabled": True}

    def crash(point: str) -> None:
        if point == "uds_after_ack_before_reply" and crashes["enabled"]:
            crashes["enabled"] = False
            raise ConnectionAbortedError(point)

    server = EmergencyServer(
        socket_path,
        harness.service,
        required_uid=os.geteuid(),
        crash_injector=crash,
    )
    await server.start()
    metadata = socket_path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_uid == os.geteuid()

    status = harness.command(
        WireCommandKind.STATUS,
        StatusCommandPayload(issue_auth_lease=False),
        policy=None,
    )
    assert await _request(socket_path, status) == b""
    durable_count = harness.state.command_count
    replay = await _request(socket_path, status)
    assert replay
    assert harness.state.command_count == durable_count

    harness.sequence = status.sequence
    from llmmaxxing.core.wire import gateway_command_digest

    harness.previous_digest = gateway_command_digest(status)
    lease_status = harness.command(
        WireCommandKind.STATUS,
        StatusCommandPayload(issue_auth_lease=True),
        policy=None,
    )
    rejected = json.loads((await _request(socket_path, lease_status)).decode())
    assert rejected == {"error": "emergency_command_not_contraction"}
    assert harness.state.command_count == durable_count

    clear = harness.command(
        WireCommandKind.CLEAR_DENY,
        ClearDenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=1,
        ),
        policy=signed_policy(harness.policy_key, bundle, bundle),
    )
    rejected = json.loads((await _request(socket_path, clear)).decode())
    assert rejected == {"error": "emergency_command_not_contraction"}
    assert harness.state.command_count == durable_count

    deny = harness.command(
        WireCommandKind.DENY,
        DenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=1,
            reason=DenyReason.EMERGENCY,
        ),
        policy=signed_policy(harness.policy_key, bundle, bundle),
    )
    response = await _request(socket_path, deny)
    assert json.loads(response)["status"] == "denied"
    await server.stop()
    assert not socket_path.exists()


@async_test
async def test_uds_requires_root_by_default_and_never_unlinks_foreign_inode(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("non-root enforcement needs an unprivileged test user")
    harness = make_harness(tmp_path / "state")
    socket_path = tmp_path / "root.sock"
    with pytest.raises(PermissionError, match="root"):
        await EmergencyServer(socket_path, harness.service).start()
    assert not socket_path.exists()

    foreign = tmp_path / "foreign.sock"
    foreign.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="existing"):
        await EmergencyServer(
            foreign,
            harness.service,
            required_uid=os.geteuid(),
        ).start()
    assert foreign.is_symlink()


@async_test
async def test_takeover_requires_signed_old_backend_fence_and_advances_once(tmp_path: Path) -> None:
    old_installation = InstallationId.new()
    new_installation = InstallationId.new()
    old = make_harness(tmp_path / "old", installation_id=old_installation)
    new = make_harness(tmp_path / "new", installation_id=new_installation)
    old_signer = ChannelSigner("old-gateway", 5, old.channel_key)
    old_trust = ChannelTrustSet(
        installation_id=old_installation,
        security_epoch=3,
        channel_epochs={1: {5: {"old-gateway": old.channel_key.public_key()}}},
    )
    coordinator = TakeoverCoordinator(
        old.state,
        old_signer,
        old_trust,
        old.service.dispatcher_gate,
    )
    receipt = await coordinator.fence_old(
        target_installation_id=new_installation,
        credential_digest="a" * 64,
        network_digest="b" * 64,
        fenced_at_ms=old.clock.now_ms(),
    )
    assert old.state.lifecycle.value == "fenced_old"
    assert (
        old.state.status(
            singleton_held=True,
            backend_ready=True,
            capacities_ready=True,
            now_ms=old.clock.now_ms(),
        ).readiness
        is not GatewayReadiness.READY
    )

    receiver = TakeoverCoordinator(
        new.state,
        new.service.ack_signer,
        old_trust,
        new.service.dispatcher_gate,
    )
    await receiver.accept(receipt)
    assert new.state.fence_epoch == receipt.payload.fence_epoch
    assert new.state.takeover_state is TakeoverState.RELEASED
    with pytest.raises(Exception, match="fence"):
        await receiver.accept(receipt)


@async_test
async def test_takeover_crash_after_receipt_is_fenced_old_until_same_receipt_resumes(
    tmp_path: Path,
) -> None:
    old_installation = InstallationId.new()
    new_installation = InstallationId.new()
    old = make_harness(tmp_path / "old", installation_id=old_installation)
    new = make_harness(tmp_path / "new", installation_id=new_installation)
    signer = ChannelSigner("old-gateway", 5, old.channel_key)
    trust = ChannelTrustSet(
        installation_id=old_installation,
        security_epoch=3,
        channel_epochs={1: {5: {"old-gateway": old.channel_key.public_key()}}},
    )
    receipt = await TakeoverCoordinator(
        old.state,
        signer,
        trust,
        old.service.dispatcher_gate,
    ).fence_old(
        target_installation_id=new_installation,
        credential_digest="c" * 64,
        network_digest="d" * 64,
        fenced_at_ms=old.clock.now_ms(),
    )
    new.state._crash = lambda point: (
        (_ for _ in ()).throw(RuntimeError(point)) if point == "takeover_after_receipt" else None
    )
    with pytest.raises(RuntimeError, match="takeover_after_receipt"):
        await TakeoverCoordinator(
            new.state,
            new.service.ack_signer,
            trust,
            new.service.dispatcher_gate,
        ).accept(receipt)
    new.state.close()

    reopened = GatewayLocalState(
        tmp_path / "new",
        installation_id=new_installation,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=new.clock,
    ).open()
    assert reopened.takeover_state is TakeoverState.FENCED_OLD
    assert reopened.deny_all
    await TakeoverCoordinator(
        reopened,
        new.service.ack_signer,
        trust,
        new.service.dispatcher_gate,
    ).accept(receipt)
    assert reopened.takeover_state is TakeoverState.RELEASED


class _InnerApp:
    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    async def __call__(self, _scope: Any, _receive: Any, send: Any) -> None:
        self.calls += 1
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def aclose(self) -> None:
        self.closed = True


@async_test
async def test_runtime_readiness_and_shutdown_own_singleton_app_and_store(tmp_path: Path) -> None:
    harness = make_harness(tmp_path / "state")
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    harness.state.renew_deny_heartbeats(harness.clock.now_ms())
    status = harness.command(
        WireCommandKind.STATUS,
        StatusCommandPayload(issue_auth_lease=True),
        policy=None,
    )
    await harness.execute(status)
    inner = _InnerApp()
    singleton = GatewaySingleton(tmp_path / "state")
    runtime = GatewayRuntime(
        inner,
        state=harness.state,
        singleton=singleton,
        backend_ready=lambda: True,
        capacities_ready=lambda: True,
    )
    await runtime.startup()
    assert runtime.status().readiness is GatewayReadiness.READY
    await runtime.shutdown()
    assert inner.closed
    assert not singleton.held
    with pytest.raises(RuntimeError, match="not open"):
        _ = harness.state.db
