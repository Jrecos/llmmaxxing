from __future__ import annotations

import asyncio
import base64
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from llmmaxxing.config.signing import ActivationEnvelope, sign_activation
from llmmaxxing.core.canonical import bundle_hash, canonical_bundle_bytes
from llmmaxxing.core.ids import CommandId, GatewayBootId, InstallationId
from llmmaxxing.core.wire import (
    GENESIS_BASE_BUNDLE_HASH,
    BaseReference,
    BundleReference,
    ChannelSealV1,
    ChannelSigner,
    ChannelTrustSet,
    ClearDenyCommandPayload,
    CommandChainGap,
    CommandInstallationMismatch,
    DenyCommandPayload,
    DenyReason,
    DenySubjectType,
    GatewayCommandV1,
    GatewayLifecycle,
    GatewayReadiness,
    PrepareCommandPayload,
    SignedActivationV1,
    StaleFenceEpoch,
    StatusCommandPayload,
    WireCommandKind,
    gateway_command_digest,
    seal_gateway_command,
    verify_gateway_ack,
)
from llmmaxxing.gateway.activation import (
    ActivationService,
    BundleLimits,
    DispatcherGate,
    GatewayLocalState,
)
from llmmaxxing.gateway.journal import InjectedCrash
from support.gateway_stack import ACCOUNT_ID, GROUP_ID, PEPPER, PEPPER_VERSION, make_bundle

ZERO_DIGEST = "0" * 64
NOW_MS = 1_800_000_000_000


def async_test(function):  # type: ignore[no-untyped-def]
    @wraps(function)
    def run(*args, **kwargs):  # type: ignore[no-untyped-def]
        return asyncio.run(function(*args, **kwargs))

    return run


class Clock:
    def __init__(self, now_ms: int = NOW_MS) -> None:
        self.value = now_ms

    def now_ms(self) -> int:
        return self.value


class GenerationGate:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def permits(self, _leg: Any, backend_manifest_hash: str) -> bool:
        return self.allowed and backend_manifest_hash == "e" * 64


@dataclass
class Applied:
    bundles: list[int] = field(default_factory=list)

    def __call__(self, bundle: Any) -> None:
        self.bundles.append(bundle.generation)


@dataclass
class Harness:
    state: GatewayLocalState
    service: ActivationService
    clock: Clock
    policy_key: Ed25519PrivateKey
    channel_key: Ed25519PrivateKey
    installation_id: InstallationId
    boot_id: GatewayBootId
    applied: Applied
    sequence: int = 0
    previous_digest: str = ZERO_DIGEST

    def command(
        self,
        kind: WireCommandKind,
        payload: Any,
        *,
        policy: SignedActivationV1 | None,
        command_id: CommandId | None = None,
        sequence: int | None = None,
        previous_digest: str | None = None,
        installation_id: InstallationId | None = None,
        fence_epoch: int | None = None,
    ) -> GatewayCommandV1:
        unsigned = GatewayCommandV1(
            command_id=command_id or CommandId.new(),
            installation_id=installation_id or self.installation_id,
            channel_epoch=1,
            security_epoch=3,
            dispatcher_fence=fence_epoch or self.state.fence_epoch,
            boot_id=self.boot_id,
            sequence=sequence or self.sequence + 1,
            previous_digest=previous_digest or self.previous_digest,
            kind=kind,
            issued_at_ms=self.clock.now_ms(),
            payload=payload.model_dump(mode="json"),
            policy=policy,
            channel_seal=ChannelSealV1(
                seal_id="control-channel",
                trust_epoch=5,
                signature="0" * 128,
            ),
        )
        return seal_gateway_command(unsigned, self.channel_key)

    async def execute(self, command: GatewayCommandV1):
        ack = await self.service.execute(command)
        self.sequence = command.sequence
        self.previous_digest = gateway_command_digest(command)
        return ack


def signed_policy(
    policy_key: Ed25519PrivateKey,
    target: Any,
    base: Any | None,
) -> SignedActivationV1:
    envelope = ActivationEnvelope(
        trust_epoch=7,
        signer_key_id="policy-primary",
        base_generation=0 if base is None else base.generation,
        base_bundle_hash=(
            GENESIS_BASE_BUNDLE_HASH if base is None else bundle_hash(canonical_bundle_bytes(base))
        ),
        target_generation=target.generation,
        target_content_hash=bundle_hash(canonical_bundle_bytes(target)),
        source_fingerprint="1" * 64,
        security_fence="2" * 64,
        key_set_fence="3" * 64,
        impact_hash="4" * 64,
    )
    return SignedActivationV1.from_signed(
        sign_activation(
            envelope,
            policy_key,
            base_bundle=base,
            target_bundle=target,
        )
    )


def prepare_payload(target: Any, base: Any | None) -> PrepareCommandPayload:
    raw = canonical_bundle_bytes(target)
    return PrepareCommandPayload(
        base=(
            None
            if base is None
            else BaseReference(
                generation=base.generation,
                bundle_hash=bundle_hash(canonical_bundle_bytes(base)),
            )
        ),
        target=BundleReference(
            generation=target.generation,
            bundle_hash=bundle_hash(raw),
        ),
        bundle_b64=base64.b64encode(raw).decode("ascii"),
    )


def make_harness(
    path: Path,
    *,
    clock: Clock | None = None,
    crash_at: str | None = None,
    generation_allowed: bool = True,
    installation_id: InstallationId | None = None,
    boot_id: GatewayBootId | None = None,
    limits: BundleLimits | None = None,
) -> Harness:
    clock = clock or Clock()
    policy_key = Ed25519PrivateKey.generate()
    channel_key = Ed25519PrivateKey.generate()
    installation_id = installation_id or InstallationId.new()
    boot_id = boot_id or GatewayBootId.new()

    def crash(point: str) -> None:
        if point == crash_at:
            raise InjectedCrash(point)

    state = GatewayLocalState(
        path,
        installation_id=installation_id,
        boot_id=boot_id,
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=clock,
        crash_injector=crash,
        limits=limits,
    ).open()
    applied = Applied()
    service = ActivationService(
        state=state,
        policy_keys={7: {"policy-primary": policy_key.public_key()}},
        channel_trust=ChannelTrustSet(
            installation_id=installation_id,
            security_epoch=3,
            channel_epochs={
                1: {
                    5: {"control-channel": channel_key.public_key()},
                    6: {"gateway-channel": channel_key.public_key()},
                }
            },
        ),
        ack_signer=ChannelSigner("gateway-channel", 6, channel_key),
        generation_gate=GenerationGate(generation_allowed),
        backend_manifest_hash="e" * 64,
        dispatcher_gate=DispatcherGate(),
        apply_bundle=applied,
        preflight_bundle=lambda _bundle: None,
        clock=clock,
        limits=limits,
    )
    return Harness(
        state,
        service,
        clock,
        policy_key,
        channel_key,
        installation_id,
        boot_id,
        applied,
    )


async def activate(harness: Harness, target: Any, base: Any | None):
    policy = signed_policy(harness.policy_key, target, base)
    prepare = harness.command(
        WireCommandKind.PREPARE,
        prepare_payload(target, base),
        policy=policy,
    )
    prepared = await harness.execute(prepare)
    commit = harness.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=target.generation,
            bundle_hash=bundle_hash(canonical_bundle_bytes(target)),
        ),
        policy=policy,
    )
    committed = await harness.execute(commit)
    return prepared, committed


@async_test
async def test_genesis_stages_then_commits_one_real_bundle_without_synthetic_base(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    target, _ = make_bundle()
    policy = signed_policy(harness.policy_key, target, None)

    prepare = harness.command(
        WireCommandKind.PREPARE,
        prepare_payload(target, None),
        policy=policy,
    )
    await harness.execute(prepare)
    status = harness.state.status(
        singleton_held=True,
        backend_ready=True,
        capacities_ready=True,
        now_ms=harness.clock.now_ms(),
    )
    assert status.active is None
    assert status.staged == BundleReference(
        generation=1,
        bundle_hash=bundle_hash(canonical_bundle_bytes(target)),
    )
    assert status.lifecycle is GatewayLifecycle.STAGED
    assert not harness.state.bundle_path(GENESIS_BASE_BUNDLE_HASH).exists()
    assert harness.applied.bundles == []

    commit = harness.command(
        WireCommandKind.COMMIT,
        status.staged,
        policy=policy,
    )
    await harness.execute(commit)
    status = harness.state.status(
        singleton_held=True,
        backend_ready=True,
        capacities_ready=True,
        now_ms=harness.clock.now_ms(),
    )
    assert status.active == BundleReference(
        generation=1,
        bundle_hash=bundle_hash(canonical_bundle_bytes(target)),
    )
    assert status.previous is None
    assert status.staged is None
    assert harness.applied.bundles == [1]


@async_test
async def test_command_auth_chain_and_dedupe_fail_before_any_new_durable_write(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    target, _ = make_bundle()
    policy = signed_policy(harness.policy_key, target, None)
    command = harness.command(
        WireCommandKind.PREPARE,
        prepare_payload(target, None),
        policy=policy,
    )
    invalid = command.model_copy(
        update={"channel_seal": command.channel_seal.model_copy(update={"signature": "f" * 128})}
    )
    before = harness.state.export_state()
    with pytest.raises(Exception, match="signature"):
        await harness.service.execute(invalid)
    assert harness.state.export_state() == before

    first_ack = await harness.execute(command)
    command_count = harness.state.command_count
    assert await harness.service.execute(command) == first_ack
    assert (
        verify_gateway_ack(
            first_ack,
            ChannelTrustSet(
                installation_id=harness.installation_id,
                security_epoch=3,
                channel_epochs={
                    1: {
                        6: {"gateway-channel": harness.channel_key.public_key()},
                    }
                },
            ),
            command=command,
        )
        == first_ack
    )
    assert harness.state.command_count == command_count

    reordered = harness.command(
        WireCommandKind.STATUS,
        StatusCommandPayload(issue_auth_lease=False),
        policy=None,
        sequence=command.sequence + 2,
        previous_digest=gateway_command_digest(command),
    )
    before = harness.state.export_state()
    with pytest.raises(CommandChainGap):
        await harness.service.execute(reordered)
    assert harness.state.export_state() == before

    wrong_installation = harness.command(
        WireCommandKind.STATUS,
        StatusCommandPayload(issue_auth_lease=False),
        policy=None,
        installation_id=InstallationId.new(),
    )
    with pytest.raises(CommandInstallationMismatch):
        await harness.service.execute(wrong_installation)
    assert harness.state.export_state() == before


@async_test
async def test_command_waiting_on_activation_gate_rechecks_dispatcher_fence(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    await activate(harness, first, None)
    policy = signed_policy(harness.policy_key, second, first)
    await harness.execute(
        harness.command(
            WireCommandKind.PREPARE,
            prepare_payload(second, first),
            policy=policy,
        )
    )
    commit = harness.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=2,
            bundle_hash=bundle_hash(canonical_bundle_bytes(second)),
        ),
        policy=policy,
    )
    async with harness.service.dispatcher_gate.hold_activation():
        pending = asyncio.create_task(harness.service.execute(commit))
        await asyncio.sleep(0)
        harness.state._fence_epoch = 2
    with pytest.raises(StaleFenceEpoch):
        await pending
    assert harness.state.active_reference is not None
    assert harness.state.active_reference.generation == 1


@async_test
async def test_prepare_rejects_noncanonical_manifest_gate_and_resource_overflow_without_writes(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path, generation_allowed=False)
    target, _ = make_bundle()
    policy = signed_policy(harness.policy_key, target, None)
    command = harness.command(
        WireCommandKind.PREPARE,
        prepare_payload(target, None),
        policy=policy,
    )
    before = harness.state.export_state()
    with pytest.raises(ValueError, match="generation"):
        await harness.service.execute(command)
    assert harness.state.export_state() == before

    with pytest.raises(ValidationError):
        BundleLimits(max_bundle_bytes=0)

    oversized = prepare_payload(target, None).model_copy(
        update={"bundle_b64": base64.b64encode(b"x" * 1025).decode("ascii")}
    )
    small = ActivationService(
        state=harness.state,
        policy_keys={7: {"policy-primary": harness.policy_key.public_key()}},
        channel_trust=ChannelTrustSet(
            installation_id=harness.installation_id,
            security_epoch=3,
            channel_epochs={1: {5: {"control-channel": harness.channel_key.public_key()}}},
        ),
        ack_signer=ChannelSigner("gateway-channel", 6, harness.channel_key),
        generation_gate=GenerationGate(),
        backend_manifest_hash="e" * 64,
        dispatcher_gate=DispatcherGate(),
        apply_bundle=harness.applied,
        preflight_bundle=lambda _bundle: None,
        clock=harness.clock,
        limits=BundleLimits(max_bundle_bytes=1024),
    )
    oversized_command = harness.command(
        WireCommandKind.PREPARE,
        oversized,
        policy=policy,
    )
    with pytest.raises(ValueError, match="bundle size"):
        await small.execute(oversized_command)
    assert harness.state.export_state() == before


@async_test
async def test_deny_overlay_is_monotonic_fresh_fail_closed_and_not_cleared_by_activation(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    await activate(harness, first, None)
    harness.state.renew_deny_heartbeats(harness.clock.now_ms())
    leg = first.policies[0].authorized_legs[0]
    assert harness.state.permits(leg, GROUP_ID)

    deny = harness.command(
        WireCommandKind.DENY,
        DenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=1,
            reason=DenyReason.COMPROMISE,
            deny_floor_generation=1,
        ),
        policy=signed_policy(harness.policy_key, first, first),
    )
    await harness.execute(deny)
    assert not harness.state.permits(leg, GROUP_ID)

    second = first.model_copy(update={"generation": 2})
    await activate(harness, second, first)
    assert not harness.state.permits(second.policies[0].authorized_legs[0], GROUP_ID)

    stale = harness.command(
        WireCommandKind.DENY,
        DenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=1,
            reason=DenyReason.MAINTENANCE,
            deny_floor_generation=1,
        ),
        policy=signed_policy(harness.policy_key, second, second),
    )
    with pytest.raises(ValueError, match="deny epoch"):
        await harness.service.execute(stale)

    clear = harness.command(
        WireCommandKind.CLEAR_DENY,
        ClearDenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=1,
        ),
        policy=signed_policy(harness.policy_key, second, second),
    )
    await harness.execute(clear)
    harness.state.renew_deny_heartbeats(harness.clock.now_ms())
    assert harness.state.permits(second.policies[0].authorized_legs[0], GROUP_ID)

    harness.clock.value += 10_001
    assert not harness.state.permits(second.policies[0].authorized_legs[0], GROUP_ID)


@async_test
async def test_auth_lease_is_exact_non_authorizing_and_readiness_tracks_all_gates(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    await activate(harness, first, None)
    harness.state.renew_deny_heartbeats(harness.clock.now_ms())

    before = harness.state.status(
        singleton_held=True,
        backend_ready=True,
        capacities_ready=True,
        now_ms=harness.clock.now_ms(),
    )
    assert before.readiness is GatewayReadiness.UNREADY
    assert before.auth_lease is None

    status_command = harness.command(
        WireCommandKind.STATUS,
        StatusCommandPayload(issue_auth_lease=True),
        policy=None,
    )
    ack = await harness.execute(status_command)
    assert ack.result["status"]["auth_lease"]["bundle"]["generation"] == 1
    after = harness.state.status(
        singleton_held=True,
        backend_ready=True,
        capacities_ready=True,
        now_ms=harness.clock.now_ms(),
    )
    assert after.readiness is GatewayReadiness.READY

    second = first.model_copy(update={"generation": 2})
    await activate(harness, second, first)
    after_activation = harness.state.status(
        singleton_held=True,
        backend_ready=True,
        capacities_ready=True,
        now_ms=harness.clock.now_ms(),
    )
    assert after_activation.readiness is GatewayReadiness.UNREADY
    assert after_activation.auth_lease is None


@async_test
async def test_corrupt_active_never_falls_back_and_corrupt_staged_is_quarantined(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    await activate(harness, first, None)
    await activate(harness, second, first)
    active = harness.state.active_reference
    previous = harness.state.previous_reference
    assert active is not None and previous is not None
    harness.state.close()

    harness.state.bundle_path(active.bundle_hash).write_bytes(b"corrupt")
    reopened = GatewayLocalState(
        tmp_path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=harness.clock,
    ).open()
    status = reopened.status(
        singleton_held=True,
        backend_ready=True,
        capacities_ready=True,
        now_ms=harness.clock.now_ms(),
    )
    assert status.lifecycle is GatewayLifecycle.RECOVERY_REQUIRED
    assert status.active is None
    assert status.previous == previous
    assert reopened.deny_all
    reopened.close()

    clean = make_harness(tmp_path / "staged")
    await activate(clean, first, None)
    policy = signed_policy(clean.policy_key, second, first)
    await clean.execute(
        clean.command(
            WireCommandKind.PREPARE,
            prepare_payload(second, first),
            policy=policy,
        )
    )
    staged = clean.state.staged_reference
    clean.state.close()
    assert staged is not None
    clean.state.bundle_path(staged.bundle_hash).write_bytes(b"corrupt")
    recovered = GatewayLocalState(
        tmp_path / "staged",
        installation_id=clean.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=clean.clock,
    ).open()
    assert recovered.active_reference is not None
    assert recovered.active_reference.generation == 1
    assert recovered.staged_reference is None
    assert recovered.lifecycle is GatewayLifecycle.APPLIED
    recovered.close()


@async_test
@pytest.mark.parametrize(
    ("failpoint", "expected_generation"),
    [
        ("active_pointer_after_write", 1),
        ("active_pointer_after_fsync", 1),
        ("active_pointer_after_rename", 1),
        ("active_pointer_after_dir_fsync", 1),
        ("active_after_commit", 2),
        ("memory_after_swap", 2),
        ("ack_after_commit", 2),
    ],
)
async def test_commit_failpoints_recover_exactly_old_or_new(
    tmp_path: Path,
    failpoint: str,
    expected_generation: int,
) -> None:
    base_path = tmp_path / failpoint
    initial = make_harness(base_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    await activate(initial, first, None)
    policy = signed_policy(initial.policy_key, second, first)
    await initial.execute(
        initial.command(
            WireCommandKind.PREPARE,
            prepare_payload(second, first),
            policy=policy,
        )
    )
    initial.state.close()

    crashing = make_harness(
        base_path,
        crash_at=failpoint,
        installation_id=initial.installation_id,
        boot_id=initial.boot_id,
    )
    crashing.policy_key = initial.policy_key
    crashing.channel_key = initial.channel_key
    crashing.service.policy_keys = {7: {"policy-primary": initial.policy_key.public_key()}}
    crashing.service.channel_trust = ChannelTrustSet(
        installation_id=initial.installation_id,
        security_epoch=3,
        channel_epochs={1: {5: {"control-channel": initial.channel_key.public_key()}}},
    )
    crashing.service.ack_signer = ChannelSigner("gateway-channel", 6, initial.channel_key)
    crashing.sequence = initial.sequence
    crashing.previous_digest = initial.previous_digest
    command = crashing.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=2,
            bundle_hash=bundle_hash(canonical_bundle_bytes(second)),
        ),
        policy=policy,
    )
    with pytest.raises(InjectedCrash):
        await crashing.service.execute(command)
    crashing.state.close()

    recovered = GatewayLocalState(
        base_path,
        installation_id=initial.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=initial.clock,
    ).open()
    assert recovered.active_reference is not None
    assert recovered.active_reference.generation == expected_generation
    assert recovered.active_reference.generation in (1, 2)
    recovered.close()


@async_test
@pytest.mark.parametrize(
    ("failpoint", "staged"),
    [
        ("bundle_after_write", False),
        ("bundle_after_fsync", False),
        ("bundle_after_rename", False),
        ("bundle_after_dir_fsync", False),
        ("staged_before_commit", False),
        ("staged_after_commit", True),
    ],
)
async def test_prepare_failpoints_never_publish_or_partially_stage(
    tmp_path: Path,
    failpoint: str,
    staged: bool,
) -> None:
    path = tmp_path / failpoint
    harness = make_harness(path, crash_at=failpoint)
    target, _ = make_bundle()
    command = harness.command(
        WireCommandKind.PREPARE,
        prepare_payload(target, None),
        policy=signed_policy(harness.policy_key, target, None),
    )
    with pytest.raises(InjectedCrash):
        await harness.service.execute(command)
    harness.state.close()
    recovered = GatewayLocalState(
        path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=harness.clock,
    ).open()
    assert recovered.active_reference is None
    assert (recovered.staged_reference is not None) is staged
    recovered.close()


@async_test
@pytest.mark.parametrize(
    ("failpoint", "applied"),
    [
        ("active_pointer_after_write", False),
        ("active_pointer_after_fsync", False),
        ("active_pointer_after_rename", False),
        ("active_pointer_after_dir_fsync", False),
        ("active_after_commit", True),
        ("memory_after_swap", True),
        ("ack_after_commit", True),
    ],
)
async def test_genesis_commit_failpoints_are_absent_or_one_real_target(
    tmp_path: Path,
    failpoint: str,
    applied: bool,
) -> None:
    path = tmp_path / f"genesis-{failpoint}"
    harness = make_harness(path)
    target, _ = make_bundle()
    policy = signed_policy(harness.policy_key, target, None)
    await harness.execute(
        harness.command(
            WireCommandKind.PREPARE,
            prepare_payload(target, None),
            policy=policy,
        )
    )
    harness.state._crash = lambda point: (
        (_ for _ in ()).throw(InjectedCrash(point)) if point == failpoint else None
    )
    command = harness.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=1,
            bundle_hash=bundle_hash(canonical_bundle_bytes(target)),
        ),
        policy=policy,
    )
    with pytest.raises(InjectedCrash):
        await harness.service.execute(command)
    harness.state.close()
    recovered = GatewayLocalState(
        path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=harness.clock,
    ).open()
    assert (recovered.active_reference is not None) is applied
    assert not recovered.bundle_path(GENESIS_BASE_BUNDLE_HASH).exists()
    recovered.close()


@async_test
@pytest.mark.parametrize(
    ("failpoint", "denied"),
    [("deny_before_commit", False), ("deny_after_commit", True)],
)
async def test_deny_failpoints_are_whole_subject_records(
    tmp_path: Path,
    failpoint: str,
    denied: bool,
) -> None:
    path = tmp_path / failpoint
    harness = make_harness(path)
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    harness.state.renew_deny_heartbeats(harness.clock.now_ms())
    harness.state._crash = lambda point: (
        (_ for _ in ()).throw(InjectedCrash(point)) if point == failpoint else None
    )
    command = harness.command(
        WireCommandKind.DENY,
        DenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=1,
            reason=DenyReason.EMERGENCY,
        ),
        policy=signed_policy(harness.policy_key, bundle, bundle),
    )
    with pytest.raises(InjectedCrash):
        await harness.service.execute(command)
    harness.state.close()
    recovered = GatewayLocalState(
        path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=harness.clock,
    ).open()
    assert (
        bool(
            recovered.status(
                singleton_held=True,
                backend_ready=True,
                capacities_ready=True,
                now_ms=harness.clock.now_ms(),
            ).deny_overlay
        )
        is denied
    )
    recovered.close()


@async_test
async def test_corrupt_deny_overlay_is_recovery_required_deny_all(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    await harness.execute(
        harness.command(
            WireCommandKind.DENY,
            DenyCommandPayload(
                subject_type=DenySubjectType.ACCOUNT,
                subject_id=str(ACCOUNT_ID),
                deny_epoch=1,
                reason=DenyReason.COMPROMISE,
            ),
            policy=signed_policy(harness.policy_key, bundle, bundle),
        )
    )
    harness.state.close()
    connection = sqlite3.connect(tmp_path / "gateway-state.sqlite3")
    connection.execute("UPDATE denies SET reason = 'maintenance'")
    connection.commit()
    connection.close()
    recovered = GatewayLocalState(
        tmp_path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=harness.clock,
    ).open()
    assert recovered.lifecycle is GatewayLifecycle.RECOVERY_REQUIRED
    assert recovered.deny_all
    assert recovered.active_reference is None
    recovered.close()


def test_wire_models_are_frozen_exact_and_require_injected_trust() -> None:
    with pytest.raises(ValidationError):
        BaseReference(generation=0, bundle_hash=GENESIS_BASE_BUNDLE_HASH)
    ref = BundleReference(generation=1, bundle_hash=GENESIS_BASE_BUNDLE_HASH)
    with pytest.raises(ValidationError):
        BundleReference.model_validate({**ref.model_dump(), "extra": True})
    with pytest.raises(ValidationError):
        ref.generation = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="channel trust"):
        ChannelTrustSet(
            installation_id=InstallationId.new(),
            security_epoch=1,
            channel_epochs={},
        )


@async_test
async def test_normal_prepare_reports_staged_before_applied(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    await activate(harness, first, None)
    await harness.execute(
        harness.command(
            WireCommandKind.PREPARE,
            prepare_payload(second, first),
            policy=signed_policy(harness.policy_key, second, first),
        )
    )
    assert harness.state.lifecycle is GatewayLifecycle.STAGED


@async_test
async def test_commit_rechecks_deny_floor_after_prepare(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    await activate(harness, first, None)
    policy = signed_policy(harness.policy_key, second, first)
    await harness.execute(
        harness.command(
            WireCommandKind.PREPARE,
            prepare_payload(second, first),
            policy=policy,
        )
    )
    await harness.execute(
        harness.command(
            WireCommandKind.DENY,
            DenyCommandPayload(
                subject_type=DenySubjectType.ACCOUNT,
                subject_id=str(ACCOUNT_ID),
                deny_epoch=1,
                reason=DenyReason.ROLLBACK_HOLD,
                deny_floor_generation=2,
            ),
            policy=signed_policy(harness.policy_key, first, first),
        )
    )
    commit = harness.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=2,
            bundle_hash=bundle_hash(canonical_bundle_bytes(second)),
        ),
        policy=policy,
    )
    with pytest.raises(ValueError, match="deny floor"):
        await harness.service.execute(commit)
    assert harness.state.active_reference is not None
    assert harness.state.active_reference.generation == 1


@async_test
async def test_deny_epoch_and_floor_highwater_survive_clear(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    policy = signed_policy(harness.policy_key, bundle, bundle)
    await harness.execute(
        harness.command(
            WireCommandKind.DENY,
            DenyCommandPayload(
                subject_type=DenySubjectType.ACCOUNT,
                subject_id=str(ACCOUNT_ID),
                deny_epoch=5,
                reason=DenyReason.COMPROMISE,
                deny_floor_generation=5,
            ),
            policy=policy,
        )
    )
    await harness.execute(
        harness.command(
            WireCommandKind.CLEAR_DENY,
            ClearDenyCommandPayload(
                subject_type=DenySubjectType.ACCOUNT,
                subject_id=str(ACCOUNT_ID),
                deny_epoch=5,
            ),
            policy=policy,
        )
    )
    stale = harness.command(
        WireCommandKind.DENY,
        DenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=4,
            reason=DenyReason.EMERGENCY,
            deny_floor_generation=5,
        ),
        policy=policy,
    )
    with pytest.raises(ValueError, match="epoch"):
        await harness.service.execute(stale)
    weaker = harness.command(
        WireCommandKind.DENY,
        DenyCommandPayload(
            subject_type=DenySubjectType.ACCOUNT,
            subject_id=str(ACCOUNT_ID),
            deny_epoch=6,
            reason=DenyReason.EMERGENCY,
        ),
        policy=policy,
    )
    with pytest.raises(ValueError, match="floor"):
        await harness.service.execute(weaker)


@async_test
async def test_new_boot_invalidates_heartbeat_and_auth_lease(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    await harness.execute(
        harness.command(
            WireCommandKind.STATUS,
            StatusCommandPayload(issue_auth_lease=True),
            policy=None,
        )
    )
    harness.state.close()
    reopened = GatewayLocalState(
        tmp_path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
        channel_epoch=1,
        security_epoch=3,
        accepted_peppers={PEPPER_VERSION: PEPPER},
        clock=harness.clock,
    ).open()
    status = reopened.status(
        singleton_held=True,
        backend_ready=True,
        capacities_ready=True,
        now_ms=harness.clock.now_ms(),
    )
    assert not status.deny_heartbeat_fresh
    assert status.auth_lease is None
    assert status.readiness is GatewayReadiness.UNREADY


@async_test
async def test_exact_durable_ack_replays_across_boot(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    command = harness.command(
        WireCommandKind.PREPARE,
        prepare_payload(bundle, None),
        policy=signed_policy(harness.policy_key, bundle, None),
    )
    ack = await harness.service.execute(command)
    harness.state.close()
    restarted = make_harness(
        tmp_path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
    )
    restarted.policy_key = harness.policy_key
    restarted.channel_key = harness.channel_key
    restarted.service.policy_keys = {7: {"policy-primary": harness.policy_key.public_key()}}
    restarted.service.channel_trust = ChannelTrustSet(
        installation_id=harness.installation_id,
        security_epoch=3,
        channel_epochs={
            1: {
                5: {"control-channel": harness.channel_key.public_key()},
                6: {"gateway-channel": harness.channel_key.public_key()},
            }
        },
    )
    restarted.service.ack_signer = ChannelSigner("gateway-channel", 6, harness.channel_key)
    assert await restarted.service.execute(command) == ack


@async_test
async def test_live_swap_failure_enters_recovery_without_applied_ack(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    await activate(harness, first, None)
    policy = signed_policy(harness.policy_key, second, first)
    await harness.execute(
        harness.command(
            WireCommandKind.PREPARE,
            prepare_payload(second, first),
            policy=policy,
        )
    )
    commit = harness.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=2,
            bundle_hash=bundle_hash(canonical_bundle_bytes(second)),
        ),
        policy=policy,
    )
    harness.service.apply_bundle = lambda _bundle: (_ for _ in ()).throw(
        ValueError("live swap failed")
    )
    with pytest.raises(ValueError, match="live swap failed"):
        await harness.service.execute(commit)
    assert harness.state.lifecycle is GatewayLifecycle.RECOVERY_REQUIRED
    assert harness.state.deny_all
    assert harness.state.ack_for(commit.command_id) is None


@async_test
async def test_prepare_disk_work_does_not_block_event_loop(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    original = harness.state.store_bundle

    def slow_store(*args: Any) -> Any:
        import time

        time.sleep(0.05)
        return original(*args)

    harness.state.store_bundle = slow_store  # type: ignore[method-assign]
    pending = asyncio.create_task(
        harness.service.execute(
            harness.command(
                WireCommandKind.PREPARE,
                prepare_payload(bundle, None),
                policy=signed_policy(harness.policy_key, bundle, None),
            )
        )
    )
    await asyncio.sleep(0.01)
    assert not pending.done()
    await pending


def test_oversized_bundle_recovery_uses_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    raw = canonical_bundle_bytes(bundle)
    expected = bundle_hash(raw)
    path = harness.state.bundle_path(expected)
    path.write_bytes(raw)
    path.write_bytes(b"x" * (harness.state.limits.max_bundle_bytes + 1))

    def forbidden(_path: Path) -> bytes:
        raise AssertionError("unbounded Path.read_bytes used")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError, match="exceeds|corrupt"):
        harness.state.load_bundle(expected)


@async_test
async def test_cancelled_commit_holds_gate_until_durable_apply_and_ack(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    await activate(harness, first, None)
    policy = signed_policy(harness.policy_key, second, first)
    await harness.execute(
        harness.command(
            WireCommandKind.PREPARE,
            prepare_payload(second, first),
            policy=policy,
        )
    )
    commit = harness.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=2,
            bundle_hash=bundle_hash(canonical_bundle_bytes(second)),
        ),
        policy=policy,
    )
    started = threading.Event()
    release = threading.Event()
    original = harness.state.commit_staged

    def blocked(*args: Any, **kwargs: Any) -> Any:
        started.set()
        release.wait(5)
        return original(*args, **kwargs)

    harness.state.commit_staged = blocked  # type: ignore[method-assign]
    pending = asyncio.create_task(harness.service.execute(commit))
    await asyncio.to_thread(started.wait, 5)
    pending.cancel()
    gate_acquired = asyncio.Event()

    async def wait_gate() -> None:
        async with harness.service.dispatcher_gate.hold_activation():
            gate_acquired.set()

    waiter = asyncio.create_task(wait_gate())
    await asyncio.sleep(0.01)
    assert not gate_acquired.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await waiter
    assert harness.state.active_reference is not None
    assert harness.state.active_reference.generation == 2
    assert harness.state.ack_for(commit.command_id) is not None


@async_test
async def test_status_heartbeat_and_command_are_one_transaction(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    harness.clock.value += 10_001
    before = harness.state.export_state()
    harness.state._crash = lambda point: (
        (_ for _ in ()).throw(InjectedCrash(point)) if point == "status_after_heartbeat" else None
    )
    command = harness.command(
        WireCommandKind.STATUS,
        StatusCommandPayload(issue_auth_lease=False),
        policy=None,
    )
    with pytest.raises(InjectedCrash):
        await harness.service.execute(command)
    after = harness.state.export_state()
    assert after["deny_heartbeat_ms"] == before["deny_heartbeat_ms"]
    assert after["command_count"] == before["command_count"]
    assert harness.state.deny_all


@async_test
async def test_command_and_auth_lease_retention_are_bounded(tmp_path: Path) -> None:
    limits = BundleLimits(max_command_records=16, max_auth_leases=2)
    harness = make_harness(tmp_path, limits=limits)
    bundle, _ = make_bundle()
    await activate(harness, bundle, None)
    for _ in range(20):
        await harness.execute(
            harness.command(
                WireCommandKind.STATUS,
                StatusCommandPayload(issue_auth_lease=True),
                policy=None,
            )
        )
    assert harness.state.command_count <= 16
    leases = harness.state.db.execute("SELECT count(*) FROM auth_leases").fetchone()
    assert leases is not None and int(leases[0]) <= 2


@async_test
async def test_prepare_preflights_live_runtime_before_any_bundle_write(
    tmp_path: Path,
) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    harness.service.preflight_bundle = lambda _bundle: (_ for _ in ()).throw(
        ValueError("runtime publication invalid")
    )
    command = harness.command(
        WireCommandKind.PREPARE,
        prepare_payload(bundle, None),
        policy=signed_policy(harness.policy_key, bundle, None),
    )
    with pytest.raises(ValueError, match="runtime publication invalid"):
        await harness.service.execute(command)
    assert not harness.state.bundle_path(bundle_hash(canonical_bundle_bytes(bundle))).exists()
    assert harness.state.command_count == 0


def test_corrupt_staged_blob_is_quarantined_for_clean_restage(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    bundle, _ = make_bundle()
    raw = canonical_bundle_bytes(bundle)
    expected = bundle_hash(raw)
    harness.state.store_bundle(raw, expected)
    path = harness.state.bundle_path(expected)
    path.write_bytes(b"corrupt")
    harness.state._quarantine_bundle(expected)
    assert not path.exists()
    harness.state.store_bundle(raw, expected)
    assert path.read_bytes() == raw


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX hard process death")
@pytest.mark.parametrize(
    ("failpoint", "expected_generation"),
    [
        ("active_pointer_after_write", 1),
        ("active_pointer_after_fsync", 1),
        ("active_pointer_after_rename", 1),
        ("active_pointer_after_dir_fsync", 1),
        ("active_after_commit", 2),
        ("memory_after_swap", 2),
        ("ack_after_commit", 2),
    ],
)
def test_commit_failpoints_survive_actual_child_process_death(
    tmp_path: Path,
    failpoint: str,
    expected_generation: int,
) -> None:
    harness = make_harness(tmp_path)
    first, _ = make_bundle()
    second = first.model_copy(update={"generation": 2})
    asyncio.run(activate(harness, first, None))
    policy = signed_policy(harness.policy_key, second, first)
    asyncio.run(
        harness.execute(
            harness.command(
                WireCommandKind.PREPARE,
                prepare_payload(second, first),
                policy=policy,
            )
        )
    )
    commit = harness.command(
        WireCommandKind.COMMIT,
        BundleReference(
            generation=2,
            bundle_hash=bundle_hash(canonical_bundle_bytes(second)),
        ),
        policy=policy,
    )
    sequence = harness.sequence
    previous_digest = harness.previous_digest
    harness.state.close()
    child = os.fork()
    if child == 0:
        child_harness = make_harness(
            tmp_path,
            installation_id=harness.installation_id,
            boot_id=harness.boot_id,
        )
        child_harness.service.policy_keys = {7: {"policy-primary": harness.policy_key.public_key()}}
        child_harness.service.channel_trust = ChannelTrustSet(
            installation_id=harness.installation_id,
            security_epoch=3,
            channel_epochs={
                1: {
                    5: {"control-channel": harness.channel_key.public_key()},
                    6: {"gateway-channel": harness.channel_key.public_key()},
                }
            },
        )
        child_harness.service.ack_signer = ChannelSigner("gateway-channel", 6, harness.channel_key)
        child_harness.sequence = sequence
        child_harness.previous_digest = previous_digest
        child_harness.state._crash = lambda point: os._exit(91) if point == failpoint else None
        asyncio.run(child_harness.service.execute(commit))
        os._exit(92)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 91
    restarted = make_harness(
        tmp_path,
        installation_id=harness.installation_id,
        boot_id=GatewayBootId.new(),
    )
    restarted.service.policy_keys = {7: {"policy-primary": harness.policy_key.public_key()}}
    restarted.service.channel_trust = ChannelTrustSet(
        installation_id=harness.installation_id,
        security_epoch=3,
        channel_epochs={
            1: {
                5: {"control-channel": harness.channel_key.public_key()},
                6: {"gateway-channel": harness.channel_key.public_key()},
            }
        },
    )
    restarted.service.ack_signer = ChannelSigner("gateway-channel", 6, harness.channel_key)
    assert restarted.state.active_reference is not None
    assert restarted.state.active_reference.generation == expected_generation
    if expected_generation == 2:
        ack = asyncio.run(restarted.service.execute(commit))
        assert ack.status.value == "applied"
