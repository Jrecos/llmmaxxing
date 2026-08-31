"""Durable reservation, segmented journal, checkpoint, and recovery contract."""

from __future__ import annotations
import asyncio
import threading

from pathlib import Path
from uuid import uuid4

import pytest

from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    BundleHash,
    DeploymentGenerationId,
    GatewayBootId,
    InstallationId,
    RequestId,
    RouteGroupId,
)
from llmmaxxing.core.models import ProviderAccount, QuotaDimension, RequestProfile
from llmmaxxing.core.reasons import Modality, TerminalOutcome
from llmmaxxing.core.state_machines import AccountState
from llmmaxxing.gateway.journal import (
    AttemptJournal,
    DurableReservation,
    InjectedCrash,
    JournalStatus,
    JournalUnavailable,
)
from llmmaxxing.gateway.runtime_state import (
    AccountBindingConflict,
    AttemptResolution,
    CircuitCause,
    CircuitState,
    CircuitValue,
    InvalidLeaseTransition,
    ProbeClassification,
    ReservationDenialReason,
    ReservationDenied,
    ReservationGranted,
    ReservationRequest,
    RuntimeIdentity,
    RuntimeState,
)


class FakeClock:
    def __init__(self, now_ms: int) -> None:
        self.value = now_ms

    def now_ms(self) -> int:
        return self.value


def limit(value: int) -> QuotaDimension:
    return QuotaDimension(status="known", value=value)


def account(
    *,
    account_id: AccountId | None = None,
    connection: str = "litellm:nan",
    provider_token: str = "nan-builders",
    binding_ref: str = "nan-primary",
) -> ProviderAccount:
    return ProviderAccount(
        account_id=account_id or AccountId.new(),
        display_name="NaN primary",
        connection=connection,
        provider_token=provider_token,
        binding_ref=binding_ref,
        credential_fingerprint="hcf1_" + "a" * 64,
        credential_epoch=1,
        parallel_limit=limit(2),
        local_parallel_ceiling=128,
        rpm_limit=limit(60),
        rpm_window_seconds=60,
        tpm_limit=limit(10_000),
        tpm_window_seconds=60,
        monthly_quota_units=limit(100_000),
        quota_units_per_attempt=1,
        monthly_reset_at_ms=2_000_000_000_000,
        state=AccountState.ACTIVE,
    )


def request(
    provider_account: ProviderAccount,
    clock: FakeClock,
    *,
    tokens: int = 10,
    quota_units: int | None = None,
) -> ReservationRequest:
    request_profile = RequestProfile(
        route_group_id=RouteGroupId.new(),
        model_alias="deepseek-v4-flash",
        modality=Modality.CHAT,
        stream=False,
        input_tokens_max=tokens,
        output_tokens_max=0,
        reasoning_tokens_max=0,
        tools_count=0,
        response_schema_present=False,
        history_turns=0,
        deadline_ms=60_000,
    )
    return ReservationRequest(
        request_id=RequestId.new(),
        attempt_id=AttemptId.new(),
        account_id=provider_account.account_id,
        deployment_generation_id=DeploymentGenerationId.from_digest("b" * 64),
        runtime_identity=RuntimeIdentity(
            installation_id=InstallationId.new(),
            dispatcher_fence=11,
            boot_id=GatewayBootId.new(),
            bundle_generation=7,
            bundle_hash=BundleHash.from_digest("c" * 64),
        ),
        deadline_at_ms=clock.now_ms() + 60_000,
        profile=request_profile,
        input_tokens_upper_bound=tokens,
        max_output_tokens=0,
        max_reasoning_tokens=0,
        quota_units=tokens if quota_units is None else quota_units,
        circuit=CircuitValue.closed(),
    )


def finish_actual(lease: object, *, tokens: int, quota_units: int) -> None:
    assert hasattr(lease, "finish")
    lease.finish(
        AttemptResolution(
            outcome=TerminalOutcome.COMPLETED,
            release_capacity=True,
            actual_starts=1,
            actual_token_units=tokens,
            actual_quota_units=quota_units,
        )
    )


class CrashController:
    def __init__(self) -> None:
        self.boundary: str | None = None
        self.hit = False

    def arm(self, boundary: str) -> None:
        self.boundary = boundary
        self.hit = False

    def __call__(self, boundary: str) -> None:
        if boundary == self.boundary and not self.hit:
            self.hit = True
            raise InjectedCrash(boundary)


def create_runtime(
    root: Path,
    clock: FakeClock,
    crash: CrashController | None = None,
    **journal_options: object,
) -> tuple[AttemptJournal, RuntimeState, ProviderAccount]:
    provider_account = account()
    journal = AttemptJournal.create(
        root,
        clock=clock,
        crash_injector=crash,
        group_commit_delay_ms=0,
        **journal_options,
    )
    view = RuntimeState((provider_account,), journal=journal, clock=clock)
    return journal, view, provider_account


def reopen_runtime(
    root: Path,
    provider_account: ProviderAccount,
    clock: FakeClock,
    **journal_options: object,
) -> tuple[AttemptJournal, RuntimeState]:
    journal = AttemptJournal.open(
        root,
        clock=clock,
        group_commit_delay_ms=0,
        **journal_options,
    )
    return journal, RuntimeState((provider_account,), journal=journal, clock=clock)


@pytest.mark.parametrize(
    ("boundary", "expected_active"),
    (
        ("before_reservation_fsync", 0),
        ("after_reservation_fsync", 1),
    ),
)
def test_crash_around_reservation_fsync_never_overgrants(
    tmp_path: Path, boundary: str, expected_active: int
) -> None:
    clock = FakeClock(1_800_000_000_000)
    crash = CrashController()
    journal, view, provider_account = create_runtime(tmp_path / boundary, clock, crash)
    crash.arm(boundary)
    with pytest.raises(InjectedCrash, match=boundary):
        view.try_reserve(request(provider_account, clock))  # type: ignore[arg-type]
    journal.close()

    reopened, recovered = reopen_runtime(tmp_path / boundary, provider_account, clock)
    try:
        capacity = recovered.account_capacity(provider_account.account_id)  # type: ignore[attr-defined]
        assert capacity.active_attempts == expected_active
        assert capacity.uncertain_attempts == expected_active
    finally:
        reopened.close()


def test_crash_after_provider_send_retains_the_full_uncertain_reservation(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    crash = CrashController()
    journal, view, provider_account = create_runtime(tmp_path / "send", clock, crash)
    granted = view.try_reserve(request(provider_account, clock, tokens=73, quota_units=19))  # type: ignore[arg-type]
    assert isinstance(granted, ReservationGranted)
    crash.arm("after_provider_send_before_terminal")
    with pytest.raises(InjectedCrash, match="after_provider_send_before_terminal"):
        granted.lease.provider_send_completed()
    journal.close()

    reopened, recovered = reopen_runtime(tmp_path / "send", provider_account, clock)
    try:
        capacity = recovered.account_capacity(provider_account.account_id)  # type: ignore[attr-defined]
        assert (capacity.active_attempts, capacity.uncertain_attempts) == (1, 1)
        assert (capacity.tpm_tokens, capacity.monthly_quota_units) == (73, 19)
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("boundary", "expected_active", "expected_tokens"),
    (
        ("before_terminal_update_fsync", 1, 80),
        ("after_terminal_update_fsync", 0, 15),
    ),
)
def test_terminal_update_is_replayed_once_across_crash(
    tmp_path: Path, boundary: str, expected_active: int, expected_tokens: int
) -> None:
    clock = FakeClock(1_800_000_000_000)
    crash = CrashController()
    journal, view, provider_account = create_runtime(tmp_path / boundary, clock, crash)
    granted = view.try_reserve(request(provider_account, clock, tokens=80, quota_units=50))  # type: ignore[arg-type]
    assert isinstance(granted, ReservationGranted)
    crash.arm(boundary)
    with pytest.raises(InjectedCrash, match=boundary):
        finish_actual(granted.lease, tokens=15, quota_units=9)

    reopened, recovered = reopen_runtime(tmp_path / boundary, provider_account, clock)
    try:
        capacity = recovered.account_capacity(provider_account.account_id)  # type: ignore[attr-defined]
        assert capacity.active_attempts == expected_active
        assert capacity.uncertain_attempts == expected_active
        assert capacity.tpm_tokens == expected_tokens
        assert capacity.monthly_quota_units == (50 if expected_active else 9)
        if expected_active:
            recovered.account_runtime(  # type: ignore[attr-defined]
                provider_account.account_id  # type: ignore[attr-defined]
            ).apply_authoritative_active_count(0)
            recovered.account_runtime(  # type: ignore[attr-defined]
                provider_account.account_id  # type: ignore[attr-defined]
            ).apply_authoritative_active_count(0)
            assert (
                recovered.account_capacity(  # type: ignore[attr-defined]
                    provider_account.account_id  # type: ignore[attr-defined]
                ).active_attempts
                == 0
            )
    finally:
        reopened.close()


@pytest.mark.parametrize("boundary", ("checkpoint_before_rename", "checkpoint_after_rename"))
def test_checkpoint_rename_crash_recovers_from_checkpoint_or_segments(
    tmp_path: Path, boundary: str
) -> None:
    clock = FakeClock(1_800_000_000_000)
    crash = CrashController()
    root = tmp_path / boundary
    journal, view, provider_account = create_runtime(root, clock, crash)
    view.checkpoint()
    granted = view.try_reserve(request(provider_account, clock, tokens=41))  # type: ignore[arg-type]
    assert isinstance(granted, ReservationGranted)
    crash.arm(boundary)
    with pytest.raises(InjectedCrash, match=boundary):
        view.checkpoint()
    journal.close()

    reopened, recovered = reopen_runtime(root, provider_account, clock)
    try:
        assert reopened.status is JournalStatus.HEALTHY
        capacity = recovered.account_capacity(provider_account.account_id)  # type: ignore[attr-defined]
        assert (capacity.active_attempts, capacity.uncertain_attempts, capacity.tpm_tokens) == (
            1,
            1,
            41,
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("boundary", ("segment_before_delete", "segment_after_delete"))
def test_segment_delete_crash_is_safe_after_two_verified_checkpoints(
    tmp_path: Path, boundary: str
) -> None:
    clock = FakeClock(1_800_000_000_000)
    crash = CrashController()
    root = tmp_path / boundary
    options = {"segment_bytes": 256, "checkpoint_every_records": 100}
    journal, view, provider_account = create_runtime(root, clock, crash, **options)

    first = view.try_reserve(request(provider_account, clock, tokens=11))  # type: ignore[arg-type]
    assert isinstance(first, ReservationGranted)

    finish_actual(first.lease, tokens=5, quota_units=5)
    view.checkpoint()

    second = view.try_reserve(request(provider_account, clock, tokens=13))  # type: ignore[arg-type]
    assert isinstance(second, ReservationGranted)
    finish_actual(second.lease, tokens=7, quota_units=7)
    crash.arm(boundary)
    with pytest.raises(InjectedCrash, match=boundary):
        view.checkpoint()
    journal.close()

    reopened, recovered = reopen_runtime(root, provider_account, clock, **options)
    try:
        assert reopened.status is JournalStatus.HEALTHY
        capacity = recovered.account_capacity(provider_account.account_id)  # type: ignore[attr-defined]
        assert (capacity.active_attempts, capacity.tpm_tokens) == (0, 12)
        assert reopened.health.verified_checkpoints >= 1
    finally:
        reopened.close()


def test_credential_attestation_highwater_survives_restart(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "credential-highwater"
    journal, view, provider_account = create_runtime(root, clock)
    rotated = ProviderAccount.model_validate(
        provider_account.model_copy(
            update={
                "credential_fingerprint": "hcf1_" + "b" * 64,
                "credential_epoch": 2,
            }
        ).model_dump(mode="python")
    )
    view.apply_publication((rotated,))
    journal.close()

    reopened = AttemptJournal.open(root, clock=clock, group_commit_delay_ms=0)
    RuntimeState((provider_account,), journal=reopened, clock=clock)
    try:
        assert reopened.status is JournalStatus.RECOVERY_REQUIRED
        assert reopened.health.reason == "invalid_runtime_state"
    finally:
        reopened.close()


def test_terminal_resolution_idempotency_survives_checkpoint_compaction(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "terminal-idempotency"
    options = {"segment_bytes": 256, "checkpoint_every_records": 100}
    journal, view, provider_account = create_runtime(root, clock, **options)
    granted = view.try_reserve(request(provider_account, clock, tokens=23, quota_units=17))
    assert isinstance(granted, ReservationGranted)
    resolution = AttemptResolution(
        outcome=TerminalOutcome.COMPLETED,
        release_capacity=True,
        actual_starts=1,
        actual_token_units=11,
        actual_quota_units=7,
    )
    receipt = granted.lease.finish(resolution)
    view.checkpoint()
    second = view.try_reserve(request(provider_account, clock, tokens=3, quota_units=3))
    assert isinstance(second, ReservationGranted)
    second.lease.finish(
        AttemptResolution(
            outcome=TerminalOutcome.COMPLETED,
            release_capacity=True,
            actual_starts=1,
            actual_token_units=3,
            actual_quota_units=3,
        )
    )
    view.checkpoint()
    journal.close()

    reopened, recovered = reopen_runtime(root, provider_account, clock, **options)
    try:
        runtime = recovered.account_runtime(provider_account.account_id)
        assert runtime.finish_attempt(granted.lease.request.attempt_id, resolution) == receipt
        with pytest.raises(InvalidLeaseTransition, match="conflicting"):
            runtime.finish_attempt(
                granted.lease.request.attempt_id,
                resolution.model_copy(update={"actual_quota_units": 6}),
            )
        assert recovered.account_capacity(provider_account.account_id).active_attempts == 0
    finally:
        reopened.close()


def test_recovery_restores_windows_quota_uncertainty_and_circuit_cas(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "state"
    journal, view, provider_account = create_runtime(root, clock)
    generation = DeploymentGenerationId.from_digest("e" * 64)

    completed = view.try_reserve(request(provider_account, clock, tokens=40, quota_units=20))  # type: ignore[arg-type]
    assert isinstance(completed, ReservationGranted)
    finish_actual(completed.lease, tokens=20, quota_units=8)
    uncertain = view.try_reserve(request(provider_account, clock, tokens=30, quota_units=15))  # type: ignore[arg-type]
    assert isinstance(uncertain, ReservationGranted)
    uncertain.lease.finish(
        AttemptResolution(
            outcome=TerminalOutcome.UPSTREAM_FAILED,
            release_capacity=False,
            actual_starts=None,
            actual_token_units=None,
            actual_quota_units=None,
        )
    )

    opened = CircuitValue(
        state=CircuitState.OPEN,
        cause=CircuitCause.TRANSIENT_FAILURE,
        opened_at_ms=clock.now_ms(),
        backoff_step=1,
        evidence_digest="sha256:" + "f" * 64,
        epoch=1,
        retry_at_ms=clock.now_ms() + 15_000,
        probe_id=None,
    )
    runtime = view.account_runtime(provider_account.account_id)  # type: ignore[attr-defined]
    assert runtime.compare_and_swap_circuit(generation, CircuitValue.closed(), opened)
    assert runtime.compare_and_swap_account_circuit(CircuitValue.closed(), opened)
    journal.close()

    reopened, recovered = reopen_runtime(root, provider_account, clock)
    try:
        capacity = recovered.account_capacity(provider_account.account_id)  # type: ignore[attr-defined]
        assert (capacity.rpm_starts, capacity.tpm_tokens, capacity.monthly_quota_units) == (
            2,
            50,
            23,
        )
        assert (capacity.active_attempts, capacity.uncertain_attempts) == (1, 1)
        recovered_runtime = recovered.account_runtime(  # type: ignore[attr-defined]
            provider_account.account_id  # type: ignore[attr-defined]
        )
        assert recovered_runtime.circuit_value(generation) == opened

        first_probe = f"probe_{uuid4()}"
        assert recovered_runtime.account_circuit_value() == opened
        assert recovered_runtime.begin_recovery_probe(first_probe)
        assert not recovered_runtime.begin_recovery_probe(f"probe_{uuid4()}")
        recovered_runtime.finish_recovery_probe(first_probe, ProbeClassification.AVAILABLE)
        assert (
            recovered.account_capacity(  # type: ignore[attr-defined]
                provider_account.account_id  # type: ignore[attr-defined]
            ).active_attempts
            == 0
        )
    finally:
        reopened.close()


def test_missing_corrupt_or_slow_recovery_fails_closed(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    missing = AttemptJournal.open(tmp_path / "missing", clock=clock, group_commit_delay_ms=0)
    assert missing.status is JournalStatus.RECOVERY_REQUIRED
    missing.close()

    root = tmp_path / "corrupt"
    journal, view, provider_account = create_runtime(root, clock)
    assert isinstance(
        view.try_reserve(request(provider_account, clock)),  # type: ignore[arg-type]
        ReservationGranted,
    )
    journal.close()
    segment = next(root.glob("segment-*.jsonl"))
    segment.write_bytes(segment.read_bytes() + b"not-json\n")

    corrupt = AttemptJournal.open(root, clock=clock, group_commit_delay_ms=0)
    corrupt_view = RuntimeState((provider_account,), journal=corrupt, clock=clock)
    try:
        assert corrupt.status is JournalStatus.RECOVERY_REQUIRED
        assert corrupt_view.try_reserve(  # type: ignore[arg-type]
            request(provider_account, clock)  # type: ignore[arg-type]
        ) == ReservationDenied(ReservationDenialReason.RECOVERY_REQUIRED)
    finally:
        corrupt.close()

    timed_root = tmp_path / "timed"
    healthy, _, timed_account = create_runtime(timed_root, clock)
    healthy.close()
    calls = 0

    def slow_monotonic() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 31.0

    slow = AttemptJournal.open(
        timed_root,
        clock=clock,
        monotonic=slow_monotonic,
        group_commit_delay_ms=0,
    )
    try:
        assert slow.status is JournalStatus.RECOVERY_REQUIRED
        assert slow.recovery.reason == "recovery_deadline_exceeded"
        assert timed_account is not None
    finally:
        slow.close()


def test_checkpoints_bound_replay_and_keep_binding_tombstones(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "bounded"
    options = {"segment_bytes": 256, "checkpoint_every_records": 2}
    journal, view, provider_account = create_runtime(root, clock, **options)

    for tokens in (3, 4, 5):
        granted = view.try_reserve(request(provider_account, clock, tokens=tokens))  # type: ignore[arg-type]
        assert isinstance(granted, ReservationGranted)
        finish_actual(granted.lease, tokens=tokens, quota_units=tokens)
    tombstone = provider_account.model_copy(update={"state": AccountState.TOMBSTONED})  # type: ignore[attr-defined]
    view.apply_publication((tombstone,))
    view.checkpoint()
    assert journal.health.verified_checkpoints == 2
    journal.close()

    reopened, recovered = reopen_runtime(root, tombstone, clock, **options)
    try:
        assert reopened.recovery.replayed_records <= 2
        with pytest.raises(AccountBindingConflict):
            recovered.apply_publication(
                (
                    account(
                        account_id=AccountId.new(),
                        connection=tombstone.connection,
                        provider_token=tombstone.provider_token,
                        binding_ref=tombstone.binding_ref,
                    ),
                )
            )
    finally:
        reopened.close()


def test_writer_group_and_delay_bounds_are_closed(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    with pytest.raises(ValueError, match="2ms"):
        AttemptJournal.create(tmp_path / "slow-group", clock=clock, group_commit_delay_ms=3)
    with pytest.raises(ValueError, match="256"):
        AttemptJournal.create(tmp_path / "large-group", clock=clock, max_group_records=257)


def test_admission_stops_at_journal_bounds_but_terminal_updates_remain_writable(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "full"
    journal, view, provider_account = create_runtime(root, clock, max_bytes=1)
    try:
        assert journal.health.status is JournalStatus.ADMISSION_STOP
        assert view.try_reserve(  # type: ignore[arg-type]
            request(provider_account, clock)  # type: ignore[arg-type]
        ) == ReservationDenied(ReservationDenialReason.JOURNAL_CAPACITY_STOP)
        assert journal.writer_limits.queue_capacity == 4096
        assert journal.writer_limits.max_group_records == 256
        assert journal.writer_limits.max_group_delay_ms <= 2
    finally:
        journal.close()

    disk_root = tmp_path / "disk"
    disk, _, _ = create_runtime(disk_root, clock, disk_usage=lambda _path: 0.8)
    try:
        assert disk.health.status is JournalStatus.ADMISSION_STOP
    finally:
        disk.close()


def test_journal_and_checkpoints_never_store_provider_or_request_secrets(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    provider_account = account(
        connection="secret-connection-marker",
        provider_token="secret-provider-marker",
        binding_ref="secret-binding-marker",
    )
    root = tmp_path / "redacted"
    journal = AttemptJournal.create(root, clock=clock, group_commit_delay_ms=0)
    view = RuntimeState((provider_account,), journal=journal, clock=clock)
    reservation = request(provider_account, clock, tokens=9)
    reservation = reservation.model_copy(
        update={"profile": reservation.profile.model_copy(update={"model_alias": "prompt-marker"})}
    )
    granted = view.try_reserve(reservation)
    assert isinstance(granted, ReservationGranted)
    granted.lease.finish(
        AttemptResolution(
            outcome=TerminalOutcome.RESPONSE_STREAM_FAILED,
            release_capacity=False,
            actual_starts=None,
            actual_token_units=None,
            actual_quota_units=None,
        )
    )
    view.checkpoint()
    journal.close()

    durable = b"".join(path.read_bytes() for path in root.iterdir() if path.is_file())
    for forbidden in (
        b"secret-connection-marker",
        b"secret-provider-marker",
        b"secret-binding-marker",
        b"prompt-marker",
        b"request_body",
        b"client_secret",
    ):
        assert forbidden not in durable

def test_writer_failure_rejects_every_queued_waiter(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    entered = threading.Event()
    release = threading.Event()

    def fail_sync(boundary: str) -> None:
        if boundary == "before_reservation_fsync":
            entered.set()
            assert release.wait(1)
            raise OSError("synthetic EIO")

    journal = AttemptJournal.create(
        tmp_path / "writer-failure",
        clock=clock,
        crash_injector=fail_sync,
        group_commit_delay_ms=0,
    )

    def durable(attempt: AttemptId) -> DurableReservation:
        return DurableReservation(
            request_id=str(RequestId.new()),
            attempt_id=str(attempt),
            account_id=str(AccountId.new()),
            deployment_generation_id=str(DeploymentGenerationId.from_digest("d" * 64)),
            installation_id=str(InstallationId.new()),
            bundle_generation=1,
            bundle_hash=str(BundleHash.from_digest("e" * 64)),
            fence_token=1,
            boot_id=str(GatewayBootId.new()),
            deadline_at_ms=clock.now_ms() + 1_000,
            profile_digest="sha256:" + "f" * 64,
            started_at_ms=clock.now_ms(),
            reserved_tokens=1,
            quota_units=1,
            monthly_reset_at_ms=2_000_000_000_000,
            circuit_epoch=0,
            circuit_probe_id=None,
            account_circuit_probe_id=None,
        )

    errors: list[BaseException] = []

    def submit(entry: DurableReservation) -> None:
        try:
            journal.reserve_before_send(entry)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=submit, args=(durable(AttemptId.new()),))
    second = threading.Thread(target=submit, args=(durable(AttemptId.new()),))
    first.start()
    assert entered.wait(1)
    second.start()
    release.set()
    first.join(1)
    second.join(1)
    try:
        assert not first.is_alive()
        assert not second.is_alive()
        assert len(errors) == 2
        assert all(isinstance(error, JournalUnavailable) for error in errors)
    finally:
        journal.close()


def test_async_cancel_reconciles_a_durable_no_send(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    entered = threading.Event()
    release = threading.Event()

    def block_sync(boundary: str) -> None:
        if boundary == "before_reservation_fsync":
            entered.set()
            assert release.wait(1)

    provider_account = account()
    journal = AttemptJournal.create(
        tmp_path / "async-cancel",
        clock=clock,
        crash_injector=block_sync,
        group_commit_delay_ms=0,
    )
    state = RuntimeState((provider_account,), journal=journal, clock=clock)

    async def exercise() -> None:
        task = asyncio.create_task(
            state.try_reserve_async(request(provider_account, clock))
        )
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
        capacity = state.account_capacity(provider_account.account_id)
        assert (
            capacity.active_attempts,
            capacity.rpm_starts,
            capacity.tpm_tokens,
            capacity.monthly_quota_units,
        ) == (0, 0, 0, 0)
    finally:
        journal.close()


def test_recovery_seeds_checkpoint_cadence_from_existing_tail(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "cadence"
    journal = AttemptJournal.create(
        root,
        clock=clock,
        group_commit_delay_ms=0,
        checkpoint_every_records=3,
    )
    journal.register_account(
        account_id=str(AccountId.new()),
        binding_digest="sha256:" + "1" * 64,
        credential_attestation_digest="sha256:" + "2" * 64,
        credential_epoch=1,
        state="active",
    )
    journal.force_checkpoint({"marker": "closed"})
    for digit in ("3", "4"):
        journal.register_account(
            account_id=str(AccountId.new()),
            binding_digest="sha256:" + digit * 64,
            credential_attestation_digest="sha256:" + "5" * 64,
            credential_epoch=1,
            state="active",
        )
    journal.close()

    reopened = AttemptJournal.open(
        root,
        clock=clock,
        group_commit_delay_ms=0,
        checkpoint_every_records=3,
    )
    try:
        assert reopened.records_until_checkpoint == 1
    finally:
        reopened.close()


def test_recovery_truncates_only_incomplete_newest_tail(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "tail"
    journal, view, provider_account = create_runtime(root, clock)
    assert isinstance(view.try_reserve(request(provider_account, clock)), ReservationGranted)
    journal.close()
    segment = sorted(root.glob("segment-*.jsonl"))[-1]
    valid_size = segment.stat().st_size
    with segment.open("ab") as stream:
        stream.write(b'{"incomplete"')

    reopened = AttemptJournal.open(root, clock=clock, group_commit_delay_ms=0)
    try:
        assert reopened.status is JournalStatus.HEALTHY
        assert segment.stat().st_size == valid_size
    finally:
        reopened.close()


def test_crash_invalidates_account_half_open_and_recovery_probe(tmp_path: Path) -> None:
    clock = FakeClock(1_800_000_000_000)
    root = tmp_path / "crashed-probes"
    journal, view, provider_account = create_runtime(root, clock)
    granted = view.try_reserve(request(provider_account, clock))
    assert isinstance(granted, ReservationGranted)
    granted.lease.finish(
        AttemptResolution(
            outcome=TerminalOutcome.UPSTREAM_FAILED,
            release_capacity=False,
            actual_starts=None,
            actual_token_units=None,
            actual_quota_units=None,
        )
    )
    runtime = view.account_runtime(provider_account.account_id)
    half_open = CircuitValue(
        state=CircuitState.HALF_OPEN,
        cause=CircuitCause.CAPACITY,
        epoch=2,
        opened_at_ms=clock.now_ms(),
        retry_at_ms=0,
        backoff_step=2,
        evidence_digest="sha256:" + "a" * 64,
        probe_id=f"probe_{uuid4()}",
    )
    assert runtime.compare_and_swap_account_circuit(CircuitValue.closed(), half_open)
    recovery_probe = f"probe_{uuid4()}"
    assert runtime.begin_recovery_probe(recovery_probe)
    journal.close()

    reopened, recovered = reopen_runtime(root, provider_account, clock)
    try:
        recovered_runtime = recovered.account_runtime(provider_account.account_id)
        account_circuit = recovered_runtime.account_circuit_value()
        assert account_circuit.state is CircuitState.OPEN
        assert account_circuit.epoch == 3
        assert recovered_runtime.begin_recovery_probe(f"probe_{uuid4()}")
    finally:
        reopened.close()
