"""Provider-account and conservative reservation invariants."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

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
from llmmaxxing.core.reasons import Modality, QuotaDimensionStatus, TerminalOutcome
from llmmaxxing.core.state_machines import AccountState
from llmmaxxing.gateway.journal import AttemptJournal
from llmmaxxing.gateway.runtime_state import (
    AccountBindingConflict,
    AttemptResolution,
    CircuitCause,
    CircuitState,
    CircuitValue,
    InvalidLeaseTransition,
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

    def advance(self, milliseconds: int) -> None:
        self.value += milliseconds


def limit(status: str = "known", value: int | None = 100) -> QuotaDimension:
    return QuotaDimension(status=status, value=value)


def account(
    *,
    account_id: AccountId | None = None,
    parallel: int = 2,
    local_parallel_ceiling: int = 128,
    rpm: int = 60,
    rpm_window_seconds: int = 60,
    tpm: int = 10_000,
    tpm_window_seconds: int = 60,
    monthly: int = 100_000,
    quota_units_per_attempt: int = 1,
    monthly_reset_at_ms: int = 2_000_000_000_000,
    state: AccountState = AccountState.ACTIVE,
    connection: str = "litellm:nan",
    provider_token: str = "nan-builders",
    binding_ref: str = "nan-primary",
    credential_epoch: int = 1,
) -> ProviderAccount:
    return ProviderAccount(
        account_id=account_id or AccountId.new(),
        display_name="NaN primary",
        connection=connection,
        provider_token=provider_token,
        binding_ref=binding_ref,
        credential_fingerprint="hcf1_" + "a" * 64,
        credential_epoch=credential_epoch,
        parallel_limit=limit(value=parallel),
        local_parallel_ceiling=local_parallel_ceiling,
        rpm_limit=limit(value=rpm),
        rpm_window_seconds=rpm_window_seconds,
        tpm_limit=limit(value=tpm),
        tpm_window_seconds=tpm_window_seconds,
        monthly_quota_units=limit(value=monthly),
        quota_units_per_attempt=quota_units_per_attempt,
        monthly_reset_at_ms=monthly_reset_at_ms,
        state=state,
    )


def profile(*, tokens: int, deadline_ms: int = 60_000) -> RequestProfile:
    return RequestProfile(
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
        deadline_ms=deadline_ms,
    )


def request(
    provider_account: ProviderAccount,
    clock: FakeClock,
    *,
    tokens: int = 10,
    quota_units: int | None = None,
    generation: str = "b",
    attempt_id: AttemptId | None = None,
    circuit: CircuitValue | None = None,
) -> ReservationRequest:
    request_profile = profile(tokens=tokens)
    return ReservationRequest(
        request_id=RequestId.new(),
        attempt_id=attempt_id or AttemptId.new(),
        account_id=provider_account.account_id,
        deployment_generation_id=DeploymentGenerationId.from_digest(generation * 64),
        runtime_identity=RuntimeIdentity(
            installation_id=InstallationId.new(),
            dispatcher_fence=11,
            boot_id=GatewayBootId.new(),
            bundle_generation=7,
            bundle_hash=BundleHash.from_digest("c" * 64),
        ),
        deadline_at_ms=clock.now_ms() + request_profile.deadline_ms,
        profile=request_profile,
        input_tokens_upper_bound=tokens,
        max_output_tokens=request_profile.output_tokens_max,
        max_reasoning_tokens=request_profile.reasoning_tokens_max,
        quota_units=tokens if quota_units is None else quota_units,
        circuit=circuit or CircuitValue.closed(),
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


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(int(datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000))


def open_view(
    tmp_path: Path, provider_account: ProviderAccount, clock: FakeClock
) -> tuple[AttemptJournal, RuntimeState]:
    journal = AttemptJournal.create(
        tmp_path / "journal",
        clock=clock,
        group_commit_delay_ms=0,
    )
    return journal, RuntimeState((provider_account,), journal=journal, clock=clock)


def test_provider_account_declares_every_dimension_and_credential_attestation() -> None:
    measured = account(rpm_window_seconds=17, tpm_window_seconds=43)
    assert measured.parallel_limit.value == 2
    assert measured.rpm_window_seconds == 17
    assert measured.tpm_window_seconds == 43
    assert measured.monthly_reset_at_ms == 2_000_000_000_000
    assert measured.credential_fingerprint.startswith("hcf1_")
    assert measured.enforced_max_in_flight == 2

    unknown = measured.model_copy(
        update={"parallel_limit": limit("unknown", None), "state": AccountState.DRAFT}
    )
    unknown = ProviderAccount.model_validate(unknown.model_dump(mode="python"))
    assert unknown.parallel_limit.status is QuotaDimensionStatus.UNKNOWN
    assert unknown.enforced_max_in_flight == 1

    with pytest.raises(ValidationError, match="parallel"):
        ProviderAccount.model_validate(
            measured.model_copy(
                update={"parallel_limit": limit("attested_absent", None)}
            ).model_dump(mode="python")
        )
    with pytest.raises(ValidationError, match="DRAFT"):
        ProviderAccount.model_validate(
            measured.model_copy(update={"parallel_limit": limit("unknown", None)}).model_dump(
                mode="python"
            )
        )
    with pytest.raises(ValidationError, match="monthly reset"):
        ProviderAccount.model_validate(
            measured.model_copy(update={"monthly_reset_at_ms": None}).model_dump(mode="python")
        )
    with pytest.raises(ValidationError, match="credential attestation"):
        ProviderAccount.model_validate(
            measured.model_copy(update={"credential_fingerprint": None}).model_dump(mode="python")
        )


def test_reservation_request_is_closed_frozen_and_binds_candidate_charge(
    clock: FakeClock,
) -> None:
    provider_account = account()
    reservation = request(provider_account, clock, tokens=31, quota_units=7)
    assert reservation.total_token_upper_bound == 31
    assert reservation.quota_units == 7
    assert reservation.profile_digest.startswith("sha256:")

    with pytest.raises(ValidationError):
        ReservationRequest.model_validate(
            {**reservation.model_dump(mode="python"), "request_body": "forbidden"}
        )
    with pytest.raises(ValidationError, match="max_output_tokens"):
        ReservationRequest.model_validate(
            {**reservation.model_dump(mode="python"), "max_output_tokens": 1}
        )
    with pytest.raises(ValidationError):
        reservation.deadline_at_ms = 1


def test_account_runtime_is_shared_across_deployment_generations_and_publications(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account(parallel=2)
    journal, view = open_view(tmp_path, provider_account, clock)
    try:
        first = view.try_reserve(request(provider_account, clock, generation="1"))
        second = view.try_reserve(request(provider_account, clock, generation="2"))
        denied = view.try_reserve(request(provider_account, clock, generation="3"))
        assert isinstance(first, ReservationGranted)
        assert isinstance(second, ReservationGranted)
        assert denied == ReservationDenied(ReservationDenialReason.PARALLEL_EXHAUSTED)

        enlarged = account(account_id=provider_account.account_id, parallel=3, credential_epoch=2)
        view.apply_publication((enlarged,))
        third = view.try_reserve(request(enlarged, clock, generation="3"))
        assert isinstance(third, ReservationGranted)
        assert view.account_capacity(provider_account.account_id).active_attempts == 3

        with pytest.raises(AccountBindingConflict):
            view.apply_publication(
                (
                    account(
                        account_id=AccountId.new(),
                        connection=provider_account.connection,
                        provider_token=provider_account.provider_token,
                        binding_ref=provider_account.binding_ref,
                    ),
                )
            )
    finally:
        journal.close()


def test_rpm_and_tpm_have_independent_rolling_windows(tmp_path: Path, clock: FakeClock) -> None:
    provider_account = account(
        parallel=5,
        rpm=2,
        rpm_window_seconds=10,
        tpm=100,
        tpm_window_seconds=60,
    )
    journal, view = open_view(tmp_path, provider_account, clock)
    try:
        for _ in range(2):
            granted = view.try_reserve(request(provider_account, clock, tokens=40))
            assert isinstance(granted, ReservationGranted)
            finish_actual(granted.lease, tokens=40, quota_units=40)

        assert view.try_reserve(request(provider_account, clock, tokens=1)) == ReservationDenied(
            ReservationDenialReason.RPM_EXHAUSTED,
            retry_at_ms=clock.now_ms() + 10_000,
        )

        clock.advance(11_000)
        assert view.try_reserve(request(provider_account, clock, tokens=30)) == ReservationDenied(
            ReservationDenialReason.TPM_EXHAUSTED,
            retry_at_ms=clock.now_ms() + 49_000,
        )

        clock.advance(50_000)
        assert isinstance(
            view.try_reserve(request(provider_account, clock, tokens=30)), ReservationGranted
        )
    finally:
        journal.close()


def test_monthly_reset_is_explicit_and_deterministic(tmp_path: Path, clock: FakeClock) -> None:
    reset_at = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp() * 1000)
    provider_account = account(parallel=3, monthly=50, monthly_reset_at_ms=reset_at)
    journal, view = open_view(tmp_path, provider_account, clock)
    try:
        granted = view.try_reserve(request(provider_account, clock, tokens=1, quota_units=30))
        assert isinstance(granted, ReservationGranted)
        finish_actual(granted.lease, tokens=1, quota_units=30)
        assert view.try_reserve(
            request(provider_account, clock, tokens=1, quota_units=30)
        ) == ReservationDenied(
            ReservationDenialReason.MONTHLY_QUOTA_EXHAUSTED,
            retry_at_ms=reset_at,
        )

        clock.value = reset_at
        assert view.try_reserve(
            request(provider_account, clock, tokens=1, quota_units=30)
        ) == ReservationDenied(ReservationDenialReason.MONTHLY_RESET_UNAVAILABLE)
        next_epoch = provider_account.model_copy(
            update={"monthly_reset_at_ms": reset_at + 2_592_000_000}
        )
        view.apply_publication((next_epoch,))
        assert isinstance(
            view.try_reserve(request(next_epoch, clock, tokens=1, quota_units=30)),
            ReservationGranted,
        )
    finally:
        journal.close()


def test_actual_usage_reconciles_but_ambiguity_retains_upper_bound(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account(parallel=1, tpm=500, monthly=500)
    journal, view = open_view(tmp_path, provider_account, clock)
    try:
        first = view.try_reserve(request(provider_account, clock, tokens=100, quota_units=100))
        assert isinstance(first, ReservationGranted)
        finish_actual(first.lease, tokens=30, quota_units=20)
        capacity = view.account_capacity(provider_account.account_id)
        assert (capacity.tpm_tokens, capacity.monthly_quota_units, capacity.active_attempts) == (
            30,
            20,
            0,
        )

        second = view.try_reserve(request(provider_account, clock, tokens=50, quota_units=50))
        assert isinstance(second, ReservationGranted)
        second.lease.finish(
            AttemptResolution(
                outcome=TerminalOutcome.CLIENT_CANCELLED,
                release_capacity=False,
                actual_starts=None,
                actual_token_units=None,
                actual_quota_units=None,
            )
        )
        capacity = view.account_capacity(provider_account.account_id)
        assert (capacity.tpm_tokens, capacity.monthly_quota_units) == (80, 70)
        assert (capacity.active_attempts, capacity.uncertain_attempts) == (1, 1)
        assert view.try_reserve(request(provider_account, clock)) == ReservationDenied(
            ReservationDenialReason.PARALLEL_EXHAUSTED
        )

        view.account_runtime(provider_account.account_id).apply_authoritative_active_count(0)
        assert view.account_capacity(provider_account.account_id).active_attempts == 0
    finally:
        journal.close()


def test_typed_resolution_is_dimension_specific_idempotent_and_fail_closed(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account(parallel=2, rpm=1, tpm=500, monthly=500)
    journal, view = open_view(tmp_path, provider_account, clock)
    try:
        granted = asyncio.run(
            view.try_reserve_async(request(provider_account, clock, tokens=100, quota_units=80))
        )
        assert isinstance(granted, ReservationGranted)
        resolution = AttemptResolution(
            outcome=TerminalOutcome.COMPLETED,
            release_capacity=True,
            actual_starts=0,
            actual_token_units=30,
            actual_quota_units=None,
        )
        first_receipt = asyncio.run(granted.lease.finish_async(resolution))
        assert granted.lease.finish(resolution) == first_receipt
        capacity = view.account_capacity(provider_account.account_id)
        assert (capacity.active_attempts, capacity.rpm_starts) == (0, 0)
        assert (capacity.tpm_tokens, capacity.monthly_quota_units) == (30, 80)

        with pytest.raises(InvalidLeaseTransition, match="conflicting"):
            granted.lease.finish(resolution.model_copy(update={"actual_quota_units": 20}))

        second = view.try_reserve(request(provider_account, clock, tokens=40, quota_units=20))
        assert isinstance(second, ReservationGranted)
        with pytest.raises(InvalidLeaseTransition, match="exceeds"):
            second.lease.finish(
                AttemptResolution(
                    outcome=TerminalOutcome.COMPLETED,
                    release_capacity=True,
                    actual_starts=1,
                    actual_token_units=41,
                    actual_quota_units=20,
                )
            )
        assert view.account_capacity(provider_account.account_id).active_attempts == 1
    finally:
        journal.close()


def test_unknown_capacity_is_never_treated_as_unlimited(tmp_path: Path, clock: FakeClock) -> None:
    measured = account()
    draft = ProviderAccount.model_validate(
        measured.model_copy(
            update={"parallel_limit": limit("unknown", None), "state": AccountState.DRAFT}
        ).model_dump(mode="python")
    )
    journal, view = open_view(tmp_path, draft, clock)
    try:
        assert view.try_reserve(request(draft, clock)) == ReservationDenied(
            ReservationDenialReason.ACCOUNT_NOT_ACTIVE
        )
    finally:
        journal.close()


def test_operational_view_is_an_immutable_advisory_snapshot(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account()
    journal, runtime_state = open_view(tmp_path, provider_account, clock)
    try:
        reservation = request(provider_account, clock, tokens=7)
        granted = runtime_state.try_reserve(reservation)
        assert isinstance(granted, ReservationGranted)
        snapshot = runtime_state.operational_view()
        assert type(snapshot).__name__ == "OperationalRuntimeView"
        assert snapshot.durable_lsn == journal.health.last_sequence
        assert snapshot.accounts[0].account_id == provider_account.account_id
        assert snapshot.attempts[0].attempt_id == reservation.attempt_id
        assert snapshot.attempts[0].installation_id == (
            reservation.runtime_identity.installation_id
        )
        assert snapshot.time_high_water_ms == clock.now_ms()
        assert not hasattr(snapshot, "try_reserve")
        assert not hasattr(snapshot, "authorize")
        with pytest.raises(FrozenInstanceError):
            snapshot.durable_lsn = 0
    finally:
        journal.close()


def test_circuit_cas_is_operational_state_not_publication_authority(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account()
    journal, view = open_view(tmp_path, provider_account, clock)
    generation = DeploymentGenerationId.from_digest("d" * 64)
    closed = CircuitValue.closed()
    opened = CircuitValue(
        state=CircuitState.OPEN,
        cause=CircuitCause.CAPACITY,
        epoch=1,
        opened_at_ms=clock.now_ms(),
        retry_at_ms=clock.now_ms() + 5_000,
        backoff_step=1,
        evidence_digest="sha256:" + "e" * 64,
        probe_id=None,
    )
    try:
        runtime = view.account_runtime(provider_account.account_id)
        assert runtime.compare_and_swap_circuit(generation, closed, opened)
        assert not runtime.compare_and_swap_circuit(generation, closed, opened)
        assert runtime.compare_and_swap_account_circuit(closed, opened)
        view.apply_publication((provider_account.model_copy(update={"display_name": "renamed"}),))
        assert runtime.circuit_value(generation) == opened
        assert not hasattr(view, "authorize")
        assert runtime.account_circuit_value() == opened
    finally:
        journal.close()


def test_local_parallel_ceiling_is_independent_of_provider_limit(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account(parallel=5, local_parallel_ceiling=2)
    journal, state = open_view(tmp_path, provider_account, clock)
    try:
        assert isinstance(state.try_reserve(request(provider_account, clock)), ReservationGranted)
        assert isinstance(state.try_reserve(request(provider_account, clock)), ReservationGranted)
        assert state.try_reserve(request(provider_account, clock)) == ReservationDenied(
            ReservationDenialReason.PARALLEL_EXHAUSTED
        )
    finally:
        journal.close()


def test_oversized_tpm_and_undercharged_quota_are_typed_denials(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account(tpm=10, quota_units_per_attempt=3)
    journal, state = open_view(tmp_path, provider_account, clock)
    try:
        oversized = state.try_reserve(request(provider_account, clock, tokens=11, quota_units=3))
        assert oversized == ReservationDenied(ReservationDenialReason.TPM_EXHAUSTED)
        undercharged = state.try_reserve(request(provider_account, clock, tokens=1, quota_units=2))
        assert undercharged == ReservationDenied(ReservationDenialReason.INVALID_QUOTA_CHARGE)
    finally:
        journal.close()


def test_publication_is_atomic_and_omissions_become_non_serving(
    tmp_path: Path, clock: FakeClock
) -> None:
    first = account(parallel=2)
    second = account(
        connection="litellm:arli",
        provider_token="arli",
        binding_ref="arli-primary",
    )
    journal = AttemptJournal.create(tmp_path / "journal", clock=clock, group_commit_delay_ms=0)
    state = RuntimeState((first, second), journal=journal, clock=clock)
    try:
        enlarged = first.model_copy(
            update={"parallel_limit": limit(value=5), "local_parallel_ceiling": 5}
        )
        conflicting = second.model_copy(
            update={
                "connection": first.connection,
                "provider_token": first.provider_token,
                "binding_ref": first.binding_ref,
            }
        )
        with pytest.raises(AccountBindingConflict):
            state.apply_publication((enlarged, conflicting))
        assert state.account_capacity(first.account_id).parallel_limit == 2

        state.apply_publication((first,))
        assert state.try_reserve(request(second, clock)) == ReservationDenied(
            ReservationDenialReason.ACCOUNT_NOT_ACTIVE
        )
    finally:
        journal.close()


def test_window_increase_retains_previously_expired_start(tmp_path: Path, clock: FakeClock) -> None:
    provider_account = account(tpm=10, tpm_window_seconds=10)
    journal, state = open_view(tmp_path, provider_account, clock)
    try:
        granted = state.try_reserve(request(provider_account, clock, tokens=8))
        assert isinstance(granted, ReservationGranted)
        finish_actual(granted.lease, tokens=8, quota_units=8)
        clock.advance(20_000)
        assert isinstance(
            state.try_reserve(request(provider_account, clock, tokens=3)),
            ReservationGranted,
        )

        enlarged = provider_account.model_copy(update={"tpm_window_seconds": 60})
        state.apply_publication((enlarged,))
        assert state.try_reserve(request(enlarged, clock, tokens=3)) == ReservationDenied(
            ReservationDenialReason.TPM_EXHAUSTED,
            retry_at_ms=clock.now_ms() + 40_000,
        )
    finally:
        journal.close()


def test_binding_digest_is_unambiguous_for_embedded_nuls(tmp_path: Path, clock: FakeClock) -> None:
    first = account(connection="a\0b", provider_token="c", binding_ref="d")
    second = account(connection="a", provider_token="b\0c", binding_ref="d")
    journal = AttemptJournal.create(tmp_path / "journal", clock=clock, group_commit_delay_ms=0)
    try:
        RuntimeState((first, second), journal=journal, clock=clock)
    finally:
        journal.close()


def test_authoritative_active_count_keeps_oldest_attempt(tmp_path: Path, clock: FakeClock) -> None:
    provider_account = account(parallel=2)
    journal, state = open_view(tmp_path, provider_account, clock)
    older_id = AttemptId("att_ffffffff-ffff-4fff-8fff-ffffffffffff")
    newer_id = AttemptId("att_00000000-0000-4000-8000-000000000000")
    try:
        older = state.try_reserve(request(provider_account, clock, attempt_id=older_id))
        assert isinstance(older, ReservationGranted)
        older.lease.finish(
            AttemptResolution(
                outcome=TerminalOutcome.UPSTREAM_FAILED,
                release_capacity=False,
                actual_starts=None,
                actual_token_units=None,
                actual_quota_units=None,
            )
        )
        clock.advance(1)
        newer = state.try_reserve(request(provider_account, clock, attempt_id=newer_id))
        assert isinstance(newer, ReservationGranted)
        newer.lease.finish(
            AttemptResolution(
                outcome=TerminalOutcome.UPSTREAM_FAILED,
                release_capacity=False,
                actual_starts=None,
                actual_token_units=None,
                actual_quota_units=None,
            )
        )
        state.account_runtime(provider_account.account_id).apply_authoritative_active_count(1)
        assert state.operational_view().attempts[0].attempt_id == older_id
    finally:
        journal.close()


def test_monthly_refund_never_changes_a_new_reset_epoch(tmp_path: Path, clock: FakeClock) -> None:
    reset_at = clock.now_ms() + 1_000
    provider_account = account(monthly=100, monthly_reset_at_ms=reset_at)
    journal, state = open_view(tmp_path, provider_account, clock)
    try:
        old = state.try_reserve(request(provider_account, clock, tokens=1, quota_units=50))
        assert isinstance(old, ReservationGranted)
        clock.advance(1_000)
        next_epoch = provider_account.model_copy(
            update={"monthly_reset_at_ms": reset_at + 2_592_000_000}
        )
        state.apply_publication((next_epoch,))
        new = state.try_reserve(request(next_epoch, clock, tokens=1, quota_units=40))
        assert isinstance(new, ReservationGranted)
        old.lease.finish(
            AttemptResolution(
                outcome=TerminalOutcome.COMPLETED,
                release_capacity=True,
                actual_starts=1,
                actual_token_units=1,
                actual_quota_units=20,
            )
        )
        assert state.account_capacity(provider_account.account_id).monthly_quota_units == 40
    finally:
        journal.close()


def test_half_open_probe_token_is_consumed_once(tmp_path: Path, clock: FakeClock) -> None:
    provider_account = account(parallel=3)
    journal, state = open_view(tmp_path, provider_account, clock)
    runtime = state.account_runtime(provider_account.account_id)
    probe_id = f"probe_{uuid4()}"
    half_open = CircuitValue(
        state=CircuitState.HALF_OPEN,
        cause=CircuitCause.CAPACITY,
        epoch=1,
        opened_at_ms=clock.now_ms(),
        retry_at_ms=0,
        backoff_step=1,
        evidence_digest="sha256:" + "a" * 64,
        probe_id=probe_id,
    )
    try:
        assert runtime.compare_and_swap_account_circuit(CircuitValue.closed(), half_open)
        first = state.try_reserve(
            request(provider_account, clock).model_copy(update={"account_circuit": half_open})
        )
        assert isinstance(first, ReservationGranted)
        second = state.try_reserve(
            request(provider_account, clock).model_copy(update={"account_circuit": half_open})
        )
        assert second == ReservationDenied(ReservationDenialReason.CIRCUIT_UNAVAILABLE)
        opened = CircuitValue(
            state=CircuitState.OPEN,
            cause=CircuitCause.CAPACITY,
            epoch=2,
            opened_at_ms=clock.now_ms(),
            retry_at_ms=clock.now_ms() + 1_000,
            backoff_step=2,
            evidence_digest="sha256:" + "b" * 64,
            probe_id=None,
        )
        assert runtime.compare_and_swap_account_circuit(half_open, opened)
        assert state.account_capacity(provider_account.account_id).consumed_probe_tokens == 0
    finally:
        journal.close()


def test_terminal_ledger_stops_before_snapshot_bound(tmp_path: Path, clock: FakeClock) -> None:
    provider_account = account(parallel=2)
    journal = AttemptJournal.create(tmp_path / "journal", clock=clock, group_commit_delay_ms=0)
    state = RuntimeState(
        (provider_account,),
        journal=journal,
        clock=clock,
        resolution_ledger_limit=1,
        resolution_retention_ms=10,
    )
    try:
        granted = state.try_reserve(request(provider_account, clock))
        assert isinstance(granted, ReservationGranted)
        finish_actual(granted.lease, tokens=10, quota_units=10)
        assert state.try_reserve(request(provider_account, clock)) == ReservationDenied(
            ReservationDenialReason.JOURNAL_CAPACITY_STOP
        )
        clock.advance(11)
        assert isinstance(
            state.try_reserve(request(provider_account, clock)), ReservationGranted
        )
    finally:
        journal.close()

def test_authoritative_count_rejects_while_live_attempt_exists(
    tmp_path: Path, clock: FakeClock
) -> None:
    provider_account = account(parallel=2)
    journal, state = open_view(tmp_path, provider_account, clock)
    try:
        uncertain = state.try_reserve(request(provider_account, clock))
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
        live = state.try_reserve(request(provider_account, clock))
        assert isinstance(live, ReservationGranted)
        with pytest.raises(ValueError, match="live active"):
            state.account_runtime(provider_account.account_id).apply_authoritative_active_count(1)
        finish_actual(live.lease, tokens=10, quota_units=10)
    finally:
        journal.close()
