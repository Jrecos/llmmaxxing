"""Shared provider-account authorization, conservative leases, and recovery state."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    BundleHash,
    DeploymentGenerationId,
    GatewayBootId,
    InstallationId,
    RequestId,
    RouteLegId,
)
from llmmaxxing.core.models import ProviderAccount, RequestProfile
from llmmaxxing.core.reasons import QuotaDimensionStatus, TerminalOutcome
from llmmaxxing.core.state_machines import AccountState
from llmmaxxing.gateway.journal import (
    AttemptJournal,
    Clock,
    DurableReservation,
    JournalHealth,
    JournalReceipt,
    JournalRecord,
    JournalStatus,
    JournalUnavailable,
    JsonValue,
)

_PROBE_ID = Annotated[str, Field(pattern=r"^probe_[0-9a-f-]{36}$")]
_MAX_INT = 2**63 - 1
_MAX_ROLLING_WINDOW_SECONDS = 86_400
_DEFAULT_RESOLUTION_LEDGER_LIMIT = 100_000


_DEFAULT_RESOLUTION_RETENTION_MS = 86_400_000


def _require_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("recovered runtime integer is invalid")
    return value


def _require_str(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("recovered runtime string is invalid")
    return value


def _require_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("recovered runtime boolean is invalid")
    return value


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class ReservationDenialReason(StrEnum):
    ACCOUNT_NOT_ACTIVE = "account_not_active"
    ACCOUNT_NOT_FOUND = "account_not_found"
    ATTEMPT_DUPLICATE = "attempt_duplicate"
    CIRCUIT_CHANGED = "circuit_changed"
    CIRCUIT_UNAVAILABLE = "circuit_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INVALID_QUOTA_CHARGE = "invalid_quota_charge"
    JOURNAL_CAPACITY_STOP = "journal_capacity_stop"
    JOURNAL_UNAVAILABLE = "journal_unavailable"
    MONTHLY_QUOTA_EXHAUSTED = "monthly_quota_exhausted"
    MONTHLY_RESET_UNAVAILABLE = "monthly_reset_unavailable"
    PARALLEL_EXHAUSTED = "parallel_exhausted"
    RECOVERY_REQUIRED = "recovery_required"
    RPM_EXHAUSTED = "rpm_exhausted"
    TPM_EXHAUSTED = "tpm_exhausted"


class AmbiguousReason(StrEnum):
    CLIENT_CANCELLED = "client_cancelled"
    CRASH_RECOVERY = "crash_recovery"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RESPONSE_STREAM_FAILED = "response_stream_failed"
    UPSTREAM_UNKNOWN = "upstream_unknown"


class ProbeClassification(StrEnum):
    AVAILABLE = "available"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    INCONCLUSIVE = "inconclusive"


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitCause(StrEnum):
    CAPACITY = "capacity"
    TRANSIENT_FAILURE = "transient_failure"
    QUOTA = "quota"
    MANUAL = "manual"


class CircuitScope(StrEnum):
    ACCOUNT = "account"
    DEPLOYMENT = "deployment"


class CircuitValue(_Frozen):
    """Persistable Task-7-ready circuit CAS value; Task 7 chooses transitions."""

    state: CircuitState
    cause: CircuitCause | None
    epoch: int = Field(ge=0)
    opened_at_ms: int = Field(ge=0, le=_MAX_INT)
    retry_at_ms: int = Field(ge=0, le=_MAX_INT)
    backoff_step: int = Field(ge=0, le=64)
    evidence_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    probe_id: _PROBE_ID | None

    @classmethod
    def closed(cls, epoch: int = 0) -> CircuitValue:
        return cls(
            state=CircuitState.CLOSED,
            cause=None,
            epoch=epoch,
            opened_at_ms=0,
            retry_at_ms=0,
            backoff_step=0,
            evidence_digest=None,
            probe_id=None,
        )

    @model_validator(mode="after")
    def _state_fields_match(self) -> Self:
        evidence = self.cause is not None and self.evidence_digest is not None
        if self.state is CircuitState.CLOSED and (
            evidence
            or self.opened_at_ms != 0
            or self.retry_at_ms != 0
            or self.backoff_step != 0
            or self.probe_id
        ):
            raise ValueError("closed circuit cannot carry open/probe state")
        if self.state is CircuitState.OPEN and (
            not evidence or self.opened_at_ms == 0 or self.retry_at_ms == 0 or self.probe_id
        ):
            raise ValueError("open circuit requires evidence/retry state and no probe")
        if self.state is CircuitState.HALF_OPEN and (
            not evidence or self.opened_at_ms == 0 or self.retry_at_ms != 0 or self.probe_id is None
        ):
            raise ValueError("half-open circuit requires evidence and one probe")
        return self


class RuntimeIdentity(_Frozen):
    """Dispatcher identity fenced into every durable provider attempt."""

    installation_id: InstallationId
    dispatcher_fence: int = Field(ge=1, le=_MAX_INT)
    boot_id: GatewayBootId
    bundle_generation: int = Field(ge=1)
    bundle_hash: BundleHash


class ReservationRequest(_Frozen):
    """Complete immutable authorization input for one candidate provider send."""

    request_id: RequestId
    attempt_id: AttemptId
    account_id: AccountId
    leg_id: RouteLegId
    deployment_generation_id: DeploymentGenerationId
    runtime_identity: RuntimeIdentity
    deadline_at_ms: int = Field(ge=1, le=_MAX_INT)
    profile: RequestProfile
    input_tokens_upper_bound: int = Field(ge=0, le=_MAX_INT)
    account_circuit: CircuitValue = Field(default_factory=CircuitValue.closed)
    max_output_tokens: int = Field(ge=0, le=_MAX_INT)
    max_reasoning_tokens: int = Field(ge=0, le=_MAX_INT)
    quota_units: int = Field(ge=0, le=_MAX_INT)
    circuit: CircuitValue

    @property
    def total_token_upper_bound(self) -> int:
        return self.input_tokens_upper_bound + self.max_output_tokens + self.max_reasoning_tokens

    @property
    def profile_digest(self) -> str:
        payload = self.profile.model_dump(mode="json", round_trip=True, warnings=False)
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @model_validator(mode="after")
    def _charge_covers_profile(self) -> Self:
        if self.input_tokens_upper_bound < self.profile.input_tokens_max:
            raise ValueError("input token upper bound cannot be below the request profile")
        if self.max_output_tokens != self.profile.output_tokens_max:
            raise ValueError("max_output_tokens must bind the request profile maximum")
        if self.max_reasoning_tokens != self.profile.reasoning_tokens_max:
            raise ValueError("max_reasoning_tokens must bind the request profile maximum")
        if self.total_token_upper_bound > _MAX_INT:
            raise ValueError("total token upper bound is out of range")
        return self


class AttemptResolution(_Frozen):
    """One durable terminal observation with independent certainty per dimension."""

    outcome: TerminalOutcome
    release_capacity: bool
    actual_starts: int | None = Field(default=None, ge=0, le=1)
    actual_token_units: int | None = Field(default=None, ge=0, le=_MAX_INT)
    actual_quota_units: int | None = Field(default=None, ge=0, le=_MAX_INT)

    @property
    def digest(self) -> str:
        payload = self.model_dump(mode="json", round_trip=True, warnings=False)
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class ReservationDenied:
    reason: ReservationDenialReason
    retry_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class AccountCapacity:
    account_id: AccountId
    state: AccountState
    parallel_limit: int
    active_attempts: int
    uncertain_attempts: int
    rpm_starts: int
    tpm_tokens: int
    monthly_quota_units: int
    account_circuit: CircuitValue
    monthly_reset_at_ms: int
    recovery_probe_in_flight: bool
    consumed_probe_tokens: int


@dataclass(slots=True)
class _Start:
    attempt_id: str
    started_at_ms: int
    tokens: int
    counted_start: bool = True


@dataclass(slots=True)
class _Attempt:
    attempt_id: str
    request_id: str
    leg_id: str
    deployment_generation_id: str
    installation_id: str
    bundle_generation: int
    bundle_hash: str
    fence_token: int
    boot_id: str
    deadline_at_ms: int
    profile_digest: str
    circuit_epoch: int
    account_circuit_epoch: int
    circuit_probe_id: str | None
    account_circuit_probe_id: str | None
    monthly_reset_at_ms: int
    started_at_ms: int
    dispatched_at_ms: int | None
    dispatch_receipt: JournalReceipt | None
    reserved_tokens: int
    quota_units: int
    state: str


@dataclass(frozen=True, slots=True)
class _Dispatch:
    request_id: str
    attempt_id: str
    leg_id: str
    deployment_generation_id: str
    circuit_epoch: int
    account_circuit_epoch: int
    circuit_probe_id: str | None
    account_circuit_probe_id: str | None
    dispatched_at_ms: int


@dataclass(frozen=True, slots=True)
class _ResolvedAttempt:
    resolution_digest: str
    receipt: JournalReceipt
    resolved_at_ms: int
    dispatch: _Dispatch | None


class AccountBindingConflict(ValueError):
    pass


class CredentialAttestationRollback(ValueError):
    pass


class InvalidLeaseTransition(ValueError):
    pass


class Lease:
    """One idempotent handle; ambiguity deliberately does not release capacity."""

    __slots__ = ("_runtime", "request", "reservation_receipt")

    def __init__(
        self,
        runtime: AccountRuntime,
        request: ReservationRequest,
        reservation_receipt: JournalReceipt,
    ) -> None:
        self._runtime = runtime
        self.request = request
        self.reservation_receipt = reservation_receipt

    @property
    def terminal(self) -> bool:
        return self.request.attempt_id in self._runtime._resolutions

    @property
    def dispatched(self) -> bool:
        attempt = self._runtime._attempts.get(self.request.attempt_id)
        return attempt is not None and attempt.dispatched_at_ms is not None

    def mark_dispatched(self) -> JournalReceipt:
        return self._runtime._mark_dispatched(self.request.attempt_id)

    async def mark_dispatched_async(self) -> JournalReceipt:
        return await asyncio.to_thread(self.mark_dispatched)

    def provider_send_completed(self) -> None:
        self._runtime._provider_send_completed(self.request.attempt_id)

    def finish(self, resolution: AttemptResolution) -> JournalReceipt:
        return self._runtime._finish(self.request.attempt_id, resolution)

    async def finish_async(self, resolution: AttemptResolution) -> JournalReceipt:
        """Run the blocking durable finish off the event loop."""
        return await asyncio.to_thread(self.finish, resolution)


@dataclass(frozen=True, slots=True)
class ReservationGranted:
    lease: Lease


class AccountRuntime:
    """Mutable counters shared by every deployment generation on one Account ID."""

    def __init__(
        self,
        account: ProviderAccount,
        *,
        journal: AttemptJournal,
        clock: Clock,
        lock: threading.RLock,
        snapshot: Callable[[], dict[str, JsonValue]],
        resolution_ledger_limit: int,
        resolution_retention_ms: int,
    ) -> None:
        self.account = account
        self._journal = journal
        self._clock = clock
        self._lock = lock
        self._snapshot = snapshot
        self._resolution_ledger_limit = resolution_ledger_limit
        self._resolution_retention_ms = resolution_retention_ms
        self._starts: dict[str, _Start] = {}
        self._attempts: dict[str, _Attempt] = {}
        self._resolutions: dict[str, _ResolvedAttempt] = {}
        self._circuits: dict[DeploymentGenerationId, CircuitValue] = {}
        self._account_circuit = CircuitValue.closed()
        self._consumed_circuit_probes: dict[str, str] = {}
        self._monthly_used = 0
        self._time_high_water_ms = clock.now_ms()
        self._monthly_reset_at_ms = self._next_monthly_reset(self._time_high_water_ms)
        self._external_uncertain_holds = 0
        self._recovery_probe_id: str | None = None

    @staticmethod
    def _dispatch(attempt: _Attempt) -> _Dispatch | None:
        if attempt.dispatched_at_ms is None:
            return None
        return _Dispatch(
            request_id=attempt.request_id,
            attempt_id=attempt.attempt_id,
            leg_id=attempt.leg_id,
            deployment_generation_id=attempt.deployment_generation_id,
            circuit_epoch=attempt.circuit_epoch,
            account_circuit_epoch=attempt.account_circuit_epoch,
            circuit_probe_id=attempt.circuit_probe_id,
            account_circuit_probe_id=attempt.account_circuit_probe_id,
            dispatched_at_ms=attempt.dispatched_at_ms,
        )

    @staticmethod
    def _restore_dispatch(value: object) -> _Dispatch | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("recovered dispatch ledger value is invalid")
        return _Dispatch(
            request_id=_require_str(value["request_id"]),
            attempt_id=_require_str(value["attempt_id"]),
            leg_id=_require_str(value["leg_id"]),
            deployment_generation_id=_require_str(
                value["deployment_generation_id"]
            ),
            circuit_epoch=_require_int(value["circuit_epoch"]),
            account_circuit_epoch=_require_int(value["account_circuit_epoch"]),
            circuit_probe_id=(
                None
                if value["circuit_probe_id"] is None
                else _require_str(value["circuit_probe_id"])
            ),
            account_circuit_probe_id=(
                None
                if value["account_circuit_probe_id"] is None
                else _require_str(value["account_circuit_probe_id"])
            ),
            dispatched_at_ms=_require_int(value["dispatched_at_ms"]),
        )

    def update_account(self, account: ProviderAccount) -> None:
        if account.account_id != self.account.account_id:
            raise ValueError("cannot replace an AccountRuntime identity")
        now_ms = self._now_ms()
        self._roll_monthly(now_ms)
        previous_reset = self._monthly_reset_at_ms
        self.account = account
        candidate_reset = self._next_monthly_reset(now_ms)
        if previous_reset == 0 or candidate_reset > previous_reset:
            self._monthly_reset_at_ms = candidate_reset

    def try_reserve(self, request: ReservationRequest) -> ReservationGranted | ReservationDenied:
        with self._lock:
            if self._journal.status is JournalStatus.RECOVERY_REQUIRED:
                return ReservationDenied(ReservationDenialReason.RECOVERY_REQUIRED)
            if self._journal.status is JournalStatus.ADMISSION_STOP:
                return ReservationDenied(ReservationDenialReason.JOURNAL_CAPACITY_STOP)
            if request.account_id != self.account.account_id:
                return ReservationDenied(ReservationDenialReason.ACCOUNT_NOT_FOUND)
            if self._recovery_probe_id is not None:
                return ReservationDenied(ReservationDenialReason.RECOVERY_REQUIRED)
            if self.account.state is not AccountState.ACTIVE:
                return ReservationDenied(ReservationDenialReason.ACCOUNT_NOT_ACTIVE)
            now_ms = self._now_ms()
            self._prune_resolutions(now_ms)
            unresolved = sum(attempt_id not in self._resolutions for attempt_id in self._attempts)
            if len(self._resolutions) + unresolved >= self._resolution_ledger_limit:
                return ReservationDenied(ReservationDenialReason.JOURNAL_CAPACITY_STOP)
            if request.quota_units < self.account.quota_units_per_attempt:
                return ReservationDenied(ReservationDenialReason.INVALID_QUOTA_CHARGE)
            if request.deadline_at_ms <= now_ms:
                return ReservationDenied(ReservationDenialReason.DEADLINE_EXCEEDED)
            if (
                request.attempt_id in self._attempts
                or request.attempt_id in self._starts
                or request.attempt_id in self._resolutions
            ):
                return ReservationDenied(ReservationDenialReason.ATTEMPT_DUPLICATE)
            if self._account_circuit != request.account_circuit:
                return ReservationDenied(ReservationDenialReason.CIRCUIT_CHANGED)
            if self._account_circuit.state is CircuitState.OPEN:
                return ReservationDenied(
                    ReservationDenialReason.CIRCUIT_UNAVAILABLE,
                    retry_at_ms=self._account_circuit.retry_at_ms,
                )
            current_circuit = self.circuit_value(request.deployment_generation_id)
            if current_circuit != request.circuit:
                return ReservationDenied(ReservationDenialReason.CIRCUIT_CHANGED)
            if current_circuit.state is CircuitState.OPEN:
                return ReservationDenied(
                    ReservationDenialReason.CIRCUIT_UNAVAILABLE,
                    retry_at_ms=current_circuit.retry_at_ms,
                )
            probe_ids = tuple(
                probe_id
                for probe_id in (
                    (
                        self._account_circuit.probe_id
                        if self._account_circuit.state is CircuitState.HALF_OPEN
                        else None
                    ),
                    (
                        current_circuit.probe_id
                        if current_circuit.state is CircuitState.HALF_OPEN
                        else None
                    ),
                )
                if probe_id is not None
            )
            if any(probe_id in self._consumed_circuit_probes for probe_id in probe_ids):
                return ReservationDenied(ReservationDenialReason.CIRCUIT_UNAVAILABLE)
            self._purge_windows(now_ms)
            self._roll_monthly(now_ms)
            if (
                self.account.monthly_quota_units.status is QuotaDimensionStatus.KNOWN
                and request.quota_units > 0
                and self._monthly_reset_at_ms == 0
            ):
                return ReservationDenied(ReservationDenialReason.MONTHLY_RESET_UNAVAILABLE)
            if self._active_count >= self.account.enforced_max_in_flight:
                return ReservationDenied(ReservationDenialReason.PARALLEL_EXHAUSTED)

            rpm = self.account.rpm_limit
            if rpm.status is QuotaDimensionStatus.KNOWN:
                assert rpm.value is not None
                rpm_starts = self._rpm_starts(now_ms)
                if len(rpm_starts) >= rpm.value:
                    return ReservationDenied(
                        ReservationDenialReason.RPM_EXHAUSTED,
                        retry_at_ms=(
                            min(start.started_at_ms for start in rpm_starts)
                            + self.account.rpm_window_seconds * 1000
                        ),
                    )

            tpm = self.account.tpm_limit
            if tpm.status is QuotaDimensionStatus.KNOWN:
                assert tpm.value is not None
                if request.total_token_upper_bound > tpm.value:
                    return ReservationDenied(ReservationDenialReason.TPM_EXHAUSTED)
                tpm_starts = self._tpm_starts(now_ms)
                projected_tokens = (
                    sum(start.tokens for start in tpm_starts) + request.total_token_upper_bound
                )
                if projected_tokens > tpm.value:
                    return ReservationDenied(
                        ReservationDenialReason.TPM_EXHAUSTED,
                        retry_at_ms=self._tpm_retry_at(
                            tpm_starts, request.total_token_upper_bound, tpm.value
                        ),
                    )

            monthly = self.account.monthly_quota_units
            if monthly.status is QuotaDimensionStatus.KNOWN:
                assert monthly.value is not None
                if self._monthly_used + request.quota_units > monthly.value:
                    return ReservationDenied(
                        ReservationDenialReason.MONTHLY_QUOTA_EXHAUSTED,
                        retry_at_ms=self._monthly_reset_at_ms,
                    )

            durable = DurableReservation(
                request_id=str(request.request_id),
                attempt_id=str(request.attempt_id),
                account_id=str(request.account_id),
                leg_id=str(request.leg_id),
                deployment_generation_id=str(request.deployment_generation_id),
                installation_id=str(request.runtime_identity.installation_id),
                bundle_generation=request.runtime_identity.bundle_generation,
                bundle_hash=str(request.runtime_identity.bundle_hash),
                fence_token=request.runtime_identity.dispatcher_fence,
                boot_id=str(request.runtime_identity.boot_id),
                deadline_at_ms=request.deadline_at_ms,
                profile_digest=request.profile_digest,
                started_at_ms=now_ms,
                reserved_tokens=request.total_token_upper_bound,
                quota_units=request.quota_units,
                monthly_reset_at_ms=self._monthly_reset_at_ms,
                circuit_epoch=request.circuit.epoch,
                account_circuit_epoch=request.account_circuit.epoch,
                circuit_probe_id=request.circuit.probe_id,
                account_circuit_probe_id=(
                    self._account_circuit.probe_id
                    if self._account_circuit.state is CircuitState.HALF_OPEN
                    else None
                ),
            )
            try:
                receipt = self._journal.reserve_before_send(durable)
            except JournalUnavailable as error:
                if str(error) == JournalStatus.ADMISSION_STOP.value:
                    return ReservationDenied(ReservationDenialReason.JOURNAL_CAPACITY_STOP)
                return ReservationDenied(ReservationDenialReason.JOURNAL_UNAVAILABLE)
            for probe_id in probe_ids:
                self._consumed_circuit_probes[probe_id] = str(request.attempt_id)

            self._starts[request.attempt_id] = _Start(
                request.attempt_id, now_ms, request.total_token_upper_bound
            )
            self._attempts[request.attempt_id] = _Attempt(
                attempt_id=str(request.attempt_id),
                request_id=str(request.request_id),
                leg_id=str(request.leg_id),
                deployment_generation_id=str(request.deployment_generation_id),
                installation_id=str(request.runtime_identity.installation_id),
                bundle_generation=request.runtime_identity.bundle_generation,
                bundle_hash=str(request.runtime_identity.bundle_hash),
                fence_token=request.runtime_identity.dispatcher_fence,
                boot_id=str(request.runtime_identity.boot_id),
                deadline_at_ms=request.deadline_at_ms,
                profile_digest=request.profile_digest,
                circuit_epoch=request.circuit.epoch,
                account_circuit_epoch=request.account_circuit.epoch,
                circuit_probe_id=request.circuit.probe_id,
                account_circuit_probe_id=(
                    self._account_circuit.probe_id
                    if self._account_circuit.state is CircuitState.HALF_OPEN
                    else None
                ),
                monthly_reset_at_ms=self._monthly_reset_at_ms,
                started_at_ms=now_ms,
                dispatched_at_ms=None,
                dispatch_receipt=None,
                reserved_tokens=request.total_token_upper_bound,
                quota_units=request.quota_units,
                state="active",
            )
            self._monthly_used += request.quota_units
            self._maybe_checkpoint_locked()
            return ReservationGranted(Lease(self, request, receipt))

    def circuit_value(self, generation: DeploymentGenerationId) -> CircuitValue:
        return self._circuits.get(generation, CircuitValue.closed())

    async def try_reserve_async(
        self, request: ReservationRequest
    ) -> ReservationGranted | ReservationDenied:
        """Run durable reservation off-loop and reconcile cancellation as no-send."""
        worker = asyncio.create_task(asyncio.to_thread(self.try_reserve, request))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            result = await asyncio.shield(worker)
            if isinstance(result, ReservationGranted):
                await result.lease.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.CLIENT_CANCELLED,
                        release_capacity=True,
                        actual_starts=0,
                        actual_token_units=0,
                        actual_quota_units=0,
                    )
                )
            raise

    def compare_and_swap_circuit(
        self,
        generation: DeploymentGenerationId,
        expected: CircuitValue,
        replacement: CircuitValue,
    ) -> bool:
        with self._lock:
            current = self.circuit_value(generation)
            if current != expected:
                return False
            if replacement.epoch < expected.epoch:
                raise ValueError("circuit epoch cannot decrease")
            if replacement == expected:
                return True
            self._journal.update_circuit(
                account_id=str(self.account.account_id),
                scope=CircuitScope.DEPLOYMENT.value,
                deployment_generation_id=str(generation),
                state=replacement.state.value,
                cause=None if replacement.cause is None else replacement.cause.value,
                opened_at_ms=replacement.opened_at_ms,
                backoff_step=replacement.backoff_step,
                evidence_digest=replacement.evidence_digest,
                epoch=replacement.epoch,
                retry_at_ms=replacement.retry_at_ms,
                probe_id=replacement.probe_id,
            )
            if expected.probe_id is not None:
                self._consumed_circuit_probes.pop(expected.probe_id, None)
            self._circuits[generation] = replacement
            self._maybe_checkpoint_locked()
            return True

    def finish_attempt(self, attempt_id: str, resolution: AttemptResolution) -> JournalReceipt:
        """Idempotently reconcile a durable attempt, including after restart."""
        return self._finish(attempt_id, resolution)

    async def finish_attempt_async(
        self, attempt_id: str, resolution: AttemptResolution
    ) -> JournalReceipt:
        return await asyncio.to_thread(self.finish_attempt, attempt_id, resolution)

    def account_circuit_value(self) -> CircuitValue:
        return self._account_circuit

    def compare_and_swap_account_circuit(
        self, expected: CircuitValue, replacement: CircuitValue
    ) -> bool:
        with self._lock:
            if self._account_circuit != expected:
                return False
            if replacement.epoch < expected.epoch:
                raise ValueError("circuit epoch cannot decrease")
            if replacement == expected:
                return True
            self._journal.update_circuit(
                account_id=str(self.account.account_id),
                scope=CircuitScope.ACCOUNT.value,
                deployment_generation_id=None,
                state=replacement.state.value,
                cause=None if replacement.cause is None else replacement.cause.value,
                opened_at_ms=replacement.opened_at_ms,
                backoff_step=replacement.backoff_step,
                evidence_digest=replacement.evidence_digest,
                epoch=replacement.epoch,
                retry_at_ms=replacement.retry_at_ms,
                probe_id=replacement.probe_id,
            )
            if expected.probe_id is not None:
                self._consumed_circuit_probes.pop(expected.probe_id, None)
            self._account_circuit = replacement
            self._maybe_checkpoint_locked()
            return True

    def capacity(self) -> AccountCapacity:
        with self._lock:
            now_ms = self._now_ms()
            self._purge_windows(now_ms)
            self._roll_monthly(now_ms)
            return AccountCapacity(
                account_id=self.account.account_id,
                state=self.account.state or AccountState.DRAFT,
                parallel_limit=self.account.enforced_max_in_flight,
                active_attempts=self._active_count,
                uncertain_attempts=self._uncertain_count,
                rpm_starts=len(self._rpm_starts(now_ms)),
                tpm_tokens=sum(start.tokens for start in self._tpm_starts(now_ms)),
                monthly_quota_units=self._monthly_used,
                monthly_reset_at_ms=self._monthly_reset_at_ms,
                recovery_probe_in_flight=self._recovery_probe_id is not None,
                account_circuit=self._account_circuit,
                consumed_probe_tokens=len(self._consumed_circuit_probes),
            )

    def apply_authoritative_active_count(self, active_count: int) -> None:
        if active_count < 0:
            raise ValueError("authoritative active count cannot be negative")
        with self._lock:
            if any(attempt.state == "active" for attempt in self._attempts.values()):
                raise ValueError("authoritative count cannot evict live active attempts")
            self._journal.record_authoritative_active_count(
                account_id=str(self.account.account_id), active_count=active_count
            )
            self._apply_authoritative_active_count(active_count)
            self._maybe_checkpoint_locked()

    def begin_recovery_probe(self, probe_id: str) -> bool:
        if not probe_id.startswith("probe_"):
            raise ValueError("recovery probe requires a typed probe ID")
        with self._lock:
            if (
                self._recovery_probe_id is not None
                or self._uncertain_count == 0
                or any(attempt.state == "active" for attempt in self._attempts.values())
            ):
                return False
            self._journal.record_recovery_probe_started(
                account_id=str(self.account.account_id), probe_id=probe_id
            )
            self._recovery_probe_id = probe_id
            self._maybe_checkpoint_locked()
            return True

    def finish_recovery_probe(self, probe_id: str, classification: ProbeClassification) -> None:
        with self._lock:
            if self._recovery_probe_id != probe_id:
                raise ValueError("recovery probe is not the serialized active probe")
            if any(attempt.state == "active" for attempt in self._attempts.values()):
                raise ValueError("recovery probe cannot evict live active attempts")
            self._journal.record_recovery_probe_finished(
                account_id=str(self.account.account_id),
                probe_id=probe_id,
                classification=classification.value,
            )
            self._apply_recovery_probe_finished(classification)
            self._maybe_checkpoint_locked()

    def _attempt_exists(self, attempt_id: str) -> bool:
        with self._lock:
            return attempt_id in self._attempts and attempt_id not in self._resolutions

    def _mark_dispatched(self, attempt_id: str) -> JournalReceipt:
        with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None or attempt_id in self._resolutions:
                raise InvalidLeaseTransition("unknown or terminal attempt cannot dispatch")
            if attempt.dispatch_receipt is not None:
                return attempt.dispatch_receipt
            dispatched_at_ms = self._now_ms()
            if dispatched_at_ms >= attempt.deadline_at_ms:
                raise InvalidLeaseTransition("attempt deadline elapsed before dispatch")
            receipt = self._journal.record_dispatch(
                attempt_id=attempt_id,
                dispatched_at_ms=dispatched_at_ms,
            )
            attempt.dispatched_at_ms = dispatched_at_ms
            attempt.dispatch_receipt = receipt
            self._maybe_checkpoint_locked()
            return receipt

    def _provider_send_completed(self, attempt_id: str) -> None:
        with self._lock:
            if attempt_id not in self._attempts or attempt_id in self._resolutions:
                raise InvalidLeaseTransition("attempt is already terminal")
            self._journal.provider_send_completed(attempt_id)

    def _finish(self, attempt_id: str, resolution: AttemptResolution) -> JournalReceipt:
        with self._lock:
            resolved = self._resolutions.get(attempt_id)
            if resolved is not None:
                if resolved.resolution_digest != resolution.digest:
                    raise InvalidLeaseTransition("conflicting terminal resolution")
                return resolved.receipt
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise InvalidLeaseTransition("unknown attempt cannot be resolved")
            if (
                resolution.actual_token_units is not None
                and resolution.actual_token_units > attempt.reserved_tokens
            ):
                raise InvalidLeaseTransition("actual token usage exceeds reservation")
            if (
                resolution.actual_quota_units is not None
                and resolution.actual_quota_units > attempt.quota_units
            ):
                raise InvalidLeaseTransition("actual quota usage exceeds reservation")

            resolved_at_ms = self._now_ms()
            receipt = self._journal.record_resolution(
                attempt_id=attempt_id,
                outcome=resolution.outcome.value,
                release_capacity=resolution.release_capacity,
                actual_starts=resolution.actual_starts,
                actual_token_units=resolution.actual_token_units,
                actual_quota_units=resolution.actual_quota_units,
                resolution_digest=resolution.digest,
                resolved_at_ms=resolved_at_ms,
            )
            start = self._starts[attempt_id]
            if resolution.actual_starts == 0:
                start.counted_start = False
            if resolution.actual_token_units is not None:
                start.tokens = resolution.actual_token_units
            if (
                resolution.actual_quota_units is not None
                and attempt.monthly_reset_at_ms == self._monthly_reset_at_ms
            ):
                self._monthly_used += resolution.actual_quota_units - attempt.quota_units
                self._monthly_used = max(0, self._monthly_used)
            if resolution.release_capacity:
                del self._attempts[attempt_id]
            else:
                attempt.state = "uncertain"
            self._resolutions[attempt_id] = _ResolvedAttempt(
                resolution.digest,
                receipt,
                resolved_at_ms,
                self._dispatch(attempt),
            )
            self._maybe_checkpoint_locked()
            return receipt

    @property
    def _active_count(self) -> int:
        return len(self._attempts) + self._external_uncertain_holds

    @property
    def _uncertain_count(self) -> int:
        return (
            sum(attempt.state == "uncertain" for attempt in self._attempts.values())
            + self._external_uncertain_holds
        )

    def _rpm_starts(self, now_ms: int) -> list[_Start]:
        floor = now_ms - self.account.rpm_window_seconds * 1000
        return [
            start
            for start in self._starts.values()
            if start.counted_start and start.started_at_ms > floor
        ]

    def _prune_resolutions(self, now_ms: int) -> None:
        cutoff = now_ms - self._resolution_retention_ms
        expired = [
            attempt_id
            for attempt_id, resolved in self._resolutions.items()
            if resolved.resolved_at_ms <= cutoff and attempt_id not in self._attempts
        ]
        for attempt_id in expired:
            del self._resolutions[attempt_id]

    def _tpm_starts(self, now_ms: int) -> list[_Start]:
        floor = now_ms - self.account.tpm_window_seconds * 1000
        return [start for start in self._starts.values() if start.started_at_ms > floor]

    def _now_ms(self) -> int:
        self._time_high_water_ms = max(self._time_high_water_ms, self._clock.now_ms())
        return self._time_high_water_ms

    def _purge_windows(self, now_ms: int) -> None:
        keep_ms = _MAX_ROLLING_WINDOW_SECONDS * 1000
        expired = [
            attempt_id
            for attempt_id, start in self._starts.items()
            if start.started_at_ms <= now_ms - keep_ms and attempt_id not in self._attempts
        ]
        for attempt_id in expired:
            del self._starts[attempt_id]

    def _tpm_retry_at(self, starts: list[_Start], requested: int, limit: int) -> int:
        total = sum(start.tokens for start in starts)
        for start in sorted(starts, key=lambda item: (item.started_at_ms, item.attempt_id)):
            total -= start.tokens
            if total + requested <= limit:
                return start.started_at_ms + self.account.tpm_window_seconds * 1000
        return max(start.started_at_ms for start in starts) + self.account.tpm_window_seconds * 1000

    def _next_monthly_reset(self, now_ms: int) -> int:
        if self.account.monthly_quota_units.status is not QuotaDimensionStatus.KNOWN:
            return 0
        reset_at_ms = self.account.monthly_reset_at_ms or 0
        return reset_at_ms if reset_at_ms > now_ms else 0

    def _roll_monthly(self, now_ms: int) -> None:
        if self._monthly_reset_at_ms and now_ms >= self._monthly_reset_at_ms:
            self._monthly_used = 0
            self._monthly_reset_at_ms = 0

    def _apply_authoritative_active_count(self, active_count: int) -> None:
        if any(attempt.state == "active" for attempt in self._attempts.values()):
            raise ValueError("authoritative count cannot evict live active attempts")
        ordered = sorted(
            self._attempts,
            key=lambda attempt_id: (
                self._attempts[attempt_id].started_at_ms,
                attempt_id,
            ),
        )
        keep = min(active_count, len(ordered))
        for attempt_id in ordered[keep:]:
            del self._attempts[attempt_id]
        self._external_uncertain_holds = max(0, active_count - keep)

    def _apply_recovery_probe_finished(self, classification: ProbeClassification) -> None:
        if classification is ProbeClassification.AVAILABLE:
            if self._external_uncertain_holds:
                self._external_uncertain_holds -= 1
            else:
                uncertain = [
                    attempt for attempt in self._attempts.values() if attempt.state == "uncertain"
                ]
                if uncertain:
                    oldest = min(
                        uncertain,
                        key=lambda attempt: (attempt.started_at_ms, attempt.attempt_id),
                    )
                    del self._attempts[oldest.attempt_id]
        self._recovery_probe_id = None

    def _maybe_checkpoint_locked(self) -> None:
        if self.account.state is not AccountState.ACTIVE:
            self._journal.force_checkpoint(self._snapshot())
        elif self._journal.checkpoint_due:
            self._journal.maybe_checkpoint(self._snapshot())

    def _snapshot_state(self) -> dict[str, JsonValue]:
        return {
            "starts": {
                attempt_id: {
                    "started_at_ms": start.started_at_ms,
                    "tokens": start.tokens,
                    "counted_start": start.counted_start,
                }
                for attempt_id, start in sorted(self._starts.items())
            },
            "attempts": {
                attempt_id: {
                    "request_id": attempt.request_id,
                    "leg_id": attempt.leg_id,
                    "deployment_generation_id": attempt.deployment_generation_id,
                    "installation_id": attempt.installation_id,
                    "bundle_generation": attempt.bundle_generation,
                    "bundle_hash": attempt.bundle_hash,
                    "fence_token": attempt.fence_token,
                    "boot_id": attempt.boot_id,
                    "deadline_at_ms": attempt.deadline_at_ms,
                    "profile_digest": attempt.profile_digest,
                    "circuit_epoch": attempt.circuit_epoch,
                    "account_circuit_epoch": attempt.account_circuit_epoch,
                    "circuit_probe_id": attempt.circuit_probe_id,
                    "account_circuit_probe_id": attempt.account_circuit_probe_id,
                    "monthly_reset_at_ms": attempt.monthly_reset_at_ms,
                    "started_at_ms": attempt.started_at_ms,
                    "dispatched_at_ms": attempt.dispatched_at_ms,
                    "dispatch_lsn": (
                        None
                        if attempt.dispatch_receipt is None
                        else attempt.dispatch_receipt.durable_lsn
                    ),
                    "dispatch_digest": (
                        None
                        if attempt.dispatch_receipt is None
                        else attempt.dispatch_receipt.record_digest
                    ),
                    "reserved_tokens": attempt.reserved_tokens,
                    "quota_units": attempt.quota_units,
                    "state": attempt.state,
                }
                for attempt_id, attempt in sorted(self._attempts.items())
            },
            "resolutions": {
                attempt_id: {
                    "resolution_digest": resolved.resolution_digest,
                    "durable_lsn": resolved.receipt.durable_lsn,
                    "record_digest": resolved.receipt.record_digest,
                    "resolved_at_ms": resolved.resolved_at_ms,
                    "dispatch": (
                        None
                        if resolved.dispatch is None
                        else {
                            "request_id": resolved.dispatch.request_id,
                            "attempt_id": resolved.dispatch.attempt_id,
                            "leg_id": resolved.dispatch.leg_id,
                            "deployment_generation_id": (
                                resolved.dispatch.deployment_generation_id
                            ),
                            "circuit_epoch": resolved.dispatch.circuit_epoch,
                            "account_circuit_epoch": (
                                resolved.dispatch.account_circuit_epoch
                            ),
                            "circuit_probe_id": resolved.dispatch.circuit_probe_id,
                            "account_circuit_probe_id": (
                                resolved.dispatch.account_circuit_probe_id
                            ),
                            "dispatched_at_ms": resolved.dispatch.dispatched_at_ms,
                        }
                    ),
                }
                for attempt_id, resolved in sorted(self._resolutions.items())
            },
            "consumed_circuit_probes": dict(sorted(self._consumed_circuit_probes.items())),
            "time_high_water_ms": self._time_high_water_ms,
            "monthly_used": self._monthly_used,
            "monthly_reset_at_ms": self._monthly_reset_at_ms,
            "external_uncertain_holds": self._external_uncertain_holds,
            "recovery_probe_id": self._recovery_probe_id,
            "account_circuit": {
                "state": self._account_circuit.state.value,
                "cause": (
                    None
                    if self._account_circuit.cause is None
                    else self._account_circuit.cause.value
                ),
                "opened_at_ms": self._account_circuit.opened_at_ms,
                "backoff_step": self._account_circuit.backoff_step,
                "evidence_digest": self._account_circuit.evidence_digest,
                "epoch": self._account_circuit.epoch,
                "retry_at_ms": self._account_circuit.retry_at_ms,
                "probe_id": self._account_circuit.probe_id,
            },
            "circuits": {
                str(generation): {
                    "state": value.state.value,
                    "cause": None if value.cause is None else value.cause.value,
                    "opened_at_ms": value.opened_at_ms,
                    "backoff_step": value.backoff_step,
                    "evidence_digest": value.evidence_digest,
                    "epoch": value.epoch,
                    "retry_at_ms": value.retry_at_ms,
                    "probe_id": value.probe_id,
                }
                for generation, value in sorted(
                    self._circuits.items(), key=lambda item: str(item[0])
                )
            },
        }

    def _restore_snapshot_state(self, value: Mapping[str, object]) -> None:
        starts = cast(Mapping[str, Mapping[str, int]], value["starts"])
        attempts = cast(Mapping[str, Mapping[str, object]], value["attempts"])
        resolutions = cast(Mapping[str, Mapping[str, object]], value["resolutions"])
        if len(resolutions) > self._resolution_ledger_limit:
            raise ValueError("recovered resolution ledger exceeds configured bound")
        circuits = cast(Mapping[str, Mapping[str, object]], value["circuits"])
        consumed_probes = cast(Mapping[str, object], value["consumed_circuit_probes"])
        account_circuit = cast(Mapping[str, object], value["account_circuit"])
        self._starts = {
            attempt_id: _Start(
                attempt_id,
                _require_int(item["started_at_ms"]),
                _require_int(item["tokens"]),
                _require_bool(item["counted_start"]),
            )
            for attempt_id, item in starts.items()
        }
        self._attempts = {
            attempt_id: _Attempt(
                attempt_id=attempt_id,
                request_id=_require_str(item["request_id"]),
                leg_id=_require_str(item["leg_id"]),
                deployment_generation_id=_require_str(item["deployment_generation_id"]),
                installation_id=_require_str(item["installation_id"]),
                bundle_generation=_require_int(item["bundle_generation"]),
                bundle_hash=_require_str(item["bundle_hash"]),
                fence_token=_require_int(item["fence_token"]),
                boot_id=_require_str(item["boot_id"]),
                deadline_at_ms=_require_int(item["deadline_at_ms"]),
                profile_digest=_require_str(item["profile_digest"]),
                circuit_epoch=_require_int(item["circuit_epoch"]),
                account_circuit_epoch=_require_int(item["account_circuit_epoch"]),
                circuit_probe_id=(
                    None
                    if item["circuit_probe_id"] is None
                    else _require_str(item["circuit_probe_id"])
                ),
                account_circuit_probe_id=(
                    None
                    if item["account_circuit_probe_id"] is None
                    else _require_str(item["account_circuit_probe_id"])
                ),
                monthly_reset_at_ms=_require_int(item["monthly_reset_at_ms"]),
                started_at_ms=_require_int(item["started_at_ms"]),
                dispatched_at_ms=(
                    None
                    if item["dispatched_at_ms"] is None
                    else _require_int(item["dispatched_at_ms"])
                ),
                dispatch_receipt=(
                    None
                    if item["dispatch_lsn"] is None
                    else JournalReceipt(
                        _require_int(item["dispatch_lsn"]),
                        _require_str(item["dispatch_digest"]),
                    )
                ),
                reserved_tokens=_require_int(item["reserved_tokens"]),
                quota_units=_require_int(item["quota_units"]),
                state=_require_str(item["state"]),
            )
            for attempt_id, item in attempts.items()
        }
        self._time_high_water_ms = _require_int(value["time_high_water_ms"])
        self._resolutions = {
            attempt_id: _ResolvedAttempt(
                _require_str(item["resolution_digest"]),
                JournalReceipt(
                    _require_int(item["durable_lsn"]),
                    _require_str(item["record_digest"]),
                ),
                _require_int(item["resolved_at_ms"]),
                self._restore_dispatch(item["dispatch"]),
            )
            for attempt_id, item in resolutions.items()
        }
        self._consumed_circuit_probes = {
            _require_str(probe_id): _require_str(attempt_id)
            for probe_id, attempt_id in consumed_probes.items()
        }
        self._monthly_used = _require_int(value["monthly_used"])
        self._monthly_reset_at_ms = _require_int(value["monthly_reset_at_ms"])
        self._external_uncertain_holds = _require_int(value["external_uncertain_holds"])
        probe = value["recovery_probe_id"]
        self._recovery_probe_id = None if probe is None else _require_str(probe)
        self._account_circuit = CircuitValue(
            state=CircuitState(_require_str(account_circuit["state"])),
            cause=(
                None
                if account_circuit["cause"] is None
                else CircuitCause(_require_str(account_circuit["cause"]))
            ),
            epoch=_require_int(account_circuit["epoch"]),
            opened_at_ms=_require_int(account_circuit["opened_at_ms"]),
            retry_at_ms=_require_int(account_circuit["retry_at_ms"]),
            backoff_step=_require_int(account_circuit["backoff_step"]),
            evidence_digest=(
                None
                if account_circuit["evidence_digest"] is None
                else _require_str(account_circuit["evidence_digest"])
            ),
            probe_id=(
                None
                if account_circuit["probe_id"] is None
                else _require_str(account_circuit["probe_id"])
            ),
        )
        self._circuits = {
            DeploymentGenerationId(generation): CircuitValue(
                state=CircuitState(_require_str(item["state"])),
                cause=(
                    None if item["cause"] is None else CircuitCause(_require_str(item["cause"]))
                ),
                epoch=_require_int(item["epoch"]),
                opened_at_ms=_require_int(item["opened_at_ms"]),
                retry_at_ms=_require_int(item["retry_at_ms"]),
                backoff_step=_require_int(item["backoff_step"]),
                evidence_digest=(
                    None
                    if item["evidence_digest"] is None
                    else _require_str(item["evidence_digest"])
                ),
                probe_id=None if item["probe_id"] is None else _require_str(item["probe_id"]),
            )
            for generation, item in circuits.items()
        }


@dataclass(frozen=True, slots=True)
class AttemptOperationalValue:
    account_id: AccountId
    request_id: RequestId
    attempt_id: AttemptId
    leg_id: RouteLegId
    deployment_generation_id: DeploymentGenerationId
    installation_id: InstallationId
    bundle_generation: int
    bundle_hash: BundleHash
    dispatcher_fence: int
    boot_id: GatewayBootId
    deadline_at_ms: int
    profile_digest: str
    circuit_epoch: int
    account_circuit_epoch: int
    circuit_probe_id: str | None
    account_circuit_probe_id: str | None
    dispatched_at_ms: int | None
    state: str


@dataclass(frozen=True, slots=True)
class DispatchOperationalValue:
    account_id: AccountId
    request_id: RequestId
    attempt_id: AttemptId
    leg_id: RouteLegId
    deployment_generation_id: DeploymentGenerationId
    circuit_epoch: int
    account_circuit_epoch: int
    circuit_probe_id: str | None
    account_circuit_probe_id: str | None
    dispatched_at_ms: int


@dataclass(frozen=True, slots=True)
class CircuitOperationalValue:
    account_id: AccountId
    deployment_generation_id: DeploymentGenerationId
    value: CircuitValue


@dataclass(frozen=True, slots=True)
class OperationalRuntimeView:
    """Immutable advisory operational snapshot; never reservation authority."""

    durable_lsn: int
    recovery_state: JournalStatus
    journal_health: JournalHealth
    time_high_water_ms: int
    accounts: tuple[AccountCapacity, ...]
    attempts: tuple[AttemptOperationalValue, ...]
    dispatches: tuple[DispatchOperationalValue, ...]
    circuits: tuple[CircuitOperationalValue, ...]


class RuntimeState:
    """Single mutable owner of journal ordering and provider-account runtimes."""

    def __init__(
        self,
        accounts: Iterable[ProviderAccount],
        *,
        journal: AttemptJournal,
        clock: Clock,
        resolution_ledger_limit: int = _DEFAULT_RESOLUTION_LEDGER_LIMIT,
        resolution_retention_ms: int = _DEFAULT_RESOLUTION_RETENTION_MS,
    ) -> None:
        if resolution_ledger_limit < 1 or resolution_ledger_limit > 100_000:
            raise ValueError("resolution ledger limit must be in [1, 100000]")
        if resolution_retention_ms < 1:
            raise ValueError("resolution retention must be positive")
        self._journal = journal
        self._clock = clock
        self._resolution_ledger_limit = resolution_ledger_limit
        self._resolution_retention_ms = resolution_retention_ms
        # ponytail: one lock preserves record/state/checkpoint order; shard only if measured.
        self._lock = threading.RLock()
        self._runtimes: dict[AccountId, AccountRuntime] = {}
        self._binding_history: dict[str, AccountId] = {}
        self._dormant_account_states: dict[AccountId, Mapping[str, object]] = {}
        self._account_binding: dict[AccountId, str] = {}
        self._credential_epoch_highwater: dict[AccountId, int] = {}
        self._credential_attestation_digest: dict[AccountId, str] = {}
        validated = tuple(ProviderAccount.model_validate(account) for account in accounts)
        for account in validated:
            self._runtimes[account.account_id] = AccountRuntime(
                account,
                journal=journal,
                clock=clock,
                lock=self._lock,
                snapshot=self._snapshot_locked,
                resolution_ledger_limit=resolution_ledger_limit,
                resolution_retention_ms=resolution_retention_ms,
            )

        if journal.recovery.status is JournalStatus.HEALTHY:
            try:
                if journal.recovery.checkpoint is not None:
                    self._restore_checkpoint(journal.recovery.checkpoint)
                for record in journal.recovery.records:
                    self._apply_record(record)
                self.apply_publication(validated)
                self._mark_recovered_attempts_uncertain()
            except (JournalUnavailable, KeyError, TypeError, ValueError):
                self._journal.enter_recovery_required("invalid_runtime_state")
        else:
            for account in validated:
                self._remember_account_without_journal(account)

    def operational_view(self) -> OperationalRuntimeView:
        with self._lock:
            accounts = tuple(
                runtime.capacity()
                for _, runtime in sorted(self._runtimes.items(), key=lambda item: str(item[0]))
            )
            attempts = tuple(
                AttemptOperationalValue(
                    account_id=account_id,
                    request_id=RequestId(attempt.request_id),
                    attempt_id=AttemptId(attempt.attempt_id),
                    leg_id=RouteLegId(attempt.leg_id),
                    deployment_generation_id=DeploymentGenerationId(
                        attempt.deployment_generation_id
                    ),
                    installation_id=InstallationId(attempt.installation_id),
                    bundle_generation=attempt.bundle_generation,
                    bundle_hash=BundleHash(attempt.bundle_hash),
                    dispatcher_fence=attempt.fence_token,
                    boot_id=GatewayBootId(attempt.boot_id),
                    deadline_at_ms=attempt.deadline_at_ms,
                    profile_digest=attempt.profile_digest,
                    circuit_epoch=attempt.circuit_epoch,
                    account_circuit_epoch=attempt.account_circuit_epoch,
                    circuit_probe_id=attempt.circuit_probe_id,
                    account_circuit_probe_id=attempt.account_circuit_probe_id,
                    dispatched_at_ms=attempt.dispatched_at_ms,
                    state=attempt.state,
                )
                for account_id, runtime in sorted(
                    self._runtimes.items(), key=lambda item: str(item[0])
                )
                for attempt in sorted(runtime._attempts.values(), key=lambda item: item.attempt_id)
            )
            dispatch_by_attempt: dict[AttemptId, DispatchOperationalValue] = {}
            for account_id, runtime in self._runtimes.items():
                dispatched = tuple(
                    dispatch
                    for attempt in runtime._attempts.values()
                    if (dispatch := runtime._dispatch(attempt)) is not None
                ) + tuple(
                    resolved.dispatch
                    for resolved in runtime._resolutions.values()
                    if resolved.dispatch is not None
                )
                for dispatch in dispatched:
                    assert dispatch is not None
                    attempt_id = AttemptId(dispatch.attempt_id)
                    dispatch_by_attempt[attempt_id] = DispatchOperationalValue(
                        account_id=account_id,
                        request_id=RequestId(dispatch.request_id),
                        attempt_id=attempt_id,
                        leg_id=RouteLegId(dispatch.leg_id),
                        deployment_generation_id=DeploymentGenerationId(
                            dispatch.deployment_generation_id
                        ),
                        circuit_epoch=dispatch.circuit_epoch,
                        account_circuit_epoch=dispatch.account_circuit_epoch,
                        circuit_probe_id=dispatch.circuit_probe_id,
                        account_circuit_probe_id=dispatch.account_circuit_probe_id,
                        dispatched_at_ms=dispatch.dispatched_at_ms,
                    )
            dispatches = tuple(
                sorted(
                    dispatch_by_attempt.values(),
                    key=lambda value: (value.dispatched_at_ms, str(value.attempt_id)),
                )
            )
            circuits = tuple(
                CircuitOperationalValue(account_id, generation, value)
                for account_id, runtime in sorted(
                    self._runtimes.items(), key=lambda item: str(item[0])
                )
                for generation, value in sorted(
                    runtime._circuits.items(), key=lambda item: str(item[0])
                )
            )
            health = self._journal.health
            return OperationalRuntimeView(
                durable_lsn=health.last_sequence,
                recovery_state=health.status,
                journal_health=health,
                time_high_water_ms=max(
                    (runtime._time_high_water_ms for runtime in self._runtimes.values()),
                    default=self._clock.now_ms(),
                ),
                accounts=accounts,
                attempts=attempts,
                dispatches=dispatches,
                circuits=circuits,
            )

    @property
    def journal_health(self) -> JournalHealth:
        return self._journal.health

    def try_reserve(self, request: ReservationRequest) -> ReservationGranted | ReservationDenied:
        if self._journal.status is JournalStatus.RECOVERY_REQUIRED:
            return ReservationDenied(ReservationDenialReason.RECOVERY_REQUIRED)
        runtime = self._runtimes.get(request.account_id)
        if runtime is None:
            return ReservationDenied(ReservationDenialReason.ACCOUNT_NOT_FOUND)
        return runtime.try_reserve(request)

    async def try_reserve_async(
        self, request: ReservationRequest
    ) -> ReservationGranted | ReservationDenied:
        runtime = self._runtimes.get(request.account_id)
        if runtime is None:
            return ReservationDenied(ReservationDenialReason.ACCOUNT_NOT_FOUND)
        return await runtime.try_reserve_async(request)

    def account_runtime(self, account_id: AccountId) -> AccountRuntime:
        try:
            return self._runtimes[account_id]
        except KeyError:
            raise KeyError(f"unknown provider account {account_id}") from None

    def account_capacity(self, account_id: AccountId) -> AccountCapacity:
        return self.account_runtime(account_id).capacity()

    def apply_publication(self, accounts: Iterable[ProviderAccount]) -> None:
        with self._lock:
            validated = tuple(ProviderAccount.model_validate(account) for account in accounts)
            account_ids = tuple(account.account_id for account in validated)
            if len(set(account_ids)) != len(account_ids):
                raise AccountBindingConflict("publication contains duplicate Account IDs")

            binding_history = dict(self._binding_history)
            account_binding = dict(self._account_binding)
            epoch_highwater = dict(self._credential_epoch_highwater)
            attestation_history = dict(self._credential_attestation_digest)
            staged: list[tuple[ProviderAccount, str, str, int, bool]] = []

            for account in validated:
                binding_digest = self._binding_digest(account)
                attestation_digest = self._attestation_digest(account)
                previous_binding = account_binding.get(account.account_id)
                if previous_binding is not None and previous_binding != binding_digest:
                    raise AccountBindingConflict("Account ID cannot change its provider binding")
                if binding_digest:
                    owner = binding_history.get(binding_digest)
                    if owner is not None and owner != account.account_id:
                        raise AccountBindingConflict(
                            "provider binding is globally unique across live and "
                            "tombstoned accounts"
                        )
                epoch = account.credential_epoch or 0
                highwater = epoch_highwater.get(account.account_id, 0)
                previous_attestation = attestation_history.get(account.account_id)
                if epoch < highwater or (
                    epoch == highwater
                    and previous_attestation is not None
                    and attestation_digest != previous_attestation
                ):
                    raise CredentialAttestationRollback(
                        "credential attestation cannot rewind or change within one epoch"
                    )
                runtime = self._runtimes.get(account.account_id)
                changed = (
                    previous_binding is None
                    or epoch > highwater
                    or runtime is None
                    or runtime.account.state != account.state
                )
                staged.append((account, binding_digest, attestation_digest, epoch, changed))
                if binding_digest:
                    binding_history[binding_digest] = account.account_id
                    account_binding[account.account_id] = binding_digest
                epoch_highwater[account.account_id] = epoch
                attestation_history[account.account_id] = attestation_digest

            omitted: list[ProviderAccount] = []
            selected = set(account_ids)
            for account_id, runtime in self._runtimes.items():
                if account_id not in selected and runtime.account.state is AccountState.ACTIVE:
                    omitted.append(
                        ProviderAccount.model_validate(
                            runtime.account.model_copy(
                                update={"state": AccountState.DISABLED}
                            ).model_dump(mode="python")
                        )
                    )

            for account, binding_digest, attestation_digest, epoch, changed in staged:
                if changed and binding_digest:
                    assert account.state is not None
                    self._journal.register_account(
                        account_id=str(account.account_id),
                        binding_digest=binding_digest,
                        credential_attestation_digest=attestation_digest,
                        credential_epoch=epoch,
                        state=account.state.value,
                    )
            for account in omitted:
                binding_digest = account_binding[account.account_id]
                self._journal.register_account(
                    account_id=str(account.account_id),
                    binding_digest=binding_digest,
                    credential_attestation_digest=attestation_history[account.account_id],
                    credential_epoch=epoch_highwater[account.account_id],
                    state=AccountState.DISABLED.value,
                )

            self._binding_history = binding_history
            self._account_binding = account_binding
            self._credential_epoch_highwater = epoch_highwater
            self._credential_attestation_digest = attestation_history
            for account, _, _, _, _ in staged:
                runtime = self._runtimes.get(account.account_id)
                if runtime is None:
                    runtime = AccountRuntime(
                        account,
                        journal=self._journal,
                        clock=self._clock,
                        lock=self._lock,
                        snapshot=self._snapshot_locked,
                        resolution_ledger_limit=self._resolution_ledger_limit,
                        resolution_retention_ms=self._resolution_retention_ms,
                    )
                    dormant = self._dormant_account_states.pop(account.account_id, None)
                    if dormant is not None:
                        runtime._restore_snapshot_state(dormant)
                    self._runtimes[account.account_id] = runtime
                else:
                    runtime.update_account(account)
            for account in omitted:
                self._runtimes[account.account_id].update_account(account)
            if omitted:
                self._journal.force_checkpoint(self._snapshot_locked())
            elif self._journal.checkpoint_due:
                self._journal.maybe_checkpoint(self._snapshot_locked())

    def checkpoint(self) -> None:
        with self._lock:
            self._journal.force_checkpoint(self._snapshot_locked())

    def _remember_account_without_journal(self, account: ProviderAccount) -> None:
        binding = self._binding_digest(account)
        if binding:
            self._binding_history[binding] = account.account_id
            self._account_binding[account.account_id] = binding
        self._credential_epoch_highwater[account.account_id] = account.credential_epoch or 0
        self._credential_attestation_digest[account.account_id] = self._attestation_digest(account)

    def _binding_digest(self, account: ProviderAccount) -> str:
        if not all((account.connection, account.provider_token, account.binding_ref)):
            return ""
        material = canonical_json_bytes(
            [account.connection, account.provider_token, account.binding_ref]
        )
        return "sha256:" + hashlib.sha256(material).hexdigest()

    def _attestation_digest(self, account: ProviderAccount) -> str:
        material = account.credential_fingerprint or ""
        return "sha256:" + hashlib.sha256(material.encode()).hexdigest()

    def _snapshot_locked(self) -> dict[str, JsonValue]:
        account_states: dict[str, JsonValue] = {
            str(account_id): cast(JsonValue, dict(state))
            for account_id, state in self._dormant_account_states.items()
        }
        account_states.update(
            {
                str(account_id): runtime._snapshot_state()
                for account_id, runtime in self._runtimes.items()
            }
        )
        return {
            "accounts": dict(sorted(account_states.items())),
            "binding_history": {
                digest: str(account_id)
                for digest, account_id in sorted(self._binding_history.items())
            },
            "account_binding": {
                str(account_id): digest
                for account_id, digest in sorted(
                    self._account_binding.items(), key=lambda item: str(item[0])
                )
            },
            "credential_epoch_highwater": {
                str(account_id): epoch
                for account_id, epoch in sorted(
                    self._credential_epoch_highwater.items(), key=lambda item: str(item[0])
                )
            },
            "credential_attestation_digest": {
                str(account_id): digest
                for account_id, digest in sorted(
                    self._credential_attestation_digest.items(), key=lambda item: str(item[0])
                )
            },
        }

    def _restore_checkpoint(self, snapshot: Mapping[str, JsonValue]) -> None:
        accounts = cast(Mapping[str, Mapping[str, object]], snapshot["accounts"])
        for account_id_text, state in accounts.items():
            account_id = AccountId(account_id_text)
            runtime = self._runtimes.get(account_id)
            if runtime is None:
                self._dormant_account_states[account_id] = state
            else:
                runtime._restore_snapshot_state(state)
        self._binding_history = {
            str(digest): AccountId(str(account_id))
            for digest, account_id in cast(
                Mapping[str, object], snapshot["binding_history"]
            ).items()
        }
        self._account_binding = {
            AccountId(str(account_id)): str(digest)
            for account_id, digest in cast(
                Mapping[str, object], snapshot["account_binding"]
            ).items()
        }
        self._credential_epoch_highwater = {
            AccountId(str(account_id)): _require_int(epoch)
            for account_id, epoch in cast(
                Mapping[str, object], snapshot["credential_epoch_highwater"]
            ).items()
        }
        self._credential_attestation_digest = {
            AccountId(str(account_id)): str(digest)
            for account_id, digest in cast(
                Mapping[str, object], snapshot["credential_attestation_digest"]
            ).items()
        }

    def _apply_record(self, record: JournalRecord) -> None:
        payload = record.payload
        if record.kind == "account_registered":
            account_id = AccountId(cast(str, payload["account_id"]))
            binding_digest = cast(str, payload["binding_digest"])
            owner = self._binding_history.get(binding_digest)
            if owner is not None and owner != account_id:
                raise AccountBindingConflict("durable binding history conflict")
            self._binding_history[binding_digest] = account_id
            self._account_binding[account_id] = binding_digest
            epoch = cast(int, payload["credential_epoch"])
            attestation = cast(str, payload["credential_attestation_digest"])
            highwater = self._credential_epoch_highwater.get(account_id, 0)
            previous = self._credential_attestation_digest.get(account_id)
            if epoch < highwater or (
                epoch == highwater and previous is not None and previous != attestation
            ):
                raise CredentialAttestationRollback("durable credential attestation conflict")
            self._credential_epoch_highwater[account_id] = epoch
            self._credential_attestation_digest[account_id] = attestation
            return
        account_id = self._record_account_id(record)
        runtime = self._runtimes[account_id]
        if record.kind == "attempt_reserved":
            attempt_id = cast(str, payload["attempt_id"])
            reset_at = cast(int, payload["monthly_reset_at_ms"])
            if runtime._monthly_reset_at_ms != reset_at:
                runtime._monthly_reset_at_ms = reset_at
                runtime._monthly_used = 0
            runtime._starts[attempt_id] = _Start(
                attempt_id,
                cast(int, payload["started_at_ms"]),
                cast(int, payload["reserved_tokens"]),
            )
            runtime._attempts[attempt_id] = _Attempt(
                attempt_id=attempt_id,
                request_id=cast(str, payload["request_id"]),
                leg_id=cast(str, payload["leg_id"]),
                deployment_generation_id=cast(str, payload["deployment_generation_id"]),
                installation_id=cast(str, payload["installation_id"]),
                bundle_generation=cast(int, payload["bundle_generation"]),
                bundle_hash=cast(str, payload["bundle_hash"]),
                fence_token=cast(int, payload["fence_token"]),
                boot_id=cast(str, payload["boot_id"]),
                deadline_at_ms=cast(int, payload["deadline_at_ms"]),
                profile_digest=cast(str, payload["profile_digest"]),
                circuit_epoch=cast(int, payload["circuit_epoch"]),
                account_circuit_epoch=cast(int, payload["account_circuit_epoch"]),
                circuit_probe_id=cast(str | None, payload["circuit_probe_id"]),
                account_circuit_probe_id=cast(str | None, payload["account_circuit_probe_id"]),
                monthly_reset_at_ms=reset_at,
                started_at_ms=cast(int, payload["started_at_ms"]),
                dispatched_at_ms=None,
                dispatch_receipt=None,
                reserved_tokens=cast(int, payload["reserved_tokens"]),
                quota_units=cast(int, payload["quota_units"]),
                state="active",
            )
            for probe_id in (
                cast(str | None, payload["account_circuit_probe_id"]),
                cast(str | None, payload["circuit_probe_id"]),
            ):
                if probe_id is not None:
                    runtime._consumed_circuit_probes[probe_id] = attempt_id
            runtime._monthly_used += cast(int, payload["quota_units"])
        elif record.kind == "attempt_dispatched":
            attempt_id = cast(str, payload["attempt_id"])
            attempt = runtime._attempts.get(attempt_id)
            if attempt is None:
                raise InvalidLeaseTransition("dispatch has no durable reservation")
            dispatched_at_ms = cast(int, payload["dispatched_at_ms"])
            receipt = JournalReceipt(record.sequence, record.digest)
            if attempt.dispatch_receipt is not None:
                if (
                    attempt.dispatched_at_ms != dispatched_at_ms
                    or attempt.dispatch_receipt != receipt
                ):
                    raise InvalidLeaseTransition("conflicting durable dispatch")
                return
            attempt.dispatched_at_ms = dispatched_at_ms
            attempt.dispatch_receipt = receipt
        elif record.kind == "attempt_resolved":
            attempt_id = cast(str, payload["attempt_id"])
            resolution_digest = cast(str, payload["resolution_digest"])
            resolved = runtime._resolutions.get(attempt_id)
            if resolved is not None:
                if resolved.resolution_digest != resolution_digest:
                    raise InvalidLeaseTransition("conflicting durable resolution")
                return
            attempt = runtime._attempts.get(attempt_id)
            if attempt is None:
                raise InvalidLeaseTransition("resolution has no durable reservation")
            actual_starts = cast(int | None, payload["actual_starts"])
            actual_tokens = cast(int | None, payload["actual_token_units"])
            actual_quota = cast(int | None, payload["actual_quota_units"])
            if actual_tokens is not None and actual_tokens > attempt.reserved_tokens:
                raise InvalidLeaseTransition("durable token usage exceeds reservation")
            if actual_quota is not None and actual_quota > attempt.quota_units:
                raise InvalidLeaseTransition("durable quota usage exceeds reservation")
            start = runtime._starts[attempt_id]
            if actual_starts == 0:
                start.counted_start = False
            if actual_tokens is not None:
                start.tokens = actual_tokens
            if actual_quota is not None:
                runtime._monthly_used += actual_quota - attempt.quota_units
                runtime._monthly_used = max(0, runtime._monthly_used)
            if cast(bool, payload["release_capacity"]):
                del runtime._attempts[attempt_id]
            else:
                attempt.state = "uncertain"
            runtime._resolutions[attempt_id] = _ResolvedAttempt(
                resolution_digest,
                JournalReceipt(record.sequence, record.digest),
                cast(int, payload["resolved_at_ms"]),
                runtime._dispatch(attempt),
            )
        elif record.kind == "attempt_uncertain":
            attempt = runtime._attempts.get(cast(str, payload["attempt_id"]))
            if attempt is not None:
                attempt.state = "uncertain"
        elif record.kind == "circuit_updated":
            value = CircuitValue(
                state=CircuitState(cast(str, payload["state"])),
                cause=(
                    None if payload["cause"] is None else CircuitCause(cast(str, payload["cause"]))
                ),
                epoch=cast(int, payload["epoch"]),
                opened_at_ms=cast(int, payload["opened_at_ms"]),
                retry_at_ms=cast(int, payload["retry_at_ms"]),
                backoff_step=cast(int, payload["backoff_step"]),
                evidence_digest=cast(str | None, payload["evidence_digest"]),
                probe_id=cast(str | None, payload["probe_id"]),
            )
            if payload["scope"] == CircuitScope.ACCOUNT.value:
                account_previous = runtime._account_circuit
                if account_previous.probe_id is not None:
                    runtime._consumed_circuit_probes.pop(account_previous.probe_id, None)
                runtime._account_circuit = value
            else:
                generation = DeploymentGenerationId(cast(str, payload["deployment_generation_id"]))
                deployment_previous = runtime._circuits.get(generation)
                if deployment_previous is not None and deployment_previous.probe_id is not None:
                    runtime._consumed_circuit_probes.pop(deployment_previous.probe_id, None)
                runtime._circuits[generation] = value
        elif record.kind == "authoritative_active_count":
            runtime._apply_authoritative_active_count(cast(int, payload["active_count"]))
        elif record.kind == "recovery_probe_started":
            runtime._recovery_probe_id = cast(str, payload["probe_id"])
        elif record.kind == "recovery_probe_finished":
            runtime._apply_recovery_probe_finished(
                ProbeClassification(cast(str, payload["classification"]))
            )

    def _record_account_id(self, record: JournalRecord) -> AccountId:
        payload = record.payload
        if "account_id" in payload:
            return AccountId(cast(str, payload["account_id"]))
        attempt_id = cast(str, payload["attempt_id"])
        for account_id, runtime in self._runtimes.items():
            if attempt_id in runtime._attempts or attempt_id in runtime._resolutions:
                return account_id
        raise ValueError("terminal record has no matching reservation")

    def _mark_recovered_attempts_uncertain(self) -> None:
        if self._journal.recovery.last_sequence == 0:
            return
        for runtime in self._runtimes.values():
            for attempt in tuple(runtime._attempts.values()):
                if attempt.state != "active":
                    continue
                self._journal.mark_attempt_uncertain(
                    attempt_id=attempt.attempt_id,
                    reason=AmbiguousReason.CRASH_RECOVERY.value,
                )
                attempt.state = "uncertain"
            if runtime._recovery_probe_id is not None:
                probe_id = runtime._recovery_probe_id
                self._journal.record_recovery_probe_finished(
                    account_id=str(runtime.account.account_id),
                    probe_id=probe_id,
                    classification=ProbeClassification.INCONCLUSIVE.value,
                )
                runtime._apply_recovery_probe_finished(ProbeClassification.INCONCLUSIVE)
            account_circuit = runtime._account_circuit
            if account_circuit.state is CircuitState.HALF_OPEN:
                runtime.compare_and_swap_account_circuit(
                    account_circuit,
                    CircuitValue(
                        state=CircuitState.OPEN,
                        cause=account_circuit.cause,
                        epoch=account_circuit.epoch + 1,
                        opened_at_ms=account_circuit.opened_at_ms,
                        retry_at_ms=runtime._now_ms(),
                        backoff_step=min(64, account_circuit.backoff_step + 1),
                        evidence_digest=account_circuit.evidence_digest,
                        probe_id=None,
                    ),
                )
            for generation, value in tuple(runtime._circuits.items()):
                if value.state is not CircuitState.HALF_OPEN:
                    continue
                runtime.compare_and_swap_circuit(
                    generation,
                    value,
                    CircuitValue(
                        state=CircuitState.OPEN,
                        cause=value.cause,
                        epoch=value.epoch + 1,
                        opened_at_ms=value.opened_at_ms,
                        retry_at_ms=runtime._now_ms(),
                        backoff_step=min(64, value.backoff_step + 1),
                        evidence_digest=value.evidence_digest,
                        probe_id=None,
                    ),
                )
        if self._journal.checkpoint_due:
            self._journal.maybe_checkpoint(self._snapshot_locked())
