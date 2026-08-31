"""Closed route eligibility, provider-failure classification, circuits, and send caps."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmmaxxing.core.canonical import bundle_hash, canonical_bundle_bytes, canonical_json_bytes
from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    DeploymentGenerationId,
    ProbeToken,
    RequestId,
    RouteLegId,
)
from llmmaxxing.core.models import (
    AuthorizedLeg,
    PolicyBundleV1,
    RequestAuthorizationCeiling,
    RequestProfile,
)
from llmmaxxing.core.reasons import (
    DispatchCause,
    FailureCause,
    FailureScope,
    RouteTrigger,
)
from llmmaxxing.core.state_machines import AccountState, KeyLifecycleState
from llmmaxxing.gateway.auth import AuthenticatedClient
from llmmaxxing.gateway.journal import JournalStatus
from llmmaxxing.gateway.runtime_state import (
    CircuitCause,
    CircuitState,
    CircuitValue,
    OperationalRuntimeView,
    RuntimeState,
)

_CAUSE_TRIGGER = {
    DispatchCause.PRIMARY: RouteTrigger.PRIMARY,
    DispatchCause.CAPACITY: RouteTrigger.CAPACITY_SPILL,
    DispatchCause.FAILURE: RouteTrigger.FAILURE_FALLBACK,
    DispatchCause.QUOTA: RouteTrigger.QUOTA_FALLBACK,
    DispatchCause.MANUAL_EMERGENCY: RouteTrigger.MANUAL_EMERGENCY,
}
_FAILURE_DISPATCH = {
    FailureCause.CAPACITY: DispatchCause.CAPACITY,
    FailureCause.TRANSIENT_FAILURE: DispatchCause.FAILURE,
    FailureCause.QUOTA: DispatchCause.QUOTA,
}
_CIRCUIT_CAUSE = {
    FailureCause.CAPACITY: CircuitCause.CAPACITY,
    FailureCause.TRANSIENT_FAILURE: CircuitCause.TRANSIENT_FAILURE,
    FailureCause.QUOTA: CircuitCause.QUOTA,
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class EmergencyActivation(_Frozen):
    """Signed contraction-safe manual leg window bound to one dispatcher fence."""

    leg_ids: tuple[RouteLegId, ...] = Field(min_length=1)
    dispatcher_fence: int = Field(ge=1)
    activated_at_ms: int = Field(ge=1)
    expires_at_ms: int = Field(ge=1)

    @model_validator(mode="after")
    def _bounded_unique_window(self) -> Self:
        if len(set(self.leg_ids)) != len(self.leg_ids):
            raise ValueError("duplicate emergency leg")
        if not self.activated_at_ms < self.expires_at_ms:
            raise ValueError("emergency activation expiry must follow activation")
        if self.expires_at_ms - self.activated_at_ms > 3_600_000:
            raise ValueError("emergency activation exceeds one hour")
        return self

    def permits(
        self,
        leg_id: RouteLegId,
        *,
        now_ms: int,
        deadline_at_ms: int,
        dispatcher_fence: int,
    ) -> bool:
        return (
            leg_id in self.leg_ids
            and dispatcher_fence == self.dispatcher_fence
            and self.activated_at_ms <= now_ms < self.expires_at_ms
            and deadline_at_ms <= self.expires_at_ms
        )


@dataclass(frozen=True, slots=True)
class RoutingContext:
    now_ms: int
    deadline_at_ms: int
    dispatcher_fence: int
    emergency: EmergencyActivation | None = None

    def __post_init__(self) -> None:
        if self.now_ms < 1 or self.deadline_at_ms < 1 or self.dispatcher_fence < 1:
            raise ValueError("routing context values must be positive")


class FailureObservation(_Frozen):
    """One bounded normalized provider response; HTTP status alone has no meaning."""

    status_code: int = Field(ge=100, le=599)
    error_code: str | None = Field(default=None, max_length=160)
    message: str = Field(max_length=4096)
    pre_response_bytes: bool

    @property
    def evidence_digest(self) -> str:
        payload = self.model_dump(mode="json", round_trip=True, warnings=False)
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class FailureRule:
    cause: FailureCause
    scope: FailureScope
    status_codes: frozenset[int]
    error_codes: frozenset[str] = frozenset()
    message_contains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cause is FailureCause.UNKNOWN or self.scope is FailureScope.UNKNOWN:
            raise ValueError("classifier rules must prove a known cause and scope")
        if not self.status_codes or any(
            status < 100 or status > 599 for status in self.status_codes
        ):
            raise ValueError("classifier rule requires bounded HTTP statuses")
        if any(not value or len(value) > 160 for value in self.error_codes):
            raise ValueError("classifier error code is empty or oversized")
        if any(not value or len(value) > 1024 for value in self.message_contains):
            raise ValueError("classifier message evidence is empty or oversized")

    def matches(self, observation: FailureObservation) -> bool:
        if observation.status_code not in self.status_codes:
            return False
        if self.error_codes and observation.error_code not in self.error_codes:
            return False
        return all(fragment in observation.message for fragment in self.message_contains)


@dataclass(frozen=True, slots=True)
class FailureClassification:
    cause: FailureCause
    scope: FailureScope
    evidence_digest: str
    pre_response_bytes: bool

    def __post_init__(self) -> None:
        if not self.evidence_digest.startswith("sha256:") or len(self.evidence_digest) != 71:
            raise ValueError("failure classification requires a SHA-256 evidence digest")
        if (self.cause is FailureCause.UNKNOWN) != (self.scope is FailureScope.UNKNOWN):
            raise ValueError("unknown failure cause and scope must agree")

    @property
    def dispatch_cause(self) -> DispatchCause | None:
        if not self.pre_response_bytes:
            return None
        return _FAILURE_DISPATCH.get(self.cause)


class FailureClassifier:
    """One deterministic classifier; zero or overlapping matches are UNKNOWN."""

    def __init__(self, rules: Iterable[FailureRule]) -> None:
        self._rules = tuple(rules)
        if not self._rules:
            raise ValueError("failure classifier requires at least one evidence rule")

    def classify(self, observation: FailureObservation) -> FailureClassification:
        matches = tuple(rule for rule in self._rules if rule.matches(observation))
        if len(matches) != 1:
            cause = FailureCause.UNKNOWN
            scope = FailureScope.UNKNOWN
        else:
            cause = matches[0].cause
            scope = matches[0].scope
        return FailureClassification(
            cause=cause,
            scope=scope,
            evidence_digest=observation.evidence_digest,
            pre_response_bytes=observation.pre_response_bytes,
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    authorized_leg: AuthorizedLeg
    cause: DispatchCause
    account_circuit: CircuitValue
    deployment_circuit: CircuitValue

    @property
    def leg_id(self) -> RouteLegId:
        return self.authorized_leg.leg_id

    @property
    def account_id(self) -> AccountId:
        return self.authorized_leg.account_id

    @property
    def generation_id(self) -> DeploymentGenerationId:
        return self.authorized_leg.generation_id


@dataclass(frozen=True, slots=True)
class DispatchedAttempt:
    attempt_id: AttemptId
    leg_id: RouteLegId
    generation_id: DeploymentGenerationId
    circuit_epoch: int
    account_circuit_epoch: int
    probe_ids: tuple[str, ...]


class AttemptBudget:
    """Request-local view of the durable sends recorded by Task 6."""

    __slots__ = ("request_id", "_attempts", "_response_started")

    def __init__(self, request_id: RequestId) -> None:
        self.request_id = request_id
        self._attempts: list[DispatchedAttempt] = []
        self._response_started = False

    @classmethod
    def from_runtime(
        cls,
        request_id: RequestId,
        view: OperationalRuntimeView,
    ) -> AttemptBudget:
        budget = cls(request_id)
        budget.sync_runtime(view)
        return budget

    def sync_runtime(self, view: OperationalRuntimeView) -> None:
        known = {attempt.attempt_id for attempt in self._attempts}
        for dispatch in view.dispatches:
            if dispatch.request_id != self.request_id or dispatch.attempt_id in known:
                continue
            self._attempts.append(
                DispatchedAttempt(
                    attempt_id=dispatch.attempt_id,
                    leg_id=dispatch.leg_id,
                    generation_id=dispatch.deployment_generation_id,
                    circuit_epoch=dispatch.circuit_epoch,
                    account_circuit_epoch=dispatch.account_circuit_epoch,
                    probe_ids=tuple(
                        probe_id
                        for probe_id in (
                            dispatch.account_circuit_probe_id,
                            dispatch.circuit_probe_id,
                        )
                        if probe_id is not None
                    ),
                )
            )
            known.add(dispatch.attempt_id)
        self._attempts.sort(key=lambda attempt: str(attempt.attempt_id))

    @property
    def attempts(self) -> tuple[DispatchedAttempt, ...]:
        return tuple(self._attempts)

    @property
    def send_closed(self) -> bool:
        return self._response_started or len(self._attempts) >= 3

    def mark_response_started(self) -> None:
        self._response_started = True

    @staticmethod
    def _probe_ids(candidate: Candidate) -> tuple[str, ...]:
        return tuple(
            probe_id
            for circuit in (candidate.account_circuit, candidate.deployment_circuit)
            if circuit.state is CircuitState.HALF_OPEN
            if (probe_id := circuit.probe_id) is not None
        )

    def can_send(self, candidate: Candidate) -> bool:
        if self.send_closed:
            return False
        if any(
            attempt.leg_id == candidate.leg_id
            and attempt.circuit_epoch == candidate.deployment_circuit.epoch
            and attempt.account_circuit_epoch == candidate.account_circuit.epoch
            for attempt in self._attempts
        ):
            return False
        generation_seen = any(
            attempt.generation_id == candidate.generation_id for attempt in self._attempts
        )
        if not generation_seen:
            return True
        probe_ids = self._probe_ids(candidate)
        return (
            candidate.cause is DispatchCause.CAPACITY
            and bool(probe_ids)
            and all(
                probe_id not in attempt.probe_ids
                for probe_id in probe_ids
                for attempt in self._attempts
            )
        )

    def record_send(self, candidate: Candidate, attempt_id: AttemptId) -> DispatchedAttempt:
        if not self.can_send(candidate):
            raise ValueError("provider send exceeds the request attempt budget")
        attempt = DispatchedAttempt(
            attempt_id=attempt_id,
            leg_id=candidate.leg_id,
            generation_id=candidate.generation_id,
            circuit_epoch=candidate.deployment_circuit.epoch,
            account_circuit_epoch=candidate.account_circuit.epoch,
            probe_ids=self._probe_ids(candidate),
        )
        self._attempts.append(attempt)
        return attempt


class RouteEngine:
    """Current complete authorization plus immutable admission-ceiling intersection."""

    def __init__(self, bundle: PolicyBundleV1) -> None:
        self.activate(bundle)

    def activate(self, bundle: PolicyBundleV1) -> None:
        validated = PolicyBundleV1.model_validate(bundle.model_dump(mode="python"))
        self.bundle = validated
        self.bundle_hash = bundle_hash(canonical_bundle_bytes(validated))
        self._keys = {key.key_id: key for key in validated.keys}
        self._policies = {policy.policy_id: policy for policy in validated.policies}
        self._groups = {group.route_group_id: group for group in validated.route_groups}
        self._accounts = {account.account_id: account for account in validated.accounts}

    def authorize(
        self,
        client: AuthenticatedClient,
        profile: RequestProfile,
    ) -> RequestAuthorizationCeiling:
        key = self._keys.get(client.key_id)
        if (
            key is None
            or key.state is not KeyLifecycleState.ENABLED
            or key.policy_id != client.policy_id
        ):
            raise ValueError("client key has no current complete authorization")
        policy = self._policies.get(client.policy_id)
        if policy is None or profile.route_group_id not in policy.route_group_ids:
            raise ValueError("route group is not authorized by the current policy")
        group = self._groups.get(profile.route_group_id)
        if group is None:
            raise ValueError("route group is absent from the current bundle")
        group_leg_ids = frozenset(leg.leg_id for leg in group.legs)
        authorized_legs = tuple(
            leg for leg in policy.authorized_legs if leg.leg_id in group_leg_ids
        )
        if not authorized_legs:
            raise ValueError("route group has no current authorized legs")
        return RequestAuthorizationCeiling(
            key_id=client.key_id,
            credential_generation=client.accepted_credential_generation,
            policy_id=policy.policy_id,
            bundle_generation=self.bundle.generation,
            bundle_hash=self.bundle_hash,
            route_group_id=profile.route_group_id,
            authorized_legs=authorized_legs,
            queue_tier=policy.queue_tier,
            queue_weight=policy.queue_weight,
            max_concurrency=policy.max_concurrency,
            max_waiters=policy.max_waiters,
            deadline_ms=policy.deadline_ms,
        )

    def quota_units(self, account_id: AccountId) -> int:
        try:
            return self._accounts[account_id].quota_units_per_attempt
        except KeyError:
            raise ValueError("candidate account is absent from the current bundle") from None

    @staticmethod
    def _circuits(
        leg: AuthorizedLeg,
        view: OperationalRuntimeView,
    ) -> tuple[CircuitValue, CircuitValue]:
        account = next((item for item in view.accounts if item.account_id == leg.account_id), None)
        account_circuit = CircuitValue.closed() if account is None else account.account_circuit
        deployment_circuit = next(
            (
                item.value
                for item in view.circuits
                if item.account_id == leg.account_id
                and item.deployment_generation_id == leg.generation_id
            ),
            CircuitValue.closed(),
        )
        return account_circuit, deployment_circuit

    def candidate(
        self,
        leg: AuthorizedLeg,
        cause: DispatchCause,
        view: OperationalRuntimeView | None = None,
    ) -> Candidate:
        circuits = (
            (CircuitValue.closed(), CircuitValue.closed())
            if view is None
            else self._circuits(leg, view)
        )
        return Candidate(leg, cause, *circuits)

    @staticmethod
    def _capabilities_allow(leg: AuthorizedLeg, profile: RequestProfile) -> bool:
        capabilities = leg.capabilities
        return (
            profile.endpoint in capabilities.endpoints
            and profile.modality in capabilities.modalities
            and (
                profile.input_tokens_max + profile.output_tokens_max + profile.reasoning_tokens_max
                <= capabilities.context_tokens
            )
            and (profile.tools_count == 0 or capabilities.tools)
            and (not profile.forced_tool_required or capabilities.forced_tool)
            and (not profile.response_schema_present or capabilities.response_schema)
        )

    @staticmethod
    def _operationally_available(leg: AuthorizedLeg, view: OperationalRuntimeView) -> bool:
        if view.recovery_state is not JournalStatus.HEALTHY:
            return False
        account = next((item for item in view.accounts if item.account_id == leg.account_id), None)
        if (
            account is None
            or account.state is not AccountState.ACTIVE
            or account.recovery_probe_in_flight
            or account.active_attempts >= account.parallel_limit
            or account.account_circuit.state is CircuitState.OPEN
        ):
            return False
        deployment = next(
            (
                item.value
                for item in view.circuits
                if item.account_id == leg.account_id
                and item.deployment_generation_id == leg.generation_id
            ),
            CircuitValue.closed(),
        )
        return deployment.state is not CircuitState.OPEN

    def eligible(
        self,
        leg: AuthorizedLeg,
        profile: RequestProfile,
        cause: DispatchCause,
        view: OperationalRuntimeView,
        context: RoutingContext,
    ) -> bool:
        if leg.capabilities.shadow or RouteTrigger.SHADOW in leg.allowed_triggers:
            return False
        if _CAUSE_TRIGGER[cause] not in leg.allowed_triggers:
            return False
        if not self._capabilities_allow(leg, profile):
            return False
        if cause is DispatchCause.MANUAL_EMERGENCY and (
            context.emergency is None
            or not context.emergency.permits(
                leg.leg_id,
                now_ms=context.now_ms,
                deadline_at_ms=context.deadline_at_ms,
                dispatcher_fence=context.dispatcher_fence,
            )
        ):
            return False
        return context.now_ms < context.deadline_at_ms and self._operationally_available(leg, view)

    def primary_capacity_unavailable(
        self,
        authorization: RequestAuthorizationCeiling,
        profile: RequestProfile,
        view: OperationalRuntimeView,
    ) -> bool:
        primaries = tuple(
            leg
            for leg in authorization.authorized_legs
            if RouteTrigger.PRIMARY in leg.allowed_triggers
            and not leg.capabilities.shadow
            and self._capabilities_allow(leg, profile)
        )
        if not primaries:
            return False
        for leg in primaries:
            account = next(
                (value for value in view.accounts if value.account_id == leg.account_id),
                None,
            )
            if account is None or account.state is not AccountState.ACTIVE:
                return False
            account_capacity = account.active_attempts >= account.parallel_limit
            account_circuit = (
                account.account_circuit.state is CircuitState.OPEN
                and account.account_circuit.cause is CircuitCause.CAPACITY
            )
            deployment = next(
                (
                    item.value
                    for item in view.circuits
                    if item.account_id == leg.account_id
                    and item.deployment_generation_id == leg.generation_id
                ),
                CircuitValue.closed(),
            )
            deployment_capacity = (
                deployment.state is CircuitState.OPEN and deployment.cause is CircuitCause.CAPACITY
            )
            if not (account_capacity or account_circuit or deployment_capacity):
                return False
        return True

    def filter(
        self,
        authorization: RequestAuthorizationCeiling,
        profile: RequestProfile,
        cause: DispatchCause,
        view: OperationalRuntimeView,
        context: RoutingContext,
        *,
        excluded_accounts: frozenset[AccountId] = frozenset(),
    ) -> tuple[Candidate, ...]:
        if profile.route_group_id != authorization.route_group_id:
            return ()
        return tuple(
            self.candidate(leg, cause, view)
            for leg in authorization.authorized_legs
            if leg.account_id not in excluded_accounts
            and self.eligible(leg, profile, cause, view, context)
        )

    def select(
        self,
        authorization: RequestAuthorizationCeiling,
        profile: RequestProfile,
        cause: DispatchCause,
        view: OperationalRuntimeView,
        context: RoutingContext,
        *,
        excluded_accounts: frozenset[AccountId] = frozenset(),
    ) -> Candidate | None:
        return next(
            iter(
                self.filter(
                    authorization,
                    profile,
                    cause,
                    view,
                    context,
                    excluded_accounts=excluded_accounts,
                )
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class CircuitProbe:
    scope: FailureScope
    account_id: AccountId
    generation_id: DeploymentGenerationId
    value: CircuitValue

    def __post_init__(self) -> None:
        if self.scope is FailureScope.UNKNOWN:
            raise ValueError("circuit probe scope must be known")
        if self.value.state is not CircuitState.HALF_OPEN:
            raise ValueError("circuit probe requires a half-open CAS value")


class CircuitController:
    """Task 7 policy transitions applied only through Task 6's durable CAS."""

    def __init__(
        self,
        runtime: RuntimeState,
        *,
        probe_factory: Callable[[], ProbeToken] = ProbeToken.new,
    ) -> None:
        self._runtime = runtime
        self._probe_factory = probe_factory

    @staticmethod
    def _backoff_ms(step: int, evidence_digest: str) -> int:
        base = (15_000, 30_000, 60_000)[min(step, 2)]
        seed = hashlib.sha256(f"{evidence_digest}:{step}".encode()).digest()
        return base + int.from_bytes(seed[:2], "big") % (base // 5 + 1)

    @staticmethod
    def _replacement(
        current: CircuitValue,
        classification: FailureClassification,
        *,
        now_ms: int,
        step: int,
    ) -> CircuitValue:
        cause = _CIRCUIT_CAUSE[classification.cause]
        return CircuitValue(
            state=CircuitState.OPEN,
            cause=cause,
            epoch=current.epoch + 1,
            opened_at_ms=now_ms,
            retry_at_ms=now_ms
            + CircuitController._backoff_ms(step, classification.evidence_digest),
            backoff_step=step,
            evidence_digest=classification.evidence_digest,
            probe_id=None,
        )

    def open(
        self,
        account_id: AccountId,
        generation_id: DeploymentGenerationId,
        classification: FailureClassification,
        *,
        now_ms: int,
    ) -> CircuitValue | None:
        if classification.cause is FailureCause.UNKNOWN:
            return None
        runtime = self._runtime.account_runtime(account_id)
        if classification.scope is FailureScope.ACCOUNT:
            current = runtime.account_circuit_value()
            if current.state is CircuitState.OPEN:
                return current
            step = min(current.backoff_step + (current.state is CircuitState.HALF_OPEN), 2)
            replacement = self._replacement(current, classification, now_ms=now_ms, step=step)
            return (
                replacement
                if runtime.compare_and_swap_account_circuit(current, replacement)
                else None
            )
        if classification.scope is FailureScope.DEPLOYMENT:
            current = runtime.circuit_value(generation_id)
            if current.state is CircuitState.OPEN:
                return current
            step = min(current.backoff_step + (current.state is CircuitState.HALF_OPEN), 2)
            replacement = self._replacement(current, classification, now_ms=now_ms, step=step)
            return (
                replacement
                if runtime.compare_and_swap_circuit(generation_id, current, replacement)
                else None
            )
        return None

    def begin_probe(
        self,
        account_id: AccountId,
        generation_id: DeploymentGenerationId,
        *,
        now_ms: int,
    ) -> CircuitProbe | None:
        runtime = self._runtime.account_runtime(account_id)
        account = runtime.account_circuit_value()
        deployment = runtime.circuit_value(generation_id)
        scope = (
            FailureScope.ACCOUNT if account.state is CircuitState.OPEN else FailureScope.DEPLOYMENT
        )
        current = account if scope is FailureScope.ACCOUNT else deployment
        if current.state is not CircuitState.OPEN or now_ms < current.retry_at_ms:
            return None
        replacement = CircuitValue(
            state=CircuitState.HALF_OPEN,
            cause=current.cause,
            epoch=current.epoch,
            opened_at_ms=current.opened_at_ms,
            retry_at_ms=0,
            backoff_step=current.backoff_step,
            evidence_digest=current.evidence_digest,
            probe_id=str(self._probe_factory()),
        )
        swapped = (
            runtime.compare_and_swap_account_circuit(current, replacement)
            if scope is FailureScope.ACCOUNT
            else runtime.compare_and_swap_circuit(generation_id, current, replacement)
        )
        return CircuitProbe(scope, account_id, generation_id, replacement) if swapped else None

    def probe_succeeded(self, probe: CircuitProbe) -> bool:
        runtime = self._runtime.account_runtime(probe.account_id)
        replacement = CircuitValue.closed(epoch=probe.value.epoch)
        if probe.scope is FailureScope.ACCOUNT:
            return runtime.compare_and_swap_account_circuit(probe.value, replacement)
        return runtime.compare_and_swap_circuit(probe.generation_id, probe.value, replacement)

    def probe_failed(
        self,
        probe: CircuitProbe,
        classification: FailureClassification,
        *,
        now_ms: int,
    ) -> CircuitValue | None:
        if classification.cause is FailureCause.UNKNOWN or classification.scope is not probe.scope:
            return None
        runtime = self._runtime.account_runtime(probe.account_id)
        replacement = self._replacement(
            probe.value,
            classification,
            now_ms=now_ms,
            step=min(probe.value.backoff_step + 1, 2),
        )
        swapped = (
            runtime.compare_and_swap_account_circuit(probe.value, replacement)
            if probe.scope is FailureScope.ACCOUNT
            else runtime.compare_and_swap_circuit(probe.generation_id, probe.value, replacement)
        )
        return replacement if swapped else None
