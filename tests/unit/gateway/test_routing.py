"""Deterministic authorization, route-trigger, circuit, and attempt-budget contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmmaxxing.core.canonical import bundle_hash, canonical_bundle_bytes
from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    BundleHash,
    DeploymentGenerationId,
    GatewayBootId,
    InstallationId,
    PolicyRevisionId,
    ProbeToken,
    RequestId,
    RouteGroupId,
    RouteLegId,
)
from llmmaxxing.core.models import (
    AuthorizedLeg,
    ClientCredentialVerifier,
    ClientKeyRecord,
    KeyPolicyRevision,
    LegCapabilities,
    PolicyBundleV1,
    ProviderAccount,
    QuotaDimension,
    RequestAuthorizationCeiling,
    RequestProfile,
    RouteGroupRevision,
    RouteLeg,
)
from llmmaxxing.core.reasons import (
    DispatchCause,
    EndpointKind,
    FailureCause,
    FailureScope,
    Modality,
    QuotaDimensionStatus,
    RequiredFeature,
    RouteStrategy,
    RouteTrigger,
    TerminalOutcome,
)
from llmmaxxing.core.state_machines import (
    AccountState,
    CredentialVerifierStatus,
    KeyLifecycleState,
)
from llmmaxxing.gateway.auth import AuthenticatedClient
from llmmaxxing.gateway.journal import AttemptJournal
from llmmaxxing.gateway.routing import (
    AttemptBudget,
    Candidate,
    CircuitController,
    EmergencyActivation,
    FailureClassifier,
    FailureObservation,
    FailureRule,
    RouteEngine,
    RoutingContext,
)
from llmmaxxing.gateway.runtime_state import (
    AttemptResolution,
    CircuitOperationalValue,
    CircuitState,
    CircuitValue,
    ReservationGranted,
    ReservationRequest,
    RuntimeIdentity,
    RuntimeState,
)

NAN = AccountId("acc_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
ARLI = AccountId("acc_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
GROUP = RouteGroupId("rg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
PRIMARY_LEG = RouteLegId("leg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SPILL_LEG = RouteLegId("leg_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
SHADOW_LEG = RouteLegId("leg_cccccccc-cccc-4ccc-8ccc-cccccccccccc")
POLICY = PolicyRevisionId("pol_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NAN_GEN = DeploymentGenerationId.from_digest("a" * 64)
ARLI_GEN = DeploymentGenerationId.from_digest("b" * 64)
SHADOW_GEN = DeploymentGenerationId.from_digest("c" * 64)
KEY_ID = "1" * 32
NOW = 1_800_000_000_000


class FakeClock:
    def __init__(self, now_ms: int = NOW) -> None:
        self.value = now_ms

    def now_ms(self) -> int:
        return self.value


class GenerationGate:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def permits(self, leg: AuthorizedLeg, backend_manifest_hash: str) -> bool:
        assert leg.generation_id
        assert len(backend_manifest_hash) == 64
        return self.allowed


def quota(value: int = 100_000) -> QuotaDimension:
    return QuotaDimension(status=QuotaDimensionStatus.KNOWN, value=value)


def account(account_id: AccountId, suffix: str, *, parallel: int = 2) -> ProviderAccount:
    return ProviderAccount(
        account_id=account_id,
        display_name=suffix,
        connection=f"litellm:{suffix}",
        provider_token=suffix,
        binding_ref=f"{suffix}-primary",
        credential_fingerprint="hcf1_" + ("a" if account_id == NAN else "b") * 64,
        credential_epoch=1,
        parallel_limit=quota(parallel),
        local_parallel_ceiling=128,
        rpm_limit=quota(60),
        rpm_window_seconds=60,
        tpm_limit=quota(1_000_000),
        tpm_window_seconds=60,
        monthly_quota_units=quota(1_000_000),
        quota_units_per_attempt=1,
        monthly_reset_at_ms=2_000_000_000_000,
        state=AccountState.ACTIVE,
    )


def capabilities(*, shadow: bool = False, context_tokens: int = 32_768) -> LegCapabilities:
    return LegCapabilities(
        endpoints=(EndpointKind.CHAT,),
        modalities=(Modality.TEXT,),
        context_tokens=context_tokens,
        tools=True,
        forced_tool=True,
        response_schema=True,
        streaming=True,
        reasoning=True,
        history_continuation=True,
        shadow=shadow,
    )


def route_leg(
    leg_id: RouteLegId,
    account_id: AccountId,
    generation_id: DeploymentGenerationId,
    order: int,
    triggers: tuple[RouteTrigger, ...],
    *,
    caps: LegCapabilities | None = None,
) -> RouteLeg:
    return RouteLeg(
        leg_id=leg_id,
        account_id=account_id,
        generation_id=generation_id,
        order=order,
        triggers=triggers,
        capabilities=caps or capabilities(),
    )


def authorized(leg: RouteLeg, *triggers: RouteTrigger) -> AuthorizedLeg:
    return AuthorizedLeg(
        leg_id=leg.leg_id,
        account_id=leg.account_id,
        generation_id=leg.generation_id,
        order=leg.order,
        allowed_triggers=triggers or leg.triggers,
        capabilities=leg.capabilities,
    )


def make_bundle(*, spill: bool = True, shadow: bool = True) -> PolicyBundleV1:
    legs = [
        route_leg(PRIMARY_LEG, NAN, NAN_GEN, 10, (RouteTrigger.PRIMARY,)),
    ]
    if spill:
        legs.append(
            route_leg(
                SPILL_LEG,
                ARLI,
                ARLI_GEN,
                20,
                (
                    RouteTrigger.CAPACITY_SPILL,
                    RouteTrigger.FAILURE_FALLBACK,
                    RouteTrigger.QUOTA_FALLBACK,
                    RouteTrigger.MANUAL_EMERGENCY,
                ),
            )
        )
    if shadow:
        legs.append(
            route_leg(
                SHADOW_LEG,
                ARLI,
                SHADOW_GEN,
                30,
                (RouteTrigger.SHADOW,),
                caps=capabilities(shadow=True),
            )
        )
    group = RouteGroupRevision(
        route_group_id=GROUP,
        name="deepseek-v4-flash",
        strategy=RouteStrategy.ORDERED_CAPACITY,
        legs=tuple(legs),
    )
    policy = KeyPolicyRevision(
        policy_id=POLICY,
        name="coding",
        route_group_ids=(GROUP,),
        authorized_legs=tuple(authorized(leg) for leg in legs),
        queue_tier=10,
        queue_weight=4,
        max_concurrency=4,
        max_waiters=16,
        deadline_ms=120_000,
    )
    key = ClientKeyRecord(
        key_id=KEY_ID,
        policy_id=POLICY,
        state=KeyLifecycleState.ENABLED,
        issued_at_s=1_780_000_000,
        expires_at_s=1_810_000_000,
        time_high_water_s=1_800_000_000,
        generation_high_water=1,
        credential_verifiers=(
            ClientCredentialVerifier(
                generation=1,
                verifier_hex="d" * 64,
                pepper_version="p1",
                not_before_s=1_780_000_000,
                not_after_s=1_810_000_000,
                status=CredentialVerifierStatus.ACTIVE,
            ),
        ),
    )
    return PolicyBundleV1(
        schema_version=1,
        generation=7,
        min_reader="1.0",
        required_features=(
            RequiredFeature.ORDERED_CAPACITY,
            RequiredFeature.WEIGHTED_FAIR_QUEUE,
        ),
        keys=(key,),
        policies=(policy,),
        accounts=(account(NAN, "nan", parallel=5), account(ARLI, "arli", parallel=2)),
        route_groups=(group,),
        backend_manifest_hash="e" * 64,
    )


def client(bundle: PolicyBundleV1) -> AuthenticatedClient:
    return AuthenticatedClient(
        key_id=KEY_ID,
        accepted_credential_generation=1,
        policy_id=POLICY,
        key_state=KeyLifecycleState.ENABLED,
        key_expires_at_s=1_810_000_000,
        applied_bundle_generation=bundle.generation,
        applied_bundle_hash=bundle_hash(canonical_bundle_bytes(bundle)),
    )


def profile(**updates: object) -> RequestProfile:
    values: dict[str, object] = {
        "route_group_id": GROUP,
        "model_alias": "deepseek-v4-flash",
        "endpoint": EndpointKind.CHAT,
        "modality": Modality.TEXT,
        "stream": False,
        "input_tokens_max": 100,
        "output_tokens_max": 100,
        "reasoning_tokens_max": 0,
        "tools_count": 0,
        "forced_tool_required": False,
        "response_schema_present": False,
        "history_turns": 0,
        "deadline_ms": 120_000,
    }
    values.update(updates)
    return RequestProfile.model_validate(values)


def context(
    *, emergency: EmergencyActivation | None = None, deadline_at_ms: int | None = None
) -> RoutingContext:
    return RoutingContext(
        now_ms=NOW,
        deadline_at_ms=deadline_at_ms or NOW + 120_000,
        dispatcher_fence=11,
        emergency=emergency,
    )


def runtime(tmp_path: Path, bundle: PolicyBundleV1) -> tuple[AttemptJournal, RuntimeState]:
    journal = AttemptJournal(tmp_path, clock=FakeClock(), create=True)
    return journal, RuntimeState(bundle.accounts, journal=journal, clock=FakeClock())


def test_authorized_leg_intersection_is_exact_and_expansions_never_enter_queue() -> None:
    bundle = make_bundle()
    ceiling = RouteEngine(bundle, GenerationGate()).authorize(client(bundle), profile())
    primary, spill, _ = ceiling.authorized_legs

    expanded = ceiling.model_copy(
        update={
            "bundle_generation": 8,
            "authorized_legs": (
                primary.model_copy(
                    update={
                        "allowed_triggers": (
                            RouteTrigger.PRIMARY,
                            RouteTrigger.CAPACITY_SPILL,
                        ),
                        "capabilities": primary.capabilities.model_copy(
                            update={"context_tokens": 1_000_000}
                        ),
                    }
                ),
                spill,
                authorized(
                    route_leg(
                        RouteLegId.new(),
                        ARLI,
                        DeploymentGenerationId.from_digest("f" * 64),
                        40,
                        (RouteTrigger.CAPACITY_SPILL,),
                    )
                ),
            ),
        }
    )
    merged = ceiling.intersection(expanded)
    assert merged.authorized_legs == ceiling.authorized_legs[:2]
    assert merged.authorized_legs[0].capabilities.context_tokens == 32_768

    contracted = ceiling.model_copy(
        update={
            "authorized_legs": (
                primary.model_copy(
                    update={
                        "allowed_triggers": (RouteTrigger.PRIMARY,),
                        "capabilities": primary.capabilities.model_copy(
                            update={"tools": False, "context_tokens": 4_096}
                        ),
                    }
                ),
                # Same leg id with a changed deployment identity is not authorization.
                spill.model_copy(
                    update={"generation_id": DeploymentGenerationId.from_digest("9" * 64)}
                ),
            )
        }
    )
    merged = ceiling.intersection(contracted)
    assert tuple(leg.leg_id for leg in merged.authorized_legs) == (PRIMARY_LEG,)
    assert not merged.authorized_legs[0].capabilities.tools
    assert merged.authorized_legs[0].capabilities.context_tokens == 4_096


def test_closed_profile_capability_semantics_reject_each_unsupported_shape(tmp_path: Path) -> None:
    bundle = make_bundle()
    engine = RouteEngine(bundle, GenerationGate())
    ceiling = engine.authorize(client(bundle), profile())
    journal, state = runtime(tmp_path, bundle)
    try:
        view = state.operational_view()
        primary = ceiling.authorized_legs[0]
        assert engine.eligible(primary, profile(), DispatchCause.PRIMARY, view, context())
        incompatible = (
            profile(endpoint=EndpointKind.RERANK),
            profile(modality=Modality.IMAGE),
            profile(input_tokens_max=32_000, output_tokens_max=1_000),
            profile(tools_count=1),
            profile(tools_count=1, forced_tool_required=True),
            profile(response_schema_present=True),
            profile(stream=True),
            profile(reasoning_tokens_max=1),
            profile(history_turns=1),
        )
        restricted = primary.model_copy(
            update={
                "capabilities": primary.capabilities.model_copy(
                    update={
                        "tools": False,
                        "forced_tool": False,
                        "response_schema": False,
                        "streaming": False,
                        "reasoning": False,
                        "history_continuation": False,
                    }
                )
            }
        )
        assert all(
            not engine.eligible(restricted, request_profile, DispatchCause.PRIMARY, view, context())
            for request_profile in incompatible
        )
    finally:
        journal.close()


def test_generation_evidence_gate_blocks_before_reservation(tmp_path: Path) -> None:
    bundle = make_bundle()
    with pytest.raises(TypeError):
        RouteEngine(bundle)  # type: ignore[call-arg]
    engine = RouteEngine(bundle, GenerationGate(allowed=False))
    ceiling = engine.authorize(client(bundle), profile())
    journal, state = runtime(tmp_path, bundle)
    try:
        assert (
            engine.filter(
                ceiling,
                profile(),
                DispatchCause.PRIMARY,
                state.operational_view(),
                context(),
            )
            == ()
        )
    finally:
        journal.close()


def test_primary_capacity_failure_and_quota_causes_never_cross_activate(tmp_path: Path) -> None:
    bundle = make_bundle()
    engine = RouteEngine(bundle, GenerationGate())
    ceiling = engine.authorize(client(bundle), profile())
    journal, state = runtime(tmp_path, bundle)
    try:
        view = state.operational_view()
        assert (
            engine.select(ceiling, profile(), DispatchCause.PRIMARY, view, context()).leg_id
            == PRIMARY_LEG
        )
        for cause in (DispatchCause.CAPACITY, DispatchCause.FAILURE, DispatchCause.QUOTA):
            assert engine.select(ceiling, profile(), cause, view, context()).leg_id == SPILL_LEG
        full_nan = replace(view.accounts[0], active_attempts=view.accounts[0].parallel_limit)
        full_view = replace(view, accounts=(full_nan, *view.accounts[1:]))
        assert (
            engine.select(ceiling, profile(), DispatchCause.PRIMARY, full_view, context()) is None
        )
        assert (
            engine.select(ceiling, profile(), DispatchCause.CAPACITY, full_view, context()).leg_id
            == SPILL_LEG
        )
        assert (
            engine.primary_blocking_cause(ceiling, profile(), full_view) is DispatchCause.CAPACITY
        )
    finally:
        journal.close()


def test_primary_fallback_requires_agreeing_open_circuit_causes(tmp_path: Path) -> None:
    bundle = make_bundle()
    engine = RouteEngine(bundle, GenerationGate())
    ceiling = engine.authorize(client(bundle), profile())
    journal, state = runtime(tmp_path, bundle)
    failure = CircuitValue(
        state=CircuitState.OPEN,
        cause="transient_failure",
        epoch=1,
        opened_at_ms=NOW,
        retry_at_ms=NOW + 15_000,
        backoff_step=0,
        evidence_digest="sha256:" + "a" * 64,
        probe_id=None,
    )
    quota_circuit = failure.model_copy(
        update={"cause": "quota", "evidence_digest": "sha256:" + "b" * 64}
    )
    try:
        view = state.operational_view()
        failed_view = replace(
            view,
            accounts=(replace(view.accounts[0], account_circuit=failure), *view.accounts[1:]),
        )
        assert (
            engine.primary_blocking_cause(ceiling, profile(), failed_view) is DispatchCause.FAILURE
        )
        mixed = replace(
            failed_view,
            circuits=(CircuitOperationalValue(NAN, NAN_GEN, quota_circuit),),
        )
        assert engine.primary_blocking_cause(ceiling, profile(), mixed) is None
    finally:
        journal.close()


def test_shadow_capability_and_trigger_are_never_serving(tmp_path: Path) -> None:
    bundle = make_bundle(spill=False, shadow=True)
    engine = RouteEngine(bundle, GenerationGate())
    ceiling = engine.authorize(client(bundle), profile())
    journal, state = runtime(tmp_path, bundle)
    try:
        view = state.operational_view()
        assert tuple(
            candidate.leg_id
            for cause in DispatchCause
            for candidate in engine.filter(ceiling, profile(), cause, view, context())
        ) == (PRIMARY_LEG,)
    finally:
        journal.close()


def test_manual_emergency_requires_leg_fence_unexpired_window_and_deadline(tmp_path: Path) -> None:
    bundle = make_bundle()
    engine = RouteEngine(bundle, GenerationGate())
    ceiling = engine.authorize(client(bundle), profile())
    journal, state = runtime(tmp_path, bundle)
    try:
        view = state.operational_view()
        valid = EmergencyActivation(
            leg_ids=(SPILL_LEG,),
            dispatcher_fence=11,
            activated_at_ms=NOW - 1_000,
            expires_at_ms=NOW + 60_000,
        )
        assert (
            engine.select(
                ceiling,
                profile(),
                DispatchCause.MANUAL_EMERGENCY,
                view,
                context(emergency=valid, deadline_at_ms=NOW + 60_000),
            ).leg_id
            == SPILL_LEG
        )
        for invalid_context in (
            replace(context(emergency=valid), dispatcher_fence=12),
            replace(context(emergency=valid), now_ms=valid.expires_at_ms),
            context(emergency=valid, deadline_at_ms=valid.expires_at_ms + 1),
            context(
                emergency=valid.model_copy(update={"leg_ids": (PRIMARY_LEG,)}),
                deadline_at_ms=NOW + 60_000,
            ),
        ):
            assert (
                engine.select(
                    ceiling,
                    profile(),
                    DispatchCause.MANUAL_EMERGENCY,
                    view,
                    invalid_context,
                )
                is None
            )
        with pytest.raises(ValidationError):
            EmergencyActivation(
                leg_ids=(SPILL_LEG,),
                dispatcher_fence=11,
                activated_at_ms=NOW,
                expires_at_ms=NOW + 3_600_001,
            )
    finally:
        journal.close()


def classifier() -> FailureClassifier:
    return FailureClassifier(
        (
            FailureRule(
                cause=FailureCause.CAPACITY,
                scope=FailureScope.ACCOUNT,
                status_codes=frozenset({429}),
                message_contains=("max_parallel_requests",),
            ),
            FailureRule(
                cause=FailureCause.TRANSIENT_FAILURE,
                scope=FailureScope.DEPLOYMENT,
                status_codes=frozenset({500, 502, 503}),
                error_codes=frozenset({"upstream_unavailable"}),
            ),
            FailureRule(
                cause=FailureCause.QUOTA,
                scope=FailureScope.ACCOUNT,
                status_codes=frozenset({429}),
                error_codes=frozenset({"monthly_quota"}),
            ),
        )
    )


def test_classifier_requires_one_match_and_unknown_403_429_never_fallback() -> None:
    classify = classifier()
    capacity = classify.classify(
        FailureObservation(
            status_code=429,
            error_code="parallel",
            message="Limit type: max_parallel_requests",
            pre_response_bytes=True,
        )
    )
    transient = classify.classify(
        FailureObservation(
            status_code=503,
            error_code="upstream_unavailable",
            message="down",
            pre_response_bytes=True,
        )
    )
    quota = classify.classify(
        FailureObservation(
            status_code=429,
            error_code="monthly_quota",
            message="spent",
            pre_response_bytes=True,
        )
    )
    assert (capacity.cause, transient.cause, quota.cause) == (
        FailureCause.CAPACITY,
        FailureCause.TRANSIENT_FAILURE,
        FailureCause.QUOTA,
    )
    assert (capacity.dispatch_cause, transient.dispatch_cause, quota.dispatch_cause) == (
        DispatchCause.CAPACITY,
        DispatchCause.FAILURE,
        DispatchCause.QUOTA,
    )
    assert all(item.evidence_digest.startswith("sha256:") for item in (capacity, transient, quota))

    for status in (403, 429):
        unknown = classify.classify(
            FailureObservation(
                status_code=status,
                error_code="provider_overload",
                message="unrecognized",
                pre_response_bytes=True,
            )
        )
        assert (unknown.cause, unknown.scope, unknown.dispatch_cause) == (
            FailureCause.UNKNOWN,
            FailureScope.UNKNOWN,
            None,
        )

    overlap = FailureClassifier(
        (
            FailureRule(
                cause=FailureCause.CAPACITY,
                scope=FailureScope.ACCOUNT,
                status_codes=frozenset({429}),
            ),
            FailureRule(
                cause=FailureCause.QUOTA,
                scope=FailureScope.ACCOUNT,
                status_codes=frozenset({429}),
            ),
        )
    ).classify(
        FailureObservation(
            status_code=429,
            error_code=None,
            message="ambiguous",
            pre_response_bytes=True,
        )
    )
    assert overlap.cause is FailureCause.UNKNOWN and overlap.dispatch_cause is None

    post_byte = classify.classify(
        FailureObservation(
            status_code=503,
            error_code="upstream_unavailable",
            message="down",
            pre_response_bytes=False,
        )
    )
    assert post_byte.cause is FailureCause.TRANSIENT_FAILURE
    assert post_byte.dispatch_cause is None


def test_account_circuit_gates_all_legs_and_stale_probe_cannot_close(tmp_path: Path) -> None:
    bundle = make_bundle()
    journal, state = runtime(tmp_path, bundle)
    probes = iter((ProbeToken("probe_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),))
    circuits = CircuitController(state, probe_factory=lambda: next(probes))
    classification = classifier().classify(
        FailureObservation(
            status_code=429,
            error_code="parallel",
            message="max_parallel_requests",
            pre_response_bytes=True,
        )
    )
    try:
        opened = circuits.open(NAN, NAN_GEN, classification, now_ms=NOW)
        assert opened is not None and opened.state is CircuitState.OPEN
        view = state.operational_view()
        engine = RouteEngine(bundle, GenerationGate())
        ceiling = engine.authorize(client(bundle), profile())
        assert engine.select(ceiling, profile(), DispatchCause.PRIMARY, view, context()) is None

        probe = circuits.begin_probe(NAN, NAN_GEN, now_ms=opened.retry_at_ms)
        assert probe is not None and probe.value.state is CircuitState.HALF_OPEN
        newer = circuits.probe_failed(probe, classification, now_ms=opened.retry_at_ms)
        assert newer is not None and newer.epoch > probe.value.epoch
        assert not circuits.probe_succeeded(probe)
        assert state.account_runtime(NAN).account_circuit_value() == newer
    finally:
        journal.close()


def test_deployment_probe_cannot_mutate_account_scope(tmp_path: Path) -> None:
    bundle = make_bundle()
    journal, state = runtime(tmp_path, bundle)
    circuits = CircuitController(
        state,
        probe_factory=lambda: ProbeToken("probe_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
    )
    transient = classifier().classify(
        FailureObservation(
            status_code=503,
            error_code="upstream_unavailable",
            message="down",
            pre_response_bytes=True,
        )
    )
    capacity = classifier().classify(
        FailureObservation(
            status_code=429,
            error_code="parallel",
            message="max_parallel_requests",
            pre_response_bytes=True,
        )
    )
    try:
        opened = circuits.open(NAN, NAN_GEN, transient, now_ms=NOW)
        assert opened is not None
        assert state.account_runtime(NAN).account_circuit_value() == CircuitValue.closed()
        probe = circuits.begin_probe(NAN, NAN_GEN, now_ms=opened.retry_at_ms)
        assert probe is not None and probe.scope is FailureScope.DEPLOYMENT
        reopened = circuits.probe_failed(probe, capacity, now_ms=opened.retry_at_ms)
        assert reopened is not None and reopened.state is CircuitState.OPEN
        assert reopened.epoch > probe.value.epoch
        assert not circuits.probe_succeeded(probe)
        assert state.account_runtime(NAN).circuit_value(NAN_GEN) == reopened
    finally:
        journal.close()


def test_abandoned_half_open_candidate_reopens_every_probe_scope(tmp_path: Path) -> None:
    bundle = make_bundle()
    journal, state = runtime(tmp_path, bundle)
    circuits = CircuitController(
        state,
        probe_factory=lambda: ProbeToken("probe_cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
    )
    classification = classifier().classify(
        FailureObservation(
            status_code=429,
            error_code="parallel",
            message="max_parallel_requests",
            pre_response_bytes=True,
        )
    )
    try:
        opened = circuits.open(NAN, NAN_GEN, classification, now_ms=NOW)
        assert opened is not None
        probe = circuits.begin_probe(NAN, NAN_GEN, now_ms=opened.retry_at_ms)
        assert probe is not None
        candidate = Candidate(
            authorized(make_bundle().route_groups[0].legs[0]),
            DispatchCause.CAPACITY,
            probe.value,
            CircuitValue.closed(),
        )
        reopened = circuits.abandon_candidate(candidate, now_ms=opened.retry_at_ms)
        assert len(reopened) == 1
        assert reopened[0] is not None
        assert reopened[0].state is CircuitState.OPEN
        assert reopened[0].epoch > probe.value.epoch
    finally:
        journal.close()


def test_attempt_budget_is_three_sends_distinct_generation_or_named_probe_and_no_post_byte() -> (
    None
):
    bundle = make_bundle()
    engine = RouteEngine(bundle, GenerationGate())
    ceiling = engine.authorize(client(bundle), profile())
    primary = engine.candidate(ceiling.authorized_legs[0], DispatchCause.PRIMARY)
    spill = engine.candidate(ceiling.authorized_legs[1], DispatchCause.CAPACITY)
    budget = AttemptBudget(RequestId.new())

    assert budget.can_send(primary)
    budget.record_send(primary, AttemptId.new())
    assert not budget.can_send(primary)  # same leg/generation/circuit epoch
    assert budget.can_send(spill)
    budget.record_send(spill, AttemptId.new())

    probe = replace(
        spill,
        deployment_circuit=spill.deployment_circuit.model_copy(
            update={
                "state": CircuitState.HALF_OPEN,
                "cause": "capacity",
                "epoch": 1,
                "opened_at_ms": NOW,
                "retry_at_ms": 0,
                "backoff_step": 0,
                "evidence_digest": "sha256:" + "a" * 64,
                "probe_id": "probe_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            }
        ),
    )
    assert budget.can_send(probe)
    budget.record_send(probe, AttemptId.new())
    assert not budget.can_send(probe)

    other = replace(
        primary,
        authorized_leg=primary.authorized_leg.model_copy(
            update={
                "leg_id": RouteLegId.new(),
                "generation_id": DeploymentGenerationId.from_digest("f" * 64),
            }
        ),
    )
    assert not budget.can_send(other)  # hard maximum three

    fresh = AttemptBudget(RequestId.new())
    fresh.mark_response_started()
    assert not fresh.can_send(primary)


def test_attempt_budget_restores_terminal_sends_from_the_durable_runtime(
    tmp_path: Path,
) -> None:
    bundle = make_bundle()
    clock = FakeClock()
    journal = AttemptJournal(tmp_path, clock=clock, create=True)
    state = RuntimeState(bundle.accounts, journal=journal, clock=clock)
    request_id = RequestId.new()
    request_profile = profile()
    identity = RuntimeIdentity(
        installation_id=InstallationId.new(),
        dispatcher_fence=11,
        boot_id=GatewayBootId.new(),
        bundle_generation=bundle.generation,
        bundle_hash=bundle_hash(canonical_bundle_bytes(bundle)),
    )
    for index in range(3):
        route = AuthorizedLeg(
            leg_id=RouteLegId.new(),
            account_id=NAN,
            generation_id=DeploymentGenerationId.from_digest(str(index + 1) * 64),
            order=index + 1,
            allowed_triggers=(RouteTrigger.PRIMARY,),
            capabilities=capabilities(),
        )
        candidate = Candidate(
            route,
            DispatchCause.PRIMARY,
            CircuitValue.closed(),
            CircuitValue.closed(),
        )
        reserved = state.try_reserve(
            ReservationRequest(
                request_id=request_id,
                attempt_id=AttemptId.new(),
                account_id=NAN,
                leg_id=route.leg_id,
                deployment_generation_id=route.generation_id,
                runtime_identity=identity,
                deadline_at_ms=NOW + 120_000,
                profile=request_profile,
                input_tokens_upper_bound=request_profile.input_tokens_max,
                max_output_tokens=request_profile.output_tokens_max,
                max_reasoning_tokens=request_profile.reasoning_tokens_max,
                quota_units=1,
                circuit=CircuitValue.closed(),
            )
        )
        assert isinstance(reserved, ReservationGranted)
        reserved.lease.mark_dispatched()
        reserved.lease.finish(
            AttemptResolution(
                outcome=TerminalOutcome.COMPLETED,
                release_capacity=True,
                actual_starts=1,
                actual_token_units=200,
                actual_quota_units=1,
            )
        )
        assert not AttemptBudget.from_runtime(request_id, state.operational_view()).can_send(
            candidate
        )
    journal.close()

    reopened = AttemptJournal.open(tmp_path, clock=clock)
    restored = RuntimeState(bundle.accounts, journal=reopened, clock=clock)
    try:
        budget = AttemptBudget.from_runtime(request_id, restored.operational_view())
        assert len(budget.attempts) == 3
        fourth = Candidate(
            AuthorizedLeg(
                leg_id=RouteLegId.new(),
                account_id=NAN,
                generation_id=DeploymentGenerationId.from_digest("f" * 64),
                order=4,
                allowed_triggers=(RouteTrigger.PRIMARY,),
                capabilities=capabilities(),
            ),
            DispatchCause.PRIMARY,
            CircuitValue.closed(),
            CircuitValue.closed(),
        )
        assert not budget.can_send(fourth)
    finally:
        reopened.close()


def test_route_engine_returns_empty_current_authority_for_removed_route() -> None:
    bundle = make_bundle()
    engine = RouteEngine(bundle, GenerationGate())
    admitted = client(bundle)
    ceiling = engine.authorize(admitted, profile())
    assert isinstance(ceiling, RequestAuthorizationCeiling)
    assert ceiling.bundle_generation == bundle.generation
    assert ceiling.bundle_hash == admitted.applied_bundle_hash
    contracted = engine.authorize(admitted, profile(route_group_id=RouteGroupId.new()))
    assert contracted.authorized_legs == ()


def test_runtime_identity_is_available_for_emergency_fence_checks() -> None:
    identity = RuntimeIdentity(
        installation_id=InstallationId.new(),
        dispatcher_fence=11,
        boot_id=GatewayBootId.new(),
        bundle_generation=7,
        bundle_hash=BundleHash.from_digest("f" * 64),
    )
    assert context().dispatcher_fence == identity.dispatcher_fence
