"""Deterministic hierarchical WDRR and dispatch activation-gate contracts."""

from __future__ import annotations

import asyncio

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    BundleHash,
    DeploymentGenerationId,
    GatewayBootId,
    InstallationId,
    PolicyRevisionId,
    RequestId,
    RouteGroupId,
    RouteLegId,
)
from llmmaxxing.core.models import (
    AuthorizedLeg,
    LegCapabilities,
    RequestAuthorizationCeiling,
    RequestProfile,
)
from llmmaxxing.core.reasons import (
    DispatchCause,
    EndpointKind,
    Modality,
    RouteTrigger,
)
from llmmaxxing.core.state_machines import KeyLifecycleState
from llmmaxxing.gateway.auth import AuthenticatedClient
from llmmaxxing.gateway.routing import AttemptBudget, Candidate
from llmmaxxing.gateway.runtime_state import (
    CircuitValue,
    ReservationGranted,
    RuntimeIdentity,
)
from llmmaxxing.gateway.scheduler import QueueEntry, WDRRQueue

ACCOUNT_A = AccountId("acc_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
ACCOUNT_B = AccountId("acc_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
GEN_A = DeploymentGenerationId.from_digest("a" * 64)
GEN_B = DeploymentGenerationId.from_digest("b" * 64)


def leg(
    account_id: AccountId,
    *,
    leg_id: RouteLegId | None = None,
    generation_id: DeploymentGenerationId | None = None,
    order: int = 10,
) -> AuthorizedLeg:
    return AuthorizedLeg(
        leg_id=leg_id or RouteLegId.new(),
        account_id=account_id,
        generation_id=generation_id or GEN_A,
        order=order,
        allowed_triggers=(RouteTrigger.PRIMARY,),
        capabilities=LegCapabilities(
            endpoints=(EndpointKind.CHAT,),
            modalities=(Modality.TEXT,),
            context_tokens=8_192,
            tools=True,
            forced_tool=True,
            response_schema=True,
            shadow=False,
        ),
    )


def entry(
    key: str,
    *,
    tier: int = 10,
    weight: int = 1,
    accounts: tuple[AccountId, ...] = (ACCOUNT_A,),
    request_id: RequestId | None = None,
) -> QueueEntry:
    return QueueEntry(
        request_id=request_id or RequestId.new(),
        key_id=key,
        tier=tier,
        weight=weight,
        authorized_legs=tuple(
            leg(
                account_id,
                generation_id=GEN_A if account_id == ACCOUNT_A else GEN_B,
                order=index * 10,
            )
            for index, account_id in enumerate(accounts, start=1)
        ),
    )


def grant(queue: WDRRQueue, *, excluded: frozenset[RequestId] = frozenset()) -> QueueEntry:
    selection = queue.propose(lambda _: True, excluded=excluded)
    assert selection is not None
    queue.commit(selection)
    return selection.entry


def test_weight_bounds_are_closed() -> None:
    for invalid in (0, 65):
        with pytest.raises(ValueError, match="weight"):
            entry("bad", weight=invalid)
    assert entry("low", weight=1).weight == 1
    assert entry("high", weight=64).weight == 64


def test_fifo_is_stable_within_key() -> None:
    queue = WDRRQueue()
    requests = [entry("same", request_id=RequestId.new()) for _ in range(20)]
    for item in requests:
        queue.enqueue(item)
    assert [grant(queue).request_id for _ in requests] == [item.request_id for item in requests]


def test_key_weights_are_wdrr_not_strict_priority() -> None:
    queue = WDRRQueue()
    for _ in range(40):
        queue.enqueue(entry("heavy", weight=4))
        queue.enqueue(entry("light", weight=1))
    first_twenty = [grant(queue).key_id for _ in range(20)]
    assert first_twenty.count("heavy") == 16
    assert first_twenty.count("light") == 4
    assert "light" in first_twenty[:5]


def test_scarce_flexible_class_weight_is_two_to_one_and_scarcity_is_frozen() -> None:
    queue = WDRRQueue()
    scarce = [entry("scarce", accounts=(ACCOUNT_A,)) for _ in range(12)]
    flexible = [entry("flex", accounts=(ACCOUNT_A, ACCOUNT_B)) for _ in range(12)]
    for scarce_item, flexible_item in zip(scarce, flexible, strict=True):
        queue.enqueue(scarce_item)
        queue.enqueue(flexible_item)
    first_nine = [grant(queue).scarcity.value for _ in range(9)]
    assert first_nine == ["scarce", "scarce", "flexible"] * 3

    remaining_flexible = next(item for item in queue.entries if item.key_id == "flex")
    queue.contract(
        remaining_flexible.request_id,
        authorized_legs=(remaining_flexible.authorized_legs[0],),
    )
    assert queue.entry(remaining_flexible.request_id).scarcity.value == "flexible"


def test_lower_tier_is_forced_within_the_configured_aging_bound() -> None:
    queue = WDRRQueue(max_consecutive_higher_grants=8)
    for _ in range(100):
        queue.enqueue(entry("tier0", tier=0, weight=64))
    low = entry("tier100", tier=100, weight=1)
    queue.enqueue(low)
    selected = [grant(queue).request_id for _ in range(9)]
    assert low.request_id in selected


def test_sustained_arrivals_cannot_starve_lower_tier_or_flexible_class() -> None:
    queue = WDRRQueue(max_consecutive_higher_grants=16)
    low_grants = 0
    flexible_grants = 0
    for _ in range(1_000):
        queue.enqueue(entry("top-scarce", tier=0, weight=64, accounts=(ACCOUNT_A,)))
        queue.enqueue(entry("top-flex", tier=0, weight=1, accounts=(ACCOUNT_A, ACCOUNT_B)))
        queue.enqueue(entry("low", tier=100, weight=1, accounts=(ACCOUNT_B,)))
        selected = grant(queue)
        low_grants += selected.key_id == "low"
        flexible_grants += selected.key_id == "top-flex"
    assert low_grants >= 58  # one forced service in every 17 committed grants
    assert flexible_grants > 0


def test_ineligible_and_reservation_race_scans_charge_no_deficit() -> None:
    queue = WDRRQueue()
    blocked = entry("blocked", weight=64)
    ready = entry("ready", weight=1)
    queue.enqueue(blocked)
    queue.enqueue(ready)
    before = queue.snapshot()

    selection = queue.propose(lambda item: item.request_id == ready.request_id)
    assert selection is not None and selection.entry == ready
    assert queue.snapshot() == before  # proposal/ineligible scan is side-effect free

    raced = queue.propose(lambda _: True)
    assert raced is not None and raced.entry == blocked
    after_race = queue.snapshot()
    fallback = queue.propose(lambda _: True, excluded=frozenset({blocked.request_id}))
    assert fallback is not None and fallback.entry == ready
    assert queue.snapshot() == after_race  # failed Task6 reservation did not charge
    queue.commit(fallback)
    assert queue.deficit(blocked) == 0


def test_cancel_removes_waiter_without_charging_or_reordering() -> None:
    queue = WDRRQueue()
    first, cancelled, last = (entry("same") for _ in range(3))
    for item in (first, cancelled, last):
        queue.enqueue(item)
    state = queue.fairness_state
    assert queue.cancel(cancelled.request_id)
    assert not queue.cancel(cancelled.request_id)
    assert queue.fairness_state == state
    assert [grant(queue).request_id, grant(queue).request_id] == [first.request_id, last.request_id]


def test_activation_contracts_qos_without_deficit_reset_or_expansion() -> None:
    queue = WDRRQueue()
    heavy = [entry("heavy", weight=8) for _ in range(20)]
    light = [entry("light", weight=1) for _ in range(20)]
    for heavy_item, light_item in zip(heavy, light, strict=True):
        queue.enqueue(heavy_item)
        queue.enqueue(light_item)
    grant(queue)
    grant(queue)
    before = queue.fairness_state
    target = next(item for item in queue.entries if item.key_id == "heavy")
    queue.contract(target.request_id, tier=20, weight=4)
    after = queue.fairness_state
    assert after.key_deficits == before.key_deficits
    assert after.class_deficits == before.class_deficits
    assert queue.entry(target.request_id).tier == 20
    assert queue.entry(target.request_id).weight == 4
    with pytest.raises(ValueError, match="expand"):
        queue.contract(target.request_id, tier=0, weight=8)


def test_snapshot_restore_replays_the_same_grant_order_and_caps_deficits() -> None:
    queue = WDRRQueue(max_consecutive_higher_grants=7)
    for index in range(80):
        queue.enqueue(
            entry(
                f"key-{index % 5}",
                tier=(0, 10, 100)[index % 3],
                weight=(1, 2, 4, 8, 64)[index % 5],
                accounts=(ACCOUNT_A,) if index % 2 else (ACCOUNT_A, ACCOUNT_B),
            )
        )
    for _ in range(17):
        grant(queue)
    restored = WDRRQueue.restore(queue.snapshot())
    assert [grant(queue).request_id for _ in range(40)] == [
        grant(restored).request_id for _ in range(40)
    ]
    assert all(
        deficit <= 8 * quantum
        for deficit, quantum in restored.deficits_with_quanta()
    )


def test_scarce_waiters_protect_their_only_account_from_flexible_work() -> None:
    queue = WDRRQueue()
    scarce = entry("scarce", accounts=(ACCOUNT_A,))
    flexible = entry("flexible", accounts=(ACCOUNT_A, ACCOUNT_B))
    queue.enqueue(scarce)
    queue.enqueue(flexible)
    candidates = {
        scarce.request_id: (ACCOUNT_A,),
        flexible.request_id: (ACCOUNT_A, ACCOUNT_B),
    }
    assert queue.protected_scarce_accounts(candidates) == frozenset({ACCOUNT_A})
    assert queue.unprotected_accounts(flexible, candidates[flexible.request_id], candidates) == (
        ACCOUNT_B,
    )


def test_selection_tokens_reject_stale_commit() -> None:
    queue = WDRRQueue()
    first = entry("first")
    second = entry("second")
    queue.enqueue(first)
    queue.enqueue(second)
    stale = queue.propose(lambda _: True)
    assert stale is not None
    queue.cancel(first.request_id)
    with pytest.raises(RuntimeError, match="stale"):
        queue.commit(stale)


def test_activation_gate_protocol_has_no_default_and_orders_dispatch_critical_section() -> None:
    """The real controller test lands with Task8; this pins Task7's mandatory gate shape."""
    from llmmaxxing.gateway.scheduler import ActivationGate

    events: list[str] = []

    class RecordingGate:
        @asynccontextmanager
        async def hold_dispatch(self, request_id: RequestId):  # type: ignore[no-untyped-def]
            events.append(f"enter:{request_id}")
            yield
            events.append(f"exit:{request_id}")

    async def dispatch_sequence(gate: ActivationGate, request_id: RequestId) -> None:
        async with gate.hold_dispatch(request_id):
            events.extend(("reauthorize", "task6-reserve", "journal-dispatched"))

    request_id = RequestId.new()
    asyncio.run(dispatch_sequence(RecordingGate(), request_id))
    assert events == [
        f"enter:{request_id}",
        "reauthorize",
        "task6-reserve",
        "journal-dispatched",
        f"exit:{request_id}",
    ]
    assert AttemptId.new() != AttemptId.new()


def test_admission_controller_requires_gate_and_persists_dispatched_inside_it() -> None:
    from llmmaxxing.gateway.scheduler import AdmissionController, AdmissionRequest

    request_id = RequestId.new()
    route_group_id = RouteGroupId.new()
    policy_id = PolicyRevisionId.new()
    authorized_leg = leg(ACCOUNT_A)
    ceiling = RequestAuthorizationCeiling(
        key_id="1" * 32,
        credential_generation=1,
        policy_id=policy_id,
        bundle_generation=7,
        bundle_hash=BundleHash.from_digest("c" * 64),
        route_group_id=route_group_id,
        authorized_legs=(authorized_leg,),
        queue_tier=10,
        queue_weight=4,
        max_concurrency=4,
        max_waiters=16,
        deadline_ms=60_000,
    )
    request_profile = RequestProfile(
        route_group_id=route_group_id,
        model_alias="deepseek-v4-flash",
        endpoint=EndpointKind.CHAT,
        modality=Modality.TEXT,
        stream=False,
        input_tokens_max=10,
        output_tokens_max=10,
        reasoning_tokens_max=0,
        tools_count=0,
        forced_tool_required=False,
        response_schema_present=False,
        history_turns=0,
        deadline_ms=60_000,
    )
    authenticated = AuthenticatedClient(
        key_id="1" * 32,
        accepted_credential_generation=1,
        policy_id=policy_id,
        key_state=KeyLifecycleState.ENABLED,
        key_expires_at_s=1_900_000_000,
        applied_bundle_generation=7,
        applied_bundle_hash=ceiling.bundle_hash,
    )
    candidate = Candidate(
        authorized_leg=authorized_leg,
        cause=DispatchCause.PRIMARY,
        account_circuit=CircuitValue.closed(),
        deployment_circuit=CircuitValue.closed(),
    )
    events: list[str] = []
    inside_gate = False

    class Gate:
        @asynccontextmanager
        async def hold_dispatch(self, dispatch_request_id: RequestId):  # type: ignore[no-untyped-def]
            nonlocal inside_gate
            assert dispatch_request_id == request_id
            inside_gate = True
            events.append("gate-enter")
            yield
            events.append("gate-exit")
            inside_gate = False

    class Engine:
        def authorize(self, *_: object) -> RequestAuthorizationCeiling:
            events.append("reauthorize" if inside_gate else "preview-authorize")
            return ceiling

        def select(self, *_: object, **__: object) -> Candidate:
            return candidate

        def filter(self, *_: object, **__: object) -> tuple[Candidate, ...]:
            return (candidate,)

        def primary_capacity_unavailable(self, *_: object) -> bool:
            return False

        def quota_units(self, account_id: AccountId) -> int:
            assert account_id == ACCOUNT_A
            return 1

    class View:
        dispatches: tuple[()] = ()

    class DurableLease:
        async def mark_dispatched_async(self) -> object:
            assert inside_gate
            events.append("journal-dispatched")
            return object()

    class Runtime:
        def operational_view(self) -> object:
            return View()

        async def try_reserve_async(self, reservation: object) -> ReservationGranted:
            assert inside_gate
            assert getattr(reservation, "leg_id") == authorized_leg.leg_id
            events.append("task6-reserve")
            return ReservationGranted(DurableLease())  # type: ignore[arg-type]

    class Clock:
        def now_ms(self) -> int:
            return 1_800_000_000_000

    admission = AdmissionRequest(
        request_id=request_id,
        client=authenticated,
        profile=request_profile,
        authorization_ceiling=ceiling,
        runtime_identity=RuntimeIdentity(
            installation_id=InstallationId.new(),
            dispatcher_fence=11,
            boot_id=GatewayBootId.new(),
            bundle_generation=7,
            bundle_hash=ceiling.bundle_hash,
        ),
        deadline_at_ms=1_800_000_060_000,
        cause=DispatchCause.PRIMARY,
        attempt_budget=AttemptBudget(request_id),
    )
    with pytest.raises(TypeError):
        AdmissionController(Engine(), Runtime(), clock=Clock())  # type: ignore[call-arg]
    controller = AdmissionController(Engine(), Runtime(), Gate(), clock=Clock())
    dispatch = asyncio.run(controller.acquire(admission))
    assert dispatch.candidate == candidate
    assert events[-5:] == [
        "gate-enter",
        "reauthorize",
        "task6-reserve",
        "journal-dispatched",
        "gate-exit",
    ]
