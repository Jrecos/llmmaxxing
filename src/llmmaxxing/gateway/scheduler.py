"""Hierarchical deterministic WDRR and activation-gated durable dispatch."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from typing import Protocol, Self

from llmmaxxing.core.ids import AccountId, AttemptId, RequestId
from llmmaxxing.core.models import (
    AuthorizedLeg,
    RequestAuthorizationCeiling,
    RequestProfile,
)
from llmmaxxing.core.reasons import DispatchCause, TerminalOutcome
from llmmaxxing.gateway.auth import (
    AuthenticatedClient,
    AuthRuntimeView,
    queued_identity_is_authorized,
)
from llmmaxxing.gateway.routing import (
    AttemptBudget,
    Candidate,
    CircuitController,
    EmergencyActivation,
    RouteEngine,
    RoutingContext,
)
from llmmaxxing.gateway.runtime_state import (
    AttemptResolution,
    Lease,
    ReservationDenialReason,
    ReservationDenied,
    ReservationGranted,
    ReservationRequest,
    RuntimeIdentity,
    RuntimeState,
)


class Scarcity(StrEnum):
    SCARCE = "scarce"
    FLEXIBLE = "flexible"


@dataclass(frozen=True, slots=True)
class QueueEntry:
    request_id: RequestId
    key_id: str
    tier: int
    weight: int
    authorized_legs: tuple[AuthorizedLeg, ...]
    scarcity: Scarcity = field(init=False)

    def __post_init__(self) -> None:
        if not self.key_id or len(self.key_id) > 160:
            raise ValueError("key id is empty or oversized")
        if self.tier < 0:
            raise ValueError("tier must be nonnegative")
        if not 1 <= self.weight <= 64:
            raise ValueError("weight must be in [1, 64]")
        if not self.authorized_legs:
            raise ValueError("queue entry requires an authorization ceiling")
        if len({leg.leg_id for leg in self.authorized_legs}) != len(self.authorized_legs):
            raise ValueError("queue entry contains duplicate authorized legs")
        accounts = {leg.account_id for leg in self.authorized_legs}
        object.__setattr__(
            self,
            "scarcity",
            Scarcity.SCARCE if len(accounts) == 1 else Scarcity.FLEXIBLE,
        )


@dataclass(frozen=True, slots=True)
class _Queued:
    entry: QueueEntry
    sequence: int


@dataclass(frozen=True, slots=True)
class FairnessState:
    class_deficits: tuple[tuple[int, Scarcity, int], ...] = ()
    class_cursors: tuple[tuple[int, Scarcity], ...] = ()
    key_deficits: tuple[tuple[int, Scarcity, str, int], ...] = ()
    key_cursors: tuple[tuple[int, Scarcity, str], ...] = ()
    higher_grant_streak: int = 0


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    queued: tuple[_Queued, ...]
    fairness: FairnessState
    next_sequence: int
    version: int
    max_consecutive_higher_grants: int


@dataclass(frozen=True, slots=True)
class QueueSelection:
    entry: QueueEntry
    queue_version: int
    fairness: FairnessState


class WDRRQueue:
    """Tier forcing over scarce:flexible 2:1 and per-key weighted DRR."""

    def __init__(self, *, max_consecutive_higher_grants: int = 64) -> None:
        if max_consecutive_higher_grants < 1:
            raise ValueError("aging grant bound must be positive")
        self.max_consecutive_higher_grants = max_consecutive_higher_grants
        self._queued: dict[RequestId, _Queued] = {}
        self._next_sequence = 0
        self._version = 0
        self._fairness = FairnessState()

    @property
    def entries(self) -> tuple[QueueEntry, ...]:
        return tuple(
            item.entry for item in sorted(self._queued.values(), key=lambda item: item.sequence)
        )

    @property
    def fairness_state(self) -> FairnessState:
        return self._fairness

    def entry(self, request_id: RequestId) -> QueueEntry:
        return self._queued[request_id].entry

    def key_position(self, request_id: RequestId) -> int:
        target = self._queued[request_id]
        return sum(
            item.entry.key_id == target.entry.key_id
            and (item.sequence, str(item.entry.request_id))
            < (target.sequence, str(target.entry.request_id))
            for item in self._queued.values()
        )

    def enqueue(self, entry: QueueEntry) -> None:
        if entry.request_id in self._queued:
            raise ValueError("duplicate queued request")
        self._queued[entry.request_id] = _Queued(entry, self._next_sequence)
        self._next_sequence += 1
        self._version += 1

    def cancel(self, request_id: RequestId) -> bool:
        if self._queued.pop(request_id, None) is None:
            return False
        self._version += 1
        return True

    @staticmethod
    def _leg_contracts(old: AuthorizedLeg, new: AuthorizedLeg) -> bool:
        return old.intersection(new) == new

    def contract(
        self,
        request_id: RequestId,
        *,
        tier: int | None = None,
        weight: int | None = None,
        authorized_legs: tuple[AuthorizedLeg, ...] | None = None,
    ) -> QueueEntry:
        queued = self._queued[request_id]
        old = queued.entry
        new_tier = old.tier if tier is None else tier
        new_weight = old.weight if weight is None else weight
        new_legs = old.authorized_legs if authorized_legs is None else authorized_legs
        if new_tier < old.tier or new_weight > old.weight:
            raise ValueError("activation cannot expand queued QoS")
        old_by_identity = {leg.identity: leg for leg in old.authorized_legs}
        if any(
            (previous := old_by_identity.get(candidate.identity)) is None
            or not self._leg_contracts(previous, candidate)
            for candidate in new_legs
        ):
            raise ValueError("activation cannot expand queued authorization")
        updated = replace(
            old,
            tier=new_tier,
            weight=new_weight,
            authorized_legs=new_legs,
        )
        object.__setattr__(updated, "scarcity", old.scarcity)
        self._queued[request_id] = _Queued(updated, queued.sequence)
        if new_tier == old.tier and new_weight < old.weight:
            class_deficits, class_cursors, key_deficits, key_cursors = self._decode_state(
                self._fairness
            )
            key = (old.tier, old.scarcity, old.key_id)
            if key in key_deficits:
                key_deficits[key] = min(key_deficits[key], 8 * new_weight)
                self._fairness = self._encode_state(
                    class_deficits,
                    class_cursors,
                    key_deficits,
                    key_cursors,
                    self._fairness.higher_grant_streak,
                )
        self._version += 1
        return updated

    @staticmethod
    def _decode_state(
        state: FairnessState,
    ) -> tuple[
        dict[tuple[int, Scarcity], int],
        dict[int, Scarcity],
        dict[tuple[int, Scarcity, str], int],
        dict[tuple[int, Scarcity], str],
    ]:
        return (
            {(tier, scarcity): value for tier, scarcity, value in state.class_deficits},
            dict(state.class_cursors),
            {
                (tier, scarcity, key_id): value
                for tier, scarcity, key_id, value in state.key_deficits
            },
            {(tier, scarcity): key_id for tier, scarcity, key_id in state.key_cursors},
        )

    @staticmethod
    def _encode_state(
        class_deficits: Mapping[tuple[int, Scarcity], int],
        class_cursors: Mapping[int, Scarcity],
        key_deficits: Mapping[tuple[int, Scarcity, str], int],
        key_cursors: Mapping[tuple[int, Scarcity], str],
        higher_grant_streak: int,
    ) -> FairnessState:
        return FairnessState(
            class_deficits=tuple(
                (tier, scarcity, value)
                for (tier, scarcity), value in sorted(
                    class_deficits.items(), key=lambda item: (item[0][0], item[0][1].value)
                )
            ),
            class_cursors=tuple(sorted(class_cursors.items())),
            key_deficits=tuple(
                (tier, scarcity, key_id, value)
                for (tier, scarcity, key_id), value in sorted(
                    key_deficits.items(),
                    key=lambda item: (item[0][0], item[0][1].value, item[0][2]),
                )
            ),
            key_cursors=tuple(
                (tier, scarcity, key_id)
                for (tier, scarcity), key_id in sorted(
                    key_cursors.items(), key=lambda item: (item[0][0], item[0][1].value)
                )
            ),
            higher_grant_streak=higher_grant_streak,
        )

    @staticmethod
    def _oldest(items: list[_Queued]) -> _Queued:
        return min(items, key=lambda item: (item.sequence, str(item.entry.request_id)))

    def propose(
        self,
        eligible: Callable[[QueueEntry], bool],
        *,
        excluded: frozenset[RequestId] = frozenset(),
    ) -> QueueSelection | None:
        available = [
            item
            for item in self._queued.values()
            if item.entry.request_id not in excluded and eligible(item.entry)
        ]
        if not available:
            return None
        tiers = sorted({item.entry.tier for item in available})
        top_tier = tiers[0]
        lower = [item for item in available if item.entry.tier > top_tier]
        if lower and self._fairness.higher_grant_streak >= self.max_consecutive_higher_grants:
            tier = self._oldest(lower).entry.tier
        else:
            tier = top_tier
        tier_items = [item for item in available if item.entry.tier == tier]

        class_deficits, class_cursors, key_deficits, key_cursors = self._decode_state(
            self._fairness
        )
        active_classes = {item.entry.scarcity for item in tier_items}
        scarcity = class_cursors.get(tier, Scarcity.SCARCE)
        for _ in range(2):
            if scarcity in active_classes:
                break
            scarcity = Scarcity.FLEXIBLE if scarcity is Scarcity.SCARCE else Scarcity.SCARCE
        quantum = 2 if scarcity is Scarcity.SCARCE else 1
        class_key = (tier, scarcity)
        class_deficit = min(class_deficits.get(class_key, 0), 8 * quantum)
        if class_deficit <= 0:
            class_deficit = min(class_deficit + quantum, 8 * quantum)
        class_items = [item for item in tier_items if item.entry.scarcity is scarcity]

        first_sequence: dict[str, int] = {}
        for item in class_items:
            first_sequence[item.entry.key_id] = min(
                item.sequence,
                first_sequence.get(item.entry.key_id, item.sequence),
            )
        keys = sorted(first_sequence, key=lambda key: (first_sequence[key], key))
        key_id = key_cursors.get(class_key, keys[0])
        if key_id not in first_sequence:
            key_id = keys[0]
        weights = [item.entry.weight for item in class_items if item.entry.key_id == key_id]
        key_quantum = min(weights)
        key_key = (tier, scarcity, key_id)
        key_deficit = min(key_deficits.get(key_key, 0), 8 * key_quantum)
        if key_deficit <= 0:
            key_deficit = min(key_deficit + key_quantum, 8 * key_quantum)
        selected = self._oldest([item for item in class_items if item.entry.key_id == key_id])

        class_deficit -= 1
        key_deficit -= 1
        class_deficits[class_key] = class_deficit
        key_deficits[key_key] = key_deficit
        other_class = Scarcity.FLEXIBLE if scarcity is Scarcity.SCARCE else Scarcity.SCARCE
        class_cursors[tier] = other_class if class_deficit == 0 else scarcity
        key_index = keys.index(key_id)
        key_cursors[class_key] = keys[(key_index + 1) % len(keys)] if key_deficit == 0 else key_id
        streak = self._fairness.higher_grant_streak + 1 if lower and tier == top_tier else 0
        return QueueSelection(
            entry=selected.entry,
            queue_version=self._version,
            fairness=self._encode_state(
                class_deficits,
                class_cursors,
                key_deficits,
                key_cursors,
                streak,
            ),
        )

    def commit(self, selection: QueueSelection) -> None:
        queued = self._queued.get(selection.entry.request_id)
        if (
            selection.queue_version != self._version
            or queued is None
            or queued.entry != selection.entry
        ):
            raise RuntimeError("stale WDRR selection")
        self._fairness = selection.fairness
        del self._queued[selection.entry.request_id]
        self._version += 1

    def deficit(self, entry: QueueEntry) -> int:
        _, _, key_deficits, _ = self._decode_state(self._fairness)
        return key_deficits.get((entry.tier, entry.scarcity, entry.key_id), 0)

    def deficits_with_quanta(self) -> tuple[tuple[int, int], ...]:
        class_deficits, _, key_deficits, _ = self._decode_state(self._fairness)
        values = [
            (value, 2 if scarcity is Scarcity.SCARCE else 1)
            for (_, scarcity), value in class_deficits.items()
        ]
        weights = {
            (item.entry.tier, item.entry.scarcity, item.entry.key_id): item.entry.weight
            for item in self._queued.values()
        }
        values.extend((value, weights.get(key, 64)) for key, value in key_deficits.items())
        return tuple(values)

    def protected_scarce_accounts(
        self,
        candidates: Mapping[RequestId, tuple[AccountId, ...]],
    ) -> frozenset[AccountId]:
        return frozenset(
            account_id
            for item in self._queued.values()
            if item.entry.scarcity is Scarcity.SCARCE
            for account_id in candidates.get(item.entry.request_id, ())
        )

    def unprotected_accounts(
        self,
        entry: QueueEntry,
        accounts: tuple[AccountId, ...],
        candidates: Mapping[RequestId, tuple[AccountId, ...]],
    ) -> tuple[AccountId, ...]:
        if entry.scarcity is Scarcity.SCARCE:
            return accounts
        protected = self.protected_scarce_accounts(candidates)
        return tuple(account_id for account_id in accounts if account_id not in protected)

    def snapshot(self) -> QueueSnapshot:
        return QueueSnapshot(
            queued=tuple(sorted(self._queued.values(), key=lambda item: item.sequence)),
            fairness=self._fairness,
            next_sequence=self._next_sequence,
            version=self._version,
            max_consecutive_higher_grants=self.max_consecutive_higher_grants,
        )

    @classmethod
    def restore(cls, snapshot: QueueSnapshot) -> Self:
        queue = cls(max_consecutive_higher_grants=snapshot.max_consecutive_higher_grants)
        if len({item.entry.request_id for item in snapshot.queued}) != len(snapshot.queued):
            raise ValueError("scheduler snapshot has duplicate requests")
        queue._queued = {item.entry.request_id: item for item in snapshot.queued}
        queue._fairness = snapshot.fairness
        queue._next_sequence = snapshot.next_sequence
        queue._version = snapshot.version
        return queue


class ActivationGate(Protocol):
    """Task 9 implementation serializes publication with provider-send permission."""

    def hold_dispatch(
        self,
        request_id: RequestId,
    ) -> AbstractAsyncContextManager[None]: ...


class AdmissionClock(Protocol):
    def now_ms(self) -> int: ...


class AuthViewProvider(Protocol):
    """Task 4/9 supplies the current deny/credential/key authorization view."""

    def current_auth_view(self) -> AuthRuntimeView: ...


class AdmissionUnavailable(RuntimeError):
    pass


class WaiterState(StrEnum):
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    request_id: RequestId
    client: AuthenticatedClient
    profile: RequestProfile
    authorization_ceiling: RequestAuthorizationCeiling
    runtime_identity: RuntimeIdentity
    deadline_at_ms: int
    cause: DispatchCause
    attempt_budget: AttemptBudget
    emergency: EmergencyActivation | None = None

    def __post_init__(self) -> None:
        if self.request_id != self.attempt_budget.request_id:
            raise ValueError("attempt budget belongs to a different request")
        if self.profile.route_group_id != self.authorization_ceiling.route_group_id:
            raise ValueError("profile and authorization ceiling route groups differ")
        if self.client.key_id != self.authorization_ceiling.key_id:
            raise ValueError("client and authorization ceiling keys differ")
        if self.deadline_at_ms < 1:
            raise ValueError("absolute request deadline must be positive")


@dataclass(slots=True)
class _Waiter:
    request: AdmissionRequest
    future: asyncio.Future[DispatchLease]
    effective_ceiling: RequestAuthorizationCeiling
    admitted_at_ms: int
    effective_deadline_at_ms: int
    fallback_once: DispatchCause | None = None
    state: WaiterState = WaiterState.QUEUED


@dataclass(slots=True)
class DispatchLease:
    """Task 6 lease plus idempotent active-permit and probe reconciliation."""

    lease: Lease
    candidate: Candidate
    attempt_id: AttemptId
    attempt_budget: AttemptBudget
    _release_active: Callable[[], None]
    _abandon_probes: Callable[[], None]
    _released: bool = False
    _release_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def terminal(self) -> bool:
        return self.lease.terminal

    def _release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._release_active()

    def finish(self, resolution: AttemptResolution) -> object:
        result = self.lease.finish(resolution)
        self._release()
        return result

    async def finish_async(self, resolution: AttemptResolution) -> object:
        worker = asyncio.create_task(self.lease.finish_async(resolution))
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await asyncio.shield(worker)
            self._release()
            raise
        self._release()
        return result

    async def cancel_before_send(self) -> object:
        self._abandon_probes()
        return await self.finish_async(
            AttemptResolution(
                outcome=TerminalOutcome.CLIENT_CANCELLED,
                release_capacity=True,
                actual_starts=0,
                actual_token_units=0,
                actual_quota_units=0,
            )
        )

    def mark_response_started(self) -> None:
        self.attempt_budget.mark_response_started()


class AdmissionController:
    """Queue requests and grant only activation-gated, durably dispatched leases."""

    def __init__(
        self,
        route_engine: RouteEngine,
        runtime: RuntimeState,
        activation_gate: ActivationGate,
        auth_view_provider: AuthViewProvider,
        circuit_controller: CircuitController,
        *,
        clock: AdmissionClock,
        attempt_id_factory: Callable[[], AttemptId] = AttemptId.new,
        max_waiters_global: int = 64,
    ) -> None:
        if max_waiters_global < 1:
            raise ValueError("global waiter bound must be positive")
        self._route_engine = route_engine
        self._runtime = runtime
        self._activation_gate = activation_gate
        self._auth_view_provider = auth_view_provider
        self._circuit_controller = circuit_controller
        self._clock = clock
        self._attempt_id_factory = attempt_id_factory
        self._max_waiters_global = max_waiters_global
        self._queue = WDRRQueue()
        self._waiters: dict[RequestId, _Waiter] = {}
        self._pump_lock = asyncio.Lock()
        self._active_lock = threading.Lock()
        self._active_by_key: dict[str, int] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _active_count(self, key_id: str) -> int:
        with self._active_lock:
            return self._active_by_key.get(key_id, 0)

    def _acquire_active(self, key_id: str, maximum: int) -> bool:
        with self._active_lock:
            current = self._active_by_key.get(key_id, 0)
            if current >= maximum:
                return False
            self._active_by_key[key_id] = current + 1
            return True

    def _release_active(self, key_id: str, *, wake: bool = True) -> None:
        with self._active_lock:
            current = self._active_by_key.get(key_id, 0)
            if current <= 1:
                self._active_by_key.pop(key_id, None)
            else:
                self._active_by_key[key_id] = current - 1
        loop = self._loop
        if wake and loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: asyncio.create_task(self.wake()))

    def _routing_context(self, waiter: _Waiter) -> RoutingContext:
        return RoutingContext(
            now_ms=self._clock.now_ms(),
            deadline_at_ms=waiter.effective_deadline_at_ms,
            dispatcher_fence=waiter.request.runtime_identity.dispatcher_fence,
            emergency=waiter.request.emergency,
        )

    def _abandon_candidate(self, candidate: Candidate) -> None:
        self._circuit_controller.abandon_candidate(
            candidate,
            now_ms=self._clock.now_ms(),
        )

    def _current_authorization(
        self,
        waiter: _Waiter,
    ) -> RequestAuthorizationCeiling:
        try:
            auth_view = self._auth_view_provider.current_auth_view()
            authorized = queued_identity_is_authorized(
                waiter.request.client,
                auth_view,
            )
        except Exception as error:
            raise AdmissionUnavailable("queued authorization state unavailable") from error
        if not authorized:
            raise AdmissionUnavailable("queued client authorization is no longer valid")
        try:
            current = self._route_engine.authorize(waiter.request.client, waiter.request.profile)
        except ValueError as error:
            waiter.effective_ceiling = waiter.effective_ceiling.model_copy(
                update={"authorized_legs": ()}
            )
            raise AdmissionUnavailable("queued route authorization contracted to empty") from error
        contracted = waiter.effective_ceiling.intersection(current)
        waiter.effective_ceiling = contracted
        waiter.effective_deadline_at_ms = min(
            waiter.effective_deadline_at_ms,
            waiter.admitted_at_ms + contracted.deadline_ms,
        )
        if not contracted.authorized_legs:
            raise AdmissionUnavailable("queued route authorization contracted to empty")
        if self._queue.key_position(waiter.request.request_id) >= contracted.max_waiters:
            raise AdmissionUnavailable("queued key waiter ceiling contracted")
        return contracted

    def _fail_waiter_locked(self, waiter: _Waiter, error: BaseException) -> None:
        self._waiters.pop(waiter.request.request_id, None)
        self._queue.cancel(waiter.request.request_id)
        if not waiter.future.done():
            waiter.future.set_exception(error)

    def _preview(self, waiter: _Waiter) -> tuple[Candidate, ...]:
        try:
            authorization = self._current_authorization(waiter)
        except AdmissionUnavailable as error:
            self._fail_waiter_locked(waiter, error)
            return ()
        queued = self._queue.entry(waiter.request.request_id)
        if (
            queued.tier != authorization.queue_tier
            or queued.weight != authorization.queue_weight
            or queued.authorized_legs != authorization.authorized_legs
        ):
            self._queue.contract(
                waiter.request.request_id,
                tier=authorization.queue_tier,
                weight=authorization.queue_weight,
                authorized_legs=authorization.authorized_legs,
            )
        if self._clock.now_ms() >= waiter.effective_deadline_at_ms:
            self._fail_waiter_locked(
                waiter, AdmissionUnavailable("queued request deadline elapsed")
            )
            return ()
        if self._active_count(waiter.request.client.key_id) >= authorization.max_concurrency:
            return ()
        view = self._runtime.operational_view()
        waiter.request.attempt_budget.sync_runtime(view)
        cause = waiter.fallback_once or waiter.request.cause
        candidates = self._route_engine.filter(
            authorization,
            waiter.request.profile,
            cause,
            view,
            self._routing_context(waiter),
        )
        if not candidates and cause is DispatchCause.PRIMARY:
            fallback = self._route_engine.primary_blocking_cause(
                authorization, waiter.request.profile, view
            )
            if fallback is not None:
                candidates = self._route_engine.filter(
                    authorization,
                    waiter.request.profile,
                    fallback,
                    view,
                    self._routing_context(waiter),
                )
        return tuple(
            candidate
            for candidate in candidates
            if waiter.request.attempt_budget.can_send(candidate)
        )

    def _previews(self) -> dict[RequestId, tuple[Candidate, ...]]:
        candidates: dict[RequestId, tuple[Candidate, ...]] = {}
        for entry in tuple(self._queue.entries):
            waiter = self._waiters.get(entry.request_id)
            if waiter is None:
                continue
            preview = self._preview(waiter)
            if preview:
                candidates[entry.request_id] = preview
        return candidates

    async def acquire(self, request: AdmissionRequest) -> DispatchLease:
        loop = asyncio.get_running_loop()
        self._loop = loop
        future: asyncio.Future[DispatchLease] = loop.create_future()
        try:
            async with self._pump_lock:
                now_ms = self._clock.now_ms()
                request.attempt_budget.sync_runtime(self._runtime.operational_view())
                if request.attempt_budget.send_closed:
                    raise AdmissionUnavailable("provider send budget is closed")
                if request.deadline_at_ms <= now_ms:
                    raise AdmissionUnavailable("request deadline elapsed before queue admission")
                if request.request_id in self._waiters:
                    raise ValueError("duplicate admission request")
                if len(self._waiters) >= self._max_waiters_global:
                    raise AdmissionUnavailable("global waiter bound reached")
                key_waiters = sum(
                    waiter.request.client.key_id == request.client.key_id
                    for waiter in self._waiters.values()
                )
                if key_waiters >= request.authorization_ceiling.max_waiters:
                    raise AdmissionUnavailable("key waiter bound reached")
                try:
                    initially_authorized = queued_identity_is_authorized(
                        request.client,
                        self._auth_view_provider.current_auth_view(),
                    )
                except Exception as error:
                    raise AdmissionUnavailable("client authorization state unavailable") from error
                if not initially_authorized:
                    raise AdmissionUnavailable(
                        "client authorization expired before queue admission"
                    )
                waiter = _Waiter(
                    request=request,
                    future=future,
                    effective_ceiling=request.authorization_ceiling,
                    admitted_at_ms=now_ms,
                    effective_deadline_at_ms=min(
                        request.deadline_at_ms,
                        now_ms + request.authorization_ceiling.deadline_ms,
                    ),
                )
                self._waiters[request.request_id] = waiter
                self._queue.enqueue(
                    QueueEntry(
                        request_id=request.request_id,
                        key_id=request.client.key_id,
                        tier=request.authorization_ceiling.queue_tier,
                        weight=request.authorization_ceiling.queue_weight,
                        authorized_legs=request.authorization_ceiling.authorized_legs,
                    )
                )
                await self._pump_locked()
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            dispatch: DispatchLease | None = None
            async with self._pump_lock:
                queued = self._waiters.pop(request.request_id, None)
                if queued is not None:
                    queued.state = WaiterState.CANCELLED
                    self._queue.cancel(request.request_id)
                if future.done() and not future.cancelled():
                    dispatch = future.result()
                elif not future.done():
                    future.cancel()
            if dispatch is not None and not dispatch.terminal:
                await asyncio.shield(dispatch.cancel_before_send())
            raise

    async def wake(self) -> None:
        async with self._pump_lock:
            await self._pump_locked()

    @staticmethod
    def _denial_cause(denial: ReservationDenied) -> DispatchCause | None:
        if denial.reason in {
            ReservationDenialReason.PARALLEL_EXHAUSTED,
            ReservationDenialReason.RPM_EXHAUSTED,
        }:
            return DispatchCause.CAPACITY
        if denial.reason in {
            ReservationDenialReason.TPM_EXHAUSTED,
            ReservationDenialReason.MONTHLY_QUOTA_EXHAUSTED,
        }:
            return DispatchCause.QUOTA
        return None

    async def _pump_locked(self) -> None:
        for waiter in self._waiters.values():
            waiter.fallback_once = None
        excluded: set[RequestId] = set()
        while self._waiters:
            previews = self._previews()
            eligible_request_ids = frozenset(previews)

            def eligible(
                entry: QueueEntry,
                request_ids: frozenset[RequestId] = eligible_request_ids,
            ) -> bool:
                return entry.request_id in request_ids

            selection = self._queue.propose(
                eligible,
                excluded=frozenset(excluded),
            )
            if selection is None:
                return
            waiter = self._waiters[selection.entry.request_id]
            preview_candidates = previews[selection.entry.request_id]
            candidate_accounts = {
                request_id: tuple(dict.fromkeys(candidate.account_id for candidate in candidates))
                for request_id, candidates in previews.items()
            }
            allowed_accounts = self._queue.unprotected_accounts(
                selection.entry,
                candidate_accounts[selection.entry.request_id],
                candidate_accounts,
            )
            if not allowed_accounts:
                excluded.add(selection.entry.request_id)
                continue

            async with self._activation_gate.hold_dispatch(waiter.request.request_id):
                try:
                    authorization = self._current_authorization(waiter)
                except AdmissionUnavailable as error:
                    self._fail_waiter_locked(waiter, error)
                    continue
                if self._clock.now_ms() >= waiter.effective_deadline_at_ms:
                    self._fail_waiter_locked(
                        waiter,
                        AdmissionUnavailable("queued request deadline elapsed"),
                    )
                    continue
                desired_cause = preview_candidates[0].cause
                if waiter.fallback_once is desired_cause:
                    waiter.fallback_once = None
                view = self._runtime.operational_view()
                waiter.request.attempt_budget.sync_runtime(view)
                candidate = self._route_engine.select(
                    authorization,
                    waiter.request.profile,
                    desired_cause,
                    view,
                    self._routing_context(waiter),
                    excluded_accounts=frozenset(
                        set(candidate_accounts[selection.entry.request_id]) - set(allowed_accounts)
                    ),
                )
                if (
                    candidate is None
                    or all(
                        candidate.authorized_leg != preview.authorized_leg
                        for preview in preview_candidates
                    )
                    or not waiter.request.attempt_budget.can_send(candidate)
                ):
                    excluded.add(selection.entry.request_id)
                    continue
                key_id = waiter.request.client.key_id
                if not self._acquire_active(key_id, authorization.max_concurrency):
                    excluded.add(selection.entry.request_id)
                    continue
                attempt_id = self._attempt_id_factory()
                reservation = ReservationRequest(
                    request_id=waiter.request.request_id,
                    attempt_id=attempt_id,
                    account_id=candidate.account_id,
                    leg_id=candidate.leg_id,
                    deployment_generation_id=candidate.generation_id,
                    runtime_identity=waiter.request.runtime_identity,
                    deadline_at_ms=waiter.effective_deadline_at_ms,
                    profile=waiter.request.profile,
                    input_tokens_upper_bound=waiter.request.profile.input_tokens_max,
                    max_output_tokens=waiter.request.profile.output_tokens_max,
                    max_reasoning_tokens=waiter.request.profile.reasoning_tokens_max,
                    quota_units=self._route_engine.quota_units(candidate.account_id),
                    account_circuit=candidate.account_circuit,
                    circuit=candidate.deployment_circuit,
                )
                try:
                    granted = await self._runtime.try_reserve_async(reservation)
                except BaseException:
                    self._abandon_candidate(candidate)
                    self._release_active(key_id, wake=False)
                    raise
                if isinstance(granted, ReservationDenied):
                    self._release_active(key_id, wake=False)
                    next_cause = self._denial_cause(granted)
                    if next_cause is not None:
                        waiter.fallback_once = next_cause
                        continue
                    excluded.add(selection.entry.request_id)
                    continue
                assert isinstance(granted, ReservationGranted)
                try:
                    await granted.lease.mark_dispatched_async()
                    waiter.request.attempt_budget.record_send(candidate, attempt_id)
                    self._queue.commit(selection)
                except asyncio.CancelledError:
                    await asyncio.shield(
                        granted.lease.finish_async(
                            AttemptResolution(
                                outcome=TerminalOutcome.CLIENT_CANCELLED,
                                release_capacity=True,
                                actual_starts=0,
                                actual_token_units=0,
                                actual_quota_units=0,
                            )
                        )
                    )
                    self._abandon_candidate(candidate)
                    self._release_active(key_id, wake=False)
                    raise
                except BaseException:
                    await granted.lease.finish_async(
                        AttemptResolution(
                            outcome=TerminalOutcome.UPSTREAM_FAILED,
                            release_capacity=True,
                            actual_starts=0,
                            actual_token_units=0,
                            actual_quota_units=0,
                        )
                    )
                    self._abandon_candidate(candidate)
                    self._release_active(key_id, wake=False)
                    raise
                waiter.state = WaiterState.DISPATCHED
                self._waiters.pop(waiter.request.request_id)
                dispatch = DispatchLease(
                    granted.lease,
                    candidate,
                    attempt_id,
                    waiter.request.attempt_budget,
                    partial(self._release_active, key_id),
                    partial(self._abandon_candidate, candidate),
                )
                if waiter.future.cancelled():
                    await dispatch.cancel_before_send()
                elif not waiter.future.done():
                    waiter.future.set_result(dispatch)
            excluded.clear()
