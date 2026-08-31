"""Small real Task 4-7 stack for Task 8 HTTP integration tests."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmmaxxing.adapters.litellm.contract import EffectiveDeployment, load_contract
from llmmaxxing.adapters.litellm.guard import deployment_generation
from llmmaxxing.control.keys import issue_key
from llmmaxxing.core.canonical import bundle_hash, canonical_bundle_bytes
from llmmaxxing.core.ids import (
    AccountId,
    GatewayBootId,
    InstallationId,
    PolicyRevisionId,
    RequestId,
    RouteGroupId,
    RouteLegId,
)
from llmmaxxing.core.models import (
    AuthorizedLeg,
    KeyPolicyRevision,
    LegCapabilities,
    PolicyBundleV1,
    ProviderAccount,
    QuotaDimension,
    RequestProfile,
    RouteGroupRevision,
    RouteLeg,
)
from llmmaxxing.core.reasons import (
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
from llmmaxxing.core.state_machines import AccountState
from llmmaxxing.gateway.app import (
    DispatchTarget,
    RequestLifecycle,
    create_app,
)
from llmmaxxing.gateway.auth import AuthenticatedClient
from llmmaxxing.gateway.ingress import IngressLimits, IngressResources
from llmmaxxing.gateway.journal import AttemptJournal
from llmmaxxing.gateway.profiler import ProfileExecutor
from llmmaxxing.gateway.routing import FailureClassifier, FailureRule
from llmmaxxing.gateway.runtime_state import RuntimeIdentity, RuntimeState
from llmmaxxing.gateway.streaming import AttemptPermitPool, ProcessHTTPClient
from support.fake_litellm import FakeLiteLLM, FaultPlan

NOW_S = 1_800_000_000
NOW_MS = NOW_S * 1_000
PEPPER = b"p" * 32
PEPPER_VERSION = "p1"
ACCOUNT_ID = AccountId("acc_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
GROUP_ID = RouteGroupId("rg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
LEG_ID = RouteLegId("leg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
POLICY_ID = PolicyRevisionId("pol_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


class Clock:
    def __init__(self, value: int = NOW_MS) -> None:
        self.value = value

    def now_ms(self) -> int:
        return self.value


class GenerationGate:
    def permits(self, leg: AuthorizedLeg, backend_manifest_hash: str) -> bool:
        return (
            leg.generation_id == TARGET.generation.generation_id
            and len(backend_manifest_hash) == 64
        )


class ActivationGate:
    @asynccontextmanager
    async def hold_dispatch(self, request_id: RequestId):  # type: ignore[no-untyped-def]
        assert request_id
        yield


@dataclass(frozen=True, slots=True)
class RuntimeIdentityProvider:
    value: RuntimeIdentity

    def current_runtime_identity(self) -> RuntimeIdentity:
        return self.value


@dataclass(frozen=True, slots=True)
class DeploymentResolver:
    target: DispatchTarget

    def resolve(self, generation_id):  # type: ignore[no-untyped-def]
        if generation_id != self.target.generation.generation_id:
            raise LookupError("unknown deployment generation")
        return self.target


class BackendAuthorization:
    def headers(self) -> dict[str, str]:
        return {"authorization": "Bearer backend-fixture"}


@dataclass(slots=True)
class Lifecycle(RequestLifecycle):
    capacity: int
    events: list[tuple[str, object]] = field(default_factory=list)
    released: bool = False

    async def profile_accepted(self, profile: RequestProfile) -> None:
        self.events.append(("profile", profile))

    async def queued(self) -> None:
        self.events.append(("queued", None))

    async def attempt_started(self, lease, *, shadow: bool) -> None:  # type: ignore[no-untyped-def]
        self.events.append(("attempt_started", (lease.attempt_id, shadow)))

    async def attempt_finished(
        self,
        lease,  # type: ignore[no-untyped-def]
        outcome: TerminalOutcome,
        *,
        uncertain: bool,
    ) -> None:
        self.events.append(("attempt_finished", (lease.attempt_id, outcome, uncertain)))

    async def finished(self, outcome: TerminalOutcome) -> None:
        self.events.append(("finished", outcome))

    async def release(self) -> None:
        if self.released:
            raise AssertionError("lifecycle released twice")
        self.released = True
        self.events.append(("release", None))


@dataclass(slots=True)
class LifecycleCapacity:
    available: bool = True
    reservations: list[tuple[RequestId, AuthenticatedClient, int]] = field(default_factory=list)
    lifecycles: list[Lifecycle] = field(default_factory=list)

    async def reserve(
        self,
        request_id: RequestId,
        client: AuthenticatedClient,
        events: int,
    ) -> RequestLifecycle | None:
        self.reservations.append((request_id, client, events))
        if not self.available:
            return None
        lifecycle = Lifecycle(events)
        self.lifecycles.append(lifecycle)
        return lifecycle


class AuthView:
    def __init__(self, bundle: PolicyBundleV1) -> None:
        self.key_index = {record.key_id: record for record in bundle.keys}
        self.legacy_key_index: dict[str, object] = {}
        self.applied_bundle_generation = bundle.generation
        self.applied_bundle_hash = bundle_hash(canonical_bundle_bytes(bundle))
        self.denied_key_ids: frozenset[str] = frozenset()
        self.accepted_peppers = {PEPPER_VERSION: PEPPER}
        self.trusted_now_s = NOW_S


@dataclass(frozen=True, slots=True)
class AuthViews:
    view: AuthView

    def current_auth_view(self) -> AuthView:
        return self.view


CONTRACT = load_contract()
DEPLOYMENT = EffectiveDeployment(
    runtime_id="runtime-primary",
    hidden_alias="fixture/chat",
    mode="chat",
    execution={"model": "openai/fixture-chat", "api_base": "https://provider.invalid/v1"},
    capabilities={"endpoints": [endpoint.name for endpoint in CONTRACT.endpoints]},
    context={"max_input_tokens": 40_000_000, "max_output_tokens": 4096},
    defaults={},
    pricing={"input_per_token": "0", "output_per_token": "0"},
    account_id=ACCOUNT_ID,
    account_binding="fixture-primary",
    credential_field="api_key",
    credential_fingerprint="hcf1_" + "a" * 64,
    credential_epoch=1,
)


@dataclass(frozen=True, slots=True)
class _TargetSeed:
    generation: Any


TARGET = _TargetSeed(deployment_generation(DEPLOYMENT, CONTRACT))


def quota(value: int) -> QuotaDimension:
    return QuotaDimension(status=QuotaDimensionStatus.KNOWN, value=value)


def make_bundle(*, shadow: bool = False) -> tuple[PolicyBundleV1, str]:
    caps = LegCapabilities(
        endpoints=tuple(EndpointKind(endpoint.name) for endpoint in CONTRACT.endpoints),
        modalities=(
            Modality.TEXT,
            Modality.EMBEDDING,
            Modality.RERANK,
            Modality.AUDIO_SPEECH,
            Modality.AUDIO_TRANSCRIPTION,
            Modality.IMAGE,
        ),
        context_tokens=40_000_000,
        tools=True,
        forced_tool=True,
        response_schema=True,
        streaming=True,
        reasoning=True,
        history_continuation=True,
        shadow=False,
    )
    primary = RouteLeg(
        leg_id=LEG_ID,
        order=10,
        triggers=(RouteTrigger.PRIMARY,),
        account_id=ACCOUNT_ID,
        generation_id=TARGET.generation.generation_id,
        capabilities=caps,
    )
    legs = [primary]
    authorized = [
        AuthorizedLeg(
            leg_id=primary.leg_id,
            account_id=primary.account_id,
            generation_id=primary.generation_id,
            order=primary.order,
            allowed_triggers=primary.triggers,
            capabilities=primary.capabilities,
        )
    ]
    if shadow:
        shadow_leg = RouteLeg(
            leg_id=RouteLegId("leg_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            order=20,
            triggers=(RouteTrigger.SHADOW,),
            account_id=ACCOUNT_ID,
            generation_id=TARGET.generation.generation_id,
            capabilities=caps.model_copy(update={"shadow": True}),
        )
        legs.append(shadow_leg)
        authorized.append(
            AuthorizedLeg(
                leg_id=shadow_leg.leg_id,
                account_id=shadow_leg.account_id,
                generation_id=shadow_leg.generation_id,
                order=shadow_leg.order,
                allowed_triggers=shadow_leg.triggers,
                capabilities=shadow_leg.capabilities,
            )
        )
    route_group = RouteGroupRevision(
        route_group_id=GROUP_ID,
        name="deepseek-v4-flash",
        strategy=RouteStrategy.ORDERED_CAPACITY,
        legs=tuple(legs),
    )
    policy = KeyPolicyRevision(
        policy_id=POLICY_ID,
        name="fixture",
        route_group_ids=(GROUP_ID,),
        authorized_legs=tuple(authorized),
        queue_tier=10,
        queue_weight=4,
        max_concurrency=4,
        max_waiters=16,
        deadline_ms=120_000,
    )
    issued = issue_key(
        policy,
        None,
        pepper=PEPPER,
        pepper_version=PEPPER_VERSION,
        now_s=NOW_S,
    )
    token = issued.reveal_once().value
    account = ProviderAccount(
        account_id=ACCOUNT_ID,
        display_name="fixture",
        connection="litellm:fixture",
        provider_token="fixture",
        binding_ref="fixture-primary",
        credential_fingerprint=DEPLOYMENT.credential_fingerprint,
        credential_epoch=1,
        parallel_limit=quota(5),
        local_parallel_ceiling=128,
        rpm_limit=quota(60),
        rpm_window_seconds=60,
        tpm_limit=quota(100_000_000),
        tpm_window_seconds=60,
        monthly_quota_units=quota(100_000_000),
        quota_units_per_attempt=1,
        monthly_reset_at_ms=2_000_000_000_000,
        state=AccountState.ACTIVE,
    )
    bundle = PolicyBundleV1(
        schema_version=1,
        generation=1,
        min_reader="1.0",
        required_features=(
            RequiredFeature.ORDERED_CAPACITY,
            RequiredFeature.WEIGHTED_FAIR_QUEUE,
        ),
        keys=(issued.record,),
        policies=(policy,),
        accounts=(account,),
        route_groups=(route_group,),
        backend_manifest_hash="e" * 64,
    )
    return bundle, token


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
                cause=FailureCause.CAPACITY,
                scope=FailureScope.ACCOUNT,
                status_codes=frozenset({403}),
                message_contains=("exceeded the maximum number of parallel requests",),
            ),
            FailureRule(
                cause=FailureCause.TRANSIENT_FAILURE,
                scope=FailureScope.DEPLOYMENT,
                status_codes=frozenset({500, 502, 503}),
                error_codes=frozenset({"upstream_unavailable"}),
            ),
        )
    )


@dataclass(slots=True)
class Stack:
    app: Any
    token: str
    fake: FakeLiteLLM
    lifecycle_capacity: LifecycleCapacity
    ingress: IngressResources
    permits: AttemptPermitPool
    runtime: RuntimeState
    journal: AttemptJournal
    profiler: ProfileExecutor
    http: ProcessHTTPClient

    async def close(self) -> None:
        await self.app.aclose()
        self.journal.close()


async def make_stack(
    tmp_path: Path,
    *plans: FaultPlan,
    limits: IngressLimits | None = None,
    lifecycle_available: bool = True,
    shadow: bool = False,
) -> Stack:
    bundle, token = make_bundle(shadow=shadow)
    clock = Clock()
    journal = AttemptJournal.create(tmp_path / "journal", clock=clock, group_commit_delay_ms=0)
    runtime = RuntimeState(bundle.accounts, journal=journal, clock=clock)
    fake = FakeLiteLLM.with_plans(*plans)
    http = ProcessHTTPClient(
        "https://litellm.invalid",
        transport=fake.transport(),
        connect_timeout_s=0.1,
        write_timeout_s=0.1,
        read_inactivity_timeout_s=0.05,
        pool_timeout_s=0.1,
    )
    profiler = ProfileExecutor()
    ingress = IngressResources(limits or IngressLimits())
    permits = AttemptPermitPool()
    lifecycle_capacity = LifecycleCapacity(available=lifecycle_available)
    identity = RuntimeIdentity(
        installation_id=InstallationId("inst_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        dispatcher_fence=1,
        boot_id=GatewayBootId("boot_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        bundle_generation=bundle.generation,
        bundle_hash=bundle_hash(canonical_bundle_bytes(bundle)),
    )
    target = DispatchTarget(
        deployment=DEPLOYMENT,
        generation=TARGET.generation,
        backend_manifest="bm1_" + "f" * 64,
    )
    app = create_app(
        contract=CONTRACT,
        bundle=bundle,
        runtime=runtime,
        activation_gate=ActivationGate(),
        auth_view_provider=AuthViews(AuthView(bundle)),
        generation_gate=GenerationGate(),
        runtime_identity_provider=RuntimeIdentityProvider(identity),
        deployment_resolver=DeploymentResolver(target),
        lifecycle_capacity=lifecycle_capacity,
        backend_authorization=BackendAuthorization(),
        clock=clock,
        failure_classifier=classifier(),
        http=http,
        profiler=profiler,
        ingress=ingress,
        permits=permits,
    )
    return Stack(
        app,
        token,
        fake,
        lifecycle_capacity,
        ingress,
        permits,
        runtime,
        journal,
        profiler,
        http,
    )


@dataclass(frozen=True, slots=True)
class ASGIResponse:
    status: int
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes
    messages: tuple[dict[str, Any], ...]
    receive_calls: int


def chat_body(*, model: str = "deepseek-v4-flash", stream: bool = False) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "stream": stream,
        },
        separators=(",", ":"),
    ).encode()


async def call_app(
    app: Any,
    token: str | None,
    *,
    path: str = "/v1/chat/completions",
    method: str = "POST",
    body: bytes | None = None,
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
    receive_messages: list[dict[str, Any]] | None = None,
    send_hook: Any = None,
    query_string: bytes = b"",
    raw_path: bytes | None = None,
) -> ASGIResponse:
    selected_body = chat_body() if body is None else body
    headers = [
        (b"host", b"gateway.invalid"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(selected_body)).encode()),
    ]
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    headers.extend(extra_headers)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": raw_path if raw_path is not None else path.encode(),
        "query_string": query_string,
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("gateway.invalid", 443),
    }
    pending = list(
        receive_messages
        if receive_messages is not None
        else [{"type": "http.request", "body": selected_body, "more_body": False}]
    )
    sent: list[dict[str, Any]] = []
    receive_calls = 0

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if send_hook is not None:
            await send_hook(message)
        sent.append(message)

    await app(scope, receive, send)
    starts = [message for message in sent if message["type"] == "http.response.start"]
    assert len(starts) == 1
    bodies = [
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    ]
    return ASGIResponse(
        starts[0]["status"],
        tuple(starts[0]["headers"]),
        b"".join(bodies),
        tuple(sent),
        receive_calls,
    )
