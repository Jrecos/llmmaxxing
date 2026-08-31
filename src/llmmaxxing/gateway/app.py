"""Task-8 ASGI Gateway composition over the mandatory Task 4-7 contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from llmmaxxing.adapters.litellm.contract import (
    AdapterContract,
    DeploymentGenerationFingerprint,
    EffectiveDeployment,
)
from llmmaxxing.adapters.litellm.dispatch import (
    DispatchError,
    prepare_dispatch,
    reconcile_dispatch,
)
from llmmaxxing.core.ids import DeploymentGenerationId, RequestId
from llmmaxxing.core.models import (
    PolicyBundleV1,
    RequestAuthorizationCeiling,
    RequestProfile,
)
from llmmaxxing.core.reasons import DispatchCause, FailureCause, RouteTrigger, TerminalOutcome
from llmmaxxing.gateway.auth import (
    AuthenticatedClient,
    ClientAuthenticationError,
    parse_client_key,
    verify_client_key,
)
from llmmaxxing.gateway.ingress import (
    ASGIReceive,
    IngressError,
    IngressRequest,
    IngressResources,
    RetainedBody,
    read_retained_body,
    validate_http_request,
)
from llmmaxxing.gateway.profiler import ProfileError, ProfileExecutor
from llmmaxxing.gateway.routing import (
    AttemptBudget,
    CircuitController,
    FailureClassifier,
    FailureObservation,
    GenerationOperationalGate,
    RouteEngine,
)
from llmmaxxing.gateway.runtime_state import AttemptResolution, RuntimeIdentity, RuntimeState
from llmmaxxing.gateway.scheduler import (
    ActivationGate,
    AdmissionClock,
    AdmissionController,
    AdmissionRequest,
    AdmissionUnavailable,
    AuthViewProvider,
    DispatchLease,
)
from llmmaxxing.gateway.streaming import (
    ASGISend,
    AttemptPermitPool,
    DownstreamDisconnected,
    DownstreamStreamError,
    PermitClass,
    ProcessHTTPClient,
    ResponseStart,
    UpstreamStreamError,
    drain_raw_response,
    read_prestart_error,
    relay_buffered_response,
    relay_raw_response,
)


class RequestLifecycle(Protocol):
    """Task 10 must durably implement this exact reserved request lifecycle."""

    async def profile_accepted(self, profile: RequestProfile) -> None: ...

    async def queued(self) -> None: ...

    async def attempt_started(self, lease: DispatchLease, *, shadow: bool) -> None: ...

    async def attempt_finished(
        self,
        lease: DispatchLease,
        outcome: TerminalOutcome,
        *,
        uncertain: bool,
    ) -> None: ...

    async def finished(self, outcome: TerminalOutcome) -> None: ...

    async def release(self) -> None: ...


class LifecycleCapacity(Protocol):
    """Mandatory durable event-capacity admission; there is deliberately no fallback."""

    async def reserve(
        self,
        request_id: RequestId,
        client: AuthenticatedClient,
        events: int,
    ) -> RequestLifecycle | None: ...


class RuntimeIdentityProvider(Protocol):
    def current_runtime_identity(self) -> RuntimeIdentity: ...


@dataclass(frozen=True, slots=True)
class DispatchTarget:
    deployment: EffectiveDeployment
    generation: DeploymentGenerationFingerprint
    backend_manifest: str

    def __post_init__(self) -> None:
        if not str(self.generation.generation_id).startswith("dg1_"):
            raise ValueError("invalid deployment generation")
        if not self.backend_manifest.startswith("bm1_") or len(self.backend_manifest) != 68:
            raise ValueError("invalid backend manifest revision")


class DeploymentResolver(Protocol):
    def resolve(self, generation_id: DeploymentGenerationId) -> DispatchTarget: ...


class BackendAuthorization(Protocol):
    """Supplies only the narrow LiteLLM inference credential at send time."""

    def headers(self) -> Mapping[str, str]: ...


class GatewayClock(AdmissionClock, Protocol):
    pass


@dataclass(frozen=True, slots=True)
class _AttemptResult:
    terminal: bool
    outcome: TerminalOutcome
    alternate: DispatchCause | None = None


@dataclass(slots=True)
class _DeadlineState:
    monotonic_at: float
    wall_at_ms: int
    response_start: ResponseStart
    expired: bool = False

    @property
    def response_started(self) -> bool:
        return self.response_start.started


@dataclass(slots=True)
class _ShadowState:
    decided: bool = False
    task: asyncio.Task[None] | None = None


_TRIGGER_FOR_CAUSE = {
    DispatchCause.CAPACITY: RouteTrigger.CAPACITY_SPILL,
    DispatchCause.FAILURE: RouteTrigger.FAILURE_FALLBACK,
    DispatchCause.QUOTA: RouteTrigger.QUOTA_FALLBACK,
}

_REQUEST_HOP_HEADERS = frozenset(
    {
        "authorization",
        "connection",
        "content-length",
        "expect",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


async def _shielded(awaitable: Any) -> Any:
    task = asyncio.create_task(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


async def _send_error(
    send: ASGISend,
    status: int,
    code: str,
    response_start: ResponseStart | None = None,
) -> None:
    body = json.dumps(
        {"error": {"type": "llmmaxxing_error", "code": code}},
        separators=(",", ":"),
    ).encode()
    start = response_start or ResponseStart()
    try:
        await start.send(
            send,
            status=status,
            headers=[
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"cache-control", b"no-store"),
            ],
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
    except asyncio.CancelledError:
        raise
    except Exception:
        return


def _bearer(request: IngressRequest) -> str:
    values = request.values(b"authorization")
    if len(values) != 1:
        raise ClientAuthenticationError()
    try:
        value = values[0].decode("ascii")
    except UnicodeDecodeError:
        raise ClientAuthenticationError() from None
    scheme, separator, token = value.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token or " " in token:
        raise ClientAuthenticationError()
    return token


def _upstream_headers(
    request: IngressRequest,
    backend: BackendAuthorization,
) -> dict[str, str]:
    connection_tokens: set[str] = set()
    for header_name, header_value in request.headers:
        if header_name == b"connection":
            connection_tokens.update(
                token.strip().lower() for token in header_value.decode("ascii", "ignore").split(",")
            )
    headers: dict[str, str] = {}
    for raw_name, raw_value in request.headers:
        name = raw_name.decode("ascii").lower()
        if (
            name in _REQUEST_HOP_HEADERS
            or name in connection_tokens
            or name.startswith("x-litellm-")
            or name.startswith("x-llmmaxxing-")
            or name in {"traceparent", "tracestate", "baggage"}
        ):
            continue
        headers[name] = raw_value.decode("latin-1")
    trusted = backend.headers()
    if not 1 <= len(trusted) <= 4:
        raise RuntimeError("backend authorization returned an invalid header set")
    for trusted_name, value in trusted.items():
        lowered = trusted_name.lower()
        if (
            lowered in _REQUEST_HOP_HEADERS - {"authorization"}
            or lowered in connection_tokens
            or not value
            or "\r" in value
            or "\n" in value
        ):
            raise RuntimeError("backend authorization returned an unsafe header")
        headers[lowered] = value
    if "authorization" not in headers:
        raise RuntimeError("backend inference authorization is absent")
    return headers


def _failure_observation(status: int, body: bytes) -> FailureObservation:
    error_code: str | None = None
    message = ""
    try:
        document = json.loads(body)
        error = document.get("error") if isinstance(document, dict) else None
        if isinstance(error, dict):
            raw_code = error.get("type") or error.get("code")
            raw_message = error.get("message")
            if isinstance(raw_code, str):
                error_code = raw_code[:160]
            if isinstance(raw_message, str):
                message = raw_message[:4096]
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return FailureObservation(
        status_code=status,
        error_code=error_code,
        message=message,
        pre_response_bytes=True,
    )


class GatewayApp:
    """Pure ASGI callable; Task 9 owns Uvicorn, singleton startup, and readiness."""

    def __init__(
        self,
        *,
        contract: AdapterContract,
        bundle: PolicyBundleV1,
        runtime: RuntimeState,
        activation_gate: ActivationGate,
        auth_view_provider: AuthViewProvider,
        generation_gate: GenerationOperationalGate,
        runtime_identity_provider: RuntimeIdentityProvider,
        deployment_resolver: DeploymentResolver,
        lifecycle_capacity: LifecycleCapacity,
        backend_authorization: BackendAuthorization,
        clock: GatewayClock,
        failure_classifier: FailureClassifier,
        http: ProcessHTTPClient,
        profiler: ProfileExecutor,
        ingress: IngressResources,
        permits: AttemptPermitPool,
    ) -> None:
        self.contract = contract
        self.runtime = runtime
        self.auth_view_provider = auth_view_provider
        self.runtime_identity_provider = runtime_identity_provider
        self.deployment_resolver = deployment_resolver
        self.lifecycle_capacity = lifecycle_capacity
        self.backend_authorization = backend_authorization
        self.clock = clock
        self.failure_classifier = failure_classifier
        self.http = http
        self.profiler = profiler
        self.ingress = ingress
        self.permits = permits
        self.route_engine = RouteEngine(bundle, generation_gate)
        self.circuits = CircuitController(runtime)
        self.admission = AdmissionController(
            self.route_engine,
            runtime,
            activation_gate,
            auth_view_provider,
            self.circuits,
            clock=clock,
            max_waiters_global=64,
        )
        self._route_groups = {
            group.name: group.route_group_id for group in self.route_engine.bundle.route_groups
        }
        self._closed = False

    async def __call__(
        self, scope: Mapping[str, Any], receive: ASGIReceive, send: ASGISend
    ) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope_type == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": "unsupported"})
            return
        inbound = await self.ingress.try_inbound()
        if inbound is None:
            await _send_error(send, 503, "inbound_limit")
            return
        try:
            await self._http(scope, receive, send)
        finally:
            await inbound.release()

    async def _lifespan(self, receive: ASGIReceive, send: ASGISend) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _http(
        self,
        scope: Mapping[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        response_start = ResponseStart()

        async def send_error(status: int, code: str) -> None:
            await _send_error(send, status, code, response_start)

        if self._closed:
            await send_error(503, "gateway_closed")
            return
        try:
            ingress_request = validate_http_request(scope, self.contract, self.ingress.limits)
        except IngressError as error:
            await send_error(error.status, error.code)
            return
        preauth = await self.ingress.try_preauth()
        if preauth is None:
            await send_error(503, "preauth_limit")
            return
        lifecycle: RequestLifecycle | None = None
        body: RetainedBody | None = None
        final_recorded = False
        deadline: _DeadlineState | None = None
        deadline_handle: asyncio.TimerHandle | None = None
        shadow = _ShadowState()
        try:
            try:
                parsed = parse_client_key(_bearer(ingress_request))
                client = verify_client_key(parsed, self.auth_view_provider.current_auth_view())
            except Exception as error:
                if not isinstance(error, ClientAuthenticationError):
                    await send_error(503, "auth_state_unavailable")
                else:
                    await send_error(401, "invalid_client_key")
                return
            deadline_ms = self.route_engine.policy_deadline_ms(client)
            loop = asyncio.get_running_loop()
            deadline = _DeadlineState(
                monotonic_at=loop.time() + deadline_ms / 1000,
                wall_at_ms=self.clock.now_ms() + deadline_ms,
                response_start=response_start,
            )
            request_task = asyncio.current_task()
            assert request_task is not None

            def expire_request() -> None:
                assert deadline is not None
                deadline.expired = True
                request_task.cancel()

            deadline_handle = loop.call_at(deadline.monotonic_at, expire_request)
            request_id = RequestId.new()
            lifecycle_events = 12 if self.route_engine.client_authorizes_shadow(client) else 10
            lifecycle = await self.lifecycle_capacity.reserve(
                request_id,
                client,
                lifecycle_events,
            )
            if lifecycle is None:
                await send_error(503, "lifecycle_capacity")
                return
            await preauth.release()
            preauth = None

            try:
                body = await read_retained_body(
                    receive,
                    ingress_request,
                    client.key_id,
                    self.ingress,
                )
                profile = await self.profiler.profile(
                    ingress_request.endpoint,
                    body,
                    ingress_request.content_type,
                    self._route_groups,
                    deadline_ms,
                    client.key_id,
                )
            except IngressError as error:
                await lifecycle.finished(
                    TerminalOutcome.CLIENT_CANCELLED
                    if error.status == 499
                    else TerminalOutcome.BACKPRESSURE_REJECTED
                )
                final_recorded = True
                await send_error(error.status, error.code)
                return
            except ProfileError as error:
                await lifecycle.finished(TerminalOutcome.UNSUPPORTED_REQUEST)
                final_recorded = True
                await send_error(error.status, error.code)
                return
            await lifecycle.profile_accepted(profile)
            authorization = self.route_engine.authorize(client, profile)
            if not authorization.authorized_legs:
                await lifecycle.finished(TerminalOutcome.AUTHZ_DENIED)
                final_recorded = True
                await send_error(403, "model_not_authorized")
                return
            identity = self.runtime_identity_provider.current_runtime_identity()
            if (
                identity.bundle_generation != authorization.bundle_generation
                or identity.bundle_hash != authorization.bundle_hash
            ):
                await lifecycle.finished(TerminalOutcome.AUTH_STATE_UNAVAILABLE)
                final_recorded = True
                await send_error(503, "runtime_identity_mismatch")
                return
            attempt_budget = AttemptBudget(request_id)
            cause = DispatchCause.PRIMARY
            await lifecycle.queued()
            while True:
                admission_request = AdmissionRequest(
                    request_id=request_id,
                    client=client,
                    profile=profile,
                    authorization_ceiling=authorization,
                    runtime_identity=identity,
                    deadline_at_ms=deadline.wall_at_ms,
                    cause=cause,
                    attempt_budget=attempt_budget,
                )
                try:
                    dispatch = await self.admission.acquire(admission_request)
                except AdmissionUnavailable:
                    await lifecycle.finished(TerminalOutcome.ROUTE_UNAVAILABLE)
                    final_recorded = True
                    await send_error(503, "route_unavailable")
                    return
                result = await self._attempt(
                    ingress_request,
                    body,
                    profile,
                    client,
                    dispatch,
                    authorization,
                    lifecycle,
                    receive,
                    send,
                    deadline,
                    admission_request,
                    shadow,
                )
                if result.terminal:
                    if shadow.task is not None:
                        if result.outcome is TerminalOutcome.CLIENT_CANCELLED:
                            shadow.task.cancel()
                        await asyncio.gather(shadow.task, return_exceptions=True)
                    await lifecycle.finished(result.outcome)
                    final_recorded = True
                    return
                assert result.alternate is not None
                cause = result.alternate
        except asyncio.CancelledError:
            if shadow.task is not None and not shadow.task.done():
                shadow.task.cancel()
                await asyncio.gather(shadow.task, return_exceptions=True)
            outcome = (
                TerminalOutcome.DEADLINE_EXCEEDED
                if deadline is not None and deadline.expired
                else TerminalOutcome.CLIENT_CANCELLED
            )
            if lifecycle is not None and not final_recorded:
                with suppress(Exception):
                    await _shielded(lifecycle.finished(outcome))
                final_recorded = True
            if deadline is not None and deadline.expired:
                if not deadline.response_started:
                    await send_error(504, "deadline_exceeded")
                return
            raise
        except Exception:
            if lifecycle is not None and not final_recorded:
                with suppress(Exception):
                    await lifecycle.finished(TerminalOutcome.UPSTREAM_FAILED)
                final_recorded = True
            if deadline is None or not deadline.response_started:
                await send_error(500, "gateway_failure")
        finally:
            if deadline_handle is not None:
                deadline_handle.cancel()
            if shadow.task is not None and not shadow.task.done():
                shadow.task.cancel()
                await asyncio.gather(shadow.task, return_exceptions=True)
            if preauth is not None:
                await _shielded(preauth.release())
            if body is not None:
                await _shielded(body.release())
            if lifecycle is not None:
                try:
                    if not final_recorded:
                        with suppress(Exception):
                            await _shielded(lifecycle.finished(TerminalOutcome.UPSTREAM_FAILED))
                finally:
                    with suppress(Exception):
                        await _shielded(lifecycle.release())

    @staticmethod
    def _authorization_has_alternate(
        dispatch: DispatchLease,
        cause: DispatchCause | None,
        authorization: RequestAuthorizationCeiling,
    ) -> bool:
        trigger = _TRIGGER_FOR_CAUSE.get(cause) if cause is not None else None
        return trigger is not None and any(
            leg.leg_id != dispatch.candidate.leg_id and trigger in leg.allowed_triggers
            for leg in authorization.authorized_legs
        )

    async def _attempt(
        self,
        ingress_request: IngressRequest,
        body: RetainedBody,
        profile: RequestProfile,
        client: AuthenticatedClient,
        dispatch: DispatchLease,
        authorization: RequestAuthorizationCeiling,
        lifecycle: RequestLifecycle,
        receive: ASGIReceive,
        send: ASGISend,
        deadline: _DeadlineState,
        admission_request: AdmissionRequest,
        shadow: _ShadowState,
    ) -> _AttemptResult:
        permit = None
        response: httpx.Response | None = None
        provider_send_started = False
        attempt_event_recorded = False
        pending_outcome: TerminalOutcome | None = None
        pending_uncertain = False

        async def record_attempt(
            outcome: TerminalOutcome,
            *,
            uncertain: bool,
        ) -> None:
            nonlocal attempt_event_recorded
            if attempt_event_recorded:
                return
            await _shielded(lifecycle.attempt_finished(dispatch, outcome, uncertain=uncertain))
            attempt_event_recorded = True

        try:
            permit = await self.permits.acquire(PermitClass.FOREGROUND)
            await _shielded(lifecycle.attempt_started(dispatch, shadow=False))
            try:
                target = self.deployment_resolver.resolve(dispatch.candidate.generation_id)
                if target.generation.generation_id != dispatch.candidate.generation_id:
                    raise ValueError("resolved deployment generation mismatch")
                prepared = prepare_dispatch(
                    self.contract,
                    endpoint=profile.endpoint.value,
                    deployment=target.deployment,
                    generation=target.generation,
                    backend_manifest=target.backend_manifest,
                )
                rewritten = await self.profiler.rewrite(
                    body,
                    ingress_request.content_type,
                    prepared,
                    self.contract.known_litellm_params,
                    client.key_id,
                )
                headers = _upstream_headers(ingress_request, self.backend_authorization)
                provider_send_started = True
                response = await self.http.send(
                    prepared.method,
                    prepared.path,
                    headers=headers,
                    content=rewritten,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                pending_outcome = TerminalOutcome.UPSTREAM_FAILED
                await dispatch.fail_before_send()
                await record_attempt(
                    TerminalOutcome.UPSTREAM_FAILED,
                    uncertain=False,
                )
                fallback_cause = DispatchCause.FAILURE
                if self._authorization_has_alternate(dispatch, fallback_cause, authorization):
                    return _AttemptResult(False, TerminalOutcome.UPSTREAM_FAILED, fallback_cause)
                await _send_error(send, 502, "upstream_connect_failure", deadline.response_start)
                return _AttemptResult(True, TerminalOutcome.UPSTREAM_FAILED)
            except (httpx.HTTPError, OSError):
                pending_outcome = TerminalOutcome.UPSTREAM_FAILED
                pending_uncertain = True
                await dispatch.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.UPSTREAM_FAILED,
                        release_capacity=False,
                    )
                )
                await record_attempt(
                    TerminalOutcome.UPSTREAM_FAILED,
                    uncertain=True,
                )
                await _send_error(send, 502, "upstream_transport_failure", deadline.response_start)
                return _AttemptResult(True, TerminalOutcome.UPSTREAM_FAILED)
            except (DispatchError, ProfileError, ValueError, RuntimeError):
                pending_outcome = TerminalOutcome.UPSTREAM_FAILED
                await dispatch.fail_before_send()
                await record_attempt(
                    TerminalOutcome.UPSTREAM_FAILED,
                    uncertain=False,
                )
                await _send_error(send, 502, "dispatch_preparation_failed", deadline.response_start)
                return _AttemptResult(True, TerminalOutcome.UPSTREAM_FAILED)

            if not 200 <= response.status_code < 300:
                try:
                    error_body = await read_prestart_error(
                        response,
                        read_inactivity_timeout_s=self.http.config.read_inactivity_timeout_s,
                    )
                    dispatch.lease.provider_send_completed()
                except UpstreamStreamError:
                    pending_outcome = TerminalOutcome.UPSTREAM_FAILED
                    pending_uncertain = True
                    await dispatch.finish_async(
                        AttemptResolution(
                            outcome=TerminalOutcome.UPSTREAM_FAILED,
                            release_capacity=False,
                        )
                    )
                    await record_attempt(
                        TerminalOutcome.UPSTREAM_FAILED,
                        uncertain=True,
                    )
                    await _send_error(
                        send, 502, "upstream_error_read_failed", deadline.response_start
                    )
                    return _AttemptResult(True, TerminalOutcome.UPSTREAM_FAILED)
                observation = _failure_observation(response.status_code, error_body)
                classification = self.failure_classifier.classify(observation)
                self.circuits.open(
                    dispatch.candidate.account_id,
                    dispatch.candidate.generation_id,
                    classification,
                    now_ms=self.clock.now_ms(),
                )
                pending_outcome = TerminalOutcome.UPSTREAM_FAILED
                await dispatch.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.UPSTREAM_FAILED,
                        release_capacity=True,
                        actual_starts=1,
                        actual_token_units=(
                            0 if classification.cause is FailureCause.CAPACITY else None
                        ),
                        actual_quota_units=(
                            0 if classification.cause is FailureCause.CAPACITY else None
                        ),
                    )
                )
                await record_attempt(
                    TerminalOutcome.UPSTREAM_FAILED,
                    uncertain=False,
                )
                alternate = classification.dispatch_cause
                if self._authorization_has_alternate(dispatch, alternate, authorization):
                    return _AttemptResult(False, TerminalOutcome.UPSTREAM_FAILED, alternate)
                try:
                    await relay_buffered_response(
                        response,
                        error_body,
                        send,
                        deadline.response_start,
                    )
                except DownstreamStreamError:
                    return _AttemptResult(True, TerminalOutcome.CLIENT_CANCELLED)
                return _AttemptResult(True, TerminalOutcome.UPSTREAM_FAILED)

            try:
                reconcile_dispatch(
                    self.contract,
                    prepared,
                    status_code=response.status_code,
                    headers=response.headers,
                    response_started=False,
                )
            except DispatchError:
                dispatch.lease.provider_send_completed()
                pending_outcome = TerminalOutcome.UPSTREAM_FAILED
                await dispatch.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.UPSTREAM_FAILED,
                        release_capacity=True,
                        actual_starts=1,
                    )
                )
                await record_attempt(
                    TerminalOutcome.UPSTREAM_FAILED,
                    uncertain=False,
                )
                await _send_error(send, 502, "deployment_receipt_mismatch", deadline.response_start)
                return _AttemptResult(True, TerminalOutcome.UPSTREAM_FAILED)

            if not shadow.decided:
                shadow.decided = True
                shadow_permit = await self.permits.try_shadow()
                if shadow_permit is not None:
                    shadow.task = asyncio.create_task(
                        self._shadow_attempt(
                            admission_request,
                            ingress_request,
                            body,
                            profile,
                            client,
                            lifecycle,
                            shadow_permit,
                            deadline,
                        )
                    )

            dispatch.mark_response_started()
            try:
                await relay_raw_response(
                    response,
                    receive,
                    send,
                    deadline.response_start,
                    read_inactivity_timeout_s=self.http.config.read_inactivity_timeout_s,
                )
            except DownstreamDisconnected:
                pending_outcome = TerminalOutcome.CLIENT_CANCELLED
                pending_uncertain = True
                await dispatch.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.CLIENT_CANCELLED,
                        release_capacity=False,
                    )
                )
                await record_attempt(
                    TerminalOutcome.CLIENT_CANCELLED,
                    uncertain=True,
                )
                return _AttemptResult(True, TerminalOutcome.CLIENT_CANCELLED)
            except (UpstreamStreamError, DownstreamStreamError):
                pending_outcome = TerminalOutcome.RESPONSE_STREAM_FAILED
                pending_uncertain = True
                await dispatch.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.RESPONSE_STREAM_FAILED,
                        release_capacity=False,
                    )
                )
                await record_attempt(
                    TerminalOutcome.RESPONSE_STREAM_FAILED,
                    uncertain=True,
                )
                return _AttemptResult(True, TerminalOutcome.RESPONSE_STREAM_FAILED)
            dispatch.lease.provider_send_completed()
            pending_outcome = TerminalOutcome.COMPLETED
            await dispatch.finish_async(
                AttemptResolution(
                    outcome=TerminalOutcome.COMPLETED,
                    release_capacity=True,
                    actual_starts=1,
                    actual_token_units=dispatch.lease.request.total_token_upper_bound,
                    actual_quota_units=dispatch.lease.request.quota_units,
                )
            )
            await record_attempt(
                TerminalOutcome.COMPLETED,
                uncertain=False,
            )
            return _AttemptResult(True, TerminalOutcome.COMPLETED)
        except asyncio.CancelledError:
            outcome = (
                TerminalOutcome.DEADLINE_EXCEEDED
                if deadline.expired
                else TerminalOutcome.CLIENT_CANCELLED
            )
            if dispatch.terminal:
                uncertain = pending_uncertain
                event_outcome = pending_outcome or outcome
            else:
                uncertain = provider_send_started
                event_outcome = outcome
                if uncertain:
                    await _shielded(
                        dispatch.finish_async(
                            AttemptResolution(
                                outcome=outcome,
                                release_capacity=False,
                            )
                        )
                    )
                else:
                    await _shielded(dispatch.fail_before_send())
            if not attempt_event_recorded:
                await _shielded(record_attempt(event_outcome, uncertain=uncertain))
            raise
        except BaseException:
            if not dispatch.terminal:
                if provider_send_started:
                    await _shielded(
                        dispatch.finish_async(
                            AttemptResolution(
                                outcome=TerminalOutcome.UPSTREAM_FAILED,
                                release_capacity=False,
                            )
                        )
                    )
                else:
                    await _shielded(dispatch.fail_before_send())
            if not attempt_event_recorded:
                with suppress(Exception):
                    await _shielded(
                        record_attempt(
                            TerminalOutcome.UPSTREAM_FAILED,
                            uncertain=provider_send_started,
                        )
                    )
            raise
        finally:
            if response is not None:
                await _shielded(response.aclose())
            if permit is not None:
                await _shielded(permit.release())

    async def _shadow_attempt(
        self,
        admission_request: AdmissionRequest,
        ingress_request: IngressRequest,
        body: RetainedBody,
        profile: RequestProfile,
        client: AuthenticatedClient,
        lifecycle: RequestLifecycle,
        permit: Any,
        deadline: _DeadlineState,
    ) -> None:
        dispatch: DispatchLease | None = None
        response: httpx.Response | None = None
        provider_send_started = False
        event_recorded = False

        async def record(
            outcome: TerminalOutcome,
            *,
            uncertain: bool,
        ) -> None:
            nonlocal event_recorded
            if dispatch is None or event_recorded:
                return
            await _shielded(lifecycle.attempt_finished(dispatch, outcome, uncertain=uncertain))
            event_recorded = True

        try:
            dispatch = await self.admission.try_acquire_shadow(admission_request)
            if dispatch is None:
                return
            await _shielded(lifecycle.attempt_started(dispatch, shadow=True))
            target = self.deployment_resolver.resolve(dispatch.candidate.generation_id)
            prepared = prepare_dispatch(
                self.contract,
                endpoint=profile.endpoint.value,
                deployment=target.deployment,
                generation=target.generation,
                backend_manifest=target.backend_manifest,
            )
            rewritten = await self.profiler.rewrite(
                body,
                ingress_request.content_type,
                prepared,
                self.contract.known_litellm_params,
                client.key_id,
            )
            headers = _upstream_headers(ingress_request, self.backend_authorization)
            provider_send_started = True
            try:
                response = await self.http.send(
                    prepared.method,
                    prepared.path,
                    headers=headers,
                    content=rewritten,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                await dispatch.fail_before_send()
                await record(TerminalOutcome.UPSTREAM_FAILED, uncertain=False)
                return
            if not 200 <= response.status_code < 300:
                await read_prestart_error(
                    response,
                    read_inactivity_timeout_s=self.http.config.read_inactivity_timeout_s,
                )
                dispatch.lease.provider_send_completed()
                await dispatch.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.UPSTREAM_FAILED,
                        release_capacity=True,
                        actual_starts=1,
                    )
                )
                await record(TerminalOutcome.UPSTREAM_FAILED, uncertain=False)
                return
            try:
                reconcile_dispatch(
                    self.contract,
                    prepared,
                    status_code=response.status_code,
                    headers=response.headers,
                    response_started=False,
                )
            except DispatchError:
                dispatch.lease.provider_send_completed()
                await dispatch.finish_async(
                    AttemptResolution(
                        outcome=TerminalOutcome.UPSTREAM_FAILED,
                        release_capacity=True,
                        actual_starts=1,
                    )
                )
                await record(TerminalOutcome.UPSTREAM_FAILED, uncertain=False)
                return
            await drain_raw_response(
                response,
                read_inactivity_timeout_s=self.http.config.read_inactivity_timeout_s,
            )
            dispatch.lease.provider_send_completed()
            await dispatch.finish_async(
                AttemptResolution(
                    outcome=TerminalOutcome.COMPLETED,
                    release_capacity=True,
                    actual_starts=1,
                    actual_token_units=dispatch.lease.request.total_token_upper_bound,
                    actual_quota_units=dispatch.lease.request.quota_units,
                )
            )
            await record(TerminalOutcome.COMPLETED, uncertain=False)
        except asyncio.CancelledError:
            outcome = (
                TerminalOutcome.DEADLINE_EXCEEDED
                if deadline.expired
                else TerminalOutcome.CLIENT_CANCELLED
            )
            if dispatch is not None and not dispatch.terminal:
                if provider_send_started:
                    await _shielded(
                        dispatch.finish_async(
                            AttemptResolution(
                                outcome=outcome,
                                release_capacity=False,
                            )
                        )
                    )
                else:
                    await _shielded(dispatch.fail_before_send())
                if not event_recorded:
                    await _shielded(record(outcome, uncertain=provider_send_started))
            raise
        except BaseException:
            if dispatch is not None and not dispatch.terminal:
                if provider_send_started:
                    await _shielded(
                        dispatch.finish_async(
                            AttemptResolution(
                                outcome=TerminalOutcome.UPSTREAM_FAILED,
                                release_capacity=False,
                            )
                        )
                    )
                else:
                    await _shielded(dispatch.fail_before_send())
            if dispatch is not None and not event_recorded:
                with suppress(Exception):
                    await _shielded(
                        record(
                            TerminalOutcome.UPSTREAM_FAILED,
                            uncertain=provider_send_started,
                        )
                    )
        finally:
            if response is not None:
                await _shielded(response.aclose())
            await _shielded(permit.release())

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.profiler.aclose()
        await self.http.aclose()


def create_app(
    *,
    contract: AdapterContract,
    bundle: PolicyBundleV1,
    runtime: RuntimeState,
    activation_gate: ActivationGate,
    auth_view_provider: AuthViewProvider,
    generation_gate: GenerationOperationalGate,
    runtime_identity_provider: RuntimeIdentityProvider,
    deployment_resolver: DeploymentResolver,
    lifecycle_capacity: LifecycleCapacity,
    backend_authorization: BackendAuthorization,
    clock: GatewayClock,
    failure_classifier: FailureClassifier,
    http: ProcessHTTPClient,
    profiler: ProfileExecutor,
    ingress: IngressResources,
    permits: AttemptPermitPool,
) -> GatewayApp:
    """Compose only mandatory production dependencies; no permissive stand-ins."""
    return GatewayApp(
        contract=contract,
        bundle=bundle,
        runtime=runtime,
        activation_gate=activation_gate,
        auth_view_provider=auth_view_provider,
        generation_gate=generation_gate,
        runtime_identity_provider=runtime_identity_provider,
        deployment_resolver=deployment_resolver,
        lifecycle_capacity=lifecycle_capacity,
        backend_authorization=backend_authorization,
        clock=clock,
        failure_classifier=failure_classifier,
        http=http,
        profiler=profiler,
        ingress=ingress,
        permits=permits,
    )
