"""Frozen contract records for the one certified LiteLLM 1.98.0 build."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from llmmaxxing.core.canonical import canonical_json_bytes

from llmmaxxing.core.ids import AccountId, DeploymentGenerationId

_HEX64 = r"^[0-9a-f]{64}$"
_SHA256 = r"^sha256:[0-9a-f]{64}$"
_MANIFEST = r"^bm1_[0-9a-f]{64}$"
_CREDENTIAL_FINGERPRINT = r"^hcf1_[0-9a-f]{64}$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class ExactLiteLLMBuild(_Frozen):
    version: Literal["1.98.0"]
    image: str = Field(pattern=r"^ghcr\.io/berriai/litellm@sha256:[0-9a-f]{64}$")
    source_commit: str = Field(pattern=_HEX64.replace("64", "40"))
    source_files: dict[str, str]

    @model_validator(mode="after")
    def _source_attestation(self) -> Self:
        if not self.source_files:
            raise ValueError("certified source-file digests are empty")
        if any(
            not path.startswith("litellm/")
            or not digest.startswith("sha256:")
            or len(digest) != 71
            for path, digest in self.source_files.items()
        ):
            raise ValueError("certified source-file attestation is malformed")
        return self


class GuardContract(_Frozen):
    mount_path: str = Field(min_length=1)
    callback_name: Literal["llmmaxxing_guard"]
    digest: str = Field(pattern=_SHA256)
    registration: Literal["last"]
    hook: Literal["async_pre_call_deployment_hook"]
    dynamic_credential_lists: Literal["unsupported"]
    credential_fingerprint: Literal["hcf1_hmac_sha256"]
    fence_fields: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _exact_fence(self) -> Self:
        expected = {
            "alias",
            "generation_id",
            "account_id",
            "account_binding",
            "backend_manifest",
            "credential_fingerprint",
            "credential_epoch",
            "contract_id",
            "endpoint",
        }
        if set(self.fence_fields) != expected or len(self.fence_fields) != len(expected):
            raise ValueError("guard fence fields do not match the certified hook")
        return self

    @property
    def active_callback_identity(self) -> str:
        return f"{self.callback_name}@{self.digest}"


class ServiceKeyContract(_Frozen):
    user_role: str = Field(min_length=1)
    allowed_routes: tuple[str, ...] = Field(min_length=1)
    models: tuple[str, ...] = Field(min_length=1)
    policy_kind: Literal["explicit_routes_hidden_models", "proxy_admin_viewer_explicit_routes"]

    @model_validator(mode="after")
    def _closed_scope(self) -> Self:
        if len(set(self.allowed_routes)) != len(self.allowed_routes):
            raise ValueError("duplicate service-key route")
        if any(
            route in {"read_only", "llm_api_routes", "llm_api"} for route in self.allowed_routes
        ):
            raise ValueError("broad service-key route groups are not certified")
        return self


class ServiceKeyContracts(_Frozen):
    inference: ServiceKeyContract
    discovery: ServiceKeyContract


class DiscoveryContract(_Frozen):
    readiness_path: Literal["/health/readiness/details"]
    callbacks_path: Literal["/active/callbacks"]
    model_info_path: Literal["/v2/model/info"]
    models_path: Literal["/v1/models"]
    page_size: int = Field(ge=1, le=1000)
    models_query: dict[str, str]


class RouterRequirements(_Frozen):
    disable_cooldowns: Literal[True]
    num_retries: Literal[0]
    fallbacks: tuple[()] = ()
    context_window_fallbacks: tuple[()] = ()
    content_policy_fallbacks: tuple[()] = ()
    hidden_alias_deployments: Literal[1]


class CertifiedEndpoint(_Frozen):
    name: str = Field(min_length=1)
    method: Literal["POST"]
    path: str = Field(pattern=r"^/v1/")
    model_locator: Literal["json.model", "multipart.model"]
    fence_locator: Literal[
        "json.metadata",
        "json.litellm_metadata",
        "multipart.metadata",
    ]
    execution_normalizers: dict[str, Literal["provider_prefix_removed"]]
    receipt_header: Literal["x-litellm-model-id"]
    guard_required: Literal[True]

class DenialProbe(_Frozen):
    protocol: Literal["http", "websocket"]
    method: Literal["GET", "POST", "DELETE"]
    path: str = Field(pattern=r"^/v1/responses")


class AdapterContract(_Frozen):
    contract_id: Literal["litellm-1.98.0"]
    litellm: ExactLiteLLMBuild
    guard: GuardContract
    service_keys: ServiceKeyContracts
    discovery: DiscoveryContract
    router: RouterRequirements
    known_litellm_params: tuple[str, ...]
    known_execution_fields: tuple[str, ...]
    endpoints: tuple[CertifiedEndpoint, ...]
    denial_probes: tuple[DenialProbe, ...]
    unsupported: tuple[str, ...]

    @model_validator(mode="after")
    def _unique_closed_contract(self) -> Self:
        if len({endpoint.name for endpoint in self.endpoints}) != len(self.endpoints):
            raise ValueError("duplicate certified endpoint name")
        endpoint_routes = {(endpoint.method, endpoint.path) for endpoint in self.endpoints}
        if len(endpoint_routes) != len(self.endpoints):
            raise ValueError("duplicate certified endpoint route")
        if len(set(self.known_litellm_params)) != len(self.known_litellm_params):
            raise ValueError("duplicate known LiteLLM parameter")
        if len(set(self.known_execution_fields)) != len(self.known_execution_fields):
            raise ValueError("duplicate known execution field")
        inference_routes = set(self.service_keys.inference.allowed_routes)
        if inference_routes != {endpoint.path for endpoint in self.endpoints}:
            raise ValueError("inference service key must name exactly the certified routes")
        discovery_routes = set(self.service_keys.discovery.allowed_routes)
        expected_discovery = {
            self.discovery.readiness_path,
            self.discovery.callbacks_path,
            self.discovery.model_info_path,
            self.discovery.models_path,
        }
        if discovery_routes != expected_discovery:
            raise ValueError("discovery service key must name exactly the discovery routes")
        expected_denials = {
            ("http", "GET", "/v1/responses/{response_id}"),
            ("http", "DELETE", "/v1/responses/{response_id}"),
            ("http", "GET", "/v1/responses/{response_id}/input_items"),
            ("http", "POST", "/v1/responses/{response_id}/cancel"),
            ("http", "POST", "/v1/responses/compact"),
            ("websocket", "GET", "/v1/responses"),
        }
        observed_denials = {
            (probe.protocol, probe.method, probe.path) for probe in self.denial_probes
        }
        if observed_denials != expected_denials or len(self.denial_probes) != len(
            expected_denials
        ):
            raise ValueError("Responses denial probes do not match pinned routes")
        return self

    def endpoint(self, name: str) -> CertifiedEndpoint:
        try:
            return next(endpoint for endpoint in self.endpoints if endpoint.name == name)
        except StopIteration as exc:
            raise ValueError(f"unsupported LiteLLM endpoint: {name}") from exc

    def validate_router_config(self, value: Mapping[str, Any]) -> None:
        settings = value.get("router_settings")
        counts = value.get("hidden_alias_deployment_counts")
        if not isinstance(settings, Mapping) or not isinstance(counts, Mapping) or not counts:
            raise ValueError("router fixture is incomplete")
        expected = self.router.model_dump(mode="python", exclude={"hidden_alias_deployments"})
        for key, required in expected.items():
            observed = settings.get(key)
            if isinstance(required, tuple):
                observed = tuple(observed) if isinstance(observed, list) else observed
            if observed != required:
                raise ValueError(f"router authority mismatch: {key}")
        if any(count != self.router.hidden_alias_deployments for count in counts.values()):
            raise ValueError("a hidden alias must map to exactly one deployment")


class TransportResponse(_Frozen):
    status_code: int = Field(ge=100, le=599)
    headers: dict[str, str]
    body: JsonValue


class EffectiveDeployment(_Frozen):
    runtime_id: str = Field(min_length=1)
    hidden_alias: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    mode: str = Field(min_length=1)
    execution: dict[str, JsonValue]
    capabilities: dict[str, JsonValue]
    context: dict[str, JsonValue]
    defaults: dict[str, JsonValue]
    pricing: dict[str, JsonValue]
    account_id: AccountId
    account_binding: str = Field(min_length=1)
    credential_field: str = Field(min_length=1)
    credential_fingerprint: str = Field(pattern=_CREDENTIAL_FINGERPRINT)
    credential_epoch: int = Field(ge=1)


class DeploymentGenerationProjection(_Frozen):
    contract_id: str = Field(min_length=1)
    hidden_alias: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    mode: str = Field(min_length=1)
    execution: dict[str, JsonValue]
    capabilities: dict[str, JsonValue]
    context: dict[str, JsonValue]
    defaults: dict[str, JsonValue]
    pricing: dict[str, JsonValue]
    account_id: AccountId
    account_binding: str = Field(min_length=1)
    credential_fingerprint: str = Field(pattern=_CREDENTIAL_FINGERPRINT)
    credential_epoch: int = Field(ge=1)


class DeploymentGenerationFingerprint(_Frozen):
    generation_id: DeploymentGenerationId
    projection: DeploymentGenerationProjection


class BuildProbe(_Frozen):
    contract_id: str
    observed_version: Literal["1.98.0"]
    certified_image: str
    source_commit: str
    guard_identity: str
    guard_is_last: Literal[True]


class CatalogModel(_Frozen):
    id: str = Field(min_length=1)
    owned_by: str = Field(min_length=1)


class DiscoverySnapshot(_Frozen):
    probe: BuildProbe
    deployments: tuple[EffectiveDeployment, ...]
    catalog: tuple[CatalogModel, ...]
    manifest_revision: str = Field(pattern=_MANIFEST)


class GuardDeploymentExpectation(_Frozen):
    runtime_id: str = Field(min_length=1)
    generation_id: DeploymentGenerationId
    credential_field: str = Field(min_length=1)
    projection: DeploymentGenerationProjection


class GuardManifest(_Frozen):
    contract_id: Literal["litellm-1.98.0"]
    backend_manifest: str = Field(pattern=_MANIFEST)
    guard_digest: str = Field(pattern=_SHA256)
    deployments: dict[str, GuardDeploymentExpectation]

    @model_validator(mode="after")
    def _nonempty_qualified_aliases(self) -> Self:
        if not self.deployments:
            raise ValueError("guard manifest deployments are empty")
        if any("/" not in alias for alias in self.deployments):
            raise ValueError("guard manifest contains an unqualified alias")
        for alias, expectation in self.deployments.items():
            if expectation.projection.hidden_alias != alias:
                raise ValueError("guard deployment alias differs from semantic projection")
            digest = hashlib.sha256(
                canonical_json_bytes(expectation.projection.model_dump(mode="json"))
            ).hexdigest()
            if expectation.generation_id != DeploymentGenerationId.from_digest(digest):
                raise ValueError("guard generation differs from semantic projection")
        return self


class PreparedDispatch(_Frozen):
    endpoint: str
    method: Literal["POST"]
    path: str
    model_locator: Literal["json.model", "multipart.model"]
    fence_locator: Literal[
        "json.metadata",
        "json.litellm_metadata",
        "multipart.metadata",
    ]
    execution_normalizers: dict[str, Literal["provider_prefix_removed"]]
    hidden_alias: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    expected_deployment_id: str
    generation_id: DeploymentGenerationId
    backend_manifest: str = Field(pattern=_MANIFEST)
    trusted_metadata: dict[str, JsonValue]


class DeploymentReceipt(_Frozen):
    endpoint: str
    deployment_id: str
    generation_id: DeploymentGenerationId


type CertifiedErrorCode = Literal[
    "unsupported_endpoint",
    "generation_mismatch",
    "receipt_missing",
    "receipt_mismatch",
    "upstream_error",
]


class CertifiedErrorDetail(_Frozen):
    code: CertifiedErrorCode
    message: str
    critical: bool
    retryable: bool


def load_contract(path: str | Path | None = None) -> AdapterContract:
    if path is None:
        path = Path(__file__).resolve().parents[4] / "compatibility/litellm-1.98.0.yaml"
    raw = json.loads(Path(path).read_text())
    return AdapterContract.model_validate(raw)
