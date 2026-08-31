"""All-or-nothing LiteLLM discovery over an injected async transport."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from llmmaxxing.adapters.litellm.contract import (
    AdapterContract,
    BuildProbe,
    CatalogModel,
    DiscoverySnapshot,
    DeploymentReceipt,
    EffectiveDeployment,
    PreparedDispatch,
    TransportResponse,
)
from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.ids import AccountId


class AdapterTransport(Protocol):
    async def request(
        self,
        method: str,
        path: str,
        *,
        key: str,
        query: Mapping[str, str] | None = None,
    ) -> TransportResponse: ...


class DiscoveryError(RuntimeError):
    """A complete certified snapshot could not be proved."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Account(_Strict):
    id: AccountId
    binding: str = Field(min_length=1)


class _Credential(_Strict):
    field: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^hcf1_[0-9a-f]{64}$")
    epoch: int = Field(ge=1)
    dynamic_list: bool

    @model_validator(mode="after")
    def _static_only(self) -> Self:
        if self.dynamic_list:
            raise ValueError("dynamic credential lists are not certified")
        return self


class _Routing(_Strict):
    num_retries: int
    fallbacks: tuple[JsonValue, ...]
    context_window_fallbacks: tuple[JsonValue, ...]
    content_policy_fallbacks: tuple[JsonValue, ...]
    cooldown_selection: bool

    @model_validator(mode="after")
    def _gateway_owns_recovery(self) -> Self:
        if self.num_retries != 0:
            raise ValueError("LiteLLM retries must be zero")
        if self.fallbacks or self.context_window_fallbacks or self.content_policy_fallbacks:
            raise ValueError("LiteLLM fallback mappings must be empty")
        if self.cooldown_selection:
            raise ValueError("LiteLLM cooldown selection must be disabled")
        return self


class _Metadata(_Strict):
    hidden_alias: bool
    execution: dict[str, JsonValue]
    capabilities: dict[str, JsonValue]
    context: dict[str, JsonValue]
    defaults: dict[str, JsonValue]
    pricing: dict[str, JsonValue]
    account: _Account
    credential: _Credential
    routing: _Routing

    @model_validator(mode="after")
    def _hidden_only(self) -> Self:
        if not self.hidden_alias:
            raise ValueError("certified deployment must be an exact hidden alias")
        if not self.execution:
            raise ValueError("execution projection is empty")
        return self


class LiteLLMAdapter:
    def __init__(self, contract: AdapterContract, transport: AdapterTransport) -> None:
        self.contract = contract
        self.transport = transport
        self._snapshot: DiscoverySnapshot | None = None

    @property
    def snapshot(self) -> DiscoverySnapshot | None:
        return self._snapshot

    async def _get(self, path: str, query: Mapping[str, str] | None = None) -> dict[str, Any]:
        response = await self.transport.request("GET", path, key="discovery", query=query)
        if response.status_code != 200:
            raise DiscoveryError(f"discovery GET {path} returned {response.status_code}")
        if not isinstance(response.body, dict):
            raise DiscoveryError(f"discovery GET {path} returned a non-object")
        return response.body

    async def probe(self) -> BuildProbe:
        readiness = await self._get(self.contract.discovery.readiness_path)
        expected_readiness = {
            "status",
            "db",
            "cache",
            "litellm_version",
            "success_callbacks",
            "use_aiohttp_transport",
            "log_level",
            "is_detailed_debug",
            "show_no_redis_warning",
        }
        if set(readiness) != expected_readiness:
            raise DiscoveryError("readiness details schema does not match the certified build")
        if readiness.get("status") != "healthy" or readiness.get("litellm_version") != self.contract.litellm.version:
            raise DiscoveryError("LiteLLM build probe mismatch")

        callback_state = await self._get(self.contract.discovery.callbacks_path)
        expected_callback_fields = {
            "alerting",
            "litellm.callbacks",
            "litellm.input_callback",
            "litellm.failure_callback",
            "litellm.success_callback",
            "litellm._async_success_callback",
            "litellm._async_failure_callback",
            "litellm._async_input_callback",
            "all_litellm_callbacks",
            "num_callbacks",
            "num_alerting",
            "litellm.request_timeout",
        }
        if set(callback_state) != expected_callback_fields:
            raise DiscoveryError("active callback schema does not match the certified build")
        callbacks = callback_state.get("litellm.callbacks")
        identity = self.contract.guard.active_callback_identity
        if not isinstance(callbacks, list) or callbacks.count(identity) != 1 or callbacks[-1] != identity:
            raise DiscoveryError("certified generation guard is missing, duplicated, or not last")
        all_callbacks = callback_state.get("all_litellm_callbacks")
        if not isinstance(all_callbacks, list) or callback_state.get("num_callbacks") != len(all_callbacks):
            raise DiscoveryError("active callback counts are malformed")
        return BuildProbe(
            contract_id=self.contract.contract_id,
            observed_version=self.contract.litellm.version,
            certified_image=self.contract.litellm.image,
            source_commit=self.contract.litellm.source_commit,
            guard_identity=identity,
            guard_is_last=True,
        )

    async def _model_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total_pages: int | None = None
        total_count: int | None = None
        page = 1
        while total_pages is None or page <= total_pages:
            body = await self._get(
                self.contract.discovery.model_info_path,
                {"page": str(page), "size": str(self.contract.discovery.page_size)},
            )
            required = {"data", "total_count", "current_page", "total_pages", "size"}
            if set(body) != required:
                raise DiscoveryError("model-info pagination schema mismatch")
            data = body.get("data")
            if (
                not isinstance(data, list)
                or not isinstance(body.get("total_count"), int)
                or not isinstance(body.get("current_page"), int)
                or not isinstance(body.get("total_pages"), int)
                or not isinstance(body.get("size"), int)
            ):
                raise DiscoveryError("model-info pagination fields are malformed")
            if body["current_page"] != page or body["size"] != self.contract.discovery.page_size:
                raise DiscoveryError("model-info pagination position mismatch")
            if page == 1:
                total_pages = body["total_pages"]
                total_count = body["total_count"]
                if total_pages < 1 or total_count < 1:
                    raise DiscoveryError("model-info discovery is empty")
            elif body["total_pages"] != total_pages or body["total_count"] != total_count:
                raise DiscoveryError("model-info pagination changed during discovery")
            if any(not isinstance(row, dict) for row in data):
                raise DiscoveryError("model-info page contains a non-object row")
            rows.extend(data)
            page += 1
        if total_count is None or len(rows) != total_count:
            raise DiscoveryError("model-info pagination count mismatch")
        return rows

    def _deployment(self, raw: Mapping[str, Any]) -> EffectiveDeployment | None:
        alias = raw.get("model_name")
        params = raw.get("litellm_params")
        info = raw.get("model_info")
        if not isinstance(alias, str) or not isinstance(params, dict) or not isinstance(info, dict):
            raise DiscoveryError("model-info row lacks model_name/litellm_params/model_info")
        lmx = info.get("llmmaxxing")
        if lmx is None:
            return None
        if unknown := set(params) - set(self.contract.known_litellm_params):
            raise DiscoveryError(f"unknown execution fields in litellm_params: {sorted(unknown)}")
        try:
            metadata = _Metadata.model_validate(lmx)
        except ValidationError as exc:
            raise DiscoveryError(f"invalid llmmaxxing deployment metadata: {exc}") from exc
        if unknown := set(metadata.execution) - set(self.contract.known_execution_fields):
            raise DiscoveryError(f"unknown execution fields: {sorted(unknown)}")
        if params.get("num_retries") != metadata.routing.num_retries:
            raise DiscoveryError("deployment retry authority mismatch")
        if params.get("custom_llm_provider") != metadata.execution.get("custom_llm_provider"):
            raise DiscoveryError("deployment provider projection mismatch")
        if params.get("model") != metadata.execution.get("model"):
            raise DiscoveryError("deployment upstream model projection mismatch")
        runtime_id = info.get("id")
        mode = info.get("mode")
        if not isinstance(runtime_id, str) or not runtime_id or not isinstance(mode, str) or not mode:
            raise DiscoveryError("deployment runtime id or mode is missing")
        return EffectiveDeployment(
            runtime_id=runtime_id,
            hidden_alias=alias,
            mode=mode,
            execution=metadata.execution,
            capabilities=metadata.capabilities,
            context=metadata.context,
            defaults=metadata.defaults,
            pricing=metadata.pricing,
            account_id=metadata.account.id,
            account_binding=metadata.account.binding,
            credential_field=metadata.credential.field,
            credential_fingerprint=metadata.credential.fingerprint,
            credential_epoch=metadata.credential.epoch,
        )

    async def discover_complete(self) -> DiscoverySnapshot:
        probe = await self.probe()
        raw_rows = await self._model_rows()
        deployments = [row for raw in raw_rows if (row := self._deployment(raw)) is not None]
        if not deployments:
            raise DiscoveryError("no certified hidden deployments discovered")
        aliases = [row.hidden_alias for row in deployments]
        runtime_ids = [row.runtime_id for row in deployments]
        if len(set(aliases)) != len(aliases):
            raise DiscoveryError("a hidden alias maps to more than one deployment")
        if len(set(runtime_ids)) != len(runtime_ids):
            raise DiscoveryError("a runtime deployment id is reused")
        deployments.sort(key=lambda row: row.hidden_alias)

        catalog_body = await self._get(
            self.contract.discovery.models_path,
            self.contract.discovery.models_query,
        )
        if set(catalog_body) != {"object", "data"} or catalog_body.get("object") != "list":
            raise DiscoveryError("expanded model catalog schema mismatch")
        catalog_data = catalog_body.get("data")
        if not isinstance(catalog_data, list) or any(not isinstance(row, dict) for row in catalog_data):
            raise DiscoveryError("expanded model catalog is malformed")
        try:
            catalog = tuple(
                sorted(
                    (
                        CatalogModel(id=row["id"], owned_by=row["owned_by"])
                        for row in catalog_data
                    ),
                    key=lambda row: row.id,
                )
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise DiscoveryError("expanded model catalog row is malformed") from exc
        catalog_ids = {row.id for row in catalog}
        if not catalog or not set(aliases) <= catalog_ids:
            raise DiscoveryError("expanded catalog omits a hidden alias")

        from llmmaxxing.adapters.litellm.guard import deployment_generation

        revision_payload = {
            "contract_id": self.contract.contract_id,
            "build": self.contract.litellm.model_dump(mode="json"),
            "deployments": [
                deployment_generation(row, self.contract).projection.model_dump(mode="json") for row in deployments
            ],
            "catalog": [row.model_dump(mode="json") for row in catalog],
        }
        revision = "bm1_" + hashlib.sha256(canonical_json_bytes(revision_payload)).hexdigest()
        snapshot = DiscoverySnapshot(
            probe=probe,
            deployments=tuple(deployments),
            catalog=catalog,
            manifest_revision=revision,
        )
        self._snapshot = snapshot
        return snapshot

    def prepare_dispatch(self, **kwargs: Any) -> PreparedDispatch:
        from llmmaxxing.adapters.litellm.dispatch import prepare_dispatch

        return prepare_dispatch(self.contract, **kwargs)

    def reconcile_dispatch(self, *args: Any, **kwargs: Any) -> DeploymentReceipt:
        from llmmaxxing.adapters.litellm.dispatch import reconcile_dispatch

        return reconcile_dispatch(self.contract, *args, **kwargs)
