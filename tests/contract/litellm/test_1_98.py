from __future__ import annotations

import asyncio
import hashlib
import json
import runpy
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest
from pydantic import ValidationError

from llmmaxxing.adapters.litellm import LiteLLMAdapter as PublicLiteLLMAdapter
from llmmaxxing.adapters.litellm.contract import (
    EffectiveDeployment,
    TransportResponse,
    load_contract,
)
from llmmaxxing.adapters.litellm.discovery import DiscoveryError, LiteLLMAdapter
from llmmaxxing.adapters.litellm.dispatch import DispatchError
from llmmaxxing.adapters.litellm.guard import build_guard_manifest, deployment_generation

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures"

def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())["response"]


class FixtureTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict[str, str]]] = []
        self.responses: dict[tuple[str, int], dict[str, Any]] = {
            ("/v2/model/info", 1): fixture("model-info-page-1.json"),
            ("/v2/model/info", 2): fixture("model-info-page-2.json"),
        }
        self.override: dict[str, dict[str, Any]] = {}

    async def request(
        self,
        method: str,
        path: str,
        *,
        key: str,
        query: Mapping[str, str] | None = None,
    ) -> TransportResponse:
        query_dict = dict(query or {})
        self.calls.append((key, method, path, query_dict))
        if path in self.override:
            body = deepcopy(self.override[path])
        elif path == "/health/readiness/details":
            body = fixture("readiness-details.json")
        elif path == "/active/callbacks":
            body = fixture("active-callbacks.json")
            digest = hashlib.sha256((ROOT / "deploy/litellm/llmmaxxing_guard.py").read_bytes()).hexdigest()
            body = json.loads(json.dumps(body).replace("__GUARD_DIGEST__", digest))
        elif path == "/v2/model/info":
            body = deepcopy(self.responses[(path, int(query_dict["page"]))])
        elif path == "/v1/models":
            body = fixture("models-expanded.json")
        else:
            return TransportResponse(status_code=404, headers={}, body={"error": "not found"})
        return TransportResponse(status_code=200, headers={"content-type": "application/json"}, body=body)


def test_public_adapter_interface() -> None:
    assert PublicLiteLLMAdapter is LiteLLMAdapter


def test_contract_binds_one_exact_build_guard_and_explicit_service_keys() -> None:
    contract = load_contract()
    assert contract.litellm.version == "1.98.0"
    assert contract.litellm.image == (
        "ghcr.io/berriai/litellm@sha256:"
        "20b5044b619055374061a6d5b7b08754cad75aeabbf82ddf4f69cc0cf80ddaf4"
    )
    assert contract.litellm.source_commit == "d8f71d7bdbd7c9873d98293f83d64c6db72847e6"
    assert not any(token in contract.litellm.version for token in (">", "<", "^", "~", "*", ","))

    guard = ROOT / contract.guard.mount_path
    assert contract.guard.digest == "sha256:" + hashlib.sha256(guard.read_bytes()).hexdigest()
    inference = contract.service_keys.inference
    assert contract.guard.credential_fingerprint == "hcf1_hmac_sha256"
    assert set(contract.guard.fence_fields) == {
        "alias",
        "generation_id",
        "account_id",
        "account_binding",
        "backend_manifest",
        "credential_fingerprint",
        "credential_epoch",
        "contract_id",
    }
    discovery = contract.service_keys.discovery
    assert inference.allowed_routes and discovery.allowed_routes
    assert inference.models == ("exact_hidden_aliases",)
    assert discovery.user_role == "proxy_admin_viewer"
    assert set(discovery.allowed_routes) == {
        "/health/readiness/details",
        "/active/callbacks",
        "/v2/model/info",
        "/v1/models",
    }
    forbidden = {"read_only", "llm_api", "models_only", "master"}
    assert inference.policy_kind not in forbidden
    assert discovery.policy_kind not in forbidden


def test_contract_certifies_only_native_receipt_endpoints_and_exact_locators() -> None:
    contract = load_contract()
    assert {(e.name, e.method, e.path, e.model_locator) for e in contract.endpoints} == {
        ("chat", "POST", "/v1/chat/completions", "json.model"),
        ("text", "POST", "/v1/completions", "json.model"),
        ("responses", "POST", "/v1/responses", "json.model"),
        ("messages", "POST", "/v1/messages", "json.model"),
        ("embeddings", "POST", "/v1/embeddings", "json.model"),
        ("rerank", "POST", "/v1/rerank", "json.model"),
        ("audio_speech", "POST", "/v1/audio/speech", "json.model"),
        ("audio_transcription", "POST", "/v1/audio/transcriptions", "multipart.model"),
        ("image", "POST", "/v1/images/generations", "json.model"),
    }
    assert all(e.guard_required and e.receipt_header == "x-litellm-model-id" for e in contract.endpoints)
    assert {
        "responses_stateful_get",
        "responses_stateful_delete",
        "responses_cancel",
        "responses_compact",
        "websocket",
        "management_mutation",
        "batch",
        "assistants",
        "realtime",
        "provider_passthrough",
    } <= set(contract.unsupported)

    selectors = json.loads((FIXTURES / "endpoint-selectors.json").read_text())["fixtures"]
    assert set(selectors) == {endpoint.name for endpoint in contract.endpoints}
    for endpoint in contract.endpoints:
        selected = selectors[endpoint.name]
        if endpoint.model_locator == "json.model":
            assert selected["body"]["model"] == "__HIDDEN_ALIAS__"
        else:
            assert selected["fields"]["model"] == "__HIDDEN_ALIAS__"


def test_router_and_key_isolation_fixtures_are_closed() -> None:
    contract = load_contract()
    router = json.loads((FIXTURES / "router-config.json").read_text())
    contract.validate_router_config(router)
    matrix = json.loads((FIXTURES / "key-route-matrix.json").read_text())["observations"]
    observed = {(row["key"], row["method"], row["path"]): row["status"] for row in matrix}
    assert observed[("inference", "GET", "/v2/model/info")] == 403
    assert observed[("discovery", "POST", "/v1/chat/completions")] == 403
    assert observed[("discovery", "POST", "/model/update")] == 403


@pytest.mark.parametrize(
    "change",
    (
        {"router_settings": {"disable_cooldowns": False}},
        {"router_settings": {"fallbacks": ["another-model"]}},
        {"hidden_alias_deployment_counts": {"lmx/electron-v1": 2}},
    ),
)
def test_router_authority_rejects_retries_fallbacks_cooldowns_and_alias_fanout(
    change: dict[str, Any],
) -> None:
    contract = load_contract()
    router = json.loads((FIXTURES / "router-config.json").read_text())
    for section, values in change.items():
        router[section].update(values)
    with pytest.raises(ValueError):
        contract.validate_router_config(router)


def test_probe_and_complete_paginated_discovery_are_exact_and_atomic() -> None:
    transport = FixtureTransport()
    adapter = LiteLLMAdapter(load_contract(), transport)
    probe = asyncio.run(adapter.probe())
    assert probe.observed_version == "1.98.0"
    assert probe.guard_is_last
    snapshot = asyncio.run(adapter.discover_complete())
    assert snapshot is adapter.snapshot
    assert [row.hidden_alias for row in snapshot.deployments] == [
        "lmx/electron-v1",
        "lmx/local-fixture",
    ]
    assert snapshot.deployments[0].execution["custom_llm_provider"] == "electron"
    assert snapshot.deployments[0].execution["region_name"] == "test-region-1"
    assert [call[2] for call in transport.calls] == [
        "/health/readiness/details",
        "/active/callbacks",
        "/health/readiness/details",
        "/active/callbacks",
        "/v2/model/info",
        "/v2/model/info",
        "/v1/models",
    ]
    model_calls = [call for call in transport.calls if call[2] == "/v2/model/info"]
    assert [call[3]["page"] for call in model_calls] == ["1", "2"]
    assert all(call[3]["size"] == "1" for call in model_calls)
    catalog_call = next(call for call in transport.calls if call[2] == "/v1/models")
    assert catalog_call[3] == {"scope": "expand", "return_wildcard_routes": "true"}

    previous = snapshot
    bad = deepcopy(fixture("model-info-page-2.json"))
    bad["current_page"] = 9
    transport.responses[("/v2/model/info", 2)] = bad
    with pytest.raises(DiscoveryError, match="pagination"):
        asyncio.run(adapter.discover_complete())
    assert adapter.snapshot is previous


def test_malformed_catalog_leaves_last_complete_snapshot_unchanged() -> None:
    transport = FixtureTransport()
    adapter = LiteLLMAdapter(load_contract(), transport)
    previous = asyncio.run(adapter.discover_complete())
    malformed = fixture("models-expanded.json")
    malformed["data"].append({"id": "broken-without-owner"})
    transport.override["/v1/models"] = malformed
    with pytest.raises(DiscoveryError, match="catalog"):
        asyncio.run(adapter.discover_complete())
    assert adapter.snapshot is previous


def test_generation_uses_jcs_semantics_not_runtime_identity() -> None:
    adapter = LiteLLMAdapter(load_contract(), FixtureTransport())
    snapshot = asyncio.run(adapter.discover_complete())
    row = snapshot.deployments[0]
    first = deployment_generation(row, adapter.contract)
    same = row.model_copy(update={"runtime_id": "runtime-electron-restarted"})
    assert deployment_generation(same, adapter.contract) == first
    changed = row.model_copy(
        update={"execution": {**row.execution, "region_name": "test-region-2"}}
    )
    assert deployment_generation(changed, adapter.contract) != first
    changed_mode = row.model_copy(update={"mode": "embedding"})
    assert deployment_generation(changed_mode, adapter.contract) != first
    assert first.projection.contract_id == adapter.contract.contract_id


def test_effective_deployment_is_frozen_strict_and_unknown_execution_fields_fail() -> None:
    adapter = LiteLLMAdapter(load_contract(), FixtureTransport())
    row = asyncio.run(adapter.discover_complete()).deployments[0]
    with pytest.raises(ValidationError):
        EffectiveDeployment.model_validate({**row.model_dump(mode="python"), "surprise": True})

    with pytest.raises(ValidationError):
        EffectiveDeployment.model_validate(
            {**row.model_dump(mode="python"), "hidden_alias": "not-provider-qualified"}
        )

    transport = FixtureTransport()
    page = deepcopy(fixture("model-info-page-2.json"))
    page["data"][0]["model_info"]["llmmaxxing"]["execution"]["new_runtime_knob"] = True
    transport.responses[("/v2/model/info", 2)] = page
    with pytest.raises(DiscoveryError, match="unknown execution"):
        asyncio.run(LiteLLMAdapter(load_contract(), transport).discover_complete())


def test_unknown_provider_discovers_generates_and_prepares_without_source_enum() -> None:
    adapter = LiteLLMAdapter(load_contract(), FixtureTransport())
    snapshot = asyncio.run(adapter.discover_complete())
    electron = next(
        row
        for row in snapshot.deployments
        if row.execution["custom_llm_provider"] == "electron"
    )
    generated = deployment_generation(electron, adapter.contract)
    prepared = adapter.prepare_dispatch(
        endpoint="chat",
        deployment=electron,
        generation=generated,
        backend_manifest=snapshot.manifest_revision,
    )
    assert prepared.hidden_alias == "lmx/electron-v1"
    assert prepared.model_locator == "json.model"
    fence = prepared.trusted_metadata["llmmaxxing_guard"]
    assert fence["alias"] == "lmx/electron-v1"
    assert fence["generation_id"] == str(generated.generation_id)
    assert "electron" not in type(electron).__annotations__.values()

    receipt = adapter.reconcile_dispatch(
        prepared,
        status_code=200,
        headers={"X-LiteLLM-Model-ID": "runtime-electron-001"},
    )
    assert receipt.deployment_id == "runtime-electron-001"
    with pytest.raises(DispatchError, match="receipt") as missing:
        adapter.reconcile_dispatch(prepared, status_code=200, headers={})
    assert missing.value.detail.critical and not missing.value.detail.retryable
    with pytest.raises(DispatchError, match="receipt"):
        adapter.reconcile_dispatch(
            prepared,
            status_code=200,
            headers={"x-litellm-model-id": "some-other-runtime"},
        )


def test_pinned_stack_launcher_materializes_exact_isolated_contract(tmp_path: Path) -> None:
    launcher = runpy.run_path(str(Path(__file__).parent / "pinned_stack.py"))
    stack = launcher["materialize_stack"](tmp_path)
    compose = json.loads(stack.compose_path.read_text())
    contract = load_contract()
    assert compose["services"]["litellm"]["image"] == contract.litellm.image
    assert compose["services"]["fake-provider"]["image"] == contract.litellm.image
    assert compose["services"]["postgres"]["image"].startswith("postgres@sha256:")
    assert compose["services"]["litellm"]["depends_on"] == {
        "fake-provider": {"condition": "service_healthy"},
        "postgres": {"condition": "service_healthy"},
    }
    config = json.loads(stack.config_path.read_text())
    assert config["litellm_settings"]["callbacks"][-1] == (
        "llmmaxxing_guard.llmmaxxing_guard"
    )
    assert {
        row["model_name"] for row in config["model_list"]
    } == {target["alias"] for target in stack.endpoint_targets.values()}
    inference, discovery = launcher["key_requests"](contract, "inference-user", "discovery-user")
    assert set(inference["allowed_routes"]) == set(contract.service_keys.inference.allowed_routes)
    assert set(discovery["allowed_routes"]) == set(contract.service_keys.discovery.allowed_routes)
    assert discovery["key_type"] == inference["key_type"] == "default"




def test_fixture_payloads_are_redacted_and_container_evidence_is_honestly_pending() -> None:
    for path in FIXTURES.glob("*.json"):
        raw = path.read_text()
        assert "sk-" not in raw
        assert "Bearer " not in raw
        data = json.loads(raw)
        provenance = data.get("_provenance")
        if provenance is not None:
            assert provenance["pinned_container_verified"] is False


def test_guard_manifest_is_strict_data_driven_and_secret_free() -> None:
    adapter = LiteLLMAdapter(load_contract(), FixtureTransport())
    snapshot = asyncio.run(adapter.discover_complete())
    manifest = build_guard_manifest(snapshot, adapter.contract)
    electron = manifest.deployments["lmx/electron-v1"]
    assert manifest.backend_manifest == snapshot.manifest_revision
    assert electron.runtime_id == "runtime-electron-001"
    assert electron.execution["custom_llm_provider"] == "electron"
    assert electron.generation_id == deployment_generation(
        snapshot.deployments[0], adapter.contract
    ).generation_id
    raw = manifest.model_dump_json()
    assert "unit-test-provider-credential" not in raw
    assert "api_key" in raw
