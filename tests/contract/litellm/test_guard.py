from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures"
GUARD_PATH = ROOT / "deploy/litellm/llmmaxxing_guard.py"


def load_guard_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("certified_llmmaxxing_guard", GUARD_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_guard() -> tuple[ModuleType, Any, dict[str, Any], str]:
    module = load_guard_module()
    hmac_key = b"unit-test-guard-fingerprint-key"
    provider_secret = "unit-test-provider-credential"
    digest = hashlib.sha256(GUARD_PATH.read_bytes()).hexdigest()
    manifest = json.loads((FIXTURES / "guard-manifest.json").read_text())
    manifest = json.loads(json.dumps(manifest).replace("__GUARD_DIGEST__", digest))
    manifest["deployments"]["lmx/electron-v1"]["projection"]["credential_fingerprint"] = (
        module.credential_fingerprint(hmac_key, provider_secret)
    )
    guard = module.LLMMaxxingGuard(manifest=manifest, hmac_key=hmac_key, guard_digest=digest)
    return module, guard, manifest, provider_secret


def request_kwargs(manifest: dict[str, Any], provider_secret: str) -> dict[str, Any]:
    expected = manifest["deployments"]["lmx/electron-v1"]
    projection = expected["projection"]
    return {
        **projection["execution"],
        "api_key": provider_secret,
        "model_info": {
            "id": expected["runtime_id"],
            "mode": projection["mode"],
            "llmmaxxing": {
                "capabilities": deepcopy(projection["capabilities"]),
                "context": deepcopy(projection["context"]),
                "defaults": deepcopy(projection["defaults"]),
                "pricing": deepcopy(projection["pricing"]),
                "account": {
                    "id": projection["account_id"],
                    "binding": projection["account_binding"],
                },
                "credential": {"epoch": projection["credential_epoch"]},
            },
        },
        "metadata": {
            "model_group": projection["hidden_alias"],
            "llmmaxxing_guard": {
                "alias": projection["hidden_alias"],
                "generation_id": expected["generation_id"],
                "account_id": projection["account_id"],
                "account_binding": projection["account_binding"],
                "backend_manifest": manifest["backend_manifest"],
                "credential_fingerprint": projection["credential_fingerprint"],
                "credential_epoch": projection["credential_epoch"],
                "contract_id": projection["contract_id"],
                "endpoint": "chat",
            },
        },
    }


async def guarded_provider_call(
    guard: Any,
    kwargs: dict[str, Any],
    calls: list[dict[str, Any]],
) -> None:
    await guard.async_pre_call_deployment_hook(kwargs, None)
    calls.append(kwargs)


def test_guard_has_stable_active_callbacks_digest_identity() -> None:
    _, guard, _, _ = make_guard()
    digest = hashlib.sha256(GUARD_PATH.read_bytes()).hexdigest()
    assert str(guard) == f"llmmaxxing_guard@sha256:{digest}"
    assert guard.callback_name == "llmmaxxing_guard"


def test_guard_registration_helper_deduplicates_and_moves_last() -> None:
    module, guard, _, _ = make_guard()
    other = object()
    callbacks = [guard, other, guard]
    module._move_guard_last(callbacks, guard)
    assert callbacks == [other, guard]


def test_guard_accepts_exact_atomic_fence_before_provider_call() -> None:
    _, guard, manifest, secret = make_guard()
    calls: list[dict[str, Any]] = []
    kwargs = request_kwargs(manifest, secret)
    result = asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None))
    assert result is kwargs
    asyncio.run(guarded_provider_call(guard, kwargs, calls))
    assert len(calls) == 1


def test_guard_accepts_identical_metadata_copy_but_rejects_conflicting_copy() -> None:
    module, guard, manifest, secret = make_guard()
    kwargs = request_kwargs(manifest, secret)
    kwargs["litellm_metadata"] = deepcopy(kwargs["metadata"])
    assert asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None)) is kwargs

    kwargs["litellm_metadata"]["llmmaxxing_guard"]["alias"] = "lmx/other"
    with pytest.raises(module.GuardViolation, match="ambiguous"):
        asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None))


def test_messages_compares_raw_provider_qualified_model_exactly() -> None:
    module, guard, manifest, secret = make_guard()
    kwargs = request_kwargs(manifest, secret)
    kwargs["metadata"]["llmmaxxing_guard"]["endpoint"] = "messages"
    assert asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None)) is kwargs

    for model in ("other/electron-v1", "electron/electron-v2", "electron-v1"):
        kwargs["model"] = model
        with pytest.raises(module.GuardViolation, match="execution"):
            asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None))


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda k: k["metadata"].update(model_group="lmx/other"), "alias"),
        (lambda k: k["model_info"].update(id="runtime-other"), "runtime"),
        (
            lambda k: k["metadata"]["llmmaxxing_guard"].update(generation_id="dg1_" + "0" * 64),
            "generation",
        ),
        (
            lambda k: k["metadata"]["llmmaxxing_guard"].update(account_binding="other"),
            "account",
        ),
        (
            lambda k: k["metadata"]["llmmaxxing_guard"].update(backend_manifest="bm1_" + "0" * 64),
            "backend",
        ),
        (
            lambda k: k["metadata"]["llmmaxxing_guard"].update(credential_epoch=8),
            "credential",
        ),
        (lambda k: k.update(api_base="https://repointed.invalid/v1"), "projection"),
        (lambda k: k["model_info"].update(mode="embedding"), "projection"),
        (
            lambda k: k["model_info"]["llmmaxxing"]["capabilities"].update(tools=False),
            "projection",
        ),
        (
            lambda k: k["model_info"]["llmmaxxing"]["context"].update(max_input_tokens=1),
            "projection",
        ),
        (
            lambda k: k["model_info"]["llmmaxxing"]["defaults"].update(temperature=1),
            "projection",
        ),
        (
            lambda k: k["model_info"]["llmmaxxing"]["pricing"].update(input_per_token="9"),
            "projection",
        ),
    ),
)
def test_alias_generation_account_backend_and_execution_drift_stop_before_provider(
    mutator: Any, message: str
) -> None:
    module, guard, manifest, secret = make_guard()
    kwargs = request_kwargs(manifest, secret)
    mutator(kwargs)
    calls: list[dict[str, Any]] = []
    with pytest.raises(module.GuardViolation, match=message):
        asyncio.run(guarded_provider_call(guard, kwargs, calls))
    assert calls == []


def test_vertex_project_and_location_are_part_of_the_live_guard_projection() -> None:
    module, _, manifest, secret = make_guard()
    expected = manifest["deployments"]["lmx/electron-v1"]["projection"]["execution"]
    expected.update(vertex_project="certified-project", vertex_location="europe-west4")
    digest = hashlib.sha256(GUARD_PATH.read_bytes()).hexdigest()
    guard = module.LLMMaxxingGuard(
        manifest=manifest,
        hmac_key=b"unit-test-guard-fingerprint-key",
        guard_digest=digest,
    )
    kwargs = request_kwargs(manifest, secret)
    assert asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None)) is kwargs

    kwargs["vertex_location"] = "us-central1"
    calls: list[dict[str, Any]] = []
    with pytest.raises(module.GuardViolation, match="projection"):
        asyncio.run(guarded_provider_call(guard, kwargs, calls))
    assert calls == []


def test_secret_swap_stops_before_provider_without_exposing_secret() -> None:
    module, guard, manifest, secret = make_guard()
    kwargs = request_kwargs(manifest, secret)
    kwargs["api_key"] = "rotated-provider-credential"
    calls: list[dict[str, Any]] = []
    with pytest.raises(module.GuardViolation, match="credential") as raised:
        asyncio.run(guarded_provider_call(guard, kwargs, calls))
    assert calls == []
    assert secret not in str(raised.value)
    assert kwargs["api_key"] not in str(raised.value)


def test_dynamic_credential_list_is_explicitly_unsupported_before_provider() -> None:
    module, guard, manifest, secret = make_guard()
    kwargs = request_kwargs(manifest, secret)
    kwargs["api_key"] = ["credential-a", "credential-b"]
    calls: list[dict[str, Any]] = []
    with pytest.raises(module.GuardViolation, match="dynamic credential"):
        asyncio.run(guarded_provider_call(guard, kwargs, calls))
    assert calls == []


def test_guard_rejects_missing_fence_and_unknown_manifest_fields() -> None:
    module, guard, manifest, secret = make_guard()
    kwargs = request_kwargs(manifest, secret)
    del kwargs["metadata"]["llmmaxxing_guard"]
    with pytest.raises(module.GuardViolation, match="fence"):
        asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None))

    kwargs = request_kwargs(manifest, secret)
    kwargs["metadata"]["llmmaxxing_guard"]["caller_extra"] = True
    with pytest.raises(module.GuardViolation, match="unknown"):
        asyncio.run(guard.async_pre_call_deployment_hook(kwargs, None))

    bad_manifest = deepcopy(manifest)
    bad_manifest["surprise"] = True
    with pytest.raises(module.GuardConfigurationError, match="unknown"):
        module.LLMMaxxingGuard(
            manifest=bad_manifest,
            hmac_key=b"unit-test-guard-fingerprint-key",
            guard_digest="0" * 64,
        )
