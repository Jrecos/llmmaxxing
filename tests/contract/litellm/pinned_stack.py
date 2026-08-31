"""Launch and clean an exact isolated LiteLLM contract stack.

Controller command (Docker required):
    uv run python tests/contract/litellm/pinned_stack.py
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmmaxxing.adapters.litellm.contract import (
    AdapterContract,
    EffectiveDeployment,
    GuardDeploymentExpectation,
    GuardManifest,
    TransportResponse,
    load_contract,
)
from llmmaxxing.adapters.litellm.discovery import LiteLLMAdapter
from llmmaxxing.adapters.litellm.guard import (
    backend_manifest_revision,
    deployment_generation,
)

POSTGRES_IMAGE = "postgres@sha256:57c72fd2a128e416c7fcc499958864df5301e940bca0a56f58fddf30ffc07777"
ACCOUNT_ID = "acc_99999999-9999-4999-8999-999999999999"
ACCOUNT_BINDING = "pinned-contract-fixture"

ENDPOINT_SPECS: dict[str, dict[str, str]] = {
    "chat": {
        "alias": "fixture/chat",
        "deployment_id": "fixture-chat",
        "mode": "chat",
        "provider": "openai",
        "model": "openai/fixture-chat",
        "api_base": "http://fake-provider:8080/v1",
    },
    "text": {
        "alias": "fixture/text",
        "deployment_id": "fixture-text",
        "mode": "completion",
        "provider": "openai",
        "model": "openai/fixture-text",
        "api_base": "http://fake-provider:8080/v1",
    },
    "embeddings": {
        "alias": "fixture/embeddings",
        "deployment_id": "fixture-embeddings",
        "mode": "embedding",
        "provider": "openai",
        "model": "openai/fixture-embeddings",
        "api_base": "http://fake-provider:8080/v1",
    },
    "rerank": {
        "alias": "fixture/rerank",
        "deployment_id": "fixture-rerank",
        "mode": "rerank",
        "provider": "cohere",
        "model": "cohere/fixture-rerank",
        "api_base": "http://fake-provider:8080",
    },
    "audio_speech": {
        "alias": "fixture/audio-speech",
        "deployment_id": "fixture-audio-speech",
        "mode": "audio_speech",
        "provider": "openai",
        "model": "openai/fixture-audio-speech",
        "api_base": "http://fake-provider:8080/v1",
    },
    "audio_transcription": {
        "alias": "fixture/audio-transcription",
        "deployment_id": "fixture-audio-transcription",
        "mode": "audio_transcription",
        "provider": "openai",
        "model": "openai/fixture-audio-transcription",
        "api_base": "http://fake-provider:8080/v1",
    },
    "image": {
        "alias": "fixture/image",
        "deployment_id": "fixture-image",
        "mode": "image_generation",
        "provider": "openai",
        "model": "openai/fixture-image",
        "api_base": "http://fake-provider:8080/v1",
    },
}


@dataclass(frozen=True)
class StackMaterial:
    compose_path: Path
    config_path: Path
    manifest_path: Path
    endpoint_targets: dict[str, dict[str, str]]
    backend_manifest: str
    master_key: str
    provider_secret: str
    fingerprint_key: str


def _fingerprint(key: str, credential: str) -> str:
    return "hcf1_" + hmac.new(key.encode(), credential.encode(), hashlib.sha256).hexdigest()


def _metadata(spec: Mapping[str, str], fingerprint: str, endpoint: str) -> dict[str, Any]:
    return {
        "hidden_alias": True,
        "execution": {
            "custom_llm_provider": spec["provider"],
            "model": spec["model"],
            "api_base": spec["api_base"],
            "allow_client_keepalive_override": False,
            "merge_reasoning_content_in_choices": False,
            "use_in_pass_through": False,
            "use_litellm_proxy": False,
            "use_xai_oauth": False,
        },
        "capabilities": {"endpoints": [endpoint], "modalities": ["text"]},
        "context": {"max_input_tokens": 8192, "max_output_tokens": 1024},
        "defaults": {},
        "pricing": {"input_per_token": "0", "output_per_token": "0"},
        "account": {"id": ACCOUNT_ID, "binding": ACCOUNT_BINDING},
        "credential": {
            "field": "api_key",
            "fingerprint": fingerprint,
            "epoch": 1,
            "dynamic_list": False,
        },
        "routing": {
            "num_retries": 0,
            "fallbacks": [],
            "context_window_fallbacks": [],
            "content_policy_fallbacks": [],
            "cooldown_selection": False,
        },
    }


def materialize_stack(root: Path) -> StackMaterial:
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    contract = load_contract()
    master_key = secrets.token_urlsafe(32)
    provider_secret = secrets.token_urlsafe(32)
    fingerprint_key = secrets.token_urlsafe(32)
    postgres_password = secrets.token_urlsafe(32)
    fingerprint = _fingerprint(fingerprint_key, provider_secret)

    model_list: list[dict[str, Any]] = []
    expected: dict[str, GuardDeploymentExpectation] = {}
    rows: list[EffectiveDeployment] = []
    for endpoint, spec in ENDPOINT_SPECS.items():
        metadata = _metadata(spec, fingerprint, endpoint)
        model_list.append(
            {
                "model_name": spec["alias"],
                "litellm_params": {
                    "model": spec["model"],
                    "custom_llm_provider": spec["provider"],
                    "api_base": spec["api_base"],
                    "api_key": "os.environ/FAKE_API_KEY",
                    "num_retries": 0,
                },
                "model_info": {
                    "id": spec["deployment_id"],
                    "mode": spec["mode"],
                    "llmmaxxing": metadata,
                },
            }
        )
        row = EffectiveDeployment(
            runtime_id=spec["deployment_id"],
            hidden_alias=spec["alias"],
            mode=spec["mode"],
            execution=metadata["execution"],
            capabilities=metadata["capabilities"],
            context=metadata["context"],
            defaults=metadata["defaults"],
            pricing=metadata["pricing"],
            account_id=ACCOUNT_ID,
            account_binding=ACCOUNT_BINDING,
            credential_field="api_key",
            credential_fingerprint=fingerprint,
            credential_epoch=1,
        )
        generation = deployment_generation(row, contract)
        rows.append(row)
        expected[spec["alias"]] = GuardDeploymentExpectation(
            runtime_id=row.runtime_id,
            generation_id=generation.generation_id,
            credential_field=row.credential_field,
            projection=generation.projection,
        )

    config = {
        "model_list": model_list,
        "litellm_settings": {"callbacks": ["llmmaxxing_guard.llmmaxxing_guard"]},
        "router_settings": {
            "disable_cooldowns": True,
            "num_retries": 0,
            "fallbacks": [],
            "context_window_fallbacks": [],
            "content_policy_fallbacks": [],
        },
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "allow_public_health_readiness_details": False,
        },
    }
    backend_manifest = backend_manifest_revision(contract, rows)
    manifest = GuardManifest(
        contract_id=contract.contract_id,
        backend_manifest=backend_manifest,
        guard_digest=contract.guard.digest,
        deployments=expected,
    )

    config_path = root / "config.yaml"
    manifest_path = root / "guard-manifest.json"
    compose_path = root / "compose.yaml"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    fake_provider = Path(__file__).with_name("fake_provider.py").resolve()
    guard = Path(__file__).resolve().parents[3] / contract.guard.mount_path
    compose = {
        "services": {
            "postgres": {
                "image": POSTGRES_IMAGE,
                "environment": {
                    "POSTGRES_USER": "litellm",
                    "POSTGRES_PASSWORD": postgres_password,
                    "POSTGRES_DB": "litellm",
                },
                "tmpfs": ["/var/lib/postgresql/data"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U litellm -d litellm"],
                    "interval": "2s",
                    "timeout": "2s",
                    "retries": 60,
                },
            },
            "fake-provider": {
                "image": contract.litellm.image,
                "entrypoint": ["python", "/fixture/fake_provider.py"],
                "ports": ["127.0.0.1::8080"],
                "volumes": [f"{fake_provider}:/fixture/fake_provider.py:ro"],
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')",
                    ],
                    "interval": "2s",
                    "timeout": "2s",
                    "retries": 30,
                },
            },
            "litellm": {
                "image": contract.litellm.image,
                "command": ["--config", "/app/config.yaml", "--port", "4000"],
                "ports": ["127.0.0.1::4000"],
                "environment": {
                    "DATABASE_URL": (
                        "postgresql://litellm:"
                        f"{urllib.parse.quote(postgres_password, safe='')}@postgres:5432/litellm"
                    ),
                    "LITELLM_MASTER_KEY": master_key,
                    "FAKE_API_KEY": provider_secret,
                    "STORE_MODEL_IN_DB": "False",
                    "LLMMAXXING_GUARD_MANIFEST": "/app/guard-manifest.json",
                    "LLMMAXXING_GUARD_FINGERPRINT_KEY": fingerprint_key,
                    "PYTHONPATH": "/app",
                },
                "volumes": [
                    f"{config_path}:/app/config.yaml:ro",
                    f"{manifest_path}:/app/guard-manifest.json:ro",
                    f"{guard}:/app/llmmaxxing_guard.py:ro",
                ],
                "depends_on": {
                    "fake-provider": {"condition": "service_healthy"},
                    "postgres": {"condition": "service_healthy"},
                },
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import urllib.request;urllib.request.urlopen('http://localhost:4000/health/readiness')",
                    ],
                    "interval": "3s",
                    "timeout": "3s",
                    "retries": 80,
                    "start_period": "10s",
                },
            },
        }
    }
    compose_path.write_text(json.dumps(compose, indent=2) + "\n")
    targets = {
        endpoint: {"alias": spec["alias"], "deployment_id": spec["deployment_id"]}
        for endpoint, spec in ENDPOINT_SPECS.items()
    }
    return StackMaterial(
        compose_path=compose_path,
        config_path=config_path,
        manifest_path=manifest_path,
        endpoint_targets=targets,
        backend_manifest=backend_manifest,
        master_key=master_key,
        provider_secret=provider_secret,
        fingerprint_key=fingerprint_key,
    )


def key_requests(
    contract: AdapterContract,
    inference_user: str,
    discovery_user: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = sorted(target["alias"] for target in ENDPOINT_SPECS.values())
    inference = {
        "user_id": inference_user,
        "key_alias": "llmmaxxing-contract-inference",
        "models": aliases,
        "allowed_routes": list(contract.service_keys.inference.allowed_routes),
        "key_type": "default",
    }
    discovery = {
        "user_id": discovery_user,
        "key_alias": "llmmaxxing-contract-discovery",
        "models": list(contract.service_keys.discovery.models),
        "allowed_routes": list(contract.service_keys.discovery.allowed_routes),
        "key_type": "default",
    }
    return inference, discovery


def _request(
    base_url: str,
    method: str,
    path: str,
    key: str,
    body: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base_url + path,
        data=raw,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            **({"Content-Type": "application/json"} if raw is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
            return response.status, payload
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read())
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        return exc.code, payload
    except (urllib.error.URLError, ConnectionError):
        return 0, {}


class _DiscoveryTransport:
    def __init__(self, base_url: str, key: str) -> None:
        self.base_url = base_url
        self.key = key

    async def request(
        self,
        method: str,
        path: str,
        *,
        key: str,
        query: Mapping[str, str] | None = None,
    ) -> TransportResponse:
        if key != "discovery":
            raise RuntimeError("launcher transport only owns the discovery key")
        target = path + (("?" + urllib.parse.urlencode(query)) if query else "")
        status, payload = await asyncio.to_thread(
            _request,
            self.base_url,
            method,
            target,
            self.key,
        )
        return TransportResponse(status_code=status, headers={}, body=payload)


def _compose(
    material: StackMaterial,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", "compose", "-f", str(material.compose_path), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(f"docker compose {' '.join(args)} failed")
    return result


def _wait_ready(base_url: str, timeout: float = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, _ = _request(base_url, "GET", "/health/readiness", "unused")
        if status == 200:
            return
        time.sleep(2)
    raise RuntimeError("pinned LiteLLM did not become ready")


def _wait_guard_last(
    base_url: str,
    master_key: str,
    identity: str,
    timeout: float = 60,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, payload = _request(base_url, "GET", "/active/callbacks", master_key)
        callbacks = payload.get("litellm.callbacks")
        if (
            status == 200
            and isinstance(callbacks, list)
            and callbacks.count(identity) == 1
            and callbacks[-1] == identity
        ):
            return
        time.sleep(0.1)
    raise RuntimeError("certified generation guard did not register once and last")


def _published_litellm_url(material: StackMaterial) -> str:
    port = _compose(material, "port", "litellm", "4000").stdout.strip().rsplit(":", 1)[-1]
    return f"http://127.0.0.1:{port}"


def _recreate_litellm(
    material: StackMaterial,
    master_key: str,
    guard_identity: str,
) -> str:
    _compose(material, "up", "-d", "--force-recreate", "--no-deps", "litellm")
    base_url = _published_litellm_url(material)
    _wait_ready(base_url)
    _wait_guard_last(base_url, master_key, guard_identity)
    return base_url


def _inspect_source_files(
    material: StackMaterial,
    contract: AdapterContract,
) -> dict[str, str]:
    paths = sorted(contract.litellm.source_files)
    script = (
        "import hashlib,json,pathlib,litellm;"
        "root=pathlib.Path(litellm.__file__).resolve().parent.parent;"
        f"paths={paths!r};"
        "print(json.dumps({p:'sha256:'+hashlib.sha256((root/p).read_bytes()).hexdigest() "
        "for p in paths},sort_keys=True))"
    )
    result = _compose(material, "exec", "-T", "litellm", "python", "-c", script)
    observed = json.loads(result.stdout)
    if observed != contract.litellm.source_files:
        raise RuntimeError("pinned image source-file digests differ from compatibility manifest")
    return observed


def _set_provider_secret(material: StackMaterial, value: str) -> None:
    compose = json.loads(material.compose_path.read_text())
    compose["services"]["litellm"]["environment"]["FAKE_API_KEY"] = value
    material.compose_path.write_text(json.dumps(compose, indent=2) + "\n")


def _run_pinned_test(environment: dict[str, str]) -> int:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/contract/litellm/test_pinned_runtime.py",
            "-q",
            "-ra",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        check=False,
    )
    return result.returncode


def _provision_key(
    base_url: str,
    master: str,
    user_id: str,
    role: str,
    request: dict[str, Any],
) -> str:
    status, _ = _request(
        base_url,
        "POST",
        "/user/new",
        master,
        {"user_id": user_id, "user_role": role, "auto_create_key": False},
    )
    if status != 200:
        raise RuntimeError("failed to create isolated LiteLLM contract user")
    status, response = _request(base_url, "POST", "/key/generate", master, request)
    key = response.get("key")
    if status != 200 or not isinstance(key, str) or not key:
        raise RuntimeError("failed to create isolated LiteLLM contract key")
    return key


def main() -> int:
    contract = load_contract()
    preserve = os.environ.get("LLMMAXXING_PINNED_DEBUG_PRESERVE") == "1"
    with tempfile.TemporaryDirectory(
        prefix="llmmaxxing-litellm-1.98-",
        delete=not preserve,
    ) as temporary:
        material = materialize_stack(Path(temporary))
        project = "lmx-contract-" + secrets.token_hex(4)
        os.environ["COMPOSE_PROJECT_NAME"] = project
        try:
            _compose(material, "up", "-d", "--wait", "--wait-timeout", "300")
            base_url = _published_litellm_url(material)
            _wait_ready(base_url)
            _wait_guard_last(
                base_url,
                material.master_key,
                contract.guard.active_callback_identity,
            )

            inference_user = "contract-inference-" + secrets.token_hex(8)
            discovery_user = "contract-discovery-" + secrets.token_hex(8)
            inference_request, discovery_request = key_requests(
                contract,
                inference_user,
                discovery_user,
            )
            inference_key = _provision_key(
                base_url,
                material.master_key,
                inference_user,
                contract.service_keys.inference.user_role,
                inference_request,
            )
            discovery_key = _provision_key(
                base_url,
                material.master_key,
                discovery_user,
                contract.service_keys.discovery.user_role,
                discovery_request,
            )

            adapter = LiteLLMAdapter(contract, _DiscoveryTransport(base_url, discovery_key))
            snapshot = asyncio.run(adapter.discover_complete())
            from llmmaxxing.adapters.litellm.guard import build_guard_manifest

            configured_manifest = GuardManifest.model_validate_json(
                material.manifest_path.read_text()
            )
            live_manifest = build_guard_manifest(snapshot, contract)
            if live_manifest != configured_manifest:
                raise RuntimeError("live backend manifest differs from immutable guard manifest")
            source_files = _inspect_source_files(material, contract)
            provider_port = (
                _compose(material, "port", "fake-provider", "8080")
                .stdout.strip()
                .rsplit(":", 1)[-1]
            )

            environment = os.environ.copy()
            environment.update(
                {
                    "LLMMAXXING_PINNED_LITELLM_URL": base_url,
                    "LLMMAXXING_PINNED_PROVIDER_URL": f"http://127.0.0.1:{provider_port}",
                    "LLMMAXXING_PINNED_DISCOVERY_KEY": discovery_key,
                    "LLMMAXXING_PINNED_INFERENCE_KEY": inference_key,
                    "LLMMAXXING_PINNED_ENDPOINT_TARGETS_JSON": json.dumps(
                        material.endpoint_targets
                    ),
                    "LLMMAXXING_PINNED_IMAGE": contract.litellm.image,
                    "LLMMAXXING_PINNED_SOURCE_FILES_JSON": json.dumps(source_files),
                }
            )

            _set_provider_secret(material, secrets.token_urlsafe(32))
            base_url = _recreate_litellm(
                material,
                material.master_key,
                contract.guard.active_callback_identity,
            )
            environment["LLMMAXXING_PINNED_LITELLM_URL"] = base_url
            swapped_environment = {
                **environment,
                "LLMMAXXING_PINNED_EXPECT_SECRET_SWAP": "1",
            }
            if _run_pinned_test(swapped_environment):
                return 1

            _set_provider_secret(material, material.provider_secret)
            base_url = _recreate_litellm(
                material,
                material.master_key,
                contract.guard.active_callback_identity,
            )
            environment["LLMMAXXING_PINNED_LITELLM_URL"] = base_url
            environment.pop("LLMMAXXING_PINNED_EXPECT_SECRET_SWAP", None)
            return _run_pinned_test(environment)
        finally:
            if preserve:
                print(
                    f"Preserving pinned stack {project} at {material.compose_path}",
                    file=sys.stderr,
                )
            else:
                _compose(material, "down", "-v", "--remove-orphans", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
