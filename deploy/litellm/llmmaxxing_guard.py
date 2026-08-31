"""Pinned LiteLLM 1.98 deployment guard, mounted directly into the proxy image.

The certified hook runs after deployment selection and before provider I/O.  It
has no dependency on the llmmaxxing package so the stock pinned image can load
this single file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # Unit contract tests intentionally do not install LiteLLM.
    class CustomLogger:  # type: ignore[no-redef]
        pass


class GuardConfigurationError(RuntimeError):
    pass


_CERTIFIED_EXECUTION_NORMALIZERS: dict[str, dict[str, str]] = {
    "chat": {},
    "text": {},
    "messages": {},
    "embeddings": {},
    "rerank": {},
    "audio_speech": {},
    "audio_transcription": {},
    "image": {},
}




class GuardViolation(RuntimeError):
    pass


def credential_fingerprint(hmac_key: bytes, credential: str) -> str:
    if not hmac_key or not isinstance(credential, str) or not credential:
        raise GuardConfigurationError("credential fingerprint input is empty")
    return "hcf1_" + hmac.new(hmac_key, credential.encode(), hashlib.sha256).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    error: type[RuntimeError] = GuardConfigurationError,
) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise error(f"unknown {label} fields: {sorted(unknown)}")
    if missing:
        raise error(f"missing {label} fields: {sorted(missing)}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validated_manifest(value: Mapping[str, Any], digest: str) -> Mapping[str, Any]:
    _require_exact_keys(
        value,
        {"contract_id", "backend_manifest", "guard_digest", "deployments"},
        "guard manifest",
    )
    if value["contract_id"] != "litellm-1.98.0":
        raise GuardConfigurationError("guard contract id mismatch")
    if value["guard_digest"] != f"sha256:{digest}":
        raise GuardConfigurationError("guard digest mismatch")
    backend = value["backend_manifest"]
    if not isinstance(backend, str) or not backend.startswith("bm1_") or len(backend) != 68:
        raise GuardConfigurationError("backend manifest id is malformed")
    deployments = value["deployments"]
    if not isinstance(deployments, dict) or not deployments:
        raise GuardConfigurationError("guard deployment manifest is empty")
    deployment_fields = {
        "runtime_id",
        "generation_id",
        "credential_field",
        "projection",
    }
    projection_fields = {
        "contract_id",
        "hidden_alias",
        "mode",
        "execution",
        "capabilities",
        "context",
        "defaults",
        "pricing",
        "account_id",
        "account_binding",
        "credential_fingerprint",
        "credential_epoch",
    }
    for alias, record in deployments.items():
        if not isinstance(alias, str) or not alias or not isinstance(record, dict):
            raise GuardConfigurationError("guard deployment entry is malformed")
        _require_exact_keys(record, deployment_fields, "guard deployment")
        if not isinstance(record["runtime_id"], str) or not record["runtime_id"]:
            raise GuardConfigurationError("guard runtime id is malformed")
        generation = record["generation_id"]
        if not isinstance(generation, str) or not generation.startswith("dg1_") or len(generation) != 68:
            raise GuardConfigurationError("guard generation id is malformed")
        if not isinstance(record["credential_field"], str) or not record["credential_field"]:
            raise GuardConfigurationError("guard credential field is empty")
        projection = record["projection"]
        if not isinstance(projection, dict):
            raise GuardConfigurationError("guard semantic projection is malformed")
        _require_exact_keys(projection, projection_fields, "guard semantic projection")
        if projection["contract_id"] != value["contract_id"] or projection["hidden_alias"] != alias:
            raise GuardConfigurationError("guard semantic projection identity mismatch")
        if not isinstance(projection["mode"], str) or not projection["mode"]:
            raise GuardConfigurationError("guard semantic projection mode is malformed")
        for field in ("execution", "capabilities", "context", "defaults", "pricing"):
            if not isinstance(projection[field], dict):
                raise GuardConfigurationError(f"guard semantic projection {field} is malformed")
        if not projection["execution"]:
            raise GuardConfigurationError("guard execution projection is empty")
        if not isinstance(projection["account_id"], str) or not projection["account_id"].startswith(
            "acc_"
        ):
            raise GuardConfigurationError("guard account id is malformed")
        if not isinstance(projection["account_binding"], str) or not projection["account_binding"]:
            raise GuardConfigurationError("guard account binding is empty")
        fingerprint = projection["credential_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint.startswith("hcf1_") or len(fingerprint) != 69:
            raise GuardConfigurationError("guard credential fingerprint is malformed")
        if (
            not isinstance(projection["credential_epoch"], int)
            or projection["credential_epoch"] < 1
        ):
            raise GuardConfigurationError("guard credential epoch is malformed")
    return _freeze(json.loads(json.dumps(value)))


def _metadata(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    found: list[Mapping[str, Any]] = []
    for name in ("metadata", "litellm_metadata"):
        value = kwargs.get(name)
        if isinstance(value, Mapping):
            found.append(value)
    fences = [
        value["llmmaxxing_guard"]
        for value in found
        if isinstance(value.get("llmmaxxing_guard"), Mapping)
    ]
    if not fences:
        raise GuardViolation("server-side guard fence is required")
    first = dict(fences[0])
    if any(dict(fence) != first for fence in fences[1:]):
        raise GuardViolation("server-side guard fence is ambiguous")
    return fences[0]


def _model_group(kwargs: Mapping[str, Any]) -> str:
    groups = {
        value["model_group"]
        for name in ("metadata", "litellm_metadata")
        if isinstance((value := kwargs.get(name)), Mapping) and isinstance(value.get("model_group"), str)
    }
    if len(groups) != 1:
        raise GuardViolation("selected hidden alias is missing or ambiguous")
    return next(iter(groups))


def _model_info(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = kwargs.get("model_info")
    if isinstance(direct, Mapping):
        return direct
    for name in ("litellm_metadata", "metadata"):
        metadata = kwargs.get(name)
        if isinstance(metadata, Mapping) and isinstance(metadata.get("model_info"), Mapping):
            return metadata["model_info"]
    raise GuardViolation("selected deployment runtime metadata is missing")


def _resolved_credential(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        raise GuardViolation("dynamic credential-list forms are unsupported")
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        value = getter()
    if not isinstance(value, str) or not value:
        raise GuardViolation("resolved provider credential is missing")
    if value.startswith("os.environ/"):
        env_name = value.removeprefix("os.environ/")
        value = os.environ.get(env_name, "")
        if not value:
            raise GuardViolation("resolved provider credential environment reference is missing")
    return value


class LLMMaxxingGuard(CustomLogger):
    callback_name = "llmmaxxing_guard"

    def __init__(self, *, manifest: Mapping[str, Any], hmac_key: bytes, guard_digest: str) -> None:
        if len(hmac_key) < 16:
            raise GuardConfigurationError("credential-fingerprint HMAC key is too short")
        if len(guard_digest) != 64 or any(ch not in "0123456789abcdef" for ch in guard_digest):
            raise GuardConfigurationError("guard digest is malformed")
        self._hmac_key = bytes(hmac_key)
        self._digest = guard_digest
        self._manifest = _validated_manifest(manifest, guard_digest)

    def __str__(self) -> str:
        return f"{self.callback_name}@sha256:{self._digest}"

    @classmethod
    def from_environment(cls) -> "LLMMaxxingGuard":
        manifest_path = os.environ.get("LLMMAXXING_GUARD_MANIFEST", "")
        key_file = os.environ.get("LLMMAXXING_GUARD_FINGERPRINT_KEY_FILE", "")
        key_text = os.environ.get("LLMMAXXING_GUARD_FINGERPRINT_KEY", "")
        if not manifest_path or not (key_file or key_text):
            raise GuardConfigurationError("guard manifest and separate fingerprint key are required")
        try:
            manifest = json.loads(Path(manifest_path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise GuardConfigurationError("guard manifest cannot be loaded") from exc
        try:
            hmac_key = Path(key_file).read_bytes().strip() if key_file else key_text.encode()
        except OSError as exc:
            raise GuardConfigurationError("guard fingerprint key cannot be loaded") from exc
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return cls(manifest=manifest, hmac_key=hmac_key, guard_digest=digest)

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: Any) -> dict[str, Any]:
        del call_type
        if kwargs.get("litellm_credential_name") is not None or kwargs.get("credentials") is not None:
            raise GuardViolation("dynamic credential-list forms are unsupported")

        fence = _metadata(kwargs)
        _require_exact_keys(
            fence,
            {
                "alias",
                "generation_id",
                "account_id",
                "account_binding",
                "backend_manifest",
                "credential_fingerprint",
                "credential_epoch",
                "contract_id",
                "endpoint",
            },
            "request fence",
            GuardViolation,
        )
        alias = _model_group(kwargs)
        if fence["alias"] != alias:
            raise GuardViolation("expected alias does not match selected alias")
        deployments = self._manifest["deployments"]
        expected = deployments.get(alias)
        if not isinstance(expected, Mapping):
            raise GuardViolation("selected alias is absent from immutable guard manifest")
        if fence["contract_id"] != self._manifest["contract_id"]:
            raise GuardViolation("contract fence mismatch")
        if fence["backend_manifest"] != self._manifest["backend_manifest"]:
            raise GuardViolation("backend manifest fence mismatch")

        info = _model_info(kwargs)
        if info.get("id") != expected["runtime_id"]:
            raise GuardViolation("selected runtime deployment mismatch")
        attestation = info.get("llmmaxxing")
        if not isinstance(attestation, Mapping):
            raise GuardViolation("selected deployment attestation is missing")
        account = attestation.get("account")
        credential = attestation.get("credential")
        if not isinstance(account, Mapping) or not isinstance(credential, Mapping):
            raise GuardViolation("selected deployment account or credential attestation is missing")
        semantics: dict[str, dict[str, Any]] = {}
        for field in ("capabilities", "context", "defaults", "pricing"):
            value = attestation.get(field)
            if not isinstance(value, Mapping):
                raise GuardViolation(f"selected deployment {field} attestation is missing")
            semantics[field] = dict(value)

        projection = expected["projection"]
        comparisons = {
            "generation_id": expected["generation_id"],
            "account_id": projection["account_id"],
            "account_binding": projection["account_binding"],
            "credential_fingerprint": projection["credential_fingerprint"],
            "credential_epoch": projection["credential_epoch"],
        }
        if any(fence.get(name) != value for name, value in comparisons.items()):
            raise GuardViolation("generation, account, or credential fence mismatch")

        endpoint = fence["endpoint"]
        normalizers = _CERTIFIED_EXECUTION_NORMALIZERS.get(endpoint)
        if normalizers is None:
            raise GuardViolation("uncertified endpoint fence")
        expected_execution = dict(projection["execution"])
        credential_value = _resolved_credential(kwargs.get(expected["credential_field"]))
        actual_fingerprint = credential_fingerprint(self._hmac_key, credential_value)
        if not hmac.compare_digest(
            actual_fingerprint,
            projection["credential_fingerprint"],
        ):
            raise GuardViolation("resolved provider credential fingerprint mismatch")
        current_projection = {
            "contract_id": self._manifest["contract_id"],
            "hidden_alias": alias,
            "mode": info.get("mode"),
            "execution": {name: kwargs.get(name) for name in projection["execution"]},
            "capabilities": semantics["capabilities"],
            "context": semantics["context"],
            "defaults": semantics["defaults"],
            "pricing": semantics["pricing"],
            "account_id": account.get("id"),
            "account_binding": account.get("binding"),
            "credential_fingerprint": actual_fingerprint,
            "credential_epoch": credential.get("epoch"),
        }
        expected_projection = {
            **dict(projection),
            "execution": expected_execution,
        }
        frozen_current = _freeze(current_projection)
        mismatches = [
            field
            for field in expected_projection
            if frozen_current[field] != expected_projection[field]
        ]
        if "execution" in mismatches:
            execution_mismatches = [
                field
                for field in expected_execution
                if frozen_current["execution"][field] != expected_execution[field]
            ]
            mismatches[mismatches.index("execution")] = (
                "execution(" + ",".join(execution_mismatches) + ")"
            )
        if mismatches:
            raise GuardViolation(
                "selected deployment semantic execution projection mismatch: "
                + ",".join(mismatches)
            )
        return kwargs


def _move_guard_last(callbacks: list[Any], guard: LLMMaxxingGuard) -> None:
    callbacks[:] = [callback for callback in callbacks if callback is not guard]
    callbacks.append(guard)


def _enforce_last_registration(guard: LLMMaxxingGuard) -> None:
    def run() -> None:
        try:
            import litellm
        except ImportError:
            return
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            callbacks = litellm.callbacks
            if isinstance(callbacks, list) and guard in callbacks:
                if callbacks.count(guard) != 1 or callbacks[-1] is not guard:
                    _move_guard_last(callbacks, guard)
            time.sleep(0.05)

    threading.Thread(
        target=run,
        name="llmmaxxing-guard-order",
        daemon=True,
    ).start()


def _configured_guard() -> LLMMaxxingGuard | None:
    configured = any(
        os.environ.get(name)
        for name in (
            "LLMMAXXING_GUARD_MANIFEST",
            "LLMMAXXING_GUARD_FINGERPRINT_KEY_FILE",
            "LLMMAXXING_GUARD_FINGERPRINT_KEY",
        )
    )
    return LLMMaxxingGuard.from_environment() if configured else None


llmmaxxing_guard = _configured_guard()
if llmmaxxing_guard is not None:
    _enforce_last_registration(llmmaxxing_guard)
custom_callback = llmmaxxing_guard
