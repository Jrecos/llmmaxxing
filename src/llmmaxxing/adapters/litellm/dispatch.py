"""Exact hidden-alias dispatch preparation and native deployment receipt fencing."""

from __future__ import annotations

from collections.abc import Mapping

from llmmaxxing.adapters.litellm.contract import (
    AdapterContract,
    CertifiedErrorDetail,
    DeploymentGenerationFingerprint,
    DeploymentReceipt,
    EffectiveDeployment,
    PreparedDispatch,
)
from llmmaxxing.adapters.litellm.guard import deployment_generation


class DispatchError(RuntimeError):
    def __init__(self, detail: CertifiedErrorDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


def _error(
    code: str,
    message: str,
    *,
    critical: bool,
    retryable: bool,
) -> DispatchError:
    return DispatchError(
        CertifiedErrorDetail(
            code=code,
            message=message,
            critical=critical,
            retryable=retryable,
        )
    )


def prepare_dispatch(
    contract: AdapterContract,
    *,
    endpoint: str,
    deployment: EffectiveDeployment,
    generation: DeploymentGenerationFingerprint,
    backend_manifest: str,
) -> PreparedDispatch:
    try:
        certified = contract.endpoint(endpoint)
    except ValueError as exc:
        raise _error(
            "unsupported_endpoint",
            str(exc),
            critical=False,
            retryable=False,
        ) from exc
    actual = deployment_generation(deployment, contract)
    if actual != generation:
        raise _error(
            "generation_mismatch",
            "prepared deployment generation does not match current semantics",
            critical=True,
            retryable=False,
        )
    metadata = {
        "alias": deployment.hidden_alias,
        "generation_id": str(generation.generation_id),
        "account_id": str(deployment.account_id),
        "account_binding": deployment.account_binding,
        "backend_manifest": backend_manifest,
        "credential_fingerprint": deployment.credential_fingerprint,
        "credential_epoch": deployment.credential_epoch,
        "contract_id": contract.contract_id,
    }
    return PreparedDispatch(
        endpoint=certified.name,
        method=certified.method,
        path=certified.path,
        model_locator=certified.model_locator,
        hidden_alias=deployment.hidden_alias,
        expected_deployment_id=deployment.runtime_id,
        generation_id=generation.generation_id,
        backend_manifest=backend_manifest,
        trusted_metadata={"llmmaxxing_guard": metadata},
    )


def reconcile_dispatch(
    contract: AdapterContract,
    prepared: PreparedDispatch,
    *,
    status_code: int,
    headers: Mapping[str, str],
    response_started: bool = False,
) -> DeploymentReceipt:
    endpoint = contract.endpoint(prepared.endpoint)
    if response_started:
        raise _error(
            "receipt_mismatch",
            "deployment receipt was not reconciled before response start",
            critical=True,
            retryable=False,
        )
    if not 200 <= status_code < 300:
        raise _error(
            "upstream_error",
            f"LiteLLM returned HTTP {status_code} before a receipt could be certified",
            critical=False,
            retryable=False,
        )
    normalized = {name.lower(): value for name, value in headers.items()}
    observed = normalized.get(endpoint.receipt_header)
    if observed is None:
        raise _error(
            "receipt_missing",
            "authoritative x-litellm-model-id receipt is missing",
            critical=True,
            retryable=False,
        )
    if observed != prepared.expected_deployment_id:
        raise _error(
            "receipt_mismatch",
            "authoritative deployment receipt does not match the prepared deployment",
            critical=True,
            retryable=False,
        )
    return DeploymentReceipt(
        endpoint=prepared.endpoint,
        deployment_id=observed,
        generation_id=prepared.generation_id,
    )
