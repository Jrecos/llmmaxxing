"""Semantic deployment generation projection and RFC8785 fingerprint."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from llmmaxxing.adapters.litellm.contract import (
    AdapterContract,
    DeploymentGenerationFingerprint,
    DeploymentGenerationProjection,
    DiscoverySnapshot,
    EffectiveDeployment,
    GuardDeploymentExpectation,
    GuardManifest,
)
from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.ids import DeploymentGenerationId


def deployment_generation(
    row: EffectiveDeployment,
    contract: AdapterContract,
) -> DeploymentGenerationFingerprint:
    projection = DeploymentGenerationProjection(
        contract_id=contract.contract_id,
        hidden_alias=row.hidden_alias,
        mode=row.mode,
        execution=row.execution,
        capabilities=row.capabilities,
        context=row.context,
        defaults=row.defaults,
        pricing=row.pricing,
        account_id=row.account_id,
        account_binding=row.account_binding,
        credential_fingerprint=row.credential_fingerprint,
        credential_epoch=row.credential_epoch,
    )
    digest = hashlib.sha256(canonical_json_bytes(projection.model_dump(mode="json"))).hexdigest()
    return DeploymentGenerationFingerprint(
        generation_id=DeploymentGenerationId.from_digest(digest),
        projection=projection,
    )


def backend_manifest_revision(
    contract: AdapterContract,
    deployments: Sequence[EffectiveDeployment],
) -> str:
    projections = sorted(
        (
            deployment_generation(row, contract).projection.model_dump(mode="json")
            for row in deployments
        ),
        key=lambda projection: projection["hidden_alias"],
    )
    payload = {
        "contract_id": contract.contract_id,
        "build": contract.litellm.model_dump(mode="json"),
        "deployments": projections,
    }
    return "bm1_" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_guard_manifest(
    snapshot: DiscoverySnapshot,
    contract: AdapterContract,
) -> GuardManifest:
    deployments = {
        row.hidden_alias: GuardDeploymentExpectation(
            runtime_id=row.runtime_id,
            generation_id=(generation := deployment_generation(row, contract)).generation_id,
            credential_field=row.credential_field,
            projection=generation.projection,
        )
        for row in snapshot.deployments
    }
    revision = backend_manifest_revision(contract, snapshot.deployments)
    if revision != snapshot.manifest_revision:
        raise ValueError("snapshot manifest revision differs from canonical guard payload")
    return GuardManifest(
        contract_id=contract.contract_id,
        backend_manifest=revision,
        guard_digest=contract.guard.digest,
        deployments=deployments,
    )
