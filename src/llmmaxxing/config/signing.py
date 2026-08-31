"""Canonical Ed25519 activation envelopes with explicit epoch/key trust."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import ValidationError

from llmmaxxing.config.plan import ImpactPlan
from llmmaxxing.core.canonical import (
    bundle_hash,
    canonical_bundle_bytes,
    canonical_json_bytes,
)
from llmmaxxing.core.wire import (
    GENESIS_BASE_BUNDLE_HASH,
    ActivationEnvelope as WireActivationEnvelope,
    SignatureVerificationError,
    UnknownSigningKey,
    UnknownTrustEpoch,
)
from llmmaxxing.core.key_lifecycle import (
    exact_policy_reassignments,
    validate_key_record_set_delta,
)
from llmmaxxing.core.models import PolicyBundleV1

class ActivationEnvelope(WireActivationEnvelope):
    """Activation claims plus the Task-3 impact-plan constructor."""

    @classmethod
    def from_plan(
        cls,
        plan: ImpactPlan,
        *,
        trust_epoch: int,
        signer_key_id: str,
    ) -> Self:
        return cls(
            trust_epoch=trust_epoch,
            signer_key_id=signer_key_id,
            base_generation=plan.base_generation,
            base_bundle_hash=plan.base_bundle_hash,
            target_generation=plan.target_generation,
            target_content_hash=plan.target_content_hash,
            source_fingerprint=plan.source_fingerprint,
            security_fence=plan.security_fence,
            key_set_fence=plan.key_set_fence,
            impact_hash=plan.impact_hash,
        )


@dataclass(frozen=True, slots=True)
class SignedActivation:
    payload: bytes
    signature: bytes
    trust_epoch: int
    signer_key_id: str


def sign_activation(
    envelope: ActivationEnvelope,
    private_key: Ed25519PrivateKey,
    *,
    base_bundle: PolicyBundleV1 | None,
    target_bundle: PolicyBundleV1,
) -> SignedActivation:
    """Validate exact key lifecycle and return a canonical detached signature."""
    envelope = ActivationEnvelope.model_validate(envelope.model_dump(mode="python"))
    target_bundle = PolicyBundleV1.model_validate(target_bundle.model_dump(mode="python"))
    if base_bundle is None:
        if (
            envelope.base_generation != 0
            or envelope.base_bundle_hash != GENESIS_BASE_BUNDLE_HASH
        ):
            raise ValueError("genesis activation requires an explicit absent base")
    else:
        base_bundle = PolicyBundleV1.model_validate(base_bundle.model_dump(mode="python"))
        validate_key_record_set_delta(
            base_bundle.keys,
            target_bundle.keys,
            policy_reassignments=exact_policy_reassignments(
                base_bundle.keys,
                target_bundle.keys,
            ),
        )
        if (
            envelope.base_generation != base_bundle.generation
            or envelope.base_bundle_hash != bundle_hash(canonical_bundle_bytes(base_bundle))
        ):
            raise ValueError("activation envelope does not match exact base bundle")
    if (
        envelope.target_generation != target_bundle.generation
        or envelope.target_content_hash != bundle_hash(canonical_bundle_bytes(target_bundle))
    ):
        raise ValueError("activation envelope does not match exact target bundle")
    payload = canonical_json_bytes(envelope.model_dump(mode="json"))
    return SignedActivation(
        payload=payload,
        signature=private_key.sign(payload),
        trust_epoch=envelope.trust_epoch,
        signer_key_id=envelope.signer_key_id,
    )


def _parse_canonical(payload: bytes) -> ActivationEnvelope:
    try:
        decoded = json.loads(payload)
        envelope = ActivationEnvelope.model_validate(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError) as error:
        raise SignatureVerificationError("invalid activation payload") from error
    if canonical_json_bytes(envelope.model_dump(mode="json")) != payload:
        raise SignatureVerificationError("activation payload is not canonical JSON")
    return envelope


def verify_activation(
    payload: bytes,
    signature: bytes,
    public_key: Ed25519PublicKey | Mapping[int, Mapping[str, Ed25519PublicKey]],
    *,
    expected_trust_epoch: int | None = None,
    expected_signer_key_id: str | None = None,
) -> ActivationEnvelope:
    """Verify a canonical activation against an explicit epoch-scoped trust set.

    A raw public key is accepted only when both expected epoch and key ID are
    supplied. This prevents an unscoped key object from silently trusting an
    envelope from an unknown epoch.
    """
    envelope = _parse_canonical(payload)
    if isinstance(public_key, Mapping):
        epoch = public_key.get(envelope.trust_epoch)
        if epoch is None:
            raise UnknownTrustEpoch(f"unknown activation trust epoch {envelope.trust_epoch}")
        trusted_key = epoch.get(envelope.signer_key_id)
        if trusted_key is None:
            raise UnknownSigningKey(
                f"unknown signing key {envelope.signer_key_id!r} in epoch {envelope.trust_epoch}"
            )
    else:
        if expected_trust_epoch is None:
            raise UnknownTrustEpoch("a raw public key requires expected_trust_epoch")
        if envelope.trust_epoch != expected_trust_epoch:
            raise UnknownTrustEpoch(f"unknown activation trust epoch {envelope.trust_epoch}")
        if expected_signer_key_id is None:
            raise UnknownSigningKey("a raw public key requires expected_signer_key_id")
        if envelope.signer_key_id != expected_signer_key_id:
            raise UnknownSigningKey(f"unknown signing key {envelope.signer_key_id!r}")
        trusted_key = public_key

    try:
        trusted_key.verify(signature, payload)
    except (InvalidSignature, ValueError) as error:
        raise SignatureVerificationError("invalid activation signature") from error
    return envelope
