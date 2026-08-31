from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from llmmaxxing.config.signing import ActivationEnvelope, sign_activation
from llmmaxxing.core.canonical import bundle_hash, canonical_bundle_bytes
from llmmaxxing.core.ids import CommandId, GatewayBootId, InstallationId
from llmmaxxing.core.wire import (
    GENESIS_BASE_BUNDLE_HASH,
    BaseReference,
    BundleReference,
    ChannelSealV1,
    ChannelSigner,
    ChannelTrustSet,
    CommandBootMismatch,
    CommandInstallationMismatch,
    GatewayCommandV1,
    PrepareCommandPayload,
    SignedActivationV1,
    SignatureVerificationError,
    StatusCommandPayload,
    UnknownChannelEpoch,
    UnknownChannelKey,
    UnknownTrustEpoch,
    WireCommandKind,
    gateway_command_digest,
    seal_gateway_command,
    verify_gateway_command,
)
from support.gateway_stack import make_bundle


def _signed_policy(private_key: Ed25519PrivateKey) -> SignedActivationV1:
    target, _ = make_bundle()
    envelope = ActivationEnvelope(
        trust_epoch=7,
        signer_key_id="policy-primary",
        base_generation=0,
        base_bundle_hash=GENESIS_BASE_BUNDLE_HASH,
        target_generation=target.generation,
        target_content_hash=bundle_hash(canonical_bundle_bytes(target)),
        source_fingerprint="1" * 64,
        security_fence="2" * 64,
        key_set_fence="3" * 64,
        impact_hash="4" * 64,
    )
    return SignedActivationV1.from_signed(
        sign_activation(envelope, private_key, base_bundle=None, target_bundle=target)
    )


def _command(
    channel_key: Ed25519PrivateKey,
    policy_key: Ed25519PrivateKey,
    installation_id: InstallationId,
    boot_id: GatewayBootId,
) -> GatewayCommandV1:
    target, _ = make_bundle()
    target_bytes = canonical_bundle_bytes(target)
    payload = PrepareCommandPayload(
        base=None,
        target=BundleReference(
            generation=target.generation,
            bundle_hash=bundle_hash(target_bytes),
        ),
        bundle_b64=__import__("base64").b64encode(target_bytes).decode("ascii"),
    )
    unsigned = GatewayCommandV1(
        command_id=CommandId.new(),
        installation_id=installation_id,
        channel_epoch=4,
        security_epoch=9,
        dispatcher_fence=3,
        boot_id=boot_id,
        sequence=1,
        previous_digest="0" * 64,
        kind=WireCommandKind.PREPARE,
        issued_at_ms=1,
        payload=payload.model_dump(mode="json"),
        policy=_signed_policy(policy_key),
        channel_seal=ChannelSealV1(
            seal_id="control-primary",
            trust_epoch=5,
            signature="0" * 128,
        ),
    )
    return seal_gateway_command(unsigned, channel_key)


def test_genesis_is_absent_base_not_a_zero_bundle() -> None:
    policy_key = Ed25519PrivateKey.generate()
    policy = _signed_policy(policy_key)
    assert policy.payload_bytes
    with pytest.raises(ValidationError):
        BaseReference(generation=0, bundle_hash=GENESIS_BASE_BUNDLE_HASH)


def test_wire_models_are_frozen_and_forbid_extra_fields() -> None:
    reference = BundleReference(generation=1, bundle_hash=GENESIS_BASE_BUNDLE_HASH)
    with pytest.raises(ValidationError):
        BundleReference.model_validate({**reference.model_dump(), "other": True})
    with pytest.raises(ValidationError):
        reference.generation = 2  # type: ignore[misc]
    signer = ChannelSigner("gateway", 1, Ed25519PrivateKey.generate())
    with pytest.raises(FrozenInstanceError):
        signer.trust_epoch = 2  # type: ignore[misc]


def test_verifier_requires_exact_injected_channel_policy_and_runtime_binding() -> None:
    installation_id = InstallationId.new()
    boot_id = GatewayBootId.new()
    channel_key = Ed25519PrivateKey.generate()
    policy_key = Ed25519PrivateKey.generate()
    command = _command(channel_key, policy_key, installation_id, boot_id)
    trust = ChannelTrustSet(
        installation_id=installation_id,
        security_epoch=9,
        channel_epochs={4: {5: {"control-primary": channel_key.public_key()}}},
    )
    verified = verify_gateway_command(
        command,
        {7: {"policy-primary": policy_key.public_key()}},
        trust,
        expected_boot_id=boot_id,
        expected_fence_epoch=3,
    )
    assert verified.command_digest == gateway_command_digest(command)
    assert verified.policy is not None and verified.policy.base is None

    with pytest.raises(CommandInstallationMismatch):
        verify_gateway_command(
            command,
            {7: {"policy-primary": policy_key.public_key()}},
            ChannelTrustSet(
                installation_id=InstallationId.new(),
                security_epoch=9,
                channel_epochs={4: {5: {"control-primary": channel_key.public_key()}}},
            ),
            expected_boot_id=boot_id,
            expected_fence_epoch=3,
        )
    with pytest.raises(CommandBootMismatch):
        verify_gateway_command(
            command,
            {7: {"policy-primary": policy_key.public_key()}},
            trust,
            expected_boot_id=GatewayBootId.new(),
            expected_fence_epoch=3,
        )
    with pytest.raises(UnknownChannelEpoch):
        verify_gateway_command(
            command,
            {7: {"policy-primary": policy_key.public_key()}},
            ChannelTrustSet(
                installation_id=installation_id,
                security_epoch=9,
                channel_epochs={8: {5: {"control-primary": channel_key.public_key()}}},
            ),
            expected_boot_id=boot_id,
            expected_fence_epoch=3,
        )
    with pytest.raises(UnknownTrustEpoch):
        verify_gateway_command(
            command,
            {7: {"policy-primary": policy_key.public_key()}},
            ChannelTrustSet(
                installation_id=installation_id,
                security_epoch=9,
                channel_epochs={4: {6: {"control-primary": channel_key.public_key()}}},
            ),
            expected_boot_id=boot_id,
            expected_fence_epoch=3,
        )
    with pytest.raises(UnknownChannelKey):
        verify_gateway_command(
            command,
            {7: {"policy-primary": policy_key.public_key()}},
            ChannelTrustSet(
                installation_id=installation_id,
                security_epoch=9,
                channel_epochs={4: {5: {"other": channel_key.public_key()}}},
            ),
            expected_boot_id=boot_id,
            expected_fence_epoch=3,
        )

    tampered = command.model_copy(
        update={
            "channel_seal": command.channel_seal.model_copy(update={"signature": "f" * 128})
        }
    )
    with pytest.raises(SignatureVerificationError):
        verify_gateway_command(
            tampered,
            {7: {"policy-primary": policy_key.public_key()}},
            trust,
            expected_boot_id=boot_id,
            expected_fence_epoch=3,
        )


def test_status_cannot_smuggle_a_policy_and_mutations_cannot_omit_one() -> None:
    channel_key = Ed25519PrivateKey.generate()
    policy_key = Ed25519PrivateKey.generate()
    installation_id = InstallationId.new()
    boot_id = GatewayBootId.new()
    command = _command(channel_key, policy_key, installation_id, boot_id)
    with pytest.raises(ValidationError):
        GatewayCommandV1.model_validate(
            {
                **command.model_dump(mode="json"),
                "kind": WireCommandKind.STATUS,
                "payload": StatusCommandPayload().model_dump(mode="json"),
            }
        )
    with pytest.raises(ValidationError):
        GatewayCommandV1.model_validate(
            {**command.model_dump(mode="json"), "policy": None}
        )
