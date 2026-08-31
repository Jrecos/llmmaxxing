from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from llmmaxxing.config.plan import StalePreview, plan_change
from llmmaxxing.config.signing import (
    ActivationEnvelope,
    SignatureVerificationError,
    SignedActivation,
    UnknownSigningKey,
    UnknownTrustEpoch,
    sign_activation,
    verify_activation,
)
from llmmaxxing.core.canonical import bundle_hash, canonical_bundle_bytes
from llmmaxxing.core.ids import (
    AccountId,
    DeploymentGenerationId,
    PolicyRevisionId,
    RouteGroupId,
    RouteLegId,
)
from llmmaxxing.core.models import (
    ClientCredentialVerifier,
    ClientKeyRecord,
    KeyPolicyRevision,
    PolicyBundleV1,
    ProviderAccount,
    QuotaDimension,
    RouteGroupRevision,
    RouteLeg,
)
from llmmaxxing.core.reasons import RouteStrategy, RouteTrigger
from llmmaxxing.core.state_machines import (
    AccountState,
    CredentialVerifierStatus,
    KeyLifecycleState,
)

ACCOUNT_ID = AccountId("acc_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
GROUP_ID = RouteGroupId("rg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
LEG_ID = RouteLegId("leg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OLD_POLICY_ID = PolicyRevisionId("pol_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NEW_POLICY_ID = PolicyRevisionId("pol_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
KEY_A = "1" * 32
KEY_B = "2" * 32
NEW_KEY = "3" * 32


def _account() -> ProviderAccount:
    quota = QuotaDimension(status="known", value=100)
    return ProviderAccount(
        account_id=ACCOUNT_ID,
        display_name="account",
        connection="secret-connection-marker",
        provider_token="secret-provider-marker",
        binding_ref="secret-binding-marker",
        credential_fingerprint="hcf1_" + "a" * 64,
        credential_epoch=1,
        parallel_limit=QuotaDimension(status="known", value=2),
        local_parallel_ceiling=128,
        rpm_limit=quota,
        rpm_window_seconds=60,
        tpm_limit=quota,
        tpm_window_seconds=60,
        monthly_quota_units=quota,
        quota_units_per_attempt=1,
        monthly_reset_at_ms=2_000_000_000_000,
        state=AccountState.ACTIVE,
    )


def _group() -> RouteGroupRevision:
    return RouteGroupRevision(
        route_group_id=GROUP_ID,
        name="model",
        strategy=RouteStrategy.ORDERED_CAPACITY,
        legs=(
            RouteLeg(
                leg_id=LEG_ID,
                order=1,
                triggers=(RouteTrigger.PRIMARY,),
                account_id=ACCOUNT_ID,
                generation_id=DeploymentGenerationId.from_digest("4" * 64),
            ),
        ),
    )


def _policy(policy_id: PolicyRevisionId, *, weight: int = 2) -> KeyPolicyRevision:
    return KeyPolicyRevision(
        policy_id=policy_id,
        name="policy",
        route_group_ids=(GROUP_ID,),
        allowed_account_ids=(ACCOUNT_ID,),
        allowed_triggers=(RouteTrigger.PRIMARY,),
        queue_tier=20,
        queue_weight=weight,
        max_concurrency=2,
        max_waiters=4,
        deadline_ms=60_000,
    )


def _key(key_id: str, policy_id: PolicyRevisionId, verifier: str) -> ClientKeyRecord:
    return ClientKeyRecord(
        key_id=key_id,
        policy_id=policy_id,
        state=KeyLifecycleState.ENABLED,
        issued_at_s=1_970_000_000,
        expires_at_s=2_000_000_000,
        time_high_water_s=1_970_000_000,
        generation_high_water=1,
        credential_verifiers=(
            ClientCredentialVerifier(
                generation=1,
                verifier_hex=verifier * 64,
                pepper_version="p1",
                not_before_s=1_970_000_000,
                not_after_s=2_000_000_000,
                status=CredentialVerifierStatus.ACTIVE,
            ),
        ),
    )


def base_bundle() -> PolicyBundleV1:
    return PolicyBundleV1(
        schema_version=1,
        generation=11,
        min_reader="1.0",
        required_features=("ordered_capacity",),
        keys=(_key(KEY_A, OLD_POLICY_ID, "5"), _key(KEY_B, OLD_POLICY_ID, "6")),
        policies=(_policy(OLD_POLICY_ID),),
        accounts=(_account(),),
        route_groups=(_group(),),
        backend_manifest_hash="7" * 64,
    )


def target_bundle(*, weight: int = 8) -> PolicyBundleV1:
    base = base_bundle()
    return PolicyBundleV1(
        schema_version=1,
        generation=12,
        min_reader=base.min_reader,
        required_features=base.required_features,
        keys=(_key(KEY_A, NEW_POLICY_ID, "5"), _key(KEY_B, OLD_POLICY_ID, "6")),
        policies=(_policy(OLD_POLICY_ID), _policy(NEW_POLICY_ID, weight=weight)),
        accounts=base.accounts,
        route_groups=base.route_groups,
        backend_manifest_hash=base.backend_manifest_hash,
    )


def test_preview_binds_applied_base_fences_and_exact_key_set() -> None:
    base = base_bundle()
    preview = plan_change(base, target_bundle())

    assert preview.base_generation == base.generation
    assert preview.base_bundle_hash == bundle_hash(canonical_bundle_bytes(base))
    assert preview.target_content_hash == bundle_hash(canonical_bundle_bytes(target_bundle()))
    assert len(preview.source_fingerprint) == 64
    assert len(preview.security_fence) == 64
    assert len(preview.key_set_fence) == 64
    assert tuple(binding.key_id for binding in preview.expected_key_bindings) == (KEY_A, KEY_B)

    changed_key_set = PolicyBundleV1.model_validate(
        {
            **base.model_dump(mode="python"),
            "keys": (*base.keys, _key(NEW_KEY, OLD_POLICY_ID, "8")),
        }
    )
    with pytest.raises(StalePreview, match="base bundle|key set"):
        preview.verify_against(changed_key_set, source_fingerprint=preview.source_fingerprint)

    changed_binding = PolicyBundleV1.model_validate(
        {
            **base.model_dump(mode="python"),
            "keys": (
                _key(KEY_A, NEW_POLICY_ID, "5"),
                _key(KEY_B, OLD_POLICY_ID, "6"),
            ),
            "policies": (*base.policies, _policy(NEW_POLICY_ID)),
        }
    )
    with pytest.raises(StalePreview):
        preview.verify_against(changed_binding, source_fingerprint=preview.source_fingerprint)

    preview.verify_against(base, source_fingerprint=preview.source_fingerprint)
    with pytest.raises(StalePreview, match="source fingerprint"):
        preview.verify_against(base, source_fingerprint="0" * 64)
    with pytest.raises(StalePreview, match="fresh source fingerprint"):
        preview.verify_against(base)


def test_semantic_diff_and_impact_hash_are_exact_and_content_sensitive() -> None:
    base = base_bundle()
    target = target_bundle()
    first = plan_change(base, target)
    second = plan_change(base, target)

    assert first == second
    assert first.diff.changed_key_ids == (KEY_A,)
    assert first.diff.changed_policy_ids == (NEW_POLICY_ID,)
    assert first.diff.changed_account_ids == ()
    assert first.diff.changed_route_group_ids == ()
    assert first.diff.changed_leg_ids == ()
    assert first.affected_key_ids == (KEY_A,)
    assert len(first.impact_hash) == 64

    changed = plan_change(base, target_bundle(weight=9))
    assert changed.diff.changed_policy_ids == (NEW_POLICY_ID,)
    assert changed.target_content_hash != first.target_content_hash
    assert changed.impact_hash != first.impact_hash


def test_affected_keys_use_each_policy_effective_authorized_leg_projection() -> None:
    spill_account_id = AccountId("acc_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    spill_leg_id = RouteLegId("leg_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    primary = _account()
    spill = ProviderAccount.model_validate(
        {
            **primary.model_dump(mode="python"),
            "account_id": spill_account_id,
            "display_name": "spill",
            "connection": "spill-connection",
            "provider_token": "spill-provider",
            "binding_ref": "spill-binding",
        }
    )

    def group(spill_digest: str) -> RouteGroupRevision:
        return RouteGroupRevision(
            route_group_id=GROUP_ID,
            name="pooled",
            strategy=RouteStrategy.ORDERED_CAPACITY,
            legs=(
                RouteLeg(
                    leg_id=LEG_ID,
                    order=10,
                    triggers=(RouteTrigger.PRIMARY,),
                    account_id=ACCOUNT_ID,
                    generation_id=DeploymentGenerationId.from_digest("4" * 64),
                ),
                RouteLeg(
                    leg_id=spill_leg_id,
                    order=20,
                    triggers=(RouteTrigger.CAPACITY_SPILL,),
                    account_id=spill_account_id,
                    generation_id=DeploymentGenerationId.from_digest(spill_digest * 64),
                ),
            ),
        )

    def trigger_policy(policy_id: PolicyRevisionId, trigger: RouteTrigger) -> KeyPolicyRevision:
        return KeyPolicyRevision(
            policy_id=policy_id,
            name=f"{trigger.value}-only",
            route_group_ids=(GROUP_ID,),
            allowed_account_ids=(ACCOUNT_ID, spill_account_id),
            allowed_triggers=(trigger,),
            queue_tier=20,
            queue_weight=2,
            max_concurrency=2,
            max_waiters=4,
            deadline_ms=60_000,
        )

    policies = (
        trigger_policy(OLD_POLICY_ID, RouteTrigger.PRIMARY),
        trigger_policy(NEW_POLICY_ID, RouteTrigger.CAPACITY_SPILL),
    )

    def bundle(generation: int, spill_digest: str) -> PolicyBundleV1:
        return PolicyBundleV1(
            schema_version=1,
            generation=generation,
            min_reader="1.0",
            required_features=("ordered_capacity",),
            keys=(
                _key(KEY_A, OLD_POLICY_ID, "5"),
                _key(KEY_B, NEW_POLICY_ID, "6"),
            ),
            policies=policies,
            accounts=(primary, spill),
            route_groups=(group(spill_digest),),
            backend_manifest_hash="7" * 64,
        )

    preview = plan_change(bundle(20, "8"), bundle(21, "9"))

    assert preview.diff.changed_leg_ids == (spill_leg_id,)
    assert preview.affected_key_ids == (KEY_B,)


def test_affected_projection_replaces_leg_triggers_with_policy_intersection() -> None:
    base = base_bundle()
    leg = base.route_groups[0].legs[0]
    expanded = leg.model_copy(
        update={"triggers": (RouteTrigger.PRIMARY, RouteTrigger.CAPACITY_SPILL)}
    )
    target = PolicyBundleV1.model_validate(
        {
            **base.model_dump(mode="python"),
            "generation": base.generation + 1,
            "route_groups": (base.route_groups[0].model_copy(update={"legs": (expanded,)}),),
        }
    )

    preview = plan_change(base, target)

    assert preview.diff.changed_leg_ids == (LEG_ID,)
    assert preview.affected_key_ids == ()


def test_plan_rejects_invalid_key_lifecycle_delta() -> None:
    base = base_bundle()
    key = base.keys[0]
    target = PolicyBundleV1.model_validate(
        {
            **base.model_dump(mode="python"),
            "generation": base.generation + 1,
            "keys": (
                key.model_copy(update={"expires_at_s": key.expires_at_s + 1}),
                base.keys[1],
            ),
        }
    )
    with pytest.raises(ValueError, match="expiry extension"):
        plan_change(base, target)

    removed = PolicyBundleV1.model_validate(
        {
            **base.model_dump(mode="python"),
            "generation": base.generation + 1,
            "keys": (base.keys[1],),
        }
    )
    with pytest.raises(ValueError, match="tombstones"):
        plan_change(base, removed)


def test_impact_plan_export_contains_no_secret_material() -> None:
    plan = plan_change(base_bundle(), target_bundle())
    exported = plan.model_dump_json()
    for forbidden in (
        "secret-connection-marker",
        "secret-provider-marker",
        "secret-binding-marker",
        "5" * 64,
        "6" * 64,
    ):
        assert forbidden not in exported
    assert "verifier_hex" not in exported
    assert "provider_token" not in exported


def test_activation_signatures_bind_explicit_epoch_and_key() -> None:
    base = base_bundle()
    target = target_bundle()
    plan = plan_change(base, target)
    envelope = ActivationEnvelope.from_plan(plan, trust_epoch=7, signer_key_id="control-primary")
    private_key = Ed25519PrivateKey.generate()

    signed = sign_activation(envelope, private_key, base_bundle=base, target_bundle=target)
    assert isinstance(signed, SignedActivation)
    assert signed.trust_epoch == 7
    assert signed.signer_key_id == "control-primary"
    assert json.loads(signed.payload)["trust_epoch"] == 7
    assert (
        verify_activation(
            signed.payload,
            signed.signature,
            {7: {"control-primary": private_key.public_key()}},
        )
        == envelope
    )
    assert (
        verify_activation(
            signed.payload,
            signed.signature,
            private_key.public_key(),
            expected_trust_epoch=7,
            expected_signer_key_id="control-primary",
        )
        == envelope
    )

    with pytest.raises(UnknownTrustEpoch):
        verify_activation(
            signed.payload,
            signed.signature,
            {8: {"control-primary": private_key.public_key()}},
        )
    with pytest.raises(UnknownSigningKey):
        verify_activation(
            signed.payload,
            signed.signature,
            {7: {"other": private_key.public_key()}},
        )


def test_activation_rejects_tampering_and_noncanonical_payloads() -> None:
    base = base_bundle()
    target = target_bundle()
    plan = plan_change(base, target)
    envelope = ActivationEnvelope.from_plan(plan, trust_epoch=4, signer_key_id="operator")
    private_key = Ed25519PrivateKey.generate()
    signed = sign_activation(envelope, private_key, base_bundle=base, target_bundle=target)
    trust = {4: {"operator": private_key.public_key()}}

    tampered = signed.payload.replace(b'"target_generation":12', b'"target_generation":13')
    with pytest.raises(SignatureVerificationError):
        verify_activation(tampered, signed.signature, trust)

    noncanonical = json.dumps(json.loads(signed.payload), indent=2).encode()
    with pytest.raises(SignatureVerificationError, match="canonical"):
        verify_activation(noncanonical, private_key.sign(noncanonical), trust)


def test_signing_revalidates_key_lifecycle_against_exact_bundles() -> None:
    base = base_bundle()
    target = target_bundle()
    envelope = ActivationEnvelope.from_plan(
        plan_change(base, target),
        trust_epoch=4,
        signer_key_id="operator",
    )
    invalid_target = target.model_copy(
        update={
            "keys": (
                target.keys[0].model_copy(update={"expires_at_s": target.keys[0].expires_at_s + 1}),
                target.keys[1],
            )
        }
    )
    with pytest.raises(ValueError, match="expiry extension"):
        sign_activation(
            envelope,
            Ed25519PrivateKey.generate(),
            base_bundle=base,
            target_bundle=invalid_target,
        )
