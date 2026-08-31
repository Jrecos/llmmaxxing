from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from llmmaxxing.config.compile import DiscoverySnapshot, compile_authoring
from llmmaxxing.config.schema import (
    AuthoringConfigV1,
    AuthoringPolicy,
    PolicyMacro,
    emit_authoring_schema,
)
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
from llmmaxxing.core.reasons import QuotaDimensionStatus, RouteStrategy, RouteTrigger
from llmmaxxing.core.state_machines import (
    AccountState,
    CredentialVerifierStatus,
    KeyLifecycleState,
)

NAN_ID = AccountId("acc_11111111-1111-4111-8111-111111111111")
ARLIAI_ID = AccountId("acc_22222222-2222-4222-8222-222222222222")
ELECTRON_ID = AccountId("acc_33333333-3333-4333-8333-333333333333")
GROUP_ID = RouteGroupId("rg_11111111-1111-4111-8111-111111111111")
ELECTRON_GROUP_ID = RouteGroupId("rg_22222222-2222-4222-8222-222222222222")
BASE_POLICY_ID = PolicyRevisionId("pol_11111111-1111-4111-8111-111111111111")
SELECTED_POLICY_ID = PolicyRevisionId("pol_22222222-2222-4222-8222-222222222222")
CLONE_POLICY_ID = PolicyRevisionId("pol_33333333-3333-4333-8333-333333333333")
SHARED_POLICY_ID = PolicyRevisionId("pol_44444444-4444-4444-8444-444444444444")
KEY_A = "a" * 32
KEY_B = "b" * 32


def _quota(value: int = 100) -> QuotaDimension:
    return QuotaDimension(status=QuotaDimensionStatus.KNOWN, value=value)


def _account(account_id: AccountId, name: str, *, state: AccountState) -> ProviderAccount:
    return ProviderAccount(
        account_id=account_id,
        display_name=name,
        connection=f"litellm:{name}",
        provider_token=f"provider-{name}",
        binding_ref=f"binding-{name}",
        max_in_flight=5,
        rpm_limit=_quota(),
        tpm_limit=_quota(1_000),
        monthly_quota_units=_quota(10_000),
        state=state,
    )


def _leg(
    leg_id: str,
    account_id: AccountId,
    order: int,
    triggers: tuple[RouteTrigger, ...],
) -> RouteLeg:
    return RouteLeg(
        leg_id=RouteLegId(leg_id),
        order=order,
        triggers=triggers,
        account_id=account_id,
        generation_id=DeploymentGenerationId.from_digest(f"{order // 10}" * 64),
    )


def _policy(policy_id: PolicyRevisionId = BASE_POLICY_ID) -> KeyPolicyRevision:
    return KeyPolicyRevision(
        policy_id=policy_id,
        name="base",
        route_group_ids=(GROUP_ID,),
        allowed_account_ids=(NAN_ID, ARLIAI_ID),
        allowed_triggers=(RouteTrigger.PRIMARY, RouteTrigger.CAPACITY_SPILL),
        queue_tier=10,
        queue_weight=4,
        max_concurrency=4,
        max_waiters=16,
        deadline_ms=60_000,
    )


def _key(key_id: str, policy_id: PolicyRevisionId = BASE_POLICY_ID) -> ClientKeyRecord:
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
                verifier_hex=("c" if key_id == KEY_A else "d") * 64,
                pepper_version="p1",
                not_before_s=1_970_000_000,
                not_after_s=2_000_000_000,
                status=CredentialVerifierStatus.ACTIVE,
            ),
        ),
    )


def base_bundle() -> PolicyBundleV1:
    nan = _account(NAN_ID, "nan", state=AccountState.ACTIVE)
    arli = _account(ARLIAI_ID, "arliai", state=AccountState.ACTIVE)
    electron = _account(ELECTRON_ID, "electron", state=AccountState.DISABLED)
    pooled = RouteGroupRevision(
        route_group_id=GROUP_ID,
        name="pooled",
        strategy=RouteStrategy.ORDERED_CAPACITY,
        legs=(
            _leg(
                "leg_11111111-1111-4111-8111-111111111111",
                NAN_ID,
                10,
                (RouteTrigger.PRIMARY, RouteTrigger.CAPACITY_SPILL),
            ),
            _leg(
                "leg_22222222-2222-4222-8222-222222222222",
                ARLIAI_ID,
                20,
                (RouteTrigger.CAPACITY_SPILL,),
            ),
        ),
    )
    electron_only = RouteGroupRevision(
        route_group_id=ELECTRON_GROUP_ID,
        name="electron-only",
        strategy=RouteStrategy.ORDERED_CAPACITY,
        legs=(
            _leg(
                "leg_33333333-3333-4333-8333-333333333333",
                ELECTRON_ID,
                30,
                (RouteTrigger.PRIMARY,),
            ),
        ),
    )
    return PolicyBundleV1(
        schema_version=1,
        generation=7,
        min_reader="1.0",
        required_features=("ordered_capacity", "weighted_fair_queue"),
        keys=(_key(KEY_A), _key(KEY_B)),
        policies=(_policy(),),
        accounts=(nan, arli, electron),
        route_groups=(pooled, electron_only),
        backend_manifest_hash="e" * 64,
    )


def discovery() -> DiscoverySnapshot:
    return DiscoverySnapshot(
        labels={
            NAN_ID: {"billing": "unlimited", "trust": "approved"},
            ARLIAI_ID: {"billing": "unlimited", "trust": "approved"},
            ELECTRON_ID: {"billing": "metered", "trust": "blocked"},
        }
    )


def _direct_policy(
    *,
    policy_id: PolicyRevisionId = SELECTED_POLICY_ID,
    route_group_ids: tuple[RouteGroupId, ...] = (GROUP_ID,),
    account_ids: tuple[AccountId, ...] | None = None,
    selector: dict[str, str] | None = None,
) -> AuthoringPolicy:
    return AuthoringPolicy(
        policy_id=policy_id,
        name="selected",
        route_group_ids=route_group_ids,
        account_ids=account_ids,
        account_selector=selector,
        allowed_triggers=(RouteTrigger.CAPACITY_SPILL, RouteTrigger.PRIMARY),
        queue_tier=20,
        queue_weight=2,
        max_concurrency=2,
        max_waiters=8,
        deadline_ms=120_000,
    )


def test_labels_are_compile_time_only_and_exact_lists_are_sorted() -> None:
    found = discovery()
    config = AuthoringConfigV1(
        policies=(_direct_policy(selector={"billing": "unlimited", "trust": "approved"}),)
    )

    compiled = compile_authoring(config, found, base_bundle())
    selected = next(
        policy for policy in compiled.policies if policy.policy_id == SELECTED_POLICY_ID
    )
    assert selected.allowed_account_ids == (NAN_ID, ARLIAI_ID)
    assert selected.allowed_triggers == (RouteTrigger.CAPACITY_SPILL, RouteTrigger.PRIMARY)
    assert compiled.generation == 8

    found.labels[ELECTRON_ID]["billing"] = "unlimited"
    found.labels[ELECTRON_ID]["trust"] = "approved"
    assert selected.allowed_account_ids == (NAN_ID, ARLIAI_ID)
    runtime_json = compiled.model_dump_json()
    assert "selector" not in runtime_json
    assert "labels" not in runtime_json

    explicit = AuthoringConfigV1(policies=(_direct_policy(account_ids=(ARLIAI_ID, NAN_ID)),))
    explicit_bundle = compile_authoring(explicit, discovery(), base_bundle())
    exact = next(
        policy for policy in explicit_bundle.policies if policy.policy_id == SELECTED_POLICY_ID
    )
    assert exact.allowed_account_ids == (NAN_ID, ARLIAI_ID)


def test_shared_rebind_is_exact_clone_copies_no_keys_and_macro_expands() -> None:
    config = AuthoringConfigV1(
        macros={
            "interactive": PolicyMacro(
                queue_tier=3,
                queue_weight=8,
                max_concurrency=2,
                max_waiters=6,
                deadline_ms=90_000,
            )
        },
        policies=(
            AuthoringPolicy(
                policy_id=CLONE_POLICY_ID,
                clone_from_policy_id=BASE_POLICY_ID,
                macro="interactive",
            ),
            AuthoringPolicy(
                policy_id=SHARED_POLICY_ID,
                clone_from_policy_id=BASE_POLICY_ID,
                rebind_shared=True,
            ),
        ),
    )

    compiled = compile_authoring(config, discovery(), base_bundle())
    assert {key.key_id: key.policy_id for key in compiled.keys} == {
        KEY_A: SHARED_POLICY_ID,
        KEY_B: SHARED_POLICY_ID,
    }
    assert not any(key.policy_id == CLONE_POLICY_ID for key in compiled.keys)
    clone = next(policy for policy in compiled.policies if policy.policy_id == CLONE_POLICY_ID)
    assert (clone.queue_tier, clone.queue_weight, clone.deadline_ms) == (3, 8, 90_000)
    assert clone.allowed_account_ids == (NAN_ID, ARLIAI_ID)


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        (
            _direct_policy(account_ids=(ELECTRON_ID,)),
            "non-active account",
        ),
        (
            _direct_policy(account_ids=(NAN_ID,), route_group_ids=(ELECTRON_GROUP_ID,)),
            "no usable route",
        ),
        (
            _direct_policy(selector={"billing": "missing"}),
            "matched no accounts",
        ),
    ),
)
def test_compiler_rejects_hard_account_state_and_empty_routes(
    policy: AuthoringPolicy, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_authoring(AuthoringConfigV1(policies=(policy,)), discovery(), base_bundle())


def test_authoring_schema_is_deterministic_and_contains_no_secret_fields() -> None:
    generated = emit_authoring_schema()
    checked_in = Path(__file__).resolve().parents[3] / "schemas" / "authoring-v1.json"
    assert checked_in.read_bytes() == generated, (
        "regenerate with: uv run python -m llmmaxxing.config.schema > schemas/authoring-v1.json"
    )
    for forbidden in (
        b"verifier_hex",
        b"private_key",
        b"connection",
        b"provider_token",
        b"binding_ref",
    ):
        assert forbidden not in generated


def test_authoring_schema_and_pydantic_share_acceptance_vectors() -> None:
    schema_validator = Draft202012Validator(AuthoringConfigV1.model_json_schema())
    exact = _direct_policy(account_ids=(NAN_ID, ARLIAI_ID)).model_dump(
        mode="json", exclude_none=True
    )
    selector = _direct_policy(selector={"billing": "unlimited", "trust": "approved"}).model_dump(
        mode="json", exclude_none=True
    )
    clone_rebind = AuthoringPolicy(
        policy_id=SHARED_POLICY_ID,
        clone_from_policy_id=BASE_POLICY_ID,
        rebind_shared=True,
    ).model_dump(mode="json", exclude_none=True)
    tier_zero = {**exact, "queue_tier": 0}

    for policy in (exact, selector, clone_rebind, tier_zero):
        document = {"schema_version": 1, "policies": [policy]}
        AuthoringConfigV1.model_validate_json(json.dumps(document))
        schema_validator.validate(document)

    # Default model_dump() retains nullable authoring fields. JSON Schema must
    # treat those nulls as absent, matching the Pydantic authoring contract.
    for policy in (
        _direct_policy(account_ids=(NAN_ID, ARLIAI_ID)).model_dump(mode="json"),
        _direct_policy(selector={"billing": "unlimited"}).model_dump(mode="json"),
        AuthoringPolicy(
            policy_id=SHARED_POLICY_ID,
            clone_from_policy_id=BASE_POLICY_ID,
        ).model_dump(mode="json"),
    ):
        document = {"schema_version": 1, "policies": [policy]}
        AuthoringConfigV1.model_validate_json(json.dumps(document))
        schema_validator.validate(document)

    invalid = (
        {**selector, "account_selector": {}},
        {**selector, "account_selector": {f"label-{index}": "x" for index in range(17)}},
        {**selector, "account_selector": {"BadLabel": "x"}},
        {**selector, "account_selector": {"billing": ""}},
        {**exact, "account_selector": {"billing": "unlimited"}},
        {**exact, "account_ids": [str(NAN_ID), str(NAN_ID)]},
        {**clone_rebind, "clone_from_policy_id": None},
        {**exact, "queue_tier": "20"},
    )
    for policy in invalid:
        document = {"schema_version": 1, "policies": [policy]}
        with pytest.raises(PydanticValidationError):
            AuthoringConfigV1.model_validate_json(json.dumps(document))
        with pytest.raises(JsonSchemaValidationError):
            schema_validator.validate(document)


def test_authoring_schema_emits_selector_membership_and_condition_constraints() -> None:
    schema = AuthoringConfigV1.model_json_schema()
    policy = schema["$defs"]["AuthoringPolicy"]
    selector = policy["properties"]["account_selector"]["anyOf"][0]

    assert selector["minProperties"] == 1
    assert selector["maxProperties"] == 16
    assert selector["propertyNames"]["pattern"] == r"^[a-z][a-z0-9_.-]{0,63}$"
    assert selector["additionalProperties"] == {
        "maxLength": 120,
        "minLength": 1,
        "type": "string",
    }
    for field in ("route_group_ids", "account_ids", "allowed_triggers"):
        assert policy["properties"][field]["uniqueItems"] is True
    assert policy["allOf"]
