from __future__ import annotations

from pathlib import Path

import pytest

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
    ClientKeyRecord,
    KeyPolicyRevision,
    PolicyBundleV1,
    ProviderAccount,
    QuotaDimension,
    RouteGroupRevision,
    RouteLeg,
)
from llmmaxxing.core.reasons import QuotaDimensionStatus, RouteStrategy, RouteTrigger
from llmmaxxing.core.state_machines import AccountState, KeyLifecycleState

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
        verifier_hex=("c" if key_id == KEY_A else "d") * 64,
        policy_id=policy_id,
        state=KeyLifecycleState.ENABLED,
        expires_at_s=2_000_000_000,
        credential_generation=1,
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
        policies=(
            _direct_policy(selector={"billing": "unlimited", "trust": "approved"}),
        )
    )

    compiled = compile_authoring(config, found, base_bundle())
    selected = next(policy for policy in compiled.policies if policy.policy_id == SELECTED_POLICY_ID)
    assert selected.allowed_account_ids == (NAN_ID, ARLIAI_ID)
    assert selected.allowed_triggers == (RouteTrigger.CAPACITY_SPILL, RouteTrigger.PRIMARY)
    assert compiled.generation == 8

    found.labels[ELECTRON_ID]["billing"] = "unlimited"
    found.labels[ELECTRON_ID]["trust"] = "approved"
    assert selected.allowed_account_ids == (NAN_ID, ARLIAI_ID)
    runtime_json = compiled.model_dump_json()
    assert "selector" not in runtime_json
    assert "labels" not in runtime_json

    explicit = AuthoringConfigV1(
        policies=(_direct_policy(account_ids=(ARLIAI_ID, NAN_ID)),)
    )
    explicit_bundle = compile_authoring(explicit, discovery(), base_bundle())
    exact = next(policy for policy in explicit_bundle.policies if policy.policy_id == SELECTED_POLICY_ID)
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
