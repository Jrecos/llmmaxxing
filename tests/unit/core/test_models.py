"""Identity, closed-enum, referential-integrity and state-machine contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmmaxxing.core.ids import (
    AccountId,
    BundleHash,
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
    RequestAuthorizationCeiling,
    RequestProfile,
    RouteGroupRevision,
    RouteLeg,
)
from llmmaxxing.core.reasons import (
    V1_FEATURES,
    Modality,
    QuotaDimensionStatus,
    RouteStrategy,
    RouteTrigger,
    TerminalOutcome,
)
from llmmaxxing.core.state_machines import (
    ActivationStage,
    KeyLifecycleState,
    key_transition,
    next_activation_stage,
)


def quota(status: str = "known", value: int | None = 1) -> dict[str, object]:
    return {"status": status, "value": value}


def account(name: str = "nan", **overrides: object) -> ProviderAccount:
    fields: dict[str, object] = {
        "account_id": AccountId.new(),
        "display_name": name,
        "connection": "litellm:primary",
        "provider_token": f"tok-{name}",
        "binding_ref": f"bind-{name}",
        "max_in_flight": 5,
        "rpm_limit": quota(value=1800),
        "tpm_limit": quota(value=6_000_000),
        "window_seconds": 60,
        "monthly_quota_units": quota(value=1_000),
    }
    fields.update(overrides)
    return ProviderAccount.model_validate(fields)


def leg(
    account_id: AccountId,
    order: int = 10,
    triggers: tuple[RouteTrigger, ...] = (RouteTrigger.PRIMARY,),
    leg_id: RouteLegId | None = None,
) -> RouteLeg:
    return RouteLeg(
        leg_id=leg_id or RouteLegId.new(),
        order=order,
        triggers=triggers,
        account_id=account_id,
        generation_id=DeploymentGenerationId.from_digest("a" * 64),
    )


def route_group(
    legs: tuple[RouteLeg, ...],
    *,
    group_id: RouteGroupId | None = None,
    strategy: RouteStrategy = RouteStrategy.ORDERED_CAPACITY,
    name: str = "dsv4",
) -> RouteGroupRevision:
    return RouteGroupRevision(
        route_group_id=group_id or RouteGroupId.new(),
        name=name,
        strategy=strategy,
        legs=legs,
    )


def policy(
    route_group_ids: tuple[RouteGroupId, ...],
    account_ids: tuple[AccountId, ...],
    *,
    policy_id: PolicyRevisionId | None = None,
    name: str = "default",
    triggers: tuple[RouteTrigger, ...] = (RouteTrigger.PRIMARY, RouteTrigger.CAPACITY_SPILL),
) -> KeyPolicyRevision:
    return KeyPolicyRevision(
        policy_id=policy_id or PolicyRevisionId.new(),
        name=name,
        route_group_ids=route_group_ids,
        allowed_account_ids=account_ids,
        allowed_triggers=triggers,
        queue_tier=1,
        queue_weight=8,
        max_concurrency=4,
        max_waiters=16,
        deadline_ms=7_200_000,
    )


def key_record(
    policy_id: PolicyRevisionId,
    key_id: str = "0" * 32,
    state: str = "enabled",
) -> ClientKeyRecord:
    return ClientKeyRecord(
        key_id=key_id,
        verifier_hex="b" * 64,
        policy_id=policy_id,
        state=state,
        expires_at_s=1_800_000_000,
        credential_generation=1,
    )


def parts() -> dict[str, object]:
    """Coherent baseline: 2 accounts, 1 group with 2 ordered legs, 1 policy, 1 key."""
    primary = account(name="nan")
    spill = account(name="spill")
    group = route_group(
        (
            leg(
                primary.account_id,
                10,
                (RouteTrigger.PRIMARY, RouteTrigger.CAPACITY_SPILL),
            ),
            leg(spill.account_id, 20, (RouteTrigger.CAPACITY_SPILL,)),
        )
    )
    pol = policy((group.route_group_id,), (primary.account_id, spill.account_id))
    return {
        "accounts": (primary, spill),
        "route_groups": (group,),
        "policies": (pol,),
        "keys": (key_record(pol.policy_id),),
    }


def bundle(**overrides: object) -> PolicyBundleV1:
    fields: dict[str, object] = {
        "schema_version": 1,
        "generation": 7,
        "min_reader": "1.0",
        "required_features": ("ordered_capacity", "weighted_fair_queue"),
        "backend_manifest_hash": "c" * 64,
    }
    fields.update(parts())
    fields.update(overrides)
    return PolicyBundleV1.model_validate(fields)


def profile_fields() -> dict[str, object]:
    return {
        "route_group_id": RouteGroupId.new(),
        "model_alias": "dsv4",
        "modality": "chat",
        "stream": True,
        "input_tokens_max": 4096,
        "output_tokens_max": 8192,
        "reasoning_tokens_max": 0,
        "tools_count": 2,
        "response_schema_present": False,
        "history_turns": 12,
        "deadline_ms": 60_000,
    }


# --- identity -------------------------------------------------------------


def test_names_never_define_identity():
    a = account()
    b = a.model_copy(update={"display_name": "renamed"})
    assert a.account_id == b.account_id


def test_generated_ids_are_unique_prefixed_and_immutable():
    left, right = AccountId.new(), AccountId.new()
    assert left != right
    assert str(left).startswith("acc_")
    with pytest.raises(AttributeError):
        left.something = 1  # str subclass: no instance dict


def test_ids_are_typed_and_not_interchangeable():
    acc = AccountId.new()
    with pytest.raises(ValueError):
        RouteLegId(str(acc))  # wrong prefix
    with pytest.raises(ValueError):
        RouteLegId(str(acc)[4:])  # bare hex lacks the typed prefix
    with pytest.raises(ValidationError):
        account(account_id=RouteGroupId.new())


def test_hash_ids_require_full_digest():
    assert str(DeploymentGenerationId.from_digest("a" * 64)).startswith("dg1_")
    assert str(BundleHash.from_digest("b" * 64)).startswith("bh_")
    with pytest.raises(ValueError):
        DeploymentGenerationId.from_digest("deadbeef")


def test_models_are_frozen_and_reject_unknown_fields():
    a = account()
    with pytest.raises(ValidationError):
        a.display_name = "other"
    with pytest.raises(ValidationError):
        ProviderAccount.model_validate(
            {"account_id": AccountId.new(), "display_name": "x", "unlimited": True}
        )


# --- closed vocabularies ---------------------------------------------------


def test_terminal_outcomes_are_closed_and_queued_is_not_one():
    assert [outcome.value for outcome in TerminalOutcome] == [
        "authz_denied",
        "auth_state_unavailable",
        "unsupported_request",
        "backpressure_rejected",
        "route_unavailable",
        "deadline_exceeded",
        "upstream_failed",
        "client_cancelled",
        "response_stream_failed",
        "completed",
    ]
    for bad in ("queued", "token_exhausted"):
        with pytest.raises(ValueError):
            TerminalOutcome(bad)


def test_route_profile_bounds_and_rejects_unknowns():
    fields = profile_fields()
    profile = RequestProfile.model_validate(fields)
    assert profile.modality is Modality.CHAT
    for bad in (
        {"modality": "telepathy"},
        {"deadline_ms": 0},
        {"deadline_ms": 9_000_001},
        {"input_tokens_max": -1},
        {"model_alias": ""},
    ):
        with pytest.raises(ValidationError):
            RequestProfile.model_validate({**fields, **bad})


def test_weighted_and_adaptive_strategy_fail_closed():
    for bad in ("weighted", "adaptive", "least_loaded", "random"):
        with pytest.raises(ValueError):
            RouteStrategy(bad)


def test_unknown_trigger_is_rejected():
    with pytest.raises(ValueError):
        RouteTrigger("panic_fallback")


def test_required_features_schema_and_pydantic_share_closed_unique_vectors():
    schema = PolicyBundleV1.model_json_schema()
    feature_schema = schema["properties"]["required_features"]
    item_ref = feature_schema["items"]["$ref"].rsplit("/", 1)[-1]
    schema_values = set(schema["$defs"][item_ref]["enum"])
    assert feature_schema["uniqueItems"] is True
    assert schema_values == V1_FEATURES

    vectors = (
        ((), True),
        (("ordered_capacity",), True),
        (tuple(sorted(V1_FEATURES)), True),
        (("ordered_capacity", "ordered_capacity"), False),
        (("time_travel_routing",), False),
    )
    for features, accepted in vectors:
        schema_accepts = (
            len(set(features)) == len(features) and set(features) <= schema_values
        )
        assert schema_accepts is accepted
        if accepted:
            parsed = bundle(required_features=features).required_features
            assert tuple(feature.value for feature in parsed) == features
        else:
            with pytest.raises(ValidationError, match="required_features"):
                bundle(required_features=features)


# --- exact references -----------------------------------------------------


def test_policy_must_reference_existing_route_group():
    p = parts()
    p["policies"] = (policy((RouteGroupId.new(),), tuple(a.account_id for a in p["accounts"])),)
    with pytest.raises(ValidationError, match="route_group"):
        bundle(**p)


def test_policy_cannot_grant_unknown_account():
    p = parts()
    b = bundle(**p)
    p["policies"] = (
        policy(
            tuple(g.route_group_id for g in b.route_groups),
            tuple(a.account_id for a in b.accounts) + (AccountId.new(),),
        ),
    )
    p["keys"] = (key_record(b.policies[0].policy_id),)
    with pytest.raises(ValidationError, match="account"):
        bundle(**p)


def test_leg_must_reference_existing_account():
    p = parts()
    gid = p["route_groups"][0].route_group_id
    p["route_groups"] = (route_group((leg(AccountId.new()),), group_id=gid),)
    with pytest.raises(ValidationError, match="account"):
        bundle(**p)


def test_key_must_reference_existing_policy():
    p = parts()
    p["keys"] = (key_record(PolicyRevisionId.new()),)
    with pytest.raises(ValidationError, match="policy"):
        bundle(**p)


def test_duplicate_ids_are_rejected():
    p = parts()
    p["accounts"] = (p["accounts"][0], p["accounts"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        bundle(**p)

    p = parts()
    shared_leg = p["route_groups"][0].legs[0]
    extra = route_group((leg(p["accounts"][0].account_id, 10, leg_id=shared_leg.leg_id),))
    p["route_groups"] = (p["route_groups"][0], extra)
    with pytest.raises(ValidationError, match="duplicate"):
        bundle(**p)

    p = parts()
    p["keys"] = (p["keys"][0], p["keys"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        bundle(**p)


def test_route_group_rejects_duplicate_order_and_normalizes_legs():
    primary = account(name="primary")
    spill = account(name="spill")
    first = leg(primary.account_id, 10, (RouteTrigger.PRIMARY,))
    second = leg(spill.account_id, 20, (RouteTrigger.CAPACITY_SPILL,))

    normalized = route_group((second, first))
    assert normalized.legs == (first, second)

    with pytest.raises(ValidationError, match="duplicate RouteLeg.order"):
        route_group((first, second.model_copy(update={"order": first.order})))


def test_account_binding_triple_is_globally_unique():
    p = parts()
    first, second = p["accounts"]
    stolen = second.model_copy(
        update={
            "connection": first.connection,
            "provider_token": first.provider_token,
            "binding_ref": first.binding_ref,
        }
    )
    p["accounts"] = (first, stolen)
    with pytest.raises(ValidationError, match="binding"):
        bundle(**p)


def test_empty_binding_triples_do_not_collide():
    first = account(name="unbound-a", connection="", provider_token="", binding_ref="")
    second = account(name="unbound-b", connection="", provider_token="", binding_ref="")
    group = route_group(
        (
            leg(first.account_id, 10, (RouteTrigger.PRIMARY,)),
            leg(second.account_id, 20, (RouteTrigger.CAPACITY_SPILL,)),
        )
    )
    pol = policy((group.route_group_id,), (first.account_id, second.account_id))

    result = bundle(
        accounts=(first, second),
        route_groups=(group,),
        policies=(pol,),
        keys=(key_record(pol.policy_id),),
    )
    assert len(result.accounts) == 2


@pytest.mark.parametrize(
    ("connection", "provider_token", "binding_ref"),
    (
        ("connection", "", ""),
        ("", "token", ""),
        ("", "", "binding"),
        ("connection", "token", ""),
        ("connection", "", "binding"),
        ("", "token", "binding"),
    ),
)
def test_partial_account_binding_always_rejects(
    connection: str, provider_token: str, binding_ref: str
):
    with pytest.raises(ValidationError, match="binding"):
        account(
            connection=connection,
            provider_token=provider_token,
            binding_ref=binding_ref,
        )


def test_active_account_requires_complete_binding():
    with pytest.raises(ValidationError, match="ACTIVE.*complete binding"):
        account(connection="", provider_token="", binding_ref="", state="active")


def test_several_keys_may_share_one_policy():
    p = parts()
    k = p["keys"][0]
    p["keys"] = (k, key_record(k.policy_id, key_id="1" * 32))
    assert len(bundle(**p).keys) == 2


def test_exactly_one_primary_leg_per_route_group():
    p = parts()
    gid = p["route_groups"][0].route_group_id
    acc = p["accounts"][0].account_id
    p["route_groups"] = (
        route_group(
            (leg(acc, 10, (RouteTrigger.PRIMARY,)), leg(acc, 20, (RouteTrigger.PRIMARY,))),
            group_id=gid,
        ),
    )
    with pytest.raises(ValidationError, match="PRIMARY"):
        bundle(**p)
    p["route_groups"] = (
        route_group((leg(acc, 10, (RouteTrigger.CAPACITY_SPILL,)),), group_id=gid),
    )
    with pytest.raises(ValidationError, match="PRIMARY"):
        bundle(**p)


def test_unknown_required_feature_is_rejected():
    with pytest.raises(ValidationError, match="feature"):
        bundle(required_features=("time_travel_routing",))


def test_duplicate_policy_triggers_are_rejected():
    p = parts()
    with pytest.raises(ValidationError, match="duplicate"):
        policy(
            p["policies"][0].route_group_ids,
            p["policies"][0].allowed_account_ids,
            policy_id=p["policies"][0].policy_id,
            triggers=(RouteTrigger.PRIMARY, RouteTrigger.PRIMARY),
        )


def test_min_reader_above_supported_version_is_rejected():
    with pytest.raises(ValidationError, match="min_reader"):
        bundle(min_reader="9.9")
    with pytest.raises(ValidationError, match="min_reader"):
        bundle(min_reader="1")


def test_schema_version_and_generation_are_bounded():
    for bad in (
        {"schema_version": 2},
        {"schema_version": 0},
        {"generation": 0},
        {"generation": -3},
        {"backend_manifest_hash": "zz"},
        {"keys": ()},
    ):
        with pytest.raises(ValidationError):
            bundle(**bad)


def test_quota_dimensions_distinguish_known_unknown_and_attested_absent():
    known = QuotaDimension(status="known", value=60)
    unknown = QuotaDimension(status="unknown")
    absent = QuotaDimension(status="attested_absent")
    assert known.status is QuotaDimensionStatus.KNOWN
    assert known.value == 60
    assert unknown.status is QuotaDimensionStatus.UNKNOWN
    assert absent.status is QuotaDimensionStatus.ATTESTED_ABSENT

    for invalid in (
        {"status": "known"},
        {"status": "unknown", "value": 60},
        {"status": "attested_absent", "value": 60},
    ):
        with pytest.raises(ValidationError, match="value"):
            QuotaDimension.model_validate(invalid)

    draft = account(rpm_limit=quota("unknown", None))
    assert draft.state.value == "draft"
    assert draft.enforced_max_in_flight == 1
    with pytest.raises(ValidationError, match="attested"):
        account(rpm_limit=quota("unknown", None), state="active")

    unlimited = account(
        rpm_limit=quota("attested_absent", None),
        tpm_limit=quota("attested_absent", None),
        monthly_quota_units=quota("attested_absent", None),
    )
    assert unlimited.state.value == "active"
    assert unlimited.enforced_max_in_flight == unlimited.max_in_flight


# --- key lifecycle / activation state machines -----------------------------


def test_key_lifecycle_is_closed_and_revocation_is_terminal():
    assert key_transition(KeyLifecycleState.DRAFT, "activate") is KeyLifecycleState.ENABLED
    assert key_transition(KeyLifecycleState.ENABLED, "suspend") is KeyLifecycleState.SUSPENDED
    assert key_transition(KeyLifecycleState.SUSPENDED, "resume") is KeyLifecycleState.ENABLED
    for state in KeyLifecycleState:
        assert key_transition(state, "revoke") is KeyLifecycleState.REVOKED
    for event in ("activate", "resume", "suspend"):
        with pytest.raises(ValueError):
            key_transition(KeyLifecycleState.REVOKED, event)
    with pytest.raises(ValueError):
        key_transition(KeyLifecycleState.DRAFT, "resume")


def test_activation_stage_is_strictly_linear():
    stage = ActivationStage.PREPARING_BACKEND
    seen = [stage]
    for _ in range(5):
        stage = next_activation_stage(stage)
        seen.append(stage)
    assert seen == [
        ActivationStage.PREPARING_BACKEND,
        ActivationStage.BACKEND_READY,
        ActivationStage.STAGING_GATEWAY,
        ActivationStage.GATEWAY_STAGED,
        ActivationStage.COMMITTING,
        ActivationStage.APPLIED,
    ]
    with pytest.raises(ValueError):
        next_activation_stage(ActivationStage.APPLIED)


# --- admission ceiling ------------------------------------------------------


_CEILING_BUNDLE = bundle()


def ceiling(**overrides: object) -> RequestAuthorizationCeiling:
    b = _CEILING_BUNDLE
    fields: dict[str, object] = {
        "key_id": "0" * 32,
        "credential_generation": 1,
        "policy_id": b.policies[0].policy_id,
        "bundle_generation": b.generation,
        "bundle_hash": BundleHash.from_digest("d" * 64),
        "route_group_id": b.route_groups[0].route_group_id,
        "allowed_account_ids": tuple(a.account_id for a in b.accounts),
        "allowed_triggers": (RouteTrigger.PRIMARY, RouteTrigger.CAPACITY_SPILL),
        "leg_ids": tuple(lg.leg_id for lg in b.route_groups[0].legs),
        "max_tier": 1,
        "max_weight": 8,
        "max_deadline_ms": 7_200_000,
    }
    fields.update(overrides)
    return RequestAuthorizationCeiling.model_validate(fields)


def test_ceiling_intersection_only_contracts():
    full = ceiling()
    current = ceiling(
        allowed_account_ids=(full.allowed_account_ids[0],),
        leg_ids=(full.leg_ids[0],),
        allowed_triggers=(RouteTrigger.PRIMARY,),
        max_weight=4,
        max_tier=2,
        max_deadline_ms=60_000,
    )
    merged = full.intersection(current)
    assert merged.allowed_account_ids == (full.allowed_account_ids[0],)
    assert merged.leg_ids == (full.leg_ids[0],)
    assert merged.allowed_triggers == (RouteTrigger.PRIMARY,)
    assert merged.max_weight == 4  # minimum bound, never the expansion
    assert merged.max_tier == 1
    assert merged.max_deadline_ms == 60_000
    assert current.intersection(full) == merged


def test_ceiling_intersection_rejects_foreign_identity():
    for other in (
        ceiling(credential_generation=2),
        ceiling(route_group_id=RouteGroupId.new()),
        ceiling(key_id="1" * 32),
    ):
        with pytest.raises(ValueError):
            ceiling().intersection(other)


def test_ceiling_can_contract_to_empty_sets():
    full = ceiling()
    foreign = ceiling(
        allowed_account_ids=(AccountId.new(),),
        leg_ids=(RouteLegId.new(),),
        allowed_triggers=(RouteTrigger.QUOTA_FALLBACK,),
    )
    merged = full.intersection(foreign)
    assert merged.allowed_account_ids == ()
    assert merged.leg_ids == ()
    assert merged.allowed_triggers == ()
