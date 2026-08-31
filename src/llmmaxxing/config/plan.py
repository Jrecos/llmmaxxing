"""Secret-free semantic impact planning with stale-preview fences."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmmaxxing.core.canonical import (
    bundle_hash,
    canonical_bundle_bytes,
    canonical_json_bytes,
    content_hash,
)
from llmmaxxing.core.ids import (
    AccountId,
    BundleHash,
    PolicyRevisionId,
    RouteGroupId,
    RouteLegId,
)
from llmmaxxing.core.key_lifecycle import (
    exact_policy_reassignments,
    validate_key_record_set_delta,
)
from llmmaxxing.core.models import PolicyBundleV1

_HEX64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_SOURCE_DOMAIN = b"llmmaxxing.source-fingerprint.v1\x00"
_SECURITY_DOMAIN = b"llmmaxxing.security-fence.v1\x00"
_KEY_SET_DOMAIN = b"llmmaxxing.key-set-fence.v1\x00"
_IMPACT_DOMAIN = b"llmmaxxing.impact.v1\x00"


class StalePreview(ValueError):
    """The applied state no longer matches the exact state reviewed in a plan."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class ExpectedKeyBinding(_Frozen):
    key_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    policy_id: PolicyRevisionId
    generation_high_water: int = Field(ge=1)


class SemanticDiff(_Frozen):
    """Exact stable identities whose semantic records were added, removed, or changed."""

    changed_key_ids: tuple[str, ...]
    changed_policy_ids: tuple[PolicyRevisionId, ...]
    changed_account_ids: tuple[AccountId, ...]
    changed_route_group_ids: tuple[RouteGroupId, ...]
    changed_leg_ids: tuple[RouteLegId, ...]
    backend_manifest_changed: bool
    min_reader_changed: bool
    required_features_changed: bool


class ImpactPlan(_Frozen):
    """Reviewable activation impact, bound to one exact applied base and source."""

    schema_version: Literal[1] = 1
    base_generation: int = Field(ge=1)
    base_bundle_hash: BundleHash
    base_source_fingerprint: _HEX64
    source_fingerprint: _HEX64
    security_fence: _HEX64
    key_set_fence: _HEX64
    expected_key_bindings: tuple[ExpectedKeyBinding, ...]
    target_generation: int = Field(ge=1)
    target_content_hash: BundleHash
    diff: SemanticDiff
    affected_key_ids: tuple[str, ...]
    impact_hash: _HEX64

    @model_validator(mode="after")
    def _ordered_exact_sets(self) -> Self:
        for name, values in (
            ("expected_key_bindings", self.expected_key_bindings),
            ("affected_key_ids", self.affected_key_ids),
            ("changed_key_ids", self.diff.changed_key_ids),
            ("changed_policy_ids", self.diff.changed_policy_ids),
            ("changed_account_ids", self.diff.changed_account_ids),
            ("changed_route_group_ids", self.diff.changed_route_group_ids),
            ("changed_leg_ids", self.diff.changed_leg_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name} in impact plan")
        return self

    def verify_against(
        self,
        current: PolicyBundleV1,
        *,
        source_fingerprint: str | None = None,
    ) -> None:
        """Raise :class:`StalePreview` unless every reviewed base fence still matches."""
        if source_fingerprint is None:
            raise StalePreview("fresh source fingerprint is required to verify an impact plan")
        current = PolicyBundleV1.model_validate(current.model_dump(mode="python"))
        if current.generation != self.base_generation:
            raise StalePreview("base bundle generation changed")
        if bundle_hash(canonical_bundle_bytes(current)) != self.base_bundle_hash:
            raise StalePreview("base bundle content changed")
        if _source_fingerprint(current) != self.base_source_fingerprint:
            raise StalePreview("base authoring source changed")
        if _security_fence(current) != self.security_fence:
            raise StalePreview("security fence changed")
        if _key_set_fence(current) != self.key_set_fence:
            raise StalePreview("key set fence changed")
        if _key_bindings(current) != self.expected_key_bindings:
            raise StalePreview("exact key bindings changed")
        if source_fingerprint != self.source_fingerprint:
            raise StalePreview("source fingerprint changed")


def _hash_json(value: object, domain: bytes) -> str:
    return content_hash(canonical_json_bytes(value), domain=domain)


def _source_fingerprint(bundle: PolicyBundleV1) -> str:
    """Hash the source-visible policy semantics without credentials or bindings."""
    policies = sorted(
        (policy.model_dump(mode="json") for policy in bundle.policies),
        key=lambda policy: policy["policy_id"],
    )
    return _hash_json(
        {
            "schema_version": bundle.schema_version,
            "min_reader": bundle.min_reader,
            "required_features": sorted(feature.value for feature in bundle.required_features),
            "policies": policies,
        },
        _SOURCE_DOMAIN,
    )


def _security_fence(bundle: PolicyBundleV1) -> str:
    """Digest security-sensitive state without exporting its source fields."""
    return _hash_json(
        {
            "keys": sorted(
                (key.model_dump(mode="json") for key in bundle.keys),
                key=lambda key: key["key_id"],
            ),
            "accounts": sorted(
                (account.model_dump(mode="json") for account in bundle.accounts),
                key=lambda account: account["account_id"],
            ),
        },
        _SECURITY_DOMAIN,
    )


def _key_bindings(bundle: PolicyBundleV1) -> tuple[ExpectedKeyBinding, ...]:
    return tuple(
        ExpectedKeyBinding(
            key_id=key.key_id,
            policy_id=key.policy_id,
            generation_high_water=key.generation_high_water,
        )
        for key in sorted(bundle.keys, key=lambda key: key.key_id)
    )


def _key_set_fence(bundle: PolicyBundleV1) -> str:
    return _hash_json(
        [binding.model_dump(mode="json") for binding in _key_bindings(bundle)],
        _KEY_SET_DOMAIN,
    )


def _changed_ids(
    before: tuple[Any, ...],
    after: tuple[Any, ...],
    id_field: str,
) -> tuple[Any, ...]:
    left = {getattr(item, id_field): item.model_dump(mode="json") for item in before}
    right = {getattr(item, id_field): item.model_dump(mode="json") for item in after}
    return tuple(
        sorted(
            (
                identity
                for identity in left.keys() | right.keys()
                if left.get(identity) != right.get(identity)
            ),
            key=str,
        )
    )


def _legs(bundle: PolicyBundleV1) -> tuple[Any, ...]:
    return tuple(leg for group in bundle.route_groups for leg in group.legs)


def _semantic_diff(base: PolicyBundleV1, target: PolicyBundleV1) -> SemanticDiff:
    return SemanticDiff(
        changed_key_ids=_changed_ids(base.keys, target.keys, "key_id"),
        changed_policy_ids=_changed_ids(base.policies, target.policies, "policy_id"),
        changed_account_ids=_changed_ids(base.accounts, target.accounts, "account_id"),
        changed_route_group_ids=_changed_ids(
            base.route_groups, target.route_groups, "route_group_id"
        ),
        changed_leg_ids=_changed_ids(_legs(base), _legs(target), "leg_id"),
        backend_manifest_changed=base.backend_manifest_hash != target.backend_manifest_hash,
        min_reader_changed=base.min_reader != target.min_reader,
        required_features_changed=frozenset(base.required_features)
        != frozenset(target.required_features),
    )


def _effective_key_projection(
    bundle: PolicyBundleV1,
    key_id: str,
) -> object:
    """Return only runtime semantics reachable through one key's policy."""
    keys = {key.key_id: key for key in bundle.keys}
    key = keys.get(key_id)
    if key is None:
        return None
    policies = {policy.policy_id: policy for policy in bundle.policies}
    groups = {group.route_group_id: group for group in bundle.route_groups}
    accounts = {account.account_id: account for account in bundle.accounts}
    policy = policies[key.policy_id]
    authorized = {leg.leg_id: leg for leg in policy.authorized_legs}
    effective_groups = []
    for group_id in policy.route_group_ids:
        group = groups[group_id]
        legs = []
        for leg in group.legs:
            granted = authorized.get(leg.leg_id)
            if granted is None:
                continue
            projected_leg = leg.model_dump(mode="json")
            projected_leg["triggers"] = [
                trigger.value for trigger in granted.allowed_triggers
            ]
            projected_leg["capabilities"] = granted.capabilities.model_dump(mode="json")
            legs.append(
                {
                    "leg": projected_leg,
                    "account": accounts[granted.account_id].model_dump(mode="json"),
                }
            )
        effective_groups.append(
            {
                "route_group_id": group.route_group_id,
                "name": group.name,
                "strategy": group.strategy,
                "legs": legs,
            }
        )
    return {
        "key": key.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "route_groups": effective_groups,
    }


def _affected_keys(
    base: PolicyBundleV1,
    target: PolicyBundleV1,
    diff: SemanticDiff,
) -> tuple[str, ...]:
    key_ids = {key.key_id for bundle in (base, target) for key in bundle.keys}
    if diff.backend_manifest_changed or diff.min_reader_changed or diff.required_features_changed:
        return tuple(sorted(key_ids))
    return tuple(
        sorted(
            key_id
            for key_id in key_ids
            if _effective_key_projection(base, key_id) != _effective_key_projection(target, key_id)
        )
    )


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def plan_change(
    base: PolicyBundleV1,
    target: PolicyBundleV1,
    *,
    source_fingerprint: str | None = None,
) -> ImpactPlan:
    """Compute a deterministic, secret-free plan from an applied base to a target."""
    base = PolicyBundleV1.model_validate(base.model_dump(mode="python"))
    target = PolicyBundleV1.model_validate(target.model_dump(mode="python"))
    if target.generation <= base.generation:
        raise ValueError("target generation must be strictly greater than applied base generation")
    validate_key_record_set_delta(
        base.keys,
        target.keys,
        policy_reassignments=exact_policy_reassignments(base.keys, target.keys),
    )

    diff = _semantic_diff(base, target)
    target_source = source_fingerprint or _source_fingerprint(target)
    fields: dict[str, object] = {
        "schema_version": 1,
        "base_generation": base.generation,
        "base_bundle_hash": bundle_hash(canonical_bundle_bytes(base)),
        "base_source_fingerprint": _source_fingerprint(base),
        "source_fingerprint": target_source,
        "security_fence": _security_fence(base),
        "key_set_fence": _key_set_fence(base),
        "expected_key_bindings": _key_bindings(base),
        "target_generation": target.generation,
        "target_content_hash": bundle_hash(canonical_bundle_bytes(target)),
        "diff": diff,
        "affected_key_ids": _affected_keys(base, target, diff),
    }
    fields["impact_hash"] = _hash_json(
        {key: _jsonable(value) for key, value in fields.items()},
        _IMPACT_DOMAIN,
    )
    return ImpactPlan.model_validate(fields)
