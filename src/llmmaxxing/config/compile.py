"""Compile authoring conveniences into an immutable exact-membership bundle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from llmmaxxing.config.schema import AuthoringConfigV1, AuthoringPolicy, PolicyMacro
from llmmaxxing.core.ids import AccountId, PolicyRevisionId
from llmmaxxing.core.models import AuthorizedLeg, KeyPolicyRevision, PolicyBundleV1
from llmmaxxing.core.state_machines import AccountState

_QUEUE_FIELDS = (
    "queue_tier",
    "queue_weight",
    "max_concurrency",
    "max_waiters",
    "deadline_ms",
)


@dataclass(slots=True)
class DiscoverySnapshot:
    """Compile-time account labels; mutating them cannot affect a compiled bundle."""

    labels: dict[AccountId, dict[str, str]]


def _pick[T](
    explicit: T | None,
    inherited: T | None,
    *,
    field: str,
    macro: PolicyMacro | None = None,
) -> T:
    if explicit is not None:
        return explicit
    if macro is not None and field in _QUEUE_FIELDS:
        return cast(T, getattr(macro, field))
    if inherited is not None:
        return inherited
    raise ValueError(f"direct policy requires {field}")


def _selected_accounts(
    spec: AuthoringPolicy,
    source: KeyPolicyRevision | None,
    labels: Mapping[AccountId, Mapping[str, str]],
) -> tuple[AccountId, ...]:
    if spec.account_ids is not None:
        selected = spec.account_ids
    elif spec.account_selector is not None:
        selected = tuple(
            account_id
            for account_id, account_labels in labels.items()
            if all(account_labels.get(key) == value for key, value in spec.account_selector.items())
        )
        if not selected:
            raise ValueError(f"account selector for policy {spec.policy_id} matched no accounts")
    elif source is not None:
        selected = tuple({leg.account_id for leg in source.authorized_legs})
    else:
        raise ValueError("direct policy requires account_ids or account_selector")
    return tuple(sorted(selected, key=str))


def _materialize_policy(
    spec: AuthoringPolicy,
    source: KeyPolicyRevision | None,
    macro: PolicyMacro | None,
    labels: Mapping[AccountId, Mapping[str, str]],
    base: PolicyBundleV1,
) -> KeyPolicyRevision:
    inherited: dict[str, Any] = source.model_dump(mode="python") if source is not None else {}
    route_group_ids = tuple(
        sorted(
            _pick(
                spec.route_group_ids,
                inherited.get("route_group_ids"),
                field="route_group_ids",
            ),
            key=str,
        )
    )
    authorization_changed = any(
        value is not None
        for value in (
            spec.route_group_ids,
            spec.account_ids,
            spec.account_selector,
            spec.allowed_triggers,
        )
    )
    if source is not None and not authorization_changed:
        authorized_legs = source.authorized_legs
    else:
        allowed_accounts = frozenset(_selected_accounts(spec, source, labels))
        inherited_triggers = (
            tuple(
                sorted(
                    {
                        trigger
                        for leg in source.authorized_legs
                        for trigger in leg.allowed_triggers
                    },
                    key=lambda trigger: trigger.value,
                )
            )
            if source is not None
            else None
        )
        allowed_triggers = frozenset(
            _pick(
                spec.allowed_triggers,
                inherited_triggers,
                field="allowed_triggers",
            )
        )
        groups = {group.route_group_id: group for group in base.route_groups}
        for account_id in allowed_accounts:
            account = next(
                (item for item in base.accounts if item.account_id == account_id),
                None,
            )
            if account is None:
                raise ValueError(
                    f"policy {spec.policy_id} grants unknown account {account_id}"
                )
            state = account.state
            if state is not AccountState.ACTIVE:
                raise ValueError(
                    f"policy {spec.policy_id} grants non-active account {account_id} "
                    f"in state {state.value if state is not None else 'unset'}"
                )
        projected: list[AuthorizedLeg] = []
        for group_id in route_group_ids:
            group = groups.get(group_id)
            if group is None:
                raise ValueError(f"policy {spec.policy_id} grants unknown route group {group_id}")
            for leg in group.legs:
                triggers = tuple(
                    trigger for trigger in leg.triggers if trigger in allowed_triggers
                )
                if leg.account_id in allowed_accounts and triggers:
                    projected.append(
                        AuthorizedLeg(
                            leg_id=leg.leg_id,
                            account_id=leg.account_id,
                            generation_id=leg.generation_id,
                            order=leg.order,
                            allowed_triggers=triggers,
                            capabilities=leg.capabilities,
                        )
                    )
        authorized_legs = tuple(projected)
        if not authorized_legs:
            raise ValueError(f"policy {spec.policy_id} has no usable route")

    values: dict[str, object] = {
        "policy_id": spec.policy_id,
        "name": _pick(spec.name, inherited.get("name"), field="name"),
        "route_group_ids": route_group_ids,
        "authorized_legs": authorized_legs,
    }
    for field in _QUEUE_FIELDS:
        values[field] = _pick(
            getattr(spec, field),
            inherited.get(field),
            field=field,
            macro=macro,
        )
    return KeyPolicyRevision.model_validate(values)


def _validate_runtime_routes(
    policy: KeyPolicyRevision,
    base: PolicyBundleV1,
) -> None:
    accounts = {account.account_id: account for account in base.accounts}
    groups = {group.route_group_id: group for group in base.route_groups}
    leg_groups = {
        leg.leg_id: group.route_group_id
        for group in base.route_groups
        for leg in group.legs
    }

    for authorized in policy.authorized_legs:
        account = accounts.get(authorized.account_id)
        if account is None:
            raise ValueError(
                f"policy {policy.policy_id} grants unknown account {authorized.account_id}"
            )
        state = account.state
        if state is not AccountState.ACTIVE:
            raise ValueError(
                f"policy {policy.policy_id} grants non-active account "
                f"{authorized.account_id} in state "
                f"{state.value if state is not None else 'unset'}"
            )

    covered = frozenset(leg_groups[leg.leg_id] for leg in policy.authorized_legs)
    for group_id in policy.route_group_ids:
        if group_id not in groups:
            raise ValueError(f"policy {policy.policy_id} grants unknown route group {group_id}")
        if group_id not in covered:
            raise ValueError(
                f"policy {policy.policy_id} has no usable route for route group {group_id}"
            )


def compile_authoring(
    config: AuthoringConfigV1,
    discovery: DiscoverySnapshot,
    base: PolicyBundleV1,
) -> PolicyBundleV1:
    """Add authoring revisions to ``base`` and materialize all runtime memberships.

    Selectors, macros and clone operations end here. The returned Gateway bundle
    contains only exact IDs and copied immutable runtime records.
    """
    config = AuthoringConfigV1.model_validate(config.model_dump(mode="python"))
    base = PolicyBundleV1.model_validate(base.model_dump(mode="python"))
    base_accounts = {account.account_id for account in base.accounts}
    unknown_discovery = sorted(set(discovery.labels) - base_accounts, key=str)
    if unknown_discovery:
        raise ValueError(
            "discovery contains account IDs absent from the applied bundle: "
            + ", ".join(map(str, unknown_discovery))
        )

    existing = {policy.policy_id: policy for policy in base.policies}
    collisions = sorted(
        {spec.policy_id for spec in config.policies} & existing.keys(),
        key=str,
    )
    if collisions:
        raise ValueError(
            "immutable policy revision ID already exists: " + ", ".join(map(str, collisions))
        )

    additions: list[KeyPolicyRevision] = []
    rebinds: dict[PolicyRevisionId, PolicyRevisionId] = {}
    for spec in config.policies:
        source = None
        if spec.clone_from_policy_id is not None:
            source = existing.get(spec.clone_from_policy_id)
            if source is None:
                raise ValueError(f"unknown clone source policy {spec.clone_from_policy_id}")
        macro = config.macros.get(spec.macro) if spec.macro is not None else None
        materialized = _materialize_policy(spec, source, macro, discovery.labels, base)
        _validate_runtime_routes(materialized, base)
        additions.append(materialized)
        if spec.rebind_shared:
            assert spec.clone_from_policy_id is not None
            if spec.clone_from_policy_id in rebinds:
                raise ValueError(f"multiple shared rebinds from policy {spec.clone_from_policy_id}")
            if not any(key.policy_id == spec.clone_from_policy_id for key in base.keys):
                raise ValueError(
                    f"shared rebind source policy {spec.clone_from_policy_id} has no keys"
                )
            rebinds[spec.clone_from_policy_id] = spec.policy_id

    keys = tuple(
        sorted(
            (
                key.model_copy(update={"policy_id": rebinds.get(key.policy_id, key.policy_id)})
                for key in base.keys
            ),
            key=lambda key: key.key_id,
        )
    )
    return PolicyBundleV1.model_validate(
        {
            **base.model_dump(mode="python"),
            "generation": base.generation + 1,
            "keys": keys,
            "policies": tuple(sorted((*base.policies, *additions), key=lambda p: str(p.policy_id))),
            "accounts": tuple(sorted(base.accounts, key=lambda account: str(account.account_id))),
            "route_groups": tuple(
                sorted(base.route_groups, key=lambda group: str(group.route_group_id))
            ),
        }
    )
