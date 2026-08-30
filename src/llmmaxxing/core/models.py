"""Pure Pydantic domain models for the immutable policy bundle.

No infrastructure imports.  Identity lives exclusively in typed ids; display
names are annotations.  Every model is frozen and forbids unknown fields, so
a bundle either matches this contract exactly or is rejected whole.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llmmaxxing.core.ids import (
    AccountId,
    BundleHash,
    DeploymentGenerationId,
    PolicyRevisionId,
    RouteGroupId,
    RouteLegId,
)
from llmmaxxing.core.reasons import (
    MAX_MIN_READER,
    V1_FEATURES,
    Modality,
    RouteStrategy,
    RouteTrigger,
)
from llmmaxxing.core.state_machines import AccountState, KeyLifecycleState

_HEX32 = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_HEX64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

_MAX_DEADLINE_MS = 9_000_000


class _Frozen(BaseModel):
    """Common contract: frozen value objects, exact fields only."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderAccount(_Frozen):
    """One real shared upstream quota boundary (not a vendor brand)."""

    account_id: AccountId
    display_name: str = Field(min_length=1, max_length=120)
    # Empty until onboarding binds the account; unbound accounts stay DRAFT.
    connection: str = ""
    provider_token: str = ""
    binding_ref: str = ""
    max_in_flight: int = Field(ge=1, le=128)
    # None means *not yet measured*: unknown never means unlimited.
    rpm_limit: int | None = Field(default=None, gt=0)
    tpm_limit: int | None = Field(default=None, gt=0)
    window_seconds: int = Field(default=60, gt=0, le=3600)
    monthly_quota_units: int | None = Field(default=None, gt=0)
    state: AccountState | None = None

    @property
    def fully_attested(self) -> bool:
        return (
            self.rpm_limit is not None
            and self.tpm_limit is not None
            and self.monthly_quota_units is not None
        )

    @property
    def enforced_max_in_flight(self) -> int:
        """Conservative parallel limit; unmeasured accounts serve one at a time."""
        return self.max_in_flight if self.fully_attested else 1

    @model_validator(mode="after")
    def _derive_and_guard_state(self) -> Self:
        if not self.fully_attested:
            if self.state not in (None, AccountState.DRAFT):
                raise ValueError(
                    "an account with unattested capacity dimensions must stay DRAFT "
                    "until every dimension is measured or attested absent"
                )
            object.__setattr__(self, "state", AccountState.DRAFT)
            return self
        if self.state is None:
            bound = bool(self.connection and self.provider_token and self.binding_ref)
            object.__setattr__(self, "state", AccountState.ACTIVE if bound else AccountState.DRAFT)
        return self


class RouteLeg(_Frozen):
    """One ordered candidate leg inside a Route Group revision."""

    leg_id: RouteLegId
    order: int = Field(ge=1)
    triggers: tuple[RouteTrigger, ...] = Field(min_length=1)
    account_id: AccountId
    generation_id: DeploymentGenerationId

    @field_validator("triggers")
    @classmethod
    def _triggers_unique(cls, v: tuple[RouteTrigger, ...]) -> tuple[RouteTrigger, ...]:
        if len(set(v)) != len(v):
            raise ValueError("duplicate leg trigger")
        return v


class RouteGroupRevision(_Frozen):
    """Immutable revision of one client-visible model and its ordered legs."""

    route_group_id: RouteGroupId
    name: str = Field(min_length=1, max_length=120)
    strategy: RouteStrategy
    legs: tuple[RouteLeg, ...] = Field(min_length=1)


class KeyPolicyRevision(_Frozen):
    """One complete immutable policy revision bound to client keys."""

    policy_id: PolicyRevisionId
    name: str = Field(min_length=1, max_length=120)
    route_group_ids: tuple[RouteGroupId, ...] = Field(min_length=1)
    allowed_account_ids: tuple[AccountId, ...] = Field(min_length=1)
    allowed_triggers: tuple[RouteTrigger, ...] = Field(min_length=1)
    queue_tier: int = Field(ge=1)
    queue_weight: int = Field(ge=1, le=64)
    max_concurrency: int = Field(ge=1)
    max_waiters: int = Field(ge=0)
    deadline_ms: int = Field(ge=1, le=_MAX_DEADLINE_MS)

    @field_validator("route_group_ids", "allowed_account_ids", "allowed_triggers")
    @classmethod
    def _no_duplicates(cls, v: tuple[object, ...]) -> tuple[object, ...]:
        if len(set(v)) != len(v):
            raise ValueError("duplicate policy membership value")
        return v


class ClientKeyRecord(_Frozen):
    """Gateway-verifiable client key record: only the HMAC verifier is stored."""

    key_id: _HEX32
    verifier_hex: _HEX64
    policy_id: PolicyRevisionId
    state: KeyLifecycleState
    expires_at_s: int = Field(gt=0)
    credential_generation: int = Field(ge=1, le=2)


class RequestProfile(_Frozen):
    """Immutable bounded profile captured for one admitted request."""

    route_group_id: RouteGroupId
    model_alias: str = Field(min_length=1, max_length=160)
    modality: Modality
    stream: bool
    input_tokens_max: int = Field(ge=0)
    output_tokens_max: int = Field(ge=0)
    reasoning_tokens_max: int = Field(ge=0)
    tools_count: int = Field(ge=0)
    response_schema_present: bool
    history_turns: int = Field(ge=0)
    deadline_ms: int = Field(ge=1, le=_MAX_DEADLINE_MS)


class RequestAuthorizationCeiling(_Frozen):
    """Admission-time ceiling; queue wakes intersect it with current authority.

    ``intersection`` is contractively safe: sets shrink, bounds take the
    minimum, and it refuses ceilings from different admitted identities.
    """

    key_id: _HEX32
    credential_generation: int = Field(ge=1, le=2)
    policy_id: PolicyRevisionId
    bundle_generation: int = Field(ge=1)
    bundle_hash: BundleHash
    route_group_id: RouteGroupId
    allowed_account_ids: tuple[AccountId, ...]
    leg_ids: tuple[RouteLegId, ...]
    allowed_triggers: tuple[RouteTrigger, ...]
    max_tier: int = Field(ge=1)
    max_weight: int = Field(ge=1, le=64)
    max_deadline_ms: int = Field(ge=1, le=_MAX_DEADLINE_MS)

    def intersection(self, other: Self) -> Self:
        if (
            self.key_id != other.key_id
            or self.credential_generation != other.credential_generation
            or self.route_group_id != other.route_group_id
        ):
            raise ValueError("cannot intersect ceilings of different admitted identities")

        def keep(ordered: tuple[object, ...], allowed: frozenset[object]) -> tuple[object, ...]:
            return tuple(item for item in ordered if item in allowed)

        return self.model_copy(
            update={
                "allowed_account_ids": keep(
                    self.allowed_account_ids, frozenset(other.allowed_account_ids)
                ),
                "leg_ids": keep(self.leg_ids, frozenset(other.leg_ids)),
                "allowed_triggers": keep(self.allowed_triggers, frozenset(other.allowed_triggers)),
                "max_tier": min(self.max_tier, other.max_tier),
                "max_weight": min(self.max_weight, other.max_weight),
                "max_deadline_ms": min(self.max_deadline_ms, other.max_deadline_ms),
            }
        )


class PolicyBundleV1(_Frozen):
    """The complete immutable enforcement snapshot applied by one generation."""

    schema_version: Literal[1]
    generation: int = Field(ge=1)
    min_reader: str
    required_features: tuple[str, ...]
    keys: tuple[ClientKeyRecord, ...] = Field(min_length=1)
    policies: tuple[KeyPolicyRevision, ...] = Field(min_length=1)
    accounts: tuple[ProviderAccount, ...] = Field(min_length=1)
    route_groups: tuple[RouteGroupRevision, ...] = Field(min_length=1)
    backend_manifest_hash: _HEX64

    @field_validator("min_reader")
    @classmethod
    def _min_reader_supported(cls, v: str) -> str:
        match = re.fullmatch(r"(\d+)\.(\d+)", v)
        if match is None:
            raise ValueError("min_reader must be '<major>.<minor>'")
        if (int(match.group(1)), int(match.group(2))) > MAX_MIN_READER:
            raise ValueError(
                f"min_reader {v!r} exceeds this build's supported reader floor "
                f"{MAX_MIN_READER[0]}.{MAX_MIN_READER[1]}"
            )
        return v

    @field_validator("required_features")
    @classmethod
    def _features_known(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(v) - V1_FEATURES)
        if unknown:
            raise ValueError(f"unknown required feature: {', '.join(unknown)}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate required feature")
        return v

    @model_validator(mode="after")
    def _exact_membership(self) -> Self:
        accounts = {a.account_id for a in self.accounts}
        groups = {g.route_group_id for g in self.route_groups}
        policies = {p.policy_id for p in self.policies}

        for label, seen in (
            ("account_id", [a.account_id for a in self.accounts]),
            ("route_group_id", [g.route_group_id for g in self.route_groups]),
            ("policy_id", [p.policy_id for p in self.policies]),
            ("key_id", [k.key_id for k in self.keys]),
            ("leg_id", [leg.leg_id for g in self.route_groups for leg in g.legs]),
        ):
            if len(set(seen)) != len(seen):
                raise ValueError(f"duplicate {label} in bundle")

        bindings = [(a.connection, a.provider_token, a.binding_ref) for a in self.accounts]
        if len(set(bindings)) != len(bindings):
            raise ValueError("duplicate account binding triple across live and tombstoned accounts")

        for group in self.route_groups:
            if missing := [leg.account_id for leg in group.legs if leg.account_id not in accounts]:
                raise ValueError(
                    f"route group {group.route_group_id} leg references unknown account: "
                    f"{', '.join(str(a) for a in missing)}"
                )
            primaries = [leg for leg in group.legs if RouteTrigger.PRIMARY in leg.triggers]
            if len(primaries) != 1:
                raise ValueError(
                    f"route group {group.route_group_id} must have exactly one PRIMARY leg, "
                    f"found {len(primaries)}"
                )

        for pol in self.policies:
            if missing_groups := [g for g in pol.route_group_ids if g not in groups]:
                raise ValueError(
                    f"policy {pol.policy_id} references unknown route_group: "
                    f"{', '.join(str(g) for g in missing_groups)}"
                )
            if missing_accounts := [a for a in pol.allowed_account_ids if a not in accounts]:
                raise ValueError(
                    f"policy {pol.policy_id} grants unknown account: "
                    f"{', '.join(str(a) for a in missing_accounts)}"
                )

        for key in self.keys:
            if key.policy_id not in policies:
                raise ValueError(f"key {key.key_id} references unknown policy")
        return self
