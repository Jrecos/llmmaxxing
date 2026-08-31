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
    Modality,
    QuotaDimensionStatus,
    RequiredFeature,
    RouteStrategy,
    RouteTrigger,
)
from llmmaxxing.core.state_machines import (
    AccountState,
    CredentialVerifierStatus,
    KeyLifecycleState,
)

_HEX32 = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_HEX64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_CREDENTIAL_FINGERPRINT = Annotated[str, Field(pattern=r"^hcf1_[0-9a-f]{64}$")]

_MAX_DEADLINE_MS = 9_000_000


class _Frozen(BaseModel):
    """Common contract: frozen value objects, exact fields only."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class QuotaDimension(_Frozen):
    """One independently bounded quota dimension and its attestation state."""

    status: QuotaDimensionStatus
    value: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _value_matches_status(self) -> Self:
        if self.status is QuotaDimensionStatus.KNOWN and self.value is None:
            raise ValueError("a known quota dimension requires a value")
        if self.status is not QuotaDimensionStatus.KNOWN and self.value is not None:
            raise ValueError("only a known quota dimension may carry a value")
        return self


class ProviderAccount(_Frozen):
    """One real shared upstream quota boundary (not a vendor brand)."""

    account_id: AccountId
    display_name: str = Field(min_length=1, max_length=120)
    # Empty until onboarding binds the account; unbound accounts stay DRAFT.
    connection: str = Field(default="", max_length=256)
    provider_token: str = Field(default="", max_length=256)
    binding_ref: str = Field(default="", max_length=256)
    credential_fingerprint: _CREDENTIAL_FINGERPRINT | None = None
    credential_epoch: int | None = Field(default=None, ge=1)
    parallel_limit: QuotaDimension
    rpm_limit: QuotaDimension
    rpm_window_seconds: int = Field(gt=0, le=3600)
    tpm_limit: QuotaDimension
    tpm_window_seconds: int = Field(gt=0, le=86_400)
    monthly_quota_units: QuotaDimension
    monthly_reset_day_utc: int | None = Field(default=None, ge=1, le=31)
    monthly_reset_hour_utc: int | None = Field(default=None, ge=0, le=23)
    state: AccountState | None = None

    @property
    def fully_attested(self) -> bool:
        return self.parallel_limit.status is QuotaDimensionStatus.KNOWN and all(
            dimension.status is not QuotaDimensionStatus.UNKNOWN
            for dimension in (
                self.rpm_limit,
                self.tpm_limit,
                self.monthly_quota_units,
            )
        )

    @property
    def enforced_max_in_flight(self) -> int:
        """Conservative parallel limit; unmeasured accounts serve one at a time."""
        if not self.fully_attested:
            return 1
        assert self.parallel_limit.value is not None
        return self.parallel_limit.value

    @model_validator(mode="after")
    def _derive_and_guard_state(self) -> Self:
        if self.parallel_limit.status is QuotaDimensionStatus.ATTESTED_ABSENT:
            raise ValueError("parallel capacity must be known or unknown, never unbounded")

        reset = (self.monthly_reset_day_utc, self.monthly_reset_hour_utc)
        if self.monthly_quota_units.status is QuotaDimensionStatus.KNOWN and not all(
            value is not None for value in reset
        ):
            raise ValueError("a known monthly quota requires explicit UTC monthly reset semantics")
        if self.monthly_quota_units.status is not QuotaDimensionStatus.KNOWN and any(
            value is not None for value in reset
        ):
            raise ValueError("monthly reset semantics apply only to a known monthly quota")

        binding = (self.connection, self.provider_token, self.binding_ref)
        if any(binding) and not all(binding):
            raise ValueError("account binding must be fully populated or fully empty")
        attestation = (self.credential_fingerprint, self.credential_epoch)
        if any(value is not None for value in attestation) and not all(
            value is not None for value in attestation
        ):
            raise ValueError("credential attestation must include fingerprint and epoch")
        bound = all(binding)
        attested_credential = all(value is not None for value in attestation)

        if not self.fully_attested:
            if self.state not in (None, AccountState.DRAFT):
                raise ValueError(
                    "an account with unattested capacity dimensions must stay DRAFT "
                    "until every dimension is measured or attested absent"
                )
            object.__setattr__(self, "state", AccountState.DRAFT)
            return self
        if self.state is AccountState.ACTIVE and not attested_credential:
            raise ValueError("an ACTIVE account requires a complete credential attestation")
        if self.state is AccountState.ACTIVE and not bound:
            raise ValueError("an ACTIVE account requires a complete binding")
        durable_state = self.state in (AccountState.DISABLED, AccountState.TOMBSTONED)
        if durable_state and not attested_credential:
            raise ValueError("a durable account state requires a complete credential attestation")
        if durable_state and not bound:
            raise ValueError("a durable account state requires a complete binding")
        if self.state is None:
            object.__setattr__(
                self,
                "state",
                AccountState.ACTIVE if bound and attested_credential else AccountState.DRAFT,
            )
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

    @model_validator(mode="after")
    def _orders_unique_and_normalized(self) -> Self:
        orders = [leg.order for leg in self.legs]
        if len(set(orders)) != len(orders):
            raise ValueError("duplicate RouteLeg.order within route group")
        object.__setattr__(
            self,
            "legs",
            tuple(sorted(self.legs, key=lambda leg: (leg.order, str(leg.leg_id)))),
        )
        return self


class KeyPolicyRevision(_Frozen):
    """One complete immutable policy revision bound to client keys."""

    policy_id: PolicyRevisionId
    name: str = Field(min_length=1, max_length=120)
    route_group_ids: tuple[RouteGroupId, ...] = Field(min_length=1)
    allowed_account_ids: tuple[AccountId, ...] = Field(min_length=1)
    allowed_triggers: tuple[RouteTrigger, ...] = Field(min_length=1)
    queue_tier: int = Field(ge=0)
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


class ClientCredentialVerifier(_Frozen):
    """One immutable verifier generation; plaintext key material never enters bundles."""

    generation: int = Field(ge=1)
    verifier_hex: _HEX64
    pepper_version: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
    not_before_s: int = Field(gt=0)
    not_after_s: int = Field(gt=0)
    status: CredentialVerifierStatus

    @model_validator(mode="after")
    def _valid_window(self) -> Self:
        if self.not_after_s < self.not_before_s:
            raise ValueError("credential not_after_s precedes not_before_s")
        return self


class ClientKeyRecord(_Frozen):
    """One logical client key with at most two verifier generations."""

    key_id: _HEX32
    policy_id: PolicyRevisionId
    state: KeyLifecycleState
    issued_at_s: int = Field(gt=0)
    expires_at_s: int = Field(gt=0)
    time_high_water_s: int = Field(gt=0)
    generation_high_water: int = Field(ge=1)
    credential_verifiers: tuple[ClientCredentialVerifier, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _valid_lifetime_and_generations(self) -> Self:
        if self.expires_at_s <= self.issued_at_s:
            raise ValueError("client key expiry must follow issuance")
        if self.expires_at_s - self.issued_at_s > 365 * 86_400:
            raise ValueError("client key lifetime exceeds 365 days")
        if self.time_high_water_s < self.issued_at_s:
            raise ValueError("trusted time high-water precedes issuance")

        generations = tuple(item.generation for item in self.credential_verifiers)
        if generations != tuple(sorted(set(generations))):
            raise ValueError("credential generations must be unique and increasing")
        if generations[-1] != self.generation_high_water:
            raise ValueError("generation high-water must equal the newest verifier generation")
        if any(
            item.not_before_s < self.issued_at_s or item.not_after_s > self.expires_at_s
            for item in self.credential_verifiers
        ):
            raise ValueError("credential window must stay within logical key lifetime")

        accepted = tuple(
            item
            for item in self.credential_verifiers
            if item.status in (CredentialVerifierStatus.ACTIVE, CredentialVerifierStatus.RETIRING)
        )
        active = tuple(
            item
            for item in self.credential_verifiers
            if item.status is CredentialVerifierStatus.ACTIVE
        )
        if len(accepted) > 2:
            raise ValueError("at most two credential generations may be accepted")
        if self.state is KeyLifecycleState.REVOKED or self.time_high_water_s >= self.expires_at_s:
            if accepted:
                raise ValueError("terminal key cannot retain accepted credential generations")
        elif len(active) != 1 or active[0].generation != self.generation_high_water:
            raise ValueError("nonterminal key requires one active newest credential generation")
        return self


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

    Lower numeric queue tiers are higher priority. Intersection therefore takes
    the maximum tier while every resource limit takes the minimum.
    """

    key_id: _HEX32
    credential_generation: int = Field(ge=1)
    policy_id: PolicyRevisionId
    bundle_generation: int = Field(ge=1)
    bundle_hash: BundleHash
    route_group_id: RouteGroupId
    allowed_account_ids: tuple[AccountId, ...]
    leg_ids: tuple[RouteLegId, ...]
    allowed_triggers: tuple[RouteTrigger, ...]
    queue_tier: int = Field(ge=0)
    queue_weight: int = Field(ge=1, le=64)
    max_concurrency: int = Field(ge=1)
    max_waiters: int = Field(ge=0)
    deadline_ms: int = Field(ge=1, le=_MAX_DEADLINE_MS)

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
                "queue_tier": max(self.queue_tier, other.queue_tier),
                "queue_weight": min(self.queue_weight, other.queue_weight),
                "max_concurrency": min(self.max_concurrency, other.max_concurrency),
                "max_waiters": min(self.max_waiters, other.max_waiters),
                "deadline_ms": min(self.deadline_ms, other.deadline_ms),
            }
        )


class PolicyBundleV1(_Frozen):
    """The complete immutable enforcement snapshot applied by one generation."""

    schema_version: Literal[1]
    generation: int = Field(ge=1)
    min_reader: str
    required_features: tuple[RequiredFeature, ...] = Field(json_schema_extra={"uniqueItems": True})
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
    def _features_unique(cls, v: tuple[RequiredFeature, ...]) -> tuple[RequiredFeature, ...]:
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

        bindings = [
            binding
            for a in self.accounts
            if any(binding := (a.connection, a.provider_token, a.binding_ref))
        ]
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
