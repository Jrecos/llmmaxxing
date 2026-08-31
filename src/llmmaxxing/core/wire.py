"""Canonical authenticated Control/Gateway wire contracts.

The models in this module are the shared boundary: Control, Gateway and CLI may
import them without importing one another.  Channel trust and signing keys are
always injected; this module deliberately has no trust defaults.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.ids import (
    AccountId,
    AckId,
    AuthLeaseId,
    BundleHash,
    CommandId,
    GatewayBootId,
    InstallationId,
    RouteGroupId,
    RouteLegId,
)

_HEX64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_HEX128 = Annotated[str, Field(pattern=r"^[0-9a-f]{128}$")]
_KEY_ID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
_SUBJECT_ID = Annotated[str, Field(min_length=1, max_length=180)]
_MAX_B64_BUNDLE = ((16 * 1024 * 1024 + 2) // 3) * 4
_ZERO_DIGEST = "0" * 64
_COMMAND_DOMAIN = b"llmmaxxing.gateway-command.v1\x00"
_ACK_DOMAIN = b"llmmaxxing.gateway-ack.v1\x00"
_FENCE_DOMAIN = b"llmmaxxing.fence-receipt.v1\x00"

GENESIS_BASE_GENERATION = 0
GENESIS_BASE_BUNDLE_HASH = BundleHash("bh_" + _ZERO_DIGEST)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class SignatureVerificationError(ValueError):
    """A canonical signed object is malformed or has an invalid signature."""


class UnknownTrustEpoch(SignatureVerificationError):
    pass


class UnknownSigningKey(SignatureVerificationError):
    pass


class UnknownChannelEpoch(SignatureVerificationError):
    pass


class UnknownChannelKey(SignatureVerificationError):
    pass


class CommandChainGap(ValueError):
    pass


class DuplicateCommand(ValueError):
    pass


class CommandInstallationMismatch(ValueError):
    pass


class CommandBootMismatch(ValueError):
    pass


class StaleSecurityEpoch(ValueError):
    pass


class StaleFenceEpoch(ValueError):
    pass


class BaseReference(_Frozen):
    """A real previously applied bundle; genesis represents absence with ``None``."""

    generation: int = Field(ge=1)
    bundle_hash: BundleHash


class BundleReference(_Frozen):
    generation: int = Field(ge=1)
    bundle_hash: BundleHash


class ActivationEnvelope(_Frozen):
    """Secret-free policy approval bound to an exact base and target bundle.

    The wire keeps the original scalar fields for the frozen v1 signature.
    ``base`` is the unambiguous application view: ``None`` is genesis, never an
    empty or generation-zero bundle.
    """

    schema_version: Literal[1] = 1
    trust_epoch: int = Field(ge=1)
    signer_key_id: _KEY_ID
    base_generation: int = Field(ge=0)
    base_bundle_hash: BundleHash
    target_generation: int = Field(ge=1)
    target_content_hash: BundleHash
    source_fingerprint: _HEX64
    security_fence: _HEX64
    key_set_fence: _HEX64
    impact_hash: _HEX64

    @model_validator(mode="after")
    def _genesis_pair_is_exact(self) -> Self:
        if (self.base_generation == GENESIS_BASE_GENERATION) != (
            self.base_bundle_hash == GENESIS_BASE_BUNDLE_HASH
        ):
            raise ValueError("genesis requires the exact absent-base sentinel pair")
        return self

    @property
    def base(self) -> BaseReference | None:
        if self.base_generation == GENESIS_BASE_GENERATION:
            return None
        return BaseReference(
            generation=self.base_generation,
            bundle_hash=self.base_bundle_hash,
        )

    @property
    def target(self) -> BundleReference:
        return BundleReference(
            generation=self.target_generation,
            bundle_hash=self.target_content_hash,
        )


class _SignedActivationLike(Protocol):
    payload: bytes
    signature: bytes
    trust_epoch: int
    signer_key_id: str


class SignedActivationV1(_Frozen):
    """JSON-safe detached policy signature."""

    payload_b64: str = Field(min_length=1, max_length=16 * 1024)
    signature: _HEX128
    trust_epoch: int = Field(ge=1)
    signer_key_id: _KEY_ID

    @classmethod
    def from_signed(cls, signed: _SignedActivationLike) -> Self:
        return cls(
            payload_b64=base64.b64encode(signed.payload).decode("ascii"),
            signature=signed.signature.hex(),
            trust_epoch=signed.trust_epoch,
            signer_key_id=signed.signer_key_id,
        )

    @property
    def payload_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.payload_b64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise SignatureVerificationError("invalid policy payload encoding") from error

    @property
    def signature_bytes(self) -> bytes:
        return bytes.fromhex(self.signature)


class ChannelSealV1(_Frozen):
    seal_id: _KEY_ID
    trust_epoch: int = Field(ge=1)
    signature: _HEX128


class WireCommandKind(StrEnum):
    PREPARE = "prepare"
    COMMIT = "commit"
    DENY = "deny"
    CLEAR_DENY = "clear_deny"
    STATUS = "status"


_MUTATING_POLICY_KINDS = frozenset(
    {
        WireCommandKind.PREPARE,
        WireCommandKind.COMMIT,
        WireCommandKind.DENY,
        WireCommandKind.CLEAR_DENY,
    }
)


class GatewayCommandV1(_Frozen):
    schema_version: Literal[1] = 1
    command_id: CommandId
    installation_id: InstallationId
    channel_epoch: int = Field(ge=1)
    security_epoch: int = Field(ge=1)
    dispatcher_fence: int = Field(ge=1)
    boot_id: GatewayBootId
    sequence: int = Field(ge=1)
    previous_digest: _HEX64
    kind: WireCommandKind
    issued_at_ms: int = Field(ge=1)
    payload: dict[str, JsonValue]
    policy: SignedActivationV1 | None
    channel_seal: ChannelSealV1

    @model_validator(mode="after")
    def _policy_presence_is_closed(self) -> Self:
        if (self.kind in _MUTATING_POLICY_KINDS) != (self.policy is not None):
            raise ValueError("command kind has invalid policy signature presence")
        return self


class PrepareCommandPayload(_Frozen):
    base: BaseReference | None
    target: BundleReference
    bundle_b64: str = Field(min_length=1, max_length=_MAX_B64_BUNDLE)

    def bundle_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.bundle_b64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("bundle is not canonical base64") from error


class StatusCommandPayload(_Frozen):
    issue_auth_lease: bool = False


class DenySubjectType(StrEnum):
    ROUTE_GROUP = "route_group"
    ACCOUNT = "account"
    LEG = "leg"


class DenyReason(StrEnum):
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    COMPROMISE = "compromise"
    ROLLBACK_HOLD = "rollback_hold"


class DenyCommandPayload(_Frozen):
    subject_type: DenySubjectType
    subject_id: _SUBJECT_ID
    deny_epoch: int = Field(ge=1)
    reason: DenyReason
    deny_floor_generation: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _typed_subject_id(self) -> Self:
        validators = {
            DenySubjectType.ROUTE_GROUP: RouteGroupId,
            DenySubjectType.ACCOUNT: AccountId,
            DenySubjectType.LEG: RouteLegId,
        }
        validators[self.subject_type](self.subject_id)
        return self


class ClearDenyCommandPayload(_Frozen):
    subject_type: DenySubjectType
    subject_id: _SUBJECT_ID
    deny_epoch: int = Field(ge=1)

    @model_validator(mode="after")
    def _typed_subject_id(self) -> Self:
        DenyCommandPayload(
            subject_type=self.subject_type,
            subject_id=self.subject_id,
            deny_epoch=self.deny_epoch,
            reason=DenyReason.MAINTENANCE,
        )
        return self


class DenyOverlayV1(_Frozen):
    subject_type: DenySubjectType
    subject_id: _SUBJECT_ID
    deny_epoch: int = Field(ge=1)
    reason: DenyReason
    deny_floor_generation: int | None = Field(default=None, ge=1)
    heartbeat_at_ms: int = Field(ge=1)


class AuthLeaseV1(_Frozen):
    schema_version: Literal[1] = 1
    lease_id: AuthLeaseId
    installation_id: InstallationId
    security_epoch: int = Field(ge=1)
    boot_id: GatewayBootId
    bundle: BundleReference
    issued_at_ms: int = Field(ge=1)
    expires_at_ms: int = Field(ge=1)

    @model_validator(mode="after")
    def _bounded_lifetime(self) -> Self:
        lifetime = self.expires_at_ms - self.issued_at_ms
        if lifetime < 1 or lifetime > 15 * 60 * 1000:
            raise ValueError("authorization lease exceeds 15 minutes")
        return self


class GatewayLifecycle(StrEnum):
    NONE = "none"
    PREPARED = "prepared"
    STAGED = "staged"
    APPLIED = "applied"
    RECOVERY_REQUIRED = "recovery_required"
    FENCED_OLD = "fenced_old"


class GatewayReadiness(StrEnum):
    READY = "ready"
    UNREADY = "unready"


class ReadinessReason(StrEnum):
    SINGLETON = "singleton_unavailable"
    NO_ACTIVE = "no_active_bundle"
    RECOVERY = "recovery_required"
    FENCED = "fenced_old"
    DENY_STALE = "deny_feed_stale"
    AUTH_LEASE = "auth_lease_invalid"
    BACKEND = "backend_unready"
    CAPACITIES = "gateway_capacities_unready"


class TakeoverState(StrEnum):
    NONE = "none"
    FENCED_OLD = "fenced_old"
    RELEASED = "released"


class GatewayStatusV1(_Frozen):
    schema_version: Literal[1] = 1
    installation_id: InstallationId
    channel_epoch: int = Field(ge=1)
    security_epoch: int = Field(ge=1)
    boot_id: GatewayBootId
    singleton_held: bool
    lifecycle: GatewayLifecycle
    active: BundleReference | None
    staged: BundleReference | None
    previous: BundleReference | None
    base: BaseReference | None
    deny_overlay: tuple[DenyOverlayV1, ...]
    deny_floor_generation: int | None = Field(default=None, ge=1)
    deny_heartbeat_fresh: bool
    auth_lease: AuthLeaseV1 | None
    dispatcher_fence: int = Field(ge=1)
    takeover_state: TakeoverState
    readiness: GatewayReadiness
    unready_reasons: tuple[ReadinessReason, ...]


class GatewayAckStatus(StrEnum):
    PREPARED = "prepared"
    APPLIED = "applied"
    DENIED = "denied"
    DENY_CLEARED = "deny_cleared"
    STATUS = "status"


class GatewayAckV1(_Frozen):
    schema_version: Literal[1] = 1
    ack_id: AckId
    command_id: CommandId
    command_digest: _HEX64
    installation_id: InstallationId
    channel_epoch: int = Field(ge=1)
    security_epoch: int = Field(ge=1)
    dispatcher_fence: int = Field(ge=1)
    boot_id: GatewayBootId
    sequence: int = Field(ge=1)
    status: GatewayAckStatus
    acknowledged_at_ms: int = Field(ge=1)
    result: dict[str, JsonValue]
    channel_seal: ChannelSealV1


class FenceReceiptPayloadV1(_Frozen):
    schema_version: Literal[1] = 1
    old_installation_id: InstallationId
    target_installation_id: InstallationId
    channel_epoch: int = Field(ge=1)
    security_epoch: int = Field(ge=1)
    old_boot_id: GatewayBootId
    credential_digest: _HEX64
    network_digest: _HEX64
    fence_epoch: int = Field(ge=1)
    fenced_at_ms: int = Field(ge=1)


class FenceReceiptV1(_Frozen):
    payload: FenceReceiptPayloadV1
    channel_seal: ChannelSealV1


@dataclass(frozen=True, slots=True)
class ChannelSigner:
    seal_id: str
    trust_epoch: int
    private_key: Ed25519PrivateKey

    def __post_init__(self) -> None:
        if not self.seal_id or self.trust_epoch < 1:
            raise ValueError("channel signer requires an explicit key and trust epoch")


@dataclass(frozen=True, slots=True)
class ChannelTrustSet:
    """Task-12-injected installation/channel/trust/key matrix."""

    installation_id: InstallationId
    security_epoch: int
    channel_epochs: Mapping[int, Mapping[int, Mapping[str, Ed25519PublicKey]]]

    def __post_init__(self) -> None:
        if self.security_epoch < 1 or not self.channel_epochs:
            raise ValueError("channel trust must be explicitly injected")
        copied: dict[int, Mapping[int, Mapping[str, Ed25519PublicKey]]] = {}
        for channel_epoch, trust_epochs in self.channel_epochs.items():
            if channel_epoch < 1 or not trust_epochs:
                raise ValueError("channel trust contains an empty epoch")
            copied_trust: dict[int, Mapping[str, Ed25519PublicKey]] = {}
            for trust_epoch, keys in trust_epochs.items():
                if trust_epoch < 1 or not keys:
                    raise ValueError("channel trust contains an empty trust epoch")
                copied_trust[trust_epoch] = MappingProxyType(dict(keys))
            copied[channel_epoch] = MappingProxyType(copied_trust)
        object.__setattr__(self, "channel_epochs", MappingProxyType(copied))


def _blanked(value: BaseModel) -> dict[str, Any]:
    payload = value.model_dump(mode="json")
    seal = payload.get("channel_seal")
    if not isinstance(seal, dict):
        raise TypeError("signed object has no channel seal")
    seal["signature"] = "0" * 128
    return payload


def gateway_command_signing_bytes(command: GatewayCommandV1) -> bytes:
    return _COMMAND_DOMAIN + canonical_json_bytes(_blanked(command))


def seal_gateway_command(
    command: GatewayCommandV1,
    private_key: Ed25519PrivateKey,
) -> GatewayCommandV1:
    signature = private_key.sign(gateway_command_signing_bytes(command)).hex()
    payload = command.model_dump(mode="json")
    payload["channel_seal"]["signature"] = signature
    return GatewayCommandV1.model_validate(payload)


def gateway_command_digest(command: GatewayCommandV1) -> str:
    payload = canonical_json_bytes(command.model_dump(mode="json"))
    return hashlib.sha256(_COMMAND_DOMAIN + payload).hexdigest()


def _parse_activation(policy: SignedActivationV1) -> ActivationEnvelope:
    payload = policy.payload_bytes
    try:
        decoded = json.loads(payload)
        envelope = ActivationEnvelope.model_validate(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError) as error:
        raise SignatureVerificationError("invalid activation payload") from error
    if canonical_json_bytes(envelope.model_dump(mode="json")) != payload:
        raise SignatureVerificationError("activation payload is not canonical JSON")
    if policy.trust_epoch != envelope.trust_epoch:
        raise UnknownTrustEpoch("policy wrapper trust epoch differs from payload")
    if policy.signer_key_id != envelope.signer_key_id:
        raise UnknownSigningKey("policy wrapper signer differs from payload")
    return envelope


def verify_signed_activation(
    policy: SignedActivationV1,
    policy_keys: Mapping[int, Mapping[str, Ed25519PublicKey]],
) -> ActivationEnvelope:
    envelope = _parse_activation(policy)
    epoch = policy_keys.get(envelope.trust_epoch)
    if epoch is None:
        raise UnknownTrustEpoch(f"unknown activation trust epoch {envelope.trust_epoch}")
    key = epoch.get(envelope.signer_key_id)
    if key is None:
        raise UnknownSigningKey(
            f"unknown signing key {envelope.signer_key_id!r} in epoch {envelope.trust_epoch}"
        )
    try:
        key.verify(policy.signature_bytes, policy.payload_bytes)
    except (InvalidSignature, ValueError) as error:
        raise SignatureVerificationError("invalid activation signature") from error
    return envelope


def _channel_key(
    seal: ChannelSealV1,
    channel_epoch: int,
    trust: ChannelTrustSet,
) -> Ed25519PublicKey:
    channel = trust.channel_epochs.get(channel_epoch)
    if channel is None:
        raise UnknownChannelEpoch(f"unknown channel epoch {channel_epoch}")
    epoch = channel.get(seal.trust_epoch)
    if epoch is None:
        raise UnknownTrustEpoch(f"unknown channel trust epoch {seal.trust_epoch}")
    key = epoch.get(seal.seal_id)
    if key is None:
        raise UnknownChannelKey(f"unknown channel key {seal.seal_id!r} in epoch {seal.trust_epoch}")
    return key


@dataclass(frozen=True, slots=True)
class VerifiedGatewayCommand:
    command: GatewayCommandV1
    command_digest: str
    policy: ActivationEnvelope | None


def authenticate_gateway_command(
    command: GatewayCommandV1,
    policy_keys: Mapping[int, Mapping[str, Ed25519PublicKey]],
    channel_trust: ChannelTrustSet,
) -> VerifiedGatewayCommand:
    """Verify immutable command identity and signatures without live boot/fence checks."""

    if command.installation_id != channel_trust.installation_id:
        raise CommandInstallationMismatch("command names another installation")
    key = _channel_key(command.channel_seal, command.channel_epoch, channel_trust)
    try:
        key.verify(
            bytes.fromhex(command.channel_seal.signature),
            gateway_command_signing_bytes(command),
        )
    except (InvalidSignature, ValueError) as error:
        raise SignatureVerificationError("invalid channel signature") from error
    policy = (
        None if command.policy is None else verify_signed_activation(command.policy, policy_keys)
    )
    return VerifiedGatewayCommand(command, gateway_command_digest(command), policy)


def verify_gateway_command(
    command: GatewayCommandV1,
    policy_keys: Mapping[int, Mapping[str, Ed25519PublicKey]],
    channel_trust: ChannelTrustSet,
    *,
    expected_boot_id: GatewayBootId,
    expected_fence_epoch: int,
) -> VerifiedGatewayCommand:
    """Authenticate a new command and require current runtime binding."""

    verified = authenticate_gateway_command(command, policy_keys, channel_trust)
    if command.boot_id != expected_boot_id:
        raise CommandBootMismatch("command names another Gateway boot")
    if command.security_epoch != channel_trust.security_epoch:
        raise StaleSecurityEpoch("command security epoch is not current")
    if command.dispatcher_fence != expected_fence_epoch:
        raise StaleFenceEpoch("command dispatcher fence is not current")
    return verified


def _seal_model[T: BaseModel](
    value: T,
    signer: ChannelSigner,
    model: type[T],
) -> T:
    payload = value.model_dump(mode="json")
    payload["channel_seal"] = {
        "seal_id": signer.seal_id,
        "trust_epoch": signer.trust_epoch,
        "signature": "0" * 128,
    }
    unsigned = model.model_validate(payload)
    domain = _ACK_DOMAIN if isinstance(unsigned, GatewayAckV1) else _FENCE_DOMAIN
    signature = signer.private_key.sign(domain + canonical_json_bytes(_blanked(unsigned))).hex()
    payload["channel_seal"]["signature"] = signature
    return model.model_validate(payload)


def seal_gateway_ack(ack: GatewayAckV1, signer: ChannelSigner) -> GatewayAckV1:
    return _seal_model(ack, signer, GatewayAckV1)


def verify_gateway_ack(
    ack: GatewayAckV1,
    channel_trust: ChannelTrustSet,
    *,
    command: GatewayCommandV1,
) -> GatewayAckV1:
    if ack.installation_id != channel_trust.installation_id:
        raise CommandInstallationMismatch("acknowledgement names another installation")
    if ack.security_epoch != channel_trust.security_epoch:
        raise StaleSecurityEpoch("acknowledgement security epoch is not current")
    if (
        ack.command_id != command.command_id
        or ack.command_digest != gateway_command_digest(command)
        or ack.channel_epoch != command.channel_epoch
        or ack.boot_id != command.boot_id
        or ack.sequence != command.sequence
    ):
        raise SignatureVerificationError("acknowledgement does not bind the exact command")
    key = _channel_key(ack.channel_seal, ack.channel_epoch, channel_trust)
    try:
        key.verify(
            bytes.fromhex(ack.channel_seal.signature),
            _ACK_DOMAIN + canonical_json_bytes(_blanked(ack)),
        )
    except (InvalidSignature, ValueError) as error:
        raise SignatureVerificationError("invalid acknowledgement signature") from error
    return ack


def seal_fence_receipt(
    receipt: FenceReceiptV1,
    signer: ChannelSigner,
) -> FenceReceiptV1:
    return _seal_model(receipt, signer, FenceReceiptV1)


def verify_fence_receipt(
    receipt: FenceReceiptV1,
    channel_trust: ChannelTrustSet,
    *,
    target_installation_id: InstallationId,
    minimum_fence_epoch: int,
) -> FenceReceiptPayloadV1:
    payload = receipt.payload
    if payload.target_installation_id != target_installation_id:
        raise CommandInstallationMismatch("fence receipt targets another installation")
    if payload.fence_epoch <= minimum_fence_epoch:
        raise StaleFenceEpoch("fence receipt does not advance dispatcher fence")
    if payload.security_epoch != channel_trust.security_epoch:
        raise StaleSecurityEpoch("fence receipt security epoch is not current")
    key = _channel_key(receipt.channel_seal, payload.channel_epoch, channel_trust)
    try:
        key.verify(
            bytes.fromhex(receipt.channel_seal.signature),
            _FENCE_DOMAIN + canonical_json_bytes(_blanked(receipt)),
        )
    except (InvalidSignature, ValueError) as error:
        raise SignatureVerificationError("invalid fence receipt signature") from error
    return payload
