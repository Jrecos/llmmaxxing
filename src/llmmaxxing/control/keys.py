"""Framework-neutral client-key issuance, rotation, and lifecycle commands."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from llmmaxxing.core.ids import PolicyRevisionId
from llmmaxxing.core.key_lifecycle import (
    DEFAULT_KEY_LIFETIME_S,
    DEFAULT_ROTATION_OVERLAP_S,
    MAX_KEY_LIFETIME_S,
    MAX_ROTATION_OVERLAP_S,
    PolicyReassignment,
    validate_key_record_delta,
)
from llmmaxxing.core.key_material import compute_client_key_verifier, format_client_key
from llmmaxxing.core.models import ClientCredentialVerifier, ClientKeyRecord, KeyPolicyRevision
from llmmaxxing.core.state_machines import (
    CredentialVerifierStatus,
    KeyLifecycleState,
    key_transition,
)

_RESPONSE_HEADERS = MappingProxyType(
    {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
)


@dataclass(frozen=True, slots=True, repr=False)
class RevealedClientKey:
    value: str = field(repr=False)
    response_headers: MappingProxyType[str, str] = _RESPONSE_HEADERS

    def __repr__(self) -> str:
        return "RevealedClientKey(value=<redacted>, cache=no-store, referrer=no-referrer)"


class IssuedOnce:
    """One-time in-memory handoff; only the verifier record survives reveal."""

    __slots__ = ("record", "_value")

    def __init__(self, record: ClientKeyRecord, value: str) -> None:
        self.record = record
        self._value: str | None = value

    @property
    def response_headers(self) -> MappingProxyType[str, str]:
        return _RESPONSE_HEADERS

    def reveal_once(self) -> RevealedClientKey:
        value = self._value
        if value is None:
            raise RuntimeError("client key was already revealed")
        self._value = None
        return RevealedClientKey(value=value)

    def __repr__(self) -> str:
        return f"IssuedOnce(key_id={self.record.key_id!r}, value=<redacted>)"


def _policy_id(policy: KeyPolicyRevision | PolicyRevisionId) -> PolicyRevisionId:
    return policy.policy_id if isinstance(policy, KeyPolicyRevision) else policy


def _validate_pepper(pepper: bytes, pepper_version: str) -> None:
    if len(pepper) < 32:
        raise ValueError("client-key pepper must contain at least 256 bits")
    if not pepper_version:
        raise ValueError("client-key pepper version is required")


def _new_material(
    *,
    key_id_bytes: bytes,
    pepper: bytes,
    pepper_version: str,
    generation: int,
    now_s: int,
    expires_at_s: int,
) -> tuple[ClientCredentialVerifier, str]:
    secret = secrets.token_bytes(32)
    verifier = ClientCredentialVerifier(
        generation=generation,
        verifier_hex=compute_client_key_verifier(pepper, key_id_bytes, secret).hex(),
        pepper_version=pepper_version,
        not_before_s=now_s,
        not_after_s=expires_at_s,
        status=CredentialVerifierStatus.ACTIVE,
    )
    return verifier, format_client_key(key_id_bytes, secret)


def issue_key(
    policy: KeyPolicyRevision | PolicyRevisionId,
    expiry: int | None = None,
    *,
    pepper: bytes,
    pepper_version: str,
    now_s: int,
    initial_state: Literal["draft", "enabled"] = "enabled",
) -> IssuedOnce:
    """Mint one 256-bit secret and return it through a one-time no-store handoff."""
    _validate_pepper(pepper, pepper_version)
    expires_at_s = now_s + DEFAULT_KEY_LIFETIME_S if expiry is None else expiry
    if expires_at_s <= now_s:
        raise ValueError("client key expiry must be in the future")
    if expires_at_s - now_s > MAX_KEY_LIFETIME_S:
        raise ValueError("client key expiry cannot exceed 365 days")
    key_id_bytes = secrets.token_bytes(16)
    verifier, value = _new_material(
        key_id_bytes=key_id_bytes,
        pepper=pepper,
        pepper_version=pepper_version,
        generation=1,
        now_s=now_s,
        expires_at_s=expires_at_s,
    )
    record = ClientKeyRecord(
        key_id=key_id_bytes.hex(),
        policy_id=_policy_id(policy),
        state=KeyLifecycleState(initial_state),
        issued_at_s=now_s,
        expires_at_s=expires_at_s,
        time_high_water_s=now_s,
        generation_high_water=1,
        credential_verifiers=(verifier,),
    )
    return IssuedOnce(record, value)


def _require_live_time(record: ClientKeyRecord, now_s: int) -> None:
    if now_s < record.time_high_water_s:
        raise ValueError("trusted clock high-water cannot move backward")
    if now_s >= record.expires_at_s:
        raise ValueError("client key is expired and terminal")


def _transition(
    record: ClientKeyRecord,
    *,
    now_s: int,
    state: KeyLifecycleState,
    credential_verifiers: tuple[ClientCredentialVerifier, ...] | None = None,
    policy_id: PolicyRevisionId | None = None,
    policy_reassignment: PolicyReassignment | None = None,
) -> ClientKeyRecord:
    candidate = ClientKeyRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "state": state,
            "time_high_water_s": now_s,
            "credential_verifiers": credential_verifiers or record.credential_verifiers,
            "policy_id": policy_id or record.policy_id,
        }
    )
    validate_key_record_delta(
        record,
        candidate,
        policy_reassignment=policy_reassignment,
    )
    return candidate


def activate_key(record: ClientKeyRecord, *, now_s: int) -> ClientKeyRecord:
    if record.state is KeyLifecycleState.REVOKED:
        raise ValueError("revoked client key is terminal")
    _require_live_time(record, now_s)
    return _transition(
        record,
        now_s=now_s,
        state=key_transition(record.state, "activate"),
    )


def suspend_key(record: ClientKeyRecord, *, now_s: int) -> ClientKeyRecord:
    if record.state is KeyLifecycleState.REVOKED:
        raise ValueError("revoked client key is terminal")
    _require_live_time(record, now_s)
    return _transition(
        record,
        now_s=now_s,
        state=key_transition(record.state, "suspend"),
    )


def resume_key(record: ClientKeyRecord, *, now_s: int) -> ClientKeyRecord:
    if record.state is KeyLifecycleState.REVOKED:
        raise ValueError("revoked client key is terminal")
    _require_live_time(record, now_s)
    return _transition(
        record,
        now_s=now_s,
        state=key_transition(record.state, "resume"),
    )


def expire_key(record: ClientKeyRecord, *, now_s: int) -> ClientKeyRecord:
    """Materialize absolute expiry without adding any grace period."""
    if now_s < record.time_high_water_s:
        raise ValueError("trusted clock high-water cannot move backward")
    if now_s < record.expires_at_s:
        raise ValueError("client key has not reached absolute expiry")
    if record.time_high_water_s >= record.expires_at_s:
        return record
    expired = tuple(
        item.model_copy(
            update={
                "status": CredentialVerifierStatus.EXPIRED,
                "not_after_s": min(item.not_after_s, record.expires_at_s),
            }
        )
        if item.status in (CredentialVerifierStatus.ACTIVE, CredentialVerifierStatus.RETIRING)
        else item
        for item in record.credential_verifiers
    )
    candidate = ClientKeyRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "time_high_water_s": now_s,
            "credential_verifiers": expired,
        }
    )
    validate_key_record_delta(record, candidate)
    return candidate


def revoke_key(record: ClientKeyRecord, *, now_s: int) -> ClientKeyRecord:
    if record.state is KeyLifecycleState.REVOKED:
        return record
    _require_live_time(record, now_s)
    retired = tuple(
        item.model_copy(
            update={
                "status": CredentialVerifierStatus.RETIRED,
                "not_after_s": min(item.not_after_s, now_s),
            }
        )
        for item in record.credential_verifiers
    )
    return _transition(
        record,
        now_s=now_s,
        state=key_transition(record.state, "revoke"),
        credential_verifiers=retired,
    )


def rotate_key(
    record: ClientKeyRecord,
    *,
    pepper: bytes,
    pepper_version: str,
    now_s: int,
    overlap_s: int = DEFAULT_ROTATION_OVERLAP_S,
) -> IssuedOnce:
    _validate_pepper(pepper, pepper_version)
    _require_live_time(record, now_s)
    if record.state is KeyLifecycleState.REVOKED:
        raise ValueError("revoked client key is terminal")
    if not 0 <= overlap_s <= MAX_ROTATION_OVERLAP_S:
        raise ValueError("credential overlap must be between zero and seven days")

    current = next(
        item
        for item in reversed(record.credential_verifiers)
        if item.status is CredentialVerifierStatus.ACTIVE
    )
    retiring = current.model_copy(
        update={
            "status": (
                CredentialVerifierStatus.RETIRING if overlap_s else CredentialVerifierStatus.RETIRED
            ),
            "not_after_s": min(record.expires_at_s, now_s + overlap_s),
        }
    )
    key_id_bytes = bytes.fromhex(record.key_id)
    generation = record.generation_high_water + 1
    verifier, value = _new_material(
        key_id_bytes=key_id_bytes,
        pepper=pepper,
        pepper_version=pepper_version,
        generation=generation,
        now_s=now_s,
        expires_at_s=record.expires_at_s,
    )
    candidate = ClientKeyRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "time_high_water_s": now_s,
            "generation_high_water": generation,
            "credential_verifiers": (retiring, verifier),
        }
    )
    validate_key_record_delta(record, candidate)
    return IssuedOnce(candidate, value)


def reassign_key_policy(
    record: ClientKeyRecord,
    reassignment: PolicyReassignment,
    *,
    now_s: int,
) -> ClientKeyRecord:
    _require_live_time(record, now_s)
    if reassignment.expected_policy_id != record.policy_id:
        raise ValueError("policy reassignment expected policy does not match")
    return _transition(
        record,
        now_s=now_s,
        state=record.state,
        policy_id=reassignment.target_policy_id,
        policy_reassignment=reassignment,
    )


__all__ = [
    "IssuedOnce",
    "PolicyReassignment",
    "RevealedClientKey",
    "activate_key",
    "expire_key",
    "issue_key",
    "reassign_key_policy",
    "resume_key",
    "revoke_key",
    "rotate_key",
    "suspend_key",
    "validate_key_record_delta",
]
