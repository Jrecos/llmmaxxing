"""Pure client-key record lifecycle invariants shared by Control and signing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llmmaxxing.core.ids import PolicyRevisionId
from llmmaxxing.core.models import ClientCredentialVerifier, ClientKeyRecord
from llmmaxxing.core.state_machines import CredentialVerifierStatus, KeyLifecycleState

DEFAULT_KEY_LIFETIME_S = 90 * 86_400
MAX_KEY_LIFETIME_S = 365 * 86_400
DEFAULT_ROTATION_OVERLAP_S = 86_400
MAX_ROTATION_OVERLAP_S = 7 * 86_400

_TERMINAL_CREDENTIAL_STATES = frozenset(
    (CredentialVerifierStatus.RETIRED, CredentialVerifierStatus.EXPIRED)
)
_ALLOWED_CREDENTIAL_TRANSITIONS = {
    CredentialVerifierStatus.ACTIVE: frozenset(
        (
            CredentialVerifierStatus.ACTIVE,
            CredentialVerifierStatus.RETIRING,
            CredentialVerifierStatus.RETIRED,
            CredentialVerifierStatus.EXPIRED,
        )
    ),
    CredentialVerifierStatus.RETIRING: frozenset(
        (
            CredentialVerifierStatus.RETIRING,
            CredentialVerifierStatus.RETIRED,
            CredentialVerifierStatus.EXPIRED,
        )
    ),
    CredentialVerifierStatus.RETIRED: frozenset((CredentialVerifierStatus.RETIRED,)),
    CredentialVerifierStatus.EXPIRED: frozenset((CredentialVerifierStatus.EXPIRED,)),
}


@dataclass(frozen=True, slots=True)
class PolicyReassignment:
    expected_policy_id: PolicyRevisionId
    target_policy_id: PolicyRevisionId


def _validated(record: ClientKeyRecord) -> ClientKeyRecord:
    return ClientKeyRecord.model_validate(record.model_dump(mode="python"))


def _validate_existing_verifier(
    before: ClientCredentialVerifier,
    after: ClientCredentialVerifier,
) -> None:
    if (
        before.verifier_hex != after.verifier_hex
        or before.pepper_version != after.pepper_version
        or before.not_before_s != after.not_before_s
    ):
        raise ValueError("credential verifier generation is immutable")
    if after.not_after_s > before.not_after_s:
        raise ValueError("credential verifier window cannot be extended")
    if after.status not in _ALLOWED_CREDENTIAL_TRANSITIONS[before.status]:
        if before.status in _TERMINAL_CREDENTIAL_STATES:
            raise ValueError("terminal credential generation cannot resurrect")
        raise ValueError("illegal credential verifier status transition")


def validate_key_record_delta(
    before: ClientKeyRecord,
    after: ClientKeyRecord,
    *,
    policy_reassignment: PolicyReassignment | None = None,
) -> None:
    """Reject any expansive or ambiguous whole-record lifecycle transition."""
    before = _validated(before)
    after = _validated(after)
    if after.key_id != before.key_id:
        raise ValueError("client key identity cannot change")
    if after.issued_at_s != before.issued_at_s:
        raise ValueError("client key issuance time cannot change")
    if before.state is KeyLifecycleState.REVOKED or before.time_high_water_s >= before.expires_at_s:
        if after != before:
            raise ValueError("terminal key identity cannot resurrect or change")
        return
    if after.time_high_water_s < before.time_high_water_s:
        raise ValueError("trusted clock high-water cannot move backward")
    if after.expires_at_s > before.expires_at_s:
        raise ValueError("client key expiry extension requires reissue")
    if after.expires_at_s < before.expires_at_s:
        raise ValueError("client key expiry cannot change silently")

    if after.policy_id != before.policy_id:
        if policy_reassignment is None:
            raise ValueError("policy reassignment requires exact expected and target policy IDs")
        if policy_reassignment.expected_policy_id != before.policy_id:
            raise ValueError("policy reassignment expected policy does not match")
        if policy_reassignment.target_policy_id != after.policy_id:
            raise ValueError("policy reassignment target policy does not match")
    elif policy_reassignment is not None and (
        policy_reassignment.expected_policy_id != before.policy_id
        or policy_reassignment.target_policy_id != after.policy_id
    ):
        raise ValueError("policy reassignment does not match the record delta")

    if after.generation_high_water < before.generation_high_water:
        raise ValueError("credential generation high-water cannot roll back")

    old = {item.generation: item for item in before.credential_verifiers}
    new = {item.generation: item for item in after.credential_verifiers}
    for generation in old.keys() & new.keys():
        _validate_existing_verifier(old[generation], new[generation])
    for generation in new.keys() - old.keys():
        if generation <= before.generation_high_water:
            raise ValueError("retired credential generation reuse is forbidden")
    if after.generation_high_water > before.generation_high_water:
        if after.generation_high_water not in new:
            raise ValueError("generation high-water must name the new verifier")
    elif new.keys() - old.keys():
        raise ValueError("new verifier requires a higher generation")

    active = next(
        (
            item
            for item in after.credential_verifiers
            if item.status is CredentialVerifierStatus.ACTIVE
        ),
        None,
    )
    if active is not None:
        for retiring in (
            item
            for item in after.credential_verifiers
            if item.status is CredentialVerifierStatus.RETIRING
        ):
            if retiring.not_after_s - active.not_before_s > MAX_ROTATION_OVERLAP_S:
                raise ValueError("credential rotation overlap exceeds seven days")


def exact_policy_reassignments(
    before: Sequence[ClientKeyRecord],
    after: Sequence[ClientKeyRecord],
) -> dict[str, PolicyReassignment]:
    """Materialize exact expected→target policy changes for reviewed publication."""
    old = {record.key_id: record for record in before}
    return {
        record.key_id: PolicyReassignment(
            expected_policy_id=old[record.key_id].policy_id,
            target_policy_id=record.policy_id,
        )
        for record in after
        if record.key_id in old and record.policy_id != old[record.key_id].policy_id
    }


def validate_key_record_set_delta(
    before: Sequence[ClientKeyRecord],
    after: Sequence[ClientKeyRecord],
    *,
    policy_reassignments: Mapping[str, PolicyReassignment] | None = None,
) -> None:
    """Validate a complete key set so terminal tombstones cannot be dropped and reused."""
    old = {record.key_id: record for record in before}
    new = {record.key_id: record for record in after}
    missing = old.keys() - new.keys()
    if missing:
        raise ValueError("client key tombstones cannot be removed")
    assignments = policy_reassignments or {}
    for key_id in old.keys() & new.keys():
        if old[key_id] != new[key_id]:
            validate_key_record_delta(
                old[key_id],
                new[key_id],
                policy_reassignment=assignments.get(key_id),
            )
