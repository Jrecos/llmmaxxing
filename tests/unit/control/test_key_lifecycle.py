from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, replace

import pytest

from llmmaxxing.control.keys import (
    PolicyReassignment,
    activate_key,
    expire_key,
    issue_key,
    reassign_key_policy,
    resume_key,
    revoke_key,
    rotate_key,
    suspend_key,
    validate_key_record_delta,
)
from llmmaxxing.core.ids import BundleHash, PolicyRevisionId, RouteGroupId
from llmmaxxing.core.models import (
    ClientCredentialVerifier,
    ClientKeyRecord,
    KeyPolicyRevision,
)
from llmmaxxing.core.state_machines import CredentialVerifierStatus, KeyLifecycleState
from llmmaxxing.gateway.auth import (
    ClientAuthenticationError,
    parse_client_key,
    queued_identity_is_authorized,
    verify_client_key,
)

NOW = 2_000_000_000
DAY = 86_400
PEPPER = b"p" * 32
POLICY_ID = PolicyRevisionId("pol_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OTHER_POLICY_ID = PolicyRevisionId("pol_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
BUNDLE_HASH = BundleHash.from_digest("a" * 64)


def policy(policy_id: PolicyRevisionId = POLICY_ID) -> KeyPolicyRevision:
    return KeyPolicyRevision(
        policy_id=policy_id,
        name="client",
        route_group_ids=(RouteGroupId("rg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),),
        allowed_account_ids=("acc_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",),
        allowed_triggers=("primary",),
        queue_tier=20,
        queue_weight=2,
        max_concurrency=2,
        max_waiters=4,
        deadline_ms=60_000,
    )


def issue(*, expiry: int | None = None):
    return issue_key(
        policy(),
        expiry,
        pepper=PEPPER,
        pepper_version="p1",
        now_s=NOW,
    )


def test_issue_defaults_to_90_days_caps_at_365_and_returns_plaintext_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real_token_bytes = __import__("secrets").token_bytes

    def counted_token_bytes(length: int) -> bytes:
        calls.append(length)
        return real_token_bytes(length)

    monkeypatch.setattr("llmmaxxing.control.keys.secrets.token_bytes", counted_token_bytes)
    issued = issue()
    assert calls[:2] == [16, 32]
    assert issued.record.expires_at_s == NOW + 90 * DAY
    assert issued.record.generation_high_water == 1
    assert issued.record.credential_verifiers[0].generation == 1
    assert issued.record.credential_verifiers[0].status is CredentialVerifierStatus.ACTIVE
    assert issued.response_headers == {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
    }

    revealed = issued.reveal_once()
    assert revealed.value.startswith("lmxk1.")
    assert revealed.response_headers == issued.response_headers
    assert revealed.value not in repr(revealed)
    assert revealed.value not in repr(issued)
    assert issued.record.credential_verifiers[0].verifier_hex not in repr(issued)
    with pytest.raises(RuntimeError, match="already revealed"):
        issued.reveal_once()

    assert issue(expiry=NOW + 365 * DAY).record.expires_at_s == NOW + 365 * DAY
    with pytest.raises(ValueError, match="365 days"):
        issue(expiry=NOW + 365 * DAY + 1)
    with pytest.raises(ValueError, match="future"):
        issue(expiry=NOW)


def test_rotation_uses_monotonic_generations_default_overlap_and_maximum() -> None:
    first = issue()
    first_value = first.reveal_once().value
    second = rotate_key(
        first.record,
        pepper=PEPPER,
        pepper_version="p1",
        now_s=NOW + DAY,
    )
    second_value = second.reveal_once().value
    assert second.record.generation_high_water == 2
    assert [item.generation for item in second.record.credential_verifiers] == [1, 2]
    assert second.record.credential_verifiers[0].status is CredentialVerifierStatus.RETIRING
    assert second.record.credential_verifiers[0].not_after_s == NOW + 2 * DAY

    third = rotate_key(
        second.record,
        pepper=PEPPER,
        pepper_version="p1",
        now_s=NOW + 2 * DAY,
        overlap_s=7 * DAY,
    )
    assert third.record.generation_high_water == 3
    assert [item.generation for item in third.record.credential_verifiers] == [2, 3]
    assert all(item.generation != 1 for item in third.record.credential_verifiers)
    with pytest.raises(ValueError, match="seven days"):
        rotate_key(
            third.record,
            pepper=PEPPER,
            pepper_version="p1",
            now_s=NOW + 3 * DAY,
            overlap_s=7 * DAY + 1,
        )

    immediate = rotate_key(
        third.record,
        pepper=PEPPER,
        pepper_version="p1",
        now_s=NOW + 3 * DAY,
        overlap_s=0,
    )
    assert immediate.record.generation_high_water == 4
    assert immediate.record.credential_verifiers[0].status is CredentialVerifierStatus.RETIRED
    assert immediate.record.credential_verifiers[0].not_after_s == NOW + 3 * DAY
    assert second_value != first_value


def test_suspend_resume_revoke_and_expiry_are_fail_closed_and_terminal() -> None:
    issued = issue(expiry=NOW + 10)
    value = issued.reveal_once().value
    suspended = suspend_key(issued.record, now_s=NOW + 1)
    assert suspended.state is KeyLifecycleState.SUSPENDED
    resumed = resume_key(suspended, now_s=NOW + 2)
    assert resumed.state is KeyLifecycleState.ENABLED
    revoked = revoke_key(resumed, now_s=NOW + 3)
    assert revoked.state is KeyLifecycleState.REVOKED
    assert all(
        verifier.status is CredentialVerifierStatus.RETIRED
        for verifier in revoked.credential_verifiers
    )
    with pytest.raises(ValueError, match="terminal"):
        resume_key(revoked, now_s=NOW + 4)
    with pytest.raises(ValueError, match="expired"):
        suspend_key(issued.record, now_s=NOW + 10)

    @dataclass(frozen=True)
    class Runtime:
        key_index: Mapping[str, ClientKeyRecord]
        applied_bundle_generation: int = 1
        applied_bundle_hash: BundleHash = BUNDLE_HASH
        denied_key_ids: Set[str] = frozenset()
        accepted_peppers: Mapping[str, bytes] = None  # type: ignore[assignment]
        trusted_now_s: int = NOW

        def __post_init__(self) -> None:
            if self.accepted_peppers is None:
                object.__setattr__(self, "accepted_peppers", {"p1": PEPPER})

    expired_record = expire_key(issued.record, now_s=NOW + 10)
    assert expired_record.time_high_water_s == NOW + 10
    assert all(
        verifier.status is CredentialVerifierStatus.EXPIRED
        for verifier in expired_record.credential_verifiers
    )
    with pytest.raises(ValueError, match="terminal"):
        resume_key(expired_record, now_s=NOW + 11)
    with pytest.raises(ValueError, match="terminal key"):
        validate_key_record_delta(expired_record, issued.record)

    live = Runtime({issued.record.key_id: issued.record})
    client = verify_client_key(parse_client_key(value), live)
    assert client.active_attempt_remains_pinned
    assert not queued_identity_is_authorized(
        client,
        replace(live, key_index={revoked.key_id: revoked}, trusted_now_s=NOW + 3),
    )
    with pytest.raises(ClientAuthenticationError):
        verify_client_key(
            parse_client_key(value),
            replace(live, key_index={revoked.key_id: revoked}, trusted_now_s=NOW + 3),
        )


def test_lifecycle_delta_rejects_generation_rollback_reuse_resurrection_and_expiry_extension() -> (
    None
):
    issued = issue()
    rotated = rotate_key(
        issued.record,
        pepper=PEPPER,
        pepper_version="p1",
        now_s=NOW + 1,
        overlap_s=0,
    ).record

    with pytest.raises(ValueError, match="generation high-water"):
        validate_key_record_delta(
            rotated,
            rotated.model_copy(update={"generation_high_water": 1}),
        )

    retired = rotated.credential_verifiers[0]
    resurrected = retired.model_copy(
        update={"status": CredentialVerifierStatus.ACTIVE, "not_after_s": rotated.expires_at_s}
    )
    with pytest.raises(ValueError, match="terminal credential|active newest"):
        validate_key_record_delta(
            rotated,
            rotated.model_copy(
                update={
                    "credential_verifiers": (
                        resurrected,
                        rotated.credential_verifiers[1],
                    )
                }
            ),
        )

    third = rotate_key(
        rotated,
        pepper=PEPPER,
        pepper_version="p1",
        now_s=NOW + 2,
        overlap_s=0,
    ).record
    reused = ClientCredentialVerifier(
        generation=1,
        verifier_hex="0" * 64,
        pepper_version="p1",
        not_before_s=NOW + 2,
        not_after_s=NOW + 2,
        status=CredentialVerifierStatus.RETIRED,
    )
    with pytest.raises(ValueError, match="reuse"):
        validate_key_record_delta(
            third,
            third.model_copy(
                update={
                    "credential_verifiers": (reused, third.credential_verifiers[1]),
                }
            ),
        )

    with pytest.raises(ValueError, match="expiry extension"):
        validate_key_record_delta(
            issued.record,
            issued.record.model_copy(update={"expires_at_s": issued.record.expires_at_s + 1}),
        )

    revoked = revoke_key(issued.record, now_s=NOW + 1)
    with pytest.raises(ValueError, match="terminal key"):
        validate_key_record_delta(revoked, issued.record)


def test_delta_bounds_overlap_requires_exact_reassignment_and_rejects_clock_rollback() -> None:
    issued = issue()
    reassignment = PolicyReassignment(
        expected_policy_id=POLICY_ID,
        target_policy_id=OTHER_POLICY_ID,
    )
    moved = reassign_key_policy(
        issued.record,
        reassignment,
        now_s=NOW + 1,
    )
    assert moved.policy_id == OTHER_POLICY_ID
    with pytest.raises(ValueError, match="policy reassignment"):
        validate_key_record_delta(issued.record, moved)
    validate_key_record_delta(issued.record, moved, policy_reassignment=reassignment)

    wrong = PolicyReassignment(
        expected_policy_id=OTHER_POLICY_ID,
        target_policy_id=POLICY_ID,
    )
    with pytest.raises(ValueError, match="expected policy"):
        reassign_key_policy(issued.record, wrong, now_s=NOW + 1)

    with pytest.raises(ValueError, match="trusted clock"):
        suspend_key(moved, now_s=NOW)

    active = moved.credential_verifiers[-1]
    overlapping = (
        active.model_copy(
            update={
                "status": CredentialVerifierStatus.RETIRING,
                "not_after_s": NOW + 8 * DAY,
            }
        ),
        active.model_copy(
            update={
                "generation": 2,
                "verifier_hex": "1" * 64,
                "not_before_s": NOW,
            }
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_key_record_delta(
            moved,
            moved.model_copy(
                update={
                    "credential_verifiers": overlapping,
                    "generation_high_water": 2,
                }
            ),
        )


def test_activation_is_draft_only() -> None:
    issued = issue()
    draft = issued.record.model_copy(update={"state": KeyLifecycleState.DRAFT})
    assert activate_key(draft, now_s=NOW + 1).state is KeyLifecycleState.ENABLED
    with pytest.raises(ValueError, match="illegal"):
        activate_key(issued.record, now_s=NOW + 1)
