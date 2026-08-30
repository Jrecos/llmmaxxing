from __future__ import annotations

import hmac
from collections.abc import Mapping, Set
from dataclasses import dataclass, replace

import pytest

from llmmaxxing.control.keys import issue_key, resume_key, rotate_key, suspend_key
from llmmaxxing.core.ids import BundleHash, PolicyRevisionId, RouteGroupId
from llmmaxxing.core.models import ClientKeyRecord, KeyPolicyRevision
from llmmaxxing.gateway import auth
from llmmaxxing.gateway.auth import (
    AuthenticatedClient,
    ClientAuthenticationError,
    parse_client_key,
    queued_identity_is_authorized,
    verify_client_key,
)

NOW = 2_000_000_000
PEPPERS = {"pepper-current": b"p" * 32, "pepper-prior": b"o" * 32}
POLICY_ID = PolicyRevisionId("pol_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
BUNDLE_HASH = BundleHash.from_digest("a" * 64)


def policy() -> KeyPolicyRevision:
    return KeyPolicyRevision(
        policy_id=POLICY_ID,
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


@dataclass(frozen=True)
class Runtime:
    key_index: Mapping[str, ClientKeyRecord]
    applied_bundle_generation: int = 7
    applied_bundle_hash: BundleHash = BUNDLE_HASH
    denied_key_ids: Set[str] = frozenset()
    accepted_peppers: Mapping[str, bytes] = None  # type: ignore[assignment]
    trusted_now_s: int = NOW

    def __post_init__(self) -> None:
        if self.accepted_peppers is None:
            object.__setattr__(self, "accepted_peppers", PEPPERS)


def issued_runtime() -> tuple[object, Runtime]:
    issued = issue_key(
        policy(),
        None,
        pepper=PEPPERS["pepper-current"],
        pepper_version="pepper-current",
        now_s=NOW,
    )
    return issued, Runtime({issued.record.key_id: issued.record})


def test_parser_accepts_only_canonical_lmxk1_material_and_rejects_query_or_multiples() -> None:
    issued, _ = issued_runtime()
    value = issued.reveal_once().value
    parsed = parse_client_key(value)

    prefix, encoded_id, encoded_secret = value.split(".")
    assert prefix == "lmxk1"
    assert len(encoded_id) == 22
    assert len(encoded_secret) == 43
    assert len(parsed.key_id_bytes) == 16
    assert len(parsed.secret) == 32

    malformed = (
        "",
        value + "=",
        value + ".extra",
        value.replace("lmxk1", "lmxk2", 1),
        value.replace(encoded_id, encoded_id[:-1], 1),
        value.replace(encoded_secret, encoded_secret[:-1] + "!", 1),
        f" {value}",
        f"Bearer {value}",
    )
    for candidate in malformed:
        with pytest.raises(ClientAuthenticationError, match="invalid client key"):
            parse_client_key(candidate)
    with pytest.raises(ClientAuthenticationError, match="invalid client key"):
        parse_client_key([value, value])
    with pytest.raises(ClientAuthenticationError, match="invalid client key"):
        parse_client_key(value, query_credentials=(value,))


def test_verify_returns_only_pre_body_identity_from_applied_runtime() -> None:
    issued, runtime = issued_runtime()
    parsed = parse_client_key(issued.reveal_once().value)

    client = verify_client_key(parsed, runtime)

    assert client == AuthenticatedClient(
        key_id=issued.record.key_id,
        accepted_credential_generation=1,
        policy_id=POLICY_ID,
        key_state=issued.record.state,
        key_expires_at_s=NOW + 90 * 86_400,
        applied_bundle_generation=7,
        applied_bundle_hash=BUNDLE_HASH,
    )
    assert client.active_attempt_remains_pinned
    for forbidden in ("route_group_id", "leg_ids", "account", "ceiling"):
        assert not hasattr(client, forbidden)


def test_unknown_id_runs_dummy_hmac_and_errors_are_non_enumerating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issued, runtime = issued_runtime()
    value = issued.reveal_once().value
    unknown_id = "A" * 22
    unknown = parse_client_key(".".join(("lmxk1", unknown_id, value.rsplit(".", 1)[1])))
    calls = 0
    real_hmac = auth.compute_client_key_verifier

    def counted(pepper: bytes, key_id: bytes, secret: bytes) -> bytes:
        nonlocal calls
        calls += 1
        return real_hmac(pepper, key_id, secret)

    compare_calls = 0
    real_compare = hmac.compare_digest

    def compared(left: bytes, right: bytes) -> bool:
        nonlocal compare_calls
        compare_calls += 1
        return real_compare(left, right)

    monkeypatch.setattr(auth, "compute_client_key_verifier", counted)
    monkeypatch.setattr(auth.hmac, "compare_digest", compared)

    with pytest.raises(ClientAuthenticationError) as unknown_error:
        verify_client_key(unknown, runtime)
    assert calls == 2
    assert compare_calls == 2

    wrong_secret = value[:-1] + ("A" if value[-1] != "A" else "B")
    with pytest.raises(ClientAuthenticationError) as known_error:
        verify_client_key(parse_client_key(wrong_secret), runtime)
    assert str(unknown_error.value) == str(known_error.value) == "invalid client key"


def test_current_and_prior_peppers_are_accepted_only_during_credential_overlap() -> None:
    issued = issue_key(
        policy(),
        None,
        pepper=PEPPERS["pepper-prior"],
        pepper_version="pepper-prior",
        now_s=NOW,
    )
    first_runtime = Runtime({issued.record.key_id: issued.record})
    first_value = issued.reveal_once().value
    rotated = rotate_key(
        issued.record,
        pepper=PEPPERS["pepper-current"],
        pepper_version="pepper-current",
        now_s=NOW + 60,
    )
    second_value = rotated.reveal_once().value
    overlap = replace(
        first_runtime,
        key_index={rotated.record.key_id: rotated.record},
        trusted_now_s=NOW + 120,
    )

    assert (
        verify_client_key(parse_client_key(first_value), overlap).accepted_credential_generation
        == 1
    )
    assert (
        verify_client_key(parse_client_key(second_value), overlap).accepted_credential_generation
        == 2
    )
    without_prior = replace(
        overlap,
        accepted_peppers={"pepper-current": PEPPERS["pepper-current"]},
    )
    with pytest.raises(ClientAuthenticationError):
        verify_client_key(parse_client_key(first_value), without_prior)
    assert (
        verify_client_key(
            parse_client_key(second_value), without_prior
        ).accepted_credential_generation
        == 2
    )
    too_many_peppers = replace(
        overlap,
        accepted_peppers={**PEPPERS, "pepper-ancient": b"a" * 32},
    )
    with pytest.raises(ClientAuthenticationError):
        verify_client_key(parse_client_key(second_value), too_many_peppers)

    after_overlap = replace(overlap, trusted_now_s=NOW + 60 + 86_400)
    with pytest.raises(ClientAuthenticationError):
        verify_client_key(parse_client_key(first_value), after_overlap)
    assert (
        verify_client_key(
            parse_client_key(second_value), after_overlap
        ).accepted_credential_generation
        == 2
    )


def test_expiry_has_no_grace_and_suspend_revoke_stop_queued_not_active() -> None:
    issued = issue_key(
        policy(),
        NOW + 1,
        pepper=PEPPERS["pepper-current"],
        pepper_version="pepper-current",
        now_s=NOW,
    )
    value = issued.reveal_once().value
    live = Runtime({issued.record.key_id: issued.record})
    client = verify_client_key(parse_client_key(value), live)

    assert queued_identity_is_authorized(client, live)
    assert client.active_attempt_remains_pinned
    suspended_record = suspend_key(issued.record, now_s=NOW)
    suspended = replace(live, key_index={issued.record.key_id: suspended_record})
    assert not queued_identity_is_authorized(client, suspended)
    assert client.active_attempt_remains_pinned
    resumed_record = resume_key(suspended_record, now_s=NOW)
    assert queued_identity_is_authorized(
        client,
        replace(live, key_index={issued.record.key_id: resumed_record}),
    )

    expired = replace(live, trusted_now_s=NOW + 1)
    assert not queued_identity_is_authorized(client, expired)
    with pytest.raises(ClientAuthenticationError):
        verify_client_key(parse_client_key(value), expired)

    denied = replace(live, denied_key_ids=frozenset({issued.record.key_id}))
    assert not queued_identity_is_authorized(client, denied)
    assert client.active_attempt_remains_pinned
