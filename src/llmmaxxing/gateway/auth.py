"""O(1) local client-key verification before any request-body allocation."""

from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Protocol

from llmmaxxing.core.ids import BundleHash, PolicyRevisionId
from llmmaxxing.core.key_material import (
    InvalidClientKeyMaterial,
    ParsedClientKey,
    compute_client_key_verifier,
    parse_client_key_material,
)
from llmmaxxing.core.models import ClientCredentialVerifier, ClientKeyRecord
from llmmaxxing.core.state_machines import CredentialVerifierStatus, KeyLifecycleState

_PUBLIC_ERROR = "invalid client key"
_ACCEPTED_STATUSES = frozenset((CredentialVerifierStatus.ACTIVE, CredentialVerifierStatus.RETIRING))
_DUMMY_PEPPER = b"\x00" * 32
_DUMMY_VERIFIER = b"\x00" * 32


class ClientAuthenticationError(ValueError):
    """Bounded non-enumerating public authentication failure."""

    def __init__(self) -> None:
        super().__init__(_PUBLIC_ERROR)


class AuthRuntimeView(Protocol):
    """Immutable authentication slice of the wider operational runtime view."""

    @property
    def key_index(self) -> Mapping[str, ClientKeyRecord]: ...

    @property
    def applied_bundle_generation(self) -> int: ...

    @property
    def applied_bundle_hash(self) -> BundleHash: ...

    @property
    def denied_key_ids(self) -> Set[str]: ...

    @property
    def accepted_peppers(self) -> Mapping[str, bytes]: ...

    @property
    def trusted_now_s(self) -> int: ...


@dataclass(frozen=True, slots=True)
class AuthenticatedClient:
    """Pre-body identity pinned to the exact accepted credential and bundle."""

    key_id: str
    accepted_credential_generation: int
    policy_id: PolicyRevisionId
    key_state: KeyLifecycleState
    key_expires_at_s: int
    applied_bundle_generation: int
    applied_bundle_hash: BundleHash

    @property
    def active_attempt_remains_pinned(self) -> bool:
        return True


def parse_client_key(
    value: str | Sequence[str] | None,
    *,
    query_credentials: Sequence[str] = (),
) -> ParsedClientKey:
    """Reject ambiguity before body read, then parse exact canonical token material."""
    if query_credentials or value is None:
        raise ClientAuthenticationError()
    if isinstance(value, str):
        candidate = value
    else:
        values = tuple(value)
        if len(values) != 1:
            raise ClientAuthenticationError()
        candidate = values[0]
    try:
        return parse_client_key_material(candidate)
    except (InvalidClientKeyMaterial, TypeError):
        raise ClientAuthenticationError() from None


def _credential_is_accepted(
    verifier: ClientCredentialVerifier,
    record: ClientKeyRecord,
    runtime_view: AuthRuntimeView,
) -> bool:
    now_s = runtime_view.trusted_now_s
    return (
        verifier.status in _ACCEPTED_STATUSES
        and verifier.pepper_version in runtime_view.accepted_peppers
        and verifier.not_before_s <= now_s < verifier.not_after_s
        and now_s < record.expires_at_s
    )


def verify_client_key(
    parsed: ParsedClientKey,
    runtime_view: AuthRuntimeView,
) -> AuthenticatedClient:
    """Verify locally with one index lookup and a fixed two constant-time compares."""
    pepper_set_is_valid = 1 <= len(runtime_view.accepted_peppers) <= 2 and all(
        len(pepper) >= 32 for pepper in runtime_view.accepted_peppers.values()
    )
    record = runtime_view.key_index.get(parsed.key_id)
    verifiers = record.credential_verifiers if record is not None else ()
    fallback_pepper = next(iter(runtime_view.accepted_peppers.values()), _DUMMY_PEPPER)
    accepted_generation: int | None = None

    for index in range(2):
        verifier = verifiers[index] if index < len(verifiers) else None
        pepper = (
            runtime_view.accepted_peppers.get(verifier.pepper_version, fallback_pepper)
            if verifier is not None
            else fallback_pepper
        )
        expected = bytes.fromhex(verifier.verifier_hex) if verifier is not None else _DUMMY_VERIFIER
        candidate = compute_client_key_verifier(pepper, parsed.key_id_bytes, parsed.secret)
        matches = hmac.compare_digest(candidate, expected)
        if (
            matches
            and verifier is not None
            and record is not None
            and _credential_is_accepted(verifier, record, runtime_view)
        ):
            accepted_generation = verifier.generation

    if (
        record is None
        or accepted_generation is None
        or record.state is not KeyLifecycleState.ENABLED
        or not pepper_set_is_valid
        or record.key_id in runtime_view.denied_key_ids
        or runtime_view.trusted_now_s >= record.expires_at_s
    ):
        raise ClientAuthenticationError()

    return AuthenticatedClient(
        key_id=record.key_id,
        accepted_credential_generation=accepted_generation,
        policy_id=record.policy_id,
        key_state=record.state,
        key_expires_at_s=record.expires_at_s,
        applied_bundle_generation=runtime_view.applied_bundle_generation,
        applied_bundle_hash=runtime_view.applied_bundle_hash,
    )


def queued_identity_is_authorized(
    client: AuthenticatedClient,
    runtime_view: AuthRuntimeView,
) -> bool:
    """Recheck queued work; active attempts deliberately never call this path."""
    record = runtime_view.key_index.get(client.key_id)
    if (
        record is None
        or record.state is not KeyLifecycleState.ENABLED
        or record.policy_id != client.policy_id
        or record.key_id in runtime_view.denied_key_ids
        or runtime_view.trusted_now_s >= record.expires_at_s
    ):
        return False
    return any(
        verifier.generation == client.accepted_credential_generation
        and _credential_is_accepted(verifier, record, runtime_view)
        for verifier in record.credential_verifiers
    )
