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
    ParsedLegacyClientKey,
    compute_client_key_verifier,
    compute_legacy_key_fingerprint,
    parse_client_key_material,
    parse_legacy_client_key_material,
)
from llmmaxxing.core.models import (
    ClientCredentialVerifier,
    ClientKeyRecord,
    LegacyClientCredentialVerifier,
)
from llmmaxxing.core.state_machines import CredentialVerifierStatus, KeyLifecycleState

_PUBLIC_ERROR = "invalid client key"
_ACCEPTED_STATUSES = frozenset((CredentialVerifierStatus.ACTIVE, CredentialVerifierStatus.RETIRING))
_DUMMY_PEPPER = b"\x00" * 32
_DUMMY_CREDENTIALS = (
    ClientCredentialVerifier(
        generation=1,
        verifier_hex="0" * 64,
        pepper_version="dummy-1",
        not_before_s=1,
        not_after_s=2,
        status=CredentialVerifierStatus.RETIRED,
    ),
    ClientCredentialVerifier(
        generation=2,
        verifier_hex="f" * 64,
        pepper_version="dummy-2",
        not_before_s=2,
        not_after_s=3,
        status=CredentialVerifierStatus.RETIRED,
    ),
)
_DUMMY_RECORD = ClientKeyRecord(
    key_id="0" * 32,
    policy_id=PolicyRevisionId("pol_00000000-0000-4000-8000-000000000000"),
    state=KeyLifecycleState.REVOKED,
    issued_at_s=1,
    expires_at_s=4,
    time_high_water_s=2,
    generation_high_water=2,
    credential_verifiers=_DUMMY_CREDENTIALS,
)


@dataclass(frozen=True, slots=True)
class LegacyKeyIndexEntry:
    key_id: str
    credential_generation: int


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
    def legacy_key_index(self) -> Mapping[str, LegacyKeyIndexEntry]: ...

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
) -> ParsedClientKey | ParsedLegacyClientKey:
    """Reject ambiguity, then parse canonical or bounded migration key material."""
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
        try:
            return parse_legacy_client_key_material(candidate)
        except (InvalidClientKeyMaterial, TypeError):
            raise ClientAuthenticationError() from None


def _credential_is_accepted(
    verifier: ClientCredentialVerifier | LegacyClientCredentialVerifier,
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


def _prepare_verifier(
    verifier: ClientCredentialVerifier,
    record: ClientKeyRecord,
    runtime_view: AuthRuntimeView,
    fallback_pepper: bytes,
) -> tuple[bytes, bytes, bool]:
    """Perform the same field preparation for real and dummy verifier slots."""
    status_is_accepted = verifier.status in _ACCEPTED_STATUSES
    pepper_is_accepted = verifier.pepper_version in runtime_view.accepted_peppers
    pepper = runtime_view.accepted_peppers.get(verifier.pepper_version, fallback_pepper)
    expected = bytes.fromhex(verifier.verifier_hex)
    now_s = runtime_view.trusted_now_s
    window_is_accepted = verifier.not_before_s <= now_s < verifier.not_after_s
    record_is_live = now_s < record.expires_at_s
    eligible = status_is_accepted and pepper_is_accepted and window_is_accepted and record_is_live
    return pepper, expected, eligible


def _legacy_index_key(pepper_version: str, fingerprint_hex: str) -> str:
    return f"{pepper_version}:{fingerprint_hex}"


def build_legacy_key_index(
    records: Sequence[ClientKeyRecord],
) -> dict[str, LegacyKeyIndexEntry]:
    index: dict[str, LegacyKeyIndexEntry] = {}
    for record in records:
        for verifier in record.legacy_verifiers:
            key = _legacy_index_key(verifier.pepper_version, verifier.fingerprint_hex)
            if key in index:
                raise ValueError("duplicate legacy credential fingerprint")
            index[key] = LegacyKeyIndexEntry(record.key_id, verifier.generation)
    return index


def _authenticated(
    record: ClientKeyRecord,
    accepted_generation: int,
    runtime_view: AuthRuntimeView,
) -> AuthenticatedClient:
    return AuthenticatedClient(
        key_id=record.key_id,
        accepted_credential_generation=accepted_generation,
        policy_id=record.policy_id,
        key_state=record.state,
        key_expires_at_s=record.expires_at_s,
        applied_bundle_generation=runtime_view.applied_bundle_generation,
        applied_bundle_hash=runtime_view.applied_bundle_hash,
    )


def _verify_legacy_client_key(
    parsed: ParsedLegacyClientKey,
    runtime_view: AuthRuntimeView,
) -> AuthenticatedClient:
    pepper_set_is_valid = 1 <= len(runtime_view.accepted_peppers) <= 2 and all(
        len(pepper) >= 32 for pepper in runtime_view.accepted_peppers.values()
    )
    slots = list(runtime_view.accepted_peppers.items())[:2]
    slots.extend((f"dummy-{index}", _DUMMY_PEPPER) for index in range(2 - len(slots)))
    accepted: tuple[ClientKeyRecord, int] | None = None
    for pepper_version, pepper in slots:
        candidate = compute_legacy_key_fingerprint(pepper, parsed.token)
        entry = runtime_view.legacy_key_index.get(
            _legacy_index_key(pepper_version, candidate.hex())
        )
        record = runtime_view.key_index.get(entry.key_id) if entry is not None else None
        verifier = (
            next(
                (
                    item
                    for item in record.legacy_verifiers
                    if entry is not None and item.generation == entry.credential_generation
                ),
                None,
            )
            if record is not None
            else None
        )
        expected = bytes.fromhex(verifier.fingerprint_hex) if verifier is not None else b"\x00" * 32
        matches = hmac.compare_digest(candidate, expected)
        if (
            matches
            and record is not None
            and verifier is not None
            and _credential_is_accepted(verifier, record, runtime_view)
        ):
            accepted = (record, verifier.generation)
    if (
        accepted is None
        or not pepper_set_is_valid
        or accepted[0].state is not KeyLifecycleState.ENABLED
        or accepted[0].key_id in runtime_view.denied_key_ids
        or runtime_view.trusted_now_s >= accepted[0].expires_at_s
    ):
        raise ClientAuthenticationError()
    return _authenticated(accepted[0], accepted[1], runtime_view)


def verify_client_key(
    parsed: ParsedClientKey | ParsedLegacyClientKey,
    runtime_view: AuthRuntimeView,
) -> AuthenticatedClient:
    """Verify locally with bounded HMAC work and no plaintext credential index."""
    if isinstance(parsed, ParsedLegacyClientKey):
        return _verify_legacy_client_key(parsed, runtime_view)
    pepper_set_is_valid = 1 <= len(runtime_view.accepted_peppers) <= 2 and all(
        len(pepper) >= 32 for pepper in runtime_view.accepted_peppers.values()
    )
    record = runtime_view.key_index.get(parsed.key_id)
    prepared_record = record if record is not None else _DUMMY_RECORD
    verifiers = record.credential_verifiers if record is not None else _DUMMY_CREDENTIALS
    fallback_pepper = next(iter(runtime_view.accepted_peppers.values()), _DUMMY_PEPPER)
    accepted_generation: int | None = None

    for index in range(2):
        verifier = verifiers[index] if index < len(verifiers) else _DUMMY_CREDENTIALS[index]
        pepper, expected, credential_is_accepted = _prepare_verifier(
            verifier,
            prepared_record,
            runtime_view,
            fallback_pepper,
        )
        candidate = compute_client_key_verifier(pepper, parsed.key_id_bytes, parsed.secret)
        matches = hmac.compare_digest(candidate, expected)
        if matches and record is not None and credential_is_accepted:
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
    return _authenticated(record, accepted_generation, runtime_view)


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
    verifiers: tuple[ClientCredentialVerifier | LegacyClientCredentialVerifier, ...] = (
        *record.credential_verifiers,
        *record.legacy_verifiers,
    )
    return any(
        verifier.generation == client.accepted_credential_generation
        and _credential_is_accepted(verifier, record, runtime_view)
        for verifier in verifiers
    )
