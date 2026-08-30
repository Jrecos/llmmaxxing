from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Mapping

from llmmaxxing.control.keys import issue_key
from llmmaxxing.core.ids import BundleHash, PolicyRevisionId, RouteGroupId
from llmmaxxing.core.models import ClientKeyRecord, KeyPolicyRevision
from llmmaxxing.gateway.auth import parse_client_key, verify_client_key

NOW = 2_000_000_000
PEPPER = b"p" * 32


class CountingIndex(dict[str, ClientKeyRecord]):
    get_calls = 0

    def get(self, key: str, default: ClientKeyRecord | None = None) -> ClientKeyRecord | None:
        self.get_calls += 1
        return super().get(key, default)


@dataclass(frozen=True)
class Runtime:
    key_index: Mapping[str, ClientKeyRecord]
    applied_bundle_generation: int
    applied_bundle_hash: BundleHash
    denied_key_ids: AbstractSet[str]
    accepted_peppers: Mapping[str, bytes]
    trusted_now_s: int


def test_auth_lookup_is_functionally_constant_with_100_000_keys() -> None:
    policy = KeyPolicyRevision(
        policy_id=PolicyRevisionId("pol_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        name="benchmark",
        route_group_ids=(RouteGroupId("rg_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),),
        allowed_account_ids=("acc_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",),
        allowed_triggers=("primary",),
        queue_tier=20,
        queue_weight=2,
        max_concurrency=2,
        max_waiters=4,
        deadline_ms=60_000,
    )
    issued = issue_key(
        policy,
        None,
        pepper=PEPPER,
        pepper_version="p1",
        now_s=NOW,
    )
    value = issued.reveal_once().value
    index = CountingIndex({f"{number:032x}": issued.record for number in range(100_000)})
    index[issued.record.key_id] = issued.record
    runtime = Runtime(
        key_index=index,
        applied_bundle_generation=1,
        applied_bundle_hash=BundleHash.from_digest("a" * 64),
        denied_key_ids=frozenset(),
        accepted_peppers={"p1": PEPPER},
        trusted_now_s=NOW,
    )

    assert verify_client_key(parse_client_key(value), runtime).key_id == issued.record.key_id
    assert index.get_calls == 1
