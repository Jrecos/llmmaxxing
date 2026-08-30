"""RFC8785-compatible canonicalization, domain-separated hashing and schema parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_models import account, bundle, key_record, leg, policy, route_group

from llmmaxxing.core.canonical import (
    bundle_hash,
    canonical_bundle_bytes,
    canonical_json_bytes,
    content_hash,
    emit_bundle_schema,
)
from llmmaxxing.core.ids import AccountId, PolicyRevisionId, RouteGroupId, RouteLegId
from llmmaxxing.core.models import PolicyBundleV1
from llmmaxxing.core.reasons import RouteTrigger


def bundle_fixture(order: str = "forward") -> PolicyBundleV1:
    """Identical content spelled with two different input orders."""
    assert order in ("forward", "reverse")
    # fixed ids: forward and reverse spellings must describe one identical bundle
    primary = account(name="nan", account_id=AccountId("acc_11111111-1111-4111-8111-111111111111"))
    spill = account(name="spill", account_id=AccountId("acc_22222222-2222-4222-9222-222222222222"))
    group_id = RouteGroupId("rg_33333333-3333-4333-8333-333333333333")
    legs = (
        leg(
            primary.account_id,
            10,
            (RouteTrigger.PRIMARY, RouteTrigger.CAPACITY_SPILL),
            leg_id=RouteLegId("leg_44444444-4444-4444-8444-444444444444"),
        ),
        leg(
            spill.account_id,
            20,
            (RouteTrigger.CAPACITY_SPILL,),
            leg_id=RouteLegId("leg_55555555-5555-4555-9555-555555555555"),
        ),
    )
    pol = policy(
        (group_id,),
        (primary.account_id, spill.account_id),
        policy_id=PolicyRevisionId("pol_66666666-6666-4666-8666-666666666666"),
    )
    if order == "forward":
        return bundle(
            accounts=(primary, spill),
            route_groups=(route_group(legs, group_id=group_id),),
            policies=(pol,),
            keys=(key_record(pol.policy_id),),
        )
    return bundle(
        accounts=(spill, primary),
        route_groups=(route_group(tuple(reversed(legs)), group_id=group_id, name="dsv4"),),
        policies=(pol,),
        keys=(key_record(pol.policy_id),),
    )


def test_canonical_hash_ignores_input_order():
    left = canonical_bundle_bytes(bundle_fixture(order="forward"))
    right = canonical_bundle_bytes(bundle_fixture(order="reverse"))
    assert left == right
    assert bundle_hash(left) == bundle_hash(right)


def test_canonical_output_is_sorted_compact_json() -> None:
    data = {"b": 1, "a": {"d": [1, 2], "c": "x"}, "e": True, "f": None}
    assert canonical_json_bytes(data) == b'{"a":{"c":"x","d":[1,2]},"b":1,"e":true,"f":null}'


def test_no_whitespace_between_tokens() -> None:
    assert canonical_json_bytes({"a": [1, {"b": 2}]}) == b'{"a":[1,{"b":2}]}'


def test_jcs_numeric_form() -> None:
    # integral floats share the integer form (ECMAScript Number->String)
    assert canonical_json_bytes({"a": 1.0}) == canonical_json_bytes({"a": 1})
    assert canonical_json_bytes({"a": 1}) != canonical_json_bytes({"a": "1"})
    assert canonical_json_bytes({"a": 1}) != canonical_json_bytes({"a": True})
    assert canonical_json_bytes({"a": 1e21}) == b'{"a":1e+21}'
    assert canonical_json_bytes({"a": -0.0}) == b'{"a":0}'
    assert canonical_json_bytes({"a": 1.5}) == b'{"a":1.5}'
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json_bytes({"a": bad})


def test_jcs_string_and_key_ordering() -> None:
    # non-ASCII is emitted as raw UTF-8, never \\u escapes
    assert canonical_json_bytes({"k": "ñ"}) == '{"k":"ñ"}'.encode()
    assert canonical_json_bytes({"k": 'a"b\\c\nd'}) == b'{"k":"a\\"b\\\\c\\nd"}'
    # keys sort by UTF-16 code units: an astral char encodes as surrogate pairs
    # (0xD83D...) and therefore precedes U+F900, unlike codepoint order
    astral, bmp = "\U0001f600", "￰"
    assert sorted([astral, bmp]) == [bmp, astral]  # codepoint order differs
    expected = json.dumps({astral: 1, bmp: 2}, ensure_ascii=False, separators=(",", ":")).encode()
    assert canonical_json_bytes({bmp: 2, astral: 1}) == expected


def test_hash_is_domain_separated_and_content_sensitive() -> None:
    payload = canonical_bundle_bytes(bundle_fixture())
    assert content_hash(payload) != content_hash(payload, domain=b"llmmaxxing-impact-v1\x00")
    assert str(bundle_hash(payload)).startswith("bh_")
    again = canonical_bundle_bytes(bundle_fixture(order="reverse"))
    assert bundle_hash(payload) == bundle_hash(again)
    assert bundle_hash(payload) != bundle_hash(canonical_bundle_bytes(bundle(generation=8)))


def test_semantic_change_changes_hash() -> None:
    base = canonical_bundle_bytes(bundle_fixture())
    acc = account(name="other")
    group = route_group((leg(acc.account_id, 10, (RouteTrigger.PRIMARY,)),))
    pol = policy((group.route_group_id,), (acc.account_id,))
    alt = bundle(
        accounts=(acc,),
        route_groups=(group,),
        policies=(pol,),
        keys=(key_record(pol.policy_id),),
    )
    assert bundle_hash(canonical_bundle_bytes(alt)) != bundle_hash(base)


def test_feature_bits_and_min_reader_are_canonicalized() -> None:
    # sorted once at canonicalization: two spellings of the same set hash equal
    base = bundle()
    left = base.model_copy(
        update={"required_features": ("weighted_fair_queue", "ordered_capacity")}
    )
    right = base.model_copy(
        update={"required_features": ("ordered_capacity", "weighted_fair_queue")}
    )
    assert canonical_bundle_bytes(left) == canonical_bundle_bytes(right)
    with_extra = base.model_copy(
        update={
            "required_features": (
                "expiry_deny_overlay",
                "ordered_capacity",
                "weighted_fair_queue",
            )
        }
    )
    assert canonical_bundle_bytes(with_extra) != canonical_bundle_bytes(right)
    other_reader = base.model_copy(update={"min_reader": "0.9"})
    assert canonical_bundle_bytes(other_reader) != canonical_bundle_bytes(right)


def test_checked_in_schema_matches_generated_output_byte_for_byte() -> None:
    generated = emit_bundle_schema()
    checked_in = Path(__file__).resolve().parents[3] / "schemas" / "bundle-v1.json"
    assert checked_in.read_bytes() == generated, (
        "regenerate with: uv run python -m llmmaxxing.core.canonical > schemas/bundle-v1.json"
    )
