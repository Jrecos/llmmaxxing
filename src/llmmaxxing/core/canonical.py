"""RFC 8785 (JCS) canonical JSON, domain-separated hashing and schema emission.

Pure module: string/bytes in, string/bytes out.  Numbers follow the ECMAScript
``Number::toString`` contract required by RFC 8785; dictionary keys are sorted
by UTF-16 code units, exactly as RFC 8785 specifies.  Bundle collections are
additionally sorted element-wise (see :func:`canonical_bundle_bytes`) so that
the content hash depends on the bundle's semantic content, never on the order
members were handed to the model constructor.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from typing import Any

from llmmaxxing.core.ids import BundleHash
from llmmaxxing.core.models import PolicyBundleV1

#: Domain prefix for bundle content hashes.  Canonical JSON never contains a
#: NUL byte (control characters are escaped), so a NUL-terminated domain makes
#: the ``domain || payload`` boundary unambiguous.
_BUNDLE_DOMAIN = b"llmmaxxing.bundle.v1\x00"

_ES6_NUMBER = re.compile(r"(\d+)(?:\.(\d*))?(?:[eE]([+-]\d+))?")
_MAX_SAFE_INTEGER = (1 << 53) - 1


def _es6_number(value: float) -> str:
    """Render a float exactly as ECMAScript ``Number::toString`` does."""
    if not math.isfinite(value):
        raise ValueError(f"non-finite number {value!r} is not representable in canonical JSON")
    if value == 0:
        return "0"  # 0.0 and -0.0 share the canonical form
    mantissa, _, exponent = repr(abs(value)).partition("e")
    exp = int(exponent) if exponent else 0
    whole, _, frac = mantissa.partition(".")
    digits = (whole + frac).lstrip("0").rstrip("0") or "0"
    # Position of the first significant digit relative to the decimal point.
    n = len(whole) + exp if whole != "0" else -(len(frac) - len(frac.lstrip("0")))
    k = len(digits)
    if k <= n <= 21:
        return ("-" if value < 0 else "") + digits + "0" * (n - k)
    if 0 < n <= 21:
        return ("-" if value < 0 else "") + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return ("-" if value < 0 else "") + "0." + "0" * -n + digits
    return (
        ("-" if value < 0 else "")
        + digits[0]
        + ("." + digits[1:] if k > 1 else "")
        + f"e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"
    )


def _serialize(value: Any) -> bytes:
    """Serialize one already-normalized JSON value to canonical bytes."""
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, str):
        # ensure_ascii=False emits raw UTF-8 (never \\u escapes); Python's
        # JSON escaping of control characters matches RFC 8785.
        return json.dumps(value, ensure_ascii=False).encode()
    if isinstance(value, int):
        if not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            raise ValueError(
                f"integer {value} exceeds the ECMAScript safe integer range"
            )
        return str(value).encode()
    if isinstance(value, float):
        return _es6_number(value).encode()
    if isinstance(value, list):
        # RFC 8785 preserves array order; callers wanting order-insensitivity
        # normalize arrays before hashing (see canonical_bundle_bytes).
        return b"[" + b",".join(_serialize(item) for item in value) + b"]"
    if isinstance(value, dict):
        return (
            b"{"
            + b",".join(
                _serialize(key) + b":" + _serialize(item)
                for key, item in sorted(value.items(), key=lambda kv: kv[0].encode("utf-16-be"))
            )
            + b"}"
        )
    raise TypeError(f"{type(value).__name__} is not representable in canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """RFC 8785 canonical JSON bytes for a JSON-compatible value."""
    return _serialize(value)


def _sort_collections(value: Any, field_name: str | None = None) -> Any:
    """Recursively normalize bundle arrays to deterministic total orders.

    Every array in a policy bundle is semantically a set (duplicate members
    are rejected at validation) except ``legs``, whose explicit unique
    ``order`` field defines precedence.
    """
    if isinstance(value, dict):
        return {
            key: _sort_collections(item, field_name=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        normalized = [_sort_collections(item) for item in value]
        if field_name == "legs":
            return sorted(normalized, key=lambda leg: (leg["order"], leg["leg_id"]))
        return sorted(normalized, key=_serialize)
    return value


def canonical_bundle_bytes(bundle: PolicyBundleV1) -> bytes:
    """Validate a full dump, then return order-independent canonical bytes."""
    dumped = bundle.model_dump(mode="python", round_trip=True, warnings=False)
    validated = PolicyBundleV1.model_validate(dumped)
    return canonical_json_bytes(_sort_collections(validated.model_dump(mode="json")))


def content_hash(payload: bytes, domain: bytes = _BUNDLE_DOMAIN) -> str:
    """Lowercase hex SHA-256 of ``domain + payload`` (domain-separated)."""
    if not domain.endswith(b"\x00"):
        raise ValueError("hash domain must be NUL-terminated for unambiguous framing")
    return hashlib.sha256(domain + payload).hexdigest()


def bundle_hash(payload: bytes) -> BundleHash:
    """Domain-separated content hash of canonical bundle bytes as a BundleHash."""
    return BundleHash.from_digest(content_hash(payload))


def emit_bundle_schema() -> bytes:
    """Deterministic JSON Schema bytes for PolicyBundleV1 (checked into schemas/)."""
    schema = PolicyBundleV1.model_json_schema()
    text = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode()


if __name__ == "__main__":  # regenerate: uv run python -m llmmaxxing.core.canonical
    sys.stdout.buffer.write(emit_bundle_schema())
