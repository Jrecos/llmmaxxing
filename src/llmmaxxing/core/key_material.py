"""Pure client-key grammar and domain-separated verifier material."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
from dataclasses import dataclass, field

CLIENT_KEY_DOMAIN = b"llmmaxxing-client-key-v1\x00"
_KEY_ID_BYTES = 16
_SECRET_BYTES = 32
_TOKEN_RE = re.compile(r"^lmxk1\.([A-Za-z0-9_-]{22})\.([A-Za-z0-9_-]{43})$")


class InvalidClientKeyMaterial(ValueError):
    """The credential is not canonical ``lmxk1`` key material."""


@dataclass(frozen=True, slots=True)
class ParsedClientKey:
    key_id: str
    key_id_bytes: bytes = field(repr=False)
    secret: bytes = field(repr=False)

    def __repr__(self) -> str:
        return f"ParsedClientKey(key_id={self.key_id!r}, secret=<redacted>)"


def _decode_canonical(value: str, expected_bytes: int) -> bytes:
    try:
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise InvalidClientKeyMaterial("invalid client key") from error
    if len(decoded) != expected_bytes or _encode(decoded) != value:
        raise InvalidClientKeyMaterial("invalid client key")
    return decoded


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def parse_client_key_material(value: str) -> ParsedClientKey:
    match = _TOKEN_RE.fullmatch(value)
    if match is None:
        raise InvalidClientKeyMaterial("invalid client key")
    key_id_bytes = _decode_canonical(match.group(1), _KEY_ID_BYTES)
    secret = _decode_canonical(match.group(2), _SECRET_BYTES)
    return ParsedClientKey(key_id=key_id_bytes.hex(), key_id_bytes=key_id_bytes, secret=secret)


def format_client_key(key_id: bytes, secret: bytes) -> str:
    if len(key_id) != _KEY_ID_BYTES or len(secret) != _SECRET_BYTES:
        raise ValueError("client key requires 128-bit ID and 256-bit secret")
    return f"lmxk1.{_encode(key_id)}.{_encode(secret)}"


def compute_client_key_verifier(pepper: bytes, key_id: bytes, secret: bytes) -> bytes:
    if len(key_id) != _KEY_ID_BYTES or len(secret) != _SECRET_BYTES:
        raise ValueError("client key verifier requires 128-bit ID and 256-bit secret")
    return hmac.new(pepper, CLIENT_KEY_DOMAIN + key_id + secret, hashlib.sha256).digest()
