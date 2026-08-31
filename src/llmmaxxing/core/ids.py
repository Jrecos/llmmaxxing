"""Typed, prefix-namespaced immutable identifiers.

Identity is content-independent: human-readable names never participate in
equality.  Every id is an immutable string carrying its own type prefix, so
values of two id types can never compare equal or be silently swapped.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, ClassVar, Self

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

_HEX64_PATTERN = r"[0-9a-f]{64}"
_UUID4_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"


class _TypedId(str):
    """Immutable ``prefix + body`` identifier.  Subclasses fix both parts."""

    __slots__ = ()

    _prefix: ClassVar[str] = ""
    _body_pattern: ClassVar[str] = ""

    def __new__(cls, value: object = ...) -> Self:
        text = str(value)
        if value is ...:
            text = f"{cls._prefix}{cls._fresh_body()}"
        if not re.fullmatch(rf"{cls._prefix}{cls._body_pattern}", text):
            raise ValueError(
                f"{text!r} is not a valid {cls.__name__} "
                f"(expected {cls._prefix}{cls._body_pattern})"
            )
        return super().__new__(cls, text)

    @classmethod
    def _fresh_body(cls) -> str:
        raise NotImplementedError

    @classmethod
    def new(cls) -> Self:
        """Mint a fresh, collision-resistant identifier."""
        return cls(...)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {
            "type": "string",
            "pattern": rf"^{cls._prefix}{cls._body_pattern}$",
            "title": cls.__name__,
        }


class _UuidId(_TypedId):
    """Identifier whose body is a UUID4."""

    __slots__ = ()
    _body_pattern = _UUID4_PATTERN

    @classmethod
    def _fresh_body(cls) -> str:
        return str(uuid.uuid4())


class _DigestId(_TypedId):
    """Identifier whose body is a lowercase SHA-256 hex digest."""

    __slots__ = ()
    _body_pattern = _HEX64_PATTERN

    @classmethod
    def _fresh_body(cls) -> str:
        raise ValueError(f"{cls.__name__} must be built from a digest via from_digest()")

    @classmethod
    def from_digest(cls, digest: str) -> Self:
        """Wrap a 64-char lowercase hex digest under this id's domain prefix."""
        return cls(f"{cls._prefix}{digest}")


class AccountId(_UuidId):
    """Immutable identity of one Provider Account (one upstream quota boundary)."""

    __slots__ = ()
    _prefix = "acc_"
class RequestId(_UuidId):
    """Immutable identity of one admitted client request."""

    __slots__ = ()
    _prefix = "req_"


class AttemptId(_UuidId):
    """Immutable identity of one possible provider send."""

    __slots__ = ()
    _prefix = "att_"


class GatewayBootId(_UuidId):
    """Identity of one Gateway process boot."""

    __slots__ = ()
    _prefix = "boot_"


class InstallationId(_UuidId):
    """Stable identity of one commissioned Gateway installation."""

    __slots__ = ()
    _prefix = "inst_"


class ProbeToken(_UuidId):
    """Unique durable token for one serialized capacity/circuit probe."""

    __slots__ = ()
    _prefix = "probe_"




class RouteGroupId(_UuidId):
    """Immutable identity of one client-visible model (Route Group)."""

    __slots__ = ()
    _prefix = "rg_"


class RouteLegId(_UuidId):
    """Immutable identity of one routing leg within a Route Group revision."""

    __slots__ = ()
    _prefix = "leg_"


class PolicyRevisionId(_UuidId):
    """Immutable identity of one complete key policy revision."""

    __slots__ = ()
    _prefix = "pol_"


class DeploymentGenerationId(_DigestId):
    """Semantic fingerprint of one LiteLLM deployment generation (``dg1_<sha256>``)."""

    __slots__ = ()
    _prefix = "dg1_"


class BundleHash(_DigestId):
    """Domain-separated SHA-256 of canonical bundle bytes (``bh_<sha256>``)."""

    __slots__ = ()
    _prefix = "bh_"
