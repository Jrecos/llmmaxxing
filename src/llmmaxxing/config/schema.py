"""Strict, secret-free V1 authoring models and deterministic schema emission."""

from __future__ import annotations

import json
import re
import sys
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llmmaxxing.core.canonical import canonical_json_bytes, content_hash
from llmmaxxing.core.ids import AccountId, PolicyRevisionId, RouteGroupId
from llmmaxxing.core.reasons import RouteTrigger

_NAME = Annotated[str, Field(min_length=1, max_length=120)]
_MACRO_NAME = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
_SOURCE_DOMAIN = b"llmmaxxing.authoring.v1\x00"


class _AuthoringModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class PolicyMacro(_AuthoringModel):
    """Reusable queue/deadline authoring defaults; never emitted to a bundle."""

    queue_tier: int = Field(ge=1)
    queue_weight: int = Field(ge=1, le=64)
    max_concurrency: int = Field(ge=1)
    max_waiters: int = Field(ge=0)
    deadline_ms: int = Field(ge=1, le=9_000_000)


class AuthoringPolicy(_AuthoringModel):
    """One new immutable policy revision, optionally cloned from an applied one."""

    policy_id: PolicyRevisionId
    name: _NAME | None = None
    clone_from_policy_id: PolicyRevisionId | None = None
    rebind_shared: bool = False
    macro: _MACRO_NAME | None = None
    route_group_ids: tuple[RouteGroupId, ...] | None = None
    account_ids: tuple[AccountId, ...] | None = None
    account_selector: dict[str, str] | None = None
    allowed_triggers: tuple[RouteTrigger, ...] | None = None
    queue_tier: int | None = Field(default=None, ge=1)
    queue_weight: int | None = Field(default=None, ge=1, le=64)
    max_concurrency: int | None = Field(default=None, ge=1)
    max_waiters: int | None = Field(default=None, ge=0)
    deadline_ms: int | None = Field(default=None, ge=1, le=9_000_000)

    @field_validator("route_group_ids", "account_ids", "allowed_triggers")
    @classmethod
    def _nonempty_unique_tuple(cls, value: tuple[object, ...] | None) -> tuple[object, ...] | None:
        if value is not None and not value:
            raise ValueError("explicit authoring membership cannot be empty")
        if value is not None and len(set(value)) != len(value):
            raise ValueError("duplicate authoring membership value")
        return value

    @field_validator("account_selector")
    @classmethod
    def _selector_is_bounded(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("account selector cannot be empty")
        if len(value) > 16:
            raise ValueError("account selector may contain at most 16 labels")
        for key, item in value.items():
            if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", key):
                raise ValueError(f"invalid account selector label {key!r}")
            if not item or len(item) > 120:
                raise ValueError(f"invalid account selector value for {key!r}")
        return value

    @model_validator(mode="after")
    def _authoring_mode_is_unambiguous(self) -> Self:
        if self.account_ids is not None and self.account_selector is not None:
            raise ValueError("use either exact account_ids or account_selector, never both")
        if self.rebind_shared and self.clone_from_policy_id is None:
            raise ValueError("rebind_shared requires clone_from_policy_id")
        if self.clone_from_policy_id == self.policy_id:
            raise ValueError("a policy revision cannot clone itself")
        return self


class AuthoringConfigV1(_AuthoringModel):
    """Secret-free authoring input applied as additions to one immutable base bundle."""

    schema_version: Literal[1] = 1
    macros: dict[_MACRO_NAME, PolicyMacro] = Field(default_factory=dict)
    policies: tuple[AuthoringPolicy, ...] = Field(min_length=1)

    @field_validator("policies")
    @classmethod
    def _policy_ids_are_unique(
        cls, value: tuple[AuthoringPolicy, ...]
    ) -> tuple[AuthoringPolicy, ...]:
        if len({policy.policy_id for policy in value}) != len(value):
            raise ValueError("duplicate authoring policy_id")
        return value

    @model_validator(mode="after")
    def _macros_exist(self) -> Self:
        missing = sorted(
            {policy.macro for policy in self.policies if policy.macro is not None}
            - self.macros.keys()
        )
        if missing:
            raise ValueError(f"unknown authoring macro(s): {', '.join(missing)}")
        return self


def authoring_fingerprint(config: AuthoringConfigV1) -> str:
    """Stable digest of the exact secret-free authoring source."""
    validated = AuthoringConfigV1.model_validate(config.model_dump(mode="python"))
    return content_hash(
        canonical_json_bytes(validated.model_dump(mode="json")),
        domain=_SOURCE_DOMAIN,
    )


def emit_authoring_schema() -> bytes:
    """Deterministic checked-in JSON Schema for :class:`AuthoringConfigV1`."""
    schema = AuthoringConfigV1.model_json_schema()
    return (json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


if __name__ == "__main__":  # regenerate: python -m llmmaxxing.config.schema
    sys.stdout.buffer.write(emit_authoring_schema())
