"""Frozen, metadata-only lifecycle event schema and canonical codec."""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmmaxxing.core.canonical import canonical_json_bytes
from llmmaxxing.core.ids import (
    AccountId,
    AttemptId,
    DeploymentGenerationId,
    EventId,
    GatewayBootId,
    InstallationId,
    RequestId,
    RouteGroupId,
)
from llmmaxxing.core.reasons import RouteTrigger, TerminalOutcome

MAX_EVENT_BYTES = 4096
_MAX_SAFE_INTEGER = (1 << 53) - 1
_KeyId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_Milliseconds = Annotated[int, Field(ge=0, le=_MAX_SAFE_INTEGER)]
_PositiveMilliseconds = Annotated[int, Field(ge=1, le=_MAX_SAFE_INTEGER)]
_Tokens = Annotated[int, Field(ge=0, le=_MAX_SAFE_INTEGER)]


class LifecycleEventKind(StrEnum):
    """Only the six V1 records covered by the fixed reservation formula."""

    REQUEST_ADMITTED = "request_admitted"
    REQUEST_QUEUED = "request_queued"
    REQUEST_ADMISSION_EVENT = "request_admission_event"
    REQUEST_TERMINAL = "request_terminal"
    ATTEMPT_RESERVED = "attempt_reserved"
    ATTEMPT_RESOLVED = "attempt_resolved"


class LifecycleReason(StrEnum):
    """Closed event reasons; raw provider/client strings never enter the spool."""

    AUTHZ_DENIED = "authz_denied"
    AUTH_STATE_UNAVAILABLE = "auth_state_unavailable"
    UNSUPPORTED_REQUEST = "unsupported_request"
    BACKPRESSURE_REJECTED = "backpressure_rejected"
    ROUTE_UNAVAILABLE = "route_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    UPSTREAM_FAILED = "upstream_failed"
    CLIENT_CANCELLED = "client_cancelled"
    RESPONSE_STREAM_FAILED = "response_stream_failed"
    COMPLETED = "completed"

    ACCOUNT_NOT_ACTIVE = "account_not_active"
    ACCOUNT_NOT_FOUND = "account_not_found"
    ATTEMPT_DUPLICATE = "attempt_duplicate"
    CIRCUIT_CHANGED = "circuit_changed"
    CIRCUIT_UNAVAILABLE = "circuit_unavailable"
    INVALID_QUOTA_CHARGE = "invalid_quota_charge"
    JOURNAL_CAPACITY_STOP = "journal_capacity_stop"
    JOURNAL_UNAVAILABLE = "journal_unavailable"
    MONTHLY_QUOTA_EXHAUSTED = "monthly_quota_exhausted"
    MONTHLY_RESET_UNAVAILABLE = "monthly_reset_unavailable"
    PARALLEL_EXHAUSTED = "parallel_exhausted"
    RECOVERY_REQUIRED = "recovery_required"
    RPM_EXHAUSTED = "rpm_exhausted"
    TPM_EXHAUSTED = "tpm_exhausted"

    CRASH_RECOVERY = "crash_recovery"
    UPSTREAM_UNKNOWN = "upstream_unknown"
    PROFILE_ACCEPTED = "profile_accepted"
    QUEUED_CONTRACTION = "queued_contraction"
    SPOOL_PRESSURE = "spool_pressure"
    PREDISPATCH_DEADLINE = "predispatch_deadline"
    TELEMETRY_SPOOL_EXHAUSTED = "telemetry_spool_exhausted"
    WRITER_UNAVAILABLE = "writer_unavailable"
    LIFECYCLE_INVARIANT = "lifecycle_invariant"

    @classmethod
    def from_outcome(cls, outcome: TerminalOutcome) -> Self:
        return cls(outcome.value)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class LifecycleTimingsV1(_Frozen):
    queue_wait_ms: _Milliseconds | None = None
    ttft_ms: _Milliseconds | None = None
    duration_ms: _Milliseconds | None = None

    @model_validator(mode="after")
    def _has_measurement(self) -> Self:
        if self.queue_wait_ms is None and self.ttft_ms is None and self.duration_ms is None:
            raise ValueError("at least one lifecycle timing is required")
        return self


class LifecycleTokenActualsV1(_Frozen):
    input: _Tokens | None = None
    output: _Tokens | None = None
    reasoning: _Tokens | None = None

    @model_validator(mode="after")
    def _has_measurement(self) -> Self:
        if self.input is None and self.output is None and self.reasoning is None:
            raise ValueError("at least one token actual is required")
        return self


_ATTEMPT_KINDS = frozenset(
    (LifecycleEventKind.ATTEMPT_RESERVED, LifecycleEventKind.ATTEMPT_RESOLVED)
)


class LifecycleEventV1(_Frozen):
    """One canonical, bounded lifecycle record safe for durable central storage."""

    schema_version: Literal[1] = 1
    event_id: EventId
    request_id: RequestId
    attempt_id: AttemptId | None = None
    kind: LifecycleEventKind
    occurred_at_ms: _PositiveMilliseconds
    installation_id: InstallationId
    boot_id: GatewayBootId
    key_id: _KeyId
    route_group_id: RouteGroupId | None = None
    account_id: AccountId | None = None
    deployment_generation_id: DeploymentGenerationId | None = None
    trigger: RouteTrigger | None = None
    outcome: TerminalOutcome | None = None
    reason: LifecycleReason | None = None
    timings_ms: LifecycleTimingsV1 | None = None
    token_actuals: LifecycleTokenActualsV1 | None = None
    spill_from_account_id: AccountId | None = None
    uncertain: bool | None = None
    headers_at_ms: _PositiveMilliseconds | None = None
    first_byte_at_ms: _PositiveMilliseconds | None = None
    final_byte_at_ms: _PositiveMilliseconds | None = None
    lease_released_at_ms: _PositiveMilliseconds | None = None
    attempts_used: Annotated[int, Field(ge=0, le=4)] | None = None

    @model_validator(mode="after")
    def _kind_fields_and_size(self) -> Self:
        is_attempt = self.kind in _ATTEMPT_KINDS
        attempt_identity = (
            self.attempt_id,
            self.account_id,
            self.deployment_generation_id,
            self.trigger,
        )
        attempt_observations = (
            self.token_actuals,
            self.spill_from_account_id,
            self.uncertain,
            self.headers_at_ms,
            self.first_byte_at_ms,
            self.final_byte_at_ms,
            self.lease_released_at_ms,
        )
        resolution_fields = (
            self.token_actuals,
            self.uncertain,
            self.headers_at_ms,
            self.first_byte_at_ms,
            self.final_byte_at_ms,
            self.lease_released_at_ms,
        )
        if is_attempt and any(value is None for value in attempt_identity):
            raise ValueError("attempt event requires complete attempt identity")
        if not is_attempt and any(value is not None for value in (*attempt_identity, *attempt_observations)):
            raise ValueError("request event forbids attempt fields")
        if self.kind is LifecycleEventKind.ATTEMPT_RESERVED:
            if self.outcome is not None or any(value is not None for value in resolution_fields):
                raise ValueError("attempt reservation forbids resolution fields")
        elif self.kind is LifecycleEventKind.ATTEMPT_RESOLVED:
            if self.outcome is None or self.uncertain is None:
                raise ValueError("attempt resolution requires outcome and certainty")
        elif self.kind is LifecycleEventKind.REQUEST_TERMINAL:
            if self.outcome is None or self.attempts_used is None:
                raise ValueError("request terminal requires outcome and attempts_used")
        elif self.outcome is not None or self.attempts_used is not None:
            raise ValueError("nonterminal request event forbids terminal fields")
        if len(canonical_event_bytes(self)) > MAX_EVENT_BYTES:
            raise ValueError(f"lifecycle event exceeds {MAX_EVENT_BYTES} bytes")
        return self


def canonical_event_bytes(event: LifecycleEventV1) -> bytes:
    """Return the sole JSON representation accepted by the spool and Task 11."""
    return canonical_json_bytes(
        event.model_dump(mode="json", exclude_none=True, round_trip=True, warnings=False)
    )


def load_event(payload: bytes) -> LifecycleEventV1:
    """Validate canonical bytes and reject alternate encodings before persistence."""
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid lifecycle event JSON") from error
    event = LifecycleEventV1.model_validate(document)
    if canonical_event_bytes(event) != payload:
        raise ValueError("lifecycle event JSON is not canonical")
    return event


def emit_event_schema() -> bytes:
    schema = LifecycleEventV1.model_json_schema()
    return (json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


if __name__ == "__main__":  # regenerate: uv run python -m llmmaxxing.telemetry.events
    sys.stdout.buffer.write(emit_event_schema())
