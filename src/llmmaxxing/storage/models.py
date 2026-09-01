"""SQLAlchemy Core schema for the Control database (schema_v1) plus row types.

``metadata`` is the single structural source of truth consumed by both the
runtime repositories and the Alembic ``0001`` migration, so the two can never
drift.  Engine-specific constraints (CHECK vocabularies, immutability
triggers) are applied by the migration itself and are intentionally absent
from the normalized comparison performed by ``tests/migration``.

Row dataclasses are plain value objects: repositories convert between them
and table rows on both engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy import MetaData

metadata = MetaData()


@dataclass(frozen=True, slots=True)
class Bundle:
    generation: int
    bundle_hash: str
    canonical_bytes: bytes
    jcs_artifact: bytes
    min_reader: str
    schema_version: int
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class PublicationHead:
    id: int
    applied_generation: int
    applied_bundle_hash: str
    dispatcher_fence: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class Activation:
    activation_id: str
    op_idempotency_key: str
    bundle_generation: int
    stage: str
    stage_record_digest: str
    base_generation: int
    impact_hash: str
    outbox_complete: bool
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class ActivationAcknowledgement:
    activation_id: str
    ack_kind: str
    boot_id: str
    acknowledged_at_ms: int
    receipt_digest: str


@dataclass(frozen=True, slots=True)
class OutboxItem:
    op_idempotency_key: str
    seq: int
    command_json: str
    dedupe_key: str
    status: str
    attempts: int
    next_attempt_at_ms: int
    last_error_class: str | None
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class KeyTombstone:
    key_id: str
    revoked_at_ms: int
    reason_class: str
    terminal_record_digest: str


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    installation_id: str
    channel_key_epoch: int
    security_epoch: int
    identity_root_digest: str
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    account_id: str
    generation_id: str
    artifact_digest: str
    suite_version: str
    assertions_json: str
    provenance: str
    expires_at_ms: int
    status: str


@dataclass(frozen=True, slots=True)
class AuditLog:
    seq: int
    occurred_at_ms: int
    actor_principal_id: str
    action: str
    subject_type: str
    subject_id: str
    record_digest: str
    prev_digest: str


@dataclass(frozen=True, slots=True)
class AuditCheckpoint:
    checkpoint_seq: int
    through_seq: int
    checkpoint_digest: str
    signer_key_id: str
    trust_epoch: int
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class TerminalLedgerEntry:
    seq: int
    subject_type: str
    subject_id: str
    tombstone_digest: str
    prev_digest: str
    at_ms: int


@dataclass(frozen=True, slots=True)
class RequestLifecycle:
    request_id: str
    admitted_at_ms: int
    key_id: str
    route_group_id: str
    policy_revision: str
    terminal_outcome: str
    attempt_count: int
    queue_wait_ms: int
    ttft_ms: int | None
    duration_ms: int | None
    ingested_at_ms: int
    source_checkpoint: str


@dataclass(frozen=True, slots=True)
class TelemetryCheckpoint:
    checkpoint_id: str
    first_seq: int
    last_seq: int
    event_count: int
    ingest_digest: str
    at_ms: int


@dataclass(frozen=True, slots=True)
class BackupReceipt:
    receipt_id: str
    mode: str
    scope_json: str
    artifacts_json: str
    verifier_status: str
    created_at_ms: int
    verified_at_ms: int | None


publications = sa.Table(
    "publications",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("applied_generation", sa.BigInteger, nullable=False),
    sa.Column("applied_bundle_hash", sa.Text, nullable=False),
    sa.Column("dispatcher_fence", sa.BigInteger, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
)

bundles = sa.Table(
    "bundles",
    metadata,
    sa.Column("generation", sa.BigInteger, primary_key=True),
    sa.Column("bundle_hash", sa.Text, nullable=False, unique=True),
    sa.Column("canonical_bytes", sa.LargeBinary, nullable=False),
    sa.Column("jcs_artifact", sa.LargeBinary, nullable=False),
    sa.Column("min_reader", sa.Text, nullable=False),
    sa.Column("schema_version", sa.BigInteger, nullable=False),
    sa.Column("created_at_ms", sa.BigInteger, nullable=False),
)

activations = sa.Table(
    "activations",
    metadata,
    sa.Column("activation_id", sa.Text, primary_key=True),
    sa.Column("op_idempotency_key", sa.Text, nullable=False, unique=True),
    sa.Column(
        "bundle_generation",
        sa.BigInteger,
        sa.ForeignKey("bundles.generation"),
        nullable=False,
    ),
    sa.Column("stage", sa.Text, nullable=False),
    sa.Column("stage_record_digest", sa.Text, nullable=False),
    sa.Column("base_generation", sa.BigInteger, nullable=False),
    sa.Column("impact_hash", sa.Text, nullable=False),
    sa.Column("outbox_complete", sa.Boolean, nullable=False),
    sa.Column("created_at_ms", sa.BigInteger, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
    sa.Index(
        "ux_activations_single_nonterminal",
        "stage",
        unique=True,
        sqlite_where=sa.text("stage != 'applied'"),
        postgresql_where=sa.text("stage != 'applied'"),
    ),
)

activation_acknowledgements = sa.Table(
    "activation_acknowledgements",
    metadata,
    sa.Column(
        "activation_id",
        sa.Text,
        sa.ForeignKey("activations.activation_id"),
        primary_key=True,
    ),
    sa.Column("ack_kind", sa.Text, primary_key=True),
    sa.Column("boot_id", sa.Text, primary_key=True),
    sa.Column("acknowledged_at_ms", sa.BigInteger, nullable=False),
    sa.Column("receipt_digest", sa.Text, nullable=False),
)

outbox = sa.Table(
    "outbox",
    metadata,
    sa.Column("op_idempotency_key", sa.Text, primary_key=True),
    sa.Column("seq", sa.BigInteger, primary_key=True),
    sa.Column("command_json", sa.Text, nullable=False),
    sa.Column("dedupe_key", sa.Text, nullable=False, unique=True),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("attempts", sa.BigInteger, nullable=False),
    sa.Column("next_attempt_at_ms", sa.BigInteger, nullable=False),
    sa.Column("last_error_class", sa.Text),
    sa.Column("created_at_ms", sa.BigInteger, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
    sa.Index("ix_outbox_status_next", "status", "next_attempt_at_ms"),
)

keys = sa.Table(
    "keys",
    metadata,
    sa.Column("key_id", sa.Text, primary_key=True),
    sa.Column("policy_revision", sa.Text, nullable=False),
    sa.Column("record_json", sa.Text, nullable=False),
    sa.Column("record_digest", sa.Text, nullable=False),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("issued_at_s", sa.BigInteger, nullable=False),
    sa.Column("expires_at_s", sa.BigInteger, nullable=False),
    sa.Column("time_high_water_s", sa.BigInteger, nullable=False),
    sa.Column("generation_high_water", sa.BigInteger, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
)

key_tombstones = sa.Table(
    "key_tombstones",
    metadata,
    sa.Column("key_id", sa.Text, primary_key=True),
    sa.Column("revoked_at_ms", sa.BigInteger, nullable=False),
    sa.Column("reason_class", sa.Text, nullable=False),
    sa.Column("terminal_record_digest", sa.Text, nullable=False),
)

policies = sa.Table(
    "policies",
    metadata,
    sa.Column("policy_id", sa.Text, primary_key=True),
    sa.Column("revision", sa.BigInteger, primary_key=True),
    sa.Column("current_record_digest", sa.Text, nullable=False),
    sa.Column("record_json", sa.Text, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
)

route_groups = sa.Table(
    "route_groups",
    metadata,
    sa.Column("route_group_id", sa.Text, primary_key=True),
    sa.Column("revision", sa.BigInteger, primary_key=True),
    sa.Column("current_record_digest", sa.Text, nullable=False),
    sa.Column("record_json", sa.Text, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
)

route_group_legs = sa.Table(
    "route_group_legs",
    metadata,
    sa.Column("leg_id", sa.Text, primary_key=True),
    sa.Column("route_group_id", sa.Text, nullable=False),
    sa.Column("revision", sa.BigInteger, nullable=False),
    sa.Column("order", sa.BigInteger, nullable=False),
    sa.Column("record_json", sa.Text, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
    sa.UniqueConstraint("route_group_id", "order", name="ux_route_group_legs_order"),
)

accounts = sa.Table(
    "accounts",
    metadata,
    sa.Column("account_id", sa.Text, primary_key=True),
    sa.Column("revision", sa.BigInteger, primary_key=True),
    sa.Column("current_record_digest", sa.Text, nullable=False),
    sa.Column("record_json", sa.Text, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
)

account_bindings = sa.Table(
    "account_bindings",
    metadata,
    sa.Column("binding_id", sa.Text, primary_key=True),
    sa.Column("account_id", sa.Text, nullable=False),
    sa.Column("provider_connection", sa.Text, nullable=False),
    sa.Column("provider_token", sa.Text, nullable=False),
    sa.Column("binding_ref", sa.Text, nullable=False),
    sa.Column("record_json", sa.Text, nullable=False),
    sa.Column("record_digest", sa.Text, nullable=False),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("tombstoned_at_ms", sa.BigInteger),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
    sa.UniqueConstraint(
        "provider_connection",
        "provider_token",
        "binding_ref",
        name="ux_account_bindings_ref",
    ),
)

identity_records = sa.Table(
    "identity_records",
    metadata,
    sa.Column("installation_id", sa.Text, primary_key=True),
    sa.Column("channel_key_epoch", sa.BigInteger, nullable=False),
    sa.Column("security_epoch", sa.BigInteger, nullable=False),
    sa.Column("identity_root_digest", sa.Text, nullable=False),
    sa.Column("created_at_ms", sa.BigInteger, nullable=False),
    sa.Column("updated_at_ms", sa.BigInteger, nullable=False),
)

evidence = sa.Table(
    "evidence",
    metadata,
    sa.Column("evidence_id", sa.Text, primary_key=True),
    sa.Column("account_id", sa.Text, nullable=False),
    sa.Column("generation_id", sa.Text, nullable=False),
    sa.Column("artifact_digest", sa.Text, nullable=False),
    sa.Column("suite_version", sa.Text, nullable=False),
    sa.Column("assertions_json", sa.Text, nullable=False),
    sa.Column("provenance", sa.Text, nullable=False),
    sa.Column("expires_at_ms", sa.BigInteger, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Index("ix_evidence_account", "account_id"),
    sa.Index("ix_evidence_generation", "generation_id"),
)

audit_log = sa.Table(
    "audit_log",
    metadata,
    sa.Column("seq", sa.Integer, primary_key=True),
    sa.Column("occurred_at_ms", sa.BigInteger, nullable=False),
    sa.Column("actor_principal_id", sa.Text, nullable=False),
    sa.Column("action", sa.Text, nullable=False),
    sa.Column("subject_type", sa.Text, nullable=False),
    sa.Column("subject_id", sa.Text, nullable=False),
    sa.Column("record_digest", sa.Text, nullable=False),
    sa.Column("prev_digest", sa.Text, nullable=False),
)

audit_checkpoints = sa.Table(
    "audit_checkpoints",
    metadata,
    sa.Column("checkpoint_seq", sa.Integer, primary_key=True),
    sa.Column(
        "through_seq",
        sa.Integer,
        sa.ForeignKey("audit_log.seq"),
        nullable=False,
    ),
    sa.Column("checkpoint_digest", sa.Text, nullable=False),
    sa.Column("signer_key_id", sa.Text, nullable=False),
    sa.Column("trust_epoch", sa.BigInteger, nullable=False),
    sa.Column("created_at_ms", sa.BigInteger, nullable=False),
)

terminal_ledger = sa.Table(
    "terminal_ledger",
    metadata,
    sa.Column("seq", sa.Integer, primary_key=True),
    sa.Column("subject_type", sa.Text, nullable=False),
    sa.Column("subject_id", sa.Text, nullable=False),
    sa.Column("tombstone_digest", sa.Text, nullable=False),
    sa.Column("prev_digest", sa.Text, nullable=False),
    sa.Column("at_ms", sa.BigInteger, nullable=False),
)

request_lifecycle = sa.Table(
    "request_lifecycle",
    metadata,
    sa.Column("request_id", sa.Text, primary_key=True),
    sa.Column("admitted_at_ms", sa.BigInteger, nullable=False),
    sa.Column("key_id", sa.Text, nullable=False),
    sa.Column("route_group_id", sa.Text, nullable=False),
    sa.Column("policy_revision", sa.Text, nullable=False),
    sa.Column("terminal_outcome", sa.Text, nullable=False),
    sa.Column("attempt_count", sa.BigInteger, nullable=False),
    sa.Column("queue_wait_ms", sa.BigInteger, nullable=False),
    sa.Column("ttft_ms", sa.BigInteger),
    sa.Column("duration_ms", sa.BigInteger),
    sa.Column("ingested_at_ms", sa.BigInteger, nullable=False),
    sa.Column("source_checkpoint", sa.Text, nullable=False),
    sa.Index("ix_request_lifecycle_outcome_admitted", "terminal_outcome", "admitted_at_ms"),
)

telemetry_checkpoints = sa.Table(
    "telemetry_checkpoints",
    metadata,
    sa.Column("checkpoint_id", sa.Text, primary_key=True),
    sa.Column("first_seq", sa.BigInteger, nullable=False),
    sa.Column("last_seq", sa.BigInteger, nullable=False),
    sa.Column("event_count", sa.BigInteger, nullable=False),
    sa.Column("ingest_digest", sa.Text, nullable=False),
    sa.Column("at_ms", sa.BigInteger, nullable=False),
)

backup_receipts = sa.Table(
    "backup_receipts",
    metadata,
    sa.Column("receipt_id", sa.Text, primary_key=True),
    sa.Column("mode", sa.Text, nullable=False),
    sa.Column("scope_json", sa.Text, nullable=False),
    sa.Column("artifacts_json", sa.Text, nullable=False),
    sa.Column("verifier_status", sa.Text, nullable=False),
    sa.Column("created_at_ms", sa.BigInteger, nullable=False),
    sa.Column("verified_at_ms", sa.BigInteger),
)

schema_meta = sa.Table(
    "schema_meta",
    metadata,
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("value", sa.Text, nullable=False),
)


def _values(instance: Any, table: sa.Table) -> dict[str, Any]:
    """Project a row dataclass onto the full column set of its table."""
    from dataclasses import fields as dc_fields

    names = {field.name for field in dc_fields(instance)}
    return {
        column.name: getattr(instance, column.name)
        for column in table.columns
        if column.name in names
    }


def bundle_row(bundle: Bundle) -> dict[str, Any]:
    return _values(bundle, bundles)


def activation_row(activation: Activation) -> dict[str, Any]:
    return _values(activation, activations)


def activation_update_row(activation: Activation) -> dict[str, Any]:
    values = _values(activation, activations)
    values.pop("activation_id")
    values.pop("op_idempotency_key")
    values.pop("created_at_ms")
    return values


def outbox_row(item: OutboxItem) -> dict[str, Any]:
    return _values(item, outbox)


def tombstone_row(tombstone: KeyTombstone) -> dict[str, Any]:
    return _values(tombstone, key_tombstones)


def identity_row(record: IdentityRecord) -> dict[str, Any]:
    return _values(record, identity_records)


def evidence_row(record: EvidenceRecord) -> dict[str, Any]:
    return _values(record, evidence)


def checkpoint_row(checkpoint: AuditCheckpoint) -> dict[str, Any]:
    return _values(checkpoint, audit_checkpoints)


def ledger_row(entry: TerminalLedgerEntry) -> dict[str, Any]:
    return _values(entry, terminal_ledger)


def request_row(record: RequestLifecycle) -> dict[str, Any]:
    return _values(record, request_lifecycle)


def telemetry_checkpoint_row(checkpoint: TelemetryCheckpoint) -> dict[str, Any]:
    return _values(checkpoint, telemetry_checkpoints)


def backup_row(receipt: BackupReceipt) -> dict[str, Any]:
    return _values(receipt, backup_receipts)
