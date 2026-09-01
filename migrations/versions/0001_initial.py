"""Control schema v1: expand-only initial tables, guards and seeds.

Revision ID: 0001
Revises:
Create Date: 2026-09-01

Structural DDL is generated from ``llmmaxxing.storage.models.metadata`` so
runtime and schema can never drift; this revision adds only what a single
dialect can express (CHECK vocabularies, immutability triggers) and the
singleton seeds.  There is deliberately **no** downgrade: migrations are
expand-only and the CLI refuses anything but forward application.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from llmmaxxing.storage import models as m

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_KEY_STATES = ("draft", "enabled", "suspended", "revoked")
_STAGES = (
    "preparing_backend",
    "backend_ready",
    "staging_gateway",
    "gateway_staged",
    "committing",
    "applied",
)
_OUTBOX = ("ready", "sent", "acked", "dead")
_EVIDENCE = ("pending", "fresh", "revalidation_due", "failed", "invalidated")
_OUTCOMES = (
    "authz_denied",
    "auth_state_unavailable",
    "unsupported_request",
    "backpressure_rejected",
    "route_unavailable",
    "deadline_exceeded",
    "upstream_failed",
    "client_cancelled",
    "response_stream_failed",
    "completed",
)
_MODES = ("file", "sqlite", "postgres")
_VERIFY = ("pending", "verified", "failed")

_IMMUTABLE = ("bundles", "audit_log", "key_tombstones")

# (table, column) pairs that must be opaque hex64 digests.
_HEX64_COLUMNS = (
    ("keys", "record_digest"),
    ("key_tombstones", "terminal_record_digest"),
    ("identity_records", "identity_root_digest"),
    ("evidence", "artifact_digest"),
    ("audit_log", "record_digest"),
    ("audit_log", "prev_digest"),
    ("audit_checkpoints", "checkpoint_digest"),
    ("terminal_ledger", "tombstone_digest"),
    ("terminal_ledger", "prev_digest"),
    ("telemetry_checkpoints", "ingest_digest"),
)


def _in(values: tuple[str, ...]) -> str:
    joined = ", ".join(f"'{value}'" for value in values)
    return f"IN ({joined})"


def _hex64_condition(dialect: str, column: str) -> str:
    if dialect == "sqlite":
        return f"length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"
    return f"{column} ~ '^[0-9a-f]{{64}}$'"


def _checks(dialect: str) -> dict[str, list[tuple[str, str]]]:
    """Table -> (constraint name, condition) pairs for the dialect.

    SQLite cannot ``ALTER TABLE ... ADD CONSTRAINT``, so the pairs are applied
    via Alembic batch mode there; Postgres gets plain ALTERs.
    """
    byte_length = "length" if dialect == "sqlite" else "octet_length"
    checks: dict[str, list[tuple[str, str]]] = {
        "publications": [("ck_publications_singleton", "id = 1")],
        "bundles": [("ck_bundles_size", f"{byte_length}(canonical_bytes) <= 16777216")],
        "activations": [("ck_activations_stage", f"stage {_in(_STAGES)}")],
        "outbox": [("ck_outbox_status", f"status {_in(_OUTBOX)}")],
        "keys": [("ck_keys_state", f"state {_in(_KEY_STATES)}")],
        "evidence": [("ck_evidence_status", f"status {_in(_EVIDENCE)}")],
        "request_lifecycle": [
            ("ck_request_outcome", f"terminal_outcome {_in(_OUTCOMES)}")
        ],
        "backup_receipts": [
            ("ck_backup_mode", f"mode {_in(_MODES)}"),
            ("ck_backup_verify", f"verifier_status {_in(_VERIFY)}"),
        ],
    }
    if dialect == "sqlite":
        # Postgres BOOLEAN is already bool-typed; SQLite needs the guard.
        checks["activations"].append(
            ("ck_activations_outbox", "outbox_complete IN (0, 1)")
        )
    for table, column in _HEX64_COLUMNS:
        checks.setdefault(table, []).append(
            (f"ck_{table}_{column}_hex64", _hex64_condition(dialect, column))
        )
    return checks


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    inject_failure = bool(getattr(bind, "info", {}).get("inject_failure", False))

    m.metadata.create_all(bind=bind)

    if inject_failure:
        raise RuntimeError("injected migration failure")

    # CHECK constraints live in the dialect-appropriate form: batch mode on
    # SQLite (which cannot ALTER TABLE ADD CONSTRAINT), plain ALTERs on
    # Postgres.  Applied before the immutability triggers because recreating a
    # table drops its triggers.
    for table, pairs in _checks(dialect).items():
        if dialect == "sqlite":
            with op.batch_alter_table(table) as batch_op:
                for name, condition in pairs:
                    batch_op.create_check_constraint(name, sa.text(condition))
        else:
            for name, condition in pairs:
                op.execute(
                    f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({condition})"
                )

    # Immutability triggers: raw UPDATE/DELETE against content-addressed or
    # append-only tables is blocked inside the engine itself.
    if dialect == "sqlite":
        for table in _IMMUTABLE:
            op.execute(
                f"CREATE TRIGGER ux_{table}_immutable_update BEFORE UPDATE ON {table} "
                f"BEGIN SELECT RAISE(ABORT, 'immutable table: {table}'); END"
            )
            op.execute(
                f"CREATE TRIGGER ux_{table}_immutable_delete BEFORE DELETE ON {table} "
                f"BEGIN SELECT RAISE(ABORT, 'immutable table: {table}'); END"
            )
    else:
        op.execute(
            "CREATE FUNCTION llmmaxxing_block_mutation() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            "RAISE EXCEPTION 'immutable table: %', TG_TABLE_NAME; END; $$"
        )
        for table in _IMMUTABLE:
            op.execute(
                f"CREATE TRIGGER ux_{table}_immutable BEFORE UPDATE OR DELETE ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION llmmaxxing_block_mutation()"
            )

    # Seeds: singleton publication head plus immutable schema metadata.
    op.execute(
        "INSERT INTO publications (id, applied_generation, applied_bundle_hash, "
        "dispatcher_fence, updated_at_ms) VALUES (1, 0, '', 0, 0)"
    )
    schema_meta = sa.table(
        "schema_meta", sa.column("key", sa.Text), sa.column("value", sa.Text)
    )
    op.bulk_insert(
        schema_meta,
        [
            {"key": "min_reader_floor", "value": "0.1.0"},
            {"key": "last_migration_id", "value": "0001"},
            {"key": "backup_required_floor", "value": "0001"},
        ],
    )
