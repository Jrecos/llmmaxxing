from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from llmmaxxing.core.ids import PolicyRevisionId
from llmmaxxing.core.models import ClientCredentialVerifier, ClientKeyRecord
from llmmaxxing.core.state_machines import CredentialVerifierStatus, KeyLifecycleState
from llmmaxxing.storage.backup import (
    audit_chain_digest,
    verify_audit_chain_from_checkpoint,
    verify_audit_chain_from_genesis,
)
from llmmaxxing.storage.migrations import migrate_database, sqlalchemy_async_url
from llmmaxxing.storage.models import (
    AuditCheckpoint,
    AuditLog,
    BackupReceipt,
    Bundle,
    IdentityRecord,
    KeyTombstone,
    OutboxItem,
    PublicationHead,
    RequestLifecycle,
    TelemetryCheckpoint,
)
from llmmaxxing.storage.postgres import PostgresStorage
from llmmaxxing.storage.repositories import ConflictError, StaleStateError, UnitOfWork
from llmmaxxing.storage.sqlite import SQLiteStorage

ZERO = "0" * 64
NOW_MS = 1_800_000_000_000
POSTGRES_DSN = os.environ.get("LLMMAXXING_TEST_POSTGRES")


@dataclass(slots=True)
class Store:
    dialect: str
    url: str
    storage: SQLiteStorage | PostgresStorage

    def uow(self) -> UnitOfWork:
        return self.storage.unit_of_work()


async def _reset_postgres(url: str) -> None:
    engine = create_async_engine(sqlalchemy_async_url(url))
    try:
        async with engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(params=("sqlite", "postgres"))
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    if request.param == "sqlite":
        path = tmp_path / "control.db"
        url = f"sqlite+aiosqlite:///{path}"
        await asyncio.to_thread(migrate_database, url)
        storage: SQLiteStorage | PostgresStorage = SQLiteStorage(path)
    else:
        if POSTGRES_DSN is None:
            pytest.skip("LLMMAXXING_TEST_POSTGRES is required for the Postgres contract")
        url = POSTGRES_DSN
        await _reset_postgres(url)
        await asyncio.to_thread(migrate_database, url)
        storage = PostgresStorage(url, installation_id="inst_repository_contract")
    harness = Store(str(request.param), url, storage)
    try:
        yield harness
    finally:
        await storage.dispose()


async def _raw_fails(url: str, statement: str, parameters: dict[str, Any]) -> None:
    engine = create_async_engine(sqlalchemy_async_url(url))
    try:
        with pytest.raises(DBAPIError):
            async with engine.begin() as connection:
                await connection.execute(text(statement), parameters)
    finally:
        await engine.dispose()


def _bundle(generation: int = 1) -> Bundle:
    return Bundle(
        generation=generation,
        bundle_hash=f"bh_{generation:062x}",
        canonical_bytes=b"{}",
        jcs_artifact=b"{}",
        min_reader="0.1",
        schema_version=1,
        created_at_ms=NOW_MS + generation,
    )


def _verifier(
    generation: int,
    *,
    status: CredentialVerifierStatus,
    not_before_s: int,
    not_after_s: int,
    digest: str | None = None,
) -> ClientCredentialVerifier:
    return ClientCredentialVerifier(
        generation=generation,
        verifier_hex=digest or f"{generation:064x}",
        pepper_version="v1",
        not_before_s=not_before_s,
        not_after_s=not_after_s,
        status=status,
    )


def _key(
    key_id: str,
    *,
    state: KeyLifecycleState = KeyLifecycleState.ENABLED,
    issued_at_s: int = 1_000_000,
    expires_at_s: int = 2_000_000,
    time_high_water_s: int = 1_100_000,
    verifiers: tuple[ClientCredentialVerifier, ...] | None = None,
) -> ClientKeyRecord:
    selected = verifiers or (
        _verifier(
            1,
            status=(
                CredentialVerifierStatus.RETIRED
                if state is KeyLifecycleState.REVOKED
                else CredentialVerifierStatus.ACTIVE
            ),
            not_before_s=issued_at_s,
            not_after_s=expires_at_s,
        ),
    )
    return ClientKeyRecord(
        key_id=key_id,
        policy_id=PolicyRevisionId.new(),
        state=state,
        issued_at_s=issued_at_s,
        expires_at_s=expires_at_s,
        time_high_water_s=time_high_water_s,
        generation_high_water=max(item.generation for item in selected),
        credential_verifiers=selected,
    )


@pytest.mark.asyncio
async def test_immutable_rows_and_identity_digest_cas(store: Store) -> None:
    bundle = _bundle()
    audit: AuditLog
    async with store.uow() as transaction:
        await transaction.publications.add_bundle(bundle)
        audit = await transaction.audit.append(
            occurred_at_ms=NOW_MS,
            actor_principal_id="principal",
            action="publish",
            subject_type="bundle",
            subject_id=bundle.bundle_hash,
            record_digest="a" * 64,
            prev_digest=ZERO,
        )
        await transaction.keys.append_tombstone(
            KeyTombstone(
                key_id="1" * 32,
                revoked_at_ms=NOW_MS,
                reason_class="operator",
                terminal_record_digest="b" * 64,
            )
        )
        await transaction.identities.add(
            IdentityRecord(
                installation_id="inst_contract",
                channel_key_epoch=1,
                security_epoch=1,
                identity_root_digest="c" * 64,
                created_at_ms=NOW_MS,
                updated_at_ms=NOW_MS,
            )
        )

    with pytest.raises(ConflictError):
        async with store.uow() as transaction:
            await transaction.publications.add_bundle(_bundle())

    for statement, parameters in (
        (
            "UPDATE bundles SET created_at_ms=:value WHERE generation=1",
            {"value": NOW_MS + 99},
        ),
        ("DELETE FROM bundles WHERE generation=1", {}),
        (
            "UPDATE audit_log SET action='forged' WHERE seq=:seq",
            {"seq": audit.seq},
        ),
        ("DELETE FROM audit_log WHERE seq=:seq", {"seq": audit.seq}),
        (
            "UPDATE key_tombstones SET reason_class='forged' WHERE key_id=:key_id",
            {"key_id": "1" * 32},
        ),
        ("DELETE FROM key_tombstones WHERE key_id=:key_id", {"key_id": "1" * 32}),
    ):
        await _raw_fails(store.url, statement, parameters)

    async with store.uow() as transaction:
        separately_loaded = await transaction.identities.get("inst_contract")
    assert separately_loaded is not None

    async with store.uow() as transaction:
        await transaction.identities.compare_and_swap(
            "inst_contract",
            separately_loaded.identity_root_digest,
            IdentityRecord(
                installation_id="inst_contract",
                channel_key_epoch=2,
                security_epoch=1,
                identity_root_digest="d" * 64,
                created_at_ms=NOW_MS,
                updated_at_ms=NOW_MS + 1,
            ),
        )

    with pytest.raises(StaleStateError):
        async with store.uow() as transaction:
            await transaction.identities.compare_and_swap(
                "inst_contract",
                separately_loaded.identity_root_digest,
                IdentityRecord(
                    installation_id="inst_contract",
                    channel_key_epoch=3,
                    security_epoch=1,
                    identity_root_digest="e" * 64,
                    created_at_ms=NOW_MS,
                    updated_at_ms=NOW_MS + 2,
                ),
            )


@pytest.mark.asyncio
async def test_head_cas_has_one_concurrent_winner_on_independent_connections(store: Store) -> None:
    async with store.uow() as transaction:
        expected = await transaction.publications.get_head()
    assert expected is not None

    ready = asyncio.Event()
    starters = 0
    starter_lock = asyncio.Lock()

    async def publish(generation: int) -> str:
        nonlocal starters
        async with starter_lock:
            starters += 1
            if starters == 2:
                ready.set()
        await ready.wait()
        try:
            async with store.uow() as transaction:
                await transaction.publications.cas_head(
                    expected,
                    PublicationHead(
                        id=1,
                        applied_generation=generation,
                        applied_bundle_hash=f"bh_{generation:062x}",
                        dispatcher_fence=generation,
                        updated_at_ms=NOW_MS + generation,
                    ),
                )
            return "won"
        except StaleStateError:
            return "stale"

    results = await asyncio.gather(publish(10), publish(11))
    assert sorted(results) == ["stale", "won"]
    async with store.uow() as transaction:
        head = await transaction.publications.get_head()
    assert head is not None
    assert head.applied_generation in (10, 11)

    if store.dialect == "sqlite":
        path = Path(store.url.removeprefix("sqlite+aiosqlite:///"))
        first = await aiosqlite.connect(path)
        second = await aiosqlite.connect(path)
        try:
            await first.execute("PRAGMA busy_timeout=50")
            await second.execute("PRAGMA busy_timeout=50")
            await first.execute("BEGIN IMMEDIATE")
            started = time.monotonic()
            with pytest.raises(aiosqlite.OperationalError, match="locked"):
                await second.execute("BEGIN IMMEDIATE")
            assert time.monotonic() - started < 1
        finally:
            await first.rollback()
            await first.close()
            await second.close()


@pytest.mark.asyncio
async def test_key_updates_run_irreversible_core_validation_before_digest_cas(store: Store) -> None:
    base = _key("2" * 32)
    async with store.uow() as transaction:
        await transaction.keys.add(base)

    terminal = _key("3" * 32, state=KeyLifecycleState.REVOKED)
    async with store.uow() as transaction:
        await transaction.keys.add(terminal)

    resurrected = terminal.model_copy(
        update={
            "state": KeyLifecycleState.ENABLED,
            "credential_verifiers": (
                terminal.credential_verifiers[0].model_copy(
                    update={"status": CredentialVerifierStatus.ACTIVE}
                ),
            ),
        }
    )
    expiry_extension = base.model_copy(update={"expires_at_s": base.expires_at_s + 1})

    old_two = _key(
        "4" * 32,
        verifiers=(
            _verifier(
                1,
                status=CredentialVerifierStatus.RETIRED,
                not_before_s=1_000_000,
                not_after_s=1_050_000,
            ),
            _verifier(
                2,
                status=CredentialVerifierStatus.ACTIVE,
                not_before_s=1_050_000,
                not_after_s=2_000_000,
            ),
        ),
    )
    async with store.uow() as transaction:
        await transaction.keys.add(old_two)
    high_water_rollback = _key(
        old_two.key_id,
        verifiers=(
            _verifier(
                1,
                status=CredentialVerifierStatus.ACTIVE,
                not_before_s=1_000_000,
                not_after_s=1_050_000,
            ),
        ),
    ).model_copy(update={"policy_id": old_two.policy_id})

    old_three = _key(
        "5" * 32,
        verifiers=(
            _verifier(
                1,
                status=CredentialVerifierStatus.RETIRED,
                not_before_s=1_000_000,
                not_after_s=1_010_000,
            ),
            _verifier(
                3,
                status=CredentialVerifierStatus.ACTIVE,
                not_before_s=1_010_000,
                not_after_s=2_000_000,
            ),
        ),
    )
    async with store.uow() as transaction:
        await transaction.keys.add(old_three)
    reused_generation = _key(
        old_three.key_id,
        verifiers=(
            _verifier(
                2,
                status=CredentialVerifierStatus.RETIRING,
                not_before_s=1_000_000,
                not_after_s=1_020_000,
                digest="9" * 64,
            ),
            old_three.credential_verifiers[1],
        ),
    ).model_copy(update={"policy_id": old_three.policy_id})

    overlap_before = _key("6" * 32)
    async with store.uow() as transaction:
        await transaction.keys.add(overlap_before)
    overlap_after = _key(
        overlap_before.key_id,
        time_high_water_s=1_100_000,
        verifiers=(
            overlap_before.credential_verifiers[0].model_copy(
                update={"status": CredentialVerifierStatus.RETIRING}
            ),
            _verifier(
                2,
                status=CredentialVerifierStatus.ACTIVE,
                not_before_s=1_100_000,
                not_after_s=2_000_000,
            ),
        ),
    ).model_copy(update={"policy_id": overlap_before.policy_id})

    for before, after, message in (
        (terminal, resurrected, "terminal"),
        (base, expiry_extension, "expiry"),
        (old_two, high_water_rollback, "high-water"),
        (old_three, reused_generation, "reuse"),
        (overlap_before, overlap_after, "seven days"),
    ):
        with pytest.raises(ValueError, match=message):
            async with store.uow() as transaction:
                await transaction.keys.update_record(before.key_id, before, after)

    async with store.uow() as transaction:
        assert await transaction.keys.get(base.key_id) == base
        assert await transaction.keys.get(terminal.key_id) == terminal


@pytest.mark.asyncio
async def test_outbox_audit_telemetry_and_backup_lifecycles_are_idempotent(store: Store) -> None:
    item = OutboxItem(
        op_idempotency_key="op-1",
        seq=1,
        command_json='{"command":"prepare"}',
        dedupe_key="prepare:1",
        status="ready",
        attempts=0,
        next_attempt_at_ms=NOW_MS,
        last_error_class=None,
        created_at_ms=NOW_MS,
        updated_at_ms=NOW_MS,
    )
    async with store.uow() as transaction:
        first = await transaction.publications.enqueue_outbox(item)
    async with store.uow() as transaction:
        replay = await transaction.publications.enqueue_outbox(
            OutboxItem(
                op_idempotency_key="op-1",
                seq=2,
                command_json=item.command_json,
                dedupe_key=item.dedupe_key,
                status="ready",
                attempts=0,
                next_attempt_at_ms=NOW_MS,
                last_error_class=None,
                created_at_ms=NOW_MS,
                updated_at_ms=NOW_MS,
            )
        )
    assert (first.op_idempotency_key, first.seq) == (replay.op_idempotency_key, replay.seq)

    first_record = "a" * 64
    second_record = "b" * 64
    first_chain = audit_chain_digest(ZERO, first_record)
    second_chain = audit_chain_digest(first_chain, second_record)
    async with store.uow() as transaction:
        first_audit = await transaction.audit.append(
            occurred_at_ms=NOW_MS + 1,
            actor_principal_id="principal",
            action="first",
            subject_type="test",
            subject_id="one",
            record_digest=first_record,
            prev_digest=ZERO,
        )
        second_audit = await transaction.audit.append(
            occurred_at_ms=NOW_MS + 2,
            actor_principal_id="principal",
            action="second",
            subject_type="test",
            subject_id="two",
            record_digest=second_record,
            prev_digest=first_chain,
        )
        checkpoint = AuditCheckpoint(
            checkpoint_seq=1,
            through_seq=second_audit.seq,
            checkpoint_digest=second_chain,
            signer_key_id="signer-1",
            trust_epoch=1,
            created_at_ms=NOW_MS + 3,
        )
        await transaction.audit.add_checkpoint(checkpoint)
        rows = await transaction.audit.list_entries(first_seq=first_audit.seq)
    assert verify_audit_chain_from_genesis(rows, genesis_digest=ZERO) == second_chain
    assert verify_audit_chain_from_checkpoint(rows, checkpoint, genesis_digest=ZERO) == ZERO

    forged = [
        AuditLog(
            seq=row.seq,
            occurred_at_ms=row.occurred_at_ms,
            actor_principal_id=row.actor_principal_id,
            action=row.action,
            subject_type=row.subject_type,
            subject_id=row.subject_id,
            record_digest=row.record_digest,
            prev_digest=("f" * 64 if row.seq == first_audit.seq else row.prev_digest),
        )
        for row in rows
    ]
    with pytest.raises(ValueError, match="audit chain"):
        verify_audit_chain_from_genesis(forged, genesis_digest=ZERO)
    with pytest.raises(ValueError, match="audit chain"):
        verify_audit_chain_from_checkpoint(forged, checkpoint, genesis_digest=ZERO)

    request = RequestLifecycle(
        request_id="req-1",
        admitted_at_ms=NOW_MS,
        key_id="2" * 32,
        route_group_id="rg-1",
        policy_revision="pol-1",
        terminal_outcome="succeeded",
        attempt_count=1,
        queue_wait_ms=5,
        ttft_ms=20,
        duration_ms=100,
        ingested_at_ms=NOW_MS + 100,
        source_checkpoint="telemetry-1",
    )
    telemetry_checkpoint = TelemetryCheckpoint(
        checkpoint_id="telemetry-1",
        first_seq=1,
        last_seq=1,
        event_count=1,
        ingest_digest="c" * 64,
        at_ms=NOW_MS + 100,
    )
    async with store.uow() as transaction:
        assert await transaction.telemetry.ingest(telemetry_checkpoint, [request]) is True
    async with store.uow() as transaction:
        assert await transaction.telemetry.ingest(telemetry_checkpoint, [request]) is False
        assert len(await transaction.telemetry.list_requests()) == 1
        assert len(await transaction.telemetry.list_checkpoints()) == 1

    receipt = BackupReceipt(
        receipt_id="backup-1",
        mode="sqlite" if store.dialect == "sqlite" else "postgres",
        scope_json='{"scope":"all"}',
        artifacts_json='{"sha256":"' + "d" * 64 + '"}',
        verifier_status="pending",
        created_at_ms=NOW_MS,
        verified_at_ms=None,
    )
    async with store.uow() as transaction:
        await transaction.backups.add(receipt)
    async with store.uow() as transaction:
        verified = await transaction.backups.mark_verified("backup-1", NOW_MS + 1)
        assert verified.verifier_status == "verified"
        assert await transaction.backups.latest_verified(created_after_ms=NOW_MS - 1) is not None
