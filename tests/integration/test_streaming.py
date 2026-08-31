"""Exact receipt-fenced raw relay, permit partitions, and uncertainty contract."""

from __future__ import annotations

import asyncio
import gzip

import pytest

from llmmaxxing.core.reasons import TerminalOutcome
from llmmaxxing.gateway.streaming import AttemptPermitPool, PermitClass
from support.fake_litellm import (
    FaultMode,
    FaultPlan,
    fixture_body,
    fixture_status,
)
from support.gateway_stack import call_app, make_stack


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def headers(response) -> dict[bytes, bytes]:  # type: ignore[no-untyped-def]
    return {name.lower(): value for name, value in response.headers}


def test_attempt_permits_are_144_with_hard_128_8_8_partitions_and_shadow_never_waits() -> None:
    async def scenario() -> None:
        pool = AttemptPermitPool()
        foreground = [await pool.acquire(PermitClass.FOREGROUND) for _ in range(128)]
        recovery = [await pool.acquire(PermitClass.RECOVERY) for _ in range(8)]
        discovery = [await pool.acquire(PermitClass.DISCOVERY) for _ in range(4)]
        qualification = [await pool.acquire(PermitClass.QUALIFICATION) for _ in range(4)]
        snapshot = pool.snapshot()
        assert snapshot.total_active == 144
        assert snapshot.foreground_active == 128
        assert snapshot.recovery_active == 8
        assert snapshot.discovery_qualification_active == 8
        assert await pool.try_shadow() is None

        blocked = asyncio.create_task(pool.acquire(PermitClass.FOREGROUND))
        await asyncio.sleep(0)
        assert not blocked.done()
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        assert pool.snapshot().foreground_waiters == 0

        await foreground.pop().release()
        shadow = await pool.try_shadow()
        assert shadow is not None
        assert shadow.shadow
        assert pool.snapshot().total_active == 144
        await shadow.release()
        await shadow.release()

        for lease in (*foreground, *recovery, *discovery, *qualification):
            await lease.release()
        assert pool.snapshot().total_active == 0
        assert pool.snapshot().total_waiters == 0

    run(scenario())


def test_process_http_client_has_exact_pool_security_and_timeout_contract(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(tmp_path, FaultPlan(FaultMode.JSON))
        try:
            config = stack.http.config
            assert config.max_connections == 160
            assert config.max_keepalive_connections == 160
            assert config.trust_env is False
            assert config.follow_redirects is False
            assert config.verify_tls is True
            assert config.retries == 0
            assert stack.http.client_count == 1
        finally:
            await stack.close()

    run(scenario())


@pytest.mark.parametrize(
    ("mode", "expected", "content_encoding"),
    (
        (FaultMode.SSE, b'data: {"token":"a"}\n\ndata: [DONE]\n\n', None),
        (FaultMode.UTF8_SPLIT, "uno \U0001f680 dos".encode(), None),
        (FaultMode.BINARY, bytes(range(256)), None),
        (
            FaultMode.COMPRESSED,
            gzip.compress(b"compressed-exact-bytes", mtime=0),
            b"gzip",
        ),
    ),
)
def test_sse_utf8_binary_and_compressed_responses_are_byte_exact_and_ordered(
    tmp_path,
    mode: FaultMode,
    expected: bytes,
    content_encoding: bytes | None,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        stack = await make_stack(tmp_path, FaultPlan(mode))
        try:
            response = await call_app(stack.app, stack.token)
            assert response.status == 200
            assert response.body == expected
            observed = headers(response)
            assert observed.get(b"content-encoding") == content_encoding
            start_index = next(
                index
                for index, message in enumerate(response.messages)
                if message["type"] == "http.response.start"
            )
            first_body_index = next(
                index
                for index, message in enumerate(response.messages)
                if message["type"] == "http.response.body" and message.get("body")
            )
            assert start_index < first_body_index
            assert stack.fake.streams[0].closed
            assert stack.ingress.retained_bytes == 0
            assert stack.permits.snapshot().total_active == 0
        finally:
            await stack.close()

    run(scenario())


def test_receipt_is_reconciled_before_response_start_and_mismatch_never_leaks_upstream_body(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        for mode in (FaultMode.MISSING_RECEIPT, FaultMode.WRONG_RECEIPT):
            stack = await make_stack(tmp_path / mode.value, FaultPlan(mode))
            try:
                response = await call_app(stack.app, stack.token)
                assert response.status == 502
                assert response.body != b'{"ok":true}'
                assert len(stack.fake.calls) == 1
                assert stack.fake.streams[0].next_calls == 0
                assert stack.ingress.retained_bytes == 0
                assert stack.permits.snapshot().total_active == 0
                lifecycle = stack.lifecycle_capacity.lifecycles[0]
                attempt = [event for event in lifecycle.events if event[0] == "attempt_finished"]
                assert len(attempt) == 1
            finally:
                await stack.close()

    run(scenario())


def test_downstream_backpressure_stops_upstream_reads_at_one_64k_chunk(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(tmp_path, FaultPlan(FaultMode.BACKPRESSURE))
        blocked = asyncio.Event()
        release = asyncio.Event()

        async def send_hook(message) -> None:  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.body" and message.get("body") and not blocked.is_set():
                blocked.set()
                await release.wait()

        task = asyncio.create_task(call_app(stack.app, stack.token, send_hook=send_hook))
        try:
            await asyncio.wait_for(blocked.wait(), timeout=1)
            assert stack.fake.streams[0].next_calls == 1
            assert stack.permits.snapshot().foreground_active == 1
            release.set()
            response = await task
            assert response.body == b"a" * 65_536 + b"b" * 65_536
            assert stack.fake.streams[0].next_calls == 2
            assert stack.permits.snapshot().total_active == 0
        finally:
            release.set()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await stack.close()

    run(scenario())


@pytest.mark.parametrize("mode", (FaultMode.RESET_AFTER_BYTE, FaultMode.STALL_AFTER_BYTE))
def test_post_byte_reset_or_stall_is_uncertain_and_never_replayed(
    tmp_path,
    mode: FaultMode,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        stack = await make_stack(
            tmp_path,
            FaultPlan(mode),
            FaultPlan(FaultMode.JSON),
        )
        try:
            response = await call_app(stack.app, stack.token)
            assert response.status == 200
            assert response.body == b"x"
            assert len(stack.fake.calls) == 1
            assert stack.fake.remaining == 1
            account = stack.runtime.operational_view().accounts[0]
            assert account.uncertain_attempts == 1
            assert account.active_attempts == 1
            assert stack.ingress.retained_bytes == 0
            assert stack.permits.snapshot().total_active == 0
            lifecycle = stack.lifecycle_capacity.lifecycles[0]
            attempt = [value for name, value in lifecycle.events if name == "attempt_finished"]
            assert len(attempt) == 1
            assert attempt[0][1] is TerminalOutcome.RESPONSE_STREAM_FAILED
            assert attempt[0][2] is True
            final = [value for name, value in lifecycle.events if name == "finished"]
            assert final == [TerminalOutcome.RESPONSE_STREAM_FAILED]
        finally:
            await stack.close()

    run(scenario())


def test_task_cancellation_after_first_byte_resolves_once_as_uncertain_without_replay(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        stack = await make_stack(
            tmp_path,
            FaultPlan(FaultMode.STALL_AFTER_BYTE),
            FaultPlan(FaultMode.JSON),
        )
        first_byte = asyncio.Event()

        async def observe(message) -> None:  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.body" and message.get("body"):
                first_byte.set()

        task = asyncio.create_task(call_app(stack.app, stack.token, send_hook=observe))
        try:
            await asyncio.wait_for(first_byte.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(stack.fake.calls) == 1
            assert stack.fake.remaining == 1
            account = stack.runtime.operational_view().accounts[0]
            assert account.uncertain_attempts == 1
            assert account.active_attempts == 1
            assert stack.fake.streams[0].closed
            assert stack.ingress.retained_bytes == 0
            assert stack.permits.snapshot().total_active == 0
            lifecycle = stack.lifecycle_capacity.lifecycles[0]
            attempts = [value for name, value in lifecycle.events if name == "attempt_finished"]
            assert len(attempts) == 1
            assert attempts[0][2] is True
            assert [value for name, value in lifecycle.events if name == "finished"] == [
                TerminalOutcome.CLIENT_CANCELLED
            ]
            assert lifecycle.released
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await stack.close()

    run(scenario())


def test_downstream_disconnect_closes_upstream_marks_uncertain_and_never_replays(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(
            tmp_path,
            FaultPlan(FaultMode.BACKPRESSURE),
            FaultPlan(FaultMode.JSON),
        )
        body_sends = 0

        async def disconnect(message) -> None:  # type: ignore[no-untyped-def]
            nonlocal body_sends
            if message["type"] == "http.response.body" and message.get("body"):
                body_sends += 1
                raise ConnectionError("downstream disconnected")

        try:
            response = await call_app(stack.app, stack.token, send_hook=disconnect)
            assert response.status == 200
            assert response.body == b""
            assert body_sends == 1
            assert len(stack.fake.calls) == 1
            assert stack.fake.remaining == 1
            assert stack.fake.streams[0].closed
            account = stack.runtime.operational_view().accounts[0]
            assert account.uncertain_attempts == 1
            assert stack.ingress.retained_bytes == 0
            assert stack.permits.snapshot().total_active == 0
        finally:
            await stack.close()

    run(scenario())


@pytest.mark.parametrize(
    ("mode", "fixture"),
    (
        (FaultMode.NAN_429, "nan_parallel_429"),
        (FaultMode.ARLIAI_403, "arliai_parallel_403"),
        (FaultMode.UPSTREAM_503, "upstream_unavailable_503"),
        (FaultMode.UNHANDLED_500, "unhandled_500"),
    ),
)
def test_pinned_normalized_error_envelopes_are_reusable_and_relayed_without_rewrite(
    tmp_path,
    mode: FaultMode,
    fixture: str,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        stack = await make_stack(tmp_path, FaultPlan(mode))
        try:
            response = await call_app(stack.app, stack.token)
            assert response.status == fixture_status(fixture)
            assert response.body == fixture_body(fixture)
            assert len(stack.fake.calls) == 1
            assert stack.ingress.retained_bytes == 0
            assert stack.permits.snapshot().total_active == 0
        finally:
            await stack.close()

    run(scenario())
