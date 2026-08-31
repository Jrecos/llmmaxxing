"""Auth-first bounded ingress and compact profiler integration contract."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from pydantic import ValidationError

from llmmaxxing.core.ids import RouteGroupId, RouteLegId
from llmmaxxing.core.key_material import compute_legacy_key_fingerprint
from llmmaxxing.core.models import (
    ClientKeyRecord,
    LegacyClientCredentialVerifier,
    PolicyBundleV1,
)
from llmmaxxing.core.reasons import Modality
from llmmaxxing.core.state_machines import CredentialVerifierStatus
from llmmaxxing.gateway.auth import (
    build_legacy_key_index,
    parse_client_key,
    verify_client_key,
)
from llmmaxxing.gateway.ingress import (
    DEFAULT_BODY_BYTES,
    DEFAULT_BODY_READERS_GLOBAL,
    DEFAULT_BODY_READERS_PER_KEY,
    DEFAULT_PREAUTH,
    DEFAULT_RETAINED_BYTES_GLOBAL,
    DEFAULT_RETAINED_BYTES_PER_KEY,
    IngressError,
    IngressLimits,
    IngressResources,
    RetainedBody,
    read_retained_body,
    validate_http_request,
)
from support.fake_litellm import FaultMode, FaultPlan
from support.gateway_stack import (
    CONTRACT,
    PEPPER,
    PEPPER_VERSION,
    AuthView,
    AuthViews,
    call_app,
    make_bundle,
    make_stack,
)


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def scope(
    path: str,
    body: bytes,
    *,
    content_type: bytes = b"application/json",
    headers: tuple[tuple[bytes, bytes], ...] = (),
) -> dict[str, object]:
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"host", b"gateway.invalid"),
            (b"content-type", content_type),
            (b"content-length", str(len(body)).encode()),
            *headers,
        ],
    }


def test_exact_seven_route_taxonomy_fails_closed_before_body(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(tmp_path, FaultPlan(FaultMode.JSON))
        try:
            exact = {(endpoint.method, endpoint.path) for endpoint in CONTRACT.endpoints}
            assert exact == {
                ("POST", "/v1/chat/completions"),
                ("POST", "/v1/completions"),
                ("POST", "/v1/embeddings"),
                ("POST", "/v1/rerank"),
                ("POST", "/v1/audio/speech"),
                ("POST", "/v1/audio/transcriptions"),
                ("POST", "/v1/images/generations"),
            }
            for path in (
                "/v1/responses",
                "/v1/responses/resp_1",
                "/v1/responses/resp_1/input_items",
                "/v1/messages",
                "/v1/messages/count_tokens",
                "/v1/realtime",
                "/v1/batches",
                "/v1/assistants",
                "/v1/chat/completions/children",
            ):
                response = await call_app(stack.app, None, path=path)
                assert response.status == 404
                assert response.receive_calls == 0
            wrong_method = await call_app(
                stack.app,
                stack.token,
                path="/v1/chat/completions",
                method="GET",
            )
            assert wrong_method.status == 405
            assert wrong_method.receive_calls == 0
            assert not stack.fake.calls
        finally:
            await stack.close()

    run(scenario())


def test_invalid_auth_and_lifecycle_backpressure_consume_zero_body_waiter_or_attempt(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        invalid = await make_stack(tmp_path / "invalid", FaultPlan(FaultMode.JSON))
        try:
            response = await call_app(invalid.app, "not-a-key")
            assert response.status == 401
            assert response.receive_calls == 0
            assert invalid.lifecycle_capacity.reservations == []
            assert invalid.ingress.retained_bytes == 0
            assert invalid.ingress.body_readers == 0
            assert not invalid.fake.calls
            assert invalid.runtime.operational_view().dispatches == ()
        finally:
            await invalid.close()

        unavailable = await make_stack(
            tmp_path / "lifecycle",
            FaultPlan(FaultMode.JSON),
            lifecycle_available=False,
        )
        try:
            response = await call_app(unavailable.app, unavailable.token)
            assert response.status == 503
            assert response.receive_calls == 0
            assert [
                reservation[2] for reservation in unavailable.lifecycle_capacity.reservations
            ] == [10]
            assert unavailable.ingress.retained_bytes == 0
            assert unavailable.ingress.body_readers == 0
            assert not unavailable.fake.calls
            assert unavailable.runtime.operational_view().dispatches == ()
        finally:
            await unavailable.close()

    run(scenario())


def test_lifecycle_capacity_is_reserved_before_body_and_body_is_released_once(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(tmp_path / "normal", FaultPlan(FaultMode.JSON))
        try:
            response = await call_app(stack.app, stack.token)
            assert response.status == 200
            assert [reservation[2] for reservation in stack.lifecycle_capacity.reservations] == [10]
            lifecycle = stack.lifecycle_capacity.lifecycles[0]
            assert lifecycle.released
            assert [name for name, _ in lifecycle.events] == [
                "profile",
                "queued",
                "attempt_started",
                "attempt_headers",
                "attempt_first_byte",
                "attempt_finished",
                "finished",
                "release",
            ]
            assert stack.ingress.retained_bytes == 0
            assert stack.ingress.retained_by_key == {}
            assert stack.ingress.body_readers == 0
            sent = json.loads(stack.fake.request_bodies[0])
            assert sent["model"] == "fixture/chat"
            assert sent["metadata"]["llmmaxxing_guard"]["generation_id"].startswith("dg1_")
            assert stack.fake.calls[0].headers["authorization"] == "Bearer backend-fixture"
        finally:
            await stack.close()

        shadow = await make_stack(
            tmp_path / "shadow",
            FaultPlan(FaultMode.JSON),
            FaultPlan(FaultMode.JSON),
            shadow=True,
        )
        try:
            response = await call_app(shadow.app, shadow.token)
            assert response.status == 200
            assert [reservation[2] for reservation in shadow.lifecycle_capacity.reservations] == [
                12
            ]
        finally:
            await shadow.close()

    run(scenario())


@pytest.mark.parametrize(
    ("headers", "status"),
    (
        (((b"content-length", b"1"),), 400),
        (((b"transfer-encoding", b"chunked"),), 400),
        (((b"content-encoding", b"gzip"),), 415),
        (((b"connection", b"upgrade"), (b"upgrade", b"websocket")), 426),
        (((b"expect", b"100-continue"),), 417),
        (((b"x-large", b"x" * 8193),), 431),
    ),
)
def test_framing_encoding_upgrade_and_header_limits_reject_before_body(
    tmp_path,
    headers: tuple[tuple[bytes, bytes], ...],
    status: int,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        stack = await make_stack(tmp_path, FaultPlan(FaultMode.JSON))
        try:
            response = await call_app(stack.app, stack.token, extra_headers=headers)
            assert response.status == status
            assert response.receive_calls == 0
            assert stack.ingress.retained_bytes == 0
            assert not stack.fake.calls
        finally:
            await stack.close()

    run(scenario())


def test_request_target_and_declared_body_limits_are_bounded_before_receive(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(tmp_path, FaultPlan(FaultMode.JSON))
        try:
            target = "/v1/chat/completions" + "x" * 8192
            response = await call_app(stack.app, stack.token, path=target)
            assert response.status == 414
            assert response.receive_calls == 0
            encoded = await call_app(
                stack.app,
                stack.token,
                raw_path=b"/v1/chat/%63ompletions",
            )
            assert encoded.status == 400
            assert encoded.receive_calls == 0
            queried = await call_app(stack.app, stack.token, query_string=b"api_key=forbidden")
            assert queried.status == 400
            assert queried.receive_calls == 0
            oversized = await call_app(
                stack.app,
                stack.token,
                extra_headers=((b"content-length", str(DEFAULT_BODY_BYTES + 1).encode()),),
            )
            assert oversized.status == 400  # duplicate framing is rejected before size
            assert oversized.receive_calls == 0
        finally:
            await stack.close()

    run(scenario())


def test_exact_ingress_caps_reader_partitions_and_retained_budgets() -> None:
    assert DEFAULT_PREAUTH == 64
    assert DEFAULT_BODY_READERS_GLOBAL == 16
    assert DEFAULT_BODY_READERS_PER_KEY == 4
    assert DEFAULT_BODY_BYTES == 32 * 1024 * 1024
    assert DEFAULT_RETAINED_BYTES_GLOBAL == 256 * 1024 * 1024
    assert DEFAULT_RETAINED_BYTES_PER_KEY == 64 * 1024 * 1024

    async def scenario() -> None:
        resources = IngressResources(
            IngressLimits(
                preauth=2,
                body_readers_global=2,
                body_readers_per_key=1,
                max_body_bytes=8,
                retained_bytes_global=12,
                retained_bytes_per_key=8,
                body_chunk_timeout_s=0.02,
                body_total_timeout_s=0.1,
            )
        )
        first = await resources.try_body_reader("a" * 32)
        assert first is not None
        assert await resources.try_body_reader("a" * 32) is None
        second = await resources.try_body_reader("b" * 32)
        assert second is not None
        assert await resources.try_body_reader("c" * 32) is None
        await first.release()
        await second.release()
        assert resources.body_readers == 0

        one = resources.new_body_reservation("a" * 32)
        assert await one.grow(8)
        two = resources.new_body_reservation("b" * 32)
        assert await two.grow(4)
        assert not await two.grow(1)
        assert not await one.grow(1)
        await one.release()
        await two.release()
        assert resources.retained_bytes == 0
        assert resources.retained_by_key == {}

    run(scenario())


def test_one_byte_slow_upload_and_cancellation_release_reader_and_retained_bytes() -> None:
    async def scenario() -> None:
        limits = IngressLimits(
            max_body_bytes=8,
            retained_bytes_global=16,
            retained_bytes_per_key=8,
            body_chunk_timeout_s=0.01,
            body_total_timeout_s=0.05,
        )
        resources = IngressResources(limits)
        request = validate_http_request(
            scope("/v1/chat/completions", b"xx"),
            CONTRACT,
            limits,
        )
        calls = 0

        async def slow_receive() -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": b"x", "more_body": True}
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        with pytest.raises(IngressError) as timeout:
            await read_retained_body(slow_receive, request, "a" * 32, resources)
        assert timeout.value.status == 408
        assert resources.retained_bytes == 0
        assert resources.body_readers == 0

        started = asyncio.Event()

        async def cancelled_receive() -> dict[str, object]:
            if not started.is_set():
                started.set()
                return {"type": "http.request", "body": b"x", "more_body": True}
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        task = asyncio.create_task(
            read_retained_body(cancelled_receive, request, "a" * 32, resources)
        )
        await started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert resources.retained_bytes == 0
        assert resources.body_readers == 0

    run(scenario())


def test_multipart_uses_quota_counted_mode_0600_spool_and_exact_release() -> None:
    async def scenario() -> None:
        boundary = "fixture-boundary"
        payload = b"a" * (1024 * 1024 + 1)
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
                "deepseek-v4-flash\r\n"
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        limits = IngressLimits(max_body_bytes=2 * 1024 * 1024)
        resources = IngressResources(limits)
        request = validate_http_request(
            scope(
                "/v1/audio/transcriptions",
                body,
                content_type=f"multipart/form-data; boundary={boundary}".encode(),
            ),
            CONTRACT,
            limits,
        )
        delivered = False

        async def receive() -> dict[str, object]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        retained = await read_retained_body(receive, request, "a" * 32, resources)
        assert retained.spooled_to_disk
        assert retained.spool_mode == 0o600
        assert resources.retained_bytes == len(body)
        assert await retained.read() == body
        await retained.release()
        await retained.release()
        assert resources.retained_bytes == 0
        assert resources.retained_by_key == {}

    run(scenario())


@pytest.mark.parametrize(
    "body",
    (
        json.dumps(
            {
                "model": "deepseek-v4-flash",
                "messages": [],
                "max_tokens": 1,
                "nested": "[DEPTH]",
            }
        ).encode(),
        json.dumps(
            {
                "model": "deepseek-v4-flash",
                "messages": [],
                "max_tokens": 1,
                "items": [0] * 100_001,
            }
        ).encode(),
        json.dumps(
            {
                "model": "deepseek-v4-flash",
                "messages": [],
                "max_tokens": 1,
                "value": "x" * (8 * 1024 * 1024 + 1),
            }
        ).encode(),
    ),
)
def test_profile_workers_enforce_depth_element_and_string_caps(tmp_path, body: bytes) -> None:
    if b"[DEPTH]" in body:
        nested: object = 0
        for _ in range(65):
            nested = [nested]
        document = json.loads(body)
        document["nested"] = nested
        body = json.dumps(document, separators=(",", ":")).encode()

    async def scenario() -> None:
        stack = await make_stack(tmp_path)
        try:
            response = await call_app(stack.app, stack.token, body=body)
            assert response.status == 422
            assert not stack.fake.calls
            assert stack.profiler.max_workers == 2
            assert stack.ingress.retained_bytes == 0
            assert stack.lifecycle_capacity.lifecycles[0].released
        finally:
            await stack.close()

    run(scenario())


def test_legacy_sk_hmac_index_authenticates_without_plaintext_in_signed_record() -> None:
    bundle, _ = make_bundle()
    token = "sk-" + "legacyfixture" * 8
    fingerprint = compute_legacy_key_fingerprint(PEPPER, token)
    canonical = bundle.keys[0]
    legacy = LegacyClientCredentialVerifier(
        generation=1,
        fingerprint_hex=fingerprint.hex(),
        pepper_version=PEPPER_VERSION,
        not_before_s=canonical.issued_at_s,
        not_after_s=canonical.expires_at_s,
        status=CredentialVerifierStatus.ACTIVE,
    )
    record = ClientKeyRecord.model_validate(
        {
            **canonical.model_dump(mode="python"),
            "credential_verifiers": (),
            "legacy_verifiers": (legacy,),
        }
    )
    legacy_bundle = PolicyBundleV1.model_validate(
        {**bundle.model_dump(mode="python"), "keys": (record,)}
    )
    runtime = AuthView(legacy_bundle)
    runtime.legacy_key_index = build_legacy_key_index(legacy_bundle.keys)

    client = verify_client_key(parse_client_key(token), AuthViews(runtime).current_auth_view())

    assert client.key_id == record.key_id
    assert client.accepted_credential_generation == 1
    serialized = json.dumps(record.model_dump(mode="json"), sort_keys=True)
    assert token not in serialized
    assert token not in repr(parse_client_key(token))
    assert len(runtime.legacy_key_index) == 1


def test_route_group_names_are_globally_unique_but_case_sensitive() -> None:
    bundle, _ = make_bundle()
    original = bundle.route_groups[0]
    second = original.model_copy(
        update={
            "route_group_id": RouteGroupId("rg_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            "legs": (
                original.legs[0].model_copy(
                    update={"leg_id": RouteLegId("leg_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")}
                ),
            ),
        }
    )
    with pytest.raises(ValidationError, match="duplicate RouteGroupRevision.name"):
        PolicyBundleV1.model_validate(
            {**bundle.model_dump(mode="python"), "route_groups": (original, second)}
        )

    distinct_case = second.model_copy(update={"name": original.name.swapcase()})
    validated = PolicyBundleV1.model_validate(
        {**bundle.model_dump(mode="python"), "route_groups": (original, distinct_case)}
    )
    assert [group.name for group in validated.route_groups] == [
        "deepseek-v4-flash",
        "DEEPSEEK-V4-FLASH",
    ]


def test_multipart_rewrite_strips_all_reserved_litellm_controls(tmp_path) -> None:
    async def scenario() -> None:
        boundary = "task8-controls"
        fields = (
            ("model", "deepseek-v4-flash"),
            ("language", "en"),
            ("num_retries", "99"),
            ("timeout", "9999"),
            ("api_base", "https://attacker.invalid"),
            ("custom_llm_provider", "attacker"),
        )
        chunks = [
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
            for name, value in fields
        ]
        chunks.append(
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + b"RIFF-fixture\r\n"
        )
        chunks.append(f"--{boundary}--\r\n".encode())
        body = b"".join(chunks)
        stack = await make_stack(tmp_path, FaultPlan(FaultMode.JSON))
        try:
            response = await call_app(
                stack.app,
                stack.token,
                path="/v1/audio/transcriptions",
                body=body,
                content_type=f"multipart/form-data; boundary={boundary}".encode(),
            )
            assert response.status == 200
            forwarded = stack.fake.request_bodies[0]
            assert b'name="language"' in forwarded
            for forbidden in (
                b'name="num_retries"',
                b'name="timeout"',
                b'name="api_base"',
                b'name="custom_llm_provider"',
            ):
                assert forbidden not in forwarded
            assert b"fixture/chat" in forwarded
            assert b"llmmaxxing_guard" in forwarded
        finally:
            await stack.close()

    run(scenario())


def test_attempt_event_failure_releases_dispatch_before_any_provider_send(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(
            tmp_path,
            FaultPlan(FaultMode.JSON),
            fail_attempt_started=True,
        )
        try:
            response = await call_app(stack.app, stack.token)
            assert response.status == 500
            assert not stack.fake.calls
            account = stack.runtime.operational_view().accounts[0]
            assert account.active_attempts == 0
            assert account.uncertain_attempts == 0
            assert stack.permits.snapshot().total_active == 0
            assert stack.lifecycle_capacity.lifecycles[0].released
        finally:
            await stack.close()

    run(scenario())


def test_lifecycle_capacity_releases_even_when_terminal_event_write_fails(tmp_path) -> None:
    async def scenario() -> None:
        stack = await make_stack(tmp_path, fail_finished=True)
        try:
            response = await call_app(stack.app, stack.token, body=b"{")
            assert response.status == 500
            assert stack.lifecycle_capacity.lifecycles[0].released
            assert stack.ingress.retained_bytes == 0
        finally:
            await stack.close()

    run(scenario())


def test_post_auth_deadline_covers_slow_body_and_releases_every_resource(tmp_path) -> None:
    async def scenario() -> None:
        limits = IngressLimits(
            body_chunk_timeout_s=1,
            body_total_timeout_s=10,
        )
        stack = await make_stack(tmp_path, limits=limits, deadline_ms=30)
        started = time.monotonic()
        try:
            response = await call_app(
                stack.app,
                stack.token,
                receive_messages=[{"type": "http.request", "body": b"{", "more_body": True}],
            )
            assert response.status == 504
            assert time.monotonic() - started < 0.5
            assert stack.ingress.retained_bytes == 0
            assert stack.ingress.body_readers == 0
            assert stack.lifecycle_capacity.lifecycles[0].released
        finally:
            await stack.close()

    run(scenario())


def test_retained_body_copy_and_prescan_do_not_block_the_event_loop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        original = RetainedBody._read_sync
        entered = threading.Event()

        def delayed_read(body: RetainedBody) -> bytes:
            entered.set()
            time.sleep(0.08)
            return original(body)

        monkeypatch.setattr(RetainedBody, "_read_sync", delayed_read)
        stack = await make_stack(tmp_path, FaultPlan(FaultMode.JSON))
        task = asyncio.create_task(call_app(stack.app, stack.token))
        try:
            while not entered.is_set():
                await asyncio.sleep(0)
            before = asyncio.get_running_loop().time()
            await asyncio.sleep(0.01)
            assert asyncio.get_running_loop().time() - before < 0.04
            assert (await task).status == 200
        finally:
            if not task.done():
                task.cancel()
            await stack.close()

    run(scenario())


def test_chat_profiler_derives_image_and_audio_and_rejects_mixed_nontext(tmp_path) -> None:
    def body(content: list[dict[str, object]]) -> bytes:
        return json.dumps(
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 8,
            },
            separators=(",", ":"),
        ).encode()

    async def scenario() -> None:
        stack = await make_stack(
            tmp_path,
            FaultPlan(FaultMode.JSON),
            FaultPlan(FaultMode.JSON),
        )
        try:
            image = await call_app(
                stack.app,
                stack.token,
                body=body(
                    [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ]
                ),
            )
            audio = await call_app(
                stack.app,
                stack.token,
                body=body(
                    [
                        {"type": "text", "text": "transcribe"},
                        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
                    ]
                ),
            )
            assert image.status == audio.status == 200
            profiles = [
                value
                for lifecycle in stack.lifecycle_capacity.lifecycles
                for name, value in lifecycle.events
                if name == "profile"
            ]
            assert [profile.modality for profile in profiles] == [
                Modality.IMAGE,
                Modality.AUDIO_TRANSCRIPTION,
            ]
            mixed = await call_app(
                stack.app,
                stack.token,
                body=body(
                    [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
                    ]
                ),
            )
            assert mixed.status == 422
            assert len(stack.fake.calls) == 2
        finally:
            await stack.close()

    run(scenario())


def test_cancelled_profile_offloads_hold_both_slots_until_work_really_finishes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        original = RetainedBody._read_sync
        release = threading.Event()
        two_entered = threading.Event()
        state_lock = threading.Lock()
        active = 0
        peak = 0

        def blocked_read(body: RetainedBody) -> bytes:
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    two_entered.set()
            release.wait(timeout=2)
            try:
                return original(body)
            finally:
                with state_lock:
                    active -= 1

        monkeypatch.setattr(RetainedBody, "_read_sync", blocked_read)
        stack = await make_stack(tmp_path, deadline_ms=30)
        tasks = [asyncio.create_task(call_app(stack.app, stack.token)) for _ in range(2)]
        try:
            assert await asyncio.to_thread(two_entered.wait, 1)
            await asyncio.sleep(0.05)
            tasks.append(asyncio.create_task(call_app(stack.app, stack.token)))
            await asyncio.sleep(0.06)
            with state_lock:
                assert peak == 2
                assert active == 2
            release.set()
            responses = await asyncio.gather(*tasks)
            assert [response.status for response in responses] == [504, 504, 504]
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await stack.close()

    run(scenario())
