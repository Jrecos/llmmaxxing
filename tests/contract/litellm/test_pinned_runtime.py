"""Opt-in contract harness for the controller-owned pinned fake-provider stack."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import socket
import ssl
import secrets
from collections.abc import Mapping
from typing import Any
from copy import deepcopy
from pathlib import Path

import pytest

from llmmaxxing.adapters.litellm.contract import TransportResponse, load_contract
from llmmaxxing.adapters.litellm.discovery import LiteLLMAdapter
from llmmaxxing.adapters.litellm.guard import deployment_generation

BASE_URL = os.environ.get("LLMMAXXING_PINNED_LITELLM_URL", "").rstrip("/")
if not BASE_URL:
    pytest.skip(
        "pinned LiteLLM container evidence pending controller execution",
        allow_module_level=True,
    )

DISCOVERY_KEY = os.environ["LLMMAXXING_PINNED_DISCOVERY_KEY"]
INFERENCE_KEY = os.environ["LLMMAXXING_PINNED_INFERENCE_KEY"]
TARGETS = json.loads(os.environ["LLMMAXXING_PINNED_ENDPOINT_TARGETS_JSON"])
PROVIDER_URL = os.environ["LLMMAXXING_PINNED_PROVIDER_URL"].rstrip("/")
SOURCE_FILES = json.loads(os.environ["LLMMAXXING_PINNED_SOURCE_FILES_JSON"])
EXPECT_SECRET_SWAP = os.environ.get("LLMMAXXING_PINNED_EXPECT_SECRET_SWAP") == "1"
FIXTURES = Path(__file__).parent / "fixtures"


def _http(
    method: str,
    path: str,
    key: str,
    *,
    query: Mapping[str, str] | None = None,
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    url = BASE_URL + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    headers = {"Authorization": f"Bearer {key}"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()

def _provider_count() -> int:
    with urllib.request.urlopen(PROVIDER_URL + "/counter", timeout=10) as response:
        payload = json.loads(response.read())
    return int(payload["provider_calls"])


def _websocket_status(path: str, key: str) -> int:
    parsed = urllib.parse.urlsplit(BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection = socket.create_connection((host, port), timeout=10)
    if parsed.scheme == "https":
        connection = ssl.create_default_context().wrap_socket(connection, server_hostname=host)
    websocket_key = base64.b64encode(secrets.token_bytes(16)).decode()
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {websocket_key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Authorization: Bearer {key}\r\n\r\n"
    )
    try:
        connection.sendall(request.encode())
        status_line = connection.recv(4096).split(b"\r\n", 1)[0]
    finally:
        connection.close()
    return int(status_line.split()[1])


class PinnedTransport:
    async def request(
        self,
        method: str,
        path: str,
        *,
        key: str,
        query: Mapping[str, str] | None = None,
    ) -> TransportResponse:
        assert key == "discovery"
        status, headers, raw = await asyncio.to_thread(
            _http,
            method,
            path,
            DISCOVERY_KEY,
            query=query,
        )
        return TransportResponse(
            status_code=status,
            headers=headers,
            body=json.loads(raw),
        )


def _multipart(fields: Mapping[str, str], file_fixture: Mapping[str, str]) -> tuple[str, bytes]:
    boundary = "llmmaxxing-certified-boundary"
    file_name = file_fixture["name"]
    file_content_type = file_fixture["content_type"]
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            )
        )
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode(),
            f"Content-Type: {file_content_type}\r\n\r\n".encode(),
            base64.b64decode(file_fixture["base64"]),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)

def _endpoint_request(
    prepared: Any,
    selected: dict[str, Any],
    trusted_metadata: dict[str, Any] | None,
) -> tuple[str, bytes]:
    selected = deepcopy(selected)
    if prepared.model_locator == "json.model":
        body = selected["body"]
        body["model"] = prepared.hidden_alias
        if trusted_metadata is not None:
            fence_field = prepared.fence_locator.removeprefix("json.")
            body[fence_field] = trusted_metadata
        return "application/json", json.dumps(body, separators=(",", ":")).encode()
    fields = selected["fields"]
    fields["model"] = prepared.hidden_alias
    if trusted_metadata is not None:
        fence_field = prepared.fence_locator.removeprefix("multipart.")
        fields[fence_field] = json.dumps(trusted_metadata, separators=(",", ":"))
    return _multipart(fields, selected["file"])


def test_pinned_build_complete_discovery_key_isolation_and_native_receipts() -> None:
    contract = load_contract()
    assert os.environ["LLMMAXXING_PINNED_IMAGE"] == contract.litellm.image
    assert SOURCE_FILES == contract.litellm.source_files
    assert set(TARGETS) == {endpoint.name for endpoint in contract.endpoints}

    adapter = LiteLLMAdapter(contract, PinnedTransport())
    snapshot = asyncio.run(adapter.discover_complete())
    by_alias = {deployment.hidden_alias: deployment for deployment in snapshot.deployments}

    status, _, _ = _http("GET", "/v2/model/info", INFERENCE_KEY)
    assert status == 403
    status, _, _ = _http(
        "POST",
        "/v1/chat/completions",
        DISCOVERY_KEY,
        body=b'{"model":"denied","messages":[]}',
        content_type="application/json",
    )
    assert status == 403
    status, _, _ = _http(
        "POST",
        "/model/update",
        DISCOVERY_KEY,
        body=b"{}",
        content_type="application/json",
    )
    assert status == 403
    for probe in contract.denial_probes:
        path = probe.path.replace("{response_id}", "resp_contract_fixture")
        if probe.protocol == "websocket":
            status = _websocket_status(path, INFERENCE_KEY)
        else:
            body = b"{}" if probe.method == "POST" else None
            status, _, _ = _http(
                probe.method,
                path,
                INFERENCE_KEY,
                body=body,
                content_type="application/json" if body is not None else None,
            )
        assert status == 403, (probe.protocol, probe.method, path, status)

    selector_fixtures = json.loads((FIXTURES / "endpoint-selectors.json").read_text())["fixtures"]
    for endpoint in contract.endpoints:
        target = TARGETS[endpoint.name]
        deployment = by_alias[target["alias"]]
        assert deployment.runtime_id == target["deployment_id"]
        prepared = adapter.prepare_dispatch(
            endpoint=endpoint.name,
            deployment=deployment,
            generation=deployment_generation(deployment, contract),
            backend_manifest=snapshot.manifest_revision,
        )
        selected = selector_fixtures[endpoint.name]
        before = _provider_count()
        content_type, raw = _endpoint_request(
            prepared,
            selected,
            prepared.trusted_metadata,
        )
        status, headers, response_body = _http(
            prepared.method,
            prepared.path,
            INFERENCE_KEY,
            body=raw,
            content_type=content_type,
        )
        if EXPECT_SECRET_SWAP:
            assert not 200 <= status < 300, (endpoint.name, status, response_body[:1000])
            assert _provider_count() == before
            continue

        content_type, missing_raw = _endpoint_request(prepared, selected, None)
        missing_status, _, _ = _http(
            prepared.method,
            prepared.path,
            INFERENCE_KEY,
            body=missing_raw,
            content_type=content_type,
        )
        assert not 200 <= missing_status < 300, (endpoint.name, missing_status)
        assert _provider_count() == before

        stale_metadata = deepcopy(prepared.trusted_metadata)
        stale_metadata["llmmaxxing_guard"]["backend_manifest"] = "bm1_" + "0" * 64
        content_type, stale_raw = _endpoint_request(prepared, selected, stale_metadata)
        stale_status, _, _ = _http(
            prepared.method,
            prepared.path,
            INFERENCE_KEY,
            body=stale_raw,
            content_type=content_type,
        )
        assert not 200 <= stale_status < 300, (endpoint.name, stale_status)
        assert _provider_count() == before

        assert status == 200, (endpoint.name, status, response_body[:1000])
        assert _provider_count() == before + 1
        receipt = adapter.reconcile_dispatch(
            prepared,
            status_code=status,
            headers=headers,
        )
        assert receipt.deployment_id == target["deployment_id"]
