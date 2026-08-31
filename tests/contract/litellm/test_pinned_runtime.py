"""Opt-in contract harness for the controller-owned pinned fake-provider stack."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

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
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_fixture["name"]}"\r\n'
            ).encode(),
            f'Content-Type: {file_fixture["content_type"]}\r\n\r\n'.encode(),
            base64.b64decode(file_fixture["base64"]),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def test_pinned_build_complete_discovery_key_isolation_and_native_receipts() -> None:
    contract = load_contract()
    assert os.environ["LLMMAXXING_PINNED_IMAGE"] == contract.litellm.image
    assert os.environ["LLMMAXXING_PINNED_SOURCE_COMMIT"] == contract.litellm.source_commit
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

    selector_fixtures = json.loads((FIXTURES / "endpoint-selectors.json").read_text())["fixtures"]
    for endpoint in contract.endpoints:
        target = TARGETS[endpoint.name]
        deployment = by_alias[target["alias"]]
        assert deployment.runtime_id == target["deployment_id"]
        prepared = adapter.prepare_dispatch(
            endpoint=endpoint.name,
            deployment=deployment,
            generation=deployment_generation(deployment, contract),
            backend_manifest=os.environ["LLMMAXXING_PINNED_BACKEND_MANIFEST"],
        )
        selected = deepcopy(selector_fixtures[endpoint.name])
        if endpoint.model_locator == "json.model":
            body = selected["body"]
            body["model"] = prepared.hidden_alias
            fence_field = prepared.fence_locator.removeprefix("json.")
            body[fence_field] = prepared.trusted_metadata
            content_type = "application/json"
            raw = json.dumps(body, separators=(",", ":")).encode()
        else:
            fields = selected["fields"]
            fields["model"] = prepared.hidden_alias
            fence_field = prepared.fence_locator.removeprefix("multipart.")
            fields[fence_field] = json.dumps(
                prepared.trusted_metadata,
                separators=(",", ":"),
            )
            content_type, raw = _multipart(fields, selected["file"])
        status, headers, response_body = _http(
            prepared.method,
            prepared.path,
            INFERENCE_KEY,
            body=raw,
            content_type=content_type,
        )
        assert status == 200, (endpoint.name, status, response_body[:1000])
        receipt = adapter.reconcile_dispatch(
            prepared,
            status_code=status,
            headers=headers,
        )
        assert receipt.deployment_id == target["deployment_id"]
