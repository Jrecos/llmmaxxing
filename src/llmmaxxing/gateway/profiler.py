"""Memory/CPU-bounded request profiling with no retained parse tree."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import multiprocessing
import re
import resource
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, cast

from llmmaxxing.adapters.litellm.contract import CertifiedEndpoint, PreparedDispatch
from llmmaxxing.core.ids import RouteGroupId
from llmmaxxing.core.models import RequestProfile
from llmmaxxing.core.reasons import EndpointKind, Modality
from llmmaxxing.gateway.ingress import DEFAULT_BODY_BYTES, RetainedBody

MAX_JSON_DEPTH = 64
MAX_JSON_ELEMENTS = 100_000
MAX_JSON_STRING_BYTES = 8 * 1024 * 1024
MAX_MULTIPART_PARTS = 1_000
MAX_MULTIPART_PART_HEADERS = 16 * 1024
MAX_TOOLS = 128
PROFILE_WORKERS = 2
PROFILE_ADDRESS_SPACE_BYTES = 768 * 1024 * 1024
PROFILE_CPU_SECONDS = 15
PROFILE_WALL_SECONDS = 20.0
SCRATCH_BYTES_GLOBAL = 128 * 1024 * 1024
SCRATCH_BYTES_PER_KEY = 64 * 1024 * 1024

_BOUNDARY = re.compile(
    r"(?:^|;)\s*boundary=(?:\"([0-9A-Za-z'()+_,./:=? -]{1,70})\"|([0-9A-Za-z'()+_,./:=?-]{1,70}))"
)
_NAME = re.compile(rb'(?:^|;)\s*name="([^"]{1,128})"(?:;|$)')


class ProfileError(ValueError):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class _ProfileResult:
    model_alias: str
    endpoint: EndpointKind
    modality: Modality
    stream: bool
    input_tokens_max: int
    output_tokens_max: int
    reasoning_tokens_max: int
    tools_count: int
    forced_tool_required: bool
    response_schema_present: bool
    history_turns: int


@dataclass(frozen=True, slots=True)
class _MultipartPart:
    header_blob: bytes
    name: str
    content: bytes


def _limit_profile_worker() -> None:
    resource.setrlimit(
        resource.RLIMIT_AS,
        (PROFILE_ADDRESS_SPACE_BYTES, PROFILE_ADDRESS_SPACE_BYTES),
    )
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (PROFILE_CPU_SECONDS, PROFILE_CPU_SECONDS),
    )


def _reject(code: str) -> NoReturn:
    raise ValueError(code)


def _pairs_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _reject("duplicate_json_key")
        result[key] = value
    return result

def _prescan_json_limits(body: bytes) -> None:
    """Bound nesting/tokens/strings without constructing attacker-controlled objects."""
    index = 0
    depth = 0
    elements = 0
    length = len(body)
    whitespace = b" \t\r\n"
    delimiters = b" \t\r\n,]}:"
    while index < length:
        byte = body[index]
        if byte in whitespace or byte in b",:":
            index += 1
            continue
        if byte in (ord("{"), ord("[")):
            depth += 1
            elements += 1
            if depth > MAX_JSON_DEPTH:
                _reject("json_depth_limit")
            index += 1
        elif byte in (ord("}"), ord("]")):
            depth -= 1
            index += 1
        elif byte == ord('"'):
            index += 1
            start = index
            escaped = False
            while index < length:
                current = body[index]
                if current == ord('"') and not escaped:
                    break
                escaped = current == ord("\\") and not escaped
                if current != ord("\\"):
                    escaped = False
                index += 1
                if index - start > MAX_JSON_STRING_BYTES:
                    _reject("json_string_limit")
            elements += 1
            index += 1
        else:
            elements += 1
            while index < length and body[index] not in delimiters:
                index += 1
        if elements > MAX_JSON_ELEMENTS:
            _reject("json_element_limit")



def _parse_json(body: bytes) -> dict[str, Any]:
    try:
        decoded = body.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda _value: _reject("invalid_json_number"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        _reject("malformed_json")
    if not isinstance(value, dict):
        _reject("json_object_required")
    _validate_tree(value)
    return cast(dict[str, Any], value)


def _validate_tree(root: object) -> None:
    stack: list[tuple[object, int]] = [(root, 1)]
    elements = 0
    while stack:
        value, depth = stack.pop()
        elements += 1
        if elements > MAX_JSON_ELEMENTS:
            _reject("json_element_limit")
        if depth > MAX_JSON_DEPTH:
            _reject("json_depth_limit")
        if isinstance(value, str):
            if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
                _reject("json_string_limit")
        elif isinstance(value, dict):
            for key, item in value.items():
                if len(key.encode("utf-8")) > MAX_JSON_STRING_BYTES:
                    _reject("json_string_limit")
                elements += 1
                if elements > MAX_JSON_ELEMENTS:
                    _reject("json_element_limit")
                stack.append((item, depth + 1))
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        elif value is not None and not isinstance(value, (bool, int, float)):
            _reject("unsupported_json_value")


def _positive_int(value: object, code: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        _reject(code)
    return value


def _model(data: Mapping[str, Any]) -> str:
    value = data.get("model")
    if not isinstance(value, str) or not 1 <= len(value) <= 160:
        _reject("invalid_model")
    return value


def _stream(data: Mapping[str, Any]) -> bool:
    value = data.get("stream", False)
    if not isinstance(value, bool):
        _reject("invalid_stream")
    return value


def _tools(data: Mapping[str, Any]) -> tuple[int, bool]:
    tools = data.get("tools", [])
    functions = data.get("functions", [])
    if not isinstance(tools, list) or not isinstance(functions, list):
        _reject("invalid_tools")
    count = len(tools) + len(functions)
    if count > MAX_TOOLS:
        _reject("tool_limit")
    choice = data.get("tool_choice", data.get("function_call"))
    forced = isinstance(choice, dict) or (
        isinstance(choice, str) and choice not in ("", "auto", "none")
    )
    return count, forced


def _schema(data: Mapping[str, Any]) -> bool:
    response_format = data.get("response_format")
    if response_format is None:
        return False
    if not isinstance(response_format, dict):
        _reject("invalid_response_format")
    return response_format.get("type") == "json_schema" or "json_schema" in response_format


def _reasoning(data: Mapping[str, Any], output_tokens: int) -> int:
    direct = data.get("max_reasoning_tokens")
    if direct is not None:
        return _positive_int(direct, "invalid_reasoning_budget", zero=True)
    reasoning = data.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            _reject("invalid_reasoning_budget")
        nested = reasoning.get("max_tokens")
        if nested is not None:
            return _positive_int(nested, "invalid_reasoning_budget", zero=True)
    return output_tokens if data.get("reasoning_effort") is not None else 0


def _profile_json(endpoint_name: str, body: bytes) -> _ProfileResult:
    data = _parse_json(body)
    alias = _model(data)
    endpoint = EndpointKind(endpoint_name)
    stream = _stream(data)
    tools_count = 0
    forced_tool = False
    response_schema = False
    history_turns = 0
    output_tokens = 0
    reasoning_tokens = 0

    if endpoint is EndpointKind.CHAT:
        messages = data.get("messages")
        if not isinstance(messages, list):
            _reject("invalid_messages")
        selected = (
            data["max_completion_tokens"]
            if "max_completion_tokens" in data
            else data.get("max_tokens")
        )
        output_tokens = _positive_int(selected, "missing_output_budget")
        reasoning_tokens = _reasoning(data, output_tokens)
        tools_count, forced_tool = _tools(data)
        response_schema = _schema(data)
        history_turns = sum(
            isinstance(message, dict)
            and message.get("role") in ("assistant", "tool", "function")
            for message in messages
        )
        modality = Modality.TEXT
    elif endpoint is EndpointKind.TEXT:
        if "prompt" not in data:
            _reject("invalid_prompt")
        output_tokens = _positive_int(data.get("max_tokens"), "missing_output_budget")
        reasoning_tokens = _reasoning(data, output_tokens)
        modality = Modality.TEXT
    elif endpoint is EndpointKind.EMBEDDINGS:
        if "input" not in data:
            _reject("invalid_embedding_input")
        if stream:
            _reject("stream_not_supported")
        modality = Modality.EMBEDDING
    elif endpoint is EndpointKind.RERANK:
        if "query" not in data or not isinstance(data.get("documents"), list):
            _reject("invalid_rerank_input")
        if stream:
            _reject("stream_not_supported")
        modality = Modality.RERANK
    elif endpoint is EndpointKind.AUDIO_SPEECH:
        if not isinstance(data.get("input"), str):
            _reject("invalid_audio_input")
        modality = Modality.AUDIO_SPEECH
    elif endpoint is EndpointKind.IMAGE:
        if not isinstance(data.get("prompt"), str):
            _reject("invalid_image_input")
        modality = Modality.IMAGE
    else:
        _reject("unsupported_json_endpoint")
    return _ProfileResult(
        alias,
        endpoint,
        modality,
        stream,
        len(body),
        output_tokens,
        reasoning_tokens,
        tools_count,
        forced_tool,
        response_schema,
        int(history_turns),
    )


def _multipart_boundary(content_type: str) -> bytes:
    match = _BOUNDARY.search(content_type)
    value = (match.group(1) or match.group(2)) if match is not None else None
    if value is None:
        _reject("multipart_boundary_missing")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        _reject("multipart_boundary_invalid")
    return encoded


def _multipart_parts(body: bytes, content_type: str) -> tuple[bytes, tuple[_MultipartPart, ...]]:
    boundary = _multipart_boundary(content_type)
    delimiter = b"--" + boundary
    segments = body.split(delimiter)
    if not segments or segments[0] != b"" or segments[-1] not in (b"--", b"--\r\n"):
        _reject("malformed_multipart")
    raw_parts = segments[1:-1]
    if not 1 <= len(raw_parts) <= MAX_MULTIPART_PARTS:
        _reject("multipart_part_limit")
    parts: list[_MultipartPart] = []
    names: set[str] = set()
    for raw_part in raw_parts:
        if not raw_part.startswith(b"\r\n") or not raw_part.endswith(b"\r\n"):
            _reject("malformed_multipart")
        header_blob, separator, content = raw_part[2:-2].partition(b"\r\n\r\n")
        if not separator or len(header_blob) > MAX_MULTIPART_PART_HEADERS:
            _reject("multipart_header_limit")
        header_lines = header_blob.split(b"\r\n")
        if len(header_lines) > 32:
            _reject("multipart_header_limit")
        disposition = next(
            (
                line.split(b":", 1)[1].strip()
                for line in header_lines
                if line.lower().startswith(b"content-disposition:") and b":" in line
            ),
            None,
        )
        match = _NAME.search(disposition) if disposition is not None else None
        if match is None:
            _reject("multipart_name_missing")
        try:
            name = match.group(1).decode("ascii")
        except UnicodeDecodeError:
            _reject("multipart_name_invalid")
        if name in names:
            _reject("duplicate_multipart_field")
        names.add(name)
        if b"filename=" not in (disposition or b""):
            if len(content) > MAX_JSON_STRING_BYTES:
                _reject("multipart_field_limit")
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                _reject("multipart_field_encoding")
        parts.append(_MultipartPart(header_blob, name, content))
    return boundary, tuple(parts)


def _profile_multipart(endpoint_name: str, body: bytes, content_type: str) -> _ProfileResult:
    _, parts = _multipart_parts(body, content_type)
    values = {part.name: part.content for part in parts if b"filename=" not in part.header_blob}
    model_raw = values.get("model")
    try:
        alias = model_raw.decode("utf-8") if model_raw is not None else ""
    except UnicodeDecodeError:
        _reject("invalid_model")
    if not 1 <= len(alias) <= 160 or "file" not in {part.name for part in parts}:
        _reject("invalid_audio_transcription")
    endpoint = EndpointKind(endpoint_name)
    if endpoint is not EndpointKind.AUDIO_TRANSCRIPTION:
        _reject("unsupported_multipart_endpoint")
    return _ProfileResult(
        alias,
        endpoint,
        Modality.AUDIO_TRANSCRIPTION,
        False,
        len(body),
        0,
        0,
        0,
        False,
        False,
        0,
    )


def _profile_worker(endpoint_name: str, body: bytes, content_type: str) -> _ProfileResult:
    if len(body) > DEFAULT_BODY_BYTES:
        _reject("body_too_large")
    if content_type.split(";", 1)[0].strip().lower() == "multipart/form-data":
        return _profile_multipart(endpoint_name, body, content_type)
    return _profile_json(endpoint_name, body)


def _rewrite_json(
    body: bytes,
    prepared: PreparedDispatch,
    known_litellm_params: tuple[str, ...],
) -> bytes:
    data = _parse_json(body)
    for field in known_litellm_params:
        if field != "model":
            data.pop(field, None)
    data["model"] = prepared.hidden_alias
    field = prepared.fence_locator.removeprefix("json.")
    existing = data.get(field)
    if existing is not None and not isinstance(existing, dict):
        _reject("invalid_metadata")
    metadata = dict(existing or {})
    metadata["llmmaxxing_guard"] = prepared.trusted_metadata["llmmaxxing_guard"]
    data[field] = metadata
    rewritten = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(rewritten) > DEFAULT_BODY_BYTES:
        _reject("rewritten_body_too_large")
    return rewritten


def _rewrite_multipart(body: bytes, content_type: str, prepared: PreparedDispatch) -> bytes:
    boundary, parts = _multipart_parts(body, content_type)
    model_field = prepared.model_locator.removeprefix("multipart.")
    fence_field = prepared.fence_locator.removeprefix("multipart.")
    rewritten: list[_MultipartPart] = []
    fence_value = json.dumps(prepared.trusted_metadata, separators=(",", ":")).encode()
    seen_fence = False
    for part in parts:
        if part.name == model_field:
            rewritten.append(_MultipartPart(part.header_blob, part.name, prepared.hidden_alias.encode()))
        elif part.name == fence_field:
            rewritten.append(_MultipartPart(part.header_blob, part.name, fence_value))
            seen_fence = True
        else:
            rewritten.append(part)
    if not seen_fence:
        rewritten.append(
            _MultipartPart(
                f'Content-Disposition: form-data; name="{fence_field}"'.encode(),
                fence_field,
                fence_value,
            )
        )
    delimiter = b"--" + boundary
    output = bytearray()
    for part in rewritten:
        output.extend(delimiter + b"\r\n")
        output.extend(part.header_blob + b"\r\n\r\n")
        output.extend(part.content + b"\r\n")
    output.extend(delimiter + b"--\r\n")
    if len(output) > DEFAULT_BODY_BYTES:
        _reject("rewritten_body_too_large")
    return bytes(output)


def _rewrite_worker(
    body: bytes,
    content_type: str,
    prepared_payload: dict[str, Any],
    known_litellm_params: tuple[str, ...],
) -> bytes:
    prepared = PreparedDispatch.model_validate(prepared_payload)
    if prepared.model_locator.startswith("multipart."):
        return _rewrite_multipart(body, content_type, prepared)
    return _rewrite_json(body, prepared, known_litellm_params)


class _ScratchReservation:
    __slots__ = ("_owner", "key_id", "amount", "_released")

    def __init__(self, owner: ProfileExecutor, key_id: str, amount: int) -> None:
        self._owner = owner
        self.key_id = key_id
        self.amount = amount
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._owner._release_scratch(self.key_id, self.amount)


class ProfileExecutor:
    """Exactly two fresh resource-limited workers; parse trees never return."""

    def __init__(self) -> None:
        self.max_workers = PROFILE_WORKERS
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=PROFILE_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_limit_profile_worker,
            max_tasks_per_child=1,
        )
        self._scratch_lock = asyncio.Lock()
        self._scratch = 0
        self._scratch_by_key: dict[str, int] = {}
        self._closed = False

    async def _reserve_scratch(self, key_id: str, amount: int) -> _ScratchReservation:
        async with self._scratch_lock:
            key_total = self._scratch_by_key.get(key_id, 0)
            if (
                amount < 0
                or self._scratch + amount > SCRATCH_BYTES_GLOBAL
                or key_total + amount > SCRATCH_BYTES_PER_KEY
            ):
                raise ProfileError(429, "scratch_limit")
            self._scratch += amount
            self._scratch_by_key[key_id] = key_total + amount
        return _ScratchReservation(self, key_id, amount)

    async def _release_scratch(self, key_id: str, amount: int) -> None:
        async with self._scratch_lock:
            key_total = self._scratch_by_key.get(key_id, 0)
            if amount > self._scratch or amount > key_total:
                raise RuntimeError("profile scratch reservation underflow")
            self._scratch -= amount
            remaining = key_total - amount
            if remaining:
                self._scratch_by_key[key_id] = remaining
            else:
                self._scratch_by_key.pop(key_id, None)

    async def _run(self, function: Any, *arguments: object) -> Any:
        if self._closed:
            raise ProfileError(503, "profile_workers_closed")
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._executor, function, *arguments)
            return await asyncio.wait_for(future, PROFILE_WALL_SECONDS)
        except ValueError as error:
            raise ProfileError(422, str(error)) from None
        except (TimeoutError, concurrent.futures.process.BrokenProcessPool):
            raise ProfileError(503, "profile_worker_unavailable") from None

    async def profile(
        self,
        endpoint: CertifiedEndpoint,
        body: RetainedBody,
        content_type: str,
        route_groups: Mapping[str, RouteGroupId],
        deadline_ms: int,
        key_id: str,
    ) -> RequestProfile:
        scratch = await self._reserve_scratch(key_id, body.size)
        try:
            raw = await body.read()
            if endpoint.model_locator == "json.model":
                try:
                    _prescan_json_limits(raw)
                except ValueError as error:
                    raise ProfileError(422, str(error)) from None
            result = await self._run(_profile_worker, endpoint.name, raw, content_type)
            assert isinstance(result, _ProfileResult)
        finally:
            await scratch.release()
        route_group_id = route_groups.get(result.model_alias)
        if route_group_id is None:
            raise ProfileError(404, "unknown_model")
        return RequestProfile(
            route_group_id=route_group_id,
            model_alias=result.model_alias,
            endpoint=result.endpoint,
            modality=result.modality,
            stream=result.stream,
            input_tokens_max=result.input_tokens_max,
            output_tokens_max=result.output_tokens_max,
            reasoning_tokens_max=result.reasoning_tokens_max,
            tools_count=result.tools_count,
            forced_tool_required=result.forced_tool_required,
            response_schema_present=result.response_schema_present,
            history_turns=result.history_turns,
            deadline_ms=deadline_ms,
        )

    async def rewrite(
        self,
        body: RetainedBody,
        content_type: str,
        prepared: PreparedDispatch,
        known_litellm_params: tuple[str, ...],
        key_id: str,
    ) -> bytes:
        scratch = await self._reserve_scratch(key_id, min(body.size * 2, DEFAULT_BODY_BYTES * 2))
        try:
            raw = await body.read()
            rewritten = await self._run(
                _rewrite_worker,
                raw,
                content_type,
                prepared.model_dump(mode="python"),
                known_litellm_params,
            )
            assert isinstance(rewritten, bytes)
            return rewritten
        finally:
            await scratch.release()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, True, cancel_futures=True)
