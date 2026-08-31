"""Secret-free fake provider for the opt-in pinned LiteLLM contract stack."""

from __future__ import annotations

import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    _counter_lock = threading.Lock()
    _provider_calls = 0

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, value: object, status: int = 200) -> None:
        raw = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if self.path == "/counter":
            with self._counter_lock:
                count = self._provider_calls
            self._send_json({"provider_calls": count})
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        with self._counter_lock:
            type(self)._provider_calls += 1
        request = self._read_json()
        model = request.get("model", "fixture-model")
        now = int(time.time())
        if self.path.endswith("/chat/completions"):
            self._send_json(
                {
                    "id": "chatcmpl_fixture",
                    "object": "chat.completion",
                    "created": now,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "fixture"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
            return
        if self.path.endswith("/completions"):
            self._send_json(
                {
                    "id": "cmpl_fixture",
                    "object": "text_completion",
                    "created": now,
                    "model": model,
                    "choices": [{"index": 0, "text": "fixture", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
            return
        if self.path.endswith("/responses"):
            self._send_json(
                {
                    "id": "resp_fixture",
                    "object": "response",
                    "created_at": now,
                    "status": "completed",
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "max_output_tokens": 1,
                    "model": model,
                    "output": [
                        {
                            "id": "msg_fixture",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "fixture",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "parallel_tool_calls": True,
                    "previous_response_id": None,
                    "reasoning": {"effort": None, "summary": None},
                    "store": False,
                    "temperature": 1.0,
                    "text": {"format": {"type": "text"}},
                    "tool_choice": "auto",
                    "tools": [],
                    "top_p": 1.0,
                    "truncation": "disabled",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    "user": None,
                    "metadata": {},
                }
            )
            return
        if self.path.endswith("/embeddings"):
            self._send_json(
                {
                    "object": "list",
                    "model": model,
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
            return
        if self.path.endswith("/rerank"):
            self._send_json(
                {
                    "id": "rerank_fixture",
                    "results": [{"index": 0, "relevance_score": 1.0}],
                    "meta": {"billed_units": {"search_units": 1}},
                }
            )
            return
        if self.path.endswith("/audio/speech"):
            raw = (
                b"RIFF$\x00\x00\x00WAVEfmt "
                b"\x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00@\x1f\x00\x00"
                b"\x01\x00\x08\x00data\x00\x00\x00\x00"
            )
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path.endswith("/audio/transcriptions"):
            self._send_json({"text": "fixture"})
            return
        if self.path.endswith("/images/generations"):
            self._send_json(
                {
                    "created": now,
                    "data": [{"url": "https://example.invalid/fixture.png"}],
                }
            )
            return
        self._send_json({"error": "not found", "path": self.path}, 404)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
