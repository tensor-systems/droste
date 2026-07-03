"""Regression: the runner must report subcalls from the same execution context
the subcall client increments.

run() builds an ExecutionContext for HTTPSubcallClient but historically did not
pass it to run_rlm, which then created its own fresh context — so the response
reported subcalls=0 regardless of how many subcalls actually ran.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from rlm_core.execution.context import create_execution_context
from rlm_runner.runner import HTTPSubcallClient, run

ROOT_REPLY = (
    "I'll query the model once and finish.\n"
    "```python\n"
    "hint = llm_query('classify: is this spam?')\n"
    "answer['content'] = 'label: ' + hint\n"
    "answer['ready'] = True\n"
    "```\n"
)


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/root":
            body = {"result": ROOT_REPLY, "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            body = {"result": "spam"}
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


def test_run_reports_actual_subcall_count() -> None:
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = run(
            {
                "model": "test-model",
                "question": "is it spam?",
                "max_iterations": 3,
                "max_depth": 2,
                "max_subcalls": 10,
                "token": "test-token",
                "root_endpoint": f"{base}/root",
                "subcall_endpoint": f"{base}/subcall",
            }
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert response["error"] is None
    assert response["ready"] is True
    assert response["answer"] == "label: spam"
    assert response["subcalls"] == 1


def _client(max_calls: int) -> tuple[HTTPSubcallClient, Any]:
    context = create_execution_context(max_calls=max_calls, max_depth=5)
    client = HTTPSubcallClient(
        endpoint="http://unused.invalid",
        token="t",
        session="",
        session_index=0,
        max_calls=max_calls,
        max_depth=5,
        context=context,
    )
    client._request = lambda payload: "ok"  # type: ignore[method-assign]
    return client, context


def test_rejected_over_limit_attempt_is_not_counted() -> None:
    client, context = _client(max_calls=1)
    assert client.llm_query("a") == "ok"
    with pytest.raises(RuntimeError, match="max subcalls exceeded"):
        client.llm_query("b")
    assert context.stats.calls_made == 1


def test_concurrent_batch_counts_each_issued_call() -> None:
    client, context = _client(max_calls=50)
    results = client.llm_batch([f"p{i}" for i in range(20)])
    assert results == ["ok"] * 20
    assert context.stats.calls_made == 20
