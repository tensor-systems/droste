"""Native ModelRelay /responses clients against a stub server.

Covers: root call with usage + auth header, input-item conversion, NDJSON
streaming with on_delta, subcall accounting into the shared ExecutionContext
(max_calls, usage), and the platform-mirroring subcall cost defaults
(bounded output, reasoning_effort="none").
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from droste.clients.modelrelay import ModelRelayClient, ModelRelaySubcallClient
from droste.exceptions import SubcallBudgetExceeded
from droste.execution.context import create_execution_context
from droste.structured import structured_batch


class StubResponsesServer:
    """Minimal native /responses stub.

    - model == "sub-model": echoes the input text as "echo: <prompt>".
    - otherwise: pops the next queued root response (or "hi").
    Streams NDJSON when the responses-stream Accept profile is requested.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.root_responses: list[str] = []
        self.fail_status: int | None = None
        self.fail_body: bytes = b'{"error":"boom"}'
        self.stream_error_midway = False
        self.stream_truncate = False  # drop the connection before completion
        self.usage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
        self.batch_error_message = "provider exploded"
        stub = self

        def subcall_content(prompt: str) -> str:
            if prompt == "structured-valid":
                return '{"count":2}'
            if prompt == "structured-malformed":
                return "not json"
            if "Original task:\nstructured-malformed" in prompt:
                return '{"count":3}'
            return f"echo: {prompt}"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                with threading.Lock():
                    stub.requests.append(payload)
                    stub.paths.append(self.path)
                    stub.headers.append({k.lower(): v for k, v in self.headers.items()})
                if stub.fail_status is not None:
                    self.send_response(stub.fail_status)
                    self.send_header("Content-Length", str(len(stub.fail_body)))
                    self.end_headers()
                    self.wfile.write(stub.fail_body)
                    return
                if self.path.endswith("/responses/batch"):
                    results = []
                    for item in payload["requests"]:
                        prompt = item["input"][-1]["content"][0]["text"]
                        if prompt == "fail":
                            results.append(
                                {
                                    "id": item["id"],
                                    "status": "error",
                                    "error": {
                                        "status": 502,
                                        "message": stub.batch_error_message,
                                        "code": "PROVIDER_ERROR",
                                    },
                                }
                            )
                            continue
                        results.append(
                            {
                                "id": item["id"],
                                "status": "success",
                                "response": {
                                    "output": [
                                        {
                                            "type": "message",
                                            "role": "assistant",
                                            "content": [
                                                {"type": "text", "text": subcall_content(prompt)}
                                            ],
                                        }
                                    ],
                                    "usage": dict(stub.usage),
                                },
                            }
                        )
                    successful = sum(item["status"] == "success" for item in results)
                    body = json.dumps(
                        {
                            "id": "batch-stub-1",
                            # Reverse wire order to prove ids restore caller order.
                            "results": list(reversed(results)),
                            "usage": {
                                "total_input_tokens": successful * stub.usage["input_tokens"],
                                "total_output_tokens": successful * stub.usage["output_tokens"],
                                "total_requests": len(results),
                                "successful_requests": successful,
                                "failed_requests": len(results) - successful,
                            },
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if payload.get("model") == "sub-model":
                    prompt = payload["input"][-1]["content"][0]["text"]
                    content = subcall_content(prompt)
                else:
                    content = stub.root_responses.pop(0) if stub.root_responses else "hi"
                accept = str(self.headers.get("Accept") or "")
                if "responses-stream" in accept:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    third = max(1, len(content) // 3)
                    pieces = [content[:third], content[third : 2 * third], content[2 * third :]]
                    self.wfile.write(
                        (
                            json.dumps({"type": "start", "model": payload.get("model")}) + "\n"
                        ).encode()
                    )
                    for i, piece in enumerate(p for p in pieces if p):
                        if stub.stream_error_midway and i == 1:
                            event = {
                                "type": "error",
                                "code": "PROVIDER_ERROR",
                                "message": "exploded",
                            }
                            self.wfile.write((json.dumps(event) + "\n").encode())
                            return
                        event = {"type": "update", "delta": piece, "stream_version": "v2"}
                        self.wfile.write((json.dumps(event) + "\n").encode())
                    if stub.stream_truncate:
                        return  # connection drops before the completion event
                    completion = {
                        "type": "completion",
                        "request_id": "req-stub-1",
                        "content": content,
                        "model": payload.get("model"),
                        "provider": "stub",
                        "stop_reason": "stop",
                        "usage": dict(stub.usage),
                    }
                    self.wfile.write((json.dumps(completion) + "\n").encode())
                    return
                body = json.dumps(
                    {
                        "id": "resp-stub-1",
                        "model": payload.get("model"),
                        "provider": "stub",
                        "stop_reason": "stop",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "text", "text": content}],
                            }
                        ],
                        "usage": dict(stub.usage),
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/api/v1"

    def shutdown(self) -> None:
        self._server.shutdown()


@pytest.fixture()
def stub_native():
    server = StubResponsesServer()
    yield server
    server.shutdown()


def test_root_returns_text_usage_and_sends_api_key_header(stub_native):
    client = ModelRelayClient(model="root-model", base_url=stub_native.base_url, api_key="mr_sk_t")
    stub_native.root_responses = ["the answer"]
    text, usage = client.responses_create(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        model="",
        return_usage=True,
    )
    assert text == "the answer"
    assert usage.total_tokens == 10
    assert client.total_usage.prompt_tokens == 7
    assert client.total_usage.completion_tokens == 3
    assert client.total_usage.total_tokens == 10
    assert client.last_provider == "stub"
    assert client.last_stop_reason == "stop"

    request = stub_native.requests[0]
    assert request["model"] == "root-model"
    assert request["input"][0] == {
        "type": "message",
        "role": "system",
        "content": [{"type": "text", "text": "sys"}],
    }
    headers = stub_native.headers[0]
    assert headers.get("x-modelrelay-api-key") == "mr_sk_t"
    assert "authorization" not in headers


def test_root_streaming_delivers_deltas(stub_native):
    deltas: list[str] = []
    client = ModelRelayClient(
        model="root-model",
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
        on_delta=deltas.append,
    )
    stub_native.root_responses = ["streamed answer text"]
    text, usage = client.responses_create(
        [{"role": "user", "content": "q"}], model="", return_usage=True
    )
    assert text == "streamed answer text"
    assert "".join(deltas) == "streamed answer text"
    assert len(deltas) > 1
    assert usage.total_tokens == 10
    assert client.total_usage.total_tokens == 10


def test_root_stream_error_fails_loudly(stub_native):
    client = ModelRelayClient(
        model="root-model",
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
        on_delta=lambda _: None,
    )
    stub_native.root_responses = ["long enough content to split"]
    stub_native.stream_error_midway = True
    with pytest.raises(RuntimeError, match="streamed an error"):
        client.responses_create([{"role": "user", "content": "q"}], model="")


def test_root_stream_truncation_fails_loudly(stub_native):
    # A dropped connection must never surface partial generated code as a
    # successful root response (codex review finding).
    client = ModelRelayClient(
        model="root-model",
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
        on_delta=lambda _: None,
    )
    stub_native.root_responses = ["long enough content to split"]
    stub_native.stream_truncate = True
    with pytest.raises(RuntimeError, match="without a completion event"):
        client.responses_create([{"role": "user", "content": "q"}], model="")


def test_root_http_error_is_bounded(stub_native):
    client = ModelRelayClient(model="root-model", base_url=stub_native.base_url, api_key="mr_sk_t")
    stub_native.fail_status = 402
    stub_native.fail_body = b'{"error":"insufficient account balance - please add funds"}'
    with pytest.raises(RuntimeError, match="HTTP 402"):
        client.responses_create([{"role": "user", "content": "q"}], model="")


def test_root_accounts_usage_before_output_validation(monkeypatch):
    client = ModelRelayClient(model="root-model", api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload: {"usage": {"input_tokens": 7, "output_tokens": 3}},
    )

    with pytest.raises(RuntimeError, match="missing output"):
        client.responses_create([{"role": "user", "content": "q"}], model="")

    assert client.total_usage.total_tokens == 10


def test_subcall_accounting_and_cost_defaults(stub_native):
    context = create_execution_context(max_calls=3, max_iterations=5)
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    assert client.llm_query("first") == "echo: first"
    assert context.stats.calls_made == 1
    assert context.stats.successful_calls == 1
    assert context.stats.total_tokens == 10
    request = stub_native.requests[0]
    # ModelRelay's server-side subcall defaults, applied client-side too.
    assert request["max_output_tokens"] == 2048
    assert request["reasoning_effort"] == "none"

    assert client.llm_batch(["a", "b"]) == ["echo: a", "echo: b"]
    assert context.stats.calls_made == 3
    assert context.stats.successful_calls == 3
    assert context.stats.total_tokens == 30
    assert client.total_usage.prompt_tokens == 21
    assert client.total_usage.completion_tokens == 9
    assert client.total_usage.total_tokens == 30
    assert stub_native.paths == ["/api/v1/responses", "/api/v1/responses/batch"]
    assert len(stub_native.requests) == 2
    assert stub_native.requests[1]["options"] == {"max_concurrent": 5, "fail_fast": False}

    with pytest.raises(SubcallBudgetExceeded, match="max subcalls exceeded"):
        client.llm_query("over budget")
    assert context.stats.calls_made == 3  # rejected attempt not counted


def test_subcall_requires_api_key():
    context = create_execution_context(max_calls=1, max_iterations=1)
    with pytest.raises(ValueError, match="api_key"):
        ModelRelaySubcallClient(model="m", context=context, api_key="")


def test_subcall_accounts_usage_before_output_validation(monkeypatch):
    context = create_execution_context(max_calls=1, max_iterations=1)
    client = ModelRelaySubcallClient(model="sub-model", context=context, api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload: {"usage": {"input_tokens": 7, "output_tokens": 3}},
    )

    with pytest.raises(RuntimeError, match="missing output"):
        client.llm_query("q")

    assert context.stats.calls_made == 1
    assert context.stats.total_tokens == 10
    assert context.stats.successful_calls == 0
    assert client.total_usage.total_tokens == 10


def test_subcall_batch_reports_ordered_item_errors_without_retry(stub_native):
    context = create_execution_context(max_calls=3, max_iterations=5)
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )

    results, errors = client.llm_batch_with_errors(["a", "fail", "c"])

    assert results == ["echo: a", "", "echo: c"]
    assert errors == [
        {
            "index": 1,
            "error": "llm_batch item 1 failed (PROVIDER_ERROR): provider exploded",
        }
    ]
    assert context.stats.calls_made == 3
    assert context.stats.successful_calls == 2
    assert context.stats.total_tokens == 20
    assert client.total_usage.total_tokens == 20
    assert stub_native.paths == ["/api/v1/responses/batch"]
    assert len(stub_native.requests) == 1


def test_all_failed_native_batch_has_attempts_but_no_success_evidence(stub_native):
    context = create_execution_context(max_calls=2, max_iterations=5)
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )

    results, errors = client.llm_batch_with_errors(["fail", "fail"])

    assert results == ["", ""]
    assert [item["index"] for item in errors] == [0, 1]
    assert context.stats.calls_made == 2
    assert context.stats.successful_calls == 0
    assert context.stats.total_tokens == 0


def test_subcall_batch_budget_rejection_is_atomic(stub_native):
    context = create_execution_context(max_calls=2, max_iterations=5)
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )

    with pytest.raises(SubcallBudgetExceeded, match="max subcalls exceeded"):
        client.llm_batch(["a", "b", "c"])

    assert context.stats.calls_made == 0
    assert stub_native.requests == []


def test_subcall_batch_item_error_is_bounded_and_redacted(stub_native):
    context = create_execution_context(max_calls=1, max_iterations=5)
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    stub_native.batch_error_message = "token=supersecret " + ("x" * 1000)

    _, errors = client.llm_batch_with_errors(["fail"])

    message = str(errors[0]["error"])
    assert "supersecret" not in message
    assert "[redacted]" in message
    assert len(message) < 600


def test_native_batch_structured_repair_keeps_one_wire_request_per_attempt(stub_native):
    context = create_execution_context(max_calls=3, max_iterations=5)
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    schema = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer"}},
        "additionalProperties": False,
    }

    result = structured_batch(
        client,
        ["structured-valid", "structured-malformed"],
        schema,
        max_repair_attempts=1,
    )

    assert result["values"] == [{"count": 2}, {"count": 3}]
    assert result["attempts"] == [1, 2]
    assert result["repairs_made"] == 1
    assert context.stats.calls_made == 3
    assert context.stats.successful_calls == 3
    assert context.stats.total_tokens == 30
    assert stub_native.paths == ["/api/v1/responses/batch", "/api/v1/responses/batch"]
