"""BYOK OpenAI-compatible client against a stub chat-completions server (#27).

Covers the launch-gating contract: root call with usage, batch subcalls with
bounded concurrency, usage/call accounting into the shared ExecutionContext,
max_calls enforcement, and HTTP error bodies surfaced (bounded + redacted).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from droste import RLMConfig, run_rlm
from droste.clients.openai_compat import (
    OpenAICompatClient,
    OpenAICompatSubcallClient,
)
from droste.execution.context import create_execution_context
from droste.testing import MockEnvironment


class StubOpenAIServer:
    """Minimal OpenAI-compatible /chat/completions stub.

    Behavior is driven by the request payload:
    - model == "sub-model": echoes the last user message as "echo: <prompt>".
    - otherwise: pops the next queued root response (or "hi" if none queued).
    Records every request payload + auth header, and tracks in-flight
    concurrency so tests can assert the client's bounded fan-out.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.auth_headers: list[str] = []
        self.root_responses: list[str] = []
        self.fail_status: int | None = None
        self.fail_body: bytes = b""
        # Simulate endpoints that 400 on unknown stream_options.
        self.reject_stream_options = False
        # Emit an SSE error chunk mid-stream (after one content chunk).
        self.stream_error_midway = False
        self.usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                with stub._lock:
                    stub.requests.append(payload)
                    stub.auth_headers.append(self.headers.get("Authorization") or "")
                    stub._in_flight += 1
                    stub.max_in_flight = max(stub.max_in_flight, stub._in_flight)
                try:
                    if stub.reject_stream_options and "stream_options" in payload:
                        msg = b'{"error": "unknown parameter: stream_options"}'
                        self.send_response(400)
                        self.send_header("Content-Length", str(len(msg)))
                        self.end_headers()
                        self.wfile.write(msg)
                        return
                    if stub.fail_status is not None:
                        self.send_response(stub.fail_status)
                        self.send_header("Content-Length", str(len(stub.fail_body)))
                        self.end_headers()
                        self.wfile.write(stub.fail_body)
                        return
                    # Hold the request briefly so concurrent calls overlap and
                    # max_in_flight actually observes the fan-out.
                    threading.Event().wait(0.05)
                    if payload.get("model") == "sub-model":
                        prompt = payload["messages"][-1]["content"]
                        content = f"echo: {prompt}"
                    else:
                        with stub._lock:
                            content = (
                                stub.root_responses.pop(0) if stub.root_responses else "hi"
                            )
                    if payload.get("stream"):
                        # SSE: content split into 3 chunks, usage in the final
                        # chunk iff the client asked for include_usage.
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        third = max(1, len(content) // 3)
                        pieces = [content[:third], content[third : 2 * third], content[2 * third :]]
                        for i, piece in enumerate(pieces):
                            if not piece:
                                continue
                            if stub.stream_error_midway and i == 1:
                                err = {"error": {"message": "provider exploded mid-stream", "code": 500}}
                                self.wfile.write(f"data: {json.dumps(err)}\n\n".encode("utf-8"))
                                self.wfile.write(b"data: [DONE]\n\n")
                                return
                            chunk = {
                                "id": "chatcmpl-stub-1",
                                "model": payload.get("model", ""),
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                        final = {
                            "id": "chatcmpl-stub-1",
                            "model": payload.get("model", ""),
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        if (payload.get("stream_options") or {}).get("include_usage"):
                            final["usage"] = dict(stub.usage)
                        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        return
                    body = json.dumps(
                        {
                            "id": "chatcmpl-stub-1",
                            "model": payload.get("model", ""),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": content},
                                    "finish_reason": "stop",
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
                finally:
                    with stub._lock:
                        stub._in_flight -= 1

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def shutdown(self) -> None:
        self._server.shutdown()


@pytest.fixture()
def stub_server():
    server = StubOpenAIServer()
    yield server
    server.shutdown()


def _subcall_client(server: StubOpenAIServer, context, **kwargs) -> OpenAICompatSubcallClient:
    kwargs.setdefault("model", "sub-model")
    kwargs.setdefault("base_url", server.base_url)
    kwargs.setdefault("api_key", "k")
    return OpenAICompatSubcallClient(context=context, **kwargs)


def test_root_responses_create_returns_text_and_usage(stub_server):
    client = OpenAICompatClient(model="root-model", base_url=stub_server.base_url, api_key="k")
    result, usage = client.responses_create(
        [{"role": "user", "content": "Question: hello"}],
        model="",
        return_usage=True,
    )
    assert result == "hi"
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (7, 3, 10)
    payload = stub_server.requests[0]
    assert payload["model"] == "root-model"
    assert payload["messages"][0]["content"] == "Question: hello"
    assert payload["max_tokens"] == 4096
    assert stub_server.auth_headers[0] == "Bearer k"
    assert client.last_stop_reason == "stop"
    assert client.last_response_id == "chatcmpl-stub-1"


def test_root_per_call_model_beats_default(stub_server):
    client = OpenAICompatClient(model="default-model", base_url=stub_server.base_url, api_key="k")
    client.responses_create([{"role": "user", "content": "x"}], model="override-model")
    assert stub_server.requests[0]["model"] == "override-model"


def test_env_vars_configure_client_and_explicit_args_win(stub_server, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", stub_server.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    client = OpenAICompatClient(model="root-model")
    client.responses_create([{"role": "user", "content": "x"}], model="")
    assert stub_server.auth_headers[0] == "Bearer env-key"

    explicit = OpenAICompatClient(model="root-model", api_key="explicit-key")
    # base_url still comes from env; the explicit key must beat OPENAI_API_KEY.
    explicit.responses_create([{"role": "user", "content": "x"}], model="")
    assert stub_server.auth_headers[1] == "Bearer explicit-key"


def test_missing_api_key_omits_authorization_header(stub_server, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatClient(model="root-model", base_url=stub_server.base_url)
    client.responses_create([{"role": "user", "content": "x"}], model="")
    assert stub_server.auth_headers[0] == ""


def test_subcall_llm_query_counts_calls_and_tokens(stub_server):
    context = create_execution_context(max_calls=10, max_depth=3)
    client = _subcall_client(stub_server, context)
    result = client.llm_query("summarize this", context="chunk text")
    assert result == "echo: chunk text\n\nsummarize this"
    assert context.stats.calls_made == 1
    assert context.stats.total_tokens == 10
    payload = stub_server.requests[0]
    assert payload["model"] == "sub-model"
    assert payload["max_tokens"] == 2048  # bounded-output default (#25 parity)
    assert "temperature" not in payload  # endpoint default unless configured


def test_subcall_cost_controls_passthrough(stub_server):
    context = create_execution_context(max_calls=10, max_depth=3)
    client = _subcall_client(
        stub_server,
        context,
        max_output_tokens=512,
        reasoning_effort="low",
        extra_body={"reasoning": {"enabled": False}},
    )
    client.llm_query("p")
    payload = stub_server.requests[0]
    assert payload["max_tokens"] == 512
    assert payload["reasoning_effort"] == "low"
    assert payload["reasoning"] == {"enabled": False}


def test_llm_batch_ordered_results_bounded_concurrency(stub_server):
    context = create_execution_context(max_calls=50, max_depth=3)
    client = _subcall_client(stub_server, context)
    prompts = [f"p{i}" for i in range(12)]
    results = client.llm_batch(prompts)
    assert results == [f"echo: p{i}" for i in range(12)]
    assert context.stats.calls_made == 12
    assert context.stats.total_tokens == 120
    assert stub_server.max_in_flight <= 5


def test_llm_batch_rejects_oversized_batches(stub_server):
    context = create_execution_context(max_calls=100, max_depth=3)
    client = _subcall_client(stub_server, context)
    with pytest.raises(ValueError, match="exceeds max 50"):
        client.llm_batch(["p"] * 51)
    assert context.stats.calls_made == 0


def test_max_calls_enforced_and_rejections_not_counted(stub_server):
    context = create_execution_context(max_calls=3, max_depth=3)
    client = _subcall_client(stub_server, context, max_calls=3)
    for _ in range(3):
        client.llm_query("ok")
    with pytest.raises(RuntimeError, match="max subcalls exceeded"):
        client.llm_query("over budget")
    # The rejected attempt must not inflate the reported subcall count.
    assert context.stats.calls_made == 3


def test_llm_batch_with_errors_reports_per_item_failures(stub_server):
    context = create_execution_context(max_calls=2, max_depth=3)
    client = _subcall_client(stub_server, context, max_calls=2)
    results, errors = client.llm_batch_with_errors(["a", "b", "c", "d"])
    assert len([r for r in results if r]) == 2
    assert len(errors) == 2
    assert all("max subcalls exceeded" in str(e["error"]) for e in errors)
    assert context.stats.calls_made == 2


def test_max_depth_enforced(stub_server):
    context = create_execution_context(max_calls=10, max_depth=0)
    client = _subcall_client(stub_server, context, max_depth=0)
    with pytest.raises(RuntimeError, match="max depth exceeded"):
        client.llm_query("p")
    assert context.stats.calls_made == 0


def test_error_body_surfaced_and_redacted(stub_server):
    stub_server.fail_status = 503
    stub_server.fail_body = (
        b'no healthy provider offers model "gemini-3.5-flash"; api_key=supersecret'
    )
    client = OpenAICompatClient(model="root-model", base_url=stub_server.base_url, api_key="k")
    with pytest.raises(RuntimeError, match=r'HTTP 503: no healthy provider offers model') as exc_info:
        client.responses_create([{"role": "user", "content": "x"}], model="")
    assert "supersecret" not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)

    context = create_execution_context(max_calls=5, max_depth=3)
    subcalls = _subcall_client(stub_server, context)
    with pytest.raises(RuntimeError, match=r"llm_query failed with HTTP 503"):
        subcalls.llm_query("p")


def test_run_rlm_end_to_end_with_byok_clients(stub_server):
    stub_server.root_responses = [
        "```python\n"
        "part = llm_query('describe part one')\n"
        "answer['content'] = part\n"
        "answer['ready'] = True\n"
        "```",
    ]
    context = create_execution_context(max_calls=10, max_depth=3)
    root = OpenAICompatClient(model="root-model", base_url=stub_server.base_url, api_key="k")
    subcalls = _subcall_client(stub_server, context)
    env = MockEnvironment(
        {
            "answer": {"content": "", "ready": False},
            "llm_query": subcalls.llm_query,
            "llm_batch": subcalls.llm_batch,
        }
    )
    result = run_rlm(
        question="what is in part one?",
        environment=env,
        root_llm=root,
        subcalls=subcalls,
        config=RLMConfig(max_iterations=2, root_model="root-model"),
        context=context,
    )
    assert result.ready
    assert result.answer == "echo: describe part one"
    assert result.sub_calls_made == 1
    # Root usage (counted by the loop) + subcall usage (counted by the client).
    assert result.tokens_used == 20


def test_message_content_tool_calls_without_text_raises():
    """A compat endpoint answering with tool_calls and null content must fail
    loudly — this client sends no tools and cannot honor them (codex #32)."""
    from droste.clients.openai_compat import _message_content

    data = {
        "choices": [
            {"message": {"content": None, "tool_calls": [{"id": "c1", "type": "function"}]}}
        ]
    }
    with pytest.raises(RuntimeError, match="tool_calls but no text"):
        _message_content(data, label="root")


def test_message_content_null_without_tool_calls_is_empty():
    from droste.clients.openai_compat import _message_content

    data = {"choices": [{"message": {"content": None}}]}
    assert _message_content(data, label="root") == ""


def test_batch_responses_preserves_explicit_zero_max_tokens(stub_server):
    """max_tokens=0 is an explicit opt-out and must not be coerced to the
    default in the batch path (codex #32)."""
    client = OpenAICompatClient(base_url=stub_server.base_url, api_key="k", model="m")
    client.batch_responses([{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}])
    assert "max_tokens" not in stub_server.requests[0]


# --- root streaming (on_delta) ---


def test_on_delta_streams_and_assembles(stub_server):
    from droste import OpenAICompatClient

    stub_server.root_responses = ["streamed answer body"]
    deltas: list[str] = []
    client = OpenAICompatClient(
        model="root-model",
        base_url=stub_server.base_url,
        api_key="k",
        on_delta=deltas.append,
    )
    text, usage = client.responses_create(
        [{"role": "user", "content": "q"}], model="root-model", return_usage=True
    )
    assert text == "streamed answer body"
    assert len(deltas) >= 2  # arrived in fragments
    assert "".join(deltas) == "streamed answer body"
    assert usage.total_tokens == 10  # include_usage honored end-to-end
    assert stub_server.requests[0]["stream"] is True
    assert stub_server.requests[0]["stream_options"] == {"include_usage": True}


def test_without_on_delta_request_does_not_stream(stub_server):
    from droste import OpenAICompatClient

    stub_server.root_responses = ["plain"]
    client = OpenAICompatClient(model="root-model", base_url=stub_server.base_url, api_key="k")
    text = client.responses_create([{"role": "user", "content": "q"}], model="root-model")
    assert text == "plain"
    assert "stream" not in stub_server.requests[0]


def test_on_delta_retries_without_stream_options(stub_server):
    # codex review (#49): endpoints that 400 on stream_options still stream.
    from droste import OpenAICompatClient

    stub_server.reject_stream_options = True
    stub_server.root_responses = ["retried stream"]
    deltas: list[str] = []
    client = OpenAICompatClient(
        model="root-model",
        base_url=stub_server.base_url,
        api_key="k",
        on_delta=deltas.append,
    )
    text, usage = client.responses_create(
        [{"role": "user", "content": "q"}], model="root-model", return_usage=True
    )
    assert text == "retried stream"
    assert "".join(deltas) == "retried stream"
    assert usage.total_tokens == 0  # no usage without include_usage — tolerated
    assert "stream_options" not in stub_server.requests[-1]


def test_streamed_error_chunk_raises(stub_server):
    # codex review (#49): an SSE error payload mid-stream must fail the call,
    # not return partial text as success.
    import pytest as _pytest

    from droste import OpenAICompatClient

    stub_server.stream_error_midway = True
    stub_server.root_responses = ["will not complete"]
    client = OpenAICompatClient(
        model="root-model", base_url=stub_server.base_url, api_key="k", on_delta=lambda _t: None
    )
    with _pytest.raises(RuntimeError, match="streamed an error"):
        client.responses_create([{"role": "user", "content": "q"}], model="root-model")
