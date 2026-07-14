"""Native Anthropic Messages client: transport, accounting, detection."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from droste import (
    AnthropicClient,
    AnthropicSubcallClient,
    create_execution_context,
)
from droste_cli.main import main


class StubAnthropicServer:
    """Minimal Anthropic /v1/messages stub.

    Behavior is driven by the request payload:
    - model == "sub-model": echoes the last user message as "echo: <prompt>".
    - otherwise: pops the next queued root response (or "hi" if none queued).
    - payload["stream"] is true: emits the Messages SSE event sequence
      (message_start / content_block_delta x3 / message_delta / message_stop),
      or an error event mid-stream when ``stream_error_midway`` is set.
    Records every request payload + headers, and tracks in-flight concurrency
    so tests can assert the client's bounded fan-out.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict[str, str]] = []
        self.root_responses: list[str] = []
        self.fail_status: int | None = None
        self.fail_body: bytes = b""
        self.stream_error_midway = False
        self.usage = {"input_tokens": 7, "output_tokens": 3}
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
                    stub.headers.append(
                        {
                            "x-api-key": self.headers.get("x-api-key") or "",
                            "anthropic-version": self.headers.get("anthropic-version") or "",
                        }
                    )
                    stub._in_flight += 1
                    stub.max_in_flight = max(stub.max_in_flight, stub._in_flight)
                try:
                    if stub.fail_status is not None:
                        self.send_response(stub.fail_status)
                        self.send_header("Content-Length", str(len(stub.fail_body)))
                        self.end_headers()
                        self.wfile.write(stub.fail_body)
                        return
                    # Hold briefly so concurrent calls overlap and
                    # max_in_flight observes the fan-out.
                    threading.Event().wait(0.05)
                    if payload.get("model") == "sub-model":
                        prompt = payload["messages"][-1]["content"]
                        content = f"echo: {prompt}"
                    else:
                        with stub._lock:
                            content = stub.root_responses.pop(0) if stub.root_responses else "hi"
                    if payload.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()

                        def sse(event: dict) -> None:
                            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))

                        sse(
                            {
                                "type": "message_start",
                                "message": {
                                    "id": "msg_stub_1",
                                    "model": payload.get("model", ""),
                                    "usage": {"input_tokens": stub.usage["input_tokens"]},
                                },
                            }
                        )
                        third = max(1, len(content) // 3)
                        pieces = [content[:third], content[third : 2 * third], content[2 * third :]]
                        for i, piece in enumerate(pieces):
                            if not piece:
                                continue
                            if stub.stream_error_midway and i == 1:
                                sse(
                                    {
                                        "type": "error",
                                        "error": {"type": "overloaded_error", "message": "boom"},
                                    }
                                )
                                return
                            sse(
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": piece},
                                }
                            )
                        sse(
                            {
                                "type": "message_delta",
                                "delta": {"stop_reason": "end_turn"},
                                "usage": {"output_tokens": stub.usage["output_tokens"]},
                            }
                        )
                        sse({"type": "message_stop"})
                        return
                    body = json.dumps(
                        {
                            "id": "msg_stub_1",
                            "model": payload.get("model", ""),
                            "content": [{"type": "text", "text": content}],
                            "stop_reason": "end_turn",
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
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def shutdown(self) -> None:
        self._server.shutdown()


@pytest.fixture()
def stub():
    server = StubAnthropicServer()
    yield server
    server.shutdown()


# --- root client ---


def test_root_create_returns_text_and_usage(stub):
    stub.root_responses = ["claude says hi"]
    client = AnthropicClient(model="claude-test", base_url=stub.base_url, api_key="sk-ant-k")
    text, usage = client.responses_create(
        [{"role": "user", "content": "q"}], model="claude-test", return_usage=True
    )
    assert text == "claude says hi"
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (7, 3, 10)
    assert stub.headers[0]["x-api-key"] == "sk-ant-k"
    assert stub.headers[0]["anthropic-version"] == "2023-06-01"
    assert client.last_stop_reason == "end_turn"


def test_system_message_lifted_to_top_level(stub):
    stub.root_responses = ["ok"]
    client = AnthropicClient(model="claude-test", base_url=stub.base_url, api_key="sk-ant-k")
    client.responses_create(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "prior"},
        ],
        model="claude-test",
    )
    sent = stub.requests[0]
    assert sent["system"] == "be terse"
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant"]


def test_max_tokens_always_present_and_temperature_omitted(stub):
    stub.root_responses = ["ok", "warm"]
    client = AnthropicClient(model="claude-test", base_url=stub.base_url, api_key="sk-ant-k")
    client.responses_create([{"role": "user", "content": "q"}], model="claude-test", max_tokens=0)
    assert stub.requests[0]["max_tokens"] == 4096  # required by the API
    assert "temperature" not in stub.requests[0]

    warm = AnthropicClient(
        model="claude-test", base_url=stub.base_url, api_key="sk-ant-k", temperature=0.6
    )
    warm.responses_create([{"role": "user", "content": "q"}], model="claude-test")
    assert stub.requests[1]["temperature"] == 0.6


def test_stop_maps_to_stop_sequences_and_extra_body_wins(stub):
    stub.root_responses = ["ok"]
    client = AnthropicClient(
        model="claude-test",
        base_url=stub.base_url,
        api_key="sk-ant-k",
        stop=["END"],
        extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    client.responses_create([{"role": "user", "content": "q"}], model="claude-test")
    sent = stub.requests[0]
    assert sent["stop_sequences"] == ["END"]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_streaming_deltas_assembly_and_usage(stub):
    stub.root_responses = ["streamed claude body"]
    deltas: list[str] = []
    client = AnthropicClient(
        model="claude-test", base_url=stub.base_url, api_key="sk-ant-k", on_delta=deltas.append
    )
    text, usage = client.responses_create(
        [{"role": "user", "content": "q"}], model="claude-test", return_usage=True
    )
    assert text == "streamed claude body"
    assert len(deltas) >= 2
    assert "".join(deltas) == "streamed claude body"
    assert usage.total_tokens == 10
    assert stub.requests[0]["stream"] is True


def test_mid_stream_error_raises(stub):
    stub.stream_error_midway = True
    stub.root_responses = ["will not complete"]
    client = AnthropicClient(
        model="claude-test", base_url=stub.base_url, api_key="sk-ant-k", on_delta=lambda _t: None
    )
    with pytest.raises(RuntimeError, match="streamed an error"):
        client.responses_create([{"role": "user", "content": "q"}], model="claude-test")


def test_http_error_is_bounded_and_redacted(stub):
    stub.fail_status = 401
    stub.fail_body = b'{"error": {"message": "invalid x-api-key sk-ant-SECRETSECRETSECRET"}}'
    client = AnthropicClient(model="claude-test", base_url=stub.base_url, api_key="sk-ant-k")
    with pytest.raises(RuntimeError, match="HTTP 401") as exc:
        client.responses_create([{"role": "user", "content": "q"}], model="claude-test")
    assert "sk-ant-SECRETSECRETSECRET" not in str(exc.value)


# --- subcall client ---


def test_subcall_client_reports_usage_without_owning_budget_policy(stub):
    ctx = create_execution_context()
    sub = AnthropicSubcallClient(
        model="sub-model", context=ctx, base_url=stub.base_url, api_key="sk-ant-k"
    )
    assert sub.output_token_limit == 2048
    assert sub.llm_query("alpha") == "echo: alpha"
    assert sub.llm_query("beta") == "echo: beta"
    assert ctx.stats.calls_made == 2
    assert ctx.stats.successful_calls == 2
    assert ctx.stats.total_tokens == 20  # 2 x (7 + 3)
    assert sub.llm_query("gamma") == "echo: gamma"
    assert ctx.stats.calls_made == 3
    assert stub.requests[0]["max_tokens"] == 2048  # bounded by default


def test_subcall_requires_positive_output_bound(stub):
    ctx = create_execution_context()
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicSubcallClient(
            model="sub-model",
            context=ctx,
            base_url=stub.base_url,
            api_key="sk-ant-k",
            max_output_tokens=0,
        )


def test_batch_bounded_concurrency(stub):
    ctx = create_execution_context()
    sub = AnthropicSubcallClient(
        model="sub-model",
        context=ctx,
        base_url=stub.base_url,
        api_key="sk-ant-k",
        max_parallel=3,
    )
    results = sub.llm_batch([f"p{i}" for i in range(9)])
    assert results == [f"echo: p{i}" for i in range(9)]
    assert stub.max_in_flight <= 3


# --- CLI provider detection ---


def _clean_provider_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_select_provider_matrix(monkeypatch):
    import argparse

    from droste_cli.main import select_provider

    def args(**kw):
        defaults = {"base_url": None, "api_key": None, "model": "m"}
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    _clean_provider_env(monkeypatch)
    # Explicit endpoint always wins compat.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert select_provider(args(base_url="http://localhost:11434/v1")) == "openai"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    assert select_provider(args()) == "openai"
    monkeypatch.delenv("OPENAI_BASE_URL")
    # Key prefix is the fact.
    assert select_provider(args(api_key="sk-ant-abc")) == "anthropic"
    assert select_provider(args(api_key="sk-proj-abc")) == "openai"
    # claude-* model + anthropic key beats a coexisting OPENAI_API_KEY.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    assert select_provider(args(model="claude-opus-4-8")) == "anthropic"
    assert select_provider(args(model="gpt-5.5")) == "openai"
    monkeypatch.delenv("OPENAI_API_KEY")
    # Anthropic env alone routes anthropic.
    assert select_provider(args(model="gpt-5.5")) == "anthropic"
    _clean_provider_env(monkeypatch)
    assert select_provider(args()) == "openai"


def test_cli_e2e_anthropic_via_env(stub, tmp_path, monkeypatch, capsys):
    _clean_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", stub.base_url)
    doc = tmp_path / "doc.txt"
    doc.write_text("anthropic content")
    stub.root_responses = [
        "```python\nanswer['content'] = context['files'][0]['text']\nanswer['ready'] = True\n```",
    ]
    code = main([str(doc), "what does it say?", "--model", "claude-test"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "anthropic content"
    assert stub.headers[0]["x-api-key"] == "sk-ant-test"


def test_cli_reasoning_effort_rejected_for_anthropic(tmp_path, monkeypatch, capsys):
    _clean_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    f = tmp_path / "a.txt"
    f.write_text("x")
    code = main([str(f), "q", "--model", "claude-test", "--reasoning-effort", "low"])
    assert code == 2
    assert "thinking" in capsys.readouterr().err


def test_cli_keyless_error_mentions_anthropic(tmp_path, monkeypatch, capsys):
    _clean_provider_env(monkeypatch)
    f = tmp_path / "a.txt"
    f.write_text("x")
    code = main([str(f), "q", "--model", "gpt-5.2-mini"])
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_clients_subpackage_reexports():
    # codex review: the droste.clients import surface must stay
    # consistent for all built-in clients. Assert names explicitly so a
    # linter can never strip the "unused" imports this test exists for.
    import droste.clients as clients

    for name in (
        "AnthropicClient",
        "AnthropicSubcallClient",
        "OpenAICompatClient",
        "OpenAICompatSubcallClient",
        "ModelRelayClient",
        "ModelRelaySubcallClient",
    ):
        assert hasattr(clients, name), name
