"""Native ModelRelay /responses clients against a stub server.

Covers: root call with usage + auth header, input-item conversion, NDJSON
streaming with on_delta, subcall usage accounting into the shared
ExecutionContext, and the platform-mirroring subcall cost defaults
(bounded output, reasoning_effort="none").
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from droste import (
    BatchItemError,
    BatchItemErrorDetails,
    CapabilityCallError,
    LLMUsageFailure,
    SubcallBatchFailure,
    TokenUsage,
)
from droste.capabilities import broker_subcalls
from droste.clients.modelrelay import (
    ModelRelayClient,
    ModelRelaySubcallClient,
    _batch_item_error_details,
    _usage_from,
)
from droste.environments import RunnerEnvironment
from droste.execution.context import create_execution_context
from droste.protocols.llm_client import CACHE_ANCHOR_MARKER
from droste.structured import _StructuredBatchEvidence, bind_structured_batch, structured_batch


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"usage": {}},
        {"usage": {"input_tokens": 1, "output_tokens": 2}},
        {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 2}},
        {"usage": {"input_tokens": 1, "output_tokens": "2", "total_tokens": 3}},
    ],
)
def test_usage_missing_or_malformed_is_unavailable(payload: object) -> None:
    assert _usage_from(payload).exact is False


def test_usage_accepts_complete_zero_and_preserves_hidden_total() -> None:
    zero = _usage_from({"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})
    hidden = _usage_from({"usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 19}})

    assert zero.exact is True and zero.total_tokens == 0
    assert hidden.exact is True and hidden.total_tokens == 19


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
        self.batch_error_fields: dict[str, object] = {}
        self.batch_item_fields: dict[str, object] = {}
        self.batch_response_fields: dict[str, object] = {}
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
                            error = {
                                "status": 502,
                                "message": stub.batch_error_message,
                                "code": "PROVIDER_ERROR",
                                **stub.batch_error_fields,
                            }
                            results.append(
                                {
                                    "id": item["id"],
                                    "status": "error",
                                    "error": error,
                                    **stub.batch_item_fields,
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
                            **stub.batch_response_fields,
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
        [
            {"role": "system", "content": "sys", CACHE_ANCHOR_MARKER: True},
            {"role": "user", "content": "q", CACHE_ANCHOR_MARKER: True},
        ],
        model="",
        return_usage=True,
    )
    assert text == "the answer"
    assert usage.total_tokens == 10
    assert client.total_usage.prompt_tokens == 7
    assert client.total_usage.completion_tokens == 3
    assert client.total_usage.total_tokens == 10
    assert client.root_requests_issued == 1
    assert client.last_provider == "stub"
    assert client.last_stop_reason == "stop"

    request = stub_native.requests[0]
    assert CACHE_ANCHOR_MARKER not in json.dumps(request)
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
    assert client.root_requests_issued == 1


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
    assert client.root_requests_issued == 1


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
    assert client.root_requests_issued == 1


def test_root_transport_failure_counts_dispatched_request(monkeypatch):
    client = ModelRelayClient(model="root-model", api_key="mr_sk_t")

    def fail_transport(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fail_transport)

    with pytest.raises(RuntimeError, match="network unavailable"):
        client.responses_create([{"role": "user", "content": "q"}], model="")

    assert client.root_requests_issued == 1


def test_root_validation_failure_before_dispatch_is_not_counted(stub_native):
    client = ModelRelayClient(base_url=stub_native.base_url, api_key="mr_sk_t")

    with pytest.raises(ValueError, match="model is required"):
        client.responses_create([{"role": "user", "content": "q"}], model="")
    with pytest.raises(TypeError, match="not JSON serializable"):
        client.responses_create(
            [{"role": "user", "content": [{"type": "text", "text": object()}]}],
            model="root-model",
        )

    assert client.root_requests_issued == 0
    assert stub_native.requests == []


def test_root_request_count_accumulates_across_calls(stub_native):
    client = ModelRelayClient(model="root-model", base_url=stub_native.base_url, api_key="mr_sk_t")

    assert client.root_requests_issued == 0
    for _ in range(3):
        client.responses_create([{"role": "user", "content": "q"}], model="")

    assert client.root_requests_issued == 3


def test_usage_accumulators_preserve_cache_breakdown_across_folds() -> None:
    usages = [
        TokenUsage(11, 2, 13, cache_read_tokens=7, cache_creation_tokens=3),
        TokenUsage(17, 5, 22, cache_read_tokens=13, cache_creation_tokens=2),
    ]
    expected = TokenUsage(
        prompt_tokens=28,
        completion_tokens=7,
        total_tokens=35,
        cache_read_tokens=20,
        cache_creation_tokens=5,
    )

    root = ModelRelayClient(model="root-model", api_key="mr_sk_t")
    for usage in usages:
        root._account_usage(usage)
    assert root.total_usage == expected

    subcall = ModelRelaySubcallClient(
        model="sub-model",
        context=create_execution_context(),
        api_key="mr_sk_t",
    )
    for usage in usages:
        subcall._account_usage(usage)
    assert subcall.total_usage == expected


def test_root_request_count_is_thread_safe(stub_native):
    client = ModelRelayClient(model="root-model", base_url=stub_native.base_url, api_key="mr_sk_t")

    def call_root(_: int) -> str:
        return client.responses_create([{"role": "user", "content": "q"}], model="")

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(call_root, range(32))) == ["hi"] * 32

    assert client.root_requests_issued == 32


def test_root_accounts_usage_before_output_validation(monkeypatch):
    client = ModelRelayClient(model="root-model", api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload: {"usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}},
    )

    with pytest.raises(LLMUsageFailure, match="missing output") as raised:
        client.responses_create([{"role": "user", "content": "q"}], model="", return_usage=True)

    assert raised.value.usage == TokenUsage(7, 3, 10, exact=True)
    assert isinstance(raised.value.cause, RuntimeError)
    assert client.total_usage.total_tokens == 10
    with pytest.raises(RuntimeError, match="missing output"):
        client.responses_create([{"role": "user", "content": "q"}], model="")


def test_subcall_usage_reporting_and_cost_defaults(stub_native):
    context = create_execution_context()
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    assert client.output_token_limit == 2048
    first = client.llm_query_with_usage("first")
    assert first.result == "echo: first" and first.usage.total_tokens == 10
    assert context.stats.calls_made == 1
    assert context.stats.successful_calls == 1
    assert context.stats.total_tokens == 0
    request = stub_native.requests[0]
    # ModelRelay's server-side subcall defaults, applied client-side too.
    assert request["max_output_tokens"] == 2048
    assert request["reasoning_effort"] == "none"

    batch = client.llm_batch_with_usage(["a", "b"])
    assert list(batch.results) == ["echo: a", "echo: b"]
    assert sum(item.total_tokens for item in batch.usage) == 20
    assert context.stats.calls_made == 3
    assert context.stats.successful_calls == 3
    assert context.stats.total_tokens == 0
    assert client.total_usage.prompt_tokens == 21
    assert client.total_usage.completion_tokens == 9
    assert client.total_usage.total_tokens == 30
    assert stub_native.paths == ["/api/v1/responses", "/api/v1/responses/batch"]
    assert len(stub_native.requests) == 2
    assert stub_native.requests[1]["options"] == {"max_concurrent": 5, "fail_fast": False}

    assert client.llm_query("one more") == "echo: one more"
    assert context.stats.calls_made == 4


def test_subcall_reports_deliberately_unbounded_output() -> None:
    context = create_execution_context()
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        api_key="mr_sk_t",
        max_output_tokens=0,
    )

    assert client.output_token_limit is None


def test_subcall_requires_api_key():
    context = create_execution_context()
    with pytest.raises(ValueError, match="api_key"):
        ModelRelaySubcallClient(model="m", context=context, api_key="")


def test_subcall_accounts_usage_before_output_validation(monkeypatch):
    context = create_execution_context()
    client = ModelRelaySubcallClient(model="sub-model", context=context, api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload: {"usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}},
    )

    with pytest.raises(LLMUsageFailure, match="missing output") as raised:
        client.llm_query_with_usage("q")

    assert raised.value.usage == TokenUsage(7, 3, 10, exact=True)
    assert isinstance(raised.value.cause, RuntimeError)
    assert context.stats.calls_made == 1
    assert context.stats.total_tokens == 0
    assert context.stats.successful_calls == 0
    assert client.total_usage.total_tokens == 10
    with pytest.raises(RuntimeError, match="missing output"):
        client.llm_query("q")


def test_native_batch_malformed_success_preserves_item_usage(monkeypatch) -> None:
    context = create_execution_context()
    client = ModelRelaySubcallClient(model="sub-model", context=context, api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload, **kwargs: {
            "results": [
                {
                    "id": "0",
                    "status": "success",
                    "response": {
                        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
                    },
                }
            ]
        },
    )

    with pytest.raises(SubcallBatchFailure, match="missing output") as raised:
        client.llm_batch_with_usage(["q"])

    assert raised.value.result.usage == (TokenUsage(7, 3, 10, exact=True),)
    assert isinstance(raised.value.cause, RuntimeError)
    assert context.stats.calls_made == 1
    assert context.stats.successful_calls == 0
    with pytest.raises(RuntimeError, match="missing output"):
        client.llm_batch(["q"])


def _native_success_result(index: int) -> dict[str, object]:
    return {
        "id": str(index),
        "status": "success",
        "response": {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"ok-{index}"}],
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        },
    }


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (None, "non-object result"),
        ({"id": "invalid"}, "invalid result id"),
        ({"id": "0", "status": "success", "response": {}}, "unexpected result id 0"),
        ({"id": "2", "status": "success", "response": {}}, "unexpected result id 2"),
        ({"id": "1", "status": "success"}, "item 1 missing response"),
        ({"id": "1", "status": "error"}, "item 1 missing error"),
        ({"id": "1", "status": "mystery"}, "item 1 has invalid status 'mystery'"),
    ],
)
def test_native_batch_structural_failure_preserves_earlier_usage(
    monkeypatch, malformed, message
) -> None:
    context = create_execution_context()
    client = ModelRelaySubcallClient(model="sub-model", context=context, api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload, **kwargs: {"results": [_native_success_result(0), malformed]},
    )

    with pytest.raises(SubcallBatchFailure, match=message) as failure:
        client.llm_batch_with_usage(["a", "b"])

    assert failure.value.result.usage == (
        TokenUsage(7, 3, 10, exact=True),
        TokenUsage.unavailable(),
    )
    assert type(failure.value.cause) is RuntimeError
    assert context.stats.calls_made == 2
    assert context.stats.successful_calls == 1
    assert client.total_usage.total_tokens == 10


def test_native_batch_final_missing_id_preserves_earlier_usage_and_public_error(
    monkeypatch,
) -> None:
    context = create_execution_context()
    client = ModelRelaySubcallClient(model="sub-model", context=context, api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload, **kwargs: {"results": [_native_success_result(0)]},
    )

    with pytest.raises(SubcallBatchFailure, match=r"missing result ids \[1\]") as failure:
        client.llm_batch_with_usage(["a", "b"])
    assert failure.value.result.usage == (
        TokenUsage(7, 3, 10, exact=True),
        TokenUsage.unavailable(),
    )
    with pytest.raises(RuntimeError, match=r"missing result ids \[1\]"):
        client.llm_batch(["a", "b"])


def test_native_batch_structural_failure_broker_settles_known_usage_once(
    monkeypatch,
) -> None:
    context = create_execution_context()
    client = ModelRelaySubcallClient(model="sub-model", context=context, api_key="mr_sk_t")
    monkeypatch.setattr(
        client._transport,
        "complete",
        lambda payload, **kwargs: {
            "results": [_native_success_result(0), {"id": "1", "status": "mystery"}]
        },
    )
    brokered = broker_subcalls(
        client,
        context.ledger,
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    )

    with pytest.raises(CapabilityCallError, match="invalid status") as failure:
        brokered.llm_batch(["a", "b"])

    assert failure.value.error.type == "RuntimeError"
    assert context.stats.calls_made == 2
    assert context.stats.successful_calls == 1
    assert context.stats.subcall_total_tokens == 10
    assert context.stats.subcall_usage_complete is False
    assert client.total_usage.total_tokens == 10
    snapshot = context.ledger.snapshot()
    assert snapshot.consumed.tokens > 10
    assert snapshot.reserved.tokens == 0


def test_subcall_batch_reports_ordered_item_errors_without_retry(stub_native):
    context = create_execution_context()
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
            "details": {
                "batch_id": "batch-stub-1",
                "item_id": "1",
                "status_code": 502,
                "code": "PROVIDER_ERROR",
            },
        }
    ]
    assert context.stats.calls_made == 3
    assert context.stats.successful_calls == 2
    assert context.stats.total_tokens == 0
    assert client.total_usage.total_tokens == 20
    assert stub_native.paths == ["/api/v1/responses/batch"]
    assert len(stub_native.requests) == 1


def test_all_failed_native_batch_has_attempts_but_no_success_evidence(stub_native):
    context = create_execution_context()
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


def test_failed_batch_item_never_trusts_a_zero_valued_usage_object(stub_native):
    context = create_execution_context()
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    stub_native.batch_item_fields = {
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    }

    value = client.llm_batch_with_errors_and_usage(["fail"])

    assert value.errors[0]["index"] == 0
    assert value.usage[0].exact is False


def test_subcall_batch_client_only_reports_mechanism_usage(stub_native):
    context = create_execution_context()
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )

    assert client.llm_batch(["a", "b", "c"]) == ["echo: a", "echo: b", "echo: c"]
    assert context.stats.calls_made == 3
    assert len(stub_native.requests) == 1


def test_subcall_batch_item_error_is_bounded_and_redacted(stub_native):
    context = create_execution_context()
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
    assert set(errors[0]) == {"index", "error", "details"}
    assert "message" not in errors[0]["details"]


def test_subcall_batch_raises_typed_error_with_allowlisted_details(stub_native):
    context = create_execution_context()
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    stub_native.batch_error_fields = {
        "request_id": "req-item-9",
        "batch_id": "batch-error-9",
        "item_id": "item-error-9",
        "layer": "gateway",
        "cause": "upstream_timeout",
        "status_code": 504,
        "code": "UPSTREAM_TIMEOUT",
        "retryable": True,
        "payload": {"must": "not survive"},
        "headers": {"authorization": "must not survive"},
    }

    with pytest.raises(BatchItemError) as caught:
        client.llm_batch(["fail"])

    assert str(caught.value) == "llm_batch item 0 failed (UPSTREAM_TIMEOUT): provider exploded"
    assert caught.value.details == BatchItemErrorDetails(
        request_id="req-item-9",
        batch_id="batch-error-9",
        item_id="item-error-9",
        layer="gateway",
        cause="upstream_timeout",
        status_code=504,
        code="UPSTREAM_TIMEOUT",
        retryable=True,
    )
    assert caught.value.details.to_dict() == {
        "request_id": "req-item-9",
        "batch_id": "batch-error-9",
        "item_id": "item-error-9",
        "layer": "gateway",
        "cause": "upstream_timeout",
        "status_code": 504,
        "code": "UPSTREAM_TIMEOUT",
        "retryable": True,
    }


def test_subcall_batch_error_details_drop_malformed_and_unknown_fields(stub_native):
    context = create_execution_context()
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    stub_native.batch_error_fields = {
        "request_id": "token=supersecret " + ("r" * 400),
        "batch_id": {"nested": "discard"},
        "item_id": ["discard"],
        "layer": {"nested": "discard"},
        "cause": ["discard"],
        "status_code": True,
        "status": "502",
        "code": "C" * 200,
        "retryable": "yes",
        "unknown": "discard",
    }

    _, errors = client.llm_batch_with_errors(["fail"])

    details = errors[0]["details"]
    assert details == {
        "request_id": "token=[redacted] " + ("r" * 239),
        "batch_id": "batch-stub-1",
        "item_id": "0",
        "code": "C" * 128,
    }
    assert len(details["request_id"]) == 256


def test_batch_item_error_details_constructor_redacts_and_to_dict_is_safe() -> None:
    details = BatchItemErrorDetails(
        request_id="Authorization: Bearer sk-directsecret123",
        batch_id="api_key=AIzaSyFAKE-KEY",
        item_id="token=supersecret",
        layer="mr_sk_directsecret123",
        cause="sk-anothersecret456",
        code="token=[redacted]",
    )

    payload = details.to_dict()
    assert payload == {
        "request_id": "Authorization: [redacted]",
        "batch_id": "api_key=[redacted]",
        "item_id": "token=[redacted]",
        "layer": "[redacted]",
        "cause": "[redacted]",
        "code": "token=[redacted]",
    }
    serialized = json.dumps(payload)
    for secret in (
        "sk-directsecret123",
        "AIzaSyFAKE-KEY",
        "supersecret",
        "mr_sk_directsecret123",
        "sk-anothersecret456",
    ):
        assert secret not in serialized

    # Reconstructing an already-sanitized value is stable, not double-redacted.
    assert BatchItemErrorDetails(**payload) == details
    with pytest.raises(FrozenInstanceError):
        details.code = "changed"  # type: ignore[misc]


def test_batch_item_error_details_constructor_intrinsically_bounds_strings() -> None:
    details = BatchItemErrorDetails(request_id="r" * 300, code="c" * 200)

    assert details.request_id == "r" * 256
    assert details.code == "c" * 128
    assert details.to_dict() == {"request_id": "r" * 256, "code": "c" * 128}


def test_batch_item_error_details_reject_out_of_contract_scalar_types() -> None:
    with pytest.raises(ValueError, match="request_id"):
        BatchItemErrorDetails(request_id="   ")
    with pytest.raises(ValueError, match="request_id"):
        BatchItemErrorDetails(request_id=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="status_code"):
        BatchItemErrorDetails(status_code=99)
    with pytest.raises(ValueError, match="status_code"):
        BatchItemErrorDetails(status_code=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="retryable"):
        BatchItemErrorDetails(retryable="yes")  # type: ignore[arg-type]


def test_batch_item_error_wire_projection_remains_fail_soft() -> None:
    details = _batch_item_error_details(
        {"id": "batch-fallback", "batch_id": {"nested": "discard"}},
        {"id": "item-fallback", "request_id": ["discard"]},
        {
            "request_id": {"nested": "discard"},
            "batch_id": ["discard"],
            "item_id": {"nested": "discard"},
            "layer": ["discard"],
            "cause": {"nested": "discard"},
            "status_code": True,
            "status": 502,
            "code": "token=supersecret",
            "retryable": "yes",
            "payload": {"must": "not survive"},
        },
    )

    assert details.to_dict() == {
        "batch_id": "batch-fallback",
        "item_id": "item-fallback",
        "code": "token=[redacted]",
    }


def test_batch_error_details_survive_broker_semantic_and_runner_paths(stub_native):
    context = create_execution_context()
    client = ModelRelaySubcallClient(
        model="sub-model",
        context=context,
        base_url=stub_native.base_url,
        api_key="mr_sk_t",
    )
    stub_native.batch_error_fields = {
        "layer": "gateway",
        "cause": "rate_limited",
        "status_code": 429,
        "retryable": True,
    }
    stub_native.batch_item_fields = {"request_id": "req-end-to-end"}
    stub_native.batch_response_fields = {"batch_id": "batch-end-to-end"}
    expected = {
        "request_id": "req-end-to-end",
        "batch_id": "batch-end-to-end",
        "item_id": "0",
        "layer": "gateway",
        "cause": "rate_limited",
        "status_code": 429,
        "code": "PROVIDER_ERROR",
        "retryable": True,
    }

    brokered = broker_subcalls(
        client,
        context.ledger,
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    )
    _, broker_errors = brokered.llm_batch_with_errors(["fail"])
    assert broker_errors[0]["details"] == expected

    evidence = _StructuredBatchEvidence()
    semantic_batch = bind_structured_batch(brokered, evidence)
    semantic_result = semantic_batch(
        ["fail"],
        {"type": "object"},
        max_repair_attempts=0,
    )
    assert semantic_result["errors"][0]["details"] == expected
    assert evidence.unresolved_batches == 1
    assert evidence.unresolved_items == 1

    environment = RunnerEnvironment(
        context={},
        registry=None,
        subcalls=client,
        max_output_chars=10_000,
        exec_timeout_ms=0,
        budget_ledger=context.ledger,
    )
    environment.execute(
        "runner_result = llm_batch_json(['fail'], {'type': 'object'}, max_repair_attempts=0)"
    )
    assert environment.globals()["runner_result"]["errors"][0]["details"] == expected


def test_native_batch_structured_repair_keeps_one_wire_request_per_attempt(stub_native):
    context = create_execution_context()
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
    assert context.stats.total_tokens == 0
    assert client.total_usage.total_tokens == 30
    assert stub_native.paths == ["/api/v1/responses/batch", "/api/v1/responses/batch"]
