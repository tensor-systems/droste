"""Regression: the runner must report subcalls from the same execution context
the subcall client increments.

run() builds an ExecutionContext for HTTPSubcallClient but historically did not
pass it to run_rlm, which then created its own fresh context — so the response
reported subcalls=0 regardless of how many subcalls actually ran.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from droste import Budget
from droste.execution.context import create_execution_context
from droste.protocols.llm_client import LLMUsageFailure, TokenUsage
from droste.protocols.subcall_client import SubcallBatchFailure, SubcallQueryResult
from droste_runner.http_clients import RootLLMClient
from droste_runner.runner import HTTPSubcallClient, run

ROOT_REPLY = (
    "I'll query the model once and finish.\n"
    "```python\n"
    "hint = llm_query('classify: is this spam?')\n"
    "answer['content'] = 'label: ' + hint\n"
    "answer['metadata'] = {'evidence_ids': ['classification-1']}\n"
    "answer['ready'] = True\n"
    "```\n"
)


def _budget(**overrides: int) -> dict[str, int]:
    return Budget(**overrides).as_dict()


def test_http_subcall_reports_only_an_explicit_output_limit() -> None:
    context = create_execution_context()
    kwargs = {
        "endpoint": "https://example.invalid/subcall",
        "token": "t",
        "session": "s",
        "session_index": 0,
        "context": context,
    }

    assert not hasattr(HTTPSubcallClient(**kwargs), "output_token_limit")
    assert HTTPSubcallClient(**kwargs, max_output_tokens=512).output_token_limit == 512


def test_http_subcall_usage_requires_explicit_provider_total() -> None:
    context = create_execution_context()
    client = HTTPSubcallClient(
        endpoint="https://example.invalid/subcall",
        token="t",
        session="s",
        session_index=0,
        context=context,
    )

    missing = client._record_usage({"input_tokens": 2, "output_tokens": 3})
    hidden = client._record_usage(
        {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 19,
            "reasoning_tokens": 1,
            "observation_basis": "exact",
        }
    )
    zero = client._record_usage(
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "observation_basis": "exact",
        }
    )

    assert missing.exact is False
    assert hidden.exact is True and hidden.total_tokens == 19
    assert zero.exact is True and zero.total_tokens == 0
    assert context.stats.subcall_usage_complete is True
    assert context.stats.total_tokens == 0


def test_http_clients_preserve_usage_when_result_is_missing(monkeypatch) -> None:
    raw = json.dumps(
        {"usage": {"input_tokens": 7, "output_tokens": "bad", "total_tokens": 19}}
    ).encode()

    class Response:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return raw

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    context = create_execution_context()
    subcall = HTTPSubcallClient(
        endpoint="https://example.invalid/subcall",
        token="t",
        session="s",
        session_index=0,
        context=context,
    )
    with pytest.raises(LLMUsageFailure, match="missing subcall result") as subcall_failure:
        subcall.llm_query_with_usage("q")
    assert subcall_failure.value.usage == TokenUsage(7, 0, 19)
    with pytest.raises(RuntimeError, match="missing subcall result"):
        subcall.llm_query("q")

    root = RootLLMClient(
        endpoint="https://example.invalid/root",
        token="t",
        default_model="root-model",
        provider=None,
        max_output_tokens=100,
        temperature=None,
        stop=None,
        session="s",
        session_index=0,
    )
    with pytest.raises(LLMUsageFailure, match="missing root result") as root_failure:
        root.responses_create(
            [{"role": "user", "content": "q"}],
            model="",
            return_usage=True,
        )
    assert root_failure.value.usage == TokenUsage(7, 0, 19)


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/root":
            body = {
                "result": ROOT_REPLY,
                "usage": {
                    "input_tokens": 1,
                    "cache_read_input_tokens": 1,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "reasoning_tokens": 0,
                    "observation_basis": "exact",
                },
            }
        else:
            body = {
                "result": "spam",
                "usage": {
                    "input_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_write_input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "reasoning_tokens": 1,
                    "observation_basis": "exact",
                },
            }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


def test_run_reports_actual_subcall_count(monkeypatch, capfd) -> None:
    from importlib import import_module

    run_module = import_module("droste_runner.run")
    real_client = run_module.HTTPSubcallClient
    resolved_concurrency: list[int] = []

    class CapturingHTTPSubcallClient(real_client):
        def __init__(self, **kwargs: Any) -> None:
            resolved_concurrency.append(kwargs["max_parallel"])
            super().__init__(**kwargs)

    monkeypatch.setattr(run_module, "HTTPSubcallClient", CapturingHTTPSubcallClient)
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = run(
            {
                "protocol_version": 8,
                "model": "test-model",
                "question": "is it spam?",
                "budget": _budget(subcalls=10, depth=2),
                "token": "test-token",
                "root_endpoint": f"{base}/root",
                "subcall_endpoint": f"{base}/subcall",
                "trace_policy_id": "local-training-v1",
                "retain_trace": ["subcall"],
                "trace_expires_at": "2026-10-14T00:00:00Z",
                "trace_host_managed_expiry": True,
                "training_allowed": True,
                "data_use_authorization_ref": "consent://trace/1",
                "data_use_purposes": ["training"],
                "root_model_revision": "root-rev",
                "subcall_model": "leaf-model",
                "subcall_model_revision": "leaf-rev",
                "root_sampling": {"temperature": 0.25},
                "subcall_sampling": {"temperature": 0},
                "subcall_concurrency": 2,
                "seed": 17,
                "source_revision": "commit-a",
            }
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    live_events = [json.loads(line) for line in capfd.readouterr().err.splitlines()]

    assert response["error"] is None
    assert response["ready"] is True
    assert response["answer"] == "label: spam"
    assert response["subcalls"] == 1
    assert response["successful_subcalls"] == 1
    assert response["extracted"] is False
    assert response["recovered_error"] is None
    assert response["answer_metadata"] == {"evidence_ids": ["classification-1"]}
    assert "stdout_truncations" not in response
    manifest = response["scaffold_manifest"]
    assert manifest["inference"] == {
        "root": {"id": "test-model", "revision": "root-rev"},
        "subcall": {"id": "leaf-model", "revision": "leaf-rev"},
        "root_sampling": {"temperature": 0.25},
        "subcall_sampling": {"temperature": 0},
        "output_limits": {"root_tokens": 4096, "subcall_tokens": 2048},
        "input_capacity": {"subcall": {"state": "unknown", "tokens": None}},
        "concurrency": 2,
        "seed": 17,
    }
    assert manifest["abis"]["runner"] == 8
    assert manifest["abis"]["trace"] == 4
    assert manifest["engine"]["source_revision"] == "commit-a"
    assert manifest["id"].startswith("sha256:")
    assert "trajectory" not in response
    usage = response["run_record"]["terminal"]["usage"]
    assert usage["root"] == {
        "input_tokens": 1,
        "cache_read_tokens": 1,
        "cache_creation_tokens": 0,
        "output_tokens": 1,
        "total_tokens": 2,
        "requests": 1,
        "successes": 1,
        "complete": True,
    }
    assert usage["subcall"] == {
        "input_tokens": 2,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
        "requests": 1,
        "successes": 1,
        "complete": True,
    }
    assert usage["total_tokens"] == 7
    capability = next(
        event for event in response["run_record"]["events"] if event["type"] == "capability"
    )
    assert capability["outcome"]["run_id"] == response["run_id"]
    live_subcalls = [event for event in live_events if event["type"] == "subcall"]
    retained_subcalls = [
        event for event in response["run_record"]["events"] if event["type"] == "subcall"
    ]
    assert live_subcalls == retained_subcalls
    assert [event["phase"] for event in live_subcalls] == [
        "start",
        "progress",
        "completion",
    ]
    assert len({event["call_id"] for event in live_subcalls}) == 1
    assert live_subcalls[1]["checkpoint"] == {"tokens": 5, "subcalls": 1}
    assert all(event["iteration"] == 1 and event["version"] == 4 for event in live_subcalls)
    assert capability["outcome"]["capability_id"]["operation"] == "llm_query"
    assert "params" not in capability["outcome"]
    assert "result" not in capability["outcome"]
    assert response["run_record"]["retention"] == {
        "policy_id": "local-training-v1",
        "retain": ["subcall"],
        "expires_at": "2026-10-14T00:00:00Z",
        "host_managed_expiry": True,
    }
    assert response["run_record"]["data_use"] == {
        "training_allowed": True,
        "authorization_ref": "consent://trace/1",
        "purposes": ["training"],
    }
    assert resolved_concurrency == [2]


def test_runner_trajectory_adds_status_without_rewriting_result(monkeypatch) -> None:
    from importlib import import_module

    import droste_runner.runner as runner_module
    from droste.loop.step import RLMResult
    from droste.loop.trajectory import IterationRecord

    execution_result = "ERROR: legitimate application output\n"

    def fake_run_rlm(*args: Any, **kwargs: Any) -> RLMResult:
        return RLMResult(
            answer=execution_result,
            ready=True,
            iterations=1,
            tokens_used=2,
            sub_calls_made=0,
            trajectory=[
                IterationRecord(
                    iteration=1,
                    llm_input=[{"role": "user", "content": "q"}],
                    llm_output="```python\nprint('x')\n```",
                    code_executed="print('x')",
                    execution_result=execution_result,
                    tokens_used=2,
                    execution_status="success",
                )
            ],
        )

    monkeypatch.setattr(import_module("droste_runner.run"), "run_rlm", fake_run_rlm)
    response = runner_module.run(
        {
            "protocol_version": 8,
            "model": "test-model",
            "question": "q",
            "budget": _budget(),
            "token": "unused",
            "root_endpoint": "http://127.0.0.1:1/root",
            "subcall_endpoint": "http://127.0.0.1:1/subcall",
        }
    )

    assert response["answer"] == execution_result
    assert "trajectory" not in response


def _client(**kwargs: Any) -> tuple[HTTPSubcallClient, Any]:
    context = create_execution_context()
    client = HTTPSubcallClient(
        endpoint="http://unused.invalid",
        token="t",
        session="",
        session_index=0,
        context=context,
        **kwargs,
    )
    client._request = lambda payload: SubcallQueryResult(  # type: ignore[method-assign]
        "ok", TokenUsage(1, 1, 2, exact=True)
    )
    return client, context


def test_http_client_reports_each_issued_call_without_owning_budget_policy() -> None:
    client, context = _client()
    assert client.llm_query("a") == "ok"
    assert client.llm_query("b") == "ok"
    assert context.stats.calls_made == 2
    assert context.stats.successful_calls == 2


def test_failed_request_is_attempted_but_not_successful() -> None:
    client, context = _client()

    def fail(_payload: dict[str, Any]) -> str:
        raise RuntimeError("request failed")

    client._request = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="request failed"):
        client.llm_query("a")

    assert context.stats.calls_made == 1
    assert context.stats.successful_calls == 0


def test_concurrent_batch_counts_each_issued_call() -> None:
    client, context = _client()
    results = client.llm_batch([f"p{i}" for i in range(20)])
    assert results == ["ok"] * 20
    assert context.stats.calls_made == 20
    assert context.stats.successful_calls == 20


# --- llm_batch / llm_batch_with_errors share one bounded fan-out (#34).


def test_llm_batch_with_errors_bounded_concurrency() -> None:
    client, _ = _client(max_parallel=2)
    lock = threading.Lock()
    overlap = threading.Event()
    active = 0
    peak = 0

    def _request(payload: dict[str, Any]) -> SubcallQueryResult:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                overlap.set()
        try:
            # Hold the first call until the deliberately staggered second
            # worker arrives. A fixed sleep races the 50 ms launch stagger in
            # the production batch path and can observe only one active call.
            overlap.wait(timeout=1)
            return SubcallQueryResult("ok", TokenUsage(1, 1, 2, exact=True))
        finally:
            with lock:
                active -= 1

    client._request = _request  # type: ignore[method-assign]
    results, errors = client.llm_batch_with_errors([f"p{i}" for i in range(30)])
    assert results == ["ok"] * 30
    assert errors == []
    assert peak == 2


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_http_subcall_rejects_invalid_concurrency(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="subcall concurrency"):
        _client(max_parallel=value)


def test_llm_batch_with_errors_enforces_prompt_cap() -> None:
    client, _ = _client()
    with pytest.raises(ValueError, match="exceeds max 50"):
        client.llm_batch_with_errors(["p"] * 51)


def test_llm_batch_with_errors_orders_errors_by_index() -> None:
    client, _ = _client()

    def _request(payload: dict[str, Any]) -> SubcallQueryResult:
        prompt = payload["prompt"]
        if prompt == "p1":
            # Finish after p3's failure so completion order != index order.
            time.sleep(0.05)
            raise RuntimeError("boom p1")
        if prompt == "p3":
            raise RuntimeError("boom p3")
        return SubcallQueryResult("ok", TokenUsage(1, 1, 2, exact=True))

    client._request = _request  # type: ignore[method-assign]
    results, errors = client.llm_batch_with_errors([f"p{i}" for i in range(5)])
    assert results == ["ok", "", "ok", "", "ok"]
    assert [e["index"] for e in errors] == [1, 3]
    assert "boom p1" in str(errors[0]["error"])
    assert "boom p3" in str(errors[1]["error"])


def test_llm_batch_raises_lowest_index_error_unwrapped() -> None:
    client, _ = _client()

    class _Boom(RuntimeError):
        pass

    def _request(payload: dict[str, Any]) -> SubcallQueryResult:
        prompt = payload["prompt"]
        if prompt in ("p2", "p4"):
            raise _Boom(f"boom {prompt}")
        return SubcallQueryResult("ok", TokenUsage(1, 1, 2, exact=True))

    client._request = _request  # type: ignore[method-assign]
    with pytest.raises(_Boom, match="boom p2") as raised:
        client.llm_batch([f"p{i}" for i in range(5)])
    assert raised.value.__cause__ is not None
    assert raised.value.__cause__.__cause__ is None


def test_http_fail_fast_batch_carries_every_error_in_index_order() -> None:
    client, _ = _client()

    class _Boom(RuntimeError):
        pass

    def _request(payload: dict[str, Any]) -> SubcallQueryResult:
        prompt = payload["prompt"]
        if prompt in ("p2", "p4"):
            raise _Boom(f"boom {prompt}")
        return SubcallQueryResult("ok", TokenUsage(1, 1, 2, exact=True))

    client._request = _request  # type: ignore[method-assign]
    with pytest.raises(SubcallBatchFailure) as failure:
        client.llm_batch_with_usage([f"p{i}" for i in range(5)])

    assert str(failure.value.cause) == "boom p2"
    assert failure.value.result.errors == (
        {"index": 2, "error": "boom p2"},
        {"index": 4, "error": "boom p4"},
    )


def test_http_fanout_batch_preserves_usage_failure_and_original_cause() -> None:
    client, _ = _client()

    def request(payload: dict[str, Any]) -> SubcallQueryResult:
        if payload["prompt"] == "bad":
            raise LLMUsageFailure(
                TokenUsage(7, 3, 19, exact=True),
                RuntimeError("malformed HTTP output"),
            )
        return SubcallQueryResult("ok", TokenUsage(2, 1, 5, exact=True))

    client._request = request  # type: ignore[method-assign]
    with pytest.raises(SubcallBatchFailure, match="malformed HTTP output") as failure:
        client.llm_batch_with_usage(["ok", "bad"])
    assert failure.value.result.usage == (
        TokenUsage(2, 1, 5, exact=True),
        TokenUsage(7, 3, 19, exact=True),
    )
    assert failure.value.result.errors == ({"index": 1, "error": "malformed HTTP output"},)
    assert type(failure.value.cause) is RuntimeError
    with pytest.raises(RuntimeError, match="malformed HTTP output"):
        client.llm_batch(["ok", "bad"])


# --- Subcall cost controls: the resolved budget always supplies the output
# ceiling; optional model controls are included only when set.


class _CapturingHandler(BaseHTTPRequestHandler):
    subcall_payloads: list[dict[str, Any]] = []
    root_payloads: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length)
        if self.path == "/root":
            type(self).root_payloads.append(json.loads(raw_body.decode("utf-8")))
            body = {
                "result": ROOT_REPLY,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "reasoning_tokens": 0,
                    "observation_basis": "exact",
                },
            }
        else:
            type(self).subcall_payloads.append(json.loads(raw_body.decode("utf-8")))
            body = {"result": "spam"}
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


def _run_with_capture(request_extra: dict[str, Any]) -> list[dict[str, Any]]:
    _CapturingHandler.subcall_payloads = []
    _CapturingHandler.root_payloads = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request: dict[str, Any] = {
            "protocol_version": 8,
            "model": "test-model",
            "question": "is it spam?",
            "budget": _budget(subcalls=10, depth=2),
            "token": "test-token",
            "root_endpoint": f"{base}/root",
            "subcall_endpoint": f"{base}/subcall",
        }
        request.update(request_extra)
        response = run(request)
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert response["error"] is None
    assert response["subcalls"] == 1
    usage = response["run_record"]["terminal"]["usage"]
    assert usage["kind"] == "partial"
    assert usage["root"]["complete"] is True
    assert usage["subcall"]["complete"] is False
    assert usage["total_tokens"] == 2
    return _CapturingHandler.subcall_payloads


def test_subcall_payload_includes_cost_controls_when_set() -> None:
    payloads = _run_with_capture(
        {
            "budget": _budget(subcalls=10, depth=2, subcall_output_tokens=1024),
            "subcall_model": "grok-4-fast-non-reasoning",
            "subcall_reasoning_effort": "none",
        }
    )
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["max_output_tokens"] == 1024
    assert payload["model"] == "grok-4-fast-non-reasoning"
    assert payload["reasoning_effort"] == "none"
    system_prompt = _CapturingHandler.root_payloads[0]["messages"][0]["content"]
    assert "subcall_output_tokens_per_call=1024" in system_prompt


def test_root_payload_includes_resolved_reasoning_effort() -> None:
    _run_with_capture({"root_reasoning_effort": "none"})

    assert len(_CapturingHandler.root_payloads) == 1
    assert _CapturingHandler.root_payloads[0]["reasoning_effort"] == "none"


def test_subcall_payload_uses_budget_ceiling_and_omits_optional_controls() -> None:
    payloads = _run_with_capture({})
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["max_output_tokens"] == 2048
    assert "model" not in payload
    assert "reasoning_effort" not in payload
    assert "reasoning_effort" not in _CapturingHandler.root_payloads[0]
    system_prompt = _CapturingHandler.root_payloads[0]["messages"][0]["content"]
    assert "subcall_output_tokens_per_call=2048" in system_prompt


def test_zero_subcall_output_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget.subcall_output_tokens must be positive"):
        budget = _budget()
        budget["subcall_output_tokens"] = 0
        run(
            {
                "protocol_version": 8,
                "model": "m",
                "question": "q",
                "budget": budget,
                "token": "t",
                "root_endpoint": "http://127.0.0.1:1/root",
                "subcall_endpoint": "http://127.0.0.1:1/subcall",
            }
        )


def test_wrapper_call_enforces_allowed_hosts():
    # The wrapper's allowed_hosts is an enforced allowlist, not prompt
    # decoration: a request-supplied base_url outside it must be refused
    # before any connection is attempted (pre-publish read-through).
    from droste_runner.runner import WrapperTransport

    wrapper = WrapperTransport(
        {
            "base_url": "http://evil.internal:9",
            "token": "t",
            "allowed_hosts": ["partner.example.com"],
        }
    )
    import pytest as _pytest

    with _pytest.raises(ValueError, match="not in allowed_hosts"):
        wrapper._call("/search", {"query": "q"})


def test_wrapper_rejects_redirect_to_disallowed_host():
    # SSRF-via-redirect: a call starting at an allowed host must still refuse
    # a 30x to a host outside the allowlist (codex review).
    from droste_runner.runner import _allowlist_opener

    opener = _allowlist_opener({"partner.example.com"})
    handler = next(h for h in opener.handlers if hasattr(h, "redirect_request"))
    import pytest as _pytest

    class _Req:
        pass

    with _pytest.raises(ValueError, match="not in allowed_hosts"):
        handler.redirect_request(
            _Req(), None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data"
        )


def test_wrapper_malformed_allowed_hosts_fails_closed():
    # A present-but-malformed allowed_hosts (string / empty list) is a config
    # error, not an allow-all — fail closed (codex review).
    import pytest as _pytest

    from droste_runner.runner import WrapperTransport

    for bad in ("partner.example.com", [], ["", "  "]):
        w = WrapperTransport({"base_url": "http://h/x", "token": "t", "allowed_hosts": bad})
        with _pytest.raises(ValueError, match="allowed_hosts"):
            w._call("/search", {"query": "q"})

    # Absent key still means allow-all (no raise for the host check; will fail
    # later at connection instead).
    w = WrapperTransport({"base_url": "http://127.0.0.1:9/x", "token": "t"})
    with _pytest.raises(ValueError, match="request failed"):
        w._call("/search", {"query": "q"})
